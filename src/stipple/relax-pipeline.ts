/**
 * Relax compute pipeline — curvature-aware surface repulsion with
 * re-projection, accelerated by the spatial grid.
 *
 * Each frame, {@link dispatch} records a single compute pass: each
 * invocation `i` reads `p_i` and `n_i` (inline), recomputes its grid cell,
 * scans the 27-cell neighborhood via `cellStart` / `cellCount` /
 * `sortedValues`, and for each neighbor `j` reads `n_j` from the shared
 * normals buffer, accumulates a linear-decay repulsion gated by a
 * curvature-inflated Euclidean cutoff and a midpoint SDF line-of-sight
 * check, projects the force onto the tangent plane at `p_i`, integrates
 * with a direct Euler position step, then Newton-projects the new position
 * back onto the time-animated SDF surface.
 *
 * Ping-pong: the pass reads from one point buffer and writes to the
 * other, depending on `readFromA`. Two static bind-group sets cover both
 * directions. The grid buffers (params, sorted values, cell start/count)
 * are not ping-ponged — they are rebuilt once per frame from
 * whichever buffer relax is about to read, so both A→B and B→A bind
 * groups bind the same grid buffer set. The normals buffer is shared
 * (not ping-ponged) and is owned by the reproject pass (per-frame,
 * before relax) and the shading pass (per-frame, after relax) — it
 * always matches whichever buffer relax reads. There is no density
 * texture — points are on the surface by construction (seeded there,
 * re-projected every frame by reproject and relax).
 */

import type { PointBuffers } from "./point-buffers.js";
import { RELAX_SHADER } from "./shaders/index.js";
import type { GridBuffers } from "./spatial-grid-pipeline.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the relax params uniform buffer in bytes (4 × f32 + 4 × u32 = 32). */
const RELAX_PARAMS_BUFFER_BYTES = 32;

/** Relaxation parameters passed to the compute shader via uniform. */
export interface RelaxParams {
    /** Per-frame wall-clock delta in seconds (capped at `MAX_DT`), threaded from the frame loop. */
    readonly dt: number;
    /** Interaction radius — pairs with inflated distance beyond this are ignored. */
    readonly radius: number;
    /** Midpoint line-of-sight threshold; pairs with `|map(m)| > alpha·d_E²` are skipped. */
    readonly alpha: number;
    /** Current animation time in seconds — threads into the SDF `map(p, time)`. */
    readonly time: number;
}

/**
 * Default relaxation parameters — initial guesses, need visual tuning.
 */
export const DEFAULT_RELAX_PARAMS: Omit<RelaxParams, "time" | "dt"> = {
    radius: 0.3,
    alpha: 0.5,
};

/**
 * Manages the relax compute pipeline, its params uniform, and the
 * ping-pong bind groups.
 *
 * The normals buffer is allocated once by {@link PointBuffers} and shared
 * between this pipeline (read for curvature-aware repulsion), the
 * reproject pipeline (written per-frame before relax), and the shading
 * pipeline (written per-frame after relax, plus once at bootstrap by the
 * seed pipeline).
 */
export class RelaxPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;
    private readonly relaxPipeline: GPUComputePipeline;
    private readonly relaxLayout: GPUBindGroupLayout;
    private readonly relaxParamsBuffer: GPUBuffer;
    private readonly relaxBindGroupAtoB: GPUBindGroup;
    private readonly relaxBindGroupBtoA: GPUBindGroup;

    /**
     * Creates the relax compute pipeline, its params uniform, and both
     * ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair plus shared normals
     *   buffer.
     * @param grid - The spatial grid buffers (params, sorted values, cell
     *   start/count). Rebuilt once per frame by `SpatialGridPipeline` from
     *   whichever buffer relax reads; not ping-ponged.
     */
    public constructor(device: GPUDevice, points: PointBuffers, grid: GridBuffers) {
        this.device = device;
        this.points = points;

        const relaxModule = device.createShaderModule({
            label: "stipple-relax-shader",
            code: RELAX_SHADER,
        });

        this.relaxPipeline = device.createComputePipeline({
            label: "stipple-relax-pipeline",
            layout: "auto",
            compute: {
                module: relaxModule,
                entryPoint: "relax_cs",
            },
        });

        this.relaxLayout = this.relaxPipeline.getBindGroupLayout(0);

        this.relaxParamsBuffer = device.createBuffer({
            label: "stipple-relax-params",
            size: RELAX_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.relaxBindGroupAtoB = device.createBindGroup({
            label: "stipple-relax-bind-A-to-B",
            layout: this.relaxLayout,
            entries: [
                { binding: 0, resource: { buffer: this.relaxParamsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.bufferB } },
                { binding: 3, resource: { buffer: points.normalsBuffer } },
                { binding: 4, resource: { buffer: grid.gridParamsBuffer } },
                { binding: 5, resource: { buffer: grid.sortedValues } },
                { binding: 6, resource: { buffer: grid.cellStart } },
                { binding: 7, resource: { buffer: grid.cellCount } },
            ],
        });
        this.relaxBindGroupBtoA = device.createBindGroup({
            label: "stipple-relax-bind-B-to-A",
            layout: this.relaxLayout,
            entries: [
                { binding: 0, resource: { buffer: this.relaxParamsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: points.bufferA } },
                { binding: 3, resource: { buffer: points.normalsBuffer } },
                { binding: 4, resource: { buffer: grid.gridParamsBuffer } },
                { binding: 5, resource: { buffer: grid.sortedValues } },
                { binding: 6, resource: { buffer: grid.cellStart } },
                { binding: 7, resource: { buffer: grid.cellCount } },
            ],
        });
    }

    /**
     * Writes the relaxation parameters and point count into the relax
     * params uniform buffer.
     *
     * Relax layout (32 bytes): `dt, radius, alpha, time` (4 × f32)
     * followed by `point_count` and three padding `u32`s.
     *
     * @param params - The parameters to upload.
     */
    private writeParams(params: RelaxParams): void {
        const buffer = new ArrayBuffer(RELAX_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = params.dt;
        f32[1] = params.radius;
        f32[2] = params.alpha;
        f32[3] = params.time;
        u32[4] = this.points.count;
        this.device.queue.writeBuffer(this.relaxParamsBuffer, 0, buffer);
    }

    /**
     * Records a relax compute pass into the given command encoder.
     *
     * Reads from one point buffer (depending on `readFromA`) and writes to
     * the other. The shared normals buffer is read (not written) by this
     * pass; it is refreshed per-frame by the reproject pass (before relax)
     * and the shading pass (after relax).
     *
     * The caller must flip `readFromA` after each dispatch to implement
     * the ping-pong swap.
     *
     * @param encoder - The command encoder to record into.
     * @param params - Relaxation parameters.
     * @param readFromA - If `true`, reads bufferA → writes bufferB;
     *   if `false`, reads bufferB → writes bufferA.
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        params: RelaxParams,
        readFromA: boolean,
    ): void {
        this.writeParams(params);

        const workgroupCount = Math.ceil(this.points.count / WORKGROUP_SIZE);

        const relaxPass = encoder.beginComputePass();
        relaxPass.setPipeline(this.relaxPipeline);
        relaxPass.setBindGroup(
            0,
            readFromA ? this.relaxBindGroupAtoB : this.relaxBindGroupBtoA,
        );
        relaxPass.dispatchWorkgroups(workgroupCount);
        relaxPass.end();
    }
}
