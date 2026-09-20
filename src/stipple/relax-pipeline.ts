/**
 * Relax compute pipeline — curvature-aware surface repulsion with
 * re-projection.
 *
 * Each frame, {@link dispatch} records two compute passes on the given
 * encoder:
 *
 * 1. **Normal precompute** — one invocation per point writes
 *    `normalize(sdfGradient(p))` into the shared {@link PointBuffers.normalsBuffer}.
 *    Runs first so the relax pass reads fresh normals matching the current
 *    positions.
 * 2. **Repulsion** — each invocation `i` reads `p_i` and `n_i` (inline),
 *    loops over all `j ≠ i` reading `n_j` from the normals buffer,
 *    accumulates a linear-decay repulsion gated by a curvature-inflated
 *    Euclidean cutoff and a midpoint SDF line-of-sight check, projects the
 *    force onto the tangent plane at `p_i`, integrates with a direct Euler
 *    position step, then Newton-projects the new position back onto the
 *    SDF surface.
 *
 * Ping-pong: both passes read from one point buffer and write to the
 * other, depending on `readFromA`. Two static bind-group sets cover both
 * directions. The normals buffer is shared (not ping-ponged). There is no
 * density texture — points are on the surface by construction (seeded
 * there, re-projected every frame).
 */

import type { PointBuffers } from "./point-buffers.js";
import { NORMAL_SHADER, RELAX_SHADER } from "./shaders.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the relax params uniform buffer in bytes (4 × f32 + 4 × u32 = 32). */
const RELAX_PARAMS_BUFFER_BYTES = 32;

/** Size of the normal params uniform buffer in bytes (4 × u32 = 16). */
const NORMAL_PARAMS_BUFFER_BYTES = 16;

/** Relaxation parameters passed to the compute shader via uniform. */
export interface RelaxParams {
    /** Time step per frame. */
    readonly dt: number;
    /** Interaction radius — pairs with inflated distance beyond this are ignored. */
    readonly radius: number;
    /** Midpoint line-of-sight threshold; pairs with `|map(m)| > alpha·d_E²` are skipped. */
    readonly alpha: number;
}

/**
 * Default relaxation parameters — initial guesses, need visual tuning.
 */
export const DEFAULT_RELAX_PARAMS: RelaxParams = {
    dt: 0.01,
    radius: 0.2,
    alpha: 0.6,
};

/**
 * Manages the normal-precompute and relax compute pipelines, their params
 * uniforms, and the ping-pong bind groups.
 *
 * The normals buffer is allocated once by {@link PointBuffers} and shared
 * between this pipeline (written by the normal sub-pass, read by the relax
 * sub-pass) and the point renderer.
 */
export class RelaxPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;
    private readonly relaxPipeline: GPUComputePipeline;
    private readonly relaxLayout: GPUBindGroupLayout;
    private readonly relaxParamsBuffer: GPUBuffer;
    private readonly relaxBindGroupAtoB: GPUBindGroup;
    private readonly relaxBindGroupBtoA: GPUBindGroup;
    private readonly normalPipeline: GPUComputePipeline;
    private readonly normalLayout: GPUBindGroupLayout;
    private readonly normalParamsBuffer: GPUBuffer;
    private readonly normalBindGroupA: GPUBindGroup;
    private readonly normalBindGroupB: GPUBindGroup;

    /**
     * Creates both compute pipelines (normal precompute + relax), their
     * params uniforms, and all bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair plus shared normals
     *   buffer.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;
        this.points = points;

        const relaxModule = device.createShaderModule({
            label: "stipple-relax-shader",
            code: RELAX_SHADER,
        });
        const normalModule = device.createShaderModule({
            label: "stipple-normal-shader",
            code: NORMAL_SHADER,
        });

        this.relaxPipeline = device.createComputePipeline({
            label: "stipple-relax-pipeline",
            layout: "auto",
            compute: {
                module: relaxModule,
                entryPoint: "relax_cs",
            },
        });
        this.normalPipeline = device.createComputePipeline({
            label: "stipple-normal-pipeline",
            layout: "auto",
            compute: {
                module: normalModule,
                entryPoint: "normal_cs",
            },
        });

        this.relaxLayout = this.relaxPipeline.getBindGroupLayout(0);
        this.normalLayout = this.normalPipeline.getBindGroupLayout(0);

        this.relaxParamsBuffer = device.createBuffer({
            label: "stipple-relax-params",
            size: RELAX_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.normalParamsBuffer = device.createBuffer({
            label: "stipple-normal-params",
            size: NORMAL_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.relaxBindGroupAtoB = device.createBindGroup({
            label: "stipple-relax-bind-A-to-B",
            layout: this.relaxLayout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferA } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: this.relaxParamsBuffer } },
                { binding: 3, resource: { buffer: points.normalsBuffer } },
            ],
        });
        this.relaxBindGroupBtoA = device.createBindGroup({
            label: "stipple-relax-bind-B-to-A",
            layout: this.relaxLayout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferB } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: this.relaxParamsBuffer } },
                { binding: 3, resource: { buffer: points.normalsBuffer } },
            ],
        });

        this.normalBindGroupA = device.createBindGroup({
            label: "stipple-normal-bind-A",
            layout: this.normalLayout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferA } },
                { binding: 1, resource: { buffer: points.normalsBuffer } },
                { binding: 2, resource: { buffer: this.normalParamsBuffer } },
            ],
        });
        this.normalBindGroupB = device.createBindGroup({
            label: "stipple-normal-bind-B",
            layout: this.normalLayout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferB } },
                { binding: 1, resource: { buffer: points.normalsBuffer } },
                { binding: 2, resource: { buffer: this.normalParamsBuffer } },
            ],
        });
    }

    /**
     * Writes the relaxation parameters into the relax params uniform
     * buffer and the point count into the normal params uniform buffer.
     *
     * Relax layout (32 bytes): `dt, radius, alpha, _pad0` (4 × f32)
     * followed by `point_count` and three padding `u32`s.
     *
     * Normal layout (16 bytes): `point_count` followed by three padding
     * `u32`s.
     *
     * @param params - The parameters to upload.
     */
    private writeParams(params: RelaxParams): void {
        const relaxBuffer = new ArrayBuffer(RELAX_PARAMS_BUFFER_BYTES);
        const relaxF32 = new Float32Array(relaxBuffer);
        const relaxU32 = new Uint32Array(relaxBuffer);
        relaxF32[0] = params.dt;
        relaxF32[1] = params.radius;
        relaxF32[2] = params.alpha;
        relaxU32[4] = this.points.count;
        this.device.queue.writeBuffer(this.relaxParamsBuffer, 0, relaxBuffer);

        const normalBuffer = new ArrayBuffer(NORMAL_PARAMS_BUFFER_BYTES);
        const normalU32 = new Uint32Array(normalBuffer);
        normalU32[0] = this.points.count;
        this.device.queue.writeBuffer(this.normalParamsBuffer, 0, normalBuffer);
    }

    /**
     * Records a normal-precompute pass followed by a relax pass into the
     * given command encoder.
     *
     * Both passes read from one point buffer (depending on `readFromA`);
     * the relax pass writes to the other. The normal pass writes to the
     * shared normals buffer, which the relax pass reads. Two separate
     * compute passes are used so the normals storage writes are visible to
     * the subsequent relax dispatch.
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

        const normalPass = encoder.beginComputePass();
        normalPass.setPipeline(this.normalPipeline);
        normalPass.setBindGroup(0, readFromA ? this.normalBindGroupA : this.normalBindGroupB);
        normalPass.dispatchWorkgroups(workgroupCount);
        normalPass.end();

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
