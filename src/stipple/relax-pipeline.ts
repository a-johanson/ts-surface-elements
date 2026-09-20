/**
 * Relax compute pipeline — 3D repulsion with surface re-projection.
 *
 * Each frame, one compute dispatch is recorded: each invocation `i` reads
 * `p_i`, `v_i`, loops over all `j ≠ i` accumulating a softened 1/r
 * repulsion force in 3D, integrates with semi-implicit Euler and damping,
 * then Newton-projects the new position back onto the SDF surface.
 *
 * Ping-pong: the dispatch reads from one point buffer and writes to the
 * other. Two static bind groups cover both directions. There is no
 * density texture and no auxiliary densities buffer — points are on the
 * surface by construction (seeded there, re-projected every frame).
 */

import type { PointBuffers } from "./point-buffers.js";
import { RELAX_SHADER } from "./shaders.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the params uniform buffer in bytes (4 × f32 + 4 × u32 = 32). */
const PARAMS_BUFFER_BYTES = 32;

/** Relaxation parameters passed to the compute shader via uniform. */
export interface RelaxParams {
    /** Time step per frame. */
    readonly dt: number;
    /** Repulsion strength between point pairs. */
    readonly kRep: number;
    /** Plummer-style softening length, prevents singularities. */
    readonly softening: number;
    /** Per-frame velocity damping (0 = frozen, 1 = no damping). */
    readonly damping: number;
}

/**
 * Default relaxation parameters — initial guesses, need visual tuning.
 */
export const DEFAULT_RELAX_PARAMS: RelaxParams = {
    dt: 0.01,
    kRep: 0.01,
    softening: 0.0001,
    damping: 0.95,
};

/**
 * Manages the relax compute pipeline, the params uniform, and the
 * ping-pong bind groups.
 */
export class RelaxPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;
    private readonly pipeline: GPUComputePipeline;
    private readonly layout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly relaxBindGroupAtoB: GPUBindGroup;
    private readonly relaxBindGroupBtoA: GPUBindGroup;

    /**
     * Creates the compute pipeline, the params uniform, and both
     * ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;
        this.points = points;

        const module = device.createShaderModule({
            label: "stipple-relax-shader",
            code: RELAX_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-relax-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "relax_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-relax-params",
            size: PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.relaxBindGroupAtoB = device.createBindGroup({
            label: "stipple-relax-bind-A-to-B",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferA } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
            ],
        });

        this.relaxBindGroupBtoA = device.createBindGroup({
            label: "stipple-relax-bind-B-to-A",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferB } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
            ],
        });
    }

    /**
     * Writes the relaxation parameters into the uniform buffer.
     *
     * Layout (32 bytes): `dt, kRep, softening, damping` (4 × f32) followed
     * by `point_count` (from the point buffer pair) and three padding
     * `u32`s.
     *
     * @param params - The parameters to upload.
     */
    private writeParams(params: RelaxParams): void {
        const buffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = params.dt;
        f32[1] = params.kRep;
        f32[2] = params.softening;
        f32[3] = params.damping;
        u32[4] = this.points.count;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a relax compute dispatch into the given command encoder.
     *
     * Reads from one point buffer and writes to the other, depending on
     * `readFromA`. The caller must flip `readFromA` after each dispatch
     * to implement the ping-pong swap.
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

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromA ? this.relaxBindGroupAtoB : this.relaxBindGroupBtoA);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
    }
}
