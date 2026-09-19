/**
 * Compute pipeline for O(n²) all-pairs gravity integration.
 *
 * Owns the WGSL shader module, compute pipeline, params uniform buffer,
 * and two pre-created bind groups for the ping-pong directions
 * (A→B and B→A). Swapping direction is just selecting which bind group
 * to set before dispatch.
 */

import type { BodyBuffers } from "./buffers.js";
import { COMPUTE_SHADER } from "./shaders.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the params uniform buffer in bytes (4 × 4 = 16). */
const PARAMS_BUFFER_BYTES = 16;

/** Simulation parameters passed to the compute shader via uniform buffer. */
export interface SimParams {
    /** Time step per frame (simulation time units). */
    readonly dt: number;
    /** Gravitational constant (normalized, not SI). */
    readonly g: number;
    /** Plummer softening length — prevents singularities at close approach. */
    readonly softening: number;
    /** Number of bodies in the simulation. */
    readonly bodyCount: number;
}

/** Default simulation parameters — tuned for ~4096 bodies in a sphere of radius 20. */
export const DEFAULT_PARAMS: SimParams = {
    dt: 0.005,
    g: 1.0,
    softening: 10.0,
    bodyCount: 8 * 1024,
};

/**
 * Manages the compute pipeline and its bind groups for the n-body
 * gravity integration step.
 *
 * The pipeline is created once at construction time. Each frame, the
 * caller invokes {@link dispatch} to record a compute pass into the
 * current frame's command encoder, choosing the ping-pong direction
 * via the `readFromA` flag.
 */
export class ComputePipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupAtoB: GPUBindGroup;
    private readonly bindGroupBtoA: GPUBindGroup;
    private readonly paramsBuffer: GPUBuffer;

    /**
     * Creates the shader module, compute pipeline, params uniform buffer,
     * and both ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param bodies - The ping-pong body buffer pair.
     * @param params - Initial simulation parameters.
     */
    public constructor(device: GPUDevice, bodies: BodyBuffers, params: SimParams) {
        this.device = device;

        const shaderModule = device.createShaderModule({
            label: "nbody-compute-shader",
            code: COMPUTE_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "nbody-compute-pipeline",
            layout: "auto",
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });

        this.paramsBuffer = device.createBuffer({
            label: "nbody-params",
            size: PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.writeParams(params);

        const layout = this.pipeline.getBindGroupLayout(0);

        this.bindGroupAtoB = device.createBindGroup({
            label: "nbody-bind-A-to-B",
            layout,
            entries: [
                { binding: 0, resource: { buffer: bodies.bufferA } },
                { binding: 1, resource: { buffer: bodies.bufferB } },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
            ],
        });

        this.bindGroupBtoA = device.createBindGroup({
            label: "nbody-bind-B-to-A",
            layout,
            entries: [
                { binding: 0, resource: { buffer: bodies.bufferB } },
                { binding: 1, resource: { buffer: bodies.bufferA } },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
            ],
        });
    }

    /**
     * Writes simulation parameters into the uniform buffer.
     *
     * Uses an `ArrayBuffer` with dual `Float32Array` / `Uint32Array`
     * views so that `bodyCount` is stored as a proper `u32`.
     *
     * @param params - The parameters to upload.
     */
    private writeParams(params: SimParams): void {
        const buffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = params.dt;
        f32[1] = params.g;
        f32[2] = params.softening;
        u32[3] = params.bodyCount;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a compute dispatch into the given command encoder.
     *
     * Reads from one body buffer and writes to the other, depending on
     * `readFromA`. The caller must flip `readFromA` after each dispatch
     * to implement the ping-pong swap.
     *
     * @param encoder - The command encoder to record into.
     * @param params - Current simulation parameters.
     * @param readFromA - If `true`, reads bufferA → writes bufferB;
     *   if `false`, reads bufferB → writes bufferA.
     */
    public dispatch(encoder: GPUCommandEncoder, params: SimParams, readFromA: boolean): void {
        this.writeParams(params);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromA ? this.bindGroupAtoB : this.bindGroupBtoA);

        const workgroupCount = Math.ceil(params.bodyCount / WORKGROUP_SIZE);
        pass.dispatchWorkgroups(workgroupCount);

        pass.end();
    }
}
