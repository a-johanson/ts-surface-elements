/**
 * Reproject compute pipeline — re-projects points onto the current animated
 * SDF surface and refreshes the shared normals buffer.
 *
 * Each frame, {@link dispatch} records a single compute pass that runs
 * *before* relax. One invocation per point reads `p_i` from the point
 * buffer relax is about to read, Newton-projects it onto the surface at
 * the current animation time (16 iterations, `alpha = 0.5` — generous
 * budget since the per-frame surface motion may be larger than relax's
 * tail re-projection assumes), writes the projected position back
 * in-place, and writes the matching surface normal into the shared
 * normals buffer.
 *
 * This ensures relax starts with on-surface positions and matching
 * normals even when the SDF has moved since the last frame. Without this
 * pass, relax would read stale positions (off the new surface) and stale
 * normals (describing the old surface), corrupting the curvature-aware
 * repulsion kernel.
 *
 * In-place `read_write` on the point buffer is safe because each
 * invocation touches only index `i`. Two bind groups (A/B) cover the
 * ping-pong alternation — reproject operates on whichever buffer the next
 * relax pass will read.
 */

import type { PointBuffers } from "./point-buffers.js";
import { REPROJECT_SHADER } from "./shaders/index.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of the reproject params uniform buffer in bytes.
 *
 * WGSL struct layout (uniform):
 *   `time: f32` (offset 0) + `point_count: u32` (offset 4)
 *   + `_pad0: u32` (offset 8) + `_pad1: u32` (offset 12)
 * Struct size rounds up to 16 (alignment 16).
 */
const REPROJECT_PARAMS_BUFFER_BYTES = 16;

/**
 * Manages the reproject compute pipeline, its params uniform, and the
 * ping-pong read bind groups.
 *
 * The normals buffer is allocated once by {@link PointBuffers} and shared
 * between this pipeline (written before relax), the seed pipeline (written
 * once at bootstrap), and the shading pipeline (written after relax).
 */
export class ReprojectPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;
    private readonly pipeline: GPUComputePipeline;
    private readonly layout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;

    /**
     * Creates the reproject compute pipeline, its params uniform, and
     * both ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair plus shared
     *   normals buffer.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;
        this.points = points;

        const module = device.createShaderModule({
            label: "stipple-reproject-shader",
            code: REPROJECT_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-reproject-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "reproject_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-reproject-params",
            size: REPROJECT_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupA = device.createBindGroup({
            label: "stipple-reproject-bind-A",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
            ],
        });

        this.bindGroupB = device.createBindGroup({
            label: "stipple-reproject-bind-B",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
            ],
        });
    }

    /**
     * Writes the animation time and point count into the params uniform
     * buffer.
     *
     * Layout (16 bytes): `time: f32` + `point_count: u32` + 2 padding
     * `u32`s.
     *
     * @param time - The current animation time in seconds.
     */
    private writeParams(time: number): void {
        const buffer = new ArrayBuffer(REPROJECT_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = time;
        u32[1] = this.points.count;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a reproject compute pass into the given command encoder.
     *
     * Reads from and writes back to one point buffer (depending on
     * `readFromA`) in-place, and writes the refreshed normals into the
     * shared normals buffer. Should be called before
     * {@link RelaxPipeline.dispatch} so relax starts with on-surface
     * positions and matching normals.
     *
     * @param encoder - The command encoder to record into.
     * @param time - The current animation time in seconds.
     * @param readFromA - If `true`, operates on bufferA; else bufferB.
     *   Should match the buffer the next relax pass will read.
     */
    public dispatch(encoder: GPUCommandEncoder, time: number, readFromA: boolean): void {
        this.writeParams(time);

        const workgroupCount = Math.ceil(this.points.count / WORKGROUP_SIZE);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromA ? this.bindGroupA : this.bindGroupB);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
    }
}
