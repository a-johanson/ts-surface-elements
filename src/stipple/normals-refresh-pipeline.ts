/**
 * Normals-refresh compute pipeline — refreshes the shared normals buffer
 * from the post-relax positions.
 *
 * On each frame where the simulation stepped, {@link dispatch} records a
 * single compute pass that runs after the relax substeps and before the
 * link pass. One invocation per point reads the buffer relax most
 * recently wrote and writes the surface normal at that position into the
 * shared {@link PointBuffers.normalsBuffer}.
 *
 * Because the pass writes normals from the buffer relax just produced,
 * the normals buffer always matches the buffer that the *next* reproject
 * + relax cycle will read — preserving the relax invariant that normals
 * match the read-side buffer across the ping-pong swap — and the link
 * pass reads normals matching the post-relax positions. The pass is
 * view-independent, so it is skipped while time is paused (frozen
 * positions imply frozen normals).
 *
 * Ping-pong: the pass reads from one point buffer (depending on
 * `readFromA`) and writes to the shared normals buffer (not
 * ping-ponged). Two static bind-group sets cover both read directions.
 */

import type { PointBuffers } from "./point-buffers.js";
import { NORMALS_REFRESH_SHADER } from "./shaders/index.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of the normals-refresh params uniform buffer in bytes.
 *
 * WGSL struct layout (uniform):
 *   `time: f32` (offset 0) + `point_count: u32` (offset 4)
 *   + `_pad0: u32` (offset 8) + `_pad1: u32` (offset 12)
 * Struct size rounds up to 16 (alignment 16).
 */
const NORMALS_REFRESH_PARAMS_BUFFER_BYTES = 16;

/**
 * Manages the normals-refresh compute pipeline, its params uniform, and
 * the ping-pong read bind groups.
 *
 * The normals buffer is allocated once by {@link PointBuffers} and
 * shared (not ping-ponged) — written once at bootstrap by the seed pass,
 * every playing frame by the reproject pass (before relax), and every
 * playing frame by this pass (after relax).
 */
export class NormalsRefreshPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;
    private readonly pipeline: GPUComputePipeline;
    private readonly layout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;

    /**
     * Creates the shader module, compute pipeline, params uniform
     * buffer, and both ping-pong read bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair plus the shared
     *   normals buffer.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;
        this.points = points;

        const module = device.createShaderModule({
            label: "stipple-normals-refresh-shader",
            code: NORMALS_REFRESH_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-normals-refresh-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "normals_refresh_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-normals-refresh-params",
            size: NORMALS_REFRESH_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupA = device.createBindGroup({
            label: "stipple-normals-refresh-bind-A",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
            ],
        });
        this.bindGroupB = device.createBindGroup({
            label: "stipple-normals-refresh-bind-B",
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
        const buffer = new ArrayBuffer(NORMALS_REFRESH_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = time;
        u32[1] = this.points.count;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a normals-refresh compute pass into the given command
     * encoder.
     *
     * Reads from one point buffer (depending on `readFromA`) — this must
     * match the buffer relax most recently wrote — and writes the
     * refreshed normals into the shared normals buffer. Should be called
     * after the final {@link RelaxPipeline.dispatch} substep and before
     * {@link LinkPipeline.dispatch}.
     *
     * @param encoder - The command encoder to record into.
     * @param time - Current animation time in seconds.
     * @param readFromA - If `true`, reads bufferA; else bufferB. Should
     *   match the buffer relax most recently wrote (i.e. the inverse of
     *   the relax `readFromA` argument).
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
