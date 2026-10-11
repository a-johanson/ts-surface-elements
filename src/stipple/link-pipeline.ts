/**
 * Link compute pipeline — connects each stipple point to its
 * best-aligned neighbor along the principal direction.
 *
 * Each frame, {@link dispatch} records a single compute pass after the
 * shading pass: one invocation per point scans the 27-cell neighborhood
 * of its spatial grid cell (the grid must have been rebuilt from the
 * post-relax positions first), scores each candidate neighbor by
 * alignment with the principal direction discounted by distance, and
 * writes the winning neighbor index — or `UINT_MAX` when no valid
 * neighbor exists — into the shared {@link LinkPipeline.linksBuffer}.
 *
 * Ping-pong: the pass reads from one point buffer (depending on
 * `readFromA` — the buffer relax most recently wrote) and writes only the
 * links buffer (not ping-ponged). Two static bind-group sets cover both
 * read directions.
 */

import type { PointBuffers } from "./point-buffers.js";
import { LINK_SHADER } from "./shaders/index.js";
import type { GridBuffers } from "./spatial-grid-pipeline.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of the link params uniform buffer in bytes (1 × f32 + 1 × u32).
 *
 * WGSL struct layout (uniform): `time: f32` (offset 0) +
 * `point_count: u32` (offset 4).
 */
const LINK_PARAMS_BUFFER_BYTES = 8;

/** Size of one link entry in bytes (one `u32` neighbor index). */
const LINK_ENTRY_BYTES = 4;

/**
 * WGSL sentinel for "no link" — matches `UINT_MAX` in the link shader.
 */
const UINT_MAX = 0xFFFFFFFF;

/**
 * Manages the link compute pipeline, its params uniform, the shared
 * links output buffer, and the ping-pong read bind groups.
 */
export class LinkPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;

    /**
     * Shared per-point link buffer (not ping-ponged): the best neighbor
     * index per point, or `UINT_MAX` when no valid neighbor exists.
     *
     * Written every frame by the link compute pass and read by the link
     * render pipeline. Pre-filled with `UINT_MAX` so a frame that renders
     * before the link compute has ever dispatched (e.g. paused at
     * startup) draws no lines instead of linking every point to index 0.
     */
    public readonly linksBuffer: GPUBuffer;

    private readonly pipeline: GPUComputePipeline;
    private readonly layout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;

    /**
     * Creates the shader module, compute pipeline, params uniform
     * buffer, links output buffer, and both ping-pong read bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair plus shared
     *   normals buffer.
     * @param grid - The spatial grid buffers (params, sorted values,
     *   cell start/count). Must have been rebuilt from the buffer the
     *   link pass will read before {@link dispatch} is called.
     */
    public constructor(device: GPUDevice, points: PointBuffers, grid: GridBuffers) {
        this.device = device;
        this.points = points;

        this.linksBuffer = device.createBuffer({
            label: "stipple-links",
            size: points.count * LINK_ENTRY_BYTES,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Uint32Array(this.linksBuffer.getMappedRange()).fill(UINT_MAX);
        this.linksBuffer.unmap();

        const module = device.createShaderModule({
            label: "stipple-link-shader",
            code: LINK_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-link-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "link_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-link-params",
            size: LINK_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupA = device.createBindGroup({
            label: "stipple-link-bind-A",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
                { binding: 3, resource: { buffer: grid.gridParamsBuffer } },
                { binding: 4, resource: { buffer: grid.sortedValues } },
                { binding: 5, resource: { buffer: grid.cellStart } },
                { binding: 6, resource: { buffer: grid.cellCount } },
                { binding: 7, resource: { buffer: this.linksBuffer } },
            ],
        });
        this.bindGroupB = device.createBindGroup({
            label: "stipple-link-bind-B",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
                { binding: 3, resource: { buffer: grid.gridParamsBuffer } },
                { binding: 4, resource: { buffer: grid.sortedValues } },
                { binding: 5, resource: { buffer: grid.cellStart } },
                { binding: 6, resource: { buffer: grid.cellCount } },
                { binding: 7, resource: { buffer: this.linksBuffer } },
            ],
        });
    }

    /**
     * Writes the animation time and point count into the link params
     * uniform buffer.
     *
     * Layout (8 bytes): `time` (f32) followed by `point_count` (u32).
     *
     * @param time - Current animation time in seconds.
     */
    private writeParams(time: number): void {
        const buffer = new ArrayBuffer(LINK_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = time;
        u32[1] = this.points.count;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a link compute pass into the given command encoder.
     *
     * Reads from one point buffer (depending on `readFromA`) — this must
     * match the buffer relax most recently wrote — plus the shared
     * normals buffer and the spatial grid rebuilt from that same buffer.
     * Writes the best-neighbor indices into the shared links buffer.
     * Should be called after the shading pass (which refreshes the
     * normals buffer from the post-relax positions) and before the
     * render pass.
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
