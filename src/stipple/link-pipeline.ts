/**
 * Link compute pipeline — connects each stipple point to its
 * best-aligned neighbor along the principal direction, canonicalized by
 * target, and emits a surface-projected subdivided curve per link.
 *
 * Each frame, {@link dispatch} records one compute pass with three
 * dispatches, run after the post-relax spatial grid rebuild:
 *
 *  1. **Owner clear** — resets the `owners` buffer to `UINT_MAX`.
 *  2. **Scan + elect** — one invocation per point scans the 27-cell
 *     neighborhood of its spatial grid cell, scores each candidate
 *     neighbor by alignment with the principal direction discounted by
 *     distance, writes the winning neighbor index — or `UINT_MAX` when
 *     no valid neighbor exists — into the internal
 *     {@link LinkPipeline.linksBuffer}, and elects each target's owner
 *     via `atomicMin` (after the dispatch, `owners[j]` is the smallest
 *     source index targeting `j`).
 *  3. **Dedup + subdiv** — one invocation per point evaluates the
 *     edge-survival predicate — owner election plus the mutual 2-cycle
 *     break, dropping the larger-index edge of each pair (decision 9 in
 *     the hatch-lines spec) — to decide whether the point's link
 *     survived; `links` itself is not mutated, it stays the raw scan
 *     output so the chain-window reads below are race-free (decision
 *     11). For surviving links it samples a cubic Hermite segment
 *     between the two stipple endpoints with C1 chain-window tangents
 *     (Step 2c in the hatch-lines spec), Newton-projects the interior
 *     samples onto the surface, and writes the curve points into
 *     {@link LinkPipeline.linkCurves}. Dropped or absent links zero
 *     their slice.
 *
 * WebGPU orders dispatches within a compute pass, so each stage's
 * writes are visible to the next without explicit barriers.
 *
 * Ping-pong: the scan and the curve subdivision both read from one
 * point buffer (depending on `readFromA` — the buffer relax most
 * recently wrote); the clear stage touches only the shared owners
 * buffer. Static bind groups cover the ping-pong read directions of
 * both stages.
 */

import type { PointBuffers } from "./point-buffers.js";
import { LINK_N_SEGMENTS, LINK_SHADER } from "./shaders/index.js";
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

/** Curve points per link (`LINK_N_SEGMENTS + 1`). */
const LINK_CURVE_STRIDE = LINK_N_SEGMENTS + 1;

/** Size of one curve point in bytes (one `vec4f`). */
const CURVE_POINT_BYTES = 16;

/**
 * WGSL sentinel for "no link" — matches `UINT_MAX` in the link shader.
 */
const UINT_MAX = 0xffffffff;

/**
 * Manages the link compute pipeline, its params uniform, the links and
 * owners working buffers, the curve output buffer, and the bind groups.
 */
export class LinkPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;

    /**
     * Per-point link buffer (not ping-ponged): the best neighbor index
     * per point, or `UINT_MAX` when no valid neighbor exists. The scan
     * stage writes every entry and is the buffer's final writer — it
     * stays the raw scan output, and consumers (the subdivision's
     * chain-window fetches and the render's interim coarse cull)
     * evaluate the survival predicate from `links` + `owners`
     * themselves (decisions 9 and 11 in the hatch-lines spec). Pre-filled
     * with `UINT_MAX` for determinism before the first dispatch.
     *
     * Internal working state, exposed readonly for the link render's
     * interim coarse visibility cull (Step 2b in the hatch-lines spec);
     * the render-side use goes away in Step 4, when per-segment
     * clearance is baked into the curve buffer.
     */
    public readonly linksBuffer: GPUBuffer;

    /**
     * Shared per-point curve buffer (not ping-ponged): for each point,
     * one `vec4f` per curve point — `.xyz` is the surface-projected
     * position, `.w` the validity flag (1.0 valid, 0.0 dropped or
     * absent link). Written every frame by the dedup + subdiv stage
     * and read by the link render pipeline. Zero-initialized, so
     * `.w = 0.0` renders nothing before the link compute has ever
     * dispatched (e.g. paused at startup).
     */
    public readonly linkCurves: GPUBuffer;

    private readonly pipeline: GPUComputePipeline;
    private readonly layout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;
    private readonly clearPipeline: GPUComputePipeline;
    private readonly clearBindGroup: GPUBindGroup;
    private readonly dedupSubdivPipeline: GPUComputePipeline;
    private readonly dedupSubdivBindGroupA: GPUBindGroup;
    private readonly dedupSubdivBindGroupB: GPUBindGroup;

    /**
     * Shared per-target owner buffer (not ping-ponged, internal): after
     * the election dispatch, `owners[j]` is the smallest source index
     * targeting `j`, or `UINT_MAX` when no source targets `j`. Cleared
     * to `UINT_MAX` at the start of every dispatch.
     */
    private readonly ownersBuffer: GPUBuffer;

    /**
     * Creates the shader module, compute pipelines, params uniform
     * buffer, curve output buffer, and the bind groups.
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

        this.ownersBuffer = device.createBuffer({
            label: "stipple-link-owners",
            size: points.count * LINK_ENTRY_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

        this.linkCurves = device.createBuffer({
            label: "stipple-link-curves",
            size: points.count * LINK_CURVE_STRIDE * CURVE_POINT_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });

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
        this.clearPipeline = device.createComputePipeline({
            label: "stipple-link-owners-clear-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "link_owner_clear_cs",
            },
        });
        this.dedupSubdivPipeline = device.createComputePipeline({
            label: "stipple-link-dedup-subdiv-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "link_dedup_subdiv_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);
        const clearLayout = this.clearPipeline.getBindGroupLayout(0);
        const dedupSubdivLayout = this.dedupSubdivPipeline.getBindGroupLayout(0);

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
                { binding: 8, resource: { buffer: this.ownersBuffer } },
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
                { binding: 8, resource: { buffer: this.ownersBuffer } },
            ],
        });

        this.clearBindGroup = device.createBindGroup({
            label: "stipple-link-owners-clear-bind",
            layout: clearLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: this.ownersBuffer } },
            ],
        });
        this.dedupSubdivBindGroupA = device.createBindGroup({
            label: "stipple-link-dedup-subdiv-bind-A",
            layout: dedupSubdivLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 7, resource: { buffer: this.linksBuffer } },
                { binding: 8, resource: { buffer: this.ownersBuffer } },
                { binding: 9, resource: { buffer: this.linkCurves } },
            ],
        });
        this.dedupSubdivBindGroupB = device.createBindGroup({
            label: "stipple-link-dedup-subdiv-bind-B",
            layout: dedupSubdivLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 7, resource: { buffer: this.linksBuffer } },
                { binding: 8, resource: { buffer: this.ownersBuffer } },
                { binding: 9, resource: { buffer: this.linkCurves } },
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
     * Records the link compute stages into the given command encoder as
     * one compute pass with three dispatches: owner clear → scan+elect →
     * dedup+subdiv.
     *
     * The scan and the curve subdivision read from one point buffer
     * (depending on `readFromA`) — this must match the buffer relax most
     * recently wrote — plus, for the scan, the shared normals buffer and
     * the spatial grid rebuilt from that same buffer. The dedup+subdiv
     * stage evaluates the survival predicate without mutating `links`
     * and writes the Hermite-sampled, surface-projected curve points
     * into the shared curve buffer. Should be called after the shading
     * pass (which refreshes the normals buffer from the post-relax
     * positions) and before the render pass.
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

        pass.setPipeline(this.clearPipeline);
        pass.setBindGroup(0, this.clearBindGroup);
        pass.dispatchWorkgroups(workgroupCount);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromA ? this.bindGroupA : this.bindGroupB);
        pass.dispatchWorkgroups(workgroupCount);

        pass.setPipeline(this.dedupSubdivPipeline);
        pass.setBindGroup(
            0,
            readFromA ? this.dedupSubdivBindGroupA : this.dedupSubdivBindGroupB,
        );
        pass.dispatchWorkgroups(workgroupCount);

        pass.end();
    }
}
