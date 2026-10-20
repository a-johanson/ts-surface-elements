/**
 * Hatch compute pipeline — grows a surface-projected hatch stroke from
 * every stipple point by integrating the principal direction field
 * (forward and backward from the point), and refreshes the stroke
 * points' view-dependent occlusion clearance.
 *
 * Each frame, {@link dispatch} records one compute pass with a single
 * dispatch on frames where the simulation stepped; {@link dispatchClearance}
 * records a second single-dispatch compute pass *every* frame (playing
 * and paused):
 *
 *  1. **Growth** (`hatch_cs`) — one invocation per point. From the
 *     stipple point, the pass marches the principal direction field —
 *     the tangent-plane direction orthogonal to the light's tangent
 *     projection, recomputed at every step point from a fresh surface
 *     normal — taking `HATCH_N_SEGMENTS_HALF` `HATCH_STEP_SIZE` steps in
 *     each direction, Newton-projecting every step onto the surface.
 *     The result is a `2 * HATCH_N_SEGMENTS_HALF + 1`-point polyline per
 *     point, centered on the stipple point, written into
 *     {@link HatchPipeline.hatchCurves} (with a validity placeholder in
 *     `.w`). Arms that hit a direction degeneracy get a zero direction
 *     and walk nowhere — their remaining slots all hold the same
 *     standing point, so every slot always holds a real surface point.
 *     The pass also writes each stroke point's surface normal into
 *     {@link HatchPipeline.hatchCurveNormals} — the center slot gets
 *     the stipple point's own normal from the shared normals buffer,
 *     grown points get the fresh surface normals that steered the walk.
 *  2. **Clearance** (`hatch_clearance_cs`) — one invocation per point,
 *     recorded by {@link dispatchClearance} as its own compute pass
 *     every frame. It traces one `rayClearance` per stroke point toward
 *     the eye, with each stroke point's surface normal taken from
 *     {@link HatchPipeline.hatchCurveNormals} and its local stroke
 *     spacing as the clearance threshold — writing the clearance into
 *     the `.w` component. Occlusion is view-dependent, so the traces must
 *     refresh even on paused frames — that is what keeps the hatch
 *     render's culling and soft fade tracking the camera while the
 *     stroke geometry is frozen.
 *
 * On playing frames the clearance pass follows the growth pass in the
 * same command encoder, so it sees the freshly written stroke slices.
 *
 * Ping-pong: growth reads one point buffer (depending on `readFromA` —
 * the buffer relax most recently wrote); clearance reads only shared
 * (non-ping-ponged) buffers. Static bind groups cover the ping-pong
 * read directions of growth.
 *
 * Unlike the link pipeline this module replaces, the hatch pass needs
 * no spatial grid: strokes are grown by marching the direction field,
 * not by scanning point neighborhoods.
 */

import type { PointBuffers } from "./point-buffers.js";
import { HATCH_N_SEGMENTS_HALF, HATCH_SHADER } from "./shaders/index.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of the hatch params uniform buffer in bytes.
 *
 * WGSL struct layout (uniform): `eye: vec3f` (offset 0, align 16) +
 * `time: f32` (offset 12) + `point_count: u32` (offset 16) +
 * `_pad: u32` (offset 20). Struct size rounds up to 32 (alignment 16).
 * The eye drives the clearance stage's occlusion traces; growth uses
 * only time and point count.
 */
const HATCH_PARAMS_BUFFER_BYTES = 32;

/** Stroke points per stipple point (`2 * HATCH_N_SEGMENTS_HALF + 1`). */
const HATCH_CURVE_STRIDE = 2 * HATCH_N_SEGMENTS_HALF + 1;

/** Size of one stroke point in bytes (one `vec4f`). */
const CURVE_POINT_BYTES = 16;

/**
 * Manages the hatch compute pipeline, its params uniform, the stroke
 * output buffers, and the bind groups.
 */
export class HatchPipeline {
    private readonly device: GPUDevice;
    private readonly points: PointBuffers;

    /**
     * Shared per-point stroke buffer (not ping-ponged): for each point,
     * one `vec4f` per stroke point — `.xyz` is the surface-projected
     * position, `.w` the per-stroke-point occlusion clearance toward
     * the eye (> 0 visible, < 0 occluded). The growth stage writes
     * positions and a validity placeholder in `.w` on playing frames;
     * the clearance stage refreshes `.w` every frame (view-dependent).
     * Read by the hatch render pipeline. Zero-initialized, so
     * `.w = 0.0` renders nothing before the hatch compute has ever
     * dispatched (e.g. paused at startup).
     */
    public readonly hatchCurves: GPUBuffer;

    /**
     * Per-stroke-point surface normals, slot-aligned with
     * {@link hatchCurves}: one `vec4f` per stroke point — `.xyz` the
     * surface normal, `.w` unused. Written by the growth stage on
     * playing frames; read by the clearance stage every frame.
     * Zero-initialized, so before the first growth dispatch every
     * normal is zero and the clearance stage's back-face guard
     * deterministically culls every slot.
     */
    private readonly hatchCurveNormals: GPUBuffer;

    private readonly pipeline: GPUComputePipeline;
    private readonly layout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;
    private readonly clearancePipeline: GPUComputePipeline;
    private readonly clearanceBindGroup: GPUBindGroup;

    /**
     * Creates the shader module, compute pipelines, params uniform
     * buffer, stroke output buffer, and the bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair plus shared
     *   normals buffer.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;
        this.points = points;

        const curveBytes = points.count * HATCH_CURVE_STRIDE * CURVE_POINT_BYTES;
        this.hatchCurves = device.createBuffer({
            label: "stipple-hatch-curves",
            size: curveBytes,
            usage: GPUBufferUsage.STORAGE,
        });
        this.hatchCurveNormals = device.createBuffer({
            label: "stipple-hatch-curve-normals",
            size: curveBytes,
            usage: GPUBufferUsage.STORAGE,
        });

        const module = device.createShaderModule({
            label: "stipple-hatch-shader",
            code: HATCH_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-hatch-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "hatch_cs",
            },
        });
        this.clearancePipeline = device.createComputePipeline({
            label: "stipple-hatch-clearance-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "hatch_clearance_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);
        const clearanceLayout = this.clearancePipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-hatch-params",
            size: HATCH_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupA = device.createBindGroup({
            label: "stipple-hatch-bind-A",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
                { binding: 3, resource: { buffer: this.hatchCurves } },
                { binding: 4, resource: { buffer: this.hatchCurveNormals } },
            ],
        });
        this.bindGroupB = device.createBindGroup({
            label: "stipple-hatch-bind-B",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
                { binding: 3, resource: { buffer: this.hatchCurves } },
                { binding: 4, resource: { buffer: this.hatchCurveNormals } },
            ],
        });

        // The clearance stage reads only the growth stage's outputs —
        // stroke points and their per-slot surface normals — plus the
        // params uniform, so a single static bind group covers it. It
        // reads no ping-ponged point buffer and not the shared normals
        // buffer either: the growth stage snapshots the stipple point's
        // own normal into the slot-normals buffer. Its derived layout
        // omits the growth stage's points_in and normals_in bindings,
        // which the clearance entry point does not use.
        this.clearanceBindGroup = device.createBindGroup({
            label: "stipple-hatch-clearance-bind",
            layout: clearanceLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 3, resource: { buffer: this.hatchCurves } },
                { binding: 4, resource: { buffer: this.hatchCurveNormals } },
            ],
        });
    }

    /**
     * Writes the eye position, animation time, and point count into
     * the hatch params uniform buffer.
     *
     * Layout (32 bytes): `eye: vec3f` + `time: f32`, then
     * `point_count: u32` + `_pad: u32`. Called by both {@link dispatch}
     * and {@link dispatchClearance}; the writes are idempotent.
     *
     * @param eye - Camera eye position in world space.
     * @param time - Current animation time in seconds.
     */
    private writeParams(eye: readonly [number, number, number], time: number): void {
        const buffer = new ArrayBuffer(HATCH_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = eye[0];
        f32[1] = eye[1];
        f32[2] = eye[2];
        f32[3] = time;
        u32[4] = this.points.count;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records the hatch growth stage into the given command encoder as
     * one single-dispatch compute pass.
     *
     * Reads one point buffer (depending on `readFromA`) — the buffer
     * relax most recently wrote — plus the shared normals buffer, and
     * writes the surface-projected stroke points (with a validity
     * placeholder in `.w`) into the shared stroke buffer, along with
     * each stroke point's surface normal into the shared slot-normals
     * buffer. Should be
     * called after the normals-refresh pass (which refreshes the
     * normals buffer from the post-relax positions), before
     * {@link dispatchClearance}, and only on frames where the
     * simulation stepped — while time is paused the stroke inputs are
     * unchanged and the last strokes stay valid.
     *
     * @param encoder - The command encoder to record into.
     * @param eye - Camera eye position in world space (written to the
     *   params uniform for the clearance stage).
     * @param time - Current animation time in seconds.
     * @param readFromA - If `true`, reads bufferA; else bufferB. Should
     *   match the buffer relax most recently wrote (i.e. the inverse of
     *   the relax `readFromA` argument).
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        eye: readonly [number, number, number],
        time: number,
        readFromA: boolean,
    ): void {
        this.writeParams(eye, time);

        const workgroupCount = Math.ceil(this.points.count / WORKGROUP_SIZE);

        const pass = encoder.beginComputePass();

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromA ? this.bindGroupA : this.bindGroupB);
        pass.dispatchWorkgroups(workgroupCount);

        pass.end();
    }

    /**
     * Records the hatch clearance stage into the given command encoder
     * as its own single-dispatch compute pass. Per point it traces one
     * `rayClearance` per stroke point toward the eye — biased origin,
     * back-face guard, and each stroke point's surface normal taken
     * from the slot-normals buffer the growth stage wrote — writing
     * the clearance into the stroke buffer's `.w` fields.
     *
     * Runs every frame — including while time is paused — because the
     * clearance is view-dependent: on paused frames it refreshes the
     * occlusion of the frozen stroke geometry against the moving
     * camera, which is what keeps the hatch render's culling and soft
     * fade live. On playing frames it must follow {@link dispatch} so
     * it sees the freshly written stroke slices.
     *
     * @param encoder - The command encoder to record into.
     * @param eye - Camera eye position in world space (the trace
     *   target).
     * @param time - Current animation time in seconds.
     */
    public dispatchClearance(
        encoder: GPUCommandEncoder,
        eye: readonly [number, number, number],
        time: number,
    ): void {
        this.writeParams(eye, time);

        const workgroupCount = Math.ceil(this.points.count / WORKGROUP_SIZE);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.clearancePipeline);
        pass.setBindGroup(0, this.clearanceBindGroup);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
    }
}
