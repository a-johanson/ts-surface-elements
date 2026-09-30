/**
 * Shading compute pipeline — per-point visibility, shadow, and normal
 * refresh.
 *
 * Each frame, {@link dispatch} records a single compute pass that runs
 * after relax and before the point render. One invocation per point
 * reads the buffer relax most recently wrote, computes the surface
 * normal from the (final) position, writes it into the shared
 * {@link PointBuffers.normalsBuffer}, then sphere-traces toward the eye
 * to sample per-corner occlusion clearances and toward the light (soft
 * shadow) and writes the result into the shared
 * {@link PointBuffers.shadingBuffer}.
 *
 * Because the shading pass writes normals from the buffer relax just
 * produced, the normals buffer always matches the buffer that the
 * *next* reproject + relax cycle will read — preserving the relax
 * invariant that normals match the read-side buffer across the ping-pong
 * swap.
 *
 * Ping-pong: the pass reads from one point buffer (depending on
 * `readFromA`) and writes to the shared normals and shading buffers
 * (neither is ping-ponged). Two static bind-group sets cover both
 * read directions.
 */

import type { PointBuffers } from "./point-buffers.js";
import { SHADING_SHADER } from "./shaders/index.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of the shading params uniform buffer in bytes.
 *
 * WGSL struct layout (uniform):
 *   `eye: vec3f` (offset 0, align 16) + `point_count: u32` (offset 12)
 *   `light_dir: vec3f` (offset 16, align 16) + `time: f32` (offset 28)
 * Struct size rounds up to 32 (alignment 16).
 */
const SHADING_PARAMS_BUFFER_BYTES = 32;

/**
 * Default light direction — matches the value previously hardcoded in
 * the debug render shader (`normalize(vec3f(0.5, 0.8, 0.6))`).
 */
export const DEFAULT_LIGHT_DIR: readonly [number, number, number] = (() => {
    const x = 0.5;
    const y = 0.8;
    const z = 0.6;
    const len = Math.hypot(x, y, z);
    return [x / len, y / len, z / len];
})();

/**
 * Manages the shading compute pipeline, its params uniform, and the
 * ping-pong read bind groups.
 *
 * The normals and shading buffers are allocated once by
 * {@link PointBuffers} and shared (not ping-ponged) — both are written
 * by this pass every frame.
 */
export class ShadingPipeline {
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
     * @param points - The ping-pong point buffer pair plus shared
     *   normals and shading buffers.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;
        this.points = points;

        const module = device.createShaderModule({
            label: "stipple-shading-shader",
            code: SHADING_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-shading-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "shading_cs",
            },
        });

        this.layout = this.pipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-shading-params",
            size: SHADING_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupA = device.createBindGroup({
            label: "stipple-shading-bind-A",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
                { binding: 3, resource: { buffer: points.shadingBuffer } },
            ],
        });
        this.bindGroupB = device.createBindGroup({
            label: "stipple-shading-bind-B",
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: points.normalsBuffer } },
                { binding: 3, resource: { buffer: points.shadingBuffer } },
            ],
        });
    }

    /**
     * Writes the eye position, light direction, and point count into the
     * params uniform buffer.
     *
     * Layout (32 bytes):
     *   `eye: vec3f + point_count: u32`,
     *   `light_dir: vec3f + time: f32`.
     *
     * @param eye - Camera eye position in world space.
     * @param lightDir - Normalized light direction in world space.
     * @param time - Current animation time in seconds.
     */
    private writeParams(
        eye: readonly [number, number, number],
        lightDir: readonly [number, number, number],
        time: number,
    ): void {
        const buffer = new ArrayBuffer(SHADING_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = eye[0];
        f32[1] = eye[1];
        f32[2] = eye[2];
        u32[3] = this.points.count;
        f32[4] = lightDir[0];
        f32[5] = lightDir[1];
        f32[6] = lightDir[2];
        f32[7] = time;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a shading compute pass into the given command encoder.
     *
     * Reads from one point buffer (depending on `readFromA`) — this must
     * match the buffer relax most recently wrote. Writes the refreshed
     * normals and the per-point shading result into the shared buffers.
     * Should be called after {@link RelaxPipeline.dispatch} and before the
     * render pass.
     *
     * @param encoder - The command encoder to record into.
     * @param eye - Camera eye position in world space.
     * @param lightDir - Normalized light direction in world space.
     * @param time - Current animation time in seconds.
     * @param readFromA - If `true`, reads bufferA; else bufferB. Should
     *   match the buffer relax most recently wrote (i.e. the inverse of
     *   the relax `readFromA` argument).
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        eye: readonly [number, number, number],
        lightDir: readonly [number, number, number],
        time: number,
        readFromA: boolean,
    ): void {
        this.writeParams(eye, lightDir, time);

        const workgroupCount = Math.ceil(this.points.count / WORKGROUP_SIZE);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromA ? this.bindGroupA : this.bindGroupB);
        pass.dispatchWorkgroups(workgroupCount);
        pass.end();
    }
}
