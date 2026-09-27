/**
 * Seed compute pipeline — 3D rejection sampling in a bounding box.
 *
 * One compute dispatch generates `point_count` points distributed near
 * the SDF surface. The shader PCG-samples candidates uniformly in
 * `[bbox_min, bbox_max]³`, accepts the first within `band` of the
 * surface, and Newton-projects the result onto the surface. A
 * closest-candidate fallback guarantees a deterministic point per
 * invocation. For each accepted point the matching surface normal is
 * also written into the shared normals buffer, so the normals buffer
 * matches the seeded point buffer before the first relax frame.
 *
 * No density texture is required — the SDF is evaluated directly via the
 * shared `map` function. The pipeline runs once at bootstrap.
 */

import { SEED_SHADER } from "./shaders/index.js";
import type { SceneBBox } from "./spatial-grid-pipeline.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of the params uniform buffer in bytes.
 *
 * WGSL struct layout (uniform):
 *   `bbox_min: vec3f` (offset 0, align 16) + `band: f32` (offset 12)
 *   `bbox_max: vec3f` (offset 16, align 16) + `point_count: u32` (offset 28)
 *   `_pad0..2: vec3u` (offset 32)
 * Struct size rounds up to 48 (alignment 16).
 */
const PARAMS_BUFFER_BYTES = 48;

/**
 * Manages the seed compute pipeline, params uniform, and bind group.
 *
 * The bounding box and band are fixed at construction (the bbox is the
 * shared `SCENE_BBOX` constant; the band is set to the relaxation radius).
 * Only `point_count` varies per dispatch, written into the uniform before
 * each dispatch. The bind group is recreated lazily when the target point
 * buffer or normals buffer changes (identified by object identity).
 */
export class SeedPipeline {
    private readonly device: GPUDevice;
    private readonly sceneBBox: SceneBBox;
    private readonly band: number;
    private readonly pipeline: GPUComputePipeline;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private lastPointBuffer: GPUBuffer | null = null;
    private lastNormalsBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;

    /**
     * Creates the shader module, compute pipeline, and params uniform
     * buffer.
     *
     * @param device - The GPU device.
     * @param sceneBBox - The fixed scene bounding box (shared with the
     *   spatial grid).
     * @param band - Acceptance band for rejection sampling (`|map(p)| < band`).
     *   Set to the relaxation radius so seed density matches the relax
     *   interaction scale.
     */
    public constructor(device: GPUDevice, sceneBBox: SceneBBox, band: number) {
        this.device = device;
        this.sceneBBox = sceneBBox;
        this.band = band;

        const shaderModule = device.createShaderModule({
            label: "stipple-seed-shader",
            code: SEED_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-seed-pipeline",
            layout: "auto",
            compute: {
                module: shaderModule,
                entryPoint: "seed_cs",
            },
        });

        this.paramsBuffer = device.createBuffer({
            label: "stipple-seed-params",
            size: PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }

    /**
     * Writes the bounding box, band, and point count into the params
     * uniform buffer.
     *
     * Layout (48 bytes):
     *   `bbox_min: vec3f + band: f32`,
     *   `bbox_max: vec3f + point_count: u32`,
     *   `_pad0..2: vec3u`.
     *
     * @param pointCount - Number of points to seed.
     */
    private writeParams(pointCount: number): void {
        const buffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = this.sceneBBox.min[0];
        f32[1] = this.sceneBBox.min[1];
        f32[2] = this.sceneBBox.min[2];
        f32[3] = this.band;
        f32[4] = this.sceneBBox.max[0];
        f32[5] = this.sceneBBox.max[1];
        f32[6] = this.sceneBBox.max[2];
        u32[7] = pointCount;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a seed compute dispatch into the given command encoder.
     *
     * Writes `pointCount` points (and matching normals) into
     * `outputBuffer` / `normalsBuffer` by rejection-sampling the SDF
     * inside the bounding box.
     *
     * @param encoder - The command encoder to record into.
     * @param outputBuffer - The point storage buffer to write into.
     * @param normalsBuffer - The shared normals buffer to write into.
     * @param pointCount - Number of points to seed.
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        outputBuffer: GPUBuffer,
        normalsBuffer: GPUBuffer,
        pointCount: number,
    ): void {
        if (
            outputBuffer !== this.lastPointBuffer ||
            normalsBuffer !== this.lastNormalsBuffer
        ) {
            this.lastPointBuffer = outputBuffer;
            this.lastNormalsBuffer = normalsBuffer;
            this.bindGroup = this.device.createBindGroup({
                label: "stipple-seed-bind",
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.paramsBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: normalsBuffer } },
                ],
            });
        }

        this.writeParams(pointCount);

        if (this.bindGroup === null) {
            throw new Error("Seed bind group was not created.");
        }

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(pointCount / WORKGROUP_SIZE));
        pass.end();
    }
}
