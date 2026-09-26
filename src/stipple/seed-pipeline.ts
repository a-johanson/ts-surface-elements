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
 * shared `map` function. The pipeline runs once at bootstrap and again
 * whenever the canvas is resized (to re-seed after camera moves, though
 * seeding is camera-independent; the resize trigger is kept for
 * deterministic restarts).
 */

import { SEED_SHADER } from "./shaders/index.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the params uniform buffer in bytes.
 *
 * WGSL struct layout (uniform):
 *   `bbox_min: vec3f` (offset 0, align 16) + `band: f32` (offset 12)
 *   `bbox_max: vec3f` (offset 16, align 16) + `point_count: u32` (offset 28)
 *   `_pad0..2: vec3u` (offset 32)
 * Struct size rounds up to 48 (alignment 16).
 */
const PARAMS_BUFFER_BYTES = 48;

/** Bounding-box and band parameters uploaded to the seed shader. */
export interface SeedParams {
    /** Inclusive lower corner of the rejection-sampling bounding box. */
    readonly bboxMin: readonly [number, number, number];
    /** Inclusive upper corner of the rejection-sampling bounding box. */
    readonly bboxMax: readonly [number, number, number];
    /** Accept a candidate when `|map(p)| < band`. */
    readonly band: number;
}

/**
 * Manages the seed compute pipeline, params uniform, and bind group.
 *
 * The bind group is recreated lazily when the target point buffer or
 * normals buffer changes (identified by object identity).
 */
export class SeedPipeline {
    private readonly device: GPUDevice;
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
     */
    public constructor(device: GPUDevice) {
        this.device = device;

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
     * Layout (32 bytes):
     *   `bbox_min: vec3f + band: f32`,
     *   `bbox_max: vec3f + point_count: u32`,
     *   `_pad0..2: vec3u`.
     *
     * @param params - Bounding box and band.
     * @param pointCount - Number of points to seed.
     */
    private writeParams(params: SeedParams, pointCount: number): void {
        const buffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = params.bboxMin[0];
        f32[1] = params.bboxMin[1];
        f32[2] = params.bboxMin[2];
        f32[3] = params.band;
        f32[4] = params.bboxMax[0];
        f32[5] = params.bboxMax[1];
        f32[6] = params.bboxMax[2];
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
     * @param params - Bounding box and band.
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        outputBuffer: GPUBuffer,
        normalsBuffer: GPUBuffer,
        pointCount: number,
        params: SeedParams,
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

        this.writeParams(params, pointCount);

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
