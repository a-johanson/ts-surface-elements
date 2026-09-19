/**
 * Seed compute pipeline — GPU-side rejection sampling of the initial
 * point distribution.
 *
 * One compute dispatch generates `point_count` points whose spatial
 * distribution follows the density texture. The shader uses a PCG hash
 * to generate pseudo-random candidates and accepts each with probability
 * equal to the density value at that location (see `stippling.md` for
 * the full algorithm).
 *
 * The seed pipeline runs once at bootstrap and again whenever the canvas
 * is resized (since the density texture changes shape). It must run
 * *after* the density compute pass in the same command encoder, since it
 * samples the density texture.
 */

import { SEED_SHADER } from "./shaders.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the params uniform buffer in bytes (4 × u32 = 16). */
const PARAMS_BUFFER_BYTES = 16;

/**
 * Manages the seed compute pipeline, params uniform, and bind group.
 *
 * The bind group is recreated lazily when either the density texture or
 * the target point buffer changes (both are identified by object
 * identity). This handles the resize case (new texture) and the future
 * ping-pong case (alternating target buffer).
 */
export class SeedPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly paramsBuffer: GPUBuffer;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private lastTexture: GPUTexture | null = null;
    private lastBuffer: GPUBuffer | null = null;
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
     * Writes the point count into the params uniform buffer.
     *
     * @param pointCount - Number of points to seed.
     */
    private writeParams(pointCount: number): void {
        const buffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
        const u32 = new Uint32Array(buffer);
        u32[0] = pointCount;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Records a seed compute dispatch into the given command encoder.
     *
     * Writes `pointCount` points into `outputBuffer` by rejection-sampling
     * the density texture. Must be called after the density pass has
     * been recorded into the same encoder.
     *
     * @param encoder - The command encoder to record into.
     * @param densityTexture - The `r32float` density texture to sample.
     * @param outputBuffer - The point storage buffer to write into.
     * @param pointCount - Number of points to seed.
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        densityTexture: GPUTexture,
        outputBuffer: GPUBuffer,
        pointCount: number,
    ): void {
        if (densityTexture !== this.lastTexture || outputBuffer !== this.lastBuffer) {
            this.lastTexture = densityTexture;
            this.lastBuffer = outputBuffer;
            this.bindGroup = this.device.createBindGroup({
                label: "stipple-seed-bind",
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: densityTexture.createView() },
                    { binding: 1, resource: { buffer: outputBuffer } },
                    { binding: 2, resource: { buffer: this.paramsBuffer } },
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
