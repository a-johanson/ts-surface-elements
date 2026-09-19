/**
 * Density compute pipeline — ray-marches the SDF scene and writes a
 * per-pixel density value into an `r32float` storage texture.
 *
 * A compute shader (rather than a fragment-shader render pass) is used
 * because `r32float` is not a renderable color-attachment format in
 * WebGPU. Storage textures support `r32float` in the core spec, so this
 * approach gives a true 32-bit float density field without compromise.
 *
 * The same GPUTexture is later bound as `texture_2d<f32>` (nearest) by
 * the debug blit, seed, and relax pipelines. The texture carries both
 * `STORAGE_BINDING` and `TEXTURE_BINDING` usage flags.
 *
 * The texture is recreated automatically when the canvas size changes.
 */

import { DENSITY_SHADER } from "./shaders.js";

/** Workgroup size — must match `@workgroup_size(8, 8, 1)` in the WGSL. */
const WORKGROUP_SIZE_X = 8;
const WORKGROUP_SIZE_Y = 8;

/** Camera uniform buffer size in bytes (5 × 16-byte aligned members = 80). */
const CAMERA_BUFFER_BYTES = 80;

/** Static camera configuration — scene-independent lens parameters. */
export interface CameraConfig {
    /** Vertical field of view in radians. */
    readonly fov: number;
    /** Look-at target in world space. */
    readonly target: readonly [number, number, number];
    /** World-space up direction. */
    readonly up: readonly [number, number, number];
}

/**
 * Manages the density compute pipeline, camera uniform, and the
 * canvas-sized `r32float` storage texture.
 *
 * The pipeline is created once at construction time. Each frame, the
 * caller invokes {@link dispatch} to record a compute pass into the
 * current frame's command encoder. The texture is recreated on resize.
 */
export class DensityPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly cameraBuffer: GPUBuffer;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly config: CameraConfig;
    private texture: GPUTexture | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private textureWidth = 0;
    private textureHeight = 0;

    /**
     * Creates the shader module, compute pipeline, and camera uniform
     * buffer. The storage texture is created lazily on the first
     * {@link dispatch} call (once the canvas has a non-zero size).
     *
     * @param device - The GPU device.
     * @param config - Static camera configuration (FOV, target, up).
     */
    public constructor(device: GPUDevice, config: CameraConfig) {
        this.device = device;
        this.config = config;

        const shaderModule = device.createShaderModule({
            label: "stipple-density-shader",
            code: DENSITY_SHADER,
        });

        this.pipeline = device.createComputePipeline({
            label: "stipple-density-pipeline",
            layout: "auto",
            compute: {
                module: shaderModule,
                entryPoint: "density_cs",
            },
        });

        this.cameraBuffer = device.createBuffer({
            label: "stipple-density-camera",
            size: CAMERA_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }

    /**
     * Recreates the storage texture if the canvas dimensions have
     * changed, and rebuilds the bind group to reference the new texture
     * view.
     *
     * @param width - Current canvas backing-store width.
     * @param height - Current canvas backing-store height.
     */
    private ensureTexture(width: number, height: number): void {
        if (
            width === this.textureWidth &&
            height === this.textureHeight &&
            this.texture !== null
        ) {
            return;
        }

        this.texture?.destroy();

        this.texture = this.device.createTexture({
            label: "stipple-density-texture",
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.textureWidth = width;
        this.textureHeight = height;

        this.bindGroup = this.device.createBindGroup({
            label: "stipple-density-bind",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.cameraBuffer } },
                { binding: 1, resource: this.texture.createView() },
            ],
        });
    }

    /**
     * Returns the current density texture.
     *
     * The texture exists only after the first {@link dispatch} call.
     *
     * @returns The `r32float` storage texture.
     * @throws {Error} If the texture has not been created yet.
     */
    public getTexture(): GPUTexture {
        if (this.texture === null) {
            throw new Error("Density texture has not been created yet.");
        }
        return this.texture;
    }

    /**
     * Writes the camera rays uniform for the current frame.
     *
     * Computes forward/right/up basis vectors and half-FOV extents from
     * the eye position, target, up vector, FOV, and canvas aspect ratio.
     * Packs the result into five 16-byte aligned struct members:
     *   `eye: vec3f + half_width: f32`,
     *   `forward: vec3f + half_height: f32`,
     *   `right: vec3f + _pad0: f32`,
     *   `up: vec3f + _pad1: f32`,
     *   `resolution: vec2u + _pad2: vec2u`.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param width - Canvas backing-store width.
     * @param height - Canvas backing-store height.
     */
    private writeCamera(
        eye: readonly [number, number, number],
        width: number,
        height: number,
    ): void {
        const { target, up, fov } = this.config;

        let fx = target[0] - eye[0];
        let fy = target[1] - eye[1];
        let fz = target[2] - eye[2];
        const fLen = Math.hypot(fx, fy, fz);
        if (fLen === 0) {
            fx = 0;
            fy = 0;
            fz = -1;
        } else {
            fx /= fLen;
            fy /= fLen;
            fz /= fLen;
        }

        let rx = fy * up[2] - fz * up[1];
        let ry = fz * up[0] - fx * up[2];
        let rz = fx * up[1] - fy * up[0];
        const rLen = Math.hypot(rx, ry, rz);
        if (rLen === 0) {
            rx = 1;
            ry = 0;
            rz = 0;
        } else {
            rx /= rLen;
            ry /= rLen;
            rz /= rLen;
        }

        const ux = ry * fz - rz * fy;
        const uy = rz * fx - rx * fz;
        const uz = rx * fy - ry * fx;

        const aspect = width / height;
        const halfH = Math.tan(fov / 2);
        const halfW = halfH * aspect;

        const buffer = new ArrayBuffer(CAMERA_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = eye[0];
        f32[1] = eye[1];
        f32[2] = eye[2];
        f32[3] = halfW;
        f32[4] = fx;
        f32[5] = fy;
        f32[6] = fz;
        f32[7] = halfH;
        f32[8] = rx;
        f32[9] = ry;
        f32[10] = rz;
        f32[11] = 0;
        f32[12] = ux;
        f32[13] = uy;
        f32[14] = uz;
        f32[15] = 0;
        u32[16] = width;
        u32[17] = height;
        u32[18] = 0;
        u32[19] = 0;

        this.device.queue.writeBuffer(this.cameraBuffer, 0, buffer);
    }

    /**
     * Records a density compute pass into the given command encoder.
     *
     * Ensures the storage texture matches the current canvas size,
     * writes the camera uniform, then dispatches enough 8×8 workgroups
     * to cover every pixel.
     *
     * @param encoder - The command encoder to record into.
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   dispatch dimensions.
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
    ): void {
        this.ensureTexture(canvas.width, canvas.height);
        this.writeCamera(eye, canvas.width, canvas.height);

        if (this.bindGroup === null) {
            throw new Error("Density bind group was not created.");
        }

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(canvas.width / WORKGROUP_SIZE_X),
            Math.ceil(canvas.height / WORKGROUP_SIZE_Y),
        );
        pass.end();
    }
}
