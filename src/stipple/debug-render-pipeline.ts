/**
 * Debug render pipeline — ray-marches the SDF per fragment and writes
 * grayscale Lambert shading directly to the canvas color attachment.
 *
 * Replaces the former density-compute + blit pair. With surface-space
 * points nothing consumes a density texture, so the SDF is visualized by a
 * single full-screen fragment shader that ray-marches `map()` and shades
 * by Lambert. The pipeline owns its `CameraRays` uniform (eye + basis +
 * half-FOV extents + resolution), written each frame from the orbit
 * camera's eye position.
 */

import { DEBUG_RENDER_SHADER } from "./shaders.js";

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
 * Manages the debug render pipeline and the `CameraRays` uniform.
 *
 * The pipeline is created once at construction time. Each frame, the
 * caller invokes {@link render} from inside an active render pass; the
 * camera uniform is uploaded beforehand via {@link writeCamera}.
 */
export class DebugRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly cameraBuffer: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;
    private readonly config: CameraConfig;

    /**
     * Creates the shader module, render pipeline, camera uniform buffer,
     * and bind group.
     *
     * @param device - The GPU device.
     * @param config - Static camera configuration (FOV, target, up).
     * @param format - The canvas texture format (e.g. `bgra8unorm`).
     */
    public constructor(device: GPUDevice, config: CameraConfig, format: GPUTextureFormat) {
        this.device = device;
        this.config = config;

        const shaderModule = device.createShaderModule({
            label: "stipple-debug-render-shader",
            code: DEBUG_RENDER_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "stipple-debug-render-pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "blit_vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "blit_fs",
                targets: [{ format }],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        this.cameraBuffer = device.createBuffer({
            label: "stipple-debug-camera",
            size: CAMERA_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const layout = this.pipeline.getBindGroupLayout(0);
        this.bindGroup = device.createBindGroup({
            label: "stipple-debug-render-bind",
            layout,
            entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }],
        });
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
     * Draws the SDF debug view (grayscale Lambert) as a full-screen pass.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`. Uploads
     * the camera uniform before drawing.
     *
     * @param pass - The active render pass encoder.
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   ray direction per fragment.
     */
    public render(
        pass: GPURenderPassEncoder,
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
    ): void {
        this.writeCamera(eye, canvas.width, canvas.height);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
    }
}
