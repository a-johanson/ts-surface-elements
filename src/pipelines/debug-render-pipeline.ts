/**
 * Debug render pipeline — ray-marches the SDF per fragment and writes
 * grayscale Lambert shading directly to the canvas color attachment.
 *
 * The SDF is visualized by a single full-screen fragment shader that
 * ray-marches `map()` and shades by Lambert. The pipeline owns its
 * `CameraRays` uniform (eye + basis + half-FOV extents + resolution),
 * written each frame from the orbit camera's eye position.
 */

import { type CameraConfig, computeCameraRays } from "./camera.js";
import { DEBUG_RENDER_SHADER } from "./shaders/debug-render-shader.js";

/** Camera uniform buffer size in bytes (5 × 16-byte aligned members = 80). */
const CAMERA_BUFFER_BYTES = 80;

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
     * @param config - Static camera configuration (FOV, up).
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
            label: "debug-render-pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "debug_vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "debug_fs",
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
     * Derives the orthonormal basis and half-FOV extents via
     * {@link computeCameraRays}, then packs the basis, the animation
     * time, and the canvas resolution into five 16-byte aligned struct
     * members:
     *   `eye: vec3f + half_width: f32`,
     *   `forward: vec3f + half_height: f32`,
     *   `right: vec3f + _pad0: f32`,
     *   `up: vec3f + time: f32`,
     *   `resolution: vec2u + _pad2: vec2u`.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param target - Look-at target point [x, y, z].
     * @param width - Canvas backing-store width.
     * @param height - Canvas backing-store height.
     * @param time - Current animation time in seconds.
     */
    private writeCamera(
        eye: readonly [number, number, number],
        target: readonly [number, number, number],
        width: number,
        height: number,
        time: number,
    ): void {
        const camera = computeCameraRays(
            eye,
            target,
            this.config.up,
            this.config.fov,
            width / height,
        );

        const buffer = new ArrayBuffer(CAMERA_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = eye[0];
        f32[1] = eye[1];
        f32[2] = eye[2];
        f32[3] = camera.halfWidth;
        f32[4] = camera.forward[0];
        f32[5] = camera.forward[1];
        f32[6] = camera.forward[2];
        f32[7] = camera.halfHeight;
        f32[8] = camera.right[0];
        f32[9] = camera.right[1];
        f32[10] = camera.right[2];
        f32[11] = 0;
        f32[12] = camera.up[0];
        f32[13] = camera.up[1];
        f32[14] = camera.up[2];
        f32[15] = time;
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
     * @param target - Look-at target point [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   ray direction per fragment.
     * @param time - Current animation time in seconds.
     */
    public render(
        pass: GPURenderPassEncoder,
        eye: readonly [number, number, number],
        target: readonly [number, number, number],
        canvas: HTMLCanvasElement,
        time: number,
    ): void {
        this.writeCamera(eye, target, canvas.width, canvas.height, time);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
    }
}
