/**
 * Debug render pipeline — ray-marches the SDF per fragment and writes
 * grayscale Lambert shading directly to the canvas color attachment, then
 * overlays a line-list wireframe of the scene bounding box on top.
 *
 * Replaces the former density-compute + blit pair. With surface-space
 * points nothing consumes a density texture, so the SDF is visualized by a
 * single full-screen fragment shader that ray-marches `map()` and shades
 * by Lambert. The pipeline owns its `CameraRays` uniform (eye + basis +
 * half-FOV extents + resolution), written each frame from the orbit
 * camera's eye position. The wireframe is a draw-order overlay (the pass
 * has no depth attachment), projected with the same camera parameters so
 * it aligns with the ray-marched view.
 */

import { viewProjection } from "./camera.js";
import type { CameraConfig } from "./point-render-pipeline.js";
import { BBOX_WIREFRAME_SHADER, DEBUG_RENDER_SHADER } from "./shaders/index.js";
import type { SceneBBox } from "./spatial-grid-pipeline.js";

/** Camera uniform buffer size in bytes (5 × 16-byte aligned members = 80). */
const CAMERA_BUFFER_BYTES = 80;

/**
 * Bounding-box wireframe uniform buffer size in bytes: `mat4x4f` (64) plus
 * two `vec3f` + pad slots (2 × 16) = 96.
 */
const BBOX_BUFFER_BYTES = 96;

/**
 * Manages the debug render pipeline and the `CameraRays` uniform.
 *
 * The pipelines are created once at construction time. Each frame, the
 * caller invokes {@link render} from inside an active render pass; the
 * camera uniform is uploaded beforehand via {@link writeCamera} and the
 * wireframe uniform via {@link writeBBox}.
 */
export class DebugRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bboxPipeline: GPURenderPipeline;
    private readonly cameraBuffer: GPUBuffer;
    private readonly bboxBuffer: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;
    private readonly bboxBindGroup: GPUBindGroup;
    private readonly config: CameraConfig;
    private readonly bbox: SceneBBox;

    /**
     * Creates the shader modules, render pipelines, camera and wireframe
     * uniform buffers, and bind groups.
     *
     * @param device - The GPU device.
     * @param config - Static camera configuration (FOV, target, up).
     * @param format - The canvas texture format (e.g. `bgra8unorm`).
     * @param bbox - The scene bounding box drawn as a wireframe overlay.
     */
    public constructor(
        device: GPUDevice,
        config: CameraConfig,
        format: GPUTextureFormat,
        bbox: SceneBBox,
    ) {
        this.device = device;
        this.config = config;
        this.bbox = bbox;

        const shaderModule = device.createShaderModule({
            label: "stipple-debug-render-shader",
            code: DEBUG_RENDER_SHADER,
        });

        const bboxModule = device.createShaderModule({
            label: "stipple-debug-bbox-wireframe-shader",
            code: BBOX_WIREFRAME_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "stipple-debug-render-pipeline",
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

        this.bboxPipeline = device.createRenderPipeline({
            label: "stipple-debug-bbox-wireframe-pipeline",
            layout: "auto",
            vertex: {
                module: bboxModule,
                entryPoint: "bbox_vs",
            },
            fragment: {
                module: bboxModule,
                entryPoint: "bbox_fs",
                targets: [{ format }],
            },
            primitive: {
                topology: "line-list",
            },
        });

        this.cameraBuffer = device.createBuffer({
            label: "stipple-debug-camera",
            size: CAMERA_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bboxBuffer = device.createBuffer({
            label: "stipple-debug-bbox-wireframe",
            size: BBOX_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const layout = this.pipeline.getBindGroupLayout(0);
        this.bindGroup = device.createBindGroup({
            label: "stipple-debug-render-bind",
            layout,
            entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }],
        });

        this.bboxBindGroup = device.createBindGroup({
            label: "stipple-debug-bbox-wireframe-bind",
            layout: this.bboxPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.bboxBuffer } }],
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
     *   `up: vec3f + time: f32`,
     *   `resolution: vec2u + _pad2: vec2u`.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param width - Canvas backing-store width.
     * @param height - Canvas backing-store height.
     * @param time - Current animation time in seconds.
     */
    private writeCamera(
        eye: readonly [number, number, number],
        width: number,
        height: number,
        time: number,
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
        f32[15] = time;
        u32[16] = width;
        u32[17] = height;
        u32[18] = 0;
        u32[19] = 0;

        this.device.queue.writeBuffer(this.cameraBuffer, 0, buffer);
    }

    /**
     * Writes the bounding-box wireframe uniform for the current frame.
     *
     * Packs the view-projection matrix and the scene bounding box into
     * three 16-byte aligned struct members:
     *   `view_proj: mat4x4f`,
     *   `bbox_min: vec3f + _pad0: f32`,
     *   `bbox_max: vec3f + _pad1: f32`.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param width - Canvas backing-store width.
     * @param height - Canvas backing-store height.
     */
    private writeBBox(
        eye: readonly [number, number, number],
        width: number,
        height: number,
    ): void {
        const viewProj = viewProjection(
            eye,
            this.config.target,
            this.config.up,
            this.config.fov,
            width / height,
        );

        const buffer = new ArrayBuffer(BBOX_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        f32.set(viewProj, 0);
        f32[16] = this.bbox.min[0];
        f32[17] = this.bbox.min[1];
        f32[18] = this.bbox.min[2];
        f32[20] = this.bbox.max[0];
        f32[21] = this.bbox.max[1];
        f32[22] = this.bbox.max[2];

        this.device.queue.writeBuffer(this.bboxBuffer, 0, buffer);
    }

    /**
     * Draws the SDF debug view (grayscale Lambert) as a full-screen pass,
     * followed by the scene bounding-box wireframe as a line-list overlay.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`. Uploads
     * the camera uniform before drawing.
     *
     * @param pass - The active render pass encoder.
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   ray direction per fragment.
     * @param time - Current animation time in seconds.
     */
    public render(
        pass: GPURenderPassEncoder,
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
        time: number,
    ): void {
        this.writeCamera(eye, canvas.width, canvas.height, time);
        this.writeBBox(eye, canvas.width, canvas.height);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
        pass.setPipeline(this.bboxPipeline);
        pass.setBindGroup(0, this.bboxBindGroup);
        pass.draw(24);
    }
}
