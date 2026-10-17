/**
 * Link render pipeline — draws the link curves connecting stipple
 * points to their best-aligned neighbors as chains of screen-space
 * stroke quads following the subdivided, surface-projected curve.
 *
 * Each instance draws one `triangle-list` quad (two triangles, six
 * vertices) between two adjacent curve points of the subdivided,
 * surface-projected curve written by the link compute pass. The quad
 * is extruded in screen space around the projected segment and shaded
 * with a 2D capsule distance field sized from the line's world-space
 * half-width, so the line reads with the screen width its 3D tube
 * would project (shrinking with depth) while consecutive sub-segments
 * weld into a continuous stroke at every viewing angle, with a soft
 * `fwidth`-based anti-aliased edge. Occlusion (Step 4 in the
 * the per-curve-point clearance the link compute's clearance stage
 * bakes into the curve buffer's `.w` fields — refreshed every frame,
 * including while time is paused, so culling and the geometric
 * retraction of crossing segments stay view-dependent.
 *
 * The curve buffer is not ping-ponged, so a single bind group covers
 * the uniform and the curve data.
 */

import { viewProjection } from "./camera.js";
import type { CameraConfig } from "./point-render-pipeline.js";
import { LINK_N_SEGMENTS, LINK_RENDER_SHADER } from "./shaders/index.js";

/**
 * Size of the link render uniform buffer in bytes: `mat4x4f` (64) +
 * `vec3f` eye (12, 16-aligned) + `vec2f` canvas_size (8) + `f32` pad
 * (4) = 96 (rounded to the struct's 16-byte alignment). The eye
 * provides the view direction for the world-units-per-pixel estimate
 * that converts the stroke's world-space half-width to pixels; the
 * canvas size drives that estimate and the screen-space projection of
 * the stroke geometry.
 */
const UNIFORM_BUFFER_BYTES = 96;

/**
 * Manages the link render pipeline, uniform buffer, and bind group.
 */
export class LinkRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly uniformBuffer: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;
    private readonly config: CameraConfig;

    /**
     * Creates the shader module, render pipeline, uniform buffer, and
     * bind group.
     *
     * @param device - The GPU device.
     * @param linkCurves - The shared curve buffer written by the link
     *   compute pass — `.xyz` positions plus per-curve-point occlusion
     *   clearance in `.w`, refreshed every frame by the clearance
     *   stage.
     * @param config - Static camera configuration (FOV, up).
     * @param format - The canvas texture format.
     */
    public constructor(
        device: GPUDevice,
        linkCurves: GPUBuffer,
        config: CameraConfig,
        format: GPUTextureFormat,
    ) {
        this.device = device;
        this.config = config;

        const shaderModule = device.createShaderModule({
            label: "stipple-link-render-shader",
            code: LINK_RENDER_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "stipple-link-render-pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "link_vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "link_fs",
                targets: [
                    {
                        format,
                        blend: {
                            color: {
                                srcFactor: "src-alpha",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "one",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        this.uniformBuffer = device.createBuffer({
            label: "stipple-link-render-uniform",
            size: UNIFORM_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const layout = this.pipeline.getBindGroupLayout(0);

        this.bindGroup = device.createBindGroup({
            label: "stipple-link-render-bind",
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: linkCurves } },
            ],
        });
    }

    /**
     * Writes the view-projection matrix, eye position, and canvas
     * backing-store size into the uniform buffer. The eye provides the
     * view direction for the world-units-per-pixel estimate that
     * converts the stroke's world-space half-width to pixels; the
     * canvas size drives that estimate and the screen-space projection
     * of the stroke geometry.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param target - Look-at target point [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   projection aspect ratio and the AA-padding scale.
     */
    private writeUniform(
        eye: readonly [number, number, number],
        target: readonly [number, number, number],
        canvas: HTMLCanvasElement,
    ): void {
        const aspect = canvas.width / canvas.height;
        const viewProj = viewProjection(eye, target, this.config.up, this.config.fov, aspect);

        const buffer = new ArrayBuffer(UNIFORM_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        f32.set(viewProj, 0);
        f32[16] = eye[0];
        f32[17] = eye[1];
        f32[18] = eye[2];
        f32[20] = canvas.width;
        f32[21] = canvas.height;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer);
    }

    /**
     * Draws all link curve segments as instanced screen-space stroke
     * quads into the given render pass.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`, before
     * the point render so points sit on top of the lines.
     *
     * @param pass - The active render pass encoder.
     * @param eye - Camera eye position [x, y, z].
     * @param target - Look-at target point [x, y, z].
     * @param canvas - The canvas whose size determines the projection
     *   aspect.
     * @param pointCount - Number of stipple points; each contributes
     *   `LINK_N_SEGMENTS` quad instances.
     */
    public render(
        pass: GPURenderPassEncoder,
        eye: readonly [number, number, number],
        target: readonly [number, number, number],
        canvas: HTMLCanvasElement,
        pointCount: number,
    ): void {
        this.writeUniform(eye, target, canvas);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6, pointCount * LINK_N_SEGMENTS);
    }
}
