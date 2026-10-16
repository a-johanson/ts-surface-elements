/**
 * Link render pipeline — draws the link curves connecting stipple
 * points to their best-aligned neighbors as chains of straight 1px
 * line segments following the surface.
 *
 * Each instance draws one `line-list` sub-segment between two adjacent
 * curve points of the subdivided, surface-projected curve written by
 * the link compute pass. The vertex shader culls links with invalid
 * curve slices and — as the interim coarse visibility cull from
 * Step 2b in the hatch-lines spec — links whose stipple endpoints are
 * both fully occluded (per-segment occlusion replaces this in
 * Step 4).
 *
 * The curve, links, and shading buffers are not ping-ponged, so a
 * single bind group covers them.
 */

import { viewProjection } from "./camera.js";
import type { CameraConfig } from "./point-render-pipeline.js";
import { LINK_N_SEGMENTS, LINK_RENDER_SHADER } from "./shaders/index.js";

/**
 * Size of the link render uniform buffer in bytes: `mat4x4f` (64) +
 * `vec3f` eye (12) + `f32` pad (4). The eye is written from Step 1 but
 * only consumed by the vertex shader once the renderer switches to
 * camera-facing quads.
 */
const UNIFORM_BUFFER_BYTES = 80;

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
     *   compute pass.
     * @param links - The internal post-dedup links buffer (the target
     *   stipple point per link, for the coarse visibility cull).
     * @param shading - The shared shading buffer (per-point center-ray
     *   clearance for the coarse visibility cull).
     * @param config - Static camera configuration (FOV, up).
     * @param format - The canvas texture format.
     */
    public constructor(
        device: GPUDevice,
        linkCurves: GPUBuffer,
        links: GPUBuffer,
        shading: GPUBuffer,
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
                topology: "line-list",
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
                { binding: 2, resource: { buffer: links } },
                { binding: 3, resource: { buffer: shading } },
            ],
        });
    }

    /**
     * Writes the view-projection matrix and eye position into the
     * uniform buffer.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param target - Look-at target point [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   projection aspect ratio.
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
        this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer);
    }

    /**
     * Draws all link curve segments as instanced lines into the given
     * render pass.
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
     *   `LINK_N_SEGMENTS` sub-segment instances.
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
        pass.draw(2, pointCount * LINK_N_SEGMENTS);
    }
}
