/**
 * Link render pipeline — draws the link segments connecting stipple
 * points to their best-aligned neighbors as straight 1px lines.
 *
 * Each instance draws one `line-list` segment: endpoint 0 is the point
 * itself, endpoint 1 is its linked neighbor (both read from the
 * post-relax point buffer). Links with no neighbor and links whose
 * endpoints are both fully occluded are pushed offscreen in the vertex
 * shader (see the link render shader).
 *
 * Two bind groups support the ping-pong point buffer pair so the render
 * can read from whichever buffer the relax pass most recently wrote to.
 * The links and shading buffers are non-ping-ponged.
 */

import { viewProjection } from "./camera.js";
import type { PointBuffers } from "./point-buffers.js";
import type { CameraConfig } from "./point-render-pipeline.js";
import { LINK_RENDER_SHADER } from "./shaders/index.js";

/**
 * Size of the link render uniform buffer in bytes: `mat4x4f` (64) +
 * `vec3f` eye (12) + `f32` pad (4). The eye is written from Step 1 but
 * only consumed by the vertex shader once the renderer switches to
 * camera-facing quads.
 */
const UNIFORM_BUFFER_BYTES = 80;

/**
 * Manages the link render pipeline, uniform buffer, and ping-pong bind
 * groups.
 */
export class LinkRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly uniformBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;
    private readonly config: CameraConfig;

    /**
     * Creates the shader module, render pipeline, uniform buffer, and
     * both ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair.
     * @param links - The shared links buffer written by the link compute
     *   pass.
     * @param config - Static camera configuration (FOV, target, up).
     * @param format - The canvas texture format.
     */
    public constructor(
        device: GPUDevice,
        points: PointBuffers,
        links: GPUBuffer,
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

        this.bindGroupA = device.createBindGroup({
            label: "stipple-link-render-bind-A",
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: links } },
                { binding: 3, resource: { buffer: points.shadingBuffer } },
            ],
        });

        this.bindGroupB = device.createBindGroup({
            label: "stipple-link-render-bind-B",
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: links } },
                { binding: 3, resource: { buffer: points.shadingBuffer } },
            ],
        });
    }

    /**
     * Writes the view-projection matrix and eye position into the
     * uniform buffer.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   projection aspect ratio.
     */
    private writeUniform(
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
    ): void {
        const aspect = canvas.width / canvas.height;
        const viewProj = viewProjection(
            eye,
            this.config.target,
            this.config.up,
            this.config.fov,
            aspect,
        );

        const buffer = new ArrayBuffer(UNIFORM_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        f32.set(viewProj, 0);
        f32[16] = eye[0];
        f32[17] = eye[1];
        f32[18] = eye[2];
        this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer);
    }

    /**
     * Draws all link segments as instanced lines into the given render
     * pass.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`, before
     * the point render so points sit on top of the lines.
     *
     * @param pass - The active render pass encoder.
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose size determines the projection
     *   aspect.
     * @param readFromBufferA - If `true`, reads from `bufferA`; else
     *   `bufferB`. Should match the buffer the relax pass most recently
     *   wrote.
     * @param instanceCount - Number of points (instances) to draw.
     */
    public render(
        pass: GPURenderPassEncoder,
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
        readFromBufferA: boolean,
        instanceCount: number,
    ): void {
        this.writeUniform(eye, canvas);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromBufferA ? this.bindGroupA : this.bindGroupB);
        pass.draw(2, instanceCount);
    }
}
