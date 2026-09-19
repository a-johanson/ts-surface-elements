/**
 * Point render pipeline — draws stipple points as screen-space billboard
 * quads overlaid on the canvas.
 *
 * Each point is drawn as a two-triangle quad centered at the point's UV
 * position (mapped to NDC). The quad radius is specified in pixels and
 * converted to NDC using the canvas resolution, producing circular discs
 * regardless of aspect ratio. The fragment shader paints a soft red disc
 * with additive blending.
 *
 * Two bind groups support the ping-pong buffer pair so the render can
 * read from whichever buffer the relax pass most recently wrote to.
 */

import type { PointBuffers } from "./point-buffers.js";
import { POINT_SHADER } from "./shaders.js";

/** Size of the point uniform buffer in bytes (vec2u + f32 + f32 = 16). */
const UNIFORM_BUFFER_BYTES = 16;

/** Default point radius in pixels. */
const DEFAULT_POINT_RADIUS_PX = 2.5;

/**
 * Manages the point render pipeline, uniform buffer, and ping-pong bind
 * groups.
 */
export class PointRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly uniformBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;

    /**
     * Creates the shader module, render pipeline, uniform buffer, and
     * both ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair.
     * @param format - The canvas texture format.
     */
    public constructor(device: GPUDevice, points: PointBuffers, format: GPUTextureFormat) {
        this.device = device;

        const shaderModule = device.createShaderModule({
            label: "stipple-point-shader",
            code: POINT_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "stipple-point-pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "point_vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "point_fs",
                targets: [
                    {
                        format,
                        blend: {
                            color: {
                                srcFactor: "src-alpha",
                                dstFactor: "one",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "one",
                                dstFactor: "one",
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
            label: "stipple-point-uniform",
            size: UNIFORM_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const layout = this.pipeline.getBindGroupLayout(0);

        this.bindGroupA = device.createBindGroup({
            label: "stipple-point-bind-A",
            layout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferA } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });

        this.bindGroupB = device.createBindGroup({
            label: "stipple-point-bind-B",
            layout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferB } },
                { binding: 1, resource: { buffer: this.uniformBuffer } },
            ],
        });
    }

    /**
     * Writes the canvas resolution and point radius into the uniform
     * buffer.
     *
     * @param canvas - The canvas whose backing-store size determines the
     *   resolution.
     * @param pointRadiusPx - Point disc radius in pixels.
     */
    private writeUniform(canvas: HTMLCanvasElement, pointRadiusPx: number): void {
        const buffer = new ArrayBuffer(UNIFORM_BUFFER_BYTES);
        const u32 = new Uint32Array(buffer);
        const f32 = new Float32Array(buffer);
        u32[0] = canvas.width;
        u32[1] = canvas.height;
        f32[2] = pointRadiusPx;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer);
    }

    /**
     * Draws all points as instanced billboard quads into the given
     * render pass.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`, after the
     * debug blit (or any other content) has been drawn.
     *
     * @param pass - The active render pass encoder.
     * @param canvas - The canvas whose size determines the NDC-to-pixel
     *   conversion.
     * @param readFromBufferA - If `true`, reads from `bufferA`; else
     *   `bufferB`. Should match the buffer the seed/relax pass most
     *   recently wrote.
     * @param instanceCount - Number of points (instances) to draw.
     * @param pointRadiusPx - Point disc radius in pixels (defaults to
     *   {@link DEFAULT_POINT_RADIUS_PX}).
     */
    public render(
        pass: GPURenderPassEncoder,
        canvas: HTMLCanvasElement,
        readFromBufferA: boolean,
        instanceCount: number,
        pointRadiusPx: number = DEFAULT_POINT_RADIUS_PX,
    ): void {
        this.writeUniform(canvas, pointRadiusPx);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromBufferA ? this.bindGroupA : this.bindGroupB);
        pass.draw(6, instanceCount);
    }
}
