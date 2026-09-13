/**
 * Render pipeline for instanced billboard rendering of n-body positions.
 *
 * Owns the WGSL shader module, render pipeline, camera uniform buffer,
 * and two bind groups for ping-pong buffer selection. Each frame the
 * caller provides the view-projection matrix and the buffer direction
 * that contains the most recently written body data.
 */

import type { BodyBuffers } from "./buffers.js";
import { RENDER_SHADER } from "./shaders.js";

/** Size of the camera uniform buffer in bytes (mat4x4f + camera_pos + inv_aspect + padding). */
const CAMERA_BUFFER_BYTES = 96;

/**
 * Manages the render pipeline and its bind groups.
 *
 * The pipeline is created once at construction time. Each frame, the
 * caller invokes {@link render} inside a render pass to draw all bodies
 * as instanced quads.
 */
export class RenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly cameraBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;

    /**
     * Creates the shader module, render pipeline, camera uniform buffer,
     * and both ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param bodies - The ping-pong body buffer pair.
     * @param format - The canvas texture format.
     */
    public constructor(device: GPUDevice, bodies: BodyBuffers, format: GPUTextureFormat) {
        this.device = device;

        const shaderModule = device.createShaderModule({
            label: "nbody-render-shader",
            code: RENDER_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "nbody-render-pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs",
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

        this.cameraBuffer = device.createBuffer({
            label: "nbody-camera",
            size: CAMERA_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const layout = this.pipeline.getBindGroupLayout(0);

        this.bindGroupA = device.createBindGroup({
            label: "nbody-render-bind-A",
            layout,
            entries: [
                { binding: 0, resource: { buffer: bodies.bufferA } },
                { binding: 1, resource: { buffer: this.cameraBuffer } },
            ],
        });

        this.bindGroupB = device.createBindGroup({
            label: "nbody-render-bind-B",
            layout,
            entries: [
                { binding: 0, resource: { buffer: bodies.bufferB } },
                { binding: 1, resource: { buffer: this.cameraBuffer } },
            ],
        });
    }

    /**
     * Writes the view-projection matrix, camera position, and inverse
     * aspect ratio into the camera uniform buffer.
     *
     * Three separate `writeBuffer` calls target the three regions of the
     * uniform buffer:
     *   offset 0:  view_proj   (64 bytes)
     *   offset 64: camera_pos  (16 bytes — vec4f, .w unused)
     *   offset 80: inv_aspect  (4 bytes)
     *
     * @param viewProj - Column-major 4×4 matrix (16 floats).
     * @param cameraPos - Camera world-space position [x, y, z].
     * @param invAspect - Inverse of the canvas aspect ratio (height / width).
     */
    private writeCamera(
        viewProj: Float32Array,
        cameraPos: readonly [number, number, number],
        invAspect: number,
    ): void {
        this.device.queue.writeBuffer(
            this.cameraBuffer,
            0,
            viewProj.buffer,
            viewProj.byteOffset,
            viewProj.byteLength,
        );
        const posData = new Float32Array([cameraPos[0], cameraPos[1], cameraPos[2], 0]);
        this.device.queue.writeBuffer(
            this.cameraBuffer,
            64,
            posData.buffer,
            posData.byteOffset,
            posData.byteLength,
        );
        const aspectData = new Float32Array([invAspect]);
        this.device.queue.writeBuffer(
            this.cameraBuffer,
            80,
            aspectData.buffer,
            aspectData.byteOffset,
            aspectData.byteLength,
        );
    }

    /**
     * Draws all bodies as instanced quads into the given render pass.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`. The
     * caller is responsible for beginning/ending the pass and submitting
     * the command buffer.
     *
     * @param pass - The active render pass encoder.
     * @param viewProj - The view-projection matrix for this frame.
     * @param cameraPos - Camera world-space position [x, y, z].
     * @param invAspect - Inverse of the canvas aspect ratio (height / width).
     * @param readFromBufferA - If `true`, reads from `bufferA`; else
     *   `bufferB`. Should match the buffer the compute pass just wrote.
     * @param instanceCount - Number of bodies (instances) to draw.
     */
    public render(
        pass: GPURenderPassEncoder,
        viewProj: Float32Array,
        cameraPos: readonly [number, number, number],
        invAspect: number,
        readFromBufferA: boolean,
        instanceCount: number,
    ): void {
        this.writeCamera(viewProj, cameraPos, invAspect);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromBufferA ? this.bindGroupA : this.bindGroupB);
        pass.draw(6, instanceCount);
    }
}
