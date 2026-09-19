/**
 * Debug render pipeline — blits the density texture to the canvas as
 * grayscale.
 *
 * Draws a full-screen triangle that loads the `r32float` density texture
 * via `textureLoad` (integer texel coords, no sampler). Background
 * (`-1.0`) maps to black; hit values (`[0, 1]`) map to `[0, 1]` gray.
 *
 * This pipeline is for development visualization. In the final version
 * it may be removed or replaced by a combined debug+stipple render pass.
 */

import { BLIT_SHADER } from "./shaders.js";

/**
 * Manages the blit render pipeline and its bind group.
 *
 * The bind group is recreated only when the density texture changes
 * (i.e. on canvas resize), not every frame.
 */
export class DebugRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private lastTexture: GPUTexture | null = null;
    private bindGroup: GPUBindGroup | null = null;

    /**
     * Creates the shader module and render pipeline targeting the
     * canvas's preferred texture format.
     *
     * @param device - The GPU device.
     * @param format - The canvas texture format (e.g. `bgra8unorm`).
     */
    public constructor(device: GPUDevice, format: GPUTextureFormat) {
        this.device = device;

        const shaderModule = device.createShaderModule({
            label: "stipple-blit-shader",
            code: BLIT_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "stipple-blit-pipeline",
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

        this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    }

    /**
     * Draws the density texture to the canvas as grayscale.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`. The
     * caller is responsible for beginning/ending the pass and submitting
     * the command buffer.
     *
     * @param pass - The active render pass encoder.
     * @param densityTexture - The `r32float` density texture to blit.
     */
    public render(pass: GPURenderPassEncoder, densityTexture: GPUTexture): void {
        if (densityTexture !== this.lastTexture) {
            this.lastTexture = densityTexture;
            this.bindGroup = this.device.createBindGroup({
                label: "stipple-blit-bind",
                layout: this.bindGroupLayout,
                entries: [{ binding: 0, resource: densityTexture.createView() }],
            });
        }

        if (this.bindGroup === null) {
            throw new Error("Blit bind group was not created.");
        }

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
    }
}
