/**
 * WebGPU initialization and canvas context management.
 *
 * This module owns the acquisition of the GPU adapter, device, and canvas
 * context configuration. It is the single entry point for obtaining the
 * low-level WebGPU objects the rest of the application needs.
 */

/** Bundle of WebGPU objects needed to render to a canvas. */
export interface GpuContext {
    /** The logical GPU device used to create buffers, pipelines, and submit commands. */
    readonly device: GPUDevice;
    /** The canvas context used to obtain the current frame's render target texture. */
    readonly context: GPUCanvasContext;
    /** The texture format the canvas expects (e.g. `bgra8unorm` or `rgba8unorm`). */
    readonly format: GPUTextureFormat;
    /** The canvas element backing this context. */
    readonly canvas: HTMLCanvasElement;
}

/**
 * Acquires a GPU adapter and device, then configures the given canvas for
 * WebGPU rendering using the browser's preferred texture format.
 *
 * @param canvas - The canvas element to render into.
 * @returns A {@link GpuContext} ready for rendering.
 * @throws {Error} If WebGPU is unavailable or no adapter/device can be obtained.
 */
export async function createGpuContext(canvas: HTMLCanvasElement): Promise<GpuContext> {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported in this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) {
        throw new Error("No suitable GPU adapter found.");
    }

    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    if (context === null) {
        throw new Error("Failed to acquire a WebGPU canvas context.");
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format,
        alphaMode: "premultiplied",
    });

    return { device, context, format, canvas };
}

/**
 * Synchronizes the canvas backing-store resolution with its CSS size,
 * accounting for `devicePixelRatio`. Should be called once per frame
 * before acquiring the current texture via `context.getCurrentTexture()`.
 *
 * The WebGPU canvas context does not need to be reconfigured after a size
 * change — the texture returned by `getCurrentTexture()` will match the
 * canvas's current `width` and `height`.
 *
 * @param ctx - The {@link GpuContext} whose canvas may have been resized.
 */
export function syncCanvasSize(ctx: GpuContext): void {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(ctx.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(ctx.canvas.clientHeight * dpr));
    if (ctx.canvas.width !== width || ctx.canvas.height !== height) {
        ctx.canvas.width = width;
        ctx.canvas.height = height;
    }
}
