import { OrbitControls } from "./orbit-controls.js";
import { DebugRenderPipeline } from "./stipple/debug-render-pipeline.js";
import { type CameraConfig, DensityPipeline } from "./stipple/density-pipeline.js";
import { createGpuContext, type GpuContext, syncCanvasSize } from "./webgpu.js";

/** Background clear color — black. */
const CLEAR_COLOR: GPUColorDict = {
    r: 0,
    g: 0,
    b: 0,
    a: 1,
};

/** Camera parameters. */
const FIELD_OF_VIEW = (45 * Math.PI) / 180;
const TARGET: readonly [number, number, number] = [0, 0, 0];
const UP: readonly [number, number, number] = [0, 1, 0];

/**
 * Returns the `#outputCanvas` element from the DOM.
 *
 * @returns The canvas element.
 * @throws {Error} If the element is missing or not a canvas.
 */
function getCanvas(): HTMLCanvasElement {
    const element = document.getElementById("outputCanvas");
    if (!(element instanceof HTMLCanvasElement)) {
        throw new Error("Expected #outputCanvas to be a canvas element.");
    }
    return element;
}

/**
 * Starts the per-frame loop.
 *
 * Each frame:
 * 1. Syncs canvas size.
 * 2. Dispatches the density compute pass (ray-march → r32float texture).
 * 3. Begins a render pass on the canvas and blits the density texture
 *    as grayscale.
 * 4. Submits the command buffer.
 *
 * @param gpu - The WebGPU context.
 * @param density - The density compute pipeline.
 * @param blit - The debug blit render pipeline.
 * @param controls - Orbit camera controls.
 */
function startFrameLoop(
    gpu: GpuContext,
    density: DensityPipeline,
    blit: DebugRenderPipeline,
    controls: OrbitControls,
): void {
    const { device } = gpu;

    const frame = (): void => {
        syncCanvasSize(gpu);

        const eye = controls.getEye();

        const encoder = device.createCommandEncoder();

        // --- Compute pass: ray-march SDF → density texture ---
        density.dispatch(encoder, eye, gpu.canvas);

        // --- Render pass: blit density to canvas ---
        const texture = gpu.context.getCurrentTexture();
        const view = texture.createView();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view,
                    clearValue: CLEAR_COLOR,
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });

        blit.render(pass, density.getTexture());
        pass.end();

        device.queue.submit([encoder.finish()]);

        requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);
}

async function bootstrap(): Promise<void> {
    const canvas = getCanvas();
    const gpu = await createGpuContext(canvas);

    const cameraConfig: CameraConfig = {
        fov: FIELD_OF_VIEW,
        target: TARGET,
        up: UP,
    };

    const density = new DensityPipeline(gpu.device, cameraConfig);
    const blit = new DebugRenderPipeline(gpu.device, gpu.format);
    const controls = new OrbitControls(canvas, {
        azimuth: 0,
        elevation: 0.15,
        radius: 12,
    });

    startFrameLoop(gpu, density, blit, controls);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void bootstrap(), {
        once: true,
    });
} else {
    void bootstrap();
}
