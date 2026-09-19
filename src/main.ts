import { OrbitControls } from "./orbit-controls.js";
import { DebugRenderPipeline } from "./stipple/debug-render-pipeline.js";
import { type CameraConfig, DensityPipeline } from "./stipple/density-pipeline.js";
import { PointBuffers } from "./stipple/point-buffers.js";
import { PointRenderPipeline } from "./stipple/point-render-pipeline.js";
import {
    DEFAULT_RELAX_PARAMS,
    type RelaxParams,
    RelaxPipeline,
} from "./stipple/relax-pipeline.js";
import { SeedPipeline } from "./stipple/seed-pipeline.js";
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

/** Number of stipple points. */
const POINT_COUNT = 4096;

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
 * 3. If the canvas was resized (or this is the first frame), dispatches
 *    the seed compute pass to regenerate the initial point distribution
 *    into buffer A, and resets the ping-pong direction to read from A.
 * 4. Dispatches the relax compute pass (sample-densities + relax,
 *    ping-pong) to redistribute points according to the density field.
 * 5. Begins a render pass that blits the density texture as grayscale,
 *    then draws all stipple points as red billboard quads on top.
 * 6. Submits the command buffer.
 *
 * @param gpu - The WebGPU context.
 * @param density - The density compute pipeline.
 * @param seed - The seed compute pipeline.
 * @param relax - The relax compute pipeline.
 * @param points - The ping-pong point buffer pair.
 * @param blit - The debug blit render pipeline.
 * @param pointRender - The point render pipeline.
 * @param params - Relaxation parameters.
 * @param controls - Orbit camera controls.
 */
function startFrameLoop(
    gpu: GpuContext,
    density: DensityPipeline,
    seed: SeedPipeline,
    relax: RelaxPipeline,
    points: PointBuffers,
    blit: DebugRenderPipeline,
    pointRender: PointRenderPipeline,
    params: RelaxParams,
    controls: OrbitControls,
): void {
    const { device } = gpu;
    let lastCanvasWidth = 0;
    let lastCanvasHeight = 0;
    let readFromA = true;

    const frame = (): void => {
        syncCanvasSize(gpu);

        const resized =
            gpu.canvas.width !== lastCanvasWidth || gpu.canvas.height !== lastCanvasHeight;

        const eye = controls.getEye();

        const encoder = device.createCommandEncoder();

        // --- Compute pass: ray-march SDF → density texture ---
        density.dispatch(encoder, eye, gpu.canvas);

        // --- Compute pass (one-shot on resize): seed buffer A ---
        if (resized) {
            seed.dispatch(encoder, density.getTexture(), points.bufferA, points.count);
            lastCanvasWidth = gpu.canvas.width;
            lastCanvasHeight = gpu.canvas.height;
            readFromA = true;
        }

        // --- Compute passes: sample densities + relax (ping-pong) ---
        relax.dispatch(encoder, density.getTexture(), points, params, readFromA);

        // Render reads the buffer that relax just wrote to.
        const renderReadsA = !readFromA;

        // --- Render pass: blit density + draw points ---
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
        pointRender.render(pass, gpu.canvas, renderReadsA, points.count);

        pass.end();

        device.queue.submit([encoder.finish()]);

        // Swap ping-pong direction for the next frame.
        readFromA = !readFromA;
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
    const seed = new SeedPipeline(gpu.device);
    const points = new PointBuffers(gpu.device, POINT_COUNT);
    const relax = new RelaxPipeline(gpu.device, points);
    const blit = new DebugRenderPipeline(gpu.device, gpu.format);
    const pointRender = new PointRenderPipeline(gpu.device, points, gpu.format);
    const controls = new OrbitControls(canvas, {
        azimuth: 0,
        elevation: 0.15,
        radius: 12,
    });

    startFrameLoop(
        gpu,
        density,
        seed,
        relax,
        points,
        blit,
        pointRender,
        DEFAULT_RELAX_PARAMS,
        controls,
    );
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void bootstrap(), {
        once: true,
    });
} else {
    void bootstrap();
}
