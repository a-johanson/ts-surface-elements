import { OrbitControls } from "./orbit-controls.js";
import { type CameraConfig, DebugRenderPipeline } from "./stipple/debug-render-pipeline.js";
import { PointBuffers } from "./stipple/point-buffers.js";
import { PointRenderPipeline } from "./stipple/point-render-pipeline.js";
import {
    DEFAULT_RELAX_PARAMS,
    type RelaxParams,
    RelaxPipeline,
} from "./stipple/relax-pipeline.js";
import { type SeedParams, SeedPipeline } from "./stipple/seed-pipeline.js";
import { DEFAULT_LIGHT_DIR, ShadingPipeline } from "./stipple/shading-pipeline.js";
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
const POINT_COUNT = 8 * 1024;

/** Bounding box for rejection sampling of the seed distribution. */
const SEED_PARAMS: SeedParams = {
    bboxMin: [-3, -2, -2],
    bboxMax: [3, 2, 2],
    band: 0.3,
};

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
 * 2. Dispatches the relax compute pass (ping-pong) to redistribute points
 *    via 3D repulsion with surface re-projection.
 * 3. Dispatches the shading compute pass to refresh the normals buffer
 *    and compute per-point visibility (occlusion by the SDF surface) and
 *    luminance (Lambert with shadow) from the relaxed positions.
 * 4. Begins a render pass that draws the SDF debug view (grayscale
 *    Lambert) and then the stipple points as red billboard quads on top
 *    (occluded points discarded in the vertex shader, color modulated by
 *    luminance).
 * 5. Submits the command buffer.
 *
 * Seeding happens once at bootstrap (before this loop starts) since
 * points live in world space and are independent of the view.
 *
 * @param gpu - The WebGPU context.
 * @param relax - The relax compute pipeline.
 * @param shading - The shading compute pipeline.
 * @param points - The ping-pong point buffer pair.
 * @param debugRender - The debug render pipeline (SDF visualization).
 * @param pointRender - The point render pipeline.
 * @param params - Relaxation parameters.
 * @param controls - Orbit camera controls.
 */
function startFrameLoop(
    gpu: GpuContext,
    relax: RelaxPipeline,
    shading: ShadingPipeline,
    points: PointBuffers,
    debugRender: DebugRenderPipeline,
    pointRender: PointRenderPipeline,
    params: RelaxParams,
    controls: OrbitControls,
): void {
    const { device } = gpu;
    let readFromA = true;

    const frame = (): void => {
        syncCanvasSize(gpu);

        const eye = controls.getEye();

        const encoder = device.createCommandEncoder();

        // --- Compute pass: relax (ping-pong) ---
        relax.dispatch(encoder, params, readFromA);

        // Shading reads the buffer that relax just wrote to.
        const shadingReadsA = !readFromA;
        shading.dispatch(encoder, eye, DEFAULT_LIGHT_DIR, shadingReadsA);

        // Render reads the same buffer shading just read.
        const renderReadsA = shadingReadsA;

        // --- Render pass: debug SDF view + draw points ---
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

        debugRender.render(pass, eye, gpu.canvas);
        pointRender.render(pass, eye, gpu.canvas, renderReadsA, points.count);

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

    const seed = new SeedPipeline(gpu.device);
    const points = new PointBuffers(gpu.device, POINT_COUNT);
    const relax = new RelaxPipeline(gpu.device, points);
    const shading = new ShadingPipeline(gpu.device, points);
    const debugRender = new DebugRenderPipeline(gpu.device, cameraConfig, gpu.format);
    const pointRender = new PointRenderPipeline(gpu.device, points, cameraConfig, gpu.format);
    const controls = new OrbitControls(canvas, {
        azimuth: 0,
        elevation: 0.15,
        radius: 12,
    });

    // --- Seed buffer A once (points are world-space; no re-seed on resize) ---
    const seedEncoder = gpu.device.createCommandEncoder();
    seed.dispatch(
        seedEncoder,
        points.bufferA,
        points.normalsBuffer,
        points.count,
        SEED_PARAMS,
    );
    gpu.device.queue.submit([seedEncoder.finish()]);

    startFrameLoop(
        gpu,
        relax,
        shading,
        points,
        debugRender,
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
