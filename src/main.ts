import { OrbitControls } from "./orbit-controls.js";
import { type CameraConfig, DebugRenderPipeline } from "./stipple/debug-render-pipeline.js";
import { PointBuffers } from "./stipple/point-buffers.js";
import { PointRenderPipeline } from "./stipple/point-render-pipeline.js";
import { RelaxPipeline } from "./stipple/relax-pipeline.js";
import { ReprojectPipeline } from "./stipple/reproject-pipeline.js";
import { SeedPipeline } from "./stipple/seed-pipeline.js";
import { DEFAULT_LIGHT_DIR, ShadingPipeline } from "./stipple/shading-pipeline.js";
import { type SceneBBox, SpatialGridPipeline } from "./stipple/spatial-grid-pipeline.js";
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

/**
 * Fixed CPU-known bounding box for the animated SDF scene.
 *
 * Shared between the seed pipeline (rejection sampling) and the spatial
 * grid (cell origin and dimensions). Chosen to safely contain the surface
 * across all animation phases. If the scene geometry changes, update this
 * constant — a too-small bbox causes only performance degradation (the
 * cell-index clamp collapses out-of-range points into boundary cells).
 */
const SCENE_BBOX: SceneBBox = {
    min: [-3, -2, -2],
    max: [3, 2, 2],
};

/** Relaxation interaction radius (also the spatial grid cell size and seed band). */
const RELAX_RADIUS = 0.3;

/** Acceptance band for seed rejection sampling — tied to the relax radius. */
const SEED_BAND = RELAX_RADIUS;

/** Whether to draw the SDF debug view behind the stipple points. */
const DRAW_DEBUG = false;

/**
 * Maximum per-frame wall-clock delta, in seconds.
 *
 * Caps the animation time step so the per-frame surface motion stays
 * bounded — guaranteeing the reproject pass's Newton iteration converges
 * even after a tab-switch or frame hitch. Animation slows during heavy
 * frame drops rather than jumping.
 */
const MAX_DT = 1 / 30;

/**
 * Target wall-clock time per relax substep, in seconds.
 *
 * The relax kernel integrates with explicit Euler (`x* = x + dt·F_tan`),
 * which goes unstable when `dt` exceeds the kernel's stable step size.
 * This constant is the empirically stable single-step dt (the former
 * fixed constant); the frame loop subdivides each frame's `dt` into
 * `ceil(dt / TARGET_SUBSTEP_DT)` sub-passes so the per-step integration
 * size stays bounded regardless of frame rate.
 */
const TARGET_SUBSTEP_DT = 0.01;

/**
 * Maximum number of relax substeps per frame.
 *
 * Caps substep count so a severe hitch doesn't explode relax cost
 * (relax is the O(n²) pass). At `MAX_DT = 1/30` and
 * `TARGET_SUBSTEP_DT = 0.01`, the uncapped count is 4, so this bound
 * only bites on pathological dt values beyond the cap.
 */
const MAX_SUBSTEPS = 4;

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
 * 2. Computes the capped wall-clock delta and accumulates animation time.
 * 3. Dispatches the reproject compute pass to re-project points onto the
 *    current animated surface and refresh the shared normals buffer.
 * 4. Dispatches the relax compute pass (ping-pong) one or more times to
 *    redistribute points via 3D repulsion with surface re-projection.
 *    The frame's `dt` is subdivided into `ceil(dt / TARGET_SUBSTEP_DT)`
 *    sub-passes (capped at `MAX_SUBSTEPS`), each flipping the ping-pong
 *    direction, so the per-step Euler integration size stays stable
 *    regardless of frame rate.
 * 5. Dispatches the shading compute pass to refresh the normals buffer
 *    and compute per-point corner occlusion clearances (sphere-traced
 *    toward the eye) and luminance (Lambert with shadow) from the relaxed
 *    positions.
 * 6. Begins a render pass that draws the SDF debug view (grayscale
 *    Lambert) and then the stipple points.
 * 7. Submits the command buffer.
 *
 * Seeding happens once at bootstrap (before this loop starts) at t=0
 * since points live in world space and are independent of the view.
 *
 * @param gpu - The WebGPU context.
 * @param reproject - The reproject compute pipeline.
 * @param relax - The relax compute pipeline.
 * @param shading - The shading compute pipeline.
 * @param points - The ping-pong point buffer pair.
 * @param debugRender - The debug render pipeline (SDF visualization).
 * @param pointRender - The point render pipeline.
 * @param controls - Orbit camera controls.
 */
function startFrameLoop(
    gpu: GpuContext,
    reproject: ReprojectPipeline,
    spatialGrid: SpatialGridPipeline,
    relax: RelaxPipeline,
    shading: ShadingPipeline,
    points: PointBuffers,
    debugRender: DebugRenderPipeline,
    pointRender: PointRenderPipeline,
    controls: OrbitControls,
): void {
    const { device } = gpu;
    let readFromA = true;
    let lastNow = performance.now() / 1000;
    let time = 0;

    const frame = (): void => {
        syncCanvasSize(gpu);

        const now = performance.now() / 1000;
        const dt = Math.min(now - lastNow, MAX_DT);
        lastNow = now;
        time += dt;

        const eye = controls.getEye();

        const encoder = device.createCommandEncoder();

        // --- Compute pass: reproject (in-place, before relax) ---
        reproject.dispatch(encoder, time, readFromA);

        // --- Compute pass: spatial grid build (before relax, reused across
        // all substeps — see decision 1). Built from the same buffer relax
        // is about to read, so cell assignments match the iterated positions.
        spatialGrid.dispatch(encoder, readFromA);

        // --- Compute pass: relax (ping-pong, substepped) ---
        // Subdivide dt so the per-step Euler size stays within the kernel's
        // stable range. Each substep flips the ping-pong direction; after
        // the loop, readFromA points at whichever buffer relax last wrote.
        const substeps = Math.min(
            Math.max(Math.ceil(dt / TARGET_SUBSTEP_DT), 1),
            MAX_SUBSTEPS,
        );
        const substepDt = dt / substeps;
        for (let s = 0; s < substeps; s++) {
            relax.dispatch(encoder, substepDt, time, readFromA);
            readFromA = !readFromA;
        }

        // Shading reads the buffer that relax most recently wrote to.
        const shadingReadsA = readFromA;
        shading.dispatch(encoder, eye, DEFAULT_LIGHT_DIR, time, shadingReadsA);

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

        if (DRAW_DEBUG) {
            debugRender.render(pass, eye, gpu.canvas, time);
        }
        pointRender.render(pass, eye, gpu.canvas, renderReadsA, points.count);

        pass.end();

        device.queue.submit([encoder.finish()]);

        // The relax substep loop advanced readFromA to the buffer relax
        // most recently wrote; next frame's reproject refreshes it in-place.
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

    const seed = new SeedPipeline(gpu.device, SCENE_BBOX, SEED_BAND);
    const points = new PointBuffers(gpu.device, POINT_COUNT);
    const reproject = new ReprojectPipeline(gpu.device, points);
    const spatialGrid = new SpatialGridPipeline(gpu.device, points, SCENE_BBOX, RELAX_RADIUS);
    const relax = new RelaxPipeline(gpu.device, points, spatialGrid, RELAX_RADIUS);
    const shading = new ShadingPipeline(gpu.device, points);
    const debugRender = new DebugRenderPipeline(gpu.device, cameraConfig, gpu.format);
    const pointRender = new PointRenderPipeline(gpu.device, points, cameraConfig, gpu.format);
    const controls = new OrbitControls(canvas, {
        azimuth: 1.9,
        elevation: 0.5,
        radius: 8,
    });

    // --- Seed buffer A once (points are world-space; no re-seed on resize) ---
    const seedEncoder = gpu.device.createCommandEncoder();
    seed.dispatch(seedEncoder, points.bufferA, points.normalsBuffer, points.count);
    gpu.device.queue.submit([seedEncoder.finish()]);

    startFrameLoop(
        gpu,
        reproject,
        spatialGrid,
        relax,
        shading,
        points,
        debugRender,
        pointRender,
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
