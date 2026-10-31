import { FrameCapture } from "./frame-capture.js";
import { OrbitControls } from "./orbit-controls.js";
import type { CameraConfig } from "./pipelines/camera.js";
import { DebugRenderPipeline } from "./pipelines/debug-render-pipeline.js";
import { TimeControls } from "./time-controls.js";
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
const UP: readonly [number, number, number] = [0, 1, 0];

/** Initial orbit target point — the orbit center and look-at target. */
const TARGET: readonly [number, number, number] = [0, 0, 0];

/** Whether to draw the SDF debug view behind the stipple points. */
const DRAW_DEBUG = true;

/**
 * Whether the canvas has a fixed size, applied once at bootstrap to both
 * the rendering attachment and the HTML element. When disabled, the
 * canvas tracks the viewport and its backing store is re-synced to the
 * CSS size (with `devicePixelRatio`) every frame.
 */
const FIXED_CANVAS_SIZE = false;

/**
 * Fixed canvas height in pixels — the long side of the portrait canvas.
 * Only used when {@link FIXED_CANVAS_SIZE} is enabled.
 */
const CANVAS_HEIGHT = 4000;

/**
 * Fixed canvas width in pixels, derived from {@link CANVAS_HEIGHT} for
 * DIN A4 portrait aspect ratio (210:297). Only used when
 * {@link FIXED_CANVAS_SIZE} is enabled.
 */
const CANVAS_WIDTH = Math.round(CANVAS_HEIGHT * (210 / 297));

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
 * @param gpu - The WebGPU context.
 * @param pointRender - The point render pipeline.
 * @param controls - Orbit camera controls.
 * @param timeControls - Animation-time playback controls.
 * @param frameCapture - Still-image export controls.
 */
function startFrameLoop(
    gpu: GpuContext,
    debugRender: DebugRenderPipeline,
    controls: OrbitControls,
    timeControls: TimeControls,
    frameCapture: FrameCapture,
): void {
    const { device } = gpu;
    let lastNow = performance.now() / 1000;
    let time = 0;

    const frame = (): void => {
        if (!FIXED_CANVAS_SIZE) {
            syncCanvasSize(gpu);
        }

        const now = performance.now() / 1000;
        const dt = Math.min(now - lastNow, MAX_DT);
        lastNow = now;

        controls.update(dt);
        const eye = controls.getEye();
        const target = controls.getTarget();

        const encoder = device.createCommandEncoder();

        if (timeControls.isPlaying) {
            time += dt;
        }

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
            debugRender.render(pass, eye, target, gpu.canvas, time);
        }

        pass.end();

        device.queue.submit([encoder.finish()]);

        frameCapture.save(time);

        requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);
}

async function bootstrap(): Promise<void> {
    const canvas = getCanvas();
    const gpu = await createGpuContext(
        canvas,
        FIXED_CANVAS_SIZE ? { width: CANVAS_WIDTH, height: CANVAS_HEIGHT } : undefined,
    );

    const cameraConfig: CameraConfig = {
        fov: FIELD_OF_VIEW,
        up: UP,
    };

    const debugRender = new DebugRenderPipeline(gpu.device, cameraConfig, gpu.format);
    const controls = new OrbitControls(canvas, {
        azimuth: 0.0,
        elevation: 0.5,
        radius: 8,
        target: TARGET,
    });
    const timeControls = new TimeControls();
    const frameCapture = new FrameCapture(canvas);

    startFrameLoop(gpu, debugRender, controls, timeControls, frameCapture);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void bootstrap(), {
        once: true,
    });
} else {
    void bootstrap();
}
