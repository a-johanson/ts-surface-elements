import { createMat4LookAt, createMat4Perspective, multiplyMat4 } from "./mat4.js";
import { BodyBuffers, createSphereSeed } from "./nbody/buffers.js";
import { ComputePipeline, DEFAULT_PARAMS, type SimParams } from "./nbody/compute-pipeline.js";
import { RenderPipeline } from "./nbody/render-pipeline.js";
import { OrbitControls } from "./orbit-controls.js";
import { createGpuContext, type GpuContext, syncCanvasSize } from "./webgpu.js";

/** Background clear color — near-black with a slight blue tint. */
const CLEAR_COLOR: GPUColorDict = {
    r: 0.02,
    g: 0.02,
    b: 0.04,
    a: 1,
};

/** Camera parameters. */
const FIELD_OF_VIEW = (45 * Math.PI) / 180;
const NEAR_PLANE = 0.1;
const FAR_PLANE = 1000;
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
 * Computes the view-projection matrix for the current canvas aspect ratio
 * and camera eye position.
 *
 * @param canvas - The canvas to derive the aspect ratio from.
 * @param eye - Camera eye position [x, y, z].
 * @returns Column-major view-projection matrix (16 floats).
 */
function computeViewProj(
    canvas: HTMLCanvasElement,
    eye: readonly [number, number, number],
): Float32Array {
    const aspect = canvas.width / canvas.height;
    const projection = createMat4Perspective(FIELD_OF_VIEW, aspect, NEAR_PLANE, FAR_PLANE);
    const view = createMat4LookAt(
        eye[0],
        eye[1],
        eye[2],
        TARGET[0],
        TARGET[1],
        TARGET[2],
        UP[0],
        UP[1],
        UP[2],
    );
    return multiplyMat4(projection, view);
}

/**
 * Starts the per-frame loop.
 *
 * Each frame:
 * 1. Syncs canvas size.
 * 2. Records a compute dispatch (gravity integration, ping-pong swap).
 * 3. Records a render pass that clears the canvas and draws all bodies
 *    as instanced billboards.
 * 4. Submits the command buffer.
 *
 * @param gpu - The WebGPU context.
 * @param bodies - The ping-pong body storage buffers.
 * @param compute - The gravity compute pipeline.
 * @param render - The billboard render pipeline.
 * @param params - Simulation parameters.
 * @param controls - Orbit camera controls.
 */
function startFrameLoop(
    gpu: GpuContext,
    bodies: BodyBuffers,
    compute: ComputePipeline,
    render: RenderPipeline,
    params: SimParams,
    controls: OrbitControls,
): void {
    const { device } = gpu;
    let readFromA = true;

    const frame = (): void => {
        syncCanvasSize(gpu);

        const eye = controls.getEye();
        const viewProj = computeViewProj(gpu.canvas, eye);

        const encoder = device.createCommandEncoder();

        // --- Compute pass: advance the simulation one step ---
        compute.dispatch(encoder, params, readFromA);

        // --- Render pass: draw all bodies ---
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

        // Render reads the buffer that compute just wrote to.
        const renderReadsA = !readFromA;
        const invAspect = gpu.canvas.height / gpu.canvas.width;
        render.render(pass, viewProj, eye, invAspect, renderReadsA, bodies.count);
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
    const seed = createSphereSeed(DEFAULT_PARAMS.bodyCount);
    const bodies = new BodyBuffers(gpu.device, seed);
    const compute = new ComputePipeline(gpu.device, bodies, DEFAULT_PARAMS);
    const renderPl = new RenderPipeline(gpu.device, bodies, gpu.format);
    const controls = new OrbitControls(canvas, {
        azimuth: 0,
        elevation: 0,
        radius: 60,
    });
    startFrameLoop(gpu, bodies, compute, renderPl, DEFAULT_PARAMS, controls);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void bootstrap(), {
        once: true,
    });
} else {
    void bootstrap();
}
