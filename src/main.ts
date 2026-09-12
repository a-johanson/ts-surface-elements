import {
    BODY_FLOATS,
    BodyBuffers,
    createSphereSeed,
    DEFAULT_BODY_COUNT,
} from "./nbody/buffers.js";
import { ComputePipeline, DEFAULT_PARAMS, type SimParams } from "./nbody/compute-pipeline.js";
import { createGpuContext, type GpuContext, syncCanvasSize } from "./webgpu.js";

/** Background clear color — near-black with a slight blue tint. */
const CLEAR_COLOR: GPUColorDict = {
    r: 0.02,
    g: 0.02,
    b: 0.04,
    a: 1,
};

/** Number of bodies to log during the one-shot readback verification. */
const VERIFY_LOG_COUNT = 3;

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
 * Logs the positions of the first {@link VERIFY_LOG_COUNT} bodies from
 * the given packed data array.
 *
 * @param label - Prefix label for the log output.
 * @param data - Packed body data (`count * {@link BODY_FLOATS}` floats).
 */
function logBodies(label: string, data: Float32Array): void {
    console.log(`${label} (first ${VERIFY_LOG_COUNT} bodies):`);
    for (let i = 0; i < VERIFY_LOG_COUNT; i += 1) {
        const offset = i * BODY_FLOATS;
        console.log(
            `  body ${i}: pos=(${data[offset].toFixed(4)}, ${data[offset + 1].toFixed(4)}, ${data[offset + 2].toFixed(4)}) mass=${data[offset + 3].toFixed(4)}`,
        );
    }
}

/**
 * Starts the per-frame loop.
 *
 * Each frame records a compute dispatch (gravity integration step)
 * followed by a render pass that clears the canvas. The compute
 * dispatch alternates between the two ping-pong buffer directions.
 *
 * After the first compute step, a one-shot readback logs the updated
 * body positions to the console for verification.
 *
 * @param gpu - The WebGPU context to render with.
 * @param bodies - The ping-pong body storage buffers.
 * @param compute - The gravity compute pipeline.
 * @param params - Simulation parameters.
 * @param seed - The initial body data (used for pre-compute logging).
 */
function startFrameLoop(
    gpu: GpuContext,
    bodies: BodyBuffers,
    compute: ComputePipeline,
    params: SimParams,
    seed: Float32Array,
): void {
    const { device } = gpu;
    let readFromA = true;
    let verifyFirstFrame = true;

    logBodies("Before compute", seed);

    const render = (): void => {
        syncCanvasSize(gpu);

        const encoder = device.createCommandEncoder();

        // --- Compute pass: advance the simulation one step ---
        compute.dispatch(encoder, params, readFromA);

        // --- Render pass: clear (bodies not yet rendered — step 4) ---
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
        pass.end();
        device.queue.submit([encoder.finish()]);

        // --- One-shot readback verification after first compute step ---
        if (verifyFirstFrame) {
            verifyFirstFrame = false;
            // The compute wrote to the *opposite* buffer.
            const writtenBufferIsA = !readFromA;
            void bodies.readback(device, writtenBufferIsA).then((data) => {
                logBodies("After 1 compute step", data);
            });
        }

        // Swap ping-pong direction for the next frame.
        readFromA = !readFromA;
        requestAnimationFrame(render);
    };

    requestAnimationFrame(render);
}

async function bootstrap(): Promise<void> {
    const canvas = getCanvas();
    const gpu = await createGpuContext(canvas);
    const seed = createSphereSeed(DEFAULT_BODY_COUNT);
    const bodies = new BodyBuffers(gpu.device, seed);
    const compute = new ComputePipeline(gpu.device, bodies, DEFAULT_PARAMS);
    startFrameLoop(gpu, bodies, compute, DEFAULT_PARAMS, seed);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void bootstrap(), {
        once: true,
    });
} else {
    void bootstrap();
}
