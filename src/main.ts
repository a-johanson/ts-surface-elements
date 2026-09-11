import { CanvasDimensions } from "./canvas-dimensions";
import { Renderer } from "./render";

const canvasDimensions = new CanvasDimensions(150, 15, 20);

/**
 * Initializes the canvas element and returns its WebGL2 context.
 *
 * @param canvasElement - The canvas element to initialize.
 * @param dimensions - Physical and pixel canvas dimensions.
 * @returns The initialized WebGL2 rendering context.
 * @throws {Error} Throws when the canvas element or WebGL2 context is unavailable.
 */
function initializeCanvas(
    canvasElement: HTMLCanvasElement,
    dimensions: CanvasDimensions,
): WebGL2RenderingContext {
    canvasElement.width = dimensions.width;
    canvasElement.height = dimensions.height;
    canvasElement.style.width = `${dimensions.width}px`;
    canvasElement.style.height = `${dimensions.height}px`;

    const context = canvasElement.getContext("webgl2");
    if (context === null) {
        throw new Error("Unable to initialize a WebGL2 rendering context.");
    }

    return context;
}

/**
 * Runs the canvas initialization and starts the animation loop.
 */
function bootstrap(): void {
    const canvasElement = document.getElementById("outputCanvas");
    if (!(canvasElement instanceof HTMLCanvasElement)) {
        throw new Error("Expected #outputCanvas to be a canvas element.");
    }

    const context = initializeCanvas(canvasElement, canvasDimensions);
    const renderer = new Renderer(context, canvasDimensions);

    let startTime: number | null = null;
    const frame = (now: number): void => {
        if (startTime === null) {
            startTime = now;
        }
        renderer.render(now - startTime);
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
} else {
    bootstrap();
}
