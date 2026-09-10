import { CanvasDimensions } from "./canvas-dimensions";
import { renderFrame } from "./render";

const canvasDimensions = new CanvasDimensions(144, 12, 12);

/**
 * Initializes the canvas element and returns its 2D context.
 *
 * @param canvasElement - The canvas element to initialize.
 * @param dimensions - Physical and pixel canvas dimensions.
 * @returns The initialized 2D rendering context.
 * @throws {Error} Throws when the canvas element or 2D context is unavailable.
 */
function initializeCanvas(
    canvasElement: HTMLCanvasElement,
    dimensions: CanvasDimensions,
): CanvasRenderingContext2D {
    canvasElement.width = dimensions.width;
    canvasElement.height = dimensions.height;
    canvasElement.style.width = `${dimensions.width}px`;
    canvasElement.style.height = `${dimensions.height}px`;

    const context = canvasElement.getContext("2d");
    if (context === null) {
        throw new Error("Unable to initialize a 2D rendering context.");
    }

    return context;
}

/**
 * Runs the canvas initialization and renders a single frame.
 */
function bootstrap(): void {
    const canvasElement = document.getElementById("outputCanvas");
    if (!(canvasElement instanceof HTMLCanvasElement)) {
        throw new Error("Expected #outputCanvas to be a canvas element.");
    }

    const context = initializeCanvas(canvasElement, canvasDimensions);
    renderFrame(context, canvasDimensions);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
} else {
    bootstrap();
}
