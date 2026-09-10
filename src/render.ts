import type { CanvasDimensions } from "./canvas-dimensions";
import { createSfc32 } from "./sfc32";

const rng = createSfc32(0x1234_5673_9abc_def0n);
const DOT_COUNT = 400;
const DOT_RADIUS = 3;

/**
 * Returns a random color string.
 *
 * @returns A CSS rgb() color.
 */
function createRandomColor(): string {
    const red = Math.floor(rng() * 256);
    const green = Math.floor(rng() * 256);
    const blue = Math.floor(rng() * 256);

    return `rgb(${red}, ${green}, ${blue})`;
}

/**
 * Renders a single frame with a white background and fixed-seed random dots.
 *
 * @param context - The initialized 2D rendering context.
 * @param dimensions - The output canvas dimensions.
 */
export function renderFrame(
    context: CanvasRenderingContext2D,
    dimensions: CanvasDimensions,
): void {
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, dimensions.width, dimensions.height);

    for (let index = 0; index < DOT_COUNT; index += 1) {
        const x = rng() * dimensions.width;
        const y = rng() * dimensions.height;

        context.beginPath();
        context.arc(x, y, DOT_RADIUS, 0, Math.PI * 2);
        context.fillStyle = createRandomColor();
        context.fill();
    }
}
