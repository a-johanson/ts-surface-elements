/**
 * Immutable physical and pixel dimensions for the output canvas.
 */
export class CanvasDimensions {
    private static readonly CENTIMETERS_PER_INCH = 2.54;

    public readonly dpi: number;
    public readonly widthCm: number;
    public readonly heightCm: number;
    public readonly width: number;
    public readonly height: number;

    /**
     * Creates a canvas dimensions value object.
     *
     * @param dpi - Target dots per inch.
     * @param widthCm - Canvas width in centimeters.
     * @param heightCm - Canvas height in centimeters.
     */
    public constructor(dpi: number, widthCm: number, heightCm: number) {
        this.dpi = dpi;
        this.widthCm = widthCm;
        this.heightCm = heightCm;
        this.width = CanvasDimensions.centimetersToPixels(widthCm, dpi);
        this.height = CanvasDimensions.centimetersToPixels(heightCm, dpi);
    }

    /**
     * Converts centimeters to pixels for a given DPI.
     *
     * @param centimeters - Physical size in centimeters.
     * @param dpi - Target dots per inch.
     * @returns The corresponding integer pixel size.
     */
    private static centimetersToPixels(centimeters: number, dpi: number): number {
        return Math.round((centimeters / CanvasDimensions.CENTIMETERS_PER_INCH) * dpi);
    }
}
