/**
 * Keyboard-triggered still-image export of the canvas.
 *
 * A key press requests an export; the frame loop fulfills requests via
 * {@link FrameCapture.save} right after presenting, so the serialized
 * image is the frame just rendered and its filename can be tagged with
 * the matching animation time.
 *
 * The controls are passive — they only record the request. The caller is
 * responsible for polling {@link save} each frame.
 */

/** Physical key code that requests an export. */
const EXPORT_KEY = "KeyL";

/**
 * Maintains the still-image export request state and processes keyboard
 * events on the window to set it.
 *
 * Modifier combinations and auto-repeats are ignored. Exports are
 * serialized as PNG at the canvas backing-store resolution. Call
 * {@link dispose} to remove the listener.
 */
export class FrameCapture {
    private readonly canvas: HTMLCanvasElement;
    private count: number = 0;
    private pending: boolean = false;

    private readonly onKeyDown: (e: KeyboardEvent) => void;

    /**
     * Creates the controls and attaches the key listener to the window.
     *
     * @param canvas - The canvas to serialize.
     */
    public constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;

        this.onKeyDown = (e: KeyboardEvent): void => {
            if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) {
                return;
            }
            if (e.code === EXPORT_KEY) {
                this.pending = true;
            }
        };

        window.addEventListener("keydown", this.onKeyDown);
    }

    /**
     * Serializes the canvas's last presented frame to a PNG and downloads
     * it, if an export was requested since the last save.
     *
     * The filename encodes an incrementing counter and the given animation
     * time in microseconds, e.g. `frame-0001-t3456789.png`.
     *
     * @param time - Animation time in seconds to encode in the filename.
     */
    public save(time: number): void {
        if (!this.pending) {
            return;
        }
        this.pending = false;
        this.count += 1;

        const name = `frame-${String(this.count).padStart(4, "0")}-t${Math.round(time * 1e3)}.png`;
        this.canvas.toBlob((blob) => {
            if (blob === null) {
                console.error("Failed to serialize the canvas to a PNG.");
                return;
            }
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = url;
            anchor.download = name;
            anchor.click();
            URL.revokeObjectURL(url);
        }, "image/png");
    }

    /**
     * Removes the event listener from the window.
     *
     * Call this when the controls are no longer needed to avoid
     * dangling references.
     */
    public dispose(): void {
        window.removeEventListener("keydown", this.onKeyDown);
    }
}
