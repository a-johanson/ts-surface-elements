/**
 * Mouse-driven orbit camera controls.
 *
 * Left-button drag orbits the camera around a fixed target point using
 * spherical coordinates (azimuth + elevation). Mouse wheel adjusts the
 * orbit radius (zoom). Right-click is left untouched so the browser
 * context menu remains available.
 *
 * The controls are passive — they only maintain spherical state. The
 * caller is responsible for reading {@link getEye} each frame and
 * feeding the result into view matrix construction.
 */

/** Configuration for initial orbit state. */
export interface OrbitState {
    /** Horizontal angle in radians (0 = looking down -Z). */
    readonly azimuth: number;
    /** Vertical angle in radians (0 = equator, clamped near ±π/2). */
    readonly elevation: number;
    /** Distance from the target point. */
    readonly radius: number;
}

/** Radians of rotation per pixel of drag. */
const ROTATE_SENSITIVITY = 0.005;

/** Radius delta per pixel of wheel scroll. */
const ZOOM_SENSITIVITY = 0.05;

/** Minimum orbit radius — prevents clipping into the scene. */
const MIN_RADIUS = 10;

/** Maximum orbit radius — prevents zooming out too far. */
const MAX_RADIUS = 200;

/** Clamp elevation to just under ±π/2 to avoid gimbal lock. */
const MAX_ELEVATION = Math.PI / 2 - 0.01;

/**
 * Maintains spherical camera state and processes mouse input on a canvas
 * element to update it.
 *
 * Listens to `pointerdown`, `pointermove`, `pointerup`, and `wheel`
 * events on the canvas. Call {@link dispose} to remove all listeners.
 */
export class OrbitControls {
    private readonly canvas: HTMLCanvasElement;
    private azimuth: number;
    private elevation: number;
    private radius: number;
    private dragging: boolean = false;
    private lastX: number = 0;
    private lastY: number = 0;

    private readonly onPointerDown: (e: PointerEvent) => void;
    private readonly onPointerMove: (e: PointerEvent) => void;
    private readonly onPointerUp: (e: PointerEvent) => void;
    private readonly onWheel: (e: WheelEvent) => void;

    /**
     * Creates the controls and attaches event listeners to the canvas.
     *
     * @param canvas - The canvas element to capture mouse input on.
     * @param state - Initial spherical state.
     */
    public constructor(canvas: HTMLCanvasElement, state: OrbitState) {
        this.canvas = canvas;
        this.azimuth = state.azimuth;
        this.elevation = state.elevation;
        this.radius = state.radius;

        this.onPointerDown = (e: PointerEvent): void => {
            if (e.button !== 0) {
                return;
            }
            this.dragging = true;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
            this.canvas.setPointerCapture(e.pointerId);
        };

        this.onPointerMove = (e: PointerEvent): void => {
            if (!this.dragging) {
                return;
            }
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;
            this.lastX = e.clientX;
            this.lastY = e.clientY;

            this.azimuth -= dx * ROTATE_SENSITIVITY;
            this.elevation += dy * ROTATE_SENSITIVITY;
            this.elevation = Math.max(-MAX_ELEVATION, Math.min(MAX_ELEVATION, this.elevation));
        };

        this.onPointerUp = (e: PointerEvent): void => {
            if (e.button !== 0) {
                return;
            }
            this.dragging = false;
            this.canvas.releasePointerCapture(e.pointerId);
        };

        this.onWheel = (e: WheelEvent): void => {
            e.preventDefault();
            const delta = e.deltaY * ZOOM_SENSITIVITY;
            this.radius = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, this.radius + delta));
        };

        canvas.addEventListener("pointerdown", this.onPointerDown);
        canvas.addEventListener("pointermove", this.onPointerMove);
        canvas.addEventListener("pointerup", this.onPointerUp);
        canvas.addEventListener("wheel", this.onWheel, { passive: false });
    }

    /**
     * Computes the camera eye position from the current spherical state.
     *
     * @returns Eye position as `[x, y, z]`.
     */
    public getEye(): readonly [number, number, number] {
        const cosElev = Math.cos(this.elevation);
        const x = this.radius * cosElev * Math.sin(this.azimuth);
        const y = this.radius * Math.sin(this.elevation);
        const z = this.radius * cosElev * Math.cos(this.azimuth);
        return [x, y, z];
    }

    /**
     * Removes all event listeners from the canvas.
     *
     * Call this when the controls are no longer needed to avoid
     * dangling references.
     */
    public dispose(): void {
        this.canvas.removeEventListener("pointerdown", this.onPointerDown);
        this.canvas.removeEventListener("pointermove", this.onPointerMove);
        this.canvas.removeEventListener("pointerup", this.onPointerUp);
        this.canvas.removeEventListener("wheel", this.onWheel);
    }
}
