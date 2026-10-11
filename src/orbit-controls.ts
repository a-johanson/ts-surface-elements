/**
 * Mouse- and keyboard-driven orbit camera controls.
 *
 * Left-button drag orbits the camera around an adjustable target point
 * using spherical coordinates (azimuth + elevation). Mouse wheel adjusts
 * the orbit radius (zoom). Holding A/D, W/S, or Q/E moves the target
 * point along the world x, z, and y axes. Right-click is left untouched
 * so the browser context menu remains available.
 *
 * The controls are passive — they only maintain spherical and target
 * state. The caller is responsible for invoking {@link update} each
 * frame to apply held-key target movement, and for reading {@link getEye}
 * and {@link getTarget} to feed view-matrix construction.
 */

/** Configuration for initial orbit state. */
export interface OrbitState {
    /** Horizontal angle in radians (0 = looking down -Z). */
    readonly azimuth: number;
    /** Vertical angle in radians (0 = equator, clamped near ±π/2). */
    readonly elevation: number;
    /** Distance from the target point. */
    readonly radius: number;
    /** Orbit center and look-at target in world space. */
    readonly target: readonly [number, number, number];
}

/** Radians of rotation per pixel of drag. */
const ROTATE_SENSITIVITY = 0.0025;

/** Radius delta per pixel of wheel scroll. */
const ZOOM_SENSITIVITY = 0.005;

/** Minimum orbit radius — prevents clipping into the scene. */
const MIN_RADIUS = 3;

/** Maximum orbit radius — prevents zooming out too far. */
const MAX_RADIUS = 30;

/** Clamp elevation to just under ±π/2 to avoid gimbal lock. */
const MAX_ELEVATION = Math.PI / 2 - 0.01;

/**
 * World-space unit direction the target moves per held physical key:
 * A/D along x, W/S along z (W pushes the target away from the viewer at
 * the default azimuth), and Q/E along y.
 */
const TARGET_KEYS: Readonly<Record<string, readonly [number, number, number]>> = {
    KeyA: [-1, 0, 0],
    KeyD: [1, 0, 0],
    KeyW: [0, 0, -1],
    KeyS: [0, 0, 1],
    KeyQ: [0, -1, 0],
    KeyE: [0, 1, 0],
};

/** Target movement speed, in world units per second. */
const TARGET_MOVE_SPEED = 1.5;

/** Minimum target position per world axis. */
const TARGET_MIN: readonly [number, number, number] = [-3, -3, -3];

/** Maximum target position per world axis. */
const TARGET_MAX: readonly [number, number, number] = [3, 3, 3];

/**
 * Maintains spherical camera state and the adjustable target point, and
 * processes mouse and keyboard input to update them.
 *
 * Listens to `pointerdown`, `pointermove`, `pointerup`, and `wheel`
 * events on the canvas, and to `keydown`, `keyup`, and `blur` events on
 * the window. Call {@link dispose} to remove all listeners.
 */
export class OrbitControls {
    private readonly canvas: HTMLCanvasElement;
    private azimuth: number;
    private elevation: number;
    private radius: number;
    private target: [number, number, number];
    private readonly heldTargetKeys: Set<string> = new Set();
    private dragging: boolean = false;
    private lastX: number = 0;
    private lastY: number = 0;

    private readonly onPointerDown: (e: PointerEvent) => void;
    private readonly onPointerMove: (e: PointerEvent) => void;
    private readonly onPointerUp: (e: PointerEvent) => void;
    private readonly onWheel: (e: WheelEvent) => void;
    private readonly onKeyDown: (e: KeyboardEvent) => void;
    private readonly onKeyUp: (e: KeyboardEvent) => void;
    private readonly onBlur: () => void;

    /**
     * Creates the controls and attaches event listeners to the canvas
     * and the window.
     *
     * @param canvas - The canvas element to capture mouse input on.
     * @param state - Initial spherical state and target point.
     */
    public constructor(canvas: HTMLCanvasElement, state: OrbitState) {
        this.canvas = canvas;
        this.azimuth = state.azimuth;
        this.elevation = state.elevation;
        this.radius = state.radius;
        this.target = [...state.target];

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

        this.onKeyDown = (e: KeyboardEvent): void => {
            if (e.ctrlKey || e.metaKey || e.altKey) {
                return;
            }
            if (e.code in TARGET_KEYS) {
                this.heldTargetKeys.add(e.code);
            }
        };

        this.onKeyUp = (e: KeyboardEvent): void => {
            this.heldTargetKeys.delete(e.code);
        };

        this.onBlur = (): void => {
            this.heldTargetKeys.clear();
        };

        canvas.addEventListener("pointerdown", this.onPointerDown);
        canvas.addEventListener("pointermove", this.onPointerMove);
        canvas.addEventListener("pointerup", this.onPointerUp);
        canvas.addEventListener("wheel", this.onWheel, { passive: false });
        window.addEventListener("keydown", this.onKeyDown);
        window.addEventListener("keyup", this.onKeyUp);
        window.addEventListener("blur", this.onBlur);
    }

    /**
     * Advances held-key target movement over the elapsed frame time.
     *
     * While A/D, W/S, or Q/E is held, the target moves along the world
     * x, z, or y axis at a fixed speed, clamped to a per-axis bounded
     * range. Call once per frame, before reading {@link getEye}
     * or {@link getTarget}.
     *
     * @param dt - Elapsed wall-clock time in seconds.
     */
    public update(dt: number): void {
        let dx = 0;
        let dy = 0;
        let dz = 0;
        for (const code of this.heldTargetKeys) {
            const direction = TARGET_KEYS[code];
            dx += direction[0];
            dy += direction[1];
            dz += direction[2];
        }
        if (dx === 0 && dy === 0 && dz === 0) {
            return;
        }

        const step = TARGET_MOVE_SPEED * dt;
        this.target[0] = Math.max(
            TARGET_MIN[0],
            Math.min(TARGET_MAX[0], this.target[0] + dx * step),
        );
        this.target[1] = Math.max(
            TARGET_MIN[1],
            Math.min(TARGET_MAX[1], this.target[1] + dy * step),
        );
        this.target[2] = Math.max(
            TARGET_MIN[2],
            Math.min(TARGET_MAX[2], this.target[2] + dz * step),
        );
    }

    /**
     * Computes the camera eye position from the current spherical state
     * and target point.
     *
     * @returns Eye position as `[x, y, z]`.
     */
    public getEye(): readonly [number, number, number] {
        const cosElev = Math.cos(this.elevation);
        const x = this.target[0] + this.radius * cosElev * Math.sin(this.azimuth);
        const y = this.target[1] + this.radius * Math.sin(this.elevation);
        const z = this.target[2] + this.radius * cosElev * Math.cos(this.azimuth);
        return [x, y, z];
    }

    /**
     * Returns the current target point.
     *
     * @returns Target position as `[x, y, z]`.
     */
    public getTarget(): readonly [number, number, number] {
        return [this.target[0], this.target[1], this.target[2]];
    }

    /**
     * Removes all event listeners from the canvas and the window.
     *
     * Call this when the controls are no longer needed to avoid
     * dangling references.
     */
    public dispose(): void {
        this.canvas.removeEventListener("pointerdown", this.onPointerDown);
        this.canvas.removeEventListener("pointermove", this.onPointerMove);
        this.canvas.removeEventListener("pointerup", this.onPointerUp);
        this.canvas.removeEventListener("wheel", this.onWheel);
        window.removeEventListener("keydown", this.onKeyDown);
        window.removeEventListener("keyup", this.onKeyUp);
        window.removeEventListener("blur", this.onBlur);
    }
}
