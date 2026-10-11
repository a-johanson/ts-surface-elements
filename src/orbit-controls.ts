/**
 * Mouse- and keyboard-driven orbit camera controls.
 *
 * Left-button drag orbits the camera around an adjustable target point
 * using spherical coordinates (azimuth + elevation). Mouse wheel adjusts
 * the orbit radius (zoom). Holding Q or E moves the target point down or
 * up along the world y axis. Right-click is left untouched so the
 * browser context menu remains available.
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

/** Physical key that lowers the target along the y axis while held. */
const LOWER_TARGET_KEY = "KeyQ";

/** Physical key that raises the target along the y axis while held. */
const RAISE_TARGET_KEY = "KeyE";

/** Target movement speed along the y axis, in world units per second. */
const TARGET_MOVE_SPEED = 1.5;

/** Minimum target y position. */
const TARGET_MIN_Y = -3;

/** Maximum target y position. */
const TARGET_MAX_Y = 3;

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
    private lowerTargetHeld: boolean = false;
    private raiseTargetHeld: boolean = false;
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
            if (e.code === LOWER_TARGET_KEY) {
                this.lowerTargetHeld = true;
            } else if (e.code === RAISE_TARGET_KEY) {
                this.raiseTargetHeld = true;
            }
        };

        this.onKeyUp = (e: KeyboardEvent): void => {
            if (e.code === LOWER_TARGET_KEY) {
                this.lowerTargetHeld = false;
            } else if (e.code === RAISE_TARGET_KEY) {
                this.raiseTargetHeld = false;
            }
        };

        this.onBlur = (): void => {
            this.lowerTargetHeld = false;
            this.raiseTargetHeld = false;
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
     * While the Q or E key is held, the target moves down or up the
     * world y axis at a fixed speed, clamped to a bounded range. Call
     * once per frame, before reading {@link getEye} or {@link getTarget}.
     *
     * @param dt - Elapsed wall-clock time in seconds.
     */
    public update(dt: number): void {
        let dy = 0;
        if (this.lowerTargetHeld) {
            dy -= TARGET_MOVE_SPEED * dt;
        }
        if (this.raiseTargetHeld) {
            dy += TARGET_MOVE_SPEED * dt;
        }
        if (dy !== 0) {
            const y = this.target[1] + dy;
            this.target[1] = Math.max(TARGET_MIN_Y, Math.min(TARGET_MAX_Y, y));
        }
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
