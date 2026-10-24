/**
 * Mouse-, touch-, and keyboard-driven orbit camera controls.
 *
 * Left-button drag or a one-finger drag orbits the camera around an
 * adjustable target point using spherical coordinates (azimuth +
 * elevation). Mouse wheel and two-finger pinch adjust the orbit radius
 * (zoom), and a two-finger drag pans the target point in the camera's
 * screen plane. Holding W/S moves the target point along the view
 * heading projected onto the ground plane, A/D strafes it left and
 * right, and Q/E moves it along the world y axis. Right-click is left
 * untouched so the browser context menu remains available.
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
const ROTATE_SENSITIVITY = 0.0035;

/** Radius delta per pixel of wheel scroll. */
const ZOOM_SENSITIVITY = 0.005;

/** Radius delta per pixel of two-finger pinch distance change. */
const PINCH_ZOOM_SENSITIVITY = 0.02;

/**
 * World units of target movement per pixel of two-finger pan, scaled by
 * the orbit radius so pan speed matches the current zoom level.
 */
const PAN_SENSITIVITY = 0.001;

/** Minimum orbit radius — prevents clipping into the scene. */
const MIN_RADIUS = 3;

/** Maximum orbit radius — prevents zooming out too far. */
const MAX_RADIUS = 30;

/** Clamp elevation to just under ±π/2 to avoid gimbal lock. */
const MAX_ELEVATION = Math.PI / 2 - 0.01;

/**
 * Movement weights per held physical key, expressed in the view-relative
 * basis as [forward, right, up]: W/S move along the view heading
 * projected onto the ground plane, A/D strafe left and right, and Q/E
 * move along the world y axis. The basis is resolved against the current
 * azimuth each time {@link OrbitControls.update} runs.
 */
const TARGET_KEYS: Readonly<Record<string, readonly [number, number, number]>> = {
    KeyW: [1, 0, 0],
    KeyS: [-1, 0, 0],
    KeyA: [0, -1, 0],
    KeyD: [0, 1, 0],
    KeyQ: [0, 0, -1],
    KeyE: [0, 0, 1],
};

/** Target movement speed, in world units per second. */
const TARGET_MOVE_SPEED = 1.5;

/** Minimum target position per world axis. */
const TARGET_MIN: readonly [number, number, number] = [-3, -3, -3];

/** Maximum target position per world axis. */
const TARGET_MAX: readonly [number, number, number] = [3, 3, 3];

/**
 * Maintains spherical camera state and the adjustable target point, and
 * processes mouse, touch, and keyboard input to update them.
 *
 * Listens to `pointerdown`, `pointermove`, `pointerup`, `pointercancel`,
 * and `wheel` events on the canvas, and to `keydown`, `keyup`, and
 * `blur` events on the window. Call {@link dispose} to remove all
 * listeners.
 */
export class OrbitControls {
    private readonly canvas: HTMLCanvasElement;
    private azimuth: number;
    private elevation: number;
    private radius: number;
    private target: [number, number, number];
    private readonly heldTargetKeys: Set<string> = new Set();
    /** Active pointers by ID, in the order they went down. */
    private readonly activePointers: Map<number, { x: number; y: number }> = new Map();
    private prevPinchDist: number = 0;
    private prevPinchMidX: number = 0;
    private prevPinchMidY: number = 0;

    private readonly onPointerDown: (e: PointerEvent) => void;
    private readonly onPointerMove: (e: PointerEvent) => void;
    private readonly onPointerEnd: (e: PointerEvent) => void;
    private readonly onWheel: (e: WheelEvent) => void;
    private readonly onKeyDown: (e: KeyboardEvent) => void;
    private readonly onKeyUp: (e: KeyboardEvent) => void;
    private readonly onBlur: () => void;

    /**
     * Creates the controls and attaches event listeners to the canvas
     * and the window.
     *
     * @param canvas - The canvas element to capture pointer input on.
     * @param state - Initial spherical state and target point.
     */
    public constructor(canvas: HTMLCanvasElement, state: OrbitState) {
        this.canvas = canvas;
        this.azimuth = state.azimuth;
        this.elevation = state.elevation;
        this.radius = state.radius;
        this.target = [...state.target];

        this.onPointerDown = (e: PointerEvent): void => {
            if (e.pointerType === "mouse" && e.button !== 0) {
                return;
            }
            this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (this.activePointers.size === 2) {
                this.beginPinch();
            }
            this.canvas.setPointerCapture(e.pointerId);
        };

        this.onPointerMove = (e: PointerEvent): void => {
            const point = this.activePointers.get(e.pointerId);
            if (!point) {
                return;
            }
            if (this.activePointers.size === 1) {
                const dx = e.clientX - point.x;
                const dy = e.clientY - point.y;
                point.x = e.clientX;
                point.y = e.clientY;
                this.rotate(dx, dy);
            } else {
                point.x = e.clientX;
                point.y = e.clientY;
                this.pinchAndPan();
            }
        };

        this.onPointerEnd = (e: PointerEvent): void => {
            if (e.pointerType === "mouse" && e.button !== 0) {
                return;
            }
            if (!this.activePointers.delete(e.pointerId)) {
                return;
            }
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
        canvas.addEventListener("pointerup", this.onPointerEnd);
        canvas.addEventListener("pointercancel", this.onPointerEnd);
        canvas.addEventListener("wheel", this.onWheel, { passive: false });
        window.addEventListener("keydown", this.onKeyDown);
        window.addEventListener("keyup", this.onKeyUp);
        window.addEventListener("blur", this.onBlur);
    }

    /**
     * Advances held-key target movement over the elapsed frame time.
     *
     * While W/S or A/D is held, the target moves along (or perpendicular
     * to) the view heading on the ground plane; while Q/E is held, it
     * moves along the world y axis. The combined movement direction is
     * normalized to a fixed speed and clamped to a per-axis bounded
     * range; no movement is applied when no keys are held or the held
     * keys cancel out. Call once per frame, before reading
     * {@link getEye} or {@link getTarget}.
     *
     * @param dt - Elapsed wall-clock time in seconds.
     */
    public update(dt: number): void {
        let forward = 0;
        let right = 0;
        let up = 0;
        for (const code of this.heldTargetKeys) {
            const weight = TARGET_KEYS[code];
            forward += weight[0];
            right += weight[1];
            up += weight[2];
        }

        // Normalize for constant speed in every direction. The length is
        // zero when nothing is held or the held keys cancel out (e.g. W
        // and S together), in which case there is nothing to move.
        const inputLength = Math.hypot(forward, right, up);
        if (inputLength === 0) {
            return;
        }
        const step = (TARGET_MOVE_SPEED * dt) / inputLength;

        // Ground-plane view basis from the current azimuth alone:
        // independent of elevation, so the heading never degenerates
        // when looking straight down. Forward points from the eye toward
        // the target, projected onto the xz-plane.
        const sinAzimuth = Math.sin(this.azimuth);
        const cosAzimuth = Math.cos(this.azimuth);
        const forwardX = -sinAzimuth;
        const forwardZ = -cosAzimuth;
        const rightX = cosAzimuth;
        const rightZ = -sinAzimuth;

        const moveX = (forward * forwardX + right * rightX) * step;
        const moveY = up * step;
        const moveZ = (forward * forwardZ + right * rightZ) * step;

        this.target[0] += moveX;
        this.target[1] += moveY;
        this.target[2] += moveZ;
        this.clampTarget();
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
        this.canvas.removeEventListener("pointerup", this.onPointerEnd);
        this.canvas.removeEventListener("pointercancel", this.onPointerEnd);
        this.canvas.removeEventListener("wheel", this.onWheel);
        window.removeEventListener("keydown", this.onKeyDown);
        window.removeEventListener("keyup", this.onKeyUp);
        window.removeEventListener("blur", this.onBlur);
    }

    /** Applies an orbit rotation from a single-pointer drag delta. */
    private rotate(dx: number, dy: number): void {
        this.azimuth -= dx * ROTATE_SENSITIVITY;
        this.elevation += dy * ROTATE_SENSITIVITY;
        this.elevation = Math.max(-MAX_ELEVATION, Math.min(MAX_ELEVATION, this.elevation));
    }

    /** Records the pinch geometry when the second pointer goes down. */
    private beginPinch(): void {
        const [first, second] = this.activePointers.values();
        this.prevPinchDist = Math.hypot(second.x - first.x, second.y - first.y);
        this.prevPinchMidX = (first.x + second.x) * 0.5;
        this.prevPinchMidY = (first.y + second.y) * 0.5;
    }

    /** Applies pinch zoom and two-finger pan from the current pointer geometry. */
    private pinchAndPan(): void {
        const [first, second] = this.activePointers.values();
        const dist = Math.hypot(second.x - first.x, second.y - first.y);
        const midX = (first.x + second.x) * 0.5;
        const midY = (first.y + second.y) * 0.5;

        this.radius = Math.max(
            MIN_RADIUS,
            Math.min(
                MAX_RADIUS,
                this.radius + (this.prevPinchDist - dist) * PINCH_ZOOM_SENSITIVITY,
            ),
        );
        this.panTarget(midX - this.prevPinchMidX, midY - this.prevPinchMidY);

        this.prevPinchDist = dist;
        this.prevPinchMidX = midX;
        this.prevPinchMidY = midY;
    }

    /**
     * Moves the target point so the scene follows a two-finger pan by
     * (`dx`, `dy`) pixels, expressed in the camera's screen-space basis
     * and scaled by the orbit radius so pan speed matches the current
     * zoom level.
     */
    private panTarget(dx: number, dy: number): void {
        const scale = PAN_SENSITIVITY * this.radius;
        const sinAzimuth = Math.sin(this.azimuth);
        const cosAzimuth = Math.cos(this.azimuth);
        const sinElevation = Math.sin(this.elevation);
        const cosElevation = Math.cos(this.elevation);

        // Screen-space basis: the ground-plane right vector and the
        // screen-up vector perpendicular to the view direction, which
        // accounts for elevation and stays well-defined everywhere
        // elevation is clamped away from exactly ±π/2.
        const rightX = cosAzimuth;
        const rightZ = -sinAzimuth;
        const upX = -sinAzimuth * sinElevation;
        const upY = cosElevation;
        const upZ = -cosAzimuth * sinElevation;

        this.target[0] += (-dx * rightX + dy * upX) * scale;
        this.target[1] += dy * upY * scale;
        this.target[2] += (-dx * rightZ + dy * upZ) * scale;
        this.clampTarget();
    }

    /** Clamps the target point to its per-axis bounded range. */
    private clampTarget(): void {
        this.target[0] = Math.max(TARGET_MIN[0], Math.min(TARGET_MAX[0], this.target[0]));
        this.target[1] = Math.max(TARGET_MIN[1], Math.min(TARGET_MAX[1], this.target[1]));
        this.target[2] = Math.max(TARGET_MIN[2], Math.min(TARGET_MAX[2], this.target[2]));
    }
}
