/**
 * Camera basis and half-FOV ray-extent construction for the debug render.
 *
 * The debug render's ray-marching uniform derives per-fragment ray
 * directions from an eye position, an orthonormal forward/right/up basis,
 * and half-FOV extents; {@link computeCameraRays} builds those from a
 * look-at eye/target/up plus FOV and canvas aspect, so the basis
 * construction lives in one place. Ray direction per fragment is
 * `normalize(forward + ndc.x * halfWidth * right + ndc.y * halfHeight * up)`.
 */

export interface CameraConfig {
    /** Vertical field of view in radians. */
    readonly fov: number;
    /** World-space up direction. */
    readonly up: readonly [number, number, number];
}

/** Orthonormal camera basis and half-FOV ray extents. */
export interface CameraRays {
    /** Unit forward direction (from eye toward target). */
    readonly forward: readonly [number, number, number];
    /** Unit right direction (forward cross `up`). */
    readonly right: readonly [number, number, number];
    /** Unit up direction (right cross forward). */
    readonly up: readonly [number, number, number];
    /** Half-FOV extent scaling NDC x offsets (tan(fov / 2) * aspect). */
    readonly halfWidth: number;
    /** Half-FOV extent scaling NDC y offsets (tan(fov / 2)). */
    readonly halfHeight: number;
}

/**
 * Builds the camera basis and half-FOV ray extents for a look-at camera.
 *
 * Degenerate cases fall back to a usable default: eye equal to target
 * maps to forward `(0, 0, -1)`, and forward parallel to `up` maps to
 * right `(1, 0, 0)`.
 *
 * @param eye - Camera eye position.
 * @param target - Look-at target.
 * @param up - World up direction.
 * @param fov - Vertical field of view in radians.
 * @param aspect - Canvas width / height.
 * @returns Basis and extents; ray direction per fragment is
 *   `normalize(forward + ndc.x * halfWidth * right + ndc.y * halfHeight * up)`.
 */
export function computeCameraRays(
    eye: readonly [number, number, number],
    target: readonly [number, number, number],
    up: readonly [number, number, number],
    fov: number,
    aspect: number,
): CameraRays {
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    const fLen = Math.hypot(fx, fy, fz);
    if (fLen === 0) {
        fx = 0;
        fy = 0;
        fz = -1;
    } else {
        fx /= fLen;
        fy /= fLen;
        fz /= fLen;
    }

    let rx = fy * up[2] - fz * up[1];
    let ry = fz * up[0] - fx * up[2];
    let rz = fx * up[1] - fy * up[0];
    const rLen = Math.hypot(rx, ry, rz);
    if (rLen === 0) {
        rx = 1;
        ry = 0;
        rz = 0;
    } else {
        rx /= rLen;
        ry /= rLen;
        rz /= rLen;
    }

    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    const halfHeight = Math.tan(fov / 2);

    return {
        forward: [fx, fy, fz],
        right: [rx, ry, rz],
        up: [ux, uy, uz],
        halfWidth: halfHeight * aspect,
        halfHeight,
    };
}
