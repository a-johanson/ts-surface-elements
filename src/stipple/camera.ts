/**
 * View-projection matrix construction shared by the render pipelines.
 *
 * The point renderer and the debug render's bounding-box wireframe both
 * project world-space geometry onto the canvas and must stay mutually
 * consistent, so the look-at and perspective helpers live here. Matrices
 * are column-major, matching the WGSL `mat4x4f` uniform layout.
 */

/** Column-major 4×4 matrix stored as 16 floats. */
export type Mat4 = number[];

/** Near plane for the perspective projection. */
const NEAR = 0.1;

/** Far plane for the perspective projection. */
const FAR = 100;

/** Orthonormal camera frame. */
export interface CameraBasis {
    /** Unit view direction, from the eye toward the target. */
    forward: [number, number, number];
    /** Unit vector pointing to the right of the view. */
    right: [number, number, number];
    /** Unit vector pointing up in the view, orthogonal to the others. */
    up: [number, number, number];
}

/**
 * Builds the orthonormal camera frame for a look-at configuration.
 *
 * Degenerate inputs (eye coinciding with the target, or view direction
 * parallel to the world up) fall back to a forward of -Z and a right of +X
 * respectively.
 *
 * @param eye - Camera eye position.
 * @param target - Look-at target.
 * @param up - World up direction.
 * @returns The forward, right, and up unit vectors.
 */
export function cameraBasis(
    eye: readonly [number, number, number],
    target: readonly [number, number, number],
    up: readonly [number, number, number],
): CameraBasis {
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

    return {
        forward: [fx, fy, fz],
        right: [rx, ry, rz],
        up: [ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx],
    };
}

/**
 * Computes the half-extents of the image plane at unit distance.
 *
 * @param fov - Vertical field of view in radians.
 * @param aspect - Canvas width / height.
 * @returns Half-width and half-height of the image plane.
 */
export function halfExtents(fov: number, aspect: number): { halfW: number; halfH: number } {
    const halfH = Math.tan(fov / 2);
    return { halfW: halfH * aspect, halfH };
}

/**
 * Builds a column-major look-at view matrix.
 *
 * @param eye - Camera eye position.
 * @param target - Look-at target.
 * @param up - World up direction.
 * @returns Column-major view matrix (16 floats).
 */
function lookAt(
    eye: readonly [number, number, number],
    target: readonly [number, number, number],
    up: readonly [number, number, number],
): Mat4 {
    const basis = cameraBasis(eye, target, up);
    const [fx, fy, fz] = basis.forward;
    const [rx, ry, rz] = basis.right;
    const [ux, uy, uz] = basis.up;

    return [
        rx,
        ux,
        -fx,
        0,
        ry,
        uy,
        -fy,
        0,
        rz,
        uz,
        -fz,
        0,
        -(rx * eye[0] + ry * eye[1] + rz * eye[2]),
        -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
        fx * eye[0] + fy * eye[1] + fz * eye[2],
        1,
    ];
}

/**
 * Builds a column-major perspective projection matrix.
 *
 * @param fov - Vertical field of view in radians.
 * @param aspect - Width / height.
 * @returns Column-major projection matrix (16 floats).
 */
function perspective(fov: number, aspect: number): Mat4 {
    const { halfW, halfH } = halfExtents(fov, aspect);
    const f = 1 / halfH;
    return [
        1 / halfW,
        0,
        0,
        0,
        0,
        f,
        0,
        0,
        0,
        0,
        (FAR + NEAR) / (NEAR - FAR),
        -1,
        0,
        0,
        (2 * FAR * NEAR) / (NEAR - FAR),
        0,
    ];
}

/**
 * Multiplies two column-major 4×4 matrices (`a * b`).
 *
 * @param a - Left matrix.
 * @param b - Right matrix.
 * @returns Column-major product (16 floats).
 */
function multiply(a: Mat4, b: Mat4): Mat4 {
    const out: Mat4 = new Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    return out as Mat4;
}

/**
 * Builds the combined view-projection matrix for a pinhole camera.
 *
 * @param eye - Camera eye position.
 * @param target - Look-at target.
 * @param up - World up direction.
 * @param fov - Vertical field of view in radians.
 * @param aspect - Canvas width / height.
 * @returns Column-major view-projection matrix (16 floats).
 */
export function viewProjection(
    eye: readonly [number, number, number],
    target: readonly [number, number, number],
    up: readonly [number, number, number],
    fov: number,
    aspect: number,
): Mat4 {
    return multiply(perspective(fov, aspect), lookAt(eye, target, up));
}
