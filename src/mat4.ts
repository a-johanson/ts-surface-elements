/**
 * Column-major 4x4 matrix utilities backed by `Float32Array(16)`.
 *
 * Element at (row, col) is stored at index `col * 4 + row`. All functions
 * return new arrays; none mutate their inputs.
 */

/**
 * Returns a new identity matrix.
 *
 * @returns Identity matrix.
 */
export function createMat4Identity(): Float32Array {
    const out = new Float32Array(16);
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    return out;
}

/**
 * Returns a perspective projection matrix.
 *
 * @param fovy - Vertical field of view in radians.
 * @param aspect - Width / height aspect ratio.
 * @param near - Near clip plane distance (positive).
 * @param far - Far clip plane distance (positive).
 * @returns Perspective projection matrix.
 */
export function createMat4Perspective(
    fovy: number,
    aspect: number,
    near: number,
    far: number,
): Float32Array {
    const f = 1 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);

    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[14] = 2 * far * near * nf;
    return out;
}

/**
 * Returns a view matrix placing the eye at the given position and looking at
 * the center point, with the given up vector.
 *
 * @param eyeX - Eye position X.
 * @param eyeY - Eye position Y.
 * @param eyeZ - Eye position Z.
 * @param centerX - Look-at target X.
 * @param centerY - Look-at target Y.
 * @param centerZ - Look-at target Z.
 * @param upX - Up vector X.
 * @param upY - Up vector Y.
 * @param upZ - Up vector Z.
 * @returns View matrix.
 */
export function createMat4LookAt(
    eyeX: number,
    eyeY: number,
    eyeZ: number,
    centerX: number,
    centerY: number,
    centerZ: number,
    upX: number,
    upY: number,
    upZ: number,
): Float32Array {
    let zx = eyeX - centerX;
    let zy = eyeY - centerY;
    let zz = eyeZ - centerZ;
    const zLen = Math.hypot(zx, zy, zz);
    if (zLen === 0) {
        zx = 0;
        zy = 0;
        zz = 1;
    } else {
        zx /= zLen;
        zy /= zLen;
        zz /= zLen;
    }

    let xx = upY * zz - upZ * zy;
    let xy = upZ * zx - upX * zz;
    let xz = upX * zy - upY * zx;
    const xLen = Math.hypot(xx, xy, xz);
    if (xLen === 0) {
        xx = 1;
        xy = 0;
        xz = 0;
    } else {
        xx /= xLen;
        xy /= xLen;
        xz /= xLen;
    }

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const out = new Float32Array(16);
    out[0] = xx;
    out[1] = yx;
    out[2] = zx;
    out[3] = 0;
    out[4] = xy;
    out[5] = yy;
    out[6] = zy;
    out[7] = 0;
    out[8] = xz;
    out[9] = yz;
    out[10] = zz;
    out[11] = 0;
    out[12] = -(xx * eyeX + xy * eyeY + xz * eyeZ);
    out[13] = -(yx * eyeX + yy * eyeY + yz * eyeZ);
    out[14] = -(zx * eyeX + zy * eyeY + zz * eyeZ);
    out[15] = 1;
    return out;
}

/**
 * Returns the matrix product `a * b` (both column-major).
 *
 * @param a - Left-hand matrix.
 * @param b - Right-hand matrix.
 * @returns Product matrix.
 */
export function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col += 1) {
        for (let row = 0; row < 4; row += 1) {
            let sum = 0;
            for (let k = 0; k < 4; k += 1) {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    return out;
}
