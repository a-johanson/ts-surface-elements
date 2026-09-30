/**
 * 3D point data layout and ping-pong storage buffer management for the
 * surface-stippling pipeline.
 *
 * Each point is a single `vec4f` (16 bytes):
 *
 * ```
 * offset 0:  pos.xyz  (vec4f — world-space position on the SDF surface, w unused)
 * ```
 *
 * Two storage buffers (`bufferA`, `bufferB`) form a ping-pong pair. The
 * seed pipeline writes the initial distribution; the relax pipeline reads
 * from one buffer and writes to the other each frame.
 *
 * A separate non-ping-pong `normalsBuffer` holds the per-point surface
 * normal (`vec4f`, 16 bytes). It is a derived quantity — written once by
 * the seed pass at bootstrap and overwritten every frame by the shading
 * pass (which runs after relax, from the buffer relax just wrote) — and is
 * shared between relax (tangent-plane projection) and the shading pass.
 * It is not ping-ponged because it has no temporal state of its own; the
 * invariant "normals match whichever buffer relax reads" is preserved
 * because the shading pass writes normals from the buffer that the *next*
 * relax pass will read.
 *
 * A second non-ping-pong `shadingBuffer` holds the per-point shading
 * result (32 bytes: a `lum` `vec4f` with the luminance in `x`, and a
 * `clearance` `vec4f` with one occlusion clearance value per quad corner).
 * It is written every frame by the shading pass and read by the point
 * renderer, which pushes fully occluded points offscreen in the vertex
 * shader, clips each ring fragment against the interpolated clearance,
 * and modulates the color by luminance.
 *
 * No CPU readback is performed — seeding, relaxation, and shading are
 * entirely GPU-side.
 */

/** Number of `f32` values per point (one `vec4f`). */
export const POINT_FLOATS = 4;

/** Size of one point in bytes. */
export const POINT_BYTES = POINT_FLOATS * Float32Array.BYTES_PER_ELEMENT;

/** Number of `f32` values per shading sample (two `vec4f`s). */
export const SHADING_SAMPLE_FLOATS = 8;

/** Size of one shading sample in bytes. */
export const SHADING_SAMPLE_BYTES = SHADING_SAMPLE_FLOATS * Float32Array.BYTES_PER_ELEMENT;

/** Buffer usage flags for point and normals storage buffers. */
const POINT_BUFFER_USAGE: GPUBufferUsageFlags =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Owns the ping-pong pair of storage buffers that hold point positions,
 * plus the shared normals and shading buffers.
 *
 * Both point buffers are initialized to zero. The seed pipeline writes the
 * initial distribution (and matching normals) into one of them; the relax
 * pipeline ping-pongs between the two. The normals buffer is
 * zero-initialized, seeded once at bootstrap, and overwritten every frame
 * by the shading pass. The shading buffer is zero-initialized and
 * overwritten every frame by the shading pass.
 */
export class PointBuffers {
    /** Number of points stored in each buffer. */
    public readonly count: number;

    /** First buffer in the ping-pong pair. */
    public readonly bufferA: GPUBuffer;

    /** Second buffer in the ping-pong pair. */
    public readonly bufferB: GPUBuffer;

    /**
     * Shared per-point surface normals buffer (not ping-ponged).
     *
     * Written once at bootstrap by the seed pass and overwritten every
     * frame by the shading pass (which runs after relax, computing
     * normals from the buffer relax just wrote). Read by the
     * relax pass, by the shading pass itself, and by the
     * point renderer (to orient tangent-plane quads).
     */
    public readonly normalsBuffer: GPUBuffer;

    /**
     * Shared per-point shading buffer (not ping-ponged).
     *
     * Written every frame by the shading pass; read by the point renderer
     * to push fully occluded points offscreen (in the vertex shader),
     * clip the ring per fragment against the interpolated corner
     * clearance, and modulate the fragment color by luminance. Packed as
     * `ShadingSample` (32 bytes): `lum.x` = luminance, `clearance` = one
     * occlusion clearance per quad corner.
     */
    public readonly shadingBuffer: GPUBuffer;

    /**
     * Creates both ping-pong point buffers plus the shared normals and
     * shading buffers, all zero-filled via `mappedAtCreation`.
     *
     * @param device - The GPU device used to create the buffers.
     * @param count - Number of points.
     */
    public constructor(device: GPUDevice, count: number) {
        this.count = count;
        // Point, normals, and shading samples are vec4f-structured: points
        // are pos + unused w, normals are vec3f + pad, shading samples pack
        // luminance and four corner clearances into two vec4f. WGSL storage
        // arrays of vec4f / 32-byte structs require 16-byte alignment, which
        // is satisfied here.
        const size = count * POINT_BYTES;
        const shadingSize = count * SHADING_SAMPLE_BYTES;

        this.bufferA = this.createBuffer(device, size);
        this.bufferB = this.createBuffer(device, size);
        this.normalsBuffer = this.createBuffer(device, size);
        this.shadingBuffer = this.createBuffer(device, shadingSize);
    }

    /**
     * Creates a single zero-filled storage buffer.
     *
     * @param device - The GPU device.
     * @param size - Buffer size in bytes.
     * @returns The created buffer.
     */
    private createBuffer(device: GPUDevice, size: number): GPUBuffer {
        const buffer = device.createBuffer({
            size,
            usage: POINT_BUFFER_USAGE,
            mappedAtCreation: true,
        });
        const mapped = new Float32Array(buffer.getMappedRange());
        mapped.fill(0);
        buffer.unmap();
        return buffer;
    }
}
