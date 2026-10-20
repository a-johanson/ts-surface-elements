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
 * the seed pass at bootstrap, every playing frame by the reproject pass
 * (before relax, from the buffer relax is about to read), and every
 * playing frame by the normals-refresh pass (after relax, from the
 * buffer relax just wrote) — and is shared between relax (tangent-plane
 * projection) and the hatch pass. It is not ping-ponged because it has no
 * temporal state of its own; the invariant "normals match whichever
 * buffer relax reads" is preserved because the normals-refresh pass
 * writes normals from the buffer that the *next* relax pass will read.
 *
 * A second non-ping-pong `luminanceBuffer` holds one `f32` luminance
 * value per point. It is written every playing frame by the reproject
 * pass (before relax, for the relax kernel's density modulation) and
 * read by the relax pass.
 *
 * No CPU readback is performed — seeding, relaxation, and normals
 * refresh are entirely GPU-side.
 */

/** Number of `f32` values per point (one `vec4f`). */
export const POINT_FLOATS = 4;

/** Size of one point in bytes. */
export const POINT_BYTES = POINT_FLOATS * Float32Array.BYTES_PER_ELEMENT;

/** Buffer usage flags for point and normals storage buffers. */
const POINT_BUFFER_USAGE: GPUBufferUsageFlags =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Owns the ping-pong pair of storage buffers that hold point positions,
 * plus the shared normals and luminance buffers.
 *
 * Both point buffers are initialized to zero. The seed pipeline writes the
 * initial distribution (and matching normals) into one of them; the relax
 * pipeline ping-pongs between the two. The normals buffer is
 * zero-initialized, seeded once at bootstrap, and refreshed every playing
 * frame by the reproject and normals-refresh passes. The luminance buffer
 * is zero-initialized and overwritten every playing frame by the
 * reproject pass.
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
     * Written once at bootstrap by the seed pass, every playing frame by
     * the reproject pass (before relax, from the buffer relax is about to
     * read), and every playing frame by the normals-refresh pass (after
     * relax, from the buffer relax just wrote). Read by the relax pass,
     * the hatch pass, and the point renderer (to orient tangent-plane
     * quads).
     */
    public readonly normalsBuffer: GPUBuffer;

    /**
     * Shared per-point luminance buffer (not ping-ponged).
     *
     * One `f32` per point: the view-independent luminance, written every
     * playing frame by the reproject pass (before relax) and read by the
     * relax pass for per-neighbor density modulation.
     */
    public readonly luminanceBuffer: GPUBuffer;

    /**
     * Creates both ping-pong point buffers plus the shared normals and
     * luminance buffers, all zero-filled via `mappedAtCreation`.
     *
     * @param device - The GPU device used to create the buffers.
     * @param count - Number of points.
     */
    public constructor(device: GPUDevice, count: number) {
        this.count = count;
        // Points and normals are vec4f-structured (pos + unused w,
        // vec3f + pad), 16 bytes each; the luminance buffer stores one
        // `f32` per point. WGSL storage arrays of vec4f are 16-byte
        // aligned, which the sizes satisfy.
        const size = count * POINT_BYTES;
        const luminanceSize = count * Float32Array.BYTES_PER_ELEMENT;

        this.bufferA = this.createBuffer(device, size);
        this.bufferB = this.createBuffer(device, size);
        this.normalsBuffer = this.createBuffer(device, size);
        this.luminanceBuffer = this.createBuffer(device, luminanceSize);
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
