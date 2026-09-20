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
 * normal (`vec4f`, 16 bytes). It is a derived quantity — overwritten every
 * frame by the relax pipeline's normal-precompute sub-pass from the
 * current positions — and is shared between relax (curvature-aware
 * repulsion) and the point renderer (future back-face culling). It is not
 * ping-ponged because it has no temporal state of its own.
 *
 * No CPU readback is performed — seeding and relaxation are entirely
 * GPU-side.
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
 * plus a single shared normals buffer.
 *
 * Both point buffers are initialized to zero. The seed pipeline writes the
 * initial distribution into one of them; the relax pipeline ping-pongs
 * between the two. The normals buffer is zero-initialized and overwritten
 * every frame by the relax pipeline's normal-precompute sub-pass.
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
     * Written every frame by the relax pipeline's normal-precompute
     * sub-pass; read by the curvature-aware relax pass and (in future) by
     * the point renderer for back-face culling.
     */
    public readonly normalsBuffer: GPUBuffer;

    /**
     * Creates both ping-pong point buffers and the shared normals buffer,
     * all zero-filled via `mappedAtCreation`.
     *
     * @param device - The GPU device used to create the buffers.
     * @param count - Number of points.
     */
    public constructor(device: GPUDevice, count: number) {
        this.count = count;
        const size = count * POINT_BYTES;

        this.bufferA = this.createBuffer(device, size);
        this.bufferB = this.createBuffer(device, size);
        this.normalsBuffer = this.createBuffer(device, size);
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
