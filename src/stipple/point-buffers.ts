/**
 * 2D point data layout and ping-pong storage buffer management for the
 * stippling pipeline.
 *
 * Each point is packed into a single `vec4f` (16 bytes):
 *
 * ```
 * offset 0:  pos.xy  (vec2f — position in [0,1] UV space)
 * offset 8:  vel.xy  (vec2f — velocity, used by the relax pass)
 * ```
 *
 * Two storage buffers (`bufferA`, `bufferB`) form a ping-pong pair. The
 * seed pipeline writes the initial distribution; the relax pipeline
 * (added in step 3) reads from one buffer and writes to the other each
 * frame.
 *
 * No CPU readback is performed — seeding and relaxation are entirely
 * GPU-side.
 */

/** Number of `f32` values per point (one `vec4f`). */
export const POINT_FLOATS = 4;

/** Size of one point in bytes. */
export const POINT_BYTES = POINT_FLOATS * Float32Array.BYTES_PER_ELEMENT;

/** Buffer usage flags for point storage buffers. */
const POINT_BUFFER_USAGE: GPUBufferUsageFlags =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Owns the ping-pong pair of storage buffers that hold point data.
 *
 * Both buffers are initialized to zero. The seed pipeline writes the
 * initial distribution into one of them; the relax pipeline ping-pongs
 * between the two.
 */
export class PointBuffers {
    /** Number of points stored in each buffer. */
    public readonly count: number;

    /** First buffer in the ping-pong pair. */
    public readonly bufferA: GPUBuffer;

    /** Second buffer in the ping-pong pair. */
    public readonly bufferB: GPUBuffer;

    /**
     * Creates both storage buffers, zero-filled via `mappedAtCreation`.
     *
     * @param device - The GPU device used to create the buffers.
     * @param count - Number of points.
     */
    public constructor(device: GPUDevice, count: number) {
        this.count = count;
        const size = count * POINT_BYTES;

        this.bufferA = this.createBuffer(device, size);
        this.bufferB = this.createBuffer(device, size);
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
