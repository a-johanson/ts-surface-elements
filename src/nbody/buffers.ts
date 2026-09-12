/**
 * Body data layout and ping-pong storage buffer management for the n-body
 * simulation.
 *
 * Each body is represented as two `vec4<f32>` values packed into 32 bytes:
 *
 * ```
 * offset 0:  pos.xyz  + mass   (vec4 — position in .xyz, mass in .w)
 * offset 16: vel.xyz  + pad    (vec4 — velocity in .xyz, unused .w)
 * ```
 *
 * Two storage buffers (`bufferA`, `bufferB`) form a ping-pong pair. Each
 * compute step reads from one and writes to the other, avoiding read-write
 * races on the same buffer within a single dispatch.
 */

/** Number of `f32` values per body (two `vec4`s). */
export const BODY_FLOATS = 8;

/** Size of one body in bytes. */
export const BODY_BYTES = BODY_FLOATS * Float32Array.BYTES_PER_ELEMENT;

/** Default number of bodies in the simulation. */
export const DEFAULT_BODY_COUNT = 512;

/** Buffer usage flags for body storage buffers. */
const BODY_BUFFER_USAGE: GPUBufferUsageFlags =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Generates initial body data as a packed `Float32Array`.
 *
 * Bodies are distributed uniformly inside a sphere of the given radius.
 * Each body receives a small random velocity and a uniform mass. The
 * seeding is deterministic given the same `Math.random` state.
 *
 * @param count - Number of bodies to generate.
 * @param radius - Radius of the sphere in simulation units.
 * @param mass - Mass of each body.
 * @returns Packed body data, length `count * {@link BODY_FLOATS}`.
 */
export function createSphereSeed(count: number, radius = 20, mass = 1): Float32Array {
    const data = new Float32Array(count * BODY_FLOATS);

    for (let i = 0; i < count; i += 1) {
        const offset = i * BODY_FLOATS;

        // Uniform distribution inside a sphere:
        // direction = uniform on unit sphere, radius = R * cbrt(u).
        const u = Math.random();
        const r = radius * Math.cbrt(u);
        const cosPhi = 2 * Math.random() - 1;
        const sinPhi = Math.sqrt(1 - cosPhi * cosPhi);
        const theta = Math.random() * Math.PI * 2;

        data[offset + 0] = r * sinPhi * Math.cos(theta); // posX
        data[offset + 1] = r * sinPhi * Math.sin(theta); // posY
        data[offset + 2] = r * cosPhi; // posZ
        data[offset + 3] = mass; // mass

        // Small random velocities — tuned later for stable orbits.
        data[offset + 4] = (Math.random() - 0.5) * 0.5; // velX
        data[offset + 5] = (Math.random() - 0.5) * 0.5; // velY
        data[offset + 6] = (Math.random() - 0.5) * 0.5; // velZ
        data[offset + 7] = 0; // pad (unused)
    }

    return data;
}

/**
 * Owns the ping-pong pair of storage buffers that hold body data.
 *
 * Both buffers are initialized with the same seed data. The compute
 * pipeline reads from one buffer and writes to the other each frame;
 * the caller tracks which direction is active and swaps after each
 * dispatch.
 */
export class BodyBuffers {
    /** Number of bodies stored in each buffer. */
    public readonly count: number;

    /** First buffer in the ping-pong pair. */
    public readonly bufferA: GPUBuffer;

    /** Second buffer in the ping-pong pair. */
    public readonly bufferB: GPUBuffer;

    /**
     * Creates both storage buffers and fills them with the given initial
     * data via `mappedAtCreation`.
     *
     * @param device - The GPU device used to create the buffers.
     * @param initialData - Packed body data; must have length
     *   `count * {@link BODY_FLOATS}`.
     * @throws {Error} If `initialData.length` is not a multiple of
     *   {@link BODY_FLOATS}.
     */
    public constructor(device: GPUDevice, initialData: Float32Array) {
        if (initialData.length % BODY_FLOATS !== 0) {
            throw new Error(
                `Initial data length ${initialData.length} is not a multiple of BODY_FLOATS (${BODY_FLOATS}).`,
            );
        }

        this.count = initialData.length / BODY_FLOATS;
        const size = initialData.length * Float32Array.BYTES_PER_ELEMENT;

        this.bufferA = this.createBuffer(device, size, initialData);
        this.bufferB = this.createBuffer(device, size, initialData);
    }

    /**
     * Creates a single storage buffer pre-filled with data.
     *
     * Uses `mappedAtCreation: true` to write initial data without a copy
     * command. The buffer is unmapped before returning.
     *
     * @param device - The GPU device.
     * @param size - Buffer size in bytes.
     * @param data - Initial data to write.
     * @returns The created and filled buffer.
     */
    private createBuffer(device: GPUDevice, size: number, data: Float32Array): GPUBuffer {
        const buffer = device.createBuffer({
            size,
            usage: BODY_BUFFER_USAGE,
            mappedAtCreation: true,
        });
        const mapped = new Float32Array(buffer.getMappedRange());
        mapped.set(data);
        buffer.unmap();
        return buffer;
    }

    /**
     * Copies the contents of one body buffer to the CPU for inspection.
     *
     * Creates a temporary `MAP_READ` staging buffer, submits a
     * `copyBufferToBuffer` command, then maps and reads the staging
     * buffer. The staging buffer is destroyed after reading.
     *
     * This is an asynchronous operation that awaits GPU completion of
     * all previously submitted commands. It is intended for debugging
     * and verification, not for per-frame use.
     *
     * @param device - The GPU device.
     * @param fromBufferA - If `true`, reads from `bufferA`; else `bufferB`.
     * @returns Packed body data from the chosen buffer.
     */
    public async readback(device: GPUDevice, fromBufferA: boolean): Promise<Float32Array> {
        const source = fromBufferA ? this.bufferA : this.bufferB;

        const staging = device.createBuffer({
            label: "nbody-readback-staging",
            size: source.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
        device.queue.submit([encoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const arrayBuffer = staging.getMappedRange();
        const copy = arrayBuffer.slice(0);
        staging.unmap();
        staging.destroy();
        return new Float32Array(copy);
    }
}
