/**
 * Spatial grid build pipeline — uniform-grid acceleration for the relax pass.
 *
 * Owns the full grid build: cell-index computation (Step 3), radix sort
 * (Steps 4–5), and cell-range table (Step 6). This module is the single
 * orchestration call site in `main.ts` for the grid rebuild that runs once
 * per frame after `reproject` and before `relax`.
 *
 * The grid uses a fixed CPU-known {@link SceneBBOX} constant as its origin
 * and dimensions, shared with the seed pipeline. `gridDims` per axis is
 * `ceil(bbox extent / radius)`, computed CPU-side at construction. The
 * `radius` (cell size) is fixed at construction — a per-dispatch radius would
 * make `gridDims` stale relative to the pre-sized `cellStart`/`cellCount`
 * buffers.
 *
 * Step 3 implements the cell-index computation stage. Step 4 constructs the
 * radix-split pipeline and its bind groups. Step 5 wires the 32 per-bit
 * dispatches into `dispatch`. The cell-range stage is added in Step 6.
 */

import type { PointBuffers } from "./point-buffers.js";
import { buildRadixSplitShader, CELL_INDEX_SHADER } from "./shaders/index.js";
import { SUBGROUP_WORKGROUP_SIZE } from "./shaders/subgroup-common.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/**
 * Size of one `RadixBitParams` uniform slot in bytes.
 *
 * WGSL struct layout (uniform): `bit: u32` + three u32 pads = 16 bytes,
 * 16-aligned.
 */
const RADIX_BIT_PARAMS_SLOT_BYTES = 16;

/**
 * Maximum number of grid cells.
 *
 * Caps `cellStart` + `cellCount` memory at ~8 MB and prevents absurd
 * configurations. The target scale is tens of thousands of points; at
 * `POINT_COUNT = 4096` and `SCENE_BBOX 6×4×4 / radius 0.3`, the grid is
 * ~3920 cells — 250× headroom under this cap. A configuration exceeding
 * the cap has far more cells than points (e.g. >1M cells for <65k points
 * means most cells are permanently empty), indicating a misconfigured
 * radius or bounding box rather than a legitimate workload.
 */
const MAX_CELLS = 1_048_576;

/**
 * Size of the `GridParams` uniform buffer in bytes.
 *
 * WGSL struct layout (uniform):
 *   `bbox_min: vec3f` (offset 0, align 16) + `cell_size: f32` (offset 12)
 *   `bbox_max: vec3f` (offset 16, align 16) + `point_count: u32` (offset 28)
 *   `grid_dims: vec3u` (offset 32, align 16) + `_pad: u32` (offset 44)
 * Struct size rounds up to 48 (alignment 16).
 */
const GRID_PARAMS_BUFFER_BYTES = 48;

/**
 * Fixed CPU-known bounding box for the animated SDF scene, shared between
 * the seed pipeline (rejection sampling) and the spatial grid (cell origin
 * and dimensions). Chosen once to safely contain the surface across all
 * animation phases.
 */
export interface SceneBBox {
    /** Inclusive lower corner of the scene bounding box. */
    readonly min: readonly [number, number, number];
    /** Inclusive upper corner of the scene bounding box. */
    readonly max: readonly [number, number, number];
}

/**
 * Manages the spatial grid build: cell-index computation, radix sort, and
 * cell-range table.
 *
 * All grid buffers are created upfront at construction. Step 3 binds the
 * cell-index stage's buffers; Step 4 constructs the radix-split pipeline;
 * Step 5 wires the per-bit dispatches (pass count = bit width of
 * `numCells - 1`, not a fixed 32) and exposes the sorted output pair via
 * `sortedKeys` / `sortedValues`. The cell-range stage is added in Step 6.
 */
export class SpatialGridPipeline {
    private readonly device: GPUDevice;
    private readonly pointCount: number;
    private readonly gridParamsBuffer: GPUBuffer;
    private readonly keysA: GPUBuffer;
    private readonly valuesA: GPUBuffer;
    private readonly keysB: GPUBuffer;
    private readonly valuesB: GPUBuffer;
    // biome-ignore lint/correctness/noUnusedPrivateClassMembers: cell-range table, wired in Step 6
    private readonly cellStart: GPUBuffer;
    // biome-ignore lint/correctness/noUnusedPrivateClassMembers: cell-range table, wired in Step 6
    private readonly cellCount: GPUBuffer;
    private readonly cellIndexPipeline: GPUComputePipeline;
    private readonly cellIndexLayout: GPUBindGroupLayout;
    private readonly cellIndexBindGroupA: GPUBindGroup;
    private readonly cellIndexBindGroupB: GPUBindGroup;
    private readonly radixBitParamsBuffer: GPUBuffer;
    private readonly radixSplitPipeline: GPUComputePipeline;
    private readonly radixSplitLayout: GPUBindGroupLayout;
    /**
     * One bind group per bit pass; even passes read A→write B, odd passes
     * read B→write A.
     */
    private readonly radixBindGroups: readonly GPUBindGroup[];
    /**
     * The buffer pair holding the final sorted `(keys, values)` after all
     * radix passes. If the pass count is even, the sort lands in A; if odd,
     * in B. Steps 6 and 7 read from these instead of hardcoding the A pair.
     */
    // biome-ignore lint/correctness/noUnusedPrivateClassMembers: read in Steps 6 and 7
    private readonly sortedKeys: GPUBuffer;
    // biome-ignore lint/correctness/noUnusedPrivateClassMembers: read in Steps 6 and 7
    private readonly sortedValues: GPUBuffer;

    /**
     * Per-axis grid dimensions from the scene bounding box and cell size
     * (relaxation radius).
     *
     * @param bbox - The scene bounding box.
     * @param cellSize - The cell size (relaxation radius).
     * @returns A 3-tuple of per-axis cell counts.
     */
    private static gridDims(
        bbox: SceneBBox,
        cellSize: number,
    ): readonly [number, number, number] {
        return [
            Math.ceil((bbox.max[0] - bbox.min[0]) / cellSize),
            Math.ceil((bbox.max[1] - bbox.min[1]) / cellSize),
            Math.ceil((bbox.max[2] - bbox.min[2]) / cellSize),
        ];
    }

    /**
     * Number of radix-split passes needed for a given cell count.
     *
     * Cell keys are linear indices in `[0, numCells-1]`. Only the bits up to
     * the most significant set bit of `numCells - 1` can vary, so higher bits
     * are always 0 and sorting them is a no-op. Returns the exact bit width,
     * skipping those no-op passes. For the default config (3920 cells) this
     * is 12 passes instead of 32 — a 2.7× reduction in sort work per frame.
     *
     * @param numCells - The total number of grid cells.
     * @returns The number of radix passes (0 if `numCells <= 1`).
     */
    private static radixPasses(numCells: number): number {
        if (numCells <= 1) {
            return 0;
        }
        return 32 - Math.clz32(numCells - 1);
    }

    /**
     * Creates the cell-index shader module, compute pipeline, params
     * uniform, and all grid storage buffers.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair.
     * @param sceneBBox - The fixed scene bounding box.
     * @param radius - The relaxation radius (cell size). Fixed at
     *   construction; changing it requires recreating the pipeline.
     */
    public constructor(
        device: GPUDevice,
        points: PointBuffers,
        sceneBBox: SceneBBox,
        radius: number,
    ) {
        this.device = device;
        this.pointCount = points.count;

        const gridDims = SpatialGridPipeline.gridDims(sceneBBox, radius);
        const numCells = gridDims[0] * gridDims[1] * gridDims[2];
        if (numCells > MAX_CELLS) {
            throw new Error(
                `Spatial grid cell count ${numCells} exceeds MAX_CELLS (${MAX_CELLS}). ` +
                    `gridDims=${gridDims.join("×")}, bbox extent=` +
                    `[${sceneBBox.max[0] - sceneBBox.min[0]}, ` +
                    `${sceneBBox.max[1] - sceneBBox.min[1]}, ` +
                    `${sceneBBox.max[2] - sceneBBox.min[2]}], ` +
                    `radius=${radius}. Reduce the bbox extent or increase the radius.`,
            );
        }

        this.gridParamsBuffer = device.createBuffer({
            label: "spatial-grid-params",
            size: GRID_PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.writeGridParams(sceneBBox, radius, gridDims, numCells);

        const keysUsage: GPUBufferUsageFlags =
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const cellRangesUsage: GPUBufferUsageFlags =
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

        this.keysA = device.createBuffer({
            label: "spatial-grid-keys-A",
            size: points.count * 4,
            usage: keysUsage,
        });
        this.valuesA = device.createBuffer({
            label: "spatial-grid-values-A",
            size: points.count * 4,
            usage: keysUsage,
        });
        this.keysB = device.createBuffer({
            label: "spatial-grid-keys-B",
            size: points.count * 4,
            usage: keysUsage,
        });
        this.valuesB = device.createBuffer({
            label: "spatial-grid-values-B",
            size: points.count * 4,
            usage: keysUsage,
        });
        this.cellStart = device.createBuffer({
            label: "spatial-grid-cell-start",
            size: numCells * 4,
            usage: cellRangesUsage,
        });
        this.cellCount = device.createBuffer({
            label: "spatial-grid-cell-count",
            size: numCells * 4,
            usage: cellRangesUsage,
        });

        const module = device.createShaderModule({
            label: "spatial-grid-cell-index-shader",
            code: CELL_INDEX_SHADER,
        });

        this.cellIndexPipeline = device.createComputePipeline({
            label: "spatial-grid-cell-index-pipeline",
            layout: "auto",
            compute: {
                module,
                entryPoint: "cell_index_cs",
            },
        });

        this.cellIndexLayout = this.cellIndexPipeline.getBindGroupLayout(0);

        this.cellIndexBindGroupA = device.createBindGroup({
            label: "spatial-grid-cell-index-bind-A",
            layout: this.cellIndexLayout,
            entries: [
                { binding: 0, resource: { buffer: this.gridParamsBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: this.keysA } },
                { binding: 3, resource: { buffer: this.valuesA } },
            ],
        });
        this.cellIndexBindGroupB = device.createBindGroup({
            label: "spatial-grid-cell-index-bind-B",
            layout: this.cellIndexLayout,
            entries: [
                { binding: 0, resource: { buffer: this.gridParamsBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: this.keysA } },
                { binding: 3, resource: { buffer: this.valuesA } },
            ],
        });

        // --- Radix split (Steps 4–5) -----------------------------------------

        const passCount = SpatialGridPipeline.radixPasses(numCells);
        const uniformSlotStride = device.limits.minUniformBufferOffsetAlignment;
        const radixParamsBytes = Math.max(passCount, 1) * uniformSlotStride;
        const radixParamsData = new ArrayBuffer(radixParamsBytes);
        const radixParamsU32 = new Uint32Array(radixParamsData);
        for (let b = 0; b < passCount; b++) {
            radixParamsU32[(b * uniformSlotStride) / 4] = b;
        }

        this.radixBitParamsBuffer = device.createBuffer({
            label: "spatial-grid-radix-bit-params",
            size: radixParamsBytes,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.radixBitParamsBuffer, 0, radixParamsData);

        const elementsPerThread = Math.ceil(this.pointCount / SUBGROUP_WORKGROUP_SIZE);
        if (this.pointCount < SUBGROUP_WORKGROUP_SIZE) {
            throw new Error(
                `Spatial grid radix sort requires pointCount >= ${SUBGROUP_WORKGROUP_SIZE} ` +
                    `(got ${this.pointCount}). The sort runs as a single workgroup; ` +
                    `fewer points would leave threads idle with no owned elements.`,
            );
        }

        const radixModule = device.createShaderModule({
            label: "spatial-grid-radix-split-shader",
            code: buildRadixSplitShader(this.pointCount, elementsPerThread),
        });

        this.radixSplitPipeline = device.createComputePipeline({
            label: "spatial-grid-radix-split-pipeline",
            layout: "auto",
            compute: {
                module: radixModule,
                entryPoint: "radix_split_cs",
            },
        });

        this.radixSplitLayout = this.radixSplitPipeline.getBindGroupLayout(0);

        // One bind group per bit pass. Each binds a different slot of the
        // uniform buffer (fixed offset = b × uniformSlotStride) and alternates
        // the read/write direction: even passes read A→write B, odd passes
        // read B→write A. After `passCount` passes, the sorted data lands in
        // A if even, B if odd — `sortedKeys` / `sortedValues` capture which.
        const bindGroups: GPUBindGroup[] = [];
        for (let b = 0; b < passCount; b++) {
            const readA = b % 2 === 0;
            const keysIn = readA ? this.keysA : this.keysB;
            const valuesIn = readA ? this.valuesA : this.valuesB;
            const keysOut = readA ? this.keysB : this.keysA;
            const valuesOut = readA ? this.valuesB : this.valuesA;
            bindGroups.push(
                device.createBindGroup({
                    label: `spatial-grid-radix-bind-bit-${b}`,
                    layout: this.radixSplitLayout,
                    entries: [
                        {
                            binding: 0,
                            resource: {
                                buffer: this.radixBitParamsBuffer,
                                offset: b * uniformSlotStride,
                                size: RADIX_BIT_PARAMS_SLOT_BYTES,
                            },
                        },
                        { binding: 1, resource: { buffer: keysIn } },
                        { binding: 2, resource: { buffer: valuesIn } },
                        { binding: 3, resource: { buffer: keysOut } },
                        { binding: 4, resource: { buffer: valuesOut } },
                    ],
                }),
            );
        }
        this.radixBindGroups = bindGroups;

        const sortedInA = passCount % 2 === 0;
        this.sortedKeys = sortedInA ? this.keysA : this.keysB;
        this.sortedValues = sortedInA ? this.valuesA : this.valuesB;
    }

    /**
     * Writes the grid parameters into the uniform buffer (once at startup).
     *
     * Layout (48 bytes):
     *   `bbox_min: vec3f + cell_size: f32`,
     *   `bbox_max: vec3f + point_count: u32`,
     *   `grid_dims: vec3u + _pad: u32`.
     *
     * @param bbox - The scene bounding box.
     * @param cellSize - The cell size (relaxation radius).
     * @param gridDims - Per-axis cell counts.
     * @param pointCount - Number of points.
     */
    private writeGridParams(
        bbox: SceneBBox,
        cellSize: number,
        gridDims: readonly [number, number, number],
        pointCount: number,
    ): void {
        const buffer = new ArrayBuffer(GRID_PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = bbox.min[0];
        f32[1] = bbox.min[1];
        f32[2] = bbox.min[2];
        f32[3] = cellSize;
        f32[4] = bbox.max[0];
        f32[5] = bbox.max[1];
        f32[6] = bbox.max[2];
        u32[7] = pointCount;
        u32[8] = gridDims[0];
        u32[9] = gridDims[1];
        u32[10] = gridDims[2];
        this.device.queue.writeBuffer(this.gridParamsBuffer, 0, buffer);
    }

    /**
     * Records the spatial grid build into the given command encoder.
     *
     * Two compute passes: (1) cell-index computation — reads the point buffer
     * indicated by `readFromA`, writes `keysA` / `valuesA`; (2) radix sort —
     * one per-bit pass per set bit in the cell key range, in a single compute
     * pass, alternating the A/B read-write direction. The implicit
     * pass-boundary barrier between the two passes ensures the cell-index
     * writes are visible to the sort. After all passes the sorted
     * `(keys, values)` lands in whichever pair `sortedKeys` / `sortedValues`
     * points to (A if even pass count, B if odd). The cell-range table
     * (Step 6) is not yet implemented.
     *
     * @param encoder - The command encoder to record into.
     * @param readFromA - If `true`, reads `points.bufferA`; else
     *   `points.bufferB`. Should match the buffer `relax` will read.
     */
    public dispatch(encoder: GPUCommandEncoder, readFromA: boolean): void {
        const cellIndexWorkgroups = Math.ceil(this.pointCount / WORKGROUP_SIZE);

        const cellIndexPass = encoder.beginComputePass();
        cellIndexPass.setPipeline(this.cellIndexPipeline);
        cellIndexPass.setBindGroup(
            0,
            readFromA ? this.cellIndexBindGroupA : this.cellIndexBindGroupB,
        );
        cellIndexPass.dispatchWorkgroups(cellIndexWorkgroups);
        cellIndexPass.end();

        if (this.radixBindGroups.length === 0) {
            return;
        }

        const sortPass = encoder.beginComputePass();
        sortPass.setPipeline(this.radixSplitPipeline);
        for (const bindGroup of this.radixBindGroups) {
            sortPass.setBindGroup(0, bindGroup);
            sortPass.dispatchWorkgroups(1);
        }
        sortPass.end();
    }
}
