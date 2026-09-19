/**
 * Relax compute pipeline — redistributes stipple points to match the
 * density texture via density-aware attraction/repulsion.
 *
 * Each frame, two compute dispatches are recorded into the command
 * encoder:
 *
 * 1. **Sample-densities dispatch** — each invocation `i` samples
 *    `d_i = density(p_i)` from the density texture once and writes it
 *    into a small `densities: array<f32>` storage buffer. This avoids
 *    O(n²) texture lookups in the relax pass.
 *
 * 2. **Relax dispatch** — each invocation `i` reads `p_i`, `v_i`, `d_i`
 *    and loops over all `j ≠ i` reading `p_j`, `d_j` from storage
 *    buffers (no texture access in the hot loop). Accumulates force per
 *    the attraction/repulsion model, integrates with semi-implicit Euler,
 *    and writes the updated point to the output buffer.
 *
 * Ping-pong: the two dispatches read from one point buffer and write to
 * the other. Four bind groups cover both directions × both passes. The
 * sample bind groups are recreated when the density texture changes
 * (canvas resize); the relax bind groups are static.
 */

import type { PointBuffers } from "./point-buffers.js";
import { RELAX_SHADER, SAMPLE_DENSITIES_SHADER } from "./shaders.js";

/** Workgroup size — must match `@workgroup_size(64)` in the WGSL. */
const WORKGROUP_SIZE = 64;

/** Size of the params uniform buffer in bytes (8 × f32/u32 = 32). */
const PARAMS_BUFFER_BYTES = 32;

/** Relaxation parameters passed to the compute shaders via uniform. */
export interface RelaxParams {
    /** Time step per frame. */
    readonly dt: number;
    /** Repulsion strength between two inside particles. */
    readonly kRep: number;
    /** Attraction strength, outside → inside. */
    readonly kAtt: number;
    /** Mild push strength, inside particle away from outside. */
    readonly kPush: number;
    /** Density-coupling factor for repulsion weakening (0 = none, 1 = full). */
    readonly alpha: number;
    /** Plummer-style softening length, prevents singularities. */
    readonly softening: number;
    /** Per-frame velocity damping (0 = frozen, 1 = no damping). */
    readonly damping: number;
    /** Number of points. */
    readonly pointCount: number;
}

/**
 * Default relaxation parameters — initial guesses, need visual tuning.
 *
 * Points are in `[0,1]²` UV space, so forces and softening are scaled
 * accordingly. With 4K points, the average nearest-neighbor distance is
 * roughly `sqrt(1/4096) ≈ 0.016`; the softening is set above this to
 * prevent singularities.
 */
export const DEFAULT_RELAX_PARAMS: RelaxParams = {
    dt: 0.001,
    kRep: 0.005,
    kAtt: 0.003,
    kPush: 0.001,
    alpha: 0.8,
    softening: 0.03,
    damping: 0.9,
    pointCount: 4096,
};

/**
 * Manages the sample-densities and relax compute pipelines, the
 * densities aux buffer, the params uniform, and all ping-pong bind
 * groups.
 */
export class RelaxPipeline {
    private readonly device: GPUDevice;
    private readonly samplePipeline: GPUComputePipeline;
    private readonly relaxPipeline: GPUComputePipeline;
    private readonly sampleLayout: GPUBindGroupLayout;
    private readonly relaxLayout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;
    private readonly densitiesBuffer: GPUBuffer;

    // Sample bind groups — recreated when the density texture changes.
    private sampleBindGroupA: GPUBindGroup | null = null;
    private sampleBindGroupB: GPUBindGroup | null = null;
    private lastTexture: GPUTexture | null = null;

    // Relax bind groups — static, created once.
    private readonly relaxBindGroupAtoB: GPUBindGroup;
    private readonly relaxBindGroupBtoA: GPUBindGroup;

    /**
     * Creates both compute pipelines, the densities buffer, the params
     * uniform, and all bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair.
     */
    public constructor(device: GPUDevice, points: PointBuffers) {
        this.device = device;

        const sampleModule = device.createShaderModule({
            label: "stipple-sample-densities-shader",
            code: SAMPLE_DENSITIES_SHADER,
        });

        this.samplePipeline = device.createComputePipeline({
            label: "stipple-sample-densities-pipeline",
            layout: "auto",
            compute: {
                module: sampleModule,
                entryPoint: "sample_densities_cs",
            },
        });

        const relaxModule = device.createShaderModule({
            label: "stipple-relax-shader",
            code: RELAX_SHADER,
        });

        this.relaxPipeline = device.createComputePipeline({
            label: "stipple-relax-pipeline",
            layout: "auto",
            compute: {
                module: relaxModule,
                entryPoint: "relax_cs",
            },
        });

        this.sampleLayout = this.samplePipeline.getBindGroupLayout(0);
        this.relaxLayout = this.relaxPipeline.getBindGroupLayout(0);

        this.paramsBuffer = device.createBuffer({
            label: "stipple-relax-params",
            size: PARAMS_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.densitiesBuffer = device.createBuffer({
            label: "stipple-densities",
            size: points.count * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE,
        });

        this.relaxBindGroupAtoB = device.createBindGroup({
            label: "stipple-relax-bind-A-to-B",
            layout: this.relaxLayout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferA } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: this.densitiesBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });

        this.relaxBindGroupBtoA = device.createBindGroup({
            label: "stipple-relax-bind-B-to-A",
            layout: this.relaxLayout,
            entries: [
                { binding: 0, resource: { buffer: points.bufferB } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: this.densitiesBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });
    }

    /**
     * Writes the relaxation parameters into the uniform buffer.
     *
     * @param params - The parameters to upload.
     */
    private writeParams(params: RelaxParams): void {
        const buffer = new ArrayBuffer(PARAMS_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        const u32 = new Uint32Array(buffer);
        f32[0] = params.dt;
        f32[1] = params.kRep;
        f32[2] = params.kAtt;
        f32[3] = params.kPush;
        f32[4] = params.alpha;
        f32[5] = params.softening;
        f32[6] = params.damping;
        u32[7] = params.pointCount;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
    }

    /**
     * Recreates the sample bind groups if the density texture has
     * changed (canvas resize).
     *
     * @param densityTexture - The current density texture.
     * @param points - The ping-pong point buffer pair.
     */
    private ensureSampleBindGroups(densityTexture: GPUTexture, points: PointBuffers): void {
        if (densityTexture === this.lastTexture) {
            return;
        }
        this.lastTexture = densityTexture;

        const view = densityTexture.createView();

        this.sampleBindGroupA = this.device.createBindGroup({
            label: "stipple-sample-bind-A",
            layout: this.sampleLayout,
            entries: [
                { binding: 0, resource: view },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: this.densitiesBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });

        this.sampleBindGroupB = this.device.createBindGroup({
            label: "stipple-sample-bind-B",
            layout: this.sampleLayout,
            entries: [
                { binding: 0, resource: view },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: this.densitiesBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });
    }

    /**
     * Records both compute dispatches (sample-densities + relax) into
     * the given command encoder.
     *
     * Reads from one point buffer and writes to the other, depending on
     * `readFromA`. The caller must flip `readFromA` after each dispatch
     * to implement the ping-pong swap.
     *
     * @param encoder - The command encoder to record into.
     * @param densityTexture - The `r32float` density texture.
     * @param points - The ping-pong point buffer pair.
     * @param params - Relaxation parameters.
     * @param readFromA - If `true`, reads bufferA → writes bufferB;
     *   if `false`, reads bufferB → writes bufferA.
     */
    public dispatch(
        encoder: GPUCommandEncoder,
        densityTexture: GPUTexture,
        points: PointBuffers,
        params: RelaxParams,
        readFromA: boolean,
    ): void {
        this.ensureSampleBindGroups(densityTexture, points);
        this.writeParams(params);

        const workgroupCount = Math.ceil(params.pointCount / WORKGROUP_SIZE);

        // --- Pass 1: sample densities from the texture into the buffer ---
        const sampleBindGroup = readFromA ? this.sampleBindGroupA : this.sampleBindGroupB;
        if (sampleBindGroup === null) {
            throw new Error("Sample bind group was not created.");
        }

        const samplePass = encoder.beginComputePass();
        samplePass.setPipeline(this.samplePipeline);
        samplePass.setBindGroup(0, sampleBindGroup);
        samplePass.dispatchWorkgroups(workgroupCount);
        samplePass.end();

        // --- Pass 2: relax points using buffer-read densities ---
        const relaxPass = encoder.beginComputePass();
        relaxPass.setPipeline(this.relaxPipeline);
        relaxPass.setBindGroup(
            0,
            readFromA ? this.relaxBindGroupAtoB : this.relaxBindGroupBtoA,
        );
        relaxPass.dispatchWorkgroups(workgroupCount);
        relaxPass.end();
    }
}
