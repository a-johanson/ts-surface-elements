/**
 * Point render pipeline — draws stipple points as tangent-plane quads
 * oriented from the shared normals buffer and projected from 3D world
 * space onto the canvas.
 *
 * The vertex shader builds an orthonormal tangent basis from the
 * pre-computed surface normal at each point and offsets the quad corners
 * in world space by `point_radius_world`, so quads lie flat on the SDF
 * surface and foreshorten with viewing angle.
 *
 * Two bind groups support the ping-pong buffer pair so the render can
 * read from whichever buffer the relax pass most recently wrote to. The
 * shared normals and shading buffers are non-ping-ponged.
 */

import type { CameraConfig } from "./debug-render-pipeline.js";
import type { PointBuffers } from "./point-buffers.js";
import { POINT_SHADER } from "./shaders.js";

/**
 * Size of the point uniform buffer in bytes: `mat4x4f` (64).
 */
const UNIFORM_BUFFER_BYTES = 64;

/** Near plane for the perspective projection. */
const NEAR = 0.1;

/** Far plane for the perspective projection. */
const FAR = 100;

/** Column-major 4×4 matrix stored as 16 floats. */
type Mat4 = number[];

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
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    const fLen = Math.hypot(fx, fy, fz) || 1;
    fx /= fLen;
    fy /= fLen;
    fz /= fLen;

    let rx = fy * up[2] - fz * up[1];
    let ry = fz * up[0] - fx * up[2];
    let rz = fx * up[1] - fy * up[0];
    const rLen = Math.hypot(rx, ry, rz) || 1;
    rx /= rLen;
    ry /= rLen;
    rz /= rLen;

    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

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
    const f = 1 / Math.tan(fov / 2);
    return [
        f / aspect,
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
 * Manages the point render pipeline, uniform buffer, and ping-pong bind
 * groups.
 */
export class PointRenderPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly uniformBuffer: GPUBuffer;
    private readonly bindGroupA: GPUBindGroup;
    private readonly bindGroupB: GPUBindGroup;
    private readonly config: CameraConfig;

    /**
     * Creates the shader module, render pipeline, uniform buffer, and
     * both ping-pong bind groups.
     *
     * @param device - The GPU device.
     * @param points - The ping-pong point buffer pair.
     * @param config - Static camera configuration (FOV, target, up).
     * @param format - The canvas texture format.
     */
    public constructor(
        device: GPUDevice,
        points: PointBuffers,
        config: CameraConfig,
        format: GPUTextureFormat,
    ) {
        this.device = device;
        this.config = config;

        const shaderModule = device.createShaderModule({
            label: "stipple-point-shader",
            code: POINT_SHADER,
        });

        this.pipeline = device.createRenderPipeline({
            label: "stipple-point-pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "point_vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "point_fs",
                targets: [
                    {
                        format,
                        blend: {
                            color: {
                                srcFactor: "src-alpha",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                            alpha: {
                                srcFactor: "one",
                                dstFactor: "one-minus-src-alpha",
                                operation: "add",
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        this.uniformBuffer = device.createBuffer({
            label: "stipple-point-uniform",
            size: UNIFORM_BUFFER_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const layout = this.pipeline.getBindGroupLayout(0);

        this.bindGroupA = device.createBindGroup({
            label: "stipple-point-bind-A",
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: points.bufferA } },
                { binding: 2, resource: { buffer: points.shadingBuffer } },
                { binding: 3, resource: { buffer: points.normalsBuffer } },
            ],
        });

        this.bindGroupB = device.createBindGroup({
            label: "stipple-point-bind-B",
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: points.bufferB } },
                { binding: 2, resource: { buffer: points.shadingBuffer } },
                { binding: 3, resource: { buffer: points.normalsBuffer } },
            ],
        });
    }

    /**
     * Writes the view-projection matrix into the uniform buffer.
     *
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose backing-store size determines the
     *   projection aspect ratio.
     */
    private writeUniform(
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
    ): void {
        const aspect = canvas.width / canvas.height;
        const view = lookAt(eye, this.config.target, this.config.up);
        const proj = perspective(this.config.fov, aspect);
        const viewProj = multiply(proj, view);

        const buffer = new ArrayBuffer(UNIFORM_BUFFER_BYTES);
        const f32 = new Float32Array(buffer);
        f32.set(viewProj, 0);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer);
    }

    /**
     * Draws all points as instanced tangent-plane quads into the given
     * render pass.
     *
     * Must be called *inside* an active `GPURenderPassEncoder`, after the
     * debug render (or any other content) has been drawn.
     *
     * @param pass - The active render pass encoder.
     * @param eye - Camera eye position [x, y, z].
     * @param canvas - The canvas whose size determines the projection
     * aspect.
     * @param readFromBufferA - If `true`, reads from `bufferA`; else
     * `bufferB`. Should match the buffer the seed/relax pass most
     * recently wrote.
     * @param instanceCount - Number of points (instances) to draw.
     */
    public render(
        pass: GPURenderPassEncoder,
        eye: readonly [number, number, number],
        canvas: HTMLCanvasElement,
        readFromBufferA: boolean,
        instanceCount: number,
    ): void {
        this.writeUniform(eye, canvas);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, readFromBufferA ? this.bindGroupA : this.bindGroupB);
        pass.draw(6, instanceCount);
    }
}
