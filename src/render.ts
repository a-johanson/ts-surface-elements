import type { CanvasDimensions } from "./canvas-dimensions";
import { createMat4LookAt, createMat4Perspective, multiplyMat4 } from "./mat4";
import { ELEMENT_FRAG, ELEMENT_VERT } from "./shaders";

const ELEMENT_COUNT = 7000;
const FIELD_OF_VIEW = (45 * Math.PI) / 180;
const NEAR_PLANE = 0.1;
const FAR_PLANE = 100;
const EYE_Z = 3.5;

/**
 * Compiles a shader and returns its handle.
 *
 * @param gl - The WebGL2 rendering context.
 * @param type - Shader type (`gl.VERTEX_SHADER` or `gl.FRAGMENT_SHADER`).
 * @param source - GLSL source code.
 * @returns Compiled shader handle.
 * @throws {Error} When compilation fails.
 */
function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (shader === null) {
        throw new Error("Unable to allocate a shader object.");
    }
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Shader compilation failed: ${info}`);
    }
    return shader;
}

/**
 * Links a vertex/fragment pair into a program.
 *
 * @param gl - The WebGL2 rendering context.
 * @param vertexSource - Vertex shader GLSL source.
 * @param fragmentSource - Fragment shader GLSL source.
 * @returns Linked program handle.
 * @throws {Error} When linking fails.
 */
function linkProgram(
    gl: WebGL2RenderingContext,
    vertexSource: string,
    fragmentSource: string,
): WebGLProgram {
    const program = gl.createProgram();
    if (program === null) {
        throw new Error("Unable to allocate a program object.");
    }
    const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(`Program linking failed: ${info}`);
    }
    return program;
}

/**
 * WebGL2 renderer that draws surface elements on an organic shape via a
 * single instanced draw call with no vertex buffers.
 */
export class Renderer {
    private readonly gl: WebGL2RenderingContext;
    private readonly program: WebGLProgram;
    private readonly vertexArray: WebGLVertexArrayObject;
    private readonly viewProjectionLocation: WebGLUniformLocation;
    private readonly timeLocation: WebGLUniformLocation;
    private readonly elementCountLocation: WebGLUniformLocation;

    /**
     * Creates the renderer, compiles shaders, sets up immutable GL state, and
     * uploads the static view-projection and element-count uniforms.
     *
     * @param gl - The WebGL2 rendering context.
     * @param dimensions - The output canvas dimensions.
     */
    public constructor(gl: WebGL2RenderingContext, dimensions: CanvasDimensions) {
        this.gl = gl;

        this.program = linkProgram(gl, ELEMENT_VERT, ELEMENT_FRAG);

        const projection = createMat4Perspective(
            FIELD_OF_VIEW,
            dimensions.width / dimensions.height,
            NEAR_PLANE,
            FAR_PLANE,
        );
        const view = createMat4LookAt(0, 0, EYE_Z, 0, 0, 0, 0, 1, 0);
        const viewProjection = multiplyMat4(projection, view);

        const vertexArray = gl.createVertexArray();
        if (vertexArray === null) {
            throw new Error("Unable to allocate a vertex array object.");
        }
        this.vertexArray = vertexArray;

        gl.useProgram(this.program);

        this.viewProjectionLocation = gl.getUniformLocation(
            this.program,
            "uViewProjection",
        ) as WebGLUniformLocation;
        this.timeLocation = gl.getUniformLocation(
            this.program,
            "uTime",
        ) as WebGLUniformLocation;
        this.elementCountLocation = gl.getUniformLocation(
            this.program,
            "uElementCount",
        ) as WebGLUniformLocation;

        gl.uniformMatrix4fv(this.viewProjectionLocation, false, viewProjection);
        gl.uniform1i(this.elementCountLocation, ELEMENT_COUNT);

        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.clearColor(0, 0, 0, 1);
    }

    /**
     * Renders one frame at the given elapsed time.
     *
     * @param timeMs - Elapsed time in milliseconds.
     */
    public render(timeMs: number): void {
        const gl = this.gl;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(this.program);
        gl.uniform1f(this.timeLocation, timeMs / 1000);
        gl.bindVertexArray(this.vertexArray);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, ELEMENT_COUNT);
    }
}
