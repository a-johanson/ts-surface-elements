/**
 * GLSL ES 3.00 shader sources for the surface-element renderer.
 *
 * A single `drawArraysInstanced` call with an empty vertex array renders
 * `uElementCount` flat quads lying on the surface of an organic shape.
 * Quad placement, orientation, and per-element shading are derived entirely
 * procedurally from `gl_VertexID`, `gl_InstanceID`, `uTime`,
 * `uViewProjection`, and `uElementCount` — no vertex buffers are used. The
 * fragment shader draws each element with anti-aliasing and discards
 * fragments outside the element.
 */

export const ELEMENT_VERT = `#version 300 es
precision highp float;

uniform mat4 uViewProjection;
uniform float uTime;
uniform int uElementCount;

out vec2 vUv;
out float vLightness;

const float GOLDEN_ANGLE = 2.39996323;
const float QUAD_HALF_SIZE = 0.015;

vec2 quadCorner(int vid) {
    return vec2(
        float(vid & 1) * 2.0 - 1.0,
        float((vid >> 1) & 1) * 2.0 - 1.0
    );
}

vec3 fibonacciPoint(int i, int n) {
    float phi = float(i) * GOLDEN_ANGLE;
    float y = 1.0 - 2.0 * (float(i) + 0.5) / float(n);
    float r = sqrt(max(1.0 - y * y, 0.0));
    return vec3(cos(phi) * r, y, sin(phi) * r);
}

mat3 rotationAxis(vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    float t = 1.0 - c;
    vec3 n = normalize(axis);
    return mat3(
        t * n.x * n.x + c,       t * n.x * n.y - s * n.z, t * n.x * n.z + s * n.y,
        t * n.x * n.y + s * n.z, t * n.y * n.y + c,       t * n.y * n.z - s * n.x,
        t * n.x * n.z - s * n.y, t * n.y * n.z + s * n.x, t * n.z * n.z + c
    );
}

void main() {
    vec2 uv = quadCorner(gl_VertexID);

    vec3 p = fibonacciPoint(gl_InstanceID, uElementCount);

    vec3 spinAxis = normalize(vec3(0.2, 1.0, 0.3));
    mat3 spin = rotationAxis(spinAxis, uTime * 0.00003);
    vec3 center = spin * p;

    vec3 N = normalize(center);

    vec3 ref = abs(N.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(ref, N));
    vec3 bitangent = cross(N, tangent);

    float lt = uTime * 0.5;
    vec3 L = normalize(vec3(cos(lt), sin(lt), 0.6));
    float diffuse = max(dot(N, L), 0.0);
    float lightness = 0.25 + 0.75 * diffuse;

    vUv = uv;
    vLightness = lightness;
    vec3 worldPos = center + (tangent * uv.x + bitangent * uv.y) * QUAD_HALF_SIZE;

    gl_Position = uViewProjection * vec4(worldPos, 1.0);
}
`;

export const ELEMENT_FRAG = `#version 300 es
precision highp float;

in vec2 vUv;
in float vLightness;
out vec4 fragColor;

const float LINE_HALF = 0.2;
const float ELEMENT_SCALE = 1.0 - LINE_HALF;

void main() {
    float dist = length(vUv);
    float sd = dist - vLightness * ELEMENT_SCALE;
    float w = fwidth(dist);
    //float alpha = 1.0 - smoothstep(LINE_HALF - w, LINE_HALF + w, abs(sd));
    float alpha = clamp((LINE_HALF - abs(sd)) / w + 0.5, 0.0, 1.0);
    if (alpha <= 0.0) discard;
    fragColor = vec4(1.0, 1.0, 1.0, alpha);
}
`;
