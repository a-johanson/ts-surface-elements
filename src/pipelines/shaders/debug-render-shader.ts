/**
 * Debug render shader — ray-marches the time-animated SDF per fragment and
 * writes a 1D periodic line pattern evaluated on the surface directly to the
 * canvas color attachment.
 *
 * The signal's phase is the projection of world position onto the
 * (animated) light direction, so its iso-lines are slices orthogonal to
 * the light. Its wavelength is inversely related to the Lambert luminance
 * modulated by the soft-shadow penumbra — darker fragments show denser
 * rings.
 *
 * Lines are drawn black on white with analytic anti-aliasing at a constant
 * screen-space thickness; where the implied line frequency becomes too high
 * to resolve, the region falls back to solid black instead of forming a
 * Moire pattern. Each ring is broken into dashes by thresholding a noise
 * field in object space that is decorrelated between rings.
 */
import { NOISE_COMMON } from "./noise.js";
import { SDF_COMMON } from "./sdf-common.js";

export const DEBUG_RENDER_SHADER = /* wgsl */ `
${SDF_COMMON}
${NOISE_COMMON}

struct CameraRays {
    eye: vec3f,
    half_width: f32,
    forward: vec3f,
    half_height: f32,
    right: vec3f,
    _pad0: f32,
    up: vec3f,
    time: f32,
    resolution: vec2u,
    _pad2: vec2u,
};

@group(0) @binding(0) var<uniform> camera: CameraRays;

@vertex
fn debug_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );
    return vec4f(pos[vid], 0.0, 1.0);
}

@fragment
fn debug_fs(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let res = camera.resolution;
    let uv = frag_coord.xy / vec2f(res);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let rd = normalize(
        camera.forward
        + ndc.x * camera.half_width * camera.right
        + ndc.y * camera.half_height * camera.up
    );

    let t = rayMarch(camera.eye, rd, camera.time);
    // fwidth must be called from uniform control flow, so there is no early
    // return for misses; their pattern result is discarded by the final select.
    let p = camera.eye + rd * max(t, 0.0);
    let n = calcNormal(p, camera.time);
    let light_dir = lightDir(camera.time);
    let ndotl = dot(n, light_dir);
    let lambert = max(ndotl, 0.0);
    let shadow = softShadow(p + n * SHADOW_BIAS, light_dir, camera.time);
    let lum = lambert * shadow;

    const PATTERN_WAVELENGTH_MIN: f32 = 0.05;
    const PATTERN_WAVELENGTH_MAX: f32 = 0.15;
    // const PATTERN_POLE_BLEND: f32 = 1.0;
    const HALF_LINE_WIDTH_PX: f32 = 0.5;
    const MOIRE_SPACING_PX: f32 = 2.0;
    const AA_FILTER_PX: f32 = 0.5;
    const BREAK_FREQUENCY: f32 = 4.0;
    const BREAK_THRESHOLD: f32 = 0.39;

    let wavelength = mix(PATTERN_WAVELENGTH_MIN, PATTERN_WAVELENGTH_MAX, lum);
    // let theta = acos(clamp(ndotl, -1.0, 1.0));
    // let mask = smoothstep(-0.7, -1.0, ndotl);
    // let phase = (dot(p, light_dir) + PATTERN_POLE_BLEND * mask * theta) / wavelength;
    // Phase in cycles (period 1). Continuous in screen space, so its screen-space
    // derivative carries the local line frequency.
    let phase = dot(p, light_dir) / wavelength;
    let freq = max(fwidth(phase), 1e-6);

    // Break noise: the ring index is an integer on the fourth axis, which
    // selects an independent noise layer per ring. Computed before the early
    // return because its screen-space derivative needs uniform control flow.
    let break_noise = fbm4(vec4f(p * BREAK_FREQUENCY, floor(phase)));
    let break_rate = max(fwidth(break_noise), 1e-5);

    if t < 0.0 {
        return vec4f(1.0);
    }

    // Distance to the nearest line (at cycle offset 0.5), linear in phase, so
    // multiplying by spacing_px yields the distance in screen-space pixels.
    let spacing_px = 1.0 / freq;
    let dist_px = abs(fract(phase) - 0.5) * spacing_px;

    // Constant screen-space line thickness, anti-aliased over a one-pixel filter.
    let line_shade = smoothstep(HALF_LINE_WIDTH_PX - AA_FILTER_PX, HALF_LINE_WIDTH_PX + AA_FILTER_PX, dist_px);

    // Break rings into dashes. Dividing the noise excess over the threshold by
    // its screen-space rate yields the signed distance in pixels to the gap
    // edge (positive inside a dash), anti-aliased over the same one-pixel
    // filter as the line. Gaps show white.
    let break_dist_px = (break_noise - BREAK_THRESHOLD) / break_rate;
    let keep = smoothstep(-AA_FILTER_PX, AA_FILTER_PX, break_dist_px);
    let shade = mix(1.0, line_shade, keep);

    // Regions whose implied line frequency is too high to resolve fall back to
    // solid black instead of aliasing into a Moire pattern.
    let moire_blend = smoothstep(MOIRE_SPACING_PX, MOIRE_SPACING_PX + 0.5, spacing_px);
    let pattern = mix(0.0, shade, moire_blend);
    return vec4f(vec3f(pattern), 1.0);
}
`;
