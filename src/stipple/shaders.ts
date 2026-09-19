/**
 * WGSL shader source strings for the stippling pipeline.
 *
 * Kept separate from pipeline TypeScript so the shader code is readable
 * and syntax-highlightable as WGSL.
 */

/**
 * Compute shader that ray-marches the SDF scene and writes a per-pixel
 * density value into an `r32float` storage texture.
 *
 * Each invocation owns one pixel (workgroup size 8×8). The density
 * encoding is:
 *   `-1.0` — ray missed the scene (background).
 *   `[0, 1]` — ray hit; value is `1 - Lambert` luminance under a fixed
 *              light direction.
 *
 * A storage texture is used instead of a render-target color attachment
 * because `r32float` is not a renderable format in WebGPU. The same
 * texture is later bound as `texture_2d<f32>` (nearest) for reads.
 */
export const DENSITY_SHADER = /* wgsl */ `
struct CameraRays {
    eye: vec3f,
    half_width: f32,
    forward: vec3f,
    half_height: f32,
    right: vec3f,
    _pad0: f32,
    up: vec3f,
    _pad1: f32,
    resolution: vec2u,
    _pad2: vec2u,
};

@group(0) @binding(0) var<uniform> camera: CameraRays;
@group(0) @binding(1) var density_out: texture_storage_2d<r32float, write>;

const MAX_DIST: f32 = 50.0;
const SURF_EPS: f32 = 0.001;
const MAX_STEPS: i32 = 96;

fn sdSphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdTorus(p: vec3f, t: vec2f) -> f32 {
    let q = vec2f(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn map(p: vec3f) -> f32 {
    let d1 = sdSphere(p - vec3f(-1.2, 0.0, 0.0), 1.5);
    let d2 = sdTorus(p - vec3f(1.2, 0.0, 0.0), vec2f(1.0, 0.35));
    let d3 = sdBox(p - vec3f(0.0, 1.6, 0.0), vec3f(1.0, 0.4, 1.0));
    let d12 = smin(d1, d2, 0.6);
    return smin(d12, d3, 0.6);
}

fn calcNormal(p: vec3f) -> vec3f {
    let e = 0.001;
    return normalize(vec3f(
        map(p + vec3f(e, 0.0, 0.0)) - map(p + vec3f(-e, 0.0, 0.0)),
        map(p + vec3f(0.0, e, 0.0)) - map(p + vec3f(0.0, -e, 0.0)),
        map(p + vec3f(0.0, 0.0, e)) - map(p + vec3f(0.0, 0.0, -e)),
    ));
}

fn rayMarch(ro: vec3f, rd: vec3f) -> f32 {
    var t = 0.0;
    for (var i: i32 = 0; i < MAX_STEPS; i = i + 1) {
        let p = ro + rd * t;
        let d = map(p);
        if (d < SURF_EPS) {
            return t;
        }
        t = t + d;
        if (t > MAX_DIST) {
            break;
        }
    }
    return -1.0;
}

@compute @workgroup_size(8, 8, 1)
fn density_cs(@builtin(global_invocation_id) gid: vec3u) {
    let res = camera.resolution;
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let frag_coord = vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let resolution_f = vec2f(res);
    let uv = frag_coord / resolution_f;
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let rd = normalize(
        camera.forward
        + ndc.x * camera.half_width * camera.right
        + ndc.y * camera.half_height * camera.up
    );

    let t = rayMarch(camera.eye, rd);
    var density: f32;
    if (t < 0.0) {
        density = -1.0;
    } else {
        let p = camera.eye + rd * t;
        let n = calcNormal(p);
        let light_dir = normalize(vec3f(0.5, 0.8, 0.6));
        let lambert = max(dot(n, light_dir), 0.0);
        density = 1.0 - lambert;
    }

    textureStore(density_out, gid.xy, vec4f(density, 0.0, 0.0, 0.0));
}
`;

/**
 * Debug render shader that blits the density texture to the canvas as
 * grayscale.
 *
 * Background (`-1.0`) maps to black; hit values (`[0, 1]`) map to
 * `[0, 1]` gray. Uses `textureLoad` with integer texel coordinates
 * derived from `@builtin(position)`, so no sampler or UV interpolation
 * is needed — the density texture is the same size as the canvas.
 */
export const BLIT_SHADER = /* wgsl */ `
@group(0) @binding(0) var density_in: texture_2d<f32>;

@vertex
fn blit_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    // Two triangles covering [-1, 1] in NDC:
    //   0:(-1,-1)  1:( 1,-1)  2:(-1, 1)
    //   3:(-1, 1)  4:( 1,-1)  5:( 1, 1)
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
fn blit_fs(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let texel = vec2u(frag_coord.xy);
    let d = textureLoad(density_in, texel, 0).r;
    let v = max(d, 0.0);
    return vec4f(v, v, v, 1.0);
}
`;

/**
 * Compute shader for GPU-side rejection sampling of the initial point
 * distribution.
 *
 * One invocation per point index. Uses a PCG hash to generate pseudo-random
 * `(x, y, r)` triples, where `(x, y)` is a candidate position in `[0,1]²`
 * UV space and `r` is compared against the density value at that position.
 * A point is accepted when `r < d` (i.e., with probability `d`). Up to
 * `N_ATTEMPTS` candidates are tried.
 *
 * If all attempts are exhausted without acceptance, the shader writes the
 * last candidate with non-negative density (if any), otherwise the first
 * candidate (even if background). This guarantees a deterministic fallback
 * — relaxation will pull stragglers inside the figure.
 *
 * The integer stride per point is `N_ATTEMPTS * 3` (two floats for the
 * position, one for the comparison value per attempt), ensuring every
 * invocation draws from a disjoint region of the hash sequence.
 */
export const SEED_SHADER = /* wgsl */ `
struct SeedParams {
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var density_in: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> points_out: array<vec4f>;
@group(0) @binding(2) var<uniform> params: SeedParams;

const N_ATTEMPTS: u32 = 256u;

fn pcg_hash(seed: u32) -> u32 {
    let state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_rand(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

@compute @workgroup_size(64)
fn seed_cs(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.point_count) {
        return;
    }

    let base = i * N_ATTEMPTS * 3u;
    let dims = textureDimensions(density_in);

    var accepted = false;
    var temp = vec2f(0.5);
    var last_hit = vec2f(0.5);
    var last_hit_valid = false;

    for (var a: u32 = 0u; a < N_ATTEMPTS; a = a + 1u) {
        let s = base + a * 3u;
        let x = pcg_rand(s + 0u);
        let y = pcg_rand(s + 1u);
        let r = pcg_rand(s + 2u);

        let texel = min(vec2u(vec2f(x, y) * vec2f(dims)), dims - vec2u(1u));
        let d = textureLoad(density_in, texel, 0).r;

        if (a == 0u) {
            temp = vec2f(x, y);
        }

        if (d >= 0.0) {
            last_hit = vec2f(x, y);
            last_hit_valid = true;
            if (r < d) {
                points_out[i] = vec4f(x, y, 0.0, 0.0);
                accepted = true;
                break;
            }
        }
    }

    if (!accepted) {
        if (last_hit_valid) {
            points_out[i] = vec4f(last_hit, 0.0, 0.0);
        } else {
            points_out[i] = vec4f(temp, 0.0, 0.0);
        }
    }
}
`;

/**
 * Render shader for stipple points as screen-space billboard quads.
 *
 * Each point is drawn as a two-triangle quad centered at the point's UV
 * position (mapped to NDC). The quad radius is specified in pixels and
 * converted to NDC using the canvas resolution, producing circular discs
 * regardless of aspect ratio.
 *
 * The fragment shader paints a soft red disc: fragments outside radius
 * 1.0 are discarded; alpha falls off smoothly near the edge. Uses
 * additive blending so overlapping stipples brighten the underlying
 * density visualization.
 */
export const POINT_SHADER = /* wgsl */ `
struct PointUniform {
    resolution: vec2u,
    point_radius_px: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> points: array<vec4f>;
@group(0) @binding(1) var<uniform> u: PointUniform;

struct VertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn point_vs(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    // Two triangles covering [-1, 1]:
    //   0:(-1,-1)  1:( 1,-1)  2:(-1, 1)
    //   3:(-1, 1)  4:( 1,-1)  5:( 1, 1)
    let corner = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );

    let point = points[iid].xy;
    // UV [0,1] → NDC [-1,1], Y flipped (UV y-down → NDC y-up).
    let center = vec2f(point.x * 2.0 - 1.0, 1.0 - point.y * 2.0);

    // Convert pixel radius to NDC half-size.
    let ndc_per_pixel_x = 2.0 / f32(u.resolution.x);
    let ndc_per_pixel_y = 2.0 / f32(u.resolution.y);
    let offset = vec2f(
        corner[vid].x * u.point_radius_px * ndc_per_pixel_x,
        corner[vid].y * u.point_radius_px * ndc_per_pixel_y,
    );

    var out: VertexOut;
    out.clip_pos = vec4f(center + offset, 0.0, 1.0);
    out.uv = corner[vid];
    return out;
}

@fragment
fn point_fs(in: VertexOut) -> @location(0) vec4f {
    let dist = length(in.uv);
    if (dist > 1.0) {
        discard;
    }
    let alpha = smoothstep(1.0, 0.75, dist);
    return vec4f(0.5, 0.1, 0.0, alpha);
}
`;
