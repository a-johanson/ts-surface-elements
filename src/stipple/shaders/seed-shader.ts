/**
 * Seed compute shader — 3D rejection sampling in a bounding box.
 *
 * One invocation per point. Generates pseudo-random candidates uniformly
 * in `[bbox_min, bbox_max]³` via a PCG hash and accepts the first whose
 * `|map(p, 0)| < band`. Accepted (and fallback) points are projected onto
 * the surface at t=0 with Newton-Raphson before being written.
 *
 * Seeding is a one-shot bootstrap at t=0 — the SDF is evaluated at its
 * initial configuration. Subsequent surface motion is handled per-frame by
 * the reproject pass.
 *
 * In addition to positions, the seed pass writes the matching surface
 * normal for each point into the shared `normals_out` buffer. The normals
 * buffer is shared (not ping-ponged) and must match whichever point buffer
 * relax reads; since seed writes bufferA, it also seeds the normals from
 * bufferA. Subsequent frames have their normals refreshed by the reproject
 * pass (before relax) and the shading pass (after relax).
 */
import { SDF_COMMON } from "./sdf-common.js";

export const SEED_SHADER = /* wgsl */ `
${SDF_COMMON}

struct SeedParams {
    bbox_min: vec3f,
    band: f32,
    bbox_max: vec3f,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SeedParams;
@group(0) @binding(1) var<storage, read_write> points_out: array<Point>;
@group(0) @binding(2) var<storage, read_write> normals_out: array<vec4f>;

const N_ATTEMPTS: u32 = 256u;
const SEED_NEWTON_ITERS: i32 = 16;
const SEED_ALPHA: f32 = 0.5;
const SEED_TIME: f32 = 0.0;

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
    var accepted = false;
    var best_p = (params.bbox_min + params.bbox_max) * 0.5;
    var best_d = 1e9;

    for (var a: u32 = 0u; a < N_ATTEMPTS; a = a + 1u) {
        let s = base + a * 3u;
        let rx = pcg_rand(s + 0u);
        let ry = pcg_rand(s + 1u);
        let rz = pcg_rand(s + 2u);
        let p = mix(params.bbox_min, params.bbox_max, vec3f(rx, ry, rz));
        let d = abs(map(p, SEED_TIME));

        if (d < best_d) {
            best_d = d;
            best_p = p;
        }

        if (d < params.band) {
            let projected = projectToSurface(p, SEED_TIME, SEED_NEWTON_ITERS, SEED_ALPHA);
            points_out[i].pos = vec4f(projected, 0.0);
            normals_out[i] = vec4f(normalize(sdfGradient(projected, SEED_TIME)), 0.0);
            accepted = true;
            break;
        }
    }

    if (!accepted) {
        let projected = projectToSurface(best_p, SEED_TIME, SEED_NEWTON_ITERS, SEED_ALPHA);
        points_out[i].pos = vec4f(projected, 0.0);
        normals_out[i] = vec4f(normalize(sdfGradient(projected, SEED_TIME)), 0.0);
    }
}
`;
