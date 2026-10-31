/**
 * Shared WGSL block: 4D integer hash, smooth value noise, and fractal sum.
 *
 * Interpolated into shaders that need cheap procedural noise. Contains only
 * functions and compile-time constants — no bindings.
 *
 * - `hash4(i)` — maps an integer lattice point to a value in [0, 1).
 * - `valueNoise4(x)` — smooth value noise in [0, 1], interpolating hashed
 *   lattice corners with a quintic fade.
 * - `fbm4(x)` — normalized sum of octaves of `valueNoise4`, in [0, 1].
 */
export const NOISE_COMMON = /* wgsl */ `
const FBM_OCTAVES: i32 = 2;
const FBM_LACUNARITY: f32 = 2.0;
const FBM_GAIN: f32 = 0.5;

fn hash4(i: vec4i) -> f32 {
    let h = bitcast<vec4u>(i) * vec4u(2654435761u, 2246822519u, 3266489917u, 668265263u);
    var s = h.x ^ h.y ^ h.z ^ h.w;
    s = s * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    s = (s >> 22u) ^ s;
    return f32(s) * (1.0 / 4294967296.0);
}

fn valueNoise4(x: vec4f) -> f32 {
    let cell = floor(x);
    let f = x - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let c = vec4i(cell);

    var acc = 0.0;
    for (var k = 0; k < 16; k++) {
        let o = vec4i(k & 1, (k >> 1) & 1, (k >> 2) & 1, (k >> 3) & 1);
        let w = select(1.0 - u, u, o == vec4i(1));
        acc += hash4(c + o) * w.x * w.y * w.z * w.w;
    }
    return acc;
}

fn fbm4(x: vec4f) -> f32 {
    var sum = 0.0;
    var norm = 0.0;
    var amp = 1.0;
    var p = x;
    for (var i = 0; i < FBM_OCTAVES; i++) {
        sum += amp * valueNoise4(p);
        norm += amp;
        amp *= FBM_GAIN;
        p *= FBM_LACUNARITY;
    }
    return sum / norm;
}
`;
