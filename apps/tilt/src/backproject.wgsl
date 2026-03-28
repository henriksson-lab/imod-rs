// Back-projection compute shader.
// Each invocation handles one (ox, oz) pixel in the output XZ slice,
// accumulating contributions from all projections via bilinear interpolation.

struct Params {
    nx: u32,            // output width
    nz: u32,            // output depth (thickness)
    n_projs: u32,       // number of projections
    in_nx: u32,         // input projection width
    in_ny: u32,         // input projection height (number of Y rows)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    center_x: f32,      // in_nx / 2.0
    center_z: f32,      // out_nz / 2.0
    out_center_x: f32,  // out_nx / 2.0
    inv_n: f32,          // 1.0 / n_projs
};

@group(0) @binding(0) var<storage, read> projections: array<f32>;
@group(0) @binding(1) var<storage, read> tilt_params: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> row_y: u32;

@compute @workgroup_size(16, 16)
fn backproject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oz = gid.y;

    if ox >= params.nx || oz >= params.nz {
        return;
    }

    let dx = f32(ox) - params.out_center_x;
    let dz = f32(oz) - params.center_z;
    let iy = row_y;
    let in_nx = params.in_nx;
    let in_ny = params.in_ny;

    var acc: f32 = 0.0;

    for (var pi: u32 = 0u; pi < params.n_projs; pi = pi + 1u) {
        let cs = tilt_params[pi];
        let cos_t = cs.x;
        let sin_t = cs.y;

        let proj_x = dx * cos_t + dz * sin_t + params.center_x;

        let px0 = i32(floor(proj_x));
        if px0 >= 0 && px0 + 1 < i32(in_nx) {
            let frac = proj_x - f32(px0);
            // projections layout: [n_projs][in_ny][in_nx] flattened
            let idx = pi * in_nx * in_ny + iy * in_nx + u32(px0);
            let v0 = projections[idx];
            let v1 = projections[idx + 1u];
            acc += v0 * (1.0 - frac) + v1 * frac;
        }
    }

    acc *= params.inv_n;
    let out_idx = oz * params.nx + ox;
    output[out_idx] = acc;
}
