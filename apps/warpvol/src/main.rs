use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::io::BufRead;

/// Apply a 3D linear or warping transform to a volume.
///
/// Reads a 3x4 transform matrix and applies it to the input volume using
/// trilinear interpolation.
#[derive(Parser)]
#[command(name = "warpvol", about = "Apply 3D transform to a volume")]
struct Args {
    /// Input volume (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Output volume (MRC)
    #[arg(short = 'o', long)]
    output: String,

    /// Transform file (3 lines of: m11 m12 m13 tx)
    #[arg(short = 't', long)]
    transform: String,

    /// Output X size (default: same as input)
    #[arg(long)]
    nx: Option<i32>,
    /// Output Y size
    #[arg(long)]
    ny: Option<i32>,
    /// Output Z size
    #[arg(long)]
    nz: Option<i32>,

    /// Fill value for out-of-bounds
    #[arg(short = 'f', long, default_value_t = 0.0)]
    fill: f32,
}

fn main() {
    let args = Args::parse();

    // Read transform: 3x4 matrix (3 rows of 4 values)
    let (m, t) = read_transform(&args.transform);

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let out_nx = args.nx.unwrap_or(h.nx) as usize;
    let out_ny = args.ny.unwrap_or(h.ny) as usize;
    let out_nz = args.nz.unwrap_or(h.nz) as usize;

    eprintln!("warpvol: {} x {} x {} -> {} x {} x {}", in_nx, in_ny, in_nz, out_nx, out_ny, out_nz);

    // Read entire volume
    let mut volume: Vec<Vec<f32>> = Vec::with_capacity(in_nz);
    for z in 0..in_nz {
        volume.push(reader.read_slice_f32(z).unwrap());
    }

    // Invert the transform for inverse mapping
    let (inv_m, inv_t) = invert_affine_3d(&m, &t);

    let in_cx = in_nx as f64 / 2.0;
    let in_cy = in_ny as f64 / 2.0;
    let in_cz = in_nz as f64 / 2.0;
    let out_cx = out_nx as f64 / 2.0;
    let out_cy = out_ny as f64 / 2.0;
    let out_cz = out_nz as f64 / 2.0;

    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, MrcMode::Float);
    out_header.add_label("warpvol: 3D affine transform applied");
    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for oz in 0..out_nz {
        let mut slice = vec![0.0f32; out_nx * out_ny];
        let dz = oz as f64 - out_cz;

        for oy in 0..out_ny {
            let dy = oy as f64 - out_cy;
            for ox in 0..out_nx {
                let dx = ox as f64 - out_cx;

                // Inverse map
                let sx = inv_m[0][0] * dx + inv_m[0][1] * dy + inv_m[0][2] * dz + inv_t[0] + in_cx;
                let sy = inv_m[1][0] * dx + inv_m[1][1] * dy + inv_m[1][2] * dz + inv_t[1] + in_cy;
                let sz = inv_m[2][0] * dx + inv_m[2][1] * dy + inv_m[2][2] * dz + inv_t[2] + in_cz;

                slice[oy * out_nx + ox] = trilinear(&volume, in_nx, in_ny, in_nz, sx, sy, sz, args.fill);
            }
        }

        let (smin, smax, smean) = min_max_mean(&slice);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&slice).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (out_nx * out_ny * out_nz) as f64) as f32).unwrap();
    eprintln!("warpvol: done -> {}", args.output);
}

fn trilinear(vol: &[Vec<f32>], nx: usize, ny: usize, nz: usize, x: f64, y: f64, z: f64, fill: f32) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let z0 = z.floor() as isize;

    if x0 < 0 || x0 + 1 >= nx as isize || y0 < 0 || y0 + 1 >= ny as isize || z0 < 0 || z0 + 1 >= nz as isize {
        return fill;
    }

    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let fz = (z - z0 as f64) as f32;
    let x0 = x0 as usize;
    let y0 = y0 as usize;
    let z0 = z0 as usize;

    let v000 = vol[z0][y0 * nx + x0];
    let v100 = vol[z0][y0 * nx + x0 + 1];
    let v010 = vol[z0][(y0 + 1) * nx + x0];
    let v110 = vol[z0][(y0 + 1) * nx + x0 + 1];
    let v001 = vol[z0 + 1][y0 * nx + x0];
    let v101 = vol[z0 + 1][y0 * nx + x0 + 1];
    let v011 = vol[z0 + 1][(y0 + 1) * nx + x0];
    let v111 = vol[z0 + 1][(y0 + 1) * nx + x0 + 1];

    let c00 = v000 * (1.0 - fx) + v100 * fx;
    let c10 = v010 * (1.0 - fx) + v110 * fx;
    let c01 = v001 * (1.0 - fx) + v101 * fx;
    let c11 = v011 * (1.0 - fx) + v111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

fn read_transform(path: &str) -> ([[f64; 3]; 3], [f64; 3]) {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let mut m = [[0.0f64; 3]; 3];
    let mut t = [0.0f64; 3];

    for (r, line) in reader.lines().enumerate() {
        if r >= 3 { break; }
        let line = line.unwrap();
        let vals: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if vals.len() >= 4 {
            m[r] = [vals[0], vals[1], vals[2]];
            t[r] = vals[3];
        }
    }
    (m, t)
}

fn invert_affine_3d(m: &[[f64; 3]; 3], t: &[f64; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = 1.0 / det;

    let inv_m = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ];

    let inv_t = [
        -(inv_m[0][0] * t[0] + inv_m[0][1] * t[1] + inv_m[0][2] * t[2]),
        -(inv_m[1][0] * t[0] + inv_m[1][1] * t[1] + inv_m[1][2] * t[2]),
        -(inv_m[2][0] * t[0] + inv_m[2][1] * t[1] + inv_m[2][2] * t[2]),
    ];

    (inv_m, inv_t)
}
