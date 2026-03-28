use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::fs;
use std::io::{BufRead, BufReader};

/// Transform a volume using a general 3D linear transformation.
///
/// Main use: transform one tomogram from a two-axis tilt series to match the other.
/// Combines an initial alignment transformation and any number of successive
/// refining transformations. Uses trilinear interpolation by default.
#[derive(Parser)]
#[command(name = "matchvol", about = "3D linear transformation of a volume")]
struct Args {
    /// Input MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Transform file(s) - each contains a 3x4 matrix (3 rows of: a11 a12 a13 dx)
    #[arg(short = 'x', long = "xffile", required = true)]
    transform_files: Vec<String>,

    /// Output file for inverse transformation
    #[arg(long = "inverse")]
    inverse_file: Option<String>,

    /// Output volume size X,Y,Z (default: same as input)
    #[arg(short = 's', long, num_args = 3, value_delimiter = ',')]
    size: Option<Vec<usize>>,

    /// Center of transformation X,Y,Z (default: center of input volume)
    #[arg(short = 'c', long, num_args = 3, value_delimiter = ',')]
    center: Option<Vec<f32>>,

    /// Interpolation order: 1=linear, 2=quadratic (default: 2)
    #[arg(long, default_value_t = 2)]
    order: usize,
}

/// A 3x3 matrix with a 3-element translation vector.
#[derive(Clone, Debug)]
struct Transform3D {
    a: [[f64; 3]; 3],
    d: [f64; 3],
}

impl Transform3D {
    fn identity() -> Self {
        Self {
            a: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            d: [0.0; 3],
        }
    }

    /// Read a 3x4 transform from a file (3 rows, each: a11 a12 a13 dx).
    fn read_from_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Cannot read transform file '{}': {}", path, e))?;
        let reader = BufReader::new(content.as_bytes());
        let mut a = [[0.0f64; 3]; 3];
        let mut d = [0.0f64; 3];
        let mut row = 0;
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let vals: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() >= 4 && row < 3 {
                a[row][0] = vals[0];
                a[row][1] = vals[1];
                a[row][2] = vals[2];
                d[row] = vals[3];
                row += 1;
            }
        }
        if row != 3 {
            return Err(format!(
                "Transform file '{}' must contain 3 rows of 4 values",
                path
            ));
        }
        Ok(Self { a, d })
    }

    /// Multiply: result = self * other (self applied after other).
    fn multiply(&self, other: &Transform3D) -> Transform3D {
        let mut a = [[0.0f64; 3]; 3];
        let mut d = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = self.a[i][0] * other.a[0][j]
                    + self.a[i][1] * other.a[1][j]
                    + self.a[i][2] * other.a[2][j];
            }
            d[i] = self.a[i][0] * other.d[0]
                + self.a[i][1] * other.d[1]
                + self.a[i][2] * other.d[2]
                + self.d[i];
        }
        Transform3D { a, d }
    }

    /// Compute inverse transform.
    fn inverse(&self) -> Transform3D {
        // Invert the 3x3 matrix using cofactors
        let m = &self.a;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < 1e-15 {
            eprintln!("WARNING: Transform matrix is singular or near-singular");
        }
        let inv_det = 1.0 / det;
        let mut ai = [[0.0f64; 3]; 3];
        ai[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        ai[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        ai[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        ai[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        ai[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        ai[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
        ai[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        ai[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        ai[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

        let mut di = [0.0f64; 3];
        for i in 0..3 {
            di[i] = -(ai[i][0] * self.d[0] + ai[i][1] * self.d[1] + ai[i][2] * self.d[2]);
        }
        Transform3D { a: ai, d: di }
    }

    /// Write transform to file.
    fn write_to_file(&self, path: &str) -> Result<(), String> {
        let mut s = String::new();
        for i in 0..3 {
            s.push_str(&format!(
                "{:10.6} {:10.6} {:10.6} {:10.3}\n",
                self.a[i][0], self.a[i][1], self.a[i][2], self.d[i]
            ));
        }
        fs::write(path, &s).map_err(|e| format!("Cannot write inverse file: {}", e))
    }
}

/// Trilinear interpolation in a 3D volume.
fn interp_trilinear(vol: &[f32], nx: usize, ny: usize, nz: usize, x: f64, y: f64, z: f64) -> f32 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let iz = z.floor() as i64;

    if ix < 0 || ix + 1 >= nx as i64 || iy < 0 || iy + 1 >= ny as i64 || iz < 0 || iz + 1 >= nz as i64
    {
        return 0.0;
    }

    let fx = (x - ix as f64) as f32;
    let fy = (y - iy as f64) as f32;
    let fz = (z - iz as f64) as f32;
    let ix = ix as usize;
    let iy = iy as usize;
    let iz = iz as usize;

    let idx = |x: usize, y: usize, z: usize| -> usize { z * ny * nx + y * nx + x };

    let v000 = vol[idx(ix, iy, iz)];
    let v100 = vol[idx(ix + 1, iy, iz)];
    let v010 = vol[idx(ix, iy + 1, iz)];
    let v110 = vol[idx(ix + 1, iy + 1, iz)];
    let v001 = vol[idx(ix, iy, iz + 1)];
    let v101 = vol[idx(ix + 1, iy, iz + 1)];
    let v011 = vol[idx(ix, iy + 1, iz + 1)];
    let v111 = vol[idx(ix + 1, iy + 1, iz + 1)];

    let c00 = v000 * (1.0 - fx) + v100 * fx;
    let c10 = v010 * (1.0 - fx) + v110 * fx;
    let c01 = v001 * (1.0 - fx) + v101 * fx;
    let c11 = v011 * (1.0 - fx) + v111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: MATCHVOL - opening input: {e}");
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let (out_nx, out_ny, out_nz) = if let Some(ref s) = args.size {
        (s[0], s[1], s[2])
    } else {
        (in_nx, in_ny, in_nz)
    };

    let center = if let Some(ref c) = args.center {
        [c[0] as f64, c[1] as f64, c[2] as f64]
    } else {
        [in_nx as f64 / 2.0, in_ny as f64 / 2.0, in_nz as f64 / 2.0]
    };

    // Read and compose transforms
    let mut combined = Transform3D::identity();
    for (i, path) in args.transform_files.iter().enumerate() {
        let xf = Transform3D::read_from_file(path).unwrap_or_else(|e| {
            eprintln!("ERROR: MATCHVOL - {e}");
            std::process::exit(1);
        });
        if i == 0 {
            combined = xf;
        } else {
            combined = combined.multiply(&xf);
        }
    }

    eprintln!("Forward matrix:");
    for i in 0..3 {
        eprintln!(
            "  {:10.6} {:10.6} {:10.6} {:10.3}",
            combined.a[i][0], combined.a[i][1], combined.a[i][2], combined.d[i]
        );
    }

    let inv = combined.inverse();
    eprintln!("Inverse matrix:");
    for i in 0..3 {
        eprintln!(
            "  {:10.6} {:10.6} {:10.6} {:10.3}",
            inv.a[i][0], inv.a[i][1], inv.a[i][2], inv.d[i]
        );
    }

    if let Some(ref inv_path) = args.inverse_file {
        inv.write_to_file(inv_path).unwrap_or_else(|e| {
            eprintln!("ERROR: MATCHVOL - {e}");
            std::process::exit(1);
        });
    }

    // Read entire input volume
    eprintln!("Reading input volume {}x{}x{}...", in_nx, in_ny, in_nz);
    let mut volume = vec![0.0f32; in_nx * in_ny * in_nz];
    for z in 0..in_nz {
        let slice = reader.read_slice_f32(z).unwrap();
        volume[z * in_ny * in_nx..(z + 1) * in_ny * in_nx].copy_from_slice(&slice);
    }

    // Output center
    let out_center = [out_nx as f64 / 2.0, out_ny as f64 / 2.0, out_nz as f64 / 2.0];

    // Set up output header
    let mut out_header =
        MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, MrcMode::Float);
    // Preserve pixel spacing
    if h.mx > 0 {
        let pixel_x = h.xlen / h.mx as f32;
        let pixel_y = h.ylen / h.my as f32;
        let pixel_z = h.zlen / h.mz as f32;
        out_header.xlen = out_nx as f32 * pixel_x;
        out_header.ylen = out_ny as f32 * pixel_y;
        out_header.zlen = out_nz as f32 * pixel_z;
    }
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;
    out_header.add_label("matchvol: 3-D transformation of tomogram");

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    // Transform volume slice by slice using inverse mapping
    eprintln!("Transforming volume to {}x{}x{}...", out_nx, out_ny, out_nz);
    for oz in 0..out_nz {
        let mut out_slice = vec![0.0f32; out_nx * out_ny];

        for oy in 0..out_ny {
            for ox in 0..out_nx {
                // Position relative to output center
                let px = ox as f64 - out_center[0];
                let py = oy as f64 - out_center[1];
                let pz = oz as f64 - out_center[2];

                // Apply inverse transform to find source position
                let sx = inv.a[0][0] * px + inv.a[0][1] * py + inv.a[0][2] * pz + inv.d[0];
                let sy = inv.a[1][0] * px + inv.a[1][1] * py + inv.a[1][2] * pz + inv.d[1];
                let sz = inv.a[2][0] * px + inv.a[2][1] * py + inv.a[2][2] * pz + inv.d[2];

                // Shift back to input volume coordinates
                let sx = sx + center[0];
                let sy = sy + center[1];
                let sz = sz + center[2];

                out_slice[oy * out_nx + ox] =
                    interp_trilinear(&volume, in_nx, in_ny, in_nz, sx, sy, sz);
            }
        }

        let (smin, smax, smean) = min_max_mean(&out_slice);
        gmin = gmin.min(smin);
        gmax = gmax.max(smax);
        gsum += smean as f64 * out_slice.len() as f64;

        writer.write_slice_f32(&out_slice).unwrap();
    }

    let total = (out_nx * out_ny * out_nz) as f64;
    let gmean = (gsum / total) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();

    eprintln!(
        "Done. Min={:.4} Max={:.4} Mean={:.4}",
        gmin, gmax, gmean
    );
}
