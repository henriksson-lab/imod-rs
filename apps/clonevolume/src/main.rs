use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_mrc::{MrcReader, MrcWriter};

/// Clone subvolumes into a target volume at specified positions and orientations.
///
/// Reads an input subvolume and a CSV file of positions/orientations, then
/// stamps copies of the subvolume into a target volume at each location.
/// Supports rotation, alpha blending, radial filtering, and masking.
#[derive(Parser)]
#[command(name = "clonevolume", version, about)]
struct Args {
    /// Input subvolume file to clone.
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Target volume to clone into.
    #[arg(long = "into")]
    into_file: String,

    /// CSV file with positions/orientations (contour,x,y,z,xAngle,yAngle,zAngle).
    #[arg(long = "at")]
    at_points: String,

    /// Output volume file.
    #[arg(short = 'o', long = "output")]
    output: String,

    /// X range to filter positions.
    #[arg(long = "x", num_args = 2)]
    x_range: Vec<f32>,

    /// Y range to filter positions.
    #[arg(long = "y", num_args = 2)]
    y_range: Vec<f32>,

    /// Z range to filter positions.
    #[arg(long = "z", num_args = 2)]
    z_range: Vec<f32>,

    /// Contour numbers to include (comma-separated list).
    #[arg(long = "contours", value_delimiter = ',')]
    contour_list: Vec<i32>,

    /// Alpha transparency (0 = fully opaque replacement, 1 = fully transparent).
    #[arg(long = "alpha", default_value_t = 0.0)]
    alpha: f32,

    /// Mask file (byte mode, non-zero = inside).
    #[arg(long = "mask")]
    mask_file: Option<String>,

    /// Minimum radius from center of input to include.
    #[arg(long = "rmin", default_value_t = 0)]
    r_min: i32,

    /// Maximum radius from center of input to include.
    #[arg(long = "rmax", default_value_t = 32767)]
    r_max: i32,
}

/// 3D bounding box.
#[derive(Debug, Clone, Copy)]
struct BBox3D {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    z_min: f32,
    z_max: f32,
}

/// Clone transform: includes rotation and translation.
#[allow(dead_code)]
struct CloneTransform {
    /// Forward transform matrix (3x3 rotation + translation) stored as 4x4 column-major.
    forward: [f32; 16],
    /// Inverse transform.
    inverse: [f32; 16],
    /// Bounding box after transformation, clipped to target volume.
    bbox: BBox3D,
}

/// Build a 4x4 identity matrix.
fn mat4_identity() -> [f32; 16] {
    let mut m = [0.0_f32; 16];
    m[0] = 1.0; m[5] = 1.0; m[10] = 1.0; m[15] = 1.0;
    m
}

/// Multiply two 4x4 matrices (column-major).
fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0_f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[row + k * 4] * b[k + col * 4];
            }
            r[row + col * 4] = s;
        }
    }
    r
}

/// Create translation matrix.
fn mat4_translate(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
    let mut m = mat4_identity();
    m[12] = tx; m[13] = ty; m[14] = tz;
    m
}

/// Create rotation around Z axis.
fn mat4_rot_z(deg: f32) -> [f32; 16] {
    let r = deg.to_radians();
    let c = r.cos();
    let s = r.sin();
    let mut m = mat4_identity();
    m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
    m
}

/// Create rotation around Y axis.
fn mat4_rot_y(deg: f32) -> [f32; 16] {
    let r = deg.to_radians();
    let c = r.cos();
    let s = r.sin();
    let mut m = mat4_identity();
    m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
    m
}

/// Create rotation around X axis.
fn mat4_rot_x(deg: f32) -> [f32; 16] {
    let r = deg.to_radians();
    let c = r.cos();
    let s = r.sin();
    let mut m = mat4_identity();
    m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
    m
}

/// Transform a 3D point by a 4x4 matrix.
fn mat4_transform(m: &[f32; 16], x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    (
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    )
}

/// Invert a 4x4 matrix (assumes affine with orthogonal rotation).
fn mat4_invert_affine(m: &[f32; 16]) -> [f32; 16] {
    // For an affine matrix M = [R t; 0 1], the inverse is [R^T -R^T*t; 0 1]
    let mut inv = mat4_identity();
    // Transpose the 3x3 rotation part
    inv[0] = m[0]; inv[1] = m[4]; inv[2] = m[8];
    inv[4] = m[1]; inv[5] = m[5]; inv[6] = m[9];
    inv[8] = m[2]; inv[9] = m[6]; inv[10] = m[10];
    // Translation = -R^T * t
    inv[12] = -(inv[0] * m[12] + inv[4] * m[13] + inv[8] * m[14]);
    inv[13] = -(inv[1] * m[12] + inv[5] * m[13] + inv[9] * m[14]);
    inv[14] = -(inv[2] * m[12] + inv[6] * m[13] + inv[10] * m[14]);
    inv
}

/// Trilinear interpolation in a 3D volume.
fn trilinear(vol: &[Vec<f32>], nx: usize, ny: usize, nz: usize,
             x: f32, y: f32, z: f32) -> f32 {
    let ix = x as i32;
    let iy = y as i32;
    let iz = z as i32;

    if ix < 0 || iy < 0 || iz < 0 || ix >= nx as i32 || iy >= ny as i32 || iz >= nz as i32 {
        return 0.0;
    }

    let dx = x - ix as f32;
    let dy = y - iy as f32;
    let dz = z - iz as f32;

    let ix = ix as usize;
    let iy = iy as usize;
    let iz = iz as usize;

    let ixh = (ix + 1).min(nx - 1);
    let iyh = (iy + 1).min(ny - 1);
    let izh = (iz + 1).min(nz - 1);

    let d11 = (1.0 - dx) * (1.0 - dy);
    let d12 = (1.0 - dx) * dy;
    let d21 = dx * (1.0 - dy);
    let d22 = dx * dy;

    let v0 = d11 * vol[iz][ix + iy * nx]
        + d12 * vol[iz][ix + iyh * nx]
        + d21 * vol[iz][ixh + iy * nx]
        + d22 * vol[iz][ixh + iyh * nx];

    let v1 = d11 * vol[izh][ix + iy * nx]
        + d12 * vol[izh][ix + iyh * nx]
        + d21 * vol[izh][ixh + iy * nx]
        + d22 * vol[izh][ixh + iyh * nx];

    v0 * (1.0 - dz) + v1 * dz
}

fn main() {
    let args = Args::parse();

    if args.alpha < 0.0 || args.alpha > 1.0 {
        eprintln!("ERROR: clonevolume - transparency must be between 0 and 1");
        process::exit(1);
    }

    // Open input subvolume
    let mut in_reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: clonevolume - could not open input {}: {}", args.input, e);
        process::exit(1);
    });
    let in_hdr = in_reader.header().clone();
    let in_nx = in_hdr.nx as usize;
    let in_ny = in_hdr.ny as usize;
    let in_nz = in_hdr.nz as usize;

    // Open target volume
    let mut into_reader = MrcReader::open(&args.into_file).unwrap_or_else(|e| {
        eprintln!("ERROR: clonevolume - could not open into file {}: {}", args.into_file, e);
        process::exit(1);
    });
    let into_hdr = into_reader.header().clone();
    let into_nx = into_hdr.nx as usize;
    let into_ny = into_hdr.ny as usize;
    let into_nz = into_hdr.nz as usize;

    // Input volume center and bounding box
    let in_center = Point3f {
        x: in_nx as f32 / 2.0,
        y: in_ny as f32 / 2.0,
        z: (in_nz as f32 - 1.0) / 2.0,
    };

    let in_bbox = BBox3D {
        x_min: 0.0, x_max: in_nx as f32,
        y_min: 0.0, y_max: in_ny as f32,
        z_min: -0.5, z_max: in_nz as f32 - 0.5,
    };

    let into_bbox = BBox3D {
        x_min: 0.0, x_max: into_nx as f32,
        y_min: 0.0, y_max: into_ny as f32,
        z_min: -0.5, z_max: into_nz as f32 - 0.5,
    };

    // Input volume corners for bounding box computation
    let corners: Vec<Point3f> = vec![
        Point3f { x: 0.0, y: 0.0, z: -0.5 },
        Point3f { x: 0.0, y: 0.0, z: in_nz as f32 - 0.5 },
        Point3f { x: 0.0, y: in_ny as f32, z: -0.5 },
        Point3f { x: 0.0, y: in_ny as f32, z: in_nz as f32 - 0.5 },
        Point3f { x: in_nx as f32, y: 0.0, z: -0.5 },
        Point3f { x: in_nx as f32, y: 0.0, z: in_nz as f32 - 0.5 },
        Point3f { x: in_nx as f32, y: in_ny as f32, z: -0.5 },
        Point3f { x: in_nx as f32, y: in_ny as f32, z: in_nz as f32 - 0.5 },
    ];

    // Range filters
    let x_range = if args.x_range.len() == 2 {
        (args.x_range[0], args.x_range[1])
    } else {
        (0.0, f32::MAX)
    };
    let y_range = if args.y_range.len() == 2 {
        (args.y_range[0], args.y_range[1])
    } else {
        (0.0, f32::MAX)
    };
    let z_range = if args.z_range.len() == 2 {
        (args.z_range[0], args.z_range[1])
    } else {
        (-0.5, f32::MAX)
    };

    // Parse the CSV location/orientation file
    let coord_file = File::open(&args.at_points).unwrap_or_else(|e| {
        eprintln!("ERROR: clonevolume - could not open location file {}: {}", args.at_points, e);
        process::exit(1);
    });
    let coord_reader = BufReader::new(coord_file);
    let mut lines = coord_reader.lines();

    // Skip header line
    lines.next();

    let mut clones: Vec<CloneTransform> = Vec::new();

    for line_result in lines {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }

        let contour: i32 = fields[0].trim().parse().unwrap_or(0);
        let x: f32 = fields[1].trim().parse().unwrap_or(0.0);
        let y: f32 = fields[2].trim().parse().unwrap_or(0.0);
        let z: f32 = fields[3].trim().parse().unwrap_or(0.0);
        let x_angle: f32 = fields[4].trim().parse().unwrap_or(0.0);
        let y_angle: f32 = fields[5].trim().parse().unwrap_or(0.0);
        let z_angle: f32 = fields[6].trim().parse().unwrap_or(0.0);

        // Filter by contour list
        if !args.contour_list.is_empty() && !args.contour_list.contains(&contour) {
            continue;
        }

        // Filter by range
        if x < x_range.0 || x > x_range.1
            || y < y_range.0 || y > y_range.1
            || z < z_range.0 || z > z_range.1
        {
            continue;
        }

        // Build forward transform: translate to center, rotate, translate to position
        let t_center = mat4_translate(-in_center.x, -in_center.y, -in_center.z);
        let rz = mat4_rot_z(z_angle);
        let ry = mat4_rot_y(y_angle);
        let rx = mat4_rot_x(x_angle);
        let t_pos = mat4_translate(x, y, z);

        let fwd = mat4_mul(&t_pos, &mat4_mul(&rx, &mat4_mul(&ry, &mat4_mul(&rz, &t_center))));

        // Compute bounding box of transformed corners
        let mut tb = BBox3D {
            x_min: f32::MAX, x_max: f32::MIN,
            y_min: f32::MAX, y_max: f32::MIN,
            z_min: f32::MAX, z_max: f32::MIN,
        };
        for c in &corners {
            let (tx, ty, tz) = mat4_transform(&fwd, c.x, c.y, c.z);
            tb.x_min = tb.x_min.min(tx);
            tb.x_max = tb.x_max.max(tx);
            tb.y_min = tb.y_min.min(ty);
            tb.y_max = tb.y_max.max(ty);
            tb.z_min = tb.z_min.min(tz);
            tb.z_max = tb.z_max.max(tz);
        }

        // Clip to target volume
        let clipped = BBox3D {
            x_min: tb.x_min.max(into_bbox.x_min),
            x_max: tb.x_max.min(into_bbox.x_max),
            y_min: tb.y_min.max(into_bbox.y_min),
            y_max: tb.y_max.min(into_bbox.y_max),
            z_min: tb.z_min.max(into_bbox.z_min),
            z_max: tb.z_max.min(into_bbox.z_max),
        };

        let inv = mat4_invert_affine(&fwd);

        clones.push(CloneTransform {
            forward: fwd,
            inverse: inv,
            bbox: clipped,
        });
    }

    println!("{} clone locations", clones.len());

    // Read input volume into memory
    let mut in_vol: Vec<Vec<f32>> = Vec::with_capacity(in_nz);
    for iz in 0..in_nz {
        let slice = in_reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: clonevolume - error reading input slice {}: {}", iz, e);
            process::exit(1);
        });
        in_vol.push(slice);
    }

    // Read optional mask volume
    let mask_vol: Option<Vec<Vec<u8>>> = if let Some(ref mask_path) = args.mask_file {
        let mut mask_reader = MrcReader::open(mask_path).unwrap_or_else(|e| {
            eprintln!("ERROR: clonevolume - could not open mask file {}: {}", mask_path, e);
            process::exit(1);
        });
        let mask_hdr = mask_reader.header();
        if mask_hdr.nx != in_hdr.nx || mask_hdr.ny != in_hdr.ny || mask_hdr.nz != in_hdr.nz {
            eprintln!("ERROR: clonevolume - input and mask volumes must be the same size");
            process::exit(1);
        }
        let mut vols = Vec::with_capacity(in_nz);
        for iz in 0..in_nz {
            let raw = mask_reader.read_slice_raw(iz).unwrap_or_else(|e| {
                eprintln!("ERROR: clonevolume - error reading mask slice: {}", e);
                process::exit(1);
            });
            vols.push(raw);
        }
        Some(vols)
    } else {
        None
    };

    // Create output writer from target header
    let out_hdr = into_hdr.clone();
    let mut writer = MrcWriter::create(&args.output, out_hdr).unwrap_or_else(|e| {
        eprintln!("ERROR: clonevolume - could not create output file: {}", e);
        process::exit(1);
    });

    let mut min_gray = f32::MAX;
    let mut max_gray = f32::MIN;
    let mut _mean_gray = 0.0_f64;
    let r_min = args.r_min as f32;
    let r_max = args.r_max as f32;
    let alpha = args.alpha;

    // Process slice by slice
    for iz in 0..into_nz {
        let mut slice = into_reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: clonevolume - error reading into slice {}: {}", iz, e);
            process::exit(1);
        });

        // Apply all clones that intersect this slice
        for clone in &clones {
            if (iz as f32) < clone.bbox.z_min || (iz as f32) > clone.bbox.z_max {
                continue;
            }

            let ix_start = clone.bbox.x_min.ceil() as i32;
            let ix_end = clone.bbox.x_max.floor() as i32;
            let iy_start = clone.bbox.y_min.ceil() as i32;
            let iy_end = clone.bbox.y_max.floor() as i32;

            for iy in iy_start..iy_end.min(into_ny as i32) {
                if iy < 0 || iy >= into_ny as i32 {
                    continue;
                }
                for ix in ix_start..ix_end.min(into_nx as i32) {
                    if ix < 0 || ix >= into_nx as i32 {
                        continue;
                    }

                    // Transform output coords back to input space
                    let out_x = ix as f32 + 0.5;
                    let out_y = iy as f32 + 0.5;
                    let out_z = iz as f32;

                    let (in_x, in_y, in_z) = mat4_transform(&clone.inverse, out_x, out_y, out_z);

                    // Check if inside input bounding box
                    let tol = 1.0e-4;
                    if in_x < in_bbox.x_min + tol || in_x > in_bbox.x_max - tol
                        || in_y < in_bbox.y_min + tol || in_y > in_bbox.y_max - tol
                        || in_z < in_bbox.z_min + tol || in_z > in_bbox.z_max - tol
                    {
                        continue;
                    }

                    // Radius check
                    if r_min > 0.0 || r_max < 32767.0 {
                        let r = ((in_x - in_center.x).powi(2)
                            + (in_y - in_center.y).powi(2)
                            + (in_z - in_center.z).powi(2))
                        .sqrt();
                        if r < r_min || r > r_max {
                            continue;
                        }
                    }

                    // Mask check
                    if let Some(ref mask) = mask_vol {
                        let mz = (in_z + 0.5) as usize;
                        let mx = (in_x + 0.5) as usize;
                        let my = (in_y + 0.5) as usize;
                        if mz < in_nz && mx < in_nx && my < in_ny {
                            if mask[mz][mx + in_nx * my] == 0 {
                                continue;
                            }
                        }
                    }

                    // Trilinear interpolation
                    let in_val = trilinear(
                        &in_vol, in_nx, in_ny, in_nz,
                        in_x - 0.5, in_y - 0.5, in_z,
                    );

                    let idx = ix as usize + iy as usize * into_nx;
                    slice[idx] = alpha * slice[idx] + (1.0 - alpha) * in_val;
                }
            }
        }

        // Compute statistics
        for &v in &slice {
            min_gray = min_gray.min(v);
            max_gray = max_gray.max(v);
            _mean_gray += v as f64;
        }

        eprint!("\rWriting slice {}...", iz);
        writer.write_slice_f32(&slice).unwrap_or_else(|e| {
            eprintln!("\nERROR: clonevolume - error writing output slice: {}", e);
            process::exit(1);
        });
    }

    eprintln!();
    println!("Finished!");
}
