use std::process;

use clap::Parser;
use imod_model::{read_model, write_model};

/// Transform IMOD model files by applying translation, rotation, and/or scaling.
///
/// Reads a .mod file, applies the specified geometric transformations to all
/// points (contours and meshes), and writes the result. Transformations are
/// applied in the order: scale, rotate (Z then Y then X), translate.
#[derive(Parser)]
#[command(name = "imodtrans", version, about)]
struct Args {
    /// Translate in X
    #[arg(long = "tx", default_value_t = 0.0)]
    tx: f32,

    /// Translate in Y
    #[arg(long = "ty", default_value_t = 0.0)]
    ty: f32,

    /// Translate in Z
    #[arg(long = "tz", default_value_t = 0.0)]
    tz: f32,

    /// Scale in X
    #[arg(long = "sx", default_value_t = 1.0)]
    sx: f32,

    /// Scale in Y
    #[arg(long = "sy", default_value_t = 1.0)]
    sy: f32,

    /// Scale in Z
    #[arg(long = "sz", default_value_t = 1.0)]
    sz: f32,

    /// Rotate around X axis (degrees)
    #[arg(long = "rx", default_value_t = 0.0)]
    rx: f32,

    /// Rotate around Y axis (degrees)
    #[arg(long = "ry", default_value_t = 0.0)]
    ry: f32,

    /// Rotate around Z axis (degrees)
    #[arg(long = "rz", default_value_t = 0.0)]
    rz: f32,

    /// Input IMOD model file
    input: String,

    /// Output IMOD model file
    output: String,
}

/// Build a 3x3 rotation matrix from Euler angles (degrees).
/// Order: Rz * Ry * Rx (rotate around X first, then Y, then Z).
fn rotation_matrix(rx_deg: f32, ry_deg: f32, rz_deg: f32) -> [[f64; 3]; 3] {
    let rx = (rx_deg as f64).to_radians();
    let ry = (ry_deg as f64).to_radians();
    let rz = (rz_deg as f64).to_radians();

    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();

    // Rz * Ry * Rx
    [
        [
            cy * cz,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
        ],
        [
            cy * sz,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
        ],
        [-sy, cy * sx, cy * cx],
    ]
}

fn transform_point(
    x: f32,
    y: f32,
    z: f32,
    scale: [f32; 3],
    rot: &[[f64; 3]; 3],
    translate: [f32; 3],
) -> (f32, f32, f32) {
    // 1. Scale
    let sx = x as f64 * scale[0] as f64;
    let sy = y as f64 * scale[1] as f64;
    let sz = z as f64 * scale[2] as f64;

    // 2. Rotate
    let rx = rot[0][0] * sx + rot[0][1] * sy + rot[0][2] * sz;
    let ry = rot[1][0] * sx + rot[1][1] * sy + rot[1][2] * sz;
    let rz = rot[2][0] * sx + rot[2][1] * sy + rot[2][2] * sz;

    // 3. Translate
    (
        (rx + translate[0] as f64) as f32,
        (ry + translate[1] as f64) as f32,
        (rz + translate[2] as f64) as f32,
    )
}

fn main() {
    let args = Args::parse();

    let mut model = match read_model(&args.input) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: imodtrans - reading {}: {}", args.input, e);
            process::exit(1);
        }
    };

    let scale = [args.sx, args.sy, args.sz];
    let translate = [args.tx, args.ty, args.tz];
    let rot = rotation_matrix(args.rx, args.ry, args.rz);

    let has_transform = args.tx != 0.0
        || args.ty != 0.0
        || args.tz != 0.0
        || args.sx != 1.0
        || args.sy != 1.0
        || args.sz != 1.0
        || args.rx != 0.0
        || args.ry != 0.0
        || args.rz != 0.0;

    if has_transform {
        for obj in &mut model.objects {
            // Transform contour points
            for cont in &mut obj.contours {
                for pt in &mut cont.points {
                    let (nx, ny, nz) = transform_point(pt.x, pt.y, pt.z, scale, &rot, translate);
                    pt.x = nx;
                    pt.y = ny;
                    pt.z = nz;
                }
            }

            // Transform mesh vertices
            for mesh in &mut obj.meshes {
                for pt in &mut mesh.vertices {
                    let (nx, ny, nz) = transform_point(pt.x, pt.y, pt.z, scale, &rot, translate);
                    pt.x = nx;
                    pt.y = ny;
                    pt.z = nz;
                }
            }
        }
    }

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: imodtrans - writing {}: {}", args.output, e);
        process::exit(1);
    }

    if has_transform {
        eprintln!(
            "Transformed model with scale=({},{},{}), rotate=({},{},{}), translate=({},{},{}) -> {}",
            args.sx, args.sy, args.sz, args.rx, args.ry, args.rz, args.tx, args.ty, args.tz,
            args.output
        );
    } else {
        eprintln!("No transform specified; model copied to {}", args.output);
    }
}
