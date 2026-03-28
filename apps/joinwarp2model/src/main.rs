use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};
use imod_mrc::MrcReader;
use imod_warp::WarpFile;

/// Convert warping control points from a join into a refining model.
///
/// Reads a warp file containing control points from serial section joining,
/// transforms the points using the applied G transforms, and creates an IMOD
/// model with point pairs across each join boundary that can be used for
/// further refinement.
#[derive(Parser)]
#[command(name = "joinwarp2model", version, about)]
struct Args {
    /// Input warp file with control points.
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output model file.
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Joined image file (for pixel spacing and dimensions).
    #[arg(short = 'j', long = "joined")]
    joined: Option<String>,

    /// Applied transform (G transform) file.
    #[arg(short = 'x', long = "xform")]
    xform_file: String,

    /// Size of joined image in X and Y (alternative to --joined).
    #[arg(long = "size", num_args = 2)]
    join_size: Vec<i32>,

    /// Pixel spacing of joined image (alternative to --joined).
    #[arg(long = "pixel")]
    pixel_spacing: Option<f32>,

    /// Offset in X and Y.
    #[arg(long = "offset", num_args = 2, default_values_t = vec![0, 0])]
    offset: Vec<i32>,

    /// Binning of joined image.
    #[arg(short = 'b', long = "binning", default_value_t = 1)]
    binning: i32,

    /// Chunk sizes (comma-separated list of section counts).
    #[arg(short = 'c', long = "chunks", value_delimiter = ',', required = true)]
    chunk_sizes: Vec<i32>,
}

/// Invert a 2D linear transform [a11 a12 dx; a21 a22 dy].
fn invert_xf(xf: &[f32; 6]) -> [f32; 6] {
    let det = xf[0] * xf[3] - xf[1] * xf[2];
    if det.abs() < 1.0e-10 {
        return *xf; // degenerate, return identity-ish
    }
    let inv_det = 1.0 / det;
    let a11 = xf[3] * inv_det;
    let a12 = -xf[1] * inv_det;
    let a21 = -xf[2] * inv_det;
    let a22 = xf[0] * inv_det;
    let dx = -(a11 * xf[4] + a12 * xf[5]);
    let dy = -(a21 * xf[4] + a22 * xf[5]);
    [a11, a12, a21, a22, dx, dy]
}

/// Apply a 2D transform centered at (cx, cy).
fn apply_xf(xf: &[f32; 6], cx: f32, cy: f32, x: f32, y: f32) -> (f32, f32) {
    let dx = x - cx;
    let dy = y - cy;
    let xout = xf[0] * dx + xf[1] * dy + xf[4] + cx;
    let yout = xf[2] * dx + xf[3] * dy + xf[5] + cy;
    (xout, yout)
}

fn main() {
    let args = Args::parse();

    // Get the warp point file
    let warp = WarpFile::from_file(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: joinwarp2model - error reading warp file {}: {}", args.input, e);
        process::exit(1);
    });

    let nz_warp = warp.sections.len();
    if nz_warp == 0 {
        println!("There are no warping transforms; no warp point model produced");
        process::exit(0);
    }

    // Check that the warp file has control points (flags bit 0)
    if warp.flags & 1 == 0 {
        eprintln!("ERROR: joinwarp2model - warp file does not contain control points");
        process::exit(1);
    }

    // Turn chunk sizes into cumulative Z values
    let mut cum_chunks = args.chunk_sizes.clone();
    for i in 1..cum_chunks.len() {
        cum_chunks[i] += cum_chunks[i - 1];
    }
    let num_chunks = cum_chunks.len();

    if nz_warp != num_chunks {
        eprintln!(
            "ERROR: joinwarp2model - number of chunks ({}) does not match transforms ({})",
            num_chunks, nz_warp
        );
        process::exit(1);
    }

    // Get joined file dimensions and pixel spacing
    let (x_join_size, y_join_size, join_pixel) = if let Some(ref joined) = args.joined {
        let reader = MrcReader::open(joined).unwrap_or_else(|e| {
            eprintln!("ERROR: joinwarp2model - error opening joined file {}: {}", joined, e);
            process::exit(1);
        });
        let hdr = reader.header();
        let pix = if hdr.mx > 0 && hdr.xlen > 0.0 {
            hdr.xlen / hdr.mx as f32
        } else {
            1.0
        };

        // Verify chunk sizes match
        if hdr.nz != cum_chunks[num_chunks - 1] {
            eprintln!(
                "ERROR: joinwarp2model - sum of chunk sizes ({}) does not match Z size ({})",
                cum_chunks[num_chunks - 1], hdr.nz
            );
            process::exit(1);
        }

        (hdr.nx, hdr.ny, pix)
    } else if args.join_size.len() >= 2 && args.pixel_spacing.is_some() {
        (args.join_size[0], args.join_size[1], args.pixel_spacing.unwrap())
    } else {
        eprintln!(
            "ERROR: joinwarp2model - must specify either --joined or both --size and --pixel"
        );
        process::exit(1);
    };

    // Read the applied transform file
    let xform_warp = WarpFile::from_file(&args.xform_file).unwrap_or_else(|e| {
        eprintln!(
            "ERROR: joinwarp2model - error reading transform file {}: {}",
            args.xform_file, e
        );
        process::exit(1);
    });

    // Verify that the transform file matches
    if xform_warp.nx != warp.nx || xform_warp.ny != warp.ny {
        eprintln!("ERROR: joinwarp2model - applied transform file dimensions do not match warp file");
        process::exit(1);
    }

    let warp_pixel = warp.pixel_size;
    let nx_warp = warp.nx as f32;
    let ny_warp = warp.ny as f32;
    let x_offset = args.offset.get(0).copied().unwrap_or(0) as f32;
    let y_offset = args.offset.get(1).copied().unwrap_or(0) as f32;
    let binning = args.binning as f32;

    // Build the model with warp points
    let mut model = ImodModel::default();
    model.xmax = x_join_size;
    model.ymax = y_join_size;
    model.zmax = cum_chunks[num_chunks - 1];

    let mut obj = ImodObject::default();
    obj.name = "Warp refine points".into();
    obj.flags |= 1 << 3; // IMOD_OBJFLAG_OPEN
    obj.pdrawsize = 50;
    obj.red = 1.0;
    obj.green = 0.0;
    obj.blue = 0.0;

    // Process each section (skip first since it's the reference)
    for iz in 1..nz_warp {
        let sec = &warp.sections[iz];
        let n_control = sec.control_x.len();

        for pt_idx in 0..n_control {
            let cx = sec.control_x[pt_idx];
            let cy = sec.control_y[pt_idx];

            // The control point position plus vector gives the position on aligned image.
            // We need the position on the unaligned image, so apply inverse linear transform.
            let xf = &sec.transforms[pt_idx];
            let linear = [xf.a11, xf.a12, xf.a21, xf.a22, xf.dx, xf.dy];
            let inv = invert_xf(&linear);

            let aligned_x = cx + xf.dx;
            let aligned_y = cy + xf.dy;

            let (xback, yback) = apply_xf(
                &inv,
                nx_warp / 2.0,
                ny_warp / 2.0,
                aligned_x,
                aligned_y,
            );

            // Scale from warp file coordinates to joined file coordinates
            let x_join = (xback - nx_warp / 2.0) * warp_pixel / join_pixel
                + x_join_size as f32 / 2.0
                - x_offset / binning;
            let y_join = (yback - ny_warp / 2.0) * warp_pixel / join_pixel
                + y_join_size as f32 / 2.0
                - y_offset / binning;
            let z_bot = cum_chunks[iz - 1] as f32 - 1.0;
            let z_top = z_bot + 1.0;

            // Create a contour with two points (one on each side of the boundary)
            let cont = ImodContour {
                points: vec![
                    Point3f { x: x_join, y: y_join, z: z_bot },
                    Point3f { x: x_join, y: y_join, z: z_top },
                ],
                ..Default::default()
            };
            obj.contours.push(cont);
        }
    }

    model.objects.push(obj);

    // Write the output model
    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: joinwarp2model - error writing model: {}", e);
        process::exit(1);
    }

    println!(
        "Wrote model with {} contours to {}",
        model.objects[0].contours.len(),
        args.output
    );
}
