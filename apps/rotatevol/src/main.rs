use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Rotate a volume around the Z axis by a specified angle.
#[derive(Parser)]
#[command(name = "rotatevol", about = "Rotate a volume around the Z axis")]
struct Args {
    /// Input MRC file
    input: String,
    /// Output MRC file
    output: String,
    /// Rotation angle in degrees (counterclockwise)
    #[arg(short = 'a', long, default_value_t = 0.0)]
    angle: f32,
    /// Fill value for areas outside the rotated image
    #[arg(short = 'f', long, default_value_t = 0.0)]
    fill: f32,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.add_label(&format!("rotatevol: {:.1} degrees", args.angle));
    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let rad = -args.angle.to_radians(); // negative for inverse mapping
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let cx = nx as f32 / 2.0;
    let cy = ny as f32 / 2.0;

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let src = reader.read_slice_f32(z).unwrap();
        let mut dst = vec![args.fill; nx * ny];

        for oy in 0..ny {
            for ox in 0..nx {
                // Inverse map: find source position
                let dx = ox as f32 - cx;
                let dy = oy as f32 - cy;
                let sx = cos_a * dx - sin_a * dy + cx;
                let sy = sin_a * dx + cos_a * dy + cy;

                // Bilinear interpolation
                let x0 = sx.floor() as isize;
                let y0 = sy.floor() as isize;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                if x0 >= 0 && x0 + 1 < nx as isize && y0 >= 0 && y0 + 1 < ny as isize {
                    let x0u = x0 as usize;
                    let y0u = y0 as usize;
                    let v00 = src[y0u * nx + x0u];
                    let v10 = src[y0u * nx + x0u + 1];
                    let v01 = src[(y0u + 1) * nx + x0u];
                    let v11 = src[(y0u + 1) * nx + x0u + 1];
                    dst[oy * nx + ox] = v00 * (1.0 - fx) * (1.0 - fy)
                        + v10 * fx * (1.0 - fy)
                        + v01 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&dst);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&dst).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32).unwrap();
    eprintln!("rotatevol: rotated {:.1} degrees", args.angle);
}
