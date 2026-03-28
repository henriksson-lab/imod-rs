use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcReader, MrcWriter, MrcHeader};
use std::io::{BufRead, BufReader};

/// Extract subsections from a stack using point files for center determination
/// and extraction coordinates, rotating each subsection relative to its
/// center-of-mass direction.
#[derive(Parser)]
#[command(name = "extstack", about = "Extract subsections from a stack")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC image file
    #[arg(short = 'o', long)]
    output: String,

    /// Reference (complete) point file for center determination (x y z per line)
    #[arg(short = 'r', long)]
    reference: String,

    /// Expected number of reference points per section
    #[arg(short = 'n', long)]
    num_ref: usize,

    /// Extraction point file (x y z per line)
    #[arg(short = 'e', long)]
    extract: String,

    /// Length of box edge parallel to radial line (pixels)
    #[arg(short = 'p', long)]
    parallel_len: usize,

    /// Length of box edge normal to radial line (pixels)
    #[arg(long)]
    normal_len: usize,

    /// Additional rotation angle in degrees (positive = CCW)
    #[arg(short = 'a', long, default_value_t = 0.0)]
    add_angle: f64,
}

fn read_points(path: &str) -> Vec<(f64, f64, i32)> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTSTACK - opening {path}: {e}");
        std::process::exit(1);
    });
    let reader = BufReader::new(f);
    let mut pts = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let x: f64 = parts[0].parse().unwrap();
            let y: f64 = parts[1].parse().unwrap();
            let z: i32 = parts[2].parse().unwrap();
            pts.push((x, y, z));
        }
    }
    pts
}

/// Bilinear interpolation of a rotated/translated subsection from a source image.
fn extract_subsection(
    src: &[f32],
    src_nx: usize,
    src_ny: usize,
    dst_nx: usize,
    dst_ny: usize,
    amat: [[f64; 2]; 2],
    xc: f64,
    yc: f64,
    xt: f64,
    yt: f64,
    fill: f32,
) -> Vec<f32> {
    let mut out = vec![fill; dst_nx * dst_ny];
    let dxc = dst_nx as f64 / 2.0;
    let dyc = dst_ny as f64 / 2.0;

    for iy in 0..dst_ny {
        for ix in 0..dst_nx {
            let dx = ix as f64 - dxc;
            let dy = iy as f64 - dyc;
            let sx = amat[0][0] * dx + amat[0][1] * dy + xc + xt;
            let sy = amat[1][0] * dx + amat[1][1] * dy + yc + yt;

            let ix0 = sx.floor() as i64;
            let iy0 = sy.floor() as i64;
            if ix0 >= 0 && ix0 + 1 < src_nx as i64 && iy0 >= 0 && iy0 + 1 < src_ny as i64 {
                let fx = (sx - ix0 as f64) as f32;
                let fy = (sy - iy0 as f64) as f32;
                let i00 = iy0 as usize * src_nx + ix0 as usize;
                let v = src[i00] * (1.0 - fx) * (1.0 - fy)
                    + src[i00 + 1] * fx * (1.0 - fy)
                    + src[i00 + src_nx] * (1.0 - fx) * fy
                    + src[i00 + src_nx + 1] * fx * fy;
                out[iy * dst_nx + ix] = v;
            }
        }
    }
    out
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTSTACK - opening input: {e}");
        std::process::exit(1);
    });
    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    // Read reference points and compute per-section centers
    let ref_pts = read_points(&args.reference);
    let max_z = ref_pts.iter().map(|p| p.2).max().unwrap_or(0) as usize;
    let mut xcen = vec![0.0_f64; max_z + 1];
    let mut ycen = vec![0.0_f64; max_z + 1];
    let mut ccnt = vec![0usize; max_z + 1];
    for &(x, y, z) in &ref_pts {
        let zi = z as usize;
        xcen[zi] += x;
        ycen[zi] += y;
        ccnt[zi] += 1;
    }
    for i in 0..=max_z {
        if ccnt[i] > 0 {
            if ccnt[i] != args.num_ref {
                eprintln!(
                    "ERROR: Section {} -- expected {} reference points, found {}",
                    i, args.num_ref, ccnt[i]
                );
                std::process::exit(1);
            }
            xcen[i] /= ccnt[i] as f64;
            ycen[i] /= ccnt[i] as f64;
        }
    }

    // Read extraction points
    let ext_pts = read_points(&args.extract);

    let nxb = args.normal_len;
    let nyb = args.parallel_len;
    let out_header = MrcHeader::new(nxb as i32, nyb as i32, 0, MrcMode::Float);
    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTSTACK - creating output: {e}");
        std::process::exit(1);
    });

    let mut nsec = 0i32;
    let mut dmin_tot = f32::MAX;
    let mut dmax_tot = f32::MIN;
    let mut dmean_sum = 0.0_f64;

    let add_angle = args.add_angle;
    let xc_img = nx as f64 / 2.0;
    let yc_img = ny as f64 / 2.0;

    for iz in 0..nz {
        let section = reader.read_slice_f32(iz).unwrap_or_else(|e| {
            eprintln!("ERROR: EXTSTACK - reading section {iz}: {e}");
            std::process::exit(1);
        });

        // Collect extraction points for this section
        let local: Vec<(f64, f64)> = ext_pts
            .iter()
            .filter(|p| p.2 as usize == iz)
            .map(|p| (p.0, p.1))
            .collect();

        if local.is_empty() {
            continue;
        }

        let cxi = if iz < xcen.len() { xcen[iz] } else { xc_img };
        let cyi = if iz < ycen.len() { ycen[iz] } else { yc_img };

        for &(px, py) in &local {
            let opp = px - cxi;
            let adj = py - cyi;
            let theta_deg = if opp != 0.0 && adj != 0.0 {
                opp.atan2(adj).to_degrees() - 180.0 + add_angle
            } else {
                add_angle
            };
            let theta = theta_deg.to_radians();
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let amat = [[cos_t, -sin_t], [sin_t, cos_t]];
            let xt = -(amat[0][0] * (px - xc_img) + amat[0][1] * (py - yc_img));
            let yt = -(amat[1][0] * (px - xc_img) + amat[1][1] * (py - yc_img));

            let (_, _, dmean_h) = min_max_mean(&section);
            let sub = extract_subsection(
                &section, nx, ny, nxb, nyb, amat, xc_img, yc_img, xt, yt, dmean_h,
            );

            let (smin, smax, smean) = min_max_mean(&sub);
            dmin_tot = dmin_tot.min(smin);
            dmax_tot = dmax_tot.max(smax);
            dmean_sum += smean as f64;

            eprintln!(" Writing new section # {nsec}");
            writer.write_slice_f32(&sub).unwrap_or_else(|e| {
                eprintln!("ERROR: EXTSTACK - writing section: {e}");
                std::process::exit(1);
            });
            nsec += 1;
        }
    }

    let dmean = if nsec > 0 {
        (dmean_sum / nsec as f64) as f32
    } else {
        0.0
    };
    writer.finish(dmin_tot, dmax_tot, dmean).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTSTACK - finalizing output: {e}");
        std::process::exit(1);
    });

    eprintln!("Extracted {nsec} subsections");
}
