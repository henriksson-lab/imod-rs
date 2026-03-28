//! xfsimplex - Find the best linear transform between two images using simplex
//! (Nelder-Mead) minimization.
//!
//! Searches for the transform that minimizes the difference (or maximizes
//! correlation) between a pair of images. Supports searching over formal
//! (6-parameter) or "natural" (rotation/mag/stretch) parameter subsets,
//! with optional binning, bandpass filtering, and edge exclusion.

use clap::Parser;
use imod_math::amoeba::{amoeba, amoeba_init};
use imod_mrc::MrcReader;
use imod_transforms::{read_xf_file, write_xf_file, LinearTransform};
use std::process;

#[derive(Parser)]
#[command(name = "xfsimplex", about = "Find transform between two images by simplex minimization")]
struct Args {
    /// First (reference) image file
    #[arg(short = 'a', long = "aimage")]
    aimage: String,

    /// Second (aligned) image file
    #[arg(short = 'b', long = "bimage")]
    bimage: String,

    /// Output transform file
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Initial transform file
    #[arg(long = "initial")]
    initial: Option<String>,

    /// Which line to use from initial transform file (0-based)
    #[arg(long = "useline", default_value = "0")]
    useline: usize,

    /// Sections to use from each file (0-based: ref, ali)
    #[arg(long = "sections", num_args = 2, value_names = ["REF", "ALI"])]
    sections: Option<Vec<i32>>,

    /// Number of variables to search: 0=formal 6, 1-6=natural subset
    #[arg(short = 'v', long = "variables", default_value = "0")]
    variables: i32,

    /// Limits on search parameters (comma-separated)
    #[arg(long = "limits", value_delimiter = ',')]
    limits: Option<Vec<f32>>,

    /// Edge fraction to ignore
    #[arg(short = 'e', long = "edge", default_value = "0.05")]
    edge: f32,

    /// Binning to apply to images
    #[arg(long = "binning", default_value = "2")]
    binning: usize,

    /// High-pass filter sigma1
    #[arg(long = "sig1", default_value = "0.0")]
    sigma1: f32,

    /// Low-pass filter radius2
    #[arg(long = "rad2", default_value = "0.0")]
    radius2: f32,

    /// High-pass filter radius1
    #[arg(long = "rad1", default_value = "0.0")]
    radius1: f32,

    /// Low-pass filter sigma2
    #[arg(long = "sig2", default_value = "0.0")]
    sigma2: f32,

    /// Float option: 0=range, 1=mean+sd, -1=none
    #[arg(long = "float", default_value = "1")]
    float_opt: i32,

    /// Use correlation coefficient instead of difference
    #[arg(long = "ccc")]
    ccc: bool,

    /// Use linear interpolation
    #[arg(long = "linear")]
    linear: bool,

    /// Coarse search tolerances (ftol, ptol)
    #[arg(long = "coarse", num_args = 2, value_names = ["FTOL", "PTOL"])]
    coarse: Option<Vec<f32>>,

    /// Final search tolerances (ftol, ptol)
    #[arg(long = "final", num_args = 2, value_names = ["FTOL", "PTOL"])]
    final_tol: Option<Vec<f32>>,

    /// Step size factor
    #[arg(long = "step", default_value = "2.0")]
    step_factor: f32,

    /// Trace output level (0=none, 1=iterations, 2=all)
    #[arg(long = "trace", default_value = "0")]
    trace: i32,
}

/// Decompose a 2x2 matrix into natural parameters: (rotation, mag, dmag, drot)
fn to_natural_params(xf: &LinearTransform) -> [f32; 6] {
    let theta1 = xf.a21.atan2(xf.a11);
    let theta2 = (-xf.a12).atan2(xf.a22);
    let theta = (theta1 + theta2) / 2.0;
    let mag1 = (xf.a11 * xf.a11 + xf.a21 * xf.a21).sqrt();
    let mag2 = (xf.a12 * xf.a12 + xf.a22 * xf.a22).sqrt();
    [
        xf.dx,
        xf.dy,
        theta.to_degrees(),
        (mag1 + mag2) / 2.0,
        mag2 - mag1,
        (theta2 - theta1).to_degrees(),
    ]
}

/// Reconstruct transform from natural parameters
fn from_natural_params(p: &[f32]) -> LinearTransform {
    let dx = p[0];
    let dy = p[1];
    let theta = p[2].to_radians();
    let mag = if p.len() > 3 { p[3] } else { 1.0 };
    let dmag = if p.len() > 4 { p[4] } else { 0.0 };
    let drot = if p.len() > 5 { p[5].to_radians() / 2.0 } else { 0.0 };

    let mag1 = mag - dmag / 2.0;
    let mag2 = mag + dmag / 2.0;
    let t1 = theta - drot;
    let t2 = theta + drot;

    LinearTransform {
        a11: mag1 * t1.cos(),
        a12: -mag2 * t2.sin(),
        a21: mag1 * t1.sin(),
        a22: mag2 * t2.cos(),
        dx,
        dy,
    }
}

/// Build a transform from parameter vector (formal or natural)
fn params_to_xf(params: &[f32], natural: i32) -> LinearTransform {
    if natural == 0 {
        // Formal: dx, dy, a11, a12, a21, a22
        LinearTransform {
            a11: params[2],
            a12: params[3],
            a21: params[4],
            a22: params[5],
            dx: params[0],
            dy: params[1],
        }
    } else {
        from_natural_params(params)
    }
}

/// Bin an image by integer factor using simple averaging
fn bin_image(data: &[f32], nx: usize, ny: usize, binning: usize) -> (Vec<f32>, usize, usize) {
    let bnx = nx / binning;
    let bny = ny / binning;
    let binsq = (binning * binning) as f32;
    let mut out = vec![0.0f32; bnx * bny];

    for by in 0..bny {
        for bx in 0..bnx {
            let mut sum = 0.0f32;
            for dy in 0..binning {
                for dx in 0..binning {
                    let sx = bx * binning + dx;
                    let sy = by * binning + dy;
                    sum += data[sy * nx + sx];
                }
            }
            out[by * bnx + bx] = sum / binsq;
        }
    }
    (out, bnx, bny)
}

/// Compute mean and standard deviation
fn mean_sd(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f64;
    let sum: f64 = data.iter().map(|&v| v as f64).sum();
    let mean = sum / n;
    let sumsq: f64 = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
    (mean as f32, (sumsq / n).sqrt() as f32)
}

/// Apply a transform to the second image and compute difference measure
fn compute_difference(
    img_a: &[f32],
    img_b: &[f32],
    nx: usize,
    ny: usize,
    xf: &LinearTransform,
    x1: usize,
    x2: usize,
    y1: usize,
    y2: usize,
    use_interp: bool,
) -> f64 {
    let inv = xf.inverse();
    let cxo = nx as f32 / 2.0;
    let cyo = ny as f32 / 2.0;
    let mut sum = 0.0f64;
    let mut count = 0u64;

    for iy in y1..y2 {
        for ix in x1..x2 {
            let xf_pos = inv.apply(cxo, cyo, ix as f32, iy as f32);
            let sx = xf_pos.0;
            let sy = xf_pos.1;

            if sx < 0.0 || sy < 0.0 || sx >= (nx - 1) as f32 || sy >= (ny - 1) as f32 {
                continue;
            }

            let val_b = if use_interp {
                // Bilinear interpolation
                let ix0 = sx as usize;
                let iy0 = sy as usize;
                let fx = sx - ix0 as f32;
                let fy = sy - iy0 as f32;
                let v00 = img_b[iy0 * nx + ix0];
                let v10 = img_b[iy0 * nx + ix0 + 1];
                let v01 = img_b[(iy0 + 1) * nx + ix0];
                let v11 = img_b[(iy0 + 1) * nx + ix0 + 1];
                v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy
            } else {
                let ix0 = (sx + 0.5) as usize;
                let iy0 = (sy + 0.5) as usize;
                img_b[iy0.min(ny - 1) * nx + ix0.min(nx - 1)]
            };

            let val_a = img_a[iy * nx + ix];
            let diff = (val_a - val_b) as f64;
            sum += diff * diff;
            count += 1;
        }
    }

    if count == 0 {
        return 1e30;
    }
    sum / count as f64
}

fn main() {
    let args = Args::parse();

    // Open images
    let mut reader_a = MrcReader::open(&args.aimage).unwrap_or_else(|e| {
        eprintln!("ERROR: XFSIMPLEX - opening {}: {}", args.aimage, e);
        process::exit(1);
    });
    let mut reader_b = MrcReader::open(&args.bimage).unwrap_or_else(|e| {
        eprintln!("ERROR: XFSIMPLEX - opening {}: {}", args.bimage, e);
        process::exit(1);
    });

    let hdr_a = reader_a.header().clone();
    let hdr_b = reader_b.header().clone();

    let orig_nx = hdr_a.nx as usize;
    let orig_ny = hdr_a.ny as usize;

    if hdr_b.nx as usize != orig_nx || hdr_b.ny as usize != orig_ny {
        eprintln!("ERROR: XFSIMPLEX - images must be the same size in X and Y");
        process::exit(1);
    }

    let (iz_ref, iz_ali) = if let Some(ref sec) = args.sections {
        (sec[0] as usize, sec[1] as usize)
    } else {
        (0, 0)
    };

    // Read initial transform if provided
    let mut init_params = if args.variables == 0 {
        vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0] // formal: dx,dy,a11,a12,a21,a22
    } else {
        vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0] // natural: dx,dy,rot,mag,dmag,drot
    };

    if let Some(ref xf_file) = args.initial {
        let xforms = read_xf_file(xf_file).unwrap_or_else(|e| {
            eprintln!("ERROR: XFSIMPLEX - reading initial transforms: {}", e);
            process::exit(1);
        });
        if args.useline >= xforms.len() {
            eprintln!("ERROR: XFSIMPLEX - transform line {} not found", args.useline);
            process::exit(1);
        }
        let xf = &xforms[args.useline];
        if args.variables == 0 {
            init_params = vec![xf.dx, xf.dy, xf.a11, xf.a12, xf.a21, xf.a22];
        } else {
            init_params = to_natural_params(xf).to_vec();
        }
    }

    // Read and bin images
    let img_a_raw = reader_a.read_slice_f32(iz_ref).unwrap_or_else(|e| {
        eprintln!("ERROR: XFSIMPLEX - reading reference image: {}", e);
        process::exit(1);
    });
    let img_b_raw = reader_b.read_slice_f32(iz_ali).unwrap_or_else(|e| {
        eprintln!("ERROR: XFSIMPLEX - reading aligned image: {}", e);
        process::exit(1);
    });

    let binning = args.binning.max(1);
    let (img_a, nx, ny) = bin_image(&img_a_raw, orig_nx, orig_ny, binning);
    let (mut img_b, _, _) = bin_image(&img_b_raw, orig_nx, orig_ny, binning);

    // Float images to match
    let (mean_a, sd_a) = mean_sd(&img_a);
    let (mean_b, sd_b) = mean_sd(&img_b);

    if args.float_opt >= 0 && sd_b > 0.0 {
        let scale = if args.float_opt == 0 {
            let (min_a, max_a) = img_a.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
            let (min_b, max_b) = img_b.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
            if max_b > min_b { (max_a - min_a) / (max_b - min_b) } else { 1.0 }
        } else {
            sd_a / sd_b
        };
        let add = if args.float_opt == 0 {
            let min_a = img_a.iter().cloned().fold(f32::MAX, f32::min);
            let min_b = img_b.iter().cloned().fold(f32::MAX, f32::min);
            min_a - scale * min_b
        } else {
            mean_a - scale * mean_b
        };
        for v in img_b.iter_mut() {
            *v = scale * *v + add;
        }
    }

    // Edge exclusion
    let matt_x = if args.edge >= 1.0 {
        (args.edge / binning as f32) as usize
    } else {
        (nx as f32 * args.edge) as usize
    };
    let matt_y = if args.edge >= 1.0 {
        (args.edge / binning as f32) as usize
    } else {
        (ny as f32 * args.edge) as usize
    };
    let x1 = matt_x;
    let x2 = nx - matt_x;
    let y1 = matt_y;
    let y2 = ny - matt_y;

    // Reduce shift parameters by binning
    let reduction = binning as f32;
    init_params[0] /= reduction;
    init_params[1] /= reduction;

    // Set up simplex search
    let natural = args.variables;
    let nvar = if natural == 0 { 6 } else { natural as usize };
    let nvar = nvar.min(init_params.len());

    // Step sizes
    let mut da = if natural == 0 {
        vec![1.0, 1.0, 0.025, 0.025, 0.025, 0.025]
    } else {
        vec![1.0, 1.0, 2.0, 0.02, 0.02, 2.0]
    };
    for d in da.iter_mut() {
        *d *= args.step_factor;
    }

    // Tolerances
    let (ftol_coarse, ptol_coarse) = if let Some(ref c) = args.coarse {
        (c[0], c[1])
    } else {
        (5e-3, 0.2)
    };
    let (ftol_final, ptol_final) = if let Some(ref f) = args.final_tol {
        (f[0], f[1])
    } else {
        (5e-4, 0.02)
    };

    let use_interp = args.linear;

    // Objective function
    let objective = |params: &[f32]| -> f32 {
        let mut full_params = init_params.clone();
        for i in 0..nvar.min(params.len()) {
            full_params[i] = params[i];
        }
        let xf = params_to_xf(&full_params, natural);
        compute_difference(&img_a, &img_b, nx, ny, &xf, x1, x2, y1, y2, use_interp) as f32
    };

    let search_params: Vec<f32> = init_params[..nvar].to_vec();
    let search_da: Vec<f32> = da[..nvar].to_vec();

    // mp = nvar + 1 (number of simplex vertices)
    let mp = nvar + 1;

    // Run simplex search (coarse then fine)
    let mut best_params = search_params.clone();

    let run_simplex = |params: &[f32], da_vec: &[f32], ftol: f32, ptol_val: f32| -> (Vec<f32>, f32, usize) {
        let mut p = vec![0.0f32; mp * nvar];
        let mut y = vec![0.0f32; mp];
        let ptol = amoeba_init(&mut p, &mut y, mp, nvar, 1.0, ptol_val, params, da_vec, &objective);
        let result = amoeba(&mut p, &mut y, mp, nvar, ftol, &ptol, &objective);
        let best_idx = result.best_index;
        let best_y = y[best_idx];
        let mut best = vec![0.0f32; nvar];
        for i in 0..nvar {
            best[i] = p[best_idx + i * mp];
        }
        (best, best_y, result.iterations)
    };

    // Coarse search
    if ftol_coarse > 0.0 && ptol_coarse > 0.0 {
        let (coarse_best, coarse_y, _iters) = run_simplex(&best_params, &search_da, ftol_coarse, ptol_coarse);
        best_params = coarse_best;
        if args.trace > 0 {
            eprintln!("Coarse search: best difference = {:.6}", coarse_y);
        }
    }

    // Fine search
    let fine_da: Vec<f32> = search_da.iter().map(|&d| d * 0.5).collect();
    let (fine_best, fine_y, fine_iters) = run_simplex(&best_params, &fine_da, ftol_final, ptol_final);
    best_params = fine_best;
    eprintln!(
        "Final search: difference = {:.6} after {} iterations",
        fine_y, fine_iters
    );

    // Reconstruct full parameter vector and build output transform
    let mut full_params = init_params.clone();
    for i in 0..nvar.min(best_params.len()) {
        full_params[i] = best_params[i];
    }

    // Scale shifts back up
    full_params[0] *= reduction;
    full_params[1] *= reduction;

    let result_xf = params_to_xf(&full_params, natural);

    write_xf_file(&args.output, &[result_xf]).unwrap_or_else(|e| {
        eprintln!("ERROR: XFSIMPLEX - writing output: {}", e);
        process::exit(1);
    });

    eprintln!(
        "Transform: a11={:.5} a12={:.5} a21={:.5} a22={:.5} dx={:.2} dy={:.2}",
        result_xf.a11, result_xf.a12, result_xf.a21, result_xf.a22, result_xf.dx, result_xf.dy
    );
}
