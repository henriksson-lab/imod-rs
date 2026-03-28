//! xftoxg - Convert section-to-section transforms (f) to global alignment
//! transforms (g).
//!
//! Reads a file of linear transforms, each mapping section N to section N-1,
//! and computes transforms to align each section to a common reference.
//! Supports global alignment (nfit=0), local polynomial fits (nfit=N), and
//! hybrid fits that combine global and local approaches.

use clap::Parser;
use imod_math::regression::polynomial_fit;
use imod_transforms::{read_xf_file, write_xf_file, LinearTransform};
use std::process;

#[derive(Parser)]
#[command(name = "xftoxg", about = "Convert section-to-section transforms to global alignment transforms")]
struct Args {
    /// Input file of f transforms (section-to-section)
    #[arg(short = 'i', long)]
    input: String,

    /// Output file of g transforms (global alignment)
    #[arg(short = 'g', long = "goutput")]
    output: Option<String>,

    /// Number of nearby sections to fit (0 = global alignment)
    #[arg(short = 'n', long = "nfit", default_value = "7")]
    nfit: i32,

    /// Reference section (1-based); forces nfit=0
    #[arg(short = 'r', long = "ref")]
    reference: Option<usize>,

    /// Order of polynomial fit (1 = linear)
    #[arg(short = 'o', long = "order", default_value = "1")]
    order: usize,

    /// Hybrid fits: # of parameters to do central alignment on (1-4)
    #[arg(short = 'm', long = "mixed", default_value = "0")]
    hybrid: i32,
}

/// Decompose a 2x2 matrix + shift into "natural" parameters:
/// (rotation_angle, overall_mag, dmag, drot, dx, dy)
fn to_natural(xf: &LinearTransform) -> [f32; 6] {
    let a11 = xf.a11;
    let a12 = xf.a12;
    let a21 = xf.a21;
    let a22 = xf.a22;

    // Rotation = average of atan2 for each row
    let theta1 = a21.atan2(a11);
    let theta2 = (-a12).atan2(a22);
    let theta = (theta1 + theta2) / 2.0;

    // Overall magnification
    let mag1 = (a11 * a11 + a21 * a21).sqrt();
    let mag2 = (a12 * a12 + a22 * a22).sqrt();
    let mag = (mag1 + mag2) / 2.0;

    // Difference in magnification (stretch)
    let dmag = mag2 - mag1;

    // Difference in rotation
    let drot = (theta2 - theta1).to_degrees();

    [theta.to_degrees(), mag, dmag, drot, xf.dx, xf.dy]
}

/// Reconstruct a LinearTransform from natural parameters
fn from_natural(nat: &[f32; 6]) -> LinearTransform {
    let theta = nat[0].to_radians();
    let mag = nat[1];
    let dmag = nat[2];
    let drot_rad = nat[3].to_radians() / 2.0;

    let mag1 = mag - dmag / 2.0;
    let mag2 = mag + dmag / 2.0;
    let theta1 = theta - drot_rad;
    let theta2 = theta + drot_rad;

    LinearTransform {
        a11: mag1 * theta1.cos(),
        a12: -mag2 * theta2.sin(),
        a21: mag1 * theta1.sin(),
        a22: mag2 * theta2.cos(),
        dx: nat[4],
        dy: nat[5],
    }
}

fn main() {
    let args = Args::parse();

    // Determine output filename
    let output = args.output.unwrap_or_else(|| {
        if args.input.ends_with("xf") {
            let mut s = args.input.clone();
            s.pop();
            s.push('g');
            s
        } else {
            eprintln!("ERROR: XFTOXG - no output file specified and input doesn't end in 'xf'");
            process::exit(1);
        }
    });

    let f_xforms = read_xf_file(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: XFTOXG - reading transform file {}: {}", args.input, e);
        process::exit(1);
    });

    let nlist = f_xforms.len();
    if nlist == 0 {
        eprintln!("ERROR: XFTOXG - no transforms in input file");
        process::exit(1);
    }

    // Handle reference section
    let mut nfit = args.nfit;
    let ref_sec = args.reference;
    if let Some(r) = ref_sec {
        if r < 1 || r > nlist {
            eprintln!("ERROR: XFTOXG - reference section {} out of range", r);
            process::exit(1);
        }
        nfit = 0;
    }
    if nfit < 0 {
        eprintln!("ERROR: XFTOXG - nfit cannot be negative");
        process::exit(1);
    }

    if nfit > 0 && nlist < 3 {
        eprintln!("WARNING: XFTOXG - Computing global alignment since there are fewer than 3 transforms");
        nfit = 0;
    }

    let order = args.order.max(1).min(10);
    let nhybrid = args.hybrid.max(0).min(4);

    // Step 1: Compute cumulative transforms (g[i] aligns section i to section 0)
    let mut g = vec![LinearTransform::identity(); nlist];
    for i in 1..nlist {
        g[i] = f_xforms[i].then(&g[i - 1]);
    }

    // Step 2: Convert to natural parameters
    let mut nat: Vec<[f32; 6]> = g.iter().map(|xf| to_natural(xf)).collect();

    // Unwrap rotation angles for continuity
    for i in 1..nlist {
        let diff = nat[i - 1][0] - nat[i][0];
        if diff.abs() > 180.0 {
            let delta = 360.0 * diff.signum();
            for j in i..nlist {
                nat[j][0] += delta;
            }
        }
    }

    // Step 3: Compute the alignment reference
    if nfit == 0 {
        // Global alignment: align to average or reference section
        let g_result = if let Some(r) = ref_sec {
            let ref_idx = r - 1;
            g[ref_idx].inverse()
        } else {
            // Average of natural parameters
            let mut avg = [0.0f32; 6];
            for n in &nat {
                for j in 0..6 {
                    avg[j] += n[j] / nlist as f32;
                }
            }
            from_natural(&avg).inverse()
        };

        let results: Vec<_> = g.iter().map(|gi| gi.then(&g_result)).collect();
        write_xf_file(&output, &results).unwrap_or_else(|e| {
            eprintln!("ERROR: XFTOXG - writing output: {}", e);
            process::exit(1);
        });
    } else {
        // Local or full polynomial fit
        let order_use = order.min(if nfit > 1 { nfit as usize - 1 } else { nlist - 1 });

        let mut results = Vec::with_capacity(nlist);

        for ilist in 0..nlist {
            let (kl_low, kl_high) = if nfit == 1 {
                (0, nlist - 1)
            } else {
                let half = nfit as usize / 2;
                let lo = if ilist >= half { ilist - half } else { 0 };
                let hi = (lo + nfit as usize - 1).min(nlist - 1);
                let lo = if hi + 1 < nfit as usize {
                    0
                } else {
                    hi + 1 - nfit as usize
                };
                (lo, hi)
            };

            let center = (kl_high + kl_low) / 2;

            // Fit polynomial to each natural parameter component
            let mut fitted_nat = [0.0f32; 6];
            let num_fit = kl_high + 1 - kl_low;
            let order_local = order_use.min(num_fit.saturating_sub(2)).max(1);

            let x_vals: Vec<f32> = (kl_low..=kl_high)
                .map(|k| (k as f32) - center as f32)
                .collect();
            let x_eval = ilist as f32 - center as f32;

            for j in 0..6 {
                let y_vals: Vec<f32> = (kl_low..=kl_high).map(|k| nat[k][j]).collect();

                if num_fit <= 2 || order_local < 1 {
                    // Simple average
                    let avg: f32 = y_vals.iter().sum::<f32>() / y_vals.len() as f32;
                    fitted_nat[j] = avg;
                } else {
                    match polynomial_fit(&x_vals, &y_vals, num_fit, order_local) {
                        Ok(fit) => {
                            let mut val = fit.intercept;
                            for p in 0..fit.slopes.len() {
                                val += fit.slopes[p] * x_eval.powi((p + 1) as i32);
                            }
                            fitted_nat[j] = val;
                        }
                        Err(_) => {
                            let avg: f32 = y_vals.iter().sum::<f32>() / y_vals.len() as f32;
                            fitted_nat[j] = avg;
                        }
                    }
                }
            }

            // Hybrid: restore global average for some components
            if nhybrid > 0 {
                let mut global_avg = [0.0f32; 6];
                for n in &nat {
                    for j in 0..6 {
                        global_avg[j] += n[j] / nlist as f32;
                    }
                }

                // nhybrid 1: rotation global, rest local
                // nhybrid 2: translation global, rest local
                // nhybrid 3: rotation + translation global
                // nhybrid 4: rotation + translation + mag global
                if nhybrid > 1 {
                    fitted_nat[4] = global_avg[4]; // dx
                    fitted_nat[5] = global_avg[5]; // dy
                }
                if nhybrid == 1 || nhybrid > 2 {
                    fitted_nat[0] = global_avg[0]; // rotation
                }
                if nhybrid > 3 {
                    fitted_nat[1] = global_avg[1]; // mag
                }
            }

            let fitted_xf = from_natural(&fitted_nat);
            let fitted_inv = fitted_xf.inverse();
            results.push(g[ilist].then(&fitted_inv));
        }

        write_xf_file(&output, &results).unwrap_or_else(|e| {
            eprintln!("ERROR: XFTOXG - writing output: {}", e);
            process::exit(1);
        });
    }

    eprintln!("{} alignment transforms written to {}", nlist, output);
}
