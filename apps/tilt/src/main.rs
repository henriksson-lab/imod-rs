mod gpu;

use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_transforms::read_tilt_file;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::fs;
use std::io::{self, BufRead};

/// Reconstruct a 3D volume from a tilt series using weighted back-projection.
///
/// Each projection (tilt image) is back-projected along its tilt angle into the
/// output volume. Uses rayon for parallel reconstruction across Y rows.
#[derive(Parser)]
#[command(name = "tilt", about = "Weighted back-projection reconstruction")]
struct Args {
    /// Input aligned tilt series (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Output reconstruction (MRC)
    #[arg(short = 'o', long)]
    output: String,

    /// Tilt angle file (.tlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Thickness of output volume in Z (pixels)
    #[arg(short = 'z', long, default_value_t = 0)]
    thickness: i32,

    /// Width of output (default: same as input)
    #[arg(short = 'w', long)]
    width: Option<i32>,

    /// Number of threads (default: all cores)
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Use GPU-accelerated back-projection (via wgpu compute shaders)
    #[arg(long)]
    gpu: bool,

    /// R-weighting parameters: cutoff,falloff (e.g. "0.35,0.05")
    #[arg(long, default_value = "0.35,0.05")]
    radial: String,

    /// Disable R-weighting filter
    #[arg(long)]
    no_weight: bool,

    /// Number of SIRT iterations (0 = plain WBP)
    #[arg(long, default_value_t = 0)]
    sirt: usize,

    /// SIRT relaxation factor
    #[arg(long, default_value_t = 1.0)]
    sirt_relax: f32,

    /// Disable cosine stretching of projections
    #[arg(long)]
    no_cosine_stretch: bool,

    /// Local alignment file (patch-based corrections: view x y dx dy per line)
    #[arg(long)]
    local_file: Option<String>,

    /// X-axis tilt file (one angle in degrees per view)
    #[arg(long)]
    xtilt_file: Option<String>,

    /// Per-view Z scaling factor file (one factor per view)
    #[arg(long)]
    zfactor_file: Option<String>,

    /// Apply log10 transform to projection pixel values before reconstruction
    #[arg(long)]
    log: bool,

    /// Density weight factor applied to output voxels (default 1.0)
    #[arg(long, default_value_t = 1.0)]
    densweight: f32,

    /// Output scale factor applied after density weighting (default 1.0)
    #[arg(long, default_value_t = 1.0)]
    outscale: f32,

    /// Output additive offset applied after scaling (default 0.0)
    #[arg(long, default_value_t = 0.0)]
    outadd: f32,

    /// Use exactly N points for the radial filter table instead of deriving from FFT size
    #[arg(long)]
    exact_filter_size: Option<usize>,

    /// Fill mode for out-of-bounds projection samples: "mean", "zero", or "edge"
    #[arg(long, default_value = "mean")]
    fill_mode: String,
}

/// Fill mode for out-of-bounds projection samples.
#[derive(Clone, Copy, PartialEq)]
enum FillMode {
    Mean,
    Zero,
    Edge,
}

/// Precomputed per-projection constants to avoid recomputing in the inner loop.
struct ProjParams {
    cos_t: f32,
    sin_t: f32,
    /// Cosine of the X-axis tilt for this view (1.0 if no xtilt)
    cos_xt: f32,
    /// Sine of the X-axis tilt for this view (0.0 if no xtilt)
    sin_xt: f32,
    /// Z scaling factor for this view (1.0 if no zfactor)
    zfactor: f32,
}

/// A single local alignment patch correction.
#[derive(Clone, Debug)]
struct LocalPatch {
    view: usize,
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
}

/// Per-view local alignment data organized as a grid for bilinear interpolation.
struct LocalAlignGrid {
    /// Patches organized by view index. Each view has a sorted grid of patches.
    views: Vec<ViewPatches>,
}

/// Patches for a single view, organized as a grid.
struct ViewPatches {
    /// Sorted unique X positions of patches
    xs: Vec<f32>,
    /// Sorted unique Y positions of patches
    ys: Vec<f32>,
    /// dx corrections, stored row-major [iy * nx + ix]
    dx: Vec<f32>,
    /// dy corrections, stored row-major [iy * nx + ix]
    dy: Vec<f32>,
}

impl ViewPatches {
    /// Bilinear interpolation of (dx, dy) at position (x, y).
    fn interpolate(&self, x: f32, y: f32) -> (f32, f32) {
        if self.xs.is_empty() || self.ys.is_empty() {
            return (0.0, 0.0);
        }
        let nx = self.xs.len();

        // Find bracketing X index
        let (ix0, fx) = Self::find_bracket(&self.xs, x);
        // Find bracketing Y index
        let (iy0, fy) = Self::find_bracket(&self.ys, y);

        let ix1 = (ix0 + 1).min(nx - 1);
        let ny = self.ys.len();
        let iy1 = (iy0 + 1).min(ny - 1);

        // Four corners
        let d00 = (self.dx[iy0 * nx + ix0], self.dy[iy0 * nx + ix0]);
        let d10 = (self.dx[iy0 * nx + ix1], self.dy[iy0 * nx + ix1]);
        let d01 = (self.dx[iy1 * nx + ix0], self.dy[iy1 * nx + ix0]);
        let d11 = (self.dx[iy1 * nx + ix1], self.dy[iy1 * nx + ix1]);

        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

        let dx = lerp(lerp(d00.0, d10.0, fx), lerp(d01.0, d11.0, fx), fy);
        let dy = lerp(lerp(d00.1, d10.1, fx), lerp(d01.1, d11.1, fx), fy);
        (dx, dy)
    }

    fn find_bracket(positions: &[f32], val: f32) -> (usize, f32) {
        if positions.len() == 1 {
            return (0, 0.0);
        }
        // Binary search for the interval
        let mut lo = 0usize;
        let mut hi = positions.len() - 1;
        if val <= positions[lo] {
            return (0, 0.0);
        }
        if val >= positions[hi] {
            return (hi, 0.0);
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if positions[mid] <= val {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = positions[hi] - positions[lo];
        let frac = if span > 1e-12 {
            (val - positions[lo]) / span
        } else {
            0.0
        };
        (lo, frac)
    }
}

/// Load local alignment patches from file and organize by view.
fn load_local_alignment(path: &str, nz: usize) -> io::Result<LocalAlignGrid> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut patches: Vec<LocalPatch> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let view: usize = parts[0].parse().unwrap_or(0);
        let x: f32 = parts[1].parse().unwrap_or(0.0);
        let y: f32 = parts[2].parse().unwrap_or(0.0);
        let dx: f32 = parts[3].parse().unwrap_or(0.0);
        let dy: f32 = parts[4].parse().unwrap_or(0.0);
        patches.push(LocalPatch { view, x, y, dx, dy });
    }

    // Organize by view
    let mut views: Vec<ViewPatches> = (0..nz)
        .map(|_| ViewPatches {
            xs: Vec::new(),
            ys: Vec::new(),
            dx: Vec::new(),
            dy: Vec::new(),
        })
        .collect();

    // Group patches by view
    let mut by_view: Vec<Vec<&LocalPatch>> = vec![Vec::new(); nz];
    for p in &patches {
        if p.view < nz {
            by_view[p.view].push(p);
        }
    }

    for (vi, vpatches) in by_view.iter().enumerate() {
        if vpatches.is_empty() {
            continue;
        }
        // Collect unique sorted X and Y positions
        let mut xs: Vec<f32> = vpatches.iter().map(|p| p.x).collect();
        let mut ys: Vec<f32> = vpatches.iter().map(|p| p.y).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup_by(|a, b| (*a - *b).abs() < 0.01);
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.dedup_by(|a, b| (*a - *b).abs() < 0.01);

        let nx = xs.len();
        let ny = ys.len();
        let mut dx_grid = vec![0.0f32; nx * ny];
        let mut dy_grid = vec![0.0f32; nx * ny];

        // Place each patch into the grid
        for p in vpatches {
            let ix = xs.iter().position(|&xv| (xv - p.x).abs() < 0.01).unwrap_or(0);
            let iy = ys.iter().position(|&yv| (yv - p.y).abs() < 0.01).unwrap_or(0);
            dx_grid[iy * nx + ix] = p.dx;
            dy_grid[iy * nx + ix] = p.dy;
        }

        views[vi] = ViewPatches {
            xs,
            ys,
            dx: dx_grid,
            dy: dy_grid,
        };
    }

    Ok(LocalAlignGrid { views })
}

/// Load a file with one floating-point value per line (for xtilt or zfactor).
fn load_per_view_file(path: &str) -> io::Result<Vec<f32>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let val: f32 = line.parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("parse error: {}", e))
        })?;
        values.push(val);
    }
    Ok(values)
}

/// Apply radial (r-) weighting to projection data in-place.
/// Each row is FFT'd, multiplied by a ramp filter with Gaussian rolloff, then inverse FFT'd.
/// If `exact_filter_size` is Some(N), the radial filter table uses exactly N points
/// instead of deriving the number of points from the FFT size.
fn apply_rweighting(proj_data: &mut [f32], in_nx: usize, in_ny: usize, nz: usize, cutoff: f32, falloff: f32, exact_filter_size: Option<usize>) {
    let total_rows = nz * in_ny;
    assert_eq!(proj_data.len(), total_rows * in_nx);

    // Process rows in parallel
    proj_data.par_chunks_mut(in_nx).for_each(|row| {
        let n = row.len();
        let mut planner = FftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(n);
        let fft_inv = planner.plan_fft_inverse(n);

        let mut buffer: Vec<Complex<f32>> = row.iter().map(|&v| Complex { re: v, im: 0.0 }).collect();

        fft_fwd.process(&mut buffer);

        // Number of points for the radial filter table
        let filter_nxc = exact_filter_size.unwrap_or(n / 2 + 1);
        // Apply ramp filter with Gaussian rolloff to both halves of the spectrum
        for i in 0..n {
            let freq_bin = if i <= n / 2 { i } else { n - i };
            let w = (freq_bin as f32) / (filter_nxc as f32);
            let weight = if w <= cutoff {
                w
            } else {
                let d = w - cutoff;
                w * (-d * d / (2.0 * falloff * falloff)).exp()
            };
            buffer[i].re *= weight;
            buffer[i].im *= weight;
        }

        fft_inv.process(&mut buffer);

        let inv_n = 1.0 / n as f32;
        for (i, v) in row.iter_mut().enumerate() {
            *v = buffer[i].re * inv_n;
        }
    });
}

/// Sample a projection row at fractional position proj_x, with fill mode for out-of-bounds.
/// Returns the interpolated value. `row` is the projection row data of length `row_len`.
#[inline]
fn sample_row_with_fill(row: &[f32], proj_x: f32, row_len: usize, fill_mode: FillMode, proj_mean: f32) -> f32 {
    let px0 = proj_x.floor() as isize;
    if px0 >= 0 && px0 + 1 < row_len as isize {
        let frac = proj_x - px0 as f32;
        row[px0 as usize] * (1.0 - frac) + row[px0 as usize + 1] * frac
    } else {
        match fill_mode {
            FillMode::Zero => 0.0,
            FillMode::Mean => proj_mean,
            FillMode::Edge => {
                if px0 < 0 {
                    row[0]
                } else if row_len > 0 {
                    row[row_len - 1]
                } else {
                    0.0
                }
            }
        }
    }
}

/// Apply cosine stretching to each projection: stretch each row by 1/cos(tilt_angle).
fn apply_cosine_stretch(proj_data: &mut [f32], in_nx: usize, in_ny: usize, nz: usize, tilt_angles: &[f32]) {
    let center = in_nx as f32 / 2.0;

    for p in 0..nz {
        let rad = tilt_angles[p] * PI / 180.0;
        let cos_t = rad.cos();
        if cos_t.abs() < 1e-6 {
            continue; // skip near-90-degree tilts
        }
        let stretch = 1.0 / cos_t;

        let proj_offset = p * in_nx * in_ny;

        // Process each row
        for iy in 0..in_ny {
            let row_offset = proj_offset + iy * in_nx;
            let orig_row: Vec<f32> = proj_data[row_offset..row_offset + in_nx].to_vec();
            let out_row = &mut proj_data[row_offset..row_offset + in_nx];

            for ox in 0..in_nx {
                let dx = ox as f32 - center;
                let src_x = dx * stretch + center;
                let sx0 = src_x.floor() as isize;
                if sx0 >= 0 && sx0 + 1 < in_nx as isize {
                    let frac = src_x - sx0 as f32;
                    out_row[ox] = orig_row[sx0 as usize] * (1.0 - frac) + orig_row[sx0 as usize + 1] * frac;
                } else if sx0 >= 0 && sx0 < in_nx as isize {
                    out_row[ox] = orig_row[sx0 as usize];
                } else {
                    out_row[ox] = 0.0;
                }
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    let tilt_angles = read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let nz = h.nz as usize;

    let out_nx = args.width.unwrap_or(h.nx) as usize;
    let out_ny = in_ny;
    let out_nz = if args.thickness > 0 {
        args.thickness as usize
    } else {
        in_nx
    };

    let n_threads = rayon::current_num_threads();
    eprintln!(
        "tilt: {} projections, {} x {} -> {} x {} x {} ({} threads)",
        nz, in_nx, in_ny, out_nx, out_ny, out_nz, n_threads
    );

    // Parse radial weighting parameters
    let (radial_cutoff, radial_falloff) = {
        let parts: Vec<&str> = args.radial.split(',').collect();
        let cutoff: f32 = parts.get(0).and_then(|s| s.trim().parse().ok()).unwrap_or(0.35);
        let falloff: f32 = parts.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(0.05);
        (cutoff, falloff)
    };
    let sirt_iters = args.sirt;
    let sirt_relax = args.sirt_relax;

    // Parse fill mode
    let fill_mode = match args.fill_mode.to_lowercase().as_str() {
        "zero" => FillMode::Zero,
        "edge" => FillMode::Edge,
        "mean" | _ => FillMode::Mean,
    };

    // Output transform parameters
    let densweight = args.densweight;
    let outscale = args.outscale;
    let outadd = args.outadd;

    // Read all projections into a flat buffer for cache-friendly access
    let mut proj_data: Vec<f32> = Vec::with_capacity(nz * in_nx * in_ny);
    for z in 0..nz {
        proj_data.extend_from_slice(&reader.read_slice_f32(z).unwrap());
    }

    // Apply log10 transform to projections if requested
    if args.log {
        eprintln!("tilt: applying log10 transform to projections");
        for v in proj_data.iter_mut() {
            *v = v.max(0.001).log10();
        }
    }

    // Compute per-projection means for fill_mode="mean"
    let proj_means: Vec<f32> = (0..nz)
        .map(|p| {
            let offset = p * in_nx * in_ny;
            let slice = &proj_data[offset..offset + in_nx * in_ny];
            let (_, _, mean) = min_max_mean(slice);
            mean
        })
        .collect();

    // Keep unfiltered copy for SIRT forward projection
    let proj_data_orig = if sirt_iters > 0 {
        Some(proj_data.clone())
    } else {
        None
    };

    // Apply R-weighting (radial filter) to projections
    if !args.no_weight {
        eprintln!("tilt: applying R-weighting (cutoff={}, falloff={})", radial_cutoff, radial_falloff);
        apply_rweighting(&mut proj_data, in_nx, in_ny, nz, radial_cutoff, radial_falloff, args.exact_filter_size);
    }

    // Apply cosine stretching
    if !args.no_cosine_stretch {
        eprintln!("tilt: applying cosine stretch");
        apply_cosine_stretch(&mut proj_data, in_nx, in_ny, nz, &tilt_angles);
    }

    // Load optional local alignment
    let local_align: Option<LocalAlignGrid> = args.local_file.as_ref().map(|path| {
        eprintln!("tilt: loading local alignment from {}", path);
        load_local_alignment(path, nz).unwrap_or_else(|e| {
            eprintln!("Error reading local alignment file: {}", e);
            std::process::exit(1);
        })
    });

    // Load optional X-axis tilt angles
    let xtilt_angles: Option<Vec<f32>> = args.xtilt_file.as_ref().map(|path| {
        eprintln!("tilt: loading X-axis tilt from {}", path);
        load_per_view_file(path).unwrap_or_else(|e| {
            eprintln!("Error reading xtilt file: {}", e);
            std::process::exit(1);
        })
    });

    // Load optional Z-factor file
    let zfactors: Option<Vec<f32>> = args.zfactor_file.as_ref().map(|path| {
        eprintln!("tilt: loading Z-factors from {}", path);
        load_per_view_file(path).unwrap_or_else(|e| {
            eprintln!("Error reading zfactor file: {}", e);
            std::process::exit(1);
        })
    });

    // Precompute trig for each projection (including xtilt and zfactor)
    let proj_params: Vec<ProjParams> = tilt_angles
        .iter()
        .take(nz)
        .enumerate()
        .map(|(i, &deg)| {
            let rad = deg * PI / 180.0;
            let xt_deg = xtilt_angles.as_ref().map(|v| v.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0);
            let xt_rad = xt_deg * PI / 180.0;
            let zf = zfactors.as_ref().map(|v| v.get(i).copied().unwrap_or(1.0)).unwrap_or(1.0);
            ProjParams {
                cos_t: rad.cos(),
                sin_t: rad.sin(),
                cos_xt: xt_rad.cos(),
                sin_xt: xt_rad.sin(),
                zfactor: zf,
            }
        })
        .collect();

    let center_x = in_nx as f32 / 2.0;
    let center_z = out_nz as f32 / 2.0;
    let out_center_x = out_nx as f32 / 2.0;
    let inv_n = 1.0 / nz as f32;

    // Parallel reconstruction: each Y row is independent
    // Process in chunks for progress reporting
    let chunk_size = 32.max(out_ny / 20);
    let mut out_header = MrcHeader::new(out_nx as i32, out_nz as i32, out_ny as i32, MrcMode::Float);
    out_header.xlen = h.xlen * out_nx as f32 / in_nx as f32;
    out_header.ylen = h.xlen * out_nz as f32 / in_nx as f32;
    out_header.zlen = h.ylen;
    out_header.mx = out_nx as i32;
    out_header.my = out_nz as i32;
    out_header.mz = out_ny as i32;
    out_header.add_label(&format!("tilt: {} projections, thickness {}", nz, out_nz));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;
    let mut rows_done = 0usize;

    // Try GPU path if requested
    let tilt_cos_sin: Vec<(f32, f32)> = proj_params
        .iter()
        .map(|pp| (pp.cos_t, pp.sin_t))
        .collect();

    let gpu_session = if args.gpu {
        match gpu::GpuSession::new(
            &proj_data,
            &tilt_cos_sin,
            in_nx,
            in_ny,
            nz,
            out_nx,
            out_nz,
        ) {
            Some(session) => {
                eprintln!("tilt: using GPU back-projection");
                Some(session)
            }
            None => {
                eprintln!("tilt: WARNING: GPU init failed, falling back to CPU");
                None
            }
        }
    } else {
        None
    };

    // Helper closure: backproject filtered proj_data into volume slices (one per Y row)
    let backproject_all = |bp_proj: &[f32]| -> Vec<Vec<f32>> {
        (0..out_ny)
            .into_par_iter()
            .map(|iy| {
                let mut slice = vec![0.0f32; out_nx * out_nz];
                for (pi, pp) in proj_params.iter().enumerate() {
                    let _center_y = in_ny as f32 / 2.0;
                    for oz in 0..out_nz {
                        let dz = oz as f32 - center_z;
                        let z_contrib = dz * pp.sin_t * pp.zfactor;
                        let slice_row = oz * out_nx;
                        for ox in 0..out_nx {
                            let dx = ox as f32 - out_center_x;
                            let mut proj_x = dx * pp.cos_t * pp.cos_xt + z_contrib + center_x;
                            let proj_y_offset = dx * pp.sin_xt;

                            // Apply local alignment correction
                            if let Some(ref la) = local_align {
                                let vp = &la.views[pi];
                                if !vp.xs.is_empty() {
                                    let (ldx, _ldy) = vp.interpolate(ox as f32, iy as f32);
                                    proj_x += ldx;
                                    // Note: ldy would shift the Y sampling, but for the
                                    // row-based approach we already selected iy. The local
                                    // dy is applied to proj_y_offset below if needed.
                                }
                            }

                            // Determine which projection row to sample
                            let sample_y = iy as f32 + proj_y_offset;
                            // Also add local dy if available
                            let sample_y = if let Some(ref la) = local_align {
                                let vp = &la.views[pi];
                                if !vp.xs.is_empty() {
                                    let (_ldx, ldy) = vp.interpolate(ox as f32, iy as f32);
                                    sample_y + ldy
                                } else {
                                    sample_y
                                }
                            } else {
                                sample_y
                            };

                            // Bilinear sampling from projection
                            let sy0 = sample_y.floor() as isize;
                            let sy_frac = sample_y - sy0 as f32;
                            let pm = proj_means[pi];

                            if sy0 >= 0 && sy0 + 1 < in_ny as isize && (proj_y_offset.abs() > 1e-6 || local_align.is_some()) {
                                let row0_offset = pi * in_nx * in_ny + sy0 as usize * in_nx;
                                let row1_offset = pi * in_nx * in_ny + (sy0 as usize + 1) * in_nx;
                                let row0 = &bp_proj[row0_offset..row0_offset + in_nx];
                                let row1 = &bp_proj[row1_offset..row1_offset + in_nx];
                                let v0 = sample_row_with_fill(row0, proj_x, in_nx, fill_mode, pm);
                                let v1 = sample_row_with_fill(row1, proj_x, in_nx, fill_mode, pm);
                                let v = v0 * (1.0 - sy_frac) + v1 * sy_frac;
                                slice[slice_row + ox] += v;
                            } else if sy0 >= 0 && sy0 < in_ny as isize {
                                let row_offset = pi * in_nx * in_ny + sy0 as usize * in_nx;
                                let row = &bp_proj[row_offset..row_offset + in_nx];
                                let v = sample_row_with_fill(row, proj_x, in_nx, fill_mode, pm);
                                slice[slice_row + ox] += v;
                            }
                        }
                    }
                }
                for v in &mut slice {
                    *v *= inv_n;
                }
                slice
            })
            .collect()
    };

    // Helper: forward-project volume slices back into sinogram space
    let forward_project = |volume: &[Vec<f32>]| -> Vec<f32> {
        let mut sino = vec![0.0f32; nz * in_nx * in_ny];
        // For each Y row (= volume slice index)
        for (iy, slice) in volume.iter().enumerate() {
            for (pi, pp) in proj_params.iter().enumerate() {
                for oz in 0..out_nz {
                    let dz = oz as f32 - center_z;
                    let z_contrib = dz * pp.sin_t * pp.zfactor;
                    let slice_row = oz * out_nx;
                    for ox in 0..out_nx {
                        let dx = ox as f32 - out_center_x;
                        let mut proj_x = dx * pp.cos_t * pp.cos_xt + z_contrib + center_x;
                        let proj_y_offset = dx * pp.sin_xt;

                        // Apply local alignment correction
                        if let Some(ref la) = local_align {
                            let vp = &la.views[pi];
                            if !vp.xs.is_empty() {
                                let (ldx, _ldy) = vp.interpolate(ox as f32, iy as f32);
                                proj_x += ldx;
                            }
                        }

                        let sample_y = iy as f32 + proj_y_offset;
                        let sample_y = if let Some(ref la) = local_align {
                            let vp = &la.views[pi];
                            if !vp.xs.is_empty() {
                                let (_ldx, ldy) = vp.interpolate(ox as f32, iy as f32);
                                sample_y + ldy
                            } else {
                                sample_y
                            }
                        } else {
                            sample_y
                        };

                        let px0 = proj_x.floor() as isize;
                        if px0 >= 0 && px0 + 1 < in_nx as isize {
                            let frac = proj_x - px0 as f32;
                            let px0u = px0 as usize;
                            let val = slice[slice_row + ox];

                            // Distribute to the nearest Y row(s)
                            let sy0 = sample_y.floor() as isize;
                            let sy_frac = sample_y - sy0 as f32;
                            if sy0 >= 0 && sy0 + 1 < in_ny as isize && (proj_y_offset.abs() > 1e-6 || local_align.is_some()) {
                                let row0 = pi * in_nx * in_ny + sy0 as usize * in_nx;
                                let row1 = pi * in_nx * in_ny + (sy0 as usize + 1) * in_nx;
                                let w0 = 1.0 - sy_frac;
                                let w1 = sy_frac;
                                sino[row0 + px0u] += val * (1.0 - frac) * w0;
                                sino[row0 + px0u + 1] += val * frac * w0;
                                sino[row1 + px0u] += val * (1.0 - frac) * w1;
                                sino[row1 + px0u + 1] += val * frac * w1;
                            } else if sy0 >= 0 && sy0 < in_ny as isize {
                                let row_offset = pi * in_nx * in_ny + sy0 as usize * in_nx;
                                sino[row_offset + px0u] += val * (1.0 - frac);
                                sino[row_offset + px0u + 1] += val * frac;
                            }
                        }
                    }
                }
            }
        }
        sino
    };

    if let Some(ref session) = gpu_session {
        // GPU path: dispatch one row at a time (no SIRT support on GPU)
        for iy in 0..out_ny {
            let mut slice = session.reconstruct_row(iy);

            // Apply output transform: (val * densweight) * outscale + outadd
            for v in slice.iter_mut() {
                *v = (*v * densweight) * outscale + outadd;
            }

            let (smin, smax, smean) = min_max_mean(&slice);
            if smin < gmin { gmin = smin; }
            if smax > gmax { gmax = smax; }
            gsum += smean as f64 * (out_nx * out_nz) as f64;
            writer.write_slice_f32(&slice).unwrap();

            rows_done += 1;
            if rows_done % chunk_size == 0 || rows_done == out_ny {
                eprintln!("  {}/{} rows (GPU)", rows_done, out_ny);
            }
        }
    } else if sirt_iters > 0 {
        // SIRT reconstruction
        let proj_orig = proj_data_orig.as_ref().unwrap();

        // Initial WBP
        eprintln!("tilt: initial WBP for SIRT...");
        let mut volume = backproject_all(&proj_data);
        eprintln!("tilt: starting {} SIRT iterations (relax={})", sirt_iters, sirt_relax);

        for iter in 0..sirt_iters {
            // Forward project current volume
            let mut reproj = forward_project(&volume);

            // Compute difference: diff = original - reprojected
            for (d, &o) in reproj.iter_mut().zip(proj_orig.iter()) {
                *d = o - *d;
            }

            // R-weight the difference
            if !args.no_weight {
                apply_rweighting(&mut reproj, in_nx, in_ny, nz, radial_cutoff, radial_falloff, args.exact_filter_size);
            }

            // Backproject the difference
            let delta = backproject_all(&reproj);

            // Update volume with relaxation
            for (vol_slice, delta_slice) in volume.iter_mut().zip(delta.iter()) {
                for (v, &d) in vol_slice.iter_mut().zip(delta_slice.iter()) {
                    *v += sirt_relax * d;
                }
            }

            eprintln!("  SIRT iteration {}/{}", iter + 1, sirt_iters);
        }

        // Write out the volume
        for slice in &mut volume {
            // Apply output transform: (val * densweight) * outscale + outadd
            for v in slice.iter_mut() {
                *v = (*v * densweight) * outscale + outadd;
            }
            let (smin, smax, smean) = min_max_mean(slice);
            if smin < gmin { gmin = smin; }
            if smax > gmax { gmax = smax; }
            gsum += smean as f64 * (out_nx * out_nz) as f64;
            writer.write_slice_f32(slice).unwrap();
        }
    } else {
        // CPU path: parallel WBP reconstruction with rayon
        for chunk_start in (0..out_ny).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(out_ny);
            let chunk_rows: Vec<usize> = (chunk_start..chunk_end).collect();

            let slices: Vec<Vec<f32>> = chunk_rows
                .par_iter()
                .map(|&iy| {
                    let mut slice = vec![0.0f32; out_nx * out_nz];

                    for (pi, pp) in proj_params.iter().enumerate() {
                        for oz in 0..out_nz {
                            let dz = oz as f32 - center_z;
                            let z_contrib = dz * pp.sin_t * pp.zfactor;
                            let slice_row = oz * out_nx;

                            for ox in 0..out_nx {
                                let dx = ox as f32 - out_center_x;
                                let mut proj_x = dx * pp.cos_t * pp.cos_xt + z_contrib + center_x;
                                let proj_y_offset = dx * pp.sin_xt;

                                // Apply local alignment correction
                                if let Some(ref la) = local_align {
                                    let vp = &la.views[pi];
                                    if !vp.xs.is_empty() {
                                        let (ldx, _ldy) = vp.interpolate(ox as f32, iy as f32);
                                        proj_x += ldx;
                                    }
                                }

                                // Determine which projection row to sample
                                let sample_y = iy as f32 + proj_y_offset;
                                let sample_y = if let Some(ref la) = local_align {
                                    let vp = &la.views[pi];
                                    if !vp.xs.is_empty() {
                                        let (_ldx, ldy) = vp.interpolate(ox as f32, iy as f32);
                                        sample_y + ldy
                                    } else {
                                        sample_y
                                    }
                                } else {
                                    sample_y
                                };

                                let sy0 = sample_y.floor() as isize;
                                let sy_frac = sample_y - sy0 as f32;
                                let pm = proj_means[pi];

                                if sy0 >= 0 && sy0 + 1 < in_ny as isize && (proj_y_offset.abs() > 1e-6 || local_align.is_some()) {
                                    let row0_offset = pi * in_nx * in_ny + sy0 as usize * in_nx;
                                    let row1_offset = pi * in_nx * in_ny + (sy0 as usize + 1) * in_nx;
                                    let row0 = &proj_data[row0_offset..row0_offset + in_nx];
                                    let row1 = &proj_data[row1_offset..row1_offset + in_nx];
                                    let v0 = sample_row_with_fill(row0, proj_x, in_nx, fill_mode, pm);
                                    let v1 = sample_row_with_fill(row1, proj_x, in_nx, fill_mode, pm);
                                    let v = v0 * (1.0 - sy_frac) + v1 * sy_frac;
                                    slice[slice_row + ox] += v;
                                } else if sy0 >= 0 && sy0 < in_ny as isize {
                                    let row_offset = pi * in_nx * in_ny + sy0 as usize * in_nx;
                                    let row = &proj_data[row_offset..row_offset + in_nx];
                                    let v = sample_row_with_fill(row, proj_x, in_nx, fill_mode, pm);
                                    slice[slice_row + ox] += v;
                                }
                            }
                        }
                    }

                    for v in &mut slice {
                        *v *= inv_n;
                    }
                    slice
                })
                .collect();

            for slice in &slices {
                // Apply output transform: (val * densweight) * outscale + outadd
                let transformed: Vec<f32> = slice.iter().map(|&v| (v * densweight) * outscale + outadd).collect();
                let (smin, smax, smean) = min_max_mean(&transformed);
                if smin < gmin { gmin = smin; }
                if smax > gmax { gmax = smax; }
                gsum += smean as f64 * (out_nx * out_nz) as f64;
                writer.write_slice_f32(&transformed).unwrap();
            }

            rows_done += chunk_end - chunk_start;
            eprintln!("  {}/{} rows", rows_done, out_ny);
        }
    }

    let gmean = (gsum / (out_nx * out_nz * out_ny) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
    eprintln!("tilt: reconstruction complete -> {}", args.output);
}
