use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};

/// Project a 3D volume along X, Y, or Z at one or more tilt angles.
///
/// Computes projection images by summing voxels along rays through the volume.
/// Supports tilt series around any axis with quadratic interpolation.
#[derive(Parser)]
#[command(name = "xyzproj", about = "Project a volume along X, Y, or Z axis")]
struct Args {
    /// Input MRC file
    input: String,
    /// Output MRC file
    output: String,

    /// Axis to tilt around: X, Y, or Z
    #[arg(short = 'a', long)]
    axis: char,

    /// X range: min,max (0-based, default: full)
    #[arg(long)]
    xminmax: Option<String>,
    /// Y range: min,max (0-based, default: full)
    #[arg(long)]
    yminmax: Option<String>,
    /// Z range: min,max (0-based, default: full)
    #[arg(long)]
    zminmax: Option<String>,

    /// Start, end, increment tilt angles in degrees (comma-separated, e.g. "0,0,1")
    #[arg(short = 't', long, default_value = "0,0,1")]
    angles: String,

    /// Output mode (1=int16, 2=float32)
    #[arg(short = 'm', long, default_value_t = 2)]
    mode: i32,

    /// Scale: add then multiply (comma-separated, e.g. "0,1")
    #[arg(long, default_value = "0,1")]
    scale: String,

    /// Fill value for areas not projected to (default: input mean)
    #[arg(long)]
    fill: Option<f32>,

    /// Use constant scaling (by vertical thickness) instead of by ray length
    #[arg(short = 'c', long)]
    constant: bool,
}

fn parse_range(s: &str) -> (usize, usize) {
    let parts: Vec<&str> = s.split(',').collect();
    (
        parts[0].trim().parse().expect("invalid range"),
        parts[1].trim().parse().expect("invalid range"),
    )
}

fn main() {
    let args = Args::parse();

    let axis = args.axis.to_ascii_uppercase();
    if !['X', 'Y', 'Z'].contains(&axis) {
        eprintln!("Error: axis must be X, Y, or Z");
        std::process::exit(1);
    }

    // Parse angles
    let angle_parts: Vec<f64> = args
        .angles
        .split(',')
        .map(|s| s.trim().parse().expect("invalid angle"))
        .collect();
    let (tilt_start, tilt_end, tilt_inc) = match angle_parts.len() {
        1 => (angle_parts[0], angle_parts[0], 1.0),
        3 => (angle_parts[0], angle_parts[1], angle_parts[2]),
        _ => {
            eprintln!("Error: angles must be 1 or 3 comma-separated values (start,end,inc)");
            std::process::exit(1);
        }
    };

    let scale_parts: Vec<f32> = args
        .scale
        .split(',')
        .map(|s| s.trim().parse().expect("invalid scale"))
        .collect();
    let (scale_add, scale_fac) = (scale_parts[0], scale_parts[1]);

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let in_nx = h.nx as usize;
    let in_ny = h.ny as usize;
    let in_nz = h.nz as usize;

    let (ix0, ix1) = args
        .xminmax
        .as_deref()
        .map_or((0, in_nx - 1), parse_range);
    let (iy0, iy1) = args
        .yminmax
        .as_deref()
        .map_or((0, in_ny - 1), parse_range);
    let (iz0, iz1) = args
        .zminmax
        .as_deref()
        .map_or((0, in_nz - 1), parse_range);

    let nx_block = ix1 + 1 - ix0;
    let ny_block = iy1 + 1 - iy0;
    let nz_block = iz1 + 1 - iz0;

    let fill = args.fill.unwrap_or(h.amean);

    // Determine number of projections
    let n_proj = if tilt_inc != 0.0 {
        ((tilt_end - tilt_start) / tilt_inc) as usize + 1
    } else {
        1
    };

    // Set up output dimensions based on axis
    let (nx_out, ny_out) = match axis {
        'X' => (ny_block, nx_block),
        'Y' => (nx_block, ny_block),
        'Z' => (nx_block, nz_block),
        _ => unreachable!(),
    };
    let nz_out = n_proj;

    // Read the volume subblock
    let mut volume = vec![0.0f32; nx_block * ny_block * nz_block];
    for iz in 0..nz_block {
        let slice = reader.read_slice_f32(iz0 + iz).unwrap();
        for iy in 0..ny_block {
            for ix in 0..nx_block {
                volume[iz * ny_block * nx_block + iy * nx_block + ix] =
                    slice[(iy0 + iy) * in_nx + (ix0 + ix)];
            }
        }
    }

    // Set up output
    let out_mode = if args.mode == 1 {
        MrcMode::Short
    } else {
        MrcMode::Float
    };

    let mut out_header =
        MrcHeader::new(nx_out as i32, ny_out as i32, nz_out as i32, out_mode);
    let delta = [h.pixel_size_x(), h.pixel_size_y(), h.pixel_size_z()];
    // Map scales depend on axis
    let (cell_x, cell_y, cell_z) = match axis {
        'X' => (
            nx_out as f32 * delta[1],
            ny_out as f32 * delta[0],
            nz_out as f32 * delta[1],
        ),
        'Y' => (
            nx_out as f32 * delta[0],
            ny_out as f32 * delta[1],
            nz_out as f32 * delta[0],
        ),
        'Z' => (
            nx_out as f32 * delta[0],
            ny_out as f32 * delta[2],
            nz_out as f32 * delta[0],
        ),
        _ => unreachable!(),
    };
    out_header.xlen = cell_x;
    out_header.ylen = cell_y;
    out_header.zlen = cell_z;
    out_header.mx = nx_out as i32;
    out_header.my = ny_out as i32;
    out_header.mz = nz_out as i32;
    out_header.add_label(&format!(
        "xyzproj: x {}-{}, y {}-{}, z {}-{} about {}",
        ix0, ix1, iy0, iy1, iz0, iz1, axis
    ));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    let scaled_fill = (fill + scale_add) * scale_fac;

    for iproj in 0..n_proj {
        let angle_deg = tilt_start + iproj as f64 * tilt_inc;
        let angle_rad = angle_deg.to_radians();
        let sin_a = angle_rad.sin() as f32;
        let cos_a = angle_rad.cos() as f32;

        let mut out_slice = vec![scaled_fill; nx_out * ny_out];

        match axis {
            'X' => {
                // Tilt around X: project along Z-Y plane
                // Output X = input Y, output Y = input X
                // For each output pixel, sum along rays in Z-Y plane
                project_around_x(
                    &volume,
                    nx_block,
                    ny_block,
                    nz_block,
                    &mut out_slice,
                    nx_out,
                    ny_out,
                    sin_a,
                    cos_a,
                    fill,
                    scale_add,
                    scale_fac,
                    args.constant,
                );
            }
            'Y' => {
                // Tilt around Y: project along X-Z plane
                project_around_y(
                    &volume,
                    nx_block,
                    ny_block,
                    nz_block,
                    &mut out_slice,
                    nx_out,
                    ny_out,
                    sin_a,
                    cos_a,
                    fill,
                    scale_add,
                    scale_fac,
                    args.constant,
                );
            }
            'Z' => {
                // Tilt around Z: project along X-Y plane
                project_around_z(
                    &volume,
                    nx_block,
                    ny_block,
                    nz_block,
                    &mut out_slice,
                    nx_out,
                    ny_out,
                    sin_a,
                    cos_a,
                    fill,
                    scale_add,
                    scale_fac,
                    args.constant,
                );
            }
            _ => unreachable!(),
        }

        let (smin, smax, smean) = min_max_mean(&out_slice);
        gmin = gmin.min(smin);
        gmax = gmax.max(smax);
        gsum += smean as f64 * (nx_out * ny_out) as f64;

        writer.write_slice_f32(&out_slice).unwrap();
    }

    let gmean = (gsum / (nx_out * ny_out * nz_out) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
}

/// Helper to access a voxel from the volume array (Z-major order).
#[inline]
fn voxel(vol: &[f32], nx: usize, ny: usize, ix: usize, iy: usize, iz: usize) -> f32 {
    vol[iz * ny * nx + iy * nx + ix]
}

/// Project around X axis: rays go through the Z-Y plane for each X slice.
/// The "slice" dimension is X (input), output X = input Y, output Y = input X.
fn project_around_x(
    vol: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    out: &mut [f32],
    nx_out: usize,
    ny_out: usize,
    sin_a: f32,
    cos_a: f32,
    fill: f32,
    scale_add: f32,
    scale_fac: f32,
    constant_scale: bool,
) {
    let cy = (nz as f32 - 1.0) / 2.0;
    let cx = (ny as f32 - 1.0) / 2.0;
    let n_ray_max = ((nz as f32).hypot(ny as f32)).ceil() as usize;
    let ray_scale = if constant_scale {
        nz as f32
    } else {
        n_ray_max as f32
    };

    for ox in 0..nx_out {
        // ox corresponds to input Y
        let _iy = ox;
        for oy in 0..ny_out {
            // oy corresponds to input X
            let ix = oy;
            let mut sum = 0.0f32;
            let mut count = 0usize;

            // Cast rays through Z-Y at this angle
            for step in 0..n_ray_max {
                let t = step as f32 - n_ray_max as f32 / 2.0;
                let sz = cy + t * cos_a;
                let sy = cx + t * sin_a;

                if sz >= 0.0 && sz < nz as f32 - 0.5 && sy >= 0.0 && sy < ny as f32 - 0.5 {
                    let iz0 = (sz as usize).min(nz - 2);
                    let iy0 = (sy as usize).min(ny - 2);
                    let dz = sz - iz0 as f32;
                    let dy = sy - iy0 as f32;

                    let v00 = voxel(vol, nx, ny, ix, iy0, iz0);
                    let v10 = voxel(vol, nx, ny, ix, iy0 + 1, iz0);
                    let v01 = voxel(vol, nx, ny, ix, iy0, iz0 + 1);
                    let v11 = voxel(vol, nx, ny, ix, iy0 + 1, iz0 + 1);
                    sum += v00 * (1.0 - dy) * (1.0 - dz)
                        + v10 * dy * (1.0 - dz)
                        + v01 * (1.0 - dy) * dz
                        + v11 * dy * dz;
                    count += 1;
                }
            }

            if count > 0 {
                let ray_fac = scale_fac / ray_scale;
                let ray_add = scale_add * scale_fac
                    + ray_fac * (n_ray_max - count) as f32 * (fill / scale_fac - scale_add);
                out[oy * nx_out + ox] = ray_fac * sum + ray_add;
            }
        }
    }
}

/// Project around Y axis: rays go through the X-Z plane for each Y row.
fn project_around_y(
    vol: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    out: &mut [f32],
    nx_out: usize,
    ny_out: usize,
    sin_a: f32,
    cos_a: f32,
    fill: f32,
    scale_add: f32,
    scale_fac: f32,
    constant_scale: bool,
) {
    let cx = (nx as f32 - 1.0) / 2.0;
    let cz = (nz as f32 - 1.0) / 2.0;
    let n_ray_max = ((nx as f32).hypot(nz as f32)).ceil() as usize;
    let ray_scale = if constant_scale {
        nz as f32
    } else {
        n_ray_max as f32
    };

    for ox in 0..nx_out {
        let _ix_center = ox as f32;
        for oy in 0..ny_out {
            let iy = oy;
            let mut sum = 0.0f32;
            let mut count = 0usize;

            for step in 0..n_ray_max {
                let t = step as f32 - n_ray_max as f32 / 2.0;
                let sx = cx + t * cos_a;
                let sz = cz - t * sin_a; // inverted angle for Y

                if sx >= 0.0 && sx < nx as f32 - 0.5 && sz >= 0.0 && sz < nz as f32 - 0.5 {
                    let ix0 = (sx as usize).min(nx - 2);
                    let iz0 = (sz as usize).min(nz - 2);
                    let dx = sx - ix0 as f32;
                    let dz = sz - iz0 as f32;

                    let v00 = voxel(vol, nx, ny, ix0, iy, iz0);
                    let v10 = voxel(vol, nx, ny, ix0 + 1, iy, iz0);
                    let v01 = voxel(vol, nx, ny, ix0, iy, iz0 + 1);
                    let v11 = voxel(vol, nx, ny, ix0 + 1, iy, iz0 + 1);
                    sum += v00 * (1.0 - dx) * (1.0 - dz)
                        + v10 * dx * (1.0 - dz)
                        + v01 * (1.0 - dx) * dz
                        + v11 * dx * dz;
                    count += 1;
                }
            }

            if count > 0 {
                let ray_fac = scale_fac / ray_scale;
                let ray_add = scale_add * scale_fac
                    + ray_fac * (n_ray_max - count) as f32 * (fill / scale_fac - scale_add);
                out[oy * nx_out + ox] = ray_fac * sum + ray_add;
            }
        }
    }
}

/// Project around Z axis: rays go through the X-Y plane for each Z slice.
fn project_around_z(
    vol: &[f32],
    nx: usize,
    ny: usize,
    _nz: usize,
    out: &mut [f32],
    nx_out: usize,
    ny_out: usize,
    sin_a: f32,
    cos_a: f32,
    fill: f32,
    scale_add: f32,
    scale_fac: f32,
    constant_scale: bool,
) {
    let cx = (nx as f32 - 1.0) / 2.0;
    let cy = (ny as f32 - 1.0) / 2.0;
    let n_ray_max = ((nx as f32).hypot(ny as f32)).ceil() as usize;
    let ray_scale = if constant_scale {
        ny as f32
    } else {
        n_ray_max as f32
    };

    for ox in 0..nx_out {
        for oy in 0..ny_out {
            let iz = oy; // output Y = input Z
            let mut sum = 0.0f32;
            let mut count = 0usize;

            for step in 0..n_ray_max {
                let t = step as f32 - n_ray_max as f32 / 2.0;
                let sx = cx + t * cos_a;
                let sy = cy + t * sin_a;

                if sx >= 0.0 && sx < nx as f32 - 0.5 && sy >= 0.0 && sy < ny as f32 - 0.5 {
                    let ix0 = (sx as usize).min(nx - 2);
                    let iy0 = (sy as usize).min(ny - 2);
                    let dx = sx - ix0 as f32;
                    let dy = sy - iy0 as f32;

                    let v00 = voxel(vol, nx, ny, ix0, iy0, iz);
                    let v10 = voxel(vol, nx, ny, ix0 + 1, iy0, iz);
                    let v01 = voxel(vol, nx, ny, ix0, iy0 + 1, iz);
                    let v11 = voxel(vol, nx, ny, ix0 + 1, iy0 + 1, iz);
                    sum += v00 * (1.0 - dx) * (1.0 - dy)
                        + v10 * dx * (1.0 - dy)
                        + v01 * (1.0 - dx) * dy
                        + v11 * dx * dy;
                    count += 1;
                }
            }

            if count > 0 {
                let ray_fac = scale_fac / ray_scale;
                let ray_add = scale_add * scale_fac
                    + ray_fac * (n_ray_max - count) as f32 * (fill / scale_fac - scale_add);
                out[oy * nx_out + ox] = ray_fac * sum + ray_add;
            }
        }
    }
}
