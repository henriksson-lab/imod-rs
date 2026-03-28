use clap::Parser;
use imod_core::MrcMode;
use imod_math::{min_max_mean, min_max_mean_sd};
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;
use imod_transforms::{read_xf_file, LinearTransform};
use std::f32::consts::PI;
use std::io::Read as IoRead;

/// Create a new image stack from sections of one or more input files.
///
/// Supports reordering sections, applying 2D transforms, mode conversion,
/// binning, scaling, antialias filtering, and float density modes.
#[derive(Parser)]
#[command(name = "newstack", about = "Create new MRC stack from input sections")]
struct Args {
    /// Input MRC file(s)
    #[arg(short = 'i', long = "input", required = true)]
    input: Vec<String>,

    /// Output MRC file
    #[arg(short = 'o', long = "output", required = true)]
    output: String,

    /// List of sections to read (comma-separated, 0-based, e.g. "0,3-5,10")
    #[arg(short = 'S', long)]
    secs: Option<String>,

    /// Output data mode (0=byte, 1=short, 2=float, 6=ushort)
    #[arg(short = 'm', long)]
    mode: Option<i32>,

    /// Binning factor
    #[arg(short = 'b', long, default_value_t = 1)]
    bin: usize,

    /// Transform file (.xf) to apply to each section
    #[arg(short = 'x', long)]
    xform: Option<String>,

    /// Fill value for areas outside the image after transforms
    #[arg(short = 'f', long, default_value_t = 0.0)]
    fill: f32,

    /// Scale min and max for output (min,max)
    #[arg(short = 's', long)]
    scale: Option<String>,

    /// Float densities to range (scale output to fill byte range)
    #[arg(short = 'F', long)]
    float_densities: bool,

    /// Float density mode: 0=scale to fill range, 1=shift to common mean,
    /// 2=scale to common mean and SD
    #[arg(long = "float", default_value_t = -1)]
    float_mode: i32,

    /// Enable antialias filtering before binning (default: true when bin > 1)
    #[arg(long = "antialias")]
    antialias: Option<bool>,

    /// Piece list file: each line has "x y z" coordinates for assembling tiles.
    /// Each input section is a tile placed at the given (x,y) position in the
    /// output for output section z. Overlapping regions use linear blending.
    #[arg(long = "piece-list")]
    piece_list: Option<String>,

    /// Taper fill edges by N pixels using a cosine ramp from the fill value
    /// to the image edge, reducing hard boundaries after transform application.
    #[arg(long = "taper", default_value_t = 0)]
    taper: usize,

    /// Distortion field file (binary). Contains a header (3x i32: grid_nx, grid_ny,
    /// spacing) followed by grid_nx*grid_ny pairs of (dx, dy) as f32. The distortion
    /// correction is applied to each pixel before interpolation.
    #[arg(long = "distort")]
    distort: Option<String>,

    /// Rotate each section by the given angle in degrees (applied after transform,
    /// before binning) using bicubic interpolation.
    #[arg(long = "rotate")]
    rotate: Option<f32>,

    /// Expand (magnify) each section by the given factor using bicubic interpolation.
    /// Output dimensions are multiplied by this factor.
    #[arg(long = "expand")]
    expand: Option<f32>,

    /// Shrink (reduce) each section by the given factor using Lanczos-2 interpolation.
    /// Output dimensions are divided by this factor.
    #[arg(long = "shrink")]
    shrink: Option<f32>,

    /// Extract a rectangular sub-region from each section: x0,y0,width,height
    /// (0-based pixel coordinates, applied before binning).
    #[arg(long = "subarea")]
    subarea: Option<String>,
}

fn parse_section_list(s: &str, max_z: usize) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.trim().parse().unwrap_or(0);
            let end: usize = b.trim().parse().unwrap_or(max_z - 1);
            for i in start..=end.min(max_z - 1) {
                result.push(i);
            }
        } else if let Ok(n) = part.parse::<usize>() {
            if n < max_z {
                result.push(n);
            }
        }
    }
    result
}

/// Lanczos-2 kernel: sinc(x) * sinc(x/2), windowed to |x| <= 2.
fn lanczos2(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= 2.0 {
        return 0.0;
    }
    let px = PI * x;
    let sinc_x = px.sin() / px;
    let sinc_x2 = (px / 2.0).sin() / (px / 2.0);
    sinc_x * sinc_x2
}

/// Cubic interpolation kernel value for 4 samples and fractional position t in [0,1].
fn cubic_interp(v0: f32, v1: f32, v2: f32, v3: f32, t: f32) -> f32 {
    let a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
    let b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
    let c = -0.5 * v0 + 0.5 * v2;
    let d = v1;
    a * t * t * t + b * t * t + c * t + d
}

/// Sample a 2D image at (sx, sy) using separable bicubic (4x4) interpolation.
/// Clamps to image boundaries; returns `fill` if completely out of range.
fn sample_bicubic(data: &[f32], nx: usize, ny: usize, sx: f32, sy: f32, fill: f32) -> f32 {
    let ix = sx.floor() as isize;
    let iy = sy.floor() as isize;
    let fx = sx - sx.floor();
    let fy = sy - sy.floor();

    // Need samples at ix-1..ix+2, iy-1..iy+2
    if ix + 2 < 0 || ix - 1 >= nx as isize || iy + 2 < 0 || iy - 1 >= ny as isize {
        return fill;
    }

    let get = |x: isize, y: isize| -> f32 {
        let cx = x.clamp(0, nx as isize - 1) as usize;
        let cy = y.clamp(0, ny as isize - 1) as usize;
        data[cy * nx + cx]
    };

    // Interpolate 4 rows horizontally, then interpolate vertically
    let mut col_vals = [0.0f32; 4];
    for j in 0..4 {
        let row_y = iy - 1 + j as isize;
        let r0 = get(ix - 1, row_y);
        let r1 = get(ix, row_y);
        let r2 = get(ix + 1, row_y);
        let r3 = get(ix + 2, row_y);
        col_vals[j] = cubic_interp(r0, r1, r2, r3, fx);
    }
    cubic_interp(col_vals[0], col_vals[1], col_vals[2], col_vals[3], fy)
}

/// Rotate image data by `angle` degrees around its center using bicubic interpolation.
fn apply_rotation(data: &[f32], nx: usize, ny: usize, angle_deg: f32, fill: f32) -> Vec<f32> {
    let rad = -angle_deg * PI / 180.0; // negative: rotate output coords back to source
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let cx = nx as f32 / 2.0;
    let cy = ny as f32 / 2.0;

    let mut out = vec![fill; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let sx = cos_a * dx + sin_a * dy + cx;
            let sy = -sin_a * dx + cos_a * dy + cy;
            out[y * nx + x] = sample_bicubic(data, nx, ny, sx, sy, fill);
        }
    }
    out
}

/// Expand (magnify) image using bicubic interpolation. Returns new data and (out_nx, out_ny).
fn apply_expand(data: &[f32], nx: usize, ny: usize, factor: f32, fill: f32) -> (Vec<f32>, usize, usize) {
    let out_nx = (nx as f32 * factor).round() as usize;
    let out_ny = (ny as f32 * factor).round() as usize;
    let mut out = vec![fill; out_nx * out_ny];
    let inv = 1.0 / factor;
    for y in 0..out_ny {
        for x in 0..out_nx {
            let sx = x as f32 * inv;
            let sy = y as f32 * inv;
            out[y * out_nx + x] = sample_bicubic(data, nx, ny, sx, sy, fill);
        }
    }
    (out, out_nx, out_ny)
}

/// Shrink (reduce) image using Lanczos-2 interpolation. Returns new data and (out_nx, out_ny).
fn apply_shrink(data: &[f32], nx: usize, ny: usize, factor: f32, fill: f32) -> (Vec<f32>, usize, usize) {
    let out_nx = (nx as f32 / factor).round() as usize;
    let out_ny = (ny as f32 / factor).round() as usize;
    let mut out = vec![fill; out_nx * out_ny];
    let radius = (2.0 * factor).ceil() as isize; // Lanczos-2 support scaled by factor
    for y in 0..out_ny {
        for x in 0..out_nx {
            let sx = (x as f32 + 0.5) * factor - 0.5;
            let sy = (y as f32 + 0.5) * factor - 0.5;
            let ix = sx.floor() as isize;
            let iy = sy.floor() as isize;
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for jj in -radius..=radius {
                let yy = iy + jj;
                if yy < 0 || yy >= ny as isize {
                    continue;
                }
                let wy = lanczos2((sy - yy as f32) / factor);
                for ii in -radius..=radius {
                    let xx = ix + ii;
                    if xx < 0 || xx >= nx as isize {
                        continue;
                    }
                    let wx = lanczos2((sx - xx as f32) / factor);
                    let w = wx * wy;
                    sum += data[yy as usize * nx + xx as usize] * w;
                    wsum += w;
                }
            }
            out[y * out_nx + x] = if wsum > 1e-10 { sum / wsum } else { fill };
        }
    }
    (out, out_nx, out_ny)
}

/// Parse subarea string "x0,y0,width,height" into (x0, y0, w, h).
fn parse_subarea(s: &str) -> (usize, usize, usize, usize) {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        eprintln!("Error: --subarea requires x0,y0,width,height (4 comma-separated values)");
        std::process::exit(1);
    }
    let vals: Vec<usize> = parts
        .iter()
        .map(|p| p.trim().parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Error: invalid subarea value '{}'", p);
            std::process::exit(1);
        }))
        .collect();
    (vals[0], vals[1], vals[2], vals[3])
}

/// Extract a rectangular sub-region from image data.
fn extract_subarea(data: &[f32], nx: usize, _ny: usize, x0: usize, y0: usize, w: usize, h: usize, fill: f32) -> Vec<f32> {
    let mut out = vec![fill; w * h];
    for y in 0..h {
        let src_y = y0 + y;
        for x in 0..w {
            let src_x = x0 + x;
            if src_x < nx && src_y * nx + src_x < data.len() {
                out[y * w + x] = data[src_y * nx + src_x];
            }
        }
    }
    out
}

/// Build a 1D Lanczos-2 low-pass filter kernel for the given bin factor.
/// The cutoff frequency is 1/(2*bin) of Nyquist, and the kernel radius is
/// 2*bin pixels (4 lobes of Lanczos-2 scaled to the bin factor).
fn build_lanczos2_kernel(bin: usize) -> Vec<f32> {
    let radius = 2 * bin; // 4 pixels in Lanczos-2 units, scaled by bin
    let len = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for i in 0..len {
        let x = (i as f32 - radius as f32) / bin as f32;
        let w = lanczos2(x);
        kernel.push(w);
        sum += w;
    }
    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Apply separable 1D convolution (horizontal then vertical) with the given kernel.
fn apply_separable_filter(data: &[f32], nx: usize, ny: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;

    // Horizontal pass
    let mut temp = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = x as isize + ki as isize - radius as isize;
                let sx = sx.clamp(0, nx as isize - 1) as usize;
                sum += data[y * nx + sx] * kv;
            }
            temp[y * nx + x] = sum;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let mut sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = y as isize + ki as isize - radius as isize;
                let sy = sy.clamp(0, ny as isize - 1) as usize;
                sum += temp[sy * nx + x] * kv;
            }
            result[y * nx + x] = sum;
        }
    }

    result
}

/// Output range limits for a given MRC mode.
fn mode_range(mode: MrcMode) -> (f32, f32) {
    match mode {
        MrcMode::Byte => (0.0, 255.0),
        MrcMode::Short => (-32768.0, 32767.0),
        MrcMode::UShort => (0.0, 65535.0),
        _ => (f32::MIN, f32::MAX), // Float: no clamping needed
    }
}

/// Represents a section source: which file and which z-index within that file.
struct SectionSource {
    file_idx: usize,
    z: usize,
}

/// Build the list of all available sections across multiple input files.
/// Returns (sources, per-file nz, nx, ny from first file).
fn build_multi_file_section_list(
    inputs: &[String],
) -> (Vec<SectionSource>, usize, usize) {
    let mut sources = Vec::new();
    let mut first_nx = 0usize;
    let mut first_ny = 0usize;

    for (file_idx, path) in inputs.iter().enumerate() {
        let reader = MrcReader::open(path).unwrap_or_else(|e| {
            eprintln!("Error opening {}: {}", path, e);
            std::process::exit(1);
        });
        let h = reader.header();
        let nx = h.nx as usize;
        let ny = h.ny as usize;
        let nz = h.nz as usize;

        if file_idx == 0 {
            first_nx = nx;
            first_ny = ny;
        } else if nx != first_nx || ny != first_ny {
            eprintln!(
                "Warning: {} has dimensions {}x{}, expected {}x{}",
                path, nx, ny, first_nx, first_ny
            );
        }

        for z in 0..nz {
            sources.push(SectionSource { file_idx, z });
        }
    }
    (sources, first_nx, first_ny)
}

/// A piece coordinate: tile placed at (x, y) in output section z.
#[derive(Clone, Debug)]
struct PieceCoord {
    x: i32,
    y: i32,
    z: i32,
}

/// Read a piece list file. Each line: "x y z" (integers).
fn read_piece_list(path: &str) -> Vec<PieceCoord> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading piece list {}: {}", path, e);
        std::process::exit(1);
    });
    let mut pieces = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let x: i32 = parts[0].parse().unwrap_or_else(|_| {
                eprintln!("Invalid piece list line: {}", line);
                std::process::exit(1);
            });
            let y: i32 = parts[1].parse().unwrap_or_else(|_| {
                eprintln!("Invalid piece list line: {}", line);
                std::process::exit(1);
            });
            let z: i32 = parts[2].parse().unwrap_or_else(|_| {
                eprintln!("Invalid piece list line: {}", line);
                std::process::exit(1);
            });
            pieces.push(PieceCoord { x, y, z });
        }
    }
    pieces
}

/// Assemble tiles into output sections using piece coordinates with linear blending.
/// Returns a map from output z -> assembled image data.
fn assemble_pieces(
    pieces: &[PieceCoord],
    sections: &[Vec<f32>],
    tile_nx: usize,
    tile_ny: usize,
    out_nx: usize,
    out_ny: usize,
    fill: f32,
) -> std::collections::BTreeMap<i32, Vec<f32>> {
    use std::collections::BTreeMap;

    // Group pieces by output z
    let mut groups: BTreeMap<i32, Vec<(usize, &PieceCoord)>> = BTreeMap::new();
    for (i, pc) in pieces.iter().enumerate() {
        groups.entry(pc.z).or_default().push((i, pc));
    }

    let mut result = BTreeMap::new();

    for (&z, tiles) in &groups {
        let mut output = vec![fill; out_nx * out_ny];
        let mut weight_map = vec![0.0f32; out_nx * out_ny];

        for &(sec_idx, pc) in tiles {
            if sec_idx >= sections.len() {
                eprintln!("Warning: piece index {} exceeds available sections", sec_idx);
                continue;
            }
            let tile_data = &sections[sec_idx];

            for ty in 0..tile_ny {
                for tx in 0..tile_nx {
                    let ox = pc.x as isize + tx as isize;
                    let oy = pc.y as isize + ty as isize;
                    if ox < 0 || ox >= out_nx as isize || oy < 0 || oy >= out_ny as isize {
                        continue;
                    }
                    let ox = ox as usize;
                    let oy = oy as usize;

                    // Compute blend weight: ramp from 0 at tile edges to 1 at center
                    let wx = {
                        let dist_from_edge = (tx as f32).min((tile_nx - 1 - tx) as f32);
                        let blend_width = (tile_nx as f32 * 0.1).max(1.0);
                        (dist_from_edge / blend_width).min(1.0)
                    };
                    let wy = {
                        let dist_from_edge = (ty as f32).min((tile_ny - 1 - ty) as f32);
                        let blend_width = (tile_ny as f32 * 0.1).max(1.0);
                        (dist_from_edge / blend_width).min(1.0)
                    };
                    let w = wx * wy;

                    let val = tile_data[ty * tile_nx + tx];
                    let idx = oy * out_nx + ox;

                    if weight_map[idx] == 0.0 {
                        output[idx] = val * w;
                    } else {
                        output[idx] += val * w;
                    }
                    weight_map[idx] += w;
                }
            }
        }

        // Normalize by accumulated weights
        for i in 0..output.len() {
            if weight_map[i] > 0.0 {
                output[i] /= weight_map[i];
            }
        }

        result.insert(z, output);
    }

    result
}

/// A distortion field loaded from a binary file.
/// Header: grid_nx (i32), grid_ny (i32), spacing (i32).
/// Data: grid_nx * grid_ny pairs of (dx: f32, dy: f32).
struct DistortionField {
    grid_nx: usize,
    grid_ny: usize,
    spacing: f32,
    /// Interleaved (dx, dy) pairs, row-major: data[2*(gy*grid_nx+gx)] = dx, +1 = dy
    data: Vec<f32>,
}

impl DistortionField {
    fn load(path: &str) -> Self {
        let mut file = std::fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("Error opening distortion field {}: {}", path, e);
            std::process::exit(1);
        });

        let mut header_buf = [0u8; 12];
        file.read_exact(&mut header_buf).unwrap_or_else(|e| {
            eprintln!("Error reading distortion header: {}", e);
            std::process::exit(1);
        });

        let grid_nx = i32::from_le_bytes(header_buf[0..4].try_into().unwrap()) as usize;
        let grid_ny = i32::from_le_bytes(header_buf[4..8].try_into().unwrap()) as usize;
        let spacing = i32::from_le_bytes(header_buf[8..12].try_into().unwrap()) as f32;

        let n_floats = grid_nx * grid_ny * 2;
        let mut float_buf = vec![0u8; n_floats * 4];
        file.read_exact(&mut float_buf).unwrap_or_else(|e| {
            eprintln!("Error reading distortion data: {}", e);
            std::process::exit(1);
        });

        let data: Vec<f32> = float_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        eprintln!(
            "newstack: loaded distortion field {}x{} spacing={}",
            grid_nx, grid_ny, spacing
        );

        DistortionField { grid_nx, grid_ny, spacing, data }
    }

    /// Look up the distortion at pixel (px, py) using bilinear interpolation
    /// of the grid. Returns (dx, dy) correction to subtract from the pixel coords.
    fn lookup(&self, px: f32, py: f32) -> (f32, f32) {
        let gx_f = px / self.spacing;
        let gy_f = py / self.spacing;

        let gx0 = (gx_f.floor() as isize).clamp(0, self.grid_nx as isize - 2) as usize;
        let gy0 = (gy_f.floor() as isize).clamp(0, self.grid_ny as isize - 2) as usize;
        let gx1 = gx0 + 1;
        let gy1 = gy0 + 1;

        let fx = (gx_f - gx0 as f32).clamp(0.0, 1.0);
        let fy = (gy_f - gy0 as f32).clamp(0.0, 1.0);

        let idx = |gx: usize, gy: usize| -> usize { 2 * (gy * self.grid_nx + gx) };

        let dx = self.data[idx(gx0, gy0)] * (1.0 - fx) * (1.0 - fy)
            + self.data[idx(gx1, gy0)] * fx * (1.0 - fy)
            + self.data[idx(gx0, gy1)] * (1.0 - fx) * fy
            + self.data[idx(gx1, gy1)] * fx * fy;

        let dy = self.data[idx(gx0, gy0) + 1] * (1.0 - fx) * (1.0 - fy)
            + self.data[idx(gx1, gy0) + 1] * fx * (1.0 - fy)
            + self.data[idx(gx0, gy1) + 1] * (1.0 - fx) * fy
            + self.data[idx(gx1, gy1) + 1] * fx * fy;

        (dx, dy)
    }
}

/// Apply distortion correction to image data: for each output pixel, look up the
/// distortion shift, sample the source at (x - dx, y - dy) using bilinear interpolation.
fn apply_distortion(
    data: &[f32],
    nx: usize,
    ny: usize,
    distort: &DistortionField,
    fill: f32,
) -> Vec<f32> {
    let src = Slice::from_data(nx, ny, data.to_vec());
    let mut out = vec![fill; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let (dx, dy) = distort.lookup(x as f32, y as f32);
            let sx = x as f32 - dx;
            let sy = y as f32 - dy;
            out[y * nx + x] = src.interpolate_bilinear(sx, sy, fill);
        }
    }
    out
}

/// Apply a cosine taper to fill-value edges of the image.
/// Pixels that are exactly `fill` near image boundaries get a smooth ramp
/// from the fill value toward the first real image pixel.
fn apply_taper(data: &mut [f32], nx: usize, ny: usize, taper: usize, fill: f32) {
    if taper == 0 {
        return;
    }

    // Build a mask: true if the pixel is "filled" (matches fill value exactly)
    let is_fill: Vec<bool> = data.iter().map(|&v| (v - fill).abs() < 1e-10).collect();

    // For each non-fill pixel near a fill region, apply a cosine ramp
    // For each fill pixel, find the distance to the nearest non-fill pixel
    // and if within taper range, blend toward the nearest non-fill value.

    // Compute distance-to-edge for fill pixels (simple horizontal + vertical scan)
    let mut dist = vec![u32::MAX; nx * ny];

    // Horizontal passes
    for y in 0..ny {
        // Left to right
        let mut d: u32 = u32::MAX;
        for x in 0..nx {
            let idx = y * nx + x;
            if !is_fill[idx] {
                d = 0;
            } else if d < u32::MAX {
                d += 1;
            }
            dist[idx] = dist[idx].min(d);
        }
        // Right to left
        d = u32::MAX;
        for x in (0..nx).rev() {
            let idx = y * nx + x;
            if !is_fill[idx] {
                d = 0;
            } else if d < u32::MAX {
                d += 1;
            }
            dist[idx] = dist[idx].min(d);
        }
    }

    // Vertical passes
    for x in 0..nx {
        let mut d: u32 = u32::MAX;
        for y in 0..ny {
            let idx = y * nx + x;
            if !is_fill[idx] {
                d = 0;
            } else if d < u32::MAX {
                d += 1;
            }
            dist[idx] = dist[idx].min(d);
        }
        d = u32::MAX;
        for y in (0..ny).rev() {
            let idx = y * nx + x;
            if !is_fill[idx] {
                d = 0;
            } else if d < u32::MAX {
                d += 1;
            }
            dist[idx] = dist[idx].min(d);
        }
    }

    // Now for fill pixels within taper distance, find nearest non-fill neighbor value
    // and blend with cosine ramp
    let original = data.to_vec();
    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            if !is_fill[idx] || dist[idx] == 0 {
                continue;
            }
            let d = dist[idx] as usize;
            if d > taper {
                continue;
            }
            // Find the nearest non-fill pixel by scanning in cardinal directions
            let mut nearest_val = fill;
            let mut best_dist = usize::MAX;
            for &(ddx, ddy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                for step in 1..=(taper + 1) {
                    let sx = x as i32 + ddx * step as i32;
                    let sy = y as i32 + ddy * step as i32;
                    if sx < 0 || sx >= nx as i32 || sy < 0 || sy >= ny as i32 {
                        break;
                    }
                    let si = sy as usize * nx + sx as usize;
                    if !is_fill[si] {
                        if step < best_dist {
                            best_dist = step;
                            nearest_val = original[si];
                        }
                        break;
                    }
                }
            }
            if best_dist <= taper {
                // Cosine ramp: 1.0 at distance 0 from edge, 0.0 at distance taper
                let t = d as f32 / taper as f32;
                let weight = 0.5 * (1.0 + (PI * t).cos()); // 1 at t=0, 0 at t=1
                data[idx] = nearest_val * weight + fill * (1.0 - weight);
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    // Build section sources from all input files
    let (all_sources, in_nx, in_ny) = build_multi_file_section_list(&args.input);
    let total_sections = all_sources.len();

    // Open first input to get header info for output
    let first_reader = MrcReader::open(&args.input[0]).unwrap_or_else(|e| {
        eprintln!("Error opening {}: {}", args.input[0], e);
        std::process::exit(1);
    });
    let in_header = first_reader.header().clone();
    drop(first_reader);

    // Determine sections to process
    let section_indices: Vec<usize> = match &args.secs {
        Some(s) => parse_section_list(s, total_sections),
        None => (0..total_sections).collect(),
    };
    let out_nz = section_indices.len();

    // Load transforms if provided
    let transforms: Option<Vec<LinearTransform>> = args.xform.as_ref().map(|path| {
        read_xf_file(path).unwrap_or_else(|e| {
            eprintln!("Error reading transform file {}: {}", path, e);
            std::process::exit(1);
        })
    });

    // Parse subarea if provided
    let subarea: Option<(usize, usize, usize, usize)> = args.subarea.as_ref().map(|s| {
        let (x0, y0, w, h) = parse_subarea(s);
        if x0 + w > in_nx || y0 + h > in_ny {
            eprintln!(
                "Error: subarea {}+{} x {}+{} exceeds input dimensions {}x{}",
                x0, w, y0, h, in_nx, in_ny
            );
            std::process::exit(1);
        }
        (x0, y0, w, h)
    });

    // Determine working dimensions after subarea extraction
    let (work_nx, work_ny) = match subarea {
        Some((_, _, w, h)) => (w, h),
        None => (in_nx, in_ny),
    };

    // Determine output dimensions accounting for expand/shrink and binning
    let (out_nx, out_ny) = {
        let mut nx = work_nx as f32;
        let mut ny = work_ny as f32;
        if let Some(factor) = args.expand {
            nx *= factor;
            ny *= factor;
        }
        if let Some(factor) = args.shrink {
            nx /= factor;
            ny /= factor;
        }
        ((nx.round() as usize) / args.bin, (ny.round() as usize) / args.bin)
    };

    // Determine effective float mode
    // --float_densities (legacy -F) is equivalent to --float 0
    let float_mode: Option<i32> = if args.float_mode >= 0 {
        Some(args.float_mode)
    } else if args.float_densities {
        Some(0)
    } else {
        None
    };

    // Determine output mode
    let out_mode = match args.mode {
        Some(m) => MrcMode::from_i32(m).unwrap_or(MrcMode::Float),
        None => {
            if float_mode.is_some() {
                MrcMode::Byte
            } else {
                in_header.data_mode().unwrap_or(MrcMode::Float)
            }
        }
    };

    // Load piece list if provided
    let piece_list: Option<Vec<PieceCoord>> = args.piece_list.as_ref().map(|path| {
        read_piece_list(path)
    });

    // Load distortion field if provided
    let distortion: Option<DistortionField> = args.distort.as_ref().map(|path| {
        DistortionField::load(path)
    });

    // Determine antialias setting
    let use_antialias = args.antialias.unwrap_or(args.bin > 1);

    // Build antialias kernel if needed
    let aa_kernel = if use_antialias && args.bin > 1 {
        Some(build_lanczos2_kernel(args.bin))
    } else {
        None
    };

    // Parse scaling
    let scale_range: Option<(f32, f32)> = args.scale.as_ref().and_then(|s| {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 2 {
            Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
        } else {
            None
        }
    });

    // =========================================================================
    // For float modes 1 and 2, we need a first pass to compute global mean/SD
    // =========================================================================
    let (global_target_mean, global_target_sd) = if matches!(float_mode, Some(1) | Some(2)) {
        eprintln!("newstack: first pass - computing global statistics...");
        let mut all_sum = 0.0_f64;
        let mut all_sum_sq = 0.0_f64;
        let mut all_count = 0u64;

        // Open readers for each file as needed
        let mut open_readers: Vec<Option<MrcReader>> =
            (0..args.input.len()).map(|_| None).collect();

        for &sec_idx in &section_indices {
            let src = &all_sources[sec_idx];

            // Open reader for this file if not already open
            if open_readers[src.file_idx].is_none() {
                open_readers[src.file_idx] =
                    Some(MrcReader::open(&args.input[src.file_idx]).unwrap());
            }
            let reader = open_readers[src.file_idx].as_mut().unwrap();
            let data = reader.read_slice_f32(src.z).unwrap();

            for &v in &data {
                let vd = v as f64;
                all_sum += vd;
                all_sum_sq += vd * vd;
            }
            all_count += data.len() as u64;
        }

        let mean = (all_sum / all_count as f64) as f32;
        let variance = (all_sum_sq / all_count as f64) - (mean as f64 * mean as f64);
        let sd = (variance.max(0.0) as f32).sqrt();
        eprintln!("newstack: global mean={:.2}, SD={:.2}", mean, sd);
        (mean, sd)
    } else {
        (0.0f32, 1.0f32)
    };

    // =========================================================================
    // Piece list assembly mode
    // =========================================================================
    if let Some(ref pieces) = piece_list {
        eprintln!(
            "newstack: assembling {} tiles using piece list ({} entries)",
            section_indices.len(),
            pieces.len()
        );

        // Compute output dimensions from piece coordinates + tile size
        let mut max_x: i32 = 0;
        let mut max_y: i32 = 0;
        let mut max_z: i32 = 0;
        for pc in pieces.iter() {
            let right = pc.x + in_nx as i32;
            let bottom = pc.y + in_ny as i32;
            if right > max_x { max_x = right; }
            if bottom > max_y { max_y = bottom; }
            if pc.z + 1 > max_z { max_z = pc.z + 1; }
        }

        let asm_nx = (max_x as usize) / args.bin;
        let asm_ny = (max_y as usize) / args.bin;
        let asm_nz = max_z as usize;

        // Read and pre-process all tile sections
        let mut processed_sections: Vec<Vec<f32>> = Vec::new();
        let xcen = in_nx as f32 / 2.0;
        let ycen = in_ny as f32 / 2.0;

        let mut open_readers: Vec<Option<MrcReader>> =
            (0..args.input.len()).map(|_| None).collect();

        for (out_z, &sec_idx) in section_indices.iter().enumerate() {
            let src = &all_sources[sec_idx];
            if open_readers[src.file_idx].is_none() {
                open_readers[src.file_idx] =
                    Some(MrcReader::open(&args.input[src.file_idx]).unwrap_or_else(|e| {
                        eprintln!("Error opening {}: {}", args.input[src.file_idx], e);
                        std::process::exit(1);
                    }));
            }
            let reader = open_readers[src.file_idx].as_mut().unwrap();
            let mut data = reader.read_slice_f32(src.z).unwrap();

            // Apply distortion correction
            if let Some(ref dist_field) = distortion {
                data = apply_distortion(&data, in_nx, in_ny, dist_field, args.fill);
            }

            // Apply transform
            if let Some(ref xforms) = transforms {
                let xf_idx = out_z.min(xforms.len() - 1);
                let xf = &xforms[xf_idx];
                let inv = xf.inverse();
                let src_slice = Slice::from_data(in_nx, in_ny, data);
                let mut dst = Slice::new(in_nx, in_ny, args.fill);
                for y in 0..in_ny {
                    for x in 0..in_nx {
                        let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
                        dst.set(x, y, src_slice.interpolate_bilinear(sx, sy, args.fill));
                    }
                }
                data = dst.data;
            }

            // Taper fill edges
            if args.taper > 0 {
                apply_taper(&mut data, in_nx, in_ny, args.taper, args.fill);
            }

            // Antialias + bin
            if let Some(ref kernel) = aa_kernel {
                data = apply_separable_filter(&data, in_nx, in_ny, kernel);
            }
            if args.bin > 1 {
                let src_slice = Slice::from_data(in_nx, in_ny, data);
                let binned = imod_slice::bin(&src_slice, args.bin);
                data = binned.data;
            }

            processed_sections.push(data);
        }

        // Adjust piece coordinates for binning
        let bin = args.bin as i32;
        let binned_pieces: Vec<PieceCoord> = pieces.iter().map(|pc| PieceCoord {
            x: pc.x / bin,
            y: pc.y / bin,
            z: pc.z,
        }).collect();

        let tile_bnx = in_nx / args.bin;
        let tile_bny = in_ny / args.bin;

        // Assemble
        let assembled = assemble_pieces(
            &binned_pieces,
            &processed_sections,
            tile_bnx,
            tile_bny,
            asm_nx,
            asm_ny,
            args.fill,
        );

        // Create output
        let mut out_header = MrcHeader::new(asm_nx as i32, asm_ny as i32, asm_nz as i32, out_mode);
        out_header.add_label(&format!(
            "newstack: assembled {} tiles into {}x{}x{}",
            pieces.len(), asm_nx, asm_ny, asm_nz
        ));

        let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
            eprintln!("Error creating {}: {}", args.output, e);
            std::process::exit(1);
        });

        let mut global_min = f32::MAX;
        let mut global_max = f32::MIN;
        let mut global_sum = 0.0_f64;
        let total_pix = (asm_nx * asm_ny * asm_nz) as f64;

        for z in 0..asm_nz {
            let data = assembled.get(&(z as i32)).cloned().unwrap_or_else(|| {
                vec![args.fill; asm_nx * asm_ny]
            });

            let (smin, smax, smean) = min_max_mean(&data);
            if smin < global_min { global_min = smin; }
            if smax > global_max { global_max = smax; }
            global_sum += smean as f64 * (asm_nx * asm_ny) as f64;

            writer.write_slice_f32(&data).unwrap();
        }

        let global_mean = (global_sum / total_pix) as f32;
        writer.finish(global_min, global_max, global_mean).unwrap();

        eprintln!(
            "newstack: piece assembly complete -> {}x{}x{} (mode {:?})",
            asm_nx, asm_ny, asm_nz, out_mode
        );
        return;
    }

    // =========================================================================
    // Standard (non-piece-list) processing path
    // =========================================================================

    // Create output header
    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, out_mode);
    let bin_f = args.bin as f32;
    out_header.xlen = in_header.xlen / bin_f * (out_nx as f32 / (in_nx as f32 / bin_f));
    out_header.ylen = in_header.ylen / bin_f * (out_ny as f32 / (in_ny as f32 / bin_f));
    out_header.zlen = in_header.zlen * out_nz as f32 / in_header.nz as f32;
    out_header.mx = out_nx as i32;
    out_header.my = out_ny as i32;
    out_header.mz = out_nz as i32;

    let label = if args.input.len() > 1 {
        format!(
            "newstack: {} sections from {} files",
            out_nz,
            args.input.len()
        )
    } else {
        format!("newstack: {} sections from {}", out_nz, args.input[0])
    };
    out_header.add_label(&label);

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("Error creating {}: {}", args.output, e);
        std::process::exit(1);
    });

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum = 0.0_f64;
    let total_pix = (out_nx * out_ny * out_nz) as f64;

    let xcen = in_nx as f32 / 2.0;
    let ycen = in_ny as f32 / 2.0;

    // Open readers for second pass
    let mut open_readers: Vec<Option<MrcReader>> =
        (0..args.input.len()).map(|_| None).collect();

    for (out_z, &sec_idx) in section_indices.iter().enumerate() {
        let src = &all_sources[sec_idx];

        // Open reader for this file if not already open
        if open_readers[src.file_idx].is_none() {
            open_readers[src.file_idx] =
                Some(MrcReader::open(&args.input[src.file_idx]).unwrap_or_else(|e| {
                    eprintln!("Error opening {}: {}", args.input[src.file_idx], e);
                    std::process::exit(1);
                }));
        }
        let reader = open_readers[src.file_idx].as_mut().unwrap();
        let mut data = reader.read_slice_f32(src.z).unwrap();

        // Apply distortion correction if provided
        if let Some(ref dist_field) = distortion {
            data = apply_distortion(&data, in_nx, in_ny, dist_field, args.fill);
        }

        // Apply transform if provided
        if let Some(ref xforms) = transforms {
            let xf_idx = out_z.min(xforms.len() - 1);
            let xf = &xforms[xf_idx];
            let inv = xf.inverse();
            let src_slice = Slice::from_data(in_nx, in_ny, data);
            let mut dst = Slice::new(in_nx, in_ny, args.fill);
            for y in 0..in_ny {
                for x in 0..in_nx {
                    let (sx, sy) = inv.apply(xcen, ycen, x as f32, y as f32);
                    dst.set(x, y, src_slice.interpolate_bilinear(sx, sy, args.fill));
                }
            }
            data = dst.data;
        }

        // Apply taper to fill edges
        if args.taper > 0 {
            apply_taper(&mut data, in_nx, in_ny, args.taper, args.fill);
        }

        // Apply rotation (after transform, before subarea/scaling/binning)
        if let Some(angle) = args.rotate {
            data = apply_rotation(&data, in_nx, in_ny, angle, args.fill);
        }

        // Extract subarea
        let mut cur_nx = in_nx;
        let mut cur_ny = in_ny;
        if let Some((x0, y0, w, h)) = subarea {
            data = extract_subarea(&data, cur_nx, cur_ny, x0, y0, w, h, args.fill);
            cur_nx = w;
            cur_ny = h;
        }

        // Apply expand (bicubic)
        if let Some(factor) = args.expand {
            let (expanded, enx, eny) = apply_expand(&data, cur_nx, cur_ny, factor, args.fill);
            data = expanded;
            cur_nx = enx;
            cur_ny = eny;
        }

        // Apply shrink (Lanczos-2)
        if let Some(factor) = args.shrink {
            let (shrunk, snx, sny) = apply_shrink(&data, cur_nx, cur_ny, factor, args.fill);
            data = shrunk;
            cur_nx = snx;
            cur_ny = sny;
        }

        // Antialias filtering before binning
        if let Some(ref kernel) = aa_kernel {
            data = apply_separable_filter(&data, cur_nx, cur_ny, kernel);
        }

        // Bin
        if args.bin > 1 {
            let src_slice = Slice::from_data(cur_nx, cur_ny, data);
            let binned = imod_slice::bin(&src_slice, args.bin);
            data = binned.data;
        }

        // Apply scaling / float modes
        if let Some((smin, smax)) = scale_range {
            let (dmin, dmax, _) = min_max_mean(&data);
            let range = dmax - dmin;
            if range > 1e-10 {
                let scale = (smax - smin) / range;
                for v in &mut data {
                    *v = (*v - dmin) * scale + smin;
                }
            }
        } else if let Some(fm) = float_mode {
            match fm {
                0 => {
                    // Scale each section to fill output range
                    let (dmin, dmax, _) = min_max_mean(&data);
                    let range = dmax - dmin;
                    let (out_lo, out_hi) = mode_range(out_mode);
                    if range > 1e-10 && out_lo.is_finite() && out_hi.is_finite() {
                        let scale = (out_hi - out_lo) / range;
                        for v in &mut data {
                            *v = (*v - dmin) * scale + out_lo;
                        }
                    } else if range > 1e-10 {
                        // Float output: just shift min to 0
                        for v in &mut data {
                            *v -= dmin;
                        }
                    }
                }
                1 => {
                    // Shift each section to the common mean
                    let (_, _, sec_mean) = min_max_mean(&data);
                    let shift = global_target_mean - sec_mean;
                    for v in &mut data {
                        *v += shift;
                    }
                }
                2 => {
                    // Scale each section to common mean and SD
                    let (_, _, sec_mean, sec_sd) = min_max_mean_sd(&data);
                    if sec_sd > 1e-10 {
                        let scale = global_target_sd / sec_sd;
                        for v in &mut data {
                            *v = (*v - sec_mean) * scale + global_target_mean;
                        }
                    } else {
                        // Zero variance section: just shift to mean
                        for v in &mut data {
                            *v = global_target_mean;
                        }
                    }
                }
                _ => {
                    eprintln!("Warning: unknown float mode {}, ignoring", fm);
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&data);
        if smin < global_min {
            global_min = smin;
        }
        if smax > global_max {
            global_max = smax;
        }
        global_sum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&data).unwrap();
    }

    let global_mean = (global_sum / total_pix) as f32;
    writer.finish(global_min, global_max, global_mean).unwrap();

    eprintln!(
        "newstack: {} x {} x {} -> {} x {} x {} (mode {:?}){}",
        in_nx,
        in_ny,
        total_sections,
        out_nx,
        out_ny,
        out_nz,
        out_mode,
        if aa_kernel.is_some() {
            " [antialias]"
        } else {
            ""
        }
    );
    if let Some(fm) = float_mode {
        eprintln!("newstack: float mode {}", fm);
    }
}
