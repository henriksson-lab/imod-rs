use clap::Parser;
use imod_core::MrcMode;
use imod_fft::cross_correlate_2d;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::io::{self, BufRead};
use std::path::Path;

/// Blend overlapping montage tiles into a single image.
///
/// Reads a montage stack where each section is a tile with known piece
/// coordinates, and blends overlapping regions using linear weighting.
/// Optionally refines tile positions via cross-correlation of overlap edges.
#[derive(Parser)]
#[command(name = "blendmont", about = "Blend montage tiles into a single image")]
struct Args {
    /// Input montage stack (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Output blended image (MRC)
    #[arg(short = 'o', long)]
    output: String,

    /// Piece list file (format: x y z per line for each section)
    #[arg(short = 'p', long)]
    pieces: String,

    /// Width of blending edge (pixels)
    #[arg(short = 'w', long, default_value_t = 50)]
    blend_width: usize,

    /// Enable edge correlation refinement of piece coordinates
    #[arg(long, default_value_t = true)]
    edge_shift: bool,

    /// Override the overlap width detected from piece coordinates (pixels)
    #[arg(long)]
    overlap_width: Option<usize>,
}

/// An edge between two overlapping tiles, with the measured shift.
struct EdgePair {
    tile_a: usize,
    tile_b: usize,
    direction: EdgeDirection,
    shift_x: f32,
    shift_y: f32,
}

#[derive(Clone, Copy)]
enum EdgeDirection {
    Horizontal, // tile_b is to the right of tile_a
    Vertical,   // tile_b is below tile_a
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let tile_nx = h.nx as usize;
    let tile_ny = h.ny as usize;
    let n_tiles = h.nz as usize;

    // Read piece coordinates
    let mut pieces = read_piece_list(&args.pieces).unwrap_or_else(|e| {
        eprintln!("Error reading piece list: {}", e);
        std::process::exit(1);
    });

    if pieces.len() != n_tiles {
        eprintln!(
            "Warning: {} tiles in stack but {} piece coordinates",
            n_tiles,
            pieces.len()
        );
    }

    // Read all tile data upfront (needed for edge correlation)
    let mut tile_data: Vec<Vec<f32>> = Vec::with_capacity(n_tiles);
    for z in 0..n_tiles {
        tile_data.push(reader.read_slice_f32(z).unwrap());
    }

    // --- Edge correlation refinement ---
    if args.edge_shift {
        let edge_pairs = find_and_correlate_edges(
            &pieces,
            &tile_data,
            tile_nx,
            tile_ny,
            args.overlap_width,
        );

        eprintln!("blendmont: found {} edge pairs for correlation", edge_pairs.len());
        for ep in &edge_pairs {
            let dir_str = match ep.direction {
                EdgeDirection::Horizontal => "H",
                EdgeDirection::Vertical => "V",
            };
            eprintln!(
                "  edge {}-{} ({}): shift dx={:.2}, dy={:.2}",
                ep.tile_a, ep.tile_b, dir_str, ep.shift_x, ep.shift_y
            );
        }

        // Apply shifts: adjust piece coordinates using a simple averaging scheme.
        // For each tile, average the suggested corrections from all its edges.
        if !edge_pairs.is_empty() {
            let mut corrections: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); pieces.len()];

            for ep in &edge_pairs {
                // The shift tells us how much tile_b should move relative to tile_a.
                // We split the correction: move tile_b by +shift/2, tile_a by -shift/2.
                let half_dx = ep.shift_x as f64 / 2.0;
                let half_dy = ep.shift_y as f64 / 2.0;

                corrections[ep.tile_a].0 -= half_dx;
                corrections[ep.tile_a].1 -= half_dy;
                corrections[ep.tile_a].2 += 1;

                corrections[ep.tile_b].0 += half_dx;
                corrections[ep.tile_b].1 += half_dy;
                corrections[ep.tile_b].2 += 1;
            }

            for (ti, (cdx, cdy, count)) in corrections.iter().enumerate() {
                if *count > 0 {
                    let adj_x = (*cdx / *count as f64).round() as i32;
                    let adj_y = (*cdy / *count as f64).round() as i32;
                    pieces[ti].0 += adj_x;
                    pieces[ti].1 += adj_y;
                    if adj_x != 0 || adj_y != 0 {
                        eprintln!(
                            "  tile {} adjusted by ({}, {})",
                            ti, adj_x, adj_y
                        );
                    }
                }
            }
        }
    }

    // Determine output dimensions
    let mut max_x = 0i32;
    let mut max_y = 0i32;
    for &(px, py, _pz) in &pieces {
        if px + tile_nx as i32 > max_x { max_x = px + tile_nx as i32; }
        if py + tile_ny as i32 > max_y { max_y = py + tile_ny as i32; }
    }

    let out_nx = max_x as usize;
    let out_ny = max_y as usize;

    eprintln!(
        "blendmont: {} tiles of {}x{} -> {}x{} output, blend={}px",
        n_tiles, tile_nx, tile_ny, out_nx, out_ny, args.blend_width
    );

    // Group tiles by Z (output section)
    let mut max_z = 0i32;
    for &(_, _, pz) in &pieces {
        if pz > max_z { max_z = pz; }
    }
    let out_nz = (max_z + 1) as usize;

    let mut out_header = MrcHeader::new(out_nx as i32, out_ny as i32, out_nz as i32, MrcMode::Float);
    out_header.add_label(&format!("blendmont: {} tiles blended", n_tiles));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for oz in 0..out_nz {
        let mut output = vec![0.0f32; out_nx * out_ny];
        let mut weights = vec![0.0f32; out_nx * out_ny];

        // Find tiles belonging to this Z
        for (ti, &(px, py, pz)) in pieces.iter().enumerate() {
            if pz as usize != oz || ti >= n_tiles {
                continue;
            }

            let data = &tile_data[ti];

            for ty in 0..tile_ny {
                for tx in 0..tile_nx {
                    let ox = px as usize + tx;
                    let oy = py as usize + ty;
                    if ox >= out_nx || oy >= out_ny {
                        continue;
                    }

                    // Edge weight: ramp from 0 to 1 over blend_width at edges
                    let bw = args.blend_width as f32;
                    let wx = (tx as f32 / bw).min(1.0).min(((tile_nx - 1 - tx) as f32) / bw).min(1.0);
                    let wy = (ty as f32 / bw).min(1.0).min(((tile_ny - 1 - ty) as f32) / bw).min(1.0);
                    let w = wx * wy;

                    let idx = oy * out_nx + ox;
                    output[idx] += data[ty * tile_nx + tx] * w;
                    weights[idx] += w;
                }
            }
        }

        // Normalize
        for i in 0..out_nx * out_ny {
            if weights[i] > 0.0 {
                output[i] /= weights[i];
            }
        }

        let (smin, smax, smean) = min_max_mean(&output);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (out_nx * out_ny) as f64;

        writer.write_slice_f32(&output).unwrap();
    }

    let gmean = (gsum / (out_nx * out_ny * out_nz) as f64) as f32;
    writer.finish(gmin, gmax, gmean).unwrap();
    eprintln!("blendmont: done -> {}", args.output);
}

/// Find overlapping tile pairs and compute cross-correlation shifts for each edge.
fn find_and_correlate_edges(
    pieces: &[(i32, i32, i32)],
    tile_data: &[Vec<f32>],
    tile_nx: usize,
    tile_ny: usize,
    override_overlap: Option<usize>,
) -> Vec<EdgePair> {
    let mut edges = Vec::new();

    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            let (ax, ay, az) = pieces[i];
            let (bx, by, bz) = pieces[j];

            // Only pair tiles on the same Z
            if az != bz {
                continue;
            }

            // Check horizontal overlap (b is to the right of a)
            if ay == by {
                let overlap_x = (ax + tile_nx as i32) - bx;
                if overlap_x > 0 && overlap_x < tile_nx as i32 {
                    let ov = override_overlap.unwrap_or(overlap_x as usize);
                    let ov = ov.min(overlap_x as usize).min(tile_nx);
                    if ov >= 4 {
                        let (dx, dy) = correlate_overlap_horizontal(
                            &tile_data[i], &tile_data[j],
                            tile_nx, tile_ny, ov,
                        );
                        edges.push(EdgePair {
                            tile_a: i,
                            tile_b: j,
                            direction: EdgeDirection::Horizontal,
                            shift_x: dx,
                            shift_y: dy,
                        });
                    }
                }
                // Check the reverse direction (a is to the right of b)
                let overlap_x_rev = (bx + tile_nx as i32) - ax;
                if overlap_x_rev > 0 && overlap_x_rev < tile_nx as i32 && overlap_x <= 0 {
                    let ov = override_overlap.unwrap_or(overlap_x_rev as usize);
                    let ov = ov.min(overlap_x_rev as usize).min(tile_nx);
                    if ov >= 4 {
                        let (dx, dy) = correlate_overlap_horizontal(
                            &tile_data[j], &tile_data[i],
                            tile_nx, tile_ny, ov,
                        );
                        edges.push(EdgePair {
                            tile_a: j,
                            tile_b: i,
                            direction: EdgeDirection::Horizontal,
                            shift_x: -dx,
                            shift_y: -dy,
                        });
                    }
                }
            }

            // Check vertical overlap (b is below a)
            if ax == bx {
                let overlap_y = (ay + tile_ny as i32) - by;
                if overlap_y > 0 && overlap_y < tile_ny as i32 {
                    let ov = override_overlap.unwrap_or(overlap_y as usize);
                    let ov = ov.min(overlap_y as usize).min(tile_ny);
                    if ov >= 4 {
                        let (dx, dy) = correlate_overlap_vertical(
                            &tile_data[i], &tile_data[j],
                            tile_nx, tile_ny, ov,
                        );
                        edges.push(EdgePair {
                            tile_a: i,
                            tile_b: j,
                            direction: EdgeDirection::Vertical,
                            shift_x: dx,
                            shift_y: dy,
                        });
                    }
                }
                // Check reverse
                let overlap_y_rev = (by + tile_ny as i32) - ay;
                if overlap_y_rev > 0 && overlap_y_rev < tile_ny as i32 && overlap_y <= 0 {
                    let ov = override_overlap.unwrap_or(overlap_y_rev as usize);
                    let ov = ov.min(overlap_y_rev as usize).min(tile_ny);
                    if ov >= 4 {
                        let (dx, dy) = correlate_overlap_vertical(
                            &tile_data[j], &tile_data[i],
                            tile_nx, tile_ny, ov,
                        );
                        edges.push(EdgePair {
                            tile_a: j,
                            tile_b: i,
                            direction: EdgeDirection::Vertical,
                            shift_x: -dx,
                            shift_y: -dy,
                        });
                    }
                }
            }
        }
    }

    edges
}

/// Cross-correlate the horizontal overlap region between tile_a (right edge)
/// and tile_b (left edge). Returns the (dx, dy) refinement shift.
fn correlate_overlap_horizontal(
    tile_a: &[f32],
    tile_b: &[f32],
    tile_nx: usize,
    tile_ny: usize,
    overlap: usize,
) -> (f32, f32) {
    // Extract the right strip of tile_a and left strip of tile_b
    let strip_w = overlap;
    let strip_h = tile_ny;

    let mut strip_a = vec![0.0f32; strip_w * strip_h];
    let mut strip_b = vec![0.0f32; strip_w * strip_h];

    for y in 0..strip_h {
        for x in 0..strip_w {
            strip_a[y * strip_w + x] = tile_a[y * tile_nx + (tile_nx - overlap + x)];
            strip_b[y * strip_w + x] = tile_b[y * tile_nx + x];
        }
    }

    correlate_strips(&strip_a, &strip_b, strip_w, strip_h)
}

/// Cross-correlate the vertical overlap region between tile_a (bottom edge)
/// and tile_b (top edge). Returns the (dx, dy) refinement shift.
fn correlate_overlap_vertical(
    tile_a: &[f32],
    tile_b: &[f32],
    tile_nx: usize,
    tile_ny: usize,
    overlap: usize,
) -> (f32, f32) {
    let strip_w = tile_nx;
    let strip_h = overlap;

    let mut strip_a = vec![0.0f32; strip_w * strip_h];
    let mut strip_b = vec![0.0f32; strip_w * strip_h];

    for y in 0..strip_h {
        for x in 0..strip_w {
            strip_a[y * strip_w + x] = tile_a[(tile_ny - overlap + y) * tile_nx + x];
            strip_b[y * strip_w + x] = tile_b[y * tile_nx + x];
        }
    }

    correlate_strips(&strip_a, &strip_b, strip_w, strip_h)
}

/// Correlate two strips and return the peak shift (dx, dy).
fn correlate_strips(strip_a: &[f32], strip_b: &[f32], w: usize, h: usize) -> (f32, f32) {
    // Pad to next power-of-2 for FFT
    let fft_w = next_power_of_2(w);
    let fft_h = next_power_of_2(h);

    let padded_a = pad_strip(strip_a, w, h, fft_w, fft_h);
    let padded_b = pad_strip(strip_b, w, h, fft_w, fft_h);

    let cc = cross_correlate_2d(&padded_a, &padded_b, fft_w, fft_h);

    // Find peak
    let mut max_val = f32::NEG_INFINITY;
    let mut mx = 0usize;
    let mut my = 0usize;
    for y in 0..fft_h {
        for x in 0..fft_w {
            let v = cc[y * fft_w + x];
            if v > max_val {
                max_val = v;
                mx = x;
                my = y;
            }
        }
    }

    // Convert to signed shift (handle wrap-around)
    let dx = if mx > fft_w / 2 { mx as f32 - fft_w as f32 } else { mx as f32 };
    let dy = if my > fft_h / 2 { my as f32 - fft_h as f32 } else { my as f32 };

    (dx, dy)
}

fn pad_strip(data: &[f32], w: usize, h: usize, fw: usize, fh: usize) -> Vec<f32> {
    let sum: f64 = data.iter().map(|&v| v as f64).sum();
    let mean = (sum / data.len() as f64) as f32;
    let mut padded = vec![mean; fw * fh];
    let ox = (fw - w) / 2;
    let oy = (fh - h) / 2;
    for y in 0..h {
        for x in 0..w {
            padded[(y + oy) * fw + (x + ox)] = data[y * w + x];
        }
    }
    padded
}

fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    // Minimum FFT size of 8 to avoid degenerate cases
    p.max(8)
}

fn read_piece_list(path: &str) -> io::Result<Vec<(i32, i32, i32)>> {
    let file = std::fs::File::open(Path::new(path))?;
    let reader = io::BufReader::new(file);
    let mut pieces = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let vals: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 {
            pieces.push((vals[0], vals[1], vals[2]));
        }
    }

    Ok(pieces)
}
