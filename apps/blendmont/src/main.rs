use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use std::io::{self, BufRead};
use std::path::Path;

/// Blend overlapping montage tiles into a single image.
///
/// Reads a montage stack where each section is a tile with known piece
/// coordinates, and blends overlapping regions using linear weighting.
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
    let pieces = read_piece_list(&args.pieces).unwrap_or_else(|e| {
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

            let tile_data = reader.read_slice_f32(ti).unwrap();

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
                    output[idx] += tile_data[ty * tile_nx + tx] * w;
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
