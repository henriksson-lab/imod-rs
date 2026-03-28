//! edpiecepoint - Edit piece coordinates or create a new piece list.
//!
//! Reads or creates a list of piece coordinates (X, Y, Z triplets),
//! allows remapping Z values, adjusting overlaps for binning, and
//! adding offsets to all coordinates.
//!
//! Translated from IMOD's edpiecepoint.f90

use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::process;

#[derive(Parser)]
#[command(
    name = "edpiecepoint",
    about = "Edit piece coordinates or create a new piece list"
)]
struct Args {
    /// Input piece list file
    #[arg(short = 'i', long)]
    input: Option<String>,

    /// Output piece list file
    #[arg(short = 'o', long)]
    output: String,

    /// Number of sections to create piece list for (instead of reading input)
    #[arg(long)]
    create: Option<i32>,

    /// Number of montage pieces in X and Y
    #[arg(long, num_args = 2, value_names = ["NX", "NY"])]
    pieces: Option<Vec<i32>>,

    /// Spacing between pieces in X and Y (negative for inverse order)
    #[arg(long, num_args = 2, value_names = ["DX", "DY"])]
    spacing: Option<Vec<i32>>,

    /// Arrange pieces in columns instead of rows
    #[arg(long, default_value_t = false)]
    columns: bool,

    /// Stack pieces: >0 forward, <0 reverse Z ordering within position
    #[arg(long, default_value_t = 0)]
    stacks: i32,

    /// Divide coordinates by binning factor
    #[arg(long)]
    divide: Option<i32>,

    /// New overlap in X and Y
    #[arg(long, num_args = 2, value_names = ["OX", "OY"])]
    overlap: Option<Vec<i32>>,

    /// Image size in X and Y (required for overlap/divide)
    #[arg(long, num_args = 2, value_names = ["NX", "NY"])]
    size: Option<Vec<i32>>,

    /// Binned image size in X and Y (for divide mode)
    #[arg(long, num_args = 2, value_names = ["NX", "NY"])]
    binned: Option<Vec<i32>>,

    /// New Z value list (comma-separated; use -1 to negate all)
    #[arg(long)]
    new_z: Option<String>,

    /// Amounts to add to all X, Y, Z coordinates
    #[arg(long, num_args = 3, value_names = ["DX", "DY", "DZ"], default_values_t = [0, 0, 0])]
    add: Vec<i32>,
}

fn read_piece_list(path: &str) -> Vec<(i32, i32, i32)> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: edpiecepoint - opening {}: {}", path, e);
        process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut pieces = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 3 {
            pieces.push((vals[0], vals[1], vals[2]));
        }
    }
    pieces
}

fn check_list(coords: &[i32], n: usize, frame_size: i32) -> (i32, i32, i32) {
    // Returns (min_coord, num_pieces, overlap)
    if n == 0 {
        return (0, 0, 0);
    }
    let mut sorted: Vec<i32> = coords.to_vec();
    sorted.sort();
    sorted.dedup();
    let min_coord = sorted[0];
    let num_pieces = sorted.len() as i32;
    if num_pieces <= 1 {
        return (min_coord, num_pieces, 0);
    }
    let spacing = sorted[1] - sorted[0];
    if spacing <= 0 {
        return (min_coord, -1, 0);
    }
    let overlap = frame_size - spacing;
    (min_coord, num_pieces, overlap)
}

fn main() {
    let args = Args::parse();

    let mut pieces: Vec<(i32, i32, i32)>;

    if let Some(nz) = args.create {
        // Create new piece list
        let (nx_frame, ny_frame) = args
            .pieces
            .as_ref()
            .map_or((1, 1), |v| (v[0], v[1]));
        let (nx_delta, ny_delta) = args
            .spacing
            .as_ref()
            .map_or((0, 0), |v| (v[0], v[1]));
        let in_columns = args.columns;
        let in_stacks = args.stacks;

        let (n_inner, n_outer) = if in_columns {
            (ny_frame, nx_frame)
        } else {
            (nx_frame, ny_frame)
        };

        let ix_base = if nx_delta < 0 {
            -(nx_frame - 1) * nx_delta
        } else {
            0
        };
        let iy_base = if ny_delta < 0 {
            -(ny_frame - 1) * ny_delta
        } else {
            0
        };

        let (nz_outer, iz_inner_start, iz_inner_end, iz_inner_inc) = if in_stacks > 0 {
            (1, 1i32, nz, 1i32)
        } else if in_stacks < 0 {
            (1, nz, 1i32, -1i32) // actually iterates nz down to 1
        } else {
            (nz, 1, 1, 1)
        };

        pieces = Vec::new();
        for iz_outer in 1..=nz_outer {
            for iouter in 1..=n_outer {
                for inner in 1..=n_inner {
                    let mut iz_inner = iz_inner_start;
                    loop {
                        if iz_inner_inc > 0 && iz_inner > iz_inner_end {
                            break;
                        }
                        if iz_inner_inc < 0 && iz_inner < iz_inner_end {
                            break;
                        }
                        let iz = iz_outer * iz_inner;
                        let (ix, iy) = if in_columns {
                            (iouter, inner)
                        } else {
                            (inner, iouter)
                        };
                        pieces.push((
                            ix_base + (ix - 1) * nx_delta,
                            iy_base + (iy - 1) * ny_delta,
                            iz - 1,
                        ));
                        iz_inner += iz_inner_inc;
                    }
                }
            }
        }
    } else if let Some(ref input_path) = args.input {
        pieces = read_piece_list(input_path);
    } else {
        eprintln!("ERROR: edpiecepoint - No input file or --create specified");
        process::exit(1);
    }

    // Handle overlap/divide adjustments
    if let Some(ref size) = args.size {
        let nx = size[0];
        let ny = size[1];
        let ix_coords: Vec<i32> = pieces.iter().map(|p| p.0).collect();
        let iy_coords: Vec<i32> = pieces.iter().map(|p| p.1).collect();
        let (min_x, nx_pieces, nx_overlap) = check_list(&ix_coords, pieces.len(), nx);
        let (min_y, ny_pieces, ny_overlap) = check_list(&iy_coords, pieces.len(), ny);

        if nx_pieces < 1 || ny_pieces < 1 {
            eprintln!("ERROR: edpiecepoint - Piece coordinates are not regularly spaced");
            process::exit(1);
        }

        let (new_nx, new_ny, new_x_overlap, new_y_overlap, ibinning) =
            if let Some(bin) = args.divide {
                if bin < 2 {
                    eprintln!("ERROR: edpiecepoint - Binning factor must be 2 or higher");
                    process::exit(1);
                }
                let (bnx, bny) = args
                    .binned
                    .as_ref()
                    .map_or((nx / bin, ny / bin), |v| (v[0], v[1]));

                let mut new_xo = nx_overlap / bin;
                let ipc = ((nx_pieces - 1) * (nx - nx_overlap)) / bin;
                if ((nx_pieces - 1) * (bnx - new_xo) - ipc).abs()
                    > ((nx_pieces - 1) * (bnx - new_xo - 1) - ipc).abs()
                {
                    new_xo += 1;
                }
                let mut new_yo = ny_overlap / bin;
                let ipc = ((ny_pieces - 1) * (ny - ny_overlap)) / bin;
                if ((ny_pieces - 1) * (bny - new_yo) - ipc).abs()
                    > ((ny_pieces - 1) * (bny - new_yo - 1) - ipc).abs()
                {
                    new_yo += 1;
                }
                (bnx, bny, new_xo, new_yo, bin)
            } else if let Some(ref ov) = args.overlap {
                (nx, ny, ov[0], ov[1], 1)
            } else {
                (nx, ny, nx_overlap, ny_overlap, 1)
            };

        let new_min_x = min_x / ibinning;
        let new_min_y = min_y / ibinning;
        for p in pieces.iter_mut() {
            let ipc_x = (p.0 - min_x) / (nx - nx_overlap);
            p.0 = new_min_x + ipc_x * (new_nx - new_x_overlap);
            let ipc_y = (p.1 - min_y) / (ny - ny_overlap);
            p.1 = new_min_y + ipc_y * (new_ny - new_y_overlap);
        }
    }

    // Get sorted unique Z values
    let mut z_list: Vec<i32> = pieces.iter().map(|p| p.2).collect();
    z_list.sort();
    z_list.dedup();

    println!("The current list of Z values in input list is:");
    let z_strs: Vec<String> = z_list.iter().map(|z| z.to_string()).collect();
    println!("{}", z_strs.join(" "));

    // Remap Z values
    let mut new_z_list = z_list.clone();
    if let Some(ref z_str) = args.new_z {
        let parsed: Vec<i32> = z_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if parsed.len() == 1 && parsed[0] == -1 {
            new_z_list = z_list.iter().map(|z| -z).collect();
        } else if parsed.len() == z_list.len() {
            new_z_list = parsed;
        } else {
            eprintln!("ERROR: edpiecepoint - Number of Z values does not correspond");
            process::exit(1);
        }
    }

    // Build Z mapping
    let _z_min = *z_list.iter().min().unwrap_or(&0);
    let mut z_map: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    for (i, &z) in z_list.iter().enumerate() {
        z_map.insert(z, new_z_list[i]);
    }

    let ix_add = args.add[0];
    let iy_add = args.add[1];
    let iz_add = args.add[2];

    // Apply remapping and offsets, filter out removal markers
    let mut output_pieces: Vec<(i32, i32, i32)> = Vec::new();
    for p in &pieces {
        let new_z = z_map.get(&p.2).copied().unwrap_or(p.2);
        if new_z < -999 || new_z > -990 {
            output_pieces.push((p.0 + ix_add, p.1 + iy_add, new_z + iz_add));
        }
        // Values -999 to -990 are used to remove pieces
    }

    // Write output
    let mut out = std::fs::File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: edpiecepoint - creating {}: {}", args.output, e);
        process::exit(1);
    });
    for (x, y, z) in &output_pieces {
        writeln!(out, "{:8}{:8}{:8}", x, y, z).unwrap();
    }
}
