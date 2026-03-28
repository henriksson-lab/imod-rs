use clap::Parser;
use imod_core::MrcMode;
use imod_math::parse_list;
use imod_mrc::MrcReader;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};

/// Substitute sections from one montage file into another, with optional
/// linear intensity scaling.  Can also be used with ordinary image stacks
/// by making dummy piece lists.
#[derive(Parser)]
#[command(name = "fixmont", about = "Fix montage piece coordinates / substitute sections")]
struct Args {
    /// Image file to correct (will be modified in place)
    #[arg(short = 'f', long)]
    file: String,

    /// Piece list for file to correct
    #[arg(short = 'p', long)]
    pieces: String,

    /// Image file with correction sections
    #[arg(short = 'c', long)]
    correction: String,

    /// Piece list for correction file
    #[arg(long)]
    corr_pieces: String,

    /// List of sections to take from correction file (ranges OK, e.g. "0-5,7,9")
    /// Use "all" for all sections.
    #[arg(short = 's', long, default_value = "all")]
    sections: String,

    /// Per-section intensity scale factors as "mult,add" pairs (comma-separated).
    /// E.g. "1.0,0.0,1.1,5.0" for two sections. Default: 1,0 for all.
    #[arg(long)]
    scale: Option<String>,
}

struct PieceCoord {
    x: i32,
    y: i32,
    z: i32,
}

fn read_piece_list(path: &str) -> Vec<PieceCoord> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: FIXMONT - opening piece list {path}: {e}");
        std::process::exit(1);
    });
    let reader = BufReader::new(f);
    let mut pieces = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .collect();
        if parts.len() >= 3 {
            pieces.push(PieceCoord {
                x: parts[0],
                y: parts[1],
                z: parts[2],
            });
        }
    }
    pieces
}

fn main() {
    let args = Args::parse();

    // Read headers
    let reader1 = MrcReader::open(&args.file).unwrap_or_else(|e| {
        eprintln!("ERROR: FIXMONT - opening file to correct: {e}");
        std::process::exit(1);
    });
    let h1 = reader1.header().clone();
    let _nx = h1.nx as usize;
    let _ny = h1.ny as usize;
    let mode = h1.mode;
    drop(reader1);

    let _same_file = args.correction == args.file;
    let mut reader_corr = MrcReader::open(&args.correction).unwrap_or_else(|e| {
        eprintln!("ERROR: FIXMONT - opening correction file: {e}");
        std::process::exit(1);
    });
    let h2 = reader_corr.header().clone();

    if h2.nx != h1.nx || h2.ny != h1.ny {
        eprintln!("ERROR: FIXMONT - input image sizes do not match");
        std::process::exit(1);
    }

    let pc_list1 = read_piece_list(&args.pieces);
    let pc_list2 = read_piece_list(&args.corr_pieces);

    // Get list of Z values to process
    let z_list: Vec<i32> = if args.sections == "all" {
        let mut zs: Vec<i32> = pc_list2.iter().map(|p| p.z).collect();
        zs.sort();
        zs.dedup();
        zs
    } else {
        parse_list(&args.sections).unwrap_or_else(|e| {
            eprintln!("ERROR: FIXMONT - parsing section list: {e}");
            std::process::exit(1);
        })
    };

    // Parse scale factors
    let mut scale_mult = vec![1.0_f32; z_list.len()];
    let mut scale_add = vec![0.0_f32; z_list.len()];
    if let Some(ref sc) = args.scale {
        let vals: Vec<f32> = sc
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();
        for i in 0..z_list.len() {
            if i * 2 < vals.len() {
                scale_mult[i] = vals[i * 2];
            }
            if i * 2 + 1 < vals.len() {
                scale_add[i] = vals[i * 2 + 1];
            }
        }
    }

    let data_mode = MrcMode::from_i32(mode);
    let is_integer = matches!(mode, 0 | 9..=15);
    let val_max: f32 = if is_integer {
        let ipow = if mode == 0 { 8 } else { mode.max(8) };
        (1i32 << ipow) as f32
    } else {
        0.0
    };

    // Open the target file for read+write
    let mut target_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&args.file)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: FIXMONT - opening file for writing: {e}");
            std::process::exit(1);
        });

    let slice_bytes = h1.slice_size_bytes();
    let data_offset = h1.data_offset();
    let mut n_replace = 0;

    for (ilis, &iz) in z_list.iter().enumerate() {
        let scmul = scale_mult[ilis];
        let scad = scale_add[ilis];
        let mut any_pc = false;

        for (ipc2, pc2) in pc_list2.iter().enumerate() {
            if pc2.z != iz {
                continue;
            }

            for (ipc1, pc1) in pc_list1.iter().enumerate() {
                if pc1.z == iz && pc1.x == pc2.x && pc1.y == pc2.y {
                    any_pc = true;

                    // Read section from correction file
                    let mut section = reader_corr.read_slice_f32(ipc2).unwrap_or_else(|e| {
                        eprintln!("ERROR: FIXMONT - reading correction section: {e}");
                        std::process::exit(1);
                    });

                    // Apply scaling
                    if is_integer {
                        for v in section.iter_mut() {
                            *v = (*v * scmul + scad).clamp(0.0, val_max);
                        }
                    } else {
                        for v in section.iter_mut() {
                            *v = *v * scmul + scad;
                        }
                    }

                    // Write section back to target file at position ipc1
                    let offset = data_offset + (ipc1 as u64 * slice_bytes as u64);
                    target_file.seek(SeekFrom::Start(offset)).unwrap();

                    // Convert f32 back to the file's native mode
                    let bytes = match data_mode {
                        Some(MrcMode::Byte) => {
                            section.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect::<Vec<u8>>()
                        }
                        Some(MrcMode::Short) => {
                            let mut buf = Vec::with_capacity(section.len() * 2);
                            for &v in &section {
                                buf.extend_from_slice(&(v as i16).to_le_bytes());
                            }
                            buf
                        }
                        Some(MrcMode::Float) => {
                            let mut buf = Vec::with_capacity(section.len() * 4);
                            for &v in &section {
                                buf.extend_from_slice(&v.to_le_bytes());
                            }
                            buf
                        }
                        Some(MrcMode::UShort) => {
                            let mut buf = Vec::with_capacity(section.len() * 2);
                            for &v in &section {
                                buf.extend_from_slice(
                                    &(v.clamp(0.0, 65535.0) as u16).to_le_bytes(),
                                );
                            }
                            buf
                        }
                        _ => {
                            eprintln!("ERROR: FIXMONT - unsupported mode {mode}");
                            std::process::exit(1);
                        }
                    };
                    target_file.write_all(&bytes).unwrap();
                }
            }
        }
        if any_pc {
            n_replace += 1;
        }
    }

    eprintln!("FIXMONT: {n_replace} sections replaced");
}
