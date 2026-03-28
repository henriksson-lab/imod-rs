//! fixboundaries - Fix boundaries in montage after parallel writing.
//!
//! Rewrites data near chunk boundaries after direct writing in parallel
//! to an output file. Takes the main image file and a boundary info file,
//! and uses the boundary file entries to rewrite guard regions.
//!
//! Translated from IMOD's fixboundaries.f90

use clap::Parser;
use imod_mrc::MrcReader;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process;

#[derive(Parser)]
#[command(
    name = "fixboundaries",
    about = "Fix boundaries in montage after parallel writing"
)]
struct Args {
    /// Main output MRC image file to fix
    main_file: String,

    /// Boundary information file (from parallel writing)
    info_file: String,
}

/// Represents boundary region info parsed from the info file
struct BoundaryRegion {
    /// Path to the guard file for this region
    guard_file: String,
    /// Section indices for the two boundaries (-1 if not applicable)
    iz_secs: [i32; 2],
    /// Starting line for each boundary region (-1 if not applicable)
    line_start: [i32; 2],
}

fn main() {
    let args = Args::parse();

    // Open main file to get dimensions
    let reader = MrcReader::open(&args.main_file).unwrap_or_else(|e| {
        eprintln!("ERROR: fixboundaries - opening main file: {}", e);
        process::exit(1);
    });
    let header = reader.header();
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;
    let mode = header.mode;

    // Compute bytes per pixel based on mode
    let bytes_per_pixel: usize = match mode {
        0 => 1, // byte
        1 => 2, // i16
        2 => 4, // f32
        6 => 2, // u16
        _ => {
            eprintln!("ERROR: fixboundaries - unsupported mode {}", mode);
            process::exit(1);
        }
    };
    let line_bytes = nx * bytes_per_pixel;

    // Parse info file to get boundary regions
    // The info file format is specific to IMOD parallel writing.
    // For now, print a message about the expected workflow.
    let info_content = std::fs::read_to_string(&args.info_file).unwrap_or_else(|e| {
        eprintln!(
            "ERROR: fixboundaries - reading info file {}: {}",
            args.info_file, e
        );
        process::exit(1);
    });

    let lines: Vec<&str> = info_content.lines().collect();
    if lines.is_empty() {
        eprintln!("ERROR: fixboundaries - empty info file");
        process::exit(1);
    }

    // Parse the info file header to get properties
    // Format: first line has num_files, if_all_sec, lines_guard
    let header_vals: Vec<i32> = lines[0]
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if header_vals.len() < 3 {
        eprintln!("ERROR: fixboundaries - invalid info file header");
        process::exit(1);
    }
    let num_files = header_vals[0] as usize;
    let if_all_sec = header_vals[1] != 0;
    let lines_guard = header_vals[2] as usize;

    // Parse boundary regions
    let mut regions: Vec<BoundaryRegion> = Vec::new();
    let mut line_idx = 1;
    for _ in 0..num_files {
        if line_idx >= lines.len() {
            break;
        }
        let guard_file = lines[line_idx].trim().to_string();
        line_idx += 1;
        if line_idx >= lines.len() {
            break;
        }
        let vals: Vec<i32> = lines[line_idx]
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        line_idx += 1;
        let iz_secs = if vals.len() >= 2 {
            [vals[0], vals[1]]
        } else {
            [-1, -1]
        };
        let line_start = if vals.len() >= 4 {
            [vals[2], vals[3]]
        } else {
            [-1, -1]
        };
        regions.push(BoundaryRegion {
            guard_file,
            iz_secs,
            line_start,
        });
    }

    // Open main file for read/write
    let mut main_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&args.main_file)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: fixboundaries - opening main file for writing: {}", e);
            process::exit(1);
        });

    let header_size = 1024 + header.next as u64;

    // Process each boundary region
    for region in &regions {
        let guard_reader = MrcReader::open(&region.guard_file).unwrap_or_else(|e| {
            eprintln!(
                "ERROR: fixboundaries - opening guard file {}: {}",
                region.guard_file, e
            );
            process::exit(1);
        });
        let guard_header = guard_reader.header();
        let guard_nx = guard_header.nx as usize;
        let guard_header_size = 1024 + guard_header.next as u64;

        let mut guard_raw = std::fs::File::open(&region.guard_file).unwrap_or_else(|e| {
            eprintln!("ERROR: fixboundaries - opening guard file: {}", e);
            process::exit(1);
        });

        if if_all_sec {
            // Write guard lines for all sections
            for iz in 0..nz {
                for j in 0..2 {
                    if region.line_start[j] >= 0 {
                        let guard_section = iz * 2 + j;
                        let guard_offset = guard_header_size
                            + (guard_section * guard_nx * bytes_per_pixel) as u64
                                * guard_header.ny as u64
                                / (guard_header.nz as u64);

                        let main_section_offset = header_size
                            + (iz as u64) * (ny as u64) * line_bytes as u64
                            + (region.line_start[j] as u64) * line_bytes as u64;

                        let mut buf = vec![0u8; line_bytes];
                        for _line in 0..lines_guard {
                            guard_raw.seek(SeekFrom::Start(guard_offset)).ok();
                            guard_raw.read_exact(&mut buf).unwrap_or_else(|_| {
                                eprintln!("ERROR: fixboundaries - reading guard file");
                                process::exit(1);
                            });
                            main_file
                                .seek(SeekFrom::Start(main_section_offset))
                                .ok();
                            main_file.write_all(&buf).unwrap_or_else(|_| {
                                eprintln!("ERROR: fixboundaries - writing to main file");
                                process::exit(1);
                            });
                        }
                    }
                }
            }
        } else {
            // Write just the two boundary sections
            for j in 0..2 {
                if region.iz_secs[j] >= 0 {
                    let guard_section_offset = guard_header_size
                        + (j as u64) * (guard_nx as u64 * bytes_per_pixel as u64)
                            * guard_header.ny as u64
                            / guard_header.nz.max(1) as u64;

                    let main_offset = header_size
                        + (region.iz_secs[j] as u64) * (ny as u64) * line_bytes as u64
                        + (region.line_start[j] as u64) * line_bytes as u64;

                    let mut buf = vec![0u8; line_bytes];
                    for line in 0..lines_guard {
                        let g_off = guard_section_offset + (line as u64) * line_bytes as u64;
                        let m_off = main_offset + (line as u64) * line_bytes as u64;

                        guard_raw.seek(SeekFrom::Start(g_off)).ok();
                        if guard_raw.read_exact(&mut buf).is_err() {
                            eprintln!("ERROR: fixboundaries - reading from guard file");
                            process::exit(1);
                        }
                        main_file.seek(SeekFrom::Start(m_off)).ok();
                        if main_file.write_all(&buf).is_err() {
                            eprintln!("ERROR: fixboundaries - writing to main file");
                            process::exit(1);
                        }
                    }
                }
            }
        }
    }

    println!(
        "Fixed {} boundary regions in {}",
        regions.len(),
        args.main_file
    );
}
