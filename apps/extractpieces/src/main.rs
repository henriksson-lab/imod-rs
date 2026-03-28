//! extractpieces - Extract piece (montage tile) coordinates from an MRC
//! extended header and write a piece list file.
//!
//! The piece list file has one line per section: ixPiece iyPiece izPiece

use clap::Parser;
use imod_mrc::MrcReader;
use std::io::Write;
use std::process;

#[derive(Parser)]
#[command(name = "extractpieces", about = "Extract piece coordinates from MRC extended header")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// Output piece list file
    #[arg(short = 'o', long)]
    output: String,
}

fn main() {
    let args = Args::parse();

    let reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTRACTPIECES - opening {}: {}", args.input, e);
        process::exit(1);
    });

    let header = reader.header();
    let nz = header.nz as usize;
    let ext_data = reader.ext_header();
    let nint = header.nint as u16;
    let nreal = header.nreal as u16;

    if ext_data.is_empty() {
        eprintln!("There are no piece coordinates in this image file");
        process::exit(0);
    }

    // Check if piece coordinates are present: bit 1 (value 2) of nreal
    if nreal & 2 == 0 {
        eprintln!("There are no piece coordinates in this image file");
        process::exit(0);
    }

    // Compute offset to piece coordinate fields and bytes per section
    // nreal bit layout: bit 0 (1)=tilt 2B, bit 1 (2)=piece 6B, bit 2 (4)=stage 4B, etc.
    let mut piece_offset = nint as usize * 2; // skip integer fields
    if nreal & 1 != 0 {
        piece_offset += 2; // skip tilt
    }

    // Bytes per section
    let mut bytes_per_section = nint as usize * 2;
    let bit_sizes: [(u16, usize); 6] = [(1, 2), (2, 6), (4, 4), (8, 2), (16, 2), (32, 4)];
    for &(bit, size) in &bit_sizes {
        if nreal & bit != 0 {
            bytes_per_section += size;
        }
    }

    if bytes_per_section == 0 {
        eprintln!("There are no piece coordinates in this image file");
        process::exit(0);
    }

    let mut pieces = Vec::with_capacity(nz);

    for iz in 0..nz {
        let base = iz * bytes_per_section + piece_offset;
        if base + 6 > ext_data.len() {
            break;
        }
        let ix = i16::from_le_bytes([ext_data[base], ext_data[base + 1]]) as i32;
        let iy = i16::from_le_bytes([ext_data[base + 2], ext_data[base + 3]]) as i32;
        let iz_piece = i16::from_le_bytes([ext_data[base + 4], ext_data[base + 5]]) as i32;
        pieces.push((ix, iy, iz_piece));
    }

    if pieces.is_empty() {
        eprintln!("There are no piece coordinates in this image file");
        process::exit(0);
    }

    let mut out = std::fs::File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTRACTPIECES - creating output file: {}", e);
        process::exit(1);
    });

    for &(ix, iy, iz) in &pieces {
        writeln!(out, "{:9}{:9}{:7}", ix, iy, iz).unwrap();
    }

    eprintln!("{} piece coordinates output to file", pieces.len());
}
