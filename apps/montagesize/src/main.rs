use clap::Parser;
use imod_mrc::MrcReader;
use std::io::{BufRead, BufReader};

/// Determine the X, Y, and Z dimensions of a montaged image file from
/// piece coordinates in the header or a separate piece list file.
#[derive(Parser)]
#[command(name = "montagesize", about = "Compute montage output size from piece list")]
struct Args {
    /// Input MRC image file
    image_file: String,

    /// Piece list file (optional; if omitted, reads from image header)
    piece_file: Option<String>,
}

struct PieceCoord {
    x: i32,
    y: i32,
    z: i32,
}

fn read_piece_list(path: &str) -> Vec<PieceCoord> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("ERROR: MONTAGESIZE - opening piece list: {e}");
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

/// Parse piece coordinates from the extended header (SERI-type: 2 shorts per section
/// for tilt + 2 shorts for piece X,Y at bytes offset 8,10 in each section's block).
/// This is a simplified version; real IMOD uses flags/nint/nreal fields.
fn read_pieces_from_header(reader: &MrcReader, nz: i32) -> Vec<PieceCoord> {
    let ext = reader.ext_header();
    let h = reader.header();

    // Check if ext header has piece coordinate flags
    // IMOD stores piece coords when nreal has bit 0 set (tilt) and uses
    // nint/nreal to determine per-section byte counts
    let nint = h.nint as usize;
    let nreal = h.nreal;
    let bytes_per_section = if nint > 0 || nreal != 0 {
        // Count bytes from nreal bit flags
        let mut nbytes = nint * 2; // nint short integers
        // Each bit in nreal means 4 bytes of real/int data
        for bit in 0..16 {
            if (nreal >> bit) & 1 != 0 {
                // bits 0,1,2,3,4 = 4 bytes each; bit 5 = 2 bytes (unused usually)
                nbytes += if bit < 5 { 4 } else { 2 };
            }
        }
        nbytes
    } else {
        0
    };

    if bytes_per_section == 0 || ext.is_empty() {
        return Vec::new();
    }

    // Check if piece coordinates flag (bit 0 of iflags = nreal bit pattern)
    // Piece coords are typically stored as: short ix, short iy after a certain offset
    // determined by previous flag bits. This simplified version checks for the
    // montage flag.
    let has_pieces = (nreal & 1) != 0; // bit 0 = tilt angle (2 bytes)
    // Actually, piece coordinates in IMOD extended header are stored when
    // the "piece coordinates" bit is set. Let's just return empty if no info.
    if !has_pieces {
        return Vec::new();
    }

    // For real IMOD piece coordinates from ext header, we would need
    // the full get_extra_header_pieces logic. Return empty for now;
    // users should provide piece list files.
    let _ = nz;
    Vec::new()
}

/// Check a piece coordinate list in one dimension: determine number of pieces,
/// overlap, and minimum coordinate.
fn check_list(coords: &[i32], frame_size: i32) -> Option<(i32, i32, i32)> {
    if coords.is_empty() {
        return None;
    }
    let mut sorted: Vec<i32> = coords.to_vec();
    sorted.sort();
    sorted.dedup();

    let n_pieces = sorted.len() as i32;
    if n_pieces <= 0 {
        return None;
    }
    let min_piece = sorted[0];

    let overlap = if n_pieces > 1 {
        // Overlap = frame_size - spacing between consecutive pieces
        let spacing = sorted[1] - sorted[0];
        frame_size - spacing
    } else {
        0
    };

    Some((min_piece, n_pieces, overlap))
}

fn main() {
    let args = Args::parse();

    let reader = MrcReader::open(&args.image_file).unwrap_or_else(|e| {
        eprintln!("ERROR: MONTAGESIZE - opening image: {e}");
        std::process::exit(1);
    });
    let h = reader.header();
    let nx = h.nx;
    let ny = h.ny;
    let nz = h.nz;

    let pieces = if let Some(ref pf) = args.piece_file {
        let p = read_piece_list(pf);
        if p.is_empty() {
            eprintln!("ERROR: MONTAGESIZE - No piece list information in the piece list file");
            std::process::exit(1);
        }
        p
    } else {
        let p = read_pieces_from_header(&reader, nz);
        if p.is_empty() {
            eprintln!("ERROR: MONTAGESIZE - No piece list information in this image file");
            eprintln!("Provide a piece list file as second argument");
            std::process::exit(1);
        }
        p
    };

    let num_pc = pieces.len().min(nz as usize);

    // Find min and max Z
    let min_z = pieces[..num_pc].iter().map(|p| p.z).min().unwrap();
    let max_z = pieces[..num_pc].iter().map(|p| p.z).max().unwrap();
    let num_sections = max_z + 1 - min_z;

    // Check X piece list
    let x_coords: Vec<i32> = pieces[..num_pc].iter().map(|p| p.x).collect();
    let y_coords: Vec<i32> = pieces[..num_pc].iter().map(|p| p.y).collect();

    let (_, nx_pieces, nx_overlap) = check_list(&x_coords, nx).unwrap_or_else(|| {
        eprintln!("ERROR: MONTAGESIZE - Piece list information not good");
        std::process::exit(1);
    });
    let (_, ny_pieces, ny_overlap) = check_list(&y_coords, ny).unwrap_or_else(|| {
        eprintln!("ERROR: MONTAGESIZE - Piece list information not good");
        std::process::exit(1);
    });

    if nx_pieces <= 0 || ny_pieces <= 0 {
        eprintln!("ERROR: MONTAGESIZE - Piece list information not good");
        std::process::exit(1);
    }

    let nx_tot_pix = nx_pieces * (nx - nx_overlap) + nx_overlap;
    let ny_tot_pix = ny_pieces * (ny - ny_overlap) + ny_overlap;

    println!(" Total NX, NY, NZ:{nx_tot_pix:10}{ny_tot_pix:10}{num_sections:10}");

    // Validate piece list size vs image Z
    if let Some(ref _pf) = args.piece_file {
        if nz < pieces.len() as i32 {
            eprintln!(
                "ERROR: MONTAGESIZE - The Z size of the image file is smaller than the size of the piece list"
            );
            std::process::exit(2);
        }
        if nz > pieces.len() as i32 {
            eprintln!(
                "ERROR: MONTAGESIZE - The Z size of the image file is larger than the size of the piece list"
            );
            std::process::exit(3);
        }
    }
}
