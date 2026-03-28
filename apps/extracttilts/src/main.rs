//! extracttilts - Extract tilt angles or other per-section metadata from MRC
//! extended headers.
//!
//! Reads an MRC file's extended header (SerialEM/IMOD format) and extracts
//! tilt angles, magnifications, stage positions, exposure doses, defocus,
//! or pixel spacing for each section.

use clap::Parser;
use imod_core::ExtHeaderType;
use imod_mrc::MrcReader;
use std::io::Write;
use std::process;

#[derive(Parser)]
#[command(name = "extracttilts", about = "Extract tilt angles or other per-section info from MRC extended header")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// Output file (default: print to stdout)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Extract tilt angles (default if nothing else specified)
    #[arg(short = 't', long)]
    tilts: bool,

    /// Extract magnifications
    #[arg(short = 'm', long)]
    mag: bool,

    /// Extract stage positions (X, Y)
    #[arg(short = 's', long)]
    stage: bool,

    /// Extract intensity (C2) values
    #[arg(long)]
    intensities: bool,

    /// Extract exposure dose
    #[arg(long)]
    exp: bool,

    /// Extract defocus values
    #[arg(long)]
    defocus: bool,

    /// Extract pixel spacing
    #[arg(long)]
    pixel: bool,

    /// Warn if extracted tilt angles look suspicious
    #[arg(short = 'w', long)]
    warn: bool,
}

/// IMOD SERI extended header: nint and nreal encode which fields are present.
/// The nreal field is a bitfield: bit 0 = tilt (2 bytes), bit 1 = piece coords
/// (6 bytes), bit 2 = stage (4 bytes), bit 3 = mag (2 bytes), bit 4 = intensity
/// (2 bytes), bit 5 = exposure dose (2 bytes).
/// Fields stored as i16 are scaled: tilt by 100, stage by 25, etc.
///
/// With FEI headers (128 bytes/section), layout is different.

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExtractType {
    Tilt,
    Mag,
    StageXY,
    Intensity,
    Dose,
    Defocus,
    Pixel,
}

fn main() {
    let args = Args::parse();

    let reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: EXTRACTTILTS - opening {}: {}", args.input, e);
        process::exit(1);
    });

    let header = reader.header();
    let nz = header.nz as usize;
    let ext_data = reader.ext_header();
    let ext_type = header.ext_header_type();

    // Determine what to extract
    let num_types = [
        args.tilts,
        args.mag,
        args.stage,
        args.intensities,
        args.exp,
        args.defocus,
        args.pixel,
    ]
    .iter()
    .filter(|&&b| b)
    .count();

    if num_types > 1 {
        eprintln!("ERROR: EXTRACTTILTS - specify only one type of data to extract");
        process::exit(1);
    }

    let extract_type = if args.mag {
        ExtractType::Mag
    } else if args.stage {
        ExtractType::StageXY
    } else if args.intensities {
        ExtractType::Intensity
    } else if args.exp {
        ExtractType::Dose
    } else if args.defocus {
        ExtractType::Defocus
    } else if args.pixel {
        ExtractType::Pixel
    } else {
        ExtractType::Tilt
    };

    if ext_data.is_empty() {
        eprintln!("ERROR: EXTRACTTILTS - no extended header data in this file");
        process::exit(1);
    }

    // Extract values based on extended header format
    let (values, values2) = match ext_type {
        ExtHeaderType::Seri | ExtHeaderType::None | ExtHeaderType::Unknown => {
            extract_seri(header, ext_data, nz, extract_type)
        }
        ExtHeaderType::Fei => extract_fei(ext_data, nz, extract_type),
        ExtHeaderType::Agar => {
            eprintln!("ERROR: EXTRACTTILTS - Agard extended header not supported");
            process::exit(1);
        }
    };

    if values.is_empty() {
        eprintln!(
            "ERROR: EXTRACTTILTS - no {:?} information in this image file",
            extract_type
        );
        process::exit(1);
    }

    // Write output
    let mut out: Box<dyn Write> = if let Some(ref path) = args.output {
        Box::new(
            std::fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("ERROR: EXTRACTTILTS - creating output file: {}", e);
                process::exit(1);
            }),
        )
    } else {
        Box::new(std::io::stdout())
    };

    for (i, &v) in values.iter().enumerate() {
        match extract_type {
            ExtractType::Tilt => writeln!(out, "{:7.2}", v).unwrap(),
            ExtractType::Mag => writeln!(out, "{:7}", v as i32).unwrap(),
            ExtractType::StageXY => {
                let v2 = values2.get(i).copied().unwrap_or(0.0);
                writeln!(out, "{:9.2}{:9.2}", v, v2).unwrap();
            }
            ExtractType::Intensity => writeln!(out, "{:8.5}", v).unwrap(),
            ExtractType::Dose => writeln!(out, "{:13.5}", v).unwrap(),
            ExtractType::Defocus | ExtractType::Pixel => writeln!(out, "{:11.3}", v).unwrap(),
        }
    }

    if args.output.is_some() {
        eprintln!("{} values output to file", values.len());
    }

    // Warn about suspicious tilts
    if extract_type == ExtractType::Tilt && args.warn {
        let near_zero = values.iter().filter(|&&v| v.abs() < 0.1).count();
        let over_95 = values.iter().filter(|&&v| v.abs() > 95.0).count();
        if values.len() > 2 && near_zero > values.len() / 2 {
            eprintln!(
                "WARNING: extracttilts - {} of the extracted tilt angles are near zero",
                near_zero
            );
        }
        if over_95 > 0 {
            eprintln!(
                "WARNING: extracttilts - {} of the extracted tilt angles are greater than 95 degrees",
                over_95
            );
        }
    }
}

/// Extract from IMOD/SerialEM style extended header.
/// nreal is a bitfield indicating which 2-byte fields are present per section.
/// nint indicates the number of integer shorts per section before the real shorts.
fn extract_seri(
    header: &imod_mrc::MrcHeader,
    ext_data: &[u8],
    nz: usize,
    extract_type: ExtractType,
) -> (Vec<f32>, Vec<f32>) {
    let nint = header.nint as u16;
    let nreal = header.nreal as u16;

    // Compute bytes per section from nreal bitfield
    // Each bit in nreal corresponds to a data item of known size
    let mut bytes_per_section; // computed below
    let mut field_found = false;
    let mut field_offset = 0usize;

    // nreal bits map to sizes
    // bit 0 (1): tilt, 2 bytes
    // bit 1 (2): piece coords, 6 bytes
    // bit 2 (4): stage XY, 4 bytes
    // bit 3 (8): magnification, 2 bytes
    // bit 4 (16): intensity/C2, 2 bytes
    // bit 5 (32): exposure dose, 4 bytes as float
    // bit 6 (64): ?? defocus or other
    let bit_sizes = [(1u16, 2usize), (2, 6), (4, 4), (8, 2), (16, 2), (32, 4)];
    let bit_types = [
        (1u16, ExtractType::Tilt),
        (4, ExtractType::StageXY),
        (8, ExtractType::Mag),
        (16, ExtractType::Intensity),
        (32, ExtractType::Dose),
    ];

    // Calculate total bytes per section and find field offset
    let mut current_offset = nint as usize * 2;
    for &(bit, size) in &bit_sizes {
        if nreal & bit != 0 {
            // Check if this is our target field
            for &(bt, ref et) in &bit_types {
                if bt == bit && *et == extract_type {
                    field_offset = current_offset;
                    field_found = true;
                }
            }
            current_offset += size;
        }
    }
    bytes_per_section = current_offset;

    if bytes_per_section == 0 {
        // Fall back: use next / nz if available
        if header.next > 0 && nz > 0 {
            bytes_per_section = header.next as usize / nz;
        }
        if bytes_per_section == 0 {
            return (vec![], vec![]);
        }
    }

    if !field_found {
        return (vec![], vec![]);
    }

    let mut values = Vec::with_capacity(nz);
    let mut values2 = Vec::with_capacity(nz);

    for iz in 0..nz {
        let base = iz * bytes_per_section + field_offset;
        if base + 2 > ext_data.len() {
            break;
        }

        match extract_type {
            ExtractType::Tilt => {
                let raw = i16::from_le_bytes([ext_data[base], ext_data[base + 1]]);
                values.push(raw as f32 / 100.0);
            }
            ExtractType::StageXY => {
                if base + 4 > ext_data.len() {
                    break;
                }
                let x = i16::from_le_bytes([ext_data[base], ext_data[base + 1]]);
                let y = i16::from_le_bytes([ext_data[base + 2], ext_data[base + 3]]);
                values.push(x as f32 / 25.0);
                values2.push(y as f32 / 25.0);
            }
            ExtractType::Mag => {
                let raw = i16::from_le_bytes([ext_data[base], ext_data[base + 1]]);
                values.push(raw as f32);
            }
            ExtractType::Intensity => {
                let raw = i16::from_le_bytes([ext_data[base], ext_data[base + 1]]);
                values.push(raw as f32 / 25000.0);
            }
            ExtractType::Dose => {
                if base + 4 > ext_data.len() {
                    break;
                }
                let raw = f32::from_le_bytes([
                    ext_data[base],
                    ext_data[base + 1],
                    ext_data[base + 2],
                    ext_data[base + 3],
                ]);
                values.push(raw);
            }
            _ => {}
        }
    }

    (values, values2)
}

/// Extract from FEI extended header (128 bytes per section).
fn extract_fei(ext_data: &[u8], nz: usize, extract_type: ExtractType) -> (Vec<f32>, Vec<f32>) {
    const FEI_RECORD_SIZE: usize = 128;

    // FEI1 header field offsets (bytes within each 128-byte record):
    // 0: a_tilt (f32) - alpha tilt
    // 4: b_tilt (f32) - beta tilt
    // 8: x_stage (f32)
    // 12: y_stage (f32)
    // 16: z_stage (f32)
    // 20: x_shift (f32)
    // 24: y_shift (f32)
    // 28: defocus (f32)
    // 32: exp_time (f32)
    // 36: mean_int (f32)
    // 40: tilt_axis (f32)
    // 44: pixel_size (f32)
    // 48: magnification (f32)

    let mut values = Vec::with_capacity(nz);
    let mut values2 = Vec::with_capacity(nz);

    for iz in 0..nz {
        let base = iz * FEI_RECORD_SIZE;
        if base + FEI_RECORD_SIZE > ext_data.len() {
            break;
        }

        let read_f32 = |off: usize| -> f32 {
            let o = base + off;
            f32::from_le_bytes([ext_data[o], ext_data[o + 1], ext_data[o + 2], ext_data[o + 3]])
        };

        match extract_type {
            ExtractType::Tilt => values.push(read_f32(0)),
            ExtractType::StageXY => {
                values.push(read_f32(8));
                values2.push(read_f32(12));
            }
            ExtractType::Mag => values.push(read_f32(48)),
            ExtractType::Intensity => values.push(read_f32(36)),
            ExtractType::Dose => values.push(read_f32(32)), // exposure time as proxy
            ExtractType::Defocus => values.push(read_f32(28)),
            ExtractType::Pixel => values.push(read_f32(44)),
        }
    }

    (values, values2)
}
