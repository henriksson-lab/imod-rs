//! alterheader - Modify MRC header fields without rewriting image data.
//!
//! Supports modifying pixel size, origin, cell dimensions, labels, min/max/mean,
//! tilt angles, axis mapping, sample size, and various fix-up operations.

use binrw::BinWrite;
use clap::Parser;
use imod_mrc::{MrcHeader, MrcReader};
use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::process;

#[derive(Parser)]
#[command(name = "alterheader", about = "Modify MRC header fields in-place")]
struct Args {
    /// Input MRC file to modify
    input: String,

    /// Set origin (x, y, z) in Angstroms
    #[arg(long, num_args = 3, value_names = ["X", "Y", "Z"])]
    org: Option<Vec<f32>>,

    /// Set cell dimensions (x, y, z) in Angstroms
    #[arg(long, num_args = 3, value_names = ["X", "Y", "Z"])]
    cel: Option<Vec<f32>>,

    /// Set pixel spacing (delta x, y, z); adjusts cell size to match
    #[arg(long, num_args = 3, value_names = ["DX", "DY", "DZ"])]
    del: Option<Vec<f32>>,

    /// Set axis mapping (columns, rows, sections)
    #[arg(long, num_args = 3, value_names = ["C", "R", "S"])]
    map: Option<Vec<i32>>,

    /// Set sample size (mx, my, mz)
    #[arg(long, num_args = 3, value_names = ["MX", "MY", "MZ"])]
    sam: Option<Vec<i32>>,

    /// Set current tilt angles (alpha, beta, gamma)
    #[arg(long, num_args = 3, value_names = ["A", "B", "G"])]
    tlt: Option<Vec<f32>>,

    /// Set original tilt angles
    #[arg(long = "firsttlt", num_args = 3, value_names = ["A", "B", "G"])]
    tlt_orig: Option<Vec<f32>>,

    /// Rotate current tilt angles by these amounts
    #[arg(long = "rottlt", num_args = 3, value_names = ["DA", "DB", "DG"])]
    tlt_rot: Option<Vec<f32>>,

    /// Recompute min/max/mean from image data
    #[arg(long)]
    mmm: bool,

    /// Recompute RMS from image data
    #[arg(long)]
    rms: bool,

    /// Fix pixel: set sample = image size, cell = image size (pixel spacing = 1)
    #[arg(long)]
    fixpixel: bool,

    /// Fix grid: set sample = image size, preserving pixel spacing
    #[arg(long = "gridfix")]
    fixgrid: bool,

    /// Set min, max, mean directly
    #[arg(long, num_args = 3, value_names = ["MIN", "MAX", "MEAN"])]
    setmmm: Option<Vec<f32>>,

    /// Invert the origin sign convention
    #[arg(long)]
    invertorg: bool,

    /// Toggle origin sign convention flag
    #[arg(long)]
    toggleorg: bool,

    /// Set space group
    #[arg(long)]
    ispg: Option<i32>,

    /// Change mode between 1 (int16) and 6 (uint16)
    #[arg(long)]
    fixmode: bool,

    /// Mark as real-space data (mode 2 for complex -> mode 2)
    #[arg(long)]
    real: bool,

    /// Mark as FFT data
    #[arg(long)]
    fft: bool,

    /// Set nxstart, nystart, nzstart
    #[arg(long, num_args = 3, value_names = ["NX", "NY", "NZ"])]
    start: Option<Vec<i32>>,

    /// Add a label string
    #[arg(long = "title")]
    title: Option<String>,

    /// Remove labels at these positions (1-based, comma-separated)
    #[arg(long = "remove")]
    remove: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Read the current header
    let reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: ALTERHEADER - opening {}: {}", args.input, e);
        process::exit(1);
    });
    let mut header = reader.header().clone();
    drop(reader);

    let mut modified = false;

    // Origin
    if let Some(ref v) = args.org {
        header.xorg = v[0];
        header.yorg = v[1];
        header.zorg = v[2];
        eprintln!("Origin set to {} {} {}", v[0], v[1], v[2]);
        modified = true;
    }

    // Cell size
    if let Some(ref v) = args.cel {
        if v[0] > 0.0 && v[1] > 0.0 && v[2] > 0.0 {
            header.xlen = v[0];
            header.ylen = v[1];
            header.zlen = v[2];
            eprintln!("Cell size set to {} {} {}", v[0], v[1], v[2]);
            modified = true;
        } else {
            eprintln!("ERROR: ALTERHEADER - cell sizes must be positive");
            process::exit(1);
        }
    }

    // Pixel spacing (delta) - adjusts cell to achieve desired spacing
    if let Some(ref v) = args.del {
        if v[0] > 0.0 && v[1] > 0.0 && v[2] > 0.0 {
            header.xlen = header.mx as f32 * v[0];
            header.ylen = header.my as f32 * v[1];
            header.zlen = header.mz as f32 * v[2];
            eprintln!("Pixel spacing set to {} {} {}", v[0], v[1], v[2]);
            modified = true;
        } else {
            eprintln!("ERROR: ALTERHEADER - pixel spacing values must be positive");
            process::exit(1);
        }
    }

    // Axis mapping
    if let Some(ref v) = args.map {
        let mut counts = [0i32; 4];
        for &m in v {
            if (1..=3).contains(&m) {
                counts[m as usize] += 1;
            }
        }
        if counts[1] == 1 && counts[2] == 1 && counts[3] == 1 {
            header.mapc = v[0];
            header.mapr = v[1];
            header.maps = v[2];
            eprintln!("Axis mapping set to {} {} {}", v[0], v[1], v[2]);
            modified = true;
        } else {
            eprintln!("ERROR: ALTERHEADER - map values must be a permutation of 1, 2, 3");
            process::exit(1);
        }
    }

    // Sample size
    if let Some(ref v) = args.sam {
        if v[0] > 0 && v[1] > 0 && v[2] > 0 {
            header.mx = v[0];
            header.my = v[1];
            header.mz = v[2];
            eprintln!("Sample size set to {} {} {}", v[0], v[1], v[2]);
            modified = true;
        } else {
            eprintln!("ERROR: ALTERHEADER - sample sizes must be positive");
            process::exit(1);
        }
    }

    // Current tilt angles
    if let Some(ref v) = args.tlt {
        header.tilt_angles[3] = v[0];
        header.tilt_angles[4] = v[1];
        header.tilt_angles[5] = v[2];
        eprintln!("Current tilt angles set to {} {} {}", v[0], v[1], v[2]);
        modified = true;
    }

    // Original tilt angles
    if let Some(ref v) = args.tlt_orig {
        header.tilt_angles[0] = v[0];
        header.tilt_angles[1] = v[1];
        header.tilt_angles[2] = v[2];
        eprintln!("Original tilt angles set to {} {} {}", v[0], v[1], v[2]);
        modified = true;
    }

    // Rotate current tilt angles
    if let Some(ref v) = args.tlt_rot {
        header.tilt_angles[3] += v[0];
        header.tilt_angles[4] += v[1];
        header.tilt_angles[5] += v[2];
        eprintln!(
            "Tilt angles rotated to {} {} {}",
            header.tilt_angles[3], header.tilt_angles[4], header.tilt_angles[5]
        );
        modified = true;
    }

    // Fix pixel: sample = image size, cell = image size => pixel spacing = 1
    if args.fixpixel {
        header.mx = header.nx;
        header.my = header.ny;
        header.mz = header.nz;
        header.xlen = header.nx as f32;
        header.ylen = header.ny as f32;
        header.zlen = header.nz as f32;
        eprintln!("Pixel spacing fixed to 1.0 1.0 1.0");
        modified = true;
    }

    // Fix grid: sample = image size, preserving pixel spacing
    if args.fixgrid {
        let dx = if header.mx > 0 {
            header.xlen / header.mx as f32
        } else {
            1.0
        };
        let dy = if header.my > 0 {
            header.ylen / header.my as f32
        } else {
            1.0
        };
        let dz = if header.mz > 0 {
            header.zlen / header.mz as f32
        } else {
            1.0
        };
        header.xlen = dx * header.nx as f32;
        header.ylen = dy * header.ny as f32;
        header.zlen = dz * header.nz as f32;
        header.mx = header.nx;
        header.my = header.ny;
        header.mz = header.nz;
        eprintln!("Grid fixed: sample = image size, pixel spacing preserved");
        modified = true;
    }

    // Set min/max/mean
    if let Some(ref v) = args.setmmm {
        header.amin = v[0];
        header.amax = v[1];
        header.amean = v[2];
        eprintln!("Min/max/mean set to {} {} {}", v[0], v[1], v[2]);
        modified = true;
    }

    // Recompute min/max/mean from data
    if args.mmm || args.rms {
        let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
            eprintln!("ERROR: ALTERHEADER - opening file: {}", e);
            process::exit(1);
        });
        let nz = header.nz as usize;
        let mut dmin = f64::MAX;
        let mut dmax = f64::MIN;
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        let mut total_n = 0u64;

        for z in 0..nz {
            let slice = reader.read_slice_f32(z).unwrap_or_else(|e| {
                eprintln!("ERROR: ALTERHEADER - reading section {}: {}", z, e);
                process::exit(1);
            });
            for &v in &slice {
                let vd = v as f64;
                if vd < dmin {
                    dmin = vd;
                }
                if vd > dmax {
                    dmax = vd;
                }
                sum += vd;
                sumsq += vd * vd;
                total_n += 1;
            }
        }
        drop(reader);

        let mean = sum / total_n as f64;
        header.amin = dmin as f32;
        header.amax = dmax as f32;
        header.amean = mean as f32;

        let rms_val = ((sumsq - total_n as f64 * mean * mean) / total_n as f64).sqrt();
        header.rms = rms_val as f32;

        eprintln!(
            "Min = {}, Max = {}, Mean = {}, RMS = {}",
            header.amin, header.amax, header.amean, header.rms
        );
        modified = true;
    }

    // Invert origin
    if args.invertorg {
        header.xorg = -header.xorg;
        header.yorg = -header.yorg;
        header.zorg = -header.zorg;
        eprintln!(
            "Origin inverted to {} {} {}",
            header.xorg, header.yorg, header.zorg
        );
        modified = true;
    }

    // Toggle origin convention
    if args.toggleorg {
        let has_flag = header.is_imod() && (header.imod_flags & 4) != 0;
        if has_flag {
            header.imod_flags &= !4;
            eprintln!("Origin flag cleared (old-style origin)");
        } else {
            if !header.is_imod() {
                header.imod_stamp = MrcHeader::IMOD_STAMP;
            }
            header.imod_flags |= 4;
            eprintln!("Origin flag set (new-style origin)");
        }
        modified = true;
    }

    // Space group
    if let Some(ispg) = args.ispg {
        header.ispg = ispg;
        eprintln!("Space group set to {}", ispg);
        modified = true;
    }

    // Fix mode between 1 and 6
    if args.fixmode {
        if header.mode == 1 {
            header.mode = 6;
            eprintln!("Mode changed from 1 to 6");
            modified = true;
        } else if header.mode == 6 {
            header.mode = 1;
            eprintln!("Mode changed from 6 to 1");
            modified = true;
        } else {
            eprintln!("ERROR: ALTERHEADER - fixmode only works for mode 1 or 6");
            process::exit(1);
        }
    }

    // Real mode
    if args.real {
        if header.mode == 3 || header.mode == 4 {
            header.mode = 2;
            eprintln!("Mode set to 2 (real/float)");
            modified = true;
        } else {
            eprintln!("Mode is already non-complex");
        }
    }

    // FFT mode
    if args.fft {
        if header.mode == 2 {
            header.mode = 4;
            eprintln!("Mode set to 4 (complex float)");
            modified = true;
        } else {
            eprintln!("Mode is already complex or not float");
        }
    }

    // Start indices
    if let Some(ref v) = args.start {
        header.nxstart = v[0];
        header.nystart = v[1];
        header.nzstart = v[2];
        eprintln!("Start indices set to {} {} {}", v[0], v[1], v[2]);
        modified = true;
    }

    // Remove labels
    if let Some(ref remove_str) = args.remove {
        let indices: Vec<usize> = imod_math::parselist::parse_list(remove_str)
            .unwrap_or_else(|e| {
                eprintln!("ERROR: ALTERHEADER - parsing remove list: {}", e);
                process::exit(1);
            })
            .into_iter()
            .map(|i| i as usize)
            .collect();

        let nlabels = header.nlabl as usize;
        let mut keep = Vec::new();
        for i in 0..nlabels {
            if !indices.contains(&(i + 1)) {
                keep.push(header.labels[i]);
            }
        }
        // Clear all labels and repopulate
        header.labels = [[0u8; 80]; 10];
        for (i, label) in keep.iter().enumerate() {
            header.labels[i] = *label;
        }
        header.nlabl = keep.len() as i32;
        eprintln!("{} labels remain after removal", keep.len());
        modified = true;
    }

    // Add title
    if let Some(ref title) = args.title {
        if !header.add_label(title) {
            eprintln!("WARNING: ALTERHEADER - all 10 label slots are full");
        } else {
            eprintln!("Label added: {}", title);
            modified = true;
        }
    }

    if !modified {
        eprintln!("No modifications requested");
        return;
    }

    // Write modified header back in-place
    let file = OpenOptions::new()
        .write(true)
        .open(&args.input)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: ALTERHEADER - opening file for writing: {}", e);
            process::exit(1);
        });
    let mut writer = BufWriter::new(file);
    writer.seek(SeekFrom::Start(0)).unwrap();
    header.write_le(&mut writer).unwrap_or_else(|e| {
        eprintln!("ERROR: ALTERHEADER - writing header: {}", e);
        process::exit(1);
    });
    writer.flush().unwrap();
    eprintln!("Header updated successfully");
}
