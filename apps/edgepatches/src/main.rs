use clap::Parser;
use imod_mrc::MrcReader;
use std::fs;
use std::io::{BufRead, BufReader, Write};

/// Compute edge patches for blending adjacent volume pieces in a supermontage.
///
/// Given an info file describing pieces and their overlaps, this program
/// determines overlap regions between adjacent volumes and writes patch
/// correlation parameters for use with corrsearch3d and blending tools.
#[derive(Parser)]
#[command(name = "edgepatches", about = "Find edge patches for blending")]
struct Args {
    /// Supermontage info file
    #[arg(short = 'f', long)]
    info: String,

    /// Run all edges
    #[arg(long)]
    all: bool,

    /// Patch size in X, Y, Z
    #[arg(long, num_args = 3, default_values_t = [100, 100, 50])]
    size: Vec<i32>,

    /// Patch spacing intervals: short-axis, long-axis, Z
    #[arg(long, num_args = 3, default_values_t = [80, 120, 50])]
    intervals: Vec<i32>,

    /// Border to exclude in XY and Z
    #[arg(long, num_args = 2, default_values_t = [50, 10])]
    borders: Vec<i32>,

    /// Fraction of long dimension to use for shift correlation
    #[arg(long, default_value_t = 0.5)]
    long_frac: f32,

    /// Force this number of patches in Z
    #[arg(long, default_value_t = 0)]
    force_z: i32,

    /// Kernel sigma for corrsearch3d
    #[arg(long, default_value_t = 0.0)]
    kernel: f32,

    /// Skip edges where patches already exist
    #[arg(long)]
    skip_done: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// A piece in the supermontage
#[derive(Debug, Clone)]
struct Piece {
    file: String,
    index: [i32; 3],  // position in the grid (x, y, z)
    size: [i32; 3],   // volume dimensions
    z_limits: Option<[i32; 2]>,
}

/// An edge between two adjacent pieces
#[derive(Debug, Clone)]
struct Edge {
    name: String,
    x_or_y: char,          // 'X' or 'Y' - which axis the edge is along
    lower_idx: [i32; 3],   // grid index of lower piece
    shift: [f32; 3],       // coordinate shift between pieces
    patch_file: Option<String>,
}

/// Parse the supermontage info file.
/// This is a simplified parser for the key-value format used by IMOD supermontage info files.
fn parse_info_file(path: &str) -> (Vec<Piece>, Vec<Edge>) {
    let file = fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("Error opening info file {}: {}", path, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);

    let mut pieces: Vec<Piece> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    let mut in_piece = false;
    let mut in_edge = false;
    let mut cur_piece = Piece {
        file: String::new(),
        index: [0; 3],
        size: [0; 3],
        z_limits: None,
    };
    let mut cur_edge = Edge {
        name: String::new(),
        x_or_y: 'X',
        lower_idx: [0; 3],
        shift: [0.0; 3],
        patch_file: None,
    };

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line == "[piece]" || line == "[Piece]" {
            if in_piece {
                pieces.push(cur_piece.clone());
            }
            if in_edge {
                edges.push(cur_edge.clone());
                in_edge = false;
            }
            in_piece = true;
            cur_piece = Piece {
                file: String::new(),
                index: [0; 3],
                size: [0; 3],
                z_limits: None,
            };
            continue;
        }

        if line == "[edge]" || line == "[Edge]" {
            if in_piece {
                pieces.push(cur_piece.clone());
                in_piece = false;
            }
            if in_edge {
                edges.push(cur_edge.clone());
            }
            in_edge = true;
            cur_edge = Edge {
                name: String::new(),
                x_or_y: 'X',
                lower_idx: [0; 3],
                shift: [0.0; 3],
                patch_file: None,
            };
            continue;
        }

        // Parse key = value pairs
        if let Some((key, val)) = line.split_once('=') {
            let key = key.trim().to_lowercase();
            let val = val.trim();

            if in_piece {
                match key.as_str() {
                    "file" => cur_piece.file = val.to_string(),
                    "index" | "position" => {
                        let nums: Vec<i32> = val.split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            cur_piece.index = [nums[0], nums[1], nums[2]];
                        }
                    }
                    "size" => {
                        let nums: Vec<i32> = val.split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            cur_piece.size = [nums[0], nums[1], nums[2]];
                        }
                    }
                    "zlimits" | "zlimit" => {
                        let nums: Vec<i32> = val.split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 2 {
                            cur_piece.z_limits = Some([nums[0], nums[1]]);
                        }
                    }
                    _ => {}
                }
            }

            if in_edge {
                match key.as_str() {
                    "name" => cur_edge.name = val.to_string(),
                    "xory" | "axis" => {
                        cur_edge.x_or_y = val.chars().next().unwrap_or('X').to_ascii_uppercase();
                    }
                    "lower" => {
                        let nums: Vec<i32> = val.split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            cur_edge.lower_idx = [nums[0], nums[1], nums[2]];
                        }
                    }
                    "shift" | "corrshift" => {
                        let nums: Vec<f32> = val.split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if nums.len() >= 3 {
                            cur_edge.shift = [nums[0], nums[1], nums[2]];
                        }
                    }
                    "patchfile" | "patch" => {
                        cur_edge.patch_file = Some(val.to_string());
                    }
                    _ => {}
                }
            }
        }
    }

    // Push last item
    if in_piece {
        pieces.push(cur_piece);
    }
    if in_edge {
        edges.push(cur_edge);
    }

    (pieces, edges)
}

/// Find a piece by its grid index
fn find_piece<'a>(pieces: &'a [Piece], idx: &[i32; 3]) -> Option<&'a Piece> {
    pieces.iter().find(|p| p.index == *idx)
}

/// Compute patch correlation parameters for an edge
fn compute_patch_params(
    lower: &Piece,
    upper: &Piece,
    edge: &Edge,
    args: &Args,
) -> Vec<String> {
    let patch_size = [args.size[0], args.size[1], args.size[2]];
    let border_xy = args.borders[0].max(0);
    let border_z = args.borders[1];

    let mut sizes = patch_size;
    let mut mins = [0i32; 3];
    let mut maxes = [0i32; 3];
    let mut numpat = [0i32; 3];

    // Z limits
    mins[2] = (border_z + 1).max(1);
    maxes[2] = lower.size[2] + 1 - mins[2];
    if let Some(zlim) = lower.z_limits {
        mins[2] = zlim[0].max(1);
        maxes[2] = zlim[1].min(lower.size[2]);
    }

    if maxes[2] - mins[2] < sizes[2] {
        sizes[2] = maxes[2] - mins[2];
    }

    // X and Y limits
    for i in 0..2 {
        mins[i] = border_xy.max(border_xy - edge.shift[i] as i32) + 1;
        maxes[i] = (lower.size[i] - border_xy)
            .min(upper.size[i] - border_xy - edge.shift[i] as i32);
    }

    // Number of patches
    let intervals = if edge.x_or_y == 'Y' {
        [args.intervals[1], args.intervals[0], args.intervals[2]]
    } else {
        [args.intervals[0], args.intervals[1], args.intervals[2]]
    };

    for i in 0..3 {
        let range = maxes[i] - mins[i] - sizes[i];
        numpat[i] = if range <= 0 {
            1
        } else {
            ((range as f32 / intervals[i] as f32).round() as i32).max(1) + 1
        };
    }
    if args.force_z > 0 {
        numpat[2] = args.force_z;
    }

    let patchname = format!("{}.patch", edge.name);

    let mut params = vec![
        format!("ReferenceFile {}", lower.file),
        format!("FileToAlign {}", upper.file),
        format!("OutputFile {}", patchname),
        format!(
            "VolumeShiftXYZ {:.6} {:.6} {:.6}",
            edge.shift[0], edge.shift[1], edge.shift[2]
        ),
        format!("PatchSizeXYZ {} {} {}", sizes[0], sizes[1], sizes[2]),
        format!("XMinAndMax {} {}", mins[0], maxes[0]),
        format!("YMinAndMax {} {}", mins[1], maxes[1]),
        format!("ZMinAndMax {} {}", mins[2], maxes[2]),
        format!("BSourceBorderXLoHi {} {}", border_xy, border_xy),
        format!("BSourceBorderYZLoHi {} {}", border_xy, border_xy),
        format!(
            "NumberOfPatchesXYZ {} {} {}",
            numpat[0], numpat[1], numpat[2]
        ),
    ];

    if args.kernel > 0.0 {
        params.push(format!("KernelSigma {:.2}", args.kernel));
    }

    params
}

fn main() {
    let args = Args::parse();

    let (pieces, edges) = parse_info_file(&args.info);

    if pieces.is_empty() {
        eprintln!("Error: no pieces found in info file {}", args.info);
        std::process::exit(1);
    }

    eprintln!(
        "Read {} pieces and {} edges from {}",
        pieces.len(),
        edges.len(),
        args.info
    );

    // Fill in piece sizes from MRC files if size is [0,0,0]
    let pieces: Vec<Piece> = pieces
        .into_iter()
        .map(|mut p| {
            if p.size == [0, 0, 0] && !p.file.is_empty() {
                if let Ok(reader) = MrcReader::open(&p.file) {
                    let h = reader.header();
                    p.size = [h.nx, h.ny, h.nz];
                }
            }
            p
        })
        .collect();

    let mut processed = 0;

    for edge in &edges {
        if !args.all {
            continue;
        }

        if args.skip_done && edge.patch_file.is_some() {
            if args.verbose {
                eprintln!("Skipping edge {} (patches exist)", edge.name);
            }
            continue;
        }

        // Find lower and upper pieces
        let lower_idx = edge.lower_idx;
        let upper_idx = if edge.x_or_y == 'X' {
            [lower_idx[0] + 1, lower_idx[1], lower_idx[2]]
        } else {
            [lower_idx[0], lower_idx[1] + 1, lower_idx[2]]
        };

        let lower = match find_piece(&pieces, &lower_idx) {
            Some(p) => p,
            None => {
                eprintln!(
                    "Warning: lower piece {:?} not found for edge {}",
                    lower_idx, edge.name
                );
                continue;
            }
        };
        let upper = match find_piece(&pieces, &upper_idx) {
            Some(p) => p,
            None => {
                eprintln!(
                    "Warning: upper piece {:?} not found for edge {}",
                    upper_idx, edge.name
                );
                continue;
            }
        };

        eprintln!("Computing patch correlations for edge {}", edge.name);

        let params = compute_patch_params(lower, upper, edge, &args);

        let numpat_line = params
            .iter()
            .find(|l| l.starts_with("NumberOfPatchesXYZ"))
            .unwrap();
        eprintln!("  {}", numpat_line);

        // Write corrsearch3d parameter file
        let param_file = format!("{}.corrsearch3d", edge.name);
        let mut f = fs::File::create(&param_file).unwrap_or_else(|e| {
            eprintln!("Error creating {}: {}", param_file, e);
            std::process::exit(1);
        });
        for line in &params {
            writeln!(f, "{}", line).unwrap();
        }

        if args.verbose {
            for line in &params {
                eprintln!("  {}", line);
            }
        }

        // Also print the corrsearch3d command to stdout
        println!("corrsearch3d -StandardInput < {}", param_file);

        processed += 1;
    }

    if processed == 0 && args.all {
        eprintln!("No edges were processed. Check info file format.");
    } else {
        eprintln!("{} edges processed", processed);
    }
}
