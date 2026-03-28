use clap::Parser;
use imod_model::{write_model, ImodModel, ImodObject};
use imod_mrc::MrcReader;

/// Find regular holes in carbon film for cryo-EM grid imaging.
///
/// Uses cross-correlation with circular templates at varying filter
/// parameters and thresholds to find holes in carbon film. Supports
/// single images and montages, with optional boundary models.
#[derive(Parser)]
#[command(name = "imodholefinder", about = "Find holes in carbon film on EM grids")]
struct Args {
    /// Input image file (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Output model file
    #[arg(short = 'o', long)]
    output: String,

    /// Boundary model file (optional, object 1 contours define search area)
    #[arg(short = 'b', long)]
    boundary: Option<String>,

    /// Summary output file
    #[arg(long)]
    summary: Option<String>,

    /// Diameter of holes (in pixels or units matching pixel size)
    #[arg(short = 'd', long)]
    diameter: f32,

    /// Spacing of holes (in pixels or units matching pixel size)
    #[arg(short = 's', long)]
    spacing: f32,

    /// Maximum error (fraction of diameter used if not specified)
    #[arg(short = 'e', long)]
    max_error: Option<f32>,

    /// Threshold percentiles (comma-separated, e.g., "2.0,3.2,4.4")
    #[arg(short = 't', long, value_delimiter = ',')]
    thresholds: Option<Vec<f32>>,

    /// Filter sigma or iterations (comma-separated)
    #[arg(short = 'f', long, value_delimiter = ',')]
    filter_sigma: Option<Vec<f32>>,

    /// Circle thicknesses (comma-separated)
    #[arg(long, value_delimiter = ',')]
    thicknesses: Option<Vec<f32>>,

    /// Number of circles per scan (comma-separated)
    #[arg(long, value_delimiter = ',')]
    num_circles: Option<Vec<i32>>,

    /// Circle step sizes (comma-separated)
    #[arg(long, value_delimiter = ',')]
    step_size: Option<Vec<f32>>,

    /// Sections to process (comma-separated list or ranges)
    #[arg(long)]
    sections: Option<String>,

    /// Diameter for intensity statistics (0 to skip)
    #[arg(long)]
    intensity_diam: Option<f32>,

    /// Raw montage file
    #[arg(long)]
    montage: Option<String>,

    /// Piece list file for montage
    #[arg(long)]
    piece_list: Option<String>,

    /// Aligned piece list file
    #[arg(long)]
    aligned_list: Option<String>,

    /// Full image binning factor
    #[arg(long)]
    binned: Option<i32>,

    /// Retain duplicate positions from overlapping pieces
    #[arg(long)]
    retain_duplicates: bool,

    /// Show piece-specific objects in output model
    #[arg(long)]
    show_pieces: bool,

    /// Verbose output level (0-2)
    #[arg(short = 'v', long, default_value = "0")]
    verbose: i32,
}

fn main() {
    let args = Args::parse();

    // Open input image
    let reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error opening input image file {}: {}", args.input, e);
        std::process::exit(1);
    });

    let header = reader.header();
    let nx = header.nx as i32;
    let ny = header.ny as i32;
    let nz = header.nz;

    let pixel_size = if header.xlen > 0.0 && header.mx > 0 {
        header.xlen / header.mx as f32
    } else {
        1.0
    };

    // Scale diameter and spacing to pixels if given in physical units
    let scale = if pixel_size > 0.0 { 1.0 / pixel_size } else { 1.0 };
    let diameter = args.diameter * scale;
    let spacing = args.spacing * scale;

    let fractional_err_max = 0.05_f32;
    let error_max = args
        .max_error
        .map(|e| e * scale)
        .unwrap_or(fractional_err_max * diameter);

    // Default parameters
    let _thresholds = args.thresholds.unwrap_or_else(|| vec![2.0, 3.2, 4.4]);
    let _sigmas = args.filter_sigma.unwrap_or_else(|| vec![1.5, 2.0, 3.0]);

    let target_diam_pix = 50.0_f32;
    let reduction = (diameter / target_diam_pix).max(1.0);
    let _diam_reduced = diameter / reduction;
    let _spacing_reduced = spacing / reduction;

    if args.verbose > 0 {
        println!("Input image: {} x {} x {}", nx, ny, nz);
        println!("Pixel size: {:.4}", pixel_size);
        println!("Hole diameter: {:.1} pixels", diameter);
        println!("Hole spacing: {:.1} pixels", spacing);
        println!("Reduction factor: {:.2}", reduction);
        println!("Error max: {:.2}", error_max);
    }

    // Determine sections to process
    let sections: Vec<i32> = if let Some(ref sec_str) = args.sections {
        parse_section_list(sec_str, nz)
    } else {
        (0..nz).collect()
    };

    // Create output model with 3 objects: found holes, missing holes, close holes
    let mut out_model = ImodModel::default();
    let mut obj_found = ImodObject::default();
    obj_found.name = "Found holes".to_string();
    obj_found.red = 1.0;
    obj_found.green = 0.0;
    obj_found.blue = 1.0;

    let mut obj_missing = ImodObject::default();
    obj_missing.name = "Missing holes".to_string();
    obj_missing.red = 0.0;
    obj_missing.green = 1.0;
    obj_missing.blue = 0.0;

    let mut obj_close = ImodObject::default();
    obj_close.name = "Close to edge".to_string();
    obj_close.red = 1.0;
    obj_close.green = 1.0;
    obj_close.blue = 0.0;

    // TODO: Implement actual hole-finding algorithm using cross-correlation
    // with circular templates, threshold scanning, and grid analysis.
    // For now, write the empty model structure.

    eprintln!(
        "Note: hole-finding algorithm not yet implemented. Writing empty model with {} sections.",
        sections.len()
    );

    out_model.xmax = nx;
    out_model.ymax = ny;
    out_model.objects.push(obj_found);
    out_model.objects.push(obj_missing);
    out_model.objects.push(obj_close);

    write_model(&args.output, &out_model).unwrap_or_else(|e| {
        eprintln!("Error writing output model {}: {}", args.output, e);
        std::process::exit(1);
    });

    if args.verbose > 0 {
        println!("Wrote output model to {}", args.output);
    }
}

fn parse_section_list(s: &str, nz: i32) -> Vec<i32> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (a.trim().parse::<i32>(), b.trim().parse::<i32>()) {
                for i in start..=end {
                    if i >= 0 && i < nz {
                        result.push(i);
                    }
                }
            }
        } else if let Ok(v) = part.parse::<i32>() {
            if v >= 0 && v < nz {
                result.push(v);
            }
        }
    }
    result
}
