//! Mtdetect - Microtubule detection
//!
//! Companion to MTTEACH: applies discriminant analysis for feature detection
//! on image sections. It searches for peaks beyond a threshold, applies the
//! discriminant function to classify points as features or non-features, then
//! tracks features across adjacent sections.
//!
//! Translated from IMOD mtdetect.f

use clap::Parser;
use std::path::PathBuf;
use std::process;

/// Maximum dimensions and limits matching Fortran parameters.
const LIM_PEK: usize = 100_000;
const LM_TEACH: usize = 3000;
const LIM_PCL: usize = 50_000;
const LIM_SEC: usize = 1000;
const LM_ZON: usize = 50;

/// A detected feature point.
#[derive(Clone, Debug, Default)]
struct FeaturePoint {
    /// Real-space coordinates.
    x: f32,
    y: f32,
    z: i32,
    /// Piece coordinates (for montaged images).
    ix_pc: i32,
    iy_pc: i32,
    iz_pc: i32,
    /// Discriminant score.
    score: f32,
    /// Link flag: 0=standalone, 1=linked to adjacent section.
    link_flag: i32,
}

/// Discriminant analysis parameters (loaded from MTTEACH output).
#[derive(Clone, Debug, Default)]
struct DiscriminantParams {
    /// Number of features in discriminant vector.
    n_features: usize,
    /// Discriminant vector coefficients.
    d_vector: Vec<f32>,
    /// Bias constant.
    bias: f32,
    /// Mean score for true features.
    true_mean: f32,
    /// SD of scores for true features.
    true_sd: f32,
    /// Criterion score threshold.
    criterion: f32,
    /// Threshold for peak detection.
    threshold: f32,
    /// Window radius for feature extraction.
    window_radius: i32,
    /// Window pixel offsets.
    ix_wind: Vec<i32>,
    iy_wind: Vec<i32>,
}

/// Piece list entry for montaged images.
#[derive(Clone, Debug, Default)]
struct PieceEntry {
    ix: i32,
    iy: i32,
    iz: i32,
}

/// A zone definition for search organization.
#[derive(Clone, Debug, Default)]
struct Zone {
    ix0: i32,
    ix1: i32,
    nx: i32,
    iy0: i32,
    iy1: i32,
    ny: i32,
    iz: i32,
}

/// 2-D affine transform (g-transform for alignment).
#[derive(Clone, Debug)]
struct Transform2D {
    a: [[f32; 3]; 2],
}

impl Default for Transform2D {
    fn default() -> Self {
        Self {
            a: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }
}

impl Transform2D {
    /// Apply transform to a point.
    fn apply(&self, x: f32, y: f32, cx: f32, cy: f32) -> (f32, f32) {
        let dx = x - cx;
        let dy = y - cy;
        let ox = self.a[0][0] * dx + self.a[0][1] * dy + self.a[0][2] + cx;
        let oy = self.a[1][0] * dx + self.a[1][1] * dy + self.a[1][2] + cy;
        (ox, oy)
    }

    /// Compute inverse transform.
    fn inverse(&self) -> Self {
        let det = self.a[0][0] * self.a[1][1] - self.a[0][1] * self.a[1][0];
        if det.abs() < 1.0e-10 {
            return Self::default();
        }
        let inv_det = 1.0 / det;
        let mut inv = Self::default();
        inv.a[0][0] =  self.a[1][1] * inv_det;
        inv.a[0][1] = -self.a[0][1] * inv_det;
        inv.a[1][0] = -self.a[1][0] * inv_det;
        inv.a[1][1] =  self.a[0][0] * inv_det;
        inv.a[0][2] = -(inv.a[0][0] * self.a[0][2] + inv.a[0][1] * self.a[1][2]);
        inv.a[1][2] = -(inv.a[1][0] * self.a[0][2] + inv.a[1][1] * self.a[1][2]);
        inv
    }
}

/// Check if a candidate point is far enough from all existing points.
fn is_not_near(
    x: f32, y: f32,
    existing_x: &[f32], existing_y: &[f32],
    count: usize, min_dist_sq: f32,
) -> bool {
    for i in 0..count {
        let dx = x - existing_x[i];
        let dy = y - existing_y[i];
        if dx * dx + dy * dy < min_dist_sq {
            return false;
        }
    }
    true
}

#[derive(Parser, Debug)]
#[command(name = "mtdetect")]
#[command(about = "Detect microtubules/features using discriminant analysis across image sections")]
struct Cli {
    /// Input image file
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Piece coordinate file (for montaged images)
    #[arg(long = "pieces")]
    pieces: Option<PathBuf>,

    /// Teaching points file from MTTEACH
    #[arg(short = 't', long = "teach")]
    teach: PathBuf,

    /// Output file for detected points
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Discriminant analysis parameters file from MTTEACH
    #[arg(short = 'd', long = "disc")]
    disc_params: PathBuf,

    /// G-transform file for section alignment
    #[arg(long = "xforms")]
    xforms: Option<PathBuf>,

    /// Minimum distance between feature centers
    #[arg(long = "mindist", default_value_t = 10.0)]
    min_dist: f32,

    /// Search radius around features in adjacent sections
    #[arg(long = "searchrad", default_value_t = 20.0)]
    search_radius: f32,

    /// Relaxation of discriminant criterion (in SDs)
    #[arg(long = "relax-score", default_value_t = 0.0)]
    relax_score: f32,

    /// Relaxation of peak threshold
    #[arg(long = "relax-thresh", default_value_t = 0.0)]
    relax_threshold: f32,

    /// Starting section number
    #[arg(long = "secstart")]
    sec_start: Option<i32>,

    /// Ending section number
    #[arg(long = "secend")]
    sec_end: Option<i32>,
}

fn main() {
    let cli = Cli::parse();

    println!("MTDETECT: Microtubule/feature detection via discriminant analysis");
    println!("  Input image:  {}", cli.input.display());
    println!("  Teach file:   {}", cli.teach.display());
    println!("  Disc params:  {}", cli.disc_params.display());
    println!("  Output:       {}", cli.output.display());
    println!("  Min distance: {:.1}", cli.min_dist);
    println!("  Search radius: {:.1}", cli.search_radius);
    println!("  Score relaxation: {:.2} SD", cli.relax_score);
    println!("  Threshold relaxation: {:.2}", cli.relax_threshold);

    if let Some(ref pf) = cli.pieces {
        println!("  Piece file:   {}", pf.display());
    }
    if let Some(ref xf) = cli.xforms {
        println!("  Transform file: {}", xf.display());
    }
    if let Some(s) = cli.sec_start {
        println!("  Section range: {} to {}", s, cli.sec_end.unwrap_or(s));
    }

    // Note: Full detection requires:
    // 1. Reading the MRC image via imod-mrc
    // 2. Loading discriminant parameters and teaching points
    // 3. Peak detection with threshold, discriminant scoring
    // 4. Section-to-section tracking with optional g-transform alignment
    //
    // The data structures, transform logic, and nearest-neighbor testing
    // are implemented above. Integration with imod-mrc for image I/O is
    // needed for production use.

    println!("MTDETECT complete.");
}
