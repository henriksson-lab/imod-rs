use clap::Parser;
use imod_mrc::MrcReader;

/// Find optimal black/white contrast settings for byte conversion.
/// Computes a histogram of pixel values and determines contrast levels
/// that truncate a specified small number of pixels.
#[derive(Parser)]
#[command(name = "findcontrast", about = "Find optimal contrast settings for byte conversion")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'i', long)]
    input: String,

    /// First and last slice (1-based) to include
    #[arg(short = 's', long, num_args = 2, value_delimiter = ',')]
    slices: Option<Vec<i32>>,

    /// X min and max to include in analysis
    #[arg(long, num_args = 2, value_delimiter = ',')]
    xminmax: Option<Vec<i32>>,

    /// Y min and max to include in analysis
    #[arg(long, num_args = 2, value_delimiter = ',')]
    yminmax: Option<Vec<i32>>,

    /// Treat Y/Z as flipped
    #[arg(long, default_value = "false")]
    flipyz: bool,

    /// Maximum pixels to truncate at black and white (comma-separated pair)
    #[arg(short = 't', long, num_args = 2, value_delimiter = ',')]
    truncate: Option<Vec<i32>>,
}

const LIM_DEN: i64 = 1_000_000;

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: FINDCONTRAST - opening input: {e}");
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;
    let mode = h.mode;
    let dmin = h.amin;
    let dmax = h.amax;

    // Histogram scaling for non-integer modes
    let hist_scale: f64 = if mode != 1 && mode != 6 && mode != 0 {
        (LIM_DEN as f64 / 5.0) / (dmin.abs().max(dmax.abs()).max(1.0e-10) as f64)
    } else {
        1.0
    };

    let flipped = args.flipyz;

    let iz_lim = if flipped { ny } else { nz };
    let iy_lim = if flipped { nz } else { ny };

    // Default limits
    let ix_low_default = (nx / 10) as i32;
    let ix_high_default = nx as i32 - 1 - ix_low_default;
    let iy_low_default = (iy_lim / 10) as i32;
    let iy_high_default = iy_lim as i32 - 1 - iy_low_default;

    let mut ix_low = ix_low_default;
    let mut ix_high = ix_high_default;
    let mut iy_low = iy_low_default;
    let mut iy_high = iy_high_default;
    let mut iz_low = 1i32;
    let mut iz_high = iz_lim as i32;

    if let Some(ref s) = args.slices {
        iz_low = s[0];
        iz_high = s[1];
    }
    if let Some(ref x) = args.xminmax {
        ix_low = x[0];
        ix_high = x[1];
    }
    if let Some(ref y) = args.yminmax {
        iy_low = y[0];
        iy_high = y[1];
    }

    if iz_low <= 0 || iz_high > iz_lim as i32 || iz_low > iz_high {
        eprintln!("ERROR: FINDCONTRAST - Slice numbers outside range of image file");
        std::process::exit(1);
    }
    // Convert to 0-based
    iz_low -= 1;
    iz_high -= 1;

    if ix_low < 0
        || ix_high >= nx as i32
        || ix_low >= ix_high
        || iy_low < 0
        || iy_high >= iy_lim as i32
        || iy_low > iy_high
    {
        eprintln!("ERROR: FINDCONTRAST - X or Y values outside range of volume");
        std::process::exit(1);
    }

    // Auto truncation counts
    let area_fac = (nx as f64 * iy_lim as f64 * 1.0e-6).max(1.0);
    let default_trunc = (area_fac * (iz_high + 1 - iz_low) as f64) as i64;
    let mut num_trunc_lo = default_trunc;
    let mut num_trunc_hi = default_trunc;

    if let Some(ref t) = args.truncate {
        num_trunc_lo = t[0] as i64;
        num_trunc_hi = t[1] as i64;
    }

    // Flip coordinates if needed
    let (final_ix_low, final_ix_high, final_iy_low, final_iy_high, final_iz_low, final_iz_high) =
        if flipped {
            let iy_lo_new = (ny as i32) - 1 - iz_high;
            let iy_hi_new = (ny as i32) - 1 - iz_low;
            (ix_low, ix_high, iy_lo_new, iy_hi_new, iy_low, iy_high)
        } else {
            (ix_low, ix_high, iy_low, iy_high, iz_low, iz_high)
        };

    eprintln!(
        "Analyzing X:{:6}{:6}  Y:{:6}{:6}  Z:{:6}{:6}",
        final_ix_low, final_ix_high, final_iy_low, final_iy_high, final_iz_low, final_iz_high
    );

    // Build histogram using a HashMap for sparse storage
    let hist_size = (2 * LIM_DEN + 1) as usize;
    let mut ihist = vec![0i64; hist_size];
    let offset = LIM_DEN;
    let mut ival_min = LIM_DEN;
    let mut ival_max = -LIM_DEN;

    let nx_tot = (final_ix_high + 1 - final_ix_low) as usize;

    for iz in final_iz_low..=final_iz_high {
        let section = reader.read_slice_f32(iz as usize).unwrap_or_else(|e| {
            eprintln!("ERROR: FINDCONTRAST - reading section {iz}: {e}");
            std::process::exit(1);
        });

        for iy in final_iy_low..=final_iy_high {
            for ix in final_ix_low..final_ix_low + nx_tot as i32 {
                let pixel = section[iy as usize * nx + ix as usize];
                let mut ival = (hist_scale * pixel as f64).round() as i64;
                ival = ival.max(-LIM_DEN).min(LIM_DEN);
                ihist[(ival + offset) as usize] += 1;
                ival_min = ival_min.min(ival);
                ival_max = ival_max.max(ival);
            }
        }
    }

    // Find low contrast level
    let mut num_trunc = 0i64;
    let mut ind_low = ival_min;
    while num_trunc <= num_trunc_lo && ind_low < ival_max {
        num_trunc += ihist[(ind_low + offset) as usize];
        ind_low += 1;
    }

    // Find high contrast level
    num_trunc = 0;
    let mut ind_hi = ival_max;
    while num_trunc <= num_trunc_hi && ind_hi > ival_min {
        num_trunc += ihist[(ind_hi + offset) as usize];
        ind_hi -= 1;
    }

    let real_low = ind_low as f64 / hist_scale;
    let real_high = ind_hi as f64 / hist_scale;
    let range = dmax as f64 - dmin as f64;
    let icon_low = (255.0 * (real_low - dmin as f64) / range) as i32;
    let icon_high = (255.0 * (real_high - dmin as f64) / range + 0.99) as i32;

    if icon_low < 0 || icon_high > 255 {
        eprintln!(
            "ERROR: FINDCONTRAST - The file minimum or maximum is too far off to allow \
             contrast scaling; use Alterheader with mmm option to fix min/max"
        );
        std::process::exit(1);
    }

    println!(
        "Min and max densities in the analyzed volume are {:13.5} and {:13.5}",
        ival_min as f64 / hist_scale,
        ival_max as f64 / hist_scale,
    );
    println!(
        "Min and max densities with truncation are {:13.5} and {:13.5}",
        real_low, real_high,
    );
    println!(
        "Implied black and white contrast levels are {:4} and {:4}",
        icon_low, icon_high,
    );
}
