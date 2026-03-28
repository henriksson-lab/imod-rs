use std::process;

use clap::Parser;
use imod_mrc::{MrcReader, MrcWriter};

/// Subtract one image from another (A - B).
///
/// Reads corresponding sections from files A and B, computes the difference
/// (A - B), and writes the result to file C. If no output file is given,
/// only statistics are printed.
#[derive(Parser)]
#[command(name = "subimage", version, about)]
struct Args {
    /// File A (subtract from this).
    #[arg(short = 'a', long = "afile")]
    afile: String,

    /// File B (subtract this off).
    #[arg(short = 'b', long = "bfile")]
    bfile: String,

    /// Output difference file (optional; if omitted, statistics only).
    #[arg(short = 'o', long = "output")]
    output: Option<String>,

    /// Section list from file A (comma-separated, ranges OK; default: all).
    #[arg(long = "asections")]
    a_sections: Option<String>,

    /// Section list from file B (same count as A; default: same as A list).
    #[arg(long = "bsections")]
    b_sections: Option<String>,

    /// Mode for output (0=byte, 1=short, 2=float).
    #[arg(short = 'm', long = "mode")]
    mode: Option<i32>,

    /// Subtract mean difference so output has zero mean.
    #[arg(long = "zero")]
    zero_mean: bool,
}

fn parse_section_list(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.trim().parse().expect("bad section number");
            let end: usize = b.trim().parse().expect("bad section number");
            for v in start..=end {
                result.push(v);
            }
        } else {
            result.push(part.parse().expect("bad section number"));
        }
    }
    result
}

fn main() {
    let args = Args::parse();

    let mut reader_a = MrcReader::open(&args.afile).unwrap_or_else(|e| {
        eprintln!("ERROR: subimage - opening {}: {}", args.afile, e);
        process::exit(1);
    });

    let ha = reader_a.header().clone();
    let nx = ha.nx as usize;
    let ny = ha.ny as usize;
    let nz_a = ha.nz as usize;

    let a_sections = if let Some(ref s) = args.a_sections {
        parse_section_list(s)
    } else {
        (0..nz_a).collect()
    };

    let b_sections = if let Some(ref s) = args.b_sections {
        parse_section_list(s)
    } else {
        a_sections.clone()
    };

    if a_sections.len() != b_sections.len() {
        eprintln!("ERROR: subimage - section counts do not match");
        process::exit(1);
    }

    let mut reader_b = MrcReader::open(&args.bfile).unwrap_or_else(|e| {
        eprintln!("ERROR: subimage - opening {}: {}", args.bfile, e);
        process::exit(1);
    });

    let hb = reader_b.header().clone();
    if hb.nx as usize != nx || hb.ny as usize != ny {
        eprintln!("ERROR: subimage - image sizes do not match");
        process::exit(1);
    }

    let out_mode = args.mode.unwrap_or(ha.mode);
    let num_sections = a_sections.len();

    let mut writer = args.output.as_ref().map(|out_path| {
        let mut oh = ha.clone();
        oh.nz = num_sections as i32;
        oh.mz = num_sections as i32;
        oh.mode = out_mode;
        oh.add_label("SUBIMAGE: Subtract section B from section A.");
        MrcWriter::create(out_path, oh).unwrap_or_else(|e| {
            eprintln!("ERROR: subimage - creating output: {}", e);
            process::exit(1);
        })
    });

    println!(" Section      Min            Max            Mean           S.D.");

    let mut global_min = f32::MAX;
    let mut global_max = f32::MIN;
    let mut global_sum = 0.0_f64;
    let mut global_sumsq = 0.0_f64;

    for i in 0..num_sections {
        let sec_a = a_sections[i];
        let sec_b = b_sections[i];

        let slice_a = reader_a.read_slice_f32(sec_a).unwrap_or_else(|e| {
            eprintln!("ERROR: subimage - reading A section {}: {}", sec_a, e);
            process::exit(1);
        });

        let slice_b = reader_b.read_slice_f32(sec_b).unwrap_or_else(|e| {
            eprintln!("ERROR: subimage - reading B section {}: {}", sec_b, e);
            process::exit(1);
        });

        // Compute mean difference if zero-mean requested
        let diff_mean: f64 = if args.zero_mean {
            let sum_a: f64 = slice_a.iter().map(|&v| v as f64).sum();
            let sum_b: f64 = slice_b.iter().map(|&v| v as f64).sum();
            (sum_a - sum_b) / (nx * ny) as f64
        } else {
            0.0
        };

        // Compute difference
        let diff: Vec<f32> = slice_a
            .iter()
            .zip(slice_b.iter())
            .map(|(&a, &b)| a - b - diff_mean as f32)
            .collect();

        // Statistics
        let mut smin = f32::MAX;
        let mut smax = f32::MIN;
        let mut ssum = 0.0_f64;
        let mut ssumsq = 0.0_f64;
        for &v in &diff {
            smin = smin.min(v);
            smax = smax.max(v);
            ssum += v as f64;
            ssumsq += (v as f64) * (v as f64);
        }
        let n = (nx * ny) as f64;
        let smean = ssum / n;
        let sd = ((ssumsq - ssum * ssum / n) / (n - 1.0)).max(0.0).sqrt();

        global_min = global_min.min(smin);
        global_max = global_max.max(smax);
        global_sum += smean;
        global_sumsq += ssumsq;

        println!(
            "{:5}{:15.4}{:15.4}{:15.4}{:15.4}",
            sec_a, smin, smax, smean, sd
        );

        if let Some(ref mut w) = writer {
            w.write_slice_f32(&diff).unwrap_or_else(|e| {
                eprintln!("ERROR: subimage - writing section: {}", e);
                process::exit(1);
            });
        }
    }

    let global_mean = global_sum / num_sections as f64;
    let total_n = (nx * ny * num_sections) as f64;
    let global_sd = ((global_sumsq - global_sum * global_sum * (nx * ny) as f64 / total_n)
        / (total_n - 1.0))
        .max(0.0)
        .sqrt();

    if num_sections > 1 {
        println!(
            " all {:15.4}{:15.4}{:15.4}{:15.4}",
            global_min, global_max, global_mean, global_sd
        );
    }

    if let Some(w) = writer {
        w.finish(global_min, global_max, global_mean as f32)
            .unwrap();
    }
}
