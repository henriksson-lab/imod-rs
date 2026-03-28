//! numericdiff - Compare numerical output between two text files.
//!
//! Reads two text files, identifies sections of numeric data, and reports
//! the maximum differences for each numeric section. Can flag sections where
//! differences exceed specified limits.
//!
//! Translated from IMOD's numericdiff.f

use clap::Parser;
use std::io::{BufRead, BufReader};
use std::process;

#[derive(Parser)]
#[command(name = "numericdiff", about = "Compare numerical output between two text files")]
struct Args {
    /// First input file
    #[arg(short = 'a', long)]
    ainput: String,

    /// Second input file (default: first file with ~ appended)
    #[arg(short = 'b', long)]
    binput: Option<String>,

    /// Maximum allowed differences per section (repeatable; one set per numeric section)
    #[arg(short = 'm', long, num_args = 1.., action = clap::ArgAction::Append)]
    max: Vec<f64>,

    /// Use general (scientific) format for output
    #[arg(short = 'g', long, default_value_t = false)]
    general: bool,

    /// Strip lines containing this string to numeric-only
    #[arg(short = 's', long)]
    strip: Option<String>,

    /// Print lines with big differences
    #[arg(long, default_value_t = false)]
    big: bool,
}

fn is_numeric_line(line: &str, strip_str: &Option<String>) -> (bool, String) {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return (false, String::new());
    }

    // If strip string matches, convert to numeric-only
    if let Some(ss) = strip_str {
        if !ss.is_empty() && trimmed.contains(ss.as_str()) {
            let cleaned: String = trimmed
                .chars()
                .map(|c| {
                    if c.is_ascii_digit()
                        || c == '.'
                        || c == '-'
                        || c == '+'
                        || c == ' '
                        || c == '\t'
                        || c == 'e'
                        || c == 'E'
                    {
                        // Keep e/E only if it looks like scientific notation
                        c
                    } else if c == ',' {
                        ' '
                    } else {
                        ' '
                    }
                })
                .collect();
            let cleaned = cleaned.trim().to_string();
            if cleaned.is_empty() {
                return (false, String::new());
            }
            return (true, cleaned);
        }
    }

    // Check if line is all numeric characters
    for c in trimmed.chars() {
        if c.is_ascii_digit()
            || c == '.'
            || c == '-'
            || c == '+'
            || c == ' '
            || c == '\t'
            || c == ','
        {
            continue;
        }
        if c == 'e' || c == 'E' {
            continue; // Allow scientific notation
        }
        return (false, String::new());
    }

    let cleaned = trimmed.replace(',', " ");
    (true, cleaned)
}

fn parse_values(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect()
}

fn output_diffs(
    diff_max: &[f64],
    max_comp: usize,
    diff_limits: &[f64],
    section: usize,
    num_error: &mut usize,
    general: bool,
) {
    print!("Section {:3} differences:", section);
    if general {
        for k in 0..max_comp {
            print!(" {:13.5e}", diff_max[k]);
        }
    } else {
        for k in 0..max_comp {
            if k == 0 {
                print!("{:9.4}", diff_max[k]);
            } else {
                print!("{:10.4}", diff_max[k]);
            }
        }
    }
    println!();

    let num_above: usize = diff_limits
        .iter()
        .enumerate()
        .take(max_comp)
        .filter(|(i, lim)| diff_max.get(*i).map_or(false, |d| d > lim))
        .count();

    if num_above > 0 {
        println!(" {} differences above limit", num_above);
        *num_error += 1;
    }
}

fn main() {
    let args = Args::parse();

    let bfile = args
        .binput
        .unwrap_or_else(|| format!("{}~", args.ainput));

    let file_a = std::fs::File::open(&args.ainput).unwrap_or_else(|e| {
        eprintln!("ERROR: numericdiff - opening {}: {}", args.ainput, e);
        process::exit(1);
    });
    let file_b = std::fs::File::open(&bfile).unwrap_or_else(|e| {
        eprintln!("ERROR: numericdiff - opening {}: {}", bfile, e);
        process::exit(1);
    });

    let mut reader_a = BufReader::new(file_a);
    let mut reader_b = BufReader::new(file_b);

    let diff_limits: Vec<f64> = args.max.clone();
    let num_limit_sets = diff_limits.len();

    let mut in_section = false;
    let mut numeric_sect = 0usize;
    let mut num_error = 0usize;
    let mut diff_max = vec![0.0f64; 1000];
    let mut max_comp = 0usize;
    let mut did_big = false;

    let mut line_a = String::new();
    let mut line_b = String::new();

    loop {
        line_a.clear();
        line_b.clear();
        let ra = reader_a.read_line(&mut line_a).unwrap_or(0);
        let rb = reader_b.read_line(&mut line_b).unwrap_or(0);
        if ra == 0 || rb == 0 {
            break;
        }

        let (a_numeric, strip_a) = is_numeric_line(&line_a, &args.strip);
        let (b_numeric, strip_b) = is_numeric_line(&line_b, &args.strip);

        if in_section && !(a_numeric && b_numeric) {
            // End of numeric section
            let section_limits = if numeric_sect <= num_limit_sets {
                &diff_limits[..num_limit_sets.min(diff_max.len())]
            } else {
                &[]
            };
            output_diffs(
                &diff_max,
                max_comp,
                section_limits,
                numeric_sect,
                &mut num_error,
                args.general,
            );
            in_section = false;
            // Skip remaining numeric lines
            while {
                let (an, _) = is_numeric_line(&line_a, &args.strip);
                an
            } {
                line_a.clear();
                if reader_a.read_line(&mut line_a).unwrap_or(0) == 0 {
                    break;
                }
            }
            while {
                let (bn, _) = is_numeric_line(&line_b, &args.strip);
                bn
            } {
                line_b.clear();
                if reader_b.read_line(&mut line_b).unwrap_or(0) == 0 {
                    break;
                }
            }
        } else if !in_section && (a_numeric || b_numeric) {
            // Start new numeric section
            numeric_sect += 1;
            diff_max = vec![0.0f64; 1000];
            max_comp = 0;
            did_big = false;

            // Skip non-numeric lines to sync
            while !a_numeric {
                line_a.clear();
                if reader_a.read_line(&mut line_a).unwrap_or(0) == 0 {
                    break;
                }
                let (an, _) = is_numeric_line(&line_a, &args.strip);
                if an {
                    break;
                }
            }
            while !b_numeric {
                line_b.clear();
                if reader_b.read_line(&mut line_b).unwrap_or(0) == 0 {
                    break;
                }
                let (bn, _) = is_numeric_line(&line_b, &args.strip);
                if bn {
                    break;
                }
            }
            in_section = true;
        }

        if in_section {
            let a_vals = parse_values(&strip_a);
            let b_vals = parse_values(&strip_b);
            let num_comp = a_vals.len().min(b_vals.len());
            max_comp = max_comp.max(num_comp);

            let mut dump = false;
            for i in 0..num_comp {
                let d = (a_vals[i] - b_vals[i]).abs();
                diff_max[i] = diff_max[i].max(d);
                if args.big && i < diff_limits.len() && d > diff_limits[i] {
                    dump = true;
                }
            }

            if dump {
                if !did_big {
                    println!();
                    println!("Section {} lines with big differences:", numeric_sect);
                    did_big = true;
                }
                println!(" {}", line_a.trim());
                println!(" {}", line_b.trim());
            }
        }
    }

    if in_section {
        let section_limits = if numeric_sect <= num_limit_sets {
            &diff_limits[..num_limit_sets.min(diff_max.len())]
        } else {
            &[]
        };
        output_diffs(
            &diff_max,
            max_comp,
            section_limits,
            numeric_sect,
            &mut num_error,
            args.general,
        );
    }

    process::exit(num_error as i32);
}
