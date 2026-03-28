//! edgemtf - Compute MTF from an edge image.
//!
//! Reads an MRC image containing an edge, computes the line-spread function
//! (derivative of the edge profile), takes its FFT to obtain the MTF curve,
//! and writes output files with the MTF and intermediate data.
//!
//! Translated from IMOD's edgemtf.f

use clap::Parser;
use imod_mrc::MrcReader;
use std::io::Write;
use std::process;

#[derive(Parser)]
#[command(name = "edgemtf", about = "Compute MTF curve from edge image")]
struct Args {
    /// Input MRC image file containing edge
    #[arg(short = 'i', long)]
    input: String,

    /// Root name for output files (generates .out and per-section .mtf files)
    #[arg(short = 'r', long)]
    rootname: String,

    /// Starting and ending sections (0-based)
    #[arg(short = 's', long, num_args = 2, value_names = ["START", "END"])]
    sections: Option<Vec<i32>>,

    /// Number of points for ring averaging of MTF
    #[arg(short = 'n', long, default_value_t = 20)]
    points: i32,

    /// Number of lines to sum (collapse) in Y
    #[arg(long, default_value_t = 3)]
    sum: i32,

    /// Number of lines to average for reference
    #[arg(long, default_value_t = 2)]
    average: i32,

    /// Normalization components
    #[arg(long, default_value_t = 4)]
    components: i32,

    /// Binning of input images (1-4, adjusts defaults)
    #[arg(short = 'b', long, default_value_t = 1)]
    binning: i32,

    /// Crossing value for MTF reporting
    #[arg(short = 'c', long, default_value_t = 0.5)]
    cross: f32,

    /// Limit on derivative computation (number of pixels)
    #[arg(short = 'z', long)]
    zero: Option<i32>,
}

fn nice_fft_size(mut n: usize) -> usize {
    // Find a size >= n that factors into small primes (2,3,5)
    n = n.max(2);
    if n % 2 != 0 {
        n += 1;
    }
    loop {
        let mut m = n;
        while m % 2 == 0 {
            m /= 2;
        }
        while m % 3 == 0 {
            m /= 3;
        }
        while m % 5 == 0 {
            m /= 5;
        }
        if m == 1 {
            return n;
        }
        n += 2;
    }
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: edgemtf - opening input: {}", e);
        process::exit(1);
    });

    let header = reader.header();
    let nx_full = header.nx as usize;
    let ny_full = header.ny as usize;
    let nz = header.nz as usize;

    let iz_start = args.sections.as_ref().map_or(0, |v| v[0] as usize);
    let iz_end = args.sections.as_ref().map_or(nz - 1, |v| v[1] as usize);

    let binning = args.binning.clamp(1, 4) as usize;
    let ncollapse_defaults = [3, 2, 1, 1];
    let nlinavg_defaults = [2, 2, 2, 1];
    let navgnorm_defaults = [4, 2, 2, 1];

    let ncollapse = if args.sum != 3 {
        args.sum as usize
    } else {
        ncollapse_defaults[binning - 1]
    };
    let _nlinavg = if args.average != 2 {
        args.average as usize
    } else {
        nlinavg_defaults[binning - 1]
    };
    let navgnorm = if args.components != 4 {
        args.components as usize
    } else {
        navgnorm_defaults[binning - 1]
    };

    // Compute FFT-friendly NX
    let mut nx = 2 * (nx_full / 2);
    while nx != nice_fft_size(nx) {
        nx -= 2;
    }
    let nxo2 = nx / 2;
    let nx21 = nxo2 + 1;
    let delx = 1.0 / nx as f32;
    let nring = args.points as usize;
    let navgring = nxo2 / nring;
    let nxlim = args.zero.map_or(nx, |z| z as usize);
    let cross = args.cross;

    // Open main output file
    let out_path = format!("{}.out", args.rootname);
    let mut out_file = std::fs::File::create(&out_path).unwrap_or_else(|e| {
        eprintln!("ERROR: edgemtf - creating output {}: {}", out_path, e);
        process::exit(1);
    });

    for kk in iz_start..=iz_end {
        // Read section
        let section_data = reader.read_slice_f32(kk).unwrap_or_else(|e| {
            eprintln!("ERROR: edgemtf - reading section {}: {}", kk, e);
            process::exit(1);
        });

        // Copy into working array (trim to nx x ny_full)
        let mut array: Vec<f32> = vec![0.0; nx * ny_full];
        for iy in 0..ny_full {
            for ix in 0..nx {
                array[iy * nx + ix] = section_data[iy * nx_full + ix];
            }
        }

        // Collapse lines
        let ny = ny_full / ncollapse;
        if ncollapse > 1 {
            let mut collapsed = vec![0.0f32; nx * ny];
            for iyn in 0..ny {
                let iyst = iyn * ncollapse;
                for ix in 0..nx {
                    let mut sum = 0.0;
                    for iy in iyst..iyst + ncollapse {
                        sum += array[iy * nx + ix];
                    }
                    collapsed[iyn * nx + ix] = sum / ncollapse as f32;
                }
            }
            array = collapsed;
        }

        // Sum all lines to get reference edge profile
        let mut ref_line = vec![0.0f32; nx];
        for ix in 0..nx {
            let mut sum = 0.0;
            for iy in 0..ny {
                sum += array[iy * nx + ix];
            }
            ref_line[ix] = sum / ny as f32;
        }

        // Compute derivative of reference
        let mut deriv = vec![0.0f32; nx];
        for i in 0..nx - 1 {
            if i < nxlim {
                deriv[i] = ref_line[i + 1] - ref_line[i];
            }
        }

        // Simple FFT magnitude computation of the LSF
        // Compute magnitude spectrum of derivative
        let mut ftmag = vec![0.0f32; nx21];
        {
            // DFT of derivative
            for k in 0..nx21 {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for n in 0..nx {
                    let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / nx as f64;
                    re += deriv[n] as f64 * angle.cos();
                    im += deriv[n] as f64 * angle.sin();
                }
                ftmag[k] = (re * re + im * im).sqrt() as f32;
            }
        }

        // Normalize
        let avg1: f32 = if navgnorm > 0 {
            ftmag[1..=navgnorm.min(nx21 - 1)].iter().sum::<f32>() / navgnorm as f32
        } else {
            1.0
        };
        let avg1 = if avg1 > 0.0 { avg1 } else { 1.0 };

        // Write MTF file for this section
        let mtf_path = format!("{}-{}.mtf", args.rootname, kk);
        let mut mtf_file = std::fs::File::create(&mtf_path).unwrap_or_else(|e| {
            eprintln!("ERROR: edgemtf - creating {}: {}", mtf_path, e);
            process::exit(1);
        });

        let mut four_last = 0.0f32;
        let mut avg_last = 1.0f32;
        let mut fcrs = -99.0f32;
        let mut istr = 1usize; // 0-based index 1 (skip DC)

        for iring in 0..nring {
            let iend = if iring == nring - 1 {
                nx21 - 1
            } else {
                istr + navgring - 1
            };
            let four = (0.5 * (iend + istr) as f64) as f32 * delx;
            let mut ring_sum = 0.0f32;
            for i in istr..=iend {
                ring_sum += ftmag[i];
            }
            let avg = ring_sum / (avg1 * (iend + 1 - istr) as f32);

            let _ = writeln!(out_file, "{:5}{:7.4}{:9.5}", kk, four, avg);
            let _ = writeln!(mtf_file, "{:7.4}{:9.5}", four, avg);

            if avg_last > cross && avg <= cross && fcrs == -99.0 {
                fcrs = four_last + (cross - avg_last) * (four - four_last) / (avg - avg_last);
                println!("MTF ={:6.2} at frequency{:8.4}/pixel", cross, fcrs);
            }
            four_last = four;
            avg_last = avg;
            istr = iend + 1;
        }

        // Write full spectrum
        let mut x = delx;
        for ix in 1..nx21 {
            let avg = ftmag[ix] / avg1;
            let _ = writeln!(out_file, "{:5}{:7.4}{:9.5}", kk + 100, x, avg);
            x += delx;
        }

        // Write reference profile
        for ix in 0..nx {
            let _ = writeln!(out_file, "{:5}{:6.1}{:12.5}", kk + 200, ix as f32, ref_line[ix]);
        }

        // Write derivative
        for ix in 0..nx - 1 {
            let _ = writeln!(
                out_file,
                "{:5}{:6.1}{:12.5}",
                kk + 300,
                ix as f32 + 0.5,
                deriv[ix]
            );
        }
    }

    println!(
        "MTF computed for sections {} to {}, output in {}",
        iz_start, iz_end, out_path
    );
}
