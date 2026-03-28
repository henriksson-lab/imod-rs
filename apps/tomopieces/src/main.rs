use clap::Parser;
use imod_mrc::MrcReader;

/// Compute piece coordinates for chopping a tomogram into pieces for
/// processing (e.g. FFT-based filtering) within memory limits.
#[derive(Parser)]
#[command(name = "tomopieces", about = "Compute piece coordinates for tomogram montaging")]
struct Args {
    /// Input tomogram file or size as NX,NY,NZ
    #[arg(short = 't', long)]
    tomogram: String,

    /// Maximum megavoxels per piece
    #[arg(short = 'm', long, default_value_t = 80.0)]
    megavox: f64,

    /// X padding extent
    #[arg(long, default_value_t = 8)]
    xpad: i32,

    /// Y padding extent
    #[arg(long, default_value_t = 4)]
    ypad: i32,

    /// Z padding extent
    #[arg(long, default_value_t = 8)]
    zpad: i32,

    /// Maximum pieces in X (0 = auto)
    #[arg(long, default_value_t = 0)]
    xmaxpiece: i32,

    /// Maximum pieces in Y (0 = auto)
    #[arg(long, default_value_t = 0)]
    ymaxpiece: i32,

    /// Maximum pieces in Z (-1 = auto)
    #[arg(long, default_value_t = -1)]
    zmaxpiece: i32,

    /// Minimum overlap between pieces
    #[arg(long, default_value_t = 4)]
    minoverlap: i32,

    /// Use no-FFT sizes (just pad, no nice-frame rounding)
    #[arg(long, default_value = "false")]
    nofft: bool,
}

/// Find the next "nice" size >= n that is even and has no prime factor > limit.
fn nice_frame(n: i32, step: i32, limit: i32) -> i32 {
    let mut size = n;
    if size % step != 0 {
        size += step - size % step;
    }
    loop {
        if is_nice(size, limit) {
            return size;
        }
        size += step;
    }
}

fn is_nice(mut n: i32, limit: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let primes = [2, 3, 5, 7, 11, 13, 17, 19];
    for &p in &primes {
        if p > limit {
            break;
        }
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

/// Pad a size for FFT or plain padding.
fn pad_nice_if_fft(nx_extra: i32, nx_pad: i32, no_fft: bool) -> i32 {
    if no_fft {
        nx_extra + 2 * nx_pad
    } else {
        nice_frame(2 * ((nx_extra + 1) / 2 + nx_pad), 2, 19)
    }
}

/// Compute extraction and back-mapping ranges for one dimension.
fn get_ranges(
    nx: i32,
    num_pieces: i32,
    min_overlap: i32,
    nx_pad: i32,
    no_fft: bool,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let np = num_pieces as usize;
    let mut out_start = vec![0i32; np];
    let mut out_end = vec![0i32; np];
    let mut back_start = vec![0i32; np];
    let mut back_end = vec![0i32; np];

    let nx_extra = (nx + (num_pieces - 1) * min_overlap + num_pieces - 1) / num_pieces;
    let nx_out = pad_nice_if_fft(nx_extra, nx_pad, no_fft);
    let nx_back_offset = (nx_out - nx_extra) / 2;

    let lap_total = nx_extra * num_pieces - nx;
    let lap_base = lap_total / (num_pieces - 1).max(1);
    let lap_extra_count = lap_total % (num_pieces - 1).max(1);

    out_start[0] = 0;
    back_start[0] = nx_back_offset;

    for ip in 0..np {
        out_end[ip] = out_start[ip] + nx_extra - 1;
        if (ip as i32) < num_pieces - 1 {
            let mut lap = lap_base;
            if (ip as i32) < lap_extra_count {
                lap += 1;
            }
            let lap_top = lap / 2;
            let lap_bottom = lap - lap_top;
            out_start[ip + 1] = out_start[ip] + nx_extra - lap;
            back_end[ip] = nx_back_offset + nx_extra - 1 - lap_top;
            back_start[ip + 1] = nx_back_offset + lap_bottom;
        } else {
            back_end[ip] = nx_back_offset + nx_extra - 1;
        }
    }

    (out_start, out_end, back_start, back_end)
}

fn main() {
    let args = Args::parse();

    // Parse tomogram size: either from file or as NX,NY,NZ string
    let (nx, ny, nz) = if let Some(dims) = parse_size_string(&args.tomogram) {
        dims
    } else {
        // Try to open as MRC file
        let reader = MrcReader::open(&args.tomogram).unwrap_or_else(|e| {
            eprintln!("ERROR: TOMOPIECES - opening tomogram: {e}");
            std::process::exit(1);
        });
        let h = reader.header();
        (h.nx, h.ny, h.nz)
    };

    let megavox = args.megavox;
    let min_overlap = args.minoverlap;
    let nx_pad = args.xpad;
    let ny_pad = args.ypad;
    let nz_pad = args.zpad;
    let no_fft = args.nofft;

    let max_layer_pieces = 100;
    let mut max_pieces_x = args.xmaxpiece;
    let mut max_pieces_y = args.ymaxpiece;
    let mut max_pieces_z = args.zmaxpiece;

    // Set defaults for max pieces
    if max_pieces_x == 0 && (max_pieces_y == 0 || max_pieces_y == 1) {
        max_pieces_x = max_layer_pieces;
    } else if max_pieces_x == 1 && max_pieces_y == 0 {
        max_pieces_y = max_layer_pieces;
    } else {
        if max_pieces_x <= 0 {
            max_pieces_x = nx / 2 - 1;
        }
        if max_pieces_y <= 0 {
            max_pieces_y = ny / 2 - 1;
        }
    }
    if max_pieces_z < 0 {
        max_pieces_z = nz / 2 - 1;
    }

    // Search for optimal piece decomposition
    let mut perim_min: f64 = 10.0 * nx as f64 * ny as f64 * nz as f64;
    let mut piece_perim_min = perim_min;
    let mut best_x = 0i32;
    let mut best_y = 0i32;
    let mut best_z = 0i32;

    for num_x in 1..=max_pieces_x {
        let nx_extra = (nx + (num_x - 1) * min_overlap + num_x - 1) / num_x;
        let nx_out = pad_nice_if_fft(nx_extra, nx_pad, no_fft);

        let limit_y = if max_pieces_y == 0 {
            max_layer_pieces / num_x
        } else {
            max_pieces_y
        };

        for num_y in 1..=limit_y {
            let ny_extra = (ny + (num_y - 1) * min_overlap + num_y - 1) / num_y;
            let ny_out = pad_nice_if_fft(ny_extra, ny_pad, no_fft);

            let mut num_z = 1i32;
            let mut too_big = true;

            while num_z <= max_pieces_z && too_big {
                let nz_extra = (nz + (num_z - 1) * min_overlap + num_z - 1) / num_z;
                let nz_out = pad_nice_if_fft(nz_extra, nz_pad, no_fft);
                if (nx_out as f64) * (ny_out as f64) * (nz_out as f64) > megavox * 1.0e6 {
                    num_z += 1;
                } else {
                    too_big = false;
                }
            }

            let perim = (nx as f64) * (ny as f64) * (num_z as f64)
                + (nx as f64) * (nz as f64) * (num_y as f64)
                + (ny as f64) * (nz as f64) * (num_x as f64);
            let piece_perim =
                (nx_out as f64) * (ny_out as f64) + (nx_out as f64) * (nz_out_val(nz, num_z, min_overlap, nz_pad, no_fft) as f64) + (ny_out as f64) * (nz_out_val(nz, num_z, min_overlap, nz_pad, no_fft) as f64);

            if !too_big && (perim < perim_min || (perim == perim_min && piece_perim < piece_perim_min))
            {
                perim_min = perim;
                piece_perim_min = piece_perim;
                best_x = num_x;
                best_y = num_y;
                best_z = num_z;
            }
        }
    }

    if best_x == 0 {
        eprintln!("ERROR: TOMOPIECES - Pieces are all too large with given maximum numbers");
        std::process::exit(1);
    }

    // Compute ranges for each dimension
    let (ix_out_s, ix_out_e, ix_back_s, ix_back_e) =
        get_ranges(nx, best_x, min_overlap, nx_pad, no_fft);
    let (iy_out_s, iy_out_e, iy_back_s, iy_back_e) =
        get_ranges(ny, best_y, min_overlap, ny_pad, no_fft);
    let (iz_out_s, iz_out_e, iz_back_s, iz_back_e) =
        get_ranges(nz, best_z, min_overlap, nz_pad, no_fft);

    // Output: number of pieces per dimension
    println!("{:4}{:4}{:4}", best_x, best_y, best_z);

    // Output: extraction ranges for each piece (x_start,x_end,y_start,y_end,z_start,z_end)
    for iz in 0..best_z as usize {
        for iy in 0..best_y as usize {
            for ix in 0..best_x as usize {
                println!(
                    "{},{},{},{},{},{}",
                    ix_out_s[ix], ix_out_e[ix], iy_out_s[iy], iy_out_e[iy], iz_out_s[iz],
                    iz_out_e[iz]
                );
            }
        }
    }

    // Output: back-mapping ranges
    for i in 0..best_x as usize {
        println!("{},{}", ix_back_s[i], ix_back_e[i]);
    }
    for i in 0..best_y as usize {
        println!("{},{}", iy_back_s[i], iy_back_e[i]);
    }
    for i in 0..best_z as usize {
        println!("{},{}", iz_back_s[i], iz_back_e[i]);
    }
}

fn nz_out_val(nz: i32, num_z: i32, min_overlap: i32, nz_pad: i32, no_fft: bool) -> i32 {
    let nz_extra = (nz + (num_z - 1) * min_overlap + num_z - 1) / num_z;
    pad_nice_if_fft(nz_extra, nz_pad, no_fft)
}

fn parse_size_string(s: &str) -> Option<(i32, i32, i32)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() == 3 {
        let x = parts[0].trim().parse::<i32>().ok()?;
        let y = parts[1].trim().parse::<i32>().ok()?;
        let z = parts[2].trim().parse::<i32>().ok()?;
        Some((x, y, z))
    } else {
        None
    }
}
