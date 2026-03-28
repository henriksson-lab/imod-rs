use clap::Parser;

/// Find the next valid FFT frame size(s) >= the given value(s).
/// By default requires no prime factor greater than 19.
#[derive(Parser)]
#[command(name = "goodframe", about = "Find good frame boundaries for FFT")]
struct Args {
    /// Use the FFTW-based nice limit instead of the default limit of 19
    #[arg(short = 'n', long)]
    fftw_limit: bool,

    /// One or more sizes to round up to a good FFT size
    sizes: Vec<i32>,
}

/// Find the smallest integer >= n that is divisible by `step` and
/// whose remaining factors are all <= `limit`.
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

/// Check whether n factors completely into primes <= limit.
fn is_nice(mut n: i32, limit: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
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

/// The FFTW "nice" limit -- FFTW handles factors up to ~13 well.
fn nice_fft_limit() -> i32 {
    13
}

fn main() {
    let args = Args::parse();

    let limit = if args.fftw_limit {
        nice_fft_limit()
    } else {
        19
    };

    if args.sizes.is_empty() {
        eprintln!("Usage: goodframe [-n] SIZE [SIZE ...]");
        std::process::exit(1);
    }

    let results: Vec<i32> = args
        .sizes
        .iter()
        .map(|&nx| nice_frame(nx, 2, limit))
        .collect();

    let formatted: Vec<String> = results.iter().map(|n| format!("{n}")).collect();
    println!(" {}", formatted.join(" "));
}
