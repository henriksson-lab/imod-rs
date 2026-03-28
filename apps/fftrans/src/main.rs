use clap::Parser;
use imod_core::MrcMode;
use imod_fft::{fft_c2r_2d, fft_r2c_2d};
use imod_math::min_max_mean;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use rustfft::num_complex::Complex;

/// Compute forward or inverse 2D FFT of MRC image sections.
///
/// Forward: real image (mode 0/1/2) -> complex FFT (mode 4, nx/2+1 complex per row).
/// Inverse: complex FFT (mode 3/4) -> real image (mode 2).
#[derive(Parser)]
#[command(name = "fftrans", about = "2D Fourier transform of MRC images")]
struct Args {
    /// Input MRC file
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Quiet mode (suppress informational output)
    #[arg(short = 'q', long)]
    quiet: bool,
}

fn main() {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: FFTRANS - opening input: {e}");
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;
    let mode = h.data_mode().unwrap_or_else(|| {
        eprintln!("ERROR: FFTRANS - unsupported input mode {}", h.mode);
        std::process::exit(1);
    });

    let is_complex = matches!(mode, MrcMode::ComplexShort | MrcMode::ComplexFloat);

    if !args.quiet {
        eprintln!("FFTRANS: Fourier Transform Program");
    }

    if is_complex {
        inverse_transform(&args, &mut reader, nx, ny, nz, &h);
    } else {
        forward_transform(&args, &mut reader, nx, ny, nz, &h);
    }
}

/// Forward FFT: real image -> complex output (MRC mode 4).
fn forward_transform(
    args: &Args,
    reader: &mut MrcReader,
    nx: usize,
    ny: usize,
    nz: usize,
    h: &MrcHeader,
) {
    if nx % 2 != 0 {
        eprintln!("ERROR: FFTRANS - Image size in X must be even");
        std::process::exit(1);
    }

    let nxc = nx / 2 + 1; // complex values per row in output

    // Output header: nx = nxc (each pixel = 2 floats), mode = 4 (ComplexFloat)
    let mut out_header = MrcHeader::new(nxc as i32, ny as i32, nz as i32, MrcMode::ComplexFloat);
    out_header.mx = nxc as i32;
    out_header.my = h.my;
    out_header.mz = h.mz;
    // Adjust cell dimensions to preserve pixel size
    if h.mx > 0 {
        out_header.xlen = h.xlen * nxc as f32 / h.mx as f32;
    }
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.add_label("fftrans: Forward Fourier Transform");

    if !args.quiet {
        eprintln!("FFTRANS: Computing forward Fourier transform");
    }

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: FFTRANS - creating output: {e}");
        std::process::exit(1);
    });

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap_or_else(|e| {
            eprintln!("ERROR: FFTRANS - reading section {z}: {e}");
            std::process::exit(1);
        });

        let fft = fft_r2c_2d(&data, nx, ny);

        // Compute stats on the complex amplitudes
        let amps: Vec<f32> = fft.iter().map(|c| c.norm()).collect();
        let (smin, smax, smean) = min_max_mean(&amps);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * amps.len() as f64;

        // Write complex data as raw bytes (pairs of f32: re, im)
        let raw = complex_to_bytes(&fft);
        writer.write_slice_raw(&raw).unwrap_or_else(|e| {
            eprintln!("ERROR: FFTRANS - writing section {z}: {e}");
            std::process::exit(1);
        });

        if !args.quiet && nz > 1 {
            eprintln!(
                " Min,Max,Mean for section {z}: {smin:.5e} {smax:.5e} {smean:.5e}"
            );
        }
    }

    let gmean = (gsum / (nxc * ny * nz) as f64) as f32;
    if !args.quiet {
        eprintln!(
            "\n Overall Min,Max,Mean values: {gmin:.5e} {gmax:.5e} {gmean:.5e}"
        );
    }
    writer.finish(gmin, gmax, gmean).unwrap();
}

/// Inverse FFT: complex input (mode 3/4) -> real output (mode 2).
fn inverse_transform(
    args: &Args,
    reader: &mut MrcReader,
    nx_header: usize,
    ny: usize,
    nz: usize,
    h: &MrcHeader,
) {
    // In MRC FFT convention, header nx = nxc = nx_real/2 + 1
    let nxc = nx_header;
    let nx_real = (nxc - 1) * 2;

    let mut out_header = MrcHeader::new(nx_real as i32, ny as i32, nz as i32, MrcMode::Float);
    out_header.mx = nx_real as i32;
    out_header.my = h.my;
    out_header.mz = h.mz;
    // Adjust cell dimensions back
    if h.mx > 0 {
        out_header.xlen = h.xlen * nx_real as f32 / h.mx as f32;
    }
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.add_label("fftrans: Inverse Fourier Transform");

    if !args.quiet {
        eprintln!("FFTRANS: Computing inverse Fourier transform");
    }

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap_or_else(|e| {
        eprintln!("ERROR: FFTRANS - creating output: {e}");
        std::process::exit(1);
    });

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    for z in 0..nz {
        // Read the complex slice as raw bytes and parse into Complex<f32>
        let raw = reader.read_slice_raw(z).unwrap_or_else(|e| {
            eprintln!("ERROR: FFTRANS - reading section {z}: {e}");
            std::process::exit(1);
        });
        let fft_data = bytes_to_complex(&raw, reader.is_swapped());

        let real = fft_c2r_2d(&fft_data, nx_real, ny);

        let (smin, smax, smean) = min_max_mean(&real);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * real.len() as f64;

        writer.write_slice_f32(&real).unwrap_or_else(|e| {
            eprintln!("ERROR: FFTRANS - writing section {z}: {e}");
            std::process::exit(1);
        });

        if !args.quiet && nz > 1 {
            eprintln!(
                " Min,Max,Mean for section {z}: {smin:.5e} {smax:.5e} {smean:.5e}"
            );
        }
    }

    let gmean = (gsum / (nx_real * ny * nz) as f64) as f32;
    if !args.quiet {
        eprintln!(
            "\n Overall Min,Max,Mean values: {gmin:.5e} {gmax:.5e} {gmean:.5e}"
        );
    }
    writer.finish(gmin, gmax, gmean).unwrap();
}

/// Convert a slice of Complex<f32> to raw bytes (little-endian pairs of f32).
fn complex_to_bytes(data: &[Complex<f32>]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for c in data {
        buf.extend_from_slice(&c.re.to_le_bytes());
        buf.extend_from_slice(&c.im.to_le_bytes());
    }
    buf
}

/// Parse raw bytes into Complex<f32> values.
fn bytes_to_complex(raw: &[u8], swapped: bool) -> Vec<Complex<f32>> {
    let conv = if swapped {
        f32::from_be_bytes as fn([u8; 4]) -> f32
    } else {
        f32::from_le_bytes
    };
    raw.chunks_exact(8)
        .map(|chunk| {
            let re = conv([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let im = conv([chunk[4], chunk[5], chunk[6], chunk[7]]);
            Complex::new(re, im)
        })
        .collect()
}
