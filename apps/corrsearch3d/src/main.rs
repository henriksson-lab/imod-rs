use clap::Parser;
use imod_core::Point3f;
use imod_fft::cross_correlate_2d;
use imod_model::{write_model, ImodContour, ImodModel, ImodObject};
use imod_mrc::MrcReader;

/// Search for corresponding positions between two volumes using 3D
/// cross-correlation of patches. Used for dual-axis tomogram combination.
///
/// Divides volume A into patches at specified positions and finds the best
/// matching position in volume B by cross-correlating XY slices at each Z.
#[derive(Parser)]
#[command(name = "corrsearch3d", about = "3D patch correlation between two volumes")]
struct Args {
    /// Volume A (MRC)
    #[arg(short = 'a', long)]
    vol_a: String,

    /// Volume B (MRC)
    #[arg(short = 'b', long)]
    vol_b: String,

    /// Output model with displacement vectors
    #[arg(short = 'o', long)]
    output: String,

    /// Patch size in X and Y (pixels)
    #[arg(short = 'p', long, default_value_t = 64)]
    patch_size: usize,

    /// Number of patches in X
    #[arg(long, default_value_t = 5)]
    nx_patches: usize,

    /// Number of patches in Y
    #[arg(long, default_value_t = 5)]
    ny_patches: usize,

    /// Number of patches in Z
    #[arg(long, default_value_t = 3)]
    nz_patches: usize,
}

fn main() {
    let args = Args::parse();

    let mut reader_a = MrcReader::open(&args.vol_a).unwrap_or_else(|e| {
        eprintln!("Error opening vol A: {}", e);
        std::process::exit(1);
    });
    let mut reader_b = MrcReader::open(&args.vol_b).unwrap_or_else(|e| {
        eprintln!("Error opening vol B: {}", e);
        std::process::exit(1);
    });

    let ha = reader_a.header().clone();
    let hb = reader_b.header().clone();
    let nx = ha.nx as usize;
    let ny = ha.ny as usize;
    let nz = ha.nz as usize;

    eprintln!(
        "corrsearch3d: vol A {}x{}x{}, vol B {}x{}x{}, patch {}x{}, grid {}x{}x{}",
        nx, ny, nz, hb.nx, hb.ny, hb.nz,
        args.patch_size, args.patch_size,
        args.nx_patches, args.ny_patches, args.nz_patches
    );

    // Read both volumes
    let vol_a: Vec<Vec<f32>> = (0..nz).map(|z| reader_a.read_slice_f32(z).unwrap()).collect();
    let vol_b: Vec<Vec<f32>> = (0..hb.nz as usize).map(|z| reader_b.read_slice_f32(z).unwrap()).collect();

    let fft_size = next_pow2(args.patch_size * 2);

    // Generate patch centers
    let mut model = ImodModel {
        name: "corrsearch3d displacements".into(),
        xmax: ha.nx,
        ymax: ha.ny,
        zmax: ha.nz,
        ..Default::default()
    };

    let mut obj = ImodObject {
        name: "patches".into(),
        red: 0.0,
        green: 1.0,
        blue: 1.0,
        ..Default::default()
    };

    let x_step = if args.nx_patches > 1 { (nx - args.patch_size) / (args.nx_patches - 1) } else { 0 };
    let y_step = if args.ny_patches > 1 { (ny - args.patch_size) / (args.ny_patches - 1) } else { 0 };
    let z_step = if args.nz_patches > 1 { (nz - 1) / (args.nz_patches - 1) } else { 0 };

    let half = args.patch_size / 2;
    let mut n_patches = 0;

    for iz in 0..args.nz_patches {
        let cz = if args.nz_patches > 1 { iz * z_step } else { nz / 2 };
        if cz >= nz { continue; }

        for iy in 0..args.ny_patches {
            let cy = half + iy * y_step.max(1);
            if cy + half > ny { continue; }

            for ix in 0..args.nx_patches {
                let cx = half + ix * x_step.max(1);
                if cx + half > nx { continue; }

                // Extract patch from vol A at this Z
                let patch_a = extract_patch(&vol_a[cz], nx, cx, cy, args.patch_size);
                let patch_b = extract_patch(&vol_b[cz.min(vol_b.len() - 1)], hb.nx as usize, cx, cy, args.patch_size);

                // Pad and cross-correlate
                let pa = pad_patch(&patch_a, args.patch_size, fft_size);
                let pb = pad_patch(&patch_b, args.patch_size, fft_size);
                let cc = cross_correlate_2d(&pa, &pb, fft_size, fft_size);

                // Find peak
                let (px, py) = find_peak(&cc, fft_size);
                let dx = if px > fft_size / 2 { px as f32 - fft_size as f32 } else { px as f32 };
                let dy = if py > fft_size / 2 { py as f32 - fft_size as f32 } else { py as f32 };

                // Store as a contour with two points: position and position+displacement
                let mut cont = ImodContour::default();
                cont.points.push(Point3f { x: cx as f32, y: cy as f32, z: cz as f32 });
                cont.points.push(Point3f { x: cx as f32 + dx, y: cy as f32 + dy, z: cz as f32 });
                obj.contours.push(cont);
                n_patches += 1;
            }
        }
    }

    model.objects.push(obj);
    write_model(&args.output, &model).unwrap();
    eprintln!("corrsearch3d: {} patches correlated, wrote {}", n_patches, args.output);
}

fn extract_patch(slice: &[f32], nx: usize, cx: usize, cy: usize, size: usize) -> Vec<f32> {
    let half = size / 2;
    let mut patch = vec![0.0f32; size * size];
    for py in 0..size {
        let sy = cy - half + py;
        for px in 0..size {
            let sx = cx - half + px;
            patch[py * size + px] = slice[sy * nx + sx];
        }
    }
    patch
}

fn pad_patch(patch: &[f32], size: usize, fft_size: usize) -> Vec<f32> {
    let sum: f64 = patch.iter().map(|&v| v as f64).sum();
    let mean = (sum / patch.len() as f64) as f32;
    let mut padded = vec![mean; fft_size * fft_size];
    let off = (fft_size - size) / 2;
    for y in 0..size {
        for x in 0..size {
            padded[(y + off) * fft_size + (x + off)] = patch[y * size + x];
        }
    }
    padded
}

fn find_peak(cc: &[f32], n: usize) -> (usize, usize) {
    let mut max_v = f32::NEG_INFINITY;
    let (mut mx, mut my) = (0, 0);
    for y in 0..n {
        for x in 0..n {
            if cc[y * n + x] > max_v { max_v = cc[y * n + x]; mx = x; my = y; }
        }
    }
    (mx, my)
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}
