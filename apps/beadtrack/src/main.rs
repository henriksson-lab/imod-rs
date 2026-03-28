use clap::Parser;
use imod_core::Point3f;
use imod_fft::cross_correlate_2d;
use imod_model::{write_model, ImodContour, ImodModel, ImodObject};
use imod_mrc::MrcReader;
use imod_transforms::read_tilt_file;

/// Track fiducial gold beads through a tilt series by template matching.
///
/// Given a seed model with initial bead positions on one or more views,
/// tracks each bead across all views using cross-correlation with an
/// extracted bead template.
#[derive(Parser)]
#[command(name = "beadtrack", about = "Track fiducial beads through a tilt series")]
struct Args {
    /// Input tilt series (MRC)
    #[arg(short = 'i', long)]
    input: String,

    /// Seed model with initial bead positions (.mod)
    #[arg(short = 's', long)]
    seed: String,

    /// Output tracked model (.mod)
    #[arg(short = 'o', long)]
    output: String,

    /// Tilt angle file (.tlt)
    #[arg(short = 't', long)]
    tilt_file: String,

    /// Bead diameter in pixels
    #[arg(short = 'd', long, default_value_t = 10.0)]
    bead_diameter: f32,

    /// Search radius in pixels
    #[arg(short = 'r', long, default_value_t = 20.0)]
    search_radius: f32,
}

fn main() {
    let args = Args::parse();

    let _tilt_angles = read_tilt_file(&args.tilt_file).unwrap_or_else(|e| {
        eprintln!("Error reading tilt file: {}", e);
        std::process::exit(1);
    });

    let mut reader = MrcReader::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let h = reader.header().clone();
    let nx = h.nx as usize;
    let ny = h.ny as usize;
    let nz = h.nz as usize;

    let seed = imod_model::read_model(&args.seed).unwrap_or_else(|e| {
        eprintln!("Error reading seed model: {}", e);
        std::process::exit(1);
    });

    // Read all sections
    let mut sections: Vec<Vec<f32>> = Vec::with_capacity(nz);
    for z in 0..nz {
        sections.push(reader.read_slice_f32(z).unwrap());
    }

    // Extract seed bead positions
    let box_size = (args.bead_diameter * 1.5).ceil() as usize | 1; // odd
    let _half_box = box_size / 2;
    let search_r = args.search_radius as usize;
    let search_size = next_pow2(box_size + 2 * search_r);

    let mut bead_seeds: Vec<(f32, f32, usize)> = Vec::new(); // (x, y, view)
    for obj in &seed.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                let view = pt.z.round() as usize;
                if view < nz {
                    bead_seeds.push((pt.x, pt.y, view));
                }
            }
        }
    }

    eprintln!(
        "beadtrack: {} seed beads, {} views, box={}px, search={}px",
        bead_seeds.len(), nz, box_size, search_r
    );

    // Track each bead across all views
    let mut output_model = ImodModel {
        name: "tracked beads".into(),
        xmax: h.nx,
        ymax: h.ny,
        zmax: h.nz,
        ..Default::default()
    };

    for (bi, &(seed_x, seed_y, seed_view)) in bead_seeds.iter().enumerate() {
        let mut positions: Vec<Option<(f32, f32)>> = vec![None; nz];
        positions[seed_view] = Some((seed_x, seed_y));

        // Extract template from seed view
        let template = extract_box(&sections[seed_view], nx, ny, seed_x, seed_y, box_size);

        // Track forward from seed
        let mut cur_x = seed_x;
        let mut cur_y = seed_y;
        for v in (seed_view + 1)..nz {
            if let Some((new_x, new_y)) = track_bead(
                &template, &sections[v], nx, ny, cur_x, cur_y, box_size, search_size,
            ) {
                positions[v] = Some((new_x, new_y));
                cur_x = new_x;
                cur_y = new_y;
            } else {
                break;
            }
        }

        // Track backward from seed
        cur_x = seed_x;
        cur_y = seed_y;
        for v in (0..seed_view).rev() {
            if let Some((new_x, new_y)) = track_bead(
                &template, &sections[v], nx, ny, cur_x, cur_y, box_size, search_size,
            ) {
                positions[v] = Some((new_x, new_y));
                cur_x = new_x;
                cur_y = new_y;
            } else {
                break;
            }
        }

        let tracked: usize = positions.iter().filter(|p| p.is_some()).count();

        // Create model object for this bead
        let mut contour = ImodContour::default();
        for (v, pos) in positions.iter().enumerate() {
            if let Some((x, y)) = pos {
                contour.points.push(Point3f { x: *x, y: *y, z: v as f32 });
            }
        }

        let mut obj = ImodObject {
            name: format!("bead{}", bi + 1),
            red: 0.0,
            green: 1.0,
            blue: 0.0,
            ..Default::default()
        };
        obj.contours.push(contour);
        output_model.objects.push(obj);

        if bi < 5 || bi == bead_seeds.len() - 1 {
            eprintln!("  bead {}: tracked in {}/{} views", bi + 1, tracked, nz);
        }
    }

    write_model(&args.output, &output_model).unwrap();
    eprintln!("beadtrack: wrote {} tracked beads to {}", bead_seeds.len(), args.output);
}

fn extract_box(image: &[f32], nx: usize, ny: usize, cx: f32, cy: f32, size: usize) -> Vec<f32> {
    let half = size / 2;
    let mut box_data = vec![0.0f32; size * size];
    let ix = cx.round() as isize;
    let iy = cy.round() as isize;

    // Compute mean for fill
    let sum: f64 = image.iter().take(1000).map(|&v| v as f64).sum();
    let fill = (sum / 1000.0) as f32;

    for by in 0..size {
        for bx in 0..size {
            let sx = ix - half as isize + bx as isize;
            let sy = iy - half as isize + by as isize;
            if sx >= 0 && sx < nx as isize && sy >= 0 && sy < ny as isize {
                box_data[by * size + bx] = image[sy as usize * nx + sx as usize];
            } else {
                box_data[by * size + bx] = fill;
            }
        }
    }
    box_data
}

fn track_bead(
    template: &[f32],
    image: &[f32],
    nx: usize,
    ny: usize,
    pred_x: f32,
    pred_y: f32,
    box_size: usize,
    fft_size: usize,
) -> Option<(f32, f32)> {
    // Extract search area centered on predicted position
    let search_box = extract_box(image, nx, ny, pred_x, pred_y, fft_size);

    // Pad template to fft_size
    let mut tmpl_padded = vec![0.0f32; fft_size * fft_size];
    let offset = (fft_size - box_size) / 2;
    for y in 0..box_size {
        for x in 0..box_size {
            tmpl_padded[(y + offset) * fft_size + (x + offset)] = template[y * box_size + x];
        }
    }

    let cc = cross_correlate_2d(&search_box, &tmpl_padded, fft_size, fft_size);

    // Find peak
    let mut max_val = f32::NEG_INFINITY;
    let mut mx = 0usize;
    let mut my = 0usize;
    for y in 0..fft_size {
        for x in 0..fft_size {
            if cc[y * fft_size + x] > max_val {
                max_val = cc[y * fft_size + x];
                mx = x;
                my = y;
            }
        }
    }

    // Convert peak to shift
    let dx = if mx > fft_size / 2 { mx as f32 - fft_size as f32 } else { mx as f32 };
    let dy = if my > fft_size / 2 { my as f32 - fft_size as f32 } else { my as f32 };

    let new_x = pred_x + dx;
    let new_y = pred_y + dy;

    // Reject if out of bounds
    if new_x < 0.0 || new_x >= nx as f32 || new_y < 0.0 || new_y >= ny as f32 {
        return None;
    }

    Some((new_x, new_y))
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}
