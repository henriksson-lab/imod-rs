use clap::Parser;
use imod_core::MrcMode;
use imod_math::min_max_mean;
use imod_model::read_model;
use imod_mrc::{MrcHeader, MrcReader, MrcWriter};
use imod_slice::Slice;

/// Erase gold beads or other features from a tomographic reconstruction
/// using an IMOD model that marks their locations.
///
/// Each model point defines the center of a bead. The bead is replaced
/// by interpolating from surrounding pixels.
#[derive(Parser)]
#[command(name = "eraser", about = "Erase gold beads from a reconstruction using a model")]
struct Args {
    /// Input MRC file (reconstruction)
    #[arg(short = 'i', long)]
    input: String,

    /// Output MRC file
    #[arg(short = 'o', long)]
    output: String,

    /// Model file with bead positions (.mod)
    #[arg(short = 'm', long)]
    model: String,

    /// Bead radius in pixels
    #[arg(short = 'r', long, default_value_t = 5.0)]
    radius: f32,

    /// Extra border around bead for sampling replacement values
    #[arg(short = 'b', long, default_value_t = 2.0)]
    border: f32,
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.model).unwrap_or_else(|e| {
        eprintln!("Error reading model: {}", e);
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

    // Collect all bead positions from model, grouped by Z
    let mut beads_by_z: Vec<Vec<(f32, f32)>> = vec![Vec::new(); nz];
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                let z = pt.z.round() as usize;
                if z < nz {
                    beads_by_z[z].push((pt.x, pt.y));
                }
            }
        }
    }

    let total_beads: usize = beads_by_z.iter().map(|b| b.len()).sum();
    eprintln!("eraser: {} beads in {} sections, radius={:.1}", total_beads, nz, args.radius);

    let mut out_header = MrcHeader::new(h.nx, h.ny, h.nz, MrcMode::Float);
    out_header.xlen = h.xlen;
    out_header.ylen = h.ylen;
    out_header.zlen = h.zlen;
    out_header.add_label(&format!("eraser: {} beads, radius={:.1}", total_beads, args.radius));

    let mut writer = MrcWriter::create(&args.output, out_header).unwrap();

    let mut gmin = f32::MAX;
    let mut gmax = f32::MIN;
    let mut gsum = 0.0_f64;

    let outer_r = args.radius + args.border;
    let r_sq = args.radius * args.radius;
    let outer_sq = outer_r * outer_r;

    for z in 0..nz {
        let data = reader.read_slice_f32(z).unwrap();
        let mut slice = Slice::from_data(nx, ny, data);

        for &(bx, by) in &beads_by_z[z] {
            // Sample ring between radius and outer_r to get replacement value
            let mut ring_sum = 0.0f32;
            let mut ring_count = 0;

            let ir = outer_r.ceil() as i32;
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let d_sq = (dx * dx + dy * dy) as f32;
                    if d_sq > r_sq && d_sq <= outer_sq {
                        let px = (bx + dx as f32).round() as isize;
                        let py = (by + dy as f32).round() as isize;
                        if px >= 0 && px < nx as isize && py >= 0 && py < ny as isize {
                            ring_sum += slice.get(px as usize, py as usize);
                            ring_count += 1;
                        }
                    }
                }
            }

            let fill = if ring_count > 0 {
                ring_sum / ring_count as f32
            } else {
                slice.statistics().2 // use image mean
            };

            // Replace pixels inside bead radius
            for dy in -ir..=ir {
                for dx in -ir..=ir {
                    let d_sq = (dx * dx + dy * dy) as f32;
                    if d_sq <= r_sq {
                        let px = (bx + dx as f32).round() as isize;
                        let py = (by + dy as f32).round() as isize;
                        if px >= 0 && px < nx as isize && py >= 0 && py < ny as isize {
                            // Smooth transition at edge
                            let d = d_sq.sqrt();
                            let weight = if d > args.radius - 1.0 {
                                args.radius - d // 0..1 at boundary
                            } else {
                                1.0
                            };
                            let weight = weight.clamp(0.0, 1.0);
                            let old = slice.get(px as usize, py as usize);
                            slice.set(px as usize, py as usize, old * (1.0 - weight) + fill * weight);
                        }
                    }
                }
            }
        }

        let (smin, smax, smean) = min_max_mean(&slice.data);
        if smin < gmin { gmin = smin; }
        if smax > gmax { gmax = smax; }
        gsum += smean as f64 * (nx * ny) as f64;

        writer.write_slice_f32(&slice.data).unwrap();
    }

    writer.finish(gmin, gmax, (gsum / (nx * ny * nz) as f64) as f32).unwrap();
}
