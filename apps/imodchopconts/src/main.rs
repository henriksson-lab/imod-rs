use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_math::parse_list;
use imod_model::{ImodContour, ImodModel, ImodObject, read_model, write_model};

/// Chop contours into pieces at Z boundaries or at specified length.
///
/// Takes an IMOD model and chops up contours that are longer than a specified
/// length into overlapping pieces. This is useful for patch tracking models
/// where long contours need to be split.
#[derive(Parser)]
#[command(name = "imodchopconts", version, about)]
struct Args {
    /// Input model file
    #[arg(short = 'i', long)]
    input: String,

    /// Output model file
    #[arg(short = 'o', long)]
    output: String,

    /// Length of new contour pieces
    #[arg(short = 'l', long)]
    length: Option<i32>,

    /// Minimum overlap between pieces (use -1 for zero overlap enforced)
    #[arg(short = 'O', long, default_value_t = 4)]
    overlap: i32,

    /// Number of pieces to split into (alternative to length)
    #[arg(short = 'n', long)]
    number: Option<i32>,

    /// Assign surface numbers to new contours
    #[arg(short = 's', long)]
    surfaces: bool,

    /// List of objects to process (e.g. "1,3-5")
    #[arg(short = 'L', long)]
    objects: Option<String>,
}

/// IMOD_OBJFLAG_THICK_CONT
const IMOD_OBJFLAG_THICK_CONT: u32 = 1 << 10;

fn main() {
    let args = Args::parse();

    let mut model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: imodchopconts - Reading model {}: {}", args.input, e);
        process::exit(1);
    });

    // Parse object list
    let obj_list: Option<Vec<i32>> = args.objects.as_ref().map(|s| {
        parse_list(s).unwrap_or_else(|e| {
            eprintln!("ERROR: imodchopconts - Bad entry in list of objects: {}", e);
            process::exit(1);
        })
    });

    let num_obj = model.objects.len();

    // Determine maximum contour length across selected objects
    let mut max_len: usize = 1;
    for (ob, obj) in model.objects.iter().enumerate() {
        if !object_in_list(ob + 1, &obj_list) {
            continue;
        }
        for cont in &obj.contours {
            max_len = max_len.max(cont.points.len());
        }
    }

    // Set up default contour length
    let mut len_contour = (model.zmax as usize).max(max_len);
    let mut min_overlap = args.overlap;
    let mut no_overlap = false;
    let len_entered = args.length.is_some();

    if let Some(len_in) = args.length {
        if len_in == -1 {
            len_contour = 16usize.max(model.zmax as usize / 5);
        } else {
            len_contour = len_in as usize;
        }
    }
    if len_contour == 0 {
        eprintln!("ERROR: imodchopconts - New contour length must be positive");
        process::exit(1);
    }

    if len_contour == 1 {
        min_overlap = 0;
    }
    if min_overlap == -1 {
        no_overlap = true;
        min_overlap = 0;
    }
    if min_overlap < -1 {
        eprintln!("ERROR: imodchopconts - Overlap cannot be negative (other than -1)");
        process::exit(1);
    }
    let min_overlap = min_overlap as usize;

    if let Some(num) = args.number {
        if len_entered {
            if let Some(l) = args.length {
                if l > 0 {
                    eprintln!("ERROR: imodchopconts - Cannot enter both -number and -length > 0");
                    process::exit(1);
                }
            }
        }
        let num = num as usize;
        len_contour = (max_len + (num - 1) * min_overlap) / num;
        println!("Maximum contour length = {}, length for new contours = {}", max_len, len_contour);
    }

    if len_contour <= min_overlap || (len_contour > 1 && len_contour < min_overlap + 2) {
        eprintln!("ERROR: imodchopconts - Contour length must be greater than the overlap{}",
                  if len_contour > 1 { " + 1" } else { "" });
        process::exit(1);
    }

    let mut num_before = 0usize;
    let mut num_after = 0usize;

    for ob in 0..num_obj {
        if !object_in_list(ob + 1, &obj_list) {
            continue;
        }

        let old_contours = std::mem::take(&mut model.objects[ob].contours);
        let num_to_cut = old_contours.len();
        num_before += num_to_cut;

        // Check if any contour exceeds length
        let local_max = old_contours.iter().map(|c| c.points.len()).max().unwrap_or(0);
        if local_max <= len_contour {
            // No cutting needed -- just put them back
            num_after += old_contours.len();
            model.objects[ob].contours = old_contours;
            model.objects[ob].flags |= IMOD_OBJFLAG_THICK_CONT;
            continue;
        }

        let mut surf = 1i32;
        let mut new_contours = Vec::new();

        for cont in &old_contours {
            let ipnt = cont.points.len();
            if ipnt <= len_contour {
                // Just copy
                let mut nc = cont.clone();
                if args.surfaces {
                    nc.surf = surf;
                }
                new_contours.push(nc);
            } else {
                let len_conts = len_contour.min(ipnt);
                let num_cont = (ipnt - 1) / (len_conts - min_overlap) + 1;
                let lap_total = num_cont * len_conts - ipnt;
                let lap_base = if num_cont > 1 { lap_total / (num_cont - 1) } else { lap_total };
                let lap_remainder = if num_cont > 1 { lap_total % (num_cont - 1) } else { 0 };

                let (len_conts, lap_base, lap_remainder) = if no_overlap {
                    let lc = ipnt / num_cont;
                    let lr = ipnt % num_cont;
                    (lc, 0usize, lr)
                } else {
                    (len_conts, lap_base, lap_remainder)
                };

                let mut pt_base = 0usize;
                for new_co in 0..num_cont {
                    let mut ind = pt_base + len_conts - 1;
                    if no_overlap && new_co < lap_remainder {
                        ind += 1;
                    }
                    ind = ind.min(ipnt - 1);

                    let mut nc = ImodContour {
                        points: cont.points[pt_base..=ind].to_vec(),
                        flags: cont.flags,
                        time: cont.time,
                        surf: if args.surfaces { surf } else { cont.surf },
                        sizes: None,
                    };
                    new_contours.push(nc);

                    // Advance base
                    pt_base += len_conts - lap_base;
                    if new_co < lap_remainder {
                        if no_overlap {
                            pt_base += 1;
                        } else {
                            pt_base -= 1;
                        }
                    }
                }
            }
            surf += 1;
        }

        num_after += new_contours.len();
        model.objects[ob].contours = new_contours;
        model.objects[ob].flags |= IMOD_OBJFLAG_THICK_CONT;
    }

    write_model(&args.output, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: imodchopconts - Writing model: {}", e);
        process::exit(1);
    });

    println!("Number of contours in selected objects changed from {} to {}", num_before, num_after);
}

fn object_in_list(ob: usize, list: &Option<Vec<i32>>) -> bool {
    match list {
        None => true,
        Some(v) => v.contains(&(ob as i32)),
    }
}
