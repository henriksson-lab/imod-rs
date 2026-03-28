use clap::Parser;
use imod_core::Point3f;
use imod_model::{read_model, write_model, ImodContour, ImodModel, ImodObject};

/// Sort/split contours into new objects based on whether their points fall
/// inside closed contours in specified mask objects.
///
/// For each combination of input object and mask object, a new object is created
/// containing the contours (or points) from the input object that fall inside
/// the mask object's contours at the corresponding Z level.
#[derive(Parser)]
#[command(name = "slashmasksort", about = "Sort masks for PEET")]
struct Args {
    /// List of input object numbers to sort/split (comma-separated, e.g., "1,4-5")
    #[arg(short = 'o', long, value_delimiter = ',')]
    objects: Vec<String>,

    /// List of mask object numbers with closed contours (comma-separated)
    #[arg(short = 'm', long, value_delimiter = ',')]
    masks: Vec<String>,

    /// Test each point and allow contours to be split into separate objects
    #[arg(short = 's')]
    split_points: bool,

    /// Split contours will be marked as open
    #[arg(short = 'C')]
    cut_open: bool,

    /// Insert new objects just after their corresponding mask objects
    #[arg(short = 'i')]
    insert_in_place: bool,

    /// Delete original objects after splitting
    #[arg(short = 'd')]
    delete_old: bool,

    /// Keep contours/points outside all masks in a separate object
    #[arg(short = 'k')]
    keep_outside: bool,

    /// Only test the first point of each contour
    #[arg(short = 'f')]
    first_pt_only: bool,

    /// Give new objects new colors
    #[arg(short = 'c')]
    new_colors: bool,

    /// Use object numbers instead of names for new object names
    #[arg(short = 'n')]
    use_obj_nums: bool,

    /// Input model file
    input: String,

    /// Output model file
    output: String,
}

fn main() {
    let args = Args::parse();

    let obj_list = parse_int_list(&args.objects);
    let mask_list = parse_int_list(&args.masks);

    if obj_list.is_empty() {
        eprintln!("ERROR: You have not entered any objects to split (-o)");
        std::process::exit(3);
    }
    if mask_list.is_empty() {
        eprintln!("ERROR: You have not entered any mask objects (-m)");
        std::process::exit(3);
    }

    let mut model = read_model(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: Problem reading imod model {}: {}", args.input, e);
        std::process::exit(3);
    });

    let num_objects = model.objects.len();

    // Validate object indices
    let mut bad = 0;
    for &ob in obj_list.iter().chain(mask_list.iter()) {
        if ob < 1 || ob as usize > num_objects {
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!(
            "ERROR: {} bad object numbers. All should be between 1 and {}.",
            bad, num_objects
        );
        std::process::exit(3);
    }

    // Check for overlapping lists
    for &ob in &obj_list {
        for &mb in &mask_list {
            if ob == mb {
                println!(
                    "WARNING: Object {} appears as both a split and mask object.",
                    ob
                );
            }
        }
    }

    // Compute Z range
    let (min_z, max_z) = compute_z_range(&model);

    println!("MODEL SUMMARY:");
    println!("  min z:  {}", min_z);
    println!("  max z:  {}", max_z);
    println!("  # of objects to split:  {}", obj_list.len());
    println!("  # of mask objects:      {}", mask_list.len());
    println!();

    // Build mask contour lookup by Z level
    let mut mask_contours: Vec<Vec<(usize, &ImodContour)>> = vec![Vec::new(); (max_z + 1) as usize];
    for (mi, &mask_ob) in mask_list.iter().enumerate() {
        let obj = &model.objects[(mask_ob - 1) as usize];
        for cont in &obj.contours {
            if cont.points.is_empty() {
                continue;
            }
            let z = cont.points[0].z as i32;
            if z >= 0 && z <= max_z {
                mask_contours[z as usize].push((mi, cont));
            }
        }
    }

    // For each split object, create new objects (one per mask object, plus optional outside)
    let num_split = obj_list.len();
    let num_masks = mask_list.len();
    let mut new_objects: Vec<Vec<ImodObject>> = Vec::new();

    // Pre-generate color palette
    let palette: Vec<(f32, f32, f32)> = (0..num_split * (num_masks + 1))
        .map(|i| {
            let hue = (i as f32 * 137.508) % 360.0;
            hsv_to_rgb(hue, 0.7, 0.9)
        })
        .collect();

    for (si, &split_ob) in obj_list.iter().enumerate() {
        let split_obj = &model.objects[(split_ob - 1) as usize];
        let split_name = if args.use_obj_nums || split_obj.name.is_empty() {
            format!("object {}", split_ob)
        } else {
            split_obj.name.clone()
        };

        let mut per_mask: Vec<ImodObject> = Vec::new();

        for (mi, &mask_ob) in mask_list.iter().enumerate() {
            let mask_obj = &model.objects[(mask_ob - 1) as usize];
            let mask_name = if args.use_obj_nums || mask_obj.name.is_empty() {
                format!("object {}", mask_ob)
            } else {
                mask_obj.name.clone()
            };

            let mut new_obj = ImodObject {
                name: format!("{}... masked by '{}'", split_name, mask_name),
                flags: split_obj.flags,
                red: split_obj.red,
                green: split_obj.green,
                blue: split_obj.blue,
                ..ImodObject::default()
            };

            if args.new_colors {
                let (r, g, b) = palette[si * num_masks + mi];
                new_obj.red = r;
                new_obj.green = g;
                new_obj.blue = b;
            }

            per_mask.push(new_obj);
        }

        // Optional outside object
        if args.keep_outside {
            let outside_obj = ImodObject {
                name: format!("{}... masked OUTSIDE", split_name),
                flags: split_obj.flags,
                red: split_obj.red,
                green: split_obj.green,
                blue: split_obj.blue,
                ..ImodObject::default()
            };
            per_mask.push(outside_obj);
        }

        new_objects.push(per_mask);
    }

    // Sort contours into new objects
    let mut num_sorted = 0;
    for (si, &split_ob) in obj_list.iter().enumerate() {
        let split_obj = &model.objects[(split_ob - 1) as usize];

        for cont in &split_obj.contours {
            if cont.points.is_empty() {
                continue;
            }

            if args.split_points && !args.first_pt_only {
                // Test each point individually
                for pt in &cont.points {
                    let z = pt.z as i32;
                    let mask_idx = if z >= 0 && (z as usize) < mask_contours.len() {
                        find_containing_mask(pt, &mask_contours[z as usize])
                    } else {
                        None
                    };

                    match mask_idx {
                        Some(mi) => {
                            let mut new_cont = ImodContour::default();
                            new_cont.points.push(*pt);
                            if args.cut_open {
                                new_cont.flags |= 1 << 3; // ICONT_OPEN
                            }
                            new_objects[si][mi].contours.push(new_cont);
                        }
                        None => {
                            if args.keep_outside {
                                let outside_idx = new_objects[si].len() - 1;
                                let mut new_cont = ImodContour::default();
                                new_cont.points.push(*pt);
                                new_objects[si][outside_idx].contours.push(new_cont);
                            }
                        }
                    }
                }
            } else {
                // Test first point (or whole contour treated as unit)
                let test_pt = &cont.points[0];
                let z = test_pt.z as i32;
                let mask_idx = if z >= 0 && (z as usize) < mask_contours.len() {
                    find_containing_mask(test_pt, &mask_contours[z as usize])
                } else {
                    None
                };

                match mask_idx {
                    Some(mi) => {
                        new_objects[si][mi].contours.push(cont.clone());
                        num_sorted += 1;
                    }
                    None => {
                        if args.keep_outside {
                            let outside_idx = new_objects[si].len() - 1;
                            new_objects[si][outside_idx].contours.push(cont.clone());
                            num_sorted += 1;
                        }
                    }
                }
            }
        }
    }

    // Add new objects to model
    for per_mask in new_objects {
        for obj in per_mask {
            model.objects.push(obj);
        }
    }

    // Optionally delete original objects (mark them empty)
    if args.delete_old {
        for &ob in &obj_list {
            model.objects[(ob - 1) as usize].contours.clear();
        }
    }

    println!("Sorted {} contours into new objects.", num_sorted);

    write_model(&args.output, &model).unwrap_or_else(|e| {
        eprintln!("Error writing output model: {}", e);
        std::process::exit(1);
    });
}

/// Test if a point is inside any of the mask contours at its Z level.
/// Returns the mask index if found, None otherwise.
fn find_containing_mask(pt: &Point3f, contours: &[(usize, &ImodContour)]) -> Option<usize> {
    for &(mi, cont) in contours {
        if point_in_contour(pt.x, pt.y, &cont.points) {
            return Some(mi);
        }
    }
    None
}

/// Point-in-polygon test using ray casting algorithm.
fn point_in_contour(x: f32, y: f32, points: &[Point3f]) -> bool {
    let n = points.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (yi, yj) = (points[i].y, points[j].y);
        let (xi, xj) = (points[i].x, points[j].x);
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn compute_z_range(model: &ImodModel) -> (i32, i32) {
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                let z = pt.z as i32;
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }
        }
    }
    if min_z > max_z {
        (0, 0)
    } else {
        (min_z, max_z)
    }
}

/// Parse a list of integers from strings like "1", "3-5", "2,4-6".
fn parse_int_list(items: &[String]) -> Vec<i32> {
    let mut result = Vec::new();
    for item in items {
        for part in item.split(',') {
            let part = part.trim();
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(start), Ok(end)) = (a.trim().parse::<i32>(), b.trim().parse::<i32>()) {
                    for i in start..=end {
                        result.push(i);
                    }
                }
            } else if let Ok(v) = part.parse::<i32>() {
                result.push(v);
            }
        }
    }
    result
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (r + m, g + m, b + m)
}
