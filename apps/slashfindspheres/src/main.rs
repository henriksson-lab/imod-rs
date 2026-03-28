use clap::Parser;
use imod_model::{read_model, write_model, ImodModel, ImodObject};
use imod_mrc::MrcReader;

/// Find spheres matching input sphere templates in an MRC image.
///
/// Given an IMOD model with scattered point objects as input templates,
/// scans the image to find additional locations that match the radial
/// intensity profile of the input spheres using cross-correlation of
/// concentric ring histograms.
///
/// Used for PEET (Particle Estimation for Electron Tomography).
#[derive(Parser)]
#[command(name = "slashfindspheres", about = "Find spheres for PEET")]
struct Args {
    /// Input MRC image file
    #[arg(short = 'I', long)]
    image: String,

    /// Input IMOD model containing template spheres
    #[arg(short = 'M', long)]
    model: String,

    /// Output IMOD model file
    #[arg(short = 'O', long)]
    output: String,

    /// List of object numbers containing input spheres (comma-separated)
    #[arg(short = 'o', long, value_delimiter = ',')]
    objects: Vec<i32>,

    /// List of extra object numbers whose spheres should not overlap output (comma-separated)
    #[arg(short = 'e', long, value_delimiter = ',')]
    extra_objects: Option<Vec<i32>>,

    /// XY gutter size in pixels around each template
    #[arg(long, default_value = "4.0")]
    gutter_xy: f32,

    /// Z gutter size in pixels
    #[arg(long)]
    gutter_z: Option<f32>,

    /// Cross-correlation cutoff (0-1, where 1 = perfect match)
    #[arg(short = 'c', long)]
    cutoff: Option<f32>,

    /// Pixel value range for scaling correlation
    #[arg(long)]
    val_range: Option<f32>,

    /// Maximum span for vesicle search
    #[arg(long, default_value = "30")]
    max_span: i32,

    /// Maximum number of output spheres
    #[arg(long, default_value = "10000")]
    max_spheres: i32,

    /// Analyze in 2D only (ignore Z)
    #[arg(short = '2', long)]
    ignore_z: bool,

    /// Allow output spheres to overlap input spheres
    #[arg(long)]
    allow_dup: bool,

    /// Number of new objects to split output into (by confidence)
    #[arg(long, default_value = "5")]
    split_level: i32,

    /// Use same colors as input objects
    #[arg(long)]
    orig_color: bool,

    /// One contour with all points per object instead of one point per contour
    #[arg(long)]
    multi_pts: bool,

    /// Z buffer size (max slices in memory)
    #[arg(long, default_value = "30")]
    z_buffer: i32,

    /// Print level (0=none, 1=basic, 5=verbose)
    #[arg(short = 'v', long, default_value = "2")]
    print_level: i32,

    /// Write output images: 1 = filtered, 2 = also cutoff values
    #[arg(long, default_value = "0")]
    write_images: i32,

    /// Filter specification (e.g., "median,5" or "sobel")
    #[arg(short = 'f', long)]
    filter: Option<String>,

    /// Test radius for early rejection
    #[arg(long, default_value = "3.0")]
    test_radius: f32,
}

fn main() {
    let args = Args::parse();

    // Read input image
    let reader = MrcReader::open(&args.image).unwrap_or_else(|e| {
        eprintln!("Error opening image file {}: {}", args.image, e);
        std::process::exit(1);
    });
    let header = reader.header();
    let (nx, ny, nz) = (header.nx, header.ny, header.nz);

    // Read input model with template spheres
    let model = read_model(&args.model).unwrap_or_else(|e| {
        eprintln!("Error reading model {}: {}", args.model, e);
        std::process::exit(1);
    });

    // Validate object list
    if args.objects.is_empty() {
        eprintln!("Error: No input object numbers specified (-o)");
        std::process::exit(1);
    }

    for &ob in &args.objects {
        if ob < 1 || ob as usize > model.objects.len() {
            eprintln!(
                "Error: Object number {} is out of range (model has {} objects)",
                ob,
                model.objects.len()
            );
            std::process::exit(1);
        }
    }

    // Collect input spheres from specified objects
    let mut input_spheres: Vec<(f32, f32, f32, f32, usize)> = Vec::new(); // x, y, z, r, obj_idx
    let zscale = model.scale.z;

    for &ob in &args.objects {
        let obj = &model.objects[(ob - 1) as usize];
        for cont in &obj.contours {
            for (p, pt) in cont.points.iter().enumerate() {
                let radius = if let Some(ref sizes) = cont.sizes {
                    if p < sizes.len() {
                        sizes[p]
                    } else {
                        obj.pdrawsize as f32
                    }
                } else {
                    obj.pdrawsize as f32
                };
                if radius > 0.0 {
                    input_spheres.push((pt.x, pt.y, pt.z, radius, (ob - 1) as usize));
                }
            }
        }
    }

    if args.print_level >= 1 {
        println!("Image size: {} x {} x {}", nx, ny, nz);
        println!("Input spheres: {}", input_spheres.len());
        println!("Z scale: {}", zscale);
    }

    // TODO: Implement the actual sphere-finding algorithm:
    // 1. Build concentric ring histograms from each input sphere
    // 2. Compute average template histogram
    // 3. Scan image voxels, apply filters, compute cross-correlation
    // 4. Find peaks above cutoff
    // 5. Remove overlapping results
    // 6. Sort by confidence and split into output objects

    eprintln!(
        "Note: sphere-finding algorithm not yet implemented. \
         Writing model with {} input spheres only.",
        input_spheres.len()
    );

    // Create output model
    let mut out_model = ImodModel::default();
    out_model.xmax = nx as i32;
    out_model.ymax = ny as i32;
    out_model.zmax = nz as i32;
    out_model.scale = model.scale;

    let split = args.split_level.max(1) as usize;
    let colors: Vec<(f32, f32, f32)> = (0..split)
        .map(|i| {
            let t = i as f32 / split.max(1) as f32;
            (1.0 - t, t, 0.0) // red to green gradient
        })
        .collect();

    for i in 0..split {
        let mut obj = ImodObject::default();
        obj.name = format!("Found spheres (level {})", i + 1);
        obj.flags |= 1 << 9; // scattered point
        let (r, g, b) = colors[i];
        obj.red = r;
        obj.green = g;
        obj.blue = b;
        out_model.objects.push(obj);
    }

    write_model(&args.output, &out_model).unwrap_or_else(|e| {
        eprintln!("Error writing output model: {}", e);
        std::process::exit(1);
    });

    if args.print_level >= 1 {
        println!("Wrote output model to {}", args.output);
    }
}
