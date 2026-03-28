use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{write_model, ImodContour, ImodModel, ImodObject};

/// Convert a WIMP model file to IMOD format.
///
/// Reads a VMS WIMP format model and writes it as an IMOD binary model.
/// WIMP objects are mapped based on their display color index, with
/// standard colors assigned for indices 247-255 (yellow, olive brown,
/// orange, red, green, blue, yellow, magenta, cyan).
#[derive(Parser)]
#[command(name = "wmod2imod", version, about)]
struct Args {
    /// Input WIMP model file
    input: String,

    /// Output IMOD model file
    output: String,

    /// X scale factor
    #[arg(short = 'x', default_value_t = 1.0)]
    xscale: f32,

    /// Y scale factor
    #[arg(short = 'y', default_value_t = 1.0)]
    yscale: f32,

    /// Z scale factor
    #[arg(short = 'z', default_value_t = 1.0)]
    zscale: f32,
}

/// WIMP display colors for indices 247-255
const WMOD_COLORS: [[f32; 3]; 9] = [
    [0.90, 0.82, 0.37], // 247: Dim Yellow
    [0.54, 0.51, 0.01], // 248: Olive Brown
    [0.94, 0.49, 0.0],  // 249: Orange
    [1.00, 0.0, 0.0],   // 250: Red
    [0.0, 1.0, 0.0],    // 251: Green
    [0.0, 0.0, 1.0],    // 252: Blue
    [1.0, 1.0, 0.0],    // 253: Yellow
    [1.0, 0.0, 1.0],    // 254: Magenta
    [0.0, 1.0, 1.0],    // 255: Cyan
];

const MAXOBJ: usize = 256;

/// IMOD object flag for open contours
const IMOD_OBJFLAG_OPEN: u32 = 1 << 3;

/// A parsed contour from WIMP format
struct WimpContour {
    display_switch: usize,
    points: Vec<Point3f>,
}

/// Read a single non-empty line from reader
fn read_line(reader: &mut impl BufRead) -> Option<String> {
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => Some(line),
        Err(_) => None,
    }
}

fn parse_wimp(reader: &mut impl BufRead) -> Vec<WimpContour> {
    let mut contours = Vec::new();
    let cont_string = "Object #:";

    loop {
        let line = match read_line(reader) {
            Some(l) => l,
            None => break,
        };

        // Look for "Object #:" marker
        if !line.contains(cont_string) {
            continue;
        }

        // Next line: "... ... ... <points>"
        let points_line = match read_line(reader) {
            Some(l) => l,
            None => break,
        };
        let num_points: usize = points_line
            .split_whitespace()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Next line: "... ... <display_switch>"
        let display_line = match read_line(reader) {
            Some(l) => l,
            None => break,
        };
        let display_switch: usize = display_line
            .split_whitespace()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Skip one line
        let _ = read_line(reader);

        // Read points
        let mut pts = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let pt_line = match read_line(reader) {
                Some(l) => l,
                None => break,
            };
            let parts: Vec<&str> = pt_line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                pts.push(Point3f { x, y, z });
            }
        }

        contours.push(WimpContour {
            display_switch,
            points: pts,
        });
    }

    contours
}

fn main() {
    let args = Args::parse();

    let fin = File::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: wmod2imod - Could not open {}: {}", args.input, e);
        process::exit(3);
    });
    let mut reader = BufReader::new(fin);

    // First pass: determine which display indices are used
    let contours = parse_wimp(&mut reader);

    let mut obj_lookup: [i32; MAXOBJ] = [-1; MAXOBJ];
    for c in &contours {
        if c.display_switch < MAXOBJ {
            obj_lookup[c.display_switch] = 0; // mark as used
        }
    }

    // Create objects for each used display index
    let mut model = ImodModel::default();
    let mut nobj = 0i32;
    for i in 0..MAXOBJ {
        if obj_lookup[i] >= 0 {
            obj_lookup[i] = nobj;
            let mut obj = ImodObject::default();
            if i >= 247 {
                obj.red = WMOD_COLORS[i - 247][0];
                obj.green = WMOD_COLORS[i - 247][1];
                obj.blue = WMOD_COLORS[i - 247][2];
            } else {
                let gray = i as f32 / 255.0;
                obj.red = gray;
                obj.green = gray;
                obj.blue = gray;
            }
            obj.name = format!("Wimp no. {}", i);
            model.objects.push(obj);
            nobj += 1;
        }
    }

    // Second pass: add contours to objects
    for c in &contours {
        if c.display_switch >= MAXOBJ {
            continue;
        }
        let obj_idx = obj_lookup[c.display_switch];
        if obj_idx < 0 {
            continue;
        }
        let obj = &mut model.objects[obj_idx as usize];
        let cont = ImodContour {
            points: c.points.clone(),
            ..Default::default()
        };
        obj.contours.push(cont);
    }

    // Set open flag for objects whose first contour spans multiple Z
    for obj in &mut model.objects {
        if !obj.contours.is_empty() {
            let first = &obj.contours[0];
            if first.points.len() < 2 {
                obj.flags |= IMOD_OBJFLAG_OPEN;
            } else if first.points[0].z != first.points[1].z {
                obj.flags |= IMOD_OBJFLAG_OPEN;
            }
        }
    }

    // Remove empty objects
    model.objects.retain(|obj| !obj.contours.is_empty());

    // Apply scale
    model.scale.z = args.zscale;

    write_model(&args.output, &model).unwrap_or_else(|e| {
        eprintln!("ERROR: wmod2imod - Writing model {}: {}", args.output, e);
        process::exit(3);
    });
}
