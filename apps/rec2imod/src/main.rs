use std::fs::File;
use std::io::{BufReader, Read};
use std::process;

use clap::Parser;
use imod_core::Point3f;
use imod_model::{ImodContour, ImodModel, ImodObject, write_model};

/// Convert HVEM3D reconstruction files to IMOD model files.
///
/// Reads an HVEM3D .rec file from an IBM PC and writes an IMOD .mod file.
/// Contours are grouped into objects by their HVEM type code. Coordinates
/// are shifted so the minimum is at the origin and optionally scaled.
#[derive(Parser)]
#[command(name = "rec2imod", version, about)]
struct Args {
    /// Magnification in kx.
    #[arg(short = 'm', long)]
    magnification: Option<f32>,

    /// Thickness of each section in um.
    #[arg(short = 't', long)]
    thickness: Option<f32>,

    /// Pixel size in mm (default 0.1).
    #[arg(short = 'p', long, default_value_t = 0.1)]
    pixel_size: f32,

    /// Z-scale (used only if no mag and thickness entered).
    #[arg(short = 'z', long)]
    zscale: Option<f32>,

    /// Scaling factor to reduce dimensions of data.
    #[arg(short = 's', long, default_value_t = 1.0)]
    scaling: f32,

    /// Maximum contour number to read.
    #[arg(short = 'c', long)]
    max_contour: Option<usize>,

    /// Input HVEM3D filename.
    input: String,

    /// Output IMOD model filename.
    output: String,
}

/// Minimal HVEM3D header data we need.
struct HvemHead {
    lastdir: usize,
    mag: f32,
    secthick: f32,
    unitsize: f32,
}

/// Minimal HVEM3D contour data.
struct HvemContour {
    npts: usize,
    type_code: u8,
    deleted: bool,
    points: Vec<[f32; 3]>,
}

/// Read a big-endian i16 from the reader.
fn read_i16_be(r: &mut impl Read) -> std::io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_be_bytes(buf))
}

/// Read a big-endian f32 from the reader.
fn read_f32_be(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_be_bytes(buf))
}

/// Read the HVEM3D header. The format is legacy IBM PC big-endian.
fn read_hvem_head(reader: &mut BufReader<File>) -> std::io::Result<HvemHead> {
    // The HVEM3D header contains fields at known offsets.
    // We read the essential fields: lastdir (number of contours),
    // magnification, section thickness.
    let mut buf = [0u8; 512];
    reader.read_exact(&mut buf)?;

    // lastdir at offset 0 (2 bytes, big-endian)
    let lastdir = i16::from_be_bytes([buf[0], buf[1]]) as usize;
    // magnification at offset 8 (4 bytes float)
    let mag = f32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
    // section thickness at offset 12
    let secthick = f32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);

    Ok(HvemHead {
        lastdir,
        mag,
        secthick,
        unitsize: 0.1,
    })
}

/// Read a single HVEM3D contour from the file.
fn read_hvem_contour(reader: &mut BufReader<File>, _index: usize) -> std::io::Result<Option<HvemContour>> {
    // Each contour record has a small header then point data.
    // Read the contour header: type, deleted flag, number of points.
    let type_code = {
        let mut b = [0u8; 1];
        if reader.read_exact(&mut b).is_err() {
            return Ok(None);
        }
        b[0]
    };
    let del_byte = {
        let mut b = [0u8; 1];
        reader.read_exact(&mut b)?;
        b[0]
    };
    let npts = read_i16_be(reader)? as usize;

    if npts == 0 {
        return Ok(Some(HvemContour {
            npts: 0,
            type_code,
            deleted: del_byte != 0,
            points: Vec::new(),
        }));
    }

    let mut points = Vec::with_capacity(npts);
    for _ in 0..npts {
        let x = read_f32_be(reader)?;
        let y = read_f32_be(reader)?;
        let z = read_f32_be(reader)?;
        points.push([x, y, z]);
    }

    Ok(Some(HvemContour {
        npts,
        type_code,
        deleted: del_byte != 0,
        points,
    }))
}

fn main() {
    let args = Args::parse();

    let fin = File::open(&args.input).unwrap_or_else(|e| {
        eprintln!("ERROR: rec2imod - could not open {}: {}", args.input, e);
        process::exit(1);
    });
    let mut reader = BufReader::new(fin);

    let mut head = read_hvem_head(&mut reader).unwrap_or_else(|e| {
        eprintln!("ERROR: rec2imod - error reading header: {}", e);
        process::exit(1);
    });
    head.unitsize = args.pixel_size;

    if let Some(mag) = args.magnification {
        head.mag = mag;
    }
    if let Some(thick) = args.thickness {
        head.secthick = thick;
    }

    let max_cont = args.max_contour.unwrap_or(head.lastdir);
    let num_contours = max_cont.min(head.lastdir);

    // Map HVEM type codes to object indices
    let mut type_to_obj: [i16; 256] = [-1; 256];
    let mut model = ImodModel::default();

    for i in 0..num_contours {
        eprint!("\rcontour {} of {}          \r", i + 1, num_contours);

        let hvem_cont = match read_hvem_contour(&mut reader, i) {
            Ok(Some(c)) => c,
            Ok(None) => continue,
            Err(_) => continue,
        };

        if hvem_cont.deleted || hvem_cont.npts == 0 {
            continue;
        }

        let tc = hvem_cont.type_code as usize;
        if type_to_obj[tc] < 0 {
            let mut obj = ImodObject::default();
            obj.name = format!("Type {} data.", i + 1);
            model.objects.push(obj);
            type_to_obj[tc] = (model.objects.len() - 1) as i16;
        }

        let obj_idx = type_to_obj[tc] as usize;
        let points: Vec<Point3f> = hvem_cont
            .points
            .iter()
            .map(|p| Point3f {
                x: p[0],
                y: p[1],
                z: p[2],
            })
            .collect();

        let cont = ImodContour {
            points,
            surf: 0,
            ..Default::default()
        };
        model.objects[obj_idx].contours.push(cont);
    }
    eprintln!("\ndone");

    // Set units to micrometers
    model.units = -6; // IMOD_UNIT_UM

    // Compute pixel size and z-scale from magnification and section thickness
    if head.mag != 0.0 {
        model.pixel_size = args.scaling * head.unitsize / head.mag;
    }
    if model.pixel_size != 0.0 && head.secthick != 0.0 {
        model.scale.z = head.secthick / model.pixel_size;
    }
    if model.scale.z == 0.0 {
        model.scale.z = args.zscale.unwrap_or(1.0);
    }

    // Shift data to origin and apply scaling
    let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                min_x = min_x.min(pt.x);
                min_y = min_y.min(pt.y);
                min_z = min_z.min(pt.z);
            }
        }
    }

    for obj in &mut model.objects {
        for cont in &mut obj.contours {
            for pt in &mut cont.points {
                pt.x = args.scaling * (pt.x - min_x);
                pt.y = args.scaling * (pt.y - min_y);
                pt.z -= min_z;
            }
        }
    }

    // Find maximum extents
    let (mut max_x, mut max_y, mut max_z) = (0.0_f32, 0.0_f32, 0.0_f32);
    for obj in &model.objects {
        for cont in &obj.contours {
            for pt in &cont.points {
                max_x = max_x.max(pt.x);
                max_y = max_y.max(pt.y);
                max_z = max_z.max(pt.z);
            }
        }
    }

    model.xmax = max_x as i32;
    model.ymax = max_y as i32;
    model.zmax = max_z as i32 + 1;

    println!(
        "Maximum X, Y and Z are {}, {}, and {}",
        model.xmax, model.ymax, model.zmax
    );

    if args.scaling == 1.0 && (max_x > 1500.0 || max_y > 1500.0) {
        let facx = 1280.0 / max_x;
        let facy = 1024.0 / max_y;
        let fac = facx.min(facy);
        println!(
            "For better results with imodmesh, it is recommended that\n\
             you rerun this program with a scaling (-s) of about {:.2},\n\
             then run reducecont with a tolerance of about 0.2",
            fac
        );
    }

    if let Err(e) = write_model(&args.output, &model) {
        eprintln!("ERROR: rec2imod - error writing model: {}", e);
        process::exit(1);
    }
}
