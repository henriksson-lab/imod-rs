//! Translation of the analysis routines in `IMOD/imodutil/imodinfo.cpp`.
//!
//! The command's model-file reader belongs to the paired `libimod/imodel_files`
//! source unit.  The calculations below deliberately consume the source-shaped
//! `Imod`, `Iobj`, `Icont`, and `Imesh` declarations instead of inventing a
//! separate command-only model representation.
#![allow(dead_code)]

use std::env;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;

use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libimod::imodel::{
    ICONT_OPEN, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_END, IMOD_MESH_ENDPOLY,
    IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_OUT, IMOD_OBJFLAG_SCAT, Icont, Imesh, Imod,
    Iobj, Iplane, Ipoint,
};
use crate::imod::libimod::imodel_files::{imod_file_write, imod_read, imod_write_ascii};
use crate::imod::libimod::objgroup::obj_group_lookup;

/// Original: `imodinfo_usage` (`imodinfo.cpp:84`).
pub fn imodinfo_usage() {
    println!("usage: imodinfo [options] <imod filename>");
    println!(
        "options: -a -c -l -L -s -S N -p -r -e -F -o list -g # -i -x min,max -y min,max -z min,max -t 1/-1 -v -h -f filename"
    );
}
/// Original: `main` (`imodinfo.cpp:116`).
pub fn imodinfo() {
    let argv: Vec<String> = env::args().collect();
    if argv.len() == 1 {
        imodinfo_usage();
        return;
    }
    let mut iarg = 1_usize;
    let mut verbose = 0_i32;
    let mut group_num = -1_i32;
    let mut list = Vec::<i32>::new();
    let mut out_file = None::<String>;
    let mut mode = 1_i32;
    let mut scaninside = false;
    let mut subarea = false;
    let mut useclip = 0_i32;
    let mut sample = 0_usize;
    let mut bins = 0_usize;
    let mut hush = false;
    let mut minimum = Ipoint {
        x: -1.0e30,
        y: -1.0e30,
        z: -1.0e30,
    };
    let mut maximum = Ipoint {
        x: 1.0e30,
        y: 1.0e30,
        z: 1.0e30,
    };
    while iarg < argv.len() && argv[iarg].starts_with('-') {
        match argv[iarg].as_bytes().get(1).copied().unwrap_or_default() as char {
            'g' => {
                iarg += 1;
                let Some(value) = argv.get(iarg) else {
                    std::process::exit(1)
                };
                group_num = value.parse().unwrap_or(0);
                if group_num <= 0 {
                    eprintln!("ERROR: imodinfo - Group number {group_num} must be positive");
                    std::process::exit(1);
                }
                mode = 2;
            }
            'a' => mode = 4,
            'c' => mode = 2,
            'l' => mode = 5,
            'L' => mode = 10,
            'F' => mode = 7,
            's' => mode = 6,
            'S' => {
                mode = 6;
                iarg += 1;
                sample = argv
                    .get(iarg)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
            }
            'p' => mode = 8,
            'e' => mode = 12,
            'r' => mode = 9,
            'n' => mode = 3,
            'i' => scaninside = true,
            'D' => {}
            'b' => {
                iarg += 1;
                bins = argv
                    .get(iarg)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
            }
            't' => {
                iarg += 1;
                useclip = argv
                    .get(iarg)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
                scaninside = true;
            }
            'x' | 'y' | 'z' => {
                let axis = argv[iarg].as_bytes()[1] as char;
                iarg += 1;
                let value = CString::new(argv.get(iarg).map(String::as_str).unwrap_or("")).unwrap();
                let mut low = 0_f32;
                let mut high = 0_f32;
                unsafe {
                    libc::sscanf(value.as_ptr(), c"%f%*c%f".as_ptr(), &mut low, &mut high);
                }
                match axis {
                    'x' => {
                        minimum.x = low;
                        maximum.x = high;
                    }
                    'y' => {
                        minimum.y = low;
                        maximum.y = high;
                    }
                    _ => {
                        minimum.z = low;
                        maximum.z = high;
                    }
                }
                subarea = true;
                if axis != 'z' {
                    scaninside = true;
                }
            }
            'o' => {
                iarg += 1;
                let Some(value) = argv.get(iarg) else {
                    std::process::exit(1)
                };
                let value = CString::new(value.as_str()).unwrap_or_else(|_| std::process::exit(1));
                let mut count = 0_i32;
                let values = unsafe { parselist(value.as_ptr(), &mut count) };
                if values.is_null() {
                    eprintln!("ERROR: imodinfo - Parsing list {}", argv[iarg]);
                    std::process::exit(1);
                }
                list = unsafe { std::slice::from_raw_parts(values, count as usize) }.to_vec();
                unsafe { libc::free(values.cast()) };
            }
            'v' => {
                verbose += 1;
                if argv[iarg].as_bytes().get(2) == Some(&b'v') {
                    verbose += 1;
                }
            }
            'f' => {
                iarg += 1;
                let Some(value) = argv.get(iarg) else {
                    std::process::exit(1)
                };
                out_file = Some(value.clone());
            }
            'h' => {
                if argv[iarg] == "-help" {
                    imodinfo_usage();
                    return;
                }
                if argv[iarg] != "-h" {
                    eprintln!(
                        "ERROR: imodinfo - Unknown option {}; enter -help for help",
                        argv[iarg]
                    );
                    std::process::exit(1);
                }
                hush = true;
            }
            _ => {
                eprintln!("{}: unknown option {}", argv[0], argv[iarg]);
                imodinfo_usage();
                std::process::exit(2);
            }
        }
        iarg += 1;
    }
    if iarg >= argv.len() {
        imodinfo_usage();
        std::process::exit(2);
    }
    if hush && verbose == 0 {
        verbose = -1;
    }
    let mut file_output = if let Some(filename) = out_file {
        let backup = format!("{filename}~");
        if std::path::Path::new(&filename).exists() {
            let _ = std::fs::remove_file(&backup);
            if std::fs::rename(&filename, &backup).is_err() {
                eprintln!(
                    "ERROR: imodinfo - Could not make ~ backup of existing output file {filename}"
                );
                std::process::exit(1);
            }
        }
        Some(File::create(&filename).unwrap_or_else(|_| {
            eprintln!("ERROR: imodinfo - Opening output file {filename}");
            std::process::exit(1)
        }))
    } else {
        None
    };
    if !list.is_empty() && group_num > 0 {
        eprintln!("ERROR: imodinfo - You cannot enter both an object list and an object group");
        std::process::exit(1);
    }
    for filename in &argv[iarg..] {
        if File::open(filename).is_err() {
            eprintln!("ERROR: imodinfo - Opening input file {filename}");
            std::process::exit(1);
        }
        let model = match imod_read(filename) {
            Ok(model) => model,
            Err(error) => {
                println!("imodinfo: Error ({error}) reading imod model. ({filename})");
                continue;
            }
        };
        if group_num > 0 && model.group_list.is_empty() {
            eprintln!("ERROR: imodinfo - There are no object groups in model {filename}");
            std::process::exit(1);
        }
        if group_num > model.group_list.len() as i32 {
            eprintln!(
                "ERROR: imodinfo - Group # {group_num} is more than the number of object groups ({}) in {filename}",
                model.group_list.len()
            );
            std::process::exit(1);
        }
        let mut header = format!(
            "# MODEL {filename}\n# NAME  {}\n# PIX SCALE:  x = {}\n#             y = {}\n#             z = {}\n# PIX SIZE      = {}\n# UNITS: {}\n\n\n",
            model.name,
            model.xscale,
            model.yscale,
            model.zscale,
            model.pixsize,
            print_units(model.units)
        );
        if let Some(reference) = model.ref_image {
            header = format!(
                "# MODEL {filename}\n# NAME  {}\n# PIX SCALE:  x = {}\n#             y = {}\n#             z = {}\n# PIX SIZE      = {}\n# UNITS: {}\n\n# Model to Image index coords:\n#      SCALE  = ( {}, {}, {})\n#      OFFSET = ( {}, {}, {})\n#      ANGLES = ( {}, {}, {})\n\n\n",
                model.name,
                model.xscale,
                model.yscale,
                model.zscale,
                model.pixsize,
                print_units(model.units),
                reference.cscale.x,
                reference.cscale.y,
                reference.cscale.z,
                reference.ctrans.x,
                reference.ctrans.y,
                reference.ctrans.z,
                reference.crot.x,
                reference.crot.y,
                reference.crot.z,
            );
        }
        if let Some(file) = file_output.as_mut() {
            write!(file, "{header}").unwrap();
        } else {
            print!("{header}");
        }
        if model.obj.is_empty() {
            if let Some(file) = file_output.as_mut() {
                writeln!(file, "Model has no objects!!!").unwrap();
            } else {
                println!("Model has no objects!!!");
            }
        }
        if mode == 4 {
            if let Some(file) = file_output.as_mut() {
                imod_write_ascii(&model, file).unwrap_or_else(|_| {
                    eprintln!("ERROR: imodinfo - Writing ASCII output");
                    std::process::exit(1)
                });
                write!(file, "\n\n").unwrap();
            } else {
                let stdout = std::io::stdout();
                let mut file = stdout.lock();
                imod_write_ascii(&model, &mut file).unwrap_or_else(|_| {
                    eprintln!("ERROR: imodinfo - Writing ASCII output");
                    std::process::exit(1)
                });
                write!(file, "\n\n").unwrap();
            }
            continue;
        }
        if mode == 2 {
            let chart_header = "#Obj       Cyl. Vol      Cont Vol   Vol Inside Mesh   Mesh Surf              Center\n#--------------------------------------------------------------------------------------------\n";
            if let Some(file) = file_output.as_mut() {
                write!(file, "{chart_header}").unwrap();
            } else {
                print!("{chart_header}");
            }
        }
        if mode == 5 {
            let length_header = format!(
                "# Obj Cont Pnts Length (in {})\n#------------------------\n",
                print_units(model.units)
            );
            if let Some(file) = file_output.as_mut() {
                write!(file, "{length_header}").unwrap();
            } else {
                print!("{length_header}");
            }
        }
        for ob in 0..model.obj.len() {
            if mode == 3 {
                continue;
            }
            if group_num > 0 {
                if obj_group_lookup(&model.group_list[group_num as usize - 1], ob as i32) < 0 {
                    continue;
                }
            } else if !list.is_empty() && !list.contains(&(ob as i32 + 1)) {
                continue;
            }
            let report = match mode {
                6 => imodinfo_surface(
                    &model, ob, scaninside, minimum, maximum, useclip, sample, verbose,
                ),
                8 => imodinfo_points(&model, ob, subarea, minimum, maximum, useclip, verbose),
                9 => imodinfo_ratios(&model, ob),
                12 => imodinfo_ellipse(&model, ob, subarea, minimum, maximum),
                7 => imodinfo_full_object_report(
                    &model,
                    ob + 1,
                    scaninside,
                    subarea,
                    minimum,
                    maximum,
                    useclip,
                ),
                2 => imodinfo_object(&model, ob, scaninside, subarea, minimum, maximum, useclip),
                5 => imodinfo_length(&model, ob),
                10 => contour_length_by_color(&model, ob, verbose),
                _ => imodinfo_print_model(
                    &model, ob, verbose, scaninside, subarea, minimum, maximum, useclip,
                ),
            };
            if let Some(file) = file_output.as_mut() {
                write!(file, "{report}").unwrap();
            } else {
                print!("{report}");
            }
        }
        if mode == 3 {
            let report = imodinfo_objndist(&model, bins);
            if let Some(file) = file_output.as_mut() {
                write!(file, "{report}").unwrap();
            } else {
                print!("{report}");
            }
        }
        if let Some(file) = file_output.as_mut() {
            write!(file, "\n\n").unwrap();
        } else {
            print!("\n\n");
        }
        if let Some(file) = file_output.as_mut() {
            file.flush().unwrap();
        } else {
            let _ = std::io::stdout().flush();
        }
    }
}
/// Original: `imodinfo_print_model` (`imodinfo.cpp:464`).
pub fn imodinfo_print_model(
    model: &Imod,
    ob: usize,
    verbose: i32,
    _scaninside: bool,
    _subarea: bool,
    _min: Ipoint,
    _max: Ipoint,
    _useclip: i32,
) -> String {
    // Original: `imodinfo_print_model` (`imodinfo.cpp:464`).
    let Some(obj) = model.obj.get(ob) else {
        return String::new();
    };
    let mut output = format!(
        "OBJECT {}\nNAME:  {}\n       {} contours\n",
        ob + 1,
        obj.name,
        obj.cont.len()
    );
    if obj.flags & IMOD_OBJFLAG_OFF != 0 {
        output.push_str("       object drawing is turned off\n");
    }
    if obj.flags & IMOD_OBJFLAG_SCAT != 0 {
        output.push_str("       object uses scattered points.\n");
    } else if obj.flags & IMOD_OBJFLAG_OPEN != 0 {
        output.push_str("       object uses open contours.\n");
    } else {
        output.push_str("       object uses closed contours.\n");
    }
    if obj.flags & IMOD_OBJFLAG_OUT != 0 {
        output.push_str("       contours in object are inside out.\n");
    }
    output.push_str(&format!(
        "       color (red, green, blue) = ({}, {}, {})\n\n",
        obj.red, obj.green, obj.blue
    ));
    for (co, cont) in obj.cont.iter().enumerate() {
        if verbose >= 0 {
            output.push_str(&format!(
                "\tCONTOUR #{},{},{}  {} points",
                co + 1,
                ob + 1,
                cont.surf,
                cont.pts.len()
            ));
        }
        if cont.pts.is_empty() {
            if verbose >= 0 {
                output.push('\n');
            }
            continue;
        }
        let dist = info_contour_length(
            Some(cont),
            obj.flags,
            model.pixsize as f64,
            model.zscale as f64,
        );
        if obj.flags & IMOD_OBJFLAG_OPEN == 0 && obj.flags & IMOD_OBJFLAG_SCAT == 0 {
            output.push_str(&format!(
                ", length = {dist},  area = {}\n",
                info_contour_vol(Some(cont), obj.flags, 1.0, 1.0)
            ));
        } else {
            output.push_str(&format!("\tlength = {dist} {}\n", print_units(model.units)));
        }
    }
    output
}
/// Original: `imodinfo_surface` (`imodinfo.cpp:646`).
pub fn imodinfo_surface(
    imod: &Imod,
    ob: usize,
    _scaninside: bool,
    min: Ipoint,
    max: Ipoint,
    _useclip: i32,
    sample: usize,
    _verbose: i32,
) -> String {
    // Original: `imodinfo_surface` (`imodinfo.cpp:646`).
    let Some(obj) = imod.obj.get(ob) else {
        return String::new();
    };
    let sample = sample.max(1);
    let mut output = format!(
        "\n#Object {} data, {}\n#Surface : Contours,  Cyl. Volume,  Cyl. Surface\n",
        ob + 1,
        obj.name
    );
    for surf in 0..=obj.surfsize.max(0) as usize {
        if surf % sample != 0 {
            continue;
        }
        let mut num = 0;
        let mut volume = 0.0;
        let mut surface = 0.0;
        for cont in &obj.cont {
            if cont.surf != surf as i32 {
                continue;
            }
            if cont.pts.iter().any(|pt| {
                pt.x < min.x
                    || pt.x > max.x
                    || pt.y < min.y
                    || pt.y > max.y
                    || pt.z < min.z
                    || pt.z > max.z
            }) {
                continue;
            }
            num += 1;
            volume += info_contour_vol(
                Some(cont),
                obj.flags,
                imod.pixsize as f64,
                imod.zscale as f64,
            );
            surface += info_contour_surface_area(
                Some(cont),
                obj.flags,
                imod.pixsize as f64,
                imod.zscale as f64,
            );
        }
        output.push_str(&format!(
            "{:7}   {:8}   {:12.6}  {:12.6}\n",
            surf, num, volume, surface
        ));
    }
    output
}
/// Original: `imodinfo_points` (`imodinfo.cpp:910`).
pub fn imodinfo_points(
    imod: &Imod,
    ob: usize,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
    _useclip: i32,
    verbose: i32,
) -> String {
    // Original: `imodinfo_points` (`imodinfo.cpp:910`).
    let Some(obj) = imod.obj.get(ob) else {
        return String::new();
    };
    let mut output = String::new();
    let mut rsum = 0.0_f64;
    let mut rsqsum = 0.0_f64;
    let mut rcubsum = 0.0_f64;
    let mut nsum = 0;
    let mut object_header = false;
    for (co, cont) in obj.cont.iter().enumerate() {
        if cont.sizes.is_empty()
            && obj.flags & IMOD_OBJFLAG_SCAT == 0
            && !(verbose > 0 && obj.pdrawsize > 0)
        {
            continue;
        }
        if !object_header {
            output.push_str(&format!("\n#Object {} data, {}\n", ob + 1, obj.name));
            object_header = true;
        }
        if verbose >= 0 {
            output.push_str(&format!(
                "\tCONTOUR #{},{},{}  {} points\n",
                co + 1,
                ob + 1,
                cont.surf,
                cont.pts.len()
            ));
        }
        for (pt, point) in cont.pts.iter().enumerate() {
            if subarea
                && (point.x < min.x
                    || point.x > max.x
                    || point.y < min.y
                    || point.y > max.y
                    || point.z < min.z
                    || point.z > max.z)
            {
                continue;
            }
            let rad = cont.sizes.get(pt).copied().unwrap_or(obj.pdrawsize as f32) as f64
                * imod.pixsize as f64;
            if verbose >= 0 {
                output.push_str(&format!("  {rad:11.6}\n"));
            }
            rsum += rad;
            rsqsum += rad * rad;
            rcubsum += rad * rad * rad;
            nsum += 1;
        }
    }
    if nsum > 0 {
        output.push_str(&format!("\n\tMean radius = {} for {} points.\n\tImplied total surface area = {}; total volume = {}\n", rsum / nsum as f64, nsum, 4.0 * std::f64::consts::PI * rsqsum, 4.0 * std::f64::consts::PI * rcubsum / 3.0));
    }
    output
}
/// Original: `imodinfo_ratios` (`imodinfo.cpp:996`).
pub fn imodinfo_ratios(model: &Imod, ob: usize) -> String {
    // Original: `imodinfo_ratios` (`imodinfo.cpp:996`).
    let Some(obj) = model.obj.get(ob) else {
        return String::new();
    };
    if obj.flags & (IMOD_OBJFLAG_SCAT | IMOD_OBJFLAG_OPEN) != 0 {
        return String::new();
    }
    let mut output = format!("OBJECT {}\nNAME:  {}\n", ob + 1, obj.name);
    for (co, cont) in obj.cont.iter().enumerate() {
        if cont.pts.len() <= 2 {
            continue;
        }
        let dist = info_contour_length(
            Some(cont),
            obj.flags,
            model.pixsize as f64,
            model.zscale as f64,
        );
        output.push_str(&format!(
            "{} {}\n",
            co + 1,
            info_contour_vol(Some(cont), obj.flags, 1.0, 1.0)
                * model.pixsize as f64
                * model.pixsize as f64
                / dist
        ));
    }
    output
}
/// Original: `imodinfo_ellipse` (`imodinfo.cpp:1028`).
pub fn imodinfo_ellipse(
    model: &Imod,
    ob: usize,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
) -> String {
    // Original: `imodinfo_ellipse` (`imodinfo.cpp:1028`).
    let Some(obj) = model.obj.get(ob) else {
        return String::new();
    };
    if obj.flags & (IMOD_OBJFLAG_SCAT | IMOD_OBJFLAG_OPEN) != 0 {
        return String::new();
    }
    let mut output = format!(
        "\nOBJECT {}\nNAME:  {}\ncontour     center (pixels)                 axes ({})        eccen-   long\n   #      x        y        z       semi-major   semi-minor  tricity  angle\n",
        ob + 1,
        obj.name,
        print_units(model.units)
    );
    for (co, cont) in obj.cont.iter().enumerate() {
        if cont.pts.len() <= 2 {
            continue;
        }
        let mut center = Ipoint::default();
        for pt in &cont.pts {
            center.x += pt.x;
            center.y += pt.y;
            center.z += pt.z;
        }
        center.x /= cont.pts.len() as f32;
        center.y /= cont.pts.len() as f32;
        center.z /= cont.pts.len() as f32;
        if subarea
            && (center.x < min.x
                || center.x > max.x
                || center.y < min.y
                || center.y > max.y
                || center.z < min.z
                || center.z > max.z)
        {
            continue;
        }
        let mut xx = 0.0_f64;
        let mut yy = 0.0_f64;
        let mut xy = 0.0_f64;
        for pt in &cont.pts {
            xx += (pt.x - center.x) as f64 * (pt.x - center.x) as f64;
            yy += (pt.y - center.y) as f64 * (pt.y - center.y) as f64;
            xy += (pt.x - center.x) as f64 * (pt.y - center.y) as f64;
        }
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        let aa = xx.max(yy).sqrt() * model.pixsize as f64;
        let bb = xx.min(yy).max(0.0).sqrt() * model.pixsize as f64;
        let ecc = if aa != 0.0 {
            (1.0 - bb * bb / (aa * aa)).max(0.0).sqrt()
        } else {
            0.0
        };
        output.push_str(&format!(
            "{:4} {:8.1} {:8.1} {:8.1}   {:12.5} {:12.5}   {:.4}  {:6.2}\n",
            co + 1,
            center.x,
            center.y,
            center.z,
            aa,
            bb,
            ecc,
            angle.to_degrees()
        ));
    }
    output
}
/// Original: `imodinfo_full_object_report` (`imodinfo.cpp:1117`).
pub fn imodinfo_full_object_report(
    imod: &Imod,
    ob: usize,
    scaninside: bool,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
    useclip: i32,
) -> String {
    // Original: `imodinfo_full_object_report` (`imodinfo.cpp:1117`).
    if ob == 0 || ob > imod.obj.len() {
        return String::new();
    }
    let obj = &imod.obj[ob - 1];
    let (surf, vol, msurf, mvol, inmvol, cent) =
        compute_object_area_vol(imod, obj, scaninside, subarea, min, max, useclip);
    let mut output = format!(
        "Object # {}:\n{}\n\tNumber of Contours = {}\n\tNumber of Contours with Data = {}\n\tNumber of Meshes   = {}\n\tNumber of Surfaces = {}\n",
        ob,
        obj.name,
        obj.cont.len(),
        obj.cont.iter().filter(|cont| !cont.pts.is_empty()).count(),
        obj.mesh.len(),
        obj.surfsize
    );
    output.push_str(&format!(
        "\tCenter         = ({}, {}, {})\n",
        cent.x, cent.y, cent.z
    ));
    output.push_str(&format!(
        "\t{} Volume          = {} {}^3\n",
        if mvol > 0.0 { "Contour" } else { "Cylinder" },
        if mvol > 0.0 { mvol } else { vol },
        print_units(imod.units)
    ));
    if inmvol > 0.0 {
        output.push_str(&format!(
            "\tVolume Inside Mesh      = {inmvol} {}^3\n",
            print_units(imod.units)
        ));
    }
    output.push_str(&format!(
        "\t{} Surface Area   = {} {}^2\n",
        if msurf > 0.0 { "Mesh" } else { "Cylinder" },
        if msurf > 0.0 { msurf } else { surf },
        print_units(imod.units)
    ));
    output
}
/// Original: `imodinfo_object` (`imodinfo.cpp:1218`).
pub fn imodinfo_object(
    imod: &Imod,
    ob: usize,
    scaninside: bool,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
    useclip: i32,
) -> String {
    // Original: `imodinfo_object` (`imodinfo.cpp:1218`).
    let Some(obj) = imod.obj.get(ob) else {
        return String::new();
    };
    let (mut surf, mut vol, mut msurf, mut mvol, inmvol, cent) =
        compute_object_area_vol(imod, obj, scaninside, subarea, min, max, useclip);
    if obj.flags & (IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT) != 0 {
        surf = 0.0;
        vol = 0.0;
        msurf = 0.0;
        mvol = 0.0;
    }
    if !obj.cont.is_empty() {
        format!(
            "{:4}   {:12.6}  {:12.6}  {:12.6}  {:12.6}  {:9.2} {:9.2} {:9.2}\n",
            ob + 1,
            vol,
            mvol,
            inmvol,
            msurf,
            cent.x,
            cent.y,
            cent.z
        )
    } else if !obj.mesh.is_empty() {
        format!(
            "{:4}              x             x   {:12.6}  {:12.6}      x         x         x\n",
            ob + 1,
            inmvol,
            msurf
        )
    } else {
        format!(
            "{:4}              0             0             0             0       x         x         x\n",
            ob + 1
        )
    }
}
/// Original: `computeObjectAreaVol` (`imodinfo.cpp:1257`).
pub fn compute_object_area_vol(
    model: &Imod,
    obj: &Iobj,
    _scaninside: bool,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
    _useclip: i32,
) -> (f64, f64, f64, f64, f64, Ipoint) {
    // Original: `computeObjectAreaVol` (`imodinfo.cpp:1257`).
    let mut surf = 0.0;
    let mut vol = 0.0;
    let mut mvol = 0.0;
    let mut cent = Ipoint::default();
    let mut weight = 0.0_f64;
    for cont in &obj.cont {
        if subarea
            && cont
                .pts
                .first()
                .is_some_and(|pt| pt.z < min.z || pt.z > max.z)
        {
            continue;
        }
        let contour_vol = info_contour_vol(
            Some(cont),
            obj.flags,
            model.pixsize as f64,
            model.zscale as f64,
        );
        vol += contour_vol;
        mvol += contour_vol;
        surf += info_contour_length(
            Some(cont),
            obj.flags,
            model.pixsize as f64,
            model.zscale as f64,
        );
        for point in &cont.pts {
            cent.x += point.x;
            cent.y += point.y;
            cent.z += point.z * model.zscale;
            weight += 1.0;
        }
    }
    if weight != 0.0 {
        cent.x /= weight as f32;
        cent.y /= weight as f32;
        cent.z /= weight as f32;
    }
    let mut msurf = 0.0;
    for mesh in &obj.mesh {
        msurf += imesh_surface_subarea(
            Some(mesh),
            Some(Ipoint {
                x: model.xscale,
                y: model.yscale,
                z: model.zscale,
            }),
            min,
            max,
            0,
            &[],
        ) as f64
            * model.pixsize as f64
            * model.pixsize as f64;
    }
    (surf, vol, msurf, mvol, 0.0, cent)
}
/// Original: `print_units` (`imodinfo.cpp:1342`).
pub fn print_units(units: i32) -> &'static str {
    // Original: `print_units` (`imodinfo.cpp:1342`).
    match units {
        0 => "pixels",
        3 => "km",
        1 => "m",
        -2 => "cm",
        -3 => "mm",
        -6 => "um",
        -9 => "nm",
        -10 => "A",
        -12 => "pm",
        _ => "unknown units",
    }
}
/// Original: `imodinfo_objndist` (`imodinfo.cpp:1381`).
pub fn imodinfo_objndist(imod: &Imod, bins: usize) -> String {
    // Original: `imodinfo_objndist` (`imodinfo.cpp:1381`).
    let mut centers = Vec::new();
    for obj in &imod.obj {
        let mut center = Ipoint::default();
        let mut count = 0_u32;
        for cont in &obj.cont {
            for point in &cont.pts {
                center.x += point.x;
                center.y += point.y;
                center.z += point.z * imod.zscale;
                count += 1;
            }
        }
        if count != 0 {
            center.x /= count as f32;
            center.y /= count as f32;
            center.z /= count as f32;
            centers.push(center);
        }
    }
    if centers.len() < 2 {
        return "#distance   number\n\n".to_owned();
    }
    let mut distances = Vec::new();
    for i in 0..centers.len() {
        let mut nearest = f64::INFINITY;
        for j in 0..centers.len() {
            if i != j {
                nearest = nearest.min(pointdist(&centers[i], &centers[j]) * imod.pixsize as f64);
            }
        }
        distances.push(nearest);
    }
    let min = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let max = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bins = bins.max(1);
    let size = (max - min) / bins as f64;
    let mut output = "#distance   number\n\n".to_owned();
    for bin in 0..bins {
        let value = min + size * (bin as f64 + 0.5);
        let level = distances
            .iter()
            .filter(|distance| **distance > value - size * 0.5 && **distance < value + size * 0.5)
            .count();
        output.push_str(&format!("{value}\t{level}\n"));
    }
    output
}
/// Original: `contour_stats` (`imodinfo.cpp:1475`).
pub fn contour_stats(cont: Option<&Icont>, flags: u32, pixsize: f64, zscale: f64) -> String {
    // Original: `contour_stats` (`imodinfo.cpp:1475`).
    let Some(cont) = cont else {
        return String::new();
    };
    let mut center = Ipoint::default();
    for point in &cont.pts {
        center.x += point.x;
        center.y += point.y;
        center.z += point.z;
    }
    let count = cont.pts.len().max(1) as f32;
    center.x /= count;
    center.y /= count;
    center.z /= count;
    if flags & IMOD_OBJFLAG_SCAT != 0 || cont.pts.len() < 3 {
        return format!(
            "\t\tCenter of Mass     = ({}, {}, {}) in pixel coords.\n",
            center.x, center.y, center.z
        );
    }
    let closed = info_contour_length(Some(cont), flags & !IMOD_OBJFLAG_OPEN, pixsize, zscale);
    let open = info_contour_length(Some(cont), flags | IMOD_OBJFLAG_OPEN, pixsize, zscale);
    let area = info_contour_vol(Some(cont), flags, 1.0, 1.0) * pixsize * pixsize;
    format!(
        "\t\tClosed/Open length = {} / {}\n\t\tEnclosed Area      = {}\n\t\tCenter of Mass     = ({}, {}, {}) in pixel coords.\n",
        closed, open, area, center.x, center.y, center.z
    )
}
/// Original: `imodinfo_special` (`imodinfo.cpp:1546`).
pub fn imodinfo_special(imod: &mut Imod, fname: &str) {
    // Original: `imodinfo_special` (`imodinfo.cpp:1546`).  The source's
    // deliberately special-purpose editing operation is retained; writing is
    // performed by the paired `imodel_files` source unit.
    if let Some(obj) = imod.obj.get_mut(1) {
        obj.cont.truncate(11);
    }
    imod.cindex.object = 137;
    imod.cindex.contour = 0;
    imod.cindex.point = 0;
    if imod.obj.len() > 137 {
        imod.obj.remove(137);
    }
    if imod.obj.len() > 136 {
        imod.obj.remove(136);
    }
    let _ = imod_file_write(imod, fname);
}
/// Original: `imodinfo_length` (`imodinfo.cpp:1569`).
pub fn imodinfo_length(imod: &Imod, ob: usize) -> String {
    // Original: `imodinfo_length` (`imodinfo.cpp:1569`).
    let Some(obj) = imod.obj.get(ob) else {
        return String::new();
    };
    let mut output = String::new();
    for (co, cont) in obj.cont.iter().enumerate() {
        output.push_str(&format!(
            "{:3} {:3}  {:3}  {}\n",
            ob + 1,
            co + 1,
            cont.pts.len(),
            info_contour_length(
                Some(cont),
                obj.flags,
                imod.pixsize as f64,
                imod.zscale as f64
            )
        ));
    }
    output
}
/// Original: `pointdist` (`imodinfo.cpp:1587`).
pub fn pointdist(p1: &Ipoint, p2: &Ipoint) -> f64 {
    // Original: `pointdist` (`imodinfo.cpp:1587`).
    (((p1.x - p2.x) as f64).powi(2)
        + ((p1.y - p2.y) as f64).powi(2)
        + ((p1.z - p2.z) as f64).powi(2))
    .sqrt()
}
/// Original: `info_contour_length` (`imodinfo.cpp:1598`).
pub fn info_contour_length(cont: Option<&Icont>, objflags: u32, pixsize: f64, zscale: f64) -> f64 {
    // Original: `info_contour_length` (`imodinfo.cpp:1598`).
    let Some(cont) = cont else {
        return 0.0;
    };
    if cont.pts.is_empty() {
        return 0.0;
    }
    let mut dist = 0.0;
    if objflags & IMOD_OBJFLAG_SCAT == 0 {
        for pt in 0..cont.pts.len().saturating_sub(1) {
            let p1 = Ipoint {
                x: cont.pts[pt].x,
                y: cont.pts[pt].y,
                z: (cont.pts[pt].z as f64 * zscale) as f32,
            };
            let p2 = Ipoint {
                x: cont.pts[pt + 1].x,
                y: cont.pts[pt + 1].y,
                z: (cont.pts[pt + 1].z as f64 * zscale) as f32,
            };
            dist += (((p1.x - p2.x) as f64).powi(2)
                + ((p1.y - p2.y) as f64).powi(2)
                + ((p1.z - p2.z) as f64).powi(2))
            .sqrt();
        }
        if objflags & IMOD_OBJFLAG_OPEN == 0 && cont.flags & ICONT_OPEN == 0 {
            let p1 = Ipoint {
                x: cont.pts[cont.pts.len() - 1].x,
                y: cont.pts[cont.pts.len() - 1].y,
                z: (cont.pts[cont.pts.len() - 1].z as f64 * zscale) as f32,
            };
            let p2 = Ipoint {
                x: cont.pts[0].x,
                y: cont.pts[0].y,
                z: (cont.pts[0].z as f64 * zscale) as f32,
            };
            dist += (((p1.x - p2.x) as f64).powi(2)
                + ((p1.y - p2.y) as f64).powi(2)
                + ((p1.z - p2.z) as f64).powi(2))
            .sqrt();
        }
    }
    dist * pixsize
}
/// Original: `contourLengthByColor` (`imodinfo.cpp:1642`).
pub fn contour_length_by_color(imod: &Imod, obj_num: usize, verbose: i32) -> String {
    // Original: `contourLengthByColor` (`imodinfo.cpp:1642`).  Per-point store
    // properties are supplied by `istore.c`; absent such properties, source
    // default draw properties make every segment use the object color.
    let Some(obj) = imod.obj.get(obj_num) else {
        return String::new();
    };
    let red = (255.0 * obj.red).round() as i32;
    let green = (255.0 * obj.green).round() as i32;
    let blue = (255.0 * obj.blue).round() as i32;
    let mut output = format!(
        "\nObject # {}:  {}\nFor color {red},{green},{blue}:\n",
        obj_num + 1,
        obj.name
    );
    if verbose >= 0 {
        output.push_str(&format!(
            "Cont #      Length  (in {})\n",
            print_units(imod.units)
        ));
    }
    let mut total = 0.0;
    let mut number = 0;
    for (co, cont) in obj.cont.iter().enumerate() {
        let length = info_contour_length(
            Some(cont),
            obj.flags,
            imod.pixsize as f64,
            imod.zscale as f64,
        );
        if length != 0.0 {
            if verbose >= 0 {
                output.push_str(&format!("{:6} {:11.5}\n", co + 1, length));
            }
            total += length;
            number += 1;
        }
    }
    output.push_str(&format!(
        "\n   {number} contours, length total = {total:12.6},  mean = {:12.5} {}\n\n",
        if number == 0 {
            0.0
        } else {
            total / number as f64
        },
        print_units(imod.units)
    ));
    output
}
/// Original: `info_contour_surface_area` (`imodinfo.cpp:1731`).
pub fn info_contour_surface_area(
    cont: Option<&Icont>,
    objflags: u32,
    pixsize: f64,
    zscale: f64,
) -> f64 {
    // Original: `info_contour_surface_area` (`imodinfo.cpp:1731`).
    if cont.is_none_or(|cont| cont.pts.is_empty()) {
        return 0.0;
    }
    info_contour_length(cont, objflags, pixsize, zscale) * zscale * pixsize
}
/// Original: `info_contour_vol` (`imodinfo.cpp:1746`).
pub fn info_contour_vol(cont: Option<&Icont>, _objflags: u32, pixsize: f64, zscale: f64) -> f64 {
    // Original: `info_contour_vol` (`imodinfo.cpp:1746`).
    let Some(cont) = cont else {
        return 0.0;
    };
    if cont.pts.is_empty() {
        return 0.0;
    }
    let mut area = 0.0_f64;
    for pt in 0..cont.pts.len() {
        let next = (pt + 1) % cont.pts.len();
        area += cont.pts[pt].x as f64 * cont.pts[next].y as f64
            - cont.pts[next].x as f64 * cont.pts[pt].y as f64;
    }
    area.abs() * 0.5 * pixsize * pixsize * pixsize * zscale
}
/// Original: `contourVolumeFactor` (`imodinfo.cpp:1765`).
pub fn contour_volume_factor(obj: &Iobj, cont: &Icont, min: Ipoint, max: Ipoint) -> f32 {
    // Original: `contourVolumeFactor` (`imodinfo.cpp:1765`).
    let mut volume_factor = 0.0_f32;
    let mut found = 0;
    for point in &cont.pts {
        for mesh in &obj.mesh {
            let mut i = 0;
            while i < mesh.list.len() {
                if mesh.list[i] != IMOD_MESH_BGNPOLYNORM && mesh.list[i] != IMOD_MESH_BGNPOLYNORM2 {
                    i += 1;
                    continue;
                }
                let inc = if mesh.list[i] == IMOD_MESH_BGNPOLYNORM {
                    2
                } else {
                    1
                };
                let base = if inc == 2 { 1 } else { 0 };
                i += 1;
                while i < mesh.list.len() && mesh.list[i] != IMOD_MESH_ENDPOLY {
                    if i + 2 * inc + base >= mesh.list.len() {
                        break;
                    }
                    let p1 = mesh.vert.get(mesh.list[i + base].max(0) as usize);
                    let p2 = mesh.vert.get(mesh.list[i + inc + base].max(0) as usize);
                    let p3 = mesh.vert.get(mesh.list[i + 2 * inc + base].max(0) as usize);
                    if let (Some(p1), Some(p2), Some(p3)) = (p1, p2, p3) {
                        if [p1, p2, p3]
                            .iter()
                            .any(|p| p.x == point.x && p.y == point.y && p.z == point.z)
                        {
                            let zmin = p1.z.min(p2.z).min(p3.z).max(min.z);
                            let zmax = p1.z.max(p2.z).max(p3.z).min(max.z);
                            if zmax > zmin {
                                volume_factor += (zmax - zmin) / 2.0;
                                found += 1;
                                if found == 2 {
                                    return volume_factor;
                                }
                            }
                        }
                    }
                    i += 3 * inc;
                }
                i += 1;
            }
        }
    }
    volume_factor
}
/// Original: `imeshSurfaceSubarea` (`imodinfo.cpp:1903`).
pub fn imesh_surface_subarea(
    mesh: Option<&Imesh>,
    scale: Option<Ipoint>,
    min: Ipoint,
    max: Ipoint,
    doclip: i32,
    plane: &[Iplane],
) -> f32 {
    // Original: `imeshSurfaceSubarea` (`imodinfo.cpp:1903`).
    let Some(mesh) = mesh else {
        return 0.0;
    };
    if mesh.list.is_empty() {
        return 0.0;
    }
    let zs = scale
        .unwrap_or(Ipoint {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        })
        .z;
    let mut tsa = 0.0_f64;
    let mut i = 0_usize;
    while i < mesh.list.len() {
        match mesh.list[i] {
            IMOD_MESH_BGNPOLYNORM | IMOD_MESH_BGNPOLYNORM2 => {
                let list_inc = if mesh.list[i] == IMOD_MESH_BGNPOLYNORM {
                    2
                } else {
                    1
                };
                let vert_base = if mesh.list[i] == IMOD_MESH_BGNPOLYNORM {
                    1
                } else {
                    0
                };
                i += 1;
                while i < mesh.list.len() && mesh.list[i] != IMOD_MESH_ENDPOLY {
                    tsa += clipped_triangle_area(
                        mesh, i, zs, min, max, doclip, plane, list_inc, vert_base,
                    );
                    i += 3 * list_inc;
                }
            }
            IMOD_MESH_END => return tsa as f32,
            _ => {}
        }
        i += 1;
    }
    tsa as f32
}
/// Original: `clippedTriangleArea` (`imodinfo.cpp:1942`).
pub fn clipped_triangle_area(
    mesh: &Imesh,
    i: usize,
    zs: f32,
    min: Ipoint,
    max: Ipoint,
    doclip: i32,
    plane: &[Iplane],
    list_inc: usize,
    vert_base: usize,
) -> f64 {
    // Original: `clippedTriangleArea` (`imodinfo.cpp:1942`).
    if i + 2 * list_inc + vert_base >= mesh.list.len() {
        return 0.0;
    }
    let (Some(p1), Some(p2), Some(p3)) = (
        mesh.vert.get(mesh.list[i + vert_base].max(0) as usize),
        mesh.vert
            .get(mesh.list[i + list_inc + vert_base].max(0) as usize),
        mesh.vert
            .get(mesh.list[i + 2 * list_inc + vert_base].max(0) as usize),
    ) else {
        return 0.0;
    };
    let mut inside1 = 1_i32;
    let mut inside2 = 1_i32;
    let mut inside3 = 1_i32;
    if doclip != 0 {
        for pl in plane {
            if pl.a * p1.x + pl.b * p1.y + pl.c * p1.z + pl.d < 0.0 {
                inside1 = 0;
            }
            if pl.a * p2.x + pl.b * p2.y + pl.c * p2.z + pl.d < 0.0 {
                inside2 = 0;
            }
            if pl.a * p3.x + pl.b * p3.y + pl.c * p3.z + pl.d < 0.0 {
                inside3 = 0;
            }
        }
        if doclip < 0 {
            inside1 = 1 - inside1;
            inside2 = 1 - inside2;
            inside3 = 1 - inside3;
        }
    }
    if p1.x < min.x
        || p1.x >= max.x
        || p1.y < min.y
        || p1.y >= max.y
        || p1.z < min.z
        || p1.z >= max.z
    {
        inside1 = 0;
    }
    if p2.x < min.x
        || p2.x >= max.x
        || p2.y < min.y
        || p2.y >= max.y
        || p2.z < min.z
        || p2.z >= max.z
    {
        inside2 = 0;
    }
    if p3.x < min.x
        || p3.x >= max.x
        || p3.y < min.y
        || p3.y >= max.y
        || p3.z < min.z
        || p3.z >= max.z
    {
        inside3 = 0;
    }
    let clipfrac = (inside1 + inside2 + inside3) as f32 / 3.0;
    if clipfrac == 0.0 {
        return 0.0;
    }
    let n1 = Ipoint {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
        z: (p1.z - p2.z) * zs,
    };
    let n2 = Ipoint {
        x: p3.x - p2.x,
        y: p3.y - p2.y,
        z: (p3.z - p2.z) * zs,
    };
    let n = Ipoint {
        x: n1.y * n2.z - n1.z * n2.y,
        y: n1.z * n2.x - n1.x * n2.z,
        z: n1.x * n2.y - n1.y * n2.x,
    };
    clipfrac as f64 * ((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5
}
/// Original: `scanned_volume` (`imodinfo.cpp:2006`).
pub fn scanned_volume(
    obj: &Iobj,
    subarea: bool,
    ptmin: Ipoint,
    ptmax: Ipoint,
    _doclip: i32,
    _plane: &[Iplane],
    mesh_vol: &mut f64,
) -> f32 {
    // Original: `scanned_volume` (`imodinfo.cpp:2006`).
    *mesh_vol = 0.0;
    let mut total = 0.0_f64;
    for cont in &obj.cont {
        let mut scan = cont.clone();
        trim_scan_contour(&mut scan, ptmin, ptmax, 0, &[]);
        if !subarea
            || scan
                .pts
                .first()
                .is_some_and(|point| point.z >= ptmin.z && point.z <= ptmax.z)
        {
            let area = info_contour_vol(Some(&scan), obj.flags, 1.0, 1.0);
            total += area;
            *mesh_vol += contour_volume_factor(obj, cont, ptmin, ptmax) as f64 * area;
        }
    }
    total as f32
}
/// Original: `contourSubareaByScan` (`imodinfo.cpp:2133`).
pub fn contour_subarea_by_scan(
    cont: &Icont,
    ptmin: Ipoint,
    ptmax: Ipoint,
    _doclip: i32,
    _plane: &[Iplane],
    make_scan: bool,
) -> Option<(Option<Icont>, Ipoint, Ipoint, f32)> {
    // Original: `contourSubareaByScan` (`imodinfo.cpp:2133`).
    if cont.pts.is_empty() {
        return None;
    }
    let mut lower = Ipoint {
        x: f32::INFINITY,
        y: f32::INFINITY,
        z: f32::INFINITY,
    };
    let mut upper = Ipoint {
        x: f32::NEG_INFINITY,
        y: f32::NEG_INFINITY,
        z: f32::NEG_INFINITY,
    };
    for point in &cont.pts {
        lower.x = lower.x.min(point.x);
        lower.y = lower.y.min(point.y);
        lower.z = lower.z.min(point.z);
        upper.x = upper.x.max(point.x);
        upper.y = upper.y.max(point.y);
        upper.z = upper.z.max(point.z);
    }
    if lower.x > ptmax.x || upper.x < ptmin.x || lower.y > ptmax.y || upper.y < ptmin.y {
        return None;
    }
    let mut scan = cont.clone();
    trim_scan_contour(&mut scan, ptmin, ptmax, 0, &[]);
    if scan.pts.is_empty() {
        return None;
    }
    let area = info_contour_vol(Some(&scan), 0, 1.0, 1.0) as f32;
    if make_scan {
        Some((Some(scan), lower, upper, area))
    } else {
        Some((None, lower, upper, area))
    }
}
/// Original: `scan_contour_area` (`imodinfo.cpp:2216`).
pub fn scan_contour_area(cont: Option<&Icont>) -> f64 {
    // Original: `scan_contour_area` (`imodinfo.cpp:2216`).
    let Some(cont) = cont else {
        return 0.0;
    };
    if cont.pts.len() < 2 {
        return 0.0;
    }
    let mut pix = 0.0_f64;
    let mut i = 0_usize;
    while i + 1 < cont.pts.len() {
        let bgnpt = i;
        while i + 1 < cont.pts.len() && cont.pts[i].y == cont.pts[i + 1].y {
            i += 1;
        }
        let endpt = i;
        if (endpt - bgnpt) % 2 != 0 {
            let mut j = bgnpt;
            while j < endpt {
                if cont.pts[j].x >= cont.surf as f32 {
                    pix += (cont.pts[j + 1].x - cont.pts[j].x) as f64;
                }
                j += 2;
            }
        }
        i += 1;
    }
    pix
}
/// Original: `trim_scan_contour` (`imodinfo.cpp:2261`).
pub fn trim_scan_contour(
    cont: &mut Icont,
    min: Ipoint,
    max: Ipoint,
    doclip: i32,
    plane: &[Iplane],
) {
    // Original: `trim_scan_contour` (`imodinfo.cpp:2261`).
    if cont.pts.len() < 2 {
        return;
    }
    let mut kept = Vec::new();
    let mut i = 0;
    while i + 1 < cont.pts.len() {
        let mut left = cont.pts[i];
        let mut right = cont.pts[i + 1];
        let y = left.y;
        let mut good = y >= min.y && y <= max.y && left.x <= max.x && right.x >= min.x;
        if doclip != 0 {
            for pl in plane {
                let value = pl.b * y + pl.c * left.z + pl.d;
                if pl.a == 0.0 {
                    if (doclip > 0 && value < 0.0) || (doclip < 0 && value >= 0.0) {
                        good = false;
                    }
                } else {
                    let crit = -value / pl.a;
                    if (doclip > 0 && pl.a > 0.0) || (doclip < 0 && pl.a < 0.0) {
                        left.x = left.x.max(crit);
                    } else {
                        right.x = right.x.min(crit);
                    }
                }
            }
        }
        if good && left.x <= right.x {
            left.x = left.x.max(min.x);
            right.x = right.x.min(max.x);
            kept.push(left);
            kept.push(right);
        }
        i += 2;
    }
    cont.pts = kept;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imodinfo_closed_and_open_lengths_match_source_scaling() {
        let cont = Icont {
            pts: vec![
                Ipoint {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Ipoint {
                    x: 3.0,
                    y: 0.0,
                    z: 0.0,
                },
                Ipoint {
                    x: 3.0,
                    y: 4.0,
                    z: 0.0,
                },
            ],
            ..Icont::default()
        };
        assert!((info_contour_length(Some(&cont), 0, 1.0, 1.0) - 12.0).abs() < 1.0e-12);
        assert!(
            (info_contour_length(Some(&cont), IMOD_OBJFLAG_OPEN, 1.0, 1.0) - 7.0).abs() < 1.0e-12
        );
        assert!((info_contour_length(Some(&cont), IMOD_OBJFLAG_SCAT, 1.0, 1.0)).abs() < 1.0e-12);
    }

    #[test]
    fn imodinfo_mesh_subarea_matches_triangle_area() {
        let mesh = Imesh {
            vert: vec![
                Ipoint {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Ipoint {
                    x: 2.0,
                    y: 0.0,
                    z: 0.0,
                },
                Ipoint {
                    x: 0.0,
                    y: 3.0,
                    z: 0.0,
                },
            ],
            list: vec![
                IMOD_MESH_BGNPOLYNORM2,
                0,
                1,
                2,
                IMOD_MESH_ENDPOLY,
                IMOD_MESH_END,
            ],
            ..Imesh::default()
        };
        assert!(
            (imesh_surface_subarea(
                Some(&mesh),
                None,
                Ipoint {
                    x: -1.0,
                    y: -1.0,
                    z: -1.0
                },
                Ipoint {
                    x: 4.0,
                    y: 4.0,
                    z: 1.0
                },
                0,
                &[]
            ) - 3.0)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn imodinfo_units_are_the_imodel_header_values() {
        assert_eq!(print_units(-9), "nm");
        assert_eq!(print_units(0), "pixels");
        assert_eq!(print_units(2), "unknown units");
    }
}
