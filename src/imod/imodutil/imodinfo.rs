//! Translation of the analysis routines in `IMOD/imodutil/imodinfo.cpp`.
//!
//! The command's model-file reader belongs to the paired `libimod/imodel_files`
//! source unit.  The calculations below deliberately consume the source-shaped
//! `Imod`, `Iobj`, `Icont`, and `Imesh` declarations instead of inventing a
//! separate command-only model representation.
#![allow(dead_code)]

use std::cell::RefCell;
use std::env;
use std::io::Write;
use std::os::fd::{FromRawFd, OwnedFd};

use crate::imod::libcfshr::b3dutil::set_or_clear_flags;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, imod_backup_file, imod_copyright, imod_prog_name, imod_version,
    number_in_list,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::simplestat::sums_to_avg_sd;
use crate::imod::libimod::icont::{
    ICONT_TEMPUSE, Nesting, imod_contour_area, imod_contour_center_of_mass,
    imod_contour_check_nesting, imod_contour_circularity, imod_contour_delete,
    imod_contour_equiv_ellipse, imod_contour_free_nests, imod_contour_free_z_tables,
    imod_contour_get_bbox, imod_contour_long_axis, imod_contour_make_z_tables,
    imod_contour_nest_levels, imodel_contour_centroid, imodel_contour_scan,
};
use crate::imod::libimod::ilabel::imod_label_print;
use crate::imod::libimod::imodel::{
    ICONT_OPEN, IMOD_CLIPSIZE, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_END,
    IMOD_MESH_ENDPOLY, IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_OUT, IMOD_OBJFLAG_SCAT,
    Icont, Imesh, Imod, Iobj, Iplane, Ipoint, imod_units,
};
use crate::imod::libimod::imodel_files::{imod_file_write, imod_read, imod_write_ascii};
use crate::imod::libimod::iplane::{imod_plane_set_from_clips, imod_planes_clip};
use crate::imod::libimod::ipoint::imod_point_delete;
use crate::imod::libimod::objgroup::obj_group_lookup;

// The numerical report columns retain `c_format_bytes` only where their `%g`,
// precision, and field-width rules are part of the command's established text
// format.  It is a pure Rust formatter returning an owned `Vec<u8>`; report
// output itself always flows through Rust's `Write` implementation.

/// Per-thread destination of every report.
///
/// `main` sets it to `stdout` (`imodinfo.cpp:150`) and `-f` reopens it on a
/// file (`:287`).  It stays one of the C streams rather than becoming
/// `std::io::stdout()`: `imodVersion`, `imodCopyright` and `imodLabelPrint`
/// all write through the C stream, and the report's `# MODEL` banner goes to
/// `printf` while its body goes to `fout`, so a second independently buffered
/// handle would reorder a redirected capture.
thread_local! {
    pub static FOUT: RefCell<ImodFile> = const { RefCell::new(ImodFile::Stdout) };
}

/// Original: `imodinfo_usage` (`imodinfo.cpp:84`).
pub fn imodinfo_usage(name: &str) {
    let _ = write!(
        &mut ImodFile::Stdout,
        "usage: {name} [options] <imod filename>\n\
         options:\n\
         \t-a\tPrint ascii readable version of IMOD model file.\n\
         \t-c\tPrint centroids of closed objects.\n\
         \t-l\tPrint lengths of contours in column output.\n\
         \t-L\tPrint lengths of contour portions by fine-grained color.\n\
         \t-s\tPrint surface information.\n\
         \t-S N\tPrint surface information for every Nth surface.\n\
         \t-p\tPrint point size information.\n\
         \t-r\tPrint ratio of length to area for closed contours.\n\
         \t-e\tPrint center and axes of equivalent ellipse for closed contours.\n\
         \t-F\tPrint full report on objects.\n\
         \t-o list\tList of objects to process (default is all).\n\
         \t-g #\tNumber of object group to process.\n\
         \t-i\tAnalyze for inside contours and adjust volume.\n\
         \t-x min,max   Compute volume, mesh area, and point count and sizes\n\
         \t-y min,max         between min and max in X (-x), Y (-y), or Z (-z).\n\
         \t-z min,max   \n\
         \t-t 1/-1\tApply clipping plane in normal (1) or inverted (-1) orientation\n\
         \t-v\tBe verbose on model output.\n\
         \t-vv\tBe more verbose on model output (prints points).\n\
         \t-h\tHush - no detailed data in standard, point, or by-color output.\n\
         \t-f filename  Write output to file.\n"
    );
}
/// Original: `main` (`imodinfo.cpp:116`).
pub fn imodinfo() {
    let argv: Vec<String> = env::args().collect();
    let progname = imod_prog_name(&argv[0]);
    setExitPrefix(format!("ERROR: {progname} - ").as_bytes());
    if argv.len() == 1 {
        imod_version(Some(&progname));
        imod_copyright();
        imodinfo_usage(&progname);
        std::process::exit(0);
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
    // `fout = stdout` (`imodinfo.cpp:150`).
    FOUT.with(|fout| *fout.borrow_mut() = ImodFile::Stdout);
    while iarg < argv.len() && argv[iarg].starts_with('-') {
        match argv[iarg].as_bytes().get(1).copied().unwrap_or_default() as char {
            'g' => {
                iarg += 1;
                let Some(value) = argv.get(iarg) else {
                    std::process::exit(1)
                };
                // `atoi(argv[++i])`: `strtol` over the longest prefix that
                // converts, and zero where nothing does.
                let text = value.as_bytes();
                let mut end = 0_usize;
                while end < text.len() && text[end].is_ascii_whitespace() {
                    end += 1;
                }
                let start = end;
                if end < text.len() && (text[end] == b'-' || text[end] == b'+') {
                    end += 1;
                }
                while end < text.len() && text[end].is_ascii_digit() {
                    end += 1;
                }
                group_num = std::str::from_utf8(&text[start..end])
                    .unwrap_or("")
                    .parse::<i32>()
                    .unwrap_or(0);
                if group_num <= 0 {
                    exit_error(format!("Group number {group_num} must be positive").as_bytes());
                }
                // The source `case 'g'` has no break and falls into `case 'c'`.
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
                let value = argv.get(iarg).map(String::as_str).unwrap_or("");
                let mut low = 0_f32;
                let mut high = 0_f32;
                // `sscanf(argv[++i], "%f%*c%f", &low, &high)`: a float, one
                // suppressed character of any kind, then a second float.  Each
                // field is stored as it converts and the scan stops at the
                // first that does not, so a bare number leaves `high` at zero.
                {
                    let text = value.as_bytes();
                    let mut scan = 0_usize;
                    for field in [&mut low, &mut high] {
                        while scan < text.len() && text[scan].is_ascii_whitespace() {
                            scan += 1;
                        }
                        let start = scan;
                        let mut end = text.len();
                        let mut scanned = None;
                        while end > start {
                            if let Some(parsed) =
                                value.get(start..end).and_then(|f| f.parse::<f32>().ok())
                            {
                                scanned = Some((parsed, end));
                                break;
                            }
                            end -= 1;
                        }
                        let Some((parsed, consumed)) = scanned else {
                            break;
                        };
                        *field = parsed;
                        scan = consumed;
                        // `%*c` reads and discards exactly one character.
                        if scan >= text.len() {
                            break;
                        }
                        scan += 1;
                    }
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
                let Ok(values) = parselist(value) else {
                    exit_error(format!("Parsing list {value}").as_bytes());
                };
                list = values;
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
                hush = true;
                if argv[iarg].len() > 2 {
                    if argv[iarg] == "-help" {
                        imodinfo_usage(&progname);
                        std::process::exit(0);
                    }
                    exit_error(
                        format!("Unknown option {}; enter -help for help", argv[iarg]).as_bytes(),
                    );
                }
            }
            _ => {
                let _ = writeln!(
                    &mut ImodFile::Stdout,
                    "{progname}: unknown option {}",
                    argv[iarg]
                );
                imodinfo_usage(&progname);
                std::process::exit(2);
            }
        }
        iarg += 1;
    }
    if iarg >= argv.len() {
        imodinfo_usage(&progname);
        std::process::exit(2);
    }
    if let Some(filename) = out_file.as_ref() {
        if imod_backup_file(filename) != 0 {
            exit_error(
                format!("Could not make ~ backup of existing output file {filename}").as_bytes(),
            );
        }
        match ImodFile::open(filename, "w") {
            Some(opened) => FOUT.with(|fout| *fout.borrow_mut() = opened),
            None => {
                exit_error(format!("Opening output file {filename}").as_bytes());
            }
        }
    }
    // `fout` is fixed from here on: the `-f` branch above is the only
    // assignment `main` makes after `fout = stdout` (`imodinfo.cpp:150, 287`).
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    if !list.is_empty() && group_num > 0 {
        exit_error(b"You cannot enter both an object list and an object group");
    }
    if hush && verbose == 0 {
        verbose = -1;
    }
    for (index, filename) in argv.iter().enumerate().skip(iarg) {
        match std::fs::File::open(filename) {
            Ok(fin) => drop(fin),
            Err(error) => {
                let errno = error.raw_os_error().unwrap_or(0);
                let message = if errno != 0 {
                    format!("Opening input file {filename} - system message: {error}")
                } else {
                    format!("Opening input file {filename}")
                };
                exit_error(message.as_bytes());
            }
        }
        let mut model = match imod_read(filename) {
            Ok(model) => model,
            Err(error) => {
                let _ = writeln!(
                    &mut ImodFile::Stdout,
                    "{progname}: Error ({error}) reading imod model. ({filename})"
                );
                continue;
            }
        };
        if group_num > 0 {
            if model.group_list.is_empty() {
                exit_error(format!("There are no object groups in model {filename}").as_bytes());
            }
            if group_num > model.group_list.len() as i32 {
                exit_error(
                    format!(
                        "Group # {group_num} is more than the number of object groups ({}) in {filename}",
                        model.group_list.len()
                    )
                    .as_bytes(),
                );
            }
        }
        let _ = index;
        let _ = writeln!(fout, "# MODEL {filename}");
        // `Imod.name` is the fixed 128-byte array `imodel_write` emits
        // whole; `%s` stops at its first NUL.
        let name_end = model
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(model.name.len());
        let _ = fout.write_all(&c_format_bytes(
            "# NAME  %s\n",
            &[CArg::Bytes(&model.name[..name_end])],
        ));
        let _ = fout.write_all(&c_format_bytes(
            "# PIX SCALE:  x = %g\n",
            &[CArg::Dbl(model.xscale as f64)],
        ));
        let _ = fout.write_all(&c_format_bytes(
            "#             y = %g\n",
            &[CArg::Dbl(model.yscale as f64)],
        ));
        let _ = fout.write_all(&c_format_bytes(
            "#             z = %g\n",
            &[CArg::Dbl(model.zscale as f64)],
        ));
        let _ = fout.write_all(&c_format_bytes(
            "# PIX SIZE      = %g\n",
            &[CArg::Dbl(model.pixsize as f64)],
        ));
        let _ = fout.write_all(b"# UNITS: ");
        print_units(model.units);
        if let Some(reference) = model.ref_image {
            let _ = fout.write_all(b"\n# Model to Image index coords:\n");
            let _ = fout.write_all(&c_format_bytes(
                "#      SCALE  = ( %g, %g, %g)\n",
                &[
                    CArg::Dbl(reference.cscale.x as f64),
                    CArg::Dbl(reference.cscale.y as f64),
                    CArg::Dbl(reference.cscale.z as f64),
                ],
            ));
            let _ = fout.write_all(&c_format_bytes(
                "#      OFFSET = ( %g, %g, %g)\n",
                &[
                    CArg::Dbl(reference.ctrans.x as f64),
                    CArg::Dbl(reference.ctrans.y as f64),
                    CArg::Dbl(reference.ctrans.z as f64),
                ],
            ));
            let _ = fout.write_all(&c_format_bytes(
                "#      ANGLES = ( %g, %g, %g)\n",
                &[
                    CArg::Dbl(reference.crot.x as f64),
                    CArg::Dbl(reference.crot.y as f64),
                    CArg::Dbl(reference.crot.z as f64),
                ],
            ));
        }
        let mut obj_list = Vec::<usize>::new();
        for ob in 0..model.obj.len() {
            if group_num > 0 {
                if obj_group_lookup(&model.group_list[group_num as usize - 1], ob as i32) >= 0 {
                    obj_list.push(ob);
                }
            } else if number_in_list(ob as i32 + 1, Some(&list), list.len() as i32, 1) != 0 {
                obj_list.push(ob);
            }
        }
        let _ = fout.write_all(b"\n\n");
        if model.obj.is_empty() {
            let _ = fout.write_all(b"Model has no objects!!!\n");
        }
        match mode {
            4 => {
                // `imodinfo.cpp:452` passes this program's own `fout`.
                //
                // `imodWriteAscii` starts with `rewind(fout)`
                // (`imodel_files.c:1645`), and that really seeks when `fout`
                // is `stdout` on a regular file: native's `-a` output under a
                // shell redirect is the ascii text written *over* the
                // `# MODEL` banner, while under a pipe `fseek` fails and the
                // text is appended.  [`ImodFile`]'s `Seek` treats the
                // standard-stream arms as unseekable, so the descriptor is
                // duplicated into a `File` arm, which seeks exactly as the C
                // stream does.  The C buffer is flushed first, and `dup`
                // shares the open file *description*, so the two handles share
                // one offset and the `\n\n` written after this lands where
                // the ascii text ended.
                let _ = fout.flush();
                // `dup` is the immediate POSIX boundary needed to preserve
                // the shared descriptor offset.  Convert it to owned Rust
                // state at the boundary so no raw descriptor escapes.
                let duplicated = unsafe { libc::dup(fout.fileno()) };
                if duplicated < 0 {
                    exit_error(b"Could not duplicate output file descriptor");
                }
                let owned = unsafe { OwnedFd::from_raw_fd(duplicated) };
                let mut out = ImodFile::File(std::rc::Rc::new(std::fs::File::from(owned)));
                imod_write_ascii(&model, &mut out);
                let _ = out.flush();
            }
            6 => {
                for ob in &obj_list {
                    imodinfo_surface(
                        &model, *ob, scaninside, minimum, maximum, useclip, sample, verbose,
                    );
                }
            }
            8 => {
                for ob in &obj_list {
                    imodinfo_points(&model, *ob, subarea, minimum, maximum, useclip, verbose);
                }
            }
            12 => {
                for ob in &obj_list {
                    imodinfo_ellipse(&model, *ob, subarea, minimum, maximum);
                }
            }
            9 => {
                for ob in &obj_list {
                    imodinfo_ratios(&model, *ob);
                }
            }
            2 => {
                let _ = fout.write_all(b"#Obj       Cyl. Vol      Cont Vol   Vol Inside Mesh   Mesh Surf              Center\n");
                let _ = fout.write_all(b"#--------------------------------------------------------------------------------------------\n");
                for ob in &obj_list {
                    imodinfo_object(&model, *ob, scaninside, subarea, minimum, maximum, useclip);
                }
            }
            3 => imodinfo_objndist(&model, bins),
            5 => {
                let _ = fout.write_all(&c_format_bytes(
                    "# Obj Cont Pnts Length (in %s)\n",
                    &[CArg::Str(imod_units(&model))],
                ));
                let _ = fout.write_all(b"#------------------------\n");
                for ob in &obj_list {
                    imodinfo_length(&model, *ob);
                }
            }
            10 => {
                for ob in &obj_list {
                    if model.obj[*ob].flags & (IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT) == 0
                        || model.obj[*ob].flags & IMOD_OBJFLAG_SCAT == 0
                    {
                        contour_length_by_color(&model, *ob, verbose);
                    }
                }
            }
            7 => {
                for ob in &obj_list {
                    imodinfo_full_object_report(
                        &model,
                        *ob + 1,
                        scaninside,
                        subarea,
                        minimum,
                        maximum,
                        useclip,
                    );
                }
            }
            _ => {
                for ob in &obj_list {
                    imodinfo_print_model(
                        &mut model, *ob, verbose, scaninside, subarea, minimum, maximum, useclip,
                    );
                }
            }
        }
        let _ = fout.write_all(b"\n\n");
    }
    std::process::exit(0);
}
/// Original: `imodinfo_print_model` (`imodinfo.cpp:464`).
pub fn imodinfo_print_model(
    model: &mut Imod,
    ob: usize,
    verbose: i32,
    scaninside: bool,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
    _useclip: i32,
) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_print_model` (`imodinfo.cpp:464`).  The source keeps
    // `Iobj *obj = &(model->obj[ob])` for the whole body; the translation takes
    // the alias afresh in each region because `contour_stats` is non-const.
    if model.obj.get(ob).is_none() {
        return;
    }
    let mut tsa = 0.0_f64;
    let mut tvol = 0.0_f64;
    let mut mvol = 0.0_f64;
    let mut inmvol = 0.0_f64;
    let mut msa = 0.0_f64;
    // `imodPlaneSetFromClips` (`imodinfo.cpp:487`) fills `plane` from the
    // object's and the current view's clip sets; `doclip` is `useclip` only
    // when it produced any.
    let view = &model.view[model.cview.clamp(0, model.view.len() as i32 - 1) as usize];
    let mut plane = [Iplane::default(); 2 * IMOD_CLIPSIZE];
    let mut n_planes = 0_i32;
    imod_plane_set_from_clips(
        Some(&model.obj[ob].clips),
        Some(&view.clips),
        &mut plane,
        2 * IMOD_CLIPSIZE as i32,
        &mut n_planes,
    );
    let doclip = if n_planes != 0 { _useclip } else { 0 };
    let obj = &model.obj[ob];
    let _ = fout.write_all(&c_format_bytes(
        "OBJECT %d\n",
        &[CArg::Int((ob as i32 + 1) as i64)],
    ));
    {
        let end = obj
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(obj.name.len());
        let _ = fout.write_all(&c_format_bytes(
            "NAME:  %s
",
            &[CArg::Bytes(&obj.name[..end])],
        ));
    }
    let _ = fout.write_all(&c_format_bytes(
        "       %d contours\n",
        &[CArg::Int((obj.cont.len() as i32) as i64)],
    ));
    if obj.flags & IMOD_OBJFLAG_OFF != 0 {
        let _ = fout.write_all(b"       object drawing is turned off\n");
    }
    if obj.flags & IMOD_OBJFLAG_SCAT != 0 {
        let _ = fout.write_all(b"       object uses scattered points.\n");
    } else if obj.flags & IMOD_OBJFLAG_OPEN != 0 {
        let _ = fout.write_all(b"       object uses open contours.\n");
    } else {
        let _ = fout.write_all(b"       object uses closed contours.\n");
    }
    if obj.flags & IMOD_OBJFLAG_OUT != 0 {
        let _ = fout.write_all(b"       contours in object are inside out.\n");
    }
    let _ = fout.write_all(&c_format_bytes(
        "       color (red, green, blue) = (%g, %g, %g)\n",
        &[
            CArg::Dbl(obj.red as f64),
            CArg::Dbl(obj.green as f64),
            CArg::Dbl(obj.blue as f64),
        ],
    ));
    let _ = fout.write_all(b"\n");
    if scaninside && model.obj[ob].flags & IMOD_OBJFLAG_OPEN == 0 {
        let mut mesh_vol = 0.0_f64;
        tvol = model.pixsize as f64
            * model.pixsize as f64
            * scanned_volume(
                &model.obj[ob],
                subarea,
                min,
                max,
                doclip,
                &plane[..n_planes.max(0) as usize],
                &mut mesh_vol,
            ) as f64;
        mvol = mesh_vol;
    } else {
        for co in 0..model.obj[ob].cont.len() {
            let obj = &model.obj[ob];
            let cont = &obj.cont[co];
            let npt = cont.pts.len();
            if verbose >= 0 {
                let _ = fout.write_all(&c_format_bytes(
                    "\tCONTOUR #%d,%d,%d  %d points",
                    &[
                        CArg::Int((co as i32 + 1) as i64),
                        CArg::Int((ob as i32 + 1) as i64),
                        CArg::Int((cont.surf) as i64),
                        CArg::Int((npt as i32) as i64),
                    ],
                ));
            }
            if verbose > 1 {
                let _ = fout.write_all(b"\n\t");
                // `imodinfo.cpp:540` prints the label to `stdout`, not to
                // `fout`.  `ImodFile::Stdout` is that same C stream, so the
                // two stay in order.
                imod_label_print(cont.label.as_ref(), &mut ImodFile::Stdout);
            }
            if cont.pts.is_empty() {
                if verbose >= 0 {
                    let _ = fout.write_all(b"\n");
                }
                continue;
            }
            if subarea && obj.flags & IMOD_OBJFLAG_OPEN == 0 {
                let coz = cont.pts[0].z.round() as i32;
                if (coz as f32) < min.z || (coz as f32) > max.z {
                    if verbose >= 0 {
                        let _ = fout.write_all(b"\n");
                    }
                    let vol_fac = contour_volume_factor(obj, cont, min, max);
                    if vol_fac > 0. {
                        // `imodContourArea` (`icont.c:324`): magnitude of the summed cross
                        // products of successive points, halved, accumulated in float.
                        let mut n = Ipoint::default();
                        if cont.pts.len() >= 3 {
                            for i in 0..cont.pts.len() {
                                let next = if i == cont.pts.len() - 1 { 0 } else { i + 1 };
                                n.x += cont.pts[i].y * cont.pts[next].z
                                    - cont.pts[i].z * cont.pts[next].y;
                                n.y += cont.pts[i].z * cont.pts[next].x
                                    - cont.pts[i].x * cont.pts[next].z;
                                n.z += cont.pts[i].x * cont.pts[next].y
                                    - cont.pts[i].y * cont.pts[next].x;
                            }
                        }
                        let area =
                            (((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5) as f32;
                        mvol += vol_fac as f64
                            * area as f64
                            * model.pixsize as f64
                            * model.pixsize as f64;
                    }
                    continue;
                }
            }
            if verbose <= 0 {
                let dist = info_contour_length(
                    Some(cont),
                    obj.flags,
                    model.pixsize as f64,
                    model.zscale as f64,
                );
                if obj.flags & IMOD_OBJFLAG_OPEN == 0 {
                    if verbose >= 0 {
                        let _ = fout.write_all(&c_format_bytes(
                            ", length = %g, ",
                            &[CArg::Dbl(dist as f64)],
                        ));
                    }
                    // `imodContourArea` (`icont.c:324`): magnitude of the summed cross
                    // products of successive points, halved, accumulated in float.
                    let mut n = Ipoint::default();
                    if cont.pts.len() >= 3 {
                        for i in 0..cont.pts.len() {
                            let next = if i == cont.pts.len() - 1 { 0 } else { i + 1 };
                            n.x +=
                                cont.pts[i].y * cont.pts[next].z - cont.pts[i].z * cont.pts[next].y;
                            n.y +=
                                cont.pts[i].z * cont.pts[next].x - cont.pts[i].x * cont.pts[next].z;
                            n.z +=
                                cont.pts[i].x * cont.pts[next].y - cont.pts[i].y * cont.pts[next].x;
                        }
                    }
                    let area = (((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5) as f32;
                    let mut sa = area as f64;
                    sa *= model.pixsize as f64 * model.pixsize as f64;
                    if verbose >= 0 {
                        let _ = fout
                            .write_all(&c_format_bytes(" area = %g\n", &[CArg::Dbl(sa as f64)]));
                    }
                    tsa += dist;
                    tvol += sa;
                    mvol += sa * contour_volume_factor(obj, cont, min, max) as f64;
                } else if verbose >= 0 {
                    let _ = fout.write_all(&c_format_bytes(
                        "\tlength = %g %s\n",
                        &[CArg::Dbl(dist as f64), CArg::Str(imod_units(model))],
                    ));
                }
            } else {
                let _ = fout.write_all(b".\n");
            }
            if verbose > 1 {
                let _ = fout.write_all(b"\t\tx\ty\tz\n");
                for point in &cont.pts {
                    let _ = fout.write_all(&c_format_bytes(
                        "\t\t%g\t%g\t%g\n",
                        &[
                            CArg::Dbl(point.x as f64),
                            CArg::Dbl(point.y as f64),
                            CArg::Dbl(point.z as f64),
                        ],
                    ));
                }
            }
            if verbose > 0 {
                let obj_flags = obj.flags;
                let pixsize = model.pixsize as f64;
                let zscale = model.zscale as f64;
                contour_stats(model.obj[ob].cont.get_mut(co), obj_flags, pixsize, zscale);
            }
        }
    }
    let obj = &model.obj[ob];
    if !obj.mesh.is_empty() {
        // `imodinfo.cpp:609-620` selects the nearest resolution first and then
        // processes only the meshes at that resolution.  Looping over every
        // mesh double-counts a multi-resolution object.
        let mscale = Ipoint {
            x: model.xscale,
            y: model.yscale,
            z: model.zscale,
        };
        let mut resol = 0;
        crate::imod::libimod::imesh::imod_mesh_nearest_res(
            &obj.mesh,
            obj.mesh.len() as i32,
            0,
            &mut resol,
        );
        for mesh in &obj.mesh {
            if crate::imod::libimod::imesh::imesh_resol(mesh.flag) != resol {
                continue;
            }
            msa += imesh_surface_subarea(
                Some(mesh),
                Some(mscale),
                min,
                max,
                doclip,
                &plane[..n_planes.max(0) as usize],
            ) as f64;
            inmvol +=
                crate::imod::libimod::imesh::imesh_volume(Some(mesh), Some(&mscale), None) as f64;
        }
        msa *= model.pixsize as f64 * model.pixsize as f64;
        // `imodinfo.cpp:620` is `pow((double)pixsize, 3.)`, not three float
        // multiplies.
        inmvol *= (model.pixsize as f64).powf(3.);
    }
    if mvol > 0.0 {
        mvol *= model.zscale as f64 * model.pixsize as f64;
        let _ = fout.write_all(&c_format_bytes(
            "\tTotal contour volume = %g\n",
            &[CArg::Dbl(mvol as f32 as f64)],
        ));
    } else if tvol > 0.0 {
        tvol *= model.zscale as f64 * model.pixsize as f64;
        let _ = fout.write_all(&c_format_bytes(
            "\n\tTotal cylinder volume = %g\n",
            &[CArg::Dbl(tvol as f32 as f64)],
        ));
    }
    if inmvol > 0. {
        let _ = fout.write_all(&c_format_bytes(
            "\tTotal volume inside mesh = %g\n",
            &[CArg::Dbl(inmvol as f32 as f64)],
        ));
    }
    if msa > 0. {
        let _ = fout.write_all(&c_format_bytes(
            "\tTotal mesh surface area = %g\n",
            &[CArg::Dbl(msa as f32 as f64)],
        ));
    } else if tsa > 0.0 {
        tsa *= model.zscale as f64 * model.pixsize as f64;
        let _ = fout.write_all(&c_format_bytes(
            "\tTotal cylinder surface area = %g\n",
            &[CArg::Dbl(tsa as f32 as f64)],
        ));
    }
    let _ = fout.write_all(b"\n");
}
/// Original: `imodinfo_surface` (`imodinfo.cpp:646`).
pub fn imodinfo_surface(
    imod: &Imod,
    ob: usize,
    scaninside: bool,
    min: Ipoint,
    max: Ipoint,
    _useclip: i32,
    sample: usize,
    _verbose: i32,
) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_surface` (`imodinfo.cpp:646`).
    let Some(obj) = imod.obj.get(ob) else {
        return;
    };
    // `imodPlaneSetFromClips` (`imodinfo.cpp:669`).
    let view = &imod.view[imod.cview.clamp(0, imod.view.len() as i32 - 1) as usize];
    let mut plane = [Iplane::default(); 2 * IMOD_CLIPSIZE];
    let mut n_planes = 0_i32;
    imod_plane_set_from_clips(
        Some(&obj.clips),
        Some(&view.clips),
        &mut plane,
        2 * IMOD_CLIPSIZE as i32,
        &mut n_planes,
    );
    let doclip = if n_planes != 0 { _useclip } else { 0 };
    let plane = &plane[..n_planes.max(0) as usize];
    let sample = sample.max(1);
    let surfsize = obj.surfsize.max(0) as usize;
    let mut sa = vec![0.0_f64; surfsize + 1];
    let mut vol = vec![0.0_f64; surfsize + 1];
    let mut mvol = vec![0.0_f64; surfsize + 1];
    let mut nofc = vec![0_i32; surfsize + 1];
    let mut extent = vec![0.0_f32; surfsize + 1];
    let mut i = 0_usize;
    while i <= surfsize {
        let mut distmax = 0.0_f32;
        for co in 0..obj.cont.len() {
            let cont = &obj.cont[co];
            if cont.pts.is_empty() || cont.surf != i as i32 {
                continue;
            }
            for cont2 in obj.cont.iter().skip(co) {
                if cont2.pts.is_empty() || cont2.surf != i as i32 {
                    continue;
                }
                for p1 in &cont.pts {
                    for p2 in &cont2.pts {
                        let delx = p1.x - p2.x;
                        let dely = p1.y - p2.y;
                        let delz = (p1.z - p2.z) * imod.zscale;
                        if delx.abs() + dely.abs() + delz.abs() < extent[i] {
                            continue;
                        }
                        let dist = delx * delx + dely * dely + delz * delz;
                        if dist > distmax {
                            distmax = dist;
                            extent[i] = (dist as f64).sqrt() as f32;
                        }
                    }
                }
            }
        }
        extent[i] *= imod.pixsize;
        i += sample;
    }
    for cont in &obj.cont {
        if cont.pts.is_empty() {
            continue;
        }
        let i = cont.surf.max(0) as usize;
        if i % sample != 0 || i > surfsize {
            continue;
        }
        let mut mean_z = 0.0_f32;
        for point in &cont.pts {
            mean_z += point.z;
        }
        let coz = (mean_z / cont.pts.len() as f32 + 0.5).floor() as i32;
        let skipz = (coz as f32) < min.z || (coz as f32) > max.z;
        let vol_fac = contour_volume_factor(obj, cont, min, max);
        if (vol_fac > 0. && !(scaninside || doclip != 0)) || !skipz {
            if let Some((_scan, _pmin, _pmax, tvol)) =
                contour_subarea_by_scan(cont, min, max, doclip, plane, false)
            {
                let tvol = tvol * imod.pixsize * imod.pixsize * imod.pixsize * imod.zscale;
                if vol_fac != 0. {
                    mvol[i] += tvol as f64 * vol_fac as f64;
                }
                if !skipz {
                    nofc[i] += 1;
                    vol[i] += tvol as f64;
                }
            }
        }
        if !(scaninside || doclip != 0 || skipz) {
            sa[i] += info_contour_surface_area(
                Some(cont),
                obj.flags,
                imod.pixsize as f64,
                imod.zscale as f64,
            );
        }
    }
    {
        let end = obj
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(obj.name.len());
        let _ = fout.write_all(&c_format_bytes(
            "\n#Object %d data, %s\n",
            &[
                CArg::Int((ob as i32 + 1) as i64),
                CArg::Bytes(&obj.name[..end]),
            ],
        ));
    }
    if !obj.mesh.is_empty() {
        let zs = imod.zscale;
        let mut max_mesh_surf = 0_i32;
        let mut max_cont_surf = 0_i32;
        let mut msa = vec![0.0_f64; surfsize + 1];
        let mut vol_in_mesh = vec![0.0_f64; surfsize + 1];
        let mut resol = 0;
        crate::imod::libimod::imesh::imod_mesh_nearest_res(
            &obj.mesh,
            obj.mesh.len() as i32,
            0,
            &mut resol,
        );

        for mesh in &obj.mesh {
            if mesh.list.is_empty() || crate::imod::libimod::imesh::imesh_resol(mesh.flag) != resol
            {
                continue;
            }
            if mesh.surf >= 0
                && mesh.surf as usize <= surfsize
                && (mesh.surf as usize % sample) == 0
            {
                let mscale = Ipoint {
                    x: imod.xscale,
                    y: imod.yscale,
                    z: imod.zscale,
                };
                vol_in_mesh[mesh.surf as usize] +=
                    crate::imod::libimod::imesh::imesh_volume(Some(mesh), Some(&mscale), None)
                        as f64
                        * (imod.pixsize as f64).powf(3.);
                if mesh.surf as i32 > max_mesh_surf {
                    max_mesh_surf = mesh.surf as i32;
                }
            }

            if sample < 2 || _verbose >= 0 {
                let mut i = 0_usize;
                while i < mesh.list.len() {
                    let mut list_inc = 0;
                    let mut vert_base = 0;
                    let mut norm_add = 0;
                    if crate::imod::libimod::imesh::imod_mesh_poly_norm_factors(
                        mesh.list[i],
                        &mut list_inc,
                        &mut vert_base,
                        &mut norm_add,
                    ) != 0
                    {
                        i += 1;

                        /* Get a total area for this polygon and find
                        the surface whose contours it matches */
                        let mut psa = 0.;
                        let mut found = false;
                        let mut psurf = 0_i32;
                        if obj.cont.is_empty() && mesh.surf >= 0 && mesh.surf as usize <= surfsize {
                            psurf = mesh.surf as i32;
                            found = true;
                        }
                        while i < mesh.list.len() && mesh.list[i] != IMOD_MESH_ENDPOLY {
                            psa += clipped_triangle_area(
                                mesh,
                                i,
                                zs,
                                min,
                                max,
                                doclip,
                                plane,
                                list_inc as usize,
                                vert_base as usize,
                            );
                            let p1 = mesh.vert[mesh.list[i + vert_base as usize] as usize];
                            i += list_inc as usize;
                            let p2 = mesh.vert[mesh.list[i + vert_base as usize] as usize];
                            i += list_inc as usize;
                            let p3 = mesh.vert[mesh.list[i + vert_base as usize] as usize];
                            i += list_inc as usize;

                            if !found {
                                /* Scan contours to find match */
                                for cont in &obj.cont {
                                    if cont.pts.is_empty()
                                        || (cont.surf.max(0) as usize % sample) != 0
                                    {
                                        continue;
                                    }
                                    for point in &cont.pts {
                                        let (xx, yy, zz) = (point.x, point.y, point.z);
                                        if zz != p1.z && zz != p2.z && zz != p3.z {
                                            break;
                                        }
                                        if (xx == p1.x && yy == p1.y && zz == p1.z)
                                            || (xx == p2.x && yy == p2.y && zz == p2.z)
                                            || (xx == p3.x && yy == p3.y && zz == p3.z)
                                        {
                                            found = true;
                                            psurf = cont.surf;
                                            break;
                                        }
                                    }
                                    if found {
                                        break;
                                    }
                                }
                            }
                        }
                        if found
                            && (psurf % sample as i32) == 0
                            && psurf >= 0
                            && psurf as usize <= surfsize
                        {
                            msa[psurf as usize] += psa * imod.pixsize as f64 * imod.pixsize as f64;
                            if psurf > max_cont_surf {
                                max_cont_surf = psurf;
                            }
                        }
                    }
                    i += 1;
                }
            }
        }
        if max_mesh_surf > 0 && obj.cont.is_empty() {
            let _ = fout.write_all(b"#Surface : Contours,  Mesh Volume,  Mesh Surface\n");
            let mut i = 0_usize;
            while i <= surfsize {
                let _ = fout.write_all(&c_format_bytes(
                    "%7d   %8d   %12.6g  %12.6g\n",
                    &[
                        CArg::Int((i as i32) as i64),
                        CArg::Int((nofc[i]) as i64),
                        CArg::Dbl(vol_in_mesh[i] as f64),
                        CArg::Dbl(msa[i] as f64),
                    ],
                ));
                i += sample;
            }
        } else if max_cont_surf > 0 && max_mesh_surf == 0 && mvol[max_cont_surf as usize] > 0. {
            let _ = fout.write_all(
                b"#Surface : Contours,  Cyl. Volume,  Cont. Volume,  Mesh Surface,  Max Extent\n",
            );
            let mut i = 0_usize;
            while i <= surfsize {
                let _ = fout.write_all(&c_format_bytes(
                    "%7d   %8d   %12.6g  %12.6g  %12.6g  %12.6g\n",
                    &[
                        CArg::Int((i as i32) as i64),
                        CArg::Int((nofc[i]) as i64),
                        CArg::Dbl(vol[i] as f64),
                        CArg::Dbl(mvol[i] as f64),
                        CArg::Dbl(msa[i] as f64),
                        CArg::Dbl(extent[i] as f64),
                    ],
                ));
                i += sample;
            }
        } else if max_cont_surf > 0 && max_mesh_surf > 0 && max_cont_surf == max_mesh_surf {
            let _ = fout.write_all(
                b"#Surface : Contours,  Cont. Volume,  Mesh Volume,  Mesh Surface,  Max Extent\n",
            );
            let mut i = 0_usize;
            while i <= surfsize {
                let _ = fout.write_all(&c_format_bytes(
                    "%7d   %8d   %12.6g  %12.6g  %12.6g  %12.6g\n",
                    &[
                        CArg::Int((i as i32) as i64),
                        CArg::Int((nofc[i]) as i64),
                        CArg::Dbl(mvol[i] as f64),
                        CArg::Dbl(vol_in_mesh[i] as f64),
                        CArg::Dbl(msa[i] as f64),
                        CArg::Dbl(extent[i] as f64),
                    ],
                ));
                i += sample;
            }
        } else {
            let _ = fout.write_all(b"#Surface : Contours,  Cyl. Volume,  Cont. Volume,  Mesh Volume,  Mesh Surface,  Max Extent\n");
            let mut i = 0_usize;
            while i <= surfsize {
                let _ = fout.write_all(&c_format_bytes(
                    "%7d   %8d   %12.6g  %12.6g  %12.6g  %12.6g  %12.6g\n",
                    &[
                        CArg::Int((i as i32) as i64),
                        CArg::Int((nofc[i]) as i64),
                        CArg::Dbl(vol[i] as f64),
                        CArg::Dbl(mvol[i] as f64),
                        CArg::Dbl(vol_in_mesh[i] as f64),
                        CArg::Dbl(msa[i] as f64),
                        CArg::Dbl(extent[i] as f64),
                    ],
                ));
                i += sample;
            }
        }
    } else {
        let _ = fout.write_all(b"#Surface : Contours,  Cyl. Volume,  Cyl. Surface,  Max. Extent\n");
        let mut i = 0_usize;
        while i <= surfsize {
            let _ = fout.write_all(&c_format_bytes(
                "%7d   %8d   %12.6g  %12.6g  %12.6g\n",
                &[
                    CArg::Int((i as i32) as i64),
                    CArg::Int((nofc[i]) as i64),
                    CArg::Dbl(vol[i] as f64),
                    CArg::Dbl(sa[i] as f64),
                    CArg::Dbl(extent[i] as f64),
                ],
            ));
            i += sample;
        }
    }
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
) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_points` (`imodinfo.cpp:910`).
    let Some(obj) = imod.obj.get(ob) else {
        return;
    };
    let mut objheader = false;
    let mut rsum = 0.0_f64;
    let mut rsqsum = 0.0_f64;
    let mut rcubsum = 0.0_f64;
    let mut nsum = 0_i32;
    let pi = 3.14159_f32;
    // `imodPlaneSetFromClips` (`imodinfo.cpp:929`).
    let view = &imod.view[imod.cview.clamp(0, imod.view.len() as i32 - 1) as usize];
    let mut plane = [Iplane::default(); 2 * IMOD_CLIPSIZE];
    let mut n_planes = 0_i32;
    imod_plane_set_from_clips(
        Some(&obj.clips),
        Some(&view.clips),
        &mut plane,
        2 * IMOD_CLIPSIZE as i32,
        &mut n_planes,
    );
    for (co, cont) in obj.cont.iter().enumerate() {
        if !cont.sizes.is_empty()
            || obj.flags & IMOD_OBJFLAG_SCAT != 0
            || (verbose > 0 && obj.pdrawsize > 0)
        {
            if !objheader {
                objheader = true;
                let end = obj
                    .name
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(obj.name.len());
                let _ = fout.write_all(&c_format_bytes(
                    "\n#Object %d data, %s\n",
                    &[
                        CArg::Int((ob as i32 + 1) as i64),
                        CArg::Bytes(&obj.name[..end]),
                    ],
                ));
            }
            if verbose >= 0 {
                let _ = fout.write_all(&c_format_bytes(
                    "\tCONTOUR #%d,%d,%d  %d points",
                    &[
                        CArg::Int((co as i32 + 1) as i64),
                        CArg::Int((ob as i32 + 1) as i64),
                        CArg::Int((cont.surf) as i64),
                        CArg::Int((cont.pts.len() as i32) as i64),
                    ],
                ));
                if subarea || (_useclip != 0 && n_planes != 0) {
                    let _ = fout.write_all(b" total, before constraints");
                }
                let _ = fout.write_all(b"\n");
            }
            for (pt, p1) in cont.pts.iter().enumerate() {
                let mut skip = subarea;
                if skip
                    && p1.x >= min.x
                    && p1.x <= max.x
                    && p1.y >= min.y
                    && p1.y <= max.y
                    && p1.z >= min.z
                    && p1.z <= max.z
                {
                    skip = false;
                }
                if !skip {
                    // `imodPointGetSize` (`iobj.c`): the point size if set,
                    // else the object's default point-draw size.
                    let size = match cont.sizes.get(pt) {
                        Some(value) if *value >= 0. => *value,
                        _ => obj.pdrawsize as f32,
                    };
                    let rad = size * imod.pixsize;
                    if verbose >= 0 {
                        let _ =
                            fout.write_all(&c_format_bytes("  %11.6g\n", &[CArg::Dbl(rad as f64)]));
                    }
                    rsum += rad as f64;
                    rsqsum += (rad * rad) as f64;
                    rcubsum += (rad * rad * rad) as f64;
                    nsum += 1;
                }
            }
        }
    }
    if nsum > 0 {
        let rad = (rsum / nsum as f64) as f32;
        let area = (4. * pi as f64 * rsqsum) as f32;
        let volume = (4. * pi as f64 * rcubsum / 3.) as f32;
        let _ = fout.write_all(&c_format_bytes("\n\tMean radius = %g for %d points.\n\tImplied total surface area = %g; total volume = %g\n", &[CArg::Dbl(rad as f64), CArg::Int((nsum) as i64), CArg::Dbl(area as f64), CArg::Dbl(volume as f64)]));
    }
}
/// Original: `imodinfo_ratios` (`imodinfo.cpp:996`).
pub fn imodinfo_ratios(model: &Imod, ob: usize) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_ratios` (`imodinfo.cpp:996`).
    let Some(obj) = model.obj.get(ob) else {
        return;
    };
    if obj.flags & IMOD_OBJFLAG_SCAT != 0 || obj.flags & IMOD_OBJFLAG_OPEN != 0 {
        return;
    }
    let _ = fout.write_all(&c_format_bytes(
        "OBJECT %d\n",
        &[CArg::Int((ob as i32 + 1) as i64)],
    ));
    {
        let end = obj
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(obj.name.len());
        let _ = fout.write_all(&c_format_bytes(
            "NAME:  %s
",
            &[CArg::Bytes(&obj.name[..end])],
        ));
    }
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
        // `imodContourArea` (`icont.c:324`): magnitude of the summed cross
        // products of successive points, halved, accumulated in float.
        let mut n = Ipoint::default();
        if cont.pts.len() >= 3 {
            for i in 0..cont.pts.len() {
                let next = if i == cont.pts.len() - 1 { 0 } else { i + 1 };
                n.x += cont.pts[i].y * cont.pts[next].z - cont.pts[i].z * cont.pts[next].y;
                n.y += cont.pts[i].z * cont.pts[next].x - cont.pts[i].x * cont.pts[next].z;
                n.z += cont.pts[i].x * cont.pts[next].y - cont.pts[i].y * cont.pts[next].x;
            }
        }
        let area = (((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5) as f32;
        let mut sa = area as f64;
        sa *= model.pixsize as f64 * model.pixsize as f64;
        let _ = fout.write_all(&c_format_bytes(
            "%d %g\n",
            &[
                CArg::Int((co as i32 + 1) as i64),
                CArg::Dbl((sa / dist) as f64),
            ],
        ));
    }
}
/// Original: `imodinfo_ellipse` (`imodinfo.cpp:1028`).
pub fn imodinfo_ellipse(model: &Imod, ob: usize, subarea: bool, min: Ipoint, max: Ipoint) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_ellipse` (`imodinfo.cpp:1028`).
    let Some(obj) = model.obj.get(ob) else {
        return;
    };
    if obj.flags & IMOD_OBJFLAG_SCAT != 0 || obj.flags & IMOD_OBJFLAG_OPEN != 0 {
        return;
    }
    let sector_crit = 44.0_f32;
    let mut sector = 0_i32;
    let mut nsum = 0_i32;
    let mut ang_sum = 0.0_f32;
    let mut ang_sum_sq = 0.0_f32;
    let mut aa_sum = 0.0_f32;
    let mut aa_sum_sq = 0.0_f32;
    let mut bb_sum = 0.0_f32;
    let mut bb_sum_sq = 0.0_f32;
    let mut ecc_sum = 0.0_f32;
    let mut ecc_sum_sq = 0.0_f32;
    let mut min_angle = 1000.0_f32;
    let mut max_angle = -1000.0_f32;
    let _ = fout.write_all(&c_format_bytes(
        "\nOBJECT %d\n",
        &[CArg::Int((ob as i32 + 1) as i64)],
    ));
    {
        let end = obj
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(obj.name.len());
        let _ = fout.write_all(&c_format_bytes(
            "NAME:  %s
",
            &[CArg::Bytes(&obj.name[..end])],
        ));
    }
    let _ = fout.write_all(&c_format_bytes(
        "contour     center (pixels)                 axes (%s)        eccen-   long\n   #      x        y        z       semi-major   semi-minor  tricity  angle\n",
        &[CArg::Str(imod_units(model))],
    ));
    for (co, cont) in obj.cont.iter().enumerate() {
        if cont.pts.len() <= 2 {
            continue;
        }
        let mut center = Ipoint::default();
        let mut aa = 0.0_f32;
        let mut bb = 0.0_f32;
        let mut angle = 0.0_f32;
        if imod_contour_equiv_ellipse(Some(cont), &mut center, &mut aa, &mut bb, &mut angle) != 0 {
            continue;
        }
        if subarea
            && !(center.x >= min.x
                && center.x <= max.x
                && center.y >= min.y
                && center.y <= max.y
                && center.z >= min.z
                && center.z <= max.z)
        {
            continue;
        }
        // This is standard definition for eccentricity, a not terribly intuitive number
        // (`imodinfo.cpp:1062`): `sqrt(1. - bb * bb / (aa * aa))` in double.
        let ecc = (1. - (bb as f64 * bb as f64) / (aa as f64 * aa as f64)).sqrt() as f32;
        aa *= model.pixsize;
        bb *= model.pixsize;
        if angle < sector_crit || angle > 180. - sector_crit {
            if nsum == 0 {
                sector = if angle < sector_crit { 1 } else { -1 };
            } else if sector < 0 && angle < sector_crit {
                angle += 180.;
            } else if sector > 0 && angle > 180. - sector_crit {
                angle -= 180.;
            }
        }
        let _ = fout.write_all(&c_format_bytes(
            "%4d %8.1f %8.1f %8.1f   %12.5g %12.5g   %.4f  %6.2f\n",
            &[
                CArg::Int((co as i32 + 1) as i64),
                CArg::Dbl(center.x as f64),
                CArg::Dbl(center.y as f64),
                CArg::Dbl(center.z as f64),
                CArg::Dbl(aa as f64),
                CArg::Dbl(bb as f64),
                CArg::Dbl(ecc as f64),
                CArg::Dbl(angle as f64),
            ],
        ));
        aa_sum += aa;
        aa_sum_sq += aa * aa;
        bb_sum += bb;
        bb_sum_sq += bb * bb;
        ecc_sum += ecc;
        ecc_sum_sq += ecc * ecc;
        ang_sum += angle;
        ang_sum_sq += angle * angle;
        nsum += 1;
        min_angle = min_angle.min(angle);
        max_angle = max_angle.max(angle);
    }
    if nsum != 0 {
        let mut aa_avg = 0.0_f32;
        let mut aa_sd = 0.0_f32;
        let mut bb_avg = 0.0_f32;
        let mut bb_sd = 0.0_f32;
        let mut ecc_avg = 0.0_f32;
        let mut ecc_sd = 0.0_f32;
        let mut ang_avg = 0.0_f32;
        let mut ang_sd = 0.0_f32;
        sums_to_avg_sd(aa_sum, aa_sum_sq, nsum, &mut aa_avg, &mut aa_sd);
        sums_to_avg_sd(bb_sum, bb_sum_sq, nsum, &mut bb_avg, &mut bb_sd);
        sums_to_avg_sd(ecc_sum, ecc_sum_sq, nsum, &mut ecc_avg, &mut ecc_sd);
        sums_to_avg_sd(ang_sum, ang_sum_sq, nsum, &mut ang_avg, &mut ang_sd);
        if ang_avg < 0. {
            ang_avg += 180.;
        }
        if ang_avg >= 180. {
            ang_avg -= 180.;
        }
        let _ = ImodFile::Stdout.write_all(&c_format_bytes("Mean                              %12.5g %12.5g   %.4f  %6.2f\n SD                               %12.5g %12.5g   %.4f  %6.2f\n", &[CArg::Dbl(aa_avg as f64), CArg::Dbl(bb_avg as f64), CArg::Dbl(ecc_avg as f64), CArg::Dbl(ang_avg as f64), CArg::Dbl(aa_sd as f64), CArg::Dbl(bb_sd as f64), CArg::Dbl(ecc_sd as f64), CArg::Dbl(ang_sd as f64)]));
        if max_angle - min_angle > 2. * sector_crit {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "WARNING: The range of angles is %.0f degrees and the mean may be inaccurate\n",
                &[CArg::Dbl((max_angle - min_angle) as f64)],
            ));
        }
    }
}
/// Original: `imodinfo_full_object_report` (`imodinfo.cpp:1117`).
pub fn imodinfo_full_object_report(
    imod: &Imod,
    ob: usize,
    scaninside: bool,
    subarea: bool,
    ptmin: Ipoint,
    ptmax: Ipoint,
    useclip: i32,
) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_full_object_report` (`imodinfo.cpp:1117`).
    if ob < 1 || ob > imod.obj.len() {
        return;
    }
    let obj = &imod.obj[ob - 1];
    // `imodUnits` (`imodel.c:1360`).
    let units = imod_units(imod);
    let _ = fout.write_all(&c_format_bytes(
        "Object # %d:\n",
        &[CArg::Int((ob as i32) as i64)],
    ));
    {
        let end = obj
            .name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(obj.name.len());
        let _ = fout.write_all(&c_format_bytes("%s\n", &[CArg::Bytes(&obj.name[..end])]));
    }
    let _ = fout.write_all(&c_format_bytes(
        "\tNumber of Contours = %d\n",
        &[CArg::Int((obj.cont.len() as i32) as i64)],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tNumber of Contours with Data = %d\n",
        &[CArg::Int(
            (obj.cont.iter().filter(|cont| !cont.pts.is_empty()).count() as i32) as i64,
        )],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tNumber of Meshes   = %d\n",
        &[CArg::Int((obj.mesh.len() as i32) as i64)],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tNumber of Surfaces = %d\n",
        &[CArg::Int((obj.surfsize) as i64)],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tColor  = (Red, Green, Blue, Alpha) = (%g, %g, %g, %g)\n",
        &[
            CArg::Dbl(obj.red as f64),
            CArg::Dbl(obj.green as f64),
            CArg::Dbl(obj.blue as f64),
            CArg::Dbl((obj.trans as f32 * 0.01_f32) as f64),
        ],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tAmbient Light  = %d\n",
        &[CArg::Int((obj.ambient as i32) as i64)],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tDiffuse Light  = %d\n",
        &[CArg::Int((obj.diffuse as i32) as i64)],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tSpecular Light = %d\n",
        &[CArg::Int((obj.specular as i32) as i64)],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "\tShininess      = %d\n",
        &[CArg::Int((obj.shininess as i32) as i64)],
    ));
    // `imodObjectGetBBox` (`iobj.c`).
    let mut lower = Ipoint {
        x: f32::MAX,
        y: f32::MAX,
        z: f32::MAX,
    };
    let mut upper = Ipoint {
        x: -f32::MAX,
        y: -f32::MAX,
        z: -f32::MAX,
    };
    if obj.cont.is_empty() {
        for mesh in &obj.mesh {
            for point in &mesh.vert {
                lower.x = lower.x.min(point.x);
                lower.y = lower.y.min(point.y);
                lower.z = lower.z.min(point.z);
                upper.x = upper.x.max(point.x);
                upper.y = upper.y.max(point.y);
                upper.z = upper.z.max(point.z);
            }
        }
    } else {
        for cont in &obj.cont {
            for point in &cont.pts {
                lower.x = lower.x.min(point.x);
                lower.y = lower.y.min(point.y);
                lower.z = lower.z.min(point.z);
                upper.x = upper.x.max(point.x);
                upper.y = upper.y.max(point.y);
                upper.z = upper.z.max(point.z);
            }
        }
    }
    let _ = fout.write_all(&c_format_bytes(
        "\n\tBounding Box   = { (%g, %g, %g), (%g, %g, %g)}\n",
        &[
            CArg::Dbl(lower.x as f64),
            CArg::Dbl(lower.y as f64),
            CArg::Dbl(lower.z as f64),
            CArg::Dbl(upper.x as f64),
            CArg::Dbl(upper.y as f64),
            CArg::Dbl(upper.z as f64),
        ],
    ));
    let (surf, vol, msurf, mvol, inmvol, cent) = compute_object_area_vol(
        imod,
        obj,
        false,
        false,
        Ipoint {
            x: -1.0e30,
            y: -1.0e30,
            z: -1.0e30,
        },
        Ipoint {
            x: 1.0e30,
            y: 1.0e30,
            z: 1.0e30,
        },
        0,
    );
    let _ = fout.write_all(&c_format_bytes(
        "\tCenter         = (%g, %g, %g)\n",
        &[
            CArg::Dbl(cent.x as f64),
            CArg::Dbl(cent.y as f64),
            CArg::Dbl(cent.z as f64),
        ],
    ));
    if mvol > 0. {
        let _ = fout.write_all(&c_format_bytes(
            "	Contour Volume          = %g %s^3
",
            &[CArg::Dbl(mvol), CArg::Str(units)],
        ));
    } else {
        let _ = fout.write_all(&c_format_bytes(
            "	Cylinder Volume         = %g %s^3
",
            &[CArg::Dbl(vol), CArg::Str(units)],
        ));
    }
    if inmvol > 0. {
        let _ = fout.write_all(&c_format_bytes(
            "	Volume Inside Mesh      = %g %s^3
",
            &[CArg::Dbl(inmvol), CArg::Str(units)],
        ));
    }
    if msurf > 0. {
        let _ = fout.write_all(&c_format_bytes(
            "	Mesh Surface Area       = %g %s^2
",
            &[CArg::Dbl(msurf), CArg::Str(units)],
        ));
    } else {
        let _ = fout.write_all(&c_format_bytes(
            "	Cylinder Surface Area   = %g %s^2
",
            &[CArg::Dbl(surf), CArg::Str(units)],
        ));
    }
    let mut num_clips = 0;
    for clip in 0..(obj.clips.count as usize).min(obj.clips.normal.len()) {
        if obj.clips.flags & (1 << clip) != 0 {
            let _ = fout.write_all(&c_format_bytes(
                "\tClip %d Normal    = (%g, %g, %g)\n",
                &[
                    CArg::Int((clip as i32) as i64),
                    CArg::Dbl(obj.clips.normal[clip].x as f64),
                    CArg::Dbl(obj.clips.normal[clip].y as f64),
                    CArg::Dbl((obj.clips.normal[clip].z / imod.zscale) as f64),
                ],
            ));
            let _ = fout.write_all(&c_format_bytes(
                "\tClip %d Point     = (%g, %g, %g)\n",
                &[
                    CArg::Int((clip as i32) as i64),
                    CArg::Dbl(obj.clips.point[clip].x as f64),
                    CArg::Dbl(obj.clips.point[clip].y as f64),
                    CArg::Dbl(obj.clips.point[clip].z as f64),
                ],
            ));
            num_clips += 1;
        }
    }
    if obj.clips.flags & (1 << 7) == 0 {
        if let Some(view) = imod.view.get(imod.cview.max(0) as usize) {
            for clip in 0..(view.clips.count as usize).min(view.clips.normal.len()) {
                if view.clips.flags & (1 << clip) != 0 {
                    num_clips += 1;
                }
            }
        }
    }
    if subarea || (useclip != 0 && num_clips != 0) {
        let _ = fout.write_all(b"    Clipped and/or subsetted values:\n");
        let (surf, vol, msurf, mvol, _inmvol, _cent) =
            compute_object_area_vol(imod, obj, scaninside, subarea, ptmin, ptmax, useclip);
        if mvol > 0. {
            let _ = fout.write_all(&c_format_bytes(
                "	Contour Volume          = %g %s^3
",
                &[CArg::Dbl(mvol), CArg::Str(units)],
            ));
        } else {
            let _ = fout.write_all(&c_format_bytes(
                "	Cylinder Volume         = %g %s^3
",
                &[CArg::Dbl(vol), CArg::Str(units)],
            ));
        }
        if msurf > 0. {
            let _ = fout.write_all(&c_format_bytes(
                "	Mesh Surface Area       = %g %s^2
",
                &[CArg::Dbl(msurf), CArg::Str(units)],
            ));
        } else if surf > 0. {
            let _ = fout.write_all(&c_format_bytes(
                "	Cylinder Surface Area   = %g %s^2
",
                &[CArg::Dbl(surf), CArg::Str(units)],
            ));
        }
    }
    let _ = fout.write_all(b"\n");
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
) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_object` (`imodinfo.cpp:1218`).
    let Some(obj) = imod.obj.get(ob) else {
        return;
    };
    let (mut surf, mut vol, mut msurf, mut mvol, inmvol, cent) =
        compute_object_area_vol(imod, obj, scaninside, subarea, min, max, useclip);
    if obj.flags & IMOD_OBJFLAG_OPEN != 0 || obj.flags & IMOD_OBJFLAG_SCAT != 0 {
        surf = 0.0;
        vol = 0.0;
        msurf = 0.0;
        mvol = 0.0;
    }
    let _ = surf;
    if !obj.cont.is_empty() {
        let _ = fout.write_all(&c_format_bytes(
            "%4d   %12.6g  %12.6g  %12.6g  %12.6g  %9.2f %9.2f %9.2f\n",
            &[
                CArg::Int((ob as i32 + 1) as i64),
                CArg::Dbl(vol as f64),
                CArg::Dbl(mvol as f64),
                CArg::Dbl(inmvol as f64),
                CArg::Dbl(msurf as f64),
                CArg::Dbl(cent.x as f64),
                CArg::Dbl(cent.y as f64),
                CArg::Dbl(cent.z as f64),
            ],
        ));
    } else if !obj.mesh.is_empty() {
        let _ = fout.write_all(&c_format_bytes(
            "%4d              x             x   %12.6g  %12.6g      x         x         x\n",
            &[
                CArg::Int((ob as i32 + 1) as i64),
                CArg::Dbl(inmvol as f64),
                CArg::Dbl(msurf as f64),
            ],
        ));
    } else {
        let _ = fout.write_all(&c_format_bytes("%4d              0             0             0             0       x         x         x\n", &[CArg::Int((ob as i32 + 1) as i64)]));
    }
}
/// Original: `computeObjectAreaVol` (`imodinfo.cpp:1257`).
pub fn compute_object_area_vol(
    model: &Imod,
    obj: &Iobj,
    scaninside: bool,
    subarea: bool,
    min: Ipoint,
    max: Ipoint,
    _useclip: i32,
) -> (f64, f64, f64, f64, f64, Ipoint) {
    // Original: `computeObjectAreaVol` (`imodinfo.cpp:1257`).
    let mut surf = 0.0_f64;
    let mut vol = 0.0_f64;
    let mut mvol = 0.0_f64;
    let mut msurf = 0.0_f64;
    let mut inmvol = 0.0_f64;
    let mut cent = Ipoint::default();
    let mut tweight = 0.0_f64;
    let zscale = model.zscale as f64;
    let pixsize = model.pixsize as f64;
    // `imodPlaneSetFromClips` (`imodinfo.cpp:1280`) fills `plane` from the
    // object's and the current view's clip sets.
    let view = &model.view[model.cview.clamp(0, model.view.len() as i32 - 1) as usize];
    let mut plane = [Iplane::default(); 2 * IMOD_CLIPSIZE];
    let mut n_planes = 0_i32;
    imod_plane_set_from_clips(
        Some(&obj.clips),
        Some(&view.clips),
        &mut plane,
        2 * IMOD_CLIPSIZE as i32,
        &mut n_planes,
    );
    let doclip = if n_planes != 0 { _useclip } else { 0 };
    let plane = &plane[..n_planes.max(0) as usize];
    if scaninside && obj.flags & IMOD_OBJFLAG_OPEN == 0 {
        let mut mesh_vol = 0.0_f64;
        vol = pixsize.powi(3)
            * zscale
            * scanned_volume(obj, subarea, min, max, doclip, plane, &mut mesh_vol) as f64;
        mvol = mesh_vol * pixsize.powi(3) * zscale;
    } else {
        for cont in &obj.cont {
            let tvol = info_contour_vol(Some(cont), obj.flags, pixsize, zscale) as f32;
            mvol += tvol as f64 * contour_volume_factor(obj, cont, min, max) as f64;
            if subarea && obj.flags & IMOD_OBJFLAG_OPEN == 0 {
                let mut mean_z = 0.0_f32;
                for point in &cont.pts {
                    mean_z += point.z;
                }
                let coz = if cont.pts.is_empty() {
                    -1
                } else {
                    (mean_z / cont.pts.len() as f32 + 0.5).floor() as i32
                };
                if (coz as f32) < min.z || (coz as f32) > max.z {
                    continue;
                }
            }
            vol += tvol as f64;
            surf += info_contour_length(Some(cont), obj.flags, pixsize, zscale);
            /* 2/24/09: Compute differently for open contours, so set a flag
            for an open contour object */
            // `imodinfo.cpp:1303` sets ICONT_TEMPUSE on the contour for the
            // duration of the call and clears it right after.  This function
            // holds the object by shared reference, so the flag is set on a
            // copy of the contour; `imodel_contour_centroid` (`icont.c:551`)
            // reads the contour and never writes it, and nothing else observes
            // the flag between the two calls.
            let mut scratch = cont.clone();
            let mut ccent = Ipoint::default();
            let mut weight = 0.0_f64;
            set_or_clear_flags(
                &mut scratch.flags,
                ICONT_TEMPUSE,
                (obj.flags & IMOD_OBJFLAG_OPEN) as i32,
            );
            imodel_contour_centroid(Some(&scratch), &mut ccent, &mut weight);
            set_or_clear_flags(&mut scratch.flags, ICONT_TEMPUSE, 0);
            tweight += weight;
            cent.x += ccent.x;
            cent.y += ccent.y;
            cent.z += ccent.z * model.zscale;
        }
        surf *= model.pixsize as f64 * model.zscale as f64;
        if tweight != 0. {
            cent.x /= tweight as f32;
            cent.y /= tweight as f32;
            cent.z /= tweight as f32;
        }
    }
    if !obj.mesh.is_empty() {
        msurf = 0.0;
        let mscale = Ipoint {
            x: model.xscale,
            y: model.yscale,
            z: model.zscale,
        };
        let mut resol = 0;
        crate::imod::libimod::imesh::imod_mesh_nearest_res(
            &obj.mesh,
            obj.mesh.len() as i32,
            0,
            &mut resol,
        );
        for mesh in &obj.mesh {
            if crate::imod::libimod::imesh::imesh_resol(mesh.flag) != resol {
                continue;
            }
            if subarea || doclip != 0 {
                msurf +=
                    imesh_surface_subarea(Some(mesh), Some(mscale), min, max, doclip, plane) as f64;
            } else {
                msurf += crate::imod::libimod::imesh::imesh_surface_area(Some(mesh), Some(&mscale))
                    as f64;
            }
            inmvol +=
                crate::imod::libimod::imesh::imesh_volume(Some(mesh), Some(&mscale), None) as f64;
        }
        msurf *= model.pixsize as f64 * model.pixsize as f64;
        inmvol *= model.pixsize as f64 * model.pixsize as f64 * model.pixsize as f64;
    }
    (surf, vol, msurf, mvol, inmvol, cent)
}
/// Original: `print_units` (`imodinfo.cpp:1342`).
pub fn print_units(units: i32) {
    // Original: `print_units` (`imodinfo.cpp:1342`).
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    let _ = fout.write_all(match units {
        0 => b"pixels".as_slice(),
        3 => b"km".as_slice(),
        1 => b"m".as_slice(),
        -2 => b"cm".as_slice(),
        -3 => b"mm".as_slice(),
        -6 => b"um".as_slice(),
        -9 => b"nm".as_slice(),
        -10 => b"A".as_slice(),
        -12 => b"pm".as_slice(),
        _ => b"unknown units".as_slice(),
    });
}
/// Original: `imodinfo_objndist` (`imodinfo.cpp:1381`).
pub fn imodinfo_objndist(imod: &Imod, bins: usize) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_objndist` (`imodinfo.cpp:1381`).
    let bins = if bins == 0 { 20 } else { bins };
    let mut pixsize = imod.pixsize as f64;
    if pixsize == 0. {
        pixsize = 1.;
    }
    let mut zscale = imod.zscale as f64;
    if zscale == 0. {
        zscale = 1.;
    }
    let _ = fout.write_all(b"#distance   number\n\n");
    if imod.obj.is_empty() {
        return;
    }
    let mut dcont = Vec::<Ipoint>::new();
    for obj in &imod.obj {
        let mut pnt = Ipoint::default();
        let mut tpt = 0_i32;
        for cont in &obj.cont {
            for point in &cont.pts {
                pnt.x += point.x;
                pnt.y += point.y;
                pnt.z += (point.z as f64 * zscale) as f32;
                tpt += 1;
            }
        }
        pnt.x /= tpt as f32;
        pnt.y /= tpt as f32;
        pnt.z /= tpt as f32;
        dcont.push(pnt);
    }
    let mut dista = vec![0.0_f64; imod.obj.len()];
    for i in 0..dcont.len().saturating_sub(1) {
        let mut mindist = pixsize * pointdist(&dcont[1], &dcont[0]);
        if i != 0 {
            mindist = pixsize * pointdist(&dcont[i], &dcont[0]);
        }
        for pt in 0..dcont.len() {
            if pt == i {
                break;
            }
            let mut dist = pixsize * pointdist(&dcont[i], &dcont[pt]);
            if dist < mindist {
                dist += 1.;
            }
            let _ = dist;
        }
        dista[i] = mindist;
    }
    let mut max = dista[0];
    let mut min = dista[0];
    for value in dista.iter().skip(1) {
        if *value < min {
            min = *value;
        }
        if *value > max {
            max = *value;
        }
    }
    let binsize = (max - min) / bins as f64;
    let mut binval = min + binsize * 0.5;
    for _bin in 0..bins {
        let mut level = 0_i32;
        let binmin = binval - binsize * 0.5;
        let binmax = binval + binsize * 0.5;
        for value in &dista {
            if *value < binmax && *value > binmin {
                level += 1;
            }
        }
        let _ = fout.write_all(&c_format_bytes(
            "%f\t%d\n",
            &[CArg::Dbl(binval as f64), CArg::Int((level) as i64)],
        ));
        binval += binsize;
    }
}
/// Original: `contour_stats` (`imodinfo.cpp:1475`).
///
/// The contour is taken by mutable reference because `imodContourCenterOfMass`
/// (`icont.c:527-533`) saves, sets and restores the contour's `flags`, so the
/// source's `Icont *cont` parameter is not const.
pub fn contour_stats(cont: Option<&mut Icont>, flags: u32, pixsize: f64, zscale: f64) -> i32 {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `contour_stats` (`imodinfo.cpp:1475`).
    let Some(cont) = cont else {
        return -1;
    };
    let mut cmass = Ipoint::default();
    if flags & IMOD_OBJFLAG_SCAT != 0 || cont.pts.len() < 3 {
        for point in &cont.pts {
            cmass.x += point.x;
            cmass.y += point.y;
            cmass.z += point.z;
        }
        let num = cont.pts.len().max(1) as f32;
        let _ = fout.write_all(&c_format_bytes(
            "\t\tCenter of Mass     = (%g, %g, %g) in pixel coords.\n",
            &[
                CArg::Dbl((cmass.x / num) as f64),
                CArg::Dbl((cmass.y / num) as f64),
                CArg::Dbl((cmass.z / num) as f64),
            ],
        ));
        return 0;
    }
    let odist = info_contour_length(Some(cont), flags | IMOD_OBJFLAG_OPEN, pixsize, zscale) as f32;
    let cdist = info_contour_length(Some(cont), flags & !IMOD_OBJFLAG_OPEN, pixsize, zscale) as f32;
    let _ = fout.write_all(&c_format_bytes(
        "\t\tClosed/Open length = %g / %g\n",
        &[CArg::Dbl(cdist as f64), CArg::Dbl(odist as f64)],
    ));

    let mut area = imod_contour_area(Some(cont));
    area *= (pixsize * pixsize) as f32;
    let _ = fout.write_all(&c_format_bytes(
        "\t\tEnclosed Area      = %g\n",
        &[CArg::Dbl(area as f64)],
    ));

    imod_contour_center_of_mass(Some(cont), &mut cmass);
    let _ = fout.write_all(&c_format_bytes(
        "\t\tCenter of Mass     = (%g, %g, %g) in pixel coords.\n",
        &[
            CArg::Dbl(cmass.x as f64),
            CArg::Dbl(cmass.y as f64),
            CArg::Dbl(cmass.z as f64),
        ],
    ));

    let mut ll = Ipoint::default();
    let mut ur = Ipoint::default();
    imod_contour_get_bbox(Some(cont), &mut ll, &mut ur);
    let _ = fout.write_all(&c_format_bytes(
        "\t\tBounding Box        = {(%g, %g), (%g, %g)}\n",
        &[
            CArg::Dbl(ll.x as f64),
            CArg::Dbl(ll.y as f64),
            CArg::Dbl(ur.x as f64),
            CArg::Dbl(ur.y as f64),
        ],
    ));

    let _ = fout.write_all(&c_format_bytes(
        "\t\tCircularity        = %g\n",
        &[CArg::Dbl(imod_contour_circularity(Some(cont)) as f64)],
    ));

    let mut aspect = 0.;
    let mut length = 0.;
    let mut orientation = imod_contour_long_axis(Some(cont), 1.0, &mut aspect, &mut length);
    orientation /= 0.01745329252;

    let _ = fout.write_all(&c_format_bytes(
        "\t\tOrientation        = %g degrees.\n",
        &[CArg::Dbl(orientation as f64)],
    ));

    let width = if length > 0. { length / aspect } else { 0. };

    if length != 0. && width != 0. {
        let _ = fout.write_all(&c_format_bytes(
            "\t\tEllipse            = %g\n",
            &[CArg::Dbl(
                (area / (0.7853981635 * length as f64 * width as f64 * pixsize * pixsize) as f32)
                    as f64,
            )],
        ));
    }

    let _ = fout.write_all(&c_format_bytes(
        "\t\tLength X Width     = %g x %g\n",
        &[CArg::Dbl(length as f64), CArg::Dbl(width as f64)],
    ));

    let _ = fout.write_all(&c_format_bytes(
        "\t\tAspect Ratio       = %g\n",
        &[CArg::Dbl(aspect as f64)],
    ));

    0
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
pub fn imodinfo_length(imod: &Imod, ob: usize) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `imodinfo_length` (`imodinfo.cpp:1569`).
    let Some(obj) = imod.obj.get(ob) else {
        return;
    };
    for (co, cont) in obj.cont.iter().enumerate() {
        let dist = info_contour_length(
            Some(cont),
            obj.flags,
            imod.pixsize as f64,
            imod.zscale as f64,
        );
        let _ = fout.write_all(&c_format_bytes(
            "%3d %3d  %3d  %g\n",
            &[
                CArg::Int((ob as i32 + 1) as i64),
                CArg::Int((co as i32 + 1) as i64),
                CArg::Int((cont.pts.len() as i32) as i64),
                CArg::Dbl(dist as f64),
            ],
        ));
    }
}
/// Original: `pointdist` (`imodinfo.cpp:1587`).
pub fn pointdist(p1: &Ipoint, p2: &Ipoint) -> f64 {
    // Original: `pointdist` (`imodinfo.cpp:1587`).  Every operand of the sum is
    // `float` in the source, so the squares and their sum are formed in single
    // precision and only the `sqrt` is done in double.
    let dist = ((p1.x - p2.x) * (p1.x - p2.x))
        + ((p1.y - p2.y) * (p1.y - p2.y))
        + ((p1.z - p2.z) * (p1.z - p2.z));
    (dist as f64).sqrt()
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
            dist += pointdist(&p1, &p2);
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
            dist += pointdist(&p1, &p2);
        }
    }
    dist * pixsize
}
/// Original: `contourLengthByColor` (`imodinfo.cpp:1642`).
pub fn contour_length_by_color(imod: &Imod, obj_num: usize, verbose: i32) {
    let mut fout = FOUT.with(|fout| fout.borrow().clone());
    // Original: `contourLengthByColor` (`imodinfo.cpp:1642`).  Per-point store
    // properties are supplied by `istore.c`; absent such properties, source
    // default draw properties make every segment use the object color.
    let Some(obj) = imod.obj.get(obj_num) else {
        return;
    };
    // `imodUnits` (`imodel.c:1360`).
    let units = imod_units(imod);
    let color = ((255.0_f32 * obj.red).round() as i32) << 16
        | ((255.0_f32 * obj.green).round() as i32) << 8
        | (255.0_f32 * obj.blue).round() as i32;
    let end = obj
        .name
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(obj.name.len());
    let _ = fout.write_all(&c_format_bytes(
        "\nObject # %d:  %s\n",
        &[
            CArg::Int((obj_num as i32 + 1) as i64),
            CArg::Bytes(&obj.name[..end]),
        ],
    ));
    let _ = fout.write_all(&c_format_bytes(
        "For color %d,%d,%d:\n",
        &[
            CArg::Int((color >> 16) as i64),
            CArg::Int(((color >> 8) & 255) as i64),
            CArg::Int((color & 255) as i64),
        ],
    ));
    if verbose >= 0 {
        let _ = fout.write_all(&c_format_bytes(
            "Cont #      Length  (in %s)\n",
            &[CArg::Str(units)],
        ));
    }
    let mut total = 0.0_f64;
    let mut num_total = 0_i32;
    for (co, cont) in obj.cont.iter().enumerate() {
        let dist = info_contour_length(Some(cont), obj.flags, 1.0, imod.zscale as f64);
        if dist != 0.0 {
            if verbose >= 0 {
                let _ = fout.write_all(&c_format_bytes(
                    "%6d %11.5g\n",
                    &[
                        CArg::Int((co as i32 + 1) as i64),
                        CArg::Dbl(dist * imod.pixsize as f64),
                    ],
                ));
            }
            total += dist;
            num_total += 1;
        }
    }
    let _ = fout.write_all(&c_format_bytes(
        "%s   %d contours, length total = %12.6g,  mean = %12.5g %s\n\n",
        &[
            CArg::Str(if verbose >= 0 { "\n" } else { "" }),
            CArg::Int(num_total as i64),
            CArg::Dbl(imod.pixsize as f64 * total),
            CArg::Dbl(imod.pixsize as f64 * total / num_total as f64),
            CArg::Str(units),
        ],
    ));
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
    // `imodContourArea` (`icont.c:324`): magnitude of the summed cross
    // products of successive points, halved, accumulated in float.
    let mut n = Ipoint::default();
    if cont.pts.len() >= 3 {
        for i in 0..cont.pts.len() {
            let next = if i == cont.pts.len() - 1 { 0 } else { i + 1 };
            n.x += cont.pts[i].y * cont.pts[next].z - cont.pts[i].z * cont.pts[next].y;
            n.y += cont.pts[i].z * cont.pts[next].x - cont.pts[i].x * cont.pts[next].z;
            n.z += cont.pts[i].x * cont.pts[next].y - cont.pts[i].y * cont.pts[next].x;
        }
    }
    let area = (((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5) as f32;
    let mut vol = area as f64;
    vol *= pixsize * pixsize * pixsize * zscale;
    vol
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
///
/// `imodContourMakeZTables` takes the object non-const in the source, but with
/// `clearFlag` 0 -- the only value this caller passes -- its single write is
/// `flags &= ~0`.  This function holds the object by shared reference, so the
/// tables are built from a copy; nothing can diverge.
pub fn scanned_volume(
    obj: &Iobj,
    _subarea: bool,
    ptmin: Ipoint,
    ptmax: Ipoint,
    doclip: i32,
    plane: &[Iplane],
    mesh_vol: &mut f64,
) -> f32 {
    let mut zmin: i32 = 0;
    let mut zmax: i32 = 0;
    let mut zlsize: i32 = 0;
    let mut nummax: i32 = 0;
    let mut numwarn: i32 = -1;
    let mut tvol: f64 = 0.;

    *mesh_vol = 0.;
    if obj.cont.is_empty() {
        return 0.;
    }

    let mut contz: Vec<i32> = Vec::new();
    let mut zlist: Vec<i32> = Vec::new();
    let mut numatz: Vec<i32> = Vec::new();
    let mut contatz: Vec<Vec<i32>> = Vec::new();
    let mut tables_obj = obj.clone();
    if imod_contour_make_z_tables(
        &mut tables_obj,
        1,
        0,
        &mut contz,
        &mut zlist,
        &mut numatz,
        &mut contatz,
        &mut zmin,
        &mut zmax,
        &mut zlsize,
        &mut nummax,
    ) != 0
    {
        return -1.;
    }

    /* Allocate space for lists of min's max's, and scan contours */
    let mut pmin: Vec<Ipoint> = vec![Ipoint::default(); nummax.max(0) as usize];
    let mut pmax: Vec<Ipoint> = vec![Ipoint::default(); nummax.max(0) as usize];
    let mut scancont: Vec<Icont> = vec![Icont::default(); nummax.max(0) as usize];
    let mut areas: Vec<f32> = vec![0.; nummax.max(0) as usize];
    let mut vol_facs: Vec<f32> = vec![0.; nummax.max(0) as usize];

    /* Get array for index to inside/outside information */
    let mut nestind: Vec<i32> = vec![0; nummax.max(0) as usize];

    for indz in 0..(zmax + 1 - zmin).max(0) {
        if ((indz + zmin) as f32) < ptmin.z || ((indz + zmin) as f32) > ptmax.z {
            continue;
        }
        let mut inbox: usize = 0;
        let mut numnests: i32 = 0;
        let mut nests: Vec<Nesting> = Vec::new();
        for kis in 0..numatz[indz as usize] as usize {
            let co = contatz[indz as usize][kis] as usize;

            if let Some((scan, lower, upper, area)) =
                contour_subarea_by_scan(&obj.cont[co], ptmin, ptmax, doclip, plane, true)
            {
                scancont[inbox] = scan.unwrap_or_default();
                pmin[inbox] = lower;
                pmax[inbox] = upper;
                areas[inbox] = area;
                vol_facs[inbox] = contour_volume_factor(obj, &obj.cont[co], ptmin, ptmax);
                nestind[inbox] = -1;
                inbox += 1;
            }
        }

        /* Look for overlapping contours as in imodmesh */
        for co in 0..inbox.saturating_sub(1) {
            for eco in (co + 1)..inbox {
                if imod_contour_check_nesting(
                    co as i32,
                    eco as i32,
                    &mut scancont[..inbox],
                    &pmin[..inbox],
                    &pmax[..inbox],
                    &mut nests,
                    &mut nestind[..inbox],
                    &mut numnests,
                    &mut numwarn,
                ) != 0
                {
                    return -1.;
                }
            }
        }

        /* Analyze inside and outside contours to determine level */
        imod_contour_nest_levels(&mut nests, &nestind[..inbox], numnests);

        /* now add up areas of non-nested and odd levels, minus even levels */
        for co in 0..inbox {
            let mut level = 1;
            if nestind[co] >= 0 {
                level = nests[nestind[co] as usize].level;
            }
            if level % 2 != 0 {
                tvol += areas[co] as f64;
                *mesh_vol += (vol_facs[co] * areas[co]) as f64;
            } else {
                tvol -= areas[co] as f64;
                *mesh_vol -= (vol_facs[co] * areas[co]) as f64;
            }
        }

        /* clean up inside the nests */
        imod_contour_free_nests(&mut nests, numnests);

        /* clean up scan conversions */
        for co in 0..inbox {
            imod_contour_delete(&mut scancont[co]);
        }
    }

    /* clean up everything else */
    imod_contour_free_z_tables(
        &mut numatz,
        &mut contatz,
        &mut contz,
        &mut zlist,
        zmin,
        zmax,
    );
    tvol as f32
}
/// Original: `contourSubareaByScan` (`imodinfo.cpp:2133`).
///
/// Returns `None` where the source returns 0 (no area to count), otherwise the
/// scan contour it created (`None` when `makeScan` is 0), the bounding box and
/// the area.
pub fn contour_subarea_by_scan(
    cont: &Icont,
    ptmin: Ipoint,
    ptmax: Ipoint,
    doclip: i32,
    plane: &[Iplane],
    make_scan: bool,
) -> Option<(Option<Icont>, Ipoint, Ipoint, f32)> {
    let mut tmpmin = Ipoint::default();
    let mut tmpmax = Ipoint::default();
    let mut clipsum = 0;
    let mut scancont: Option<Icont> = None;
    let mut area = 0f32;

    if cont.pts.is_empty() {
        return None;
    }

    /* Get limits and test in X and Y */
    imod_contour_get_bbox(Some(cont), &mut tmpmin, &mut tmpmax);
    if tmpmin.x > ptmax.x || tmpmax.x < ptmin.x || tmpmin.y > ptmax.y || tmpmax.y < ptmin.y {
        return None;
    }

    if doclip != 0 {
        /* Evaluate the corners of the bounding box: if all on
        wrong side of clip plane, skip */
        let mut corner = tmpmin;
        clipsum = imod_planes_clip(plane, plane.len() as i32, &corner);
        corner.y = tmpmax.y;
        clipsum += imod_planes_clip(plane, plane.len() as i32, &corner);
        corner.x = tmpmax.x;
        clipsum += imod_planes_clip(plane, plane.len() as i32, &corner);
        corner.y = tmpmin.y;
        clipsum += imod_planes_clip(plane, plane.len() as i32, &corner);
        if (clipsum == 0 && doclip > 0) || (clipsum == 4 && doclip < 0) {
            return None;
        }
    }

    /* Start with true area of untrimmed contour */
    area = imod_contour_area(Some(cont));

    /* If contour is not wholly inside the box, or is not all on
    the good side of the clip plane, need to trim scan
    contour down */
    if tmpmin.x < ptmin.x
        || tmpmax.x > ptmax.x
        || tmpmin.y < ptmin.y
        || tmpmax.y > ptmax.y
        || (doclip != 0 && clipsum > 0 && clipsum < 4)
    {
        let mut scan = imodel_contour_scan(Some(cont))?;

        /* need to copy the z coordinate over! */
        if !scan.pts.is_empty() {
            scan.pts[0].z = cont.pts[0].z;
        }

        let frac1 = scan_contour_area(Some(&scan));
        trim_scan_contour(&mut scan, ptmin, ptmax, doclip, plane);
        if scan.pts.is_empty() {
            /* If contour now empty, delete and skip it */
            imod_contour_delete(&mut scan);
            return None;
        }
        imod_contour_get_bbox(Some(&scan), &mut tmpmin, &mut tmpmax);

        /* Adjust true area down by ratio of trimmed to original
        scan-contour area */
        let frac2 = scan_contour_area(Some(&scan));
        if frac1 != 0. && frac2 < frac1 {
            area *= (frac2 / frac1) as f32;
        }
        scancont = Some(scan);
    }

    /* Make scan contour if necessary and requested; delete if not requested */
    if make_scan && scancont.is_none() {
        let mut scan = imodel_contour_scan(Some(cont))?;
        if !scan.pts.is_empty() {
            scan.pts[0].z = cont.pts[0].z;
        }
        scancont = Some(scan);
    } else if !make_scan && scancont.is_some() {
        imod_contour_delete(scancont.as_mut().unwrap());
        scancont = None;
    }

    Some((scancont, tmpmin, tmpmax, area))
}
/// Original: `scan_contour_area` (`imodinfo.cpp:2216`).
pub fn scan_contour_area(cont: Option<&Icont>) -> f64 {
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
        // `imodinfo.cpp:2233` reads `pts[i+1].y` at `i == psize - 1`, one past
        // the array, and only then tests `i == psize`; the bound here stops
        // one step earlier, which is the same for the paired scan contours
        // this is ever called on.
        while i + 1 < cont.pts.len() && cont.pts[i].y == cont.pts[i + 1].y {
            i += 1;
        }
        let endpt = i;

        /* check for odd amount of scans, shouldn't happen! */
        if (endpt - bgnpt) % 2 != 0 {
            let mut j = bgnpt;
            while j < endpt {
                // `xmin` and `xmax` are `int` in the source, so both ends of
                // the scan line are truncated toward zero before subtracting
                // and the difference is an integer count of pixels, not the
                // float width.
                let xmin = cont.pts[j].x as i32;
                let xmax = cont.pts[j + 1].x as i32;
                if xmin >= cont.surf {
                    pix += (xmax - xmin) as f64;
                }
                j += 1;
                j += 1;
            }
        }
        i += 1;
    }
    pix
}
/// Original: `trim_scan_contour` (`imodinfo.cpp:2261`).
///
/// `doclip` is the source's by-value parameter and it is *assigned zero*
/// inside the plane loop whenever the plane's `a` is negligible.  The loop
/// bound `ipl < (doclip ? nPlanes : 1)` re-reads it, so the first such plane
/// ends the loop -- only the planes before it ever trim anything.
pub fn trim_scan_contour(
    cont: &mut Icont,
    min: Ipoint,
    max: Ipoint,
    mut doclip: i32,
    plane: &[Iplane],
) {
    let n_planes = plane.len() as i32;

    if cont.pts.len() < 2 {
        return;
    }
    let contz = cont.pts[0].z;

    /* DNM 9/26/04: just loop on multiple planes.  Fix probable bug; take max
    of clipping crit and overall min, min of clipping crit and overall max */
    let mut ipl = 0;
    while ipl < if doclip != 0 { n_planes } else { 1 } {
        let pl = plane.get(ipl as usize).copied().unwrap_or_default();
        let mut tmin = min;
        let mut tmax = max;
        let mut ylast = 1.0e20_f32;
        if doclip != 0 {
            let pmag = ((pl.a * pl.a + pl.b * pl.b + pl.c * pl.c) as f64).sqrt() as f32;

            /* If a is small, compute y limit; but if b is small also, skip
            checking for these limits.  In either case set doclip to 0 */
            if pl.a < 1.0e-10 * pmag && pl.a > -1.0e-10 * pmag {
                if pl.b > 1.0e-10 * pmag || pl.b < -1.0e-10 * pmag {
                    let crit = -(pl.c * contz + pl.d) / pl.b;
                    if (doclip > 0 && pl.b > 0.) || (doclip < 0 && pl.b < 0.) {
                        if tmin.y < crit {
                            tmin.y = crit;
                        }
                    } else if tmax.y > crit {
                        tmax.y = crit;
                    }
                }
                doclip = 0;
            }
        }

        let mut i: i32 = 0;
        while i < cont.pts.len() as i32 {
            let yline = cont.pts[i as usize].y;
            if yline != ylast && doclip != 0 {
                /* If its a new line and clip needs to be checked, compute the
                limit in X for this Y and assign it to min or max */
                tmin = min;
                tmax = max;
                ylast = yline;
                let crit = -(pl.b * yline + pl.c * contz + pl.d) / pl.a;
                if (doclip > 0 && pl.a > 0.) || (doclip < 0 && pl.a < 0.) {
                    if tmin.x < crit {
                        tmin.x = crit;
                    }
                } else if tmax.x > crit {
                    tmax.x = crit;
                }
            }

            if (i as usize) + 1 < cont.pts.len() && yline == cont.pts[i as usize + 1].y {
                if yline < tmin.y
                    || yline > tmax.y
                    || cont.pts[i as usize].x > tmax.x
                    || cont.pts[i as usize + 1].x < tmin.x
                {
                    /* If line is out of bounds in y or x, delete 2 points */
                    imod_point_delete(cont, i);
                    imod_point_delete(cont, i);
                    i -= 1;
                } else {
                    /* otherwise, check and truncate the left and right ends
                    of the scan line */
                    if cont.pts[i as usize].x < tmin.x {
                        cont.pts[i as usize].x = tmin.x;
                    }
                    if cont.pts[i as usize + 1].x > tmax.x {
                        cont.pts[i as usize + 1].x = tmax.x;
                    }
                    i += 1;
                }
            }
            i += 1;
        }
        ipl += 1;
    }
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
        // `print_units` writes the unit name to `fout`; drive it through a
        // temporary file and read back what the source formats there.
        let path = std::env::temp_dir().join(format!(
            "imod-rs-imodinfo-print-units-{}.txt",
            std::process::id()
        ));
        let name = path.to_str().unwrap().to_string();
        for (units, expected) in [(-9, "nm"), (0, "pixels"), (2, "unknown units")] {
            FOUT.with(|fout| {
                *fout.borrow_mut() = ImodFile::open(&name, "w").unwrap();
            });
            print_units(units);
            FOUT.with(|fout| {
                let _ = fout.borrow_mut().flush();
                *fout.borrow_mut() = ImodFile::Stdout;
            });
            assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);
        }
        let _ = std::fs::remove_file(path);
    }
}
