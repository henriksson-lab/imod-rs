//! Translation of `IMOD/pysrc/transferfid`: transfers fiducials (or a
//! boundary model) from one axis of a dual-axis tilt series to the other.
//!
//! The script's top level is [`transferfid`]; its functions are
//! [`get_minimum_angle`], [`rotation_transform`], [`xf_mult`] and
//! [`xf_invert`].  The search for the best-matching pair of views is
//! `tiltmatch::search_pairs`.  `newstack`, `clipmodel`, `remapmodel`,
//! `xfmodel`, `imodtrans`, `model2point`, `imodinfo`, `beadtrack` and
//! `repackseed` are our own programs and run in process through
//! `imodpy::run_cmd`; the model pixel size (`imodinfo -h`), the count of
//! failed fiducials (`beadtrack`) and the correspondence report
//! (`repackseed`) are still read from their printed text, as the script
//! reads them.

use super::imodpy::{
    FLOAT_VALUE, INT_VALUE, MrcInfo, OptionValue, add_imod_bin_ignore_sighup,
    extract_program_entries, find_root_axis_and_extensions, fmtstr, get_mrc, option_value,
    pass_on_key_interrupt, print_pid, prnstr, py_float, py_round, py_str_float, read_text_file,
    run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_in_out_file, pip_get_integer,
    pip_get_string, pip_read_or_parse_options, python_uncaught,
};
use super::pysed::{PysedSrc, pysed};
use super::tiltmatch::{
    clean_exit_error, cleanup, get_temp_components, get_temp_names, search_pairs,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// `def getMinimumAngle(setname, src, AA, lines, nz)` (`transferfid:14`):
/// find the view with the minimum tilt angle from either the tilt file or
/// track.com.  Also return the tilt angles in an array.
fn get_minimum_angle(
    setname: &str,
    src: &str,
    aa: &str,
    lines: &[String],
    nz: i64,
    com_ext: &str,
) -> (i64, Vec<f64>) {
    let tilt_file = format!("{setname}{src}.rawtlt");
    let mut angles: Vec<f64> = Vec::new();
    if !Path::new(&tilt_file).exists() {
        // Try to find starting and increment and compute from them
        let floats = |option: &str| match option_value(lines, option, 2, false, 0, None, None) {
            Some(OptionValue::Floats(values)) if !values.is_empty() => Some(values),
            _ => None,
        };
        let first = floats("FirstTiltAngle");
        let increment = floats("TiltIncrement");
        let (Some(first), Some(increment)) = (first, increment) else {
            exit_error(&format!(
                "{tilt_file} not found - it is needed unless you enter the zero-tilt view number for {aa} with -z{src} or track{src}{com_ext} has starting angle and increment"
            ))
        };
        if increment[0].abs() < (0.000001 * first[0]).abs() {
            exit_error("Tilt increment too small to find zero tilt view number");
        }
        let zero = 1 + (-first[0] / increment[0] + 0.5).floor() as i64;
        if zero <= 0 {
            exit_error("Cannot find zero tilt view number from first angle and increment");
        }
        for iz in 0..nz {
            // Fixed in translation (BUGS.md, `transferfid`): the source
            // appends `first + iz * increment` -- the two one-element lists
            // concatenated and repeated, not an angle.
            angles.push(first[0] + iz as f64 * increment[0]);
        }
        return (zero, angles);
    }

    // Find minimum tilt angle in rawtlt file
    let ang_lines = read_text_file(&tilt_file, None, false, None).unwrap_or_default();
    let mut amin = 1.0e20_f64;
    let mut zero: i64 = -1;
    for (i, line) in ang_lines.iter().enumerate() {
        if !line.trim().is_empty() {
            let Some(value) = py_float(line) else {
                exit_error(&format!(
                    "Converting lines in {tilt_file} to floating point values"
                ))
            };
            angles.push(value);
            let ang = value.abs();
            if ang < amin {
                amin = ang;
                zero = i as i64 + 1;
            }
        }
    }
    if zero <= 0 {
        exit_error(&format!(
            "Cannot find a minimum tilt angle from {tilt_file}"
        ));
    }
    (zero, angles)
}

/// `def rotationTransform(angle)` (`transferfid:57`): a rotation matrix in
/// the 6-element form xpx, xpy, ypx, ypy, dx, dy.
fn rotation_transform(angle: f64) -> [f64; 6] {
    let sina = angle.to_radians().sin();
    let cosa = angle.to_radians().cos();
    [cosa, -sina, sina, cosa, 0., 0.]
}

/// `def xfMult(f1, f2)` (`transferfid:63`): multiply two matrices, where f1
/// is applied first, f2 second.
fn xf_mult(f1: &[f64; 6], f2: &[f64; 6]) -> [f64; 6] {
    let mut tmp = [0.0_f64; 6];
    tmp[0] = (f2[0] * f1[0]) + (f2[2] * f1[1]);
    tmp[1] = (f2[1] * f1[0]) + (f2[3] * f1[1]);
    tmp[2] = (f2[0] * f1[2]) + (f2[2] * f1[3]);
    tmp[3] = (f2[1] * f1[2]) + (f2[3] * f1[3]);
    tmp[4] = (f2[0] * f1[4]) + (f2[2] * f1[5]) + f2[4];
    tmp[5] = (f2[1] * f1[4]) + (f2[3] * f1[5]) + f2[5];
    tmp
}

/// `def xfInvert(f)` (`transferfid:74`): return the inverse of a matrix.
/// Defined in the source and never called.
#[allow(dead_code)]
fn xf_invert(f: &[f64; 6]) -> [f64; 6] {
    let mut tmp = [0.0_f64; 6];
    let denom = f[0] * f[3] - f[2] * f[1];
    tmp[0] = f[3] / denom;
    tmp[2] = -f[2] / denom;
    tmp[1] = -f[1] / denom;
    tmp[3] = f[0] / denom;
    tmp[4] = -(tmp[0] * f[4] + tmp[2] * f[5]);
    tmp[5] = -(tmp[1] * f[4] + tmp[3] * f[5]);
    tmp
}

/// The script's top level (`transferfid:86-516`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn transferfid(arguments: &[OsString]) -> i32 {
    let progname = "transferfid";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 transferfid
    let options: Vec<String> = [
        "s:Setname:CH:",
        "b:TransferBtoA:B:",
        "ia:AImageFile:FN:",
        "ib:BImageFile:FN:",
        "f:FiducialModel:FN:",
        "o:SeedModel:FN:",
        "boundary:BoundaryModel:FN:",
        "n:ViewsToSearch:I:",
        "za:ACenterView:I:",
        "zb:BCenterView:I:",
        "a:AngleOfRotation:I:",
        "x:MirrorXaxis:I:",
        "m:RunMidas:B:",
        "scan:ScanRotationMaxAndStep:FP:",
        "c:CorrespondingCoordFile:FN:",
        "lowest:LowestTiltTransformFile:FN:",
        "t:LeaveTempFiles:B:",
        ":PID:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 0);
    pass_on_key_interrupt(true);

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    // Set names of temp files, this gets pid, tmpRoot, and tmpDir in the module globals
    let tmp_minxf = get_temp_names(progname);
    let (tmp_root, _tmp_dir, pid) = get_temp_components();

    let tmp_stack = format!("{tmp_root}stack{pid}");
    let tmp_twoxf = format!("{tmp_root}twoxf{pid}");
    let tmp_clip = format!("{tmp_root}clip{pid}");
    let tmp_xfmod = format!("{tmp_root}xfmod{pid}");
    let tmp_seed = format!("{tmp_root}seed{pid}");
    let tmp_map1 = format!("{tmp_root}map1{pid}");
    let tmp_map2 = format!("{tmp_root}map2{pid}");
    let tmp_map3 = format!("{tmp_root}map3{pid}");
    let tmp_trans = format!("{tmp_root}trans{pid}");

    let setname = pip_get_in_out_file("Setname", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if setname.is_empty() {
        exit_error("You must enter the setname (root name of dataset)");
    }

    let mut src = "a";
    let mut dst = "b";
    let mut aa = "A";
    let mut bb = "B";
    let if_b_to_a = pip_get_boolean("TransferBtoA", 0).unwrap_or(0);
    if if_b_to_a != 0 {
        src = "b";
        dst = "a";
        aa = "B";
        bb = "A";
    }

    // Get the com extension and if this fails, get it specifically from track files
    let (mut com_ext, _dual_num, _setroot, _type_ext, _stack_ext) =
        find_root_axis_and_extensions(-1, None);
    if com_ext.is_empty() {
        // Fixed in translation (BUGS.md, `transferfid`): the source tests
        // `trackb.pcm` for the .com flag and `tracka.com` for the .pcm flag.
        let acom_exists = Path::new("tracka.com").exists();
        let bcom_exists = Path::new("trackb.com").exists();
        let apcm_exists = Path::new("tracka.pcm").exists();
        let bpcm_exists = Path::new("trackb.pcm").exists();
        if acom_exists && bcom_exists && !(apcm_exists && bpcm_exists) {
            com_ext = "com".to_owned();
        } else if !(acom_exists && bcom_exists) && apcm_exists && bpcm_exists {
            com_ext = "pcm".to_owned();
        } else {
            exit_error("Cannot determine command file extension: both .com and .pcm files exist");
        }
    }

    let com_ext = format!(".{com_ext}");

    let mut out_file = format!("{setname}{dst}.seed");
    let mut image_a = pip_get_string("AImageFile", "").unwrap_or_default();
    let mut image_b = pip_get_string("BImageFile", "").unwrap_or_default();
    let fid_default = format!("{setname}{src}.fid");
    let fid_file = pip_get_string("FiducialModel", &fid_default).unwrap_or(fid_default);
    let if_fid_file = 1 - pip_get_err_no();
    let correspond = pip_get_string("CorrespondingCoordFile", "").unwrap_or_default();

    // Get possible boundary file, check if conflicts, get default output file
    let boundary_file = pip_get_string("BoundaryModel", "").unwrap_or_default();
    if !boundary_file.is_empty() {
        if if_fid_file != 0 || !correspond.is_empty() {
            exit_error("You cannot enter a fiducial model or the -c option with a boundary model");
        }
        out_file = String::new();
        let from = format!("{setname}{src}");
        if boundary_file.starts_with(&from) {
            out_file = boundary_file.replacen(&from, &format!("{setname}{dst}"), 1);
        }
    }

    let out_file = pip_get_string("SeedModel", &out_file).unwrap_or(out_file);
    if out_file.is_empty() {
        exit_error(&format!(
            "You must enter an output file with -o; the boundary model name does not start with {setname}{src}"
        ));
    }

    let nviews = pip_get_integer("ViewsToSearch", 5).unwrap_or(5) as i64;
    let mut zero_a = pip_get_integer("ACenterView", -1).unwrap_or(-1) as i64;
    let mut zero_b = pip_get_integer("BCenterView", -1).unwrap_or(-1) as i64;
    let lowest_xf_file = pip_get_string("LowestTiltTransformFile", "").unwrap_or_default();
    if nviews < 0 {
        exit_error("The number of views to sample must be positive");
    }

    // swap inputs for filename and center z's if going backwards
    if if_b_to_a != 0 {
        std::mem::swap(&mut image_a, &mut image_b);
        std::mem::swap(&mut zero_a, &mut zero_b);
    }

    // Get the A track command file and insist it be PIP version; get A image file if needed
    let tracka = format!("track{src}{com_ext}");
    if !Path::new(&tracka).exists() {
        exit_error(&format!("Cannot find {tracka} command file"));
    }
    let track_lines = read_text_file(&tracka, None, false, None).unwrap_or_default();

    let Some(track_lines) = extract_program_entries(&track_lines, "beadtrack", "-Standard") else {
        exit_error(&format!(
            "Old version of {tracka} cannot be used; convert it by opening and closing the fiducial tracking panel in etomo"
        ))
    };

    let string_value =
        |lines: &[String], option: &str| match option_value(lines, option, 0, false, 0, None, None)
        {
            Some(OptionValue::String(value)) => value,
            _ => String::new(),
        };
    if image_a.is_empty() {
        image_a = string_value(&track_lines, "ImageFile");
    }

    // Get B image file from trackb.com if needed
    let mut b_lines: Vec<String> = Vec::new();
    if image_b.is_empty() || zero_b < 0 || !lowest_xf_file.is_empty() || !boundary_file.is_empty() {
        let trackb = format!("track{dst}{com_ext}");
        if !Path::new(&trackb).exists() {
            exit_error(&format!(
                "Cannot find {trackb} command file; it is needed unless you enter the {bb} image file with -i{dst}"
            ));
        }
        b_lines = read_text_file(&trackb, None, false, None).unwrap_or_default();
        if image_b.is_empty() {
            image_b = string_value(&b_lines, "ImageFile");
        }
        if image_b.is_empty() {
            exit_error(&format!("Cannot find the {bb} image file name in {trackb}"));
        }
    }

    // Make sure image files exist and fid file too
    for imfile in [&image_a, &image_b] {
        if !Path::new(imfile).exists() {
            exit_error(&format!("Image file {imfile} does not exist"));
        }
    }

    if !boundary_file.is_empty() && !Path::new(&boundary_file).exists() {
        exit_error(&format!(
            "Boundary model file {boundary_file} does not exist"
        ));
    } else if !Path::new(&fid_file).exists() {
        exit_error(&format!("Fiducial file {fid_file} does not exist"));
    }

    // Get image sizes and pixel size; an error here is uncaught in the source
    let full_header = |file: &str| match get_mrc(file, true, false) {
        Ok(MrcInfo::All(nx, ny, nz, _, px, _, _, ox, oy, oz, ..)) => {
            (nx as i64, ny as i64, nz as i64, px, ox, oy, oz)
        }
        Ok(_) => unreachable!(),
        Err(error) => python_uncaught(&format!("imodpy.ImodpyError: {error}")),
    };
    let (nxa, nya, nza, pxa, oxa, oya, oza) = full_header(&image_a);
    let (nxb, nyb, nzb, pxb, oxb, oyb, ozb) = full_header(&image_b);
    // `expandAfac` is the int 1 or a float, written with `str()`
    let mut expand_afac = 1.0_f64;
    let mut expand_afac_text = "1".to_owned();
    if ((pxb - pxa) / pxa).abs() > 0.025 {
        prnstr(
            &format!(
                "WARNING: - {progname}: Pixel sizes do not match: {image_a} = {}, {image_b} = {}; scaling to compensate",
                py_str_float(pxa),
                py_str_float(pxb)
            ),
            "\n",
            false,
        );
        prnstr("", "\n", false);
        expand_afac = pxa / pxb;
        expand_afac_text = py_str_float(expand_afac);
    }

    // Get the view at minimum tilt if needed for one reason or another
    let mut zero_a_view = zero_a;
    let mut zero_b_view = zero_b;
    let mut angles_a: Vec<f64> = Vec::new();
    let mut angles_b: Vec<f64> = Vec::new();
    if zero_a < 0 || !lowest_xf_file.is_empty() || !boundary_file.is_empty() {
        (zero_a_view, angles_a) = get_minimum_angle(&setname, src, aa, &track_lines, nza, &com_ext);
    }
    if zero_b < 0 || !lowest_xf_file.is_empty() || !boundary_file.is_empty() {
        (zero_b_view, angles_b) = get_minimum_angle(&setname, dst, bb, &b_lines, nzb, &com_ext);
    }
    if zero_a < 0 {
        zero_a = zero_a_view;
    }
    if zero_b < 0 {
        zero_b = zero_b_view;
    }

    zero_a -= 1;
    zero_b -= 1;

    let asec_start = zero_a - nviews.div_euclid(2);
    let asec_end = asec_start + nviews - 1;
    let bsec_start = zero_b - nviews.div_euclid(2);
    let bsec_end = bsec_start + nviews - 1;
    let lowest_asec = asec_end.min(asec_start.max(zero_a_view - 1));
    let lowest_bsec = bsec_end.min(bsec_start.max(zero_b_view - 1));

    // Check section numbers
    if asec_start < 0 || asec_end >= nza {
        exit_error(&format!(
            "The starting or ending section numbers for {aa} are out of range ({asec_start} and {asec_end})"
        ));
    }
    if bsec_start < 0 || bsec_end >= nzb {
        exit_error(&format!(
            "The starting or ending section numbers for {bb} are out of range ({bsec_start} and {bsec_end})"
        ));
    }

    let (asec_best, bsec_best, _junk1, _junk2) = search_pairs(
        progname,
        zero_a,
        zero_b,
        nviews,
        nviews,
        &image_a,
        &image_b,
        nxa,
        nxb,
        nya,
        nyb,
        aa,
        bb,
        &lowest_xf_file,
        lowest_asec,
        lowest_bsec,
        "",
        0,
        0,
        &expand_afac_text,
    );

    // A `runcmd` the source does not guard: its ImodpyError is uncaught
    let unguarded = |command: &str| {
        if let Err(error) = run_cmd(command, None, None, None, &[]) {
            python_uncaught(&format!("imodpy.ImodpyError: {error}"));
        }
    };

    // Find the pixel size of the model and a scale factor
    let mut mod_file = fid_file.clone();
    if !boundary_file.is_empty() {
        mod_file = boundary_file.clone();
    }
    let info_lines = match run_cmd(
        &format!("imodinfo -h \"{mod_file}\""),
        None,
        None,
        None,
        &[],
    ) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => clean_exit_error("Extracting pixel size from model"),
    };
    let mut mod_pixel = 0.0_f64;
    for l in &info_lines {
        if l.contains("SCALE  =") {
            let l = l.replace([',', '(', ')'], "");
            mod_pixel = l.split_whitespace().nth(3).and_then(py_float).unwrap_or(0.);
            break;
        }
    }

    if mod_pixel == 0. {
        cleanup();
        exit_error("Getting model scale value");
    }

    let mod_scale = pxa / mod_pixel;

    // Get the best transform
    let mut minxf: Vec<String> = Vec::new();
    if Path::new(&tmp_minxf).exists() {
        minxf = read_text_file(&tmp_minxf, None, false, None).unwrap_or_default();
    }
    if minxf.is_empty() {
        cleanup();
        exit_error("No alignment was computed, cannot continue");
    }

    // do what is needed with boundary file
    let zadd: String;
    let xf_input_mod: String;
    let map_output_mod: String;
    let stack_for_trans: String;
    let new_opt: &str;
    if !boundary_file.is_empty() {
        // Need to find view with most points
        unguarded(&fmtstr(
            "model2point \"{}\" \"{}\"",
            &[boundary_file.clone(), tmp_map1.clone()],
        ));
        let bound_lines = read_text_file(&tmp_map1, None, false, None).unwrap_or_default();
        if bound_lines.is_empty() {
            exit_error("The boundary model has no points");
        }
        let mut nearest_z: i64 = -100;
        for line in &bound_lines {
            let Some(value) = line.split_whitespace().nth(2).and_then(py_float) else {
                exit_error(&format!(
                    "Reading Z coordinate from line in point file: {line}"
                ))
            };
            let ptz = py_round(value) as i64;
            if ((ptz - asec_best) as f64).abs() < ((nearest_z - asec_best) as f64).abs() {
                nearest_z = ptz;
            }
        }

        // Set up transform and a starting matrix
        let axis_rotation = match option_value(
            &track_lines,
            "RotationAngle",
            FLOAT_VALUE,
            true,
            1,
            None,
            None,
        ) {
            Some(OptionValue::Floats(values)) => Some(values[0]),
            _ => None,
        };
        let xfsplit: Vec<&str> = minxf[0].split_whitespace().collect();
        let mut a_to_bmat = [0.0_f64; 6];
        for (slot, index) in a_to_bmat.iter_mut().zip(0..6) {
            match xfsplit.get(index).and_then(|text| py_float(text)) {
                Some(value) => *slot = value,
                None => exit_error("Converting transform values to floats"),
            }
        }

        let mut base_mat = [1., 0., 0., 1., 0., 0.];

        // If angles are available, apply a stretch perpendicular to X by the ratio of the
        // cosines of asecBest and the nearest view with contours
        if angles_a.len() as i64 > asec_best.max(nearest_z) && angles_b.len() as i64 > bsec_best {
            if let Some(axis_rotation) = axis_rotation {
                // A negative index counts from the end of a Python list
                let angle = |index: i64| -> f64 {
                    let at = if index < 0 {
                        index + angles_a.len() as i64
                    } else {
                        index
                    };
                    match angles_a.get(at as usize) {
                        Some(value) if at >= 0 => *value,
                        _ => python_uncaught("IndexError: list index out of range"),
                    }
                };
                base_mat = rotation_transform(-axis_rotation);
                base_mat[0] *=
                    angle(asec_best).to_radians().cos() / angle(nearest_z).to_radians().cos();

                base_mat = xf_mult(&base_mat, &rotation_transform(axis_rotation));
            }
        }

        // Get the full transform and write it
        let full_mat = xf_mult(&base_mat, &a_to_bmat);
        let _ = write_text_file(
            &tmp_twoxf,
            &[full_mat
                .iter()
                .map(|value| py_str_float(*value))
                .collect::<Vec<_>>()
                .join("  ")],
            false,
        );
        zadd = (bsec_best - asec_best).to_string();
        xf_input_mod = boundary_file.clone();
        map_output_mod = out_file.clone();
        stack_for_trans = image_b.clone();
        new_opt = "";
    } else {
        // Fiducial transfer operations
        prnstr(
            &format!(
                "Transferring fiducials from view {} in {aa} to view {} in {bb} with Beadtrack:",
                asec_best + 1,
                bsec_best + 1
            ),
            "\n",
            false,
        );
        prnstr("              (Type Ctrl-C to interrupt)", "\n", false);

        // Stack the two best sections
        zadd = "0.0".to_owned();
        xf_input_mod = tmp_map1.clone();
        map_output_mod = tmp_map2.clone();
        stack_for_trans = tmp_stack.clone();
        new_opt = "-new 1";
        let _ = write_text_file(
            &tmp_twoxf,
            &["1 0 0 1 0 0".to_owned(), minxf[0].clone()],
            false,
        );
        if run_cmd(
            &fmtstr(
                "newstack -sec {} -sec {} -xform {} -use 0,1 -float 2 \"{}\" \"{}\" \"{}\"",
                &[
                    bsec_best.to_string(),
                    asec_best.to_string(),
                    tmp_twoxf.clone(),
                    image_b.clone(),
                    image_a.clone(),
                    tmp_stack.clone(),
                ],
            ),
            None,
            None,
            None,
            &[],
        )
        .is_err()
        {
            clean_exit_error("Stacking two best views");
        }

        // clip out the model and remap it to z = 1
        let clipcom = vec![
            format!("InputFile {fid_file}"),
            format!("OutputFile {tmp_clip}"),
            format!(
                "ZMinAndMax {},{}",
                py_str_float(asec_best as f64 - 0.5),
                py_str_float(asec_best as f64 + 0.5)
            ),
            "KeepEmptyContours".to_owned(),
        ];
        if run_cmd("clipmodel -StandardInput", Some(&clipcom), None, None, &[]).is_err() {
            clean_exit_error(&format!("Clipping out best view from {aa} fiducial model"));
        }
        if run_cmd(
            &format!("remapmodel -new 1 \"{tmp_clip}\" \"{tmp_map1}\""),
            None,
            None,
            None,
            &[],
        )
        .is_err()
        {
            clean_exit_error(&format!("Remapping {aa} fiducials to section 1"));
        }
    }

    // Common operations
    // transform model then adjust its coordinates to new center
    let xadd = mod_scale * (nxb - nxa) as f64 / 2.;
    let yadd = mod_scale * (nyb - nya) as f64 / 2.;
    if run_cmd(
        &format!(
            "xfmodel -xforms \"{tmp_twoxf}\" -scale {} \"{xf_input_mod}\" \"{tmp_xfmod}\"",
            py_str_float(mod_scale)
        ),
        None,
        None,
        None,
        &[],
    )
    .is_err()
    {
        clean_exit_error(&format!("Transforming {aa} model to match {bb} image"));
    }

    // Need to modify the image reference data in the model to match B if there was scaling
    // or if the origins are not the same between the files
    let mut remap_in = tmp_xfmod.clone();
    if expand_afac != 1. || oxa != oxb || oya != oyb || oza != ozb {
        remap_in = tmp_trans.clone();
        if run_cmd(
            &format!("imodtrans -I \"{stack_for_trans}\" \"{tmp_xfmod}\" \"{tmp_trans}\""),
            None,
            None,
            None,
            &[],
        )
        .is_err()
        {
            clean_exit_error("Changing scale information in transformed model");
        }
    }

    if run_cmd(
        &format!(
            "remapmodel {new_opt} -add {},{},{zadd} \"{remap_in}\" \"{map_output_mod}\"",
            py_str_float(xadd),
            py_str_float(yadd)
        ),
        None,
        None,
        None,
        &[],
    )
    .is_err()
    {
        clean_exit_error(&format!("Recentering transformed {aa} model"));
    }

    // Done if it is a boundary model
    if !boundary_file.is_empty() {
        cleanup();
        return done(0);
    }

    // Prepare the blendmont command; keep tracking parameters but modify for two untilted
    // images
    let sedcom: Vec<String> = vec![
        format!("?^ImageFile?s?[ \t].*? {tmp_stack}?"),
        format!("?^InputSeedModel?s?[ \t].*? {tmp_map2}?"),
        format!("?^OutputModel?s?[ \t].*? {tmp_seed}?"),
        "?^RotationAngle?s?[ \t].*? 0?".to_owned(),
        "?^FirstTiltAngle?d".to_owned(),
        "?^TiltIncrement?d".to_owned(),
        "?^TiltFile?d".to_owned(),
        "?^TiltAngles?d".to_owned(),
        "?^SkipViews?d".to_owned(),
        "?^ShiftsNearZeroTilt?d".to_owned(),
        "?^SeparateGroup?d".to_owned(),
        "?^RoundsOfTracking?s?[ \t].*? 1?".to_owned(),
        "?^RotationAngle?a?TiltAngles 0,0?".to_owned(),
    ];
    let mut sedlines = match pysed(
        &sedcom,
        PysedSrc::Lines(&track_lines),
        None,
        false,
        '?',
        false,
    ) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(message) => exit_error(&message),
    };

    // If there is local tracking, definitely track objects together
    // Fixed in translation (BUGS.md, `transferfid`): the source reads the
    // entry as a string (so `'0'` counts as on) and appends the option to
    // `sedcom` after `pysed` has already run, so it never reaches beadtrack.
    if let Some(OptionValue::Integers(iflocal)) = option_value(
        &track_lines,
        "LocalAreaTracking",
        INT_VALUE,
        false,
        0,
        None,
        None,
    ) {
        if iflocal.first().is_some_and(|value| *value != 0) {
            sedlines.push("TrackObjectsTogether".to_owned());
        }
    }

    let tracklog = match run_cmd("beadtrack -StandardInput", Some(&sedlines), None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => clean_exit_error(&format!(
            "Running Beadtrack to get fiducials onto {bb} view"
        )),
    };

    match tracklog
        .last()
        .and_then(|line| line.split_whitespace().last().map(str::to_owned))
    {
        Some(btnum) => prnstr(
            &format!("Number of fiducials that failed to transfer: {btnum}"),
            "\n",
            false,
        ),
        None => {
            cleanup();
            exit_error("Finding # failed message in track output");
        }
    }

    // Remap seed model to the section in B
    if run_cmd(
        &format!("remapmodel -new {bsec_best},-999 \"{tmp_seed}\" \"{tmp_map3}\""),
        None,
        None,
        None,
        &[],
    )
    .is_err()
    {
        clean_exit_error(&format!("Remapping seed model up to view in {bb}"));
    }

    // Repack the model to remove empty points, and pass through mapping report
    // First find out if the fid.xyz is available and has contour data
    let mut xyz_name = format!("{setname}{src}fid.xyz");
    if Path::new(&xyz_name).exists() {
        let xyzlines = read_text_file(&xyz_name, None, false, None).unwrap_or_default();
        let Some(last) = xyzlines.last() else {
            python_uncaught("IndexError: list index out of range")
        };
        if last.split_whitespace().count() < 6 {
            xyz_name = String::new();
        }
    } else {
        xyz_name = String::new();
    }

    let comlines = vec![
        fid_file.clone(),
        xyz_name,
        tmp_map3.clone(),
        out_file.clone(),
        correspond.clone(),
        format!("{asec_best},{bsec_best},{if_b_to_a}"),
    ];
    let rep_lines = match run_cmd("repackseed", Some(&comlines), None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => clean_exit_error("Repacking seed model and establishing correspondence"),
    };
    let mut do_out = false;
    for l in &rep_lines {
        do_out = do_out || l.contains("follow");
        if do_out {
            prnstr(l.trim(), "\n", false);
        }
    }

    cleanup();
    done(0)
}
