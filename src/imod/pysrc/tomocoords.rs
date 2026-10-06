//! Translation of `IMOD/pysrc/tomocoords.py`: functions shared by
//! `subtomosetup`, `rawtiltcoords` and `ctf3dsetup` for getting the common
//! options, reading point positions and file headers, and working out CTF and
//! raw-stack parameters.
//!
//! `imodinfo`, `imodextract`, `model2point`, `header` and `xfmodel` are run
//! through [`run_cmd`], which runs this crate's own programs in process where
//! the command line allows.  `getmrc`/`getmrcsize` read the header in process.
//!
//! Python values keep their types: positions, pixel sizes and angles are
//! floats (doubles); `optionValue(..., numVal = 1)` scalars come back as the
//! first element of the one-element list [`option_value`] returns.

use super::imodpy::{
    FLOAT_VALUE, MrcInfo, OptionValue, STRING_VALUE, cleanup_files, complete_and_check_com_file,
    exit_from_imod_error, fmtstr, get_mrc, get_mrc_size, option_value, os_path_splitext, prnstr,
    py_fixed, py_float, py_int, py_round, py_str_float, read_text_file, run_cmd,
};
use super::pip::{
    exit_error, pip_get_err_no, pip_get_float, pip_get_integer, pip_get_string,
    pip_read_or_parse_options,
};
use regex::Regex;
use std::path::Path;

/// `yzRatioCrit` (`tomocoords.py:9`).
const YZ_RATIO_CRIT: f64 = 2.;

/// The tuple `getCommonOptions` returns (`tomocoords.py:53-54`).
pub struct CommonOptions {
    pub root_name: String,
    pub vol_name: String,
    pub center_file: String,
    pub point_file: String,
    pub model_file: String,
    pub obj_list: String,
    pub com_file: String,
    pub reorient_in: i32,
    pub entered_orient: i32,
}

/// The tuple `getPointsAndHeaders` returns (`tomocoords.py:228-234`), field
/// for field.  The sizes are Python ints, the pixel sizes and origins floats.
pub struct PointsAndHeaders {
    pub pid: String,
    pub clean_list: Vec<String>,
    pub point_list: Vec<[f64; 3]>,
    pub ali_binning: i32,
    pub reorient: i32,
    pub com_lines: Vec<String>,
    pub nx_raw: i32,
    pub ny_raw: i32,
    pub nz_raw: i32,
    pub pix_x_raw: f64,
    pub pix_y_raw: f64,
    pub pix_z_raw: f64,
    pub nx_ali: i32,
    pub ny_ali: i32,
    pub nz_ali: i32,
    pub pix_x_ali: f64,
    pub pix_y_ali: f64,
    pub pix_z_ali: f64,
    pub orig_x_ali: f64,
    pub orig_y_ali: f64,
    pub orig_z_ali: f64,
    pub nx_vol: i32,
    pub ny_vol: i32,
    pub nz_vol: i32,
    pub pix_x_vol: f64,
    pub pix_y_vol: f64,
    pub pix_z_vol: f64,
    pub orig_x_vol: f64,
    pub orig_y_vol: f64,
    pub orig_z_vol: f64,
    pub nx_full: i32,
    pub ny_full: i32,
    pub nz_full: i32,
    pub pix_x_full: f64,
    pub pix_y_full: f64,
    pub pix_z_full: f64,
    pub orig_x_full: f64,
    pub orig_y_full: f64,
    pub orig_z_full: f64,
}

/// The tuple `getCTFoptionsCheckIfCorrected` returns (`tomocoords.py:273-274`).
/// `None` stands for the Python `None` an absent option gives.
pub struct CtfOptions {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub xaxis_tilt: f64,
    pub tilt_use_gpu: i32,
    pub ctf_xtilt: Option<f64>,
    pub ctf_use_gpu: Option<i32>,
    pub ctf_pix_size: Option<f64>,
    pub ctf_input: Option<String>,
    pub ctf_output: Option<String>,
}

/// Matches `getCommonOptions` (`tomocoords.py:16`): do initial option startup
/// and get the options that are common between the programs.
pub fn get_common_options(argv: &[String], options: &[String], progname: &str) -> CommonOptions {
    let (_opts, _nonopts) = pip_read_or_parse_options(argv, options, progname, 1, 0, 1);

    // Get the options
    let root_name = pip_get_string("RootName", "").unwrap_or_default();
    if root_name.is_empty() {
        exit_error("A root name must be entered");
    }
    let vol_name = pip_get_string("VolumeModeled", "").unwrap_or_default();
    if vol_name.is_empty() {
        exit_error("The name of the volume on which points were picked must be entered");
    }

    // Get position file and determine if it is model or point file
    let center_file = pip_get_string("CenterPositionFile", "").unwrap_or_default();
    if center_file.is_empty() {
        exit_error("You must enter a model or point file name with -center");
    }
    let mut point_file = String::new();
    let mut model_file = String::new();
    match run_cmd(
        &format!("imodinfo -h \"{center_file}\""),
        None,
        None,
        Some("stdout"),
        &[],
    ) {
        Ok(info_lines) => {
            for line in info_lines.unwrap_or_default() {
                if (line.contains("Error") && line.contains("eading imod model"))
                    || line.contains("Model has no objects")
                {
                    point_file = center_file.clone();
                    break;
                }
            }
        }
        Err(_) => point_file = center_file.clone(),
    }

    if point_file.is_empty() {
        model_file = center_file.clone();
    }

    let obj_list = pip_get_string("ObjectsToUse", "").unwrap_or_default();
    if !obj_list.is_empty() && !point_file.is_empty() {
        exit_error("You cannot enter a list of objects with a point file");
    }

    let com_file = pip_get_string("CommandFile", "tilt.com").unwrap_or_default();
    let reorient_in = pip_get_integer("ReorientionType", 0).unwrap_or(0);
    let entered_orient = 1 - pip_get_err_no();
    CommonOptions {
        root_name,
        vol_name,
        center_file,
        point_file,
        model_file,
        obj_list,
        com_file,
        reorient_in,
        entered_orient,
    }
}

/// Matches `getPointsAndHeaders` (`tomocoords.py:59`): get the coordinates
/// from a model or point file, read the headers of all relevant files, and
/// work out the orientation with all the needed messages.
#[allow(clippy::too_many_arguments)]
pub fn get_points_and_headers(
    model_file: &str,
    obj_list: &str,
    point_file: &str,
    progname: &str,
    stack_name: &str,
    ali_name: &str,
    vol_name: &str,
    full_rec: Option<&str>,
    entered_orient: i32,
    reorient: i32,
    com_file: &str,
    use_raw: bool,
) -> PointsAndHeaders {
    let mut reorient = reorient;

    // Get the coordinate list
    let pid = format!(".{}", std::process::id());
    let mut clean_list: Vec<String> = Vec::new();
    let mut descrip = "";
    let mut point_file = point_file.to_owned();
    if !model_file.is_empty() {
        let mut mod_convert = model_file.to_owned();
        descrip = "temporary";

        // Extract objects
        if !obj_list.is_empty() {
            mod_convert = format!("{model_file}.obj{pid}");
            clean_list.push(mod_convert.clone());
            if run_cmd(
                &fmtstr(
                    "imodextract \"{}\" \"{}\" \"{}\"",
                    &[
                        obj_list.to_owned(),
                        model_file.to_owned(),
                        mod_convert.clone(),
                    ],
                ),
                None,
                None,
                None,
                &[],
            )
            .is_err()
            {
                cleanup_files(&clean_list);
                exit_from_imod_error(progname);
            }
        }

        // Convert to point list, with -scale option to compensate for subset loading
        point_file = format!("{model_file}.pt{pid}");
        clean_list.push(point_file.clone());
        if run_cmd(
            &fmtstr(
                "model2point -scale -float \"{}\" \"{}\"",
                &[mod_convert.clone(), point_file.clone()],
            ),
            None,
            None,
            None,
            &[],
        )
        .is_err()
        {
            cleanup_files(&clean_list);
            exit_from_imod_error(progname);
        }
    }

    // Now read in the point file and process the lines
    let point_lines = read_text_file(
        &point_file,
        Some(&format!("{descrip} point file")),
        true,
        None,
    );
    cleanup_files(&clean_list);
    let point_lines = match point_lines {
        Ok(lines) => lines,
        Err(message) => exit_error(&message),
    };

    let mut point_list: Vec<[f64; 3]> = Vec::new();
    for line in &point_lines {
        let lsplit: Vec<&str> = line.split_whitespace().collect();
        if lsplit.len() < 3 {
            exit_error(&format!(
                "There are not three values on the line in {descrip} point file: {line}"
            ));
        }
        match (
            py_float(lsplit[0]),
            py_float(lsplit[1]),
            py_float(lsplit[2]),
        ) {
            (Some(x), Some(y), Some(z)) => point_list.push([x, y, z]),
            _ => exit_error(&format!(
                "Converting a value to a floating point number on the line in {descrip} point file: {line}"
            )),
        }
    }

    // Get the file headers
    // Return raw stack info for aligned if using raw
    let all = |file: &str| -> (i32, i32, i32, f64, f64, f64, f64, f64, f64) {
        match get_mrc(file, true, false) {
            Ok(MrcInfo::All(nx, ny, nz, _mode, px, py, pz, ox, oy, oz, _, _, _)) => {
                (nx, ny, nz, px, py, pz, ox, oy, oz)
            }
            _ => exit_from_imod_error(progname),
        }
    };
    let (
        nx_raw,
        ny_raw,
        nz_raw,
        pix_x_raw,
        pix_y_raw,
        pix_z_raw,
        mut orig_x_ali,
        mut orig_y_ali,
        mut orig_z_ali,
    ) = all(stack_name);
    let (nx_ali, ny_ali, nz_ali, pix_x_ali, pix_y_ali, pix_z_ali);
    if use_raw {
        (nx_ali, ny_ali, nz_ali, pix_x_ali, pix_y_ali, pix_z_ali) =
            (nx_raw, ny_raw, nz_raw, pix_x_raw, pix_y_raw, pix_z_raw);
    } else {
        let values = all(ali_name);
        (nx_ali, ny_ali, nz_ali, pix_x_ali, pix_y_ali, pix_z_ali) =
            (values.0, values.1, values.2, values.3, values.4, values.5);
        (orig_x_ali, orig_y_ali, orig_z_ali) = (values.6, values.7, values.8);
    }
    let (
        nx_vol,
        ny_vol,
        nz_vol,
        pix_x_vol,
        pix_y_vol,
        pix_z_vol,
        orig_x_vol,
        orig_y_vol,
        orig_z_vol,
    ) = all(vol_name);
    let (
        nx_full,
        ny_full,
        nz_full,
        pix_x_full,
        pix_y_full,
        pix_z_full,
        orig_x_full,
        orig_y_full,
        orig_z_full,
    ) = match full_rec.filter(|name| !name.is_empty()) {
        Some(name) => all(name),
        None => (0, 0, 0, 0., 0., 0., 0., 0., 0.),
    };

    let head_lines = match run_cmd(&format!("header \"{vol_name}\""), None, None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => exit_from_imod_error(progname),
    };
    // `int(round(pixXali / pixXraw))`.  `BUGS.md` (tomocoords
    // `getPointsAndHeaders`), fixed in translation: a zero pixel size in the
    // raw stack header raises a ZeroDivisionError that this `except
    // ImodpyError` does not catch, ending in a traceback; it is reported as
    // an error instead.
    if pix_x_raw == 0. {
        exit_error(&format!(
            "The pixel size in the header of {stack_name} is zero"
        ));
    }
    let ali_binning = py_round(pix_x_ali / pix_x_raw) as i32;

    // Deduce post-processing of the volume: look for tilt angles and clip flipyz
    let mut tilt_x_orig = -999.0_f64;
    let mut tilt_x_cur = 0.0_f64;
    let mut orient_title = -2;
    for line in &head_lines {
        if line.contains("ilt angles") && tilt_x_orig < -990. {
            let lsplit: Vec<&str> = line.split_whitespace().collect();
            if lsplit.len() < 9 {
                exit_error(&format!(
                    "Tilt angles line in header output from {vol_name} has too few items"
                ));
            }
            match (
                py_float(lsplit[lsplit.len() - 6]),
                py_float(lsplit[lsplit.len() - 3]),
            ) {
                (Some(orig), Some(cur)) => {
                    tilt_x_orig = orig;
                    tilt_x_cur = cur;
                }
                _ => exit_error(&format!(
                    "Converting tilt angles to floating point values in header output from {vol_name}"
                )),
            }
        }
        if line.to_lowercase().contains("clip: flipyz") {
            orient_title = 1;
        }
        if line.to_lowercase().contains("clip: rotx") {
            orient_title = -1;
        }
    }

    // Set orientation values based on size and angles
    let mut orient_size = -2;
    if (nz_vol as f64) < ny_vol as f64 / YZ_RATIO_CRIT {
        orient_size = 1;
    }
    if (ny_vol as f64) < nz_vol as f64 / YZ_RATIO_CRIT {
        orient_size = 0;
    }

    let mut orient_angle = -2;
    if tilt_x_orig == 90. && tilt_x_cur == 0. {
        orient_angle = -1;
    } else if tilt_x_orig == 0. && tilt_x_cur == 90. {
        orient_angle = orient_size;
    }

    // Give information only when orientation entered
    if entered_orient != 0 {
        if reorient < 0 && orient_title == 1 {
            prnstr(
                "Using specified reorientation even though title indicates rotation around X",
                "\n",
                false,
            );
        }
        if reorient < 0 && orient_angle != -1 {
            prnstr(
                "Using specified reorientation even though header angles indicate  rotation around X",
                "\n",
                false,
            );
        }
        if reorient >= 0 && orient_title == -1 {
            prnstr(
                "Using specified reorientation even though title indicates swapping of Y and Z",
                "\n",
                false,
            );
        }
        if reorient >= 0 && orient_angle == -1 {
            prnstr(
                "Using specified reorientation even though header angles indicate swapping of Y and Z",
                "\n",
                false,
            );
        }
    } else {
        // Otherwise look for consistency, inform of decision in almost all cases, warn of
        // possible inconsistency, exit if not conclusive
        if orient_angle == -1 {
            reorient = -1;
            if orient_title == 1 {
                prnstr(
                    &format!(
                        "WARNING: {progname} - Assuming reorientation by rotation around  X because of header angles, but there is a title  indicating swapping of Y and Z"
                    ),
                    "\n",
                    false,
                );
            }
            if orient_size == 0 {
                prnstr(
                    &format!(
                        "WARNING: {progname} - Assuming reorientation by rotation around X because of header angles, even though Z dimension is much bigger than Y"
                    ),
                    "\n",
                    false,
                );
            }
        } else if orient_angle == -2 {
            exit_error(
                "You must enter -reorient to indicate reorientation type, because header angles are not consistent with known types",
            );
        } else {
            if orient_title == -1 {
                exit_error(
                    "You must enter -reorient to indicate reorientation type, because header angles are not consistent with title indicating rotation around X",
                );
            }
            if orient_title == 1 {
                if orient_size == 0 {
                    exit_error(
                        "You must enter -reorient to indicate reorientation type, because there is a title indicating swapping of Y and Z but the Z dimension is much bigger than Y",
                    );
                }
                reorient = 1;
                if orient_size == 1 {
                    prnstr(
                        "Assuming reorientation by swapping Y and Z as indicated by title",
                        "\n",
                        false,
                    );
                } else {
                    prnstr(
                        &format!(
                            "WARNING: {progname} - Assuming reorientation by swapping Y and Z as indicated by title, but Y and Z dimensions do not clearly support this assumption"
                        ),
                        "\n",
                        false,
                    );
                }
            } else {
                if orient_size < 0 {
                    exit_error(
                        "You must enter -reorient to indicate reorientation type, because Y and Z dimensions do not clearly indicate orientation",
                    );
                }
                reorient = orient_size;
                if reorient != 0 {
                    prnstr(
                        "Assuming reorientation by swapping Y and Z because of Y and Z dimensions, even though there is no flipyz title",
                        "\n",
                        false,
                    );
                } else {
                    prnstr(
                        "Assuming no reorientation because of Y and Z dimensions",
                        "\n",
                        false,
                    );
                }
            }
        }
    }

    let com_lines = read_text_file(com_file, None, false, None).unwrap_or_default();
    PointsAndHeaders {
        pid,
        clean_list,
        point_list,
        ali_binning,
        reorient,
        com_lines,
        nx_raw,
        ny_raw,
        nz_raw,
        pix_x_raw,
        pix_y_raw,
        pix_z_raw,
        nx_ali,
        ny_ali,
        nz_ali,
        pix_x_ali,
        pix_y_ali,
        pix_z_ali,
        orig_x_ali,
        orig_y_ali,
        orig_z_ali,
        nx_vol,
        ny_vol,
        nz_vol,
        pix_x_vol,
        pix_y_vol,
        pix_z_vol,
        orig_x_vol,
        orig_y_vol,
        orig_z_vol,
        nx_full,
        ny_full,
        nz_full,
        pix_x_full,
        pix_y_full,
        pix_z_full,
        orig_x_full,
        orig_y_full,
        orig_z_full,
    }
}

/// Matches `getCTFoptionsCheckIfCorrected` (`tomocoords.py:237`): gets shared
/// values when doing CTF correction for 3D CTF, checks whether an aligned
/// stack has CTF correction title, returns size in X, y, z as well as those
/// values.
///
/// An empty `raw_stack` is the source's `None`/`''`.  `BUGS.md` (tomocoords
/// `getCTFoptionsCheckIfCorrected`), fixed in translation: with no
/// `InputStack` in the CTF command file the source calls
/// `os.path.exists(None)` and dies with a TypeError traceback; here the
/// missing name is reported through the source's own "does not exist"
/// error, spelled `None` as Python's `str()` would.
pub fn get_ctf_options_check_if_corrected(
    progname: &str,
    tilt_lines: &[String],
    ctf_lines: &[String],
    raw_stack: &str,
    do_filter: bool,
) -> CtfOptions {
    let float1 = |lines: &[String], option: &str| match option_value(
        lines, option, 2, true, 1, None, None,
    ) {
        Some(OptionValue::Floats(values)) => Some(values[0]),
        _ => None,
    };
    let int1 = |lines: &[String], option: &str| match option_value(
        lines, option, 1, true, 1, None, None,
    ) {
        Some(OptionValue::Integers(values)) => Some(values[0]),
        _ => None,
    };
    let string = |lines: &[String], option: &str| match option_value(
        lines, option, 0, true, 0, None, None,
    ) {
        Some(OptionValue::String(value)) => Some(value),
        _ => None,
    };
    let xaxis_tilt = float1(tilt_lines, "xaxistilt");
    let tilt_use_gpu = int1(tilt_lines, "UseGPU");
    let ctf_xtilt = float1(ctf_lines, "XAxisTilt");
    let ctf_use_gpu = int1(ctf_lines, "UseGPU");
    let ctf_input = string(ctf_lines, "InputStack");
    let ctf_output = string(ctf_lines, "OutputFileName");
    let mut ctf_pix_size = float1(ctf_lines, "PixelSize");
    if raw_stack.is_empty()
        && !ctf_input
            .as_deref()
            .is_some_and(|name| Path::new(name).exists())
    {
        exit_error(&format!(
            "Input file in ctf correction command file, {}, does not exist",
            ctf_input.as_deref().unwrap_or("None")
        ));
    }
    let (nx, ny, nz);
    if !raw_stack.is_empty() {
        match get_mrc(raw_stack, false, false) {
            Ok(MrcInfo::Basic(x, y, z, _mode, px, _py, _pz)) => {
                (nx, ny, nz) = (x, y, z);
                ctf_pix_size = Some(px);
            }
            _ => exit_from_imod_error(progname),
        }
    } else {
        let input = ctf_input.clone().unwrap_or_default();
        let head_lines = match run_cmd(&format!("header {input}"), None, None, None, &[]) {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => exit_from_imod_error(progname),
        };
        for line in &head_lines {
            if line.to_lowercase().contains("ctfphaseflip") {
                exit_error(&format!(
                    "Ctfphaseflip appears to have been run already on the input stack for CTF correction, {input}"
                ));
            }
            if do_filter
                && line
                    .to_lowercase()
                    .contains("mtffilter: dose weight filtered")
            {
                exit_error(&format!(
                    "Filtering was requested, but dose weighting appears to have been done already on the input stack, {input}"
                ));
            }
        }

        match get_mrc_size(&input) {
            Ok(size) => (nx, ny, nz) = size,
            Err(_) => exit_from_imod_error(progname),
        }
    }

    CtfOptions {
        nx,
        ny,
        nz,
        xaxis_tilt: xaxis_tilt.unwrap_or(0.),
        tilt_use_gpu: tilt_use_gpu.unwrap_or(-1),
        ctf_xtilt,
        ctf_use_gpu,
        ctf_pix_size,
        ctf_input,
        ctf_output,
    }
}

/// Matches `checkXtiltCtfVsRec` (`tomocoords.py:277`): test for X tilt
/// consistency between CTF correction and reconstruction and if it matters.
pub fn check_xtilt_ctf_vs_rec(
    xaxis_tilt: f64,
    ctf_xtilt: Option<f64>,
    warn_xtilt_crit: f64,
    ysize_nm: f64,
    slab_nm: f64,
    slab_text: &str,
    progname: &str,
) {
    let xaxis_tilt = if xaxis_tilt == 0. { 0. } else { xaxis_tilt };
    if let Some(ctf) = ctf_xtilt.filter(|ctf| (xaxis_tilt - ctf).abs() >= 0.1) {
        prnstr(
            &fmtstr(
                "WARNING: {} - X-axis tilt for CTF correction is non-zero ({})  but it differs from X-tilt for reconstruction ({})",
                &[
                    progname.to_owned(),
                    py_str_float(ctf),
                    py_str_float(xaxis_tilt),
                ],
            ),
            "\n",
            false,
        );
    } else {
        let mut warn_start =
            "The difference in X-axis tilts between CTF correction and reconstruction";
        let ctf = match ctf_xtilt {
            Some(ctf) => ctf,
            None => {
                warn_start = "X-axis tilt is not included in the CTF correction but the X-tilt in reconstruction";
                0.
            }
        };
        let error_nm =
            0.5 * ysize_nm * (xaxis_tilt.to_radians().tan() - ctf.to_radians().tan()).abs();
        if error_nm / slab_nm > warn_xtilt_crit {
            prnstr(
                &format!(
                    "WARNING: {progname} - {warn_start} is big enough to give an error in Z-height of {} nm, more than {} of the {slab_text} thickness ({} nm)",
                    py_fixed(error_nm, 0, 0),
                    py_str_float(warn_xtilt_crit),
                    py_fixed(slab_nm, 0, 0)
                ),
                "\n",
                false,
            );
        }
    }
}

/// Matches `findSplitComNumber` (`tomocoords.py:303`): find the number of
/// command files created by splittilt or splitcorrection.
///
/// `runcmd` returns lines with their endings, so the source's `prnstr(l)`
/// of a warning prints a blank line after it; [`run_cmd`] strips them, and
/// the ending is put back here.
pub fn find_split_com_number(splitout: &[String], descrip: &str) -> i32 {
    let mut retval: i32 = -1;
    let reg = Regex::new(r"^([ 0-9]*).* files for [ 0-9]*chunks created.*$").unwrap();
    for l in splitout {
        if l.starts_with("WARNING:") {
            prnstr(&format!("{l}\n"), "\n", false);
        }
        if reg.is_match(l) {
            let numstr = reg.replace(l, "${1}").into_owned();
            if !numstr.is_empty() {
                // `int(numstr)`: surrounding blanks are ignored; a ValueError
                // would end the script with a traceback.
                match py_int(&numstr) {
                    Some(value) => retval = value as i32,
                    None => {
                        eprintln!("ValueError: invalid literal for int() with base 10: '{numstr}'");
                        crate::imod::libcfshr::b3dutil::exit(1)
                    }
                }
            }
        }
    }
    if retval < 0 {
        exit_error(&format!("Cannot determine com file number from {descrip}"));
    }
    retval
}

/// Matches `getOrDeriveComFile` (`tomocoords.py:320`): look for a file name
/// specified by the given option, and if not present, try to derive it from
/// the name of the tilt com file by substituting the defaultName string for
/// 'tilt'.
pub fn get_or_derive_com_file(
    option: &str,
    default_name: &str,
    tilt_com_file: &str,
    descrip: &str,
) -> String {
    let mut com_file = pip_get_string(option, "").unwrap_or_default();
    if !com_file.is_empty() {
        (com_file, _) = complete_and_check_com_file(&com_file);
    } else {
        if !tilt_com_file.contains("tilt") {
            exit_error(&format!(
                "You must enter the name of the {descrip} erasing command file since {tilt_com_file} does not contain \"tilt\""
            ));
        }
        com_file = tilt_com_file.replace("tilt", default_name);
        if !Path::new(&com_file).exists() {
            exit_error(&format!(
                "You must enter the name of the {descrip} command file, the file {com_file} dose not exist"
            ));
        }
    }

    com_file
}

/// Matches `getAxisAngleAndTranspose` (`tomocoords.py:338`): get the option
/// for the axis angle, and if not entered, look for an align log file and
/// find the angle there.  Determine if X and Y need to be transposed.
/// `axis_let` `None` is the Python `None`.
pub fn get_axis_angle_and_transpose(axis_let: Option<&str>) -> (f64, bool) {
    let mut axis_angle = pip_get_float("AxisAngle", 0.).unwrap_or(0.);
    if pip_get_err_no() != 0 {
        let align_log = axis_let.map(|letter| format!("align{letter}.log"));
        if axis_let.is_none() || !Path::new(align_log.as_deref().unwrap_or("")).exists() {
            let rot_file = axis_let.map(|letter| format!("rotation{letter}.xf"));
            if axis_let.is_none() || !Path::new(rot_file.as_deref().unwrap_or("")).exists() {
                exit_error("The tilt axis angle must be entered, it cannot be deduced");
            }
            let rot_file = rot_file.unwrap_or_default();
            let xf_lines = read_text_file(
                &rot_file,
                Some("file with rotation transform for the axis angle"),
                false,
                None,
            )
            .unwrap_or_default();
            let lsplit: Vec<&str> = xf_lines
                .first()
                .map(|line| line.split_whitespace().collect())
                .unwrap_or_default();
            match (
                lsplit.first().and_then(|text| py_float(text)),
                lsplit.get(1).and_then(|text| py_float(text)),
            ) {
                (Some(a11), Some(a12)) => axis_angle = (-a12).atan2(a11).to_degrees(),
                _ => exit_error(&format!(
                    "The tilt axis angle must be entered; there was an error converting values from {rot_file}"
                )),
            }
        } else {
            let align_log = align_log.unwrap_or_default();
            let log_lines = read_text_file(&align_log, Some("alignment log file"), false, None)
                .unwrap_or_default();
            let mut found = false;
            for line in &log_lines {
                let tag = "rotation angle is";
                if let Some(ind) = line.find(tag) {
                    let line = &line[ind + tag.len()..];
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    if let Some(value) = lsplit.first().and_then(|text| py_float(text)) {
                        axis_angle = value;
                        found = true;
                    }
                }
            }

            if !found {
                exit_error(&format!(
                    "The tilt axis angle must be entered, it cannot be determined from {align_log}"
                ));
            }
        }
    }
    let mut theta = axis_angle;
    if theta <= -45. {
        theta += 360.;
    }
    let mut rot_flip = py_round(theta / 90.) as i32;
    rot_flip = 0.max(3.min(rot_flip));
    let transpose_xy = rot_flip % 2 != 0;
    (axis_angle, transpose_xy)
}

/// Matches `getEssentialRawOptions` (`tomocoords.py:388`): gets the align
/// transform file option or sets up expected name, and tests if it exists;
/// also gets the binning/reduction and the pixel size entry.
pub fn get_essential_raw_options(raw_root: &str) -> (String, i32, f64) {
    // Get the align transform file
    let mut ali_xform_file = pip_get_string("AlignTransformFile", "").unwrap_or_default();
    if ali_xform_file.is_empty() {
        if raw_root.is_empty() {
            exit_error("The alignment transform file must be entered, its name cannot be deduced");
        }
        ali_xform_file = format!("{raw_root}.xf");
    }

    if !Path::new(&ali_xform_file).exists() {
        exit_error(&format!(
            "The alignment transform file does not exist: {ali_xform_file}"
        ));
    }

    // Determine binning
    let raw_binning = pip_get_integer("FourierReduceByFactor", 1).unwrap_or(1);
    let raw_pix_size = pip_get_float("RawPixelSize", 0.).unwrap_or(0.);

    (ali_xform_file, raw_binning, raw_pix_size)
}

/// Matches `backTransformEraseModel` (`tomocoords.py:411`): back-transforms
/// the eraser model file found in eraseLines with the aliXformFile.
/// rawPixSize has to be the pixel size (Angstroms or small integers) from the
/// raw stack header because transform shifts are in unbinned pixels and
/// shifts need to be scaled to the model pixel size.
///
/// `BUGS.md` (tomocoords `backTransformEraseModel`), fixed in translation:
/// the source names `eraseComFile` and `progname`, which are globals of the
/// calling scripts and not names in this module, so a missing `ModelFile`
/// entry or a failed `imodinfo`/`xfmodel` ended in a NameError traceback.
/// The caller's two values are passed in (`erase_com_file`, `progname`) and
/// the source's own messages are given.
pub fn back_transform_erase_model(
    erase_lines: &[String],
    ali_xform_file: &str,
    raw_pix_size: f64,
    erase_com_file: &str,
    progname: &str,
) -> String {
    let erase_fid = match option_value(erase_lines, "ModelFile", STRING_VALUE, false, 0, None, None)
    {
        Some(OptionValue::String(value)) => value,
        _ => String::new(),
    };
    if erase_fid.is_empty() {
        exit_error(&format!(
            "Cannot find model file name in eraser com file: {erase_com_file}"
        ));
    }
    if !Path::new(&erase_fid).exists() {
        exit_error(&format!(
            "Model file for gold erasing does not exist: {erase_fid}"
        ));
    }

    // Get model pixel size to determine scaling of xf's
    let asc_lines = match run_cmd(&format!("imodinfo -h {erase_fid}"), None, None, None, &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => exit_from_imod_error(progname),
    };
    let mut model_pix = 0.0_f64;
    let tag = "SCALE  = (";
    for line in &asc_lines {
        if let Some(ind) = line.find(tag) {
            let line = &line[ind + tag.len()..];
            let lsplit: Vec<&str> = line.split(',').collect();
            match py_float(lsplit[0]) {
                Some(value) => model_pix = value,
                None => exit_error(&format!(
                    "Converting the SCALE entry to a number in the imodinfo -a output from {erase_fid}"
                )),
            }
        }
    }

    if model_pix == 0. {
        exit_error(&format!(
            "Failed to find the SCALE entry in the imodinfo -a output from {erase_fid}"
        ));
    }
    let mod_scale = raw_pix_size / model_pix;

    // The magic of origins takes care of a mismatch in size
    let (erase_root, ext) = os_path_splitext(&erase_fid);
    let raw_erase_fid = format!("{erase_root}_raw{ext}");
    if run_cmd(
        &fmtstr(
            "xfmodel -back -scale {} -xforms {} {} {}",
            &[
                py_str_float(mod_scale),
                ali_xform_file.to_owned(),
                erase_fid.clone(),
                raw_erase_fid.clone(),
            ],
        ),
        None,
        None,
        None,
        &[],
    )
    .is_err()
    {
        exit_from_imod_error(progname);
    }

    raw_erase_fid
}

/// Matches `getFallbackRawPixel` (`tomocoords.py:452`): returns a pixel size
/// in nanometers based on the header value if it is not 1.0, otherwise tries
/// to obtain the pixel size from track.com.
pub fn get_fallback_raw_pixel(header_pix_size: f64, axis_let: Option<&str>, com_ext: &str) -> f64 {
    let raw_pix_size;
    if header_pix_size == 1.0 {
        let Some(axis_let) = axis_let else {
            exit_error(
                "The raw stack seems not to have a true pixel size and it cannot be deduced, it must be entered",
            );
        };
        let track_com = format!("track{axis_let}.{com_ext}");
        let track_lines = read_text_file(&track_com, None, false, None).unwrap_or_default();
        raw_pix_size = match option_value(
            &track_lines,
            "PixelSize",
            FLOAT_VALUE,
            false,
            1,
            None,
            None,
        ) {
            Some(OptionValue::Floats(values)) if values[0] != 0. => values[0],
            _ => exit_error(&format!(
                "The raw stack seems not to have a true pixel size and it cannot be found in {track_com}; it must be entered"
            )),
        };
    } else {
        raw_pix_size = header_pix_size / 10.;
    }
    raw_pix_size
}

/// Matches `checkForDistortion` (`tomocoords.py:474`): check whether prenewst
/// or newst was run with distortion or mag gradient correction.  Checks log
/// files first, then com files if there are no log files.  Returns 1 if found
/// in one file, or 2 if found in both; name of distortion file or None; name
/// of mag gradient file or None; image binning (`None` for the source's 0,
/// otherwise the text after the separator).  Issues a warning if action is 1
/// or an error if action > 1.
pub fn check_for_distortion(
    axis_let: Option<&str>,
    action: i32,
    progname: &str,
) -> (i32, Option<String>, Option<String>, Option<String>) {
    let mut dist = None;
    let mut grad = None;
    let mut bin = None;
    let mut sep = "=";
    let Some(axis_let) = axis_let else {
        return (0, None, None, None);
    };
    let mut count = 0;
    for ext in [".log", ".com"] {
        count = 0;
        let mut found = 0;

        for step in ["pre", ""] {
            let fname = format!("{step}newst{axis_let}{ext}");
            if Path::new(&fname).exists() {
                found += 1;
                let tlines = read_text_file(&fname, None, false, None).unwrap_or_default();
                let mut in_file = 0;
                for line in &tlines {
                    let line = line.trim();
                    // `line[ind + len(sep):]` with `ind` -1 when not found
                    let after = |ind: Option<usize>| -> String {
                        let start = ind.map_or(0, |ind| ind + sep.len());
                        line[start..].trim().to_owned()
                    };
                    let ind = line.find(sep);
                    if ind.is_some_and(|ind| ind > 0) && line.starts_with("DistortionField") {
                        in_file = 1;
                        dist = Some(after(ind));
                    }
                    if ind.is_some_and(|ind| ind > 0) && line.starts_with("GradientFile") {
                        in_file = 1;
                        grad = Some(after(ind));
                    }
                    if line.starts_with("ImagesAreBinned") {
                        bin = Some(after(ind));
                    }
                }

                count += in_file;
            }
        }

        sep = " ";
        if found != 0 {
            break;
        }
    }

    if count != 0 && action != 0 {
        let mess = "It appears that distortion correction was used in aligning images; reconstruction from raw images cannot be done correctly";
        if action > 1 {
            exit_error(mess);
        }
        prnstr(&format!("WARNING: {progname} - {mess}"), "\n", false);
    }
    (count, dist, grad, bin)
}

/// Matches `setAxisLetter` (`tomocoords.py:519`): sets the axis letter from
/// the root name and whether it is dual axis.
pub fn set_axis_letter(root_name: &str, if_dual: i32) -> Option<String> {
    let mut axis_let = None;
    if if_dual == 0 {
        axis_let = Some(String::new());
    } else if if_dual == 2 {
        if root_name.ends_with('a') {
            axis_let = Some("a".to_owned());
        } else if root_name.ends_with('b') {
            axis_let = Some("b".to_owned());
        }
    }
    axis_let
}

/// Matches `getRawExtensionFallback` (`tomocoords.py:533`): returns the
/// passed raw extension if it is OK, otherwise looks for .mrc, .st, .hdf and
/// returns an extension from whichever is found and number of candidate files
/// found.
pub fn get_raw_extension_fallback(raw_ext: &str, root_name: &str) -> (String, i32) {
    if !raw_ext.is_empty() {
        return (raw_ext.to_owned(), 1);
    }
    let mut raw_ext = raw_ext.to_owned();
    let mut num_candid = 0;
    for ext in ["st", "mrc", "hdf"] {
        if Path::new(&format!("{root_name}.{ext}")).exists() {
            num_candid += 1;
            raw_ext = ext.to_owned();
        }
    }

    (raw_ext, num_candid)
}
