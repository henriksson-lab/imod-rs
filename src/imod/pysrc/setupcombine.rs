//! Translation of `IMOD/pysrc/setupcombine`.
//!
//! A Python command script: its six functions are translated one for one
//! below, and its top level is [`setupcombine`], translated statement by
//! statement.  The module globals those functions read (`rootname`,
//! `stackExt`, `IMODversion`, `srcdir`, `nameStyle`, `changeList`,
//! `dfltComDir`, `origComDir`, `autoPatch`, `useVolMatch`, `extraTargets`,
//! `ranlist`, `ranind`, `chunkedHDF`, `sumname`, `tmppath`, `recfile`,
//! `tmpmatfile`, `warnToStdOut`) are the fields of [`Globals`], which the top
//! level owns and passes to them.
//!
//! The templates are read from `$IMOD_DIR/com`, as the source does.
//!
//! Python values: PIP floats and `optionValue` floats are doubles, and
//! `'{}'.format(x)` of one is its `repr` ([`py_str_float`]); `//` floors;
//! `round()` rounds half to even.  A value the source leaves undefined on
//! some path and later reads is an `Option`, and reading it as `None` is the
//! source's uncaught `NameError`.

use super::comchanger::{Change, modify_for_change_list, process_change_options};
use super::copytomocoms::path_join;
use super::imodpy::{
    MrcInfo, OptionValue, add_imod_bin_ignore_sighup, add_output_format_var_to_lines,
    auto_patch_number, call_own_program, dataset_filename, default_com_extension,
    exit_from_imod_error, find_root_axis_and_extensions, get_imod_version, get_montage_size,
    get_mrc, get_mrc_size, get_naming_style, make_backup_file, map_type_extension_to_style,
    option_value, os_path_splitext, patch_size_from_entry, prnstr, py_int_floordiv, py_round,
    read_text_file, run_cmd, set_root_and_extension, write_text_file,
};
use super::imodpy::{py_fixed, py_float, py_int, py_str_float};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_two_integers, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add};
use crate::imod::flib::distort::xf2rotmagstr::xf2rotmagstr_compute;
use crate::imod::flib::image::tomopieces::{tomopieces_output_lines, tomopieces_recording};
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::parse_input_params::set_exit_prefix;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

const XYBORDERS: [i64; 5] = [24, 36, 54, 68, 80]; // border sizes per increment of minimum
const BORDERINC: i64 = 1000; // dimension
const DELPATCHXY: i64 = 100;
const NZPATCH3: i64 = 75; // Thickness at which to do 3 patches in Z
const NZPATCH4: i64 = 150; // Thickness at which to do 4 patches in Z
const MAXPIXELS: i64 = 50000; // KPixels for maximum FFT
const TAPERPADXZ: i64 = 8; // Size of taper/pad for 3D ffts in X and Z
const TAPERPADY: i64 = 4; // Size of taper/pad for 3D ffts in Y
const MAXPIECEY: i64 = 1; // Maximum pieces in Y
const MINOVERLAP: i64 = 10; // Minimum overlap between pieces

/// The module globals the script's functions use (see the module comment).
#[derive(Default)]
pub struct Globals {
    /// `rootname`
    pub rootname: String,
    /// `stackExt`
    pub stack_ext: String,
    /// `IMODversion`
    pub imod_version: String,
    /// `srcdir`
    pub srcdir: String,
    /// `nameStyle`
    pub name_style: i32,
    /// `changeList`
    pub change_list: Vec<Change>,
    /// `dfltComDir`
    pub dflt_com_dir: String,
    /// `origComDir`
    pub orig_com_dir: String,
    /// `autoPatch`
    pub auto_patch: String,
    /// `useVolMatch`
    pub use_vol_match: i32,
    /// `extraTargets`
    pub extra_targets: String,
    /// `ranlist`
    pub ranlist: Vec<String>,
    /// `ranind`
    pub ranind: usize,
    /// `chunkedHDF`
    pub chunked_hdf: i32,
    /// `sumname`
    pub sumname: String,
    /// `tmppath`
    pub tmppath: String,
    /// `recfile`
    pub recfile: String,
    /// `tmpmatfile`
    pub tmpmatfile: String,
    /// `warnToStdOut`
    pub warn_to_std_out: i32,
}

/// Rust-only stand-in for an uncaught Python exception: the traceback's
/// first and last lines on standard error and exit status 1.
fn py_crash(message: &str) -> ! {
    let _ = std::io::stdout().flush();
    eprintln!("Traceback (most recent call last):");
    eprintln!("{message}");
    crate::imod::libcfshr::b3dutil::exit(1)
}

/// Matches `readTiltOptions` (`setupcombine:22`): read needed values from one
/// tilt.com.  Returns `(xaxis, offset, zshift, xshift, binning, sliceStart,
/// sliceEnd)`.
pub fn read_tilt_options(tilt: &str) -> (f64, f64, f64, f64, i32, i32, i32) {
    let comlines = read_text_file(tilt, Some("tilt command file"), false, None).unwrap_or_default();
    let floats = |option: &str| -> Option<Vec<f64>> {
        match option_value(&comlines, option, 2, true, 0, None, None) {
            Some(OptionValue::Floats(values)) => Some(values),
            _ => None,
        }
    };
    let ints = |option: &str| -> Option<Vec<i32>> {
        match option_value(&comlines, option, 1, true, 0, None, None) {
            Some(OptionValue::Integers(values)) => Some(values),
            _ => None,
        }
    };
    let tiltarr = floats("xaxistilt");
    let mut xaxis = 0.;
    if let Some(values) = tiltarr.as_ref().filter(|values| !values.is_empty()) {
        xaxis = values[0];
    }
    let tiltarr = floats("offset");
    let mut offset = 0.;
    if let Some(values) = tiltarr.as_ref().filter(|values| !values.is_empty()) {
        offset = values[0];
    }
    let tiltarr = floats("shift");
    let mut xshift = 0.;
    let mut zshift = 0.;
    if let Some(values) = tiltarr.as_ref().filter(|values| !values.is_empty()) {
        xshift = values[0];
    }
    if let Some(values) = tiltarr.as_ref().filter(|values| values.len() > 1) {
        zshift = values[1];
    }
    let mut binning = 1;
    let tiltarr = ints("imagebinned");
    if let Some(values) = tiltarr.as_ref().filter(|values| !values.is_empty()) {
        binning = values[0];
    }
    // Fixed in translation (BUGS.md): native divides by an `IMAGEBINNED 0`
    // (`setupcombine:43`, ZeroDivisionError); `tilt` itself takes
    // `B3DMAX(1, imageBinned)` (`tilt.cpp:3119`), and so does this.
    if binning < 1 {
        binning = 1;
    }
    zshift /= binning as f64;
    let mut slice_start = 0;
    let mut slice_end = 0;
    let tiltarr = ints("slice");
    if let Some(values) = tiltarr.as_ref().filter(|values| values.len() > 1) {
        slice_start = values[0];
        slice_end = values[1];
    }

    (
        xaxis,
        offset,
        zshift,
        xshift,
        binning,
        slice_start,
        slice_end,
    )
}

/// Matches `getSeriesSizeAndAngle` (`setupcombine:55`): determine binned
/// original size and axis rotation angle for one axis.  `None` sizes are the
/// source's `(None, None, None)` return.
pub fn get_series_size_and_angle(
    g: &Globals,
    progname: &str,
    axis: &str,
) -> (Option<i64>, Option<i64>, Option<f64>) {
    let mut xsize: i64 = 0;
    let mut ysize: i64 = 0;
    let mut angle: Option<f64> = None;
    let xfname = format!("{}{axis}.xf", g.rootname);
    let tltname = format!("{}{axis}.tlt", g.rootname);
    let recname = dataset_filename(&format!("{axis}.rec"), None, None);
    let stackname = format!("{}{axis}.{}", g.rootname, g.stack_ext);
    let plname = format!("{}{axis}.pl", g.rootname);
    if !(Path::new(&recname).exists() && Path::new(&stackname).exists()) {
        return (None, None, None);
    }

    // First get the axis angle from the rotation of the zero-tilt view
    if Path::new(&xfname).exists() {
        // Direct call (owner rule, 2026-09-26): xf2rotmagstr's transforms in
        // place of parsing its output lines (`setupcombine:82-108`).  The
        // script kept the lines from the first containing "rot=" (dropping
        // the warping-file note) and took the third word of the chosen line
        // with commas blanked, the `rot={f8.2}` field: so the angle is rounded
        // as `f8.2` prints it, and a field printed without a leading blank ran
        // into `rot=` and left `mag=`, which does not convert.
        let rot_file = xfname.clone();
        let rot_result = match call_own_program(
            &format!("xf2rotmagstr \"{xfname}\""),
            &["xf2rotmagstr"],
            None,
            true,
            move || {
                set_exit_prefix("ERROR: XF2ROTMAGSTR -");
                xf2rotmagstr_compute(&rot_file)
            },
        ) {
            Ok((Some(result), _)) => result,
            _ => exit_from_imod_error(progname),
        };
        // The number of lines the script kept: the transforms, or the
        // warping-file note alone when there are none.
        let num_lines = if rot_result.transforms.is_empty() {
            usize::from(rot_result.from_warp_file)
        } else {
            rot_result.transforms.len()
        };

        // Assume the middle view, but if tlt file exists, read it and find minimum angle
        let mut middle = num_lines / 2;
        if Path::new(&tltname).exists() {
            let tiltlines = read_text_file(&tltname, None, false, None).unwrap_or_default();
            if tiltlines.len() == num_lines {
                let mut min_tilt: f64 = 200.;
                let mut min_line = middle;
                let mut failed = false;
                for ind in 0..tiltlines.len() {
                    // `float()` strips surrounding whitespace
                    let Some(tilt) = py_float(&tiltlines[ind]) else {
                        failed = true;
                        break;
                    };
                    if tilt.abs() < min_tilt.abs() {
                        min_tilt = tilt;
                        min_line = ind;
                    }
                }
                if !failed {
                    middle = min_line;
                }
            }
        }

        // With whatever line, convert the angle if possible.  Need negative to fit
        // definition of axis angle
        if middle >= num_lines {
            py_crash("IndexError: list index out of range");
        }
        // The warping-file note has no angle ("warping" does not convert).
        if let Some(transform) = rot_result.transforms.get(middle) {
            let field = format_f(f64::from(transform.theta), 8, 2);
            if field.starts_with(' ') {
                if let Some(value) = py_float(&field) {
                    angle = Some(-value);
                }
            }
        }
    }

    let (mut nx_stk, mut ny_stk, xpix_stk) = match get_mrc(&stackname, false, false) {
        Ok(MrcInfo::Basic(nx, ny, _, _, xpix, ..)) => (nx as i64, ny as i64, xpix as f64),
        _ => exit_from_imod_error(progname),
    };
    let xpix_rec = match get_mrc(&recname, false, false) {
        Ok(MrcInfo::Basic(_, _, _, _, xpix, ..)) => xpix as f64,
        _ => exit_from_imod_error(progname),
    };
    if xpix_stk == 0. {
        py_crash("ZeroDivisionError: float division by zero");
    }
    // Header pixel sizes: finite for any file IMOD writes (`getmrc` divides the
    // cell size by the sampling); a NaN/inf header value would make Python raise
    let binning = py_round(xpix_rec / xpix_stk) as i64;

    // If it is montage, get the size, forget it if it fails
    if Path::new(&plname).exists() {
        if let Ok((nx, ny, _nz)) = get_montage_size(&stackname, Some(&plname)) {
            nx_stk = nx as i64;
            ny_stk = ny as i64;
            if binning == 0 {
                py_crash("ZeroDivisionError: integer division or modulo by zero");
            }
            xsize = py_int_floordiv(nx_stk, binning);
            ysize = py_int_floordiv(ny_stk, binning);
        }
    }
    // Otherwise, we have the size already
    else {
        if binning == 0 {
            py_crash("ZeroDivisionError: integer division or modulo by zero");
        }
        xsize = py_int_floordiv(nx_stk, binning);
        ysize = py_int_floordiv(ny_stk, binning);
    }

    (Some(xsize), Some(ysize), angle)
}

/// Matches `editApplyChangeListAndWrite` (`setupcombine:127`): edits the com
/// file with the current sed commands, applies a change list if any, writes
/// com file and writes a default copy to dfltcoms if different, and a copy to
/// origcoms.  The source appends the version line to the caller's `sedcom`,
/// so it takes the list mutably.
pub fn edit_apply_change_list_and_write(
    g: &Globals,
    sedcom: &mut Vec<String>,
    com_name: &str,
    do_change: bool,
) {
    sedcom.push(format!(
        "/####CreatedVersion####/s/# .*/#{}/",
        g.imod_version
    ));
    let (src_root, _ext) = os_path_splitext(com_name);
    let mut sed_lines = pysed(
        sedcom,
        PysedSrc::File(&format!("{}/{src_root}.com", g.srcdir)),
        None,
        false,
        '/',
        false,
    )
    .ok()
    .flatten()
    .unwrap_or_default();

    // set output format if appropriate
    add_output_format_var_to_lines(&mut sed_lines, g.name_style, None, false);

    if do_change && !g.change_list.is_empty() {
        let temp_lines = sed_lines.clone();
        sed_lines = match modify_for_change_list(&sed_lines, &src_root, "", &g.change_list, false) {
            Ok(lines) => lines,
            Err(message) => exit_error(&message),
        };
        if !g.dflt_com_dir.is_empty() && temp_lines != sed_lines {
            let _ = write_text_file(&path_join(&g.dflt_com_dir, com_name), &temp_lines, false);
        }
    }
    let _ = write_text_file(com_name, &sed_lines, false);
    if do_change && !g.orig_com_dir.is_empty() {
        let _ = write_text_file(&path_join(&g.orig_com_dir, com_name), &sed_lines, false);
    }
}

/// Matches `makeCombineCom` (`setupcombine:146`): make a copy of combine.com or
/// edit it as needed.
pub fn make_combine_com(g: &Globals) {
    let mut sedcom = vec![format!(
        "/####CreatedVersion####/s/# .*/#{}/",
        g.imod_version
    )];
    if !g.auto_patch.is_empty() || g.use_vol_match != 0 {
        if !g.auto_patch.is_empty() {
            let mut final_ = g.auto_patch.clone();
            if !g.extra_targets.is_empty() {
                final_ += &format!(" -extra {}", g.extra_targets);
            }
            sedcom.push(format!("/autopatchfit -final/s/-final E/-final {final_}/"));
            sedcom.push("/goto patchcorr/s//goto autopatch/".to_owned());
        }

        if g.use_vol_match != 0 {
            sedcom.push("/goto solvematch/s//goto dualvolmatch/".to_owned());
        }
    }

    let _ = pysed(
        &sedcom,
        PysedSrc::File(&format!("{}/combine.com", g.srcdir)),
        Some("combine.com"),
        false,
        '/',
        false,
    );
}

/// Matches `commonFFTLines` (`setupcombine:162`).
pub fn common_fft_lines(g: &Globals, sumnum: i64) -> Vec<String> {
    let Some(line) = g.ranlist.get(g.ranind) else {
        py_crash("IndexError: list index out of range");
    };
    let ixyz = line.split(',').collect::<Vec<_>>();
    if ixyz.len() < 6 {
        py_crash("IndexError: list index out of range");
    }
    let out_file = if g.chunked_hdf != 0 {
        g.sumname.clone()
    } else {
        g.tmppath.clone() + &dataset_filename(".rec", Some(&format!("sum{sumnum}")), None)
    };
    vec![
        "$echo".to_owned(),
        "#".to_owned(),
        "$combinefft -StandardInput".to_owned(),
        format!("AInputFFT\t{}", g.recfile),
        format!("BInputFFT\t{}", g.tmpmatfile),
        format!("OutputFFT\t{out_file}"),
        format!("XMinAndMax\t{},{}", ixyz[0], ixyz[1]),
        format!("YMinAndMax\t{},{}", ixyz[2], ixyz[3]),
        format!("ZMinAndMax\t{},{}", ixyz[4], ixyz[5]),
        format!("TaperPadsInXYZ\t{TAPERPADXZ},{TAPERPADY},{TAPERPADXZ}"),
    ]
}

/// Matches `warning` (`setupcombine:181`): a blank line and the warning, on
/// standard error unless `-warnings` sends them to standard output.
pub fn warning(g: &Globals, text: &str) {
    if g.warn_to_std_out != 0 {
        prnstr(" ", "\n", false);
        prnstr(&format!("WARNING: {text}\n"), "\n", false);
    } else {
        let _ = std::io::stdout().flush();
        eprint!(" \n");
        eprint!("WARNING: {text}\n\n");
    }
}

/// The script's top level (`setupcombine:190-839`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn setupcombine(arguments: &[OsString]) -> i32 {
    let progname = "setupcombine";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let mut g = Globals::default();

    //
    // Setup runtime environment
    let imod_dir = match std::env::var("IMOD_DIR") {
        Ok(value) => {
            add_imod_bin_ignore_sighup();
            value
        }
        Err(_) => {
            print!("{prefix} IMOD_DIR is not defined!\n");
            let _ = std::io::stdout().flush();
            return 1;
        }
    };

    let to_srcname = "g5a";
    let from_srcname = "g5b";
    let tmp_srcname = "g5tmpdir";
    let root_srcname = "g5";
    let backupname = "./savework";
    let backupsed = r".\/savework";

    g.srcdir = path_join(&imod_dir, "com");
    if !Path::new(&g.srcdir).exists() {
        exit_error(&format!(
            "Source directory for command files, {}, not found",
            g.srcdir
        ));
    }

    // Get IMOD version
    g.imod_version = match get_imod_version() {
        Some(version) if !version.is_empty() => version,
        _ => exit_error("Getting IMOD version"),
    };

    // Set up to save coms if copytomocoms set up directories
    g.orig_com_dir = "origcoms".to_owned();
    g.dflt_com_dir = "dfltcoms".to_owned();
    if !Path::new(&g.orig_com_dir).exists() {
        g.orig_com_dir = String::new();
    }
    if !Path::new(&g.dflt_com_dir).exists() {
        g.dflt_com_dir = String::new();
    }

    let mut sedtranskey = "gibberish".to_owned();
    let mut fromlet = "b";
    let mut tolet = "a";
    let mut delregion = "RegionModel".to_owned();

    // Fallbacks from ../manpages/autodoc2man 3 1 setupcombine
    let options: Vec<String> = [
        "name:RootName:FN:",
        "atob:MatchAtoB:B:",
        "tolist:ToVolPointList:LI:",
        "fromlist:FromVolPointList:LI:",
        "transfer:TransferPointFile:FN:",
        "uselist:UsePointList:LI:",
        "surfaces:SurfaceModelType:I:",
        "initial:InitialVolumeMatching:B:",
        "patchsize:PatchTypeOrXYZ:CH:",
        "autopatch:AutoPatchFinalSize:CH:",
        "extra:ExtraResidualTargets:CH:",
        "xlimits:XLowerAndUpper:IP:",
        "ylimits:YLowerAndUpper:IP:",
        "zlimits:ZLowerAndUpper:IP:",
        "regionmod:PatchRegionModel:FN:",
        "lowradius:LowFromBothRadius:F:",
        "wedgefrac:WedgeReductionFraction:F:",
        "change:ChangeParametersFile:FNM:",
        "one:OneParameterChange:CHM:",
        "tempdir:TemporaryDirectory:FN:",
        "noclean:NoTempCleanup:B:",
        "info:InfoOnPatchSizes:B:",
        "only:OnlyMakeCombineCom:B:",
        "style:NamingStyle:I:",
        "stackext:StackExtension:CH:",
        "chunked:ChunkedHDFFiles:I:",
        "warnings:WarningsToStandardOut:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 0);

    // Get patch size info first and exit
    let if_info = pip_get_boolean("InfoOnPatchSizes", 0).unwrap_or(0);
    if if_info != 0 {
        for size in ["S", "M", "L", "E"] {
            let (patchnx, patchny, patchnz, _err) = patch_size_from_entry(size);
            prnstr(
                &format!("{size}:  {patchnx}  {patchny}  {patchnz}"),
                "\n",
                false,
            );
        }
        let _ = std::io::stdout().flush();
        return 0;
    }

    // Now get info about the data set
    let (mut com_ext, _dual_num, _setroot, type_ext, stack_ext) =
        find_root_axis_and_extensions(-1, None);
    if com_ext.is_empty() {
        com_ext = default_com_extension();
    }
    let com_ext = format!(".{com_ext}");

    g.stack_ext = pip_get_string("StackExtension", &stack_ext).unwrap_or_default();
    if g.stack_ext.is_empty() {
        exit_error(
            &("Raw stack extension must be entered; it cannot be determined from ".to_owned()
                + "dataset files"),
        );
    }

    let (mut name_style, type_ext) = match get_naming_style(type_ext.as_deref(), false, false) {
        Ok(result) => result,
        Err(message) => exit_error(&message),
    };
    let type_ext = type_ext.unwrap_or_default();
    if name_style < 0 {
        name_style = map_type_extension_to_style(&type_ext, false);
    }
    g.name_style = name_style;

    let solvematch = format!("solvematch{com_ext}");
    let dualvolmatch = format!("dualvolmatch{com_ext}");
    let matchvol1 = format!("matchvol1{com_ext}");
    let patchcorr = format!("patchcorr{com_ext}");
    let matchorwarp = format!("matchorwarp{com_ext}");
    let warpvol = format!("warpvol{com_ext}");
    let matchvol2 = format!("matchvol2{com_ext}");
    let volcombine = format!("volcombine{com_ext}");

    // Get options
    let make_combine_only = pip_get_boolean("OnlyMakeCombineCom", 0).unwrap_or(0);
    g.rootname = pip_get_string("RootName", "").unwrap_or_default();
    if g.rootname.is_empty() && make_combine_only == 0 {
        exit_error("A root name must be entered");
    }
    let rootname = g.rootname.clone();

    g.warn_to_std_out = pip_get_boolean("WarningsToStandardOut", 0).unwrap_or(0);
    let matchatob = pip_get_boolean("MatchAtoB", 0).unwrap_or(0);
    if matchatob != 0 {
        fromlet = "a";
        tolet = "b";
    }

    let mut corrlist1 = pip_get_string("ToVolPointList", "/").unwrap_or_default();
    let mut corrlist2 = pip_get_string("FromVolPointList", "/").unwrap_or_default();

    let transfile = pip_get_string("TransferPointFile", "").unwrap_or_default();
    if !transfile.is_empty() {
        sedtranskey = "#TransferCoordinateFile".to_owned();
    }

    let mut uselist = pip_get_string("UsePointList", "/").unwrap_or_default();

    // All lists must have / converted to \/ to be used in sed commands
    corrlist1 = corrlist1.replace('/', r"\/");
    corrlist2 = corrlist2.replace('/', r"\/");
    uselist = uselist.replace('/', r"\/");

    g.use_vol_match = pip_get_boolean("InitialVolumeMatching", 0).unwrap_or(0);
    let modsurf;
    if g.use_vol_match != 0 {
        modsurf = 2;
    } else {
        modsurf = pip_get_integer("SurfaceModelType", -10).unwrap_or(-10);
        if (modsurf < -2 || modsurf > 2) && make_combine_only == 0 {
            exit_error(
                &("You must enter either the -initial option or the -surface option ".to_owned()
                    + "with a value between -2 and 2"),
            );
        }
    }

    let patchin = pip_get_string("PatchTypeOrXYZ", "M").unwrap_or_default();
    let (patchnx, patchny, patchnz, err) = patch_size_from_entry(&patchin);
    let (patchnx, patchny, mut patchnz) = (patchnx as i64, patchny as i64, patchnz as i64);
    if err != 0 && make_combine_only == 0 {
        exit_error(
            &("You must enter one of S, M, L, or E or sizes in X,Y,Z for ".to_owned()
                + "the -patchsize option"),
        );
    }

    g.auto_patch = pip_get_string("AutoPatchFinalSize", "").unwrap_or_default();
    g.extra_targets = String::new();
    let (mut autonx, mut autony, mut autonz) = (0i64, 0i64, 0i64);
    if !g.auto_patch.is_empty() {
        let (nx, ny, nz, err) = patch_size_from_entry(&g.auto_patch);
        (autonx, autony, autonz) = (nx as i64, ny as i64, nz as i64);
        if err != 0 {
            exit_error(
                &("You must enter one of S, M, L, or E or sizes in X,Y,Z for ".to_owned()
                    + "the -autopatch option"),
            );
        }
        if autonx < patchnx || autony < patchny || autonz < patchnz {
            exit_error(
                &("The final patch size for Autopatchfit must not be smaller than ".to_owned()
                    + "the starting patch size"),
            );
        }

        // Handle #, #, # in this entry, which etomo sends in, and handle no comma or space
        // and comma in extra targets
        if g.auto_patch.contains(' ') {
            g.auto_patch = format!("{autonx},{autony},{autonz}");
        }
        g.extra_targets = pip_get_string("ExtraResidualTargets", "").unwrap_or_default();
        if !g.extra_targets.is_empty() && g.extra_targets.contains(' ') {
            g.extra_targets = g.extra_targets.replace(' ', ",");
            while g.extra_targets.contains(",,") {
                g.extra_targets = g.extra_targets.replace(",,", ",");
            }
        }
    }

    set_root_and_extension(&rootname, &type_ext);

    // If making combine.com only, do it now that preconditions are done
    if make_combine_only != 0 {
        make_backup_file("combine.com");
        make_combine_com(&g);
        let _ = std::io::stdout().flush();
        return 0;
    }

    // Limits
    let (patchxl, patchxu) = pip_get_two_integers("XLowerAndUpper", (0, 0)).unwrap_or((0, 0));
    let (mut patchxl, mut patchxu) = (patchxl as i64, patchxu as i64);
    let noxlim = pip_get_err_no();
    let (patchyl, patchyu) = pip_get_two_integers("YLowerAndUpper", (0, 0)).unwrap_or((0, 0));
    let (mut patchyl, mut patchyu) = (patchyl as i64, patchyu as i64);
    let noylim = pip_get_err_no();
    let (patchzl, patchzu) = pip_get_two_integers("ZLowerAndUpper", (0, 0)).unwrap_or((0, 0));
    let (mut patchzl, mut patchzu) = (patchzl as i64, patchzu as i64);
    let no_z_entered = pip_get_err_no();
    if no_z_entered != 0 && g.auto_patch.is_empty() {
        exit_error("You must enter the -zlimits option");
    }

    // Region model
    let regionmod = pip_get_string("PatchRegionModel", "").unwrap_or_default();
    if !regionmod.is_empty() {
        delregion = "gibberish".to_owned();
        if !Path::new(&regionmod).exists() {
            warning(&g, &format!("file {regionmod} does not yet exist."));
        }
    }

    // Volcombine entries, Temp directory and cleanup.  The defaults are the
    // integer 0, which `'{}'.format` prints as `0`.
    let low_radius = pip_get_float("LowFromBothRadius", 0.).unwrap_or(0.);
    let low_radius_text = if pip_get_err_no() == 1 {
        "0".to_owned()
    } else {
        py_str_float(low_radius)
    };
    let wedge_reduction = pip_get_float("WedgeReductionFraction", 0.).unwrap_or(0.);
    let wedge_reduction_text = if pip_get_err_no() == 1 {
        "0".to_owned()
    } else {
        py_str_float(wedge_reduction)
    };
    let tmproot = pip_get_string("TemporaryDirectory", "").unwrap_or_default();
    let handclean = pip_get_boolean("NoTempCleanup", 0).unwrap_or(0);
    g.change_list = process_change_options(
        "ChangeParametersFile",
        "OneParameterChange",
        "comparam",
        4,
        false,
    );
    g.chunked_hdf = pip_get_integer("ChunkedHDFFiles", 0).unwrap_or(0);

    // Get tilt.com filenames and values from them
    let toname = format!("{rootname}{tolet}");
    let fromname = format!("{rootname}{fromlet}");
    let tilta = format!("tilt{tolet}{com_ext}");
    let tiltb = format!("tilt{fromlet}{com_ext}");

    let (xaxisa, angoffa, zshifta, xaxisb, angoffb, zshiftb): (f64, f64, f64, f64, f64, f64);
    let (binninga, binningb): (i32, i32);
    // Fixed in translation (BUGS.md): the source assigns these only when both
    // tilt files exist (`setupcombine:406-411`) and reads `xshifta`
    // unconditionally at `:444`, so native always dies with NameError without
    // them; like the other values of that branch they default to 0 here
    // (the slice limits already start at 0).
    let (mut xshifta, mut xshiftb): (Option<f64>, Option<f64>) = (Some(0.), Some(0.));
    let (mut slice_starta, mut slice_enda, mut slice_startb, mut slice_endb) = (0, 0, 0, 0);
    if Path::new(&tilta).exists() && Path::new(&tiltb).exists() {
        let (xa, aa, za, xsa, ba, ssa, sea) = read_tilt_options(&tilta);
        let (xb, ab, zb, xsb, bb, ssb, seb) = read_tilt_options(&tiltb);
        (xaxisa, angoffa, zshifta, binninga) = (xa, aa, za, ba);
        (xaxisb, angoffb, zshiftb, binningb) = (xb, ab, zb, bb);
        (xshifta, xshiftb) = (Some(xsa), Some(xsb));
        (slice_starta, slice_enda, slice_startb, slice_endb) = (ssa, sea, ssb, seb);
    } else {
        (xaxisa, xaxisb, angoffa, angoffb, zshifta, zshiftb) = (0., 0., 0., 0., 0., 0.);
        (binninga, binningb) = (0, 0);
        if g.use_vol_match == 0 {
            warning(
                &g,
                &format!(
                    "CANNOT FIND tilta{com_ext} or tiltb{com_ext}; CANNOT SET X-AXIS TILTS CORRECTLY"
                ),
            );
        }
    }

    // set up some filenames
    g.recfile = dataset_filename(".rec", Some(&toname), None);
    let matfile = dataset_filename(".mat", Some(&fromname), None);
    let origfile = dataset_filename(".rec", Some(&fromname), None);
    let from_stack = format!("{fromname}.{}", g.stack_ext);
    let from_ali = dataset_filename(".ali", Some(&fromname), None);
    let to_ali = dataset_filename(".ali", Some(&toname), None);
    let invfile = "inverse.xf";
    let atlt = format!("{toname}.tlt");
    let btlt = format!("{fromname}.tlt");
    let _sum_name = dataset_filename(".rec", Some("sum"), None);
    let recfile = g.recfile.clone();

    // Get file size
    let (nx, nz, ny): (i64, i64, i64);
    if Path::new(&recfile).exists() {
        match get_mrc_size(&recfile) {
            Ok((x, z, y)) => (nx, nz, ny) = (x as i64, z as i64, y as i64),
            Err(_) => exit_from_imod_error(progname),
        }
    } else {
        (nx, nz, ny) = (1024, 60, 1024);
        warning(
            &g,
            &format!("{recfile} NOT FOUND; SETTING SIZE TO {nx} {nz} {ny} FOR TEST PURPOSES"),
        );
    }

    // Set Z limits to full range if using automated processing
    if no_z_entered != 0 {
        patchzl = 1;
        patchzu = nz;
    }

    // Determine if reconstructions are centered, needed for two kinds of output
    let mut recons_centered = false;
    let xshifta = xshifta.unwrap_or(0.);
    if xshifta == 0.
        && xshiftb.unwrap_or(0.) == 0.
        && Path::new(&from_stack).exists()
        && Path::new(&format!("{fromname}.xf")).exists()
        && binninga != 0
        && ((slice_starta == 0 && slice_enda == 0) || Path::new(&to_ali).exists())
        && ((slice_startb == 0 && slice_endb == 0) || Path::new(&from_ali).exists())
    {
        recons_centered = true;
        let mut failed = false;
        if slice_starta != 0 || slice_enda != 0 {
            match get_mrc_size(&to_ali) {
                Ok((_nxali, nyali, _nzali)) => {
                    let diff = (slice_starta + slice_enda).div_euclid(2) - nyali;
                    if diff > 4 || diff < -4 {
                        recons_centered = false;
                    }
                }
                Err(_) => failed = true,
            }
        }

        if !failed && recons_centered && (slice_startb != 0 || slice_endb != 0) {
            match get_mrc_size(&from_ali) {
                Ok((_nxali, nyali, _nzali)) => {
                    let diff = (slice_startb + slice_endb).div_euclid(2) - nyali;
                    if diff > 4 || diff < -4 {
                        recons_centered = false;
                    }
                }
                Err(_) => failed = true,
            }
        }
        if failed {
            recons_centered = false;
        }
    }

    // Get the default border size and process the patch limits
    let minsize = nx.min(ny);
    let borderindex = minsize
        .div_euclid(BORDERINC)
        .min(XYBORDERS.len() as i64 - 1);
    let xyborder = XYBORDERS[borderindex as usize].min(minsize.div_euclid(4));

    if noxlim != 0 {
        patchxl = xyborder;
        patchxu = nx - xyborder;
    }
    if patchxl < 0 || patchxl > nx || patchxl >= patchxu {
        exit_error(&format!(
            "X limits ({patchxl}, {patchxu}) out of range or out of order"
        ));
    }
    if noylim != 0 {
        patchyl = xyborder;
        patchyu = ny - xyborder;
    }
    if patchyl < 0 || patchyl > ny || patchyl >= patchyu {
        exit_error(&format!(
            "Y limits ({patchyl}, {patchyu}) out of range or out of order"
        ));
    }

    // NEW: make Z positive
    if patchzl == 0 {
        patchzl = 1;
    }
    if patchzl <= 0 || patchzl > nz || patchzl >= patchzu {
        exit_error(&format!(
            "Z limits ({patchzl}, {patchzu}) out of range or out of order"
        ));
    }

    let mut npatchx = (patchxu + DELPATCHXY / 2 - patchxl).div_euclid(DELPATCHXY);
    let mut npatchy = (patchyu + DELPATCHXY / 2 - patchyl).div_euclid(DELPATCHXY);

    // set number of patches in Z based on criteria, but if patch thickness
    // is greater than two-thirds of extent, do only one row of patches
    let mut npatchz: i64 = 2;
    if patchzu - patchzl >= NZPATCH3 {
        npatchz = 3;
    }
    if patchzu - patchzl >= NZPATCH4 {
        npatchz = 4;
    }

    if !g.auto_patch.is_empty() {
        npatchx = auto_patch_number(autonx as i32, patchxl as i32, patchxu as i32, false, 0) as i64;
        npatchy = auto_patch_number(autony as i32, patchyl as i32, patchyu as i32, false, 0) as i64;
        npatchz =
            npatchz.max(
                auto_patch_number(autonz as i32, patchzl as i32, patchzu as i32, true, 0) as i64,
            );
    }

    if (patchnz as f64) > (2 * (patchzu - patchzl)) as f64 / 3. {
        npatchz = 1;
        warning(&g, "Only one layer of patches will be computed in Z");
    }

    // If patch thickness is greater than extent cut it down
    if patchnz > patchzu + 1 - patchzl {
        patchnz = patchzu + 1 - patchzl;
        warning(
            &g,
            &format!("Patch thickness set to {patchnz} to fit within Z limits"),
        );
    }

    // Get tilt series size and angle and set up for modifications
    let (nx_astack, ny_astack, a_angle) = get_series_size_and_angle(&g, progname, "a");
    let (nx_bstack, ny_bstack, b_angle) = get_series_size_and_angle(&g, progname, "b");
    // Python truth values: None and 0 are false
    let a_angle_true = a_angle.is_some_and(|angle| angle != 0.);
    let b_angle_true = b_angle.is_some_and(|angle| angle != 0.);
    let nx_astack_true = nx_astack.is_some_and(|size| size != 0);
    let nx_bstack_true = nx_bstack.is_some_and(|size| size != 0);
    let mut axis_angle = 0.;
    if a_angle_true && b_angle_true {
        axis_angle = (a_angle.unwrap() + b_angle.unwrap()) / 2.;
    } else if a_angle_true {
        axis_angle = a_angle.unwrap();
    } else if b_angle_true {
        axis_angle = b_angle.unwrap();
    }

    let kpixel = ((nx * nz).div_euclid(100) * ny).div_euclid(10);
    let mut mbytes = ((6 * kpixel) + kpixel.div_euclid(2)).div_euclid(1000);
    let lim1 = ((4 * kpixel) + kpixel.div_euclid(2) + (8 * MAXPIXELS)).div_euclid(1000);
    let lim2 = ((2 * kpixel) + (10 * MAXPIXELS)).div_euclid(1000);
    mbytes = mbytes.max(lim1).max(lim2);

    prnstr(
        &format!("{mbytes} MBytes of disk space will be needed for combining"),
        "\n",
        false,
    );

    let tmpdir;
    let ifmkdir;
    let sedtmpdir;
    let sedmatfile;
    let sedtempkey;
    if !tmproot.is_empty() {
        if !Path::new(&tmproot).is_dir() {
            exit_error(&format!("{tmproot} does not exist or is not a directory"));
        }
        // `os.access(tmproot, os.W_OK)`: the POSIX `access` call itself
        let writable = std::ffi::CString::new(tmproot.as_bytes())
            .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
            .unwrap_or(false);
        if !writable {
            exit_error(&format!(
                " You do not have permission to write in {tmproot}"
            ));
        }
        tmpdir = format!("{tmproot}/combine.{}", std::process::id());
        g.tmppath = format!("{tmpdir}/");
        ifmkdir = "$if";
        g.tmpmatfile = g.tmppath.clone() + &matfile;
        sedtmpdir = tmpdir.replace('/', r"\/");
        sedmatfile = sedtmpdir.clone() + r"\/" + &matfile;
        sedtempkey = "TemporaryDir";
    } else {
        tmpdir = String::new();
        g.tmppath = String::new();
        ifmkdir = "#$if";
        g.tmpmatfile = matfile.clone();
        sedtmpdir = String::new();
        sedmatfile = matfile.clone();
        sedtempkey = "gibberish";
    }

    g.sumname = dataset_filename(".rec", Some("sum"), None);
    let sumname = g.sumname.clone();
    let tmppath = g.tmppath.clone();
    let tmpmatfile = g.tmpmatfile.clone();

    // If there is an existing link to sum.rec, remove it
    if std::fs::symlink_metadata(&sumname).is_ok_and(|meta| meta.file_type().is_symlink()) {
        let _ = std::fs::remove_file(&sumname);
    }

    // BUT NO LONGER MAKE sum.rec in the temporary directory
    if !tmpdir.is_empty() {
        if handclean != 0 {
            prnstr(
                &format!(
                    "sum.rec will be assembled in the current directory.\n{matfile} will be left in {tmpdir}\nYou are responsible for deleting $tmpdir and its contents\n when you are done with these files."
                ),
                "\n",
                false,
            );
        } else {
            prnstr(
                &format!(
                    "Your temporary directory is {tmpdir}\nIt and its contents will be deleted when combine.com finishes successfully."
                ),
                "\n",
                false,
            );
        }
    }

    prnstr(" ", "\n", false);
    prnstr(
        &format!(
            "The number of patches for Corrsearch3d is {npatchx} in X, {npatchz} in Y, and {npatchy} in Z"
        ),
        "\n",
        false,
    );
    prnstr(
        "   (Y and Z are not flipped in entries to corrsearch3d in patchcorr.com)",
        "\n",
        false,
    );

    make_backup_file("combine.com");
    make_backup_file(&solvematch);
    make_backup_file(&dualvolmatch);
    make_backup_file(&matchvol1);
    make_backup_file(&matchvol2);
    make_backup_file(&warpvol);
    make_backup_file(&patchcorr);
    make_backup_file(&matchorwarp);
    make_backup_file(&volcombine);

    let base_changes = vec![
        format!("s/{from_srcname}.rec/{origfile}/g"),
        format!("s/{to_srcname}.rec/{recfile}/g"),
        format!("s/{from_srcname}.matmod/{fromname}.matmod/g"),
        format!("s/{from_srcname}.mat/{matfile}/g"),
        format!("s/{from_srcname}/{fromname}/g"),
        format!("s/{to_srcname}/{toname}/g"),
        format!("s/{matfile}/{sedmatfile}/g"),
    ];

    let mut sedcom = base_changes.clone();
    sedcom.extend([
        format!("/^ACorrespondenceList/s/[ \t].*/\t{corrlist1}/"),
        format!("/^BCorrespondenceList/s/[ \t].*/\t{corrlist2}/"),
        format!(
            "/^XAxisTilts/s/[ \t].*/\t{},{}/",
            py_str_float(xaxisa),
            py_str_float(xaxisb)
        ),
        format!(
            "/^AngleOffsets/s/[ \t].*/\t{},{}/",
            py_str_float(angoffa),
            py_str_float(angoffb)
        ),
        format!(
            "/^ZShifts/s/[ \t].*/\t{},{}/",
            py_str_float(zshifta),
            py_str_float(zshiftb)
        ),
        format!("/^SurfacesOrUseModel/s/[ \t].*/\t{modsurf}/"),
        format!("/{sedtranskey}/s/.*/TransferCoordinateFile\t{transfile}/"),
        format!("/^UsePoints/s/[ \t].*/\t{uselist}/"),
        format!("/^MatchingAtoB/s/[ \t].*/\t{matchatob}/"),
    ]);
    edit_apply_change_list_and_write(&g, &mut sedcom, &solvematch, true);

    let mut sedcom = vec![
        format!("s/{root_srcname}/{rootname}/g"),
        format!("/^MatchAtoB/s/[ \t].*/\t{matchatob}/"),
    ];
    sedcom.extend(sed_del_and_add(
        "NamingStyle",
        &g.name_style.to_string(),
        "RootName",
        '/',
    ));
    edit_apply_change_list_and_write(&g, &mut sedcom, &dualvolmatch, true);

    let mut sedcom = base_changes.clone();
    sedcom.extend([
        format!("s/{tmp_srcname}/{sedtmpdir}/g"),
        format!("/mkdir/s/.*if/{ifmkdir}/"),
        format!("/{sedtempkey}/s/#//"),
        format!(r"/OutputSizeXYZ/s/\//{nx} {nz} {ny}/g"),
    ]);
    edit_apply_change_list_and_write(&g, &mut sedcom, &matchvol2, false);

    edit_apply_change_list_and_write(&g, &mut sedcom, &warpvol, false);

    sedcom.push(format!("/savework-file/s//{backupsed}/g"));
    edit_apply_change_list_and_write(&g, &mut sedcom, &matchvol1, false);

    let mut sedcom = base_changes.clone();
    sedcom.extend([
        format!("/^PatchSizeXYZ/s/[ \t].*/\t{patchnx},{patchnz},{patchny}/"),
        format!("/^NumberOfPatchesXYZ/s/[ \t].*/\t{npatchx},{npatchz},{npatchy}/"),
        format!("/^XMinAndMax/s/[ \t].*/\t{patchxl},{patchxu}/"),
        format!("/^YMinAndMax/s/[ \t].*/\t{patchzl},{patchzu}/"),
        format!("/^ZMinAndMax/s/[ \t].*/\t{patchyl},{patchyu}/"),
        format!("/^BSourceBorder/s/[ \t].*/\t{xyborder},{xyborder}/"),
        format!("/^RegionModel/s/[ \t].*/\t{regionmod}/"),
        format!("/^{delregion}/d"),
    ]);

    // If either angle is there and either stack size, add angle, then stack sizes
    if recons_centered && (a_angle_true || b_angle_true) && (nx_astack_true || nx_bstack_true) {
        if nx_astack_true || nx_bstack_true {
            sedcom.push(format!(
                "/^FlipYZMes/a/AxisRotationAngle\t{}/",
                py_fixed(axis_angle, 0, 2)
            ));
        }
        if nx_astack_true {
            sedcom.push(format!(
                "/^FlipYZMes/a/TiltSeriesSizeXY\t{},{}/",
                nx_astack.unwrap(),
                ny_astack.unwrap()
            ));
        }
        if nx_bstack_true {
            sedcom.push(format!(
                "/^FlipYZMes/a/BTiltSeriesSizeXY\t{},{}/",
                nx_bstack.unwrap(),
                ny_bstack.unwrap()
            ));
        }
    }

    // Add entries for autopatchfit
    if !g.auto_patch.is_empty() {
        sedcom.push("/^FlipYZMes/a/LocalSDNumBinnings\t4/".to_owned());
        sedcom.push("/^FlipYZMes/a/BoxSizeForLocalSD\t32,8,32/".to_owned());
        sedcom.push("/^FlipYZMes/a/EliminateByLocalSD\t1,0.5/".to_owned());
    }

    edit_apply_change_list_and_write(&g, &mut sedcom, &patchcorr, true);

    // Make matchorwarp
    let mut sedcom = base_changes.clone();
    sedcom.extend([
        format!("s/{tmp_srcname}/{sedtmpdir}/g"),
        format!("/mkdir/s/.*if/{ifmkdir}/"),
        format!("/{sedtempkey}/s/#//"),
    ]);
    if !g.auto_patch.is_empty() {
        sedcom.push("/^WarpLimits/a/ExtentToFit\t3/".to_owned());
        sedcom.push("/^WarpLimits/a/StructureCriteria\t0.5,0.57,0.65/".to_owned());
    }
    if !regionmod.is_empty() {
        sedcom.push(format!("/^WarpLimits/a/ModelFile\t{regionmod}/"));
    }

    edit_apply_change_list_and_write(&g, &mut sedcom, &matchorwarp, true);

    // Do combine.com
    make_combine_com(&g);

    // Start volcombine, get as strings
    let mut sedcom = base_changes.clone();
    sedcom.extend([
        format!("/set combinefft_lowboth =/s/=.*/= {low_radius_text}/"),
        format!("/set combinefft_reduce =/s/=.*/= {wedge_reduction_text}/"),
        format!("/savework-file/s//{backupsed}/g"),
        format!("/####CreatedVersion####/s/# .*/#{}/", g.imod_version),
    ]);
    let volcombine_root = volcombine
        .chars()
        .take(volcombine.chars().count().saturating_sub(4))
        .collect::<String>();
    let mut voltext = pysed(
        &sedcom,
        PysedSrc::File(&format!("{}/{volcombine_root}.com", g.srcdir)),
        None,
        false,
        '/',
        false,
    )
    .ok()
    .flatten()
    .unwrap_or_default();
    add_output_format_var_to_lines(&mut voltext, g.name_style, None, false);

    // If we really want this to run with no recfile, supply the test size to -tomo
    let megamax = MAXPIXELS.div_euclid(1000);
    let mut chunk_arg = "";
    if g.chunked_hdf != 0 {
        chunk_arg = "-hdf";
    }
    let tomocom = format!(
        "tomopieces -tomo \"{recfile}\" -mega {megamax} -xpad {TAPERPADXZ} -ypad {TAPERPADY} -zpad {TAPERPADXZ}  -min {MINOVERLAP} -ymax {MAXPIECEY} {chunk_arg}"
    );
    // Direct call (owner rule, 2026-09-26): tomopieces returns its ranges and
    // `ranlist` is built from them rather than from its captured output
    // (`setupcombine:700`).  The values are integers the script copies into
    // volcombine.com as text, so the lines are exactly what it printed; the
    // PIP-fallback lines the loop below eats can no longer be among them.
    // Tomopieces still reads its options through PIP from the words of
    // `tomocom`.
    let megamax_text = megamax.to_string();
    let mut words: Vec<String> = vec![
        "tomopieces".to_owned(),
        "-tomo".to_owned(),
        recfile.clone(),
        "-mega".to_owned(),
        megamax_text,
    ];
    words.extend(
        format!(
            "-xpad {TAPERPADXZ} -ypad {TAPERPADY} -zpad {TAPERPADXZ}  -min {MINOVERLAP} -ymax {MAXPIECEY} {chunk_arg}"
        )
        .split_whitespace()
        .map(str::to_owned),
    );
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    let sink = std::sync::Arc::new(std::sync::Mutex::new(None));
    let recorder = std::sync::Arc::clone(&sink);
    g.ranlist = match call_own_program(&tomocom, &words, None, true, move || {
        tomopieces_recording(recorder)
    }) {
        Ok(_) => match sink.lock().expect("tomopieces result").take() {
            Some(result) => tomopieces_output_lines(&result),
            None => Vec::new(),
        },
        Err(_) => exit_from_imod_error(progname),
    };

    for i in 0..g.ranlist.len() {
        g.ranlist[i] = g.ranlist[i].trim_end_matches(['\r', '\n']).to_owned();
    }

    // Eat any lines with PIP fallback output
    while !g.ranlist.is_empty() {
        let first = &g.ranlist[0];
        if first.trim().is_empty()
            || first.contains('a')
            || first.contains('e')
            || first.contains('i')
            || first.contains('o')
        {
            g.ranlist.remove(0);
        } else {
            break;
        }
    }

    // Set up filltomo command and check all the conditions for specifying valid area
    let mut fillcom = vec![
        format!("MatchedToTomogram\t{recfile}"),
        format!("SourceTomogram\t{origfile}"),
        format!("InverseTransformFile\t{invfile}"),
    ];

    if recons_centered {
        // Get the size of the raw stack from montage or stack
        // If anything goes wrong, just forget it
        let mut raw_size = String::new();
        if Path::new(&format!("newst{fromlet}{com_ext}")).exists() {
            if let Ok((nxraw, nyraw, nzraw)) = get_mrc_size(&format!("{fromname}.{}", g.stack_ext))
            {
                raw_size = format!("{nxraw} {nyraw} {nzraw}");
            }
        } else if Path::new(&format!("blend{fromlet}{com_ext}")).exists() {
            if let Ok((nxraw, nyraw, nzraw)) = get_montage_size(
                &format!("{fromname}.{}", g.stack_ext),
                Some(&format!("{fromname}.pl")),
            ) {
                raw_size = format!("{nxraw} {nyraw} {nzraw}");
            }
        }

        // Things are good, add the source commands
        if !raw_size.is_empty() {
            fillcom.extend([
                format!("ImagesAreBinned\t{binningb}"),
                format!("SourceRawStackSize\t{raw_size}"),
                format!("SourceStackTransforms\t{fromname}.xf"),
            ]);
        }
    }

    voltext.extend([
        "$set nonomatch".to_owned(),
        format!(
            "$\\rm -f {tmppath}*.mat~ {tmppath}mat.fft* {tmppath}rec.fft* {tmppath}sum.fft* {sumname}* {tmppath}sum[1-9]*.rec* {tmppath}sum[1-9]*_rec.{type_ext}*"
        ),
        "#".to_owned(),
        "$echo STATUS: RUNNING FILLTOMO TO FILL IN GRAY AREAS IN THE .MAT FILE".to_owned(),
        "$echo ".to_owned(),
        "#".to_owned(),
        "$filltomo -StandardInput".to_owned(),
        format!("FillTomogram\t{tmpmatfile}"),
    ]);
    voltext.extend(fillcom.iter().cloned());
    let int_field = |line: &str, index: usize| -> i64 {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        let Some(field) = fields.get(index) else {
            py_crash("IndexError: list index out of range");
        };
        match py_int(field) {
            Some(value) => value,
            None => py_crash(&format!(
                "ValueError: invalid literal for int() with base 10: '{field}'"
            )),
        }
    };
    let Some(first_line) = g.ranlist.first().cloned() else {
        py_crash("IndexError: list index out of range");
    };
    let npiecex = int_field(&first_line, 0);
    let npiecey = int_field(&first_line, 1);
    let npiecez = int_field(&first_line, 2);
    g.ranind = 1;
    let ranline = |g: &Globals, index: usize| -> String {
        match g.ranlist.get(index) {
            Some(line) => line.clone(),
            None => py_crash("IndexError: list index out of range"),
        }
    };
    if g.chunked_hdf != 0 {
        let line = ranline(&g, 1);
        let nx_chunk = int_field(&line, 0);
        let ny_chunk = int_field(&line, 1);
        let nz_chunk = int_field(&line, 2);
        g.ranind = 2;
        let _ixyz = ranline(&g, g.ranind);
        voltext.extend([
            "#".to_owned(),
            "$echo STATUS: INITIALIZING CHUNKED OUTPUT FILE".to_owned(),
        ]);
        voltext.extend(common_fft_lines(&g, 1));
        voltext.push(format!("ChunkSizeForHDF\t{nx_chunk} {ny_chunk} {nz_chunk}"));
    }

    let mut sumnum: i64 = 0;
    let npiecetot = npiecex * npiecey * npiecez;
    for _znum in 0..npiecez {
        for _ynum in 0..npiecey {
            for _xnum in 0..npiecex {
                sumnum += 1;
                voltext.extend([
                    "#".to_owned(),
                    format!("$dopiece{sumnum}:"),
                    format!(
                        "$echo STATUS: EXTRACTING AND COMBINING PIECE  {sumnum}  of {npiecetot}"
                    ),
                ]);
                voltext.extend(common_fft_lines(&g, sumnum));
                g.ranind += 1;
                voltext.extend([
                    format!("InverseTransformFile\t{invfile}"),
                    format!("ATiltFile\t{atlt}"),
                    format!("BTiltFile\t{btlt}"),
                    "ReductionFraction\t$combinefft_reduce".to_owned(),
                    "LowFromBothRadius\t$combinefft_lowboth".to_owned(),
                ]);

                if g.chunked_hdf != 0 {
                    let line = ranline(&g, g.ranind);
                    let mmpl = line.split(',').collect::<Vec<_>>();
                    if mmpl.len() < 9 {
                        py_crash("IndexError: list index out of range");
                    }
                    g.ranind += 1;
                    voltext.extend([
                        format!("XSaveStartAndEnd\t{} {}", mmpl[0], mmpl[1]),
                        format!("YSaveStartAndEnd\t{} {}", mmpl[2], mmpl[3]),
                        format!("ZSaveStartAndEnd\t{} {}", mmpl[4], mmpl[5]),
                        format!("PlaceChunkAtXYZ\t{} {} {}", mmpl[6], mmpl[7], mmpl[8]),
                    ]);
                }
            }
        }
    }

    voltext.extend([
        "#".to_owned(),
        "$echo STATUS: REASSEMBLING PIECES".to_owned(),
    ]);
    if g.chunked_hdf == 0 {
        voltext.extend([
            "$echo".to_owned(),
            "#".to_owned(),
            "$assemblevol -StandardInput".to_owned(),
            format!("OutputFile {sumname}"),
        ]);

        for _xnum in 0..npiecex {
            voltext.push(format!("StartEndToExtractInX {}", ranline(&g, g.ranind)));
            g.ranind += 1;
        }
        for _ynum in 0..npiecey {
            voltext.push(format!("StartEndToExtractInY {}", ranline(&g, g.ranind)));
            g.ranind += 1;
        }
        for _znum in 0..npiecez {
            voltext.push(format!("StartEndToExtractInZ {}", ranline(&g, g.ranind)));
            g.ranind += 1;
        }

        for sumnum in 0..npiecetot.max(0) {
            voltext.push(format!(
                "InputFile {tmppath}{}",
                dataset_filename(".rec", Some(&format!("sum{}", sumnum + 1)), None)
            ));
        }
    }

    voltext.extend([
        "#".to_owned(),
        "$echo ".to_owned(),
        "$echo STATUS: RUNNING FILLTOMO ON FINAL VOLUME".to_owned(),
        "$echo ".to_owned(),
        "#".to_owned(),
        format!("$\\rm -f {tmppath}sum[1-9]*.rec {tmppath}sum[1-9]*_rec.{type_ext}*"),
        "$filltomo -StandardInput".to_owned(),
        format!("FillTomogram\t{sumname}"),
    ]);
    voltext.extend(fillcom.iter().cloned());

    if !tmpdir.is_empty() && handclean == 0 {
        voltext.push(format!("$\\rm -r {tmpdir}"));
    }
    voltext.push(format!("$if (-e {backupname}) {backupname}"));

    let mut text = String::new();
    for line in &voltext {
        text.push_str(line);
        text.push('\n');
    }
    if std::fs::write(&volcombine, text).is_err() {
        exit_error(&format!("Opening or writing to {volcombine}"));
    }

    let _ = std::io::stdout().flush();
    0
}
