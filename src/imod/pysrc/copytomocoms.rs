//! Translation of `IMOD/pysrc/copytomocoms`.
//!
//! A Python command script: its seven functions are translated one for one
//! below, and its top level is [`copytomocoms`], translated statement by
//! statement.  The module globals those functions read or extend
//! (`srcdir`, `dstext`, `backuplist`, `changeList`, `nameStyle`, `setlet`,
//! `origComDir`, `dfltComDir`, `srcname`, `stackname`, `srcToDest`, `first`,
//! `zsize`, `IMODversion`) are the fields of [`Globals`], which the top level
//! owns and passes to them.
//!
//! The templates are read from `$IMOD_DIR/com`, as the source does
//! (`srcdir = os.path.join(IMOD_DIR, 'com')`); an installation must point
//! `IMOD_DIR` at a directory holding IMOD's `com` templates.
//!
//! `header`, `alterheader` and `clip` are this crate's programs and run in
//! process through `runcmd`; `montagesize`, `goodframe`, `extracttilts`,
//! `extractpieces` and `extractmagrad` run in process once they are in the
//! command table and as processes until then.
//!
//! Python values: PIP floats are doubles and `'{}'.format(x)` of one is its
//! `repr` ([`py_str_float`]); `//` floors (every divisor here is a positive
//! literal, so `div_euclid`); `round()` rounds half to even; `int()` of a
//! float truncates.  Values the script leaves dynamically typed -- the
//! voltage, Cs, defocus and `reversed` entries, which may be a number, a
//! directive's string, or `None` -- are [`PyVal`]s.

use super::batchruntomo::{PyVal, py_str_float};
use super::comchanger::{
    Change, get_setupset_value, modify_for_change_list, process_change_options,
};
use super::imodpy::{
    ImodpyError, MrcInfo, OptionValue, add_imod_bin_ignore_sighup, add_output_format_var_to_lines,
    allowed_raw_stack_extensions, call_own_program, com_extension_from_option, dataset_filename,
    default_naming_style, exit_from_imod_error, get_image_format, get_imod_version, get_mrc,
    get_naming_style, header_in_process, make_backup_file, make_current_dir_writable, option_value,
    prnstr, read_text_file, run_cmd, run_goodframe, set_root_and_extension,
    standard_type_extensions, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_two_floats, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add};
use crate::imod::clip::processing::clip_recording_stat;
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The module globals the script's functions use (see the module comment).
pub struct Globals {
    /// `srcdir`
    pub srcdir: String,
    /// `srcext`
    pub srcext: String,
    /// `dstext`
    pub dstext: String,
    /// `backuplist`
    pub backuplist: Vec<String>,
    /// `backupline`
    pub backupline: String,
    /// `IMODversion`
    pub imod_version: String,
    /// `changeList`
    pub change_list: Vec<Change>,
    /// `nameStyle`
    pub name_style: i32,
    /// `setlet`
    pub setlet: String,
    /// `origComDir`
    pub orig_com_dir: String,
    /// `dfltComDir`
    pub dflt_com_dir: String,
    /// `srcname`
    pub srcname: String,
    /// `stackname`
    pub stackname: String,
    /// `srcToDest`
    pub src_to_dest: String,
    /// `first`
    pub first: f64,
    /// `zsize`: an `int` from the header, or the `str` that the montage
    /// branch assigns from `montagesize` output (`copytomocoms:657`).
    pub zsize: PyVal,
}

/// `os.path.join(a, b)` (`posixpath.join`) for the two-part joins the
/// script makes.
pub fn path_join(a: &str, b: &str) -> String {
    if b.starts_with('/') || a.is_empty() {
        b.to_owned()
    } else if a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// Matches `writeComFile` (`copytomocoms:38`): write a com file with the given
/// lines, and write it to the origcoms directory if that is still defined.
pub fn write_com_file(g: &Globals, dstfile: &str, sedlines: &[String]) {
    let _ = write_text_file(dstfile, sedlines, false);
    if !g.orig_com_dir.is_empty() {
        let _ = write_text_file(&path_join(&g.orig_com_dir, dstfile), sedlines, false);
    }
}

/// Matches `applyChangeListAndWrite` (`copytomocoms:45`): if there are
/// directive changes, first write the default file, then apply changes.
///
/// The source's `addOutputFormatVarToLines` inserts into the caller's list;
/// no caller reads its list afterwards, so it is taken by value here.
pub fn apply_change_list_and_write(
    g: &Globals,
    dst_root: &str,
    mut sedlines: Vec<String>,
    dstfile: &str,
    do_changes: bool,
    output_format: Option<&str>,
) {
    // set output format if appropriate
    add_output_format_var_to_lines(&mut sedlines, g.name_style, output_format, false);

    if do_changes && !g.change_list.is_empty() {
        let temp_lines = sedlines.clone();
        sedlines =
            match modify_for_change_list(&sedlines, dst_root, &g.setlet, &g.change_list, false) {
                Ok(lines) => lines,
                Err(message) => exit_error(&message),
            };
        if !g.dflt_com_dir.is_empty() && temp_lines != sedlines {
            let _ = write_text_file(&path_join(&g.dflt_com_dir, dstfile), &temp_lines, false);
        }
    }
    write_com_file(g, dstfile, &sedlines);
}

/// Matches `editAndWrite` (`copytomocoms:61`): define source and destination
/// files, back up the destination file and add it to the backuplist, run
/// pysed, add the savework command to the output, and write the output file.
pub fn edit_and_write(
    g: &mut Globals,
    src_root: &str,
    dst_root: &str,
    mut sedcom: Vec<String>,
    do_changes: bool,
    output_format: Option<&str>,
) {
    let srcfile = path_join(&g.srcdir, &format!("{src_root}{}", g.srcext));
    let dstfile = format!("{dst_root}{}", g.dstext);
    g.backuplist.push(dstfile.clone());
    make_backup_file(&dstfile);
    sedcom.push(format!(
        "/####CreatedVersion####/s/# .*/#{}/",
        g.imod_version
    ));
    let mut sedlines = pysed(&sedcom, PysedSrc::File(&srcfile), None, false, '/', false)
        .ok()
        .flatten()
        .unwrap_or_default();
    sedlines.push(g.backupline.clone());

    apply_change_list_and_write(g, dst_root, sedlines, &dstfile, do_changes, output_format);
}

/// Matches `isFileFromFEI` (`copytomocoms:74`): look for the Fei title to
/// determine if it is an FEI file; also returns a pixel size from an mdoc.
pub fn is_file_from_fei(progname: &str, filename: &str) -> (bool, f64) {
    let mut retval = false;
    let mut pixel = 0.;
    // Direct call (owner rule, 2026-09-26): `header_in_process` builds the
    // lines `header` prints -- titles and the mdoc pixel-size line included
    // -- from the file, without running the program (`copytomocoms:85`).
    let headout = match header_in_process(&format!("header \"{filename}\""), filename, false, None)
    {
        Ok(lines) => lines,
        Err(_) => exit_from_imod_error(progname),
    };

    for l in &headout {
        if l.starts_with("Fei Company") {
            retval = true;
        }
        if l.contains("Pixel size in nanometers") && l.contains("from mdoc") {
            let lsplit = l.split_whitespace().collect::<Vec<_>>();
            if lsplit.len() >= 6 && lsplit[4] == "=" {
                if let Ok(value) = lsplit[5].parse::<f64>() {
                    pixel = value * 10.;
                }
            }
        }
    }

    (retval, pixel)
}

/// Matches `anglesPassThrough` (`copytomocoms:97`): determine whether tilt
/// angles pass through 0, 90, and -90 with given increment.
pub fn angles_pass_through(g: &Globals, inc_test: f64) -> (bool, bool, bool) {
    let mut pass0 = false;
    let mut pass90 = false;
    let mut pass_min90 = false;
    let mut last_tilt = g.first;
    let zsize = match &g.zsize {
        PyVal::Int(value) => *value,
        // `range(1, zsize)` with the `str` the montage branch stored
        _ => {
            eprintln!("Traceback (most recent call last):");
            eprintln!("TypeError: 'str' object cannot be interpreted as an integer");
            std::process::exit(1)
        }
    };
    for ind in 1..zsize {
        let tilt = g.first + ind as f64 * inc_test;
        if last_tilt.min(tilt) <= 0. && tilt.max(last_tilt) > 0. {
            pass0 = true;
        }
        if last_tilt.min(tilt) <= 90. && tilt.max(last_tilt) > 90. {
            pass90 = true;
        }
        if last_tilt.min(tilt) <= -90. && tilt.max(last_tilt) > -90. {
            pass_min90 = true;
        }
        last_tilt = tilt;
    }

    (pass0, pass90, pass_min90)
}

/// Matches `sedForImageFiles` (`copytomocoms:117`): return set of sed strings
/// for changing filenames given some extensions to handle, which would
/// include a . and possibly text before the dot.
pub fn sed_for_image_files(g: &Globals, extensions: &[&str]) -> Vec<String> {
    // Do the stack
    let mut sedcoms = vec![format!("s/{}.st/{}/g", g.srcname, g.stackname)];

    // Do each extension and end with the generic change for non-image files
    for ext in extensions {
        sedcoms.push(format!(
            "s/{}{}/{}/g",
            g.srcname,
            ext,
            dataset_filename(ext, None, None)
        ));
    }
    sedcoms.push(g.src_to_dest.clone());
    sedcoms
}

/// Matches `warning` (`copytomocoms:134`): prints to stderr AND adds another
/// newline, since etomo parses this output for multiline warnings terminated
/// by a blank line.
pub fn warning(text: &str) {
    eprint!("\nWARNING: {text}\n\n");
}

/// Matches `printInfo` (`copytomocoms:138`): INFO lines are output to the
/// etomo_err log but only selected ones go to the project log.
pub fn print_info(text: &str) {
    eprint!("\nINFO: {text}\n\n");
}

/// The script's top level (`copytomocoms:142-1578`).  Returns the status of
/// its final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn copytomocoms(arguments: &[OsString]) -> i32 {
    let progname = "copytomocoms";
    let prefix = format!("ERROR: {progname} - ");

    const SAMPLESIZE: i32 = 20; // Size of sample tomogram
    const SAMPALISIZE: i32 = 72; // Size of aligned stack for sampling
    const MINBORDER1: i32 = 33; // Minimum border size
    const MINBORDER2: i32 = 43; // Minimum size if y dimension > borderstep1
    const BORDERSTEP1: i32 = 500; // threshold for using minborder2
    const BORDERSTEP2: i32 = 1000; // threshold for linearly increasing size
    const BORDERFAC: i32 = 10; // controls linear increase of border size
    const CCDBORDERFAC: f64 = 2.0; // extra size of border for CCD images
    const RESIDUAL_SCALE: i32 = 10; // scaling for tiltalign residuals
    const MONTAGE_XCORR_MAX: i32 = 2048; // Max size for blended stack for doing xcorr
    let backupname = "./savework";
    const TILTSIZES: [i32; 10] = [950, 1400, 1900, 2850, 3800, 5700, 7600, 11400, 15200, 22800];
    const TILTSCALES: [i32; 10] = [1000, 700, 500, 330, 250, 170, 125, 80, 60, 40];

    let single_srcname = "g5a";
    let seedname = "empty.seed";
    let later_bname = "laterbsetup.com";
    let mut later_blines: Vec<String> = Vec::new();
    let formats_for_ext = |ext: &str| -> Option<&'static str> {
        match ext {
            "MRC" => Some("MRC"),
            "HDF" => Some("HDF"),
            "TIF" => Some("TIF"),
            "TIFF" => Some("TIF"),
            _ => None,
        }
    };
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

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

    let mut g = Globals {
        srcdir: String::new(),
        srcext: ".com".to_owned(),
        dstext: String::new(),
        backuplist: Vec::new(),
        backupline: format!("$if (-e {backupname}) {backupname}"),
        imod_version: String::new(),
        change_list: Vec::new(),
        name_style: 0,
        setlet: String::new(),
        orig_com_dir: "origcoms".to_owned(),
        dflt_com_dir: "dfltcoms".to_owned(),
        srcname: String::new(),
        stackname: String::new(),
        src_to_dest: String::new(),
        first: 0.,
        zsize: PyVal::None,
    };

    let mut subdistort = "gibberish".to_owned();
    let mut voltsub = "gibberish";
    let mut sphersub = "gibberish";
    let mut defocsub = "gibberish";
    let mut ctfconfsub = "gibberish";
    let mut logbase = 0;
    g.srcname = single_srcname.to_owned();
    let possible_stack_exts = allowed_raw_stack_extensions();

    // Set the com directory
    g.srcdir = path_join(&imod_dir, "com");
    if !Path::new(&g.srcdir).exists() {
        exit_error(&format!(
            "Source directory for command files, {}, not found",
            g.srcdir
        ));
    }

    // Get IMOD version and day created
    match get_imod_version() {
        Some(version) if !version.is_empty() => g.imod_version = version,
        _ => exit_error("Getting IMOD version"),
    }

    // `(datetime.now() - datetime.strptime('1/1/2020', '%m/%d/%Y')).days`: both
    // are naive local times, so this is the number of local calendar days
    // since 1 January 2020
    let created_day = {
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe {
            libc::localtime_r(&seconds, &mut tm);
        }
        // Days from the civil date to 1970-01-01 (proleptic Gregorian)
        let days_from_civil = |year: i64, month: i64, day: i64| -> i64 {
            let year = if month <= 2 { year - 1 } else { year };
            let era = year.div_euclid(400);
            let yoe = year - era * 400;
            let mp = (month + 9) % 12;
            let doy = (153 * mp + 2) / 5 + day - 1;
            let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
            era * 146097 + doe - 719468
        };
        days_from_civil(
            1900 + tm.tm_year as i64,
            1 + tm.tm_mon as i64,
            tm.tm_mday as i64,
        ) - days_from_civil(2020, 1, 1)
    };

    // Fallbacks from ../manpages/autodoc2man 3 1 copytomocoms
    let options: Vec<String> = [
        "name:RootName:FN:",
        "stackext:StackExtension:CH:",
        "dual:DualAxis:B:",
        "montage:MontagedImages:B:",
        "backup:BackupDirectory:FN:",
        "pixel:PixelSize:F:",
        "gold:GoldBeadSize:F:",
        "rotation:RotationAngle:F:",
        "brotation:BRotationAngle:F:",
        "firstinc:FirstAndIncAngle:FP:",
        "bfirstinc:BFirstAndIncAngle:FP:",
        "userawtlt:UseRawtltFile:B:",
        "buserawtlt:BUseRawtltFile:B:",
        "extract:ExtractAngles:B:",
        "bextract:BExtractAngles:B:",
        "angles:TiltAngles:LI:",
        "bangles:BTiltAngles:LI:",
        "twodir:TwoDirectionsAngle:F:",
        "btwodir:BTwoDirectionsAngle:F:",
        "reversed:ReversedBidirectional:B:",
        "breversed:BReversedBidirectional:B:",
        "dosesym:DoseSymmetricAngle:F:",
        "bdosesym:BDoseSymmetricAngle:F:",
        "skip:ViewsToSkip:LI:",
        "bskip:BViewsToSkip:LI:",
        "distort:DistortionField:FN:",
        "binning:BinningOfImages:F:",
        "gradient:GradientTable:FN:",
        "focus:FocusWasAdjusted:B:",
        "bfocus:BFocusWasAdjusted:B:",
        "voltage:VoltageInKV:I:",
        "Cs:SphericalAberration:F:",
        "ctfnoise:NoiseConfigFile:FN:",
        "defocus:Defocus:F:",
        "CTFfiles:CopyCTFfiles:I:",
        "fei:SetFEIPixelSize:B:",
        "change:ChangeParametersFile:FNM:",
        "one:OneParameterChange:CHM:",
        "style:NamingStyle:I:",
        "halffloat:HalfFloatModeOutput:I:",
        "pcm:MakeComExtensionPcm:I:",
        "xsize:XImageSize:I:",
        "ysize:YImageSize:I:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 0);

    // Get options
    let rootname = pip_get_string("RootName", "").unwrap_or_default();
    if rootname.is_empty() {
        exit_error("A root name must be entered");
    }

    let naxis = pip_get_boolean("DualAxis", 0).unwrap_or(0) + 1;
    let montage = pip_get_boolean("MontagedImages", 0).unwrap_or(0);
    let backupdir = pip_get_string("BackupDirectory", "").unwrap_or_default();
    let backup = 1 - pip_get_err_no();
    let mut xsize = pip_get_integer("XImageSize", -1).unwrap_or(-1);
    let mut ysize = pip_get_integer("YImageSize", -1).unwrap_or(-1);
    let set_fei_pixel = pip_get_boolean("SetFEIPixelSize", 0).unwrap_or(0);
    let mut stack_ext = pip_get_string("StackExtension", "").unwrap_or_default();

    let mut make_pcm = 0;
    if std::env::var("TEST_USE_PCM_FOR_COM").is_ok_and(|value| !value.is_empty()) {
        make_pcm = 1;
    }
    let out_com_ext = com_extension_from_option(make_pcm);

    let (name_style_dflt, mut file_type_ext) = default_naming_style();
    let mut env_name_style = -1;
    if let Ok(value) = std::env::var("TEST_NAMING_STYLE")
        && !value.is_empty()
    {
        if let Ok(style) = value.trim().parse::<i32>() {
            env_name_style = style;
            if env_name_style >= 0 && (env_name_style as usize) < standard_type_extensions().len() {
                file_type_ext = standard_type_extensions()[env_name_style as usize].clone();
            } else {
                env_name_style = -1;
            }
        }
    }

    let (mut name_style, style_ext) = match get_naming_style(Some(&file_type_ext), false, false) {
        Ok(result) => result,
        Err(message) => exit_error(&message),
    };
    let file_type_ext = style_ext.unwrap_or_default();
    if name_style < 0 && env_name_style >= 0 {
        name_style = env_name_style;
    }
    if name_style < 0 {
        name_style = name_style_dflt;
    }
    g.name_style = name_style;

    // Get the change list and eliminate the option that kept it from being applied to newst
    g.change_list = process_change_options(
        "ChangeParametersFile",
        "OneParameterChange",
        "comparam",
        4,
        false,
    );
    for ind in (0..g.change_list.len()).rev() {
        if g.change_list[ind][0].starts_with("newst")
            && g.change_list[ind][1] == "newstack"
            && g.change_list[ind][2] == "SizeToOutputInXandY"
        {
            g.change_list.remove(ind);
        }
    }

    let copy_arg_list = process_change_options(
        "ChangeParametersFile",
        "OneParameterChange",
        "setupset",
        -3,
        false,
    );

    let pixsize = pip_get_float("PixelSize", -1.).unwrap_or(-1.);
    if pixsize < 0. {
        exit_error("You must enter a pixel size in nanometers");
    }

    let beadnm = pip_get_float("GoldBeadSize", -1.).unwrap_or(-1.);
    if beadnm < 0. {
        exit_error("You must enter a bead size in nanometers");
    }

    let mut axisangle = pip_get_float("RotationAngle", 0.).unwrap_or(0.);
    if pip_get_err_no() != 0 {
        exit_error("You must enter an axis rotation angle");
    }

    let baxisangle = pip_get_float("BRotationAngle", axisangle).unwrap_or(axisangle);
    //if PipGetErrNo() and naxis > 1:

    let mut angles = pip_get_string("TiltAngles", "").unwrap_or_default();
    let ifangles = 1 - pip_get_err_no();
    let bangles = pip_get_string("BTiltAngles", "").unwrap_or_default();
    let ifbangles = 1 - pip_get_err_no();
    let (first, mut increm) = pip_get_two_floats("FirstAndIncAngle", (0., 0.)).unwrap_or((0., 0.));
    g.first = first;
    let mut firstinc = 1 - pip_get_err_no();
    let (bfirst, bincrem) = pip_get_two_floats("BFirstAndIncAngle", (0., 0.)).unwrap_or((0., 0.));
    let bfirstinc = 1 - pip_get_err_no();
    let mut userawtlt = pip_get_boolean("UseRawtltFile", 0).unwrap_or(0);
    let mut buserawtlt = pip_get_boolean("BUseRawtltFile", 0).unwrap_or(0);
    let mut ifextract = pip_get_boolean("ExtractAngles", 0).unwrap_or(0);
    if ifextract != 0 {
        userawtlt = 1;
    }
    let bifextract = pip_get_boolean("BExtractAngles", 0).unwrap_or(0);
    if bifextract != 0 {
        buserawtlt = 1;
    }

    if ifangles + firstinc + userawtlt != 1 {
        if ifextract != 0 {
            exit_error("You must enter one and only one of -angles, -firstinc, and -extract");
        }
        exit_error("You must enter one and only one of -angles, -firstinc, and -userawtlt");
    }
    if naxis > 1 && ifbangles + bfirstinc + buserawtlt != 1 {
        if bifextract != 0 {
            exit_error("You must enter one and only one of -bangles, -bfirstinc, and -bextract");
        }
        exit_error("You must enter one and only one of -bangles, -bfirstinc, and -buserawtlt");
    }

    let mut two_dir_angle = pip_get_float("TwoDirectionsAngle", 0.).unwrap_or(0.);
    let mut two_dir = 1 - pip_get_err_no();
    let btwo_dir_angle = pip_get_float("BTwoDirectionsAngle", 0.).unwrap_or(0.);
    let btwo_dir = 1 - pip_get_err_no();
    let mut dose_sym_angle = pip_get_float("DoseSymmetricAngle", 0.).unwrap_or(0.);
    let mut dose_sym = 1 - pip_get_err_no();
    // Fixed in translation (BUGS.md): `copytomocoms:316` reads
    // `DoseSymmetricAngle` a second time, so native never reads -bdosesym and
    // the B axis inherits the A axis's value; this reads the B option.
    let bdose_sym_angle = pip_get_float("BDoseSymmetricAngle", 0.).unwrap_or(0.);
    let bdose_sym = 1 - pip_get_err_no();
    if (two_dir != 0 && dose_sym != 0) || (btwo_dir != 0 && bdose_sym != 0) {
        exit_error("You cannot enter both -twodir and -dosesym");
    }
    let mut excludelistin = pip_get_string("ViewsToSkip", "").unwrap_or_default();
    let bexcludelistin = pip_get_string("BViewsToSkip", "").unwrap_or_default();
    let mut distort = pip_get_string("DistortionField", "").unwrap_or_default();
    let binning = pip_get_float("BinningOfImages", 1.).unwrap_or(1.);
    let gradient = pip_get_string("GradientTable", "").unwrap_or_default();
    let mut focus_adj = pip_get_boolean("FocusWasAdjusted", 0).unwrap_or(0);
    let b_focus_adj = pip_get_boolean("BFocusWasAdjusted", 0).unwrap_or(0);

    // try: ... except ValueError: the conversions of directive strings
    let converting_error = |name: &str, kind: &str| -> ! {
        exit_error(&format!(
            "Converting directive setupset.copyarg.{name} to {kind} value"
        ))
    };
    let mut voltage = PyVal::Int(pip_get_integer("VoltageInKV", -1).unwrap_or(-1) as i64);
    if voltage.int() < 0 {
        voltage = match get_setupset_value(&copy_arg_list, "copyarg", "voltage") {
            Some(text) => PyVal::Str(text),
            None => PyVal::None,
        };
        if voltage.truthy() {
            voltage = match voltage.text().trim().parse::<i64>() {
                Ok(value) => PyVal::Int(value),
                Err(_) => converting_error("voltage", "integer"),
            };
        }
    }
    if voltage.truthy() && voltage.is_int() && voltage.int() > 0 {
        voltsub = "Voltage";
    }

    let mut spherab = PyVal::Float(pip_get_float("SphericalAberration", -1.).unwrap_or(-1.));
    if spherab.float() < 0. {
        spherab = match get_setupset_value(&copy_arg_list, "copyarg", "Cs") {
            Some(text) => PyVal::Str(text),
            None => PyVal::None,
        };
        if spherab.truthy() {
            spherab = match spherab.text().trim().parse::<f64>() {
                Ok(value) => PyVal::Float(value),
                Err(_) => converting_error("Cs", "float"),
            };
        }
    }
    if spherab.truthy() && !spherab.is_str() && spherab.float() > 0. {
        sphersub = "SphericalAberration";
    }

    let mut defocus = PyVal::Float(pip_get_float("Defocus", 0.).unwrap_or(0.));
    if pip_get_err_no() == 0 {
        defocsub = "ExpectedDefocus";
    } else {
        defocus = match get_setupset_value(&copy_arg_list, "copyarg", "defocus") {
            Some(text) => match text.trim().parse::<f64>() {
                Ok(value) => {
                    defocsub = "ExpectedDefocus";
                    PyVal::Float(value)
                }
                Err(_) => converting_error("defocus", "float"),
            },
            None => PyVal::None,
        };
    }

    let mut half_float_opt = pip_get_integer("HalfFloatModeOutput", 0).unwrap_or(0);
    if pip_get_err_no() != 0 {
        if let Some(half_str) = get_setupset_value(&copy_arg_list, "copyarg", "halffloat") {
            half_float_opt = match half_str.trim().parse::<i32>() {
                Ok(value) => value,
                Err(_) => converting_error("halffloat", "integer"),
            };
        }
    }

    // `reversed` is only tested for truth: an int from the option, or a
    // directive's string (true when not empty) or None
    let mut reversed = pip_get_boolean("ReversedBidirectional", 0).unwrap_or(0) != 0;
    if pip_get_err_no() != 0 {
        reversed = get_setupset_value(&copy_arg_list, "copyarg", "reversed")
            .is_some_and(|value| !value.is_empty());
    }

    let mut breversed = pip_get_boolean("BReversedBidirectional", 0).unwrap_or(0) != 0;
    if pip_get_err_no() != 0 {
        breversed = get_setupset_value(&copy_arg_list, "copyarg", "breversed")
            .is_some_and(|value| !value.is_empty());
    }

    // This is a full path to be used in sed commands: convert \ to / and escape /
    let mut ctfconf = match pip_get_string("NoiseConfigFile", "") {
        Ok(value) if !value.is_empty() => PyVal::Str(value),
        _ => match get_setupset_value(&copy_arg_list, "copyarg", "ctfnoise") {
            Some(value) => PyVal::Str(value),
            None => PyVal::None,
        },
    };
    if ctfconf.truthy() {
        let mut text = ctfconf.text().replace('\\', "/");
        text = text.replace('/', "\\/");
        ctfconf = PyVal::Str(text);
        ctfconfsub = "ConfigFile";
    }

    let ctf_only = pip_get_integer("CopyCTFfiles", 0).unwrap_or(0);
    if ctf_only != 0 && (ctf_only < 0 || ctf_only > 3) {
        exit_error("The entry for -CTFonly must be 1, 2, or 3");
    }

    // str() / '{}'.format() of a dynamically typed value
    let py_str = |value: &PyVal| -> String {
        match value {
            PyVal::None => "None".to_owned(),
            PyVal::Int(value) => value.to_string(),
            PyVal::Float(value) => py_str_float(*value),
            PyVal::Str(value) => value.clone(),
        }
    };

    // SKIP erase and logbase

    // figure out stack extension if not entered and make sure things are legal for old style
    let mut num_found = 0;
    if name_style <= 0 {
        if !stack_ext.is_empty() && stack_ext != "st" {
            exit_error(
                &("The raw stack can have only extension \".st\" because the old ".to_owned()
                    + "naming style is being used"),
            );
        }
        stack_ext = "st".to_owned();
    } else if stack_ext.is_empty() {
        for ind in 0..possible_stack_exts.len() {
            let ext = &possible_stack_exts[ind];
            if (naxis == 1 && Path::new(&format!("{rootname}.{ext}")).exists())
                || (naxis == 2
                    && (Path::new(&format!("{rootname}a.{ext}")).exists()
                        || Path::new(&format!("{rootname}b.{ext}")).exists()))
            {
                num_found += 1;
                stack_ext = ext.clone();
            }
        }

        if num_found == 0 {
            exit_error(
                &("You must enter -stackext because the new naming style is being used "
                    .to_owned()
                    + "but no raw stack exists with the allowed extensions"),
            );
        }
        if num_found > 1 {
            exit_error(
                &("You must enter -stackext because the new naming style is being used "
                    .to_owned()
                    + "and there are possible raw stacks with several of the allowed extensions"),
            );
        }
    }

    // Set some names based on axis (`piecename` too, which the loop sets again
    // before any use)
    if naxis == 1 {
        g.stackname = format!("{rootname}.{stack_ext}");
    } else {
        g.stackname = format!("{rootname}a.{stack_ext}");
    }

    // On Windows, make sure the current directory is rxw or at least writeable
    if make_current_dir_writable(".").is_some_and(|message| !message.is_empty()) {
        exit_error("Cannot make the current directory writable or write files to it");
    }

    // Copy a distortion file to the current directory if it is elsewhere
    if !distort.is_empty() {
        distort = distort.replace('\\', "/");
        // `os.path.split`
        let split = distort.rfind('/').map_or(0, |index| index + 1);
        let mut dist_dir = distort[..split].to_owned();
        if !dist_dir.is_empty() && dist_dir.chars().any(|character| character != '/') {
            dist_dir = dist_dir.trim_end_matches('/').to_owned();
        }
        let new_distort = distort[split..].to_owned();
        if ctf_only == 0 && !dist_dir.is_empty() && !Path::new(&new_distort).exists() {
            // `shutil.copyfile`: the contents only
            if std::fs::read(&distort)
                .and_then(|contents| std::fs::write(&new_distort, contents))
                .is_err()
            {
                exit_error(&format!(
                    "Copying distortion file {distort} to current directory"
                ));
            }
        }

        // Then set the parameters
        distort = new_distort;
        subdistort = "DistortionField".to_owned();
    }

    // Make the origcoms and dfltcoms directories if needed
    for (com_dir_name, com_descr) in [("origcoms", "original"), ("dfltcoms", "default")] {
        let mut com_dir = com_dir_name.to_owned();
        if Path::new(&com_dir).exists() {
            if !Path::new(&com_dir).is_dir() {
                warning(&format!(
                    "Cannot make directory for {com_descr} com files; a regular file named {com_dir} already exists"
                ));
                com_dir = String::new();
            }
            // `os.access(comDir, os.W_OK)`: the POSIX `access` call itself
            let writable = std::ffi::CString::new(com_dir.as_bytes())
                .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
                .unwrap_or(false);
            if !writable {
                if make_current_dir_writable(&com_dir).is_some() {
                    warning(&format!(
                        "Cannot write to the directory for {com_descr} com files, {com_dir}"
                    ));
                    com_dir = String::new();
                }
            }
        } else {
            match std::fs::create_dir(&com_dir) {
                Ok(()) => {
                    if let Some(err_str) =
                        make_current_dir_writable(&com_dir).filter(|message| !message.is_empty())
                    {
                        warning(&err_str);
                        com_dir = String::new();
                    }
                }
                Err(_) => {
                    warning(&format!("Error making directory for {com_descr} com files"));
                    com_dir = String::new();
                }
            }
        }

        if com_dir.is_empty() && com_descr == "original" {
            g.orig_com_dir = String::new();
        }
        if com_dir.is_empty() && com_descr == "default" {
            g.dflt_com_dir = String::new();
        }
    }

    // SKIP ALLOWING STACKS IN BACKUP DIRECTORY AND MAKING LINKS TO THEM

    // Make sure X and Y size was entered if stack does not exist
    if !Path::new(&g.stackname).exists() && (xsize <= 0 || ysize <= 0) {
        exit_error("Stack is not present; you must enter -xsize and -ysize");
    }

    // Bead size in pixels, and radius for centroid
    let beadsize = beadnm / pixsize;
    let beadrad = 0.5 * (beadsize + 3.);
    let mut eraserad = if beadrad < 3. {
        1.1
    } else if beadrad < 5. {
        2.1
    } else if beadrad < 7. {
        3.0
    } else {
        4.2
    };

    // 7/18/13: Make the boxes stay 3.3x the bead size; max this with the older formula
    // 9/29/19: Constrain to multiples of 2 instead of 8 and adjust formulas down by 5-7
    // to go for smaller sizes so that the ratio of box to bead size is more comparable
    // for big and medium sized beads
    // `max()` keeps the first of equal values
    let mut box_real = 3.3 * beadsize + 2.;
    if 2. * beadsize + 20. > box_real {
        box_real = 2. * beadsize + 20.;
    }
    if 32. > box_real {
        box_real = 32.;
    }
    let boxsize = 512.min(2 * (box_real / 2.) as i32);

    // See if tracking parameters need to be scaled for bead size
    let mut track_scale_seds: Vec<String> = Vec::new();
    let mut track_lines = read_text_file(
        &path_join(&g.srcdir, &format!("track{}", g.srcext)),
        None,
        false,
        None,
    )
    .unwrap_or_default();
    if !g.change_list.is_empty() {
        track_lines =
            match modify_for_change_list(&track_lines, "track", &g.setlet, &g.change_list, false) {
                Ok(lines) => lines,
                Err(message) => exit_error(&message),
            };
    }

    let min_diam_for_scaling = match option_value(
        &track_lines,
        "MinDiamForParamScaling",
        2,
        false,
        1,
        None,
        None,
    ) {
        Some(OptionValue::Floats(values)) => Some(values[0]),
        _ => None,
    };
    if let Some(min_diam) = min_diam_for_scaling
        && beadsize > min_diam
    {
        // Do the scaling
        let scaling = beadsize / min_diam;
        let dca_opt = "DeletionCriterionMinAndSD";
        for option in [
            "DistanceRescueCriterion",
            "PostFitRescueResidual",
            "MaxRescueDistance",
        ] {
            if let Some(OptionValue::Floats(crit)) =
                option_value(&track_lines, option, 2, false, 1, None, None)
            {
                track_scale_seds.push(format!("/^{option}/s/[ \t].*/\t{:.2}/", scaling * crit[0]));
            }
        }

        if let Some(OptionValue::Floats(del_crit_arr)) =
            option_value(&track_lines, dca_opt, 2, false, 0, None, None)
            && del_crit_arr.len() > 1
        {
            track_scale_seds.push(format!(
                "/^{dca_opt}/s/[ \t].*/\t{:.3},{:.2}/",
                scaling * del_crit_arr[0],
                del_crit_arr[1]
            ));
        }
    }

    // If 0 radius entered, just set eraser param generously large
    if beadnm == 0. {
        eraserad = 4.2;
    }

    // Values set in the loop that the script reads on later passes or after it
    let mut piece_xsize: Option<i32> = None;
    let mut piece_ysize: Option<i32> = None;
    let mut pixelx: Option<f64> = None;
    let mut is_fei = false;
    let name_error = |name: &str| -> ! {
        eprintln!("Traceback (most recent call last):");
        eprintln!("NameError: name '{name}' is not defined");
        std::process::exit(1)
    };

    // LOOP ON THE AXES
    for iaxis in 0..naxis {
        let mut setname = rootname.clone();
        g.dstext = out_com_ext.clone();
        let mut modext = ".mod".to_owned();
        let mut logext = ".log".to_owned();
        let mut recext = ".rec".to_owned();
        let mut subgradient = "gibberish".to_owned();
        let mut sep_group_text = String::new();
        let mut mag_view_text = String::new();
        let mut half_floats = half_float_opt > 1;
        if naxis > 1 {
            g.setlet = "a".to_owned();
            if iaxis != 0 {
                g.setlet = "b".to_owned();
            }
            setname = format!("{rootname}{}", g.setlet);
            g.dstext = format!("{}{out_com_ext}", g.setlet);
            modext = format!("{}.mod", g.setlet);
            logext = format!("{}.log", g.setlet);
            recext = format!("{}.rec", g.setlet);
            if iaxis != 0 {
                std::mem::swap(&mut xsize, &mut ysize);
                axisangle = baxisangle;
                excludelistin = bexcludelistin.clone();
                ifextract = bifextract;
                // Fixed in translation (BUGS.md): `copytomocoms:552` assigns
                // `bangles = angles`, the wrong way round, so native gives the
                // B axis the A axis's -angles list; this uses -bangles.
                angles = bangles.clone();
                g.first = bfirst;
                increm = bincrem;
                userawtlt = buserawtlt;
                firstinc = bfirstinc;
                two_dir = btwo_dir;
                two_dir_angle = btwo_dir_angle;
                dose_sym = bdose_sym;
                dose_sym_angle = bdose_sym_angle;
                reversed = breversed;
                focus_adj = b_focus_adj;
            }
        }

        let dstname = setname.clone();
        g.stackname = format!("{setname}.{stack_ext}");
        let piecename = format!("{setname}.pl");
        g.src_to_dest = format!("s/{}/{dstname}/g", g.srcname);
        set_root_and_extension(&setname, &file_type_ext);

        // Fix axis angle to be within expected range
        while axisangle < -180. {
            axisangle += 360.;
        }
        while axisangle > 180. {
            axisangle -= 360.;
        }

        // Look for expansion factor in newstack changes
        let mut expand_fac = String::new();
        for change in &g.change_list {
            if (change[0] == "newst" || change[0] == format!("newst{}", g.setlet))
                && change[1] == "newstack"
                && change[2] == "ExpandByFactor"
            {
                expand_fac = change[3].clone();
                break;
            }
        }

        // GET X AND Y SIZE FROM STACK IF IT EXISTS
        let stackname = g.stackname.clone();
        let stack_exists = Path::new(&stackname).exists();
        let mut raw_zsize = -1;
        let mut mode = 0;
        let mut mindens = 0.;
        let mut meandens = 0.;
        if stack_exists {
            // Determine if file is from FEI and try to fix the pixel before getting it
            // Etomo is looking for "Pixel spacing" in INFO lines
            if set_fei_pixel != 0 {
                let (fei, mdoc_pixel) = is_file_from_fei(progname, &stackname);
                is_fei = fei;
                if is_fei || mdoc_pixel != 0. {
                    let alter_lines = if is_fei {
                        vec!["FEIPIXEL".to_owned(), "-1".to_owned(), "done".to_owned()]
                    } else {
                        let pixel = py_str_float(mdoc_pixel);
                        vec![
                            "DEL".to_owned(),
                            format!("{pixel},{pixel},{pixel}"),
                            "done".to_owned(),
                        ]
                    };
                    let alter_out = match run_cmd(
                        &format!("alterheader \"{stackname}\""),
                        Some(&alter_lines),
                        None,
                        None,
                        &[],
                    ) {
                        Ok(lines) => lines.unwrap_or_default(),
                        Err(_) => exit_from_imod_error(progname),
                    };
                    let lastind = alter_out.len().saturating_sub(1);
                    let line_at = |index: usize| alter_out.get(index).cloned().unwrap_or_default();
                    if line_at(lastind).contains("header") {
                        let mut warnout = if is_fei {
                            format!(
                                "Pixel spacing of FEI file {stackname} was not set.  Output from alterheader:\n"
                            )
                        } else {
                            format!(
                                "Pixel spacing of file {stackname} was not set from value in mdoc.  Output from alterheader:\n"
                            )
                        };
                        if line_at(lastind).contains("not being") {
                            warnout +=
                                &format!("{}\n", line_at(lastind.saturating_sub(1)).trim_end());
                        }
                        warnout += line_at(lastind).trim_end();
                        if line_at(lastind.saturating_sub(1))
                            .contains("has already been transferred")
                        {
                            print_info(&warnout);
                        } else {
                            warning(&warnout);
                        }
                    } else {
                        // Batchruntomo looks for 'Pixel spacing was set in FEI file'
                        if is_fei {
                            print_info(&format!(
                                "Pixel spacing was set in FEI file {stackname} from value in extended header"
                            ));
                        } else {
                            print_info(&format!(
                                "Pixel spacing was set in file {stackname} from value in mdoc file"
                            ));
                        }
                    }
                }
            }

            // Regardless of file type, run this full header once here
            xsize = -1;
            let header_result: Result<(), ImodpyError> = (|| {
                // `getmrc` values are the text `header` printed, read as
                // Python floats (doubles)
                let as_double = |value: f64| -> f64 { value };
                if let MrcInfo::All(x, y, z, m, px, _py, _pz, _ox, _oy, _oz, dmin, _dmax, dmean) =
                    get_mrc(&stackname, true, false)?
                {
                    xsize = x;
                    ysize = y;
                    g.zsize = PyVal::Int(z as i64);
                    mode = m;
                    pixelx = Some(as_double(px));
                    mindens = as_double(dmin);
                    meandens = as_double(dmean);
                }
                raw_zsize = g.zsize.int() as i32;
                if (mode == 2 || mode == 12) && half_float_opt == 1 {
                    half_floats = true;
                }
                if half_floats && name_style == 2 {
                    exit_error("Half-float output is not supported in HDF files");
                }
                if mode == 16 {
                    exit_error(
                        &("Color images (mode 16) cannot be processed; use Newstack to "
                            .to_owned()
                            + "convert to another mode"),
                    );
                }
                if montage != 0 {
                    piece_xsize = Some(xsize);
                    piece_ysize = Some(ysize);
                    let mut montcmd = format!("montagesize {stackname}");
                    if Path::new(&piecename).exists() {
                        montcmd += &format!(" {piecename}");
                    }
                    let montlines = run_cmd(&montcmd, None, None, None, &[])?.unwrap_or_default();
                    for line in &montlines {
                        if line.contains("Total") {
                            if let Some(ind) = line.find(':').filter(|ind| *ind > 0) {
                                let lsplit = line[ind + 1..].split_whitespace().collect::<Vec<_>>();
                                if lsplit.len() > 1 {
                                    let to_int = |text: &str| -> i32 {
                                        match text.parse::<i32>() {
                                            Ok(value) => value,
                                            Err(_) => {
                                                eprintln!("Traceback (most recent call last):");
                                                eprintln!(
                                                    "ValueError: invalid literal for int() with base 10: '{text}'"
                                                );
                                                std::process::exit(1)
                                            }
                                        }
                                    };
                                    xsize = to_int(lsplit[0]);
                                    ysize = to_int(lsplit[1]);
                                    if lsplit.len() > 2 {
                                        // Fixed in translation (BUGS.md): native
                                        // stores the string (`copytomocoms:657`) and
                                        // `anglesPassThrough`'s `range(1, zsize)` then
                                        // raises TypeError; this stores the integer.
                                        g.zsize = match lsplit[2].parse::<i64>() {
                                            Ok(value) => PyVal::Int(value),
                                            Err(_) => PyVal::Str(lsplit[2].to_owned()),
                                        };
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                Ok(())
            })();
            let _ = header_result;
            if xsize <= 0 {
                exit_error(&format!(
                    "Extracting image size from header of stack {stackname}"
                ));
            }

            // Check for inverted increment
            if firstinc != 0 {
                let (input_pass0, input_pass90, input_pass_min90) = angles_pass_through(&g, increm);
                let (inv_pass0, inv_pass90, inv_pass_min90) = angles_pass_through(&g, -increm);
                if inv_pass0
                    && !inv_pass90
                    && !inv_pass_min90
                    && ((input_pass90 && !input_pass0 && !input_pass_min90)
                        || (input_pass_min90 && !input_pass0 && !input_pass90))
                {
                    let mut mess = "Tilt angles pass through ".to_owned();
                    if input_pass90 {
                        mess += "90 degrees ";
                    } else {
                        mess += "-90 degrees ";
                    }
                    if naxis > 1 {
                        mess += &format!("for the {} axis ", g.setlet.to_uppercase());
                    }
                    warning(&format!(
                        "{mess}but not through 0 degrees; does the starting angle ({}) or the increment ({}) have the wrong sign?",
                        py_str_float(g.first),
                        py_str_float(increm)
                    ));
                }
            }
        } else if iaxis != 0 {
            // Etomo is looking for "dual axis set without B stack"
            print_info(&format!(
                "Setting up dual axis set without B stack {stackname}\n   assuming it will be added later"
            ));
            let mut infol =
                format!("{stackname} does not exist yet, assuming size {xsize} x {ysize}\n")
                    + "   If the size is different, you will need to fix FULLIMAGE in "
                    + "tiltb.com\n   in order to use parallel processing";
            if montage != 0 {
                infol += &format!(
                    "\n   Run   \"montagesize {stackname}\"    to get the raw image size (NX, NY, and NZ)\n"
                );
                infol +=
                    "   Then run   \"goodframe NX NY\"   to get the numbers to put in tiltb.com";
            }
            print_info(&infol);
        }

        // GET TILT ANGLE AND AXIS INFORMATION AND FULL OUTPUT SIZE
        let mut newstwidth = String::new();
        let mut fullx = xsize;
        let mut fully = ysize;
        if montage != 0 {
            (fullx, fully) = run_goodframe(xsize, ysize);
            if fullx < 0 {
                exit_error("Running goodframe to get full aligned stack size");
            }
        }

        // Also get the rotation relative to square to get indent in Y from rotation
        let mut indentangle;
        if (axisangle > 45. && axisangle < 135.) || (axisangle < -45. && axisangle > -135.) {
            std::mem::swap(&mut fullx, &mut fully);
            newstwidth = fullx.to_string();
            if axisangle > 0. {
                indentangle = 90. - axisangle;
            } else {
                indentangle = -90. - axisangle;
            }
        } else if axisangle > 90. {
            indentangle = 180. - axisangle;
        } else {
            indentangle = -180. - axisangle;
        }

        if indentangle < 0. {
            indentangle = -indentangle;
        }
        let blxstart = xsize.div_euclid(2) - fullx.div_euclid(2);
        let blxend = blxstart + fullx - 1;
        let blystart = ysize.div_euclid(2) - fully.div_euclid(2);
        let blyend = blystart + fully - 1;

        // EXTRACT A RAWTLT FILE
        let mut tiltspec = format!("{setname}.rawtlt");
        let mut no_tilts = !Path::new(&tiltspec).exists();
        if no_tilts {
            if ifextract != 0 {
                if stack_exists {
                    let extractlines = match run_cmd(
                        &format!("extracttilts -warn \"{stackname}\" \"{tiltspec}\""),
                        None,
                        None,
                        None,
                        &[],
                    ) {
                        Ok(lines) => lines.unwrap_or_default(),
                        Err(_) => exit_from_imod_error(progname),
                    };
                    if let Some(last) = extractlines.last() {
                        prnstr(last, "\n", false);
                    }
                    no_tilts = false;
                } else {
                    print_info(&format!(
                        "{stackname} does not exist yet.\n  When it does, you will have to run:\n  extracttilts {stackname} {tiltspec}"
                    ));
                    later_blines.push(format!("$extracttilts {stackname} {tiltspec}"));
                }
            } else if userawtlt != 0 {
                warning(&format!(
                    "{tiltspec} does not exist yet.  You need to create this file before running commands."
                ));
            }
        }

        // Check the tilt file for all the same angle or small range
        if !no_tilts && raw_zsize > 2 {
            prnstr(" ", "\n", false);
            let angle_list = read_text_file(&tiltspec, None, false, None).unwrap_or_default();
            let mut min_angle: f64 = 999.;
            let mut max_angle: f64 = -999.;
            for ind in 0..angle_list.len() {
                if angle_list[ind].trim().is_empty() {
                    continue;
                }
                match angle_list[ind].trim().parse::<f64>() {
                    Ok(f_angle) => {
                        // `min()`/`max()` keep the first of equal values
                        if f_angle < min_angle {
                            min_angle = f_angle;
                        }
                        if f_angle > max_angle {
                            max_angle = f_angle;
                        }
                    }
                    Err(_) => exit_error(&format!(
                        "Converting {} to float from {tiltspec}",
                        angle_list[ind]
                    )),
                }
            }

            if min_angle > max_angle {
                exit_error(&format!("There are no tilt angles in {tiltspec}"));
            }
            if max_angle - min_angle < 0.001 {
                exit_error(&format!(
                    "The tilt angles in {tiltspec} are all {max_angle:.2}"
                ));
            }
            if max_angle - min_angle < 1.001 {
                exit_error(&format!(
                    "The tilt angles in {tiltspec} only range from {min_angle:.2} to {max_angle:.2}"
                ));
            }
        }

        // Set up a SeparateGroup if twodir angle was entered
        let mut num_bidir_views = 0;
        if two_dir != 0 {
            let mut angle_list: Vec<String> = Vec::new();
            if !angles.is_empty() {
                angle_list = angles
                    .replace(',', " ")
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect();
            } else if firstinc != 0 {
                let num_fake = (2. * g.first / increm).abs().round_ties_even() as i64;
                for ind in 0..num_fake {
                    angle_list.push(py_str_float(g.first + ind as f64 * increm));
                }
            } else if no_tilts {
                warning(&format!(
                    "{tiltspec} does not exist yet so SeparateGroup entries could not be defined for track{}.com and align{}.com",
                    g.setlet, g.setlet
                ));
            } else {
                angle_list = read_text_file(&tiltspec, None, false, None).unwrap_or_default();
            }

            if !angle_list.is_empty() {
                let mut num_interval = 0;
                let mut interval_sum = 0.;
                let mut min_diff = 10000.;
                let mut f_angles: Vec<f64> = Vec::new();
                let mut ind_min: i64 = 0;

                // Convert the numbers and find the one closest to defined angle and the mean
                // tilt interval in the neighborhood
                for ind in 0..angle_list.len() {
                    let f_angle = match angle_list[ind].trim().parse::<f64>() {
                        Ok(value) => value,
                        Err(_) => exit_error(&format!(
                            "Converting {} to float from {tiltspec}",
                            angle_list[ind]
                        )),
                    };

                    f_angles.push(f_angle);
                    let diff = (f_angle - two_dir_angle).abs();
                    if diff < 10. && ind != 0 {
                        interval_sum += (f_angle - f_angles[ind - 1]).abs();
                        num_interval += 1;
                    }
                    if diff < min_diff {
                        min_diff = diff;
                        ind_min = ind as i64;
                    }
                }

                // If the closest one is within 1/4 of the tilt interval, look again to make
                // sure we have the first one within that range; otherwise just take closest
                let mean_interval = interval_sum / 1.max(num_interval) as f64;
                if num_interval != 0 && min_diff <= 0.25 * mean_interval {
                    for ind in 0..f_angles.len() {
                        if (f_angles[ind] - two_dir_angle).abs() <= 0.25 * mean_interval {
                            ind_min = ind as i64;
                            break;
                        }
                    }
                }

                if ind_min == 0
                    && angle_list.len() > 1
                    && min_diff > 0.75 * (f_angles[1] - f_angles[0]).abs()
                {
                    exit_error(&format!(
                        "The bidirectional starting angle ({}) is past the start of the tilt series, which starts at {}",
                        py_str_float(two_dir_angle),
                        py_str_float(f_angles[ind_min as usize])
                    ));
                }

                if reversed {
                    ind_min -= 1;
                    if ind_min < 0 {
                        exit_error(&format!(
                            "The bidirectional starting angle ({}) is at the start of the tilt series; for a reversed series this means it is no longer bidirectional",
                            py_str_float(two_dir_angle)
                        ));
                    }
                }

                if ind_min == angle_list.len() as i64 - 1 {
                    exit_error(&format!(
                        "The bidirectional starting angle ({}) is at or past the end of the tilt series, which ends at {}",
                        py_str_float(two_dir_angle),
                        py_str_float(f_angles[ind_min as usize])
                    ));
                }

                sep_group_text = format!("/^RotationAngle/a/SeparateGroup\t1-{}/", ind_min + 1);
                mag_view_text = format!("/^RotationAngle/a/ViewsWithMagChanges\t{}/", ind_min + 2);
                num_bidir_views = ind_min + 1;
            }
        }

        // IF MONTAGED CCD, JUST EXTRACT THE PIECE LIST IF NEEDED
        if montage != 0 && !Path::new(&piecename).exists() {
            if stack_exists {
                prnstr(
                    "Extracting piece list file from the image file...",
                    "\n",
                    false,
                );
                let extractlines = match run_cmd(
                    &format!("extractpieces \"{stackname}\" \"{piecename}\""),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => exit_from_imod_error(progname),
                };
                if let Some(last) = extractlines.last() {
                    prnstr(last, "\n", false);
                }
            } else {
                print_info(&format!(
                    "{stackname} does not exist yet.\n  When it does, you will have to run:\n  extractpieces {stackname} {piecename}"
                ));
                later_blines.push(format!("$extractpieces {stackname} {piecename}"));
            }
        }

        // Set up mag gradient file
        if !gradient.is_empty() {
            let gradspec = format!("{setname}.maggrad");
            let mut gradok = 1;
            if !Path::new(&gradspec).exists() {
                // Use standard input in case of spaces in gradient table
                let mut input = vec![
                    format!("InputImageFile {stackname}"),
                    format!("OutputFile {gradspec}"),
                    format!("GradientTable {gradient}"),
                    format!("RotationAngle {}", py_str_float(axisangle)),
                ];
                // Fixed in translation (BUGS.md): with the stack absent native
                // reads the never-assigned `pixelx` (`copytomocoms:879`,
                // NameError); an unknown header pixel size is treated like the
                // unset one (1.0), so the entered -pixel is passed on.
                let pixelx_value = pixelx.unwrap_or(1.);
                if pixelx_value == 1. {
                    input.push(format!("PixelSize {}", py_str_float(pixsize)));
                }

                if stack_exists {
                    prnstr(
                        "Extracting mag gradients from the image file...",
                        "\n",
                        false,
                    );
                    let mut extractlines: Vec<String> = Vec::new();
                    match run_cmd(
                        "extractmagrad -StandardInput",
                        Some(&input),
                        None,
                        None,
                        &[],
                    ) {
                        Ok(lines) => extractlines = lines.unwrap_or_default(),
                        Err(_) => {
                            warning(&format!("Could not extract mag gradients from {stackname}"));
                            gradok = 0;
                        }
                    }

                    if extractlines.len() > 1 {
                        prnstr(&extractlines[extractlines.len() - 2], "\n", false);
                        prnstr(&extractlines[extractlines.len() - 1], "\n", false);
                    }
                } else {
                    print_info(&format!(
                        "{stackname} does not exist yet.\n  When it does, you will have to run:\n  extractmagrad {stackname} -rot {} -grad {gradient} {stackname} {gradspec}",
                        py_str_float(axisangle)
                    ));
                    later_blines.push("$extractmagrad -StandardInput".to_owned());
                    later_blines.extend(input);
                }
            }

            if gradok != 0 {
                subgradient = "GradientFile".to_owned();
            }
        }

        // Set up tilt angle variables
        let tiltopt;
        if !angles.is_empty() {
            tiltopt = -1;
            tiltspec = angles.clone();
        } else if firstinc != 0 {
            tiltopt = 1;
            tiltspec = format!("{},{}", py_str_float(g.first), py_str_float(increm));
        } else {
            tiltopt = 0;
            tiltspec = format!("{setname}.rawtlt");
        }

        // Handle text for PIP-style tilt options
        let mut inc_del_text = "TiltIncrement";
        let mut tilt_inc_text = "gibberish232";
        let mut tilt_start = tiltspec.clone();
        let mut tilt_inc = "1".to_owned();
        let tilt_type_text;
        if tiltopt < 0 {
            tilt_type_text = "TiltAngles";
        } else if tiltopt == 0 {
            tilt_type_text = "TiltFile";
        } else {
            tilt_type_text = "FirstTiltAngle";
            tilt_start = py_str_float(g.first);
            tilt_inc = py_str_float(increm);
            tilt_inc_text = "TiltIncrement";
            inc_del_text = "gibberish434";
        }

        let excludelist = excludelistin.replace(' ', "");

        let mut wipeexclude = "ExcludeList";
        let mut wipeskip = "SkipViews";
        if !excludelist.is_empty() {
            wipeexclude = "gibberish";
            wipeskip = "gibberish";
        }

        g.backuplist.extend([
            piecename.clone(),
            format!("{setname}.rawtlt"),
            format!("track{logext}"),
            format!("align{logext}"),
            format!("findsec{logext}"),
            format!("tomopitch{logext}"),
        ]);

        // Adjust mode in tilt output if float input, in newst output if byte
        // Also adjust base for log if maximum denisty is negative and minimum is
        // very negative and mode is 1
        let mut tiltmode = 1;
        let mut newstmode = String::new();
        if stack_exists {
            if mode == 2 {
                tiltmode = 2;
            }
            if half_floats {
                tiltmode = 12;
                newstmode = "ModeToOutput 12".to_owned();
            } else if mode == 0 {
                newstmode = "ModeToOutput 1".to_owned();
            }
            if mode == 1 {
                // Etomo sets metadata from these two INFO lines simply by looking for 32768
                // And the INFO is recognized with 'Setting logarithm offset'
                if meandens < -1000. && mindens < -10000. {
                    logbase = 32768;
                    print_info(&format!(
                        "Setting logarithm offset for Tilt to 32768 because the mean density of\n   {stackname} is negative and the minimum density is < -10000"
                    ));
                } else {
                    if set_fei_pixel == 0 {
                        (is_fei, _) = is_file_from_fei(progname, &stackname);
                    }

                    if is_fei {
                        if mindens < 0. {
                            logbase = 32768;
                            print_info(&format!(
                                "Setting logarithm offset for Tilt to 32768 because {stackname}\n   came from FEI software and has a minimum < 0"
                            ));
                        } else {
                            warning(&format!(
                                "You may need a logarithm offset of 32768 for Tilt because {stackname}\n   came from FEI software, unless you already shifted the data up by 32768"
                            ));
                        }
                    }

                    if !is_fei && meandens < 0. {
                        warning(&format!(
                            "{stackname} has a negative mean density ({meandens:.1}) and you may need\n   a logarithm offset when running Tilt"
                        ));
                    }
                }
            }
        }

        let mut wipepl = "gibberish9876543".to_owned();

        // GET CTFPLOTTER AND CTFCORRECTION first, so that we can skip rest of loop
        if ctf_only != 2 {
            let mut sedcom = sed_for_image_files(&g, &[]);
            sedcom.extend([
                format!("/^{ctfconfsub}/s/[ \t].*/\t{}/", py_str(&ctfconf)),
                format!("/^{voltsub}/s/[ \t].*/\t{}/", py_str(&voltage)),
                format!("/^{sphersub}/s/[ \t].*/\t{}/", py_str(&spherab)),
                format!("/^{defocsub}/s/[ \t].*/\t{}/", py_str(&defocus)),
                format!("/^PixelSize/s/[ \t].*/\t{}/", py_str_float(pixsize)),
                format!("/^AxisAngle/s/[ \t].*/\t{}/", py_str_float(axisangle)),
            ]);
            if !ctfconf.truthy() {
                sedcom.push("/ConfigFile/d".to_owned());
            }
            if two_dir != 0 && num_bidir_views != 0 {
                sedcom.push(format!(
                    "/^DefocusFile/a/BidirectionalNumViews\t{num_bidir_views}/"
                ));
            }
            if !excludelist.is_empty() {
                sedcom.push(format!("/^DefocusFile/a/ViewsToSkip\t{excludelist}/"));
            }

            edit_and_write(&mut g, "ctfplotter", "ctfplotter", sedcom, true, None);
        }

        if ctf_only != 1 {
            let mut sedcom = sed_for_image_files(&g, &[".ali", "_ctfcorr.ali"]);
            sedcom.extend([
                format!("/^{voltsub}/s/[ \t].*/\t{}/", py_str(&voltage)),
                format!("/^{sphersub}/s/[ \t].*/\t{}/", py_str(&spherab)),
                format!("/^UnbinnedPixelSize/s/[ \t].*/\t{}/", py_str_float(pixsize)),
                format!("/^PixelSize/s/[ \t].*/\t{}/", py_str_float(pixsize)),
            ]);
            if !expand_fac.is_empty() {
                sedcom.extend(sed_del_and_add(
                    "ExpandedByFactor",
                    &expand_fac,
                    "PixelSize",
                    '/',
                ));
            }
            if half_floats {
                sedcom.extend(sed_del_and_add(
                    "$setenv",
                    "IMOD_WRITE_FLOATS_16BIT 1",
                    "####CreatedVersion",
                    '/',
                ));
            }
            edit_and_write(&mut g, "ctfcorrection", "ctfcorrection", sedcom, true, None);
        }

        if ctf_only != 0 {
            continue;
        }

        wipepl = piecename.clone();
        let tiltxf = format!("{setname}.tltxf");
        let tracksrc = dataset_filename(".preali", None, None);

        // GET ERASER - a special case because name must be preserved as well as file format
        // set from that
        let sedcom = vec![
            format!("s/{}.st/{stackname}/g", g.srcname),
            format!("s/{}_fixed.st/{setname}_fixed.{stack_ext}/g", g.srcname),
            g.src_to_dest.clone(),
            format!(
                "s/^MaximumRadius.*/MaximumRadius   {}/",
                py_str_float(eraserad)
            ),
        ];
        let mut stack_format: Option<&str> = None;
        if let Some(format) = formats_for_ext(&stack_ext.to_uppercase())
            && name_style > 0
        {
            stack_format = Some(format);
        }
        let user_format = std::env::var("IMOD_OUTPUT_FORMAT").unwrap_or_default();
        if stack_format.is_none()
            && !user_format.is_empty()
            && (user_format == "JPG" || user_format == "JPEG")
        {
            stack_format = Some("MRC");
        }
        edit_and_write(&mut g, "eraser", "eraser", sedcom, true, stack_format);

        // GET XCORR
        let dstfile = format!("xcorr{}", g.dstext);
        g.backuplist.extend([
            dstfile.clone(),
            format!("{setname}.prexf"),
            format!("{setname}.prexg"),
            format!("{setname}.tltxf"),
        ]);
        make_backup_file(&dstfile);

        let mut dstlines = vec![
            "# THIS IS A COMMAND FILE TO RUN TILTXCORR AND DETERMINE CROSS-CORRELATION".to_owned(),
            "# ALIGNMENT OF A TILT SERIES".to_owned(),
            "#".to_owned(),
        ];

        let mut xcsrc = stackname.clone();
        let mut preblend = 0;
        if montage != 0 {
            // IF MONTAGED, START THE FILE WITH PREBLEND - LIMIT SIZE
            dstlines.extend([
                "# Set the following goto to 'doxcorr' to skip the blend".to_owned(),
                "$goto doblend".to_owned(),
                "$doblend:".to_owned(),
            ]);
            preblend = 1;
            let srcfile = path_join(&g.srcdir, &format!("blend{}", g.srcext));
            let (Some(piece_x), Some(piece_y)) = (piece_xsize, piece_ysize) else {
                name_error("pieceXsize")
            };
            let xcorr_max = piece_x.max(piece_y).max(MONTAGE_XCORR_MAX);
            let xcxsize = xsize.min(xcorr_max);
            let xcysize = ysize.min(xcorr_max);
            let xstart = (xsize - xcxsize).div_euclid(2);
            let xend = (xsize + xcxsize).div_euclid(2) - 1;
            let ystart = (ysize - xcysize).div_euclid(2);
            let yend = (ysize + xcysize).div_euclid(2) - 1;
            xcsrc = dataset_filename(".bl", None, None);
            let mut sedcom = sed_for_image_files(&g, &[".ali"]);
            sedcom.extend([
                format!(
                    "/^ImageOutputFile/s/{}/{xcsrc}/",
                    dataset_filename(".ali", None, None)
                ),
                "/^TransformFile/s/^T/#T/".to_owned(),
                format!("/^#*{subdistort}/s/.*/DistortionField\t{distort}/"),
                format!(
                    "/^ImagesAreBinned/s/1/{}/",
                    c_format("%g", &[CArg::Dbl(binning)])
                ),
                format!("/^#*{subgradient}/s/^#//"),
                format!("/^AdjustedFocus/s/0/{focus_adj}/"),
                "/^SloppyMontage/s/0/1/".to_owned(),
                "/^ShiftPieces/s/0/1/".to_owned(),
                "/AdjustOrigin/d".to_owned(),
                format!("/^StartingAndEndingX/s/[ \t].*/\t{xstart} {xend}/"),
                format!("/^StartingAndEndingY/s/[ \t].*/\t{ystart} {yend}/"),
                "/mrctaper/d".to_owned(),
                format!("/####CreatedVersion####/s/# .*/#{}/", g.imod_version),
            ]);
            if half_floats {
                sedcom.extend(sed_del_and_add(
                    "ModeToOutput",
                    "12",
                    "ImageOutputFile",
                    '/',
                ));
            }
            let sedlines = pysed(&sedcom, PysedSrc::File(&srcfile), None, false, '/', false)
                .ok()
                .flatten()
                .unwrap_or_default();
            dstlines.extend(sedlines);
            dstlines.push("$doxcorr:".to_owned());
        }

        // Now do tiltxcorr commands
        let srcfile = path_join(&g.srcdir, &format!("xcorr{}", g.srcext));
        let mut sedcom = sed_for_image_files(&g, &[]);
        sedcom.extend([
            format!("/^FirstTiltAngle/s/.*/{tilt_type_text}\t{tilt_start}/"),
            format!("/^{tilt_inc_text}/s/[ \t].*/\t{tilt_inc}/"),
            format!("/^{inc_del_text}/d"),
            format!("/^RotationAngle/s/[ \t].*/\t{}/", py_str_float(axisangle)),
            format!("/^#*SkipViews.*/s//SkipViews\t{excludelist}/"),
            format!("/^#*{wipeskip}/d"),
            format!("/^InputFile/s/{stackname}/{xcsrc}/"),
            format!("/####CreatedVersion####/s/# .*/#{}/", g.imod_version),
        ]);
        sedcom.extend(sed_del_and_add(
            "PixelSize",
            &py_str_float(pixsize),
            "OutputFile",
            '/',
        ));

        if !mag_view_text.is_empty() {
            sedcom.push(mag_view_text.clone());
        }
        if dose_sym != 0 {
            sedcom.extend(sed_del_and_add(
                "AngleOffset",
                &py_str_float(-dose_sym_angle),
                "RotationAngle",
                '/',
            ));
        }
        let sedlines = pysed(&sedcom, PysedSrc::File(&srcfile), None, false, '/', false)
            .ok()
            .flatten()
            .unwrap_or_default();
        dstlines.extend(sedlines);
        dstlines.push(g.backupline.clone());
        apply_change_list_and_write(&g, "xcorr", dstlines, &dstfile, true, None);

        // Set up lines for adding distortion corrections
        let dist_seds = vec![
            format!("/^#*{subdistort}/s/.*/DistortionField\t{distort}/"),
            format!(
                "/^ImagesAreBinned/s/1/{}/",
                c_format("%g", &[CArg::Dbl(binning)])
            ),
            format!("/^#*{subgradient}/s/^#//"),
        ];

        if montage == 0 {
            // GET PRENEWST
            let mut sedcom = sed_for_image_files(&g, &[".preali"]);
            sedcom.extend(dist_seds.iter().cloned());
            edit_and_write(&mut g, "prenewst", "prenewst", sedcom, true, None);

            // GET NEWST
            let mut sedcom = sed_for_image_files(&g, &[".ali"]);
            sedcom.extend(dist_seds.iter().cloned());
            sedcom.push(format!(
                "/^SizeToOutputInXandY/s/[ \t].*/\t{newstwidth},{SAMPLESIZE}/"
            ));
            if !newstmode.is_empty() {
                sedcom.push(format!("/^OutputFile/a/{newstmode}/"));
            }
            edit_and_write(&mut g, "ccdnewst", "newst", sedcom, true, None);
        } else {
            // OR, GET PREBLEND AND BLEND
            let mut sedcom = sed_for_image_files(&g, &[".preali"]);
            sedcom.extend(dist_seds.iter().cloned());
            sedcom.push(format!("/^AdjustedFocus/s/0/{focus_adj}/"));
            if half_floats {
                sedcom.extend(sed_del_and_add(
                    "ModeToOutput",
                    "12",
                    "ImageOutputFile",
                    '/',
                ));
            }
            edit_and_write(&mut g, "ccdpreblend", "preblend", sedcom, true, None);

            let mut sedcom = sed_for_image_files(&g, &[".ali"]);
            sedcom.extend(dist_seds.iter().cloned());
            sedcom.extend([
                format!("/^AdjustedFocus/s/0/{focus_adj}/"),
                "/^SloppyMontage/s/0/1/".to_owned(),
                "/^ShiftPieces/s/0/1/".to_owned(),
                "/^ReadInXcorrs/s/0/1/".to_owned(),
                "/^OldEdgeFunctions/s/0/1/".to_owned(),
                format!("/^StartingAndEndingX/s/[ \t].*/\t{blxstart} {blxend}/"),
                format!("/^StartingAndEndingY/s/[ \t].*/\t{blystart} {blyend}/"),
            ]);
            if half_floats {
                sedcom.extend(sed_del_and_add(
                    "ModeToOutput",
                    "12",
                    "ImageOutputFile",
                    '/',
                ));
            }
            edit_and_write(&mut g, "blend", "blend", sedcom, true, None);
            g.backuplist.push(format!("{setname}.ecd"));
        }

        // GET TRACK: WIPE OUT PL ENTRY WHEN NO MONTAGE

        let mut sedcom = sed_for_image_files(&g, &[".preali"]);
        sedcom.extend(track_scale_seds.iter().cloned());
        sedcom.extend([
            format!("/^FirstTiltAngle/s/.*/{tilt_type_text}\t{tilt_start}/"),
            format!("/^{tilt_inc_text}/s/[ \t].*/\t{tilt_inc}/"),
            format!("/^{inc_del_text}/d"),
            format!("/^RotationAngle/s/[ \t].*/\t{}/", py_str_float(axisangle)),
            format!("/^#*SkipViews.*/s//SkipViews\t{excludelist}/"),
            format!("/^#*{wipeskip}/d"),
            format!("/^BeadDiameter/s/[ \t].*/\t{beadsize:.2}/"),
            format!("/^BoxSizeXandY/s/[ \t].*/\t{boxsize},{boxsize}/"),
            format!("s/{wipepl}//"),
        ]);
        sedcom.extend(sed_del_and_add(
            "PixelSize",
            &py_str_float(pixsize),
            "RotationAngle",
            '/',
        ));
        if !sep_group_text.is_empty() {
            sedcom.push(sep_group_text.clone());
        }
        if dose_sym != 0 {
            sedcom.extend(sed_del_and_add(
                "AngleOffset",
                &py_str_float(-dose_sym_angle),
                "RotationAngle",
                '/',
            ));
        }
        edit_and_write(&mut g, "track", "track", sedcom, true, None);
        g.backuplist
            .extend([format!("{setname}seed"), format!("{setname}fid")]);

        // COPY EMPTY SEED MODELS
        let seedhere = format!("{setname}.seed");
        if !Path::new(&seedhere).exists() {
            // `shutil.copyfile`: the contents only
            let source = path_join(&g.srcdir, seedname);
            if let Err(error) =
                std::fs::read(&source).and_then(|contents| std::fs::write(&seedhere, contents))
            {
                eprintln!("Traceback (most recent call last):");
                eprintln!("{error}");
                std::process::exit(1);
            }
        }

        // GET ALIGN; zero FixXYZCoords IF SINGLE AXIS; SET OUTPUT TO .TLTXF FOR CCD

        let mut fixxyz = 1;
        if naxis == 1 {
            fixxyz = 0;
        }

        // set the frame size bigger if needed for a montage set

        let mut alxsize = xsize;
        let mut alysize = ysize;
        if preblend != 0 {
            (alxsize, alysize) = run_goodframe(xsize, ysize);
            if alxsize < 0 {
                exit_error("Running goodframe to get full prealigned stack size");
            }
        }

        // 5/3/02: change to using image file name, keep dimensions commented out

        let srcfile = path_join(&g.srcdir, &format!("align{}", g.srcext));
        let dstfile = format!("align{}", g.dstext);
        make_backup_file(&dstfile);
        let mut sedcom = sed_for_image_files(&g, &[]);
        sedcom.extend([
            format!("/^FirstTiltAngle/s/.*/{tilt_type_text}\t{tilt_start}/"),
            format!("/^{tilt_inc_text}/s/[ \t].*/\t{tilt_inc}/"),
            format!("/^{inc_del_text}/d"),
            format!("/^RotationAngle/s/[ \t].*/\t{}/", py_str_float(axisangle)),
            format!("/ImageSizeXandY/s/[ \t].*/\t{alxsize},{alysize}/"),
            format!("/^#*ExcludeList.*/s//ExcludeList\t{excludelist}/"),
            format!("/^ImageFile/s/[ \t].*/\t{tracksrc}/"),
            format!("/^OutputTransformFile/s/[ \t].*/\t{tiltxf}/"),
            format!("/^FixXYZCoordinates/s/.*/FixXYZCoordinates\t{fixxyz}/"),
            format!("/^#*{wipeexclude}/d"),
            format!("/####CreatedVersion####/s/# .*/#{}/", g.imod_version),
        ]);
        sedcom.extend(sed_del_and_add(
            "UnbinnedPixelSize",
            &py_str_float(pixsize),
            "RotationAngle",
            '/',
        ));
        sedcom.extend(sed_del_and_add(
            "CreatedDayStamp",
            &created_day.to_string(),
            "RotationAngle",
            '/',
        ));

        if !sep_group_text.is_empty() {
            sedcom.push(sep_group_text.clone());
        }
        if dose_sym != 0 {
            sedcom.extend(sed_del_and_add(
                "AngleOffset",
                &py_str_float(-dose_sym_angle),
                "RotationAngle",
                '/',
            ));
        }
        let mut sedlines = pysed(&sedcom, PysedSrc::File(&srcfile), None, false, '/', false)
            .ok()
            .flatten()
            .unwrap_or_default();

        sedlines.extend([
            "#".to_owned(),
            "# COMBINE TILT TRANSFORMS WITH PREALIGNMENT TRANSFORMS".to_owned(),
            "#".to_owned(),
            "$xfproduct -StandardInput".to_owned(),
            format!("InputFile1 {setname}.prexg"),
            format!("InputFile2 {setname}.tltxf"),
            format!("OutputFile {setname}_fid.xf"),
            format!("$b3dcopy -p \"{setname}_fid.xf\" \"{setname}.xf\""),
            format!("$b3dcopy -p \"{setname}.tlt\" \"{setname}_fid.tlt\""),
            "#".to_owned(),
            "# CONVERT RESIDUAL FILE TO MODEL".to_owned(),
            "#".to_owned(),
            format!(
                "$if (-e \"{setname}.resid\") patch2imod -s {RESIDUAL_SCALE} \"{setname}.resid\" \"{setname}.resmod\""
            ),
            g.backupline.clone(),
        ]);
        apply_change_list_and_write(&g, "align", sedlines, &dstfile, true, None);
        g.backuplist.extend([
            dstfile.clone(),
            format!("{setname}local.xf"),
            format!("{setname}fid.xyz"),
            format!("{setname}.xf"),
            format!("{setname}.tlt"),
            format!("{setname}.resid"),
            format!("{setname}.resmod"),
            format!("{setname}.3dmod"),
        ]);

        // GET MTFFILTER
        let mut sedcom = sed_for_image_files(&g, &[".ali", "_filt.ali"]);
        sedcom.push(format!(
            "/^PixelSize/s/[ \t].*/\t{}/",
            py_str_float(pixsize)
        ));
        if voltage == PyVal::Int(200) {
            sedcom.push(format!("/^{voltsub}/s/[ \t].*/\t{}/", py_str(&voltage)));
        }
        if !expand_fac.is_empty() {
            sedcom.extend(sed_del_and_add(
                "ExpandedByFactor",
                &expand_fac,
                "PixelSize",
                '/',
            ));
        }
        if half_floats {
            sedcom.extend(sed_del_and_add("ModeToOutput", "12", "PixelSize", '/'));
        }
        if two_dir != 0 && num_bidir_views > 0 {
            sedcom.push(format!(
                "/^PixelSize/a/BidirectionalNumViews\t{num_bidir_views}/"
            ));
            if reversed {
                sedcom.push("/^PixelSize/a/ReversedBidirectional\t1/".to_owned());
            }
        }

        let mut mdoc_ext = "mrc".to_owned();
        if name_style > 0 {
            sedcom.extend(sed_del_and_add(
                "DoseWeightingFile",
                &format!("{setname}.{stack_ext}.mdoc"),
                "PixelSize",
                '/',
            ));
            mdoc_ext = stack_ext.clone();
        }

        // Make an mdoc file from an HDF input so Etomo works without complication
        let mut stack_type = stack_ext.to_uppercase();
        if name_style == 0 {
            if let Ok(format) = get_image_format(&stackname) {
                stack_type = format;
            }
        }
        if stack_type == "HDF" {
            let mdoc_name = format!("{setname}.{mdoc_ext}.mdoc");
            if !Path::new(&mdoc_name).exists() {
                match run_cmd(
                    &format!("extracttilts -attrib {stackname} {mdoc_name}"),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(_) => print_info(&format!(
                        "Extracted mdoc file {mdoc_name} from HDF input file"
                    )),
                    Err(_) => warning(&format!(
                        "Failed to extract mdoc file {mdoc_name} from HDF input file"
                    )),
                }
            }
        }

        edit_and_write(&mut g, "mtffilter", "mtffilter", sedcom, true, None);

        // GET TILT
        let mut delexclude = "gibberish";
        if excludelist.is_empty() {
            delexclude = "EXCLUDELIST";
        }

        // Get a scaling for tilt appropriate to the X size: look up in table
        let mut tiltscale = 0;
        for i in (0..TILTSIZES.len()).rev() {
            tiltscale = TILTSCALES[i];
            let tiltsize = TILTSIZES[i];
            if fullx >= tiltsize {
                break;
            }
        }
        let mut tiltscale_text = tiltscale.to_string();

        // Get mean and SD from 3 sections in file
        let mut sd_avg: f64 = 0.;
        if stack_exists
            && raw_zsize > 3
            && std::fs::metadata(&stackname).map_or(0, |meta| meta.len()) > 1000000
        {
            let clipcom = format!(
                "clip stat -2d -iz {},{},{} {stackname}",
                raw_zsize.div_euclid(3),
                raw_zsize.div_euclid(2),
                (2 * raw_zsize).div_euclid(3)
            );
            // Direct call (owner rule, 2026-09-26): clip records the mean and
            // SD of each section row it prints, used in place of parsing
            // those rows (`copytomocoms:1164-1182`).  The script took the last
            // two words of lines 2-4 of the report, the `%9.4f  %9.4f`
            // columns, so each value is rounded as that format prints it.  A
            // successful run on three sections prints two heading lines, the
            // three rows and a summary, so the script's line-count and
            // field-count checks hold exactly when three rows come back.
            let mut words: Vec<String> = vec!["clip".to_owned(), "stat".to_owned()];
            words.extend(
                format!(
                    "-2d -iz {},{},{}",
                    raw_zsize.div_euclid(3),
                    raw_zsize.div_euclid(2),
                    (2 * raw_zsize).div_euclid(3)
                )
                .split_whitespace()
                .map(str::to_owned),
            );
            words.push(stackname.clone());
            let words: Vec<&str> = words.iter().map(String::as_str).collect();
            let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let recorder = std::sync::Arc::clone(&sink);
            if call_own_program(&clipcom, &words, None, true, move || {
                clip_recording_stat(recorder)
            })
            .is_err()
            {
                exit_from_imod_error(progname);
            }
            let rows = sink.lock().expect("clip stat rows").clone();
            let mut mean_avg = 0.;
            sd_avg = 0.;
            if rows.len() < 3 {
                exit_error("Wrong number of lines from running clip stat on 3 images");
            }
            let rounded = |value: f64| -> Option<f64> {
                c_format("%9.4f", &[CArg::Dbl(value)])
                    .trim()
                    .parse::<f64>()
                    .ok()
            };
            for &(mean, sd) in &rows[..3] {
                match (rounded(mean), rounded(sd)) {
                    (Some(mean), Some(sd)) => {
                        mean_avg += mean / 3.;
                        sd_avg += sd / 3.;
                    }
                    _ => exit_error(
                        &("Converting value to float on line from running clip stat on "
                            .to_owned()
                            + "3 images"),
                    ),
                }
            }
            let _ = mean_avg;
        }

        // Check if a template is turning off the log by modifying key lines
        if !g.change_list.is_empty() {
            let test_lines = vec![
                "$tilt -StandardInput".to_owned(),
                format!("LOG {logbase}"),
                format!("SCALE 0 {tiltscale_text}"),
            ];
            let test_lines =
                match modify_for_change_list(&test_lines, "tilt", &g.setlet, &g.change_list, false)
                {
                    Ok(lines) => lines,
                    Err(message) => exit_error(&message),
                };

            // Set up to divide scale by 5000 if log line is gone, but if the
            // if the log is found or scale is already appropriate for linear, do not divide it
            let mut div_scale = 5000.;
            for line in &test_lines {
                if line.starts_with("LOG") {
                    div_scale = 1.;
                    break;
                }

                // Fixed in translation (BUGS.md): `copytomocoms:1336` calls
                // `atof`, undefined in Python 3, so native's `except Exception`
                // swallows a NameError and a SCALE already below 3 is still
                // divided by 5000; this converts it as `float()` would.
                if line.starts_with("SCALE") {
                    let lsplit: Vec<&str> = line.split_whitespace().collect();
                    if let Some(Ok(new_scale)) = lsplit.get(2).map(|t| t.parse::<f64>()) {
                        if new_scale < 3. {
                            div_scale = 1.;
                        }
                    }
                }
            }

            if div_scale > 1. {
                let mut scale = tiltscale as f64 / div_scale;

                // If linear scaling, adjust it for the SD of the input values
                // This targets an SD of ~200, based on an input SD of 250 giving an
                // output of 325 with scaling 0.2
                // Fixed in translation (BUGS.md): with no `clip stat` sample
                // `sdAvg` is 0 and native raises ZeroDivisionError at
                // `copytomocoms:1350`; with no SD to adjust against, the scale
                // is left at the linear value.
                if sd_avg != 0. {
                    scale *= 150. / sd_avg;
                }
                let num_dig = 0.max((2.31 - scale.log10()).floor() as i32) as usize;
                tiltscale_text = format!("{scale:.num_dig$}");
            }
        }

        let mut sedcom = sed_for_image_files(&g, &[".ali", ".rec"]);
        sedcom.extend([
            format!("/^LOG/s/LOG.*/LOG {logbase}/"),
            format!("/MODE/s/MODE.*/MODE {tiltmode}/"),
            format!("/SCALE/s/SCALE.*/SCALE 0 {tiltscale_text}/"),
            format!("/FULLIMAGE/s//FULLIMAGE {fullx} {fully}/"),
            format!("/EXCLUDELIST/s//EXCLUDELIST {excludelist}/"),
            format!("/{delexclude}/d"),
        ]);
        if sd_avg > 0. {
            sedcom.extend(sed_del_and_add(
                "ReferenceSDofScaling",
                &format!("{sd_avg:.4}"),
                "SCALE",
                '/',
            ));
        }
        if !expand_fac.is_empty() {
            sedcom.extend(sed_del_and_add(
                "ExpandedByFactor",
                &expand_fac,
                "SCALE",
                '/',
            ));
        }

        edit_and_write(&mut g, "tilt", "tilt", sedcom, true, None);

        // COMPUTE OFFSET FOR SAMPLING

        // get indent in Y where 3/4 of the X extent has data after rotation
        // but limit it to 1/4 of Y extent
        let mut rotindent = (0.25 * fullx as f64 * (indentangle * 3.14159 / 180.).tan()) as i32;
        rotindent = rotindent.min(fully.div_euclid(4));

        // Limit offsets to 1/4 of Y extent generally, and sample sizes to half of
        // Y extent, in case of tiny tomograms
        //
        let mut border = MINBORDER1;
        if fully > BORDERSTEP1 {
            border = MINBORDER2;
        }
        if fully > BORDERSTEP2 {
            border = MINBORDER2 + (fully - BORDERSTEP2).div_euclid(BORDERFAC);
        }
        border = (border as f64 * CCDBORDERFAC) as i32;
        border = border.max(rotindent);
        border = (fully.div_euclid(4) - 1).min(border);
        let sampalisize = SAMPALISIZE.min(fully.div_euclid(2) - 2);
        let samplesize = SAMPLESIZE.min(sampalisize - 2);
        let offset = fully.div_euclid(2) - border;
        let halfsampali = sampalisize.div_euclid(2).min(border - 10);
        let sampali = 2 * halfsampali;
        let slmin = halfsampali - samplesize.div_euclid(2);
        let slmax = slmin + samplesize - 1;
        let _sllimit = sampalisize - 1;

        let offarr = [0, offset, -offset];
        let mut bmtarr: Vec<String> = Vec::new();
        for base in ["mid", "top", "bot"] {
            bmtarr.push(dataset_filename(&recext, Some(base), None));
        }

        let dstfile = format!("sample{}", g.dstext);
        g.backuplist.push(dstfile.clone());
        make_backup_file(&dstfile);

        // MAKE UP SAMPLE FILE WITH VARIOUS CASES
        let mut dstlines = vec![
            "# THIS IS A COMMAND FILE TO MAKE 3 TOMOGRAM SAMPLES".to_owned(),
            "#".to_owned(),
            format!("####CreatedVersion#### {}", g.imod_version),
            "# ".to_owned(),
            "# The sample aligned stacks will each be this number of pixels in Y".to_owned(),
            format!(">numLines = {sampalisize}"),
            "# The sample tomograms will have this number of slices".to_owned(),
            format!(">numSlices = {samplesize}"),
            "#".to_owned(),
        ];

        add_output_format_var_to_lines(&mut dstlines, g.name_style, None, false);
        dstlines.push(format!("$b3dcopy tilt{0} tilt_sample{0}", g.dstext));

        // Get the montage min and max values into this array
        let mut mont_lims = ">montYlims = [".to_owned();
        for ind in 0..3 {
            let offind = offarr[ind];
            let ymin = ysize.div_euclid(2) + offind - halfsampali;
            let ymax = ysize.div_euclid(2) + offind + halfsampali;
            mont_lims += &format!("{ymin}, {ymax}");
            if ind < 2 {
                mont_lims += ", ";
            } else {
                mont_lims += "]";
            }
        }
        dstlines.push(mont_lims);

        // Output the lines needed to run slicesforsample and revise the ali size and slices
        dstlines.extend(
            [
                ">noSFS = False".to_owned(),
                ">try:".to_owned(),
                format!(
                    ">  sliceOut = runcmd('slicesforsample {sampali} {offset} {fully} {ysize} tilt{}', inStderr = 'stdout')",
                    g.dstext
                ),
                ">  lsplit = sliceOut[0].split()".to_owned(),
                ">  newLines = int(lsplit[0])".to_owned(),
                ">  if newLines:".to_owned(),
                ">    numLines = newLines".to_owned(),
                ">    for ind in range(6):".to_owned(),
                ">      montYlims[ind] = int(lsplit[ind + 1])".to_owned(),
                ">except ImodpyError:".to_owned(),
                ">  scriptErr = False".to_owned(),
                ">  for l in getErrStrings():".to_owned(),
                ">    if 'ERROR:' in l:".to_owned(),
                ">      scriptErr = True".to_owned(),
                ">      prnstr(l, end='', file=log)".to_owned(),
                ">    else:".to_owned(),
                ">      prnstr('ERROR: ' + l, end='', file=log)".to_owned(),
                ">  if scriptErr:".to_owned(),
                ">    closeExit(1)".to_owned(),
                ">  noSFS = True".to_owned(),
                ">except Exception:".to_owned(),
                ">  prnstr('ERROR: sample com file - converting output from slicesforsample', file = log)"
                    .to_owned(),
                ">  closeExit(1)".to_owned(),
            ],
        );

        let ali_out = dataset_filename("_pos.ali", None, None);
        for ind in 0..3 {
            let offind = offarr[ind];
            let ystart = fully.div_euclid(2) + offind - halfsampali;
            dstlines.push(format!(">ymin = montYlims[{}]", 2 * ind));
            dstlines.push(format!(">ymax = montYlims[{}]", 2 * ind + 1));
            let sedlines;
            if montage == 0 {
                let sedcom = vec![
                    format!("/^OffsetsInXandY/s/[ \t].*/\t 0,{offind}/"),
                    "/^SizeToOutputInXandY/s/,.*/,%numLines/".to_owned(),
                    format!("/^OutputFile/s/[ \t].*/ \t{ali_out}/"),
                    "/if (-e/d".to_owned(),
                    "/AdjustOrigin/d".to_owned(),
                    "/TaperAtFill/d".to_owned(),
                    "/THIS IS A COMMAND FILE/s//THESE ARE COMMANDS/".to_owned(),
                    format!("/####CreatedVersion####/s/# .*/#{}/", g.imod_version),
                ];
                sedlines = pysed(
                    &sedcom,
                    PysedSrc::File(&format!("newst{}", g.dstext)),
                    None,
                    false,
                    '/',
                    false,
                )
                .ok()
                .flatten()
                .unwrap_or_default();
            } else {
                let sedcom = vec![
                    "/^OldEdgeFunctions/s/0/1/".to_owned(),
                    "/^StartingAndEndingY/s/[ \t].*/\t%ymin %ymax/".to_owned(),
                    format!("/^ImageOutputFile/s/[ \t].*/ \t{ali_out}/"),
                    "/if (-e/d".to_owned(),
                    "/AdjustOrigin/d".to_owned(),
                    "/mrctaper/d".to_owned(),
                    "/Command file/s//Commands/".to_owned(),
                    format!("/####CreatedVersion####/s/# .*/#{}/", g.imod_version),
                ];
                let mut lines = pysed(
                    &sedcom,
                    PysedSrc::File(&format!("blend{}", g.dstext)),
                    None,
                    false,
                    '/',
                    false,
                )
                .ok()
                .flatten()
                .unwrap_or_default();
                if !g.change_list.is_empty() {
                    lines = match modify_for_change_list(
                        &lines,
                        "sample",
                        &g.setlet,
                        &g.change_list,
                        false,
                    ) {
                        Ok(lines) => lines,
                        Err(message) => exit_error(&message),
                    };
                }
                sedlines = lines;
            }

            dstlines.extend(sedlines);
            dstlines.push(">if noSFS:".to_owned());
            dstlines.push(format!(
                "$  sampletilt {slmin} {slmax} {ystart} {setname} {} tilt_sample{} {ali_out}",
                bmtarr[ind], g.dstext
            ));
            dstlines.push(">else:".to_owned());
            dstlines.push(format!(
                "$  sampletilt {slmin} {slmax} {ystart} {setname} {} tilt_sample{} {sampali} %numLines %numSlices {ali_out}",
                bmtarr[ind], g.dstext
            ));
        }

        dstlines.push(g.backupline.clone());
        if !g.change_list.is_empty() {
            dstlines =
                match modify_for_change_list(&dstlines, "sample", &g.setlet, &g.change_list, false)
                {
                    Ok(lines) => lines,
                    Err(message) => exit_error(&message),
                };
        }
        write_com_file(&g, &dstfile, &dstlines);

        // SKIP FINDSEC until it is useful!

        // GET TOMOPITCH
        let sedcom = vec![
            format!("/^SpacingInY/s/[ \t].*/\t-{offset}/"),
            format!(r"s/\.mod/{modext}/g"),
            format!(r"s/\.rec/{recext}/g"),
        ];
        edit_and_write(&mut g, "tomopitch", "tomopitch", sedcom, true, None);
        g.backuplist.extend([
            format!("top{modext}"),
            format!("mid{modext}"),
            format!("bot{modext}"),
            format!("tomopitch{modext}"),
        ]);
    }

    if ctf_only != 0 {
        return 0;
    }

    // OUTPUT THE BACKUP FILE

    if backup != 0 {
        if naxis == 2 {
            g.backuplist.extend(
                [
                    "combine.com",
                    "solve.xf",
                    "refine.xf",
                    "inverse.xf",
                    "warp.xf",
                    "patch.out",
                    "combine.log",
                ]
                .map(str::to_owned),
            );
        }
        let mut dstlines = vec![
            "#!/usr/bin/env python".to_owned(),
            "import os, shutil".to_owned(),
            format!("backupdir = r'{backupdir}'"),
            "backuplist = [ \\".to_owned(),
        ];
        let mut line = String::new();
        for num in 0..g.backuplist.len() {
            line += &format!("'{}'", g.backuplist[num]);
            if num < g.backuplist.len() - 1 {
                line += ", ";
            }
            if (num + 1) % 4 == 0 {
                line += "\\";
                dstlines.push(line);
                line = String::new();
            }
        }

        line += "]";
        dstlines.extend(
            [
                line.as_str(),
                "for flnm in backuplist:",
                "  if os.path.exists(flnm):",
                "    flback = backupdir + '/' + flnm",
                "    if not os.path.exists(flback) or \\",
                "       os.stat(flback).st_mtime < os.stat(flnm).st_mtime:",
                "      try:",
                "        shutil.copyfile(flnm, flback)",
                "      except:",
                "        pass",
            ]
            .map(str::to_owned),
        );

        let _ = write_text_file(backupname, &dstlines, false);

        // The chmod failed sporadically with broken pipe on Mac OS X Python 2.6.1
        // So do it only on Windows, use the OS call otherwise.  Note that cygwin python
        // CAN set the desired bits with os.chmod wih native python cannot
        // (`sys.platform` is never win32 or cygwin here, so `needChmod` stays True)

        // If it fails fall through to using python method.  Note that if cygwin does not exist
        // vmstopy will run savework explicitly with python
        {
            use std::os::unix::fs::PermissionsExt as _;
            if std::fs::set_permissions(backupname, std::fs::Permissions::from_mode(0o755)).is_err()
            {
                exit_error(&format!("Setting permissions of {backupname}"));
            }
        }
    }

    // Skip all the verbiage at the end
    if !later_blines.is_empty() {
        print_info(&format!(
            "After the B stack is present, you can do all needed extractions with\n   the command:\n   subm {later_bname}"
        ));
        let _ = write_text_file(later_bname, &later_blines, false);
    }
    prnstr("All command files successfully created", "\n", false);
    0
}
