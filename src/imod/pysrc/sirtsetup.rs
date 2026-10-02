//! Translation of `IMOD/pysrc/sirtsetup`.
//!
//! A Python command script: its five functions are translated one for one
//! below, and its top level is [`sirtsetup`], translated statement by
//! statement.  The module globals the functions read (`scMinMax`, `trimarg`,
//! `srecname`) are passed to them as arguments.
//!
//! `header` (through `getmrc`/`getmrcsize`) is this crate's own program and
//! runs in process.  `splittilt` is also ours, but a Python-script
//! translation whose PIP and `imodpy` state are process-global, so
//! `runcmd` runs it as a child process and its report is parsed as the
//! script does (`findSplitComNumber`), as `dualvolmatch` does for
//! `matchrotpairs` (`CLAUDE.md`, "Direct calls instead of parsed output").
//!
//! Python semantics carried explicitly: `//` floors (`div_euclid` where the
//! divisor is positive), every float is a double, `'{}'.format` of a float
//! is its `repr` ([`py_str_float`]), and `'{:f}'`/`'{:.3f}'` are C's
//! `%f`/`%.3f` (both round the exact binary value half to even).
//!
//! Fixed in translation (`BUGS.md`, sirtsetup): the empty X-tilt file error
//! named an undefined variable (`NameError`); the de-duplication of
//! the leave list called `list.remove(ind)` (a *value*, `ValueError` when
//! absent) and stopped one pair short; a missing `SCALE` line with `LOG`
//! and local alignments took `len(None)`; resuming from a vertical-slice
//! file looked for `.vsrN` where every such file is written `.vsrNN`; and
//! the one-processor difference-reconstruction file was named with a fixed
//! `.com` instead of the command file extension.

use super::imodpy::{
    MrcInfo, OptionValue, add_imod_bin_ignore_sighup, clean_chunk_files,
    complete_and_check_com_file, convert_to_integer, dataset_filename, exit_from_imod_error,
    find_root_axis_and_extensions, get_mrc, get_mrc_size, get_naming_style, glob_glob,
    make_backup_file, option_value, os_path_splitext, parallel_boundary_size, parse_list, prnstr,
    read_text_file, run_cmd, set_root_and_extension,
};
use super::imodpy::{py_fixed, py_float, py_int, py_str_float};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_string, pip_get_two_floats, pip_get_two_integers,
    pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_modify};
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use regex::Regex;
use std::ffi::OsString;
use std::fs::File;
use std::io::Write as _;
use std::path::Path;

const PROGNAME: &str = "sirtsetup";
const SIRTEXT: &str = "srec";
const INTEXT: &str = "sint";
const TRIMEXT: &str = "strm";
const VSEXT: &str = "vsr";

/// Matches `findSplitComNumber` (`sirtsetup:21`): find the number of command
/// files created by splittilt.
pub fn find_split_com_number(splitout: &[String], descrip: &str) -> i32 {
    let mut retval: i32 = -1;
    let reg = Regex::new(r"^.* files for ([ 0-9]*)chunks created.*$").unwrap();
    for l in splitout {
        if l.starts_with("WARNING:") {
            prnstr(l, "\n", false);
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
        exit_error(&format!(
            "Cannot determine com file number from splittilt on {descrip}"
        ));
    }
    retval
}

/// Matches `tryRename` (`sirtsetup:35`): simple rename function in a try block.
pub fn try_rename(sirtcom: &str, comname: &str) {
    if std::fs::rename(sirtcom, comname).is_err() {
        exit_error(&format!("Renaming {sirtcom} to {comname}"));
    }
}

/// Matches `outputScalingLines` (`sirtsetup:42`): output scaling commands.
pub fn output_scaling_lines(
    comf: &mut File,
    recnum: i32,
    sc_min_max: &str,
    trimarg: &str,
    srecname: &str,
) -> std::io::Result<()> {
    if !sc_min_max.is_empty() {
        let convname = dataset_filename(&format!(".{INTEXT}{recnum:02}"), None, None);
        writeln!(comf, "$b3dremove {convname}")?;
        writeln!(
            comf,
            "$newstack -mode 1 -scale {sc_min_max} {srecname} {convname}"
        )?;
    }
    if !trimarg.is_empty() {
        let convname = dataset_filename(&format!(".{TRIMEXT}{recnum:02}"), None, None);
        writeln!(comf, "$b3dremove {convname}")?;
        writeln!(comf, "$trimvol {trimarg} {srecname} {convname}")?;
    }
    Ok(())
}

/// Matches `commandsForFinish` (`sirtsetup:54`): output final commands.  The
/// source closes the file here; it is flushed and closed by the caller's
/// drop.
pub fn commands_for_finish(comf: &mut File, sirtname: &str, testmode: i32) -> std::io::Result<()> {
    writeln!(comf, "$findsirtdiffs {sirtname}")?;
    if testmode < 2 {
        writeln!(comf, "$b3dremove -g {sirtname}-[0-9]*.*")?;
    }
    comf.flush()
}

/// Matches `extractRecNum` (`sirtsetup:61`): extract and convert number from
/// name.
pub fn extract_rec_num(recname: &str, basename: &str, base_ext: &str) -> i32 {
    let chars: Vec<char> = recname.chars().collect();
    let start = basename.chars().count().min(chars.len());
    let end = if !base_ext.is_empty() {
        chars.len().saturating_sub(base_ext.chars().count())
    } else {
        chars.len()
    };
    let num_text: String = if start < end {
        chars[start..end].iter().collect()
    } else {
        String::new()
    };
    convert_to_integer(&num_text, &format!("rec number in {recname}"))
}

/// Writes one of the script's command files: `open`, the writes in `body`,
/// and the `except:` that reports `action + " file: " + comname`.
macro_rules! write_com {
    ($comname:expr, |$comf:ident| $body:block) => {{
        let comname: &str = &$comname;
        let mut action = "Opening";
        let result: std::io::Result<()> = (|| {
            let mut $comf = File::create(comname)?;
            action = "Writing to";
            $body
            Ok(())
        })();
        if result.is_err() {
            exit_error(&format!("{action} file: {comname}"));
        }
    }};
}

/// The script's top level (`sirtsetup:67-819`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn sirtsetup(arguments: &[OsString]) -> i32 {
    let prefix = format!("ERROR: {PROGNAME} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    // Initializations (defaults are in Pip calls)
    let mut recchunk = "";
    let mut projchunk = "";
    let mut target_chunks = String::new();
    let mut boundpixels = parallel_boundary_size(2048);

    // Fallbacks from ../manpages/autodoc2man 3 1 sirtsetup
    let options: Vec<String> = [
        "co:CommandFile:FN:",
        "naming:NamingStyle:I:",
        "nu:NumberOfProcessors:I:",
        "ra:ChunksPerProcessor:I:",
        "st:StartFromZero:B:",
        "re:ResumeFromIteration:I:",
        "it:IterationsToRun:I:",
        "le:LeaveIterations:LI:",
        "sk:SkipVertSliceOutput:B:",
        "cl:CleanUpPastStart:B:",
        "su:SubareaSize:IP:",
        "yo:YOffsetOfSubarea:I:",
        "sc:ScaleToInteger:FP:",
        "tr:TrimvolOptions:CH:",
        "fl:FlatFilterFraction:F:",
        "rd:RadiusAndSigma:FP:",
        "falloff:FalloffIsTrueSigma:B:",
        "cs:ConstrainSign:B:",
        "ch:SeparateRecChunks:B:",
        "pc:SeparateProjChunks:B:",
        "bo:BoundaryPixels:I:",
        "mo:OutputMode:I:",
        "te:TestMode:I:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, PROGNAME, 1, 1, 0);

    // Get the com file name, derive a root name and new com file name, check exists
    let comfile = pip_get_in_out_file("CommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    let (comfile, mut rootname) = complete_and_check_com_file(&comfile);

    let (com_ext, _dual_num, _setroot, type_ext, _stack_ext) =
        find_root_axis_and_extensions(0, Some(&comfile));
    let mut com_ext = com_ext;
    if com_ext.is_empty() {
        com_ext = comfile
            .chars()
            .skip(comfile.chars().count().saturating_sub(3))
            .collect();
    }
    let com_ext = format!(".{com_ext}");

    let (_opt_name_style, type_ext) = match get_naming_style(type_ext.as_deref(), false, false) {
        Ok(result) => result,
        Err(message) => exit_error(&message),
    };
    let mut type_ext = type_ext.unwrap_or_default();

    // Get options
    let mut numproc = pip_get_integer("NumberOfProcessors", 8).unwrap_or(8);
    let proc_entered = 1 - pip_get_err_no();
    let chunk_per_proc = pip_get_integer("ChunksPerProcessor", 0).unwrap_or(0);

    if pip_get_boolean("SeparateRecChunks", 0).unwrap_or(0) != 0 {
        recchunk = "-c";
    }
    if pip_get_boolean("SeparateProjChunks", 0).unwrap_or(0) != 0 {
        projchunk = "-c";
    }

    boundpixels = pip_get_integer("BoundaryPixels", boundpixels).unwrap_or(boundpixels);
    let flatfrac = pip_get_float("FlatFilterFraction", 1.0).unwrap_or(1.0);
    let true_sigma = pip_get_boolean("FalloffIsTrueSigma", 0).unwrap_or(0);
    let mut sigma: f64 = 0.05;
    if true_sigma != 0 {
        sigma = 0.035;
    }
    let radius: f64;
    (radius, sigma) = pip_get_two_floats("RadiusAndSigma", (0.4, sigma)).unwrap_or((0.4, sigma));

    let mut iterations = pip_get_integer("IterationsToRun", 10).unwrap_or(10);
    let iter_entered = 1 - pip_get_err_no();
    let leave_str = pip_get_string("LeaveIterations", "").unwrap_or_default();

    let resume_iter = pip_get_integer("ResumeFromIteration", -1).unwrap_or(-1);
    let mut startfirst = pip_get_boolean("StartFromZero", 0).unwrap_or(0) != 0;
    if startfirst && resume_iter > 0 {
        exit_error("You cannot enter both StartFromZero and ResumeFromIteration");
    }

    let mut using_for_sirt = false;
    if !startfirst && rootname.ends_with("_for_sirt") {
        using_for_sirt = true;
        rootname = rootname
            .chars()
            .take(rootname.chars().count() - 9)
            .collect();
    }

    let cleanup = pip_get_boolean("CleanUpPastStart", 0).unwrap_or(0);

    let (mut sub_xsize, sub_ysize) = pip_get_two_integers("SubareaSize", (0, 0)).unwrap_or((0, 0));
    let do_subarea = 1 - pip_get_err_no();
    let sub_offset = pip_get_integer("YOffsetOfSubarea", 0).unwrap_or(0);

    let sign_constraint = pip_get_integer("ConstrainSign", 0).unwrap_or(0);
    let mode = pip_get_integer("OutputMode", 2).unwrap_or(2);
    let skip_vert = pip_get_integer("SkipVertSliceOutput", 0).unwrap_or(0);

    let (scale_min, scale_max) = pip_get_two_floats("ScaleToInteger", (0., 0.)).unwrap_or((0., 0.));
    let mut sc_min_max = String::new();
    if scale_min != 0. || scale_max != 0. {
        sc_min_max = format!(
            "{},{}",
            py_fixed(scale_min, 0, 6),
            py_fixed(scale_max, 0, 6)
        );
    }
    let trimarg = pip_get_string("TrimvolOptions", "").unwrap_or_default();
    if !sc_min_max.is_empty() && !trimarg.is_empty() {
        exit_error("You cannot enter both scaling and trimming options");
    }

    let testmode = pip_get_integer("TestMode", 0).unwrap_or(0);

    let sirtname = format!("{rootname}_sirt");

    // read com file and get options from it
    let comlines =
        read_text_file(&comfile, Some("tilt command file"), false, None).unwrap_or_default();
    let ints = |value: Option<OptionValue>| match value {
        Some(OptionValue::Integers(values)) => Some(values),
        _ => None,
    };
    let floats = |value: Option<OptionValue>| match value {
        Some(OptionValue::Floats(values)) => Some(values),
        _ => None,
    };
    let strings = |value: Option<OptionValue>| match value {
        Some(OptionValue::String(value)) => Some(value),
        _ => None,
    };
    let mut alifile = strings(option_value(&comlines, "inputproj", 0, true, 0, None, None));
    let mut recfile = strings(option_value(
        &comlines,
        "outputfile",
        0,
        true,
        0,
        None,
        None,
    ));
    let shift_arr = floats(option_value(&comlines, "shift", 2, true, 0, None, None));
    let thickness_arr = ints(option_value(&comlines, "thickness", 1, true, 0, None, None));
    let logbase = floats(option_value(&comlines, "log", 2, true, 0, None, None));
    let xtilt_arr = floats(option_value(&comlines, "xaxistilt", 2, true, 0, None, None));
    let xtilt_file = strings(option_value(&comlines, "xtiltfile", 0, true, 0, None, None));
    // `numVal = 1` returns the single value itself
    let use_gpu =
        ints(option_value(&comlines, "UseGPU", 1, true, 1, None, None)).map(|values| values[0]);
    let zfactors = strings(option_value(
        &comlines,
        "zfactorfile",
        0,
        true,
        0,
        None,
        None,
    ));
    let localali = strings(option_value(&comlines, "localfile", 0, true, 0, None, None));
    let substart = ints(option_value(
        &comlines,
        "subsetstart",
        1,
        true,
        0,
        None,
        None,
    ));
    let binning_arr = ints(option_value(
        &comlines,
        "imagebinned",
        1,
        true,
        0,
        None,
        None,
    ));
    let scale_arr = floats(option_value(&comlines, "scale", 2, true, 0, None, None));
    let mask_option =
        ints(option_value(&comlines, "mask", 1, true, 1, None, None)).map(|values| values[0]);
    let mut binning = 1;
    if let Some(values) = binning_arr.as_ref().filter(|values| !values.is_empty()) {
        binning = values[0];
    }
    let mut xtilt: f64 = 0.;
    if let Some(values) = xtilt_arr.as_ref().filter(|values| !values.is_empty()) {
        xtilt = values[0];
    }

    // If GPU is used and number of procs not entered, assume 1 not 8
    if use_gpu.is_some_and(|value| value >= 0) && proc_entered == 0 {
        numproc = 1;
    }

    if numproc > 1 && chunk_per_proc > 0 {
        target_chunks = format!("-t {} -m {}", numproc * chunk_per_proc, numproc);
    }

    // Figure out if it can be done with internal SIRT
    let mut simple_bp = localali.is_none() && zfactors.is_none();
    if simple_bp && let Some(xtilt_file) = xtilt_file.filter(|file| !file.is_empty()) {
        let xtlines =
            read_text_file(&xtilt_file, Some("X-tilt file"), false, None).unwrap_or_default();
        if xtlines.is_empty() {
            // Defined behaviour: the source names `xtiltfile`, an undefined
            // variable, and dies with a NameError (BUGS.md, sirtsetup).
            exit_error(&format!("The file of X-axis tilts, {xtilt_file}, is empty"));
        }
        let py_float = |text: &str| -> f64 {
            match py_float(text) {
                Some(value) => value,
                None => {
                    eprintln!("ValueError: could not convert string to float: '{text}'");
                    crate::imod::libcfshr::b3dutil::exit(1)
                }
            }
        };
        let firstxt = py_float(&xtlines[0]);
        for i in 0..xtlines.len() {
            if (py_float(&xtlines[i]) - firstxt).abs() > 1.0e-5 {
                simple_bp = false;
            }
        }
        // ELSE ON FOR: the loop has no break, so this always runs
        xtilt += firstxt;
    }

    let doing_vert = simple_bp && xtilt != 0.;

    // Get the input image file name from the command file if necessary
    if alifile.is_none() {
        let tilt_space = Regex::new(r"^\s*\$\s*tilt\s").unwrap();
        let tilt_end = Regex::new(r"^\s*\$\s*tilt$").unwrap();
        let mut ind = 0usize;
        let mut broke = false;
        for index in 0..comlines.len() {
            ind = index;
            if tilt_space.is_match(&comlines[ind]) || tilt_end.is_match(&comlines[ind]) {
                broke = true;
                break;
            }
        }
        if !broke {
            exit_error(&format!("tilt command not found in com file {comfile}"));
        }
        while ind + 1 < comlines.len() {
            ind += 1;
            if !comlines[ind].trim().starts_with('#') {
                alifile = Some(comlines[ind].trim().to_owned());
                break;
            }
        }
        if recfile.is_none() {
            while ind + 1 < comlines.len() {
                ind += 1;
                if !comlines[ind].trim().starts_with('#') {
                    recfile = Some(comlines[ind].trim().to_owned());
                    break;
                }
            }
        }
        if alifile.is_none() || recfile.is_none() {
            exit_error("Cannot find input and output file names in command file");
        }
    }
    let alifile = alifile.unwrap_or_default();

    // Make sure ali exists and get its size
    if !Path::new(&alifile).exists() {
        exit_error(&format!("{alifile} does not exist yet"));
    }
    let (mut alix, mut aliy, aliz, ali_mode) = match get_mrc(&alifile, false, false) {
        Ok(MrcInfo::Basic(nx, ny, nz, mode, ..)) => (nx, ny, nz, mode),
        _ => exit_from_imod_error(PROGNAME),
    };

    // Make sure SUBSETSTART is there
    let substart = match substart {
        Some(values) if values.len() >= 2 => values,
        _ => exit_error("The command file needs to have a SUBSETSTART entry"),
    };
    let mut ss_xstart = substart[0];
    let mut ss_ystart = substart[1];

    // Check size and offset of a subarea
    if do_subarea != 0 {
        let fullx = alix;
        let fully = aliy;
        sub_xsize = 2 * (sub_xsize + 1).div_euclid(2);
        if sub_xsize > alix || sub_ysize > aliy {
            exit_error("Subarea size must be smaller than the aligned stack size");
        }
        // Kept as native (BUGS.md, sirtsetup): the chained comparison
        // `subXsize < 32 < subYsize < 1` can never be true.
        if sub_xsize < 32 && 32 < sub_ysize && sub_ysize < 1 {
            exit_error("The subarea size is too small");
        }
        if aliy.div_euclid(2) + sub_ysize.div_euclid(2) + sub_offset > aliy
            || aliy.div_euclid(2) - sub_ysize.div_euclid(2) + sub_offset < 0
        {
            exit_error(
                &("The subarea offset is too large and the subarea goes ".to_owned()
                    + "outside the image"),
            );
        }
        alix = sub_xsize;
        aliy = sub_ysize;
        ss_xstart += binning * (fullx - alix).div_euclid(2);
        ss_ystart += binning * ((fully - aliy).div_euclid(2) + sub_offset);
    }

    // Get some clip arguments for doing stats (the script computes them and
    // never uses them)
    let mut _iytext = String::new();
    if let Some(values) = thickness_arr.as_ref().filter(|values| !values.is_empty()) {
        _iytext = format!("-iy {}", values[0].div_euclid(2 * binning));
    }
    let _ixtext = format!("-ix {}", alix.div_euclid(2));
    let mut mask_size = 2.max(alix.div_euclid(500));
    if let Some(mask_option) = mask_option {
        mask_size = mask_size.max(mask_option);
    }

    // Get the z shift if any, for substituting in a SHIFT line
    let mut zshift: f64 = 0.;
    if let Some(values) = shift_arr.as_ref().filter(|values| values.len() > 1) {
        zshift = values[1];
    }

    let lastslice = aliy * binning - 1;

    // `recfile` is None here only when `inputproj` was found and `outputfile`
    // was not; the source then fails on `os.path.splitext(None)`.
    let Some(recfile) = recfile else {
        eprintln!("TypeError: expected str, bytes or os.PathLike object, not NoneType");
        crate::imod::libcfshr::b3dutil::exit(1)
    };

    // Get the setname and pull off _rec for standard extension style
    let mut setname = os_path_splitext(&recfile).0;
    let mut descrip_separator = ".";
    if !type_ext.is_empty() {
        descrip_separator = "_";
        let chars: Vec<char> = setname.chars().collect();
        if chars.len() > 4 && chars[chars.len() - 4] == '_' {
            setname = chars[..chars.len() - 4].iter().collect();
        } else {
            type_ext = String::new();
        }
    }

    // adjust it to add _sub or substitute this for _full
    if do_subarea != 0 {
        if setname.ends_with("_full") {
            setname = setname.chars().take(setname.chars().count() - 5).collect();
        }
        setname += "_sub";
    }

    set_root_and_extension(&setname, &type_ext);

    // Get starting reconstruction #: first look for 3 digit ones, then two
    let mut lastnum = 0;
    let mut lastrec = String::new();
    let mut basename = String::new();
    let mut base_ext = String::new();
    if !startfirst {
        basename = format!("{setname}{descrip_separator}{SIRTEXT}");
        if !type_ext.is_empty() {
            base_ext = format!(".{type_ext}");
        }

        // Get the hundreds list, which is all we need if resume >= 100
        let mut reclist = glob_glob(&format!("{basename}[0-9][0-9][0-9]{base_ext}"));
        if resume_iter < 100 {
            // if resume < 100, save the list if resuming, get lower list either
            // in this case or if hundreds list empty
            if resume_iter > 0 {
                let _hundred_list = reclist;
                reclist = Vec::new();
            }
            if reclist.is_empty() {
                reclist = glob_glob(&format!("{basename}[0-9][0-9]{base_ext}"));
            }
        }

        // Now see if resume iteration is present
        if resume_iter > 0 {
            for rec in &reclist {
                let num = extract_rec_num(rec, &basename, &base_ext);
                if num == resume_iter {
                    lastnum = resume_iter;
                    lastrec = rec.clone();
                    break;
                }
            }

            // Error if didn't find it
            if lastnum == 0 {
                exit_error(
                    &("There is no \"srec\" file for the iteration you want to ".to_owned()
                        + "resume from"),
                );
            }
        }
        // Or if not resuming, sort and find last one on list; or start at 0
        else if !reclist.is_empty() {
            reclist.sort();
            lastrec = reclist[reclist.len() - 1].clone();
            lastnum = extract_rec_num(&lastrec, &basename, &base_ext);
            prnstr(
                &format!("Starting from last reconstruction # {lastnum}"),
                "\n",
                false,
            );
        } else {
            startfirst = true;
        }
    }

    // If resuming and doing vertical internal slices, look for the vertical slice rec
    let mut starting_vsrec: Option<String> = None;
    if !startfirst && doing_vert {
        // Defined behaviour: `{:02d}` as every vertical slice file is named
        // (the source's `{}` never found one below iteration 10; BUGS.md).
        let vsrec = dataset_filename(
            &format!(
                ".{VSEXT}{:02}",
                extract_rec_num(&lastrec, &basename, &base_ext)
            ),
            None,
            None,
        );
        if Path::new(&vsrec).exists() {
            starting_vsrec = Some(vsrec);
        }
    }

    // Make sure the rec to resume from is the right size
    if !startfirst {
        let (usex, _usey, usez) = match get_mrc_size(&lastrec) {
            Ok(size) => size,
            Err(_) => exit_from_imod_error(PROGNAME),
        };

        // Also check starting vert slice rec and cancel if it doesn't match
        if let Some(vsrec) = starting_vsrec.clone() {
            let (vsrx, _vsry, vsrz) = match get_mrc_size(&vsrec) {
                Ok(size) => size,
                Err(_) => exit_from_imod_error(PROGNAME),
            };
            if usex != vsrx || usez != vsrz {
                prnstr(
                    &format!(
                        "WARNING: {PROGNAME} - Vertical slice file {vsrec} does not have the right dimensions; using interpolation to resume from regular reconstruction"
                    ),
                    "\n",
                    false,
                );
                starting_vsrec = None;
            }
        }
        if usex != alix || usez != aliy {
            exit_error(&format!(
                "The X/Z size of the existing file {lastrec} ({usex}/{usez}), does not match the X/Y size of the aligned stack ({alix}/{aliy})"
            ));
        }
    }

    // Set up cleanup lists
    let mut cleanlist = String::new();
    if cleanup != 0 {
        for ext in [SIRTEXT, INTEXT, TRIMEXT, VSEXT] {
            let basename = format!("{setname}{descrip_separator}{ext}");
            let reclist = glob_glob(&format!("{basename}[0-9][0-9]*{type_ext}"));
            for rec in &reclist {
                let chars: Vec<char> = rec.chars().collect();
                let mut lastchar = chars.len();
                while !chars[lastchar - 1].is_ascii_digit() {
                    lastchar -= 1;
                }
                let start = basename.chars().count();
                let text: String = chars[start..lastchar].iter().collect();
                let num = match py_int(&text) {
                    Some(value) => value as i32,
                    None => {
                        eprintln!("ValueError: invalid literal for int() with base 10: '{text}'");
                        crate::imod::libcfshr::b3dutil::exit(1)
                    }
                };
                if num > lastnum || startfirst {
                    cleanlist += &format!("{rec} ");
                }
            }
        }

        if !cleanlist.is_empty() {
            prnstr(
                "INFO: The following files will be deleted when the run starts",
                "\n",
                false,
            );
            prnstr(&format!("INFO: {cleanlist}"), "\n", false);
        }
    }

    // Get proper name of the input aligned stack after processing
    let make_log = logbase.is_some() && !simple_bp;
    // Defined behaviour: no SCALE line is not a scale of two values (the
    // source takes `len(None)`; BUGS.md, sirtsetup).
    let change_scale =
        make_log && scale_arr.as_ref().is_some_and(|values| values.len() == 2) && ali_mode != 2;
    let mut aliuse = alifile.clone();
    let mut densin = alifile.clone();
    if do_subarea != 0 || make_log {
        let (mut aliroot, mut aliext) = os_path_splitext(&alifile);
        if !type_ext.is_empty() {
            match aliroot.rfind('_') {
                Some(ind) if ind > 0 && ind < aliroot.len() - 1 => {
                    aliext = format!(".{}", &aliroot[ind + 1..]);
                    aliroot = aliroot[..ind].to_owned();
                }
                _ => exit_error(&format!(
                    "Cannot extract descriptive text between _ and extension from {alifile}"
                )),
            }
        }

        if do_subarea != 0 {
            aliroot += "_sub";
            densin = dataset_filename(&aliext, Some(&aliroot), None);
        }
        if make_log {
            aliext += "log10";
        }

        aliuse = dataset_filename(&aliext, Some(&aliroot), None);
    }

    let mut scale_line = String::new();
    if change_scale {
        let scale_arr = scale_arr.as_ref().unwrap();
        scale_line = format!(
            r"/^\s*SCALE.*/s//SCALE  {} {}/",
            py_str_float(scale_arr[0]),
            py_str_float(scale_arr[1] / 5000.)
        );
    }

    // Make sure modified input stack exists and has right size if restarting
    if !startfirst && aliuse != alifile {
        if !Path::new(&aliuse).exists() {
            exit_error(&format!("{aliuse} does not exist"));
        }
        let (usex, usey, usez) = match get_mrc_size(&aliuse) {
            Ok(size) => size,
            Err(_) => exit_from_imod_error(PROGNAME),
        };
        if usex != alix || usey != aliy || usez != aliz {
            exit_error(&format!(
                "The existing file {aliuse} is {usex}x{usey}x{usez}, not the expected size of {alix}x{aliy}x{aliz}"
            ));
        }
    }

    // Clean up previous stuff that splittilt might not get
    clean_chunk_files(&sirtname, false);

    // Figure out iterations and leave list
    let mut leave_list: Vec<i32> = Vec::new();
    if !leave_str.is_empty() {
        leave_list = parse_list(&leave_str).unwrap_or_default();
        if leave_list.is_empty() {
            exit_error(&format!("Parsing the leave list {leave_str}"));
        }

        // Sort the list and remove duplicates.  Defined behaviour: the
        // duplicate at `ind` is removed and every adjacent pair is checked
        // (the source's `leaveList.remove(ind)` removes the *value* ind, or
        // raises ValueError, and its loop stops one pair short; BUGS.md).
        leave_list.sort();
        let mut ind = 0usize;
        while ind + 1 < leave_list.len() {
            if leave_list[ind] == leave_list[ind + 1] {
                leave_list.remove(ind);
            } else {
                ind += 1;
            }
        }

        let last_leave = leave_list[leave_list.len() - 1];

        // If no iterations entered, set it from the last entry on leave list
        if iter_entered == 0 {
            iterations = last_leave - lastnum;
            if iterations <= 0 {
                exit_error(&format!(
                    "Trying to resume with iteration {}, but the last entry on the list to retain ({last_leave}) is before this iteration ",
                    lastnum + 1
                ));
            }
        }

        // Check validity of leave values
        for &leave in &leave_list {
            if leave <= lastnum || leave > lastnum + iterations {
                exit_error(&format!(
                    "A value on the list to retain ({leave}) is outside the range of iterations being done ({} to {})",
                    lastnum + 1,
                    lastnum + iterations
                ));
            }
        }
    }

    // Set up loop index, and add previous reconstruction and last iteration to
    // leave list to protect both of them
    // then for internal SIRT make length of list be # of iterations
    let mut loop_iter = 0usize;
    let mut do_vert = "";
    if lastnum != 0 {
        leave_list.insert(0, lastnum);
    }
    if leave_list.is_empty() || leave_list[leave_list.len() - 1] != lastnum + iterations {
        leave_list.push(lastnum + iterations);
    }
    if simple_bp {
        prnstr("Doing SIRT internally in the Tilt program", "\n", false);
        do_vert = "-v";
        iterations = leave_list.len() as i32;
        if lastnum != 0 {
            loop_iter = 1;
        }
    }

    // Make an initial reconstruction or starting internal SIRT
    let sirtcom = format!("{sirtname}.com");
    let mut comnum: i32 = 1;
    let mut made_start = false;
    let mut srecname = String::new();
    let mut internal_iter = 0;
    if startfirst {
        if simple_bp {
            // For simple bp, get the first iteration # from leaveList and make
            // sure that is the last number on leaving the loop
            internal_iter = leave_list[loop_iter];
            loop_iter += 1;
            srecname = dataset_filename(&format!(".{SIRTEXT}{internal_iter:02}"), None, None);
            lastnum = internal_iter;
        } else {
            srecname = dataset_filename(&format!(".{SIRTEXT}00"), None, None);
        }
    }

    // For cleanup, or for regular taking the log or subarea when first starting,
    // make com to do this and advance the com number
    if !cleanlist.is_empty() || (startfirst && (make_log || do_subarea != 0)) {
        comnum += 1;
        let comname = format!("{sirtname}-001-sync{com_ext}");
        write_com!(comname, |comf| {
            if !cleanlist.is_empty() {
                writeln!(comf, "$b3dremove {cleanlist}")?;
            }
            if startfirst && do_subarea != 0 {
                writeln!(comf, "$b3dremove {densin}")?;
                writeln!(
                    comf,
                    "$newstack -size {alix},{aliy} -off 0,{sub_offset} {alifile} {densin}"
                )?;
            }

            if startfirst && make_log {
                writeln!(comf, "$b3dremove {aliuse}")?;
                writeln!(
                    comf,
                    "$densnorm -log {} -ignore {densin} {aliuse}",
                    py_fixed(logbase.as_ref().unwrap()[0], 0, 6)
                )?;
            }
        });
    }

    // Set up the base sed commands common to all command files
    let mut sedbase: Vec<String> = vec![
        r"/^\s*SLICE/d".to_owned(),
        r"/^\s*WIDTH/d".to_owned(),
        r"/^\s*MASK/d".to_owned(),
        r"/^\s*RADIAL/d".to_owned(),
        r"/^\s*FalloffIsTrueSigma/d".to_owned(),
        r"/^\s*HammingLikeFilter/d".to_owned(),
        r"/^\s*ExactFilterSize/d".to_owned(),
        r"/^\s*FakeSIRTiterations/d".to_owned(),
        format!(r"/^\s*MODE.*/s//MODE   {mode}/"),
        format!(r"/^\s*SHIFT.*/s//SHIFT 0.0 {}/", py_str_float(zshift)),
        format!(r"/^\s*SUBSETSTART.*/s//SUBSETSTART   {ss_xstart} {ss_ystart}/"),
        format!("/THICKNESS/a/SLICE  0 {lastslice}/"),
        sed_modify("ActionIfGPUFails", "2,2", '/'),
    ];
    if true_sigma != 0 {
        sedbase.push("/THICKNESS/a/FalloffIsTrueSigma  1/".to_owned());
    }
    let radial = format!(
        "/THICKNESS/a/RADIAL   {}/",
        format!("{} {}", py_fixed(radius, 0, 3), py_fixed(sigma, 0, 3))
    );
    let run_pysed = |sedlist: &[String]| {
        let _ = pysed(
            sedlist,
            PysedSrc::Lines(&comlines),
            Some(&sirtcom),
            true,
            '/',
            false,
        );
    };

    if startfirst {
        // Compose the commands for first bp.  Strip log for regular, not for simple
        let mut sedlist = sedbase.clone();
        sedlist.extend([
            format!("/{recfile}/s//{srecname}/"),
            format!("/{alifile}/s//{aliuse}/"),
            radial.clone(),
            format!("/THICKNESS/a/MASK  {mask_size}/"),
            format!(
                "/THICKNESS/a/FlatFilterFraction  {}/",
                py_fixed(flatfrac, 0, 6)
            ),
        ]);

        if simple_bp {
            sedlist.push(format!("/THICKNESS/a/SIRTIterations   {internal_iter}/"));
            sedlist.push("/THICKNESS/a/StartingIteration  1/".to_owned());
            if sign_constraint != 0 {
                sedlist.push(format!("/THICKNESS/a/ConstrainSign  {sign_constraint}/"));
            }
            if doing_vert && skip_vert == 0 {
                let vsrec = dataset_filename(&format!(".{VSEXT}{internal_iter:02}"), None, None);
                sedlist.push(format!("/THICKNESS/a/VertSliceOutputFile  {vsrec}/"));
                starting_vsrec = Some(vsrec);
            }
        } else {
            sedlist.push(r"/^\s*LOG/d".to_owned());
            if change_scale {
                sedlist.push(scale_line.clone());
            }
        }
        run_pysed(&sedlist);

        if numproc > 1 {
            let cmdline = format!(
                "splittilt -d {alix},{aliy} -n {numproc} {target_chunks} -o {recchunk} -b {boundpixels} -i {comnum} {do_vert} {sirtcom}"
            );
            match run_cmd(&cmdline, None, None, None, &[]) {
                Ok(splitout) => {
                    let numchunk =
                        find_split_com_number(&splitout.unwrap_or_default(), "initial run");

                    // If the number started past 1 there was no -start file, so add 1
                    // Otherwise keep track so count can be adjusted at end
                    if comnum > 1 && recchunk.is_empty() {
                        comnum += 1;
                    } else if recchunk.is_empty() {
                        made_start = true;
                    }
                    comnum += numchunk + 1;
                }
                Err(_) => {
                    prnstr("Splittilt failed on initial run", "\n", false);
                    exit_from_imod_error(PROGNAME);
                }
            }
        }
        // One processor, do not split the file, just rename it
        else {
            try_rename(&sirtcom, &format!("{sirtname}-{comnum:03}-sync{com_ext}"));
            comnum += 1;
        }

        // Have to write a finish file now for simple case done in one shot
        // or a scaling file for simple BP
        if loop_iter as i32 == iterations
            || (simple_bp && (!sc_min_max.is_empty() || !trimarg.is_empty()))
        {
            let mut comname = format!("{sirtname}-finish{com_ext}");
            if (loop_iter as i32) < iterations {
                comname = format!("{sirtname}-{comnum:03}-sync{com_ext}");
                comnum += 1;
            }
            write_com!(comname, |comf| {
                if !sc_min_max.is_empty() || !trimarg.is_empty() {
                    output_scaling_lines(&mut comf, lastnum, &sc_min_max, &trimarg, &srecname)?;
                }
                if loop_iter as i32 == iterations {
                    commands_for_finish(&mut comf, &sirtname, testmode)?;
                }
            });
        }
    }

    let _projname = dataset_filename(".proj", None, None);
    let diffname = dataset_filename(".diff", None, None);
    let _drecname = dataset_filename(".drec", None, None);

    // Start the iterations
    while (loop_iter as i32) < iterations {
        let recnum = if simple_bp {
            leave_list[loop_iter]
        } else {
            lastnum + 1
        };
        loop_iter += 1;

        srecname = dataset_filename(&format!(".{SIRTEXT}{recnum:02}"), None, None);
        let lastname = dataset_filename(&format!(".{SIRTEXT}{lastnum:02}"), None, None);

        // Set rec file to reproject from a vert slice file if any
        let mut rec_to_reproj = lastname.clone();
        if let Some(vsrec) = &starting_vsrec {
            rec_to_reproj = vsrec.clone();
        }

        // Make commands for reprojection or SIRT iterations
        let mut sedlist = sedbase.clone();
        sedlist.extend([
            format!("/{alifile}/s//{aliuse}/"),
            format!("/THICKNESS/a/RecFileToReproj  {rec_to_reproj}/"),
        ]);

        let this_chunk;
        if simple_bp {
            sedlist.push(format!("/{recfile}/s//{srecname}/"));
            sedlist.push(radial.clone());
            sedlist.push(format!("/THICKNESS/a/MASK  {mask_size}/"));
            sedlist.push(format!(
                "/THICKNESS/a/SIRTIterations  {}/",
                recnum - lastnum
            ));
            sedlist.push(format!("/THICKNESS/a/StartingIteration  {lastnum}/"));
            if sign_constraint != 0 {
                sedlist.push(format!("/THICKNESS/a/ConstrainSign  {sign_constraint}/"));
            }

            // If using starting vertical slices, indicate so and use them only once
            // unless...  If outputting vertical slices, set up filename to use it on
            // next round
            if starting_vsrec.is_some() {
                sedlist.push("/THICKNESS/a/VertForSIRTInput/".to_owned());
                starting_vsrec = None;
            }
            if doing_vert && skip_vert == 0 {
                let vsrec = dataset_filename(&format!(".{VSEXT}{recnum:02}"), None, None);
                sedlist.push(format!("/THICKNESS/a/VertSliceOutputFile  {vsrec}/"));
                starting_vsrec = Some(vsrec);
            }
            this_chunk = recchunk;
        } else {
            sedlist.push(format!("/{recfile}/s//{diffname}/"));
            sedlist.push(radial.clone());
            sedlist.push("/THICKNESS/a/ViewsToReproj  0/".to_owned());
            sedlist.push("/THICKNESS/a/SIRTSubtraction/".to_owned());
            sedlist.push(r"/^\s*LOG/d".to_owned());
            if change_scale {
                sedlist.push(scale_line.clone());
            }
            this_chunk = projchunk;
        }

        run_pysed(&sedlist);
        if numproc > 1 {
            let cmdline = format!(
                "splittilt -d {alix},{aliy} -n {numproc} {target_chunks} -o {this_chunk} -b {boundpixels} -i {comnum} {do_vert} {sirtcom}"
            );
            match run_cmd(&cmdline, None, None, None, &[]) {
                Ok(splitout) => {
                    let numchunk = find_split_com_number(
                        &splitout.unwrap_or_default(),
                        &format!("projection for iteration {recnum}"),
                    );

                    comnum += numchunk + 1;
                    if this_chunk.is_empty() {
                        comnum += 1;
                    }
                }
                Err(_) => {
                    prnstr(
                        &format!("Splittilt failed setting up projection for iteration {recnum}"),
                        "\n",
                        false,
                    );
                    exit_from_imod_error(PROGNAME);
                }
            }
        }
        // One processor, do not split the file, just rename it
        else {
            try_rename(&sirtcom, &format!("{sirtname}-{comnum:03}-sync{com_ext}"));
            comnum += 1;
        }

        if !simple_bp {
            // Make the error reconstruction and subtract from rec
            let mut sedlist = sedbase.clone();
            sedlist.extend([
                format!("/{recfile}/s//{srecname}/"),
                format!("/{alifile}/s//{diffname}/"),
                r"/^\s*LOG/d".to_owned(),
                radial.clone(),
                format!("/THICKNESS/a/MASK  {mask_size}/"),
                "/THICKNESS/a/FlatFilterFraction  2/".to_owned(),
                format!("/THICKNESS/a/BaseRecFile  {lastname}/"),
                format!("/THICKNESS/a/StartingIteration  {recnum}/"),
                "/THICKNESS/a/SubtractFromBase   -1/".to_owned(),
            ]);
            if change_scale {
                sedlist.push(scale_line.clone());
            }

            if sign_constraint != 0 {
                sedlist.push(format!("/THICKNESS/a/ConstrainSign  {sign_constraint}/"));
            }

            run_pysed(&sedlist);
            if numproc > 1 {
                let cmdline = format!(
                    "splittilt -d {alix},{aliy} -n {numproc} {target_chunks} -o {recchunk} -b {boundpixels} -i {comnum} {sirtcom}"
                );
                match run_cmd(&cmdline, None, None, None, &[]) {
                    Ok(splitout) => {
                        let numchunk = find_split_com_number(
                            &splitout.unwrap_or_default(),
                            &format!("difference reconstruction for iteration # {recnum}"),
                        );

                        comnum += numchunk + 1;
                        if recchunk.is_empty() {
                            comnum += 1;
                        }
                    }
                    Err(_) => {
                        prnstr(
                            &format!(
                                "Splittilt failed setting up difference reconstruction for iteration # {recnum}"
                            ),
                            "\n",
                            false,
                        );
                        exit_from_imod_error(PROGNAME);
                    }
                }
            } else {
                // Defined behaviour: the command file extension, where the
                // source writes a fixed `.com` here only (BUGS.md, sirtsetup).
                try_rename(&sirtcom, &format!("{sirtname}-{comnum:03}-sync{com_ext}"));
                comnum += 1;
            }
        }

        // Put out a sync file that subtracts diff reconstruction from last recon
        // if appropriate and manages conversions and leaving
        if loop_iter as i32 == iterations
            || !simple_bp
            || !sc_min_max.is_empty()
            || !trimarg.is_empty()
        {
            let mut comname = format!("{sirtname}-{comnum:03}-sync{com_ext}");

            if loop_iter as i32 == iterations {
                comname = format!("{sirtname}-finish{com_ext}");
            }
            let mut advance = false;
            write_com!(comname, |comf| {
                if !simple_bp {
                    let mut deldiff = diffname.as_str();
                    if testmode != 0 {
                        deldiff = "";
                    }
                    writeln!(comf, "$b3dremove {deldiff} {lastname}~")?;
                }

                // Delete something not on the leave list unconditionally, or delete
                // something on the list if it got converted
                if !leave_list.contains(&lastnum) {
                    writeln!(comf, "$b3dremove {lastname}")?;
                } else if !sc_min_max.is_empty() || !trimarg.is_empty() {
                    let mut lastconv =
                        dataset_filename(&format!(".{INTEXT}{lastnum:02}"), None, None);
                    if !trimarg.is_empty() {
                        lastconv = dataset_filename(&format!(".{TRIMEXT}{lastnum:02}"), None, None);
                    }
                    let mut rmstring = format!("$if (-e {lastconv}) b3dremove {lastname}");
                    if doing_vert && skip_vert == 0 {
                        rmstring += &format!(
                            " {}",
                            dataset_filename(&format!(".{VSEXT}{lastnum:02}"), None, None)
                        );
                    }
                    writeln!(comf, "{rmstring}")?;
                }

                // Set up conversion if requested
                if leave_list.contains(&recnum) && (!sc_min_max.is_empty() || !trimarg.is_empty()) {
                    output_scaling_lines(&mut comf, recnum, &sc_min_max, &trimarg, &srecname)?;
                }

                if loop_iter as i32 == iterations {
                    commands_for_finish(&mut comf, &sirtname, testmode)?;
                } else {
                    advance = true;
                }
            });
            if advance {
                comnum += 1;
            }
        }

        // End of loop at last
        lastnum = recnum;
    }

    if made_start {
        comnum += 1;
    }

    // Copy the tilt.com that was used for etomo to use as a checkpoint on resumability
    if !using_for_sirt {
        let usedname = format!("{rootname}_for_sirt{com_ext}");
        make_backup_file(&usedname);
        // `shutil.copyfile`: the contents only, not the permission bits
        if std::fs::read(&comfile)
            .and_then(|bytes| std::fs::write(&usedname, bytes))
            .is_err()
        {
            prnstr(
                &format!("WARNING: {PROGNAME} - Failed to copy {comfile} to {usedname}"),
                "\n",
                false,
            );
        }
    }

    prnstr(
        &format!("{comnum} command files created and ready to run with:"),
        "\n",
        false,
    );
    prnstr(
        &format!("  processchunks machine_list {sirtname}"),
        "\n",
        false,
    );
    if numproc == 1 {
        prnstr("Or with:", "\n", false);
        prnstr(&format!("  subm {sirtname}*{com_ext}"), "\n", false);
    }

    let _ = std::io::stdout().flush();
    0
}
