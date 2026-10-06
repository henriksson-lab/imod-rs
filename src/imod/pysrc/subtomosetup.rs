//! Translation of `IMOD/pysrc/subtomosetup`: makes command files for
//! reconstructing subvolumes around points, optionally with CTF correction
//! at several Z levels, 2D filtering, gold erasing, unaligned input or a new
//! aligned stack binning.
//!
//! A Python command script; its one function is [`make_next_com_name`] and
//! its top level is [`subtomosetup`].  The shared functions come from
//! `tomocoords.py` ([`super::tomocoords`]).  `newstack`, `header`,
//! `imodinfo`, `imodextract`, `model2point` and `xfmodel` are our own
//! programs and run in process through `imodpy::run_cmd` where the command
//! line allows; `splitcorrection` is a Python-script translation, which
//! `run_cmd` runs as a child.
//!
//! Python values keep their types: positions, shifts, pixel sizes and Z
//! extents are floats written with `str()` ([`py_str_float`]); sizes,
//! binnings and slice numbers are ints, with Python's flooring `//`.

use super::imodpy::{
    BOOL_VALUE, FLOAT_VALUE, INT_VALUE, OptionValue, STRING_VALUE, balanced_group_limits,
    clean_chunk_files, cleanup_files, dataset_filename, exit_from_imod_error,
    find_root_axis_and_extensions, fmtstr, get_mrc_size, option_value, os_path_abspath,
    os_path_splitext, parallel_boundary_size, prnstr, py_fixed, py_int_floordiv, py_int_mod,
    py_round, py_round_ndigits, py_str_float, py_true_div, read_text_file, run_cmd,
    write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_three_integers,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add, sed_modify};
use super::tomocoords::{
    back_transform_erase_model, check_for_distortion, check_xtilt_ctf_vs_rec,
    find_split_com_number, get_axis_angle_and_transpose, get_common_options,
    get_ctf_options_check_if_corrected, get_essential_raw_options, get_fallback_raw_pixel,
    get_or_derive_com_file, get_points_and_headers, get_raw_extension_fallback, set_axis_letter,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// `xzShiftDecimals` (`subtomosetup:11`).
const XZ_SHIFT_DECIMALS: i32 = 2;
/// `maxChunks` (`subtomosetup:12`).
const MAX_CHUNKS: i64 = 99990;
/// `minChunks` (`subtomosetup:13`).
const MIN_CHUNKS: i64 = 8;
/// `maxChunkPerProc` (`subtomosetup:14`).
const MAX_CHUNK_PER_PROC: i64 = 10;
/// `minChunkPerProc` (`subtomosetup:15`).
const MIN_CHUNK_PER_PROC: i64 = 5;
/// `warnXtiltCrit` (`subtomosetup:16`).
const WARN_XTILT_CRIT: f64 = 0.3;
/// `debug` (`subtomosetup:17`).
const DEBUG: i32 = 0;

/// Matches `makeNextComName` (`subtomosetup:20`): compose the next command
/// file name and increment the number, making a sync if indicated.
/// `root_with_dir` and `com_num` are the script's globals.
pub fn make_next_com_name(com_num: &mut i32, root_with_dir: &str, do_sync: bool) -> String {
    let com_name = if do_sync {
        format!("{root_with_dir}-{:03}-sync.com", *com_num)
    } else {
        format!("{root_with_dir}-{:03}.com", *com_num)
    };
    *com_num += 1;
    com_name
}

/// The script's top level (`subtomosetup:31-809`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn subtomosetup(arguments: &[OsString]) -> i32 {
    let progname = "subtomosetup";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };
    let lines_of = |result: Result<Option<Vec<String>>, String>| -> Vec<String> {
        result.ok().flatten().unwrap_or_default()
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        super::imodpy::add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 subtomosetup
    let options: Vec<String> = [
        "root:RootName:CH:",
        "center:CenterPositionFile:FN:",
        "volume:VolumeModeled:FN:",
        "raw:RawStackFile:FN:",
        "axis:AxisAngle:F:",
        "objects:ObjectsToUse:LI:",
        "size:SizeInXYZ:IT:",
        "dir:DirectoryForOutput:FN:",
        "chunk:DirectoryForChunkFiles:FN:",
        "skip:SkipSubVolNumbers:B:",
        "stackvols:MakeVolumeStacks:I:",
        "com:CommandFile:FN:",
        "binali:NewAlignedBinning:I:",
        "newstcom:NewstackComFile:FN:",
        "unaligned:UseUnalignedImages:B:",
        "reduce:FourierReduceByFactor:I:",
        "zlevels:NumberOfZLevels:I:",
        "extent:ExtentOfZLevelsInNm:I:",
        "invert:InvertZLevelOffsets:I:",
        "adjust:AdjustForAlignZShift:B:",
        "ctfcom:CorrectionComFile:FN:",
        "erase:EraseFiducials:B:",
        "goldcom:GoldEraserComFile:FN:",
        "filter:FilterIn2D:B:",
        "2dcom:2DFilterComFile:FN:",
        "gpu:WhenToUseGPU:I:",
        "pixel:RawPixelSize:F:",
        "xform:AlignTransformFile:FN:",
        "reorient:ReorientionType:I:",
        "proc:ProcessorNumber:I:",
        "runs:RunsPerChunk:I:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let common = get_common_options(&argv, &options, progname);
    let root_name = common.root_name;
    let vol_name = common.vol_name;
    let center_file = common.center_file;
    let point_file = common.point_file;
    let model_file = common.model_file;
    let obj_list = common.obj_list;
    let com_file = common.com_file;
    let reorient_in = common.reorient_in;
    let entered_orient = common.entered_orient;

    let mut check_list: Vec<(String, String)> = Vec::new();
    let bound_pixels = parallel_boundary_size(2048);
    let mut out_dir = pip_get_string("DirectoryForOutput", "").unwrap_or_default();
    let skip_vol_numbers = pip_get_boolean("SkipSubVolNumbers", 0).unwrap_or(0) != 0;
    // `PipGetString('DirectoryForChunkFiles', 0)`: the default 0 is falsy
    let com_dir = pip_get_string("DirectoryForChunkFiles", "").unwrap_or_default();
    let (x_size, y_size, z_size) =
        pip_get_three_integers("SizeInXYZ", (0, 0, 0)).unwrap_or((0, 0, 0));
    if x_size < 1 || y_size < 1 || z_size < 1 {
        exit_error("Positive sizes must be entered for the reconstructions");
    }
    let (x_size, y_size, z_size) = (x_size as i64, y_size as i64, z_size as i64);

    let make_vol_stacks = pip_get_integer("MakeVolumeStacks", 0).unwrap_or(0) as i64;
    if pip_get_err_no() == 0 && make_vol_stacks < 5 {
        exit_error("The volume stack entry must be at least 5");
    }
    let mut num_zlevels = pip_get_integer("NumberOfZLevels", 0).unwrap_or(0) as i64;
    let max_zextent = pip_get_float("ExtentOfZLevelsInNm", 0.).unwrap_or(0.);
    if num_zlevels != 0 && max_zextent != 0. {
        exit_error("You cannot enter both a number of Z levels and their extent");
    }
    let do_ctf = num_zlevels > 0 || max_zextent > 0.;
    let mut invert_zoffsets = pip_get_integer("InvertZLevelOffsets", -1).unwrap_or(-1);
    let adjust_for_z_shift = pip_get_boolean("AdjustForAlignZShift", 0).unwrap_or(0) != 0;

    let mut ctf_com_file = String::new();
    let mut ctf_lines: Vec<String> = Vec::new();
    if do_ctf {
        ctf_com_file = get_or_derive_com_file(
            "CorrectionComFile",
            "ctfcorrection",
            &com_file,
            "CTF correction",
        );
        ctf_lines = read_text_file(
            &ctf_com_file,
            Some("CTF correction command file"),
            false,
            None,
        )
        .unwrap_or_default();
        check_list.push((
            ctf_com_file.clone(),
            "CTF correction command file".to_owned(),
        ));
    }

    let do_erase = pip_get_boolean("EraseFiducials", 0).unwrap_or(0) != 0;
    let mut erase_com_file = String::new();
    if do_erase {
        erase_com_file =
            get_or_derive_com_file("GoldEraserComFile", "golderaser", &com_file, "erasing gold");
        check_list.push((
            erase_com_file.clone(),
            "gold erasing command file".to_owned(),
        ));
    }

    // Determine filtering
    let do_filter = pip_get_boolean("FilterIn2D", 0).unwrap_or(0) != 0;
    let mut mtf_lines: Vec<String> = Vec::new();
    if do_filter {
        let mtf_com_file =
            get_or_derive_com_file("2DFilterComFile", "mtffilter", &com_file, "2D filtering");
        mtf_lines = read_text_file(
            &mtf_com_file,
            Some("2D filtering command file"),
            false,
            None,
        )
        .unwrap_or_default();
    }

    let (mut com_ext, if_dual, raw_root, type_ext, raw_ext) =
        find_root_axis_and_extensions(0, None);
    let axis_let = set_axis_letter(&root_name, if_dual);

    // Get raw stack name
    let mut stack_name = pip_get_string("RawStackFile", "").unwrap_or_default();
    if stack_name.is_empty() {
        let (raw_ext, num_candid) = get_raw_extension_fallback(&raw_ext, &root_name);

        if num_candid == 0 || num_candid > 1 {
            exit_error("The raw stack name must be entered, it cannot be deduced");
        }

        stack_name = format!("{root_name}.{raw_ext}");
    }

    // Get the aligned stack name, or give up and assume it
    let type_ext = type_ext.unwrap_or_default();
    let ali_name = if !type_ext.is_empty() {
        dataset_filename(".ali", Some(&root_name), Some(&type_ext))
    } else {
        let tilt_lines = read_text_file(&com_file, None, false, None).unwrap_or_default();
        match option_value(&tilt_lines, "inputproj", 0, true, 0, None, None) {
            Some(OptionValue::String(value)) if !value.is_empty() => value,
            _ => format!("{root_name}.ali"),
        }
    };

    // Get the axis angle and whether X/Y are transposed
    let (axis_angle, transpose_xy) = get_axis_angle_and_transpose(axis_let.as_deref());

    // See if using raw input
    let use_raw_input = pip_get_boolean("UseUnalignedImages", 0).unwrap_or(0) != 0;
    let mut split_corr_size = String::new();
    let mut ali_xform_file = String::new();
    let mut raw_binning: i64 = 1;
    let mut raw_pix_size = 0.0_f64;
    if use_raw_input {
        let raw_binning_entry;
        (ali_xform_file, raw_binning_entry, raw_pix_size) = get_essential_raw_options(&raw_root);
        raw_binning = raw_binning_entry as i64;
        let (nx_corr, ny_corr, nz_corr) = match get_mrc_size(&stack_name) {
            Ok(size) => size,
            Err(_) => exit_from_imod_error(progname),
        };

        split_corr_size = format!(
            "-size {},{},{}",
            py_int_floordiv(nx_corr as i64, raw_binning),
            py_int_floordiv(ny_corr as i64, raw_binning),
            nz_corr
        );
    }

    // See if remaking aligned stack
    let new_ali_binning = pip_get_integer("NewAlignedBinning", 0).unwrap_or(0) as i64;
    let mut newst_lines: Vec<String> = Vec::new();
    let mut old_ali_bin: i64 = 1;
    if new_ali_binning > 0 {
        if use_raw_input {
            exit_error("You cannot use unaligned images and specify a new aligned stack binning");
        }
        let newst_com_file = get_or_derive_com_file(
            "NewstackComFile",
            "newst",
            &com_file,
            "making new aligned stack",
        );
        newst_lines = read_text_file(&newst_com_file, None, false, None).unwrap_or_default();

        // Make modifications now to binning and to size if it is present
        let mut sedcom = sed_del_and_add(
            "BinByFactor",
            &new_ali_binning.to_string(),
            "TransformFile",
            '/',
        );
        let ali_out_size = match option_value(
            &newst_lines,
            "SizeToOutputInXandY",
            INT_VALUE,
            false,
            0,
            None,
            None,
        ) {
            Some(OptionValue::Integers(values)) => Some(values),
            _ => None,
        };
        old_ali_bin =
            match option_value(&newst_lines, "BinByFactor", INT_VALUE, false, 1, None, None) {
                Some(OptionValue::Integers(values)) if values[0] != 0 => values[0] as i64,
                _ => 1,
            };
        if let Some(mut ali_out_size) = ali_out_size {
            // `BUGS.md` (subtomosetup `SizeToOutputInXandY`), fixed in
            // translation: a one-value entry makes the source index past the
            // list and die with an IndexError traceback.
            if ali_out_size.len() < 2 {
                exit_error(
                    "The SizeToOutputInXandY entry in the newstack command file needs two values",
                );
            }
            let mut out = [0i64; 2];
            for ind in 0..2 {
                out[ind] = py_int_floordiv(
                    ali_out_size[ind] as i64 * old_ali_bin + new_ali_binning - 1,
                    new_ali_binning,
                );
                ali_out_size[ind] = out[ind] as i32;
            }
            sedcom.push(sed_modify(
                "SizeToOutputInXandY",
                &format!("{},{}", out[0], out[1]),
                '/',
            ));
        }
        newst_lines = lines_of(pysed(
            &sedcom,
            PysedSrc::Lines(&newst_lines),
            None,
            false,
            '/',
            false,
        ));

        let mut run_lines: Vec<String> = Vec::new();
        let mut got_run = false;

        // Get the lines for running it
        for line in &newst_lines {
            if got_run && line.starts_with('$') {
                break;
            }
            if !got_run && line.starts_with('$') && line.contains("newstack") {
                got_run = true;
            } else if got_run {
                run_lines.push(line.clone());
            }
        }

        if !got_run {
            exit_error("Could not find newstack line in command file");
        }

        // Make a stack with one section in order to get the size and header correct
        let sedcom = sed_del_and_add("SectionsToRead ", "0", "TransformFile", '/');
        run_lines = lines_of(pysed(
            &sedcom,
            PysedSrc::Lines(&run_lines),
            None,
            false,
            '/',
            false,
        ));
        if run_cmd("newstack -StandardInput", Some(&run_lines), None, None, &[]).is_err() {
            exit_from_imod_error(progname);
        }
    }

    // Check that all the files exist
    check_list.extend([
        (stack_name.clone(), "raw stack".to_owned()),
        (vol_name.clone(), "modeled volume".to_owned()),
        (com_file.clone(), "command file".to_owned()),
        (center_file.clone(), "center position file".to_owned()),
    ]);
    if !use_raw_input && new_ali_binning <= 0 {
        check_list.push((ali_name.clone(), "aligned stack".to_owned()));
    }
    let mut align_com = String::new();
    if do_ctf && adjust_for_z_shift {
        let Some(axis) = axis_let.as_deref() else {
            exit_error("Cannot determine axis for finding align command file");
        };
        align_com = format!("align{axis}.com");
        check_list.push((align_com.clone(), "align command file".to_owned()));
    }

    for (name, descrip) in &check_list {
        if !Path::new(name).exists() {
            exit_error(&format!("The {descrip}, {name}, does not exist"));
        }
    }

    // Check and create output directory
    if !out_dir.is_empty() {
        if Path::new(&out_dir).exists() {
            if !Path::new(&out_dir).is_dir() {
                exit_error(
                    "The specified name for output directory already exists and is not a directory",
                );
            }
        } else {
            if std::fs::create_dir(&out_dir).is_err() {
                exit_error(&format!("Making directory for output, {out_dir}"));
            }

            // `os.path.relpath(outDir)`: the absolute path relative to the
            // absolute current directory, with `..` for each component of
            // the current directory past the common prefix
            let path_list: Vec<String> = os_path_abspath(&out_dir)
                .split('/')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect();
            let start_list: Vec<String> = os_path_abspath(".")
                .split('/')
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect();
            let common = path_list
                .iter()
                .zip(start_list.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let mut rel_list: Vec<String> = vec!["..".to_owned(); start_list.len() - common];
            rel_list.extend(path_list[common..].iter().cloned());
            out_dir = if rel_list.is_empty() {
                ".".to_owned()
            } else {
                rel_list.join("/")
            };
            if out_dir.starts_with('/') {
                prnstr(
                    &format!(
                        "WARNING: {progname} - The output directory cannot be converted to a relative path and may not work on other machines"
                    ),
                    "\n",
                    false,
                );
            }
        }
    }

    if !com_dir.is_empty() {
        if Path::new(&com_dir).exists() {
            if !Path::new(&com_dir).is_dir() {
                exit_error(
                    "The specified name for command file directory already exists and is not a directory",
                );
            }
        } else if std::fs::create_dir(&com_dir).is_err() {
            exit_error(&format!("Making directory for command files, {com_dir}"));
        }
    }

    // Get possible entries for runs per chunk and # of processors
    let mut num_proc = pip_get_integer("ProcessorNumber", 0).unwrap_or(0) as i64;
    let zero_proc_entered = pip_get_err_no() == 0 && num_proc == 0;
    let num_runs_per_chunk = pip_get_integer("RunsPerChunk", 10).unwrap_or(10) as i64;
    if pip_get_err_no() == 0 && num_proc > 0 {
        exit_error("You cannot enter both -proc and -runs");
    }

    let full_rec: Option<&str> = None;
    let ph = get_points_and_headers(
        &model_file,
        &obj_list,
        &point_file,
        progname,
        &stack_name,
        &ali_name,
        &vol_name,
        full_rec,
        entered_orient,
        reorient_in,
        &com_file,
        use_raw_input,
    );
    let mut point_list = ph.point_list;
    let mut ali_binning = ph.ali_binning as i64;
    let reorient = ph.reorient;
    let tilt_lines = ph.com_lines;
    let (mut nx_raw, mut ny_raw, nz_raw) = (ph.nx_raw as i64, ph.ny_raw as i64, ph.nz_raw as i64);
    let pix_x_raw = ph.pix_x_raw;
    let pix_y_raw = ph.pix_y_raw;
    let (mut nx_ali, mut ny_ali, mut nz_ali) =
        (ph.nx_ali as i64, ph.ny_ali as i64, ph.nz_ali as i64);
    let (mut pix_x_ali, mut pix_y_ali) = (ph.pix_x_ali, ph.pix_y_ali);
    let (mut orig_x_ali, mut orig_y_ali) = (ph.orig_x_ali, ph.orig_y_ali);
    let (pix_x_vol, pix_y_vol, pix_z_vol) = (ph.pix_x_vol, ph.pix_y_vol, ph.pix_z_vol);
    let (orig_x_vol, orig_y_vol, orig_z_vol) = (ph.orig_x_vol, ph.orig_y_vol, ph.orig_z_vol);

    // Fix nzAli to be raw size for new ali binning or raw input, for raw it is used
    // only for getting the maxSlices
    if new_ali_binning > 0 || use_raw_input {
        nz_ali = nz_raw;
    }

    if matches!(
        option_value(&tilt_lines, "ExpandedByFactor", FLOAT_VALUE, false, 1, None, None),
        Some(OptionValue::Floats(values)) if values[0] != 0.
    ) {
        exit_error("Reconstructions with an expansion factor applied are not (yet) supported");
    }

    let mut xaxis_tilt = 0.0_f64;
    let mut ctf_xtilt: Option<f64> = None;
    let mut tilt_use_gpu: i32;
    let mut ctf_use_gpu: i32 = -1;
    let mut ctf_pix_size: f64;
    let mut ctf_input: String;
    let mut ctf_output: String;
    let mut z_shift_adjustment = 0.0_f64;
    if do_ctf {
        let mut raw_arg = "";
        if use_raw_input {
            raw_arg = &stack_name;
        }
        let ctf_options = get_ctf_options_check_if_corrected(
            progname,
            &tilt_lines,
            &ctf_lines,
            raw_arg,
            do_filter,
        );
        xaxis_tilt = ctf_options.xaxis_tilt;
        tilt_use_gpu = ctf_options.tilt_use_gpu;
        ctf_xtilt = ctf_options.ctf_xtilt;
        ctf_input = ctf_options.ctf_input.unwrap_or_default();
        // `BUGS.md` (subtomosetup CTF entries), fixed in translation: with no
        // `OutputFileName` (or no `PixelSize` for an aligned stack) in the
        // CTF command file, the source passes `None` to `os.path.splitext`
        // (or divides it) and dies with a TypeError; ctf3dsetup's error for
        // the same missing entries is given instead.
        let (Some(output), Some(pixel)) = (ctf_options.ctf_output, ctf_options.ctf_pix_size) else {
            exit_error(&format!(
                "Cannot find needed information in {ctf_com_file} (PixelSize, input or output file)"
            ));
        };
        ctf_output = output;
        ctf_pix_size = pixel;
        if !use_raw_input {
            split_corr_size = format!("-size {nx_ali},{ny_ali},{nz_ali}");
        }
        ctf_use_gpu = match ctf_options.ctf_use_gpu {
            Some(value) if value >= 0 => value,
            _ => tilt_use_gpu,
        };

        if adjust_for_z_shift {
            let ali_lines = read_text_file(&align_com, Some("align command file"), false, None)
                .unwrap_or_default();
            // `BUGS.md` (ctf3dsetup/subtomosetup `AxisZShift`), fixed in
            // translation: the source leaves `startingZshift` undefined
            // (NameError) when tilt.com has no SHIFT entry with two values,
            // and adds `None` (TypeError) when align.com has no
            // `AxisZShift`; both missing values are 0 here, the default of
            // tilt and tiltalign.
            let align_zshift =
                match option_value(&ali_lines, "AxisZShift", FLOAT_VALUE, false, 1, None, None) {
                    Some(OptionValue::Floats(values)) => values[0],
                    _ => 0.,
                };
            let mut starting_zshift = 0.0_f64;
            if let Some(OptionValue::Floats(shift_arr)) =
                option_value(&tilt_lines, "shift", 2, true, 0, None, None)
                && shift_arr.len() > 1
            {
                starting_zshift = shift_arr[1];
            }

            z_shift_adjustment = align_zshift + starting_zshift;
        }
    } else {
        ctf_output = ali_name.clone();
        ctf_input = ali_name.clone();
        ctf_pix_size = pix_x_raw;
        tilt_use_gpu = match option_value(&tilt_lines, "UseGPU", 1, true, 1, None, None) {
            Some(OptionValue::Integers(values)) => values[0],
            _ => -1,
        };
    }

    // Restore the previous aligned stack
    if new_ali_binning > 0 {
        cleanup_files(std::slice::from_ref(&ali_name));
        if Path::new(&format!("{ali_name}~")).exists() {
            let _ = std::fs::rename(format!("{ali_name}~"), &ali_name);
        }
    }

    let (ali_root, ali_ext) = os_path_splitext(&ctf_output);
    let override_gpu = pip_get_integer("WhenToUseGPU", -1).unwrap_or(-1);
    if pip_get_err_no() == 0 {
        if override_gpu == 0 {
            ctf_use_gpu = -1;
            tilt_use_gpu = -1;
        } else if override_gpu == 1 {
            ctf_use_gpu = 0;
            tilt_use_gpu = 0;
        } else if override_gpu > 1 {
            ctf_use_gpu = 0;
            tilt_use_gpu = -1;
        }
    }

    let (mut com_root, ext) = os_path_splitext(&com_file);
    if com_ext.is_empty() {
        com_ext = ext;
    }
    com_root.push_str("-sub");
    let mut root_with_dir = com_root.clone();
    let mut opt_com_dir = String::new();
    if !com_dir.is_empty() {
        root_with_dir = format!("{com_dir}/{com_root}");
        opt_com_dir = format!("-dir \"{com_dir}\"");
    }
    clean_chunk_files(&root_with_dir, false);
    let mut com_num = 1;
    let mut temp_stacks = String::new();

    // Handle setting all the items for "aligned stack" when using raw input
    // The ...ali values from the above calls are actually raw stack values
    if use_raw_input {
        check_for_distortion(axis_let.as_deref(), 1, progname);
        ali_binning = raw_binning;
        if raw_pix_size == 0. {
            raw_pix_size = get_fallback_raw_pixel(ctf_pix_size, axis_let.as_deref(), &com_ext);
        }
        let _header_pix_size = ctf_pix_size * raw_binning as f64;
        if transpose_xy {
            (nx_ali, ny_ali) = (ny_ali, nx_ali);
            orig_x_ali += pix_x_raw * (nx_ali - nx_raw) as f64 / 2.;
            orig_y_ali += pix_y_raw * (ny_ali - ny_raw) as f64 / 2.;
        }
        nx_ali = py_int_floordiv(nx_ali, raw_binning);
        ny_ali = py_int_floordiv(ny_ali, raw_binning);
        pix_x_ali *= raw_binning as f64;
        pix_y_ali *= raw_binning as f64;
        ctf_pix_size = raw_pix_size * raw_binning as f64;
    }

    // Make command file for making new aligned stack
    // adjust pixel size from com file for CTF correction by change in binning
    if new_ali_binning > 0 {
        if DEBUG != 0 {
            let value_of = |option: &str| match option_value(
                &newst_lines,
                option,
                STRING_VALUE,
                false,
                0,
                None,
                None,
            ) {
                Some(OptionValue::String(value)) => value,
                _ => "None".to_owned(),
            };
            prnstr(
                &format!(
                    "newst {} -> {}",
                    value_of("InputFile"),
                    value_of("OutputFile")
                ),
                "\n",
                false,
            );
        }
        let _ = write_text_file(
            &make_next_com_name(&mut com_num, &root_with_dir, true),
            &newst_lines,
            false,
        );
        ctf_pix_size = (ctf_pix_size * new_ali_binning as f64) / old_ali_bin as f64;
    }

    // If using raw input, set up input name and optional binning com file
    if use_raw_input {
        ctf_input = stack_name.clone();
        if raw_binning > 1 {
            ctf_input = format!("{ali_root}_red{raw_binning}_tmp{ali_ext}");
            temp_stacks.push_str(&format!(" {ctf_input}"));
            let newst_lines = vec![format!(
                "$newstack -ftreduce {raw_binning} {stack_name} {ctf_input}"
            )];
            if DEBUG != 0 {
                prnstr(
                    &format!("newstack reduce {stack_name} -> {ctf_input}"),
                    "\n",
                    false,
                );
            }
            let _ = write_text_file(
                &make_next_com_name(&mut com_num, &root_with_dir, true),
                &newst_lines,
                false,
            );
        }
    }

    // If filtering, do it in an initial command file
    if do_filter {
        let filt_input = ctf_input.clone();
        ctf_input = format!("{ali_root}_filt_tmp{ali_ext}");
        temp_stacks.push_str(&format!(" {ctf_input}"));
        let mtfsed = vec![
            sed_modify("InputFile", &filt_input, '|'),
            sed_modify("OutputFile", &ctf_input, '|'),
            sed_modify("PixelSize", &py_str_float(ctf_pix_size), '|'),
        ];
        let mtf_mod = lines_of(pysed(
            &mtfsed,
            PysedSrc::Lines(&mtf_lines),
            None,
            false,
            '|',
            false,
        ));
        if DEBUG != 0 {
            prnstr(
                &format!("mtffilter {filt_input} -> {ctf_input}"),
                "\n",
                false,
            );
        }
        let _ = write_text_file(
            &make_next_com_name(&mut com_num, &root_with_dir, true),
            &mtf_mod,
            false,
        );
    }

    if !do_ctf {
        ctf_output = ctf_input.clone();
    }

    let mut erase_mod: Vec<String> = Vec::new();
    let mut erase_name = String::new();
    if do_erase {
        let erase_lines = read_text_file(
            &erase_com_file,
            Some("Gold erasing command file"),
            false,
            None,
        )
        .unwrap_or_default();
        erase_name = format!("{ali_root}_erase_tmp{ali_ext}");
        let mut erase_sed = vec![
            sed_modify("InputFile", &ctf_output, '|'),
            sed_modify("OutputFile", &erase_name, '|'),
        ];
        if use_raw_input {
            let raw_erase_fid = back_transform_erase_model(
                &erase_lines,
                &ali_xform_file,
                10. * raw_pix_size,
                &erase_com_file,
                progname,
            );
            erase_sed.push(sed_modify("ModelFile", &raw_erase_fid, '|'));
            temp_stacks.push_str(&format!(" {raw_erase_fid}"));
        }

        erase_mod = lines_of(pysed(
            &erase_sed,
            PysedSrc::Lines(&erase_lines),
            None,
            false,
            '|',
            false,
        ));
        erase_mod.push(format!("$b3drename \"{erase_name}\" \"{ctf_output}\""));
    }

    let mut thickness = y_size;
    let mut num_slices = py_int_floordiv(z_size, ali_binning);
    if reorient != 0 {
        thickness = z_size;
        num_slices = py_int_floordiv(y_size, ali_binning);
    }

    // Get values needed for making volume stack and model
    let x_center = py_int_floordiv(x_size, ali_binning) as f64 / 2.;
    let y_center = py_int_floordiv(y_size, ali_binning) as f64 / 2.;
    let z_final = py_int_floordiv(z_size, ali_binning);
    let z_center = z_final as f64 / 2. - 0.5;

    // ASSUMING CENTER ALIGNED STACK, could use origins to overcome this
    if transpose_xy {
        (nx_raw, ny_raw) = (ny_raw, nx_raw);
    }
    let sssx = py_int_floordiv(nx_raw - nx_ali * ali_binning, 2);
    let sssy = py_int_floordiv(ny_raw - ny_ali * ali_binning, 2);

    let mut sedcom_base = vec![
        sed_modify("IMAGEBINNED", &ali_binning.to_string(), '|'),
        sed_modify("FULLIMAGE", &format!("{nx_raw} {ny_raw}"), '|'),
        sed_modify("SUBSETSTART", &format!("{sssx} {sssy}"), '|'),
        sed_modify("THICKNESS", &thickness.to_string(), '|'),
        "|savework|d".to_owned(),
    ];
    sedcom_base.extend(sed_del_and_add(
        "WIDTH",
        &x_size.to_string(),
        "THICKNESS",
        '|',
    ));
    sedcom_base.extend(sed_del_and_add("XSubsetLoadRatio", "1.2", "THICKNESS", '|'));

    if ctf_output != ali_name {
        sedcom_base.push(sed_modify("InputProjections", &ctf_output, '|'));
    }
    if use_raw_input {
        sedcom_base.extend(sed_del_and_add("UseUnalignedImages", "1", "THICKNESS", '|'));
        sedcom_base.extend(sed_del_and_add(
            "AlignTransformFile",
            &ali_xform_file,
            "THICKNESS",
            '|',
        ));
        sedcom_base.push(sed_modify("IMAGEBINNED", &raw_binning.to_string(), '|'));
    }

    if !type_ext.is_empty() && type_ext != "mrc" {
        sedcom_base.push("|IMOD_OUTPUT_FORMAT|d".to_owned());
    }

    let mut max_slices: i64 = 0;
    if do_ctf {
        // Work out GPU for CTF which can be independent: if it is on it stays on
        // Do CTF in parallel if multiple procs explicitly entered and tilt is using GPUs,
        // or if not using a GPU for CTF and no number was entered - assume 8 in thta case
        max_slices = 0;
        let mut proc_for_ctf = num_proc;
        if num_proc == 0 {
            proc_for_ctf = 8;
        }
        if num_proc > 1 || (num_proc == 0 && ctf_use_gpu < 0) {
            let num_chunks = 3 * proc_for_ctf;
            max_slices = 1.max(py_int_floordiv(nz_ali + num_chunks - 1, num_chunks));
        }
    }

    // Get format string for particle names to get equal digits on all
    let mut num_dec = 1;
    let mut num_pts = point_list.len() as i64;
    let mut del_match = "-[0-9]".to_owned();
    while num_pts > 9 {
        num_dec += 1;
        num_pts = py_int_floordiv(num_pts, 10);
        del_match.push_str("[0-9]");
    }
    let num_pts = point_list.len();

    // Loop through all the points, getting their command file text and z shifts
    let mut num_pt_tot: i64 = 0;
    let mut min_zshift = 1.0e10_f64;
    let mut max_zshift = -1.0e10_f64;
    let mut shift_list: Vec<f64> = Vec::new();
    let mut sed_list: Vec<Vec<String>> = Vec::new();
    let mut vol_names: Vec<String> = Vec::new();
    for pt_num in 0..num_pts {
        let mut num_use = num_pt_tot + 1;
        if skip_vol_numbers {
            num_use = pt_num as i64 + 1;
        }

        let mut chunk_base = format!("{root_name}-{num_use:0num_dec$}");

        // Use a forward slash so output is stable and tests work with Windows python
        if !out_dir.is_empty() {
            chunk_base = format!("{out_dir}/{chunk_base}");
        }
        let mut chunk_name = format!("{chunk_base}.mrc");
        if reorient != 0 {
            chunk_name = format!("{chunk_base}.tmp");
        }
        let mut sedcom = sedcom_base.clone();
        sedcom.push(sed_modify("OutputFile", &chunk_name, '|'));
        let point = &mut point_list[pt_num];

        // Convert points from a point list by the header transformation to match scaled
        // values that came in from model conversion
        if model_file.is_empty() {
            point[0] = point[0] * pix_x_vol - orig_x_vol;
            point[1] = point[1] * pix_y_vol - orig_y_vol;
            point[2] = point[2] * pix_z_vol - orig_z_vol;
        }
        let point = *point;

        // Now need to get slice range and X/Z shifts.  X is easy and invariant
        let x_in_ali = (point[0] + orig_x_ali) / pix_x_ali;
        let x_shift = ali_binning as f64 * (nx_ali as f64 / 2. - x_in_ali);

        let (y_in_ali, z_shift);
        // For no reorientation, Y comes from Z, Y from Y; z shift is negative of coordinate
        if reorient == 0 {
            y_in_ali = (point[2] + orig_y_ali) / pix_y_ali;
            z_shift = -(ali_binning as f64) * (point[1]) / pix_x_ali;

        // For rotation, Y comes from Y, Z from inversion of Y
        } else if reorient < 0 {
            y_in_ali = (point[1] + orig_y_ali) / pix_y_ali;
            z_shift = ali_binning as f64 * (point[2]) / pix_x_ali;

        // For flip, the origins were not swapped in the header, so undo the origin that was
        // applied and adjust by origin that should have been applied
        } else {
            y_in_ali = (point[1] + orig_y_vol - orig_z_vol - orig_y_ali) / pix_y_ali;
            z_shift = -(ali_binning as f64) * (point[2] + orig_z_vol - orig_y_vol) / pix_x_ali;
        }

        // Get the slice range, skip if too far out of range
        let mut slice_start = py_round(y_in_ali - num_slices as f64 / 2.) as i64 * ali_binning;
        let mut slice_end = slice_start + num_slices * ali_binning - 1;
        if y_in_ali < num_slices as f64 / 6. || y_in_ali > ny_ali as f64 - num_slices as f64 / 6. {
            prnstr(
                &format!(
                    "WARNING: {progname} - Point # {} is skipped; it requires too many Y slices outside the reconstructable range for this aligned stack",
                    pt_num + 1
                ),
                "\n",
                false,
            );
            continue;
        }

        // And set up for blank slices if partly out of the range
        // Get the binned slice range that Tilt will use.  It uses slices numbered from 1
        // and rounds up when binning, but (sl0 + 1 + bin - 1) / bin = sl0 / bin + 1
        // numbered from 1, which is oddly just sl0 / bin numbered from 0.
        let mut newst_range = String::new();
        let bin_slice_start = py_int_floordiv(slice_start, ali_binning);
        let bin_slice_end = py_int_floordiv(slice_end, ali_binning);
        if slice_start < 0 || bin_slice_end >= ny_ali {
            let num_blank;
            if slice_start < 0 {
                num_blank = num_slices - (bin_slice_end + 1);
                newst_range = format!("{}-{bin_slice_end}", -num_blank);
                slice_start = 0;
            } else {
                num_blank = num_slices - (ny_ali - bin_slice_start);
                newst_range = format!("0-{}", num_slices - 1);
                slice_end = ny_ali * ali_binning - 1;
            }
            prnstr(
                &format!(
                    "WARNING: {progname} - Point # {} is near the edge of the aligned stack in Y and requires {num_blank} blank slices",
                    pt_num + 1
                ),
                "\n",
                false,
            );
        }

        // Finish the sed com, process the lines
        sedcom.extend(sed_del_and_add(
            "SHIFT",
            &format!(
                "{} {}",
                py_str_float(py_round_ndigits(x_shift, XZ_SHIFT_DECIMALS)),
                py_str_float(py_round_ndigits(z_shift, XZ_SHIFT_DECIMALS))
            ),
            "THICKNESS",
            '|',
        ));
        sedcom.extend(sed_del_and_add(
            "SLICE",
            &format!("{slice_start} {slice_end}"),
            "THICKNESS",
            '|',
        ));
        if override_gpu >= 0 {
            sedcom.extend(sed_del_and_add(
                "UseGPU",
                &tilt_use_gpu.to_string(),
                "THICKNESS",
                '|',
            ));
        }
        let mut sed_lines = lines_of(pysed(
            &sedcom,
            PysedSrc::Lines(&tilt_lines),
            None,
            false,
            '|',
            false,
        ));

        // Add blank slices if needed
        if !newst_range.is_empty() {
            sed_lines.extend([
                format!(
                    "$newstack -blank -sec {newst_range} \"{chunk_name}\" \"{chunk_base}.tmp2\""
                ),
                format!("$b3dremove \"{chunk_name}\""),
            ]);
            chunk_name = format!("{chunk_base}.tmp2");
        }

        // Add final reorientation
        if reorient != 0 {
            let mut oper = "rotx";
            if reorient > 0 {
                oper = "flipyz";
            }
            sed_lines.extend([
                format!("$clip {oper} \"{chunk_name}\" \"{chunk_base}.mrc\""),
                format!("$b3dremove \"{chunk_name}\""),
            ]);
        }

        // Maintain min/max, save the lines and the z shift
        // (Python's `min`/`max` keep the first argument unless the second is
        // strictly less/greater)
        if z_shift < min_zshift {
            min_zshift = z_shift;
        }
        if z_shift + 0.01 > max_zshift {
            max_zshift = z_shift + 0.01;
        }
        sed_list.push(sed_lines);
        shift_list.push(z_shift);
        num_pt_tot += 1;
        if make_vol_stacks != 0 {
            vol_names.push(format!("{chunk_base}.mrc"));
        }
    }

    if num_pt_tot == 0 {
        exit_error("There are no points to do because all points are being skipped");
    }

    // if doing CTF, figure out the range of Z etc
    let del_zpixels;
    let num_in_level: Vec<i64>;
    let pt_ind_list: Vec<Vec<usize>>;
    let mut defocus_tol: i64 = 0;
    if do_ctf {
        let ub_pix_size = ctf_pix_size / ali_binning as f64;
        let full_zpixels = max_zshift - min_zshift;
        let full_znm = full_zpixels * ub_pix_size;
        if max_zextent > 0. {
            num_zlevels = (full_znm / max_zextent).ceil() as i64;
        }
        if num_zlevels < 2 && max_zextent > 0. {
            prnstr(
                &format!(
                    "WARNING: {progname} - The Z extent of {} nm will result in only one Z level because the range of center positions is {} nm",
                    py_fixed(max_zextent, 0, 0),
                    py_fixed(full_znm, 0, 0)
                ),
                "\n",
                false,
            );
        }

        let z_extent_nm = py_true_div(full_znm, num_zlevels as f64);
        prnstr(
            &format!(
                "CTF corrections will be computed at {num_zlevels} levels that are {} nm thick",
                py_fixed(z_extent_nm, 0, 0)
            ),
            "\n",
            false,
        );
        del_zpixels = z_extent_nm / ub_pix_size;
        let mut levels = Vec::new();
        let mut indices = Vec::new();
        let invert_opt = option_value(
            &ctf_lines,
            "InvertTiltAngles",
            BOOL_VALUE,
            false,
            0,
            None,
            None,
        );
        if invert_zoffsets < 0 && invert_opt == Some(OptionValue::Boolean(true)) {
            invert_zoffsets = 1;
        }
        let z_nm_int = py_round(z_extent_nm) as i64;
        defocus_tol = match option_value(&ctf_lines, "DefocusTol", INT_VALUE, false, 1, None, None)
        {
            Some(OptionValue::Integers(values)) if values[0] != 0 => {
                (values[0] as i64).min(z_nm_int)
            }
            _ => z_nm_int,
        };

        // Test for X tilt consistency and if it matters
        check_xtilt_ctf_vs_rec(
            xaxis_tilt,
            ctf_xtilt,
            WARN_XTILT_CRIT,
            0.1 * ny_ali as f64 * pix_x_ali,
            z_extent_nm,
            "level",
            progname,
        );

        // Find the number in each level
        for level in 0..num_zlevels {
            let shift_low = min_zshift + level as f64 * del_zpixels;
            let shift_high = shift_low + del_zpixels;
            let mut num_tmp = 0;
            let mut level_inds = Vec::new();
            for ind in 0..num_pt_tot as usize {
                if shift_list[ind] >= shift_low && shift_list[ind] < shift_high {
                    num_tmp += 1;
                    level_inds.push(ind);
                }
            }
            levels.push(num_tmp);
            indices.push(level_inds);
        }
        num_in_level = levels;
        pt_ind_list = indices;
    } else {
        num_zlevels = 1;
        del_zpixels = max_zshift - min_zshift;
        num_in_level = vec![num_pt_tot];
        pt_ind_list = vec![(0..num_pt_tot as usize).collect()];
    }

    // No processors entered and using GPU for tilt, assume 1
    if num_proc == 0 && tilt_use_gpu >= 0 {
        num_proc = 1;
    }

    // Loop on levels if any
    let mut ctf_com_out = String::new();
    for level in 0..num_zlevels as usize {
        let num_pts_lvl = num_in_level[level];
        if num_pts_lvl == 0 {
            continue;
        }
        let mut max_chunk_lvl = MAX_CHUNKS;
        let mut num_optimal: i64 = 1000;
        if do_ctf {
            max_chunk_lvl =
                py_round((0.9 * MAX_CHUNKS as f64 * num_pts_lvl as f64) / num_pt_tot as f64) as i64;
            num_optimal = py_int_floordiv(num_optimal * num_pts_lvl, num_pt_tot);
        }

        let mut num_chunks: i64 = 0;
        if num_proc > 1 || zero_proc_entered {
            // If # of processors entered, try for a large # of chunks per processor but lower it
            // to give fewer than 1000 chunks; in any case limit chunks to maximum and to # pts
            let mut num_chunk_per_proc = MAX_CHUNK_PER_PROC;
            while num_chunk_per_proc >= MIN_CHUNK_PER_PROC {
                num_chunks = num_pts_lvl
                    .min(num_chunk_per_proc * num_proc)
                    .min(max_chunk_lvl);
                if num_chunks < num_optimal {
                    break;
                }
                num_chunk_per_proc -= 1;
            }
        } else {
            // Otherwise base it on default or entered # of runs per chunks; but it must be
            // raised if that gives too many chunks, or lower it if it is too few
            let min_runs = py_int_floordiv(num_pts_lvl, max_chunk_lvl) + 1;
            let min_chunks_lvl = num_pts_lvl.min(MIN_CHUNKS);
            let runs_per_chunk_lvl = num_runs_per_chunk.max(min_runs);
            num_chunks = min_chunks_lvl.max(py_int_floordiv(
                num_pts_lvl + runs_per_chunk_lvl - 1,
                runs_per_chunk_lvl,
            ));
        }

        // `BUGS.md` (subtomosetup `-proc 0`), fixed in translation: with
        // `-proc 0` entered and no GPU for Tilt, the loop above gives zero
        // chunks and the division below raises a ZeroDivisionError; this is
        // reported as an error instead.
        if num_chunks == 0 {
            exit_error(
                "The runs cannot be divided into chunks for 0 processors; enter a positive number with -proc",
            );
        }
        let runs_per_chunk_lvl = py_int_floordiv(num_pts_lvl, num_chunks);

        if do_ctf {
            let mut offset_pix =
                ((level as f64 + 0.5) * del_zpixels + min_zshift + z_shift_adjustment)
                    / ali_binning as f64;
            if invert_zoffsets > 0 {
                offset_pix = -offset_pix;
            }
            let mut ctfsed = vec![sed_modify("InputStack", &ctf_input, '|')];
            ctfsed.extend(sed_del_and_add(
                "OffsetInZ",
                &py_str_float(py_round_ndigits(offset_pix, 2)),
                "DefocusFile",
                '|',
            ));
            ctfsed.extend(sed_del_and_add(
                "UseGPU",
                &ctf_use_gpu.to_string(),
                "DefocusFile",
                '|',
            ));
            ctfsed.extend(sed_del_and_add(
                "DefocusTol",
                &defocus_tol.to_string(),
                "DefocusFile",
                '|',
            ));

            if use_raw_input || new_ali_binning > 0 {
                ctfsed.push(sed_modify("PixelSize", &py_str_float(ctf_pix_size), '|'));
            }

            if use_raw_input {
                ctfsed.push("|TransformFile|d".to_owned());
                ctfsed.extend(sed_del_and_add(
                    "XAxisTilt",
                    &py_str_float(xaxis_tilt),
                    "DefocusFile",
                    '|',
                ));
                ctfsed.extend(sed_del_and_add(
                    "AxisAngle",
                    &py_str_float(axis_angle),
                    "DefocusFile",
                    '|',
                ));
            }

            let ctf_mod = lines_of(pysed(
                &ctfsed,
                PysedSrc::Lines(&ctf_lines),
                None,
                false,
                '|',
                false,
            ));
            if DEBUG != 0 {
                let output = match option_value(
                    &ctf_mod,
                    "OutputFileName",
                    STRING_VALUE,
                    false,
                    0,
                    None,
                    None,
                ) {
                    Some(OptionValue::String(value)) => value,
                    _ => "None".to_owned(),
                };
                prnstr(
                    &format!("ctfcorrection  {ctf_input} -> {output}"),
                    "\n",
                    false,
                );
            }

            if max_slices != 0 {
                ctf_com_out = "ctfcorrection.tmp.com".to_owned();
                let _ = write_text_file(&ctf_com_out, &ctf_mod, false);
                let split_lines = match run_cmd(
                    &fmtstr(
                        "splitcorrection -i {} -o -m {} -b {} -uni {} {} -r {} {}",
                        &[
                            com_num.to_string(),
                            max_slices.to_string(),
                            bound_pixels.to_string(),
                            split_corr_size.clone(),
                            opt_com_dir.clone(),
                            com_root.clone(),
                            ctf_com_out.clone(),
                        ],
                    ),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => exit_from_imod_error(progname),
                };
                let num_added = find_split_com_number(&split_lines, "output of splitcorrection");
                com_num += num_added;
            } else {
                ctf_com_out = make_next_com_name(&mut com_num, &root_with_dir, true);
                let _ = write_text_file(&ctf_com_out, &ctf_mod, false);
            }
        }

        if do_erase {
            if DEBUG != 0 {
                prnstr(
                    &format!("eraser  {ctf_output} <-> {erase_name}"),
                    "\n",
                    false,
                );
            }
            erase_name = make_next_com_name(&mut com_num, &root_with_dir, true);
            let _ = write_text_file(&erase_name, &erase_mod, false);
        }

        // Loop on chunks
        let mut ind_in_level = 0;
        for chunk in 0..num_chunks {
            let mut num_in_chunk = runs_per_chunk_lvl;
            if chunk < py_int_mod(num_pts_lvl, num_chunks) {
                num_in_chunk += 1;
            }

            let com_name = make_next_com_name(&mut com_num, &root_with_dir, false);
            let mut chunk_lines: Vec<String> = Vec::new();
            if !type_ext.is_empty() && type_ext != "mrc" {
                chunk_lines = vec!["$setenv IMOD_OUTPUT_FORMAT MRC".to_owned()];
            }
            for _ind_in_chunk in 0..num_in_chunk {
                let pt_ind = pt_ind_list[level][ind_in_level];
                if DEBUG != 0 {
                    let value_of = |option: &str| match option_value(
                        &sed_list[pt_ind],
                        option,
                        STRING_VALUE,
                        false,
                        0,
                        None,
                        None,
                    ) {
                        Some(OptionValue::String(value)) => value,
                        _ => "None".to_owned(),
                    };
                    prnstr(
                        &format!(
                            "tilt  {} -> {}",
                            value_of("InputProjections"),
                            value_of("OutputFile")
                        ),
                        "\n",
                        false,
                    );
                }
                ind_in_level += 1;

                // Add lines to chunk
                chunk_lines.extend(sed_list[pt_ind].iter().cloned());
            }

            // Write the file
            let _ = write_text_file(&com_name, &chunk_lines, false);
        }

        if do_ctf {
            let com_name = make_next_com_name(&mut com_num, &root_with_dir, true);
            let _ = write_text_file(&com_name, &[format!("$b3dremove {ctf_output}")], false);
        }
    }

    if make_vol_stacks != 0 {
        // Get number of stacks and format for numbering
        let num_stacks = py_int_floordiv(num_pt_tot + make_vol_stacks - 1, make_vol_stacks);
        let mut num_dec = 1;
        let mut num_pts = num_stacks;
        while num_pts > 9 {
            num_dec += 1;
            num_pts = py_int_floordiv(num_pts, 10);
        }

        // Make one sync file; loop on stacks
        let com_name = make_next_com_name(&mut com_num, &root_with_dir, true);
        let mut com_lines: Vec<String> = Vec::new();
        for group in 0..num_stacks {
            // Get range of subvols and name for output
            let (start, end) =
                balanced_group_limits(num_pt_tot as i32, num_stacks as i32, group as i32);
            let mut stack_root = format!("{root_name}-vol{:0num_dec$}", group + 1);
            if !out_dir.is_empty() {
                stack_root = format!("{out_dir}/{stack_root}");
            }
            com_lines.extend([
                "$setenv IMOD_OUTPUT_FORMAT MRC".to_owned(),
                "$newstack -StandardInput".to_owned(),
                format!("OutputFile {stack_root}.mrc"),
            ]);

            // Add the input file names and accumulate point lines
            let mut pt_lines: Vec<String> = Vec::new();
            for ind in start..end + 1 {
                com_lines.push(format!("InputFile {}", vol_names[ind as usize]));
                let cont_time = ind + 1 - start;
                pt_lines.push(format!(
                    "1 {cont_time} {} {} {} {cont_time}",
                    py_str_float(x_center),
                    py_str_float(y_center),
                    py_str_float(z_center)
                ));
            }

            // Write point file, add commands to convert to model, and remove point file
            let _ = write_text_file(&format!("{stack_root}.pt"), &pt_lines, false);
            com_lines.push(format!(
                "$alterheader -volstack {z_final} \"{stack_root}.mrc\""
            ));
            com_lines.push(format!(
                "$point2model -scat -time -sphere 3 -image \"{stack_root}.mrc\" \"{stack_root}.pt\" \"{stack_root}.mod\""
            ));
            com_lines.push(format!("$b3dremove \"{stack_root}.pt\""));
        }

        // After all runs, remove the subvols
        let mut clean_root = format!("{root_name}{del_match}");
        if !out_dir.is_empty() {
            clean_root = format!("{out_dir}/{clean_root}");
        }
        com_lines.push(format!("$b3dremove -g \"{clean_root}*.mrc\""));
        let _ = write_text_file(&com_name, &com_lines, false);
    }

    // Write finish file
    let mut finlines = vec![format!(
        "$b3dremove -g \"{root_with_dir}-[0-9][0-9][0-9]*.com*\" \"{root_with_dir}-[0-9][0-9][0-9]*.log*\" \"{root_with_dir}-finish*.com*\""
    )];
    if !temp_stacks.is_empty() {
        finlines.push(format!("$b3dremove {temp_stacks}"));
    }
    if do_ctf && max_slices > 0 {
        finlines.push(format!("$b3dremove {ctf_com_out}"));
    }
    let _ = write_text_file(&format!("{root_with_dir}-finish.com"), &finlines, false);
    prnstr(
        &format!("Created {com_num} command files for {num_pt_tot} subtomograms; run them with:"),
        "\n",
        false,
    );
    prnstr(
        &format!("    \"subm {root_with_dir}*.com\"   or   \"processchunks ... {root_with_dir}\""),
        "\n",
        false,
    );
    if do_ctf && max_slices != 0 {
        cleanup_files(&[ctf_com_out]);
    }
    done(0)
}
