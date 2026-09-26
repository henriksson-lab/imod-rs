//! Translation of `IMOD/pysrc/matchorwarp`.
//!
//! A Python command script: its two functions are [`usable_error`] and
//! [`run_findwarp`], and its top level is [`matchorwarp`], translated
//! statement by statement.  The module globals those functions read or set
//! (`usableMat`, `meanResids`, `maxResids`, `meanIncreased`, `warplimit`,
//! `findbase`, `volbase`, the exclusion limits, `extent`, `clipsize`,
//! `objname`, `trial`, `iterInd`, `iterStr`, the stopping criteria and the
//! `matFile` names) are the fields of [`Globals`].
//!
//! `refinematch`, `matchvol`, `warpvol` and `patch2imod` run through
//! `runcmd`, in process once they are this crate's commands (none of their
//! output is parsed: refinematch's is only its exit status).  `findwarp`,
//! whose `[FWP1]`/`[FWP2]` lines the script parsed, is a direct call since
//! 2026-09-26 that returns those values (`findwarp_recording`); the
//! patchcorr command file (`vmstopy -x -q` in the source) through the
//! in-process runner ([`crate::imod::comrun::run_com_as_command`]).  PIP
//! floats are doubles and `'{}'.format`/`str` of one is its `repr`
//! ([`py_str_float`]).

use super::batchruntomo::py_str_float;
use super::imodpy::{
    ImodpyError, OptionValue, add_imod_bin_ignore_sighup, call_own_program, cleanup_files,
    complete_and_check_com_file, exit_from_imod_error, get_err_strings, get_last_exit_status,
    get_mrc_size, option_value, os_path_splitext, prnstr, read_text_file, run_cmd,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_string, pip_get_three_floats, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_modify};
use crate::imod::flib::model::findwarp::{FindwarpResult, findwarp_recording};
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The module globals (see the module comment).
#[derive(Default)]
pub struct Globals {
    /// `usableMat`
    pub usable_mat: bool,
    /// `meanResids`
    pub mean_resids: Vec<f64>,
    /// `maxResids`
    pub max_resids: Vec<f64>,
    /// `meanIncreased`
    pub mean_increased: bool,
    /// `warplimit`
    pub warplimit: String,
    /// `findbase`
    pub findbase: Vec<String>,
    /// `volbase`
    pub volbase: Vec<String>,
    /// `xlower`, `xupper`, `ylower`, `yupper`, `zlower`, `zupper`
    pub xlower: i32,
    pub xupper: i32,
    pub ylower: i32,
    pub yupper: i32,
    pub zlower: i32,
    pub zupper: i32,
    /// `extent`
    pub extent: i32,
    /// `clipsize`
    pub clipsize: String,
    /// `objname`
    pub objname: String,
    /// `trial`
    pub trial: i32,
    /// `iterInd`
    pub iter_ind: usize,
    /// `iterStr`
    pub iter_str: String,
    /// `countingMeanCrit`, `meanStopCrit`, `maxStopCrit`
    pub counting_mean_crit: f64,
    pub mean_stop_crit: f64,
    pub max_stop_crit: f64,
    /// `matRoot`, `matExt`, `matFile`
    pub mat_root: String,
    pub mat_ext: String,
    pub mat_file: String,
}

/// Matches `usableError` (`matchorwarp:15`): exit with a message that the
/// current warp file is usable if that is the case.
pub fn usable_error(g: &Globals, progname: &str, action: &str, iter_num: usize) -> ! {
    if !g.usable_mat {
        exit_from_imod_error(progname);
    }
    let err_strings = get_err_strings();
    for l in &err_strings {
        prnstr(l.trim(), "\n", false);
    }
    exit_error(&format!(
        "{action} failed on iteration {iter_num} but match file from last iteration is still usable"
    ))
}

/// Matches `runFindwarp` (`matchorwarp:27`): run Findwarp, setting parameters
/// and extracting residuals from the results, and making decision on stopping
/// when the error increases.  `Err` is an `ImodpyError` the source lets
/// propagate to its caller.
pub fn run_findwarp(
    g: &mut Globals,
    solvefile: &str,
    patchfile: &str,
    warpfile: &str,
    residualfile: &str,
    vectormodel: &str,
    iteration: usize,
) -> Result<i32, ImodpyError> {
    let mut use_limit = g.warplimit.clone();
    if iteration != 0 {
        let mut lim_split: Vec<String> = regex::Regex::new(r"[,\s]+")
            .unwrap()
            .split(&g.warplimit)
            .map(str::to_owned)
            .collect();
        let last = lim_split.last().cloned().unwrap_or_default();
        let Ok(highest) = last.trim().parse::<f64>() else {
            let _ = std::io::stdout().flush();
            eprintln!("Traceback (most recent call last):");
            eprintln!("ValueError: could not convert string to float: '{last}'");
            std::process::exit(1);
        };
        lim_split.push(py_str_float(highest * 1.25));
        lim_split.push(py_str_float(highest * 1.5));
        use_limit = lim_split.join(",");
    }

    let mut comlines = g.findbase.clone();
    comlines.extend([
        format!("TargetMeanResidual {use_limit}"),
        format!("InitialTransformFile {solvefile}"),
        format!("OutputFile {warpfile}"),
        format!("PatchFile {patchfile}"),
        "ZeroExitCodeIfTooHigh 1".to_owned(),
    ]);
    if !residualfile.is_empty() {
        comlines.push(format!("ResidualPatchOutput {residualfile}"));
    }
    if g.xlower != 0 || g.xupper != 0 {
        comlines.push(format!("XSkipLeftAndRight {},{}", g.xlower, g.xupper));
    }
    if g.ylower != 0 || g.yupper != 0 {
        comlines.push(format!("YSkipLowerAndUpper {},{}", g.ylower, g.yupper));
    }
    if g.zlower != 0 || g.zupper != 0 {
        comlines.push(format!("ZSkipLowerAndUpper {},{}", g.zlower, g.zupper));
    }
    if g.extent != 0 {
        comlines.push(format!("MinExtentToFit {}", g.extent));
    }

    let savestat;
    let mut mean: f64 = 0.;
    let mut maxr: f64 = 0.;
    // Direct call (owner rule, 2026-09-26): findwarp records the values of
    // its `[FWP1]`/`[FWP2]` reports into a `FindwarpResult`, used here in place
    // of parsing those lines (`matchorwarp:70-86`).  Findwarp still reads its
    // options through PIP from `comlines`, and its report is echoed as before.
    let sink = std::sync::Arc::new(std::sync::Mutex::new(FindwarpResult::default()));
    let recorder = std::sync::Arc::clone(&sink);
    match call_own_program(
        "findwarp -StandardInput",
        &["findwarp", "-StandardInput"],
        Some(&comlines),
        true,
        move || findwarp_recording(recorder),
    ) {
        Ok((_, find_out)) => {
            let mut stat = 0;
            for line in &find_out {
                prnstr(line.trim_end(), "\n", false);
            }
            let result = sink.lock().expect("findwarp result").clone();
            // The script took the first two numbers it could `float()` from
            // each `[FWP1]` line, `Mean residual has an average of{f8.3} and a
            // maximum of{f8.3}`: so each value is rounded as `f8.3` prints it,
            // and a field printed without a leading blank (8 characters or
            // asterisks) ran into the word `of` and was not a number.
            for fit in &result.fits {
                for value in [fit.dev_mean_avg, fit.dev_mean_max] {
                    let field = format_f(f64::from(value), 8, 3);
                    if !field.starts_with(' ') {
                        continue;
                    }
                    if let Ok(val) = field.trim().parse::<f64>() {
                        if mean == 0. {
                            mean = val;
                        } else if maxr == 0. {
                            maxr = val;
                            break;
                        }
                    }
                }
            }
            if result.failed_above_target.is_some() {
                stat = 2;
            }
            savestat = stat;

            g.mean_resids.push(mean);
            g.max_resids.push(maxr);
        }
        Err(_) => {
            savestat = get_last_exit_status();
            // The source's error strings keep their line endings, so each
            // `prnstr` is followed by an empty line
            for line in get_err_strings() {
                prnstr(&format!("{line}\n"), "\n", false);
            }
        }
    }

    if savestat != 0 && savestat != 2 {
        if iteration > 1 {
            return Ok(savestat);
        }
        let _ = std::io::stdout().flush();
        std::process::exit(1);
    }
    if !vectormodel.is_empty() {
        prnstr(" ", "\n", false);
        run_cmd(
            &format!(
                "patch2imod {} -n \"{}\" \"{residualfile}\" \"{vectormodel}\"",
                g.clipsize, g.objname
            ),
            None,
            None,
            None,
            &[],
        )?;
        prnstr(&format!("MATCHORWARP: Created {vectormodel}"), "\n", false);
    }

    if savestat == 0 && iteration != 0 {
        let mut stop = false;
        let mut mess = String::new();
        let iter_ind = g.iter_ind;
        let index_error = || -> ! {
            let _ = std::io::stdout().flush();
            eprintln!("Traceback (most recent call last):");
            eprintln!("IndexError: list index out of range");
            std::process::exit(1)
        };
        if iter_ind >= g.mean_resids.len() || iter_ind >= g.max_resids.len() || iter_ind < 1 {
            index_error();
        }
        let mean_resids = &g.mean_resids;
        let max_resids = &g.max_resids;
        if mean_resids[0] != 0. && (mean_resids[iter_ind] > mean_resids[0] * g.mean_stop_crit) {
            stop = true;
            mess = format!(
                "mean residual ({:.3}) is higher than on the first iteration ({:.3})",
                mean_resids[iter_ind], mean_resids[0]
            );
        } else if (mean_resids[iter_ind - 1] != 0.
            && (mean_resids[iter_ind] > mean_resids[iter_ind - 1] * g.mean_stop_crit))
            || (max_resids[iter_ind - 1] != 0.
                && (max_resids[iter_ind] > max_resids[iter_ind - 1] * g.max_stop_crit))
        {
            stop = true;
            mess = format!(
                "mean or max residual increased too much (from {:.3}, {:.3} to {:.3}, {:.3})",
                mean_resids[iter_ind - 1],
                mean_resids[iter_ind],
                max_resids[iter_ind - 1],
                max_resids[iter_ind]
            );
        } else if mean_resids[iter_ind - 1] != 0.
            && mean_resids[iter_ind] > mean_resids[iter_ind - 1] * g.counting_mean_crit
        {
            if g.mean_increased {
                stop = true;
                if iter_ind < 2 {
                    index_error();
                }
                mess = format!(
                    "mean residual increased on two successive iterations ({:.3} -> {:.3} -> {:.3})",
                    mean_resids[iter_ind - 2],
                    mean_resids[iter_ind - 1],
                    mean_resids[iter_ind]
                );
            } else {
                // Fixed in translation (BUGS.md): `matchorwarp:115` sets
                // `meanIncreased = False` here, so native's two-successive-
                // increases stop can never fire; this records the increase.
                g.mean_increased = true;
            }
        }

        if stop {
            prnstr(
                &format!(
                    "WARNING: Stopped on iteration {} without applying new transforms",
                    g.iter_str
                ),
                "\n",
                false,
            );
            prnstr(&format!(" because {mess}"), "\n", false);
            prnstr("", "\n", false);
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }
    }

    // If succeed, run warpvol
    if savestat == 0 {
        prnstr(" ", "\n", false);
        if g.trial != 0 {
            prnstr("MATCHORWARP: Findwarp found a good warping", "\n", false);
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }

        prnstr(
            "MATCHORWARP: Findwarp found a good warping: next running Warpvol",
            "\n",
            false,
        );
        prnstr(" ", "\n", true);
        if iteration != 0 {
            let wv_name = format!("_warp{iteration}");
            let ren_name = if g.mat_root.ends_with("_mat") {
                format!(
                    "{}{wv_name}_mat{}",
                    &g.mat_root[..g.mat_root.len() - 4],
                    g.mat_ext
                )
            } else {
                format!("{}{wv_name}{}", g.mat_root, g.mat_ext)
            };
            let action = format!("Renaming {} to {ren_name}", g.mat_file);
            prnstr(&action, "\n", false);
            if std::fs::rename(&g.mat_file, &ren_name).is_err() {
                prnstr(
                    &format!(
                        "WARNING: Matchorwarp - Failed to rename {} to {ren_name}",
                        g.mat_file
                    ),
                    "\n",
                    false,
                );
            }
        }

        g.usable_mat = false;
        let mut comlines = g.volbase.clone();
        comlines.push(format!("TransformFile {warpfile}"));

        let _ = std::io::stdout().flush();
        run_cmd(
            "warpvol -StandardInput",
            Some(&comlines),
            Some("stdout"),
            None,
            &[],
        )?;
        return Ok(0);
    }

    Ok(savestat)
}

/// The script's top level (`matchorwarp:157-439`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn matchorwarp(arguments: &[OsString]) -> i32 {
    let progname = "matchorwarp";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let mut g = Globals::default();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    g.clipsize = String::new();

    // Fallbacks from ../manpages/autodoc2man 3 1 matchorwarp
    let options: Vec<String> = [
        "inputvolume:InputVolume:FN:",
        "outputvolume:OutputVolume:FN:",
        "size:SizeXYZorVolume:CH:",
        "refinelimit:RefineLimit:F:",
        "warplimit:WarpLimits:CH:",
        "structurecrit:StructureCriteria:CH:",
        "extentfit:ExtentToFit:FN:",
        "modelfile:ModelFile:FN:",
        "patchfile:PatchFile:FN:",
        "solvefile:SolveFile:FN:",
        "refinefile:RefineFile:FN:",
        "inversefile:InverseFile:FN:",
        "warpfile:WarpFile:FN:",
        "residualfile:ResidualFile:FN:",
        "vectormodel:VectorModel:FN:",
        "clipsize:ClipPlaneBoxSize:I:",
        "tempdir:TemporaryDirectory:FN:",
        "xlowerexclude:XLowerExclude:I:",
        "xupperexclude:XUpperExclude:I:",
        "ylowerexclude:YLowerExclude:I:",
        "yupperexclude:YUpperExclude:I:",
        "zlowerexclude:ZLowerExclude:I:",
        "zupperexclude:ZUpperExclude:I:",
        "linear:LinearInterpolation:B:",
        "trial:TrialMode:B:",
        "iterations:IterationsToRun:I:",
        // Fixed in translation (BUGS.md): the source's fallback table says
        // `CriteriaToStopIterating` (`matchorwarp:198`) but the program reads
        // `StopIteratingCriteria` (the autodoc's name), so without the autodoc
        // native exits "Illegal option" for -iterations 2 or more.
        "stop:StopIteratingCriteria:FT:",
        "patchcorr:PatchcorrCommandFile:FN:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 3, 1, 1);

    // Get all the options
    let recfile = pip_get_in_out_file("InputVolume", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if recfile.is_empty() {
        exit_error("An input volume must be entered");
    }
    let matfile = pip_get_in_out_file("OutputVolume", 1)
        .ok()
        .flatten()
        .unwrap_or_default();
    if matfile.is_empty() {
        exit_error("An output volume must be entered");
    }

    let sizein = pip_get_string("size", "").unwrap_or_default();
    if sizein.is_empty() {
        exit_error("-size must be entered with nx,ny,nz or file being matched to");
    }
    let patchfile = pip_get_string("patchfile", "patch.out").unwrap_or_default();
    let modelfile = pip_get_string("modelfile", "").unwrap_or_default();
    if !modelfile.is_empty() && !Path::new(&modelfile).exists() {
        exit_error(&format!("Model file {modelfile} does not exist"));
    }

    let solvefile = pip_get_string("solvefile", "solve.xf").unwrap_or_default();
    let refinefile = pip_get_string("refinefile", "refine.xf").unwrap_or_default();
    let inversefile = pip_get_string("inversefile", "inverse.xf").unwrap_or_default();
    let warpfile = pip_get_string("warpfile", "warp.xf").unwrap_or_default();
    let residualfile = pip_get_string("residualfile", "").unwrap_or_default();

    let vectormodel = pip_get_string("vectormodel", "").unwrap_or_default();
    let clipin = pip_get_integer("clipsize", 0).unwrap_or(0);
    if pip_get_err_no() == 0 {
        g.clipsize = format!("-c {clipin}");
    }
    let refinelimit = pip_get_float("refinelimit", 0.3).unwrap_or(0.3);
    g.warplimit = pip_get_string("warplimit", "0.2,0.27,0.35").unwrap_or_default();
    let struct_crit = pip_get_string("structurecrit", "").unwrap_or_default();
    let num_iterations = pip_get_integer("IterationsToRun", 1).unwrap_or(1);
    let mut patch_root = String::new();
    let mut patch_ext = String::new();
    let mut patch_com_root = String::new();
    let mut patch_lines: Vec<String> = Vec::new();
    if num_iterations > 1 {
        let patch_com = pip_get_string("PatchcorrCommandFile", "patchcorr.com").unwrap_or_default();
        let (patch_com, root) = complete_and_check_com_file(&patch_com);
        patch_com_root = root;
        patch_lines = read_text_file(&patch_com, Some("patchcorr command file"), false, None)
            .unwrap_or_default();
        let string_value = |option: &str| -> Option<String> {
            match option_value(&patch_lines, option, 0, false, 0, None, None) {
                Some(OptionValue::String(value)) => Some(value),
                _ => None,
            }
        };
        // Fixed in translation (BUGS.md): both messages concatenate
        // `patchFile` (`matchorwarp:240,245`), which is `None` when the first
        // test fails (native: uncaught TypeError); both name the command file.
        let patch_file = string_value("OutputFile");
        let Some(patch_file) = patch_file.filter(|name| !name.is_empty()) else {
            exit_error(&format!(
                "Cannot find name of patch output file in {patch_com}"
            ));
        };
        (patch_root, patch_ext) = os_path_splitext(&patch_file);

        let mat_file = string_value("FileToAlign");
        let Some(mat_file) = mat_file.filter(|name| !name.is_empty()) else {
            exit_error(&format!("Cannot find name of file to align in {patch_com}"));
        };
        (g.mat_root, g.mat_ext) = os_path_splitext(&mat_file);
        g.mat_file = mat_file;
        let (counting, mean_stop, max_stop) =
            pip_get_three_floats("StopIteratingCriteria", (1.03, 1.25, 2.))
                .unwrap_or((1.03, 1.25, 2.));
        (g.counting_mean_crit, g.mean_stop_crit, g.max_stop_crit) = (counting, mean_stop, max_stop);
    }

    let tempdir = pip_get_string("tempdir", "").unwrap_or_default();
    let linear = pip_get_boolean("linear", 0).unwrap_or(0);

    g.xlower = pip_get_integer("xlowerexclude", 0).unwrap_or(0);
    g.xupper = pip_get_integer("xupperexclude", 0).unwrap_or(0);
    g.ylower = pip_get_integer("ylowerexclude", 0).unwrap_or(0);
    g.yupper = pip_get_integer("yupperexclude", 0).unwrap_or(0);
    g.zlower = pip_get_integer("zlowerexclude", 0).unwrap_or(0);
    g.zupper = pip_get_integer("zupperexclude", 0).unwrap_or(0);
    g.trial = pip_get_boolean("trial", 0).unwrap_or(0);
    g.extent = pip_get_integer("extentfit", 0).unwrap_or(0);

    if !Path::new(&recfile).exists() {
        exit_error(&format!("Input volume {recfile} does not exist"));
    }
    if !Path::new(&patchfile).exists() {
        exit_error(&format!("Input file {patchfile} does not exist"));
    }
    if !Path::new(&solvefile).exists() {
        exit_error(&format!("Input file {solvefile} does not exist"));
    }

    // The size entry: If it is not an existing file, use as is and hope it is numbers
    // if it is a file, get the nx, ny, nz of it
    let mut size = sizein.clone();
    if Path::new(&sizein).exists() {
        match get_mrc_size(&sizein) {
            Ok((nx, ny, nz)) => size = format!("{nx},{ny},{nz}"),
            Err(_) => exit_from_imod_error(progname),
        }
    }

    if !vectormodel.is_empty() && residualfile.is_empty() {
        exit_error("A residual file must be specified to make a vector model");
    }

    // Set up name for object in output model and figure out if skipping warp
    g.objname = "Values are residuals".to_owned();
    if !g.clipsize.is_empty() {
        g.objname = "Values are residuals; clip planes exist".to_owned();
    }

    // Setup base lines for refinematch /findwarp
    g.findbase = vec![format!("VolumeOrSizeXYZ {sizein}")];
    if !residualfile.is_empty() {
        g.findbase
            .push(format!("ResidualPatchOutput {residualfile}"));
    }
    if !modelfile.is_empty() {
        g.findbase.push(format!("RegionModel {modelfile}"));
    }
    if !struct_crit.is_empty() {
        g.findbase.push("ExtraValueSelection 5,1".to_owned());
        g.findbase.push(format!("SelectionCriteria {struct_crit}"));
    }

    // Setup base lines for warpvol/matchvol
    g.volbase = vec![
        format!("InputFile {recfile}"),
        format!("OutputFile  {matfile}"),
        format!("OutputSizeXYZ {size}"),
    ];
    if !tempdir.is_empty() {
        g.volbase.push(format!("TemporaryDirectory {tempdir}"));
    }
    if linear != 0 {
        g.volbase.push("InterpolationOrder 1".to_owned());
    }

    // Run refinematch
    // The flush is needed because in old python (2.5 or below) the output printed from the
    // runcmd somehow gets ahead of this output
    prnstr(
        "MATCHORWARP: Running Refinematch to try to find single transformation",
        "\n",
        true,
    );

    let skipwarp =
        g.warplimit == "0" || g.warplimit == "0." || g.warplimit == ".0" || g.warplimit == "0.0";

    let mut comlines = g.findbase.clone();
    comlines.extend([
        format!("MeanResidualLimit {}", py_str_float(refinelimit)),
        format!("OutputFile {refinefile}"),
        format!("PatchFile {patchfile}"),
    ]);
    if !residualfile.is_empty() {
        comlines.push(format!("ResidualPatchOutput {residualfile}"));
    }

    g.mean_resids = Vec::new();
    g.max_resids = Vec::new();

    let outer: Result<(), ImodpyError> = (|| {
        let savestat = match run_cmd(
            "refinematch -StandardInput",
            Some(&comlines),
            Some("stdout"),
            None,
            &[],
        ) {
            Ok(_) => 0,
            Err(_) => get_last_exit_status(),
        };

        // Look for status 2 specifically, it is the code used when above the limit
        if savestat != 0 && savestat != 2 {
            let _ = std::io::stdout().flush();
            std::process::exit(1);
        }

        // If exiting either because of success or because warp is being skipped,
        // write the vector model now
        if (savestat == 0 || skipwarp) && !vectormodel.is_empty() {
            prnstr(" ", "\n", false);
            run_cmd(
                &format!(
                    "patch2imod {} -n \"{}\" \"{residualfile}\" \"{vectormodel}\"",
                    g.clipsize, g.objname
                ),
                None,
                None,
                None,
                &[],
            )?;
            prnstr(&format!("MATCHORWARP: Created {vectormodel}"), "\n", false);
        }

        if savestat == 0 {
            prnstr(" ", "\n", false);
            if g.trial != 0 {
                prnstr(
                    "MATCHORWARP: Refinematch found a good transformation",
                    "\n",
                    false,
                );
                let _ = std::io::stdout().flush();
                std::process::exit(0);
            }

            // If refinematch did not have error exit, run matchvol
            prnstr(
                "MATCHORWARP: Refinematch found a good transformation: next running Matchvol",
                "\n",
                false,
            );
            prnstr(" ", "\n", true);

            let mut comlines = g.volbase.clone();
            comlines.extend([
                format!("TransformFile {solvefile}"),
                format!("TransformFile {refinefile}"),
                format!("InverseFile {inversefile}"),
            ]);
            run_cmd(
                "matchvol -StandardInput",
                Some(&comlines),
                Some("stdout"),
                None,
                &[],
            )?;
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }

        // If there is an error exit from refinematch, run findwarp as long as warplimit not 0
        if skipwarp {
            prnstr(" ", "\n", false);
            prnstr(
                &format!(
                    "ERROR: MATCHORWARP - Refinematch gave a mean residual error above {} and warping is disabled",
                    py_str_float(refinelimit)
                ),
                "\n",
                false,
            );
            let _ = std::io::stdout().flush();
            std::process::exit(1);
        }

        prnstr(" ", "\n", false);
        prnstr(
            "MATCHORWARP: Running Findwarp to find a warping with given residual limits",
            "\n",
            true,
        );

        let savestat = run_findwarp(
            &mut g,
            &solvefile,
            &patchfile,
            &warpfile,
            &residualfile,
            &vectormodel,
            0,
        )?;
        if savestat != 0 {
            prnstr(" ", "\n", false);
            exit_error(
                "You need to get better patches, edit patches, or eliminate rows or columns",
            );
        }
        if num_iterations < 2 {
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }
        Ok(())
    })();
    if outer.is_err() {
        exit_from_imod_error(progname);
    }

    let (warp_root, warp_ext) = os_path_splitext(&warpfile);
    let (residual_root, residual_ext) = os_path_splitext(&residualfile);
    let (vector_root, vector_ext) = os_path_splitext(&vectormodel);
    let mut last_warp_xf = warpfile.clone();
    g.mean_increased = false;

    for iter_ind in 1..num_iterations.max(1) as usize {
        g.iter_ind = iter_ind;
        g.iter_str = (iter_ind + 1).to_string();
        let iter_str = g.iter_str.clone();
        g.usable_mat = true;

        let mut action = String::new();
        let result: Result<(), ImodpyError> = (|| {
            // Run warpvol to get filled transforms
            let (last_root, last_ext) = os_path_splitext(&last_warp_xf);
            let filled_file = format!("{last_root}-filled{last_ext}");
            let mut comlines = g.volbase.clone();
            comlines.extend([
                format!("TransformFile {last_warp_xf}"),
                format!("FilledInOutputFile {filled_file}"),
            ]);

            action = "Getting filled-in file from Warpvol".to_owned();
            run_cmd(
                "warpvol -StandardInput",
                Some(&comlines),
                Some("stdout"),
                None,
                &[],
            )?;

            // Modify the patch com and run patchcorr with new mat file
            let patch_out = format!("{patch_root}{iter_str}{patch_ext}");
            let sedcom = vec![
                sed_modify("OutputFile", &patch_out, '/'),
                "/InitialShiftXYZ/d".to_owned(),
                format!("/patch2imod/s/[^ ]*\\.out/{patch_out}/"),
                format!("/patch2imod/s/\\.mod/{iter_str}.mod/"),
            ];
            let patch_com = format!("{patch_com_root}-tmp.com");
            let patch_log = format!("{patch_com_root}{iter_str}.log");
            let _ = pysed(
                &sedcom,
                PysedSrc::Lines(&patch_lines),
                Some(&patch_com),
                false,
                '/',
                false,
            );

            prnstr(" ", "\n", false);
            action = format!("Running {patch_com}");
            prnstr(
                &format!(
                    "MATCHORWARP: Running Corrsearch3d to get new patch vectors for iteration {iter_str}"
                ),
                "\n",
                true,
            );

            // `runcmd('vmstopy -x -q ...')`: the in-process runner now
            // (owner, 2026-09-26: no Python, no pipes)
            crate::imod::comrun::run_com_as_command(&patch_com, &patch_log)?;
            cleanup_files(&[patch_com.clone()]);

            // Get names for next findwarp
            last_warp_xf = format!("{warp_root}{iter_str}{warp_ext}");
            // Fixed in translation (BUGS.md): with no -residualfile (and so no
            // -vectormodel) native builds these from empty roots
            // (`matchorwarp:424-425`), i.e. the bare iteration number, and
            // findwarp and patch2imod both write a file named e.g. `2`; an
            // empty name here means "no such output", as on the first pass.
            let resid_iter = if residualfile.is_empty() {
                String::new()
            } else {
                format!("{residual_root}{iter_str}{residual_ext}")
            };
            let vector_iter = if vectormodel.is_empty() {
                String::new()
            } else {
                format!("{vector_root}{iter_str}{vector_ext}")
            };
            action = "Running Findwarp".to_owned();
            prnstr(" ", "\n", false);
            prnstr(
                &format!(
                    "MATCHORWARP: Running Findwarp with new patch vectors for iteration {iter_str}"
                ),
                "\n",
                true,
            );
            let savestat = run_findwarp(
                &mut g,
                &filled_file,
                &patch_out,
                &last_warp_xf,
                &resid_iter,
                &vector_iter,
                iter_ind,
            )?;

            if savestat != 0 {
                usable_error(&g, progname, &action, iter_ind + 1);
            }
            Ok(())
        })();
        if result.is_err() {
            usable_error(&g, progname, &action, iter_ind + 1);
        }
    }

    let _ = std::io::stdout().flush();
    0
}
