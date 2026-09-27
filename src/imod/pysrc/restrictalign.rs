//! Translation of `IMOD/pysrc/restrictalign`.
//!
//! A Python command script: each `def` is a method of [`Ra`], which holds the
//! module-level variables those functions read and assign as globals, and
//! the top level is [`restrictalign`], translated statement by statement.
//!
//! Python semantics carried over: the tiltalign parameter array mixes
//! Python `None` (an option missing from the command file) with ints, so
//! `origParam` is `Option<i64>` where `None != 0` as in Python, and the
//! working arrays are plain `i64` (the source's `None -> 0` copy); a boolean
//! entry (`ProjectionStretch`) is its int value (`True == 1`).  Floats are
//! Python floats (doubles); `round()` rounds half to even; `{:.Nf}` formats
//! are Rust's, which round the exact binary value as Python does.
//!
//! Commands: `tiltalign -StandardInput` and `imodinfo` run through
//! [`run_cmd`], which runs our own programs in process; their printed output
//! is still parsed as text (the leave-out error lines, `WARNING` lines, the
//! `CONTOUR`/`contour`/`max` lines) -- tiltalign has no recording entry
//! point and imodinfo is being reworked concurrently.  `getmrcsize` uses
//! `header_in_process`; `submfg` runs through `run_cmd` (a Python-script
//! translation, so a child process).
//!
//! Upstream defects fixed in translation (`BUGS.md`, restrictalign):
//! `-order` absent without `-cross` crashed (`len(None)`); the
//! `'#ImageSizeXnadY'` typo left the fallback image size commented out; the
//! `len(imageSize) > 2` test made the two-value `ImageSizeXandY` fallback
//! dead; a missing `mess +=` dropped the "max line not found" message;
//! an empty contour in a patch-tracking model raised `IndexError`; the
//! verbose error print raised `IndexError` for a robust run reporting only
//! two errors; and `-cross` with a single fiducial, or robust fitting that
//! failed on the initial values with no parameter change, reached undefined
//! names (`NameError`).

use super::imodpy::{
    ImodpyError, OptionValue, add_imod_bin_ignore_sighup, convert_to_integer, exit_from_imod_error,
    get_err_strings, get_mrc_size, make_backup_file, option_value, print_pid, prnstr, py_int,
    read_text_file, run_cmd,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_integer_array, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add, sed_modify};
use super::vmstopy::python_float;
use std::ffi::OsString;
use std::io::Write as _;

const PROGNAME: &str = "restrictalign";

// A set of indices to the variable array
// MAKE SURE GROUP FOLLOWS OPT IN EVERY CASE
const ROT_OPT: usize = 0;
const ROT_GROUP: usize = 1;
const MAG_OPT: usize = 2;
const MAG_GROUP: usize = 3;
const TILT_OPT: usize = 4;
const TILT_GROUP: usize = 5;
const SKEW_OPT: usize = 6;
const SKEW_GROUP: usize = 7;
const XSTRETCH_OPT: usize = 8;
const XSTRETCH_GROUP: usize = 9;
const XTILT_OPT: usize = 10;
const XTILT_GROUP: usize = 11;
const BEAM_TILT_OPT: usize = 12;
const PROJ_STRETCH: usize = 13;
const LOC_ROT_OPT: usize = 14;
const LOC_MAG_OPT: usize = 16;
const LOC_TILT_OPT: usize = 18;
const LOC_XSTR_OPT: usize = 20;
const LOC_XSTR_GROUP: usize = 21;
const LOC_SKEW_OPT: usize = 22;
const LOC_SKEW_GROUP: usize = 23;

// Tiltalign option values
const TA_ONE_ROT: i64 = -1;
const TA_GROUP_ROT: i64 = 3;
const TA_ALL_ROT: i64 = 1;
const TA_ALL_MAG: i64 = 1;
const TA_GROUP_MAG: i64 = 3;
const TA_ALL_TILT: i64 = 2;
const TA_GROUP_TILT: i64 = 5;
const TA_GROUP_SKEW: i64 = 3;
const TA_GROUP_XSTRETCH: i64 = 3;
const TA_LINEAR_XTILT: i64 = 3;
const TA_BLOCK_XTILT: i64 = 4;

// Restrictions
const RES_GROUP_ROTS: i64 = 1;
const RES_ONE_ROT: i64 = 2;
const RES_FIX_TILTS: i64 = 3;
const RES_GROUP_MAGS: i64 = 4;
const RES_FIX_MAGS: i64 = 5;

// Order values for cross-validation
const CV_TEST_STRETCH: i64 = 1;
const CV_TEST_XTILT: i64 = 2;
const CV_TEST_TILT: i64 = 3;
const CV_TEST_ROT: i64 = 4;
const CV_TEST_MAG: i64 = 5;
const CV_TEST_SINGLES: i64 = 6;

// Module-level constants (`restrictalign:774-783`)
const MIN_MAG_GROUPING: i64 = 4;
const MIN_ROT_GROUPING: i64 = 5;
const MIN_TILT_GROUPING: i64 = 5;
const DFLT_SKEW_GROUPING: i64 = 11;
const DFLT_XSTRETCH_GROUPING: i64 = 7;
const MIN_BEADS_FOR_CV_ONLY: i64 = 5;
const MIN_BEADS_FOR_BEAM_TILT: i64 = 4;

/// Python truthiness of an `origParam` entry (`None` and 0 are false).
fn truthy(value: Option<i64>) -> bool {
    value.is_some_and(|v| v != 0)
}

/// The script's module-level variables that its functions read or assign
/// through `global`.
#[derive(Default)]
struct Ra {
    num_views: i64,
    num_beads: i64,
    num_points: i64,
    final_changes: bool,
    local_align: bool,
    test_local: i32,
    cross_validate: i32,
    test_local_area: bool,
    target_size: bool,
    robust_align: bool,
    verbose: bool,
    skip_beam_tilt: bool,
    big_grouping: i64,
    orig_param: Vec<Option<i64>>,
    orig_required: Vec<i64>,
    orig_area_or_num: Vec<i64>,
    sedcom: Vec<String>,
    ta_lines: Vec<String>,
    bad_robust: bool,
    doing_robust: bool,
    cum_non_rob_time: f64,
    cum_robust_time: f64,
    new_errors: Vec<f64>,
    new_param: Vec<i64>,
    next_param: Vec<i64>,
    next_required: Vec<i64>,
    next_area_or_num: Vec<i64>,
    new_required: Vec<i64>,
    new_area_or_num: Vec<i64>,
    last_err_diff: f64,
    last_errors: Vec<f64>,
    prev_last_errs: Vec<f64>,
    prev_err_diff: f64,
    cur_err_diff: f64,
    prev_errors: Vec<f64>,
    prev_param: Vec<i64>,
    prev_required: Vec<i64>,
    prev_area_or_num: Vec<i64>,
    was_tested: Vec<i64>,
}

impl Ra {
    /// Matches `groupedUnknowns` (`restrictalign:48`).
    fn grouped_unknowns(&self, size: i64) -> i64 {
        if size <= 1 {
            return self.num_views;
        }
        (self.num_views + 1.max(size) - 1).div_euclid(1.max(size)) + 1
    }

    /// Matches `prnFinal` (`restrictalign:53`).
    fn prn_final(&self, strn: &str) {
        if self.final_changes {
            prnstr(strn, "\n", false);
        }
    }

    /// Matches `measuredToUnknown` (`restrictalign:58`).
    fn measured_to_unknown(&self, param: &[i64]) -> f64 {
        let measured = self.num_points as f64 * 2.;
        let mut unknowns = 3. * (self.num_beads - 1) as f64 + 2. * (self.num_views - 1) as f64;
        if param[ROT_OPT] == TA_ALL_ROT {
            unknowns += self.num_views as f64;
        } else if param[ROT_OPT] == TA_GROUP_ROT {
            unknowns += self.grouped_unknowns(param[ROT_GROUP]) as f64;
        } else if param[ROT_OPT] == TA_ONE_ROT {
            unknowns += 1.;
        }
        if param[TILT_OPT] == TA_GROUP_TILT {
            unknowns += self.grouped_unknowns(param[TILT_GROUP]) as f64;
        } else if param[TILT_OPT] == TA_ALL_TILT {
            unknowns += (self.num_views - 1) as f64;
        }
        if param[MAG_OPT] == TA_ALL_MAG {
            unknowns += (self.num_views - 1) as f64;
        } else if param[MAG_OPT] == TA_GROUP_MAG {
            unknowns += self.grouped_unknowns(param[MAG_GROUP]) as f64;
        }
        if param[SKEW_OPT] == TA_GROUP_SKEW {
            unknowns += self.grouped_unknowns(param[SKEW_GROUP]) as f64;
        }
        if param[XSTRETCH_OPT] == TA_GROUP_XSTRETCH {
            unknowns += self.grouped_unknowns(param[XSTRETCH_GROUP]) as f64;
        }
        if param[XTILT_OPT] == TA_LINEAR_XTILT || param[XTILT_OPT] == TA_BLOCK_XTILT {
            unknowns += self.grouped_unknowns(param[XTILT_GROUP]) as f64;
        }
        if param[BEAM_TILT_OPT] != 0 {
            unknowns += 1.;
        }
        if param[PROJ_STRETCH] != 0 {
            unknowns += 1.;
        }
        measured / unknowns
    }

    /// Matches `setOptAndGrouping` (`restrictalign:106`).
    fn set_opt_and_grouping(
        &mut self,
        param: &[i64],
        opt_ind: usize,
        group_opt: i64,
        variable: &str,
        prefix: &str,
    ) {
        let opt = param[opt_ind];
        let mut did_mess = false;
        let group_ind = opt_ind + 1;
        if self.orig_param[opt_ind] != Some(opt) {
            if opt == 0 {
                self.prn_final(&format!("Turned off solving for {variable}"));
            } else if opt == group_opt {
                self.prn_final(&format!(
                    "Turned on grouping of {variable}s to {}",
                    param[group_ind]
                ));
                did_mess = true;
            } else if variable == "rotation" && opt == TA_ONE_ROT {
                self.prn_final("Switched to solving for one rotation");
            }
            self.sedcom.push(sed_modify(
                &format!("{prefix}Option"),
                &opt.to_string(),
                '/',
            ));
        }
        if opt == group_opt && self.orig_param[group_ind] != Some(param[group_ind]) {
            if !did_mess {
                self.prn_final(&format!(
                    "Changed {variable} grouping to {}",
                    param[group_ind]
                ));
            }
            self.sedcom.push(sed_modify(
                &format!("{prefix}DefaultGrouping"),
                &param[group_ind].to_string(),
                '/',
            ));
        }
    }

    /// Matches `buildUpSedcom` (`restrictalign:127`).
    fn build_up_sedcom(
        &mut self,
        param: &[i64],
        robust_off: &str,
        required: &[i64],
        area_or_num: &[i64],
    ) {
        self.sedcom = Vec::new();
        let mut cross_val_opt = 1;
        let op = &self.orig_param;
        if self.local_align
            && self.test_local == 0
            && (!self.final_changes || self.cross_validate == 0)
        {
            if self.cross_validate == 0 {
                self.prn_final("Turned off local alignments");
            }
            self.sedcom.push(sed_modify("LocalAlignments", "0", '/'));
        }

        // This is not allowed to happen
        if !self.local_align && self.test_local != 0 {
            self.prn_final("Turned on local alignments");
            self.sedcom.push(sed_modify("LocalAlignments", "1", '/'));
        }

        // batchruntomo looking for 'off robust'
        if !robust_off.is_empty() {
            self.prn_final(&format!(
                "Turned off robust fitting because {robust_off}     [rsa2]"
            ));
            self.sedcom.extend(sed_del_and_add(
                "RobustFitting",
                "0",
                "OutputTransformFile",
                '/',
            ));
        }
        let op = op.clone();
        if truthy(op[XSTRETCH_OPT]) && param[XSTRETCH_OPT] == 0 {
            self.prn_final("Turned off solving for X stretch");
            self.sedcom.push(sed_modify("XStretchOption", "0", '/'));
            self.sedcom
                .push(sed_modify("LocalXStretchOption", "0", '/'));
        }
        if param[XSTRETCH_OPT] != 0
            && (op[XSTRETCH_OPT] != Some(param[XSTRETCH_OPT])
                || op[XSTRETCH_GROUP] != Some(param[XSTRETCH_GROUP]))
        {
            self.prn_final(&format!(
                "Set grouping of X stretch to {}",
                param[XSTRETCH_GROUP]
            ));
            self.sedcom.push(sed_modify(
                "XStretchOption",
                &param[XSTRETCH_OPT].to_string(),
                '/',
            ));
            self.sedcom.push(sed_modify(
                "XStretchDefaultGrouping",
                &param[XSTRETCH_GROUP].to_string(),
                '/',
            ));
        }

        if truthy(op[SKEW_OPT]) && param[SKEW_OPT] == 0 {
            self.prn_final("Turned off solving for skew");
            self.sedcom.push(sed_modify("SkewOption", "0", '/'));
            self.sedcom.push(sed_modify("LocalSkewOption", "0", '/'));
        }
        if param[SKEW_OPT] != 0
            && (op[SKEW_OPT] != Some(param[SKEW_OPT]) || op[SKEW_GROUP] != Some(param[SKEW_GROUP]))
        {
            self.prn_final(&format!("Set grouping of skew to {}", param[SKEW_GROUP]));
            self.sedcom
                .push(sed_modify("SkewOption", &param[SKEW_OPT].to_string(), '/'));
            self.sedcom.push(sed_modify(
                "SkewDefaultGrouping",
                &param[SKEW_GROUP].to_string(),
                '/',
            ));
        }

        if truthy(op[PROJ_STRETCH]) && param[PROJ_STRETCH] == 0 {
            self.prn_final("Turned off solving for projection stretch");
            self.sedcom.push("/ProjectionStretch/d".to_owned());
        }

        self.set_opt_and_grouping(param, ROT_OPT, TA_GROUP_ROT, "rotation", "Rot");
        self.set_opt_and_grouping(param, TILT_OPT, TA_GROUP_TILT, "tilt angle", "Tilt");
        self.set_opt_and_grouping(param, MAG_OPT, TA_GROUP_MAG, "magnification", "Mag");

        self.set_opt_and_grouping(
            param,
            LOC_ROT_OPT,
            TA_GROUP_ROT,
            "local rotation",
            "LocalRot",
        );
        self.set_opt_and_grouping(
            param,
            LOC_TILT_OPT,
            TA_GROUP_TILT,
            "local tilt angle",
            "LocalTilt",
        );
        self.set_opt_and_grouping(
            param,
            LOC_MAG_OPT,
            TA_GROUP_MAG,
            "local magnification",
            "LocalMag",
        );
        self.set_opt_and_grouping(
            param,
            LOC_XSTR_OPT,
            TA_GROUP_XSTRETCH,
            "local X-stretch",
            "LocalXStretch",
        );
        self.set_opt_and_grouping(
            param,
            LOC_SKEW_OPT,
            TA_GROUP_SKEW,
            "local skew",
            "LocalSkew",
        );

        if param[XTILT_OPT] != 0
            && (op[XTILT_OPT] != Some(param[XTILT_OPT])
                || op[XTILT_GROUP] != Some(param[XTILT_GROUP]))
        {
            self.prn_final("Switched to solving for single X-tilt");
            self.sedcom.push(sed_modify(
                "XTiltOption",
                &param[XTILT_OPT].to_string(),
                '/',
            ));
            self.sedcom.push(sed_modify(
                "XTiltDefaultGrouping",
                &param[XTILT_GROUP].to_string(),
                '/',
            ));
        }

        if truthy(op[XTILT_OPT]) && param[XTILT_OPT] == 0 {
            self.prn_final("Turned off solving for X-axis tilt");
            self.sedcom.push(sed_modify("XTiltOption", "0", '/'));
        }

        let beam_opt = param[BEAM_TILT_OPT];
        if (!truthy(op[BEAM_TILT_OPT]) && beam_opt != 0)
            || (truthy(op[BEAM_TILT_OPT]) && op[BEAM_TILT_OPT] != Some(beam_opt))
        {
            if beam_opt != 0 {
                self.prn_final("Added beam tilt solution because solving for only one rotation");
            } else {
                self.prn_final("Turned off solving for beam tilt");
            }
            self.sedcom.extend(sed_del_and_add(
                "BeamTiltOption",
                &beam_opt.to_string(),
                "OutputTransformFile",
                '/',
            ));
        }

        if self.test_local_area {
            cross_val_opt = 2;
            if required[0] != self.orig_required[0] || required[1] != self.orig_required[1] {
                let req_str = format!("{},{}", required[0], required[1]);
                self.prn_final(&format!("Changed required # of fiducials to {req_str}"));
                self.sedcom
                    .push(sed_modify("MinFidsTotalAndEachSurface", &req_str, '/'));
            }
            if area_or_num[0] != self.orig_area_or_num[0]
                || area_or_num[1] != self.orig_area_or_num[1]
            {
                let area_str = format!("{},{}", area_or_num[0], area_or_num[1]);
                if self.target_size {
                    self.prn_final(&format!("Changed target area size to  {area_str}"));
                    self.sedcom
                        .push(sed_modify("TargetPatchSizeXandY", &area_str, '/'));
                } else {
                    self.prn_final(&format!("Changed number of local areas to  {area_str}"));
                    self.sedcom
                        .push(sed_modify("NumberOfLocalPatchesXandY", &area_str, '/'));
                }
            }
        }

        // For a test, add the cross-validation option and the one to do contours
        if !self.final_changes {
            self.sedcom.extend(sed_del_and_add(
                "CrossValidate",
                &cross_val_opt.to_string(),
                "OutputTransformFile",
                '/',
            ));
            if self.cross_validate > 1 {
                self.sedcom.extend(sed_del_and_add(
                    "LeaveOutPredictAndPad",
                    "0,0",
                    "OutputTransformFile",
                    '/',
                ));
            }
        }
    }

    /// Matches `runTiltalignExtractErrors` (`restrictalign:221`).
    fn run_tiltalign_extract_errors(&mut self, descrip: &str, robust: &str) -> Vec<f64> {
        let mut errors: Vec<f64> = Vec::new();
        self.bad_robust = false;
        let run_lines = pysed(
            &self.sedcom,
            PysedSrc::Lines(&self.ta_lines),
            None,
            false,
            '/',
            false,
        )
        .ok()
        .flatten()
        .unwrap_or_default();
        let mut warn_lines: Vec<String> = Vec::new();
        let mut tag = "Global";
        if self.test_local != 0 {
            tag = "Local";
        }

        match run_cmd(
            "tiltalign -StandardInput",
            Some(&run_lines),
            None,
            None,
            &[],
        ) {
            Ok(out_lines) => {
                for line in out_lines.unwrap_or_default() {
                    // Look for robust failure and turn off robust for further tests, pass on
                    // warning
                    let lower = line.to_lowercase();
                    if lower.contains("too few") && lower.contains("robust fitting") {
                        self.bad_robust = true;
                    } else if line.starts_with("WARNING") {
                        warn_lines.push(line.trim().to_owned());
                        if line.contains("rotation angle") && line.contains("closer to") {
                            let mut angles: Vec<f64> = Vec::new();
                            for word in line.split_whitespace() {
                                if let Some(value) = python_float(word) {
                                    angles.push(value);
                                }
                            }

                            // Tiltalign's criterion in 15 and it can pull in a correct angle
                            // with 30, so make criterion for fail somewhat higher than 15
                            if angles.len() == 2 && (angles[1] - angles[0]).abs() > 22. {
                                exit_error(
                                    &("Fix the initial rotation angle as suggested by that "
                                        .to_owned()
                                        + "warning before trying to optimize the parameters"),
                                );
                            }
                        }
                    }

                    // For a leave-out line, extract the values
                    if line.contains(tag) && line.contains("leave-out") {
                        let lsplit: Vec<&str> = line.split_whitespace().collect();
                        for ind in 0..lsplit.len().saturating_sub(1) {
                            if lsplit[ind].ends_with("):") {
                                match python_float(lsplit[ind + 1]) {
                                    Some(value) => errors.push(value),
                                    None => exit_error(&format!(
                                        "Converting {} to floating point number",
                                        lsplit[ind + 1]
                                    )),
                                }
                            }
                            if lsplit[ind].starts_with('w')
                                && lsplit[ind].contains('g')
                                && lsplit[ind].contains('t')
                            {
                                match python_float(lsplit[ind + 1]) {
                                    Some(value) => errors.push(value),
                                    None => exit_error(&format!(
                                        "Converting {} to floating point number",
                                        lsplit[ind + 1]
                                    )),
                                }
                            }
                        }
                    }
                }
            }
            Err(ImodpyError { .. }) => {
                let err_str = get_err_strings();
                prnstr(
                    &format!("WARNING: Tiltalign failed with error with {descrip}{robust}:"),
                    "\n",
                    false,
                );
                for line in err_str {
                    prnstr(&line.trim().replace("ERROR:", "   (error):"), "\n", false);
                }
                errors = vec![-2.];
                if !robust.is_empty() {
                    errors = vec![-2., -2., -2., -2.];
                }
                return errors;
            }
        }

        if errors.is_empty() {
            exit_error("Could not find leave-out errors in Tiltalign output");
        }
        if self.bad_robust && errors.len() > 1 {
            errors.pop();
            if errors.len() > 1 {
                errors.pop();
            }
        }

        if !warn_lines.is_empty() {
            prnstr(
                &format!("WARNING: Tiltalign gave warning with {descrip}{robust}:"),
                "\n",
                false,
            );
            for line in &warn_lines {
                prnstr(&format!("    {line}"), "\n", false);
            }
        }
        if self.bad_robust {
            errors.extend([-1., -1., -1.]);
        }
        errors
    }

    /// Matches `doTiltalignRuns` (`restrictalign:299`).
    ///
    /// Fixed in translation: the verbose report indexed `errors[2]` and
    /// `errors[3]` before the list is padded to four, so a robust run that
    /// reported only two errors raised `IndexError`; the report reads the
    /// padding value, -1, there instead.
    fn do_tiltalign_runs(
        &mut self,
        param: &[i64],
        descrip: &str,
        required: &[i64],
        area_or_num: &[i64],
    ) -> Vec<f64> {
        let mut no_robust = "";
        if self.robust_align && !self.doing_robust {
            no_robust = "for eval";
        }
        self.build_up_sedcom(param, no_robust, required, area_or_num);
        let start_time = std::time::SystemTime::now();
        let mut errors = self.run_tiltalign_extract_errors(descrip, "");
        self.cum_non_rob_time += start_time.elapsed().map(|d| d.as_secs_f64()).unwrap_or(0.);
        if errors.len() > 1 && !(self.robust_align && self.doing_robust) {
            exit_error(
                &("Inconsistent output from Tiltalign: weighted error despite ".to_owned()
                    + "robust fitting turned off"),
            );
        }
        if self.robust_align && self.doing_robust {
            if errors.len() < 2 {
                exit_error(
                    &("Could not find weighted leave-out error in Tiltalign output with "
                        .to_owned()
                        + "robust fitting"),
                );
            }
            if errors[1] == -1. {
                self.doing_robust = false;
            }
        }

        if self.verbose {
            let e = |ind: usize| errors.get(ind).copied().unwrap_or(-1.);
            if errors[0] >= 0.5 {
                if self.doing_robust {
                    prnstr(
                        &format!(
                            "{:.3} {:.3} {:.3} {:.3}: errors with {descrip}",
                            e(0),
                            e(1),
                            e(2),
                            e(3)
                        ),
                        "",
                        false,
                    );
                } else {
                    prnstr(&format!("{:.3}: error with {descrip}", e(0)), "", false);
                }
            } else if self.doing_robust {
                prnstr(
                    &format!(
                        "{:.4} {:.4} {:.4} {:.4}: errors with {descrip}",
                        e(0),
                        e(1),
                        e(2),
                        e(3)
                    ),
                    "",
                    false,
                );
            } else {
                prnstr(&format!("{:.4}: error with {descrip}", e(0)), "", false);
            }
        }

        if errors.len() < 4 {
            for _ind in errors.len()..4 {
                errors.push(-1.);
            }
        }
        errors
    }

    /// Matches `compareNextParam` (`restrictalign:342`).
    fn compare_next_param(&mut self, descrip: &str) -> i32 {
        let next_param = self.next_param.clone();
        let next_required = self.next_required.clone();
        let next_area_or_num = self.next_area_or_num.clone();
        let errors =
            self.do_tiltalign_runs(&next_param, descrip, &next_required, &next_area_or_num);
        let mut diff = -999.;
        self.prev_last_errs = self.last_errors.clone();
        self.prev_err_diff = self.last_err_diff;
        self.last_err_diff = -999.;
        let ne = &self.new_errors;
        if self.doing_robust && errors[3] > 0. && ne[3] > 0. {
            diff = ((ne[2] - errors[2]) / ne[2] + (ne[3] - errors[3]) / ne[3]) / 2.;
        } else if errors[0] > 0. {
            diff = (ne[0] - errors[0]) / ne[0];
        }

        // Get difference from last error too, so that local area can count up unique
        // ones that are worse
        let le = &self.last_errors;
        if self.doing_robust && errors[3] > 0. && le[3] > 0. {
            self.last_err_diff = ((le[2] - errors[2]) / le[2] + (le[3] - errors[3]) / le[3]) / 2.;
            self.last_errors = errors.clone();
        } else if errors[0] > 0. {
            self.last_err_diff = (le[0] - errors[0]) / le[0];
            self.last_errors = errors.clone();
        }

        // Give output if verbose or if it is better, do assignments to best if it is better
        if self.verbose {
            if diff > 0. || diff == -999. {
                prnstr(" ", "\n", false);
            } else if diff < 0. {
                prnstr(&format!(" -  {:.2}% higher", -100. * diff), "\n", false);
            } else {
                prnstr(" -  the same", "\n", false);
            }
        }
        if diff > 0. {
            // A new best one, copy to the "new" params and errors, and save the current
            // "new" params in prec so it is possible to revert
            self.cur_err_diff = diff;
            self.prev_errors = self.new_errors.clone();
            self.prev_param = self.new_param.clone();
            self.prev_required = self.new_required.clone();
            self.prev_area_or_num = self.new_area_or_num.clone();
            self.new_errors = errors;
            self.new_param = self.next_param.clone();
            self.new_required = self.next_required.clone();
            self.new_area_or_num = self.next_area_or_num.clone();
            prnstr(
                &format!("Leave-out error {:.2}% lower with {descrip}", diff * 100.),
                "\n",
                false,
            );
            1
        } else {
            // If not better, set up nextParam as copy of current best
            self.next_param = self.new_param.clone();
            self.next_required = self.new_required.clone();
            self.next_area_or_num = self.new_area_or_num.clone();
            if diff == 0. {
                return 0;
            }
            -1
        }
    }

    /// Matches `revertToStep` (`restrictalign:394`).
    fn revert_to_step(
        &mut self,
        err_diff: f64,
        last_errs: &[f64],
        errors: &[f64],
        param: &[i64],
        required: &[i64],
        area_or_num: &[i64],
    ) {
        self.last_errors = last_errs.to_vec();
        self.last_err_diff = err_diff;
        self.new_errors = errors.to_vec();
        self.new_param = param.to_vec();
        self.new_required = required.to_vec();
        self.new_area_or_num = area_or_num.to_vec();
        self.next_param = self.new_param.clone();
        self.next_required = self.new_required.clone();
        self.next_area_or_num = self.new_area_or_num.clone();
    }

    /// Matches `cvTestStretch` (`restrictalign:410`).
    fn cv_test_stretch(&mut self, take_one_step: i32) -> i32 {
        // Test with default grouping of stretch
        let np = &self.new_param;
        let need_group_skew = np[SKEW_OPT] != 0
            && (np[SKEW_OPT] != TA_GROUP_SKEW || np[SKEW_GROUP] < DFLT_SKEW_GROUPING);
        let need_group_xstr = np[XSTRETCH_OPT] != 0
            && (np[XSTRETCH_OPT] != TA_GROUP_XSTRETCH
                || np[XSTRETCH_GROUP] < DFLT_XSTRETCH_GROUPING);
        if (need_group_skew || need_group_xstr) && self.was_tested[XSTRETCH_OPT] < 1 {
            if need_group_skew {
                self.next_param[SKEW_OPT] = TA_GROUP_SKEW;
                self.next_param[SKEW_GROUP] = self.new_param[SKEW_GROUP].max(DFLT_SKEW_GROUPING);
            }
            if need_group_xstr {
                self.next_param[XSTRETCH_OPT] = TA_GROUP_XSTRETCH;
                self.next_param[XSTRETCH_GROUP] =
                    self.new_param[XSTRETCH_GROUP].max(DFLT_XSTRETCH_GROUPING);
            }
            self.was_tested[XSTRETCH_OPT] = 1;
            if self.compare_next_param("standard grouping of X-stretch and skew") > 0
                && take_one_step != 0
            {
                return 1;
            }
        }

        // Test with large grouping or no stretch
        if ((self.new_param[XSTRETCH_OPT] != 0
            && self.next_param[XSTRETCH_GROUP] < self.big_grouping)
            || (self.new_param[SKEW_OPT] != 0 && self.next_param[SKEW_GROUP] < self.big_grouping))
            && self.was_tested[XSTRETCH_OPT] < 2
        {
            if self.new_param[SKEW_OPT] != 0 {
                self.next_param[SKEW_OPT] = TA_GROUP_SKEW;
                self.next_param[SKEW_GROUP] = self.big_grouping;
            }
            if self.new_param[XSTRETCH_OPT] != 0 {
                self.next_param[XSTRETCH_OPT] = TA_GROUP_XSTRETCH;
                self.next_param[XSTRETCH_GROUP] = self.big_grouping;
            }
            self.was_tested[XSTRETCH_OPT] = 2;
            if self.compare_next_param("large grouping of X-stretch and skew") > 0
                && take_one_step != 0
            {
                return 1;
            }
        }

        if (self.next_param[XSTRETCH_OPT] != 0 || self.next_param[SKEW_OPT] != 0)
            && self.was_tested[XSTRETCH_OPT] < 3
        {
            self.next_param[SKEW_OPT] = 0;
            self.next_param[XSTRETCH_OPT] = 0;
            self.was_tested[XSTRETCH_OPT] = 3;
            if self.compare_next_param("not solving for X-stretch or skew") > 0 {
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestXTilt` (`restrictalign:454`).
    fn cv_test_xtilt(&mut self) -> i32 {
        // Test not solving for many x-tilts
        if self.new_param[XTILT_OPT] != 0
            && (self.new_param[XTILT_GROUP] < self.num_views
                || self.new_param[XTILT_OPT] == TA_LINEAR_XTILT)
            && self.was_tested[XTILT_OPT] < 1
        {
            self.next_param[XTILT_OPT] = TA_BLOCK_XTILT;
            self.next_param[XTILT_GROUP] = 2 * self.num_views;
            self.was_tested[XTILT_OPT] = 1;
            if self.compare_next_param("solving for only a single X-tilt") > 0 {
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestTilt` (`restrictalign:470`).
    fn cv_test_tilt(&mut self, take_one_step: i32) -> i32 {
        // Check tilt with standard grouping
        if (self.new_param[TILT_OPT] == TA_ALL_TILT
            || (self.new_param[TILT_OPT] == TA_GROUP_TILT
                && self.new_param[TILT_GROUP] < MIN_TILT_GROUPING))
            && self.was_tested[TILT_OPT] < 1
        {
            self.next_param[TILT_OPT] = TA_GROUP_TILT;
            self.next_param[TILT_GROUP] = self.next_param[TILT_GROUP].max(MIN_TILT_GROUPING);
            self.was_tested[TILT_OPT] = 1;
            if self.compare_next_param("standard grouping of tilt") > 0 && take_one_step != 0 {
                return 1;
            }
        }

        // Then tilt with big group or not at all
        if self.new_param[TILT_OPT] == TA_GROUP_TILT
            && self.new_param[TILT_GROUP] < self.big_grouping
            && self.was_tested[TILT_OPT] < 2
        {
            self.next_param[TILT_OPT] = TA_GROUP_TILT;
            self.next_param[TILT_GROUP] = self.big_grouping;
            self.was_tested[TILT_OPT] = 2;
            if self.compare_next_param("large grouping of tilt") > 0 && take_one_step != 0 {
                return 1;
            }
        }

        if self.new_param[TILT_OPT] != 0 && self.was_tested[TILT_OPT] < 3 {
            self.next_param[TILT_OPT] = 0;
            self.was_tested[TILT_OPT] = 3;
            if self.compare_next_param("not solving for tilt") > 0 {
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestRotation` (`restrictalign:502`).
    fn cv_test_rotation(&mut self, take_one_step: i32) -> i32 {
        // rotation standard grouping
        if (self.new_param[ROT_OPT] == TA_ALL_ROT
            || (self.new_param[ROT_OPT] == TA_GROUP_ROT
                && self.new_param[ROT_GROUP] < MIN_ROT_GROUPING))
            && self.was_tested[ROT_OPT] < 1
        {
            self.next_param[ROT_OPT] = TA_GROUP_ROT;
            self.next_param[ROT_GROUP] = MIN_ROT_GROUPING.max(self.next_param[ROT_GROUP]);
            self.was_tested[ROT_OPT] = 1;
            if self.compare_next_param("standard grouping of rotation") > 0 && take_one_step != 0 {
                return 1;
            }
        }

        // Rotation large grouping and solving for one
        if (self.new_param[ROT_OPT] == TA_ALL_ROT
            || (self.new_param[ROT_OPT] == TA_GROUP_ROT
                && self.new_param[ROT_GROUP] < self.big_grouping))
            && self.was_tested[ROT_OPT] < 2
        {
            self.next_param[ROT_OPT] = TA_GROUP_ROT;
            self.next_param[ROT_GROUP] = self.big_grouping;
            self.was_tested[ROT_OPT] = 2;
            if self.compare_next_param("large grouping of rotation") > 0 && take_one_step != 0 {
                return 1;
            }
        }

        if self.new_param[ROT_OPT] != TA_ONE_ROT && self.was_tested[ROT_OPT] < 3 {
            self.next_param[ROT_OPT] = TA_ONE_ROT;
            self.was_tested[ROT_OPT] = 3;
            if self.compare_next_param("solving for one rotation") > 0 && take_one_step != 0 {
                return 1;
            }
        }

        // Finally try fixed if solving for one
        if self.new_param[ROT_OPT] == TA_ONE_ROT && self.was_tested[ROT_OPT] < 4 {
            self.next_param[ROT_OPT] = 0;
            self.was_tested[ROT_OPT] = 4;
            if self.compare_next_param("rotation fixed at initial value") > 0 && take_one_step != 0
            {
                return 1;
            }
        }

        if self.new_param[ROT_OPT] == TA_ONE_ROT
            && self.orig_param[ROT_OPT] != Some(TA_ONE_ROT)
            && !self.skip_beam_tilt
            && self.num_beads >= MIN_BEADS_FOR_BEAM_TILT
            && self.was_tested[ROT_OPT] < 5
        {
            self.next_param[BEAM_TILT_OPT] = 2;
            self.was_tested[ROT_OPT] = 5;
            if self.compare_next_param("solving for beam tilt when solving for only one tilt") > 0 {
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestMagnification` (`restrictalign:549`).
    fn cv_test_magnification(&mut self, take_one_step: i32) -> i32 {
        // Magnification standard grouping
        if (self.new_param[MAG_OPT] == TA_ALL_MAG
            || (self.new_param[MAG_OPT] == TA_GROUP_MAG
                && self.new_param[MAG_GROUP] < MIN_MAG_GROUPING))
            && self.was_tested[MAG_OPT] < 1
        {
            self.next_param[MAG_OPT] = TA_GROUP_MAG;
            self.next_param[MAG_GROUP] = MIN_MAG_GROUPING.max(self.next_param[MAG_GROUP]);
            self.was_tested[MAG_OPT] = 1;
            if self.compare_next_param("standard grouping of magnification") > 0
                && take_one_step != 0
            {
                return 1;
            }
        }

        // Magnification large grouping and fixed
        if (self.new_param[MAG_OPT] == TA_ALL_MAG
            || (self.new_param[MAG_OPT] == TA_GROUP_MAG
                && self.new_param[MAG_GROUP] < self.big_grouping))
            && self.was_tested[MAG_OPT] < 2
        {
            self.next_param[MAG_OPT] = TA_GROUP_MAG;
            self.next_param[MAG_GROUP] = self.big_grouping;
            self.was_tested[MAG_OPT] = 2;
            if self.compare_next_param("large grouping of magnification") > 0 && take_one_step != 0
            {
                return 1;
            }
        }
        if self.next_param[MAG_OPT] != 0 && self.was_tested[MAG_OPT] < 3 {
            self.next_param[MAG_OPT] = 0;
            self.was_tested[MAG_OPT] = 3;
            if self.compare_next_param("not solving for magnification") > 0 {
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestSingleVars` (`restrictalign:579`).
    fn cv_test_single_vars(&mut self, take_one_step: i32) -> i32 {
        // Single-variable items
        if self.new_param[BEAM_TILT_OPT] != 0 && self.was_tested[BEAM_TILT_OPT] < 1 {
            self.next_param[BEAM_TILT_OPT] = 0;
            self.was_tested[BEAM_TILT_OPT] = 1;
            if self.compare_next_param("not solving for beam tilt") > 0 && take_one_step != 0 {
                if take_one_step > 1 {
                    return BEAM_TILT_OPT as i32;
                }
                return 1;
            }
        }

        if self.new_param[XTILT_OPT] != 0 && self.was_tested[XTILT_OPT] < 2 {
            self.next_param[XTILT_OPT] = 0;
            self.was_tested[XTILT_OPT] = 2;
            if self.compare_next_param("not solving for x-axis tilt") > 0 && take_one_step != 0 {
                if take_one_step > 1 {
                    return XTILT_OPT as i32;
                }
                return 1;
            }
        }

        if self.new_param[PROJ_STRETCH] != 0 && self.was_tested[PROJ_STRETCH] < 1 {
            self.next_param[PROJ_STRETCH] = 0;
            self.was_tested[PROJ_STRETCH] = 1;
            if self.compare_next_param("not solving for projection stretch") > 0 {
                if take_one_step > 1 {
                    return PROJ_STRETCH as i32;
                }
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestLocalStretch` (`restrictalign:613`).
    fn cv_test_local_stretch(&mut self, take_one_step: i32) -> i32 {
        // Stretch/skew is a pain, first test grouping if none, then increase it, then
        // turn it off
        let def_skew = DFLT_SKEW_GROUPING.max(self.new_param[LOC_SKEW_GROUP]);
        let def_stretch = DFLT_XSTRETCH_GROUPING.max(self.new_param[LOC_XSTR_GROUP]);
        let medium_skew = (def_skew * 2).min((def_skew + self.big_grouping).div_euclid(2));
        let medium_stretch = (def_stretch * 2).min((def_stretch + self.big_grouping).div_euclid(2));

        for (grp_skew, grp_stretch, test_val, type_text) in [
            (def_skew, def_stretch, 1, ""),
            (medium_skew, medium_stretch, 2, "more "),
            (self.big_grouping, self.big_grouping, 3, "large "),
        ] {
            let np = &self.new_param;
            let need_group_skew = np[LOC_SKEW_OPT] != 0
                && (np[LOC_SKEW_OPT] != TA_GROUP_SKEW || np[LOC_SKEW_GROUP] < grp_skew);
            let need_group_xstr = np[LOC_XSTR_OPT] != 0
                && (np[LOC_XSTR_OPT] != TA_GROUP_XSTRETCH || np[LOC_XSTR_GROUP] < grp_stretch);

            if (need_group_xstr || need_group_skew) && self.was_tested[LOC_XSTR_OPT] < test_val {
                if need_group_xstr {
                    self.next_param[LOC_XSTR_OPT] = TA_GROUP_XSTRETCH;
                    self.next_param[LOC_XSTR_GROUP] = grp_stretch;
                }
                if need_group_skew {
                    self.next_param[LOC_SKEW_OPT] = TA_GROUP_SKEW;
                    self.next_param[LOC_SKEW_GROUP] = grp_skew;
                }
                self.was_tested[LOC_XSTR_OPT] = test_val;
                if self
                    .compare_next_param(&format!("{type_text}grouping of local X-stretch and skew"))
                    > 0
                    && take_one_step != 0
                {
                    return 1;
                }
            }
        }

        // Turn off both variables
        if (self.next_param[LOC_XSTR_OPT] != 0 || self.next_param[LOC_SKEW_OPT] != 0)
            && self.was_tested[LOC_XSTR_OPT] < 4
        {
            self.next_param[LOC_XSTR_OPT] = 0;
            self.next_param[LOC_SKEW_OPT] = 0;
            self.was_tested[LOC_XSTR_OPT] = 4;
            if self.compare_next_param("not solving for local X-stretch and skew") > 0 {
                return 1;
            }
        }

        0
    }

    /// Matches `cvTestLocalTiltRotMag` (`restrictalign:655`).
    fn cv_test_local_tilt_rot_mag(
        &mut self,
        opt: usize,
        group_opt: i64,
        label: &str,
        take_one_step: i32,
    ) -> i32 {
        let def_group = self.new_param[opt + 1];
        let medium_group = (def_group * 2).min((def_group + self.big_grouping).div_euclid(2));

        for (group, test_val, type_text) in [
            (def_group, 1, ""),
            (medium_group, 2, "more "),
            (self.big_grouping, 3, "large "),
        ] {
            // Test grouping if not, or increase it
            if self.new_param[opt] != 0
                && (self.new_param[opt] != group_opt || self.new_param[opt + 1] < group)
                && self.was_tested[opt] < test_val
            {
                self.next_param[opt] = group_opt;
                self.next_param[opt + 1] = group;
                self.was_tested[opt] = test_val;
                if self.compare_next_param(&format!("{type_text}grouping of local {label}")) > 0
                    && take_one_step != 0
                {
                    return 1;
                }
            }
        }

        if self.next_param[opt] != 0 && self.was_tested[opt] < 4 {
            self.next_param[opt] = 0;
            self.was_tested[opt] = 4;
            if self.compare_next_param(&format!("not solving for local {label}")) > 0 {
                return 1;
            }
        }

        0
    }
}

/// Matches `fillAndTestOrder` (`restrictalign:92`).
///
/// Fixed in translation: without `-cross` the source calls this with the
/// `None` that `PipGetIntegerArray` returns for an absent `-order`, and
/// `len(None)` raises `TypeError`; here an absent entry leaves the default
/// order in place.
fn fill_and_test_order(order_arr: Option<&[i64]>, ord_use: &mut [i64], name: &str) {
    let order_arr = order_arr.unwrap_or(&[]);
    for ind in 0..order_arr.len().min(ord_use.len()) {
        ord_use[ind] = order_arr[ind];
    }
    for &action in ord_use.iter() {
        if ord_use.iter().filter(|&&a| a == action).count() > 1 {
            exit_error(&format!(
                "{name} {action} is in the order list more than once"
            ));
        }
        if action < 1 || action > ord_use.len() as i64 {
            exit_error(&format!(
                "{name} entry {action} is outside the allowed range of 1 to {}",
                ord_use.len()
            ));
        }
    }
}

/// Matches `getNumUnchoppedConts` (`restrictalign:677`).
///
/// Fixed in translation: a contour with no points after another contour made
/// the source index `newCont[0]` of an empty list (`IndexError`); an empty
/// contour here matches no point, so it counts as a new track.
fn get_num_unchopped_conts(model_file: &str) -> (i64, i64) {
    let mod_lines = match run_cmd(
        &format!("imodinfo -a \"{model_file}\""),
        None,
        None,
        None,
        &[],
    ) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => exit_from_imod_error(PROGNAME),
    };
    let mut prev_cont: Vec<String> = Vec::new();
    let mut true_beads: i64 = 0;
    let mut true_points: i64 = 0;
    let mut ind: usize = 0;

    // loop on lines to find contours
    while ind < mod_lines.len() {
        let line = &mod_lines[ind];
        ind += 1;
        if line.starts_with("contour ") {
            // Found a contour, make sure it can all be read
            let lsplit: Vec<&str> = line.split_whitespace().collect();
            if lsplit.len() != 4 {
                continue;
            }
            let Some(num_pts) = py_int(lsplit[3]) else {
                exit_error(
                    &("Converting # of points to integer in output from imodinfo -a to "
                        .to_owned()
                        + "analyze for chopped contours"),
                )
            };
            if ind as i64 + num_pts > mod_lines.len() as i64 {
                exit_error(
                    &("Output from imodinfo -a to analyze for chopped contours is ".to_owned()
                        + "truncated"),
                );
            }

            // Shallow copy the lines now
            let start = ind.min(mod_lines.len());
            let end = (ind as i64 + num_pts).max(start as i64) as usize;
            let new_cont: Vec<String> = mod_lines[start..end].to_vec();
            true_points += num_pts;
            if !prev_cont.is_empty() {
                // Look for duplicate point in previous contour, if found, subtract from
                // total points
                let mut found = false;
                for jnd in 0..prev_cont.len() {
                    let point = &prev_cont[jnd];
                    if new_cont.first() == Some(point) {
                        true_points -= (prev_cont.len() - jnd) as i64;
                        found = true;
                        break;
                    }
                }
                if !found {
                    // ELSE ON FOR
                    true_beads += 1;
                }
            } else {
                true_beads += 1;
            }

            prev_cont = new_cont;
            ind = (ind as i64 + num_pts).max(0) as usize;
        }
    }

    (true_beads, true_points)
}

/// Integers of an `optionValue(..., INT_VALUE, ...)` return.
fn int_values(value: Option<OptionValue>) -> Option<Vec<i64>> {
    match value {
        Some(OptionValue::Integers(values)) => {
            Some(values.iter().map(|value| *value as i64).collect())
        }
        _ => None,
    }
}

/// The script's top level (`restrictalign:723-1565`).  Returns the status of
/// its `sys.exit`; `exitError` paths end the process.
pub fn restrictalign(arguments: &[OsString]) -> i32 {
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

    // Fallbacks from ../manpages/autodoc2man 3 1 restrictalign
    let options: Vec<String> = [
        "align:AlignCommandFile:FN:",
        "fiducials:NumberOfFiducials:I:",
        "views:NumberOfViews:I:",
        "target:TargetMeasurementRatio:F:",
        "minimum:MinMeasurementRatio:F:",
        "cross:UseCrossValidation:I:",
        "local:LocalAlignValidation:I:",
        "benefit:MinRobustBenefit:F:",
        "order:OrderOfRestrictions:IA:",
        "cvorder:CrossValTestOrder:IA:",
        "onestep:OneStepPerVariableTest:I:",
        "permute:TestPermutations:IA:",
        "skipbeam:SkipBeamTiltWithOneRot:B:",
        "trial:TrialMode:B:",
        "verbose:VerboseOutput:B:",
        ":PID:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    // default in adoc = 1, 4, 3, 2, 5 will override this
    let mut order: Vec<i64> = vec![
        RES_GROUP_ROTS,
        RES_GROUP_MAGS,
        RES_FIX_TILTS,
        RES_ONE_ROT,
        RES_FIX_MAGS,
    ];
    let mut cv_order: Vec<i64> = vec![
        CV_TEST_STRETCH,
        CV_TEST_XTILT,
        CV_TEST_TILT,
        CV_TEST_ROT,
        CV_TEST_MAG,
        CV_TEST_SINGLES,
    ];
    let mut ra = Ra::default();
    ra.final_changes = false;
    let mut one_step_per_var: i32 = 1;

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, PROGNAME, 1, 1, 0);

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    let comfile = pip_get_in_out_file("AlignCommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if comfile.is_empty() {
        exit_error("The name of the Tiltalign command file must be entered");
    }

    let mut com_lines = read_text_file(&comfile, None, false, None).unwrap_or_default();

    // Cross-validation
    ra.cross_validate = pip_get_integer("UseCrossValidation", 0).unwrap_or(0);
    ra.test_local = pip_get_integer("LocalAlignValidation", 0).unwrap_or(0);
    let min_benefit_for_eval = pip_get_float("MinRobustBenefit", 2.).unwrap_or(2.);
    ra.test_local_area = ra.test_local > 1;
    let test_local_vars = ra.test_local > 0 && ra.test_local != 2;

    // Get the order of actions and check it
    if ra.cross_validate != 0 {
        if std::env::var_os("TILTALIGN_SKIP_CROSS_VAL").is_some_and(|v| !v.is_empty()) {
            // SAFETY: the script is single-threaded at this point, as the
            // Python is.
            unsafe { std::env::remove_var("TILTALIGN_SKIP_CROSS_VAL") };
        }
        let order_arr = pip_get_integer_array("CrossValTestOrder", 0);
        if pip_get_err_no() == 0 {
            fill_and_test_order(order_arr.as_deref(), &mut cv_order, "Variable");
        }
    } else {
        let order_arr = pip_get_integer_array("OrderOfRestrictions", 0);
        fill_and_test_order(order_arr.as_deref(), &mut order, "Restriction");
    }

    let permute_entry = pip_get_integer_array("TestPermutations", 0);
    let mut do_permute = 1 - pip_get_err_no();
    let mut permute_arr: Vec<i64> = permute_entry.unwrap_or_default();
    if do_permute != 0 {
        one_step_per_var = 0;
    }
    one_step_per_var =
        pip_get_integer("OneStepPerVariableTest", one_step_per_var).unwrap_or(one_step_per_var);

    if do_permute != 0 {
        if ra.cross_validate == 0 {
            exit_error("Permutations can be done only with cross-validation");
        }
        if one_step_per_var != 0 {
            exit_error("Permutations cannot be done with taking one step per variable");
        }
        for &action in &permute_arr {
            if permute_arr.iter().filter(|&&a| a == action).count() > 1 {
                exit_error(&format!(
                    "Variable {action} is in the permutation list more than once"
                ));
            }
            if action < 1 || action > cv_order.len() as i64 {
                exit_error(&format!(
                    "Permutation entry {action} is outside the allowed range of 1 to {}",
                    cv_order.len()
                ));
            }
        }
    }

    ra.num_views = pip_get_integer("NumberOfViews", -1).unwrap_or(-1) as i64;
    if pip_get_err_no() == 0 && ra.num_views < 1 {
        exit_error("The number of views entered must be positive");
    }

    ra.num_beads = pip_get_integer("NumberOfFiducials", -1).unwrap_or(-1) as i64;
    if pip_get_err_no() == 0 && ra.num_beads < 1 {
        exit_error("The number of beads entered must be positive");
    }
    let target_ratio = pip_get_float("TargetMeasurementRatio", 3.6).unwrap_or(3.6); // default in adoc
    let min_ratio = pip_get_float("MinMeasurementRatio", 3.2).unwrap_or(3.2); // default in adoc
    ra.skip_beam_tilt = pip_get_boolean("SkipBeamTiltWithOneRot", 0).unwrap_or(0) != 0;

    // Other options
    let trial_mode = pip_get_boolean("TrialMode", 0).unwrap_or(0);
    ra.verbose = pip_get_boolean("VerboseOutput", 0).unwrap_or(0) != 0;
    let model_file = match option_value(&com_lines, "ModelFile", 0, false, 0, None, None) {
        Some(OptionValue::String(value)) => value,
        _ => String::new(),
    };

    // Get the model file and number of beads from the model file
    ra.num_points = 0;
    let mut patch_track = false;
    if ra.num_beads < 0 {
        if model_file.is_empty() {
            exit_error(&format!(
                "The number of fiducials was not entered and a ModelFile entry cannot be found in {comfile}"
            ));
        }
        if !std::path::Path::new(&model_file).exists() {
            exit_error(&format!(
                "The number of fiducials was not entered and the model file {model_file} does not exist"
            ));
        }
        let info_lines = match run_cmd(&format!("imodinfo \"{model_file}\""), None, None, None, &[])
        {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => exit_error(&format!(
                "The number of fiducials was not entered and an error occurred running imodinfo on the model file {model_file}"
            )),
        };

        ra.num_beads = 0;
        let con_match = regex::Regex::new(r"^\s*CONTOUR\s#\S*\s*([0-9]*)\s*points.*")
            .expect("contour regular expression");
        for line in &info_lines {
            if line.starts_with("# NAME") && line.contains("Patch Tracking Model") {
                patch_track = true;
            }
            if con_match.is_match(line) {
                let cont_points = convert_to_integer(
                    &con_match.replace(line, "${1}"),
                    "number or contour points in imodinfo output",
                ) as i64;
                if cont_points > 1 {
                    ra.num_beads += 1;
                    ra.num_points += cont_points;
                }
            }
        }

        if ra.num_beads == 0 {
            exit_error(&format!(
                "The number of fiducials was not entered and the model file {model_file} has no contours with more than one point"
            ));
        }
    }

    // Determine the current state of the parameters
    ra.orig_param = vec![None; 25];

    for (opt, prefix) in [
        (ROT_OPT, "Rot"),
        (MAG_OPT, "Mag"),
        (TILT_OPT, "Tilt"),
        (SKEW_OPT, "Skew"),
        (XSTRETCH_OPT, "XStretch"),
        (XTILT_OPT, "XTilt"),
        (LOC_ROT_OPT, "LocalRot"),
        (LOC_MAG_OPT, "LocalMag"),
        (LOC_TILT_OPT, "LocalTilt"),
        (LOC_SKEW_OPT, "LocalSkew"),
        (LOC_XSTR_OPT, "LocalXStretch"),
    ] {
        ra.orig_param[opt] = int_values(option_value(
            &com_lines,
            &format!("{prefix}Option"),
            1,
            false,
            1,
            None,
            None,
        ))
        .map(|values| values[0]);
        ra.orig_param[opt + 1] = int_values(option_value(
            &com_lines,
            &format!("{prefix}DefaultGrouping"),
            1,
            false,
            1,
            None,
            None,
        ))
        .map(|values| values[0]);
    }

    ra.orig_param[BEAM_TILT_OPT] = int_values(option_value(
        &com_lines,
        "BeamTiltOption",
        1,
        false,
        1,
        None,
        None,
    ))
    .map(|values| values[0]);
    ra.orig_param[PROJ_STRETCH] =
        match option_value(&com_lines, "ProjectionStretch", 3, false, 0, None, None) {
            Some(OptionValue::Boolean(value)) => Some(value as i64),
            _ => None,
        };
    ra.local_align = matches!(
        option_value(&com_lines, "LocalAlignments", 3, false, 0, None, None),
        Some(OptionValue::Boolean(true))
    );
    ra.robust_align = matches!(
        option_value(&com_lines, "RobustFitting", 3, false, 0, None, None),
        Some(OptionValue::Boolean(true))
    );
    let mut image_file = match option_value(&com_lines, "ImageFile", 0, false, 0, None, None) {
        Some(OptionValue::String(value)) => value,
        _ => String::new(),
    };
    ra.orig_required = vec![0, 0];
    ra.new_required = vec![0, 0];
    ra.orig_area_or_num = vec![0, 0];
    ra.new_area_or_num = vec![0, 0];
    let mut robust_off_for_eval = false;
    let mut robust_off = String::new();
    if ra.test_local != 0 && !ra.local_align {
        exit_error("Local alignments must be turned on to test them");
    }

    if ra.test_local != 0 && do_permute != 0 {
        do_permute = 0;
        prnstr(
            "WARNING: permutations are not tested with local alignments",
            "\n",
            false,
        );
    }

    let mut permute_inds: Vec<usize> = Vec::new();
    let mut permute_list: Vec<Vec<i64>> = Vec::new();
    if do_permute != 0 {
        // For permutations, remove items from array that are not being solved for
        if permute_arr.contains(&CV_TEST_STRETCH)
            && ra.orig_param[XSTRETCH_OPT] == Some(0)
            && ra.orig_param[SKEW_OPT] == Some(0)
        {
            let pos = permute_arr
                .iter()
                .position(|&a| a == CV_TEST_STRETCH)
                .unwrap();
            permute_arr.remove(pos);
        }

        for (opt, var) in [
            (ROT_OPT, CV_TEST_ROT),
            (TILT_OPT, CV_TEST_TILT),
            (MAG_OPT, CV_TEST_MAG),
        ] {
            if permute_arr.contains(&var) && ra.orig_param[opt] == Some(0) {
                let pos = permute_arr.iter().position(|&a| a == var).unwrap();
                permute_arr.remove(pos);
            }
        }

        if permute_arr.len() < 2 {
            do_permute = 0;
        } else {
            // Make cross-index from variable in order array to position in permutation
            // array
            for &action in &permute_arr {
                for ind in 0..cv_order.len() {
                    if action == cv_order[ind] {
                        permute_inds.push(ind);
                        break;
                    }
                }
            }
        }

        // `itertools.permutations(permuteArr)`: every ordering, in lexicographic
        // order of the element positions
        let count = permute_arr.len();
        let mut indices: Vec<usize> = (0..count).collect();
        loop {
            permute_list.push(indices.iter().map(|&i| permute_arr[i]).collect());
            let Some(pivot) = (1..count).rev().find(|&i| indices[i - 1] < indices[i]) else {
                break;
            };
            let swap = (pivot..count)
                .rev()
                .find(|&j| indices[j] > indices[pivot - 1])
                .unwrap();
            indices.swap(pivot - 1, swap);
            indices[pivot..].reverse();
        }
    }

    // Get the needed options when testing local area size
    let mut image_size: Option<Vec<i64>> = None;
    if ra.test_local_area {
        let orig_required = int_values(option_value(
            &com_lines,
            "MinFidsTotalAndEachSurface",
            1,
            false,
            0,
            None,
            None,
        ));
        let target_size = int_values(option_value(
            &com_lines,
            "TargetPatchSizeXandY",
            1,
            false,
            0,
            None,
            None,
        ))
        .filter(|v| !v.is_empty());
        let num_areas = int_values(option_value(
            &com_lines,
            "NumberOfLocalPatchesXandY",
            1,
            false,
            0,
            None,
            None,
        ))
        .filter(|v| !v.is_empty());
        image_size = int_values(option_value(
            &com_lines,
            "ImageSizeXandY",
            1,
            false,
            0,
            None,
            None,
        ))
        .filter(|v| !v.is_empty());
        let orig_required = match orig_required {
            Some(values) if values.len() >= 2 => values,
            _ => exit_error(
                &("Option for required number of fiducials in local areas not found ".to_owned()
                    + "or has only one value"),
            ),
        };
        ra.orig_required = orig_required;
        if ra.orig_required[0] + 1 > ra.num_beads.div_euclid(2) {
            exit_error("There are not enough beads to evaluate areas by requiring more beads");
        }
        if target_size.is_some() && num_areas.is_some() {
            exit_error(
                &("Command file has both TargetPatchSizeXandY and ".to_owned()
                    + "NumberOfLocalPatchesXandY options"),
            );
        }
        if target_size.is_none() && num_areas.is_none() {
            exit_error(
                &("Command file has neither TargetPatchSizeXandY nor ".to_owned()
                    + "NumberOfLocalPatchesXandY option"),
            );
        }
        ra.target_size = target_size.is_some();
        if let Some(target_size) = target_size {
            if target_size.len() < 2 {
                exit_error("TargetPatchSizeXandY entry in command file has only one value");
            }
            ra.orig_area_or_num = target_size;
        }

        if let Some(num_areas) = num_areas {
            if num_areas.len() < 2 {
                exit_error("NumberOfLocalPatchesXandY entry in command file has only one value");
            }
            ra.orig_area_or_num = num_areas;
        }

        ra.new_required = ra.orig_required.clone();
        ra.new_area_or_num = ra.orig_area_or_num.clone();
    }

    // Get the image file and number of views from it if not entered
    // Or get the image size if that is needed
    // Or fall back to ImageSize Entry
    // or fall back to the imodinfo max values
    //
    // Fixed in translation: the source tests `len(imageSize) > 2` on the
    // two-value ImageSizeXandY entry, so that fallback could never apply; the
    // entry is used here when it has both values.
    let mut need_size = ra.test_local_area && ra.target_size;
    let im_size_ok = !need_size || image_size.as_ref().is_some_and(|v| v.len() > 1);
    let mut full_nx: i64 = 0;
    let mut full_ny: i64 = 0;
    if ra.num_views < 0 || need_size {
        let mut mess = "Need to determine ".to_owned();
        if ra.num_views < 0 {
            mess += "# of views ";
        }
        if need_size && !im_size_ok {
            if ra.num_views < 0 {
                mess += "and ";
            }
            mess += "image size";
        }
        mess += "; ";

        image_file = match option_value(&com_lines, "ImageFile", 0, false, 0, None, None) {
            Some(OptionValue::String(value)) => value,
            _ => String::new(),
        };
        full_nx = 0;
        if !image_file.is_empty() && std::path::Path::new(&image_file).exists() {
            match get_mrc_size(&image_file) {
                Ok((nx, ny, nz)) => {
                    full_nx = nx as i64;
                    full_ny = ny as i64;
                    if ra.num_views < 0 {
                        ra.num_views = nz as i64;
                    }
                }
                Err(_) => {
                    mess +=
                        &format!("an error occurred running header on image file {image_file}, ");
                }
            }
        } else if image_file.is_empty() {
            mess += &format!("there is no ImageFile entry in {comfile}, ");
        } else {
            mess += &format!("image file {image_file} does not exist, ");
        }

        if full_nx == 0 && need_size {
            if let Some(size) = image_size.as_ref().filter(|v| v.len() > 1) {
                full_nx = size[0];
                full_ny = size[1];
                need_size = false;
            }
        }

        if ra.num_views < 0 || need_size {
            if model_file.is_empty() {
                mess += &format!("and a ModelFile entry cannot be found in {comfile}");
            } else {
                match run_cmd(
                    &format!("imodinfo -a \"{model_file}\""),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(mod_lines) => {
                        let mut value_error = false;
                        for line in mod_lines.unwrap_or_default() {
                            if line.starts_with("max") {
                                let lsplit: Vec<&str> = line.split_whitespace().collect();
                                if lsplit.len() >= 4 {
                                    if need_size {
                                        match (py_int(lsplit[1]), py_int(lsplit[2])) {
                                            (Some(nx), Some(ny)) => {
                                                full_nx = nx;
                                                full_ny = ny;
                                                need_size = false;
                                            }
                                            _ => {
                                                value_error = true;
                                                break;
                                            }
                                        }
                                    }
                                    if ra.num_views < 0 {
                                        match py_int(lsplit[3]) {
                                            Some(nz) => ra.num_views = nz,
                                            None => {
                                                value_error = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        if value_error {
                            mess +=
                                "and there was an error converting max values from the model file";
                        } else if ra.num_views < 0 || need_size {
                            // Fixed in translation: the source evaluates this string
                            // without `mess +=`, so it never reached the message
                            mess += &format!(
                                "and the \"max\" line could not be found in imodinfo output on {model_file}"
                            );
                        }
                    }
                    Err(_) => {
                        mess += &format!("and there was an error running imodinfo on {model_file}");
                    }
                }
            }
        }

        if ra.num_views < 0 || need_size {
            exit_error(&mess);
        }
    }

    // If patch tracking, get true number of contours
    if patch_track {
        (ra.num_beads, ra.num_points) = get_num_unchopped_conts(&model_file);
    }

    // if no model, now set number of points assuming a complete model
    if ra.num_points == 0 {
        ra.num_points = ra.num_beads * ra.num_views;
    }

    if ra.verbose && ra.num_points != 0 {
        if patch_track {
            prnstr(
                &format!(
                    "{} full tracks, {} unique points, {:.1} average points per view",
                    ra.num_beads,
                    ra.num_points,
                    ra.num_points as f64 / ra.num_views as f64
                ),
                "\n",
                false,
            );
        } else {
            prnstr(
                &format!(
                    "{} beads, {} total points, {:.1} average points per view",
                    ra.num_beads,
                    ra.num_points,
                    ra.num_points as f64 / ra.num_views as f64
                ),
                "\n",
                false,
            );
        }
    }

    // modify com lines to use ImageSize if image file doesn't exist
    //
    // Fixed in translation: the source tests `'#ImageSizeXnadY'`, a typo that
    // never matches, so the size entry stayed commented out while the image
    // file entry was removed.
    if !image_file.is_empty()
        && !std::path::Path::new(&image_file).exists()
        && int_values(option_value(
            &com_lines,
            "#ImageSizeXandY",
            1,
            false,
            0,
            None,
            None,
        ))
        .is_some_and(|v| !v.is_empty())
    {
        let mut temp_lines: Vec<String> = Vec::new();
        for line in &com_lines {
            if line.starts_with("ImageFile") {
                continue;
            }
            let mut line = line.clone();
            if line.starts_with("#ImageSizeXandY") {
                line = line.chars().skip(1).collect();
            }
            temp_lines.push(line);
        }

        com_lines = temp_lines;
    }

    // Copy parameter and change None entry to 0
    ra.new_param = ra
        .orig_param
        .iter()
        .map(|value| value.unwrap_or(0))
        .collect();

    let one_bead = ra.num_beads == 1;
    let mut boosted_ratio = false;

    let mut new_ratio = ra.measured_to_unknown(&ra.new_param);
    if ra.verbose {
        prnstr(
            &format!("Original estimated ratio of measurements to unknowns: {new_ratio:.2}"),
            "\n",
            false,
        );
    }

    // batchruntomo looking for 'No restriction'
    if new_ratio >= target_ratio && ra.cross_validate == 0 {
        prnstr(
            &format!("{PROGNAME}: No restriction of parameters needed"),
            "\n",
            false,
        );
        return 0;
    }

    if new_ratio < target_ratio && (ra.cross_validate == 0 || ra.num_beads < MIN_BEADS_FOR_CV_ONLY)
    {
        for order_ind in -1..order.len() as i64 {
            let new_param = ra.new_param.clone();
            let mut next_param = new_param.clone();
            let restrict;

            // Turn off hard variables on first round
            if order_ind < 0 {
                if new_param[SKEW_OPT] != 0 {
                    next_param[SKEW_OPT] = 0;
                }
                if new_param[XSTRETCH_OPT] != 0 {
                    next_param[XSTRETCH_OPT] = 0;
                }
                if new_param[XTILT_OPT] != 0
                    && (new_param[XTILT_GROUP] < ra.num_views || ra.num_beads < 3)
                {
                    next_param[XTILT_OPT] = 0;
                }
                if new_param[TILT_OPT] == TA_ALL_TILT
                    || (new_param[TILT_OPT] == TA_GROUP_TILT
                        && new_param[TILT_GROUP] < MIN_TILT_GROUPING)
                {
                    next_param[TILT_OPT] = TA_GROUP_TILT;
                    next_param[TILT_GROUP] = next_param[TILT_GROUP].max(MIN_TILT_GROUPING);
                }
                restrict = 0;
            } else {
                restrict = order[order_ind as usize];
            }

            // Handle switching to grouped rots of minimum group size and to one rot
            if restrict == RES_GROUP_ROTS
                && (new_param[ROT_OPT] == TA_ALL_ROT
                    || (new_param[ROT_OPT] == TA_GROUP_ROT
                        && new_param[ROT_GROUP] < MIN_ROT_GROUPING))
            {
                next_param[ROT_OPT] = TA_GROUP_ROT;
                next_param[ROT_GROUP] = MIN_ROT_GROUPING.max(next_param[ROT_GROUP]);
            } else if restrict == RES_ONE_ROT && new_param[ROT_OPT] > 0 {
                next_param[ROT_OPT] = TA_ONE_ROT;
                if !ra.skip_beam_tilt && ra.num_beads >= MIN_BEADS_FOR_BEAM_TILT {
                    next_param[BEAM_TILT_OPT] = 2;
                }
            }

            // Handle fixing tilts
            if restrict == RES_FIX_TILTS || one_bead {
                next_param[TILT_OPT] = 0;
            }

            // Handle fixing mags or grouping them
            if restrict == RES_FIX_MAGS || one_bead {
                next_param[MAG_OPT] = 0;
            } else if restrict == RES_GROUP_MAGS
                && (new_param[MAG_OPT] == TA_ALL_MAG
                    || (new_param[MAG_OPT] == TA_GROUP_MAG
                        && new_param[MAG_GROUP] < MIN_MAG_GROUPING))
            {
                next_param[MAG_GROUP] = MIN_MAG_GROUPING;
                next_param[MAG_OPT] = TA_GROUP_MAG;
            }

            // Fix everything else if one bead; skip beam tilt and projection stretch for 2
            // or 3
            if one_bead {
                next_param[ROT_OPT] = 0;
            }
            if ra.num_beads < MIN_BEADS_FOR_BEAM_TILT {
                next_param[PROJ_STRETCH] = 0;
                next_param[BEAM_TILT_OPT] = 0;
            }

            // Get the ratio on the next restriction and see if it is good enough or if
            // there is just one bead
            let next_ratio = ra.measured_to_unknown(&next_param);
            ra.next_param = next_param.clone();
            if one_bead || next_ratio >= target_ratio {
                // Adopt the next parameter set if one bead, or last ratio below the
                // minimum, or the next one is closer to target
                if one_bead
                    || new_ratio < min_ratio
                    || (new_ratio - target_ratio).abs() > (next_ratio - target_ratio).abs()
                {
                    ra.new_param = next_param;
                    new_ratio = next_ratio;
                }
                break;
            }

            // Otherwise shift the next set into the "new" set for the next iteration
            ra.new_param = next_param;
            new_ratio = next_ratio;
        }

        if ra.robust_align && new_ratio < min_ratio {
            robust_off = "ratio of measurements to unknowns is too low".to_owned();
        }

        if ra.cross_validate != 0 && !one_bead {
            prnstr(
                "Initial changes were made to boost ratio of measurements to unknowns:",
                "\n",
                false,
            );
            ra.final_changes = true;
            boosted_ratio = true;
            let (param, required, area) = (
                ra.new_param.clone(),
                ra.new_required.clone(),
                ra.new_area_or_num.clone(),
            );
            ra.build_up_sedcom(&param, &robust_off, &required, &area);
            ra.final_changes = false;
        }
    }

    // Variables the source defines only in the cross-validation branch
    let mut robust_orig = false;
    let mut benefit_orig: Option<f64> = None;
    let mut orig_errors: Vec<f64> = Vec::new();
    let mut did_cross_val = false;

    if ra.cross_validate != 0 && ra.num_beads > 1 {
        did_cross_val = true;
        // Extract the input to tiltalign
        ra.ta_lines = Vec::new();
        let mut got_ta = false;
        for line in &com_lines {
            if line.starts_with('$') {
                if got_ta {
                    break;
                } else if line.contains("tiltalign") && line.contains("-St") {
                    got_ta = true;
                }
            } else if got_ta {
                ra.ta_lines.push(line.clone());
            }
        }

        if !got_ta {
            exit_error("Could not find input to Tiltalign in command file");
        }

        // Turn off ridiculous ones with 2 or 3 beads
        if ra.num_beads < MIN_BEADS_FOR_BEAM_TILT {
            ra.new_param[SKEW_OPT] = 0;
            ra.new_param[XSTRETCH_OPT] = 0;
            ra.new_param[XTILT_OPT] = 0;
            ra.new_param[PROJ_STRETCH] = 0;
            ra.new_param[BEAM_TILT_OPT] = 0;
        }

        // Get baseline run
        ra.cum_non_rob_time = 0.;
        ra.cum_robust_time = 0.;
        ra.doing_robust = ra.robust_align && new_ratio >= min_ratio;
        robust_orig = ra.doing_robust;
        let (param, required, area) = (
            ra.new_param.clone(),
            ra.new_required.clone(),
            ra.new_area_or_num.clone(),
        );
        ra.new_errors = ra.do_tiltalign_runs(&param, "initial values", &required, &area);

        // Check if robust was bad and whether benefit is worth evaluating with
        if ra.doing_robust {
            if ra.bad_robust {
                prnstr(
                    &("Turning off robust alignments for evaluation; it failed with initial "
                        .to_owned()
                        + "values"),
                    "\n",
                    false,
                );
                ra.doing_robust = false;
            } else {
                let benefit = 100. * (ra.new_errors[1] - ra.new_errors[3]) / ra.new_errors[1];
                benefit_orig = Some(benefit);
                if benefit < min_benefit_for_eval {
                    if ra.verbose {
                        prnstr(" ", "\n", false);
                    }
                    if benefit < 0. {
                        prnstr(
                            &("Turning off robust alignments for evaluation; it has negative "
                                .to_owned()
                                + "benefit"),
                            "\n",
                            false,
                        );
                    } else {
                        prnstr(
                            &format!(
                                "Turning off robust alignments for evaluation; the benefit is only {benefit:.1}%"
                            ),
                            "\n",
                            false,
                        );
                    }
                    ra.doing_robust = false;
                }
            }

            if !ra.doing_robust {
                robust_off_for_eval = true;
                ra.new_errors =
                    ra.do_tiltalign_runs(&param, "new initial values", &required, &area);
            }
        }

        if ra.new_errors[0] < 0. || ra.new_errors[1] == -2. {
            exit_error("Tiltalign failed on runs with initial parameters");
        }
        if ra.verbose {
            prnstr("", "\n", false);
        }

        ra.big_grouping = ra.num_views.div_euclid(2);
        orig_errors = ra.new_errors.clone();
        let param_for_restart = ra.new_param.clone();
        ra.last_errors = ra.new_errors.clone();
        ra.last_err_diff = 0.;
        ra.next_param = ra.new_param.clone();
        ra.next_required = ra.new_required.clone();
        ra.next_area_or_num = ra.new_area_or_num.clone();
        let mut last_required = ra.new_required[0];
        let mut num_same_min = 0;

        ra.was_tested = vec![0; 25];
        let mut changed = 1;
        let mut areas_finished = false;
        let mut ord_ind: usize = 0;
        while changed != 0 {
            changed = 0;

            // Loop on local tests twice to allow order to be tested
            for local_loop in [0, 1] {
                if ra.test_local_area
                    && ((local_loop != 0 && ra.test_local > 3)
                        || (local_loop == 0 && ra.test_local < 4))
                {
                    // Test local area required number and size
                    while !areas_finished {
                        // Step the required numbers up and increase size or drop number
                        // maybe.  Terminate only when bead number is too high, just fix the
                        // area size or number if that reaches a limit
                        ra.next_required[0] = (last_required + 1)
                            .max((1.1 * last_required as f64).round_ties_even() as i64);
                        if ra.next_required[0] > ra.num_beads.div_euclid(2) {
                            ra.next_required[0] = ra.num_beads.div_euclid(2);
                            if ra.next_required[0] == last_required {
                                areas_finished = true;
                                break;
                            }
                        }

                        last_required = ra.next_required[0];
                        let mut ratio = ra.next_required[0] as f64 / ra.orig_required[0] as f64;
                        ra.next_required[1] =
                            (ra.orig_required[1] as f64 * ratio).round_ties_even() as i64;
                        ratio = ratio.sqrt();

                        if ra.target_size {
                            let last_area_or_num = ra.next_area_or_num.clone();
                            ra.next_area_or_num[0] = 5
                                * (ra.orig_area_or_num[0] as f64 * ratio / 5.).round_ties_even()
                                    as i64;
                            ra.next_area_or_num[1] = 5
                                * (ra.orig_area_or_num[1] as f64 * ratio / 5.).round_ties_even()
                                    as i64;
                            if (ra.next_area_or_num[0] * ra.next_area_or_num[1]) as f64
                                > 0.45 * full_nx as f64 * full_ny as f64
                            {
                                ra.next_area_or_num = last_area_or_num;
                            }
                        } else {
                            ra.next_area_or_num[0] =
                                (ra.orig_area_or_num[0] as f64 / ratio).ceil() as i64;
                            ra.next_area_or_num[1] =
                                (ra.orig_area_or_num[1] as f64 / ratio).ceil() as i64;
                            if ra.next_area_or_num[0] < 2 && ra.next_area_or_num[1] < 2 {
                                if full_nx > full_ny {
                                    ra.next_area_or_num[0] = 2;
                                    ra.next_area_or_num[1] = 1;
                                } else {
                                    ra.next_area_or_num[0] = 1;
                                    ra.next_area_or_num[1] = 2;
                                }
                            }
                        }

                        let better = ra.compare_next_param(&format!(
                            "area requirements {},{} and {},{}",
                            ra.next_required[0],
                            ra.next_required[1],
                            ra.next_area_or_num[0],
                            ra.next_area_or_num[1]
                        ));

                        // Continue until two unique results that are higher than the best one
                        if better > 0 {
                            num_same_min = 0;
                            changed = 1;
                            if one_step_per_var != 0 {
                                break;
                            }
                        }

                        if better < 0 && ra.last_err_diff != 0. {
                            num_same_min += 1;
                            if num_same_min > 1 {
                                areas_finished = true;
                                break;
                            }
                        }
                    }
                }

                // Test local variables
                if test_local_vars
                    && ((local_loop != 0 && ra.test_local < 4)
                        || (local_loop == 0 && ra.test_local > 3))
                {
                    for _loop in 0..cv_order.len() {
                        let var_to_test = cv_order[ord_ind];
                        ord_ind += 1;
                        if ord_ind >= cv_order.len() {
                            ord_ind = 0;
                        }
                        if var_to_test == CV_TEST_STRETCH {
                            changed += ra.cv_test_local_stretch(one_step_per_var);
                            if changed != 0 && one_step_per_var != 0 {
                                break;
                            }
                        }

                        if var_to_test == CV_TEST_TILT {
                            changed += ra.cv_test_local_tilt_rot_mag(
                                LOC_TILT_OPT,
                                TA_GROUP_TILT,
                                "tilt",
                                one_step_per_var,
                            );
                            if changed != 0 && one_step_per_var != 0 {
                                break;
                            }
                        }

                        if var_to_test == CV_TEST_ROT {
                            changed += ra.cv_test_local_tilt_rot_mag(
                                LOC_ROT_OPT,
                                TA_GROUP_ROT,
                                "rotation",
                                one_step_per_var,
                            );
                            if changed != 0 && one_step_per_var != 0 {
                                break;
                            }
                        }

                        if var_to_test == CV_TEST_MAG {
                            changed += ra.cv_test_local_tilt_rot_mag(
                                LOC_MAG_OPT,
                                TA_GROUP_MAG,
                                "magnification",
                                one_step_per_var,
                            );
                            if changed != 0 && one_step_per_var != 0 {
                                break;
                            }
                        }
                    }
                }
            }

            // This breaks the while loop not the loop on which tests to do
            if one_step_per_var == 0 || changed == 0 {
                break;
            }
        }

        // Test permutations
        if do_permute != 0 {
            let mut best_diff = 10000.;
            let mut best_errors: Vec<f64> = Vec::new();
            let mut best_param: Vec<i64> = Vec::new();
            for permutation in &permute_list {
                // Load the permutation into the order list ans start from original state
                for ind in 0..permutation.len() {
                    cv_order[permute_inds[ind]] = permutation[ind];
                }

                ra.new_errors = orig_errors.clone();
                ra.new_param = param_for_restart.clone();
                ra.last_errors = ra.new_errors.clone();
                ra.last_err_diff = 0.;
                ra.next_param = ra.new_param.clone();
                ra.was_tested = vec![0; 25];

                let mut ord_text = String::new();
                for ord_ind in 0..cv_order.len() {
                    ord_text += &format!(" {}", cv_order[ord_ind]);
                }

                // Do classic full test of each variable in turn
                for ord_ind in 0..cv_order.len() {
                    let var_to_test = cv_order[ord_ind];
                    if var_to_test == CV_TEST_STRETCH {
                        ra.cv_test_stretch(0);
                    }

                    if var_to_test == CV_TEST_XTILT {
                        ra.cv_test_xtilt();
                    }

                    if var_to_test == CV_TEST_TILT {
                        ra.cv_test_tilt(0);
                    }

                    if var_to_test == CV_TEST_ROT {
                        ra.cv_test_rotation(0);
                    }

                    if var_to_test == CV_TEST_MAG {
                        ra.cv_test_magnification(0);
                    }

                    if var_to_test == CV_TEST_SINGLES {
                        ra.cv_test_single_vars(0);
                    }
                }

                // Keep track of best one
                let ne = &ra.new_errors;
                let diff = if ne[2] > 0. && orig_errors[2] > 0. {
                    ((ne[1] - orig_errors[1]) / orig_errors[1]
                        + (ne[2] - orig_errors[2]) / orig_errors[2])
                        / 2.
                } else {
                    (ne[0] - orig_errors[0]) / orig_errors[0]
                };
                if diff < 0. {
                    prnstr(
                        &format!(
                            "Permutation{ord_text} reduced leave-out error by {:.1}%",
                            -diff * 100.
                        ),
                        "\n",
                        false,
                    );
                } else {
                    prnstr(
                        &format!("Permutation{ord_text} did not reduce leave-out error"),
                        "\n",
                        false,
                    );
                }

                if diff < best_diff {
                    best_errors = ra.new_errors.clone();
                    best_param = ra.new_param.clone();
                    best_diff = diff;
                }
            }

            prnstr(
                &format!("Biggest change was {:.2}%", -best_diff * 100.),
                "\n",
                false,
            );
            ra.new_param = best_param;
            ra.new_errors = best_errors;
        } else if ra.test_local == 0 {
            // Regular non-local variables in defined order, but with possibility of
            // one step per var
            let mut changed = 1;
            while changed != 0 {
                changed = 0;
                let mut comp_dir = 1.;
                if one_step_per_var > 2 {
                    comp_dir = -1.;
                }

                let mut best_diff = -comp_dir * 10000.;
                let mut best_prev_err_diff = 0.;
                let mut best_last_errors: Vec<f64> = Vec::new();
                let mut best_errors: Vec<f64> = Vec::new();
                let mut best_param: Vec<i64> = Vec::new();
                let mut best_required: Vec<i64> = Vec::new();
                let mut best_area_or_num: Vec<i64> = Vec::new();
                let mut best_var: i64 = 0;
                let mut best_opt: usize = 0;
                let var_names = [
                    "X-stretch and skew",
                    "X-tilt",
                    "tilt",
                    "rotation",
                    "magnification",
                    "single variable",
                ];
                let mut var_opt: usize = 0;
                let mut result: i32 = 0;
                for ord_ind in 0..cv_order.len() {
                    let var_to_test = cv_order[ord_ind];
                    if var_to_test == CV_TEST_STRETCH {
                        var_opt = XSTRETCH_OPT;
                        result = ra.cv_test_stretch(one_step_per_var);
                    }

                    if var_to_test == CV_TEST_XTILT {
                        var_opt = XTILT_OPT;
                        result = ra.cv_test_xtilt();
                    }

                    if var_to_test == CV_TEST_TILT {
                        var_opt = TILT_OPT;
                        result = ra.cv_test_tilt(one_step_per_var);
                    }

                    if var_to_test == CV_TEST_ROT {
                        var_opt = ROT_OPT;
                        result = ra.cv_test_rotation(one_step_per_var);
                    }

                    if var_to_test == CV_TEST_MAG {
                        var_opt = MAG_OPT;
                        result = ra.cv_test_magnification(one_step_per_var);
                    }

                    if var_to_test == CV_TEST_SINGLES {
                        result = ra.cv_test_single_vars(one_step_per_var);
                        var_opt = result as usize;
                    }

                    // If taking one step, mark as changed if there is any result on this
                    // loop, and if doing best or worst step, keep track of that step
                    if result != 0 && one_step_per_var != 0 {
                        changed += 1;
                        if one_step_per_var > 1 {
                            ra.was_tested[var_opt] -= 1;
                        }

                        if one_step_per_var > 1 && comp_dir * ra.cur_err_diff > comp_dir * best_diff
                        {
                            best_prev_err_diff = ra.prev_err_diff;
                            best_last_errors = ra.last_errors.clone();
                            best_errors = ra.new_errors.clone();
                            best_param = ra.new_param.clone();
                            best_required = ra.new_required.clone();
                            best_area_or_num = ra.new_area_or_num.clone();
                            best_diff = ra.cur_err_diff;
                            best_var = var_to_test;
                            best_opt = var_opt;
                            let (e, l, p, pp, r, a) = (
                                ra.prev_err_diff,
                                ra.prev_last_errs.clone(),
                                ra.prev_errors.clone(),
                                ra.prev_param.clone(),
                                ra.prev_required.clone(),
                                ra.prev_area_or_num.clone(),
                            );
                            ra.revert_to_step(e, &l, &p, &pp, &r, &a);
                        }
                    }
                }

                // end of loop, done if not one step, otherwise repeat until no change
                if one_step_per_var == 0 {
                    break;
                }

                // If doing best or worst step, now apply that step
                if changed != 0 && one_step_per_var > 1 {
                    prnstr(
                        &format!(
                            "Changing {} for {:.2}% improvement",
                            var_names[(best_var - 1) as usize],
                            100. * best_diff
                        ),
                        "\n",
                        false,
                    );
                    ra.revert_to_step(
                        best_prev_err_diff,
                        &best_last_errors,
                        &best_errors,
                        &best_param,
                        &best_required,
                        &best_area_or_num,
                    );
                    ra.was_tested[best_opt] += 1;
                }
            }
        }

        if robust_off.is_empty() && ra.robust_align && !ra.doing_robust && !robust_off_for_eval {
            robust_off = "robust fitting failed".to_owned();
        }
    }

    // After all that, are there any changes?  If not, exit
    let mut no_param_change = true;
    if ra.test_local_area {
        no_param_change = ra.new_required[0] == ra.orig_required[0]
            && ra.new_required[1] == ra.orig_required[1]
            && ra.new_area_or_num[0] == ra.orig_area_or_num[0]
            && ra.new_area_or_num[1] == ra.orig_area_or_num[1];
    }

    if no_param_change && (ra.test_local == 0 || test_local_vars) {
        for ind in 0..ra.orig_param.len() {
            let orig = ra.orig_param[ind];
            if (!truthy(orig) && ra.new_param[ind] != 0)
                || (truthy(orig) && orig != Some(ra.new_param[ind]))
            {
                no_param_change = false;
                break;
            }
        }
    }

    if ra.cross_validate != 0 && robust_orig && robust_off.is_empty() {
        let benefit;
        if no_param_change {
            // Fixed in translation: when robust fitting failed on the initial
            // values, the source reads the undefined `benefitOrig` here
            // (`NameError`); robust fitting is turned off as failed instead.
            match benefit_orig {
                Some(value) => benefit = value,
                None => {
                    robust_off = "robust fitting failed".to_owned();
                    benefit = 1.;
                }
            }
        } else if !robust_off_for_eval {
            benefit = 100. * (ra.new_errors[1] - ra.new_errors[3]) / ra.new_errors[1];
        } else {
            let final_errors = ra.new_errors.clone();
            ra.doing_robust = true;
            let save_test_loc = ra.test_local;
            if ra.local_align && ra.test_local == 0 {
                ra.test_local = 1;
            }
            let (param, required, area) = (
                ra.new_param.clone(),
                ra.new_required.clone(),
                ra.new_area_or_num.clone(),
            );
            ra.new_errors =
                ra.do_tiltalign_runs(&param, "robust alignment back on", &required, &area);
            benefit = 100. * (ra.new_errors[1] - ra.new_errors[3]) / ra.new_errors[1];
            ra.new_errors = final_errors;
            ra.test_local = save_test_loc;
            if ra.verbose {
                prnstr(&format!(" - benefit now {benefit:.1}%"), "\n", false);
            }
        }

        if benefit <= 0. {
            robust_off = format!("robust fitting gives no benefit ({benefit:.1}%)");
        }
    }

    if no_param_change && robust_off.is_empty() {
        prnstr(
            &format!("{PROGNAME}: No restriction of parameters needed"),
            "\n",
            false,
        );
        return 0;
    }

    ra.final_changes = true;

    let mut out_file = comfile.clone();
    let mut change_text = format!("Changed {comfile}");
    if trial_mode != 0 {
        let (root, ext) = super::imodpy::os_path_splitext(&comfile);
        out_file = format!("{root}_new{ext}");
        change_text = format!("Wrote {out_file}");
    }
    // Fixed in translation: with -cross and a single fiducial no
    // cross-validation runs, and the source reached the undefined
    // `origErrors` (`NameError`); that case reports as the ratio-based path.
    if ra.cross_validate != 0 && did_cross_val {
        if no_param_change {
            prnstr(
                &format!("{PROGNAME}: {change_text} because {robust_off}"),
                "\n",
                false,
            );
        } else {
            let ne = &ra.new_errors;
            let diff = if ne[2] > 0. && orig_errors[2] > 0. {
                ((ne[1] - orig_errors[1]) / orig_errors[1]
                    + (ne[2] - orig_errors[2]) / orig_errors[2])
                    / 2.
            } else {
                (ne[0] - orig_errors[0]) / orig_errors[0]
            };
            if diff < 0. {
                prnstr(
                    &format!(
                        "{PROGNAME}: {change_text} to reduce errors of points left out by {:.1}%",
                        -diff * 100.
                    ),
                    "\n",
                    false,
                );
                if boosted_ratio {
                    prnstr(
                        &(" (after initial changes to boost the measurement/unknown ratio -"
                            .to_owned()
                            + "see log file)"),
                        "\n",
                        false,
                    );
                }
            } else {
                prnstr(
                    &format!(
                        "{PROGNAME}: {change_text} just to boost ratio of measurements to unknowns"
                    ),
                    "\n",
                    false,
                );
            }
        }
    } else if no_param_change {
        prnstr(
            &format!(
                "{PROGNAME}: {change_text} given the measured/unknown ratio of ~{new_ratio:.1}"
            ),
            "\n",
            false,
        );
    } else {
        prnstr(
            &format!(
                "{PROGNAME}: {change_text} to achieve measured/unknown ratio of ~{new_ratio:.1}"
            ),
            "\n",
            false,
        );
    }

    // Now that we know what to do, build up the sed command for the changes and list them
    let (param, required, area) = (
        ra.new_param.clone(),
        ra.new_required.clone(),
        ra.new_area_or_num.clone(),
    );
    ra.build_up_sedcom(&param, &robust_off, &required, &area);
    make_backup_file(&out_file);
    let _ = pysed(
        &ra.sedcom,
        PysedSrc::Lines(&com_lines),
        Some(&out_file),
        false,
        '/',
        false,
    );
    if ra.cross_validate != 0 {
        prnstr(
            "Rerunning the alignment with the final file    [rsa1]",
            "\n",
            false,
        );
        if run_cmd(&format!("submfg {out_file}"), None, None, None, &[]).is_err() {
            exit_from_imod_error(PROGNAME);
        }
    }
    let _ = ra.cum_robust_time;
    0
}
