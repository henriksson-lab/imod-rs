//! Translation of `IMOD/pysrc/autofidseed`: finds a fiducial seed model.
//!
//! A Python command script: its nine functions are translated one for one
//! as methods of [`Afs`], which holds the module globals they read and
//! write, and its top level is [`autofidseed`], translated statement by
//! statement.
//!
//! Programs run: `imodfindbeads`, `imodmop`, `newstack`, `tiltxcorr`,
//! `clipmodel`, `beadtrack`, `point2model`, `sortbeadsurfs` and
//! `pickbestseed`, all ours and all in this process.  Where the script
//! parses a program's printed report, the program is called directly and
//! records the values (`imodfindbeads_recording`, `clipmodel_recording`,
//! `pickbestseed_recording`; `CLAUDE.md`, "Direct calls instead of parsed
//! output"), each rounded as it was printed; the rest go through `runcmd`,
//! which runs a command of ours in process.  Still a text parse: the `View`
//! lines of `tiltxcorr` when finding shifts near zero tilt (no recording
//! entry point in `tiltxcorr`).
//!
//! Python semantics carried explicitly: `//` floors, every float is a
//! double, `str()`/`'{}'.format` of a float is its `repr`
//! ([`py_str_float`]), `'{:.Nf}'` is C's `%.Nf`, `round()` rounds half to
//! even, `int()` of a float truncates, a negative list index counts from
//! the end.
//!
//! Fixed in translation (`BUGS.md`, autofidseed): `cleanupFiles(comSaved)`
//! passed a string, so every one-character file name in it was removed
//! instead of the saved command file; `pidInfo` could be undefined (or
//! `None`) when a message about temporary files was composed; and the
//! re-run of `pickbestseed` with a lower target replaced the last entry of
//! its input, which is the `WeightsForScore` line whenever weights were
//! added, instead of the target entry.

use super::batchruntomo::py_str_float;
use super::imodpy::{
    MrcInfo, OptionValue, add_imod_bin_ignore_sighup, call_own_program, cleanup_files,
    convert_to_integer, exit_from_imod_error, get_mrc, get_mrc_pixel, get_mrc_size, glob_glob,
    make_current_dir_writable, option_value, os_path_splitext, parse_list, prnstr, read_text_file,
    run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_integer, pip_get_string,
    pip_get_two_integers, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add};
use crate::imod::flib::model::clipmodel::{ClipmodelResult, clipmodel_recording};
use crate::imod::imodutil::imodfindbeads::{
    FindbeadsReport, FindbeadsResult, imodfindbeads_recording,
};
use crate::imod::imodutil::pickbestseed::{PickbestseedResult, pickbestseed_recording};
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use regex::Regex;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};

const PROGNAME: &str = "autofidseed";
const STRING_VALUE: i32 = 0;
const INT_VALUE: i32 = 1;
const FLOAT_VALUE: i32 = 2;
const BOOL_VALUE: i32 = 3;

// Extensions for files that are always cleaned up, files that are left for resuming
// and files that are left for user edification only.  Make all extensions unique
const CLEAN_EXTS: [&str; 6] = [".seed", ".xyzpt", ".surf", ".mop", ".mopxf", ".moptrim"];
const RESUME_EXTS: [&str; 3] = [".track", ".xyzmod", ".elong"];
const INFO_EXTS: [&str; 2] = ["pkmod", ".sortmod"];

/// The module globals of the script that its functions read or assign.
#[derive(Default)]
pub struct Afs {
    prefix: String,
    leavetmp: i32,
    tmpdir: String,
    tmproot: String,
    pid: String,
    /// `pidInfo`: undefined until assigned (`None`), and `None` when an info
    /// file has no `PID` line.
    pid_info: Option<String>,
    track_start: i32,
    track_end: i32,
    mid_view: i32,
    min_tilt_ind: i32,
    tilt_included: Vec<f64>,
    included_views: Vec<i32>,
    seed_views: Vec<i32>,
    num_seed_views: i32,
    num_views: i32,
    image_file: String,
    tmppeak: String,
    bead_size: f64,
    spacing: f64,
    linear: i32,
    peak_fraction: f64,
    guess: i64,
    average_fallback: i64,
    storage_fallback: i64,
    light_beads: bool,
    using_sobel: bool,
    ksigma: f64,
    bound_model: String,
    exclude_areas: i32,
    prexg_file: Option<String>,
    binning: i32,
    adjust_size: i32,
    find_options: String,
    /// `newBeadSize`: the int 0 until `runFindBeads` assigns a float.
    new_bead_size: Option<f64>,
    target_number: f64,
    target_density: f64,
    nx: i32,
    ny: i32,
    vec_angles: [[f64; 2]; 2],
    vec_lengths: [[f64; 2]; 2],
    max_length: f64,
}

/// `list[index]` with Python's negative indexing.
macro_rules! py_index {
    ($list:expr, $index:expr) => {{
        let list = &$list;
        let index: i64 = $index as i64;
        if index < 0 {
            list[(list.len() as i64 + index) as usize]
        } else {
            list[index as usize]
        }
    }};
}

impl Afs {
    /// Matches `cleanup` (`autofidseed:13`): clean up all the temp files
    /// before exiting.
    ///
    /// Defined behaviour: an undefined or `None` `pidInfo` (the source's
    /// NameError / TypeError) reads as this run's `pid`.
    pub fn cleanup(&self, exts: &[&str], pidstr: &str) {
        let left_str_base;
        if self.leavetmp != 0 {
            left_str_base = "All".to_owned();
        } else {
            left_str_base = "Some".to_owned();
            for ext in exts {
                let cleanlist = glob_glob(&format!("{}*{ext}*", self.tmproot));
                cleanup_files(&cleanlist);
            }
        }
        let mut left_str = left_str_base;

        if self.leavetmp != 0 || exts.len() == CLEAN_EXTS.len() {
            let pid_info = self.pid_info.clone().unwrap_or_else(|| self.pid.clone());
            left_str += &format!(" temporary files left as {}/afs{pid_info}*", self.tmpdir);
            if pidstr != pid_info && self.leavetmp != 0 {
                left_str += &format!(" or afs{pidstr}*");
            }
            prnstr(&left_str, "\n", false);
        }
    }

    /// Matches `cleanExitError` (`autofidseed:31`): cleanup files, issue top
    /// message if any, and do the Imod error exit.
    pub fn clean_exit_error(&self, message: &str) -> ! {
        let mut all: Vec<&str> = CLEAN_EXTS.to_vec();
        all.extend(RESUME_EXTS);
        all.extend(INFO_EXTS);
        self.cleanup(&all, &self.pid);
        if !message.is_empty() {
            prnstr(&format!("{}{message}", self.prefix), "\n", false);
        }
        exit_from_imod_error(PROGNAME)
    }

    /// Matches `setBorderSize` (`autofidseed:39`): compute a default border
    /// size if needed.
    pub fn set_border_size(size_for_border: i32, entered: i32) -> i32 {
        let low_border_pix_per_k = 32;
        let high_border_pix_per_k = 24;
        let low_high_breakpoint = 2048;
        if entered > 0 {
            return entered;
        }
        let mut border = (low_border_pix_per_k * size_for_border).div_euclid(1024);
        if size_for_border > low_high_breakpoint {
            border = (high_border_pix_per_k * low_high_breakpoint
                + low_border_pix_per_k * (size_for_border - low_high_breakpoint))
                .div_euclid(1024);
        }
        border
    }

    /// Matches `fillFirstGap` (`autofidseed:50`): fill in first gap in view
    /// list going out from middle in given direction.
    pub fn fill_first_gap(&self, view_list: &[i32], look_dir: i32, midz: i32) -> i32 {
        let range_lo: Vec<i32> = (self.track_start..=midz).rev().collect();
        let range_hi: Vec<i32> = (midz..self.track_end + 1).collect();
        let mut track_range: Vec<i32> = range_lo.iter().chain(range_hi.iter()).copied().collect();
        if look_dir > 0 {
            track_range = range_hi.iter().chain(range_lo.iter()).copied().collect();
        }
        for view in track_range {
            if !view_list.contains(&view) {
                return view;
            }
        }
        // ELSE ON FOR
        exit_error("Inconsistency picking seed views")
    }

    /// Matches `selectSeedViews` (`autofidseed:63`): select set of seed views
    /// given the number desired.
    pub fn select_seed_views(&self, num_needed: i32) -> Vec<i32> {
        let ang_crit = 1.9;
        let tilt = |view: i32| -> f64 { py_index!(self.tilt_included, view) };
        let track_start = self.track_start;
        let track_end = self.track_end;
        let mid_view = self.mid_view;
        let min_tilt_ind = self.min_tilt_ind;

        // If all views in range are needed, just return that to avoid violating assumptions
        // of the logic below
        if num_needed == track_end + 1 - track_start {
            return (track_start..track_end + 1).collect();
        }

        // Get the 3 basic views at least 2 degrees apart inside the range
        let mut seed_views = vec![mid_view - 1, mid_view, mid_view + 1];
        for ind in [0usize, 2] {
            while seed_views[ind] > track_start
                && seed_views[ind] < track_end
                && (tilt(mid_view) - tilt(seed_views[ind])).abs() < ang_crit
            {
                seed_views[ind] += ind as i32 - 1;
            }
        }

        // If the min tilt view is not in the list, force it in
        if !seed_views.contains(&min_tilt_ind) {
            // Find which one is closest to the zero tilt
            let mut sub_ind: i32 = 0;
            for ind in [1i32, 2] {
                if tilt(seed_views[sub_ind as usize]).abs() > tilt(seed_views[ind as usize]).abs() {
                    sub_ind = ind;
                }
            }

            // put the min tilt view in that spot and set up to walk out from there in steps
            seed_views[sub_ind as usize] = min_tilt_ind;
            let dir_list: [i32; 2];
            let end_inds: [i32; 2];
            if sub_ind == 1 {
                dir_list = [-1, 1];
                end_inds = [track_start, track_end];
            } else if sub_ind == 0 {
                dir_list = [1, 1];
                end_inds = [track_end - 1, track_end];
            } else {
                dir_list = [-1, -1];
                end_inds = [track_start + 1, track_start];
            }

            // Take two steps in the listed directions and with the proper end point for
            // each step and find the next view at the correct separation
            for loop_ in [0usize, 1] {
                let ind_dir = dir_list[loop_];
                let ind = sub_ind + ind_dir;
                seed_views[ind as usize] = seed_views[sub_ind as usize] + ind_dir;
                while seed_views[ind as usize] != end_inds[loop_]
                    && (tilt(seed_views[sub_ind as usize]) - tilt(seed_views[ind as usize])).abs()
                        < ang_crit
                {
                    seed_views[ind as usize] += ind_dir;
                }
                if dir_list[0] == dir_list[1] {
                    sub_ind += ind_dir;
                }
            }
        }

        let mut check_dir: i32 = -1;
        let mut end_view = seed_views[0];
        let mut divided = [0, 0];
        let mid_z = seed_views[1];
        for loop_ in [0usize, 1] {
            if seed_views.len() as i32 >= num_needed {
                return seed_views;
            }

            let mut view_try = end_view + check_dir;

            // Try to extend by another 2 degrees
            while view_try > track_start
                && view_try < track_end
                && (tilt(end_view) - tilt(view_try)).abs() < ang_crit
            {
                view_try += check_dir;
            }

            // Find insertion between the two with most balanced intervals
            let mut min_view = -1;
            let mut min_interval: f64 = 0.;
            if end_view - check_dir != mid_z {
                let check_lo = (end_view - check_dir).min(mid_z + check_dir);
                let check_hi = (end_view - check_dir).max(mid_z + check_dir) + 1;
                let mut min_diff = 1000.;
                for view in check_lo..check_hi {
                    let interval1 = (tilt(end_view) - tilt(view)).abs();
                    let interval2 = (tilt(mid_z) - tilt(view)).abs();
                    let diff = (interval1 - interval2).abs();
                    if diff < min_diff {
                        min_diff = diff;
                        min_view = view;
                        min_interval = interval1.min(interval2);
                    }
                }
            }

            // If there was any, and either the extension is invalid or the minimum
            // interval by dividing this range is larger than the extension interval,
            // divide range
            if min_view >= 0
                && (view_try < track_start
                    || view_try > track_end
                    || min_interval > 1.5 * (tilt(end_view) - tilt(view_try)).abs())
            {
                view_try = min_view;
                divided[loop_] = 1;
            }

            // And if there is still nothing valid, just fill first gap from this direction
            if view_try < track_start || view_try > track_end {
                view_try = self.fill_first_gap(&seed_views, check_dir, mid_z);
            }
            seed_views.push(view_try);
            seed_views.sort();
            check_dir = -check_dir;
            end_view = seed_views[3];
        }

        // Next select 6th and 7th ones
        let mut end_ind: usize = 0;
        let mut mid_ind: usize = 2;
        for loop_ in [0usize, 1] {
            if seed_views.len() as i32 >= num_needed {
                return seed_views;
            }
            let mut view_try = -1;
            let end_view = seed_views[end_ind];

            // If this side was divided previously, try again to go outside
            if divided[loop_] != 0 {
                view_try = end_view + check_dir;
                while view_try > track_start
                    && view_try < track_end
                    && (tilt(end_view) - tilt(view_try)).abs() < ang_crit
                {
                    view_try += check_dir;
                }
            }

            // If nothing picked yet, find biggest gap in range
            if view_try < track_start || view_try > track_end {
                let fill_ind = (end_ind + mid_ind) / 2;

                // There must be a gap in range
                if mid_z + 2 * check_dir != end_view {
                    // If there is no gap on one side or other of filled value, use the
                    // other side
                    if mid_z + check_dir == seed_views[fill_ind] {
                        view_try = (seed_views[fill_ind] + end_view).div_euclid(2);
                    } else if seed_views[fill_ind] + check_dir == end_view {
                        view_try = (mid_z + seed_views[fill_ind]).div_euclid(2);

                        // Otherwise pick the biggest angle gap
                    } else if (tilt(mid_z) - tilt(seed_views[fill_ind])).abs()
                        > (tilt(end_view) - tilt(seed_views[fill_ind])).abs()
                    {
                        view_try = (mid_z + seed_views[fill_ind]).div_euclid(2);
                    } else {
                        view_try = (seed_views[fill_ind] + end_view).div_euclid(2);
                    }
                }
            }

            // And if there is still nothing valid, just fill first gap from this direction
            if view_try < track_start || view_try > track_end {
                view_try = self.fill_first_gap(&seed_views, check_dir, mid_z);
            }
            seed_views.push(view_try);
            seed_views.sort();
            check_dir = -check_dir;
            mid_ind = 3;
            end_ind = 5;
        }

        seed_views
    }

    /// Matches `runFindBeads` (`autofidseed:195`): run imodfindbeads with
    /// current selection of views.  Returns the total number of peaks.
    pub fn run_find_beads(&mut self) -> i32 {
        let mut sect_opt = format!(
            "SectionsToDo {}",
            self.included_views[self.seed_views[0] as usize]
        );
        let mut view_str = format!(
            " on views {}",
            self.included_views[self.seed_views[0] as usize] + 1
        );
        for i in 1..self.num_seed_views as usize {
            let seed_vw = self.included_views[self.seed_views[i] as usize];
            sect_opt += &format!(",{seed_vw}");
            if i < self.num_seed_views as usize - 1 {
                view_str += &format!(", {}", seed_vw + 1);
            } else {
                view_str += &format!(", and {}", seed_vw + 1);
            }
        }

        let mut comlines = vec![
            format!("InputImageFile {}", self.image_file),
            format!("OutputModelFile {}", self.tmppeak),
            sect_opt,
            format!("BeadSize {}", py_str_float(self.bead_size)),
            format!("MinSpacing {}", py_str_float(self.spacing)),
            format!("LinearInterpolation {}", self.linear),
            format!("StorageThreshold {}", py_str_float(-self.peak_fraction)),
            format!("MinGuessNumBeads {}", self.guess),
            format!(
                "FallbackThresholds {},{}",
                self.average_fallback * self.num_seed_views as i64,
                self.storage_fallback * self.num_seed_views as i64
            ),
        ];
        if self.light_beads {
            comlines.push("LightBeads 1".to_owned());
        }
        if self.using_sobel {
            comlines.push(format!(
                "KernelSigma {}",
                c_format("%.3f", &[CArg::Dbl(self.ksigma)])
            ));
        }
        if !self.bound_model.is_empty() {
            comlines.push(format!("AreaModel {}", self.bound_model));
            if self.exclude_areas != 0 {
                comlines.push("ExcludeInsideAreas 1".to_owned());
            }
        }
        if let Some(prexg_file) = self.prexg_file.as_ref().filter(|file| !file.is_empty()) {
            comlines.push(format!("PrealignTransformFile {prexg_file}"));
            comlines.push(format!("ImagesAreBinned {}", self.binning));
        }
        if self.adjust_size != 0 {
            comlines.push("AdjustSizes 1".to_owned());
        }

        if !self.find_options.is_empty() {
            for opt in self.find_options.split(" -") {
                comlines.push(opt.trim_start_matches('-').to_owned());
            }
        }

        prnstr(&format!("RUNNING IMODFINDBEADS{view_str}"), "\n", false);
        prnstr(" ", "\n", false);
        // Direct call (CLAUDE.md, "Direct calls instead of parsed output"):
        // imodfindbeads records its reports into a `FindbeadsResult`, used here
        // in place of parsing its last lines and its `Adjusted parameters`
        // line (`autofidseed:236-252`); its report is echoed as before.
        let sink = Arc::new(Mutex::new(FindbeadsResult::default()));
        let recorder = Arc::clone(&sink);
        let findlines = match call_own_program(
            "imodfindbeads -StandardInput",
            &["imodfindbeads", "-StandardInput"],
            Some(&comlines),
            true,
            move || imodfindbeads_recording(recorder),
        ) {
            Ok((_, lines)) => lines,
            Err(_) => self.clean_exit_error(&format!("Running imodfindbeads{view_str}")),
        };
        let reports = sink.lock().expect("imodfindbeads result").reports.clone();

        for l in &findlines {
            prnstr(l, "", false);
        }
        for report in &reports {
            if let FindbeadsReport::AdjustedSize(size) = report
                && self.adjust_size != 0
            {
                // `float(l.split()[-1])` of the line printed with `%.2f`
                match c_format("%.2f", &[CArg::Dbl(*size as f64)]).parse::<f64>() {
                    Ok(value) => self.new_bead_size = Some(value),
                    Err(_) => exit_error("Converting new bead size to float"),
                }
            }
        }

        // The last two reports are the last two lines the script tested:
        // `peaks are above` (threshold or histogram dip) or `using fallback`,
        // then `total peaks being ...`, or `Failed to find dip`.
        if reports.len() < 2 {
            eprintln!("IndexError: list index out of range");
            std::process::exit(1)
        }
        let second = &reports[reports.len() - 2];
        let last = &reports[reports.len() - 1];
        let second_ok = matches!(
            second,
            FindbeadsReport::PeaksAboveThreshold { .. }
                | FindbeadsReport::PeaksAboveDip { .. }
                | FindbeadsReport::UsingFallback
        );
        let total = match last {
            FindbeadsReport::TotalPeaksStored { num, text }
                if text.contains("total peaks being") =>
            {
                Some(*num)
            }
            _ => None,
        };
        let (true, Some(num_peaks_tot)) = (second_ok, total) else {
            if matches!(
                last,
                FindbeadsReport::FailedToFindDip | FindbeadsReport::UsingFallback
            ) {
                exit_error("Cannot proceed; imodfindbeads cannot identify the gold beads");
            }
            exit_error("Output from imodfindbeads does not end in the expected way")
        };
        prnstr(" ", "\n", false);
        num_peaks_tot
    }

    /// Matches `reviseNumSeedViews` (`autofidseed:270`): find out if a higher
    /// number of seed views is needed to try to reach target.
    pub fn revise_num_seed_views(&self, peaks_tot: i32, ns_views: i32) -> i32 {
        let more_view_crit5 = 1.; // Not all points are usable
        let more_view_crit7 = 0.5; // This is unlikely to help more so make the threshold low
        let num_per_view = peaks_tot.div_euclid(ns_views);
        let mut targ_num = self.target_number;
        let mut needed = 3;
        if self.target_density > 0. {
            targ_num = self.target_density * self.nx as f64 * self.ny as f64 * 1.0e-6;
        }
        if (num_per_view as f64) < more_view_crit5 * targ_num {
            needed = self.num_views.min(5);
        }
        if (num_per_view as f64) < more_view_crit7 * targ_num {
            needed = self.num_views.min(7);
        }
        needed
    }

    /// Matches `vectorsMatch` (`autofidseed:284`): test whether two of the
    /// shift-near-zero vectors match.
    pub fn vectors_match(&self, cor1: usize, pek1: usize, cor2: usize, pek2: usize) -> bool {
        let max_vec_angle_diff = 15.;
        let max_vec_len_diff = 0.33;
        if self.vec_angles[cor1][pek1] < -900. || self.vec_angles[cor2][pek2] < -900. {
            return false;
        }
        let mut diff = self.vec_angles[cor1][pek1] - self.vec_angles[cor2][pek2];
        if diff < -180. {
            diff += 360.;
        }
        if diff > 180. {
            diff -= 360.;
        }
        if diff.abs() > max_vec_angle_diff {
            return false;
        }
        (self.vec_lengths[cor1][pek1] - self.vec_lengths[cor2][pek2]).abs() / self.max_length
            < max_vec_len_diff
    }
}

/// The script's top level (`autofidseed:299-1336`).  The script ends
/// without `sys.exit`, so its status is 0; error paths exit the process as
/// `exitError` does.
pub fn autofidseed(arguments: &[OsString]) -> i32 {
    let mut g = Afs {
        prefix: format!("ERROR: {PROGNAME} - "),
        ..Afs::default()
    };
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{} IMOD_DIR is not defined!\n", g.prefix);
        let _ = std::io::stdout().flush();
        return 1;
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 autofidseed
    let options: Vec<String> = [
        "track:TrackCommandFile:FN:",
        "append:AppendToSeedModel:B:",
        "guess:MinGuessNumBeads:I:",
        "spacing:MinSpacing:F:",
        "size:BeadSize:F:",
        "adjust:AdjustSizes:B:",
        "peak:PeakStorageFraction:F:",
        "find:FindBeadOptions:CH:",
        "views:NumberOfSeedViews:I:",
        "shifts:ShiftsNearZeroFraction:F:",
        "justshifts:JustFindShiftsNearZero:I:",
        "boundary:BoundaryModel:FN:",
        "exclude:ExcludeInsideAreas:B:",
        "border:BordersInXandY:IP:",
        "two:TwoSurfaces:B:",
        "number:TargetNumberOfBeads:I:",
        "density:TargetDensityOfBeads:F:",
        "ratio:MaxMajorToMinorRatio:F:",
        "elongated:ElongatedPointsAllowed:I:",
        "cluster:ClusteredPointsAllowed:I:",
        "lower:LowerTargetForClustered:F:",
        "subarea:SubareaSize:I:",
        "sort:SortAreasMinNumAndSize:IP:",
        "ignore:IgnoreSurfaceData:LI:",
        "drop:DropTracks:LI:",
        "pick:PickSeedOptions:CH:",
        "remove:RemoveTempFiles:I:",
        "output:OutputSeedModel:FN:",
        "info:InfoFile:FN:",
        "tempdir:TemporaryDirectory:FN:",
        "leave:LeaveTempFiles:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, PROGNAME, 2, 0, 0);
    // SAFETY: the script's own process environment; nothing else runs.
    unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "0") };

    g.spacing = 0.85;
    let opt_local_points = 20.;
    let opt_local_size = 1000.;
    let min_local_size = 500;
    let min_local_points = 10.;
    let min_points_for_local = 20;
    let mut min_subarea_num = 50;
    let mut min_subarea_size = 2500;
    let mut min_do_tilt: f64 = 40.;
    g.leavetmp = 0;
    let target_to_guess_frac = 0.25;
    let guess_frac_if_just_shifts = 0.5;
    let target_to_average_fb = 0.33;
    let target_to_storage_fb = 2.0;
    let min_size_for_weight_scaling = 12.5;
    let kernel_sigma_default = 0.5; // THE DEFAULT IN BEADTRACK
    let usable_sec_peak_frac = 0.07;
    g.new_bead_size = None;
    let cand_model = "clusterElong.mod";

    let floats = |value: Option<OptionValue>| match value {
        Some(OptionValue::Floats(values)) => Some(values),
        _ => None,
    };
    let ints = |value: Option<OptionValue>| match value {
        Some(OptionValue::Integers(values)) => Some(values),
        _ => None,
    };
    let strings = |value: Option<OptionValue>| match value {
        Some(OptionValue::String(value)) => Some(value),
        _ => None,
    };
    let boolean = |value: Option<OptionValue>| match value {
        Some(OptionValue::Boolean(value)) => Some(value),
        _ => None,
    };
    // `optionValue(lines, option, type[, numVal = 1])`: `numVal = 1` returns
    // the single value itself
    let float1 = |lines: &[String], option: &str| -> Option<f64> {
        floats(option_value(
            lines,
            option,
            FLOAT_VALUE,
            false,
            1,
            None,
            None,
        ))
        .map(|values| values[0])
    };

    // Get track command file
    let trackcom = pip_get_string("TrackCommandFile", "").unwrap_or_default();
    if trackcom.is_empty() {
        exit_error("You must enter the name of a beadtrack command file");
    }

    // Compose info file name
    let (troot, _ext) = os_path_splitext(&trackcom);
    let mut info_suffix = "";
    if troot.ends_with('a') {
        info_suffix = "a";
    }
    if troot.ends_with('b') {
        info_suffix = "b";
    }
    let mut info_name = format!("{PROGNAME}{info_suffix}.info");

    // Set root of names for temp files
    let deftmpdir = format!("{PROGNAME}{info_suffix}.dir");
    g.tmpdir = pip_get_string("TemporaryDirectory", &deftmpdir).unwrap_or(deftmpdir.clone());
    if !Path::new(&g.tmpdir).exists() && std::fs::create_dir(&g.tmpdir).is_err() {
        exit_error(&format!("Making the temporary directory {}", g.tmpdir));
    }

    // `os.access(tmpdir, os.W_OK)`: the POSIX `access` call itself
    let writable = |path: &str| -> bool {
        std::ffi::CString::new(path)
            .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
            .unwrap_or(false)
    };
    if !(Path::new(&g.tmpdir).is_dir() && writable(&g.tmpdir))
        && make_current_dir_writable(&g.tmpdir).is_some()
    {
        exit_error(&format!("Cannot write to temporary directory {}", g.tmpdir));
    }

    g.pid = format!("{}.", std::process::id());
    g.tmproot = "afs".to_owned();
    if !g.tmpdir.is_empty() {
        g.tmproot = format!("{}/{}", g.tmpdir, g.tmproot);
    }

    // Get alternate name,, keep track if using default
    info_name = pip_get_string("InfoFile", &info_name).unwrap_or(info_name);
    let default_info = pip_get_err_no();

    // Get info file
    let mut old_valid = Path::new(&info_name).exists();
    if old_valid {
        let info_lines = read_text_file(&info_name, None, false, None).unwrap_or_default();
        g.pid_info = strings(option_value(
            &info_lines,
            "PID",
            STRING_VALUE,
            false,
            0,
            None,
            None,
        ));
    }

    // See if cleaning up, so we can exit now
    let clean = pip_get_integer("RemoveTempFiles", 0).unwrap_or(0);
    if clean != 0 && old_valid && g.pid_info.as_deref().is_some_and(|pid| !pid.is_empty()) {
        let mut exts: Vec<&str> = RESUME_EXTS.to_vec();
        exts.extend(INFO_EXTS);
        let pid_info = g.pid_info.clone().unwrap();
        g.cleanup(&exts, &pid_info);
        old_valid = false;
        if clean < 0 {
            if g.tmpdir == deftmpdir {
                let cleanlist = glob_glob(&format!("{}/*", g.tmpdir));
                cleanup_files(&cleanlist);
                if std::fs::remove_dir(&g.tmpdir).is_err() {
                    prnstr(
                        &format!("WARNING: could not remove temporary directory {}", g.tmpdir),
                        "\n",
                        false,
                    );
                }
            }
            let _ = std::io::stdout().flush();
            return 0;
        }
    }

    // Get more options
    let just_shifts = pip_get_integer("JustFindShiftsNearZero", 0).unwrap_or(0);
    g.guess = pip_get_integer("MinGuessNumBeads", 0).unwrap_or(0) as i64;
    let mut if_guess = 1 - pip_get_err_no();
    let two_surf = pip_get_boolean("TwoSurfaces", 0).unwrap_or(0);
    let append_to_seed = pip_get_boolean("AppendToSeedModel", 0).unwrap_or(0);
    g.spacing = pip_get_float("MinSpacing", g.spacing).unwrap_or(g.spacing);
    g.peak_fraction = pip_get_float("PeakStorageFraction", 1.0).unwrap_or(1.0);
    g.adjust_size = pip_get_boolean("AdjustSizes", 0).unwrap_or(0);
    g.find_options = pip_get_string("FindBeadOptions", "").unwrap_or_default();
    g.target_number = pip_get_integer("TargetNumberOfBeads", 0).unwrap_or(0) as f64;
    g.target_density = pip_get_float("TargetDensityOfBeads", 0.).unwrap_or(0.);
    if g.target_number < 1. && g.target_density <= 0. {
        exit_error(
            &("You must enter a positive number or density of beads as the target for ".to_owned()
                + "the seed model"),
        );
    }
    g.bound_model = pip_get_string("BoundaryModel", "").unwrap_or_default();
    g.exclude_areas = pip_get_boolean("ExcludeInsideAreas", 0).unwrap_or(0);
    if !g.bound_model.is_empty() && !Path::new(&g.bound_model).exists() {
        exit_error(&format!("Boundary model {} does not exist", g.bound_model));
    }
    (min_subarea_num, min_subarea_size) = pip_get_two_integers(
        "SortAreasMinNumAndSize",
        (min_subarea_num, min_subarea_size),
    )
    .unwrap_or((min_subarea_num, min_subarea_size));
    let if_pick = 1 - pip_get_err_no();
    let subarea_size = pip_get_integer("SubareaSize", 0).unwrap_or(0);
    if if_pick != 0 && subarea_size != 0 {
        exit_error("You cannot enter both -sort and -subarea");
    }
    g.leavetmp = pip_get_boolean("LeaveTempFiles", 0).unwrap_or(0);
    let shifts_frac = pip_get_float("ShiftsNearZeroFraction", 0.2).unwrap_or(0.2);
    if just_shifts > 0 {
        g.target_number = just_shifts as f64;
        if shifts_frac <= 0. || shifts_frac >= 10. {
            exit_error(
                &("You cannot enter -justshifts and -shifts with a value that disables "
                    .to_owned()
                    + "finding shifts"),
            );
        }
        if if_guess == 0 {
            g.guess = (guess_frac_if_just_shifts * just_shifts as f64).round_ties_even() as i64;
            if_guess = 1;
        }
    }

    // Get elongated and clustered and separate the latter if it is an old type of entry
    let mut elongated = pip_get_integer("ElongatedPointsAllowed", 0).unwrap_or(0);
    let mut clustered = pip_get_integer("ClusteredPointsAllowed", 0).unwrap_or(0);
    clustered = 4.min(0.max(clustered));
    elongated = 3.min(0.max(elongated));
    if clustered > 1 {
        if elongated != 0 {
            exit_error("You cannot enter a value > 1 for -cluster if you enter -elongated");
        }
        elongated = clustered - 1;
        clustered = 1;
    }

    let low_target = pip_get_float("LowerTargetForClustered", 0.).unwrap_or(0.);
    let max_major_minor = pip_get_float("MaxMajorToMinorRatio", 0.).unwrap_or(0.);
    let pick_options = pip_get_string("PickSeedOptions", "").unwrap_or_default();
    let mut seed_file = pip_get_string("OutputSeedModel", "").unwrap_or_default();
    let (mut xborder, mut yborder) =
        pip_get_two_integers("BordersInXandY", (0, 0)).unwrap_or((0, 0));
    g.num_seed_views = pip_get_integer("NumberOfSeedViews", 3).unwrap_or(3);
    let views_entered = 1 - pip_get_err_no();
    if g.num_seed_views < 3 || g.num_seed_views > 7 {
        exit_error("The number of views to use as seeds must be between 3 and 7");
    }

    let drop_str = pip_get_string("DropTracks", "").unwrap_or_default();
    let mut drop_list: Vec<i32> = Vec::new();
    if !drop_str.is_empty() {
        // `parselist` returns None for an error; the script then fails on
        // `len(None)` below
        drop_list = match parse_list(&drop_str) {
            Some(list) => list,
            None => {
                eprintln!("TypeError: object of type 'NoneType' has no len()");
                std::process::exit(1)
            }
        };
    }
    let ignore_str = pip_get_string("IgnoreSurfaceData", "").unwrap_or_default();
    let mut ignore_list: Vec<i32> = Vec::new();
    if !ignore_str.is_empty() {
        ignore_list = match parse_list(&ignore_str) {
            Some(list) => list,
            None => {
                eprintln!("TypeError: object of type 'NoneType' has no len()");
                std::process::exit(1)
            }
        };
    }

    for ind in 0..drop_list.len() {
        drop_list[ind] -= 1;
    }
    for ind in 0..ignore_list.len() {
        ignore_list[ind] -= 1;
    }

    // Read com file and get critical entries
    let raw_tracklines = read_text_file(&trackcom, None, false, None).unwrap_or_default();

    // find beadtrack command line then end of input (from transferfid)
    let mut startline: i64 = -1;
    let mut endline = raw_tracklines.len();
    for i in 0..endline {
        let line = raw_tracklines[i].trim();
        if line.starts_with('$') && line.contains("beadtrack") && line.contains("-Standard") {
            startline = i as i64 + 1;
        } else if startline > 0 && line.starts_with('$') {
            endline = i;
        }
    }

    if startline < 0 {
        exit_error(&format!(
            "Old version of {trackcom} cannot be used; convert it by opening and closing the fiducial tracking panel in etomo"
        ));
    }

    let tracklines: Vec<String> = if (startline as usize) < endline {
        raw_tracklines[startline as usize..endline].to_vec()
    } else {
        Vec::new()
    };

    g.image_file = strings(option_value(
        &tracklines,
        "ImageFile",
        STRING_VALUE,
        false,
        0,
        None,
        None,
    ))
    .unwrap_or_default();
    if seed_file.is_empty() {
        seed_file = strings(option_value(
            &tracklines,
            "InputSeedModel",
            STRING_VALUE,
            false,
            0,
            None,
            None,
        ))
        .unwrap_or_default();
    }
    g.light_beads = boolean(option_value(
        &tracklines,
        "LightBeads",
        BOOL_VALUE,
        false,
        0,
        None,
        None,
    ))
    .unwrap_or(false);
    let rotation_arr = floats(option_value(
        &tracklines,
        "RotationAngle",
        FLOAT_VALUE,
        false,
        0,
        None,
        None,
    ));
    let do_tilt_arr = floats(option_value(
        &tracklines,
        "MinTiltRangeToFindAngles",
        FLOAT_VALUE,
        false,
        0,
        None,
        None,
    ));
    if let Some(values) = do_tilt_arr.as_ref().filter(|values| !values.is_empty())
        && values[0] > min_do_tilt
    {
        min_do_tilt = values[0];
    }
    if !Path::new(&g.image_file).exists() {
        exit_error(&format!(
            "The prealigned stack {} does not exist yet",
            g.image_file
        ));
    }

    let nz;
    (g.nx, g.ny, nz) = match get_mrc_size(&g.image_file) {
        Ok(size) => size,
        Err(_) => exit_from_imod_error(PROGNAME),
    };

    // Get the border sizes
    xborder = Afs::set_border_size((g.nx + g.ny).div_euclid(2), xborder);
    yborder = Afs::set_border_size((g.nx + g.ny).div_euclid(2), yborder);

    // Get the binning and prealign file
    g.binning = 1;
    let bin_arr = ints(option_value(
        &tracklines,
        "ImagesAreBinned",
        1,
        false,
        0,
        None,
        None,
    ));
    g.prexg_file = strings(option_value(
        &tracklines,
        "PrealignTransformFile",
        STRING_VALUE,
        false,
        0,
        None,
        None,
    ));
    if let Some(values) = bin_arr.as_ref().filter(|values| !values.is_empty()) {
        g.binning = values[0];
    }

    // Get the box size and determine if finding shifts near zero tilt
    let box_size_arr = ints(option_value(
        &tracklines,
        "BoxSizeXandY",
        INT_VALUE,
        false,
        2,
        None,
        None,
    ));
    let mut pixel_size = float1(&tracklines, "PixelSize");
    if pixel_size.is_none()
        && let Ok(pix_spacing) = get_mrc_pixel(&g.image_file)
        && pix_spacing > 0.
        && pix_spacing != 1.0
    {
        pixel_size = Some(pix_spacing * g.binning as f64 / 10.);
    }

    let mut find_shifts_near_zero = false;
    if box_size_arr.is_some() && pixel_size.is_some() && shifts_frac > 0. && shifts_frac < 10. {
        find_shifts_near_zero = true;
    }
    if just_shifts > 0 && !find_shifts_near_zero {
        exit_error(&format!(
            "Cannot find shifts near zero tilt; {trackcom} is missing pixel size or box size"
        ));
    }

    // Get the unbinned bead size from the track com file regardless
    let track_diameter = float1(&tracklines, "BeadDiameter");
    if track_diameter.is_some_and(|diameter| diameter <= 0.) {
        exit_error(&format!(
            "The BeadDiameter entry in {trackcom} is not positive"
        ));
    }
    g.bead_size = pip_get_float("BeadSize", 0.).unwrap_or(0.);

    // If no -size entered and produce the value needed for findbeads
    if pip_get_err_no() != 0 {
        let Some(track_diameter) = track_diameter else {
            exit_error(&format!(
                "There is no BeadDiameter entry in {trackcom}; fix this or enter a size with -size"
            ))
        };
        g.bead_size = track_diameter / g.binning as f64;
    }
    let mut size_for_weights = g.bead_size;

    // Get filtering information if sobel activated
    g.using_sobel =
        boolean(option_value(&tracklines, "Sobel", 3, false, 0, None, None)).unwrap_or(false);
    g.linear = 0;
    if g.using_sobel {
        let kernel_arr = floats(option_value(&tracklines, "Kernel", 2, false, 0, None, None));
        let scalable_sigma = float1(&tracklines, "ScalableSigma");
        if let Some(kernel_arr) = kernel_arr.filter(|values| !values.is_empty()) {
            g.ksigma = kernel_arr[0];
            if g.ksigma >= 1.49 {
                g.linear = 1;
            }
        } else if let Some(scalable_sigma) = scalable_sigma.filter(|sigma| *sigma != 0.) {
            g.ksigma = scalable_sigma * g.bead_size;
        } else {
            g.ksigma = kernel_sigma_default;
        }
    }

    // Get tilt angle options and make list of tilt angles
    let tilt_file = strings(option_value(
        &tracklines,
        "TiltFile",
        0,
        false,
        0,
        None,
        None,
    ));
    let mut tilt_angles: Vec<f64> = Vec::new();
    if let Some(tilt_file) = tilt_file.filter(|file| !file.is_empty()) {
        let tilt_lines = read_text_file(&tilt_file, None, false, None).unwrap_or_default();
        for i in 0..tilt_lines.len() {
            if !tilt_lines[i].trim().is_empty() {
                match tilt_lines[i].trim().parse::<f64>() {
                    Ok(value) => tilt_angles.push(value),
                    Err(_) => exit_error(&format!(
                        "Converting lines in {tilt_file} to floating point values"
                    )),
                }
            }
        }
    } else {
        let first = floats(option_value(
            &tracklines,
            "FirstTiltAngle",
            2,
            false,
            0,
            None,
            None,
        ));
        let increment = floats(option_value(
            &tracklines,
            "TiltIncrement",
            2,
            false,
            0,
            None,
            None,
        ));
        let (Some(first), Some(increment)) = (
            first.filter(|values| !values.is_empty()),
            increment.filter(|values| !values.is_empty()),
        ) else {
            exit_error(
                &("The track command file must have either a tilt angle file or starting "
                    .to_owned()
                    + "and increment tilt angles"),
            )
        };
        for i in 0..nz {
            tilt_angles.push(first[0] + i as f64 * increment[0]);
        }
    }

    // Apply whatever angle offset is in track entry
    let angle_offset = float1(&tracklines, "AngleOffset");
    if let Some(angle_offset) = angle_offset.filter(|offset| *offset != 0.) {
        for ind in 0..tilt_angles.len() {
            tilt_angles[ind] += angle_offset;
        }
    }

    // Get the exclude list, which is numbered from 1
    let skip_list_str = strings(option_value(
        &tracklines,
        "SkipViews",
        0,
        false,
        0,
        None,
        None,
    ));
    let mut exclude_list: Vec<i32> = Vec::new();
    if let Some(skip_list_str) = skip_list_str.filter(|text| !text.is_empty()) {
        exclude_list = parse_list(&skip_list_str).unwrap_or_default();
    }

    // Make a list of included views and get the number of them
    g.num_views = tilt_angles.len() as i32;
    g.included_views = Vec::new();
    g.tilt_included = Vec::new();
    for iv in 0..g.num_views {
        if !exclude_list.contains(&(iv + 1)) {
            g.included_views.push(iv);
            g.tilt_included.push(tilt_angles[iv as usize]);
        }
    }

    let num_included = g.included_views.len() as i32;

    // Find minimum tilt view and highest tilt
    g.min_tilt_ind = 0;
    let mut highest_tilt: f64 = 0.;
    if num_included < 3 {
        exit_error("There must be at least 3 views in the tilt series and not in a skip list");
    }
    g.num_seed_views = num_included.min(g.num_seed_views);

    for iv in 0..g.num_views as usize {
        highest_tilt = highest_tilt.max(tilt_angles[iv].abs());
    }

    for incl in 0..num_included as usize {
        if g.tilt_included[g.min_tilt_ind as usize].abs() + 0.1 >= g.tilt_included[incl].abs() {
            g.min_tilt_ind = incl as i32;
        }
    }

    // Get view range for tracking: do 11 views unless range is > 20 deg, or 9 views unless
    // range is still > 20 deg; or 7 views
    let tilt_of = |g: &Afs, view: i32| -> f64 { py_index!(g.tilt_included, view) };
    for view_inc in [10, 8, 6] {
        g.track_start = 0.max(g.min_tilt_ind - view_inc / 2);
        g.track_end = (num_included - 1).min(g.track_start + view_inc);
        g.track_start = 0.max(g.track_end - view_inc);
        if (tilt_of(&g, g.track_start) - tilt_of(&g, g.track_end)).abs() <= 20.1 {
            break;
        }
    }

    // For very fine increments, increase the view range to have at least 7.5 degree track
    if (tilt_of(&g, g.track_start) - tilt_of(&g, g.track_end)).abs() < 7.4 {
        for view_inc in (12..80).step_by(2) {
            g.track_start = 0.max(g.min_tilt_ind - view_inc / 2);
            g.track_end = (num_included - 1).min(g.track_start + view_inc);
            g.track_start = 0.max(g.track_end - view_inc);
            if (tilt_of(&g, g.track_start) - tilt_of(&g, g.track_end)).abs() >= 7.4 {
                break;
            }
        }
    }

    // midView, trackStart, trackEnd and seed lists will all be included view indexes
    // so need to take includedViews[view] to get true Z
    g.mid_view = (g.track_end + g.track_start).div_euclid(2);

    // Convert a density to a target number
    // If there is a boundary model, find out the total area from imodfindbeads
    if g.target_density > 0. {
        g.target_density *= (g.binning * g.binning) as f64;
        let mut total_area = g.nx as f64 * g.ny as f64 * 1.0e-6;
        if !g.bound_model.is_empty() {
            let mut comlines = vec![
                format!("InputImageFile {}", g.image_file),
                format!("AreaModel {}", g.bound_model),
                format!(
                    "QueryAreaOnSection {}",
                    g.included_views[g.min_tilt_ind as usize]
                ),
            ];
            if g.exclude_areas != 0 {
                comlines.push("ExcludeInsideAreas 1".to_owned());
            }
            // Direct call: the area imodfindbeads reports (`Area (megapixels)
            // included in analysis = %.3f`), rounded as printed.
            let sink = Arc::new(Mutex::new(FindbeadsResult::default()));
            let recorder = Arc::clone(&sink);
            if call_own_program(
                "imodfindbeads -StandardInput",
                &["imodfindbeads", "-StandardInput"],
                Some(&comlines),
                true,
                move || imodfindbeads_recording(recorder),
            )
            .is_err()
            {
                g.clean_exit_error("Running imodfindbeads to determine area being analyzed");
            }
            let mut found = false;
            for report in &sink.lock().expect("imodfindbeads result").reports {
                if let FindbeadsReport::Area(area) = report {
                    match c_format("%.3f", &[CArg::Dbl(*area)]).trim().parse::<f64>() {
                        Ok(value) => total_area = value,
                        Err(_) => {
                            exit_error("Converting area being analyzed from imodfindbeads output")
                        }
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                // ELSE ON FOR
                exit_error("Cannot find area being analyzed from imodfindbeads output");
            }
        }

        g.target_number = g.target_density * total_area;
    }

    // Now it is possible to convert to get fallbacks for the guess and thresholds
    if if_guess == 0 {
        g.guess = 1.max((g.target_number * target_to_guess_frac).round_ties_even() as i64);
    }
    g.average_fallback = 1.max((g.target_number * target_to_average_fb).round_ties_even() as i64);
    g.storage_fallback = 1.max((g.target_number * target_to_storage_fb).round_ties_even() as i64);

    // `int(os.stat(f).st_mtime)`: whole seconds, truncated
    let mtime = |file: &str| -> i64 {
        match std::fs::metadata(file).and_then(|meta| meta.modified()) {
            Ok(time) => match time.duration_since(std::time::UNIX_EPOCH) {
                Ok(duration) => duration.as_secs() as i64,
                Err(error) => -(error.duration().as_secs_f64().ceil() as i64),
            },
            Err(error) => {
                eprintln!("FileNotFoundError: {error}: '{file}'");
                std::process::exit(1)
            }
        }
    };
    let track_mtime = mtime(&trackcom);
    let image_mtime = mtime(&g.image_file);
    let bead_str = py_str_float(g.bead_size);
    let spacing_str = py_str_float(g.spacing);
    let mut bound_mtime = 0;
    if !g.bound_model.is_empty() {
        bound_mtime = mtime(&g.bound_model);
    }

    // `os.path.split(trackcom)`
    let comname = match trackcom.rfind('/') {
        Some(index) => trackcom[index + 1..].to_owned(),
        None => trackcom.clone(),
    };
    let com_saved = format!("{}/{comname}", g.tmpdir);

    // Process info file fully now
    let mut adj_size_arr: Option<Vec<f64>> = None;
    let mut two_surf_info: Option<bool> = None;
    let mut zero_shifts: Option<String> = None;
    if old_valid {
        let info_lines = read_text_file(&info_name, None, false, None).unwrap_or_default();
        let opt =
            |option: &str, kind: i32| option_value(&info_lines, option, kind, false, 0, None, None);
        g.pid_info = strings(opt("PID", 0));
        let track_info = strings(opt("TrackCom", 0));
        let track_time_arr = ints(opt("TrackTime", 1));
        let image_time_arr = ints(opt("ImageTime", 1));
        let bead_info = strings(opt("BeadSize", 0));
        let guess_info_arr = ints(opt("MinGuess", 1));
        let peak_frac_info_arr = floats(opt("PeakFraction", 2));
        let spacing_info = strings(opt("Spacing", 0));
        two_surf_info = boolean(opt("TwoSurf", 3));
        let bound_info = strings(opt("BoundFile", 0));
        let bound_time_arr = ints(opt("BoundTime", 1));
        let num_peak_arr = ints(opt("NumPeaks", 1));
        let num_seed_arr = ints(opt("NumViews", 1));
        let view_enter_info = boolean(opt("ViewsEntered", 3));
        adj_size_arr = floats(opt("AdjustSizes", 2));
        let zero_frac_arr = floats(opt("ZeroShiftsFrac", 2));
        zero_shifts = strings(opt("ShiftsNearZeroTilt", 0));
        let nonempty = |values: &Option<Vec<i32>>| values.as_ref().is_some_and(|v| !v.is_empty());
        let nonempty_f = |values: &Option<Vec<f64>>| values.as_ref().is_some_and(|v| !v.is_empty());

        // Need to check contents of track.com if it doesn't match time
        let mut track_matches = false;
        if track_info.as_deref() == Some(trackcom.as_str())
            && nonempty(&track_time_arr)
            && track_time_arr.as_ref().unwrap()[0] as i64 != track_mtime
            && Path::new(&com_saved).exists()
        {
            let saved_lines = read_text_file(&com_saved, None, false, None).unwrap_or_default();
            if saved_lines.len() == raw_tracklines.len() {
                track_matches = saved_lines == raw_tracklines;
            }
        }

        // Now ready to evaluate validity
        let two_surf_ok = match two_surf_info {
            Some(info) => two_surf <= info as i32,
            // `0 <= None` is a TypeError in Python 3; an info file without
            // the entry is not valid for resuming
            None => false,
        };
        old_valid = just_shifts <= 0
            && g.pid_info.as_deref().is_some_and(|pid| !pid.is_empty())
            && track_info.as_deref() == Some(trackcom.as_str())
            && nonempty(&track_time_arr)
            && (track_time_arr.as_ref().unwrap()[0] as i64 == track_mtime || track_matches)
            && nonempty(&image_time_arr)
            && image_time_arr.as_ref().unwrap()[0] as i64 == image_mtime
            && bead_info.as_deref() == Some(bead_str.as_str())
            && nonempty(&guess_info_arr)
            && guess_info_arr.as_ref().unwrap()[0] as i64 == g.guess
            && nonempty_f(&peak_frac_info_arr)
            && peak_frac_info_arr.as_ref().unwrap()[0] == g.peak_fraction
            && spacing_info.as_deref() == Some(spacing_str.as_str())
            && two_surf_ok
            && (nonempty(&num_seed_arr)
                && nonempty(&num_peak_arr)
                && ((views_entered == 0 && !view_enter_info.unwrap_or(false))
                    || (views_entered != 0
                        && num_seed_arr.as_ref().unwrap()[0] == g.num_seed_views)))
            && nonempty_f(&adj_size_arr)
            && adj_size_arr.as_ref().unwrap()[0].round_ties_even() as i32 == g.adjust_size
            && nonempty_f(&zero_frac_arr)
            && zero_frac_arr.as_ref().unwrap()[0] == shifts_frac
            && ((bound_info.as_deref().is_some_and(|info| !info.is_empty())
                && bound_info.as_deref() == Some(g.bound_model.as_str())
                && nonempty(&bound_time_arr)
                && bound_mtime == bound_time_arr.as_ref().unwrap()[0] as i64)
                || (bound_info.as_deref().is_none_or(str::is_empty) && g.bound_model.is_empty()));

        if old_valid && views_entered == 0 {
            let needed = g.revise_num_seed_views(
                num_peak_arr.as_ref().unwrap()[0],
                num_seed_arr.as_ref().unwrap()[0],
            );
            if needed > num_seed_arr.as_ref().unwrap()[0] {
                old_valid = false;
                g.num_seed_views = needed;
            }
        }

        // Passed that test (!), now make sure all the required files are still there
        if old_valid {
            let pid_info = g.pid_info.clone().unwrap();
            'outer: for ind in 0..num_seed_arr.as_ref().unwrap()[0] {
                if !old_valid {
                    break;
                }
                for ext in RESUME_EXTS {
                    if ext == RESUME_EXTS[1] && !two_surf_info.unwrap_or(false) {
                        continue;
                    }
                    if !Path::new(&format!("{}{pid_info}{ind}{ext}", g.tmproot)).exists() {
                        old_valid = false;
                        continue 'outer;
                    }
                }
            }
        }

        // If not resuming and there was an old PID, clean up old files
        if let Some(pid_info) = g.pid_info.clone().filter(|pid| !pid.is_empty())
            && !old_valid
        {
            let leave_save = g.leavetmp;
            g.leavetmp = 0;
            let mut exts: Vec<&str> = RESUME_EXTS.to_vec();
            exts.extend(INFO_EXTS);
            g.cleanup(&exts, &pid_info);
            g.leavetmp = leave_save;
        }

        // Adjust number of seed views if resuming
        if old_valid {
            g.num_seed_views = num_seed_arr.as_ref().unwrap()[0];
            let adj = adj_size_arr.as_ref().unwrap();
            if adj.len() > 1 && adj[1] > 0. {
                size_for_weights = adj[1];
                prnstr(
                    &format!(
                        "Adjusted parameters in previous run for new bead size of {}    [AFS2]",
                        py_str_float(adj[1])
                    ),
                    "\n",
                    false,
                );
            }
            if let Some(zero_shifts) = &zero_shifts {
                prnstr(
                    &format!(
                        "Adjusted parameters in previous run with shifts per view entry of {zero_shifts}    [AFS4]"
                    ),
                    "\n",
                    false,
                );
            }
        }
    }

    // If making a new info file with default name, make sure current directory is writable
    if default_info != 0 && !old_valid && !writable(".") {
        exit_error(
            "You cannot write to the current directory; use -info to make the info file elsewhere",
        );
    }

    // Check validity of drop and ignore lists
    if !drop_list.is_empty() || !ignore_list.is_empty() {
        if !old_valid {
            exit_error(
                &("You cannot list tracks to drop or ignore unless resuming with ".to_owned()
                    + "existing tracks"),
            );
        }
        let mut num_drop = 0;
        let mut num_ignore = 0;
        for i in 0..g.num_seed_views {
            if drop_list.contains(&i) {
                num_drop += 1;
            }
            if ignore_list.contains(&i) {
                num_ignore += 1;
            }
        }
        if num_drop == g.num_seed_views {
            exit_error("The list of tracks to drop includes all tracks");
        }
        if num_ignore == g.num_seed_views {
            exit_error("The list of tracks to ignore surface data from includes all tracks");
        }
    }

    g.seed_views = g.select_seed_views(g.num_seed_views);
    let mut skip_list = String::new();
    let track_vw_start = g.included_views[g.track_start as usize];
    let track_vw_end = g.included_views[g.track_end as usize];
    if track_vw_start == 1 {
        skip_list = "1".to_owned();
    } else if track_vw_start > 1 {
        skip_list = format!("1-{track_vw_start}");
    }
    if track_vw_end < nz - 1 && !skip_list.is_empty() {
        skip_list += ",";
    }
    if track_vw_end == nz - 2 {
        skip_list += &nz.to_string();
    } else if track_vw_end < nz - 2 {
        skip_list += &format!("{}-{nz}", track_vw_end + 2);
    }
    if skip_list.is_empty() {
        skip_list = "0".to_owned();
    }

    // Ready to find the beads
    let mut num_peaks_tot = 0;
    if !old_valid {
        g.pid_info = Some(g.pid.clone());
        g.tmppeak = format!("{}{}{}", g.tmproot, g.pid, INFO_EXTS[0]);
        num_peaks_tot = g.run_find_beads();

        // Unless number of views was specified, check if there are not enough and
        // retrack with more views
        if views_entered == 0 {
            g.num_seed_views = g.revise_num_seed_views(num_peaks_tot, g.num_seed_views);
            if g.num_seed_views > 3 {
                prnstr(
                    "REDOING IMODFINDBEADS with more views to try to get more points",
                    "\n",
                    false,
                );
                g.seed_views = g.select_seed_views(g.num_seed_views);
                num_peaks_tot = g.run_find_beads();
            }
        }
    }

    // Skip views to use views at fixed intervals if there are lots of them
    // But include the seed views so every track is the same
    let skip_interval = (g.track_end + 1 - g.track_start).div_euclid(11);
    if skip_interval > 1 {
        let mut last_included = g.track_start;
        for view in g.track_start + 1..g.track_end {
            if view - last_included < skip_interval && !g.seed_views.contains(&view) {
                skip_list += &format!(",{}", g.included_views[view as usize] + 1);
            } else {
                last_included = view;
            }
        }
    }

    // Finding shifts near zero: get the range of Z values for imodmop
    let mut shifts_line = String::new();
    let mut adjust_sed: Vec<String> = Vec::new();
    if find_shifts_near_zero && !old_valid {
        // Find seed view nearest zero
        let mut near_zero: i64 = -1;
        let mut min_diff: f64 = 100.;
        for ind in 0..g.seed_views.len() {
            let diff = (g.seed_views[ind] as f64 - g.min_tilt_ind as f64).abs();
            if near_zero < 0 || diff < min_diff {
                min_diff = diff;
                near_zero = ind as i64;
            }
        }

        if near_zero == 0 {
            near_zero += 1;
        }
        if near_zero >= g.seed_views.len() as i64 - 1 {
            near_zero -= 1;
        }
        let view_near_zero = py_index!(g.seed_views, near_zero);
        let view_below_zero = py_index!(g.seed_views, near_zero - 1);
        let view_above_zero = py_index!(g.seed_views, near_zero + 1);

        let min_z = py_index!(g.included_views, view_below_zero);
        let mid_z = py_index!(g.included_views, view_near_zero);
        let max_z = py_index!(g.included_views, view_above_zero);
        let diam = (1.5 * g.bead_size) as i64;
        let mop_file = format!("{}{}mop", g.tmproot, g.pid);
        let dmean = match get_mrc(&g.image_file, true, false) {
            Ok(MrcInfo::All(.., dmean)) => dmean,
            _ => exit_from_imod_error(PROGNAME),
        };
        let mopcmd = format!(
            "imodmop -tube 1 -diam {diam} -fv {} -zmin {min_z},{max_z} -planar {} \"{}\" {mop_file}",
            py_str_float(dmean),
            g.tmppeak,
            g.image_file
        );
        if run_cmd(&mopcmd, None, None, None, &[]).is_err() {
            g.clean_exit_error("Running imodmop to isolate bead component of images");
        }

        let mop_trim = format!("{}{}moptrim", g.tmproot, g.pid);
        let newstcmd = format!(
            "newstack -sec {},{},{} {mop_file} {mop_trim}",
            0,
            mid_z - min_z,
            max_z - min_z
        );
        if run_cmd(&newstcmd, None, None, None, &[]).is_err() {
            g.clean_exit_error("Running Newstack to extract imodmop output for correlation");
        }

        let tilt = |view: i32| -> f64 { py_index!(g.tilt_included, view) };
        let increment = (tilt(view_above_zero) - tilt(view_below_zero)).abs() / 2.;
        let max_swing = 800. * increment.to_radians().sin();
        let pixel = pixel_size.unwrap();
        let length = (2. * max_swing / (g.binning as f64 * pixel)) as i64;
        let mut xcorr_com = vec![
            format!("InputFile {mop_trim}"),
            format!("OutputFile {}{}mopxf", g.tmproot, g.pid),
            format!(
                "TiltAngles {},{},{}",
                py_str_float(tilt(view_below_zero)),
                py_str_float(tilt(view_near_zero)),
                py_str_float(tilt(view_above_zero))
            ),
            "FilterSigma1    0.03".to_owned(),
            "FilterRadius2   0.25".to_owned(),
            "FilterSigma2    0.05".to_owned(),
            format!("SecondPeakBoxSize {length},{diam}"),
        ];
        if let Some(rotation) = rotation_arr.as_ref().filter(|values| !values.is_empty()) {
            xcorr_com.push(format!("RotationAngle {}", py_str_float(rotation[0])));
        }

        // The script parses tiltxcorr's `View` report lines (text parse kept:
        // tiltxcorr has no recording entry point yet)
        let xcorr_lines = match run_cmd(
            "tiltxcorr -StandardInput",
            Some(&xcorr_com),
            None,
            None,
            &[],
        ) {
            Ok(lines) => lines.unwrap_or_default(),
            Err(_) => {
                g.clean_exit_error("Running Tiltxcorr to determine bead shifts near zero tilt")
            }
        };

        let mut shifts = [[0f64; 6]; 2];
        g.vec_angles = [[-999., -999.], [-999., -999.]];
        g.vec_lengths = [[0., 0.], [0., 0.]];
        g.max_length = 0.;

        let letters = Regex::new(r"[a-zA-DF-Z,]").unwrap();
        let mut found_views: Option<Vec<String>> = Some(Vec::new());
        for line in &xcorr_lines {
            if line.starts_with("View") {
                let nums = letters.replace_all(line, "").into_owned();
                let num_split: Vec<&str> = nums.split_whitespace().collect();
                found_views.as_mut().unwrap().push(num_split[0].to_owned());
            }
        }

        // Set up how to deal with the sign of the output based on order of views
        // Tilt angle is irrelevant, the shifts are per view
        let mut div_by_views = [
            view_above_zero - view_near_zero,
            view_near_zero - view_below_zero,
        ];
        let mut sign: f64 = 0.;
        let mut keep_sign = false;
        let fv: Vec<&str> = found_views
            .as_ref()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect();
        if fv == ["3", "1"] {
            sign = -1.;
            keep_sign = false;
        } else if fv == ["2", "3"] {
            sign = -1.;
            keep_sign = true;
            div_by_views = [
                view_near_zero - view_below_zero,
                view_above_zero - view_near_zero,
            ];
        } else if fv == ["2", "1"] {
            sign = 1.;
            keep_sign = true;
        } else {
            found_views = None;
            prnstr(
                &("WARNING: View numbers in output from tiltxcorr finding shifts ".to_owned()
                    + "near zero do not have expected values"),
                "\n",
                false,
            );
            prnstr("", "\n", false);
            for line in &xcorr_lines {
                if line.starts_with("View") {
                    prnstr(line.trim_end(), "\n", false);
                }
            }
        }

        let mut sh_ind = 0usize;
        for line in &xcorr_lines {
            if found_views.is_some() && line.starts_with("View") {
                let nums = letters.replace_all(line, "").into_owned();
                let num_split: Vec<&str> = nums.split_whitespace().collect();

                // Convert numbers, apply sign to shift
                for ind in 1..num_split.len().min(7) {
                    match num_split[ind].parse::<f64>() {
                        Ok(value) => {
                            shifts[sh_ind][ind - 1] = value;
                            if ind != 3 && ind != 6 {
                                shifts[sh_ind][ind - 1] *= sign / div_by_views[sh_ind] as f64;
                            }
                        }
                        Err(_) => {
                            let mut all: Vec<&str> = CLEAN_EXTS.to_vec();
                            all.extend(RESUME_EXTS);
                            all.extend(INFO_EXTS);
                            g.cleanup(&all, &g.pid);
                            exit_error(&format!(
                                "Converting number to float in tiltxcorr output line {line}"
                            ));
                        }
                    }
                }

                // Get angle and length for primary, then for secondary if it is strong
                // enough
                g.vec_angles[sh_ind][0] = shifts[sh_ind][1].atan2(shifts[sh_ind][0]).to_degrees();
                g.vec_lengths[sh_ind][0] =
                    (shifts[sh_ind][0].powi(2) + shifts[sh_ind][1].powi(2)).sqrt();
                g.max_length = g.max_length.max(g.vec_lengths[sh_ind][0]);
                if shifts[sh_ind][5] > usable_sec_peak_frac * shifts[sh_ind][2] {
                    g.vec_angles[sh_ind][1] =
                        shifts[sh_ind][4].atan2(shifts[sh_ind][3]).to_degrees();
                    g.vec_lengths[sh_ind][1] =
                        (shifts[sh_ind][3].powi(2) + shifts[sh_ind][4].powi(2)).sqrt();
                    g.max_length = g.max_length.max(g.vec_lengths[sh_ind][1]);
                }

                sh_ind += 1;
                if !keep_sign {
                    sign = 1.;
                }
            }
        }

        if found_views.is_some() && (shifts[0][2] == 0. || shifts[1][2] == 0.) {
            g.max_length = 0.;
            prnstr(
                "WARNING: Output from tiltxcorr finding shifts near zero is not usable",
                "\n",
                false,
            );
        }

        // Proceed only if both primaries are within a reasonable distance of 0 and
        // threshold met
        let box0 = box_size_arr.as_ref().unwrap()[0];
        let binning = g.binning as f64;
        let pair =
            |a: f64, b: f64| -> String { c_format("%.1f,%.1f", &[CArg::Dbl(a), CArg::Dbl(b)]) };
        if g.max_length < length as f64 / 2. && g.max_length * binning > shifts_frac * box0 as f64 {
            // If primary angles match, set first shift and look for second
            if g.vectors_match(0, 0, 1, 0) || g.vectors_match(0, 1, 1, 1) {
                if g.vectors_match(0, 0, 1, 0) {
                    shifts_line = pair(
                        binning * (shifts[0][0] + shifts[1][0]) / 2.,
                        binning * (shifts[0][1] + shifts[1][1]) / 2.,
                    );
                }
                if g.vectors_match(0, 1, 1, 1) {
                    if !shifts_line.is_empty() {
                        shifts_line += ",";
                    }
                    shifts_line += &pair(
                        binning * (shifts[0][3] + shifts[1][3]) / 2.,
                        binning * (shifts[0][4] + shifts[1][4]) / 2.,
                    );
                }
            }
            // Otherwise look for crossed matches and use them if found
            else if g.vectors_match(0, 0, 1, 1) || g.vectors_match(1, 0, 0, 1) {
                if g.vectors_match(0, 0, 1, 1) {
                    shifts_line = pair(
                        binning * (shifts[0][0] + shifts[1][3]) / 2.,
                        binning * (shifts[0][1] + shifts[1][4]) / 2.,
                    );
                }
                if g.vectors_match(1, 0, 0, 1) {
                    if !shifts_line.is_empty() {
                        shifts_line += ",";
                    }
                    shifts_line += &pair(
                        binning * (shifts[1][0] + shifts[0][3]) / 2.,
                        binning * (shifts[1][1] + shifts[0][4]) / 2.,
                    );
                }
            }

            if !shifts_line.is_empty() {
                adjust_sed.extend(sed_del_and_add(
                    "ShiftsNearZeroTilt",
                    &shifts_line,
                    "BeadDiameter",
                    '?',
                ));
                prnstr(
                    &format!(
                        "Tracking parameters adjusted with shifts per view of {shifts_line}   [AFS3]"
                    ),
                    "\n",
                    false,
                );
            }
        }

        if just_shifts > 0 {
            let newinfo = vec![format!("PID {}", g.pid)];
            let _ = write_text_file(&info_name, &newinfo, false);
            g.cleanup(&CLEAN_EXTS, &g.pid);
            let _ = std::io::stdout().flush();
            return 0;
        }
    }

    // Adjust tracking parameters if the bead size changed by 5% as the man page said
    // existing beadSize value here is binned
    if let Some(new_bead_size) = g.new_bead_size.filter(|size| *size != 0.)
        && !old_valid
        && g.adjust_size != 0
        && (new_bead_size - g.bead_size).abs() > 0.05 * g.bead_size
    {
        let binning = g.binning as f64;
        let unbinned_new_size = new_bead_size * binning;
        adjust_sed.push(format!(
            "?^BeadDiameter?s?[ \t].*? {}?",
            py_str_float(unbinned_new_size)
        ));
        prnstr(
            &format!(
                "Tracking parameters adjusted for new unbinned bead size of {}   [AFS1]",
                c_format("%.2f", &[CArg::Dbl(unbinned_new_size)])
            ),
            "\n",
            false,
        );
        size_for_weights = new_bead_size;
        let min_diam = float1(&tracklines, "MinDiamForParamScaling");

        // If above the diameter for scaling in the com file, rescale the parameters
        if let (Some(min_diam), Some(track_diameter)) = (
            min_diam.filter(|diam| *diam != 0.),
            track_diameter.filter(|diam| *diam != 0.),
        ) {
            let mut old_scale = 1.;
            let mut new_scale = 1.;
            if track_diameter > min_diam {
                old_scale = track_diameter / min_diam;
            }
            if new_bead_size * binning > min_diam {
                new_scale = new_bead_size * binning / min_diam;
            }
            let scale = new_scale / old_scale;
            if scale != 1. {
                for option in [
                    "DistanceRescueCriterion",
                    "PostFitRescueResidual",
                    "MaxRescueDistance",
                ] {
                    let crit = float1(&tracklines, option);
                    if let Some(crit) = crit.filter(|crit| *crit != 0.) {
                        adjust_sed.push(format!(
                            "?^{option}?s?[ \t].*? {}?",
                            c_format("%.2f", &[CArg::Dbl(crit * scale)])
                        ));
                    }
                }
                let crit_arr = floats(option_value(
                    &tracklines,
                    "DeletionCriterionMinAndSD",
                    2,
                    false,
                    0,
                    None,
                    None,
                ));
                if let Some(crit_arr) = crit_arr.filter(|values| values.len() > 1) {
                    adjust_sed.push(format!(
                        "?^DeletionCriterionMinAndSD?s?[ \t].*? {}?",
                        c_format(
                            "%.3f,%.2f",
                            &[CArg::Dbl(crit_arr[0] * scale), CArg::Dbl(crit_arr[1])]
                        )
                    ));
                }
                let box_arr = ints(option_value(
                    &tracklines,
                    "BoxSizeXandY",
                    1,
                    false,
                    0,
                    None,
                    None,
                ));
                if let Some(box_arr) = box_arr.filter(|values| values.len() > 1) {
                    let new_xbox = 2 * (box_arr[0] as f64 * scale / 2.).round_ties_even() as i64;
                    let new_ybox = 2 * (box_arr[1] as f64 * scale / 2.).round_ties_even() as i64;
                    adjust_sed.push(format!("?^BoxSizeXandY?s?[ \t].*? {new_xbox},{new_ybox}?"));
                }
            }
        }
    }

    // Put out a track file with adjusted parameters as well as use them here
    if !adjust_sed.is_empty() {
        let (adj_root, adj_ext) = os_path_splitext(&trackcom);
        let adj_file = format!("{adj_root}_adjusted{adj_ext}");
        let _ = pysed(
            &adjust_sed,
            PysedSrc::Lines(&raw_tracklines),
            Some(&adj_file),
            false,
            '?',
            false,
        );
    }

    // Extract the seed models
    let mut tmpseed: Vec<String> = Vec::new();
    let mut tmptrack: Vec<String> = Vec::new();
    let mut tmpsurf: Vec<String> = Vec::new();
    let mut tmpxyzpt: Vec<String> = Vec::new();
    let mut tmpxyzmod: Vec<String> = Vec::new();
    let mut tmpelong: Vec<String> = Vec::new();
    let mut tmpsortmod: Vec<String> = Vec::new();

    let pid_info = g.pid_info.clone().unwrap_or_else(|| g.pid.clone());
    for ind in 0..g.num_seed_views as usize {
        let (tr, pid) = (g.tmproot.clone(), g.pid.clone());
        tmpseed.push(format!("{tr}{pid}{ind}{}", CLEAN_EXTS[0]));
        tmptrack.push(format!("{tr}{pid_info}{ind}{}", RESUME_EXTS[0]));
        tmpxyzpt.push(format!("{tr}{pid}{ind}{}", CLEAN_EXTS[1]));
        tmpsurf.push(format!("{tr}{pid}{ind}{}", CLEAN_EXTS[2]));
        tmpxyzmod.push(format!("{tr}{pid_info}{ind}{}", RESUME_EXTS[1]));
        tmpelong.push(format!("{tr}{pid_info}{ind}{}", RESUME_EXTS[2]));
        tmpsortmod.push(format!("{tr}{pid_info}{ind}{}", INFO_EXTS[1]));

        let seed_vw = g.included_views[g.seed_views[ind] as usize];
        let view_str = (seed_vw + 1).to_string();

        if !old_valid {
            // Direct call: clipmodel records the point counts it reports on its
            // `Number of points reduced from ... to ...` line, used here in
            // place of parsing that line's last word (`autofidseed:726-742`).
            let sink = Arc::new(Mutex::new(ClipmodelResult::default()));
            let recorder = Arc::clone(&sink);
            let zmin = format!("{seed_vw},{seed_vw}");
            let words = [
                "clipmodel",
                "-keep",
                "-zmin",
                zmin.as_str(),
                g.tmppeak.as_str(),
                tmpseed[ind].as_str(),
            ];
            if call_own_program(
                &format!(
                    "clipmodel -keep -zmin {zmin} \"{}\" \"{}\"",
                    g.tmppeak, tmpseed[ind]
                ),
                &words,
                None,
                true,
                move || clipmodel_recording(recorder),
            )
            .is_err()
            {
                g.clean_exit_error(&format!(
                    "Running clipmodel to extract seed for view {view_str}"
                ));
            }
            let mut num_peaks = -1;
            let mut found = false;
            let points = sink
                .lock()
                .expect("clipmodel result")
                .points_reduced
                .clone();
            if let Some(&(_, kept)) = points.first() {
                {
                    num_peaks = kept;
                    if (num_peaks as f64) < (num_peaks_tot as f64 / g.num_seed_views as f64) / 3. {
                        exit_error(&format!(
                            "Found only {num_peaks} out of {num_peaks_tot} points on view {}",
                            seed_vw + 1
                        ));
                    }
                    found = true;
                }
            }
            if !found {
                g.clean_exit_error(&format!(
                    "Clipmodel did not give expected output when extracting points for view {view_str}"
                ));
            }

            let mut local_size: i64 = 1000;
            let mut local_track = 0;
            if num_peaks >= min_points_for_local {
                local_track = 1;
                let density = num_peaks as f64 / (g.nx as f64 * g.ny as f64);
                local_size = (min_local_size as i64)
                    .max(((opt_local_points * opt_local_size / density).powf(0.333)) as i64);
                if density * (local_size as f64).powi(2) < min_local_points {
                    local_size = (min_local_points / density).sqrt() as i64;
                }
                if local_size as f64 > 0.8 * g.nx as f64 && local_size as f64 > 0.8 * g.ny as f64 {
                    local_track = 0;
                }
            }

            let mut sedcom = vec![
                format!("?^InputSeedModel?s?[ \t].*? {}?", tmpseed[ind]),
                format!("?^OutputModel?s?[ \t].*? {}?", tmptrack[ind]),
                "?^SkipViews?d".to_owned(),
                "?^RoundsOfTracking?s?[ \t].*? 2?".to_owned(),
                format!("?^LocalAreaTracking?s?[ \t].*? {local_track}?"),
                format!("?^LocalAreaTargetSize?s?[ \t].*? {local_size}?"),
                format!(
                    "?^MinTiltRangeToFindAngles?s?[ \t].*? {}?",
                    py_str_float(min_do_tilt)
                ),
                format!("?^OutputModel?a?ElongationOutputFile {}?", tmpelong[ind]),
                format!("?^OutputModel?a?SkipViews {skip_list}?"),
            ];
            if two_surf != 0 {
                sedcom.push(format!("?^OutputModel?a?XYZOutputFile {}?", tmpxyzpt[ind]));
            }
            if !adjust_sed.is_empty() {
                sedcom.extend(adjust_sed.iter().cloned());
            }

            let sedlines = pysed(
                &sedcom,
                PysedSrc::Lines(&tracklines),
                None,
                false,
                '?',
                false,
            )
            .ok()
            .flatten()
            .unwrap_or_default();
            prnstr(
                &format!("RUNNING BEADTRACK with seed from view {view_str}"),
                "\n",
                true,
            );
            if run_cmd("beadtrack -StandardInput", Some(&sedlines), None, None, &[]).is_err() {
                g.clean_exit_error(&format!("Running beadtrack with seed from view {view_str}"));
            }

            // Convert point list to model file
            if two_surf != 0 {
                let pointcom = format!(
                    "point2model -values -1 -sphere {} \"{}\" \"{}\"",
                    ((g.bead_size + 2.) / 2.) as i64,
                    tmpxyzpt[ind],
                    tmpxyzmod[ind]
                );
                if run_cmd(&pointcom, None, None, None, &[]).is_err() {
                    g.clean_exit_error(&format!(
                        "Running point2model with XYZ data from view {view_str}"
                    ));
                }
            }
        }
    }

    // The info file can now be saved for a new run
    if !old_valid {
        let mut newinfo = vec![
            format!("PID {}", g.pid),
            format!("TrackCom {trackcom}"),
            format!("TrackTime {track_mtime}"),
            format!("ImageTime {image_mtime}"),
            format!("BeadSize {bead_str}"),
            format!("MinGuess {}", g.guess),
            format!("PeakFraction {}", py_str_float(g.peak_fraction)),
            format!("Spacing {spacing_str}"),
            format!("TwoSurf {two_surf}"),
            format!("NumPeaks {num_peaks_tot}"),
            format!("NumViews {}", g.num_seed_views),
            format!("ViewsEntered {views_entered}"),
            format!("ZeroShiftsFrac {}", py_str_float(shifts_frac)),
            format!(
                "AdjustSizes {} {}",
                g.adjust_size,
                match g.new_bead_size {
                    Some(size) => py_str_float(size),
                    None => "0".to_owned(),
                }
            ),
        ];
        if !g.bound_model.is_empty() {
            newinfo.push(format!("BoundFile {}", g.bound_model));
            newinfo.push(format!("BoundTime {bound_mtime}"));
        }
        if !shifts_line.is_empty() {
            newinfo.push(format!("ShiftsNearZeroTilt {shifts_line}"));
        }
        let _ = write_text_file(&info_name, &newinfo, false);
        // Defined behaviour: the saved command file itself (the source passes
        // the name as a string, so it removes every one-character file name
        // in it instead; BUGS.md, autofidseed).
        cleanup_files(std::slice::from_ref(&com_saved));
        // `shutil.copyfile`: the contents only
        if std::fs::read(&trackcom)
            .and_then(|bytes| std::fs::write(&com_saved, bytes))
            .is_err()
        {
            prnstr(
                &format!(
                    "WARNING: autofidseed - failed to copy {trackcom} to {}",
                    g.tmpdir
                ),
                "\n",
                false,
            );
        }
    }

    // Sort the beads onto two surfaces for each model
    if two_surf != 0 {
        for ind in 0..g.num_seed_views {
            if drop_list.contains(&ind) || ignore_list.contains(&ind) {
                continue;
            }
            let ind = ind as usize;
            let view_str = (g.included_views[g.seed_views[ind] as usize] + 1).to_string();
            let mut sortcom = vec![
                format!("TextFileWithSurfaces {}", tmpsurf[ind]),
                "ValuesToRestrainSorting 1".to_owned(),
                "FlipYandZ 0".to_owned(),
                format!("InputFile {}", tmpxyzmod[ind]),
                format!("OutputFile {}", tmpsortmod[ind]),
            ];
            if subarea_size != 0 {
                sortcom.push(format!("SubareaSize {subarea_size}"));
            } else {
                sortcom.push(format!(
                    "PickAreasMinNumAndSize {min_subarea_num} {min_subarea_size}"
                ));
            }
            prnstr(" ", "\n", false);
            prnstr(
                &format!("RUNNING SORTBEADSURFS with XYZ positions from view {view_str}"),
                "\n",
                false,
            );
            if run_cmd(
                "sortbeadsurfs -Stand",
                Some(&sortcom),
                Some("stdout"),
                None,
                &[],
            )
            .is_err()
            {
                g.cleanup(&CLEAN_EXTS, &g.pid);
                prnstr(
                    &format!(
                        "{}Running sortbeadsurfs with XYZ values from view {view_str}",
                        g.prefix
                    ),
                    "\n",
                    false,
                );
                exit_from_imod_error(PROGNAME);
            }
        }
    }

    // Now run pickbestseed; for this we need the seed view with minimum tilt
    let mut lowest_tilt: f64 = 1000.;
    let mut zero_view: i32 = 0;
    for ind in 0..g.num_seed_views {
        let view = g.seed_views[ind as usize];
        if !drop_list.contains(&ind) && g.tilt_included[view as usize].abs() < lowest_tilt.abs() {
            zero_view = view;
            lowest_tilt = g.tilt_included[zero_view as usize];
        }
    }

    let mut pickcom = vec![
        format!("OutputSeedModel {seed_file}"),
        format!("ImageSizeXandY {} {}", g.nx, g.ny),
        format!("BordersInXandY {xborder} {yborder}"),
        format!("BeadSize {}", py_str_float(g.bead_size)),
        format!("MiddleZvalue {}", g.included_views[zero_view as usize]),
        format!("CandidateModel {}/{cand_model}", g.tmpdir),
    ];
    for ind in 0..g.num_seed_views {
        if !drop_list.contains(&ind) {
            let i = ind as usize;
            pickcom.push(format!("TrackedModel {}", tmptrack[i]));
            pickcom.push(format!("ElongationFile {}", tmpelong[i]));
            pickcom.push(format!(
                "SeedZvalue {}",
                g.included_views[g.seed_views[i] as usize]
            ));
            if !ignore_list.contains(&ind) {
                pickcom.push(format!("SurfaceFile {}", tmpsurf[i]));
            }
        }
    }

    if two_surf != 0 {
        pickcom.push("TwoSurfaces".to_owned());
    }
    if append_to_seed != 0 {
        pickcom.push("AppendToSeedModel".to_owned());
    }
    if !g.bound_model.is_empty() {
        pickcom.push(format!("BoundaryModel {}", g.bound_model));
        if g.exclude_areas != 0 {
            pickcom.push("ExcludeInsideAreas 1".to_owned());
        }
    }
    if clustered != 0 {
        pickcom.push(format!("ClusteredPointsAllowed {clustered}"));
    }
    if elongated != 0 {
        pickcom.push(format!("ElongatedPointsAllowed {elongated}"));
    }
    if (clustered != 0 || elongated != 0) && low_target != 0. {
        pickcom.push(format!(
            "LowerTargetForClustered {}",
            py_str_float(low_target)
        ));
    }
    if let Some(rotation) = rotation_arr.as_ref().filter(|values| !values.is_empty()) {
        pickcom.push(format!("RotationAngle {}", py_str_float(rotation[0])));
        pickcom.push(format!("HighestTiltAngle {}", py_str_float(highest_tilt)));
    }

    // Process the additional options if any
    let mut weight_entered = false;
    if !pick_options.is_empty() {
        for opt in pick_options.split(" -") {
            pickcom.push(opt.trim_start_matches('-').to_owned());
            let opt_split: Vec<&str> = opt.trim_start_matches('-').split_whitespace().collect();
            let Some(first) = opt_split.first() else {
                eprintln!("IndexError: list index out of range");
                std::process::exit(1)
            };
            if "WeightsForScore".starts_with(first) || "weights".starts_with(first) {
                weight_entered = true;
            }
        }
    }

    // Defined behaviour: the lower-target rerun replaces this entry (the
    // source replaces `pickcom[-1]`, which is the weights line when one
    // follows; BUGS.md, autofidseed).
    let target_index = pickcom.len();
    if g.target_density > 0. {
        pickcom.push(format!(
            "TargetDensityOfBeads {}",
            py_str_float(g.target_density)
        ));
    } else {
        pickcom.push(format!("TargetNumberOfBeads {}", g.target_number as i64));
    }

    // Unless weights were entered, adjust the weights for large beads so that
    // distance-based measures have appropriately less weight
    if !weight_entered && size_for_weights > min_size_for_weight_scaling {
        let scale = min_size_for_weight_scaling / size_for_weights;
        pickcom.push(format!(
            "WeightsForScore 1.,1.,{}",
            c_format("%.4f,%.4f", &[CArg::Dbl(scale), CArg::Dbl(scale)])
        ));
    }

    prnstr(" ", "\n", false);
    prnstr("RUNNING PICKBESTSEED", "\n", false);
    prnstr(" ", "\n", false);
    // The script parses pickbestseed's `Final:` line
    let pick_failed = || -> ! {
        g.cleanup(&CLEAN_EXTS, &g.pid);
        prnstr(
            &format!("{}Running pickbestseed on tracked models", g.prefix),
            "\n",
            false,
        );
        exit_from_imod_error(PROGNAME)
    };
    // Direct call: pickbestseed records the counts of its `Final:` line
    // (`total points accepted = N  -  on bottom = B , on top = T`), used here
    // in place of parsing that line (`autofidseed:1303-1318`).
    let sink = Arc::new(Mutex::new(PickbestseedResult::default()));
    let recorder = Arc::clone(&sink);
    let pick_lines = match call_own_program(
        "pickbestseed -StandardInput",
        &["pickbestseed", "-StandardInput"],
        Some(&pickcom),
        true,
        move || pickbestseed_recording(recorder),
    ) {
        Ok((_, lines)) => lines,
        Err(_) => pick_failed(),
    };
    for l in &pick_lines {
        prnstr(l, "", false);
    }
    let result = sink.lock().expect("pickbestseed result").clone();
    if two_surf != 0 && max_major_minor > 0. {
        let mut found = false;
        if let Some(total) = result.final_total {
            {
                // The script collected the numbers after each `=` from the end
                // of the line: on top, on bottom, total.
                let mut num_beads: Vec<i32> = Vec::new();
                if let Some((bottom, top)) = result.on_bottom_top {
                    num_beads.push(top);
                    num_beads.push(bottom);
                }
                num_beads.push(total);
                if num_beads.len() < 3 {
                    exit_error(
                        &("Unable to find enough point numbers in last line to assess ".to_owned()
                            + "ratio between the surfaces"),
                    );
                }
                let major = num_beads[0].max(num_beads[1]);
                let minor = num_beads[0].min(num_beads[1]);
                if major as f64 > max_major_minor * minor as f64 + 1. {
                    prnstr(" ", "\n", false);
                    prnstr(
                        &("The ratio between surfaces is too high; running again with ".to_owned()
                            + "lower target"),
                        "\n",
                        false,
                    );
                    prnstr(" ", "\n", false);
                    let new_target = (2. * max_major_minor * minor as f64) as i64 + 1;
                    pickcom[target_index] = format!("TargetNumberOfBeads {new_target}");
                    pickcom.push("LimitMajorityToTarget".to_owned());
                    if run_cmd(
                        "pickbestseed -StandardInput",
                        Some(&pickcom),
                        Some("stdout"),
                        None,
                        &[],
                    )
                    .is_err()
                    {
                        pick_failed();
                    }
                }

                found = true;
            }
        }
        if !found {
            // ELSE ON FOR
            exit_error("Unable to find line starting with Final: for final counts");
        }
    }

    g.cleanup(&CLEAN_EXTS, &g.pid);
    let _ = std::io::stdout().flush();
    0
}
