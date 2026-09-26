//! Translation of `IMOD/pysrc/tiltmatch.py`, the module that searches pairs
//! of views from two tilt series for the best match (used by
//! `matchrotpairs` and `transferfid`).
//!
//! One function per `def`.  The module globals (`tmpRoot`, `pid`,
//! `leaveTmp`, `tmpDir`, `modPrefix`, `modProgname`) live in a
//! `thread_local!` so a program run in process starts from the module's
//! initial values, as a fresh interpreter would.  The module constants are
//! `const`s.  Python ints are `i64`, Python floats `f64`.
//!
//! **`KeyboardInterrupt`.**  The source's `searchPairs` wraps its whole loop
//! in `try: ... except KeyboardInterrupt: pass`, so Ctrl-C ends the search
//! and returns what was found so far.  Python raises it in the interpreter
//! when SIGINT arrives, which in practice is while `runcmd` waits for a
//! child (the caller has done `passOnKeyInterrupt(True)`).  The translation
//! records SIGINT in [`KEY_INTERRUPT`] (the handler is installed by the
//! calling program) and takes the `except` arm at the next statement after
//! a `runcmd`, *before* looking at that command's status -- an interrupted
//! child is Python's `KeyboardInterrupt`, not an `ImodpyError`.
//!
//! **Uncaught exceptions** (`UnboundLocalError` on `asecBest` when no pair
//! was ever scored, `IndexError` on an empty transform file) end the process
//! with the traceback's last line on standard error and status 1, as the
//! interpreter does.

use super::batchruntomo::py_str_float;
use super::imodpy::{
    ImodpyError, call_own_program, cleanup_files, exit_from_imod_error, get_err_strings, glob_glob,
    imod_temp_dir, prnstr, py_int, read_text_file, run_cmd, write_text_file,
};
use super::pip::{exit_error, pip_get_boolean, pip_get_integer, pip_get_two_floats};
use crate::imod::flib::image::xfsimplex::{xfsimplex_final_line, xfsimplex_recording};
use std::cell::RefCell;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

// Filter for tiltxcorr
const SIGMA1: f64 = 0.03;
const SIGMA2: f64 = 0.05;
const RADIUS2BASE: f64 = 0.25;

// Search limit parameters
const XY_LIMIT_FRAC: i64 = 5;
const ROT_LIMIT: i64 = 15;
const OTHER_LIMITS: &str = ".1,.05,2";

// Number of wins more in one rotation direction than the other that will
// make it abandon the other direction
const DIRECTION_WIN_THRESH: i64 = 4;
const BIG_WIN_FACTORS: [f64; 3] = [1.5, 2., 3.];
const BIG_WIN_THRESH: [i64; 3] = [3, 2, 1];

// Criteria for when to stop doing rotation scan and pick a mean or median angle
const NUM_ANG_FOR_MEAN: usize = 4;
const NUM_ANG_FOR_MEDIAN: usize = 5;
const ANGLE_MEAN_RANGE_CRIT: f64 = 1.;
const ANGLE_MEDIAN_CRIT: f64 = 0.8;

/// Rust-only: set by the calling program's SIGINT handler; see the module
/// documentation.
pub static KEY_INTERRUPT: AtomicBool = AtomicBool::new(false);

/// Rust-only: the SIGINT handler a caller installs after
/// `passOnKeyInterrupt(True)`, standing for Python's default
/// `KeyboardInterrupt` raising.
pub extern "C" fn key_interrupt(_signal: libc::c_int) {
    KEY_INTERRUPT.store(true, Ordering::SeqCst);
}

/// The module globals (`tiltmatch.py:35-41`).
struct ModuleGlobals {
    tmp_root: String,
    pid: String,
    leave_tmp: i32,
    tmp_dir: String,
    mod_prefix: String,
    mod_progname: String,
}

thread_local! {
    static GLOBALS: RefCell<ModuleGlobals> = RefCell::new(ModuleGlobals {
        tmp_root: String::new(),
        pid: String::new(),
        leave_tmp: 0,
        tmp_dir: String::new(),
        mod_prefix: "ERROR: ".to_owned(),
        mod_progname: String::new(),
    });
}

/// Rust-only: an exception the source does not catch.
fn uncaught(message: &str) -> ! {
    let _ = std::io::stdout().flush();
    eprintln!("Traceback (most recent call last):");
    eprintln!("{message}");
    std::process::exit(1)
}

/// Matches `cleanup` (`tiltmatch.py:44`).
pub fn cleanup() {
    let (leave_tmp, tmp_dir, pid, tmp_root) = GLOBALS.with(|g| {
        let g = g.borrow();
        (
            g.leave_tmp,
            g.tmp_dir.clone(),
            g.pid.clone(),
            g.tmp_root.clone(),
        )
    });
    if leave_tmp != 0 {
        prnstr(
            &format!("Temporary files left in {tmp_dir} as *{pid}"),
            "\n",
            false,
        );
    } else {
        let clean_list = glob_glob(&format!("{tmp_root}*{pid}*"));
        cleanup_files(&clean_list);
    }
}

/// Matches `cleanExitError` (`tiltmatch.py:53`).
pub fn clean_exit_error(message: &str) -> ! {
    cleanup();
    let (mod_prefix, mod_progname) = GLOBALS.with(|g| {
        let g = g.borrow();
        (g.mod_prefix.clone(), g.mod_progname.clone())
    });
    if !message.is_empty() {
        prnstr(&format!("{mod_prefix}{message}"), "\n", false);
    }
    exit_from_imod_error(&mod_progname)
}

/// Matches `getTempNames` (`tiltmatch.py:60`).
pub fn get_temp_names(progname: &str) -> String {
    GLOBALS.with(|g| {
        let mut g = g.borrow_mut();
        g.mod_prefix = format!("ERROR: {progname} - ");
        g.mod_progname = progname.to_owned();

        g.tmp_dir = imod_temp_dir();
        g.pid = format!(".{}", std::process::id());
        g.tmp_root = progname.to_owned();
        if !g.tmp_dir.is_empty() {
            g.tmp_root = format!("{}/{progname}.", g.tmp_dir);
        }

        format!("{}minxf{}", g.tmp_root, g.pid)
    })
}

/// Matches `getTempComponents` (`tiltmatch.py:76`).
pub fn get_temp_components() -> (String, String, String) {
    GLOBALS.with(|g| {
        let g = g.borrow();
        (g.tmp_root.clone(), g.tmp_dir.clone(), g.pid.clone())
    })
}

/// Matches `makeSecList` (`tiltmatch.py:81`).
pub fn make_sec_list(zero_a: i64, nviews_a: i64) -> (Vec<i64>, i64) {
    let mut asec_list = vec![zero_a];
    let mut ind = 1;
    while ind < nviews_a {
        for dir in [1, -1] {
            if (asec_list.len() as i64) < nviews_a {
                asec_list.push(zero_a + ind * dir);
            }
        }
        ind += 1;
    }
    let min = *asec_list.iter().min().expect("non-empty list");
    (asec_list, min)
}

/// Matches `evaluateWins` (`tiltmatch.py:94`).  The two lists are updated
/// in place, as the source's list arguments are.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_wins(
    pm0: f64,
    pm1: f64,
    mut plus_win: i64,
    mut minus_win: i64,
    mut pm_start: i64,
    mut pm_end: i64,
    plus_big: &mut [i64],
    minus_big: &mut [i64],
) -> (i64, i64, i64, i64) {
    if pm0 < pm1 {
        minus_win += 1;
    }
    if pm0 > pm1 {
        plus_win += 1;
    }
    if plus_win >= minus_win + DIRECTION_WIN_THRESH {
        pm_start = 1;
    }
    if minus_win >= plus_win + DIRECTION_WIN_THRESH {
        pm_end = 0;
    }
    for ind in 0..BIG_WIN_FACTORS.len() {
        if pm0 * BIG_WIN_FACTORS[ind] < pm1 {
            minus_big[ind] += 1;
        }
        if pm0 > pm1 * BIG_WIN_FACTORS[ind] {
            plus_big[ind] += 1;
        }
        if plus_big[ind] >= BIG_WIN_THRESH[ind] && minus_win == 0 {
            pm_start = 1;
        }
        if minus_big[ind] >= BIG_WIN_THRESH[ind] && plus_win == 0 {
            pm_end = 0;
        }
    }
    (plus_win, minus_win, pm_start, pm_end)
}

/// Rust-only: Python's two-argument `min(a, b)`, which keeps `a` unless
/// `b < a`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Rust-only: `'{:W.Pf}'.format(x)` of a Python float (`nan`/`inf` spelled
/// as Python spells them).
pub fn py_fixed(value: f64, width: usize, precision: usize) -> String {
    let text = if value.is_nan() {
        "nan".to_owned()
    } else if value.is_infinite() {
        if value < 0. { "-inf" } else { "inf" }.to_owned()
    } else {
        format!("{value:.precision$}")
    };
    format!("{text:>width$}")
}

/// Rust-only: `float(text)`; `None` is the ValueError.
fn py_float(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    let body = lower.trim_start_matches(['+', '-']);
    if body == "nan" || body == "inf" || body == "infinity" {
        return trimmed.parse::<f64>().ok();
    }
    if trimmed.is_empty()
        || !trimmed
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'))
    {
        return None;
    }
    trimmed.parse::<f64>().ok()
}

/// Rust-only: the `except KeyboardInterrupt` arm of `searchPairs`.
struct Interrupted;

/// Matches `searchPairs` (`tiltmatch.py:116`).  Returns `(asecBest,
/// bsecBest, diffList, allXfList)`.
#[allow(clippy::too_many_arguments)]
pub fn search_pairs(
    progname: &str,
    zero_a: i64,
    zero_b: i64,
    nviews_a: i64,
    nviews_b: i64,
    image_a: &str,
    image_b: &str,
    nxa: i64,
    nxb: i64,
    nya: i64,
    nyb: i64,
    aa: &str,
    bb: &str,
    lowest_xf_file: &str,
    lowest_asec: i64,
    lowest_bsec: i64,
    distort: &str,
    bilinear: i64,
    all_xf_out: i32,
    expand_afac: i64,
) -> (i64, i64, Vec<Vec<f64>>, Vec<Vec<String>>) {
    // Get common temp filenames and the additional one needed here
    let tmp_minxf = get_temp_names(progname);
    let (tmp_root, _tmp_dir, pid) = get_temp_components();
    let tmp_imgb = format!("{tmp_root}imgb{pid}");
    let tmp_img_amr = format!("{tmp_root}imgamr{pid}");
    let tmp_img_apr = format!("{tmp_root}imgapr{pid}");
    let tmp_img_amx = format!("{tmp_root}imgamx{pid}");
    let tmp_img_apx = format!("{tmp_root}imgapx{pid}");
    let tmp_xcxf = format!("{tmp_root}xcxf{pid}");
    let tmp_xf1 = format!("{tmp_root}xf1{pid}");
    let tmp_xf2 = format!("{tmp_root}xf2{pid}");
    let tmp_rot90 = format!("{tmp_root}rot90{pid}");

    // Define search direction, mirroring, whether to use midas
    let mut pm_start: i64 = 0;
    let mut pm_end: i64 = 1;
    let angle = pip_get_integer("AngleOfRotation", 0).unwrap_or(0);
    if angle < 0 {
        pm_end = 0;
    }
    if angle > 0 {
        pm_start = 1;
    }
    let midas = pip_get_boolean("RunMidas", 0).unwrap_or(0);
    let mirror = pip_get_integer("MirrorXaxis", 0).unwrap_or(0);
    let leave_tmp = pip_get_boolean("LeaveTempFiles", 0).unwrap_or(0);
    GLOBALS.with(|g| g.borrow_mut().leave_tmp = leave_tmp);
    let mut mir_start: i64 = 0;
    let mut mir_end: i64 = 1;
    if mirror > 0 {
        mir_start = 1;
    }
    if mirror < 0 {
        mir_end = 0;
    }

    // Get rotation scan variables and set 3-state flag for find, use, or skip angle
    let (scan_rot_max, scan_rot_step) =
        pip_get_two_floats("ScanRotationMaxAndStep", (20., 4.)).unwrap_or((20., 4.));
    let mut find_rotation: i64 = 1;
    let mut best_angles: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut best_rotation: f64 = 0.;
    if scan_rot_max == 0. {
        find_rotation = 0;
    } else if scan_rot_step == 0. {
        find_rotation = -1;
        best_rotation = scan_rot_max;
    }

    // Set the binning needed to get image size to 512 or less unless the size is
    // bigger than 4K, in which case bin to 1024.  Limit binning to 4 between 2048
    // and 4096.  Set limits on X/Y in search
    let size = ((nxa * nya) as f64).sqrt().floor() as i64;
    let mut limit = 512;
    if size >= 4096 {
        limit = 1024;
    }
    let mut simplex_binning = (size + limit - 1).div_euclid(limit);
    if size < 4096 && simplex_binning > 4 {
        simplex_binning = 4;
    }
    let xlimit = nxb.div_euclid(XY_LIMIT_FRAC);
    let ylimit = nyb.div_euclid(XY_LIMIT_FRAC);

    // Control the binning in tiltxcorr since speed is more important than high precision
    let xcorr_binning = 12.min(1.max((size as f64 / 900.).round_ties_even() as i64));

    // Relax the filter above binning of 5, where antialiasing will be applied
    let mut radius2 = RADIUS2BASE;
    if xcorr_binning > 5 {
        radius2 = py_min(0.5, RADIUS2BASE + 0.05 * (xcorr_binning - 5) as f64);
    }

    // Set up lists to do sections from center out
    let (asec_list, asec_start) = make_sec_list(zero_a, nviews_a);
    let (bsec_list, bsec_start) = make_sec_list(zero_b, nviews_b);

    if all_xf_out != 0 && (midas == 0 && pm_start == 0 && pm_end == 1) {
        exit_error(
            "You need to use midas or specify the rotation direction to get all transforms written",
        );
    }
    if all_xf_out != 0 && mirror == 0 {
        mir_end = 0;
    }

    // set up for midas
    //
    if midas != 0 {
        if pm_end != pm_start {
            pm_end = 0;
        }
    } else {
        prnstr(
            "Finding the best matched pair of views in the two series:",
            "\n",
            false,
        );
        prnstr("              (Type Ctrl-C to end search)", "\n", false);
    }

    // Loop on section from b, section from a, and -/+90 rotations
    let mut diff_min: f64 = 2000000000.;
    let mut diff_lowest_tilt = diff_min;
    let mut plus_win: i64 = 0;
    let mut minus_win: i64 = 0;
    let ind = BIG_WIN_FACTORS.len();
    let mut plus_big = vec![0i64; ind];
    let mut minus_big = vec![0i64; ind];
    let mut mirror_win: i64 = 0;
    let mut regular_win: i64 = 0;
    let mut mirror_big = vec![0i64; ind];
    let mut regular_big = vec![0i64; ind];
    let reg_mir_text = ["regular", "mirror "];
    let mut rot90sec: [i64; 4] = [-1; 4];
    let tmp_img_amp = [&tmp_img_amr, &tmp_img_apr, &tmp_img_amx, &tmp_img_apx];

    // Make list of unfound differences.
    // You cannot use [[-1] * nA] * nB because that makes shallow copies of the inner list
    let mut diff_list: Vec<Vec<f64>> = Vec::new();
    for _ in 0..nviews_b {
        diff_list.push(vec![-1.; nviews_a.max(0) as usize]);
    }
    let mut all_xf_list: Vec<Vec<String>> = Vec::new();
    if all_xf_out != 0 {
        for _ in 0..nviews_b {
            all_xf_list.push(vec!["1 0 0 1 0 0".to_owned(); nviews_a.max(0) as usize]);
        }
    }

    let mut asec_best: Option<i64> = None;
    let mut bsec_best: Option<i64> = None;
    let mut best_ang: Option<f64> = None;
    let mut tmp_init_xf: Option<String> = None;

    // `runcmd` with the interrupt test first: a SIGINT during the command is
    // the KeyboardInterrupt, whatever the command's status
    let run = |command: &str,
               input: Option<&[String]>|
     -> Result<Result<Vec<String>, ImodpyError>, Interrupted> {
        let result = run_cmd(command, input, None, None, &[]).map(Option::unwrap_or_default);
        if KEY_INTERRUPT.load(Ordering::SeqCst) {
            return Err(Interrupted);
        }
        Ok(result)
    };

    let _: Result<(), Interrupted> = (|| {
        for asec_ind in 0..nviews_a {
            let asec = asec_list[asec_ind as usize];
            for bsec_ind in 0..nviews_b {
                let bsec = bsec_list[bsec_ind as usize];
                let mut plus_minus = pm_start;
                let mut pm_diffs: [f64; 4] = [1.0e37; 4];
                while plus_minus <= pm_end {
                    let mut mir_ind = mir_start;
                    while mir_ind <= mir_end {
                        let pmrx_ind = (plus_minus + 2 * mir_ind) as usize;
                        let tmp_imga = tmp_img_amp[pmrx_ind];
                        let rotstr;
                        let pm_angle: i64;
                        if plus_minus != 0 {
                            if mir_ind != 0 {
                                rotstr = format!("0 -{expand_afac} -{expand_afac} 0 0 0");
                            } else {
                                rotstr = format!("0 -{expand_afac} {expand_afac} 0 0 0");
                            }
                            pm_angle = 90;
                        } else {
                            if mir_ind != 0 {
                                rotstr = format!("0 {expand_afac} {expand_afac} 0 0 0");
                            } else {
                                rotstr = format!("0 {expand_afac} -{expand_afac} 0 0 0");
                            }
                            pm_angle = -90;
                        }

                        let _ = write_text_file(&tmp_rot90, &[rotstr], false);
                        if rot90sec[pmrx_ind] != asec {
                            // extract the rotated section from A if it is needed
                            match run(
                                &format!(
                                    "newstack -sec {asec} -xform {tmp_rot90} -size {nxb},{nyb} -use 0 {distort} \"{image_a}\" \"{tmp_imga}\""
                                ),
                                None,
                            )? {
                                Ok(_) => rot90sec[pmrx_ind] = asec,
                                Err(_) => clean_exit_error(&format!(
                                    "Extracting rotated section from {aa}"
                                )),
                            }
                        }

                        let ref_sec;
                        let ref_image: &str;
                        if !distort.is_empty() {
                            // If undistorting, need to extract the reference section too,
                            // otherwise the reference is the stack
                            match run(
                                &format!(
                                    "newstack -sec {bsec} -use 0 {distort} \"{image_b}\" \"{tmp_imgb}\""
                                ),
                                None,
                            )? {
                                Ok(_) => {
                                    ref_sec = 0;
                                    ref_image = &tmp_imgb;
                                }
                                Err(_) => clean_exit_error(&format!(
                                    "Extracting distortion-corrected section from {bb}"
                                )),
                            }
                        } else {
                            ref_sec = bsec;
                            ref_image = image_b;
                        }

                        if midas != 0 {
                            if asec_ind == 0 && bsec_ind == 0 && mir_ind == mir_start {
                                // first time, run midas
                                if bilinear != 0 {
                                    prnstr(
                                        "Starting midas - you should align as well as possible,",
                                        "\n",
                                        false,
                                    );
                                } else {
                                    prnstr(
                                        "Starting midas - you should align translation and rotation,",
                                        "\n",
                                        false,
                                    );
                                }
                                prnstr(
                                    " and save the transform to the already-defined output file",
                                    "\n",
                                    false,
                                );
                                prnstr(" ", "\n", false);
                                if run(
                                    &format!(
                                        "midas -D -r \"{ref_image}\" -rz {ref_sec} \"{tmp_imga}\" \"{tmp_xf1}\""
                                    ),
                                    None,
                                )?
                                .is_err()
                                {
                                    let mut true_error = true;
                                    if Path::new(&tmp_xf1).exists() {
                                        // If the xf file exists, then test if there is an ERROR
                                        // output AND see if it has a negative status to ignore
                                        // a Qt crash on exit
                                        let err_str = get_err_strings();
                                        for line in &err_str {
                                            if line.contains("ERROR:") {
                                                break;
                                            }
                                            if line.contains("with status") {
                                                let lsplit: Vec<&str> =
                                                    line.split_whitespace().collect();
                                                if let Some(status) =
                                                    lsplit.last().and_then(|s| py_int(s))
                                                    && status < 0
                                                {
                                                    true_error = false;
                                                }
                                            }
                                        }
                                    }

                                    if true_error {
                                        clean_exit_error("");
                                    }
                                }

                                if !Path::new(&tmp_xf1).exists() {
                                    cleanup();
                                    exit_error("Transform file not found - cannot proceed");
                                }

                                tmp_init_xf = Some(tmp_xf1.clone());
                                prnstr(
                                    "Finding the best matched pair of views in the two series:",
                                    "\n",
                                    false,
                                );
                                prnstr("              (Type Ctrl-C to end search)", "\n", false);
                            }
                        } else {
                            // Run tiltxcorr if no midas
                            let mut angle_opt = String::new();
                            if find_rotation > 0 {
                                angle_opt = format!(
                                    "ScanRotationMaxAndStep {} {}",
                                    py_str_float(scan_rot_max),
                                    py_str_float(scan_rot_step)
                                );
                            } else if find_rotation < 0 {
                                angle_opt = format!(
                                    "ScanRotationMaxAndStep {} 0.",
                                    py_str_float(best_rotation)
                                );
                                best_ang = Some(best_rotation);
                            }
                            let mut xccom = vec![
                                format!("InputFile {tmp_imga}"),
                                format!("OutputFile {tmp_xcxf}"),
                                "TiltAngles 0".to_owned(),
                                format!("ReferenceFile {ref_image}"),
                                format!("ReferenceView {}", ref_sec + 1),
                                format!("BinningToApply {xcorr_binning}"),
                                format!("FilterRadius2 {}", py_str_float(radius2)),
                                format!("FilterSigma1 {}", py_str_float(SIGMA1)),
                                format!("FilterSigma2 {}", py_str_float(SIGMA2)),
                            ];
                            if !angle_opt.is_empty() {
                                xccom.push(angle_opt);
                            }
                            if xcorr_binning > 5 {
                                xccom.push("AntialiasFilter 4".to_owned());
                            }

                            let xc_lines = match run("tiltxcorr -StandardInput", Some(&xccom))? {
                                Ok(lines) => lines,
                                Err(_) => clean_exit_error(
                                    "Running tiltxcorr to get initial correlation alignment",
                                ),
                            };

                            // Extract rotation from the tiltxcorr output
                            if find_rotation > 0 {
                                let mut found = false;
                                for line in &xc_lines {
                                    if line.contains("Best angle in") {
                                        let ind = line.rfind('=').map_or(-1, |i| i as i64);
                                        if ind > 0 {
                                            let value = match py_float(&line[ind as usize + 1..]) {
                                                Some(value) => value,
                                                None => exit_error(
                                                    "Converting best angle output from Tiltxcorr to float",
                                                ),
                                            };
                                            best_ang = Some(value);
                                            best_angles[pmrx_ind].push(value);
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                if !found {
                                    // ELSE ON FOR
                                    exit_error("Cannot find rotation angle in output of Tiltxcorr");
                                }
                            }

                            // Run xfsimplex looking for rotation only for legacy run (0,0
                            // entered for -scan) or if angle being used is at end of scan range
                            if find_rotation == 0
                                || ((best_ang
                                    .unwrap_or_else(|| {
                                        uncaught("NameError: name 'bestAng' is not defined")
                                    })
                                    .abs()
                                    - scan_rot_max)
                                    .abs()
                                    < 0.1
                                    && scan_rot_step != 0.)
                            {
                                let xfcom = vec![
                                    format!("AImageFile {ref_image}"),
                                    format!("BImageFile {tmp_imga}"),
                                    format!("OutputFile {tmp_xf1}"),
                                    format!("SectionsToUse {ref_sec} 0"),
                                    format!("InitialTransformFile {tmp_xcxf}"),
                                    "VariablesToSearch 3".to_owned(),
                                    format!("BinningToApply {simplex_binning}"),
                                    format!("LimitsOnSearch {xlimit},{ylimit},{ROT_LIMIT}"),
                                ];
                                tmp_init_xf = Some(tmp_xf1.clone());
                                if run("xfsimplex -StandardInput", Some(&xfcom))?.is_err() {
                                    clean_exit_error("Running first xfsimplex with rotation only");
                                }
                            } else {
                                tmp_init_xf = Some(tmp_xcxf.clone());
                            }
                        }

                        // Run xfsimplex again from there, looking for full transform
                        let xfcom = vec![
                            format!("AImageFile {ref_image}"),
                            format!("BImageFile {tmp_imga}"),
                            format!("OutputFile {tmp_xf2}"),
                            format!("SectionsToUse {ref_sec} 0"),
                            format!(
                                "InitialTransformFile {}",
                                tmp_init_xf.as_deref().unwrap_or_else(|| uncaught(
                                    "NameError: name 'tmpInitXf' is not defined"
                                ))
                            ),
                            "VariablesToSearch 6".to_owned(),
                            format!("LinearInterpolation {bilinear}"),
                            format!("BinningToApply {simplex_binning}"),
                            format!("LimitsOnSearch {xlimit},{ylimit},{ROT_LIMIT},{OTHER_LIMITS}"),
                        ];
                        // Direct call (owner rule, 2026-09-26): xfsimplex records its
                        // closing values, used in place of parsing its report
                        // (`tiltmatch.py:292-296`).  The script took the second word of
                        // the next-to-last line: with natural variables searched (always
                        // here, `VariablesToSearch 6`) that is FORMAT 72's line, whose
                        // text is rebuilt from the values so the difference is rounded
                        // (`f14.7`) and split exactly as printed; otherwise it was
                        // "  FINAL VALUES", which does not convert.
                        let sink = std::sync::Arc::new(std::sync::Mutex::new(None));
                        let recorder = std::sync::Arc::clone(&sink);
                        let called = call_own_program(
                            "xfsimplex -StandardInput",
                            &["xfsimplex", "-StandardInput"],
                            Some(&xfcom),
                            true,
                            move || xfsimplex_recording(recorder),
                        );
                        if KEY_INTERRUPT.load(Ordering::SeqCst) {
                            return Err(Interrupted);
                        }
                        if called.is_err() {
                            clean_exit_error("Running second xfsimplex with full transform")
                        }
                        let simplex = *sink.lock().expect("xfsimplex result");

                        // `simpLines[len(simpLines) - 2]`; anything failing is caught
                        // by `except Exception`
                        let diff = (|| -> Option<f64> {
                            let simplex = simplex?;
                            if !simplex.natural {
                                return None;
                            }
                            let line = xfsimplex_final_line(&simplex);
                            let diffspl: Vec<&str> = line.split_whitespace().collect();
                            py_float(diffspl.get(1)?)
                        })();
                        let diff = match diff {
                            Some(diff) => diff,
                            None => {
                                cleanup();
                                exit_error("Extracting difference value from Xfsimplex output");
                            }
                        };
                        prnstr(
                            &format!(
                                "{aa} {} {bb} {} rotation {pm_angle:3} {} difference {}",
                                asec + 1,
                                bsec + 1,
                                reg_mir_text[mir_ind as usize],
                                py_fixed(diff, 11, 6)
                            ),
                            "",
                            false,
                        );

                        // Accumulate transform list if requested
                        if all_xf_out != 0 {
                            if run(
                                &format!("xfproduct \"{tmp_rot90}\" \"{tmp_xf2}\" \"{tmp_xf1}\""),
                                None,
                            )?
                            .is_err()
                            {
                                clean_exit_error("Taking product of 90 degree and found transform");
                            }
                            let one_xf =
                                read_text_file(&tmp_xf1, None, false, None).unwrap_or_default();
                            let first = one_xf
                                .first()
                                .cloned()
                                .unwrap_or_else(|| uncaught("IndexError: list index out of range"));
                            all_xf_list[(bsec - bsec_start) as usize]
                                [(asec - asec_start) as usize] = first;
                        }

                        // Keep track of minimum and transform there
                        if diff < diff_min {
                            prnstr("*", "\n", true);
                            diff_min = diff;
                            asec_best = Some(asec);
                            bsec_best = Some(bsec);
                            if run(
                                &format!("xfproduct \"{tmp_rot90}\" \"{tmp_xf2}\" \"{tmp_minxf}\""),
                                None,
                            )?
                            .is_err()
                            {
                                clean_exit_error("Taking product of 90 degree and found transform");
                            }
                        } else {
                            prnstr(" ", "\n", true);
                        }

                        //  Accumulate differences from plus and minus
                        pm_diffs[pmrx_ind] = diff;

                        // If XF file at lowest tilt requested, output one for best lowest tilt
                        // pair
                        if !lowest_xf_file.is_empty()
                            && asec == lowest_asec
                            && bsec == lowest_bsec
                            && diff < diff_lowest_tilt
                        {
                            diff_lowest_tilt = diff;
                            if run(
                                &format!(
                                    "xfproduct \"{tmp_rot90}\" \"{tmp_xf2}\" \"{lowest_xf_file}\""
                                ),
                                None,
                            )?
                            .is_err()
                            {
                                clean_exit_error("Taking product of 90 degree and found transform");
                            }
                        }

                        mir_ind += 1; // End of mirror loop
                    }

                    plus_minus += 1; // end of plusMinus loop
                }

                // If there are both plus and minus or regular and mirrored, count who wins
                // and stop doing a consistent loser, or a big loser sooner
                // First evaluate plus and minus
                if pm_start < pm_end {
                    (plus_win, minus_win, pm_start, pm_end) = evaluate_wins(
                        py_min(pm_diffs[0], pm_diffs[2]),
                        py_min(pm_diffs[1], pm_diffs[3]),
                        plus_win,
                        minus_win,
                        pm_start,
                        pm_end,
                        &mut plus_big,
                        &mut minus_big,
                    );
                }

                // Use same logic for regular/mirror
                if mir_start < mir_end {
                    (regular_win, mirror_win, mir_start, mir_end) = evaluate_wins(
                        py_min(pm_diffs[0], pm_diffs[1]),
                        py_min(pm_diffs[2], pm_diffs[3]),
                        regular_win,
                        mirror_win,
                        mir_start,
                        mir_end,
                        &mut regular_big,
                        &mut mirror_big,
                    );
                }

                // `min(pmDiffs)`: the first of the smallest
                let mut min_diff = pm_diffs[0];
                for value in &pm_diffs[1..] {
                    if *value < min_diff {
                        min_diff = *value;
                    }
                }
                diff_list[(bsec - bsec_start) as usize][(asec - asec_start) as usize] = min_diff;

                // Once there is a single direction, see if the angles are consistent enough
                // to stop scanning for them
                let pm_ind = (pm_end + 2 * mir_end) as usize;
                let num_ang = best_angles[pm_ind].len();
                if pm_start == pm_end
                    && mir_start == mir_end
                    && find_rotation > 0
                    && num_ang >= NUM_ANG_FOR_MEAN
                {
                    // `sum()` starts from the int 0
                    let mut sum = 0.;
                    for value in &best_angles[pm_ind] {
                        sum += value;
                    }
                    let mean_angle = sum / num_ang as f64;
                    let mut min_angle = best_angles[pm_ind][0];
                    let mut max_angle = best_angles[pm_ind][0];
                    for value in &best_angles[pm_ind][1..] {
                        if *value < min_angle {
                            min_angle = *value;
                        }
                        if *value > max_angle {
                            max_angle = *value;
                        }
                    }
                    if mean_angle - min_angle < ANGLE_MEAN_RANGE_CRIT
                        && max_angle - mean_angle < ANGLE_MEAN_RANGE_CRIT
                    {
                        best_rotation = mean_angle;
                        find_rotation = -1;
                    } else if num_ang >= NUM_ANG_FOR_MEDIAN {
                        // `list.sort()` is stable and compares with `<` only
                        best_angles[pm_ind].sort_by(|a, b| {
                            if a < b {
                                std::cmp::Ordering::Less
                            } else if b < a {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        });
                        let angles = &best_angles[pm_ind];
                        let median = if num_ang % 2 != 0 {
                            angles[num_ang / 2]
                        } else {
                            (angles[num_ang / 2] + angles[num_ang / 2 - 1]) / 2.
                        };
                        if (median - angles[1] < ANGLE_MEDIAN_CRIT
                            && max_angle - median < ANGLE_MEDIAN_CRIT)
                            || (median - min_angle < ANGLE_MEDIAN_CRIT
                                && angles[num_ang - 2] - median < ANGLE_MEDIAN_CRIT)
                        {
                            best_rotation = median;
                            find_rotation = -1;
                        }
                    }
                }
            }
        }
        Ok(())
    })();

    let asec_best = asec_best.unwrap_or_else(|| {
        uncaught("UnboundLocalError: local variable 'asecBest' referenced before assignment")
    });
    let bsec_best = bsec_best.unwrap_or(0);
    (asec_best, bsec_best, diff_list, all_xf_list)
}
