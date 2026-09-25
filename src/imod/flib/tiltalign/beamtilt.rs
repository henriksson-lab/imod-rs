//! Translation of `IMOD/flib/tiltalign/beamtilt.cpp` — has `searchBeamTilt`
//! and `runMetro` routines.
//!
//! Compiled into `tiltalign` only, so `errorExit` is the `tiltalign` build,
//! `error_exit::<false>`.
//!
//! # File-scope statics and globals
//!
//! - `static AlignVariables *av;` (`beamtilt.cpp:18`), set by
//!   `beamtiltSetPointers`, is a parameter (`alivar.rs`).
//! - `funct` (`funct.cpp:66`), the callback both routines hand to
//!   `metroSearch` and call directly, reaches its `EvalFunct` through the
//!   `sEvalFunct` static; the translated [`funct`] takes the `EvalFunct`,
//!   `AlignVariables` and `ArrayMaxes` explicitly, so both routines take the
//!   `EvalFunct` and `ArrayMaxes` too and pass `metroSearch` a closure over
//!   the three.
//! - `extern double functWallCum; extern int functNumCalls;` are
//!   [`FUNCT_WALL_CUM`]/[`FUNCT_NUM_CALLS`] in `funct.rs`.
//!
//! # `searchBeamTilt`'s first parameter
//!
//! `float &beamTilt` is, at its only call site, `av->beamTilt`
//! (`tiltalign.cpp:905`; the source comment says "It is the variable in
//! alivar so we use it exclusively by reference"), and `funct` reads the same
//! member through `av` during every `runMetro`.  A Rust `&mut` to one field of
//! the `AlignVariables` that `funct` also borrows cannot exist, so the
//! parameter is dropped and the function reads and writes `av.beam_tilt`
//! directly, which is exactly what the aliased reference does.
//!
//! # Arithmetic
//!
//! `sqrt(fnew * rmsScale)` is the C++ `float` overload (`sqrtf` in the
//! reference object); `pow(2., numCuts)` is the `double` one.

use std::io::Write;
use std::sync::atomic::Ordering;

use super::alivar::AlignVariables;
use super::arraymaxes::ArrayMaxes;
use super::evalfunct::EvalFunct;
use super::funct::{FUNCT_NUM_CALLS, FUNCT_WALL_CUM, funct};
use super::utilfuncs::{copy_array, error_exit};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::metro::metro_search;
use crate::imod::libcfshr::minimize1d::minimize1d;
use crate::imod::libcfshr::simplestat::ls_fit2;

/// Original: `beamtiltSetPointers` (`beamtilt.cpp:20`).
///
/// The source stores the pointer in a file-scope static; the functions of this
/// module take it as a parameter instead, so there is nothing to store.
pub fn beamtilt_set_pointers(av_in: &mut AlignVariables) {
    let _ = av_in;
}

/// Original: `searchBeamTilt` (`beamtilt.cpp:38`).
///
/// searchBeamTilt will perform a one - dimensional search for the beam tilt
/// that minimizes the alignment error.  beamTilt (here `av.beam_tilt`, module
/// doc) starts with an initial value and is returned with the final value.
/// `binStepIni` and `binStepFinal` are initial and final step sizes in the
/// binary search.  `scanStep` is the step size in the full scan through the
/// last interval of the binary search.  Other parameters are as passed to
/// [`run_metro`].
///
/// `fmin` is uninitialised in the source until the scan's first point sets it
/// (`j == 2`, which always runs since `numScan >= 3`).
///
/// Upstream, kept as written: `btMax = 5.` is compared with the beam tilt in
/// radians, so the out-of-range guard is 5 radians and its warning prints
/// 286.5 (`BUGS.md`).
#[allow(clippy::too_many_arguments)]
pub fn search_beam_tilt(
    av: &mut AlignVariables,
    eval_funct: &mut EvalFunct,
    mx: &ArrayMaxes,
    bin_step_ini: f32,
    bin_step_final: f32,
    scan_step: f32,
    num_var_search: i32,
    var: &mut [f32],
    varerr: &mut [f32],
    grad: &mut [f32],
    h: &mut [f32],
    if_local: i32,
    facm: f32,
    ncycle: i32,
    rms_scale: f32,
    f_final: &mut f32,
    kount_init: &mut i32,
    metro_error: &mut i32,
    get_fill_points: i32,
) {
    const LIMSCAN: usize = 200;
    let bt_orig: f32;
    let bt_max: f32;
    let mut f_scan = [0f32; LIMSCAN];
    let mut xx = [0f32; LIMSCAN / 2];
    let mut xxsq = [0f32; LIMSCAN / 2];
    let mut brackets = [0f32; 14];
    let mut bt_min: f32 = 0.;
    let mut aa: f32 = 0.;
    let mut bb: f32 = 0.;
    let mut cc: f32 = 0.;
    let xmin: f32;
    let scan_int: f32;
    let mut num_scan: i32;
    let mut nfit: i32;
    let mut i_min: i32;
    let istr: i32;
    let iend: i32;
    let mut kount: i32 = 0;
    let mut num_cuts: i32;
    let mut fnew: f32 = 0.;
    let mut fmin: f32 = 0.;
    let f_above: f32;
    let bt_above: f32;
    let f_below: f32;
    let bt_below: f32;
    let dtor: f32 = 0.0174532_f64 as f32;
    let mut stdout = ImodFile::Stdout;

    bt_max = 5.;
    bt_orig = av.beam_tilt;
    num_cuts = -1;
    loop {
        //
        // restart every run at output of first one
        // 11/29/13: this was probably because of a high risk of error 2 when starting from a
        // previous solution, which is now much reduced, and could be ignored
        if num_cuts >= 0 {
            copy_array(var, 1, num_var_search, varerr, 1);
        }
        run_metro(
            av,
            eval_funct,
            mx,
            num_var_search,
            var,
            varerr,
            grad,
            h,
            if_local,
            facm,
            ncycle,
            1,
            rms_scale,
            &mut fnew,
            &mut kount,
            metro_error,
            0,
            0,
        );
        let _ = stdout.write_all(&c_format_bytes(
            " For beam tilt =%6.2f, %4d cycles,           final   F :  %14.6f\n",
            &[
                CArg::Dbl((av.beam_tilt / dtor) as f64),
                CArg::Int(kount as i64),
                CArg::Dbl((fnew * rms_scale).sqrt() as f64),
            ],
        ));
        //
        // save count and output state of first run
        if num_cuts < 0 {
            *kount_init = kount;
            copy_array(varerr, 1, num_var_search, var, 1);
        }
        //
        // Find the next step from current minimum and test if out of range
        let cur = av.beam_tilt;
        i_min = minimize1d(
            cur,
            fnew,
            bin_step_ini * dtor,
            0,
            &mut num_cuts,
            &mut brackets,
            &mut av.beam_tilt,
        );
        let d = av.beam_tilt - bt_orig;
        if (if d >= 0. { d } else { -d }) > bt_max {
            let _ = stdout.write_all(&c_format_bytes(
                "\nWARNING: No minimum error found for beam tilt change up to %.1f\n\
WARNING: Returning to original beam tilt = %.1f\n\n",
                &[
                    CArg::Dbl((bt_max / dtor) as f64),
                    CArg::Dbl((bt_orig / dtor) as f64),
                ],
            ));
            av.beam_tilt = bt_orig;
            copy_array(var, 1, num_var_search, varerr, 1);
            run_metro(
                av,
                eval_funct,
                mx,
                num_var_search,
                var,
                varerr,
                grad,
                h,
                if_local,
                facm,
                -ncycle,
                1,
                rms_scale,
                f_final,
                &mut kount,
                metro_error,
                0,
                0,
            );
            return;
        }
        //
        // Then test if step size is now small enough
        if bin_step_ini as f64 / 2f64.powf(num_cuts as f64) < 0.98 * bin_step_final as f64 {
            bt_below = brackets[0];
            bt_above = brackets[2];
            f_below = brackets[7];
            f_above = brackets[9];
            break;
        }
    }
    let _ = i_min;
    //
    // Now scan from below to above, find a minimum excluding the endpoints
    num_scan = (((bt_above - bt_below) / (scan_step * dtor)) as f64 + 1. + 0.5).floor() as i32;
    num_scan = if 3
        > (if (LIMSCAN as i32) < num_scan {
            LIMSCAN as i32
        } else {
            num_scan
        }) {
        3
    } else if (LIMSCAN as i32) < num_scan {
        LIMSCAN as i32
    } else {
        num_scan
    };
    scan_int = (bt_above - bt_below) / (num_scan - 1) as f32;
    f_scan[0] = f_below;
    f_scan[(num_scan - 1) as usize] = f_above;
    for j in 2..=num_scan - 1 {
        av.beam_tilt = bt_below + (j - 1) as f32 * scan_int;
        copy_array(var, 1, num_var_search, varerr, 1);
        let mut fs = f_scan[(j - 1) as usize];
        run_metro(
            av,
            eval_funct,
            mx,
            num_var_search,
            var,
            varerr,
            grad,
            h,
            if_local,
            facm,
            -ncycle,
            1,
            rms_scale,
            &mut fs,
            &mut kount,
            metro_error,
            0,
            0,
        );
        f_scan[(j - 1) as usize] = fs;
        let _ = stdout.write_all(&c_format_bytes(
            " For beam tilt =%6.2f, %4d cycles,           final   F :  %14.6f\n",
            &[
                CArg::Dbl((av.beam_tilt / dtor) as f64),
                CArg::Int(kount as i64),
                CArg::Dbl((f_scan[(j - 1) as usize] * rms_scale).sqrt() as f64),
            ],
        ));
        if j == 2 || f_scan[(j - 1) as usize] < fmin {
            fmin = f_scan[(j - 1) as usize];
            bt_min = av.beam_tilt;
            i_min = j;
        }
    }
    //
    // See if the change is monotonic from the minimum; if so fit to 3 points,
    // or 5 if range is not very big, if not fit to up to half the scan
    for i in 1..=num_scan {
        f_scan[(i - 1) as usize] = f_scan[(i - 1) as usize] - fmin;
    }
    nfit = 3;
    if (((if f_below > f_above { f_below } else { f_above }) / fmin) as f64) < 1.05 {
        nfit = 5;
    }
    for i in i_min + 1..=num_scan {
        if f_scan[(i - 1) as usize] < f_scan[(i - 1 - 1) as usize] {
            nfit = num_scan / 2;
        }
    }
    for i in 1..=i_min - 1 {
        if f_scan[(i - 1) as usize] < f_scan[(i + 1 - 1) as usize] {
            nfit = num_scan / 2;
        }
    }
    let lim2 = (LIMSCAN / 2) as i32;
    nfit = if 3 > (if lim2 < nfit { lim2 } else { nfit }) {
        3
    } else if lim2 < nfit {
        lim2
    } else {
        nfit
    };
    istr = if 1 > i_min - nfit / 2 {
        1
    } else {
        i_min - nfit / 2
    };
    iend = if num_scan < i_min + nfit / 2 {
        num_scan
    } else {
        i_min + nfit / 2
    };
    for i in istr..=iend {
        xx[(i + 1 - istr - 1) as usize] = (i - i_min) as f32;
        xxsq[(i + 1 - istr - 1) as usize] = ((i - i_min) * (i - i_min)) as f32;
    }
    //
    // Do fit and compute minimum from derivative, then return to actual
    // minimum if the parabola is upside down or min is outside fit
    ls_fit2(
        &xx,
        &xxsq,
        &f_scan[(istr - 1) as usize..],
        iend + 1 - istr,
        &mut aa,
        &mut bb,
        Some(&mut cc),
    );
    xmin = (i_min as f64 - 0.5 * aa as f64 / bb as f64) as f32;
    // print *,'imin is', iMin, ' fitting ', istr, ' to', iend
    // print *,'coeffs: ', aa, bb, cc, '  xmin', xmin
    if bb < 0. || xmin < istr as f32 || xmin > iend as f32 {
        av.beam_tilt = bt_min;
        if f_below < fmin {
            av.beam_tilt = bt_below;
            fmin = f_below;
        }
        if f_above < fmin {
            av.beam_tilt = bt_above;
        }
        let _ = stdout.write_all(
            b"\nWARNING: Fit to beam tilt scan failed to give minimum; using scan minimum\n\n",
        );
    } else {
        av.beam_tilt = bt_below + (xmin - 1.) * scan_int;
    }
    //
    // Run finally at the solved beam tilt
    copy_array(var, 1, num_var_search, varerr, 1);
    run_metro(
        av,
        eval_funct,
        mx,
        num_var_search,
        var,
        varerr,
        grad,
        h,
        if_local,
        facm,
        -ncycle,
        1,
        rms_scale,
        f_final,
        &mut kount,
        metro_error,
        0,
        get_fill_points,
    );
    let _ = stdout.write_all(&c_format_bytes(
        "\n Solved beam tilt =%6.2f, %4d cycles,        Final   F :  %14.6f\n",
        &[
            CArg::Dbl((av.beam_tilt / dtor) as f64),
            CArg::Int(kount as i64),
            CArg::Dbl((*f_final * rms_scale).sqrt() as f64),
        ],
    ));
    let _ = stdout.flush();
}

/// Original: `runMetro` (`beamtilt.cpp:188`).
///
/// runMetro runs the metro routine, varying the step size as needed and
/// reporting errors as appropriate.  `numVarSearch` is the number of
/// variables; `var` is the variable vector; `varerr` is used for temporary
/// storage of the `var` array between trials; `grad` is an array for
/// gradients; `h` is the array for the Hessian matrix and a few vectors;
/// `ifLocal` is nonzero if doing local alignments; `facm` is the metro factor
/// or initial step size; `ncycle` is the limit on the number of cycles, or the
/// negative of the limit to suppress some output; `ifHush` not equal to zero
/// suppresses the Final F output; `rmsScale` is used to scale the sum squared
/// error before taken sqrt; `fFinal` is the final error measure; `kount` is
/// the cycle count; `metroError` is maintained with a count of total errors;
/// `ignoreError2` can be set true to ignore error 2 when starting from a
/// previous solution; `getFillPoints` true will have it compute fillin points
/// on final call to funct.
#[allow(clippy::too_many_arguments)]
pub fn run_metro(
    av: &mut AlignVariables,
    eval_funct: &mut EvalFunct,
    mx: &ArrayMaxes,
    num_var_search: i32,
    var: &mut [f32],
    varerr: &mut [f32],
    grad: &mut [f32],
    h: &mut [f32],
    if_local: i32,
    facm: f32,
    ncycle: i32,
    if_hush: i32,
    rms_scale: f32,
    f_final: &mut f32,
    kount: &mut i32,
    metro_error: &mut i32,
    ignore_error2: i32,
    get_fill_points: i32,
) {
    const MAX_METRO_TRIALS: i32 = 5;
    let trial_scale: [f32; MAX_METRO_TRIALS as usize] = [1.0, 0.9, 1.1, 0.75, 0.5];
    let mut ier: i32;
    let mut metro_loop: i32;
    let mut f_init: f32 = 0.;
    let mut f: f32 = 0.;
    let eps: f32;
    let mut metro_fac: f32 = 0.;
    let mut stdout = ImodFile::Stdout;
    //double wallStart;
    //
    // save the variable list for multiple trials
    //
    copy_array(varerr, 1, num_var_search, var, 1);
    metro_loop = 1;
    ier = 1;
    eps = 0.00001_f64 as f32;
    //
    // Clear flag for computation of fill-in points
    av.project_fill_points = 0;
    // if (ncycle < 0) eps = eps / 5.
    while metro_loop <= MAX_METRO_TRIALS && ier != 0 && ier != 3 {
        av.first_funct = 1;
        *FUNCT_WALL_CUM.lock().unwrap_or_else(|e| e.into_inner()) = 0.;
        FUNCT_NUM_CALLS.store(0, Ordering::Relaxed);
        //wallStart = wallTime();
        funct(eval_funct, av, mx, num_var_search, var, &mut f_init, grad);
        if metro_loop == 1 && ncycle > 0 {
            let _ = stdout.write_all(&c_format_bytes(
                "\n Variable Metric minimization                  Initial F:   %14.6f\n",
                &[CArg::Dbl((f_init * rms_scale).sqrt() as f64)],
            ));
        }
        //
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Call variable metric minimizer
        //
        metro_fac = facm * trial_scale[(metro_loop - 1) as usize];
        metro_search(
            num_var_search,
            var,
            &mut |n: i32, x: &mut [f32], fv: &mut f32, g: &mut [f32]| {
                funct(eval_funct, av, mx, n, x, fv, g)
            },
            &mut f,
            grad,
            metro_fac,
            eps,
            ncycle,
            &mut ier,
            h,
            kount,
            rms_scale,
        );

        // Metro timing: funct was 3/4 of the time for global run, 1/2 to 2/3 for a run with
        // xyz fixed.  So parallelize it if/when it is is in its final state.
        /*printf("metro time %.4f funct time %.4f  calls %d  ms/call %.2f  %.2f\n",
        wallTime() - wallStart, functWallCum, functNumCalls, 1000. *
         (wallTime() - wallStart - functWallCum) / functNumCalls,
         1000. * functWallCum / functNumCalls);  */
        metro_loop = metro_loop + 1;
        if ier == 2 && ignore_error2 != 0 {
            ier = 0;
        }
        //
        // For errors except limit reached, give warning message and restart
        //
        if ier != 0 && ier != 3 {
            if ier == 1 {
                let _ = stdout.write_all(b"\nMinimization error #1 - DG > 0\n");
            }
            if ier == 2 {
                let _ = stdout.write_all(b"\nMinimization error #2 - Linear search lost\n");
            }
            if ier == 4 {
                let _ =
                    stdout.write_all(b"\nMinimization error #4 - Matrix non-positive definite\n");
            }

            if metro_loop <= MAX_METRO_TRIALS {
                let _ = stdout.write_all(&c_format_bytes(
                    "\nRestarting with metro step factor of %.3f\n",
                    &[CArg::Dbl(metro_fac as f64)],
                ));
                copy_array(var, 1, num_var_search, varerr, 1);
            }
        }
    }
    if ier == 0 && metro_loop > 2 {
        let _ = stdout.write_all(b"Search succeeded with this step factor\n");
    }

    // Final call to FUNCT: set flag for fill points as it was passed in
    av.project_fill_points = get_fill_points;
    funct(eval_funct, av, mx, num_var_search, var, f_final, grad);
    if if_hush == 0 {
        let _ = stdout.write_all(&c_format_bytes(
            "\n Number of cycles : %5d                      Final   F :  %14.6f\n",
            &[
                CArg::Int(*kount as i64),
                CArg::Dbl((*f_final * rms_scale).sqrt() as f64),
            ],
        ));
    }
    let _ = stdout.flush();
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Error returns:
    if ier != 0 {
        if ier != 3 {
            error_exit::<false>("Search failed even after varying step factor", if_local);
        } else {
            error_exit::<false>("Minimization error #3 - Iteration limit exceeded", 1);
        }
        *metro_error += 1;
    }
}
