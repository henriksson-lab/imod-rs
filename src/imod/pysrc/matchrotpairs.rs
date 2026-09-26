//! Translation of `IMOD/pysrc/matchrotpairs`.
//!
//! A Python command script: its one function is
//! [`parabolic_fit_position`], and its top level is [`matchrotpairs`],
//! translated statement by statement.  The search itself is
//! `tiltmatch.searchPairs` ([`super::tiltmatch::search_pairs`]).
//!
//! The source's final `try` catches only `KeyboardInterrupt`; the
//! translation records SIGINT (see [`super::tiltmatch`]) and takes that arm
//! after the one `runcmd` in it.  Values are Python ints (`i64`) and floats
//! (`f64`).

use super::imodpy::{
    ImodpyError, add_imod_bin_ignore_sighup, get_mrc_size, make_backup_file, os_path_splitext,
    pass_on_key_interrupt, print_pid, prnstr, read_text_file, run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_in_out_file, pip_get_integer, pip_get_string,
    pip_get_two_integers, pip_read_or_parse_options,
};
use super::tiltmatch::{
    KEY_INTERRUPT, clean_exit_error, cleanup, get_temp_components, get_temp_names, key_interrupt,
    py_fixed, search_pairs,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Matches `parabolicFitPosition` (`matchrotpairs:13`).
pub fn parabolic_fit_position(y1: f64, y2: f64, y3: f64) -> f64 {
    let mut cx = 0.;
    let denom = 2. * (y1 + y3 - 2. * y2);
    if denom.abs() > (1.0e-2 * (y1 - y3)).abs() {
        cx = (y1 - y3) / denom;
    }
    // `max(-0.5, min(0.5, cx))`: Python's `min`/`max` keep the first
    // argument unless the second compares strictly less/greater
    let inner = if cx < 0.5 { cx } else { 0.5 };
    if inner > -0.5 { inner } else { -0.5 }
}

/// The script's top level (`matchrotpairs:22-187`).  Returns the status of
/// its final `sys.exit(0)`; error paths exit the process as `exitError`
/// does.
pub fn matchrotpairs(arguments: &[OsString]) -> i32 {
    let progname = "matchrotpairs";
    let prefix = format!("ERROR: {progname} - ");
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

    // An exception the source does not catch
    let uncaught = |message: &str| -> ! {
        let _ = std::io::stdout().flush();
        eprintln!("Traceback (most recent call last):");
        eprintln!("{message}");
        std::process::exit(1)
    };

    // Fallbacks from ../manpages/autodoc2man 3 1 matchrotpairs
    let options: Vec<String> = [
        "ia:AImageFile:FN:",
        "ib:BImageFile:FN:",
        "output:OutputFile:FN:",
        "za:AStartingEndingViews:IP:",
        "zb:BStartingEndingViews:IP:",
        "swap:SwapAandB:B:",
        "a:AngleOfRotation:I:",
        "mirror:MirrorXaxis:I:",
        "d:DistortionFile:FN:",
        "b:ImagesAreBinned:I:",
        "m:RunMidas:B:",
        "scan:ScanRotationMaxAndStep:FP:",
        "nearest:NearestNeighbor:B:",
        "x:WriteAllTransforms:B:",
        "t:LeaveTempFiles:B:",
        ":PID:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 0);
    pass_on_key_interrupt(true);
    // Python raises KeyboardInterrupt on SIGINT; `runcmd` passes it on
    unsafe {
        libc::signal(
            libc::SIGINT,
            key_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    // Set names of temp files
    let tmp_minxf = get_temp_names(progname);
    let (tmp_root, _tmp_dir, pid) = get_temp_components();

    let _tmp_stack = format!("{tmp_root}stack{pid}");
    let _tmp_twoxf = format!("{tmp_root}twoxf{pid}");
    let _tmp_xfmod = format!("{tmp_root}xfmod{pid}");
    let _tmp_midxf = format!("{tmp_root}midxf{pid}");

    let mut image_a = pip_get_in_out_file("AImageFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    let mut image_b = pip_get_in_out_file("BImageFile", 1)
        .ok()
        .flatten()
        .unwrap_or_default();
    let mut aa = "A";
    let mut bb = "B";
    let mut aview_opt = "AStartingEndingViews";
    let mut bview_opt = "BStartingEndingViews";
    let if_bto_a = pip_get_boolean("SwapAandB", 0).unwrap_or(0);

    // Swap files, letters, view range if doing B to A
    if if_bto_a != 0 {
        aa = "B";
        bb = "A";
        std::mem::swap(&mut image_a, &mut image_b);
        aview_opt = "BStartingEndingViews";
        bview_opt = "AStartingEndingViews";
    }

    let out_file = pip_get_in_out_file("OutputFile", 2)
        .ok()
        .flatten()
        .unwrap_or_default();
    if image_a.is_empty() || image_b.is_empty() || out_file.is_empty() {
        exit_error("You must enter two input files and an output file");
    }

    // Make sure image files exist
    for imfile in [&image_a, &image_b] {
        if !Path::new(imfile).exists() {
            exit_error(&format!("Image file {imfile} does not exist"));
        }
    }

    // Get image sizes
    let size = |file: &str| -> (i64, i64, i64) {
        match get_mrc_size(file) {
            Ok((x, y, z)) => (x as i64, y as i64, z as i64),
            Err(ImodpyError { arguments }) => {
                uncaught(&format!("imodpy.ImodpyError: {}", arguments.join("\n")))
            }
        }
    };
    let (nxa, nya, nza) = size(&image_a);
    let (nxb, nyb, nzb) = size(&image_b);

    // Get starting and ending section numbers
    let (asec_start, asec_end) = pip_get_two_integers(aview_opt, (1, nza as i32))
        .map(|(a, b)| (a as i64, b as i64))
        .unwrap_or((1, nza));
    if asec_start < 1 || asec_end > nza || asec_start > asec_end {
        exit_error(&format!(
            "Starting and ending views from {aa} are out of range or out of order"
        ));
    }

    let (bsec_start, bsec_end) = pip_get_two_integers(bview_opt, (1, nzb as i32))
        .map(|(a, b)| (a as i64, b as i64))
        .unwrap_or((1, nzb));
    if bsec_start < 1 || bsec_end > nzb || bsec_start > bsec_end {
        exit_error(&format!(
            "Starting and ending views from {bb} are out of range or out of order"
        ));
    }

    // Convert to center views and number of views
    let zero_a = (asec_start + asec_end - 1).div_euclid(2);
    let zero_b = (bsec_start + bsec_end - 1).div_euclid(2);
    let nviews_a = asec_end + 1 - asec_start;
    let nviews_b = bsec_end + 1 - bsec_start;

    // get distortion, nearest neighbor, all transform options
    let mut distort = String::new();
    let mut bilinear: i64 = 1;
    let distort_file = pip_get_string("DistortionFile", "").unwrap_or_default();
    if !distort_file.is_empty() {
        let image_binned = pip_get_integer("ImagesAreBinned", -1).unwrap_or(-1);
        distort = format!("-dist \"{distort_file}\"");
        if image_binned > 0 {
            distort += &format!(" -image {image_binned}");
        }
    }

    let nearest = pip_get_boolean("NearestNeighbor", 0).unwrap_or(0);
    if nearest != 0 {
        bilinear = 0;
    }

    let all_xf_out = pip_get_boolean("WriteAllTransforms", 0).unwrap_or(0);
    let (out_root, _ext) = os_path_splitext(&out_file);

    // Run the search
    let (asec_best, bsec_best, diff_list, all_xf_list) = search_pairs(
        progname, zero_a, zero_b, nviews_a, nviews_b, &image_a, &image_b, nxa, nxb, nya, nyb, aa,
        bb, "", 0, 0, &distort, bilinear, all_xf_out, 1,
    );

    // `try: ... except KeyboardInterrupt: pass`
    'body: {
        prnstr(
            &format!(
                "Views in best pair: {aa} {}  {bb} {}",
                asec_best + 1,
                bsec_best + 1
            ),
            "\n",
            false,
        );

        // Get interpolated position or report that search is at end of range
        let ind_a = asec_best - (asec_start - 1);
        let ind_b = bsec_best - (bsec_start - 1);
        if ind_a > 0 && ind_a < nviews_a - 1 && ind_b > 0 && ind_b < nviews_b - 1 {
            // Python list indexing: a negative index counts from the end, and
            // one past the end is an uncaught IndexError
            let at = |row: i64, col: i64| -> f64 {
                let fix = |index: i64, len: usize| -> usize {
                    let index = if index < 0 { index + len as i64 } else { index };
                    if index < 0 || index as usize >= len {
                        uncaught("IndexError: list index out of range");
                    }
                    index as usize
                };
                let row = &diff_list[fix(row, diff_list.len())];
                row[fix(col, row.len())]
            };
            let x1 = at(ind_b - 1, ind_a);
            let x3 = at(ind_b + 1, ind_a);
            let y1 = at(ind_b, ind_a - 1);
            let y3 = at(ind_b, ind_a + 1);
            if x1 > 0. && x3 > 0. && y1 > 0. && y3 > 0. {
                let interp_a = parabolic_fit_position(-x1, -at(ind_b, ind_a), -x3);
                let interp_b = parabolic_fit_position(-y1, -at(ind_b, ind_a), -y3);
                prnstr(
                    &format!(
                        "Interpolated view numbers:  {}  {}",
                        py_fixed((asec_best + 1) as f64 + interp_a, 0, 1),
                        py_fixed((bsec_best + 1) as f64 + interp_b, 0, 1)
                    ),
                    "\n",
                    false,
                );
            }
        } else {
            prnstr("Best pair is at end of search range", "\n", false);
        }

        // Produce a standard transform file with this transform in second line
        let best_stack = format!("{out_root}.stack");
        let mut minxf: Vec<String> = Vec::new();
        make_backup_file(&out_file);
        if Path::new(&tmp_minxf).exists() {
            minxf = read_text_file(&tmp_minxf, None, false, None).unwrap_or_default();
        }
        if minxf.is_empty() {
            cleanup();
            exit_error("No alignment was computed, cannot continue");
        }
        let _ = write_text_file(
            &out_file,
            &["1 0 0 1 0 0".to_owned(), minxf[0].clone()],
            false,
        );

        // Stack the two best sections unless nearest neighbor
        if nearest == 0 {
            let result = run_cmd(
                &format!(
                    "newstack -sec {bsec_best} -sec {asec_best} -float 2 -size {},{} -use 0,1 -float 2 {distort} {image_b} {image_a} \"{best_stack}\"",
                    nxa.max(nxb),
                    nya.max(nyb)
                ),
                None,
                None,
                None,
                &[],
            );
            if KEY_INTERRUPT.load(Ordering::SeqCst) {
                break 'body;
            }
            if result.is_err() {
                clean_exit_error("Stacking two best views");
            }
        }

        // Output transform files
        if all_xf_out != 0 {
            for view in 0..nviews_b {
                let all_name = format!("{out_root}-{}.xf", view + 1);
                let _ = write_text_file(&all_name, &all_xf_list[view as usize], false);
            }
        }
    }

    cleanup();
    0
}
