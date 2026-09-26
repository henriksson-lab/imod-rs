//! Translation of `IMOD/pysrc/autopatchfit`.
//!
//! A Python command script: its one function is [`failure_log`], and its top
//! level is [`autopatchfit`], translated statement by statement.  The
//! patchcorr and matchorwarp command files, which the source runs with
//! `vmstopy -x -q`, run through the in-process runner
//! ([`crate::imod::comrun::run_com_as_command`]).  Floats are Python floats (doubles); `round()` rounds half to even.

use super::imodpy::{
    OptionValue, add_imod_bin_ignore_sighup, auto_patch_number, default_com_extension,
    find_root_axis_and_extensions, make_backup_file, option_value, patch_size_from_entry, prnstr,
    read_text_file, run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_integer, pip_get_string, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_modify};
use std::ffi::OsString;
use std::io::Write as _;

/// `PATCHXY` and `PATCHZ` (`imodpy.py:1590-1591`).
const PATCHXY: [i64; 4] = [64, 80, 100, 120];
const PATCHZ: [i64; 4] = [32, 40, 50, 60];

/// Matches `failureLog` (`autopatchfit:12`).
pub fn failure_log(com_name: &str, log_lines: &[String]) -> ! {
    for line in log_lines {
        if line.contains("ERROR:") && !line.contains("-StandardInput: exited") {
            prnstr(line, "\n", false);
        }
    }
    exit_error(&format!("{com_name} failed with unrecoverable error"))
}

/// The script's top level (`autopatchfit:19-307`).  Every path ends the
/// process through `sys.exit` or `exitError`; the returned status is the
/// source's `sys.exit(0)` on success.
pub fn autopatchfit(arguments: &[OsString]) -> i32 {
    let progname = "autopatchfit";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 autopatchfit
    let options: Vec<String> = [
        "final:FinalPatchTypeOrXYZ:CH:",
        "extra:ExtraResidualTargets:CH:",
        "high:HighDensityFinalTrial:I:",
        "trial:TrialMode:B:",
        "skip:SkipFirstPatchcorr:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    let (mut com_ext, _dual_num, _setroot, _type_ext, _stack_ext) =
        find_root_axis_and_extensions(-1, None);
    if com_ext.is_empty() {
        com_ext = default_com_extension();
    }
    let patchcom = format!("patchcorr.{com_ext}");
    let mowcom = format!("matchorwarp.{com_ext}");

    // Get options
    let final_size = pip_get_string("FinalPatchTypeOrXYZ", "L").unwrap_or_default();
    let (nx_final, ny_final, nz_final, err) = patch_size_from_entry(&final_size);
    let (nx_final, ny_final, nz_final) = (nx_final as i64, ny_final as i64, nz_final as i64);
    if err != 0 {
        exit_error(
            &("You must enter one of S, M, L, or E or sizes in X,Y,Z for ".to_owned()
                + "the -final option"),
        );
    }

    let extra_target = pip_get_string("ExtraResidualTargets", "").unwrap_or_default();
    let mut high_density = pip_get_integer("HighDensityFinalTrial", 0).unwrap_or(0);
    let trial_mode = pip_get_boolean("TrialMode", 0).unwrap_or(0);
    let skip_first = pip_get_boolean("SkipFirstPatchcorr", 0).unwrap_or(0);

    // Get patchcorr and determine high density setting unless entered
    let patch_lines = read_text_file(&patchcom, None, false, None).unwrap_or_default();

    // Determine if there is an initial shift entry, note that if the user leaves X and Y
    // fields blank in Etomo, this is a ",value/"
    if high_density == 0 {
        high_density = -1;
        let initial_shift = option_value(&patch_lines, "InitialShiftXYZ", 0, false, 0, None, None);
        if matches!(&initial_shift, Some(OptionValue::String(value)) if !value.is_empty()) {
            high_density = 1;
        }
    }

    // First manage the trial flag in matchorwarp and pull off extra targets
    // Identify if standard input form of file
    let mut mow_lines = read_text_file(&mowcom, None, false, None).unwrap_or_default();
    let mut std_input_line: i64 = -1;
    let mut trial_line: i64 = -1;
    let mut warp_line: i64 = -1;
    let mut warp_opt = "-warplimit";
    for line_ind in 0..mow_lines.len() {
        let line = mow_lines[line_ind].trim().to_owned();
        if line.contains("matchorwarp") && line.contains("-StandardI") {
            std_input_line = line_ind as i64;
            warp_opt = "WarpLimits";
        }
        if std_input_line >= 0 && line.starts_with("WarpLimits") {
            if !extra_target.is_empty() {
                mow_lines[line_ind] = mow_lines[line_ind].replace(&format!(",{extra_target}"), "");
            }
            warp_line = line_ind as i64;
        }
        if std_input_line < 0 && !line.starts_with('#') && line.contains("-warplimit") {
            warp_line = line_ind as i64;
            if !extra_target.is_empty() {
                mow_lines[line_ind] = mow_lines[line_ind].replace(&format!(",{extra_target}"), "");
            }
            if trial_mode != 0 {
                mow_lines[line_ind] =
                    mow_lines[line_ind].replace("-warplimit", "-trial -warplimit");
                break;
            }
        }
        if std_input_line >= 0 && line.starts_with("Trial") {
            trial_line = line_ind as i64;
        }
        if std_input_line < 0
            && trial_mode == 0
            && !line.starts_with('#')
            && line.contains("-trial")
        {
            mow_lines[line_ind] = mow_lines[line_ind].replace("-trial", "");
        }
    }

    if warp_line < 0 {
        exit_error(&format!(
            "Cannot find existing residual targets in {mowcom}"
        ));
    }

    // Pull out or put in the Trial line, adjusting warpLine as needed
    if std_input_line >= 0 {
        if trial_mode != 0 && trial_line < 0 {
            mow_lines.insert((std_input_line + 1) as usize, "TrialMode 1".to_owned());
            warp_line += 1;
        } else if trial_mode == 0 && trial_line >= 0 {
            mow_lines.remove(trial_line as usize);
            if trial_line < warp_line {
                warp_line -= 1;
            }
        }
    }

    make_backup_file(&mowcom);
    let _ = write_text_file(&mowcom, &mow_lines, false);

    // Make sure matchorwarp can be modified for extra target before starting
    if !extra_target.is_empty() {
        let line = mow_lines[warp_line as usize].clone();
        let replaced = line.trim().replace('\t', " ");
        let lsplit = replaced.split_whitespace().collect::<Vec<_>>();
        for ind in 0..lsplit.len().saturating_sub(1) {
            if lsplit[ind] == warp_opt {
                let cur_target = lsplit[ind + 1];
                if !cur_target.ends_with(extra_target.as_str()) {
                    let new_target = format!("{cur_target},{extra_target}");
                    mow_lines[warp_line as usize] = line.replace(cur_target, &new_target);
                    break;
                }
            }
        }
    }

    // Figure out the maximum number of runs in advance
    let ints = |option: &str, num_val: usize| -> Option<Vec<i64>> {
        match option_value(&patch_lines, option, 1, false, num_val, None, None) {
            Some(OptionValue::Integers(values)) if !values.is_empty() => {
                Some(values.iter().map(|value| *value as i64).collect())
            }
            _ => None,
        }
    };
    let nxyz_patch_cur = ints("PatchSizeXYZ", 3);
    let num_xyz_orig = ints("NumberOfPatchesXYZ", 3);
    let x_min_max = ints("XMinAndMax", 2);
    let z_min_max = ints("YMinAndMax", 2);
    let y_min_max = ints("ZMinAndMax", 2);
    let (
        Some(nxyz_patch_cur),
        Some(num_xyz_orig),
        Some(x_min_max),
        Some(y_min_max),
        Some(z_min_max),
    ) = (
        nxyz_patch_cur,
        num_xyz_orig,
        x_min_max,
        y_min_max,
        z_min_max,
    )
    else {
        exit_error(&format!(
            "Cannot find one of patch size, number of patches, or X, Y or Z limits in {patchcom}"
        ));
    };

    let nx_curr = nxyz_patch_cur[0];
    let ny_curr = nxyz_patch_cur[2];
    let nz_curr = nxyz_patch_cur[1];
    if nx_final < nx_curr || ny_final < ny_curr || nz_final < nz_curr {
        exit_error("Final patch size cannot be smaller that current size in any dimension");
    }

    let mut max_trials: i64;
    let mut cur_ind: i64 = -1;
    let mut final_ind: i64 = -1;
    let mut num_steps: i64 = 0;
    let mut step_factor: f64 = 1.;
    if nx_final == nx_curr && ny_final == ny_curr && nz_final == nz_curr {
        max_trials = 1;
    } else {
        // Try to match up size with the stock ones
        for ind in 0..PATCHXY.len() {
            if nx_final == ny_final && nx_final == PATCHXY[ind] && nz_final == PATCHZ[ind] {
                final_ind = ind as i64;
            }
            if nx_curr == PATCHXY[ind] && ny_curr == PATCHXY[ind] && nz_curr == PATCHZ[ind] {
                cur_ind = ind as i64;
            }
        }

        if cur_ind >= 0 && final_ind >= 0 {
            max_trials = final_ind + 1 - cur_ind;
        } else {
            // If no size match, target steps of 1.25, round number of steps up so the steps
            // will be no bigger than ~1.3
            let max_factor = (nx_final as f64 / nx_curr as f64)
                .max(ny_final as f64 / ny_curr as f64)
                .max(nz_final as f64 / nz_curr as f64);
            num_steps = 1i64.max((max_factor.ln() / 1.25f64.ln() + 0.75) as i64);
            step_factor = (max_factor.ln() / num_steps as f64).exp();
            max_trials = num_steps + 1;
        }
    }

    if high_density > 0 {
        max_trials += 1;
    }

    // Start loop on trials
    let mut nx_new = nx_curr;
    let mut ny_new = ny_curr;
    let mut nz_new = nz_curr;
    let mut cum_factor: f64 = 1.;
    let z_range = z_min_max[1] + 1 - z_min_max[0];
    let nz_limit = if num_xyz_orig[1] == 1 {
        z_range
    } else {
        (z_range * 3).div_euclid(2)
    };

    for trial in 0..max_trials.max(0) {
        let final_trial = trial == max_trials - 1;

        // Modify to the next size or density after the first trial
        if trial != 0 {
            let density_ind;
            if !(high_density > 0 && final_trial) {
                density_ind = 0;
                if cur_ind >= 0 && final_ind >= 0 {
                    cur_ind += 1;
                    nx_new = PATCHXY[cur_ind as usize];
                    ny_new = nx_new;
                    nz_new = PATCHZ[cur_ind as usize];
                } else if trial < num_steps {
                    cum_factor *= step_factor;
                    nx_new = 2 * (nx_curr as f64 * cum_factor / 2.).round_ties_even() as i64;
                    ny_new = 2 * (ny_curr as f64 * cum_factor / 2.).round_ties_even() as i64;
                    nz_new = 2 * (nz_curr as f64 * cum_factor / 2.).round_ties_even() as i64;
                } else {
                    nx_new = nx_final;
                    ny_new = ny_final;
                    nz_new = nz_final;
                }

                nz_new = nz_new.min(nz_limit);
            } else {
                density_ind = 1;
            }

            let num_xnew = auto_patch_number(
                nx_new as i32,
                x_min_max[0] as i32,
                x_min_max[1] as i32,
                false,
                density_ind,
            );
            let num_ynew = auto_patch_number(
                ny_new as i32,
                y_min_max[0] as i32,
                y_min_max[1] as i32,
                false,
                density_ind,
            );
            let mut num_znew = auto_patch_number(
                nz_new as i32,
                z_min_max[0] as i32,
                z_min_max[1] as i32,
                true,
                density_ind,
            );
            if num_xyz_orig[1] == 1 {
                num_znew = 1;
            }
            let sedcom = vec![
                sed_modify("PatchSizeXYZ", &format!("{nx_new},{nz_new},{ny_new}"), '/'),
                sed_modify(
                    "NumberOfPatchesXYZ",
                    &format!("{num_xnew},{num_znew},{num_ynew}"),
                    '/',
                ),
            ];
            if trial == 1 {
                make_backup_file(&patchcom);
            }
            let _ = pysed(
                &sedcom,
                PysedSrc::Lines(&patch_lines),
                Some(&patchcom),
                false,
                '/',
                false,
            );
            prnstr(
                &format!(
                    "AUTOPATCHFIT - Changing to patch size {nx_new} {ny_new} {nz_new}, number {num_xnew} {num_ynew} {num_znew}"
                ),
                "\n",
                false,
            );
        } else {
            prnstr(
                &format!(
                    "AUTOPATCHFIT - Using initial patch size {nx_new} {ny_new} {nz_new}, number {} {} {}",
                    num_xyz_orig[0], num_xyz_orig[2], num_xyz_orig[1]
                ),
                "\n",
                false,
            );
        }

        // Modify matchorwarp if there are extra criteria on last round
        if final_trial && !extra_target.is_empty() {
            prnstr(
                &format!("AUTOPATCHFIT - Adding {extra_target} to warp residual limits"),
                "\n",
                false,
            );
            let _ = write_text_file(&mowcom, &mow_lines, false);
        }

        if trial != 0 || skip_first == 0 {
            prnstr(&format!("AUTOPATCHFIT - Running {patchcom}"), "\n", true);
            // `runcmd('vmstopy -x -q ...')`: the in-process runner now
            // (owner, 2026-09-26: no Python, no pipes)
            if crate::imod::comrun::run_com_as_command(&patchcom, "patchcorr.log").is_err() {
                let log_lines =
                    read_text_file("patchcorr.log", None, false, None).unwrap_or_default();
                failure_log(&patchcom, &log_lines);
            }
        }

        prnstr(&format!("AUTOPATCHFIT - Running {mowcom}"), "\n", true);
        make_backup_file("matchorwarp.log");
        match crate::imod::comrun::run_com_as_command(&mowcom, "matchorwarp.log") {
            Ok(_) => {
                // Success!
                let log_lines =
                    read_text_file("matchorwarp.log", None, false, None).unwrap_or_default();
                let mut refine_res = String::new();
                let mut warp_res = String::new();
                for line in &log_lines {
                    let upper = line.to_uppercase();
                    if upper.contains("FOUND A GOOD") {
                        if upper.contains("REFINEMATCH") && !refine_res.is_empty() {
                            prnstr(
                                &format!(
                                    "AUTOPATCHFIT - Refinematch found a good transformation, mean residual {refine_res}"
                                ),
                                "\n",
                                false,
                            );
                            break;
                        }
                        if upper.contains("FINDWARP") && !warp_res.is_empty() {
                            prnstr(
                                &format!(
                                    "AUTOPATCHFIT - Findwarp found a good warping, mean residual {warp_res}"
                                ),
                                "\n",
                                false,
                            );
                            break;
                        }
                    }
                    if line.contains("Mean residual") {
                        let replaced = line.replace(',', " ");
                        for token in replaced.split_whitespace() {
                            if token.contains('.') {
                                if token.parse::<f64>().is_ok() {
                                    if line.contains("has") {
                                        warp_res = token.to_owned();
                                    } else {
                                        refine_res = token.to_owned();
                                    }
                                }
                                break;
                            }
                        }
                    }
                }

                let _ = std::io::stdout().flush();
                std::process::exit(0);
            }
            Err(_) => {
                let log_lines =
                    read_text_file("matchorwarp.log", None, false, None).unwrap_or_default();
                let mut found = false;
                for line in &log_lines {
                    if line.to_uppercase().contains("FINDWARP - FAILED TO FIND") {
                        prnstr(&line.replace("ERROR: ", ""), "\n", false);
                        found = true;
                        break;
                    }
                }
                if !found {
                    // ELSE ON FOR
                    failure_log(&mowcom, &log_lines);
                }

                // Save the current patches unless it is the last round
                if !final_trial {
                    let patch_name = format!("patch_{nx_new}x{ny_new}x{nz_new}.out");
                    make_backup_file(&patch_name);
                    if std::fs::rename("patch.out", &patch_name).is_err() {
                        exit_error(&format!("Renaming patch.out to {patch_name}"));
                    }
                    prnstr(
                        &format!("AUTOPATCHFIT - Renamed patch.out as {patch_name}"),
                        "\n",
                        false,
                    );
                }
            }
        }
    }

    // End of loop with no success.  How to leave things?
    exit_error("Could not get patch correlations with an acceptable fit")
}
