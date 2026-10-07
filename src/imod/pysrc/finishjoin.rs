//! Translation of `IMOD/pysrc/finishjoin`: completes the joining of serial
//! tomograms.
//!
//! The script's top level is [`finishjoin`] and its one function is
//! [`cleanup`].  `xftoxg`, `xfproduct`, `maxjoinsize` and `newstack` are our
//! own programs and run in process through `imodpy::run_cmd`.  The
//! `chmod u+rw` the script runs only on Windows is translated (`cfg!(windows)`).

use super::imodpy::{
    add_imod_bin_ignore_sighup, dataset_filename, exit_from_imod_error, fmtstr, get_mrc_size,
    get_type_ext_allowing_extra, glob_glob, make_backup_file, print_pid, prnstr, py_float, py_int,
    py_str_float, read_text_file, run_cmd, set_output_format_if_needed, set_root_and_extension,
    write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_non_option_arg, pip_get_string, pip_get_two_integers,
    pip_number_of_entries, pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// `def cleanup()` (`finishjoin:16`): cleanup temp files at end.
fn cleanup(joinroot: &str) {
    for f in glob_glob(&format!("{joinroot}.tmp?*")) {
        if std::fs::remove_file(&f).is_err() {
            break;
        }
    }
}

/// The script's top level (`finishjoin:25-366`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn finishjoin(arguments: &[OsString]) -> i32 {
    let progname = "finishjoin";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Initializations
    let mut newmode = String::new();
    let mut newsize: Option<[i64; 2]> = None;
    let mut trial = "";
    let mut fitopt = "-nfit 0";
    let mut refarg = String::new();
    let mut binning: i32 = 1;
    let mut name_style: i64 = 0;

    // Fallbacks from ../manpages/autodoc2man 3 1 finishjoin
    let options: Vec<String> = [
        "name:RootName:CH:",
        "use:UseSliceRange:IPM:",
        "ref:ReferenceTomogram:I:",
        "size:SizeInXandY:IP:",
        "offset:OffsetInXandY:IP:",
        "angle:AngleRange:F:",
        "trial:TrialInterval:I:",
        "binning:BinningForTrial:I:",
        "maxsize:MaximumSizeOnly:B:",
        "local:LocalFits:B:",
        "xform:TransformFile:FN:",
        "gaps:FillGaps:B:",
        "no:NoImage:B:",
        ":PID:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 3, 0, 0);

    let do_pid = pip_get_boolean("PID", 0).unwrap_or(0);
    print_pid(do_pid != 0);

    let joinroot = pip_get_in_out_file("RootName", 0)
        .ok()
        .flatten()
        .unwrap_or_default();

    let infoname = format!("{joinroot}.info");
    if !Path::new(&infoname).exists() {
        exit_error(&format!("Join Info file {infoname} not found"));
    }

    // Read in info file and get the number of tomos, version, etc from header line
    // 2nd version info file has default size in it
    // The fifth entry was added as a version number at version 3
    let infolines =
        read_text_file(&infoname, Some("join Info file"), false, None).unwrap_or_default();
    let mut ntomo: i64 = 0;
    let mut ifsquoze: i64 = 0;
    let mut version: i64 = 1;
    let parsed = (|| -> Option<()> {
        let info1: Vec<&str> = infolines.first()?.trim().split_whitespace().collect();
        ntomo = py_int(info1.first()?)?;
        ifsquoze = py_int(info1.get(1)?)?;
        version = 1;
        if info1.len() > 2 {
            version = 2;
        }
        if info1.len() > 4 {
            version = py_int(info1[4])?;
        }
        if version > 3 {
            newmode = (*info1.get(5)?).to_owned();
        }
        if version > 4 {
            name_style = py_int(info1.get(6)?)?;
        }
        if version > 1 {
            newsize = Some([py_int(info1.get(2)?)?, py_int(info1.get(3)?)?]);
        }
        Some(())
    })();
    if parsed.is_none() {
        exit_error("Getting information from first line of info file");
    }

    // Make sure the number of slice entries is right
    let num_by_opt = pip_number_of_entries("UseSliceRange").unwrap_or(0) as i64;
    let num_non_opts = num_non_opts as i64;
    if (num_by_opt != ntomo && num_non_opts != ntomo + 1) || num_by_opt + num_non_opts != ntomo + 1
    {
        if num_by_opt + num_non_opts == ntomo + 1 {
            exit_error(
                "You must enter slice ranges either with -slice or as non-option arguments, not both ways",
            );
        }
        if num_by_opt == 0 && num_non_opts == ntomo {
            exit_error(
                "You must enter the rootname as a non-option argument if you enter slice ranges that way",
            );
        }
        exit_error("Number of slice ranges does not match number of tomograms");
    }

    // Get the rest of the options
    let nref = pip_get_integer("ReferenceTomogram", 0).unwrap_or(0);
    if pip_get_err_no() == 0 {
        if nref <= 0 || nref as i64 > ntomo {
            exit_error("Reference tomogram number out of range");
        }
        refarg = format!("-ref {nref}");
    }

    // Size entry overrides the info file option
    let (xsize, ysize) = pip_get_two_integers("SizeInXandY", (0, 0)).unwrap_or((0, 0));
    if pip_get_err_no() == 0 {
        newsize = Some([xsize as i64, ysize as i64]);
    }

    let (xoffset, yoffset) = pip_get_two_integers("OffsetInXandY", (0, 0)).unwrap_or((0, 0));
    let angle_range = pip_get_float("AngleRange", 50.).unwrap_or(50.);
    let trial_interval = pip_get_integer("TrialInterval", 0).unwrap_or(0);
    if trial_interval > 0 {
        trial = "_trial";
        binning = pip_get_integer("BinningForTrial", 1).unwrap_or(1);
        if !(1..=16).contains(&binning) {
            exit_error("Binning out of range of allowed values");
        }
    }

    let size_only = pip_get_boolean("MaximumSizeOnly", 0).unwrap_or(0);
    let local_fits = pip_get_boolean("LocalFits", 0).unwrap_or(0);
    if local_fits != 0 {
        fitopt = " ";
        if !refarg.is_empty() {
            exit_error("You cannot use local fitting together with a reference section");
        }
    }

    let xform_file = pip_get_string("TransformFile", "").unwrap_or_default();
    let gaps = pip_get_boolean("FillGaps", 0).unwrap_or(0);
    let no_image = pip_get_boolean("NoImage", 0).unwrap_or(0);

    // A missing line is an uncaught IndexError in the source
    let info_line = |index: usize| -> &str {
        match infolines.get(index) {
            Some(line) => line,
            None => super::pip::python_uncaught("IndexError: list index out of range"),
        }
    };

    // Build up tomo list and inversion and scale lists
    // Early versions just had list of filenames then list of inversion flags
    let mut matlist: Vec<String> = Vec::new();
    let invertlist: Vec<String>;
    let scalelist: Vec<String>;
    let linesave;
    if version < 3 {
        matlist = info_line(1).split_whitespace().map(str::to_owned).collect();
        invertlist = info_line(2).split_whitespace().map(str::to_owned).collect();
        scalelist = Vec::new();
        linesave = 0;

    // Later versions have lines of information for all tomos, then tomo file names
    // one per line to handle spaces
    } else {
        invertlist = info_line(1).split_whitespace().map(str::to_owned).collect();
        scalelist = info_line(2).split_whitespace().map(str::to_owned).collect();
        linesave = 3;
        if (scalelist.len() as i64) < ntomo {
            exit_error(&format!("{infoname} does not have enough scaling entries"));
        }
        for i in 0..ntomo.max(0) as usize {
            matlist.push(info_line(3 + i).trim().to_owned());
        }
    }

    // If trial mode with binning, adjust newsize (and not offset - 1/31/07)
    if binning > 1 {
        if let Some(size) = newsize.as_mut() {
            size[0] = size[0].div_euclid(binning as i64);
            size[1] = size[1].div_euclid(binning as i64);
        }
    }

    // Process the Z ranges
    let mut rangelist: Vec<String> = Vec::new();
    for itomo in 0..ntomo.max(0) as usize {
        let Some(matname) = matlist.get(itomo) else {
            super::pip::python_uncaught("IndexError: list index out of range")
        };

        // Get Z size of tomogram
        let nz: i64;
        if Path::new(matname).exists() {
            match get_mrc_size(matname) {
                Ok((_, _, z)) => nz = z as i64,
                Err(_) => exit_from_imod_error(progname),
            }
        } else {
            nz = 100;
            prnstr(
                &format!(
                    "WARNING: {matname}  NO LONGER EXISTS.  SETTING THICKNESS TO 100 FOR TESTING"
                ),
                "\n",
                false,
            );
        }

        // Get slice entries from either place
        let (mut zst, mut znd): (i64, i64);
        if num_by_opt != 0 {
            let (first, second) =
                pip_get_two_integers("UseSliceRange", (-1, -1)).unwrap_or((-1, -1));
            (zst, znd) = (first as i64, second as i64);
        } else {
            let slice_str = pip_get_non_option_arg(itomo as i32 + 1).unwrap_or_default();
            let slice_spl: Vec<&str> = slice_str.split([',', ' ']).collect();
            if slice_spl.len() != 2 {
                exit_error(&format!(
                    "Slice range {slice_str} does not , have two numbers"
                ));
            }
            match (py_int(slice_spl[0]), py_int(slice_spl[1])) {
                (Some(first), Some(second)) => (zst, znd) = (first, second),
                _ => exit_error(&format!(
                    "Converting slice range {slice_str} to two numbers"
                )),
            }
        }

        zst -= 1;
        znd -= 1;
        if gaps == 0 && (zst < 0 || zst >= nz || znd < 0 || znd >= nz) {
            exit_error(&format!(
                "The Z range for tomogram # {}  has coordinates out of range",
                itomo + 1
            ));
        }
        let invert = match invertlist.get(itomo).map(|text| py_int(text)) {
            Some(Some(value)) => value,
            Some(None) => super::pip::python_uncaught(&format!(
                "ValueError: invalid literal for int() with base 10: '{}'",
                invertlist[itomo]
            )),
            None => super::pip::python_uncaught("IndexError: list index out of range"),
        };
        if (zst < znd && invert != 0) || (zst >= znd && invert == 0) {
            (zst, znd) = (znd, zst);
        }

        // Set up z range as simple range or as list of slices at interval
        // DIFFERENCE: interval = 1 gives simple list
        let zrange;
        if trial_interval <= 1 {
            zrange = format!("{zst}-{znd}");
        } else {
            let mut range = zst.to_string();
            let mut idir: i64 = 1;
            if zst > znd {
                idir = -1;
            }
            let interval = trial_interval as i64;
            let mut iz = zst + idir * interval;
            // Fixed in translation (BUGS.md, `finishjoin`): the source never
            // sets `zlast` before this loop, so a range no longer than the
            // interval is a NameError on the first tomogram and compares
            // with the previous tomogram's value after that.
            let mut zlast = zst;
            while idir * (iz - znd) < 0 {
                range += &format!(",{iz}");
                zlast = iz;
                iz += idir * interval;
            }
            if zlast != znd {
                range += &format!(",{znd}");
            }
            zrange = range;
        }
        rangelist.push(zrange);
    }

    let tomoxf = format!("{joinroot}.tomoxf");
    make_backup_file(&tomoxf);

    // Read the transform file and make sure there is either 6 or 1 entry on first line
    let xflines = read_text_file(
        &format!("{joinroot}.xf"),
        Some("alignment transform file"),
        false,
        None,
    )
    .unwrap_or_default();
    if xflines.is_empty() {
        exit_error(&format!(
            "The alignment transform file {joinroot}.xf is empty"
        ));
    }

    let lenfirst = xflines[0].split_whitespace().count();
    let warping = lenfirst == 1;
    if lenfirst == 6 || warping {
        // If so and there is either 1 entry or the right number of lines, copy file
        if xflines.len() as i64 == ntomo || warping {
            if std::fs::copy(format!("{joinroot}.xf"), &tomoxf).is_err() {
                exit_error(&format!("Copying {joinroot}.xf to {tomoxf}"));
            }

            // Persistent broken pipes even in RHEL5/python 2.4, so just do in
            // Windows (`finishjoin:237-246`)
            if cfg!(windows)
                && super::imodpy::run_cmd(
                    &format!("chmod u+rw {tomoxf}"),
                    None,
                    None,
                    Some("stdout"),
                    &[],
                )
                .is_err()
            {
                let _ = crate::imod::libcfshr::b3dutil::py_chmod(&tomoxf, 0o644);
            }

        // Otherwise, need to start with a unit transform line and
        } else {
            let unit = "1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000";
            let mut outlines = vec![unit.to_owned()];
            for l in &xflines {
                let lsp: Vec<&str> = l.split_whitespace().collect();
                let value = |index: usize| -> f64 {
                    match lsp.get(index).and_then(|text| py_float(text)) {
                        Some(value) => value,
                        None => exit_error(&format!("Interpreting lines of {tomoxf}")),
                    }
                };
                if value(0) != 1.
                    || value(3) != 1.
                    || value(1) != 0.
                    || value(2) != 0.
                    || value(4) != 0.
                    || value(5) != 0.
                {
                    outlines.push(l.clone());
                }
            }

            if outlines.len() as i64 != ntomo {
                exit_error(&format!(
                    "There are {} lines in {joinroot}.xf with non-unit transform; there should be {} for joining {ntomo} tomograms",
                    outlines.len() - 1,
                    ntomo - 1
                ));
            }
        }
    } else {
        exit_error(&format!(
            "{joinroot}.xf does not appear to be a linear or warping transform file"
        ));
    }

    // Take care of extension style and output format
    let type_ext = get_type_ext_allowing_extra(name_style as i32).unwrap_or_default();
    set_root_and_extension(&joinroot, &type_ext);
    set_output_format_if_needed(&type_ext, true);

    // Now run some programs
    let tomoxg = format!("{joinroot}.tomoxg");
    enum Failure {
        Imodpy,
        Os,
    }
    let result: Result<(), Failure> = (|| {
        // Get the G transforms
        run_cmd(
            &fmtstr(
                "xftoxg -range {} {} {} {} {}",
                &[
                    py_str_float(angle_range),
                    refarg.clone(),
                    fitopt.to_owned(),
                    tomoxf.clone(),
                    tomoxg.clone(),
                ],
            ),
            None,
            None,
            None,
            &[],
        )
        .map_err(|_| Failure::Imodpy)?;

        // Adjust for squeezing
        let tmpxg = format!("{joinroot}.tmpxg");
        if ifsquoze != 0 && !warping {
            let _ = std::fs::remove_file(&tmpxg);
            std::fs::rename(&tomoxg, &tmpxg).map_err(|_| Failure::Os)?;
            run_cmd(
                &fmtstr(
                    "xfproduct {0}.sqzxf {0}.tmpxg {0}.tmpsqz",
                    &[joinroot.clone()],
                ),
                None,
                None,
                None,
                &[],
            )
            .map_err(|_| Failure::Imodpy)?;
            run_cmd(
                &fmtstr(
                    "xfproduct {0}.tmpsqz {0}.xpndxf {0}.tomoxg",
                    &[joinroot.clone()],
                ),
                None,
                None,
                None,
                &[],
            )
            .map_err(|_| Failure::Imodpy)?;
        }

        //  If a refinement file was entered, form product with that
        if !xform_file.is_empty() {
            let _ = std::fs::remove_file(&tmpxg);
            std::fs::rename(&tomoxg, &tmpxg).map_err(|_| Failure::Os)?;
            let mut scale_opt = String::new();

            // If warping and there was squeezing, need to scale down the refine transform
            // when multiplying by warp transforms
            if ifsquoze != 0 && warping {
                let sqzlines = read_text_file(
                    &format!("{joinroot}.sqzxf"),
                    Some("file with squeezing transform"),
                    false,
                    None,
                )
                .unwrap_or_default();
                if sqzlines.is_empty() {
                    exit_error("No lines in .sqzxf file");
                }
                let sqztext = match sqzlines[0].split_whitespace().next() {
                    Some(text) => text.to_owned(),
                    None => super::pip::python_uncaught("IndexError: list index out of range"),
                };
                scale_opt = format!("-scale 1.,{}", sqztext.trim());
            }

            run_cmd(
                &fmtstr(
                    "xfproduct {0} {1}.tmpxg {2} {1}.tomoxg",
                    &[scale_opt, joinroot.clone(), xform_file.clone()],
                ),
                None,
                None,
                None,
                &[],
            )
            .map_err(|_| Failure::Imodpy)?;

        // But if no refinement is done, copy a warping tomoxg for use by joinwarp2model
        } else if warping {
            let warpxg = format!("{joinroot}.warpxg");
            make_backup_file(&warpxg);
            if std::fs::copy(&tomoxg, &warpxg).is_err() {
                exit_error(&format!("Copying {tomoxg} to {warpxg}"));
            }
        }

        // Now if max size is wanted, call the program
        if size_only != 0 {
            if linesave == 0 {
                exit_error("INFO FILE IS TOO OLD TO GET THE MAXIMUM SIZE FROM");
            }
            run_cmd(
                &format!("maxjoinsize {ntomo} {linesave} {joinroot}"),
                None,
                Some("stdout"),
                None,
                &[],
            )
            .map_err(|_| Failure::Imodpy)?;
            cleanup(&joinroot);
            let _ = std::io::stdout().flush();
            crate::imod::libcfshr::b3dutil::exit(0);
        }

        // Compose the newstack command
        let mut newstin: Vec<String> = Vec::new();
        for i in 0..ntomo.max(0) as usize {
            newstin.push(format!("InputFile {}", matlist[i]));
            newstin.push(format!("SectionsToRead {}", rangelist[i]));
            if !scalelist.is_empty() {
                newstin.push(format!("MultiplyAndAdd {}", scalelist[i]));
            }
        }

        if let Some(size) = newsize {
            newstin.push(format!("SizeToOutputInXandY {},{}", size[0], size[1]));
        }
        if !newmode.is_empty() {
            newstin.push(format!("ModeToOutput {newmode}"));
        }
        if gaps != 0 {
            newstin.push("BlankOutput".to_owned());
        }
        newstin.extend([
            "OneTransformPerFile".to_owned(),
            format!("BinByFactor {binning}"),
            format!(
                "OutputFile {}",
                dataset_filename(&format!("{trial}.join"), None, None)
            ),
            format!("OffsetsInXandY {xoffset},{yoffset}"),
            format!("TransformFile {tomoxg}"),
        ]);
        if no_image != 0 {
            let _ = write_text_file("finishjoinNewst.input", &newstin, false);
        } else {
            run_cmd(
                "newstack -StandardInput",
                Some(&newstin),
                Some("stdout"),
                None,
                &[],
            )
            .map_err(|_| Failure::Imodpy)?;
            prnstr(
                "Truncations are a normal result of the scaling for density matching.",
                "\n",
                false,
            );
        }
        Ok(())
    })();
    match result {
        Ok(()) => {}
        Err(Failure::Imodpy) => {
            cleanup(&joinroot);
            exit_from_imod_error(progname);
        }
        Err(Failure::Os) => {
            cleanup(&joinroot);
            exit_error(&format!("Renaming {tomoxg} to {joinroot}.tmpxg"));
        }
    }

    cleanup(&joinroot);
    done(0)
}
