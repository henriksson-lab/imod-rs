//! Translation of `IMOD/pysrc/splittilt`.
//!
//! A Python command script with no functions; its top level is
//! [`splittilt`], translated statement by statement.  `header` (through
//! `getmrcsize`) is this crate's own program and runs in process.
//!
//! Python semantics carried explicitly: `//` and `%` floor (the operands
//! here are non-negative sizes, so `div_euclid`/`rem_euclid` are used where
//! the divisor is positive), `round()` rounds half to even, `int()` of a
//! float truncates toward zero, and every float is a double.

use super::imodpy;
use super::imodpy::{
    OptionValue, add_imod_bin_ignore_sighup, clean_chunk_files, complete_and_check_com_file,
    dataset_filename, exit_from_imod_error, find_root_axis_and_extensions, get_mrc_size,
    get_naming_style, option_value, os_path_splitext, parallel_boundary_size, prnstr,
    py_int_floordiv, py_int_of_float, py_round, py_true_div, read_text_file,
    set_root_and_extension, standard_type_extensions, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_string, pip_get_two_integers, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_modify};
use regex::Regex;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`splittilt:1-437`).  It ends without a
/// `sys.exit`, so its status is 0; error paths exit the process as
/// `exitError` does.
pub fn splittilt(arguments: &[OsString]) -> i32 {
    let progname = "splittilt";
    let prefix = format!("ERROR: {progname} - ");
    let mut penalty: f64 = 1.33;
    let maxextrapct = 102;
    let mut numproc = 8;
    let mut minslices = 50;
    let minratio = 2;
    let targetratio = 5;
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

    // Initializations (defaults are above or in Pip calls)
    let mut oldstyle = "#".to_owned();
    let boundext = "rbound";
    let vs_bound_ext = "vsbound";
    let mut boundpixels = parallel_boundary_size(2048);

    // Fallbacks from ../manpages/autodoc2man 3 1 splittilt
    let options: Vec<String> = [
        ":CommandFile:FN:",
        "outroot:RootNameOfOutput:CH:",
        "naming:NamingStyle:I:",
        "n:ProcessorNumber:I:",
        "s:SliceMinimum:I:",
        "t:TargetChunks:I:",
        "m:ChunkMinimum:I:",
        "p:OldStyleXtiltPenalty:F:",
        "v:VerticalSlices:B:",
        "c:SeparateChunks:B:",
        "b:BoundaryPixels:I:",
        "i:InitialComNumber:I:",
        "o:OpenForMoreComs:B:",
        "unique:UniqueInfoFile:B:",
        "d:DimensionsOfStack:IP:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 0);

    // Get the com file name, derive a root name and new com file name, check exists
    let comfile = pip_get_in_out_file("CommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    let (comfile, rootname) = complete_and_check_com_file(&comfile);

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
    let rootname = pip_get_string("RootNameOfOutput", &rootname).unwrap_or(rootname);
    numproc = pip_get_integer("ProcessorNumber", numproc).unwrap_or(numproc);
    minslices = pip_get_integer("SliceMinimum", minslices).unwrap_or(minslices);
    penalty = pip_get_float("OldStyleXtiltPenalty", penalty).unwrap_or(penalty);
    let mut targetslabs = pip_get_integer("TargetChunks", 0).unwrap_or(0);
    let mut minslabs = pip_get_integer("ChunkMinimum", 0).unwrap_or(0);
    let vertical = pip_get_boolean("VerticalSlices", 0).unwrap_or(0);
    let direct = pip_get_boolean("SeparateChunks", 0).unwrap_or(0) == 0;
    boundpixels = pip_get_integer("BoundaryPixels", boundpixels).unwrap_or(boundpixels);
    let mut startnum = pip_get_integer("InitialComNumber", 1).unwrap_or(1);
    let ifstartnum = 1 - pip_get_err_no();
    let leaveopen = pip_get_boolean("OpenForMoreComs", 0).unwrap_or(0);
    let dimens = pip_get_two_integers("DimensionsOfStack", (0, 0)).unwrap_or((0, 0));
    let if_dimens = 1 - pip_get_err_no();
    let unique_info = pip_get_boolean("UniqueInfoFile", 0).unwrap_or(0);

    // Set min and target slabs if not entered
    if minslabs == 0 {
        minslabs = minratio * numproc;
    }
    if targetslabs == 0 {
        targetslabs = targetratio * numproc;
    }
    targetslabs = minslabs.max(targetslabs);

    // Collect info from command file
    let mut comlines =
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
    let xtilt_arr = floats(option_value(&comlines, "xaxistilt", 2, true, 0, None, None));
    let mut fullimage = ints(option_value(&comlines, "fullimage", 1, true, 0, None, None));
    let thick_arr = ints(option_value(&comlines, "thickness", 1, true, 0, None, None));
    let mut slices = ints(option_value(&comlines, "slice", 1, true, 0, None, None));
    let localali = strings(option_value(&comlines, "localfile", 0, true, 0, None, None));
    let binning_arr = ints(option_value(
        &comlines,
        "imagebinned",
        1,
        true,
        0,
        None,
        None,
    ));
    let expanded_fac = floats(option_value(
        &comlines,
        "expandedbyfactor",
        2,
        true,
        1,
        None,
        None,
    ))
    .map(|values| values[0]);
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
    let rectoproj = strings(option_value(
        &comlines,
        "recfiletoreproj",
        0,
        true,
        0,
        None,
        None,
    ));
    let sliceproj = floats(option_value(&comlines, "reproject", 2, true, 0, None, None));
    let width_arr = ints(option_value(&comlines, "width", 1, true, 0, None, None));
    let intsirt = ints(option_value(
        &comlines,
        "sirtiterations",
        1,
        true,
        0,
        None,
        None,
    ));
    let xtilt_file = strings(option_value(&comlines, "xtiltfile", 0, true, 0, None, None));
    let zfactors = strings(option_value(
        &comlines,
        "zfactorfile",
        0,
        true,
        0,
        None,
        None,
    ));
    let vs_out_file = strings(option_value(
        &comlines,
        "vertsliceoutputfile",
        0,
        true,
        0,
        None,
        None,
    ));

    let mut binval = 1;
    if let Some(values) = binning_arr.as_ref().filter(|values| !values.is_empty()) {
        binval = values[0];
    }
    let mut xaxistilt = 0.;
    if let Some(values) = xtilt_arr.as_ref().filter(|values| !values.is_empty()) {
        xaxistilt = values[0];
    }
    let mut reproj = 0;
    if intsirt.is_none()
        && (rectoproj.as_deref().is_some_and(|value| !value.is_empty())
            || sliceproj.as_ref().is_some_and(|values| !values.is_empty()))
    {
        reproj = 1;
    }
    let expanded_fac = match expanded_fac {
        Some(value) if value != 0. => value,
        _ => 1.,
    };

    // Figure out if vertical slices are even possible
    let mut vert_possible = localali.is_none() && zfactors.is_none() && reproj == 0;
    if vert_possible && let Some(xtilt_file) = xtilt_file.filter(|file| !file.is_empty()) {
        let xtlines =
            read_text_file(&xtilt_file, Some("X-tilt file"), false, None).unwrap_or_default();
        let py_float = |text: &str| -> f64 {
            match imodpy::py_float(text) {
                Some(value) => value,
                None => {
                    eprintln!("ValueError: could not convert string to float: '{text}'");
                    crate::imod::libcfshr::b3dutil::exit(1)
                }
            }
        };
        let firstxt = py_float(&xtlines[0]);
        let mut broke = false;
        for i in 0..xtlines.len() {
            if (py_float(&xtlines[i]) - firstxt).abs() > 1.0e-5 {
                vert_possible = false;
                broke = true;
                break;
            }
        }
        if !broke {
            xaxistilt += firstxt;
        }
    }

    // Get the input and output image file names from the command file if necessary
    // DUPLICATE OF SIRTSETUP
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
    // `recfile` is None here only when `inputproj` was found and `outputfile` was
    // not; the source then fails on `os.path.splitext(None)`.
    let Some(recfile) = recfile else {
        eprintln!("TypeError: expected str, bytes or os.PathLike object, not NoneType");
        crate::imod::libcfshr::b3dutil::exit(1)
    };

    let fullimage_true = fullimage.as_ref().is_some_and(|values| !values.is_empty());
    let slices_true = slices.as_ref().is_some_and(|values| !values.is_empty());
    let noali = !Path::new(&alifile).exists();
    if noali && if_dimens == 0 && !fullimage_true && !slices_true {
        exit_error(
            &("Command file has neither a SLICE nor a FULLIMAGE entry and ".to_owned()
                + "image file does not exist yet"),
        );
    }

    if direct
        && noali
        && if_dimens == 0
        && !fullimage_true
        && !width_arr.as_ref().is_some_and(|values| !values.is_empty())
    {
        exit_error(
            &("Command file has neither a WIDTH nor a FULLIMAGE entry and ".to_owned()
                + "image file does not exist yet"),
        );
    }

    // Python's `a // b` on ints and `a / b` on floats raise ZeroDivisionError
    // for a zero divisor, and `int(x)` / `int(round(x))` / `math.floor(x)` of a
    // float raise for NaN or infinity (IMAGEBINNED 0, `-n 0`, `-penalty 1`, a
    // zero width, entries and X-tilt values Python's `float()` reads as
    // `inf`/`nan`); none is caught, so each is a traceback and exit status 1.
    let floordiv = |a: i32, b: i32| py_int_floordiv(i64::from(a), i64::from(b)) as i32;
    let float_div = py_true_div;
    let int_of = |value: f64| py_int_of_float(value) as i32;

    // Divide thickness by the binning for computations
    let thickness = match thick_arr.as_ref().filter(|values| !values.is_empty()) {
        Some(values) => int_of(float_div(expanded_fac * values[0] as f64, binval as f64).floor()),
        None => exit_error("Command file has no THICKNESS entry"),
    };

    // Extract root and extension for making filenames: if it is a type extension and there
    // is and _rec or _xxx before it, pull that off, otherwise drop back to no type extension
    let (mut recroot, mut recext) = os_path_splitext(&recfile);
    let recroot_chars = recroot.chars().collect::<Vec<_>>();
    if !type_ext.is_empty()
        && recroot_chars.len() > 4
        && recroot_chars[recroot_chars.len() - 4] == '_'
    {
        recext = recroot_chars[recroot_chars.len() - 3..].iter().collect();
        recroot = recroot_chars[..recroot_chars.len() - 4].iter().collect();
    } else {
        recext = recext.chars().skip(1).collect();
        type_ext = String::new();
    }
    set_root_and_extension(&recroot, &type_ext);

    // Remove any previous files now in case the number has changed or
    // direct/indirect mode.  Processchunks takes care of other files
    clean_chunk_files(&rootname, ifstartnum != 0);

    // Get the size from the supplied dimension or from the aligned stack instead
    // of relying on FULLIMAGE if possible, and scale them up by binning
    //
    if !noali || if_dimens != 0 {
        let mut sizes = if if_dimens != 0 {
            vec![dimens.0, dimens.1]
        } else {
            match get_mrc_size(&alifile) {
                Ok((nx, ny, nz)) => vec![nx, ny, nz],
                Err(_) => exit_from_imod_error(progname),
            }
        };

        sizes[0] *= binval;
        sizes[1] *= binval;
        fullimage = Some(sizes);
    } else {
        prnstr(
            &(format!("WARNING: {progname} - aligned stack not found; sizes will")
                + " be taken from FULLIMAGE entry"),
            "\n",
            false,
        );
    }

    let mut firstslice = 0;
    let mut numslices = 0;
    if let Some(values) = fullimage.as_ref().filter(|values| !values.is_empty()) {
        firstslice = 0;
        numslices = floordiv(values[1] + binval - 1, binval);
    }

    if let Some(values) = slices.as_ref().filter(|values| !values.is_empty()) {
        firstslice = floordiv(int_of(py_round(expanded_fac * values[0] as f64)), binval);
        numslices =
            floordiv(int_of(py_round(expanded_fac * values[1] as f64)), binval) + 1 - firstslice;
    }

    // Get the width before possibly changing binval
    let widthnum = match width_arr.as_ref().filter(|values| !values.is_empty()) {
        Some(values) => floordiv(int_of((expanded_fac * values[0] as f64).floor()), binval),
        None => match fullimage.as_ref() {
            Some(values) => floordiv(values[0], binval),
            None => {
                eprintln!("TypeError: 'NoneType' object is not subscriptable");
                crate::imod::libcfshr::b3dutil::exit(1)
            }
        },
    };

    // If reprojecting from tomo, need to get real number of slices and starting one
    let mut slicedel = "SLICE";
    if reproj != 0 && rectoproj.as_deref().is_some_and(|value| !value.is_empty()) {
        slicedel = "ZMinAndMax";
        slices = ints(option_value(
            &comlines,
            "ZMinAndMax",
            1,
            true,
            0,
            None,
            None,
        ));
        firstslice = 0;
        if let Some(values) = slices.as_ref().filter(|values| !values.is_empty()) {
            firstslice = values[0];
            numslices = values[1] + 1 - firstslice;
        }
        binval = 1;
    }

    // Start with target size, make sure bigger than minimum
    let mut slabsize = minslices.max(floordiv(numslices, targetslabs));

    if vert_possible && xaxistilt != 0. {
        // If no locals or Z factors and X axis tilt, go for maximum # of slabs
        // that has a percentage of extra slices within a minimal limit, down to the
        // "min" # of slabs
        let mut nslabs = targetslabs;
        let extrathick = (thickness as f64 * (0.01745329 * xaxistilt).sin()).abs();
        let mut extranum = 0;
        while nslabs >= minslabs {
            slabsize = minslices.max(floordiv(numslices, nslabs));
            nslabs -= 1;

            // Get percent of extra slices required
            extranum = int_of(float_div(
                100. * (slabsize as f64 + extrathick),
                slabsize as f64,
            ));
            if extranum <= maxextrapct {
                break;
            }
        }

        let pennum = int_of(100. * penalty);

        // If extra is less than penalty, proceed
        // Otherwise, drop to old-style tilting unless vertical specified
        if extranum > pennum {
            if vertical == 0 {
                oldstyle = "XTILTINTERP 0".to_owned();
            } else {
                // If vertical specified, compute optimum size that just breaks
                // even with penalty for old-style tilting, but limit it
                // However, in this case allow it to go down to one chunk per processor
                slabsize = int_of(float_div(extrathick, penalty - 1.));
                let maxsize = floordiv(numslices, numproc);
                slabsize = minslices.max(maxsize.min(slabsize));
            }
        }
    }

    let numslabs = 1.max(floordiv(
        numslices + py_int_floordiv(i64::from(slabsize), 2) as i32,
        slabsize,
    ));
    let slabsize = numslices.div_euclid(numslabs);
    let remainder = numslices.rem_euclid(numslabs);

    // Now that slab size is known, get # of bound lines
    //
    let mut boundlines = floordiv(boundpixels + widthnum - 1, widthnum);
    if reproj != 0 {
        boundlines = boundlines.min(slabsize.div_euclid(2) + 1);
    } else if slabsize == 1 {
        boundlines = boundlines.min(thickness.div_euclid(2) + 1);
    } else {
        boundlines = boundlines.min(thickness - 1);
    }

    // Manage output file type: pass through incoming line or...
    if !comlines
        .iter()
        .any(|line| line.starts_with("$setenv IMOD_OUTPUT_FORMAT"))
    {
        // ELSE ON FOR: protect against strange settings on other machines
        let mut out_format = std::env::var("IMOD_OUTPUT_FORMAT").unwrap_or_default();
        if out_format.is_empty() || !standard_type_extensions().contains(&out_format) {
            out_format = "MRC".to_owned();
        }
        comlines.insert(0, format!("$setenv IMOD_OUTPUT_FORMAT {out_format}"));
    }

    let mut templist: Vec<String> = Vec::new();
    let mut recsed = recfile.clone();
    let mintotslice = binval * firstslice;
    let maxtotslice = binval * (firstslice + numslices - 1);
    let mut totsed = "gibberish";
    let mut boundsed = "gibberish";
    let mut boundfile = format!("{rootname}-bound.info");
    if unique_info != 0 {
        boundfile = format!("{rootname}-bound-{startnum:03}.info");
    }

    let mut total_coms = 0;
    if reproj != 0 {
        boundfile = format!("{rootname}-rpbound.info");
        if unique_info != 0 {
            boundfile = format!("{rootname}-rpbound-{startnum:03}.info");
        }
    }
    let vs_out_file = vs_out_file.filter(|file| !file.is_empty());
    let mut vs_bound_file = String::new();
    if vs_out_file.is_some() {
        vs_bound_file = format!("{rootname}-vsbound.info");
    }
    let mut boundtext: Vec<String> = Vec::new();
    let mut vs_bound_text: Vec<String> = Vec::new();
    if direct {
        totsed = "THICKNESS";
        recsed = "gibberish".to_owned();
        boundsed = "THICKNESS";
        let mut thiscom = format!("{rootname}-start{com_ext}");
        if ifstartnum != 0 {
            thiscom = format!("{rootname}-{startnum:03}-sync{com_ext}");
            startnum += 1;
        }

        let sedcom = vec![
            format!(r"|^\s*{slicedel}|d"),
            "|savework|d".to_owned(),
            format!("|^ *THICKNESS|a|{slicedel} -1 -1|"),
            format!("|^ *THICKNESS|a|TOTALSLICES {mintotslice} {maxtotslice}|"),
            sed_modify("ActionIfGPUFails", "2,2", '|'),
        ];
        let mut sedlines = pysed(&sedcom, PysedSrc::Lines(&comlines), None, true, '|', false)
            .ok()
            .flatten()
            .unwrap_or_default();
        sedlines.push("$sync".to_owned());
        let _ = write_text_file(&thiscom, &sedlines, false);
        total_coms += 1;

        let boundhead = format!("1 {reproj} {widthnum} {boundlines} {numslabs}");
        boundtext = vec![boundhead.clone()];
        if vs_out_file.is_some() {
            vs_bound_text = vec![boundhead];
        }
    }

    let firstofall = firstslice;
    let mut num = 0;
    for slab in 1..numslabs + 1 {
        num = slab;
        let numrec = num + startnum - 1;
        let numtext = format!("{numrec:03}");
        let thiscom = format!("{rootname}-{numtext}{com_ext}");
        let tempname = dataset_filename(&format!("-{numtext}.{recext}"), None, None);
        templist.push(tempname.clone());
        let mut lastslice = firstslice + slabsize - 1;
        if num <= remainder {
            lastslice += 1;
        }

        // Get unbinned first and last slices for output
        let ubfirst = firstslice * binval;
        let ublast = lastslice * binval;

        // Modify the command file: delete existing slice, get rid of savework,
        // Set the new slice command and the xtiltinterp control
        let mut sedcom = vec![
            format!("|{recsed}|s||{tempname}|"),
            format!(r"|^\s*{slicedel}|d"),
            "|savework|d".to_owned(),
            format!("|^ *THICKNESS|a|{slicedel} {ubfirst} {ublast}|"),
            format!("|^ *THICKNESS|a|{oldstyle}|"),
            format!("|{totsed}|a|TOTALSLICES {mintotslice} {maxtotslice}|"),
            format!("|{boundsed}|a|BoundaryInfoFile {boundfile}|"),
            sed_modify("ActionIfGPUFails", "2,2", '|'),
        ];
        if vs_out_file.is_some() {
            sedcom.push(format!("|{boundsed}|a|VertBoundaryFile {vs_bound_file}|"));
        }
        let mut sedlines = pysed(&sedcom, PysedSrc::Lines(&comlines), None, true, '|', false)
            .ok()
            .flatten()
            .unwrap_or_default();
        sedlines.insert(0, "$sync".to_owned());
        let _ = write_text_file(&thiscom, &sedlines, false);
        total_coms += 1;
        if direct {
            boundtext.push(format!("{recroot}-{numtext}.{boundext}"));
            let mut boundstart = firstslice - firstofall;
            let mut boundend = lastslice - firstofall;
            if reproj != 0 {
                boundend -= boundlines - 1;
            }
            if num == 1 {
                boundstart = -1;
            }
            if num == numslabs {
                boundend = -1;
            }
            let boundtmp = if reproj != 0 {
                format!("-1 {boundstart} -1 {boundend}")
            } else {
                format!("{boundstart} 0 {boundend} -1")
            };
            boundtext.push(boundtmp.clone());
            if vs_out_file.is_some() {
                vs_bound_text.push(format!("{recroot}-{numtext}.{vs_bound_ext}"));
                vs_bound_text.push(boundtmp);
            }
        }
        firstslice = lastslice + 1;
    }

    let mut finish = format!("{rootname}-finish{com_ext}");
    let mut cleanup = format!(
        "$b3dremove -g {rootname}-[0-9][0-9][0-9]*{com_ext}* {rootname}-[0-9][0-9][0-9]*.log* "
    );
    let mut cleanbound = format!("\"{boundfile}\"");
    if vs_out_file.is_some() {
        cleanbound += &format!(" \"{vs_bound_file}\"");
    }
    if leaveopen != 0 {
        finish = format!("{rootname}-{:03}-sync{com_ext}", num + startnum);
        cleanup = "$b3dremove -g ".to_owned();
        cleanbound = String::new();
    }

    let mut finishlines: Vec<String>;
    if !direct && reproj == 0 {
        finishlines = vec![
            "$newstack -StandardInput".to_owned(),
            format!("OutputFile {recfile}"),
        ];
        for num in 0..numslabs as usize {
            finishlines.push(format!("InputFile {}", templist[num]));
        }
        finishlines.push(format!("{cleanup}\"{recroot}-[0-9][0-9][0-9]*.{recext}\""));
    } else if !direct {
        finishlines = vec![
            "$assemblevol -StandardInput".to_owned(),
            format!("OutputFile {recfile}"),
            format!("NumberOfFilesInY {numslabs}"),
        ];

        for num in 0..numslabs as usize {
            finishlines.push(format!("InputFile {}", templist[num]));
        }

        finishlines.push(format!("{cleanup}\"{recroot}-[0-9][0-9][0-9]*.{recext}\""));
    } else {
        finishlines = vec![
            format!("$fixboundaries \"{recfile}\" \"{boundfile}\""),
            format!("$collectmmm pixels= \"{rootname}\" {numslabs} \"{recfile}\" {startnum}"),
            format!("{cleanup}\"{recroot}-[0-9][0-9][0-9]*.{boundext}\" {cleanbound}"),
        ];
        if let Some(vs_out_file) = &vs_out_file {
            finishlines.insert(
                1,
                format!("$fixboundaries \"{vs_out_file}\" \"{vs_bound_file}\""),
            );
            finishlines.push(format!(
                "{cleanup}\"{recroot}-[0-9][0-9][0-9]*.{vs_bound_ext}\""
            ));
        }
    }

    let _ = write_text_file(&finish, &finishlines, false);
    total_coms += 1;
    if direct {
        let _ = write_text_file(&boundfile, &boundtext, false);
        if vs_out_file.is_some() {
            let _ = write_text_file(&vs_bound_file, &vs_bound_text, false);
        }
    }

    prnstr(
        &format!("{total_coms} command files for {numslabs} chunks created and ready to run"),
        "\n",
        false,
    );
    prnstr(
        "  with processchunks or parallel processing interface in Etomo",
        "\n",
        false,
    );
    0
}
