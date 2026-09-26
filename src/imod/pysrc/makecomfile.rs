//! Translation of `IMOD/pysrc/makecomfile`.
//!
//! A Python command script with no functions; its top level is
//! [`makecomfile`], translated statement by statement.  `header` (through
//! `getmrcsize`) is this crate's own program and runs in process;
//! `montagesize` and `goodframe` are processes.
//!
//! Python values: `PipGetFloat`/`PipGetTwoFloats` give doubles and `str()` of
//! one is `repr` ([`py_str_float`]); `round()` rounds half to even.

use super::batchruntomo::py_str_float;
use super::comchanger::{Change, modify_for_change_list, process_change_options};
use super::imodpy::{
    OptionValue, add_imod_bin_ignore_sighup, add_output_format_var_to_lines, dataset_filename,
    exit_from_imod_error, find_root_axis_and_extensions, get_montage_size, get_mrc_size,
    get_naming_style, make_backup_file, map_type_extension_to_style, option_value,
    os_path_splitext, read_text_file, run_goodframe, set_root_and_extension, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_float, pip_get_in_out_file,
    pip_get_integer, pip_get_string, pip_get_two_floats, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed};
use regex::Regex;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`makecomfile:1-473`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn makecomfile(arguments: &[OsString]) -> i32 {
    let progname = "makecomfile";
    let prefix = format!("ERROR: {progname} - ");
    let orig_com_dir = "origcoms";
    let dflt_com_dir = "dfltcoms";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 makecomfile
    let options: Vec<String> = [
        "input:InputFile:FN:",
        "output:OutputFile:FN:",
        "root:RootNameOfDataFiles:CH:",
        "style:NamingStyle:I:",
        "stackext:StackExtension:CH:",
        "single:SingleAxis:B:",
        "binning:BinningOfImages:I:",
        "bead:BeadSize:F:",
        "use:Use3dfindAliInput:B:",
        "shift:ShiftInY:F:",
        "halffloat:HalfFloatModeOutput:I:",
        "thickness:ThicknessToMake:I:",
        "find:FindBeadsInVolume:I:",
        "gpu:UseGPU:I:",
        "procs:NumberOfProcessors:I:",
        "local:LocalAlignValidation:I:",
        "ratios:TargetAndMinRatios:FP:",
        "skipbeam:SkipBeamTiltWithOneRot:B:",
        "change:ChangeParametersFile:FNM:",
        "one:OneParameterChange:CHM:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    // Table of com files and whether they need rootname, binning, and input file, and a
    // type extension for setting IMOD_OUTPUT_FORMAT
    // indexes to these elements.  Note tilt_3dfind_reproject must stay before tilt_3dfind
    let need_root = 1;
    let need_bin = 2;
    let need_infile = 3;
    let need_type_ext = 4;
    let type_table: [(&str, [bool; 4]); 16] = [
        ("xcorr_pt", [true, true, true, false]),
        ("autofidseed", [false, false, false, false]),
        ("transferfid", [true, false, false, false]),
        ("cryoposition", [true, false, false, true]),
        ("newst_3dfind", [true, true, true, false]),
        ("blend_3dfind", [true, true, true, false]),
        ("tilt_3dfind_reproject", [true, false, true, false]),
        ("tilt_3dfind", [true, true, true, false]),
        ("findbeads3d", [true, true, false, false]),
        ("golderaser", [true, false, false, true]),
        ("sirtsetup", [false, false, false, true]),
        ("ctf3dsetup", [false, false, true, false]),
        ("restrictalign", [false, false, true, false]),
        ("reducefiltvol", [true, true, true, true]),
        ("extenderasemod", [true, false, false, false]),
        ("varypatchtrack", [false, false, false, false]),
    ];
    // `typeTable[comType][need...]`: the flags are elements 1-4 of each row
    let table_flag = |com_type: usize, index: usize| type_table[com_type].1[index - 1];

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 1);

    let outfile = pip_get_in_out_file("OutputFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if outfile.is_empty() {
        exit_error("Output com file must be entered");
    }

    // Look up type from output name
    let com_type = match type_table
        .iter()
        .position(|entry| outfile.starts_with(entry.0))
    {
        Some(ind) => ind,
        // ELSE ON FOR
        None => exit_error(&format!(
            "{outfile} is not a recognized type of output com file"
        )),
    };
    let type_name = type_table[com_type].0;

    let reduce_filt = type_name == "reducefiltvol";

    let mut infile = String::new();
    let mut comlines: Vec<String> = Vec::new();
    if table_flag(com_type, need_infile) {
        infile = pip_get_string("InputFile", "").unwrap_or_default();
        if infile.is_empty() {
            exit_error("Input file must be entered for this type of com file");
        }
        if !reduce_filt {
            comlines = read_text_file(&infile, None, false, None).unwrap_or_default();
        }
    }

    // Set up axis letter and base output name without axis
    let mut axislet = "";
    let (mut baseout, _ext) = os_path_splitext(&outfile);
    let single = pip_get_string("SingleAxis", "").unwrap_or_default();
    if single.is_empty() {
        for letter in ["a", "b"] {
            if baseout.ends_with(letter) {
                axislet = letter;
            }
        }
    }

    if !axislet.is_empty() {
        baseout.pop();
    }

    let (_out_root, mut com_ext) = os_path_splitext(&outfile);
    if com_ext != ".pcm" && com_ext != ".com" {
        com_ext = String::new();
    }

    // If need root name or missing an extension, need to get all the options, then get
    // the various items from dataset files, and use the ones needed
    let need_names = table_flag(com_type, need_root);
    let need_style = table_flag(com_type, need_type_ext);
    // `rootname` and `nameStyle` are assigned only in this branch; a later use of
    // either when it did not run is a NameError in the source
    let mut rootname: Option<String> = None;
    let mut name_style: Option<i32> = None;
    let mut type_ext: Option<String> = None;
    let mut stack_ext = String::new();
    if need_names || need_style || com_ext.is_empty() {
        let mut root_value = pip_get_string("RootNameOfDataFiles", "").unwrap_or_default();
        stack_ext = pip_get_string("StackExtension", "").unwrap_or_default();
        let (style, style_ext) = match get_naming_style(None, false, false) {
            Ok(result) => result,
            Err(message) => exit_error(&message),
        };
        type_ext = style_ext;

        if root_value.is_empty() || style < 0 || stack_ext.is_empty() || com_ext.is_empty() {
            let (ds_com_ext, _dual_num, root, ds_type_ext, ds_stack_ext) =
                find_root_axis_and_extensions(0, None);
            if need_names && root_value.is_empty() {
                if !root.is_empty() {
                    root_value = root;
                } else {
                    exit_error(
                        &("Root name of dataset files must be entered, it cannot be ".to_owned()
                            + "determined from dataset"),
                    );
                }
            }
            if (need_names || need_style) && style < 0 {
                if ds_type_ext.is_some() {
                    type_ext = ds_type_ext;
                } else {
                    exit_error(
                        &("Naming style must be entered, it cannot be determined from ".to_owned()
                            + "dataset"),
                    );
                }
            }
            if need_names && stack_ext.is_empty() {
                if !ds_stack_ext.is_empty() {
                    stack_ext = ds_stack_ext;
                } else {
                    exit_error(
                        &("Raw stack extension must be entered, it cannot be determined "
                            .to_owned()
                            + "from dataset"),
                    );
                }
            }
            if com_ext.is_empty() {
                if !ds_com_ext.is_empty() {
                    com_ext = format!(".{ds_com_ext}");
                } else {
                    exit_error(
                        &("Cannot determine extension of command files from dataset or "
                            .to_owned()
                            + "output file"),
                    );
                }
            }
        }

        if need_names {
            set_root_and_extension(&root_value, type_ext.as_deref().unwrap_or(""));
        }
        rootname = Some(root_value);
        name_style = Some(style);
    }

    // `rootname` and `nameStyle` are assigned only in the branch above; using
    // one when it did not run would be an uncaught NameError in the source.
    // Every type that uses them sets `needRoot` or `needTypeExt`, so the
    // branch always ran for them.
    let name_error = |name: &str| -> ! {
        eprintln!("Traceback (most recent call last):");
        eprintln!("NameError: name '{name}' is not defined");
        std::process::exit(1)
    };
    let rootname_value = || -> String {
        match &rootname {
            Some(value) => value.clone(),
            None => name_error("rootname"),
        }
    };

    // `binning` is an int, or a float for reducefiltvol
    let mut binning: i32 = 0;
    let mut binning_float: f64 = 0.;
    if table_flag(com_type, need_bin) {
        if reduce_filt {
            binning_float = pip_get_float("BinningOfImages", 0.).unwrap_or(0.);
            if binning_float <= 0. {
                exit_error("Binning of images must be entered");
            }
        } else {
            binning = pip_get_integer("BinningOfImages", 0).unwrap_or(0);
            if binning <= 0 {
                exit_error("Binning of images must be entered");
            }
        }
    }

    // Get the change list.  Base changes are ones set up here with defaukt entries,
    // final changes are ones specified by arguments, which override any other changes
    let mut sedcom: Vec<String> = Vec::new();
    let mut base_changes: Vec<Change> = Vec::new();
    let mut final_changes: Vec<Change> = Vec::new();
    let change_list = process_change_options(
        "ChangeParametersFile",
        "OneParameterChange",
        "comparam",
        4,
        false,
    );
    let setup_list = process_change_options(
        "ChangeParametersFile",
        "OneParameterChange",
        "setupset",
        -3,
        false,
    );
    let std_lines = vec![
        format!("# Command file for running {type_name} created by makecomfile"),
        format!("${type_name} -StandardInput"),
    ];
    let with_std = |more: Vec<String>| -> Vec<String> {
        let mut lines = std_lines.clone();
        lines.extend(more);
        lines
    };

    // XCORR_PT
    if type_name == "xcorr_pt" {
        let rootname = rootname_value();
        let tiltxcorr = Regex::new(r"^\$ *tiltxcorr").unwrap();
        match comlines.iter().position(|line| tiltxcorr.is_match(line)) {
            Some(ind) => comlines = comlines[ind..].to_vec(),
            None => exit_error("Cannot find tiltxcorr command line in file"),
        }

        // If the preali does not exist or there is an error reading the header, fall back
        // to the raw stack size, enhanced properly for montage
        let mut need_raw = true;
        let mut nx: f64 = 0.;
        let mut ny: f64 = 0.;
        if Path::new(&dataset_filename(".preali", None, None)).exists() {
            if let Ok((x, y, _z)) = get_mrc_size(&dataset_filename(".preali", None, None)) {
                nx = x as f64;
                ny = y as f64;
                need_raw = false;
            }
        }

        if need_raw {
            // Get the raw stack size, assuming a montage if there is a .pl
            let (x, y) = if Path::new(&format!("{rootname}.pl")).exists() {
                let (montx, monty, _montz) = match get_montage_size(
                    &format!("{rootname}.{stack_ext}"),
                    Some(&format!("{rootname}.pl")),
                ) {
                    Ok(size) => size,
                    Err(_) => exit_from_imod_error(progname),
                };
                let (x, y) = run_goodframe(montx, monty);
                if x < 0 {
                    exit_error("Running goodframe to get prealigned stack size");
                }
                (x, y)
            } else {
                match get_mrc_size(&format!("{rootname}.{stack_ext}")) {
                    Ok((x, y, _z)) => (x, y),
                    Err(_) => exit_from_imod_error(progname),
                }
            };
            nx = x as f64 / binning as f64;
            ny = y as f64 / binning as f64;
        }

        let fid_ext = "_pt.fid";
        let mut last_line = String::new();
        if comlines
            .last()
            .is_some_and(|line| line.contains("savework"))
        {
            last_line = comlines.pop().unwrap_or_default();
        }
        let mut new_lines = vec!["$goto doxcorr".to_owned(), "$doxcorr:".to_owned()];
        new_lines.extend(comlines);
        new_lines.extend([
            "$dochop:".to_owned(),
            "$imodchopconts -StandardInput".to_owned(),
            format!("InputModel {rootname}{fid_ext}"),
            format!("OutputModel {rootname}.fid"),
            "MinimumOverlap 4".to_owned(),
            "AssignSurfaces 1".to_owned(),
        ]);
        comlines = new_lines;
        if !last_line.is_empty() {
            comlines.push(last_line);
        }

        let xborder = (0.05 * nx + 0.5) as i32;
        let yborder = (0.05 * ny + 0.5) as i32;
        sedcom = vec![
            format!("/^OutputFile/s/[ \t].*/\t{rootname}{fid_ext}/"),
            format!(
                "/^InputFile/s/[ \t].*/\t{}/",
                dataset_filename(".preali", None, None)
            ),
            format!("/^OutputFile/a/PrealignmentTransformFile\t{rootname}.prexg/"),
            "/^CumulativeCorrelation/d".to_owned(),
            "/^SearchMag/d".to_owned(),
        ];

        let prefix = [format!("{baseout}{axislet}"), "tiltxcorr".to_owned()];
        base_changes = vec![
            [
                &prefix[..],
                &["BordersInXandY".to_owned(), format!("{xborder},{yborder}")],
            ]
            .concat(),
            [
                &prefix[..],
                &["IterateCorrelations".to_owned(), "1".to_owned()],
            ]
            .concat(),
        ];
        final_changes = vec![
            [
                &prefix[..],
                &["ImagesAreBinned".to_owned(), binning.to_string()],
            ]
            .concat(),
        ];

    // AUTOFIDSEED
    } else if type_name == "autofidseed" {
        comlines = with_std(vec![
            format!("TrackCommandFile\ttrack{axislet}{com_ext}"),
            "MinSpacing\t0.85".to_owned(),
            "PeakStorageFraction\t1.0".to_owned(),
        ]);
        if axislet == "b" {
            comlines.push("AppendToSeedModel\t1".to_owned());
        }

    // TRANSFERFID
    } else if type_name == "transferfid" {
        comlines = with_std(vec![
            format!("Setname\t{}", rootname_value()),
            "CorrespondingCoordFile\ttransferfid.coord".to_owned(),
        ]);

    // CRYOPOSITION
    } else if type_name == "cryoposition" {
        let thickness = pip_get_integer("ThicknessToMake", 0).unwrap_or(0);
        if thickness == 0 {
            exit_error("Thickness must be entered");
        }
        let prefix = [format!("{baseout}{axislet}"), "cryoposition".to_owned()];
        comlines = with_std(vec![format!("RootName {}", rootname_value())]);
        let bead_size = pip_get_float("BeadSize", 0.).unwrap_or(0.);
        if pip_get_err_no() == 0 {
            comlines.push(format!("BeadSize {}", py_str_float(bead_size)));
        }

        final_changes = vec![
            [
                &prefix[..],
                &["ThicknessOfTomograms".to_owned(), thickness.to_string()],
            ]
            .concat(),
        ];
        for option in ["FindBeadsInVolume", "UseGPU"] {
            let value = pip_get_integer(option, 0).unwrap_or(0);
            if pip_get_err_no() == 0 {
                final_changes.push([&prefix[..], &[option.to_owned(), value.to_string()]].concat());
            }
        }

    // NEWST_3DFIND/BLEND_3DFIND
    } else if type_name == "newst_3dfind" || type_name == "blend_3dfind" {
        let mut match_opt = "^OutputFile";
        if type_name == "blend_3dfind" {
            match_opt = "^ImageOutputFile";
        }
        sedcom = vec![format!(
            "/{match_opt}/s/[ \t].*/\t{}/",
            dataset_filename("_3dfind.ali", None, None)
        )];
        let prefix = if type_name == "newst_3dfind" {
            [format!("{baseout}{axislet}"), "newstack".to_owned()]
        } else {
            [format!("{baseout}{axislet}"), "blendmont".to_owned()]
        };
        final_changes = vec![
            [
                &prefix[..],
                &["BinByFactor".to_owned(), binning.to_string()],
            ]
            .concat(),
        ];
        if type_name == "blend_3dfind" {
            base_changes = vec![
                [
                    &prefix[..],
                    &["OldEdgeFunctions".to_owned(), "1".to_owned()],
                ]
                .concat(),
            ];
        }

    // TILT_3DFIND
    } else if type_name == "tilt_3dfind" {
        let thickness = pip_get_integer("ThicknessToMake", 0).unwrap_or(0);
        if thickness == 0 {
            exit_error("Thickness must be entered");
        }
        let yshift = pip_get_float("ShiftInY", 0.).unwrap_or(0.);
        let use3d = pip_get_boolean("Use3dfindAliInput", 0).unwrap_or(0);
        sedcom = vec![format!(
            "/^OutputFile/s/[ \t].*/\t{}/",
            dataset_filename("_3dfind.rec", None, None)
        )];
        if use3d != 0 {
            sedcom.push(format!(
                "/^InputProjections/s/[ \t].*/\t{}/",
                dataset_filename("_3dfind.ali", None, None)
            ));
        }
        let prefix = [format!("{baseout}{axislet}"), "tilt".to_owned()];
        final_changes = vec![
            [
                &prefix[..],
                &["IMAGEBINNED".to_owned(), binning.to_string()],
            ]
            .concat(),
            [
                &prefix[..],
                &["THICKNESS".to_owned(), thickness.to_string()],
            ]
            .concat(),
        ];
        if yshift != 0. {
            final_changes.push(
                [
                    &prefix[..],
                    &["SHIFT".to_owned(), format!("0. {yshift:.2}")],
                ]
                .concat(),
            );
        }

    // FINDBEADS3D
    } else if type_name == "findbeads3d" {
        let rootname = rootname_value();
        let bead_size = pip_get_float("BeadSize", 0.).unwrap_or(0.);
        if pip_get_err_no() != 0 {
            exit_error("BeadSize must be entered");
        }
        let prefix = [format!("{baseout}{axislet}"), "findbeads3d".to_owned()];
        comlines = with_std(vec![
            format!("InputFile\t{}", dataset_filename("_3dfind.rec", None, None)),
            format!("OutputFile\t{rootname}_3dfind.mod"),
            "MinRelativeStrength\t0.05".to_owned(),
            "MinSpacing\t0.9".to_owned(),
            "StorageThreshold\t0".to_owned(),
        ]);
        if Path::new(&format!("{rootname}.tlt")).exists() {
            comlines.extend([
                format!("TiltFile {rootname}.tlt"),
                "YAxisElongated".to_owned(),
            ]);
        }
        for change in &change_list {
            if (change[0] == "newst" || change[0] == format!("newst{axislet}"))
                && change[1] == "newstack"
                && change[2] == "ExpandByFactor"
            {
                comlines.push(format!("ExpandedByFactor\t{}", change[3]));
                break;
            }
        }
        final_changes = vec![
            [
                &prefix[..],
                &["BinningOfVolume".to_owned(), binning.to_string()],
            ]
            .concat(),
            [
                &prefix[..],
                &["BeadSize".to_owned(), format!("{bead_size:.2}")],
            ]
            .concat(),
        ];

    // GOLDERASER
    } else if type_name == "golderaser" {
        let rootname = rootname_value();
        let bead_size = pip_get_float("BeadSize", 0.).unwrap_or(0.);
        if pip_get_err_no() != 0 {
            exit_error("BeadSize must be entered");
        }

        // The option overrides any directives, it will go in finalChanges
        let half_floats = pip_get_integer("HalfFloatModeOutput", 0).unwrap_or(0);
        let half_final = pip_get_err_no() == 0;

        // If there is no option entry, a copyarg directive means put option in explicitly
        // Then a comparam directive (!) could override that
        let mut half_in_com: Option<String> = None;
        if !half_final {
            for change in &setup_list {
                if change[0] == "copyarg" && change[1] == "halffloat" {
                    half_in_com = Some(change[2].clone());
                    break;
                }
            }
        }

        comlines = vec![
            "# Command file for running ccderaser created by makecomfile".to_owned(),
            "$ccderaser -StandardInput".to_owned(),
            format!("InputFile\t{}", dataset_filename(".ali", None, None)),
            format!("OutputFile\t{}", dataset_filename("_erase.ali", None, None)),
            format!("ModelFile\t{rootname}_erase.fid"),
            "MergePatches\t1".to_owned(),
            "ExcludeAdjacent\t1".to_owned(),
            "CircleObjects\t/".to_owned(),
            "SkipTurnedOffPoints\t1".to_owned(),
            "PolynomialOrder\t0".to_owned(),
        ];
        if let Some(half) = half_in_com {
            comlines.push(format!("HalfFloatModeOutput\t{half}"));
        }
        let prefix = [format!("{baseout}{axislet}"), "ccderaser".to_owned()];
        final_changes = vec![
            [
                &prefix[..],
                &["BetterRadius".to_owned(), format!("{:.2}", bead_size / 2.)],
            ]
            .concat(),
        ];
        if half_final {
            final_changes.push(
                [
                    &prefix[..],
                    &["HalfFloatModeOutput".to_owned(), half_floats.to_string()],
                ]
                .concat(),
            );
        }

    // TILT_3DFIND_REPROJECT
    } else if type_name == "tilt_3dfind_reproject" {
        let rootname = rootname_value();
        sedcom = vec![
            format!("/^OutputFile/s/[ \t].*/\t{rootname}_erase.fid/"),
            format!("/^OutputFile/a/ProjectModel\t{rootname}_3dfind.mod/"),
        ];

    // SIRTSETUP
    } else if type_name == "sirtsetup" {
        let name_style = match name_style {
            Some(value) => value,
            None => name_error("nameStyle"),
        };
        comlines = with_std(vec![
            format!("CommandFile\ttilt{axislet}{com_ext}"),
            "RadiusAndSigma\t0.4,0.035".to_owned(),
            "FalloffIsTrueSigma\t1".to_owned(),
            "StartFromZero".to_owned(),
            format!("NamingStyle {name_style}"),
        ]);

    // CTF3DSETUP
    } else if type_name == "ctf3dsetup" {
        let thickness = pip_get_integer("ThicknessToMake", 0).unwrap_or(0);
        if thickness == 0 {
            exit_error("Slab thickness must be entered");
        }
        comlines = with_std(vec![
            format!("TiltCommandFile {infile}"),
            format!("SlabThicknessInNm {thickness}"),
        ]);
        let num_procs = pip_get_integer("NumberOfProcessors", 0).unwrap_or(0);
        if num_procs != 0 {
            comlines.push(format!("NumberOfProcessors {num_procs}"));
        }

    // RESTRICTALIGN
    } else if type_name == "restrictalign" {
        let test_locals = pip_get_integer("LocalAlignValidation", 0).unwrap_or(0);
        let (target_ratio, min_ratio) =
            pip_get_two_floats("TargetAndMinRatios", (0., 0.)).unwrap_or((0., 0.));
        let skip_beam = pip_get_boolean("SkipBeamTiltWithOneRot", 0).unwrap_or(0);
        comlines = with_std(vec![
            format!("AlignCommandFile {infile}"),
            "UseCrossValidation 1".to_owned(),
            format!("LocalAlignValidation {test_locals}"),
        ]);
        if target_ratio > 0. {
            comlines.push(format!(
                "TargetMeasurementRatio {}",
                py_str_float(target_ratio)
            ));
        }
        if min_ratio > 0. {
            comlines.push(format!("MinMeasurementRatio {}", py_str_float(min_ratio)));
        }
        if skip_beam != 0 {
            comlines.push("SkipBeamTiltWithOneRot 1".to_owned());
        }

    // REDUCEFILTVOL
    } else if reduce_filt {
        let int_bin = binning_float.round_ties_even() as i64;
        // `binning` stays a float unless it is integral, when it becomes the
        // string fmtstr('{:.1f}', binning)
        let mut binning_text = py_str_float(binning_float);
        if (int_bin as f64 - binning_float).abs() <= 0.001 {
            binning_text = format!("{binning_float:.1}");
        }
        let mut out_name = dataset_filename(".red###filt", None, None);
        out_name = out_name.replace("###", &binning_text);
        comlines = with_std(vec![
            format!("InputFile  {infile}"),
            format!("OutputFile {out_name}"),
        ]);

    // EXTENDERASEMOD
    } else if type_name == "extenderasemod" {
        let replace = pip_get_float("ReplaceAboveAngle", -1.).unwrap_or(-1.);
        comlines = with_std(vec![format!("RootName {}", rootname_value())]);
        if replace >= 0. {
            comlines.push(format!("ReplaceAboveAngle {replace:.2}"));
        }
        if Path::new(&format!("autofidseed{axislet}{com_ext}")).exists() {
            let afs_lines =
                read_text_file(&format!("autofidseed{axislet}{com_ext}"), None, false, None)
                    .unwrap_or_default();
            let adjust_size = option_value(&afs_lines, "AdjustSizes", 3, false, 0, None, None);
            let mut asval = 0;
            if adjust_size == Some(OptionValue::Boolean(true)) {
                asval = 1;
            }
            comlines.push(format!("AdjustSizes {asval}"));
        }

    // VARYPATCHTRACK
    } else if type_name == "varypatchtrack" {
        comlines = with_std(vec![
            format!("XCorrCommandFile   xcorr_pt{axislet}{com_ext}"),
            format!("AlignCommandFile   align{axislet}{com_ext}"),
            "NumFilterSteps  8".to_owned(),
            "NumPatchSizeSteps  8".to_owned(),
        ]);
    }

    // Do common tasks and finish
    let mut sedlines = if !sedcom.is_empty() {
        pysed(&sedcom, PysedSrc::Lines(&comlines), None, false, '/', false)
            .ok()
            .flatten()
            .unwrap_or_default()
    } else {
        comlines
    };

    // Insert an output format if file type indicates it
    if need_style && let Some(type_ext) = type_ext.as_deref().filter(|ext| !ext.is_empty()) {
        add_output_format_var_to_lines(
            &mut sedlines,
            map_type_extension_to_style(type_ext, false),
            None,
            false,
        );
    }

    // `os.access(dir, os.W_OK)`: the POSIX `access` call itself
    let writable_dir = |directory: &str| -> bool {
        Path::new(directory).exists()
            && Path::new(directory).is_dir()
            && std::ffi::CString::new(directory)
                .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
                .unwrap_or(false)
    };

    if !base_changes.is_empty() || !final_changes.is_empty() || !change_list.is_empty() {
        // Make a default com file with directive and final changes left out - not clear about
        // whether final changes should be in
        if (!change_list.is_empty() || !final_changes.is_empty()) && writable_dir(dflt_com_dir) {
            let dfltlines =
                match modify_for_change_list(&sedlines, &baseout, axislet, &base_changes, false) {
                    Ok(lines) => lines,
                    Err(message) => exit_error(&message),
                };
            let _ = write_text_file(&format!("{dflt_com_dir}/{outfile}"), &dfltlines, false);
        }

        let mut all_changes = base_changes.clone();
        all_changes.extend(change_list.iter().cloned());
        all_changes.extend(final_changes.iter().cloned());
        sedlines = match modify_for_change_list(&sedlines, &baseout, axislet, &all_changes, false) {
            Ok(lines) => lines,
            Err(message) => exit_error(&message),
        };
    }

    make_backup_file(&outfile);
    let _ = write_text_file(&outfile, &sedlines, false);
    if writable_dir(orig_com_dir) {
        let _ = write_text_file(&format!("{orig_com_dir}/{outfile}"), &sedlines, false);
    }

    0
}
