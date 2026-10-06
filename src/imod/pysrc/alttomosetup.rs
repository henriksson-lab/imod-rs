//! Translation of `IMOD/pysrc/alttomosetup`: sets up command files for
//! processing alternate stacks (even/odd frame sums or another stack) through
//! the steps from aligned stack to tomogram, swapping the stacks in and out
//! with `swaptomostacks`.
//!
//! A Python command script; its functions are [`next_com_name`],
//! [`read_and_get_output`], [`make_trimvol_command_from_edf`] and
//! [`get_aligned_stack_size_from_program`], and its top level is
//! [`alttomosetup`].  `newstack`/`blendmont` (for the aligned stack size) and
//! `header` (through `getmrcsize`) are our own programs and run in process
//! through `imodpy::run_cmd`; `swaptomostacks`, `splitcorrection` and
//! `splittilt` are Python-script translations, which `run_cmd` runs as
//! children.  `montagesize` is one of ours too and `run_cmd` runs it in
//! process; its last line is still parsed from the printed text.

use super::imodpy::{
    INT_VALUE, OptionValue, STRING_VALUE, clean_chunk_files, cleanup_files, dataset_filename,
    exit_from_imod_error, extract_program_entries, find_root_axis_and_extensions, fmtstr,
    get_mrc_size, glob_glob, option_value, prnstr, py_int, py_int_floordiv, read_text_file,
    run_cmd, set_root_and_extension, write_finish_and_message, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_integer, pip_get_string, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed, sed_del_and_add};
use super::tomocoords::find_split_com_number;
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// Matches `nextComName` (`alttomosetup:12`): increments the number and
/// returns the next command file name, a sync file unless `sync` is false.
pub fn next_com_name(com_num: &mut i32, out_root: &str, sync: bool) -> String {
    *com_num += 1;
    if sync {
        format!("{out_root}-{:03}-sync.com", *com_num)
    } else {
        format!("{out_root}-{:03}.com", *com_num)
    }
}

/// Matches `readAndGetOutput` (`alttomosetup:24`): read a com file for
/// processing step and find the name of the output file.  `axis_ext` is the
/// script's global `axisExt`.
pub fn read_and_get_output(com_base: &str, out_opt: &str, axis_ext: &str) -> (Vec<String>, String) {
    let com_name = format!("{com_base}{axis_ext}");
    let lines = read_text_file(&com_name, None, false, None).unwrap_or_default();
    let out_name = match option_value(&lines, out_opt, STRING_VALUE, false, 0, None, None) {
        Some(OptionValue::String(value)) if !value.is_empty() => value,
        _ => exit_error(&format!("Cannot find output file name in {com_name}")),
    };
    (lines, out_name)
}

/// Matches `makeTrimvolCommandFromEDF` (`alttomosetup:37`): read an edf file
/// and find all the Trimvol-related tags needed to compose a trimvol command.
/// Make up such a command.  Return the nx, ny, nz of input to the trimmimg
/// and the command, or -1, -1, -1 and an error string.
///
/// `BUGS.md` (alttomosetup `makeTrimvolCommandFromEDF`), two fixes in
/// translation.  The source assigns a value to every tag that occurs
/// *anywhere* in the line, so the `...Trimvol.ScaleXMin=` line also sets
/// `XMin` (and likewise for the other `Scale` tags), and whichever of the two
/// lines comes later in the edf decides the `-x`/`-y` trimming; here a tag
/// takes the value only when the key before `=` is the tag or ends in `.`
/// and the tag.  And the fixed-scaling branch appends the literal text
/// `FixedScaleMax` (and spaces around the comma, which trimvol's `-c`
/// integer pair cannot parse) instead of the value; here it is
/// `-c <FixedScaleMin>,<FixedScaleMax>`.
pub fn make_trimvol_command_from_edf(rootname: &str) -> (i32, i32, i32, String) {
    let tags = [
        "XMin",
        "XMax",
        "YMin",
        "YMax",
        "ZMin",
        "ZMax",
        "ScaleXMin",
        "ScaleXMax",
        "ScaleYMin",
        "ScaleYMax",
        "SectionScaleMin",
        "SectionScaleMax",
        "TrimvolFlipped",
        "SwapYZ",
        "RotateX",
        "ConvertToBytes",
        "FixedScaling",
        "FixedScaleMin",
        "FixedScaleMax",
        "Input.NColumns",
        "Input.NRows",
        "Input.NSections",
    ];
    let opts = ["x", "y", "z", "sx", "sy", "sz"];
    let edf_name = format!("{rootname}.edf");
    if !Path::new(&edf_name).exists() {
        return (-1, -1, -1, format!("Etomo file {edf_name} dose not exist"));
    }
    let edf_lines = match read_text_file(&edf_name, Some("Etomo data file"), true, None) {
        Ok(lines) => lines,
        Err(message) => return (-1, -1, -1, message),
    };

    // Extract the values from the edf file
    let mut values: HashMap<&str, String> = HashMap::new();
    for line in &edf_lines {
        if line.contains("Trimvol") {
            for tag in tags {
                if let Some(ind) = line.find('=')
                    && ind > 0
                {
                    let key = &line[..ind];
                    if key == tag || key.ends_with(&format!(".{tag}")) {
                        values.insert(tag, line[ind + 1..].to_owned());
                    }
                }
            }
        }
    }
    let is_true = |tag: &str| values.get(tag).is_some_and(|value| value == "true");

    // Compose the command line
    let mut command = "$trimvol -f".to_owned();
    if is_true("TrimvolFlipped") {
        if is_true("RotateX") {
            command.push_str(" -rx");
        } else if is_true("SwapYZ") {
            command.push_str(" -yz");
        }
    }

    let mut num_pairs = 3;
    if is_true("ConvertToBytes") {
        if is_true("FixedScaling") {
            if let (Some(min), Some(max)) =
                (values.get("FixedScaleMin"), values.get("FixedScaleMax"))
            {
                command.push_str(&format!(" -c {min},{max}"));
            }
        } else {
            num_pairs = 6;
        }
    }

    for ind in 0..num_pairs {
        if let (Some(low), Some(high)) = (values.get(tags[2 * ind]), values.get(tags[2 * ind + 1]))
        {
            command.push_str(&fmtstr(
                " -{} {},{}",
                &[opts[ind].to_owned(), low.clone(), high.clone()],
            ));
        }
    }

    // `int(values[...])` in `try: ... except Exception: pass`
    let int_of = |tag: &str| {
        values
            .get(tag)
            .and_then(|value| py_int(value))
            .map_or(0, |value| value as i32)
    };
    let col = int_of("Input.NColumns");
    let row = int_of("Input.NRows");
    let sec = int_of("Input.NSections");

    (col, row, sec, command)
}

/// Matches `getAlignedStackSizeFromProgram` (`alttomosetup:110`): use the
/// PrintXYSizeAndExit option to newstack or blendmont to find out aligned
/// stack size.
pub fn get_aligned_stack_size_from_program(
    newst_exists: bool,
    newst_lines: &[String],
    progname: &str,
) -> (i32, i32) {
    // Extract the input lines and add the option
    let mut prog = "blendmont";
    if newst_exists {
        prog = "newstack";
    }
    let mut size_com = extract_program_entries(newst_lines, prog, "Standard").unwrap_or_default();
    if size_com.is_empty() {
        exit_error(&format!(
            "Could not find {prog} input lines in command file"
        ));
    }
    size_com.push("PrintXYSizeAndExit".to_owned());

    // Run the program
    let size_lines = match run_cmd(
        &format!("{prog} -Standard"),
        Some(&size_com),
        None,
        None,
        &[],
    ) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => {
            prnstr("Error trying to determine aligned stack size", "\n", false);
            exit_from_imod_error(progname);
        }
    };
    if size_lines.is_empty() {
        exit_error(
            "Empty output when running newstack or blendmont to determine aligned stack size",
        );
    }

    // Get the size
    let size_split: Vec<&str> = size_lines[size_lines.len() - 1]
        .split_whitespace()
        .collect();
    if size_split.len() != 4 || size_split[0] != "Output" || size_split[1] != "size:" {
        exit_error(
            "Unexpected output from running newstack or blendmont to determine aligned stack size",
        );
    }
    match (py_int(size_split[2]), py_int(size_split[3])) {
        (Some(nx_ali), Some(ny_ali)) => (nx_ali as i32, ny_ali as i32),
        _ => exit_error(
            "Converting aligned stack output file size from newstack or blendmont to integer",
        ),
    }
}

/// The script's top level (`alttomosetup:150-459`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn alttomosetup(arguments: &[OsString]) -> i32 {
    let progname = "alttomosetup";
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
        super::imodpy::add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 alttomosetup
    let options: Vec<String> = [
        "rootname:RootnameToProcess:CH:",
        "evenodd:EvenAndOddPairs:B:",
        "axis:AxisToProcess:CH:",
        "preproc:PreprocessForExtremes:I:",
        "ctf:CorrectCTF:B:",
        "erase:EraseFiducials:B:",
        "filter:FilterIn2D:B:",
        "trim:TrimVolume:B:",
        "clean:CleanUpIntermediates:B:",
        "procs:NumberOfProcessors:I:",
        "restore:JustRestoreInitialSet:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    // PIP startup and help
    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 1, 1);

    let chunks_per_proc = 3;

    // Get options
    let even_odd = pip_get_boolean("EvenAndOddPairs", 0).unwrap_or(0) != 0;
    let from_root = pip_get_string("RootnameToProcess", "").unwrap_or_default();
    if even_odd && !from_root.is_empty() {
        exit_error("You cannot enter both -evenodd and -rootname");
    }
    if !(even_odd || !from_root.is_empty()) {
        exit_error("You must enter either -evenodd or -rootname");
    }

    let pre_process = pip_get_integer("PreprocessForExtremes", 0).unwrap_or(0);
    let correct_ctf = pip_get_boolean("CorrectCTF", 0).unwrap_or(0) != 0;
    let erase_gold = pip_get_boolean("EraseFiducials", 0).unwrap_or(0) != 0;
    let filter_2d = pip_get_boolean("FilterIn2D", 0).unwrap_or(0) != 0;
    let do_trim = pip_get_boolean("TrimVolume", 0).unwrap_or(0) != 0;
    let num_procs = pip_get_integer("NumberOfProcessors", 0).unwrap_or(0);
    let clean_up = pip_get_boolean("CleanUpIntermediates", 0).unwrap_or(0) != 0;
    let do_axis = pip_get_string("AxisToProcess", "").unwrap_or_default();
    let just_restore = pip_get_boolean("JustRestoreInitialSet", 0).unwrap_or(0) != 0;
    if !do_axis.is_empty() && do_axis.to_uppercase() != "A" && do_axis.to_uppercase() != "B" {
        exit_error("Entry for -axis must be a, A, b, or B");
    }

    // Get properties of data set and insist they are all deducible
    let (com_ext, dual_num, root_name, type_ext, stack_ext) =
        find_root_axis_and_extensions(0, None);
    let single = dual_num != 2;
    if com_ext.is_empty()
        || dual_num < 0
        || root_name.is_empty()
        || stack_ext.is_empty()
        || type_ext.is_none()
    {
        exit_error(
            "There are non-standard files in the directory and not all of the data set can be determined",
        );
    }
    let type_ext = type_ext.unwrap_or_default();

    // Error checks
    if single && !do_axis.is_empty() {
        exit_error("This appears to be a single-axis data set, you cannot enter -axis");
    }
    if dual_num == 2 && do_trim {
        exit_error("This appears to be a dual-axis data set, you cannot enter -trim");
    }
    if dual_num == 2 && even_odd {
        exit_error("This appears to be a dual-axis data set and you cannot use -evenodd");
    }

    // Set up number of loops to run on the steps
    let mut num_loop = 1;
    if even_odd || (dual_num == 2 && do_axis.is_empty()) {
        num_loop = 2;
    }

    set_root_and_extension(&root_name, &type_ext);

    // Get the trimvol command if possible
    let (mut ncol, mut nrow, mut nsec, mut trim_command) = (0, 0, 0, String::new());
    if do_trim {
        (ncol, nrow, nsec, trim_command) = make_trimvol_command_from_edf(&root_name);
        if ncol < 0 {
            exit_error(&trim_command);
        }
    }

    let out_root = "alttomo";
    let mut com_num = 0;

    clean_chunk_files(out_root, false);
    let bound_list = glob_glob(&format!("{out_root}-bound-*.info"));
    if !bound_list.is_empty() {
        cleanup_files(&bound_list);
    }

    // Set up possible default axis letter and entry for -single option to swaptomostacks
    let mut axis_let = String::new();
    let mut single_opt = "";
    if !do_axis.is_empty() {
        axis_let = do_axis.to_lowercase();
        single_opt = "-single";
    }

    // Loop, set up root names for swapping and axis letter
    let mut use_procs = 0;
    for loop_index in 0..num_loop {
        let mut alt_root = from_root.clone();
        let mut set_root = root_name.clone();
        let mut to_root = format!("{set_root}_primts");
        if even_odd {
            alt_root = format!("{root_name}{}", ["_even", "_odd"][loop_index]);
        }
        if dual_num != 0 && do_axis.is_empty() {
            axis_let = ["a", "b"][loop_index].to_owned();
        }
        if !do_axis.is_empty() {
            set_root.push_str(&axis_let);
            alt_root.push_str(&axis_let);
            to_root.push_str(&axis_let);
        }
        let axis_ext = format!("{axis_let}.{com_ext}");
        let axis_root = format!("{root_name}{axis_let}");

        // Set up names for testing whether stacks are already swapped in
        let mut to_test = to_root.clone();
        let mut alt_test = alt_root.clone();
        if dual_num != 0 && do_axis.is_empty() {
            to_test.push_str(&axis_let);
            alt_test.push_str(&axis_let);
        }

        // Test if the _primts stack exists and the alt stack does not; if so swap back
        if Path::new(&format!("{to_test}.{stack_ext}")).exists()
            && !Path::new(&format!("{alt_test}.{stack_ext}")).exists()
        {
            if !just_restore {
                prnstr(
                    &format!(
                        "WARNING: altomosetup - the alternate stack {alt_test} is already swapped in; running swaptomostacks to restore files"
                    ),
                    "\n",
                    false,
                );
            } else {
                prnstr(
                    &format!(
                        "The alternate stack {alt_test} is swapped in; running swaptomostacks to restore files"
                    ),
                    "\n",
                    false,
                );
            }
            if run_cmd(
                &fmtstr(
                    "swaptomostacks {} -root \"{}\" -from \"{}\" -to \"{}\"",
                    &[
                        single_opt.to_owned(),
                        set_root.clone(),
                        to_root.clone(),
                        alt_root.clone(),
                    ],
                ),
                None,
                None,
                None,
                &[],
            )
            .is_err()
            {
                exit_from_imod_error(progname);
            }
        }

        if just_restore {
            if even_odd {
                continue;
            }
            return done(0);
        }

        // Get newstack or blendmont lines for aligned stack
        let newst_com = format!("newst{axis_ext}");
        let blend_com = format!("blend{axis_ext}");
        let newst_exists = Path::new(&newst_com).exists();
        let blend_exists = Path::new(&blend_com).exists();
        if !(newst_exists || blend_exists) {
            exit_error(&format!("Neither {newst_com} nor {blend_com} exists"));
        }
        if newst_exists && blend_exists {
            exit_error(&format!(
                "Both {newst_com} and {blend_com} exist, cannot tell which to use"
            ));
        }
        let newst_lines = if newst_exists {
            read_text_file(&newst_com, None, false, None).unwrap_or_default()
        } else {
            read_text_file(&blend_com, None, false, None).unwrap_or_default()
        };

        // Get tilt.com and its input and output, detect GPU and set up default # of procs
        let tilt_name = format!("tilt{axis_ext}");
        let tilt_lines = read_text_file(&tilt_name, None, false, None).unwrap_or_default();
        let string_of =
            |option: &str| match option_value(&tilt_lines, option, 0, true, 0, None, None) {
                Some(OptionValue::String(value)) => value,
                _ => String::new(),
            };
        let rec_name = string_of("outputfile");
        let ali_name = string_of("inputproj");
        if rec_name.is_empty() || ali_name.is_empty() {
            exit_error(&format!(
                "Cannot find name of input file or output file in {tilt_name}"
            ));
        }

        let use_gpu = match option_value(&tilt_lines, "UseGPU", INT_VALUE, true, 1, None, None) {
            Some(OptionValue::Integers(values)) => values[0],
            _ => -1,
        };
        use_procs = num_procs;
        if use_procs == 0 {
            use_procs = 8;
            if use_gpu >= 0 {
                use_procs = 1;
            }
        }

        // When more than one processing unit, Splittilt will need size
        let (mut nx_ali, mut ny_ali) = (0, 0);
        if use_procs > 1 {
            (nx_ali, ny_ali) =
                get_aligned_stack_size_from_program(newst_exists, &newst_lines, progname);
        }

        // If trimming, get size of current rec file
        if do_trim && let Ok((nx, ny, nz)) = get_mrc_size(&rec_name) {
            if (ncol != 0 && nx != ncol) || (nrow != 0 && ny != nrow) || (nsec != 0 && nz != nsec) {
                exit_error(&fmtstr(
                    "Size of current output from Tilt ({}x{}x{}) does not match size when Trimvol was run ({}x{}x{})",
                    &[
                        nx.to_string(),
                        ny.to_string(),
                        nz.to_string(),
                        ncol.to_string(),
                        nrow.to_string(),
                        nsec.to_string(),
                    ],
                ));
            }
        }

        // Set up the swap on first loop or both if even/odd
        let mut com_lines: Vec<String> = Vec::new();
        if loop_index == 0 || even_odd {
            com_lines.push(fmtstr(
                "$swaptomostacks {} -check -root {} -from {} -to {}",
                &[
                    single_opt.to_owned(),
                    set_root.clone(),
                    alt_root.clone(),
                    to_root.clone(),
                ],
            ));
        }

        // Eraser and archive
        if pre_process != 0 {
            let xray_lines =
                read_text_file(&format!("eraser{axis_ext}"), None, false, None).unwrap_or_default();
            com_lines.extend(xray_lines.iter().cloned());
            let mut erase_out = match option_value(
                &xray_lines,
                "OutputFile",
                STRING_VALUE,
                false,
                0,
                None,
                None,
            ) {
                Some(OptionValue::String(value)) => value,
                _ => String::new(),
            };
            let stack_name = format!("{axis_root}.{stack_ext}");
            if erase_out.is_empty() {
                erase_out = format!("{axis_root}_fixed.{stack_ext}");
            }
            com_lines.push(format!(
                "$b3drename {stack_name} {axis_root}_orig.{stack_ext}"
            ));
            com_lines.push(format!("$b3drename {erase_out} {axis_root}.{stack_ext}"));
            if pre_process > 1 {
                com_lines.push(format!("$archiveorig {stack_name}"));
            }
        }

        com_lines.extend(newst_lines.iter().cloned());
        let _ = write_text_file(
            &next_com_name(&mut com_num, out_root, true),
            &com_lines,
            false,
        );
        com_lines = Vec::new();

        // CTF correction
        if correct_ctf {
            let (ctf_lines, ctf_out) =
                read_and_get_output("ctfcorrection", "OutputFileName", &axis_ext);
            let ctf_mod = pysed(
                &sed_del_and_add("UseGPU", &use_gpu.to_string(), "DefocusFile", '/'),
                PysedSrc::Lines(&ctf_lines),
                None,
                false,
                '/',
                false,
            )
            .ok()
            .flatten()
            .unwrap_or_default();
            if use_procs > 1 {
                let main_stack = format!("{axis_root}.{stack_ext}");
                let nz_ali;
                if blend_exists {
                    let mut mscom = fmtstr("montagesize \"{}\"", &[main_stack.clone()]);
                    if Path::new(&format!("{axis_root}.pl")).exists() {
                        mscom.push_str(&format!(" \"{axis_root}.pl\""));
                    }
                    let mont_size_lines = match run_cmd(&mscom, None, None, None, &[]) {
                        Ok(lines) => lines.unwrap_or_default(),
                        Err(_) => exit_from_imod_error(progname),
                    };
                    // `BUGS.md` (alttomosetup `montagesize`), fixed in
                    // translation: with no output the source's IndexError
                    // handler indexes the empty list again and dies with a
                    // traceback; the handler's message is given with an
                    // empty last line instead.
                    let last = mont_size_lines.last().cloned().unwrap_or_default();
                    let mont_split: Vec<&str> = last.split_whitespace().collect();
                    nz_ali = match mont_split.last().and_then(|text| py_int(text)) {
                        Some(value) => value as i32,
                        None => exit_error(&format!(
                            "Getting NZ from last value in montagesize output: {last}"
                        )),
                    };
                } else {
                    nz_ali = match get_mrc_size(&main_stack) {
                        Ok((_nx_raw, _ny_raw, nz)) => nz,
                        Err(_) => exit_from_imod_error(progname),
                    };
                }

                // Split correction if multiple procs
                let num_chunks = use_procs * chunks_per_proc;
                let max_slices = 1.max(py_int_floordiv(
                    (nz_ali + num_chunks - 1) as i64,
                    num_chunks as i64,
                ));

                let temp_name = format!("ctfcorrection_tmp{axis_ext}");
                let _ = write_text_file(&temp_name, &ctf_mod, false);
                let split_lines = match run_cmd(
                    &fmtstr(
                        "splitcorrection -i {} -o -m {} -uni -size {},{},{} -r {} \"{}\"",
                        &[
                            (com_num + 1).to_string(),
                            max_slices.to_string(),
                            nx_ali.to_string(),
                            ny_ali.to_string(),
                            nz_ali.to_string(),
                            out_root.to_owned(),
                            temp_name.clone(),
                        ],
                    ),
                    None,
                    None,
                    None,
                    &[],
                ) {
                    Ok(lines) => lines.unwrap_or_default(),
                    Err(_) => exit_from_imod_error(progname),
                };

                let num_added = find_split_com_number(&split_lines, "output of splitcorrection");
                com_num += num_added;
                cleanup_files(&[temp_name]);
            } else {
                // Or just copy the com lines
                com_lines = ctf_mod;
            }

            com_lines.push(format!("$b3drename {ctf_out} {ali_name}"));
        }

        // Gold erasing
        if erase_gold {
            let (gold_lines, erase_out) =
                read_and_get_output("golderaser", "OutputFile", &axis_ext);
            com_lines.extend(gold_lines);
            com_lines.push(format!("$b3drename {erase_out} {ali_name}"));
        }

        // 2D filtering
        if filter_2d {
            let (filt_lines, filt_out) = read_and_get_output("mtffilter", "OutputFile", &axis_ext);
            com_lines.extend(filt_lines);
            com_lines.push(format!("$b3drename {filt_out} {ali_name}"));
        }

        if !com_lines.is_empty() {
            let _ = write_text_file(
                &next_com_name(&mut com_num, out_root, true),
                &com_lines,
                false,
            );
            com_lines = Vec::new();
        }

        // Tilt!  Split up if multiple procs, or use the lines as is
        if use_procs > 1 {
            let split_com = vec![
                format!("CommandFile  {tilt_name}"),
                format!("RootNameOfOutput  {out_root}"),
                format!("ProcessorNumber  {use_procs}"),
                format!("TargetChunks  {}", use_procs * chunks_per_proc),
                format!("InitialComNumber  {}", com_num + 1),
                "OpenForMoreComs  1".to_owned(),
                "UniqueInfoFile  1".to_owned(),
                format!("DimensionsOfStack {nx_ali},{ny_ali}"),
            ];
            let split_lines = match run_cmd(
                "splittilt -StandardInput",
                Some(&split_com),
                None,
                None,
                &[],
            ) {
                Ok(lines) => lines.unwrap_or_default(),
                Err(_) => exit_from_imod_error(progname),
            };

            let num_added = find_split_com_number(&split_lines, "output of splittilt");
            com_num += num_added;
        } else {
            com_lines.extend(tilt_lines.iter().cloned());
        }

        // Trim
        if do_trim {
            let trim_name = dataset_filename(".rec", Some(&axis_root), None);
            com_lines.push(format!("{trim_command} {rec_name} {trim_name}"));
        }

        // Cleanup
        if clean_up {
            com_lines.push(format!("$b3dremove {ali_name}"));
            if do_trim {
                com_lines.push(format!("$b3dremove {rec_name}"));
            }
        }

        // Swap back at end or for each loop of even/odd
        if even_odd || loop_index == num_loop - 1 {
            com_lines.push(fmtstr(
                "$swaptomostacks {} -root {} -from {} -to {}",
                &[
                    single_opt.to_owned(),
                    set_root.clone(),
                    to_root.clone(),
                    alt_root.clone(),
                ],
            ));
        }

        if !com_lines.is_empty() {
            let _ = write_text_file(
                &next_com_name(&mut com_num, out_root, true),
                &com_lines,
                false,
            );
        }
    }

    if !just_restore {
        write_finish_and_message(
            Some(&mut Vec::new()),
            out_root,
            com_num,
            use_procs < 2,
            ".com",
        );
    }

    done(0)
}
