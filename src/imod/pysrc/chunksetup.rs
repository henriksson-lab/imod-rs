//! Translation of `IMOD/pysrc/chunksetup`.
//!
//! A Python command script with no functions; its top level is
//! [`chunksetup`], translated statement by statement.  `tomopieces` is not a
//! command of this crate, so `runcmd` runs it as a process.

use super::imodpy::{
    add_imod_bin_ignore_sighup, add_output_format_var_to_lines, call_own_program,
    clean_chunk_files, default_com_extension, exit_from_imod_error, map_type_extension_to_style,
    os_path_splitext, prnstr, read_text_file, run_cmd, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_integer, pip_get_non_option_arg,
    pip_get_string, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed};
use crate::imod::flib::image::tomopieces::{tomopieces_output_lines, tomopieces_recording};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`chunksetup:1-283`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn chunksetup(arguments: &[OsString]) -> i32 {
    let progname = "chunksetup";
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

    // Initializations - defaults for filename identifiers and tomopieces call
    let masterin = "INPUTFILE";
    let masterout = "OUTPUTFILE";
    let mut mega = String::new();
    let mut nofft = String::new();

    // Fallbacks from ../manpages/autodoc2man 3 1 chunksetup
    let options: Vec<String> = [
        "master:MasterComFile:FN:",
        "command:OneLineCommand:CH:",
        "input:InputImageFile:FN:",
        "output:OutputImageFile:FN:",
        "suffix:SuffixForOutputName:CH:",
        "format:FormatOfOutputFile:CH:",
        "p:PaddingPixels:I:",
        "o:OverlapPixels:I:",
        "m:MegavoxelMaximum:I:",
        "xm:XMaximumPieces:I:",
        "ym:YMaximumPieces:I:",
        "no:NoFFTSizes:B:",
        "param:ParameterFile:PF:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, nonopts) = pip_read_or_parse_options(&argv, &options, progname, 3, 2, 1);

    // Get file names
    let mut in_ind = 0;

    let mut mastercom = pip_get_string("MasterComFile", "").unwrap_or_default();
    let command = pip_get_string("OneLineCommand", "").unwrap_or_default();
    let mut outfile = pip_get_string("OutputImageFile", "").unwrap_or_default();
    let suffix = pip_get_string("SuffixForOutputName", "").unwrap_or_default();
    let mut infile = pip_get_string("InputImageFile", "").unwrap_or_default();
    if !mastercom.is_empty() && !command.is_empty() {
        exit_error("You cannot enter both a command file and a one-line command");
    }
    if !outfile.is_empty() && !suffix.is_empty() {
        exit_error(
            &("You cannot enter both an output file name and a suffix for making the".to_owned()
                + " name"),
        );
    }
    let mut num_allowed = 3;
    if !mastercom.is_empty() || !command.is_empty() {
        num_allowed -= 1;
    }
    if !outfile.is_empty() || !suffix.is_empty() {
        num_allowed -= 1;
    }
    if !infile.is_empty() {
        num_allowed -= 1;
    }
    if num_allowed != nonopts {
        exit_error(&format!(
            "There are {nonopts} non-option arguments and there must be {num_allowed} with the options entered"
        ));
    }

    if mastercom.is_empty() && command.is_empty() {
        mastercom = pip_get_non_option_arg(0).unwrap_or_default();
        in_ind = 1;
    }

    let mut out_ind = in_ind;
    if infile.is_empty() {
        infile = pip_get_non_option_arg(in_ind).unwrap_or_default();
        out_ind = in_ind + 1;
    }
    if infile.is_empty() {
        exit_error("Image input file must be entered");
    }

    if outfile.is_empty() && suffix.is_empty() {
        outfile = pip_get_non_option_arg(out_ind).unwrap_or_default();
    }

    // Figure out final output format
    let mut name_style = 1;
    let mut out_format = std::env::var("IMOD_OUTPUT_FORMAT").ok();
    let mut oformat = pip_get_string("FormatOfOutputFile", "").unwrap_or_default();
    let allowed = ["HDF", "MRC", "TIFF", "TIF"];
    if !oformat.is_empty() {
        if !allowed.contains(&oformat.to_uppercase().as_str()) {
            exit_error(&format!("Output format {oformat} is not a valid format"));
        }
        if oformat.to_uppercase() == "TIFF" {
            oformat = "TIF".to_owned();
        }
        out_format = Some(oformat.clone());
    }

    let out_format = match out_format.filter(|format| !format.is_empty()) {
        Some(format) => {
            let mut format = format.to_lowercase();
            if format == "tiff" {
                format = "tif".to_owned();
            }
            name_style = map_type_extension_to_style(&format, true);
            if name_style == 0 {
                name_style = 1;
            }
            format
        }
        None => "mrc".to_owned(),
    };

    // Compose output name
    if !suffix.is_empty() {
        let (mut out_root, mut out_ext) = os_path_splitext(&infile);
        if out_ext.starts_with('.') {
            out_ext = out_ext[1..].to_owned();
        }
        if !allowed.contains(&out_ext.to_uppercase().as_str()) {
            out_root += &format!("_{out_ext}");
        }
        outfile = format!("{out_root}_{suffix}.{out_format}");
    }

    // Any names that are going to be split for directory need to have forward slashes
    let fullroot;
    let rootname;
    let mut com_ext;
    if !mastercom.is_empty() {
        mastercom = mastercom.replace('\\', "/");
        (fullroot, com_ext) = os_path_splitext(&mastercom);
        if com_ext != ".com" && com_ext != ".pcm" {
            com_ext = format!(".{}", default_com_extension());
        }
        // `os.path.basename`
        rootname = fullroot[fullroot.rfind('/').map_or(0, |index| index + 1)..].to_owned();
    } else {
        let com_split = command.split_whitespace().collect::<Vec<_>>();
        fullroot = format!("{}-cs", com_split[0]);
        rootname = fullroot.clone();
        com_ext = format!(".{}", default_com_extension());
        mastercom = format!("{fullroot}{com_ext}");
    }

    let allname = format!("{fullroot}-all{com_ext}");
    let finishname = format!("{fullroot}-finish{com_ext}");

    // The input and output files may be relative to the location of the command
    // file which need not be in current dir
    // `os.path.dirname`
    let mut masthead = mastercom[..mastercom.rfind('/').map_or(0, |index| index + 1)].to_owned();
    if !masthead.is_empty() && masthead.chars().any(|character| character != '/') {
        masthead = masthead.trim_end_matches('/').to_owned();
    }
    let mut infromhere = infile.clone();
    if !masthead.is_empty() {
        // `os.path.join(masthead, infile)`
        infromhere = if infile.starts_with('/') {
            infile.clone()
        } else if masthead.ends_with('/') {
            format!("{masthead}{infile}")
        } else {
            format!("{masthead}/{infile}")
        };
    }

    if command.is_empty() && !Path::new(&mastercom).exists() {
        exit_error(&format!("Command file {mastercom} does not exist"));
    }
    if !Path::new(&infromhere).exists() {
        exit_error(&format!("Image file file {infile} does not exist"));
    }

    // Get rest of arguments
    let pad = pip_get_integer("PaddingPixels", 8).unwrap_or(8);
    let overin = pip_get_integer("OverlapPixels", 8).unwrap_or(8);
    let overlap = format!("-min {overin}");
    let megain = pip_get_integer("MegavoxelMaximum", 0).unwrap_or(0);
    if pip_get_err_no() == 0 {
        mega = format!("-mega {megain}");
    }
    let maxxp = pip_get_integer("XMaximumPieces", -1).unwrap_or(-1);
    let maxyp = pip_get_integer("YMaximumPieces", -1).unwrap_or(-1);
    let nof = pip_get_boolean("NoFFTSizes", 0).unwrap_or(0);
    if nof != 0 {
        nofft = "-nofft".to_owned();
    }

    let master_lines = if command.is_empty() {
        read_text_file(&mastercom, Some("master command file"), false, None).unwrap_or_default()
    } else {
        vec![format!("${command} \"{masterin}\"  \"{masterout}\"")]
    };

    let mut testin = false;
    let mut testout = false;
    let mut broke = false;
    for l in &master_lines {
        if l.contains(masterin) {
            testin = true;
        }
        if l.contains(masterout) {
            testout = true;
        }
        if testout && testin {
            broke = true;
            break;
        }
    }
    if !broke {
        exit_error(&format!(
            "The master command file does not contain both {masterin} and {masterout}"
        ));
    }

    // Any filenames passed in a command line that could have a path must be quoted
    let tomopcom = format!(
        "tomopieces -tomo \"{infromhere}\" {mega} {nofft} -xp {pad} -yp {pad} -zp {pad} {overlap} -xmax {maxxp} -ymax {maxyp}"
    );
    // Direct call (owner rule, 2026-09-26): tomopieces returns its ranges and
    // `ranlist` is built from them rather than from its captured output
    // (`chunksetup:196`).  The values are integers the script copies into the
    // command files as text, so the lines are exactly what it printed.
    // Tomopieces still reads its options through PIP from the words of
    // `tomopcom`.
    let tail = format!(
        "{mega} {nofft} -xp {pad} -yp {pad} -zp {pad} {overlap} -xmax {maxxp} -ymax {maxyp}"
    );
    let mut words: Vec<&str> = vec!["tomopieces", "-tomo", &infromhere];
    words.extend(tail.split_whitespace());
    let sink = std::sync::Arc::new(std::sync::Mutex::new(None));
    let recorder = std::sync::Arc::clone(&sink);
    let ranlist = match call_own_program(&tomopcom, &words, None, true, move || {
        tomopieces_recording(recorder)
    }) {
        Ok(_) => match sink.lock().expect("tomopieces result").take() {
            Some(result) => tomopieces_output_lines(&result),
            None => Vec::new(),
        },
        Err(_) => exit_from_imod_error(progname),
    };

    // Remove any previous files now in case the number has changed
    clean_chunk_files(&fullroot, false);

    // Get the number of pieces in X/Y/Z
    if ranlist.len() < 3 {
        exit_error("First line of output from tomopieces does not have correct form");
    }
    let npl = ranlist[0].split_whitespace().collect::<Vec<_>>();
    if npl.len() < 3 {
        exit_error("First line of output from tomopieces does not have correct form");
    }
    let (npiecex, npiecey, npiecez) = match (
        npl[0].parse::<i32>(),
        npl[1].parse::<i32>(),
        npl[2].parse::<i32>(),
    ) {
        (Ok(x), Ok(y), Ok(z)) => (x, y, z),
        _ => exit_error("First line of output from tomopieces does not have correct form"),
    };
    let npiecetot = npiecex * npiecey * npiecez;
    if ranlist.len() as i32 != 1 + npiecetot + npiecex + npiecey + npiecez {
        exit_error("Output from tomopieces does not have correct number of lines");
    }

    // Make the chunk coms
    let mut allines = vec![
        "# THIS IS A COMMAND FILE TO RUN ALL THE PIECES AND PUT THEM TOGETHER".to_owned(),
        "#".to_owned(),
    ];
    for ipc in 0..npiecetot {
        let numtext = format!("-{:03}", ipc + 1);
        let minmaxes = ranlist[(ipc + 1) as usize]
            .trim()
            .split(',')
            .collect::<Vec<_>>();
        let comfile = format!("{fullroot}{numtext}{com_ext}");
        let imagein = format!("{rootname}{numtext}.in");
        let imageout = format!("{rootname}{numtext}.out");
        let mut comlines = vec![
            "$taperoutvol -StandardInput".to_owned(),
            format!("InputFile {infile}"),
            format!("OutputFile {imagein}"),
            format!("XMinAndMax {},{}", minmaxes[0], minmaxes[1]),
            format!("YMinAndMax {},{}", minmaxes[2], minmaxes[3]),
            format!("ZMinAndMax {},{}", minmaxes[4], minmaxes[5]),
            format!("TaperPadsInXYZ {pad},{pad},{pad}"),
        ];
        if !nofft.is_empty() {
            comlines.push("NoFFTSizes".to_owned());
        }

        let sed = vec![
            format!("/{masterin}/s//{imagein}/g"),
            format!("/{masterout}/s//{imageout}/g"),
        ];
        comlines.extend(
            pysed(
                &sed,
                PysedSrc::Lines(&master_lines),
                None,
                false,
                '/',
                false,
            )
            .ok()
            .flatten()
            .unwrap_or_default(),
        );
        comlines.extend([String::new(), format!("$b3dremove {imagein}")]);

        let _ = write_text_file(&comfile, &comlines, false);

        allines.extend([
            format!("$echo Working on piece  {}  of {}", ipc + 1, npiecetot),
            format!("$vmstopy -x {comfile} {rootname}{numtext}.log"),
        ]);
    }

    // Make the reassembly file
    let mut finishlines = vec![
        "# THIS COMMAND FILE REASSEMBLES THE PIECES".to_owned(),
        "#".to_owned(),
        "$assemblevol -StandardInput".to_owned(),
        format!("OutputFile {outfile}"),
    ];
    add_output_format_var_to_lines(&mut finishlines, name_style, Some(&out_format), true);
    let mut ranind = (npiecetot + 1) as usize;
    for _ipc in 0..npiecex {
        finishlines.push(format!("StartEndToExtractInX {}", ranlist[ranind].trim()));
        ranind += 1;
    }
    for _ipc in 0..npiecey {
        finishlines.push(format!("StartEndToExtractInY {}", ranlist[ranind].trim()));
        ranind += 1;
    }
    for _ipc in 0..npiecez {
        finishlines.push(format!("StartEndToExtractInZ {}", ranlist[ranind].trim()));
        ranind += 1;
    }

    for ipc in 0..npiecetot {
        let numtext = format!("-{:03}", ipc + 1);
        finishlines.push(format!("InputFile {rootname}{numtext}.out"));
    }

    finishlines.push(format!(
        "$b3dremove -g {rootname}-[0-9][0-9][0-9]*{com_ext}* {rootname}-[0-9][0-9][0-9]*.log* {rootname}-[0-9][0-9][0-9]*.out*"
    ));
    allines.extend([
        "$echo Reassembling pieces".to_owned(),
        format!("$vmstopy -x {finishname} {rootname}-finish.log"),
    ]);

    let _ = write_text_file(&finishname, &finishlines, false);
    let _ = write_text_file(&allname, &allines, false);

    if !suffix.is_empty() {
        prnstr(&format!("Output file: {outfile}    [CHS1]"), "\n", false);
        prnstr("", "\n", false);
    }

    // `os.path.basename(allname)`
    let all_base = &allname[allname.rfind('/').map_or(0, |index| index + 1)..];
    prnstr(
        &format!(
            "{npiecetot} command files were generated and are ready to run.

You can run {all_base} (with subm) to run all command files in 
sequence, reassemble the volume, and clean up intermediate image and 
command files.

If you have multiple processors available, you can use 
  processchunks machine_list {rootname}
to do these operations in parallel, where machine_list is a comma-separated 
list of available machines or just the number of local processors"
        ),
        "\n",
        false,
    );

    0
}
