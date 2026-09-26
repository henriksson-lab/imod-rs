//! Translation of `IMOD/pysrc/splitcombine`.
//!
//! A Python command script: its one function is [`warning`], and its top
//! level is [`splitcombine`], translated statement by statement.

use super::imodpy::{
    OptionValue, add_imod_bin_ignore_sighup, clean_chunk_files, complete_and_check_com_file,
    option_value, prnstr, read_text_file, write_text_file,
};
use super::pip::{
    exit_error, pip_get_err_no, pip_get_in_out_file, pip_get_string, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed};
use regex::Regex;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// Matches `warning` (`splitcombine:11`): a blank line and the warning on
/// standard error.
pub fn warning(text: &str) {
    let _ = std::io::stdout().flush();
    eprint!(" \n");
    eprint!("WARNING: {text}\n\n");
}

/// The script's top level (`splitcombine:15-192`).  Returns the status of its
/// final `sys.exit(0)`; error paths exit the process as `exitError` does.
pub fn splitcombine(arguments: &[OsString]) -> i32 {
    let progname = "splitcombine";
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

    // Fallbacks from ../manpages/autodoc2man 3 1 splitcombine
    let options: Vec<String> = [
        "comfile:CommandFile:FN:",
        "tempdir:TemporaryDirectory:FN:",
        "local:LocalTempPath:FN:",
        "global:GlobalTempPath:FN:",
        "help:Usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 0, 0, 0);

    let comfile = pip_get_in_out_file("CommandFile", 0)
        .ok()
        .flatten()
        .unwrap_or_else(|| "volcombine".to_owned());
    let (comfile, rootname) = complete_and_check_com_file(&comfile);
    let com_ext = comfile
        .chars()
        .skip(comfile.chars().count().saturating_sub(4))
        .collect::<String>();

    let tempdir = pip_get_string("TemporaryDirectory", "").unwrap_or_default();

    let mut local = pip_get_string("LocalTempPath", "gibberish").unwrap_or_default();
    let iflocal = 1 - pip_get_err_no();
    let mut globdir = pip_get_string("GlobalTempPath", "reallyjunk").unwrap_or_default();
    if iflocal + 1 - pip_get_err_no() == 1 {
        exit_error("If you enter one of -local and -global, you must enter both");
    }

    // Escape both forward and backward slashes since this is going into pysedcd
    if iflocal != 0 {
        local = local.replace('\\', "\\\\");
        local = local.replace('/', "\\/");
        globdir = globdir.replace('\\', "\\\\");
        globdir = globdir.replace('/', "\\/");
    }

    // Read command file
    let comlines = read_text_file(&comfile, None, false, None).unwrap_or_default();
    let mut option_line1: Option<String> = None;
    let mut option_line2: Option<String> = None;
    let mut option_line3: Option<String> = None;
    let mut option_line4: Option<String> = None;
    let mut sect_starts: Vec<usize> = Vec::new();
    let mut got_assemble = false;
    let opt1_match = Regex::new("set *combinefft_red.*=").unwrap();
    let opt2_match = Regex::new("set *combinefft_low.*=").unwrap();
    let mut using_tmp = false;
    let mut using_usr_tmp = false;
    let mut got_init_chunk = false;
    let mut lock_name = String::new();

    for ln in 0..comlines.len() {
        let line = &comlines[ln];
        if !got_assemble && line.contains("COMBINING PIECE") {
            sect_starts.push(ln);
        }
        if !got_assemble && line.contains("ASSEMBLING") {
            sect_starts.push(ln);
            got_assemble = true;
        }
        if option_line1.as_deref().is_none_or(str::is_empty) && opt1_match.is_match(line) {
            option_line1 = Some(line.clone());
        }
        if option_line2.as_deref().is_none_or(str::is_empty) && opt2_match.is_match(line) {
            option_line2 = Some(line.clone());
        }
        if option_line3.as_deref().is_none_or(str::is_empty)
            && line.contains("setenv IMOD_BRIEF_HEADER")
        {
            option_line3 = Some(line.clone());
        }
        // Fixed in translation (BUGS.md): `splitcombine:91` tests optionLine3
        // here, so native keeps the last IMOD_OUTPUT_FORMAT line before the
        // IMOD_BRIEF_HEADER one (or none after it); this keeps the first.
        if option_line4.as_deref().is_none_or(str::is_empty)
            && line.contains("setenv IMOD_OUTPUT_FORMAT")
        {
            option_line4 = Some(line.clone());
        }
        if !using_usr_tmp && line.contains("/usr/tmp") {
            using_usr_tmp = true;
        }
        if !using_tmp && line.contains("/tmp") {
            using_tmp = true;
        }
        if !got_init_chunk && line.contains("INITIALIZING CHUNKED") {
            got_init_chunk = true;
        }
    }

    let num_chunks = sect_starts.len() as i64 - 1;
    if num_chunks < 1 || !got_assemble {
        exit_error("The command file is missing chunks or the assemblevol section");
    }
    let num_chunks = num_chunks as usize;

    if using_usr_tmp {
        warning("This command file accesses /usr/tmp and will not run on multiple machines");
    } else if using_tmp {
        warning("This command file accesses /tmp and may not run on multiple machines");
    }

    // Try to extract the master temporary directory from the first chunk
    let first_chunk = &comlines[sect_starts[0]..sect_starts[1]];
    let string_value = |option: &str| -> Option<String> {
        match option_value(first_chunk, option, 0, false, 0, None, None) {
            Some(OptionValue::String(value)) => Some(value),
            _ => None,
        }
    };
    let input_ffta = string_value("AInputFFT");
    let input_fftb = string_value("BInputFFT");
    let output_fft = string_value("OutputFFT");
    let fft_opts = [&input_ffta, &input_fftb, &output_fft];
    let mut sumdir = String::new();
    for line in fft_opts {
        if let Some(line) = line.as_deref().filter(|line| !line.is_empty()) {
            let line = line.replace('\\', "/");
            // `os.path.dirname`: through the last separator, with trailing
            // separators removed unless that is all there is
            let index = line.rfind('/').map_or(0, |index| index + 1);
            let mut head = line[..index].to_owned();
            if !head.is_empty() && head.chars().any(|c| c != '/') {
                head = head.trim_end_matches('/').to_owned();
            }
            sumdir = head;
            if !sumdir.is_empty() {
                break;
            }
        }
    }

    if !sumdir.is_empty() {
        // `os.access(sumdir, os.W_OK)`: the POSIX `access` call itself
        let writable = std::ffi::CString::new(sumdir.as_bytes())
            .map(|path| unsafe { libc::access(path.as_ptr(), libc::W_OK) } == 0)
            .unwrap_or(false);
        if !Path::new(&sumdir).is_dir() || !writable {
            exit_error(&format!("Unable to write sum*.rec to directory {sumdir}"));
        }
    }

    let tmprec = "$tmpdir\\/rec.";
    let tmpmat = "$tmpdir\\/mat.";

    // Remove any previous files now in case the number has changed
    clean_chunk_files(&rootname, false);

    let mut localcom = vec![format!("/{local}/s//{globdir}/g")];
    let mut sumname = String::new();
    if got_init_chunk {
        let Some(output_fft) = output_fft.as_deref().filter(|name| !name.is_empty()) else {
            exit_error("Cannot get name of output file for combine");
        };
        // `os.path.basename`
        sumname = output_fft[output_fft.rfind('/').map_or(0, |index| index + 1)..].to_owned();
        lock_name = format!("{sumname}.lock");
        localcom.push(format!("/INITIALIZING CHUNKED/a/$b3dtouch {lock_name}/"));
        localcom.push(format!("/TaperPadsInXYZ/a/LockFileForHDF  {lock_name}/"));
    }
    let _ = pysed(
        &localcom,
        PysedSrc::Lines(&comlines[0..sect_starts[0]]),
        Some(&format!("{rootname}-start{com_ext}")),
        false,
        '/',
        false,
    );

    for num in 1..num_chunks + 1 {
        let comname = format!("{rootname}-{num:03}{com_ext}");
        let mut outlines = vec!["$set tmpext = `hostname`.$$".to_owned()];
        if !tempdir.is_empty() {
            outlines.push(format!("$set tmpdir = \"{tempdir}\""));
        } else {
            outlines.extend([
                "$set tmpdir = /usr/tmp".to_owned(),
                "$if ($?IMOD_DIR) then".to_owned(),
                "$if (-e \"$IMOD_DIR/bin/settmpdir\") source \"$IMOD_DIR/bin/settmpdir\""
                    .to_owned(),
                "$endif".to_owned(),
            ]);
        }
        for option_line in [&option_line1, &option_line2, &option_line3, &option_line4] {
            if let Some(line) = option_line.as_deref().filter(|line| !line.is_empty()) {
                outlines.push(line.to_owned());
            }
        }
        outlines.extend_from_slice(&comlines[sect_starts[num - 1]..sect_starts[num]]);

        // doctor the filenames.  Need to replace all leading paths before rec. and
        // mat. to get rid of temporary directory.  Match all back to space or tab
        // But need to put escapes in front of the $tmpdir entries at start of line
        // This is all for backward compatibility.  Had to match .st and .fft explicitly
        // to avoid matching new style filenames
        let mut sedcom = localcom.clone();
        sedcom.extend([
            "/STATUS:/d".to_owned(),
            r"/rec\.st/s//rec.st.$tmpext/g".to_owned(),
            r"/mat\.st/s//mat.st.$tmpext/g".to_owned(),
            r"/rec\.fft/s//rec.fft.$tmpext/g".to_owned(),
            r"/mat\.fft/s//mat.fft.$tmpext/g".to_owned(),
            r"/^[^ 	]*rec\.st".to_owned() + "/s//\\" + tmprec + "st/g",
            r"/^[^ 	]*mat\.st/s//\\".to_owned() + tmpmat + "st/g",
            r"/[ 	][^ 	]*rec\.st/s// ".to_owned() + tmprec + "st/g",
            r"/[ 	][^ 	]*mat\.st/s// ".to_owned() + tmpmat + "st/g",
            r"/^[^ 	]*rec\.fft/s//\\".to_owned() + tmprec + "fft/g",
            r"/^[^ 	]*mat\.fft/s//\\".to_owned() + tmpmat + "fft/g",
            r"/[ 	][^ 	]*rec\.fft/s// ".to_owned() + tmprec + "fft/g",
            r"/[ 	][^ 	]*mat\.fft/s// ".to_owned() + tmpmat + "fft/g",
        ]);
        let sedlines = pysed(&sedcom, PysedSrc::Lines(&outlines), None, false, '/', false)
            .ok()
            .flatten()
            .unwrap_or_default();
        let _ = write_text_file(&comname, &sedlines, false);
    }

    let mut outlines = comlines[sect_starts[num_chunks]..].to_vec();
    if got_init_chunk {
        outlines.push(format!(
            "$collectmmm pixels= {rootname} {num_chunks} {sumname}"
        ));
    }
    outlines.push(format!(
        "$b3dremove -g {rootname}-[0-9][0-9][0-9]*{com_ext}* {rootname}-[0-9][0-9][0-9]*.log* {lock_name}"
    ));
    let _ = pysed(
        &localcom,
        PysedSrc::Lines(&outlines),
        // Fixed in translation (BUGS.md): native always writes
        // `<root>-finish.com` (`splitcombine:189`), while the start and chunk
        // files and processchunks' finish lookup (`processchunks.cpp:1291-1293`)
        // use the command file's own extension.
        Some(&format!("{rootname}-finish{com_ext}")),
        false,
        '/',
        false,
    );
    prnstr(
        &format!(
            "{} command files created and ready to run with\n  processchunks or the parallel processing interface",
            num_chunks + 2
        ),
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    0
}
