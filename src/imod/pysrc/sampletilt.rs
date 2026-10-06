//! Translation of `IMOD/pysrc/sampletilt`: runs tilt for a sample tomogram.
//!
//! A Python command script with no functions; its top level is
//! [`sampletilt`].  `tilt` is this crate's own program; `runcmd` runs it in
//! process (`imodpy::run_cmd`).

use super::imodpy::{
    add_imod_bin_ignore_sighup, cleanup_files, exit_from_imod_error, fmtstr, prnstr, py_int,
    py_int_floordiv, read_text_file, run_cmd,
};
use super::pysed::{PysedSrc, pysed};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`sampletilt:1-91`).  Returns the status of its
/// `sys.exit`; error paths exit the process.
pub fn sampletilt(arguments: &[OsString]) -> i32 {
    let progname = "sampletilt";
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

    let num_args = argv.len() as i64 - 1;
    if py_int_floordiv(num_args, 2) != 3 && py_int_floordiv(num_args + 1, 2) != 5 {
        prnstr(
            &format!(
                "{prefix}Need 6, 7, 9, or 10 params: slice start & end, ystart, set name, rec name, com file name, [aliFile] | [orig and new # of lines, # of slices [aliFile]]"
            ),
            "\n",
            false,
        );
        return done(1);
    }

    let mut slstart = argv[1].clone();
    let mut slend = argv[2].clone();
    let mut ystart = argv[3].clone();
    let setname = &argv[4];
    let recname = &argv[5];
    let tiltname = &argv[6];
    let mut ali_name = String::new();
    if num_args == 7 || num_args == 10 {
        ali_name = argv[num_args as usize].clone();
    }

    if num_args == 9 || num_args == 10 {
        // The 3 new parameters allow us to supercede the starting and ending slice numbers
        // at the beginning, which are still there in case old sampletilt is run
        match (
            py_int(&argv[7]),
            py_int(&argv[8]),
            py_int(&argv[9]),
            py_int(&ystart),
        ) {
            (Some(orig_lines), Some(num_lines), Some(num_slices), Some(ystart_value)) => {
                ystart = (ystart_value - py_int_floordiv(num_lines - orig_lines, 2)).to_string();
                let start = py_int_floordiv(num_lines, 2) - py_int_floordiv(num_slices, 2);
                slstart = start.to_string();
                slend = (start + num_slices - 1).to_string();
            }
            _ => {
                prnstr(
                    &format!(
                        "{prefix}Error converting number of input lines or output slices to integer"
                    ),
                    "\n",
                    false,
                );
                return done(1);
            }
        }
    }

    let mut sedcom = vec![
        "/^[$#]/d".to_owned(),
        fmtstr("/^SUBSETSTART.*/s//SUBSETSTART 0 {}/", &[ystart]),
        fmtstr(
            "/^OutputFile/s/[ \t]{}.*/ {}/",
            &[setname.clone(), recname.clone()],
        ),
        "/^SLICE/d".to_owned(),
        "/^WIDTH/d".to_owned(),
        "/^AdjustOrigin/d".to_owned(),
        fmtstr("/^THICKNESS/a/SLICE {} {} 1/", &[slstart, slend]),
    ];
    if !ali_name.is_empty() {
        sedcom.push(fmtstr(
            "/^InputProjections/s/[ \t].*/ {}/",
            &[ali_name.clone()],
        ));
    }

    let mut tilt_lines = read_text_file(tiltname, None, false, None).unwrap_or_default();
    for line in tilt_lines.iter_mut() {
        *line = line.trim_start().to_owned();
    }
    let tiltcom = match pysed(&sedcom, PysedSrc::Lines(&tilt_lines), None, true, '/', false) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(message) => super::pip::exit_error(&message),
    };

    match run_cmd("tilt -StandardInput", Some(&tiltcom), Some("stdout"), None, &[]) {
        Ok(_) => {
            if !ali_name.is_empty() {
                cleanup_files(&[ali_name]);
            }
        }
        Err(_) => {
            if !ali_name.is_empty() {
                cleanup_files(&[ali_name]);
            }
            exit_from_imod_error(progname);
        }
    }

    done(0)
}
