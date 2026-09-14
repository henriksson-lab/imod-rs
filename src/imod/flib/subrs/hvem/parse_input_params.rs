//! Translation of `IMOD/flib/subrs/hvem/parse_input_params.f90`.
//!
//! This is the Fortran front end to the PIP package: every program unit in the
//! source file maps to one function here, layered over the already-translated
//! C engine in [`crate::imod::libcfshr::parse_params`] exactly as the Fortran
//! layers over `libcfshr` through `pip_fwrap.c`.  The Fortran `character*(*)`
//! dummy arguments are represented by `&str`, `logical` by `bool`, and the
//! `common / exitprefix /` block by the module-level [`EXIT_PREFIX`] storage.
#![allow(dead_code)]

use crate::imod::libcfshr::autodoc::adoc_set_current;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::parse_params::{
    pip_add_option, pip_allow_comma_defaults, pip_exit_on_error as pip_exit_on_error_fw,
    pip_get_boolean, pip_get_error, pip_get_non_option_arg, pip_get_string, pip_initialize,
    pip_next_arg, pip_number_of_args, pip_print_entries, pip_print_help, pip_read_option_file,
    pip_read_prog_defaults, pip_read_stdin_if_set, pip_set_error,
};
use std::io::{self, Write};

/// `character*32 prefix` in `common / exitprefix / prefix`
/// (`parse_input_params.f90:235,246`).  A Fortran common block starts as zero
/// bytes, and `trim` strips only trailing blanks, so the uninitialised state is
/// reproduced with NUL fill rather than blank fill.
static mut EXIT_PREFIX: [u8; 32] = [0; 32];

/// `integer bufferSize / parameter (bufferSize = 1024)`
/// (`parse_input_params.f90:70`).
const BUFFER_SIZE: usize = 1024;

/// Original Fortran `PipParseInput` (`parse_input_params.f90:18`).
pub fn pip_parse_input(
    options: &[&str],
    num_options: i32,
    separator: char,
    num_opt_arg: &mut i32,
    num_non_opt_arg: &mut i32,
) -> i32 {
    unsafe {
        //
        // initialize then pass the options one by one
        //
        let mut result = pip_initialize(num_options);
        if result != 0 {
            return result;
        }
        if separator == ' ' {
            for i in 1..=num_options {
                result = pip_add_option(options[(i - 1) as usize].as_bytes());
                if result != 0 {
                    return result;
                }
            }
        } else {
            //
            // if options are all in one string with a separator
            //
            let all: Vec<u8> = options[0].as_bytes().to_vec();
            let mut ind_str = 1_i32;
            let mut len_all = all.len() as i32;
            while len_all > 0 && all[(len_all - 1) as usize] == b' ' {
                len_all -= 1;
            }
            for _i in 1..=num_options {
                let mut j = ind_str;
                let mut ind_end = 0_i32;
                while j <= len_all && ind_end == 0 {
                    if all[(j - 1) as usize] == separator as u8 {
                        ind_end = j - 1;
                    }
                    j += 1;
                }
                if j > len_all {
                    ind_end = len_all;
                }
                if ind_str > ind_end {
                    pip_set_error(b"Too few options in string");
                    return -1;
                }
                result = pip_add_option(&all[(ind_str - 1) as usize..ind_end as usize]);
                if result != 0 {
                    return result;
                }
                ind_str = j;
            }
        }

        pip_parse_entries(num_opt_arg, num_non_opt_arg)
    }
}

/// Original Fortran `PipParseEntries` (`parse_input_params.f90:68`).
pub fn pip_parse_entries(num_opt_arg: &mut i32, num_non_opt_arg: &mut i32) -> i32 {
    unsafe {
        //
        // pass the arguments in one by one
        //
        let arguments: Vec<std::ffi::OsString> = std::env::args_os().collect();
        let iargc = arguments.len() as i32 - 1;
        for i in 1..=iargc {
            // `call getarg(i, string)` into `character*(bufferSize)`, then
            // `PipNextArg(string)` which trims the blank padding back off in
            // `pipf2cstr`; both truncation and the trim are reproduced here.
            let mut bytes =
                std::os::unix::ffi::OsStrExt::as_bytes(arguments[i as usize].as_os_str()).to_vec();
            bytes.truncate(BUFFER_SIZE);
            let mut len_trim = bytes.len();
            while len_trim > 0 && bytes[len_trim - 1] == b' ' {
                len_trim -= 1;
            }
            if len_trim == BUFFER_SIZE {
                pip_set_error(b"Input argument too long for buffer in PipParseEntries");
                return -1;
            }
            bytes.truncate(len_trim);
            let result = pip_next_arg(&bytes);
            if result < 0 {
                return result;
            }
            if result > 0 && i == iargc {
                pip_set_error(
                    b"A value was expected but not found for the last option on the command line",
                );
                return -1;
            }
        }
        //
        // Or read stdin if the flag is set to do it when no arguments
        //
        if iargc == 0 {
            let result = pip_read_stdin_if_set();
            if result != 0 {
                return result;
            }
        }
        //
        // get numbers to return
        //
        pip_number_of_args(num_opt_arg, num_non_opt_arg);
        pip_print_entries();
        0
    }
}

/// Original Fortran `PipGetLogical` (`parse_input_params.f90:114`).
pub fn pip_get_logical(option: &str, value: &mut bool) -> i32 {
    unsafe {
        let mut intval = 0_i32;
        let mut result = 0_i32;
        let ierr = pip_get_boolean(option.as_bytes(), &mut intval);
        if ierr != 0 {
            result = ierr;
        } else {
            *value = intval != 0;
        }
        result
    }
}

/// Original Fortran `PipReadOrParseOptions` (`parse_input_params.f90:136`).
pub fn pip_read_or_parse_options(
    options: &[&str],
    num_options: i32,
    prog_name: &str,
    exit_string: &str,
    interactive: bool,
    min_args: i32,
    num_in_files: i32,
    num_out_files: i32,
    num_opt_arg: &mut i32,
    num_non_opt_arg: &mut i32,
) {
    unsafe {
        //
        // First try to read autodoc file
        //
        pip_allow_comma_defaults(1);
        let mut ierr = pip_read_option_file(prog_name.as_bytes(), 1, 0);
        pip_exit_on_error(0, exit_string);
        //
        // If that is OK, go parse the entries;
        // otherwise print error message and use fallback option list
        //
        if ierr == 0 {
            ierr = pip_parse_entries(num_opt_arg, num_non_opt_arg);
        } else {
            let mut error_string: Vec<u8> = Vec::new();
            ierr = pip_get_error(&mut error_string);
            // `errString` is `character*240`, so `write(*, '(a, a)')` emits the
            // blank-padded 240-character variable, not the trimmed message.
            let mut padded: Vec<u8> = error_string;
            padded.truncate(240);
            padded.resize(240, b' ');
            println!("PIP WARNING: {}", String::from_utf8_lossy(&padded));
            println!(" Using fallback options in main program");
            ierr = pip_parse_input(options, num_options, '@', num_opt_arg, num_non_opt_arg);
            if ierr == 0 {
                pip_read_prog_defaults(prog_name.as_bytes());
            }
        }
        //
        // Process help input
        //
        if interactive && *num_opt_arg + *num_non_opt_arg == 0 {
            return;
        }
        if *num_opt_arg + *num_non_opt_arg < min_args || pip_get_boolean(b"help", &mut ierr) == 0 {
            pip_print_help(prog_name.as_bytes(), 0, num_in_files, num_out_files);
            std::process::exit(0);
        }
    }
}

/// Original Fortran `PipGetInOutFile` (`parse_input_params.f90:185`).
pub fn pip_get_in_out_file(
    option: &str,
    non_opt_arg_no: i32,
    prompt: &str,
    filename: &mut String,
) -> i32 {
    unsafe {
        let mut result = 0_i32;
        let mut num_opt_arg = 0_i32;
        let mut num_non_opt_arg = 0_i32;
        pip_number_of_args(&mut num_opt_arg, &mut num_non_opt_arg);
        //
        // if there is PIP input, first look for explicit option by the name
        // then get the given non-option argument if there are enough
        //
        if num_opt_arg + num_non_opt_arg > 0 {
            let mut value: Vec<u8> = Vec::new();
            result = pip_get_string(option.as_bytes(), &mut value);
            if result == 0 {
                *filename = String::from_utf8_lossy(&value).into_owned();
            }
            if result != 0 {
                if num_non_opt_arg < non_opt_arg_no {
                    return result;
                }
                value.clear();
                result = pip_get_non_option_arg(non_opt_arg_no - 1, &mut value);
                if result == 0 {
                    *filename = String::from_utf8_lossy(&value).into_owned();
                }
            }
        } else {
            //
            // Otherwise get interactive input with the prompt
            //
            print!(" {}{}", prompt, ": ");
            let _ = io::stdout().flush();
            let mut line = String::new();
            let _ = io::stdin().read_line(&mut line);
            *filename = line.trim_end_matches(['\r', '\n']).to_owned();
        }
        result
    }
}

/// Original Fortran `PipExitOnError` (`parse_input_params.f90:218`).
///
/// `PipExitOnErrorFW` reaches C through `pipf2cstr`, which strips the blank
/// padding of the Fortran character argument, so the C `sExitPrefix` loses the
/// trailing blank of a literal such as `'ERROR: HEADER - '`; `setExitPrefix`
/// keeps the literal as written and trims it again inside `exitError`.
pub fn pip_exit_on_error(if_use_stderr: i32, message: &str) {
    unsafe {
        pip_exit_on_error_fw(if_use_stderr, message.trim_end_matches(' ').as_bytes());
    }
    set_exit_prefix(message);
}

/// Original Fortran `exitError` (`parse_input_params.f90:231`).
///
/// The source writes to unit 6 with a leading blank record, so the message
/// lands on stdout and not on stderr, and then exits with status 1.
///
/// The write goes through libc stdout, not Rust's stream.  In a native
/// program gfortran's unit 6 and libc stdout interleave in program order even
/// under a pipe, but in this translation Rust's `println!` flushes per line
/// while libc's stdout is block buffered, so a `b3dError` diagnostic issued
/// before this one would be overtaken by it.  `newstack` on a truncated input
/// is the observed case: the source prints two `ERROR: mrcReadSectionAny -
/// reading data from file.` lines and then `ERROR: NEWSTACK - Reading image
/// file`, and a `println!` here put the last line first.
pub fn exit_error(message: &str) -> ! {
    let prefix = unsafe {
        let stored = &*core::ptr::addr_of!(EXIT_PREFIX);
        let mut length = stored.len();
        while length > 0 && stored[length - 1] == b' ' {
            length -= 1;
        }
        String::from_utf8_lossy(&stored[..length]).into_owned()
    };
    // `write(*,'(/,a,a,a)') trim(prefix), ' ', trim(message)`
    // (`parse_input_params.f90:236`).
    let text = message.trim_end_matches(' ');
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "\n%s%s%s\n",
        &[
            CArg::Bytes(prefix.as_bytes()),
            CArg::Str(" "),
            CArg::Bytes(text.as_bytes()),
        ],
    ));
    std::process::exit(1);
}

/// Original Fortran `setExitPrefix` (`parse_input_params.f90:243`).
pub fn set_exit_prefix(message: &str) {
    unsafe {
        let stored = &mut *core::ptr::addr_of_mut!(EXIT_PREFIX);
        stored.fill(b' ');
        let bytes = message.as_bytes();
        let count = bytes.len().min(stored.len());
        stored[..count].copy_from_slice(&bytes[..count]);
    }
}

/// Original Fortran `memoryError` (`parse_input_params.f90:254`).
pub fn memory_error(ierr: i32, message: &str) {
    if ierr != 0 {
        exit_error(&format!("Failure to allocate {}", message));
    }
}

/// Original Fortran `memoryErrorUC` (`parse_input_params.f90:264`).
pub fn memory_error_uc(ierr: i32, message: &str) {
    if ierr != 0 {
        exit_error(&format!("FAILURE TO ALLOCATE {}", message));
    }
}

/// Original Fortran `setCurrentAdocOrExit` (`parse_input_params.f90:275`).
/// `indAdoc` is the source's 1-based autodoc index, and the Fortran wrapper
/// `adocsetcurrent` (`adoc_fwrap.c:159`) subtracts one before the C entry
/// point, so this does the same.
pub fn set_current_adoc_or_exit(ind_adoc: i32, message: &str) {
    if unsafe { adoc_set_current(ind_adoc - 1) } != 0 {
        exit_error(&format!(
            "Selecting {} autodoc as current one",
            message.trim_end_matches(' ')
        ));
    }
}
