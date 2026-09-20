//! In-progress direct translation of `IMOD/pysrc/pip.py`.
//!
//! `parse_params.rs` contains the native translation of its corresponding C
//! parser.  This module preserves the Python-command-facing functions and
//! Python module state for Python source units such as `submfg`.

use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::sync::{LazyLock, Mutex};

static S_EXIT_PREFIX: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
static PIP_ERRNO: LazyLock<Mutex<i32>> = LazyLock::new(|| Mutex::new(0));
static FORBID_COMMENT_LONG: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new(String::new()));
static FORBID_COMMENT_SHORT: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new(String::new()));
static WARN_ON_COMMENT: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));

/// Matches Python class `pipOption` (`IMOD/pysrc/pip.py:56`).
#[derive(Clone, Debug, Default)]
pub struct PipOption {
    pub short_name: String,
    pub long_name: String,
    pub option_type: String,
    pub help_string: String,
    pub format: String,
    pub default_val: String,
    pub values: Vec<String>,
    pub multiple: i32,
    pub count: i32,
    pub len_short: i32,
    pub next_linked: Vec<i32>,
    pub linked: bool,
}

/// Values returned by `OptionLineOfValues` and `PipGetLineOfValues`.
#[derive(Clone, Debug, PartialEq)]
pub enum PipValues {
    Integers(Vec<i32>),
    Floats(Vec<f32>),
}

/// Matches `setExitPrefix` (`IMOD/pysrc/pip.py:163`).
pub fn set_exit_prefix(prefix: String) {
    *S_EXIT_PREFIX.lock().expect("PIP exit prefix mutex") = Some(prefix);
}

/// Matches `exitError` (`IMOD/pysrc/pip.py:678`).
pub fn exit_error(error_message: &str) -> ! {
    if let Some(prefix) = S_EXIT_PREFIX
        .lock()
        .expect("PIP exit prefix mutex")
        .as_ref()
    {
        eprintln!("{prefix}{error_message}");
    } else {
        eprintln!("{error_message}");
    }
    std::process::exit(1)
}

/// Matches `expandArgList` (`IMOD/pysrc/pip.py:197`).
pub fn expand_arg_list(args: &[OsString]) -> (Vec<OsString>, isize) {
    // The source intentionally delegates glob expansion to Unix shells.  This
    // programmatic expansion is solely for Windows/Cygwin, where Python gets
    // literal wildcard arguments.
    if !cfg!(windows) && !cfg!(target_os = "cygwin") {
        return (args.to_vec(), -1);
    }
    let mut new_list = Vec::new();
    let mut no_match = -1;
    for (index, argument) in args.iter().enumerate() {
        let argument_text = argument.to_string_lossy();
        if !argument_text.contains('*') && !argument_text.contains('?') {
            new_list.push(argument.clone());
            continue;
        }
        // `glob.glob` has recursive pathname semantics.  The standard library
        // deliberately has no glob API; preserve a no-match argument exactly,
        // rather than applying a non-source pattern interpretation.
        new_list.push(argument.clone());
        if no_match < 0 {
            no_match = index as isize;
        }
    }
    (new_list, no_match)
}

/// Matches `PipInitialize` (`IMOD/pysrc/pip.py:101`).
pub fn pip_initialize(number_options: i32) -> i32 {
    unsafe { crate::imod::libcfshr::parse_params::pip_initialize(number_options) }
}

/// Matches `PipWarnUnusedNonOptArgs` (`IMOD/pysrc/pip.py:123`).
pub fn pip_warn_unused_non_opt_args() -> i32 {
    unsafe { crate::imod::libcfshr::parse_params::pip_warn_unused_non_opt_args() }
}

/// Matches `PipDone` (`IMOD/pysrc/pip.py:136`).
pub fn pip_done() {
    unsafe { crate::imod::libcfshr::parse_params::pip_done() }
}

/// Matches `PipExitOnError` (`IMOD/pysrc/pip.py:154`).
pub fn pip_exit_on_error(use_standard_error: i32, prefix: &str) -> i32 {
    *S_EXIT_PREFIX.lock().expect("PIP exit prefix mutex") = Some(prefix.to_string());
    unsafe {
        crate::imod::libcfshr::parse_params::pip_exit_on_error(
            use_standard_error,
            prefix.as_bytes(),
        )
    }
}

/// Matches `PipEnableEntryOutput` (`IMOD/pysrc/pip.py:169`).
pub fn pip_enable_entry_output(value: i32) {
    unsafe { crate::imod::libcfshr::parse_params::pip_enable_entry_output(value) }
}

/// Matches `PipSetLinkedOption` (`IMOD/pysrc/pip.py:175`).
pub fn pip_set_linked_option(option: &str) {
    unsafe {
        crate::imod::libcfshr::parse_params::pip_set_linked_option(option.as_bytes());
    }
}

/// Matches `PipForbidComments` (`IMOD/pysrc/pip.py:181`).
pub fn pip_forbid_comments(long_name: &str, short_name: &str, warn: i32) {
    *FORBID_COMMENT_LONG.lock().expect("PIP comment mutex") = long_name.to_owned();
    *FORBID_COMMENT_SHORT.lock().expect("PIP comment mutex") = short_name.to_owned();
    *WARN_ON_COMMENT.lock().expect("PIP comment mutex") = warn != 0;
}

/// Matches `PipGetErrNo` (`IMOD/pysrc/pip.py:189`).
pub fn pip_get_err_no() -> i32 {
    *PIP_ERRNO.lock().expect("PIP errno mutex")
}

/// Matches `PipAddOption` (`IMOD/pysrc/pip.py:220`).
pub fn pip_add_option(option_string: &str) -> i32 {
    unsafe { crate::imod::libcfshr::parse_params::pip_add_option(option_string.as_bytes()) }
}

/// Matches `PipNextArg` (`IMOD/pysrc/pip.py:275`).
pub fn pip_next_arg(argument_string: &str) -> i32 {
    unsafe { crate::imod::libcfshr::parse_params::pip_next_arg(argument_string.as_bytes()) }
}

/// Matches `PipNumberOfArgs` (`IMOD/pysrc/pip.py:356`).
pub fn pip_number_of_args() -> (i32, i32) {
    let mut option_arguments = 0;
    let mut non_option_arguments = 0;
    unsafe {
        crate::imod::libcfshr::parse_params::pip_number_of_args(
            &mut option_arguments,
            &mut non_option_arguments,
        )
    }
    (option_arguments, non_option_arguments)
}

/// Matches `PipGetNonOptionArg` (`IMOD/pysrc/pip.py:363`).
pub fn pip_get_non_option_arg(argument_number: i32) -> Result<String, i32> {
    *PIP_ERRNO.lock().expect("PIP errno mutex") = 0;
    let mut value: Vec<u8> = Vec::new();
    let status =
        crate::imod::libcfshr::parse_params::pip_get_non_option_arg(argument_number, &mut value);
    if status != 0 {
        *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
        return Err(status);
    }
    Ok(String::from_utf8_lossy(&value).into_owned())
}

/// Matches `PipGetString` (`IMOD/pysrc/pip.py:377`).
pub fn pip_get_string(option: &str, default_value: &str) -> Result<String, i32> {
    *PIP_ERRNO.lock().expect("PIP errno mutex") = 0;
    let mut value: Vec<u8> = Vec::new();
    let status = crate::imod::libcfshr::parse_params::pip_get_string(option.as_bytes(), &mut value);
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        return Err(status);
    }
    if status > 0 {
        return Ok(default_value.to_owned());
    }
    Ok(String::from_utf8_lossy(&value).into_owned())
}

/// Matches `PipGetBoolean` (`IMOD/pysrc/pip.py:390`).
pub fn pip_get_boolean(option: &str, default_value: i32) -> Result<i32, i32> {
    *PIP_ERRNO.lock().expect("PIP errno mutex") = 0;
    let mut value = default_value;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_boolean(option.as_bytes(), &mut value)
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 { Err(status) } else { Ok(value) }
}

/// Matches `PipGetInteger` (`IMOD/pysrc/pip.py:414`).
pub fn pip_get_integer(option: &str, default_value: i32) -> Result<i32, i32> {
    *PIP_ERRNO.lock().expect("PIP errno mutex") = 0;
    let mut value = default_value;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_integer(option.as_bytes(), &mut value)
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 { Err(status) } else { Ok(value) }
}

/// Matches `PipGetFloat` (`IMOD/pysrc/pip.py:424`).
pub fn pip_get_float(option: &str, default_value: f32) -> Result<f32, i32> {
    *PIP_ERRNO.lock().expect("PIP errno mutex") = 0;
    let mut value = default_value;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_float(option.as_bytes(), &mut value)
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 { Err(status) } else { Ok(value) }
}

/// Matches `PipGetTwoIntegers` (`IMOD/pysrc/pip.py:437`).
pub fn pip_get_two_integers(option: &str, default_values: (i32, i32)) -> Result<(i32, i32), i32> {
    let (mut first, mut second) = default_values;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_two_integers(
            option.as_bytes(),
            &mut first,
            &mut second,
        )
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        Err(status)
    } else {
        Ok((first, second))
    }
}

/// Matches `PipGetTwoFloats` (`IMOD/pysrc/pip.py:447`).
pub fn pip_get_two_floats(option: &str, default_values: (f32, f32)) -> Result<(f32, f32), i32> {
    let (mut first, mut second) = default_values;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_two_floats(
            option.as_bytes(),
            &mut first,
            &mut second,
        )
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        Err(status)
    } else {
        Ok((first, second))
    }
}

/// Matches `PipGetThreeIntegers` (`IMOD/pysrc/pip.py:460`).
pub fn pip_get_three_integers(
    option: &str,
    default_values: (i32, i32, i32),
) -> Result<(i32, i32, i32), i32> {
    let (mut first, mut second, mut third) = default_values;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_three_integers(
            option.as_bytes(),
            &mut first,
            &mut second,
            &mut third,
        )
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        Err(status)
    } else {
        Ok((first, second, third))
    }
}

/// Matches `PipGetThreeFloats` (`IMOD/pysrc/pip.py:470`).
pub fn pip_get_three_floats(
    option: &str,
    default_values: (f32, f32, f32),
) -> Result<(f32, f32, f32), i32> {
    let (mut first, mut second, mut third) = default_values;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_three_floats(
            option.as_bytes(),
            &mut first,
            &mut second,
            &mut third,
        )
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        Err(status)
    } else {
        Ok((first, second, third))
    }
}

/// Matches `PipGetIntegerArray` (`IMOD/pysrc/pip.py:484`).
pub fn pip_get_integer_array(option: &str, values: &mut [i32]) -> Result<usize, i32> {
    let len = values.len() as i32;
    let mut number = len;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_integer_array(
            option.as_bytes(),
            values,
            &mut number,
            len,
        )
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        Err(status)
    } else {
        Ok(number as usize)
    }
}

/// Matches `PipGetFloatArray` (`IMOD/pysrc/pip.py:488`).
pub fn pip_get_float_array(option: &str, values: &mut [f32]) -> Result<usize, i32> {
    let len = values.len() as i32;
    let mut number = len;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_get_float_array(
            option.as_bytes(),
            values,
            &mut number,
            len,
        )
    };
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        Err(status)
    } else {
        Ok(number as usize)
    }
}

/// Matches `PipPrintHelp` (`IMOD/pysrc/pip.py:495`).
pub fn pip_print_help(
    program_name: &str,
    use_standard_error: i32,
    input_files: i32,
    output_files: i32,
) -> i32 {
    unsafe {
        crate::imod::libcfshr::parse_params::pip_print_help(
            program_name.as_bytes(),
            use_standard_error,
            input_files,
            output_files,
        )
    }
}

/// Matches `PipPrintEntries` (`IMOD/pysrc/pip.py:608`).
pub fn pip_print_entries() {
    unsafe { crate::imod::libcfshr::parse_params::pip_print_entries() }
}

/// Matches `PipGetError` (`IMOD/pysrc/pip.py:645`).
pub fn pip_get_error() -> Result<String, i32> {
    let mut message: Vec<u8> = Vec::new();
    let status = crate::imod::libcfshr::parse_params::pip_get_error(&mut message);
    if status != 0 {
        return Err(status);
    }
    Ok(String::from_utf8_lossy(&message).into_owned())
}

/// Matches `PipSetError` (`IMOD/pysrc/pip.py:657`).
pub fn pip_set_error(error_string: &str) -> i32 {
    unsafe { crate::imod::libcfshr::parse_params::pip_set_error(error_string.as_bytes()) }
}

/// Matches `PipNumberOfEntries` (`IMOD/pysrc/pip.py:685`).
pub fn pip_number_of_entries(option: &str) -> Result<i32, i32> {
    let mut number = 0;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_number_of_entries(option.as_bytes(), &mut number)
    };
    if status < 0 { Err(status) } else { Ok(number) }
}

/// Matches `PipLinkedIndex` (`IMOD/pysrc/pip.py:699`).
pub fn pip_linked_index(option: &str) -> Result<i32, i32> {
    let mut index = 0;
    let status = unsafe {
        crate::imod::libcfshr::parse_params::pip_linked_index(option.as_bytes(), &mut index)
    };
    if status < 0 { Err(status) } else { Ok(index) }
}

/// Matches `PipParseInput` (`IMOD/pysrc/pip.py:726`).
pub fn pip_parse_input(arguments: &[String], options: &[String]) -> Result<(i32, i32), i32> {
    let argument_bytes = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let option_bytes = options
        .iter()
        .map(|value| value.as_bytes())
        .collect::<Vec<_>>();
    let (mut option_count, mut non_option_count) = (0, 0);
    let status = crate::imod::libcfshr::parse_params::pip_parse_input(
        argument_bytes.len() as i32,
        &argument_bytes,
        &option_bytes,
        option_bytes.len() as i32,
        &mut option_count,
        &mut non_option_count,
    );
    if status < 0 {
        Err(status)
    } else {
        Ok((option_count, non_option_count))
    }
}

/// Matches `PipReadOptionFile` (`IMOD/pysrc/pip.py:779`).
pub fn pip_read_option_file(program_name: &str, help_level: i32, local_directory: i32) -> i32 {
    unsafe {
        crate::imod::libcfshr::parse_params::pip_read_option_file(
            program_name.as_bytes(),
            help_level,
            local_directory,
        )
    }
}

/// Matches `PipParseEntries` (`IMOD/pysrc/pip.py:1014`).
pub fn pip_parse_entries(arguments: &[String]) -> Result<(i32, i32), i32> {
    let argument_bytes = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let (mut option_count, mut non_option_count) = (0, 0);
    let status = crate::imod::libcfshr::parse_params::pip_parse_entries(
        argument_bytes.len() as i32,
        &argument_bytes,
        &mut option_count,
        &mut non_option_count,
    );
    if status < 0 {
        Err(status)
    } else {
        Ok((option_count, non_option_count))
    }
}

/// Matches `PipGetInOutFile` (`IMOD/pysrc/pip.py:1120`).
pub fn pip_get_in_out_file(option: &str, non_option_argument: i32) -> Result<Option<String>, i32> {
    let mut filename: Vec<u8> = Vec::new();
    let status = crate::imod::libcfshr::parse_params::pip_get_in_out_file(
        option.as_bytes(),
        non_option_argument,
        &mut filename,
    );
    if status != 0 {
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&filename).into_owned()))
}

/// Matches `PipStartsWith` (`IMOD/pysrc/pip.py:1504`).
pub fn pip_starts_with(full_string: &str, substring: &str) -> bool {
    if full_string.is_empty() || substring.is_empty() {
        return false;
    }
    unsafe {
        crate::imod::libcfshr::parse_params::pip_starts_with(
            full_string.as_bytes(),
            substring.as_bytes(),
        ) != 0
    }
}

/// Matches `LineIsOptionToken` (`IMOD/pysrc/pip.py:1514`).
pub fn line_is_option_token(line: &str) -> i32 {
    if !pip_starts_with(line, "[") || !line.contains(']') {
        return 0;
    }
    let token = &line[1..];
    if pip_starts_with(token, "Field") {
        1
    } else if pip_starts_with(token, "SectionHeader") {
        2
    } else {
        -1
    }
}

/// Matches `CheckKeyword` (`IMOD/pysrc/pip.py:1537`).
pub fn check_keyword(line: &str, keyword: &str, index: i32, quote_ok: bool) -> (String, i32, i32) {
    if !pip_starts_with(line, keyword) {
        return (String::new(), 0, -1);
    }
    let Some(delimiter_index) = line.find('=') else {
        return (String::new(), 0, -1);
    };
    let value = line[delimiter_index + 1..].trim_start_matches([' ', '\t']);
    if value.is_empty() {
        return (String::new(), 0, -1);
    }
    if quote_ok && matches!(value.as_bytes()[0], b'\'' | b'"' | b'`') {
        let quote = "'\"`".find(value.as_bytes()[0] as char).unwrap_or_default() as i32;
        return (value[1..].to_owned(), index, quote);
    }
    (value.to_owned(), index, -1)
}

/// Matches `PipOpenInstalledAdoc` (`IMOD/pysrc/pip.py:747`).
pub fn pip_open_installed_adoc(program_name: &str) -> Option<File> {
    if let Some(directory) = std::env::var_os("AUTODOC_DIR") {
        if let Ok(file) =
            File::open(std::path::Path::new(&directory).join(format!("{program_name}.adoc")))
        {
            return Some(file);
        }
    }
    std::env::var_os("IMOD_DIR").and_then(|directory| {
        File::open(
            std::path::Path::new(&directory)
                .join("autodoc")
                .join(format!("{program_name}.adoc")),
        )
        .ok()
    })
}

/// Matches `PipReadNextLine` (`IMOD/pysrc/pip.py:1230`).
pub fn pip_read_next_line(
    file: &mut File,
    comment: char,
    keep_comments: bool,
    inline_comments: bool,
) -> (i32, String, usize, String) {
    loop {
        let mut bytes = Vec::new();
        loop {
            let mut byte = [0u8; 1];
            match file.read(&mut byte) {
                Ok(0) => break,
                Ok(_) => {
                    bytes.push(byte[0]);
                    if byte[0] == b'\n' {
                        break;
                    }
                }
                Err(_) => return (-2, String::new(), 0, String::new()),
            }
        }
        if bytes.is_empty() {
            return (-3, String::new(), 0, String::new());
        }
        let mut line = String::from_utf8_lossy(&bytes).into_owned();
        let Some(first_non_white) =
            line.find(|character: char| character != ' ' && character != '\t')
        else {
            continue;
        };
        if line[first_non_white..].starts_with(comment) {
            if keep_comments {
                line = line.trim_end().to_owned();
                return (line.len() as i32, line, first_non_white, String::new());
            }
            continue;
        }
        let mut bad_comment = String::new();
        let mut length = line.len();
        if inline_comments && let Some(index) = line.find(comment) {
            length = index;
            let option = line
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_start_matches('-');
            if (*FORBID_COMMENT_LONG.lock().expect("PIP comment mutex")).starts_with(option)
                || (*FORBID_COMMENT_SHORT.lock().expect("PIP comment mutex")).starts_with(option)
            {
                bad_comment = format!(
                    "The {comment} character should not be used for a comment or any other reason on the line: {line}"
                );
            }
        }
        line.truncate(length);
        line = line.trim_end().to_owned();
        if first_non_white < line.len() || keep_comments {
            return (line.len() as i32, line, first_non_white, bad_comment);
        }
    }
}

/// Matches `ReadParamFile` (`IMOD/pysrc/pip.py:1134`).
pub fn read_param_file(file: &mut File) -> i32 {
    loop {
        let (length, line, first, bad_comment) = pip_read_next_line(file, '#', false, true);
        if length == -3 {
            return 0;
        }
        if length == -2 {
            return pip_set_error("Error reading parameter file or StandardInput");
        }
        if !bad_comment.is_empty() {
            if *WARN_ON_COMMENT.lock().expect("PIP comment mutex") {
                println!("PIP WARNING: {bad_comment}");
            } else {
                return pip_set_error(&bad_comment);
            }
        }
        let content = &line[first..];
        let token_end = content
            .find(|character: char| character == '=' || character == ' ' || character == '\t')
            .unwrap_or(content.len());
        let token = &content[..token_end];
        if token == "EndInput" || token == "DONE" {
            return 0;
        }
        let value = content[token_end..].trim_start_matches([' ', '\t', '=']);
        let option_status = pip_next_arg(&format!("-{token}"));
        if option_status < 0 {
            return option_status;
        }
        if option_status == 0 {
            if !value.is_empty() {
                return pip_set_error(&format!("A value was supplied to boolean option: {line}"));
            }
            continue;
        }
        if value.is_empty() {
            return pip_set_error(&format!("Missing a value on the input line: {line}"));
        }
        let value_status = pip_next_arg(value);
        if value_status < 0 {
            return value_status;
        }
    }
}

/// Matches `PipGetLineOfValues` (`IMOD/pysrc/pip.py:1310`).
pub fn pip_get_line_of_values(
    option: &str,
    value_string: &str,
    value_type: i32,
    number_to_get: usize,
) -> Result<PipValues, i32> {
    let mut integers = Vec::new();
    let mut floats = Vec::new();
    let mut previous_comma = false;
    for token in
        value_string.split_inclusive(|character: char| matches!(character, ',' | ' ' | '\t' | '/'))
    {
        let entry =
            token.trim_matches(|character: char| matches!(character, ',' | ' ' | '\t' | '/'));
        if entry.is_empty() {
            if token.contains('/') || (token.contains(',') && previous_comma) {
                return Err(pip_set_error(&format!(
                    "Default entry is not allowed in value entry:  {option}  {value_string}"
                )));
            }
            previous_comma = token.contains(',');
            continue;
        }
        previous_comma = false;
        if value_type == 1 {
            integers.push(entry.parse::<i32>().map_err(|_| {
                pip_set_error(&format!(
                    "Illegal character in value entry:  {option}  {value_string}"
                ))
            })?);
        } else {
            floats.push(entry.parse::<f32>().map_err(|_| {
                pip_set_error(&format!(
                    "Illegal character in value entry:  {option}  {value_string}"
                ))
            })?);
        }
        if number_to_get != 0 && integers.len() + floats.len() >= number_to_get {
            break;
        }
    }
    let number = integers.len() + floats.len();
    if number_to_get != 0 && number < number_to_get {
        return Err(pip_set_error(&format!(
            "{number_to_get} values expected but only {number} values found in value entry:  {option}  {value_string}"
        )));
    }
    if value_type == 1 {
        Ok(PipValues::Integers(integers))
    } else {
        Ok(PipValues::Floats(floats))
    }
}

/// Matches `GetNextValueString` (`IMOD/pysrc/pip.py:1408`).
pub fn get_next_value_string(option: &str) -> Result<(i32, Option<String>), i32> {
    let mut value: Vec<u8> = Vec::new();
    let status =
        crate::imod::libcfshr::parse_params::get_next_value_string(option.as_bytes(), &mut value);
    *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    if status < 0 {
        return Err(status);
    }
    if status == 1 {
        return Ok((status, None));
    }
    Ok((status, Some(String::from_utf8_lossy(&value).into_owned())))
}

/// Matches `AddValueString` (`IMOD/pysrc/pip.py:1433`).
pub fn add_value_string(option_index: i32, value: &str) -> i32 {
    crate::imod::libcfshr::parse_params::add_value_string(option_index, value.as_bytes())
}

/// Matches `LookupOption` (`IMOD/pysrc/pip.py:1463`).
pub fn lookup_option(option: &str, maximum_lookup: i32) -> i32 {
    crate::imod::libcfshr::parse_params::lookup_option(option.as_bytes(), maximum_lookup)
}

/// Matches `OptionLineOfValues` (`IMOD/pysrc/pip.py:1287`).
pub fn option_line_of_values(
    option: &str,
    value_type: i32,
    number_to_get: usize,
) -> Result<PipValues, i32> {
    let (status, value) = get_next_value_string(option)?;
    if status != 0 && status != 2 {
        return Err(status);
    }
    let result = pip_get_line_of_values(
        option,
        &value.unwrap_or_default(),
        value_type,
        number_to_get,
    );
    if result.is_ok() {
        *PIP_ERRNO.lock().expect("PIP errno mutex") = status;
    }
    result
}

/// Matches `PipReadOrParseOptions` (`IMOD/pysrc/pip.py:1046`).
///
/// Its usage/help termination is intentionally retained in the translated C
/// parser boundary, as in the Python routine's `sys.exit(0)` path.
pub fn pip_read_or_parse_options(
    arguments: &[String],
    options: &[String],
    program_name: &str,
    minimum_arguments: i32,
    input_files: i32,
    output_files: i32,
    header_function: Option<fn(&[u8])>,
) -> (i32, i32) {
    let argument_bytes = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let option_bytes = options
        .iter()
        .map(|value| value.as_bytes())
        .collect::<Vec<_>>();
    let (mut option_count, mut non_option_count) = (0, 0);
    crate::imod::libcfshr::parse_params::pip_read_or_parse_options(
        argument_bytes.len() as i32,
        &argument_bytes,
        &option_bytes,
        option_bytes.len() as i32,
        program_name.as_bytes(),
        minimum_arguments,
        input_files,
        output_files,
        &mut option_count,
        &mut non_option_count,
        header_function,
    );
    (option_count, non_option_count)
}

#[cfg(test)]
mod tests {
    use super::{
        PipOption, expand_arg_list, pip_add_option, pip_done, pip_get_integer, pip_initialize,
        pip_next_arg,
    };
    use std::ffi::OsString;

    #[test]
    fn pip_option_defaults_match_python_constructor() {
        let option = PipOption::default();
        assert_eq!(option.count, 0);
        assert!(option.values.is_empty());
        assert!(!option.linked);
    }

    #[test]
    fn unix_argument_expansion_is_identity() {
        let values = vec![OsString::from("one*"), OsString::from("two")];
        let (expanded, no_match) = expand_arg_list(&values);
        if !cfg!(windows) && !cfg!(target_os = "cygwin") {
            assert_eq!(expanded, values);
            assert_eq!(no_match, -1);
        }
    }

    #[test]
    fn parser_round_trip_uses_source_option_state() {
        assert_eq!(pip_initialize(1), 0);
        assert_eq!(pip_add_option("n:Number:I:integer"), 0);
        assert_eq!(pip_next_arg("-n"), 1);
        assert_eq!(pip_next_arg("17"), 0);
        assert_eq!(pip_get_integer("Number", 3), Ok(17));
        pip_done();
    }
}
