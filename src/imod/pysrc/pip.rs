//! Translation of `IMOD/pysrc/pip.py`, the Python PIP (Parse Input Params)
//! module that every `IMOD/pysrc` script imports.
//!
//! `pip.py` is **not** a binding to the C parser in `libcfshr/parse_params.c`;
//! it is a separate reimplementation with its own behaviour, and a translated
//! Python program must use this module rather than
//! `libcfshr::parse_params` to match its original.  Observable differences
//! from the C parser that this module therefore reproduces include:
//!
//! - `PipGetLineOfValues` starts with `gotComma = 0`, so a leading comma
//!   (`-x ,5,6`) is skipped where the C (`gotComma = 1`) rejects it; values
//!   are parsed with Python `int()`/`float()` (underscores accepted, no hex
//!   floats, results are Python ints and doubles) and arrays are unbounded.
//! - `PipNextArg` tests `argString.strip()[0]`: an argument with leading
//!   white space can be an option, and an empty argument raises `IndexError`.
//! - `ReadParamFile` takes the token end from a match on
//!   `lineStr[indst:]` but uses it as an index into `lineStr`, so native
//!   splits an indented parameter line at the wrong place; fixed here
//!   (BUGS.md), the match offset is added to `indst`.
//! - `PipPrintHelp` prints "(Successive entries accumulate)" for linked
//!   options too, strips only `\fR`/`\fI`/`\fB` from formats, counts in
//!   characters, and `PipPrintEntries` prints its banner even with no entries.
//! - `PipReadOrParseOptions` parses `progDefaults.adoc` with a regular
//!   expression rather than through the autodoc reader, swallowing `IOError`.
//!
//! Module globals are process-global statics, as a Python module's are.
//! Where the Python raises an uncaught exception, the translation writes a
//! traceback line on stderr and exits with status 1, as the interpreter does
//! (see [`python_uncaught`]).

use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering::Relaxed};

// The global defines, tables and variables (`pip.py:23-98`)
const NON_OPTION_STRING: &str = "NonOptionArgument";
const STANDARD_INPUT_STRING: &str = "StandardInput";
const STANDARD_INPUT_END: &str = "EndInput";
const LOOKUP_NOT_FOUND: i32 = -1;
const LOOKUP_AMBIGUOUS: i32 = -2;
const OPTFILE_DIR: &str = "autodoc";
const OPTFILE_EXT: &str = "adoc";
const OPTDIR_VARIABLE: &str = "AUTODOC_DIR";
const DEFAULTS_FILE: &str = "progDefaults.adoc";
const DEFAULTS_DIR: &str = "com";
const DEFAULT_SUB_STR: &str = "%{default}";
const PRINTENTRY_VARIABLE: &str = "PIP_PRINT_ENTRIES";
const OPEN_DELIM: &str = "[";
const CLOSE_DELIM: &str = "]";
const VALUE_DELIM: &str = "=";
const PATH_SEPARATOR: &str = "/";
/// `PIP_INTEGER` (`pip.py:40`).
pub const PIP_INTEGER: i32 = 1;
/// `PIP_FLOAT` (`pip.py:41`).
pub const PIP_FLOAT: i32 = 2;

const S_TYPES: [&str; 13] = [
    "B", "PF", "LI", "I", "F", "IP", "FP", "IT", "FT", "IA", "FA", "CH", "FN",
];
const S_TYPE_FOR_USAGE: [&str; 14] = [
    "Boolean",
    "File",
    "List",
    "Int",
    "Float",
    "2 ints",
    "2 floats",
    "3 ints",
    "3 floats",
    "Ints",
    "Floats",
    "String",
    "File",
    "Unknown argument type",
];
const S_NUM_TYPES: usize = 13;
// `sQuoteTypes = """'"`"""` (`pip.py:51`) -- note the order differs from the C's.
const S_QUOTE_TYPES: [char; 3] = ['\'', '"', '`'];

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

/// The list `OptionLineOfValues`/`PipGetLineOfValues` return: Python ints or
/// Python floats (doubles).
#[derive(Clone, Debug, PartialEq)]
pub enum PipValues {
    Integers(Vec<i64>),
    Floats(Vec<f64>),
}

static S_OPT_TABLE: Mutex<Option<Vec<PipOption>>> = Mutex::new(None);
static S_TABLE_SIZE: AtomicI32 = AtomicI32::new(0);
static S_NUM_OPTIONS: AtomicI32 = AtomicI32::new(0);
static S_NON_OPT_IND: AtomicI32 = AtomicI32::new(0);
static S_ERROR_STRING: Mutex<Option<String>> = Mutex::new(None);
static S_EXIT_PREFIX: Mutex<Option<String>> = Mutex::new(None);
static S_PROGRAM_NAME: Mutex<String> = Mutex::new(String::new());
static S_ERROR_DEST: AtomicI32 = AtomicI32::new(0);
static S_NEXT_OPTION: AtomicI32 = AtomicI32::new(0);
static S_NEXT_ARG_BELONGS_TO: AtomicI32 = AtomicI32::new(-1);
static S_NUM_OPTION_ARGUMENTS: AtomicI32 = AtomicI32::new(0);
static S_ALLOW_DEFAULTS: AtomicI32 = AtomicI32::new(0);
static S_OUTPUT_MANPAGE: AtomicI32 = AtomicI32::new(0);
static S_PRINT_ENTRIES: AtomicI32 = AtomicI32::new(-1);
// `None` is the import-time value `VALUE_DELIM`.
static S_VALUE_DELIM: Mutex<Option<String>> = Mutex::new(None);
static S_DONE_ENDS: AtomicI32 = AtomicI32::new(0);
static S_TAKE_STD_IN: AtomicI32 = AtomicI32::new(0);
static S_NON_OPT_LINES: AtomicI32 = AtomicI32::new(0);
static S_NO_ABBREVS: AtomicI32 = AtomicI32::new(0);
static S_NOT_FOUND_OK: AtomicI32 = AtomicI32::new(0);
static S_LINKED_OPTION: Mutex<Option<String>> = Mutex::new(None);
static S_TEST_ABBREV_FOR_USAGE: AtomicBool = AtomicBool::new(false);
static S_FORBID_COMMENT_LONG: Mutex<String> = Mutex::new(String::new());
static S_FORBID_COMMENT_SHORT: Mutex<String> = Mutex::new(String::new());
static S_WARN_ON_COMMENT: AtomicBool = AtomicBool::new(false);
static S_HIGHEST_NON_OPT_GOTTEN: AtomicI32 = AtomicI32::new(-1);
static PIP_ERRNO: AtomicI32 = AtomicI32::new(0);

/// Python runtime: an uncaught exception prints a traceback on stderr and
/// the interpreter exits with status 1 (after flushing `sys.stdout`).
fn python_uncaught(exception: &str) -> ! {
    let _ = std::io::stdout().flush();
    eprintln!("Traceback (most recent call last):");
    eprintln!("{exception}");
    std::process::exit(1)
}

/// Python runtime: `str.isspace` for one character.  Rust's
/// `char::is_whitespace` is the Unicode `White_Space` property, which lacks
/// the four separators U+001C..U+001F that Python counts as space.
fn python_isspace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// Python runtime: `s[a:b]` on a string held as characters, with Python's
/// clamping of out-of-range and crossed bounds.
fn python_slice(characters: &[char], start: usize, end: usize) -> String {
    let end = end.min(characters.len());
    if start >= end {
        return String::new();
    }
    characters[start..end].iter().collect()
}

/// Python runtime: builtin `int(s)` on a string (base 10).  Surrounding
/// white space is stripped, a sign is accepted, and single underscores may
/// separate digits.  Python ints are unbounded; values beyond `i128` are
/// rejected here and larger-than-`i64` values wrap -- a documented limit,
/// not source behaviour.  Non-ASCII decimal digits, which Python accepts,
/// are not.
fn python_int(text: &str) -> Option<i64> {
    let text = text.trim_matches(python_isspace);
    let (sign, digits) = match text.as_bytes().first() {
        Some(b'+') => ("", &text[1..]),
        Some(b'-') => ("-", &text[1..]),
        _ => ("", text),
    };
    let bytes = digits.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_digit() || !bytes[bytes.len() - 1].is_ascii_digit() {
        return None;
    }
    let mut cleaned = String::from(sign);
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'_' {
            if !bytes[index - 1].is_ascii_digit() || !bytes[index + 1].is_ascii_digit() {
                return None;
            }
        } else if byte.is_ascii_digit() {
            cleaned.push(byte as char);
        } else {
            return None;
        }
    }
    cleaned.parse::<i128>().ok().map(|value| value as i64)
}

/// Python runtime: builtin `float(s)`.  Surrounding white space is
/// stripped and single underscores between digits are accepted; the rest
/// (sign, `inf`/`infinity`/`nan` in any case, exponent, leading or trailing
/// `.`) is what Rust's `f64` parser also accepts.  Hexadecimal is rejected by
/// both.
fn python_float(text: &str) -> Option<f64> {
    let text = text.trim_matches(python_isspace);
    let bytes = text.as_bytes();
    let mut cleaned = String::new();
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'_' {
            if index == 0
                || index + 1 >= bytes.len()
                || !bytes[index - 1].is_ascii_digit()
                || !bytes[index + 1].is_ascii_digit()
            {
                return None;
            }
        } else {
            cleaned.push(byte as char);
        }
    }
    cleaned.parse::<f64>().ok()
}

/// Python runtime: a text-mode file object as `pip.py` uses it --
/// `open(name, "r")` (universal newlines, strict UTF-8) or `sys.stdin`
/// (newline `"\n"` only, `surrogateescape`, approximated here by lossy
/// decoding).  A decoding error on a file surfaces from `readline`, which
/// the callers' bare `except:` turns into their read-error return; Python
/// decodes in 8 KiB chunks, so it can surface a line or so earlier than
/// here.
pub struct PythonTextFile {
    reader: Box<dyn BufRead>,
    universal_newlines: bool,
}

impl PythonTextFile {
    /// `open(name, "r")`; a directory raises `IsADirectoryError` in Python
    /// (an `IOError`), which `File::open` on Linux would not.
    pub fn open(name: &str) -> std::io::Result<PythonTextFile> {
        let file = File::open(name)?;
        PythonTextFile::from_file(file)
    }

    /// Wrap an already opened file (as `open` returns it).
    pub fn from_file(file: File) -> std::io::Result<PythonTextFile> {
        if file.metadata()?.is_dir() {
            return Err(std::io::Error::from_raw_os_error(libc::EISDIR));
        }
        Ok(PythonTextFile {
            reader: Box::new(BufReader::new(file)),
            universal_newlines: true,
        })
    }

    /// `sys.stdin`.
    pub fn stdin() -> PythonTextFile {
        PythonTextFile {
            reader: Box::new(std::io::stdin().lock()),
            universal_newlines: false,
        }
    }

    /// `f.readline()`: the next line including its `"\n"`, or `""` at end
    /// of file; `Err` for a read or decode error.
    pub fn readline(&mut self) -> Result<String, ()> {
        let mut bytes = Vec::new();
        if self.universal_newlines {
            loop {
                let buffer = self.reader.fill_buf().map_err(|_| ())?;
                if buffer.is_empty() {
                    break;
                }
                let byte = buffer[0];
                self.reader.consume(1);
                if byte == b'\n' {
                    bytes.push(b'\n');
                    break;
                }
                if byte == b'\r' {
                    bytes.push(b'\n');
                    let buffer = self.reader.fill_buf().map_err(|_| ())?;
                    if buffer.first() == Some(&b'\n') {
                        self.reader.consume(1);
                    }
                    break;
                }
                bytes.push(byte);
            }
            String::from_utf8(bytes).map_err(|_| ())
        } else {
            self.reader.read_until(b'\n', &mut bytes).map_err(|_| ())?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }
}

/// Take the option table out of its lock for indexing; Python subscripting
/// `sOptTable` after `PipDone` set it to `None` raises `TypeError`.
fn opt_table(guard: &mut Option<Vec<PipOption>>) -> &mut Vec<PipOption> {
    match guard {
        Some(table) => table,
        None => python_uncaught("TypeError: 'NoneType' object is not subscriptable"),
    }
}

/// Matches `PipInitialize` (`IMOD/pysrc/pip.py:101`).
pub fn pip_initialize(num_opts: i32) -> i32 {
    S_NUM_OPTIONS.store(num_opts, Relaxed);
    S_TABLE_SIZE.store(num_opts + 2, Relaxed);
    S_NON_OPT_IND.store(num_opts, Relaxed);
    let mut table = Vec::new();

    // Initialize the table
    for _ in 0..num_opts + 2 {
        table.push(PipOption::default());
    }

    // In the last slots, put non-option arguments, and also put the
    //  name for the standard input option for easy checking on duplication
    let non_opt_ind = num_opts as usize;
    table[non_opt_ind].long_name = NON_OPTION_STRING.to_owned();
    table[non_opt_ind + 1].short_name = STANDARD_INPUT_STRING.to_owned();
    table[non_opt_ind + 1].long_name = STANDARD_INPUT_END.to_owned();
    table[non_opt_ind].multiple = 1;
    *S_OPT_TABLE.lock().unwrap() = Some(table);
    0
}

/// Matches `PipWarnUnusedNonOptArgs` (`IMOD/pysrc/pip.py:123`).
pub fn pip_warn_unused_non_opt_args() -> i32 {
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    let non_opt = &table[S_NON_OPT_IND.load(Relaxed) as usize];
    let highest = S_HIGHEST_NON_OPT_GOTTEN.load(Relaxed);
    let unused = (non_opt.count - 1) - highest;
    if unused > 0 {
        print!("\nWARNING: Extra non-option arguments not used by the program:");
        for ind in highest + 1..non_opt.count {
            print!("  {}", non_opt.values[ind as usize]);
        }
        print!("\n\n");
    }
    unused
}

/// Matches `PipDone` (`IMOD/pysrc/pip.py:136`).
///
/// `sLinkedOption = None` there is a local assignment (the name is not in
/// the function's `global` list), so the linked option survives.
pub fn pip_done() {
    pip_warn_unused_non_opt_args();
    *S_OPT_TABLE.lock().unwrap() = None;
    S_TABLE_SIZE.store(0, Relaxed);
    S_NUM_OPTIONS.store(0, Relaxed);
    *S_ERROR_STRING.lock().unwrap() = None;
    S_NEXT_OPTION.store(0, Relaxed);
    S_NEXT_ARG_BELONGS_TO.store(-1, Relaxed);
    S_NUM_OPTION_ARGUMENTS.store(0, Relaxed);
    S_ALLOW_DEFAULTS.store(0, Relaxed);
}

/// Matches `PipExitOnError` (`IMOD/pysrc/pip.py:154`).
pub fn pip_exit_on_error(use_std_err: i32, prefix: &str) -> i32 {
    S_ERROR_DEST.store(use_std_err, Relaxed);
    *S_EXIT_PREFIX.lock().unwrap() = Some(prefix.to_owned());
    0
}

/// Matches `setExitPrefix` (`IMOD/pysrc/pip.py:163`).
pub fn set_exit_prefix(prefix: String) {
    *S_EXIT_PREFIX.lock().unwrap() = Some(prefix);
}

/// Matches `PipEnableEntryOutput` (`IMOD/pysrc/pip.py:169`).
pub fn pip_enable_entry_output(val: i32) {
    S_PRINT_ENTRIES.store(val, Relaxed);
}

/// Matches `PipSetLinkedOption` (`IMOD/pysrc/pip.py:175`).
pub fn pip_set_linked_option(option: &str) {
    *S_LINKED_OPTION.lock().unwrap() = Some(option.to_owned());
}

/// Matches `PipForbidComments` (`IMOD/pysrc/pip.py:181`).
pub fn pip_forbid_comments(long_name: &str, short_name: &str, warn: i32) {
    *S_FORBID_COMMENT_SHORT.lock().unwrap() = short_name.to_owned();
    *S_FORBID_COMMENT_LONG.lock().unwrap() = long_name.to_owned();
    S_WARN_ON_COMMENT.store(warn != 0, Relaxed);
}

/// Matches `PipGetErrNo` (`IMOD/pysrc/pip.py:189`).
pub fn pip_get_err_no() -> i32 {
    PIP_ERRNO.load(Relaxed)
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

/// Matches `PipAddOption` (`IMOD/pysrc/pip.py:220`).
pub fn pip_add_option(option_string: &str) -> i32 {
    let next_option = S_NEXT_OPTION.load(Relaxed);
    if next_option >= S_NUM_OPTIONS.load(Relaxed) {
        pip_set_error("Attempting to add more options than were originally specified");
        return -1;
    }

    let mut parts: Vec<String> = option_string.splitn(4, ':').map(String::from).collect();
    if parts.len() < 4 {
        pip_set_error(&format!(
            "Option does not have three colons in it:  {option_string}"
        ));
        return -1;
    }

    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    let next = next_option as usize;
    table[next].short_name = parts[0].clone();
    let new_slen = parts[0].chars().count() as i32;
    table[next].len_short = new_slen;
    table[next].long_name = parts[1].clone();

    // If type ends in M or L, set multiple flag and strip M or L
    if parts[2].ends_with('M') || parts[2].ends_with('L') {
        table[next].multiple = 1;
        table[next].linked = parts[2].ends_with('L');
        if parts[2].chars().count() > 1 {
            let length = parts[2].len();
            parts[2] = parts[2][0..length - 1].to_owned();
        } else {
            parts[2] = String::new();
        }
    }

    table[next].option_type = parts[2].clone();
    table[next].help_string = parts[3].clone();

    let non_opt_ind = S_NON_OPT_IND.load(Relaxed);
    for ind in 0..S_TABLE_SIZE.load(Relaxed) {
        // after checking existing ones, skip to NonOptionArg and
        // StandardInput entries
        if ind >= next_option && ind < non_opt_ind {
            continue;
        }

        let old_short = &table[ind as usize].short_name;
        let old_long = &table[ind as usize].long_name;
        let old_slen = table[ind as usize].len_short;
        if ((pip_starts_with(&parts[0], old_short) || pip_starts_with(old_short, &parts[0]))
            && ((new_slen > 1 && old_slen > 1) || (new_slen == 1 && old_slen == 1)))
            || pip_starts_with(old_long, &parts[0])
            || pip_starts_with(&parts[0], old_long)
            || pip_starts_with(old_short, &parts[1])
            || pip_starts_with(&parts[1], old_short)
            || pip_starts_with(old_long, &parts[1])
            || pip_starts_with(&parts[1], old_long)
        {
            let temp_str = format!(
                "Option {}  {} is ambiguous with option {}  {}",
                parts[0], parts[1], old_short, old_long
            );
            drop(guard);
            pip_set_error(&temp_str);
            print!("{temp_str}\n");
            return -1;
        }
    }

    S_NEXT_OPTION.store(next_option + 1, Relaxed);
    0
}

/// Matches `PipNextArg` (`IMOD/pysrc/pip.py:275`).
///
/// Returns `None` where the Python returns `None` (an `AddValueString`
/// failure on the linked-option lookup).
pub fn pip_next_arg(arg_string: &str) -> Option<i32> {
    // If we are expecting a value for an option, add string to the option
    let belongs_to = S_NEXT_ARG_BELONGS_TO.load(Relaxed);
    if belongs_to >= 0 {
        let mut err = add_value_string(belongs_to, arg_string);

        // Check whether this option was for reading from parameter file
        let is_param_file = {
            let mut guard = S_OPT_TABLE.lock().unwrap();
            opt_table(&mut guard)[belongs_to as usize].option_type == "PF"
        };
        if matches!(err, None | Some(0)) && is_param_file {
            let mut param_file = match PythonTextFile::open(arg_string) {
                Ok(file) => file,
                Err(_) => {
                    let temp_str = format!("Error opening parameter file {arg_string}");
                    pip_set_error(&temp_str);
                    return Some(-1);
                }
            };

            err = Some(read_param_file(&mut param_file));
            // `close(paramFile)` is a NameError swallowed by its bare
            // `except:`; the file is closed when it goes out of scope.
        }
        S_NEXT_ARG_BELONGS_TO.store(-1, Relaxed);
        return err;
    }

    // Is it a legal option starting with - or -- ?
    let stripped = arg_string.trim_matches(python_isspace);
    let Some(first) = stripped.chars().next() else {
        python_uncaught("IndexError: string index out of range");
    };
    let mut arg_string = arg_string;
    if first == '-' {
        arg_string = stripped;
        let characters: Vec<char> = arg_string.chars().collect();
        let mut ind_start = 1;
        if characters.len() > 1 && characters[1] == '-' {
            ind_start = 2;
        }
        if characters.len() == ind_start {
            pip_set_error("Illegal argument: - or --");
            return Some(-1);
        }
        let rest = &arg_string[ind_start..];

        // First check for StandardInput
        if STANDARD_INPUT_STRING.starts_with(rest) {
            let err = read_param_file(&mut PythonTextFile::stdin());
            return Some(err);
        }

        // Next check if it is a potential numeric non-option arg
        S_NOT_FOUND_OK.store(1, Relaxed);
        for ch in rest.chars() {
            // Python `str.isdigit` also accepts non-ASCII digits.
            if ch != '-' && ch != ',' && ch != '.' && ch != ' ' && !ch.is_ascii_digit() {
                S_NOT_FOUND_OK.store(0, Relaxed);
                break;
            }
        }

        // Lookup the option among true defined options
        let err = lookup_option(rest, S_NEXT_OPTION.load(Relaxed));

        // Process as an option unless it could be numeric and was not found
        if !(S_NOT_FOUND_OK.load(Relaxed) != 0 && err == LOOKUP_NOT_FOUND) {
            S_NOT_FOUND_OK.store(0, Relaxed);
            if err < 0 {
                return Some(err);
            }

            S_NUM_OPTION_ARGUMENTS.fetch_add(1, Relaxed);

            // For an option with value, setup to get argument next time and
            // return an indicator that there had better be another
            let is_boolean = {
                let mut guard = S_OPT_TABLE.lock().unwrap();
                opt_table(&mut guard)[err as usize].option_type == "B"
            };
            if !is_boolean {
                S_NEXT_ARG_BELONGS_TO.store(err, Relaxed);
                return Some(1);
            } else {
                // for a boolean option, set the argument with a 1
                return add_value_string(err, "1");
            }
        }
    }

    // A non-option argument
    S_NOT_FOUND_OK.store(0, Relaxed);
    let (expanded, _no_match_ind) = expand_arg_list(&[OsString::from(arg_string)]);
    for arg in expanded {
        let err = add_value_string(S_NON_OPT_IND.load(Relaxed), &arg.to_string_lossy());
        if err.is_none() {
            return None;
        }
    }
    Some(0)
}

/// Matches `PipNumberOfArgs` (`IMOD/pysrc/pip.py:356`).
pub fn pip_number_of_args() -> (i32, i32) {
    PIP_ERRNO.store(0, Relaxed);
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    (
        S_NUM_OPTION_ARGUMENTS.load(Relaxed),
        table[S_NON_OPT_IND.load(Relaxed) as usize].count,
    )
}

/// Matches `PipGetNonOptionArg` (`IMOD/pysrc/pip.py:363`).  `Err(-1)` is
/// the Python's `None`.
pub fn pip_get_non_option_arg(arg_no: i32) -> Result<String, i32> {
    PIP_ERRNO.store(0, Relaxed);
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    let non_opt = &table[S_NON_OPT_IND.load(Relaxed) as usize];
    if arg_no >= non_opt.count {
        drop(guard);
        pip_set_error("Requested a non-option argument beyond the number available");
        PIP_ERRNO.store(-1, Relaxed);
        return Err(-1);
    }
    S_HIGHEST_NON_OPT_GOTTEN.store(S_HIGHEST_NON_OPT_GOTTEN.load(Relaxed).max(arg_no), Relaxed);
    // A negative index counts from the end in Python.
    let index = if arg_no < 0 {
        non_opt.count + arg_no
    } else {
        arg_no
    };
    if index < 0 {
        drop(guard);
        python_uncaught("IndexError: list index out of range");
    }
    Ok(non_opt.values[index as usize].clone())
}

/// Matches `PipGetString` (`IMOD/pysrc/pip.py:377`).  `Err` is the
/// Python's `None` (with `pipErrno` holding the code).
pub fn pip_get_string(option: &str, string: &str) -> Result<String, i32> {
    PIP_ERRNO.store(0, Relaxed);
    let retval = get_next_value_string(option);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(string.to_owned());
    }
    Ok(retval.unwrap_or_default())
}

/// Matches `PipGetBoolean` (`IMOD/pysrc/pip.py:390`).
pub fn pip_get_boolean(option: &str, val: i32) -> Result<i32, i32> {
    PIP_ERRNO.store(0, Relaxed);
    let str_ptr = get_next_value_string(option);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno > 0 {
        return Ok(val);
    }
    let str_ptr = str_ptr.unwrap_or_default();
    if matches!(
        str_ptr.as_str(),
        "1" | "T" | "TRUE" | "ON" | "t" | "true" | "on"
    ) {
        Ok(1)
    } else if matches!(
        str_ptr.as_str(),
        "0" | "F" | "FALSE" | "OFF" | "f" | "false" | "off"
    ) {
        Ok(0)
    } else {
        let temp_str = format!("Illegal entry for boolean option {option}: {str_ptr}");
        pip_set_error(&temp_str);
        PIP_ERRNO.store(-1, Relaxed);
        Err(-1)
    }
}

/// Matches `PipGetInteger` (`IMOD/pysrc/pip.py:414`).  The Python int is
/// returned as `i32` (see [`python_int`]).
pub fn pip_get_integer(option: &str, val: i32) -> Result<i32, i32> {
    let num = 1;
    let retval = pip_get_integer_array(option, num);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(val);
    }
    Ok(retval.unwrap_or_default()[0] as i32)
}

/// Matches `PipGetFloat` (`IMOD/pysrc/pip.py:424`).  A Python float is a
/// double.
pub fn pip_get_float(option: &str, val: f64) -> Result<f64, i32> {
    let num = 1;
    let retval = pip_get_float_array(option, num);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(val);
    }
    Ok(retval.unwrap_or_default()[0])
}

/// Matches `PipGetTwoIntegers` (`IMOD/pysrc/pip.py:437`).
pub fn pip_get_two_integers(option: &str, val: (i32, i32)) -> Result<(i32, i32), i32> {
    let num = 2;
    let retval = pip_get_integer_array(option, num);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(val);
    }
    let retval = retval.unwrap_or_default();
    Ok((retval[0] as i32, retval[1] as i32))
}

/// Matches `PipGetTwoFloats` (`IMOD/pysrc/pip.py:447`).
pub fn pip_get_two_floats(option: &str, val: (f64, f64)) -> Result<(f64, f64), i32> {
    let num = 2;
    let retval = pip_get_float_array(option, num);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(val);
    }
    let retval = retval.unwrap_or_default();
    Ok((retval[0], retval[1]))
}

/// Matches `PipGetThreeIntegers` (`IMOD/pysrc/pip.py:460`).
pub fn pip_get_three_integers(option: &str, val: (i32, i32, i32)) -> Result<(i32, i32, i32), i32> {
    let num = 3;
    let retval = pip_get_integer_array(option, num);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(val);
    }
    let retval = retval.unwrap_or_default();
    Ok((retval[0] as i32, retval[1] as i32, retval[2] as i32))
}

/// Matches `PipGetThreeFloats` (`IMOD/pysrc/pip.py:470`).
pub fn pip_get_three_floats(option: &str, val: (f64, f64, f64)) -> Result<(f64, f64, f64), i32> {
    let num = 3;
    let retval = pip_get_float_array(option, num);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno < 0 {
        return Err(errno);
    }
    if errno == 1 {
        return Ok(val);
    }
    let retval = retval.unwrap_or_default();
    Ok((retval[0], retval[1], retval[2]))
}

/// Matches `PipGetIntegerArray` (`IMOD/pysrc/pip.py:484`).  `numToGet` 0
/// returns every value on the line.
pub fn pip_get_integer_array(option: &str, num_to_get: usize) -> Option<Vec<i64>> {
    match option_line_of_values(option, PIP_INTEGER, num_to_get) {
        Some(PipValues::Integers(values)) => Some(values),
        _ => None,
    }
}

/// Matches `PipGetFloatArray` (`IMOD/pysrc/pip.py:488`).
pub fn pip_get_float_array(option: &str, num_to_get: usize) -> Option<Vec<f64>> {
    match option_line_of_values(option, PIP_FLOAT, num_to_get) {
        Some(PipValues::Floats(values)) => Some(values),
        _ => None,
    }
}

/// Matches `PipPrintHelp` (`IMOD/pysrc/pip.py:495`).
pub fn pip_print_help(
    prog_name: &str,
    use_std_err: i32,
    input_files: i32,
    output_files: i32,
) -> i32 {
    let mut num_real = 0;
    let helplim = 74usize;
    let mut out: Box<dyn Write> = if use_std_err != 0 {
        Box::new(std::io::stderr())
    } else {
        Box::new(std::io::stdout())
    };
    let indent4 = "    ";
    let num_options = S_NUM_OPTIONS.load(Relaxed);
    let output_manpage = S_OUTPUT_MANPAGE.load(Relaxed);
    let mut descriptions: Option<&[&str]> = None;
    // The table is only read here; a snapshot keeps the lock free for the
    // `LookupOption` calls below.
    let table = {
        let mut guard = S_OPT_TABLE.lock().unwrap();
        opt_table(&mut guard).clone()
    };

    for i in 0..num_options as usize {
        if !table[i].short_name.is_empty() || !table[i].long_name.is_empty() {
            num_real += 1;
        }
    }

    if output_manpage == 0 {
        let _ = write!(out, "Usage: {prog_name} ");
        if num_options != 0 {
            let _ = write!(out, "[Options]");
        }
        if input_files != 0 {
            let _ = write!(out, " input_file");
        }
        if input_files > 1 {
            let _ = write!(out, "s...");
        }
        if output_files != 0 {
            let _ = write!(out, " output_file");
        }
        if output_files > 1 {
            let _ = write!(out, "s...");
        }
        let _ = write!(out, "\n");

        if num_real == 0 {
            return 0;
        }
        let _ = write!(
            out,
            "Options can be abbreviated, current short name abbreviations are in parentheses\n"
        );
        let _ = write!(out, "Options:\n");
        descriptions = Some(&S_TYPE_FOR_USAGE);
    }

    S_TEST_ABBREV_FOR_USAGE.store(true, Relaxed);
    for i in 0..num_options as usize {
        let sname = table[i].short_name.clone();
        let lname = table[i].long_name.clone();
        let mut indent_str = "";

        // Try to look up an abbreviation of the short name
        let mut abbrev: Option<String> = None;
        let sname_chars: Vec<char> = sname.chars().collect();
        if !sname_chars.is_empty() {
            for j in 1..sname_chars.len() {
                let prefix = python_slice(&sname_chars, 0, j);
                if lookup_option(&prefix, num_options) == i as i32 {
                    abbrev = Some(prefix);
                    break;
                }
            }
        }

        if !lname.is_empty() || !sname.is_empty() {
            if output_manpage <= 0 {
                indent_str = indent4;
            }

            let _ = write!(out, " ");
            if !sname.is_empty() {
                let _ = write!(out, "-{sname}");
            }
            if let Some(abbrev) = &abbrev {
                let _ = write!(out, " (-{abbrev})");
            }
            if !sname.is_empty() && !lname.is_empty() {
                let _ = write!(out, "  OR  ");
            }
            if !lname.is_empty() {
                let _ = write!(out, "-{lname}");
            }
            let mut jj = 0;
            for j in 0..S_NUM_TYPES {
                jj = j;
                if table[i].option_type == S_TYPES[j] {
                    break;
                }
            }

            if table[i].option_type != "B" {
                let Some(descriptions) = descriptions else {
                    python_uncaught(
                        "UnboundLocalError: cannot access local variable 'descriptions'",
                    );
                };
                let mut format = descriptions[jj].to_owned();
                if !table[i].format.is_empty() {
                    format = table[i]
                        .format
                        .replace("\\fR", "")
                        .replace("\\fI", "")
                        .replace("\\fB", "");
                }
                let _ = write!(out, "   {format}");
            }
        }

        let _ = write!(out, "\n");

        // Print help string, breaking up line as needed
        if !table[i].help_string.is_empty() {
            let mut sname = table[i].help_string.clone();
            if !table[i].default_val.is_empty() {
                sname = sname.replace(DEFAULT_SUB_STR, &table[i].default_val);
            }
            let mut sname: Vec<char> = sname.chars().collect();
            let mut opt_len = sname.len();
            let mut new_line_pt = sname.iter().position(|&c| c == '\n');
            while opt_len > helplim || new_line_pt.is_some() {
                // Break string at newline
                // Or break string at last space before limit
                let j = match new_line_pt {
                    Some(point) if point <= helplim => point,
                    _ => {
                        // `sname.rfind(' ', 1, helplim)`
                        let end = helplim.min(sname.len());
                        let mut found = 0;
                        if end > 1 {
                            if let Some(position) = sname[1..end].iter().rposition(|&c| c == ' ') {
                                found = position + 1;
                            }
                        }
                        found
                    }
                };

                let _ = write!(out, "{indent_str}{}\n", python_slice(&sname, 0, j));
                sname = if j + 1 <= sname.len() {
                    sname[j + 1..].to_vec()
                } else {
                    Vec::new()
                };
                new_line_pt = sname.iter().position(|&c| c == '\n');
                opt_len = opt_len.wrapping_sub(j + 1);
            }

            let _ = write!(out, "{indent_str}{}\n", sname.iter().collect::<String>());
        }

        if table[i].multiple != 0 {
            let _ = write!(out, "{indent_str}(Successive entries accumulate)\n");
        }
    }

    S_TEST_ABBREV_FOR_USAGE.store(false, Relaxed);
    let _ = out.flush();
    0
}

/// Matches `PipPrintEntries` (`IMOD/pysrc/pip.py:608`).
pub fn pip_print_entries() {
    if S_PRINT_ENTRIES.load(Relaxed) < 0 {
        S_PRINT_ENTRIES.store(0, Relaxed);
        if let Ok(val) = std::env::var(PRINTENTRY_VARIABLE) {
            S_PRINT_ENTRIES.store(python_int(&val).map_or(0, |value| value as i32), Relaxed);
        }
    }
    if S_PRINT_ENTRIES.load(Relaxed) == 0 {
        return;
    }
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    print!(
        "\n*** Entries to program {} ***\n",
        S_PROGRAM_NAME.lock().unwrap()
    );
    for i in 0..S_NUM_OPTIONS.load(Relaxed) as usize {
        let sname = &table[i].short_name;
        let lname = &table[i].long_name;
        if (!lname.is_empty() || !sname.is_empty()) && table[i].count != 0 {
            let mut name = lname;
            if lname.is_empty() {
                name = sname;
            }
            for j in 0..table[i].count as usize {
                print!("  {name} = {}\n", table[i].values[j]);
            }
        }
    }
    let non_opt = &table[S_NON_OPT_IND.load(Relaxed) as usize];
    if non_opt.count != 0 {
        print!("  Non-option arguments:");
        for j in 0..non_opt.count as usize {
            if non_opt.values[j].contains(' ') {
                print!("   \"{}\"", non_opt.values[j]);
            } else {
                print!("   {}", non_opt.values[j]);
            }
        }
        print!("\n");
    }
    print!("*** End of entries ***\n\n");
    let _ = std::io::stdout().flush();
}

/// Matches `PipGetError` (`IMOD/pysrc/pip.py:645`).
pub fn pip_get_error() -> String {
    PIP_ERRNO.store(0, Relaxed);
    let error_string = S_ERROR_STRING.lock().unwrap().clone();
    match error_string {
        Some(error_string) if !error_string.is_empty() => error_string,
        _ => {
            PIP_ERRNO.store(-1, Relaxed);
            String::new()
        }
    }
}

/// Matches `PipSetError` (`IMOD/pysrc/pip.py:657`).
///
/// With `sErrorDest` set, the source's `outFile = stderr` names an undefined
/// global and raises `NameError` before anything else happens.  Returns -1
/// where the Python returns `None` (no message and no prefix).
pub fn pip_set_error(err_string: &str) -> i32 {
    if S_ERROR_DEST.load(Relaxed) != 0 {
        python_uncaught("NameError: name 'stderr' is not defined");
    }
    *S_ERROR_STRING.lock().unwrap() = Some(err_string.to_owned());
    let exit_prefix = S_EXIT_PREFIX.lock().unwrap().clone().unwrap_or_default();
    if err_string.is_empty() && exit_prefix.is_empty() {
        // `pipErrno = -1` there is a local assignment.
        return -1;
    }

    if !exit_prefix.is_empty() {
        let mut error_string = err_string.to_owned();
        if error_string.is_empty() {
            error_string = "Unspecified error".to_owned();
            *S_ERROR_STRING.lock().unwrap() = Some(error_string.clone());
        }
        print!("{exit_prefix}{error_string}\n");
        let _ = std::io::stdout().flush();
        std::process::exit(1);
    }

    0
}

/// Matches `exitError` (`IMOD/pysrc/pip.py:678`).
pub fn exit_error(error_mess: &str) -> ! {
    pip_set_error(error_mess);
    let _ = std::io::stdout().flush();
    std::process::exit(1)
}

/// Matches `PipNumberOfEntries` (`IMOD/pysrc/pip.py:685`).
pub fn pip_number_of_entries(option: &str) -> Result<i32, i32> {
    PIP_ERRNO.store(0, Relaxed);
    let err = lookup_option(option, S_NON_OPT_IND.load(Relaxed) + 1);
    if err < 0 {
        PIP_ERRNO.store(err, Relaxed);
        return Err(err);
    }
    let mut guard = S_OPT_TABLE.lock().unwrap();
    Ok(opt_table(&mut guard)[err as usize].count)
}

/// Matches `PipLinkedIndex` (`IMOD/pysrc/pip.py:699`).  Unlike the C, it
/// does not check that the option is linked; indexing an empty
/// `nextLinked` raises `IndexError`.
pub fn pip_linked_index(option: &str) -> Result<i32, i32> {
    PIP_ERRNO.store(0, Relaxed);
    let num_options = S_NUM_OPTIONS.load(Relaxed);
    let err = lookup_option(option, num_options);
    if err < 0 {
        PIP_ERRNO.store(err, Relaxed);
        return Err(err);
    }

    let mut ind = 0;
    let multiple = {
        let mut guard = S_OPT_TABLE.lock().unwrap();
        opt_table(&mut guard)[err as usize].multiple
    };
    if multiple != 0 {
        ind = multiple - 1;
    }

    // Set up to use count from non-option args, but use the count from the linked option
    // instead if it was entered at all.  This allows other non-option args to be used
    let mut which = 0;
    let linked_option = S_LINKED_OPTION.lock().unwrap().clone();
    if let Some(linked_option) = linked_option {
        let ilink = lookup_option(&linked_option, num_options);
        if ilink < 0 {
            PIP_ERRNO.store(ilink, Relaxed);
            return Err(ilink);
        }
        let mut guard = S_OPT_TABLE.lock().unwrap();
        if opt_table(&mut guard)[ilink as usize].count != 0 {
            which = 1;
        }
    }
    let mut guard = S_OPT_TABLE.lock().unwrap();
    match opt_table(&mut guard)[err as usize]
        .next_linked
        .get((2 * ind + which) as usize)
    {
        Some(&value) => Ok(value),
        None => {
            drop(guard);
            python_uncaught("IndexError: list index out of range")
        }
    }
}

/// Matches `PipParseInput` (`IMOD/pysrc/pip.py:726`).  `Err` is the
/// Python's `None`; its `pipErrno = err` assignments are local (the name is
/// not declared global there), so a failed `PipAddOption` leaves the global
/// untouched.
pub fn pip_parse_input(argv: &[String], options: &[String]) -> Result<(i32, i32), i32> {
    // Initialize
    let num_opts = options.len() as i32;
    let err = pip_initialize(num_opts);
    if err != 0 {
        return Err(err);
    }

    // add the options
    for option in options {
        let err = pip_add_option(option);
        if err != 0 {
            return Err(err);
        }
    }

    pip_parse_entries(argv)
}

/// Matches `PipOpenInstalledAdoc` (`IMOD/pysrc/pip.py:747`).
pub fn pip_open_installed_adoc(prog_name: &str) -> Option<File> {
    let mut opt_file: Option<File> = None;
    if let Ok(pip_dir) = std::env::var(OPTDIR_VARIABLE)
        && !pip_dir.is_empty()
    {
        let big_str = format!("{pip_dir}{PATH_SEPARATOR}{prog_name}.{OPTFILE_EXT}");
        opt_file = File::open(&big_str)
            .ok()
            .filter(|file| file.metadata().is_ok_and(|meta| !meta.is_dir()));
    }

    if opt_file.is_none()
        && let Ok(pip_dir) = std::env::var("IMOD_DIR")
        && !pip_dir.is_empty()
    {
        let big_str = format!(
            "{pip_dir}{PATH_SEPARATOR}{OPTFILE_DIR}{PATH_SEPARATOR}{prog_name}.{OPTFILE_EXT}"
        );
        opt_file = File::open(&big_str)
            .ok()
            .filter(|file| file.metadata().is_ok_and(|meta| !meta.is_dir()));
    }

    opt_file
}

/// Matches `PipReadOptionFile` (`IMOD/pysrc/pip.py:779`).
pub fn pip_read_option_file(prog_name: &str, help_level: i32, local_dir: i32) -> i32 {
    let mut opt_file: Option<PythonTextFile> = None;
    *S_PROGRAM_NAME.lock().unwrap() = prog_name.to_owned();

    // If local directory not set, look for environment variable pointing
    // directly to where the file should be
    if local_dir == 0 {
        opt_file = pip_open_installed_adoc(prog_name)
            .and_then(|file| PythonTextFile::from_file(file).ok());

    // If local directory set, set up name with ../ as many times as specified
    // and look for file there
    } else if local_dir > 0 {
        let mut big_str = String::new();
        for _ in 0..local_dir {
            // The source indents the name building and the open inside this
            // loop, so the path accumulates on each pass.
            big_str += "..";
            big_str += PATH_SEPARATOR;
            big_str += &format!("{OPTFILE_DIR}{PATH_SEPARATOR}{prog_name}.{OPTFILE_EXT}");
            opt_file = PythonTextFile::open(&big_str).ok();
        }
    }

    // If there is still no file, look in current directory
    if opt_file.is_none() {
        let big_str = format!("{prog_name}.{OPTFILE_EXT}");
        opt_file = PythonTextFile::open(&big_str).ok();

        if opt_file.is_none() {
            let big_str = format!(
                "Autodoc file {prog_name}.{OPTFILE_EXT} was not found or not readable.\nCheck environment variable settings of {OPTDIR_VARIABLE} and IMOD_DIR\nor place autodoc file in current directory"
            );
            pip_set_error(&big_str);
            return -1;
        }
    }
    let mut opt_file = opt_file.unwrap();

    let mut num_opts = 0;
    let mut opt_list: Vec<String> = Vec::new();
    let mut format_list: Vec<String> = Vec::new();
    let mut default_list: Vec<String> = Vec::new();
    let mut long_name = String::new();
    let mut short_name = String::new();
    let mut option_type = String::new();
    let mut help_str = [String::new(), String::new(), String::new()];
    let mut format_str = String::new();
    let mut default_str = String::new();
    let mut reading_opt = 0;
    let mut in_quote_index: i32 = -1;
    let mut is_section = 0;
    let mut last_ind: i32 = 0;

    loop {
        let (line_len, big_str, indst, _bad_comment) = pip_read_next_line(&mut opt_file, '#', 0, 0);
        if line_len == -2 {
            pip_set_error("Error reading autodoc file");
            return -1;
        }

        // Count up option entries
        let big_chars: Vec<char> = big_str.chars().collect();
        let text_str = python_slice(&big_chars, indst, big_chars.len());
        let is_option = line_is_option_token(&text_str);
        if is_option != 0 {
            num_opts += 1;
        }

        // Look for new keyword-value delimiter before any options
        if num_opts == 0 {
            let new_delim;
            (new_delim, last_ind, in_quote_index) =
                check_keyword(&text_str, "KeyValueDelimiter", 0, 0);
            if !new_delim.is_empty() {
                *S_VALUE_DELIM.lock().unwrap() = Some(new_delim);
            }
        }

        if reading_opt != 0 && (line_len == -3 || is_option != 0) {
            // If we were reading options, it is time to add them if we are at
            // end of file or if we have reached a new token of any kind

            // Pick the closest help string that was read in if the given one
            // does not match (there has got to be an easier way!)
            let help_ind = if help_level <= 1 {
                if !help_str[0].is_empty() {
                    0
                } else if !help_str[1].is_empty() {
                    1
                } else {
                    2
                }
            } else if help_level == 2 {
                if !help_str[1].is_empty() {
                    1
                } else if !help_str[0].is_empty() {
                    0
                } else {
                    2
                }
            } else if !help_str[2].is_empty() {
                2
            } else if !help_str[1].is_empty() {
                1
            } else {
                0
            };

            // If it is a section header, get rid of the names
            if is_section != 0 {
                long_name = String::new();
                short_name = String::new();
            }

            let opt_str = format!(
                "{short_name}:{long_name}:{option_type}:{}",
                help_str[help_ind]
            );
            opt_list.push(opt_str);
            format_list.push(format_str.clone());
            default_list.push(default_str.clone());

            // Clean up
            long_name = String::new();
            short_name = String::new();
            option_type = String::new();
            help_str = [String::new(), String::new(), String::new()];
            format_str = String::new();
            default_str = String::new();
            reading_opt = 0;
        }

        if line_len == -3 {
            break;
        }

        // If reading options, look for the various keywords
        if reading_opt != 0 {
            let value_delim = S_VALUE_DELIM
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| VALUE_DELIM.to_owned());
            // If the last string gotten was a help string and the line does not contain the
            // value delimiter or we are in a quote, then append it to the last string
            if last_ind != 0 && (in_quote_index >= 0 || !text_str.contains(value_delim.as_str())) {
                let last = (last_ind - 1) as usize;
                if help_str[last].ends_with('.') {
                    help_str[last] += "  ";
                } else {
                    help_str[last] += " ";
                }

                // Replace leading ^ with a newline
                let mut text_str = text_str.clone();
                if text_str.starts_with('^') {
                    text_str = text_str.replacen('^', "\n", 1);
                }

                // If inside quotes, look for quote at end and say it is the end of accepting
                // continuation lines, and as a protection, also say a blank line ends it
                let text_chars: Vec<char> = text_str.chars().collect();
                let lentx = text_chars.len();
                if in_quote_index >= 0
                    && (lentx == 0
                        || text_chars[lentx - 1] == S_QUOTE_TYPES[in_quote_index as usize])
                {
                    if lentx != 0 {
                        help_str[last] += &python_slice(&text_chars, 0, lentx - 1);
                    }
                    last_ind = 0;
                    in_quote_index = -1;
                } else {
                    help_str[last] += &text_str;
                }

            // Otherwise look for each keyword of interest, but zero last index
            } else {
                last_ind = 0;
                let retval = check_keyword(&text_str, "short", 0, 0);
                if !retval.0.is_empty() {
                    (short_name, last_ind, in_quote_index) = retval;
                }
                let retval = check_keyword(&text_str, "long", 0, 0);
                if !retval.0.is_empty() {
                    (long_name, last_ind, in_quote_index) = retval;
                }
                let retval = check_keyword(&text_str, "type", 0, 0);
                if !retval.0.is_empty() {
                    (option_type, last_ind, in_quote_index) = retval;
                }
                let retval = check_keyword(&text_str, "format", 0, 0);
                if !retval.0.is_empty() {
                    (format_str, last_ind, in_quote_index) = retval;
                }
                let retval = check_keyword(&text_str, "default", 0, 0);
                if !retval.0.is_empty() {
                    (default_str, last_ind, in_quote_index) = retval;
                }

                // Check for usage if at help level 1 or if we haven't got
                // either of the other strings yet
                if help_level <= 1 || !(!help_str[1].is_empty() || !help_str[2].is_empty()) {
                    let retval = check_keyword(&text_str, "usage", 1, 1);
                    if !retval.0.is_empty() {
                        (help_str[0], last_ind, in_quote_index) = retval;
                    }
                }

                // Check for tooltip if at level 2 or if at level 1 and haven't
                // got usage, or at level 3 and haven't got manpage
                if help_level == 2
                    || (help_level <= 1 && help_str[0].is_empty())
                    || (help_level >= 3 && help_str[2].is_empty())
                {
                    let retval = check_keyword(&text_str, "tooltip", 2, 1);
                    if !retval.0.is_empty() {
                        (help_str[1], last_ind, in_quote_index) = retval;
                    }
                }

                // Check for manpage if at level 3 or if at level 2 and haven't
                // got tip, or at level 1 and haven't got tip or usage
                if help_level >= 3
                    || (help_level == 2 && help_str[1].is_empty())
                    || (help_level <= 1 && !(!help_str[1].is_empty() || !help_str[0].is_empty()))
                {
                    let retval = check_keyword(&text_str, "manpage", 3, 1);
                    if !retval.0.is_empty() {
                        (help_str[2], last_ind, in_quote_index) = retval;
                    }
                }

                // If that was a line with quoted string, check for quote at end of line and
                // close out the help string if so */
                if in_quote_index >= 0 && last_ind != 0 {
                    let last = (last_ind - 1) as usize;
                    let help_chars: Vec<char> = help_str[last].chars().collect();
                    let lentx = help_chars.len();
                    if lentx != 0 && help_chars[lentx - 1] == S_QUOTE_TYPES[in_quote_index as usize]
                    {
                        help_str[last] = python_slice(&help_chars, 0, lentx - 1);
                        in_quote_index = -1;
                        last_ind = 0;
                    }
                }
            }

        // But if not reading options, check for a new option token and start
        // reading if one is found.  But first take a Field value as default
        // long option name
        } else if is_option > 0 {
            last_ind = 0;
            in_quote_index = -1;
            reading_opt = 1;
            is_section = is_option - 1;
            if is_section == 0 {
                let text_chars: Vec<char> = text_str.chars().collect();
                let retval = check_keyword(
                    &python_slice(&text_chars, OPEN_DELIM.len(), text_chars.len()),
                    "Field",
                    0,
                    0,
                );
                if !retval.0.is_empty() {
                    let field: Vec<char> = retval.0.chars().collect();
                    long_name = python_slice(&field, 0, field.len() - 1)
                        .trim_end_matches(python_isspace)
                        .to_owned();
                }
            }
        }
    }

    // Initialize and process option strings
    pip_initialize(num_opts);

    // add the options
    for i in 0..num_opts as usize {
        let err = pip_add_option(&opt_list[i]);
        if err != 0 {
            return err;
        }
        let mut guard = S_OPT_TABLE.lock().unwrap();
        let table = opt_table(&mut guard);
        // `sOptTable[sNextOption - 1]`: a Python -1 index is the last slot.
        let next = S_NEXT_OPTION.load(Relaxed) - 1;
        let slot = if next < 0 {
            (table.len() as i32 + next) as usize
        } else {
            next as usize
        };
        table[slot].format = format_list[i].clone();
        table[slot].default_val = default_list[i].clone();
    }

    0
}

/// Matches `PipParseEntries` (`IMOD/pysrc/pip.py:1014`).  `Err` is the
/// Python's `None`.
pub fn pip_parse_entries(argv: &[String]) -> Result<(i32, i32), i32> {
    PIP_ERRNO.store(0, Relaxed);
    let argc = argv.len();

    // Special case: no arguments and flag set to take stdin automatically
    if argc == 0 && S_TAKE_STD_IN.load(Relaxed) != 0 {
        let err = read_param_file(&mut PythonTextFile::stdin());
        if err != 0 {
            PIP_ERRNO.store(err, Relaxed);
            return Err(err);
        }
    } else {
        // parse the arguments
        for i in 1..argc {
            let Some(err) = pip_next_arg(&argv[i]) else {
                python_uncaught(
                    "TypeError: '<' not supported between instances of 'NoneType' and 'int'",
                );
            };
            if err < 0 {
                PIP_ERRNO.store(err, Relaxed);
                return Err(err);
            }
            if err != 0 && i == argc - 1 {
                pip_set_error(
                    "A value was expected but not found for the last option on the command line",
                );
                PIP_ERRNO.store(-1, Relaxed);
                return Err(-1);
            }
        }
    }

    pip_print_entries();
    Ok(pip_number_of_args())
}

/// Matches `PipReadOrParseOptions` (`IMOD/pysrc/pip.py:1046`).
pub fn pip_read_or_parse_options(
    argv: &[String],
    options: &[String],
    prog_name: &str,
    min_args: i32,
    num_in_files: i32,
    num_out_files: i32,
) -> (i32, i32) {
    // Startup with fallback
    let ierr = pip_read_option_file(prog_name, 0, 0);
    pip_exit_on_error(0, &format!("ERROR: {prog_name} - "));
    let (num_opt_args, num_non_opt_args);
    let unpack = "TypeError: cannot unpack non-iterable NoneType object";
    if ierr == 0 {
        (num_opt_args, num_non_opt_args) =
            pip_parse_entries(argv).unwrap_or_else(|_| python_uncaught(unpack));
    } else {
        let err_string = pip_get_error();
        if options.is_empty() {
            pip_set_error(&err_string);
        }
        if !err_string.is_empty() {
            print!("PIP WARNING: {err_string}\nUsing fallback options in main program\n\n");
        }

        (num_opt_args, num_non_opt_args) =
            pip_parse_input(argv, options).unwrap_or_else(|_| python_uncaught(unpack));
    }

    // Output usage and exit if not enough arguments
    let exit_save = S_EXIT_PREFIX.lock().unwrap().take();
    let help = pip_get_boolean("help", 0);
    *S_EXIT_PREFIX.lock().unwrap() = exit_save;
    if matches!(help, Ok(value) if value != 0) || num_opt_args + num_non_opt_args < min_args {
        pip_print_help(prog_name, 0, num_in_files, num_out_files);
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }

    if ierr == 0 {
        return (num_opt_args, num_non_opt_args);
    }

    // If no autodoc found, open the master list of program defaults
    let Ok(pip_dir) = std::env::var("IMOD_DIR") else {
        return (num_opt_args, num_non_opt_args);
    };
    if pip_dir.is_empty() {
        return (num_opt_args, num_non_opt_args);
    }
    let Ok(mut def_file) = PythonTextFile::open(&format!(
        "{pip_dir}{PATH_SEPARATOR}{DEFAULTS_DIR}{PATH_SEPARATOR}{DEFAULTS_FILE}"
    )) else {
        // `except IOError: pass`
        return (num_opt_args, num_non_opt_args);
    };

    // Read lines, determine when enter and leave the program section, add lines
    // to dictionary
    let prog_match = regex::Regex::new(&format!(r"\[\s*Program\s*=\s*{prog_name}\s*\]"))
        .unwrap_or_else(|_| python_uncaught("re.error: bad pattern"));
    let mut in_prog = false;
    let mut def_dict: Vec<(String, String)> = Vec::new();
    loop {
        let Ok(line) = def_file.readline() else {
            // A decoding error is not an IOError and is not caught.
            python_uncaught("UnicodeDecodeError: 'utf-8' codec can't decode byte");
        };
        if line.is_empty() {
            break;
        }
        let line = line.trim_matches(python_isspace);
        if prog_match.is_match(line) {
            if in_prog {
                break;
            }
            in_prog = true;
        } else if in_prog {
            let lsplit: Vec<&str> = line.split('=').collect();
            if lsplit.len() == 2 {
                let key = lsplit[0].trim_matches(python_isspace).to_owned();
                let value = lsplit[1].trim_matches(python_isspace).to_owned();
                match def_dict.iter_mut().find(|(existing, _)| *existing == key) {
                    Some(entry) => entry.1 = value,
                    None => def_dict.push((key, value)),
                }
            }
        }
    }

    // Look up each option in dictionary and assign default if found
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    for ind in 0..S_NUM_OPTIONS.load(Relaxed) as usize {
        if let Some((_, value)) = def_dict
            .iter()
            .find(|(key, _)| *key == table[ind].long_name)
        {
            table[ind].default_val = value.clone();
        }
    }

    (num_opt_args, num_non_opt_args)
}

/// Matches `PipGetInOutFile` (`IMOD/pysrc/pip.py:1120`).  `Ok(None)` is the
/// Python's `None`.
///
/// Note the source tests `pipErrno`, which is 2 when `PipGetString` returned
/// a default from `progDefaults.adoc`, so a defaulted option falls through to
/// the non-option argument.
pub fn pip_get_in_out_file(option: &str, non_opt_arg_no: i32) -> Result<Option<String>, i32> {
    let retval = pip_get_string(option, "");
    if PIP_ERRNO.load(Relaxed) == 0 {
        return Ok(retval.ok());
    }
    let count = {
        let mut guard = S_OPT_TABLE.lock().unwrap();
        opt_table(&mut guard)[S_NON_OPT_IND.load(Relaxed) as usize].count
    };
    if non_opt_arg_no >= count {
        PIP_ERRNO.store(1, Relaxed);
        return Ok(None);
    }
    Ok(pip_get_non_option_arg(non_opt_arg_no).ok())
}

/// Matches `ReadParamFile` (`IMOD/pysrc/pip.py:1134`).
///
/// Fixed in translation (BUGS.md): the token end is found on
/// `lineStr[indst:]` but native uses it as an index into `lineStr`
/// (`pip.py:1164-1168`), so on an indented line the token is cut short and
/// the value starts inside the option name; here the offset is taken
/// relative to `indst`, as `parse_params.c:1696` does.
pub fn read_param_file(p_file: &mut PythonTextFile) -> i32 {
    loop {
        // If non-option lines are allowed, set flag that it is OK for
        // LookupOption to not find the option, but only for the given number
        // of lines at the start of the input
        let non_opt_count = {
            let mut guard = S_OPT_TABLE.lock().unwrap();
            opt_table(&mut guard)[S_NON_OPT_IND.load(Relaxed) as usize].count
        };
        S_NOT_FOUND_OK.store(
            (S_NUM_OPTION_ARGUMENTS.load(Relaxed) == 0
                && non_opt_count < S_NON_OPT_LINES.load(Relaxed)) as i32,
            Relaxed,
        );
        let (line_len, line_str, indst, bad_comment) = pip_read_next_line(p_file, '#', 0, 1);
        if line_len == -3 {
            break;
        }
        if line_len == -2 {
            pip_set_error(&format!(
                "Error reading parameter file or {STANDARD_INPUT_STRING}"
            ));
            return -1;
        }
        if !bad_comment.is_empty() {
            if S_WARN_ON_COMMENT.load(Relaxed) {
                print!("PIP WARNING: {bad_comment}");
            } else {
                pip_set_error(&bad_comment);
                return -1;
            }
        }

        // Find token
        let line: Vec<char> = line_str.chars().collect();
        let line_len = line_len as usize;
        let mut indnd = line_len;
        if let Some(position) = line[indst.min(line.len())..]
            .iter()
            .position(|&c| c == '=' || c == ' ' || c == '\t')
        {
            indnd = indst + position;
        }
        let token = python_slice(&line, indst, indnd);

        // Done if it matches end of input string
        if token == STANDARD_INPUT_END || (S_DONE_ENDS.load(Relaxed) != 0 && token == "DONE") {
            break;
        }

        // Look up option
        let opt_num = lookup_option(&token, S_NUM_OPTIONS.load(Relaxed));
        if opt_num < 0 {
            // If no option, process special case if in-line non-options allowed
            // or error out
            if S_NOT_FOUND_OK.load(Relaxed) != 0 {
                let err = add_value_string(
                    S_NON_OPT_IND.load(Relaxed),
                    &python_slice(&line, indst, line.len()),
                );
                if let Some(err) = err
                    && err != 0
                {
                    return err;
                }
                continue;
            } else {
                return opt_num;
            }
        }

        let (is_param_file, is_boolean) = {
            let mut guard = S_OPT_TABLE.lock().unwrap();
            let option_type = &opt_table(&mut guard)[opt_num as usize].option_type;
            (option_type == "PF", option_type == "B")
        };
        if is_param_file {
            pip_set_error(&format!(
                "Trying to open a parameter file while reading a parameter file or {STANDARD_INPUT_STRING}"
            ));
            return -1;
        }

        // Find first non-white space, passing over at most one equals sign
        let mut indst = indnd + 1;
        let mut got_equals = 0;
        while indst < line_len {
            if line[indst] == '=' {
                if got_equals != 0 {
                    pip_set_error(&format!("Two = signs in input line:  {line_str}"));
                    return -1;
                }
                got_equals = 1;
            } else if line[indst] != ' ' && line[indst] != '\t' {
                break;
            }
            indst += 1;
        }

        // If there is a string, get one; if not, get a "1" for boolean,
        // otherwise it is an error
        let token = if indst < line_len {
            python_slice(&line, indst, line.len())
        } else if is_boolean {
            "1".to_owned()
        } else {
            pip_set_error(&format!("Missing a value on the input line:  {line_str}"));
            return -1;
        };

        // Add the token as a value string and increment argument number
        let err = add_value_string(opt_num, &token);
        if let Some(err) = err
            && err != 0
        {
            return err;
        }
        S_NUM_OPTION_ARGUMENTS.fetch_add(1, Relaxed);
    }

    S_NOT_FOUND_OK.store(0, Relaxed);
    0
}

/// Matches `PipReadNextLine` (`IMOD/pysrc/pip.py:1230`).
///
/// Returns `(lineLen, lineStr, indst, badComment)`; `lineLen` and `indst`
/// count characters, as Python's do.
pub fn pip_read_next_line(
    p_file: &mut PythonTextFile,
    comment: char,
    keep_comments: i32,
    in_line_comments: i32,
) -> (i32, String, usize, String) {
    let mut line_str;
    let mut line_len;
    let mut indst;
    let mut bad_comment;
    loop {
        bad_comment = String::new();
        match p_file.readline() {
            Ok(line) => {
                line_str = line;
                line_len = line_str.chars().count();
                if line_len == 0 {
                    return (-3, String::new(), 0, String::new());
                }
            }
            Err(()) => return (-2, String::new(), 0, String::new()),
        }

        // Get first non-white space
        let characters: Vec<char> = line_str.chars().collect();
        let Some(first) = characters.iter().position(|&c| c != ' ' && c != '\t') else {
            continue;
        };
        indst = first;

        // If it is a comment, skip or strip line ending and return
        if characters[indst] == comment {
            if keep_comments != 0 {
                line_str = line_str.trim_end_matches(python_isspace).to_owned();
                line_len = line_str.chars().count();
                break;
            } else {
                continue;
            }
        }

        // adjust line length to remove comment, if we have in-line comments
        if in_line_comments != 0
            && let Some(str_ptr) = characters.iter().position(|&c| c == comment)
        {
            line_len = str_ptr;

            // Look for forbidden in-line comments
            let forbid_long = S_FORBID_COMMENT_LONG.lock().unwrap().clone();
            let forbid_short = S_FORBID_COMMENT_SHORT.lock().unwrap().clone();
            if !forbid_long.is_empty() || !forbid_short.is_empty() {
                let lsplit: Vec<&str> = line_str
                    .split(python_isspace)
                    .filter(|part| !part.is_empty())
                    .collect();
                let option = lsplit[0].trim_start_matches('-');
                if (!forbid_long.is_empty() && forbid_long.starts_with(option))
                    || (!forbid_short.is_empty() && forbid_short.starts_with(option))
                {
                    bad_comment = format!(
                        "The {comment} character should not be used for a comment or any other reason on the line: {line_str}"
                    );
                }
            }
        }

        // adjust line length back further to remove white space and newline
        line_str = python_slice(&characters, 0, line_len)
            .trim_end_matches(python_isspace)
            .to_owned();
        line_len = line_str.chars().count();

        // Return if something is on line or we are keeping comments
        if indst < line_len || keep_comments != 0 {
            break;
        }
    }

    (line_len as i32, line_str, indst, bad_comment)
}

/// Matches `OptionLineOfValues` (`IMOD/pysrc/pip.py:1287`).
pub fn option_line_of_values(option: &str, val_type: i32, num_to_get: usize) -> Option<PipValues> {
    // Get string  and save pointer to it for error messages
    let str_ptr = get_next_value_string(option);
    let errno = PIP_ERRNO.load(Relaxed);
    if errno != 0 && errno != 2 {
        return None;
    }
    let save_errno = errno;
    let retval = pip_get_line_of_values(option, &str_ptr.unwrap_or_default(), val_type, num_to_get);
    if PIP_ERRNO.load(Relaxed) == 0 {
        PIP_ERRNO.store(save_errno, Relaxed);
    }
    retval
}

/// Matches `PipGetLineOfValues` (`IMOD/pysrc/pip.py:1310`).
///
/// Unlike the C, `gotComma` starts at 0 (a leading comma is skipped), the
/// values are Python ints/floats, and there is no array-size limit.
pub fn pip_get_line_of_values(
    option: &str,
    str_ptr: &str,
    val_type: i32,
    num_to_get: usize,
) -> Option<PipValues> {
    PIP_ERRNO.store(0, Relaxed);
    let full_str = str_ptr;
    let mut int_array: Vec<i64> = Vec::new();
    let mut float_array: Vec<f64> = Vec::new();
    let mut num_got = 0usize;
    let mut got_comma = 0;
    let allow_defaults = S_ALLOW_DEFAULTS.load(Relaxed);
    let mut str_ptr: Vec<char> = str_ptr.chars().collect();

    while !str_ptr.is_empty() {
        let match_start = str_ptr
            .iter()
            .position(|&c| c == ' ' || c == ',' || c == '\t' || c == '/');
        let end_ptr;
        if match_start.is_none() {
            // no object means read a number to end of string
            end_ptr = str_ptr.len();
        } else if match_start == Some(0) {
            // separator at start means advance by one byte and continue
            // if defaults allowed and a specific number are expected,
            // / means stop processing and mark all values as received
            if str_ptr[0] == '/' {
                if allow_defaults != 0 && num_to_get != 0 {
                    num_got = num_to_get;
                    break;
                }

                let temp_str = format!(
                    "Default entry with a / is not allowed in value entry:  {option}  {full_str}"
                );
                pip_set_error(&temp_str);
                PIP_ERRNO.store(-1, Relaxed);
                return None;
            }

            // special handling of commas to allow default values
            if str_ptr[0] == ',' {
                // If already have a comma, skip an array value if defaults
                // allowed
                if got_comma != 0 {
                    if allow_defaults != 0 && num_to_get != 0 {
                        num_got += 1;
                        if num_got >= num_to_get {
                            break;
                        }
                    } else {
                        let temp_str = format!(
                            "Default entries with commas are not allowed in value entry:  {option}  {full_str}"
                        );
                        pip_set_error(&temp_str);
                        PIP_ERRNO.store(-1, Relaxed);
                        return None;
                    }
                }
                got_comma = 1;
            }

            str_ptr.remove(0);
            continue;
        } else {
            // otherwise, this should be the end index for the conversion
            end_ptr = match_start.unwrap();
        }

        // convert number in a try block
        let text = python_slice(&str_ptr, 0, end_ptr);
        let converted = if val_type == PIP_INTEGER {
            python_int(&text).map(|value| int_array.push(value))
        } else {
            python_float(&text).map(|value| float_array.push(value))
        };
        if converted.is_none() {
            let temp_str = format!("Illegal character in value entry:  {option}  {full_str}");
            pip_set_error(&temp_str);
            PIP_ERRNO.store(-1, Relaxed);
            return None;
        }
        num_got += 1;

        // Mark that there is no comma after we have a number
        got_comma = 0;

        // Done if at end of line, or if count is fulfilled
        if end_ptr == str_ptr.len() || (num_to_get != 0 && num_got >= num_to_get) {
            break;
        }

        // Otherwise advance to separator and continue
        str_ptr = str_ptr[end_ptr..].to_vec();
    }

    // If not enough values found, return error
    if num_to_get > 0 && num_got < num_to_get {
        let temp_str = format!(
            "{num_to_get} values expected but only {num_got} values found in value entry:  {option}  {full_str}"
        );
        pip_set_error(&temp_str);
        PIP_ERRNO.store(-1, Relaxed);
        return None;
    }

    if val_type == PIP_INTEGER {
        Some(PipValues::Integers(int_array))
    } else {
        Some(PipValues::Floats(float_array))
    }
}

/// Matches `GetNextValueString` (`IMOD/pysrc/pip.py:1408`).  `pipErrno` is
/// set < 0 for an invalid option, 1 if the option was not entered, 2 if a
/// default is being returned.
pub fn get_next_value_string(option: &str) -> Option<String> {
    PIP_ERRNO.store(0, Relaxed);
    let err = lookup_option(option, S_NON_OPT_IND.load(Relaxed) + 1);
    if err < 0 {
        PIP_ERRNO.store(err, Relaxed);
        return None;
    }
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let optp = &mut opt_table(&mut guard)[err as usize];
    if optp.count == 0 {
        if !optp.default_val.is_empty() {
            PIP_ERRNO.store(2, Relaxed);
            return Some(optp.default_val.clone());
        }
        PIP_ERRNO.store(1, Relaxed);
        return None;
    }

    let mut index = 0;
    if optp.multiple != 0 {
        index = optp.multiple - 1;
    }
    if optp.multiple != 0 && optp.multiple < optp.count {
        optp.multiple += 1;
    }
    Some(optp.values[index as usize].clone())
}

/// Matches `AddValueString` (`IMOD/pysrc/pip.py:1433`).  `None` is the
/// Python's `None` (the linked-option lookup failed).
pub fn add_value_string(opt_ind: i32, str_ptr: &str) -> Option<i32> {
    // Add the index of the next non-option arg and the index of a linked option if
    // one is defined to the array for these
    let linked = {
        let mut guard = S_OPT_TABLE.lock().unwrap();
        let table = opt_table(&mut guard);
        let non_opt_count = table[S_NON_OPT_IND.load(Relaxed) as usize].count;
        let optp = &mut table[opt_ind as usize];
        if optp.linked {
            optp.next_linked.push(non_opt_count);
        }
        optp.linked
    };
    if linked {
        let mut ind = 0;
        let linked_option = S_LINKED_OPTION.lock().unwrap().clone();
        if let Some(linked_option) = linked_option
            && !linked_option.is_empty()
        {
            let err = lookup_option(&linked_option, S_NUM_OPTIONS.load(Relaxed));
            if err < 0 {
                PIP_ERRNO.store(err, Relaxed);
                return None;
            }
            let mut guard = S_OPT_TABLE.lock().unwrap();
            ind = opt_table(&mut guard)[err as usize].count;
        }
        let mut guard = S_OPT_TABLE.lock().unwrap();
        opt_table(&mut guard)[opt_ind as usize]
            .next_linked
            .push(ind);
    }

    // If we accept multiple values or have none yet, append;
    // otherwise set first element
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let optp = &mut opt_table(&mut guard)[opt_ind as usize];
    if optp.multiple != 0 || optp.count == 0 {
        optp.values.push(str_ptr.to_owned());
        optp.count += 1;
    } else {
        optp.values[0] = str_ptr.to_owned();
        optp.count = 1;
    }
    Some(0)
}

/// Matches `LookupOption` (`IMOD/pysrc/pip.py:1463`).
pub fn lookup_option(option: &str, max_lookup: i32) -> i32 {
    let lenopt = option.chars().count() as i32;
    let mut found = LOOKUP_NOT_FOUND;
    let no_abbrevs = S_NO_ABBREVS.load(Relaxed);

    // Look at all of the options specified by maxLookup
    let mut guard = S_OPT_TABLE.lock().unwrap();
    let table = opt_table(&mut guard);
    for i in 0..max_lookup {
        let sname = &table[i as usize].short_name;
        let lname = &table[i as usize].long_name;
        let len_short = table[i as usize].len_short;
        let starts = pip_starts_with(sname, option);

        // First test for single letter short name match - if it passes, skip ambiguity test
        if lenopt == 1 && starts && len_short == 1 {
            found = i;
            break;
        }

        if (starts && (no_abbrevs == 0 || lenopt == len_short))
            || (pip_starts_with(lname, option)
                && (no_abbrevs == 0 || lenopt == lname.chars().count() as i32))
        {
            // If it is found, it's an error if one has already been found
            if found == LOOKUP_NOT_FOUND {
                found = i;
            } else {
                if !S_TEST_ABBREV_FOR_USAGE.load(Relaxed) {
                    let temp_str = format!(
                        "An option specified by \"{option}\" is ambiguous between option {sname} -  {lname}  and option {} -  {}",
                        table[found as usize].short_name, table[found as usize].long_name
                    );
                    drop(guard);
                    pip_set_error(&temp_str);
                }
                return LOOKUP_AMBIGUOUS;
            }
        }
    }
    drop(guard);

    // Set error string unless flag set that non-options are OK
    if found == LOOKUP_NOT_FOUND && S_NOT_FOUND_OK.load(Relaxed) == 0 {
        let temp_str = format!("Illegal option: {option}");
        pip_set_error(&temp_str);
    }
    found
}

/// Matches `PipStartsWith` (`IMOD/pysrc/pip.py:1504`).
pub fn pip_starts_with(str1: &str, str2: &str) -> bool {
    if str1.is_empty() || str2.is_empty() {
        return false;
    }
    str1.starts_with(str2)
}

/// Matches `LineIsOptionToken` (`IMOD/pysrc/pip.py:1514`).
pub fn line_is_option_token(line: &str) -> i32 {
    // It is not a token unless it starts with open delim and contains close
    if !pip_starts_with(line, OPEN_DELIM) || !line.contains(CLOSE_DELIM) {
        return 0;
    }

    // It must then contain "Field" right after delim to be an option
    let openlen = OPEN_DELIM.len();
    if pip_starts_with(&line[openlen..], "Field") {
        return 1;
    }
    if pip_starts_with(&line[openlen..], "SectionHeader") {
        return 2;
    }

    -1
}

/// Matches `CheckKeyword` (`IMOD/pysrc/pip.py:1537`).  Returns
/// `(value, index, quoteIndex)`.
pub fn check_keyword(line: &str, keyword: &str, index: i32, quote_ok: i32) -> (String, i32, i32) {
    let value_delim = S_VALUE_DELIM
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| VALUE_DELIM.to_owned());
    let line_chars: Vec<char> = line.chars().collect();
    let line_len = line_chars.len();

    // First make sure line starts with it
    if !pip_starts_with(line, keyword) {
        return (String::new(), 0, -1);
    }

    // Now look for delimiter
    let Some(byte_start) = line.find(value_delim.as_str()) else {
        return (String::new(), 0, -1);
    };
    let mut val_start = line[..byte_start].chars().count();

    // Eat spaces after the delimiter and return if nothing left
    // In other words, a key with no value is the same as having no key at all
    val_start += value_delim.chars().count();
    while val_start < line_len && (line_chars[val_start] == ' ' || line_chars[val_start] == '\t') {
        val_start += 1;
    }
    if val_start >= line_len {
        return (String::new(), 0, -1);
    }

    // Look for quote if directed to
    let mut ind_quote = -1;
    if quote_ok != 0 && S_QUOTE_TYPES.contains(&line_chars[val_start]) {
        ind_quote = S_QUOTE_TYPES
            .iter()
            .position(|&c| c == line_chars[val_start])
            .unwrap() as i32;
        val_start += 1;
    }

    (
        python_slice(&line_chars, val_start, line_len),
        index,
        ind_quote,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        PipOption, PipValues, expand_arg_list, pip_add_option, pip_done, pip_get_integer,
        pip_get_line_of_values, pip_initialize, pip_next_arg,
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
        assert_eq!(pip_next_arg("-n"), Some(1));
        assert_eq!(pip_next_arg("17"), Some(0));
        assert_eq!(pip_get_integer("Number", 3), Ok(17));
        pip_done();
    }

    #[test]
    fn leading_comma_is_skipped_as_in_python() {
        // `pip.py:1318` starts with gotComma = 0; the C starts with 1.
        assert_eq!(
            pip_get_line_of_values("X", ",5,1_0", 1, 2),
            Some(PipValues::Integers(vec![5, 10]))
        );
    }
}
