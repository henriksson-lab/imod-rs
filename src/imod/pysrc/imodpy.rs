//! Direct Rust translation of the process/error portion of `IMOD/pysrc/imodpy.py`.

use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{LazyLock, Mutex};

static ERR_STRINGS: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static ERR_STATUS: LazyLock<Mutex<i32>> = LazyLock::new(|| Mutex::new(0));
static RUN_RETRY_LIMIT: LazyLock<Mutex<i32>> = LazyLock::new(|| Mutex::new(10));
static RUN_MAX_TIME_FOR_RETRY: LazyLock<Mutex<f64>> = LazyLock::new(|| Mutex::new(0.5));
static RAISE_KEY_INTERRUPT: LazyLock<Mutex<bool>> = LazyLock::new(|| Mutex::new(false));
static FILE_TYPE_EXTENSION: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new(String::new()));
static CURRENT_ROOTNAME: LazyLock<Mutex<String>> = LazyLock::new(|| Mutex::new(String::new()));
static MOC_RENAMES_OK: AtomicBool = AtomicBool::new(true);
/// Rust-only: the unit number [`header_in_process`] opens on; `header` itself
/// always uses unit 1 (`header.f90:132`).
static HEADER_UNIT: AtomicI32 = AtomicI32::new(1);

/// Matches Python class `ImodpyError` (`IMOD/pysrc/imodpy.py:159`).
#[derive(Clone, Debug)]
pub struct ImodpyError {
    pub arguments: Vec<String>,
}

/// Return variants of `optionValue` (`IMOD/pysrc/imodpy.py:1153`).
#[derive(Clone, Debug, PartialEq)]
pub enum OptionValue {
    String(String),
    Integers(Vec<i32>),
    Floats(Vec<f32>),
    Boolean(bool),
}

/// Return shapes of `getmrc` (`IMOD/pysrc/imodpy.py:456`).
#[derive(Clone, Debug, PartialEq)]
pub enum MrcInfo {
    Basic(i32, i32, i32, i32, f32, f32, f32),
    All(
        i32,
        i32,
        i32,
        i32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
        f32,
    ),
    AngleLines([Option<f32>; 5]),
}

impl Display for ImodpyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.arguments.join("\n"))
    }
}

impl std::error::Error for ImodpyError {}

/// Matches `getErrStrings` (`IMOD/pysrc/imodpy.py:167`).
pub fn get_err_strings() -> Vec<String> {
    ERR_STRINGS.lock().expect("imodpy errors mutex").clone()
}

/// Matches `runcmd` (`IMOD/pysrc/imodpy.py:176`).
pub fn run_cmd(
    command: &str,
    input: Option<&[String]>,
    outfile: Option<&str>,
    in_stderr: Option<&str>,
    ignore_status: &[i32],
) -> Result<Option<Vec<String>>, ImodpyError> {
    let command = avoid_local_com_file(command);
    let command = command.as_str();

    // Owner decision, 2026-09-24: a command of ours runs in this process (see
    // `run_cmd_in_process`).  Only when standard error is left alone: the
    // in-process runner redirects standard input and output, not error.
    if in_stderr.is_none()
        && let Some((own, words)) = own_command_words(command)
    {
        return run_own_command(command, own, words, input, outfile, ignore_status, false);
    }

    // Set up flags for whether to collect output or send to stderr
    *ERR_STATUS.lock().expect("imodpy status mutex") = 0;
    let mut process = Command::new("sh");
    process.arg("-c").arg(command);
    if input.is_some() {
        process.stdin(Stdio::piped());
    }
    if outfile == Some("stdout") {
        process.stdout(Stdio::inherit());
    } else {
        process.stdout(Stdio::piped());
    }
    if outfile == Some("stdout") || in_stderr == Some("stdout") {
        process.stderr(Stdio::inherit());
    } else if in_stderr == Some("pipe") {
        process.stderr(Stdio::piped());
    }
    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Starting command {command}: {error}");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
    };
    if let Some(lines) = input {
        if let Some(mut stdin) = child.stdin.take() {
            for line in lines {
                if stdin
                    .write_all(line.as_bytes())
                    .and_then(|_| stdin.write_all(b"\n"))
                    .is_err()
                {
                    break;
                }
            }
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(error) => {
            let message = format!("Waiting for command {command}: {error}");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
    };
    let status = output.status.code().unwrap_or(1);
    *ERR_STATUS.lock().expect("imodpy status mutex") = status;
    let standard_error = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() && !ignore_status.contains(&status) {
        let mut errors = standard_error
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if errors.is_empty() {
            errors.push(format!("{command}: exit status {status}"));
        }
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = errors.clone();
        return Err(ImodpyError { arguments: errors });
    }
    if let Some(filename) = outfile.filter(|name| *name != "stdout") {
        if let Err(error) = fs::write(filename, &output.stdout) {
            let message = format!("Writing to file: {filename}  - {error}");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
        return Ok(None);
    }
    if outfile == Some("stdout") {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_owned)
            .collect(),
    ))
}

/// `runcmd` (`IMOD/pysrc/imodpy.py:176`) for a command this crate itself
/// provides, run **in this process** rather than through `sh -c`.
///
/// Owner decision, 2026-09-24: where a Python script ran one of our own
/// commands, the translation calls the translated program instead of
/// spawning a process (see `CLAUDE.md`, "Our own commands are called in
/// process").  [`run_cmd`] itself now routes every such command here, so this
/// entry point differs from it only in the shape of the returned lines: they
/// keep their line endings, as Python's `splitlines(True)` does, where
/// [`run_cmd`] has always returned them without.  A command that is not one
/// of ours goes to [`run_cmd`], and so to `sh -c`.
///
/// With no `outfile` the program's standard output is collected and
/// returned; with `outfile == Some("stdout")` it goes straight to standard
/// output; `input` lines are fed to its standard input, each ended by a
/// newline.  A non-zero exit status raises [`ImodpyError`] with `errStrings`
/// set to every collected line containing `ERROR:` followed by
/// `"<cmd>: exited with status <n>"`, as `runcmd` does after the retry loop.
/// Standard error is not redirected, as `runcmd` leaves it with `inStderr`
/// unset.  The retry-on-broken-pipe loop and `RUNCMD_VERBOSE` echo have no
/// counterpart: there is no pipe to break.
pub fn run_cmd_in_process(
    command: &str,
    input: Option<&[String]>,
    outfile: Option<&str>,
) -> Result<Option<Vec<String>>, ImodpyError> {
    let command = avoid_local_com_file(command);
    let command = command.as_str();
    match own_command_words(command) {
        Some((own, words)) => run_own_command(command, own, words, input, outfile, &[], true),
        None => {
            let ignore: [i32; 0] = [];
            run_cmd(command, input, outfile, None, &ignore)
        }
    }
}

/// Rust-only: splits `command` into words as `sh -c` would and returns them
/// with the command table's entry for the first word, when that word names a
/// command of ours that may run in process (`commands::Command::in_process`)
/// and the line needs nothing else from a shell.
///
/// Blanks separate words outside quotes; `"..."`/`'...'` quoting and
/// backslash escapes are removed.  Any unquoted character that would make
/// the shell do more than split words -- a pipe, redirection, command
/// separator, background `&`, parameter or command substitution, glob,
/// subshell, tilde or comment -- returns `None`, and the line is left to the
/// shell.  So does a `$` or backquote inside double quotes, where the shell
/// would still expand it.
fn own_command_words(
    command: &str,
) -> Option<(&'static crate::imod::commands::Command, Vec<String>)> {
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\n' => {
                if in_word {
                    words.push(std::mem::take(&mut current));
                    in_word = false;
                }
            }
            '\'' => {
                in_word = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    current.push(q);
                }
            }
            '"' => {
                in_word = true;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => break,
                        '$' | '`' => return None,
                        '\\' if matches!(chars.peek(), Some('"' | '\\' | '$' | '`')) => {
                            current.push(chars.next().unwrap());
                        }
                        _ => current.push(q),
                    }
                }
            }
            '\\' => {
                in_word = true;
                if let Some(q) = chars.next() {
                    current.push(q);
                }
            }
            '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '*' | '?' | '[' => {
                return None;
            }
            '~' | '#' if !in_word => return None,
            _ => {
                in_word = true;
                current.push(c);
            }
        }
    }
    if in_word {
        words.push(current);
    }
    let own = crate::imod::commands::find(words.first()?)?;
    if !own.in_process {
        return None;
    }
    Some((own, words))
}

/// Rust-only: runs a command of ours in this process with `runcmd`'s
/// contract (see [`run_cmd_in_process`]).  `keep_ends` selects the shape of
/// the collected lines: with their endings (`splitlines(True)`) for
/// [`run_cmd_in_process`], without for [`run_cmd`].  `ignore_status` is
/// `runcmd`'s `ignoreStatus`.
fn run_own_command(
    command: &str,
    own: &'static crate::imod::commands::Command,
    words: Vec<String>,
    input: Option<&[String]>,
    outfile: Option<&str>,
    ignore_status: &[i32],
    keep_ends: bool,
) -> Result<Option<Vec<String>>, ImodpyError> {
    // The program's `argv[0]` is the path a command link would have,
    // `<bindir>/<name>`, as `src/bin/imod.rs` records it for a subcommand.
    let mut argv: Vec<OsString> = Vec::with_capacity(words.len());
    argv.push(match std::env::current_exe() {
        Ok(path) => path.with_file_name(&words[0]).into_os_string(),
        Err(_) => OsString::from(&words[0]),
    });
    argv.extend(words.iter().skip(1).map(OsString::from));
    let joined = input.map(|lines| {
        let mut text = String::new();
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
        text
    });
    let collect = outfile.is_none();
    let capture = outfile != Some("stdout");
    *ERR_STATUS.lock().expect("imodpy status mutex") = 0;
    let (status, output) = match crate::imod::commands::run_in_process(
        own,
        argv,
        joined.as_deref().map(str::as_bytes),
        capture,
    ) {
        Ok(result) => result,
        Err(error) => {
            let message = format!("command {command}: {error}\n");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
    };
    let mut lines: Vec<String> = Vec::new();
    if collect {
        let text = String::from_utf8_lossy(&output);
        lines = if keep_ends {
            text.split_inclusive('\n').map(str::to_owned).collect()
        } else {
            text.lines().map(str::to_owned).collect()
        };
    }
    if status != 0 {
        *ERR_STATUS.lock().expect("imodpy status mutex") = status;
    }
    if status != 0 && !ignore_status.contains(&status) {
        let mut errors: Vec<String> = lines
            .iter()
            .filter(|line| line.contains("ERROR:"))
            .cloned()
            .collect();
        // `exit_from_imod_error` supplies the final line ending itself.
        errors.push(format!("{command}: exited with status {status}"));
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = errors.clone();
        return Err(ImodpyError { arguments: errors });
    }
    if let Some(filename) = outfile.filter(|name| *name != "stdout") {
        // `runcmd` hands a file object to the child as its standard output;
        // the only callers pass `'stdout'`, so a named file keeps
        // `run_cmd`'s shape of writing the collected text.
        if let Err(error) = fs::write(filename, &output) {
            let message = format!("Writing to file: {filename}  - {error}");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
        return Ok(None);
    }
    if !collect {
        return Ok(None);
    }
    Ok(Some(lines))
}

/// Matches `setRetryLimit` (`IMOD/pysrc/imodpy.py:353`).
pub fn set_retry_limit(number_retries: i32, maximum_time: Option<f64>) {
    *RUN_RETRY_LIMIT.lock().expect("imodpy retry mutex") = number_retries.max(0);
    if let Some(value) = maximum_time {
        *RUN_MAX_TIME_FOR_RETRY
            .lock()
            .expect("imodpy retry-time mutex") = value;
    }
}

/// Matches `passOnKeyInterrupt` (`IMOD/pysrc/imodpy.py:365`).
pub fn pass_on_key_interrupt(pass_on: bool) {
    *RAISE_KEY_INTERRUPT.lock().expect("imodpy interrupt mutex") = pass_on;
}

/// Matches `getLastExitStatus` (`IMOD/pysrc/imodpy.py:371`).
pub fn get_last_exit_status() -> i32 {
    *ERR_STATUS.lock().expect("imodpy status mutex")
}

/// Matches `bkgdProcess` (`IMOD/pysrc/imodpy.py:376`).
pub fn bkgd_process(
    command_array: &[OsString],
    outfile: Option<&str>,
    errfile: Option<&str>,
    return_on_error: bool,
    append: bool,
) -> Result<(), ImodpyError> {
    if command_array.is_empty() {
        let message = "Starting background process: empty command".to_owned();
        if return_on_error {
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
        crate::imod::pysrc::pip::exit_error(&message);
    }
    let mut process = Command::new(&command_array[0]);
    process.args(&command_array[1..]);
    if let Some(filename) = outfile {
        let mut options = OpenOptions::new();
        options.create(true).write(true);
        if append {
            options.append(true);
        } else {
            options.truncate(true);
        }
        match options.open(filename) {
            Ok(file) => process.stdout(Stdio::from(file)),
            Err(error) => {
                let message = format!("Opening {filename} for output  - {error}");
                if return_on_error {
                    return Err(ImodpyError {
                        arguments: vec![message],
                    });
                }
                crate::imod::pysrc::pip::exit_error(&message);
            }
        };
    } else {
        process.stdout(Stdio::inherit());
    }
    if errfile == Some("stdout") {
        // Python passes subprocess.STDOUT, not the parent's stderr stream.
        // On the Unix source target, reopen the inherited stdout descriptor
        // for the spawned child's stderr.
        #[cfg(unix)]
        match OpenOptions::new().write(true).open("/dev/stdout") {
            Ok(file) => process.stderr(Stdio::from(file)),
            Err(_) => process.stderr(Stdio::inherit()),
        };
        #[cfg(not(unix))]
        process.stderr(Stdio::inherit());
    } else if errfile == Some("devnull") {
        process.stderr(Stdio::null());
    } else if let Some(filename) = errfile {
        let mut options = OpenOptions::new();
        options.create(true).write(true);
        if append {
            options.append(true);
        } else {
            options.truncate(true);
        }
        match options.open(filename) {
            Ok(file) => process.stderr(Stdio::from(file)),
            Err(error) => {
                let message = format!("Opening {filename} for error output  - {error}");
                if return_on_error {
                    return Err(ImodpyError {
                        arguments: vec![message],
                    });
                }
                crate::imod::pysrc::pip::exit_error(&message);
            }
        };
    }
    match process.spawn() {
        Ok(_) => Ok(()),
        Err(error) => {
            let message = format!(
                "Starting background process {}  - {error}",
                command_array[0].to_string_lossy()
            );
            if return_on_error {
                Err(ImodpyError {
                    arguments: vec![message],
                })
            } else {
                crate::imod::pysrc::pip::exit_error(&message)
            }
        }
    }
}

/// Matches `multiCharSplit` (`IMOD/pysrc/imodpy.py:451`).
pub fn multi_char_split(line: &str, characters: &str) -> Vec<String> {
    line.split(|character| characters.contains(character))
        .filter(|entry| !entry.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Rust-only: runs our own translated `header` program's logic **in process**
/// and returns the lines of its standard output that the `pysrc` callers read.
///
/// Owner decision (2026-09-24): where a `pysrc` translation ran one of *this
/// crate's own* commands through `runcmd` (`sh -c`, resolved through `PATH`),
/// it now calls the translated library directly.  Through `PATH` the command
/// could silently be native IMOD's `header`, and a shell plus an exec per
/// query was the slow part of `getmrc`.  This is a deliberate deviation from
/// upstream's process-based design; genuinely external tools stay processes.
///
/// The lines are built with the formats `header.rs` / `irdhdr.rs` /
/// `wrap_iiunit.rs` print them with, so every value a caller parses back has
/// been rounded exactly as the printed text rounded it.  `command` is the
/// command line that used to be run; it only names the failure.
///
/// * `stdin_name` true: the file name arrived as the `-StandardInput` line
///   `InputFile <file>`, so it gets PIP's `ReadParamFile` treatment
///   (`parse_params.c:1665-1760`): an in-line `#` comment is cut, trailing
///   blanks/CR are stripped, leading blanks and at most one `=` are skipped,
///   and an empty value is an error.  False: a command-line argument.
///   Either way `PipGetString` fills `character*320 inFile` (`header.f90:21`),
///   truncating past 320 bytes, and trailing blanks are trimmed.
/// * `silent`: `None` is plain `header` (full output); `Some(false)` is
///   `-si -mo -pi` and `Some(true)` adds `-ori -min -max -mean`.
///
/// Full output reproduces, in order: the `imopen` file line and file-type line
/// (`wrap_iiunit.f90`), the multi-volume note, from `irdhdr` the `Pixel
/// spacing` line, the titles (first 79 bytes, FORMAT 1020) and the `idtype`
/// line, then every line `header.f90:170-340` prints (extended header, mdoc).
/// The other `irdhdr` lines carry none of the keys any caller looks for
/// (`Pixel spacing`, `size in nanometers =`, `axis`+`angle`, `This is a`,
/// `ctfPhaseFlip`, `CTF correct`) and none precede the file-type line, so the
/// first-nine-lines window of `getImageFormat` is unchanged.  Silent output
/// reproduces the numeric lines only: the notes `irdhdr`/`header` can print
/// before them are exactly the lines `getmrc` discards, and the mdoc lines
/// after them are past the indices it reads.
///
/// Errors: every path on which `header` exits non-zero (`imopen` failure,
/// unreadable extended header, `get_extra_header_items` failure) returns an
/// `ImodpyError` as `runcmd` raised one — the `ERROR:` lines, then
/// `<command>: exited with status 1` (`imodpy.py:335-344`) — and sets the
/// error strings and last exit status the same way.
pub fn header_in_process(
    command: &str,
    file: &str,
    stdin_name: bool,
    silent: Option<bool>,
) -> Result<Vec<String>, ImodpyError> {
    use crate::imod::libcfshr::autodoc::{
        ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_clear, adoc_get_float,
        adoc_get_number_of_sections, adoc_get_section_name, adoc_open_image_metadata,
    };
    use crate::imod::libcfshr::b3dutil::{
        b3d_get_error, b3d_get_store_error, b3d_set_store_error, extra_is_nbytes_and_flags,
    };
    use crate::imod::libcfshr::extraheader::{
        get_extra_header_items, get_extra_header_value, get_fei_ext_head_angle_scale,
    };
    use crate::imod::libiimod::iimage::ii_allow_multi_volume;
    use crate::imod::libiimod::unit_fileio::{
        MAX_UNIT, iiu_close, iiu_exit_on_error, iiu_file_info, iiu_get_exit_on_error, iiu_open,
        iiu_ret_chunk_sizes, iiu_ret_num_volumes,
    };
    use crate::imod::libiimod::unit_header::{
        iiu_ret_basic_head, iiu_ret_data_type, iiu_ret_delta, iiu_ret_extended_data,
        iiu_ret_extended_type, iiu_ret_imod_flags, iiu_ret_labels, iiu_ret_num_extended,
        iiu_ret_origin, iiu_ret_size,
    };

    *ERR_STATUS.lock().expect("imodpy status mutex") = 0;
    // `runcmd`'s failure branch (`imodpy.py:335-344`): the output's `ERROR:`
    // lines, then the exit-status line.
    let fail = |messages: Vec<String>| -> ImodpyError {
        let mut err_strings = messages
            .iter()
            .flat_map(|message| message.lines())
            .filter(|line| line.contains("ERROR:"))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        err_strings.push(format!("{command}: exited with status 1"));
        *ERR_STATUS.lock().expect("imodpy status mutex") = 1;
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = err_strings.clone();
        ImodpyError {
            arguments: err_strings,
        }
    };

    // The file name as `header` ends up holding it in `inFile`.
    let mut value = file.as_bytes();
    if stdin_name {
        // `PipReadNextLine` reads one line of "InputFile <file>"; with the
        // fixed prefix, the token is always `InputFile`.
        if let Some(newline) = value.iter().position(|&byte| byte == b'\n') {
            value = &value[..newline];
        }
        if let Some(comment) = value.iter().position(|&byte| byte == b'#') {
            value = &value[..comment];
        }
        while let [rest @ .., b' ' | b'\t' | b'\n' | b'\r'] = value {
            value = rest;
        }
        let mut got_equals = false;
        loop {
            match value.first() {
                Some(b'=') if got_equals => {
                    return Err(fail(vec![format!(
                        "ERROR: HEADER - Two = signs in input line:  InputFile {file}"
                    )]));
                }
                Some(b'=') => got_equals = true,
                Some(b' ' | b'\t') => {}
                _ => break,
            }
            value = &value[1..];
        }
        if value.is_empty() {
            return Err(fail(vec![format!(
                "ERROR: HEADER - Missing a value on the input line:  InputFile {file}"
            )]));
        }
    }
    let mut length = value.len().min(320);
    while length > 0 && value[length - 1] == b' ' {
        length -= 1;
    }
    let in_file = String::from_utf8_lossy(&value[..length]).into_owned();

    // Fortran `Gw.d` editing, the closure `header.rs` and `irdhdr.rs` carry.
    let g_edit = |value: f32, w: usize, d: i32| -> String {
        if value.is_nan() {
            return format!("{:>w$}", "NaN");
        }
        if value.is_infinite() {
            let text = match (value < 0.0, w) {
                (false, 8..) => "Infinity",
                (false, _) => "Inf",
                (true, 9..) => "-Infinity",
                (true, _) => "-Inf",
            };
            return format!("{text:>w$}");
        }
        let magnitude = value.abs();
        let mut digits = String::new();
        let mut exponent = 1_i32;
        if magnitude != 0.0 {
            let scientific = format!("{:.*e}", (d - 1) as usize, magnitude);
            let (mantissa, power) = scientific.split_once('e').unwrap();
            digits = mantissa.replace('.', "");
            exponent = power.parse::<i32>().unwrap() + 1;
        }
        if (0..=d).contains(&exponent) {
            let mut text = format!("{:.*}", (d - exponent) as usize, value);
            if exponent == d {
                text.push('.');
            }
            format!("{:>1$}    ", text, w - 4)
        } else {
            format!(
                "{:>1$}",
                format!(
                    "{}0.{}E{}{:02}",
                    if value < 0.0 { "-" } else { "" },
                    digits,
                    if exponent < 0 { '-' } else { '+' },
                    exponent.abs()
                ),
                w
            )
        }
    };

    // `header` runs with the unit layer's defaults: it exits on an open
    // error, and prints the error.  Here the failure has to come back as a
    // value, so exit is off and the message is stored, not printed.
    // `iiuOpen("")` would also install an MRC check function globally
    // (`unit_fileio.c:224`); `header` fails on an empty name either way.
    if in_file.is_empty() {
        return Err(fail(vec!["ERROR: iiuOpen - Could not open ''".to_owned()]));
    }
    // Every error path below leaves exit-on-error off and the store flag on
    // until the unit is closed, so that nothing can exit this process.
    let saved_exit = iiu_get_exit_on_error();
    let saved_store = b3d_get_store_error();
    // The unit's own store flag too: the first `findNewUnit` of a process
    // copies it into `b3dSetStoreError` (`unit_fileio.c:985-986`).
    iiu_exit_on_error(0, 1);
    b3d_set_store_error(1);
    ii_allow_multi_volume(1);
    let im_unit = HEADER_UNIT.load(Ordering::SeqCst);
    let ierr = unsafe { iiu_open(im_unit, &in_file, "RO") };
    ii_allow_multi_volume(0);
    if ierr != 0 {
        let message = b3d_get_error();
        // With exit-on-error off, `iiuOpen` returns with the unit still
        // marked in use (`unit_fileio.c:233-236`).  When `iiOpen` itself
        // failed its `iiFile` is NULL, and any later `iiuClose` of that unit
        // number -- including the implicit one when the number is reused --
        // dereferences it; that unit number is abandoned instead.  Every
        // other failure has a file to close.
        if message.contains("Could not open") {
            HEADER_UNIT.store((im_unit % (MAX_UNIT - 1)) + 1, Ordering::SeqCst);
        } else {
            unsafe { iiu_close(im_unit) };
        }
        iiu_exit_on_error(saved_exit, -1);
        b3d_set_store_error(saved_store);
        return Err(fail(vec![message]));
    }

    let mut out = String::new();
    let mut result = Ok(());
    unsafe {
        let (mut num_kbytes, mut itype, mut iflags) = (0, 0, 0);
        iiu_file_info(im_unit, &mut num_kbytes, &mut itype, &mut iflags);
        let num_volumes = iiu_ret_num_volumes(im_unit);
        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut mode = 0_i32;
        let mut dmin = 0.0_f32;
        let mut dmax = 0.0_f32;
        let mut dmean = 0.0_f32;
        // `irdhdr`: basic head, then the sizes again from `iiuRetSize`.
        iiu_ret_basic_head(
            im_unit,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        let nxyzst;
        (nxyz, mxyz, nxyzst) = iiu_ret_size(im_unit);
        let _ = nxyzst;
        if let Some(do_all) = silent {
            // `header.f90:151-161,165-167`.
            out.push_str(&format!("{:8}{:8}{:8}\n", nxyz[0], nxyz[1], nxyz[2]));
            if iflags & (1 << 8) != 0 {
                mode = 12;
            }
            out.push_str(&format!("{:4}\n", mode));
            let delta = iiu_ret_delta(im_unit);
            out.push_str(&format!(
                "{}{}{}\n",
                g_edit(delta[0], 15, 5),
                g_edit(delta[1], 15, 5),
                g_edit(delta[2], 15, 5)
            ));
            if do_all {
                let origin = iiu_ret_origin(im_unit);
                out.push_str(&format!(
                    "{}{}{}\n",
                    g_edit(origin[0], 15, 5),
                    g_edit(origin[1], 15, 5),
                    g_edit(origin[2], 15, 5)
                ));
                out.push_str(&format!("{}\n", g_edit(dmin, 13, 5)));
                out.push_str(&format!("{}\n", g_edit(dmax, 13, 5)));
                out.push_str(&format!("{}\n", g_edit(dmean, 13, 5)));
            }
        } else {
            // `iiuOpenPrint` (`wrap_iiunit.rs:37-73`).
            let (mut nx_tile, mut ny_tile, mut nz_chunk) = (0, 0, 0);
            iiu_ret_chunk_sizes(im_unit, &mut nx_tile, &mut ny_tile, &mut nz_chunk);
            if nx_tile > 0 || ny_tile > 0 {
                if itype == 5 && nx_tile == 0 {
                    nx_tile = nxyz[0];
                }
                if ny_tile == 0 {
                    ny_tile = nxyz[1];
                }
            }
            if num_kbytes < 0 {
                out.push_str(&format!("\n RO image file on unit{:4} : {}\n", 1, in_file));
            } else {
                out.push_str(&format!(
                    "\n RO image file on unit{:4} : {}     Size= {:10} K\n",
                    1, in_file, num_kbytes
                ));
            }
            match itype {
                1 if ny_tile > 0 && nx_tile == 0 => out.push_str(&format!(
                    "\n                    This is a TIFF file (in strips of{:7} x{:7}).\n",
                    nxyz[0], ny_tile
                )),
                1 if ny_tile > 0 => out.push_str(&format!(
                    "\n                    This is a TIFF file (in tiles of{:7} x{:7}).\n",
                    nx_tile, ny_tile
                )),
                1 => out.push_str("\n                    This is a TIFF file.\n"),
                5 if nx_tile > 0 => out.push_str(&format!(
                    "\n                    This is an HDF file (in chunks of{:7} x{:7} x{:5}).\n",
                    nx_tile, ny_tile, nz_chunk
                )),
                5 => out.push_str("\n                    This is an HDF file.\n"),
                6 => out.push_str("\n                    This is a JPEG file.\n"),
                7 => out.push_str("\n                    This is an image series file.\n"),
                2 => (),
                _ => out.push_str("\n                    This is a non-MRC file.\n"),
            }
            if iflags & 1 != 0 {
                out.push_str("\n                    This is a byte-swapped file.\n");
            }
            if num_volumes > 1 {
                out.push_str(&format!(
                    "This is the header for the first of{:4} volumes, use -vol # to see others\n",
                    num_volumes
                ));
            }

            // `irdhdr` (`irdhdr.rs`): the Pixel spacing line, the titles and
            // the data-type line.
            let delta = iiu_ret_delta(im_unit);
            out.push_str(&format!(
                " Pixel spacing (Angstroms).............. {}{}{}\n",
                g_edit(delta[0], 11, 4),
                g_edit(delta[1], 11, 4),
                g_edit(delta[2], 11, 4)
            ));
            let mut labels = [[0_u8; 80]; 10];
            let mut num_labels = 0;
            iiu_ret_labels(im_unit, &mut labels, &mut num_labels);
            out.push_str(&format!(" {:5} Titles :\n", num_labels));
            for label in labels.iter().take(num_labels.clamp(0, 10) as usize) {
                out.push_str(&format!("{}\n", String::from_utf8_lossy(&label[..79])));
            }
            let (idtype, _lensnum, nd1, nd2, vd1, vd2) = iiu_ret_data_type(im_unit);
            let lxyz = [' ', 'X', 'Y', 'Z'];
            let axis = if (1..=3).contains(&nd1) {
                lxyz[nd1 as usize]
            } else {
                ' '
            };
            match idtype {
                1 => out.push_str(&format!(
                    "      TILT data set, axis= {} delta,start angle= {:8.2}{:8.2}\n\n",
                    axis, vd1, vd2
                )),
                2 => out.push_str(&format!(
                    " SERIAL STEREO data set, axis= {} left angle= {:8.2} right angle= {:8.2}\n\n",
                    axis, vd1, vd2
                )),
                3 => out.push_str(&format!(
                    "      AVERAGED data set, Navg,Noffset   =  {:6}{:6}\n\n",
                    nd1, nd2
                )),
                4 => out.push_str(&format!(
                    "      AVG STEREO data set, Navg,Noffset= {:3}{:3} L,R angles= {:8.2}{:8.2}\n\n",
                    nd1, nd2, vd1, vd2
                )),
                _ => (),
            }

            // `header.rs:346-673`, printing into `out`.
            let mut found_pixel = false;
            let mut found_axis_rot = false;
            let mut nbsym = iiu_ret_num_extended(im_unit);
            'extended: {
                if nbsym <= 0 {
                    break 'extended;
                }
                let mut extended_data = Vec::new();
                if iiu_ret_extended_data(im_unit, &mut extended_data) != 0 {
                    result = Err(vec!["ERROR: HEADER - Reading extended header".to_owned()]);
                    break 'extended;
                }
                nbsym = extended_data.len() as i32;
                let array: Vec<f32> = extended_data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect();
                let [num_int, num_real] = iiu_ret_extended_type(im_unit);
                if extra_is_nbytes_and_flags(num_int, num_real) == 0 && num_real >= 12 {
                    // Agard/old FEI type
                    let mut tiltaxis = array[(num_int + 10) as usize];
                    if (-360.0..=360.0).contains(&tiltaxis) {
                        if tiltaxis < -180.0 {
                            tiltaxis += 360.0;
                        }
                        if tiltaxis > 180.0 {
                            tiltaxis -= 360.0;
                        }
                        if labels[0][..4] == *b"Fei " {
                            out.push_str(&format!(
                                "          Tilt axis rotation angle = {:7.1}{}\n",
                                -tiltaxis, " (Corrected sign)"
                            ));
                        } else {
                            out.push_str(&format!(
                                "          Tilt axis rotation angle = {:7.1}\n",
                                tiltaxis
                            ));
                        }
                        found_axis_rot = true;
                    }
                    let mut pixel = array[(num_int + 11) as usize];
                    if array[(num_int + 11) as usize] > 0.05
                        && array[(num_int + 11) as usize] < 100000.0
                    {
                        pixel /= 10.0;
                    } else {
                        pixel *= 1.0e9;
                    }
                    let (iflags, _if_imod) = iiu_ret_imod_flags(im_unit);
                    if pixel > 0.005 && pixel < 10000.0 && iflags & 2 == 0 {
                        let mut i_binning = 0_i32;
                        for j in (0..3).rev() {
                            i_binning = delta[j].round() as i32;
                            if (delta[j] - i_binning as f32).abs() > 1.0e-6
                                || i_binning <= 0
                                || i_binning > 4
                            {
                                i_binning = 0;
                                break;
                            }
                        }
                        if i_binning == 1 {
                            out.push_str(&format!(
                                "          Pixel size in nanometers ={}\n",
                                g_edit(pixel, 11, 4)
                            ));
                        } else if i_binning > 1 && i_binning < 5 {
                            out.push_str(&format!(
                                "          Pixel size in nanometers ={}{}{:2}{}\n",
                                g_edit(pixel * i_binning as f32, 11, 4),
                                " (Assumed binning of",
                                i_binning,
                                ")"
                            ));
                        } else {
                            out.push_str(&format!(
                                "          Original/extended header pixel size in nanometers ={}\n",
                                g_edit(pixel, 11, 4)
                            ));
                        }
                        found_pixel = true;
                    }
                }
                // New FEI type
                if num_int == -3 {
                    let mut byte_value = 0_u8;
                    let mut short_value = 0_i16;
                    let mut mask = 0_i32;
                    let mut j = 0_i32;
                    let mut tiltaxis = 0.0_f32;
                    let mut axis8 = 0.0_f64;
                    if get_extra_header_value(
                        &extended_data,
                        8,
                        3,
                        &mut byte_value,
                        &mut short_value,
                        &mut mask,
                        &mut tiltaxis,
                        &mut axis8,
                    ) == 0
                        && get_extra_header_value(
                            &extended_data,
                            140,
                            4,
                            &mut byte_value,
                            &mut short_value,
                            &mut j,
                            &mut tiltaxis,
                            &mut axis8,
                        ) == 0
                        && mask & (1 << 12) != 0
                    {
                        tiltaxis = (axis8 * get_fei_ext_head_angle_scale(&extended_data)) as f32;
                        if (-360.0..=360.0).contains(&tiltaxis) {
                            if tiltaxis < -180.0 {
                                tiltaxis += 360.0;
                            }
                            if tiltaxis > 180.0 {
                                tiltaxis -= 360.0;
                            }
                            out.push_str(&format!(
                                "          Tilt axis rotation angle = {:7.1}{}\n",
                                -tiltaxis, " (Corrected sign)"
                            ));
                            found_axis_rot = true;
                        }
                    }
                }
                // SerialEM type
                if extra_is_nbytes_and_flags(num_int, num_real) != 0 {
                    let type_name = [
                        "Tilt angles",
                        "Piece coordinates",
                        "Stage positions",
                        "Magnifications",
                        "Intensities",
                        "Exposure doses",
                    ];
                    let extract_com = [
                        "extracttilts",
                        "extractpieces",
                        "extracttilts -stage",
                        "extracttilts -mag",
                        "extracttilts -int",
                        "extracttilts -exp",
                    ];
                    out.push_str("\nExtended header from SerialEM contains:\n");
                    for j in 0..6 {
                        if (num_real / (1 << j)) % 2 != 0 {
                            out.push_str(&format!(
                                "  {:17} - Extract with \"{}\"\n",
                                type_name[j], extract_com[j]
                            ));
                        }
                    }
                } else {
                    let mut tilts = vec![0.0_f32; nxyz[2].max(0) as usize + 9];
                    let mut iz_piece = vec![0_i32; nxyz[2].max(0) as usize + 9];
                    for j in 0..nxyz[2].max(0) as usize {
                        iz_piece[j] = j as i32;
                    }
                    let mut ierr = 0;
                    if get_extra_header_items(
                        &extended_data,
                        nbsym,
                        num_int,
                        num_real,
                        nxyz[2],
                        1,
                        &mut tilts,
                        None,
                        &mut ierr,
                        &iz_piece,
                    ) != 0
                    {
                        result = Err(vec![format!("ERROR: HEADER - ERROR: {}", b3d_get_error())]);
                        break 'extended;
                    }
                    if ierr > 0 {
                        out.push_str(
                            "Extended header has tilt angles - extract with \"extracttilts\"\n",
                        );
                    }
                }
            }

            if result.is_ok() {
                // If no axis rotation in extended header, look for it in labels
                if !found_axis_rot {
                    for label in labels.iter().take(num_labels.clamp(0, 10) as usize) {
                        if String::from_utf8_lossy(&label[..80]).contains("Tilt axis angle") {
                            found_axis_rot = true;
                            break;
                        }
                    }
                }
                // if no pixel in extended header,
                if !found_pixel {
                    found_pixel = delta[0] != 1.0 || delta[1] != 1.0 || delta[2] != 1.0;
                }
                // Look for mdoc file in either case
                if !found_pixel || !found_axis_rot {
                    let (mut montage, mut num_sect, mut i_type_adoc) = (0, 0, 0);
                    let ind_adoc = adoc_open_image_metadata(
                        in_file.as_bytes(),
                        1,
                        &mut montage,
                        &mut num_sect,
                        &mut i_type_adoc,
                    );
                    if ind_adoc >= 0 {
                        if !found_pixel {
                            let mut pixel = 0.0_f32;
                            if adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut pixel) == 0
                            {
                                out.push_str(&format!(
                                    "          Pixel size in nanometers ={}{}\n",
                                    g_edit(pixel / 10.0, 11, 4),
                                    "  , from mdoc"
                                ));
                            }
                        }
                        if !found_axis_rot {
                            let num_labels = adoc_get_number_of_sections(b"T").unwrap_or(-1);
                            for j in 0..num_labels {
                                let Ok(name) = adoc_get_section_name(b"T", j) else {
                                    continue;
                                };
                                let temp_label_str = String::from_utf8_lossy(&name);
                                let fei_label = temp_label_str.contains("TiltAxisAngle");
                                if !(fei_label || temp_label_str.contains("Tilt axis angle")) {
                                    continue;
                                }
                                if let Some((_, rest)) = temp_label_str.split_once('=') {
                                    let extract = rest.trim_start();
                                    let end = extract
                                        .find([',', ' ', '\t', '/'])
                                        .unwrap_or(extract.len());
                                    if let Ok(tilt_axis) = extract[..end].parse::<f32>() {
                                        if fei_label {
                                            let mut rot_angle = 0.0_f32;
                                            if adoc_get_float(
                                                ADOC_ZVALUE_NAME,
                                                0,
                                                b"RotationAngle",
                                                &mut rot_angle,
                                            ) == 0
                                            {
                                                if (-(rot_angle + 90.0) - tilt_axis).abs() < 0.11 {
                                                    out.push_str(&format!(
                                                        "          Tilt axis rotation angle = {:7.1}{}\n",
                                                        rot_angle, "  (from RotationAngle in mdoc)"
                                                    ));
                                                } else if ((rot_angle - 90.0) - tilt_axis).abs()
                                                    < 0.11
                                                {
                                                    out.push_str(&format!(
                                                        "          Tilt axis rotation angle = {:7.1}{}\n",
                                                        tilt_axis, "  (from mdoc)"
                                                    ));
                                                } else if (-(rot_angle - 90.0) - tilt_axis).abs()
                                                    < 0.11
                                                {
                                                    out.push_str(&format!(
                                                        "          Tilt axis rotation angle = {:7.1}{}\n",
                                                        -tilt_axis, "  (corrected sign, from mdoc)"
                                                    ));
                                                }
                                            }
                                        } else {
                                            out.push_str(&format!(
                                                "          Tilt axis rotation angle = {:7.1}{}\n",
                                                tilt_axis, "  (from mdoc)"
                                            ));
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        // The `header` process ended here and took its
                        // autodoc with it; in process it must be released.
                        adoc_clear(ind_adoc);
                    }
                }
            }
        }
        iiu_close(im_unit);
    }
    iiu_exit_on_error(saved_exit, -1);
    b3d_set_store_error(saved_store);
    match result {
        Ok(()) => Ok(out.lines().map(str::to_owned).collect()),
        Err(messages) => Err(fail(messages)),
    }
}

/// Matches `getmrc` (`IMOD/pysrc/imodpy.py:456`); the `header` process is replaced by
/// [`header_in_process`] (owner decision 2026-09-24).
pub fn get_mrc(file: &str, do_all: bool, angle_line_values: bool) -> Result<MrcInfo, ImodpyError> {
    if angle_line_values {
        // `imodpy.py:477`: `runcmd("header -StandardInput", input)` with
        // `input = ["InputFile " + file]`.  Owner decision (2026-09-24): our
        // own `header` runs in process instead of through `sh -c`/`PATH`.
        let lines = header_in_process("header -StandardInput", file, true, None)?;
        let mut values = [None; 5];
        for line in lines
            .iter()
            .filter(|line| line.to_ascii_lowercase().contains("axis") && line.contains("angle"))
        {
            let tokens = multi_char_split(line, " ,=");
            for index in 0..tokens.len().saturating_sub(1) {
                match tokens[index].as_str() {
                    "angle" | "binning" | "bidir" => {
                        values[["angle", "binning", "spot", "camera", "bidir"]
                            .iter()
                            .position(|key| *key == tokens[index])
                            .unwrap()] = tokens[index + 1].parse().ok()
                    }
                    "spot" | "camera" => {
                        values[["angle", "binning", "spot", "camera", "bidir"]
                            .iter()
                            .position(|key| *key == tokens[index])
                            .unwrap()] = tokens[index + 1]
                            .parse::<i32>()
                            .ok()
                            .map(|value| value as f32)
                    }
                    _ => {}
                }
            }
        }
        return Ok(MrcInfo::AngleLines(values));
    }
    let command = if do_all {
        "header -si -mo -pi -ori -min -max -mean -StandardInput"
    } else {
        "header -si -mo -pi -StandardInput"
    };
    // `imodpy.py:500-505`: `runcmd(command, input)`, now in process (owner
    // decision 2026-09-24); the lines carry the `3i8`/`i4`/`3g15.5`/`g13.5`
    // text `header` prints, so the values parsed below are rounded as before.
    let mut lines = header_in_process(command, file, true, Some(do_all))?;
    let needed = if do_all { 7 } else { 3 };
    while lines.len() >= needed
        && (lines[0].trim().is_empty()
            || ["a", "e", "i", "o"]
                .iter()
                .any(|letter| lines[0].contains(letter)))
    {
        lines.remove(0);
    }
    if lines.len() < needed {
        return Err(ImodpyError {
            arguments: vec![format!("header {file}: too few lines of output")],
        });
    }
    let dimensions = lines[0]
        .split_whitespace()
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ImodpyError {
            arguments: vec![format!("header {file}: invalid -si output")],
        })?;
    let pixels = lines[2]
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ImodpyError {
            arguments: vec![format!("header {file}: invalid -pi output")],
        })?;
    if dimensions.len() < 3 || pixels.len() < 3 {
        return Err(ImodpyError {
            arguments: vec![format!("header {file}: too few numbers")],
        });
    }
    let mode = lines[1].trim().parse::<i32>().map_err(|_| ImodpyError {
        arguments: vec![format!("header {file}: invalid mode")],
    })?;
    if !do_all {
        return Ok(MrcInfo::Basic(
            dimensions[0],
            dimensions[1],
            dimensions[2],
            mode,
            pixels[0],
            pixels[1],
            pixels[2],
        ));
    }
    let origin = lines[3]
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ImodpyError {
            arguments: vec![format!("header {file}: invalid origin")],
        })?;
    if origin.len() < 3 {
        return Err(ImodpyError {
            arguments: vec![format!("header {file}: too few origin numbers")],
        });
    }
    Ok(MrcInfo::All(
        dimensions[0],
        dimensions[1],
        dimensions[2],
        mode,
        pixels[0],
        pixels[1],
        pixels[2],
        origin[0],
        origin[1],
        origin[2],
        lines[4].trim().parse().unwrap_or(0.),
        lines[5].trim().parse().unwrap_or(0.),
        lines[6].trim().parse().unwrap_or(0.),
    ))
}

/// Matches `getmrcsize` (`IMOD/pysrc/imodpy.py:549`).
pub fn get_mrc_size(file: &str) -> Result<(i32, i32, i32), ImodpyError> {
    match get_mrc(file, false, false)? {
        MrcInfo::Basic(x, y, z, ..) => Ok((x, y, z)),
        _ => unreachable!(),
    }
}

/// Matches `getmrcpixel` (`IMOD/pysrc/imodpy.py:557`); the `header` process is replaced by
/// [`header_in_process`] (owner decision 2026-09-24).
pub fn get_mrc_pixel(file: &str) -> Result<f32, ImodpyError> {
    // `imodpy.py:565`: `runcmd("header -StandardInput", input)`, now in
    // process; the `Pixel spacing` (g11.4) and `size in nanometers` (g11.4)
    // lines are the text `header` prints, so the pixel keeps its rounding.
    let lines = header_in_process("header -StandardInput", file, true, None)?;
    let mut pixel = None;
    for line in lines {
        if let Some(value) = line
            .split(".. ")
            .nth(1)
            .filter(|_| line.contains("Pixel spacing"))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<f32>().ok())
        {
            pixel = Some(value);
        }
        if let Some(value) = line
            .split('=')
            .nth(1)
            .filter(|_| line.contains("size in nanometers ="))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<f32>().ok())
        {
            pixel = Some(10. * value);
        }
    }
    pixel.ok_or_else(|| ImodpyError {
        arguments: vec![format!("header {file}: cannot find pixel size")],
    })
}

/// Matches `getMontageSize` (`IMOD/pysrc/imodpy.py:596`); `montagesize` remains external.
pub fn get_montage_size(
    stack: &str,
    piece_list: Option<&str>,
) -> Result<(i32, i32, i32), ImodpyError> {
    let mut command = format!("montagesize \"{stack}\"");
    if let Some(piece_list) = piece_list.filter(|piece_list| Path::new(piece_list).exists()) {
        command.push_str(&format!(" \"{piece_list}\""));
    }
    let lines = run_cmd(&command, None, None, None, &[])?.unwrap_or_default();
    let line = lines.last().ok_or_else(|| ImodpyError {
        arguments: vec![format!("{command}: No output returned")],
    })?;
    let values = line
        .split_once("NZ:")
        .map(|(_, values)| {
            values
                .split_whitespace()
                .map(str::parse::<i32>)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .ok()
        .flatten()
        .filter(|values| values.len() >= 3)
        .ok_or_else(|| ImodpyError {
            arguments: vec![format!(
                "{command}: Uninterpretable output on line with NZ:"
            )],
        })?;
    Ok((values[0], values[1], values[2]))
}

/// Matches `runGoodframe` (`IMOD/pysrc/imodpy.py:624`); `goodframe` remains external.
pub fn run_goodframe(x_size: i32, y_size: i32) -> (i32, i32) {
    let Ok(Some(lines)) = run_cmd(
        &format!("goodframe {x_size} {y_size}"),
        None,
        None,
        None,
        &[],
    ) else {
        return (-1, -1);
    };
    let Some(line) = lines.last() else {
        return (-2, -2);
    };
    let values = line
        .split_whitespace()
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>();
    match values {
        Ok(values) if values.len() >= 2 => (values[0], values[1]),
        _ => (-2, -2),
    }
}

/// Matches `getImageFormat` (`IMOD/pysrc/imodpy.py:639`); the `header` process is replaced by
/// [`header_in_process`] (owner decision 2026-09-24).
pub fn get_image_format(file: &str) -> Result<String, ImodpyError> {
    // `imodpy.py:645`: `runcmd("header -StandardInput", input)`, now in process.
    let lines = header_in_process("header -StandardInput", file, true, None)?;
    for (description, format) in [
        ("a TIFF", "TIFF"),
        ("an HDF", "HDF"),
        ("a non-MRC", "likeMRC"),
    ] {
        if lines
            .iter()
            .take(9)
            .any(|line| line.contains(&format!("This is {description}")))
        {
            return Ok(format.to_owned());
        }
    }
    Ok("MRC".to_owned())
}

/// Matches `readTextFile` (`IMOD/pysrc/imodpy.py:1080`).
pub fn read_text_file(
    filename: &str,
    description: Option<&str>,
    return_on_error: bool,
    maximum_lines: Option<usize>,
) -> Result<Vec<String>, String> {
    match fs::read_to_string(filename) {
        Ok(text) => Ok(text
            .lines()
            .take(maximum_lines.unwrap_or(usize::MAX))
            .map(|line| line.trim_end_matches([' ', '\t', '\r', '\n']).to_owned())
            .collect()),
        Err(error) => {
            let message = format!(
                "Opening {} {}: {error}",
                description.unwrap_or(" "),
                filename
            );
            if return_on_error {
                Err(message)
            } else {
                crate::imod::pysrc::pip::exit_error(&message)
            }
        }
    }
}

/// Matches `writeTextFile` (`IMOD/pysrc/imodpy.py:1127`).
pub fn write_text_file(
    filename: &str,
    strings: &[String],
    return_on_error: bool,
) -> Result<(), String> {
    let mut contents = String::new();
    for line in strings {
        contents.push_str(line);
        contents.push('\n');
    }
    match fs::write(filename, contents) {
        Ok(()) => Ok(()),
        Err(error) => {
            let message = format!("Opening file: {filename}  - {error}");
            if return_on_error {
                Err(message)
            } else {
                crate::imod::pysrc::pip::exit_error(&message)
            }
        }
    }
}

/// Matches `convertToInteger` (`IMOD/pysrc/imodpy.py:1242`).
pub fn convert_to_integer(value_string: &str, description: &str) -> i32 {
    match value_string.parse() {
        Ok(value) => value,
        Err(_) => crate::imod::pysrc::pip::exit_error(&format!(
            "Converting {description} ({value_string}) to integer"
        )),
    }
}

/// Matches `optionValue` (`IMOD/pysrc/imodpy.py:1153`).
pub fn option_value(
    lines: &[String],
    option: &str,
    value_type: i32,
    ignore_case: bool,
    number_values: usize,
    other_separator: Option<char>,
    empty_return: Option<&str>,
) -> Option<OptionValue> {
    let mut result = None;
    for line in lines {
        let trimmed = line.trim_start();
        let matches = if ignore_case {
            trimmed[..trimmed.len().min(option.len())].eq_ignore_ascii_case(option)
        } else {
            trimmed.starts_with(option)
        };
        if !matches
            || trimmed.starts_with(&format!("# {option}"))
            || trimmed.starts_with(&format!("#{option}"))
        {
            continue;
        }
        let after_option = &trimmed[option.len()..];
        let value = if let Some(separator) = other_separator {
            after_option
                .split_once(separator)
                .map(|(_, value)| value)
                .unwrap_or("")
        } else {
            after_option.trim_start()
        };
        let value = value.split('#').next().unwrap_or("").trim();
        if value_type > 2 {
            let lower = value.to_ascii_lowercase();
            result = match lower.as_str() {
                "0" | "f" | "off" | "false" => Some(OptionValue::Boolean(false)),
                "1" | "t" | "on" | "true" | "" => Some(OptionValue::Boolean(true)),
                _ if value == option => Some(OptionValue::Boolean(true)),
                _ => {
                    prnstr(
                        &format!(
                            "WARNING: optionValue - Boolean entry found with improper value ({lower}) in: {line}"
                        ),
                        "\n",
                        false,
                    );
                    result
                }
            };
        } else if value.is_empty() {
            result = empty_return.map(|entry| OptionValue::String(entry.to_owned()));
            if result.is_none() {
                prnstr(
                    &format!("WARNING: optionValue - No value for option in: {line}"),
                    "\n",
                    false,
                );
            }
        } else if value_type <= 0 {
            result = Some(OptionValue::String(value.to_owned()));
        } else {
            let replaced = value.replace(',', " ");
            let entries = replaced.split_whitespace().collect::<Vec<_>>();
            if number_values != 0 && entries.len() < number_values {
                return None;
            }
            let limit = if number_values == 0 {
                entries.len()
            } else {
                number_values
            };
            if value_type == 1 {
                match entries[..limit]
                    .iter()
                    .map(|entry| entry.parse::<i32>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(values) => result = Some(OptionValue::Integers(values)),
                    Err(_) => {
                        prnstr(
                            &format!(
                                "WARNING: optionValue - Bad character in numeric entry in: {line}"
                            ),
                            "\n",
                            false,
                        );
                        return None;
                    }
                }
            } else {
                match entries[..limit]
                    .iter()
                    .map(|entry| entry.parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(values) => result = Some(OptionValue::Floats(values)),
                    Err(_) => {
                        prnstr(
                            &format!(
                                "WARNING: optionValue - Bad character in numeric entry in: {line}"
                            ),
                            "\n",
                            false,
                        );
                        return None;
                    }
                }
            }
        }
    }
    result
}

/// Matches `completeAndCheckComFile` (`IMOD/pysrc/imodpy.py:1251`).
pub fn complete_and_check_com_file(command_file: &str) -> (String, String) {
    if command_file.is_empty() {
        crate::imod::pysrc::pip::exit_error("A command file must be entered");
    }
    let (root, complete) = if command_file.ends_with(".com") || command_file.ends_with(".pcm") {
        (
            command_file[..command_file.len() - 4].to_owned(),
            command_file.to_owned(),
        )
    } else {
        let root = command_file.trim_end_matches('.').to_owned();
        let com = format!("{root}.com");
        let pcm = format!("{root}.pcm");
        if Path::new(&com).exists() && Path::new(&pcm).exists() {
            crate::imod::pysrc::pip::exit_error(&format!(
                "Both {com} and {pcm} exist; specify which"
            ));
        }
        if Path::new(&com).exists() {
            (root, com)
        } else if Path::new(&pcm).exists() {
            (root, pcm)
        } else {
            crate::imod::pysrc::pip::exit_error(&format!("Neither {com} nor {pcm} exists"));
        }
    };
    (root, complete)
}

/// Matches `cleanupFiles` (`IMOD/pysrc/imodpy.py:1317`).
pub fn cleanup_files(files: &[String]) {
    let mut remaining = files.to_vec();
    for _ in 0..10 {
        remaining
            .retain(|filename| fs::remove_file(filename).is_err() && Path::new(filename).exists());
        if remaining.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Matches `cleanChunkFiles` (`IMOD/pysrc/imodpy.py:1296`).
pub fn clean_chunk_files(root_name: &str, log_only: bool) {
    let directory = Path::new(root_name)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let base = Path::new(root_name)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let suffix = name.strip_prefix(&format!("{base}-"));
            let numbered = suffix.is_some_and(|suffix| {
                suffix
                    .as_bytes()
                    .get(0..3)
                    .is_some_and(|digits| digits.iter().all(u8::is_ascii_digit))
            });
            let start_or_finish = name == format!("{base}-start.log")
                || name == format!("{base}-finish.log")
                || (!log_only
                    && ["com", "pcm"].iter().any(|extension| {
                        name == format!("{base}-start.{extension}")
                            || name == format!("{base}-finish.{extension}")
                    }));
            let selected = (numbered
                && (name.ends_with(".log")
                    || (!log_only && (name.ends_with(".com") || name.ends_with(".pcm")))))
                || start_or_finish;
            if selected {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

/// Matches `balancedGroupLimits` (`IMOD/pysrc/imodpy.py:1792`).
pub fn balanced_group_limits(total: i32, groups: i32, group_index: i32) -> (i32, i32) {
    let base = total / groups;
    let remainder = total % groups;
    let start = group_index * base + group_index.min(remainder);
    let end = (group_index + 1) * base + (group_index + 1).min(remainder) - 1;
    (start, end)
}

/// Matches `fmtstr` (`IMOD/pysrc/imodpy.py:1801`) on supported modern Python versions.
pub fn fmtstr(format_string: &str, values: &[String]) -> String {
    let mut result = String::new();
    let mut value_index = 0usize;
    let mut characters = format_string.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '{' && characters.peek() == Some(&'}') {
            characters.next();
            result.push_str(values.get(value_index).map(String::as_str).unwrap_or(""));
            value_index += 1;
        } else {
            result.push(character);
        }
    }
    result
}

/// Matches `prnstr` (`IMOD/pysrc/imodpy.py:1897`) for standard output.
pub fn prnstr(string: &str, end: &str, flush: bool) {
    print!("{string}{end}");
    if flush {
        let _ = std::io::stdout().flush();
    }
}

/// Matches `datasetFilename` (`IMOD/pysrc/imodpy.py:657`).
pub fn dataset_filename(suffix: &str, root: Option<&str>, type_extension: Option<&str>) -> String {
    let root = root
        .map(str::to_owned)
        .unwrap_or_else(|| CURRENT_ROOTNAME.lock().expect("imodpy root mutex").clone());
    let type_extension = type_extension.map(str::to_owned).unwrap_or_else(|| {
        FILE_TYPE_EXTENSION
            .lock()
            .expect("imodpy extension mutex")
            .clone()
    });
    if type_extension.is_empty() {
        return root + suffix;
    }
    let suffix = match suffix.rfind('.') {
        Some(index) => format!("{}_{}", &suffix[..index], &suffix[index + 1..]),
        None => suffix.to_owned(),
    };
    format!("{root}{suffix}.{type_extension}")
}

/// Matches `setRootAndExtension` (`IMOD/pysrc/imodpy.py:672`).
pub fn set_root_and_extension(root: &str, extension: &str) {
    *CURRENT_ROOTNAME.lock().expect("imodpy root mutex") = root.to_owned();
    *FILE_TYPE_EXTENSION.lock().expect("imodpy extension mutex") = extension.to_owned();
}

/// Matches `findRootAxisAndExtensions` (`IMOD/pysrc/imodpy.py:689`).
pub fn find_root_axis_and_extensions(
    force_single: i32,
    use_tilt: Option<&str>,
) -> (String, i32, String, Option<String>, String) {
    let mut root = String::new();
    let mut type_extension = None;
    let mut stack_extension = String::new();
    let mut dual_number = 0;
    let mut tilt_sums = [0i32; 2];
    let mut eraser_sums = [0i32; 2];
    let mut tilt_extension = String::new();
    let mut eraser_extension = String::new();
    let mut passed_tilt = use_tilt
        .filter(|file| Path::new(file).exists())
        .map(str::to_owned);
    let mut tried_extension = None::<String>;
    if let Some(file) = &passed_tilt {
        if let Some(extension) = Path::new(file)
            .extension()
            .map(|extension| extension.to_string_lossy().into_owned())
            .filter(|extension| extension == "com" || extension == "pcm")
        {
            tilt_extension = extension.clone();
            tried_extension = Some(extension);
        } else {
            passed_tilt = None;
        }
    }
    for (index, extension) in ["com", "pcm"].iter().enumerate() {
        if force_single >= 0 {
            if tried_extension.is_none() && Path::new(&format!("tilt.{extension}")).exists() {
                tilt_extension = (*extension).to_owned();
                tilt_sums[index] += 1;
            }
            if Path::new(&format!("eraser.{extension}")).exists() {
                eraser_extension = (*extension).to_owned();
                eraser_sums[index] += 1;
            }
        }
        if force_single <= 0 {
            if tried_extension.is_none()
                && Path::new(&format!("tilta.{extension}")).exists()
                && Path::new(&format!("tiltb.{extension}")).exists()
            {
                tilt_extension = (*extension).to_owned();
                tilt_sums[index] += 2;
            }
            if Path::new(&format!("erasera.{extension}")).exists()
                && Path::new(&format!("eraserb.{extension}")).exists()
            {
                eraser_extension = (*extension).to_owned();
                eraser_sums[index] += 2;
            }
        }
    }
    if tilt_sums.iter().min().copied().unwrap_or(0) > 0
        || eraser_sums.iter().min().copied().unwrap_or(0) > 0
        || tilt_sums.iter().max().copied().unwrap_or(0) > 2
        || eraser_sums.iter().max().copied().unwrap_or(0) > 2
        || (tilt_sums.iter().max().copied().unwrap_or(0) > 0
            && eraser_sums.iter().max().copied().unwrap_or(0) > 0
            && (tilt_sums.iter().max() != eraser_sums.iter().max()
                || tilt_extension != eraser_extension))
    {
        return (String::new(), -1, String::new(), None, String::new());
    }
    let command_extension = if !tilt_extension.is_empty() {
        tilt_extension.clone()
    } else {
        eraser_extension.clone()
    };
    let mut use_extension = format!(".{command_extension}");
    if tilt_sums.iter().max().copied().unwrap_or(0) > 1
        || eraser_sums.iter().max().copied().unwrap_or(0) > 1
    {
        dual_number = 2;
        use_extension = format!("a.{command_extension}");
    }
    let mut tilt_root_failed = false;
    if !tilt_extension.is_empty() {
        let tilt_file = passed_tilt.unwrap_or_else(|| format!("tilt{use_extension}"));
        let tilt_lines = read_text_file(&tilt_file, None, true, None).ok();
        let track_lines = read_text_file(&format!("track{use_extension}"), None, true, None).ok();
        let input_line = tilt_lines.as_ref().and_then(|lines| {
            match option_value(lines, "InputProjections", 0, false, 0, None, None) {
                Some(OptionValue::String(value)) => Some(value),
                _ => None,
            }
        });
        let image_line = track_lines.as_ref().and_then(|lines| {
            match option_value(lines, "ImageFile", 0, false, 0, None, None) {
                Some(OptionValue::String(value)) => Some(value),
                _ => None,
            }
        });
        if let (Some(input), Some(image)) = (input_line, image_line) {
            let input_path = Path::new(&input);
            let image_path = Path::new(&image);
            let mut input_root = input_path.with_extension("").to_string_lossy().into_owned();
            let mut image_root = image_path.with_extension("").to_string_lossy().into_owned();
            let input_extension = input_path
                .extension()
                .map(|extension| extension.to_string_lossy().into_owned())
                .unwrap_or_default();
            let image_extension = image_path
                .extension()
                .map(|extension| extension.to_string_lossy().into_owned())
                .unwrap_or_default();
            if input_extension == "ali" && image_extension == "preali" {
                type_extension = Some(String::new());
            } else if !input_extension.is_empty()
                && input_extension == image_extension
                && input_root.ends_with("ali")
                && image_root.ends_with("preali")
            {
                type_extension = Some(input_extension);
                input_root.truncate(input_root.len() - 3);
                image_root.truncate(image_root.len() - 6);
            }
            if dual_number != 0 {
                if input_root.ends_with('a') || input_root.ends_with('b') {
                    input_root.pop();
                    image_root.pop();
                } else {
                    tilt_root_failed = true;
                }
            }
            if input_root == image_root && !tilt_root_failed {
                root = input_root;
            } else {
                tilt_root_failed = true;
            }
        }
    }
    if !eraser_extension.is_empty()
        && let Ok(lines) = read_text_file(&format!("eraser{use_extension}"), None, true, None)
    {
        if let Some(OptionValue::String(input)) =
            option_value(&lines, "InputFile", 0, false, 0, None, None)
        {
            let path = Path::new(&input);
            let root2 = path.with_extension("").to_string_lossy().into_owned();
            stack_extension = path
                .extension()
                .map(|extension| extension.to_string_lossy().into_owned())
                .unwrap_or_default();
            if root.is_empty() && tilt_root_failed {
                if dual_number == 0 {
                    root = root2;
                } else if root2.ends_with('a') {
                    root = root2[..root2.len() - 1].to_owned();
                }
            }
        }
    }
    (
        command_extension,
        dual_number,
        root,
        type_extension,
        stack_extension,
    )
}

/// Matches `defaultNamingStyle` (`IMOD/pysrc/imodpy.py:866`).
pub fn default_naming_style() -> (i32, String) {
    (1, "mrc".to_owned())
}

/// Matches `getNamingStyle` (`IMOD/pysrc/imodpy.py:846`).
pub fn get_naming_style(
    default: Option<&str>,
    require: bool,
    allow_extra: bool,
) -> Result<(i32, Option<String>), String> {
    let style = crate::imod::pysrc::pip::pip_get_integer("NamingStyle", -1)
        .map_err(|_| "Cannot get NamingStyle option".to_owned())?;
    let maximum = if allow_extra { 3 } else { 2 };
    if style > maximum {
        return Err(format!("Naming style must be less than {maximum}"));
    }
    if style < 0 && default.is_none() && require {
        return Err(
            "Cannot determine file naming style from the command file; you must enter -name"
                .to_owned(),
        );
    }
    let extension = if style >= 0 {
        get_type_ext_allowing_extra(style)
    } else {
        default.map(str::to_owned)
    };
    Ok((style, extension))
}

/// Matches `comExtensionFromOption` (`IMOD/pysrc/imodpy.py:872`).
pub fn com_extension_from_option(default_value: i32) -> String {
    match crate::imod::pysrc::pip::pip_get_integer("MakeComExtensionPcm", default_value)
        .unwrap_or(default_value)
    {
        value if value > 0 => ".pcm".to_owned(),
        value if value < 0 => ".com".to_owned(),
        _ => ".com".to_owned(),
    }
}

/// Matches `defaultComExtension` (`IMOD/pysrc/imodpy.py:883`).
pub fn default_com_extension() -> String {
    "com".to_owned()
}

/// Matches `standardTypeExtensions` (`IMOD/pysrc/imodpy.py:888`).
pub fn standard_type_extensions() -> Vec<String> {
    vec![String::new(), "mrc".to_owned(), "hdf".to_owned()]
}

/// Matches `getTypeExtAllowingExtra` (`IMOD/pysrc/imodpy.py:893`).
pub fn get_type_ext_allowing_extra(naming_style: i32) -> Option<String> {
    match naming_style {
        0 => Some(String::new()),
        1 => Some("mrc".to_owned()),
        2 => Some("hdf".to_owned()),
        3 => Some("tif".to_owned()),
        _ => None,
    }
}

/// Matches `allowedRawStackExtensions` (`IMOD/pysrc/imodpy.py:903`).
pub fn allowed_raw_stack_extensions() -> Vec<String> {
    vec![
        "st".to_owned(),
        "mrc".to_owned(),
        "hdf".to_owned(),
        "tif".to_owned(),
        "tiff".to_owned(),
    ]
}

/// Matches `mapTypeExtensionToStyle` (`IMOD/pysrc/imodpy.py:912`).
pub fn map_type_extension_to_style(type_extension: &str, allow_extra: bool) -> i32 {
    if type_extension.is_empty() {
        return 0;
    }
    for (index, extension) in standard_type_extensions().iter().enumerate() {
        if type_extension == extension {
            return index as i32;
        }
    }
    if allow_extra && type_extension == "tif" {
        return 3;
    }
    0
}

/// Matches `addOutputFormatVarToLines` (`IMOD/pysrc/imodpy.py:928`).
pub fn add_output_format_var_to_lines(
    lines: &mut Vec<String>,
    naming_style: i32,
    output_format: Option<&str>,
    allow_extra: bool,
) {
    let output_format = output_format.map(str::to_owned).or_else(|| {
        if naming_style > 0 {
            get_type_ext_allowing_extra(naming_style)
                .filter(|extension| allow_extra || extension != "tif")
                .map(|extension| extension.to_uppercase())
        } else {
            None
        }
    });
    if let Some(output_format) = output_format {
        let line = format!("$setenv IMOD_OUTPUT_FORMAT {output_format}");
        match lines.iter().position(|entry| !entry.starts_with('#')) {
            Some(index) => lines.insert(index, line),
            None => lines.push(line),
        }
    }
}

/// Matches `setOutputFormatIfNeeded` (`IMOD/pysrc/imodpy.py:948`).
pub fn set_output_format_if_needed(type_extension: &str, allow_extra: bool) {
    let current = std::env::var("IMOD_OUTPUT_FORMAT").ok();
    let output = if !type_extension.is_empty() {
        Some(type_extension.to_ascii_uppercase())
    } else if current.as_deref().is_some_and(|format| {
        !standard_type_extensions()
            .iter()
            .any(|extension| extension.eq_ignore_ascii_case(format))
            && (!allow_extra || !format.eq_ignore_ascii_case("tif"))
    }) {
        Some("MRC".to_owned())
    } else {
        None
    };
    if let Some(output) = output {
        unsafe {
            std::env::set_var("IMOD_OUTPUT_FORMAT", output);
        }
    }
}

/// Matches `makeBackupFile` (`IMOD/pysrc/imodpy.py:961`).
pub fn make_backup_file(filename: &str) {
    if Path::new(filename).exists() {
        let backup = format!("{filename}~");
        if Path::new(&backup).exists() {
            let _ = fs::remove_file(&backup);
        }
        if let Err(error) = fs::rename(filename, &backup) {
            prnstr(
                &format!("WARNING: Failed to rename existing file {filename} to {backup}: {error}"),
                "\n",
                false,
            );
        }
    }
}

/// Matches `isFileNewer` (`IMOD/pysrc/imodpy.py:975`).
pub fn is_file_newer(test_file: &str, old_file: &str) -> bool {
    fs::metadata(test_file)
        .and_then(|test| fs::metadata(old_file).map(|old| (test, old)))
        .and_then(|(test, old)| {
            test.modified()
                .and_then(|test_time| old.modified().map(|old_time| test_time > old_time))
        })
        .unwrap_or(false)
}

/// Matches `exitFromImodError` (`IMOD/pysrc/imodpy.py:981`).
pub fn exit_from_imod_error(program_name: &str) -> ! {
    let errors = get_err_strings();
    let mut prior = None;
    for line in errors {
        if let Some(previous) = prior.replace(line) {
            // `prnstr(line, end='')`: the Python lines keep their endings;
            // producers here that stored a line without one get it back.
            let end = if previous.ends_with('\n') { "" } else { "\n" };
            prnstr(&previous, end, false);
        }
    }
    // `prnstr("ERROR: " + pn + " - " + line, end='')` on stdout; the stored
    // final line carries no line ending here, so one is written.
    prnstr(
        &format!("ERROR: {program_name} - {}", prior.unwrap_or_default()),
        "\n",
        true,
    );
    std::process::exit(1)
}

/// Matches `parselist` (`IMOD/pysrc/imodpy.py:994`).
pub fn parse_list(line: &str) -> Option<Vec<i32>> {
    let mut list = Vec::new();
    let mut dash_last = false;
    let mut negative_number = false;
    let mut got_comma = false;
    let mut got_number = false;
    let characters = line.as_bytes();
    if characters.is_empty() {
        return Some(list);
    }
    if characters[0] == b'/' {
        return None;
    }
    let mut index = 0usize;
    let mut last_number = 0i32;
    while index < characters.len() {
        let next = characters[index];
        if next.is_ascii_digit() {
            got_number = true;
            let mut number_start = index;
            while index < characters.len() && characters[index].is_ascii_digit() {
                index += 1;
            }
            if negative_number {
                number_start -= 1;
            }
            let number = line[number_start..index].parse::<i32>().ok()?;
            let increment = if dash_last && last_number > number {
                -1
            } else {
                1
            };
            let mut value = if dash_last {
                last_number + increment
            } else {
                number
            };
            while increment * value <= increment * number {
                list.push(value);
                value += increment;
            }
            last_number = number;
            negative_number = false;
            dash_last = false;
            got_comma = false;
            continue;
        }
        if next != b',' && next != b' ' && next != b'-' {
            return None;
        }
        if next == b',' {
            got_comma = true;
        }
        if next == b'-' {
            if dash_last || !got_number || got_comma {
                negative_number = true;
            } else {
                dash_last = true;
            }
        }
        index += 1;
    }
    Some(list)
}

/// Matches `extractProgramEntries` (`IMOD/pysrc/imodpy.py:1345`).
pub fn extract_program_entries(
    command_lines: &[String],
    program_name: &str,
    start_key: &str,
) -> Option<Vec<String>> {
    let mut start_line = None;
    let mut end_line = command_lines.len();
    for (index, line) in command_lines.iter().enumerate() {
        let line = line.trim();
        if line.starts_with('$') {
            if start_line.is_some() {
                end_line = index;
                break;
            }
            if line.contains(program_name) && line.contains(start_key) {
                start_line = Some(index + 1);
            }
        }
    }
    start_line.map(|start_line| command_lines[start_line..end_line].to_vec())
}

/// Matches `getIMODversion` (`IMOD/pysrc/imodpy.py:1421`).
///
/// `imodpy.py:1423` ran `imodinfo` with no arguments and took the third word
/// of its first line, `imodVersion`'s `"%s Version %s %s %s\n"` with
/// `VERSION_NAME` (`b3dutil.c:151`).  Owner decision (2026-09-24): the
/// process is gone and that word is returned directly.  `VERSION_NAME` is generated from
/// `IMOD/.version` (`setup2:411`) and `imod_version` in `b3dutil.rs` writes
/// it inline, so it is repeated here; the two must change together.
pub fn get_imod_version() -> Option<String> {
    Some("5.2.17".to_owned())
}

/// Matches `imodIsAbsPath` (`IMOD/pysrc/imodpy.py:1431`) outside Cygwin's `cygpath` boundary.
pub fn imod_is_abs_path(path: &str) -> bool {
    Path::new(path).is_absolute()
}

/// Matches `imodAbsPath` (`IMOD/pysrc/imodpy.py:1443`).
pub fn imod_abs_path(path: &str) -> String {
    // Cygwin will not work with a windows path, so convert it
    let cygwin = cygwin_path(path);
    let mut absp = std::path::absolute(&cygwin)
        .unwrap_or_else(|_| Path::new(&cygwin).to_path_buf())
        .to_string_lossy()
        .into_owned();
    if cfg!(windows) || cfg!(target_os = "cygwin") {
        absp = get_cygpath(true, &absp, "-m");
    }
    absp
}

/// Matches `newPsutilAPI` (`IMOD/pysrc/imodpy.py:1453`).
pub fn new_psutil_api(version: &str) -> bool {
    version
        .split('.')
        .next()
        .and_then(|part| part.parse::<i32>().ok())
        .is_some_and(|major| major > 1)
}

/// Matches `imodTempDir` (`IMOD/pysrc/imodpy.py:1486`).
///
/// `os.access(path, os.W_OK)` is read here as "some write permission bit is
/// set", which is what `std` exposes without a `libc::access` call.
pub fn imod_temp_dir() -> String {
    let windows = cfg!(windows) || cfg!(target_os = "cygwin");
    if let Some(imodtemp) = std::env::var_os("IMOD_TMPDIR") {
        let imodtemp = get_cygpath(windows, &imodtemp.to_string_lossy(), "-m");
        let path = Path::new(&imodtemp);
        if path.exists()
            && path.is_dir()
            && fs::metadata(path).is_ok_and(|metadata| !metadata.permissions().readonly())
        {
            return imodtemp;
        }
    }
    let imodtemp = get_cygpath(windows, "/usr/tmp", "-m");
    let path = Path::new(&imodtemp);
    if path.exists() && fs::metadata(path).is_ok_and(|metadata| !metadata.permissions().readonly())
    {
        return imodtemp;
    }
    let imodtemp = get_cygpath(windows, "/tmp", "-m");
    let path = Path::new(&imodtemp);
    if path.exists() && fs::metadata(path).is_ok_and(|metadata| !metadata.permissions().readonly())
    {
        return imodtemp;
    }
    ".".to_owned()
}

/// Matches `getCygpath` (`IMOD/pysrc/imodpy.py:1365`).
pub fn get_cygpath(windows: bool, path: &str, type_argument: &str) -> String {
    if windows {
        if let Ok(Some(lines)) = run_cmd(
            &format!("cygpath {type_argument} \"{path}\""),
            None,
            None,
            Some("stdout"),
            &[],
        ) {
            if let Some(line) = lines.first() {
                return line.trim().to_owned();
            }
        }
    }
    path.to_owned()
}

/// Matches `cygwinPath` (`IMOD/pysrc/imodpy.py:1378`); the conversion command remains an external boundary.
pub fn cygwin_path(path: &str) -> String {
    if cfg!(target_os = "cygwin") {
        if let Ok(Some(lines)) = run_cmd(
            &format!("cygpath \"{path}\""),
            None,
            None,
            Some("stdout"),
            &[],
        ) {
            if let Some(line) = lines.first() {
                return line.trim().to_owned();
            }
        }
        let converted = path.replace('\\', "/");
        let bytes = converted.as_bytes();
        if bytes.len() > 2 && bytes[1] == b':' && bytes[2] == b'/' {
            return format!(
                "/cygdrive/{}{}",
                (bytes[0] as char).to_ascii_lowercase(),
                &converted[2..]
            );
        }
        return converted;
    }
    path.to_owned()
}

/// Matches `printPID` (`IMOD/pysrc/imodpy.py:1408`).
pub fn print_pid(do_print: bool) {
    if !do_print {
        return;
    }
    let prefix = if cfg!(windows) {
        "Windows "
    } else if cfg!(target_os = "cygwin") {
        "Cygwin "
    } else {
        ""
    };
    eprint!("{prefix}Python PID: {}\n", std::process::id());
}

/// Matches `addIMODbinIgnoreSIGHUP` (`IMOD/pysrc/imodpy.py:1397`).
pub fn add_imod_bin_ignore_sighup() {
    let Some(directory) = std::env::var_os("IMOD_DIR") else {
        return;
    };
    let bin = Path::new(&cygwin_path(&directory.to_string_lossy())).join("bin");
    let mut path = std::ffi::OsString::from(bin);
    path.push(if cfg!(windows) { ";" } else { ":" });
    path.push(std::env::var_os("PATH").unwrap_or_default());
    unsafe {
        std::env::set_var("PATH", path);
    }
    if !cfg!(windows) {
        unsafe {
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
        }
    }
}

/// Matches `imodNice` (`IMOD/pysrc/imodpy.py:1463`) at the POSIX process-priority boundary.
pub fn imod_nice(nice_increment: i32) -> i32 {
    if cfg!(windows) {
        return if nice_increment < 4 { 0 } else { 1 };
    }
    unsafe {
        libc::nice(nice_increment);
    }
    0
}

/// Matches `setLibPath` (`IMOD/pysrc/imodpy.py:1510`).
pub fn set_lib_path() {
    let Some(qt_directory) = std::env::var_os("IMOD_QTLIBDIR") else {
        return;
    };
    let Some(imod_directory) = std::env::var_os("IMOD_DIR") else {
        return;
    };
    let variable = if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let mut path = std::ffi::OsString::from(&qt_directory);
    path.push(if cfg!(windows) { ";" } else { ":" });
    path.push(Path::new(&imod_directory).join("lib"));
    if let Some(previous) = std::env::var_os(variable) {
        path.push(if cfg!(windows) { ";" } else { ":" });
        path.push(previous);
    }
    unsafe {
        std::env::set_var(variable, path);
    }
    if cfg!(target_os = "macos") {
        let mut frameworks = std::ffi::OsString::from(qt_directory);
        if let Some(previous) = std::env::var_os("DYLD_FRAMEWORK_PATH") {
            frameworks.push(":");
            frameworks.push(previous);
        }
        unsafe {
            std::env::set_var("DYLD_FRAMEWORK_PATH", frameworks);
        }
    }
}

/// Matches `avoidLocalComFile` (`IMOD/pysrc/imodpy.py:1531`).
pub fn avoid_local_com_file(command: &str) -> String {
    if !cfg!(windows) {
        return command.to_owned();
    }
    let mut parts = command.split_whitespace();
    let Some(program) = parts.next() else {
        return command.to_owned();
    };
    if Path::new(program).components().count() != 1
        || !Path::new(&format!("{program}.com")).is_file()
    {
        return command.to_owned();
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let full = directory.join(program);
        if full.is_file()
            || full.with_extension("exe").is_file()
            || full.with_extension("cmd").is_file()
        {
            let mut rewritten = format!("\"{}\"", full.display());
            for argument in parts {
                rewritten.push_str(&format!(" \"{argument}\""));
            }
            return rewritten;
        }
    }
    command.to_owned()
}

/// Matches `makeCurrentDirWritable` (`IMOD/pysrc/imodpy.py:1567`).
pub fn make_current_dir_writable(subdirectory: &str) -> Option<String> {
    if !cfg!(windows) && !cfg!(target_os = "cygwin") {
        return None;
    }
    match run_cmd(
        &format!("chmod u+rwx {subdirectory}"),
        None,
        None,
        Some("stdout"),
        &[],
    ) {
        Ok(_) => None,
        Err(error) => Some(error.to_string()),
    }
}

/// Matches `initializeMoveOrCopy` (`IMOD/pysrc/imodpy.py:1653`).
pub fn initialize_move_or_copy(_skip_windows_copy: i32) {
    MOC_RENAMES_OK.store(true, Ordering::SeqCst);
}

/// Matches `moveOrCopyWithRetry` (`IMOD/pysrc/imodpy.py:1672`); Unix `cp -rf` remains an external process boundary.
pub fn move_or_copy_with_retry(
    from_file: &str,
    to_directory: &str,
    message: &str,
    copy: bool,
    trials: i32,
) -> i32 {
    let source = Path::new(from_file);
    let destination = Path::new(to_directory).join(source.file_name().unwrap_or_default());
    if !copy && MOC_RENAMES_OK.load(Ordering::SeqCst) && fs::rename(source, &destination).is_ok() {
        return 0;
    }
    MOC_RENAMES_OK.store(false, Ordering::SeqCst);
    for trial in 0..trials {
        let result = run_cmd(
            &format!("cp -rf \"{from_file}\" \"{to_directory}\""),
            None,
            None,
            None,
            &[],
        );
        if result.is_ok() {
            if !copy {
                let _ = if source.is_dir() {
                    fs::remove_dir_all(source)
                } else {
                    fs::remove_file(source)
                };
            }
            return 0;
        }
        if trial + 1 < trials {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    prnstr(
        &format!("An error occurred {message} to {to_directory} :"),
        "\n",
        false,
    );
    1
}

/// Matches `patchSizeFromEntry` (`IMOD/pysrc/imodpy.py:1597`).
pub fn patch_size_from_entry(patch_entry: &str) -> (i32, i32, i32, i32) {
    let index = "SMLE".find(patch_entry.to_ascii_uppercase().as_str());
    if let Some(index) = index.filter(|_| patch_entry.len() == 1) {
        let xy = [64, 80, 100, 120][index] as i32;
        return (xy, xy, [32, 40, 50, 60][index], 0);
    }
    let values = patch_entry
        .split(',')
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>();
    match values {
        Ok(values) if values.len() == 3 => (values[0], values[1], values[2], 0),
        _ => (0, 0, 0, 1),
    }
}

/// Matches `autoPatchNumber` (`IMOD/pysrc/imodpy.py:1619`).
pub fn auto_patch_number(
    size: i32,
    lower: i32,
    upper: i32,
    if_z: bool,
    density_index: usize,
) -> i32 {
    let mut delta = [80, 40][density_index];
    if if_z {
        delta = delta.min(3 * size / 4).min([30, 20][density_index]);
    }
    let value = (upper - lower - size) as f64 / delta as f64 + 1.0;
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 || (fraction == 0.5 && (floor as i32) % 2 != 0) {
        floor as i32 + 1
    } else {
        floor as i32
    }
}

/// Matches `parallelBoundarySize` (`IMOD/pysrc/imodpy.py:1629`).
pub fn parallel_boundary_size(default_value: i32) -> i32 {
    std::env::var("PARALLEL_BOUNDARY_SIZE")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| *value > default_value)
        .unwrap_or(default_value)
}

/// Matches `elapsedTimeComponents` (`IMOD/pysrc/imodpy.py:1642`), with Unix seconds as input.
pub fn elapsed_time_components(start_time: f64) -> (i32, i32, i32) {
    let used = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        - start_time;
    let minutes = (used / 60.0) as i32;
    let seconds_float = used - 60.0 * minutes as f64;
    let seconds = seconds_float as i32;
    let fraction = ((seconds_float - seconds as f64) * 10.0).round() as i32;
    (minutes, seconds, fraction)
}

/// Matches `writeFinishAndMessage` (`IMOD/pysrc/imodpy.py:1775`).
pub fn write_finish_and_message(
    assemble_lines: Option<&mut Vec<String>>,
    output_root: &str,
    command_number: i32,
    subm_ok: bool,
    command_extension: &str,
) {
    let mut command_number = command_number;
    if let Some(lines) = assemble_lines {
        lines.push(format!("$b3dremove -g {output_root}-[0-9][0-9][0-9]*{command_extension}* {output_root}-[0-9][0-9][0-9]*.log* {output_root}-finish*{command_extension}*"));
        let _ = write_text_file(
            &format!("{output_root}-finish{command_extension}"),
            lines,
            false,
        );
        command_number += 1;
    }
    prnstr(
        &format!("{command_number} command files created with root name {output_root} and"),
        "\n",
        false,
    );
    prnstr(
        " ready to run with processchunks or parallel processing interface in Etomo",
        "\n",
        false,
    );
    if subm_ok {
        prnstr("Or with:", "\n", false);
        prnstr(
            &format!("  subm {output_root}*{command_extension}"),
            "\n",
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{OptionValue, option_value, parse_list};

    #[test]
    fn parses_forward_backward_and_negative_ranges() {
        assert_eq!(parse_list("1-3,7,5-3"), Some(vec![1, 2, 3, 7, 5, 4, 3]));
        assert_eq!(parse_list("-3--1"), Some(vec![-3, -2, -1]));
        assert_eq!(parse_list("/1-3"), None);
    }

    #[test]
    fn obtains_last_option_value_and_numeric_lists() {
        let lines = vec![
            "Size 1,2,3".to_owned(),
            "Size 4,5,6 # overridden".to_owned(),
        ];
        assert_eq!(
            option_value(&lines, "Size", 1, false, 0, None, None),
            Some(OptionValue::Integers(vec![4, 5, 6]))
        );
    }
}
