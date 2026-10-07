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
/// `pyVersion` (`imodpy.py:113-116`) for the Python 3 interpreter the
/// scripts run under (3.12 in the reference setup: `100 * 3 + 10 * 12` does
/// not apply past 3.9, so `10000 * 3 + 100 * 12`).
const PY_VERSION: i32 = 31200;
static MOC_RENAMES_OK: AtomicBool = AtomicBool::new(true);
/// `mocRecursiveCopyOK`, `mocUseXcopy`, `mocXcopyExists` (`imodpy.py:138-141`).
static MOC_RECURSIVE_COPY_OK: AtomicBool = AtomicBool::new(true);
static MOC_USE_XCOPY: AtomicBool = AtomicBool::new(false);
static MOC_XCOPY_EXISTS: AtomicBool = AtomicBool::new(false);
/// Rust-only: the unit number [`header_in_process`] opens on; `header` itself
/// always uses unit 1 (`header.f90:132`).
static HEADER_UNIT: AtomicI32 = AtomicI32::new(1);

/// Matches Python class `ImodpyError` (`IMOD/pysrc/imodpy.py:159`).
#[derive(Clone, Debug)]
pub struct ImodpyError {
    pub arguments: Vec<String>,
}

/// `STRING_VALUE`, `INT_VALUE`, `FLOAT_VALUE`, `BOOL_VALUE`: the `optionValue`
/// value types (module constants of `IMOD/pysrc/imodpy.py`).
pub const STRING_VALUE: i32 = 0;
pub const INT_VALUE: i32 = 1;
pub const FLOAT_VALUE: i32 = 2;
pub const BOOL_VALUE: i32 = 3;

/// Return variants of `optionValue` (`IMOD/pysrc/imodpy.py:1153`).
#[derive(Clone, Debug, PartialEq)]
pub enum OptionValue {
    String(String),
    Integers(Vec<i32>),
    Floats(Vec<f64>),
    Boolean(bool),
}

/// Return shapes of `getmrc` (`IMOD/pysrc/imodpy.py:456`).  The float
/// members are Python floats -- doubles parsed from `header`'s text.
#[derive(Clone, Debug, PartialEq)]
pub enum MrcInfo {
    Basic(i32, i32, i32, i32, f64, f64, f64),
    All(
        i32,
        i32,
        i32,
        i32,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    ),
    /// `[axis angle, binning, spot, camera, bidir angle]`; spot and camera
    /// are Python ints, held here as their exact double values.
    AngleLines([Option<f64>; 5]),
}

impl Display for ImodpyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.arguments.join("\n"))
    }
}

impl std::error::Error for ImodpyError {}

/// Rust-only: records a failed command run the way `runcmd` does on a
/// non-zero status (`errStrings`, `errStatus`) and returns the error to
/// raise, for callers that run a command without `runcmd`
/// (`comrun::run_com_as_command`).
pub fn set_run_error(errors: Vec<String>, status: i32) -> ImodpyError {
    *ERR_STATUS.lock().expect("imodpy status mutex") = status;
    *ERR_STRINGS.lock().expect("imodpy errors mutex") = errors.clone();
    ImodpyError { arguments: errors }
}

/// Matches `getErrStrings` (`IMOD/pysrc/imodpy.py:167`).
pub fn get_err_strings() -> Vec<String> {
    ERR_STRINGS.lock().expect("imodpy errors mutex").clone()
}

/// Matches `runcmd` (`IMOD/pysrc/imodpy.py:176`) under Python 3, where
/// `useSubprocess` is always true.
///
/// Shapes of the arguments: `outfile` is `None` (collect and return the
/// output), `"stdout"`, or the name of a file standing for the source's open
/// file object (opened here with `'w'`); `in_stderr` is `None`, `"stdout"`,
/// `"pipe"`, or likewise a file name.  The returned lines have their endings
/// removed; they are split at Python's `str.splitlines` boundaries.
///
/// The source's retry-on-`Broken pipe` loop is not translated: on Python 3
/// `Popen.communicate` swallows `BrokenPipeError` itself, so the exception
/// it retries on is never raised.
pub fn run_cmd(
    command: &str,
    input: Option<&[String]>,
    outfile: Option<&str>,
    in_stderr: Option<&str>,
    ignore_status: &[i32],
) -> Result<Option<Vec<String>>, ImodpyError> {
    let command = avoid_local_com_file(command);
    let command = command.as_str();

    // Set up flags for whether to collect output or send to stderr
    let verbose = std::env::var("RUNCMD_VERBOSE").ok().as_deref() == Some("1");
    if verbose {
        prnstr("+++++++++++++++++++++++++", "\n", false);
        prnstr("   runcmd running command:", "\n", false);
        prnstr(command, "\n", false);
        if let Some(lines) = input.filter(|lines| !lines.is_empty()) {
            prnstr("   With input:", "\n", false);
            for l in lines {
                prnstr(l, "\n", false);
            }
        }
        let _ = std::io::stdout().flush();
    }

    // Owner decision, 2026-09-24: a command of ours runs in this process (see
    // `run_cmd_in_process`).  Only when standard error is left alone: the
    // in-process runner redirects standard input and output, not error.
    if in_stderr.is_none()
        && let Some((own, words)) = own_command_words(command, true)
    {
        return run_own_command(
            command,
            own,
            words,
            input,
            outfile,
            ignore_status,
            false,
            verbose,
        );
    }

    *ERR_STATUS.lock().expect("imodpy status mutex") = 0;
    let collect = outfile.is_none();
    let to_stdout = outfile == Some("stdout");
    let raise = |message: String| -> ImodpyError {
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
        ImodpyError {
            arguments: vec![message],
        }
    };
    // `str(sys.exc_info()[1])` of an OSError
    let exc_text = |error: &std::io::Error, name: &str| -> String {
        let text = error.to_string();
        match error.raw_os_error() {
            Some(errno) => format!(
                "[Errno {errno}] {}: '{name}'",
                text.strip_suffix(&format!(" (os error {errno})"))
                    .unwrap_or(&text)
            ),
            None => text,
        }
    };

    // The subprocess interface: input must be all one string
    let mut joined: Option<Vec<u8>> = None;
    if let Some(lines) = input.filter(|lines| !lines.is_empty()) {
        let mut text = String::new();
        for l in lines {
            text.push_str(l);
            text.push('\n');
        }
        joined = Some(text.into_bytes());
    }

    // Rust-only (owner goal 2026-10-05: nothing depends on native IMOD at
    // run time): a command of ours that does not run in process -- a window
    // program such as `genhstplt`, or a Python-script translation -- runs as
    // a child of our own binary when this process is the `imod` launcher,
    // not as whatever `PATH` finds.  Everything else about `runcmd` is
    // unchanged.
    let own_child = own_command_words(command, false).and_then(|(_, words)| {
        let exe = std::env::current_exe().ok()?;
        if exe.file_name()?.to_string_lossy() != format!("imod{}", std::env::consts::EXE_SUFFIX) {
            return None;
        }
        Some((exe, words))
    });
    // `Popen(cmd, shell=True)` runs `/bin/sh -c cmd` with `argv[0]` "sh"
    let mut process = match &own_child {
        Some((exe, words)) => {
            let mut process = Command::new(exe);
            crate::imod::libcfshr::b3dutil::command_arg0(
                &mut process,
                exe.with_file_name(&words[0]),
            );
            process.args(&words[1..]);
            process
        }
        None => crate::imod::libcfshr::b3dutil::shell_command(command),
    };
    // `Popen(..., stdin=PIPE)` in all three of the source's forms, so the
    // command reads the given input and then end-of-file, never the caller's
    // standard input
    process.stdin(Stdio::piped());
    let mut merged: Option<std::io::PipeReader> = None;
    let setup: Result<(), String> = (|| {
        // Run it three different ways depending on where output goes
        if to_stdout {
            // `Popen(cmd, shell=True, stdin=PIPE)`: inStderr is ignored
            process.stdout(Stdio::inherit());
            process.stderr(Stdio::inherit());
            return Ok(());
        }
        // The standard output target: a pipe to collect, or the file
        let out_file = match outfile {
            None => None,
            Some(filename) => Some(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(filename)
                    .map_err(|error| exc_text(&error, filename))?,
            ),
        };
        match in_stderr {
            // `stderr=STDOUT`: standard error joins standard output,
            // whether that is collected or bound for `outfile`
            Some("stdout") => match out_file {
                Some(file) => {
                    let second = file.try_clone().map_err(|error| error.to_string())?;
                    process.stdout(Stdio::from(file));
                    process.stderr(Stdio::from(second));
                }
                None => {
                    let (reader, writer) = std::io::pipe().map_err(|error| error.to_string())?;
                    let second = writer.try_clone().map_err(|error| error.to_string())?;
                    process.stdout(writer);
                    process.stderr(second);
                    merged = Some(reader);
                }
            },
            other => {
                match out_file {
                    Some(file) => {
                        process.stdout(Stdio::from(file));
                    }
                    None => {
                        process.stdout(Stdio::piped());
                    }
                }
                match other {
                    None => {
                        process.stderr(Stdio::inherit());
                    }
                    // `PIPE`: read by `communicate` and dropped
                    Some("pipe") => {
                        process.stderr(Stdio::piped());
                    }
                    Some(filename) => {
                        let file = OpenOptions::new()
                            .create(true)
                            .write(true)
                            .truncate(true)
                            .open(filename)
                            .map_err(|error| exc_text(&error, filename))?;
                        process.stderr(Stdio::from(file));
                    }
                }
            }
        }
        Ok(())
    })();
    let spawned = setup.and_then(|()| process.spawn().map_err(|error| exc_text(&error, "sh")));
    let mut child = match spawned {
        Ok(child) => child,
        Err(exception) => {
            return Err(raise(format!("command {command}: {exception}\n")));
        }
    };
    // The parent's copies of a merged pipe's write end live in `process`
    drop(process);

    // `p.communicate(input)`: the input is written while the output is read,
    // and a command that stops reading early (a broken pipe) is not an error
    let writer = child.stdin.take().map(|mut stdin| {
        std::thread::spawn(move || {
            if let Some(bytes) = joined {
                let _ = stdin.write_all(&bytes);
            }
        })
    });
    let waited = match merged {
        Some(mut reader) => {
            let mut bytes = Vec::new();
            let _ = std::io::Read::read_to_end(&mut reader, &mut bytes);
            child.wait().map(|status| std::process::Output {
                status,
                stdout: bytes,
                stderr: Vec::new(),
            })
        }
        None => child.wait_with_output(),
    };
    if let Some(writer) = writer {
        let _ = writer.join();
    }
    let output = match waited {
        Ok(output) => output,
        Err(error) => {
            return Err(raise(format!("command {command}: {error}\n")));
        }
    };
    let mut output_lines: Option<Vec<String>> = None;
    if collect {
        let kout = String::from_utf8_lossy(&output.stdout);
        if !kout.is_empty() {
            // `kout.splitlines(True)`, with the endings then removed
            let mut lines = Vec::new();
            let mut current = String::new();
            let mut chars = kout.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '\n' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}'
                    | '\u{2028}' | '\u{2029}' => lines.push(std::mem::take(&mut current)),
                    '\r' => {
                        if chars.peek() == Some(&'\n') {
                            chars.next();
                        }
                        lines.push(std::mem::take(&mut current));
                    }
                    _ => current.push(c),
                }
            }
            if !current.is_empty() {
                lines.push(current);
            }
            output_lines = Some(lines);
        }
    }
    // `p.returncode` is minus the signal number for a killed command
    let ec = output.status.code().unwrap_or_else(|| {
        crate::imod::libcfshr::b3dutil::exit_signal(&output.status).map_or(1, |signal| -signal)
    });
    if ec != 0 {
        *ERR_STATUS.lock().expect("imodpy status mutex") = ec;
    }

    if verbose {
        if let Some(lines) = &output_lines {
            prnstr("    Output:", "\n", false);
            for l in lines {
                prnstr(l, "\n", false);
            }
        }
        prnstr("-------------------------", "\n", true);
    }

    if ec != 0 && !ignore_status.contains(&ec) {
        // look thru the output for 'ERROR' line(s) and put them before this.
        // The lines are stored without their endings, as the in-process
        // route stores them; `exit_from_imod_error` supplies the endings.
        let mut errors: Vec<String> = Vec::new();
        if let Some(lines) = &output_lines {
            errors = lines
                .iter()
                .filter(|line| line.contains("ERROR:"))
                .cloned()
                .collect();
        }
        errors.push(format!("{command}: exited with status {ec}"));
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = errors.clone();
        return Err(ImodpyError { arguments: errors });
    }

    if collect {
        // `output` is None when nothing was printed
        return Ok(Some(output_lines.unwrap_or_default()));
    }
    Ok(None)
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
    match own_command_words(command, true) {
        Some((own, words)) => {
            run_own_command(command, own, words, input, outfile, &[], true, false)
        }
        None => {
            let ignore: [i32; 0] = [];
            run_cmd(command, input, outfile, None, &ignore)
        }
    }
}

/// Rust-only: calls one of our own programs **directly** and returns the
/// value it computed, with `runcmd`'s error contract.
///
/// Owner rule, 2026-09-26 (`CLAUDE.md`, "Wherever we control both sides, use
/// a direct function call now"): a script that ran our program and parsed its
/// printed output instead calls the program's compute function (or, for a
/// program not yet split, runs it with its reported values recorded into a
/// result struct) and uses the returned values.  `body` runs through
/// `commands::call_in_process`, so it gets a fresh command environment
/// (Fortran unit table, PIP state, exit via `b3dutil::exit`) exactly as
/// [`run_cmd_in_process`] gave the whole program.  `words` are the program
/// name and any arguments it still reads through PIP, and `input` its
/// standard-input lines (each given a newline); `command` is the command
/// line the script used to run, and names the command in `errStrings`.
///
/// With `capture` set, what the program printed is returned as lines that
/// keep their endings (`splitlines(True)`), for a script that echoes the
/// program's report; otherwise it goes straight to standard output and the
/// list is empty.  The value is `None` when the program ended through `exit`
/// rather than returning it.  A non-zero exit status sets `errStrings` to the
/// captured `ERROR:` lines followed by `"<command>: exited with status <n>"`
/// and the last exit status, as `runcmd` does, and raises [`ImodpyError`].
pub fn call_own_program<R: Send + 'static>(
    command: &str,
    words: &[&str],
    input: Option<&[String]>,
    capture: bool,
    body: impl FnOnce() -> R + Send + 'static,
) -> Result<(Option<R>, Vec<String>), ImodpyError> {
    let joined = input.map(|lines| {
        let mut text = String::new();
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
        text
    });
    *ERR_STATUS.lock().expect("imodpy status mutex") = 0;
    let (status, value, output) = match crate::imod::commands::call_in_process(
        words,
        joined.as_deref().map(str::as_bytes),
        capture,
        body,
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
    let text = String::from_utf8_lossy(&output);
    let lines: Vec<String> = text.split_inclusive('\n').map(str::to_owned).collect();
    if status != 0 {
        *ERR_STATUS.lock().expect("imodpy status mutex") = status;
        let mut errors: Vec<String> = lines
            .iter()
            .filter(|line| line.contains("ERROR:"))
            .map(|line| line.trim_end_matches(['\r', '\n']).to_owned())
            .collect();
        errors.push(format!("{command}: exited with status {status}"));
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = errors.clone();
        return Err(ImodpyError { arguments: errors });
    }
    Ok((value, lines))
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
    require_in_process: bool,
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
    if require_in_process && !own.in_process {
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
    verbose: bool,
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
    // `RUNCMD_VERBOSE` trailer (`imodpy.py:327-332`)
    if verbose {
        if collect && !lines.is_empty() {
            prnstr("    Output:", "\n", false);
            for l in &lines {
                let end = if l.ends_with('\n') { "" } else { "\n" };
                prnstr(l, end, false);
            }
        }
        prnstr("-------------------------", "\n", true);
    }
    if let Some(filename) = outfile.filter(|name| *name != "stdout") {
        // `runcmd` hands a file object to the child as its standard output,
        // so the file gets the output whether or not the command then fails
        // (`gputilttest` reads its log after a failed `tilt`).
        if let Err(error) = fs::write(filename, &output) {
            let message = format!("Writing to file: {filename}  - {error}");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            return Err(ImodpyError {
                arguments: vec![message],
            });
        }
        if status == 0 || ignore_status.contains(&status) {
            return Ok(None);
        }
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
///
/// Only the source's `useSubprocess` branch exists (it is always taken on
/// Python 3).  `errfile == Some("stdout")` is `subprocess.STDOUT`: the
/// child's standard error goes wherever its standard output goes -- into
/// `outfile` when one is given, otherwise to this process's standard output.
/// An error is `action + "  - " + str(exception)`, with the OSError text in
/// Python's `[Errno n] strerror: 'name'` form.
pub fn bkgd_process(
    command_array: &[OsString],
    outfile: Option<&str>,
    errfile: Option<&str>,
    return_on_error: bool,
    append: bool,
) -> Result<(), ImodpyError> {
    // `str(sys.exc_info()[1])` of an OSError
    let exc_text = |error: &std::io::Error, name: &str| -> String {
        let text = error.to_string();
        match error.raw_os_error() {
            Some(errno) => format!(
                "[Errno {errno}] {}: '{name}'",
                text.strip_suffix(&format!(" (os error {errno})"))
                    .unwrap_or(&text)
            ),
            None => text,
        }
    };
    let mut action = String::new();
    let result: Result<(), String> = (|| {
        // If subprocess is allowed, open the files if any
        let mut outf: Option<fs::File> = None;
        let mut errf: Option<Stdio> = None;
        let err_to_stdout = errfile == Some("stdout");
        if let Some(outfile) = outfile.filter(|name| !name.is_empty()) {
            let mut options = OpenOptions::new();
            options.create(true).write(true);
            if append && Path::new(outfile).exists() {
                options.append(true);
            } else {
                options.truncate(true);
            }
            action = format!("Opening {outfile} for output");
            outf = Some(
                options
                    .open(outfile)
                    .map_err(|error| exc_text(&error, outfile))?,
            );
        }
        if let Some(errfile) = errfile.filter(|name| !name.is_empty() && *name != "stdout") {
            action = format!("Opening {errfile} for error output");
            if errfile == "devnull" {
                errf = Some(Stdio::null());
            } else {
                let mut options = OpenOptions::new();
                options.create(true).write(true);
                if append && Path::new(errfile).exists() {
                    options.append(true);
                } else {
                    options.truncate(true);
                }
                errf = Some(Stdio::from(
                    options
                        .open(errfile)
                        .map_err(|error| exc_text(&error, errfile))?,
                ));
            }
        }

        let program = command_array
            .first()
            .map(|name| name.to_string_lossy().into_owned());
        action = format!(
            "Starting background process {}",
            program.clone().unwrap_or_default()
        );
        let Some(program) = program else {
            return Err("list index out of range".to_owned());
        };
        let mut process = Command::new(&command_array[0]);
        process.args(&command_array[1..]);
        // `stderr=STDOUT`: the child's standard error duplicates its
        // standard output descriptor, whatever that is
        if err_to_stdout {
            let duplicate: std::io::Result<std::fs::File> = match &outf {
                Some(file) => file.try_clone(),
                None => {
                    #[cfg(unix)]
                    {
                        use std::os::fd::AsFd;
                        std::io::stdout()
                            .as_fd()
                            .try_clone_to_owned()
                            .map(std::fs::File::from)
                    }
                    #[cfg(windows)]
                    {
                        use std::os::windows::io::AsHandle;
                        std::io::stdout()
                            .as_handle()
                            .try_clone_to_owned()
                            .map(std::fs::File::from)
                    }
                }
            };
            errf = Some(Stdio::from(
                duplicate.map_err(|error| exc_text(&error, &program))?,
            ));
        }
        if let Some(file) = outf {
            process.stdout(Stdio::from(file));
        }
        if let Some(stdio) = errf {
            process.stderr(stdio);
        }
        // Use detached flag on Windows, although it may not be needed.  In
        // fact, unless stderr is going to a file it keeps it from running
        // there (`imodpy.py:415-420`)
        #[cfg(windows)]
        {
            let err_to_file = (outfile.is_some_and(|name| !name.is_empty())
                && errfile == Some("stdout"))
                || errfile
                    .is_some_and(|name| !name.is_empty() && name != "stdout" && name != "devnull");
            if err_to_file {
                const DETACHED_PROCESS: u32 = 0x0000_0008;
                std::os::windows::process::CommandExt::creation_flags(
                    &mut process,
                    DETACHED_PROCESS,
                );
            }
        }
        process
            .spawn()
            .map(|_| ())
            .map_err(|error| exc_text(&error, &program))
    })();
    match result {
        Ok(()) => Ok(()),
        Err(exception) => {
            let err_string = format!("{action}  - {exception}");
            if return_on_error {
                return Err(ImodpyError {
                    arguments: vec![err_string],
                });
            }
            crate::imod::pysrc::pip::exit_error(&err_string)
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
/// spacing` line, the `Space group,# extra bytes` line, the titles (first 79 bytes, FORMAT 1020) and the `idtype`
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
        iiu_ret_origin, iiu_ret_size, iiu_ret_space_group,
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
            // FORMAT 1020's `Space group,# extra bytes,idtype,lens .` line
            // (`irdhdr.f90:119`, 4I9), which `copyheader` reads.
            let (idtype, lensnum, _, _, _, _) = iiu_ret_data_type(im_unit);
            out.push_str(&format!(
                " Space group,# extra bytes,idtype,lens .{:>9}{:>9}{:>9}{:>9}\n\n",
                iiu_ret_space_group(im_unit),
                iiu_ret_num_extended(im_unit),
                idtype,
                lensnum
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
///
/// Python's `int()`/`float()` of a `header` token that is not a number raise
/// a ValueError the source does not catch outside the angle-line loop; that
/// is reported here as an `ImodpyError` naming the token.
pub fn get_mrc(file: &str, do_all: bool, angle_line_values: bool) -> Result<MrcInfo, ImodpyError> {
    let raise = |message: String| -> ImodpyError {
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
        ImodpyError {
            arguments: vec![message],
        }
    };
    let py_int = |text: &str| -> Result<i32, ImodpyError> {
        crate::imod::pysrc::imodpy::py_int(text)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| raise(format!("invalid literal for int() with base 10: '{text}'")))
    };
    let py_float = |text: &str| -> Result<f64, ImodpyError> {
        crate::imod::pysrc::imodpy::py_float(text)
            .ok_or_else(|| raise(format!("could not convert string to float: '{text}'")))
    };
    if angle_line_values {
        let mut retval: [Option<f64>; 5] = [None; 5];
        let keys = ["angle", "binning", "spot", "camera", "bidir"];
        let types = [1, 1, 0, 0, 1];
        // `imodpy.py:477`: `runcmd("header -StandardInput", input)` with
        // `input = ["InputFile " + file]`.  Owner decision (2026-09-24): our
        // own `header` runs in process instead of through `sh -c`/`PATH`.
        let hdrout = header_in_process("header -StandardInput", file, true, None)?;
        for line in &hdrout {
            if line.to_lowercase().contains("axis") && line.contains("angle") {
                // Here is  good way to split on multiple characters; filter removes empty
                // strings from the separators
                // `runcmd` returns `kout.splitlines(True)`, so the source's
                // `line` still ends in its newline, which becomes part of the
                // last token (or a token of its own after trailing blanks);
                // `header_in_process` returns the lines without it.
                let line_with_end = format!("{line}\n");
                // Defined behaviour (BUGS.md, "fixed in translation"): the
                // source calls `len()` on `multiCharSplit`'s result, a
                // `filter` object on Python 3, and dies with a TypeError that
                // no caller catches.  This is the loop the code clearly
                // intends -- `list(multiCharSplit(line, ' ,='))` -- which is
                // also what it did under Python 2.
                let tokens = multi_char_split(&line_with_end, " ,=");
                for ind in 0..tokens.len().saturating_sub(1) {
                    for key_ind in 0..keys.len() {
                        if tokens[ind] == keys[key_ind] {
                            // Python's `float()`/`int()` strip surrounding
                            // whitespace, including the line's newline.
                            let converted = if types[key_ind] != 0 {
                                crate::imod::pysrc::imodpy::py_float(&tokens[ind + 1])
                            } else {
                                crate::imod::pysrc::imodpy::py_int(&tokens[ind + 1])
                                    .and_then(|value| i32::try_from(value).ok())
                                    .map(f64::from)
                            };
                            match converted {
                                Some(value) => retval[key_ind] = Some(value),
                                None => {
                                    return Err(raise(format!(
                                        "header {file}: Error converting value to integer or float in title {line}\n"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }
        return Ok(MrcInfo::AngleLines(retval));
    }
    // `imodpy.py:500-505`: `runcmd(command, input)`, now in process (owner
    // decision 2026-09-24); the lines carry the `3i8`/`i4`/`3g15.5`/`g13.5`
    // text `header` prints, so the values parsed below are rounded as before.
    let (mut hdrout, needed) = if do_all {
        (
            header_in_process(
                "header -si -mo -pi -ori -min -max -mean -StandardInput",
                file,
                true,
                Some(true),
            )?,
            7,
        )
    } else {
        (
            header_in_process("header -si -mo -pi -StandardInput", file, true, Some(false))?,
            3,
        )
    };

    // Eat any lines with PIP fallback output
    while hdrout.len() >= needed {
        if hdrout[0].trim().is_empty()
            || hdrout[0].contains('a')
            || hdrout[0].contains('e')
            || hdrout[0].contains('i')
            || hdrout[0].contains('o')
        {
            hdrout.remove(0);
        } else {
            break;
        }
    }

    if hdrout.len() < needed {
        return Err(raise(format!("header {file}: too few lines of output\n")));
    }

    let nxyz: Vec<&str> = hdrout[0].split_whitespace().collect();
    let pxyz: Vec<&str> = hdrout[2].split_whitespace().collect();
    let orixyz: Vec<&str> = if do_all {
        hdrout[3].split_whitespace().collect()
    } else {
        Vec::new()
    };
    if nxyz.len() < 3 || pxyz.len() < 3 || (do_all && orixyz.len() < 3) {
        let bad_line = if nxyz.len() < 3 {
            format!("-si option: {}", hdrout[0].trim())
        } else if pxyz.len() < 3 {
            format!("-pi option: {}", hdrout[2].trim())
        } else {
            format!("-ori option: {}", hdrout[3].trim())
        };
        return Err(raise(format!(
            "header {file}: too few numbers on line for {bad_line}\n"
        )));
    }
    let ix = py_int(nxyz[0])?;
    let iy = py_int(nxyz[1])?;
    let iz = py_int(nxyz[2])?;
    let mode = py_int(&hdrout[1])?;
    let px = py_float(pxyz[0])?;
    let py = py_float(pxyz[1])?;
    let pz = py_float(pxyz[2])?;

    if !do_all {
        return Ok(MrcInfo::Basic(ix, iy, iz, mode, px, py, pz));
    }

    let orix = py_float(orixyz[0])?;
    let oriy = py_float(orixyz[1])?;
    let oriz = py_float(orixyz[2])?;
    let minv = py_float(&hdrout[4])?;
    let maxv = py_float(&hdrout[5])?;
    let meanv = py_float(&hdrout[6])?;
    Ok(MrcInfo::All(
        ix, iy, iz, mode, px, py, pz, orix, oriy, oriz, minv, maxv, meanv,
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
pub fn get_mrc_pixel(file: &str) -> Result<f64, ImodpyError> {
    let raise = |message: String| -> ImodpyError {
        *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
        ImodpyError {
            arguments: vec![message],
        }
    };
    // `imodpy.py:565`: `runcmd("header -StandardInput", input)`, now in
    // process; the `Pixel spacing` (g11.4) and `size in nanometers` (g11.4)
    // lines are the text `header` prints, so the pixel keeps its rounding.
    let hdrout = header_in_process("header -StandardInput", file, true, None)?;
    let mut pixel = -1.0f64;
    for line in &hdrout {
        let conversion_error = || {
            raise(format!(
                "header {file}: error converting pixel size to float"
            ))
        };
        if line.contains("Pixel spacing") {
            let dot_ind = line.find(".. ").map_or(1, |index| index as i64 + 2);
            if dot_ind < 3 {
                return Err(raise(format!("header {file}: cannot find pixel sizes")));
            }
            let lsplit: Vec<&str> = line[dot_ind as usize..].split_whitespace().collect();
            if lsplit.len() < 3 {
                return Err(raise(format!(
                    "header {file}: pixel sizes not interpretable"
                )));
            }
            pixel = py_float(lsplit[0]).ok_or_else(conversion_error)?;
        }

        if line.contains("size in nanometers =") {
            let ind = line.find('=').unwrap() + 1;
            let lsplit: Vec<&str> = line[ind..].split_whitespace().collect();
            if !lsplit.is_empty() {
                pixel = 10. * py_float(lsplit[0]).ok_or_else(conversion_error)?;
            }
        }
    }

    if pixel < 0. {
        return Err(raise(format!("header {file}: cannot find pixel size")));
    }

    Ok(pixel)
}

/// Matches `getMontageSize` (`IMOD/pysrc/imodpy.py:596`); `montagesize` remains external.
pub fn get_montage_size(
    stack: &str,
    pl_name: Option<&str>,
) -> Result<(i32, i32, i32), ImodpyError> {
    let mut command = format!("montagesize \"{stack}\"");
    if let Some(pl_name) = pl_name.filter(|name| !name.is_empty() && Path::new(name).exists()) {
        command.push_str(&format!(" \"{pl_name}\""));
    }
    let size_lines = run_cmd(&command, None, None, None, &[])?;
    let mut problem = "No output returned";
    let parsed: Option<(i32, i32, i32)> = (|| {
        let lines = size_lines.as_ref()?;
        let line = lines.last()?;
        let chars: Vec<char> = line.chars().collect();
        // `line.find('NZ:')`, or -5 so that `line[-2:]` is taken
        let start = match line.find("NZ:") {
            Some(index) => line[..index].chars().count() as i64 + 3,
            None => {
                problem = "Line with NZ: not found in output";
                chars.len() as i64 - 2
            }
        };
        let rest: String = chars[start.clamp(0, chars.len() as i64) as usize..]
            .iter()
            .collect();
        let lsplit: Vec<&str> = rest.split_whitespace().collect();
        problem = "Uninterpretable output on line with NZ:";
        let raw_xsize = i32::try_from(py_int(lsplit.first()?)?).ok()?;
        let raw_ysize = i32::try_from(py_int(lsplit.get(1)?)?).ok()?;
        let zsize = i32::try_from(py_int(lsplit.get(2)?)?).ok()?;
        Some((raw_xsize, raw_ysize, zsize))
    })();
    match parsed {
        Some(sizes) => Ok(sizes),
        None => {
            // Fixed in translation (BUGS.md): native sets `errStrings =
            // command + ': ' + problem` (`imodpy.py:618`), a string, so a caller
            // iterating it prints one character per line; here it is a
            // one-line list, as `runcmd` leaves it.
            let message = format!("{command}: {problem}");
            *ERR_STRINGS.lock().expect("imodpy errors mutex") = vec![message.clone()];
            Err(ImodpyError {
                arguments: vec![message],
            })
        }
    }
}

/// Matches `runGoodframe` (`IMOD/pysrc/imodpy.py:624`); `goodframe` remains external.
pub fn run_goodframe(nx: i32, ny: i32) -> (i32, i32) {
    let goodout = match run_cmd(&format!("goodframe {nx} {ny}"), None, None, None, &[]) {
        Ok(lines) => lines,
        Err(_) => return (-1, -1),
    };
    let parsed: Option<(i32, i32)> = (|| {
        let goodout = goodout?;
        let gsplit: Vec<&str> = goodout.last()?.split_whitespace().collect();
        let gfnx = i32::try_from(py_int(gsplit.first()?)?).ok()?;
        let gfny = i32::try_from(py_int(gsplit.get(1)?)?).ok()?;
        Some((gfnx, gfny))
    })();
    parsed.unwrap_or((-2, -2))
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
    let mut descrip = description.unwrap_or("");
    if descrip.is_empty() {
        descrip = " ";
    }
    match fs::read_to_string(filename) {
        // Python opens the file with universal newlines, so `\r\n` and a lone
        // `\r` both end a line
        Ok(text) => Ok(text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .lines()
            .take(maximum_lines.unwrap_or(usize::MAX))
            .map(|line| line.trim_end_matches([' ', '\t', '\r', '\n']).to_owned())
            .collect()),
        Err(error) => {
            // `str(sys.exc_info()[1])` of the OSError: `[Errno n] strerror: 'name'`
            let text = error.to_string();
            let exc_info = match error.raw_os_error() {
                Some(errno) => format!(
                    "[Errno {errno}] {}: '{filename}'",
                    text.strip_suffix(&format!(" (os error {errno})"))
                        .unwrap_or(&text)
                ),
                None => text.clone(),
            };
            let message = format!("Opening {descrip} {filename}: {exc_info}");
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
            // `str(sys.exc_info()[1])` of the OSError: `[Errno n] strerror: 'name'`
            let text = error.to_string();
            let exc_info = match error.raw_os_error() {
                Some(errno) => format!(
                    "[Errno {errno}] {}: '{filename}'",
                    text.strip_suffix(&format!(" (os error {errno})"))
                        .unwrap_or(&text)
                ),
                None => text.clone(),
            };
            let message = format!("Opening file: {filename}  - {exc_info}");
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
    match py_int(value_string) {
        Some(value) => value as i32,
        None => crate::imod::pysrc::pip::exit_error(&format!(
            "Converting {description} ({value_string}) to integer"
        )),
    }
}

/// Matches `optionValue` (`IMOD/pysrc/imodpy.py:1153`).
///
/// The source's three regular expressions are used as written: `optre`
/// anchors the option at the start of the line, `comre` rejects a line with
/// `# option` **anywhere** in it, and `subre` takes the value after the
/// *last* occurrence of the option and the rest of that word.  A line that
/// `subre` does not match is left whole by `re.sub`, so its whole stripped
/// text becomes the value.  Floats are Python floats (doubles).  A
/// `numVal == 1` scalar is returned as a one-element list.
pub fn option_value(
    lines: &[String],
    option: &str,
    value_type: i32,
    ignore_case: bool,
    number_values: usize,
    other_separator: Option<char>,
    empty_return: Option<&str>,
) -> Option<OptionValue> {
    let mut sep = r"\s".to_owned();
    if let Some(other) = other_separator {
        sep = format!(r"\s*{}", other);
    }
    let build = |pattern: String| {
        regex::RegexBuilder::new(&pattern)
            .case_insensitive(ignore_case)
            .build()
            .expect("optionValue regular expression")
    };
    let optre = build(format!(r"^\s*{option}"));
    let subre = build(format!(r".*{option}[^\s]*{sep}([^#]*).*"));
    let comre = build(format!(r"\s*#\s*{option}"));
    let mut retval = None;
    for line in lines {
        if optre.is_match(line) && !comre.is_match(line) {
            let valstr = subre.replace_all(line, "${1}").trim().to_owned();
            if value_type > 2 {
                // A boolean can be any of these values, but it can also be a line with no
                // separator, which requires a separate re test
                let bval = valstr.to_lowercase();
                if bval == "0" || bval == "f" || bval == "off" || bval == "false" {
                    retval = Some(OptionValue::Boolean(false));
                } else if bval == "1"
                    || bval == "t"
                    || bval == "on"
                    || bval == "true"
                    || bval.is_empty()
                    || valstr == option
                    || (other_separator.is_none()
                        && build(format!(r".*{option}[^\s]*([^#]*).*"))
                            .replace_all(line, "${1}")
                            .trim()
                            .is_empty())
                {
                    retval = Some(OptionValue::Boolean(true));
                } else {
                    prnstr(
                        &format!(
                            "WARNING: optionValue - Boolean entry found with improper value ({bval}) in: {line}"
                        ),
                        "\n",
                        false,
                    );
                }
            } else if valstr.is_empty() {
                if let Some(entry) = empty_return.filter(|entry| !entry.is_empty()) {
                    retval = Some(OptionValue::String(entry.to_owned()));
                } else {
                    prnstr(
                        &format!("WARNING: optionValue - No value for option in: {line}"),
                        "\n",
                        false,
                    );
                }
            } else if value_type <= 0 {
                retval = Some(OptionValue::String(valstr));
            } else {
                let replaced = valstr.replace(',', " ");
                let splits = replaced.split_whitespace().collect::<Vec<_>>();
                let mut num_conv = splits.len();
                if number_values != 0 {
                    if num_conv < number_values {
                        return None;
                    }
                    num_conv = number_values;
                }
                if value_type == 1 {
                    let mut values = Vec::new();
                    for val in &splits[..num_conv] {
                        match py_int(val).and_then(|value| i32::try_from(value).ok()) {
                            Some(value) => values.push(value),
                            None => {
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
                    retval = Some(OptionValue::Integers(values));
                } else {
                    let mut values = Vec::new();
                    for val in &splits[..num_conv] {
                        match py_float(val) {
                            Some(value) => values.push(value),
                            None => {
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
                    retval = Some(OptionValue::Floats(values));
                }
            }
        }
    }
    retval
}

/// Matches `completeAndCheckComFile` (`IMOD/pysrc/imodpy.py:1251`).
///
/// Returns `(comfile, rootname)` in the source's order.
pub fn complete_and_check_com_file(comfile: &str) -> (String, String) {
    if comfile.is_empty() {
        crate::imod::pysrc::pip::exit_error("A command file must be entered");
    }
    let mut comfile = comfile.to_owned();

    let is_com = comfile.ends_with(".com");
    let is_pcm = comfile.ends_with(".pcm");
    let rootname = if is_com || is_pcm {
        py_slice_end(&comfile, 4)
    } else if comfile.ends_with('.') {
        comfile.trim_end_matches('.').to_owned()
    } else {
        comfile.clone()
    };

    // Look for the one we know it is, or both if it had neither extension
    let mut pcm_exists = false;
    let mut com_exists = false;
    if !is_com {
        pcm_exists = Path::new(&format!("{rootname}.pcm")).exists();
    }
    if !is_pcm {
        com_exists = Path::new(&format!("{rootname}.com")).exists();
    }

    // Handle lack of specified file with extension known
    if (is_com && !com_exists) || (is_pcm && !pcm_exists) {
        crate::imod::pysrc::pip::exit_error(&format!("Command file {comfile} does not exist"));
    }

    // Handle possibilities of looking for either extension: error or assign name
    if !(is_com || is_pcm) {
        if com_exists && pcm_exists {
            crate::imod::pysrc::pip::exit_error(&format!(
                "The full command file name must be entered because both {rootname}.com and {rootname}.pcm exist"
            ));
        }
        if com_exists {
            comfile = format!("{rootname}.com");
        } else if pcm_exists {
            comfile = format!("{rootname}.pcm");
        } else {
            crate::imod::pysrc::pip::exit_error(&format!(
                "The full command file name must be entered because neither {rootname}.com nor {rootname}.pcm exist"
            ));
        }
    }

    (comfile, rootname)
}

/// Matches `cleanupFiles` (`IMOD/pysrc/imodpy.py:1317`).
pub fn cleanup_files(files: &[String]) {
    // Even wih a wait of 0.01 it only took two trials, so try 0.1 for this
    let retry_wait = std::time::Duration::from_millis(100);
    let max_trials = 10;
    let mut num_to_do = files.len();
    let mut still_to_do = vec![1; files.len()];
    let mut trial = 0;
    while trial < max_trials && num_to_do != 0 {
        trial += 1;
        for ind in 0..files.len() {
            if still_to_do[ind] != 0 {
                let removed = match fs::remove_file(&files[ind]) {
                    Ok(()) => 1,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
                    Err(_) => -1,
                };
                if removed >= 0 {
                    still_to_do[ind] = 0;
                    num_to_do -= 1;
                }
            }
        }

        if num_to_do != 0 {
            std::thread::sleep(retry_wait);
        }
    }
}

/// Matches `cleanChunkFiles` (`IMOD/pysrc/imodpy.py:1296`).
pub fn clean_chunk_files(rootname: &str, log_only: bool) {
    let mut rmlist = glob_glob(&format!("{rootname}-[0-9][0-9][0-9]*.log"));
    if Path::new(&format!("{rootname}-start.log")).exists() {
        rmlist.push(format!("{rootname}-start.log"));
    }
    if Path::new(&format!("{rootname}-finish.log")).exists() {
        rmlist.push(format!("{rootname}-finish.log"));
    }
    if !log_only {
        for ext in [".com", ".pcm"] {
            rmlist.extend(glob_glob(&format!("{rootname}-[0-9][0-9][0-9]*{ext}")));
            if Path::new(&format!("{rootname}-start{ext}")).exists() {
                rmlist.push(format!("{rootname}-start{ext}"));
            }
            if Path::new(&format!("{rootname}-finish{ext}")).exists() {
                rmlist.push(format!("{rootname}-finish{ext}"));
            }
        }
    }
    for filename in &rmlist {
        if fs::remove_file(filename).is_err() {
            break;
        }
    }
}

/// Matches `balancedGroupLimits` (`IMOD/pysrc/imodpy.py:1792`).
pub fn balanced_group_limits(total: i32, groups: i32, group_index: i32) -> (i32, i32) {
    let base = py_int_floordiv(total as i64, groups as i64) as i32;
    let remainder = py_int_mod(total as i64, groups as i64) as i32;
    let start = group_index * base + group_index.min(remainder);
    let end = (group_index + 1) * base + (group_index + 1).min(remainder) - 1;
    (start, end)
}

/// Matches `fmtstr` (`IMOD/pysrc/imodpy.py:1801`), which on Python 3.1 or
/// later is `string.format(*args)` (`imodpy.py:1818-1819`); the pre-2.7
/// rewriting below that line is never reached.
///
/// The arguments arrive already converted to text (the caller applies
/// Python's `str()` or format), so this is `str.format` over `str`
/// arguments: `{{` and `}}` are literal braces, `{}` takes the next argument
/// and `{N}` argument `N`, and a `:spec` applies a string format
/// specification -- `[[fill]align][width][.precision]`, left-aligned by
/// default.  A malformed field, which Python rejects with a ValueError,
/// panics.
pub fn fmtstr(string_in: &str, args: &[String]) -> String {
    let chars: Vec<char> = string_in.chars().collect();
    let mut result = String::new();
    let mut auto_index = 0usize;
    // Python refuses to mix `{}` and `{N}` in one string
    let mut numbering: Option<bool> = None;
    let mut ind = 0usize;
    while ind < chars.len() {
        let c = chars[ind];
        if c == '}' {
            assert!(
                chars.get(ind + 1) == Some(&'}'),
                "Single '}}' encountered in format string: {string_in}"
            );
            result.push('}');
            ind += 2;
            continue;
        }
        if c != '{' {
            result.push(c);
            ind += 1;
            continue;
        }
        if chars.get(ind + 1) == Some(&'{') {
            result.push('{');
            ind += 2;
            continue;
        }
        let close = chars[ind + 1..]
            .iter()
            .position(|c| *c == '}')
            .map(|offset| ind + 1 + offset)
            .unwrap_or_else(|| panic!("Single '{{' encountered in format string: {string_in}"));
        let field: String = chars[ind + 1..close].iter().collect();
        let (name, spec) = match field.split_once(':') {
            Some((name, spec)) => (name, spec),
            None => (field.as_str(), ""),
        };
        assert!(
            numbering.is_none_or(|automatic| automatic == name.is_empty()),
            "cannot switch between automatic and manual field numbering in {string_in}"
        );
        numbering = Some(name.is_empty());
        let arg_index = if name.is_empty() {
            auto_index += 1;
            auto_index - 1
        } else {
            name.parse::<usize>()
                .unwrap_or_else(|_| panic!("Unsupported replacement field {{{field}}}"))
        };
        let value = args
            .get(arg_index)
            .unwrap_or_else(|| panic!("Replacement index {arg_index} out of range"));
        // Format specification for a `str`
        let spec_chars: Vec<char> = spec.chars().collect();
        let mut fill = ' ';
        let mut align = '<';
        let mut pos = 0usize;
        if spec_chars.len() >= 2 && matches!(spec_chars[1], '<' | '>' | '^') {
            fill = spec_chars[0];
            align = spec_chars[1];
            pos = 2;
        } else if !spec_chars.is_empty() && matches!(spec_chars[0], '<' | '>' | '^') {
            align = spec_chars[0];
            pos = 1;
        }
        let rest: String = spec_chars[pos..].iter().collect();
        let rest = rest.strip_suffix('s').unwrap_or(&rest);
        let (width_text, precision_text) = match rest.split_once('.') {
            Some((width, precision)) => (width, Some(precision)),
            None => (rest, None),
        };
        let mut text: String = value.clone();
        if let Some(precision) = precision_text {
            let precision = precision
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Unsupported format spec {spec}"));
            text = text.chars().take(precision).collect();
        }
        let width = if width_text.is_empty() {
            0
        } else {
            width_text
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Unsupported format spec {spec}"))
        };
        let count = text.chars().count();
        if count < width {
            let pad = width - count;
            let (left, right) = match align {
                '>' => (pad, 0),
                '^' => (pad / 2, pad - pad / 2),
                _ => (0, pad),
            };
            result.extend(std::iter::repeat_n(fill, left));
            result.push_str(&text);
            result.extend(std::iter::repeat_n(fill, right));
        } else {
            result.push_str(&text);
        }
        ind = close + 1;
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
    // Find tilt com file under either extension and determine axis state
    let mut root = String::new();
    let mut type_ext: Option<String> = None;
    let mut stack_ext = String::new();
    let mut dual_num = 0;
    let mut tilt_sum = [0i32; 2];
    let mut eraser_sum = [0i32; 2];
    let mut tilt_ext = String::new();
    let mut eraser_ext = String::new();
    let mut try_ext: Option<String> = None;
    let mut use_tilt = use_tilt.filter(|file| !file.is_empty()).map(str::to_owned);

    // Test the passed tilt file: if it is not there or extension is bad, clear the markers
    // if it is there, set the com ex
    if let Some(file) = use_tilt.clone() {
        if Path::new(&file).exists() {
            let (_tilt_root, ext) = os_path_splitext(&file);
            if ext == ".com" || ext == ".pcm" {
                tilt_ext = ext[1..].to_owned();
                try_ext = Some(ext);
            } else {
                try_ext = None;
            }
        } else {
            use_tilt = None;
        }
    }

    // Check tilt and eraser, single and dual unless forced to do only one
    // If already got extension from passed-in tilt, skip tilt, but otherwise do try to
    // analyze tilt[ab].com
    for (com_ext, ind) in [("com", 0usize), ("pcm", 1usize)] {
        if force_single >= 0 {
            if try_ext.is_none() && Path::new(&format!("tilt.{com_ext}")).exists() {
                tilt_ext = com_ext.to_owned();
                tilt_sum[ind] += 1;
            }
            if Path::new(&format!("eraser.{com_ext}")).exists() {
                eraser_ext = com_ext.to_owned();
                eraser_sum[ind] += 1;
            }
        }
        if force_single <= 0 {
            if try_ext.is_none()
                && Path::new(&format!("tilta.{com_ext}")).exists()
                && Path::new(&format!("tiltb.{com_ext}")).exists()
            {
                tilt_ext = com_ext.to_owned();
                tilt_sum[ind] += 2;
            }
            if Path::new(&format!("erasera.{com_ext}")).exists()
                && Path::new(&format!("eraserb.{com_ext}")).exists()
            {
                eraser_ext = com_ext.to_owned();
                eraser_sum[ind] += 2;
            }
        }
    }

    // Require only com or pcm to appear, require no contamination between single and dual
    // names unless it was forced to be one or the other, but do not require both to exist
    let max_tilt = tilt_sum[0].max(tilt_sum[1]);
    let max_eraser = eraser_sum[0].max(eraser_sum[1]);
    if tilt_sum[0].min(tilt_sum[1]) > 0
        || eraser_sum[0].min(eraser_sum[1]) > 0
        || max_tilt > 2
        || max_eraser > 2
        || (max_tilt.min(max_eraser) > 0 && (max_tilt != max_eraser || eraser_ext != tilt_ext))
    {
        return (String::new(), -1, String::new(), None, String::new());
    }

    // Finalize com extension now
    let mut com_ext = tilt_ext.clone();
    if com_ext.is_empty() {
        com_ext = eraser_ext.clone();
    }
    let mut use_ext = format!(".{com_ext}");
    if max_tilt > 1 || max_eraser > 1 {
        dual_num = 2;
        use_ext = format!("a.{com_ext}");
    }

    // Then read the file options from the tilt file
    let mut tilt_root_failed = false;
    if !tilt_ext.is_empty() {
        let use_tilt = use_tilt.unwrap_or_else(|| format!("tilt{use_ext}"));

        // Tilt output is variable, so just for redundancy in test, read track file too
        let tilt_lines = read_text_file(&use_tilt, None, true, None);
        let track_lines = read_text_file(&format!("track{use_ext}"), None, true, None);
        let bad_tilt = tilt_lines.is_err();
        let bad_track = track_lines.is_err();
        if !(bad_tilt && bad_track) {
            let mut input_line = String::new();
            let mut image_line = String::new();
            let mut descrip = "";
            if let Ok(lines) = &tilt_lines {
                if let Some(OptionValue::String(value)) =
                    option_value(lines, "InputProjections", 0, false, 0, None, None)
                {
                    input_line = value;
                }
                descrip = ".ali";
            }
            if let Ok(lines) = &track_lines {
                if let Some(OptionValue::String(value)) =
                    option_value(lines, "ImageFile", 0, false, 0, None, None)
                {
                    image_line = value;
                }
                descrip = ".preali";
            }

            // If both read OK and option lines were found, proceed with two-name analysis
            if !input_line.is_empty() && !image_line.is_empty() {
                let (mut img_root, img_ext) = os_path_splitext(&image_line);
                let (new_root, inp_ext) = os_path_splitext(&input_line);
                root = new_root;

                // If the extensions are descriptive style, file type extension is blank,
                // if they match then it is a file-type extension
                if inp_ext == ".ali" && img_ext == ".preali" {
                    type_ext = Some(String::new());
                } else if inp_ext.len() > 1
                    && inp_ext == img_ext
                    && root.ends_with("ali")
                    && img_root.ends_with("preali")
                {
                    type_ext = Some(inp_ext[1..].to_owned());
                    root = py_slice_end(&root, 4);
                    img_root = py_slice_end(&img_root, 7);
                }

                // Make sure the rootname is sensible for dual axis
                if dual_num != 0 {
                    if root.ends_with('a') || root.ends_with('b') {
                        root = py_slice_end(&root, 1);
                        img_root = py_slice_end(&img_root, 1);
                    } else {
                        root = String::new();
                        tilt_root_failed = true;
                    }
                }

                if root != img_root {
                    root = String::new();
                    tilt_root_failed = true;
                }

            // Otherwise just make sure the entry from one file matches the expected one
            // and set the type extension and root that way
            } else if !input_line.is_empty() || !image_line.is_empty() {
                let (new_root, inp_ext) = if !input_line.is_empty() {
                    os_path_splitext(&input_line)
                } else {
                    os_path_splitext(&image_line)
                };
                root = new_root;
                if inp_ext == descrip {
                    type_ext = Some(String::new());
                } else if inp_ext.len() > 1 && root.ends_with(&descrip[1..]) {
                    type_ext = Some(inp_ext[1..].to_owned());
                    root = py_slice_end(&root, descrip.len());
                }

                // And rootname must still be good for dual axis
                if dual_num != 0 {
                    if root.ends_with('a') || root.ends_with('b') {
                        root = py_slice_end(&root, 1);
                    } else {
                        root = String::new();
                        tilt_root_failed = true;
                    }
                }
            }
        }
    }

    // Next get raw stack extension from eraser.com
    if !eraser_ext.is_empty() {
        if let Ok(eraser_lines) = read_text_file(&format!("eraser{use_ext}"), None, true, None) {
            if let Some(OptionValue::String(input_line)) =
                option_value(&eraser_lines, "InputFile", 0, false, 0, None, None)
            {
                let (root2, inp_ext) = os_path_splitext(&input_line);
                if inp_ext.len() > 1 {
                    stack_ext = inp_ext[1..].to_owned();
                }

                // This gives us another shot at the rootname if tilt reading failed
                if root.is_empty() && tilt_root_failed {
                    if dual_num != 0 {
                        if root2.ends_with('a') {
                            root = py_slice_end(&root2, 1);
                        }
                    } else {
                        root = root2;
                    }
                }
            }
        }
    }

    // Return what was successfully gotten
    (com_ext, dual_num, root, type_ext, stack_ext)
}

/// Rust-only: Python's `s[:-n]` for `n > 0`, counted in characters as
/// Python counts code points (the empty string when `n` reaches past the
/// start).
pub fn py_slice_end(text: &str, n: usize) -> String {
    let keep = text.chars().count().saturating_sub(n);
    text.chars().take(keep).collect()
}

/// Rust-only stand-in for Python's `os.path.splitext` (`posixpath.py`,
/// `genericpath._splitext`): the extension runs from the last `.` of the
/// last path component, provided that component has a character other than
/// `.` before it; otherwise the extension is empty.  `Path::extension`
/// differs for `name.` (Python keeps `.` as the extension).
pub fn os_path_splitext(path: &str) -> (String, String) {
    let sep_index = path.rfind('/').map(|index| index as isize).unwrap_or(-1);
    let dot_index = path.rfind('.').map(|index| index as isize).unwrap_or(-1);
    if dot_index > sep_index {
        // skip all leading dots
        let mut filename_index = (sep_index + 1) as usize;
        let dot = dot_index as usize;
        while filename_index < dot {
            if path.as_bytes()[filename_index] != b'.' {
                return (path[..dot].to_owned(), path[dot..].to_owned());
            }
            filename_index += 1;
        }
    }
    (path.to_owned(), String::new())
}

/// Rust-only stand-in for Python's `int(str)` in base 10: surrounding
/// whitespace is ignored, one `+` or `-` sign is allowed, and single
/// underscores may separate digits.  `None` is the ValueError.  Values past
/// `i64` (Python ints are unbounded) are also `None`.
pub fn py_int(text: &str) -> Option<i64> {
    // `int()`/`float()` strip Unicode whitespace but, unlike `str.strip`,
    // not U+001C..U+001F (checked with python3), which is Rust's `trim`
    let trimmed = text.trim();
    let (negative, digits) = match trimmed.as_bytes().first() {
        Some(b'-') => (true, &trimmed[1..]),
        Some(b'+') => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    let cleaned: String = digits.chars().filter(|c| *c != '_').collect();
    if !cleaned.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let value = cleaned.parse::<i64>().ok()?;
    Some(if negative { -value } else { value })
}

/// Rust-only: Python's builtin `float(str)` (CPython `PyOS_string_to_double`
/// behind `float_from_string`): surrounding whitespace is ignored; `inf`,
/// `infinity` and `nan` in any case, with an optional sign; otherwise
/// `[sign] (digits [. [digits]] | . digits) [(e|E) [sign] digits]`, where a
/// single `_` may separate two digits.  `None` is the ValueError.  Rust's
/// `str::parse::<f64>` rejects the whitespace and the underscores and accepts
/// forms Python does not.
pub fn py_float(text: &str) -> Option<f64> {
    // `int()`/`float()` strip Unicode whitespace but, unlike `str.strip`,
    // not U+001C..U+001F (checked with python3), which is Rust's `trim`
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    let mut pos = 0;
    let negative = match bytes.first() {
        Some(b'-') => {
            pos = 1;
            true
        }
        Some(b'+') => {
            pos = 1;
            false
        }
        _ => false,
    };
    let lower = trimmed[pos..].to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if lower == "nan" {
        return Some(if negative { -f64::NAN } else { f64::NAN });
    }
    // One run of digits with single underscores between digits; returns
    // the count of digits taken.
    let digit_run = |pos: &mut usize, cleaned: &mut String| -> Option<usize> {
        let mut count = 0;
        while *pos < bytes.len() {
            let byte = bytes[*pos];
            if byte.is_ascii_digit() {
                cleaned.push(byte as char);
                count += 1;
                *pos += 1;
            } else if byte == b'_'
                && count > 0
                && *pos + 1 < bytes.len()
                && bytes[*pos + 1].is_ascii_digit()
            {
                *pos += 1;
            } else if byte == b'_' {
                return None;
            } else {
                break;
            }
        }
        Some(count)
    };
    let mut cleaned = String::new();
    if negative {
        cleaned.push('-');
    }
    let int_digits = digit_run(&mut pos, &mut cleaned)?;
    let mut frac_digits = 0;
    if pos < bytes.len() && bytes[pos] == b'.' {
        cleaned.push('.');
        pos += 1;
        frac_digits = digit_run(&mut pos, &mut cleaned)?;
    }
    if int_digits + frac_digits == 0 {
        return None;
    }
    if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
        cleaned.push('e');
        pos += 1;
        if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
            cleaned.push(bytes[pos] as char);
            pos += 1;
        }
        if digit_run(&mut pos, &mut cleaned)? == 0 {
            return None;
        }
    }
    if pos != bytes.len() {
        return None;
    }
    cleaned.parse::<f64>().ok()
}

/// Rust-only: an uncaught Python exception -- the interpreter prints the
/// traceback, ending in the `Type: message` line, on standard error and
/// exits with status 1.  Only that last line is reproduced.
pub fn py_raise(exception: &str) -> ! {
    eprintln!("{exception}");
    crate::imod::libcfshr::b3dutil::exit(1)
}

/// Rust-only: Python's `int(x)` (or `int(round(x))` after [`py_round`]) of a
/// float, as the exception it raises when the script does not catch it:
/// `ValueError` for NaN, `OverflowError` for an infinity.  Rust's `as`
/// saturates instead.  A finite value truncates toward zero as `as` does
/// (Python ints are unbounded; values past `i64` saturate here).
pub fn py_try_int_of_float(value: f64) -> Result<i64, String> {
    if value.is_nan() {
        Err("ValueError: cannot convert float NaN to integer".to_owned())
    } else if value.is_infinite() {
        Err("OverflowError: cannot convert float infinity to integer".to_owned())
    } else {
        Ok(value as i64)
    }
}

/// Rust-only: [`py_try_int_of_float`] where the exception is uncaught.
pub fn py_int_of_float(value: f64) -> i64 {
    py_try_int_of_float(value).unwrap_or_else(|exception| py_raise(&exception))
}

/// Rust-only: Python's true division `a / b` of numbers, raising
/// `ZeroDivisionError` (uncaught) for a zero divisor where Rust gives an
/// infinity or NaN.
pub fn py_true_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        py_raise("ZeroDivisionError: float division by zero");
    }
    a / b
}

/// Rust-only: Python's `'%W.Pf' % x` / `'{:W.Pf}'.format(x)` of a float,
/// right-aligned in `width`.  Rust's `{:.P}` agrees except that it spells a
/// NaN `NaN` where Python writes `nan` (for either sign of NaN); `inf` and
/// `-inf` are the same in both.
pub fn py_fixed(value: f64, width: usize, precision: usize) -> String {
    let text = if value.is_nan() {
        "nan".to_owned()
    } else if value.is_infinite() {
        if value < 0. { "-inf" } else { "inf" }.to_owned()
    } else {
        format!("{value:.precision$}")
    };
    format!("{text:>width$}")
}

/// Rust-only: Python's builtin `round(number)` on a float (`float.__round__`
/// with no `ndigits`), which rounds a half to the even integer: `round(2.5)`
/// is 2 and `round(-0.5)` is 0.  Rust's `f64::round` rounds a half away from
/// zero.  The result is returned as a float; the scripts wrap it in `int()`.
/// Python raises for NaN and infinity where this returns them unchanged.
pub fn py_round(value: f64) -> f64 {
    value.round_ties_even()
}

/// Rust-only: Python's builtin `round(number, ndigits)` on a float for
/// `ndigits >= 0` (`float.__round__`, `double_round` in CPython's
/// `floatobject.c`): the exact binary value is rounded half to even to
/// `ndigits` decimals (`_Py_dg_dtoa` mode 3) and that decimal is read back
/// (`_Py_dg_strtod`).  So `round(2.675, 2)` is 2.67 (the double is below the
/// tie) and `round(-62.055, 2)` is -62.05, where scaling by 100 and rounding
/// gives -62.06.  Rust's `{:.N}` formats the exact value with the same
/// half-even rule.  Zero, NaN and infinity come back unchanged, as in CPython,
/// and so does any value when `ndigits` exceeds CPython's `NDIGITS_MAX` (323).
pub fn py_round_ndigits(value: f64, ndigits: i32) -> f64 {
    assert!(
        ndigits >= 0,
        "py_round_ndigits: negative ndigits not used by the scripts"
    );
    if ndigits > 323 || value == 0.0 || !value.is_finite() {
        return value;
    }
    format!("{:.*}", ndigits as usize, value)
        .parse::<f64>()
        .unwrap_or(value)
}

/// Rust-only: Python's `a // b` on ints, which floors (`-7 // 2` is -4 and
/// `7 // -2` is -4).  Rust's `/` truncates toward zero, and `div_euclid`
/// matches Python only for a positive divisor.  A zero divisor raises
/// `ZeroDivisionError` (uncaught), as in Python.
pub fn py_int_floordiv(a: i64, b: i64) -> i64 {
    if b == 0 {
        py_raise("ZeroDivisionError: integer division or modulo by zero");
    }
    let quotient = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        quotient - 1
    } else {
        quotient
    }
}

/// Rust-only: Python's `a % b` on ints, whose result takes the divisor's
/// sign (`-7 % 2` is 1, `7 % -2` is -1).  Rust's `%` takes the dividend's,
/// and `rem_euclid` is never negative.
pub fn py_int_mod(a: i64, b: i64) -> i64 {
    if b == 0 {
        py_raise("ZeroDivisionError: integer modulo by zero");
    }
    let remainder = a % b;
    if remainder != 0 && ((remainder < 0) != (b < 0)) {
        remainder + b
    } else {
        remainder
    }
}

/// Rust-only: Python's `a % b` on floats (`float_rem` in CPython's
/// `floatobject.c`): `fmod`, moved to the divisor's sign, with a zero result
/// carrying the divisor's sign.
pub fn py_float_mod(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        py_raise("ZeroDivisionError: float modulo by zero");
    }
    let mut remainder = a % b;
    if remainder != 0.0 {
        if (b < 0.0) != (remainder < 0.0) {
            remainder += b;
        }
    } else {
        remainder = 0.0_f64.copysign(b);
    }
    remainder
}

/// Rust-only: Python's `a // b` on floats (`float_floor_div` /
/// `_float_div_mod` in CPython's `floatobject.c`).  It is computed from
/// `fmod`, not as `floor(a / b)`, and the two differ: `1 // 0.1` is 9.0 in
/// Python while `(1.0 / 0.1).floor()` is 10.0.
pub fn py_float_floordiv(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        py_raise("ZeroDivisionError: float floor division by zero");
    }
    let remainder = a % b;
    let mut div = (a - remainder) / b;
    if remainder != 0.0 && (b < 0.0) != (remainder < 0.0) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        floordiv
    } else {
        0.0_f64.copysign(a / b)
    }
}

/// Rust-only: `repr(float)` / `str(float)` (Python 3, `float_repr_style`
/// `'short'`): the shortest decimal that round-trips, `.0` forced on an
/// integral value in positional form, and exponent form outside
/// `1e-4 ..= 1e16` (where an integral mantissa stays bare, `1e-05`).
///
/// Two formatters that both produce "the shortest round-tripping decimal"
/// still differ when two equally short decimals both round-trip: CPython's
/// `_Py_dg_dtoa` mode 0 takes the one nearest the exact value, rounding a
/// half to even, while Rust's `{:e}` does not (`811212085039910.25` is
/// `811212085039910.2` in Python and `...910.3` from `{}`).  So the digit
/// count comes from `{:e}` and the digits from `{:.*e}`, which rounds the
/// exact value half to even at that count.  Rust's `{}` also prints `1` for
/// `1.0` and never uses exponent form, so a string built by a script with
/// `str()` or `'{}'.format()` needs this.
pub fn py_str_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value < 0.0 {
            "-inf".to_owned()
        } else {
            "inf".to_owned()
        };
    }
    let shortest = format!("{value:e}");
    let num_digits = shortest
        .split_once('e')
        .map_or(shortest.as_str(), |(mantissa, _)| mantissa)
        .chars()
        .filter(|c| c.is_ascii_digit())
        .count();
    let scientific = format!("{:.*e}", num_digits.saturating_sub(1), value);
    let (mantissa, exponent_text) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let exponent: i32 = exponent_text.parse().unwrap_or(0);
    if exponent < -4 || exponent >= 16 {
        // `repr` keeps an integral mantissa bare: `1e-05`, `1e+16`
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    if exponent < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-exponent - 1) as usize));
    }
    let int_len = exponent as usize + 1;
    if digits.len() <= int_len {
        format!("{sign}{digits}{}.0", "0".repeat(int_len - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..int_len], &digits[int_len..])
    }
}

/// Rust-only stand-in for Python's `os.path.normpath` (`posixpath.py`):
/// collapses repeated separators and `.` components and resolves `..`
/// lexically, keeping exactly two leading slashes but reducing three or more
/// to one, and returning `.` for an empty result.  `Path::components` differs:
/// it keeps `..` and folds `//` to `/`.
pub fn os_path_normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_owned();
    }
    let mut initial_slashes = if path.starts_with('/') { 1 } else { 0 };
    // POSIX allows one or two initial slashes, but treats three or more
    // as single slash.
    if initial_slashes == 1 && path.starts_with("//") && !path.starts_with("///") {
        initial_slashes = 2;
    }
    let mut new_comps: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".."
            || (initial_slashes == 0 && new_comps.is_empty())
            || new_comps.last() == Some(&"..")
        {
            new_comps.push(comp);
        } else if !new_comps.is_empty() {
            new_comps.pop();
        }
    }
    let mut result = "/".repeat(initial_slashes);
    result.push_str(&new_comps.join("/"));
    if result.is_empty() {
        return ".".to_owned();
    }
    result
}

/// Rust-only stand-in for Python's `os.path.abspath` (`posixpath.py`):
/// joins a relative path onto `os.getcwd()` and normalises it with
/// [`os_path_normpath`], without touching the file system beyond the working
/// directory -- so, unlike `fs::canonicalize`, symbolic links are kept and
/// the path need not exist, and unlike `std::path::absolute`, `..` is
/// resolved.
pub fn os_path_abspath(path: &str) -> String {
    if path.starts_with('/') {
        return os_path_normpath(path);
    }
    let cwd = std::env::current_dir()
        .map(|dir| dir.to_string_lossy().into_owned())
        .unwrap_or_default();
    // `os.path.join(cwd, path)`
    let mut joined = cwd;
    if !joined.is_empty() && !joined.ends_with('/') {
        joined.push('/');
    }
    joined.push_str(path);
    os_path_normpath(&joined)
}

/// Rust-only stand-in for Python's `glob.glob(pattern)` (`glob.py`, Python
/// 3.12) for a pattern whose directory part holds no wildcard, which is every
/// pattern the translated scripts pass: without a wildcard the pattern itself
/// is returned if it exists (`os.path.lexists`); otherwise the directory is
/// listed in `os.scandir` order -- `read_dir`, both being `readdir` -- names
/// starting with `.` are skipped unless the pattern's name part does, and the
/// rest are matched with `fnmatch` (`*`, `?`, `[...]`, `[!...]`).
pub fn glob_glob(pattern: &str) -> Vec<String> {
    let has_magic = |text: &str| text.contains(['*', '?', '[']);
    if !has_magic(pattern) {
        if fs::symlink_metadata(pattern).is_ok() {
            return vec![pattern.to_owned()];
        }
        return Vec::new();
    }
    let (dirname, basename) = match pattern.rfind('/') {
        Some(index) => (&pattern[..index + 1], &pattern[index + 1..]),
        None => ("", pattern),
    };
    // `fnmatch.translate`
    let chars = basename.chars().collect::<Vec<_>>();
    let mut regex_text = String::from("(?s:");
    let mut i = 0;
    let n = chars.len();
    while i < n {
        let c = chars[i];
        i += 1;
        if c == '*' {
            if !regex_text.ends_with(".*") {
                regex_text.push_str(".*");
            }
        } else if c == '?' {
            regex_text.push('.');
        } else if c == '[' {
            let mut j = i;
            if j < n && chars[j] == '!' {
                j += 1;
            }
            if j < n && chars[j] == ']' {
                j += 1;
            }
            while j < n && chars[j] != ']' {
                j += 1;
            }
            if j >= n {
                regex_text.push_str("\\[");
            } else {
                let mut stuff = chars[i..j].iter().collect::<String>();
                stuff = stuff.replace('\\', "\\\\");
                i = j + 1;
                if let Some(rest) = stuff.strip_prefix('!') {
                    stuff = format!("^{rest}");
                } else if stuff.starts_with('^') || stuff.starts_with('[') {
                    stuff = format!("\\{stuff}");
                }
                regex_text.push('[');
                regex_text.push_str(&stuff);
                regex_text.push(']');
            }
        } else {
            regex_text.push_str(&regex::escape(&c.to_string()));
        }
    }
    regex_text.push_str(")\\z");
    let Ok(matcher) = regex::Regex::new(&format!("^{regex_text}")) else {
        return Vec::new();
    };
    let list_dir = if dirname.is_empty() { "." } else { dirname };
    let mut result = Vec::new();
    if let Ok(entries) = fs::read_dir(list_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') && !basename.starts_with('.') {
                continue;
            }
            if matcher.is_match(&name) {
                result.push(format!("{dirname}{name}"));
            }
        }
    }
    result
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
    // `if not outputFormat`: an empty format counts as none
    let output_format = output_format
        .filter(|format| !format.is_empty())
        .map(str::to_owned)
        .or_else(|| {
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
        let backname = format!("{filename}~");
        let renamed: std::io::Result<()> = (|| {
            if Path::new(&backname).exists() {
                fs::remove_file(&backname)?;
            }
            fs::rename(filename, &backname)
        })();
        if renamed.is_err() {
            prnstr(
                &format!("WARNING: Failed to rename existing file {filename} to {backname}"),
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
    crate::imod::libcfshr::b3dutil::exit(1)
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
    let mut absp = os_path_abspath(&cygwin_path(path));
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
        .and_then(py_int)
        .is_some_and(|major| major > 1)
}

/// Matches `imodTempDir` (`IMOD/pysrc/imodpy.py:1486`).
pub fn imod_temp_dir() -> String {
    // `os.access(path, os.W_OK)`: the POSIX `access` call itself, which asks
    // about this process's real user rather than the permission bits alone
    let writable = |path: &str| -> bool {
        crate::imod::libcfshr::b3dutil::os_access(path, crate::imod::libcfshr::b3dutil::W_OK)
    };
    let windows = cfg!(windows) || cfg!(target_os = "cygwin");
    if let Some(imodtemp) = std::env::var_os("IMOD_TMPDIR") {
        let imodtemp = get_cygpath(windows, &imodtemp.to_string_lossy(), "-m");
        if Path::new(&imodtemp).exists() && Path::new(&imodtemp).is_dir() && writable(&imodtemp) {
            return imodtemp;
        }
    }
    let imodtemp = get_cygpath(windows, "/usr/tmp", "-m");
    if Path::new(&imodtemp).exists() && writable(&imodtemp) {
        return imodtemp;
    }
    let imodtemp = get_cygpath(windows, "/tmp", "-m");
    if Path::new(&imodtemp).exists() && writable(&imodtemp) {
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
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGHUP, libc::SIG_IGN);
    }
}

/// Matches `imodNice` (`IMOD/pysrc/imodpy.py:1463`) at the POSIX process-priority boundary.
pub fn imod_nice(nice_increment: i32) -> i32 {
    #[cfg(windows)]
    {
        // `imodpy.py:1467-1480`: `psutil.Process(os.getpid()).nice(priority)`,
        // which is `SetPriorityClass` on the current process.
        if nice_increment < 4 {
            return 0;
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn SetPriorityClass(process: *mut core::ffi::c_void, class: u32) -> i32;
        }
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        const IDLE_PRIORITY_CLASS: u32 = 0x0000_0040;
        let priority = if nice_increment <= 15 {
            BELOW_NORMAL_PRIORITY_CLASS
        } else {
            IDLE_PRIORITY_CLASS
        };
        // SAFETY: the pseudo-handle of the current process.
        unsafe { SetPriorityClass(GetCurrentProcess(), priority) };
        return 0;
    }
    #[cfg(unix)]
    unsafe {
        libc::nice(nice_increment);
    }
    0
}

/// Rust-only: the Win32 calls behind the `psutil` methods IMOD's Windows
/// Python arms use (`imodkillgroup`, `b3dwinps`, `imodNice`).  `psutil` is a
/// third-party module that Windows IMOD requires; these are the calls it makes
/// there.  The `psutil.Process` object is represented by its PID.
#[cfg(windows)]
pub mod psutil {
    type Handle = *mut core::ffi::c_void;
    const TH32CS_SNAPPROCESS: u32 = 0x0000_0002;
    const PROCESS_TERMINATE: u32 = 0x0001;
    const PROCESS_SUSPEND_RESUME: u32 = 0x0800;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const INVALID_HANDLE_VALUE: Handle = -1_isize as Handle;

    #[repr(C)]
    struct ProcessEntry32W {
        dw_size: u32,
        cnt_usage: u32,
        th32_process_id: u32,
        th32_default_heap_id: usize,
        th32_module_id: u32,
        cnt_threads: u32,
        th32_parent_process_id: u32,
        pc_pri_class_base: i32,
        dw_flags: u32,
        sz_exe_file: [u16; 260],
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> Handle;
        fn Process32FirstW(snapshot: Handle, entry: *mut ProcessEntry32W) -> i32;
        fn Process32NextW(snapshot: Handle, entry: *mut ProcessEntry32W) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn QueryFullProcessImageNameW(
            process: Handle,
            flags: u32,
            name: *mut u16,
            size: *mut u32,
        ) -> i32;
        fn TerminateProcess(process: Handle, exit_code: u32) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
        fn GetProcessTimes(
            process: Handle,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
    }
    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn OpenProcessToken(process: Handle, access: u32, token: *mut Handle) -> i32;
        fn GetTokenInformation(
            token: Handle,
            class: i32,
            information: *mut core::ffi::c_void,
            length: u32,
            return_length: *mut u32,
        ) -> i32;
        fn LookupAccountSidW(
            system: *const u16,
            sid: *mut core::ffi::c_void,
            name: *mut u16,
            name_length: *mut u32,
            domain: *mut u16,
            domain_length: *mut u32,
            use_: *mut i32,
        ) -> i32;
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtSuspendProcess(process: Handle) -> i32;
    }

    /// `psutil.process_iter()` with `ppid()`: (PID, parent PID) pairs.
    pub fn process_list() -> Vec<(i64, i64)> {
        let mut list = Vec::new();
        // SAFETY: the snapshot handle is checked and closed; the entry has
        // its size set as the API requires.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return list;
            }
            let mut entry: ProcessEntry32W = core::mem::zeroed();
            entry.dw_size = core::mem::size_of::<ProcessEntry32W>() as u32;
            let mut ok = Process32FirstW(snapshot, &mut entry);
            while ok != 0 {
                list.push((
                    entry.th32_process_id as i64,
                    entry.th32_parent_process_id as i64,
                ));
                ok = Process32NextW(snapshot, &mut entry);
            }
            CloseHandle(snapshot);
        }
        list
    }

    /// `proc.exe()`, or `None` for `AccessDenied`/`NoSuchProcess`.
    pub fn process_exe(pid: i64) -> Option<String> {
        // SAFETY: the process handle is checked and closed.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
            if handle.is_null() {
                return None;
            }
            let mut name = [0u16; 32768];
            let mut size = name.len() as u32;
            let ok = QueryFullProcessImageNameW(handle, 0, name.as_mut_ptr(), &mut size);
            CloseHandle(handle);
            if ok == 0 {
                return None;
            }
            Some(String::from_utf16_lossy(&name[..size as usize]))
        }
    }

    fn with_process(
        pid: i64,
        access: u32,
        call: impl FnOnce(Handle) -> bool,
    ) -> Result<(), String> {
        // SAFETY: the process handle is checked and closed.
        unsafe {
            let handle = OpenProcess(access, 0, pid as u32);
            if handle.is_null() {
                return Err(std::io::Error::last_os_error().to_string());
            }
            let ok = call(handle);
            let error = std::io::Error::last_os_error();
            CloseHandle(handle);
            if ok { Ok(()) } else { Err(error.to_string()) }
        }
    }

    /// `proc.username()`: `DOMAIN\\user` of the process's token owner, or
    /// `None` for `AccessDenied`/`NoSuchProcess`.
    pub fn username(pid: i64) -> Option<String> {
        const TOKEN_QUERY: u32 = 0x0008;
        const TOKEN_USER: i32 = 1;
        // SAFETY: handles are checked and closed; the token buffer is sized
        // by the first call, and its first field is the SID pointer.
        unsafe {
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
            if process.is_null() {
                return None;
            }
            let mut token: Handle = core::ptr::null_mut();
            let opened = OpenProcessToken(process, TOKEN_QUERY, &mut token) != 0;
            CloseHandle(process);
            if !opened {
                return None;
            }
            let mut length = 0_u32;
            GetTokenInformation(token, TOKEN_USER, core::ptr::null_mut(), 0, &mut length);
            let mut buffer = vec![0_usize; (length as usize).div_ceil(8).max(1)];
            let ok = GetTokenInformation(
                token,
                TOKEN_USER,
                buffer.as_mut_ptr().cast(),
                length,
                &mut length,
            ) != 0;
            CloseHandle(token);
            if !ok {
                return None;
            }
            let sid = buffer[0] as *mut core::ffi::c_void;
            let mut name = [0_u16; 256];
            let mut domain = [0_u16; 256];
            let mut name_length = name.len() as u32;
            let mut domain_length = domain.len() as u32;
            let mut use_ = 0_i32;
            if LookupAccountSidW(
                core::ptr::null(),
                sid,
                name.as_mut_ptr(),
                &mut name_length,
                domain.as_mut_ptr(),
                &mut domain_length,
                &mut use_,
            ) == 0
            {
                return None;
            }
            Some(format!(
                "{}\\{}",
                String::from_utf16_lossy(&domain[..domain_length as usize]),
                String::from_utf16_lossy(&name[..name_length as usize])
            ))
        }
    }

    /// `proc.create_time()`: seconds since the epoch, as a float.
    pub fn create_time(pid: i64) -> Option<f64> {
        // SAFETY: the process handle is checked and closed.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
            if handle.is_null() {
                return None;
            }
            let (mut creation, mut exit, mut kernel, mut user) = (0_u64, 0_u64, 0_u64, 0_u64);
            let ok = GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) != 0;
            CloseHandle(handle);
            if !ok {
                return None;
            }
            // FILETIME counts 100 ns from 1601; psutil subtracts the epoch.
            Some((creation as f64 - 116_444_736_000_000_000.0) / 10_000_000.0)
        }
    }

    /// `proc.suspend()`.
    pub fn suspend(pid: i64) -> Result<(), String> {
        with_process(pid, PROCESS_SUSPEND_RESUME, |handle| unsafe {
            NtSuspendProcess(handle) >= 0
        })
    }

    /// `proc.kill()`: psutil terminates with exit code `SIGTERM` (15).
    pub fn kill(pid: i64) -> Result<(), String> {
        with_process(pid, PROCESS_TERMINATE, |handle| unsafe {
            TerminateProcess(handle, 15) != 0
        })
    }
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
        // `imodpy.py:1578-1585`: set the user bits with `os.chmod`, and when
        // that fails, report whether a file can be written there at all
        Err(_) => {
            let mode = fs::metadata(".")
                .map(|meta| {
                    crate::imod::libcfshr::b3dutil::py_st_mode(&meta, Path::new(".")) & 0o7777
                })
                .unwrap_or(0);
            if crate::imod::libcfshr::b3dutil::py_chmod(".", mode | 0o400 | 0o200 | 0o100).is_err()
            {
                let test_file = format!("{subdirectory}/writetest.tmp");
                let err_str =
                    write_text_file(&test_file, &["Test for writability".to_owned()], true).err();
                cleanup_files(&[test_file]);
                return err_str;
            }
            None
        }
    }
}

/// Matches `initializeMoveOrCopy` (`IMOD/pysrc/imodpy.py:1653`).  Set
/// `skip_win_copy` to 1 to not use robocopy, 2 to not use xcopy either.
pub fn initialize_move_or_copy(skip_win_copy: i32) {
    MOC_RENAMES_OK.store(true, Ordering::SeqCst);
    MOC_XCOPY_EXISTS.store(false, Ordering::SeqCst);

    // If in Windows, see if recursive copy available with native Windows commands
    // xcopy is deprecated so prefer robocopy
    if cfg!(windows) {
        let xcopy_exists = Path::new("C:/Windows/system32/xcopy.exe").exists() && skip_win_copy < 2;
        MOC_XCOPY_EXISTS.store(xcopy_exists, Ordering::SeqCst);
        if skip_win_copy > 0 || !Path::new("C:/Windows/system32/robocopy.exe").exists() {
            if xcopy_exists {
                MOC_USE_XCOPY.store(true, Ordering::SeqCst);
            } else {
                MOC_RECURSIVE_COPY_OK.store(false, Ordering::SeqCst);
            }
        }
    }
}

/// Matches `moveOrCopyWithRetry` (`IMOD/pysrc/imodpy.py:1672`), with its
/// Windows arm (`ROBOCOPY`/`XCOPY`/`COPY` through `runcmd`); the Cygwin
/// variants of that arm are not translated.  `cp -rf` is an external process
/// boundary.
///
/// Not translated: the `numTrials > 1` branch's `shutil.copy`/`shutil.move`
/// for a **directory** (`shutil.move`'s `copytree` fallback); a file is
/// handled.  No caller in this crate reaches this function (its source
/// callers are `serieswatcher` and `framewatcher`); see TOFIX.md.
pub fn move_or_copy_with_retry(
    from_file: &str,
    to_dir: &str,
    mess: &str,
    if_copy: bool,
    num_trials: i32,
) -> i32 {
    let moving_dir = Path::new(from_file).is_dir();
    let from_file = os_path_normpath(from_file);
    let to_dir = os_path_normpath(to_dir);
    // `os.path.basename`
    let basename = |path: &str| -> String { path.rsplit('/').next().unwrap_or(path).to_owned() };

    // If moving and renames have been OK so far or not tested yet, try one os.rename
    // and if anything goes wrong, mark renames as bad and fall back to copy/deletes
    let is_windows = cfg!(windows);
    // `os.stat(...).st_dev`, the volume on Windows: the path's drive prefix
    let same_device = || -> bool {
        let prefix = |path: &str| {
            fs::canonicalize(path).ok().and_then(|full| {
                full.components()
                    .next()
                    .map(|first| first.as_os_str().to_owned())
            })
        };
        prefix(&from_file).is_some() && prefix(&from_file) == prefix(&to_dir)
    };
    // But do not try this on Windows unless it is the same file system
    if MOC_RENAMES_OK.load(Ordering::SeqCst) && !if_copy && (!is_windows || same_device()) {
        if fs::rename(&from_file, format!("{to_dir}/{}", basename(&from_file))).is_ok() {
            return 0;
        }
        MOC_RENAMES_OK.store(false, Ordering::SeqCst);
    }

    let mut last_error = String::new();
    for trial in 0..num_trials {
        let attempt: Result<(), String> = (|| {
            // If multiple trials, or if moving file and no directory copy
            // available at all, use the dog-slow shutil
            if num_trials > 1 || (moving_dir && !MOC_RECURSIVE_COPY_OK.load(Ordering::SeqCst)) {
                let real_dst = format!("{to_dir}/{}", basename(&from_file));
                if if_copy {
                    // `shutil.copy`: contents and permission bits
                    fs::copy(&from_file, &real_dst).map_err(|error| error.to_string())?;
                } else {
                    // `shutil.move`
                    if Path::new(&real_dst).exists() {
                        return Err(format!("Destination path '{real_dst}' already exists"));
                    }
                    if fs::rename(&from_file, &real_dst).is_err() {
                        fs::copy(&from_file, &real_dst).map_err(|error| error.to_string())?;
                        fs::remove_file(&from_file).map_err(|error| error.to_string())?;
                    }
                }
            } else {
                // Otherwise, set up to copy file or directory, using the extended
                // copy command in Windows for directories
                let mut ignore: Vec<i32> = Vec::new();
                let mut move_file_with_robo = false;
                let mut move_dir_with_robo = false;
                let command = if is_windows {
                    let recursive_ok = MOC_RECURSIVE_COPY_OK.load(Ordering::SeqCst);
                    let use_xcopy = MOC_USE_XCOPY.load(Ordering::SeqCst);
                    // Move a single file with robocopy if that exists; good exit status is 3
                    move_file_with_robo = !if_copy && !moving_dir && recursive_ok && !use_xcopy;
                    if move_file_with_robo {
                        // `os.path.dirname`
                        let mut from_dir = Path::new(&from_file)
                            .parent()
                            .map(|dir| dir.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if from_dir.is_empty() {
                            from_dir = ".".to_owned();
                        }
                        ignore = vec![1, 3];
                        fmtstr(
                            "ROBOCOPY /S /MOV \"{}\" \"{}\" \"{}\"",
                            &[from_dir, to_dir.clone(), basename(&from_file)],
                        )
                    } else if !moving_dir {
                        // Or copy a single file if copying, or if can't move with robocopy
                        fmtstr(
                            "{}COPY /Y \"{}\" \"{}\"",
                            &[String::new(), from_file.clone(), to_dir.clone()],
                        )
                    } else {
                        // Or copy directory with xcopy or robocopy: the good exit status is 1
                        let command = if use_xcopy {
                            "XCOPY /Y /S /I".to_owned()
                        } else {
                            ignore = vec![1];
                            move_dir_with_robo = true;
                            "ROBOCOPY /S /MOVE".to_owned()
                        };
                        let to_file = format!("{to_dir}/{}", basename(&from_file));
                        fmtstr("{} \"{}\" \"{}\"", &[command, from_file.clone(), to_file])
                    }
                } else {
                    fmtstr("cp -rf \"{}\" \"{}\"", &[from_file.clone(), to_dir.clone()])
                };
                // `runcmd` raises a bare `ImodpyError`, whose `str()` is empty
                run_cmd(&command, None, None, None, &ignore).map_err(|_| String::new())?;

                // Now for move, remove the tree or file if not done with robocopy
                if !if_copy && !move_file_with_robo && !move_dir_with_robo {
                    if moving_dir {
                        fs::remove_dir_all(&from_file).map_err(|error| error.to_string())?;
                    } else {
                        fs::remove_file(&from_file).map_err(|error| error.to_string())?;
                    }
                }
            }
            Ok(())
        })();
        match attempt {
            Ok(()) => return 0,
            // Just catch anything here and retry or give up
            Err(error) => {
                last_error = error;
                if trial < num_trials - 1 {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
    }

    prnstr(
        &format!("An error occurred {mess} to {to_dir} :"),
        "\n",
        false,
    );
    prnstr(&format!("    {last_error}"), "\n", false);
    1
}

/// Matches `patchSizeFromEntry` (`IMOD/pysrc/imodpy.py:1597`).
pub fn patch_size_from_entry(patch_entry: &str) -> (i32, i32, i32, i32) {
    let index = "SMLE".find(patch_entry.to_ascii_uppercase().as_str());
    if let Some(index) = index.filter(|_| patch_entry.len() == 1) {
        let xy = [64, 80, 100, 120][index] as i32;
        return (xy, xy, [32, 40, 50, 60][index], 0);
    }
    // `int()` strips surrounding whitespace, which the `#, #, #` form Etomo
    // sends to setupcombine relies on
    let values = patch_entry
        .split(',')
        .map(|value| py_int(value).and_then(|value| i32::try_from(value).ok()))
        .collect::<Option<Vec<_>>>();
    match values {
        Some(values) if values.len() == 3 => (values[0], values[1], values[2], 0),
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
        delta = delta
            .min(py_int_floordiv(3 * size as i64, 4) as i32)
            .min([30, 20][density_index]);
    }
    // A Z size under 2 makes `delta` 0: Python raises ZeroDivisionError, which
    // no caller catches.  Otherwise the quotient is finite: integer operands
    // over a nonzero integer.
    py_round(py_true_div((upper - lower - size) as f64, delta as f64) + 1.0) as i32
}

/// Matches `parallelBoundarySize` (`IMOD/pysrc/imodpy.py:1629`).
pub fn parallel_boundary_size(default_value: i32) -> i32 {
    if let Ok(val_str) = std::env::var("PARALLEL_BOUNDARY_SIZE")
        && !val_str.is_empty()
        && let Some(new_val) = py_int(&val_str)
        && new_val > default_value as i64
    {
        return new_val as i32;
    }
    default_value
}

/// Matches `elapsedTimeComponents` (`IMOD/pysrc/imodpy.py:1642`), with Unix seconds as input.
pub fn elapsed_time_components(start_time: f64) -> (i32, i32, i32) {
    let used = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        - start_time;
    // finite: a difference of two wall-clock times
    let minutes = (used / 60.0) as i32;
    let seconds_float = used - 60.0 * minutes as f64;
    let seconds = seconds_float as i32;
    // Python 3's `round` rounds half to even
    let fraction = ((seconds_float - seconds as f64) * 10.0).round_ties_even() as i32;
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
    use super::{
        OptionValue, option_value, parse_list, py_float_floordiv, py_float_mod, py_int_floordiv,
        py_int_mod, py_round, py_round_ndigits, py_str_float,
    };
    use super::{py_fixed, py_float, py_int, py_try_int_of_float};

    #[test]
    fn python_int_of_float_raises_like_cpython() {
        assert_eq!(py_try_int_of_float(-2.7), Ok(-2));
        assert_eq!(
            py_try_int_of_float(f64::NAN),
            Err("ValueError: cannot convert float NaN to integer".to_owned())
        );
        assert_eq!(
            py_try_int_of_float(f64::NEG_INFINITY),
            Err("OverflowError: cannot convert float infinity to integer".to_owned())
        );
    }

    /// `float()`/`int()`/`'%W.Pf'` as CPython 3 gives them (`python3 -c`).
    #[test]
    fn python_float_int_and_fixed_match_cpython() {
        assert_eq!(py_float(" 1.5\n"), Some(1.5));
        assert_eq!(py_float("1_000.000_1e-0_3"), Some(1.0000001));
        assert_eq!(py_float("5."), Some(5.0));
        assert_eq!(py_float(".5"), Some(0.5));
        assert_eq!(py_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(py_float("+NaN").is_some_and(f64::is_nan));
        for bad in [
            "1__0", "_1", "1_", "1._5", ".", "e5", "1e", "0x10", "1 2", "infinit", "\u{1c}7",
        ] {
            assert_eq!(py_float(bad), None, "{bad:?}");
        }
        assert_eq!(py_int(" 4\n"), Some(4));
        assert_eq!(py_int("1_0"), Some(10));
        assert_eq!(py_int("1.0"), None);
        assert_eq!(py_fixed(f64::NAN, 8, 3), "     nan");
        assert_eq!(py_fixed(-f64::NAN, 0, 2), "nan");
        assert_eq!(py_fixed(f64::NEG_INFINITY, 0, 1), "-inf");
        assert_eq!(py_fixed(-0.0, 0, 3), "-0.000");
        assert_eq!(py_fixed(2.675, 6, 2), "  2.67");
    }

    /// Expected values are CPython 3's (`python3 -c`), at the ties where the
    /// Rust builtins differ.
    #[test]
    fn python_round_matches_cpython_at_ties() {
        assert_eq!(py_round(12.5), 12.);
        assert_eq!(py_round(2.5), 2.);
        assert_eq!(py_round(3.5), 4.);
        assert_eq!(py_round(-0.5), 0.);
        // `round(x, 2)` rounds the exact binary value: scaling by 100 first
        // gave -62.06 and 3024.58
        assert_eq!(py_round_ndigits(-62.055, 2), -62.05);
        assert_eq!(py_round_ndigits(3024.585, 2), 3024.59);
        assert_eq!(py_round_ndigits(56327.395, 2), 56327.39);
        assert_eq!(py_round_ndigits(2.675, 2), 2.67);
        assert_eq!(py_round_ndigits(0.125, 2), 0.12);
        assert_eq!(py_round_ndigits(1.0005, 3), 1.0);
    }

    #[test]
    fn python_float_str_matches_cpython_repr() {
        let cases = [
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123.456, "123.456"),
            // two equally short decimals round-trip; CPython takes the even one
            (811212085039910.25, "811212085039910.2"),
            (1573626427739.15625, "1573626427739.1562"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for (value, text) in cases {
            assert_eq!(py_str_float(value), text, "{value:e}");
        }
    }

    #[test]
    fn python_floor_division_and_modulo() {
        assert_eq!(py_int_floordiv(-7, 2), -4);
        assert_eq!(py_int_floordiv(7, -2), -4);
        assert_eq!(py_int_floordiv(1021, -2), -511);
        assert_eq!(py_int_mod(-7, 2), 1);
        assert_eq!(py_int_mod(7, -2), -1);
        assert_eq!(py_float_floordiv(1.0, 0.1), 9.0);
        assert_eq!(py_float_floordiv(-7.5, 2.0), -4.0);
        assert_eq!(py_float_mod(-7.5, 2.0), 0.5);
        assert_eq!(py_float_mod(7.5, -2.0), -0.5);
    }

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

/// Rust-only: rewrites a pattern written for Python's `re` module into the
/// `regex` crate's syntax, for the constructs the two spell differently.
/// Inside a character class Python takes `[` literally and has no set
/// operations, while the crate starts a nested class at `[` and treats `&&`,
/// `--` and `~~` as intersection, difference and symmetric difference; those
/// are escaped.  Everything else (escapes, groups, quantifiers) is the same in
/// both for the patterns the scripts build.  `sorttiltframes` builds its
/// pattern from user-entered delimiters, `[` and `]` by default.
pub fn py_regex(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() + 8);
    let mut ind = 0usize;
    let mut in_class = false;
    let mut class_start = 0usize;
    while ind < chars.len() {
        let c = chars[ind];
        if c == '\\' {
            out.push(c);
            if let Some(&next) = chars.get(ind + 1) {
                out.push(next);
            }
            ind += 2;
            continue;
        }
        if !in_class {
            if c == '[' {
                in_class = true;
                class_start = ind;
            }
            out.push(c);
            ind += 1;
            continue;
        }
        // Inside a class: a `]` right after `[` or `[^` is a literal.
        let first =
            ind == class_start + 1 || (ind == class_start + 2 && chars[class_start + 1] == '^');
        if c == ']' && !first {
            in_class = false;
            out.push(c);
        } else if c == '[' || (c == ']' && first) {
            out.push('\\');
            out.push(c);
        } else if matches!(c, '&' | '-' | '~') && chars.get(ind + 1) == Some(&c) {
            out.push('\\');
            out.push(c);
        } else {
            out.push(c);
        }
        ind += 1;
    }
    out
}
