//! The in-process command file runner: runs an IMOD `.com` file without
//! `vmstocsh | tcsh` or `vmstopy` + `python`.
//!
//! **Not a translated unit.**  Upstream IMOD runs a command file by
//! converting it to a shell script (`vmstocsh`, `IMOD/flib/image/vmstocsh.f`)
//! or to a Python script (`vmstopy`, `IMOD/pysrc/vmstopy`) and running that
//! with `tcsh` or `python`.  This module does the same work in this process,
//! owner decisions of 2026-09-26: *"long term I do not want our software to
//! rely on any piping. it is not very portable and is fragile"* and
//! *"whenever possible, definitely do direct function calls instead of
//! piping, if within control"*.  See `CLAUDE.md`, "The in-process command
//! file runner".
//!
//! # How it works
//!
//! The command file is converted by [`vmstopy::convert`] -- the faithful
//! translation of `vmstopy`'s parser, verified byte for byte against the
//! native script -- and the Python text that produces is then *executed* here
//! by a small interpreter for exactly the statements `vmstopy` emits.  So the
//! language (labels and `$goto`, `$exit`, `$if ($status) goto`, `$set`,
//! `$setenv`, `$echo`, `$if (-e file) command`, continuation lines, standard
//! input lines, `%var`/`$var`/`${ENV}` substitution, `$$`, `` `hostname` ``,
//! `rm`/`cp`/`mv`/`mkdir` rewriting, nested `$vmstocsh x.log < x.com | csh
//! -ef`) is whatever `vmstopy` makes of it, and the runtime semantics are
//! those of the generated script:
//!
//! * the log is backed up to `log~` and truncated, and every program's
//!   standard output **and standard error** go to it
//!   (`runcmd(command, input, log, 'stdout')`);
//! * the first program that exits non-zero stops the run: the log gets
//!   `ERROR: <command>: exited with status <n>` and the status is 1, unless
//!   the next line is `$if ($status) goto label`, in which case the lines
//!   from `label:` to `$exit n` run instead;
//! * `$exit n` ends the run with status `n`; success writes
//!   `SUCCESSFULLY COMPLETED` (and `CHUNK DONE` with `-c`) to the log;
//! * an undefined `${ENV}` gives `ERROR: Environment variable not defined:
//!   'ENV'` and status 1; any other runtime error `ERROR: Unknown error
//!   running commands: ...` and status 1;
//! * a construct `vmstopy` rejects is rejected with the same message, and
//!   nothing runs.
//!
//! Where `vmstocsh | tcsh -ef` differs from that (standard error is not
//! logged; the log is truncated by the first command rather than at the
//! start; `$if ($status) goto` is unreachable under `-e`), the runner follows
//! `vmstopy`: it is the default of `submfg` and the converter `processchunks`
//! uses.  Output *files* and the exit status (zero or not) agree between the
//! two for the command files IMOD writes.
//!
//! # How a command runs
//!
//! After `vmstopy`'s substitutions, a command line that needs nothing from a
//! shell but word splitting and quote removal is run
//!
//! * **in process** through [`crate::imod::commands::run_in_process`] when its
//!   first word names a command-table entry with `in_process` set, or is
//!   `b3dcopy`, `b3dremove -g` (which globs its own arguments) or `sync`;
//! * as a child process of **this crate's own binary** when it names another
//!   table entry (the Python-script translations, `processchunks`, ...) and
//!   this process is the `imod` launcher;
//! * otherwise -- a program this crate does not provide, or a line that needs
//!   a shell (pipes, redirection, globs, `$`) -- through `/bin/sh -c`, as the
//!   generated script's `runcmd` does.  **This fallback is temporary**; it
//!   goes when every program the command files name is translated.
//!
//! Standard input lines are fed to the program as text; a program with none
//! reads end-of-file, never the caller's standard input.
//!
//! # Not done
//!
//! `-n` (niceness) is applied only by the `runcom` command (`-n`), since it
//! cannot be undone in process; `runcom -P` prints `Runcom PID: <pid>` to
//! standard error where the script's `printPID(True)` printed `Python PID:`
//! (processchunks finds the job's PID after `PID:`); the Windows branches of
//! `vmstopy` are not exercised.  Python statements written into
//! a command file with `>` run only if they use the statements `vmstopy`
//! itself emits; anything else fails before the first command runs.
//!
//! Environment changes (`$setenv`, `PIP_PRINT_ENTRIES`, `-e`, `-f`, `-b`,
//! `$IMOD_DIR/bin` on the path) are made to this process's environment for
//! the run and undone afterwards, as a child process's would vanish.  The
//! runner redirects descriptors 1 and 2 while a program runs in process, so
//! it must not run concurrently with anything else writing to them.

use crate::imod::commands;
use crate::imod::pysrc::imodpy;
use crate::imod::pysrc::vmstopy::{self, VmstopyOptions};
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Options for [`run_com_file`]: those of `vmstopy` that affect the script,
/// and where standard error goes.
#[derive(Clone, Debug)]
pub struct ComOptions {
    /// The log file; `None` is the command file's name with its extension
    /// replaced by `.log`, as `submfg` names it.
    pub log: Option<PathBuf>,
    /// `-c`, `-k`, `-t`, `-e`, `-f`, `-b` of `vmstopy`.
    pub vmstopy: VmstopyOptions,
    /// Whether programs' standard error goes to the log, as the `vmstopy`
    /// script sends it (default); false leaves it on this process's standard
    /// error, as `vmstocsh | tcsh` does.
    pub stderr_to_log: bool,
}

impl Default for ComOptions {
    fn default() -> Self {
        ComOptions {
            log: None,
            vmstopy: VmstopyOptions::default(),
            stderr_to_log: true,
        }
    }
}

/// How one command of the file was run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunMode {
    /// Through the command table, in this process.
    InProcess,
    /// This crate's own binary, as a child process.
    OwnBinary,
    /// `/bin/sh -c`.
    Shell,
    /// A nested command file (`$vmstocsh x.log < x.com | csh -ef`).
    Nested,
}

/// One command the run executed.
#[derive(Clone, Debug)]
pub struct ComStep {
    /// The command line after substitution.
    pub command: String,
    /// Its exit status (for a nested file, that run's status).
    pub status: i32,
    pub mode: RunMode,
}

/// What [`run_com_file`] did.
#[derive(Clone, Debug)]
pub struct ComResult {
    /// The exit status the generated script would have had.
    pub status: i32,
    /// Every command run, in order.
    pub steps: Vec<ComStep>,
    /// A failure before anything ran: the `vmstopy` conversion error
    /// (`ERROR: vmstopy - ...`) or the log file could not be opened.
    pub error: Option<String>,
    /// The log file used.
    pub log: PathBuf,
}

/// Runs the command file `path` in this process; see the module comment.
pub fn run_com_file(path: &Path, options: &ComOptions) -> ComResult {
    let log = options
        .log
        .clone()
        .unwrap_or_else(|| path.with_extension("log"));
    let fail = |message: String, log: PathBuf| ComResult {
        status: 1,
        steps: Vec::new(),
        error: Some(message),
        log,
    };
    // `vmstopy` opens the command file (`vmstopy:162-166`) and converts it
    let com = match File::open(path) {
        Ok(file) if !path.is_dir() => file,
        _ => {
            return fail(
                format!("ERROR: vmstopy - Opening command file {}", path.display()),
                log,
            );
        }
    };
    let mut script: Vec<u8> = Vec::new();
    if let Err(message) =
        vmstopy::convert(com, &log.to_string_lossy(), &options.vmstopy, &mut script)
    {
        return fail(format!("ERROR: vmstopy - {message}"), log);
    }
    run_script(&String::from_utf8_lossy(&script), &log, options)
}

/// Runs the command file `com` with its log in `log`, as the Python scripts'
/// `runcmd('vmstopy -x -q ' + com + ' ' + log)` did, and returns what that
/// `runcmd` returned: `Ok` on success, otherwise an `ImodpyError` whose error
/// strings (`getErrStrings`) are the runner's pre-run error, if any, and
/// `<com>: exited with status <n>`.  `vmstopy -x -q` printed nothing and
/// exited 1 on any failure; the status here is the script's own.
pub fn run_com_as_command(com: &str, log: &str) -> Result<(), imodpy::ImodpyError> {
    let options = ComOptions {
        log: Some(PathBuf::from(log)),
        ..ComOptions::default()
    };
    let result = run_com_file(Path::new(com), &options);
    if result.status == 0 {
        return Ok(());
    }
    let mut errors: Vec<String> = result.error.into_iter().collect();
    errors.push(format!("{com}: exited with status {}", result.status));
    Err(imodpy::set_run_error(errors, result.status))
}

/// Runs a script produced by [`vmstopy::convert`] for the log `log`
/// (`vmstopy -x` executes the script it wrote with this).
pub fn run_script(script: &str, log: &Path, options: &ComOptions) -> ComResult {
    let fail = |message: String| ComResult {
        status: 1,
        steps: Vec::new(),
        error: Some(message),
        log: log.to_path_buf(),
    };
    // The boilerplate up to `printPID(True)` is fixed text for the options;
    // it is carried out directly below rather than interpreted.
    let body = match script.find("\nprintPID(True)\n") {
        Some(at) => &script[at + "\nprintPID(True)\n".len()..],
        None => return fail("ERROR: runcom - Unrecognized script from vmstopy".to_owned()),
    };
    let program = match parse(body) {
        Ok(program) => program,
        Err(message) => {
            return fail(format!(
                "ERROR: runcom - Command file uses a construct the runner cannot execute: {message}"
            ));
        }
    };

    // The environment is this run's, as a child process's would be.
    let saved_env: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    let set_env = |name: &str, value: &OsString| unsafe { std::env::set_var(name, value) };
    // addIMODbinIgnoreSIGHUP (the signal is not changed in process)
    if let Some(imod_dir) = std::env::var_os("IMOD_DIR") {
        let mut path = OsString::from(Path::new(&imod_dir).join("bin"));
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        set_env("PATH", &path);
    }
    set_env("PIP_PRINT_ENTRIES", &OsString::from("1"));
    for addto in &options.vmstopy.add_to_front {
        let mut path = OsString::from(addto);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        set_env("PATH", &path);
    }
    for addto in &options.vmstopy.add_to_back {
        let mut path = std::env::var_os("PATH").unwrap_or_default();
        path.push(":");
        path.push(addto);
        set_env("PATH", &path);
    }
    for (var, val) in &options.vmstopy.envars {
        set_env(var, &OsString::from(val));
    }

    // Back up and open log file
    let log_name = log.to_string_lossy().into_owned();
    imodpy::make_backup_file(&log_name);
    let result = match File::create(log) {
        Err(_) => {
            imodpy::prnstr(
                &format!("ERROR: Cannot open log file {log_name} for writing"),
                "\n",
                false,
            );
            let _ = std::io::stdout().flush();
            fail(format!(
                "ERROR: Cannot open log file {log_name} for writing"
            ))
        }
        Ok(file) => {
            let mut interp = Interp {
                log: file,
                chunk: options.vmstopy.chunk,
                stderr_to_log: options.stderr_to_log,
                options: options.clone(),
                globals: HashMap::from([("log".to_owned(), Value::Log)]),
                functions: HashMap::new(),
                err_strings: Vec::new(),
                current_exc: None,
                nested: HashMap::new(),
                steps: Vec::new(),
            };
            let status = match interp.exec_block(&program) {
                Ok(()) => 0,
                Err(Signal::Exit(status)) => status,
                // An exception outside every `try` ends the interpreter with
                // a traceback and status 1
                Err(Signal::Raise(exc)) => {
                    let _ = writeln!(std::io::stderr(), "{}: {}", exc.kind, exc.message);
                    1
                }
            };
            ComResult {
                status,
                steps: interp.steps,
                error: None,
                log: log.to_path_buf(),
            }
        }
    };

    // Restore the environment
    let now: Vec<OsString> = std::env::vars_os().map(|(name, _)| name).collect();
    for name in now {
        if !saved_env.iter().any(|(saved, _)| *saved == name) {
            unsafe { std::env::remove_var(&name) };
        }
    }
    for (name, value) in &saved_env {
        if std::env::var_os(name).as_ref() != Some(value) {
            unsafe { std::env::set_var(name, value) };
        }
    }
    result
}

// ---------------------------------------------------------------------------
// The script language: the subset of Python that `vmstopy` writes.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Name(String),
    Num(String),
    Str(String),
    Op(String),
}

/// One logical line: its indentation and tokens.
struct Line {
    indent: usize,
    toks: Vec<Tok>,
}

/// Python string-literal escapes (`\\`, `\'`, `\"`, `\n`, `\t`, `\r`, `\a`,
/// `\b`, `\f`, `\v`, octal, `\x`, `\u`, `\U`, backslash-newline); an unknown
/// escape keeps its backslash, as Python does.
fn decode_escapes(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' || i + 1 >= chars.len() {
            out.push(c);
            i += 1;
            continue;
        }
        let e = chars[i + 1];
        i += 2;
        match e {
            '\n' => {}
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'a' => out.push('\u{7}'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'v' => out.push('\u{b}'),
            '0'..='7' => {
                let mut value = e.to_digit(8).unwrap();
                let mut n = 1;
                while n < 3 && i < chars.len() && chars[i].is_digit(8) {
                    value = value * 8 + chars[i].to_digit(8).unwrap();
                    i += 1;
                    n += 1;
                }
                out.extend(char::from_u32(value));
            }
            'x' | 'u' | 'U' => {
                let len = match e {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let digits: String = chars[i..(i + len).min(chars.len())].iter().collect();
                match u32::from_str_radix(&digits, 16)
                    .ok()
                    .and_then(char::from_u32)
                {
                    Some(ch) if digits.len() == len => {
                        out.push(ch);
                        i += len;
                    }
                    _ => {
                        out.push('\\');
                        out.push(e);
                    }
                }
            }
            _ => {
                out.push('\\');
                out.push(e);
            }
        }
    }
    out
}

/// Splits `text` into logical lines of tokens, as Python's tokenizer does
/// for this subset: comments and blank lines dropped, newlines inside
/// brackets and triple-quoted strings joined.
///
/// One deliberate difference: a triple-quoted string followed directly by
/// more of its quote character (`"""abc""""`, which `vmstopy` writes for a
/// standard input line ending in `"`) is a `SyntaxError` in Python, so the
/// whole native script fails to run; here the extra quotes belong to the
/// string, which is what the line meant.
fn tokenize(text: &str) -> Result<Vec<Line>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut lines = Vec::new();
    let mut i = 0;
    let n = chars.len();
    while i < n {
        // start of a logical line: indentation
        let mut indent = 0;
        while i < n && chars[i] == ' ' {
            indent += 1;
            i += 1;
        }
        let mut toks: Vec<Tok> = Vec::new();
        let mut depth = 0i32;
        loop {
            if i >= n {
                break;
            }
            let c = chars[i];
            if c == '\n' {
                i += 1;
                if depth > 0 {
                    continue;
                }
                break;
            }
            if c == ' ' || c == '\t' || c == '\r' {
                i += 1;
                continue;
            }
            if c == '#' {
                while i < n && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            if c == '\\' && i + 1 < n && chars[i + 1] == '\n' {
                i += 2;
                continue;
            }
            if c == '"' || c == '\'' {
                let triple = i + 2 < n && chars[i + 1] == c && chars[i + 2] == c;
                let mut raw = String::new();
                if triple {
                    i += 3;
                    loop {
                        if i >= n {
                            return Err("unterminated string".to_owned());
                        }
                        if chars[i] == '\\' && i + 1 < n {
                            raw.push(chars[i]);
                            raw.push(chars[i + 1]);
                            i += 2;
                            continue;
                        }
                        if i + 2 < n && chars[i] == c && chars[i + 1] == c && chars[i + 2] == c {
                            // a longer run of quotes closes at its end
                            let mut run = 3;
                            while i + run < n && chars[i + run] == c {
                                run += 1;
                            }
                            for _ in 3..run {
                                raw.push(c);
                            }
                            i += run;
                            break;
                        }
                        raw.push(chars[i]);
                        i += 1;
                    }
                } else {
                    i += 1;
                    loop {
                        if i >= n || chars[i] == '\n' {
                            return Err("unterminated string".to_owned());
                        }
                        if chars[i] == '\\' && i + 1 < n {
                            raw.push(chars[i]);
                            raw.push(chars[i + 1]);
                            i += 2;
                            continue;
                        }
                        if chars[i] == c {
                            i += 1;
                            break;
                        }
                        raw.push(chars[i]);
                        i += 1;
                    }
                }
                toks.push(Tok::Str(decode_escapes(&raw)));
                continue;
            }
            if c.is_alphabetic() || c == '_' {
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                toks.push(Tok::Name(chars[start..i].iter().collect()));
                continue;
            }
            if c.is_ascii_digit() || (c == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()) {
                let start = i;
                while i < n
                    && (chars[i].is_ascii_alphanumeric()
                        || chars[i] == '.'
                        || chars[i] == '_'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && matches!(chars[i - 1], 'e' | 'E')
                            && !chars[start..i].iter().any(|d| matches!(d, 'x' | 'X'))))
                {
                    i += 1;
                }
                toks.push(Tok::Num(chars[start..i].iter().collect()));
                continue;
            }
            let two: String = chars[i..(i + 2).min(n)].iter().collect();
            if two == "==" || two == "!=" {
                toks.push(Tok::Op(two));
                i += 2;
                continue;
            }
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
            if "()[]{},:.=+-".contains(c) {
                toks.push(Tok::Op(c.to_string()));
                i += 1;
                continue;
            }
            return Err(format!("unexpected character {c:?}"));
        }
        if !toks.is_empty() {
            lines.push(Line { indent, toks });
        }
    }
    Ok(lines)
}

#[derive(Clone, Debug)]
enum Expr {
    Str(String),
    Num(Value),
    Name(String),
    Attr(Box<Expr>, String),
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>),
    Index(Box<Expr>, Box<Expr>),
    List(Vec<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Compare(bool, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
enum Stmt {
    Assign(Expr, Expr),
    Expr(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    Try(Vec<Stmt>, Vec<(Option<String>, Vec<Stmt>)>),
    Def(String, Vec<Stmt>),
    Nothing,
}

/// A Python value of the subset.
#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    /// `sys.exc_info()`
    ExcInfo,
    /// `sys.exc_info()[1]`
    Exc(PyExc),
    /// The script's `log` file object
    Log,
}

#[derive(Clone, Debug)]
struct PyExc {
    kind: &'static str,
    message: String,
}

enum Signal {
    Exit(i32),
    Raise(PyExc),
}

fn parse(text: &str) -> Result<Vec<Stmt>, String> {
    let lines = tokenize(text)?;
    let mut pos = 0;
    let block = parse_block(&lines, &mut pos, 0)?;
    if pos != lines.len() {
        return Err("unexpected indentation".to_owned());
    }
    Ok(block)
}

/// The statements at exactly `indent`, from `pos` on.
fn parse_block(lines: &[Line], pos: &mut usize, indent: usize) -> Result<Vec<Stmt>, String> {
    let mut stmts = Vec::new();
    while *pos < lines.len() && lines[*pos].indent >= indent {
        if lines[*pos].indent > indent {
            return Err("unexpected indentation".to_owned());
        }
        let toks = &lines[*pos].toks;
        *pos += 1;
        let first = &toks[0];
        let is = |name: &str| *first == Tok::Name(name.to_owned());
        let body = |pos: &mut usize| -> Result<Vec<Stmt>, String> {
            if *pos >= lines.len() || lines[*pos].indent <= indent {
                return Err("expected an indented block".to_owned());
            }
            let inner = lines[*pos].indent;
            parse_block(lines, pos, inner)
        };
        let header_end = |toks: &[Tok]| -> Result<(), String> {
            if toks.last() != Some(&Tok::Op(":".to_owned())) {
                return Err("expected ':'".to_owned());
            }
            Ok(())
        };
        if is("import") || is("from") {
            stmts.push(Stmt::Nothing);
        } else if is("pass") {
            stmts.push(Stmt::Nothing);
        } else if is("def") {
            header_end(toks)?;
            let Some(Tok::Name(name)) = toks.get(1) else {
                return Err("bad def".to_owned());
            };
            // `closeExit` and `printErrorExit` are the runner's own (see
            // `Interp::call`); their bodies are not interpreted
            if matches!(name.as_str(), "closeExit" | "printErrorExit") {
                while *pos < lines.len() && lines[*pos].indent > indent {
                    *pos += 1;
                }
                stmts.push(Stmt::Nothing);
                continue;
            }
            if toks.get(2) != Some(&Tok::Op("(".to_owned()))
                || toks.get(3) != Some(&Tok::Op(")".to_owned()))
            {
                return Err(format!("def {name} with parameters"));
            }
            let stmts_body = body(pos)?;
            stmts.push(Stmt::Def(name.clone(), stmts_body));
        } else if is("if") {
            header_end(toks)?;
            let cond = parse_expr_all(&toks[1..toks.len() - 1])?;
            let then = body(pos)?;
            let mut otherwise = Vec::new();
            if *pos < lines.len()
                && lines[*pos].indent == indent
                && lines[*pos].toks[0] == Tok::Name("else".to_owned())
            {
                *pos += 1;
                otherwise = body(pos)?;
            }
            stmts.push(Stmt::If(cond, then, otherwise));
        } else if is("try") {
            if toks.len() != 2 {
                return Err("bad try".to_owned());
            }
            let tried = body(pos)?;
            let mut handlers = Vec::new();
            while *pos < lines.len()
                && lines[*pos].indent == indent
                && lines[*pos].toks[0] == Tok::Name("except".to_owned())
            {
                let head = &lines[*pos].toks;
                *pos += 1;
                header_end(head)?;
                let class = match &head[1..head.len() - 1] {
                    [] => None,
                    [Tok::Name(class)] => Some(class.clone()),
                    _ => return Err("bad except".to_owned()),
                };
                handlers.push((class, body(pos)?));
            }
            if handlers.is_empty() {
                return Err("try without except".to_owned());
            }
            stmts.push(Stmt::Try(tried, handlers));
        } else if let Some(eq) = {
            // an assignment's `=` is outside every bracket
            let mut depth = 0i32;
            toks.iter().position(|t| {
                if let Tok::Op(o) = t {
                    match o.as_str() {
                        "(" | "[" | "{" => depth += 1,
                        ")" | "]" | "}" => depth -= 1,
                        "=" => return depth == 0,
                        _ => {}
                    }
                }
                false
            })
        } {
            let target = parse_expr_all(&toks[..eq])?;
            let value = parse_expr_all(&toks[eq + 1..])?;
            if !matches!(target, Expr::Name(_) | Expr::Index(..)) {
                return Err("bad assignment".to_owned());
            }
            stmts.push(Stmt::Assign(target, value));
        } else {
            stmts.push(Stmt::Expr(parse_expr_all(toks)?));
        }
    }
    Ok(stmts)
}

fn parse_expr_all(toks: &[Tok]) -> Result<Expr, String> {
    let mut pos = 0;
    let expr = parse_expr(toks, &mut pos).map_err(|error| format!("{error} in {toks:?}"))?;
    if pos != toks.len() {
        return Err(format!("unexpected {:?}", toks[pos]));
    }
    Ok(expr)
}

fn parse_expr(toks: &[Tok], pos: &mut usize) -> Result<Expr, String> {
    if toks.get(*pos) == Some(&Tok::Name("not".to_owned())) {
        *pos += 1;
        return Ok(Expr::Not(Box::new(parse_expr(toks, pos)?)));
    }
    let left = parse_sum(toks, pos)?;
    match toks.get(*pos) {
        Some(Tok::Op(op)) if op == "==" || op == "!=" => {
            let equal = op == "==";
            *pos += 1;
            let right = parse_sum(toks, pos)?;
            Ok(Expr::Compare(equal, Box::new(left), Box::new(right)))
        }
        _ => Ok(left),
    }
}

fn parse_sum(toks: &[Tok], pos: &mut usize) -> Result<Expr, String> {
    let mut left = parse_postfix(toks, pos)?;
    while toks.get(*pos) == Some(&Tok::Op("+".to_owned())) {
        *pos += 1;
        let right = parse_postfix(toks, pos)?;
        left = Expr::Add(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_postfix(toks: &[Tok], pos: &mut usize) -> Result<Expr, String> {
    let op = |s: &str| Tok::Op(s.to_owned());
    let mut expr = match toks.get(*pos) {
        Some(Tok::Str(text)) => {
            let mut text = text.clone();
            *pos += 1;
            // adjacent literals concatenate
            while let Some(Tok::Str(more)) = toks.get(*pos) {
                text += more;
                *pos += 1;
            }
            Expr::Str(text)
        }
        Some(Tok::Num(text)) => {
            *pos += 1;
            Expr::Num(parse_number(text).ok_or(format!("bad number {text}"))?)
        }
        Some(Tok::Name(name)) => {
            *pos += 1;
            match name.as_str() {
                "None" => Expr::Num(Value::None),
                "True" => Expr::Num(Value::Bool(true)),
                "False" => Expr::Num(Value::Bool(false)),
                _ => Expr::Name(name.clone()),
            }
        }
        Some(Tok::Op(o)) if o == "[" => {
            *pos += 1;
            let mut items = Vec::new();
            while toks.get(*pos) != Some(&op("]")) {
                items.push(parse_expr(toks, pos)?);
                if toks.get(*pos) == Some(&op(",")) {
                    *pos += 1;
                } else if toks.get(*pos) != Some(&op("]")) {
                    return Err("bad list".to_owned());
                }
            }
            *pos += 1;
            Expr::List(items)
        }
        Some(Tok::Op(o)) if o == "-" => {
            *pos += 1;
            match parse_postfix(toks, pos)? {
                Expr::Num(Value::Int(i)) => Expr::Num(Value::Int(-i)),
                Expr::Num(Value::Float(f)) => Expr::Num(Value::Float(-f)),
                _ => return Err("unsupported unary minus".to_owned()),
            }
        }
        Some(Tok::Op(o)) if o == "(" => {
            *pos += 1;
            let inner = parse_expr(toks, pos)?;
            if toks.get(*pos) != Some(&op(")")) {
                return Err("expected ')'".to_owned());
            }
            *pos += 1;
            inner
        }
        other => return Err(format!("unexpected {other:?}")),
    };
    loop {
        match toks.get(*pos) {
            Some(Tok::Op(o)) if o == "(" => {
                *pos += 1;
                let mut args = Vec::new();
                let mut keywords = Vec::new();
                while toks.get(*pos) != Some(&op(")")) {
                    if let (Some(Tok::Name(key)), Some(Tok::Op(eq))) =
                        (toks.get(*pos), toks.get(*pos + 1))
                        && eq == "="
                    {
                        *pos += 2;
                        keywords.push((key.clone(), parse_expr(toks, pos)?));
                    } else {
                        args.push(parse_expr(toks, pos)?);
                    }
                    if toks.get(*pos) == Some(&op(",")) {
                        *pos += 1;
                    } else if toks.get(*pos) != Some(&op(")")) {
                        return Err("bad call".to_owned());
                    }
                }
                *pos += 1;
                expr = Expr::Call(Box::new(expr), args, keywords);
            }
            Some(Tok::Op(o)) if o == "." => {
                *pos += 1;
                let Some(Tok::Name(name)) = toks.get(*pos) else {
                    return Err("bad attribute".to_owned());
                };
                *pos += 1;
                expr = Expr::Attr(Box::new(expr), name.clone());
            }
            Some(Tok::Op(o)) if o == "[" => {
                *pos += 1;
                let index = parse_expr(toks, pos)?;
                if toks.get(*pos) != Some(&op("]")) {
                    return Err("expected ']'".to_owned());
                }
                *pos += 1;
                expr = Expr::Index(Box::new(expr), Box::new(index));
            }
            _ => return Ok(expr),
        }
    }
}

/// A Python numeric literal.
fn parse_number(text: &str) -> Option<Value> {
    let clean = text.replace('_', "");
    let lower = clean.to_ascii_lowercase();
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(digits) = lower.strip_prefix(prefix) {
            return i64::from_str_radix(digits, radix).ok().map(Value::Int);
        }
    }
    if clean.chars().all(|c| c.is_ascii_digit()) {
        return clean.parse::<i64>().ok().map(Value::Int);
    }
    clean.parse::<f64>().ok().map(Value::Float)
}

impl Value {
    /// Python `str(value)`.
    fn to_str(&self) -> String {
        match self {
            Value::None => "None".to_owned(),
            Value::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => imodpy::py_str_float(*f),
            Value::Str(s) => s.clone(),
            Value::List(items) => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|item| match item {
                        Value::Str(s) => format!("'{s}'"),
                        other => other.to_str(),
                    })
                    .collect();
                format!("[{}]", parts.join(", "))
            }
            Value::ExcInfo => "(exc_info)".to_owned(),
            Value::Exc(exc) => exc.message.clone(),
            Value::Log => "<log>".to_owned(),
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Value::None => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(items) => !items.is_empty(),
            Value::ExcInfo | Value::Exc(_) | Value::Log => true,
        }
    }
}

fn raise<T>(kind: &'static str, message: String) -> Result<T, Signal> {
    Err(Signal::Raise(PyExc { kind, message }))
}

/// The dotted name an attribute chain spells (`os.path.exists`).
fn dotted(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.clone()),
        Expr::Attr(base, name) => Some(format!("{}.{name}", dotted(base)?)),
        _ => None,
    }
}

/// `str(sys.exc_info()[1])` of an OSError.
fn os_error_text(error: &std::io::Error, name: &str) -> String {
    let text = error.to_string();
    match error.raw_os_error() {
        Some(errno) => format!(
            "[Errno {errno}] {}: '{name}'",
            text.strip_suffix(&format!(" (os error {errno})"))
                .unwrap_or(&text)
        ),
        None => text,
    }
}

struct Interp {
    log: File,
    chunk: bool,
    stderr_to_log: bool,
    options: ComOptions,
    globals: HashMap<String, Value>,
    functions: HashMap<String, Vec<Stmt>>,
    /// `imodpy.errStrings`, without line endings
    err_strings: Vec<String>,
    current_exc: Option<PyExc>,
    /// `X.com.py` -> (`X.com`, `X.log`) for a nested file converted by
    /// `runcmd("vmstopy X.com X.log X.com.py")`
    nested: HashMap<String, (String, String)>,
    steps: Vec<ComStep>,
}

impl Interp {
    /// The options of a nested conversion: the script runs
    /// `vmstopy [-t ]X.com X.log X.com.py` (`vmstopy:582`), so only `-t`
    /// carries over; `-e`/`-f`/`-b` reach it through the environment it
    /// inherits.
    fn nested_options(&self) -> VmstopyOptions {
        VmstopyOptions {
            test: self.options.vmstopy.test,
            ..VmstopyOptions::default()
        }
    }

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<(), Signal> {
        for stmt in stmts {
            self.exec(stmt)?;
        }
        Ok(())
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<(), Signal> {
        match stmt {
            Stmt::Nothing => Ok(()),
            Stmt::Def(name, body) => {
                self.functions.insert(name.clone(), body.clone());
                Ok(())
            }
            Stmt::Expr(expr) => self.eval(expr).map(|_| ()),
            Stmt::Assign(target, value) => {
                let value = self.eval(value)?;
                match target {
                    Expr::Name(name) => {
                        self.globals.insert(name.clone(), value);
                        Ok(())
                    }
                    Expr::Index(base, key) if dotted(base).as_deref() == Some("os.environ") => {
                        let key = self.eval(key)?;
                        let (Value::Str(key), Value::Str(value)) = (key, value) else {
                            return raise("TypeError", "str expected, not other type".to_owned());
                        };
                        unsafe { std::env::set_var(key, value) };
                        Ok(())
                    }
                    _ => raise("TypeError", "unsupported assignment".to_owned()),
                }
            }
            Stmt::If(cond, then, otherwise) => {
                if self.eval(cond)?.truthy() {
                    self.exec_block(then)
                } else {
                    self.exec_block(otherwise)
                }
            }
            Stmt::Try(body, handlers) => match self.exec_block(body) {
                Err(Signal::Raise(exc)) => {
                    for (class, handler) in handlers {
                        let matches = match class.as_deref() {
                            None | Some("Exception") => true,
                            Some(class) => class == exc.kind,
                        };
                        if matches {
                            let saved = self.current_exc.replace(exc);
                            let result = self.exec_block(handler);
                            self.current_exc = saved;
                            return result;
                        }
                    }
                    Err(Signal::Raise(exc))
                }
                other => other,
            },
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Value, Signal> {
        match expr {
            Expr::Str(text) => Ok(Value::Str(text.clone())),
            Expr::Num(value) => Ok(value.clone()),
            Expr::Name(name) => match self.globals.get(name) {
                Some(value) => Ok(value.clone()),
                None => raise("NameError", format!("name '{name}' is not defined")),
            },
            Expr::List(items) => {
                let mut values = Vec::new();
                for item in items {
                    values.push(self.eval(item)?);
                }
                Ok(Value::List(values))
            }
            Expr::Add(left, right) => {
                let left = self.eval(left)?;
                let right = self.eval(right)?;
                match (left, right) {
                    (Value::Str(a), Value::Str(b)) => Ok(Value::Str(a + &b)),
                    (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_add(b))),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                    (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                    (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                    (Value::List(mut a), Value::List(b)) => {
                        a.extend(b);
                        Ok(Value::List(a))
                    }
                    _ => raise("TypeError", "unsupported operand type(s) for +".to_owned()),
                }
            }
            Expr::Not(inner) => Ok(Value::Bool(!self.eval(inner)?.truthy())),
            Expr::Compare(equal, left, right) => {
                let left = self.eval(left)?;
                let right = self.eval(right)?;
                let same = match (&left, &right) {
                    (Value::None, Value::None) => true,
                    (Value::None, _) | (_, Value::None) => false,
                    (Value::Str(a), Value::Str(b)) => a == b,
                    _ => left.to_str() == right.to_str(),
                };
                Ok(Value::Bool(same == *equal))
            }
            Expr::Index(base, key) => {
                if dotted(base).as_deref() == Some("os.environ") {
                    let key = self.eval(key)?.to_str();
                    return match std::env::var(&key) {
                        Ok(value) => Ok(Value::Str(value)),
                        Err(_) => raise("KeyError", format!("'{key}'")),
                    };
                }
                let base = self.eval(base)?;
                let key = self.eval(key)?;
                match (base, key) {
                    (Value::ExcInfo, Value::Int(1)) => match &self.current_exc {
                        Some(exc) => Ok(Value::Exc(exc.clone())),
                        None => Ok(Value::None),
                    },
                    (Value::List(items), Value::Int(i)) => {
                        let len = items.len() as i64;
                        let at = if i < 0 { i + len } else { i };
                        match items.get(at as usize).filter(|_| at >= 0) {
                            Some(item) => Ok(item.clone()),
                            None => raise("IndexError", "list index out of range".to_owned()),
                        }
                    }
                    _ => raise("TypeError", "unsupported subscript".to_owned()),
                }
            }
            Expr::Attr(..) => match dotted(expr).as_deref() {
                Some("os.pathsep") => Ok(Value::Str(":".to_owned())),
                Some("os.sep") => Ok(Value::Str("/".to_owned())),
                _ => raise("AttributeError", "unsupported attribute".to_owned()),
            },
            Expr::Call(func, args, keywords) => {
                let Some(name) = dotted(func) else {
                    return raise("TypeError", "object is not callable".to_owned());
                };
                let mut values = Vec::new();
                for arg in args {
                    values.push(self.eval(arg)?);
                }
                let mut kw: Vec<(String, Expr)> = Vec::new();
                for (key, value) in keywords {
                    kw.push((key.clone(), value.clone()));
                }
                self.call(&name, values, &kw)
            }
        }
    }

    fn call(
        &mut self,
        name: &str,
        args: Vec<Value>,
        keywords: &[(String, Expr)],
    ) -> Result<Value, Signal> {
        let arg_str = |i: usize| -> String { args.get(i).map(Value::to_str).unwrap_or_default() };
        let missing = |count: usize| -> Result<Value, Signal> {
            raise(
                "TypeError",
                format!("{name}() missing {count} required positional argument"),
            )
        };
        match name {
            "str" => Ok(Value::Str(arg_str(0))),
            "sys.exc_info" => Ok(Value::ExcInfo),
            "os.getpid" => Ok(Value::Int(std::process::id() as i64)),
            "socket.gethostname" => {
                let mut buffer = [0u8; 256];
                unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) };
                let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
                Ok(Value::Str(
                    String::from_utf8_lossy(&buffer[..end]).into_owned(),
                ))
            }
            "os.path.exists" => {
                if args.is_empty() {
                    return missing(1);
                }
                Ok(Value::Bool(Path::new(&arg_str(0)).exists()))
            }
            "os.getenv" => Ok(std::env::var(arg_str(0))
                .map(Value::Str)
                .unwrap_or(Value::None)),
            "imodTempDir" => Ok(Value::Str(imodpy::imod_temp_dir())),
            "cygwinPath" => Ok(Value::Str(arg_str(0))),
            "makeBackupFile" => {
                imodpy::make_backup_file(&arg_str(0));
                Ok(Value::None)
            }
            "os.mkdir" => {
                use std::os::unix::fs::DirBuilderExt as _;
                let mode = match args.get(1) {
                    Some(Value::Int(mode)) => *mode as u32,
                    _ => 0o777,
                };
                let path = arg_str(0);
                match std::fs::DirBuilder::new().mode(mode).create(&path) {
                    Ok(()) => Ok(Value::None),
                    Err(error) => raise("OSError", os_error_text(&error, &path)),
                }
            }
            "shutil.rmtree" => {
                let path = arg_str(0);
                let ignore = args.get(1).is_some_and(Value::truthy);
                match std::fs::remove_dir_all(&path) {
                    Err(error) if !ignore => raise("OSError", os_error_text(&error, &path)),
                    _ => Ok(Value::None),
                }
            }
            "prnstr" => {
                let mut end = "\n".to_owned();
                let mut to_log = false;
                for (key, value) in keywords {
                    match key.as_str() {
                        "end" => end = self.eval(value)?.to_str(),
                        "file" => to_log = dotted(value).as_deref() == Some("log"),
                        "flush" => {}
                        _ => return raise("TypeError", format!("unexpected keyword {key}")),
                    }
                }
                let text = format!("{}{end}", arg_str(0));
                if to_log {
                    let _ = self.log.write_all(text.as_bytes());
                } else {
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                }
                Ok(Value::None)
            }
            "getErrStrings" => Ok(Value::List(
                self.err_strings
                    .iter()
                    .map(|l| Value::Str(format!("{l}\n")))
                    .collect(),
            )),
            "cleanupFiles" => {
                if let Some(Value::List(files)) = args.first() {
                    for file in files {
                        let _ = std::fs::remove_file(file.to_str());
                    }
                }
                Ok(Value::None)
            }
            "printErrorExit" => {
                // for l in getErrStrings(): prnstr('ERROR: ' + l, end='', file=log)
                for l in self.err_strings.clone() {
                    let _ = self.log.write_all(format!("ERROR: {l}\n").as_bytes());
                }
                if args.first().is_some_and(Value::truthy) {
                    return self.call("closeExit", vec![Value::Int(1)], &[]);
                }
                Ok(Value::None)
            }
            "closeExit" => {
                let Some(code) = args.first() else {
                    return missing(1);
                };
                let code = match code {
                    Value::Int(code) => *code as i32,
                    Value::None => 0,
                    Value::Bool(b) => *b as i32,
                    // sys.exit(non-integer) prints it and exits 1
                    _ => 1,
                };
                if code == 0 {
                    let _ = self.log.write_all(b"SUCCESSFULLY COMPLETED\n");
                    if self.chunk {
                        let _ = self.log.write_all(b"CHUNK DONE\n");
                    }
                }
                let _ = self.log.flush();
                Err(Signal::Exit(code))
            }
            "runcmd" => self.runcmd(args, keywords),
            _ => match self.functions.get(name).cloned() {
                Some(body) => {
                    self.exec_block(&body)?;
                    Ok(Value::None)
                }
                None => raise("NameError", format!("name '{name}' is not defined")),
            },
        }
    }

    /// `runcmd(command, input, log, 'stdout')` (`imodpy.py:176`) and the
    /// nested-conversion call `runcmd("vmstopy X.com X.log X.com.py")`.
    fn runcmd(&mut self, args: Vec<Value>, keywords: &[(String, Expr)]) -> Result<Value, Signal> {
        if !keywords.is_empty() {
            return raise("TypeError", "runcmd keywords are not supported".to_owned());
        }
        let command = match args.first() {
            Some(Value::Str(command)) => command.clone(),
            _ => return raise("TypeError", "runcmd needs a command string".to_owned()),
        };

        // `runcmd("vmstopy [-t ]X.com X.log X.com.py")`: the nested file is
        // converted now, as the native call would, and run when the script
        // runs `python -u X.com.py`.
        if args.len() == 1 {
            let words: Vec<&str> = command.split_whitespace().collect();
            let words: Vec<&str> = words.into_iter().filter(|w| *w != "-t").collect();
            if words.len() == 4 && words[0] == "vmstopy" {
                let (com, log, py) = (words[1], words[2], words[3]);
                let file = match File::open(com) {
                    Ok(file) if !Path::new(com).is_dir() => file,
                    _ => {
                        self.err_strings = vec![
                            format!("ERROR: vmstopy - Opening command file {com}"),
                            format!("{command}: exited with status 1"),
                        ];
                        return raise("ImodpyError", String::new());
                    }
                };
                let mut sink: Vec<u8> = Vec::new();
                if let Err(message) = vmstopy::convert(file, log, &self.nested_options(), &mut sink)
                {
                    self.err_strings = vec![
                        format!("ERROR: vmstopy - {message}"),
                        format!("{command}: exited with status 1"),
                    ];
                    return raise("ImodpyError", String::new());
                }
                self.nested
                    .insert(py.to_owned(), (com.to_owned(), log.to_owned()));
                return Ok(Value::None);
            }
            return raise(
                "TypeError",
                format!("runcmd with collected output is not supported: {command}"),
            );
        }
        let input: Vec<String> = match args.get(1) {
            Some(Value::List(items)) => items.iter().map(Value::to_str).collect(),
            // `input = '[]'` of the nested form: a non-empty string
            Some(Value::Str(text)) if !text.is_empty() => vec![text.clone()],
            _ => Vec::new(),
        };
        if args.len() != 4
            || !matches!(args[2], Value::Log)
            || !matches!(&args[3], Value::Str(to) if to == "stdout")
        {
            return raise(
                "TypeError",
                "runcmd needs (command, input, log, 'stdout')".to_owned(),
            );
        }

        if std::env::var("RUNCMD_VERBOSE").ok().as_deref() == Some("1") {
            println!("+++++++++++++++++++++++++");
            println!("   runcmd running command:");
            println!("{command}");
            if !input.is_empty() {
                println!("   With input:");
                for l in &input {
                    println!("{l}");
                }
            }
            let _ = std::io::stdout().flush();
        }

        let mut stdin_text = String::new();
        for l in &input {
            stdin_text.push_str(l);
            stdin_text.push('\n');
        }

        let (status, mode) = if let Some(rest) = command.strip_prefix("python -u ")
            && let Some((com, log)) = self.nested.get(rest).cloned()
        {
            let options = ComOptions {
                log: Some(PathBuf::from(log)),
                vmstopy: self.nested_options(),
                stderr_to_log: self.stderr_to_log,
            };
            let result = run_com_file(Path::new(&com), &options);
            if let Some(error) = &result.error {
                let _ = self.log.write_all(format!("{error}\n").as_bytes());
            }
            let status = result.status;
            self.steps.extend(result.steps);
            (status, RunMode::Nested)
        } else {
            self.run_program(&command, stdin_text.as_bytes())
        };
        self.steps.push(ComStep {
            command: command.clone(),
            status,
            mode,
        });
        if status != 0 {
            self.err_strings = vec![format!("{command}: exited with status {status}")];
            return raise("ImodpyError", String::new());
        }
        Ok(Value::None)
    }

    /// Runs one command line with `input` on its standard input and its
    /// output in the log; returns the exit status as `runcmd` sees it
    /// (minus the signal number for a killed child).
    fn run_program(&mut self, command: &str, input: &[u8]) -> (i32, RunMode) {
        let _ = self.log.flush();
        if let Some((words, globbed)) = shell_words(command)
            && !words.is_empty()
        {
            let name = words[0].as_str();
            let own_glob = name == "b3dremove"
                && words[1..]
                    .iter()
                    .take_while(|w| w.starts_with('-'))
                    .any(|w| w == "-g");
            if name == "sync" && words.len() == 1 {
                unsafe { libc::sync() };
                return (0, RunMode::InProcess);
            }
            if let Some(entry) = commands::find(name)
                && (!globbed || own_glob)
            {
                let in_process = entry.in_process || name == "b3dcopy" || name == "b3dremove";
                let mut argv: Vec<OsString> = Vec::with_capacity(words.len());
                argv.push(match std::env::current_exe() {
                    Ok(path) => path.with_file_name(name).into_os_string(),
                    Err(_) => OsString::from(name),
                });
                argv.extend(words.iter().skip(1).map(OsString::from));
                if in_process {
                    return (self.run_in_process(entry, argv, input), RunMode::InProcess);
                }
                // Another of our commands: our own binary, when this is it
                if let Ok(exe) = std::env::current_exe()
                    && exe.file_name().is_some_and(|base| base == "imod")
                {
                    let mut process = std::process::Command::new(&exe);
                    std::os::unix::process::CommandExt::arg0(&mut process, &argv[0]);
                    process.args(&argv[1..]);
                    return (self.spawn(process, input), RunMode::OwnBinary);
                }
            }
        }
        // `Popen(cmd, shell=True)` runs `/bin/sh -c cmd` with `argv[0]` "sh"
        let mut process = std::process::Command::new("/bin/sh");
        std::os::unix::process::CommandExt::arg0(&mut process, "sh");
        process.arg("-c").arg(command);
        (self.spawn(process, input), RunMode::Shell)
    }

    /// A child process with the log as its standard output (and error).
    fn spawn(&mut self, mut process: std::process::Command, input: &[u8]) -> i32 {
        use std::process::Stdio;
        process.stdin(Stdio::piped());
        match self.log.try_clone() {
            Ok(out) => {
                process.stdout(Stdio::from(out));
            }
            Err(_) => return 1,
        }
        if self.stderr_to_log {
            match self.log.try_clone() {
                Ok(err) => {
                    process.stderr(Stdio::from(err));
                }
                Err(_) => return 1,
            }
        }
        let mut child = match process.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = self
                    .log
                    .write_all(format!("command could not be started: {error}\n").as_bytes());
                return 127;
            }
        };
        let bytes = input.to_vec();
        let writer = child.stdin.take().map(|mut stdin| {
            std::thread::spawn(move || {
                let _ = stdin.write_all(&bytes);
            })
        });
        let status = child.wait();
        if let Some(writer) = writer {
            let _ = writer.join();
        }
        match status {
            Ok(status) => status.code().unwrap_or_else(|| {
                std::os::unix::process::ExitStatusExt::signal(&status).map_or(1, |s| -s)
            }),
            Err(_) => 1,
        }
    }

    /// Runs a command-table entry in this process with descriptors 1 (and
    /// 2) on the log.
    fn run_in_process(
        &mut self,
        entry: &'static commands::Command,
        argv: Vec<OsString>,
        input: &[u8],
    ) -> i32 {
        use std::os::fd::AsRawFd as _;
        unsafe { libc::fflush(std::ptr::null_mut()) };
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        let log_fd = self.log.as_raw_fd();
        let saved_out = unsafe { libc::dup(1) };
        let saved_err = if self.stderr_to_log {
            unsafe { libc::dup(2) }
        } else {
            -1
        };
        unsafe {
            libc::dup2(log_fd, 1);
            if saved_err >= 0 {
                libc::dup2(log_fd, 2);
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            commands::run_in_process(entry, argv, Some(input), false)
        }));
        unsafe { libc::fflush(std::ptr::null_mut()) };
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe {
            libc::dup2(saved_out, 1);
            libc::close(saved_out);
            if saved_err >= 0 {
                libc::dup2(saved_err, 2);
                libc::close(saved_err);
            }
        }
        match result {
            Ok(Ok((status, _))) => status,
            Ok(Err(error)) => {
                let _ = self
                    .log
                    .write_all(format!("command could not be run: {error}\n").as_bytes());
                1
            }
            // A panic: what the shell reports for a program that aborted
            Err(_) => 134,
        }
    }
}

/// Splits `command` into words as `sh -c` would when the line needs nothing
/// from a shell but word splitting and quote removal, and says whether an
/// unquoted glob character (`*`, `?`, `[`) occurred.  Any other unquoted
/// character that would make the shell do more -- a pipe, redirection,
/// command separator, background `&`, parameter or command substitution,
/// subshell, tilde or comment -- returns `None`, and the line is left to
/// the shell.  So does a `$` or backquote inside double quotes.
fn shell_words(command: &str) -> Option<(Vec<String>, bool)> {
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut globbed = false;
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
                let mut closed = false;
                for q in chars.by_ref() {
                    if q == '\'' {
                        closed = true;
                        break;
                    }
                    current.push(q);
                }
                if !closed {
                    return None;
                }
            }
            '"' => {
                in_word = true;
                let mut closed = false;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => {
                            closed = true;
                            break;
                        }
                        '$' | '`' => return None,
                        '\\' if matches!(chars.peek(), Some('"' | '\\' | '$' | '`' | '\n')) => {
                            let next = chars.next().unwrap();
                            if next != '\n' {
                                current.push(next);
                            }
                        }
                        _ => current.push(q),
                    }
                }
                if !closed {
                    return None;
                }
            }
            '\\' => {
                in_word = true;
                match chars.next() {
                    Some('\n') => {}
                    Some(q) => current.push(q),
                    None => return None,
                }
            }
            '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' => return None,
            '*' | '?' | '[' => {
                globbed = true;
                in_word = true;
                current.push(c);
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
    Some((words, globbed))
}

/// Rust-only command `runcom`: `runcom [options] comfile [logfile]` runs a
/// command file with [`run_com_file`] and exits with its status.  The
/// options are `vmstopy`'s `-c`, `-k`, `-t`, `-e VAR[=val]`, `-f dir`,
/// `-b dir`, and `-s`, which leaves standard error off the log.
pub fn runcom() {
    let argv: Vec<String> = crate::imod::libcfshr::b3dutil::program_args_os()
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let usage = || {
        print!(
            "Usage: runcom [options] comfile [logfile]\n  \
             Runs an IMOD command file in process (instead of vmstocsh | tcsh or vmstopy)\n  \
             The log file defaults to the command file's root with .log\n  \
             Options:\n    -c will add output of CHUNK DONE at end\n    \
             -e VAR=val or -e VAR will set an environment variable\n    \
             -f dir will put dir on the front of the path\n    \
             -b dir will put dir on the back of the path\n    \
             -k will keep backslashes instead of converting to forward slashes\n    \
             -t will prefix every command with echo2\n    \
             -n #  will set niceness of job to #\n    \
             -P will print the process ID to standard error, as the vmstopy\n       \
             script does (processchunks reads it to kill the job)\n    \
             -s will leave standard error of programs out of the log\n    \
             -S will run the script vmstopy wrote, read from standard input,\n       \
             instead of a command file (eTomo's `python -u` step)\n"
        );
        let _ = std::io::stdout().flush();
    };
    let mut options = ComOptions::default();
    let mut script_on_stdin = false;
    let mut ind = 1;
    let fail = |message: &str| -> ! {
        print!("ERROR: runcom - {message}\n");
        let _ = std::io::stdout().flush();
        crate::imod::libcfshr::b3dutil::exit(1)
    };
    while ind < argv.len() && argv[ind].starts_with('-') {
        let option = argv[ind].as_str();
        let mut value = || -> String {
            ind += 1;
            match argv.get(ind) {
                Some(value) => value.clone(),
                None => fail(&format!("Option {option} needs a value")),
            }
        };
        match option {
            "-h" => {
                usage();
                crate::imod::libcfshr::b3dutil::exit(0)
            }
            "-c" => options.vmstopy.chunk = true,
            "-k" => options.vmstopy.keep_backslash = true,
            "-t" => options.vmstopy.test = true,
            "-s" => options.stderr_to_log = false,
            "-S" => script_on_stdin = true,
            // `imodNice(n)` in the vmstopy script: this process is the job
            "-n" => {
                let text = value();
                match imodpy::py_int(&text).and_then(|nice| i32::try_from(nice).ok()) {
                    Some(nice) => unsafe {
                        libc::nice(nice);
                    },
                    None => fail("Converting \"nice\" value to integer"),
                }
            }
            // `printPID(True)` in the vmstopy script
            "-P" => {
                let _ = write!(std::io::stderr(), "Runcom PID: {}\n", std::process::id());
                let _ = std::io::stderr().flush();
            }
            "-e" => {
                let var = value();
                let (var, val) = match var.split_once('=') {
                    Some((var, val)) => (var.to_owned(), val.to_owned()),
                    None => (var, String::new()),
                };
                options.vmstopy.envars.push((var, val));
            }
            "-f" => {
                let dir = value();
                options.vmstopy.add_to_front.push(dir);
            }
            "-b" => {
                let dir = value();
                options.vmstopy.add_to_back.push(dir);
            }
            _ => fail(&format!("Unrecognized argument {option}")),
        }
        ind += 1;
    }
    if script_on_stdin {
        // The script `ComScriptProcess.execPython` pipes into `python -u`: the
        // log is the one it backs up and opens (`makeBackupFile('<log>')`).
        let mut script = String::new();
        let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut script);
        let log = script
            .lines()
            .find_map(|line| {
                line.strip_prefix("makeBackupFile('")
                    .and_then(|rest| rest.strip_suffix("')"))
            })
            .map(PathBuf::from);
        let Some(log) = log else {
            fail("No log file in the script on standard input")
        };
        let result = run_script(&script, &log, &options);
        if let Some(error) = &result.error {
            print!("{error}\n");
            let _ = std::io::stdout().flush();
        }
        crate::imod::libcfshr::b3dutil::exit(result.status)
    }
    if ind >= argv.len() || argv.len() - ind > 2 {
        usage();
        crate::imod::libcfshr::b3dutil::exit(1)
    }
    if let Some(log) = argv.get(ind + 1) {
        options.log = Some(PathBuf::from(log));
    }
    let result = run_com_file(Path::new(&argv[ind]), &options);
    if let Some(error) = &result.error {
        print!("{error}\n");
        let _ = std::io::stdout().flush();
    }
    crate::imod::libcfshr::b3dutil::exit(result.status)
}
