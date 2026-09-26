//! Translation of `IMOD/pysrc/vmstopy`.
//!
//! Translates a "command" file into a Python script with logging.  The
//! script's functions are [`quote_substitute`] (`quoteSubstitute`) and the
//! usage printer; its top level is [`vmstopy`], translated statement by
//! statement, except that the part of the top level that reads the command
//! file and writes the script (`vmstopy:193-627`) is the separate function
//! [`convert`], so that the in-process command file runner
//! ([`crate::imod::comrun`]) parses a command file with exactly this code.
//! A `closeErrorExit` call in that part becomes the `Err` return of
//! [`convert`], which [`vmstopy`] hands to [`close_error_exit`].
//!
//! Upstream defects fixed here (`BUGS.md`, "vmstopy"):
//!
//! * the backup-command branch (`vmstopy:474-477`) refers to `backupmatch`,
//!   a name that does not exist (the pattern is `backupMatch`), so any
//!   command file with `$if (-e f) \mv f f~` made vmstopy die with a
//!   `NameError`; the pattern the code evidently meant is used, and the
//!   file name written into `makeBackupFile(...)` is quoted, which that
//!   branch (never having run) also lacked;
//! * `os.mkdir("dir", 0766)` (`vmstopy:497`) is a Python 2 octal literal and
//!   a `SyntaxError` under Python 3, so every script converted from a file
//!   with `$mkdir` could not run at all; `0o766` is written instead;
//! * `line.split()[0]` (`vmstopy:549`) raises `IndexError` on a command line
//!   that is empty after the `$`; the empty command is kept instead.

use super::imodpy::prnstr;
use regex::Regex;
use std::ffi::OsString;
use std::io::Write;
use std::path::Path;

/// The script's module-level switches that `quoteSubstitute` reads, and the
/// options [`convert`] needs from the command line.
#[derive(Clone, Debug, Default)]
pub struct VmstopyOptions {
    /// `-c`: add output of CHUNK DONE at end
    pub chunk: bool,
    /// `-k`: keep backslashes instead of converting to forward slashes
    pub keep_backslash: bool,
    /// `-t`: prefix every command with `echo2 `
    pub test: bool,
    /// `-e VAR=val` or `-e VAR`, in order
    pub envars: Vec<(String, String)>,
    /// `-f dir`
    pub add_to_front: Vec<String>,
    /// `-b dir`
    pub add_to_back: Vec<String>,
    /// `-n #`
    pub nice_val: Option<i64>,
}

/// Module globals `quoteSubstitute` reads (`anyset`, `needPID`,
/// `keepBackslash`).
struct QuoteState {
    anyset: bool,
    need_pid: bool,
    keep_backslash: bool,
}

/// `quoteSubstitute` (`vmstopy:13-42`): enclose a line in quotes and change
/// variables.
fn quote_substitute(lin: &str, state: &QuoteState) -> String {
    let mut lin = lin.to_owned();
    if state.need_pid {
        lin = lin.replace("$$", "\"\"\" + str(os.getpid()) + \"\"\"");
    }
    if state.anyset {
        lin = lin.replace("\\$", "%");
        lin = lin.replace('$', "%");
    }

    // Take care of replacing \ with / as long as it is not escaping a "
    // (`lin.find('\\')` is -1, true, when there is none, and 0, false, only
    // when the line starts with a backslash)
    if !state.keep_backslash && lin.find('\\') != Some(0) {
        let backslash = Regex::new(r#"\\([^"])"#).unwrap();
        lin = backslash.replace_all(&lin, "/${1}").into_owned();
        if lin.ends_with('\\') {
            lin = format!("{}/", &lin[..lin.len() - 1]);
        }
    }

    // Enclose in quotes
    lin = format!("\"\"\"{lin}\"\"\"");
    let envar_match = Regex::new(r"%\{([A-Z0-9_]+)\}").unwrap();
    let regvar_match = Regex::new(r".*%(\w+)").unwrap();
    let regvar_sub = Regex::new(r"%(\w+)").unwrap();
    let mut indv = lin.find('%');
    while indv.is_some() {
        let envar = envar_match.is_match(&lin);
        let regvar = regvar_match.is_match(&lin);
        if envar {
            lin = envar_match
                .replacen(&lin, 1, "\"\"\" + os.environ[\"${1}\"] + \"\"\"")
                .into_owned();
        } else if regvar {
            lin = regvar_sub
                .replacen(&lin, 1, "\"\"\" + str(${1}) + \"\"\"")
                .into_owned();
        } else {
            break;
        }
        indv = lin.find('%');
    }

    lin
}

/// `closeErrorExit` (`vmstopy:46-52`): print error message, close file and
/// exit.  `out` is the script's output (closing it is dropping it) and
/// `outfile` its name; returns the status of its `sys.exit`.
fn close_error_exit(
    message: &str,
    mut out: Box<dyn Write>,
    usetemp: bool,
    outfile: Option<&str>,
) -> i32 {
    let _ = out.flush();
    prnstr(&format!("ERROR: vmstopy - {message}"), "\n", false);
    drop(out);
    if usetemp && let Some(name) = outfile {
        let _ = std::fs::remove_file(name);
    }
    let _ = std::io::stdout().flush();
    1
}

/// `printUsage` (`vmstopy:55-70`).  Returns the status of its `sys.exit`.
fn print_usage() -> i32 {
    prnstr(
        "Usage: vmstopy [options] comfile logfile [pyscript]",
        "\n",
        false,
    );
    prnstr(
        "  Converts an IMOD command file to a python script",
        "\n",
        false,
    );
    prnstr(
        "  Outputs script on standard out if neither \"pyscript\" nor \"-x\" is given",
        "\n",
        false,
    );
    prnstr("  Options:", "\n", false);
    prnstr("    -x will execute the script", "\n", false);
    prnstr("    -q will suppress messages", "\n", false);
    prnstr("    -c will add output of CHUNK DONE at end", "\n", false);
    prnstr(
        "    -e VAR=val or -e VAR will set an environment variable",
        "\n",
        false,
    );
    prnstr(
        "    -f dir will put dir on the front of the path",
        "\n",
        false,
    );
    prnstr(
        "    -b dir will put dir on the back of the path",
        "\n",
        false,
    );
    prnstr(
        "    -k will keep backslashes instead of converting to forward slashes",
        "\n",
        false,
    );
    prnstr("    -n #  will set niceness of job to #", "\n", false);
    prnstr(
        "    -p #  will set permissions of output script to executable",
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    0
}

/// The script's top level (`vmstopy:1-654`).  Returns the status of its
/// `sys.exit`.
pub fn vmstopy(arguments: &[OsString]) -> i32 {
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let flush = || {
        let _ = std::io::stdout().flush();
    };

    // IMOD_DIR must be defined; `sys.path` gets its pylib (`vmstopy:78-87`)
    if std::env::var_os("IMOD_DIR").is_none() {
        print!("ERROR: vmstopy - IMOD_DIR is not defined!\n");
        flush();
        return 1;
    }

    // Process arguments
    let lenarg = argv.len();
    let mut argind = 1usize;
    let mut execute = false;
    let mut quiet = false;
    let mut set_perm = false;
    let mut options = VmstopyOptions::default();
    if lenarg < 2 || argv[1] == "-h" {
        return print_usage();
    }

    while lenarg - argind > 2 {
        let oarg = argv[argind].clone();
        if oarg.starts_with('-') {
            if oarg == "-h" {
                return print_usage();
            }
            if oarg == "-x" {
                execute = true;
            } else if oarg == "-p" {
                set_perm = true;
            } else if oarg == "-q" {
                quiet = true;
            } else if oarg == "-c" {
                options.chunk = true;
            } else if oarg == "-k" {
                options.keep_backslash = true;
            } else if oarg == "-t" {
                options.test = true;
            } else if oarg == "-e" {
                argind += 1;
                let mut var = argv[argind].clone();
                let mut val = String::new();
                if let Some(ind) = var.find('=') {
                    val = var[ind + 1..].to_owned();
                    var.truncate(ind);
                }
                options.envars.push((var, val));
            } else if oarg == "-f" {
                argind += 1;
                options.add_to_front.push(argv[argind].clone());
            } else if oarg == "-b" {
                argind += 1;
                options.add_to_back.push(argv[argind].clone());
            } else if oarg == "-n" {
                argind += 1;
                // int(): surrounding whitespace, a sign and underscores
                // between digits are accepted
                let text = argv[argind].trim();
                let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
                let valid = !digits.is_empty()
                    && digits.chars().all(|c| c.is_ascii_digit() || c == '_')
                    && !digits.starts_with('_')
                    && !digits.ends_with('_')
                    && !digits.contains("__");
                match text.replace('_', "").parse::<i64>() {
                    Ok(value) if valid => options.nice_val = Some(value),
                    _ => {
                        prnstr(
                            "ERROR: vmstopy - Converting \"nice\" value to integer",
                            "\n",
                            false,
                        );
                        flush();
                        return 1;
                    }
                }
            } else {
                prnstr(
                    &format!("ERROR: vmstopy - Unrecognized argument {oarg}"),
                    "\n",
                    false,
                );
                flush();
                return 1;
            }
            argind += 1;
        } else {
            break;
        }
    }

    if lenarg - argind < 2 {
        prnstr(
            "ERROR: vmstopy - command file and log file name are required",
            "\n",
            false,
        );
        flush();
        return 1;
    }

    // Open the com file and possibly output file
    let mut outfile: Option<String> = None;
    let mut usetemp = false;
    let logname = argv[argind + 1].clone();
    let com = match std::fs::File::open(&argv[argind]) {
        Ok(file) if !Path::new(&argv[argind]).is_dir() => file,
        _ => {
            prnstr(
                &format!("ERROR: vmstopy - Opening command file {}", argv[argind]),
                "\n",
                false,
            );
            flush();
            return 1;
        }
    };
    let mut out: Box<dyn Write> = Box::new(std::io::stdout());
    if execute || lenarg - argind > 2 {
        let name = if lenarg - argind > 2 {
            argv[argind + 2].clone()
        } else {
            usetemp = true;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0);
            let (min, sec) = ((now / 60) % 60, now % 60);
            format!("{}{}", argv[argind], min * 60 + sec)
        };
        match std::fs::File::create(&name) {
            Ok(file) => out = Box::new(std::io::BufWriter::new(file)),
            Err(_) => {
                prnstr(
                    "ERROR: vmstopy - Opening file for Python script",
                    "\n",
                    false,
                );
                flush();
                return 1;
            }
        }
        outfile = Some(name);
    }
    let closeout = outfile.is_some();

    let converted = convert(com, &logname, &options, &mut *out);
    if let Err(message) = converted {
        return close_error_exit(&message, out, usetemp, outfile.as_deref());
    }
    let _ = out.flush();
    if closeout {
        drop(out);
    }

    let mut retval = 0;
    if set_perm && let Some(name) = outfile.as_ref().filter(|_| !usetemp) {
        use std::os::unix::fs::PermissionsExt as _;
        let result = std::fs::metadata(name).and_then(|meta| {
            let mut mode = (meta.permissions().mode() & 0o7777) | 0o100;
            if mode & 0o040 != 0 {
                mode |= 0o010;
            }
            if mode & 0o004 != 0 {
                mode |= 0o001;
            }
            std::fs::set_permissions(name, std::fs::Permissions::from_mode(mode))
        });
        if result.is_err()
            && std::env::var_os("IMOD_PERMISSION_ERROR_OK").is_none_or(|value| value.is_empty())
        {
            prnstr(
                "ERROR: vmstopy - Making script file executable",
                "\n",
                false,
            );
            retval = 1;
        }
    }

    // Run with a new python interpreter if called for
    if execute && let Some(name) = &outfile {
        if !quiet {
            prnstr("Executing Python script...  ", "", false);
        }
        // Changed (owner, 2026-09-26, no Python or pipes): the script is
        // executed by the in-process runner instead of `python -u`.  Its
        // `imodNice(n)` is applied here, to this process, which is the job.
        if let Some(nice) = options.nice_val {
            unsafe {
                libc::nice(nice as libc::c_int);
            }
        }
        let script = std::fs::read_to_string(name).unwrap_or_default();
        let run = crate::imod::comrun::run_script(
            &script,
            Path::new(&logname),
            &crate::imod::comrun::ComOptions {
                log: Some(logname.clone().into()),
                vmstopy: options.clone(),
                stderr_to_log: true,
            },
        );
        let run_result = if run.status == 0 {
            Ok(())
        } else {
            let mut errors: Vec<String> = run.error.into_iter().collect();
            errors.push(format!("{name}: exited with status {}", run.status));
            Err(super::imodpy::set_run_error(errors, run.status))
        };
        match run_result {
            Ok(_) => {
                if !quiet {
                    prnstr("DONE!", "\n", false);
                }
            }
            Err(_) => {
                let mut nolines = true;
                for l in super::imodpy::get_err_strings() {
                    if !l.contains(name.as_str()) {
                        // the stored lines have lost their endings
                        prnstr(&l, "\n", false);
                        nolines = false;
                    }
                }
                retval = 1;
                if nolines && !quiet {
                    prnstr(
                        "ERROR: vmstopy - Executing the script; see log for error",
                        "\n",
                        false,
                    );
                }
            }
        }
    }

    if usetemp
        && let Some(name) = &outfile
        && std::fs::remove_file(name).is_err()
    {
        prnstr(
            &format!("ERROR: vmstopy - Removing temporary file {name}"),
            "\n",
            false,
        );
        retval = 1;
    }

    flush();
    retval
}

/// The part of the top level of `vmstopy` that reads the command file `com`
/// and writes the Python script to `out` (`vmstopy:193-627`), for the log
/// file `logname`.  `Err(message)` is a call of `closeErrorExit(message)`;
/// whatever was already written to `out` stays there, as it does in the
/// source.
pub fn convert(
    com: impl std::io::Read,
    logname: &str,
    options: &VmstopyOptions,
    out: &mut dyn Write,
) -> Result<(), String> {
    let windows = cfg!(windows) || cfg!(target_os = "cygwin");
    let mut com = com;
    let test_dummy_prog = if options.test { "echo2 " } else { "" };
    let test_opt = if options.test { "-t " } else { "" };
    let chunk = options.chunk;
    let mut anyset = false;
    let mut need_shutil = false;
    let mut need_socket = false;
    let mut need_pid = false;
    let mut has_python = false;
    let mut indented_com = false;
    // prnstr(text, file=out) and prnstr(text, end='', file=out)
    let mut p = |text: &str| {
        let _ = out.write_all(text.as_bytes());
        let _ = out.write_all(b"\n");
    };

    // Output the boilerplate.  Log must be opened binary to keep cygwin from
    // double-converting \r\n to \r\r\n
    //
    p("#!/usr/bin/env python");
    p("# Set up environment: NOHUP, IMOD on path");
    p("import os, sys");
    p("if os.getenv('IMOD_DIR') != None:");
    p("  IMOD_DIR = os.environ['IMOD_DIR']");
    p("  if sys.platform == 'cygwin' and sys.version_info[0] > 2:");
    p("    IMOD_DIR = IMOD_DIR.replace('\\\\', '/')");
    p("    if IMOD_DIR[1] == ':' and IMOD_DIR[2] == '/':");
    p("      IMOD_DIR = '/cygdrive/' + IMOD_DIR[0].lower() + IMOD_DIR[2:]");
    p("  sys.path.insert(0, os.path.join(IMOD_DIR, 'pylib'))");
    p("  from imodpy import *");
    p("  addIMODbinIgnoreSIGHUP()");
    p("else:");
    p("  log.write('ERROR: IMOD_DIR is not defined\\n')");
    p("  sys.exit(1)");
    p("os.environ['PIP_PRINT_ENTRIES'] = '1'");

    // Add components to the path
    for addto in &options.add_to_front {
        p(&format!(
            "os.environ[\"PATH\"] = cygwinPath(\"{addto}\") + os.pathsep + os.environ[\"PATH\"]"
        ));
    }
    for addto in &options.add_to_back {
        p(&format!(
            "os.environ[\"PATH\"] = os.environ[\"PATH\"] + os.pathsep + cygwinPath(\"{addto}\")"
        ));
    }

    // Set environment variables
    for (var, val) in &options.envars {
        p(&format!("os.environ['{var}'] = '{val}'"));
    }

    p("setLibPath()");
    p("# Back up and open log file");
    p(&format!("makeBackupFile('{logname}')"));
    p("try:");
    p(&format!("  log = open('{logname}', 'wb')"));
    p("except Exception:");
    p(&format!(
        "  prnstr('ERROR: Cannot open log file {logname} for writing')"
    ));
    p("  sys.exit(1)");

    //Set niceness if any
    if let Some(nice_val) = options.nice_val.filter(|&value| value != 0) {
        p(&format!("if imodNice({nice_val}):"));
        p("  prnstr('INFO: Cannot change process priority; psutil is not installed', file=log)");
    }

    // Output the PID, flush stderr needed for Windows Python
    p("printPID(True)");

    // Read the file, throwing away comments, labels, selected items, or all lines while
    // seeking a label.  Keep blank lines, they could be input lines
    let label_match = Regex::new(r"^\$ *(\w+): *$").unwrap();
    let exit_match = Regex::new(r"^\$ *exit *$").unwrap();
    let errexit_match = Regex::new(r"^\$( *)exit *([0-9]*) *$").unwrap();
    let goto_match = Regex::new(r"^\$ *goto +(\w+) *$").unwrap();
    let statgo_match = Regex::new(r"^\$ *if +\( *\$status *\) *goto +(\w+) *$").unwrap();
    let comment_match = Regex::new(r"^ *#|\$!").unwrap();
    let set_match = Regex::new(r"^\$( *)set +([^=]+= *)([^ ].*)").unwrap();
    let nono_match = Regex::new(r"^\$( *)set +nonomatch").unwrap();
    let sync_match = Regex::new(r"^\$( *)sync$").unwrap();
    let shifts_match = Regex::new(r"^\$( *)matchshifts").unwrap();
    let envar_match = Regex::new(r"\$\{[A-Z0-9_]+\}").unwrap();
    let rmr_match = Regex::new(r"^\$( *)\\?rm -rf? +").unwrap();
    let mut seek_label: Option<String> = None;
    let mut seek_error: Option<String> = None;
    let mut seek_exit = false;
    let mut lines: Vec<String> = Vec::new();
    let mut errlines: Vec<String> = Vec::new();

    // `com.readline()` in text mode: UTF-8, universal newlines
    let mut bytes = Vec::new();
    if com.read_to_end(&mut bytes).is_err() {
        return Err("Reading from command file".to_owned());
    }
    let text = match String::from_utf8(bytes) {
        Ok(text) => text.replace("\r\n", "\n").replace('\r', "\n"),
        Err(_) => return Err("Reading from command file".to_owned()),
    };
    let mut raw_lines: Vec<&str> = text.split('\n').collect();
    if text.is_empty() || text.ends_with('\n') {
        raw_lines.pop();
    }
    for raw in raw_lines {
        // Need to strip line endings right away; DOS endings confuse cygwin python
        let line = raw.trim_end_matches(['\r', '\n']).to_owned();
        let mut keep = true;

        // Find out if there is an error function to be made
        if errlines.is_empty() && statgo_match.is_match(&line) {
            seek_error = Some(statgo_match.replace_all(&line, "${1}").into_owned());
        }

        // Make sure we know about need to substitute variables before any processing
        if !anyset && (set_match.is_match(&line) || envar_match.is_match(&line)) {
            anyset = true;
        }

        // Look for some other features
        if !need_socket && line.contains("`hostname`") {
            need_socket = true;
        }
        if !need_shutil && rmr_match.is_match(&line) {
            need_shutil = true;
        }
        if !need_pid && line.contains("$$") {
            need_pid = true;
        }
        if !has_python && line.starts_with('>') {
            has_python = true;
        }
        if !indented_com && line.starts_with("$ ") {
            indented_com = true;
        }

        // If we've reached the error label, start saving lines and looking for exit
        if seek_error.is_some()
            && Some(label_match.replace_all(&line, "${1}").into_owned()) == seek_error
        {
            seek_exit = true;
            seek_error = None;
            keep = false;
        } else if seek_exit {
            keep = false;
            errlines.push(line.trim_end().to_owned());
            if errexit_match.is_match(&line) {
                seek_exit = false;
            }
        } else if seek_label.is_some() {
            keep = false;
            if Some(label_match.replace_all(&line, "${1}").into_owned()) == seek_label {
                seek_label = None;
            }
        } else if label_match.is_match(&line) {
            keep = false;
        } else if goto_match.is_match(&line) {
            seek_label = Some(goto_match.replace_all(&line, "${1}").into_owned());
            keep = false;
        } else if exit_match.is_match(&line) {
            break;
        } else if comment_match.is_match(&line) {
            keep = false;
        } else if nono_match.is_match(&line) {
            keep = false;
        } else if windows && sync_match.is_match(line.trim_end()) {
            keep = false;
        } else if shifts_match.is_match(&line) {
            keep = false;
        }

        if keep {
            lines.push(line);
        }
    }

    if need_shutil {
        p("import shutil");
    }
    if need_socket {
        p("import socket");
    }
    if has_python {
        indented_com = false;
    }

    p("");
    p("def closeExit(exitCode):");
    p("  if not exitCode:");
    p("    prnstr('SUCCESSFULLY COMPLETED', file=log)");
    if chunk {
        p("    prnstr('CHUNK DONE', file=log)");
    }
    p("  log.close()");
    p("  sys.exit(exitCode)");

    p("");
    p("def printErrorExit(doExit):");
    p("  for l in getErrStrings():");
    p("    prnstr('ERROR: ' + l, end='', file=log)");
    p("  if doExit:");
    p("    closeExit(1)");

    let tryline = errlines.len();
    if tryline != 0 {
        p("");
        p("def errorFunc():");
        let mut joined = errlines.clone();
        joined.extend(lines);
        lines = joined;
    }

    let echo_match = Regex::new(r"^\$( *)echo *([^ ].*)").unwrap();
    let setenv_match = Regex::new(r"^\$( *)setenv +(\S*) *").unwrap();
    let continue_match = Regex::new(r"\\ *$").unwrap();
    let exist_match = Regex::new(r"^\$( *)if *\( *!? *-e +([^) ]+) *\) *(.*)").unwrap();
    let exist_quote_match =
        Regex::new(r#"^\$( *)if *\( *!? *-e +(['"][^'"]+['"]) *\) *(.*)"#).unwrap();
    let rm_match = Regex::new(r"\\?rm (-f )?").unwrap();
    let remove_match = Regex::new(r"^\$( *)b3dremove +([^-])").unwrap();
    let vms_match = Regex::new(r".*vmstocsh +(\w+.log) *< *(\w+.com).*").unwrap();
    let backup_match =
        Regex::new(r"^\$( *)if *\( *-e +([^) ]+) *\) *\\?mv +([^ ]*) +([^ ~]*)~ *").unwrap();
    let copy_match = Regex::new(r"\\?cp (-f )?").unwrap();
    let mkdir_match = Regex::new(r"^\$( *)mkdir *").unwrap();
    let echo_start = Regex::new(r"^\$ *echo").unwrap();
    let echo_empty = Regex::new(r"^\$ *echo *$").unwrap();
    let echo_empty_sub = Regex::new(r"\$( *)echo *").unwrap();
    let set_tmpdir = Regex::new(r"^\$ *set tmpdir").unwrap();
    let set_tmpdir_sub = Regex::new(r"\$( *)set.*$").unwrap();
    let not_exist = Regex::new(r"if *\( *! *-e").unwrap();
    let print_match = Regex::new(r"\bprint +[^>]").unwrap();
    let print_sub = Regex::new(r"\bprint ").unwrap();
    let dollar_spaces = Regex::new(r"^\$ *").unwrap();
    let indent_sub = Regex::new(r"^\$( *).*").unwrap();
    let quote_state = QuoteState {
        anyset,
        need_pid,
        keep_backslash: options.keep_backslash,
    };

    // If there are indented command lines and no python statements, remove the spaces
    // before trying to process the lines
    if indented_com {
        for line in lines.iter_mut() {
            if line.starts_with("$ ") {
                *line = dollar_spaces.replace_all(line, "$$").into_owned();
            }
        }
    }

    let mut ind = 0usize;
    while ind < lines.len() {
        if ind == tryline {
            p("");
            p("try:");
        }

        let mut line = lines[ind].clone();
        ind += 1;

        // Replace $echo with >print and substitute variable
        if echo_start.is_match(&line) {
            if echo_match.is_match(&line) {
                let prncom = echo_match.replace_all(&line, ">${1}print ").into_owned();
                let message = echo_match.replace_all(&line, "${2}").into_owned();
                let message = quote_substitute(message.trim_matches('"'), &quote_state);
                line = prncom + &message;
            } else if echo_empty.is_match(&line) {
                line = echo_empty_sub
                    .replace_all(&line, ">${1}print \" \"")
                    .into_owned();
            }
        }

        // Replace $set tmpdir and following lines
        if set_tmpdir.is_match(&line) {
            let mut add_ind = 0;
            if ind < lines.len() && lines[ind].contains("settmpdir") {
                add_ind = 1;
            } else if ind + 2 < lines.len()
                && lines[ind + 1].contains("settmpdir")
                && lines[ind].contains("if ")
                && lines[ind + 2].contains("endif")
            {
                add_ind = 3;
            }
            if add_ind != 0 {
                line = set_tmpdir_sub
                    .replace_all(&line, ">${1}tmpdir = imodTempDir()")
                    .into_owned();
                ind += add_ind;
            }
        }

        // Replace $set with > and quote a non-numeric value
        if set_match.is_match(&line) {
            let setcom = set_match.replace_all(&line, ">${1}${2}").into_owned();
            let mut value = set_match.replace_all(&line, "${3}").into_owned();
            let numeric = python_float(&value).is_some();
            if !value.contains('"') && !value.contains('\'') && !numeric {
                value = format!("'{value}'");
                if need_pid {
                    value = value.replace("$$", "' + str(os.getpid()) + '");
                }
                if need_socket {
                    value = value.replace("`hostname`", "' + socket.gethostname() + '");
                }
            }

            line = setcom + &value;
        }

        // Replace $setenv with setting of environ
        if setenv_match.is_match(&line) {
            line = setenv_match
                .replace_all(&line, ">${1}os.environ['${2}'] = '")
                .into_owned()
                + "'";
        }

        // Look for 'if (-e file)' construct and replace with added line
        let exist_test = exist_match.is_match(&line);
        let quoted_test = exist_quote_match.is_match(&line);
        if exist_test || quoted_test {
            let mut needsub = true;

            // But intercept a backup command and use function
            // (the source reads `backupmatch`, an undefined name: fixed)
            if backup_match.is_match(&line) {
                let testfile = backup_match.replace_all(&line, "${2}").into_owned();
                let fromfile = backup_match.replace_all(&line, "${3}").into_owned();
                let tofile = backup_match.replace_all(&line, "${4}").into_owned();
                if testfile == fromfile && fromfile == tofile {
                    line = backup_match
                        .replace_all(&line, ">${1}makeBackupFile(\"${3}\")")
                        .into_owned();
                    needsub = false;
                }
            }

            let use_match = if quoted_test {
                &exist_quote_match
            } else {
                &exist_match
            };
            if needsub {
                let addline = use_match.replace_all(&line, "$$${1}  ${3}").into_owned();
                lines.insert(ind, addline);
                if not_exist.is_match(&line) {
                    line = if quoted_test {
                        use_match
                            .replace_all(&line, ">${1}if not os.path.exists(${2}):")
                            .into_owned()
                    } else {
                        use_match
                            .replace_all(&line, ">${1}if not os.path.exists(\"${2}\"):")
                            .into_owned()
                    };
                } else {
                    line = if quoted_test {
                        use_match
                            .replace_all(&line, ">${1}if os.path.exists(${2}):")
                            .into_owned()
                    } else {
                        use_match
                            .replace_all(&line, ">${1}if os.path.exists(\"${2}\"):")
                            .into_owned()
                    };
                }
            }
        }

        // Replace mkdir with a python function (`0766` in the source is a
        // Python 2 literal and a SyntaxError under Python 3: fixed)
        if mkdir_match.is_match(&line) {
            line = mkdir_match
                .replace_all(line.trim_end(), ">${1}os.mkdir(\"")
                .into_owned()
                + "\", 0o766)";
        }

        // Replace rm -r with a python function
        if rmr_match.is_match(&line) {
            line = rmr_match
                .replace_all(&line, ">${1}shutil.rmtree(\"")
                .into_owned()
                + "\", True)";
        }

        // Replace exit n with a call to closeExit
        if errexit_match.is_match(&line) {
            line = errexit_match
                .replace_all(&line, ">${1}closeExit(${2})")
                .into_owned();
        }

        // For Windows, add a -g to b3dremove; other shells glob for us and we can't count
        // on Windows Python not being used to run, even if this is cygwin python
        if windows && remove_match.is_match(&line) {
            line = remove_match
                .replace_all(&line, "$$${1}b3dremove -g ${2}")
                .into_owned();
        }

        // Python line: just replace print with prnstr( and put file=log at end
        if line.starts_with('>') {
            if print_match.is_match(&line) {
                line = print_sub.replace_all(&line, "prnstr(").into_owned();
                line += ", file=log)";
            }
            p(&format!("  {}", &line[1..]));
        } else if line.starts_with('$') {
            // First gather continuation lines
            while continue_match.is_match(&line) {
                if ind >= lines.len() {
                    return Err("Continued line at end of command file".to_owned());
                }
                line = continue_match.replace_all(&line, " ").into_owned() + &lines[ind];
                ind += 1;
            }

            let indent = indent_sub.replace_all(&line, "${1}").into_owned();
            let mut vmspy = String::new();

            // Convert vmstocsh to vmstopy and run the converted file (for combine.com)
            if vms_match.is_match(&line) {
                let vmslog = vms_match.replace_all(&line, "${1}").into_owned();
                let vmscom = vms_match.replace_all(&line, "${2}").into_owned();
                vmspy = format!("{vmscom}.py");
                while !line.contains("csh -ef") {
                    if ind >= lines.len() {
                        return Err("Cannot find csh -ef after a vmstocsh".to_owned());
                    }
                    line = lines[ind].clone();
                    ind += 1;
                }
                p(&format!("  {indent}try:"));
                p(&format!(
                    "  {indent}  runcmd(\"vmstopy {test_opt}{vmscom} {vmslog} {vmspy}\")"
                ));
                p(&format!("  {indent}except ImodpyError:"));
                p(&format!("  {indent}  printErrorExit(1)"));
                p(&format!("  {indent}command = 'python -u {vmspy}'"));
                p(&format!("  {indent}input = '[]'"));
            } else {
                // Now strip $ and indented spaces, quote and substitute
                line = dollar_spaces.replace_all(&line, "").into_owned();
                if rm_match.find(&line).is_some_and(|found| found.start() == 0) {
                    line = rm_match.replace_all(&line, "b3dremove -g ").into_owned();
                }
                if copy_match
                    .find(&line)
                    .is_some_and(|found| found.start() == 0)
                {
                    let mut subtext = "b3dcopy ".to_owned();
                    if ind < lines.len() && lines[ind].contains("chmod") {
                        ind += 1;
                        subtext += "-p ";
                    }
                    line = copy_match.replace_all(&line, subtext.as_str()).into_owned();
                }
                if line.contains('\\') {
                    line = line.replace("\\rm -f ", "rm -f ");
                    line = line.replace("\\rm ", "rm -f ");
                    line = line.replace("\\mv ", "mv -f ");
                }

                // For windows, see if it is a script and try to run with the interpreter
                // (`line.split()[0]` raises IndexError on an empty command: fixed)
                let comstr = line.split_whitespace().next().unwrap_or("").to_owned();
                if windows
                    && !comstr.is_empty()
                    && Path::new(&comstr).exists()
                    && let Ok(text) = std::fs::read_to_string(&comstr)
                {
                    let firstline = text.split_inclusive('\n').next().unwrap_or("");
                    if firstline.starts_with("#!") {
                        let firstline = firstline.replace("#!", "");
                        let firstline = firstline.trim();
                        let lsplit: Vec<&str> = firstline.split_whitespace().collect();
                        let mut combase = lsplit
                            .first()
                            .map(|path| path.rsplit(['/', '\\']).next().unwrap_or(""))
                            .unwrap_or("")
                            .to_owned();
                        if lsplit.len() > 1 {
                            if combase == "env" {
                                combase = lsplit[1].to_owned();
                            } else {
                                combase += &format!(" {}", lsplit[1]);
                            }
                        }
                        line = format!("{combase} {line}");
                    }
                }

                // Escape a terminal quote then wrap command in """
                if line.ends_with('"') {
                    let linelen = line.len();
                    if linelen > 2 && line.as_bytes()[linelen - 2] != b'\\' {
                        line = format!("{}\\\"", &line[..linelen - 1]);
                    }
                }
                line = quote_substitute(&format!("{test_dummy_prog}{line}"), &quote_state);

                p(&format!("  {indent}command = {line}"));

                // gather input if any
                let mut text = format!("  {indent}input = [");
                let mut inputout = false;
                while ind < lines.len()
                    && !(lines[ind].starts_with('$') || lines[ind].starts_with('>'))
                {
                    let line = quote_substitute(&lines[ind], &quote_state);
                    ind += 1;
                    if inputout {
                        text += &format!(",\n{indent}{}", " ".repeat(11));
                    }
                    text += &line;
                    inputout = true;
                }

                text += "]";
                p(&text);
            }

            p(&format!("  {indent}try:"));
            p(&format!(
                "  {indent}  runcmd(command, input, log, 'stdout')"
            ));
            p(&format!("  {indent}except ImodpyError:"));
            if !vmspy.is_empty() {
                p(&format!("  {indent}  cleanupFiles([\"{vmspy}\"])"));
            }
            if ind < lines.len() && !errlines.is_empty() && statgo_match.is_match(&lines[ind]) {
                p(&format!("  {indent}  printErrorExit(0)"));
                p(&format!("  {indent}  errorFunc()"));
                ind += 1;
            } else {
                p(&format!("  {indent}  printErrorExit(1)"));
            }

            if !vmspy.is_empty() {
                p(&format!("  {indent}cleanupFiles([\"{vmspy}\"])"));
            }
        }
        // skip blank lines, object to anything else
        else if !line.is_empty() {
            return Err(format!("Expected command or Python line: {line}"));
        }
    }

    p("except KeyError:");
    p("  prnstr('ERROR: Environment variable not defined: ' +  str(sys.exc_info()[1]), file=log)");
    p("  closeExit(1)");
    p("except Exception:");
    p("  prnstr('ERROR: Unknown error running commands: ' +  str(sys.exc_info()[1]), file=log)");
    p("  closeExit(1)");
    p("closeExit(0)");
    Ok(())
}

/// Python's `float(text)`: the value, or `None` where it raises
/// `ValueError`.  Surrounding whitespace, a sign, `inf`/`infinity`/`nan` in
/// any case, and single underscores between digits are accepted.
pub fn python_float(text: &str) -> Option<f64> {
    let text = text.trim_matches(|c: char| c.is_whitespace());
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    let lower = unsigned.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" || lower == "nan" {
        return text.parse::<f64>().ok();
    }
    if unsigned.is_empty()
        || !unsigned
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-' | '_'))
    {
        return None;
    }
    // underscores only between two digits
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_'
            && !(i > 0
                && chars[i - 1].is_ascii_digit()
                && chars.get(i + 1).is_some_and(|next| next.is_ascii_digit()))
        {
            return None;
        }
    }
    text.replace('_', "").parse::<f64>().ok()
}
