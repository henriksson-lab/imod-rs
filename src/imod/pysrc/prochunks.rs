//! Translation of `IMOD/pysrc/prochunks.py`: functions for running
//! processchunks in the background.
use super::imodpy::{
    OptionValue, bkgd_process, imod_abs_path, option_value, os_path_normpath, prnstr,
    read_text_file, write_text_file,
};
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// The module global `finishSetAndQuit` of `prochunks.py`.  `from imodpy
/// import *` (`prochunks.py:10`) binds a copy of `imodpy.finishSetAndQuit`
/// (`False`) in this module's namespace, and `global finishSetAndQuit`
/// (`prochunks.py:27`) assigns that copy -- so it is this module's own flag,
/// shared by every caller in the process and distinct from the flag of the
/// same name in a calling script.
static FINISH_SET_AND_QUIT: AtomicBool = AtomicBool::new(false);

/// The 16-element `param` list of `prochunks.py`, one field per index
/// `TopQuitPci` .. `CheckNamePci` (`prochunks.py:13-15`), in index order.
/// `FinishedPci` and `ErrorPci` are declared by the source and never used.
#[derive(Debug, Default)]
pub struct ProChunksParam {
    /// `TopQuitPci`: the last `checkForProChunksQuit` result.
    pub top_quit: String,
    /// `NumDonePci`: the first word of the last `DONE SO FAR` line -- a
    /// string, as `l.split()[0]` leaves it (`prochunks.py:129`), or the
    /// initial `0`.
    pub num_done: String,
    /// `ElapsedPci`
    pub elapsed: f64,
    /// `GotLinesAtPci`
    pub got_lines_at: f64,
    /// `OpenedPci`
    pub opened: bool,
    /// `ReadOKPci`
    pub read_ok: bool,
    /// `GotPIDPci`
    pub got_pid: bool,
    /// `FinishedPci` (unused by the source)
    pub finished: i32,
    /// `ErrorPci` (unused by the source)
    pub error: i32,
    /// `CheckTimePci`
    pub check_time: f64,
    /// `InChunkErrorPci`
    pub in_chunk_error: bool,
    /// `GotDoneSoFarPci`
    pub got_done_so_far: bool,
    /// `WherePci`: a `tell()` position, or `None`.
    pub where_: Option<u64>,
    /// `OutFilePci`: the opened output file.
    pub out_file: Option<File>,
    /// `OutNamePci`
    pub out_name: String,
    /// `CheckNamePci`
    pub check_name: String,
}

/// Matches `checkForProChunksQuit` (`IMOD/pysrc/prochunks.py:25`).
///
/// The source returns the integer `0` when there is no top check file and
/// `''` otherwise; both are false to every caller, and both are the empty
/// string here.
pub fn check_for_pro_chunks_quit(
    top_check_file: Option<&str>,
    pro_chunk_check_file: Option<&str>,
    look_for_pause: bool,
    send_pause_for_f: bool,
) -> String {
    let pause_mess = "RECEIVED SIGNAL TO FINISH CURRENT OPERATIONS AND EXIT";
    let top_check_file = match top_check_file {
        Some(name) if !name.is_empty() && std::path::Path::new(name).exists() => name,
        _ => return String::new(),
    };
    let checklines = match read_text_file(top_check_file, None, true, None) {
        Ok(lines) => lines,
        Err(_) => return String::new(),
    };
    if checklines.is_empty() {
        return String::new();
    }
    let last = &checklines[checklines.len() - 1];
    if last.starts_with('Q') {
        if let Some(check) = pro_chunk_check_file.filter(|name| !name.is_empty()) {
            let _ = write_text_file(check, &["Q".to_owned()], true);
        }
        return "Q".to_owned();
    }
    let is_p = last.starts_with('P');
    let is_f = last.starts_with('F');
    if is_f || (look_for_pause && is_p) {
        if is_p || send_pause_for_f {
            if let Some(check) = pro_chunk_check_file.filter(|name| !name.is_empty()) {
                let _ = write_text_file(check, &["P".to_owned()], true);
            }
        }
        if !FINISH_SET_AND_QUIT.load(Ordering::SeqCst) {
            prnstr(pause_mess, "\n", true);
        }
        FINISH_SET_AND_QUIT.store(true, Ordering::SeqCst);
        return last[..1].to_owned();
    }
    String::new()
}

/// Matches `startProcesschunks` (`IMOD/pysrc/prochunks.py:55`).  Returns
/// the error message `bkgdProcess` returns with `returnOnErr`, or `None`.
pub fn start_processchunks(
    com_array: &[OsString],
    outfile: &str,
    pro_chunk_check_file: &str,
    param: &mut ProChunksParam,
) -> Option<String> {
    param.elapsed = 0.;
    param.got_lines_at = -1.;
    param.opened = false;
    param.read_ok = false;
    param.got_pid = false;
    param.check_time = 0.;
    param.in_chunk_error = false;
    param.got_done_so_far = false;
    param.where_ = None;
    param.top_quit = String::new();
    param.num_done = "0".to_owned();
    param.out_file = None;
    param.out_name = outfile.to_owned();
    param.check_name = pro_chunk_check_file.to_owned();

    let mut full = vec![
        OsString::from("processchunks"),
        OsString::from("-P"),
        OsString::from("-g"),
        OsString::from("-c"),
        OsString::from(pro_chunk_check_file),
    ];
    full.extend_from_slice(com_array);
    match bkgd_process(&full, Some(outfile), Some("stdout"), true, false) {
        Ok(()) => None,
        Err(error) => Some(error.arguments.join("\n")),
    }
}

/// Matches `checkProChunksLog` (`IMOD/pysrc/prochunks.py:89`).  Returns
/// `(error, finished, topQuit, numDone, message)`.
///
/// The output file is read as Python's text-mode `readlines()` reads it:
/// from the current position to end of file, with a final line lacking its
/// ending returned as a line.  Bytes are decoded as UTF-8 with replacement
/// where Python would raise `UnicodeDecodeError`.
pub fn check_pro_chunks_log(
    top_check_file: Option<&str>,
    param: &mut ProChunksParam,
    look_for_pause: bool,
    send_pause_for_f: bool,
    print_output: bool,
) -> (i32, i32, String, String, String) {
    let mut finished = 0;
    let mut error = 0;
    let mut message = String::new();
    let _sleep_time = 0.2;
    let starting_time_out = 60.;
    let read_error_timeout = 30.;
    let check_interval = 5.;

    let io_result: std::io::Result<()> = (|| {
        // Seek to the last location before trying again if nothing was gotten.  This is
        // a trick from stackoverflow.com and is needed on Mac
        if let Some(position) = param.where_ {
            if let Some(file) = param.out_file.as_mut() {
                let _ = file.seek(SeekFrom::Start(position));
            }
        }

        // Keep track of whether it opened and whether it read lines without an error
        if !param.opened {
            param.out_file = Some(File::open(&param.out_name)?);
            param.opened = true;
        } else {
            let file = param
                .out_file
                .as_mut()
                .expect("processchunks output file opened");
            param.where_ = Some(file.stream_position()?);
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.split_inclusive('\n').collect();
            param.read_ok = true;
            if !lines.is_empty() {
                // Keep track of last time lines were gotten without error and look for
                // various terminations
                param.got_lines_at = param.elapsed;
                param.where_ = None;
                for l in lines {
                    if print_output && param.got_done_so_far {
                        prnstr(l.trim(), "\n", false);
                    }
                    if l.contains("DONE SO FAR") {
                        param.got_done_so_far = true;
                        if let Some(first) = l.split_whitespace().next() {
                            param.num_done = first.to_owned();
                        }
                    }
                    if l.starts_with("CHUNK ERROR:") {
                        param.in_chunk_error = true;
                    } else if l.starts_with("END CHUNK ERROR") {
                        param.in_chunk_error = false;
                    } else if !param.in_chunk_error && l.starts_with("ERROR:") {
                        message = format!("processchunks {}", l.trim());
                        finished = -1;
                        if !l.contains("has given processing error 1 times - giving up") {
                            error = 1;
                        }
                        break;
                    } else if l.starts_with("Finished reassembling") {
                        finished = 1;
                        break;
                    } else if l.contains("retain") && l.contains("existing") {
                        finished = -2;
                        break;
                    } else if l.contains("PID:") {
                        param.got_pid = true;
                    }
                }
            }
        }
        Ok(())
    })();
    if io_result.is_err() {
        param.read_ok = false;
    }

    if finished != 0 {

        // Check for various timeouts and quit
    } else if !param.opened && param.elapsed > starting_time_out {
        message =
            "Timeout occurred before processchunks output file could be opened for monitoring"
                .to_owned();
        error = 1;
    } else if param.read_ok && param.got_lines_at < 0. && param.elapsed > starting_time_out {
        message = "Timeout occurred before processchunks started".to_owned();
        error = 1;
    } else if param.read_ok && !param.got_pid && param.elapsed > starting_time_out {
        message = format!(
            "Processchunks apparently failed to run; check {} for messages",
            imod_abs_path(&param.out_name)
        );
        error = 1;

    // If can't get the output, tell processchunks to quit
    } else if param.opened
        && !param.read_ok
        && param.elapsed - param.got_lines_at > read_error_timeout
    {
        // Fixed in translation (BUGS.md): native writes to
        // `proChunkCheckFile` (`prochunks.py:178`), which is not defined in
        // this function (NameError, uncaught); the check file it means is
        // `param[CheckNamePci]`.
        let _ = write_text_file(&param.check_name, &["Q".to_owned()], true);
        thread::sleep(Duration::from_secs_f64(5.));
        message = "Unable to read processchunks output file without an error".to_owned();
        error = 1;

    // Check for Q ourselves in case processchunks is lost
    } else if param.elapsed - param.check_time > check_interval {
        param.check_time = param.elapsed;
        param.top_quit = check_for_pro_chunks_quit(
            top_check_file,
            Some(&param.check_name),
            look_for_pause,
            send_pause_for_f,
        );
        if param.top_quit == "Q" {
            error = 1;
        }
    }

    if param.opened && (error != 0 || finished != 0) {
        param.out_file = None;
    }

    (
        error,
        finished,
        param.top_quit.clone(),
        param.num_done.clone(),
        message,
    )
}

/// Matches `runProcesschunks` (`IMOD/pysrc/prochunks.py:215`).  Returns
/// `(error, finished, topQuit, numDone, message)`.
///
/// The source's `except KeyboardInterrupt` arm has no counterpart: an
/// interrupt ends this process.
pub fn run_processchunks(
    com_array: &[OsString],
    outfile: &str,
    top_check_file: Option<&str>,
    pro_chunk_check_file: &str,
    look_for_pause: bool,
    send_pause_for_f: bool,
    print_output: bool,
) -> (i32, i32, String, String, String) {
    let sleep_time = 0.2;
    let mut param = ProChunksParam::default();
    let message = start_processchunks(com_array, outfile, pro_chunk_check_file, &mut param);
    if let Some(message) = message {
        // Fixed in translation (BUGS.md): native's `return (-1, 0, 0,
        // numDone, message)` (`prochunks.py:221`) reads `numDone` before any
        // assignment (UnboundLocalError, uncaught); no chunks are done, so
        // it returns 0 done with the start-up message.
        return (-1, 0, String::new(), "0".to_owned(), message);
    }
    loop {
        let (error, finished, top_quit, num_done, message) = check_pro_chunks_log(
            top_check_file,
            &mut param,
            look_for_pause,
            send_pause_for_f,
            print_output,
        );
        if error != 0 || finished != 0 {
            return (error, finished, top_quit, num_done, message);
        }
        param.elapsed += sleep_time;
        thread::sleep(Duration::from_secs_f64(sleep_time));
    }
}

/// Matches `transferRemoteDirectory` (`IMOD/pysrc/prochunks.py:246`).
/// Returns `(remoteDataDir, errMess)`.
pub fn transfer_remote_directory(
    remote_start_dir: &str,
    starting_dir: &str,
    dataset_dir: &str,
) -> (String, String) {
    let abs_data_dir = imod_abs_path(dataset_dir)
        .trim_end_matches(['/', '\\'])
        .to_owned();
    let abs_start_dir = imod_abs_path(starting_dir)
        .trim_end_matches(['/', '\\'])
        .to_owned();
    // `os.path.commonprefix`: the longest common leading run of characters
    let mut prefix: String = abs_start_dir
        .chars()
        .zip(abs_data_dir.chars())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a)
        .collect();
    let mut remote_data_dir = String::new();
    let mut err_mess = String::new();
    // `startRemnant` is assigned only on two of the paths below
    let mut start_remnant: Option<String> = None;

    // It is not a usable prefix if it is just a drive/mount point
    let pchars: Vec<char> = prefix.chars().collect();
    if pchars.len() == 1
        || prefix == "//"
        || prefix == "\\\\"
        || (pchars.len() == 3 && pchars[1] == ':' && (pchars[2] == '/' || pchars[2] == '\\'))
    {
        prefix = String::new();
    }

    if abs_data_dir == abs_start_dir {
        remote_data_dir = remote_start_dir.to_owned();
    } else if !prefix.is_empty()
        && abs_data_dir.starts_with(&prefix)
        && abs_start_dir.starts_with(&prefix)
    {
        // `os.path.join(a, b)`
        let join = |a: &str, b: &str| -> String {
            if b.starts_with('/') {
                return b.to_owned();
            }
            let mut joined = a.to_owned();
            if !joined.is_empty() && !joined.ends_with('/') {
                joined.push('/');
            }
            joined.push_str(b);
            joined
        };
        if prefix == abs_data_dir {
            let remnant = abs_start_dir[prefix.len()..].to_owned();
            if let Some(ind) = remote_start_dir.find(&remnant).filter(|ind| *ind > 0) {
                remote_data_dir = remote_start_dir[0..ind].to_owned();
            }
            start_remnant = Some(remnant);
        } else {
            let mut data_remnant = &abs_data_dir[prefix.len()..];
            if data_remnant.starts_with('/') || data_remnant.starts_with('\\') {
                data_remnant = &data_remnant[1..];
            }
            if prefix == abs_start_dir {
                remote_data_dir = os_path_normpath(&join(remote_start_dir, data_remnant));
            } else {
                let remnant = abs_start_dir[prefix.len()..].to_owned();
                if let Some(ind) = remote_start_dir.find(&remnant).filter(|ind| *ind > 0) {
                    remote_data_dir =
                        os_path_normpath(&join(&remote_start_dir[0..ind], data_remnant));
                }
                start_remnant = Some(remnant);
            }
        }
    }

    if remote_data_dir.is_empty() {
        if !prefix.is_empty() {
            // Fixed in translation (BUGS.md): for equal directories with an
            // empty remote entry native reads the never-assigned
            // `startRemnant` (`prochunks.py:284`, UnboundLocalError); the
            // remnant of equal directories is empty, and the error message
            // is returned with it.
            let start_remnant = start_remnant.unwrap_or_default();
            err_mess = format!(
                "Cannot translate remote directory entry {remote_start_dir} to work with {dataset_dir} - common prefix of {abs_start_dir} and {abs_data_dir} is {prefix} , remnant of {start_remnant} is not in remote dir"
            );
        } else {
            err_mess = format!(
                "Cannot translate remote directory entry {remote_start_dir} to work with {dataset_dir} - there is no common prefix of {abs_start_dir} and {abs_data_dir}"
            );
        }
    }

    (remote_data_dir, err_mess)
}

/// Matches `getTranslationFromRemoteDir` (`IMOD/pysrc/prochunks.py:295`).
/// Returns `(comDirRoot, remoteRoot)`.
pub fn get_translation_from_remote_dir(
    full_lines: &[String],
    common_lines: &[String],
    comfile: &str,
) -> (String, String) {
    let mut remote_root = match option_value(full_lines, "RemoteDirectory", 0, false, 0, None, None)
    {
        Some(OptionValue::String(value)) if !value.is_empty() => value,
        _ => return (String::new(), String::new()),
    };

    // `os.path.dirname(comfile)`
    let comdir = match comfile.rfind('/') {
        Some(index) => {
            let head = &comfile[..index + 1];
            if !head.chars().all(|c| c == '/') {
                head.trim_end_matches('/')
            } else {
                head
            }
        }
        None => "",
    };
    let mut com_dir_root = imod_abs_path(comdir)
        .trim_end_matches(['/', '\\'])
        .to_owned();
    remote_root = remote_root.trim_end_matches(['/', '\\']).to_owned();

    // `os.path.split(p)`: (head, tail), with trailing slashes stripped from a
    // head that is not all slashes
    let split = |p: &str| -> (String, String) {
        let i = p.rfind('/').map_or(0, |index| index + 1);
        let (mut head, tail) = (p[..i].to_owned(), p[i..].to_owned());
        if !head.is_empty() && !head.chars().all(|c| c == '/') {
            head = head.trim_end_matches('/').to_owned();
        }
        (head, tail)
    };

    // Strip off directories that match, quit when last directory does not
    while !remote_root.is_empty() && !com_dir_root.is_empty() {
        let remote_temp = split(&remote_root);
        let com_temp = split(&com_dir_root);
        if remote_temp.1 != com_temp.1 {
            break;
        }
        remote_root = remote_temp.0;
        com_dir_root = com_temp.0;
    }

    if remote_root.chars().count() > 1 || com_dir_root.chars().count() > 1 {
        let mut broke = false;
        for line in common_lines {
            if line.contains("TranslatePathsFrom") && line.contains(&com_dir_root) {
                broke = true;
                break;
            }
        }
        if !broke {
            return (com_dir_root, remote_root);
        }
    }

    (String::new(), String::new())
}
