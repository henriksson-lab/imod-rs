//! Translation of `IMOD/pysrc/prochunks.py`.
use super::imodpy::{bkgd_process, read_text_file, write_text_file};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::thread;
use std::time::Duration;

pub const TOP_QUIT_PCI: usize = 0;
pub const NUM_DONE_PCI: usize = 1;
pub const ELAPSED_PCI: usize = 2;
pub const GOT_LINES_AT_PCI: usize = 3;
pub const OPENED_PCI: usize = 4;
pub const READ_OK_PCI: usize = 5;
pub const GOT_PID_PCI: usize = 6;
pub const FINISHED_PCI: usize = 7;
pub const ERROR_PCI: usize = 8;
pub const CHECK_TIME_PCI: usize = 9;
pub const IN_CHUNK_ERROR_PCI: usize = 10;
pub const GOT_DONE_SO_FAR_PCI: usize = 11;
pub const WHERE_PCI: usize = 12;
pub const OUT_FILE_PCI: usize = 13;
pub const OUT_NAME_PCI: usize = 14;
pub const CHECK_NAME_PCI: usize = 15;

/// Matches `checkForProChunksQuit` (`IMOD/pysrc/prochunks.py:23`).
pub fn check_for_pro_chunks_quit(
    top_check_file: Option<&str>,
    pro_chunk_check_file: Option<&str>,
    look_for_pause: bool,
    send_pause_for_f: bool,
    finish_set_and_quit: &mut bool,
) -> String {
    let Some(top) = top_check_file.filter(|name| !name.is_empty()) else {
        return String::new();
    };
    let Ok(lines) = read_text_file(top, None, true, None) else {
        return String::new();
    };
    let Some(line) = lines.last() else {
        return String::new();
    };
    let code = line.chars().next().unwrap_or_default();
    if code == 'Q' {
        if let Some(check) = pro_chunk_check_file {
            let _ = write_text_file(check, &["Q".to_owned()], true);
        }
        return "Q".to_owned();
    }
    if code == 'F' || (look_for_pause && code == 'P') {
        if (code == 'P' || send_pause_for_f) && pro_chunk_check_file.is_some() {
            let _ = write_text_file(pro_chunk_check_file.unwrap(), &["P".to_owned()], true);
        }
        *finish_set_and_quit = true;
        return code.to_string();
    }
    String::new()
}

/// Matches `startProcesschunks` (`IMOD/pysrc/prochunks.py:59`).
pub fn start_processchunks(
    command: &[OsString],
    outfile: &str,
    check_file: &str,
) -> Result<(), String> {
    let mut full = vec![
        OsString::from("processchunks"),
        OsString::from("-P"),
        OsString::from("-g"),
        OsString::from("-c"),
        OsString::from(check_file),
    ];
    full.extend_from_slice(command);
    bkgd_process(&full, Some(outfile), Some("stdout"), true, true)
        .map_err(|error| error.arguments.join("\n"))
}

/// Matches `checkProChunksLog` (`IMOD/pysrc/prochunks.py:90`).
pub fn check_pro_chunks_log(
    top_check_file: Option<&str>,
    outfile: &str,
    check_file: &str,
    elapsed: f64,
    look_for_pause: bool,
    send_pause_for_f: bool,
    finish_set_and_quit: &mut bool,
) -> (i32, i32, String, i32, String) {
    let Ok(text) = fs::read_to_string(outfile) else {
        return (0, 0, String::new(), 0, String::new());
    };
    let mut finished = 0;
    let mut error = 0;
    let mut done = 0;
    let mut message = String::new();
    let mut in_chunk_error = false;
    for line in text.lines() {
        if line.contains("DONE SO FAR") {
            done = line
                .split_whitespace()
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(done);
        }
        if line.starts_with("CHUNK ERROR:") {
            in_chunk_error = true;
        } else if line.starts_with("END CHUNK ERROR") {
            in_chunk_error = false;
        } else if !in_chunk_error && line.starts_with("ERROR:") {
            message = format!("processchunks {line}");
            finished = -1;
            if !line.contains("has given processing error 1 times - giving up") {
                error = 1;
            }
            break;
        } else if line.starts_with("Finished reassembling") {
            finished = 1;
            break;
        } else if line.contains("retain") && line.contains("existing") {
            finished = -2;
            break;
        }
    }
    let quit = if elapsed > 5.0 {
        check_for_pro_chunks_quit(
            top_check_file,
            Some(check_file),
            look_for_pause,
            send_pause_for_f,
            finish_set_and_quit,
        )
    } else {
        String::new()
    };
    if quit == "Q" {
        error = 1;
    }
    (error, finished, quit, done, message)
}

/// Matches `runProcesschunks` (`IMOD/pysrc/prochunks.py:220`).
pub fn run_processchunks(
    command: &[OsString],
    outfile: &str,
    top_check_file: Option<&str>,
    check_file: &str,
    look_for_pause: bool,
    send_pause_for_f: bool,
) -> (i32, i32, String, i32, String) {
    if let Err(message) = start_processchunks(command, outfile, check_file) {
        return (-1, 0, String::new(), 0, message);
    }
    let mut elapsed = 0.0;
    let mut finish = false;
    loop {
        let result = check_pro_chunks_log(
            top_check_file,
            outfile,
            check_file,
            elapsed,
            look_for_pause,
            send_pause_for_f,
            &mut finish,
        );
        if result.0 != 0 || result.1 != 0 {
            return result;
        }
        elapsed += 0.2;
        thread::sleep(Duration::from_millis(200));
    }
}

/// Matches `transferRemoteDirectory` (`IMOD/pysrc/prochunks.py:259`).
pub fn transfer_remote_directory(
    remote_start_dir: &str,
    starting_dir: &str,
    dataset_dir: &str,
) -> (String, String) {
    let start = match fs::canonicalize(starting_dir) {
        Ok(path) => path,
        Err(_) => {
            return (
                String::new(),
                format!(
                    "Cannot translate remote directory entry {remote_start_dir} to work with {dataset_dir}"
                ),
            );
        }
    };
    let data = match fs::canonicalize(dataset_dir) {
        Ok(path) => path,
        Err(_) => {
            return (
                String::new(),
                format!(
                    "Cannot translate remote directory entry {remote_start_dir} to work with {dataset_dir}"
                ),
            );
        }
    };
    let common = start
        .components()
        .zip(data.components())
        .take_while(|(a, b)| a == b)
        .count();
    if common == 0 {
        return (
            String::new(),
            format!(
                "Cannot translate remote directory entry {remote_start_dir} to work with {dataset_dir} - there is no common prefix of {starting_dir} and {dataset_dir}"
            ),
        );
    }
    let up = start.components().count().saturating_sub(common);
    let mut remote = std::path::PathBuf::from(remote_start_dir);
    for _ in 0..up {
        remote.pop();
    }
    for component in data.components().skip(common) {
        remote.push(component.as_os_str());
    }
    (remote.to_string_lossy().into_owned(), String::new())
}

/// Matches `getTranslationFromRemoteDir` (`IMOD/pysrc/prochunks.py:309`).
pub fn get_translation_from_remote_dir(
    full_lines: &[String],
    common_lines: &[String],
    comfile: &str,
) -> (String, String) {
    let Some(mut remote) = full_lines
        .iter()
        .filter_map(|line| line.split_once('='))
        .find_map(|(key, value)| (key.trim() == "RemoteDirectory").then(|| value.trim().to_owned()))
        .filter(|value| !value.is_empty())
    else {
        return (String::new(), String::new());
    };
    let mut com = std::path::PathBuf::from(comfile)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .to_path_buf();
    while let (Some(remote_name), Some(com_name)) =
        (std::path::Path::new(&remote).file_name(), com.file_name())
    {
        if remote_name != com_name {
            break;
        }
        remote = std::path::Path::new(&remote)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""))
            .to_string_lossy()
            .into_owned();
        com.pop();
    }
    if remote.len() <= 1 && com.to_string_lossy().len() <= 1 {
        return (String::new(), String::new());
    }
    let com_text = com.to_string_lossy().into_owned();
    if common_lines
        .iter()
        .any(|line| line.contains("TranslatePathsFrom") && line.contains(&com_text))
    {
        return (String::new(), String::new());
    }
    (com_text, remote)
}
