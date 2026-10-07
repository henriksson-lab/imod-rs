//! Translation of `IMOD/pysrc/b3dtouch`: the equivalent of the Unix-type
//! touch command.
//!
//! A Python command script with no functions; its top level is
//! [`b3dtouch`].

use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`b3dtouch:1-27`).  Returns the status of its
/// `sys.exit`, 0 when it falls off the end.
pub fn b3dtouch(arguments: &[OsString]) -> i32 {
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let mut out = std::io::stdout();
    if argv.len() < 2 {
        let _ = out.write_all(
            b"Usage: b3dtouch filename
    Updates the modification time of an existing file or creates a new one
    if it does not exist.\n",
        );
        let _ = out.flush();
        return 0;
    }

    let fname = &argv[1];
    let mess;
    let result = if Path::new(fname).exists() {
        mess = "Updating time of existing file ";
        // `os.utime(fname, None)`: access and modification times set to now.
        #[cfg(unix)]
        {
            let path = std::ffi::CString::new(fname.as_bytes()).unwrap_or_default();
            // SAFETY: `utime` with a NULL times pointer sets both to the current
            // time; `path` is a valid C string.
            if unsafe { libc::utime(path.as_ptr(), std::ptr::null()) } == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        }
        // CPython's Windows `os.utime(path, None)`: `SetFileTime` to the
        // current time on a handle opened for writing attributes.
        #[cfg(not(unix))]
        {
            let now = std::time::SystemTime::now();
            std::fs::OpenOptions::new()
                .write(true)
                .open(fname)
                .and_then(|file| {
                    file.set_times(
                        std::fs::FileTimes::new()
                            .set_accessed(now)
                            .set_modified(now),
                    )
                })
        }
    } else {
        mess = "Creating a new empty file ";
        // `open(fname, 'a').close()`
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(fname)
            .map(|_| ())
    };
    if let Err(error) = result {
        // `str(sys.exc_info()[1])` of an OSError
        let text = error.to_string();
        let text = match error.raw_os_error() {
            Some(errno) => format!(
                "[Errno {errno}] {}: '{fname}'",
                text.strip_suffix(&format!(" (os error {errno})"))
                    .unwrap_or(&text)
            ),
            None => text,
        };
        let _ = out.write_all(format!("Error: b3dtouch - {mess}{fname} : {text}\n").as_bytes());
        let _ = out.flush();
        return 1;
    }
    0
}
