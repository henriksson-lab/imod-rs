//! Translation of `IMOD/pysrc/b3dcopy`.
//!
//! A Python command script with no functions: a utility to copy a file and
//! make sure permission is set for the copy.  If the output file is a
//! directory, it copies the file into that directory.  Its top level is
//! [`b3dcopy`], translated statement by statement.
//!
//! Command files run it (`$b3dcopy -p "ts_fid.xf" "ts.xf"` in `align.com`),
//! and `vmstopy` converts `cp`/`\cp` lines to it, so the in-process command
//! file runner ([`crate::imod::comrun`]) calls it directly.

use std::ffi::OsString;
use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;

/// The script's top level (`b3dcopy:9-85`).  Returns the status of its
/// `sys.exit`.
pub fn b3dcopy(arguments: &[OsString]) -> i32 {
    let _retry_wait = 0.2;
    let max_trials = 10;
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
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

    let report_errors =
        std::env::var_os("REPORT_B3DCOPY_ERRORS").is_some_and(|value| !value.is_empty());
    let mut doperm = 0;
    if argv.len() > 1 && argv[1] == "-p" {
        doperm = 1;
    }

    if argv.len() < 3 + doperm {
        print!("ERROR: b3dcopy - not enough arguments\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    let fromfile = argv[1 + doperm].clone();
    let mut tofile = argv[2 + doperm].clone();
    if !Path::new(&fromfile).exists() {
        print!("ERROR: b3dcopy - file to copy does not exist: {fromfile}\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    if Path::new(&tofile).is_dir() {
        let conv = fromfile.replace('\\', "/");
        // os.path.basename, then os.path.join
        let base = conv.rsplit('/').next().unwrap_or("").to_owned();
        tofile = if tofile.ends_with('/') || tofile.is_empty() {
            format!("{tofile}{base}")
        } else {
            format!("{tofile}/{base}")
        };
    }

    if Path::new(&tofile).exists()
        && let Err(error) = std::fs::remove_file(&tofile)
    {
        print!(
            "WARNING: b3dcopy - Could not remove existing {tofile} : {}\n",
            exc_text(&error, &tofile)
        );
    }

    let mut info = String::new();
    let mut copied = false;
    for trial in 0..max_trials {
        // shutil.copyfile: contents only, refusing a directory source
        let result = if Path::new(&fromfile).is_dir() {
            Err(std::io::Error::from_raw_os_error(libc::EISDIR))
        } else {
            std::fs::File::open(&fromfile).and_then(|mut source| {
                let mut dest = std::fs::File::create(&tofile)?;
                std::io::copy(&mut source, &mut dest).map(|_| ())
            })
        };
        match result {
            Ok(()) => {
                if trial > 0 {
                    print!("It took {} tries to copy {fromfile}\n", trial + 1);
                }
                copied = true;
                break;
            }
            Err(error) => {
                info = exc_text(&error, &fromfile);
                if report_errors {
                    print!(
                        "WARNING: b3dcopy - Error copying {fromfile} to {tofile} on trial {} : {info}\n",
                        trial + 1
                    );
                }
            }
        }
    }
    if !copied {
        // ELSE ON FOR
        print!(
            "WARNING: b3dcopy - Could not copy {fromfile} to {tofile} in {max_trials} tries: {info}\n"
        );
    }

    if doperm == 0 {
        let _ = std::io::stdout().flush();
        return 0;
    }

    let mut set = false;
    for trial in 0..max_trials {
        let result = std::fs::metadata(&tofile).and_then(|meta| {
            let mode = (meta.permissions().mode() & 0o7777) | 0o400 | 0o200;
            std::fs::set_permissions(&tofile, std::fs::Permissions::from_mode(mode))
        });
        match result {
            Ok(()) => {
                if trial > 0 {
                    print!(
                        "It took {} tries to set permissions of {tofile}\n",
                        trial + 1
                    );
                }
                set = true;
                break;
            }
            Err(error) => {
                info = exc_text(&error, &tofile);
                if std::env::var_os("IMOD_PERMISSION_ERROR_OK")
                    .is_some_and(|value| !value.is_empty())
                {
                    set = true;
                    break;
                }
                if report_errors {
                    print!(
                        "WARNING: b3dcopy - Error setting permissions of {tofile} on trial {} : {info}\n",
                        trial + 1
                    );
                }
            }
        }
    }
    if !set {
        // ELSE ON FOR
        print!(
            "WARNING: b3dcopy - Could not set permissions of {tofile} in {max_trials} tries: {info}\n"
        );
    }

    let _ = std::io::stdout().flush();
    0
}
