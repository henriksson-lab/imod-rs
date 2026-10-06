//! Translation of `IMOD/pysrc/b3dhostname`: prints the host name on all
//! platforms.
//!
//! A Python command script with no functions; its top level is
//! [`b3dhostname`].

use std::io::Write as _;

/// The script's top level (`b3dhostname:1-10`).  Returns the status of its
/// `sys.exit(0)`.
///
/// `platform.node()` is `os.uname().nodename`, the kernel's host name, which
/// POSIX `gethostname` returns (the one foreign call).
pub fn b3dhostname() -> i32 {
    let mut buffer = [0u8; 256];
    // SAFETY: `gethostname` writes at most `buffer.len()` bytes into it.
    let failed = unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) } != 0;
    let node = if failed {
        String::new()
    } else {
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        String::from_utf8_lossy(&buffer[..end]).into_owned()
    };
    let mut out = std::io::stdout();
    let _ = out.write_all(format!("{node}\n").as_bytes());
    let _ = out.flush();
    0
}
