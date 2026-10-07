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
    let node = crate::imod::libcfshr::b3dutil::host_name();
    let mut out = std::io::stdout();
    let _ = out.write_all(format!("{node}\n").as_bytes());
    let _ = out.flush();
    0
}
