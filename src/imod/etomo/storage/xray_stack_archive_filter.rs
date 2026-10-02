//! `IMOD/Etomo/src/etomo/storage/XrayStackArchiveFilter.java`.
//!
//! `implements java.io.FilenameFilter`: accepts the numbered gzip archives of an
//! x-ray-erased raw stack.
//!
//! TODO 2206 - Use the current raw stack extension (source comment).

use std::path::Path;

use regex::Regex;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `XrayStackArchiveFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct XrayStackArchiveFilter {}

impl XrayStackArchiveFilter {
    /// Java's implicit `XrayStackArchiveFilter()`.
    pub fn new() -> XrayStackArchiveFilter {
        XrayStackArchiveFilter {}
    }

    /// Java `accept(File, String)`: `name.matches(".*_xray.st.gz.\\d+")`.  Java's `.`
    /// does not match line terminators, and `\d` is ASCII only.
    pub fn accept(&self, _dir: &Path, name: &str) -> bool {
        if Regex::new(r"^(?:[^\n\r\x{85}\x{2028}\x{2029}]*_xray[^\n\r\x{85}\x{2028}\x{2029}]st[^\n\r\x{85}\x{2028}\x{2029}]gz[^\n\r\x{85}\x{2028}\x{2029}][0-9]+)$")
            .unwrap()
            .is_match(name)
        {
            return true;
        }
        false
    }
}
