//! `IMOD/Etomo/src/etomo/ui/BrowsingDirectory.java`.
//!
//! A Java interface with two methods.
#![allow(dead_code)]

use std::path::PathBuf;

/// Java `BrowsingDirectory`.
pub trait BrowsingDirectory {
    /// Java `getBrowsingDir`.  Returns a valid browsing directory.
    fn get_browsing_dir(&self) -> Option<PathBuf>;

    /// Java `setBrowsingDir(File)`.  Attempts to set a valid browsing directory from
    /// file.
    fn set_browsing_dir(&self, file: Option<&std::path::Path>);
}
