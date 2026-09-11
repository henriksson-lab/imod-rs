//! `IMOD/Etomo/src/etomo/util/FileModifiedFlag.java`.
//!
//! A class to hold the last read state of a file.  Can be used to prevent unnecessary
//! reads.
//!
//! Copyright: Copyright (c) 2005
//!
//! Organization:
//! Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells (BL3DEM),
//! University of Colorado
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::LONG_NULL_VALUE;
use crate::imod::etomo::util::utilities::{
    java_io_file_get_absolute_path, java_io_file_last_modified,
};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private class `FileModifiedFlag`.  The `file` field is a `java.io.File`,
/// which is a path holder, so it is the path string here as it is everywhere else in the
/// translated `etomo` units.
#[derive(Clone, Debug)]
pub struct FileModifiedFlag {
    /// Java field `file`.
    file: String,
    /// Java field `lastModified`, initialised to `EtomoNumber.LONG_NULL_VALUE`.
    last_modified: i64,
}

impl FileModifiedFlag {
    /// Java `FileModifiedFlag(File)`.
    pub fn new(file: &str) -> FileModifiedFlag {
        FileModifiedFlag {
            file: file.to_string(),
            last_modified: LONG_NULL_VALUE,
        }
    }

    /// Java `isModifiedSinceLastRead`.
    pub fn is_modified_since_last_read(&self) -> bool {
        let file_last_modified = java_io_file_last_modified(&self.file);
        self.last_modified == LONG_NULL_VALUE || file_last_modified > self.last_modified
    }

    /// Java `setReadingNow`.
    pub fn set_reading_now(&mut self) {
        self.last_modified = java_io_file_last_modified(&self.file);
    }

    /// Java `getLastModified`.
    pub fn get_last_modified(&self) -> i64 {
        self.last_modified
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.last_modified = LONG_NULL_VALUE;
    }
}

/// Java `toString`.
impl std::fmt::Display for FileModifiedFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}",
            java_io_file_get_absolute_path(&self.file),
            self.last_modified
        )
    }
}
