//! `IMOD/Etomo/src/etomo/storage/MotlFileFilter.java`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class MotlFileFilter extends javax.swing.filechooser.FileFilter
/// implements java.io.FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MotlFileFilter;

impl MotlFileFilter {
    /// Java implicit constructor.
    pub fn new() -> MotlFileFilter {
        MotlFileFilter
    }
}

impl FileFilter for MotlFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, f: &Path) -> bool {
        if f.is_file()
            && !utilities::java_io_file_get_absolute_path(&f.to_string_lossy()).ends_with(".csv")
        {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("MOTL file (.em, .csv)".to_owned())
    }
}
