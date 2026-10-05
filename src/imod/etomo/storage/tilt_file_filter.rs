//! `IMOD/Etomo/src/etomo/storage/TiltFileFilter.java`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public class TiltFileFilter extends FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TiltFileFilter;

impl TiltFileFilter {
    /// Java implicit constructor.
    pub fn new() -> TiltFileFilter {
        TiltFileFilter
    }
}

impl FileFilter for TiltFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, f: &Path) -> bool {
        if f.is_file()
            && !utilities::java_io_file_get_absolute_path(&f.to_string_lossy())
                .ends_with(dataset_files::TILT_FILE_EXT)
        {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some(format!("Tilt Angles File ({})", dataset_files::TILT_FILE_EXT))
    }
}
