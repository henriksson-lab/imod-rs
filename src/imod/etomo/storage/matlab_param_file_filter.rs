//! `IMOD/Etomo/src/etomo/storage/MatlabParamFileFilter.java`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class MatlabParamFileFilter extends FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatlabParamFileFilter;

impl MatlabParamFileFilter {
    /// Java implicit constructor.
    pub fn new() -> MatlabParamFileFilter {
        MatlabParamFileFilter
    }
}

impl FileFilter for MatlabParamFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, f: &Path) -> bool {
        if f.is_file()
            && !utilities::java_io_file_get_absolute_path(&f.to_string_lossy())
                .ends_with(dataset_files::MATLAB_PARAM_FILE_EXT)
        {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some(format!(
            "MATLAB param file ({})",
            dataset_files::MATLAB_PARAM_FILE_EXT
        ))
    }
}
