//! `IMOD/Etomo/src/etomo/storage/PeetAndMatlabParamFileFilter.java`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class PeetAndMatlabParamFileFilter extends FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PeetAndMatlabParamFileFilter;

impl PeetAndMatlabParamFileFilter {
    /// Java implicit constructor.
    pub fn new() -> PeetAndMatlabParamFileFilter {
        PeetAndMatlabParamFileFilter
    }
}

impl FileFilter for PeetAndMatlabParamFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, f: &Path) -> bool {
        let absolute_path = utilities::java_io_file_get_absolute_path(&f.to_string_lossy());
        if f.is_file()
            && !absolute_path.ends_with(dataset_files::MATLAB_PARAM_FILE_EXT)
            && !absolute_path.ends_with(DataFileType::Peet.extension().unwrap_or("null"))
        {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some(format!(
            "PEET file or Matlap param file({}, {})",
            dataset_files::MATLAB_PARAM_FILE_EXT,
            DataFileType::Peet.extension().unwrap_or("null")
        ))
    }
}
