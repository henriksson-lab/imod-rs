//! `IMOD/Etomo/src/etomo/storage/ModelFileFilter.java`.
//!
//! The file chooser filter for 3dmod model files (`.mod`).

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class ModelFileFilter extends FileFilter implements
/// java.io.FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelFileFilter;

impl ModelFileFilter {
    /// Java implicit constructor.
    pub fn new() -> ModelFileFilter {
        ModelFileFilter
    }
}

impl FileFilter for ModelFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, f: &Path) -> bool {
        let file_path = utilities::java_io_file_get_absolute_path(&f.to_string_lossy());
        // If this is a file test its extension, all others should return true
        if f.is_file() && !file_path.ends_with(dataset_files::MODEL_EXT) {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("Model file".to_string())
    }
}
