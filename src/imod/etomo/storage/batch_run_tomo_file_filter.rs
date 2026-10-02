//! `IMOD/Etomo/src/etomo/storage/BatchRunTomoFileFilter.java`.

use crate::imod::etomo::r#type::data_file_type::DataFileType;
use std::path::Path;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java final `BatchRunTomoFileFilter` (a plain class, not a Swing filter).
#[derive(Default)]
pub struct BatchRunTomoFileFilter;

impl BatchRunTomoFileFilter {
    /// Java implicit `BatchRunTomoFileFilter()`.
    pub fn new() -> BatchRunTomoFileFilter {
        BatchRunTomoFileFilter
    }

    /// Java `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        if file.is_dir() {
            return true;
        }
        let file_name = file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if file_name.ends_with(DataFileType::BatchRunTomo.extension().unwrap())
            && file_name.chars().count() > 4
        {
            return true;
        }
        false
    }
}
