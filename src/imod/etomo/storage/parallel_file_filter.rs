//! `IMOD/Etomo/src/etomo/storage/ParallelFileFilter.java`.

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use std::path::Path;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ParallelFileFilter extends javax.swing.filechooser.FileFilter implements
/// java.io.FileFilter`.
#[derive(Default)]
pub struct ParallelFileFilter;

impl ParallelFileFilter {
    /// Java implicit `ParallelFileFilter()`.
    pub fn new() -> ParallelFileFilter {
        ParallelFileFilter
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
        if file_name.ends_with(DataFileType::Parallel.extension().unwrap())
            && file_name.chars().count() > 4
        {
            return true;
        }
        false
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> String {
        format!(
            "Parallel process data file ({})",
            DataFileType::Parallel.extension().unwrap()
        )
    }
}

impl FileFilter for ParallelFileFilter {
    fn accept(&self, file: &Path) -> bool {
        ParallelFileFilter::accept(self, file)
    }
    fn get_description(&self) -> Option<String> {
        Some(ParallelFileFilter::get_description(self))
    }
}
