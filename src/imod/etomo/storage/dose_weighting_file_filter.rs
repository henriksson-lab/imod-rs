//! `IMOD/Etomo/src/etomo/storage/DoseWeightingFileFilter.java`.
//!
//! `DoseWeightingFileFilter extends javax.swing.filechooser.FileFilter`.  The
//! two abstract methods are inherent methods (as in the other storage file
//! filters) and are also the `jdk::FileFilter` implementation, so the filter
//! can be handed to a file chooser as `Rc<dyn FileFilter>`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;

/// Java `public final class DoseWeightingFileFilter extends FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DoseWeightingFileFilter {}

impl DoseWeightingFileFilter {
    /// Java's implicit `DoseWeightingFileFilter()`.
    pub fn new() -> DoseWeightingFileFilter {
        DoseWeightingFileFilter {}
    }

    /// Java `accept(File)`: whether the given file is accepted by this filter.
    pub fn accept(&self, file: Option<&Path>) -> bool {
        let Some(file) = file else {
            return false;
        };
        if !file.is_file() {
            return false;
        }
        // Java `file.getName()`: never null for a File built from a path.
        let name = file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        name.ends_with(".txt") || name.ends_with(".mdoc")
    }

    /// Java `getDescription()`: the description of this filter.
    pub fn get_description(&self) -> &'static str {
        "Dose information file (.mdoc,.txt)"
    }
}

impl FileFilter for DoseWeightingFileFilter {
    fn accept(&self, file: &Path) -> bool {
        DoseWeightingFileFilter::accept(self, Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(DoseWeightingFileFilter::get_description(self).to_string())
    }
}
