//! `IMOD/Etomo/src/etomo/storage/PeetFileFilter.java`.
//!
//! The Swing `FileFilter`/`java.io.FileFilter` parents only define the two
//! methods implemented by this unit, so they are represented as inherent Rust
//! methods.
#![allow(dead_code)]

use crate::imod::etomo::r#type::data_file_type::DataFileType;
use std::path::Path;

/// Java `PeetFileFilter extends DataFileFilter`.
pub struct PeetFileFilter {
    /// Java final `acceptDirectories`.
    accept_directories: bool,
}

impl PeetFileFilter {
    /// Java `PeetFileFilter()`.
    pub fn new() -> Self {
        Self {
            accept_directories: true,
        }
    }

    /// Java `PeetFileFilter(boolean)`.
    pub fn new_with_accept_directories(accept_directories: bool) -> Self {
        Self { accept_directories }
    }

    /// Java `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        if file.is_dir() {
            return self.accept_directories;
        }
        let file_name = file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        file_name.ends_with(DataFileType::Peet.extension().unwrap()) && file_name.len() > 4
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> String {
        format!(
            "PEET data file ({})",
            DataFileType::Peet.extension().unwrap()
        )
    }
}

impl Default for PeetFileFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn accepts_only_nonempty_peet_file_names_and_configured_directories() {
        assert!(PeetFileFilter::new().accept(Path::new("particles.epe")));
        assert!(!PeetFileFilter::new().accept(Path::new(".epe")));
        assert!(!PeetFileFilter::new().accept(Path::new("particles.edf")));
        assert_eq!(
            PeetFileFilter::new().get_description(),
            "PEET data file (.epe)"
        );
    }
}
