//! `IMOD/Etomo/src/etomo/storage/SerialSectionsFileFilter.java`.
#![allow(dead_code)]

use crate::imod::etomo::r#type::data_file_type::DataFileType;
use std::path::Path;

/// Java final `SerialSectionsFileFilter`, whose `DataFileFilter` superclass
/// contributes no instance fields used by this source unit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SerialSectionsFileFilter;

impl SerialSectionsFileFilter {
    /// Java implicit `SerialSectionsFileFilter()` constructor.
    pub fn new() -> Self {
        Self
    }

    /// Java override `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        if file.is_dir() {
            return false;
        }
        let Some(file_name) = file.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        file_name.ends_with(DataFileType::SerialSections.extension().unwrap())
            && file_name.len() > 4
    }

    /// Java override `getDescription()`.
    pub fn get_description(&self) -> String {
        format!(
            "Serial sections data file ({})",
            DataFileType::SerialSections.extension().unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn accepts_only_nonempty_serial_sections_files() {
        let root = std::env::temp_dir().join(format!(
            "imod_rs_serial_sections_filter_{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("sections.ess");
        fs::write(&file, []).unwrap();
        assert!(SerialSectionsFileFilter::new().accept(&file));
        assert!(!SerialSectionsFileFilter::new().accept(&root));
        assert!(!SerialSectionsFileFilter::new().accept(&root.join(".ess")));
        fs::remove_file(file).unwrap();
        fs::remove_dir(root).unwrap();
    }
    #[test]
    fn description_matches_java() {
        assert_eq!(
            SerialSectionsFileFilter::new().get_description(),
            "Serial sections data file (.ess)"
        );
    }
}
