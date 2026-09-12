//! `IMOD/Etomo/src/etomo/storage/EtomoFileFilter.java`.

use std::path::Path;

/// Java `EtomoFileFilter`, whose `DataFileFilter` superclass supplies only the
/// Swing file-filter type at this call site.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EtomoFileFilter;

impl EtomoFileFilter {
    /// Java `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        !file.is_dir()
            && (!file.is_file()
                || file.to_string_lossy().ends_with(
                    crate::imod::etomo::r#type::data_file_type::DataFileType::Recon
                        .extension()
                        .unwrap(),
                ))
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "Etomo data file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn accepts_only_reconstruction_data_files_and_not_directories() {
        let filter = EtomoFileFilter;
        assert!(filter.accept(Path::new("uncreated.edf")));
        assert_eq!(filter.get_description(), "Etomo data file");
    }
}
