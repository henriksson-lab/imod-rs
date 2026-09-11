//! `IMOD/Etomo/src/etomo/storage/DataFileFilter.java`.

use crate::imod::etomo::r#type::data_file_type::DataFileType;
use std::path::Path;

pub struct DataFileFilter {
    files_only: bool,
}
impl DataFileFilter {
    /// Java `DataFileFilter()`.
    pub fn new() -> Self {
        Self { files_only: false }
    }
    /// Java `DataFileFilter(boolean)`.
    pub fn new_with_files_only(files_only: bool) -> Self {
        Self { files_only }
    }
    /// Java `accept(File)`; Swing's FileChooser role does not change its predicate.
    pub fn accept(&self, file: &Path) -> bool {
        if self.files_only && !file.is_file() {
            return false;
        }
        if file.is_file()
            && ![
                DataFileType::Recon,
                DataFileType::Join,
                DataFileType::Parallel,
                DataFileType::Peet,
                DataFileType::SerialSections,
                DataFileType::BatchRunTomo,
            ]
            .iter()
            .any(|file_type| {
                file.to_string_lossy()
                    .ends_with(file_type.extension().unwrap())
            })
        {
            return false;
        }
        true
    }
    /// Java `getDescription()`.
    pub fn get_description(&self) -> String {
        format!(
            "Data file ({}, {}, {}, {}, {}, {})",
            DataFileType::Recon.extension().unwrap(),
            DataFileType::Join.extension().unwrap(),
            DataFileType::Parallel.extension().unwrap(),
            DataFileType::Peet.extension().unwrap(),
            DataFileType::SerialSections.extension().unwrap(),
            DataFileType::BatchRunTomo.extension().unwrap()
        )
    }
}
impl Default for DataFileFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::DataFileFilter;
    use std::path::Path;
    #[test]
    fn accepts_known_etomo_data_extensions() {
        let filter = DataFileFilter::new();
        assert!(filter.accept(Path::new("not-created.edf")));
        assert_eq!(
            filter.get_description(),
            "Data file (.edf, .ejf, .epp, .epe, .ess, .ebt)"
        );
    }
}
