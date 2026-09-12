//! `IMOD/Etomo/src/etomo/storage/JoinFileFilter.java`.
//!
//! Swing's `FileFilter` superclass is a presentation boundary; this source
//! unit supplies the Join-specific acceptance predicate and description.
#![allow(dead_code)]

use crate::imod::etomo::r#type::data_file_type::DataFileType;
use std::path::Path;

/// Java `JoinFileFilter extends DataFileFilter`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JoinFileFilter;

impl JoinFileFilter {
    /// Java implicit `JoinFileFilter()` constructor.
    pub fn new() -> Self {
        Self
    }

    /// Java override `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        // Java returns true for every non-file, including directories and
        // nonexistent chooser candidates; only an existing file is filtered.
        !file.is_file()
            || file
                .to_string_lossy()
                .ends_with(DataFileType::Join.extension().unwrap())
    }

    /// Java override `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "Join data file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn accept_preserves_the_file_only_extension_test() {
        let root =
            std::env::temp_dir().join(format!("imod_rs_join_file_filter_{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let join = root.join("joined.ejf");
        let other = root.join("joined.edf");
        fs::write(&join, []).unwrap();
        fs::write(&other, []).unwrap();
        let filter = JoinFileFilter::new();
        assert!(filter.accept(&root));
        assert!(filter.accept(&join));
        assert!(!filter.accept(&other));
        fs::remove_file(join).unwrap();
        fs::remove_file(other).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn description_matches_java() {
        assert_eq!(JoinFileFilter::new().get_description(), "Join data file");
    }
}
