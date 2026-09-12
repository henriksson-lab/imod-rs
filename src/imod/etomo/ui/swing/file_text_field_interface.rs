//! `IMOD/Etomo/src/etomo/ui/swing/FileTextFieldInterface.java`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The `javax.swing.filechooser.FileFilter` presentation boundary referenced
/// by `FileTextFieldInterface`.
pub trait FileFilter {
    fn accept(&self, file: &Path) -> bool;
    fn get_description(&self) -> String;
}

/// Java package-private `FileTextFieldInterface`.
///
/// Java permits the `File` and `FileFilter` return values to be null; their
/// direct Rust representations are consequently optional.
pub trait FileTextFieldInterface {
    /// Java `getFile()`.
    fn get_file(&self) -> Option<PathBuf>;

    /// Java `setFile(File)`.
    fn set_file(&mut self, file: Option<&Path>);

    /// Java `getFileFilter()`.
    fn get_file_filter(&self) -> Option<&dyn FileFilter>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Field(Option<PathBuf>);

    impl FileTextFieldInterface for Field {
        fn get_file(&self) -> Option<PathBuf> {
            self.0.clone()
        }

        fn set_file(&mut self, file: Option<&Path>) {
            self.0 = file.map(Path::to_path_buf);
        }

        fn get_file_filter(&self) -> Option<&dyn FileFilter> {
            None
        }
    }

    #[test]
    fn nullable_file_boundary_is_retained() {
        let mut field = Field::default();
        assert_eq!(field.get_file(), None);
        field.set_file(Some(Path::new("volume.rec")));
        assert_eq!(field.get_file(), Some(PathBuf::from("volume.rec")));
        field.set_file(None);
        assert_eq!(field.get_file(), None);
        assert!(field.get_file_filter().is_none());
    }
}
