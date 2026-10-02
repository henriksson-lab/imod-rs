//! `IMOD/Etomo/src/etomo/ui/swing/ActionTarget.java`.
//!
//! The source declares the small package-local contract used by a file-browse
//! button: the selected file is assigned to its target, and the target exposes
//! its expanded text.  `File` values can be null at this boundary, hence the
//! `Option<&Path>` argument.
#![allow(dead_code)]

use std::path::Path;

/// Java `ActionTarget`.
///
/// This is an explicit GUI boundary.  Swing owns the file chooser and calls
/// these methods after the user has selected (or cleared) a file; Rust keeps
/// the source interface and does not provide a chooser substitute here.
pub trait ActionTarget {
    /// Java `setTargetFile(File)`.
    fn set_target_file(&self, file: Option<&Path>);

    /// Java `getExpandedValue()`.
    fn get_expanded_value(&self) -> Option<String>;
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::ActionTarget;

    struct Target {
        target_file: std::cell::RefCell<Option<String>>,
    }

    impl ActionTarget for Target {
        fn set_target_file(&self, file: Option<&Path>) {
            *self.target_file.borrow_mut() = file.map(|file| file.display().to_string());
        }

        fn get_expanded_value(&self) -> Option<String> {
            self.target_file.borrow().clone()
        }
    }

    #[test]
    fn file_and_null_are_valid_target_values() {
        let target = Target {
            target_file: std::cell::RefCell::new(None),
        };
        target.set_target_file(Some(Path::new("/tmp/input.mrc")));
        assert_eq!(
            target.get_expanded_value().as_deref(),
            Some("/tmp/input.mrc")
        );
        target.set_target_file(None);
        assert_eq!(target.get_expanded_value(), None);
    }
}
