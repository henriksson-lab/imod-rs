//! `IMOD/Etomo/src/etomo/ui/swing/Controller.java`.
//!
//! This source unit is the small package-private controller contract shared by
//! `ControlMediator` and appearance-owning Swing components.  File selection
//! belongs to the native GUI boundary: Java can return `null` from either
//! chooser method, represented here by `None`.
#![allow(dead_code)]

use std::path::PathBuf;

/// Java package-private `Controller`.
///
/// Implementors own their editable/enabled state and their native chooser
/// presentation.  This trait intentionally does not invent a Rust picker.
pub trait Controller {
    /// Java `isControl()`.
    fn is_control(&self) -> bool;

    /// Java `setEditable(boolean)`.
    fn set_editable(&mut self, editable: bool);

    /// Java `setEnabled(boolean)`.
    fn set_enabled(&mut self, enabled: bool);

    /// Java `selectFile()`.
    fn select_file(&mut self) -> Option<PathBuf>;

    /// Java `selectMultipleFiles()`.
    fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>>;
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::Controller;

    #[derive(Default)]
    struct TestController {
        control: bool,
        editable: bool,
        enabled: bool,
        file: Option<PathBuf>,
        files: Option<Vec<PathBuf>>,
    }

    impl Controller for TestController {
        fn is_control(&self) -> bool {
            self.control
        }

        fn set_editable(&mut self, editable: bool) {
            self.editable = editable;
        }

        fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        fn select_file(&mut self) -> Option<PathBuf> {
            self.file.clone()
        }

        fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>> {
            self.files.clone()
        }
    }

    #[test]
    fn source_control_and_mutator_methods_remain_implementor_owned() {
        let mut controller = TestController {
            control: true,
            ..Default::default()
        };
        assert!(controller.is_control());
        controller.set_editable(true);
        controller.set_enabled(true);
        assert!(controller.editable);
        assert!(controller.enabled);
    }

    #[test]
    fn source_file_chooser_methods_preserve_java_null_and_file_results() {
        let mut controller = TestController::default();
        assert_eq!(controller.select_file(), None);
        assert_eq!(controller.select_multiple_files(), None);

        controller.file = Some(PathBuf::from("one.mrc"));
        controller.files = Some(vec![PathBuf::from("two.mrc"), PathBuf::from("three.mrc")]);
        assert_eq!(controller.select_file(), Some(PathBuf::from("one.mrc")));
        assert_eq!(
            controller.select_multiple_files(),
            Some(vec![PathBuf::from("two.mrc"), PathBuf::from("three.mrc")])
        );
    }
}
