//! `IMOD/Etomo/src/etomo/ui/swing/ControlTarget.java`.
//!
//! This source unit is the package-private Swing target contract used by
//! `ControlMediator`.  File-picker presentation and listener delivery remain
//! explicit GUI boundaries; this trait deliberately supplies no replacement
//! GUI behavior.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::control_state::ControlState;

/// Java package-private `ControlTarget`.
pub trait ControlTarget {
    /// Java `clear()`.
    fn clear(&mut self);

    /// Java `setText(File)`.
    fn set_text_file(&mut self, file: &Path);

    /// Java `setText(File[])`.
    fn set_text_files(&mut self, files: &[PathBuf]);

    /// Java `getLabel()`.
    fn get_label(&self) -> String;

    /// Java `setComponentControl(boolean, ControlState)`.
    fn set_component_control(&mut self, control: bool, state: &ControlState);

    /// Java `setEnableControl(boolean, ControlState)`.
    fn set_enable_control(&mut self, control: bool, state: &ControlState);

    /// Java `sendControlEvent()`.
    fn send_control_event(&mut self);

    /// Java `isLocalDir(String)`.
    fn is_local_dir(&self, current_directory: &str) -> bool;
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{ControlState, ControlTarget};

    #[derive(Default)]
    struct Target {
        files: Vec<PathBuf>,
        component_control: Option<bool>,
        enable_control: Option<bool>,
        events: usize,
    }

    impl ControlTarget for Target {
        fn clear(&mut self) {
            self.files.clear();
        }

        fn set_text_file(&mut self, file: &Path) {
            self.files = vec![file.to_owned()];
        }

        fn set_text_files(&mut self, files: &[PathBuf]) {
            self.files = files.to_vec();
        }

        fn get_label(&self) -> String {
            "input files".to_owned()
        }

        fn set_component_control(&mut self, control: bool, _state: &ControlState) {
            self.component_control = Some(control);
        }

        fn set_enable_control(&mut self, control: bool, _state: &ControlState) {
            self.enable_control = Some(control);
        }

        fn send_control_event(&mut self) {
            self.events += 1;
        }

        fn is_local_dir(&self, current_directory: &str) -> bool {
            self.files.iter().all(|file| {
                file.parent()
                    .is_some_and(|parent| parent == Path::new(current_directory))
            })
        }
    }

    #[test]
    fn single_and_multiple_file_source_methods_are_distinct() {
        let mut target = Target::default();
        target.set_text_file(Path::new("/data/one.mrc"));
        assert_eq!(target.files, vec![PathBuf::from("/data/one.mrc")]);
        target.set_text_files(&[
            PathBuf::from("/data/two.mrc"),
            PathBuf::from("/data/three.mrc"),
        ]);
        assert_eq!(
            target.files,
            vec![
                PathBuf::from("/data/two.mrc"),
                PathBuf::from("/data/three.mrc")
            ]
        );
    }

    #[test]
    fn control_and_event_source_methods_remain_target_owned() {
        let mut target = Target::default();
        target.set_component_control(true, &ControlState::OVERRIDE);
        target.set_enable_control(false, &ControlState::ENABLE);
        target.send_control_event();
        assert_eq!(target.component_control, Some(true));
        assert_eq!(target.enable_control, Some(false));
        assert_eq!(target.events, 1);
        assert_eq!(target.get_label(), "input files");
    }

    #[test]
    fn local_directory_check_is_provided_by_the_target() {
        let mut target = Target::default();
        target.set_text_file(Path::new("/data/one.mrc"));
        assert!(target.is_local_dir("/data"));
        assert!(!target.is_local_dir("/other"));
        target.clear();
        assert!(target.files.is_empty());
    }
}
