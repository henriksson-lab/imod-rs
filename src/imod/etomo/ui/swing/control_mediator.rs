//! `IMOD/Etomo/src/etomo/ui/swing/ControlMediator.java`.
//!
//! Native file-picker presentation remains at the GUI boundary.  The mediator
//! keeps the Java dispatch and notification ordering intact.
#![allow(dead_code)]

use super::control_mode::{CLEAR, ControlMode, SELECT_FILE, SELECT_MULTIPLE_FILES};
use super::control_state::ControlState;
use super::control_target::ControlTarget;
use super::controller::Controller;

/// Java package-private singleton `ControlMediator`.
pub struct ControlMediator;

impl ControlMediator {
    /// Java `static ControlMediator INSTANCE`.
    pub const INSTANCE: Self = Self;

    /// Java `controlEvent(Controller, ControlTarget, ControlMode)`.
    ///
    /// The Java method recognizes `ControlState` through `instanceof`; Rust
    /// models the inherited source type with `Deref`, so the source overload
    /// is represented directly by [`Self::control_event_state`].
    pub fn control_event(
        &self,
        controller: &mut dyn Controller,
        target: Option<&mut dyn ControlTarget>,
        mode: &ControlMode,
    ) {
        let Some(target) = target else { return };
        // Java uses `==` for these singleton modes, i.e. reference identity,
        // not the value equality supplied by Rust's derived `PartialEq`.
        if std::ptr::eq(mode, &*CLEAR) {
            target.clear();
        } else if std::ptr::eq(mode, &*SELECT_FILE) {
            if let Some(file) = controller.select_file() {
                target.set_text_file(&file);
            }
        } else if std::ptr::eq(mode, &*SELECT_MULTIPLE_FILES) {
            if let Some(files) = controller.select_multiple_files() {
                target.set_text_files(&files);
            }
        }
        target.send_control_event();
    }

    /// Java `controlEvent(Controller, ControlTarget, ControlState)`.
    pub fn control_event_state(
        &self,
        controller: Option<&mut dyn Controller>,
        target: Option<&mut dyn ControlTarget>,
        state: Option<&ControlState>,
    ) {
        let (Some(controller), Some(target), Some(state)) = (controller, target, state) else {
            return;
        };
        let control = controller.is_control();
        if state.is_component_display() {
            target.set_component_control(control, state);
        } else if state.is_enable_display() {
            target.set_enable_control(control, state);
            return;
        }
        target.send_control_event();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[derive(Default)]
    struct TestController {
        control: bool,
        file: Option<PathBuf>,
        files: Option<Vec<PathBuf>>,
        select_file_count: usize,
        select_multiple_files_count: usize,
    }

    impl Controller for TestController {
        fn is_control(&self) -> bool {
            self.control
        }
        fn set_editable(&mut self, _editable: bool) {}
        fn set_enabled(&mut self, _enabled: bool) {}
        fn select_file(&mut self) -> Option<PathBuf> {
            self.select_file_count += 1;
            self.file.clone()
        }
        fn select_multiple_files(&mut self) -> Option<Vec<PathBuf>> {
            self.select_multiple_files_count += 1;
            self.files.clone()
        }
    }

    #[derive(Default)]
    struct TestTarget {
        clear_count: usize,
        files: Vec<PathBuf>,
        component_controls: Vec<bool>,
        enable_controls: Vec<bool>,
        send_count: usize,
    }

    impl ControlTarget for TestTarget {
        fn clear(&mut self) {
            self.clear_count += 1;
        }
        fn set_text_file(&mut self, file: &std::path::Path) {
            self.files = vec![file.to_owned()];
        }
        fn set_text_files(&mut self, files: &[PathBuf]) {
            self.files = files.to_vec();
        }
        fn get_label(&self) -> String {
            String::new()
        }
        fn set_component_control(&mut self, control: bool, _state: &ControlState) {
            self.component_controls.push(control);
        }
        fn set_enable_control(&mut self, control: bool, _state: &ControlState) {
            self.enable_controls.push(control);
        }
        fn send_control_event(&mut self) {
            self.send_count += 1;
        }
        fn is_local_dir(&self, _current_directory: &str) -> bool {
            false
        }
    }

    #[test]
    fn clear_and_file_modes_notify_after_their_target_change() {
        let mut controller = TestController {
            file: Some(PathBuf::from("one.mrc")),
            files: Some(vec![PathBuf::from("two.mrc"), PathBuf::from("three.mrc")]),
            ..Default::default()
        };
        let mut target = TestTarget::default();
        ControlMediator::INSTANCE.control_event(&mut controller, Some(&mut target), &CLEAR);
        ControlMediator::INSTANCE.control_event(&mut controller, Some(&mut target), &SELECT_FILE);
        ControlMediator::INSTANCE.control_event(
            &mut controller,
            Some(&mut target),
            &SELECT_MULTIPLE_FILES,
        );
        assert_eq!(target.clear_count, 1);
        assert_eq!(
            target.files,
            vec![PathBuf::from("two.mrc"), PathBuf::from("three.mrc")]
        );
        assert_eq!(target.send_count, 3);
        assert_eq!(controller.select_file_count, 1);
        assert_eq!(controller.select_multiple_files_count, 1);
    }

    #[test]
    fn null_picker_results_still_notify_like_java() {
        let mut controller = TestController::default();
        let mut target = TestTarget::default();
        ControlMediator::INSTANCE.control_event(&mut controller, Some(&mut target), &SELECT_FILE);
        ControlMediator::INSTANCE.control_event(
            &mut controller,
            Some(&mut target),
            &SELECT_MULTIPLE_FILES,
        );
        assert!(target.files.is_empty());
        assert_eq!(target.send_count, 2);
    }

    #[test]
    fn control_state_preserves_enable_early_return_and_override_notification() {
        let mut controller = TestController {
            control: true,
            ..Default::default()
        };
        let mut target = TestTarget::default();
        ControlMediator::INSTANCE.control_event_state(
            Some(&mut controller),
            Some(&mut target),
            Some(&ControlState::OVERRIDE),
        );
        ControlMediator::INSTANCE.control_event_state(
            Some(&mut controller),
            Some(&mut target),
            Some(&ControlState::ENABLE),
        );
        assert_eq!(target.component_controls, vec![true]);
        assert_eq!(target.enable_controls, vec![true]);
        assert_eq!(target.send_count, 1);
    }

    #[test]
    fn null_state_event_arguments_are_noops() {
        let mut target = TestTarget::default();
        ControlMediator::INSTANCE.control_event_state(
            None,
            Some(&mut target),
            Some(&ControlState::OVERRIDE),
        );
        assert_eq!(target.send_count, 0);
    }
}
