//! `IMOD/Etomo/src/etomo/ui/swing/ControlMediator.java`.
//!
//! ControlMediator allows a controller and its target to communicate.

use super::control_mode::{self, ControlMode};
use super::control_state::{self, ControlState};
use super::control_target::ControlTarget;
use super::controller::Controller;

/// Java `static ControlMediator INSTANCE = new ControlMediator()`.
pub static INSTANCE: ControlMediator = ControlMediator {};

/// Java package-private `final class ControlMediator` (no fields).
pub struct ControlMediator {}

impl ControlMediator {
    /// Java `controlEvent(Controller, ControlTarget, ControlMode)`.
    ///
    /// Execute a control event.  After this is done the target is notified so
    /// so it can send control events to its listeners.
    pub fn control_event_controller_control_target_control_mode(
        &self,
        controller: &dyn Controller,
        target: Option<&dyn ControlTarget>,
        mode: Option<&ControlMode>,
    ) {
        // `mode instanceof ControlState`: OVERRIDE and ENABLE are the only
        // ControlState instances (see control_state.rs), so the test is whether
        // `mode` is the ControlMode part of one of them.
        let state: Option<&'static ControlState> = match mode {
            Some(mode) if std::ptr::eq(mode, &**control_state::OVERRIDE) => {
                Some(&*control_state::OVERRIDE)
            }
            Some(mode) if std::ptr::eq(mode, &**control_state::ENABLE) => {
                Some(&*control_state::ENABLE)
            }
            _ => None,
        };
        if let Some(state) = state {
            self.control_event_controller_control_target_control_state(
                Some(controller),
                target,
                Some(state),
            );
            return;
        }
        let Some(target) = target else {
            return;
        };
        if mode.is_some_and(|mode| std::ptr::eq(mode, &*control_mode::CLEAR)) {
            target.clear();
        } else if mode.is_some_and(|mode| std::ptr::eq(mode, &*control_mode::SELECT_FILE)) {
            let file = controller.select_file();
            if let Some(file) = file {
                target.set_text_file(Some(&file));
            }
        } else if mode
            .is_some_and(|mode| std::ptr::eq(mode, &*control_mode::SELECT_MULTIPLE_FILES))
        {
            let files = controller.select_multiple_files();
            if let Some(files) = files {
                target.set_text_file_array(Some(&files));
            }
        }
        target.send_control_event();
    }

    /// Java `controlEvent(Controller, ControlTarget, ControlState)`.
    ///
    /// Set a control state on or off.  After this is done the target is
    /// notified so so it can send control events to its listeners.
    pub fn control_event_controller_control_target_control_state(
        &self,
        controller: Option<&dyn Controller>,
        target: Option<&dyn ControlTarget>,
        state: Option<&'static ControlState>,
    ) {
        let (Some(controller), Some(state), Some(target)) = (controller, state, target) else {
            return;
        };
        // Java: `controller != null && controller.isControl()`.
        let control = controller.is_control();
        if state.is_component_display() {
            target.set_component_control(control, Some(state));
        } else if state.is_enable_display() {
            target.set_enable_control(control, Some(state));
            return;
        }
        target.send_control_event();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::path::{Path, PathBuf};

    struct C(bool);
    impl Controller for C {
        fn is_control(&self) -> bool {
            self.0
        }
        fn set_editable(&self, _: bool) {}
        fn set_enabled(&self, _: bool) {}
        fn select_file(&self) -> Option<PathBuf> {
            Some(PathBuf::from("/tmp/x"))
        }
        fn select_multiple_files(&self) -> Option<Vec<PathBuf>> {
            None
        }
    }
    #[derive(Default)]
    struct T {
        log: RefCell<Vec<String>>,
        sent: Cell<i32>,
    }
    impl ControlTarget for T {
        fn clear(&self) {
            self.log.borrow_mut().push("clear".into());
        }
        fn set_text_file(&self, file: Option<&Path>) {
            let file = file.unwrap();
            self.log.borrow_mut().push(file.display().to_string());
        }
        fn set_text_file_array(&self, _: Option<&[PathBuf]>) {}
        fn get_label(&self) -> Option<String> {
            None
        }
        fn set_component_control(&self, control: bool, _: Option<&'static ControlState>) {
            self.log.borrow_mut().push(format!("component {control}"));
        }
        fn set_enable_control(&self, control: bool, _: Option<&'static ControlState>) {
            self.log.borrow_mut().push(format!("enable {control}"));
        }
        fn send_control_event(&self) {
            self.sent.set(self.sent.get() + 1);
        }
        fn is_local_dir(&self, _: Option<&str>) -> bool {
            true
        }
    }

    #[test]
    fn modes_and_states_dispatch_like_the_java() {
        let t = T::default();
        let c = C(true);
        INSTANCE.control_event_controller_control_target_control_mode(
            &c,
            Some(&t),
            Some(&*control_mode::SELECT_FILE),
        );
        INSTANCE.control_event_controller_control_target_control_mode(
            &c,
            Some(&t),
            Some(&**control_state::ENABLE),
        );
        INSTANCE.control_event_controller_control_target_control_mode(
            &c,
            Some(&t),
            Some(&**control_state::OVERRIDE),
        );
        assert_eq!(
            *t.log.borrow(),
            vec!["/tmp/x", "enable true", "component true"]
        );
        // SELECT_FILE and OVERRIDE send; ENABLE returns before sending.
        assert_eq!(t.sent.get(), 2);
    }
}
