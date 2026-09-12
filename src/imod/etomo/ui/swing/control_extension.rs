//! `IMOD/Etomo/src/etomo/ui/swing/ControlExtension.java`.
//!
//! This deprecated source unit has no implementation: its sole public method
//! returns Java `null`.  `ControlTarget` operations are retained as an
//! explicit Swing boundary so that the source-shaped signature is available
//! without inventing native UI behavior.
#![allow(dead_code)]

use std::path::PathBuf;

use super::combo_box_efield::ControlState;

/// Java `ControlTarget`; the target itself belongs to the Swing boundary.
pub trait ControlTarget {
    fn clear(&mut self);
    fn set_text_file(&mut self, file: PathBuf);
    fn set_text_files(&mut self, files: Vec<PathBuf>);
    fn get_label(&self) -> Option<String>;
    fn set_component_control(&mut self, control: bool, state: ControlState);
    fn set_enable_control(&mut self, control: bool, state: ControlState);
    fn send_control_event(&mut self);
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool;
}

/// Java package-private `ControlExtension`.
///
/// The Java constructor stores no state and its `ControlState` argument is
/// deliberately unused.
pub struct ControlExtension;

impl ControlExtension {
    /// Java private `ControlExtension(ControlState)`.
    fn new(_control_state: &ControlState) -> Self {
        Self
    }

    /// Java `setControlled(ControlTarget, boolean, ControlState, ControlState)`.
    ///
    /// The deprecated Java implementation returns `null`; Rust represents
    /// that exact absent result with `None`.
    #[deprecated(note = "ControlExtension was replaced by ControlState in IMOD")]
    pub fn set_controlled<T: ControlTarget>(
        &self,
        _target: &mut T,
        _controlled: bool,
        _cur_state: &ControlState,
        _new_state: &ControlState,
    ) -> Option<ControlState> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Target;

    impl ControlTarget for Target {
        fn clear(&mut self) {}
        fn set_text_file(&mut self, _file: PathBuf) {}
        fn set_text_files(&mut self, _files: Vec<PathBuf>) {}
        fn get_label(&self) -> Option<String> {
            None
        }
        fn set_component_control(&mut self, _control: bool, _state: ControlState) {}
        fn set_enable_control(&mut self, _control: bool, _state: ControlState) {}
        fn send_control_event(&mut self) {}
        fn is_local_dir(&self, _current_directory: Option<&str>) -> bool {
            false
        }
    }

    #[test]
    #[allow(deprecated)]
    fn set_controlled_preserves_java_null_result() {
        let extension = ControlExtension::new(&ControlState::Override);
        let mut target = Target;
        assert_eq!(
            extension.set_controlled(
                &mut target,
                true,
                &ControlState::Override,
                &ControlState::Enable,
            ),
            None
        );
    }
}
