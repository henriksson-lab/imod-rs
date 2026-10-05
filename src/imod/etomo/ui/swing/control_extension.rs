//! `IMOD/Etomo/src/etomo/ui/swing/ControlExtension.java`.
//!
//! A deprecated, unused class (nothing in the Java constructs it): a private
//! constructor that ignores its argument and one method that returns `null`.

use super::control_state::ControlState;
use super::control_target::ControlTarget;

/// Java `final class ControlExtension`.
pub struct ControlExtension;

impl ControlExtension {
    /// Java private `ControlExtension(ControlState)`.
    #[allow(dead_code)]
    fn new(_control_state: Option<&'static ControlState>) -> ControlExtension {
        ControlExtension
    }

    /// Java `@deprecated public final ControlState setControlled(ControlTarget,
    /// boolean, ControlState, ControlState)`: returns `null`.
    pub fn set_controlled(
        &self,
        _target: Option<&dyn ControlTarget>,
        _controlled: bool,
        _cur_state: Option<&'static ControlState>,
        _new_state: Option<&'static ControlState>,
    ) -> Option<&'static ControlState> {
        None
    }
}
