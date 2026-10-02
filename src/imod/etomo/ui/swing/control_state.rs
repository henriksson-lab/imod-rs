//! `IMOD/Etomo/src/etomo/ui/swing/ControlState.java`.
//!
//! `final class ControlState extends ControlMode`: the `ControlMode` part is the
//! embedded `base`, reached through `Deref`.  The constructor is effectively
//! private to this file (its `DisplayType` argument is a private nested class),
//! so `OVERRIDE` and `ENABLE` are the only instances there are; `ControlMediator`
//! relies on that for its `instanceof ControlState` test.

use std::ops::Deref;
use std::sync::LazyLock;

use super::control_mode::ControlMode;
use crate::imod::etomo::ui::shared_strings;

/// Java `static ControlState OVERRIDE =
/// new ControlState("override", SharedStrings.OVERRIDE_TEXT, DisplayType.OVERRIDE)`.
pub static OVERRIDE: LazyLock<ControlState> = LazyLock::new(|| {
    ControlState::new(
        Some("override"),
        Some(shared_strings::OVERRIDE_TEXT),
        DisplayType::Override,
    )
});

/// Java `static ControlState ENABLE = new ControlState("enable", null, DisplayType.ENABLE)`.
pub static ENABLE: LazyLock<ControlState> =
    LazyLock::new(|| ControlState::new(Some("enable"), None, DisplayType::Enable));

/// Java `private static final class DisplayType` with its two instances
/// `OVERRIDE` and `ENABLE`, compared by identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DisplayType {
    Override,
    Enable,
}

/// Java `final class ControlState extends ControlMode`.
#[derive(Debug)]
pub struct ControlState {
    base: ControlMode,
    /// Java `private final String controlString`.
    control_string: Option<String>,
    /// Java `private final DisplayType displayType`.
    display_type: DisplayType,
}

impl Deref for ControlState {
    type Target = ControlMode;
    fn deref(&self) -> &ControlMode {
        &self.base
    }
}

impl ControlState {
    /// Java package-private `ControlState(String, String, DisplayType)`.
    fn new(
        field_name: Option<&str>,
        control_string: Option<&str>,
        display_type: DisplayType,
    ) -> ControlState {
        ControlState {
            base: ControlMode::new(field_name),
            control_string: control_string.map(str::to_owned),
            display_type,
        }
    }

    /// Java package-private `isComponentDisplay()`.
    pub fn is_component_display(&self) -> bool {
        self.display_type == DisplayType::Override
    }

    /// Java package-private `isEnableDisplay()`.
    pub fn is_enable_display(&self) -> bool {
        self.display_type == DisplayType::Enable
    }

    /// Java package-private `getControlString()`.
    pub fn get_control_string(&self) -> Option<&str> {
        self.control_string.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_states() {
        assert!(OVERRIDE.is_component_display());
        assert!(!OVERRIDE.is_enable_display());
        assert_eq!(OVERRIDE.get_control_string(), Some(">OVERRIDE<"));
        assert_eq!(OVERRIDE.get_field_name(), Some("override"));
        assert!(ENABLE.is_enable_display());
        assert_eq!(ENABLE.get_control_string(), None);
    }
}
