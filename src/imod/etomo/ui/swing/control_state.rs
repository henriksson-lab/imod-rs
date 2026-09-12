//! `IMOD/Etomo/src/etomo/ui/swing/ControlState.java`.
//!
//! This source unit is a stateful `ControlMode`.  It only describes which
//! control presentation the Swing boundary must apply; it does not render or
//! mutate a GUI component itself.
#![allow(dead_code)]

use std::ops::Deref;
use std::sync::LazyLock;

use crate::imod::etomo::ui::shared_strings;

use super::control_mode::ControlMode;

/// Java private static final `ControlState.DisplayType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DisplayType {
    Override,
    Enable,
}

/// Java package-private `ControlState extends ControlMode`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ControlState {
    control_mode: ControlMode,
    control_string: Option<String>,
    display_type: DisplayType,
}

impl ControlState {
    /// Java package-private `ControlState(String, String, DisplayType)`.
    pub(crate) fn new(
        field_name: Option<String>,
        control_string: Option<String>,
        display_type: DisplayType,
    ) -> Self {
        Self {
            control_mode: ControlMode::new(field_name),
            control_string,
            display_type,
        }
    }

    /// Java `isComponentDisplay()`.
    pub(crate) fn is_component_display(&self) -> bool {
        self.display_type == DisplayType::Override
    }

    /// Java `isEnableDisplay()`.
    pub(crate) fn is_enable_display(&self) -> bool {
        self.display_type == DisplayType::Enable
    }

    /// Java `getControlString()`.
    pub(crate) fn get_control_string(&self) -> Option<&str> {
        self.control_string.as_deref()
    }

    /// Java package-private static `ControlState.OVERRIDE`.
    pub(crate) const OVERRIDE: LazyLock<Self> = LazyLock::new(|| {
        Self::new(
            Some("override".to_owned()),
            Some(shared_strings::OVERRIDE_TEXT.to_owned()),
            DisplayType::Override,
        )
    });

    /// Java package-private static `ControlState.ENABLE`.
    pub(crate) const ENABLE: LazyLock<Self> =
        LazyLock::new(|| Self::new(Some("enable".to_owned()), None, DisplayType::Enable));
}

/// Preserve the Java `extends ControlMode` relationship without duplicating
/// inherited `ControlMode` methods in this source unit.
impl Deref for ControlState {
    type Target = ControlMode;

    fn deref(&self) -> &Self::Target {
        &self.control_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_state_keeps_its_control_mode_and_override_display() {
        assert_eq!(ControlState::OVERRIDE.get_field_name(), Some("override"));
        assert_eq!(ControlState::OVERRIDE.to_string(), Some("override"));
        assert!(ControlState::OVERRIDE.has_field_name());
        assert!(ControlState::OVERRIDE.is_component_display());
        assert!(!ControlState::OVERRIDE.is_enable_display());
        assert_eq!(
            ControlState::OVERRIDE.get_control_string(),
            Some(shared_strings::OVERRIDE_TEXT)
        );
    }

    #[test]
    fn enable_state_keeps_java_null_control_string_and_enable_display() {
        assert_eq!(ControlState::ENABLE.get_field_name(), Some("enable"));
        assert!(!ControlState::ENABLE.is_component_display());
        assert!(ControlState::ENABLE.is_enable_display());
        assert_eq!(ControlState::ENABLE.get_control_string(), None);
    }

    #[test]
    fn constructor_preserves_nullable_control_string_separately_from_field_name() {
        let state = ControlState::new(None, Some("control".to_owned()), DisplayType::Enable);
        assert_eq!(state.get_field_name(), None);
        assert_eq!(state.get_control_string(), Some("control"));
        assert!(state.is_enable_display());
    }
}
