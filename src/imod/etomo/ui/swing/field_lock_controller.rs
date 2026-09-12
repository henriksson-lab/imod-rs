//! `IMOD/Etomo/src/etomo/ui/swing/FieldLockController.java`.
//!
//! Swing widgets are represented as explicit state boundaries.  The controller
//! retains the Java source unit's state transitions, including the distinction
//! between a text component's enabled and editable states.
#![allow(dead_code)]

/// State used by Java `JToggleButton` at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JToggleButton {
    pub enabled: bool,
    pub selected: bool,
}

/// State used by Java `JButton` at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JButton {
    pub enabled: bool,
}

/// State used by Java `JTextComponent` at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JTextComponent {
    pub enabled: bool,
    pub editable: bool,
}

/// State used by Java `JSpinner` at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JSpinner {
    pub enabled: bool,
}

/// Java package-private final `FieldLockController`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldLockController {
    pub toggle_button: Option<JToggleButton>,
    pub button: Option<JButton>,
    pub text_component: Option<JTextComponent>,
    pub spinner: Option<JSpinner>,
    pub read_only: bool,
    pub debug: bool,
    pub enabled: bool,
    pub editable: bool,
    pub locked: bool,
    checkpoint: Option<InternalState>,
}

impl FieldLockController {
    /// Java private `FieldLockController(JToggleButton, JButton, JTextComponent, JSpinner, boolean, boolean)`.
    fn new(
        toggle_button: Option<JToggleButton>,
        button: Option<JButton>,
        editable_text_component: Option<JTextComponent>,
        spinner: Option<JSpinner>,
        read_only: bool,
        debug: bool,
    ) -> Self {
        let mut value = Self {
            toggle_button,
            button,
            text_component: editable_text_component,
            spinner,
            read_only,
            debug,
            enabled: true,
            editable: true,
            locked: false,
            checkpoint: None,
        };
        value.apply_state();
        value
    }

    /// Java `getToggleButtonInstance(JToggleButton)`.
    pub fn get_toggle_button_instance(toggle_button: JToggleButton) -> Self {
        Self::new(Some(toggle_button), None, None, None, false, false)
    }

    /// Java `getToggleButtonInstance(JToggleButton, boolean)`.
    pub fn get_toggle_button_debug_instance(toggle_button: JToggleButton, debug: bool) -> Self {
        Self::new(Some(toggle_button), None, None, None, false, debug)
    }

    /// Java `getButtonInstance(JButton)`.
    pub fn get_button_instance(button: JButton) -> Self {
        Self::new(None, Some(button), None, None, false, false)
    }

    /// Java `getTextComponentInstance(JTextComponent)`.
    pub fn get_text_component_instance(text_component: JTextComponent) -> Self {
        Self::new(None, None, Some(text_component), None, false, false)
    }

    /// Java `getTextComponentInstance(JTextComponent, boolean)`.
    pub fn get_text_component_read_only_instance(
        text_component: JTextComponent,
        read_only: bool,
    ) -> Self {
        Self::new(None, None, Some(text_component), None, read_only, false)
    }

    /// Java `getSpinnerInstance(JSpinner)`.
    pub fn get_spinner_instance(spinner: JSpinner) -> Self {
        Self::new(None, None, None, Some(spinner), false, false)
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&mut self, locked: bool) -> bool {
        self.locked = locked;
        self.apply_state()
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) -> bool {
        self.editable = editable;
        self.apply_state()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) -> bool {
        self.enabled = enabled;
        self.apply_state()
    }

    /// Java `applyToggleButtonSelectionState()`.
    pub fn apply_toggle_button_selection_state(&mut self) {
        self.apply_state();
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.editable
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Java `applyState()`.
    fn apply_state(&mut self) -> bool {
        let changed = self.apply_state_toggle_button(true);
        let can_enable = !matches!(
            self.toggle_button.as_ref(),
            Some(toggle_button) if !toggle_button.enabled || !toggle_button.selected
        );
        let changed = self.apply_state_button(can_enable) || changed;
        let changed = self.apply_state_text_component(can_enable) || changed;
        self.apply_state_spinner(can_enable) || changed
    }

    /// Java `applyState(Component, boolean)`, specialized for `JToggleButton`.
    fn apply_state_toggle_button(&mut self, can_enable: bool) -> bool {
        let Some(toggle_button) = &mut self.toggle_button else {
            return false;
        };
        let previous_enabled = toggle_button.enabled;
        toggle_button.enabled =
            can_enable && !self.read_only && !self.locked && self.editable && self.enabled;
        previous_enabled != toggle_button.enabled
    }

    /// Java `applyState(Component, boolean)`, specialized for `JButton`.
    fn apply_state_button(&mut self, can_enable: bool) -> bool {
        let Some(button) = &mut self.button else {
            return false;
        };
        let previous_enabled = button.enabled;
        button.enabled =
            can_enable && !self.read_only && !self.locked && self.editable && self.enabled;
        previous_enabled != button.enabled
    }

    /// Java `applyState(JTextComponent, boolean)`.
    fn apply_state_text_component(&mut self, can_enable: bool) -> bool {
        let Some(text_component) = &mut self.text_component else {
            return false;
        };
        let previous_enabled = text_component.enabled;
        text_component.enabled = can_enable && self.enabled;
        let changed = previous_enabled != text_component.enabled;
        let previous_editable = text_component.editable;
        text_component.editable = !self.read_only && !self.locked && self.editable;
        previous_editable != text_component.editable || changed
    }

    /// Java `applyState(Component, boolean)`, specialized for `JSpinner`.
    fn apply_state_spinner(&mut self, can_enable: bool) -> bool {
        let Some(spinner) = &mut self.spinner else {
            return false;
        };
        let previous_enabled = spinner.enabled;
        spinner.enabled =
            can_enable && !self.read_only && !self.locked && self.editable && self.enabled;
        previous_enabled != spinner.enabled
    }

    /// Java `toChangedInternalStateString()`.
    pub fn to_changed_internal_state_string(&mut self) -> Option<String> {
        let new_checkpoint = InternalState::new(self);
        if !self
            .checkpoint
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.equals(Some(&new_checkpoint)))
        {
            self.checkpoint = Some(new_checkpoint);
            return self
                .checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.to_string(self));
        }
        None
    }
}

/// Java private inner `InternalState`.
#[derive(Clone, Debug, Eq, PartialEq)]
struct InternalState {
    enabled: bool,
    editable: bool,
    locked: bool,
    toggle_button_enabled: Option<bool>,
    button_enabled: Option<bool>,
    text_component_enabled: Option<bool>,
    text_component_editable: Option<bool>,
    spinner_enabled: Option<bool>,
}

impl InternalState {
    /// Java private `InternalState(boolean, boolean, boolean)`.
    fn new(field_lock_controller: &FieldLockController) -> Self {
        Self {
            enabled: field_lock_controller.enabled,
            editable: field_lock_controller.editable,
            locked: field_lock_controller.locked,
            toggle_button_enabled: field_lock_controller
                .toggle_button
                .as_ref()
                .map(|toggle_button| toggle_button.enabled),
            button_enabled: field_lock_controller
                .button
                .as_ref()
                .map(|button| button.enabled),
            text_component_enabled: field_lock_controller
                .text_component
                .as_ref()
                .map(|text_component| text_component.enabled),
            text_component_editable: field_lock_controller
                .text_component
                .as_ref()
                .map(|text_component| text_component.editable),
            spinner_enabled: field_lock_controller
                .spinner
                .as_ref()
                .map(|spinner| spinner.enabled),
        }
    }

    /// Java `equals(InternalState)`.
    fn equals(&self, internal_state: Option<&Self>) -> bool {
        let Some(internal_state) = internal_state else {
            return false;
        };
        self == internal_state
    }

    /// Java `toString()`.
    fn to_string(&self, field_lock_controller: &FieldLockController) -> String {
        let mut description = format!(
            "[readOnly:{},enabled:{},editable:{},locked:{}",
            field_lock_controller.read_only, self.enabled, self.editable, self.locked
        );
        if let Some(toggle_button_enabled) = self.toggle_button_enabled {
            description.push_str(&format!("\n[toggleButton enabled:{toggle_button_enabled}]"));
        }
        if let Some(button_enabled) = self.button_enabled {
            description.push_str(&format!("\n[button enabled:{button_enabled}]"));
        }
        if let (Some(text_component_enabled), Some(text_component_editable)) =
            (self.text_component_enabled, self.text_component_editable)
        {
            description.push_str(&format!(
                "\n[textComponent enabled:{text_component_enabled},editable:{text_component_editable}]"
            ));
        }
        if let Some(spinner_enabled) = self.spinner_enabled {
            description.push_str(&format!("\n[spinner enabled:{spinner_enabled}]"));
        }
        description.push(']');
        description
    }
}

#[cfg(test)]
mod tests {
    use super::{FieldLockController, JButton, JSpinner, JTextComponent, JToggleButton};

    #[test]
    fn text_component_keeps_enabled_and_editable_state_distinct() {
        let mut controller = FieldLockController::get_text_component_instance(JTextComponent {
            enabled: true,
            editable: true,
        });
        assert!(controller.set_locked(true));
        assert_eq!(
            controller.text_component.as_ref(),
            Some(&JTextComponent {
                enabled: true,
                editable: false,
            })
        );
        assert!(controller.set_enabled(false));
        assert_eq!(
            controller.text_component.as_ref(),
            Some(&JTextComponent {
                enabled: false,
                editable: false,
            })
        );
    }

    #[test]
    fn read_only_text_component_can_be_enabled_without_becoming_editable() {
        let mut controller = FieldLockController::get_text_component_read_only_instance(
            JTextComponent {
                enabled: true,
                editable: true,
            },
            true,
        );
        assert_eq!(
            controller.text_component.as_ref(),
            Some(&JTextComponent {
                enabled: true,
                editable: false,
            })
        );
        assert!(controller.set_enabled(false));
        assert!(controller.set_enabled(true));
        assert!(!controller.text_component.as_ref().unwrap().editable);
    }

    #[test]
    fn toggle_selection_controls_other_components() {
        let mut controller = FieldLockController::new(
            Some(JToggleButton {
                enabled: true,
                selected: true,
            }),
            Some(JButton { enabled: true }),
            None,
            Some(JSpinner { enabled: true }),
            false,
            false,
        );
        controller.toggle_button.as_mut().unwrap().selected = false;
        controller.apply_toggle_button_selection_state();
        assert!(controller.toggle_button.as_ref().unwrap().enabled);
        assert!(!controller.button.as_ref().unwrap().enabled);
        assert!(!controller.spinner.as_ref().unwrap().enabled);
    }

    #[test]
    fn changed_internal_state_matches_java_format_and_only_reports_changes() {
        let mut controller = FieldLockController::get_button_instance(JButton { enabled: true });
        assert_eq!(
            controller.to_changed_internal_state_string(),
            Some(
                "[readOnly:false,enabled:true,editable:true,locked:false\n[button enabled:true]]"
                    .to_owned()
            )
        );
        assert_eq!(controller.to_changed_internal_state_string(), None);
        controller.set_locked(true);
        assert_eq!(
            controller.to_changed_internal_state_string(),
            Some(
                "[readOnly:false,enabled:true,editable:true,locked:true\n[button enabled:false]]"
                    .to_owned()
            )
        );
    }
}
