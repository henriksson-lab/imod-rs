//! `IMOD/Etomo/src/etomo/ui/swing/FieldLockController.java`.
//!
//! Controls the real enabled and editable states of a field (a toggle button, a
//! button, a text component and/or a spinner) from its locked, editable and enabled
//! settings.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;

/// Java `FieldLockController`.
pub struct FieldLockController {
    /// Java `toggleButton` (optional).
    toggle_button: Option<Rc<JComponent>>,
    /// Java `button` (optional).
    button: Option<Rc<JComponent>>,
    /// Java `textComponent` (optional).
    text_component: Option<Rc<JComponent>>,
    /// Java `spinner` (optional).
    spinner: Option<Rc<JComponent>>,
    /// Java `readOnly`.  Controls the component's editable state if available.  If
    /// not, controls the component's enabled state.
    read_only: bool,
    /// Java `debug`.
    #[allow(dead_code)]
    debug: bool,
    // Controller state
    /// Java `enabled`.
    enabled: Cell<bool>,
    /// Java `editable`.
    editable: Cell<bool>,
    /// Java `locked`.
    locked: Cell<bool>,
    /// Java `checkpoint`.
    checkpoint: RefCell<Option<InternalState>>,
}

impl FieldLockController {
    /// Java private `FieldLockController(JToggleButton, JButton, JTextComponent,
    /// JSpinner, boolean, boolean)`.
    fn new(
        toggle_button: Option<Rc<JComponent>>,
        button: Option<Rc<JComponent>>,
        editable_text_component: Option<Rc<JComponent>>,
        spinner: Option<Rc<JComponent>>,
        read_only: bool,
        debug: bool,
    ) -> Rc<FieldLockController> {
        let controller = Rc::new(FieldLockController {
            toggle_button,
            button,
            text_component: editable_text_component,
            spinner,
            read_only,
            debug,
            enabled: Cell::new(true),
            editable: Cell::new(true),
            locked: Cell::new(false),
            checkpoint: RefCell::new(None),
        });
        controller.apply_state_void();
        controller
    }

    /// Java static `getToggleButtonInstance(JToggleButton)`.
    pub fn get_toggle_button_instance_j_toggle_button(
        toggle_button: &Rc<JComponent>,
    ) -> Rc<FieldLockController> {
        FieldLockController::new(Some(toggle_button.clone()), None, None, None, false, false)
    }

    /// Java static `getToggleButtonInstance(JToggleButton, boolean)`.
    pub fn get_toggle_button_instance_j_toggle_button_boolean(
        toggle_button: &Rc<JComponent>,
        debug: bool,
    ) -> Rc<FieldLockController> {
        FieldLockController::new(Some(toggle_button.clone()), None, None, None, false, debug)
    }

    /// Java static `getButtonInstance(JButton)`.
    pub fn get_button_instance(button: &Rc<JComponent>) -> Rc<FieldLockController> {
        FieldLockController::new(None, Some(button.clone()), None, None, false, false)
    }

    /// Java static `getTextComponentInstance(JTextComponent)`.
    pub fn get_text_component_instance_j_text_component(
        text_component: &Rc<JComponent>,
    ) -> Rc<FieldLockController> {
        FieldLockController::new(None, None, Some(text_component.clone()), None, false, false)
    }

    /// Java static `getTextComponentInstance(JTextComponent, boolean)`.
    pub fn get_text_component_instance_j_text_component_boolean(
        text_component: &Rc<JComponent>,
        read_only: bool,
    ) -> Rc<FieldLockController> {
        FieldLockController::new(None, None, Some(text_component.clone()), None, read_only, false)
    }

    /// Java static `getSpinnerInstance(JSpinner)`.
    pub fn get_spinner_instance(spinner: &Rc<JComponent>) -> Rc<FieldLockController> {
        FieldLockController::new(None, None, None, Some(spinner.clone()), false, false)
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) -> bool {
        self.locked.set(locked);
        self.apply_state_void()
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) -> bool {
        self.editable.set(editable);
        self.apply_state_void()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) -> bool {
        self.enabled.set(enabled);
        self.apply_state_void()
    }

    /// Java `applyToggleButtonSelectionState()`.
    pub fn apply_toggle_button_selection_state(&self) {
        self.apply_state_void();
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.locked.get()
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.editable.get()
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java private `applyState()`.  Applies the state of the readOnly, locked,
    /// editable, and enabled member variables.  If the toggle button and some other
    /// field are both present, the toggle button is assumed to control the other
    /// field.  Returns true if any field's editable or enabled state changed.
    fn apply_state_void(&self) -> bool {
        let mut changed = self.apply_state_component_boolean(self.toggle_button.as_ref(), true);
        let mut can_enable = true;
        if let Some(toggle_button) = &self.toggle_button {
            if !toggle_button.is_enabled() || !toggle_button.is_selected() {
                can_enable = false;
            }
        }
        changed = self.apply_state_component_boolean(self.button.as_ref(), can_enable) || changed;
        changed =
            self.apply_state_j_text_component_boolean(self.text_component.as_ref(), can_enable)
                || changed;
        self.apply_state_component_boolean(self.spinner.as_ref(), can_enable) || changed
    }

    /// Java private `applyState(Component, boolean)`.  Returns true if the enabled
    /// state changed.
    fn apply_state_component_boolean(
        &self,
        component: Option<&Rc<JComponent>>,
        can_enable: bool,
    ) -> bool {
        let Some(component) = component else {
            return false;
        };
        let prev_enabled = component.is_enabled();
        component.set_enabled(
            can_enable
                && !self.read_only
                && !self.locked.get()
                && self.editable.get()
                && self.enabled.get(),
        );
        prev_enabled != component.is_enabled()
    }

    /// Java private `applyState(JTextComponent, boolean)`.  Returns true if the enabled
    /// or editable state changed.
    fn apply_state_j_text_component_boolean(
        &self,
        text_component: Option<&Rc<JComponent>>,
        can_enable: bool,
    ) -> bool {
        let Some(text_component) = text_component else {
            return false;
        };
        let prev_enabled = text_component.is_enabled();
        text_component.set_enabled(can_enable && self.enabled.get());
        let changed = prev_enabled != text_component.is_enabled();
        let prev_editable = text_component.is_editable();
        text_component.set_editable(!self.read_only && !self.locked.get() && self.editable.get());
        (prev_editable != text_component.is_editable()) || changed
    }

    /// Java `toChangedInternalStateString()`.
    pub fn to_changed_internal_state_string(&self) -> Option<String> {
        let no_checkpoint = self.checkpoint.borrow().is_none();
        if no_checkpoint {
            let checkpoint = InternalState::new(
                self,
                self.enabled.get(),
                self.editable.get(),
                self.locked.get(),
            );
            let string = checkpoint.to_string(self);
            *self.checkpoint.borrow_mut() = Some(checkpoint);
            return Some(string);
        }
        let new_checkpoint =
            InternalState::new(self, self.enabled.get(), self.editable.get(), self.locked.get());
        let equal = self
            .checkpoint
            .borrow()
            .as_ref()
            .unwrap()
            .equals(self, Some(&new_checkpoint));
        if !equal {
            let string = new_checkpoint.to_string(self);
            *self.checkpoint.borrow_mut() = Some(new_checkpoint);
            return Some(string);
        }
        None
    }

    // Java: a commented-out `applyState(boolean, boolean, boolean)` follows in the
    // source (`FieldLockController.java:230-300`); it is not compiled.
}

/// Java private inner class `InternalState`.  Its methods read the outer instance's
/// fields, so they take the controller.
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
    fn new(outer: &FieldLockController, enabled: bool, editable: bool, locked: bool) -> InternalState {
        let toggle_button_enabled = outer.toggle_button.as_ref().map(|c| c.is_enabled());
        let button_enabled = outer.button.as_ref().map(|c| c.is_enabled());
        let (text_component_enabled, text_component_editable) = match &outer.text_component {
            Some(text_component) => (
                Some(text_component.is_enabled()),
                Some(text_component.is_editable()),
            ),
            None => (None, None),
        };
        let spinner_enabled = outer.spinner.as_ref().map(|c| c.is_enabled());
        InternalState {
            enabled,
            editable,
            locked,
            toggle_button_enabled,
            button_enabled,
            text_component_enabled,
            text_component_editable,
            spinner_enabled,
        }
    }

    /// Java `equals(InternalState)`.
    fn equals(&self, outer: &FieldLockController, is: Option<&InternalState>) -> bool {
        let Some(is) = is else {
            return false;
        };
        if self.enabled != is.enabled || self.editable != is.editable || self.locked != is.locked {
            return false;
        }
        // Only check the components that exist in the instance.
        if outer.toggle_button.is_some() && self.toggle_button_enabled != is.toggle_button_enabled {
            return false;
        }
        if outer.button.is_some() && self.button_enabled != is.button_enabled {
            return false;
        }
        if outer.text_component.is_some()
            && (self.text_component_enabled != is.text_component_enabled
                || self.text_component_editable != is.text_component_editable)
        {
            return false;
        }
        if outer.spinner.is_some() && self.spinner_enabled != is.spinner_enabled {
            return false;
        }
        true
    }

    /// Java `toString()`.
    fn to_string(&self, outer: &FieldLockController) -> String {
        let java_boolean = |value: Option<bool>| match value {
            Some(value) => value.to_string(),
            None => "null".to_string(),
        };
        let mut descr = String::new();
        descr.push_str(&format!(
            "[readOnly:{},enabled:{},editable:{},locked:{}",
            outer.read_only, self.enabled, self.editable, self.locked
        ));
        if outer.toggle_button.is_some() {
            descr.push_str(&format!(
                "\n[toggleButton enabled:{}]",
                java_boolean(self.toggle_button_enabled)
            ));
        }
        if outer.button.is_some() {
            descr.push_str(&format!("\n[button enabled:{}]", java_boolean(self.button_enabled)));
        }
        if outer.text_component.is_some() {
            descr.push_str(&format!(
                "\n[textComponent enabled:{},editable:{}]",
                java_boolean(self.text_component_enabled),
                java_boolean(self.text_component_editable)
            ));
        }
        if outer.spinner.is_some() {
            descr.push_str(&format!("\n[spinner enabled:{}]", java_boolean(self.spinner_enabled)));
        }
        descr.push(']');
        descr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unselected_toggle_button_disables_controlled_button() {
        let toggle = JComponent::new_check_box("t");
        let button = JComponent::new_button("b");
        let controller = FieldLockController::get_toggle_button_instance_j_toggle_button(&toggle);
        assert!(toggle.is_enabled());
        assert!(!controller.set_locked(false));
        assert!(controller.set_locked(true));
        assert!(!toggle.is_enabled());
        let _ = button;
    }
}
