//! `IMOD/Etomo/src/etomo/ui/swing/ToggleCoordinator.java`.
//!
//! Coordinates a header toggle button with the toggle buttons in its column: the
//! toggler selects every enabled target, and is enabled only while at least one
//! target is.

use super::check_box_cell::CheckBoxCell;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ComponentKind, JComponent, PropertyChangeListener,
};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Java private `ENABLED_PROPERTY`.
const ENABLED_PROPERTY: &str = "enabled";

/// `component instanceof JToggleButton`: `JToggleButton`, `JCheckBox` and
/// `JRadioButton`.
fn is_toggle_button(component: &JComponent) -> bool {
    matches!(
        component.kind(),
        ComponentKind::ToggleButton | ComponentKind::CheckBox | ComponentKind::RadioButton
    )
}

/// Java package-private `ToggleCoordinator`.
pub struct ToggleCoordinator {
    /// Java final `targetEnabledChangeListener` (inner `TargetEnabledChangeListener`).
    target_enabled_change_listener: PropertyChangeListener,
    /// Java final `tbToggler`.
    tb_toggler: Option<Rc<JComponent>>,
    /// Java `targets`, initialised to null.
    targets: RefCell<Option<Vec<Rc<JComponent>>>>,
}

impl ToggleCoordinator {
    /// Java `ToggleCoordinator(CheckBoxCell)`.
    pub fn new(cbc_toggler: Option<&CheckBoxCell>) -> Rc<ToggleCoordinator> {
        let mut tb_toggler = None;
        if let Some(cbc_toggler) = cbc_toggler {
            let component = cbc_toggler.get_component();
            if is_toggle_button(&component) {
                tb_toggler = Some(component);
            }
        }
        let coordinator = Rc::new_cyclic(|this: &Weak<ToggleCoordinator>| {
            let weak = this.clone();
            let target_enabled_change_listener: PropertyChangeListener =
                Rc::new(move |property_name: &str, new_value: bool| {
                    if let Some(coordinator) = weak.upgrade() {
                        coordinator.target_enabled_property_change(property_name, new_value);
                    }
                });
            ToggleCoordinator {
                target_enabled_change_listener,
                tb_toggler,
                targets: RefCell::new(None),
            }
        });
        if let Some(tb_toggler) = &coordinator.tb_toggler {
            let weak = Rc::downgrade(&coordinator);
            let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(coordinator) = weak.upgrade() {
                    coordinator.toggle_action_performed(Some(event));
                }
            });
            tb_toggler.add_action_listener(listener);
            coordinator.apply_enabled_state(None);
        }
        coordinator
    }

    /// Java final `addTarget(CheckBoxCell)`.
    pub fn add_target(&self, cbc_target: Option<&CheckBoxCell>) {
        let Some(cbc_target) = cbc_target else {
            return;
        };
        let component = cbc_target.get_component();
        if !is_toggle_button(&component) {
            return;
        }
        let target = component;
        {
            let mut targets = self.targets.borrow_mut();
            if targets.is_none() {
                *targets = Some(Vec::new());
            }
            targets.as_mut().unwrap().push(target.clone());
        }
        self.apply_enabled_state(Some(target.is_enabled()));
        target.add_property_change_listener(
            ENABLED_PROPERTY,
            self.target_enabled_change_listener.clone(),
        );
    }

    /// Java final `deleteTarget(CheckBoxCell)`.
    pub fn delete_target(&self, cbc_target: Option<&CheckBoxCell>) {
        let Some(cbc_target) = cbc_target else {
            return;
        };
        let component = cbc_target.get_component();
        if !is_toggle_button(&component) {
            return;
        }
        let target = component;
        target.remove_property_change_listener(&self.target_enabled_change_listener);
        {
            let mut targets = self.targets.borrow_mut();
            let Some(targets) = targets.as_mut() else {
                return;
            };
            // List.remove(Object): the first element equal (identical) to target.
            if let Some(index) = targets.iter().position(|t| Rc::ptr_eq(t, &target)) {
                targets.remove(index);
            }
        }
        self.apply_enabled_state(None);
    }

    /// Java private `applyEnabledState(Boolean)`.
    fn apply_enabled_state(&self, enabled: Option<bool>) {
        let Some(tb_toggler) = &self.tb_toggler else {
            return;
        };
        if enabled.is_none() || tb_toggler.is_enabled() != enabled.unwrap() {
            tb_toggler.set_enabled(!self.all_targets_disabled());
        }
    }

    /// Java private `allTargetsDisabled()`.
    fn all_targets_disabled(&self) -> bool {
        let targets = self.targets.borrow();
        let Some(targets) = targets.as_ref() else {
            return true;
        };
        for target in targets {
            if target.is_enabled() {
                return false;
            }
        }
        true
    }

    /// Java inner `TargetEnabledChangeListener.propertyChange(PropertyChangeEvent)`.
    fn target_enabled_property_change(&self, property_name: &str, new_value: bool) {
        if ENABLED_PROPERTY != property_name {
            return;
        }
        self.apply_enabled_state(Some(new_value));
    }

    /// Java inner `ToggleActionListener.actionPerformed(ActionEvent)`.
    fn toggle_action_performed(&self, event: Option<&ActionEvent>) {
        let Some(event) = event else {
            return;
        };
        let Some(tb_toggler) = &self.tb_toggler else {
            return;
        };
        let targets = self.targets.borrow().clone();
        let Some(targets) = targets else {
            return;
        };
        if !Rc::ptr_eq(event.get_source(), tb_toggler) {
            return;
        }
        let selected = tb_toggler.is_selected();
        for target in &targets {
            if target.is_enabled() {
                target.set_selected(selected);
            }
        }
    }
}
