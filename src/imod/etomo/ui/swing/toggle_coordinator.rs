//! `IMOD/Etomo/src/etomo/ui/swing/ToggleCoordinator.java`.
//!
//! The concrete `JToggleButton` instances and their property/action listeners
//! belong to Swing.  This source unit records the exact state and listener
//! consequences at that boundary.
#![allow(dead_code)]

/// Java `JToggleButton` state used by this coordinator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToggleButtonBoundary {
    pub identity_hash_code: usize,
    pub enabled: bool,
    pub selected: bool,
    pub action_listener_registered: bool,
    pub enabled_property_listener_registered: bool,
}

/// Java package-private `ToggleCoordinator`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToggleCoordinator {
    /// Java final `tbToggler`.
    pub tb_toggler: Option<ToggleButtonBoundary>,
    /// Java `targets`; `None` represents Java null, distinct from an empty Vector.
    pub targets: Option<Vec<ToggleButtonBoundary>>,
}

impl ToggleCoordinator {
    /// Java `ToggleCoordinator(CheckBoxCell)` after its `getComponent()`
    /// runtime-type test.  `None` retains either Java null input or a non-toggle
    /// component.
    pub fn new(toggler: Option<ToggleButtonBoundary>) -> Self {
        let mut value = Self {
            tb_toggler: toggler,
            targets: None,
        };
        if let Some(toggler) = &mut value.tb_toggler {
            toggler.action_listener_registered = true;
            value.apply_enabled_state(None);
        }
        value
    }

    /// Java `addTarget(CheckBoxCell)` after its null/type tests.
    pub fn add_target(&mut self, target: Option<ToggleButtonBoundary>) {
        let Some(mut target) = target else { return };
        if self.targets.is_none() {
            self.targets = Some(Vec::new());
        }
        let enabled = target.enabled;
        target.enabled_property_listener_registered = true;
        self.targets.as_mut().expect("assigned above").push(target);
        self.apply_enabled_state(Some(enabled));
    }

    /// Java `deleteTarget(CheckBoxCell)` after its null/type tests.
    pub fn delete_target(&mut self, target: Option<&ToggleButtonBoundary>) {
        let Some(target) = target else { return };
        if let Some(targets) = &mut self.targets {
            if let Some(index) = targets
                .iter()
                .position(|value| value.identity_hash_code == target.identity_hash_code)
            {
                targets[index].enabled_property_listener_registered = false;
                targets.remove(index);
            }
        } else {
            return;
        }
        self.apply_enabled_state(None);
    }

    /// Java private `applyEnabledState(Boolean)`.
    fn apply_enabled_state(&mut self, enabled: Option<bool>) {
        let all_targets_disabled = self.all_targets_disabled();
        let Some(toggler) = &mut self.tb_toggler else {
            return;
        };
        if enabled.is_none() || toggler.enabled != enabled.expect("checked above") {
            toggler.enabled = !all_targets_disabled;
        }
    }

    /// Java private `allTargetsDisabled()`.
    fn all_targets_disabled(&self) -> bool {
        self.targets
            .as_ref()
            .is_none_or(|targets| !targets.iter().any(|target| target.enabled))
    }

    /// Java inner `TargetEnabledChangeListener.propertyChange(PropertyChangeEvent)`.
    /// `property_name_enabled` and `new_value` retain the Java event's relevant
    /// two fields; identity selects the observed target.
    pub fn property_change(
        &mut self,
        identity_hash_code: usize,
        property_name_enabled: bool,
        new_value: Option<bool>,
    ) {
        let Some(new_value) = new_value else { return };
        if !property_name_enabled {
            return;
        }
        if let Some(targets) = &mut self.targets {
            if let Some(target) = targets
                .iter_mut()
                .find(|target| target.identity_hash_code == identity_hash_code)
            {
                target.enabled = new_value;
            }
        }
        self.apply_enabled_state(Some(new_value));
    }

    /// Java inner `ToggleActionListener.actionPerformed(ActionEvent)`.
    /// The action event carries Swing source identity; only a toggler event
    /// changes enabled target selection.
    pub fn action_performed(&mut self, source_identity_hash_code: Option<usize>) {
        let Some(toggler) = &self.tb_toggler else {
            return;
        };
        if source_identity_hash_code != Some(toggler.identity_hash_code) {
            return;
        }
        let selected = toggler.selected;
        let Some(targets) = &mut self.targets else {
            return;
        };
        for target in targets {
            if target.enabled {
                target.selected = selected;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn button(identity_hash_code: usize, enabled: bool, selected: bool) -> ToggleButtonBoundary {
        ToggleButtonBoundary {
            identity_hash_code,
            enabled,
            selected,
            action_listener_registered: false,
            enabled_property_listener_registered: false,
        }
    }
    #[test]
    fn constructor_and_target_enable_events_follow_java_toggler_state() {
        let mut coordinator = ToggleCoordinator::new(Some(button(1, true, false)));
        assert!(!coordinator.tb_toggler.as_ref().unwrap().enabled);
        coordinator.add_target(Some(button(2, true, false)));
        assert!(coordinator.tb_toggler.as_ref().unwrap().enabled);
        assert!(coordinator.targets.as_ref().unwrap()[0].enabled_property_listener_registered);
        coordinator.property_change(2, true, Some(false));
        assert!(!coordinator.tb_toggler.as_ref().unwrap().enabled);
    }
    #[test]
    fn action_only_updates_enabled_targets_and_requires_toggler_identity() {
        let mut coordinator = ToggleCoordinator::new(Some(button(1, true, true)));
        coordinator.add_target(Some(button(2, true, false)));
        coordinator.add_target(Some(button(3, false, false)));
        coordinator.action_performed(Some(99));
        assert!(!coordinator.targets.as_ref().unwrap()[0].selected);
        coordinator.action_performed(Some(1));
        assert!(coordinator.targets.as_ref().unwrap()[0].selected);
        assert!(!coordinator.targets.as_ref().unwrap()[1].selected);
    }
    #[test]
    fn delete_removes_listener_and_recomputes_enabled_state() {
        let mut coordinator = ToggleCoordinator::new(Some(button(1, true, false)));
        let target = button(2, true, false);
        coordinator.add_target(Some(target.clone()));
        coordinator.delete_target(Some(&target));
        assert!(coordinator.targets.as_ref().unwrap().is_empty());
        assert!(!coordinator.tb_toggler.as_ref().unwrap().enabled);
    }
}
