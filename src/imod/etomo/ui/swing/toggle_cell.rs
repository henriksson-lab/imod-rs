//! `IMOD/Etomo/src/etomo/ui/swing/ToggleCell.java`.
//!
//! This is the common package-private table-cell contract.  `JPanel`,
//! `GridBagLayout`, `GridBagConstraints`, and listener registration remain
//! native Swing boundaries; no widget implementation is substituted here.
#![allow(dead_code)]

use super::cell::{CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary};

/// Java package-private `ToggleCell`.
pub trait ToggleCell {
    /// Java `getLabel()`.
    fn get_label(&self) -> &str;

    /// Java `setLabel(String)`.
    fn set_label(&mut self, label: &str);

    /// Java `setSelected(boolean)`.
    fn set_selected(&mut self, selected: bool);

    /// Java `addActionListener(ActionListener)`; invocation remains native.
    fn add_action_listener(&mut self);

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.
    fn add(
        &mut self,
        panel: &mut CellPanelBoundary,
        layout: &mut CellGridBagLayoutBoundary,
        constraints: &mut CellGridBagConstraintsBoundary,
    );

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool;

    /// Java `getHeight()`.
    fn get_height(&self) -> i32;

    /// Java `getWidth()`.
    fn get_width(&self) -> i32;

    /// Java `setWarning(boolean)`.
    fn set_warning(&mut self, warning: bool);

    /// Java `addChangeListener(ChangeListener)`; invocation remains native.
    fn add_change_listener(&mut self);

    /// Java `setEnabled(boolean)`.
    fn set_enabled(&mut self, enabled: bool);

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Cell {
        label: String,
        selected: bool,
        enabled: bool,
        warning: bool,
        action_listeners: usize,
        change_listeners: usize,
    }
    impl ToggleCell for Cell {
        fn get_label(&self) -> &str {
            &self.label
        }
        fn set_label(&mut self, label: &str) {
            self.label = label.into();
        }
        fn set_selected(&mut self, selected: bool) {
            self.selected = selected;
        }
        fn add_action_listener(&mut self) {
            self.action_listeners += 1;
        }
        fn add(
            &mut self,
            _: &mut CellPanelBoundary,
            _: &mut CellGridBagLayoutBoundary,
            _: &mut CellGridBagConstraintsBoundary,
        ) {
        }
        fn is_selected(&self) -> bool {
            self.selected
        }
        fn get_height(&self) -> i32 {
            0
        }
        fn get_width(&self) -> i32 {
            0
        }
        fn set_warning(&mut self, warning: bool) {
            self.warning = warning;
        }
        fn add_change_listener(&mut self) {
            self.change_listeners += 1;
        }
        fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }
        fn is_enabled(&self) -> bool {
            self.enabled
        }
    }

    #[test]
    fn source_contract_keeps_selection_enabled_and_listener_operations_distinct() {
        let mut cell = Cell::default();
        cell.set_label("Queue");
        cell.set_selected(true);
        cell.set_enabled(true);
        cell.set_warning(true);
        cell.add_action_listener();
        cell.add_change_listener();
        assert_eq!(cell.get_label(), "Queue");
        assert!(cell.is_selected() && cell.is_enabled() && cell.warning);
        assert_eq!((cell.action_listeners, cell.change_listeners), (1, 1));
    }
}
