//! `IMOD/Etomo/src/etomo/ui/swing/ButtonCell.java`.
//!
//! `JButton`, `JToggleButton`, their listener dispatch, borders, and painting
//! are Swing-owned.  This source unit keeps their observable state at that
//! explicit GUI boundary while preserving ButtonCell's lock and table-cell
//! behaviour.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use super::field_lock_controller::{FieldLockController, JButton, JToggleButton};
use super::multi_line_button::ButtonBoundary;
use super::ui_utilities::{Icon, UiUtilities};

static NEXT_BUTTON_CELL_IDENTITY_HASH_CODE: AtomicUsize = AtomicUsize::new(1);

/// Java package-private final `ButtonCell`.
#[derive(Clone, Debug, PartialEq)]
pub struct ButtonCell {
    /// Java `AbstractButton button`; native widget implementation boundary.
    pub button: ButtonBoundary,
    /// The constructed runtime subtype: Java `JToggleButton` when true,
    /// otherwise Java `JButton`.
    pub toggle: bool,
    /// Java `FieldLockController fieldLockController`.
    pub field_lock_controller: FieldLockController,
    /// Java `Object.hashCode()` identity allocated by the Rust UI boundary.
    pub identity_hash_code: usize,
    /// The inherited `InputCell.setBackground` calls made by this source unit.
    pub background_refresh_count: usize,
}

impl ButtonCell {
    /// Java private `ButtonCell(Icon, String, boolean)`.
    pub fn new(icon: Option<Icon>, title: Option<&str>, toggle: bool) -> Self {
        let mut button = ButtonBoundary::default();
        if let Some(icon) = icon {
            button.icon = Some(icon);
            button.abstract_button.icon = Some(icon);
        } else if let Some(title) = title {
            button.text = Some(title.to_owned());
        }
        button.border = Some("BevelBorder.RAISED".into());
        button.abstract_button.preferred_size = Some(UiUtilities::get_preferred_size(
            &button.abstract_button,
            title,
        ));

        Self {
            button,
            toggle,
            field_lock_controller: if toggle {
                FieldLockController::get_toggle_button_instance(JToggleButton {
                    enabled: true,
                    selected: false,
                })
            } else {
                FieldLockController::get_button_instance(JButton { enabled: true })
            },
            identity_hash_code: NEXT_BUTTON_CELL_IDENTITY_HASH_CODE.fetch_add(1, Ordering::Relaxed),
            background_refresh_count: 0,
        }
    }

    /// Java static `getInstance(Icon)`.
    pub fn get_instance(icon: Option<Icon>) -> Self {
        Self::new(icon, None, false)
    }

    /// Java static `getToggleInstance(String)`.
    pub fn get_toggle_instance(title: Option<&str>) -> Self {
        Self::new(None, title, true)
    }

    /// Java `setName(String, String, String)`; deliberately unimplemented in
    /// the original source pending HeaderCell support.
    pub fn set_name(
        &mut self,
        _reference1: Option<&str>,
        _reference2: Option<&str>,
        _reference3: Option<&str>,
    ) {
    }

    /// Java `getName`.
    pub fn get_name(&self) -> Option<&str> {
        self.button.name.as_deref()
    }

    /// Java `getUniqueActionCommand`.
    pub fn get_unique_action_command(&self) -> String {
        format!("etomo.ui.swing.ButtonCell@{:x}", self.identity_hash_code)
    }

    /// Java `getComponent`; the concrete Swing component is a GUI boundary.
    pub fn get_component(&self) -> &ButtonBoundary {
        &self.button
    }

    /// Java `getText`.
    pub fn get_text(&self) -> Option<&str> {
        self.button.text.as_deref()
    }

    /// Java `TableComponent.getPreferredWidth`.
    pub fn get_preferred_width(&self) -> i32 {
        UiUtilities::get_preferred_width_button(
            &self.button.abstract_button,
            self.button.text.as_deref(),
        )
    }

    /// Java `getFieldType`; `UITestFieldType.BUTTON.toString()`.
    pub fn get_field_type(&self) -> &'static str {
        "bn"
    }

    /// Java `getWidth`.
    pub fn get_width(&self) -> i32 {
        self.button.width
    }

    /// Java `setSelected`.
    pub fn set_selected(&mut self, selected: bool) {
        self.button.selected = selected;
        if let Some(toggle_button) = &mut self.field_lock_controller.toggle_button {
            toggle_button.selected = selected;
        }
    }

    /// Java `isSelected`.
    pub fn is_selected(&self) -> bool {
        self.button.selected
    }

    /// Java `setActionCommand`.
    pub fn set_action_command(&mut self, input: Option<&str>) {
        self.button.action_command = input.map(str::to_owned);
    }

    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.button.action_command.as_deref()
    }

    /// Java `addActionListener`; listener invocation is a native GUI boundary.
    pub fn add_action_listener(&mut self) {
        self.button.action_listener_count += 1;
    }

    /// Java `setLocked`.
    pub fn set_locked(&mut self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.button.enabled = self
                .field_lock_controller
                .toggle_button
                .as_ref()
                .map_or_else(
                    || self.field_lock_controller.button.as_ref().unwrap().enabled,
                    |toggle_button| toggle_button.enabled,
                );
            self.background_refresh_count += 1;
        }
    }

    /// Java `setEditable`.
    pub fn set_editable(&mut self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.button.enabled = self
                .field_lock_controller
                .toggle_button
                .as_ref()
                .map_or_else(
                    || self.field_lock_controller.button.as_ref().unwrap().enabled,
                    |toggle_button| toggle_button.enabled,
                );
            self.background_refresh_count += 1;
        }
    }

    /// Java `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
        self.button.enabled = self
            .field_lock_controller
            .toggle_button
            .as_ref()
            .map_or_else(
                || self.field_lock_controller.button.as_ref().unwrap().enabled,
                |toggle_button| toggle_button.enabled,
            );
    }

    /// Java `isLocked`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `isEditable`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `setDisabledIcon`.
    pub fn set_disabled_icon(&mut self, icon: Option<Icon>) {
        self.button.disabled_icon = icon;
    }

    /// Java `setToolTipText`; TooltipFormatter remains the GUI presentation
    /// boundary until its source unit is translated.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.button.tooltip = text.map(str::to_owned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::panel::Dimension;
    use crate::imod::etomo::ui::swing::ui_utilities::{FontMetrics, Insets};

    #[test]
    fn factories_preserve_source_button_subtype_content_and_bevel_setup() {
        let icon = Icon {
            width: 13,
            height: 9,
        };
        let icon_cell = ButtonCell::get_instance(Some(icon));
        let toggle_cell = ButtonCell::get_toggle_instance(Some("Use"));

        assert!(!icon_cell.toggle);
        assert_eq!(icon_cell.button.icon, Some(icon));
        assert_eq!(icon_cell.button.text, None);
        assert_eq!(
            icon_cell.button.border.as_deref(),
            Some("BevelBorder.RAISED")
        );
        assert_eq!(
            icon_cell.button.abstract_button.preferred_size,
            Some(Dimension {
                width: 13,
                height: 9,
            })
        );
        assert!(toggle_cell.toggle);
        assert_eq!(toggle_cell.get_text(), Some("Use"));
        assert_eq!(toggle_cell.get_field_type(), "bn");
    }

    #[test]
    fn table_component_and_button_delegates_use_the_wrapped_button() {
        let mut cell = ButtonCell::get_toggle_instance(Some("Go"));
        cell.button.abstract_button.insets = Insets {
            left: 2,
            right: 3,
            ..Default::default()
        };
        cell.button.abstract_button.font_metrics = Some(FontMetrics {
            average_char_width: 4,
            wide_char_width: 5,
            height: 8,
        });
        cell.button.width = 44;
        cell.set_selected(true);
        cell.set_action_command(Some("run"));
        cell.add_action_listener();
        cell.set_disabled_icon(Some(Icon {
            width: 2,
            height: 3,
        }));
        cell.set_tool_tip_text(Some("Tip"));

        assert!(cell.is_selected());
        assert_eq!(
            cell.field_lock_controller
                .toggle_button
                .as_ref()
                .unwrap()
                .selected,
            true
        );
        assert_eq!(cell.get_action_command(), Some("run"));
        assert_eq!(cell.button.action_listener_count, 1);
        assert_eq!(cell.get_preferred_width(), 18);
        assert_eq!(cell.get_width(), 44);
        assert_eq!(cell.button.disabled_icon.unwrap().height, 3);
        assert_eq!(cell.button.tooltip.as_deref(), Some("Tip"));
        assert!(cell.get_component().selected);
    }

    #[test]
    fn lock_controller_follows_button_enable_policy_and_refreshes_only_on_change() {
        let mut cell = ButtonCell::get_instance(None);

        cell.set_locked(true);
        assert!(cell.is_locked());
        assert!(!cell.button.enabled);
        assert!(!cell.field_lock_controller.button.as_ref().unwrap().enabled);
        assert_eq!(cell.background_refresh_count, 1);
        cell.set_locked(true);
        assert_eq!(cell.background_refresh_count, 1);

        cell.set_enabled(false);
        assert!(!cell.is_enabled());
        cell.set_locked(false);
        assert!(!cell.button.enabled);
        assert_eq!(cell.background_refresh_count, 1);
        cell.set_enabled(true);
        assert!(cell.button.enabled);
        cell.set_editable(false);
        assert!(!cell.is_editable());
        assert!(!cell.button.enabled);
        assert_eq!(cell.background_refresh_count, 2);
    }

    #[test]
    fn name_is_intentionally_unimplemented_and_identity_command_is_unique() {
        let mut first = ButtonCell::get_instance(None);
        let second = ButtonCell::get_instance(None);
        first.button.name = Some("existing".into());
        first.set_name(Some("table"), Some("row"), Some("column"));

        assert_eq!(first.get_name(), Some("existing"));
        assert_ne!(
            first.get_unique_action_command(),
            second.get_unique_action_command()
        );
        assert!(
            first
                .get_unique_action_command()
                .starts_with("etomo.ui.swing.ButtonCell@")
        );
    }
}
