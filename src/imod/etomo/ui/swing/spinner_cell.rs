//! `IMOD/Etomo/src/etomo/ui/swing/SpinnerCell.java`.
#![allow(dead_code)]
use super::colors::{self, ColorUiResource};
use super::field_lock_controller::{FieldLockController, JSpinner};
use super::section_table_panel::HeaderCell;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::util::utilities;
/// Source-observable `JSpinner` and `DefaultEditor` state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpinnerCellBoundary {
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub name: Option<String>,
    pub tooltip: Option<String>,
    pub text_tooltip: Option<String>,
    pub text_background: ColorUiResource,
    pub text_foreground: ColorUiResource,
    pub text_disabled_foreground: ColorUiResource,
    pub horizontal_alignment_left: bool,
    pub width: i32,
    pub visible: bool,
    pub change_listener_count: usize,
    pub focus_listener_count: usize,
}
impl SpinnerCellBoundary {
    fn new(minimum: i32, maximum: i32) -> Self {
        Self {
            value: minimum,
            minimum,
            maximum,
            name: None,
            tooltip: None,
            text_tooltip: None,
            text_background: colors::BACKGROUND,
            text_foreground: colors::CELL_FOREGROUND,
            text_disabled_foreground: colors::CELL_FOREGROUND,
            horizontal_alignment_left: true,
            width: 0,
            visible: true,
            change_listener_count: 0,
            focus_listener_count: 0,
        }
    }
}
/// Java package-private final `SpinnerCell`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpinnerCell {
    pub disabled_value: Option<i32>,
    pub saved_value: Option<i32>,
    pub spinner: SpinnerCellBoundary,
    pub minimum_value: i32,
    pub field_lock_controller: FieldLockController,
    pub warning: bool,
    pub warning_tooltip: Option<String>,
    pub table_header: Option<String>,
    pub row_header: Option<String>,
    pub column_header: Option<String>,
}
impl SpinnerCell {
    /// Java `getIntInstance(int,int)`.
    pub fn get_int_instance(minimum: i32, maximum: i32) -> Self {
        Self::new(minimum, maximum, None)
    }
    /// Java `getNamedIntInstance(int,int,String,String,String)`.
    pub fn get_named_int_instance(
        minimum: i32,
        maximum: i32,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) -> Self {
        let reference = utilities::concatenate(reference1, reference2, reference3, Some(" "));
        Self::new(minimum, maximum, reference.as_deref())
    }
    /// Java private `SpinnerCell(int,int,String)`.
    fn new(minimum: i32, maximum: i32, reference: Option<&str>) -> Self {
        let spinner = SpinnerCellBoundary::new(minimum, maximum);
        let mut value = Self {
            disabled_value: None,
            saved_value: None,
            spinner,
            minimum_value: minimum,
            field_lock_controller: FieldLockController::get_spinner_instance(JSpinner {
                enabled: true,
            }),
            warning: false,
            warning_tooltip: None,
            table_header: None,
            row_header: None,
            column_header: None,
        };
        value.set_background();
        value.set_foreground();
        if let Some(reference) = reference {
            value.set_name(reference);
        }
        value
    }
    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        self.spinner.value.to_string()
    }
    /// Java overridden `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        let was_enabled = self.is_enabled();
        self.field_lock_controller.set_enabled(enabled);
        if let Some(disabled_value) = self.disabled_value {
            if enabled {
                if self.spinner.value == disabled_value {
                    if let Some(saved) = self.saved_value {
                        self.spinner.value = saved;
                    }
                }
            } else {
                if was_enabled {
                    self.saved_value = Some(self.spinner.value.max(self.minimum_value));
                }
                self.spinner.value = disabled_value;
            }
        }
    }
    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }
    /// Java `setLocked(boolean)`.
    pub fn set_locked(&mut self, locked: bool) {
        self.field_lock_controller.set_locked(locked);
    }
    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }
    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.field_lock_controller.set_editable(editable);
    }
    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }
    /// Java `setDisabledValue(int)`.
    pub fn set_disabled_value(&mut self, disabled_value: i32) {
        self.disabled_value = Some(disabled_value);
    }
    /// Java `getComponent()`.
    pub fn get_component(&self) -> &SpinnerCellBoundary {
        &self.spinner
    }
    /// Java `setValue(int)`.
    pub fn set_value(&mut self, value: i32) {
        self.saved_value = Some(value);
        if self.is_enabled() || self.disabled_value.is_none() {
            self.spinner.value = value;
        }
    }
    /// Java `setValue(String)`.
    pub fn set_value_string(&mut self, value: &str) {
        if let Ok(value) = value.parse() {
            self.set_value(value);
        }
    }
    /// Java `getFieldType()`.
    pub fn get_field_type(&self) -> &'static str {
        "sp"
    }
    /// Java `getIntValue()`.
    pub fn get_int_value(&self) -> i32 {
        self.spinner.value
    }
    /// Java `getStringValue()`.
    pub fn get_string_value(&self) -> String {
        self.get_int_value().to_string()
    }
    /// Java `getText()`.
    pub fn get_text(&self) -> String {
        self.get_string_value()
    }
    /// Java `getWidth()`.
    pub fn get_width(&self) -> i32 {
        self.spinner.width
    }
    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&mut self) {
        self.spinner.change_listener_count += 1;
    }
    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&mut self) {
        self.spinner.focus_listener_count += 1;
    }
    /// Java `removeChangeListener(ChangeListener)`.
    pub fn remove_change_listener(&mut self) {
        self.spinner.change_listener_count = self.spinner.change_listener_count.saturating_sub(1);
    }
    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.spinner.tooltip = text.map(str::to_owned);
        self.spinner.text_tooltip = text.map(str::to_owned);
    }
    /// Java overridden `setBackground(ColorUIResource)`.
    pub fn set_background_color(&mut self, color: ColorUiResource) {
        self.spinner.text_background = color;
    }
    /// Java inherited `setBackground()`.
    pub fn set_background(&mut self) {
        self.spinner.text_background = if self.warning {
            if self.is_enabled() {
                colors::WARNING_BACKGROUND
            } else {
                colors::WARNING_BACKGROUND_NOT_EDITABLE
            }
        } else if self.is_enabled() {
            colors::BACKGROUND
        } else {
            colors::Colors::get_cell_not_editable_background()
        };
    }
    /// Java `setWarning(boolean,String)` inherited from `InputCell`.
    pub fn set_warning(&mut self, warning: bool, tooltip: Option<&str>) {
        self.warning = warning;
        self.warning_tooltip = tooltip.map(str::to_owned);
        self.set_background();
    }
    /// Java `setForeground()`.
    pub fn set_foreground(&mut self) {
        self.spinner.text_foreground = colors::CELL_FOREGROUND;
        self.spinner.text_disabled_foreground = colors::CELL_FOREGROUND;
    }
    /// Java overridden `setName(String,String,String)`.
    pub fn set_name_three(
        &mut self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        if let Some(reference) =
            utilities::concatenate(reference1, reference2, reference3, Some(" "))
        {
            self.set_name(&reference);
        }
    }
    /// Java `setName(String)`.
    pub fn set_name(&mut self, reference: &str) {
        self.spinner.name = utilities::convert_label_to_name(Some(reference), true)
            .map(|name| format!("sp{SEPARATOR_CHAR}{name}"));
    }
    /// Java `getName()`.
    pub fn get_name(&self) -> Option<&str> {
        self.spinner.name.as_deref()
    }
    /// Java inherited `setHeaders`.
    pub fn set_headers(&mut self, table: &str, row: &HeaderCell, column: &HeaderCell) {
        self.table_header = Some(table.into());
        self.row_header = Some(row.text.clone());
        self.column_header = Some(column.text.clone());
        let reference = utilities::convert_label_to_name_three(
            Some(table),
            Some(&row.text),
            Some(&column.text),
            true,
        )
        .unwrap_or_default();
        self.spinner.name = Some(format!("sp{SEPARATOR_CHAR}{reference}"));
    }
    /// Java inherited `remove()`.
    pub fn remove(&mut self) {
        self.spinner.visible = false;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_value_restores_prior_enabled_value() {
        let mut cell = SpinnerCell::get_int_instance(2, 9);
        cell.set_value(6);
        cell.set_disabled_value(-1);
        cell.set_enabled(false);
        assert_eq!(cell.get_int_value(), -1);
        cell.set_enabled(true);
        assert_eq!(cell.get_int_value(), 6);
    }
    #[test]
    fn headers_rebuild_spinner_name() {
        let mut cell = SpinnerCell::get_int_instance(1, 2);
        cell.set_headers("Table", &HeaderCell::new("Row"), &HeaderCell::new("Col"));
        assert_eq!(cell.get_name(), Some("sp.table-row-col"));
    }
}
