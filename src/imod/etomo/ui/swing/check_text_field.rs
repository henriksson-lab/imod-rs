//! `IMOD/Etomo/src/etomo/ui/swing/CheckTextField.java`.
//!
//! The Java `JPanel`/`BoxLayout`, `ActionListener`, and document listener are
//! explicit native GUI boundaries.  This module preserves the paired checkbox
//! and text-field state and the Java unit's update/validation ordering.
#![allow(dead_code)]

use super::check_box::{BooleanFieldSetting, CheckBox};
use super::labeled_text_field::{FieldValidationFailedException, TextFieldSetting};
use super::panel::Dimension;
use super::text_field::TextField;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type as NumericType, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

/// Java `FieldSettingBundle` at this unit's `FieldSettingInterface` boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldSettingBundle {
    pub boolean_setting: Option<BooleanFieldSetting>,
    pub text_setting: Option<TextFieldSetting>,
}
impl FieldSettingBundle {
    pub fn add_boolean_setting(&mut self, setting: Option<&BooleanFieldSetting>) {
        self.boolean_setting = setting.cloned();
    }
    pub fn add_text_setting(&mut self, setting: Option<&TextFieldSetting>) {
        self.text_setting = setting.cloned();
    }
}

/// Source-visible `JPanel` state owned by this unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JPanelBoundary {
    pub visible: bool,
    pub box_layout_x_axis: bool,
    pub child_count: usize,
}
impl Default for JPanelBoundary {
    fn default() -> Self {
        Self {
            visible: true,
            box_layout_x_axis: false,
            child_count: 0,
        }
    }
}

/// Java final `CheckTextField`.
#[derive(Clone, Debug)]
pub struct CheckTextField {
    pub pnl_root: JPanelBoundary,
    pub check_box: CheckBox,
    pub text_field: TextField,
    pub label: String,
    pub numeric_type: Option<NumericType>,
    pub field_type: Option<FieldType>,
    pub required: bool,
    pub directive_def: Option<String>,
    pub debug: bool,
    pub document_listener_count: usize,
}

impl CheckTextField {
    fn new(field_type: Option<FieldType>, label: &str, numeric_type: Option<NumericType>) -> Self {
        let mut value = Self {
            pnl_root: JPanelBoundary::default(),
            check_box: CheckBox::new(),
            text_field: TextField::new(field_type, Some(label), None),
            label: label.into(),
            numeric_type,
            field_type,
            required: false,
            directive_def: None,
            debug: false,
            document_listener_count: 0,
        };
        value.set_label(label);
        value
    }
    pub fn get_instance(field_type: Option<FieldType>, label: &str) -> Self {
        let mut value = Self::new(field_type, label, None);
        value.create_panel();
        value.update_display();
        value.add_listeners();
        value
    }
    pub fn get_numeric_instance(
        field_type: Option<FieldType>,
        label: &str,
        numeric_type: NumericType,
    ) -> Self {
        let mut value = Self::new(field_type, label, Some(numeric_type));
        value.create_panel();
        value.update_display();
        value.add_listeners();
        value
    }
    pub fn get_name(&self) -> Option<&str> {
        self.check_box.get_name()
    }
    /// Java `equals(Object)`: caller supplies the native child identity match.
    pub fn equals_child(&self, is_check_box: bool, is_text_field: bool) -> bool {
        is_check_box || is_text_field
    }
    pub fn equals_document(&self, document_identity: usize) -> bool {
        self.text_field.get_document_identity() == document_identity
    }
    pub fn set_label(&mut self, label: &str) {
        self.check_box.set_text(Some(label));
        self.text_field.set_name(Some(label));
    }
    pub fn set_alternate_label(&mut self, label: &str) {
        self.check_box.set_alternate_text(Some(label));
        self.text_field.set_name(Some(label));
    }
    pub fn switch_labels(&mut self, alternate: bool) {
        self.check_box.switch_text(alternate);
        let label = self.check_box.get_text().map(str::to_owned);
        self.text_field.set_name(label.as_deref());
    }
    pub fn checkpoint_values(&mut self, checkbox_value: bool, text_value: &str) {
        self.check_box.checkpoint_value(checkbox_value);
        self.text_field.checkpoint_value(text_value);
    }
    pub fn checkpoint(&mut self) {
        self.check_box.checkpoint();
        self.text_field.checkpoint();
    }
    pub fn reset_to_checkpoint(&mut self) {
        self.check_box.reset_to_checkpoint();
        self.text_field.reset_to_checkpoint();
    }
    pub fn set_columns(&mut self) {
        if self.field_type.is_some() {
            self.text_field.set_columns();
        }
    }
    pub fn is_different_from_checkpoint_default(&self) -> bool {
        self.check_box.is_different_from_checkpoint(false)
            || self.text_field.is_different_from_checkpoint(false)
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.text_field.set_debug(input);
    }
    pub fn set_enabled(&mut self, enable: bool) {
        self.check_box.set_enabled(enable);
        self.update_display();
    }
    fn update_display(&mut self) {
        self.text_field
            .set_enabled(self.check_box.is_enabled() && self.check_box.is_selected());
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.check_box.set_editable(editable);
        self.text_field.set_editable(editable);
    }
    pub fn is_enabled(&self) -> bool {
        self.check_box.is_enabled()
    }
    fn create_panel(&mut self) {
        self.pnl_root.box_layout_x_axis = true;
        self.pnl_root.child_count = 2;
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.text_field.set_tool_tip_text(text);
        self.check_box.set_tool_tip_text(text);
    }
    pub fn set_unformatted_tooltip(&mut self, text: Option<&str>) -> Option<&str> {
        self.text_field.set_unformatted_tooltip(text);
        self.check_box.set_unformatted_tooltip(text)
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.text_field.has_unformatted_tooltip() || self.check_box.has_unformatted_tooltip()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        self.text_field
            .use_unformatted_tooltip(param_descr, directive_descr);
        self.check_box
            .use_unformatted_tooltip(param_descr, directive_descr);
    }
    pub fn set_check_box_unformatted_tooltip(&mut self, text: Option<&str>) {
        self.check_box.set_unformatted_tooltip(text);
    }
    pub fn set_field_unformatted_tooltip(&mut self, text: Option<&str>) {
        self.text_field.set_unformatted_tooltip(text);
    }
    pub fn set_alternate_tooltip_text(&mut self, text: Option<&str>) {
        self.text_field.set_alternate_tooltip_text(text);
        self.check_box.set_alternate_tooltip_text(text);
    }
    pub fn switch_tooltips(&mut self, alternate: bool) {
        self.text_field.switch_tooltips(alternate);
        self.check_box.switch_tooltips(alternate);
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.visible = visible;
    }
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
    }
    pub fn backup(&mut self) {
        self.check_box.backup();
        self.text_field.backup();
    }
    pub fn clear(&mut self) {
        self.check_box.clear();
        self.text_field.clear();
        self.update_display();
    }
    pub fn clear_field_highlight(&mut self) {
        self.check_box.clear_field_highlight();
        self.text_field.clear_field_highlight();
    }
    pub fn equals_default_value(&self) -> bool {
        self.check_box.equals_default_value() && self.text_field.equals_default_value()
    }
    pub fn equals_default_value_string(&self, value: &str) -> bool {
        self.check_box.equals_default_value_string(Some(value))
            && self.text_field.equals_default_value_string(value)
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.check_box.equals_field_highlight() && self.text_field.equals_field_highlight()
    }
    pub fn equals_field_highlight_string(&self, value: &str) -> bool {
        self.check_box.equals_field_highlight_string(Some(value))
            && self.text_field.equals_field_highlight_string(value)
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        self.check_box.equals_selected_string_value(value)
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.text_field.get_tooltip()
    }
    pub fn is_boolean(&self) -> bool {
        true
    }
    pub fn is_debug(&self) -> bool {
        self.debug || self.check_box.is_debug()
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        self.check_box.is_different_from_checkpoint(always_check)
            || self.text_field.is_different_from_checkpoint(always_check)
    }
    pub fn get_checkpoint(&self) -> FieldSettingBundle {
        let mut bundle = FieldSettingBundle::default();
        bundle.add_boolean_setting(self.check_box.get_checkpoint());
        bundle.add_text_setting(self.text_field.get_checkpoint());
        bundle
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    pub fn get_field_highlight(&self) -> FieldSettingBundle {
        let mut bundle = FieldSettingBundle::default();
        bundle.add_boolean_setting(self.check_box.get_field_highlight());
        bundle.add_text_setting(self.text_field.get_field_highlight());
        bundle
    }
    pub fn is_empty(&self) -> bool {
        java_lang_string_matches_whitespace(&self.text_field.get_text())
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.check_box.is_field_highlight_set() || self.text_field.is_field_highlight_set()
    }
    pub fn is_required(&self) -> bool {
        self.text_field.is_required()
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn restore_from_backup(&mut self) {
        self.check_box.restore_from_backup();
        self.text_field.restore_from_backup();
        self.update_display();
    }
    pub fn set_checkpoint(&mut self, input: &FieldSettingBundle) {
        self.check_box
            .set_checkpoint(input.boolean_setting.as_ref());
        self.text_field.set_checkpoint(input.text_setting.as_ref());
    }
    pub fn set_field_highlight_boolean(&mut self, value: bool) {
        self.check_box.set_field_highlight(value);
    }
    pub fn set_field_highlight(&mut self, input: &FieldSettingBundle) {
        self.check_box
            .set_field_highlight_setting(input.boolean_setting.as_ref());
        self.text_field
            .set_field_highlight_setting(input.text_setting.as_ref());
    }
    pub fn set_field_highlight_text(&mut self, text: &str) {
        self.text_field.set_field_highlight(text);
    }
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        self.check_box.set_preformatted_tooltip(tooltip);
        self.text_field.set_preformatted_tooltip(tooltip);
    }
    pub fn set_value_boolean(&mut self, value: bool) {
        self.check_box.set_value(value);
        self.update_display();
    }
    pub fn set_value(&mut self, value: Option<&str>) {
        self.text_field.set_value(value);
    }
    pub fn use_default_value(&mut self) {
        self.check_box.use_default_value(None);
        self.text_field.use_default_value();
        self.update_display();
    }
    pub fn set_check_box_tool_tip_text(&mut self, text: Option<&str>) {
        self.check_box.set_tool_tip_text(text);
    }
    pub fn set_field_tool_tip_text(&mut self, text: Option<&str>) {
        self.text_field.set_tool_tip_text(text);
    }
    pub fn get_root_component(&self) -> &JPanelBoundary {
        &self.pnl_root
    }
    fn add_listeners(&mut self) {
        self.check_box.add_action_listener();
    }
    pub fn add_action_listener(&mut self) {
        self.check_box.add_action_listener();
    }
    pub fn add_document_listener(&mut self) {
        self.document_listener_count += 1;
    }
    pub fn set_text(&mut self, input: Option<&str>) {
        self.text_field.set_text(input);
    }
    pub fn set_non_empty_text(&mut self, input: Option<&str>, nonempty: bool) {
        if !nonempty || input.is_some_and(|value| !value.is_empty()) {
            self.set_text(input);
        }
    }
    pub fn get_action_command(&self) -> Option<&str> {
        self.check_box.get_action_command()
    }
    pub fn get_label(&self) -> &str {
        &self.label
    }
    pub fn set_selected(&mut self, selected: bool) {
        self.check_box.set_selected(selected);
        self.update_display();
    }
    pub fn set_selected_number(&mut self, selected: Option<&ConstEtomoNumber>, allow_empty: bool) {
        if allow_empty || selected.is_some_and(|selected| !selected.is_null()) {
            self.set_selected(selected.is_some_and(ConstEtomoNumber::is));
        }
    }
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &JPanelBoundary {
        &self.pnl_root
    }
    pub fn set_required(&mut self, required: bool) {
        self.required = required;
        self.text_field.set_required(required);
    }
    pub fn get_text_validated(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.text_field
            .get_text_validated(do_validation && self.text_field.is_enabled())
    }
    pub fn get_text(&self) -> String {
        self.text_field.get_text()
    }
    pub fn get_description(&self) -> String {
        self.get_quoted_label()
    }
    pub fn get_quoted_label(&self) -> String {
        utilities::quote_label(self.check_box.get_text()).unwrap_or_default()
    }
    pub fn get_size(&self) -> Dimension {
        self.text_field.get_size()
    }
    pub fn set_text_preferred_width(&mut self, width: i32) {
        self.text_field.set_preferred_width(width);
    }
    pub fn set_text_field_visible(&mut self, visible: bool) {
        self.text_field.set_visible(visible);
    }
    /// Java private `action`, invoked by the registered checkbox listener.
    pub fn action(&mut self) {
        self.update_display();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkbox_selection_drives_text_enabled_state() {
        let mut field = CheckTextField::get_instance(Some(FieldType::Integer), "Use bin:");
        assert!(!field.text_field.is_enabled());
        field.set_selected(true);
        assert!(field.text_field.is_enabled());
        field.set_enabled(false);
        assert!(!field.text_field.is_enabled());
    }
    #[test]
    fn checkpoint_and_restore_preserve_both_source_children() {
        let mut field = CheckTextField::get_instance(Some(FieldType::String), "Input:");
        field.set_selected(true);
        field.set_text(Some("one"));
        field.checkpoint();
        field.set_selected(false);
        field.set_text(Some("two"));
        assert!(field.is_different_from_checkpoint(false));
        field.reset_to_checkpoint();
        assert!(field.is_selected());
        assert_eq!(field.get_text(), "one");
    }
    #[test]
    fn validation_runs_only_when_selected_text_is_enabled() {
        let mut field = CheckTextField::get_instance(Some(FieldType::Integer), "Bin:");
        field.set_required(true);
        assert!(field.get_text_validated(true).is_ok());
        field.set_selected(true);
        assert!(field.get_text_validated(true).is_err());
    }
}
