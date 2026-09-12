//! `IMOD/Etomo/src/etomo/ui/swing/TextField.java`.
//!
//! `JTextField`, its document, font metrics, focus listeners, and pixel sizing
//! are native GUI boundaries.  The Rust model retains the Java unit's state and
//! its field/checkpoint/validation-facing operations.
#![allow(dead_code)]

use crate::imod::etomo::r#type::const_etomo_number::Type as NumericType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::labeled_text_field::{
    FieldValidationFailedException, LabeledTextField, TextFieldSetting, ValidationSet,
};
use super::panel::Dimension;

/// Java package-private `TextField`.
#[derive(Clone, Debug)]
pub struct TextField {
    pub text_field: LabeledTextField,
    pub field_type: Option<FieldType>,
    pub location_descr: Option<String>,
    pub reference: Option<String>,
    pub required: bool,
    pub file_must_exist: bool,
    pub directive_def: Option<String>,
    pub fixed_size: Option<Dimension>,
    pub debug: bool,
    pub alternate_tooltip: Option<String>,
    pub tooltip: Option<String>,
    pub overridable_field_displayer_1: bool,
    pub overridable_field_displayer_2: bool,
}

impl TextField {
    /// Java `TextField(FieldType, String, String)`.
    pub fn new(
        field_type: Option<FieldType>,
        reference: Option<&str>,
        location_descr: Option<&str>,
    ) -> Self {
        let reference = reference.map(str::to_owned);
        let text_field = LabeledTextField::new(
            field_type.unwrap_or(FieldType::String),
            reference.as_deref().unwrap_or_default(),
        );
        Self {
            text_field,
            field_type,
            location_descr: location_descr.map(str::to_owned),
            reference,
            required: false,
            file_must_exist: false,
            directive_def: None,
            fixed_size: None,
            debug: false,
            alternate_tooltip: None,
            tooltip: None,
            overridable_field_displayer_1: false,
            overridable_field_displayer_2: false,
        }
    }
    pub fn set_columns_value(&mut self, columns: i32) {
        self.text_field.set_columns(columns);
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.text_field.set_debug(input);
    }
    pub fn get_preferred_width(&self) -> i32 {
        self.fixed_size
            .map(|size| size.width)
            .or_else(|| self.text_field.text_preferred_size.map(|size| size.width))
            .unwrap_or_else(|| self.text_field.text.chars().count() as i32)
    }
    pub fn is_boolean(&self) -> bool {
        false
    }
    pub fn is_debug(&self) -> bool {
        self.debug || self.text_field.is_debug()
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(|text| format!("<html>{text}"));
        self.text_field.set_tool_tip_text(self.tooltip.as_deref());
    }
    pub fn set_unformatted_tooltip(&mut self, text: Option<&str>) -> Option<&str> {
        self.text_field.unformatted_tooltip = text.map(str::to_owned);
        self.text_field.unformatted_tooltip.as_deref()
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.text_field.has_unformatted_tooltip()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        self.text_field
            .use_unformatted_tooltip(param_descr, directive_descr);
        self.tooltip = self.text_field.tooltip.clone();
    }
    pub fn set_preformatted_tooltip(&mut self, tooltip: Option<&str>) {
        self.text_field.set_tool_tip_text(tooltip);
        if self.alternate_tooltip.is_some() {
            self.tooltip = tooltip.map(str::to_owned);
        }
    }
    pub fn set_alternate_tooltip_text(&mut self, text: Option<&str>) {
        if self.tooltip.is_none() {
            self.tooltip = self
                .text_field
                .tooltip
                .clone()
                .filter(|text| !text.is_empty());
        }
        self.alternate_tooltip = text.map(|text| format!("<html>{text}"));
    }
    pub fn switch_tooltips(&mut self, alternate: bool) {
        self.text_field.set_tool_tip_text(if alternate {
            self.alternate_tooltip.as_deref()
        } else {
            self.tooltip.as_deref()
        });
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.text_field.tooltip.as_deref()
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &Self {
        self
    }
    pub fn set_alignment_x(&mut self, alignment_x: f32) {
        self.text_field.set_alignment_x(alignment_x);
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.text_field.set_enabled(enabled);
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.text_field.set_editable(editable);
    }
    pub fn is_enabled(&self) -> bool {
        self.text_field.is_enabled()
    }
    pub fn is_editable(&self) -> bool {
        self.text_field.is_editable()
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        self.text_field.is_different_from_checkpoint(always_check)
    }
    pub fn backup(&mut self) {
        self.text_field.backup();
    }
    pub fn restore_from_backup(&mut self) {
        self.text_field.restore_from_backup();
    }
    pub fn clear(&mut self) {
        self.text_field.clear();
    }
    pub fn is_selected(&self) -> bool {
        false
    }
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
        self.text_field
            .set_directive_default_value(self.directive_def.clone());
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    /// Java `getDocument`; document identity stays at the native widget boundary.
    pub fn get_document_identity(&self) -> usize {
        self as *const Self as usize
    }
    pub fn set_columns(&mut self) {
        if let Some(field_type) = self.field_type {
            self.text_field.set_columns(field_type.get_columns());
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.text_field.set_visible(visible);
    }
    pub fn use_default_value(&mut self) {
        self.text_field.use_default_value();
    }
    pub fn equals_default_value(&self) -> bool {
        self.text_field.equals_default_value()
    }
    pub fn equals_default_value_string(&self, value: &str) -> bool {
        self.text_field.equals_default_value_string(value)
    }
    pub fn checkpoint(&mut self) {
        self.text_field.checkpoint();
    }
    pub fn checkpoint_value(&mut self, value: &str) {
        self.text_field.checkpoint_value(value);
    }
    pub fn reset_to_checkpoint(&mut self) {
        self.text_field.reset_to_checkpoint();
    }
    pub fn get_checkpoint(&self) -> Option<&TextFieldSetting> {
        self.text_field.get_checkpoint()
    }
    pub fn set_checkpoint(&mut self, input: Option<&TextFieldSetting>) {
        self.text_field.set_checkpoint(input);
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.text_field.is_field_highlight_set()
    }
    pub fn set_field_highlight_setting(&mut self, input: Option<&TextFieldSetting>) {
        self.text_field.set_field_highlight_setting(input);
    }
    pub fn set_field_highlight(&mut self, value: &str) {
        self.text_field.set_field_highlight(value);
    }
    pub fn set_field_highlight_boolean(&mut self, value: bool) {
        self.text_field.set_field_highlight_boolean(value);
    }
    pub fn get_field_highlight(&self) -> Option<&TextFieldSetting> {
        self.text_field.get_field_highlight()
    }
    pub fn clear_field_highlight(&mut self) {
        self.text_field.clear_field_highlight();
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.text_field.equals_field_highlight()
    }
    pub fn equals_field_highlight_string(&self, value: &str) -> bool {
        self.text_field.equals_field_highlight_string(value)
    }
    pub fn focus_gained(&mut self) {}
    pub fn focus_lost(&mut self) {
        self.text_field.focus_lost();
    }
    pub fn set_reference(&mut self, reference: Option<&str>) {
        self.reference = reference.map(str::to_owned);
        self.text_field
            .set_label(self.reference.as_deref().unwrap_or_default());
    }
    pub fn set_text(&mut self, text: Option<&str>) {
        self.text_field.set_text(text.unwrap_or_default());
    }
    pub fn set_required(&mut self, required: bool) {
        self.required = required;
        self.text_field.set_required(required);
    }
    pub fn set_file_must_exist(&mut self, file_must_exist: bool) {
        self.file_must_exist = file_must_exist;
    }
    pub fn get_text_validated(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.text_field.get_text_validated(do_validation)
    }
    pub fn is_required(&self) -> bool {
        self.required && self.is_enabled()
    }
    pub fn get_description(&self) -> String {
        format!(
            "{}{}",
            self.get_quoted_label(),
            self.location_descr
                .as_ref()
                .map(|location| format!(" in {location}"))
                .unwrap_or_default()
        )
    }
    pub fn set_value_text_field(&mut self, input: Option<&Self>) {
        if let Some(input) = input {
            self.set_text(Some(&input.get_text()));
        } else {
            self.clear();
        }
    }
    pub fn set_value(&mut self, value: Option<&str>) {
        self.set_text(value);
    }
    pub fn set_value_boolean(&mut self, _value: bool) {}
    pub fn get_text(&self) -> String {
        self.text_field.get_text()
    }
    pub fn is_empty(&self) -> bool {
        self.get_text().trim().is_empty()
    }
    pub fn get_quoted_label(&self) -> String {
        self.text_field.get_quoted_label()
    }
    pub fn get_maximum_size(&self) -> Dimension {
        self.fixed_size
            .or(self.text_field.text_preferred_size)
            .unwrap_or_default()
    }
    pub fn set_maximum_size(&mut self, size: Dimension) {
        self.fixed_size = Some(size);
    }
    pub fn set_text_preferred_width(&mut self, min_width: f64) {
        let mut size = self.text_field.text_preferred_size.unwrap_or_default();
        size.width = min_width as i32;
        self.text_field.set_text_preferred_size(size);
    }
    pub fn set_text_preferred_size(&mut self, size: Dimension) {
        self.fixed_size = Some(size);
        self.text_field.set_text_preferred_size(size);
    }
    pub fn set_size(&mut self, size: Dimension) {
        self.text_field.set_text_preferred_size(size);
    }
    pub fn set_preferred_width(&mut self, width: i32) {
        self.text_field.set_text_preferred_width(width);
    }
    pub fn get_size(&self) -> Dimension {
        self.text_field.text_preferred_size.unwrap_or_default()
    }
    pub fn get_preferred_size(&self) -> Dimension {
        self.get_size()
    }
    pub fn get_name(&self) -> &str {
        self.text_field.get_name()
    }
    pub fn is_visible(&self) -> bool {
        self.text_field.is_visible()
    }
    pub fn set_name(&mut self, reference: Option<&str>) {
        self.reference = reference.map(str::to_owned);
        self.text_field
            .set_name(self.reference.as_deref().unwrap_or_default());
    }
    pub fn set_validation_set(&mut self, input: Option<ValidationSet>) {
        self.text_field.set_validation_set(input);
    }
    pub fn set_number_must_be_positive(&mut self, input: bool) {
        self.text_field.set_number_must_be_positive(input);
    }
    pub fn set_overridable_field_displayers(
        &mut self,
        field_displayer_1: bool,
        field_displayer_2: bool,
    ) {
        self.overridable_field_displayer_1 = field_displayer_1;
        self.overridable_field_displayer_2 = field_displayer_2;
    }
    pub fn get_numeric_type(&self) -> Option<NumericType> {
        self.field_type.and_then(FieldType::get_numeric_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_text_checkpoint_visibility_and_name_are_retained() {
        let mut field = TextField::new(Some(FieldType::Integer), Some("Bin:"), None);
        assert_eq!(field.get_name(), "tf.bin");
        field.set_text(Some("2"));
        field.checkpoint();
        field.set_text(Some("3"));
        assert!(field.is_different_from_checkpoint(false));
        field.set_visible(false);
        assert!(!field.is_different_from_checkpoint(false));
        assert!(field.is_different_from_checkpoint(true));
    }
    #[test]
    fn source_highlight_and_required_validation_are_retained() {
        let mut field = TextField::new(Some(FieldType::Integer), Some("Bin:"), None);
        field.set_required(true);
        assert!(field.get_text_validated(true).is_err());
        field.set_text(Some("2"));
        field.set_field_highlight("2");
        assert!(field.equals_field_highlight());
    }
}
