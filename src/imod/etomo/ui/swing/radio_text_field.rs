//! `IMOD/Etomo/src/etomo/ui/swing/RadioTextField.java`.
//!
//! The `JPanel`/`BoxLayout`, `ButtonGroup`, and native widget listener delivery
//! stay at the GUI boundary.  The Java source unit's paired radio/text field
//! ownership and all its value, display, checkpoint, and tooltip transitions
//! are retained here.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use crate::imod::etomo::{r#type::const_etomo_number::ConstEtomoNumber, ui::field_type::FieldType};

use super::{
    check_box::BooleanFieldSetting,
    labeled_text_field::{FieldValidationFailedException, TextFieldSetting},
    radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup},
    radio_button_interface::RadioButtonInterface,
    text_field::TextField,
};

/// Java final `rootPanel` state at the native JPanel boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RadioTextFieldPanelBoundary {
    pub box_layout_x_axis: bool,
    pub child_count: usize,
}

/// Java `FieldSettingBundle` composed by this class.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadioTextFieldSettingBundle {
    pub boolean_setting: Option<BooleanFieldSetting>,
    pub text_setting: Option<TextFieldSetting>,
}

/// Java public final `RadioTextField`.
#[derive(Clone, Debug)]
pub struct RadioTextField {
    pub root_panel: RadioTextFieldPanelBoundary,
    pub radio_button: RadioButton,
    pub text_field: TextField,
    pub debug: bool,
    pub directive_def: Option<String>,
    pub enabled: bool,
    pub editable: bool,
}

impl RadioTextField {
    /// Java private six-argument constructor.  `None` group is the source's
    /// permitted null group boundary used by existing direct consumers.
    fn new_full(
        field_type: FieldType,
        label: Option<&str>,
        enumerated_type: Option<EnumeratedTypeBoundary>,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
        location_descr: Option<&str>,
        alternate_label: Option<&str>,
    ) -> Self {
        let mut radio_button = match enumerated_type {
            Some(enumerated_type) => RadioButton::new_with_enumerated_type(
                label.map(str::to_owned),
                enumerated_type,
                group,
            ),
            None => match group {
                Some(group) => RadioButton::new_in_group(label.unwrap_or_default(), group),
                None => RadioButton::new(label.unwrap_or_default()),
            },
        };
        let mut text_field = TextField::new(Some(field_type), label, location_descr);
        if let Some(alternate_label) = alternate_label {
            radio_button.set_name(alternate_label);
            text_field.set_name(Some(alternate_label));
        }
        let mut value = Self {
            root_panel: RadioTextFieldPanelBoundary {
                box_layout_x_axis: false,
                child_count: 0,
            },
            radio_button,
            text_field,
            debug: false,
            directive_def: None,
            enabled: true,
            editable: true,
        };
        value.init();
        value
    }

    /// Compatibility construction of Java `getInstance(FieldType,String,ButtonGroup)`.
    pub fn get_instance(
        field_type: FieldType,
        label: &str,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
    ) -> Self {
        Self::new_full(field_type, Some(label), None, group, None, None)
    }

    /// Java `getInstanceWithAlternateLabel`.
    pub fn get_instance_with_alternate_label(
        field_type: FieldType,
        label: &str,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
        alternate_label: &str,
    ) -> Self {
        Self::new_full(
            field_type,
            Some(label),
            None,
            group,
            None,
            Some(alternate_label),
        )
    }

    /// Java `getInstance(FieldType,EnumeratedType,ButtonGroup)`.
    pub fn get_enum_instance(
        field_type: FieldType,
        enumerated_type: EnumeratedTypeBoundary,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
    ) -> Self {
        Self::new_full(field_type, None, Some(enumerated_type), group, None, None)
    }

    /// Java `getInstance(FieldType,String,ButtonGroup,String)`.
    pub fn get_location_instance(
        field_type: FieldType,
        label: &str,
        group: Option<Rc<RefCell<RadioButtonGroup>>>,
        location_descr: &str,
    ) -> Self {
        Self::new_full(
            field_type,
            Some(label),
            None,
            group,
            Some(location_descr),
            None,
        )
    }

    /// Existing translated direct-consumer construction; equivalent to a null Java group.
    pub fn new(field_type: FieldType, label: &str) -> Self {
        Self::get_instance(field_type, label, None)
    }

    /// Java private `init()`.
    fn init(&mut self) {
        self.root_panel.box_layout_x_axis = true;
        self.root_panel.child_count = 2;
        self.update_display();
    }

    pub fn get_name(&self) -> Option<&str> {
        self.radio_button.get_name()
    }
    pub fn get_field(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    /// Rust GUI-boundary equivalent of Java `getContainer()`.
    pub fn getContainer(&self) -> &RadioTextFieldPanelBoundary {
        &self.root_panel
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn is_boolean(&self) -> bool {
        true
    }
    pub fn is_debug(&self) -> bool {
        self.debug || self.radio_button.is_debug()
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        self.radio_button.equals_selected_string_value(value)
    }
    pub fn set_text_preferred_width(&mut self, min_width: f64) {
        self.text_field.set_text_preferred_width(min_width);
    }
    pub fn set_text(&mut self, text: &str) {
        self.text_field.set_text(Some(text));
    }
    pub fn set_text_number(&mut self, text: impl std::fmt::Display) {
        self.text_field.set_text(Some(&text.to_string()));
    }
    pub fn set_text_allow_empty(&mut self, text: Option<&str>, allow_empty: bool) {
        if allow_empty || text.is_some_and(|value| !value.is_empty()) {
            self.text_field.set_text(text);
        }
    }
    pub fn backup(&mut self) {
        self.radio_button.backup();
        self.text_field.backup();
    }
    pub fn restore_from_backup(&mut self) {
        self.radio_button.restore_from_backup();
        self.text_field.restore_from_backup();
        self.update_display();
    }
    pub fn clear(&mut self) {
        self.radio_button.clear();
        self.text_field.clear();
        self.update_display();
    }
    pub fn set_value(&mut self, input: Option<&Self>) {
        if let Some(input) = input {
            self.radio_button.set_value(Some(&input.radio_button));
            self.text_field
                .set_value(Some(&input.text_field.get_text()));
        } else {
            self.radio_button.set_value(None);
            self.text_field.set_value(None);
        }
        self.update_display();
    }
    pub fn set_value_string(&mut self, value: Option<&str>) {
        self.text_field.set_value(value);
    }
    pub fn set_value_boolean(&mut self, value: bool) {
        self.radio_button.set_value_boolean(value);
        self.update_display();
    }
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
    }
    pub fn use_default_value(&mut self) {
        self.radio_button.use_default_value(None);
        self.text_field.use_default_value();
        self.update_display();
    }
    pub fn equals_default_value(&self) -> bool {
        self.radio_button.equals_default_value() && self.text_field.equals_default_value()
    }
    pub fn equals_default_value_string(&self, value: &str) -> bool {
        self.radio_button.equals_default_value_string(Some(value))
            && self.text_field.equals_default_value_string(value)
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.radio_button.is_field_highlight_set() || self.text_field.is_field_highlight_set()
    }
    pub fn set_field_highlight(&mut self, text: &str) {
        self.text_field.set_field_highlight(text);
    }
    pub fn set_field_highlight_boolean(&mut self, value: bool) {
        self.radio_button.set_field_highlight(value);
    }
    pub fn get_field_highlight(&self) -> RadioTextFieldSettingBundle {
        RadioTextFieldSettingBundle {
            boolean_setting: self.radio_button.get_field_highlight().cloned(),
            text_setting: self.text_field.get_field_highlight().cloned(),
        }
    }
    pub fn set_field_highlight_bundle(&mut self, input: &RadioTextFieldSettingBundle) {
        self.radio_button
            .set_field_highlight_setting(input.boolean_setting.as_ref());
        self.text_field
            .set_field_highlight_setting(input.text_setting.as_ref());
    }
    pub fn clear_field_highlight(&mut self) {
        self.text_field.clear_field_highlight();
        self.radio_button.clear_field_highlight();
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.text_field.equals_field_highlight() && self.radio_button.equals_field_highlight()
    }
    pub fn equals_field_highlight_string(&self, value: &str) -> bool {
        self.text_field.equals_field_highlight_string(value)
            && self.radio_button.equals_field_highlight_string(Some(value))
    }
    pub fn checkpoint(&mut self) {
        self.radio_button.checkpoint();
        self.text_field.checkpoint();
    }
    pub fn set_checkpoint(&mut self, input: &RadioTextFieldSettingBundle) {
        self.radio_button
            .set_checkpoint(input.boolean_setting.as_ref());
        self.text_field.set_checkpoint(input.text_setting.as_ref());
    }
    pub fn get_checkpoint(&self) -> RadioTextFieldSettingBundle {
        RadioTextFieldSettingBundle {
            boolean_setting: self.radio_button.get_checkpoint().cloned(),
            text_setting: self.text_field.get_checkpoint().cloned(),
        }
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        self.radio_button.is_different_from_checkpoint(always_check)
            || self.text_field.is_different_from_checkpoint(always_check)
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
        self.radio_button.set_debug(debug);
    }
    pub fn set_label(&mut self, label: &str) {
        self.radio_button.set_text(label);
        self.text_field.set_reference(Some(label));
    }
    pub fn set_required(&mut self, required: bool) {
        self.text_field.set_required(required);
    }
    pub fn get_label(&self) -> &str {
        self.radio_button.get_text()
    }
    pub fn get_description(&self) -> Option<String> {
        self.radio_button.get_description()
    }
    pub fn get_quoted_label(&self) -> Option<String> {
        self.radio_button.get_quoted_label()
    }
    pub fn is_required(&self) -> bool {
        self.text_field.is_required()
    }
    pub fn get_text(&self, do_validation: bool) -> Result<String, FieldValidationFailedException> {
        self.text_field
            .get_text_validated(do_validation)
            .map(|value| {
                if value.trim().is_empty() {
                    String::new()
                } else {
                    value
                }
            })
    }
    pub fn get_text_unvalidated(&self) -> String {
        let value = self.text_field.get_text();
        if value.trim().is_empty() {
            String::new()
        } else {
            value
        }
    }
    pub fn is_empty(&self) -> bool {
        self.text_field.is_empty()
    }
    pub fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.radio_button.set_enabled(enabled && self.editable);
        self.update_display();
    }
    /// Java private `updateDisplay()`.
    fn update_display(&mut self) {
        self.text_field
            .set_enabled(self.enabled && self.radio_button.is_selected());
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
        if self.enabled {
            self.radio_button.set_enabled(editable);
            self.text_field.set_editable(editable);
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.radio_button.set_visible(visible);
        self.text_field.set_visible(visible);
    }
    pub fn set_selected(&mut self, selected: bool) {
        self.radio_button.set_selected(selected);
        self.update_display();
    }
    pub fn set_selected_number(&mut self, selected: Option<&ConstEtomoNumber>, allow_empty: bool) {
        if allow_empty || selected.is_some_and(|value| !value.is_null()) {
            self.set_selected(selected.is_some_and(ConstEtomoNumber::is));
        }
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.radio_button.set_tool_tip_text(text);
        self.text_field.set_tool_tip_text(text);
    }
    pub fn set_unformatted_tooltip(&mut self, text: Option<&str>) -> Option<&str> {
        self.radio_button.set_unformatted_tooltip(text);
        self.text_field.set_unformatted_tooltip(text)
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.radio_button.has_unformatted_tooltip() || self.text_field.has_unformatted_tooltip()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        self.radio_button
            .use_unformatted_tooltip(param_descr, directive_descr);
        self.text_field
            .use_unformatted_tooltip(param_descr, directive_descr);
    }
    pub fn set_text_field_unformatted_tooltip(&mut self, text: Option<&str>) {
        self.text_field.set_unformatted_tooltip(text);
    }
    pub fn set_radio_button_tool_tip_text(&mut self, text: Option<&str>) {
        self.radio_button.set_tool_tip_text(text);
    }
    pub fn set_text_field_tool_tip_text(&mut self, text: Option<&str>) {
        self.text_field.set_tool_tip_text(text);
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.text_field.get_tooltip()
    }
    pub fn add_action_listener(&mut self) {
        self.radio_button.add_action_listener();
    }
    pub fn get_action_command(&self) -> &str {
        self.radio_button.get_action_command()
    }
    /// Java `validate()`: returns null for valid state.
    pub fn validate(&self) -> Option<&'static str> {
        let radio_name = self.radio_button.get_name().unwrap_or_default();
        let text_name = self.text_field.get_name();
        if !text_name.starts_with("tf") || !radio_name.ends_with(&text_name[2..]) {
            return Some("Fields should have the same name, except for the prefix");
        }
        if !self.enabled && self.text_field.is_enabled() {
            return Some("Fields should enable and disable together");
        }
        if !self.radio_button.is_selected() && self.text_field.is_enabled() {
            return Some("Text field should be disabled when radio button is not selected");
        }
        if self.enabled && self.radio_button.is_selected() && !self.text_field.is_enabled() {
            return Some("text field should be enabled when radio button is selected");
        }
        None
    }
}

impl RadioButtonInterface for RadioTextField {
    fn msg_selected(&mut self) {
        self.update_display();
    }
    fn get_enumerated_type(&self) -> Option<&EnumeratedTypeBoundary> {
        self.radio_button.get_enumerated_type()
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_display_tracks_selected_and_enabled_source_state() {
        let mut value = RadioTextField::new(FieldType::Integer, "Count");
        assert!(!value.text_field.is_enabled());
        value.set_selected(true);
        assert!(value.text_field.is_enabled());
        value.set_enabled(false);
        assert!(!value.text_field.is_enabled());
        assert!(value.validate().is_none());
    }
}
