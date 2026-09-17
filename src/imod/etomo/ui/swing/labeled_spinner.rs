//! `IMOD/Etomo/src/etomo/ui/swing/LabeledSpinner.java`.
//!
//! The `JPanel`, `JLabel`, `JSpinner`, `SpinnerNumberModel`, formatted editor,
//! and its Swing listeners are a GUI boundary.  This module retains every piece
//! of source-owned state and the source's control, naming, range, checkpoint,
//! tooltip, and field-highlight rules, so a native GUI adapter has one direct
//! place to attach those components.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, INTEGER_NULL_VALUE};
use crate::imod::etomo::ui::swing::labeled_text_field::{
    BACKGROUND, BLACK, FIELD_HIGHLIGHT, TextFieldSetting,
};
use crate::imod::etomo::ui::swing::panel::Dimension;
use crate::imod::etomo::ui::swing::ui_utilities::Color;
use crate::imod::etomo::util::utilities;

/// The source `SpinnerNumberModel` values retained at the native Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpinnerNumberModel {
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub step_size: i32,
}

/// Java package-private final `LabeledSpinner`.
#[derive(Clone, Debug)]
pub struct LabeledSpinner {
    /// Java `defaultValue`, which is final and is used for null input.
    pub default_value: i32,
    /// Java `model`, `minimum`, and `maximum`.
    pub model: SpinnerNumberModel,
    pub minimum: i32,
    pub maximum: i32,
    pub orig_label_foreground: Option<Color>,
    pub orig_text_foreground: Option<Color>,
    /// Java `DirectiveDef` and `AutodocAttributeRetriever` are storage boundaries.
    pub directive_default_value: Option<String>,
    pub backup: Option<TextFieldSetting>,
    pub default_value_setting: Option<TextFieldSetting>,
    pub checkpoint: Option<TextFieldSetting>,
    pub field_highlight: Option<TextFieldSetting>,
    pub enabled: bool,
    pub editable: bool,
    pub unformatted_tooltip: Option<String>,
    /// State of Java `panel`, `label`, spinner, and `JFormattedTextField`.
    pub panel_visible: bool,
    pub panel_maximum_size: Option<Dimension>,
    pub panel_alignment_x: f32,
    pub label: String,
    pub label_enabled: bool,
    pub label_foreground: Color,
    pub spinner_name: String,
    pub spinner_enabled: bool,
    pub spinner_preferred_size: Option<Dimension>,
    pub spinner_maximum_size: Option<Dimension>,
    pub spinner_foreground: Color,
    pub text_background: Color,
    pub tooltip: Option<String>,
    /// Actual listener registration is Swing-specific; retain each attachment count.
    pub change_listener_count: usize,
    pub focus_listener_count: usize,
    pub mouse_listener_count: usize,
}

impl LabeledSpinner {
    /// Java private `LabeledSpinner(String,int,int,int,int,int,int)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spin_label: &str,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
        default_value: i32,
        hgap: i32,
    ) -> Self {
        let name = utilities::convert_label_to_name(Some(spin_label), true).unwrap_or_default();
        let spinner_name = format!("sp{SEPARATOR_CHAR}{name}");
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!("{spinner_name} = ");
        }
        // Java asks AWT for the maximum width and font heights.  Those values are
        // native-widget state; the source's height rule is represented by the
        // conventional unscaled one-font unit here.
        let maximum_size = Dimension {
            width: i32::MAX,
            height: 2,
        };
        let _ = hgap; // Java adds a rigid area only when this is positive.
        Self {
            default_value,
            model: SpinnerNumberModel {
                value,
                minimum,
                maximum,
                step_size,
            },
            minimum,
            maximum,
            orig_label_foreground: None,
            orig_text_foreground: None,
            directive_default_value: None,
            backup: None,
            default_value_setting: None,
            checkpoint: None,
            field_highlight: None,
            enabled: true,
            editable: true,
            unformatted_tooltip: None,
            panel_visible: true,
            panel_maximum_size: None,
            panel_alignment_x: 0.5,
            label: spin_label.to_owned(),
            label_enabled: true,
            label_foreground: BLACK,
            spinner_name,
            spinner_enabled: true,
            spinner_preferred_size: None,
            spinner_maximum_size: Some(maximum_size),
            spinner_foreground: BLACK,
            text_background: BACKGROUND,
            tooltip: None,
            change_listener_count: 0,
            focus_listener_count: 0,
            mouse_listener_count: 0,
        }
    }

    /// Java `getDefaultedInstance`.
    pub fn get_defaulted_instance(
        spin_label: &str,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
        default_value: i32,
    ) -> Self {
        Self::new(
            spin_label,
            value,
            minimum,
            maximum,
            step_size,
            default_value,
            0,
        )
    }

    /// Java five-argument `getInstance`.
    pub fn get_instance(
        spin_label: &str,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
    ) -> Self {
        Self::new(spin_label, value, minimum, maximum, step_size, value, 0)
    }

    /// Java six-argument `getInstance`.
    pub fn get_instance_with_gap(
        spin_label: &str,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
        hgap: i32,
    ) -> Self {
        Self::new(spin_label, value, minimum, maximum, step_size, value, hgap)
    }

    pub fn get_name(&self) -> &str {
        &self.spinner_name
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn is_boolean(&self) -> bool {
        false
    }
    pub fn is_debug(&self) -> bool {
        ARGUMENTS.lock().unwrap().is_debug()
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }

    /// Java `setMax`.
    pub fn set_max(&mut self, max: i32) {
        self.maximum = max;
        self.model.maximum = max;
    }
    /// Java `setModel`.
    pub fn set_model(&mut self, value: i32, minimum: i32, maximum: i32, step_size: i32) {
        self.minimum = minimum;
        self.maximum = maximum;
        self.model = SpinnerNumberModel {
            value,
            minimum,
            maximum,
            step_size,
        };
    }
    /// Java `getContainer`; the returned Swing panel is this boundary object.
    pub fn get_container(&self) -> &Self {
        self
    }
    pub fn get_label(&self) -> &str {
        &self.label
    }
    pub fn get_description(&self) -> String {
        self.get_quoted_label()
    }
    pub fn get_quoted_label(&self) -> String {
        utilities::quote_label(Some(&self.label)).unwrap_or_default()
    }

    pub fn checkpoint(&mut self) {
        self.checkpoint = Some(TextFieldSetting {
            field_type: crate::imod::etomo::ui::field_type::FieldType::Integer,
            value: Some(self.get_text()),
        });
    }
    pub fn get_checkpoint(&self) -> Option<&TextFieldSetting> {
        self.checkpoint.as_ref()
    }
    pub fn set_checkpoint(&mut self, input: Option<&TextFieldSetting>) {
        if self.checkpoint.is_none() && input.is_some_and(TextFieldSetting::is_set) {
            self.checkpoint = Some(TextFieldSetting::new(
                crate::imod::etomo::ui::field_type::FieldType::Integer,
            ));
        }
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint.copy(input);
        }
    }
    pub fn backup(&mut self) {
        self.backup = Some(TextFieldSetting {
            field_type: crate::imod::etomo::ui::field_type::FieldType::Integer,
            value: Some(self.get_value().to_string()),
        });
    }
    /// Java `setDirectiveDef`; lookup is retained as storage-boundary input.
    pub fn set_directive_default_value(&mut self, value: Option<String>) {
        self.directive_default_value = value;
    }
    pub fn get_directive_default_value(&self) -> Option<&str> {
        self.directive_default_value.as_deref()
    }
    pub fn equals_default_value(&self) -> bool {
        self.default_value_setting
            .as_ref()
            .is_some_and(|value| value.is_set() && value.equals(&self.get_text()))
    }
    pub fn equals_default_value_string(&self, value: &str) -> bool {
        self.default_value_setting
            .as_ref()
            .is_some_and(|setting| setting.is_set() && setting.equals(value))
    }
    pub fn use_default_value(&mut self) {
        if self.directive_default_value.is_none() {
            if let Some(value) = self.default_value_setting.as_mut() {
                value.reset();
            }
            return;
        }
        if self.default_value_setting.is_none() {
            self.default_value_setting = Some(TextFieldSetting {
                field_type: crate::imod::etomo::ui::field_type::FieldType::Integer,
                value: self.directive_default_value.clone(),
            });
        }
        if let Some(value) = self
            .default_value_setting
            .as_ref()
            .and_then(TextFieldSetting::get_value)
            .map(str::to_owned)
        {
            self.set_text(&value);
        }
    }
    pub fn restore_from_backup(&mut self) {
        if let Some(value) = self.backup.as_mut().and_then(|value| value.value.take()) {
            self.set_value_string(Some(&value));
        }
    }
    pub fn clear(&mut self) {
        self.model.value = self.minimum;
    }
    pub fn set_value_field(&mut self, input: Option<&str>) {
        match input {
            None => self.clear(),
            Some(value) => self.set_text(value),
        }
    }
    pub fn is_selected(&self) -> bool {
        false
    }
    pub fn is_empty(&self) -> bool {
        self.get_text().chars().all(|c| c.is_ascii_whitespace())
    }
    pub fn is_required(&self) -> bool {
        false
    }
    /// Java exposes no Spinner validation; `FieldDisplayer` is therefore not reached.
    pub fn get_text_validated(&self, _do_validation: bool) -> String {
        self.get_text()
    }
    pub fn get_text(&self) -> String {
        self.get_value().to_string()
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(TextFieldSetting::is_set)
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|value| value.equals(&self.get_text()))
    }
    pub fn equals_field_highlight_string(&self, value: &str) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|setting| setting.equals(value))
    }
    pub fn set_field_highlight(&mut self, value: Option<&str>) {
        if self.field_highlight.is_none() && value.is_some() {
            self.field_highlight = Some(TextFieldSetting::new(
                crate::imod::etomo::ui::field_type::FieldType::Integer,
            ));
            self.change_listener_count += 1;
            self.focus_listener_count += 1;
        }
        if let Some(field_highlight) = self.field_highlight.as_mut() {
            field_highlight.value = value.map(str::to_owned);
        }
        self.update_field_highlight();
    }
    pub fn clear_field_highlight(&mut self) {
        if self
            .field_highlight
            .as_ref()
            .is_some_and(TextFieldSetting::is_set)
        {
            self.field_highlight.as_mut().unwrap().reset();
            self.update_field_highlight();
        }
    }
    pub fn get_field_highlight(&self) -> Option<&TextFieldSetting> {
        self.field_highlight.as_ref()
    }
    pub fn set_field_highlight_setting(&mut self, input: Option<&TextFieldSetting>) {
        if self.field_highlight.is_none() && input.is_some_and(TextFieldSetting::is_set) {
            self.field_highlight = Some(TextFieldSetting::new(
                crate::imod::etomo::ui::field_type::FieldType::Integer,
            ));
            self.change_listener_count += 1;
            self.focus_listener_count += 1;
        }
        if let Some(field_highlight) = self.field_highlight.as_mut() {
            field_highlight.copy(input);
        }
        self.update_field_highlight();
    }
    pub fn clear_field_highlight_value(&mut self) {
        self.field_highlight
            .as_mut()
            .expect("Java fieldHighlight.reset() null dereference")
            .reset();
        self.update_field_highlight();
    }
    /// Java `stateChanged` and `focusLost` share this action.
    pub fn update_field_highlight(&mut self) {
        if self.field_highlight.is_none() || !self.is_enabled() {
            return;
        }
        let highlighted = self
            .field_highlight
            .as_ref()
            .is_some_and(|value| value.is_set() && value.equals(&self.get_value().to_string()));
        if highlighted {
            if self.orig_text_foreground.is_none() {
                self.orig_text_foreground = Some(self.spinner_foreground);
            }
            if self.orig_label_foreground.is_none() {
                self.orig_label_foreground = Some(self.label_foreground);
            }
            self.label_foreground = FIELD_HIGHLIGHT;
            self.spinner_foreground = FIELD_HIGHLIGHT;
        } else {
            if let Some(color) = self.orig_text_foreground {
                self.spinner_foreground = color;
            }
            if let Some(color) = self.orig_label_foreground {
                self.label_foreground = color;
            }
        }
    }
    pub fn reset_to_checkpoint(&mut self) {
        if let Some(value) = self
            .checkpoint
            .as_ref()
            .and_then(TextFieldSetting::get_value)
            .map(str::to_owned)
        {
            self.set_text(&value);
        }
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.is_visible()) {
            return false;
        }
        self.checkpoint
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals(&self.get_value().to_string()))
    }
    pub fn get_value(&self) -> i32 {
        self.model.value
    }
    pub fn is_in_range(&self, number: Option<&ConstEtomoNumber>) -> bool {
        let Some(number) = number else {
            return true;
        };
        number.is_null()
            || (number.ge_long(self.minimum as i64)
                && (self.maximum <= self.minimum || number.le_int(self.maximum)))
    }
    pub fn set_value_const_etomo_number(&mut self, value: &ConstEtomoNumber) {
        if value.is_null() {
            self.model.value = self.default_value;
        } else {
            self.model.value = value.get_number().int_value();
        }
    }
    pub fn set_text(&mut self, value: &str) {
        self.set_value_string(Some(value));
    }
    pub fn set_value_string(&mut self, value: Option<&str>) {
        let value = value.filter(|value| !value.chars().all(|c| c.is_ascii_whitespace()));
        self.model.value = value
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(self.default_value);
    }
    pub fn set_value_string_nonempty(&mut self, value: Option<&str>, nonempty: bool) {
        if !nonempty || value.is_some_and(|value| !value.is_empty()) {
            self.set_value_string(value);
        }
    }
    pub fn set_value_int(&mut self, value: i32) {
        self.model.value = if value == INTEGER_NULL_VALUE {
            self.default_value
        } else {
            value
        };
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.spinner_enabled = enabled && self.editable;
        self.label_enabled = enabled && self.editable;
        if enabled && self.editable {
            self.update_field_highlight();
        }
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
        if self.enabled {
            self.spinner_enabled = editable;
        }
        if self.enabled && editable {
            self.update_field_highlight();
        }
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn is_visible(&self) -> bool {
        self.panel_visible
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.panel_visible = visible;
    }
    pub fn set_highlight(&mut self, highlight: bool) {
        self.text_background = if highlight {
            Color {
                red: 255,
                green: 255,
                blue: 0,
            }
        } else {
            BACKGROUND
        };
    }
    pub fn set_text_preferred_size(&mut self, size: Dimension) {
        self.spinner_preferred_size = Some(size);
    }
    pub fn set_text_maximum_size(&mut self, size: Dimension) {
        self.spinner_maximum_size = Some(size);
    }
    pub fn set_preferred_width(&mut self, width: i32, font_size_adjustment: f32) {
        let mut size = self.spinner_preferred_size.unwrap_or_default();
        size.width = (width as f32 * font_size_adjustment).round() as i32;
        self.spinner_preferred_size = Some(size);
        self.spinner_maximum_size = Some(size);
    }
    pub fn set_maximum_size(&mut self, size: Dimension) {
        self.panel_maximum_size = Some(size);
    }
    pub fn get_label_preferred_size(&self) -> Dimension {
        Dimension {
            width: self.label.chars().count() as i32,
            height: 1,
        }
    }
    pub fn set_alignment_x(&mut self, alignment: f32) {
        self.panel_alignment_x = alignment;
    }
    pub fn set_unformatted_tooltip(&mut self, text: &str) -> String {
        self.unformatted_tooltip = Some(text.to_owned());
        text.to_owned()
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.is_some()
    }
    /// Java `TooltipFormatter` is a separate UI boundary; all three input segments are preserved.
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        let tooltip = [
            self.unformatted_tooltip.take(),
            param_descr.map(str::to_owned),
            directive_descr.map(str::to_owned),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        self.set_tool_tip_text(Some(&tooltip));
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        if let Some(tooltip) = tooltip {
            self.tooltip = Some(tooltip.to_owned());
        }
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.tooltip.as_deref()
    }
    pub fn add_mouse_listener(&mut self) {
        self.mouse_listener_count += 3;
    }
    pub fn add_change_listener(&mut self) {
        self.change_listener_count += 1;
    }

    /// Java `focusGained`; it intentionally has no source side effect.
    #[allow(non_snake_case)]
    pub fn focusGained(&mut self) {}

    /// Java private `getTextField`.  The native spinner owns its editable text
    /// state directly, so this is its read boundary rather than a Swing editor.
    #[allow(non_snake_case)]
    pub fn getTextField(&self) -> &Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_name_defaults_ranges_and_nulls_are_preserved() {
        let mut spinner = LabeledSpinner::get_defaulted_instance("Pixel size", 3, 1, 9, 2, 4);
        assert_eq!(spinner.get_name(), "sp.pixel-size");
        spinner.set_value_string(Some("\t"));
        assert_eq!(spinner.get_value(), 4);
        spinner.clear();
        assert_eq!(spinner.get_value(), 1);
        assert!(spinner.is_in_range(None));
        spinner.set_max(2);
        assert_eq!(spinner.model.maximum, 2);
    }

    #[test]
    fn highlight_checkpoint_backup_and_control_rules_match_source() {
        let mut spinner = LabeledSpinner::get_instance("Count", 2, 0, 8, 1);
        spinner.set_field_highlight(Some("2"));
        assert_eq!(spinner.spinner_foreground, FIELD_HIGHLIGHT);
        spinner.set_value_int(3);
        spinner.update_field_highlight();
        assert_eq!(spinner.spinner_foreground, BLACK);
        spinner.checkpoint();
        spinner.backup();
        spinner.set_value_int(5);
        assert!(spinner.is_different_from_checkpoint(true));
        spinner.restore_from_backup();
        assert_eq!(spinner.get_value(), 3);
        spinner.set_editable(false);
        assert!(spinner.is_enabled());
        assert!(!spinner.spinner_enabled);
        assert!(spinner.label_enabled);
    }
}
