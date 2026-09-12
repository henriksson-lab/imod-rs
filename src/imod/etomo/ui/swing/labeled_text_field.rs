//! `IMOD/Etomo/src/etomo/ui/swing/LabeledTextField.java`.
//!
//! The `JPanel`, `JLabel`, `JTextField`, document, and listener calls in the
//! Java unit are represented by their source-visible state.  Installing actual
//! Swing listeners and painting colours remain a native GUI boundary; keeping
//! that boundary explicit lets eTomo's field naming, checkpointing, formatting,
//! validation, and highlight rules run unchanged in a Rust GUI backend.
#![allow(dead_code)]

use std::path::Path;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

use super::panel::Dimension;
use super::ui_utilities::{Color, UiUtilities};

/// Java `Colors.FIELD_HIGHLIGHT` as an explicit theme boundary.
pub const FIELD_HIGHLIGHT: Color = Color {
    red: 0,
    green: 0,
    blue: 255,
};
pub const BACKGROUND: Color = Color {
    red: 255,
    green: 255,
    blue: 255,
};
pub const BLACK: Color = Color {
    red: 0,
    green: 0,
    blue: 0,
};

/// Java `TextFieldSetting` state used by this source unit.  Its storage class
/// has not yet become a separate translated module, so this is the exact
/// text/value state at the `TextFieldSetting` dependency boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextFieldSetting {
    pub field_type: FieldType,
    pub value: Option<String>,
}

impl TextFieldSetting {
    pub fn new(field_type: FieldType) -> Self {
        Self {
            field_type,
            value: None,
        }
    }
    pub fn is_set(&self) -> bool {
        self.value.is_some()
    }
    pub fn set(&mut self, value: impl Into<String>) {
        self.value = Some(value.into());
    }
    pub fn reset(&mut self) {
        self.value = None;
    }
    pub fn get_value(&self) -> Option<&str> {
        self.value.as_deref()
    }
    pub fn equals(&self, value: &str) -> bool {
        self.value.as_deref() == Some(value)
    }
    pub fn copy(&mut self, input: Option<&TextFieldSetting>) {
        self.value = input.and_then(|input| input.value.clone());
    }
}

/// Source data owned by Java `ValidationSet` at this unit's dependency boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationSet {
    pub number_must_be_positive: bool,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub parsable_string: bool,
}

/// Java `FieldValidationFailedException`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldValidationFailedException(pub String);

impl std::fmt::Display for FieldValidationFailedException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for FieldValidationFailedException {}

/// Java `LabeledTextField`, including the state held by its three Swing widgets.
#[derive(Clone, Debug)]
pub struct LabeledTextField {
    pub field_type: FieldType,
    pub location_descr: Option<String>,
    pub max_array_size: Option<usize>,
    pub debug: bool,
    pub orig_text_foreground: Option<Color>,
    pub orig_label_foreground: Option<Color>,
    /// Java `DirectiveDef`; directive-default lookup is an explicit storage boundary.
    pub directive_default_value: Option<String>,
    pub max_decimal_places: Option<i32>,
    pub backup: Option<TextFieldSetting>,
    pub default_value: Option<TextFieldSetting>,
    pub field_highlight: Option<TextFieldSetting>,
    pub checkpoint_value: Option<TextFieldSetting>,
    pub required: bool,
    pub validation_set: Option<ValidationSet>,
    pub unformatted_tooltip: Option<String>,
    pub label: String,
    pub text: String,
    pub name: String,
    pub action_command: String,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub columns: i32,
    pub alignment_x: f32,
    pub horizontal_alignment_right: bool,
    pub text_preferred_size: Option<Dimension>,
    pub text_minimum_size: Option<Dimension>,
    pub panel_preferred_size: Option<Dimension>,
    pub tooltip: Option<String>,
    pub text_foreground: Color,
    pub label_foreground: Color,
    pub background: Color,
    /// Listener attachment is a native widget boundary, retained as counts.
    pub action_listener_count: usize,
    pub focus_listener_count: usize,
    pub document_listener_count: usize,
    pub mouse_listener_count: usize,
}

impl LabeledTextField {
    /// Java private `LabeledTextField(FieldType,int,String,int,String)`.
    pub fn new_with_max_array_size_and_gap(
        field_type: FieldType,
        max_array_size: i32,
        tf_label: &str,
        _hgap: i32,
        location_descr: Option<&str>,
    ) -> Self {
        let mut field = Self {
            field_type,
            location_descr: location_descr.map(str::to_owned),
            max_array_size: (max_array_size >= 0).then_some(max_array_size as usize),
            debug: false,
            orig_text_foreground: None,
            orig_label_foreground: None,
            directive_default_value: None,
            max_decimal_places: None,
            backup: None,
            default_value: None,
            field_highlight: None,
            checkpoint_value: None,
            required: false,
            validation_set: None,
            unformatted_tooltip: None,
            label: String::new(),
            text: String::new(),
            name: String::new(),
            action_command: tf_label.to_owned(),
            enabled: true,
            editable: true,
            visible: true,
            columns: field_type.get_columns(),
            alignment_x: 0.5,
            horizontal_alignment_right: field_type == FieldType::File,
            text_preferred_size: None,
            text_minimum_size: None,
            panel_preferred_size: None,
            tooltip: None,
            text_foreground: BLACK,
            label_foreground: BLACK,
            background: BACKGROUND,
            action_listener_count: 0,
            focus_listener_count: 0,
            document_listener_count: 0,
            mouse_listener_count: 0,
        };
        field.set_label(tf_label);
        field
    }

    /// Java `LabeledTextField(FieldType,String)`.
    pub fn new(field_type: FieldType, tf_label: &str) -> Self {
        Self::new_with_max_array_size_and_gap(field_type, -1, tf_label, 0, None)
    }
    /// Java `LabeledTextField(FieldType,int,String)`.
    pub fn new_with_max_array_size(
        field_type: FieldType,
        max_array_size: i32,
        tf_label: &str,
    ) -> Self {
        Self::new_with_max_array_size_and_gap(field_type, max_array_size, tf_label, 0, None)
    }
    /// Java package-private `(FieldType,String,String)` constructor.
    pub fn new_with_location(field_type: FieldType, tf_label: &str, location_descr: &str) -> Self {
        Self::new_with_max_array_size_and_gap(field_type, -1, tf_label, 0, Some(location_descr))
    }
    /// Java package-private `(FieldType,String,int)` constructor.
    pub fn new_with_gap(field_type: FieldType, tf_label: &str, hgap: i32) -> Self {
        Self::new_with_max_array_size_and_gap(field_type, -1, tf_label, hgap, None)
    }
    /// Java `getNumericInstance(String, EtomoNumber.Type)`; `floating_point` is DOUBLE.
    pub fn get_numeric_instance(tf_label: &str, floating_point: bool) -> Self {
        Self::new(
            if floating_point {
                FieldType::FloatingPoint
            } else {
                FieldType::Integer
            },
            tf_label,
        )
    }
    /// Java `getNumericInstance(String)`.
    pub fn get_integer_instance(tf_label: &str) -> Self {
        Self::get_numeric_instance(tf_label, false)
    }

    pub fn set_max_decimal_places(&mut self, digits: i32) {
        self.max_decimal_places = Some(digits);
    }
    /// Java `isDebug()`: field-local state ORs the director argument.
    pub fn is_debug(&self) -> bool {
        self.debug || ARGUMENTS.lock().unwrap().is_debug()
    }
    /// Java private `paramString()`.
    pub fn param_string(&self) -> String {
        format!("label={},textField={}", self.label, self.text)
    }
    /// Java `equals(Object)` at the native `JTextField` identity boundary.
    pub fn equals_text_field_identity(
        &self,
        text_field_identity: usize,
        self_identity: usize,
    ) -> bool {
        text_field_identity == self_identity
    }
    /// Java package-private `equals(Document)` at the native document boundary.
    pub fn equals_document(
        &self,
        document_identity: usize,
        text_field_document_identity: usize,
    ) -> bool {
        document_identity == text_field_document_identity
    }
    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn is_boolean(&self) -> bool {
        false
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }
    /// Java `setName(String)`.
    pub fn set_name(&mut self, tf_label: &str) {
        let name = utilities::convert_label_to_name(Some(tf_label), false).unwrap_or_default();
        self.name = format!("tf{SEPARATOR_CHAR}{name}");
    }
    pub fn checkpoint(&mut self) {
        self.checkpoint_value = Some(TextFieldSetting {
            field_type: self.field_type,
            value: Some(self.text.clone()),
        });
    }
    pub fn checkpoint_value(&mut self, value: impl Into<String>) {
        self.checkpoint_value = Some(TextFieldSetting {
            field_type: self.field_type,
            value: Some(value.into()),
        });
    }
    pub fn backup(&mut self) {
        self.backup = Some(TextFieldSetting {
            field_type: self.field_type,
            value: Some(self.text.clone()),
        });
    }
    pub fn restore_from_backup(&mut self) {
        if let Some(backup) = self.backup.as_mut()
            && let Some(value) = backup.value.take()
        {
            self.set_text(&value);
        }
    }
    /// `setDirectiveDef`; the `DirectiveDef` identity itself is an untranslated storage boundary.
    pub fn set_directive_default_value(&mut self, value: Option<String>) {
        self.directive_default_value = value;
    }
    pub fn use_default_value(&mut self) {
        if self.directive_default_value.is_none() {
            if let Some(default_value) = self.default_value.as_mut() {
                default_value.reset();
            }
            return;
        }
        if self.default_value.is_none() {
            self.default_value = Some(TextFieldSetting {
                field_type: self.field_type,
                value: self.directive_default_value.clone(),
            });
        }
        if let Some(value) = self
            .default_value
            .as_ref()
            .and_then(TextFieldSetting::get_value)
            .map(str::to_owned)
        {
            self.set_text(&value);
        }
    }
    pub fn equals_default_value(&self) -> bool {
        self.default_value
            .as_ref()
            .is_some_and(|value| value.equals(&self.text))
    }
    pub fn equals_default_value_string(&self, value: &str) -> bool {
        self.default_value
            .as_ref()
            .is_some_and(|setting| setting.equals(value))
    }
    pub fn set_checkpoint(&mut self, input: Option<&TextFieldSetting>) {
        if input.is_some_and(TextFieldSetting::is_set) || self.checkpoint_value.is_some() {
            let checkpoint = self
                .checkpoint_value
                .get_or_insert_with(|| TextFieldSetting::new(self.field_type));
            checkpoint.copy(input);
        }
    }
    pub fn get_checkpoint(&self) -> Option<&TextFieldSetting> {
        self.checkpoint_value.as_ref()
    }
    pub fn reset_to_checkpoint(&mut self) {
        if let Some(value) = self
            .checkpoint_value
            .as_ref()
            .and_then(TextFieldSetting::get_value)
            .map(str::to_owned)
        {
            self.set_text(&value);
        }
    }
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(TextFieldSetting::is_set)
    }
    pub fn set_field_highlight(&mut self, value: &str) {
        self.field_highlight = Some(TextFieldSetting {
            field_type: self.field_type,
            value: Some(value.into()),
        });
        self.focus_listener_count += 1;
        self.update_field_highlight();
    }
    pub fn set_field_highlight_boolean(&mut self, _value: bool) {}
    pub fn set_field_highlight_setting(&mut self, input: Option<&TextFieldSetting>) {
        if input.is_some_and(TextFieldSetting::is_set) || self.field_highlight.is_some() {
            let highlight = self
                .field_highlight
                .get_or_insert_with(|| TextFieldSetting::new(self.field_type));
            highlight.copy(input);
            self.update_field_highlight();
        }
    }
    pub fn clear_field_highlight(&mut self) {
        if self.is_field_highlight_set() {
            self.field_highlight.as_mut().unwrap().reset();
            self.update_field_highlight();
        }
    }
    pub fn get_field_highlight(&self) -> Option<&TextFieldSetting> {
        self.field_highlight.as_ref()
    }
    pub fn get_field_type(&self) -> FieldType {
        self.field_type
    }
    pub fn equals_field_highlight(&self) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|value| value.equals(&self.text))
    }
    pub fn equals_field_highlight_string(&self, value: &str) -> bool {
        self.field_highlight
            .as_ref()
            .is_some_and(|setting| setting.equals(value))
    }
    pub fn focus_gained(&mut self) {}
    pub fn focus_lost(&mut self) {
        self.update_field_highlight();
    }
    /// Java `updateFieldHighlight`.
    pub fn update_field_highlight(&mut self) {
        if self.field_highlight.is_none() || !self.enabled {
            return;
        }
        if self.equals_field_highlight() {
            if self.orig_text_foreground.is_none() {
                self.orig_text_foreground = Some(self.text_foreground);
            }
            if self.orig_label_foreground.is_none() {
                self.orig_label_foreground = Some(self.label_foreground);
            }
            self.text_foreground = FIELD_HIGHLIGHT;
            self.label_foreground = FIELD_HIGHLIGHT;
        } else {
            if let Some(color) = self.orig_text_foreground {
                self.text_foreground = color;
            }
            if let Some(color) = self.orig_label_foreground {
                self.label_foreground = color;
            }
        }
    }
    pub fn get_action_command(&self) -> &str {
        &self.label
    }
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
    pub fn add_focus_listener(&mut self) {
        self.focus_listener_count += 1;
    }
    pub fn remove_focus_listener(&mut self) {
        self.focus_listener_count = self.focus_listener_count.saturating_sub(1);
    }
    pub fn is_different_from_checkpoint_default(&self) -> bool {
        self.is_different_from_checkpoint(false)
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.enabled || !self.visible) {
            false
        } else {
            self.checkpoint_value
                .as_ref()
                .is_none_or(|value| !value.equals(&self.text))
        }
    }
    pub fn clear(&mut self) {
        self.text.clear();
    }
    pub fn set_value(&mut self, input: Option<&str>) {
        self.set_text(input.unwrap_or_default());
    }
    pub fn set_value_boolean(&mut self, _value: bool) {}
    pub fn is_selected(&self) -> bool {
        false
    }
    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    /// Java `getComponent()` and `getContainer()` both expose Java's `panel`.
    pub fn get_component(&self) -> &Self {
        self
    }
    pub fn get_container(&self) -> &Self {
        self
    }
    /// Java package-private `getField()` at the native `JTextField` boundary.
    pub fn get_field(&self) -> &str {
        &self.text
    }
    pub fn equals_text(&self, text: Option<&str>) -> bool {
        match text {
            Some(text) => self.text.trim() == text.trim(),
            None => false,
        }
    }
    pub fn set_highlight(&mut self, highlight: bool) {
        self.background = if highlight {
            Color {
                red: 255,
                green: 255,
                blue: 0,
            }
        } else {
            BACKGROUND
        };
    }
    pub fn get_label(&self) -> &str {
        &self.label
    }
    pub fn get_quoted_label(&self) -> String {
        utilities::quote_label(Some(&self.label)).unwrap_or_default()
    }
    pub fn set_label(&mut self, label: &str) {
        self.label = label.into();
        self.set_name(label);
    }
    pub fn set_required(&mut self, required: bool) {
        self.required = required;
    }
    pub fn set_validation_set(&mut self, input: Option<ValidationSet>) {
        match (&mut self.validation_set, input) {
            (None, input) => self.validation_set = input,
            (Some(current), Some(input)) => *current = input,
            (Some(current), None) => *current = ValidationSet::default(),
        };
    }
    pub fn set_number_must_be_positive(&mut self, value: bool) {
        self.validation_set
            .get_or_insert_default()
            .number_must_be_positive = value;
    }
    pub fn set_minimum(&mut self, value: f64) {
        self.validation_set.get_or_insert_default().minimum = Some(value);
    }
    pub fn set_maximum(&mut self, value: f64) {
        self.validation_set.get_or_insert_default().maximum = Some(value);
    }
    pub fn set_parsable_string(&mut self, value: bool) {
        self.validation_set.get_or_insert_default().parsable_string = value;
    }
    pub fn is_required(&self) -> bool {
        self.required && self.enabled
    }
    pub fn get_description(&self) -> String {
        format!(
            "{}{}",
            self.get_quoted_label(),
            self.location_descr
                .as_ref()
                .map(|value| format!(" in {value}"))
                .unwrap_or_default()
        )
    }
    /// Java `handleValidation(String, FieldDisplayer, FieldDisplayer)`.
    /// FieldDisplayer notification is an explicit GUI boundary; this value is
    /// the source's rejection path.
    pub fn handle_validation(&self, _errmsg: &str) -> bool {
        false
    }
    /// Java `getText(boolean, FieldDisplayer, FieldDisplayer)`; FieldDisplayer notification is a GUI boundary.
    pub fn get_text_validated(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        if do_validation && self.enabled {
            self.validate_text()?;
        }
        Ok(self.text.clone())
    }
    pub fn get_text(&self) -> String {
        self.text.clone()
    }
    pub fn is_empty(&self) -> bool {
        self.text
            .chars()
            .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r'))
    }
    pub fn set_text_file(&mut self, file: &Path) {
        self.text = utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
    }
    pub fn set_text(&mut self, text: &str) {
        self.text = Self::round_text(text, self.max_decimal_places);
    }
    pub fn set_text_number(&mut self, value: impl std::fmt::Display) {
        self.set_text(&value.to_string());
    }
    pub fn set_non_empty_text(&mut self, text: Option<&str>) {
        if let Some(text) = text.filter(|text| !text.is_empty()) {
            self.set_text(text);
        }
    }
    pub fn set_text_allow_empty(&mut self, text: Option<&str>, allow_empty: bool) {
        if allow_empty || text.is_some_and(|text| !text.is_empty()) {
            self.set_text(text.unwrap_or_default());
        }
    }
    pub fn set_text_field_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            self.update_field_highlight();
        }
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            self.update_field_highlight();
        }
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn is_editable(&self) -> bool {
        self.editable
    }
    pub fn is_visible(&self) -> bool {
        self.visible
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    pub fn add_document_listener(&mut self) {
        self.document_listener_count += 1;
    }
    pub fn set_text_preferred_size(&mut self, size: Dimension) {
        self.text_preferred_size = Some(size);
    }
    pub fn set_preferred_width(&mut self, width: i32, user_font_size: Option<i32>) {
        self.text_preferred_size = Some(UiUtilities::calc_new_text_field_size(
            self.text_preferred_size,
            width,
            true,
            user_font_size,
        ));
    }
    pub fn set_text_preferred_width(&mut self, min_width: i32) {
        let mut size = self.text_preferred_size.unwrap_or_default();
        size.width = min_width;
        self.text_preferred_size = Some(size);
    }
    pub fn set_minimum_width(&mut self, min_width: i32) {
        let mut size = self.text_preferred_size.unwrap_or_default();
        size.width = min_width;
        self.text_minimum_size = Some(size);
    }
    pub fn get_label_preferred_size(&self) -> Dimension {
        Dimension {
            width: self.label.chars().count() as i32,
            height: 1,
        }
    }
    pub fn set_columns(&mut self, columns: i32) {
        self.columns = columns;
    }
    pub fn set_alignment_x(&mut self, alignment: f32) {
        self.alignment_x = alignment;
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    pub fn set_unformatted_tooltip(&mut self, text: &str) -> String {
        self.unformatted_tooltip = Some(text.into());
        text.into()
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.is_some()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        let text = [
            self.unformatted_tooltip.take(),
            param_descr.map(str::to_owned),
            directive_descr.map(str::to_owned),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        self.set_tool_tip_text(Some(&text));
    }
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        if let Some(tooltip) = tooltip {
            self.tooltip = Some(tooltip.into());
        }
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.tooltip.as_deref()
    }
    pub fn add_mouse_listener(&mut self) {
        self.mouse_listener_count += 3;
    }

    fn round_text(text: &str, max_decimal_places: Option<i32>) -> String {
        let Some(digits) = max_decimal_places else {
            return text.to_owned();
        };
        if digits < 0 {
            return text.to_owned();
        }
        match text.parse::<f64>() {
            Ok(value) => format!("{value:.precision$}", precision = digits as usize),
            Err(_) => text.to_owned(),
        }
    }
    /// Source `FieldValidator.validateText` branch made local until `FieldValidator.java` lands.
    fn validate_text(&self) -> Result<(), FieldValidationFailedException> {
        let text = self.text.trim();
        if self.required && text.is_empty() {
            return Err(FieldValidationFailedException(format!(
                "{} is required",
                self.get_description()
            )));
        }
        if text.is_empty() {
            return Ok(());
        }
        let numeric = matches!(
            self.field_type,
            FieldType::Integer
                | FieldType::FloatingPoint
                | FieldType::IntegerPair
                | FieldType::FloatingPointPair
                | FieldType::IntegerTriple
                | FieldType::FloatingPointArray
                | FieldType::IntegerArray
                | FieldType::IntegerList
                | FieldType::MatlabIntegerArray
        );
        if !numeric {
            return Ok(());
        }
        let values: Vec<&str> = text
            .split(|c: char| {
                c == ','
                    || c.is_ascii_whitespace()
                    || (matches!(self.field_type, FieldType::IntegerList) && c == '-')
                    || (matches!(self.field_type, FieldType::MatlabIntegerArray) && c == ':')
            })
            .filter(|value| !value.is_empty())
            .collect();
        if self.field_type.has_required_size()
            && values.len() != self.field_type.required_size() as usize
        {
            return Err(FieldValidationFailedException(format!(
                "{} requires {} values",
                self.get_description(),
                self.field_type.required_size()
            )));
        }
        if self.max_array_size.is_some_and(|max| values.len() > max) {
            return Err(FieldValidationFailedException(format!(
                "{} has too many values",
                self.get_description()
            )));
        }
        for value in values {
            let number: Result<f64, FieldValidationFailedException> = if matches!(
                self.field_type,
                FieldType::Integer
                    | FieldType::IntegerPair
                    | FieldType::IntegerTriple
                    | FieldType::IntegerArray
                    | FieldType::IntegerList
                    | FieldType::MatlabIntegerArray
            ) {
                value.parse::<i64>().map(|v| v as f64).map_err(|_| {
                    FieldValidationFailedException(format!("{} is invalid", self.get_description()))
                })
            } else {
                value.parse::<f64>().map_err(|_| {
                    FieldValidationFailedException(format!("{} is invalid", self.get_description()))
                })
            };
            let number = number?;
            if let Some(set) = &self.validation_set {
                if set.number_must_be_positive && number <= 0.0
                    || set.minimum.is_some_and(|minimum| number < minimum)
                    || set.maximum.is_some_and(|maximum| number > maximum)
                {
                    return Err(FieldValidationFailedException(format!(
                        "{} is invalid",
                        self.get_description()
                    )));
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for LabeledTextField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[label:{}]", self.label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_name_checkpoint_and_highlight_follow_text() {
        let mut field = LabeledTextField::new(FieldType::FloatingPoint, "Pixel size:");
        assert_eq!(field.name, "tf.pixel-size");
        field.set_text("1.25");
        field.checkpoint();
        assert!(!field.is_different_from_checkpoint(true));
        field.set_field_highlight("1.25");
        assert_eq!(field.text_foreground, FIELD_HIGHLIGHT);
        field.set_text("2");
        field.focus_lost();
        assert_eq!(field.text_foreground, BLACK);
    }
    #[test]
    fn source_validation_obeys_required_array_and_range_state() {
        let mut field =
            LabeledTextField::new_with_max_array_size(FieldType::IntegerArray, 2, "Values:");
        field.set_required(true);
        assert!(field.get_text_validated(true).is_err());
        field.set_text("1, 2, 3");
        assert!(field.get_text_validated(true).is_err());
        field.set_text("1, 2");
        field.set_minimum(2.0);
        assert!(field.get_text_validated(true).is_err());
        field.set_text("2, 3");
        assert!(field.get_text_validated(true).is_ok());
    }
    #[test]
    fn disabled_field_skips_validation_and_checkpoint_difference() {
        let mut field = LabeledTextField::new(FieldType::Integer, "Bin:");
        field.set_text("bad");
        field.set_enabled(false);
        assert!(field.get_text_validated(true).is_ok());
        assert!(!field.is_different_from_checkpoint(false));
    }
}
