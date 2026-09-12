//! `IMOD/Etomo/src/etomo/ui/swing/FieldCell.java`.
//!
//! `JTextField`, `FontMetrics`, `FocusListener`, colours, and borders belong to
//! the Swing renderer.  Their observable state is retained here so that the
//! source unit's field naming, path expansion, locking, and value semantics are
//! usable by the Rust UI without claiming to implement Swing itself.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, DOUBLE_NULL_VALUE, INTEGER_NULL_VALUE, Type,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::util::utilities;

use super::action_target::ActionTarget;
use super::field_lock_controller::{FieldLockController, JTextComponent};
use super::panel::Dimension;
use super::ui_utilities::{Color, FontMetrics};

/// Java `ParsedElementType`, at the unported parser boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParsedElementType {
    NonMatlabNumber,
    MatlabNumber,
}

/// Java `UITestFieldType.TEXT_FIELD`.
const TEXT_FIELD: &str = "TextField";
const CELL_FOREGROUND: Color = Color {
    red: 0,
    green: 0,
    blue: 0,
};
const CELL_NOT_IN_USE_FOREGROUND: Color = Color {
    red: 128,
    green: 128,
    blue: 128,
};
const CELL_DISABLED_FOREGROUND: Color = Color {
    red: 128,
    green: 128,
    blue: 128,
};

/// State held by Java `JTextField` for this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldCellTextField {
    pub text: String,
    pub name: Option<String>,
    pub tooltip: Option<String>,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub foreground: Color,
    pub disabled_text_color: Color,
    pub horizontal_alignment: i32,
    pub width: i32,
    pub border_right: i32,
    pub preferred_size: Dimension,
    pub focus_listener_count: usize,
    pub selection_start: usize,
    pub selection_end: usize,
}

impl Default for FieldCellTextField {
    fn default() -> Self {
        Self {
            text: String::new(),
            name: None,
            tooltip: None,
            enabled: true,
            editable: true,
            visible: true,
            foreground: CELL_FOREGROUND,
            disabled_text_color: CELL_FOREGROUND,
            horizontal_alignment: 0,
            width: 0,
            border_right: 0,
            preferred_size: Dimension {
                width: 0,
                height: 0,
            },
            focus_listener_count: 0,
            selection_start: 0,
            selection_end: 0,
        }
    }
}

/// Java `TextFieldState`, owned here until its source unit is separately translated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldCellTextFieldState {
    pub editable_field: bool,
    pub parsed_element_type: Option<ParsedElementType>,
    pub root_dir: Option<PathBuf>,
    pub debug: bool,
    pub expanded: bool,
    pub parent: Option<PathBuf>,
}

impl FieldCellTextFieldState {
    fn new(
        editable_field: bool,
        parsed_element_type: Option<ParsedElementType>,
        root_dir: Option<&Path>,
    ) -> Self {
        Self {
            editable_field,
            parsed_element_type,
            root_dir: root_dir.map(Path::to_path_buf),
            debug: false,
            expanded: true,
            parent: None,
        }
    }
    fn apply_expanded_to_field_text(&mut self, text: &str) -> String {
        if !self.expanded {
            let path = Path::new(text);
            self.parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .map(Path::to_path_buf);
            return path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        }
        if Path::new(text).components().count() == 1 {
            if let Some(parent) = &self.parent {
                return parent.join(text).to_string_lossy().into_owned();
            }
        }
        self.parent = None;
        text.to_owned()
    }
    fn expand_field_text(&mut self, expand: bool, text: &str) -> String {
        if self.expanded == expand {
            return text.to_owned();
        }
        self.expanded = expand;
        self.apply_expanded_to_field_text(text)
    }
    fn convert_to_field_text(&mut self, value: Option<&str>) -> String {
        let value = value.unwrap_or_default();
        if self.expanded || Path::new(value).components().count() == 1 {
            self.parent = None;
            return value.to_owned();
        }
        self.parent = Path::new(value)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf);
        Path::new(value)
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
    }
    fn convert_file_to_field_text(&mut self, file: Option<&Path>) -> String {
        let Some(file) = file else {
            return String::new();
        };
        let value = if let Some(root_dir) = &self.root_dir {
            file.strip_prefix(root_dir)
                .unwrap_or(file)
                .to_string_lossy()
                .into_owned()
        } else {
            file.to_string_lossy().into_owned()
        };
        self.convert_to_field_text(Some(&value))
    }
    fn convert_to_contracted_string(&self, text: &str) -> String {
        if !self.expanded {
            return text.to_owned();
        }
        Path::new(text)
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
    }
    fn convert_to_expanded_string(&mut self, text: &str) -> String {
        if text
            .chars()
            .all(|character| matches!(character, ' ' | '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r'))
        {
            self.parent = None;
            return String::new();
        }
        if self.expanded || self.parent.is_none() || Path::new(text).components().count() > 1 {
            return text.to_owned();
        }
        self.parent
            .as_ref()
            .unwrap()
            .join(text)
            .to_string_lossy()
            .into_owned()
    }
    fn extract_end_value(&self, text: &str) -> i32 {
        let text = text.trim();
        if text.len() <= 1 || !text.chars().skip(1).any(|character| character == '-') {
            return INTEGER_NULL_VALUE;
        }
        let Some(index) = text[1..].find('-').map(|index| index + 1) else {
            return INTEGER_NULL_VALUE;
        };
        text[index + 1..]
            .trim()
            .parse()
            .unwrap_or(INTEGER_NULL_VALUE)
    }
}

/// Java package-private final `FieldCell`.
#[derive(Clone, Debug)]
pub struct FieldCell {
    pub text_field: FieldCellTextField,
    pub parsed_element_type: Option<ParsedElementType>,
    pub field_lock_controller: FieldLockController,
    pub in_use: bool,
    pub font_metrics: Option<FontMetrics>,
    /// Java `DirectiveDef`, retained as an external storage identity.
    pub directive_def: Option<String>,
    pub unformatted_tooltip: Option<String>,
    pub state: FieldCellTextFieldState,
}

impl FieldCell {
    fn new(
        editable: bool,
        parsed_element_type: ParsedElementType,
        root_dir: Option<&Path>,
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Self {
        let mut field = Self {
            text_field: FieldCellTextField::default(),
            parsed_element_type: Some(parsed_element_type),
            field_lock_controller: FieldLockController::get_text_component_read_only_instance(
                JTextComponent {
                    enabled: true,
                    editable: true,
                },
                !editable,
            ),
            in_use: true,
            font_metrics: None,
            directive_def: None,
            unformatted_tooltip: None,
            state: FieldCellTextFieldState::new(editable, Some(parsed_element_type), root_dir),
        };
        let header = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        if let Some(header) = header.as_deref() {
            field.set_name(header);
        }
        field.set_background();
        field.set_foreground();
        field.set_font();
        field.set_expanded();
        field
    }
    fn new_from_state(state: &FieldCellTextFieldState) -> Self {
        let mut field = Self {
            text_field: FieldCellTextField::default(),
            parsed_element_type: None,
            field_lock_controller: FieldLockController::get_text_component_instance(
                JTextComponent {
                    enabled: true,
                    editable: true,
                },
            ),
            in_use: true,
            font_metrics: None,
            directive_def: None,
            unformatted_tooltip: None,
            state: state.clone(),
        };
        field.set_background();
        field.set_foreground();
        field.set_font();
        field.set_expanded();
        field
    }
    pub fn get_instance(field_cell: &FieldCell) -> Self {
        let mut instance = Self::new_from_state(&field_cell.state);
        instance.in_use = field_cell.in_use;
        instance.set_value(&field_cell.get_expanded_value());
        instance.add_listeners();
        instance.text_field.tooltip = field_cell.text_field.tooltip.clone();
        instance
    }
    pub fn get_editable_instance() -> Self {
        let mut instance = Self::new(true, ParsedElementType::NonMatlabNumber, None, None, None);
        instance.add_listeners();
        instance
    }
    pub fn get_editable_matlab_instance() -> Self {
        let mut instance = Self::new(true, ParsedElementType::MatlabNumber, None, None, None);
        instance.add_listeners();
        instance
    }
    pub fn get_ineditable_instance() -> Self {
        let mut instance = Self::new(false, ParsedElementType::NonMatlabNumber, None, None, None);
        instance.set_editable(false);
        instance.add_listeners();
        instance
    }
    pub fn get_named_ineditable_instance(header_label: &str) -> Self {
        let mut instance = Self::new(
            false,
            ParsedElementType::NonMatlabNumber,
            None,
            Some(header_label),
            None,
        );
        instance.set_editable(false);
        instance.add_listeners();
        instance
    }
    pub fn get_named_ineditable_instance_two(header_label1: &str, header_label2: &str) -> Self {
        let mut instance = Self::new(
            false,
            ParsedElementType::NonMatlabNumber,
            None,
            Some(header_label1),
            Some(header_label2),
        );
        instance.set_editable(false);
        instance.add_listeners();
        instance
    }
    pub fn get_expandable_instance(root_dir: &Path) -> Self {
        let mut instance = Self::new(
            true,
            ParsedElementType::NonMatlabNumber,
            Some(root_dir),
            None,
            None,
        );
        instance.add_listeners();
        instance
    }
    pub fn get_expandable_ineditable_instance(root_dir: &Path) -> Self {
        let mut instance = Self::new(
            false,
            ParsedElementType::NonMatlabNumber,
            Some(root_dir),
            None,
            None,
        );
        instance.set_editable(false);
        instance.add_listeners();
        instance
    }
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
    fn set_name(&mut self, reference: &str) {
        if let Some(name) = utilities::convert_label_to_name(Some(reference), true) {
            self.text_field.name = Some(format!("{TEXT_FIELD}{SEPARATOR_CHAR}{name}"));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {DEFAULT_DELIMITER} ",
                    self.text_field.name.as_deref().unwrap()
                );
            }
        }
    }
    pub fn get_name(&self) -> Option<&str> {
        self.text_field.name.as_deref()
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn is_boolean(&self) -> bool {
        false
    }
    pub fn set_debug(&mut self, input: bool) {
        self.state.debug = input;
    }
    pub fn set_root_dir(&mut self, input: &Path) {
        self.state =
            FieldCellTextFieldState::new(self.is_editable(), self.parsed_element_type, Some(input));
    }
    fn add_listeners(&mut self) {
        self.text_field.focus_listener_count += 1;
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn set_directive_def(&mut self, directive_def: Option<String>) {
        self.directive_def = directive_def;
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    pub fn get_preferred_size(&self) -> Dimension {
        self.text_field.preferred_size
    }
    pub fn get_preferred_width(&mut self) -> i32 {
        let metrics = self.font_metrics.get_or_insert_default();
        metrics.string_width(&self.text_field.text)
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.text_field.visible = visible;
    }
    pub fn set_locked(&mut self, locked: bool) {
        self.field_lock_controller.set_locked(locked);
        let text_component = self.field_lock_controller.text_component.as_ref().unwrap();
        self.text_field.enabled = text_component.enabled;
        self.text_field.editable = text_component.editable;
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.field_lock_controller.set_editable(editable);
        let text_component = self.field_lock_controller.text_component.as_ref().unwrap();
        self.text_field.enabled = text_component.enabled;
        self.text_field.editable = text_component.editable;
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.field_lock_controller.set_enabled(enabled) {
            let text_component = self.field_lock_controller.text_component.as_ref().unwrap();
            self.text_field.enabled = text_component.enabled;
            self.text_field.editable = text_component.editable;
            self.set_background();
            if self.is_enabled() && self.is_editable() {
                self.set_foreground();
            } else {
                self.text_field.foreground = CELL_DISABLED_FOREGROUND;
                self.text_field.disabled_text_color = CELL_DISABLED_FOREGROUND;
            }
        }
    }
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }
    pub fn set_in_use(&mut self, in_use: bool) {
        self.in_use = in_use;
        self.set_foreground();
    }
    pub fn is_empty(&self) -> bool {
        self.text_field.text.trim().is_empty()
    }
    /// Java `backup`; source deliberately leaves this unimplemented.
    pub fn backup(&mut self) {}
    /// Java `restoreFromBackup`; source deliberately leaves this unimplemented.
    pub fn restore_from_backup(&mut self) {}
    /// Java `checkpoint`; source deliberately leaves this unimplemented.
    pub fn checkpoint(&mut self) {}
    /// Java `getCheckpoint`; Java returns null.
    pub fn get_checkpoint(&self) -> Option<String> {
        None
    }
    /// Java `isDifferentFromCheckpoint`; Java returns false.
    pub fn is_different_from_checkpoint(&self, _always_check: bool) -> bool {
        false
    }
    /// Java `setCheckpoint`; source deliberately leaves this unimplemented.
    pub fn set_checkpoint(&mut self, _input: Option<&str>) {}
    pub fn equals_default_value(&self) -> bool {
        false
    }
    pub fn equals_default_value_string(&self, _value: &str) -> bool {
        false
    }
    /// Java `useDefaultValue`; source deliberately leaves this unimplemented.
    pub fn use_default_value(&mut self) {}
    pub fn is_field_highlight_set(&self) -> bool {
        false
    }
    /// Java `clearFieldHighlight`; source deliberately leaves this unimplemented.
    pub fn clear_field_highlight(&mut self) {}
    pub fn equals_field_highlight(&self) -> bool {
        false
    }
    pub fn equals_field_highlight_string(&self, _value: &str) -> bool {
        false
    }
    /// Java `getFieldHighlight`; Java returns null.
    pub fn get_field_highlight(&self) -> Option<String> {
        None
    }
    /// Java's boolean overload has an empty body.
    pub fn set_field_highlight(&mut self, _value: bool) {}
    /// Java's FieldSettingInterface overload is explicitly unimplemented.
    pub fn set_field_highlight_setting(&mut self, _setting: Option<&str>) {}
    /// Java's String overload is explicitly unimplemented.
    pub fn set_field_highlight_string(&mut self, _value: &str) {}
    pub fn set_target_file(&mut self, file: Option<&Path>) {
        self.set_file(file);
    }
    pub fn set_file(&mut self, file: Option<&Path>) {
        self.set_value_file(file);
    }
    pub fn set_value_file(&mut self, file: Option<&Path>) {
        self.text_field.text = self.state.convert_file_to_field_text(file);
    }
    pub fn set_value(&mut self, value: &str) {
        self.text_field.text = self.state.convert_to_field_text(Some(value));
    }
    pub fn set_value_allow_empty(&mut self, value: Option<&str>, allow_empty: bool) {
        if allow_empty || value.is_some_and(|value| !value.is_empty()) {
            self.set_value(value.unwrap_or_default());
        }
    }
    pub fn set_value_option(&mut self, value: Option<&str>) {
        if let Some(value) = value {
            self.set_value(value);
        } else {
            self.clear();
        }
    }
    /// Java's `setValue(boolean)` overload has an empty body.
    pub fn set_boolean_value(&mut self, _value: bool) {}
    pub fn get_contracted_value(&self) -> String {
        self.state
            .convert_to_contracted_string(&self.text_field.text)
    }
    pub fn get_expanded_value(&self) -> String {
        self.state
            .clone()
            .convert_to_expanded_string(&self.text_field.text)
    }
    pub fn get_file(&self) -> Option<PathBuf> {
        let value = self.get_expanded_value();
        (!value.is_empty()).then(|| PathBuf::from(value))
    }
    pub fn expand(&mut self, expand: bool) {
        self.text_field.text = self.state.expand_field_text(expand, &self.text_field.text);
    }
    pub fn set_horizontal_alignment(&mut self, _alignment: i32) {
        self.text_field.horizontal_alignment = 0;
    }
    pub fn set_expanded(&mut self) {
        self.text_field.text = self
            .state
            .apply_expanded_to_field_text(&self.text_field.text);
    }
    pub fn set_range_value(&mut self, start: i32, end: i32) {
        self.set_value(&format!("{start} - {end}"));
    }
    pub fn reset_value(&mut self) {
        self.state.parent = None;
        self.text_field.text.clear();
    }
    pub fn clear(&mut self) {
        self.reset_value();
    }
    pub fn set_int_value(&mut self, value: i32) {
        self.set_value(&value.to_string());
    }
    pub fn set_double_value(&mut self, value: f64) {
        self.set_value(&value.to_string());
    }
    pub fn set_long_value(&mut self, value: i64) {
        self.set_value(&value.to_string());
    }
    pub fn set_etomo_number_value(&mut self, value: &ConstEtomoNumber) {
        self.set_value(&value.to_string());
    }
    pub fn get_end_value(&self) -> i32 {
        self.state.extract_end_value(&self.text_field.text)
    }
    pub fn get_field_type(&self) -> &'static str {
        TEXT_FIELD
    }
    /// Java `getDescription`; source returns the empty string.
    pub fn get_description(&self) -> &'static str {
        ""
    }
    /// Java `getQuotedLabel`; a FieldCell cannot have a label.
    pub fn get_quoted_label(&self) -> Option<String> {
        None
    }
    pub fn get_value(&self) -> &str {
        &self.text_field.text
    }
    pub fn is_required(&self) -> bool {
        false
    }
    /// Java validation is unavailable in this class, so both overloads return the widget text.
    pub fn get_text(&self, _do_validation: bool) -> &str {
        &self.text_field.text
    }
    pub fn get_text_with_displayers(
        &self,
        _do_validation: bool,
        _field_displayer1: Option<&str>,
        _field_displayer2: Option<&str>,
    ) -> &str {
        &self.text_field.text
    }
    pub fn is_selected(&self) -> bool {
        false
    }
    pub fn get_int_value(&self) -> i32 {
        self.text_field.text.parse().unwrap_or(INTEGER_NULL_VALUE)
    }
    pub fn get_double_value(&self) -> f64 {
        self.text_field.text.parse().unwrap_or(DOUBLE_NULL_VALUE)
    }
    pub fn get_etomo_number(&self) -> ConstEtomoNumber {
        let mut number = EtomoNumber::new();
        number.set_string(Some(&self.text_field.text));
        number.base
    }
    pub fn get_etomo_number_with_type(&self, r#type: Type) -> ConstEtomoNumber {
        let mut number = EtomoNumber::new_with_type(Some(r#type));
        number.set_string(Some(&self.text_field.text));
        number.base
    }
    pub fn get_width(&self) -> i32 {
        self.text_field.width
    }
    pub fn get_right_border(&self) -> i32 {
        self.text_field.border_right
    }
    pub fn set_tooltip_text(&mut self, text: impl Into<String>) {
        self.text_field.tooltip = Some(text.into());
    }
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        if let Some(tooltip) = tooltip {
            self.text_field.tooltip = Some(tooltip.to_owned());
        }
    }
    pub fn set_unformatted_tooltip(&mut self, text: impl Into<String>) -> String {
        let text = text.into();
        self.unformatted_tooltip = Some(text.clone());
        text
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.is_some()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        let mut text = self.unformatted_tooltip.take().unwrap_or_default();
        for description in [param_descr, directive_descr].into_iter().flatten() {
            if !description.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(description);
            }
        }
        self.set_tooltip_text(text);
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.text_field.tooltip.as_deref()
    }
    pub fn equals(&self, comp: &str) -> bool {
        self.get_value() == comp
    }
    pub fn focus_gained(&mut self) {
        self.text_field.selection_start = 0;
        self.text_field.selection_end = self.text_field.text.len();
    }
    pub fn focus_lost(&mut self) {
        self.text_field.selection_start = 0;
        self.text_field.selection_end = 0;
    }
    fn set_background(&mut self) {}
    fn set_font(&mut self) {}
    fn set_foreground(&mut self) {
        if self.in_use {
            self.text_field.foreground = CELL_FOREGROUND;
            self.text_field.disabled_text_color = CELL_FOREGROUND;
        } else {
            self.text_field.foreground = CELL_NOT_IN_USE_FOREGROUND;
            self.text_field.disabled_text_color = CELL_NOT_IN_USE_FOREGROUND;
        }
    }
}

impl std::fmt::Display for FieldCell {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.text_field.text)
    }
}

/// Java `FieldCell implements ActionTarget`.
impl ActionTarget for FieldCell {
    fn set_target_file(&mut self, file: Option<&Path>) {
        FieldCell::set_target_file(self, file);
    }

    fn get_expanded_value(&self) -> String {
        FieldCell::get_expanded_value(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contracted_path_restores_parent_on_expand() {
        let mut field = FieldCell::get_editable_instance();
        field.set_value("/tmp/example.mrc");
        field.expand(false);
        assert_eq!(field.get_value(), "example.mrc");
        assert_eq!(field.get_expanded_value(), "/tmp/example.mrc");
        field.expand(true);
        assert_eq!(field.get_value(), "/tmp/example.mrc");
    }
    #[test]
    fn lock_keeps_field_enabled_but_not_editable() {
        let mut field = FieldCell::get_editable_instance();
        field.set_locked(true);
        assert!(field.text_field.enabled);
        assert!(!field.text_field.editable);
        assert!(
            field
                .field_lock_controller
                .text_component
                .as_ref()
                .unwrap()
                .enabled
        );
        assert!(
            !field
                .field_lock_controller
                .text_component
                .as_ref()
                .unwrap()
                .editable
        );
        field.set_enabled(false);
        assert!(!field.text_field.enabled);
    }
}
