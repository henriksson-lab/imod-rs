//! `IMOD/Etomo/src/etomo/ui/swing/TextEfield.java`.
//!
//! The native Swing `JTextField`, `JLabel`, `JPanel`, focus listeners, and
//! GridBag layout are deliberately represented as state at the GUI boundary.
//! This retains the source-visible field, label, control, naming, validation,
//! flag, appearance, and file-path behaviour without pretending that Rust owns
//! a Swing widget.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

/// Java `ControlState`; its concrete source unit remains an explicit boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlState {
    Enabled,
    Disabled,
}

/// Java `FlagType`, represented by its `isTemplate()` property.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlagType {
    Template,
    Errors,
}
impl FlagType {
    pub fn is_template(self) -> bool {
        self == Self::Template
    }
}

/// Source-visible `JTextField` state at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextField {
    pub text: String,
    pub name: Option<String>,
    pub columns: i32,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub preferred_width: Option<i32>,
    pub tooltip: Option<String>,
    pub focus_listener_count: usize,
}
impl Default for TextField {
    fn default() -> Self {
        Self {
            text: String::new(),
            name: None,
            columns: 0,
            enabled: true,
            editable: true,
            visible: true,
            preferred_width: None,
            tooltip: None,
            focus_listener_count: 0,
        }
    }
}

/// Java package-private `ValidationExtension` state used by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ValidationExtension {
    pub required: bool,
    pub location_descr: Option<String>,
    pub file_must_exist: bool,
    pub file_only: bool,
    pub must_be_positive: bool,
}
impl ValidationExtension {
    fn get_location_addon(&self) -> String {
        self.location_descr
            .as_ref()
            .map_or_else(String::new, |value| format!(" in {value}"))
    }
}

/// Java package-private `TextEfield`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextEfield {
    /// Java `textField`.
    pub text_field: TextField,
    pub field_type: Option<FieldType>,
    pub label_text: String,
    pub use_label: bool,
    pub label_enabled: bool,
    pub label_tooltip: Option<String>,
    pub use_control_component: bool,
    pub control_override: bool,
    pub component_visible: bool,
    pub debug: bool,
    pub enable_control_state: Option<ControlState>,
    pub control_listener_count: usize,
    pub validation_extension: Option<ValidationExtension>,
    pub backup: Option<String>,
    pub checkpoint: Option<String>,
    pub template_value: Option<String>,
    pub flag_errors: bool,
    /// Java nullable `flagExtension` presence.
    pub flag_extension_created: bool,
    pub default_to_filename: bool,
    pub limit_displayed_file_path: i32,
    pub full_file_path: Option<String>,
    pub prevent_blank_template: Option<String>,
    pub directive_def: Option<String>,
    pub in_grid_bag: bool,
}

impl TextEfield {
    /// Java package-private `TextEfield(String, FieldType, boolean, boolean, boolean,
    /// boolean, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        label_text: impl Into<String>,
        field_type: Option<FieldType>,
        use_label: bool,
        use_control_component: bool,
        enabled_field: bool,
        editable_component: bool,
        use_grid_bag: bool,
        default_to_file_name: bool,
    ) -> Self {
        let label_text = label_text.into();
        let mut field = Self {
            text_field: TextField {
                enabled: enabled_field,
                editable: editable_component,
                ..Default::default()
            },
            field_type,
            label_text,
            use_label,
            label_enabled: enabled_field,
            label_tooltip: None,
            use_control_component,
            control_override: false,
            component_visible: true,
            debug: false,
            enable_control_state: None,
            control_listener_count: 0,
            validation_extension: None,
            backup: None,
            checkpoint: None,
            template_value: None,
            flag_errors: false,
            flag_extension_created: false,
            default_to_filename: default_to_file_name,
            limit_displayed_file_path: 0,
            full_file_path: None,
            prevent_blank_template: None,
            directive_def: None,
            in_grid_bag: use_grid_bag,
        };
        field.set_name();
        field
    }

    /// Java `getInstance`.
    pub fn get_instance(label_text: impl Into<String>, field_type: FieldType) -> Self {
        Self::new(
            label_text,
            Some(field_type),
            false,
            false,
            true,
            true,
            false,
            false,
        )
    }
    /// Java `getLabeledInstance`.
    pub fn get_labeled_instance(label_text: impl Into<String>, field_type: FieldType) -> Self {
        Self::new(
            label_text,
            Some(field_type),
            true,
            false,
            true,
            true,
            false,
            false,
        )
    }
    /// Java `getOverrideInstance`; `DirectiveValueType.getInstance` is outside this unit.
    pub fn get_override_instance(
        label_text: impl Into<String>,
        field_type: Option<FieldType>,
    ) -> Self {
        Self::new(
            label_text, field_type, false, true, true, true, false, false,
        )
    }
    /// Java `getDisabledInstance`.
    pub fn get_disabled_instance(label_text: impl Into<String>, field_type: FieldType) -> Self {
        Self::new(
            label_text,
            Some(field_type),
            false,
            false,
            false,
            true,
            false,
            false,
        )
    }

    /// Java private `setName`.
    pub fn set_name(&mut self) {
        if let Some(name) = utilities::convert_label_to_name(Some(&self.label_text), false) {
            self.text_field.name = Some(format!("tf{SEPARATOR_CHAR}{name}"));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {DEFAULT_DELIMITER} ",
                    self.text_field.name.as_deref().unwrap()
                );
            }
        }
    }
    pub fn get_label(&self) -> &str {
        &self.label_text
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    /// Java `getComponent`; the boolean represents the source container's visibility.
    pub fn get_component(&self) -> bool {
        self.component_visible
    }
    pub fn set_columns(&mut self) {
        if let Some(field_type) = self.field_type {
            self.text_field.columns = field_type.get_columns();
        }
    }
    pub fn set_preferred_width(&mut self, new_width: i32) {
        self.text_field.preferred_width = Some(new_width);
    }
    pub fn get_preferred_width(&self) -> Option<i32> {
        self.text_field.preferred_width
    }
    pub fn get_file(&self) -> Option<PathBuf> {
        let text = self.get_text();
        (!text.is_empty()).then(|| PathBuf::from(text))
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> String {
        if self.is_override() {
            return String::new();
        }
        if self.limit_displayed_file_path > 0 {
            self.full_file_path
                .clone()
                .unwrap_or_else(|| self.text_field.text.clone())
        } else {
            self.text_field.text.clone()
        }
    }
    pub fn is_empty(&self) -> bool {
        self.get_text().trim().is_empty()
    }
    pub fn add_focus_listener(&mut self) {
        self.text_field.focus_listener_count += 1;
    }
    pub fn remove_focus_listener(&mut self) {
        self.text_field.focus_listener_count =
            self.text_field.focus_listener_count.saturating_sub(1);
    }
    /// Java `equals(String)`.
    pub fn equals(&self, compare_text: Option<&str>) -> bool {
        match (Some(self.get_text()), compare_text) {
            (Some(text), Some(compare)) => {
                text.is_empty() && compare.is_empty() || text.trim() == compare.trim()
            }
            _ => false,
        }
    }
    pub fn set_tooltip(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.text_field.tooltip = Some(text.clone());
        if self.use_label {
            self.label_tooltip = Some(text);
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.component_visible = visible;
    }
    pub fn is_visible(&self) -> bool {
        self.component_visible
    }
    pub fn set_file(&mut self, file_path: Option<&str>) {
        if file_path.is_some_and(|value| !value.trim().is_empty()) {
            self.set_text_file(Path::new(file_path.unwrap()));
        } else {
            self.clear();
        }
    }
    /// Java overloaded `setText(File)`.
    pub fn set_text_file(&mut self, file: &Path) {
        let text = if file.is_absolute() {
            file.to_string_lossy().into_owned()
        } else {
            std::env::current_dir()
                .map(|directory| directory.join(file).to_string_lossy().into_owned())
                .unwrap_or_else(|_| {
                    file.file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default()
                })
        };
        self.set_text(text);
    }
    /// Java overloaded `setText(int)`.
    pub fn set_text_int(&mut self, text: i32) {
        self.set_text(text.to_string());
    }
    /// Java overloaded `setText(String)`.
    pub fn set_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        // `ValueManipulationExtension.createDisplayedFilePath(String, FieldType)`:
        // it acts only on FILE fields and truncates at a system file separator.
        let displayed = if self.field_type != Some(FieldType::File)
            || self.limit_displayed_file_path <= 0
            || text.len() <= self.limit_displayed_file_path as usize
            || !text.contains(std::path::MAIN_SEPARATOR)
        {
            self.full_file_path = None;
            text
        } else {
            let is_local = self.default_to_filename
                && Path::new(&text)
                    .parent()
                    .and_then(|path| path.canonicalize().ok())
                    == std::env::current_dir().ok();
            self.full_file_path = Some(text.clone());
            if is_local {
                Path::new(&text)
                    .file_name()
                    .map_or_else(|| text.clone(), |name| name.to_string_lossy().into_owned())
            } else {
                let start = text
                    .len()
                    .saturating_sub((self.limit_displayed_file_path - 3).max(0) as usize);
                match text[start..]
                    .find(std::path::MAIN_SEPARATOR)
                    .map(|index| start + index)
                    .or_else(|| text.rfind(std::path::MAIN_SEPARATOR))
                {
                    Some(index) => format!("...{}", &text[index..]),
                    None => text,
                }
            }
        };
        self.set_text_internal(Some(displayed));
    }
    /// Java private `setTextInternal`.
    pub fn set_text_internal(&mut self, text: Option<String>) {
        match text {
            None => self.clear(),
            Some(text) if text.is_empty() => self.clear(),
            Some(text) => {
                self.text_field.text = text;
                self.update_flag_extension();
            }
        }
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    pub fn set_directive_def(&mut self, directive_def: Option<String>) {
        self.directive_def = directive_def;
    }
    pub fn is_template_value(&self) -> bool {
        self.template_value.is_some()
    }

    pub fn is_override(&self) -> bool {
        self.use_control_component && self.control_override
    }
    pub fn set_component_control(&mut self, control: bool, _control_state: Option<ControlState>) {
        if self.use_control_component {
            self.control_override = control;
            self.text_field.visible = !control;
        }
    }
    pub fn add_control_listener(&mut self) {
        self.control_listener_count += 1;
    }
    /// Java `sendControlEvent`; listener invocation is a native Swing boundary.
    pub fn send_control_event(&self) -> usize {
        self.control_listener_count
    }
    pub fn set_enable_control(&mut self, control: bool, control_state: Option<ControlState>) {
        self.enable_control_state = control.then_some(control_state).flatten();
    }

    /// Java `clear`.
    pub fn clear(&mut self) {
        self.text_field.text.clear();
        self.full_file_path = None;
        if let Some(template) = self.prevent_blank_template.clone() {
            self.set_text(template);
        }
        self.update_flag_extension();
    }
    pub fn set_substitute_text(&mut self, text: impl Into<String>) {
        self.text_field.text = text.into();
    }
    pub fn add_value_manipulation_listener(&mut self) {
        self.add_focus_listener();
    }
    pub fn set_required(&mut self, required: bool) {
        self.validation_extension.get_or_insert_default().required = required;
    }
    pub fn set_location_descr(&mut self, location_descr: Option<String>) {
        self.validation_extension
            .get_or_insert_default()
            .location_descr = location_descr;
    }
    pub fn set_file_must_exist(&mut self, file_must_exist: bool) {
        self.validation_extension
            .get_or_insert_default()
            .file_must_exist = file_must_exist;
    }
    pub fn set_file_only(&mut self, file_only: bool) {
        self.validation_extension.get_or_insert_default().file_only = file_only;
    }
    pub fn set_must_be_positive(&mut self, must_be_positive: bool) {
        self.validation_extension
            .get_or_insert_default()
            .must_be_positive = must_be_positive;
    }

    /// Java overloaded validation `getText`; `FieldValidator` is a separate source unit.
    pub fn get_text_validated(&self, do_validation: bool) -> Result<String, String> {
        let text = self.get_text();
        if !do_validation || self.is_override() || !self.is_enabled() {
            return Ok(text);
        }
        let extension = self.validation_extension.as_ref();
        let prefix = format!(
            "\"{}\"{}",
            self.label_text,
            extension.map_or_else(String::new, ValidationExtension::get_location_addon)
        );
        if extension.is_some_and(|value| value.required) && text.trim().is_empty() {
            return Err(format!("{prefix} is required"));
        }
        if extension.is_some_and(|value| value.file_must_exist)
            && !text.trim().is_empty()
            && !Path::new(&text).exists()
        {
            return Err(format!("{prefix} does not exist"));
        }
        if extension.is_some_and(|value| value.file_only)
            && !text.trim().is_empty()
            && Path::new(&text).is_dir()
        {
            return Err(format!("{prefix} must be a file"));
        }
        if extension.is_some_and(|value| value.must_be_positive)
            && text.trim().parse::<f64>().map_or(true, |value| value <= 0.)
        {
            return Err(format!("{prefix} must be positive"));
        }
        Ok(text)
    }
    pub fn is_valid(&self) -> bool {
        self.get_text_validated(true).is_ok()
    }

    /// Java `createAppearanceExtension`; appearance is state carried by `TextField`.
    pub fn create_appearance_extension(&mut self, enabled_field: bool, editable_component: bool) {
        self.text_field.enabled = enabled_field;
        self.text_field.editable = editable_component;
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.text_field.enabled = enabled;
        self.label_enabled = enabled;
    }
    pub fn is_enabled(&self) -> bool {
        self.text_field.enabled
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.text_field.editable = editable;
    }
    pub fn is_editable(&self) -> bool {
        self.text_field.editable
    }
    pub fn set_limit_displayed_file_path(&mut self, max_file_path_size: i32) {
        let text = self.get_text();
        self.limit_displayed_file_path = max_file_path_size;
        self.set_text(text);
    }

    pub fn update_flag_extension(&mut self) {}
    pub fn create_flag_extension(&mut self) -> bool {
        if !self.flag_extension_created {
            self.flag_extension_created = true;
            self.create_appearance_extension(true, true);
            return true;
        }
        false
    }
    pub fn get_flag_type(&self) -> Option<FlagType> {
        self.template_value
            .as_ref()
            .map(|_| FlagType::Template)
            .or(self.flag_errors.then_some(FlagType::Errors))
    }
    pub fn add_flag_origin_listener(&mut self) {
        self.add_focus_listener();
    }
    pub fn flag_template(&mut self, template_value: impl Into<String>) {
        self.template_value = Some(template_value.into());
        self.prevent_blank_template = self.template_value.clone();
    }
    pub fn set_flag_errors(&mut self) {
        self.flag_errors = true;
    }
    pub fn clear_template_value(&mut self) {
        self.template_value = None;
        self.flag_errors = false;
        self.prevent_blank_template = None;
    }
    pub fn set_template_value(&mut self) {
        if let Some(value) = self.template_value.clone() {
            self.set_text(value);
        }
    }
    /// Java `addFlagDisplay` / `addFinalFlagDisplay`; displays are Swing boundaries.
    pub fn add_flag_display(&mut self) {
        let _ = self.create_flag_extension();
    }
    pub fn add_final_flag_display(&mut self) {
        let _ = self.create_flag_extension();
    }
    pub fn set_field_highlight(&mut self, value: impl Into<String>) {
        self.flag_template(value);
    }

    pub fn backup(&mut self) {
        self.backup = Some(self.get_text());
    }
    pub fn checkpoint(&mut self) {
        self.checkpoint = Some(self.get_text());
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        always_check
            && self
                .checkpoint
                .as_deref()
                .is_some_and(|value| value != self.get_text())
    }
    pub fn restore_from_backup(&mut self) {
        if let Some(value) = self.backup.clone() {
            self.set_text(value);
        }
    }
    pub fn remove(&mut self) {
        self.in_grid_bag = false;
    }
    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)` at the direct Swing boundary.
    pub fn add(&mut self) {
        self.in_grid_bag = true;
    }
    /// Java source TODO `setText(File[])`.
    pub fn set_text_files(&mut self, _files: &[PathBuf]) {}
    /// Java source TODO `isLocalDir(String)`.
    pub fn is_local_dir(&self, _current_directory: &str) -> bool {
        false
    }
}
impl std::fmt::Display for TextEfield {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_uitest_name_and_columns_are_retained() {
        let mut field = TextEfield::get_labeled_instance("Input file:", FieldType::File);
        field.set_columns();
        assert_eq!(field.text_field.name.as_deref(), Some("tf.input-file"));
        assert_eq!(field.text_field.columns, 15);
    }
    #[test]
    fn override_hides_value_and_component_control_hides_text_widget() {
        let mut field = TextEfield::get_override_instance("Override", Some(FieldType::String));
        field.set_text("ordinary");
        field.set_component_control(true, Some(ControlState::Enabled));
        assert_eq!(field.get_text(), "");
        assert!(!field.text_field.visible);
    }
    #[test]
    fn file_display_limit_retains_full_path() {
        let mut field = TextEfield::get_instance("File", FieldType::File);
        field.set_limit_displayed_file_path(4);
        field.set_text("/one/two/file.mrc");
        assert_eq!(field.get_text(), "/one/two/file.mrc");
        assert_eq!(field.text_field.text, ".../file.mrc");
    }
}
