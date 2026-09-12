//! `IMOD/Etomo/src/etomo/ui/swing/FileTextField2.java`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::file_chooser::{FileChooser, FileChooserReturnValue, FileChooserSelectionMode};
use super::file_text_field_interface::{FileFilter, FileTextFieldInterface};
use super::panel::Dimension;
use crate::imod::etomo::base_manager::BaseManager;

/// Java `ResultListener` callback at the native event boundary.
pub trait ResultListener {
    fn process_result(&mut self, source: &FileTextField2, cancelled: bool);
}

/// Java public final `FileTextField2`.  Swing component objects are recorded as
/// source-observable layout/event state; filesystem paths keep Java nullability.
pub struct FileTextField2 {
    pub text: String,
    pub label: String,
    pub labeled: bool,
    pub peet: bool,
    pub alternate_layout: bool,
    pub manager_property_user_dir: Option<PathBuf>,
    pub axis_id_present: bool,
    pub component_order: Vec<String>,
    pub button_enabled: bool,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub background: Option<String>,
    pub adjusted_field_width: i32,
    pub columns: Option<i32>,
    pub file_selection_mode: Option<FileChooserSelectionMode>,
    pub file_filter: Option<Rc<dyn FileFilter>>,
    pub absolute_path: bool,
    pub use_text_as_file_chooser_dir: bool,
    pub turn_off_file_hiding: bool,
    pub origin_reference: Option<PathBuf>,
    pub origin: Option<PathBuf>,
    pub origin_etomo_run_dir: bool,
    pub browsing_directory: Option<PathBuf>,
    pub text_entry_policy: bool,
    pub tooltip: Option<String>,
    pub unformatted_tooltip: Option<String>,
    pub checkpoint: Option<String>,
    pub backup: Option<String>,
    pub required: bool,
    pub debug: bool,
    pub result_listener_count: usize,
    pub focus_listener_count: usize,
    pub directive_def: Option<String>,
    pub default_value: Option<String>,
    pub field_highlight: Option<String>,
}
impl Default for FileTextField2 {
    fn default() -> Self {
        Self::new(None, false, "", false, false, false)
    }
}
impl FileTextField2 {
    /// Java private `FileTextField2(BaseManager, AxisID, String, boolean, boolean, boolean)`.
    pub fn new(
        manager: Option<&dyn BaseManager>,
        axis_id_present: bool,
        label: &str,
        labeled: bool,
        peet: bool,
        alternate_layout: bool,
    ) -> Self {
        let manager_property_user_dir = manager
            .and_then(BaseManager::get_property_user_dir)
            .map(PathBuf::from);
        let mut value = Self {
            text: String::new(),
            label: label.into(),
            labeled,
            peet,
            alternate_layout,
            manager_property_user_dir,
            axis_id_present,
            component_order: Vec::new(),
            button_enabled: true,
            enabled: true,
            editable: true,
            visible: true,
            background: None,
            adjusted_field_width: 250,
            columns: None,
            file_selection_mode: None,
            file_filter: None,
            absolute_path: false,
            use_text_as_file_chooser_dir: false,
            turn_off_file_hiding: false,
            origin_reference: None,
            origin: None,
            origin_etomo_run_dir: false,
            browsing_directory: None,
            text_entry_policy: true,
            tooltip: None,
            unformatted_tooltip: None,
            checkpoint: None,
            backup: None,
            required: false,
            debug: false,
            result_listener_count: 0,
            focus_listener_count: 0,
            directive_def: None,
            default_value: None,
            field_highlight: None,
        };
        value.create_panel();
        value.add_listeners();
        value
    }
    pub fn get_unlabeled_peet_instance(manager: &dyn BaseManager, name: &str) -> Self {
        Self::new(Some(manager), false, name, false, true, false)
    }
    pub fn get_unlabeled_alt_layout_instance(manager: &dyn BaseManager, name: &str) -> Self {
        Self::new(Some(manager), false, name, false, false, true)
    }
    pub fn get_unlabeled_instance(manager: &dyn BaseManager, name: &str) -> Self {
        Self::new(Some(manager), false, name, false, false, false)
    }
    pub fn get_peet_instance(manager: &dyn BaseManager, axis_id_present: bool, name: &str) -> Self {
        Self::new(Some(manager), axis_id_present, name, true, true, false)
    }
    pub fn get_instance(manager: &dyn BaseManager, name: &str) -> Self {
        Self::new(Some(manager), false, name, true, false, false)
    }
    pub fn get_alt_layout_instance(manager: &dyn BaseManager, name: &str) -> Self {
        Self::new(Some(manager), false, name, true, false, true)
    }
    pub fn get_name(&self) -> &str {
        &self.label
    }
    pub fn create_panel(&mut self) {
        self.component_order.clear();
        if self.labeled {
            self.component_order.push("label".into());
        }
        self.component_order.push("field".into());
        self.component_order.push("button".into());
        if self.alternate_layout {
            self.component_order.push("horizontal-glue".into());
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn set_background(&mut self, color: Option<&str>) {
        self.background = color.map(str::to_owned);
    }
    pub fn get_preferred_width(&self) -> i32 {
        self.label.len() as i32 + self.adjusted_field_width + 22
    }
    pub fn add_listeners(&mut self) {
        self.result_listener_count = self.result_listener_count.max(1);
    }
    pub fn add_result_listener(&mut self, listener_present: bool) {
        if listener_present {
            self.result_listener_count += 1;
        }
    }
    pub fn add_focus_listener(&mut self) {
        self.focus_listener_count += 1;
    }
    pub fn is_text(&self) -> bool {
        true
    }
    pub fn is_boolean(&self) -> bool {
        false
    }
    pub fn is_debug(&self) -> bool {
        self.debug
    }
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }
    pub fn get_quoted_label(&self) -> String {
        format!("'{}'", self.label.split(':').next().unwrap_or_default())
    }
    /// Java `actionPerformed(ActionEvent)`: `selected_file` models native chooser selection.
    pub fn action_performed(&mut self, selected_file: Option<&Path>) -> FileChooserReturnValue {
        let mut chooser =
            FileChooser::new_with_browsing_file(self.get_file_chooser_location().as_deref());
        chooser.set_file_selection_mode(
            self.file_selection_mode
                .unwrap_or(FileChooserSelectionMode::FilesAndDirectories),
        );
        chooser.set_file_filter(self.file_filter.clone());
        let result = chooser.show_open_dialog(selected_file);
        if result == FileChooserReturnValue::ApproveOption {
            self.set_file(selected_file);
            if let Some(file) = selected_file {
                self.browsing_directory = file.parent().map(Path::to_path_buf);
            }
        }
        result
    }
    pub fn set_adjusted_field_width(&mut self, width: f64) {
        self.adjusted_field_width = width.round() as i32;
    }
    pub fn set_absolute_path(&mut self, input: bool) {
        self.absolute_path = input;
    }
    pub fn set_origin_etomo_run_dir(&mut self, input: bool) {
        self.origin_etomo_run_dir = input;
    }
    pub fn set_columns(&mut self, columns: i32) {
        self.columns = Some(columns);
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    pub fn set_preferred_width(&mut self, width: f64) {
        self.adjusted_field_width = width.round() as i32;
    }
    pub fn set_origin(&mut self, input: Option<&Path>) {
        self.origin = input.map(Path::to_path_buf);
    }
    pub fn set_origin_reference(&mut self, input: Option<&FileTextField2>) {
        self.origin_reference = input.and_then(FileTextField2::get_file);
    }
    pub fn set_use_text_as_file_chooser_dir(&mut self, input: bool) {
        self.use_text_as_file_chooser_dir = input;
    }
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
    pub fn exists(&self) -> bool {
        self.get_file().is_some_and(|file| file.exists())
    }
    pub fn get_file_validated(&self, _do_validation: bool) -> Option<PathBuf> {
        self.get_file()
    }
    pub fn equals(&self, input: Option<&FileTextField2>) -> bool {
        input.is_some_and(|input| self.get_file() == input.get_file())
    }
    pub fn checkpoint(&mut self) {
        self.checkpoint = Some(self.text.clone());
    }
    pub fn set_checkpoint(&mut self, input: Option<&str>) {
        self.checkpoint = input.map(str::to_owned);
    }
    pub fn get_checkpoint(&self) -> Option<&str> {
        self.checkpoint.as_deref()
    }
    pub fn backup(&mut self) {
        self.backup = Some(self.text.clone());
    }
    pub fn restore_from_backup(&mut self) {
        if let Some(value) = self.backup.take() {
            self.text = value;
        }
    }
    /// Java `getDirectiveDef`.
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    /// Java `setDirectiveDef(DirectiveDef)`; the untranslated autodoc object is
    /// represented by its source-visible definition string.
    pub fn set_directive_def(&mut self, directive_def: Option<&str>) {
        self.directive_def = directive_def.map(str::to_owned);
        self.default_value = self.directive_def.clone();
    }
    /// Java `useDefaultValue`.
    pub fn use_default_value(&mut self) {
        if let Some(default_value) = self.default_value.clone() {
            self.text = default_value;
        }
    }
    /// Java `equalsDefaultValue()`.
    pub fn equals_default_value(&self) -> bool {
        self.default_value
            .as_deref()
            .is_some_and(|value| value == self.text)
    }
    /// Java `equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, value: &str) -> bool {
        self.default_value
            .as_deref()
            .is_some_and(|default| default == value)
    }
    /// Java `isFieldHighlightSet`.
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight.is_some()
    }
    /// Java `getFieldHighlight`.
    pub fn get_field_highlight(&self) -> Option<&str> {
        self.field_highlight.as_deref()
    }
    /// Java `setFieldHighlight(String)`.
    pub fn set_field_highlight(&mut self, value: Option<&str>) {
        self.field_highlight = value.map(str::to_owned);
    }
    /// Java overloaded `setFieldHighlight(boolean)`, deliberately empty.
    pub fn set_field_highlight_boolean(&mut self, _value: bool) {}
    /// Java overloaded `setFieldHighlight(FieldSettingInterface)`.
    pub fn set_field_highlight_setting(&mut self, value: Option<&str>) {
        self.field_highlight = value.map(str::to_owned);
    }
    /// Java `equalsFieldHighlight()`.
    pub fn equals_field_highlight(&self) -> bool {
        self.field_highlight
            .as_deref()
            .is_some_and(|value| value == self.text)
    }
    /// Java `equalsFieldHighlight(String)`.
    pub fn equals_field_highlight_string(&self, value: &str) -> bool {
        self.field_highlight
            .as_deref()
            .is_some_and(|highlight| highlight == value)
    }
    /// Java `clearFieldHighlight`.
    pub fn clear_field_highlight(&mut self) {
        self.field_highlight = None;
    }
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        (always_check || (self.enabled && self.visible))
            && self
                .checkpoint
                .as_deref()
                .is_some_and(|value| value != self.text)
    }
    pub fn set_browsing_directory(&mut self, browsing_dir: Option<&Path>) {
        self.browsing_directory = browsing_dir.map(Path::to_path_buf);
    }
    pub fn get_origin_dir(&self) -> Option<PathBuf> {
        self.get_origin_for_file_chooser()
            .or_else(|| self.manager_property_user_dir.clone())
    }
    pub fn get_origin_for_file_chooser(&self) -> Option<PathBuf> {
        if let Some(reference) = self
            .origin_reference
            .as_ref()
            .filter(|p| !p.as_os_str().is_empty())
        {
            return Some(if reference.is_dir() {
                reference.clone()
            } else {
                reference.parent().unwrap_or(reference).to_path_buf()
            });
        }
        if let Some(origin) = self.origin.as_ref().filter(|p| p.is_dir()) {
            return Some(origin.clone());
        }
        if self.origin_etomo_run_dir {
            return std::env::current_dir().ok();
        }
        None
    }
    pub fn get_file_chooser_location(&self) -> Option<PathBuf> {
        if self.use_text_as_file_chooser_dir {
            let file = self.get_file()?;
            if file.is_dir() {
                return Some(file);
            }
            if let Some(parent) = file.parent() {
                return Some(parent.to_path_buf());
            }
        }
        self.get_origin_for_file_chooser()
    }
    pub fn set_file_selection_mode(&mut self, input: FileChooserSelectionMode) {
        self.file_selection_mode = Some(input);
    }
    pub fn set_turn_off_file_hiding(&mut self, input: bool) {
        self.turn_off_file_hiding = input;
    }
    pub fn set_file_filter(&mut self, input: Option<Rc<dyn FileFilter>>) {
        self.file_filter = input;
    }
    pub fn is_required(&self) -> bool {
        self.required && self.enabled
    }
    pub fn get_text_validated(&self, do_validation: bool) -> Result<String, String> {
        if do_validation && self.is_required() && self.is_empty() {
            Err(format!("{} is required", self.get_quoted_label()))
        } else {
            Ok(self.text.clone())
        }
    }
    pub fn get_description(&self) -> String {
        self.get_quoted_label()
    }
    pub fn get_text(&self) -> &str {
        &self.text
    }
    pub fn set_text(&mut self, text: Option<&str>) {
        self.text = text.unwrap_or_default().to_owned();
    }
    pub fn set_text_allow_empty(&mut self, text: Option<&str>, allow_empty: bool) {
        if allow_empty || text.is_some_and(|text| !text.is_empty()) {
            self.set_text(text);
        }
    }
    pub fn clear(&mut self) {
        self.text.clear();
    }
    pub fn set_value(&mut self, input: Option<&str>) {
        self.set_text(input);
    }
    pub fn is_selected(&self) -> bool {
        false
    }
    pub fn set_text_entry_policy(&mut self, input: bool) {
        if !input {
            self.editable = false;
        }
        self.text_entry_policy = input;
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.button_enabled = enabled && self.editable;
    }
    pub fn set_editable(&mut self, editable: bool) {
        if self.text_entry_policy {
            self.editable = editable;
        }
        if self.enabled {
            self.button_enabled = editable;
        }
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    pub fn set_unformatted_tooltip(&mut self, text: Option<&str>) -> Option<&str> {
        self.unformatted_tooltip = text.map(str::to_owned);
        self.unformatted_tooltip.as_deref()
    }
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.is_some()
    }
    pub fn use_unformatted_tooltip(
        &mut self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        self.tooltip = Some(format!(
            "{}{}{}",
            self.unformatted_tooltip.take().unwrap_or_default(),
            param_descr.unwrap_or_default(),
            directive_descr.unwrap_or_default()
        ));
    }
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        self.tooltip = tooltip.map(str::to_owned);
    }
    pub fn get_tooltip(&self) -> Option<&str> {
        self.tooltip.as_deref()
    }
    pub fn set_field_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    pub fn set_button_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
}
impl FileTextFieldInterface for FileTextField2 {
    fn get_file(&self) -> Option<PathBuf> {
        if self.is_empty() {
            return None;
        }
        let file = PathBuf::from(&self.text);
        if file.is_absolute() {
            Some(file)
        } else {
            Some(self.get_origin_dir().unwrap_or_default().join(file))
        }
    }
    fn set_file(&mut self, file: Option<&Path>) {
        let Some(file) = file else {
            self.text.clear();
            return;
        };
        self.text = if self.absolute_path {
            self.get_origin_dir()
                .unwrap_or_default()
                .join(file)
                .to_string_lossy()
                .into_owned()
        } else if let Some(origin) = self.get_origin_dir() {
            file.strip_prefix(origin)
                .unwrap_or(file)
                .to_string_lossy()
                .into_owned()
        } else {
            file.to_string_lossy().into_owned()
        };
    }
    fn get_file_filter(&self) -> Option<&dyn FileFilter> {
        self.file_filter.as_deref()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_file_and_origin_follow_source_rule() {
        let mut field = FileTextField2::default();
        field.set_origin(Some(Path::new("/tmp")));
        field.set_text(Some("x.rec"));
        assert_eq!(field.get_file(), Some(PathBuf::from("/tmp/x.rec")));
        field.set_absolute_path(true);
        field.set_file(Some(Path::new("x2.rec")));
        assert!(field.get_text().ends_with("/tmp/x2.rec"));
    }
    #[test]
    fn text_entry_policy_does_not_restore_editability() {
        let mut field = FileTextField2::default();
        field.set_text_entry_policy(false);
        field.set_editable(true);
        assert!(!field.editable);
    }
}
