//! `IMOD/Etomo/src/etomo/ui/swing/FileTextField.java`.
#![allow(dead_code)]

use super::file_text_field_interface::{FileFilter, FileTextFieldInterface};
use super::panel::Dimension;
use std::path::{Path, PathBuf};

/// Java final package-private `FileTextField`; Swing widgets remain an explicit boundary.
#[derive(Clone, Debug)]
pub struct FileTextField {
    pub label: Option<String>,
    pub text: String,
    pub file: Option<PathBuf>,
    pub property_user_dir: Option<PathBuf>,
    pub file_selection_mode: Option<i32>,
    pub checkpoint_value: Option<String>,
    pub prev_chooser_dir: Option<PathBuf>,
    pub use_prev_chooser_dir: bool,
    pub text_preferred_width: Option<i32>,
    pub visible: bool,
    pub enabled: bool,
    pub editable: bool,
    pub button_enabled: bool,
    pub show_partial_path: bool,
    pub action_listener_count: usize,
    pub tooltip: Option<String>,
    pub button_tooltip: Option<String>,
    pub panel_alignment_x: f32,
    pub debug: bool,
    pub required: bool,
    pub file_must_exist: bool,
}
impl FileTextField {
    /// Java private `FileTextField(String, boolean, String, boolean)`.
    pub fn new(
        label: &str,
        labeled: bool,
        property_user_dir: Option<&Path>,
        show_partial_path: bool,
    ) -> Self {
        Self {
            label: labeled.then(|| label.into()),
            text: String::new(),
            file: None,
            property_user_dir: property_user_dir.map(Path::to_path_buf),
            file_selection_mode: None,
            checkpoint_value: None,
            prev_chooser_dir: None,
            use_prev_chooser_dir: false,
            text_preferred_width: Some(250),
            visible: true,
            enabled: true,
            editable: !show_partial_path,
            button_enabled: true,
            show_partial_path,
            action_listener_count: 0,
            tooltip: None,
            button_tooltip: None,
            panel_alignment_x: 0.5,
            debug: false,
            required: false,
            file_must_exist: false,
        }
    }
    pub fn get_unlabeled_instance(action_command: &str) -> Self {
        Self::new(action_command, false, None, false)
    }
    pub fn get_unlabeled_partial_path_instance(action_command: &str) -> Self {
        Self::new(action_command, false, None, true)
    }
    pub fn get_partial_path_instance(label: &str) -> Self {
        Self::new(label, true, None, true)
    }
    pub fn set_text_preferred_width(&mut self, width: i32) {
        self.text_preferred_width = Some(width);
        if self.show_partial_path {
            self.fill_with_partial_path();
        }
    }
    pub fn set_alignment_x(&mut self, alignment: f32) {
        self.panel_alignment_x = alignment;
    }
    pub fn get_action_command(&self) -> &str {
        self.label.as_deref().unwrap_or_default()
    }
    /// Java `getContainer`; the `JPanel` is the native presentation boundary.
    pub fn get_container(&self) -> &Self {
        self
    }
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
    pub fn add_action(&mut self, property_user_dir: Option<&Path>, file_selection_mode: i32) {
        self.property_user_dir = property_user_dir.map(Path::to_path_buf);
        self.file_selection_mode = Some(file_selection_mode);
        self.add_action_listener();
    }
    pub fn action(&mut self, selected_file: Option<&Path>) {
        if let Some(file) = selected_file {
            self.set_file(Some(file));
        }
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    /// The Rust frontend supplies its optional chooser selection.
    pub fn actionPerformed(&mut self, selected_file: Option<&Path>) {
        self.action(selected_file);
    }
    pub fn set_use_prev_chooser_dir(&mut self, use_prev_chooser_dir: bool) {
        self.use_prev_chooser_dir = use_prev_chooser_dir;
    }
    pub fn clear(&mut self) {
        self.file = None;
        self.text.clear();
        self.prev_chooser_dir = None;
    }
    pub fn checkpoint(&mut self) {
        self.checkpoint_value = Some(self.get_text().to_owned());
    }
    pub fn reset_to_checkpoint(&mut self) {
        if let Some(value) = self.checkpoint_value.clone() {
            self.set_text(Some(&value));
        }
    }
    pub fn set_field_editable(&mut self, editable: bool) {
        if !self.show_partial_path {
            self.editable = editable;
        }
    }
    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    pub fn set_editable(&mut self, editable: bool) {
        if !self.show_partial_path {
            self.editable = editable;
        }
        self.button_enabled = editable;
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.button_enabled = enabled;
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn set_button_enabled(&mut self, enabled: bool) {
        self.button_enabled = enabled;
    }
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
    pub fn is_editable(&self) -> bool {
        self.editable
    }
    /// Java `setRequired`.
    pub fn set_required(&mut self, required: bool) {
        self.required = required;
    }
    /// Java `setFileMustExist`.
    pub fn set_file_must_exist(&mut self, file_must_exist: bool) {
        self.file_must_exist = file_must_exist;
    }
    pub fn exists(&mut self) -> bool {
        self.update_internal_values();
        self.file.as_ref().is_some_and(|file| file.exists())
    }
    pub fn get_file_validated(&mut self, _do_validation: bool) -> Option<PathBuf> {
        self.update_internal_values();
        self.file.clone()
    }
    pub fn get_file_name(&mut self) -> Option<String> {
        self.update_internal_values();
        self.file
            .as_ref()
            .and_then(|file| file.file_name())
            .map(|name| name.to_string_lossy().into_owned())
    }
    pub fn get_file_absolute_path(&mut self) -> Option<PathBuf> {
        self.update_internal_values();
        self.file.as_ref().map(|file| {
            if file.is_absolute() {
                file.clone()
            } else {
                std::env::current_dir().unwrap_or_default().join(file)
            }
        })
    }
    pub fn set_text(&mut self, text: Option<&str>) {
        self.set_internal_values(
            text.filter(|text| !text.trim().is_empty())
                .map(PathBuf::from),
        );
    }
    pub fn get_text(&self) -> &str {
        &self.text
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
        self.button_tooltip = text.map(str::to_owned);
    }
    pub fn set_field_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    pub fn set_button_tool_tip_text(&mut self, text: Option<&str>) {
        self.button_tooltip = text.map(str::to_owned);
    }
    fn update_internal_values(&mut self) {
        if !self.show_partial_path {
            self.file = (!self.text.trim().is_empty()).then(|| PathBuf::from(&self.text));
        }
    }
    fn set_internal_values(&mut self, file: Option<PathBuf>) {
        self.file = file;
        self.prev_chooser_dir = self
            .file
            .as_ref()
            .and_then(|file| file.parent())
            .map(Path::to_path_buf);
        self.text = self
            .file
            .as_ref()
            .map(|file| file.to_string_lossy().into_owned())
            .unwrap_or_default();
        if self.show_partial_path {
            self.fill_with_partial_path();
        }
    }
    fn fill_with_partial_path(&mut self) {
        if !self.show_partial_path {
            return;
        }
        let Some(file) = self.file.as_ref() else {
            self.text.clear();
            return;
        };
        let absolute = if file.is_absolute() {
            file.clone()
        } else {
            std::env::current_dir().unwrap_or_default().join(file)
        };
        let limit = self.text_preferred_width.unwrap_or(250).max(1) as usize;
        let value = absolute.to_string_lossy();
        if value.len() <= limit {
            self.text = value.into_owned();
            return;
        }
        let parent = absolute
            .parent()
            .and_then(Path::file_name)
            .map(|x| x.to_string_lossy())
            .unwrap_or_default();
        let name = absolute
            .file_name()
            .map(|x| x.to_string_lossy())
            .unwrap_or_default();
        self.text = format!(".../{parent}/{name}");
    }
}
impl FileTextFieldInterface for FileTextField {
    fn get_file(&self) -> Option<PathBuf> {
        self.file.clone()
    }
    fn set_file(&mut self, file: Option<&Path>) {
        self.set_internal_values(file.map(Path::to_path_buf));
    }
    fn get_file_filter(&self) -> Option<&dyn FileFilter> {
        None
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_path_retains_backing_file() {
        let mut field = FileTextField::get_partial_path_instance("x");
        field.set_file(Some(Path::new("/long/parent/a.rec")));
        assert_eq!(field.get_file(), Some(PathBuf::from("/long/parent/a.rec")));
        assert!(field.get_text().contains("a.rec"));
    }
}
