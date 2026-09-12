//! `IMOD/Etomo/src/etomo/ui/swing/ComboBoxEfield.java`.
//!
//! `JComboBox`, `EfieldContainer`, GridBag placement, and listener dispatch
//! are GUI boundaries.  The source unit's selection, flag and saved state are
//! retained here.
#![allow(dead_code)]

use super::appearance_extension::FlagType;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::util::utilities;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComboBoxOption {
    pub value: Option<String>,
    pub descr: Option<String>,
    pub include_value: bool,
}
impl ComboBoxOption {
    pub fn get_display_string(&self) -> Option<String> {
        if self.descr.is_none() {
            return self.value.clone();
        }
        if self.value.is_none() || !self.include_value {
            return self.descr.clone();
        }
        Some(format!(
            "{}: {}",
            self.value.as_deref().unwrap_or_default(),
            self.descr.as_deref().unwrap_or_default()
        ))
    }
    pub fn equals_text(&self, text: &str) -> bool {
        let text = text.trim();
        self.get_display_string().as_deref() == Some(text)
            || self
                .value
                .as_deref()
                .is_some_and(|v| v.eq_ignore_ascii_case(text))
            || self
                .descr
                .as_deref()
                .is_some_and(|v| v.eq_ignore_ascii_case(text))
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComboBoxItem {
    Empty,
    Option(ComboBoxOption),
    Text(String),
}
impl ComboBoxItem {
    pub fn equals_text(&self, text: &str) -> bool {
        match self {
            Self::Empty => false,
            Self::Option(v) => v.equals_text(text),
            Self::Text(v) => v == text,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlState {
    Override,
    Enable,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ControlComponentModuleBoundary {
    pub container_name: Option<String>,
    pub override_selected: bool,
    pub component_control_return: bool,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GridBagExtensionBoundary {
    pub add_count: usize,
    pub remove_count: usize,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextFlagExtensionBoundary {
    pub debug: bool,
    pub update_count: usize,
    pub flag_errors: bool,
    pub template_value: Option<String>,
    pub display_count: usize,
    pub final_display_count: usize,
}

/// Java package-private `ComboBoxEfield`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComboBoxEfield {
    pub combo_box_items: Vec<ComboBoxItem>,
    pub combo_box_name: Option<String>,
    pub combo_box_selected_index: i32,
    pub combo_box_visible: bool,
    pub combo_box_enabled: bool,
    pub combo_box_editable: bool,
    pub combo_box_tool_tip_text: Option<String>,
    pub combo_box_focus_listener_count: usize,
    pub combo_box_item_listener_count: usize,
    pub label: String,
    pub include_value: bool,
    pub control_component: Option<ControlComponentModuleBoundary>,
    pub container_visible: bool,
    pub state_extension_backup: Option<Option<String>>,
    pub state_extension_checkpoint: Option<Option<String>>,
    pub appearance_extension_created: bool,
    pub appearance_extension_enabled: bool,
    pub appearance_extension_editable: bool,
    pub appearance_extension_flag_type: Option<FlagType>,
    pub flag_extension: Option<TextFlagExtensionBoundary>,
    pub grid_bag_extension: Option<GridBagExtensionBoundary>,
    pub choice_list_set: bool,
    pub debug: bool,
    pub value_manipulation_extension: Option<String>,
    pub empty_index: i32,
    pub directive_def: Option<String>,
}
impl ComboBoxEfield {
    /// Java `ComboBoxEfield(String, boolean, boolean)`.
    pub fn new(label: &str, include_value: bool, include_control_component: bool) -> Self {
        let mut v = Self {
            combo_box_items: Vec::new(),
            combo_box_name: None,
            combo_box_selected_index: -1,
            combo_box_visible: true,
            combo_box_enabled: true,
            combo_box_editable: false,
            combo_box_tool_tip_text: None,
            combo_box_focus_listener_count: 0,
            combo_box_item_listener_count: 0,
            label: label.into(),
            include_value,
            control_component: include_control_component
                .then_some(ControlComponentModuleBoundary::default()),
            container_visible: true,
            state_extension_backup: None,
            state_extension_checkpoint: None,
            appearance_extension_created: false,
            appearance_extension_enabled: true,
            appearance_extension_editable: false,
            appearance_extension_flag_type: None,
            flag_extension: None,
            grid_bag_extension: None,
            choice_list_set: false,
            debug: false,
            value_manipulation_extension: None,
            empty_index: 0,
            directive_def: None,
        };
        v.set_name(label);
        v.combo_box_items.push(ComboBoxItem::Empty);
        v.combo_box_selected_index = 0;
        v
    }
    pub fn get_instance(label: &str, include_value: bool) -> Self {
        Self::new(label, include_value, false)
    }
    pub fn get_override_instance(label: &str, include_value: bool) -> Self {
        Self::new(label, include_value, true)
    }
    pub fn set_name(&mut self, text: &str) {
        let name = utilities::convert_label_to_name(Some(text), true).unwrap_or_default();
        self.combo_box_name = Some(format!("combo-box{SEPARATOR_CHAR}{name}"));
        if let Some(v) = &mut self.control_component {
            v.container_name = Some(text.into());
        }
    }
    pub fn get_label(&self) -> &str {
        &self.label
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
        if let Some(v) = &mut self.flag_extension {
            v.debug = debug;
        }
    }
    pub fn is_debug(&self) -> bool {
        self.debug
    }
    pub fn set_choice_list(&mut self, choices: &[Option<ComboBoxOption>]) {
        self.choice_list_set = true;
        for choice in choices.iter().flatten() {
            let mut c = choice.clone();
            c.include_value = self.include_value;
            self.combo_box_items.push(ComboBoxItem::Option(c));
        }
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component_visible(&self) -> bool {
        self.container_visible
    }
    pub fn set_choice_list_set(&mut self, set: bool) {
        self.choice_list_set = set;
    }
    pub fn set_selected_index(&mut self, index: i32) {
        self.combo_box_selected_index = index;
    }
    pub fn set_text_file(&mut self, file: Option<&Path>) {
        if let Some(file) = file {
            self.set_text(&file.to_string_lossy());
        } else {
            self.set_text("");
        }
    }
    pub fn set_text(&mut self, text: &str) {
        if text.trim().is_empty() {
            self.clear();
            return;
        }
        let size = self.combo_box_items.len() as i32;
        for i in (self.empty_index + 1)..size {
            if self.combo_box_items[i as usize].equals_text(text) {
                self.set_selected_index(i);
                self.update_flag_extension();
                return;
            }
        }
        self.combo_box_items.push(ComboBoxItem::Text(text.into()));
        self.set_selected_index(size);
        if self.choice_list_set {
            self.create_flag_extension();
            if let Some(v) = &mut self.flag_extension {
                v.flag_errors = true;
            }
        }
        self.update_flag_extension();
    }
    pub fn get_text(&self) -> Option<String> {
        if self.is_controlled() {
            return Some(String::new());
        }
        match self
            .combo_box_items
            .get(self.combo_box_selected_index as usize)?
        {
            ComboBoxItem::Empty => None,
            ComboBoxItem::Option(v) => v.value.clone(),
            ComboBoxItem::Text(v) => Some(v.clone()),
        }
    }
    pub fn is_controlled(&self) -> bool {
        false
    }
    pub fn get_selected_index(&self) -> i32 {
        self.combo_box_selected_index
    }
    pub fn is_empty(&self) -> bool {
        let i = self.get_selected_index();
        i == -1 || (self.empty_index != -1 && i == self.empty_index)
    }
    pub fn equals(&self, text: Option<&str>) -> bool {
        match (
            self.combo_box_items
                .get(self.combo_box_selected_index as usize),
            text,
        ) {
            (None | Some(ComboBoxItem::Empty), None) => true,
            (Some(v), Some(text)) => v.equals_text(text),
            _ => false,
        }
    }
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.combo_box_tool_tip_text = Some(text.into());
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.container_visible = visible;
    }
    pub fn set_combo_box_visible(&mut self, visible: bool) {
        self.combo_box_visible = visible;
    }
    pub fn add_item(&mut self, option: ComboBoxOption) {
        self.combo_box_items.push(ComboBoxItem::Option(option));
    }
    pub fn is_visible(&self) -> bool {
        self.container_visible
    }
    pub fn get_directive_def(&self) -> Option<&str> {
        self.directive_def.as_deref()
    }
    pub fn set_directive_def(&mut self, value: Option<String>) {
        self.directive_def = value;
    }
    pub fn is_template_value(&self) -> bool {
        self.get_flag_type().is_some_and(FlagType::is_template)
    }
    pub fn is_valid(&self) -> bool {
        !self.choice_list_set
            || self.is_controlled()
            || !matches!(
                self.combo_box_items
                    .get(self.combo_box_selected_index as usize),
                Some(ComboBoxItem::Text(_))
            )
    }
    pub fn clear(&mut self) {
        if self.empty_index != -1 {
            self.set_selected_index(self.empty_index);
        } else if let Some(v) = self.value_manipulation_extension.clone() {
            self.set_text(&v);
        }
        self.update_flag_extension();
    }
    pub fn add_value_manipulation_listener(&mut self) {
        self.combo_box_focus_listener_count += 1;
    }
    pub fn is_override(&self) -> bool {
        self.control_component
            .as_ref()
            .is_some_and(|v| v.override_selected)
    }
    pub fn set_component_control(&mut self, control: bool, _state: ControlState) {
        if let Some(v) = &mut self.control_component {
            v.component_control_return = control;
            self.combo_box_visible = !v.component_control_return;
        }
    }
    pub fn send_control_event(&mut self) {}
    pub fn set_enable_control(&mut self, _control: bool, _state: ControlState) {}
    pub fn create_appearance_extension(&mut self) {
        if !self.appearance_extension_created {
            self.appearance_extension_created = true;
            self.appearance_extension_enabled = self.combo_box_enabled;
            self.appearance_extension_editable = self.combo_box_editable;
        }
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.create_appearance_extension();
        self.appearance_extension_editable = editable;
        self.combo_box_editable = editable;
    }
    pub fn is_enabled(&self) -> bool {
        if self.appearance_extension_created {
            self.appearance_extension_enabled
        } else {
            self.combo_box_enabled
        }
    }
    pub fn is_editable(&self) -> bool {
        self.appearance_extension_created && self.appearance_extension_editable
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.combo_box_enabled = enabled;
        if self.appearance_extension_created {
            self.appearance_extension_enabled = enabled;
        }
    }
    pub fn update_flag_extension(&mut self) {
        if let Some(v) = &mut self.flag_extension {
            v.update_count += 1;
        }
    }
    pub fn create_flag_extension(&mut self) -> bool {
        if self.flag_extension.is_none() {
            self.flag_extension = Some(TextFlagExtensionBoundary {
                debug: self.debug,
                ..Default::default()
            });
            true
        } else {
            false
        }
    }
    pub fn get_flag_type(&self) -> Option<FlagType> {
        self.appearance_extension_flag_type
    }
    pub fn add_flag_origin_listener(&mut self) {
        self.combo_box_item_listener_count += 1;
    }
    pub fn flag_template(&mut self, value: Option<&str>) {
        let Some(value) = value else { return };
        self.create_flag_extension();
        if let Some(v) = &mut self.flag_extension {
            v.template_value = Some(value.into());
        }
        self.update_flag_extension();
        if self.value_manipulation_extension.is_none() {
            self.value_manipulation_extension = Some(value.into());
        }
    }
    pub fn clear_template_value(&mut self) {
        if let Some(v) = &mut self.flag_extension {
            v.template_value = None;
            v.flag_errors = false;
        }
        self.value_manipulation_extension = None;
    }
    pub fn set_template_value(&mut self) {
        if let Some(v) = self
            .flag_extension
            .as_ref()
            .and_then(|v| v.template_value.clone())
        {
            self.set_text(&v);
        }
    }
    pub fn add_flag_display(&mut self) {
        self.create_flag_extension();
        if let Some(v) = &mut self.flag_extension {
            v.display_count += 1;
        }
        self.update_flag_extension();
    }
    pub fn add_final_flag_display(&mut self) {
        self.create_flag_extension();
        if let Some(v) = &mut self.flag_extension {
            v.final_display_count += 1;
        }
        self.update_flag_extension();
    }
    pub fn set_field_highlight(&mut self, value: Option<&str>) {
        self.flag_template(value);
    }
    pub fn create_state_extension(&mut self) {
        if self.state_extension_backup.is_none() {
            self.state_extension_backup = Some(None);
            self.state_extension_checkpoint = Some(None);
        }
    }
    pub fn backup(&mut self) {
        self.create_state_extension();
        self.state_extension_backup = Some(self.get_text());
    }
    pub fn checkpoint(&mut self) {
        self.create_state_extension();
        self.state_extension_checkpoint = Some(self.get_text());
    }
    pub fn is_different_from_checkpoint(&self, _always_check: bool) -> bool {
        self.state_extension_checkpoint
            .as_ref()
            .is_some_and(|v| v != &self.get_text())
    }
    pub fn restore_from_backup(&mut self) {
        if self.state_extension_backup.is_some() {
            match self.state_extension_backup.clone().flatten() {
                Some(v) => self.set_text(&v),
                None => self.clear(),
            }
        }
    }
    pub fn remove(&mut self) {
        if let Some(v) = &mut self.grid_bag_extension {
            v.remove_count += 1;
        }
    }
    pub fn add(&mut self) {
        if self.grid_bag_extension.is_none() {
            self.grid_bag_extension = Some(GridBagExtensionBoundary::default());
        }
        if let Some(v) = &mut self.grid_bag_extension {
            v.add_count += 1;
        }
    }
    pub fn set_text_files(&mut self, _files: Vec<PathBuf>) {}
    pub fn is_local_dir(&self, _current_directory: Option<&str>) -> bool {
        false
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_preserves_empty_item_and_name() {
        let f = ComboBoxEfield::get_instance("Choice:", true);
        assert_eq!(f.combo_box_selected_index, 0);
        assert_eq!(f.combo_box_items, vec![ComboBoxItem::Empty]);
        assert_eq!(f.combo_box_name.as_deref(), Some("combo-box.choice"));
    }
    #[test]
    fn options_copy_include_value_and_match_description() {
        let mut f = ComboBoxEfield::get_instance("Choice", true);
        f.set_choice_list(&[Some(ComboBoxOption {
            value: Some("a".into()),
            descr: Some("Alpha".into()),
            include_value: false,
        })]);
        f.set_text("alpha");
        assert_eq!(f.get_text().as_deref(), Some("a"));
        assert!(f.is_valid());
        assert!(matches!(&f.combo_box_items[1], ComboBoxItem::Option(v) if v.include_value));
    }
    #[test]
    fn unknown_choice_is_flagged_invalid() {
        let mut f = ComboBoxEfield::get_instance("Choice", false);
        f.set_choice_list(&[]);
        f.set_text("unknown");
        assert!(!f.is_valid());
        assert!(f.flag_extension.as_ref().unwrap().flag_errors);
    }
    #[test]
    fn backup_and_template_value_follow_source() {
        let mut f = ComboBoxEfield::get_instance("Choice", false);
        f.set_text("saved");
        f.backup();
        f.set_field_highlight(Some("template"));
        f.set_template_value();
        assert_eq!(f.get_text().as_deref(), Some("template"));
        f.restore_from_backup();
        assert_eq!(f.get_text().as_deref(), Some("saved"));
        f.clear_template_value();
        assert!(f.value_manipulation_extension.is_none());
    }
}
