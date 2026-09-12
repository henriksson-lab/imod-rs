//! `IMOD/Etomo/src/etomo/ui/swing/SerialSectionsStartupDialog.java`.
//!
//! Native file chooser, window, dataset validation, and manager process calls remain
//! named boundaries. The saved startup state and source UI transitions live here.
#![allow(dead_code)]

use super::context_menu::{ContextMenu, MouseEvent};
use crate::imod::etomo::r#type::{axis_id::AxisID, dialog_type::DialogType, view_type::ViewType};
use std::path::{Path, PathBuf};

pub const NAME: &str = "Starting Serial Sections";
pub const VIEW_TYPE_LABEL: &str = "Frame Type";

/// Java `SerialSectionsStartupData` state created by `saveState`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SerialSectionsStartupData {
    pub stack: PathBuf,
    pub view_type: ViewType,
    pub mdoc_metadata_file_status: bool,
    pub distortion_field: Option<PathBuf>,
    pub images_are_binned: i32,
}
impl SerialSectionsStartupData {
    pub fn get_root_name(&self) -> Option<String> {
        self.stack
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
    }
    pub fn validate(&self) -> Option<String> {
        if self.stack.as_os_str().is_empty() {
            Some("Stack is required".into())
        } else {
            None
        }
    }
}

/// Genuine manager, dataset validation, busy-status, and window boundaries.
pub trait SerialSectionsStartupDialogApplicationManager {
    fn validate_dataset_name(&mut self, stack: &Path) -> bool;
    fn validate_view_type(&mut self, view_type: ViewType, stack: &Path) -> bool;
    fn complete_startup(&mut self, axis_id: AxisID);
    fn cancel_startup(&mut self);
    fn set_startup_data(&mut self, startup_data: SerialSectionsStartupData);
    fn remove_busy_status_listeners(&mut self);
}
/// Java `ContextPopup` request from `popUpContextMenu`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SerialSectionsStartupContextPopup {
    pub mouse_event: MouseEvent,
    pub standard_menu: bool,
}

/// Java `SerialSectionsStartupDialog` source fields, excluding Swing-owned widgets.
pub struct SerialSectionsStartupDialog {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub root_name: String,
    pub stack: Option<PathBuf>,
    pub distortion_field: Option<PathBuf>,
    pub view_type: ViewType,
    pub images_are_binned: i32,
    pub mdoc_metadata_file_selected: bool,
    pub mdoc_metadata_file_enabled: bool,
    pub startup_data: Option<SerialSectionsStartupData>,
    pub displayed: bool,
    pub disposed: bool,
    pub constructed: bool,
    pub listeners_added: bool,
    pub tooltips_set: bool,
    pub context_popup: Option<SerialSectionsStartupContextPopup>,
}
impl SerialSectionsStartupDialog {
    /// Java private `SerialSectionsStartupDialog(SerialSectionsManager, AxisID)`.
    pub fn new(axis_id: AxisID) -> Self {
        Self {
            axis_id,
            dialog_type: DialogType::SerialSectionsStartup,
            root_name: NAME.into(),
            stack: None,
            distortion_field: None,
            view_type: ViewType::SingleView,
            images_are_binned: 1,
            mdoc_metadata_file_selected: false,
            mdoc_metadata_file_enabled: false,
            startup_data: None,
            displayed: false,
            disposed: false,
            constructed: false,
            listeners_added: false,
            tooltips_set: false,
            context_popup: None,
        }
    }
    /// Java static `getInstance`.
    pub fn get_instance(axis_id: AxisID) -> Self {
        let mut instance = Self::new(axis_id);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }
    /// Java `createPanel`; panel geometry remains the native GUI boundary.
    pub fn create_panel(&mut self) {
        self.constructed = true;
        self.mdoc_metadata_file_enabled = false;
    }
    pub fn add_listeners(&mut self) {
        self.listeners_added = true;
    }
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.context_popup = Some(SerialSectionsStartupContextPopup {
            mouse_event,
            standard_menu: true,
        });
    }
    pub fn display(&mut self) {
        self.displayed = true;
        self.disposed = false;
    }
    /// Java `validate` direct `DatasetTool` calls.
    pub fn validate<M: SerialSectionsStartupDialogApplicationManager>(
        &self,
        manager: &mut M,
    ) -> bool {
        let Some(stack) = self.get_stack() else {
            return false;
        };
        manager.validate_dataset_name(&stack)
            && manager.validate_view_type(self.get_view_type(), &stack)
    }
    /// Java `done`.
    pub fn done<M: SerialSectionsStartupDialogApplicationManager>(&mut self, manager: &mut M) {
        self.dispose(manager);
        if let Some(startup_data) = self.startup_data.clone() {
            manager.set_startup_data(startup_data);
        }
    }
    pub fn reset_saved_state(&mut self) {
        self.startup_data = None;
    }
    /// Java `action(ActionEvent)`, narrowed to button action command.
    pub fn action<M: SerialSectionsStartupDialogApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
    ) {
        self.update_display();
        if command == "OK" {
            if !self.validate(manager) || !self.save_state() {
                return;
            }
            manager.complete_startup(self.axis_id);
        } else if command == "Cancel" {
            self.reset_saved_state();
            self.dispose(manager);
            manager.cancel_startup();
        }
    }
    /// Java `processResult(Object, boolean)`.
    pub fn process_result(&mut self) {
        self.mdoc_metadata_file_enabled = self.validate_cb_mdoc_metadata_file();
    }
    pub fn validate_cb_mdoc_metadata_file(&self) -> bool {
        let Some(stack) = self.stack.as_ref() else {
            return false;
        };
        let mdoc = PathBuf::from(format!("{}.mdoc", stack.display()));
        let pl = stack.with_extension("pl");
        mdoc.exists() && !pl.exists() && self.view_type == ViewType::Montage
    }
    pub fn get_mdoc_metadata_file_status(&self) -> bool {
        self.validate_cb_mdoc_metadata_file() && self.mdoc_metadata_file_selected
    }
    pub fn get_startup_data(&self) -> Option<&SerialSectionsStartupData> {
        self.startup_data.as_ref()
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> bool {
        self.constructed
    }
    pub fn get_distortion_field(&self) -> Option<PathBuf> {
        self.startup_data
            .as_ref()
            .and_then(|data| data.distortion_field.clone())
            .or_else(|| self.distortion_field.clone())
    }
    pub fn get_property_user_dir(&self) -> Option<PathBuf> {
        self.get_stack()
            .and_then(|stack| stack.parent().map(Path::to_path_buf))
    }
    pub fn get_stack(&self) -> Option<PathBuf> {
        self.startup_data
            .as_ref()
            .map(|data| data.stack.clone())
            .or_else(|| self.stack.clone())
    }
    pub fn get_view_type(&self) -> ViewType {
        self.startup_data
            .as_ref()
            .map(|data| data.view_type)
            .unwrap_or(self.view_type)
    }
    pub fn get_root_name(&self) -> Option<String> {
        self.startup_data
            .as_ref()
            .and_then(SerialSectionsStartupData::get_root_name)
            .or_else(|| {
                self.stack.as_ref().and_then(|path| {
                    path.file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                })
            })
    }
    /// Java `saveState`.
    pub fn save_state(&mut self) -> bool {
        let Some(stack) = self.stack.clone() else {
            self.reset_saved_state();
            return false;
        };
        let startup_data = SerialSectionsStartupData {
            stack,
            view_type: self.view_type,
            mdoc_metadata_file_status: self.get_mdoc_metadata_file_status(),
            distortion_field: self.distortion_field.clone(),
            images_are_binned: self.images_are_binned,
        };
        if startup_data.validate().is_some() {
            self.reset_saved_state();
            return false;
        }
        self.startup_data = Some(startup_data);
        true
    }
    pub fn dispose<M: SerialSectionsStartupDialogApplicationManager>(&mut self, manager: &mut M) {
        self.displayed = false;
        self.disposed = true;
        manager.remove_busy_status_listeners();
    }
    pub fn window_closing<M: SerialSectionsStartupDialogApplicationManager>(
        &mut self,
        manager: &mut M,
    ) {
        self.dispose(manager);
        manager.cancel_startup();
    }
    pub fn set_tooltips(&mut self) {
        self.tooltips_set = true;
    }
    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        if self.stack.is_none() {
            return;
        }
        self.mdoc_metadata_file_enabled =
            self.view_type == ViewType::Montage && self.validate_cb_mdoc_metadata_file();
    }
}
impl ContextMenu for SerialSectionsStartupDialog {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.pop_up_context_menu(mouse_event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn construction_matches_source_defaults() {
        let dialog = SerialSectionsStartupDialog::get_instance(AxisID::Only);
        assert!(dialog.constructed);
        assert!(dialog.listeners_added);
        assert_eq!(dialog.images_are_binned, 1);
        assert!(!dialog.mdoc_metadata_file_enabled);
    }
    #[test]
    fn save_state_prefers_saved_values_for_accessors() {
        let mut dialog = SerialSectionsStartupDialog::get_instance(AxisID::Only);
        dialog.stack = Some(PathBuf::from("/data/input.st"));
        dialog.distortion_field = Some(PathBuf::from("/data/distort.idf"));
        dialog.view_type = ViewType::Montage;
        assert!(dialog.save_state());
        dialog.stack = Some(PathBuf::from("/other.st"));
        assert_eq!(dialog.get_stack(), Some(PathBuf::from("/data/input.st")));
        assert_eq!(dialog.get_root_name().as_deref(), Some("input"));
    }
    #[test]
    fn context_popup_preserves_standard_menu_request() {
        let mut dialog = SerialSectionsStartupDialog::new(AxisID::Only);
        dialog.pop_up_context_menu(MouseEvent {
            x: 8,
            y: 9,
            right_mouse_button: true,
        });
        assert_eq!(
            dialog.context_popup.unwrap().mouse_event,
            MouseEvent {
                x: 8,
                y: 9,
                right_mouse_button: true
            }
        );
    }
}
