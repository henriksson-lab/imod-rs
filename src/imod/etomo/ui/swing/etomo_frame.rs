//! Translation of `EtomoFrame.java`, the shared MainFrame/SubFrame state.
#![allow(dead_code)]

use super::etomo_menu::{EtomoMenu, ManagerMenuState, MenuTarget};
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `FrameType` values used by these three source units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameType {
    Main,
    Sub,
}
/// A Swing `ActionEvent` reduced to the source-used action command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionEvent {
    pub action_command: String,
}
impl ActionEvent {
    pub fn new(action_command: impl Into<String>) -> Self {
        Self {
            action_command: action_command.into(),
        }
    }
}
/// Source-owned visible frame state sent to the existing Slint surface.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FramePresentation {
    pub title: String,
    pub visible: bool,
    pub packed: bool,
    pub repaint_count: u64,
    pub message: Option<(String, String)>,
}
/// Rust state for Java's abstract `EtomoFrame` fields and inherited frame state.
#[derive(Clone, Debug)]
pub struct EtomoFrame {
    pub main: bool,
    pub menu: EtomoMenu,
    pub main_panel_present: bool,
    pub current_manager_present: bool,
    pub single_frame: bool,
    pub presentation: FramePresentation,
    pub axis_id: Option<AxisID>,
    pub saved_location: Option<(i32, i32)>,
}
impl EtomoFrame {
    /// `EtomoFrame()`.
    pub fn new() -> Self {
        Self::new_single_frame(false)
    }
    /// `EtomoFrame(boolean)`.
    pub fn new_single_frame(single_frame: bool) -> Self {
        Self {
            main: false,
            menu: EtomoMenu::get_instance(false),
            main_panel_present: false,
            current_manager_present: false,
            single_frame,
            presentation: FramePresentation::default(),
            axis_id: None,
            saved_location: None,
        }
    }
    /// Abstract Java `EtomoFrame.register`; MainFrame/SubFrame implement it.
    pub fn register(&mut self) -> Result<(), String> {
        Err("EtomoFrame.register is abstract; use MainFrame or SubFrame".into())
    }
    /// Abstract Java `EtomoFrame.getFrameType`; MainFrame/SubFrame implement it.
    pub fn get_frame_type(&self) -> Result<FrameType, String> {
        Err("EtomoFrame.getFrameType is abstract; use MainFrame or SubFrame".into())
    }
    /// `EtomoFrame.initialize` and private `getMenus`.
    pub fn initialize(&mut self) {
        self.menu = EtomoMenu::get_instance(false);
    }
    /// `EtomoFrame.saveLocation`.
    pub fn save_location(&mut self, ignore_loc: bool, location: (i32, i32)) {
        if !ignore_loc {
            self.saved_location = Some(location);
        }
    }
    /// `EtomoFrame.moveSubFrame`; ownership is represented by `SubFrame`.
    pub fn move_sub_frame(&mut self) {}
    /// `EtomoFrame.toFront`.
    pub fn to_front(&mut self, _axis_id: AxisID) {
        self.presentation.visible = true;
    }
    pub fn is_menu_3dmod_startup_window(&self) -> bool {
        self.menu.is_menu_3dmod_startup_window()
    }
    pub fn is_menu_save_enabled(&self) -> bool {
        self.menu.is_menu_save_enabled()
    }
    pub fn is_menu_3dmod_bin_by_2(&self) -> bool {
        self.menu.is_menu_3dmod_bin_by_2()
    }
    pub fn set_menu_3dmod_startup_window(&mut self, value: bool) {
        self.menu.set_menu_3dmod_startup_window(value);
    }
    pub fn set_menu_3dmod_bin_by_2(&mut self, value: bool) {
        self.menu.set_menu_3dmod_bin_by_2(value);
    }
    /// `EtomoFrame.menuToolsAction`.
    pub fn menu_tools_action(&self, event: &ActionEvent) -> MenuTarget {
        self.menu.menu_tools_action(&event.action_command)
    }
    /// `EtomoFrame.menuFileAction`; current manager/director calls are returned as their direct target.
    pub fn menu_file_action(&self, event: &ActionEvent) -> MenuTarget {
        self.menu
            .menu_file_action(&event.action_command)
            .unwrap_or_else(|target| target)
    }
    pub fn close(&mut self) {
        self.current_manager_present = false;
    }
    pub fn cancel(&mut self) {}
    pub fn save(&mut self, _axis_id: AxisID) -> Result<(), String> {
        if !self.current_manager_present {
            return Err("EtomoFrame.save: no current BaseManager".into());
        }
        Ok(())
    }
    pub fn save_as(&mut self) -> Result<(), String> {
        self.save(self.get_axis_id().unwrap_or(AxisID::Only))
    }
    pub fn menu_file_mru_list_action(&self, event: &ActionEvent) -> MenuTarget {
        MenuTarget::UnportedTarget(event.action_command.clone())
    }
    pub fn menu_help_action(&self, event: &ActionEvent) -> MenuTarget {
        self.menu.menu_help_action(&event.action_command)
    }
    /// `EtomoFrame.menuViewAction` (fit is presentation-owned; all other commands remain errors).
    pub fn menu_view_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        if event.action_command == self.menu.menu_fit_window.action_command {
            self.pack();
            Ok(())
        } else {
            Err(format!(
                "Cannot handle menu command in EtomoFrame: {}",
                event.action_command
            ))
        }
    }
    /// `EtomoFrame.menuOptionsAction`.
    pub fn menu_options_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        match event.action_command.as_str() {
            "Open 3dmod with Startup Window" => Ok(()),
            "Open 3dmod Binned by 2" => Ok(()),
            "Settings" => Err("Etomo SettingsDialog.java is not yet translated".into()),
            _ => Err(format!(
                "Cannot handle menu command in EtomoFrame: {}",
                event.action_command
            )),
        }
    }
    pub fn set_enabled(&mut self, manager: Option<ManagerMenuState>) {
        self.menu.set_enabled(manager);
    }
    pub fn set_mru_file_labels(&mut self, list: &[String]) {
        self.menu.set_mru_file_labels(list);
    }
    pub fn set_enabled_new_tomogram_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_tomogram(value);
    }
    pub fn set_enabled_log_window_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_log_window(value);
    }
    pub fn set_enabled_new_join_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_join(value);
    }
    pub fn set_enabled_new_generic_parallel_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_generic_parallel(value);
    }
    pub fn set_enabled_new_anisotropic_diffusion_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_anisotropic_diffusion(value);
    }
    pub fn set_enabled_new_batch_run_tomo_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_batch_run_tomo(value);
    }
    pub fn set_enabled_new_peet_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_peet(value);
    }
    pub fn set_enabled_new_serial_sections_menu_item(&mut self, value: bool) {
        self.menu.set_enabled_new_serial_sections(value);
    }
    pub fn repaint(&mut self, _axis_id: AxisID) {
        self.presentation.repaint_count += 1;
    }
    pub fn display_message(
        &mut self,
        _manager: bool,
        message: impl Into<String>,
        title: impl Into<String>,
        _axis: Option<AxisID>,
    ) {
        self.presentation.message = Some((message.into(), title.into()));
    }
    pub fn display_info_message(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        title: impl Into<String>,
        axis: AxisID,
    ) {
        self.display_message(manager, message, title, Some(axis));
    }
    /// `displayMessage(BaseManager, String[], String, AxisID)` overload.
    pub fn display_message_lines(
        &mut self,
        manager: bool,
        message: &[String],
        title: impl Into<String>,
        axis: AxisID,
    ) {
        self.display_message(manager, message.join("\n"), title, Some(axis));
    }
    pub fn display_error_message(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        title: impl Into<String>,
        axis: AxisID,
    ) {
        self.display_message(manager, message, title, Some(axis));
    }
    pub fn display_warning_message(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        title: impl Into<String>,
        axis: AxisID,
    ) {
        self.display_message(manager, message, title, Some(axis));
    }
    pub fn display_yes_no_cancel_message(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        axis: AxisID,
    ) -> i32 {
        self.display_message(manager, message, "", Some(axis));
        2
    }
    pub fn display_yes_no_message(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        axis: AxisID,
    ) -> bool {
        self.display_message(manager, message, "", Some(axis));
        false
    }
    /// `displayYesNoMessage(BaseManager, String[], AxisID)` overload.
    pub fn display_yes_no_message_lines(
        &mut self,
        manager: bool,
        message: &[String],
        axis: AxisID,
    ) -> bool {
        self.display_yes_no_message(manager, message.join("\n"), axis)
    }
    pub fn display_delete_message(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        axis: AxisID,
    ) -> bool {
        self.display_message(manager, message, "", Some(axis));
        false
    }
    pub fn display_yes_no_warning_dialog(
        &mut self,
        manager: bool,
        message: impl Into<String>,
        axis: AxisID,
    ) -> bool {
        self.display_message(manager, message, "", Some(axis));
        false
    }
    pub fn get_param_filename(&self) -> Result<bool, String> {
        Err("Etomo FileChooser.java is not yet translated".into())
    }
    pub fn pack_axis(&mut self, _axis_id: AxisID) {
        self.pack();
    }
    pub fn pack_axis_force(&mut self, _axis_id: AxisID, _force: bool) {
        self.pack();
    }
    pub fn pack(&mut self) {
        self.presentation.packed = true;
    }
    pub fn get_menus(&self) -> &EtomoMenu {
        &self.menu
    }
    pub fn open_data_file_dialog(&self) -> Result<Option<std::path::PathBuf>, String> {
        Err("Etomo FileChooser.java is not yet translated".into())
    }
    pub fn get_axis_id(&self) -> Option<AxisID> {
        self.axis_id
    }
    pub fn get_other_frame(&self) -> bool {
        self.single_frame
    }
    pub fn get_frame(&self, _axis_id: AxisID) -> &Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_menu_state_is_not_duplicated() {
        let mut frame = EtomoFrame::new();
        frame.set_menu_3dmod_bin_by_2(true);
        assert!(frame.is_menu_3dmod_bin_by_2());
    }
}
