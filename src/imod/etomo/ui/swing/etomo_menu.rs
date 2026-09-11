//! `IMOD/Etomo/src/etomo/ui/swing/EtomoMenu.java`.
//!
//! This module owns menu state.  Visual menus (Slint) and `UIHarness` submit
//! their Java action-command strings here; there is no second menu-state
//! model.  A selected command is returned as a direct target or as an explicit
//! `UnportedTarget`, never silently executed.
#![allow(dead_code)]

pub const RECON_LABEL: &str = "Build Tomogram";
pub const JOIN_LABEL: &str = "Join Serial Tomograms";
pub const GENERIC_LABEL: &str = "Generic Parallel Process";
pub const NAD_LABEL: &str = "Nonlinear Anisotropic Diffusion";
pub const BATCH_RUN_TOMO_LABEL: &str = "Batch Tomograms";
pub const PEET_LABEL: &str = "Subvolume Averaging (PEET)";
pub const FLATTEN_VOLUME_LABEL: &str = "Flatten Volume";
pub const GPU_TILT_TEST_LABEL: &str = "Test GPU";
pub const ALIGN_FRAMES_LABEL: &str = "Align Frames";
pub const SERIAL_SECTIONS_LABEL: &str = "Align Serial Sections / Blend Montages";
pub const N_MRU_FILE_MAX: usize = 10;
pub const TOP_ANCHOR: &str = "#TOP";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MenuItem {
    pub action_command: String,
    pub enabled: bool,
    pub visible: bool,
    pub selected: bool,
}
impl MenuItem {
    pub fn new(command: &str) -> Self {
        Self {
            action_command: command.into(),
            enabled: true,
            visible: true,
            selected: false,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectiveFileType {
    Scope,
    System,
    User,
    Batch,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolType {
    FlattenVolume,
    GpuTiltTest,
    AlignFrames,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuTarget {
    Save,
    SaveAs,
    Close,
    Cancel,
    Exit,
    Open,
    Tomosnapshot,
    New(&'static str),
    Tool(ToolType),
    View(&'static str),
    Option(&'static str),
    Guide(&'static str),
    Directive(DirectiveFileType),
    Mru(usize),
    UnportedTarget(String),
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ManagerMenuState {
    pub setup_done: bool,
    pub can_change_param_file_name: bool,
    pub dual_axis: bool,
    pub can_save_directives: bool,
}

/// Java `EtomoMenu` fields, with each `JMenuItem` represented by its action
/// command and state.  Swing's JMenuBar composition is presentation-only.
#[derive(Clone, Debug)]
pub struct EtomoMenu {
    pub dataset: bool,
    pub savable: bool,
    pub peet_available: bool,
    pub menu_open: MenuItem,
    pub menu_save: MenuItem,
    pub menu_save_as: MenuItem,
    pub menu_close: MenuItem,
    pub menu_cancel: MenuItem,
    pub menu_exit: MenuItem,
    pub menu_tomosnapshot: MenuItem,
    pub menu_export_batch: MenuItem,
    pub menu_new_tomogram: MenuItem,
    pub menu_new_join: MenuItem,
    pub menu_new_peet: MenuItem,
    pub menu_serial_sections: MenuItem,
    pub menu_new_anisotropic_diffusion: MenuItem,
    pub menu_new_batch_run_tomo: MenuItem,
    pub menu_new_generic_parallel: MenuItem,
    pub menu_save_scope: MenuItem,
    pub menu_save_system: MenuItem,
    pub menu_save_user: MenuItem,
    pub menu_flatten_volume: MenuItem,
    pub menu_gpu_tilt_test: MenuItem,
    pub menu_align_frames: MenuItem,
    pub menu_log_window: MenuItem,
    pub menu_axis_a: MenuItem,
    pub menu_axis_b: MenuItem,
    pub menu_axis_both: MenuItem,
    pub menu_fit_window: MenuItem,
    pub menu_settings: MenuItem,
    pub menu_3dmod_startup_window: MenuItem,
    pub menu_3dmod_bin_by_2: MenuItem,
    pub menu_tomo_guide: MenuItem,
    pub menu_imod_guide: MenuItem,
    pub menu_3dmod_guide: MenuItem,
    pub menu_etomo_guide: MenuItem,
    pub menu_join_guide: MenuItem,
    pub menu_peet_guide: MenuItem,
    pub peet_help_item: MenuItem,
    pub menu_batch_guide: MenuItem,
    pub menu_help_about: MenuItem,
    pub menu_mru_list: [MenuItem; N_MRU_FILE_MAX],
}
impl EtomoMenu {
    /// `getInstance(AbstractFrame)`.
    pub fn get_instance(peet_available: bool) -> Self {
        Self::new(true, true, peet_available)
    }
    /// `getInstance(ManagerFrame, boolean)`.
    pub fn get_manager_instance(savable: bool, peet_available: bool) -> Self {
        Self::new(false, savable, peet_available)
    }
    /// Constructor + `createPanel` + `addListeners`.
    pub fn new(dataset: bool, savable: bool, peet_available: bool) -> Self {
        let mut menu = Self {
            dataset,
            savable,
            peet_available,
            menu_open: MenuItem::new("Open..."),
            menu_save: MenuItem::new("Save"),
            menu_save_as: MenuItem::new("Save As..."),
            menu_close: MenuItem::new("Close"),
            menu_cancel: MenuItem::new("Cancel"),
            menu_exit: MenuItem::new("Exit"),
            menu_tomosnapshot: MenuItem::new("Run Tomosnapshot"),
            menu_export_batch: MenuItem::new("Export Batch Directive File"),
            menu_new_tomogram: MenuItem::new(RECON_LABEL),
            menu_new_join: MenuItem::new(JOIN_LABEL),
            menu_new_peet: MenuItem::new(PEET_LABEL),
            menu_serial_sections: MenuItem::new(SERIAL_SECTIONS_LABEL),
            menu_new_anisotropic_diffusion: MenuItem::new(NAD_LABEL),
            menu_new_batch_run_tomo: MenuItem::new(BATCH_RUN_TOMO_LABEL),
            menu_new_generic_parallel: MenuItem::new(GENERIC_LABEL),
            menu_save_scope: MenuItem::new("Save Scope Template"),
            menu_save_system: MenuItem::new("Save System Template"),
            menu_save_user: MenuItem::new("Save User Template"),
            menu_flatten_volume: MenuItem::new(FLATTEN_VOLUME_LABEL),
            menu_gpu_tilt_test: MenuItem::new(GPU_TILT_TEST_LABEL),
            menu_align_frames: MenuItem::new(ALIGN_FRAMES_LABEL),
            menu_log_window: MenuItem::new("Show/Hide Log Window"),
            menu_axis_a: MenuItem::new("Axis A"),
            menu_axis_b: MenuItem::new("Axis B"),
            menu_axis_both: MenuItem::new("Both Axes"),
            menu_fit_window: MenuItem::new("Fit Window"),
            menu_settings: MenuItem::new("Settings"),
            menu_3dmod_startup_window: MenuItem::new("Open 3dmod with Startup Window"),
            menu_3dmod_bin_by_2: MenuItem::new("Open 3dmod Binned by 2"),
            menu_tomo_guide: MenuItem::new("Tomography Guide"),
            menu_imod_guide: MenuItem::new("Imod Users Guide"),
            menu_3dmod_guide: MenuItem::new("3dmod Users Guide"),
            menu_etomo_guide: MenuItem::new("Etomo Users Guide"),
            menu_join_guide: MenuItem::new("Join Users Guide"),
            menu_peet_guide: MenuItem::new("PEET Users Guide"),
            peet_help_item: MenuItem::new("PEET Help"),
            menu_batch_guide: MenuItem::new("Batch Interface Guide"),
            menu_help_about: MenuItem::new("About"),
            menu_mru_list: std::array::from_fn(|_| {
                let mut i = MenuItem::new("");
                i.visible = false;
                i
            }),
        };
        menu.create_panel();
        menu.add_listeners();
        menu
    }
    /// `createPanel` (structure is exposed to Slint through item state).
    pub fn create_panel(&mut self) {
        if !(self.dataset || self.savable) {
            self.menu_open.visible = false;
            self.menu_save.visible = false;
            self.menu_save_as.visible = false;
            self.menu_close.visible = false;
            self.menu_cancel.visible = false;
            self.menu_exit.visible = false;
        }
        if !self.dataset {
            self.menu_open.visible = false;
            self.menu_tomosnapshot.visible = false;
            self.menu_export_batch.visible = false;
        }
        if !self.peet_available {
            self.menu_peet_guide.visible = false;
            self.peet_help_item.visible = false;
        }
    }
    /// `addListeners`; commands are already stable Swing action commands.
    pub fn add_listeners(&mut self) {}
    /// `setEnabled(BaseManager)`.
    pub fn set_enabled(&mut self, manager: Option<ManagerMenuState>) {
        let Some(m) = manager else {
            for i in [
                &mut self.menu_save,
                &mut self.menu_save_as,
                &mut self.menu_close,
                &mut self.menu_axis_a,
                &mut self.menu_axis_b,
                &mut self.menu_axis_both,
                &mut self.menu_export_batch,
                &mut self.menu_save_scope,
                &mut self.menu_save_system,
                &mut self.menu_save_user,
            ] {
                i.enabled = false;
            }
            return;
        };
        self.menu_save.enabled = m.setup_done;
        self.menu_save_as.enabled = m.can_change_param_file_name;
        self.menu_close.enabled = true;
        for i in [
            &mut self.menu_axis_a,
            &mut self.menu_axis_b,
            &mut self.menu_axis_both,
        ] {
            i.enabled = m.dual_axis;
        }
        for i in [
            &mut self.menu_export_batch,
            &mut self.menu_save_scope,
            &mut self.menu_save_system,
            &mut self.menu_save_user,
        ] {
            i.enabled = m.can_save_directives;
        }
    }
    /// `setEnabled(EtomoMenu)`.
    pub fn set_enabled_from(&mut self, main: &EtomoMenu) {
        for (to, from) in [
            (&mut self.menu_new_tomogram, &main.menu_new_tomogram),
            (&mut self.menu_new_join, &main.menu_new_join),
            (
                &mut self.menu_new_generic_parallel,
                &main.menu_new_generic_parallel,
            ),
            (
                &mut self.menu_new_anisotropic_diffusion,
                &main.menu_new_anisotropic_diffusion,
            ),
            (
                &mut self.menu_new_batch_run_tomo,
                &main.menu_new_batch_run_tomo,
            ),
            (&mut self.menu_new_peet, &main.menu_new_peet),
            (&mut self.menu_serial_sections, &main.menu_serial_sections),
            (&mut self.menu_save_as, &main.menu_save_as),
            (&mut self.menu_axis_a, &main.menu_axis_a),
            (&mut self.menu_axis_b, &main.menu_axis_b),
            (&mut self.menu_axis_both, &main.menu_axis_both),
        ] {
            to.enabled = from.enabled;
        }
    }
    /// `setMRUFileLabels`.
    pub fn set_mru_file_labels(&mut self, files: &[String]) {
        for i in 0..N_MRU_FILE_MAX {
            let value = files.get(i).map(String::as_str).unwrap_or("");
            self.menu_mru_list[i].action_command = value.into();
            self.menu_mru_list[i].visible = !value.is_empty();
        }
    }
    pub fn menu_file_action(&self, command: &str) -> Result<MenuTarget, MenuTarget> {
        match command {
            "Save" => Ok(MenuTarget::Save),
            "Save As..." => Ok(MenuTarget::SaveAs),
            "Close" => Ok(MenuTarget::Close),
            "Cancel" => Ok(MenuTarget::Cancel),
            _ => Err(MenuTarget::UnportedTarget(command.into())),
        }
    }
    pub fn menu_tools_action(&self, command: &str) -> MenuTarget {
        match command {
            FLATTEN_VOLUME_LABEL => MenuTarget::Tool(ToolType::FlattenVolume),
            GPU_TILT_TEST_LABEL => MenuTarget::Tool(ToolType::GpuTiltTest),
            ALIGN_FRAMES_LABEL => MenuTarget::Tool(ToolType::AlignFrames),
            _ => MenuTarget::UnportedTarget(command.into()),
        }
    }
    /// `menuHelpAction`: direct `imodqtassist` targets are described; process/dialog targets remain explicit.
    pub fn menu_help_action(&self, command: &str) -> MenuTarget {
        let guide = match command {
            "Tomography Guide" => Some("tomoguide.html#TOP"),
            "Imod Users Guide" => Some("guide.html#TOP"),
            "3dmod Users Guide" => Some("3dmodguide.html#TOP"),
            "Etomo Users Guide" => Some("UsingEtomo.html#TOP"),
            "Join Users Guide" => Some("tomojoin.html#TOP"),
            "PEET Users Guide" => Some("PEETmanual.html#TOP"),
            "Batch Interface Guide" => Some("batchGuide.html#TOP"),
            _ => None,
        };
        guide
            .map(MenuTarget::Guide)
            .unwrap_or_else(|| MenuTarget::UnportedTarget(command.into()))
    }
    pub fn is_menu_3dmod_startup_window(&self) -> bool {
        self.menu_3dmod_startup_window.selected
    }
    pub fn is_menu_3dmod_bin_by_2(&self) -> bool {
        self.menu_3dmod_bin_by_2.selected
    }
    pub fn set_menu_3dmod_startup_window(&mut self, value: bool) {
        self.menu_3dmod_startup_window.selected = value
    }
    pub fn set_menu_3dmod_bin_by_2(&mut self, value: bool) {
        self.menu_3dmod_bin_by_2.selected = value
    }
    pub fn do_click_file_exit(&self) -> MenuTarget {
        MenuTarget::Exit
    }
    pub fn set_enabled_log_window(&mut self, v: bool) {
        self.menu_log_window.enabled = v
    }
    pub fn set_enabled_new_tomogram(&mut self, v: bool) {
        self.menu_new_tomogram.enabled = v
    }
    pub fn set_enabled_new_join(&mut self, v: bool) {
        self.menu_new_join.enabled = v
    }
    pub fn set_enabled_new_generic_parallel(&mut self, v: bool) {
        self.menu_new_generic_parallel.enabled = v
    }
    pub fn set_enabled_new_anisotropic_diffusion(&mut self, v: bool) {
        self.menu_new_anisotropic_diffusion.enabled = v
    }
    pub fn set_enabled_new_batch_run_tomo(&mut self, v: bool) {
        self.menu_new_batch_run_tomo.enabled = v
    }
    pub fn set_enabled_new_peet(&mut self, v: bool) {
        self.menu_new_peet.enabled = v
    }
    pub fn set_enabled_new_serial_sections(&mut self, v: bool) {
        self.menu_serial_sections.enabled = v
    }
    pub fn is_menu_save_enabled(&self) -> bool {
        self.menu_save.enabled
    }
    pub fn equals(&self, item: &MenuItem, command: &str) -> bool {
        item.action_command == command
    }
    pub fn get_directive_file_type(&self, command: &str) -> Option<DirectiveFileType> {
        match command {
            "Save Scope Template" => Some(DirectiveFileType::Scope),
            "Save System Template" => Some(DirectiveFileType::System),
            "Save User Template" => Some(DirectiveFileType::User),
            "Export Batch Directive File" => Some(DirectiveFileType::Batch),
            _ => None,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mru_fills_and_hides_all_slots() {
        let mut m = EtomoMenu::get_instance(false);
        m.set_mru_file_labels(&["a.edf".into(), "".into(), "c.edf".into()]);
        assert!(m.menu_mru_list[0].visible);
        assert!(!m.menu_mru_list[1].visible);
        assert!(m.menu_mru_list[2].visible);
        assert!(!m.menu_mru_list[9].visible)
    }
    #[test]
    fn source_action_commands_match() {
        let m = EtomoMenu::get_instance(false);
        assert_eq!(m.menu_file_action("Save"), Ok(MenuTarget::Save));
        assert_eq!(
            m.menu_tools_action(FLATTEN_VOLUME_LABEL),
            MenuTarget::Tool(ToolType::FlattenVolume)
        );
        assert_eq!(
            m.get_directive_file_type("Save User Template"),
            Some(DirectiveFileType::User)
        );
    }
    #[test]
    fn manager_state_controls_source_items() {
        let mut m = EtomoMenu::get_instance(false);
        m.set_enabled(None);
        assert!(!m.is_menu_save_enabled());
        m.set_enabled(Some(ManagerMenuState {
            setup_done: true,
            can_change_param_file_name: true,
            dual_axis: true,
            can_save_directives: true,
        }));
        assert!(m.is_menu_save_enabled());
        assert!(m.menu_axis_a.enabled);
    }
}
