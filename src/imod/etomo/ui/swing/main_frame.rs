//! Translation of `MainFrame.java` using the existing `WindowSwitch` state.
#![allow(dead_code)]
use super::etomo_frame::{ActionEvent, EtomoFrame, FrameType};
use super::sub_frame::{SubFrame, SubFrameMainFrameBoundary};
use super::window_switch::{WindowMainPanel, WindowManager, WindowSwitch};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::unique_key::UniqueKey;

pub const ETOMO_TITLE: &str = "Etomo";
pub const NAME: &str = "main-frame";
/// MainPanel's source-owned, frame-visible state.  Its actual process panels are
/// deliberately not recreated here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MainPanelState {
    pub axis_a: bool,
    pub axis_b: bool,
    pub dual_axis: bool,
    pub status_bar_text: String,
    pub display_state_saved: bool,
}
impl WindowMainPanel for MainPanelState {
    fn save_display_state(&mut self) {
        self.display_state_saved = true;
    }
}
pub struct MainPanelManager {
    pub panel: Option<MainPanelState>,
}
impl WindowManager<MainPanelState> for MainPanelManager {
    fn get_main_panel(&mut self) -> Option<MainPanelState> {
        self.panel.take()
    }
}
/// Java `MainFrame` fields. `WindowSwitch` is the pre-existing authoritative
/// tabs/window-menu implementation rather than a parallel frame abstraction.
pub struct MainFrame {
    pub frame: EtomoFrame,
    pub window_switch: WindowSwitch<MainPanelState>,
    pub root_panel_name: String,
    pub title: String,
    pub mru_list: Vec<String>,
    pub registered: bool,
    pub main_panel: Option<MainPanelState>,
    pub sub_frame: Option<SubFrame>,
}
impl MainFrame {
    pub fn new() -> Self {
        let mut frame = EtomoFrame::new();
        frame.main = true;
        frame.main_frame_registered = true;
        frame.initialize();
        Self {
            frame,
            window_switch: WindowSwitch::new(),
            root_panel_name: format!("panel:{}", NAME),
            title: ETOMO_TITLE.into(),
            mru_list: vec![],
            registered: true,
            main_panel: None,
            sub_frame: None,
        }
    }
    pub fn register(&mut self) -> Result<(), String> {
        if self.registered {
            return Err("Only one instance of MainFrame is allowed.".into());
        }
        self.registered = true;
        self.frame.main = true;
        Ok(())
    }
    pub fn get_frame_type(&self) -> FrameType {
        FrameType::Main
    }
    pub fn update_frame(&mut self, state: Option<super::etomo_menu::ManagerMenuState>) {
        self.frame.set_enabled(state);
    }
    /// `setCurrentManager`; BaseManager's MainPanel declared target is not yet a
    /// Rust value, so callers supply the directly equivalent panel state.
    pub fn set_current_manager(
        &mut self,
        panel: Option<MainPanelState>,
        manager_key: Option<&UniqueKey>,
        new_window: bool,
        _manager_stamp: bool,
    ) {
        self.frame.current_manager_present = panel.is_some();
        self.main_panel = panel;
        self.frame.main_panel_present = self.main_panel.is_some();
        self.frame.main_panel_axis_type = self.main_panel.as_ref().map(|panel| {
            if panel.dual_axis {
                crate::imod::etomo::r#type::axis_type::AxisType::DualAxis
            } else {
                crate::imod::etomo::r#type::axis_type::AxisType::SingleAxis
            }
        });
        self.frame.main_panel_showing_setup = false;
        self.frame.main_panel_showing_axis_a =
            self.main_panel.as_ref().is_some_and(|panel| panel.axis_a);
        self.frame.main_panel_showing_both_axis = self
            .main_panel
            .as_ref()
            .is_some_and(|panel| panel.axis_a && panel.axis_b);
        if self.main_panel.is_none() {
            self.title = ETOMO_TITLE.into();
            self.hide_axis_b();
            return;
        }
        self.title = format!("dataset - {ETOMO_TITLE}");
        if let Some(sub_frame) = &mut self.sub_frame {
            sub_frame.frame.current_manager_present = self.frame.current_manager_present;
            sub_frame.set_main_panel(
                format!("B Axis - {} ", self.title),
                None,
                self.main_panel.clone(),
            );
        }
        if let Some(key) = manager_key {
            let _ = self.window_switch.get_panel(Some(key));
            self.select_window_menu_item(key, new_window);
        }
        if new_window {
            self.show_axis_a();
        }
    }
    pub fn show_hide_log(&mut self) -> Result<(), String> {
        if self.frame.current_manager_present {
            Err("BaseManager.showHideLog UI target is not yet translated".into())
        } else {
            Ok(())
        }
    }
    pub fn get_main_panel(&self) -> Option<&MainPanelState> {
        self.main_panel.as_ref()
    }
    pub fn set_current_manager_default(
        &mut self,
        panel: Option<MainPanelState>,
        key: Option<&UniqueKey>,
    ) {
        self.set_current_manager(panel, key, false, true);
    }
    pub fn set_mru_file_labels(&mut self, list: &[String]) {
        self.mru_list = list.to_vec();
        self.frame.set_mru_file_labels(list);
        if let Some(sub) = &mut self.sub_frame {
            sub.frame.set_mru_file_labels(list);
        }
    }
    pub fn pop_up_context_menu(&self) -> Result<(), String> {
        Err("ContextPopup.java is not yet translated".into())
    }
    pub fn add_window(
        &mut self,
        manager: &mut MainPanelManager,
        axis_id: AxisID,
        key: Option<UniqueKey>,
    ) {
        self.window_switch.add(manager, axis_id, key);
    }
    pub fn remove_window(&mut self, key: Option<&UniqueKey>) {
        self.window_switch.remove(key);
    }
    pub fn rename_window(&mut self, old: Option<&UniqueKey>, new: Option<UniqueKey>) {
        self.window_switch.rename(old, new);
    }
    pub fn select_window_menu_item_default(&mut self, key: &UniqueKey) {
        self.select_window_menu_item(key, false);
    }
    pub fn select_window_menu_item(&mut self, key: &UniqueKey, new_window: bool) {
        self.window_switch.select_window(Some(key), new_window);
    }
    pub fn menu_view_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        match event.action_command.as_str() {
            "Axis A" => {
                self.show_axis_a();
                Ok(())
            }
            "Axis B" => {
                self.show_axis_b();
                Ok(())
            }
            "Both Axes" => {
                self.show_both_axis();
                Ok(())
            }
            "Show/Hide Log Window" => self.show_hide_log(),
            _ => self.frame.menu_view_action(event),
        }
    }
    /// Direct endpoint for `UIHarness`'s Slint `menu-command` callback.
    /// Slint uses stable path commands while Swing's `ActionEvent` carries its
    /// visible label, so this is the one source-owned conversion point.
    pub fn dispatch_harness_command(
        &mut self,
        command: &str,
    ) -> Result<Option<super::etomo_menu::MenuTarget>, String> {
        let source_command = match command {
            "view.axis-a" => {
                self.show_axis_a();
                return Ok(None);
            }
            "view.axis-b" => {
                self.show_axis_b();
                return Ok(None);
            }
            "view.both-axes" => {
                self.show_both_axis();
                return Ok(None);
            }
            "view.log-window" => {
                self.show_hide_log()?;
                return Ok(None);
            }
            "view.fit-window" => {
                self.frame.pack();
                return Ok(None);
            }
            "file.save" => "Save",
            "file.save-as" => "Save As...",
            "file.close" => "Close",
            "file.cancel" => "Cancel",
            "file.exit" => "Exit",
            "file.open" => "Open...",
            "file.tomosnapshot" => "Run Tomosnapshot",
            "file.new.tomogram" => super::etomo_menu::RECON_LABEL,
            "file.new.join" => super::etomo_menu::JOIN_LABEL,
            "file.new.peet" => super::etomo_menu::PEET_LABEL,
            "file.new.serial-sections" => super::etomo_menu::SERIAL_SECTIONS_LABEL,
            "file.new.anisotropic-diffusion" => super::etomo_menu::NAD_LABEL,
            "file.new.generic-parallel" => super::etomo_menu::GENERIC_LABEL,
            "file.new.batch-runtomo" => super::etomo_menu::BATCH_RUN_TOMO_LABEL,
            "tools.flatten-volume" => super::etomo_menu::FLATTEN_VOLUME_LABEL,
            "tools.gpu-tilt-test" => super::etomo_menu::GPU_TILT_TEST_LABEL,
            "tools.align-frames" => super::etomo_menu::ALIGN_FRAMES_LABEL,
            "help.tomography-guide" => "Tomography Guide",
            "help.imod-guide" => "Imod Users Guide",
            "help.3dmod-guide" => "3dmod Users Guide",
            "help.etomo-guide" => "Etomo Users Guide",
            "help.join-guide" => "Join Users Guide",
            "help.peet-guide" => "PEET Users Guide",
            "help.batch-guide" => "Batch Interface Guide",
            "options.settings" => {
                return self
                    .frame
                    .menu_options_action(&ActionEvent::new("Settings"))
                    .map(|_| None);
            }
            _ => return Err(format!("Unknown UIHarness menu command: {command}")),
        };
        let event = ActionEvent::new(source_command);
        if command.starts_with("tools.") {
            Ok(Some(self.frame.menu_tools_action(&event)))
        } else if command.starts_with("help.") {
            Ok(Some(self.frame.menu_help_action(&event)))
        } else {
            Ok(Some(self.frame.menu_file_action(&event)))
        }
    }
    pub fn set_title(&mut self, axis: Option<AxisID>) {
        self.frame.presentation.title = if self
            .main_panel
            .as_ref()
            .is_some_and(|panel| panel.dual_axis)
        {
            match axis {
                Some(AxisID::First) => format!("A Axis - {}", self.title),
                Some(AxisID::Second) => format!("B Axis - {}", self.title),
                _ => self.title.clone(),
            }
        } else {
            self.title.clone()
        };
    }
    pub fn hide_axis_b(&mut self) {
        self.set_title(None);
        if let Some(sub) = &mut self.sub_frame {
            sub.set_visible(false);
        }
        self.frame.pack();
    }
    pub fn show_axis_a(&mut self) {
        self.set_title(Some(AxisID::First));
        if let Some(panel) = &mut self.main_panel {
            panel.axis_a = true;
            panel.axis_b = false;
        }
        self.frame.main_panel_showing_axis_a = true;
        self.frame.main_panel_showing_both_axis = false;
        if let Some(sub) = &mut self.sub_frame {
            sub.set_visible(false);
        }
        self.frame.axis_id = Some(AxisID::First);
        self.frame.pack();
    }
    pub fn show_axis_b(&mut self) {
        self.set_title(Some(AxisID::Second));
        if let Some(panel) = &mut self.main_panel {
            panel.axis_a = false;
            panel.axis_b = true;
        }
        self.frame.main_panel_showing_axis_a = false;
        self.frame.main_panel_showing_both_axis = false;
        if let Some(sub) = &mut self.sub_frame {
            sub.set_visible(false);
        }
        self.frame.axis_id = Some(AxisID::Second);
        self.frame.pack();
    }
    pub fn show_both_axis(&mut self) {
        self.set_title(Some(AxisID::First));
        if let Some(panel) = &mut self.main_panel {
            panel.axis_a = true;
            panel.axis_b = true;
        }
        self.frame.main_panel_showing_axis_a = true;
        self.frame.main_panel_showing_both_axis = true;
        if self.sub_frame.is_none() {
            self.sub_frame = Some(SubFrame::new());
            self.frame.sub_frame_registered = true;
        }
        let main_panel = self.main_panel.clone();
        let mru_list = self.mru_list.clone();
        let main_menu = self.frame.menu.clone();
        let current_manager_present = self.frame.current_manager_present;
        if let Some(sub) = &mut self.sub_frame {
            sub.frame.current_manager_present = current_manager_present;
            sub.initialize(
                format!("B Axis - {} ", self.title),
                None,
                main_panel,
                &mru_list,
                &main_menu,
            );
            sub.set_visible(true);
        }
        self.frame.axis_id = Some(AxisID::First);
        self.frame.pack();
    }
    pub fn process_window_event(
        &mut self,
        closing: bool,
        test: bool,
    ) -> Option<super::etomo_menu::MenuTarget> {
        if closing && !test {
            Some(self.frame.menu.do_click_file_exit())
        } else {
            None
        }
    }
}

impl SubFrameMainFrameBoundary for MainFrame {
    fn show_axis_a(&mut self) {
        Self::show_axis_a(self);
    }

    fn menu_view_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        Self::menu_view_action(self, event)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn axis_menu_routes_to_source_owned_presentation() {
        let mut f = MainFrame::new();
        f.main_panel = Some(MainPanelState {
            dual_axis: true,
            ..Default::default()
        });
        f.menu_view_action(&ActionEvent::new("Both Axes")).unwrap();
        assert!(f.main_panel.as_ref().unwrap().axis_a);
        assert!(f.main_panel.as_ref().unwrap().axis_b);
        assert!(f.sub_frame.as_ref().unwrap().frame.presentation.visible);
    }
    #[test]
    fn slint_command_routes_to_existing_menu_target() {
        let mut frame = MainFrame::new();
        assert_eq!(
            frame.dispatch_harness_command("help.imod-guide").unwrap(),
            Some(super::super::etomo_menu::MenuTarget::Guide(
                "guide.html#TOP"
            ))
        );
    }
}
