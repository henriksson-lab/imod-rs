//! Translation of `SubFrame.java`.
#![allow(dead_code)]
use super::etomo_frame::{ActionEvent, EtomoFrame, FrameType};
use super::etomo_menu::EtomoMenu;
use super::main_frame::MainPanelState;
use crate::imod::etomo::r#type::axis_id::AxisID;

pub trait SubFrameMainFrameBoundary {
    fn show_axis_a(&mut self);
    fn menu_view_action(&mut self, event: &ActionEvent) -> Result<(), String>;
}

pub trait SubFrameManagerBoundary {
    fn remove_busy_status_listener(&mut self);
    fn add_busy_status_listener(&mut self);
}
/// Java `SubFrame` retains its own `EtomoMenu`, but copies its visual menu
/// state from MainFrame exactly as the Swing constructor does.
pub struct SubFrame {
    pub frame: EtomoFrame,
    pub pnl_status_present: bool,
    pub pnl_busy_status_present: bool,
    pub busy_status_b: bool,
    pub main_panel: Option<MainPanelState>,
    pub current_manager_listener_registered: bool,
    pub status_bar: String,
    pub root_panel_name: String,
    pub root_panel_axis_present: bool,
    pub root_panel_status_present: bool,
    pub root_panel_validated: bool,
    pub location: (i32, i32),
}
impl SubFrame {
    pub fn new() -> Self {
        let mut frame = EtomoFrame::new();
        frame.main_frame_registered = true;
        frame.sub_frame_registered = true;
        Self {
            frame,
            pnl_status_present: true,
            pnl_busy_status_present: true,
            busy_status_b: false,
            main_panel: None,
            current_manager_listener_registered: false,
            status_bar: String::new(),
            root_panel_name: "panel:sub-frame".into(),
            root_panel_axis_present: false,
            root_panel_status_present: false,
            root_panel_validated: false,
            location: (0, 0),
        }
    }
    pub fn register(&mut self) -> Result<(), String> {
        if self.frame.main {
            return Err("Only one instance of SubFrame is allowed.".into());
        }
        self.frame.main = false;
        Ok(())
    }
    pub fn initialize(
        &mut self,
        title: String,
        mut current_manager: Option<&mut dyn SubFrameManagerBoundary>,
        main_panel: Option<MainPanelState>,
        mru_list: &[String],
        main_menu: &EtomoMenu,
    ) {
        self.frame.initialize();
        if self.current_manager_listener_registered {
            if let Some(current_manager) = current_manager.as_deref_mut() {
                current_manager.remove_busy_status_listener();
            }
        }
        self.current_manager_listener_registered = false;
        if let Some(current_manager) = current_manager {
            current_manager.add_busy_status_listener();
            self.current_manager_listener_registered = true;
            self.frame.current_manager_present = true;
        }
        self.frame.presentation.title = title;
        self.main_panel = main_panel;
        self.frame.main_panel_present = self.main_panel.is_some();
        self.status_bar = self
            .main_panel
            .as_ref()
            .map_or_else(String::new, |main_panel| main_panel.status_bar_text.clone());
        self.frame.menu.set_enabled_from(main_menu);
        self.frame.set_mru_file_labels(mru_list);
        self.frame
            .menu
            .set_menu_3dmod_startup_window(main_menu.is_menu_3dmod_startup_window());
        self.frame
            .menu
            .set_menu_3dmod_bin_by_2(main_menu.is_menu_3dmod_bin_by_2());
        self.set_visible(true);
    }
    pub fn get_frame_type(&self) -> FrameType {
        FrameType::Sub
    }
    pub fn process_window_event(
        &mut self,
        closing: bool,
        main_frame: &mut dyn SubFrameMainFrameBoundary,
    ) {
        if closing {
            main_frame.show_axis_a();
        }
    }
    pub fn msg_busy_status_changed(&mut self, axis: AxisID, busy: bool) {
        self.run(axis, busy);
    }
    /// Rust UI-dispatch form of `SetBusyStatus::run`.
    pub fn run(&mut self, axis: AxisID, enabled: bool) {
        if axis == AxisID::Second {
            self.busy_status_b = enabled;
        }
    }
    pub fn set_main_panel(
        &mut self,
        title: String,
        mut current_manager: Option<&mut dyn SubFrameManagerBoundary>,
        main_panel: Option<MainPanelState>,
    ) {
        if self.current_manager_listener_registered {
            if let Some(current_manager) = current_manager.as_deref_mut() {
                current_manager.remove_busy_status_listener();
            }
        }
        self.current_manager_listener_registered = false;
        if let Some(current_manager) = current_manager {
            current_manager.add_busy_status_listener();
            self.current_manager_listener_registered = true;
            self.frame.current_manager_present = true;
        }
        self.main_panel = main_panel;
        self.frame.main_panel_present = self.main_panel.is_some();
        self.frame.presentation.title = title;
        self.status_bar = self
            .main_panel
            .as_ref()
            .map_or_else(String::new, |main_panel| main_panel.status_bar_text.clone());
    }
    pub fn menu_view_action(
        &mut self,
        event: &ActionEvent,
        main_frame: &mut dyn SubFrameMainFrameBoundary,
    ) -> Result<(), String> {
        if event.action_command == self.frame.menu.menu_axis_a.action_command
            || event.action_command == self.frame.menu.menu_axis_b.action_command
            || event.action_command == self.frame.menu.menu_axis_both.action_command
            || event.action_command == self.frame.menu.menu_log_window.action_command
        {
            return main_frame.menu_view_action(event);
        }
        self.frame.menu_view_action(event)
    }
    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.set_axis();
        }
        self.frame.presentation.visible = visible;
    }
    pub fn set_axis(&mut self) {
        self.root_panel_axis_present = self
            .main_panel
            .as_ref()
            .is_some_and(|main_panel| main_panel.axis_a && main_panel.axis_b);
        self.root_panel_status_present = true;
        self.root_panel_validated = true;
        self.frame.axis_id = Some(AxisID::Second);
    }
    pub fn move_sub_frame(
        &mut self,
        main_location: (i32, i32),
        main_width: i32,
        device: (i32, i32, i32, i32),
    ) {
        if !self.frame.presentation.visible {
            return;
        }
        let mut x = device.0 + main_location.0 + main_width;
        if x > device.0 + device.2 {
            x = (device.0 + device.2) / 2;
        }
        self.location = (x, device.1 + main_location.1);
    }
}
impl Default for SubFrame {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct ManagerMock {
        calls: Vec<&'static str>,
    }

    impl SubFrameManagerBoundary for ManagerMock {
        fn remove_busy_status_listener(&mut self) {
            self.calls.push("remove");
        }

        fn add_busy_status_listener(&mut self) {
            self.calls.push("add");
        }
    }

    #[derive(Default)]
    struct MainFrameMock {
        calls: Vec<&'static str>,
    }

    impl SubFrameMainFrameBoundary for MainFrameMock {
        fn show_axis_a(&mut self) {
            self.calls.push("show-axis-a");
        }

        fn menu_view_action(&mut self, _event: &ActionEvent) -> Result<(), String> {
            self.calls.push("menu-view-action");
            Ok(())
        }
    }

    #[test]
    fn only_axis_b_updates_busy_label() {
        let mut f = SubFrame::new();
        f.msg_busy_status_changed(AxisID::First, true);
        assert!(!f.busy_status_b);
        f.msg_busy_status_changed(AxisID::Second, true);
        assert!(f.busy_status_b);
    }

    #[test]
    fn source_initialization_panel_listener_and_main_frame_routes_are_preserved() {
        let mut frame = SubFrame::new();
        let mut manager = ManagerMock::default();
        let main_menu = EtomoMenu::get_instance(false);
        let panel = MainPanelState {
            axis_a: true,
            axis_b: true,
            dual_axis: true,
            status_bar_text: "busy".into(),
            ..Default::default()
        };

        frame.initialize(
            "B Axis - dataset".into(),
            Some(&mut manager),
            Some(panel.clone()),
            &["dataset.edf".into()],
            &main_menu,
        );

        assert_eq!(manager.calls, ["add"]);
        assert!(frame.frame.presentation.visible);
        assert!(frame.root_panel_axis_present);
        assert!(frame.root_panel_status_present);
        assert!(frame.root_panel_validated);
        assert_eq!(frame.status_bar, "busy");

        let mut main_frame = MainFrameMock::default();
        frame
            .menu_view_action(&ActionEvent::new("Axis A"), &mut main_frame)
            .unwrap();
        frame.process_window_event(true, &mut main_frame);
        assert_eq!(main_frame.calls, ["menu-view-action", "show-axis-a"]);

        frame.set_main_panel(
            "B Axis - replacement".into(),
            Some(&mut manager),
            Some(panel),
        );
        assert_eq!(manager.calls, ["add", "remove", "add"]);
    }

    #[test]
    fn source_move_sub_frame_returns_when_hidden_and_wraps_at_device_edge() {
        let mut frame = SubFrame::new();
        frame.move_sub_frame((15, 7), 10, (100, 200, 20, 30));
        assert_eq!(frame.location, (0, 0));

        frame.set_visible(true);
        frame.move_sub_frame((15, 7), 10, (100, 200, 20, 30));
        assert_eq!(frame.location, (60, 207));
    }
}
