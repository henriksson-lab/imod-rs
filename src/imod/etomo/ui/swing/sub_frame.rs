//! Translation of `SubFrame.java`.
#![allow(dead_code)]
use super::etomo_frame::{ActionEvent, EtomoFrame, FrameType};
use crate::imod::etomo::r#type::axis_id::AxisID;
/// Java `SubFrame` retains its own `EtomoMenu`, but copies its visual menu
/// state from MainFrame exactly as the Swing constructor does.
#[derive(Clone, Debug)]
pub struct SubFrame {
    pub frame: EtomoFrame,
    pub busy_status_b: bool,
    pub status_bar: String,
    pub root_panel_name: String,
    pub location: (i32, i32),
}
impl SubFrame {
    pub fn new() -> Self {
        Self {
            frame: EtomoFrame::new(),
            busy_status_b: false,
            status_bar: String::new(),
            root_panel_name: "panel:sub-frame".into(),
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
    pub fn initialize(&mut self, title: String, mru_list: &[String]) {
        self.frame.initialize();
        self.frame.presentation.title = title;
        self.frame.set_mru_file_labels(mru_list);
        self.frame.presentation.visible = true;
    }
    pub fn get_frame_type(&self) -> FrameType {
        FrameType::Sub
    }
    pub fn process_window_event(&mut self, closing: bool) -> bool {
        closing
    }
    pub fn msg_busy_status_changed(&mut self, axis: AxisID, busy: bool) {
        if axis == AxisID::Second {
            self.busy_status_b = busy;
        }
    }
    pub fn set_main_panel(&mut self, title: String, status: String) {
        self.frame.presentation.title = title;
        self.status_bar = status;
    }
    pub fn menu_view_action(&mut self, event: &ActionEvent) -> Result<(), String> {
        self.frame.menu_view_action(event)
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.frame.presentation.visible = visible;
        if visible {
            self.set_axis();
        }
    }
    pub fn set_axis(&mut self) {
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
    #[test]
    fn only_axis_b_updates_busy_label() {
        let mut f = SubFrame::new();
        f.msg_busy_status_changed(AxisID::First, true);
        assert!(!f.busy_status_b);
        f.msg_busy_status_changed(AxisID::Second, true);
        assert!(f.busy_status_b);
    }
}
