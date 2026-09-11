//! `IMOD/Etomo/src/etomo/ui/swing/ProcessControlPanel.java`.
//! Native `SimpleToggleButton`/`ColoredStateText` controls are represented by
//! their source-visible text, selection, color, and listener boundaries.
#![allow(dead_code)]
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
pub const TEXT_STATES: [&str; 3] = ["Not Started", "In Progress", "Complete"];
pub const COLOR_NOT_STARTED: (u8, u8, u8) = (191, 0, 0);
pub const COLOR_IN_PROGRESS: (u8, u8, u8) = (191, 0, 191);
pub const COLOR_COMPLETE: (u8, u8, u8) = (0, 153, 0);
pub const COLOR_STATE: [(u8, u8, u8); 3] = [COLOR_NOT_STARTED, COLOR_IN_PROGRESS, COLOR_COMPLETE];
pub struct ProcessControlPanel {
    pub command: String,
    pub dialog_type: DialogType,
    pub compact_display: bool,
    pub selected_state: usize,
    pub button_selected: bool,
    pub button_action_command: String,
    pub button_text: String,
    pub button_foreground: (u8, u8, u8),
    pub tooltip: Option<String>,
    pub action_listener_present: bool,
    pub mouse_listener_present: bool,
}
impl ProcessControlPanel {
    pub fn new(dialog_type: DialogType, compact_display: bool) -> Self {
        let command = if compact_display {
            dialog_type.get_compact_label()
        } else {
            dialog_type.to_string()
        };
        let mut panel = Self {
            command,
            dialog_type,
            compact_display,
            selected_state: 0,
            button_selected: false,
            button_action_command: dialog_type.to_string(),
            button_text: String::new(),
            button_foreground: COLOR_NOT_STARTED,
            tooltip: None,
            action_listener_present: false,
            mouse_listener_present: false,
        };
        panel.update_label();
        panel
    }
    pub fn get_command(&self) -> String {
        self.dialog_type.to_string()
    }
    pub fn get_dialog_type(&self) -> DialogType {
        self.dialog_type
    }
    pub fn set_button_action_listener(&mut self) {
        self.action_listener_present = true;
    }
    pub fn get_container(&self) -> bool {
        true
    }
    pub fn set_state(&mut self, state: ProcessState) {
        self.selected_state = match state {
            ProcessState::NotStarted => 0,
            ProcessState::InProgress => 1,
            ProcessState::Complete => 2,
        };
        self.update_label();
    }
    pub fn set_selected(&mut self, state: bool) {
        self.button_selected = state;
    }
    fn update_label(&mut self) {
        let state = if self.compact_display {
            ""
        } else {
            TEXT_STATES[self.selected_state]
        };
        self.button_text = if state.is_empty() {
            format!("<HTML><CENTER>{}</CENTER>", self.command)
        } else {
            format!("<HTML><CENTER>{}<br>{}</CENTER>", self.command, state)
        };
        self.button_foreground = COLOR_STATE[self.selected_state];
    }
    pub fn add_mouse_listener(&mut self) {
        self.mouse_listener_present = true;
    }
    pub fn set_tool_tip_text(&mut self, text: impl Into<String>) {
        self.tooltip = Some(text.into());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    #[test]
    fn state_controls_html_label_and_color() {
        let mut p = ProcessControlPanel::new(DialogType::SetupRecon, false);
        p.set_state(ProcessState::Complete);
        assert!(p.button_text.contains("Complete"));
        assert_eq!(p.button_foreground, COLOR_COMPLETE);
    }
    #[test]
    fn compact_display_omits_state() {
        let mut p = ProcessControlPanel::new(DialogType::SetupRecon, true);
        p.set_state(ProcessState::InProgress);
        assert!(!p.button_text.contains("In Progress"));
    }
}
