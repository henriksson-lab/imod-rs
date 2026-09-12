//! `IMOD/Etomo/src/etomo/ui/swing/ProcessControlPanel.java`.
//! `SimpleToggleButton` and `ColoredStateText` retain their source-visible
//! control state at the native Swing boundary.
#![allow(dead_code)]
use super::simple_toggle_button::SimpleToggleButton;
use super::tooltip_formatter::TooltipFormatter;
use super::ui_utilities::Color;
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
    pub button_run: SimpleToggleButton,
    pub panel_root_tooltip: Option<String>,
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
            button_run: {
                let mut button = SimpleToggleButton::new();
                button.button.action_command = Some(dialog_type.to_string());
                button
            },
            panel_root_tooltip: None,
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
        self.button_run.button.action_listener_count += 1;
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
        self.button_run.button.selected = state;
    }
    fn update_label(&mut self) {
        let state = if self.compact_display {
            ""
        } else {
            TEXT_STATES[self.selected_state]
        };
        self.button_run.set_text(Some(&if state.is_empty() {
            format!("<HTML><CENTER>{}</CENTER>", self.command)
        } else {
            format!("<HTML><CENTER>{}<br>{}</CENTER>", self.command, state)
        }));
        let color = COLOR_STATE[self.selected_state];
        self.button_run.button.foreground = Some(Color {
            red: color.0 as i32,
            green: color.1 as i32,
            blue: color.2 as i32,
        });
    }
    pub fn add_mouse_listener(&mut self) {
        self.button_run.button.mouse_listener_count += 1;
    }
    pub fn set_tool_tip_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        let tooltip = TooltipFormatter::instance().format(Some(&text));
        self.panel_root_tooltip = tooltip.clone();
        self.button_run.button.tooltip = tooltip;
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
        assert!(
            p.button_run
                .button
                .text
                .as_deref()
                .unwrap()
                .contains("Complete")
        );
        assert_eq!(
            p.button_run.button.foreground,
            Some(Color {
                red: 0,
                green: 153,
                blue: 0,
            })
        );
    }
    #[test]
    fn compact_display_omits_state() {
        let mut p = ProcessControlPanel::new(DialogType::SetupRecon, true);
        p.set_state(ProcessState::InProgress);
        assert!(
            !p.button_run
                .button
                .text
                .as_deref()
                .unwrap()
                .contains("In Progress")
        );
    }
}
