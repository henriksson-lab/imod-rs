//! `IMOD/Etomo/src/etomo/ui/swing/AxisProgressPanel.java`.
//!
//! `ProgressPanel`, `SimpleButton`, and `BusyStatusMediator` remain their own
//! untranslated source units.  This module keeps exactly the state and call
//! ordering AxisProgressPanel contributes around those widget boundaries.
#![allow(dead_code)]

use super::progress_panel::ProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;

/// Java `KILL_BUTTON_LABEL`.
pub const KILL_BUTTON_LABEL: &str = "Kill Process";
/// Java `BusyStatusMediator.DELAY`, unavailable until that direct unit is
/// translated.  The delay itself is intentionally not emulated here: blocking
/// the Rust GUI event loop is not a representation of Swing's queued thread.
pub const BUSY_STATUS_MEDIATOR_DELAY_BOUNDARY: &str = "BusyStatusMediator.DELAY";

/// Fields of Java's final `AxisProgressPanel`.
pub struct AxisProgressPanel {
    pub pnl_root_visible: bool,
    pub button_kill_process_enabled: bool,
    pub button_kill_process_tooltip: Option<String>,
    pub manager: &'static dyn BaseManager,
    pub progress_panel: ProgressPanel,
    /// The exact messages passed to absent BusyStatusMediator.
    pub busy_status_kill_button_messages: Vec<(AxisID, bool)>,
    pub axis_id: AxisID,
    pub process_can_be_killed: bool,
    pub background: Option<String>,
}
impl AxisProgressPanel {
    /// `AxisProgressPanel(AxisID, BaseManager)`.
    fn new(axis_id: Option<AxisID>, manager: &'static dyn BaseManager) -> Self {
        Self {
            pnl_root_visible: true,
            button_kill_process_enabled: true,
            button_kill_process_tooltip: None,
            manager,
            progress_panel: ProgressPanel::get_instance(
                Some("No process"),
                manager,
                axis_id.unwrap_or(AxisID::Only),
            ),
            busy_status_kill_button_messages: vec![],
            axis_id: axis_id.unwrap_or(AxisID::Only),
            process_can_be_killed: true,
            background: None,
        }
    }
    /// `getInstance(AxisID, BaseManager)`.
    pub fn get_instance(axis_id: Option<AxisID>, manager: &'static dyn BaseManager) -> Self {
        let mut instance = Self::new(axis_id, manager);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }
    /// `correctAxisID(AxisType)`.
    pub fn correct_axis_id(&mut self, axis_type: AxisType) {
        if self.axis_id == AxisID::Second {
            return;
        }
        self.axis_id = if axis_type == AxisType::DualAxis {
            AxisID::First
        } else {
            AxisID::Only
        };
    }
    /// `isStopped()`.
    pub fn is_stopped(&self) -> bool {
        self.progress_panel.is_stopped()
    }
    /// `createPanel()`.
    fn create_panel(&mut self) {
        self.button_kill_process_enabled = false;
        self.busy_status_kill_button_messages
            .push((self.axis_id, false));
    }
    /// `getComponent()`; native JPanel is owned by the GUI presentation boundary.
    pub fn get_component(&self) -> bool {
        self.pnl_root_visible
    }
    /// `setVisible(boolean)`.
    pub fn set_visible(&mut self, input: bool) {
        self.pnl_root_visible = input;
        self.progress_panel.set_visible(input);
    }
    /// `setTooltips()`.
    fn set_tooltips(&mut self) {
        self.button_kill_process_tooltip = Some("Press to end the current process.".into());
    }
    /// `addListeners()`; `action_performed` is the direct Rust event endpoint.
    fn add_listeners(&mut self) {}
    /// `setBackground(Color)`; Color is a native widget type, retained as its CSS/RGB presentation string.
    pub fn set_background(&mut self, color: impl Into<String>) {
        let color = color.into();
        self.background = Some(color.clone());
        self.progress_panel.set_background(color);
    }
    /// `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self) {
        self.manager.kill(Some(self.axis_id));
    }
    /// `setProcessCanBeKilled(boolean)`.
    pub fn set_process_can_be_killed(&mut self, input: bool) {
        self.process_can_be_killed = input;
    }
    /// `setButtonKillProcessEnabled(boolean)`; BusyStatusMediator synchronization and
    /// delay are the explicit mediator boundary, while both message order and final
    /// button state match Java.
    fn set_button_kill_process_enabled(&mut self, enabled: bool) {
        if enabled {
            self.busy_status_kill_button_messages
                .push((self.axis_id, true));
        }
        self.button_kill_process_enabled = enabled;
        if !enabled {
            self.busy_status_kill_button_messages
                .push((self.axis_id, false));
        }
    }
    /// `setProgressBar(String, int, boolean)`.
    pub fn set_progress_bar(&mut self, label: &str, n_steps: i32, indeterminate_mode: bool) {
        self.progress_panel.set_label(label);
        self.progress_panel.set_minimum(0);
        self.progress_panel.set_maximum(n_steps, indeterminate_mode);
        self.progress_panel.set_value(0);
        self.set_button_kill_process_enabled(self.process_can_be_killed);
    }
    /// `setStaticProgressBar(String)`.
    pub fn set_static_progress_bar(&mut self, label: &str) {
        self.progress_panel.set_label(label);
    }
    /// `setProgressBarValue(int)`.
    pub fn set_progress_bar_value(&mut self, value: i32) {
        self.progress_panel.set_value(value);
    }
    /// `setProgressBarValue(int, StandardBarString, String, boolean)`.
    pub fn set_progress_bar_value_standard(
        &mut self,
        value: i32,
        standard: Option<StandardBarString>,
        bar: &str,
        _print: bool,
    ) {
        self.progress_panel
            .set_value_standard(value, standard, Some(bar), _print);
    }
    /// `setEmergencyMonitorBarString(StandardBarString, String)`.
    pub fn set_emergency_monitor_bar_string(&mut self, standard: StandardBarString, bar: &str) {
        self.progress_panel
            .set_emergency_monitor_bar_string(standard, bar);
    }
    /// `isProgressBarStopped(AxisID)`; Java ignores its AxisID argument.
    pub fn is_progress_bar_stopped(&self, _axis_id: AxisID) -> bool {
        self.progress_panel.is_progress_bar_stopped()
    }
    /// `startProgressBar(String, ProcessName)`.
    pub fn start_progress_bar(&mut self, label: &str, _process_name: Option<ProcessName>) {
        self.progress_panel.set_label(label);
        self.progress_panel.start();
        self.set_button_kill_process_enabled(self.process_can_be_killed);
    }
    /// `startProgressBar(int, String, boolean)`.
    pub fn start_progress_bar_value(&mut self, value: i32, bar: &str, print: bool) {
        self.progress_panel.start_indeterminate_mode();
        self.set_progress_bar_value_standard(value, None, bar, print);
        self.set_button_kill_process_enabled(self.process_can_be_killed);
    }
    /// `setProgressBar(String, boolean)`.
    pub fn set_progress_bar_indeterminate(&mut self, label: &str, indeterminate_mode: bool) {
        self.progress_panel.set_label(label);
        if indeterminate_mode {
            self.progress_panel.start_indeterminate_mode();
        }
    }
    /// `stopProgressBar(ProcessEndState, String)`. ProcessEndState is not yet a Rust
    /// unit; its source only crosses to ProgressPanel, so its exact textual state is retained.
    pub fn stop_progress_bar(&mut self, process_end_state: &str, status: Option<&str>) {
        self.progress_panel.stop(Some(process_end_state), status);
        self.process_can_be_killed = true;
        self.set_button_kill_process_enabled(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn panel() -> AxisProgressPanel {
        AxisProgressPanel::get_instance(
            Some(AxisID::First),
            DirectiveEditorManager::new(None, None, None, None),
        )
    }
    #[test]
    fn button_messages_preserve_source_order() {
        let mut p = panel();
        assert_eq!(
            p.busy_status_kill_button_messages,
            vec![(AxisID::First, false)]
        );
        p.set_progress_bar("align", 3, false);
        assert_eq!(
            p.busy_status_kill_button_messages.last(),
            Some(&(AxisID::First, true))
        );
        p.stop_progress_bar("DONE", None);
        assert_eq!(
            p.busy_status_kill_button_messages.last(),
            Some(&(AxisID::First, false))
        );
    }
    #[test]
    fn second_axis_is_never_corrected() {
        let mut p = AxisProgressPanel::get_instance(
            Some(AxisID::Second),
            DirectiveEditorManager::new(None, None, None, None),
        );
        p.correct_axis_id(AxisType::SingleAxis);
        assert_eq!(p.axis_id, AxisID::Second);
    }
    #[test]
    fn progress_value_and_label_reach_the_direct_widget_boundary() {
        let mut p = panel();
        p.set_progress_bar("x", 4, false);
        p.set_progress_bar_value(2);
        assert_eq!(p.progress_panel.task_label, "x");
        assert_eq!(p.progress_panel.value, 2);
    }
}
