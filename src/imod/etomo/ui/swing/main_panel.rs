//! `IMOD/Etomo/src/etomo/ui/swing/MainPanel.java`.
//!
//! MainPanel is abstract.  Its concrete `AxisProcessPanel`,
//! `AxisProgressPanel`, `ScrollPanel`, and `ParallelPanel` targets remain
//! explicit source-unit boundaries rather than invented replacement widgets.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::parallel_panel::ParallelPanel;
use super::scroll_panel::ScrollPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use std::fmt::Display;
use std::path::Path;

pub const STATUS_BAR_EMPTY_TITLE: &str = "No data set loaded";
pub const STATUS_BAR_BASE_TITLE: &str = "Data file: ";
pub const ESTIMATED_MENU_HEIGHT: i32 = 60;
pub const EXTRA_SCREEN_WIDTH_MULTIPLIER: i32 = 2;

/// Java private static `StateVariable<V>`.
#[derive(Clone, Debug)]
pub struct StateVariable<V: Clone + PartialEq + Display> {
    pub printed: bool,
    pub cur_value: Option<V>,
    pub changed: bool,
}
impl<V: Clone + PartialEq + Display> StateVariable<V> {
    pub fn new() -> Self {
        Self {
            printed: false,
            cur_value: None,
            changed: false,
        }
    }
    pub fn set(&mut self, value: Option<V>) {
        self.set_print(value, false);
    }
    pub fn set_print(&mut self, value: Option<V>, print: bool) {
        self.changed = !self.printed || self.cur_value != value;
        if self.changed && print {
            println!(
                "MainPanel:StateVariable:set:printed:{},curValue:{},newValue:{}",
                self.printed,
                self.cur_value
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "null".into()),
                value
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "null".into())
            );
        }
        self.cur_value = value;
    }
    pub fn is_changed(&self) -> bool {
        self.changed
    }
    pub fn to_string_java(&mut self) -> Option<String> {
        self.printed = true;
        self.cur_value.as_ref().map(ToString::to_string)
    }
}
impl<V: Clone + PartialEq + Display> Default for StateVariable<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Java private static `ProgressPanelState`; values are kept even before the
/// concrete AxisProgressPanel source unit exists.
#[derive(Clone, Debug)]
pub struct ProgressPanelState {
    pub diagnostics: bool,
    pub killable: StateVariable<bool>,
    pub label: StateVariable<String>,
    pub value: StateVariable<i32>,
    pub n_steps: StateVariable<i32>,
    pub pause_enabled: StateVariable<bool>,
    pub bar_string: StateVariable<String>,
    pub stopped: StateVariable<bool>,
    pub process_name: StateVariable<ProcessName>,
    pub process_end_state: StateVariable<String>,
    pub status_string: StateVariable<String>,
}
impl ProgressPanelState {
    pub fn new() -> Self {
        Self {
            diagnostics: false,
            killable: StateVariable::new(),
            label: StateVariable::new(),
            value: StateVariable::new(),
            n_steps: StateVariable::new(),
            pause_enabled: StateVariable::new(),
            bar_string: StateVariable::new(),
            stopped: StateVariable::new(),
            process_name: StateVariable::new(),
            process_end_state: StateVariable::new(),
            status_string: StateVariable::new(),
        }
    }
    pub fn print_killable_state(&mut self, descr: &str, _dump_stack: bool, value: bool) -> bool {
        self.killable.set(Some(value));
        self.print_changed(descr, self.killable.is_changed())
    }
    pub fn print_state_steps(
        &mut self,
        descr: &str,
        _dump_stack: bool,
        label: &str,
        steps: i32,
        pause: bool,
    ) -> bool {
        self.label.set(Some(label.into()));
        self.n_steps.set(Some(steps));
        self.pause_enabled.set(Some(pause));
        self.print_changed(
            descr,
            self.label.is_changed() || self.n_steps.is_changed() || self.pause_enabled.is_changed(),
        )
    }
    pub fn print_state_label(&mut self, descr: &str, _dump_stack: bool, label: &str) -> bool {
        self.label.set(Some(label.into()));
        self.print_changed(descr, self.label.is_changed())
    }
    pub fn print_state_value(&mut self, descr: &str, _dump_stack: bool, value: i32) -> bool {
        self.value.set(Some(value));
        self.print_changed(descr, self.value.is_changed())
    }
    pub fn print_state_value_string(
        &mut self,
        descr: &str,
        _dump_stack: bool,
        value: i32,
        bar: &str,
    ) -> bool {
        self.value.set_print(Some(value), false);
        self.bar_string.set_print(Some(bar.into()), false);
        self.print_changed(
            descr,
            self.value.is_changed() || self.bar_string.is_changed(),
        )
    }
    pub fn print_stopped_state(&mut self, descr: &str, _dump_stack: bool, stopped: bool) -> bool {
        self.stopped.set(Some(stopped));
        self.print_changed(descr, self.stopped.is_changed())
    }
    pub fn print_state_process(
        &mut self,
        descr: &str,
        _dump_stack: bool,
        label: &str,
        process: Option<ProcessName>,
    ) -> bool {
        self.label.set(Some(label.into()));
        self.process_name.set(process);
        self.print_changed(
            descr,
            self.label.is_changed() || self.process_name.is_changed(),
        )
    }
    pub fn print_state_end(&mut self, descr: &str, _dump_stack: bool, end: &str) -> bool {
        self.process_end_state.set(Some(end.into()));
        self.print_changed(descr, self.process_end_state.is_changed())
    }
    pub fn print_state_end_status(
        &mut self,
        descr: &str,
        _dump_stack: bool,
        end: &str,
        status: Option<&str>,
    ) -> bool {
        self.process_end_state.set(Some(end.into()));
        self.status_string.set(status.map(str::to_string));
        self.print_changed(
            descr,
            self.process_end_state.is_changed() || self.status_string.is_changed(),
        )
    }
    fn print_changed(&self, descr: &str, changed: bool) -> bool {
        if self.diagnostics && changed {
            println!("{descr}");
            true
        } else {
            false
        }
    }
}
impl Default for ProgressPanelState {
    fn default() -> Self {
        Self::new()
    }
}

/// Java's abstract declarations.  Their direct concrete panel return types are
/// intentionally not replaced by an abstraction layer.
pub trait MainPanelActions {
    fn create_axis_panel_a(&mut self, axis_id: AxisID) -> Result<(), String>;
    fn create_axis_panel_b(&mut self) -> Result<(), String>;
    fn reset_axis_panels(&mut self) -> Result<(), String>;
    fn add_axis_panel_a(&mut self) -> Result<(), String>;
    fn add_axis_panel_b(&mut self) -> Result<(), String>;
    fn is_axis_panel_a_null(&self) -> bool;
    fn is_axis_panel_b_null(&self) -> bool;
    fn hide_axis_panel_a(&mut self) -> bool;
    fn hide_axis_panel_b(&mut self) -> bool;
    fn show_axis_panel_a(&mut self) -> Result<(), String>;
    fn show_axis_panel_b(&mut self) -> Result<(), String>;
    fn map_base_axis_process_panel(&mut self, axis_id: AxisID) -> Result<(), String>;
    fn get_data_file_filter(&self) -> Result<(), String>;
    fn save_display_state(&mut self);
    fn get_axis_panel_a(&mut self) -> Result<(), String>;
    fn get_axis_panel_b(&mut self) -> Result<(), String>;
    fn set_state(&mut self, axis_id: AxisID) -> Result<(), String>;
    fn map_axis_progress_panel(&mut self, axis_id: AxisID) -> Result<(), String>;
}

/// Java MainPanel fields.  A `bool` for a child panel records whether Java has
/// instantiated it; method calls on the child are their named source boundary.
pub struct MainPanel {
    pub status_bar: String,
    pub panel_center: Vec<AxisID>,
    pub progress_panel_state_a: ProgressPanelState,
    pub scroll_a: Option<ScrollPanel>,
    pub scroll_pane_a: Option<i32>,
    pub scroll_b: Option<ScrollPanel>,
    pub scroll_pane_b: Option<i32>,
    pub manager: &'static dyn BaseManager,
    pub axis_progress_panel_a: Option<AxisProgressPanel>,
    pub axis_progress_panel_b: Option<AxisProgressPanel>,
    /// Concrete process widgets are supplied by the native frontend for each axis.
    pub axis_process_panel_a: Option<AxisProcessPanel>,
    pub axis_process_panel_b: Option<AxisProcessPanel>,
    pub showing_both_axis: bool,
    pub showing_axis_a: bool,
    pub showing_setup: bool,
    pub axis_type: AxisType,
    pub busy_status_a: bool,
    pub busy_status_b: Option<bool>,
    pub progress_panel_state_b: Option<ProgressPanelState>,
    pub debug: bool,
    pub vertical_scroll_a: Option<i32>,
    pub vertical_scroll_b: Option<i32>,
    pub vertical_scroll_always: bool,
    pub repaint_count: u64,
}
impl MainPanel {
    /// `MainPanel(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Self {
        manager.add_busy_status_listener(None);
        Self {
            status_bar: STATUS_BAR_EMPTY_TITLE.into(),
            panel_center: vec![],
            progress_panel_state_a: ProgressPanelState::new(),
            scroll_a: None,
            scroll_pane_a: None,
            scroll_b: None,
            scroll_pane_b: None,
            manager,
            axis_progress_panel_a: None,
            axis_progress_panel_b: None,
            axis_process_panel_a: None,
            axis_process_panel_b: None,
            showing_both_axis: false,
            showing_axis_a: true,
            showing_setup: false,
            axis_type: AxisType::NotSet,
            busy_status_a: false,
            busy_status_b: None,
            progress_panel_state_b: None,
            debug: false,
            vertical_scroll_a: None,
            vertical_scroll_b: None,
            vertical_scroll_always: false,
            repaint_count: 0,
        }
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    pub fn create_status_border_layout() -> i32 {
        2
    }
    pub fn get_status_bar_text(&self) -> &str {
        &self.status_bar
    }
    pub fn get_status(&self) -> String {
        if self.status_bar == STATUS_BAR_EMPTY_TITLE {
            String::new()
        } else {
            self.status_bar
                .strip_prefix(STATUS_BAR_BASE_TITLE)
                .unwrap_or(&self.status_bar)
                .into()
        }
    }
    pub fn repaint(&mut self) {
        self.repaint_count += 1;
    }
    pub fn get_scroll_a(&self) -> Option<ScrollPanel> {
        self.scroll_a
    }
    pub fn get_scroll_b(&self) -> Option<ScrollPanel> {
        self.scroll_b
    }
    /// `setStatusBarText(File, BaseMetaData, LogWindow)` with the unavailable
    /// metadata/log calls represented by the exact source inputs it consumes.
    pub fn set_status_bar_text(&mut self, param_file: Option<&Path>, has_metadata: bool) {
        if !has_metadata {
            self.status_bar = STATUS_BAR_EMPTY_TITLE.into();
            return;
        }
        let Some(file) = param_file else {
            self.status_bar = format!("{STATUS_BAR_BASE_TITLE}NOT SAVED");
            return;
        };
        let mut name = file.to_string_lossy().to_string();
        const MAX: usize = 60;
        if STATUS_BAR_BASE_TITLE.len() + name.len() > MAX {
            name = format!(
                "...{}",
                &name[name.len() - (MAX - STATUS_BAR_BASE_TITLE.len() - 3)..]
            );
        }
        self.status_bar = format!("{STATUS_BAR_BASE_TITLE}{name}");
    }
    pub fn set_status_bar_text_to_directory(&mut self, directory: Option<&str>, max: usize) {
        let Some(mut value) = directory else {
            self.status_bar.clear();
            return;
        };
        if value.trim().is_empty() {
            self.status_bar.clear();
        } else if value.len() > max {
            value = &value[value.len() - max..];
            self.status_bar = format!("...{value}");
        } else {
            self.status_bar = value.into();
        }
    }
    /// Java `getProgressPanel(AxisID)`.  This is deliberately owned by the
    /// MainPanel source unit: concrete main panels only decide whether they
    /// expose the returned panel for a particular axis.
    pub fn get_progress_panel(&mut self, axis_id: AxisID) -> &mut AxisProgressPanel {
        if axis_id == AxisID::Second {
            if self.axis_progress_panel_b.is_none() {
                self.axis_progress_panel_b =
                    Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
            }
            return self.axis_progress_panel_b.as_mut().unwrap();
        }
        if self.axis_progress_panel_a.is_none() {
            self.axis_progress_panel_a =
                Some(AxisProgressPanel::get_instance(Some(axis_id), self.manager));
        }
        self.axis_progress_panel_a.as_mut().unwrap()
    }
    /// Attach a native axis-process panel to its source axis.
    pub fn set_axis_process_panel(&mut self, axis: AxisID, panel: AxisProcessPanel) {
        if axis == AxisID::Second {
            self.axis_process_panel_b = Some(panel);
        } else {
            self.axis_process_panel_a = Some(panel);
        }
    }
    fn axis_process_panel(&self, axis: AxisID) -> Option<&AxisProcessPanel> {
        if axis == AxisID::Second {
            self.axis_process_panel_b.as_ref()
        } else {
            self.axis_process_panel_a.as_ref()
        }
    }
    fn axis_process_panel_mut(&mut self, axis: AxisID) -> Option<&mut AxisProcessPanel> {
        if axis == AxisID::Second {
            self.axis_process_panel_b.as_mut()
        } else {
            self.axis_process_panel_a.as_mut()
        }
    }
    /// Java `getParallelPanel(AxisID)`.
    pub fn get_parallel_panel(&self, axis: AxisID) -> Option<&ParallelPanel> {
        self.axis_process_panel(axis)
            .and_then(AxisProcessPanel::get_parallel_panel)
    }
    /// Native availability form of Java `getParallelPauseButton(AxisID)`.
    pub fn get_parallel_pause_button(&self, axis: AxisID) -> Option<bool> {
        self.get_parallel_panel(axis)
            .map(ParallelPanel::get_parallel_pause_button)
    }
    /// Native availability form of Java `getParallelResumeButton(AxisID)`.
    pub fn get_parallel_resume_button(&mut self, axis: AxisID) -> Option<bool> {
        self.axis_process_panel_mut(axis)
            .and_then(|panel| panel.parallel_panel.as_mut())
            .map(ParallelPanel::get_parallel_resume_button)
    }
    /// Native visibility form of Java `getParallelStatusPanel(AxisID)`.
    pub fn get_parallel_status_panel(&self, axis: AxisID) -> Option<bool> {
        self.axis_process_panel(axis)
            .map(AxisProcessPanel::get_parallel_status_panel)
    }
    /// Java `done()`: notify both instantiated process panels.
    pub fn done(&mut self) {
        if let Some(panel) = self.axis_process_panel_a.as_mut() {
            panel.done();
        }
        if let Some(panel) = self.axis_process_panel_b.as_mut() {
            panel.done();
        }
    }
    /// Native scheduling form of Java `SetBusyStatus.run()`.
    pub fn run(&mut self, axis: AxisID, enabled: bool) {
        self.msg_busy_status_changed(axis, enabled);
    }
    pub fn set_divider_location(&mut self, _value: f64) {}
    pub fn map_progress_panel_state(&mut self, axis: AxisID) -> &mut ProgressPanelState {
        if axis == AxisID::Second {
            self.progress_panel_state_b
                .get_or_insert_with(ProgressPanelState::new)
        } else {
            &mut self.progress_panel_state_a
        }
    }
    pub fn set_process_can_be_killed(&mut self, value: bool, axis: AxisID) {
        self.map_progress_panel_state(axis).print_killable_state(
            "setProcessCanBeKilled",
            false,
            value,
        );
    }
    pub fn set_progress_bar_indeterminate(&mut self, label: &str, axis: AxisID, _mode: bool) {
        self.map_progress_panel_state(axis)
            .print_state_label("setProgressBar", false, label);
    }
    pub fn set_progress_bar(&mut self, label: &str, steps: i32, mode: bool, axis: AxisID) -> bool {
        self.set_progress_bar_pause(label, steps, mode, axis, false)
    }
    pub fn set_progress_bar_pause(
        &mut self,
        label: &str,
        steps: i32,
        _mode: bool,
        axis: AxisID,
        pause: bool,
    ) -> bool {
        self.map_progress_panel_state(axis).print_state_steps(
            "setProgressBar",
            false,
            label,
            steps,
            pause,
        )
    }
    pub fn set_static_progress_bar(&mut self, label: &str, axis: AxisID) {
        self.map_progress_panel_state(axis)
            .print_state_label("setStaticProgressBar", false, label);
    }
    pub fn set_progress_bar_value_plain(&mut self, value: i32, axis: AxisID) {
        self.map_progress_panel_state(axis)
            .print_state_value("setProgressBarValue", false, value);
    }
    pub fn set_progress_bar_value(&mut self, value: i32, bar: &str, axis: AxisID) -> bool {
        self.map_progress_panel_state(axis)
            .print_state_value_string("setProgressBarValue", false, value, bar)
    }
    pub fn set_progress_bar_value_standard(
        &mut self,
        value: i32,
        _standard: StandardBarString,
        bar: &str,
        axis: AxisID,
    ) -> bool {
        self.set_progress_bar_value(value, bar, axis)
    }
    pub fn set_emergency_monitor_bar_string(
        &mut self,
        standard: StandardBarString,
        from: Option<&str>,
        to: Option<&str>,
        renamed: bool,
        failed: bool,
        axis: AxisID,
    ) {
        self.map_progress_panel_state(axis)
            .bar_string
            .set(Some(standard.build_bar_string(from, to, renamed, failed)));
    }
    pub fn start_progress_bar(&mut self, label: &str, axis: AxisID, process: Option<ProcessName>) {
        self.map_progress_panel_state(axis).print_state_process(
            "startProgressBar",
            false,
            label,
            process,
        );
    }
    pub fn stop_progress_bar(&mut self, axis: AxisID, end: &str, status: Option<&str>) {
        self.map_progress_panel_state(axis).print_state_end_status(
            "stopProgressBar",
            false,
            end,
            status,
        );
    }
    /// Java `isProgressBarStopped(AxisID)`.
    pub fn is_progress_bar_stopped(&mut self, axis: AxisID) -> bool {
        self.get_progress_panel(axis).is_progress_bar_stopped(axis)
    }

    /// Native state form of Java `showProcessingPanel(AxisType)`.
    /// Concrete axis-process widgets are frontend-owned; MainPanel retains the source
    /// axis, scrolling, and selected-pane transitions.
    pub fn show_processing_panel(&mut self, axis_type: AxisType) {
        self.axis_type = axis_type;
        self.panel_center.clear();
        self.showing_both_axis = false;
        let first_axis = if axis_type == AxisType::SingleAxis {
            AxisID::Only
        } else {
            AxisID::First
        };
        self.axis_progress_panel_a = Some(AxisProgressPanel::get_instance(
            Some(first_axis),
            self.manager,
        ));
        self.scroll_a = Some(ScrollPanel::new());
        self.scroll_pane_a = Some(0);
        if axis_type == AxisType::SingleAxis {
            self.axis_progress_panel_b = None;
            self.scroll_b = None;
            self.scroll_pane_b = None;
        } else {
            self.axis_progress_panel_b = Some(AxisProgressPanel::get_instance(
                Some(AxisID::Second),
                self.manager,
            ));
            self.scroll_b = Some(ScrollPanel::new());
            self.scroll_pane_b = Some(0);
        }
        self.set_axis_a();
    }
    pub fn get_vertical_scroll_bar_value(&self, axis: AxisID) -> Option<i32> {
        if axis == AxisID::Second {
            self.vertical_scroll_b
        } else {
            self.vertical_scroll_a
        }
    }
    pub fn set_vertical_scroll_bar_value(&mut self, axis: AxisID, value: Option<i32>) {
        if axis == AxisID::Second {
            self.vertical_scroll_b = value
        } else {
            self.vertical_scroll_a = value
        }
    }
    pub fn set_scroll_bar_increments(&mut self) {}
    pub fn show_both_axis(&mut self) -> Option<i32> {
        if self.axis_type != AxisType::DualAxis || self.showing_both_axis {
            return None;
        }
        self.showing_both_axis = true;
        self.showing_axis_a = true;
        self.scroll_pane_b
    }
    pub fn is_showing_both_axis(&self) -> bool {
        self.showing_both_axis
    }
    pub fn is_showing_axis_a(&self) -> bool {
        self.showing_axis_a
    }
    pub fn msg_busy_status_changed(&mut self, axis: AxisID, status: bool) {
        if axis == AxisID::Second {
            self.busy_status_b = Some(status)
        } else {
            self.busy_status_a = status
        }
    }
    pub fn create_busy_status_label() -> (&'static str, &'static str) {
        ("busy-status", "busy-status")
    }
    pub fn show_axis_a(&mut self) {
        self.panel_center.clear();
        self.set_axis_a();
    }
    /// Java private `setAxisA()`.
    pub fn set_axis_a(&mut self) {
        self.showing_both_axis = false;
        self.showing_axis_a = true;
        if self.manager.is_valid() && self.scroll_pane_a.is_some() {
            self.panel_center.push(AxisID::First)
        }
    }
    pub fn show_axis_b(&mut self) {
        self.panel_center.clear();
        self.showing_both_axis = false;
        self.showing_axis_a = false;
        if self.scroll_pane_b.is_some() {
            self.panel_center.push(AxisID::Second)
        };
        if self.busy_status_b.is_none() {
            self.busy_status_b = Some(false)
        }
    }
    pub fn set_vertical_scroll_bar_policy(&mut self, always: bool) {
        self.vertical_scroll_always = always;
    }
    pub fn repaint_window(&mut self) {
        self.repaint_container();
        self.repaint();
    }
    /// Native retained-tree equivalent of Java recursive `repaintContainer(Container)`.
    pub fn repaint_container(&mut self) {
        self.repaint_count += self.scroll_a.is_some() as u64 + self.scroll_b.is_some() as u64;
    }
    pub fn get_axis_type(&self) -> AxisType {
        self.axis_type
    }
    pub fn is_showing_setup(&self) -> bool {
        self.showing_setup
    }
    pub fn set_showing_setup(&mut self, input: bool) {
        self.showing_setup = input;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn panel() -> MainPanel {
        MainPanel::new(DirectiveEditorManager::new(None, None, None, None))
    }
    #[test]
    fn status_title_matches_java_shortening() {
        let mut p = panel();
        p.set_status_bar_text(
            Some(Path::new(
                "/a/very/long/path/whose/dataset/name/needs/to/be/shortened.edf",
            )),
            true,
        );
        assert!(p.get_status_bar_text().starts_with("Data file: ..."));
        assert!(p.get_status().starts_with("..."));
    }
    #[test]
    fn b_progress_state_is_lazy_and_axis_specific() {
        let mut p = panel();
        p.set_progress_bar("x", 4, false, AxisID::Second);
        assert!(p.progress_panel_state_b.is_some());
        assert!(p.progress_panel_state_a.n_steps.cur_value.is_none());
    }
    #[test]
    fn dual_axis_display_follows_source_guard() {
        let mut p = panel();
        p.axis_type = AxisType::DualAxis;
        p.scroll_pane_b = Some(7);
        assert_eq!(p.show_both_axis(), Some(7));
        assert!(p.is_showing_both_axis());
        assert_eq!(p.show_both_axis(), None);
    }
}
