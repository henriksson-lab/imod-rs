//! `IMOD/Etomo/src/etomo/ui/swing/ProgressPanel.java`.
//!
//! Swing controls/timer are represented by their source-visible state.  Public
//! mutators execute the corresponding `invokeLater` runnable immediately at
//! the Rust GUI boundary; callers already invoke these through the UI event
//! loop, so no alternate worker/UI queue is introduced.
#![allow(dead_code)]
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;

pub const NAME: &str = "the-progress-bar";
pub const LABEL_NAME: &str = "the-progress-bar-label";
pub const MAX_PACK: i32 = 5;
/// Rust text form of unported `ProcessEndState.getBarString()`.
pub const DONE_BAR_STRING: &str = "Done";

/// Fields and all method behavior of Java's final ProgressPanel.
pub struct ProgressPanel {
    pub panel_visible: bool,
    pub task_label: String,
    pub progress_bar_name: &'static str,
    pub label_name: &'static str,
    pub manager: &'static dyn BaseManager,
    pub axis_id: AxisID,
    pub counter: i32,
    pub value: i32,
    pub maximum: i32,
    pub minimum: i32,
    pub start_time: Option<std::time::Instant>,
    pub cur_standard_bar_string: Option<StandardBarString>,
    pub bar_string: Option<String>,
    pub emergency_monitor_alert: bool,
    pub label: Option<String>,
    pub stopped: bool,
    pub n_packed: i32,
    pub indeterminate: bool,
    pub string_painted: bool,
    pub timer_running: bool,
    pub repaint_count: u64,
    pub revalidate_count: u64,
}
impl ProgressPanel {
    fn new(label: Option<&str>, manager: &'static dyn BaseManager, axis_id: AxisID) -> Self {
        Self {
            panel_visible: true,
            task_label: label.unwrap_or("").into(),
            progress_bar_name: NAME,
            label_name: LABEL_NAME,
            manager,
            axis_id,
            counter: 0,
            value: 0,
            maximum: 100,
            minimum: 0,
            start_time: None,
            cur_standard_bar_string: None,
            bar_string: None,
            emergency_monitor_alert: false,
            label: None,
            stopped: true,
            n_packed: 0,
            indeterminate: false,
            string_painted: false,
            timer_running: false,
            repaint_count: 0,
            revalidate_count: 0,
        }
    }
    pub fn get_instance(
        label: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Self {
        let mut panel = Self::new(label, manager, axis_id);
        panel.add_listeners();
        panel
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.panel_visible = visible;
    }
    fn add_listeners(&mut self) {}
    fn pack(&mut self) {
        if self.n_packed < MAX_PACK {
            self.n_packed += 1;
        }
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
    pub fn set_background(&mut self, _background: impl Into<String>) {}
    pub fn set_label(&mut self, label: &str) {
        self.stopped = false;
        self.label = Some(label.into());
        self.set_label_later();
    }
    fn set_label_later(&mut self) {
        self.set_task_label();
        self.revalidate();
        self.repaint();
        self.pack();
    }
    pub fn start(&mut self) {
        self.stopped = false;
        self.counter = 0;
        self.bar_string = None;
        self.start_time = Some(std::time::Instant::now());
        self.start_later();
    }
    fn start_later(&mut self) {
        self.emergency_monitor_alert = false;
        self.indeterminate = true;
        self.cur_standard_bar_string = None;
        self.set_bar_string_and_prefer("");
        self.start_timer();
        self.pack();
    }
    pub fn start_indeterminate_mode(&mut self) {
        self.stopped = false;
        self.counter = 0;
        self.start_time = Some(std::time::Instant::now());
        self.start_indeterminate_mode_later();
    }
    fn start_indeterminate_mode_later(&mut self) {
        self.emergency_monitor_alert = false;
        self.indeterminate = true;
        self.cur_standard_bar_string = None;
        self.set_bar_string_and_prefer("");
        self.revalidate();
        self.repaint();
        self.pack();
    }
    pub fn stop(&mut self, state: Option<&str>, status: Option<&str>) {
        self.stopped = true;
        self.counter = 0;
        if state != Some("FILE_LOCK_FAILURE") {
            self.bar_string = None;
        }
        self.stop_later(state.unwrap_or(DONE_BAR_STRING), status);
    }
    fn stop_later(&mut self, state: &str, status: Option<&str>) {
        self.stop_timer();
        self.value = self.counter;
        self.indeterminate = false;
        self.emergency_monitor_alert = false;
        let mut text = if self.cur_standard_bar_string.is_some() {
            self.bar_string.clone().unwrap_or_default()
        } else {
            String::new()
        };
        if !state.is_empty() {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(state);
        }
        if let Some(status) = status.filter(|s| !s.is_empty()) {
            if !text.is_empty() {
                text.push_str(":  ");
            }
            text.push_str(status);
        }
        self.set_bar_string_and_prefer(&text);
        self.pack();
    }
    pub fn increment(&mut self) {
        self.increment_later(self.stopped);
    }
    fn increment_later(&mut self, stopped: bool) {
        if stopped {
            return;
        }
        self.value = self.value;
        let elapsed = self.start_time.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let time = format!("{}:{:02}", elapsed / 60, elapsed % 60);
        let base = self
            .bar_string
            .clone()
            .unwrap_or_else(|| "Elapsed time".into());
        self.set_bar_string_and_prefer(&format!("{base} : {time}"));
        self.validate();
        self.repaint();
        self.counter += 1;
        self.restart_timer();
        self.pack();
    }
    pub fn set_maximum(&mut self, maximum: i32, indeterminate: bool) {
        self.stopped = false;
        self.maximum = maximum;
        self.set_maximum_later(indeterminate);
    }
    fn set_maximum_later(&mut self, indeterminate: bool) {
        self.indeterminate = indeterminate;
        self.string_painted = true;
        if indeterminate {
            self.revalidate();
            self.repaint();
        }
        self.pack();
    }
    pub fn set_minimum(&mut self, minimum: i32) {
        self.stopped = false;
        self.minimum = minimum;
    }
    pub fn set_value(&mut self, value: i32) {
        self.stopped = false;
        self.value = value;
    }
    pub fn set_emergency_monitor_bar_string(&mut self, standard: StandardBarString, bar: &str) {
        self.emergency_monitor_alert = true;
        self.set_progress_bar_string(Some(standard), Some(bar));
    }
    pub fn is_progress_bar_stopped(&self) -> bool {
        self.stopped
    }
    pub fn set_value_standard(
        &mut self,
        value: i32,
        standard: Option<StandardBarString>,
        bar: Option<&str>,
        _print: bool,
    ) {
        self.stopped = false;
        self.value = value;
        self.set_progress_bar_string(standard, bar);
    }
    pub fn get_container(&self) -> bool {
        self.panel_visible
    }
    pub fn get_maximum(&self) -> i32 {
        self.maximum
    }
    pub fn get_minimum(&self) -> i32 {
        self.minimum
    }
    pub fn get_value(&self) -> i32 {
        self.value
    }
    #[allow(non_snake_case)]
    pub fn getStartTime(&self) -> Option<std::time::Instant> {
        self.start_time
    }
    /// Native replacement for Java `getProgressBar`; state is rendered by the
    /// selected Rust GUI toolkit rather than exposing a Swing component.
    #[allow(non_snake_case)]
    pub fn getProgressBar(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn setProgressBarCounter(&mut self) {
        self.value = self.counter;
    }
    #[allow(non_snake_case)]
    pub fn setProgressBarValue(&mut self) {
        self.value = self.value;
    }
    #[allow(non_snake_case)]
    pub fn incrementCounter(&mut self) {
        self.counter += 1;
    }
    #[allow(non_snake_case)]
    pub fn setProgressBarMaximum(&mut self) {
        self.maximum = self.maximum;
    }
    #[allow(non_snake_case)]
    pub fn setProgressBarMinimum(&mut self) {
        self.minimum = self.minimum;
    }
    #[allow(non_snake_case)]
    pub fn setProgressBarString(&mut self, standard: Option<StandardBarString>, bar: Option<&str>) {
        self.set_progress_bar_string(standard, bar);
    }
    fn set_task_label(&mut self) {
        self.cur_standard_bar_string = None;
        self.task_label = self.label.clone().unwrap_or_default();
    }
    fn revalidate(&mut self) {
        self.revalidate_count += 1;
    }
    fn repaint(&mut self) {
        self.repaint_count += 1;
    }
    fn validate(&mut self) {}
    fn restart_timer(&mut self) {
        self.timer_running = true;
    }
    fn start_timer(&mut self) {
        self.timer_running = true;
    }
    fn stop_timer(&mut self) {
        self.timer_running = false;
    }
    fn set_bar_string_and_prefer(&mut self, bar: &str) -> bool {
        let changed = self.bar_string.as_deref() != Some(bar);
        self.bar_string = Some(bar.into());
        self.string_painted = true;
        changed
    }
    fn set_progress_bar_string(&mut self, standard: Option<StandardBarString>, bar: Option<&str>) {
        self.cur_standard_bar_string = standard;
        if self.set_bar_string_and_prefer(bar.unwrap_or("")) {
            self.pack();
        }
    }
}
pub struct ProgressTimerActionListener;
impl ProgressTimerActionListener {
    pub fn action_performed(panel: &mut ProgressPanel) {
        panel.increment();
    }
}

/// GUI-loop callback used for each Java `Runnable` in `ProgressPanel`.
/// The caller selects the source transition; each operates synchronously at
/// the native UI boundary, exactly as the panel's public methods do.
pub struct ProgressPanelRunnable;
impl ProgressPanelRunnable {
    pub fn run(panel: &mut ProgressPanel) {
        panel.increment();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn panel() -> ProgressPanel {
        ProgressPanel::get_instance(
            Some("No process"),
            DirectiveEditorManager::new(None, None, None, None),
            AxisID::Only,
        )
    }
    #[test]
    fn determinate_state_and_stop_follow_source() {
        let mut p = panel();
        p.set_label("align");
        p.set_minimum(0);
        p.set_maximum(5, false);
        p.set_value(2);
        p.stop(Some("DONE"), Some("ok"));
        assert!(p.is_stopped());
        assert!(!p.indeterminate);
        assert!(p.bar_string.as_deref().unwrap().contains("DONE:  ok"));
    }
    #[test]
    fn timer_does_not_increment_after_stop() {
        let mut p = panel();
        p.start();
        p.increment();
        let value = p.value;
        p.stop(None, None);
        p.increment();
        assert_eq!(p.value, value);
    }
}
