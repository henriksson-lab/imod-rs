//! `IMOD/Etomo/src/etomo/ui/swing/ParallelPanel.java`.
//!
//! ProcessorTable, PanelHeader, ProcessingMethodMediator, process parameters,
//! and native controls are direct source dependencies.  Until their own units
//! exist, this module retains the source-owned table selection, visibility,
//! persistence, and button state instead of replacing them with another UI.
#![allow(dead_code)]
use super::axis_process_panel::AxisProcessPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
pub use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use std::collections::BTreeMap;

pub const STORE_PREPEND: &str = "ProcessorTable";
pub const TITLE: &str = "Parallel Processing";
pub const NON_RUNNABLE_TITLE: &str = "Resources";
pub const RESUME_LABEL: &str = "Resume";
pub const PAUSE_LABEL: &str = "Pause";
pub const FIELD_LABEL: &str = "Parallel processing";
pub const MAX_CPUS_STRING: &str = ":  Maximum number of cores recommended is ";
pub const CPUS_SELECTED_LABEL: &str = "Cores: ";
pub const GPUS_SELECTED_LABEL: &str = "GPUs: ";

/// Java private static `ParallelPanelActionListener`.
pub struct ParallelPanelActionListener;
impl ParallelPanelActionListener {
    pub fn action_performed(panel: &mut ParallelPanel, command: Option<&str>) {
        panel.action(command);
    }
}

/// Java `ParallelPanel` fields. ProcessorTable identities are source table kinds;
/// their actual monitor/table widgets remain direct boundaries.
pub struct ParallelPanel {
    pub manager: &'static dyn BaseManager,
    pub axis_id: AxisID,
    pub parent_axis_id: AxisID,
    pub popup_chunk_warnings: bool,
    pub interface_type: InterfaceType,
    pub visible: bool,
    pub open: bool,
    pub pause_enabled: bool,
    pub processing_method_locked: bool,
    pub processing_running: bool,
    pub runnable: bool,
    pub outside_resume_control: bool,
    pub current_table: ProcessingMethod,
    pub secondary_table: Option<ProcessingMethod>,
    pub cpu_table_present: bool,
    pub queue_table_present: bool,
    pub gpu_table_present: bool,
    pub table_visible: BTreeMap<String, bool>,
    pub table_stopped: BTreeMap<String, bool>,
    pub queues_enabled: bool,
    pub queues_selected: bool,
    pub queues_editable: bool,
    pub root_visible: bool,
    pub body_visible: bool,
    pub header_title: String,
    pub header_less: bool,
    pub pause_button_enabled: bool,
    pub pause_button_visible: bool,
    pub resume_button_enabled: bool,
    pub resume_button_visible: bool,
    pub save_defaults_visible: bool,
    pub restart_load_count: u32,
    pub cpus_selected: String,
    pub secondary_cpus_selected: String,
    pub cpus_selected_label: String,
    pub secondary_cpus_selected_label: String,
    pub secondary_cpus_visible: bool,
    pub nice: i32,
    pub version: String,
    pub queue_events: Vec<QueueTableEvent>,
}
impl ParallelPanel {
    /// `getInstance(...)` plus private constructor and `addListeners`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        parent: &AxisProcessPanel,
        popup_chunk_warnings: bool,
        runnable: bool,
        interface_type: InterfaceType,
        has_queues: bool,
        has_gpus: bool,
    ) -> Self {
        let title = if runnable { TITLE } else { NON_RUNNABLE_TITLE }.to_string();
        let mut table_visible = BTreeMap::new();
        table_visible.insert("cpu".into(), true);
        table_visible.insert("queue".into(), false);
        table_visible.insert("gpu".into(), false);
        let mut table_stopped = BTreeMap::new();
        table_stopped.insert("cpu".into(), true);
        table_stopped.insert("queue".into(), true);
        table_stopped.insert("gpu".into(), true);
        let mut panel = Self {
            manager,
            axis_id,
            parent_axis_id: parent.axis_id,
            popup_chunk_warnings,
            interface_type,
            visible: true,
            open: true,
            pause_enabled: false,
            processing_method_locked: false,
            processing_running: false,
            runnable,
            outside_resume_control: false,
            current_table: ProcessingMethod::PpCpu,
            secondary_table: None,
            cpu_table_present: true,
            queue_table_present: has_queues,
            gpu_table_present: has_gpus,
            table_visible,
            table_stopped,
            queues_enabled: has_queues,
            queues_selected: false,
            queues_editable: has_queues,
            root_visible: runnable,
            body_visible: true,
            header_title: title,
            header_less: false,
            pause_button_enabled: false,
            pause_button_visible: runnable,
            resume_button_enabled: true,
            resume_button_visible: runnable,
            save_defaults_visible: true,
            restart_load_count: 0,
            cpus_selected: String::new(),
            secondary_cpus_selected: String::new(),
            cpus_selected_label: CPUS_SELECTED_LABEL.into(),
            secondary_cpus_selected_label: GPUS_SELECTED_LABEL.into(),
            secondary_cpus_visible: false,
            nice: 0,
            version: String::new(),
            queue_events: vec![],
        };
        if has_queues {
            panel.send_queue_table_event(panel.get_queue_table_displayed_event());
        }
        panel
    }
    pub fn add_listeners(&mut self) {}
    pub fn is_runnable(&self) -> bool {
        self.runnable
    }
    pub fn reset_results(&mut self) {}
    pub fn set_pause_enabled(&mut self, enabled: bool) {
        self.pause_enabled = enabled;
        self.pause_button_enabled = enabled;
    }
    pub fn set_cpus_selected(&mut self, cpus: i32) {
        self.cpus_selected = cpus.to_string();
    }
    pub fn set_secondary_cpus_selected(&mut self, cpus: i32) {
        self.secondary_cpus_selected = cpus.to_string();
    }
    pub fn get_cpus_selected(&self, do_validation: bool) -> Result<String, String> {
        if do_validation && self.cpus_selected.trim().is_empty() {
            Err("FieldValidationFailedException".into())
        } else {
            Ok(self.cpus_selected.clone())
        }
    }
    pub fn get_cpus_selected_int(&self, do_validation: bool) -> Result<i32, String> {
        Ok(self.get_cpus_selected(do_validation)?.parse().unwrap_or(0))
    }
    pub fn get_cpus_selected_label(&self) -> &str {
        &self.cpus_selected_label
    }
    pub fn get_no_cpus_selected_error_message(&self) -> &'static str {
        "No CPUs selected"
    }
    pub fn get_secondary_no_cpus_selected_error_message(&self) -> Option<&'static str> {
        self.secondary_table.map(|_| "No secondary CPUs selected")
    }
    pub fn get_container(&self) -> bool {
        self.root_visible
    }
    pub fn get_parallel_pause_button(&self) -> bool {
        self.pause_button_enabled
    }
    pub fn get_parallel_resume_button(&mut self) -> bool {
        self.outside_resume_control = true;
        self.resume_button_enabled
    }
    pub fn set_process_info(&mut self, root_name: Option<&str>) {
        if let Some(root) = root_name {
            self.header_title = format!("{TITLE} {root}");
        }
    }
    /// private `action(String)`; BaseManager resume/preferences and ProcessorTable
    /// monitor calls remain exact named dependencies.
    pub fn action(&mut self, command: Option<&str>) {
        match command {
            Some(RESUME_LABEL) => {
                self.processing_running = true;
                self.resume_button_enabled = false;
            }
            Some(PAUSE_LABEL) => {
                self.manager.pause(Some(self.axis_id));
            }
            Some("Save As Defaults") => {}
            Some("Restart Load") => {
                self.restart_load_count += 1;
            }
            Some("Use a cluster") => {
                if self.is_queues() {
                    self.set_processing_method(Some(ProcessingMethod::Queue));
                } else {
                    self.set_processing_method(None);
                }
                self.send_queue_table_event(self.get_queue_table_displayed_event());
            }
            _ => {}
        }
    }
    pub fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        match event {
            QueueTableEvent::PreventDisplay => self.queues_enabled = false,
            QueueTableEvent::Display => {
                self.queues_enabled = true;
                self.queues_selected = true;
            }
            QueueTableEvent::AllowDisplay => self.queues_enabled = true,
            _ => {}
        };
        self.action(Some("Use a cluster"));
    }
    pub fn get_queue_table_displayed_event(&self) -> QueueTableEvent {
        if self.is_queues() {
            QueueTableEvent::Displayed
        } else {
            QueueTableEvent::Hidden
        }
    }
    pub fn send_queue_table_event(&mut self, event: QueueTableEvent) {
        self.queue_events.push(event);
    }
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        let key = format!("{}.version", Self::store_prepend(prepend));
        if let Some(v) = props.get(&key) {
            self.version = v.clone();
        }
    }
    pub fn store(&mut self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        self.version = "1.1".into();
        props.insert(
            format!("{}.version", Self::store_prepend(prepend)),
            self.version.clone(),
        );
    }
    fn store_prepend(prepend: Option<&str>) -> String {
        if prepend.is_none_or(|v| v.is_empty()) {
            STORE_PREPEND.into()
        } else {
            format!("{}.{}", prepend.unwrap(), STORE_PREPEND)
        }
    }
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.processing_method_locked = lock;
        self.update_processing_method_lock();
    }
    fn update_processing_method_lock(&mut self) {
        self.queues_editable = !self.processing_method_locked && !self.processing_running;
    }
    pub fn set_runnable(&mut self, runnable: bool) {
        self.runnable = runnable;
        self.header_title = if runnable { TITLE } else { NON_RUNNABLE_TITLE }.into();
        self.pause_button_visible = runnable;
        self.resume_button_visible = runnable;
    }
    pub fn set_limited(&mut self, _limited: bool) {}
    pub fn set_queue(
        &mut self,
        registering: bool,
        origin_present: bool,
        method: Option<ProcessingMethod>,
    ) {
        if registering
            && origin_present
            && self.queues_enabled
            && method == Some(ProcessingMethod::Queue)
        {
            self.queues_selected = true;
        }
    }
    pub fn set_processing_method(&mut self, method: Option<ProcessingMethod>) {
        let method = if self.is_queues() {
            Some(ProcessingMethod::Queue)
        } else {
            method
        };
        let Some(method) = method else {
            self.stop_current_table();
            return;
        };
        if method.is_local() {
            self.stop_current_table();
            return;
        }
        if self.current_table != method {
            self.table_visible
                .insert(Self::table_key(self.current_table).into(), false);
            self.stop_current_table();
            self.current_table = method;
            self.table_visible
                .insert(Self::table_key(method).into(), true);
            self.table_stopped
                .insert(Self::table_key(method).into(), false);
            self.cpus_selected_label = if method == ProcessingMethod::PpGpu {
                GPUS_SELECTED_LABEL
            } else {
                CPUS_SELECTED_LABEL
            }
            .into();
        } else if self.table_stopped[Self::table_key(method)] {
            self.table_stopped
                .insert(Self::table_key(method).into(), false);
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.root_visible = visible;
        if visible {
            self.table_stopped
                .insert(Self::table_key(self.current_table).into(), false);
        } else {
            self.stop_current_table();
        }
    }
    pub fn set_secondary_processing_method(&mut self, method: Option<ProcessingMethod>) {
        self.secondary_table = method.filter(|m| !m.is_local());
        self.secondary_cpus_visible = self.secondary_table.is_some();
        if let Some(m) = self.secondary_table {
            self.secondary_cpus_selected_label = if m == ProcessingMethod::PpGpu {
                GPUS_SELECTED_LABEL
            } else {
                CPUS_SELECTED_LABEL
            }
            .into();
        }
    }
    pub fn stop_table(&mut self) {
        self.stop_current_table();
    }
    pub fn end_table(&mut self) {
        self.stop_current_table();
    }
    pub fn get_table(&self, method: Option<ProcessingMethod>) -> ProcessingMethod {
        method.unwrap_or(self.current_table)
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.current_table
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        self.secondary_table
    }
    pub fn msg_ending_process(&mut self) {
        self.processing_running = false;
        self.update_processing_method_lock();
    }
    pub fn msg_killing_process(&mut self) {
        self.pause_button_enabled = false;
        if !self.outside_resume_control {
            self.resume_button_enabled = false;
        }
    }
    pub fn msg_pausing_process(&mut self) {
        if !self.outside_resume_control {
            self.resume_button_enabled = true;
        }
    }
    pub fn msg_process_done(&mut self) {
        if !self.outside_resume_control {
            self.resume_button_enabled = true;
        }
    }
    pub fn msg_process_started(&mut self) {
        if !self.outside_resume_control {
            self.resume_button_enabled = false;
        }
    }
    pub fn expand_open_close(&mut self, expanded: bool) {
        self.open = expanded;
        self.body_visible = expanded;
    }
    pub fn expand_more_less(&mut self, expanded: bool) {
        self.set_more_less(expanded);
    }
    pub fn msg_selection_changed(&mut self) {
        if self.header_less {
            self.set_more_less(true);
            self.set_more_less(false);
        }
    }
    fn set_more_less(&mut self, more: bool) {
        self.header_less = !more;
        self.save_defaults_visible = more;
    }
    pub fn is_cb_use_gpu(&self) -> bool {
        self.current_table == ProcessingMethod::PpGpu
    }
    fn is_queues(&self) -> bool {
        self.queue_table_present && self.queues_enabled && self.queues_selected
    }
    pub fn get_use_queue_checkbox(&self) -> bool {
        self.queues_selected
    }
    fn stop_current_table(&mut self) {
        self.table_stopped
            .insert(Self::table_key(self.current_table).into(), true);
    }
    fn table_key(method: ProcessingMethod) -> &'static str {
        match method {
            ProcessingMethod::PpCpu => "cpu",
            ProcessingMethod::PpGpu => "gpu",
            ProcessingMethod::Queue => "queue",
            ProcessingMethod::LocalCpu | ProcessingMethod::LocalGpu => "cpu",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::axis_progress_panel::AxisProgressPanel;
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn panel() -> ParallelPanel {
        let m = DirectiveEditorManager::new(None, None, None, None);
        let a = AxisProgressPanel::get_instance(Some(AxisID::First), m);
        let parent = AxisProcessPanel::new(
            AxisID::First,
            m,
            false,
            true,
            InterfaceType::DirectiveEditor,
            false,
            a,
        );
        ParallelPanel::get_instance(
            m,
            AxisID::First,
            &parent,
            false,
            true,
            InterfaceType::DirectiveEditor,
            true,
            true,
        )
    }
    #[test]
    fn queue_overrides_current_table() {
        let mut p = panel();
        p.queues_selected = true;
        p.set_processing_method(Some(ProcessingMethod::PpGpu));
        assert_eq!(p.get_processing_method(), ProcessingMethod::Queue);
    }
    #[test]
    fn hide_stops_and_show_starts_current_monitor() {
        let mut p = panel();
        p.set_visible(false);
        assert!(p.table_stopped["cpu"]);
        p.set_visible(true);
        assert!(!p.table_stopped["cpu"]);
    }
    #[test]
    fn storage_uses_source_prepend() {
        let mut p = panel();
        let mut props = BTreeMap::new();
        p.store(&mut props, Some("x"));
        assert_eq!(props["x.ProcessorTable.version"], "1.1");
    }
}
