//! `IMOD/Etomo/src/etomo/ui/swing/CpuGpuPanel.java`.
//!
//! Swing painting, `Network`, `CpuAdoc`, the user configuration, and the
//! processing-method mediator are explicit application boundaries.  This unit
//! retains the source panel's selection, locking, availability, and mediator
//! dispatch policy.
#![allow(dead_code)]

use super::check_box::CheckBox;
use super::parallel_panel::{FIELD_LABEL, MAX_CPUS_STRING};
use super::process_interface::ProcessInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java `PanelId` values examined by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PanelId {
    Tilt,
    Tilt3dFind,
    Other,
}

/// `ConstMetaData` fields queried by `setParameters(ConstMetaData, ...)`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConstMetaDataBoundary {
    pub default_gpu_processing: bool,
    pub default_parallel: bool,
}

/// Java mutable `MetaData.setTiltParallel` call boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetaDataBoundary {
    pub tilt_parallel: Option<bool>,
}

/// Java `ConstTiltParam`/`TiltParam` GPU property.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TiltParamBoundary {
    pub use_gpu: bool,
}

/// Java `ConstCtfPhaseFlipParam`/`CtfPhaseFlipParam` GPU property.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CtfPhaseFlipParamBoundary {
    pub use_gpu: bool,
}

/// Java `ProcessingMethodMediator` calls emitted by this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessingMethodMediatorBoundary {
    pub registered: bool,
    pub gpu_listener_count: usize,
    pub method_calls: Vec<(bool, ProcessingMethod)>,
    pub run_method: Option<ProcessingMethod>,
}

impl ProcessingMethodMediatorBoundary {
    /// Java `register(ProcessInterface)`.
    pub fn register(&mut self) {
        self.registered = true;
    }
    /// Java `deregister(ProcessInterface)`.
    pub fn deregister(&mut self) {
        self.registered = false;
    }
    /// Java `setMethod(ProcessInterface, ProcessingMethod)`.
    pub fn set_method(&mut self, alt_stack_interface: bool, method: ProcessingMethod) {
        self.method_calls.push((alt_stack_interface, method));
    }
    /// Java `getRunMethodForProcessInterface(ProcessingMethod)`.
    pub fn get_run_method_for_process_interface(
        &self,
        method: ProcessingMethod,
    ) -> ProcessingMethod {
        self.run_method.unwrap_or(method)
    }
}

/// Java final `CpuGpuPanel` fields.  The boolean component hierarchy records
/// the source's `JPanel`/`BoxLayout` construction at the native-GUI boundary.
pub struct CpuGpuPanel {
    pub pnl_root_box_layout_axis: i32,
    pub pnl_root_component_order: Vec<&'static str>,
    pub cb_parallel_process: CheckBox,
    pub cb_use_gpu: CheckBox,
    pub axis_id: AxisID,
    pub mediator: ProcessingMethodMediatorBoundary,
    pub l_max_gpus: Option<String>,
    pub l_max_gpus_enabled: bool,
    pub l_max_gpus_visible: bool,
    pub always_parallel: bool,
    pub processing_method_locked: bool,
    pub gpus_available: bool,
    pub non_local_host_gpus_available: bool,
    pub local_gpu_available: bool,
    pub gpu_enabled: bool,
    pub use_max_gpus: bool,
    pub gpus_for_queue_available: bool,
    pub max_gpus: i32,
    pub use_queue_check_box: Option<CheckBox>,
    pub non_queue_gpu_checkbox_status: bool,
    pub alt_stack_process_interface: bool,
    pub local_gpus_int: i32,
    pub max_tilt_cpus: Option<String>,
    pub listeners_added: bool,
}

impl CpuGpuPanel {
    /// Java private `CpuGpuPanel(BaseManager, AxisID, int)`.  Network and
    /// CpuAdoc lookups are supplied to `get_instance` as their explicit boundary values.
    pub fn new(axis_id: AxisID, max_gpu: i32, total_gpus: i32) -> Self {
        let l_max_gpus = if max_gpu != -1 && total_gpus > 1 {
            Some(format!(": Maximum number of GPUs recommended is {max_gpu}"))
        } else {
            None
        };
        let mut mediator = ProcessingMethodMediatorBoundary::default();
        mediator.register();
        Self {
            pnl_root_box_layout_axis: 1,
            pnl_root_component_order: vec![],
            cb_parallel_process: CheckBox::new_with_text(FIELD_LABEL),
            cb_use_gpu: CheckBox::new_with_text("Use the GPU"),
            axis_id,
            mediator,
            l_max_gpus,
            l_max_gpus_enabled: true,
            l_max_gpus_visible: true,
            always_parallel: false,
            processing_method_locked: false,
            gpus_available: false,
            non_local_host_gpus_available: false,
            local_gpu_available: true,
            gpu_enabled: true,
            use_max_gpus: true,
            gpus_for_queue_available: false,
            max_gpus: max_gpu,
            use_queue_check_box: None,
            non_queue_gpu_checkbox_status: false,
            alt_stack_process_interface: false,
            local_gpus_int: 0,
            max_tilt_cpus: None,
            listeners_added: false,
        }
    }

    /// Java static `getInstance` plus `createPanel`, `setTooltips`, `init`, and `addListeners`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance(
        axis_id: AxisID,
        panel_id: PanelId,
        max_gpu: i32,
        use_max_cpu: bool,
        box_layout_axis: i32,
        total_gpus: i32,
        non_local_only_gpu: bool,
        any_queue_gpu: bool,
        non_local_host_gpu: bool,
        local_host_gpu: bool,
        max_tilt_cpus: Option<&str>,
    ) -> Self {
        let mut instance = Self::new(axis_id, max_gpu, total_gpus);
        instance.max_tilt_cpus = max_tilt_cpus.map(str::to_owned);
        instance.create_panel(box_layout_axis);
        instance.set_tooltips();
        instance.init(
            panel_id,
            use_max_cpu,
            non_local_only_gpu,
            any_queue_gpu,
            non_local_host_gpu,
            local_host_gpu,
        );
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(int)`; `0` and `1` are BoxLayout X/Y axes.
    pub fn create_panel(&mut self, mut box_layout_axis: i32) {
        if box_layout_axis != 0 {
            box_layout_axis = 1;
        }
        self.pnl_root_box_layout_axis = box_layout_axis;
        self.pnl_root_component_order = vec!["pnlParallelProcess", "pnlUseGpu"];
    }

    /// Java private `init(PanelId, boolean)` with Network query results passed directly.
    pub fn init(
        &mut self,
        panel_id: PanelId,
        use_max_cpu: bool,
        non_local_only_gpu: bool,
        any_queue_gpu: bool,
        non_local_host_gpu: bool,
        local_host_gpu: bool,
    ) {
        if matches!(panel_id, PanelId::Tilt | PanelId::Tilt3dFind) && use_max_cpu {
            if let Some(max_cpus) = self
                .max_tilt_cpus
                .as_deref()
                .filter(|value| !value.is_empty())
            {
                self.cb_parallel_process
                    .set_text(Some(&format!("{FIELD_LABEL}{MAX_CPUS_STRING}{max_cpus}")));
            }
        }
        self.gpus_available = non_local_only_gpu;
        self.gpus_for_queue_available = any_queue_gpu;
        self.non_local_host_gpus_available = non_local_host_gpu;
        self.local_gpu_available = local_host_gpu;
        self.update_display();
        self.mediator
            .set_method(false, self.get_processing_method());
    }

    /// Java private `isGpusAvailable`.
    pub fn is_gpus_available(&self) -> bool {
        self.use_queue_check_box
            .as_ref()
            .is_some_and(CheckBox::is_selected)
            .then_some(self.gpus_for_queue_available)
            .unwrap_or(self.gpus_available)
    }
    /// Java private `isLocalGpuAvailable`.
    pub fn is_local_gpu_available(&self) -> bool {
        !self
            .use_queue_check_box
            .as_ref()
            .is_some_and(CheckBox::is_selected)
            && self.local_gpu_available
    }
    /// Java `reregisterProcessingMethodMediator`.
    pub fn reregister_processing_method_mediator(&mut self) {
        self.mediator.register();
        self.mediator
            .set_method(false, self.get_processing_method());
    }
    /// Java `isParallelProcess`.
    pub fn is_parallel_process(&self) -> bool {
        self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected()
    }
    /// Java `setParameters(ConstMetaData, ConstEtomoNumber)`; `parallel_process` is Java-nullable.
    pub fn set_parameters_metadata(
        &mut self,
        meta_data: ConstMetaDataBoundary,
        parallel_process: Option<bool>,
    ) {
        self.cb_use_gpu
            .set_selected(meta_data.default_gpu_processing);
        if self.non_local_host_gpus_available || !self.cb_use_gpu.is_selected() {
            self.cb_parallel_process
                .set_selected(parallel_process.unwrap_or(meta_data.default_parallel));
        }
        self.update_display();
        self.mediator.set_method(
            self.alt_stack_process_interface,
            self.get_processing_method(),
        );
    }
    /// Java `getParameters(PanelId, MetaData)`.
    pub fn get_parameters_metadata(&self, _panel_id: PanelId, meta_data: &mut MetaDataBoundary) {
        meta_data.tilt_parallel = Some(self.is_parallel_process());
    }
    /// Java `setParameters(ConstTiltParam, boolean)`.
    pub fn set_parameters_tilt(&mut self, tilt_param: TiltParamBoundary, initialize: bool) {
        if !initialize {
            self.cb_use_gpu.set_selected(tilt_param.use_gpu);
        }
        self.update_display();
        self.mediator.set_method(
            self.alt_stack_process_interface,
            self.get_processing_method(),
        );
    }
    /// Java `getParameters(TiltParam)`.
    pub fn get_parameters_tilt(&self, tilt_param: &mut TiltParamBoundary) {
        tilt_param.use_gpu = self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected();
    }
    /// Java `getParameters(CtfPhaseFlipParam)`.
    pub fn get_parameters_ctf_phase_flip(&self, param: &mut CtfPhaseFlipParamBoundary) {
        param.use_gpu = self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected();
    }
    /// Java `setParameters(ConstCtfPhaseFlipParam, boolean)`.
    pub fn set_parameters_ctf_phase_flip(
        &mut self,
        param: CtfPhaseFlipParamBoundary,
        initialize: bool,
    ) {
        if !initialize {
            self.cb_use_gpu.set_selected(param.use_gpu);
        }
        self.update_display();
        self.mediator
            .set_method(false, self.get_processing_method());
    }
    /// Java `addListeners` (Swing listener installation remains a boundary).
    pub fn add_listeners(&mut self) {
        self.listeners_added = true;
        self.mediator.gpu_listener_count += 1;
    }
    /// Java `setUseQueueCheckBox(ButtonComponent)`.
    pub fn set_use_queue_check_box(&mut self, use_queue_check_box: Option<CheckBox>) {
        if use_queue_check_box.is_some() && self.use_queue_check_box.is_none() {
            self.use_queue_check_box = use_queue_check_box;
        }
    }
    /// Java no-op `queueTableEventAction`.
    pub fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}
    /// Java no-op `addQueueTableListener`.
    pub fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    /// Java no-op `removeQueueTableListener`.
    pub fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    /// Java `getComponent`; Swing component identity is represented by root order.
    pub fn get_component(&self) -> &[&'static str] {
        &self.pnl_root_component_order
    }
    /// Java `actionPerformed(ActionEvent)` with `action_command` supplied by Swing.
    pub fn action_performed(&mut self, action_command: Option<&str>) {
        if self.local_gpus_int <= 1 {
            if self.cb_parallel_process.get_action_command() == action_command {
                if self.cb_parallel_process.is_selected()
                    && !self.non_local_host_gpus_available
                    && self.cb_use_gpu.is_selected()
                {
                    self.cb_use_gpu.set_selected(false);
                }
            } else if self.cb_use_gpu.get_action_command() == action_command {
                if self.cb_use_gpu.is_selected()
                    && !self.non_local_host_gpus_available
                    && self.cb_parallel_process.is_selected()
                {
                    self.cb_parallel_process.set_selected(false);
                }
            } else if self
                .use_queue_check_box
                .as_ref()
                .and_then(CheckBox::get_action_command)
                == action_command
            {
                if self
                    .use_queue_check_box
                    .as_ref()
                    .is_some_and(CheckBox::is_selected)
                {
                    self.non_queue_gpu_checkbox_status = self.cb_use_gpu.is_selected();
                } else {
                    self.cb_use_gpu
                        .set_selected(self.non_queue_gpu_checkbox_status);
                }
            }
        }
        self.update_display();
        self.mediator.set_method(
            self.alt_stack_process_interface,
            self.get_processing_method(),
        );
    }
    /// Java `getRunMethodForProcessInterface`.
    pub fn get_run_method_for_process_interface(&self) -> ProcessingMethod {
        self.mediator
            .get_run_method_for_process_interface(self.get_processing_method())
    }
    /// Java `msgProcessingMethodChanged(boolean, boolean)`.
    pub fn msg_processing_method_changed(&mut self, always_parallel: bool, use_max_gpus: bool) {
        self.always_parallel = always_parallel;
        self.use_max_gpus = use_max_gpus;
        self.update_display();
        self.mediator
            .set_method(false, self.get_processing_method());
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.mediator.deregister();
    }
    /// Java `lockProcessingMethod`.
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.processing_method_locked = lock;
        self.update_display();
    }
    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        self.cb_parallel_process.set_visible(!self.always_parallel);
        self.cb_parallel_process
            .set_enabled(!self.processing_method_locked);
        let parallel = (self.cb_parallel_process.is_enabled()
            && self.cb_parallel_process.is_selected())
            || self.always_parallel;
        let enabled = ((self.is_gpus_available() || self.is_local_gpu_available()) && parallel
            || (self.is_local_gpu_available()
                && !self.cb_parallel_process.is_selected()
                && !self.always_parallel))
            && !self.processing_method_locked;
        self.cb_use_gpu.set_enabled(enabled);
        if self.l_max_gpus.is_some() {
            self.l_max_gpus_enabled = enabled;
            self.l_max_gpus_visible = self.use_max_gpus;
        }
    }
    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        let parallel = self.always_parallel
            || (self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected());
        if parallel {
            return if self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected() {
                ProcessingMethod::PpGpu
            } else {
                ProcessingMethod::PpCpu
            };
        }
        if self.local_gpu_available && self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected()
        {
            ProcessingMethod::LocalGpu
        } else {
            ProcessingMethod::LocalCpu
        }
    }
    /// Java `isProcessingMethodValid`.
    pub fn is_processing_method_valid(&self) -> bool {
        !self.always_parallel
    }
    /// Java `getSecondaryProcessingMethod`.
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    /// Java private `setTooltips`.
    pub fn set_tooltips(&mut self) {
        self.cb_parallel_process.set_tool_tip_text(Some(
            "Check to distribute the process across multiple computers.",
        ));
        self.cb_use_gpu.set_tool_tip_text(Some(
            "Check to run the process on one or more graphics cards.",
        ));
    }
    /// Java misspelled `registerProcesingMethodMediator`.
    pub fn register_procesing_method_mediator(&mut self) {
        self.mediator.register();
    }
    /// Java misspelled `deregisterProcesingMethodMediator`.
    pub fn deregister_procesing_method_mediator(&mut self) {
        self.mediator.deregister();
    }
    /// Java `setMethod`.
    pub fn set_method(&mut self, processing_method: ProcessingMethod) {
        self.mediator.set_method(false, processing_method);
    }
    /// Java `isUseGpu`.
    pub fn is_use_gpu(&self) -> bool {
        self.cb_use_gpu.is_enabled() && self.cb_use_gpu.is_selected()
    }
    /// Java `updateGpu`; source intentionally ignores `disableGpu`.
    pub fn update_gpu(&mut self, _disable_gpu: bool) {
        self.update_display();
    }
    /// Java `setAltStackProcessInterface`.
    pub fn set_alt_stack_process_interface(&mut self, origin_present: bool) {
        self.alt_stack_process_interface = origin_present;
    }
}

impl QueueTableListener for CpuGpuPanel {
    fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        Self::queue_table_event_action(self, event);
    }
}

impl ProcessInterface for CpuGpuPanel {
    type QueueCheckBox = CheckBox;

    fn update_gpu(&mut self, disable_gpu: bool) {
        Self::update_gpu(self, disable_gpu);
    }

    fn get_processing_method(&self) -> ProcessingMethod {
        Self::get_processing_method(self)
    }

    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        Self::get_secondary_processing_method(self)
    }

    fn lock_processing_method(&mut self, lock: bool) {
        Self::lock_processing_method(self, lock);
    }

    fn set_method(&mut self, processing_method: ProcessingMethod) {
        Self::set_method(self, processing_method);
    }

    fn is_use_gpu(&self) -> bool {
        Self::is_use_gpu(self)
    }

    fn set_use_queue_check_box(&mut self, use_queue_checkbox: Option<CheckBox>) {
        Self::set_use_queue_check_box(self, use_queue_checkbox);
    }

    fn add_queue_table_listener(&mut self, listener: &mut dyn QueueTableListener) {
        Self::add_queue_table_listener(self, listener);
    }

    fn remove_queue_table_listener(&mut self, listener: &mut dyn QueueTableListener) {
        Self::remove_queue_table_listener(self, listener);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_gpu_and_parallel_gpu_follow_source_enablement() {
        let mut panel = CpuGpuPanel::get_instance(
            AxisID::Only,
            PanelId::Tilt,
            2,
            false,
            1,
            2,
            false,
            false,
            false,
            true,
            None,
        );
        panel.cb_use_gpu.set_selected(true);
        panel.update_display();
        assert_eq!(panel.get_processing_method(), ProcessingMethod::LocalGpu);
        panel.cb_parallel_process.set_selected(true);
        panel.update_display();
        assert!(panel.cb_use_gpu.is_enabled());
        assert_eq!(panel.get_processing_method(), ProcessingMethod::PpGpu);
    }
    #[test]
    fn queue_selection_saves_and_restores_non_queue_gpu_state() {
        let mut panel = CpuGpuPanel::new(AxisID::Only, -1, 0);
        let mut queue = CheckBox::new_with_text("Use a queue");
        queue.set_selected(true);
        let command = queue.get_action_command().unwrap().to_owned();
        panel.cb_use_gpu.set_selected(true);
        panel.set_use_queue_check_box(Some(queue));
        panel.action_performed(Some(&command));
        assert!(panel.non_queue_gpu_checkbox_status);
        panel
            .use_queue_check_box
            .as_mut()
            .unwrap()
            .set_selected(false);
        panel.cb_use_gpu.set_selected(false);
        panel.action_performed(Some(&command));
        assert!(panel.cb_use_gpu.is_selected());
    }
}
