//! `IMOD/Etomo/src/etomo/ui/swing/SubtomoSetupCpuGpuPanel.java`.
//!
//! Swing construction and listener dispatch, Network GPU discovery, the
//! SubtomoSetup autodoc, the processing-method mediator, and the owning
//! `SubtomogramsPanel` remain explicit boundaries.  This unit retains the
//! source panel's CPU/GPU selection and queue transition policy.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use super::check_box::CheckBox;
use super::cpu_gpu_panel::ProcessingMethodMediatorBoundary;
use super::process_interface::ProcessInterface;
use super::radio_button::{RadioButton, RadioButtonGroup};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

pub const USE_CPUS_ONLY: &str = "Use CPU cores only";
pub const USE_GPU_FOR_RECON_AND_CTF_CORRECT: &str =
    "Use the GPU for reconstruction and CTF correction";
pub const USE_CPU_FOR_RECON_AND_GPU_FOR_CTF_CORRECT: &str =
    "Use CPU cores for reconstruction and one GPU for CTF correction";

/// Java `ApplicationManager`, `Network`, `AutodocFactory`, and
/// `EtomoAutodoc.getTooltip` calls made by this source unit.
pub trait SubtomoSetupCpuGpuApplicationManager {
    fn get_processing_method_mediator(&self, axis_id: AxisID) -> ProcessingMethodMediatorBoundary;
    fn is_local_host_gpu_processing_enabled(&self, axis_id: AxisID) -> bool;
    /// `None` represents the source's caught autodoc read/lock failure or a
    /// missing autodoc.  The three strings are values 0, 1, and 2 of
    /// `WhenToUseGPU` after `EtomoAutodoc.getTooltip` formatting.
    fn subtomo_setup_when_to_use_gpu_tooltips(&self) -> Option<[String; 3]>;
    /// Java `Network.isAnyQueueGpu()`.
    fn is_any_queue_gpu(&self) -> bool;
}

/// Java `SubtomogramsPanel.isExtentOfZLevelsInNm` callback.
pub trait SubtomoSetupSubtomogramsPanel {
    fn is_extent_of_z_levels_in_nm(&self) -> bool;
}

/// Java private `CpuGpuSelection`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuGpuSelection {
    CpuOnly,
    GpuOnly,
    MixCpuGpu,
}

/// Source-visible `JPanel` / `BoxLayout` hierarchy created by
/// `createSubtomoSetupPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SubtomoSetupCpuGpuPanelLayout {
    pub root_box_layout_x_axis: bool,
    pub root_component_order: Vec<String>,
    pub processing_resources_box_layout_y_axis: bool,
    pub processing_resources_border_title: Option<String>,
    pub processing_resources_component_order: Vec<String>,
    pub cpus_only_box_layout_x_axis: bool,
    pub cpus_only_component_order: Vec<String>,
    pub gpu_for_recon_and_ctf_correct_box_layout_x_axis: bool,
    pub gpu_for_recon_and_ctf_correct_component_order: Vec<String>,
    pub cpu_for_recon_and_gpu_for_ctf_correct_box_layout_x_axis: bool,
    pub cpu_for_recon_and_gpu_for_ctf_correct_component_order: Vec<String>,
}

/// Java final `SubtomoSetupCpuGpuPanel` fields, parameterized at its direct
/// application, process, and owning-panel boundaries.
pub struct SubtomoSetupCpuGpuPanel<M, S, P> {
    pub pnl_root: SubtomoSetupCpuGpuPanelLayout,
    pub bg_processing_resources_to_use: Rc<RefCell<RadioButtonGroup>>,
    pub rb_cpus_only: RadioButton,
    pub rb_gpu_for_recon_and_ctf_correct: RadioButton,
    pub rb_cpu_for_recon_and_gpu_for_ctf_correct: RadioButton,
    pub manager: M,
    pub axis_id: AxisID,
    pub process_interface: P,
    pub mediator: ProcessingMethodMediatorBoundary,
    pub subtomograms_panel: S,
    pub local_gpu_available: bool,
    pub use_queue_check_box: Option<CheckBox>,
    pub gpus_for_queue_available: bool,
    pub mix_cpu_gpu_action_event: bool,
    pub curr_cpu_gpu_selection: Option<CpuGpuSelection>,
    pub listeners_added: bool,
}

impl<M, S, P> SubtomoSetupCpuGpuPanel<M, S, P>
where
    M: SubtomoSetupCpuGpuApplicationManager,
    S: SubtomoSetupSubtomogramsPanel,
    P: ProcessInterface<QueueCheckBox = CheckBox>,
{
    /// Java private constructor.
    pub fn new(subtomograms_panel: S, manager: M, axis_id: AxisID, process_interface: P) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mediator = manager.get_processing_method_mediator(axis_id);
        let local_gpu_available = manager.is_local_host_gpu_processing_enabled(axis_id);
        Self {
            pnl_root: SubtomoSetupCpuGpuPanelLayout::default(),
            bg_processing_resources_to_use: group.clone(),
            rb_cpus_only: RadioButton::new_in_group(USE_CPUS_ONLY, group.clone()),
            rb_gpu_for_recon_and_ctf_correct: RadioButton::new_in_group(
                USE_GPU_FOR_RECON_AND_CTF_CORRECT,
                group.clone(),
            ),
            rb_cpu_for_recon_and_gpu_for_ctf_correct: RadioButton::new_in_group(
                USE_CPU_FOR_RECON_AND_GPU_FOR_CTF_CORRECT,
                group,
            ),
            manager,
            axis_id,
            process_interface,
            mediator,
            subtomograms_panel,
            local_gpu_available,
            use_queue_check_box: None,
            gpus_for_queue_available: false,
            mix_cpu_gpu_action_event: false,
            curr_cpu_gpu_selection: None,
            listeners_added: false,
        }
    }

    /// Java static `getSubtomoSetupInstance`.
    pub fn get_subtomo_setup_instance(
        subtomograms_panel: S,
        manager: M,
        axis_id: AxisID,
        process_interface: P,
    ) -> Self {
        let mut instance = Self::new(subtomograms_panel, manager, axis_id, process_interface);
        instance.create_subtomo_setup_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createSubtomoSetupPanel`.
    pub fn create_subtomo_setup_panel(&mut self) {
        self.rb_cpus_only.set_selected(true);
        self.process_interface
            .set_method(self.get_processing_method());
        self.pnl_root.root_box_layout_x_axis = true;
        self.pnl_root.root_component_order = vec!["pnlProcessingResourcesToUse".into()];
        self.pnl_root.processing_resources_box_layout_y_axis = true;
        self.pnl_root.processing_resources_border_title =
            Some("Processing resources to use".into());
        self.pnl_root.processing_resources_component_order = vec![
            "pnlCpusOnly".into(),
            "pnlGpuForReconAndCtfCorrect".into(),
            "pnlCpuForReconAndGpuForCtfCorrect".into(),
        ];
        self.pnl_root.cpus_only_box_layout_x_axis = true;
        self.pnl_root.cpus_only_component_order =
            vec!["rbCpusOnly".into(), "horizontalGlue".into()];
        self.pnl_root
            .gpu_for_recon_and_ctf_correct_box_layout_x_axis = true;
        self.pnl_root.gpu_for_recon_and_ctf_correct_component_order =
            vec!["rbGpuForReconAndCtfCorrect".into(), "horizontalGlue".into()];
        self.pnl_root
            .cpu_for_recon_and_gpu_for_ctf_correct_box_layout_x_axis = true;
        self.pnl_root
            .cpu_for_recon_and_gpu_for_ctf_correct_component_order = vec![
            "rbCpuForReconAndGpuForCtfCorrect".into(),
            "horizontalGlue".into(),
        ];
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.rb_cpus_only.add_action_listener();
        self.rb_gpu_for_recon_and_ctf_correct.add_action_listener();
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct
            .add_action_listener();
        self.mediator.gpu_listener_count += 3;
        self.listeners_added = true;
    }

    /// Java `getComponent`; native component identity is the retained layout.
    pub fn get_component(&self) -> &SubtomoSetupCpuGpuPanelLayout {
        &self.pnl_root
    }

    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.rb_gpu_for_recon_and_ctf_correct.is_selected() {
            ProcessingMethod::PpGpu
        } else {
            ProcessingMethod::PpCpu
        }
    }

    /// Java `actionPerformed(ActionEvent)` with Swing's action command passed
    /// through this native-GUI boundary.
    pub fn action_performed(&mut self, action_command: &str) {
        if self.rb_cpus_only.get_action_command() == action_command {
            self.process_interface
                .set_method(self.get_processing_method());
        } else if self.rb_gpu_for_recon_and_ctf_correct.get_action_command() == action_command {
            self.process_interface
                .set_method(self.get_processing_method());
        } else if self
            .rb_cpu_for_recon_and_gpu_for_ctf_correct
            .get_action_command()
            == action_command
        {
            self.mix_cpu_gpu_action_event = true;
            self.process_interface
                .set_method(self.get_processing_method());
            self.mix_cpu_gpu_action_event = false;
        } else if self
            .use_queue_check_box
            .as_ref()
            .and_then(CheckBox::get_action_command)
            == Some(action_command)
        {
            if self
                .use_queue_check_box
                .as_ref()
                .is_some_and(CheckBox::is_selected)
            {
                self.set_curr_cpu_gpu_selection();
                self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                    .set_enabled(false);
                if self.manager.is_any_queue_gpu() {
                    self.rb_gpu_for_recon_and_ctf_correct.set_selected(true);
                    self.process_interface
                        .set_method(self.get_processing_method());
                } else {
                    self.rb_cpus_only.set_selected(true);
                    self.rb_gpu_for_recon_and_ctf_correct.set_enabled(false);
                }
            } else {
                self.rb_gpu_for_recon_and_ctf_correct.set_enabled(true);
                if self.subtomograms_panel.is_extent_of_z_levels_in_nm() {
                    self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                        .set_enabled(true);
                }
                self.revert_to_non_queue_cpu_gpu_selection();
            }
        }
        self.update_display();
    }

    /// Java `setTooltips`; errors caught by Java are represented by no tooltip
    /// data at the storage/autodoc boundary.
    pub fn set_tooltips(&mut self) {
        if let Some([cpu_only, gpu_only, mix_cpu_gpu]) =
            self.manager.subtomo_setup_when_to_use_gpu_tooltips()
        {
            self.rb_cpus_only.set_tool_tip_text(Some(&cpu_only));
            self.rb_gpu_for_recon_and_ctf_correct
                .set_tool_tip_text(Some(&gpu_only));
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                .set_tool_tip_text(Some(&mix_cpu_gpu));
        }
    }

    /// Java `isSelectedCpusOnly`.
    pub fn is_selected_cpus_only(&self) -> bool {
        self.rb_cpus_only.is_selected()
    }
    /// Java `setSelectedCpusOnly`.
    pub fn set_selected_cpus_only(&mut self, selected: bool) {
        self.rb_cpus_only.set_selected(selected);
        self.process_interface
            .set_method(self.get_processing_method());
    }
    /// Java `isSelectedGpuForReconAndCtfCorrect`.
    pub fn is_selected_gpu_for_recon_and_ctf_correct(&self) -> bool {
        self.rb_gpu_for_recon_and_ctf_correct.is_selected()
    }
    /// Java `setSelectedGpuForReconAndCtfCorrect`.
    pub fn set_selected_gpu_for_recon_and_ctf_correct(&mut self, selected: bool) {
        self.rb_gpu_for_recon_and_ctf_correct.set_selected(selected);
        self.process_interface
            .set_method(self.get_processing_method());
    }
    /// Java `setEnabledGpuForReconAndCtfCorrect`.
    pub fn set_enabled_gpu_for_recon_and_ctf_correct(&mut self, enabled: bool) {
        self.rb_gpu_for_recon_and_ctf_correct.set_enabled(enabled);
    }
    /// Java `isSelectedCpuForReconAndGpuForCtfCorrect`.
    pub fn is_selected_cpu_for_recon_and_gpu_for_ctf_correct(&self) -> bool {
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct.is_selected()
    }
    /// Java `setSelectedCpuForReconAndGpuForCtfCorrect`.
    pub fn set_selected_cpu_for_recon_and_gpu_for_ctf_correct(&mut self, selected: bool) {
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct
            .set_selected(selected);
        self.process_interface
            .set_method(self.get_processing_method());
    }
    /// Java `setEnabledCpuForReconAndGpuForCtfCorrect`.
    pub fn set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(&mut self, enabled: bool) {
        if self
            .use_queue_check_box
            .as_ref()
            .is_none_or(|check_box| !check_box.is_selected())
        {
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                .set_enabled(enabled);
        }
    }
    /// Java `updateGpu`; source deliberately ignores `disableGpu`.
    pub fn update_gpu(&mut self, _disable_gpu: bool) {
        self.update_display();
    }
    /// Java `getSecondaryProcessingMethod`.
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    /// Java no-op `lockProcessingMethod`.
    pub fn lock_processing_method(&mut self, _lock: bool) {}
    /// Java no-op `setMethod`.
    pub fn set_method(&mut self, _processing_method: ProcessingMethod) {}
    /// Java `isUseGpu`.
    pub fn is_use_gpu(&self) -> bool {
        self.rb_gpu_for_recon_and_ctf_correct.is_enabled()
            && self.rb_gpu_for_recon_and_ctf_correct.is_selected()
    }

    /// Java `setUseQueueCheckBox`.
    pub fn set_use_queue_check_box(&mut self, use_queue_check_box: Option<CheckBox>) {
        if use_queue_check_box.is_some() && self.use_queue_check_box.is_none() {
            let mut use_queue_check_box = use_queue_check_box.unwrap();
            use_queue_check_box.add_action_listener();
            self.use_queue_check_box = Some(use_queue_check_box);
        }
    }

    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        self.rb_cpu_for_recon_and_gpu_for_ctf_correct.set_enabled(
            (self.use_queue_check_box.is_none()
                || !self
                    .use_queue_check_box
                    .as_ref()
                    .is_some_and(CheckBox::is_selected))
                && self.subtomograms_panel.is_extent_of_z_levels_in_nm(),
        );
    }

    /// Java private `setCurrCpuGpuSelection`.
    pub fn set_curr_cpu_gpu_selection(&mut self) {
        self.curr_cpu_gpu_selection = Some(if self.rb_cpus_only.is_selected() {
            CpuGpuSelection::CpuOnly
        } else if self.rb_gpu_for_recon_and_ctf_correct.is_selected() {
            CpuGpuSelection::GpuOnly
        } else {
            CpuGpuSelection::MixCpuGpu
        });
    }

    /// Java private `revertToNonQueueCpuGpuSelection`.
    pub fn revert_to_non_queue_cpu_gpu_selection(&mut self) {
        if self.curr_cpu_gpu_selection == Some(CpuGpuSelection::CpuOnly) {
            self.rb_cpus_only.set_selected(true);
            self.process_interface
                .set_method(self.get_processing_method());
        } else if self.curr_cpu_gpu_selection == Some(CpuGpuSelection::GpuOnly) {
            self.rb_gpu_for_recon_and_ctf_correct.set_selected(true);
            self.process_interface
                .set_method(self.get_processing_method());
        } else if self.subtomograms_panel.is_extent_of_z_levels_in_nm() {
            self.rb_cpu_for_recon_and_gpu_for_ctf_correct
                .set_selected(true);
            self.process_interface
                .set_method(self.get_processing_method());
        }
    }

    /// Java `isMixCpuGpuActionEvent`.
    pub fn is_mix_cpu_gpu_action_event(&self) -> bool {
        self.mix_cpu_gpu_action_event
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Manager {
        queue_gpu: bool,
    }
    impl SubtomoSetupCpuGpuApplicationManager for Manager {
        fn get_processing_method_mediator(
            &self,
            _axis_id: AxisID,
        ) -> ProcessingMethodMediatorBoundary {
            ProcessingMethodMediatorBoundary::default()
        }
        fn is_local_host_gpu_processing_enabled(&self, _axis_id: AxisID) -> bool {
            true
        }
        fn subtomo_setup_when_to_use_gpu_tooltips(&self) -> Option<[String; 3]> {
            Some(["cpu".into(), "gpu".into(), "mix".into()])
        }
        fn is_any_queue_gpu(&self) -> bool {
            self.queue_gpu
        }
    }
    struct Subtomograms {
        extent_nm: bool,
    }
    impl SubtomoSetupSubtomogramsPanel for Subtomograms {
        fn is_extent_of_z_levels_in_nm(&self) -> bool {
            self.extent_nm
        }
    }
    #[derive(Default)]
    struct Process {
        methods: Vec<ProcessingMethod>,
    }
    impl crate::imod::etomo::ui::queue_table_listener::QueueTableListener for Process {
        fn queue_table_event_action(
            &mut self,
            _event: crate::imod::etomo::ui::queue_table_event::QueueTableEvent,
        ) {
        }
    }
    impl ProcessInterface for Process {
        type QueueCheckBox = CheckBox;
        fn update_gpu(&mut self, _disable_gpu: bool) {}
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::PpCpu
        }
        fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
            None
        }
        fn lock_processing_method(&mut self, _lock: bool) {}
        fn set_method(&mut self, method: ProcessingMethod) {
            self.methods.push(method);
        }
        fn is_use_gpu(&self) -> bool {
            false
        }
        fn set_use_queue_check_box(&mut self, _use_queue_checkbox: Option<CheckBox>) {}
        fn add_queue_table_listener(
            &mut self,
            _listener: &mut dyn crate::imod::etomo::ui::queue_table_listener::QueueTableListener,
        ) {
        }
        fn remove_queue_table_listener(
            &mut self,
            _listener: &mut dyn crate::imod::etomo::ui::queue_table_listener::QueueTableListener,
        ) {
        }
    }
    fn panel(
        queue_gpu: bool,
        extent_nm: bool,
    ) -> SubtomoSetupCpuGpuPanel<Manager, Subtomograms, Process> {
        SubtomoSetupCpuGpuPanel::get_subtomo_setup_instance(
            Subtomograms { extent_nm },
            Manager { queue_gpu },
            AxisID::Only,
            Process::default(),
        )
    }

    #[test]
    fn construction_selects_cpu_installs_gpu_listeners_and_tooltips() {
        let panel = panel(true, true);
        assert!(panel.is_selected_cpus_only());
        assert_eq!(
            panel.process_interface.methods,
            vec![ProcessingMethod::PpCpu]
        );
        assert_eq!(panel.mediator.gpu_listener_count, 3);
        assert_eq!(
            panel.rb_gpu_for_recon_and_ctf_correct.get_tooltip(),
            Some("<html>gpu")
        );
        assert_eq!(
            panel.pnl_root.processing_resources_border_title.as_deref(),
            Some("Processing resources to use")
        );
    }

    #[test]
    fn queue_gpu_saves_gpu_selection_then_restores_it() {
        let mut panel = panel(true, true);
        panel.set_selected_gpu_for_recon_and_ctf_correct(true);
        let mut queue = CheckBox::new_with_text("Use queue");
        let command = queue.get_action_command().unwrap().to_owned();
        queue.set_selected(true);
        panel.set_use_queue_check_box(Some(queue));
        panel.action_performed(&command);
        assert_eq!(panel.curr_cpu_gpu_selection, Some(CpuGpuSelection::GpuOnly));
        assert!(panel.is_selected_gpu_for_recon_and_ctf_correct());
        panel
            .use_queue_check_box
            .as_mut()
            .unwrap()
            .set_selected(false);
        panel.action_performed(&command);
        assert!(panel.is_selected_gpu_for_recon_and_ctf_correct());
    }

    #[test]
    fn queue_without_gpu_selects_cpu_and_disables_gpu_choice() {
        let mut panel = panel(false, true);
        let mut queue = CheckBox::new_with_text("Use queue");
        let command = queue.get_action_command().unwrap().to_owned();
        queue.set_selected(true);
        panel.set_use_queue_check_box(Some(queue));
        panel.action_performed(&command);
        assert!(panel.is_selected_cpus_only());
        assert!(!panel.rb_gpu_for_recon_and_ctf_correct.is_enabled());
    }
}
