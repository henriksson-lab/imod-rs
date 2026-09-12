//! `IMOD/Etomo/src/etomo/ui/swing/FilterFullVolumePanel.java`.
//!
//! Widget installation and the concrete `ParallelManager` remain GUI/application
//! boundaries.  The fields, parameter transfer, validation order, listener
//! registration, and three action branches are retained from the Java source.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::filter_full_volume_parent::FilterFullVolumeParent;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::tilt_panel::Deferred3dmodButton;

pub const FILTER_FULL_VOLUME_LABEL: &str = "Filter Full Volume";
pub const MEMORY_PER_CHUNK_LABEL: &str = "Memory per chunk";
pub const MEMORY_TO_VOXEL: i32 = 36;
pub const MEMORY_PER_CHUNK_DEFAULT: i32 = 14 * MEMORY_TO_VOXEL;
pub const CLEANUP_LABEL: &str = "Clean Up Subdirectory";

/// Java `Spinner` state and calls used solely by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Spinner {
    pub label: String,
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub step: i32,
    pub tooltip: Option<String>,
}

impl Spinner {
    /// Java `Spinner.getLabeledInstance`.
    pub fn get_labeled_instance(
        label: &str,
        value: i32,
        minimum: i32,
        maximum: i32,
        step: i32,
    ) -> Self {
        Self {
            label: label.into(),
            value,
            minimum,
            maximum,
            step,
            tooltip: None,
        }
    }
    /// Java `Spinner.getValue`.
    pub fn get_value(&self) -> i32 {
        self.value
    }
    /// Java `Spinner.setValue`.
    pub fn set_value(&mut self, value: i32) {
        self.value = value;
    }
    /// Java `Spinner.setToolTipText`.
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.tooltip = Some(text.into());
    }
}

/// Java `ParallelMetaData` calls performed by this panel.
pub trait ParallelMetaData {
    fn set_k_value(&mut self, value: String);
    fn set_iteration(&mut self, value: i32);
    fn set_memory_per_chunk(&mut self, value: i32);
    fn set_overlap_times_four(&mut self, value: bool);
    fn k_value(&self) -> String;
    fn iteration(&self) -> i32;
    fn memory_per_chunk(&self) -> i32;
    fn overlap_times_four(&self) -> bool;
}

/// Java `AnisotropicDiffusionParam` calls performed by this panel.
pub trait AnisotropicDiffusionParam {
    fn set_k_value(&mut self, value: String);
    fn set_iteration(&mut self, value: i32);
}

/// Java `ChunksetupParam` calls performed by this panel.
pub trait ChunksetupParam {
    fn set_memory_per_chunk(&mut self, value: i32);
    fn set_overlap(&mut self, value: i32);
    fn set_overlap_times_four(&mut self, value: bool);
}

/// The direct `ParallelManager` and processing-mediator dispatch boundary.
pub trait FilterFullVolumePanelManager {
    fn chunksetup(
        &mut self,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
    );
    fn imod_anisotropic_diffusion_output(
        &mut self,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        load_with_flipping: bool,
    );
    fn anisotropic_diffusion_suffix(&self) -> String;
}

/// Source-visible Swing hierarchy state created by `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FilterFullVolumePanelLayout {
    pub root_component_order: Vec<&'static str>,
    pub fields_component_order: Vec<&'static str>,
    pub buttons_component_order: Vec<&'static str>,
    pub overlap_checkbox_centered: bool,
    pub root_border: Option<String>,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub deferred_button_set: bool,
}

/// Java final `FilterFullVolumePanel` field state.
#[derive(Clone, Debug)]
pub struct FilterFullVolumePanel {
    pub pnl_root: FilterFullVolumePanelLayout,
    pub btn_run_filter_full_volume: MultiLineButton,
    pub ltf_k_value: LabeledTextField,
    pub sp_iteration: Spinner,
    pub sp_memory_per_chunk: Spinner,
    pub btn_view_filtered_volume: MultiLineButton,
    pub btn_cleanup: MultiLineButton,
    pub cb_overlap_times_four: CheckBox,
    pub dialog_type: DialogType,
    pub actions_attached: bool,
}

impl FilterFullVolumePanel {
    /// Java private `FilterFullVolumePanel(ParallelManager, DialogType, FilterFullVolumeParent)`.
    pub fn new(dialog_type: DialogType) -> Self {
        Self {
            pnl_root: FilterFullVolumePanelLayout::default(),
            btn_run_filter_full_volume: MultiLineButton::new_with_label(Some(
                FILTER_FULL_VOLUME_LABEL,
            )),
            ltf_k_value: LabeledTextField::new(FieldType::FloatingPoint, "K value: "),
            sp_iteration: Spinner::get_labeled_instance("Iterations: ", 10, 1, 200, 1),
            sp_memory_per_chunk: Spinner::get_labeled_instance(
                "Memory per chunk (MB): ",
                MEMORY_PER_CHUNK_DEFAULT,
                MEMORY_TO_VOXEL,
                30 * MEMORY_TO_VOXEL,
                MEMORY_TO_VOXEL,
            ),
            btn_view_filtered_volume: MultiLineButton::new_with_label(Some("View Filtered Volume")),
            btn_cleanup: MultiLineButton::new_with_label(Some(CLEANUP_LABEL)),
            cb_overlap_times_four: CheckBox::new_with_text(
                "Overlap chunks by 4 times # of iterations",
            ),
            dialog_type,
            actions_attached: false,
        }
    }

    /// Java `getInstance`, including `createPanel`, `setTooltips`, and `addListeners`.
    pub fn get_instance<M: FilterFullVolumePanelManager>(
        manager: &M,
        dialog_type: DialogType,
    ) -> Self {
        let mut instance = Self::new(dialog_type);
        instance.create_panel();
        instance.set_tooltips(manager);
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_run_filter_full_volume.add_action_listener();
        self.btn_view_filtered_volume.add_action_listener();
        self.btn_cleanup.add_action_listener();
        self.pnl_root.listener_count = 3;
        self.actions_attached = true;
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.ltf_k_value.set_text_preferred_width(4);
        self.ltf_k_value.set_required(true);
        self.pnl_root.root_component_order = vec!["fields", "overlap-times-four", "buttons"];
        self.pnl_root.fields_component_order =
            vec!["k-value", "iterations", "memory-per-chunk", "glue"];
        self.pnl_root.buttons_component_order =
            vec!["run-filter-full-volume", "view-filtered-volume", "cleanup"];
        self.pnl_root.overlap_checkbox_centered = true;
        self.pnl_root.root_border = Some(FILTER_FULL_VOLUME_LABEL.into());
        self.pnl_root.deferred_button_set = true;
    }

    /// Java `getComponent`; the frontend installs this retained layout state.
    pub fn get_component(&self) -> &FilterFullVolumePanelLayout {
        &self.pnl_root
    }

    /// Java overloaded `getParameters(ParallelMetaData)`.
    pub fn get_parameters_meta_data<M: ParallelMetaData>(&self, meta_data: &mut M) {
        meta_data.set_k_value(self.ltf_k_value.text.clone());
        meta_data.set_iteration(self.sp_iteration.get_value());
        meta_data.set_memory_per_chunk(self.sp_memory_per_chunk.get_value());
        meta_data.set_overlap_times_four(self.cb_overlap_times_four.is_selected());
    }

    /// Java `getMemoryPerChunk`.
    pub fn get_memory_per_chunk(&self) -> i32 {
        self.sp_memory_per_chunk.get_value()
    }

    /// Java `setParameters(ParallelMetaData)`.
    pub fn set_parameters_meta_data<M: ParallelMetaData>(&mut self, meta_data: &M) {
        self.ltf_k_value.set_text(&meta_data.k_value());
        self.sp_iteration.set_value(meta_data.iteration());
        self.sp_memory_per_chunk
            .set_value(meta_data.memory_per_chunk());
        self.cb_overlap_times_four
            .set_selected(meta_data.overlap_times_four());
    }

    /// Java overloaded `getParameters(AnisotropicDiffusionParam, boolean)`.
    pub fn get_parameters_anisotropic_diffusion<P: AnisotropicDiffusionParam>(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        match self.ltf_k_value.get_text_validated(do_validation) {
            Ok(value) => {
                param.set_k_value(value);
                param.set_iteration(self.sp_iteration.get_value());
                true
            }
            Err(_) => false,
        }
    }

    /// Java overloaded `getParameters(ChunksetupParam)`.
    pub fn get_parameters_chunksetup<P: ChunksetupParam>(&self, param: &mut P) {
        param.set_memory_per_chunk(self.sp_memory_per_chunk.get_value());
        param.set_overlap(self.sp_iteration.get_value());
        param.set_overlap_times_four(self.cb_overlap_times_four.is_selected());
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: FilterFullVolumePanelManager, P: FilterFullVolumeParent>(
        &self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
        parent: &mut P,
    ) {
        if command
            == self
                .btn_run_filter_full_volume
                .button
                .action_command
                .as_deref()
                .unwrap_or(FILTER_FULL_VOLUME_LABEL)
        {
            if !parent.init_subdir() {
                return;
            }
            manager.chunksetup(
                deferred_3dmod_button,
                run_3dmod_menu_options,
                self.dialog_type,
                parent.get_processing_method(),
            );
        } else if command
            == self
                .btn_cleanup
                .button
                .action_command
                .as_deref()
                .unwrap_or(CLEANUP_LABEL)
        {
            parent.clean_up();
        } else if command
            == self
                .btn_view_filtered_volume
                .button
                .action_command
                .as_deref()
                .unwrap_or("View Filtered Volume")
        {
            manager.imod_anisotropic_diffusion_output(
                run_3dmod_menu_options,
                parent.is_load_with_flipping(),
            );
        }
    }

    /// Java private `setTooltips`.
    pub fn set_tooltips<M: FilterFullVolumePanelManager>(&mut self, manager: &M) {
        self.btn_run_filter_full_volume
            .set_tool_tip_text(Some(&format!(
                "Run diffusion on the full volume in chunks, creates a {}file.",
                manager.anisotropic_diffusion_suffix()
            )));
        self.ltf_k_value
            .set_tool_tip_text(Some("K threshold value for running on full volume"));
        self.sp_iteration
            .set_tool_tip_text("Number of iterations to run on full volume");
        self.sp_memory_per_chunk.set_tool_tip_text("Maximum memory in megabytes to use while running diffusion on one chunk. Reduce if there is less memory per processor or if you want to break the job into more chunks.  The number of voxels in each chunk will be 1/36 of this memory limit or less.");
        self.btn_view_filtered_volume
            .set_tool_tip_text(Some("View filtered volume (filename.nad) in 3dmod"));
        self.btn_cleanup.set_tool_tip_text(Some(
            "Remove subdirectory with all temporary and test files (naddir.filename).",
        ));
        self.cb_overlap_times_four.set_tool_tip_text(Some("Increase overlap to 4 times # of iterations (default is equal to # of iterations) to eliminate minor effects of cutting volume into chunks."));
        self.pnl_root.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
    use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
    use crate::imod::etomo::ui::swing::process_interface::ProcessInterface;

    #[derive(Default)]
    struct Meta {
        k: String,
        iteration: i32,
        memory: i32,
        overlap: bool,
    }
    impl ParallelMetaData for Meta {
        fn set_k_value(&mut self, value: String) {
            self.k = value;
        }
        fn set_iteration(&mut self, value: i32) {
            self.iteration = value;
        }
        fn set_memory_per_chunk(&mut self, value: i32) {
            self.memory = value;
        }
        fn set_overlap_times_four(&mut self, value: bool) {
            self.overlap = value;
        }
        fn k_value(&self) -> String {
            self.k.clone()
        }
        fn iteration(&self) -> i32 {
            self.iteration
        }
        fn memory_per_chunk(&self) -> i32 {
            self.memory
        }
        fn overlap_times_four(&self) -> bool {
            self.overlap
        }
    }
    #[derive(Default)]
    struct Manager {
        calls: Vec<&'static str>,
    }
    impl FilterFullVolumePanelManager for Manager {
        fn chunksetup(
            &mut self,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
            _: ProcessingMethod,
        ) {
            self.calls.push("chunksetup");
        }
        fn imod_anisotropic_diffusion_output(&mut self, _: Option<Run3dmodMenuOptions>, _: bool) {
            self.calls.push("imod");
        }
        fn anisotropic_diffusion_suffix(&self) -> String {
            ".nad".into()
        }
    }
    #[derive(Default)]
    struct Parent {
        initialized: bool,
        cleaned: bool,
        flipping: bool,
    }
    impl QueueTableListener for Parent {
        fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}
    }
    impl ProcessInterface for Parent {
        type QueueCheckBox = CheckBox;
        fn update_gpu(&mut self, _disable_gpu: bool) {}
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::PpCpu
        }
        fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
            None
        }
        fn lock_processing_method(&mut self, _lock: bool) {}
        fn set_method(&mut self, _processing_method: ProcessingMethod) {}
        fn is_use_gpu(&self) -> bool {
            false
        }
        fn set_use_queue_check_box(&mut self, _use_queue_checkbox: Option<CheckBox>) {}
        fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
        fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    }
    impl FilterFullVolumeParent for Parent {
        fn clean_up(&mut self) {
            self.cleaned = true;
        }
        fn get_volume(&self) -> String {
            String::new()
        }
        fn init_subdir(&mut self) -> bool {
            self.initialized
        }
        fn is_load_with_flipping(&self) -> bool {
            self.flipping
        }
    }
    #[derive(Default)]
    struct DiffusionParam {
        k: String,
        iteration: i32,
    }
    impl AnisotropicDiffusionParam for DiffusionParam {
        fn set_k_value(&mut self, value: String) {
            self.k = value;
        }
        fn set_iteration(&mut self, value: i32) {
            self.iteration = value;
        }
    }
    #[derive(Default)]
    struct ChunkParam {
        memory: i32,
        overlap: i32,
        overlap_times_four: bool,
    }
    impl ChunksetupParam for ChunkParam {
        fn set_memory_per_chunk(&mut self, value: i32) {
            self.memory = value;
        }
        fn set_overlap(&mut self, value: i32) {
            self.overlap = value;
        }
        fn set_overlap_times_four(&mut self, value: bool) {
            self.overlap_times_four = value;
        }
    }
    #[test]
    fn creates_source_defaults_and_transfers_metadata() {
        let panel = FilterFullVolumePanel::get_instance(
            &Manager::default(),
            DialogType::AnisotropicDiffusion,
        );
        assert_eq!(panel.get_memory_per_chunk(), 504);
        assert!(panel.ltf_k_value.required);
        assert_eq!(panel.pnl_root.listener_count, 3);
        let mut meta = Meta::default();
        panel.get_parameters_meta_data(&mut meta);
        assert_eq!(
            (meta.iteration, meta.memory, meta.overlap),
            (10, 504, false)
        );
    }
    #[test]
    fn action_retains_init_cleanup_and_view_branches() {
        let panel = FilterFullVolumePanel::get_instance(
            &Manager::default(),
            DialogType::AnisotropicDiffusion,
        );
        let mut manager = Manager::default();
        let mut parent = Parent::default();
        panel.action(
            FILTER_FULL_VOLUME_LABEL,
            None,
            None,
            &mut manager,
            &mut parent,
        );
        assert!(manager.calls.is_empty());
        parent.initialized = true;
        panel.action(
            FILTER_FULL_VOLUME_LABEL,
            None,
            None,
            &mut manager,
            &mut parent,
        );
        panel.action(CLEANUP_LABEL, None, None, &mut manager, &mut parent);
        panel.action(
            "View Filtered Volume",
            None,
            None,
            &mut manager,
            &mut parent,
        );
        assert_eq!(manager.calls, ["chunksetup", "imod"]);
        assert!(parent.cleaned);
    }

    #[test]
    fn validation_and_chunk_parameter_overloads_retain_source_order() {
        let mut panel = FilterFullVolumePanel::new(DialogType::AnisotropicDiffusion);
        panel.create_panel();
        assert!(!panel.get_parameters_anisotropic_diffusion(&mut DiffusionParam::default(), true));
        panel.ltf_k_value.set_text("2.5");
        panel.sp_iteration.set_value(17);
        panel.sp_memory_per_chunk.set_value(720);
        panel.cb_overlap_times_four.set_selected(true);
        let mut diffusion = DiffusionParam::default();
        let mut chunk = ChunkParam::default();
        assert!(panel.get_parameters_anisotropic_diffusion(&mut diffusion, true));
        panel.get_parameters_chunksetup(&mut chunk);
        assert_eq!((diffusion.k, diffusion.iteration), ("2.5".into(), 17));
        assert_eq!(
            (chunk.memory, chunk.overlap, chunk.overlap_times_four),
            (720, 17, true)
        );
    }
}
