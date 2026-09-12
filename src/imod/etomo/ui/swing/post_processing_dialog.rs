//! `IMOD/Etomo/src/etomo/ui/swing/PostProcessingDialog.java`.
//!
//! The Java source owns the five-panel tab coordinator. Native tab widgets,
//! its processing mediator, UIHarness and comscript/filesystem calls remain
//! explicit source boundaries.
#![allow(dead_code)]

use super::{
    alt_stack_panel::{
        AltStackMetaDataBoundary, AltStackPanel, AltStackTiltParamBoundary,
        AltTomoSetupParamBoundary,
    },
    check_box::CheckBox,
    context_menu::{ContextMenu, MouseEvent},
    flatten_volume_panel::{FlattenVolumeMetaData, FlattenVolumePanel, WarpVolParamBoundary},
    process_dialog::ProcessDialogApplicationManager,
    process_interface::ProcessInterface,
    squeeze_vol_panel::{
        ConstReduceFiltVolParam, ConstSqueezevolParamBoundary, MakecomfileParamBoundary,
        SqueezeVolMetaDataBoundary, SqueezeVolPanel, SqueezeVolPanelApplicationManager,
    },
    subtomograms_panel::{
        ConstSubtomogramsMetaData, SubtomoSetupParam, SubtomogramsMetaData, SubtomogramsPanel,
        SubtomogramsPanelManager,
    },
    trimvol_panel::{
        ReconScreenStateBoundary, TrimvolInputFileState, TrimvolMetaDataBoundary, TrimvolPanel,
        TrimvolParamBoundary,
    },
};
use crate::imod::etomo::r#type::{
    axis_id::AxisID, dialog_type::DialogType, processing_method::ProcessingMethod,
};
use crate::imod::etomo::ui::{
    queue_table_event::QueueTableEvent, queue_table_listener::QueueTableListener,
};

/// Java inner `Tab`, preserving its source tab index values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Tab {
    #[default]
    TrimVol,
    Flatten,
    SqueezeVol,
    AltStack,
    Subtomograms,
}

impl Tab {
    /// Java `Tab.getInstance(int)`.
    pub fn get_instance(index: i32) -> Self {
        match index {
            1 => Self::Flatten,
            2 => Self::SqueezeVol,
            3 => Self::AltStack,
            4 => Self::Subtomograms,
            _ => Self::TrimVol,
        }
    }

    /// Java `Tab.toInt()`.
    pub fn to_int(self) -> i32 {
        match self {
            Self::TrimVol => 0,
            Self::Flatten => 1,
            Self::SqueezeVol => 2,
            Self::AltStack => 3,
            Self::Subtomograms => 4,
        }
    }
}

/// Direct `ApplicationManager`/mediator/FileType/comscript operations used by
/// this unit. No filesystem or scheduler substitute is fabricated here.
pub trait PostProcessingDialogManager:
    ProcessDialogApplicationManager + SubtomogramsPanelManager
{
    fn is_dual_axis(&self) -> bool;
    fn is_montage(&self) -> bool;
    fn rootname_even_exists(&self) -> bool;
    fn rootname_odd_exists(&self) -> bool;
    fn register_processing_method(&mut self, method: ProcessingMethod);
    fn set_processing_method(&mut self, method: ProcessingMethod);
    fn add_queue_listener_on_switch_dialog(&mut self);
    fn move_sub_frame(&mut self);
    fn done_post_processing(&mut self);
    fn load_alt_tomo_setup(&mut self, retry: bool) -> bool;
    fn touch_alt_tomo_setup_comscript(&mut self);
    fn get_alt_tomo_setup_param(&self) -> AltTomoSetupParamBoundary;
    fn gold_eraser_com_file_exists(&self) -> (bool, bool, bool);
    fn eraser_log_file_exists(&self) -> (bool, bool, bool);
    fn ctf_correction_log_file_exists(&self) -> (bool, bool, bool);
    fn gold_eraser_log_file_exists(&self) -> (bool, bool, bool);
    fn mtf_filter_log_file_exists(&self) -> (bool, bool, bool);
}

/// The source-observable `TabbedPane` attachment and listener state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostProcessingTabbedPane {
    pub labels: Vec<&'static str>,
    pub selected_index: i32,
    pub attached_panel: Vec<Option<Tab>>,
    pub mouse_listener_present: bool,
    pub change_listener_present: bool,
}

/// Java private static `TabChangeListener`.  The Java listener retains its
/// enclosing dialog; Rust passes that same source owner at its GUI callback
/// boundary rather than introducing shared mutable ownership.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TabChangeListener;

impl TabChangeListener {
    /// Java private `TabChangeListener(PostProcessingDialog)`.
    pub fn new() -> Self {
        Self
    }

    /// Java `stateChanged(ChangeEvent)`.
    pub fn state_changed<M: PostProcessingDialogManager>(
        &self,
        adaptee: &mut PostProcessingDialog,
        manager: &mut M,
    ) {
        adaptee.change_tab(manager);
    }
}

/// The fields of Java `MetaData` reached by this coordinator, grouped by the
/// translated child-panel boundaries plus the source-owned current tab.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PostProcessingMetaDataBoundary {
    pub trimvol: TrimvolMetaDataBoundary,
    pub flatten: FlattenVolumeMetaData,
    pub squeeze_vol: SqueezeVolMetaDataBoundary,
    pub alt_stack: AltStackMetaDataBoundary,
    pub subtomo_reorientation_type_none: bool,
    pub subtomo_reorientation_type_flipped: bool,
    pub subtomo_reorientation_type_rotated: bool,
    pub subtomo_make_volume_stacks: bool,
    pub subtomo_make_volume_stacks_value: i32,
    pub subtomo_new_aligned_binning: i32,
    pub subtomo_fourier_reduce_by_factor: i32,
    pub subtomo_extent_of_z_levels_in_nm: String,
    pub post_cur_tab: Option<i32>,
}

impl ConstSubtomogramsMetaData for PostProcessingMetaDataBoundary {
    fn subtomo_reorientation_type_none(&self) -> bool {
        self.subtomo_reorientation_type_none
    }
    fn subtomo_reorientation_type_flipped(&self) -> bool {
        self.subtomo_reorientation_type_flipped
    }
    fn subtomo_reorientation_type_rotated(&self) -> bool {
        self.subtomo_reorientation_type_rotated
    }
    fn subtomo_make_volume_stacks(&self) -> bool {
        self.subtomo_make_volume_stacks
    }
    fn subtomo_make_volume_stacks_value(&self) -> i32 {
        self.subtomo_make_volume_stacks_value
    }
    fn subtomo_new_aligned_binning(&self) -> i32 {
        self.subtomo_new_aligned_binning
    }
    fn subtomo_fourier_reduce_by_factor(&self) -> i32 {
        self.subtomo_fourier_reduce_by_factor
    }
    fn subtomo_extent_of_z_levels_in_nm(&self) -> String {
        self.subtomo_extent_of_z_levels_in_nm.clone()
    }
}

impl SubtomogramsMetaData for PostProcessingMetaDataBoundary {
    fn set_subtomo_reorientation_type_none(&mut self, value: bool) {
        self.subtomo_reorientation_type_none = value;
    }
    fn set_subtomo_reorientation_type_flipped(&mut self, value: bool) {
        self.subtomo_reorientation_type_flipped = value;
    }
    fn set_subtomo_reorientation_type_rotated(&mut self, value: bool) {
        self.subtomo_reorientation_type_rotated = value;
    }
    fn set_subtomo_make_volume_stacks(&mut self, value: i32) {
        self.subtomo_make_volume_stacks_value = value;
    }
    fn set_subtomo_new_aligned_binning(&mut self, value: i32) {
        self.subtomo_new_aligned_binning = value;
    }
    fn set_subtomo_fourier_reduce_by_factor(&mut self, value: i32) {
        self.subtomo_fourier_reduce_by_factor = value;
    }
    fn set_subtomo_extent_of_z_levels_in_nm(&mut self, value: String) {
        self.subtomo_extent_of_z_levels_in_nm = value;
    }
}

/// Java `PostProcessingDialog` fields. `None` for `subtomograms_panel` is the
/// source's explicit dual-axis/montage branch.
pub struct PostProcessingDialog {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub trimvol_panel: TrimvolPanel,
    pub flatten_volume_panel: FlattenVolumePanel,
    pub squeeze_vol_panel: SqueezeVolPanel,
    pub subtomograms_panel: Option<SubtomogramsPanel>,
    pub alt_stack_panel: AltStackPanel,
    pub tabbed_pane: PostProcessingTabbedPane,
    pub cur_tab: Tab,
    pub open_alt_stack_first_time: bool,
    pub displayed: bool,
    pub root_box_layout_y_axis: bool,
    pub root_border_title: &'static str,
    pub execute_button_text: &'static str,
    /// Java's active Subtomograms `setUseQueueCheckBox` direct component boundary.
    pub queue_checkbox_installed: bool,
}

impl PostProcessingDialog {
    /// Java private `PostProcessingDialog(ApplicationManager, boolean)`.
    pub fn new<M: PostProcessingDialogManager>(
        manager: &mut M,
        trimvol_input_file_missing: bool,
    ) -> Self {
        let axis_id = AxisID::Only;
        let dialog_type = DialogType::PostProcessing;
        let subtomograms_panel = (!manager.is_dual_axis() && !manager.is_montage())
            .then(|| SubtomogramsPanel::get_instance(manager, axis_id, dialog_type));
        let mut result = Self {
            axis_id,
            dialog_type,
            trimvol_panel: TrimvolPanel::new(axis_id, dialog_type, trimvol_input_file_missing),
            flatten_volume_panel: FlattenVolumePanel::get_post_instance(
                axis_id,
                dialog_type,
                super::multi_line_button::MultiLineButton::new_with_label(Some("Flatten")),
                super::multi_line_button::MultiLineButton::new_with_label(Some("Run Flattenwarp")),
            ),
            squeeze_vol_panel: SqueezeVolPanel::get_instance(axis_id, dialog_type),
            subtomograms_panel,
            alt_stack_panel: AltStackPanel::get_instance(
                axis_id,
                dialog_type,
                manager.is_dual_axis(),
                manager.rootname_even_exists(),
                manager.rootname_odd_exists(),
                Default::default(),
            ),
            tabbed_pane: PostProcessingTabbedPane {
                labels: vec!["Trim vol", "Flatten", "Reduce/filt vol", "Alt Stack"],
                selected_index: Tab::TrimVol.to_int(),
                attached_panel: vec![Some(Tab::TrimVol), None, None, None],
                mouse_listener_present: false,
                change_listener_present: false,
            },
            cur_tab: Tab::TrimVol,
            open_alt_stack_first_time: true,
            displayed: true,
            root_box_layout_y_axis: true,
            root_border_title: "Post Processing",
            execute_button_text: "Done",
            queue_checkbox_installed: false,
        };
        if result.subtomograms_panel.is_some() {
            result.tabbed_pane.labels.push("Subtomograms");
            result.tabbed_pane.attached_panel.push(None);
        }
        manager.register_processing_method(result.get_processing_method());
        manager.set_processing_method(result.get_processing_method());
        result
    }

    /// Java static `getInstance(ApplicationManager, boolean)`.
    pub fn get_instance<M: PostProcessingDialogManager>(
        manager: &mut M,
        trimvol_input_file_missing: bool,
    ) -> Self {
        let mut instance = Self::new(manager, trimvol_input_file_missing);
        instance.add_listeners();
        instance.tabbed_pane.selected_index = Tab::TrimVol.to_int();
        instance
    }

    /// Java private `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.tabbed_pane.mouse_listener_present = true;
        self.tabbed_pane.change_listener_present = true;
    }

    /// Java private `changeTab(ConstEtomoNumber)`; `None` represents null/isNull.
    pub fn change_tab_index<M: PostProcessingDialogManager>(
        &mut self,
        manager: &mut M,
        index: Option<i32>,
    ) {
        if let Some(index) = index {
            self.tabbed_pane.selected_index = index;
            self.change_tab(manager);
        }
    }

    /// Java private `changeTab()`.
    pub fn change_tab<M: PostProcessingDialogManager>(&mut self, manager: &mut M) {
        if let Some(slot) = self
            .tabbed_pane
            .attached_panel
            .get_mut(self.cur_tab.to_int() as usize)
        {
            *slot = None;
        }
        self.cur_tab = Tab::get_instance(self.tabbed_pane.selected_index);
        if self.cur_tab == Tab::AltStack && self.open_alt_stack_first_time {
            self.create_alt_tomo_setup_com_file(manager);
        }
        if self.cur_tab == Tab::Subtomograms && self.subtomograms_panel.is_some() {
            manager.add_queue_listener_on_switch_dialog();
        }
        if let Some(slot) = self
            .tabbed_pane
            .attached_panel
            .get_mut(self.cur_tab.to_int() as usize)
        {
            *slot = Some(self.cur_tab);
        }
        if self.cur_tab == Tab::AltStack {
            self.alt_stack_panel.check_if_files_exist(
                manager.gold_eraser_com_file_exists(),
                manager.eraser_log_file_exists(),
                manager.ctf_correction_log_file_exists(),
                manager.gold_eraser_log_file_exists(),
                manager.mtf_filter_log_file_exists(),
            );
        }
        manager.set_processing_method(self.get_processing_method());
        SubtomogramsPanelManager::pack(manager, self.axis_id);
        manager.move_sub_frame();
    }

    /// Java `setParameters(ConstSqueezevolParam)`; the second argument is the
    /// source `ApplicationManager.isSqueezevolFlipped()` boundary.
    pub fn set_parameters_squeezevol(
        &mut self,
        param: &ConstSqueezevolParamBoundary,
        squeezevol_flipped: bool,
    ) {
        self.squeeze_vol_panel
            .set_parameters_squeezevol(param, squeezevol_flipped);
    }

    /// Java `setParameters(ReduceFiltVolParam, boolean, boolean)`.  MRC pixel
    /// count and input orientation remain the direct image-metadata boundary.
    pub fn set_parameters_reduce_filt_vol<P: ConstReduceFiltVolParam>(
        &mut self,
        param: &P,
        dialog_not_exists: bool,
        com_file_exists: bool,
        trim_vol_pixel_area: usize,
        input_flipped: bool,
    ) {
        self.squeeze_vol_panel.set_parameters_reduce_filt_vol(
            param,
            dialog_not_exists,
            com_file_exists,
            trim_vol_pixel_area,
            input_flipped,
        );
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&mut self, state: &ReconScreenStateBoundary) {
        self.trimvol_panel.set_parameters_recon_screen_state(state);
        self.squeeze_vol_panel
            .set_parameters_screen_state(state.screen_state.clone());
    }

    /// Java `setParameters(SubtomoSetupParam)`.
    pub fn set_parameters_subtomo_setup<P: SubtomoSetupParam>(&mut self, param: &P) {
        if let Some(panel) = &mut self.subtomograms_panel {
            panel.set_parameters(param);
        }
    }

    /// Java `initParameters(TrimvolParam)`.
    pub fn init_parameters(&mut self, param: &TrimvolParamBoundary) {
        self.trimvol_panel.init_parameters(param);
    }

    /// Java `setParameters(ConstMetaData, boolean)`.
    pub fn set_parameters_metadata<M: PostProcessingDialogManager>(
        &mut self,
        manager: &mut M,
        metadata: &PostProcessingMetaDataBoundary,
        dialog_exists: bool,
    ) {
        self.trimvol_panel
            .set_parameters(&metadata.trimvol, dialog_exists);
        self.flatten_volume_panel
            .set_parameters_metadata(&metadata.flatten);
        self.squeeze_vol_panel
            .set_parameters_meta_data(&metadata.squeeze_vol);
        if let Some(panel) = &mut self.subtomograms_panel {
            panel.set_parameters_metadata(metadata);
        }
        self.alt_stack_panel
            .set_parameters_metadata(&metadata.alt_stack);
        self.change_tab_index(manager, metadata.post_cur_tab);
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_metadata(&mut self, metadata: &mut PostProcessingMetaDataBoundary) {
        self.trimvol_panel.get_parameters(&mut metadata.trimvol);
        self.flatten_volume_panel
            .get_parameters_metadata(&mut metadata.flatten);
        self.squeeze_vol_panel
            .get_parameters_meta_data(&mut metadata.squeeze_vol);
        if let Some(panel) = &self.subtomograms_panel {
            panel.get_parameters_metadata(metadata);
        }
        self.alt_stack_panel
            .get_parameters_metadata(&mut metadata.alt_stack);
        metadata.post_cur_tab = Some(self.cur_tab.to_int());
    }

    /// Java `getParametersForTrimvol(MetaData)`.
    pub fn get_parameters_for_trimvol(&self, metadata: &mut TrimvolMetaDataBoundary) {
        self.trimvol_panel.get_parameters_for_trimvol(metadata);
    }

    /// Java `setParameters(ConstWarpVolParam)`.
    pub fn set_parameters_warp_vol(&mut self, param: &WarpVolParamBoundary) {
        self.flatten_volume_panel.set_parameters_warp_vol(param);
    }

    /// Java `getParameters(WarpVolParam, boolean)`.
    pub fn get_parameters_warp_vol<M: super::flatten_volume_panel::FlattenVolumePanelManager>(
        &self,
        param: &mut WarpVolParamBoundary,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        self.flatten_volume_panel
            .get_parameters_warp_vol(param, do_validation, manager)
    }

    /// Java `getParameters(MakecomfileParam, boolean)`.
    pub fn get_parameters_makecomfile<M: SqueezeVolPanelApplicationManager>(
        &self,
        param: &mut MakecomfileParamBoundary,
        do_validation: bool,
        manager: &M,
    ) -> bool {
        self.squeeze_vol_panel
            .get_parameters_makecomfile(param, do_validation, manager)
    }

    /// Java `setParameters(TiltParam, boolean)`.
    pub fn set_parameters_tilt(&mut self, param: &AltStackTiltParamBoundary, initialize: bool) {
        self.alt_stack_panel.set_parameters_tilt(param, initialize);
    }

    /// Java `getFlattenWarpDisplay()`.
    pub fn get_flatten_warp_display(&mut self) -> &mut FlattenVolumePanel {
        &mut self.flatten_volume_panel
    }

    /// Java `getTrimvolDisplay()`.
    pub fn get_trimvol_display(&self) -> &TrimvolPanel {
        &self.trimvol_panel
    }

    /// Java `getSubtomoSetupDisplay()`; `None` retains the source's explicit
    /// dual-axis/montage null branch.
    pub fn get_subtomo_setup_display(&self) -> Option<&SubtomogramsPanel> {
        self.subtomograms_panel.as_ref()
    }

    /// Java `getAltStackDisplay()`.
    pub fn get_alt_stack_display(&self) -> &AltStackPanel {
        self.alt_stack_panel.get_alt_stack_display()
    }

    /// Java `getReduceFiltVolDisplay()`.
    pub fn get_reduce_filt_vol_display(&mut self) -> &mut SqueezeVolPanel {
        self.squeeze_vol_panel.get_reduce_filt_vol_display()
    }

    /// Java `setStartupWarnings(TrimvolInputFileState)`.
    pub fn set_startup_warnings(&mut self, state: TrimvolInputFileState) -> bool {
        self.trimvol_panel.set_startup_warnings(state)
    }

    /// Java `getParameters(TrimvolParam, boolean)`; `output_format` is the
    /// source MetaData image-output-format boundary.
    pub fn get_parameters_trimvol(
        &mut self,
        param: &mut TrimvolParamBoundary,
        do_validation: bool,
        output_format: Option<String>,
    ) -> bool {
        self.trimvol_panel
            .get_parameters_trimvol(param, do_validation, output_format)
    }

    /// Java override `done()`.
    pub fn done<M: PostProcessingDialogManager>(&mut self, manager: &mut M) {
        manager.done_post_processing();
        self.squeeze_vol_panel.done();
        self.trimvol_panel.done();
        self.flatten_volume_panel.done();
        self.displayed = false;
    }

    /// Java `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.cur_tab == Tab::Subtomograms {
            if let Some(panel) = &self.subtomograms_panel {
                return match panel.get_processing_method() {
                    1 => ProcessingMethod::PpGpu,
                    _ => ProcessingMethod::PpCpu,
                };
            }
        }
        if self.cur_tab == Tab::AltStack {
            return self
                .alt_stack_panel
                .get_processing_method()
                .unwrap_or(ProcessingMethod::LocalCpu);
        }
        ProcessingMethod::LocalCpu
    }

    /// Java `getSecondaryProcessingMethod()`, whose source body returns null.
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java `lockProcessingMethod(boolean)`, whose source body is empty.
    pub fn lock_processing_method(&mut self, _lock: bool) {}

    /// Java `setMethod(ProcessingMethod)`.
    pub fn set_method<M: PostProcessingDialogManager>(
        &mut self,
        manager: &mut M,
        method: ProcessingMethod,
    ) {
        manager.set_processing_method(method);
    }

    /// Java `updateGpu(boolean)`.
    pub fn update_gpu(&mut self, disable: bool) {
        if self.cur_tab == Tab::Subtomograms {
            if let Some(panel) = &mut self.subtomograms_panel {
                panel.update_gpu(disable);
            }
        } else if self.cur_tab == Tab::AltStack {
            self.alt_stack_panel.update_gpu(disable);
        }
    }

    /// Java `isUseGpu()`.
    pub fn is_use_gpu(&self) -> bool {
        self.cur_tab == Tab::Subtomograms
            && self
                .subtomograms_panel
                .as_ref()
                .is_some_and(SubtomogramsPanel::is_use_gpu)
    }

    /// Java `setUseQueueCheckBox(ButtonComponent)`.
    pub fn set_use_queue_check_box(&mut self, _use_queue_checkbox: Option<CheckBox>) {
        if self.cur_tab == Tab::Subtomograms {
            if let Some(panel) = &mut self.subtomograms_panel {
                panel.set_use_queue_check_box();
                self.queue_checkbox_installed = true;
            }
        }
    }

    /// Java private `createAltTomoSetupComFile()`.
    pub fn create_alt_tomo_setup_com_file<M: PostProcessingDialogManager>(
        &mut self,
        manager: &mut M,
    ) {
        if !manager.load_alt_tomo_setup(false) {
            manager.touch_alt_tomo_setup_comscript();
            manager.load_alt_tomo_setup(true);
        }
        let param = manager.get_alt_tomo_setup_param();
        self.alt_stack_panel.set_parameters(&param);
        self.open_alt_stack_first_time = false;
    }
}

impl ContextMenu for PostProcessingDialog {
    /// Java `popUpContextMenu(MouseEvent)`, whose source body is empty.
    fn pop_up_context_menu(&mut self, _mouse_event: MouseEvent) {}
}

impl QueueTableListener for PostProcessingDialog {
    /// Java inherited queue listener boundary; no direct source handling here.
    fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}
}

impl ProcessInterface for PostProcessingDialog {
    type QueueCheckBox = CheckBox;

    fn update_gpu(&mut self, disable_gpu: bool) {
        PostProcessingDialog::update_gpu(self, disable_gpu);
    }
    fn get_processing_method(&self) -> ProcessingMethod {
        PostProcessingDialog::get_processing_method(self)
    }
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        PostProcessingDialog::get_secondary_processing_method(self)
    }
    fn lock_processing_method(&mut self, lock: bool) {
        PostProcessingDialog::lock_processing_method(self, lock);
    }
    /// The Java implementation only forwards to its mediator if it is non-null;
    /// callers with a concrete manager use the inherent source method above.
    fn set_method(&mut self, _processing_method: ProcessingMethod) {}
    fn is_use_gpu(&self) -> bool {
        PostProcessingDialog::is_use_gpu(self)
    }
    fn set_use_queue_check_box(&mut self, use_queue_checkbox: Option<Self::QueueCheckBox>) {
        PostProcessingDialog::set_use_queue_check_box(self, use_queue_checkbox);
    }
    fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[derive(Default)]
    struct Manager {
        dual_axis: bool,
        calls: Vec<String>,
    }

    impl ProcessDialogApplicationManager for Manager {
        fn is_advanced(&self, _dialog_type: DialogType, _axis_id: AxisID) -> bool {
            false
        }
    }

    impl SubtomogramsPanelManager for Manager {
        fn property_user_dir(&self) -> PathBuf {
            PathBuf::from(".")
        }
        fn total_gpus(&self, _axis: AxisID) -> i32 {
            0
        }
        fn output_processchunks_exists(&self, _axis: AxisID, _directory: &Path) -> bool {
            false
        }
        fn confirm_existing_output(&mut self, _axis: AxisID) -> bool {
            true
        }
        fn subtomo_setup(&mut self, _axis: AxisID, _dialog: DialogType, _method: i32) {}
        fn open_files_in_imod(&mut self, _axis: AxisID, _names: Vec<String>, _subdir: PathBuf) {}
        fn pack(&mut self, _axis: AxisID) {
            self.calls.push("pack".into());
        }
    }

    impl PostProcessingDialogManager for Manager {
        fn is_dual_axis(&self) -> bool {
            self.dual_axis
        }
        fn is_montage(&self) -> bool {
            false
        }
        fn rootname_even_exists(&self) -> bool {
            false
        }
        fn rootname_odd_exists(&self) -> bool {
            false
        }
        fn register_processing_method(&mut self, _method: ProcessingMethod) {
            self.calls.push("register".into());
        }
        fn set_processing_method(&mut self, _method: ProcessingMethod) {
            self.calls.push("set-method".into());
        }
        fn add_queue_listener_on_switch_dialog(&mut self) {
            self.calls.push("queue".into());
        }
        fn move_sub_frame(&mut self) {
            self.calls.push("move".into());
        }
        fn done_post_processing(&mut self) {
            self.calls.push("done".into());
        }
        fn load_alt_tomo_setup(&mut self, retry: bool) -> bool {
            self.calls
                .push(if retry { "alt-load-retry" } else { "alt-load" }.into());
            true
        }
        fn touch_alt_tomo_setup_comscript(&mut self) {
            self.calls.push("alt-touch".into());
        }
        fn get_alt_tomo_setup_param(&self) -> AltTomoSetupParamBoundary {
            AltTomoSetupParamBoundary::default()
        }
        fn gold_eraser_com_file_exists(&self) -> (bool, bool, bool) {
            (false, false, false)
        }
        fn eraser_log_file_exists(&self) -> (bool, bool, bool) {
            (false, false, false)
        }
        fn ctf_correction_log_file_exists(&self) -> (bool, bool, bool) {
            (false, false, false)
        }
        fn gold_eraser_log_file_exists(&self) -> (bool, bool, bool) {
            (false, false, false)
        }
        fn mtf_filter_log_file_exists(&self) -> (bool, bool, bool) {
            (false, false, false)
        }
    }

    #[test]
    fn single_axis_creates_subtomograms_and_alt_stack_is_lazy() {
        let mut manager = Manager::default();
        let mut dialog = PostProcessingDialog::get_instance(&mut manager, false);
        assert!(dialog.subtomograms_panel.is_some());
        assert_eq!(dialog.tabbed_pane.labels.len(), 5);
        assert!(dialog.tabbed_pane.change_listener_present);
        dialog.change_tab_index(&mut manager, Some(Tab::AltStack.to_int()));
        assert!(!dialog.open_alt_stack_first_time);
        assert_eq!(
            manager.calls,
            [
                "register",
                "set-method",
                "alt-load",
                "set-method",
                "pack",
                "move"
            ]
        );
    }

    #[test]
    fn dual_axis_uses_source_null_subtomograms_branch() {
        let mut manager = Manager {
            dual_axis: true,
            ..Default::default()
        };
        let dialog = PostProcessingDialog::get_instance(&mut manager, false);
        assert!(dialog.subtomograms_panel.is_none());
        assert_eq!(dialog.tabbed_pane.labels.len(), 4);
    }

    #[test]
    fn subtomogram_metadata_and_processing_method_follow_active_source_tab() {
        let mut manager = Manager::default();
        let mut dialog = PostProcessingDialog::get_instance(&mut manager, false);
        let metadata = PostProcessingMetaDataBoundary {
            subtomo_reorientation_type_flipped: true,
            subtomo_make_volume_stacks: true,
            subtomo_make_volume_stacks_value: 25,
            subtomo_new_aligned_binning: 2,
            subtomo_fourier_reduce_by_factor: 3,
            subtomo_extent_of_z_levels_in_nm: "12".into(),
            ..Default::default()
        };
        dialog.set_parameters_metadata(&mut manager, &metadata, true);
        let panel = dialog.subtomograms_panel.as_mut().unwrap();
        assert!(panel.rb_reorientation_type_flipped.is_selected());
        panel.subtomo_setup_cpu_gpu_panel.cpus_only = false;
        panel.subtomo_setup_cpu_gpu_panel.gpu_recon_ctf = true;
        dialog.cur_tab = Tab::Subtomograms;
        assert_eq!(dialog.get_processing_method(), ProcessingMethod::PpGpu);

        let mut written = PostProcessingMetaDataBoundary::default();
        dialog.get_parameters_metadata(&mut written);
        assert!(written.subtomo_reorientation_type_flipped);
        assert_eq!(written.subtomo_make_volume_stacks_value, 25);
    }
}
