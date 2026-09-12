//! `IMOD/Etomo/src/etomo/ui/swing/AltStackPanel.java`.
//!
//! Swing construction, autodoc lookup, file-type lookup, `CpuGpuPanel`, and
//! `ApplicationManager` are direct boundaries.  This module retains the panel's
//! source-owned state, validation order, axis policy, and dispatch arguments.
#![allow(dead_code)]

use super::check_box::CheckBox;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonGroup};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field_type::FieldType;
use std::cell::RefCell;
use std::rc::Rc;

pub const EVEN_ODD_PAIRS_LABEL: &str = "Process even and odd pairs";
pub const ROOTNAME_ALTERNATIVE_STACK_LABEL: &str = "Rootname of alternative stack ";
pub const AXES_TO_PROCESS_LABEL: &str = "Axes to process:";
pub const AXES_TO_PROCESS_BOTH_LABEL: &str = "Both";
pub const AXES_TO_PROCESS_A_ONLY_LABEL: &str = "A only";
pub const AXES_TO_PROCESS_B_ONLY_LABEL: &str = "B only";
pub const PREPROCESS_LABEL: &str = "Preprocess";
pub const ARCHIVE_ORIGINAL_STACK_LABEL: &str = "Archive original stack";
pub const CORRECT_CTF_LABEL: &str = "Correct CTF";
pub const ERASE_GOLD_LABEL: &str = "Erase gold";
pub const FILTER_IN_2D_LABEL: &str = "Filter in 2D";
pub const TRIM_VOLUME_LABEL: &str = "Trim volume";
pub const CLEAN_UP_INTERMEDIATE_FILES_LABEL: &str = "Clean up intermediate files";
pub const BUTTON_OPEN_RECON_ALTERNATIVE_LABEL: &str = "Open Alternative Tomogram in 3dmod";
pub const BUTTON_OPEN_RECON_EVEN_ODD_LABEL: &str = "Open Even and Odd Tomograms in 3dmod";
pub const BUTTON_OPEN_RECON_AXIS_A_LABEL: &str = "Open Axis A Tomogram in 3dmod";
pub const BUTTON_OPEN_RECON_AXIS_B_LABEL: &str = "Open Axis B Tomogram in 3dmod";

/// Java `TomogramState` queries used by this panel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AltStackTomogramState {
    pub alt_tomo_trim_vol_checked: bool,
    pub post_proc_trim_vol_input_n_rows_null: bool,
}

/// Java `AltTomoSetupParam` state reached by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AltTomoSetupParamBoundary {
    pub even_and_odd_pairs: Option<bool>,
    pub rootname_to_process: Option<String>,
    pub axis_to_process: Option<String>,
    pub preprocess_for_extremes: Option<i32>,
    pub correct_ctf: bool,
    pub erase_fiducials: bool,
    pub filter_in_2d: bool,
    pub trim_volume: bool,
    pub clean_up_intermediates: bool,
}

/// Java metadata fields used by the two `getParameters` and `setParameters` paths.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AltStackMetaDataBoundary {
    pub alt_tomo_rootname: String,
    pub alt_tomo_trim_volume: bool,
    pub alt_tomo_archive_orig_stack: bool,
    pub tilt_parallel: Option<String>,
}

/// Java `TiltParam`/CPU-GPU panel boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AltStackTiltParamBoundary {
    pub cpu_gpu_parameters_written: bool,
}

/// Direct `CpuGpuPanel` state and operations used by Java `AltStackPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CpuGpuPanelBoundary {
    pub processing_method: Option<ProcessingMethod>,
    pub update_gpu_calls: Vec<bool>,
    pub parameters_written: bool,
    pub parameters_set: bool,
    pub alt_stack_process_interface_set: bool,
    pub reregistered: bool,
}

impl CpuGpuPanelBoundary {
    pub fn msg_processing_method_changed(&mut self, _from_display: bool, _force: bool) {}
    pub fn update_gpu(&mut self, disable: bool) {
        self.update_gpu_calls.push(disable);
    }
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        self.processing_method
    }
    pub fn get_parameters_metadata(&mut self, _metadata: &mut AltStackMetaDataBoundary) {
        self.parameters_written = true;
    }
    pub fn get_parameters_tilt(&mut self, tilt: &mut AltStackTiltParamBoundary) {
        tilt.cpu_gpu_parameters_written = true;
    }
    pub fn set_parameters_metadata(&mut self, _metadata: &AltStackMetaDataBoundary) {
        self.parameters_set = true;
    }
    pub fn set_parameters_tilt(&mut self, _tilt: &AltStackTiltParamBoundary, _initialize: bool) {
        self.parameters_set = true;
    }
    pub fn reregister_processing_method_mediator(&mut self) {
        self.reregistered = true;
    }
}

/// Direct `ApplicationManager` operations invoked by Java `AltStackPanel`.
pub trait AltStackPanelApplicationManager {
    fn is_dual_axis(&self) -> bool;
    fn state(&self) -> AltStackTomogramState;
    fn alt_tomo_setup(
        &mut self,
        axis_id: AxisID,
        dialog_type: DialogType,
        method: Option<ProcessingMethod>,
    );
    fn alt_tomo_setup_restore_swapped_files(
        &mut self,
        axis_id: AxisID,
        dialog_type: DialogType,
        method: Option<ProcessingMethod>,
    );
    fn open_alternative_tomogram(
        &mut self,
        rootname: &str,
        axis_id: AxisID,
        dual_axis: bool,
        options: Option<Run3dmodMenuOptions>,
        trimmed: bool,
    );
    fn open_even_odd_files_in_imod(
        &mut self,
        axis_id: AxisID,
        full_tomogram: bool,
        options: Option<Run3dmodMenuOptions>,
        trimmed: bool,
    );
}

/// Java final `AltStackPanel` source-visible state.  Swing hierarchy is captured
/// by the component-boundary flags instead of synthesising a parallel widget tree.
#[derive(Clone, Debug)]
pub struct AltStackPanel {
    pub cb_process_even_odd_pairs: CheckBox,
    pub ltf_rootname_of_alt_stack: LabeledTextField,
    pub cb_preprocess: CheckBox,
    pub cb_archive_orig_stack: CheckBox,
    pub cb_correct_ctf: CheckBox,
    pub cb_erase_gold: CheckBox,
    pub cb_filter_in_2d: CheckBox,
    pub cb_trim_volume: CheckBox,
    pub cb_clean_up_intermediate_files: CheckBox,
    pub btn_run_alt_tomo_setup: MultiLineButton,
    pub btn_open_recon_in_3dmod_1: MultiLineButton,
    pub btn_open_recon_in_3dmod_2: MultiLineButton,
    pub btn_restore_swapped_files: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub dual_axis: bool,
    pub cpu_gpu_panel: CpuGpuPanelBoundary,
    pub rb_both: Option<RadioButton>,
    pub rb_a_only: Option<RadioButton>,
    pub rb_b_only: Option<RadioButton>,
    pub tomogram_state: AltStackTomogramState,
    pub is_rootname_even_file: bool,
    pub is_rootname_odd_file: bool,
    pub even_odd_pairs_files_found: bool,
    pub is_gold_eraser_com_file: bool,
    pub is_gold_eraser_com_file_a: bool,
    pub is_gold_eraser_com_file_b: bool,
    pub is_eraser_log_file: bool,
    pub is_eraser_log_file_a: bool,
    pub is_eraser_log_file_b: bool,
    pub is_ctf_correction_log_file: bool,
    pub is_ctf_correction_log_file_a: bool,
    pub is_ctf_correction_log_file_b: bool,
    pub is_gold_eraser_log_file: bool,
    pub is_gold_eraser_log_file_a: bool,
    pub is_gold_eraser_log_file_b: bool,
    pub is_mtf_filter_log_file: bool,
    pub is_mtf_filter_log_file_a: bool,
    pub is_mtf_filter_log_file_b: bool,
    pub is_dialog_opened_first_time: bool,
    pub panel_created: bool,
    pub packed: bool,
    pub context_menu_requested: bool,
}

impl AltStackPanel {
    /// Java private constructor.  File existence comes from the caller-owned
    /// `FileType` boundary, preserving the constructor's exact decision.
    pub fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        dual_axis: bool,
        rootname_even_exists: bool,
        rootname_odd_exists: bool,
        state: AltStackTomogramState,
    ) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let (rb_both, rb_a_only, rb_b_only) = if dual_axis {
            (
                Some(RadioButton::new_in_group(
                    AXES_TO_PROCESS_BOTH_LABEL,
                    group.clone(),
                )),
                Some(RadioButton::new_in_group(
                    AXES_TO_PROCESS_A_ONLY_LABEL,
                    group.clone(),
                )),
                Some(RadioButton::new_in_group(
                    AXES_TO_PROCESS_B_ONLY_LABEL,
                    group,
                )),
            )
        } else {
            (None, None, None)
        };
        let mut result = Self {
            cb_process_even_odd_pairs: CheckBox::new_with_text(EVEN_ODD_PAIRS_LABEL),
            ltf_rootname_of_alt_stack: LabeledTextField::new(
                FieldType::String,
                ROOTNAME_ALTERNATIVE_STACK_LABEL,
            ),
            cb_preprocess: CheckBox::new_with_text(PREPROCESS_LABEL),
            cb_archive_orig_stack: CheckBox::new_with_text(ARCHIVE_ORIGINAL_STACK_LABEL),
            cb_correct_ctf: CheckBox::new_with_text(CORRECT_CTF_LABEL),
            cb_erase_gold: CheckBox::new_with_text(ERASE_GOLD_LABEL),
            cb_filter_in_2d: CheckBox::new_with_text(FILTER_IN_2D_LABEL),
            cb_trim_volume: CheckBox::new_with_text(TRIM_VOLUME_LABEL),
            cb_clean_up_intermediate_files: CheckBox::new_with_text(
                CLEAN_UP_INTERMEDIATE_FILES_LABEL,
            ),
            btn_run_alt_tomo_setup: MultiLineButton::new_with_label(Some("Run")),
            btn_open_recon_in_3dmod_1: MultiLineButton::new_with_label(Some(
                "Open Tomogram in 3dmod",
            )),
            btn_open_recon_in_3dmod_2: MultiLineButton::new_with_label(Some(
                "Open Tomogram B in 3dmod",
            )),
            btn_restore_swapped_files: MultiLineButton::new_with_label(Some(
                "Restore Swapped Files",
            )),
            axis_id,
            dialog_type,
            dual_axis,
            cpu_gpu_panel: CpuGpuPanelBoundary::default(),
            rb_both,
            rb_a_only,
            rb_b_only,
            tomogram_state: state,
            is_rootname_even_file: rootname_even_exists,
            is_rootname_odd_file: rootname_odd_exists,
            even_odd_pairs_files_found: rootname_even_exists && rootname_odd_exists,
            is_gold_eraser_com_file: false,
            is_gold_eraser_com_file_a: false,
            is_gold_eraser_com_file_b: false,
            is_eraser_log_file: false,
            is_eraser_log_file_a: false,
            is_eraser_log_file_b: false,
            is_ctf_correction_log_file: false,
            is_ctf_correction_log_file_a: false,
            is_ctf_correction_log_file_b: false,
            is_gold_eraser_log_file: false,
            is_gold_eraser_log_file_a: false,
            is_gold_eraser_log_file_b: false,
            is_mtf_filter_log_file: false,
            is_mtf_filter_log_file_a: false,
            is_mtf_filter_log_file_b: false,
            is_dialog_opened_first_time: true,
            panel_created: false,
            packed: false,
            context_menu_requested: false,
        };
        result
            .cb_process_even_odd_pairs
            .set_text(Some(EVEN_ODD_PAIRS_LABEL));
        result
            .cb_process_even_odd_pairs
            .set_alternate_text(Some(&format!("{EVEN_ODD_PAIRS_LABEL} (Files not found)")));
        if !result.even_odd_pairs_files_found {
            result.cb_process_even_odd_pairs.switch_text(true);
        }
        result
    }

    /// Java static `getInstance`.
    pub fn get_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        dual_axis: bool,
        rootname_even_exists: bool,
        rootname_odd_exists: bool,
        state: AltStackTomogramState,
    ) -> Self {
        let mut instance = Self::new(
            axis_id,
            dialog_type,
            dual_axis,
            rootname_even_exists,
            rootname_odd_exists,
            state,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }
    pub fn get_component(&self) -> bool {
        self.panel_created
    }
    fn add_listeners(&mut self) {
        for cb in [
            &mut self.cb_process_even_odd_pairs,
            &mut self.cb_preprocess,
            &mut self.cb_archive_orig_stack,
            &mut self.cb_correct_ctf,
            &mut self.cb_erase_gold,
            &mut self.cb_filter_in_2d,
            &mut self.cb_trim_volume,
            &mut self.cb_clean_up_intermediate_files,
        ] {
            cb.add_action_listener();
        }
        for rb in [&mut self.rb_both, &mut self.rb_a_only, &mut self.rb_b_only]
            .into_iter()
            .flatten()
        {
            rb.add_action_listener();
        }
        for button in [
            &mut self.btn_run_alt_tomo_setup,
            &mut self.btn_open_recon_in_3dmod_1,
            &mut self.btn_open_recon_in_3dmod_2,
            &mut self.btn_restore_swapped_files,
        ] {
            button.add_action_listener();
        }
    }
    pub fn pop_up_context_menu(&mut self) {
        self.context_menu_requested = true;
    }
    fn create_panel(&mut self) {
        if let Some(rb_both) = &mut self.rb_both {
            rb_both.set_selected(true);
        }
        self.cpu_gpu_panel.msg_processing_method_changed(true, true);
        self.set_checkboxes_first_time_only();
        self.cb_trim_volume.set_enabled(!self.dual_axis);
        self.ltf_rootname_of_alt_stack.set_required(self.dual_axis);
        self.panel_created = true;
        self.update_display();
    }
    fn update_display(&mut self) {
        let even_odd = self.cb_process_even_odd_pairs.is_enabled()
            && self.cb_process_even_odd_pairs.is_selected();
        self.ltf_rootname_of_alt_stack.set_enabled(!even_odd);
        if !self.dual_axis {
            self.ltf_rootname_of_alt_stack
                .set_required(self.ltf_rootname_of_alt_stack.is_enabled());
        }
        self.cb_archive_orig_stack
            .set_enabled(self.cb_preprocess.is_enabled() && self.cb_preprocess.is_selected());
        if self.dual_axis {
            self.btn_open_recon_in_3dmod_1
                .set_text(BUTTON_OPEN_RECON_AXIS_A_LABEL);
            self.btn_open_recon_in_3dmod_2
                .set_text(BUTTON_OPEN_RECON_AXIS_B_LABEL);
        } else if even_odd {
            self.btn_open_recon_in_3dmod_1
                .set_text(BUTTON_OPEN_RECON_EVEN_ODD_LABEL);
        } else {
            self.btn_open_recon_in_3dmod_1
                .set_text(BUTTON_OPEN_RECON_ALTERNATIVE_LABEL);
        }
        self.packed = true;
    }
    fn set_tool_tip_text(&mut self) { /* `AutodocFactory` lookup is the direct storage boundary. */
    }
    pub fn action<M: AltStackPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        action_command: Option<&str>,
        run_options: Option<Run3dmodMenuOptions>,
        even_exists: bool,
        odd_exists: bool,
    ) {
        let Some(command) = action_command else {
            return;
        };
        if self
            .cb_process_even_odd_pairs
            .equals_action_command(Some(command))
        {
            if self.cb_process_even_odd_pairs.is_selected() {
                if !self.even_odd_pairs_files_found {
                    self.is_rootname_even_file = even_exists;
                    self.is_rootname_odd_file = odd_exists;
                    if even_exists && odd_exists {
                        self.even_odd_pairs_files_found = true;
                        self.cb_process_even_odd_pairs.switch_text(false);
                        self.cb_process_even_odd_pairs.disable_warning();
                    } else {
                        self.cb_process_even_odd_pairs.enable_warning(true);
                    }
                } else {
                    self.cb_process_even_odd_pairs.disable_warning();
                }
            } else {
                self.cb_process_even_odd_pairs.disable_warning();
            }
        } else if self
            .rb_both
            .as_ref()
            .is_some_and(|v| v.get_action_command() == command)
            || self
                .rb_a_only
                .as_ref()
                .is_some_and(|v| v.get_action_command() == command)
            || self
                .rb_b_only
                .as_ref()
                .is_some_and(|v| v.get_action_command() == command)
        {
            self.enable_or_disable_cb_erase_gold();
        } else if self
            .btn_run_alt_tomo_setup
            .get_action_command()
            .or(self.btn_run_alt_tomo_setup.get_text())
            == Some(command)
        {
            manager.alt_tomo_setup(self.axis_id, self.dialog_type, self.get_processing_method());
        } else if self
            .btn_open_recon_in_3dmod_1
            .get_action_command()
            .or(self.btn_open_recon_in_3dmod_1.get_text())
            == Some(command)
        {
            if self.dual_axis {
                let root = self.ltf_rootname_of_alt_stack.get_text();
                manager.open_alternative_tomogram(&root, AxisID::First, true, run_options, false);
            } else {
                self.open_alt_stack_single_axis_tomograms(manager, run_options);
            }
        } else if self
            .btn_open_recon_in_3dmod_2
            .get_action_command()
            .or(self.btn_open_recon_in_3dmod_2.get_text())
            == Some(command)
        {
            let root = self.ltf_rootname_of_alt_stack.get_text();
            manager.open_alternative_tomogram(&root, AxisID::Second, true, run_options, false);
        } else if self
            .btn_restore_swapped_files
            .get_action_command()
            .or(self.btn_restore_swapped_files.get_text())
            == Some(command)
        {
            manager.alt_tomo_setup_restore_swapped_files(
                self.axis_id,
                self.dialog_type,
                self.get_processing_method(),
            );
        }
        self.update_display();
    }
    pub fn open_alt_stack_single_axis_tomograms<M: AltStackPanelApplicationManager>(
        &self,
        manager: &mut M,
        options: Option<Run3dmodMenuOptions>,
    ) {
        let trimmed = self.tomogram_state.alt_tomo_trim_vol_checked;
        if self.cb_process_even_odd_pairs.check_box.visible
            && self.cb_process_even_odd_pairs.is_enabled()
            && self.cb_process_even_odd_pairs.is_selected()
        {
            manager.open_even_odd_files_in_imod(self.axis_id, !trimmed, options, trimmed);
        } else {
            manager.open_alternative_tomogram(
                &self.ltf_rootname_of_alt_stack.get_text(),
                self.axis_id,
                false,
                options,
                trimmed,
            );
        }
    }
    pub fn action_performed<M: AltStackPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        command: Option<&str>,
        even_exists: bool,
        odd_exists: bool,
    ) {
        self.action(manager, command, None, even_exists, odd_exists);
    }
    pub fn update_gpu(&mut self, disable: bool) {
        self.update_display();
        self.cpu_gpu_panel.update_gpu(disable);
    }
    pub fn get_processing_method(&self) -> Option<ProcessingMethod> {
        self.cpu_gpu_panel.get_processing_method()
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    pub fn lock_processing_method(&mut self, _lock: bool) {}
    pub fn set_method(&mut self, processing_method: ProcessingMethod) {
        self.cpu_gpu_panel.processing_method = Some(processing_method);
    }
    pub fn is_use_gpu(&self) -> bool {
        false
    }
    pub fn add_queue_listener(&mut self) {}
    pub fn reregister_processing_method_mediator(&mut self) {
        self.cpu_gpu_panel.reregister_processing_method_mediator();
    }
    /// Java `checkIfFilesExist`; file checks are caller-provided results from `FileType.exists`.
    pub fn check_if_files_exist(
        &mut self,
        gold_com: (bool, bool, bool),
        eraser: (bool, bool, bool),
        ctf: (bool, bool, bool),
        gold_log: (bool, bool, bool),
        mtf: (bool, bool, bool),
    ) {
        if self.dual_axis {
            self.is_gold_eraser_com_file_a = gold_com.1;
            self.is_gold_eraser_com_file_b = gold_com.2;
            if self.is_dialog_opened_first_time {
                self.is_eraser_log_file_a = eraser.1;
                self.is_eraser_log_file_b = eraser.2;
                self.is_ctf_correction_log_file_a = ctf.1;
                self.is_ctf_correction_log_file_b = ctf.2;
                self.is_gold_eraser_log_file_a = gold_log.1;
                self.is_gold_eraser_log_file_b = gold_log.2;
                self.is_mtf_filter_log_file_a = mtf.1;
                self.is_mtf_filter_log_file_b = mtf.2;
                self.is_dialog_opened_first_time = false;
            }
        } else {
            self.is_gold_eraser_com_file = gold_com.0;
            if self.is_dialog_opened_first_time {
                self.is_eraser_log_file = eraser.0;
                self.is_ctf_correction_log_file = ctf.0;
                self.is_gold_eraser_log_file = gold_log.0;
                self.is_mtf_filter_log_file = mtf.0;
                self.is_dialog_opened_first_time = false;
            }
        }
        self.enable_or_disable_cb_erase_gold();
        self.update_display();
    }
    fn set_checkboxes_first_time_only(&mut self) {
        if self.dual_axis {
            self.cb_preprocess
                .set_selected(self.is_eraser_log_file_a && self.is_eraser_log_file_b);
            self.cb_correct_ctf.set_selected(
                self.is_ctf_correction_log_file_a && self.is_ctf_correction_log_file_b,
            );
            self.cb_erase_gold.set_selected(
                self.is_gold_eraser_log_file_a
                    && self.is_gold_eraser_log_file_b
                    && self.is_gold_eraser_com_file_a
                    && self.is_gold_eraser_com_file_b,
            );
            self.cb_filter_in_2d
                .set_selected(self.is_mtf_filter_log_file_a && self.is_mtf_filter_log_file_b);
        } else {
            self.cb_preprocess.set_selected(self.is_eraser_log_file);
            self.cb_correct_ctf
                .set_selected(self.is_ctf_correction_log_file);
            self.cb_erase_gold
                .set_selected(self.is_gold_eraser_log_file && self.is_gold_eraser_com_file);
            self.cb_filter_in_2d
                .set_selected(self.is_mtf_filter_log_file);
            self.cb_trim_volume
                .set_selected(!self.tomogram_state.post_proc_trim_vol_input_n_rows_null);
        }
    }
    pub fn get_parameters_metadata(&mut self, metadata: &mut AltStackMetaDataBoundary) {
        self.cpu_gpu_panel.get_parameters_metadata(metadata);
        metadata.alt_tomo_rootname = self.ltf_rootname_of_alt_stack.get_text();
        metadata.alt_tomo_trim_volume = self.cb_trim_volume.is_selected();
        metadata.alt_tomo_archive_orig_stack = self.cb_archive_orig_stack.is_selected();
    }
    pub fn get_parameters(
        &self,
        param: &mut AltTomoSetupParamBoundary,
        do_validation: bool,
    ) -> bool {
        if self.cb_process_even_odd_pairs.check_box.visible
            && self.cb_process_even_odd_pairs.is_enabled()
            && self.cb_process_even_odd_pairs.is_selected()
        {
            param.even_and_odd_pairs = Some(true);
            param.rootname_to_process = None;
        } else {
            match self
                .ltf_rootname_of_alt_stack
                .get_text_validated(do_validation)
            {
                Ok(value) => {
                    param.rootname_to_process = Some(value);
                    param.even_and_odd_pairs = None;
                }
                Err(_) => return false,
            }
        }
        param.axis_to_process = if self
            .rb_a_only
            .as_ref()
            .is_some_and(|v| v.is_enabled() && v.is_selected())
        {
            Some("A".into())
        } else if self
            .rb_b_only
            .as_ref()
            .is_some_and(|v| v.is_enabled() && v.is_selected())
        {
            Some("B".into())
        } else {
            None
        };
        param.preprocess_for_extremes = Some(
            if self.cb_preprocess.is_enabled() && self.cb_preprocess.is_selected() {
                if self.cb_archive_orig_stack.is_selected() {
                    2
                } else {
                    1
                }
            } else {
                0
            },
        );
        param.correct_ctf = self.cb_correct_ctf.is_enabled() && self.cb_correct_ctf.is_selected();
        param.erase_fiducials = self.cb_erase_gold.is_enabled() && self.cb_erase_gold.is_selected();
        param.filter_in_2d =
            self.cb_filter_in_2d.is_enabled() && self.cb_filter_in_2d.is_selected();
        param.trim_volume = self.cb_trim_volume.is_enabled() && self.cb_trim_volume.is_selected();
        param.clean_up_intermediates = self.cb_clean_up_intermediate_files.is_enabled()
            && self.cb_clean_up_intermediate_files.is_selected();
        true
    }
    pub fn set_parameters_metadata(&mut self, metadata: &AltStackMetaDataBoundary) {
        self.cpu_gpu_panel.set_parameters_metadata(metadata);
        self.ltf_rootname_of_alt_stack
            .set_text(&metadata.alt_tomo_rootname);
        self.cb_trim_volume
            .set_selected(metadata.alt_tomo_trim_volume);
        self.cb_archive_orig_stack
            .set_selected(metadata.alt_tomo_archive_orig_stack);
        self.update_display();
    }
    pub fn set_parameters_tilt(&mut self, tilt: &AltStackTiltParamBoundary, initialize: bool) {
        self.cpu_gpu_panel.set_parameters_tilt(tilt, initialize);
    }
    pub fn set_parameters(&mut self, param: &AltTomoSetupParamBoundary) {
        self.cb_process_even_odd_pairs
            .set_selected(param.even_and_odd_pairs.unwrap_or(false));
        if let Some(root) = &param.rootname_to_process {
            self.ltf_rootname_of_alt_stack.set_text(root);
        }
        if let Some(axis) = &param.axis_to_process {
            if axis.eq_ignore_ascii_case("b") {
                if let Some(button) = &mut self.rb_b_only {
                    button.set_selected(true);
                }
            }
            if axis.eq_ignore_ascii_case("a") {
                if let Some(button) = &mut self.rb_a_only {
                    button.set_selected(true);
                }
            }
        }
        if let Some(value) = param.preprocess_for_extremes {
            self.cb_preprocess.set_selected(value != 0);
            self.cb_archive_orig_stack.set_enabled(value == 2);
            self.cb_archive_orig_stack.set_selected(value == 2);
        }
        self.cb_correct_ctf.set_selected(param.correct_ctf);
        self.cb_erase_gold.set_selected(param.erase_fiducials);
        self.cb_filter_in_2d.set_selected(param.filter_in_2d);
        self.cb_trim_volume.set_selected(param.trim_volume);
        self.cb_clean_up_intermediate_files
            .set_selected(param.clean_up_intermediates);
    }
    fn enable_or_disable_cb_erase_gold(&mut self) {
        let enabled = if self.dual_axis {
            if self.rb_both.as_ref().is_some_and(RadioButton::is_selected) {
                self.is_gold_eraser_com_file_a && self.is_gold_eraser_com_file_b
            } else if self
                .rb_a_only
                .as_ref()
                .is_some_and(RadioButton::is_selected)
            {
                self.is_gold_eraser_com_file_a
            } else if self
                .rb_b_only
                .as_ref()
                .is_some_and(RadioButton::is_selected)
            {
                self.is_gold_eraser_com_file_b
            } else {
                self.cb_erase_gold.is_enabled()
            }
        } else {
            self.is_gold_eraser_com_file
        };
        self.cb_erase_gold.set_enabled(enabled);
    }
    pub fn get_parameters_tilt(&mut self, tilt: &mut AltStackTiltParamBoundary) -> bool {
        if !self.is_dialog_opened_first_time {
            self.cpu_gpu_panel.get_parameters_tilt(tilt);
            true
        } else {
            false
        }
    }
    pub fn get_alt_stack_display(&self) -> &Self {
        self
    }
    pub fn get_axis_id(&self) -> Option<AxisID> {
        if self.rb_both.is_none() {
            Some(AxisID::Only)
        } else if self
            .rb_a_only
            .as_ref()
            .is_some_and(RadioButton::is_selected)
        {
            Some(AxisID::First)
        } else if self
            .rb_b_only
            .as_ref()
            .is_some_and(RadioButton::is_selected)
        {
            Some(AxisID::Second)
        } else {
            None
        }
    }
    pub fn set_use_queue_check_box(&mut self) {}
    pub fn add_queue_table_listener(&mut self) {}
    pub fn remove_queue_table_listener(&mut self) {}
    pub fn queue_table_event_action(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        dual: bool,
        state: AltStackTomogramState,
        calls: Vec<String>,
    }
    impl AltStackPanelApplicationManager for Manager {
        fn is_dual_axis(&self) -> bool {
            self.dual
        }
        fn state(&self) -> AltStackTomogramState {
            self.state
        }
        fn alt_tomo_setup(&mut self, axis: AxisID, _: DialogType, _: Option<ProcessingMethod>) {
            self.calls.push(format!("run:{}", axis.key()));
        }
        fn alt_tomo_setup_restore_swapped_files(
            &mut self,
            axis: AxisID,
            _: DialogType,
            _: Option<ProcessingMethod>,
        ) {
            self.calls.push(format!("restore:{}", axis.key()));
        }
        fn open_alternative_tomogram(
            &mut self,
            root: &str,
            axis: AxisID,
            dual: bool,
            _: Option<Run3dmodMenuOptions>,
            trimmed: bool,
        ) {
            self.calls
                .push(format!("open:{root}:{}:{dual}:{trimmed}", axis.key()));
        }
        fn open_even_odd_files_in_imod(
            &mut self,
            axis: AxisID,
            full: bool,
            _: Option<Run3dmodMenuOptions>,
            trimmed: bool,
        ) {
            self.calls
                .push(format!("evenodd:{}:{full}:{trimmed}", axis.key()));
        }
    }
    #[test]
    fn display_and_even_odd_dispatch_follow_source() {
        let mut panel = AltStackPanel::get_instance(
            AxisID::Only,
            DialogType::TomogramGeneration,
            false,
            false,
            false,
            AltStackTomogramState::default(),
        );
        assert!(panel.panel_created);
        assert_eq!(
            panel.btn_open_recon_in_3dmod_1.get_text(),
            Some(BUTTON_OPEN_RECON_ALTERNATIVE_LABEL)
        );
        panel.cb_process_even_odd_pairs.set_selected(true);
        let cmd = panel
            .cb_process_even_odd_pairs
            .get_action_command()
            .unwrap()
            .to_owned();
        let mut manager = Manager::default();
        panel.action(&mut manager, Some(&cmd), None, true, true);
        assert!(panel.even_odd_pairs_files_found);
        assert!(!panel.ltf_rootname_of_alt_stack.is_enabled());
        let open = panel
            .btn_open_recon_in_3dmod_1
            .get_text()
            .unwrap()
            .to_owned();
        panel.action(&mut manager, Some(&open), None, true, true);
        assert_eq!(manager.calls, ["evenodd:Only:true:false"]);
    }
    #[test]
    fn parameter_validation_and_dual_axis_selection_follow_source() {
        let mut panel = AltStackPanel::get_instance(
            AxisID::First,
            DialogType::TomogramGeneration,
            true,
            false,
            false,
            AltStackTomogramState::default(),
        );
        panel.ltf_rootname_of_alt_stack.set_text("alt");
        panel.rb_a_only.as_mut().unwrap().set_selected(true);
        panel.cb_preprocess.set_selected(true);
        panel.cb_archive_orig_stack.set_selected(true);
        let mut param = AltTomoSetupParamBoundary::default();
        assert!(panel.get_parameters(&mut param, true));
        assert_eq!(param.rootname_to_process.as_deref(), Some("alt"));
        assert_eq!(param.axis_to_process.as_deref(), Some("A"));
        assert_eq!(param.preprocess_for_extremes, Some(2));
        assert_eq!(panel.get_axis_id(), Some(AxisID::First));
    }
}
