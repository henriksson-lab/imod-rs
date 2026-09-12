//! `IMOD/Etomo/src/etomo/ui/swing/SetupCombinePanel.java`.
//!
//! Swing construction, MRC-header reads, the process-result factory, and the
//! concrete `ApplicationManager`/`TomogramCombinationDialog` calls are GUI and
//! application boundaries.  The source-owned controls, validation order,
//! matching-direction coordinate swap, and action dispatch remain here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use super::check_box::CheckBox;
use super::final_combine_fields::FinalCombineFields;
use super::final_combine_panel::{NO_VOLCOMBINE_TITLE, VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP};
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::patch_size_panel::{
    CombineParams as PatchSizeCombineParams, ConstCombineParams as PatchSizeConstCombineParams,
    PatchSizePanel,
};
use super::radio_button::{RadioButton, RadioButtonGroup};
use super::solvematch_panel::{CombineParameters, FiducialMatch, SolvematchPanel};
use crate::imod::etomo::r#type::{
    axis_id::AxisID, dialog_type::DialogType, processing_method::ProcessingMethod,
};
use crate::imod::etomo::ui::field_type::FieldType;

pub const TOMOGRAM_SIZE_CHANGED_STRING: &str =
    "THE TOMOGRAM HAS CHANGED - check min and max values";

/// Java `MatchMode`, whose original typesafe enum is only consumed by this unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchMode {
    AToB,
    BToA,
}

/// Java `TomogramState` calls made by `isChanged`.
pub trait SetupCombineTomogramState {
    fn combine_scripts_created(&self) -> bool;
    fn combine_match_mode(&self) -> Option<MatchMode>;
}

/// Direct `CombineParams` calls.  It extends the existing patch-size and
/// solvematch parameter boundaries just as Java `CombineParams` implements
/// both groups of accessors.
pub trait SetupCombineParams: PatchSizeCombineParams + CombineParameters {
    fn match_mode(&self) -> Option<MatchMode>;
    fn use_patch_region_model(&self) -> bool;
    fn patch_x_min(&self) -> String;
    fn patch_x_max(&self) -> String;
    fn patch_y_min(&self) -> String;
    fn patch_y_max(&self) -> String;
    fn patch_z_min(&self) -> String;
    fn patch_z_max(&self) -> String;
    fn max_patch_z_max(&self) -> i32;
    fn temp_directory(&self) -> String;
    fn manual_cleanup(&self) -> bool;
    fn extra_residual_targets(&self) -> Option<String>;
    fn set_extra_residual_targets(&mut self, value: String);
    fn reset_extra_residual_targets(&mut self);
    fn set_match_mode(&mut self, b_to_a: bool);
    fn set_default_patch_region_model(&mut self);
    fn set_patch_region_model(&mut self, value: String);
    fn set_patch_x_min(&mut self, value: i32);
    fn set_patch_x_max(&mut self, value: i32);
    fn set_patch_y_min(&mut self, value: i32);
    fn set_patch_y_max(&mut self, value: i32);
    fn set_patch_z_min(&mut self, value: String);
    fn set_patch_z_max(&mut self, value: String);
    fn set_max_patch_z_max(&mut self, value: i32);
    fn set_temp_directory(&mut self, value: String);
    fn set_manual_cleanup(&mut self, value: bool);
}

/// Java `ConstPatchcrawl3DParam` accessors.
pub trait SetupCombinePatchcrawl3DParam: PatchSizeConstCombineParams {
    fn x_low(&self) -> String;
    fn x_high(&self) -> String;
    fn y_low(&self) -> String;
    fn y_high(&self) -> String;
    fn z_low(&self) -> String;
    fn z_high(&self) -> String;
}

/// MRC read boundary used by `setAutoPatchZ` and `resetXandY`.
pub trait SetupCombineMrcHeader {
    fn read(&mut self) -> Result<bool, String>;
    fn n_rows(&self) -> i32;
    fn n_columns(&self) -> i32;
    fn n_sections(&self) -> i32;
    fn xy_border(&self) -> i32;
}

/// Direct manager calls from `SetupCombinePanel`.
pub trait SetupCombineApplicationManager {
    fn tomogram_size_changed(&self, match_b_to_a: bool, axis: AxisID) -> bool;
    fn create_combine_scripts(&mut self, button: &MultiLineButton) -> bool;
    fn combine(
        &mut self,
        button: &MultiLineButton,
        dialog_type: DialogType,
        initial: bool,
        parallel: bool,
        no_volcombine: bool,
    );
    fn imod_patch_region_model(&mut self);
    fn imod_full_volume(&mut self, axis: AxisID);
    fn set_processing_method(&mut self, method: ProcessingMethod);
}

/// Direct `TomogramCombinationDialog` calls.
pub trait SetupCombinePanelParent {
    fn synchronize(&mut self, include_this: bool);
    fn synchronize_from_current_tab(&mut self);
    fn update_display(&mut self);
}

/// Source-visible Swing hierarchy retained as renderer input.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SetupCombinePanelLayout {
    pub root_visible: bool,
    pub to_selector_visible: bool,
    pub patch_and_min_max_visible: bool,
    pub volcombine_controls_visible: bool,
    pub temp_directory_visible: bool,
    pub to_selector_body_visible: bool,
    pub patch_and_min_max_body_visible: bool,
    pub volcombine_body_visible: bool,
    pub temp_directory_body_visible: bool,
    pub tomogram_size_warning_visible: bool,
    pub tomogram_size_warning: String,
    pub binning_warning: String,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub component_order: Vec<&'static str>,
}

/// Java `SetupCombinePanel` fields, with native widget construction kept at
/// the presentation boundary.
pub struct SetupCombinePanel {
    pub dialog_type: DialogType,
    pub pnl_root: SetupCombinePanelLayout,
    pub bg_to_selector: Rc<RefCell<RadioButtonGroup>>,
    pub rb_b_to_a: RadioButton,
    pub rb_a_to_b: RadioButton,
    pub cb_patch_region_model: CheckBox,
    pub btn_patch_region_model: MultiLineButton,
    pub ltf_x_min: LabeledTextField,
    pub ltf_x_max: LabeledTextField,
    pub ltf_y_min: LabeledTextField,
    pub ltf_y_max: LabeledTextField,
    pub ltf_z_min: LabeledTextField,
    pub ltf_z_max: LabeledTextField,
    pub ltf_temp_directory: LabeledTextField,
    pub cb_manual_cleanup: CheckBox,
    pub btn_imod_volume_a: MultiLineButton,
    pub btn_imod_volume_b: MultiLineButton,
    pub btn_defaults: MultiLineButton,
    pub cb_no_volcombine: CheckBox,
    pub cb_auto_patch_final_size: CheckBox,
    pub ltf_extra_residual_targets: LabeledTextField,
    pub psp_patch_type_or_xyz: PatchSizePanel,
    pub psp_auto_patch_final_size: PatchSizePanel,
    pub btn_create: MultiLineButton,
    pub btn_combine: MultiLineButton,
    pub cb_parallel_process: CheckBox,
    pub pnl_solvematch: SolvematchPanel,
    pub max_z_max: i32,
    pub processing_method_locked: bool,
    pub match_b_to_a: bool,
}

impl SetupCombinePanel {
    /// Java private constructor plus `getInstance` construction sequence.
    pub fn get_instance(dialog_type: DialogType, parallel_process_check_box_text: &str) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut panel = Self {
            dialog_type,
            pnl_root: SetupCombinePanelLayout {
                root_visible: true,
                to_selector_visible: true,
                patch_and_min_max_visible: true,
                volcombine_controls_visible: true,
                temp_directory_visible: true,
                to_selector_body_visible: true,
                patch_and_min_max_body_visible: true,
                volcombine_body_visible: true,
                temp_directory_body_visible: false,
                component_order: vec![
                    "effect-warning",
                    "to-selector",
                    "solvematch",
                    "patch-and-min-max",
                    "volcombine-controls",
                    "temp-directory",
                    "buttons",
                ],
                ..Default::default()
            },
            bg_to_selector: group.clone(),
            rb_b_to_a: RadioButton::new_in_group("Match the B tomogram to A", group.clone()),
            rb_a_to_b: RadioButton::new_in_group("Match the A tomogram to B", group),
            cb_patch_region_model: CheckBox::new_with_text("Use patch region model"),
            btn_patch_region_model: MultiLineButton::new_with_label(Some(
                "Create/Edit Patch Region Model",
            )),
            ltf_x_min: LabeledTextField::new(FieldType::Integer, "X axis min: "),
            ltf_x_max: LabeledTextField::new(FieldType::Integer, "X axis max: "),
            ltf_y_min: LabeledTextField::new(FieldType::Integer, "Y axis min: "),
            ltf_y_max: LabeledTextField::new(FieldType::Integer, "Y axis max: "),
            ltf_z_min: LabeledTextField::new(FieldType::Integer, "Z axis min: "),
            ltf_z_max: LabeledTextField::new(FieldType::Integer, "Z axis max: "),
            ltf_temp_directory: LabeledTextField::new(FieldType::String, "Temporary directory: "),
            cb_manual_cleanup: CheckBox::new_with_text("Manual cleanup"),
            btn_imod_volume_a: MultiLineButton::new_with_label(Some("3dmod Volume A")),
            btn_imod_volume_b: MultiLineButton::new_with_label(Some("3dmod Volume B")),
            btn_defaults: MultiLineButton::new_with_label(Some("Defaults")),
            cb_no_volcombine: CheckBox::new_with_text(NO_VOLCOMBINE_TITLE),
            cb_auto_patch_final_size: CheckBox::new_with_text("Use Automatic Patch Fitting"),
            ltf_extra_residual_targets: LabeledTextField::new(
                FieldType::String,
                "Extra warping limits: ",
            ),
            psp_patch_type_or_xyz: PatchSizePanel::get_instance(false),
            psp_auto_patch_final_size: PatchSizePanel::get_instance(true),
            btn_create: MultiLineButton::new_with_label(Some("Create Combine Scripts")),
            btn_combine: MultiLineButton::new_with_label(Some("Start Combine")),
            cb_parallel_process: CheckBox::new_with_text(parallel_process_check_box_text),
            pnl_solvematch: SolvematchPanel::get_instance(
                "Setup",
                "combine-setup-solvematch",
                dialog_type,
                false,
            ),
            max_z_max: 0,
            processing_method_locked: false,
            match_b_to_a: false,
        };
        panel.create_panel();
        panel.set_tool_tip_text();
        panel.add_listeners();
        panel
    }

    /// Java `createPanel`; layout is carried in `pnl_root` for a renderer.
    pub fn create_panel(&mut self) {
        for field in [
            &mut self.ltf_x_min,
            &mut self.ltf_x_max,
            &mut self.ltf_y_min,
            &mut self.ltf_y_max,
            &mut self.ltf_z_min,
            &mut self.ltf_z_max,
        ] {
            field.set_required(true);
            field.set_number_must_be_positive(true);
        }
        self.cb_auto_patch_final_size.set_selected(true);
        self.update_patch_region_model();
    }
    pub fn add_listeners(&mut self) {
        self.pnl_root.listener_count = 11;
    }
    pub fn remove_listeners(&mut self) {
        self.pnl_root.listener_count = self.pnl_root.listener_count.saturating_sub(2);
    }
    pub fn get_container(&self) -> &SetupCombinePanelLayout {
        &self.pnl_root
    }
    pub fn show<M: SetupCombineApplicationManager>(&mut self, manager: &M, enable_combine: bool) {
        self.pnl_solvematch.show(true);
        self.update_tomogram_size_warning(manager, enable_combine);
    }
    pub fn set_deferred_3dmod_buttons(&mut self) {
        self.pnl_solvematch
            .set_deferred_3dmod_buttons(Some(&self.btn_combine));
    }
    pub fn update_tomogram_size_warning<M: SetupCombineApplicationManager>(
        &mut self,
        manager: &M,
        _enable_combine: bool,
    ) {
        let changed = manager.tomogram_size_changed(self.match_b_to_a, AxisID::Only);
        self.pnl_root.tomogram_size_warning_visible = changed;
        if changed {
            self.pnl_root.tomogram_size_warning = TOMOGRAM_SIZE_CHANGED_STRING.into();
        }
    }
    pub fn get_combine_result_display(&self) -> &MultiLineButton {
        &self.btn_combine
    }
    pub fn get_metadata_parameters(&self) -> bool {
        self.cb_parallel_process.is_selected()
    }
    pub fn update_display<M: SetupCombineApplicationManager>(
        &mut self,
        manager: &M,
        enable_combine: bool,
    ) {
        self.btn_combine.set_enabled(enable_combine);
        self.update_tomogram_size_warning(manager, enable_combine);
        let auto = self.cb_auto_patch_final_size.is_selected();
        self.psp_auto_patch_final_size.set_enabled(auto);
        self.ltf_extra_residual_targets.set_enabled(auto);
        self.pnl_solvematch.update_display();
    }
    pub fn set_metadata_parameters(
        &mut self,
        combine_volcombine_parallel: Option<bool>,
        default_parallel: bool,
    ) {
        self.cb_parallel_process
            .set_enabled(!self.processing_method_locked);
        self.cb_parallel_process
            .set_selected(combine_volcombine_parallel.unwrap_or(default_parallel));
    }
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.processing_method_locked = lock;
        self.cb_parallel_process.set_enabled(!lock);
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected() {
            ProcessingMethod::PpCpu
        } else {
            ProcessingMethod::LocalCpu
        }
    }
    pub fn send_processing_method_message<M: SetupCombineApplicationManager>(
        &self,
        manager: &mut M,
    ) {
        manager.set_processing_method(self.get_processing_method());
    }
    pub fn set_no_volcombine(&mut self, value: bool) {
        self.cb_no_volcombine.set_selected(value);
    }
    pub fn is_no_volcombine(&self) -> bool {
        self.cb_no_volcombine.is_selected()
    }
    pub fn set_parallel(&mut self, value: bool) {
        self.cb_parallel_process.set_selected(value);
    }
    pub fn set_parallel_enabled(&mut self, value: bool) {
        self.cb_parallel_process.set_enabled(value);
    }
    pub fn is_parallel(&self) -> bool {
        self.cb_parallel_process.is_selected()
    }
    pub fn is_parallel_enabled(&self) -> bool {
        self.cb_parallel_process.is_enabled()
    }
    pub fn is_use_corresponding_points(&self) -> bool {
        self.pnl_solvematch.is_use_corresponding_points()
    }
    pub fn set_use_corresponding_points(&mut self, value: bool) {
        self.pnl_solvematch.set_use_corresponding_points(value);
    }
    pub fn is_enabled(&self) -> bool {
        true
    }
    pub fn is_initial_volume_matching(&self) -> bool {
        self.pnl_solvematch.is_initial_volume_matching()
    }
    pub fn set_initial_volume_matching(&mut self, value: bool) {
        self.pnl_solvematch.set_initial_volume_matching(value);
    }
    pub fn get_match_mode(&self) -> MatchMode {
        if self.rb_b_to_a.is_selected() {
            MatchMode::BToA
        } else {
            MatchMode::AToB
        }
    }
    pub fn set_match_mode(&mut self, value: Option<MatchMode>) {
        if let Some(value) = value {
            self.set_b_to_a(value);
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.root_visible = visible;
        self.pnl_root.to_selector_visible = visible;
        self.pnl_root.patch_and_min_max_visible = visible;
        self.pnl_root.volcombine_controls_visible = visible;
        self.pnl_root.temp_directory_visible = visible;
        self.pnl_solvematch.set_visible(visible);
    }
    pub fn expand_global(&mut self, _advanced: bool) {}
    pub fn expand(&mut self, header: &str, expanded: bool) {
        match header {
            "Tomogram Matching Relationship" => self.pnl_root.to_selector_body_visible = expanded,
            "Patch Parameters for Refining Alignment" => {
                self.pnl_root.patch_and_min_max_body_visible = expanded
            }
            "Volcombine Controls" => self.pnl_root.volcombine_body_visible = expanded,
            "Intermediate Data Storage" => self.pnl_root.temp_directory_body_visible = expanded,
            _ => {}
        }
    }
    pub fn set_b_to_a(&mut self, match_mode: MatchMode) {
        if match_mode == MatchMode::BToA {
            self.rb_b_to_a.set_selected(true);
            self.match_b_to_a = true;
        } else {
            self.rb_a_to_b.set_selected(true);
            self.match_b_to_a = false;
        }
    }
    pub fn set_combine_parameters<P: SetupCombineParams>(&mut self, params: &P, init: bool) {
        self.set_match_mode(params.match_mode());
        self.pnl_solvematch.set_combine_parameters(params, init);
        self.psp_patch_type_or_xyz
            .set_parameters_combine_params(params);
        self.psp_auto_patch_final_size
            .set_parameters_combine_params(params);
        self.cb_patch_region_model
            .set_selected(params.use_patch_region_model());
        self.ltf_x_min.set_text(&params.patch_x_min());
        self.ltf_x_max.set_text(&params.patch_x_max());
        self.ltf_y_min.set_text(&params.patch_y_min());
        self.ltf_y_max.set_text(&params.patch_y_max());
        self.ltf_z_min.set_text(&params.patch_z_min());
        self.ltf_z_max.set_text(&params.patch_z_max());
        self.max_z_max = params.max_patch_z_max();
        self.ltf_temp_directory.set_text(&params.temp_directory());
        self.cb_manual_cleanup.set_selected(params.manual_cleanup());
        if let Some(value) = params.extra_residual_targets() {
            self.ltf_extra_residual_targets.set_text(&value);
        }
        self.update_patch_region_model();
    }
    pub fn set_patchcrawl3d_parameters<P: SetupCombinePatchcrawl3DParam>(&mut self, params: &P) {
        self.psp_patch_type_or_xyz
            .set_parameters_combine_params(params);
        self.ltf_x_min.set_text(&params.x_low());
        self.ltf_x_max.set_text(&params.x_high());
        self.ltf_y_min.set_text(&params.z_low());
        self.ltf_y_max.set_text(&params.z_high());
        self.ltf_z_min.set_text(&params.y_low());
        self.ltf_z_max.set_text(&params.y_high());
    }
    pub fn get_combine_parameters<P: SetupCombineParams>(
        &self,
        params: &mut P,
        validate: bool,
    ) -> Result<bool, String> {
        if !self.psp_patch_type_or_xyz.get_parameters(params, validate)
            || !self
                .psp_auto_patch_final_size
                .get_parameters(params, validate)
        {
            return Ok(false);
        }
        if self.ltf_extra_residual_targets.is_enabled() {
            params.set_extra_residual_targets(
                self.ltf_extra_residual_targets
                    .get_text_validated(validate)
                    .map_err(|e| e.to_string())?,
            );
        } else {
            params.reset_extra_residual_targets();
        }
        params.set_match_mode(self.rb_b_to_a.is_selected());
        if !self.pnl_solvematch.get_combine_parameters(params, validate) {
            return Ok(false);
        }
        if self.cb_patch_region_model.is_selected() {
            params.set_default_patch_region_model();
        } else {
            params.set_patch_region_model(String::new());
        }
        let parse = |field: &LabeledTextField| -> Result<i32, String> {
            field
                .get_text_validated(validate)
                .map_err(|e| e.to_string())?
                .parse()
                .map_err(|e: std::num::ParseIntError| format!("{} {e}", field.get_label()))
        };
        params.set_patch_x_min(parse(&self.ltf_x_min)?);
        params.set_patch_x_max(parse(&self.ltf_x_max)?);
        params.set_patch_y_min(parse(&self.ltf_y_min)?);
        params.set_patch_y_max(parse(&self.ltf_y_max)?);
        params.set_patch_z_min(
            self.ltf_z_min
                .get_text_validated(validate)
                .map_err(|e| e.to_string())?,
        );
        params.set_patch_z_max(
            self.ltf_z_max
                .get_text_validated(validate)
                .map_err(|e| e.to_string())?,
        );
        params.set_max_patch_z_max(self.max_z_max);
        params.set_temp_directory(
            self.ltf_temp_directory
                .get_text_validated(validate)
                .map_err(|e| e.to_string())?,
        );
        params.set_manual_cleanup(self.cb_manual_cleanup.is_selected());
        Ok(true)
    }
    pub fn set_use_patch_region_model(&mut self, value: bool) {
        self.cb_patch_region_model.set_selected(value);
        self.update_patch_region_model();
    }
    pub fn is_use_patch_region_model(&self) -> bool {
        self.cb_patch_region_model.is_selected()
    }
    pub fn set_x_min(&mut self, value: &str) {
        self.ltf_x_min.set_text(value);
    }
    pub fn get_x_min(&self) -> String {
        self.ltf_x_min.get_text()
    }
    pub fn set_x_max(&mut self, value: &str) {
        self.ltf_x_max.set_text(value);
    }
    pub fn get_x_max(&self) -> String {
        self.ltf_x_max.get_text()
    }
    pub fn set_y_min(&mut self, value: &str) {
        self.ltf_y_min.set_text(value);
    }
    pub fn get_y_min(&self) -> String {
        self.ltf_y_min.get_text()
    }
    pub fn set_y_max(&mut self, value: &str) {
        self.ltf_y_max.set_text(value);
    }
    pub fn get_y_max(&self) -> String {
        self.ltf_y_max.get_text()
    }
    pub fn set_z_min(&mut self, value: &str) {
        self.ltf_z_min.set_text(value);
    }
    pub fn get_z_min(&self) -> String {
        self.ltf_z_min.get_text()
    }
    pub fn get_combine_process_result_display(&self) -> &MultiLineButton {
        &self.btn_combine
    }
    pub fn set_z_max(&mut self, value: &str) {
        self.ltf_z_max.set_text(value);
    }
    pub fn get_z_max(&self) -> String {
        self.ltf_z_max.get_text()
    }
    pub fn get_surfaces_or_models(&self) -> FiducialMatch {
        self.pnl_solvematch.get_surfaces_or_models()
    }
    pub fn set_surfaces_or_models(&mut self, value: FiducialMatch) {
        self.pnl_solvematch.set_surfaces_or_models(value);
    }
    pub fn set_binning_warning(&mut self, warning: bool) {
        self.pnl_root.binning_warning = if warning {
            "WARNING:  Coordinates must be selected from an unbinned 3dmod".into()
        } else {
            String::new()
        };
    }
    pub fn is_bin_by_2(&self) -> bool {
        self.pnl_solvematch.is_bin_by_2()
    }
    pub fn set_bin_by_2(&mut self, value: bool) {
        self.pnl_solvematch.set_bin_by_2(value);
    }
    pub fn set_fiducial_match_list_a(&mut self, value: &str) {
        self.pnl_solvematch.set_fiducial_match_list_a(value);
    }
    pub fn set_use_list(&mut self, value: &str) {
        self.pnl_solvematch.set_use_list(value);
    }
    pub fn get_use_list(&self, validate: bool) -> Result<String, FieldValidationFailedException> {
        self.pnl_solvematch.get_use_list(validate)
    }
    pub fn set_fiducial_match_list_b(&mut self, value: &str) {
        self.pnl_solvematch.set_fiducial_match_list_b(value);
    }
    pub fn get_fiducial_match_list_a(
        &self,
        validate: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.pnl_solvematch.get_fiducial_match_list_a(validate)
    }
    pub fn get_fiducial_match_list_b(
        &self,
        validate: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.pnl_solvematch.get_fiducial_match_list_b(validate)
    }
    pub fn action<M: SetupCombineApplicationManager, T: SetupCombinePanelParent>(
        &mut self,
        manager: &mut M,
        parent: &mut T,
        command: &str,
    ) {
        parent.synchronize(true);
        if Some(command) == self.btn_create.get_action_command() {
            let enable_combine = manager.create_combine_scripts(&self.btn_create);
            self.update_tomogram_size_warning(manager, enable_combine);
            parent.update_display();
        } else if Some(command) == self.btn_combine.get_action_command() {
            manager.combine(
                &self.btn_combine,
                self.dialog_type,
                self.pnl_solvematch.is_initial_volume_matching(),
                self.is_parallel(),
                self.is_no_volcombine(),
            );
        } else if Some(command) == self.cb_parallel_process.get_action_command() {
            self.send_processing_method_message(manager);
        } else if Some(command) == self.btn_defaults.get_action_command() {
            parent.update_display();
        } else if Some(command) == self.btn_patch_region_model.get_action_command() {
            manager.imod_patch_region_model();
        } else if Some(command) == self.btn_imod_volume_a.get_action_command() {
            manager.imod_full_volume(AxisID::First);
        } else if Some(command) == self.btn_imod_volume_b.get_action_command() {
            manager.imod_full_volume(AxisID::Second);
        } else {
            parent.update_display();
        }
    }
    pub fn set_auto_patch_z<H: SetupCombineMrcHeader>(&mut self, header: &mut H) {
        if self.cb_auto_patch_final_size.is_selected() {
            if self.ltf_z_min.is_empty() {
                self.ltf_z_min.set_text("1");
            }
            if self.ltf_z_max.is_empty() && header.read().is_ok() {
                self.ltf_z_max.set_text_number(header.n_rows());
            }
        }
    }
    pub fn reset_x_and_y<H: SetupCombineMrcHeader, T: SetupCombinePanelParent>(
        &mut self,
        header: &mut H,
        parent: &mut T,
    ) {
        if header.read().unwrap_or(false) {
            let border = header.xy_border();
            self.ltf_x_min.set_text_number(border);
            self.ltf_x_max.set_text_number(header.n_columns() - border);
            self.ltf_y_min.set_text_number(border);
            self.ltf_y_max.set_text_number(header.n_sections() - border);
            parent.synchronize_from_current_tab();
        }
    }
    pub fn rb_match_to_action<T: SetupCombinePanelParent>(&mut self, parent: &mut T) {
        self.update_match_to();
        parent.update_display();
    }
    pub fn is_changed<M: SetupCombineApplicationManager, S: SetupCombineTomogramState>(
        &self,
        manager: &M,
        state: &S,
    ) -> bool {
        !state.combine_scripts_created()
            || state.combine_match_mode() != Some(self.get_match_mode())
            || manager.tomogram_size_changed(self.match_b_to_a, AxisID::Only)
    }
    pub fn update_match_to(&mut self) {
        if (self.match_b_to_a && self.rb_a_to_b.is_selected())
            || (!self.match_b_to_a && self.rb_b_to_a.is_selected())
        {
            std::mem::swap(&mut self.ltf_x_min.text, &mut self.ltf_y_min.text);
            std::mem::swap(&mut self.ltf_x_max.text, &mut self.ltf_y_max.text);
        }
        self.match_b_to_a = !self.rb_a_to_b.is_selected();
    }
    pub fn cb_patch_region_action(&mut self) {
        self.update_patch_region_model();
    }
    pub fn update_patch_region_model(&mut self) {
        self.btn_patch_region_model
            .set_enabled(self.cb_patch_region_model.is_selected());
    }
    pub fn pop_up_context_menu(&self) -> ContextPopup {
        ContextPopup {
            man_page_labels: vec!["Solvematch", "Matchshifts", "Patchcrawl3d", "Matchorwarp"],
            man_pages: vec![
                "solvematch.html",
                "matchshifts.html",
                "patchcrawl3d.html",
                "matchorwarp.html",
            ],
            log_file_labels: vec![
                "Transferfid",
                "Solvematch",
                "Patchcorr",
                "Matchorwarp",
                "Volcombine",
            ],
            log_files: vec![
                "transferfid.log",
                "solvematch.log",
                "patchcorr.log",
                "matchorwarp.log",
                "volcombine.log",
            ],
        }
    }
    pub fn set_tool_tip_text(&mut self) {
        self.rb_b_to_a.set_tool_tip_text(Some(
            "Transform the B tomogram into the same orientation as the A tomogram.",
        ));
        self.rb_a_to_b.set_tool_tip_text(Some(
            "Transform the A tomogram into the same orientation as the B tomogram.",
        ));
        self.cb_parallel_process
            .set_tool_tip_text(Some(VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP));
        self.pnl_root.tooltip_initialized = true;
    }
}

impl FinalCombineFields for SetupCombinePanel {
    fn set_use_patch_region_model(&mut self, use_patch_region_model: bool) {
        SetupCombinePanel::set_use_patch_region_model(self, use_patch_region_model);
    }
    fn is_use_patch_region_model(&self) -> bool {
        SetupCombinePanel::is_use_patch_region_model(self)
    }
    fn set_x_min(&mut self, x_min: &str) {
        SetupCombinePanel::set_x_min(self, x_min);
    }
    fn get_x_min(&self) -> String {
        SetupCombinePanel::get_x_min(self)
    }
    fn set_x_max(&mut self, x_max: &str) {
        SetupCombinePanel::set_x_max(self, x_max);
    }
    fn get_x_max(&self) -> String {
        SetupCombinePanel::get_x_max(self)
    }
    fn set_y_min(&mut self, y_min: &str) {
        SetupCombinePanel::set_y_min(self, y_min);
    }
    fn get_y_min(&self) -> String {
        SetupCombinePanel::get_y_min(self)
    }
    fn set_y_max(&mut self, y_max: &str) {
        SetupCombinePanel::set_y_max(self, y_max);
    }
    fn get_y_max(&self) -> String {
        SetupCombinePanel::get_y_max(self)
    }
    fn set_z_min(&mut self, z_min: &str) {
        SetupCombinePanel::set_z_min(self, z_min);
    }
    fn get_z_min(&self) -> String {
        SetupCombinePanel::get_z_min(self)
    }
    fn set_z_max(&mut self, z_max: &str) {
        SetupCombinePanel::set_z_max(self, z_max);
    }
    fn get_z_max(&self) -> String {
        SetupCombinePanel::get_z_max(self)
    }
    fn set_parallel(&mut self, parallel: bool) {
        SetupCombinePanel::set_parallel(self, parallel);
    }
    fn is_parallel(&self) -> bool {
        SetupCombinePanel::is_parallel(self)
    }
    fn set_parallel_enabled(&mut self, parallel_enabled: bool) {
        SetupCombinePanel::set_parallel_enabled(self, parallel_enabled);
    }
    fn is_parallel_enabled(&self) -> bool {
        SetupCombinePanel::is_parallel_enabled(self)
    }
    fn set_no_volcombine(&mut self, no_volcombine: bool) {
        SetupCombinePanel::set_no_volcombine(self, no_volcombine);
    }
    fn is_no_volcombine(&self) -> bool {
        SetupCombinePanel::is_no_volcombine(self)
    }
    fn is_enabled(&self) -> bool {
        SetupCombinePanel::is_enabled(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextPopup {
    pub man_page_labels: Vec<&'static str>,
    pub man_pages: Vec<&'static str>,
    pub log_file_labels: Vec<&'static str>,
    pub log_files: Vec<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matching_direction_swaps_xy_bounds() {
        let mut p = SetupCombinePanel::get_instance(DialogType::TomogramCombination, "parallel");
        p.set_b_to_a(MatchMode::BToA);
        p.set_x_min("1");
        p.set_y_min("2");
        p.set_x_max("3");
        p.set_y_max("4");
        p.rb_a_to_b.set_selected(true);
        p.update_match_to();
        assert_eq!(p.get_x_min(), "2");
        assert_eq!(p.get_y_max(), "3");
        assert!(!p.match_b_to_a);
    }
    #[test]
    fn patch_region_model_enables_its_button() {
        let mut p = SetupCombinePanel::get_instance(DialogType::TomogramCombination, "parallel");
        assert!(!p.btn_patch_region_model.is_enabled());
        p.set_use_patch_region_model(true);
        assert!(p.btn_patch_region_model.is_enabled());
    }
    #[test]
    fn auto_patch_fills_empty_z_values_from_mrc_header() {
        struct H;
        impl SetupCombineMrcHeader for H {
            fn read(&mut self) -> Result<bool, String> {
                Ok(true)
            }
            fn n_rows(&self) -> i32 {
                44
            }
            fn n_columns(&self) -> i32 {
                0
            }
            fn n_sections(&self) -> i32 {
                0
            }
            fn xy_border(&self) -> i32 {
                0
            }
        }
        let mut p = SetupCombinePanel::get_instance(DialogType::TomogramCombination, "parallel");
        p.set_auto_patch_z(&mut H);
        assert_eq!(p.get_z_min(), "1");
        assert_eq!(p.get_z_max(), "44");
    }
}
