//! `IMOD/Etomo/src/etomo/ui/swing/FinalCombinePanel.java`.
//!
//! Swing layout, autodoc lookup, file existence checks, and concrete
//! `ApplicationManager`/`TomogramCombinationDialog` invocations remain direct
//! presentation boundaries.  This source unit retains its controls, parameter
//! transfers, advanced/open state, validation ordering, and action dispatch.
#![allow(dead_code)]

use super::{
    check_box::CheckBox,
    final_combine_fields::FinalCombineFields,
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    multi_line_button::MultiLineButton,
    panel_header::{ExpandButton, PanelHeader},
};
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions, r#type::dialog_type::DialogType,
    ui::field_type::FieldType,
};

pub const NO_VOLCOMBINE_TITLE: &str = "Stop before running volcombine";
pub const VOLCOMBINE_PARALLEL_PROCESSING_TOOL_TIP: &str =
    "Check to distribute the volcombine process across multiple computers.";
pub const KERNEL_SIGMA_LABEL: &str = "Kernel filtering with sigma: ";

/// Java `ConstPatchcrawl3DParam` and mutable `Patchcrawl3DParam` boundary.
pub trait Patchcrawl3DParameters {
    fn get(&self, key: &str) -> Option<String>;
    fn is_kernel_sigma_active(&self) -> bool;
    fn set(&mut self, key: &str, value: String);
    fn set_use_boundary_model(&mut self, value: bool);
    fn set_kernel_sigma(&mut self, active: bool, value: String);
}
/// Java `ConstMatchorwarpParam` and `MatchorwarpParam` boundary.
pub trait MatchorwarpParameters {
    fn get(&self, key: &str) -> Option<String>;
    fn is_set(&self, key: &str) -> bool;
    fn linear_interpolation(&self) -> bool;
    fn use_model_file(&self) -> bool;
    fn set(&mut self, key: &str, value: String);
    fn reset(&mut self, key: &str);
    fn set_default_model_file(&mut self);
    fn set_model_file(&mut self, value: String);
    fn set_linear_interpolation(&mut self, value: bool);
}
/// Java `SetParam` direct calls.
pub trait SetParameters {
    fn is_valid(&self) -> bool;
    fn value(&self) -> String;
    fn set_value(&mut self, value: String);
}
/// Direct manager calls from `FinalCombinePanel.action`.
pub trait FinalCombinePanelApplicationManager {
    fn patchcorr_combine(
        &mut self,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        parallel: bool,
        not_run_volcombine: bool,
    );
    fn matchorwarp_combine(
        &mut self,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        parallel: bool,
        not_run_volcombine: bool,
    );
    fn matchorwarp_trial(&mut self);
    fn splitcombine(
        &mut self,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        parallel: bool,
        not_run_volcombine: bool,
    );
    fn volcombine(&mut self, options: Option<Run3dmodMenuOptions>, dialog_type: DialogType);
    fn imod_patch_vector_model(&mut self, correlation_coefficients: bool);
    fn model_to_patch(&mut self);
    fn imod_patch_region_model(&mut self, options: Option<Run3dmodMenuOptions>);
    fn imod_matched_to_tomogram(&mut self, options: Option<Run3dmodMenuOptions>);
    fn imod_combined_tomogram(&mut self, options: Option<Run3dmodMenuOptions>);
    fn set_processing_method_parallel(&mut self, parallel: bool);
}
/// Direct `TomogramCombinationDialog` calls.
pub trait FinalCombinePanelParent {
    fn synchronize_final(&mut self);
    fn final_tab_enabled(&self) -> bool;
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FinalCombinePanelLayout {
    pub root_visible: bool,
    /// Java `tomogramCombinationDialog.isTabEnabled(lblFinal)` boundary.
    pub final_tab_enabled: bool,
    pub patch_region_model_visible: bool,
    pub patchcorr_visible: bool,
    pub matchorwarp_visible: bool,
    pub volcombine_visible: bool,
    pub patch_region_model_body_visible: bool,
    pub patchcorr_body_visible: bool,
    pub matchorwarp_body_visible: bool,
    pub volcombine_body_visible: bool,
    pub boundary_visible: bool,
    pub initial_shift_visible: bool,
    pub kernel_sigma_visible: bool,
    pub refine_limit_visible: bool,
    pub linear_interpolation_visible: bool,
    pub reduction_factor_visible: bool,
    pub low_from_both_radius_visible: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub patch_vector_model_exists: bool,
    pub component_order: Vec<&'static str>,
}

/// Java `FinalCombinePanel` fields represented by their source labels.
pub struct FinalCombinePanel {
    pub dialog_type: DialogType,
    pub pnl_root: FinalCombinePanelLayout,
    pub patch_region_model_header: PanelHeader,
    pub patchcorr_header: PanelHeader,
    pub matchorwarp_header: PanelHeader,
    pub volcombine_header: PanelHeader,
    pub cb_use_patch_region_model: CheckBox,
    pub ltf_x_patch_size: LabeledTextField,
    pub ltf_y_patch_size: LabeledTextField,
    pub ltf_z_patch_size: LabeledTextField,
    pub ltf_x_n_patches: LabeledTextField,
    pub ltf_y_n_patches: LabeledTextField,
    pub ltf_z_n_patches: LabeledTextField,
    pub ltf_x_low: LabeledTextField,
    pub ltf_x_high: LabeledTextField,
    pub ltf_y_low: LabeledTextField,
    pub ltf_y_high: LabeledTextField,
    pub ltf_z_low: LabeledTextField,
    pub ltf_z_high: LabeledTextField,
    pub ltf_initial_shift_x: LabeledTextField,
    pub ltf_initial_shift_y: LabeledTextField,
    pub ltf_initial_shift_z: LabeledTextField,
    pub cb_kernel_sigma: CheckBox,
    pub tf_kernel_sigma: LabeledTextField,
    pub ltf_warp_limit: LabeledTextField,
    pub ltf_refine_limit: LabeledTextField,
    pub ltf_x_lower_exclude: LabeledTextField,
    pub ltf_x_upper_exclude: LabeledTextField,
    pub ltf_z_lower_exclude: LabeledTextField,
    pub ltf_z_upper_exclude: LabeledTextField,
    pub cb_use_linear_interpolation: CheckBox,
    pub cb_no_volcombine: CheckBox,
    pub cb_parallel_process: CheckBox,
    pub ltf_reduction_factor: LabeledTextField,
    pub ltf_low_from_both_radius: LabeledTextField,
    pub btn_patchcorr_restart: MultiLineButton,
    pub btn_matchorwarp_restart: MultiLineButton,
    pub btn_matchorwarp_trial: MultiLineButton,
    pub btn_volcombine_restart: MultiLineButton,
    pub btn_patch_size_increase: MultiLineButton,
    pub btn_patch_size_decrease: MultiLineButton,
    pub btn_patch_region_model: MultiLineButton,
    pub btn_patch_vector_model: MultiLineButton,
    pub btn_patch_vector_ccc_model: MultiLineButton,
    pub btn_replace_patch_out: MultiLineButton,
    pub btn_imod_matched_to: MultiLineButton,
    pub btn_imod_combined: MultiLineButton,
}

impl FinalCombinePanel {
    /// Java constructor `FinalCombinePanel(...)`; ProcessResultDisplayFactory is an explicit caller boundary.
    pub fn new(dialog_type: DialogType, parallel_process_check_box_text: &str) -> Self {
        let mut panel = Self {
            dialog_type,
            pnl_root: FinalCombinePanelLayout {
                root_visible: true,
                final_tab_enabled: true,
                patch_region_model_visible: true,
                patchcorr_visible: true,
                matchorwarp_visible: true,
                volcombine_visible: true,
                patch_region_model_body_visible: true,
                patchcorr_body_visible: true,
                matchorwarp_body_visible: true,
                volcombine_body_visible: true,
                boundary_visible: true,
                initial_shift_visible: true,
                kernel_sigma_visible: true,
                refine_limit_visible: true,
                linear_interpolation_visible: true,
                reduction_factor_visible: true,
                low_from_both_radius_visible: true,
                component_order: vec![
                    "patch-region-model",
                    "patchcorr",
                    "matchorwarp",
                    "volcombine",
                    "buttons",
                ],
                ..Default::default()
            },
            patch_region_model_header: PanelHeader::new(
                "Patch Region Model",
                true,
                true,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            patchcorr_header: PanelHeader::new(
                "Patchcorr Parameters",
                true,
                true,
                dialog_type,
                true,
                true,
                true,
                true,
                true,
            ),
            matchorwarp_header: PanelHeader::new(
                "Matchorwarp Parameters",
                true,
                true,
                dialog_type,
                true,
                true,
                true,
                true,
                true,
            ),
            volcombine_header: PanelHeader::new(
                "Volcombine Parameters",
                true,
                true,
                dialog_type,
                true,
                true,
                true,
                true,
                true,
            ),
            cb_use_patch_region_model: CheckBox::new_with_text("Use patch region model"),
            ltf_x_patch_size: LabeledTextField::new(FieldType::Integer, "X patch size :"),
            ltf_y_patch_size: LabeledTextField::new(FieldType::Integer, "Z patch size :"),
            ltf_z_patch_size: LabeledTextField::new(FieldType::Integer, "Y patch size :"),
            ltf_x_n_patches: LabeledTextField::new(FieldType::Integer, "Number of X patches :"),
            ltf_y_n_patches: LabeledTextField::new(FieldType::Integer, "Number of Z patches :"),
            ltf_z_n_patches: LabeledTextField::new(FieldType::Integer, "Number of Y patches :"),
            ltf_x_low: LabeledTextField::new(FieldType::Integer, "X Low :"),
            ltf_x_high: LabeledTextField::new(FieldType::Integer, "X high :"),
            ltf_y_low: LabeledTextField::new(FieldType::Integer, "Z Low :"),
            ltf_y_high: LabeledTextField::new(FieldType::Integer, "Z high :"),
            ltf_z_low: LabeledTextField::new(FieldType::Integer, "Y Low :"),
            ltf_z_high: LabeledTextField::new(FieldType::Integer, "Y high :"),
            ltf_initial_shift_x: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Initial shift in X:",
            ),
            ltf_initial_shift_y: LabeledTextField::new(FieldType::FloatingPoint, "Z:"),
            ltf_initial_shift_z: LabeledTextField::new(FieldType::FloatingPoint, "Y:"),
            cb_kernel_sigma: CheckBox::new_with_text(KERNEL_SIGMA_LABEL),
            tf_kernel_sigma: LabeledTextField::new(FieldType::FloatingPoint, KERNEL_SIGMA_LABEL),
            ltf_warp_limit: LabeledTextField::new(FieldType::String, "Warping residual limits: "),
            ltf_refine_limit: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Residual limit for single transform: ",
            ),
            ltf_x_lower_exclude: LabeledTextField::new(
                FieldType::Integer,
                "Number of columns to exclude on left (in X): ",
            ),
            ltf_x_upper_exclude: LabeledTextField::new(
                FieldType::Integer,
                "Number of columns to exclude on right (in X): ",
            ),
            ltf_z_lower_exclude: LabeledTextField::new(
                FieldType::Integer,
                "Number of rows to exclude on bottom (in Y): ",
            ),
            ltf_z_upper_exclude: LabeledTextField::new(
                FieldType::Integer,
                "Number of rows to exclude on top (in Y): ",
            ),
            cb_use_linear_interpolation: CheckBox::new_with_text("Use linear interpolation"),
            cb_no_volcombine: CheckBox::new_with_text(NO_VOLCOMBINE_TITLE),
            cb_parallel_process: CheckBox::new_with_text(parallel_process_check_box_text),
            ltf_reduction_factor: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Reduction factor for matching amplitudes in combined FFT: ",
            ),
            ltf_low_from_both_radius: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Radius below which to average components from both tomograms: ",
            ),
            btn_patchcorr_restart: MultiLineButton::new_with_label(Some("Restart at Patchcorr")),
            btn_matchorwarp_restart: MultiLineButton::new_with_label(Some(
                "Restart at Matchorwarp",
            )),
            btn_matchorwarp_trial: MultiLineButton::new_with_label(Some("Matchorwarp Trial Run")),
            btn_volcombine_restart: MultiLineButton::new_with_label(Some("Restart at Volcombine")),
            btn_patch_size_increase: MultiLineButton::new_with_label(Some("Patch Size +20%")),
            btn_patch_size_decrease: MultiLineButton::new_with_label(Some("Patch Size -20%")),
            btn_patch_region_model: MultiLineButton::new_with_label(Some(
                "Create/Edit Patch Region Model",
            )),
            btn_patch_vector_model: MultiLineButton::new_with_label(Some(
                "Examine Patch Vector Model",
            )),
            btn_patch_vector_ccc_model: MultiLineButton::new_with_label(Some(
                "Open Vector Model with Correlations",
            )),
            btn_replace_patch_out: MultiLineButton::new_with_label(Some("Replace Patch Vectors")),
            btn_imod_matched_to: MultiLineButton::new_with_label(Some(
                "Open Volume Being Matched To",
            )),
            btn_imod_combined: MultiLineButton::new_with_label(Some("Open Combined Volume")),
        };
        panel.tf_kernel_sigma.set_enabled(false);
        // Swing's AbstractButton supplies the label as its default action
        // command; retain that Java listener identity in the Rust boundary.
        for button in [
            &mut panel.btn_patchcorr_restart,
            &mut panel.btn_matchorwarp_restart,
            &mut panel.btn_matchorwarp_trial,
            &mut panel.btn_volcombine_restart,
            &mut panel.btn_patch_size_increase,
            &mut panel.btn_patch_size_decrease,
            &mut panel.btn_patch_region_model,
            &mut panel.btn_patch_vector_model,
            &mut panel.btn_patch_vector_ccc_model,
            &mut panel.btn_replace_patch_out,
            &mut panel.btn_imod_matched_to,
            &mut panel.btn_imod_combined,
        ] {
            let command = button.get_unformatted_label().map(str::to_owned);
            button.set_action_command(command.as_deref());
        }
        let parallel_command = panel.cb_parallel_process.get_text().map(str::to_owned);
        panel
            .cb_parallel_process
            .set_action_command(parallel_command.as_deref());
        panel
            .cb_kernel_sigma
            .set_action_command(Some(KERNEL_SIGMA_LABEL));
        panel.add_listeners();
        panel.set_tool_tip_text();
        panel
    }
    /// Java `removeListeners`.
    pub fn remove_listeners(&mut self) {
        self.pnl_root.listener_count = self.pnl_root.listener_count.saturating_sub(3);
    }
    /// Java constructor listener bindings.
    pub fn add_listeners(&mut self) {
        self.pnl_root.listener_count = 13;
    }
    pub fn update_advanced(&mut self, state: bool) {
        self.update_advanced_patchcorr(state);
        self.update_advanced_matchorwarp(state);
        self.update_advanced_volcombine(state);
    }
    pub fn update_advanced_patchcorr(&mut self, state: bool) {
        self.pnl_root.boundary_visible = state;
        self.pnl_root.initial_shift_visible = state;
        self.pnl_root.kernel_sigma_visible = state;
    }
    pub fn update_advanced_matchorwarp(&mut self, state: bool) {
        self.pnl_root.refine_limit_visible = state;
        self.pnl_root.linear_interpolation_visible = state;
        self.ltf_refine_limit.set_visible(state);
        self.cb_use_linear_interpolation.set_visible(state);
    }
    pub fn update_advanced_volcombine(&mut self, state: bool) {
        self.pnl_root.reduction_factor_visible = state;
        self.pnl_root.low_from_both_radius_visible = state;
        self.ltf_reduction_factor.set_visible(state);
        self.ltf_low_from_both_radius.set_visible(state);
    }
    pub fn get_patchcorr_process_result_display(&self) -> &MultiLineButton {
        &self.btn_patchcorr_restart
    }
    pub fn get_imod_combined_button(&self) -> &MultiLineButton {
        &self.btn_imod_combined
    }
    pub fn get_matchorwarp_process_result_display(&self) -> &MultiLineButton {
        &self.btn_matchorwarp_restart
    }
    pub fn get_volcombine_process_result_display(&self) -> &MultiLineButton {
        &self.btn_volcombine_restart
    }
    pub fn set_use_patch_region_model(&mut self, value: bool) {
        self.cb_use_patch_region_model.set_selected(value);
    }
    pub fn is_use_patch_region_model(&self) -> bool {
        self.cb_use_patch_region_model.is_selected()
    }
    pub fn is_parallel(&self) -> bool {
        self.cb_parallel_process.is_selected()
    }
    pub fn is_parallel_enabled(&self) -> bool {
        self.cb_parallel_process.is_enabled()
    }
    pub fn set_x_min(&mut self, value: &str) {
        self.ltf_x_low.set_text(value);
    }
    pub fn get_x_min(&self) -> String {
        self.ltf_x_low.get_text()
    }
    pub fn set_x_max(&mut self, value: &str) {
        self.ltf_x_high.set_text(value);
    }
    pub fn get_x_max(&self) -> String {
        self.ltf_x_high.get_text()
    }
    /// Java's UI swaps command Y/Z labels deliberately.
    pub fn set_y_min(&mut self, value: &str) {
        self.ltf_z_low.set_text(value);
    }
    pub fn get_y_min(&self) -> String {
        self.ltf_z_low.get_text()
    }
    pub fn set_y_max(&mut self, value: &str) {
        self.ltf_z_high.set_text(value);
    }
    pub fn get_y_max(&self) -> String {
        self.ltf_z_high.get_text()
    }
    pub fn set_z_min(&mut self, value: &str) {
        self.ltf_y_low.set_text(value);
    }
    pub fn get_z_min(&self) -> String {
        self.ltf_y_low.get_text()
    }
    pub fn set_z_max(&mut self, value: &str) {
        self.ltf_y_high.set_text(value);
    }
    pub fn get_z_max(&self) -> String {
        self.ltf_y_high.get_text()
    }
    pub fn is_run_volcombine(&self) -> bool {
        !self.cb_no_volcombine.is_selected()
    }
    pub fn set_run_volcombine(&mut self, value: bool) {
        self.cb_no_volcombine.set_selected(!value);
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
    pub fn set_parameters_combine(
        &mut self,
        wedge_reduction_fraction: &str,
        low_from_both_radius: &str,
    ) {
        self.ltf_reduction_factor.set_text(wedge_reduction_fraction);
        self.ltf_low_from_both_radius.set_text(low_from_both_radius);
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.patch_region_model_visible = visible;
        self.pnl_root.patchcorr_visible = visible;
        self.pnl_root.matchorwarp_visible = visible;
        self.pnl_root.volcombine_visible = visible;
    }
    /// Java `expand(GlobalExpandButton)` deliberately has an empty body.
    pub fn expand_global(&mut self) {}
    pub fn expand(&mut self, button: &ExpandButton) {
        let open = button.is_expanded();
        if self
            .patch_region_model_header
            .btn_open_close
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.pnl_root.patch_region_model_body_visible = open;
        } else if self
            .patchcorr_header
            .btn_open_close
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.pnl_root.patchcorr_body_visible = open;
        } else if self
            .patchcorr_header
            .btn_advanced_basic
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.update_advanced_patchcorr(open);
        } else if self
            .matchorwarp_header
            .btn_open_close
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.pnl_root.matchorwarp_body_visible = open;
        } else if self
            .matchorwarp_header
            .btn_advanced_basic
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.update_advanced_matchorwarp(open);
        } else if self
            .volcombine_header
            .btn_open_close
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.pnl_root.volcombine_body_visible = open;
        } else if self
            .volcombine_header
            .btn_advanced_basic
            .as_ref()
            .is_some_and(|b| b.name == button.name)
        {
            self.update_advanced_volcombine(open);
        }
    }
    pub fn get_volcombine_button_name(&self) -> &'static str {
        "volcombine"
    }
    pub fn set_patchcrawl_3d_params<P: Patchcrawl3DParameters>(&mut self, p: &P) {
        for (key, f) in [
            ("xPatchSize", &mut self.ltf_x_patch_size),
            ("yPatchSize", &mut self.ltf_y_patch_size),
            ("zPatchSize", &mut self.ltf_z_patch_size),
            ("nX", &mut self.ltf_x_n_patches),
            ("nY", &mut self.ltf_y_n_patches),
            ("nZ", &mut self.ltf_z_n_patches),
            ("xLow", &mut self.ltf_x_low),
            ("xHigh", &mut self.ltf_x_high),
            ("yLow", &mut self.ltf_y_low),
            ("yHigh", &mut self.ltf_y_high),
            ("zLow", &mut self.ltf_z_low),
            ("zHigh", &mut self.ltf_z_high),
            ("initialShiftX", &mut self.ltf_initial_shift_x),
            ("initialShiftY", &mut self.ltf_initial_shift_y),
            ("initialShiftZ", &mut self.ltf_initial_shift_z),
        ] {
            if let Some(value) = p.get(key) {
                f.set_text(&value);
            }
        }
        self.cb_use_patch_region_model
            .set_selected(p.get("useBoundaryModel").as_deref() == Some("true"));
        self.cb_kernel_sigma
            .set_selected(p.is_kernel_sigma_active());
        if let Some(value) = p.get("kernelSigma") {
            self.tf_kernel_sigma.set_text(&value);
        }
        self.update_kernel_sigma();
    }
    pub fn set_reduction_factor_params<P: SetParameters>(&mut self, p: Option<&P>) {
        if let Some(p) = p.filter(|p| p.is_valid()) {
            self.ltf_reduction_factor.set_text(&p.value());
        }
    }
    pub fn set_low_from_both_radius_params<P: SetParameters>(&mut self, p: Option<&P>) {
        if let Some(p) = p.filter(|p| p.is_valid()) {
            self.ltf_low_from_both_radius.set_text(&p.value());
        }
    }
    pub fn get_reduction_factor_param<P: SetParameters>(
        &self,
        p: Option<&mut P>,
        validation: bool,
    ) -> bool {
        p.is_some_and(|p| {
            self.ltf_reduction_factor
                .get_text_validated(validation)
                .map(|v| {
                    p.set_value(v);
                })
                .is_ok()
        })
    }
    pub fn get_low_from_both_radius_param<P: SetParameters>(
        &self,
        p: Option<&mut P>,
        validation: bool,
    ) -> bool {
        p.is_some_and(|p| {
            self.ltf_low_from_both_radius
                .get_text_validated(validation)
                .map(|v| {
                    p.set_value(v);
                })
                .is_ok()
        })
    }
    pub fn enable_reduction_factor(&mut self, enabled: bool) {
        self.ltf_reduction_factor.set_enabled(enabled);
    }
    pub fn enable_low_from_both_radius(&mut self, enabled: bool) {
        self.ltf_low_from_both_radius.set_enabled(enabled);
    }
    pub fn get_patchcrawl_3d_params<P: Patchcrawl3DParameters>(
        &self,
        p: &mut P,
        validation: bool,
    ) -> Result<bool, String> {
        p.set_use_boundary_model(self.cb_use_patch_region_model.is_selected());
        for (label, key, f) in [
            ("X patch size :", "xPatchSize", &self.ltf_x_patch_size),
            ("Z patch size :", "yPatchSize", &self.ltf_y_patch_size),
            ("Y patch size :", "zPatchSize", &self.ltf_z_patch_size),
            ("Number of X patches :", "nX", &self.ltf_x_n_patches),
            ("Number of Z patches :", "nY", &self.ltf_y_n_patches),
            ("Number of Y patches :", "nZ", &self.ltf_z_n_patches),
            ("X Low :", "xLow", &self.ltf_x_low),
            ("X high :", "xHigh", &self.ltf_x_high),
            ("Z Low :", "yLow", &self.ltf_y_low),
            ("Z high :", "yHigh", &self.ltf_y_high),
            ("Y Low :", "zLow", &self.ltf_z_low),
            ("Y high :", "zHigh", &self.ltf_z_high),
        ] {
            let text = f
                .get_text_validated(validation)
                .map_err(|_| format!("{label} validation failed"))?;
            text.parse::<i32>().map_err(|e| format!("{label} {e}"))?;
            p.set(key, text);
        }
        for (key, f) in [
            ("initialShiftX", &self.ltf_initial_shift_x),
            ("initialShiftY", &self.ltf_initial_shift_y),
            ("initialShiftZ", &self.ltf_initial_shift_z),
        ] {
            p.set(
                key,
                f.get_text_validated(validation)
                    .map_err(|_| "initial shift validation failed".to_string())?,
            );
        }
        p.set_kernel_sigma(
            self.cb_kernel_sigma.is_selected(),
            self.tf_kernel_sigma
                .get_text_validated(validation)
                .map_err(|_| "Kernel filtering with sigma: validation failed".to_string())?,
        );
        Ok(true)
    }
    pub fn set_matchorwarp_params<P: MatchorwarpParameters>(&mut self, p: &P) {
        for (key, f) in [
            ("warpLimits", &mut self.ltf_warp_limit),
            ("refineLimit", &mut self.ltf_refine_limit),
            ("xLowerExclude", &mut self.ltf_x_lower_exclude),
            ("xUpperExclude", &mut self.ltf_x_upper_exclude),
            ("zLowerExclude", &mut self.ltf_z_lower_exclude),
            ("zUpperExclude", &mut self.ltf_z_upper_exclude),
        ] {
            if p.is_set(key) {
                if let Some(v) = p.get(key) {
                    f.set_text(&v);
                }
            }
        }
        self.cb_use_linear_interpolation
            .set_selected(p.linear_interpolation());
        self.cb_use_patch_region_model
            .set_selected(p.use_model_file());
    }
    pub fn get_matchorwarp_params<P: MatchorwarpParameters>(
        &self,
        p: &mut P,
        validation: bool,
    ) -> Result<bool, FieldValidationFailedException> {
        if self.cb_use_patch_region_model.is_selected() {
            p.set_default_model_file();
        } else {
            p.set_model_file(String::new());
        }
        for (key, f) in [
            ("warpLimits", &self.ltf_warp_limit),
            ("refineLimit", &self.ltf_refine_limit),
        ] {
            p.set(key, f.get_text_validated(validation)?);
        }
        for (key, f) in [
            ("xLowerExclude", &self.ltf_x_lower_exclude),
            ("xUpperExclude", &self.ltf_x_upper_exclude),
            ("zLowerExclude", &self.ltf_z_lower_exclude),
            ("zUpperExclude", &self.ltf_z_upper_exclude),
        ] {
            if f.is_empty() {
                p.reset(key);
            } else {
                p.set(key, f.get_text_validated(validation)?);
            }
        }
        p.set_linear_interpolation(self.cb_use_linear_interpolation.is_selected());
        Ok(true)
    }
    pub fn pop_up_context_menu(&self) -> (&'static str, [&'static str; 3]) {
        (
            "Patch Problems in Combining",
            ["patchcorr.log", "matchorwarp.log", "volcombine.log"],
        )
    }
    pub fn action<A: FinalCombinePanelApplicationManager, T: FinalCombinePanelParent>(
        &mut self,
        command: &str,
        manager: &mut A,
        parent: &mut T,
        options: Option<Run3dmodMenuOptions>,
    ) {
        parent.synchronize_final();
        if command
            == self
                .btn_patch_size_decrease
                .get_action_command()
                .unwrap_or_default()
        {
            for f in [
                &mut self.ltf_x_patch_size,
                &mut self.ltf_y_patch_size,
                &mut self.ltf_z_patch_size,
            ] {
                if let Ok(value) = f.get_text_validated(true).and_then(|v| {
                    v.parse::<i32>()
                        .map_err(|e| FieldValidationFailedException(e.to_string()))
                }) {
                    f.set_text_number(((value as f32 / 1.2).round()) as i32);
                }
            }
        } else if command
            == self
                .btn_patch_size_increase
                .get_action_command()
                .unwrap_or_default()
        {
            for f in [
                &mut self.ltf_x_patch_size,
                &mut self.ltf_y_patch_size,
                &mut self.ltf_z_patch_size,
            ] {
                if let Ok(value) = f.get_text_validated(true).and_then(|v| {
                    v.parse::<i32>()
                        .map_err(|e| FieldValidationFailedException(e.to_string()))
                }) {
                    f.set_text_number(((value as f32 * 1.2).round()) as i32);
                }
            }
        } else if command
            == self
                .btn_patchcorr_restart
                .get_action_command()
                .unwrap_or_default()
        {
            manager.patchcorr_combine(
                options,
                self.dialog_type,
                self.is_parallel(),
                !self.is_run_volcombine(),
            );
        } else if command
            == self
                .btn_matchorwarp_restart
                .get_action_command()
                .unwrap_or_default()
        {
            manager.matchorwarp_combine(
                options,
                self.dialog_type,
                self.is_parallel(),
                !self.is_run_volcombine(),
            );
        } else if command
            == self
                .btn_matchorwarp_trial
                .get_action_command()
                .unwrap_or_default()
        {
            manager.matchorwarp_trial();
        } else if command
            == self
                .btn_volcombine_restart
                .get_action_command()
                .unwrap_or_default()
        {
            if self.is_parallel() {
                manager.splitcombine(options, self.dialog_type, true, !self.is_run_volcombine());
            } else {
                manager.volcombine(options, self.dialog_type);
            }
        } else if command
            == self
                .btn_patch_vector_model
                .get_action_command()
                .unwrap_or_default()
        {
            manager.imod_patch_vector_model(false);
        } else if command
            == self
                .btn_patch_vector_ccc_model
                .get_action_command()
                .unwrap_or_default()
        {
            manager.imod_patch_vector_model(true);
        } else if command
            == self
                .btn_replace_patch_out
                .get_action_command()
                .unwrap_or_default()
        {
            manager.model_to_patch();
        } else if self
            .cb_parallel_process
            .equals_action_command(Some(command))
        {
            manager.set_processing_method_parallel(self.is_parallel());
        } else if self.cb_kernel_sigma.equals_action_command(Some(command)) {
            self.update_kernel_sigma();
        } else if command
            == self
                .btn_patch_region_model
                .get_action_command()
                .unwrap_or_default()
        {
            manager.imod_patch_region_model(options);
        } else if command
            == self
                .btn_imod_matched_to
                .get_action_command()
                .unwrap_or_default()
        {
            manager.imod_matched_to_tomogram(options);
        } else if command
            == self
                .btn_imod_combined
                .get_action_command()
                .unwrap_or_default()
        {
            manager.imod_combined_tomogram(options);
        }
    }
    pub fn get_processing_method(&self) -> &'static str {
        if self.cb_parallel_process.is_enabled() && self.cb_parallel_process.is_selected() {
            "PP_CPU"
        } else {
            "LOCAL_CPU"
        }
    }
    pub fn update_kernel_sigma(&mut self) {
        self.tf_kernel_sigma
            .set_enabled(self.cb_kernel_sigma.is_selected());
    }
    pub fn update_patch_vector_model_display(&mut self, exists: bool) {
        self.pnl_root.patch_vector_model_exists = exists;
        self.btn_patch_vector_model.set_enabled(exists);
        self.btn_replace_patch_out.set_enabled(exists);
    }
    pub fn set_tool_tip_text(&mut self) {
        self.pnl_root.tooltip_initialized = true;
    }
}

impl FinalCombineFields for FinalCombinePanel {
    fn set_use_patch_region_model(&mut self, use_patch_region_model: bool) {
        FinalCombinePanel::set_use_patch_region_model(self, use_patch_region_model);
    }
    fn is_use_patch_region_model(&self) -> bool {
        FinalCombinePanel::is_use_patch_region_model(self)
    }
    fn set_x_min(&mut self, x_min: &str) {
        FinalCombinePanel::set_x_min(self, x_min);
    }
    fn get_x_min(&self) -> String {
        FinalCombinePanel::get_x_min(self)
    }
    fn set_x_max(&mut self, x_max: &str) {
        FinalCombinePanel::set_x_max(self, x_max);
    }
    fn get_x_max(&self) -> String {
        FinalCombinePanel::get_x_max(self)
    }
    fn set_y_min(&mut self, y_min: &str) {
        FinalCombinePanel::set_y_min(self, y_min);
    }
    fn get_y_min(&self) -> String {
        FinalCombinePanel::get_y_min(self)
    }
    fn set_y_max(&mut self, y_max: &str) {
        FinalCombinePanel::set_y_max(self, y_max);
    }
    fn get_y_max(&self) -> String {
        FinalCombinePanel::get_y_max(self)
    }
    fn set_z_min(&mut self, z_min: &str) {
        FinalCombinePanel::set_z_min(self, z_min);
    }
    fn get_z_min(&self) -> String {
        FinalCombinePanel::get_z_min(self)
    }
    fn set_z_max(&mut self, z_max: &str) {
        FinalCombinePanel::set_z_max(self, z_max);
    }
    fn get_z_max(&self) -> String {
        FinalCombinePanel::get_z_max(self)
    }
    fn set_parallel(&mut self, parallel: bool) {
        FinalCombinePanel::set_parallel(self, parallel);
    }
    fn is_parallel(&self) -> bool {
        FinalCombinePanel::is_parallel(self)
    }
    fn set_parallel_enabled(&mut self, parallel_enabled: bool) {
        FinalCombinePanel::set_parallel_enabled(self, parallel_enabled);
    }
    fn is_parallel_enabled(&self) -> bool {
        FinalCombinePanel::is_parallel_enabled(self)
    }
    fn set_no_volcombine(&mut self, no_volcombine: bool) {
        FinalCombinePanel::set_no_volcombine(self, no_volcombine);
    }
    fn is_no_volcombine(&self) -> bool {
        FinalCombinePanel::is_no_volcombine(self)
    }
    fn is_enabled(&self) -> bool {
        self.pnl_root.final_tab_enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn y_and_z_boundary_coordinates_follow_java_ui_swap() {
        let mut p = FinalCombinePanel::new(DialogType::TomogramCombination, "parallel");
        p.set_y_min("7");
        p.set_z_min("9");
        assert_eq!(p.ltf_z_low.get_text(), "7");
        assert_eq!(p.ltf_y_low.get_text(), "9");
    }
    #[test]
    fn advanced_state_changes_source_groups() {
        let mut p = FinalCombinePanel::new(DialogType::TomogramCombination, "parallel");
        p.update_advanced(false);
        assert!(
            !p.pnl_root.boundary_visible
                && !p.pnl_root.refine_limit_visible
                && !p.pnl_root.reduction_factor_visible
        );
    }
    #[test]
    fn patch_size_actions_use_java_rounding() {
        let mut p = FinalCombinePanel::new(DialogType::TomogramCombination, "parallel");
        p.ltf_x_patch_size.set_text("10");
        p.ltf_y_patch_size.set_text("10");
        p.ltf_z_patch_size.set_text("10");
        struct A;
        impl FinalCombinePanelApplicationManager for A {
            fn patchcorr_combine(
                &mut self,
                _: Option<Run3dmodMenuOptions>,
                _: DialogType,
                _: bool,
                _: bool,
            ) {
            }
            fn matchorwarp_combine(
                &mut self,
                _: Option<Run3dmodMenuOptions>,
                _: DialogType,
                _: bool,
                _: bool,
            ) {
            }
            fn matchorwarp_trial(&mut self) {}
            fn splitcombine(
                &mut self,
                _: Option<Run3dmodMenuOptions>,
                _: DialogType,
                _: bool,
                _: bool,
            ) {
            }
            fn volcombine(&mut self, _: Option<Run3dmodMenuOptions>, _: DialogType) {}
            fn imod_patch_vector_model(&mut self, _: bool) {}
            fn model_to_patch(&mut self) {}
            fn imod_patch_region_model(&mut self, _: Option<Run3dmodMenuOptions>) {}
            fn imod_matched_to_tomogram(&mut self, _: Option<Run3dmodMenuOptions>) {}
            fn imod_combined_tomogram(&mut self, _: Option<Run3dmodMenuOptions>) {}
            fn set_processing_method_parallel(&mut self, _: bool) {}
        }
        struct T;
        impl FinalCombinePanelParent for T {
            fn synchronize_final(&mut self) {}
            fn final_tab_enabled(&self) -> bool {
                true
            }
        }
        let c = p
            .btn_patch_size_increase
            .get_action_command()
            .unwrap()
            .to_owned();
        p.action(&c, &mut A, &mut T, None);
        assert_eq!(p.ltf_x_patch_size.get_text(), "12");
    }
}
