//! `IMOD/Etomo/src/etomo/ui/swing/SqueezeVolPanel.java`.
//!
//! Swing construction, autodoc/MRC I/O, and concrete application-manager calls
//! are frontend boundaries.  This unit keeps the source-owned state, parameter
//! transfer, validation, enablement, and action routing explicit.
#![allow(dead_code)]

use super::{
    check_box::CheckBox,
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    multi_line_button::{BaseScreenState, MultiLineButton},
    radio_button::{RadioButton, RadioButtonGroup},
    text_efield::TextEfield,
};
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{axis_id::AxisID, dialog_type::DialogType},
    ui::field_type::FieldType,
};
use std::{cell::RefCell, rc::Rc};

pub const USE_TRIM_VOL_OUTPUT_LABEL: &str = "Use the trimvol output";
pub const USE_FLATTEN_OUTPUT_LABEL: &str = "Use the flatten output";
pub const REDUCE_BY_OVERALL_FACTOR_LABEL: &str = "Reduce by overall factor (>=1)";
pub const FACTOR_IN_Z_LABEL: &str = "Factor in Z ";
pub const FILTER_NONE_LABEL: &str = "None";
pub const FILTER_GAUSSIAN_LOW_PASS_LABEL: &str = "Gaussian low-pass";
pub const FILTER_DECONVOLUTION_LABEL: &str = "Deconvolution";
pub const LOW_PASS_CUTOFF_SIGMA_LABEL: &str = "Low pass cutoff and sigma (1/pixel) ";
pub const DECONVOLUTION_STRENGTH_LABEL: &str = "Deconvolution strength ";
pub const SNR_FALLOFF_LABEL: &str = "SNR falloff ";
pub const HIGH_PASS_FILTER_CUTOFF_LABEL: &str = "High pass filter cutoff (fraction of Nyquist) ";
pub const DEFOCUS_LABEL: &str = "Defocus (microns) ";
pub const PHASE_SHIFT_LABEL: &str = "Phase shift (degrees) ";
pub const DATA_MODE_OF_OUTPUT_LABEL: &str = "Data mode of output ";
pub const BTN_IMOD_REDUCE_FILT_VOL_LABEL: &str = "Open Output Volume in 3dmod";

/// Direct Java `ReduceFiltVolParam` calls.
pub trait ReduceFiltVolParam {
    fn set_input_file(&mut self, file: String, flipped: bool);
    fn set_reduction_factor(&mut self, value: String);
    fn reset_reduction_factor(&mut self);
    fn set_z_reduction_factor(&mut self, value: String);
    fn reset_z_reduction_factor(&mut self);
    fn set_low_pass_radius_sigma(&mut self, value: String, validation: bool) -> Option<String>;
    fn reset_low_pass_radius_sigma(&mut self);
    fn set_deconvolution_strength(&mut self, value: String);
    fn reset_deconvolution_strength(&mut self);
    fn set_snr_falloff(&mut self, value: String);
    fn reset_snr_falloff(&mut self);
    fn set_high_pass_nyquist(&mut self, value: String);
    fn reset_high_pass_nyquist(&mut self);
    fn set_defocus_in_microns(&mut self, value: String);
    fn reset_defocus_in_microns(&mut self);
    fn set_phase_shift(&mut self, value: String);
    fn reset_phase_shift(&mut self);
    fn set_mode_to_output(&mut self, value: String);
    fn set_setup_chunks_if_memory_error(&mut self, value: bool);
    fn set_output_file(&mut self, value: String);
}
/// Reads from Java `ConstReduceFiltVolParam`.
pub trait ConstReduceFiltVolParam {
    fn value(&self, key: &str) -> Option<String>;
    fn is_set(&self, key: &str) -> bool;
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConstSqueezevolParamBoundary {
    pub reduction_factor_x: String,
    pub reduction_factor_y: String,
    pub reduction_factor_z: String,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SqueezeVolMetaDataBoundary {
    pub post_squeeze_vol_input_trim_vol: bool,
    pub post_reduce_filt_vol_reduction_factor: String,
    pub post_reduce_filt_vol_z_reduction_factor: String,
    pub post_reduce_filt_vol_low_pass_radius_sigma: String,
    pub post_reduce_filt_vol_deconvolution_strength: String,
    pub post_reduce_filt_vol_snr_falloff: String,
    pub post_reduce_filt_vol_high_pass_nyquist: String,
    pub post_reduce_filt_vol_defocus_in_microns: String,
    pub post_reduce_filt_vol_phase_shift: String,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MakecomfileParamBoundary {
    pub input_file: Option<String>,
    pub reduction_factor: Option<String>,
}

/// The direct application-manager and FileType boundary used by this unit.
pub trait SqueezeVolPanelApplicationManager {
    fn is_squeezevol_flipped(&self) -> bool;
    fn is_trimvol_flipped(&self) -> bool;
    fn is_result_set_flatten_flipped(&self) -> bool;
    fn is_flatten_flipped(&self) -> bool;
    fn trim_vol_output_file_name(&self, axis_id: AxisID) -> String;
    fn flatten_output_file_name(&self, axis_id: AxisID) -> String;
    fn reduce_filt_vol_output_file_name(&self, axis_id: AxisID, reduction_factor: f64) -> String;
    fn reduce_filt_vol(&mut self, options: Option<Run3dmodMenuOptions>, dialog_type: DialogType);
    fn imod_reduced_filtered_volume(
        &mut self,
        options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        output_file: String,
    );
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SqueezeVolPanelLayout {
    pub root_border: Option<String>,
    pub root_component_order: Vec<&'static str>,
    pub filtering_component_order: Vec<&'static str>,
    pub button_component_order: Vec<&'static str>,
    pub listener_count: usize,
    pub tooltips_initialized: bool,
    pub context_popup_title: Option<String>,
}

/// Java final `SqueezeVolPanel` field state.
#[derive(Clone, Debug)]
pub struct SqueezeVolPanel {
    pub pnl_root: SqueezeVolPanelLayout,
    pub rb_input_file_trim_vol: RadioButton,
    pub rb_input_file_flatten_warp: RadioButton,
    pub cb_reduce_by_overall_factor: CheckBox,
    pub tf_reduce_by_overall_factor: TextEfield,
    pub ltf_factor_in_z: LabeledTextField,
    pub rb_filtering_none: RadioButton,
    pub rb_filtering_gaussian: RadioButton,
    pub rb_filtering_deconvolution: RadioButton,
    pub ltf_low_pass_cutoff_sigma: LabeledTextField,
    pub ltf_deconvolution_strength: LabeledTextField,
    pub ltf_snr_falloff: LabeledTextField,
    pub ltf_high_pass_filter_cutoff: LabeledTextField,
    pub ltf_defocus: LabeledTextField,
    pub ltf_phase_shift: LabeledTextField,
    pub ltf_data_mode_of_output: LabeledTextField,
    pub btn_reduce_filt_vol: MultiLineButton,
    pub btn_imod_reduce_filt_vol: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
}
impl SqueezeVolPanel {
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let input = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let filter = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root: SqueezeVolPanelLayout::default(),
            rb_input_file_trim_vol: RadioButton::new_in_group(
                USE_TRIM_VOL_OUTPUT_LABEL,
                input.clone(),
            ),
            rb_input_file_flatten_warp: RadioButton::new_in_group(USE_FLATTEN_OUTPUT_LABEL, input),
            cb_reduce_by_overall_factor: CheckBox::new_with_text(REDUCE_BY_OVERALL_FACTOR_LABEL),
            tf_reduce_by_overall_factor: TextEfield::new(
                REDUCE_BY_OVERALL_FACTOR_LABEL,
                Some(FieldType::FloatingPoint),
                true,
                false,
                true,
                true,
                false,
                false,
            ),
            ltf_factor_in_z: LabeledTextField::new(FieldType::FloatingPoint, FACTOR_IN_Z_LABEL),
            rb_filtering_none: RadioButton::new_in_group(FILTER_NONE_LABEL, filter.clone()),
            rb_filtering_gaussian: RadioButton::new_in_group(
                FILTER_GAUSSIAN_LOW_PASS_LABEL,
                filter.clone(),
            ),
            rb_filtering_deconvolution: RadioButton::new_in_group(
                FILTER_DECONVOLUTION_LABEL,
                filter,
            ),
            ltf_low_pass_cutoff_sigma: LabeledTextField::new(
                FieldType::FloatingPointPair,
                LOW_PASS_CUTOFF_SIGMA_LABEL,
            ),
            ltf_deconvolution_strength: LabeledTextField::new(
                FieldType::FloatingPoint,
                DECONVOLUTION_STRENGTH_LABEL,
            ),
            ltf_snr_falloff: LabeledTextField::new(FieldType::FloatingPoint, SNR_FALLOFF_LABEL),
            ltf_high_pass_filter_cutoff: LabeledTextField::new(
                FieldType::FloatingPoint,
                HIGH_PASS_FILTER_CUTOFF_LABEL,
            ),
            ltf_defocus: LabeledTextField::new(FieldType::FloatingPoint, DEFOCUS_LABEL),
            ltf_phase_shift: LabeledTextField::new(FieldType::FloatingPoint, PHASE_SHIFT_LABEL),
            ltf_data_mode_of_output: LabeledTextField::new(
                FieldType::FloatingPoint,
                DATA_MODE_OF_OUTPUT_LABEL,
            ),
            btn_reduce_filt_vol: MultiLineButton::new_with_label(Some("Reduce/Filter Volume")),
            btn_imod_reduce_filt_vol: MultiLineButton::new_with_label(Some(
                BTN_IMOD_REDUCE_FILT_VOL_LABEL,
            )),
            axis_id,
            dialog_type,
        }
    }
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }
    pub fn add_listeners(&mut self) {
        self.rb_input_file_trim_vol.add_action_listener();
        self.rb_input_file_flatten_warp.add_action_listener();
        self.cb_reduce_by_overall_factor.add_action_listener();
        self.ltf_factor_in_z.add_action_listener();
        self.rb_filtering_none.add_action_listener();
        self.rb_filtering_gaussian.add_action_listener();
        self.rb_filtering_deconvolution.add_action_listener();
        self.ltf_low_pass_cutoff_sigma.add_action_listener();
        self.ltf_deconvolution_strength.add_action_listener();
        self.ltf_snr_falloff.add_action_listener();
        self.ltf_high_pass_filter_cutoff.add_action_listener();
        self.ltf_defocus.add_action_listener();
        self.ltf_phase_shift.add_action_listener();
        self.ltf_data_mode_of_output.add_action_listener();
        self.btn_reduce_filt_vol.add_action_listener();
        self.btn_imod_reduce_filt_vol.add_action_listener();
        self.pnl_root.listener_count = 17;
    }
    pub fn pop_up_context_menu(&mut self) {
        self.pnl_root.context_popup_title = Some("Reducefilt".into());
    }
    pub fn create_panel(&mut self) {
        self.rb_input_file_trim_vol.set_selected(true);
        self.rb_filtering_none.set_selected(true);
        self.tf_reduce_by_overall_factor.set_preferred_width(70);
        self.ltf_factor_in_z.set_preferred_width(70, None);
        self.ltf_low_pass_cutoff_sigma
            .set_number_must_be_positive(true);
        self.ltf_deconvolution_strength.set_required(true);
        self.ltf_deconvolution_strength
            .set_number_must_be_positive(true);
        self.ltf_snr_falloff.set_number_must_be_positive(true);
        self.ltf_high_pass_filter_cutoff
            .set_number_must_be_positive(true);
        self.ltf_high_pass_filter_cutoff.set_minimum(0.0);
        self.ltf_high_pass_filter_cutoff.set_maximum(1.0);
        self.ltf_defocus.set_number_must_be_positive(true);
        self.pnl_root.root_border = Some("Reduce and/or Filter Volume".into());
        self.pnl_root.root_component_order = vec![
            "input-file",
            "reduce-by-factor",
            "filtering",
            "data-mode",
            "buttons",
        ];
        self.pnl_root.filtering_component_order = vec![
            "radio-buttons",
            "low-pass",
            "deconvolution",
            "high-pass",
            "defocus",
        ];
        self.pnl_root.button_component_order =
            vec!["reduce-filt-vol", "horizontal-glue", "imod-reduce-filt-vol"];
        self.update_display(false);
    }
    pub fn set_tool_tip_text(&mut self) {
        self.btn_reduce_filt_vol
            .set_tool_tip_text(Some("Run reducefiltvol on the input volume"));
        self.btn_imod_reduce_filt_vol
            .set_tool_tip_text(Some("View the reduced and/or filtered volume"));
        self.pnl_root.tooltips_initialized = true;
    }
    pub fn set_parameters_squeezevol(
        &mut self,
        param: &ConstSqueezevolParamBoundary,
        squeezevol_flipped: bool,
    ) {
        if param.reduction_factor_x != "1.25" {
            self.tf_reduce_by_overall_factor
                .set_text(param.reduction_factor_x.clone());
        }
        let reduction = if squeezevol_flipped {
            &param.reduction_factor_z
        } else {
            &param.reduction_factor_y
        };
        if reduction != "1.25" {
            self.ltf_factor_in_z.set_text(reduction);
        }
    }
    pub fn set_parameters_reduce_filt_vol<P: ConstReduceFiltVolParam>(
        &mut self,
        param: &P,
        dialog_not_exists: bool,
        com_file_exists: bool,
        trim_vol_pixel_area: usize,
        input_flipped: bool,
    ) {
        let file_too_big = dialog_not_exists && trim_vol_pixel_area > 2_000_000;
        if file_too_big {
            self.cb_reduce_by_overall_factor.set_selected(true);
        }
        let reduction = param.is_set("reduction-factor");
        let z_reduction = param.is_set("z-reduction-factor");
        if reduction || z_reduction {
            self.cb_reduce_by_overall_factor.set_selected(true);
            if let Some(v) = param.value("reduction-factor") {
                self.tf_reduce_by_overall_factor.set_text(v)
            }
            if let Some(v) = param.value("z-reduction-factor") {
                self.ltf_factor_in_z.set_text(&v)
            }
        } else if com_file_exists {
            self.cb_reduce_by_overall_factor.set_selected(false)
        }
        if let Some(v) = param.value("low-pass-radius-sigma") {
            self.rb_filtering_gaussian.set_selected(true);
            self.ltf_low_pass_cutoff_sigma.set_text(&v)
        }
        if let Some(v) = param.value("deconvolution-strength") {
            self.rb_filtering_deconvolution.set_selected(true);
            self.ltf_deconvolution_strength.set_text(&v)
        }
        for (key, field) in [
            ("snr-falloff", &mut self.ltf_snr_falloff),
            ("high-pass-nyquist", &mut self.ltf_high_pass_filter_cutoff),
            ("defocus-in-microns", &mut self.ltf_defocus),
            ("phase-shift", &mut self.ltf_phase_shift),
            ("mode-to-output", &mut self.ltf_data_mode_of_output),
        ] {
            if let Some(v) = param.value(key) {
                field.set_text(&v)
            }
        }
        self.update_display(input_flipped);
        if !com_file_exists && !file_too_big {
            self.cb_reduce_by_overall_factor.set_selected(
                !self.tf_reduce_by_overall_factor.is_empty()
                    || (!self.ltf_factor_in_z.is_empty() && self.ltf_factor_in_z.is_enabled()),
            );
        }
        self.update_display(input_flipped);
    }
    pub fn get_parameters_meta_data(&self, m: &mut SqueezeVolMetaDataBoundary) {
        m.post_squeeze_vol_input_trim_vol = self.rb_input_file_trim_vol.is_selected();
        m.post_reduce_filt_vol_reduction_factor = self.tf_reduce_by_overall_factor.get_text();
        m.post_reduce_filt_vol_z_reduction_factor = self.ltf_factor_in_z.text.clone();
        m.post_reduce_filt_vol_low_pass_radius_sigma = self.ltf_low_pass_cutoff_sigma.text.clone();
        m.post_reduce_filt_vol_deconvolution_strength =
            self.ltf_deconvolution_strength.text.clone();
        m.post_reduce_filt_vol_snr_falloff = self.ltf_snr_falloff.text.clone();
        m.post_reduce_filt_vol_high_pass_nyquist = self.ltf_high_pass_filter_cutoff.text.clone();
        m.post_reduce_filt_vol_defocus_in_microns = self.ltf_defocus.text.clone();
        m.post_reduce_filt_vol_phase_shift = self.ltf_phase_shift.text.clone();
    }
    pub fn set_parameters_meta_data(&mut self, m: &SqueezeVolMetaDataBoundary) {
        self.rb_input_file_trim_vol
            .set_selected(m.post_squeeze_vol_input_trim_vol);
        if !self.rb_input_file_trim_vol.is_selected() {
            self.rb_input_file_flatten_warp.set_selected(true)
        }
        self.tf_reduce_by_overall_factor
            .set_text(m.post_reduce_filt_vol_reduction_factor.clone());
        self.ltf_factor_in_z
            .set_text(&m.post_reduce_filt_vol_z_reduction_factor);
        self.ltf_low_pass_cutoff_sigma
            .set_text(&m.post_reduce_filt_vol_low_pass_radius_sigma);
        self.ltf_deconvolution_strength
            .set_text(&m.post_reduce_filt_vol_deconvolution_strength);
        self.ltf_snr_falloff
            .set_text(&m.post_reduce_filt_vol_snr_falloff);
        self.ltf_high_pass_filter_cutoff
            .set_text(&m.post_reduce_filt_vol_high_pass_nyquist);
        self.ltf_defocus
            .set_text(&m.post_reduce_filt_vol_defocus_in_microns);
        self.ltf_phase_shift
            .set_text(&m.post_reduce_filt_vol_phase_shift);
    }
    pub fn get_parameters_reduce_filt_vol<
        P: ReduceFiltVolParam,
        M: SqueezeVolPanelApplicationManager,
    >(
        &self,
        param: &mut P,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        let flipped = self.is_input_file_flipped(manager);
        let file = if self.rb_input_file_trim_vol.is_selected() {
            manager.trim_vol_output_file_name(self.axis_id)
        } else {
            manager.flatten_output_file_name(self.axis_id)
        };
        param.set_input_file(file, flipped);
        if self.cb_reduce_by_overall_factor.is_selected()
            && !self.tf_reduce_by_overall_factor.is_empty()
        {
            match self
                .tf_reduce_by_overall_factor
                .get_text_validated(do_validation)
            {
                Ok(v) => param.set_reduction_factor(v),
                Err(_) => return false,
            }
        } else {
            param.reset_reduction_factor()
        }
        if self.ltf_factor_in_z.is_enabled() && !self.ltf_factor_in_z.is_empty() {
            match self.ltf_factor_in_z.get_text_validated(do_validation) {
                Ok(v) => param.set_z_reduction_factor(v),
                Err(_) => return false,
            }
        } else {
            param.reset_z_reduction_factor()
        }
        if self.ltf_low_pass_cutoff_sigma.is_enabled() && !self.ltf_low_pass_cutoff_sigma.is_empty()
        {
            let v = match self
                .ltf_low_pass_cutoff_sigma
                .get_text_validated(do_validation)
            {
                Ok(v) => v,
                Err(_) => return false,
            };
            if let Some(error) = param.set_low_pass_radius_sigma(v, do_validation) {
                manager.open_message_dialog(
                    format!("\"{}\" {error}", LOW_PASS_CUTOFF_SIGMA_LABEL),
                    "Syntax Error",
                    self.axis_id,
                );
                return false;
            }
        } else {
            param.reset_low_pass_radius_sigma()
        }
        macro_rules! f {
            ($field:ident,$set:ident,$reset:ident) => {
                if self.$field.is_enabled() {
                    match self.$field.get_text_validated(do_validation) {
                        Ok(v) => param.$set(v),
                        Err(_) => return false,
                    }
                } else {
                    param.$reset()
                }
            };
        }
        f!(
            ltf_deconvolution_strength,
            set_deconvolution_strength,
            reset_deconvolution_strength
        );
        f!(ltf_snr_falloff, set_snr_falloff, reset_snr_falloff);
        f!(
            ltf_high_pass_filter_cutoff,
            set_high_pass_nyquist,
            reset_high_pass_nyquist
        );
        f!(
            ltf_defocus,
            set_defocus_in_microns,
            reset_defocus_in_microns
        );
        f!(ltf_phase_shift, set_phase_shift, reset_phase_shift);
        match self
            .ltf_data_mode_of_output
            .get_text_validated(do_validation)
        {
            Ok(v) => param.set_mode_to_output(v),
            Err(_) => return false,
        };
        param.set_setup_chunks_if_memory_error(true);
        match self.get_output_filename(do_validation, manager) {
            Ok(v) => param.set_output_file(v),
            Err(_) => return false,
        };
        true
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_screen_state(&mut self, screen_state: BaseScreenState) {
        self.btn_reduce_filt_vol.set_screen_state(screen_state);
    }
    pub fn done(&mut self) {
        self.pnl_root.listener_count = self.pnl_root.listener_count.saturating_sub(1)
    }
    pub fn get_component(&self) -> &SqueezeVolPanelLayout {
        &self.pnl_root
    }
    pub fn action<M: SqueezeVolPanelApplicationManager>(
        &mut self,
        command: &str,
        options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if command == "Reduce/Filter Volume" {
            manager.reduce_filt_vol(options, self.dialog_type)
        } else if command == BTN_IMOD_REDUCE_FILT_VOL_LABEL {
            if let Ok(file) = self.get_output_filename(false, manager) {
                manager.imod_reduced_filtered_volume(options, self.axis_id, file)
            }
        }
        let flipped = self.is_input_file_flipped(manager);
        self.update_display(flipped)
    }
    pub fn get_parameters_makecomfile<M: SqueezeVolPanelApplicationManager>(
        &self,
        param: &mut MakecomfileParamBoundary,
        do_validation: bool,
        manager: &M,
    ) -> bool {
        param.input_file = Some(if self.rb_input_file_trim_vol.is_selected() {
            manager.trim_vol_output_file_name(self.axis_id)
        } else {
            manager.flatten_output_file_name(self.axis_id)
        });
        if self.cb_reduce_by_overall_factor.is_selected()
            && !self.tf_reduce_by_overall_factor.is_empty()
        {
            if let Ok(v) = self
                .tf_reduce_by_overall_factor
                .get_text_validated(do_validation)
            {
                param.reduction_factor = Some(v)
            }
        }
        true
    }
    pub fn update_display(&mut self, input_flipped: bool) {
        self.tf_reduce_by_overall_factor
            .set_enabled(self.cb_reduce_by_overall_factor.is_selected());
        self.ltf_factor_in_z
            .set_enabled(self.cb_reduce_by_overall_factor.is_selected() && input_flipped);
        self.ltf_low_pass_cutoff_sigma
            .set_enabled(self.rb_filtering_gaussian.is_selected());
        let d = self.rb_filtering_deconvolution.is_selected();
        self.ltf_deconvolution_strength.set_enabled(d);
        self.ltf_snr_falloff.set_enabled(d);
        self.ltf_high_pass_filter_cutoff.set_enabled(d);
        self.ltf_defocus.set_enabled(d);
        self.ltf_phase_shift.set_enabled(d)
    }
    pub fn is_input_file_flipped<M: SqueezeVolPanelApplicationManager>(&self, m: &M) -> bool {
        if self.rb_input_file_trim_vol.is_selected() {
            m.is_trimvol_flipped()
        } else if m.is_result_set_flatten_flipped() && !m.is_flatten_flipped() {
            false
        } else {
            true
        }
    }
    pub fn get_reduce_filt_vol_display(&mut self) -> &mut Self {
        self
    }
    pub fn get_output_filename<M: SqueezeVolPanelApplicationManager>(
        &self,
        do_validation: bool,
        m: &M,
    ) -> Result<String, FieldValidationFailedException> {
        let mut factor = 1.0;
        if self.tf_reduce_by_overall_factor.is_enabled()
            && !self.tf_reduce_by_overall_factor.is_empty()
        {
            let v = self
                .tf_reduce_by_overall_factor
                .get_text_validated(do_validation)
                .map_err(FieldValidationFailedException)?;
            if let Ok(n) = v.parse() {
                factor = n
            }
        }
        Ok(m.reduce_filt_vol_output_file_name(self.axis_id, factor))
    }
    pub fn display(&self) {}
    pub fn display_ui_component(&self) {}
    pub fn set_field_displayer(&mut self) {}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        trim: bool,
        calls: Vec<String>,
    }
    impl SqueezeVolPanelApplicationManager for Manager {
        fn is_squeezevol_flipped(&self) -> bool {
            false
        }
        fn is_trimvol_flipped(&self) -> bool {
            self.trim
        }
        fn is_result_set_flatten_flipped(&self) -> bool {
            false
        }
        fn is_flatten_flipped(&self) -> bool {
            false
        }
        fn trim_vol_output_file_name(&self, _: AxisID) -> String {
            "trim.rec".into()
        }
        fn flatten_output_file_name(&self, _: AxisID) -> String {
            "flat.rec".into()
        }
        fn reduce_filt_vol_output_file_name(&self, _: AxisID, n: f64) -> String {
            format!("reduce{n}.rec")
        }
        fn reduce_filt_vol(&mut self, _: Option<Run3dmodMenuOptions>, _: DialogType) {
            self.calls.push("reduce".into())
        }
        fn imod_reduced_filtered_volume(
            &mut self,
            _: Option<Run3dmodMenuOptions>,
            _: AxisID,
            file: String,
        ) {
            self.calls.push(file)
        }
        fn open_message_dialog(&mut self, message: String, _: &str, _: AxisID) {
            self.calls.push(message)
        }
    }
    #[test]
    fn defaults() {
        let p = SqueezeVolPanel::get_instance(AxisID::Only, DialogType::PostProcessing);
        assert!(p.rb_input_file_trim_vol.is_selected());
        assert!(p.rb_filtering_none.is_selected());
        assert!(!p.ltf_factor_in_z.is_enabled());
        assert_eq!(p.pnl_root.listener_count, 17)
    }
    #[test]
    fn output_name() {
        let mut p = SqueezeVolPanel::get_instance(AxisID::Only, DialogType::PostProcessing);
        let m = Manager::default();
        p.cb_reduce_by_overall_factor.set_selected(true);
        p.tf_reduce_by_overall_factor.set_text("2.5");
        p.update_display(false);
        assert_eq!(p.get_output_filename(true, &m).unwrap(), "reduce2.5.rec")
    }
    #[test]
    fn actions() {
        let mut p = SqueezeVolPanel::get_instance(AxisID::Only, DialogType::PostProcessing);
        let mut m = Manager::default();
        p.action("Reduce/Filter Volume", None, &mut m);
        p.action(BTN_IMOD_REDUCE_FILT_VOL_LABEL, None, &mut m);
        assert_eq!(m.calls, vec!["reduce", "reduce1.rec"])
    }
}
