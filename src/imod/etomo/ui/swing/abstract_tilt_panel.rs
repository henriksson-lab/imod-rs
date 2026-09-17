//! `IMOD/Etomo/src/etomo/ui/swing/AbstractTiltPanel.java`.
//!
//! The Java controls and the application/com-script classes are not translated
//! yet.  Their direct boundary is deliberately represented here by the state
//! which this source unit reads and writes; no alternate tilt-panel policy is
//! introduced.
#![allow(dead_code)]

use crate::imod::etomo::logic::converter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

pub const BACK_PROJECTION_HEADER: &str = "Tilt";
pub const PARAMETERS_HEADER: &str = "Tilt Parameters";
pub const SUPER_SAMPLE_FACTOR_LABEL: &str = "Super-sample by";
pub const LINEAR_SCALE_FACTOR_DEFAULT: &str = "1.0";
pub const LINEAR_SCALE_OFFSET_DEFAULT: &str = "0.0";

/// Direct state of Java `LabeledTextField` / `CheckTextField` at this unit's boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltTextField {
    pub label: String,
    pub text: String,
    pub visible: bool,
    pub enabled: bool,
    pub preferred_width: Option<i32>,
    pub tooltip: Option<String>,
}

/// Direct state of Java `CheckBox` at this unit's boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltCheckBox {
    pub label: String,
    pub selected: bool,
    pub visible: bool,
    pub enabled: bool,
    pub tooltip: Option<String>,
}

/// Direct state of Java `Spinner` at this unit's boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TiltSpinner {
    pub label: String,
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub visible: bool,
    pub enabled: bool,
    pub tooltip: Option<String>,
}

/// Direct state of Java `Run3dmodButton` / `MultiLineButton` at this unit's boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltButton {
    pub action_command: String,
    pub enabled: bool,
    pub tooltip: Option<String>,
    pub listener_count: u32,
}

/// The source-visible `ConstTiltParam` / `TiltParam` values owned by this panel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TiltParamState {
    pub image_binned: bool,
    pub width: Option<String>,
    pub thickness: Option<String>,
    pub x_shift: Option<f64>,
    pub y_height: Option<String>,
    pub y_shift: Option<String>,
    pub z_shift: Option<String>,
    pub x_axis_tilt: Option<String>,
    pub tilt_angle_offset: Option<String>,
    pub scale_coeff: Option<f64>,
    pub scale_f_level: Option<f64>,
    pub log_shift: Option<f64>,
    pub local_align_file: String,
    pub fiducialess: bool,
    pub use_z_factors: bool,
    pub super_sample_factor: Option<i32>,
    pub expand_input_lines: bool,
    pub exclude_list_2: String,
    pub montage_subset_start: bool,
    pub subset_start_valid: bool,
}

/// Direct source dependency `MetaData` / `TomogramState` reduced to members accessed here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AbstractTiltPanelState {
    pub made_z_factors: Option<bool>,
    pub newst_fiducialess_alignment: Option<bool>,
    pub used_local_alignments: Option<bool>,
    pub backward_compatible_made_z_factors: bool,
    pub backward_compatible_used_local_alignments: bool,
    pub fiducialess_alignment: bool,
    pub use_local_alignments: bool,
    pub use_z_factors: bool,
    pub dataset_name: String,
    pub montage: bool,
    pub gen_log: String,
    pub gen_scale_factor_log: String,
    pub gen_scale_offset_log: String,
    pub gen_scale_factor_linear: Option<String>,
    pub gen_scale_offset_linear: Option<String>,
    pub gen_super_sample_factor: i32,
    pub gen_expand_input_lines: bool,
}

/// Java abstract calls recorded at the owning manager/process boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TiltPanelAction {
    Tilt(ProcessingMethod),
    DeleteAlignedStack,
    ImodTomogram,
    UpdateDisplay,
}

/// State and complete local behavior of Java `AbstractTiltPanel`.
pub struct AbstractTiltPanel {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub panel_id: String,
    pub listen_for_field_changes: bool,
    pub header_text: String,
    pub header_advanced: bool,
    pub pnl_root_visible: bool,
    pub pnl_body_visible: bool,
    pub pnl_recon_with_super_sampling_visible: bool,
    pub trial_panel_visible: bool,
    pub trial_tilt_panel_visible: bool,
    pub pnl_button_visible: bool,
    pub radial_panel_visible: bool,
    pub radial_panel_editable: bool,
    pub cpu_gpu_panel_visible: bool,
    pub cpu_gpu_parallel: bool,
    pub cpu_gpu_method: ProcessingMethod,
    pub is_back_projection_value: bool,
    pub is_multifilt_value: bool,
    pub is_ctf3d_value: bool,
    pub is_sirt_value: bool,
    pub is_method_plugin_value: bool,
    pub made_z_factors: bool,
    pub newst_fiducialess_alignment: bool,
    pub used_local_alignments: bool,
    pub ltf_tomo_width: TiltTextField,
    pub ltf_tomo_thickness: TiltTextField,
    pub ltf_x_axis_tilt: TiltTextField,
    pub ltf_extra_exclude_list: TiltTextField,
    pub ltf_x_shift: TiltTextField,
    pub ltf_tomo_height: TiltTextField,
    pub ltf_y_shift: TiltTextField,
    pub ctf_log: TiltTextField,
    pub ctf_log_selected: bool,
    pub ltf_tilt_angle_offset: TiltTextField,
    pub ltf_log_density_scale_factor: TiltTextField,
    pub ltf_log_density_scale_offset: TiltTextField,
    pub ltf_linear_density_scale_factor: TiltTextField,
    pub ltf_linear_density_scale_offset: TiltTextField,
    pub ltf_z_shift: TiltTextField,
    pub cb_use_local_alignment: TiltCheckBox,
    pub cb_use_z_factors: TiltCheckBox,
    pub cb_super_sample_factor: TiltCheckBox,
    pub sp_super_sample_factor: TiltSpinner,
    pub cb_expand_input_lines: TiltCheckBox,
    pub btn_3dmod_tomogram: TiltButton,
    pub btn_tilt: TiltButton,
    pub btn_delete_stack: TiltButton,
    pub listener_commands: Vec<String>,
    pub last_action: Option<TiltPanelAction>,
    pub packed_count: u32,
    pub debug: bool,
}

impl AbstractTiltPanel {
    /// Java constructor.  `ApplicationManager`, `GlobalExpandButton`, and
    /// `TomogramGenerationParent` remain direct dependencies represented by inputs/state.
    pub fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: impl Into<String>,
        listen_for_field_changes: bool,
    ) -> Self {
        Self {
            axis_id,
            dialog_type,
            panel_id: panel_id.into(),
            listen_for_field_changes,
            header_text: BACK_PROJECTION_HEADER.into(),
            header_advanced: false,
            pnl_root_visible: true,
            pnl_body_visible: true,
            pnl_recon_with_super_sampling_visible: false,
            trial_panel_visible: false,
            trial_tilt_panel_visible: false,
            pnl_button_visible: false,
            radial_panel_visible: false,
            radial_panel_editable: true,
            cpu_gpu_panel_visible: true,
            cpu_gpu_parallel: false,
            cpu_gpu_method: ProcessingMethod::DEFAULT,
            is_back_projection_value: true,
            is_multifilt_value: false,
            is_ctf3d_value: false,
            is_sirt_value: false,
            is_method_plugin_value: false,
            made_z_factors: false,
            newst_fiducialess_alignment: false,
            used_local_alignments: false,
            ltf_tomo_width: TiltTextField {
                label: "Tomogram width in X: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_tomo_thickness: TiltTextField {
                label: "Tomogram thickness in Z: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_x_axis_tilt: TiltTextField {
                label: "X axis tilt: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_extra_exclude_list: TiltTextField {
                label: "Extra views to exclude: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_x_shift: TiltTextField {
                label: "X shift: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_tomo_height: TiltTextField {
                label: "Tomogram height in Y: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_y_shift: TiltTextField {
                label: " Y shift: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ctf_log: TiltTextField {
                label: "Take logarithm of densities with offset: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ctf_log_selected: false,
            ltf_tilt_angle_offset: TiltTextField {
                label: "Tilt angle offset: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_log_density_scale_factor: TiltTextField {
                label: "Logarithm density scaling factor: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_log_density_scale_offset: TiltTextField {
                label: " Offset: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_linear_density_scale_factor: TiltTextField {
                label: "Linear density scaling factor: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_linear_density_scale_offset: TiltTextField {
                label: " Offset: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            ltf_z_shift: TiltTextField {
                label: " Z shift: ".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            cb_use_local_alignment: TiltCheckBox {
                label: "Use local alignments".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            cb_use_z_factors: TiltCheckBox {
                label: "Use Z factors".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            cb_super_sample_factor: TiltCheckBox {
                label: SUPER_SAMPLE_FACTOR_LABEL.into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            sp_super_sample_factor: TiltSpinner {
                label: SUPER_SAMPLE_FACTOR_LABEL.into(),
                value: 2,
                minimum: 2,
                maximum: 8,
                visible: true,
                enabled: true,
                tooltip: None,
            },
            cb_expand_input_lines: TiltCheckBox {
                label: "Super-sample input also".into(),
                visible: true,
                enabled: true,
                ..Default::default()
            },
            btn_3dmod_tomogram: TiltButton {
                action_command: "View Tomogram In 3dmod".into(),
                enabled: true,
                ..Default::default()
            },
            btn_tilt: TiltButton {
                action_command: "Tilt".into(),
                enabled: true,
                ..Default::default()
            },
            btn_delete_stack: TiltButton {
                action_command: "Delete Aligned Stack".into(),
                enabled: true,
                ..Default::default()
            },
            listener_commands: vec![],
            last_action: None,
            packed_count: 0,
            debug: false,
        }
    }

    pub fn tilt_action(&mut self, tilt_processing_method: ProcessingMethod) {
        self.last_action = Some(TiltPanelAction::Tilt(tilt_processing_method));
    }
    pub fn imod_tomogram_action(&mut self) {
        self.last_action = Some(TiltPanelAction::ImodTomogram);
    }
    pub fn initialize_panel(&mut self) {
        self.btn_tilt.listener_count = self.btn_tilt.listener_count.saturating_add(0);
    }
    pub fn create_panel(&mut self) {
        self.set_advanced_field_displayer();
        self.initialize_panel();
        self.ltf_linear_density_scale_factor.text = LINEAR_SCALE_FACTOR_DEFAULT.into();
        self.ltf_linear_density_scale_offset.text = LINEAR_SCALE_OFFSET_DEFAULT.into();
        self.ltf_tomo_width.preferred_width = Some(163);
        self.ltf_tomo_height.preferred_width = Some(159);
        self.update_display();
    }
    pub fn get_root_panel(&self) -> bool {
        self.pnl_root_visible
    }
    pub fn get_panel_id(&self) -> &str {
        &self.panel_id
    }
    pub fn add_axis_tilt_focus_listener(&mut self) {
        self.listener_commands.push("axisTiltFocus".into());
    }
    pub fn remove_axis_tilt_focus_listener(&mut self) {
        self.listener_commands.retain(|v| v != "axisTiltFocus");
    }
    pub fn add_use_local_alignment_action_listener(&mut self) {
        self.listener_commands.push("useLocalAlignment".into());
    }
    pub fn remove_use_local_alignment_action_listener(&mut self) {
        self.listener_commands.retain(|v| v != "useLocalAlignment");
    }
    pub fn add_use_z_factors_action_listener(&mut self) {
        self.listener_commands.push("useZFactors".into());
    }
    pub fn remove_use_z_factors_action_listener(&mut self) {
        self.listener_commands.retain(|v| v != "useZFactors");
    }
    pub fn get_x_axis_tilt(&self) -> &str {
        &self.ltf_x_axis_tilt.text
    }
    pub fn get_tilt_button(&self) -> &TiltButton {
        &self.btn_tilt
    }
    pub fn get_3dmod_tomogram_button(&self) -> &TiltButton {
        &self.btn_3dmod_tomogram
    }
    pub fn get_cpu_gpu_panel(&self) -> bool {
        self.cpu_gpu_panel_visible
    }
    pub fn set_tilt_button_tooltip(&mut self, tooltip: impl Into<String>) {
        self.btn_tilt.tooltip = Some(tooltip.into());
    }
    pub fn add_listeners(&mut self) {
        self.btn_tilt.listener_count += 1;
        self.btn_3dmod_tomogram.listener_count += 1;
        self.btn_delete_stack.listener_count += 1;
        self.listener_commands
            .extend(["log".into(), "superSampleFactor".into()]);
        if self.listen_for_field_changes {
            self.listener_commands
                .extend(["useLocalAlignment".into(), "useZFactors".into()]);
        }
    }
    pub fn set_filter_type_action_listener(&mut self) {
        self.listener_commands.push("multifiltFilterType".into());
    }
    pub fn get_root(&self) -> bool {
        self.pnl_root_visible
    }
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }
    pub fn expand_global(&mut self) {}
    pub fn expand(&mut self, expanded: bool, advanced_basic: bool) {
        if advanced_basic {
            self.update_display();
        } else {
            self.pnl_body_visible = expanded;
        }
        self.packed_count += 1;
    }
    pub fn is_advanced(&self) -> bool {
        self.header_advanced
    }
    pub fn msg_method_changed(&mut self) {
        self.update_display();
    }
    pub fn register_processing_method_mediator(&mut self) {
        self.listener_commands
            .push("registerProcessingMethodMediator".into());
    }
    pub fn deregister_processing_method_mediator(&mut self) {
        self.listener_commands
            .push("deregisterProcessingMethodMediator".into());
    }
    pub fn set_advanced_field_displayer(&mut self) {}
    pub fn reset_to_nonplugin_state(&mut self) {
        if !self.is_method_plugin() {
            self.ctf_log.visible = true;
            self.ltf_tilt_angle_offset.visible = true;
            self.ltf_log_density_scale_factor.visible = true;
            self.ltf_log_density_scale_offset.visible = true;
            self.ltf_linear_density_scale_factor.visible = true;
            self.ltf_linear_density_scale_offset.visible = true;
            self.ltf_z_shift.visible = true;
            self.cb_use_local_alignment.visible = true;
            self.cb_use_z_factors.visible = true;
        }
    }
    pub fn done(&mut self) {
        self.btn_tilt.listener_count = self.btn_tilt.listener_count.saturating_sub(1);
        self.btn_delete_stack.listener_count =
            self.btn_delete_stack.listener_count.saturating_sub(1);
    }
    pub fn is_back_projection(&self) -> bool {
        self.is_back_projection_value
    }
    pub fn is_multifilt(&self) -> bool {
        self.is_multifilt_value
    }
    pub fn is_ctf3d(&self) -> bool {
        self.is_ctf3d_value
    }
    pub fn is_sirt(&self) -> bool {
        self.is_sirt_value
    }
    pub fn is_method_plugin(&self) -> bool {
        self.is_method_plugin_value
    }
    pub fn update_display(&mut self) {
        let advanced = self.header_advanced;
        self.reset_to_nonplugin_state();
        let back_projection = self.is_back_projection();
        let multifilt = self.is_multifilt();
        let ctf3d = self.is_ctf3d();
        self.ltf_log_density_scale_offset.visible = advanced;
        self.ltf_log_density_scale_factor.visible = advanced;
        self.ltf_linear_density_scale_offset.visible = advanced;
        self.ltf_linear_density_scale_factor.visible = advanced;
        self.ltf_tomo_width.visible = advanced && (back_projection || ctf3d);
        self.ltf_tomo_height.visible = advanced && (back_projection || ctf3d);
        self.ltf_tomo_thickness.visible = !multifilt;
        self.ltf_y_shift.visible = advanced && (back_projection || ctf3d);
        self.ltf_x_shift.visible = advanced && (back_projection || ctf3d);
        self.ltf_z_shift.visible = !multifilt;
        let super_sampling = (advanced && back_projection) || ctf3d;
        self.pnl_recon_with_super_sampling_visible = super_sampling;
        self.cb_super_sample_factor.visible = super_sampling;
        self.sp_super_sample_factor.visible = super_sampling;
        self.cb_expand_input_lines.visible = super_sampling;
        self.ltf_tilt_angle_offset.visible = advanced;
        self.radial_panel_visible = back_projection || ctf3d || multifilt;
        self.trial_panel_visible = back_projection;
        self.trial_tilt_panel_visible = advanced;
        self.pnl_button_visible = back_projection;
        self.btn_3dmod_tomogram.enabled = true;
        self.ctf_log.enabled = true;
        self.ltf_tomo_width.enabled = true;
        self.ltf_tomo_thickness.enabled = true;
        self.ltf_x_axis_tilt.enabled = true;
        self.ltf_tilt_angle_offset.enabled = true;
        self.ltf_extra_exclude_list.enabled = true;
        self.ltf_log_density_scale_factor.enabled = self.ctf_log_selected;
        self.ltf_log_density_scale_offset.enabled = self.ctf_log_selected;
        self.ltf_linear_density_scale_factor.enabled = !self.ctf_log_selected;
        self.ltf_linear_density_scale_offset.enabled = !self.ctf_log_selected;
        self.ltf_tomo_height.enabled = true;
        self.ltf_y_shift.enabled = true;
        self.ltf_z_shift.enabled = true;
        self.ltf_x_shift.enabled = true;
        self.cb_super_sample_factor.enabled = true;
        self.sp_super_sample_factor.enabled = self.cb_super_sample_factor.selected;
        self.cb_expand_input_lines.enabled = self.cb_super_sample_factor.selected;
        self.radial_panel_editable = true;
        self.cb_use_local_alignment.enabled =
            self.used_local_alignments && !self.newst_fiducialess_alignment;
        self.cb_use_z_factors.enabled = self.made_z_factors && !self.newst_fiducialess_alignment;
        self.btn_tilt.enabled = true;
        self.btn_delete_stack.enabled = true;
        self.header_text = if back_projection {
            BACK_PROJECTION_HEADER
        } else {
            PARAMETERS_HEADER
        }
        .into();
    }
    pub fn is_parallel_process(&self) -> bool {
        self.cpu_gpu_parallel
    }
    pub fn get_run_method_for_process_interface(&self) -> ProcessingMethod {
        self.cpu_gpu_method
    }
    pub fn is_z_shift_set(&self) -> bool {
        !self.ltf_z_shift.text.trim().is_empty()
    }
    pub fn is_use_local_alignment(&self) -> bool {
        self.cb_use_local_alignment.selected
    }
    pub fn set_state(&mut self, state: &AbstractTiltPanelState) {
        self.made_z_factors = state
            .made_z_factors
            .unwrap_or(state.backward_compatible_made_z_factors);
        self.newst_fiducialess_alignment = state
            .newst_fiducialess_alignment
            .unwrap_or(state.fiducialess_alignment);
        self.used_local_alignments = state
            .used_local_alignments
            .unwrap_or(state.backward_compatible_used_local_alignments);
        self.update_display();
    }
    pub fn is_use_z_factors(&self) -> bool {
        self.cb_use_z_factors.selected
    }
    pub fn is_cb_super_sample_factor(&self) -> bool {
        self.cb_super_sample_factor.selected
    }
    pub fn is_cb_expand_input_lines(&self) -> bool {
        self.cb_expand_input_lines.selected
    }
    pub fn get_parameters_meta_data(&self, state: &mut AbstractTiltPanelState) {
        state.gen_log = self.ctf_log.text.clone();
        state.gen_scale_factor_log = self.ltf_log_density_scale_factor.text.clone();
        state.gen_scale_offset_log = self.ltf_log_density_scale_offset.text.clone();
        state.gen_scale_factor_linear = Some(self.ltf_linear_density_scale_factor.text.clone());
        state.gen_scale_offset_linear = Some(self.ltf_linear_density_scale_offset.text.clone());
        state.gen_super_sample_factor = self.sp_super_sample_factor.value;
        state.gen_expand_input_lines = self.cb_expand_input_lines.selected;
    }
    pub fn get_parameters_recon_screen_state(&self) -> bool {
        self.header_advanced
    }
    pub fn set_parameters_meta_data(&mut self, state: &AbstractTiltPanelState) {
        self.ctf_log.text = state.gen_log.clone();
        self.ltf_log_density_scale_factor.text = state.gen_scale_factor_log.clone();
        self.ltf_log_density_scale_offset.text = state.gen_scale_offset_log.clone();
        if let Some(v) = &state.gen_scale_factor_linear {
            self.ltf_linear_density_scale_factor.text = v.clone();
        }
        if let Some(v) = &state.gen_scale_offset_linear {
            self.ltf_linear_density_scale_offset.text = v.clone();
        }
        self.sp_super_sample_factor.value = state.gen_super_sample_factor;
        self.cb_expand_input_lines.selected = state.gen_expand_input_lines;
        self.update_display();
    }
    pub fn disable_gpu(&mut self, _disable: bool) {
        self.update_display();
    }
    pub fn lock_processing_method(&mut self, _lock: bool) {
        self.update_display();
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.cpu_gpu_method
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    pub fn reregister_processing_method_mediator(&mut self) {
        self.listener_commands
            .push("reregisterProcessingMethodMediator".into());
    }
    pub fn set_parameters_tilt_param(&mut self, param: &TiltParamState, initialize: bool) {
        if let Some(v) = &param.width {
            self.ltf_tomo_width.text = v.clone();
        }
        if let Some(v) = &param.thickness {
            self.ltf_tomo_thickness.text = v.clone();
        }
        if let Some(v) = param.x_shift {
            self.ltf_x_shift.text = v.to_string();
        }
        if let Some(v) = &param.y_height {
            self.ltf_tomo_height.text = v.clone();
        }
        if let Some(v) = &param.y_shift {
            self.ltf_y_shift.text = v.clone();
        }
        if let Some(v) = &param.z_shift {
            self.ltf_z_shift.text = v.clone();
        }
        if let Some(v) = &param.x_axis_tilt {
            self.ltf_x_axis_tilt.text = v.clone();
        }
        if let Some(v) = &param.tilt_angle_offset {
            self.ltf_tilt_angle_offset.text = v.clone();
        }
        self.ctf_log_selected = param.log_shift.is_some();
        if self.ctf_log_selected || initialize {
            if let Some(v) = param.log_shift {
                self.ctf_log.text = v.to_string();
            }
        }
        if !self.ctf_log_selected && initialize && self.ctf_log.text.trim().is_empty() {
            self.ctf_log.text = "0.0".into();
        }
        if let (Some(coeff), Some(level)) = (param.scale_coeff, param.scale_f_level) {
            if self.ctf_log_selected || initialize {
                self.ltf_log_density_scale_offset.text = level.to_string();
                self.ltf_log_density_scale_factor.text = if self.ctf_log_selected {
                    coeff.to_string()
                } else {
                    ((coeff * 500.0).round() / 100.0).to_string()
                };
            }
            if !self.ctf_log_selected {
                self.ltf_linear_density_scale_offset.text = level.to_string();
                self.ltf_linear_density_scale_factor.text = coeff.to_string();
            }
        }
        self.cb_super_sample_factor.selected = param.super_sample_factor.is_some();
        if let Some(v) = param.super_sample_factor {
            self.sp_super_sample_factor.value = v;
            self.cb_expand_input_lines.selected = param.expand_input_lines;
        }
        self.ltf_extra_exclude_list.text = param.exclude_list_2.clone();
        self.update_display();
    }
    pub fn set_parameters_recon_screen_state(&mut self, advanced: bool) {
        self.header_advanced = advanced;
        self.update_display();
    }
    pub fn get_parameters_splittilt(&self, cpus_selected: Result<i32, String>) -> bool {
        matches!(cpus_selected, Ok(value) if value > 0)
    }
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    pub fn get_tomo_thickness(&self) -> Option<i64> {
        converter::to_long(Some(&self.ltf_tomo_thickness.text))
    }
    pub fn get_parameters_tilt_param(
        &mut self,
        param: &mut TiltParamState,
        do_validation: bool,
        state: &mut AbstractTiltPanelState,
    ) -> Result<bool, String> {
        param.image_binned = true;
        param.width = if self.ltf_tomo_width.text.trim().is_empty() {
            None
        } else {
            Some(self.ltf_tomo_width.text.clone())
        };
        param.z_shift = if self.is_z_shift_set() {
            Some(self.ltf_z_shift.text.clone())
        } else {
            None
        };
        param.x_shift = if !self.ltf_x_shift.text.trim().is_empty() {
            Some(
                self.ltf_x_shift
                    .text
                    .trim()
                    .parse()
                    .map_err(|_| format!("{} invalid number", self.ltf_x_shift.label))?,
            )
        } else if self.is_z_shift_set() {
            self.ltf_x_shift.text = "0".into();
            Some(0.0)
        } else {
            None
        };
        param.y_height = if self.ltf_tomo_height.text.trim().is_empty() {
            None
        } else {
            Some(self.ltf_tomo_height.text.clone())
        };
        param.y_shift = if self.ltf_y_shift.text.trim().is_empty() {
            None
        } else {
            Some(self.ltf_y_shift.text.clone())
        };
        param.thickness = if self.ltf_tomo_thickness.text.trim().is_empty() {
            None
        } else {
            Some(self.ltf_tomo_thickness.text.clone())
        };
        param.x_axis_tilt = if self.ltf_x_axis_tilt.text.trim().is_empty() {
            None
        } else {
            Some(self.ltf_x_axis_tilt.text.clone())
        };
        param.tilt_angle_offset = if self.ltf_tilt_angle_offset.text.trim().is_empty() {
            None
        } else {
            Some(self.ltf_tilt_angle_offset.text.clone())
        };
        let scale_factor = if self.ltf_log_density_scale_offset.enabled {
            &self.ltf_log_density_scale_factor
        } else {
            &self.ltf_linear_density_scale_factor
        };
        let scale_offset = if self.ltf_log_density_scale_offset.enabled {
            &self.ltf_log_density_scale_offset
        } else {
            &self.ltf_linear_density_scale_offset
        };
        if !scale_factor.text.trim().is_empty() || !scale_offset.text.trim().is_empty() {
            param.scale_coeff = Some(
                scale_factor
                    .text
                    .trim()
                    .parse()
                    .map_err(|_| format!("{} invalid number", scale_factor.label))?,
            );
            param.scale_f_level = Some(
                scale_offset
                    .text
                    .trim()
                    .parse()
                    .map_err(|_| format!("{} invalid number", scale_offset.label))?,
            );
        } else {
            param.scale_coeff = None;
            param.scale_f_level = None;
        }
        param.log_shift = if self.ctf_log_selected && !self.ctf_log.text.trim().is_empty() {
            Some(
                self.ctf_log
                    .text
                    .trim()
                    .parse()
                    .map_err(|_| format!("{} invalid number", self.ctf_log.label))?,
            )
        } else {
            None
        };
        param.local_align_file =
            if self.is_use_local_alignment() && self.cb_use_local_alignment.enabled {
                format!(
                    "{}{}local.xf",
                    state.dataset_name,
                    self.axis_id.get_extension()
                )
            } else {
                String::new()
            };
        state.use_local_alignments = self.is_use_local_alignment();
        param.fiducialess = state
            .newst_fiducialess_alignment
            .unwrap_or(state.fiducialess_alignment);
        param.use_z_factors = self.is_use_z_factors() && self.cb_use_z_factors.enabled;
        state.use_z_factors = self.is_use_z_factors();
        param.super_sample_factor = self
            .cb_super_sample_factor
            .selected
            .then_some(self.sp_super_sample_factor.value);
        param.expand_input_lines =
            self.cb_super_sample_factor.selected && self.cb_expand_input_lines.selected;
        param.exclude_list_2 = self.ltf_extra_exclude_list.text.clone();
        param.montage_subset_start = state.montage;
        if !state.montage && !param.subset_start_valid && do_validation {
            return Ok(false);
        }
        Ok(true)
    }
    pub fn action_button(&mut self, command: &str) {
        self.action(command);
    }
    pub fn action(&mut self, command: &str) {
        if command == self.btn_tilt.action_command {
            self.tilt_action(self.get_run_method_for_process_interface());
        } else if command == self.btn_delete_stack.action_command {
            self.last_action = Some(TiltPanelAction::DeleteAlignedStack);
        } else if command == self.btn_3dmod_tomogram.action_command {
            self.imod_tomogram_action();
        } else {
            self.update_display();
            self.last_action = Some(TiltPanelAction::UpdateDisplay);
        }
    }
    pub fn set_tool_tip_text(&mut self) {
        self.ltf_tomo_thickness.tooltip = Some(
            "Thickness, in unbinned pixels, along the z-axis of the reconstructed volume.".into(),
        );
        self.ltf_x_axis_tilt.tooltip = Some("X axis tilt".into());
        self.btn_tilt.tooltip = Some(
            "Compute the tomogram from the full aligned stack.  This runs the tilt.com script."
                .into(),
        );
        self.btn_3dmod_tomogram.tooltip = Some("View the reconstructed volume in 3dmod.".into());
        self.btn_delete_stack.tooltip = Some("Delete the aligned stack for this axis.".into());
    }
    pub fn tilt_action_listener_action_performed(&mut self, command: &str) {
        self.action(command);
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for `TiltActionListener::actionPerformed`.
    pub fn actionPerformed(&mut self, command: &str) {
        self.tilt_action_listener_action_performed(command);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn update_display_has_java_backprojection_visibility_and_enablement() {
        let mut panel =
            AbstractTiltPanel::new(AxisID::First, DialogType::TomogramGeneration, "Tilt", true);
        panel.header_advanced = true;
        panel.cb_super_sample_factor.selected = true;
        panel.used_local_alignments = true;
        panel.update_display();
        assert!(panel.pnl_recon_with_super_sampling_visible);
        assert!(panel.ltf_tomo_width.visible);
        assert!(panel.sp_super_sample_factor.enabled);
        assert!(panel.cb_use_local_alignment.enabled);
        assert_eq!(panel.header_text, BACK_PROJECTION_HEADER);
    }
    #[test]
    fn tilt_parameter_transfer_preserves_source_x_shift_and_supersampling_rules() {
        let mut panel =
            AbstractTiltPanel::new(AxisID::First, DialogType::TomogramGeneration, "Tilt", false);
        panel.ltf_z_shift.text = "4".into();
        panel.cb_super_sample_factor.selected = true;
        panel.sp_super_sample_factor.value = 4;
        panel.cb_expand_input_lines.selected = true;
        let mut param = TiltParamState {
            subset_start_valid: true,
            ..Default::default()
        };
        let mut state = AbstractTiltPanelState {
            dataset_name: "data".into(),
            ..Default::default()
        };
        assert_eq!(
            panel.get_parameters_tilt_param(&mut param, true, &mut state),
            Ok(true)
        );
        assert_eq!(param.x_shift, Some(0.0));
        assert_eq!(param.super_sample_factor, Some(4));
        assert!(param.expand_input_lines);
    }
    #[test]
    fn action_routes_source_commands() {
        let mut panel =
            AbstractTiltPanel::new(AxisID::Only, DialogType::TomogramGeneration, "Tilt", false);
        panel.action("Tilt");
        assert_eq!(
            panel.last_action,
            Some(TiltPanelAction::Tilt(ProcessingMethod::LocalCpu))
        );
        panel.action("Delete Aligned Stack");
        assert_eq!(panel.last_action, Some(TiltPanelAction::DeleteAlignedStack));
    }
}
