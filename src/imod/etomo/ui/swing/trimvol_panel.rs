//! `IMOD/Etomo/src/etomo/ui/swing/TrimvolPanel.java`.
//!
//! Swing widgets and the `ApplicationManager`, `TrimvolParam`, `MetaData`,
//! `VolumeRangePanel`, and `RubberbandPanel` are direct boundaries.  The
//! source-owned policy, state changes, validation order, and call arguments
//! are retained here.
#![allow(dead_code)]

use super::check_box::CheckBox;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::{BaseScreenState, MultiLineButton};
use super::radio_button::{RadioButton, RadioButtonGroup};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::{axis_id::AxisID, dialog_type::DialogType};
use crate::imod::etomo::ui::field_type::FieldType;
use std::{cell::RefCell, rc::Rc};

pub const SCALING_ERROR_TITLE: &str = "Scaling Panel Error";
pub const FIXED_SCALE_MIN_LABEL: &str = "black: ";
pub const FIXED_SCALE_MAX_LABEL: &str = " white: ";
pub const SECTION_SCALE_MIN_LABEL: &str = "Z min: ";
pub const SECTION_SCALE_MAX_LABEL: &str = " Z max: ";
pub const SWAP_YZ_LABEL: &str = "Swap Y and Z dimensions";
pub const REORIENTATION_GROUP_LABEL: &str = "Reorientation:";
pub const COMBINED_TOMOGRAM_KEY: &str = "combined tomogram";

/// Fields of Java `TrimvolParam` used by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrimvolParamBoundary {
    pub x_min: String,
    pub x_max: String,
    pub y_min: String,
    pub y_max: String,
    pub z_min: String,
    pub z_max: String,
    pub scale_x_min: String,
    pub scale_x_max: String,
    pub scale_y_min: String,
    pub scale_y_max: String,
    pub swap_yz: bool,
    pub rotate_x: bool,
    pub convert_to_bytes: bool,
    pub fixed_scaling: bool,
    pub flipped_volume: bool,
    pub fixed_scale_min: String,
    pub fixed_scale_max: String,
    pub section_scale_min: String,
    pub section_scale_max: String,
    pub format_of_output_file: Option<String>,
}

/// Fields of Java `ConstMetaData`/`MetaData` reached by this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrimvolMetaDataBoundary {
    pub x_min: String,
    pub x_max: String,
    pub y_min: String,
    pub y_max: String,
    pub z_min: String,
    pub z_max: String,
    pub scale_x_min: String,
    pub scale_x_max: String,
    pub scale_y_min: String,
    pub scale_y_max: String,
    pub post_trimvol_swap_yz: bool,
    pub post_trimvol_rotate_x: bool,
    pub post_trimvol_convert_to_bytes: bool,
    pub post_trimvol_fixed_scaling: bool,
    pub post_trimvol_fixed_scale_min: String,
    pub post_trimvol_fixed_scale_max: String,
    pub post_trimvol_section_scale_min: String,
    pub post_trimvol_section_scale_max: String,
    pub post_trimvol_scaling_new_style_z: Option<(String, String)>,
    pub image_output_format: Option<String>,
}

/// Java `TrimvolInputFileState` queries in `setStartupWarnings`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrimvolInputFileState {
    pub n_columns_changed: bool,
    pub n_rows_changed: bool,
    pub n_sections_changed: bool,
}

/// The state owned by direct Java `VolumeRangePanel` dependency.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VolumeRangePanelBoundary {
    pub x_min: String,
    pub x_max: String,
    pub y_min: String,
    pub y_max: String,
    pub z_min: String,
    pub z_max: String,
    pub validation_result: bool,
    pub trimvol_input_file_missing: bool,
    pub last_rubberband_coordinates: Option<Vec<String>>,
}

/// The state owned by direct Java `RubberbandPanel` dependency.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RubberbandPanelBoundary {
    pub x_min: String,
    pub x_max: String,
    pub y_min: String,
    pub y_max: String,
    pub enabled: bool,
    pub validation_result: bool,
    pub trimvol_input_file_missing: bool,
}

/// Java `ReconScreenState` at `setParameters(ReconScreenState)`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReconScreenStateBoundary {
    pub screen_state: BaseScreenState,
}

/// Java `ContextPopup` construction retained at the presentation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrimvolContextPopup {
    pub title: &'static str,
    pub guide: &'static str,
    pub man_page_label: [String; 1],
    pub man_page: [String; 1],
    pub axis_id: AxisID,
}

/// The four `ApplicationManager` calls in `TrimvolPanel.java`.
pub trait TrimvolPanelApplicationManager {
    fn trim_volume(
        &mut self,
        deferred_3dmod_button_present: bool,
        options: Run3dmodMenuOptions,
        dialog_type: DialogType,
    );
    fn imod_get_rubberband_coordinates(&mut self, key: &str, axis_id: AxisID) -> Vec<String>;
    fn imod_combined_tomogram(&mut self, options: Run3dmodMenuOptions);
    fn imod_trimmed_volume(&mut self, options: Run3dmodMenuOptions, axis_id: AxisID);
}

/// Java `TrimvolPanel` source-visible state.
#[derive(Clone, Debug)]
pub struct TrimvolPanel {
    pub pnl_trimvol_title: &'static str,
    pub pnl_scale_title: &'static str,
    pub cb_convert_to_bytes: CheckBox,
    pub rb_scale_fixed: RadioButton,
    pub ltf_fixed_scale_min: LabeledTextField,
    pub ltf_fixed_scale_max: LabeledTextField,
    pub rb_scale_section: RadioButton,
    pub ltf_section_scale_min: LabeledTextField,
    pub ltf_section_scale_max: LabeledTextField,
    pub rb_none: RadioButton,
    pub rb_swap_yz: RadioButton,
    pub rb_rotate_x: RadioButton,
    pub warning: String,
    pub warning_visible: bool,
    pub btn_imod_full: MultiLineButton,
    pub btn_trimvol: MultiLineButton,
    pub btn_imod_trim: MultiLineButton,
    pub btn_get_coordinates: MultiLineButton,
    pub volume_range_panel: VolumeRangePanelBoundary,
    pub pnl_scale_rubberband: RubberbandPanelBoundary,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub trimvol_input_file_missing: bool,
    pub button_action_listener_installed: bool,
    pub scaling_listener_installed: bool,
    pub context_popup: Option<TrimvolContextPopup>,
    pub last_scaling_error: Option<String>,
}

impl TrimvolPanel {
    /// Java `TrimvolPanel(ApplicationManager, AxisID, DialogType, boolean)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType, trimvol_input_file_missing: bool) -> Self {
        let scale_group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let orientation_group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut btn_trimvol = MultiLineButton::new_toggle(Some("Trim Volume"), true);
        btn_trimvol.set_enabled(!trimvol_input_file_missing);
        let mut result = Self {
            pnl_trimvol_title: "Volume Trimming",
            pnl_scale_title: "Scaling",
            cb_convert_to_bytes: CheckBox::new_with_text("Convert to bytes"),
            rb_scale_fixed: RadioButton::new_in_group(
                "Scale to match contrast  ",
                scale_group.clone(),
            ),
            ltf_fixed_scale_min: LabeledTextField::new(FieldType::Integer, FIXED_SCALE_MIN_LABEL),
            ltf_fixed_scale_max: LabeledTextField::new(FieldType::Integer, FIXED_SCALE_MAX_LABEL),
            rb_scale_section: RadioButton::new_in_group(
                "Find scaling from sections  ",
                scale_group,
            ),
            ltf_section_scale_min: LabeledTextField::new(
                FieldType::Integer,
                SECTION_SCALE_MIN_LABEL,
            ),
            ltf_section_scale_max: LabeledTextField::new(
                FieldType::Integer,
                SECTION_SCALE_MAX_LABEL,
            ),
            rb_none: RadioButton::new_in_group("None", orientation_group.clone()),
            rb_swap_yz: RadioButton::new_in_group(SWAP_YZ_LABEL, orientation_group.clone()),
            rb_rotate_x: RadioButton::new_in_group("Rotate around X axis", orientation_group),
            warning: String::new(),
            warning_visible: true,
            btn_imod_full: MultiLineButton::new_with_label(Some("3dmod Full Volume")),
            btn_trimvol,
            btn_imod_trim: MultiLineButton::new_with_label(Some("3dmod Trimmed Volume")),
            btn_get_coordinates: MultiLineButton::new_with_label(Some(
                "Get XYZ Volume Range From 3dmod",
            )),
            volume_range_panel: VolumeRangePanelBoundary {
                validation_result: true,
                trimvol_input_file_missing,
                ..Default::default()
            },
            pnl_scale_rubberband: RubberbandPanelBoundary {
                validation_result: true,
                trimvol_input_file_missing,
                ..Default::default()
            },
            axis_id,
            dialog_type,
            trimvol_input_file_missing,
            button_action_listener_installed: true,
            scaling_listener_installed: true,
            context_popup: None,
            last_scaling_error: None,
        };
        // Java `AbstractButton.getActionCommand()` defaults to its text.  The
        // generic native button boundary does not manufacture that Swing
        // default, so retain it at this source unit's listener boundary.
        result
            .btn_imod_full
            .set_action_command(Some("3dmod Full Volume"));
        result.btn_trimvol.set_action_command(Some("Trim Volume"));
        result
            .btn_imod_trim
            .set_action_command(Some("3dmod Trimmed Volume"));
        result
            .btn_get_coordinates
            .set_action_command(Some("Get XYZ Volume Range From 3dmod"));
        result.set_tool_tip_text();
        result.set_scale_state();
        result
    }
    /// Java `getContainer()`.
    pub fn get_container(&self) -> &Self {
        self
    }
    /// Java `initParameters(TrimvolParam)`.
    pub fn init_parameters(&mut self, param: &TrimvolParamBoundary) {
        if self.trimvol_input_file_missing {
            return;
        }
        if param.swap_yz {
            self.rb_swap_yz.set_selected(true)
        } else if param.rotate_x {
            self.rb_rotate_x.set_selected(true)
        } else {
            self.rb_none.set_selected(true)
        }
        if param.fixed_scaling {
            self.rb_scale_fixed.set_selected(true)
        } else {
            self.ltf_section_scale_min
                .set_text(&param.section_scale_min);
            self.ltf_section_scale_max
                .set_text(&param.section_scale_max);
            self.rb_scale_section.set_selected(true)
        }
        self.volume_range_panel.x_min.clone_from(&param.x_min);
        self.volume_range_panel.x_max.clone_from(&param.x_max);
        self.volume_range_panel.y_min.clone_from(&param.y_min);
        self.volume_range_panel.y_max.clone_from(&param.y_max);
        self.volume_range_panel.z_min.clone_from(&param.z_min);
        self.volume_range_panel.z_max.clone_from(&param.z_max);
        self.pnl_scale_rubberband
            .x_min
            .clone_from(&param.scale_x_min);
        self.pnl_scale_rubberband
            .x_max
            .clone_from(&param.scale_x_max);
        self.pnl_scale_rubberband
            .y_min
            .clone_from(&param.scale_y_min);
        self.pnl_scale_rubberband
            .y_max
            .clone_from(&param.scale_y_max);
        self.set_scale_state();
    }
    /// Java `setParameters(ConstMetaData, boolean)`.
    pub fn set_parameters(&mut self, meta: &TrimvolMetaDataBoundary, dialog_exists: bool) {
        if !dialog_exists || self.trimvol_input_file_missing {
            return;
        }
        self.volume_range_panel.x_min.clone_from(&meta.x_min);
        self.volume_range_panel.x_max.clone_from(&meta.x_max);
        self.volume_range_panel.y_min.clone_from(&meta.y_min);
        self.volume_range_panel.y_max.clone_from(&meta.y_max);
        self.volume_range_panel.z_min.clone_from(&meta.z_min);
        self.volume_range_panel.z_max.clone_from(&meta.z_max);
        if meta.post_trimvol_swap_yz {
            self.rb_swap_yz.set_selected(true)
        } else if meta.post_trimvol_rotate_x {
            self.rb_rotate_x.set_selected(true)
        } else {
            self.rb_none.set_selected(true)
        }
        self.cb_convert_to_bytes
            .set_selected(meta.post_trimvol_convert_to_bytes);
        if meta.post_trimvol_fixed_scaling {
            self.ltf_fixed_scale_min
                .set_text(&meta.post_trimvol_fixed_scale_min);
            self.ltf_fixed_scale_max
                .set_text(&meta.post_trimvol_fixed_scale_max);
            self.rb_scale_fixed.set_selected(true)
        } else {
            self.ltf_section_scale_min
                .set_text(&meta.post_trimvol_section_scale_min);
            self.ltf_section_scale_max
                .set_text(&meta.post_trimvol_section_scale_max);
            self.rb_scale_section.set_selected(true)
        }
        self.set_scale_state();
        self.pnl_scale_rubberband
            .x_min
            .clone_from(&meta.scale_x_min);
        self.pnl_scale_rubberband
            .x_max
            .clone_from(&meta.scale_x_max);
        self.pnl_scale_rubberband
            .y_min
            .clone_from(&meta.scale_y_min);
        self.pnl_scale_rubberband
            .y_max
            .clone_from(&meta.scale_y_max);
    }
    /// Java `setStartupWarnings(TrimvolInputFileState)`.
    pub fn set_startup_warnings(&mut self, state: TrimvolInputFileState) -> bool {
        if state.n_columns_changed || state.n_rows_changed {
            self.warning = if state.n_sections_changed {
                "Min and max values have been restored to defaults"
            } else {
                "X,Y values have been restored to defaults"
            }
            .into();
            return true;
        }
        if state.n_sections_changed {
            self.warning = "Z values have been restored to defaults".into();
            return true;
        }
        self.warning_visible = false;
        false
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_parameters(&self, meta: &mut TrimvolMetaDataBoundary) {
        if self.trimvol_input_file_missing {
            return;
        }
        meta.x_min.clone_from(&self.volume_range_panel.x_min);
        meta.x_max.clone_from(&self.volume_range_panel.x_max);
        meta.y_min.clone_from(&self.volume_range_panel.y_min);
        meta.y_max.clone_from(&self.volume_range_panel.y_max);
        meta.z_min.clone_from(&self.volume_range_panel.z_min);
        meta.z_max.clone_from(&self.volume_range_panel.z_max);
        meta.post_trimvol_swap_yz = self.rb_swap_yz.is_selected();
        meta.post_trimvol_rotate_x = self.rb_rotate_x.is_selected();
        meta.post_trimvol_convert_to_bytes = self.cb_convert_to_bytes.is_selected();
        meta.post_trimvol_fixed_scaling = self.rb_scale_fixed.is_selected();
        meta.post_trimvol_fixed_scale_min = self.ltf_fixed_scale_min.get_text();
        meta.post_trimvol_fixed_scale_max = self.ltf_fixed_scale_max.get_text();
        meta.post_trimvol_section_scale_min = self.ltf_section_scale_min.get_text();
        meta.post_trimvol_section_scale_max = self.ltf_section_scale_max.get_text();
        meta.scale_x_min
            .clone_from(&self.pnl_scale_rubberband.x_min);
        meta.scale_x_max
            .clone_from(&self.pnl_scale_rubberband.x_max);
        meta.scale_y_min
            .clone_from(&self.pnl_scale_rubberband.y_min);
        meta.scale_y_max
            .clone_from(&self.pnl_scale_rubberband.y_max);
    }
    /// Java `getParametersForTrimvol(MetaData)`.
    pub fn get_parameters_for_trimvol(&self, meta: &mut TrimvolMetaDataBoundary) {
        if self.trimvol_input_file_missing {
            return;
        }
        meta.x_min.clone_from(&self.volume_range_panel.x_min);
        meta.x_max.clone_from(&self.volume_range_panel.x_max);
        meta.y_min.clone_from(&self.volume_range_panel.y_min);
        meta.y_max.clone_from(&self.volume_range_panel.y_max);
        meta.z_min.clone_from(&self.volume_range_panel.z_min);
        meta.z_max.clone_from(&self.volume_range_panel.z_max);
        meta.post_trimvol_scaling_new_style_z = Some((
            self.ltf_section_scale_min.get_text(),
            self.ltf_section_scale_max.get_text(),
        ));
    }
    /// Java overloaded `getParameters(TrimvolParam, boolean)`.
    pub fn get_parameters_trimvol(
        &mut self,
        param: &mut TrimvolParamBoundary,
        do_validation: bool,
        output_format: Option<String>,
    ) -> bool {
        if self.trimvol_input_file_missing {
            return true;
        }
        if !self.volume_range_panel.validation_result {
            return false;
        }
        param.x_min.clone_from(&self.volume_range_panel.x_min);
        param.x_max.clone_from(&self.volume_range_panel.x_max);
        param.y_min.clone_from(&self.volume_range_panel.y_min);
        param.y_max.clone_from(&self.volume_range_panel.y_max);
        param.z_min.clone_from(&self.volume_range_panel.z_min);
        param.z_max.clone_from(&self.volume_range_panel.z_max);
        param.flipped_volume = true;
        param.swap_yz = self.rb_swap_yz.is_selected();
        param.rotate_x = self.rb_rotate_x.is_selected();
        param.format_of_output_file = output_format;
        param.convert_to_bytes = self.cb_convert_to_bytes.is_selected();
        param.fixed_scaling = self.rb_scale_fixed.is_selected();
        let (min, max, min_label, max_label) = if param.fixed_scaling {
            (
                &self.ltf_fixed_scale_min,
                &self.ltf_fixed_scale_max,
                FIXED_SCALE_MIN_LABEL,
                FIXED_SCALE_MAX_LABEL,
            )
        } else {
            (
                &self.ltf_section_scale_min,
                &self.ltf_section_scale_max,
                SECTION_SCALE_MIN_LABEL,
                SECTION_SCALE_MAX_LABEL,
            )
        };
        let min = match min.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let max = match max.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        for (value, label) in [(&min, min_label), (&max, max_label)] {
            if value.trim().parse::<i32>().is_err() {
                self.last_scaling_error = Some(format!("{label}must be an integer"));
                return false;
            }
        }
        if param.fixed_scaling {
            param.fixed_scale_min = min;
            param.fixed_scale_max = max
        } else {
            param.section_scale_min = min;
            param.section_scale_max = max
        }
        if !self.pnl_scale_rubberband.validation_result {
            return false;
        }
        param
            .scale_x_min
            .clone_from(&self.pnl_scale_rubberband.x_min);
        param
            .scale_x_max
            .clone_from(&self.pnl_scale_rubberband.x_max);
        param
            .scale_y_min
            .clone_from(&self.pnl_scale_rubberband.y_min);
        param
            .scale_y_max
            .clone_from(&self.pnl_scale_rubberband.y_max);
        true
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&mut self, state: &ReconScreenStateBoundary) {
        if !self.trimvol_input_file_missing {
            let key = self.btn_trimvol.get_button_state_key();
            self.btn_trimvol
                .set_button_state(state.screen_state.get_button_state(key.as_deref()));
        }
    }
    /// Java `setSwapYZ(boolean)`.
    pub fn set_swap_yz(&mut self, input: bool) {
        if input {
            self.rb_swap_yz.set_selected(true)
        }
    }
    /// Java `setRotateX(boolean)`.
    pub fn set_rotate_x(&mut self, input: bool) {
        if input {
            self.rb_rotate_x.set_selected(true)
        }
    }
    /// Java `setConvertToBytes(boolean)`.
    pub fn set_convert_to_bytes(&mut self, input: bool) {
        self.cb_convert_to_bytes.set_selected(input)
    }
    /// Java `setSectionScaleMin(String)`.
    pub fn set_section_scale_min(&mut self, input: &str) {
        self.ltf_section_scale_min.set_text(input)
    }
    /// Java `setSectionScaleMax(String)`.
    pub fn set_section_scale_max(&mut self, input: &str) {
        self.ltf_section_scale_max.set_text(input)
    }
    /// Java `setXMin(String)`.
    pub fn set_x_min(&mut self, input: &str) {
        self.volume_range_panel.x_min = input.into()
    }
    /// Java `setXMax(String)`.
    pub fn set_x_max(&mut self, input: &str) {
        self.volume_range_panel.x_max = input.into()
    }
    /// Java `setYMin(String)`.
    pub fn set_y_min(&mut self, input: &str) {
        self.volume_range_panel.y_min = input.into()
    }
    /// Java `setYMax(String)`.
    pub fn set_y_max(&mut self, input: &str) {
        self.volume_range_panel.y_max = input.into()
    }
    /// Java `setZMin(String)`.
    pub fn set_z_min(&mut self, input: &str) {
        self.volume_range_panel.z_min = input.into()
    }
    /// Java `setZMax(String)`.
    pub fn set_z_max(&mut self, input: &str) {
        self.volume_range_panel.z_max = input.into()
    }
    /// Java `setScaleXMin(String)`.
    pub fn set_scale_x_min(&mut self, input: &str) {
        self.pnl_scale_rubberband.x_min = input.into()
    }
    /// Java `setScaleXMax(String)`.
    pub fn set_scale_x_max(&mut self, input: &str) {
        self.pnl_scale_rubberband.x_max = input.into()
    }
    /// Java `setScaleYMin(String)`.
    pub fn set_scale_y_min(&mut self, input: &str) {
        self.pnl_scale_rubberband.y_min = input.into()
    }
    /// Java `setScaleYMax(String)`.
    pub fn set_scale_y_max(&mut self, input: &str) {
        self.pnl_scale_rubberband.y_max = input.into()
    }
    /// Java `setRubberbandContainerZMin(String)`.
    pub fn set_rubberband_container_z_min(&mut self, input: &str) {
        if self.rb_scale_section.is_selected() {
            self.ltf_section_scale_min.set_text(input)
        }
    }
    /// Java `setRubberbandContainerZMax(String)`.
    pub fn set_rubberband_container_z_max(&mut self, input: &str) {
        if self.rb_scale_section.is_selected() {
            self.ltf_section_scale_max.set_text(input)
        }
    }
    /// Java private `setScaleState()`.
    fn set_scale_state(&mut self) {
        let convert = self.cb_convert_to_bytes.is_selected();
        self.rb_scale_fixed.set_enabled(convert);
        self.rb_scale_section.set_enabled(convert);
        let fixed = convert && self.rb_scale_fixed.is_selected();
        self.ltf_fixed_scale_min.set_enabled(fixed);
        self.ltf_fixed_scale_max.set_enabled(fixed);
        let section = convert && self.rb_scale_section.is_selected();
        self.ltf_section_scale_min.set_enabled(section);
        self.ltf_section_scale_max.set_enabled(section);
        self.pnl_scale_rubberband.enabled = section;
    }
    /// Java `scaleAction(ActionEvent)`.
    pub fn scale_action(&mut self) {
        self.set_scale_state()
    }
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: TrimvolPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        deferred: bool,
        options: Run3dmodMenuOptions,
    ) {
        if self.btn_trimvol.get_action_command() == Some(command) {
            manager.trim_volume(deferred, options, self.dialog_type)
        } else if self.btn_get_coordinates.get_action_command() == Some(command) {
            let values =
                manager.imod_get_rubberband_coordinates(COMBINED_TOMOGRAM_KEY, AxisID::Only);
            self.volume_range_panel.last_rubberband_coordinates = Some(values.clone());
            if values.len() >= 4 {
                self.set_x_min(&values[0]);
                self.set_x_max(&values[1]);
                self.set_y_min(&values[2]);
                self.set_y_max(&values[3]);
            }
        } else if self.btn_imod_full.get_action_command() == Some(command) {
            manager.imod_combined_tomogram(options)
        } else if self.btn_imod_trim.get_action_command() == Some(command) {
            manager.imod_trimmed_volume(options, self.axis_id)
        }
    }
    /// Java `done()`.
    pub fn done(&mut self) {
        self.btn_trimvol.remove_action_listener();
        self.button_action_listener_installed = false
    }
    /// Java private `cbConvertToBytesAction(ActionEvent)`.
    pub fn cb_convert_to_bytes_action(&mut self) {
        let state = self.cb_convert_to_bytes.is_selected();
        self.rb_scale_fixed.set_enabled(state);
        self.ltf_fixed_scale_max.set_enabled(state);
        self.ltf_fixed_scale_min.set_enabled(state);
        self.rb_scale_section.set_enabled(state);
        self.ltf_section_scale_min.set_enabled(state);
        self.ltf_section_scale_max.set_enabled(state);
    }
    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&mut self) {
        self.cb_convert_to_bytes.set_tool_tip_text(Some(
            "Scale densities to bytes with extreme densities truncated.",
        ));
        self.rb_scale_fixed.set_tool_tip_text(Some(
            "Set the scaling to match the contrast in a 3dmod display.",
        ));
        self.ltf_fixed_scale_min.set_tool_tip_text(Some(
            "Enter the black contrast slider setting (0-254) that gives the desired contrast.",
        ));
        self.ltf_fixed_scale_max.set_tool_tip_text(Some(
            "Enter the white contrast slider setting (1-255) that gives the desired contrast.",
        ));
        self.rb_scale_section.set_tool_tip_text(Some("Set the scaling based on the range of contrast in a subset of sections and XY volume.  Exclude areas with extreme densities that can be truncated (gold particles)."));
        self.ltf_section_scale_min.set_tool_tip_text(Some(
            "Minimum Z section of the subset to analyze for contrast range.",
        ));
        self.ltf_section_scale_max.set_tool_tip_text(Some(
            "Maximum Z section of the subset to analyze for contrast range.",
        ));
        self.rb_none.set_tool_tip_text(Some("Do not change the orientation of the output volume.  The file will need to be flipped when loaded into 3dmod."));
        self.rb_swap_yz.set_tool_tip_text(Some("Flip Y and Z in the output volume so that the file does not need to be flipped when loaded into 3dmod."));
        self.btn_imod_full
            .set_tool_tip_text(Some("View the original, untrimmed volume in 3dmod."));
        self.btn_get_coordinates.set_tool_tip_text(Some("After pressing the 3dmod Full Volume button, press shift-B in the ZaP window.  Create a rubberband around the volume range.  Then press this button to retrieve X and Y coordinates."));
        self.btn_trimvol.set_tool_tip_text(Some(
            "Trim the original volume with the parameters given above.",
        ));
        self.btn_imod_trim
            .set_tool_tip_text(Some("View the trimmed volume."));
    }
    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self) {
        self.context_popup = Some(TrimvolContextPopup {
            title: "POST-PROCESSING",
            guide: "TOMO_GUIDE",
            man_page_label: ["Trimvol".into()],
            man_page: ["trimvol.html".into()],
            axis_id: self.axis_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        calls: Vec<String>,
    }
    impl TrimvolPanelApplicationManager for Manager {
        fn trim_volume(&mut self, deferred: bool, _: Run3dmodMenuOptions, dialog: DialogType) {
            self.calls.push(format!("trim:{deferred}:{dialog:?}"))
        }
        fn imod_get_rubberband_coordinates(&mut self, _: &str, _: AxisID) -> Vec<String> {
            self.calls.push("rubberband".into());
            vec!["1".into(), "20".into(), "3".into(), "40".into()]
        }
        fn imod_combined_tomogram(&mut self, _: Run3dmodMenuOptions) {
            self.calls.push("full".into())
        }
        fn imod_trimmed_volume(&mut self, _: Run3dmodMenuOptions, axis: AxisID) {
            self.calls.push(format!("trimmed:{axis:?}"))
        }
    }
    #[test]
    fn scaling_state_and_validation_follow_source() {
        let mut panel = TrimvolPanel::new(AxisID::Only, DialogType::PostProcessing, false);
        panel.cb_convert_to_bytes.set_selected(true);
        panel.rb_scale_fixed.set_selected(true);
        panel.scale_action();
        assert!(panel.ltf_fixed_scale_min.is_enabled());
        assert!(!panel.ltf_section_scale_min.is_enabled());
        panel.ltf_fixed_scale_min.set_text("12");
        panel.ltf_fixed_scale_max.set_text("254");
        let mut param = TrimvolParamBoundary::default();
        assert!(panel.get_parameters_trimvol(&mut param, true, Some("MRC".into())));
        assert_eq!(
            (
                param.fixed_scale_min.as_str(),
                param.fixed_scale_max.as_str()
            ),
            ("12", "254")
        );
        assert!(param.flipped_volume);
    }
    #[test]
    fn action_and_rubberband_routes_keep_source_arguments() {
        let mut panel = TrimvolPanel::new(AxisID::Second, DialogType::PostProcessing, false);
        let mut manager = Manager::default();
        let command = panel
            .btn_get_coordinates
            .get_action_command()
            .unwrap()
            .to_owned();
        panel.action(
            &mut manager,
            &command,
            false,
            Run3dmodMenuOptions::default(),
        );
        let command = panel.btn_imod_trim.get_action_command().unwrap().to_owned();
        panel.action(
            &mut manager,
            &command,
            false,
            Run3dmodMenuOptions::default(),
        );
        assert_eq!(panel.volume_range_panel.x_max, "20");
        assert_eq!(manager.calls, ["rubberband", "trimmed:Second"]);
    }
    #[test]
    fn metadata_and_input_warnings_match_source_branches() {
        let mut panel = TrimvolPanel::new(AxisID::Only, DialogType::PostProcessing, false);
        assert!(panel.set_startup_warnings(TrimvolInputFileState {
            n_columns_changed: true,
            n_rows_changed: false,
            n_sections_changed: true
        }));
        assert_eq!(
            panel.warning,
            "Min and max values have been restored to defaults"
        );
        assert!(!panel.set_startup_warnings(TrimvolInputFileState::default()));
        assert!(!panel.warning_visible);
        panel.set_x_min("2");
        panel.set_section_scale_min("5");
        let mut data = TrimvolMetaDataBoundary::default();
        panel.get_parameters(&mut data);
        assert_eq!(
            (
                data.x_min.as_str(),
                data.post_trimvol_section_scale_min.as_str()
            ),
            ("2", "5")
        );
    }
}
