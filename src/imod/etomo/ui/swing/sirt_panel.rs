//! `IMOD/Etomo/src/etomo/ui/swing/SirtPanel.java`.
//!
//! Swing layout, file choosing, autodoc I/O, and `ApplicationManager` process
//! execution are explicit frontend boundaries.  The panel's source-owned
//! widget state, resume discovery, enablement, parameter transfer, and action
//! dispatch are retained here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonGroup};

pub const RESUME_FROM_LAST_ITERATION_LABEL: &str = "Resume from last iteration";

/// Java `SirtsetupParam` calls made by `SirtPanel`.
pub trait SirtsetupParam {
    fn set_leave_iterations(&mut self, value: String) -> Result<(), String>;
    fn set_subarea_size(&mut self, value: String) -> Result<(), String>;
    fn set_y_offset_of_subarea(&mut self, value: String) -> Result<(), String>;
    fn reset_subarea_size(&mut self);
    fn reset_y_offset_of_subarea(&mut self);
    fn set_scale_to_integer(&mut self, value: bool);
    fn set_clean_up_past_start(&mut self, value: bool);
    fn set_flat_filter_fraction(&mut self, value: String) -> Result<(), String>;
    fn set_skip_vert_slice_output(&mut self, value: bool);
    fn set_start_from_zero(&mut self, value: bool);
    fn set_resume_from_iteration(&mut self, value: i32);
    fn reset_resume_from_iteration(&mut self);
    fn set_resume(&mut self, value: bool);
}

/// Read methods of Java `SirtsetupParam` used by the overload of `setParameters`.
pub trait ConstSirtsetupParam {
    fn leave_iterations(&self) -> String;
    fn subarea_size(&self) -> Option<String>;
    fn y_offset_of_subarea(&self) -> Option<String>;
    fn scale_to_integer_is_null(&self) -> bool;
    fn clean_up_past_start(&self) -> bool;
    fn flat_filter_fraction(&self) -> String;
    fn skip_vert_slice_output(&self) -> bool;
    fn start_from_zero(&self) -> bool;
    fn resume_from_iteration(&self) -> Option<i32>;
}

/// Source-facing metadata operations from the two metadata overloads.
pub trait SirtPanelMetaData {
    fn gen_subarea(&self, axis_id: AxisID) -> bool;
    fn gen_subarea_size(&self, axis_id: AxisID) -> String;
    fn gen_y_offset_of_subarea(&self, axis_id: AxisID) -> String;
    fn set_gen_subarea(&mut self, axis_id: AxisID, value: bool);
    fn set_gen_subarea_size(&mut self, axis_id: AxisID, value: String);
    fn set_gen_y_offset_of_subarea(&mut self, axis_id: AxisID, value: String);
}

/// Java `ReconScreenState` reads and writes specific to this source unit.
pub trait SirtPanelReconScreenState {
    fn get_button_state(&self, key: Option<&str>) -> bool;
    fn set_button_state(&mut self, key: Option<&str>, state: bool);
    fn tomo_gen_sirt_header_state(&self) -> (bool, bool);
    fn set_tomo_gen_sirt_header_state(&mut self, value: (bool, bool));
}

/// In-memory `ReconScreenState` portion used by `SirtPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SirtPanelReconScreenStateBoundary {
    pub button_states: std::collections::BTreeMap<String, bool>,
    pub tomo_gen_sirt_header_state: (bool, bool),
}
impl SirtPanelReconScreenState for SirtPanelReconScreenStateBoundary {
    fn get_button_state(&self, key: Option<&str>) -> bool {
        key.and_then(|key| self.button_states.get(key))
            .copied()
            .unwrap_or(false)
    }
    fn set_button_state(&mut self, key: Option<&str>, state: bool) {
        if let Some(key) = key {
            self.button_states.insert(key.into(), state);
        }
    }
    fn tomo_gen_sirt_header_state(&self) -> (bool, bool) {
        self.tomo_gen_sirt_header_state
    }
    fn set_tomo_gen_sirt_header_state(&mut self, value: (bool, bool)) {
        self.tomo_gen_sirt_header_state = value;
    }
}

/// Java `ApplicationManager` calls reached by this source unit.
pub trait SirtPanelApplicationManager {
    fn sirtsetup(&mut self, axis_id: AxisID, dialog_type: DialogType);
    fn open_files_in_imod(&mut self, axis_id: AxisID, files: Vec<String>);
    fn use_sirt(&mut self, axis_id: AxisID, dialog_type: DialogType, file: String);
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
}

/// File-system and chooser boundary corresponding to `SirtOutputFileFilter` and `JFileChooser`.
pub trait SirtPanelFileChooser {
    fn sirt_output_files(&self, subarea: bool) -> Vec<String>;
    fn choose_sirt_output_files(&mut self, multiple: bool) -> Option<Vec<String>>;
    fn confirm_use_sirt(&mut self, file: &str) -> bool;
}

/// Source-visible Swing hierarchy and listeners installed by `createPanel`/`addListeners`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SirtPanelLayout {
    pub root_visible: bool,
    pub subarea_border: bool,
    pub sirtsetup_border: bool,
    pub sirtsetup_body_visible: bool,
    pub subarea_size_enabled: bool,
    pub y_offset_enabled: bool,
    pub radius_and_sigma_editable: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
}

/// Complete state owned by Java `SirtPanel`; `radiusAndSigmaPanel` remains the
/// separately translated radial-panel dependency at this unit's boundary.
pub struct SirtPanel {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub layout: SirtPanelLayout,
    pub cb_subarea: CheckBox,
    pub ltf_y_offset_of_subarea: LabeledTextField,
    pub ltf_subarea_size: LabeledTextField,
    pub ltf_leave_iterations: LabeledTextField,
    pub cb_scale_to_integer: CheckBox,
    pub btn_sirt: MultiLineButton,
    pub btn_3dmod_sirt: MultiLineButton,
    pub btn_use_sirt: MultiLineButton,
    pub cb_clean_up_past_start: CheckBox,
    pub ltf_flat_filter_fraction: LabeledTextField,
    pub rb_start_from_zero: RadioButton,
    pub rb_resume_from_last_iteration: RadioButton,
    pub rb_resume_from_iteration: RadioButton,
    pub cmb_resume_from_iteration: Vec<i32>,
    pub cmb_resume_from_iteration_selected: Option<usize>,
    pub cmb_resume_from_iteration_enabled: bool,
    pub cb_skip_vert_slice_output: CheckBox,
    pub num_files: usize,
    pub different_from_checkpoint_flag: bool,
    pub radius_and_sigma_editable: bool,
    pub sirt_method_selected: bool,
    pub sirt_setup_params_advanced: bool,
}

impl SirtPanel {
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut instance = Self {
            axis_id,
            dialog_type,
            layout: SirtPanelLayout::default(),
            cb_subarea: CheckBox::new_with_text("Reconstruct subarea"),
            ltf_y_offset_of_subarea: LabeledTextField::new(
                FieldType::FloatingPoint,
                " Offset in Y: ",
            ),
            ltf_subarea_size: LabeledTextField::new(FieldType::IntegerPair, "Size in X and Y: "),
            ltf_leave_iterations: LabeledTextField::new(
                FieldType::IntegerList,
                "Iteration #'s to retain: ",
            ),
            cb_scale_to_integer: CheckBox::new_with_text("Scale retained volumes to integers"),
            btn_sirt: MultiLineButton::new_with_label(Some("Run SIRT")),
            btn_3dmod_sirt: MultiLineButton::new_with_label(Some("View Tomogram(s) In 3dmod")),
            btn_use_sirt: MultiLineButton::new_toggle(Some("Use SIRT Output File"), true),
            cb_clean_up_past_start: CheckBox::new_with_text(
                "Delete existing reconstructions after starting point",
            ),
            ltf_flat_filter_fraction: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Flat filter fraction: ",
            ),
            rb_start_from_zero: RadioButton::new_in_group("Start from beginning", group.clone()),
            rb_resume_from_last_iteration: RadioButton::new_in_group(
                RESUME_FROM_LAST_ITERATION_LABEL,
                group.clone(),
            ),
            rb_resume_from_iteration: RadioButton::new_in_group(
                "Go back, resume from iteration:",
                group,
            ),
            cmb_resume_from_iteration: Vec::new(),
            cmb_resume_from_iteration_selected: None,
            cmb_resume_from_iteration_enabled: false,
            cb_skip_vert_slice_output: CheckBox::new_with_text(
                "Do not make vertical slice output files used for resuming",
            ),
            num_files: 0,
            different_from_checkpoint_flag: false,
            radius_and_sigma_editable: true,
            sirt_method_selected: true,
            sirt_setup_params_advanced: true,
        };
        instance
            .btn_sirt
            .create_button_state_key(Some(instance.dialog_type));
        instance
            .btn_use_sirt
            .create_button_state_key(Some(instance.dialog_type));
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    fn create_panel(&mut self) {
        self.layout.root_visible = true;
        self.layout.subarea_border = true;
        self.layout.sirtsetup_border = true;
        self.layout.sirtsetup_body_visible = true;
        self.rb_start_from_zero.set_selected(true);
        self.cb_clean_up_past_start.set_selected(true);
        self.update_display();
    }

    fn add_listeners(&mut self) {
        self.layout.listener_count = 9;
    }

    pub fn msg_field_changed(&mut self, different_from_checkpoint: bool) {
        self.different_from_checkpoint_flag = different_from_checkpoint;
        self.update_display();
    }

    pub fn update_display(&mut self) {
        self.ltf_flat_filter_fraction
            .set_visible(self.is_advanced());
        let subarea = self.cb_subarea.is_selected();
        self.ltf_subarea_size.set_enabled(subarea);
        self.ltf_y_offset_of_subarea.set_enabled(subarea);
        let enable_resume = self.num_files > 0
            && !self.is_different_from_checkpoint()
            && !self.different_from_checkpoint_flag;
        self.rb_resume_from_last_iteration
            .set_enabled(enable_resume);
        self.rb_resume_from_iteration.set_enabled(enable_resume);
        self.cmb_resume_from_iteration_enabled =
            enable_resume && self.rb_resume_from_iteration.is_selected();
        if !enable_resume
            && (self.rb_resume_from_last_iteration.is_selected()
                || self.rb_resume_from_iteration.is_selected())
        {
            self.rb_start_from_zero.set_selected(true);
        }
        let resume = self.is_resume();
        self.ltf_subarea_size.set_enabled(subarea && !resume);
        self.ltf_y_offset_of_subarea.set_enabled(subarea && !resume);
        self.radius_and_sigma_editable = !resume;
        self.cmb_resume_from_iteration_enabled = self.rb_resume_from_iteration.is_enabled()
            && self.rb_resume_from_iteration.is_selected();
        self.layout.subarea_size_enabled = self.ltf_subarea_size.is_enabled();
        self.layout.y_offset_enabled = self.ltf_y_offset_of_subarea.is_enabled();
        self.layout.radius_and_sigma_editable = self.radius_and_sigma_editable;
    }

    pub fn is_resume_enabled(&self) -> bool {
        self.rb_resume_from_last_iteration.is_enabled()
    }
    pub fn is_resume(&self) -> bool {
        self.rb_resume_from_last_iteration.is_selected()
            || self.rb_resume_from_iteration.is_selected()
    }
    pub fn msg_sirt_succeeded<F: SirtPanelFileChooser>(&mut self, chooser: &F) {
        self.load_resume_from(chooser);
    }
    pub fn msg_method_changed(&mut self, sirt: bool) {
        self.sirt_method_selected = sirt;
        self.layout.root_visible = sirt;
    }
    pub fn done(&mut self) {
        self.layout.listener_count = self.layout.listener_count.saturating_sub(2);
    }

    /// Java overloaded `getParameters(ReconScreenState)`.
    pub fn get_parameters_screen_state<S: SirtPanelReconScreenState>(&self, screen_state: &mut S) {
        screen_state.set_button_state(
            self.btn_sirt.state_key.as_deref(),
            self.btn_sirt.get_button_state(),
        );
        screen_state.set_button_state(
            self.btn_use_sirt.state_key.as_deref(),
            self.btn_use_sirt.get_button_state(),
        );
        screen_state.set_tomo_gen_sirt_header_state((
            self.layout.sirtsetup_body_visible,
            self.is_advanced(),
        ));
    }

    /// Java overloaded `setParameters(ReconScreenState)`.
    pub fn set_parameters_screen_state<S: SirtPanelReconScreenState>(&mut self, screen_state: &S) {
        let (open, advanced) = screen_state.tomo_gen_sirt_header_state();
        self.expand_open_close(open);
        self.expand_advanced_basic(advanced);
        let sirt_state = screen_state.get_button_state(self.btn_sirt.state_key.as_deref());
        let use_sirt_state = screen_state.get_button_state(self.btn_use_sirt.state_key.as_deref());
        self.btn_sirt.set_button_state(sirt_state);
        self.btn_use_sirt.set_button_state(use_sirt_state);
    }

    pub fn get_parameters_metadata<M: SirtPanelMetaData>(&self, metadata: &mut M) {
        metadata.set_gen_subarea(self.axis_id, self.cb_subarea.is_selected());
        metadata.set_gen_subarea_size(self.axis_id, self.ltf_subarea_size.get_text());
        metadata.set_gen_y_offset_of_subarea(self.axis_id, self.ltf_y_offset_of_subarea.get_text());
    }
    pub fn set_parameters_metadata<M: SirtPanelMetaData, F: SirtPanelFileChooser>(
        &mut self,
        metadata: &M,
        chooser: &F,
    ) {
        self.cb_subarea
            .set_selected(metadata.gen_subarea(self.axis_id));
        self.ltf_subarea_size
            .set_text(&metadata.gen_subarea_size(self.axis_id));
        self.ltf_y_offset_of_subarea
            .set_text(&metadata.gen_y_offset_of_subarea(self.axis_id));
        self.load_resume_from(chooser);
    }

    pub fn get_parameters_sirtsetup<P: SirtsetupParam, M: SirtPanelApplicationManager>(
        &self,
        param: &mut P,
        manager: &mut M,
        do_validation: bool,
    ) -> bool {
        if do_validation && self.ltf_leave_iterations.is_empty() {
            manager.open_message_dialog(
                format!("{} is empty.", self.ltf_leave_iterations.label),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        if param
            .set_leave_iterations(self.ltf_leave_iterations.get_text())
            .is_err()
        {
            return false;
        }
        if self.cb_subarea.is_selected() {
            if self.ltf_subarea_size.is_empty() {
                manager.open_message_dialog(
                    format!("{} is empty.", self.ltf_subarea_size.label),
                    "Entry Error",
                    self.axis_id,
                );
                return false;
            }
            if param
                .set_subarea_size(self.ltf_subarea_size.get_text())
                .is_err()
                || param
                    .set_y_offset_of_subarea(self.ltf_y_offset_of_subarea.get_text())
                    .is_err()
            {
                return false;
            }
        } else {
            param.reset_subarea_size();
            param.reset_y_offset_of_subarea();
        }
        param.set_scale_to_integer(self.cb_scale_to_integer.is_selected());
        param.set_clean_up_past_start(self.cb_clean_up_past_start.is_selected());
        if param
            .set_flat_filter_fraction(self.ltf_flat_filter_fraction.get_text())
            .is_err()
        {
            return false;
        }
        param.set_skip_vert_slice_output(self.cb_skip_vert_slice_output.is_selected());
        let mut resume = false;
        if self.rb_start_from_zero.is_selected() {
            param.set_start_from_zero(true);
            param.reset_resume_from_iteration();
        } else if self.rb_resume_from_last_iteration.is_enabled()
            && self.rb_resume_from_last_iteration.is_selected()
        {
            param.set_start_from_zero(false);
            param.reset_resume_from_iteration();
            resume = true;
        } else if self.rb_resume_from_iteration.is_enabled()
            && self.rb_resume_from_iteration.is_selected()
        {
            let Some(index) = self.cmb_resume_from_iteration_selected else {
                return false;
            };
            let Some(value) = self.cmb_resume_from_iteration.get(index) else {
                return false;
            };
            param.set_start_from_zero(false);
            param.set_resume_from_iteration(*value);
            resume = true;
        } else {
            manager.open_message_dialog(
                "Please select an enabled starting option.".into(),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        param.set_resume(resume);
        true
    }

    pub fn set_parameters_sirtsetup<P: ConstSirtsetupParam>(&mut self, param: &P) {
        self.ltf_leave_iterations
            .set_text(&param.leave_iterations());
        if let Some(value) = param.subarea_size() {
            self.ltf_subarea_size.set_text(&value);
        }
        if let Some(value) = param.y_offset_of_subarea() {
            self.ltf_y_offset_of_subarea.set_text(&value);
        }
        self.cb_scale_to_integer
            .set_selected(!param.scale_to_integer_is_null());
        self.cb_clean_up_past_start
            .set_selected(param.clean_up_past_start());
        self.ltf_flat_filter_fraction
            .set_text(&param.flat_filter_fraction());
        self.cb_skip_vert_slice_output
            .set_selected(param.skip_vert_slice_output());
        if param.start_from_zero() {
            self.rb_start_from_zero.set_selected(true);
        } else if param.resume_from_iteration().is_some() {
            self.rb_resume_from_iteration.set_selected(true);
        } else {
            self.rb_resume_from_last_iteration.set_selected(true);
        }
        self.update_display();
    }

    /// Java `loadResumeFrom`, with matching numeric sort and descending pulldown order.
    pub fn load_resume_from<F: SirtPanelFileChooser>(&mut self, chooser: &F) {
        self.cmb_resume_from_iteration.clear();
        self.cmb_resume_from_iteration_selected = None;
        let mut values: Vec<i32> = chooser
            .sirt_output_files(self.cb_subarea.is_selected())
            .into_iter()
            .filter_map(|file| {
                let digits: String = file
                    .chars()
                    .rev()
                    .take_while(char::is_ascii_digit)
                    .collect();
                (!digits.is_empty())
                    .then(|| digits.chars().rev().collect::<String>().parse().ok())
                    .flatten()
            })
            .collect();
        values.sort_unstable();
        for value in values.iter().rev() {
            self.cmb_resume_from_iteration.push(*value);
        }
        if let Some(value) = values.last() {
            self.rb_resume_from_last_iteration
                .set_text(format!("{RESUME_FROM_LAST_ITERATION_LABEL}: {value}"));
        } else {
            self.rb_resume_from_last_iteration
                .set_text(RESUME_FROM_LAST_ITERATION_LABEL);
        }
        if values.len() > 1 {
            self.cmb_resume_from_iteration_selected = Some(1);
        }
        self.num_files = values.len();
        self.update_display();
    }

    pub fn open_files_in_imod<F: SirtPanelFileChooser, M: SirtPanelApplicationManager>(
        &mut self,
        chooser: &mut F,
        manager: &mut M,
    ) {
        let files = chooser.sirt_output_files(self.cb_subarea.is_selected());
        let files = if files.len() == 1 {
            files
        } else {
            chooser.choose_sirt_output_files(true).unwrap_or_default()
        };
        if !files.is_empty() {
            manager.open_files_in_imod(self.axis_id, files);
        }
    }
    pub fn use_sirt<F: SirtPanelFileChooser, M: SirtPanelApplicationManager>(
        &mut self,
        chooser: &mut F,
        manager: &mut M,
    ) {
        let files = chooser.sirt_output_files(self.cb_subarea.is_selected());
        let file = if files.len() == 1 {
            chooser
                .confirm_use_sirt(&files[0])
                .then(|| files[0].clone())
        } else {
            chooser
                .choose_sirt_output_files(false)
                .and_then(|mut files| files.pop())
        };
        if let Some(file) = file {
            manager.use_sirt(self.axis_id, self.dialog_type, file);
        }
    }
    pub fn expand_open_close(&mut self, expanded: bool) {
        self.layout.sirtsetup_body_visible = expanded;
    }
    pub fn expand_advanced_basic(&mut self, advanced: bool) {
        self.sirt_setup_params_advanced = advanced;
        self.ltf_flat_filter_fraction.set_visible(advanced);
    }
    pub fn is_advanced(&self) -> bool {
        self.sirt_setup_params_advanced
    }
    pub fn is_ctf3d(&self) -> bool {
        false
    }
    pub fn is_multifilt(&self) -> bool {
        false
    }
    pub fn checkpoint(&mut self, subarea_size: &str, y_offset: &str) {
        self.ltf_subarea_size.checkpoint_value(subarea_size);
        self.ltf_y_offset_of_subarea.checkpoint_value(y_offset);
        self.update_display();
    }
    pub fn is_different_from_checkpoint(&self) -> bool {
        self.ltf_subarea_size.is_different_from_checkpoint(false)
            || self
                .ltf_y_offset_of_subarea
                .is_different_from_checkpoint(false)
    }
    pub fn action<F: SirtPanelFileChooser, M: SirtPanelApplicationManager>(
        &mut self,
        action_command: &str,
        chooser: &mut F,
        manager: &mut M,
    ) {
        if self.btn_sirt.get_action_command() == Some(action_command) {
            manager.sirtsetup(self.axis_id, self.dialog_type);
        } else if self.btn_3dmod_sirt.get_action_command() == Some(action_command) {
            self.open_files_in_imod(chooser, manager);
        } else if self.btn_use_sirt.get_action_command() == Some(action_command) {
            self.use_sirt(chooser, manager);
        } else if action_command == self.rb_start_from_zero.get_action_command()
            || action_command == self.rb_resume_from_last_iteration.get_action_command()
            || action_command == self.rb_resume_from_iteration.get_action_command()
        {
            self.update_display();
        } else if self.cb_subarea.get_action_command() == Some(action_command) {
            self.load_resume_from(chooser);
        }
    }
    pub fn document_action(&mut self) {
        self.update_display();
    }
    fn set_tool_tip_text(&mut self) {
        self.layout.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Files(Vec<String>);
    impl SirtPanelFileChooser for Files {
        fn sirt_output_files(&self, _: bool) -> Vec<String> {
            self.0.clone()
        }
        fn choose_sirt_output_files(&mut self, _: bool) -> Option<Vec<String>> {
            Some(self.0.clone())
        }
        fn confirm_use_sirt(&mut self, _: &str) -> bool {
            true
        }
    }
    #[test]
    fn resume_numbers_are_sorted_descending_and_select_previous() {
        let mut panel = SirtPanel::get_instance(AxisID::Only, DialogType::TomogramGeneration);
        panel.load_resume_from(&Files(vec![
            "x.srec003".into(),
            "x.srec012".into(),
            "x.srec002".into(),
        ]));
        assert_eq!(panel.cmb_resume_from_iteration, vec![12, 3, 2]);
        assert_eq!(panel.cmb_resume_from_iteration_selected, Some(1));
        assert_eq!(panel.num_files, 3);
    }
    #[test]
    fn resume_is_disabled_by_checkpoint_difference() {
        let mut panel = SirtPanel::get_instance(AxisID::Only, DialogType::TomogramGeneration);
        panel.load_resume_from(&Files(vec!["x.srec003".into()]));
        panel.msg_field_changed(true);
        assert!(!panel.is_resume_enabled());
        assert!(panel.rb_start_from_zero.is_selected());
    }

    #[test]
    fn screen_state_retains_header_and_process_button_state() {
        let mut panel = SirtPanel::get_instance(AxisID::Only, DialogType::TomogramGeneration);
        panel.expand_open_close(false);
        panel.expand_advanced_basic(false);
        panel.btn_sirt.set_button_state(true);
        let mut state = SirtPanelReconScreenStateBoundary::default();
        panel.get_parameters_screen_state(&mut state);
        let mut restored = SirtPanel::get_instance(AxisID::Only, DialogType::TomogramGeneration);
        restored.set_parameters_screen_state(&state);
        assert!(!restored.layout.sirtsetup_body_visible);
        assert!(restored.btn_sirt.get_button_state());
    }
}
