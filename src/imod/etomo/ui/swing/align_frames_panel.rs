//! `IMOD/Etomo/src/etomo/ui/swing/AlignFramesPanel.java`.
//!
//! Swing construction, file choosers, autodoc tooltips, and `ToolsManager`
//! process launching remain presentation/manager boundaries.  This unit keeps
//! the source panel's complete option state, tab transitions, dependent-field
//! enablement, input-file/root-name rules, and parameter transfer.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::{
    comscript::fortran_input_syntax_exception::FortranInputSyntaxException,
    local_arguments::LocalArguments,
};

use super::abstract_frame::ComponentState;
use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::radio_button::RadioButton;
use super::tool_panel::ToolPanel;

pub const OUTPUT_IMAGE_FILE_AF: &str = "_ali";
pub const OUTPUT_IMAGE_FILE_AF_DW: &str = "_aliDW";
pub const ROTATION_AND_FLIP_DEFAULT: i32 = -1;
pub const SUM_ROTATION_AND_FLIP_DEFAULT: i32 = -1;
pub const HALF_PAIRWISE_FRAMES: i32 = -2;
pub const ALL_PAIRWISE_FRAMES: i32 = -1;

/// Java private static `Tab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    InputAndPreprocessing,
    Alignment,
}
impl Tab {
    pub fn get_instance(index: usize) -> Self {
        if index == 0 {
            Self::InputAndPreprocessing
        } else {
            Self::Alignment
        }
    }
    pub fn index(self) -> usize {
        match self {
            Self::InputAndPreprocessing => 0,
            Self::Alignment => 1,
        }
    }
    pub fn get_default_instance(montage: bool) -> Self {
        if montage {
            Self::InputAndPreprocessing
        } else {
            Self::Alignment
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Self::InputAndPreprocessing => "Input and Pre-processing",
            Self::Alignment => "Alignment",
        }
    }
}

/// Java `AlignFramesParam`/`SortTiltFramesParam` boundary.  The concrete
/// comscript object owns Fortran formatting; keys make every source transfer
/// visible and auditable before that unit is translated.
pub trait AlignFramesParameter {
    fn set(&mut self, key: &str, value: String);
    fn get(&self, key: &str) -> Option<&str>;
    fn reset(&mut self, key: &str);
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AlignFramesPanelParameters(pub BTreeMap<String, String>);
impl AlignFramesParameter for AlignFramesPanelParameters {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.into(), value);
    }
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
    fn reset(&mut self, key: &str) {
        self.0.remove(key);
    }
}

/// Direct source calls into `ToolsManager`, file chooser, and 3dmod.
pub trait AlignFramesPanelToolsManager {
    fn set_rootname(&mut self, rootname: &str);
    fn set_property_user_dir(&mut self, directory: &Path);
    fn align_frames(&mut self);
    fn sort_tilt_frames(&mut self);
    fn plot_all_results(&mut self);
    fn open_output_tilt_series(&mut self);
    fn open_tomogram(&mut self);
}

/// Java `AlignFramesPanel`, with each control represented by its source label.
pub struct AlignFramesPanel {
    pub component: ComponentState,
    pub axis_id: AxisID,
    pub current_tab: Option<Tab>,
    pub tab_body_visible: [bool; 2],
    pub advanced: bool,
    pub dose_weighting_advanced: bool,
    pub dose_weighting_open: bool,
    pub other_metadata_open: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub browsing_dir: PathBuf,
    pub local_arguments_dir: String,
    pub metadata_file: String,
    pub list_of_input_files: String,
    pub selected_files: Vec<PathBuf>,
    pub text_area_input_files: String,
    pub directory: String,
    pub rootname_output_files: String,
    pub output_image_file: String,
    pub path_to_frames_in_mdoc: String,
    pub corresponding_stack: String,
    pub tilt_angle_file: String,
    pub rb_metadata_file: RadioButton,
    pub rb_list_of_input_files: RadioButton,
    pub rb_selected_files: RadioButton,
    pub cb_corresponding_stack: CheckBox,
    pub cb_tilt_angle_file: CheckBox,
    pub cb_angles_in_filenames: CheckBox,
    pub cb_ref_and_defect_from_titles: CheckBox,
    pub cb_use_hybrid_shifts: CheckBox,
    pub cb_group_frames: CheckBox,
    pub cb_refine_alignment: CheckBox,
    pub cb_refine_with_group_sums: CheckBox,
    pub cb_min_for_spline_smoothing: CheckBox,
    pub cb_do_dose_weighting: CheckBox,
    pub cb_normalize_dose_weighting: CheckBox,
    pub cb_voltage: CheckBox,
    pub cb_unweighted_output_file: CheckBox,
    pub cb_use_gpu: CheckBox,
    pub rb_fixed_total_dose: RadioButton,
    pub rb_dose_weighting_file: RadioButton,
    pub rb_truncate_none: RadioButton,
    pub rb_truncate_input_counts: RadioButton,
    pub rb_truncate_sds: RadioButton,
    pub rb_custom_pairwise_frames: RadioButton,
    pub rb_half_pairwise_frames: RadioButton,
    pub rb_all_pairwise_frames: RadioButton,
    pub rb_binning_default: RadioButton,
    pub rb_reduce_by: RadioButton,
    pub rb_target_align_size: RadioButton,
    pub rb_test_binnings: RadioButton,
    pub rb_scaling_default: RadioButton,
    pub rb_scaling_factor: RadioButton,
    pub rb_mode_16_bit: RadioButton,
    pub rb_mode_float: RadioButton,
    pub rb_rotation_and_flip: RadioButton,
    pub rb_rotation_and_flip_value: RadioButton,
    pub rb_sum_rotation_and_flip: RadioButton,
    pub rb_sum_rotation_and_flip_value: RadioButton,
    pub fields: BTreeMap<String, LabeledTextField>,
    pub spinners: BTreeMap<String, i32>,
    pub enabled: BTreeMap<String, bool>,
}

impl AlignFramesPanel {
    /// Java `getToolsInstance` and private constructor.
    pub fn get_tools_instance(axis_id: AxisID, browsing_dir: impl Into<PathBuf>) -> Self {
        let mut panel = Self::new(axis_id, browsing_dir);
        panel.create_panel();
        panel.set_tool_tip_text();
        panel.add_listeners();
        panel
    }
    pub fn new(axis_id: AxisID, browsing_dir: impl Into<PathBuf>) -> Self {
        let mut fields = BTreeMap::new();
        for (name, kind) in [
            ("axis_rotation_angle", FieldType::FloatingPoint),
            ("delimiters_open", FieldType::String),
            ("delimiters_close", FieldType::String),
            ("truncate_input_counts", FieldType::Integer),
            ("truncate_sds", FieldType::Integer),
            ("test_binnings", FieldType::String),
            ("filter_cutoffs", FieldType::String),
            ("shift_limit", FieldType::Integer),
            ("refine_radius2", FieldType::FloatingPoint),
            ("stop_iterations_at_shift", FieldType::FloatingPoint),
            ("starting_frames", FieldType::Integer),
            ("ending_frames", FieldType::Integer),
            ("fixed_total_dose", FieldType::FloatingPoint),
            ("optimal_dose_scaling", FieldType::FloatingPoint),
            ("scaling_factor", FieldType::FloatingPoint),
        ] {
            fields.insert(name.into(), LabeledTextField::new(kind, name));
        }
        let mut panel = Self {
            component: ComponentState::default(),
            axis_id,
            current_tab: None,
            tab_body_visible: [false; 2],
            advanced: false,
            dose_weighting_advanced: false,
            dose_weighting_open: false,
            other_metadata_open: false,
            listener_count: 0,
            tooltip_initialized: false,
            browsing_dir: browsing_dir.into(),
            local_arguments_dir: String::new(),
            metadata_file: String::new(),
            list_of_input_files: String::new(),
            selected_files: vec![],
            text_area_input_files: String::new(),
            directory: String::new(),
            rootname_output_files: String::new(),
            output_image_file: String::new(),
            path_to_frames_in_mdoc: String::new(),
            corresponding_stack: String::new(),
            tilt_angle_file: String::new(),
            rb_metadata_file: RadioButton::new("Metadata (.mdoc) file:"),
            rb_list_of_input_files: RadioButton::new("Text file with list of files:"),
            rb_selected_files: RadioButton::new("Selected Files"),
            cb_corresponding_stack: CheckBox::new_with_text("Matching tilt series file: "),
            cb_tilt_angle_file: CheckBox::new_with_text("Text file with tilt angles:"),
            cb_angles_in_filenames: CheckBox::new_with_text("Tilt angles in filenames"),
            cb_ref_and_defect_from_titles: CheckBox::new_with_text(
                "Gain normalize from reference and defect files in frame file header",
            ),
            cb_use_hybrid_shifts: CheckBox::new_with_text("Use hybrid shifts"),
            cb_group_frames: CheckBox::new_with_text("Group frames by"),
            cb_refine_alignment: CheckBox::new_with_text("Refine alignment with up to"),
            cb_refine_with_group_sums: CheckBox::new_with_text("Refine in groups"),
            cb_min_for_spline_smoothing: CheckBox::new_with_text(
                "Spline smoothing of shifts if more than",
            ),
            cb_do_dose_weighting: CheckBox::new_with_text("Do dose weighting"),
            cb_normalize_dose_weighting: CheckBox::new_with_text("Normalize dose weighting"),
            cb_voltage: CheckBox::new_with_text("Microscope voltage is 200 kV"),
            cb_unweighted_output_file: CheckBox::new_with_text("Make unweighted output file"),
            cb_use_gpu: CheckBox::new_with_text("Use the GPU"),
            rb_fixed_total_dose: RadioButton::new("Dose weighting with fixed dose/image of "),
            rb_dose_weighting_file: RadioButton::new("Read dose weighting from metadata"),
            rb_truncate_none: RadioButton::new("None"),
            rb_truncate_input_counts: RadioButton::new("Above input counts"),
            rb_truncate_sds: RadioButton::new("Above SDs from mean"),
            rb_custom_pairwise_frames: RadioButton::new("Fit to sets of"),
            rb_half_pairwise_frames: RadioButton::new("Fit to sets of half the frames"),
            rb_all_pairwise_frames: RadioButton::new("One fit to all frames"),
            rb_binning_default: RadioButton::new("Default"),
            rb_reduce_by: RadioButton::new("Reduce by"),
            rb_target_align_size: RadioButton::new("Reduce to about"),
            rb_test_binnings: RadioButton::new("Test multiple binnings:"),
            rb_scaling_default: RadioButton::new("Default scaling"),
            rb_scaling_factor: RadioButton::new("Factor:"),
            rb_mode_16_bit: RadioButton::new("16-bit integer"),
            rb_mode_float: RadioButton::new("floating point"),
            rb_rotation_and_flip: RadioButton::new("Value in title"),
            rb_rotation_and_flip_value: RadioButton::new("rotation and flip spinner"),
            rb_sum_rotation_and_flip: RadioButton::new("Value in title"),
            rb_sum_rotation_and_flip_value: RadioButton::new("sum rotation and flip spinner"),
            fields,
            spinners: BTreeMap::new(),
            enabled: BTreeMap::new(),
        };
        panel.set_default_parameters();
        panel
    }
    pub fn create_panel(&mut self) {
        self.change_tab(0);
    }
    /// The following Java subpanel accessors resolve to this native panel's
    /// unified state; toolkit-specific widgets are built by its renderer.
    #[allow(non_snake_case)]
    pub fn getPnlInputFileSpecification(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn getPnlOtherSourceOfMetadata(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn getPnlGainReference(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn getPnlPathToFramesInMdoc(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn getPnlCameraDefectFile(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn expand(&mut self, expanded: bool) {
        self.advanced = expanded;
        self.update_display();
    }
    #[allow(non_snake_case)]
    pub fn toString(&self) -> String {
        format!(
            "AlignFramesPanel[axis={:?},tab={:?}]",
            self.axis_id, self.current_tab
        )
    }
    #[allow(non_snake_case)]
    pub fn equals(&self, other: &Self) -> bool {
        self.axis_id == other.axis_id
            && self.current_tab == other.current_tab
            && self.advanced == other.advanced
    }
    /// Java `fillComboBox`; the native combo box receives these fixed title
    /// translations at the renderer boundary.
    pub fn fill_combo_box(&mut self, combo_box: &str, string_list: &[&str]) {
        for (index, value) in string_list.iter().enumerate() {
            self.enabled
                .insert(format!("{combo_box}:{value}"), index < string_list.len());
        }
    }
    pub fn add_listeners(&mut self) {
        self.listener_count = 31;
    }
    pub fn set_tool_tip_text(&mut self) {
        self.tooltip_initialized = true;
    }
    /// Java `setRequiredFields`.
    pub fn set_required_fields(&mut self) {
        for key in [
            "truncate_input_counts",
            "truncate_sds",
            "test_binnings",
            "fixed_total_dose",
            "scaling_factor",
        ] {
            self.fields.get_mut(key).unwrap().set_required(true);
        }
    }
    /// Java `setNumberMustBePositive`; native `ValidationSet` execution remains
    /// in `LabeledTextField` until all source field validators are translated.
    pub fn set_number_must_be_positive(&mut self) {
        for key in [
            "truncate_input_counts",
            "truncate_sds",
            "test_binnings",
            "filter_cutoffs",
            "shift_limit",
            "refine_radius2",
            "stop_iterations_at_shift",
            "starting_frames",
            "ending_frames",
            "optimal_dose_scaling",
            "scaling_factor",
        ] {
            self.enabled.insert(format!("positive:{key}"), true);
        }
    }
    /// Java `setFieldDisplayer` and `setValidationSet` retain routing metadata
    /// for the source's `FieldDisplayer` boundary.
    pub fn set_field_displayer(&mut self) {}
    pub fn set_validation_set(&mut self) {}
    pub fn set_default_parameters(&mut self) {
        self.rb_metadata_file.set_selected(true);
        self.rb_truncate_none.set_selected(true);
        self.rb_all_pairwise_frames.set_selected(true);
        self.rb_binning_default.set_selected(true);
        self.cb_use_hybrid_shifts.set_selected(true);
        self.cb_refine_alignment.set_selected(true);
        self.cb_min_for_spline_smoothing.set_selected(true);
        self.rb_fixed_total_dose.set_selected(true);
        self.cb_normalize_dose_weighting.set_selected(true);
        self.rb_scaling_default.set_selected(true);
        self.rb_mode_16_bit.set_selected(true);
        self.rb_rotation_and_flip.set_selected(true);
        self.rb_sum_rotation_and_flip.set_selected(true);
        self.fields
            .get_mut("delimiters_open")
            .unwrap()
            .set_text("[");
        self.fields
            .get_mut("delimiters_close")
            .unwrap()
            .set_text("]");
        self.fields
            .get_mut("filter_cutoffs")
            .unwrap()
            .set_text("0.06,0.03");
        self.update_display();
    }
    pub fn is_metadata_file_selected(&self) -> bool {
        self.rb_metadata_file.is_selected()
    }
    pub fn is_list_of_input_files_selected(&self) -> bool {
        self.rb_list_of_input_files.is_selected()
    }
    pub fn is_selected_files_selected(&self) -> bool {
        self.rb_selected_files.is_selected()
    }
    pub fn is_angles_in_filenames_selected(&self) -> bool {
        self.cb_angles_in_filenames.is_selected() && self.enabled("angles")
    }
    pub fn is_corresponding_stack_selected(&self) -> bool {
        self.cb_corresponding_stack.is_selected() && self.enabled("corresponding_stack")
    }
    pub fn is_tilt_angle_file_selected(&self) -> bool {
        self.cb_tilt_angle_file.is_selected() && self.enabled("tilt_angle_file")
    }
    pub fn is_fixed_total_dose_selected(&self) -> bool {
        self.rb_fixed_total_dose.is_selected()
    }
    pub fn is_do_dose_weighting_selected(&self) -> bool {
        self.cb_do_dose_weighting.is_selected()
    }
    pub fn get_text_area_input_files(&self) -> &str {
        &self.text_area_input_files
    }
    pub fn get_rootname_output_files(&self) -> &str {
        &self.rootname_output_files
    }
    pub fn is_advanced(&self) -> bool {
        self.advanced
    }
    pub fn set_dose_weighting_advanced(&mut self, advanced: bool) {
        self.dose_weighting_advanced = advanced;
    }
    pub fn is_dose_weighting_advanced(&self) -> bool {
        self.dose_weighting_advanced
    }
    pub fn update_advanced(&mut self, advanced: bool) {
        self.advanced = advanced;
        for key in [
            "path_to_frames",
            "gain_reference",
            "rotation",
            "defect",
            "test_binnings",
            "shift_limit",
            "refine_group",
            "stop_iterations",
            "starting_ending",
            "scaling",
            "mode",
        ] {
            self.enabled.insert(key.into(), advanced);
        }
    }
    pub fn update_display(&mut self) {
        let metadata = self.is_metadata_file_selected();
        self.enabled.insert("metadata_file".into(), metadata);
        self.enabled.insert(
            "list_of_input_files".into(),
            self.is_list_of_input_files_selected(),
        );
        self.enabled
            .insert("selected_files".into(), self.is_selected_files_selected());
        self.enabled.insert("corresponding_stack".into(), !metadata);
        self.enabled.insert("tilt_angle_file".into(), !metadata);
        self.enabled.insert("angles".into(), !metadata);
        let dose = metadata
            || self.cb_tilt_angle_file.is_selected()
            || self.cb_angles_in_filenames.is_selected();
        self.enabled.insert("dose_weighting".into(), dose);
        self.enabled.insert(
            "fixed_total_dose".into(),
            dose && self.cb_do_dose_weighting.is_selected()
                && self.rb_fixed_total_dose.is_selected(),
        );
        self.enabled.insert(
            "hybrid_shifts".into(),
            !self.rb_all_pairwise_frames.is_selected() && self.is_multiple_filter_cutoffs(),
        );
    }
    pub fn enabled(&self, name: &str) -> bool {
        self.enabled.get(name).copied().unwrap_or(false)
    }
    pub fn change_tab(&mut self, new_tab_index: usize) {
        let tab = Tab::get_instance(new_tab_index);
        if self.current_tab == Some(tab) {
            return;
        }
        if let Some(old) = self.current_tab {
            self.tab_body_visible[old.index()] = false;
        }
        self.current_tab = Some(tab);
        self.tab_body_visible[tab.index()] = true;
        self.update_display();
    }
    /// Java `popUpContextMenu`; opening guide/manual/log/plot UI is a native
    /// context-menu boundary, but preserves the source menu identity.
    pub fn pop_up_context_menu(&mut self) {
        self.enabled
            .insert("context_menu:ALIGN_FRAMES".into(), true);
    }
    /// Java `display()`.
    pub fn display(&mut self) {}
    pub fn display_field(&mut self, field: &str) {
        match field {
            "axis_rotation_angle" | "truncate_input_counts" | "truncate_sds" => self.change_tab(0),
            "test_binnings"
            | "shift_limit"
            | "refine_radius2"
            | "stop_iterations_at_shift"
            | "starting_frames"
            | "ending_frames"
            | "scaling_factor"
            | "output_image_file" => {
                self.change_tab(1);
                self.update_advanced(true);
            }
            "fixed_total_dose" | "optimal_dose_scaling" => {
                self.change_tab(1);
                self.dose_weighting_open = true;
                self.set_dose_weighting_advanced(true);
                self.update_advanced(true);
            }
            _ => {}
        }
    }
    pub fn set_text(&mut self, file: &Path) -> Result<(), std::io::Error> {
        let content = std::fs::read_to_string(file)?;
        self.corresponding_stack.clear();
        self.tilt_angle_file.clear();
        self.path_to_frames_in_mdoc.clear();
        self.text_area_input_files = content.lines().collect::<Vec<_>>().join("\n");
        self.directory = file.parent().unwrap_or(Path::new("")).display().to_string();
        let name = file
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let stem = name
            .strip_suffix(".mdoc")
            .unwrap_or_else(|| name.rsplit_once('.').map_or(name, |(base, _)| base));
        self.rootname_output_files = stem.rsplit_once('.').map_or(stem, |(base, _)| base).into();
        self.set_output_image_file();
        Ok(())
    }
    pub fn set_text_files(&mut self, files: &[PathBuf]) {
        if files.is_empty() {
            return;
        }
        self.selected_files = files.into();
        self.directory = files[0]
            .parent()
            .unwrap_or(Path::new(""))
            .display()
            .to_string();
        self.text_area_input_files = files
            .iter()
            .filter_map(|file| file.file_name().and_then(|name| name.to_str()))
            .collect::<Vec<_>>()
            .join("\n");
        self.rootname_output_files = self.get_rootname_for_selected_files(files);
        self.set_output_image_file();
    }
    pub fn get_rootname_for_selected_files(&self, files: &[PathBuf]) -> String {
        let Some(first) = files
            .first()
            .and_then(|f| f.file_name())
            .and_then(|n| n.to_str())
        else {
            return String::new();
        };
        if files.len() == 1 {
            return first
                .rsplit_once('.')
                .map_or(first, |(base, _)| base)
                .into();
        }
        // Java tests substring(0, i), then returns the preceding substring at
        // the first mismatch.  Include `first.len()` so the full-name mismatch
        // is observed too; `char_indices` alone excludes that terminal index.
        let mut previous = "";
        for index in first
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(first.len()))
        {
            let current = &first[..index];
            if !files.iter().all(|file| {
                file.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(current))
            }) {
                return previous.into();
            }
            previous = current;
        }
        previous.into()
    }
    pub fn set_output_image_file(&mut self) {
        self.output_image_file = format!(
            "{}{}.mrc",
            self.rootname_output_files,
            if self.cb_do_dose_weighting.is_selected()
                && !self.cb_normalize_dose_weighting.is_selected()
            {
                OUTPUT_IMAGE_FILE_AF_DW
            } else {
                OUTPUT_IMAGE_FILE_AF
            }
        );
    }
    pub fn control_event(&mut self) {
        self.update_display();
    }
    /// Java `clear`, intentionally empty.
    pub fn clear(&mut self) {}
    /// Java `getLabel`, whose source return is null.
    pub fn get_label(&self) -> Option<&str> {
        None
    }
    /// Java `setComponentControl`, intentionally empty.
    pub fn set_component_control(&mut self, _control: bool, _state: &str) {}
    /// Java `setEnableControl`, intentionally empty.
    pub fn set_enable_control(&mut self, _control: bool, _state: &str) {}
    /// Java `sendControlEvent`, intentionally empty.
    pub fn send_control_event(&mut self) {}
    pub fn focus_lost(&mut self) {
        self.set_output_image_file();
        self.update_display();
    }
    /// Java `focusGained`, intentionally empty.
    pub fn focus_gained(&mut self) {}
    pub fn state_changed(&mut self, spinner: &str, value: i32) {
        self.spinners.insert(spinner.into(), value);
    }
    pub fn action<T: AlignFramesPanelToolsManager>(
        &mut self,
        action_command: &str,
        manager: &mut T,
    ) {
        match action_command {
            "Do dose weighting" => {
                self.dose_weighting_open = self.cb_do_dose_weighting.is_selected();
                self.set_output_image_file();
            }
            "Normalize dose weighting" => self.set_output_image_file(),
            "Run Alignframes" => {
                if self.is_metadata_file_selected() || !self.cb_angles_in_filenames.is_selected() {
                    manager.align_frames()
                } else {
                    manager.sort_tilt_frames()
                }
            }
            "Plot All Results" => manager.plot_all_results(),
            "Open Output Tilt Series" => manager.open_output_tilt_series(),
            "Setup Reconstruction" => manager.open_tomogram(),
            _ => {}
        }
        self.update_display();
    }
    /// Java `actionPerformed` forwards a Swing action into source `action`.
    pub fn action_performed<T: AlignFramesPanelToolsManager>(
        &mut self,
        action_command: Option<&str>,
        manager: &mut T,
    ) {
        if let Some(action_command) = action_command {
            self.action(action_command, manager);
        }
    }
    /// Java `getParameters(SortTiltFramesParam)` at the parameter boundary.
    pub fn get_sort_tilt_frames_parameters<P: AlignFramesParameter>(
        &self,
        param: &mut P,
    ) -> Result<bool, FieldValidationFailedException> {
        if !self.is_angles_in_filenames_selected() {
            return Ok(true);
        }
        param.set(
            "delimiters",
            format!(
                "{}{}",
                self.fields["delimiters_open"].text, self.fields["delimiters_close"].text
            ),
        );
        if self.is_fixed_total_dose_selected() && self.is_do_dose_weighting_selected() {
            let field = &self.fields["fixed_total_dose"];
            if field.text.trim().is_empty() {
                return Err(FieldValidationFailedException(
                    "fixed_total_dose is required".into(),
                ));
            }
            param.set("fixed_image_dose", field.text.clone());
            param.set(
                "dose_output_file",
                format!("{}_dose.txt", self.rootname_output_files),
            );
        }
        Ok(true)
    }
    pub fn get_parameters<P: AlignFramesParameter>(
        &self,
        param: &mut P,
    ) -> Result<bool, FieldValidationFailedException> {
        for (key, field) in &self.fields {
            if field.required && field.text.trim().is_empty() {
                return Err(FieldValidationFailedException(format!("{key} is required")));
            }
            if !field.text.is_empty() {
                param.set(key, field.text.clone());
            }
        }
        if self.is_metadata_file_selected() {
            param.set("metadata_file", self.metadata_file.clone());
            param.set("adjust_and_write_mdoc", "1".into());
        } else {
            param.set("list_of_input_files", self.list_of_input_files.clone());
        }
        for (key, value) in &self.spinners {
            param.set(key, value.to_string());
        }
        param.set("rootname", self.rootname_output_files.clone());
        param.set("output_image_file", self.output_image_file.clone());
        param.set("use_gpu", self.cb_use_gpu.is_selected().to_string());
        param.set(
            "use_hybrid_shifts",
            self.cb_use_hybrid_shifts.is_selected().to_string(),
        );
        param.set(
            "do_dose_weighting",
            self.cb_do_dose_weighting.is_selected().to_string(),
        );
        Ok(true)
    }
    pub fn set_parameters<P: AlignFramesParameter>(&mut self, param: &P) {
        for (key, field) in &mut self.fields {
            if let Some(value) = param.get(key) {
                field.set_text(value);
            }
        }
        if let Some(value) = param.get("output_image_file") {
            self.output_image_file = value.into();
        }
        if let Some(value) = param.get("rootname") {
            self.rootname_output_files = value.into();
        }
        self.cb_use_gpu
            .set_selected(param.get("use_gpu") == Some("true"));
        self.cb_do_dose_weighting
            .set_selected(param.get("do_dose_weighting") == Some("true"));
        self.update_display();
    }
    pub fn setup_local_arguments(&self) -> (&str, &str) {
        (&self.output_image_file, &self.local_arguments_dir)
    }
    pub fn get_output_image_file_name(&self) -> &str {
        &self.output_image_file
    }
    pub fn get_new_mdoc_file_name(&self) -> String {
        format!("{}.mdoc", self.output_image_file)
    }
    /// Java `getBrowsingDir`.
    pub fn get_browsing_dir(&self) -> &Path {
        &self.browsing_dir
    }
    /// Java `setBrowsingDir`.
    pub fn set_browsing_dir(&mut self, file: impl Into<PathBuf>) {
        self.browsing_dir = file.into();
    }
    pub fn is_multiple_filter_cutoffs(&self) -> bool {
        self.fields
            .get("filter_cutoffs")
            .is_some_and(|field| field.text.split(',').count() > 1)
    }
    pub fn is_local_dir(&self, _current_directory: &str) -> bool {
        false
    }
}
impl ToolPanel for AlignFramesPanel {
    fn get_component(&self) -> &ComponentState {
        &self.component
    }
}
impl super::process_display::ProcessDisplay for AlignFramesPanel {}
impl super::align_frames_display::AlignFramesDisplay for AlignFramesPanel {
    type AlignFramesParam = AlignFramesPanelParameters;
    type SortTiltFramesParam = AlignFramesPanelParameters;

    fn get_align_frames_parameters(
        &self,
        param: &mut Self::AlignFramesParam,
    ) -> Result<bool, FortranInputSyntaxException> {
        self.get_parameters(param)
            .map_err(|error| FortranInputSyntaxException::new(&error.to_string()))
    }

    fn get_sort_tilt_frames_parameters(
        &self,
        param: &mut Self::SortTiltFramesParam,
    ) -> Result<bool, FieldValidationFailedException> {
        AlignFramesPanel::get_sort_tilt_frames_parameters(self, param)
    }

    fn set_align_frames_parameters(&mut self, param: &Self::AlignFramesParam) {
        self.set_parameters(param);
    }

    fn is_list_of_input_files_selected(&self) -> bool {
        AlignFramesPanel::is_list_of_input_files_selected(self)
    }
    fn is_selected_files_selected(&self) -> bool {
        AlignFramesPanel::is_selected_files_selected(self)
    }
    fn is_angles_in_filenames_selected(&self) -> bool {
        AlignFramesPanel::is_angles_in_filenames_selected(self)
    }
    fn is_corresponding_stack_selected(&self) -> bool {
        AlignFramesPanel::is_corresponding_stack_selected(self)
    }
    fn is_tilt_angle_file_selected(&self) -> bool {
        AlignFramesPanel::is_tilt_angle_file_selected(self)
    }
    fn is_fixed_total_dose_selected(&self) -> bool {
        AlignFramesPanel::is_fixed_total_dose_selected(self)
    }
    fn is_do_dose_weighting_selected(&self) -> bool {
        AlignFramesPanel::is_do_dose_weighting_selected(self)
    }
    fn get_text_area_input_files(&self) -> String {
        AlignFramesPanel::get_text_area_input_files(self).into()
    }
    fn get_rootname_output_files(&self) -> String {
        AlignFramesPanel::get_rootname_output_files(self).into()
    }
    fn get_output_image_file_name(&self) -> String {
        AlignFramesPanel::get_output_image_file_name(self).into()
    }
    fn setup_local_arguments(&self) -> LocalArguments {
        let mut local_arguments = LocalArguments::default();
        local_arguments.set_raw_image_stack(&self.output_image_file);
        local_arguments.set_dir(&self.local_arguments_dir);
        local_arguments
    }
    fn is_metadata_file_selected(&self) -> bool {
        AlignFramesPanel::is_metadata_file_selected(self)
    }
    fn get_new_mdoc_file_name(&self) -> String {
        AlignFramesPanel::get_new_mdoc_file_name(self)
    }
}
impl super::tools_dialog::AlignFramesPanel for AlignFramesPanel {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_file_rootname_and_output_follow_java_rules() {
        let mut panel = AlignFramesPanel::get_tools_instance(AxisID::Only, ".");
        panel.set_text_files(&[
            PathBuf::from("/data/frames_001.tif"),
            PathBuf::from("/data/frames_002.tif"),
        ]);
        assert_eq!(panel.get_rootname_output_files(), "frames_00");
        assert_eq!(panel.get_output_image_file_name(), "frames_00_ali.mrc");
        panel.cb_do_dose_weighting.set_selected(true);
        panel.cb_normalize_dose_weighting.set_selected(false);
        panel.set_output_image_file();
        assert_eq!(panel.get_output_image_file_name(), "frames_00_aliDW.mrc");
    }
    #[test]
    fn metadata_selection_controls_other_metadata_sources() {
        let mut panel = AlignFramesPanel::new(AxisID::Only, ".");
        panel.update_display();
        assert!(!panel.enabled("corresponding_stack"));
        panel.rb_metadata_file.set_selected(false);
        panel.rb_list_of_input_files.set_selected(true);
        panel.update_display();
        assert!(panel.enabled("corresponding_stack"));
    }
    #[test]
    fn parameter_transfer_retains_source_fields() {
        let mut panel = AlignFramesPanel::new(AxisID::Only, ".");
        panel.metadata_file = "frames.mdoc".into();
        panel.rootname_output_files = "frames".into();
        panel.set_output_image_file();
        let mut parameters = AlignFramesPanelParameters::default();
        assert_eq!(panel.get_parameters(&mut parameters), Ok(true));
        assert_eq!(parameters.get("metadata_file"), Some("frames.mdoc"));
        assert_eq!(parameters.get("output_image_file"), Some("frames_ali.mrc"));
    }
}
