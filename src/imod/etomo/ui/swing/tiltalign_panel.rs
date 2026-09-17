//! `IMOD/Etomo/src/etomo/ui/swing/TiltalignPanel.java`.
//!
//! The native widget tree and `ApplicationManager` process dispatch are explicit
//! boundaries.  The panel's source-owned options, tab state, enablement rules,
//! validation, and Tiltalign parameter transfer are retained here.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::radio_button::RadioButton;
use super::radio_text_field::RadioTextField;
use super::tiltxcorr_panel::CheckTextField;

pub const MIN_LOCAL_PATCH_SIZE_LABEL: &str = "Min. local patch size or overlap factor (x,y): ";
pub const MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL: &str = "Overlap factor (x,y): ";
pub const AUTOMAPPED_OPTION: i32 = 3;
pub const SINGLE_OPTION: i32 = 1;
pub const FIXED_OPTION: i32 = 0;
pub const BEAM_SEARCH_OPTION: i32 = 1;

/// Native retained description of Java's radio-box and variable-panel builders.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltalignLayout {
    pub radio_boxes: Vec<Vec<String>>,
    pub variable_panels: Vec<String>,
    pub general_tab_created: bool,
    pub global_tab_created: bool,
    pub local_tab_created: bool,
}

/// Java `Tab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    General,
    GlobalVariables,
    LocalVariables,
}
impl Tab {
    pub fn get_instance(index: usize) -> Self {
        match index {
            1 => Self::GlobalVariables,
            2 => Self::LocalVariables,
            _ => Self::General,
        }
    }
    pub fn index(self) -> usize {
        match self {
            Self::General => 0,
            Self::GlobalVariables => 1,
            Self::LocalVariables => 2,
        }
    }
    /// Java private `Tab.getIndex()`.
    pub fn get_index(self) -> usize {
        self.index()
    }
}

/// Java `LocalAlignValidation`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalAlignValidation {
    AreaRequirements,
    Variables,
    Both,
}
impl LocalAlignValidation {
    pub fn value(self) -> i32 {
        match self {
            Self::AreaRequirements => 2,
            Self::Variables => 1,
            Self::Both => 3,
        }
    }
    pub fn get_instance(value: i32) -> Self {
        match value {
            1 => Self::Variables,
            3 => Self::Both,
            _ => Self::AreaRequirements,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::AreaRequirements => "area requirements",
            Self::Variables => "variables",
            Self::Both => "both",
        }
    }
    /// Java `EnumeratedType.isDefault()`.
    pub fn is_default(self) -> bool {
        self == Self::AreaRequirements
    }
    /// Java `EnumeratedType.getValue()`.
    pub fn get_value(self) -> i32 {
        self.value()
    }
    /// Java `toString()`.
    pub fn to_string_java(self) -> &'static str {
        self.label()
    }
    /// Java `EnumeratedType.getLabel()`.
    pub fn get_label(self) -> &'static str {
        self.label()
    }
}

/// The source-facing `TiltalignParam`, `ConstTiltalignParam`, metadata and
/// Makecomfile boundary.  The concrete parameter implementation below is useful
/// to native GUI frontends and makes every source option auditable by key.
pub trait TiltalignParameter {
    fn value(&self, key: &str) -> Option<String>;
    fn set_value(&mut self, key: &str, value: String);
    fn reset_value(&mut self, key: &str);
    fn option(&self, key: &str) -> i32;
    fn set_option(&mut self, key: &str, value: i32);
    fn flag(&self, key: &str) -> bool;
    fn set_flag(&mut self, key: &str, value: bool);
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltalignPanelParameters {
    pub values: BTreeMap<String, String>,
    pub options: BTreeMap<String, i32>,
    pub flags: BTreeMap<String, bool>,
}
impl TiltalignParameter for TiltalignPanelParameters {
    fn value(&self, key: &str) -> Option<String> {
        self.values.get(key).cloned()
    }
    fn set_value(&mut self, key: &str, value: String) {
        self.values.insert(key.into(), value);
    }
    fn reset_value(&mut self, key: &str) {
        self.values.remove(key);
    }
    fn option(&self, key: &str) -> i32 {
        self.options.get(key).copied().unwrap_or_default()
    }
    fn set_option(&mut self, key: &str, value: i32) {
        self.options.insert(key.into(), value);
    }
    fn flag(&self, key: &str) -> bool {
        self.flags.get(key).copied().unwrap_or(false)
    }
    fn set_flag(&mut self, key: &str, value: bool) {
        self.flags.insert(key.into(), value);
    }
}

/// Direct Java `ApplicationManager.restrictalign` operation.
pub trait TiltalignPanelApplicationManager {
    fn restrictalign(&mut self, axis_id: AxisID);
}

/// Complete source-visible state of Java `TiltalignPanel`.  Widget layout is
/// represented by the tab/body flags, preserving Swing as a renderer boundary.
pub struct TiltalignPanel {
    pub axis_id: AxisID,
    pub current_tab: Tab,
    pub general_body_visible: bool,
    pub global_body_visible: bool,
    pub local_body_visible: bool,
    pub local_tab_enabled: bool,
    pub patch_tracking: bool,
    pub created_day_stamp_imod_5_0_1: bool,
    pub advanced: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub last_validation_message: Option<String>,
    pub layout: TiltalignLayout,

    pub ltf_residual_threshold: LabeledTextField,
    pub rb_resid_all_views: RadioButton,
    pub rb_resid_neighboring: RadioButton,
    pub rb_single_fiducial_surface: RadioButton,
    pub rb_dual_fiducial_surfaces: RadioButton,
    pub ltf_exclude_list: LabeledTextField,
    pub ltf_separate_view_groups: LabeledTextField,
    pub ltf_tilt_angle_offset: LabeledTextField,
    pub ltf_tilt_axis_z_shift: LabeledTextField,
    pub ctf_robust_fitting_and_k_factor_scaling: CheckTextField,
    pub cb_weight_whole_tracks: CheckBox,
    pub ltf_metro_factor: LabeledTextField,
    pub ltf_maximum_cycles: LabeledTextField,
    pub cb_local_alignments: CheckBox,
    pub rtf_target_patch_size_x_and_y: RadioTextField,
    pub rtf_n_local_patches: RadioTextField,
    pub ltf_min_local_patch_size: LabeledTextField,
    pub ltf_min_local_fiducials: LabeledTextField,
    pub cb_fix_xyz_coordinates: CheckBox,
    pub rb_tilt_angle_fixed: RadioButton,
    pub rb_tilt_angle_all: RadioButton,
    pub rb_tilt_angle_automap: RadioButton,
    pub ltf_tilt_angle_group_size: LabeledTextField,
    pub ltf_tilt_angle_non_default_groups: LabeledTextField,
    pub rb_magnification_fixed: RadioButton,
    pub rb_magnification_all: RadioButton,
    pub rb_magnification_automap: RadioButton,
    pub ltf_magnification_reference_view: LabeledTextField,
    pub ltf_magnification_group_size: LabeledTextField,
    pub ltf_magnification_non_default_groups: LabeledTextField,
    pub rb_distortion_disabled: RadioButton,
    pub rb_distortion_full_solution: RadioButton,
    pub rb_distortion_skew: RadioButton,
    pub ltf_xstretch_group_size: LabeledTextField,
    pub ltf_xstretch_non_default_groups: LabeledTextField,
    pub ltf_skew_group_size: LabeledTextField,
    pub ltf_skew_non_default_groups: LabeledTextField,
    pub cb_local_rotation: CheckBox,
    pub ltf_local_rotation_group_size: LabeledTextField,
    pub ltf_local_rotation_non_default_groups: LabeledTextField,
    pub cb_local_tilt_angle: CheckBox,
    pub ltf_local_tilt_angle_group_size: LabeledTextField,
    pub ltf_local_tilt_angle_non_default_groups: LabeledTextField,
    pub cb_local_magnification: CheckBox,
    pub ltf_local_magnification_group_size: LabeledTextField,
    pub ltf_local_magnification_non_default_groups: LabeledTextField,
    pub rb_local_distortion_disabled: RadioButton,
    pub rb_local_distortion_full_solution: RadioButton,
    pub rb_local_distortion_skew: RadioButton,
    pub ltf_local_xstretch_group_size: LabeledTextField,
    pub ltf_local_xstretch_non_default_groups: LabeledTextField,
    pub ltf_local_skew_group_size: LabeledTextField,
    pub ltf_local_skew_non_default_groups: LabeledTextField,
    pub rb_rotation_none: RadioButton,
    pub rb_rotation_all: RadioButton,
    pub rb_rotation_automap: RadioButton,
    pub rb_rotation_one: RadioButton,
    pub ltf_rotation_angle: LabeledTextField,
    pub ltf_rotation_group_size: LabeledTextField,
    pub ltf_rotation_non_default_groups: LabeledTextField,
    pub cb_projection_stretch: CheckBox,
    pub rb_no_beam_tilt: RadioButton,
    pub rtf_fixed_beam_tilt: RadioTextField,
    pub rb_solve_for_beam_tilt: RadioButton,
    pub cb_x_tilt_automap_same: CheckBox,
    pub ltf_target_measurement_ratio: LabeledTextField,
    pub ltf_min_measurement_ratio: LabeledTextField,
    pub local_align_validation: LocalAlignValidation,
    pub cb_cross_validate: CheckBox,
}

impl TiltalignPanel {
    pub fn get_instance(axis_id: AxisID) -> Self {
        let mut panel = Self::new(axis_id);
        panel.add_listeners();
        panel
    }
    pub fn new(axis_id: AxisID) -> Self {
        let mut panel = Self {
            axis_id,
            current_tab: Tab::General,
            general_body_visible: true,
            global_body_visible: false,
            local_body_visible: false,
            local_tab_enabled: false,
            patch_tracking: false,
            created_day_stamp_imod_5_0_1: false,
            advanced: false,
            listener_count: 0,
            tooltip_initialized: false,
            last_validation_message: None,
            layout: TiltalignLayout::default(),
            ltf_residual_threshold: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Threshold for residual report: ",
            ),
            rb_resid_all_views: RadioButton::new("All views"),
            rb_resid_neighboring: RadioButton::new("Neighboring views"),
            rb_single_fiducial_surface: RadioButton::new(
                "Do not sort fiducials into 2 surfaces for analysis",
            ),
            rb_dual_fiducial_surfaces: RadioButton::new(
                "Assume fiducials on 2 surfaces for analysis",
            ),
            ltf_exclude_list: LabeledTextField::new(
                FieldType::IntegerList,
                "List of views to exclude: ",
            ),
            ltf_separate_view_groups: LabeledTextField::new(
                FieldType::IntegerList,
                "Separate view groups: ",
            ),
            ltf_tilt_angle_offset: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Total tilt angle offset: ",
            ),
            ltf_tilt_axis_z_shift: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Tilt axis z shift: ",
            ),
            ctf_robust_fitting_and_k_factor_scaling: CheckTextField::new(
                FieldType::FloatingPoint,
                "Do robust fitting with tuning factor:",
            ),
            cb_weight_whole_tracks: CheckBox::new_with_text(
                "Find weights for contours, not points",
            ),
            ltf_metro_factor: LabeledTextField::new(FieldType::FloatingPoint, "Metro factor: "),
            ltf_maximum_cycles: LabeledTextField::new(FieldType::Integer, "Iteration limit: "),
            cb_local_alignments: CheckBox::new_with_text("Enable local alignments"),
            rtf_target_patch_size_x_and_y: RadioTextField::new(
                FieldType::IntegerPair,
                "Target patch size (x,y): ",
            ),
            rtf_n_local_patches: RadioTextField::new(
                FieldType::IntegerPair,
                "# of local patches (x,y): ",
            ),
            ltf_min_local_patch_size: LabeledTextField::new(
                FieldType::FloatingPointPair,
                MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL,
            ),
            ltf_min_local_fiducials: LabeledTextField::new(
                FieldType::IntegerPair,
                "Min. # of fiducials (total, each surface): ",
            ),
            cb_fix_xyz_coordinates: CheckBox::new_with_text("Use global X-Y-Z coordinates"),
            rb_tilt_angle_fixed: RadioButton::new("Fixed tilt angles"),
            rb_tilt_angle_all: RadioButton::new("Solve for all except minimum tilt"),
            rb_tilt_angle_automap: RadioButton::new("Group tilt angles "),
            ltf_tilt_angle_group_size: LabeledTextField::new(FieldType::Integer, "Group size: "),
            ltf_tilt_angle_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default grouping: ",
            ),
            rb_magnification_fixed: RadioButton::new("Fixed magnification at 1.0"),
            rb_magnification_all: RadioButton::new("Solve for all magnifications"),
            rb_magnification_automap: RadioButton::new("Group magnifications"),
            ltf_magnification_reference_view: LabeledTextField::new(
                FieldType::Integer,
                "Reference view: ",
            ),
            ltf_magnification_group_size: LabeledTextField::new(FieldType::Integer, "Group size: "),
            ltf_magnification_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default grouping: ",
            ),
            rb_distortion_disabled: RadioButton::new("Disabled"),
            rb_distortion_full_solution: RadioButton::new("Full solution"),
            rb_distortion_skew: RadioButton::new("Skew only"),
            ltf_xstretch_group_size: LabeledTextField::new(
                FieldType::Integer,
                "X stretch group size: ",
            ),
            ltf_xstretch_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "X stretch non-default grouping: ",
            ),
            ltf_skew_group_size: LabeledTextField::new(FieldType::Integer, "Skew group size: "),
            ltf_skew_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Skew non-default grouping: ",
            ),
            cb_local_rotation: CheckBox::new_with_text("Enable"),
            ltf_local_rotation_group_size: LabeledTextField::new(
                FieldType::Integer,
                "Group size: ",
            ),
            ltf_local_rotation_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default grouping: ",
            ),
            cb_local_tilt_angle: CheckBox::new_with_text("Enable"),
            ltf_local_tilt_angle_group_size: LabeledTextField::new(
                FieldType::Integer,
                "Group size: ",
            ),
            ltf_local_tilt_angle_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default grouping: ",
            ),
            cb_local_magnification: CheckBox::new_with_text("Enable"),
            ltf_local_magnification_group_size: LabeledTextField::new(
                FieldType::Integer,
                "Group size: ",
            ),
            ltf_local_magnification_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default grouping: ",
            ),
            rb_local_distortion_disabled: RadioButton::new("Disabled"),
            rb_local_distortion_full_solution: RadioButton::new("Full solution"),
            rb_local_distortion_skew: RadioButton::new("Skew only"),
            ltf_local_xstretch_group_size: LabeledTextField::new(
                FieldType::Integer,
                "X stretch group size: ",
            ),
            ltf_local_xstretch_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "X stretch non-default grouping: ",
            ),
            ltf_local_skew_group_size: LabeledTextField::new(
                FieldType::Integer,
                "Skew group size: ",
            ),
            ltf_local_skew_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Skew non-default grouping: ",
            ),
            rb_rotation_none: RadioButton::new("No rotation"),
            rb_rotation_all: RadioButton::new("Solve for all rotations"),
            rb_rotation_automap: RadioButton::new("Group rotations"),
            rb_rotation_one: RadioButton::new("One rotation"),
            ltf_rotation_angle: LabeledTextField::new(FieldType::FloatingPoint, "Rotation angle: "),
            ltf_rotation_group_size: LabeledTextField::new(FieldType::Integer, "Group size: "),
            ltf_rotation_non_default_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default grouping: ",
            ),
            cb_projection_stretch: CheckBox::new_with_text(
                "Solve for single stretch during projection",
            ),
            rb_no_beam_tilt: RadioButton::new("No beam tilt"),
            rtf_fixed_beam_tilt: RadioTextField::new(
                FieldType::FloatingPoint,
                "Fixed beam tilt (degrees): ",
            ),
            rb_solve_for_beam_tilt: RadioButton::new("Solve for beam tilt"),
            cb_x_tilt_automap_same: CheckBox::new_with_text(
                "Solve for X axis tilt between separate groups",
            ),
            ltf_target_measurement_ratio: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Fallback ratio of measurements to unknowns:  Target ",
            ),
            ltf_min_measurement_ratio: LabeledTextField::new(FieldType::FloatingPoint, "Minimum "),
            local_align_validation: LocalAlignValidation::AreaRequirements,
            cb_cross_validate: CheckBox::new_with_text(
                "Compute prediction errors for points left out of test fits",
            ),
        };
        panel.set_default_parameters();
        panel.set_first_tab();
        panel.set_tool_tip_text();
        panel
    }

    pub fn add_listeners(&mut self) {
        self.listener_count = 25;
    }
    /// Native action-command form of Java's panel and nested listener `actionPerformed`.
    pub fn action_performed<M: TiltalignPanelApplicationManager>(
        &mut self,
        action_command: Option<&str>,
        manager: &mut M,
    ) {
        let Some(action_command) = action_command else {
            return;
        };
        match action_command {
            "restrictalign" => self.action_restrictalign(manager),
            "target-patch-size" | "local-patches" => self.set_min_local_patch_size_label(),
            "no-beam-tilt" | "fixed-beam-tilt" | "solve-beam-tilt" | "robust-fitting" => {
                self.update_display()
            }
            _ => {}
        }
    }
    /// Java `getUIComponent()`; the retained state is the native component.
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    /// Java `getComponent()` native root component identity.
    pub fn get_component(&self) -> &'static str {
        "tiltalign-panel-root"
    }
    /// Java overloaded `createRadioBox` helpers, represented as ordered labels.
    pub fn create_radio_box<I, S>(&mut self, items: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.layout
            .radio_boxes
            .push(items.into_iter().map(Into::into).collect());
        self.layout.radio_boxes.len() - 1
    }
    /// Java `createGeneralTab()` native retained layout.
    pub fn create_general_tab(&mut self) {
        self.layout.general_tab_created = true;
        self.general_body_visible = self.current_tab == Tab::General;
    }
    /// Java `createGlobalSolutionTab()` native retained layout.
    pub fn create_global_solution_tab(&mut self) {
        self.layout.global_tab_created = true;
        self.global_body_visible = self.current_tab == Tab::GlobalVariables;
    }
    /// Java `createLocalSolutionTab()` native retained layout.
    pub fn create_local_solution_tab(&mut self) {
        self.layout.local_tab_created = true;
        self.local_body_visible = self.current_tab == Tab::LocalVariables;
    }
    /// Java overloaded `createVariablePanel` helpers, represented by their stable title.
    pub fn create_variable_panel(&mut self, title: impl Into<String>) {
        self.layout.variable_panels.push(title.into());
    }
    pub fn change_tab(&mut self, index: usize) {
        self.update_tab(false);
        self.current_tab = Tab::get_instance(index);
        self.update_tab(true);
    }
    /// Java `TabChangeListener.stateChanged(ChangeEvent)`.
    pub fn state_changed(&mut self, selected_index: usize) {
        self.change_tab(selected_index);
    }
    pub fn update_tab(&mut self, visible: bool) {
        match self.current_tab {
            Tab::General => self.general_body_visible = visible,
            Tab::GlobalVariables => self.global_body_visible = visible,
            Tab::LocalVariables => self.local_body_visible = visible,
        }
    }
    pub fn action_restrictalign<M: TiltalignPanelApplicationManager>(&mut self, manager: &mut M) {
        manager.restrictalign(self.axis_id);
    }
    pub fn expand(&mut self, advanced: bool) {
        self.update_advanced(advanced);
    }
    pub fn update_advanced_beam_tilt(&mut self, advanced: bool) {
        self.advanced = advanced;
    }
    pub fn set_min_local_patch_size_label(&mut self) {
        if self.rtf_target_patch_size_x_and_y.is_selected() {
            self.ltf_min_local_patch_size
                .set_label(MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL);
        } else if !self.created_day_stamp_imod_5_0_1 && self.rtf_n_local_patches.is_selected() {
            self.ltf_min_local_patch_size
                .set_label(MIN_LOCAL_PATCH_SIZE_LABEL);
        }
    }

    pub fn set_parameters<P: TiltalignParameter>(&mut self, params: &P) {
        macro_rules! text {
            ($field:ident, $key:literal) => {
                self.$field
                    .set_text(&params.value($key).unwrap_or_default())
            };
        }
        macro_rules! flag {
            ($field:ident, $key:literal) => {
                self.$field.set_selected(params.flag($key))
            };
        }
        self.rb_dual_fiducial_surfaces
            .set_selected(params.option("surfaces_to_analyze") == 2);
        self.rb_single_fiducial_surface
            .set_selected(params.option("surfaces_to_analyze") != 2);
        let residual = params
            .value("residual_report_criterion")
            .unwrap_or_default()
            .parse::<f64>()
            .unwrap_or(0.0);
        self.ltf_residual_threshold
            .set_text(&residual.abs().to_string());
        self.rb_resid_neighboring.set_selected(residual < 0.0);
        self.rb_resid_all_views.set_selected(residual >= 0.0);
        flag!(cb_cross_validate, "cross_validate");
        text!(ltf_exclude_list, "exclude_list");
        self.ltf_exclude_list
            .set_enabled(params.flag("exclude_list_available"));
        text!(ltf_separate_view_groups, "separate_group");
        text!(ltf_tilt_angle_offset, "angle_offset");
        text!(ltf_tilt_axis_z_shift, "axis_z_shift");
        flag!(ctf_robust_fitting_and_k_factor_scaling, "robust_fitting");
        text!(ctf_robust_fitting_and_k_factor_scaling, "k_factor_scaling");
        flag!(cb_weight_whole_tracks, "weight_whole_tracks");
        text!(ltf_metro_factor, "metro_factor");
        text!(ltf_maximum_cycles, "maximum_cycles");
        flag!(cb_local_alignments, "local_alignments");
        flag!(cb_fix_xyz_coordinates, "fix_xyz_coordinates");
        text!(ltf_min_local_patch_size, "min_size_or_overlap_x_and_y");
        text!(ltf_min_local_fiducials, "min_fids_total_and_each_surface");
        text!(rtf_target_patch_size_x_and_y, "target_patch_size_x_and_y");
        self.rtf_target_patch_size_x_and_y
            .set_selected(params.flag("target_patch_size_x_and_y_active"));
        text!(rtf_n_local_patches, "number_of_local_patches_x_and_y");
        self.rtf_n_local_patches
            .set_selected(params.flag("number_of_local_patches_x_and_y_active"));
        self.created_day_stamp_imod_5_0_1 = params.flag("created_day_stamp_imod_5_0_1");
        self.set_min_local_patch_size_label();
        self.set_solution_buttons(params);
        self.enable_fields();
        self.update_display();
    }

    pub fn set_solution_buttons<P: TiltalignParameter>(&mut self, params: &P) {
        macro_rules! text {
            ($field:ident, $key:literal) => {
                self.$field
                    .set_text(&params.value($key).unwrap_or_default())
            };
        }
        let tilt = params.option("tilt_option");
        self.rb_tilt_angle_fixed.set_selected(tilt == 0);
        self.rb_tilt_angle_all.set_selected(tilt == 2);
        self.rb_tilt_angle_automap.set_selected(tilt == 5);
        text!(ltf_tilt_angle_group_size, "tilt_default_grouping");
        text!(ltf_tilt_angle_non_default_groups, "tilt_nondefault_group");
        let mag = params.option("mag_option");
        self.rb_magnification_fixed.set_selected(mag == 0);
        self.rb_magnification_all.set_selected(mag == 1);
        self.rb_magnification_automap
            .set_selected(mag == AUTOMAPPED_OPTION);
        text!(ltf_magnification_reference_view, "mag_reference_view");
        text!(ltf_magnification_group_size, "mag_default_grouping");
        text!(ltf_magnification_non_default_groups, "mag_nondefault_group");
        let rot = params.option("rot_option");
        self.rb_rotation_none.set_selected(rot == 0);
        self.rb_rotation_all.set_selected(rot == 1);
        self.rb_rotation_automap
            .set_selected(rot == AUTOMAPPED_OPTION);
        self.rb_rotation_one.set_selected(rot == SINGLE_OPTION);
        text!(ltf_rotation_angle, "rotation_angle");
        text!(ltf_rotation_group_size, "rot_default_grouping");
        text!(ltf_rotation_non_default_groups, "rot_nondefault_group");
        self.set_distortion_buttons(
            params.option("x_stretch_option"),
            params.option("skew_option"),
            false,
        );
        self.set_distortion_buttons(
            params.option("local_x_stretch_option"),
            params.option("local_skew_option"),
            true,
        );
        macro_rules! text_more { ($($field:ident, $key:literal),* $(,)?) => { $(text!($field,$key);)* }; }
        text_more!(
            ltf_xstretch_group_size,
            "x_stretch_default_grouping",
            ltf_xstretch_non_default_groups,
            "x_stretch_nondefault_group",
            ltf_skew_group_size,
            "skew_default_grouping",
            ltf_skew_non_default_groups,
            "skew_nondefault_group",
            ltf_local_xstretch_group_size,
            "local_x_stretch_default_grouping",
            ltf_local_xstretch_non_default_groups,
            "local_x_stretch_nondefault_group",
            ltf_local_skew_group_size,
            "local_skew_default_grouping",
            ltf_local_skew_non_default_groups,
            "local_skew_nondefault_group",
            ltf_local_rotation_group_size,
            "local_rot_default_grouping",
            ltf_local_rotation_non_default_groups,
            "local_rot_nondefault_group",
            ltf_local_tilt_angle_group_size,
            "local_tilt_default_grouping",
            ltf_local_tilt_angle_non_default_groups,
            "local_tilt_nondefault_group",
            ltf_local_magnification_group_size,
            "local_mag_default_grouping",
            ltf_local_magnification_non_default_groups,
            "local_mag_nondefault_group"
        );
        self.cb_local_rotation
            .set_selected(params.option("local_rot_option") != 0);
        self.cb_local_tilt_angle
            .set_selected(params.option("local_tilt_option") != 0);
        self.cb_local_magnification
            .set_selected(params.option("local_mag_option") != 0);
        self.cb_projection_stretch
            .set_selected(params.flag("projection_stretch"));
        self.cb_x_tilt_automap_same
            .set_selected(params.flag("x_tilt_automap_same"));
        let beam = params.option("beam_tilt_option");
        let fixed = params.value("fixed_or_initial_beam_tilt");
        self.rb_solve_for_beam_tilt
            .set_selected(beam == BEAM_SEARCH_OPTION);
        self.rtf_fixed_beam_tilt
            .set_selected(beam != BEAM_SEARCH_OPTION && fixed.is_some());
        self.rb_no_beam_tilt
            .set_selected(beam != BEAM_SEARCH_OPTION && fixed.is_none());
        if let Some(value) = fixed {
            self.rtf_fixed_beam_tilt.set_text(&value);
        }
    }
    pub fn set_distortion_buttons(&mut self, x_stretch: i32, skew: i32, local: bool) {
        let full = x_stretch == 3 && skew == 3;
        let disabled = x_stretch == 0 && skew == 0;
        if local {
            self.rb_local_distortion_disabled.set_selected(disabled);
            self.rb_local_distortion_full_solution.set_selected(full);
            self.rb_local_distortion_skew
                .set_selected(!disabled && !full);
        } else {
            self.rb_distortion_disabled.set_selected(disabled);
            self.rb_distortion_full_solution.set_selected(full);
            self.rb_distortion_skew.set_selected(!disabled && !full);
        }
    }

    pub fn get_parameters<P: TiltalignParameter>(
        &mut self,
        params: &mut P,
        do_validation: bool,
    ) -> Result<bool, FieldValidationFailedException> {
        macro_rules! set_text {
            ($field:ident, $key:literal) => {
                params.set_value($key, self.$field.get_text_validated(do_validation)?)
            };
        }
        macro_rules! set_flag {
            ($field:ident, $key:literal) => {
                params.set_flag($key, self.$field.is_selected())
            };
        }
        params.set_option(
            "surfaces_to_analyze",
            if self.rb_dual_fiducial_surfaces.is_selected() {
                2
            } else {
                1
            },
        );
        let mut residual = self
            .ltf_residual_threshold
            .get_text_validated(do_validation)?
            .parse::<f64>()
            .map_err(|err| {
                FieldValidationFailedException(format!(
                    "{} {err}",
                    self.ltf_residual_threshold.label
                ))
            })?;
        if self.rb_resid_neighboring.is_selected() {
            residual = -residual
        };
        params.set_value("residual_report_criterion", residual.to_string());
        set_flag!(cb_cross_validate, "cross_validate");
        if self.ltf_exclude_list.is_enabled() {
            set_text!(ltf_exclude_list, "exclude_list")
        };
        set_text!(ltf_separate_view_groups, "separate_group");
        set_text!(ltf_tilt_angle_offset, "angle_offset");
        set_text!(ltf_tilt_axis_z_shift, "axis_z_shift");
        set_flag!(ctf_robust_fitting_and_k_factor_scaling, "robust_fitting");
        params.set_value(
            "k_factor_scaling",
            self.ctf_robust_fitting_and_k_factor_scaling
                .get_text(do_validation)?,
        );
        if self.cb_weight_whole_tracks.is_enabled() {
            set_flag!(cb_weight_whole_tracks, "weight_whole_tracks")
        } else {
            params.reset_value("weight_whole_tracks")
        };
        set_text!(ltf_metro_factor, "metro_factor");
        set_text!(ltf_maximum_cycles, "maximum_cycles");
        set_flag!(cb_local_alignments, "local_alignments");
        params.set_flag(
            "target_patch_size_x_and_y_active",
            self.rtf_target_patch_size_x_and_y.is_selected(),
        );
        params.set_value(
            "target_patch_size_x_and_y",
            self.rtf_target_patch_size_x_and_y.get_text(do_validation)?,
        );
        params.set_flag(
            "number_of_local_patches_x_and_y_active",
            self.rtf_n_local_patches.is_selected(),
        );
        params.set_value(
            "number_of_local_patches_x_and_y",
            self.rtf_n_local_patches.get_text(do_validation)?,
        );
        set_text!(ltf_min_local_patch_size, "min_size_or_overlap_x_and_y");
        set_text!(ltf_min_local_fiducials, "min_fids_total_and_each_surface");
        set_flag!(cb_fix_xyz_coordinates, "fix_xyz_coordinates");
        params.set_option(
            "tilt_option",
            if self.rb_tilt_angle_automap.is_selected() {
                5
            } else if self.rb_tilt_angle_all.is_selected() {
                2
            } else {
                0
            },
        );
        set_text!(ltf_tilt_angle_group_size, "tilt_default_grouping");
        set_text!(ltf_tilt_angle_non_default_groups, "tilt_nondefault_group");
        params.set_option(
            "mag_option",
            if self.rb_magnification_automap.is_selected() {
                AUTOMAPPED_OPTION
            } else if self.rb_magnification_all.is_selected() {
                1
            } else {
                0
            },
        );
        set_text!(ltf_magnification_reference_view, "mag_reference_view");
        set_text!(ltf_magnification_group_size, "mag_default_grouping");
        set_text!(ltf_magnification_non_default_groups, "mag_nondefault_group");
        params.set_option(
            "rot_option",
            if self.rb_rotation_one.is_selected() {
                SINGLE_OPTION
            } else if self.rb_rotation_automap.is_selected() {
                AUTOMAPPED_OPTION
            } else if self.rb_rotation_all.is_selected() {
                1
            } else {
                0
            },
        );
        set_text!(ltf_rotation_angle, "rotation_angle");
        set_text!(ltf_rotation_group_size, "rot_default_grouping");
        set_text!(ltf_rotation_non_default_groups, "rot_nondefault_group");
        self.get_distortion_parameters(params, false, do_validation)?;
        self.get_distortion_parameters(params, true, do_validation)?;
        set_flag!(cb_projection_stretch, "projection_stretch");
        params.set_option(
            "local_rot_option",
            if self.cb_local_rotation.is_selected() {
                params.option("local_rot_option").max(5)
            } else {
                0
            },
        );
        params.set_option(
            "local_tilt_option",
            if self.cb_local_tilt_angle.is_selected() {
                params.option("local_tilt_option").max(5)
            } else {
                0
            },
        );
        params.set_option(
            "local_mag_option",
            if self.cb_local_magnification.is_selected() {
                params.option("local_mag_option").max(5)
            } else {
                0
            },
        );
        set_text!(ltf_local_rotation_group_size, "local_rot_default_grouping");
        set_text!(
            ltf_local_rotation_non_default_groups,
            "local_rot_nondefault_group"
        );
        set_text!(
            ltf_local_tilt_angle_group_size,
            "local_tilt_default_grouping"
        );
        set_text!(
            ltf_local_tilt_angle_non_default_groups,
            "local_tilt_nondefault_group"
        );
        set_text!(
            ltf_local_magnification_group_size,
            "local_mag_default_grouping"
        );
        set_text!(
            ltf_local_magnification_non_default_groups,
            "local_mag_nondefault_group"
        );
        if self.rb_no_beam_tilt.is_selected() {
            params.set_option("beam_tilt_option", FIXED_OPTION);
            params.reset_value("fixed_or_initial_beam_tilt")
        } else if self.rtf_fixed_beam_tilt.is_selected() {
            params.set_option("beam_tilt_option", FIXED_OPTION);
            params.set_value(
                "fixed_or_initial_beam_tilt",
                self.rtf_fixed_beam_tilt.get_text(do_validation)?,
            )
        } else if self.rb_solve_for_beam_tilt.is_selected() {
            params.set_option("beam_tilt_option", BEAM_SEARCH_OPTION);
            params.reset_value("fixed_or_initial_beam_tilt")
        };
        set_flag!(cb_x_tilt_automap_same, "x_tilt_automap_same");
        Ok(true)
    }
    pub fn get_distortion_parameters<P: TiltalignParameter>(
        &self,
        params: &mut P,
        local: bool,
        do_validation: bool,
    ) -> Result<(), FieldValidationFailedException> {
        let (disabled, full, xgs, xng, sgs, sng, prefix) = if local {
            (
                &self.rb_local_distortion_disabled,
                &self.rb_local_distortion_full_solution,
                &self.ltf_local_xstretch_group_size,
                &self.ltf_local_xstretch_non_default_groups,
                &self.ltf_local_skew_group_size,
                &self.ltf_local_skew_non_default_groups,
                "local_",
            )
        } else {
            (
                &self.rb_distortion_disabled,
                &self.rb_distortion_full_solution,
                &self.ltf_xstretch_group_size,
                &self.ltf_xstretch_non_default_groups,
                &self.ltf_skew_group_size,
                &self.ltf_skew_non_default_groups,
                "",
            )
        };
        params.set_option(
            &format!("{prefix}skew_option"),
            if disabled.is_selected() {
                FIXED_OPTION
            } else {
                AUTOMAPPED_OPTION
            },
        );
        params.set_option(
            &format!("{prefix}x_stretch_option"),
            if full.is_selected() {
                AUTOMAPPED_OPTION
            } else {
                FIXED_OPTION
            },
        );
        params.set_value(
            &format!("{prefix}x_stretch_default_grouping"),
            xgs.get_text_validated(do_validation)?,
        );
        params.set_value(
            &format!("{prefix}x_stretch_nondefault_group"),
            xng.get_text_validated(do_validation)?,
        );
        params.set_value(
            &format!("{prefix}skew_default_grouping"),
            sgs.get_text_validated(do_validation)?,
        );
        params.set_value(
            &format!("{prefix}skew_nondefault_group"),
            sng.get_text_validated(do_validation)?,
        );
        Ok(())
    }

    pub fn set_default_parameters(&mut self) {
        self.local_align_validation = LocalAlignValidation::AreaRequirements;
        self.rb_single_fiducial_surface.set_selected(true);
        self.rb_resid_all_views.set_selected(true);
        self.rb_tilt_angle_fixed.set_selected(true);
        self.rb_magnification_fixed.set_selected(true);
        self.rb_rotation_none.set_selected(true);
        self.rb_distortion_disabled.set_selected(true);
        self.rb_local_distortion_disabled.set_selected(true);
        self.rb_no_beam_tilt.set_selected(true);
        self.enable_fields();
    }
    pub fn set_patch_tracking(&mut self, input: bool) {
        self.patch_tracking = input;
        self.update_display();
    }
    pub fn set_surfaces_to_analyze(&mut self, surfaces: i32) {
        if surfaces == 1 {
            self.rb_single_fiducial_surface.set_selected(true);
            self.cb_weight_whole_tracks.set_selected(true)
        } else if surfaces == 2 {
            self.rb_dual_fiducial_surfaces.set_selected(true);
            self.cb_weight_whole_tracks.set_selected(false)
        }
    }
    pub fn is_valid(&mut self) -> bool {
        if self.rtf_fixed_beam_tilt.is_selected()
            && self.rtf_fixed_beam_tilt.get_text_unvalidated().is_empty()
        {
            self.last_validation_message = Some(format!(
                "{} can not be empty when it is selected.",
                self.rtf_fixed_beam_tilt.get_label()
            ));
            return false;
        }
        if self.patch_tracking && self.rb_dual_fiducial_surfaces.is_selected() {
            self.last_validation_message=Some("Patch tracking puts fiducials only on one side. Select one fiducial surface for better results.".into())
        };
        true
    }
    pub fn set_first_tab(&mut self) {
        self.change_tab(Tab::General.index());
    }
    pub fn update_advanced(&mut self, state: bool) {
        self.update_advanced_beam_tilt(state);
        for field in [
            &mut self.ltf_metro_factor,
            &mut self.ltf_maximum_cycles,
            &mut self.ltf_magnification_reference_view,
            &mut self.ltf_rotation_non_default_groups,
            &mut self.ltf_tilt_angle_non_default_groups,
            &mut self.ltf_magnification_non_default_groups,
            &mut self.ltf_xstretch_non_default_groups,
            &mut self.ltf_skew_non_default_groups,
            &mut self.ltf_local_rotation_non_default_groups,
            &mut self.ltf_local_tilt_angle_non_default_groups,
            &mut self.ltf_local_magnification_non_default_groups,
            &mut self.ltf_local_xstretch_non_default_groups,
            &mut self.ltf_local_skew_non_default_groups,
            &mut self.ltf_min_local_patch_size,
        ] {
            field.set_visible(state);
        }
        self.cb_fix_xyz_coordinates.set_visible(state);
    }
    pub fn enable_local_alignment_dependents(&mut self) {
        let state = self.cb_local_alignments.is_selected();
        self.local_tab_enabled = state;
        self.ltf_target_measurement_ratio.set_enabled(!state);
        self.ltf_min_measurement_ratio.set_enabled(!state);
        self.rtf_target_patch_size_x_and_y.set_enabled(state);
        self.rtf_n_local_patches.set_enabled(state);
        self.ltf_min_local_patch_size.set_enabled(state);
        self.ltf_min_local_fiducials.set_enabled(state);
        self.cb_fix_xyz_coordinates.set_enabled(state);
    }
    pub fn enable_fields(&mut self) {
        self.enable_local_alignment_dependents();
        self.enable_rotation_solution_fields();
        self.enable_tilt_angle_solution_fields();
        self.enable_magnification_solution_fields();
        self.enable_distortion_solution_fields();
        self.enable_local_rotation_solution_fields();
        self.enable_local_tilt_angle_solution_fields();
        self.enable_local_magnification_solution_fields();
        self.enable_local_distortion_solution_fields();
    }
    pub fn enable_tilt_angle_solution_fields(&mut self) {
        let state = self.rb_tilt_angle_automap.is_selected();
        self.ltf_tilt_angle_group_size.set_enabled(state);
        self.ltf_tilt_angle_non_default_groups.set_enabled(state);
    }
    pub fn enable_magnification_solution_fields(&mut self) {
        let state = self.rb_magnification_automap.is_selected();
        self.ltf_magnification_group_size.set_enabled(state);
        self.ltf_magnification_non_default_groups.set_enabled(state);
    }
    pub fn enable_rotation_solution_fields(&mut self) {
        let state = self.rb_rotation_automap.is_selected();
        self.ltf_rotation_group_size.set_enabled(state);
        self.ltf_rotation_non_default_groups.set_enabled(state);
        self.ltf_rotation_angle
            .set_enabled(self.rb_rotation_none.is_selected());
    }
    pub fn set_distortion_solution_state(&mut self) {
        if self.rb_distortion_disabled.is_selected() {
            self.rb_local_distortion_disabled.set_selected(true)
        } else {
            self.rb_tilt_angle_automap.set_selected(true);
            if self.rb_distortion_full_solution.is_selected() {
                self.rb_local_distortion_full_solution.set_selected(true)
            } else {
                self.rb_local_distortion_skew.set_selected(true)
            }
        }
        self.enable_fields();
    }
    pub fn enable_distortion_solution_fields(&mut self) {
        let x = self.rb_distortion_full_solution.is_selected();
        let skew = x || self.rb_distortion_skew.is_selected();
        self.ltf_xstretch_group_size.set_enabled(x);
        self.ltf_xstretch_non_default_groups.set_enabled(x);
        self.ltf_skew_group_size.set_enabled(skew);
        self.ltf_skew_non_default_groups.set_enabled(skew);
    }
    pub fn enable_local_rotation_solution_fields(&mut self) {
        let state = self.cb_local_rotation.is_selected();
        self.ltf_local_rotation_group_size.set_enabled(state);
        self.ltf_local_rotation_non_default_groups
            .set_enabled(state);
    }
    pub fn enable_local_tilt_angle_solution_fields(&mut self) {
        let state = self.cb_local_tilt_angle.is_selected();
        self.ltf_local_tilt_angle_group_size.set_enabled(state);
        self.ltf_local_tilt_angle_non_default_groups
            .set_enabled(state);
    }
    pub fn enable_local_magnification_solution_fields(&mut self) {
        let state = self.cb_local_magnification.is_selected();
        self.ltf_local_magnification_group_size.set_enabled(state);
        self.ltf_local_magnification_non_default_groups
            .set_enabled(state);
    }
    pub fn enable_local_distortion_solution_fields(&mut self) {
        let x = self.rb_local_distortion_full_solution.is_selected();
        let skew = x || self.rb_local_distortion_skew.is_selected();
        self.ltf_local_xstretch_group_size.set_enabled(x);
        self.ltf_local_xstretch_non_default_groups.set_enabled(x);
        self.ltf_local_skew_group_size.set_enabled(skew);
        self.ltf_local_skew_non_default_groups.set_enabled(skew);
    }
    pub fn update_display(&mut self) {
        let disable = (self.rb_distortion_full_solution.is_selected()
            || self.rb_distortion_skew.is_selected())
            && self.rb_solve_for_beam_tilt.is_selected();
        let all = self.rb_rotation_all.is_selected();
        let auto = self.rb_rotation_automap.is_selected();
        self.rb_rotation_all.set_enabled(all || !disable);
        self.rb_rotation_automap.set_enabled(auto || !disable);
        let disable = (all || auto) && self.rb_solve_for_beam_tilt.is_selected();
        let full = self.rb_distortion_full_solution.is_selected();
        let skew = self.rb_distortion_skew.is_selected();
        self.rb_distortion_full_solution
            .set_enabled(full || !disable);
        self.rb_distortion_skew.set_enabled(skew || !disable);
        self.rb_solve_for_beam_tilt.set_enabled(
            self.rb_solve_for_beam_tilt.is_selected() || !((all || auto) && (full || skew)),
        );
        self.cb_weight_whole_tracks.set_enabled(
            self.patch_tracking && self.ctf_robust_fitting_and_k_factor_scaling.is_selected(),
        );
    }
    pub fn set_tool_tip_text(&mut self) {
        self.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_enablement_and_residual_sign_are_preserved() {
        let mut panel = TiltalignPanel::get_instance(AxisID::Only);
        let mut p = TiltalignPanelParameters::default();
        p.values
            .insert("residual_report_criterion".into(), "-2.5".into());
        p.options.insert("surfaces_to_analyze".into(), 2);
        panel.set_parameters(&p);
        assert!(panel.rb_resid_neighboring.is_selected());
        assert!(panel.rb_dual_fiducial_surfaces.is_selected());
        let mut out = TiltalignPanelParameters::default();
        panel.get_parameters(&mut out, false).unwrap();
        assert_eq!(
            out.value("residual_report_criterion").as_deref(),
            Some("-2.5")
        );
    }
    #[test]
    fn local_alignment_controls_tab_and_patch_fields() {
        let mut panel = TiltalignPanel::get_instance(AxisID::First);
        panel.cb_local_alignments.set_selected(true);
        panel.enable_fields();
        assert!(panel.local_tab_enabled);
        // `RadioTextField.setEnabled` enables the *radio*
        // (`RadioTextField.java:437-441`), while `updateDisplay` gates the text
        // field on `enabled && radioButton.isSelected()` (`:445`).  Selecting
        // "local alignments" only enables the control; the text field stays
        // disabled until this field's own radio is chosen.  This assertion used
        // to expect the text field enabled immediately, which no state in the
        // Java produces.
        assert!(
            panel
                .rtf_target_patch_size_x_and_y
                .radio_button
                .is_enabled()
        );
        assert!(!panel.rtf_target_patch_size_x_and_y.text_field.is_enabled());
        panel.rtf_target_patch_size_x_and_y.set_selected(true);
        assert!(panel.rtf_target_patch_size_x_and_y.text_field.is_enabled());
        // And clearing local alignments disables both again (`:437-445`).
        panel.cb_local_alignments.set_selected(false);
        panel.enable_fields();
        assert!(!panel.local_tab_enabled);
        assert!(!panel.rtf_target_patch_size_x_and_y.text_field.is_enabled());
    }
    #[test]
    fn fixed_beam_tilt_requires_value() {
        let mut panel = TiltalignPanel::get_instance(AxisID::Only);
        panel.rtf_fixed_beam_tilt.set_selected(true);
        assert!(!panel.is_valid());
    }
}
