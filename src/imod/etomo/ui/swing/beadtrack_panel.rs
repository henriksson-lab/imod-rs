//! `IMOD/Etomo/src/etomo/ui/swing/BeadtrackPanel.java`.
//!
//! The concrete Swing component tree, autodoc lookup, and `ApplicationManager`
//! process calls are explicit boundaries.  This module retains the Java panel's
//! field state, visibility/enabling rules, validation ordering, and action
//! dispatch so a Rust GUI adapter has the same controller contract.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::panel_header::{BaseScreenState, ExpandButton, PanelHeader};
use super::text_efield::TextEfield;

pub const TRACK_LABEL: &str = "Track Seed Model";
pub const USE_MODEL_LABEL: &str = "Track with Fiducial Model as Seed";
const VIEW_SKIP_LIST_LABEL: &str = "View skip list";
pub const LIGHT_BEADS_LABEL: &str = "Light fiducial markers";

/// Every `BeadtrackParam` property used by this source unit.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BeadtrackField {
    SkipViews,
    AdditionalViewGroups,
    LightBeads,
    SobelFilterCentering,
    ScalableSigmaForSobel,
    LowPassCutoffInverseNm,
    ImagesAreBinned,
    FillGaps,
    TiltAngleGroups,
    MagnificationGroups,
    SearchBoxPixels,
    FiducialExtrapolationParams,
    RescueAttemptParams,
    RescueRelaxationParams,
    MeanResidChangeLimits,
    DeletionParams,
    TiltDefaultGrouping,
    MagnificationGroupSize,
    MinViewsForTiltalign,
    MaxGapSize,
    MaxBeadsToAverage,
    DistanceRescueCriterion,
    PostFitRescueResidual,
    DensityRelaxationPostFit,
    MaxRescueDistance,
    MinTiltRangeToFindAxis,
    MinTiltRangeToFindAngles,
    BeadDiameter,
    LocalAreaTracking,
    LocalAreaTargetSize,
    MinBeadsInArea,
    MinOverlapBeads,
    MaxViewsInAlign,
    RoundsOfTracking,
}

/// The translated `BeadtrackParam` boundary.  Text preserves Java's field input
/// until the real parameter object performs its source-specific conversion.
pub trait BeadtrackParam {
    fn get(&self, field: BeadtrackField) -> Option<String>;
    fn set(&mut self, field: BeadtrackField, value: String) -> Result<(), String>;
    fn validate(&self, _field: BeadtrackField, _label: &str) -> Option<String> {
        None
    }
}

/// Direct manager/UI calls made by `BeadtrackPanel.java`.
pub trait BeadtrackPanelApplicationManager<P: BeadtrackParam> {
    fn stack_binning(&self, axis_id: AxisID) -> String;
    fn fiducial_model_track(
        &mut self,
        axis_id: AxisID,
        button: &MultiLineButton,
        dialog_type: DialogType,
    );
    fn make_fiducial_model_seed_model(&mut self, axis_id: AxisID) -> bool;
    fn imod_fix_fiducials(
        &mut self,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
        button: &MultiLineButton,
        skip_list: Option<String>,
    );
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
    fn pack(&mut self, axis_id: AxisID);
}

/// Source-visible Swing layout and listener state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BeadtrackPanelLayout {
    pub root_visible: bool,
    pub body_visible: bool,
    pub expert_body_visible: bool,
    pub expert_visible: bool,
    pub fill_gaps_visible: bool,
    pub local_area_visible: bool,
    pub track_visible: bool,
    pub autofidseed_padding: bool,
    pub body_component_order: Vec<String>,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
}

/// Java `BeadtrackPanel`.
pub struct BeadtrackPanel {
    pub panel_beadtrack_x: BeadtrackPanelLayout,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub header: PanelHeader,
    pub expert_parameters_header: PanelHeader,
    pub btn_track: MultiLineButton,
    pub btn_use_model: MultiLineButton,
    pub btn_fix_model: MultiLineButton,
    pub ltf_view_skip_list: LabeledTextField,
    pub ltf_additional_view_sets: LabeledTextField,
    pub ltf_tilt_angle_group_size: LabeledTextField,
    pub ltf_tilt_angle_groups: LabeledTextField,
    pub ltf_magnification_group_size: LabeledTextField,
    pub ltf_magnification_groups: LabeledTextField,
    pub ltf_n_min_views: LabeledTextField,
    pub ltf_bead_diameter: LabeledTextField,
    pub cb_light_beads: CheckBox,
    pub cb_fill_gaps: CheckBox,
    pub ltf_max_gap: LabeledTextField,
    pub ltf_min_tilt_range_to_find_axis: LabeledTextField,
    pub ltf_min_tilt_range_to_find_angle: LabeledTextField,
    pub ltf_search_box_pixels: LabeledTextField,
    pub ltf_max_fiducials_avg: LabeledTextField,
    pub ltf_fiducial_extrapolation_params: LabeledTextField,
    pub ltf_rescue_attempt_params: LabeledTextField,
    pub ltf_min_rescue_distance: LabeledTextField,
    pub ltf_rescue_relaxtion_params: LabeledTextField,
    pub ltf_residual_distance_limit: LabeledTextField,
    pub ltf_mean_resid_change_limits: LabeledTextField,
    pub ltf_deletion_params: LabeledTextField,
    pub ltf_density_relaxation_post_fit: LabeledTextField,
    pub ltf_max_rescue_distance: LabeledTextField,
    pub cb_local_area_tracking: CheckBox,
    pub ltf_local_area_target_size: LabeledTextField,
    pub ltf_min_beads_in_area: LabeledTextField,
    pub ltf_min_overlap_beads: LabeledTextField,
    pub ltf_max_views_in_align: LabeledTextField,
    pub ltf_rounds_of_tracking: LabeledTextField,
    pub cb_sobel_filter_centering: CheckBox,
    pub ltf_scalable_sigma_for_sobel: LabeledTextField,
    pub tf_low_pass_cutoff_inverse_nm: TextEfield,
    pub autofidseed_mode: bool,
}

impl BeadtrackPanel {
    /// Java private constructor `BeadtrackPanel(...)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut result = Self {
            panel_beadtrack_x: BeadtrackPanelLayout {
                root_visible: true,
                body_visible: true,
                expert_body_visible: true,
                expert_visible: true,
                fill_gaps_visible: true,
                local_area_visible: true,
                track_visible: true,
                body_component_order: vec![
                    "viewSkipList".into(),
                    "additionalViewSets".into(),
                    "tiltAngleGroupSize".into(),
                    "tiltAngleGroups".into(),
                    "magnificationGroupSize".into(),
                    "magnificationGroups".into(),
                    "nMinViews".into(),
                    "beadDiameter".into(),
                    "lightBeads".into(),
                    "sobelFilterCentering".into(),
                    "scalableSigmaForSobel".into(),
                    "lowPassCutoffInverseNm".into(),
                    "fillGaps".into(),
                    "maxGap".into(),
                    "localAreaTracking".into(),
                    "localAreaTargetSize".into(),
                    "minBeadsInArea".into(),
                    "minOverlapBeads".into(),
                    "maxViewsInAlign".into(),
                    "roundsOfTracking".into(),
                    "expertParameters".into(),
                    "track".into(),
                    "fixModel".into(),
                    "useModel".into(),
                ],
                ..Default::default()
            },
            axis_id,
            dialog_type,
            header: PanelHeader::new(
                "Beadtracker",
                true,
                true,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            expert_parameters_header: PanelHeader::new(
                "Expert Parameters",
                true,
                false,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            btn_track: MultiLineButton::new_with_label(Some(TRACK_LABEL)),
            btn_use_model: MultiLineButton::new_with_label(Some(USE_MODEL_LABEL)),
            btn_fix_model: MultiLineButton::new_with_label(Some("Fix Fiducial Model")),
            ltf_view_skip_list: LabeledTextField::new(FieldType::IntegerList, "View skip list: "),
            ltf_additional_view_sets: LabeledTextField::new(
                FieldType::IntegerList,
                "Separate view groups: ",
            ),
            ltf_tilt_angle_group_size: LabeledTextField::new(
                FieldType::Integer,
                "Tilt angle group size: ",
            ),
            ltf_tilt_angle_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default tilt angle groups: ",
            ),
            ltf_magnification_group_size: LabeledTextField::new(
                FieldType::Integer,
                "Magnification group size: ",
            ),
            ltf_magnification_groups: LabeledTextField::new(
                FieldType::IntegerTriple,
                "Non-default magnification groups: ",
            ),
            ltf_n_min_views: LabeledTextField::new(
                FieldType::Integer,
                "Minimum # of views for tilt alignment: ",
            ),
            ltf_bead_diameter: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Unbinned bead diameter: ",
            ),
            cb_light_beads: CheckBox::new_with_text(LIGHT_BEADS_LABEL),
            cb_fill_gaps: CheckBox::new_with_text("Fill seed model gaps"),
            ltf_max_gap: LabeledTextField::new(FieldType::Integer, "Maximum gap size: "),
            ltf_min_tilt_range_to_find_axis: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Minimum tilt range for finding axis: ",
            ),
            ltf_min_tilt_range_to_find_angle: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Minimum tilt range for finding angles: ",
            ),
            ltf_search_box_pixels: LabeledTextField::new(
                FieldType::IntegerPair,
                "Search box size (pixels): ",
            ),
            ltf_max_fiducials_avg: LabeledTextField::new(
                FieldType::Integer,
                "Maximum # of views for fiducial avg.: ",
            ),
            ltf_fiducial_extrapolation_params: LabeledTextField::new(
                FieldType::IntegerPair,
                "Fiducial extrapolation limits: ",
            ),
            ltf_rescue_attempt_params: LabeledTextField::new(
                FieldType::FloatingPointPair,
                "Rescue attempt criteria: ",
            ),
            ltf_min_rescue_distance: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Distance criterion for rescue (pixels): ",
            ),
            ltf_rescue_relaxtion_params: LabeledTextField::new(
                FieldType::FloatingPointPair,
                "Rescue relaxation factors: ",
            ),
            ltf_residual_distance_limit: LabeledTextField::new(
                FieldType::FloatingPoint,
                "First pass residual limit for deletion: ",
            ),
            ltf_mean_resid_change_limits: LabeledTextField::new(
                FieldType::IntegerPair,
                "Residual change limits: ",
            ),
            ltf_deletion_params: LabeledTextField::new(
                FieldType::FloatingPointPair,
                "Deletion residual parameters: ",
            ),
            ltf_density_relaxation_post_fit: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Second pass density relaxation: ",
            ),
            ltf_max_rescue_distance: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Second pass maximum rescue distance: ",
            ),
            cb_local_area_tracking: CheckBox::new_with_text("Local tracking"),
            ltf_local_area_target_size: LabeledTextField::new(
                FieldType::Integer,
                "Local area size: ",
            ),
            ltf_min_beads_in_area: LabeledTextField::new(
                FieldType::Integer,
                "Minimum beads in area: ",
            ),
            ltf_min_overlap_beads: LabeledTextField::new(
                FieldType::Integer,
                "Minimum beads overlapping: ",
            ),
            ltf_max_views_in_align: LabeledTextField::new(
                FieldType::Integer,
                "Max. # views to include in align: ",
            ),
            ltf_rounds_of_tracking: LabeledTextField::new(
                FieldType::Integer,
                "Rounds of tracking: ",
            ),
            cb_sobel_filter_centering: CheckBox::new_with_text("Refine center with Sobel filter"),
            ltf_scalable_sigma_for_sobel: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Sobel sigma relative to bead size: ",
            ),
            tf_low_pass_cutoff_inverse_nm: TextEfield::get_labeled_instance(
                "Overall low-pass filter cutoff (/nm): ",
                FieldType::FloatingPoint,
            ),
            autofidseed_mode: false,
        };
        result.btn_track.set_action_command(Some(TRACK_LABEL));
        result
            .btn_use_model
            .set_action_command(Some(USE_MODEL_LABEL));
        result
            .btn_fix_model
            .set_action_command(Some("Fix Fiducial Model"));
        result
            .ltf_scalable_sigma_for_sobel
            .set_max_decimal_places(3);
        result.set_tool_tip_text();
        result
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut result = Self::new(axis_id, dialog_type);
        result.add_listeners();
        result
    }
    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.cb_local_area_tracking.add_action_listener();
        self.cb_sobel_filter_centering.add_action_listener();
        self.btn_track.add_action_listener();
        self.btn_use_model.add_action_listener();
        self.btn_fix_model.add_action_listener();
        self.panel_beadtrack_x.listener_count = 5;
    }
    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.panel_beadtrack_x.root_visible = visible;
    }
    /// Java `updateAutofidseed(boolean)`.
    pub fn update_autofidseed(&mut self, input: bool) {
        if input == self.autofidseed_mode {
            return;
        }
        self.autofidseed_mode = input;
        self.panel_beadtrack_x.autofidseed_padding = input;
        let visible = !input;
        self.ltf_tilt_angle_group_size.set_visible(visible);
        self.ltf_tilt_angle_groups.set_visible(visible);
        self.ltf_magnification_group_size.set_visible(visible);
        self.ltf_magnification_groups.set_visible(visible);
        self.ltf_n_min_views.set_visible(visible);
        self.ltf_bead_diameter.set_visible(visible);
        self.panel_beadtrack_x.fill_gaps_visible = visible;
        self.ltf_max_gap.set_visible(visible);
        self.panel_beadtrack_x.local_area_visible = visible;
        self.ltf_local_area_target_size.set_visible(visible);
        self.ltf_min_beads_in_area.set_visible(visible);
        self.ltf_min_overlap_beads.set_visible(visible);
        self.ltf_max_views_in_align.set_visible(visible);
        self.ltf_rounds_of_tracking.set_visible(visible);
        self.ltf_min_tilt_range_to_find_axis.set_visible(visible);
        self.ltf_min_tilt_range_to_find_angle.set_visible(visible);
        self.ltf_search_box_pixels.set_visible(visible);
        self.panel_beadtrack_x.expert_visible = visible;
        self.panel_beadtrack_x.track_visible = visible;
        self.update_advanced(self.header.is_advanced());
    }
    /// Java `expand(GlobalExpandButton)`.
    pub fn expand_global(&mut self) {}
    /// Java `expand(ExpandButton)`.
    pub fn expand<P: BeadtrackParam, M: BeadtrackPanelApplicationManager<P>>(
        &mut self,
        manager: &mut M,
        button: &ExpandButton,
    ) {
        if self.expert_parameters_header.equals_open_close(button) {
            self.panel_beadtrack_x.expert_body_visible = button.is_expanded();
        } else if self.header.equals_open_close(button) {
            self.panel_beadtrack_x.body_visible = button.is_expanded();
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        manager.pack(self.axis_id);
    }
    /// Java overloaded `setParameters(BaseScreenState)`.
    pub fn set_parameters_screen_state(&mut self, screen_state: &mut dyn BaseScreenState) {
        self.expert_parameters_header
            .set_button_states_with_default(Some(screen_state), false);
        self.header.set_button_states(Some(screen_state));
    }
    /// Java overloaded `getParameters(BaseScreenState)`.
    pub fn get_parameters_screen_state(&mut self, screen_state: &mut dyn BaseScreenState) {
        self.expert_parameters_header
            .get_button_states(Some(screen_state));
        self.header.get_button_states(Some(screen_state));
    }
    /// Java private `setEnabled()`.
    pub fn set_enabled(&mut self) {
        let local = self.cb_local_area_tracking.is_selected();
        self.ltf_local_area_target_size.set_enabled(local);
        self.ltf_min_beads_in_area.set_enabled(local);
        self.ltf_min_overlap_beads.set_enabled(local);
        self.ltf_scalable_sigma_for_sobel
            .set_enabled(self.cb_sobel_filter_centering.is_selected());
    }
    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&mut self, state: bool) {
        self.cb_light_beads.set_visible(state);
        if self.autofidseed_mode {
            return;
        }
        self.ltf_tilt_angle_group_size.set_visible(state);
        self.ltf_tilt_angle_groups.set_visible(state);
        self.ltf_magnification_group_size.set_visible(state);
        self.ltf_magnification_groups.set_visible(state);
        self.ltf_n_min_views.set_visible(state);
        self.ltf_bead_diameter.set_visible(state);
        self.ltf_max_gap.set_visible(state);
        self.ltf_min_tilt_range_to_find_axis.set_visible(state);
        self.ltf_min_tilt_range_to_find_angle.set_visible(state);
        self.ltf_search_box_pixels.set_visible(state);
        self.panel_beadtrack_x.expert_visible = state;
        self.ltf_min_beads_in_area.set_visible(state);
        self.ltf_min_overlap_beads.set_visible(state);
        self.ltf_rounds_of_tracking.set_visible(state);
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_track.remove_action_listener();
        self.btn_fix_model.remove_action_listener();
        self.panel_beadtrack_x.listener_count =
            self.panel_beadtrack_x.listener_count.saturating_sub(2);
    }
    /// Java private `setToolTipText`; autodoc lookup is a storage boundary.
    pub fn set_tool_tip_text(&mut self) {
        self.btn_track.set_tool_tip_text(Some(
            "Run Beadtrack to produce fiducial model from seed model.",
        ));
        self.btn_fix_model
            .set_tool_tip_text(Some("Load fiducial model into 3dmod."));
        self.btn_use_model.set_tool_tip_text(Some("Turn the output of Beadtrack (fiducial model) into a new seed model and then track.  Your original seed model will be moved into an _orig.seed file."));
        self.panel_beadtrack_x.tooltip_initialized = true;
    }

    /// Java `setParameters(BeadtrackParam, boolean)` at the parameter-object boundary.
    pub fn set_parameters<P: BeadtrackParam>(&mut self, params: &P, for_transfer_fid: bool) {
        let get = |field| params.get(field).unwrap_or_default();
        self.cb_light_beads
            .set_selected(get(BeadtrackField::LightBeads) == "true");
        self.cb_sobel_filter_centering
            .set_selected(get(BeadtrackField::SobelFilterCentering) == "true");
        self.ltf_scalable_sigma_for_sobel
            .set_text(&get(BeadtrackField::ScalableSigmaForSobel));
        self.tf_low_pass_cutoff_inverse_nm
            .set_text(get(BeadtrackField::LowPassCutoffInverseNm));
        if !for_transfer_fid {
            self.ltf_view_skip_list
                .set_text(&get(BeadtrackField::SkipViews));
            self.ltf_additional_view_sets
                .set_text(&get(BeadtrackField::AdditionalViewGroups));
            self.ltf_tilt_angle_group_size
                .set_text(&get(BeadtrackField::TiltDefaultGrouping));
            self.ltf_tilt_angle_groups
                .set_text(&get(BeadtrackField::TiltAngleGroups));
            self.ltf_magnification_group_size
                .set_text(&get(BeadtrackField::MagnificationGroupSize));
            self.ltf_magnification_groups
                .set_text(&get(BeadtrackField::MagnificationGroups));
            self.ltf_n_min_views
                .set_text(&get(BeadtrackField::MinViewsForTiltalign));
            self.ltf_bead_diameter
                .set_text(&get(BeadtrackField::BeadDiameter));
            self.cb_fill_gaps
                .set_selected(get(BeadtrackField::FillGaps) == "true");
            self.ltf_max_gap.set_text(&get(BeadtrackField::MaxGapSize));
            self.ltf_min_tilt_range_to_find_axis
                .set_text(&get(BeadtrackField::MinTiltRangeToFindAxis));
            self.ltf_min_tilt_range_to_find_angle
                .set_text(&get(BeadtrackField::MinTiltRangeToFindAngles));
            self.ltf_search_box_pixels
                .set_text(&get(BeadtrackField::SearchBoxPixels));
            self.ltf_max_fiducials_avg
                .set_text(&get(BeadtrackField::MaxBeadsToAverage));
            self.ltf_fiducial_extrapolation_params
                .set_text(&get(BeadtrackField::FiducialExtrapolationParams));
            self.ltf_rescue_attempt_params
                .set_text(&get(BeadtrackField::RescueAttemptParams));
            self.ltf_min_rescue_distance
                .set_text(&get(BeadtrackField::DistanceRescueCriterion));
            self.ltf_rescue_relaxtion_params
                .set_text(&get(BeadtrackField::RescueRelaxationParams));
            self.ltf_residual_distance_limit
                .set_text(&get(BeadtrackField::PostFitRescueResidual));
            self.ltf_density_relaxation_post_fit
                .set_text(&get(BeadtrackField::DensityRelaxationPostFit));
            self.ltf_max_rescue_distance
                .set_text(&get(BeadtrackField::MaxRescueDistance));
            self.ltf_mean_resid_change_limits
                .set_text(&get(BeadtrackField::MeanResidChangeLimits));
            self.ltf_deletion_params
                .set_text(&get(BeadtrackField::DeletionParams));
            self.cb_local_area_tracking
                .set_selected(get(BeadtrackField::LocalAreaTracking) == "true");
            self.ltf_local_area_target_size
                .set_text(&get(BeadtrackField::LocalAreaTargetSize));
            self.ltf_min_beads_in_area
                .set_text(&get(BeadtrackField::MinBeadsInArea));
            self.ltf_min_overlap_beads
                .set_text(&get(BeadtrackField::MinOverlapBeads));
            self.ltf_max_views_in_align
                .set_text(&get(BeadtrackField::MaxViewsInAlign));
            self.ltf_rounds_of_tracking
                .set_text(&get(BeadtrackField::RoundsOfTracking));
        }
        self.set_enabled();
    }

    /// Java `getParameters(BeadtrackParam, boolean)`.
    pub fn get_parameters<P: BeadtrackParam, M: BeadtrackPanelApplicationManager<P>>(
        &self,
        manager: &mut M,
        params: &mut P,
        do_validation: bool,
    ) -> Result<bool, String> {
        let text = |field: &LabeledTextField| {
            field
                .get_text_validated(do_validation)
                .map_err(|e: FieldValidationFailedException| e.0)
        };
        let mut values = Vec::new();
        values.push((BeadtrackField::SkipViews, text(&self.ltf_view_skip_list)?));
        values.push((
            BeadtrackField::AdditionalViewGroups,
            text(&self.ltf_additional_view_sets)?,
        ));
        values.push((
            BeadtrackField::LightBeads,
            self.cb_light_beads.is_selected().to_string(),
        ));
        values.push((
            BeadtrackField::SobelFilterCentering,
            self.cb_sobel_filter_centering.is_selected().to_string(),
        ));
        values.push((
            BeadtrackField::ScalableSigmaForSobel,
            text(&self.ltf_scalable_sigma_for_sobel)?,
        ));
        values.push((
            BeadtrackField::LowPassCutoffInverseNm,
            self.tf_low_pass_cutoff_inverse_nm.get_text(),
        ));
        values.push((
            BeadtrackField::ImagesAreBinned,
            manager.stack_binning(self.axis_id),
        ));
        values.push((
            BeadtrackField::FillGaps,
            self.cb_fill_gaps.is_selected().to_string(),
        ));
        for (kind, field) in [
            (BeadtrackField::TiltAngleGroups, &self.ltf_tilt_angle_groups),
            (
                BeadtrackField::MagnificationGroups,
                &self.ltf_magnification_groups,
            ),
            (BeadtrackField::SearchBoxPixels, &self.ltf_search_box_pixels),
            (
                BeadtrackField::FiducialExtrapolationParams,
                &self.ltf_fiducial_extrapolation_params,
            ),
            (
                BeadtrackField::RescueAttemptParams,
                &self.ltf_rescue_attempt_params,
            ),
            (
                BeadtrackField::RescueRelaxationParams,
                &self.ltf_rescue_relaxtion_params,
            ),
            (
                BeadtrackField::MeanResidChangeLimits,
                &self.ltf_mean_resid_change_limits,
            ),
            (BeadtrackField::DeletionParams, &self.ltf_deletion_params),
        ] {
            values.push((kind, text(field)?));
        }
        for (kind, field) in [
            (
                BeadtrackField::TiltDefaultGrouping,
                &self.ltf_tilt_angle_group_size,
            ),
            (
                BeadtrackField::MagnificationGroupSize,
                &self.ltf_magnification_group_size,
            ),
            (BeadtrackField::MinViewsForTiltalign, &self.ltf_n_min_views),
            (BeadtrackField::MaxGapSize, &self.ltf_max_gap),
            (
                BeadtrackField::MaxBeadsToAverage,
                &self.ltf_max_fiducials_avg,
            ),
            (
                BeadtrackField::DistanceRescueCriterion,
                &self.ltf_min_rescue_distance,
            ),
            (
                BeadtrackField::PostFitRescueResidual,
                &self.ltf_residual_distance_limit,
            ),
            (
                BeadtrackField::DensityRelaxationPostFit,
                &self.ltf_density_relaxation_post_fit,
            ),
            (
                BeadtrackField::MaxRescueDistance,
                &self.ltf_max_rescue_distance,
            ),
            (
                BeadtrackField::MinTiltRangeToFindAxis,
                &self.ltf_min_tilt_range_to_find_axis,
            ),
            (
                BeadtrackField::MinTiltRangeToFindAngles,
                &self.ltf_min_tilt_range_to_find_angle,
            ),
            (BeadtrackField::BeadDiameter, &self.ltf_bead_diameter),
            (
                BeadtrackField::LocalAreaTargetSize,
                &self.ltf_local_area_target_size,
            ),
            (BeadtrackField::MinBeadsInArea, &self.ltf_min_beads_in_area),
            (BeadtrackField::MinOverlapBeads, &self.ltf_min_overlap_beads),
            (
                BeadtrackField::MaxViewsInAlign,
                &self.ltf_max_views_in_align,
            ),
            (
                BeadtrackField::RoundsOfTracking,
                &self.ltf_rounds_of_tracking,
            ),
        ] {
            values.push((kind, text(field)?));
        }
        values.push((
            BeadtrackField::LocalAreaTracking,
            self.cb_local_area_tracking.is_selected().to_string(),
        ));
        for (kind, value) in values {
            params.set(kind, value)?;
            if let Some(message) = params.validate(kind, &format!("{kind:?}")) {
                manager.open_message_dialog(message.clone(), "FieldInterface Error", self.axis_id);
                return Err(message);
            }
        }
        Ok(true)
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<P: BeadtrackParam, M: BeadtrackPanelApplicationManager<P>>(
        &mut self,
        manager: &mut M,
        command: &str,
        options: Option<Run3dmodMenuOptions>,
    ) -> Result<(), String> {
        if Some(command) == self.btn_track.get_action_command() {
            manager.fiducial_model_track(self.axis_id, &self.btn_track, self.dialog_type);
        } else if Some(command) == self.btn_use_model.get_action_command() {
            if manager.make_fiducial_model_seed_model(self.axis_id) {
                manager.fiducial_model_track(self.axis_id, &self.btn_use_model, self.dialog_type);
            }
        } else if Some(command) == self.cb_local_area_tracking.get_text()
            || Some(command) == self.cb_sobel_filter_centering.get_text()
        {
            self.set_enabled();
        } else if Some(command) == self.btn_fix_model.get_action_command() {
            let skip = self
                .ltf_view_skip_list
                .get_text_validated(true)
                .map_err(|e| e.0)?
                .trim()
                .to_owned();
            if skip.chars().any(char::is_whitespace) {
                manager.open_message_dialog(
                    format!("{VIEW_SKIP_LIST_LABEL} cannot contain embedded spaces."),
                    "Entry Error",
                    self.axis_id,
                );
                return Ok(());
            }
            manager.imod_fix_fiducials(
                self.axis_id,
                options,
                &self.btn_fix_model,
                (!skip.is_empty()).then_some(skip),
            );
        }
        Ok(())
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<P: BeadtrackParam, M: BeadtrackPanelApplicationManager<P>>(
        &mut self,
        manager: &mut M,
        command: &str,
    ) -> Result<(), String> {
        self.action(manager, command, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Param(BTreeMap<BeadtrackField, String>);
    impl BeadtrackParam for Param {
        fn get(&self, field: BeadtrackField) -> Option<String> {
            self.0.get(&field).cloned()
        }
        fn set(&mut self, field: BeadtrackField, value: String) -> Result<(), String> {
            self.0.insert(field, value);
            Ok(())
        }
    }
    #[derive(Default)]
    struct Manager {
        calls: Vec<String>,
        messages: Vec<String>,
    }
    impl BeadtrackPanelApplicationManager<Param> for Manager {
        fn stack_binning(&self, _: AxisID) -> String {
            "2".into()
        }
        fn fiducial_model_track(&mut self, _: AxisID, b: &MultiLineButton, _: DialogType) {
            self.calls.push(b.get_action_command().unwrap().into());
        }
        fn make_fiducial_model_seed_model(&mut self, _: AxisID) -> bool {
            true
        }
        fn imod_fix_fiducials(
            &mut self,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: &MultiLineButton,
            skip: Option<String>,
        ) {
            self.calls.push(format!("fix:{skip:?}"));
        }
        fn open_message_dialog(&mut self, m: String, _: &str, _: AxisID) {
            self.messages.push(m)
        }
        fn pack(&mut self, _: AxisID) {}
    }
    #[test]
    fn autofidseed_and_advanced_follow_source_visibility() {
        let mut panel = BeadtrackPanel::get_instance(AxisID::Only, DialogType::FiducialModel);
        panel.update_autofidseed(true);
        assert!(!panel.ltf_bead_diameter.is_visible());
        assert!(!panel.panel_beadtrack_x.track_visible);
        panel.update_autofidseed(false);
        panel.update_advanced(true);
        assert!(panel.ltf_bead_diameter.is_visible());
    }
    #[test]
    fn action_rejects_embedded_skip_spaces_and_tracks() {
        let mut panel = BeadtrackPanel::get_instance(AxisID::Only, DialogType::FiducialModel);
        let mut manager = Manager::default();
        panel.ltf_view_skip_list.set_text("1 2");
        panel
            .action::<Param, _>(&mut manager, "Fix Fiducial Model", None)
            .unwrap();
        assert_eq!(manager.messages.len(), 1);
        panel
            .action::<Param, _>(&mut manager, TRACK_LABEL, None)
            .unwrap();
        assert_eq!(manager.calls, vec![TRACK_LABEL]);
    }
    #[test]
    fn parameters_retain_manager_binning_and_local_switch() {
        let mut panel = BeadtrackPanel::get_instance(AxisID::Only, DialogType::FiducialModel);
        let mut manager = Manager::default();
        panel.cb_local_area_tracking.set_selected(true);
        panel.set_enabled();
        let mut param = Param::default();
        assert!(
            panel
                .get_parameters(&mut manager, &mut param, true)
                .unwrap()
        );
        assert_eq!(
            param.get(BeadtrackField::ImagesAreBinned).as_deref(),
            Some("2")
        );
        assert!(panel.ltf_local_area_target_size.is_enabled());
    }
}
