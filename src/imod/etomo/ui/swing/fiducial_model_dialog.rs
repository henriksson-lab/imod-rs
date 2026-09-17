//! `IMOD/Etomo/src/etomo/ui/swing/FiducialModelDialog.java`.
//!
//! Swing hierarchy construction, `ApplicationManager`, process-result factory,
//! autodoc, and filesystem queries remain direct boundaries.  This source unit
//! owns the method/seed-model/tab switching, all dialog field state, enablement,
//! parameter transfer ordering, popup selection, and command dispatch.
#![allow(dead_code)]

use super::{
    beadtrack_panel::BeadtrackPanel,
    check_box::CheckBox,
    check_box_spinner::CheckBoxSpinner,
    context_popup::{ContextPopup, MouseEvent},
    labeled_text_field::LabeledTextField,
    multi_line_button::MultiLineButton,
    process_dialog::{ProcessDialog, ProcessDialogApplicationManager},
    radio_button::RadioButton,
    radio_text_field::RadioTextField,
    raptor_panel::RaptorPanel,
    run_3dmod_button::Run3dmodButton,
    tiltxcorr_panel::TiltxcorrPanel,
    transferfid_panel::TransferfidPanel,
};
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{axis_id::AxisID, axis_type::AxisType, dialog_type::DialogType},
    ui::field_type::FieldType,
};

pub const SEEDING_NOT_DONE_LABEL: &str = "Seed Fiducial Model";
pub const AUTOFIDSEED_NEW_MODEL_LABEL: &str = "Generate Seed Model";
const SEEDING_DONE_LABEL: &str = "View Seed Model";
const AUTOFIDSEED_APPEND_LABEL: &str = "Add Points to Seed Model";
const AUTOFIDSEED_NEW_MODEL_TITLE: &str = "Generate seed model automatically";
const AUTOFIDSEED_APPEND_TITLE: &str = "Add points to seed model automatically";

/// Java `TrackingMethod` selection stored by `bgMethod`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackingMethod {
    Seed,
    PatchTracking,
    Raptor,
}
impl TrackingMethod {
    fn index(self) -> usize {
        match self {
            Self::Seed => 0,
            Self::PatchTracking => 1,
            Self::Raptor => 2,
        }
    }
}

/// Java nested `SeedModelEnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedModelEnumeratedType {
    Manual,
    Auto,
    Transfer,
}
impl SeedModelEnumeratedType {
    fn index(self) -> usize {
        match self {
            Self::Manual => 0,
            Self::Auto => 1,
            Self::Transfer => 2,
        }
    }
    pub fn is_default(self) -> bool {
        self == Self::Manual
    }
    pub fn get_label(self) -> Option<&'static str> {
        None
    }
    pub fn get_value(self) -> i32 {
        self.index() as i32
    }
    pub fn get_instance(value: Option<&str>) -> Option<Self> {
        match value {
            Some("Manual") => Some(Self::Manual),
            Some("Auto") => Some(Self::Auto),
            Some("Transfer") => Some(Self::Transfer),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "Manual",
            Self::Auto => "Auto",
            Self::Transfer => "Transfer",
        }
    }
}

/// Java nested `SeedAndTrackTab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedAndTrackTab {
    Seed,
    Track,
}
impl SeedAndTrackTab {
    fn index(self) -> usize {
        match self {
            Self::Seed => 0,
            Self::Track => 1,
        }
    }
    fn get_instance(index: i32) -> Self {
        if index == 1 { Self::Track } else { Self::Seed }
    }
    fn title(self) -> &'static str {
        match self {
            Self::Seed => "Seed Model",
            Self::Track => "Track Beads",
        }
    }
}

/// Java nested `RunRaptorTab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunRaptorTab {
    Raptor,
    Track,
}
impl RunRaptorTab {
    fn index(self) -> usize {
        match self {
            Self::Raptor => 0,
            Self::Track => 1,
        }
    }
    fn get_instance(index: i32) -> Self {
        if index == 1 {
            Self::Track
        } else {
            Self::Raptor
        }
    }
    fn title(self) -> &'static str {
        match self {
            Self::Raptor => "Run RAPTOR",
            Self::Track => "Track Beads",
        }
    }
}

/// Direct `ApplicationManager` calls made by `FiducialModelDialog.java`.
pub trait FiducialModelDialogApplicationManager: ProcessDialogApplicationManager {
    fn is_dual_axis(&self) -> bool;
    fn is_seeding_done(&self, axis_id: AxisID) -> bool;
    fn autofidseed_initial_model_exists(&self, axis_id: AxisID) -> bool;
    fn sorted_models_exist(&self, axis_id: AxisID) -> bool;
    fn clustered_elongated_model_exists(&self, axis_id: AxisID) -> bool;
    fn adjusted_track_com_exists(&self, axis_id: AxisID) -> bool;
    fn tracking_adjusted(&self, axis_id: AxisID) -> Result<bool, String>;
    fn pack(&mut self, axis_id: AxisID);
    fn move_sub_frame(&mut self);
    fn imod_seed_model(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn autofidseed(&mut self, axis_id: AxisID, just_find_shifts_near_zero: bool);
    fn imod_initial_bead_model(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn imod_sorted_models(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn imod_clustered_elongated_model(
        &mut self,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
    );
    fn cleanup_autofidseed(&mut self, axis_id: AxisID);
    fn imod_boundary_model(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn use_track_adjusted_comfile(&mut self, axis_id: AxisID);
    fn done_fiducial_model_dialog(&mut self, axis_id: AxisID);
}

/// Concrete GUI parameter boundary for the source's `AutofidseedParam` reads/writes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AutofidseedFields {
    pub just_find_shifts_near_zero: Option<String>,
    pub min_guess_num_beads: String,
    pub min_spacing: String,
    pub peak_storage_fraction: String,
    pub boundary_model: bool,
    pub exclude_inside_areas: bool,
    pub adjust_sizes: bool,
    pub borders_in_x_and_y: String,
    pub two_surfaces: bool,
    pub append_to_seed_model: bool,
    pub target_number: Option<String>,
    pub target_density: Option<String>,
    pub max_major_to_minor_ratio: String,
    pub clustered_points_allowed: bool,
    pub elongated_points_allowed: Option<i32>,
    pub lower_target_for_clustered: Option<String>,
    pub ignore_surface_data: String,
    pub drop_tracks: String,
}

/// The `MetaData`/`ConstMetaData` portion directly read and written by this
/// dialog.  The subordinate Raptor and Tiltxcorr calls stay at their own
/// translated parameter boundaries.
pub trait FiducialModelMetaData:
    super::raptor_panel::RaptorMetaData + super::tiltxcorr_panel::TiltxcorrMetaData
{
    fn track_method(&self, axis: AxisID) -> Option<String>;
    fn set_track_method(&mut self, axis: AxisID, value: String);
    fn seed_model_manual(&self, axis: AxisID) -> bool;
    fn seed_model_auto(&self, axis: AxisID) -> bool;
    fn seed_model_transfer(&self, axis: AxisID) -> bool;
    fn set_seed_model_manual(&mut self, axis: AxisID, value: bool);
    fn set_seed_model_auto(&mut self, axis: AxisID, value: bool);
    fn set_seed_model_transfer(&mut self, axis: AxisID, value: bool);
    fn exclude_inside_areas(&self, axis: AxisID) -> bool;
    fn set_exclude_inside_areas(&mut self, axis: AxisID, value: bool);
    fn just_find_shifts_near_zero(&self, axis: AxisID) -> String;
    fn set_just_find_shifts_near_zero(&mut self, axis: AxisID, value: String);
    fn target_number_of_beads(&self, axis: AxisID) -> String;
    fn set_target_number_of_beads(&mut self, axis: AxisID, value: String);
    fn target_density_of_beads(&self, axis: AxisID) -> String;
    fn set_target_density_of_beads(&mut self, axis: AxisID, value: String);
    fn elongated_points_allowed(&self, axis: AxisID) -> Option<i32>;
    fn set_elongated_points_allowed(&mut self, axis: AxisID, value: i32);
    fn lower_target_for_clustered(&self, axis: AxisID) -> String;
    fn set_lower_target_for_clustered(&mut self, axis: AxisID, value: String);
    fn advanced(&self, axis: AxisID) -> bool;
    fn set_advanced(&mut self, axis: AxisID, value: bool);
    fn seed_and_track_tab(&self, axis: AxisID) -> i32;
    fn set_seed_and_track_tab(&mut self, axis: AxisID, value: i32);
    fn raptor_tab(&self, axis: AxisID) -> i32;
    fn set_raptor_tab(&mut self, axis: AxisID, value: i32);
}

/// The Java panel layout exposed to the native GUI adapter, in exact high-level order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiducialModelDialogLayout {
    pub root_box_layout_y_axis: bool,
    pub root_border_title: &'static str,
    pub method_panel_items: Vec<&'static str>,
    pub seed_tabs: Vec<&'static str>,
    pub raptor_tabs: Vec<&'static str>,
    pub seed_model_items: Vec<&'static str>,
    pub mouse_listener_count: usize,
    pub action_listener_count: usize,
}

/// Java final `FiducialModelDialog` fields.
pub struct FiducialModelDialog<'a> {
    pub process_dialog: ProcessDialog<'a>,
    pub axis_type: AxisType,
    pub layout: FiducialModelDialogLayout,
    pub rb_method_seed: RadioButton,
    pub rb_method_patch_tracking: RadioButton,
    pub rb_method_raptor: RadioButton,
    pub rb_seed_model_manual: RadioButton,
    pub rb_seed_model_auto: RadioButton,
    pub rb_seed_model_transfer: RadioButton,
    pub cb_boundary_model: CheckBox,
    pub cb_exclude_inside_areas: CheckBox,
    pub cb_adjust_sizes: CheckBox,
    pub cb_two_surfaces: CheckBox,
    pub cb_append_to_seed_model: CheckBox,
    pub cb_clustered_points_allowed_clustered: CheckBox,
    pub cbs_elongated_points_allowed: CheckBoxSpinner,
    pub ltf_borders_in_x_and_y: LabeledTextField,
    pub ltf_min_guess_num_beads: LabeledTextField,
    pub ltf_min_spacing: LabeledTextField,
    pub ltf_peak_storage_fraction: LabeledTextField,
    pub ltf_ignore_surface_data: LabeledTextField,
    pub ltf_drop_tracks: LabeledTextField,
    pub ltf_max_major_to_minor_ratio: LabeledTextField,
    pub ltf_lower_target_for_clustered: LabeledTextField,
    pub ltf_just_find_shifts_near_zero: LabeledTextField,
    pub rtf_target_number_of_beads: RadioTextField,
    pub rtf_target_density_of_beads: RadioTextField,
    pub btn_boundary_model: Run3dmodButton,
    pub btn_3dmod_autofidseed: Run3dmodButton,
    pub btn_3dmod_initial_bead_finding: Run3dmodButton,
    pub btn_3dmod_bead_selection_and_sorting: Run3dmodButton,
    pub btn_3dmod_clustered_elongated_model: Run3dmodButton,
    pub btn_seed: Run3dmodButton,
    pub btn_autofidseed: Run3dmodButton,
    pub btn_cleanup: MultiLineButton,
    pub btn_use_adjusted_track_com: MultiLineButton,
    pub btn_just_find_shifts_near_zero: MultiLineButton,
    pub pnl_beadtrack: BeadtrackPanel,
    pub pnl_transferfid: Option<TransferfidPanel>,
    pub tiltxcorr_panel: TiltxcorrPanel,
    pub raptor_panel: Option<RaptorPanel>,
    pub transferfid_enabled: bool,
    pub cur_method: TrackingMethod,
    pub cur_seed_and_track_tab: SeedAndTrackTab,
    pub cur_seed_model: SeedModelEnumeratedType,
    pub cur_run_raptor_tab: RunRaptorTab,
    pub context_popup: Option<ContextPopup>,
}

impl<'a> FiducialModelDialog<'a> {
    /// Java static `getInstance(ApplicationManager,AxisID,AxisType)`.
    pub fn get_instance<M: FiducialModelDialogApplicationManager>(
        manager: &'a M,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> Self {
        let mut process_dialog =
            ProcessDialog::new(manager, axis_id, DialogType::FiducialModel, Box::new(|| {}));
        process_dialog.add_exit_buttons();
        process_dialog.btn_execute.set_text("Done");
        let mut seed = Run3dmodButton::get_3dmod_instance(SEEDING_NOT_DONE_LABEL, true);
        let mut auto = Run3dmodButton::get_3dmod_instance(AUTOFIDSEED_NEW_MODEL_LABEL, true);
        seed.add_action_listener();
        auto.add_action_listener();
        let mut just = MultiLineButton::new();
        just.set_text("Just Find Shifts Near Zero");
        just.set_enabled(false);
        let mut adjusted = MultiLineButton::new();
        adjusted.set_text("Use Adjusted Track Com");
        adjusted.set_enabled(false);
        let raptor_panel = (axis_type != AxisType::DualAxis || axis_id != AxisID::Second)
            .then(|| RaptorPanel::get_instance(axis_id, DialogType::FiducialModel));
        let transfer = manager
            .is_dual_axis()
            .then(|| TransferfidPanel::get_instance(axis_id, DialogType::FiducialModel));
        let mut value = Self {
            process_dialog,
            axis_type,
            layout: FiducialModelDialogLayout {
                root_box_layout_y_axis: true,
                root_border_title: "Fiducial Model Generation",
                method_panel_items: vec![
                    "Make seed and track",
                    "Use patch tracking to make fiducial model",
                    "Run RAPTOR and fix",
                ],
                seed_tabs: vec![
                    SeedAndTrackTab::Seed.title(),
                    SeedAndTrackTab::Track.title(),
                ],
                raptor_tabs: vec![RunRaptorTab::Raptor.title(), RunRaptorTab::Track.title()],
                seed_model_items: vec![
                    "Make seed model manually",
                    AUTOFIDSEED_NEW_MODEL_TITLE,
                    "Transfer seed model from the other axis",
                ],
                mouse_listener_count: 9,
                action_listener_count: 19,
            },
            rb_method_seed: RadioButton::new("Make seed and track"),
            rb_method_patch_tracking: RadioButton::new("Use patch tracking to make fiducial model"),
            rb_method_raptor: RadioButton::new("Run RAPTOR and fix"),
            rb_seed_model_manual: RadioButton::new("Make seed model manually"),
            rb_seed_model_auto: RadioButton::new(AUTOFIDSEED_NEW_MODEL_TITLE),
            rb_seed_model_transfer: RadioButton::new("Transfer seed model from the other axis"),
            cb_boundary_model: CheckBox::new_with_text("Use boundary model"),
            cb_exclude_inside_areas: CheckBox::new_with_text("Exclude inside boundary contours"),
            cb_adjust_sizes: CheckBox::new_with_text("Find and adjust bead size"),
            cb_two_surfaces: CheckBox::new_with_text("Select beads on two surfaces"),
            cb_append_to_seed_model: CheckBox::new_with_text("Add beads to existing model"),
            cb_clustered_points_allowed_clustered: CheckBox::new_with_text("Allow clustered beads"),
            cbs_elongated_points_allowed: CheckBoxSpinner::get_instance_with_value(
                "Allow elongated beads of severity: ",
                1,
                1,
                3,
            ),
            ltf_borders_in_x_and_y: LabeledTextField::new(
                FieldType::IntegerPair,
                "Borders in X & Y: ",
            ),
            ltf_min_guess_num_beads: LabeledTextField::new(
                FieldType::Integer,
                "Estimated number of beads in sample: ",
            ),
            ltf_min_spacing: LabeledTextField::new(FieldType::FloatingPoint, "Minimum spacing: "),
            ltf_peak_storage_fraction: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Fraction of peaks to store: ",
            ),
            ltf_ignore_surface_data: LabeledTextField::new(
                FieldType::IntegerList,
                "Ignore sorting in tracked models: ",
            ),
            ltf_drop_tracks: LabeledTextField::new(FieldType::IntegerList, "Drop tracked models: "),
            ltf_max_major_to_minor_ratio: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Maximum ratio between surfaces: ",
            ),
            ltf_lower_target_for_clustered: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Lower target number for allowing ",
            ),
            ltf_just_find_shifts_near_zero: LabeledTextField::new(
                FieldType::Integer,
                "Estimated number of beads in sample",
            ),
            rtf_target_number_of_beads: RadioTextField::new(FieldType::Integer, "Total number:"),
            rtf_target_density_of_beads: RadioTextField::new(
                FieldType::FloatingPoint,
                "Density (per megapixel):",
            ),
            btn_boundary_model: Run3dmodButton::get_3dmod_instance(
                "Create/Edit Boundary Model",
                true,
            ),
            btn_3dmod_autofidseed: Run3dmodButton::get_3dmod_instance("Open Seed Model", true),
            btn_3dmod_initial_bead_finding: Run3dmodButton::get_3dmod_instance(
                "Open Initial Bead Model",
                true,
            ),
            btn_3dmod_bead_selection_and_sorting: Run3dmodButton::get_3dmod_instance(
                "Open Sorted 3D Models",
                true,
            ),
            btn_3dmod_clustered_elongated_model: Run3dmodButton::get_3dmod_instance(
                "Open Clustered / Elongated Model",
                true,
            ),
            btn_seed: seed,
            btn_autofidseed: auto,
            btn_cleanup: MultiLineButton::new(),
            btn_use_adjusted_track_com: adjusted,
            btn_just_find_shifts_near_zero: just,
            pnl_beadtrack: BeadtrackPanel::get_instance(axis_id, DialogType::FiducialModel),
            pnl_transferfid: transfer,
            tiltxcorr_panel: TiltxcorrPanel::get_patch_tracking_instance(
                axis_id,
                DialogType::FiducialModel,
            ),
            raptor_panel,
            transferfid_enabled: false,
            cur_method: TrackingMethod::Seed,
            cur_seed_and_track_tab: SeedAndTrackTab::Seed,
            cur_seed_model: SeedModelEnumeratedType::Manual,
            cur_run_raptor_tab: RunRaptorTab::Raptor,
            context_popup: None,
        };
        value.rb_method_seed.set_selected(true);
        value.rb_seed_model_manual.set_selected(true);
        value.rtf_target_number_of_beads.set_selected(true);
        value.ltf_just_find_shifts_near_zero.required = true;
        value.ltf_just_find_shifts_near_zero.columns = 3;
        value.create_panel();
        value.set_tool_tip_text();
        value.add_listeners();
        value
    }

    /// Java private `createPanel()`; component-tree attachment remains native GUI state in `layout`.
    fn create_panel(&mut self) {
        self.update_advanced(self.process_dialog.btn_advanced.is_expanded());
        self.update_display_without_manager();
    }
    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        self.btn_cleanup.add_action_listener();
        self.btn_use_adjusted_track_com.add_action_listener();
        self.btn_just_find_shifts_near_zero.add_action_listener();
    }
    /// Java static `getUseRaptorResultLabel()`.
    pub fn get_use_raptor_result_label() -> &'static str {
        super::raptor_panel::USE_RAPTOR_RESULT_LABEL
    }
    /// Java private `updateMethod()`.
    pub fn update_method<M: FiducialModelDialogApplicationManager>(&mut self, manager: &mut M) {
        if self.cur_method == TrackingMethod::Seed {
            self.change_seed_and_track_tab(manager);
        } else if self.cur_method == TrackingMethod::Raptor {
            self.change_run_raptor_tab(manager);
        } else {
            manager.pack(self.process_dialog.axis_id);
        }
    }
    /// Java private `changeSeedAndTrackTab()`.
    pub fn change_seed_and_track_tab<M: FiducialModelDialogApplicationManager>(
        &mut self,
        manager: &mut M,
    ) {
        if self.cur_seed_and_track_tab == SeedAndTrackTab::Seed
            && self.cur_seed_model == SeedModelEnumeratedType::Auto
        {
            self.update_seed_model(manager);
        } else {
            manager.pack(self.process_dialog.axis_id);
        }
        manager.move_sub_frame();
    }
    /// Java private `changeRunRaptorTab()`.
    pub fn change_run_raptor_tab<M: FiducialModelDialogApplicationManager>(
        &mut self,
        manager: &mut M,
    ) {
        manager.pack(self.process_dialog.axis_id);
        manager.move_sub_frame();
    }
    /// Java private `updateSeedModel()`.
    pub fn update_seed_model<M: FiducialModelDialogApplicationManager>(&mut self, manager: &mut M) {
        manager.pack(self.process_dialog.axis_id);
    }
    /// Java `updateDisplay()`.
    pub fn update_display<M: FiducialModelDialogApplicationManager>(&mut self, manager: &M) {
        if manager.is_seeding_done(self.process_dialog.axis_id) {
            self.btn_seed.multi_line_button.set_text(SEEDING_DONE_LABEL);
        } else {
            self.btn_seed
                .multi_line_button
                .set_text(SEEDING_NOT_DONE_LABEL);
        }
        self.update_display_without_manager();
    }
    fn update_display_without_manager(&mut self) {
        let boundary = self.cb_boundary_model.is_selected();
        self.btn_boundary_model.set_enabled(boundary);
        self.cb_exclude_inside_areas.set_enabled(boundary);
        if self.cb_append_to_seed_model.is_selected() {
            self.btn_autofidseed
                .multi_line_button
                .set_text(AUTOFIDSEED_APPEND_LABEL);
            self.rb_seed_model_auto.radio_button.text = AUTOFIDSEED_APPEND_TITLE.into();
        } else {
            self.btn_autofidseed
                .multi_line_button
                .set_text(AUTOFIDSEED_NEW_MODEL_LABEL);
            self.rb_seed_model_auto.radio_button.text = AUTOFIDSEED_NEW_MODEL_TITLE.into();
        }
        self.ltf_lower_target_for_clustered.set_enabled(
            (self.cb_clustered_points_allowed_clustered.is_selected()
                || self.cbs_elongated_points_allowed.is_selected())
                && !self.rtf_target_density_of_beads.is_selected(),
        );
    }
    /// Java `expand(GlobalExpandButton)`.
    pub fn expand<M: FiducialModelDialogApplicationManager>(&mut self, manager: &mut M) {
        self.update_advanced(self.process_dialog.btn_advanced.is_expanded());
        manager.pack(self.process_dialog.axis_id);
    }
    /// Java `expand(ExpandButton)`, intentionally empty.
    pub fn expand_button(&mut self) {}
    /// Java private `updateAdvanced(boolean)`.
    pub fn update_advanced(&mut self, advanced: bool) {
        self.pnl_beadtrack.update_advanced(advanced);
        if let Some(panel) = &mut self.pnl_transferfid {
            panel.update_advanced(advanced);
        }
        self.tiltxcorr_panel.update_advanced(advanced);
        self.cb_adjust_sizes.set_visible(advanced);
        self.ltf_borders_in_x_and_y.set_visible(advanced);
        self.ltf_min_guess_num_beads.set_visible(advanced);
        self.ltf_min_spacing.set_visible(advanced);
        self.ltf_peak_storage_fraction.set_visible(advanced);
        self.ltf_ignore_surface_data.set_visible(advanced);
        self.ltf_drop_tracks.set_visible(advanced);
        self.ltf_max_major_to_minor_ratio.set_visible(advanced);
        self.cb_clustered_points_allowed_clustered
            .set_visible(advanced);
        self.cbs_elongated_points_allowed.set_visible(advanced);
        self.ltf_lower_target_for_clustered.set_visible(advanced);
        self.btn_3dmod_clustered_elongated_model
            .multi_line_button
            .set_visible(advanced);
    }
    /// Java `updateEnabled()`.
    pub fn update_enabled<M: FiducialModelDialogApplicationManager>(&mut self, manager: &M) {
        if let Some(panel) = &mut self.pnl_transferfid {
            panel.set_enabled(self.transferfid_enabled);
        }
        self.rb_seed_model_transfer
            .set_enabled(self.transferfid_enabled);
        if !self.transferfid_enabled {
            self.rb_seed_model_auto.set_selected(true);
            self.cur_seed_model = SeedModelEnumeratedType::Auto;
        }
        self.btn_3dmod_initial_bead_finding
            .set_enabled(manager.autofidseed_initial_model_exists(self.process_dialog.axis_id));
        let models = manager.sorted_models_exist(self.process_dialog.axis_id);
        self.btn_3dmod_bead_selection_and_sorting
            .set_enabled(models);
        self.ltf_ignore_surface_data.set_enabled(models);
        self.btn_3dmod_clustered_elongated_model
            .set_enabled(manager.clustered_elongated_model_exists(self.process_dialog.axis_id));
        self.btn_use_adjusted_track_com.set_enabled(
            manager.adjusted_track_com_exists(self.process_dialog.axis_id)
                && manager
                    .tracking_adjusted(self.process_dialog.axis_id)
                    .unwrap_or(true),
        );
    }
    /// Java `setBeadtrackParams(BeadtrackParam,boolean)`.
    pub fn set_beadtrack_params<
        P: super::beadtrack_panel::BeadtrackParam + super::raptor_panel::RaptorBeadtrackParam,
    >(
        &mut self,
        parameter: &P,
        for_transfer_fid: bool,
    ) {
        if !for_transfer_fid {
            if let Some(raptor) = &mut self.raptor_panel {
                raptor.set_beadtrack_params(parameter);
            }
        }
        self.pnl_beadtrack
            .set_parameters(parameter, for_transfer_fid);
    }
    /// Java `setTransferFidParams()` is delegated at its concrete manager boundary.
    pub fn set_transfer_fid_params<
        P: super::transferfid_panel::TransferfidParam,
        M: super::transferfid_panel::TransferfidPanelApplicationManager<P>,
    >(
        &mut self,
        manager: &M,
    ) {
        if let Some(panel) = &mut self.pnl_transferfid {
            panel.set_parameters(manager);
        }
    }
    /// Java `getBeadTrackDisplay()`.
    pub fn get_bead_track_display(&self) -> &BeadtrackPanel {
        &self.pnl_beadtrack
    }
    /// Java `getTiltxcorrDisplay()`.
    pub fn get_tiltxcorr_display(&self) -> &TiltxcorrPanel {
        &self.tiltxcorr_panel
    }
    /// Java overloaded `getParameters(BaseScreenState)`.
    pub fn get_screen_state_parameters<
        S: super::panel_header::BaseScreenState + super::tiltxcorr_panel::TiltxcorrScreenState,
    >(
        &mut self,
        screen_state: &mut S,
    ) {
        self.pnl_beadtrack.get_parameters_screen_state(screen_state);
        self.tiltxcorr_panel
            .get_parameters_screen_state(screen_state);
    }
    /// Java overloaded `setParameters(ReconScreenState)`.
    pub fn set_screen_state_parameters<
        S: super::panel_header::BaseScreenState + super::tiltxcorr_panel::TiltxcorrScreenState,
    >(
        &mut self,
        screen_state: &mut S,
    ) {
        self.pnl_beadtrack.set_parameters_screen_state(screen_state);
        self.tiltxcorr_panel
            .set_parameters_screen_state(screen_state);
    }
    /// Java `getParameters(RunraptorParam,boolean)`.
    pub fn get_runraptor_parameters<
        P: super::raptor_panel::RaptorRunraptorParam,
        M: super::raptor_panel::RaptorPanelApplicationManager,
    >(
        &mut self,
        parameter: &mut P,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        self.raptor_panel.as_mut().map_or(true, |panel| {
            panel.get_parameters(parameter, do_validation, manager)
        })
    }
    /// Java overloaded `getParameters(MetaData)`.  Calls into the concrete
    /// metadata object are represented by `FiducialModelMetaData`; the member
    /// panel metadata calls are their separately translated source boundaries.
    pub fn get_metadata_parameters<M: FiducialModelMetaData>(&self, metadata: &mut M) {
        let axis = self.process_dialog.axis_id;
        if let Some(panel) = &self.raptor_panel {
            panel.get_metadata_parameters(metadata);
        }
        self.tiltxcorr_panel.get_parameters_meta_data(metadata);
        metadata.set_track_method(
            axis,
            match self.cur_method {
                TrackingMethod::Seed => "Seed",
                TrackingMethod::PatchTracking => "PatchTracking",
                TrackingMethod::Raptor => "Raptor",
            }
            .into(),
        );
        metadata
            .set_seed_model_manual(axis, self.cur_seed_model == SeedModelEnumeratedType::Manual);
        metadata.set_seed_model_auto(axis, self.cur_seed_model == SeedModelEnumeratedType::Auto);
        metadata.set_seed_model_transfer(
            axis,
            self.cur_seed_model == SeedModelEnumeratedType::Transfer,
        );
        metadata.set_exclude_inside_areas(axis, self.cb_exclude_inside_areas.is_selected());
        metadata
            .set_just_find_shifts_near_zero(axis, self.ltf_just_find_shifts_near_zero.text.clone());
        metadata.set_target_number_of_beads(
            axis,
            self.rtf_target_number_of_beads
                .get_text(false)
                .unwrap_or_default(),
        );
        metadata.set_target_density_of_beads(
            axis,
            self.rtf_target_density_of_beads
                .get_text(false)
                .unwrap_or_default(),
        );
        metadata.set_elongated_points_allowed(
            axis,
            self.cbs_elongated_points_allowed.get_value().int_value(),
        );
        metadata
            .set_lower_target_for_clustered(axis, self.ltf_lower_target_for_clustered.text.clone());
        metadata.set_advanced(axis, self.process_dialog.btn_advanced.is_expanded());
        metadata.set_seed_and_track_tab(axis, self.cur_seed_and_track_tab.index() as i32);
        metadata.set_raptor_tab(axis, self.cur_run_raptor_tab.index() as i32);
    }
    /// Java overloaded `setParameters(ConstMetaData)`.
    pub fn set_metadata_parameters<M: FiducialModelMetaData>(&mut self, metadata: &M) {
        let axis = self.process_dialog.axis_id;
        self.cur_method = match metadata.track_method(axis).as_deref() {
            Some("PatchTracking") => TrackingMethod::PatchTracking,
            Some("Raptor") if axis != AxisID::Second => TrackingMethod::Raptor,
            _ => TrackingMethod::Seed,
        };
        self.tiltxcorr_panel.set_parameters_meta_data(metadata);
        self.cur_seed_model = if metadata.seed_model_transfer(axis) {
            SeedModelEnumeratedType::Transfer
        } else if metadata.seed_model_auto(axis) {
            SeedModelEnumeratedType::Auto
        } else {
            SeedModelEnumeratedType::Manual
        };
        self.ltf_just_find_shifts_near_zero
            .set_text(&metadata.just_find_shifts_near_zero(axis));
        self.cb_exclude_inside_areas
            .set_selected(metadata.exclude_inside_areas(axis));
        self.rtf_target_number_of_beads
            .set_text(&metadata.target_number_of_beads(axis));
        self.rtf_target_density_of_beads
            .set_text(&metadata.target_density_of_beads(axis));
        if let Some(value) = metadata.elongated_points_allowed(axis) {
            self.cbs_elongated_points_allowed.set_selected(true);
            self.cbs_elongated_points_allowed
                .set_value_string(&value.to_string());
        } else {
            self.cbs_elongated_points_allowed.set_selected(false);
        }
        self.ltf_lower_target_for_clustered
            .set_text(&metadata.lower_target_for_clustered(axis));
        self.process_dialog
            .btn_advanced
            .change_state(metadata.advanced(axis));
        self.cur_seed_and_track_tab =
            SeedAndTrackTab::get_instance(metadata.seed_and_track_tab(axis));
        self.cur_run_raptor_tab = RunRaptorTab::get_instance(metadata.raptor_tab(axis));
        self.update_advanced(self.process_dialog.btn_advanced.is_expanded());
        self.update_display_without_manager();
    }
    /// The Raptor half of Java `setParameters(ConstMetaData)`, whose Rust
    /// `RaptorPanel` source unit makes its ApplicationManager an explicit input.
    pub fn set_metadata_raptor_parameters<
        M: FiducialModelMetaData,
        A: super::raptor_panel::RaptorPanelApplicationManager,
    >(
        &mut self,
        metadata: &M,
        manager: &A,
    ) {
        if let Some(panel) = &mut self.raptor_panel {
            panel.set_parameters(metadata, manager);
        }
    }
    /// Java `setParameters(ConstTiltxcorrParam)`.
    pub fn set_tiltxcorr_parameters<P: super::tiltxcorr_panel::ConstTiltxcorrParam>(
        &mut self,
        parameter: &P,
    ) {
        self.tiltxcorr_panel.set_parameters_tiltxcorr(parameter);
    }
    /// Java `setParameters(ImodchopcontsParam)`.
    pub fn set_imodchopconts_parameters<P: super::tiltxcorr_panel::ImodchopcontsParam>(
        &mut self,
        parameter: &P,
    ) {
        self.tiltxcorr_panel.set_parameters_imodchopconts(parameter);
    }
    /// Java `getTransferFidParams(boolean)`.
    pub fn get_transfer_fid_params<
        P: super::transferfid_panel::TransferfidParam,
        M: super::transferfid_panel::TransferfidPanelApplicationManager<P>,
    >(
        &self,
        manager: &mut M,
        do_validation: bool,
    ) -> bool {
        self.pnl_transferfid
            .as_ref()
            .map_or(true, |panel| panel.get_parameters(manager, do_validation))
    }
    /// Java overloaded `getTransferFidParams(TransferfidParam,boolean)`.
    pub fn get_transfer_fid_parameters_into<
        P: super::transferfid_panel::TransferfidParam,
        M: super::transferfid_panel::TransferfidPanelApplicationManager<P>,
    >(
        &self,
        manager: &mut M,
        parameter: &mut P,
        do_validation: bool,
    ) -> bool {
        self.pnl_transferfid.as_ref().map_or(true, |panel| {
            panel.get_parameters_into(manager, parameter, do_validation)
        })
    }
    /// Java `setParameters(AutofidseedParam,boolean)`.
    pub fn set_autofidseed_parameters(&mut self, p: &AutofidseedFields, for_transfer_fid: bool) {
        self.cb_adjust_sizes.set_selected(p.adjust_sizes);
        self.ltf_min_guess_num_beads
            .set_text(&p.min_guess_num_beads);
        self.ltf_min_spacing.set_text(&p.min_spacing);
        self.ltf_peak_storage_fraction
            .set_text(&p.peak_storage_fraction);
        self.cb_two_surfaces.set_selected(p.two_surfaces);
        self.cb_clustered_points_allowed_clustered
            .set_selected(p.clustered_points_allowed);
        self.cbs_elongated_points_allowed
            .set_selected(p.elongated_points_allowed.is_some());
        if let Some(value) = p.elongated_points_allowed {
            self.cbs_elongated_points_allowed
                .set_value_string(&value.to_string());
        }
        self.ltf_lower_target_for_clustered
            .set_text(p.lower_target_for_clustered.as_deref().unwrap_or(""));
        self.rtf_target_number_of_beads
            .set_selected(p.target_number.is_some());
        self.rtf_target_number_of_beads
            .set_text(p.target_number.as_deref().unwrap_or(""));
        self.rtf_target_density_of_beads
            .set_selected(p.target_density.is_some());
        self.rtf_target_density_of_beads
            .set_text(p.target_density.as_deref().unwrap_or(""));
        if !for_transfer_fid {
            self.cb_boundary_model.set_selected(p.boundary_model);
            self.cb_exclude_inside_areas
                .set_selected(p.exclude_inside_areas);
            self.ltf_borders_in_x_and_y.set_text(&p.borders_in_x_and_y);
            self.cb_append_to_seed_model
                .set_selected(p.append_to_seed_model);
            self.ltf_ignore_surface_data
                .set_text(&p.ignore_surface_data);
            self.ltf_drop_tracks.set_text(&p.drop_tracks);
        }
        self.update_display_without_manager();
    }
    /// Java `getParameters(AutofidseedParam,boolean,boolean)` after the parameter-object boundary.
    pub fn get_autofidseed_parameters(
        &self,
        just_find_shifts_near_zero: bool,
    ) -> AutofidseedFields {
        AutofidseedFields {
            just_find_shifts_near_zero: just_find_shifts_near_zero
                .then(|| self.ltf_just_find_shifts_near_zero.text.clone()),
            min_guess_num_beads: self.ltf_min_guess_num_beads.text.clone(),
            min_spacing: self.ltf_min_spacing.text.clone(),
            peak_storage_fraction: self.ltf_peak_storage_fraction.text.clone(),
            boundary_model: self.cb_boundary_model.is_selected(),
            exclude_inside_areas: self.cb_exclude_inside_areas.is_enabled()
                && self.cb_exclude_inside_areas.is_selected(),
            adjust_sizes: self.cb_adjust_sizes.is_selected(),
            borders_in_x_and_y: self.ltf_borders_in_x_and_y.text.clone(),
            two_surfaces: self.cb_two_surfaces.is_selected(),
            append_to_seed_model: self.cb_append_to_seed_model.is_selected(),
            target_number: self.rtf_target_number_of_beads.is_selected().then(|| {
                self.rtf_target_number_of_beads
                    .get_text(false)
                    .unwrap_or_default()
            }),
            target_density: self.rtf_target_density_of_beads.is_selected().then(|| {
                self.rtf_target_density_of_beads
                    .get_text(false)
                    .unwrap_or_default()
            }),
            max_major_to_minor_ratio: self.ltf_max_major_to_minor_ratio.text.clone(),
            clustered_points_allowed: self.cb_clustered_points_allowed_clustered.is_selected(),
            elongated_points_allowed: self.cbs_elongated_points_allowed.is_selected().then(|| {
                match self.cbs_elongated_points_allowed.get_value() {
                    Number::Integer(value) => value,
                    _ => 0,
                }
            }),
            lower_target_for_clustered: self
                .ltf_lower_target_for_clustered
                .is_enabled()
                .then(|| self.ltf_lower_target_for_clustered.text.clone()),
            ignore_surface_data: self.ltf_ignore_surface_data.text.clone(),
            drop_tracks: self.ltf_drop_tracks.text.clone(),
        }
    }
    /// Java `setTransferfidEnabled(boolean)`.
    pub fn set_transferfid_enabled(&mut self, file_exists: bool) {
        self.transferfid_enabled = file_exists;
    }
    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        let automatic = self.cur_method == TrackingMethod::Seed
            && self.cur_seed_and_track_tab == SeedAndTrackTab::Seed
            && self.cur_seed_model == SeedModelEnumeratedType::Auto;
        let (labels, pages, anchor) = if automatic {
            (
                vec![
                    "Autofidseed",
                    "Imodfindbeads",
                    "Beadtrack",
                    "Sortbeadsurfs",
                    "Pickbestseed",
                ],
                vec![
                    "autofidseed.html",
                    "imodfindbeads.html",
                    "beadtrack.html",
                    "sortbeadsurfs.html",
                    "pickbestseed.html",
                ],
                "AutomaticSeed",
            )
        } else {
            (
                vec!["Autofidseed", "Beadtrack", "Transferfid", "3dmod"],
                vec![
                    "autofidseed.html",
                    "beadtrack.html",
                    "transferfid.html",
                    "3dmod.html",
                ],
                "GETTING FIDUCIAL",
            )
        };
        let man_page_label = labels.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let man_page = pages.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let log_file_label = vec!["Autofidseed".into(), "Track".into(), "Transferfid".into()];
        let log_file = vec![
            format!(
                "autofidseed{}.log",
                self.process_dialog.axis_id.get_extension()
            ),
            format!("track{}.log", self.process_dialog.axis_id.get_extension()),
            "transferfid.log".into(),
        ];
        self.context_popup = ContextPopup::new_log_files(
            mouse_event,
            Some(anchor),
            super::context_popup::TOMO_GUIDE,
            &man_page_label,
            &man_page,
            &log_file_label,
            &log_file,
            self.process_dialog.axis_id,
            None,
        )
        .ok();
    }
    /// Java private `setToolTipText()`; autodoc-generated tooltips remain an explicit boundary.
    fn set_tool_tip_text(&mut self) {
        self.btn_seed
            .set_tool_tip_text("Open new or existing seed model in 3dmod.");
        self.btn_autofidseed.set_tool_tip_text(
            "Run Autofidseed to find beads, track them through 11 views, and select a seed model.",
        );
        self.btn_cleanup
            .set_tool_tip_text(Some("Delete the temporary directory."));
        self.btn_use_adjusted_track_com.set_tool_tip_text(Some("For tracking this seed, use the com file with an adjusted bead size or information on large shifts between views."));
    }
    /// Java `action(String,Deferred3dmodButton,Run3dmodMenuOptions)`.
    pub fn action<M: FiducialModelDialogApplicationManager>(
        &mut self,
        command: &str,
        options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if command == self.rb_method_seed.get_action_command() {
            self.cur_method = TrackingMethod::Seed;
            self.update_method(manager);
        } else if command == self.rb_method_patch_tracking.get_action_command() {
            self.cur_method = TrackingMethod::PatchTracking;
            self.update_method(manager);
        } else if command == self.rb_method_raptor.get_action_command() {
            self.cur_method = TrackingMethod::Raptor;
            self.update_method(manager);
        } else if command == self.rb_seed_model_manual.get_action_command() {
            self.cur_seed_model = SeedModelEnumeratedType::Manual;
            self.update_seed_model(manager);
        } else if command == self.rb_seed_model_auto.get_action_command() {
            self.cur_seed_model = SeedModelEnumeratedType::Auto;
            self.update_seed_model(manager);
        } else if command == self.rb_seed_model_transfer.get_action_command() {
            self.cur_seed_model = SeedModelEnumeratedType::Transfer;
            self.update_seed_model(manager);
        } else if Some(command) == self.btn_seed.get_action_command()
            || Some(command) == self.btn_3dmod_autofidseed.get_action_command()
        {
            manager.imod_seed_model(self.process_dialog.axis_id, options);
        } else if Some(command) == self.btn_autofidseed.get_action_command() {
            manager.autofidseed(self.process_dialog.axis_id, false);
        } else if command
            == self
                .btn_just_find_shifts_near_zero
                .get_action_command()
                .unwrap_or("")
        {
            manager.autofidseed(self.process_dialog.axis_id, true);
        } else if Some(command) == self.btn_3dmod_initial_bead_finding.get_action_command() {
            manager.imod_initial_bead_model(self.process_dialog.axis_id, options);
        } else if Some(command)
            == self
                .btn_3dmod_bead_selection_and_sorting
                .get_action_command()
        {
            manager.imod_sorted_models(self.process_dialog.axis_id, options);
        } else if Some(command)
            == self
                .btn_3dmod_clustered_elongated_model
                .get_action_command()
        {
            manager.imod_clustered_elongated_model(self.process_dialog.axis_id, options);
        } else if command == self.btn_cleanup.get_action_command().unwrap_or("") {
            manager.cleanup_autofidseed(self.process_dialog.axis_id);
            self.update_enabled(manager);
        } else if Some(command) == self.btn_boundary_model.get_action_command() {
            manager.imod_boundary_model(self.process_dialog.axis_id, options);
        } else if command
            == self
                .btn_use_adjusted_track_com
                .get_action_command()
                .unwrap_or("")
        {
            manager.use_track_adjusted_comfile(self.process_dialog.axis_id);
            self.update_enabled(manager);
        }
        self.update_display(manager);
    }
    /// Java override `done()`.
    pub fn done<M: FiducialModelDialogApplicationManager>(&mut self, manager: &mut M) {
        manager.done_fiducial_model_dialog(self.process_dialog.axis_id);
        if let Some(panel) = &mut self.pnl_transferfid {
            panel.done();
        }
        if let Some(panel) = &mut self.raptor_panel {
            panel.done();
        }
        self.pnl_beadtrack.done();
        self.tiltxcorr_panel.done();
        self.process_dialog.set_displayed(false);
    }
}

/// Source-shaped parameter adapter.  The full typed overloads remain on the
/// dialog; this mirrors Java's `getParameters`/`setParameters` UI boundary.
pub struct FiducialModelDialogParameters;
impl FiducialModelDialogParameters {
    #[allow(non_snake_case)]
    pub fn getParameters(dialog: &FiducialModelDialog<'_>, near_zero: bool) -> AutofidseedFields {
        dialog.get_autofidseed_parameters(near_zero)
    }
    #[allow(non_snake_case)]
    pub fn setParameters(
        dialog: &mut FiducialModelDialog<'_>,
        fields: &AutofidseedFields,
        for_transfer_fid: bool,
    ) {
        dialog.set_autofidseed_parameters(fields, for_transfer_fid);
    }
}

/// Native replacement for the Java action listener.
pub struct FiducialModelDialogActionListener;
impl FiducialModelDialogActionListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed<M: FiducialModelDialogApplicationManager>(
        dialog: &mut FiducialModelDialog<'_>,
        command: &str,
        manager: &mut M,
    ) {
        dialog.action(command, None, manager);
    }
}

/// Native replacement for the Java tab/change listener.
pub struct FiducialModelDialogChangeListener;
impl FiducialModelDialogChangeListener {
    #[allow(non_snake_case)]
    pub fn stateChanged(dialog: &mut FiducialModelDialog<'_>) {
        dialog.update_display_without_manager();
    }
}

impl FiducialModelDialog<'_> {
    #[allow(non_snake_case)]
    pub fn toString(&self) -> String {
        format!(
            "FiducialModelDialog[axis={:?}, method={:?}]",
            self.process_dialog.axis_id, self.cur_method
        )
    }
    #[allow(non_snake_case)]
    pub fn equals(&self, other: &Self) -> bool {
        self.process_dialog.axis_id == other.process_dialog.axis_id
            && self.cur_method == other.cur_method
            && self.cur_seed_model == other.cur_seed_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        events: Vec<String>,
    }
    impl ProcessDialogApplicationManager for Manager {
        fn is_advanced(&self, _: DialogType, _: AxisID) -> bool {
            false
        }
    }
    impl FiducialModelDialogApplicationManager for Manager {
        fn is_dual_axis(&self) -> bool {
            false
        }
        fn is_seeding_done(&self, _: AxisID) -> bool {
            false
        }
        fn autofidseed_initial_model_exists(&self, _: AxisID) -> bool {
            true
        }
        fn sorted_models_exist(&self, _: AxisID) -> bool {
            true
        }
        fn clustered_elongated_model_exists(&self, _: AxisID) -> bool {
            false
        }
        fn adjusted_track_com_exists(&self, _: AxisID) -> bool {
            true
        }
        fn tracking_adjusted(&self, _: AxisID) -> Result<bool, String> {
            Ok(true)
        }
        fn pack(&mut self, _: AxisID) {}
        fn move_sub_frame(&mut self) {}
        fn imod_seed_model(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.events.push("seed".into())
        }
        fn autofidseed(&mut self, _: AxisID, near: bool) {
            self.events.push(format!("auto:{near}"))
        }
        fn imod_initial_bead_model(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {}
        fn imod_sorted_models(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {}
        fn imod_clustered_elongated_model(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {}
        fn cleanup_autofidseed(&mut self, _: AxisID) {}
        fn imod_boundary_model(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {}
        fn use_track_adjusted_comfile(&mut self, _: AxisID) {}
        fn done_fiducial_model_dialog(&mut self, _: AxisID) {}
    }
    #[test]
    fn source_parameter_round_trip_preserves_conditional_fields() {
        let manager = Manager::default();
        let mut d = FiducialModelDialog::get_instance(&manager, AxisID::Only, AxisType::SingleAxis);
        d.cb_boundary_model.set_selected(true);
        d.rtf_target_number_of_beads.set_text("43");
        let p = d.get_autofidseed_parameters(false);
        assert_eq!(p.target_number.as_deref(), Some("43"));
        assert_eq!(p.just_find_shifts_near_zero, None);
    }
    #[test]
    fn append_switches_source_labels() {
        let manager = Manager::default();
        let mut d = FiducialModelDialog::get_instance(&manager, AxisID::Only, AxisType::SingleAxis);
        d.cb_append_to_seed_model.set_selected(true);
        d.update_display(&manager);
        assert_eq!(
            d.btn_autofidseed.multi_line_button.get_text(),
            Some(AUTOFIDSEED_APPEND_LABEL)
        );
    }
}
