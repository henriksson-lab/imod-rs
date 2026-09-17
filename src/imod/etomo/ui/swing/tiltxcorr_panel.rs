//! `IMOD/Etomo/src/etomo/ui/swing/TiltxcorrPanel.java`.
//!
//! The Swing widgets, autodoc lookup, and `ApplicationManager` are presentation
//! boundaries.  This unit retains the Java panel's field state, enablement,
//! validation, parameter transfer, and action routing without replacing the
//! application's process policy.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;
use std::collections::BTreeMap;

use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::radio_text_field::RadioTextField;

pub const PANEL_ID_CROSS_CORRELATION: &str = "Cross Correlation";
pub const PANEL_ID_PATCH_TRACKING: &str = "Patch Tracking";
pub const ITERATE_CORRELATIONS_DEFAULT: i32 = 1;
pub const ITERATE_CORRELATIONS_MIN: i32 = 0;
pub const ITERATE_CORRELATIONS_MAX: i32 = 20;

/// Java `PanelId` values used by this unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PanelId {
    CrossCorrelation,
    PatchTracking,
}

/// Java `CheckTextField` source-visible state used by `TiltxcorrPanel`.
#[derive(Clone, Debug)]
pub struct CheckTextField {
    pub check_box: CheckBox,
    pub field: LabeledTextField,
}
impl CheckTextField {
    pub fn new(field_type: FieldType, label: &str) -> Self {
        Self {
            check_box: CheckBox::new_with_text(label),
            field: LabeledTextField::new(field_type, label),
        }
    }
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }
    pub fn set_selected(&mut self, value: bool) {
        self.check_box.set_selected(value);
    }
    pub fn set_text(&mut self, value: &str) {
        self.field.set_text(value);
    }
    pub fn get_text(&self, validation: bool) -> Result<String, FieldValidationFailedException> {
        self.field.get_text_validated(validation)
    }
    pub fn set_enabled(&mut self, value: bool) {
        self.check_box.set_enabled(value);
        self.field.set_enabled(value);
    }
    pub fn is_enabled(&self) -> bool {
        self.field.is_enabled()
    }
}

/// Java `Spinner` state needed by this unit.
#[derive(Clone, Debug)]
pub struct Spinner {
    pub label: String,
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub tooltip: Option<String>,
}
impl Spinner {
    pub fn new(label: &str, value: i32, minimum: i32, maximum: i32) -> Self {
        Self {
            label: label.into(),
            value,
            minimum,
            maximum,
            tooltip: None,
        }
    }
    pub fn set_value(&mut self, value: i32) {
        self.value = value;
    }
}

/// Java `ImodchopcontsParam` calls made by this panel.
pub trait ImodchopcontsParam {
    fn minimum_overlap(&self) -> Option<String>;
    fn length_of_pieces_is_null(&self) -> bool;
    fn length_of_pieces_is_default(&self) -> bool;
    fn length_of_pieces(&self) -> Option<String>;
    fn set_minimum_overlap(&mut self, value: String);
    fn set_length_of_pieces_default(&mut self);
    fn set_length_of_pieces(&mut self, value: String);
    fn reset_length_of_pieces(&mut self);
}

/// Java `TiltxcorrParam` calls made by `getParameters`.
pub trait TiltxcorrParam {
    fn set_value(&mut self, key: &str, value: String) -> Result<(), String>;
    fn reset_value(&mut self, key: &str);
    fn set_flag(&mut self, key: &str, value: bool);
    fn set_iterate_correlations(&mut self, value: i32) -> Option<String>;
}

/// Java `ConstTiltxcorrParam` reads made by `setParameters`.
pub trait ConstTiltxcorrParam {
    fn value(&self, key: &str) -> Option<String>;
    fn is_set(&self, key: &str) -> bool;
    fn flag(&self, key: &str) -> bool;
    fn iterate_correlations(&self) -> i32;
}

/// Java metadata reads/writes retained at the metadata boundary.
pub trait TiltxcorrMetaData {
    fn get_value(&self, key: &str, axis_id: AxisID) -> Option<String>;
    fn is_set(&self, key: &str, axis_id: AxisID) -> bool;
    fn set_value(&mut self, key: &str, axis_id: AxisID, value: String);
    fn rotation_angle(&self, axis_id: AxisID) -> f64;
    fn tilt_angle_spec(&self, axis_id: AxisID) -> String;
}

/// Java `BaseScreenState` calls made through this panel's `PanelHeader`.
pub trait TiltxcorrScreenState {
    fn get_button_state(&mut self, key: &str, default: bool) -> bool;
    fn set_button_state(&mut self, key: &str, state: bool);
}

/// In-memory source-shaped `BaseScreenState` for native frontends and tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltxcorrPanelScreenState {
    pub button_states: BTreeMap<String, bool>,
}
impl TiltxcorrScreenState for TiltxcorrPanelScreenState {
    fn get_button_state(&mut self, key: &str, default: bool) -> bool {
        self.button_states.get(key).copied().unwrap_or(default)
    }
    fn set_button_state(&mut self, key: &str, state: bool) {
        self.button_states.insert(key.into(), state);
    }
}

/// Direct `ApplicationManager` calls from `TiltxcorrPanel.java`.
pub trait TiltxcorrPanelApplicationManager {
    fn pre_cross_correlate(&mut self, axis_id: AxisID, dialog_type: DialogType);
    fn tiltxcorr(
        &mut self,
        axis_id: AxisID,
        dialog_type: DialogType,
        run_tiltxcorr: bool,
        length_of_pieces: bool,
        options: Option<Run3dmodMenuOptions>,
    );
    fn imod_model(
        &mut self,
        image_file_type: &str,
        model_file_type: &str,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
    );
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
    fn meta_data(&self) -> &dyn TiltxcorrMetaData;
}

/// All Swing layout state directly set by `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltxcorrPanelLayout {
    pub root_visible: bool,
    pub body_visible: bool,
    pub advanced_visible: bool,
    pub advanced2_visible: bool,
    pub angle_offset_visible: bool,
    pub shift_limits_visible: bool,
    pub mag_changes_visible: bool,
    pub patch_layout_border: bool,
    pub boundary_model_enabled: bool,
    pub absolute_cosine_stretch_enabled: bool,
    pub length_of_pieces_enabled: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
}

/// Complete Java `TiltxcorrPanel` field state.  Widget component hierarchy and
/// listener registration remain native GUI boundaries represented by `layout`.
pub struct TiltxcorrPanel {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub panel_id: PanelId,
    pub mag_changes_mode: bool,
    pub layout: TiltxcorrPanelLayout,
    pub cb_exclude_central_peak: CheckBox,
    pub ltf_test_output: LabeledTextField,
    pub ltf_filter_sigma1: LabeledTextField,
    pub ltf_filter_radius2: LabeledTextField,
    pub ltf_filter_sigma2: LabeledTextField,
    pub ltf_trim: LabeledTextField,
    pub ltf_x_min: LabeledTextField,
    pub ltf_x_max: LabeledTextField,
    pub ltf_y_min: LabeledTextField,
    pub ltf_y_max: LabeledTextField,
    pub ltf_pad_percent: LabeledTextField,
    pub ltf_taper_percent: LabeledTextField,
    pub cb_cumulative_correlation: CheckBox,
    pub cb_absolute_cosine_stretch: CheckBox,
    pub cb_no_cosine_stretch: CheckBox,
    pub ltf_view_range: LabeledTextField,
    pub ltf_angle_offset: LabeledTextField,
    pub ltf_skip_views: LabeledTextField,
    pub ltf_size_of_patches_x_and_y: LabeledTextField,
    pub rtf_overlap_of_patches_x_and_y: RadioTextField,
    pub rtf_number_of_patches_x_and_y: RadioTextField,
    pub sp_iterate_correlations: Spinner,
    pub ltf_shift_limits_x_and_y: LabeledTextField,
    pub ctf_length_of_pieces_minimum_overlap: CheckTextField,
    pub cb_boundary_model: CheckBox,
    pub rb_length_of_pieces_default: bool,
    pub rtf_length_of_pieces: RadioTextField,
    pub ctf_mag_changes: Option<CheckTextField>,
    pub skip_views: Option<String>,
    pub last_context_popup: Option<String>,
    pub actions_attached: bool,
    pub header_open: bool,
    pub header_advanced: bool,
}

impl TiltxcorrPanel {
    fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: PanelId,
        mag_changes_mode: bool,
    ) -> Self {
        Self {
            axis_id,
            dialog_type,
            panel_id,
            mag_changes_mode,
            layout: TiltxcorrPanelLayout {
                root_visible: true,
                body_visible: true,
                advanced_visible: true,
                advanced2_visible: true,
                angle_offset_visible: true,
                shift_limits_visible: true,
                mag_changes_visible: true,
                ..Default::default()
            },
            cb_exclude_central_peak: CheckBox::new_with_text(
                "Exclude central peak due to fixed pattern noise",
            ),
            ltf_test_output: LabeledTextField::new(FieldType::String, "Test output: "),
            ltf_filter_sigma1: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Low frequency rolloff sigma: ",
            ),
            ltf_filter_radius2: LabeledTextField::new(
                FieldType::FloatingPoint,
                "High frequency cutoff radius: ",
            ),
            ltf_filter_sigma2: LabeledTextField::new(
                FieldType::FloatingPoint,
                "High frequency rolloff sigma: ",
            ),
            ltf_trim: LabeledTextField::new(FieldType::IntegerPair, "Pixels to trim (x,y): "),
            ltf_x_min: LabeledTextField::new(FieldType::Integer, "X axis min "),
            ltf_x_max: LabeledTextField::new(FieldType::Integer, "Max "),
            ltf_y_min: LabeledTextField::new(FieldType::Integer, "Y axis min "),
            ltf_y_max: LabeledTextField::new(FieldType::Integer, "Max "),
            ltf_pad_percent: LabeledTextField::new(FieldType::IntegerPair, "Pixels to pad (x,y): "),
            ltf_taper_percent: LabeledTextField::new(
                FieldType::IntegerPair,
                "Pixels to taper (x,y): ",
            ),
            cb_cumulative_correlation: CheckBox::new_with_text("Cumulative correlation"),
            cb_absolute_cosine_stretch: CheckBox::new_with_text("Absolute Cosine Stretch"),
            cb_no_cosine_stretch: CheckBox::new_with_text("No Cosine Stretch"),
            ltf_view_range: LabeledTextField::new(
                FieldType::IntegerPair,
                "View range (start,end): ",
            ),
            ltf_angle_offset: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Tilt angle offset: ",
            ),
            ltf_skip_views: LabeledTextField::new(FieldType::IntegerList, "Views to skip: "),
            ltf_size_of_patches_x_and_y: LabeledTextField::new(
                FieldType::IntegerPair,
                "Size of patches (X,Y): ",
            ),
            rtf_overlap_of_patches_x_and_y: RadioTextField::new(
                FieldType::FloatingPointPair,
                "Fractional overlap of patches (X,Y): ",
            ),
            rtf_number_of_patches_x_and_y: RadioTextField::new(
                FieldType::IntegerPair,
                "Number of patches (X,Y): ",
            ),
            sp_iterate_correlations: Spinner::new(
                "Iterations to increase subpixel accuracy: ",
                ITERATE_CORRELATIONS_DEFAULT,
                ITERATE_CORRELATIONS_MIN,
                ITERATE_CORRELATIONS_MAX,
            ),
            ltf_shift_limits_x_and_y: LabeledTextField::new(
                FieldType::IntegerPair,
                "Limits on shifts from correlation (X,Y): ",
            ),
            ctf_length_of_pieces_minimum_overlap: CheckTextField::new(
                FieldType::Integer,
                "Break contours into pieces with overlap: ",
            ),
            cb_boundary_model: CheckBox::new_with_text("Use boundary model"),
            rb_length_of_pieces_default: true,
            rtf_length_of_pieces: RadioTextField::new(FieldType::Integer, "Use length"),
            ctf_mag_changes: (panel_id == PanelId::CrossCorrelation).then(|| {
                CheckTextField::new(FieldType::IntegerList, "Find mag change at view(s):")
            }),
            skip_views: None,
            last_context_popup: None,
            actions_attached: false,
            header_open: true,
            header_advanced: true,
        }
    }
    pub fn get_cross_correlation_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        mag_changes_mode: bool,
    ) -> Self {
        let mut value = Self::new(
            axis_id,
            dialog_type,
            PanelId::CrossCorrelation,
            mag_changes_mode,
        );
        value.create_panel();
        value.set_tool_tip_text();
        value.add_listeners();
        value
    }
    pub fn get_patch_tracking_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut value = Self::new(axis_id, dialog_type, PanelId::PatchTracking, false);
        value.create_panel();
        value.set_tool_tip_text();
        value.add_listeners();
        value
    }
    fn create_panel(&mut self) {
        if self.panel_id == PanelId::PatchTracking {
            self.rtf_overlap_of_patches_x_and_y.set_text("0.33,0.33");
            self.rtf_overlap_of_patches_x_and_y.set_selected(true);
            self.ltf_filter_sigma1.set_text("0.03");
            self.ltf_filter_radius2.set_text("0.25");
            self.ltf_filter_sigma2.set_text("0.05");
            self.layout.patch_layout_border = true;
        }
        self.update_panel();
    }
    fn add_listeners(&mut self) {
        self.actions_attached = true;
        self.layout.listener_count = if self.ctf_mag_changes.is_some() { 9 } else { 8 };
    }
    pub fn done(&mut self) {
        self.actions_attached = false;
    }
    pub fn get_panel_id(&self) -> PanelId {
        self.panel_id
    }
    pub fn pop_up_context_menu(&mut self) {
        if self.panel_id == PanelId::PatchTracking {
            self.last_context_popup = Some(format!(
                "PatchTracking:xcorr_pt{}.log",
                self.axis_id.get_extension()
            ));
        }
    }
    pub fn update_advanced(&mut self, state: bool) {
        self.layout.advanced_visible = state;
        self.layout.advanced2_visible = state;
        if self.panel_id == PanelId::PatchTracking {
            self.layout.angle_offset_visible = state;
            self.layout.shift_limits_visible = state;
        }
        if !self.mag_changes_mode && self.ctf_mag_changes.is_some() {
            self.layout.mag_changes_visible = state;
        }
    }
    pub fn expand_open_close(&mut self, expanded: bool) {
        self.layout.body_visible = expanded;
    }
    pub fn expand_advanced_basic(&mut self, expanded: bool) {
        self.header_advanced = expanded;
        self.update_advanced(expanded);
    }
    /// Java overloaded `setParameters(BaseScreenState)`.
    pub fn set_parameters_screen_state<S: TiltxcorrScreenState>(&mut self, screen_state: &mut S) {
        self.header_open = screen_state.get_button_state("Tiltxcorr.openClose", true);
        self.header_advanced = screen_state.get_button_state("Tiltxcorr.advancedBasic", true);
        self.expand_open_close(self.header_open);
        self.update_advanced(self.header_advanced);
    }
    /// Java overloaded `getParameters(BaseScreenState)`.
    pub fn get_parameters_screen_state<S: TiltxcorrScreenState>(&self, screen_state: &mut S) {
        screen_state.set_button_state("Tiltxcorr.openClose", self.header_open);
        screen_state.set_button_state("Tiltxcorr.advancedBasic", self.header_advanced);
    }
    pub fn set_parameters_imodchopconts<P: ImodchopcontsParam>(&mut self, param: &P) {
        if self.panel_id == PanelId::PatchTracking {
            self.ctf_length_of_pieces_minimum_overlap
                .set_text(param.minimum_overlap().as_deref().unwrap_or_default());
            let set = !param.length_of_pieces_is_null();
            self.ctf_length_of_pieces_minimum_overlap.set_selected(set);
            if set {
                self.rb_length_of_pieces_default = param.length_of_pieces_is_default();
                if !self.rb_length_of_pieces_default {
                    self.rtf_length_of_pieces.set_selected(true);
                    self.rtf_length_of_pieces
                        .set_text(param.length_of_pieces().as_deref().unwrap_or_default());
                }
            }
            self.update_panel();
        }
    }
    pub fn set_parameters_tiltxcorr<P: ConstTiltxcorrParam>(&mut self, param: &P) {
        macro_rules! field {
            ($field:ident, $key:literal) => {
                if let Some(value) = param.value($key) {
                    self.$field.set_text(&value);
                }
            };
        }
        field!(ltf_angle_offset, "AngleOffset");
        if param.is_set("BordersInXandY") {
            field!(ltf_trim, "BordersInXandY");
        }
        field!(ltf_x_min, "XMin");
        field!(ltf_x_max, "XMax");
        field!(ltf_y_min, "YMin");
        field!(ltf_y_max, "YMax");
        field!(ltf_pad_percent, "PadsInXandY");
        field!(ltf_taper_percent, "TaperPercent");
        field!(ltf_test_output, "TestOutput");
        field!(ltf_view_range, "StartingEndingViews");
        field!(ltf_skip_views, "SkipViews");
        if param.is_set("FilterSigma1") {
            field!(ltf_filter_sigma1, "FilterSigma1");
        }
        if param.is_set("FilterRadius2") {
            field!(ltf_filter_radius2, "FilterRadius2");
        }
        if param.is_set("FilterSigma2") {
            field!(ltf_filter_sigma2, "FilterSigma2");
        }
        if self.panel_id == PanelId::CrossCorrelation {
            self.cb_exclude_central_peak
                .set_selected(param.flag("ExcludeCentralPeak"));
            self.cb_cumulative_correlation
                .set_selected(param.flag("CumulativeCorrelation"));
            self.cb_absolute_cosine_stretch
                .set_selected(param.flag("AbsoluteCosineStretch"));
            self.cb_no_cosine_stretch
                .set_selected(param.flag("NoCosineStretch"));
        } else {
            field!(ltf_size_of_patches_x_and_y, "SizeOfPatchesXAndY");
            field!(ltf_shift_limits_x_and_y, "ShiftLimitsXAndY");
            self.sp_iterate_correlations
                .set_value(param.iterate_correlations());
            self.cb_boundary_model
                .set_selected(param.flag("BoundaryModel"));
            if param.is_set("OverlapOfPatchesXAndY") {
                self.rtf_overlap_of_patches_x_and_y.set_selected(true);
                self.rtf_overlap_of_patches_x_and_y
                    .set_text(&param.value("OverlapOfPatchesXAndY").unwrap_or_default());
            }
            if param.is_set("NumberOfPatchesXAndY") {
                self.rtf_number_of_patches_x_and_y.set_selected(true);
                self.rtf_number_of_patches_x_and_y
                    .set_text(&param.value("NumberOfPatchesXAndY").unwrap_or_default());
            }
        }
        if let Some(mag) = &mut self.ctf_mag_changes {
            mag.set_selected(param.flag("SearchMagChanges"));
            mag.set_text(
                param
                    .value("ViewsWithMagChanges")
                    .as_deref()
                    .unwrap_or_default(),
            );
        }
        self.update_panel();
    }
    pub fn set_parameters_meta_data<M: TiltxcorrMetaData>(&mut self, metadata: &M) {
        if self.panel_id == PanelId::PatchTracking {
            if metadata.is_set("TrackOverlapOfPatchesXAndY", self.axis_id) {
                self.rtf_overlap_of_patches_x_and_y.set_text(
                    metadata
                        .get_value("TrackOverlapOfPatchesXAndY", self.axis_id)
                        .as_deref()
                        .unwrap_or_default(),
                );
            }
            self.rtf_number_of_patches_x_and_y.set_text(
                metadata
                    .get_value("TrackNumberOfPatchesXAndY", self.axis_id)
                    .as_deref()
                    .unwrap_or_default(),
            );
            self.rtf_length_of_pieces.set_text(
                metadata
                    .get_value("LengthOfPieces", self.axis_id)
                    .as_deref()
                    .unwrap_or_default(),
            );
        }
    }
    pub fn get_parameters_meta_data<M: TiltxcorrMetaData>(&self, metadata: &mut M) {
        if self.panel_id == PanelId::PatchTracking {
            metadata.set_value(
                "TrackOverlapOfPatchesXAndY",
                self.axis_id,
                self.rtf_overlap_of_patches_x_and_y.get_text_unvalidated(),
            );
            metadata.set_value(
                "TrackNumberOfPatchesXAndY",
                self.axis_id,
                self.rtf_number_of_patches_x_and_y.get_text_unvalidated(),
            );
            metadata.set_value(
                "LengthOfPieces",
                self.axis_id,
                self.rtf_length_of_pieces.get_text_unvalidated(),
            );
        }
    }
    pub fn get_parameters_imodchopconts<P: ImodchopcontsParam>(
        &self,
        param: &mut P,
        validation: bool,
    ) -> bool {
        if self.panel_id != PanelId::PatchTracking {
            return true;
        }
        let Ok(overlap) = self
            .ctf_length_of_pieces_minimum_overlap
            .get_text(validation)
        else {
            return false;
        };
        param.set_minimum_overlap(overlap);
        if self.ctf_length_of_pieces_minimum_overlap.is_selected() {
            if self.rb_length_of_pieces_default {
                param.set_length_of_pieces_default();
            } else {
                let Ok(value) = self.rtf_length_of_pieces.get_text(validation) else {
                    return false;
                };
                param.set_length_of_pieces(value);
            }
        } else {
            param.reset_length_of_pieces();
        }
        true
    }
    pub fn get_parameters_tiltxcorr<P: TiltxcorrParam>(
        &self,
        param: &mut P,
        validation: bool,
        metadata: Option<&dyn TiltxcorrMetaData>,
    ) -> Result<bool, String> {
        let set = |p: &mut P, key: &str, field: &LabeledTextField| -> Result<(), String> {
            p.set_value(
                key,
                field
                    .get_text_validated(validation)
                    .map_err(|e| e.to_string())?,
            )
        };
        set(param, "TestOutput", &self.ltf_test_output)?;
        if self.panel_id == PanelId::CrossCorrelation {
            param.set_flag(
                "ExcludeCentralPeak",
                self.cb_exclude_central_peak.is_selected(),
            );
        } else {
            if let Some(error) = param.set_iterate_correlations(self.sp_iterate_correlations.value)
            {
                return Err(format!("{}: {error}", self.sp_iterate_correlations.label));
            }
            param.set_value("InputFile", "PREALIGNED_STACK".into())?;
            param.set_value("OutputFile", "FIDUCIAL_PATCH_TRACKING_MODEL".into())?;
            if self.cb_boundary_model.is_selected() {
                param.set_value("BoundaryModel", "PATCH_TRACKING_BOUNDARY_MODEL".into())?;
            } else {
                param.reset_value("BoundaryModel");
            }
            if let Some(metadata) = metadata {
                param.set_value("TiltAngleSpec", metadata.tilt_angle_spec(self.axis_id))?;
                param.set_value(
                    "RotationAngle",
                    metadata.rotation_angle(self.axis_id).to_string(),
                )?;
            }
        }
        for (key, field) in [
            ("AngleOffset", &self.ltf_angle_offset),
            ("BordersInXandY", &self.ltf_trim),
            ("XMin", &self.ltf_x_min),
            ("XMax", &self.ltf_x_max),
            ("YMin", &self.ltf_y_min),
            ("YMax", &self.ltf_y_max),
            ("PadsInXAndY", &self.ltf_pad_percent),
            ("TapersInXAndY", &self.ltf_taper_percent),
            ("StartingEndingViews", &self.ltf_view_range),
            ("SkipViews", &self.ltf_skip_views),
            ("FilterSigma1", &self.ltf_filter_sigma1),
            ("FilterRadius2", &self.ltf_filter_radius2),
            ("FilterSigma2", &self.ltf_filter_sigma2),
        ] {
            set(param, key, field)?;
        }
        if self.panel_id == PanelId::CrossCorrelation {
            param.set_flag(
                "CumulativeCorrelation",
                self.cb_cumulative_correlation.is_selected(),
            );
            param.set_flag(
                "AbsoluteCosineStretch",
                self.cb_absolute_cosine_stretch.is_selected(),
            );
            param.set_flag("NoCosineStretch", self.cb_no_cosine_stretch.is_selected());
        } else {
            set(
                param,
                "SizeOfPatchesXAndY",
                &self.ltf_size_of_patches_x_and_y,
            )?;
            if self.rtf_overlap_of_patches_x_and_y.is_selected() {
                param.set_value(
                    "OverlapOfPatchesXAndY",
                    self.rtf_overlap_of_patches_x_and_y
                        .get_text(validation)
                        .map_err(|e| e.to_string())?,
                )?;
            } else {
                param.reset_value("OverlapOfPatchesXAndY");
            }
            if self.rtf_number_of_patches_x_and_y.is_selected() {
                param.set_value(
                    "NumberOfPatchesXAndY",
                    self.rtf_number_of_patches_x_and_y
                        .get_text(validation)
                        .map_err(|e| e.to_string())?,
                )?;
            } else {
                param.reset_value("NumberOfPatchesXAndY");
            }
            set(param, "ShiftLimitsXAndY", &self.ltf_shift_limits_x_and_y)?;
            param.set_value("PrealignmentTransformFile", "default".into())?;
            param.set_value("ImagesAreBinned", "PREALIGNED_STACK".into())?;
        }
        if let Some(mag) = &self.ctf_mag_changes {
            param.set_flag("SearchMagChanges", mag.is_selected());
            param.set_value(
                "ViewsWithMagChanges",
                mag.get_text(validation).map_err(|e| e.to_string())?,
            )?;
        }
        Ok(true)
    }
    pub fn set_visible(&mut self, state: bool) {
        self.layout.root_visible = state;
    }
    pub fn update_panel(&mut self) {
        let absolute = self.cb_cumulative_correlation.is_selected()
            && !self.cb_no_cosine_stretch.is_selected();
        self.cb_absolute_cosine_stretch.set_enabled(absolute);
        if !absolute {
            self.cb_absolute_cosine_stretch.set_selected(false);
        }
        self.layout.absolute_cosine_stretch_enabled = absolute;
        self.layout.boundary_model_enabled = self.cb_boundary_model.is_selected();
        if let Some(mag) = &mut self.ctf_mag_changes {
            let cumulative_enabled = !mag.is_selected() || !mag.is_enabled();
            self.cb_cumulative_correlation
                .set_enabled(cumulative_enabled);
            mag.set_enabled(!self.cb_cumulative_correlation.is_selected() || !cumulative_enabled);
        }
        let enable = self.ctf_length_of_pieces_minimum_overlap.is_selected();
        self.layout.length_of_pieces_enabled = enable;
        self.rtf_length_of_pieces.set_enabled(enable);
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.panel_id == PanelId::PatchTracking && self.ltf_size_of_patches_x_and_y.is_empty() {
            return Err(format!(
                "{} is required.",
                self.ltf_size_of_patches_x_and_y.get_label()
            ));
        }
        Ok(())
    }
    pub fn action<M: TiltxcorrPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        action_command: &str,
        options: Option<Run3dmodMenuOptions>,
    ) {
        match action_command {
            "Tiltxcorr" if self.panel_id == PanelId::CrossCorrelation => {
                manager.pre_cross_correlate(self.axis_id, self.dialog_type)
            }
            "Tiltxcorr" | "Imodchopconts" => {
                let run = action_command == "Tiltxcorr";
                if !run || self.validate().is_ok() {
                    manager.tiltxcorr(
                        self.axis_id,
                        self.dialog_type,
                        run,
                        self.ctf_length_of_pieces_minimum_overlap.is_selected(),
                        options,
                    );
                }
            }
            "Open Tracked Patches" => {
                manager.imod_model("PREALIGNED_STACK", "FIDUCIAL_MODEL", self.axis_id, options)
            }
            "Create Boundary Model" => manager.imod_model(
                "PREALIGNED_STACK",
                "PATCH_TRACKING_BOUNDARY_MODEL",
                self.axis_id,
                options,
            ),
            _ => self.update_panel(),
        }
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<M: TiltxcorrPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        action_command: &str,
    ) {
        self.action(manager, action_command, None);
    }
    fn set_tool_tip_text(&mut self) {
        self.layout.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_tracking_validation_and_enablement_follow_source() {
        let mut p =
            TiltxcorrPanel::get_patch_tracking_instance(AxisID::First, DialogType::FiducialModel);
        assert!(p.validate().is_err());
        p.ltf_size_of_patches_x_and_y.set_text("128,128");
        assert!(p.validate().is_ok());
        p.cb_cumulative_correlation.set_selected(true);
        p.cb_no_cosine_stretch.set_selected(false);
        p.update_panel();
        assert!(p.layout.absolute_cosine_stretch_enabled);
        p.cb_boundary_model.set_selected(true);
        p.ctf_length_of_pieces_minimum_overlap.set_selected(true);
        p.update_panel();
        assert!(p.layout.boundary_model_enabled && p.layout.length_of_pieces_enabled);
    }
    #[test]
    fn advanced_visibility_keeps_mag_change_exception() {
        let mut p = TiltxcorrPanel::get_cross_correlation_instance(
            AxisID::Only,
            DialogType::CoarseAlignment,
            true,
        );
        p.update_advanced(false);
        assert!(!p.layout.advanced_visible);
        assert!(p.layout.mag_changes_visible);
    }
}
