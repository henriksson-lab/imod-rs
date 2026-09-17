//! `IMOD/Etomo/src/etomo/ui/swing/TomogramPositioningDialog.java`.
//!
//! Process launching, comscript parameter objects, metadata persistence and
//! Swing painting are named collaborator boundaries.  The dialog-owned state,
//! display transitions, parameter forwarding order, and `CalcPanel` logic are
//! translated here.
#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleType {
    None,
    PlasticSection,
    Cryo,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewType {
    Montage,
    Single,
}

/// Java `CheckBox` / process-button state used by this dialog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Control {
    pub text: String,
    pub selected: bool,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub tooltip: Option<String>,
    pub listener_count: usize,
}
impl Control {
    fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            selected: false,
            enabled: true,
            editable: true,
            visible: true,
            tooltip: None,
            listener_count: 0,
        }
    }
}
/// Java `LabeledTextField` / `TextField` state used by this dialog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Field {
    pub text: String,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub tooltip: Option<String>,
}
impl Field {
    fn new() -> Self {
        Self {
            text: String::new(),
            enabled: true,
            editable: true,
            visible: true,
            tooltip: None,
        }
    }
}

/// Calls to Java `TomogramPositioningExpert`.
pub trait TomogramPositioningExpert {
    fn sample_action(&mut self, deferred: bool);
    fn tomopitch(&mut self);
    fn final_align(&mut self);
    fn fiducialess_action(&mut self);
    fn create_boundary(&mut self);
    fn done_dialog(&mut self);
}
/// Java `TomogramState` values inspected by `updateDisplay` and parameter reads.
pub trait TomogramStateBoundary {
    fn sample_fiducialess(&self) -> Option<bool>;
    fn sample_angle_offset(&self) -> String;
    fn sample_axis_z_shift(&self) -> String;
    fn sample_x_axis_tilt(&self) -> String;
}
/// The source's multiple comscript objects are deliberately separate traits.
pub trait TiltalignParamBoundary {
    fn set_angle_offset(&mut self, value: String);
    fn set_axis_z_shift(&mut self, value: String);
}
pub trait TomopitchParamBoundary {
    fn set_scale_factor(&mut self, value: bool);
    fn set_extra_thickness(&mut self, value: String);
    fn set_no_x_axis_tilt(&mut self, value: bool);
    fn set_angle_offset_old(&mut self, value: String);
    fn set_z_shift_old(&mut self, value: String);
    fn set_x_axis_tilt_old(&mut self, value: String);
}
pub trait MakecomfileParamBoundary {
    fn set_thickness_to_make(&mut self, value: String);
}
pub trait TiltParamBoundary {
    fn set_use_gpu(&mut self, value: bool);
    fn set_fiducialess(&mut self, value: bool);
    fn set_z_shift(&mut self, value: String);
    fn set_tilt_angle_offset(&mut self, value: String);
    fn set_x_axis_tilt(&mut self, value: String);
    fn set_image_binned(&mut self, value: i32);
    fn set_thickness(&mut self, value: String);
    fn reset_subset_start(&mut self);
}
pub trait CryoPositionParamBoundary {
    fn set_find_beads_in_volume(&mut self, value: bool);
    fn set_bead_size(&mut self, value: String);
    fn reset_bead_size(&mut self);
    fn set_thickness_of_tomograms(&mut self, value: String);
}
pub trait MetaDataBoundary {
    fn whole_tomogram_sample(&self) -> bool;
    fn set_whole_tomogram_sample(&mut self, value: bool);
    fn set_pos_binning(&mut self, value: i32);
    fn set_sample_type(&mut self, value: SampleType);
    fn set_sample_thickness(&mut self, value: String);
    fn set_has_gold_beads(&mut self, value: bool);
    fn set_positioning_bead_size(&mut self, value: String);
    fn set_extra_thickness(&mut self, value: String);
    fn set_extra_thickness_cryo(&mut self, value: String);
}
/// Java `ConstMetaData` values reached by `setParameters(ConstMetaData)` and
/// `setFiducialess(ConstMetaData)`.
pub trait ConstMetaDataBoundary {
    fn default_gpu_processing(&self) -> bool;
    fn pos_binning(&self) -> i32;
    fn sample_thickness(&self) -> String;
    fn sample_type(&self) -> SampleType;
    fn positioning_new_dialog(&self) -> bool;
    fn fiducial_diameter_available(&self) -> bool;
    fn has_gold_beads(&self) -> bool;
    fn positioning_bead_size(&self) -> Option<String>;
    fn extra_thickness(&self) -> String;
    fn extra_thickness_cryo(&self) -> String;
    fn fiducialess_alignment(&self) -> bool;
}
/// Java `ConstTiltParam`, `ConstTiltalignParam`, and `ConstTomopitchParam`
/// accessor boundaries.
pub trait ConstTiltParamBoundary {
    fn use_gpu(&self) -> bool;
    fn x_axis_tilt(&self) -> String;
    fn thickness(&self) -> String;
    fn tilt_angle_offset(&self) -> String;
    fn z_shift(&self) -> String;
}
pub trait ConstTiltalignParamBoundary {
    fn angle_offset(&self) -> String;
    fn axis_z_shift(&self) -> String;
}
pub trait ConstTomopitchParamBoundary {
    fn extra_thickness(&self) -> Option<String>;
    fn no_x_axis_tilt(&self) -> bool;
}
pub trait TomopitchLogBoundary {
    fn angle_offset_original(&self) -> Option<String>;
    fn angle_offset_added(&self) -> Option<String>;
    fn angle_offset_total(&self) -> Option<String>;
    fn axis_z_shift_original(&self) -> Option<String>;
    fn axis_z_shift_added(&self) -> Option<String>;
    fn axis_z_shift_total(&self) -> Option<String>;
    fn x_axis_tilt_original(&self) -> Option<String>;
    fn x_axis_tilt_added(&self) -> Option<String>;
    fn x_axis_tilt_total(&self) -> Option<String>;
    fn thickness(&self) -> Option<String>;
}

/// Java public static final nested `CalcPanel`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalcPanel {
    pub label: String,
    pub original: Field,
    pub added: Field,
    pub total: Field,
    pub more: bool,
    pub visible: bool,
    pub enabled: bool,
    pub tooltip: Option<String>,
}
impl CalcPanel {
    pub const ADDED_KEY: &'static str = "Added";
    pub const MAX_DIGITS: i32 = 6;
    pub fn new(label: impl Into<String>) -> Self {
        let mut original = Field::new();
        original.editable = false;
        original.text = "0.0".into();
        let mut added = Field::new();
        added.editable = false;
        added.text = "0.0".into();
        let mut total = Field::new();
        total.text = "0.0".into();
        let mut value = Self {
            label: label.into(),
            original,
            added,
            total,
            more: true,
            visible: true,
            enabled: true,
            tooltip: None,
        };
        value.update_display(false);
        value
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.total.enabled = enabled;
    }
    pub fn set_tool_tip_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.tooltip = Some(text.clone());
        self.original.tooltip = Some(text.clone());
        self.added.tooltip = Some(text.clone());
        self.total.tooltip = Some(text);
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn set(&mut self, total: Option<&str>) {
        self.update_display(false);
        self.set_number(total, false);
    }
    pub fn set_double(&mut self, total: f64) {
        self.set(Some(&total.to_string()));
    }
    pub fn set_three(
        &mut self,
        original: Option<&str>,
        added: Option<&str>,
        total: Option<&str>,
    ) -> bool {
        if total.is_none_or(str::is_empty) {
            return false;
        }
        self.update_display(true);
        self.set_number(original, true);
        self.set_number(added, true);
        self.set_number(total, false);
        true
    }
    fn set_number(&mut self, number: Option<&str>, original_or_added: bool) {
        let value = number.filter(|v| !v.is_empty()).unwrap_or("0.0").to_owned();
        if original_or_added {
            if self.original.text == "0.0" {
                self.original.text = value;
            } else {
                self.added.text = value;
            }
        } else {
            self.total.text = value;
        }
    }
    pub fn update_display(&mut self, more: bool) {
        if self.more != more {
            self.more = more;
            self.original.visible = more;
            self.added.visible = more;
        }
    }
    pub fn get_original(&self, _do_validation: bool) -> Result<String, ()> {
        Ok(self.original.text.clone())
    }
    pub fn get_total(&self, _do_validation: bool) -> Result<String, ()> {
        Ok(self.total.text.clone())
    }
    pub fn get_total_unvalidated(&self) -> String {
        self.total.text.clone()
    }
}

/// Java final `TomogramPositioningDialog`.
pub struct TomogramPositioningDialog<S: TomogramStateBoundary, E: TomogramPositioningExpert> {
    pub ltf_sample_thickness: Field,
    pub ltf_extra_thickness: Field,
    pub ltf_thickness: Field,
    pub cb_fiducialess: Control,
    pub ltf_rotation: Field,
    pub spin_binning: i32,
    pub cb_whole_tomogram: Control,
    pub btn_create_boundary: Control,
    pub pnl_final_align_visible: bool,
    pub cp_angle_offset: CalcPanel,
    pub cp_tilt_axis_z_shift: CalcPanel,
    pub cp_x_axis_tilt: CalcPanel,
    pub cp_tilt_angle_offset: CalcPanel,
    pub cp_z_shift: CalcPanel,
    pub cb_use_gpu: Control,
    pub cb_sample_type_auto: Control,
    pub cb_sample_type_cryo: Control,
    pub cb_has_gold_beads: Control,
    pub tf_bead_size: Field,
    pub l_bead_size_enabled: bool,
    pub ltf_extra_thickness_cryo: Field,
    pub cb_no_x_axis_tilt: Control,
    pub btn_sample: Control,
    pub btn_tomopitch: Control,
    pub btn_align: Control,
    pub expert: E,
    pub state: S,
    pub view_type: ViewType,
    pub displayed: bool,
    pub root_mouse_listener_count: usize,
    pub packed: bool,
}
impl<S: TomogramStateBoundary, E: TomogramPositioningExpert> TomogramPositioningDialog<S, E> {
    pub const SAMPLE_TOMOGRAMS_LABEL: &'static str = "Create Sample Tomograms";
    pub const SAMPLE_TOMOGRAMS_TOOLTIP: &'static str =
        "Build 3 sample tomograms for finding location and angles of section.";
    pub const HAS_GOLD_BEADS_LABEL: &'static str = "Sample has gold beads of size:";
    pub const CREATE_BOUNDARY_LABEL: &'static str = "Create Boundary Model";
    pub const EXTRA_THICKNESS_LABEL: &'static str = "Added border thickness (unbinned): ";
    /// Java private constructor, after manager/display-factory collaborator construction.
    pub fn new(expert: E, state: S, view_type: ViewType) -> Self {
        let mut dialog = Self {
            ltf_sample_thickness: Field::new(),
            ltf_extra_thickness: Field::new(),
            ltf_thickness: Field::new(),
            cb_fiducialess: Control::new("Coarse alignment only"),
            ltf_rotation: Field::new(),
            spin_binning: 3,
            cb_whole_tomogram: Control::new("Use whole tomogram"),
            btn_create_boundary: Control::new(Self::CREATE_BOUNDARY_LABEL),
            pnl_final_align_visible: true,
            cp_angle_offset: CalcPanel::new("Angle offset"),
            cp_tilt_axis_z_shift: CalcPanel::new("Z shift"),
            cp_x_axis_tilt: CalcPanel::new("X axis tilt"),
            cp_tilt_angle_offset: CalcPanel::new("Tilt angle offset"),
            cp_z_shift: CalcPanel::new("Z shift"),
            cb_use_gpu: Control::new("Use the GPU"),
            cb_sample_type_auto: Control::new("Find boundary model automatically"),
            cb_sample_type_cryo: Control::new("Do positioning for cryo sample"),
            cb_has_gold_beads: Control::new(Self::HAS_GOLD_BEADS_LABEL),
            tf_bead_size: Field::new(),
            l_bead_size_enabled: true,
            ltf_extra_thickness_cryo: Field::new(),
            cb_no_x_axis_tilt: Control::new("Keep X-axis tilt at zero"),
            btn_sample: Control::new(Self::SAMPLE_TOMOGRAMS_LABEL),
            btn_tomopitch: Control::new("Compute Pitch"),
            btn_align: Control::new("Final Alignment"),
            expert,
            state,
            view_type,
            displayed: true,
            root_mouse_listener_count: 0,
            packed: true,
        };
        dialog.set_tool_tip_text();
        dialog
    }
    /// Java static `getInstance`.
    pub fn get_instance(expert: E, state: S, view_type: ViewType) -> Self {
        let mut instance = Self::new(expert, state, view_type);
        instance.add_listeners();
        instance
    }
    pub fn add_listeners(&mut self) {
        self.cb_fiducialess.listener_count += 1;
        self.cb_whole_tomogram.listener_count += 1;
        self.btn_sample.listener_count += 1;
        self.btn_create_boundary.listener_count += 1;
        self.btn_tomopitch.listener_count += 1;
        self.btn_align.listener_count += 1;
        self.cb_sample_type_auto.listener_count += 1;
        self.cb_sample_type_cryo.listener_count += 1;
        self.cb_has_gold_beads.listener_count += 1;
        self.root_mouse_listener_count += 1;
    }
    pub fn is_fiducialess(&self) -> bool {
        self.cb_fiducialess.selected
    }
    pub fn set_image_rotation(&mut self, tilt_axis_angle: impl Into<String>) {
        self.ltf_rotation.text = tilt_axis_angle.into();
    }
    pub fn get_image_rotation(&self, _do_validation: bool) -> Result<String, ()> {
        Ok(self.ltf_rotation.text.clone())
    }
    pub fn get_align_params<P: TiltalignParamBoundary, M: MetaDataBoundary>(
        &self,
        param: &mut P,
        metadata: &mut M,
        validation: bool,
    ) -> bool {
        let (Ok(angle), Ok(shift)) = (
            self.cp_angle_offset.get_total(validation),
            self.cp_tilt_axis_z_shift.get_total(validation),
        ) else {
            return false;
        };
        param.set_angle_offset(angle);
        param.set_axis_z_shift(shift);
        self.update_metadata(metadata);
        true
    }
    pub fn get_tomopitch_param<P: TomopitchParamBoundary, M: MetaDataBoundary>(
        &self,
        param: &mut P,
        metadata: &mut M,
        validation: bool,
    ) -> bool {
        param.set_scale_factor(self.cb_whole_tomogram.selected);
        let thickness = if !self.is_sample_type_cryo() {
            self.ltf_extra_thickness.text.clone()
        } else {
            self.ltf_extra_thickness_cryo.text.clone()
        };
        if validation && thickness.is_empty() {
            return false;
        }
        param.set_extra_thickness(thickness);
        param.set_no_x_axis_tilt(self.cb_no_x_axis_tilt.selected);
        param.set_angle_offset_old(self.state.sample_angle_offset());
        param.set_z_shift_old(self.state.sample_axis_z_shift());
        param.set_x_axis_tilt_old(self.state.sample_x_axis_tilt());
        self.update_metadata(metadata);
        true
    }
    pub fn get_parameters_makecomfile<P: MakecomfileParamBoundary>(
        &self,
        param: &mut P,
        validation: bool,
    ) -> bool {
        if validation && self.ltf_sample_thickness.text.is_empty() {
            return false;
        }
        param.set_thickness_to_make(self.ltf_sample_thickness.text.clone());
        true
    }
    pub fn get_tilt_params_for_sample<P: TiltParamBoundary>(
        &self,
        param: &mut P,
        validation: bool,
    ) -> bool {
        if validation && self.ltf_sample_thickness.text.is_empty() {
            return false;
        }
        param.set_thickness(self.ltf_sample_thickness.text.clone());
        true
    }
    pub fn is_whole_tomogram(&self) -> bool {
        self.cb_whole_tomogram.selected
    }
    pub fn update_metadata<M: MetaDataBoundary>(&self, metadata: &mut M) {
        let whole = self.is_whole_tomogram();
        if whole != metadata.whole_tomogram_sample() {
            metadata.set_whole_tomogram_sample(whole);
        }
        metadata.set_pos_binning(self.spin_binning);
    }
    pub fn get_binning(&self) -> i32 {
        if !self.is_whole_tomogram() {
            1
        } else {
            self.spin_binning
        }
    }
    pub fn get_tilt_params<P: TiltParamBoundary, M: MetaDataBoundary>(
        &self,
        param: &mut P,
        metadata: &mut M,
        validation: bool,
    ) -> bool {
        param.set_use_gpu(self.cb_use_gpu.enabled && self.cb_use_gpu.selected);
        let fiducialess = self.is_fiducialess();
        param.set_fiducialess(fiducialess);
        if fiducialess {
            let (Ok(z), Ok(angle)) = (
                self.cp_z_shift.get_total(validation),
                self.cp_tilt_angle_offset.get_total(validation),
            ) else {
                return false;
            };
            param.set_z_shift(z);
            param.set_tilt_angle_offset(angle);
        }
        if self.is_sample_type_cryo() {
            param.set_x_axis_tilt("0".into());
        } else if let Ok(value) = self.cp_x_axis_tilt.get_total(validation) {
            param.set_x_axis_tilt(value);
        } else {
            return false;
        }
        param.set_image_binned(self.get_binning());
        if validation && self.ltf_thickness.text.is_empty() {
            return false;
        }
        param.set_thickness(self.ltf_thickness.text.clone());
        self.update_metadata(metadata);
        true
    }
    pub fn get_sample_type(&self) -> SampleType {
        if !self.cb_sample_type_auto.selected {
            SampleType::None
        } else if self.cb_sample_type_cryo.selected {
            SampleType::Cryo
        } else {
            SampleType::PlasticSection
        }
    }
    pub fn set_sample_type(&mut self, input: SampleType) {
        self.cb_sample_type_auto.selected =
            input == SampleType::PlasticSection || input == SampleType::Cryo;
        self.cb_sample_type_cryo.selected = input == SampleType::Cryo;
        self.update_display();
    }
    pub fn update_display(&mut self) {
        let enable = self
            .state
            .sample_fiducialess()
            .is_none_or(|value| value == self.is_fiducialess());
        let fiducialess = self.is_fiducialess();
        self.btn_tomopitch.enabled = enable;
        self.cp_angle_offset.set_enabled(enable && !fiducialess);
        self.cp_tilt_axis_z_shift
            .set_enabled(enable && !fiducialess);
        self.btn_align.enabled = enable && !fiducialess;
        self.cp_tilt_angle_offset.set_enabled(enable);
        self.cp_z_shift.set_enabled(enable);
        self.cp_x_axis_tilt.set_enabled(enable);
        self.ltf_thickness.enabled = enable;
        self.ltf_rotation.enabled = fiducialess;
        self.pnl_final_align_visible = !fiducialess;
        self.cp_tilt_angle_offset.set_visible(fiducialess);
        self.cp_z_shift.set_visible(fiducialess);
        let auto = self.cb_sample_type_auto.selected;
        self.cb_sample_type_cryo.enabled = auto && self.view_type != ViewType::Montage;
        let cryo = self.is_sample_type_cryo();
        self.cb_has_gold_beads.enabled = cryo;
        self.tf_bead_size.enabled = cryo;
        self.l_bead_size_enabled = cryo;
        self.ltf_extra_thickness.visible = !cryo;
        self.ltf_extra_thickness_cryo.visible = cryo;
        self.cb_whole_tomogram.editable = !cryo;
        if cryo {
            self.cb_whole_tomogram.selected = true;
        }
        let whole = self.cb_whole_tomogram.selected;
        self.spin_binning = if !cryo && whole {
            self.spin_binning
        } else {
            self.spin_binning
        };
        if !auto {
            self.btn_create_boundary.text = Self::CREATE_BOUNDARY_LABEL.into();
            if whole {
                self.btn_sample.text = "Create Whole Tomogram".into();
                self.btn_sample.tooltip =
                    Some("Create whole tomogram for drawing positioning model.".into());
            } else {
                self.btn_sample.text = Self::SAMPLE_TOMOGRAMS_LABEL.into();
                self.btn_sample.tooltip = Some(Self::SAMPLE_TOMOGRAMS_TOOLTIP.into());
            }
        } else {
            self.btn_create_boundary.text = "View Boundary Model".into();
            if !cryo && !whole {
                self.btn_sample.text = "Create Samples & Boundary Model".into();
            } else if !cryo {
                self.btn_sample.text = "Create Tomogram & Boundary Model".into();
            } else {
                self.btn_sample.text = "Find Boundary Model for Cryo".into();
            }
            self.btn_sample.tooltip =
                Some("Builds a sample tomogram and creates a boundary model.".into());
        }
        let has_beads = self.cb_has_gold_beads.enabled && self.cb_has_gold_beads.selected;
        self.tf_bead_size.enabled = has_beads;
        self.l_bead_size_enabled = has_beads;
    }
    pub fn is_sample_type_auto(&self) -> bool {
        self.cb_sample_type_auto.selected
    }
    pub fn is_sample_type_cryo(&self) -> bool {
        self.cb_sample_type_cryo.enabled && self.cb_sample_type_cryo.selected
    }
    pub fn is_tomopitch_button(&self) -> bool {
        self.btn_tomopitch.selected
    }
    /// Java `setParameters(ConstMetaData)`.  GPU availability and default
    /// bead-diameter calculation remain application-manager boundaries, so
    /// callers provide their already-resolved values.
    pub fn set_parameters_metadata<C: ConstMetaDataBoundary>(
        &mut self,
        metadata: &C,
        gpu_available: bool,
        default_bead_diameter_pixels: Option<String>,
    ) {
        self.cb_use_gpu.enabled = gpu_available;
        self.cb_use_gpu.selected = metadata.default_gpu_processing();
        self.spin_binning = metadata.pos_binning();
        self.ltf_sample_thickness.text = metadata.sample_thickness();
        self.set_sample_type(metadata.sample_type());
        if metadata.positioning_new_dialog() {
            self.cb_has_gold_beads.selected = metadata.fiducial_diameter_available();
            if let Some(value) = default_bead_diameter_pixels {
                self.tf_bead_size.text = value;
            }
        } else {
            self.cb_has_gold_beads.selected = metadata.has_gold_beads();
            if let Some(value) = metadata.positioning_bead_size() {
                self.tf_bead_size.text = value;
            }
        }
        self.ltf_extra_thickness.text = metadata.extra_thickness();
        self.ltf_extra_thickness_cryo.text = metadata.extra_thickness_cryo();
    }
    pub fn get_parameters_metadata<M: MetaDataBoundary>(&self, metadata: &mut M) {
        metadata.set_sample_type(self.get_sample_type());
        metadata.set_sample_thickness(self.ltf_sample_thickness.text.clone());
        metadata.set_has_gold_beads(self.cb_has_gold_beads.selected);
        metadata.set_positioning_bead_size(self.tf_bead_size.text.clone());
        metadata.set_extra_thickness(self.ltf_extra_thickness.text.clone());
        metadata.set_extra_thickness_cryo(self.ltf_extra_thickness_cryo.text.clone());
    }
    /// Java `setParameters(CryoPositionParam)`.
    pub fn set_parameters_cryo(&mut self, bead_size: Option<String>) {
        self.cb_has_gold_beads.selected = bead_size.is_some();
        if let Some(value) = bead_size {
            self.tf_bead_size.text = value;
        }
    }
    pub fn get_parameters_cryo<P: CryoPositionParamBoundary>(
        &self,
        param: &mut P,
        validation: bool,
    ) -> bool {
        let beads = self.cb_has_gold_beads.selected;
        param.set_find_beads_in_volume(beads);
        if beads {
            if validation && self.tf_bead_size.text.is_empty() {
                return false;
            }
            param.set_bead_size(self.tf_bead_size.text.clone());
        } else {
            param.reset_bead_size();
        }
        if validation && self.ltf_sample_thickness.text.is_empty() {
            return false;
        }
        param.set_thickness_of_tomograms(self.ltf_sample_thickness.text.clone());
        true
    }
    pub fn is_align_button(&self) -> bool {
        self.btn_align.selected
    }
    pub fn is_align_button_enabled(&self) -> bool {
        self.btn_align.enabled
    }
    /// Java `setButtonState(ReconScreenState)`, after that storage unit has
    /// resolved its three keyed button states.
    pub fn set_button_state(&mut self, sample: bool, tomopitch: bool, align: bool) {
        self.btn_sample.selected = sample;
        self.btn_tomopitch.selected = tomopitch;
        self.btn_align.selected = align;
    }
    /// Java `setTiltParam(ConstTiltParam, boolean)`.
    pub fn set_tilt_param<P: ConstTiltParamBoundary>(&mut self, param: &P, initialize: bool) {
        if !initialize {
            self.cb_use_gpu.selected = param.use_gpu();
        }
        self.cp_x_axis_tilt.set(Some(&param.x_axis_tilt()));
        self.ltf_thickness.text = param.thickness();
        self.cp_tilt_angle_offset
            .set(Some(&param.tilt_angle_offset()));
        self.cp_z_shift.set(Some(&param.z_shift()));
    }
    pub fn roll_align_com_angles(&mut self) {
        self.cp_angle_offset.update_display(false);
        self.cp_tilt_axis_z_shift.update_display(false);
    }
    pub fn roll_tilt_com_angles(&mut self) {
        self.cp_x_axis_tilt.update_display(false);
    }
    /// Java `setAlignParam(ConstTiltalignParam)`.
    pub fn set_align_param<P: ConstTiltalignParamBoundary>(&mut self, param: &P) {
        self.cp_angle_offset.set(Some(&param.angle_offset()));
        self.cp_tilt_axis_z_shift.set(Some(&param.axis_z_shift()));
    }
    /// Java `setParametersFiducialess(TiltParam, MetaData)`, after metadata's
    /// axis-specific `isFiducialess` query has been supplied.
    pub fn set_parameters_fiducialess<P: TiltParamBoundary>(
        &mut self,
        param: &mut P,
        metadata_fiducialess: bool,
    ) {
        param.set_fiducialess(metadata_fiducialess);
        if self.is_fiducialess() {
            self.cp_tilt_angle_offset.set(None);
            self.cp_z_shift.set(None);
        } else {
            param.set_tilt_angle_offset(self.cp_tilt_angle_offset.get_total_unvalidated());
            param.set_z_shift(self.cp_z_shift.get_total_unvalidated());
        }
        param.reset_subset_start();
    }
    /// Java `setFiducialess(ConstMetaData)`.
    pub fn set_fiducialess<C: ConstMetaDataBoundary>(&mut self, metadata: &C) {
        self.cb_fiducialess.selected = metadata.fiducialess_alignment();
        self.update_display();
    }
    /// Java `setParameters(TomopitchLog)`.
    pub fn set_parameters_tomopitch_log<L: TomopitchLogBoundary>(&mut self, log: &L) -> bool {
        let mut missing_data = !self.cp_angle_offset.set_three(
            log.angle_offset_original().as_deref(),
            log.angle_offset_added().as_deref(),
            log.angle_offset_total().as_deref(),
        );
        missing_data = !self.cp_tilt_axis_z_shift.set_three(
            log.axis_z_shift_original().as_deref(),
            log.axis_z_shift_added().as_deref(),
            log.axis_z_shift_total().as_deref(),
        );
        missing_data = !self.cp_x_axis_tilt.set_three(
            log.x_axis_tilt_original().as_deref(),
            log.x_axis_tilt_added().as_deref(),
            log.x_axis_tilt_total().as_deref(),
        );
        self.cp_tilt_angle_offset.set_three(
            log.angle_offset_original().as_deref(),
            log.angle_offset_added().as_deref(),
            log.angle_offset_total().as_deref(),
        );
        self.cp_z_shift.set_three(
            log.axis_z_shift_original().as_deref(),
            log.axis_z_shift_added().as_deref(),
            log.axis_z_shift_total().as_deref(),
        );
        if let Some(thickness) = log.thickness() {
            self.ltf_thickness.text = thickness;
        } else {
            missing_data = true;
        }
        self.packed = true;
        !missing_data
    }
    /// Java `setTomopitchParam(ConstTomopitchParam)`.
    pub fn set_tomopitch_param<P: ConstTomopitchParamBoundary>(&mut self, param: &P) {
        if let Some(extra_thickness) = param.extra_thickness() {
            if !self.is_sample_type_cryo() {
                self.ltf_extra_thickness.text = extra_thickness;
            } else {
                self.ltf_extra_thickness_cryo.text = extra_thickness;
            }
            self.cb_no_x_axis_tilt.selected = param.no_x_axis_tilt();
        }
    }
    pub fn set_whole_tomogram(&mut self, state: bool) {
        self.cb_whole_tomogram.selected = state;
    }
    /// Java `popUpContextMenu(MouseEvent)`: actual popup construction is a GUI boundary.
    pub fn pop_up_context_menu(&self) -> ([&'static str; 6], [&'static str; 3]) {
        (
            [
                "Tomopitch",
                "Findsection",
                "Cryoposition",
                "Newstack",
                "3dmod",
                "Tilt",
            ],
            ["Tomopitch", "Sample", "Cryoposition"],
        )
    }
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action(&mut self, command: &str, deferred: bool) {
        if command == self.btn_sample.text {
            self.expert.sample_action(deferred);
        } else if command == self.btn_tomopitch.text {
            self.expert.tomopitch();
        } else if command == self.btn_align.text {
            self.expert.final_align();
        } else if command == self.cb_fiducialess.text {
            self.expert.fiducialess_action();
        } else if command == self.btn_create_boundary.text {
            self.expert.create_boundary();
        } else {
            self.update_display();
        }
    }
    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`, whose
    /// deferred-button argument is null.
    pub fn actionPerformed(&mut self, command: &str) {
        self.action(command, false);
    }
    pub fn done(&mut self) {
        self.expert.done_dialog();
        self.btn_sample.listener_count = self.btn_sample.listener_count.saturating_sub(1);
        self.btn_tomopitch.listener_count = self.btn_tomopitch.listener_count.saturating_sub(1);
        self.btn_align.listener_count = self.btn_align.listener_count.saturating_sub(1);
        self.displayed = false;
    }
    pub fn set_tool_tip_text(&mut self) {
        self.ltf_sample_thickness.tooltip = Some("Thickness of sample slices, or unbinned thickness of whole tomogram.  Make this much larger than expected section thickness to see borders of section.".into());
        self.btn_sample.tooltip = Some(Self::SAMPLE_TOMOGRAMS_TOOLTIP.into());
        self.btn_create_boundary.tooltip = Some("Open samples in 3dmod to make a model with lines along top and bottom edges of the section in each sample.".into());
        self.btn_tomopitch.tooltip = Some("Run tomopitch.  This will compute the positioning values and adjust the totals shown here.".into());
        self.cb_use_gpu.tooltip =
            Some("Check to run the tilt process on the graphics card.".into());
        self.tf_bead_size.tooltip = Some("Size of gold beads in unbinned pixels".into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct State {
        fiducialess: Option<bool>,
    }
    impl TomogramStateBoundary for State {
        fn sample_fiducialess(&self) -> Option<bool> {
            self.fiducialess
        }
        fn sample_angle_offset(&self) -> String {
            "1".into()
        }
        fn sample_axis_z_shift(&self) -> String {
            "2".into()
        }
        fn sample_x_axis_tilt(&self) -> String {
            "3".into()
        }
    }
    #[derive(Default)]
    struct Expert {
        calls: Vec<&'static str>,
    }
    impl TomogramPositioningExpert for Expert {
        fn sample_action(&mut self, _: bool) {
            self.calls.push("sample")
        }
        fn tomopitch(&mut self) {
            self.calls.push("pitch")
        }
        fn final_align(&mut self) {
            self.calls.push("align")
        }
        fn fiducialess_action(&mut self) {
            self.calls.push("fid")
        }
        fn create_boundary(&mut self) {
            self.calls.push("boundary")
        }
        fn done_dialog(&mut self) {
            self.calls.push("done")
        }
    }
    #[test]
    fn cryo_display_forces_whole_tomogram_and_controls_visibility() {
        let mut dialog = TomogramPositioningDialog::get_instance(
            Expert::default(),
            State::default(),
            ViewType::Single,
        );
        dialog.set_sample_type(SampleType::Cryo);
        assert!(dialog.cb_whole_tomogram.selected);
        assert!(!dialog.ltf_extra_thickness.visible);
        assert!(dialog.ltf_extra_thickness_cryo.visible);
    }
    #[test]
    fn fiducialess_display_disables_final_alignment_and_exposes_tilt_fields() {
        let mut dialog = TomogramPositioningDialog::get_instance(
            Expert::default(),
            State::default(),
            ViewType::Single,
        );
        dialog.cb_fiducialess.selected = true;
        dialog.update_display();
        assert!(!dialog.btn_align.enabled);
        assert!(!dialog.pnl_final_align_visible);
        assert!(dialog.cp_z_shift.visible);
    }
    #[test]
    fn done_calls_expert_removes_process_listeners_and_hides_dialog() {
        let mut dialog = TomogramPositioningDialog::get_instance(
            Expert::default(),
            State::default(),
            ViewType::Single,
        );
        dialog.done();
        assert_eq!(dialog.expert.calls, vec!["done"]);
        assert!(!dialog.displayed);
        assert_eq!(dialog.btn_align.listener_count, 0);
    }
}
