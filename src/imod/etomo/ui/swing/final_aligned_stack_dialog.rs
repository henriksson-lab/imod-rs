//! `IMOD/Etomo/src/etomo/ui/swing/FinalAlignedStackDialog.java`.
//!
//! The native panel construction, file chooser, autodoc lookup, and process
//! execution graph are GUI/application boundaries.  This source unit owns the
//! four-tab state machine, field/metadata/comscript transfer, dose-weighting
//! enablement policy, warnings, and action routing of the Java dialog.
#![allow(dead_code)]

use crate::imod::etomo::r#type::axis_id::AxisID;

pub const USE_CTF_CORRECTION_LABEL: &str = "Use CTF Correction";
pub const CTF_TAB_LABEL: &str = "Correct CTF";
pub const USE_FILTERED_STACK_LABEL: &str = "Use Filtered Stack";
pub const MTF_FILTER_TAB_LABEL: &str = "2D Filter";
pub const FINAL_ALIGNED_STACK_TAB_LABEL: &str = "Create";
pub const CTF_CORRECTION_LABEL: &str = "Correct CTF";
const MTF_FILE_LABEL: &str = "MTF file: ";
const FIXED_IMAGE_DOSE_LABEL: &str = "No dose file; use fixed dose per image: ";

/// Java private static final inner `Tab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Newst,
    CtfCorrection,
    CcdEraser,
    MtfFilter,
}
impl Tab {
    pub const DEFAULT: Self = Self::Newst;
    pub fn get_instance(index: usize) -> Self {
        match index {
            1 => Self::CtfCorrection,
            2 => Self::CcdEraser,
            3 => Self::MtfFilter,
            _ => Self::Newst,
        }
    }
    pub fn to_int(self) -> usize {
        match self {
            Self::Newst => 0,
            Self::CtfCorrection => 1,
            Self::CcdEraser => 2,
            Self::MtfFilter => 3,
        }
    }
    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Newst => "NEWST",
            Self::CtfCorrection => "CTF_CORRECTION",
            Self::CcdEraser => "CCD_ERASER",
            Self::MtfFilter => "MTF_FILTER",
        }
    }
}

/// Java private static final inner `TypeOfDoseFile`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypeOfDoseFile {
    ImageDose,
    AccumulatedAndImageDose,
    PriorAndCumulativeDose,
    MdocFile,
}
impl TypeOfDoseFile {
    pub const DEFAULT: Self = Self::MdocFile;
    pub fn get_instance(value: Option<&str>) -> Self {
        match value {
            Some("1") => Self::ImageDose,
            Some("2") => Self::AccumulatedAndImageDose,
            Some("3") => Self::PriorAndCumulativeDose,
            Some("4") => Self::MdocFile,
            _ => Self::DEFAULT,
        }
    }
    pub fn get_value(self) -> i32 {
        match self {
            Self::ImageDose => 1,
            Self::AccumulatedAndImageDose => 2,
            Self::PriorAndCumulativeDose => 3,
            Self::MdocFile => 4,
        }
    }
    pub fn get_label(self) -> &'static str {
        match self {
            Self::ImageDose => "Dose for each image",
            Self::AccumulatedAndImageDose => "Prior dose and dose of image",
            Self::PriorAndCumulativeDose => "Cumulative dose before and after image",
            Self::MdocFile => "Metadata in .mdoc file",
        }
    }
    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }
    pub fn enable_bidirectional_num_views(self) -> bool {
        matches!(self, Self::ImageDose | Self::MdocFile)
    }
}

/// Values passed from/to Java `MetaData`; external model ownership is explicit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FinalAlignedStackMetaData {
    pub dialog_saved: bool,
    pub erase_beads_initialized: bool,
    pub ctf_correction_parallel: bool,
    pub use_x_axis_tilt: bool,
    pub x_axis_tilt: String,
    pub scale_by_ctf_power: String,
    pub low_pass_radius_sigma: String,
    pub mtf_file: String,
    pub maximum_inverse: String,
    pub inverse_rolloff_radius_sigma: String,
    pub use_fixed_image_dose: bool,
    pub fixed_image_dose: String,
    pub dose_weighting_file: String,
    pub type_of_dose_file: String,
    pub voltage_200: bool,
    pub optimal_dose_scaling: String,
    pub bidirectional_num_views: String,
    pub ctf3d_setup_slab_thickness_set: bool,
}

/// Direct `MTFFilterParam` values used by the source dialog.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MtfFilterParameters {
    pub low_pass_radius_sigma: Option<String>,
    pub mtf_file: Option<String>,
    pub maximum_inverse: Option<String>,
    pub inverse_rolloff_radius_sigma: Option<String>,
    pub fixed_image_dose: Option<String>,
    pub type_of_dose_file: Option<TypeOfDoseFile>,
    pub dose_weighting_file: Option<String>,
    pub voltage_200: Option<bool>,
    pub optimal_dose_scaling: Option<String>,
    pub bidirectional_num_views: Option<String>,
    pub starting_and_ending_z: String,
}

/// The direct execution/application calls in `action`, `done`, and `changeTab`.
pub trait FinalAlignedStackDialogExpert {
    fn mtffilter(&mut self, deferred: bool);
    fn use_mtf_filter(&mut self);
    fn ctf_plotter(&mut self);
    fn ctf_correction(&mut self, deferred: bool, parallel: bool);
    fn use_ctf_correction(&mut self);
    fn enable_use_filter(&mut self);
    fn done_dialog(&mut self);
}
pub trait FinalAlignedStackDialogApplication {
    fn imod_mtf_filter(&mut self, axis_id: AxisID);
    fn imod_ctf_correction(&mut self, axis_id: AxisID);
    fn warning(&mut self, title: &str, message: &str, axis_id: AxisID);
    fn pack(&mut self, axis_id: AxisID);
    fn move_sub_frame(&mut self);
}

/// The complete source-owned dialog state. `*_enabled`, visibility fields and
/// `mounted_tab` are the corresponding Swing component state at the GUI edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalAlignedStackDialog {
    pub axis_id: AxisID,
    pub cur_tab: Tab,
    pub selected_tab: usize,
    pub mounted_tab: Option<Tab>,
    pub trial_tilt: bool,
    pub processing_method_locked: bool,
    pub valid_autodoc: bool,
    pub erase_beads_initialized: bool,
    pub dialog_saved: bool,
    pub montage: bool,
    pub fiducialess: bool,
    pub advanced: bool,
    pub filter_header_open: bool,
    pub ctf_header_open: bool,
    pub voltage: String,
    pub spherical_aberration: String,
    pub invert_tilt_angles: bool,
    pub amplitude_contrast: String,
    pub scan_defocus_range: String,
    pub expected_defocus: String,
    pub phase_shift_in_degrees: String,
    pub offset_to_add: String,
    pub interpolation_width: String,
    pub defocus_tol: String,
    pub config_file: String,
    pub use_expected_defocus: bool,
    pub x_axis_tilt: String,
    pub use_x_axis_tilt: bool,
    pub scale_by_ctf_power: String,
    pub use_scale_by_ctf_power: bool,
    pub minimum_zero_spacing: String,
    pub low_pass_radius_sigma: String,
    pub mtf_file: String,
    pub maximum_inverse: String,
    pub inverse_rolloff_radius_sigma: String,
    pub starting_and_ending_z: String,
    pub dose_weight_filtering: bool,
    pub fixed_image_dose_selected: bool,
    pub fixed_image_dose: String,
    pub dose_weighting_file: String,
    pub type_of_dose_file: TypeOfDoseFile,
    pub voltage_200: bool,
    pub optimal_dose_scaling: String,
    pub bidirectional_num_views: String,
    pub filter_enabled: bool,
    pub view_filter_enabled: bool,
    pub use_filter_enabled: bool,
    pub ctf_correction_enabled: bool,
    pub view_ctf_correction_enabled: bool,
    pub use_ctf_correction_enabled: bool,
    pub ctf_plotter_enabled: bool,
    pub config_file_enabled: bool,
    pub uniform_filtering_enabled: bool,
    pub inverse_params_enabled: bool,
    pub dose_weighting_file_enabled: bool,
    pub type_of_dose_file_enabled: bool,
    pub fixed_image_dose_enabled: bool,
    pub bidirectional_num_views_enabled: bool,
    pub advanced_filter_visible: bool,
    pub advanced_ctf_visible: bool,
    pub ctf3d_message_visible: bool,
    pub ctf3d_filter_message_visible: bool,
    pub action_listeners_installed: bool,
    pub displayed: bool,
    /// Source Swing insertion order for the four lazy-mounted tab panels.
    pub tab_components: [Vec<String>; 4],
    pub image_rotation: String,
    pub aligned_stack_binning_updated: bool,
    pub mediator_reregistered: bool,
    pub processing_method: String,
}

impl FinalAlignedStackDialog {
    pub fn get_instance(axis_id: AxisID, montage: bool, cur_tab: Tab) -> Self {
        let mut instance = Self {
            axis_id,
            cur_tab,
            selected_tab: cur_tab.to_int(),
            mounted_tab: Some(cur_tab),
            montage,
            trial_tilt: false,
            processing_method_locked: false,
            valid_autodoc: false,
            erase_beads_initialized: false,
            dialog_saved: false,
            fiducialess: false,
            advanced: false,
            filter_header_open: false,
            ctf_header_open: false,
            voltage: String::new(),
            spherical_aberration: String::new(),
            invert_tilt_angles: false,
            amplitude_contrast: String::new(),
            scan_defocus_range: String::new(),
            expected_defocus: String::new(),
            phase_shift_in_degrees: String::new(),
            offset_to_add: String::new(),
            interpolation_width: String::new(),
            defocus_tol: String::new(),
            config_file: String::new(),
            use_expected_defocus: false,
            x_axis_tilt: String::new(),
            use_x_axis_tilt: false,
            scale_by_ctf_power: String::new(),
            use_scale_by_ctf_power: false,
            minimum_zero_spacing: String::new(),
            low_pass_radius_sigma: String::new(),
            mtf_file: String::new(),
            maximum_inverse: String::new(),
            inverse_rolloff_radius_sigma: String::new(),
            starting_and_ending_z: String::new(),
            dose_weight_filtering: false,
            fixed_image_dose_selected: false,
            fixed_image_dose: String::new(),
            dose_weighting_file: String::new(),
            type_of_dose_file: TypeOfDoseFile::DEFAULT,
            voltage_200: false,
            optimal_dose_scaling: String::new(),
            bidirectional_num_views: String::new(),
            filter_enabled: true,
            view_filter_enabled: true,
            use_filter_enabled: true,
            ctf_correction_enabled: true,
            view_ctf_correction_enabled: true,
            use_ctf_correction_enabled: true,
            ctf_plotter_enabled: true,
            config_file_enabled: true,
            uniform_filtering_enabled: true,
            inverse_params_enabled: true,
            dose_weighting_file_enabled: false,
            type_of_dose_file_enabled: false,
            fixed_image_dose_enabled: false,
            bidirectional_num_views_enabled: false,
            advanced_filter_visible: false,
            advanced_ctf_visible: false,
            ctf3d_message_visible: false,
            ctf3d_filter_message_visible: false,
            action_listeners_installed: false,
            displayed: true,
            tab_components: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            image_rotation: String::new(),
            aligned_stack_binning_updated: false,
            mediator_reregistered: false,
            processing_method: "Local".into(),
        };
        instance.layout_newst_panel();
        instance.layout_ctf_correction_panel();
        instance.layout_ccd_eraser();
        instance.layout_filter_panel();
        instance.add_listeners();
        instance.update_advanced();
        instance.update_display();
        instance
    }
    pub fn add_listeners(&mut self) {
        self.action_listeners_installed = true;
    }
    pub fn get_tilt3d_find_button_label() -> &'static str {
        "Find 3D Beads"
    }
    pub fn get_reproject_model_button_label() -> &'static str {
        "Reproject Model"
    }
    pub fn get_use_erased_stack_label() -> &'static str {
        "Use Erased Stack"
    }
    pub fn get_erased_stack_tab_label() -> &'static str {
        "Erase Beads"
    }
    pub fn is_fiducialess(&self) -> bool {
        self.fiducialess
    }
    pub fn set_filter_button_enabled(&mut self, enable: bool) {
        self.filter_enabled = enable;
    }
    pub fn set_view_filter_button_enabled(&mut self, enable: bool) {
        self.view_filter_enabled = enable;
    }
    pub fn set_voltage(&mut self, input: impl ToString) {
        self.voltage = input.to_string();
    }
    pub fn get_config_file(&self) -> &str {
        &self.config_file
    }
    pub fn set_config_file(&mut self, input: impl Into<String>) {
        self.config_file = input.into();
    }
    pub fn set_spherical_aberration(&mut self, input: impl ToString) {
        self.spherical_aberration = input.to_string();
    }
    pub fn set_invert_tilt_angles(&mut self, input: bool) {
        self.invert_tilt_angles = input;
    }
    pub fn set_amplitude_contrast(&mut self, input: impl ToString) {
        self.amplitude_contrast = input.to_string();
    }
    pub fn set_defocus_tol(&mut self, input: impl ToString) {
        self.defocus_tol = input.to_string();
    }
    pub fn set_scan_defocus_range(&mut self, input: impl Into<String>) {
        self.scan_defocus_range = input.into();
    }
    pub fn set_expected_defocus(&mut self, input: impl ToString) {
        self.expected_defocus = input.to_string();
    }
    pub fn set_phase_shift_in_degrees(&mut self, input: impl Into<String>) {
        self.phase_shift_in_degrees = input.into();
    }
    pub fn set_offset_to_add(&mut self, input: impl ToString) {
        self.offset_to_add = input.to_string();
    }
    pub fn get_defocus_tol(&self, _: bool) -> &str {
        &self.defocus_tol
    }
    pub fn get_scan_defocus_range(&self, _: bool) -> &str {
        &self.scan_defocus_range
    }
    pub fn get_expected_defocus(&self, _: bool) -> &str {
        &self.expected_defocus
    }
    pub fn get_phase_shift_in_degrees(&self, _: bool) -> &str {
        &self.phase_shift_in_degrees
    }
    pub fn get_offset_to_add(&self, _: bool) -> &str {
        &self.offset_to_add
    }
    pub fn set_use_filter_enabled(&mut self, enable: bool) {
        self.use_filter_enabled = enable;
    }
    pub fn set_interpolation_width(&mut self, input: impl ToString) {
        self.interpolation_width = input.to_string();
    }
    pub fn set_fiducialess_alignment(&mut self, input: bool) {
        self.fiducialess = input;
    }
    pub fn set_image_rotation(&mut self, input: impl Into<String>) {
        self.image_rotation = input.into();
    }
    pub fn set_filter_header_state(&mut self, state: bool) {
        self.filter_header_open = state;
    }
    pub fn set_ctf_correction_header_state(&mut self, state: bool) {
        self.ctf_header_open = state;
    }
    pub fn get_cur_tab(&self) -> Tab {
        self.cur_tab
    }
    pub fn get_interpolation_width(&self, _: bool) -> &str {
        &self.interpolation_width
    }
    pub fn get_starting_and_ending_z(&self) -> &str {
        &self.starting_and_ending_z
    }
    pub fn get_ctf_phase_flip_x_axis_tilt(&self, _: bool) -> Option<&str> {
        self.use_x_axis_tilt.then_some(self.x_axis_tilt.as_str())
    }
    pub fn get_scale_by_ctf_power(&self, _: bool) -> Option<&str> {
        self.use_scale_by_ctf_power
            .then_some(self.scale_by_ctf_power.as_str())
    }
    pub fn get_minimum_zero_spacing(&self, _: bool) -> &str {
        &self.minimum_zero_spacing
    }
    pub fn set_ctf_phase_flip_x_axis_tilt(
        &mut self,
        x_axis_tilt: impl Into<String>,
        from_tilt_com: bool,
    ) {
        let value = x_axis_tilt.into();
        if !from_tilt_com && !value.is_empty() {
            self.use_x_axis_tilt = true;
        }
        if !value.is_empty() {
            self.x_axis_tilt = value;
        }
    }
    pub fn set_scale_by_ctf_power(&mut self, input: impl Into<String>) {
        let value = input.into();
        self.use_scale_by_ctf_power = !value.is_empty();
        if self.use_scale_by_ctf_power {
            self.scale_by_ctf_power = value;
        }
    }
    pub fn set_minimum_zero_spacing(&mut self, input: impl Into<String>) {
        self.minimum_zero_spacing = input.into();
    }
    pub fn set_use_expected_defocus(&mut self, input: bool) {
        self.use_expected_defocus = input;
        self.update_ctf_plotter();
    }
    pub fn get_voltage(&self, _: bool) -> &str {
        &self.voltage
    }
    pub fn get_spherical_aberration(&self, _: bool) -> &str {
        &self.spherical_aberration
    }
    pub fn get_invert_tilt_angles(&self) -> bool {
        self.invert_tilt_angles
    }
    pub fn get_amplitude_contrast(&self, _: bool) -> &str {
        &self.amplitude_contrast
    }
    pub fn get_filter_header_state(&self) -> bool {
        self.filter_header_open
    }
    pub fn get_ctf_correction_header_state(&self) -> bool {
        self.ctf_header_open
    }
    pub fn set_starting_and_ending_z(&mut self, input: impl Into<String>) {
        self.starting_and_ending_z = input.into();
    }
    pub fn expand_global(&mut self) {}
    pub fn expand(&mut self, filter: bool, expanded: bool) {
        if filter {
            self.update_advanced_filter(expanded);
        } else {
            self.update_advanced_ctf_correction(expanded);
        }
    }
    pub fn update_advanced_filter(&mut self, advanced: bool) {
        self.advanced_filter_visible = advanced;
    }
    pub fn update_advanced_ctf_correction(&mut self, advanced: bool) {
        self.advanced_ctf_visible = advanced;
    }
    pub fn update_aligned_stack_binning(&mut self) {
        self.aligned_stack_binning_updated = true;
    }
    pub fn is_parallel_process(&self) -> bool {
        self.processing_method_locked
    }
    pub fn is_use_expected_defocus(&self) -> bool {
        self.use_expected_defocus
    }
    pub fn is_ctf_phase_flip_x_axis_tilt_empty(&self) -> bool {
        self.x_axis_tilt.is_empty()
    }
    pub fn set_tilt_com_parameters(&mut self, x_axis_tilt: impl Into<String>) {
        self.set_ctf_phase_flip_x_axis_tilt(x_axis_tilt, true);
    }
    pub fn layout_ccd_eraser(&mut self) {
        self.tab_components[Tab::CcdEraser.to_int()] = vec!["EraseGoldPanel".into()];
    }
    pub fn layout_ctf_correction_panel(&mut self) {
        self.tab_components[Tab::CtfCorrection.to_int()] = vec![
            "PanelHeader:CTF Correction".into(),
            "Voltage".into(),
            "SphericalAberration".into(),
            "AmplitudeContrast".into(),
            "InvertTiltAngles".into(),
            "CTF Plotter".into(),
            "CTF Phase Flip".into(),
            "CpuGpuPanel".into(),
            "InterpolationWidth".into(),
            "CtfPhaseFlipXAxisTilt".into(),
            "ScaleByCtfPower".into(),
            "DefocusTol".into(),
            "MinimumZeroSpacing".into(),
            "CtfCorrectionButtons".into(),
        ];
    }
    pub fn layout_newst_panel(&mut self) {
        self.tab_components[Tab::Newst.to_int()] = vec!["NewstackOrBlendmontPanel".into()];
    }
    pub fn layout_filter_panel(&mut self) {
        self.tab_components[Tab::MtfFilter.to_int()] = vec![
            "PanelHeader:2D Filtering (optional)".into(),
            "DoseWeightFiltering".into(),
            "UniformFiltering".into(),
            "LowPassRadiusSigma".into(),
            "InverseFilteringParameters".into(),
            MTF_FILE_LABEL.into(),
            "MaximumInverse".into(),
            "InverseRolloffRadiusSigma".into(),
            FIXED_IMAGE_DOSE_LABEL.into(),
            "TypeOfDoseFile".into(),
            "DoseWeightingFile".into(),
            "Voltage200".into(),
            "OptimalDoseScaling".into(),
            "BidirectionalNumViews".into(),
            "StartingAndEndingZ".into(),
            "MtfFilterButtons".into(),
        ];
    }
    pub fn starting_and_ending_z_key_released<E: FinalAlignedStackDialogExpert>(
        &mut self,
        expert: &mut E,
    ) {
        expert.enable_use_filter();
    }
    pub fn update_ctf_plotter(&mut self) {
        let enable = !self.use_expected_defocus;
        self.config_file_enabled = enable;
        self.ctf_plotter_enabled = enable;
    }
    pub fn update_display(&mut self) {
        let dose = self.dose_weight_filtering;
        self.uniform_filtering_enabled = !dose;
        self.inverse_params_enabled = !dose;
        self.fixed_image_dose_enabled = dose;
        let fixed = self.fixed_image_dose_selected;
        self.dose_weighting_file_enabled = dose && !fixed;
        self.type_of_dose_file_enabled = dose && !fixed;
        self.bidirectional_num_views_enabled =
            dose && self.type_of_dose_file.enable_bidirectional_num_views();
    }
    pub fn get_mtf_filter_parameters(&self, _: bool) -> MtfFilterParameters {
        MtfFilterParameters {
            low_pass_radius_sigma: self
                .uniform_filtering_enabled
                .then_some(self.low_pass_radius_sigma.clone()),
            mtf_file: self
                .uniform_filtering_enabled
                .then_some(self.mtf_file.clone()),
            maximum_inverse: self
                .uniform_filtering_enabled
                .then_some(self.maximum_inverse.clone()),
            inverse_rolloff_radius_sigma: self
                .uniform_filtering_enabled
                .then_some(self.inverse_rolloff_radius_sigma.clone()),
            fixed_image_dose: self
                .fixed_image_dose_enabled
                .then_some(self.fixed_image_dose.clone()),
            type_of_dose_file: self
                .type_of_dose_file_enabled
                .then_some(self.type_of_dose_file),
            dose_weighting_file: self
                .dose_weighting_file_enabled
                .then_some(self.dose_weighting_file.clone()),
            voltage_200: self.dose_weight_filtering.then_some(self.voltage_200),
            optimal_dose_scaling: self
                .dose_weight_filtering
                .then_some(self.optimal_dose_scaling.clone()),
            bidirectional_num_views: self
                .bidirectional_num_views_enabled
                .then_some(self.bidirectional_num_views.clone()),
            starting_and_ending_z: self.starting_and_ending_z.clone(),
        }
    }
    pub fn set_mtf_filter_parameters(&mut self, param: &MtfFilterParameters) {
        self.dose_weight_filtering =
            param.type_of_dose_file.is_some() || param.fixed_image_dose.is_some();
        if let Some(v) = &param.mtf_file {
            self.mtf_file = v.clone()
        }
        if let Some(v) = &param.maximum_inverse {
            self.maximum_inverse = v.clone()
        }
        if let Some(v) = &param.low_pass_radius_sigma {
            self.low_pass_radius_sigma = v.clone()
        }
        if let Some(v) = &param.inverse_rolloff_radius_sigma {
            self.inverse_rolloff_radius_sigma = v.clone()
        }
        self.fixed_image_dose_selected = param.fixed_image_dose.is_some();
        if let Some(v) = &param.fixed_image_dose {
            self.fixed_image_dose = v.clone()
        }
        if let Some(v) = param.type_of_dose_file {
            self.type_of_dose_file = v
        }
        if let Some(v) = &param.dose_weighting_file {
            if !v.starts_with('.') {
                self.dose_weighting_file = v.clone()
            }
        }
        if let Some(v) = param.voltage_200 {
            self.voltage_200 = v
        }
        if let Some(v) = &param.optimal_dose_scaling {
            self.optimal_dose_scaling = v.clone()
        }
        if let Some(v) = &param.bidirectional_num_views {
            self.bidirectional_num_views = v.clone()
        }
        self.starting_and_ending_z = param.starting_and_ending_z.clone();
        self.update_display();
    }
    pub fn get_meta_data(&mut self) -> FinalAlignedStackMetaData {
        self.dialog_saved = true;
        FinalAlignedStackMetaData {
            dialog_saved: true,
            erase_beads_initialized: self.erase_beads_initialized,
            ctf_correction_parallel: self.is_parallel_process(),
            use_x_axis_tilt: self.use_x_axis_tilt,
            x_axis_tilt: self.x_axis_tilt.clone(),
            scale_by_ctf_power: self.scale_by_ctf_power.clone(),
            low_pass_radius_sigma: self.low_pass_radius_sigma.clone(),
            mtf_file: self.mtf_file.clone(),
            maximum_inverse: self.maximum_inverse.clone(),
            inverse_rolloff_radius_sigma: self.inverse_rolloff_radius_sigma.clone(),
            use_fixed_image_dose: self.fixed_image_dose_selected,
            fixed_image_dose: self.fixed_image_dose.clone(),
            dose_weighting_file: self.dose_weighting_file.clone(),
            type_of_dose_file: self.type_of_dose_file.get_value().to_string(),
            voltage_200: self.voltage_200,
            optimal_dose_scaling: self.optimal_dose_scaling.clone(),
            bidirectional_num_views: self.bidirectional_num_views.clone(),
            ctf3d_setup_slab_thickness_set: self.ctf3d_message_visible,
        }
    }
    pub fn set_meta_data(&mut self, meta: &FinalAlignedStackMetaData) {
        self.dialog_saved = meta.dialog_saved;
        self.erase_beads_initialized = meta.erase_beads_initialized;
        self.use_x_axis_tilt = meta.use_x_axis_tilt;
        self.x_axis_tilt = meta.x_axis_tilt.clone();
        self.scale_by_ctf_power = meta.scale_by_ctf_power.clone();
        self.low_pass_radius_sigma = meta.low_pass_radius_sigma.clone();
        self.mtf_file = meta.mtf_file.clone();
        self.maximum_inverse = meta.maximum_inverse.clone();
        self.inverse_rolloff_radius_sigma = meta.inverse_rolloff_radius_sigma.clone();
        self.fixed_image_dose_selected = meta.use_fixed_image_dose;
        self.fixed_image_dose = meta.fixed_image_dose.clone();
        if !meta.dose_weighting_file.starts_with('.') {
            self.dose_weighting_file = meta.dose_weighting_file.clone()
        }
        self.type_of_dose_file = TypeOfDoseFile::get_instance(Some(&meta.type_of_dose_file));
        self.voltage_200 = meta.voltage_200;
        self.optimal_dose_scaling = meta.optimal_dose_scaling.clone();
        self.bidirectional_num_views = meta.bidirectional_num_views.clone();
        self.ctf3d_message_visible = meta.ctf3d_setup_slab_thickness_set;
        self.ctf3d_filter_message_visible = meta.ctf3d_setup_slab_thickness_set;
        self.update_display();
    }
    pub fn update_advanced(&mut self) {
        self.update_advanced_filter(self.advanced);
        self.update_advanced_ctf_correction(self.advanced);
    }
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.processing_method_locked = lock;
    }
    pub fn get_processing_method(&self) -> &'static str {
        if self.processing_method_locked {
            "Parallel"
        } else {
            "Local"
        }
    }
    pub fn get_secondary_processing_method(&self) -> &'static str {
        "Local"
    }
    pub fn reregister_processing_method_mediator(&mut self) {
        self.mediator_reregistered = true;
    }
    pub fn change_tab<A: FinalAlignedStackDialogApplication>(
        &mut self,
        selected_index: usize,
        app: &mut A,
        use_ctf_warning: &mut bool,
        use_erased_warning: &mut bool,
        use_filter_warning: &mut bool,
    ) {
        let previous = self.cur_tab;
        self.cur_tab = Tab::get_instance(selected_index);
        self.selected_tab = selected_index;
        if previous == self.cur_tab {
            return;
        }
        self.mounted_tab = Some(self.cur_tab);
        if previous != Tab::CcdEraser
            && self.cur_tab == Tab::CcdEraser
            && !self.erase_beads_initialized
        {
            self.erase_beads_initialized = true;
        }
        self.reregister_processing_method_mediator();
        app.pack(self.axis_id);
        if previous == Tab::CtfCorrection && *use_ctf_warning {
            app.warning("Entry Warning", "To use the CTF correction go back to the Correct CTF tab and press the \"Use CTF Correction\" button.",self.axis_id);
            *use_ctf_warning = false;
        } else if previous == Tab::CcdEraser && *use_erased_warning {
            app.warning("Entry Warning", "To use the stack with the erased beads go back to the Erase Beads tab and press the \"Use Erased Stack\" button.",self.axis_id);
            *use_erased_warning = false;
        } else if previous == Tab::MtfFilter && *use_filter_warning {
            app.warning("Entry Warning", "To use the MTF filtered stack go back to the 2D Filter tab and press the \"Use Filtered Stack\" button.",self.axis_id);
            *use_filter_warning = false;
        }
        app.move_sub_frame();
    }
    pub fn pop_up_context_menu(&self) -> (&'static str, Vec<&'static str>, Vec<String>) {
        match self.cur_tab {
            Tab::CtfCorrection => (
                "CorrectingCTF",
                vec!["ctfplotter.html", "ctfphaseflip.html", "3dmod.html"],
                vec!["ctfplotter.log".into(), "ctfcorrection.log".into()],
            ),
            Tab::MtfFilter => (
                "Filtering2D",
                vec!["mtffilter.html", "3dmod.html"],
                vec!["mtffilter.log".into()],
            ),
            _ if self.montage => (
                "FinalAligned",
                vec!["blendmont.html", "3dmod.html"],
                vec!["blend.log".into()],
            ),
            _ => (
                "FinalAligned",
                vec!["newstack.html", "3dmod.html"],
                vec!["newst.log".into()],
            ),
        }
    }
    pub fn action<E: FinalAlignedStackDialogExpert, A: FinalAlignedStackDialogApplication>(
        &mut self,
        command: &str,
        deferred: bool,
        expert: &mut E,
        app: &mut A,
    ) {
        match command {
            "Filter" => expert.mtffilter(deferred),
            USE_FILTERED_STACK_LABEL => expert.use_mtf_filter(),
            "View Filtered Stack" => app.imod_mtf_filter(self.axis_id),
            "Use expected defocus" => self.update_ctf_plotter(),
            "Run Ctfplotter" => expert.ctf_plotter(),
            CTF_CORRECTION_LABEL => expert.ctf_correction(deferred, self.is_parallel_process()),
            "View CTF Correction" => app.imod_ctf_correction(self.axis_id),
            USE_CTF_CORRECTION_LABEL => expert.use_ctf_correction(),
            _ => self.update_display(),
        }
    }
    pub fn done<E: FinalAlignedStackDialogExpert>(&mut self, expert: &mut E) {
        expert.done_dialog();
        self.action_listeners_installed = false;
        self.displayed = false;
    }
    pub fn set_method(&mut self, processing_method: &str) {
        self.processing_method = processing_method.into();
    }
    pub fn is_use_gpu(&self) -> bool {
        false
    }
    pub fn set_use_queue_check_box(&mut self) {}
    pub fn update_gpu(&mut self, _: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dose_weighting_matches_source_enablement() {
        let mut d = FinalAlignedStackDialog::get_instance(AxisID::Only, false, Tab::DEFAULT);
        d.dose_weight_filtering = true;
        d.fixed_image_dose_selected = false;
        d.type_of_dose_file = TypeOfDoseFile::PriorAndCumulativeDose;
        d.update_display();
        assert!(!d.uniform_filtering_enabled);
        assert!(d.dose_weighting_file_enabled);
        assert!(!d.bidirectional_num_views_enabled);
    }
    #[test]
    fn ctf_x_axis_tilt_source_rule() {
        let mut d = FinalAlignedStackDialog::get_instance(AxisID::Only, false, Tab::DEFAULT);
        d.set_ctf_phase_flip_x_axis_tilt("2.5", true);
        assert!(!d.use_x_axis_tilt);
        d.set_ctf_phase_flip_x_axis_tilt("2.5", false);
        assert!(d.use_x_axis_tilt);
    }
    #[test]
    fn type_of_dose_file_defaults() {
        assert_eq!(TypeOfDoseFile::get_instance(None), TypeOfDoseFile::MdocFile);
        assert!(TypeOfDoseFile::MdocFile.enable_bidirectional_num_views());
    }
}
