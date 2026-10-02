//! `IMOD/Etomo/src/etomo/ui/swing/TiltalignPanel.java`.
//!
//! Java `final class TiltalignPanel implements Expandable, ActionListener,
//! UIComponent, SwingComponent`.  An EDT object: created as `Rc<Self>` by
//! [`TiltalignPanel::get_instance`], every method takes `&self`, mutable state
//! lives in `Cell`s.  `this` (handed to `PanelHeader`, to
//! `GlobalExpandButton.register` and to `btnRestrictalign.addActionListener`)
//! is the weak self reference made by `Rc::new_cyclic`.
//!
//! The three enum-constructed radio buttons of the local-alignment validation
//! group are built from [`LocalAlignValidation`]; `RadioButton`'s enum
//! constructors select the instance whose `isDefault()` is true
//! (RadioButton.java:114-116), so the group comes up with `AREA_REQUIREMENTS`
//! (`LocalAlignValidation.DEFAULT`) selected.

use crate::imod::etomo::ui::field::Field;
use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tiltalign_param::{self, ConstTiltalignParam};
use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::makecomfile_param::{self, MakecomfileParam};
use crate::imod::etomo::comscript::restrictalign_param::{self, RestrictalignParam};
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, ChangeEvent, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::autodoc::section::Section;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_etomo_number::{self, ConstEtomoNumber};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::ui_component::UIComponent;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::check_box::CheckBox;
use super::check_text_field::CheckTextField;
use super::etched_border::EtchedBorder;
use super::etomo_button_group::EtomoButtonGroup;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::fixed_dim as FixedDim;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::{EnumeratedTypeRef, RadioButtonInterface};
use super::radio_text_field::RadioTextField;
use super::single_line_button::SingleLineButton;
use super::spaced_panel::SpacedPanel;
use super::swing_component::SwingComponent;
use super::tabbed_pane::TabbedPane;
use super::ui_expert_utilities::UIExpertUtilities;
use super::ui_harness;
use crate::imod::etomo::jdk::Dimension;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java private static final `MIN_LOCAL_PATCH_SIZE_LABEL`.
const MIN_LOCAL_PATCH_SIZE_LABEL: &str = "Min. local patch size or overlap factor (x,y): ";
/// Java private static final `MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL`.
const MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL: &str = "Overlap factor (x,y): ";

/// Java private static `NullImagesAreBinnedReported` (class-wide, so shared by
/// both axes' panels).
static NULL_IMAGES_ARE_BINNED_REPORTED: AtomicBool = AtomicBool::new(false);

/// The two exceptions Java `getParameters(TiltalignParam, boolean)` throws:
/// the declared `FortranInputSyntaxException` and the unchecked
/// `NumberFormatException` (from `Double.parseDouble`), each rethrown with the
/// offending field's label prepended.  `ApplicationManager.updateAlignCom`
/// catches both.
#[derive(Debug)]
pub enum TiltalignParamsException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `NumberFormatException`; the value is its message.
    NumberFormatException(String),
}

/// Java private static final nested class `Tab`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    /// Java `GENERAL`.
    General,
    /// Java `GLOBAL_VARIABLES`.
    GlobalVariables,
    /// Java `LOCAL_VARIABLES`.
    LocalVariables,
}

impl Tab {
    /// Java `GENERAL_INDEX`.
    const GENERAL_INDEX: i32 = 0;
    /// Java `GLOBAL_VARIABLES_INDEX`.
    const GLOBAL_VARIABLES_INDEX: i32 = 1;
    /// Java `LOCAL_VARIABLES_INDEX`.
    const LOCAL_VARIABLES_INDEX: i32 = 2;

    /// Java private `getIndex()`.
    fn get_index(self) -> i32 {
        match self {
            Tab::General => Tab::GENERAL_INDEX,
            Tab::GlobalVariables => Tab::GLOBAL_VARIABLES_INDEX,
            Tab::LocalVariables => Tab::LOCAL_VARIABLES_INDEX,
        }
    }

    /// Java private static `getInstance(int)`: get the tab associated with the
    /// index.  Default: SETUP
    fn get_instance(index: i32) -> Tab {
        match index {
            Tab::GENERAL_INDEX => Tab::General,
            Tab::GLOBAL_VARIABLES_INDEX => Tab::GlobalVariables,
            Tab::LOCAL_VARIABLES_INDEX => Tab::LocalVariables,
            _ => Tab::General,
        }
    }
}

/// Java private static final nested class `LocalAlignValidation implements
/// EnumeratedType`.  The three instances are `static`s so that identity (Java
/// `==`, and `EtomoButtonGroup`'s `HashMap` key) is preserved.  A copy (as held
/// by an `EnumeratedTypeRef`) compares equal to its instance: the three
/// instances differ in both fields.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalAlignValidation {
    /// Java private final `label`.
    label: &'static str,
    /// Java private final `value` (an `EtomoNumber` set once to this int).
    value: i32,
}

/// Java `LocalAlignValidation.AREA_REQUIREMENTS`.
pub static AREA_REQUIREMENTS: LocalAlignValidation = LocalAlignValidation {
    label: "Area requirements",
    value: 2,
};
/// Java `LocalAlignValidation.VARIABLES`.
pub static VARIABLES: LocalAlignValidation = LocalAlignValidation {
    label: "Variables",
    value: 1,
};
/// Java `LocalAlignValidation.BOTH`.
pub static BOTH: LocalAlignValidation = LocalAlignValidation {
    label: "Both",
    value: 3,
};
/// Java `LocalAlignValidation.DEFAULT`.
static DEFAULT: &LocalAlignValidation = &AREA_REQUIREMENTS;

impl LocalAlignValidation {
    /// Java field `value`: `new EtomoNumber()` then `value.set(int)`.
    fn value_number(&self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(self.value);
        value
    }

    /// Java private static `getInstance(String)`.
    fn get_instance(value: Option<&str>) -> &'static LocalAlignValidation {
        let Some(value) = value else {
            return DEFAULT;
        };
        if AREA_REQUIREMENTS.value_number().equals_string(Some(value)) {
            return &AREA_REQUIREMENTS;
        }
        if VARIABLES.value_number().equals_string(Some(value)) {
            return &VARIABLES;
        }
        if BOTH.value_number().equals_string(Some(value)) {
            return &BOTH;
        }
        DEFAULT
    }
}

impl EnumeratedType for LocalAlignValidation {
    /// Java `isDefault()`: `this == DEFAULT`.
    fn is_default(&self) -> bool {
        *self == *DEFAULT
    }

    /// Java `getValue()`.
    fn get_value(&self) -> ConstEtomoNumber {
        self.value_number().base
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        Some(self.label.to_string())
    }
}

/// Java `toString()`: the label.
impl std::fmt::Display for LocalAlignValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

/// Java `final class TiltalignPanel implements Expandable, ActionListener,
/// UIComponent, SwingComponent`.
pub struct TiltalignPanel {
    /// Java `this`.
    this: Weak<TiltalignPanel>,
    /// Java private final `axisID`.
    axis_id: AxisID,

    /// Java private final `tabPane`.
    tab_pane: Rc<TabbedPane>,

    // General pane
    pnl_general: Rc<EtomoPanel>,
    pnl_general_body: Rc<JComponent>,

    ltf_residual_threshold: Rc<LabeledTextField>,

    rb_resid_all_views: Rc<RadioButton>,
    rb_resid_neighboring: Rc<RadioButton>,
    bg_residual_threshold: Rc<ButtonGroup>,
    pnl_residual_threshold: Rc<EtomoPanel>,

    rb_single_fiducial_surface: Rc<RadioButton>,
    rb_dual_fiducial_surfaces: Rc<RadioButton>,
    bg_fiducial_surfaces: Rc<ButtonGroup>,
    pnl_fiducial_surfaces: Rc<EtomoPanel>,

    ltf_exclude_list: Rc<LabeledTextField>,
    ltf_separate_view_groups: Rc<LabeledTextField>,

    pnl_volume_parameters: Rc<EtomoPanel>,
    ltf_tilt_angle_offset: Rc<LabeledTextField>,
    ltf_tilt_axis_z_shift: Rc<LabeledTextField>,

    pnl_minimization_params: Rc<EtomoPanel>,
    pnl_metro_factor: Rc<JComponent>,
    ltf_metro_factor: Rc<LabeledTextField>,
    ltf_maximum_cycles: Rc<LabeledTextField>,

    pnl_local_parameters: Rc<EtomoPanel>,
    pnl_local_parameters_body: Rc<EtomoPanel>,
    pnl_local_patches: Rc<SpacedPanel>,
    cb_local_alignments: Rc<CheckBox>,
    bg_local_patches: Rc<ButtonGroup>,
    rtf_target_patch_size_xand_y: Rc<RadioTextField>,
    rtf_n_local_patches: Rc<RadioTextField>,
    ltf_min_local_patch_size: Rc<LabeledTextField>,
    ltf_min_local_fiducials: Rc<LabeledTextField>,
    cb_fix_xyz_coordinates: Rc<CheckBox>,

    // Global variables pane
    pnl_global_variable: Rc<EtomoPanel>,
    pnl_global_variable_body: Rc<JComponent>,

    // Tilt angle pane
    rb_tilt_angle_fixed: Rc<RadioButton>,
    rb_tilt_angle_all: Rc<RadioButton>,
    rb_tilt_angle_automap: Rc<RadioButton>,
    bg_tilt_angle_solution: Rc<ButtonGroup>,
    pnl_tilt_angle_solution: Rc<EtomoPanel>,

    ltf_tilt_angle_group_size: Rc<LabeledTextField>,
    ltf_tilt_angle_non_default_groups: Rc<LabeledTextField>,

    // Magnfication pane
    rb_magnification_fixed: Rc<RadioButton>,
    rb_magnification_all: Rc<RadioButton>,
    /// Java private (non-final) `rbMagnificationAutomap`.
    rb_magnification_automap: Rc<RadioButton>,
    bg_magnification_solution: Rc<ButtonGroup>,
    pnl_magnification_solution: Rc<EtomoPanel>,

    ltf_magnification_reference_view: Rc<LabeledTextField>,
    ltf_magnification_group_size: Rc<LabeledTextField>,
    ltf_magnification_non_default_groups: Rc<LabeledTextField>,

    // GlobalDistortion pane
    pnl_distortion_solution: Rc<EtomoPanel>,
    rb_distortion_disabled: Rc<RadioButton>,
    rb_distortion_full_solution: Rc<RadioButton>,
    rb_distortion_skew: Rc<RadioButton>,
    bg_distortion_solution: Rc<ButtonGroup>,

    ltf_xstretch_group_size: Rc<LabeledTextField>,
    ltf_xstretch_non_default_groups: Rc<LabeledTextField>,

    ltf_skew_group_size: Rc<LabeledTextField>,
    ltf_skew_non_default_groups: Rc<LabeledTextField>,

    // Local variables pane
    pnl_local_solution: Rc<EtomoPanel>,
    pnl_local_solution_body: Rc<JComponent>,

    // Local rotation pane
    pnl_local_rotation_solution: Rc<EtomoPanel>,
    cb_local_rotation: Rc<CheckBox>,

    ltf_local_rotation_group_size: Rc<LabeledTextField>,
    ltf_local_rotation_non_default_groups: Rc<LabeledTextField>,

    // Local tilt angle pane
    pnl_local_tilt_angle_solution: Rc<EtomoPanel>,
    cb_local_tilt_angle: Rc<CheckBox>,

    ltf_local_tilt_angle_group_size: Rc<LabeledTextField>,
    ltf_local_tilt_angle_non_default_groups: Rc<LabeledTextField>,

    // Local magnfication pane
    pnl_local_magnification_solution: Rc<EtomoPanel>,
    cb_local_magnification: Rc<CheckBox>,

    ltf_local_magnification_group_size: Rc<LabeledTextField>,
    ltf_local_magnification_non_default_groups: Rc<LabeledTextField>,

    // Local distortion pane
    pnl_local_distortion_solution: Rc<EtomoPanel>,
    rb_local_distortion_disabled: Rc<RadioButton>,
    rb_local_distortion_full_solution: Rc<RadioButton>,
    rb_local_distortion_skew: Rc<RadioButton>,
    bg_local_distortion_solution: Rc<ButtonGroup>,

    ltf_local_xstretch_group_size: Rc<LabeledTextField>,
    ltf_local_xstretch_non_default_groups: Rc<LabeledTextField>,

    ltf_local_skew_group_size: Rc<LabeledTextField>,
    ltf_local_skew_non_default_groups: Rc<LabeledTextField>,

    // Rotation pane
    rb_rotation_none: Rc<RadioButton>,
    rb_rotation_all: Rc<RadioButton>,
    rb_rotation_automap: Rc<RadioButton>,
    rb_rotation_one: Rc<RadioButton>,
    bg_rotation_solution: Rc<ButtonGroup>,
    pnl_rotation_solution: Rc<EtomoPanel>,
    ltf_rotation_angle: Rc<LabeledTextField>,
    ltf_rotation_group_size: Rc<LabeledTextField>,
    ltf_rotation_non_default_groups: Rc<LabeledTextField>,
    cb_projection_stretch: Rc<CheckBox>,
    /// Java private final `appMgr`.
    app_mgr: &'static ApplicationManager,
    bg_beam_tilt_option: Rc<ButtonGroup>,
    rb_no_beam_tilt: Rc<RadioButton>,
    rtf_fixed_beam_tilt: Rc<RadioTextField>,
    rb_solve_for_beam_tilt: Rc<RadioButton>,
    pnl_single_variables: Rc<EtomoPanel>,
    pnl_single_variables_body: Rc<JComponent>,

    ctf_robust_fitting_and_k_factor_scaling: Rc<CheckTextField>,
    cb_weight_whole_tracks: Rc<CheckBox>,
    cb_x_tilt_automap_same: Rc<CheckBox>,
    ltf_target_measurement_ratio: Rc<LabeledTextField>,
    ltf_min_measurement_ratio: Rc<LabeledTextField>,
    /// Java private (non-final) `btnRestrictalign`.
    btn_restrictalign: Rc<SingleLineButton>,
    l_local_align_validation: Rc<JComponent>,
    bg_local_align_validation: Rc<EtomoButtonGroup>,
    rb_local_align_validation_area_requirements: Rc<RadioButton>,
    rb_local_align_validation_variables: Rc<RadioButton>,
    rb_local_align_validation_both: Rc<RadioButton>,

    cb_cross_validate: Rc<CheckBox>,

    /// Java private final `phSingleVariables`.
    ph_single_variables: Rc<PanelHeader>,

    /// Java private `currentTab`.
    current_tab: Cell<Tab>,
    /// Java private `patchTracking`.
    patch_tracking: Cell<bool>,
    /// Java private `createdDayStampIMOD_5_0_1`.
    created_day_stamp_imod_5_0_1: Cell<bool>,
}

impl TiltalignPanel {
    /// Java private constructor `TiltalignPanel(AxisID, ApplicationManager,
    /// GlobalExpandButton)`.
    fn new(
        axis: AxisID,
        app_mgr: &'static ApplicationManager,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<TiltalignPanel> {
        let panel = Rc::new_cyclic(|this: &Weak<TiltalignPanel>| {
            // Field initializers, in declaration order.
            let tab_pane = TabbedPane::new();
            let pnl_general = EtomoPanel::new();
            let pnl_general_body = JComponent::new_panel();
            let ltf_residual_threshold = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Threshold for residual report: "),
            );
            let rb_resid_all_views = RadioButton::new_string(Some("All views"));
            let rb_resid_neighboring = RadioButton::new_string(Some("Neighboring views"));
            let bg_residual_threshold = ButtonGroup::new();
            let pnl_residual_threshold = EtomoPanel::new();
            let rb_single_fiducial_surface =
                RadioButton::new_string(Some("Do not sort fiducials into 2 surfaces for analysis"));
            let rb_dual_fiducial_surfaces =
                RadioButton::new_string(Some("Assume fiducials on 2 surfaces for analysis"));
            let bg_fiducial_surfaces = ButtonGroup::new();
            let pnl_fiducial_surfaces = EtomoPanel::new();
            let ltf_exclude_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("List of views to exclude: "),
            );
            let ltf_separate_view_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some("Separate view groups: "),
            );
            let pnl_volume_parameters = EtomoPanel::new();
            let ltf_tilt_angle_offset = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Total tilt angle offset: "),
            );
            let ltf_tilt_axis_z_shift = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Tilt axis z shift: "),
            );
            let pnl_minimization_params = EtomoPanel::new();
            let pnl_metro_factor = JComponent::new_panel();
            let ltf_metro_factor = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Metro factor: "),
            );
            let ltf_maximum_cycles = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Iteration limit: "),
            );
            let pnl_local_parameters = EtomoPanel::new();
            let pnl_local_parameters_body = EtomoPanel::new();
            let pnl_local_patches = SpacedPanel::get_instance_boolean(true);
            let cb_local_alignments = CheckBox::new_string(Some("Enable local alignments"));
            let bg_local_patches = ButtonGroup::new();
            let rtf_target_patch_size_xand_y =
                RadioTextField::get_instance_field_type_string_button_group(
                    FieldType::IntegerPair,
                    Some("Target patch size (x,y): "),
                    Some(&bg_local_patches),
                );
            let rtf_n_local_patches = RadioTextField::get_instance_field_type_string_button_group(
                FieldType::IntegerPair,
                Some("# of local patches (x,y): "),
                Some(&bg_local_patches),
            );
            let ltf_min_local_patch_size = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointPair,
                Some(MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL),
            );
            let ltf_min_local_fiducials = LabeledTextField::new_field_type_string(
                FieldType::IntegerPair,
                Some("Min. # of fiducials (total, each surface): "),
            );
            let cb_fix_xyz_coordinates = CheckBox::new_string(Some("Use global X-Y-Z coordinates"));

            // Global variables pane
            let pnl_global_variable = EtomoPanel::new();
            let pnl_global_variable_body = JComponent::new_panel();

            // Tilt angle pane
            let rb_tilt_angle_fixed = RadioButton::new_string(Some("Fixed tilt angles"));
            let rb_tilt_angle_all =
                RadioButton::new_string(Some("Solve for all except minimum tilt"));
            let rb_tilt_angle_automap = RadioButton::new_string(Some("Group tilt angles "));
            let bg_tilt_angle_solution = ButtonGroup::new();
            let pnl_tilt_angle_solution = EtomoPanel::new();

            let ltf_tilt_angle_group_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Group size: "));
            let ltf_tilt_angle_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default grouping: "),
            );

            // Magnfication pane
            let rb_magnification_fixed =
                RadioButton::new_string(Some("Fixed magnification at 1.0"));
            let rb_magnification_all =
                RadioButton::new_string(Some("Solve for all magnifications"));
            let rb_magnification_automap = RadioButton::new_string(Some("Group magnifications"));
            let bg_magnification_solution = ButtonGroup::new();
            let pnl_magnification_solution = EtomoPanel::new();

            let ltf_magnification_reference_view = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Reference view: "),
            );
            let ltf_magnification_group_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Group size: "));
            let ltf_magnification_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default grouping: "),
            );

            // GlobalDistortion pane
            let pnl_distortion_solution = EtomoPanel::new();
            let rb_distortion_disabled = RadioButton::new_string(Some("Disabled"));
            let rb_distortion_full_solution = RadioButton::new_string(Some("Full solution"));
            let rb_distortion_skew = RadioButton::new_string(Some("Skew only"));
            let bg_distortion_solution = ButtonGroup::new();

            let ltf_xstretch_group_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("X stretch group size: "),
            );
            let ltf_xstretch_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("X stretch non-default grouping: "),
            );

            let ltf_skew_group_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Skew group size: "),
            );
            let ltf_skew_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Skew non-default grouping: "),
            );

            // Local variables pane
            let pnl_local_solution = EtomoPanel::new();
            let pnl_local_solution_body = JComponent::new_panel();

            // Local rotation pane
            let pnl_local_rotation_solution = EtomoPanel::new();
            let cb_local_rotation = CheckBox::new_string(Some("Enable"));

            let ltf_local_rotation_group_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Group size: "));
            let ltf_local_rotation_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default grouping: "),
            );

            // Local tilt angle pane
            let pnl_local_tilt_angle_solution = EtomoPanel::new();
            let cb_local_tilt_angle = CheckBox::new_string(Some("Enable"));

            let ltf_local_tilt_angle_group_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Group size: "));
            let ltf_local_tilt_angle_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default grouping: "),
            );

            // Local magnfication pane
            let pnl_local_magnification_solution = EtomoPanel::new();
            let cb_local_magnification = CheckBox::new_string(Some("Enable"));

            let ltf_local_magnification_group_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Group size: "));
            let ltf_local_magnification_non_default_groups =
                LabeledTextField::new_field_type_string(
                    FieldType::IntegerTriple,
                    Some("Non-default grouping: "),
                );

            // Local distortion pane
            let pnl_local_distortion_solution = EtomoPanel::new();
            let rb_local_distortion_disabled = RadioButton::new_string(Some("Disabled"));
            let rb_local_distortion_full_solution = RadioButton::new_string(Some("Full solution"));
            let rb_local_distortion_skew = RadioButton::new_string(Some("Skew only"));
            let bg_local_distortion_solution = ButtonGroup::new();

            let ltf_local_xstretch_group_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("X stretch group size: "),
            );
            let ltf_local_xstretch_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("X stretch non-default grouping: "),
            );

            let ltf_local_skew_group_size = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Skew group size: "),
            );
            let ltf_local_skew_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Skew non-default grouping: "),
            );

            // Rotation pane
            let rb_rotation_none = RadioButton::new_string(Some("No rotation"));
            let rb_rotation_all = RadioButton::new_string(Some("Solve for all rotations"));
            let rb_rotation_automap = RadioButton::new_string(Some("Group rotations"));
            let rb_rotation_one = RadioButton::new_string(Some("One rotation"));
            let bg_rotation_solution = ButtonGroup::new();
            let pnl_rotation_solution = EtomoPanel::new();
            let ltf_rotation_angle = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Rotation angle: "),
            );
            let ltf_rotation_group_size =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some("Group size: "));
            let ltf_rotation_non_default_groups = LabeledTextField::new_field_type_string(
                FieldType::IntegerTriple,
                Some("Non-default grouping: "),
            );
            let cb_projection_stretch =
                CheckBox::new_string(Some("Solve for single stretch during projection"));
            let bg_beam_tilt_option = ButtonGroup::new();
            let rb_no_beam_tilt = RadioButton::new_string(Some("No beam tilt"));
            let rtf_fixed_beam_tilt = RadioTextField::get_instance_field_type_string_button_group(
                FieldType::FloatingPoint,
                Some("Fixed beam tilt (degrees): "),
                Some(&bg_beam_tilt_option),
            );
            let rb_solve_for_beam_tilt = RadioButton::new_string(Some("Solve for beam tilt"));
            let pnl_single_variables = EtomoPanel::new();
            let pnl_single_variables_body = JComponent::new_panel();

            let ctf_robust_fitting_and_k_factor_scaling = CheckTextField::get_instance(
                FieldType::FloatingPoint,
                "Do robust fitting with tuning factor:",
            );
            let cb_weight_whole_tracks =
                CheckBox::new_string(Some("Find weights for contours, not points"));
            let cb_x_tilt_automap_same =
                CheckBox::new_string(Some("Solve for X axis tilt between separate groups"));
            let ltf_target_measurement_ratio = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Fallback ratio of measurements to unknowns:  Target "),
            );
            let ltf_min_measurement_ratio =
                LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("Minimum "));
            let btn_restrictalign = SingleLineButton::new_string(Some("Run Cross-Validation"));
            let l_local_align_validation = JComponent::new_label("Test local ");
            let bg_local_align_validation = EtomoButtonGroup::new();
            // The enum constructor selects the button whose EnumeratedType
            // isDefault() (RadioButton.java:114-116): AREA_REQUIREMENTS.
            // Java `new RadioButton(String, EnumeratedType, ButtonGroup)` with
            // the EtomoButtonGroup: the constructor's `group.add(this)` dispatches
            // to `EtomoButtonGroup.add`, which records the button's model under
            // its enumerated type.  The Rust constructor takes a plain
            // ButtonGroup, so the button is added to the EtomoButtonGroup here.
            let rb_local_align_validation_area_requirements =
                RadioButton::new_string_enumerated_type_button_group(
                    AREA_REQUIREMENTS.get_label().as_deref(),
                    Some(EnumeratedTypeRef::new(AREA_REQUIREMENTS.clone())),
                    None,
                );
            bg_local_align_validation.add(
                &rb_local_align_validation_area_requirements.get_abstract_button(),
                Some(RadioButtonModel::new(Some(Rc::downgrade(
                    &rb_local_align_validation_area_requirements,
                )
                    as Weak<dyn RadioButtonInterface>))
                    as Rc<dyn AbstractRadioButtonModel>),
            );
            // Java `new RadioButton(String, EnumeratedType, ButtonGroup)` with
            // the EtomoButtonGroup: the constructor's `group.add(this)` dispatches
            // to `EtomoButtonGroup.add`, which records the button's model under
            // its enumerated type.  The Rust constructor takes a plain
            // ButtonGroup, so the button is added to the EtomoButtonGroup here.
            let rb_local_align_validation_variables =
                RadioButton::new_string_enumerated_type_button_group(
                    VARIABLES.get_label().as_deref(),
                    Some(EnumeratedTypeRef::new(VARIABLES.clone())),
                    None,
                );
            bg_local_align_validation.add(
                &rb_local_align_validation_variables.get_abstract_button(),
                Some(RadioButtonModel::new(Some(
                    Rc::downgrade(&rb_local_align_validation_variables)
                        as Weak<dyn RadioButtonInterface>,
                )) as Rc<dyn AbstractRadioButtonModel>),
            );
            // Java `new RadioButton(String, EnumeratedType, ButtonGroup)` with
            // the EtomoButtonGroup: the constructor's `group.add(this)` dispatches
            // to `EtomoButtonGroup.add`, which records the button's model under
            // its enumerated type.  The Rust constructor takes a plain
            // ButtonGroup, so the button is added to the EtomoButtonGroup here.
            let rb_local_align_validation_both =
                RadioButton::new_string_enumerated_type_button_group(
                    BOTH.get_label().as_deref(),
                    Some(EnumeratedTypeRef::new(BOTH.clone())),
                    None,
                );
            bg_local_align_validation.add(
                &rb_local_align_validation_both.get_abstract_button(),
                Some(
                    RadioButtonModel::new(Some(Rc::downgrade(&rb_local_align_validation_both)
                        as Weak<dyn RadioButtonInterface>))
                        as Rc<dyn AbstractRadioButtonModel>,
                ),
            );

            let cb_cross_validate = CheckBox::new_string(Some(
                "Compute prediction errors for points left out of test fits",
            ));

            // Constructor body.
            let axis_id = axis;
            tab_pane.get_component().set_border_title(
                EtchedBorder::new(Some("Tiltalign Parameters"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
            let expandable: Weak<dyn Expandable> = this.clone();
            let ph_single_variables = PanelHeader::get_advanced_basic_only_no_separator_instance(
                Some(&format!(
                    "{}{}",
                    "Single Variables: Beam Tilt,", " X Tilt, Projection Stretch"
                )),
                Some(expandable),
                Some(DialogType::FineAlignment),
                Some(global_advanced_button.clone()),
                true,
            );

            TiltalignPanel {
                this: this.clone(),
                axis_id,
                tab_pane,
                pnl_general,
                pnl_general_body,
                ltf_residual_threshold,
                rb_resid_all_views,
                rb_resid_neighboring,
                bg_residual_threshold,
                pnl_residual_threshold,
                rb_single_fiducial_surface,
                rb_dual_fiducial_surfaces,
                bg_fiducial_surfaces,
                pnl_fiducial_surfaces,
                ltf_exclude_list,
                ltf_separate_view_groups,
                pnl_volume_parameters,
                ltf_tilt_angle_offset,
                ltf_tilt_axis_z_shift,
                pnl_minimization_params,
                pnl_metro_factor,
                ltf_metro_factor,
                ltf_maximum_cycles,
                pnl_local_parameters,
                pnl_local_parameters_body,
                pnl_local_patches,
                cb_local_alignments,
                bg_local_patches,
                rtf_target_patch_size_xand_y,
                rtf_n_local_patches,
                ltf_min_local_patch_size,
                ltf_min_local_fiducials,
                cb_fix_xyz_coordinates,
                pnl_global_variable,
                pnl_global_variable_body,
                rb_tilt_angle_fixed,
                rb_tilt_angle_all,
                rb_tilt_angle_automap,
                bg_tilt_angle_solution,
                pnl_tilt_angle_solution,
                ltf_tilt_angle_group_size,
                ltf_tilt_angle_non_default_groups,
                rb_magnification_fixed,
                rb_magnification_all,
                rb_magnification_automap,
                bg_magnification_solution,
                pnl_magnification_solution,
                ltf_magnification_reference_view,
                ltf_magnification_group_size,
                ltf_magnification_non_default_groups,
                pnl_distortion_solution,
                rb_distortion_disabled,
                rb_distortion_full_solution,
                rb_distortion_skew,
                bg_distortion_solution,
                ltf_xstretch_group_size,
                ltf_xstretch_non_default_groups,
                ltf_skew_group_size,
                ltf_skew_non_default_groups,
                pnl_local_solution,
                pnl_local_solution_body,
                pnl_local_rotation_solution,
                cb_local_rotation,
                ltf_local_rotation_group_size,
                ltf_local_rotation_non_default_groups,
                pnl_local_tilt_angle_solution,
                cb_local_tilt_angle,
                ltf_local_tilt_angle_group_size,
                ltf_local_tilt_angle_non_default_groups,
                pnl_local_magnification_solution,
                cb_local_magnification,
                ltf_local_magnification_group_size,
                ltf_local_magnification_non_default_groups,
                pnl_local_distortion_solution,
                rb_local_distortion_disabled,
                rb_local_distortion_full_solution,
                rb_local_distortion_skew,
                bg_local_distortion_solution,
                ltf_local_xstretch_group_size,
                ltf_local_xstretch_non_default_groups,
                ltf_local_skew_group_size,
                ltf_local_skew_non_default_groups,
                rb_rotation_none,
                rb_rotation_all,
                rb_rotation_automap,
                rb_rotation_one,
                bg_rotation_solution,
                pnl_rotation_solution,
                ltf_rotation_angle,
                ltf_rotation_group_size,
                ltf_rotation_non_default_groups,
                cb_projection_stretch,
                app_mgr,
                bg_beam_tilt_option,
                rb_no_beam_tilt,
                rtf_fixed_beam_tilt,
                rb_solve_for_beam_tilt,
                pnl_single_variables,
                pnl_single_variables_body,
                ctf_robust_fitting_and_k_factor_scaling,
                cb_weight_whole_tracks,
                cb_x_tilt_automap_same,
                ltf_target_measurement_ratio,
                ltf_min_measurement_ratio,
                btn_restrictalign,
                l_local_align_validation,
                bg_local_align_validation,
                rb_local_align_validation_area_requirements,
                rb_local_align_validation_variables,
                rb_local_align_validation_both,
                cb_cross_validate,
                ph_single_variables,
                current_tab: Cell::new(Tab::General),
                patch_tracking: Cell::new(false),
                created_day_stamp_imod_5_0_1: Cell::new(false),
            }
        });
        let expandable: Weak<dyn Expandable> = Rc::downgrade(&panel) as Weak<dyn Expandable>;
        global_advanced_button.register_expandable(expandable);
        // Create the tabs
        panel.create_general_tab();
        panel.create_global_solution_tab();
        panel.create_local_solution_tab();
        panel.set_tool_tip_text();
        panel
    }

    /// Java static `getInstance(AxisID, ApplicationManager, GlobalExpandButton)`:
    /// construct a local instance and add listeners.
    pub fn get_instance(
        axis: AxisID,
        app_mgr: &'static ApplicationManager,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<TiltalignPanel> {
        let tiltalign_panel = TiltalignPanel::new(axis, app_mgr, global_advanced_button);
        tiltalign_panel.add_listeners();
        tiltalign_panel
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // ResidualRadioListener: `actionPerformed` is empty in the Java.
        let residual_radio_listener: ActionListener = Rc::new(|_event| {});
        self.rb_resid_all_views
            .add_action_listener(residual_radio_listener.clone());
        self.rb_resid_neighboring
            .add_action_listener(residual_radio_listener);
        // FiducialRadioListener: `actionPerformed` is empty in the Java.
        let fiducial_radio_listener: ActionListener = Rc::new(|_event| {});
        self.rb_single_fiducial_surface
            .add_action_listener(fiducial_radio_listener.clone());
        self.rb_dual_fiducial_surfaces
            .add_action_listener(fiducial_radio_listener);
        // LocalAlignmentsListener
        let local_alignments_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_local_alignment_dependents();
                }
            })
        };
        self.cb_local_alignments
            .add_action_listener(Some(local_alignments_listener));
        // RotationRadioListener
        let rotation_radio_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_rotation_solution_fields();
                    panel.update_display();
                }
            })
        };
        self.rb_rotation_none
            .add_action_listener(rotation_radio_listener.clone());
        self.rb_rotation_all
            .add_action_listener(rotation_radio_listener.clone());
        self.rb_rotation_automap
            .add_action_listener(rotation_radio_listener.clone());
        self.rb_rotation_one
            .add_action_listener(rotation_radio_listener);
        // TiltAngleRadioListener
        let tilt_angle_radio_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_tilt_angle_solution_fields();
                }
            })
        };
        self.rb_tilt_angle_fixed
            .add_action_listener(tilt_angle_radio_listener.clone());
        self.rb_tilt_angle_all
            .add_action_listener(tilt_angle_radio_listener.clone());
        self.rb_tilt_angle_automap
            .add_action_listener(tilt_angle_radio_listener);
        // MagnificationRadioListener
        let magnification_radio_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_magnification_solution_fields();
                }
            })
        };
        self.rb_magnification_fixed
            .add_action_listener(magnification_radio_listener.clone());
        self.rb_magnification_all
            .add_action_listener(magnification_radio_listener.clone());
        self.rb_magnification_automap
            .add_action_listener(magnification_radio_listener);
        // DistortionRadioListener
        let distortion_radio_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.set_distortion_solution_state();
                    panel.update_display();
                }
            })
        };
        self.rb_distortion_disabled
            .add_action_listener(distortion_radio_listener.clone());
        self.rb_distortion_full_solution
            .add_action_listener(distortion_radio_listener.clone());
        self.rb_distortion_skew
            .add_action_listener(distortion_radio_listener);
        // LocalRotationCheckListener
        let local_rotation_check_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_local_rotation_solution_fields();
                }
            })
        };
        self.cb_local_rotation
            .add_action_listener(Some(local_rotation_check_listener));
        // LocalTiltAngleCheckListener
        let local_tilt_angle_check_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_local_tilt_angle_solution_fields();
                }
            })
        };
        self.cb_local_tilt_angle
            .add_action_listener(Some(local_tilt_angle_check_listener));
        // LocalMagnificationCheckListener
        let local_magnification_check_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_local_magnification_solution_fields();
                }
            })
        };
        self.cb_local_magnification
            .add_action_listener(Some(local_magnification_check_listener));
        // LocalDistortionRadioListener
        let local_distortion_radio_listener: ActionListener = {
            let panel = self.this.clone();
            Rc::new(move |_event| {
                if let Some(panel) = panel.upgrade() {
                    panel.enable_local_distortion_solution_fields();
                }
            })
        };
        self.rb_local_distortion_disabled
            .add_action_listener(local_distortion_radio_listener.clone());
        self.rb_local_distortion_full_solution
            .add_action_listener(local_distortion_radio_listener.clone());
        self.rb_local_distortion_skew
            .add_action_listener(local_distortion_radio_listener);
        // TPActionListener
        let tp_action_listener: ActionListener = {
            let tiltalign_panel = self.this.clone();
            Rc::new(move |action_event| {
                if let Some(tiltalign_panel) = tiltalign_panel.upgrade() {
                    tiltalign_panel.action(action_event);
                }
            })
        };
        self.rtf_target_patch_size_xand_y
            .add_action_listener(tp_action_listener.clone());
        self.rtf_n_local_patches
            .add_action_listener(tp_action_listener.clone());
        // TabChangeListener
        {
            let tiltalign_panel = self.this.clone();
            self.tab_pane
                .get_component()
                .add_change_listener(Rc::new(move |change_event| {
                    if let Some(tiltalign_panel) = tiltalign_panel.upgrade() {
                        tiltalign_panel.change_tab(change_event);
                    }
                }));
        }
        self.rb_no_beam_tilt
            .add_action_listener(tp_action_listener.clone());
        self.rtf_fixed_beam_tilt
            .add_action_listener(tp_action_listener.clone());
        self.rb_solve_for_beam_tilt
            .add_action_listener(tp_action_listener.clone());
        self.ctf_robust_fitting_and_k_factor_scaling
            .add_action_listener(tp_action_listener);
        // btnRestrictalign.addActionListener(this)
        {
            let this = self.this.clone();
            self.btn_restrictalign
                .add_action_listener(Rc::new(move |action_event| {
                    if let Some(this) = this.upgrade() {
                        this.action_performed(Some(action_event));
                    }
                }));
        }
    }

    // <p>updates done</p>

    /// Java private `changeTab(ChangeEvent)`.
    fn change_tab(&self, _change_event: &ChangeEvent) {
        self.update_tab(false);
        self.current_tab.set(Tab::get_instance(
            self.tab_pane.get_component().get_selected_tab(),
        ));
        self.update_tab(true);
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.app_mgr))
        });
        ui_harness::INSTANCE.with(|harness| harness.move_sub_frame());
    }

    /// Java private `updateTab(boolean)`.
    fn update_tab(&self, visible: bool) {
        if self.current_tab.get() == Tab::General {
            self.pnl_general_body.set_visible(visible);
        } else if self.current_tab.get() == Tab::GlobalVariables {
            self.pnl_global_variable_body.set_visible(visible);
        } else if self.current_tab.get() == Tab::LocalVariables {
            self.pnl_local_solution_body.set_visible(visible);
        }
    }

    /// Java private `updateDisplay()`: more field enabling.  Enable/disable
    /// global radio buttons based on which radio buttons are selected.
    fn update_display(&self) {
        // Beam Tilt
        // Don't disable a field which is selected. This shouldn't come up unless the
        // comscript was changed. In this case the problem will be handled by a tiltalign
        // error message.
        // Rotation
        // Solve for all rotations:
        let mut keep_enabled = self.rb_rotation_all.is_selected();
        // Group and solve rotations: disable if full or skew AND solve beam tilt
        // selected.
        let mut disable = (self.rb_distortion_full_solution.is_selected()
            || self.rb_distortion_skew.is_selected())
            && self.rb_solve_for_beam_tilt.is_selected();
        self.rb_rotation_all.set_enabled(keep_enabled || !disable);
        // Group Rotations:
        keep_enabled = self.rb_rotation_automap.is_selected();
        self.rb_rotation_automap
            .set_enabled(keep_enabled || !disable);
        // Distortion
        // Distortion: Full solution:
        keep_enabled = self.rb_distortion_full_solution.is_selected();
        // Full and skew: disabled if All rotations or Group rotations AND solve beam
        // tilt is on
        disable = (self.rb_rotation_all.is_selected() || self.rb_rotation_automap.is_selected())
            && self.rb_solve_for_beam_tilt.is_selected();
        self.rb_distortion_full_solution
            .set_enabled(keep_enabled || !disable);
        // Distortion: Skew only:
        keep_enabled = self.rb_distortion_skew.is_selected();
        self.rb_distortion_skew
            .set_enabled(keep_enabled || !disable);
        // Beam Tilt
        // Solve for beam tilt:
        keep_enabled = self.rb_solve_for_beam_tilt.is_selected();
        // Solve for beam tilt: disable if All rotations or Group rotations AND Full or
        // Skew distortion selected.
        disable = (self.rb_rotation_all.is_selected() || self.rb_rotation_automap.is_selected())
            && (self.rb_distortion_full_solution.is_selected()
                || self.rb_distortion_skew.is_selected());
        self.rb_solve_for_beam_tilt
            .set_enabled(keep_enabled || !disable);
        self.cb_weight_whole_tracks.set_enabled(
            self.patch_tracking.get() && self.ctf_robust_fitting_and_k_factor_scaling.is_selected(),
        );
    }

    /// Java private `action(ActionEvent)`.
    fn action(&self, action_event: &ActionEvent) {
        let action_command = action_event.get_action_command();
        if action_command
            == self
                .rtf_target_patch_size_xand_y
                .get_action_command()
                .as_deref()
        {
            self.set_min_local_patch_size_label();
        } else if action_command == self.rtf_n_local_patches.get_action_command().as_deref() {
            self.set_min_local_patch_size_label();
        } else if action_command == self.rb_no_beam_tilt.get_action_command().as_deref() {
            self.update_display();
        } else if action_command == self.rtf_fixed_beam_tilt.get_action_command().as_deref() {
            self.update_display();
        } else if action_command == self.rb_solve_for_beam_tilt.get_action_command().as_deref() {
            self.update_display();
        } else if action_command
            == self
                .ctf_robust_fitting_and_k_factor_scaling
                .get_action_command()
                .as_deref()
        {
            self.update_display();
        }
    }

    /// Java `actionPerformed(ActionEvent)` (`implements ActionListener`).
    pub fn action_performed(&self, action_event: Option<&ActionEvent>) {
        let Some(action_event) = action_event else {
            return;
        };
        let Some(action_command) = action_event.get_action_command() else {
            return;
        };
        if Some(action_command) == self.btn_restrictalign.get_action_command().as_deref() {
            self.app_mgr.restrictalign(self.axis_id, None);
        }
    }

    /// Java private `updateAdvancedBeamTilt(boolean)`.
    fn update_advanced_beam_tilt(&self, advanced: bool) {
        self.pnl_single_variables_body.set_visible(advanced);
    }

    /// Java private `setMinLocalPatchSizeLabel()`.
    fn set_min_local_patch_size_label(&self) {
        if self.rtf_target_patch_size_xand_y.is_selected() {
            self.ltf_min_local_patch_size
                .set_label(Some(MIN_LOCAL_PATCH_SIZE_OVERLAP_ONLY_LABEL));
        } else if !self.created_day_stamp_imod_5_0_1.get() && self.rtf_n_local_patches.is_selected()
        {
            self.ltf_min_local_patch_size
                .set_label(Some(MIN_LOCAL_PATCH_SIZE_LABEL));
        }
    }

    /// Java `setParameters(ConstTiltalignParam)`: set the values of the panel
    /// using a constant tiltalign parameter object.
    pub fn set_parameters_const_tiltalign_param(&self, params: &ConstTiltalignParam) {
        // General panel parameters
        if params.get_surfaces_to_analyze().get_int() == 2 {
            self.rb_dual_fiducial_surfaces.set_selected_boolean(true);
        } else {
            self.rb_single_fiducial_surface.set_selected_boolean(true);
        }

        self.ltf_residual_threshold
            .set_text_double(params.get_residual_report_criterion().get_double().abs());
        if params.get_residual_report_criterion().get_double() < 0.0 {
            self.rb_resid_neighboring.set_selected_boolean(true);
        } else {
            self.rb_resid_all_views.set_selected_boolean(true);
        }
        self.cb_cross_validate
            .set_selected_boolean(params.is_cross_validate());

        if params.is_exclude_list_available() {
            self.ltf_exclude_list.set_enabled(true);
            self.ltf_exclude_list
                .set_text_string(Some(&params.get_exclude_list()));
        } else {
            self.ltf_exclude_list.set_enabled(false);
        }

        self.ltf_separate_view_groups
            .set_text_string(Some(&params.get_separate_group()));
        self.ltf_tilt_angle_offset
            .set_text_string(Some(&params.get_angle_offset().to_string()));
        self.ltf_tilt_axis_z_shift
            .set_text_string(Some(&params.get_axis_z_shift().to_string()));

        self.ctf_robust_fitting_and_k_factor_scaling
            .set_selected_boolean(params.is_robust_fitting());
        self.ctf_robust_fitting_and_k_factor_scaling
            .set_text_string(Some(&params.get_k_factor_scaling()));
        if self.cb_weight_whole_tracks.is_enabled() {
            self.cb_weight_whole_tracks
                .set_selected_boolean(params.is_weight_whole_tracks());
        }
        self.ltf_metro_factor
            .set_text_string(Some(&params.get_metro_factor().to_string()));
        self.ltf_maximum_cycles
            .set_text_string(Some(&params.get_maximum_cycles().to_string()));

        self.cb_local_alignments
            .set_selected_boolean(params.get_local_alignments().is());
        if !params.is_target_patch_size_xand_y_empty() {
            self.rtf_target_patch_size_xand_y.set_selected_boolean(true);
            self.rtf_target_patch_size_xand_y
                .set_text_string(Some(&params.get_target_patch_size_xand_y()));
        } else if !params.is_number_of_local_patches_xand_y_empty() {
            self.rtf_n_local_patches.set_selected_boolean(true);
            self.rtf_n_local_patches
                .set_text_string(Some(&params.get_number_of_local_patches_xand_y()));
        }
        self.created_day_stamp_imod_5_0_1
            .set(params.is_created_day_stamp_imod_5_0_1());
        if params.is_created_day_stamp_imod_5_0_1() {
            self.rtf_n_local_patches
                .set_label(Some("# of full-field patches (x, y)"));
        }
        self.set_min_local_patch_size_label();
        self.ltf_min_local_patch_size
            .set_text_string(Some(&params.get_min_size_or_overlap_xand_y()));
        self.ltf_min_local_fiducials
            .set_text_string(Some(&params.get_min_fids_total_and_each_surface()));
        self.cb_fix_xyz_coordinates
            .set_selected_boolean(params.get_fix_xyz_coordinates().is());

        // Tilt angle solution parameters
        let mut solution_type = params.get_tilt_option().get_int();
        if solution_type == 0 {
            self.rb_tilt_angle_fixed.set_selected_boolean(true);
        }
        if solution_type == 2 {
            self.rb_tilt_angle_all.set_selected_boolean(true);
        }
        if solution_type == 5 {
            self.rb_tilt_angle_automap.set_selected_boolean(true);
        }
        self.ltf_tilt_angle_group_size
            .set_text_string(Some(&params.get_tilt_default_grouping().to_string()));
        self.ltf_tilt_angle_non_default_groups
            .set_text_string(Some(&params.get_tilt_nondefault_group()));

        // Magnification solution parameters
        // TODO what to do if the magnification type is not one of the cases
        // below
        self.ltf_magnification_reference_view
            .set_text_string(Some(&params.get_mag_reference_view().to_string()));
        solution_type = params.get_mag_option().get_int();
        if solution_type == 0 {
            self.rb_magnification_fixed.set_selected_boolean(true);
        }
        if solution_type == 1 {
            self.rb_magnification_all.set_selected_boolean(true);
        }
        if solution_type == const_tiltalign_param::AUTOMAPPED_OPTION {
            self.rb_magnification_automap.set_selected_boolean(true);
        }
        self.ltf_magnification_group_size
            .set_text_string(Some(&params.get_mag_default_grouping().to_string()));
        self.ltf_magnification_non_default_groups
            .set_text_string(Some(&params.get_mag_nondefault_group()));

        // Rotation solution parameters
        solution_type = params.get_rot_option().get_int();
        if solution_type == 0 {
            self.rb_rotation_none.set_selected_boolean(true);
        }
        if solution_type == 1 {
            self.rb_rotation_all.set_selected_boolean(true);
        }
        if solution_type == const_tiltalign_param::AUTOMAPPED_OPTION {
            self.rb_rotation_automap.set_selected_boolean(true);
        }
        if solution_type == const_tiltalign_param::SINGLE_OPTION {
            self.rb_rotation_one.set_selected_boolean(true);
        }
        self.ltf_rotation_angle
            .set_text_string(Some(&params.get_rotation_angle().to_string()));
        self.ltf_rotation_group_size
            .set_text_string(Some(&params.get_rot_default_grouping().to_string()));
        self.ltf_rotation_non_default_groups
            .set_text_string(Some(&params.get_rot_nondefault_group()));

        // Compression solution parameters
        // (commented out in the Java: ltfCompressionReferenceView,
        // rbCompressionAll, rbCompressionAutomapLinear/Fixed,
        // ltfCompressionGroupSize, ltfCompressionAdditionalGroups)
        // Global distortion solution type
        let mut x_stretch_solution_type = params.get_x_stretch_option().get_int();
        let mut skew_solution_type = params.get_skew_option().get_int();
        if x_stretch_solution_type == 0 && skew_solution_type == 0 {
            self.rb_distortion_disabled.set_selected_boolean(true);
        } else if x_stretch_solution_type == 3 && skew_solution_type == 3 {
            self.rb_distortion_full_solution.set_selected_boolean(true);
        } else {
            self.rb_distortion_skew.set_selected_boolean(true);
        }
        self.ltf_xstretch_group_size
            .set_text_string(Some(&params.get_x_stretch_default_grouping().to_string()));
        self.ltf_xstretch_non_default_groups
            .set_text_string(Some(&params.get_x_stretch_nondefault_group()));
        // skew solution parameters
        self.ltf_skew_group_size
            .set_text_string(Some(&params.get_skew_default_grouping().to_string()));
        self.ltf_skew_non_default_groups
            .set_text_string(Some(&params.get_skew_nondefault_group()));

        self.cb_projection_stretch
            .set_selected_boolean(params.get_projection_stretch().is());

        // Local rotation solution parameters
        // NOTE this is brittle since we are mapping a numeric value to a boolean
        // at David's request
        solution_type = params.get_local_rot_option().get_int();
        if solution_type == 0 {
            self.cb_local_rotation.set_selected_boolean(false);
        } else {
            self.cb_local_rotation.set_selected_boolean(true);
        }
        self.ltf_local_rotation_group_size
            .set_text_string(Some(&params.get_local_rot_default_grouping().to_string()));
        self.ltf_local_rotation_non_default_groups
            .set_text_string(Some(&params.get_local_rot_nondefault_group()));

        // Local tilt angle solution parameters
        solution_type = params.get_local_tilt_option().get_int();
        if solution_type == 0 {
            self.cb_local_tilt_angle.set_selected_boolean(false);
        } else {
            self.cb_local_tilt_angle.set_selected_boolean(true);
        }
        self.ltf_local_tilt_angle_group_size
            .set_text_string(Some(&params.get_local_tilt_default_grouping().to_string()));
        self.ltf_local_tilt_angle_non_default_groups
            .set_text_string(Some(&params.get_local_tilt_nondefault_group()));

        // Local magnification solution parameters
        solution_type = params.get_local_mag_option().get_int();
        if solution_type == 0 {
            self.cb_local_magnification.set_selected_boolean(false);
        } else {
            self.cb_local_magnification.set_selected_boolean(true);
        }
        self.ltf_local_magnification_group_size
            .set_text_string(Some(&params.get_local_mag_default_grouping().to_string()));
        self.ltf_local_magnification_non_default_groups
            .set_text_string(Some(&params.get_local_mag_nondefault_group()));

        // Local distortion solution type
        x_stretch_solution_type = params.get_local_x_stretch_option().get_int();
        skew_solution_type = params.get_local_skew_option().get_int();
        if x_stretch_solution_type == 0 && skew_solution_type == 0 {
            self.rb_local_distortion_disabled.set_selected_boolean(true);
        } else if x_stretch_solution_type == 3 && skew_solution_type == 3 {
            self.rb_local_distortion_full_solution
                .set_selected_boolean(true);
        } else {
            self.rb_local_distortion_skew.set_selected_boolean(true);
        }
        self.ltf_local_xstretch_group_size.set_text_string(Some(
            &params.get_local_x_stretch_default_grouping().to_string(),
        ));
        self.ltf_local_xstretch_non_default_groups
            .set_text_string(Some(&params.get_local_x_stretch_nondefault_group()));
        // Local skew solution parameters
        self.ltf_local_skew_group_size
            .set_text_string(Some(&params.get_local_skew_default_grouping().to_string()));
        self.ltf_local_skew_non_default_groups
            .set_text_string(Some(&params.get_local_skew_nondefault_group()));

        // Beam tilt
        if params
            .get_beam_tilt_option()
            .equals_int(const_tiltalign_param::BEAM_SEARCH_OPTION)
        {
            self.rb_solve_for_beam_tilt.set_selected_boolean(true);
        } else {
            // rbNoBeamTilt and rtfFixedBeamTilt are both option 0. Option 1 is not working.
            // (The Java also tests `fixedOrInitialBeamTilt == null`; the Rust
            // getter returns a value, never null.)
            let fixed_or_initial_beam_tilt = params.get_fixed_or_initial_beam_tilt();
            if fixed_or_initial_beam_tilt.is_null() {
                self.rb_no_beam_tilt.set_selected_boolean(true);
            } else {
                self.rtf_fixed_beam_tilt.set_selected_boolean(true);
                self.rtf_fixed_beam_tilt
                    .set_text_const_etomo_number(&fixed_or_initial_beam_tilt);
            }
        }

        self.cb_x_tilt_automap_same
            .set_selected_boolean(params.is_x_tilt_option_automap_same());
        // Set the UI to match the data
        self.enable_fields();
        self.update_display();
    }

    /// Java `setParameters(RestrictalignParam)`.
    pub fn set_parameters_restrictalign_param(&self, param: &RestrictalignParam) {
        if param.is_target_measurement_ratio() {
            self.ltf_target_measurement_ratio
                .set_text_string(Some(&param.get_target_measurement_ratio()));
        }
        if param.is_min_measurement_ratio() {
            self.ltf_min_measurement_ratio
                .set_text_string(Some(&param.get_min_measurement_ratio()));
        }
        self.bg_local_align_validation
            .set_selected(Some(&EnumeratedTypeRef::new(
                LocalAlignValidation::get_instance(Some(&param.get_local_align_validation()))
                    .clone(),
            )));
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_target_patch_size_x_and_y(
            self.rtf_target_patch_size_xand_y.get_text_void().as_deref(),
        );
        meta_data.set_number_of_local_patches_x_and_y(
            self.rtf_n_local_patches.get_text_void().as_deref(),
        );
        meta_data.set_no_beam_tilt_selected(self.axis_id, self.rb_no_beam_tilt.is_selected());
        meta_data
            .set_fixed_beam_tilt_selected(self.axis_id, self.rtf_fixed_beam_tilt.is_selected());
        meta_data.set_fixed_beam_tilt(
            self.axis_id,
            self.rtf_fixed_beam_tilt.get_text_void().as_deref(),
        );
        meta_data.set_weight_whole_tracks(self.axis_id, self.cb_weight_whole_tracks.is_selected());
        meta_data.set_target_measurement_ratio(
            self.axis_id,
            self.ltf_target_measurement_ratio.get_text_void().as_deref(),
        );
        meta_data.set_min_measurement_ratio(
            self.axis_id,
            self.ltf_min_measurement_ratio.get_text_void().as_deref(),
        );
        // ((RadioButton.RadioButtonModel) bgLocalAlignValidation.getSelection())
        // .getEnumeratedType().getValue()
        let value = (self
            .bg_local_align_validation
            .get_button_group()
            .get_selection())
        .and_then(|button| button.get_model())
        .and_then(|model| {
            model
                .as_any()
                .downcast_ref::<RadioButtonModel>()
                .and_then(|model| model.get_enumerated_type())
        })
        .map(|enumerated_type| enumerated_type.get_value());
        meta_data.set_fine_local_align_validation(self.axis_id, value.as_ref());
    }

    /// Java `setDefaultParameters()`.
    pub fn set_default_parameters(&self) {
        self.ltf_target_measurement_ratio.use_default_value();
        self.ltf_min_measurement_ratio.use_default_value();
    }

    /// Java `setParameters(ConstMetaData)`.
    ///
    /// Backwards compatibility: setParameters(ConstMetaData) must be called
    /// before setParameters(TiltalignParam).
    pub fn set_parameters_const_meta_data(&self, meta_data: &MetaData) {
        self.rtf_target_patch_size_xand_y
            .set_text_string(Some(&meta_data.get_target_patch_size_x_and_y()));
        self.rtf_n_local_patches
            .set_text_string(Some(&meta_data.get_number_of_local_patches_x_and_y()));
        self.rb_no_beam_tilt
            .set_selected_boolean(meta_data.get_no_beam_tilt_selected(self.axis_id).is());
        self.rtf_fixed_beam_tilt
            .set_selected_boolean(meta_data.get_fixed_beam_tilt_selected(self.axis_id).is());
        self.rtf_fixed_beam_tilt
            .set_text_const_etomo_number(&meta_data.get_fixed_beam_tilt(self.axis_id));
        self.cb_weight_whole_tracks
            .set_selected_boolean(meta_data.get_weight_whole_tracks(self.axis_id));
        if meta_data.is_target_measurement_ratio_set(self.axis_id) {
            self.ltf_target_measurement_ratio
                .set_text_string(Some(&meta_data.get_target_measurement_ratio(self.axis_id)));
        }
        if meta_data.is_min_measurement_ratio_set(self.axis_id) {
            self.ltf_min_measurement_ratio
                .set_text_string(Some(&meta_data.get_min_measurement_ratio(self.axis_id)));
        }
        self.bg_local_align_validation
            .set_selected(Some(&EnumeratedTypeRef::new(
                LocalAlignValidation::get_instance(Some(
                    &meta_data.get_fine_local_align_validation(self.axis_id),
                ))
                .clone(),
            )));
        self.update_display();
    }

    /// Java `setParameters(BaseScreenState)`.
    pub fn set_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.ph_single_variables
            .set_button_states_base_screen_state(Some(screen_state));
    }

    /// Java `getParameters(RestrictalignParam, boolean)`.
    pub fn get_parameters_restrictalign_param_boolean(
        &self,
        param: &mut RestrictalignParam,
        do_validation: bool,
    ) -> bool {
        // Java try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            if self.ltf_target_measurement_ratio.is_enabled() {
                param.set_target_measurement_ratio(
                    self.ltf_target_measurement_ratio
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                param.set_min_measurement_ratio(
                    self.ltf_min_measurement_ratio
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
            } else {
                param.reset_target_measurement_ratio();
                param.reset_min_measurement_ratio();
            }
            if self
                .rb_local_align_validation_area_requirements
                .is_enabled()
            {
                let value = (self
                    .bg_local_align_validation
                    .get_button_group()
                    .get_selection())
                .and_then(|button| button.get_model())
                .and_then(|model| {
                    model
                        .as_any()
                        .downcast_ref::<RadioButtonModel>()
                        .and_then(|model| model.get_enumerated_type())
                })
                .map(|enumerated_type| enumerated_type.get_value());
                param.set_local_align_validation(value.as_ref());
            } else {
                param.reset_local_align_validation();
            }
            Ok(true)
        })();
        match result {
            Ok(value) => value,
            Err(_) => false,
        }
    }

    /// Java `getParameters(MakecomfileParam, boolean)`.
    pub fn get_parameters_makecomfile_param_boolean(
        &self,
        param: &mut MakecomfileParam,
        do_validation: bool,
    ) -> bool {
        // Java try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if self
                .rb_local_align_validation_area_requirements
                .is_enabled()
            {
                // The group always has a selection: LocalAlignValidation.DEFAULT
                // selects itself when its radio button is constructed.
                let enumerated_type = (self
                    .bg_local_align_validation
                    .get_button_group()
                    .get_selection())
                .and_then(|button| button.get_model())
                .and_then(|model| {
                    model
                        .as_any()
                        .downcast_ref::<RadioButtonModel>()
                        .and_then(|model| model.get_enumerated_type())
                })
                .expect("bgLocalAlignValidation has a selected RadioButtonModel");
                param.set_local_align_validation(enumerated_type.get_value().get_int());
            } else {
                param.reset_local_align_validation();
            }
            if self.ltf_target_measurement_ratio.is_enabled() {
                let mut target_and_min_ratios =
                    FortranInputString::new(makecomfile_param::TARGET_AND_MIN_RATIOS_NPARAMS);
                target_and_min_ratios.set_index_string(
                    0,
                    self.ltf_target_measurement_ratio
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                target_and_min_ratios.set_index_string(
                    1,
                    self.ltf_min_measurement_ratio
                        .get_text_boolean(do_validation)?
                        .as_deref(),
                );
                param.set_target_and_min_ratios(&target_and_min_ratios);
            } else {
                param.reset_target_and_min_ratios();
            }
            Ok(())
        })();
        if result.is_err() {
            return false;
        }
        true
    }

    /// Java `setPatchTracking(boolean)`.
    pub fn set_patch_tracking(&self, input: bool) {
        self.patch_tracking.set(input);
        self.update_display();
    }

    /// Java `setSurfacesToAnalyze(int)`: selects a fiducial surface radio
    /// button depending on surfacesToAnalyze.  Only the surfacesToAnalyze
    /// values 1 and 2 have an effect.
    pub fn set_surfaces_to_analyze(&self, surfaces_to_analyze: i32) {
        if surfaces_to_analyze == 1 {
            self.rb_single_fiducial_surface.set_selected_boolean(true);
            self.cb_weight_whole_tracks.set_selected_boolean(true);
        } else if surfaces_to_analyze == 2 {
            self.rb_dual_fiducial_surfaces.set_selected_boolean(true);
            self.cb_weight_whole_tracks.set_selected_boolean(false);
        }
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.ph_single_variables
            .get_button_states(Some(screen_state));
    }

    /// Java `getParameters(TiltalignParam, boolean) throws
    /// FortranInputSyntaxException`.
    ///
    /// Get the values from the panel by updating tiltalign parameter object.
    /// Currently this makes the assumption that the argument contains valid
    /// parameters and that only the known parameters will be changed.
    /// getParameters(MetaData) must be called before
    /// getParameters(TiltalignParam).
    pub fn get_parameters_tiltalign_param_boolean(
        &self,
        params: &mut TiltalignParam,
        do_validation: bool,
    ) -> Result<bool, TiltalignParamsException> {
        /// The exceptions the Java `try` blocks of this method see.
        enum Thrown {
            FieldValidationFailed(FieldValidationFailedException),
            FortranInputSyntax(FortranInputSyntaxException),
            NumberFormat(String),
        }
        let app_mgr = self.app_mgr;
        let axis_id = self.axis_id;
        // Java outer try { ... } catch (FieldValidationFailedException e) {
        // return false; }
        let outer = (|| -> Result<(), Thrown> {
            let mut bad_parameter = String::new();
            let images_are_binned = UIExpertUtilities::INSTANCE
                .get_stack_binning_base_manager_axis_id_file_type_boolean(
                    app_mgr,
                    axis_id,
                    &file_type::CLASS.prealigned_stack,
                    true,
                );
            if images_are_binned != const_etomo_number::INTEGER_NULL_VALUE {
                params.set_images_are_binned(images_are_binned);
                NULL_IMAGES_ARE_BINNED_REPORTED.store(false, Ordering::Relaxed);
            } else if !NULL_IMAGES_ARE_BINNED_REPORTED.load(Ordering::Relaxed) && do_validation {
                let message = format!(
                    "{}{}{}{}{}",
                    "The coarse aligned stack is missing and the stack binning will be assumed to ",
                    "be ",
                    params.get_images_are_binned(),
                    " when running Tiltalign.  If this is not correct, remake the coarse ",
                    "aligned stack; otherwise residual errors will be scaled wrong."
                );
                ui_harness::INSTANCE.with(|harness| {
                    harness.open_warning_message_dialog_base_manager_ui_component_string_string(
                        Some(app_mgr),
                        Some(self as &dyn UIComponent),
                        &message,
                        "Missing coarse aligned stack",
                    )
                });
                NULL_IMAGES_ARE_BINNED_REPORTED.store(true, Ordering::Relaxed);
            }
            params.update_image_file();
            // Java inner try { ... } catch (FortranInputSyntaxException except)
            // ... catch (NumberFormatException except).
            let inner = (|| -> Result<(), Thrown> {
                if self.rb_dual_fiducial_surfaces.is_selected() {
                    params.set_surfaces_to_analyze(2);
                } else {
                    params.set_surfaces_to_analyze(1);
                }

                bad_parameter = self.ltf_residual_threshold.get_label();
                let mut resid = const_etomo_number::java_lang_double_value_of(
                    &self
                        .ltf_residual_threshold
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .unwrap_or_default(),
                )
                .map_err(Thrown::NumberFormat)?;
                if self.rb_resid_neighboring.is_selected() {
                    resid *= -1.0;
                }
                params.set_residual_report_criterion(resid);
                params.set_cross_validate(self.cb_cross_validate.is_selected());

                // Currently only supports Exclude list or blank entries
                bad_parameter = self.ltf_exclude_list.get_label();
                if self.ltf_exclude_list.is_enabled() {
                    params.set_exclude_list(
                        self.ltf_exclude_list
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    );
                }
                bad_parameter = self.ltf_separate_view_groups.get_label();
                params.set_separate_group(
                    self.ltf_separate_view_groups
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_tilt_angle_offset.get_label();
                params.set_angle_offset(
                    self.ltf_tilt_angle_offset
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_tilt_axis_z_shift.get_label();
                params.set_axis_z_shift(
                    self.ltf_tilt_axis_z_shift
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self
                    .ctf_robust_fitting_and_k_factor_scaling
                    .get_label()
                    .to_string();
                params
                    .set_robust_fitting(self.ctf_robust_fitting_and_k_factor_scaling.is_selected());
                params.set_k_factor_scaling(
                    self.ctf_robust_fitting_and_k_factor_scaling
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                if self.cb_weight_whole_tracks.is_enabled() {
                    bad_parameter = self
                        .cb_weight_whole_tracks
                        .get_text_void()
                        .unwrap_or_else(|| "null".to_string());
                    params.set_weight_whole_tracks(self.cb_weight_whole_tracks.is_selected());
                } else {
                    params.reset_weight_whole_tracks();
                }

                bad_parameter = self.ltf_metro_factor.get_label();
                params.set_metro_factor(
                    self.ltf_metro_factor
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_maximum_cycles.get_label();
                params.set_maximum_cycles(
                    self.ltf_maximum_cycles
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self
                    .cb_local_alignments
                    .get_text_void()
                    .unwrap_or_else(|| "null".to_string());
                params.set_local_alignments(self.cb_local_alignments.is_selected());

                bad_parameter = self
                    .rtf_target_patch_size_xand_y
                    .get_label()
                    .unwrap_or_else(|| "null".to_string());
                params.set_target_patch_size_xand_y_active(
                    self.rtf_target_patch_size_xand_y.is_selected(),
                );
                params
                    .set_target_patch_size_xand_y(
                        self.rtf_target_patch_size_xand_y
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self
                    .rtf_n_local_patches
                    .get_label()
                    .unwrap_or_else(|| "null".to_string());
                params.set_number_of_local_patches_xand_y_active(
                    self.rtf_n_local_patches.is_selected(),
                );
                params
                    .set_number_of_local_patches_xand_y(
                        self.rtf_n_local_patches
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_min_local_patch_size.get_label();
                params
                    .set_min_size_or_overlap_xand_y(
                        self.ltf_min_local_patch_size
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_min_local_fiducials.get_label();
                params
                    .set_min_fids_total_and_each_surface(
                        self.ltf_min_local_fiducials
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self
                    .cb_fix_xyz_coordinates
                    .get_text_void()
                    .unwrap_or_else(|| "null".to_string());
                params.set_fix_xyz_coordinates(self.cb_fix_xyz_coordinates.is_selected());

                // Tilt angle pane
                let mut r#type = 0;
                if self.rb_tilt_angle_fixed.is_selected() {
                    r#type = 0;
                }
                if self.rb_tilt_angle_all.is_selected() {
                    r#type = 2;
                }
                if self.rb_tilt_angle_automap.is_selected() {
                    r#type = 5;
                }
                params.set_tilt_option(r#type);
                bad_parameter = self.ltf_tilt_angle_group_size.get_label();
                params.set_tilt_default_grouping(
                    self.ltf_tilt_angle_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_tilt_angle_non_default_groups.get_label();
                params
                    .set_tilt_nondefault_group(
                        self.ltf_tilt_angle_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // Magnification pane
                bad_parameter = self.ltf_magnification_reference_view.get_label();
                params.set_mag_reference_view(
                    self.ltf_magnification_reference_view
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                if self.rb_magnification_fixed.is_selected() {
                    r#type = 0;
                }
                if self.rb_magnification_all.is_selected() {
                    r#type = 1;
                }
                if self.rb_magnification_automap.is_selected() {
                    r#type = const_tiltalign_param::AUTOMAPPED_OPTION;
                }
                params.set_mag_option(r#type);

                bad_parameter = self.ltf_magnification_group_size.get_label();
                params.set_mag_default_grouping(
                    self.ltf_magnification_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_magnification_non_default_groups.get_label();
                params
                    .set_mag_nondefault_group(
                        self.ltf_magnification_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // Rotation pane
                if self.rb_rotation_none.is_selected() {
                    r#type = 0;
                }
                if self.rb_rotation_all.is_selected() {
                    r#type = 1;
                }
                if self.rb_rotation_automap.is_selected() {
                    r#type = const_tiltalign_param::AUTOMAPPED_OPTION;
                }
                if self.rb_rotation_one.is_selected() {
                    r#type = const_tiltalign_param::SINGLE_OPTION;
                }
                params.set_rot_option(r#type);
                bad_parameter = self.ltf_rotation_angle.get_label();
                params.set_rotation_angle(
                    self.ltf_rotation_angle
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );
                bad_parameter = self.ltf_rotation_group_size.get_label();
                params.set_rot_default_grouping(
                    self.ltf_rotation_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );
                bad_parameter = self.ltf_rotation_non_default_groups.get_label();
                params
                    .set_rot_nondefault_group(
                        self.ltf_rotation_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // Distortion pane
                r#type = 0;
                let _ = r#type;
                // Set the necessary types for distortion xstretch and skew
                if self.rb_distortion_disabled.is_selected() {
                    params.set_skew_option(const_tiltalign_param::FIXED_OPTION);
                    params.set_x_stretch_option(const_tiltalign_param::FIXED_OPTION);
                } else {
                    params.set_skew_option(const_tiltalign_param::AUTOMAPPED_OPTION);
                    if self.rb_distortion_full_solution.is_selected() {
                        params.set_x_stretch_option(const_tiltalign_param::AUTOMAPPED_OPTION);
                    } else {
                        params.set_x_stretch_option(const_tiltalign_param::FIXED_OPTION);
                    }
                }

                bad_parameter = self.ltf_skew_group_size.get_label();
                params.set_skew_default_grouping(
                    self.ltf_skew_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_skew_non_default_groups.get_label();
                params
                    .set_skew_nondefault_group(
                        self.ltf_skew_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_xstretch_group_size.get_label();
                params.set_x_stretch_default_grouping(
                    self.ltf_xstretch_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_xstretch_non_default_groups.get_label();
                params
                    .set_x_stretch_nondefault_group(
                        self.ltf_xstretch_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self
                    .cb_projection_stretch
                    .get_text_void()
                    .unwrap_or_else(|| "null".to_string());
                params.set_projection_stretch(self.cb_projection_stretch.is_selected());

                // Get the local alignment parameters
                // Rotation pane
                // NOTE this only works if 0 and 5 are valid local tilt angle codes
                r#type = 0;
                if self.cb_local_rotation.is_selected() {
                    r#type = params.get_local_rot_option().get_display_integer();
                }
                params.set_local_rot_option(r#type);
                bad_parameter = self.ltf_local_rotation_group_size.get_label();
                params.set_local_rot_default_grouping(
                    self.ltf_local_rotation_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_local_rotation_non_default_groups.get_label();
                params
                    .set_local_rot_nondefault_group(
                        self.ltf_local_rotation_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // Tilt angle pane
                r#type = 0;
                if self.cb_local_tilt_angle.is_selected() {
                    r#type = params.get_local_tilt_option().get_display_integer();
                }
                params.set_local_tilt_option(r#type);
                bad_parameter = self.ltf_local_tilt_angle_group_size.get_label();
                params.set_local_tilt_default_grouping(
                    self.ltf_local_tilt_angle_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_local_tilt_angle_non_default_groups.get_label();
                params
                    .set_local_tilt_nondefault_group(
                        self.ltf_local_tilt_angle_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // Local magnification pane
                if self.cb_local_magnification.is_selected() {
                    let local_mag_option = params.get_local_mag_option().get_display_integer();
                    params.set_local_mag_option(local_mag_option);
                } else {
                    params.set_local_mag_option(0);
                }

                bad_parameter = self.ltf_local_magnification_group_size.get_label();
                params.set_local_mag_default_grouping(
                    self.ltf_local_magnification_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_local_magnification_non_default_groups.get_label();
                params
                    .set_local_mag_nondefault_group(
                        self.ltf_local_magnification_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                // Distortion pane
                r#type = 0;
                let _ = r#type;
                if self.rb_local_distortion_disabled.is_selected() {
                    params.set_local_skew_option(const_tiltalign_param::FIXED_OPTION);
                    params.set_local_x_stretch_option(const_tiltalign_param::FIXED_OPTION);
                } else {
                    params.set_local_skew_option(const_tiltalign_param::AUTOMAPPED_OPTION);
                    if self.rb_local_distortion_full_solution.is_selected() {
                        params.set_local_x_stretch_option(const_tiltalign_param::AUTOMAPPED_OPTION);
                    } else {
                        params.set_local_x_stretch_option(const_tiltalign_param::FIXED_OPTION);
                    }
                }
                bad_parameter = self.ltf_local_skew_group_size.get_label();
                params.set_local_skew_default_grouping(
                    self.ltf_local_skew_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_local_skew_non_default_groups.get_label();
                params
                    .set_local_skew_nondefault_group(
                        self.ltf_local_skew_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;

                bad_parameter = self.ltf_local_xstretch_group_size.get_label();
                params.set_local_x_stretch_default_grouping(
                    self.ltf_local_xstretch_group_size
                        .get_text_boolean(do_validation)
                        .map_err(Thrown::FieldValidationFailed)?
                        .as_deref(),
                );

                bad_parameter = self.ltf_local_xstretch_non_default_groups.get_label();
                params
                    .set_local_x_stretch_nondefault_group(
                        self.ltf_local_xstretch_non_default_groups
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    )
                    .map_err(Thrown::FortranInputSyntax)?;
                // params needs to have other values set before it can set OutputZFactorFile
                params.set_output_z_factor_file();

                if self.rb_no_beam_tilt.is_selected() {
                    bad_parameter = self
                        .rb_no_beam_tilt
                        .get_text_void()
                        .unwrap_or_else(|| "null".to_string());
                    params.set_beam_tilt_option(const_tiltalign_param::FIXED_OPTION);
                    params.reset_fixed_or_initial_beam_tilt();
                } else if self.rtf_fixed_beam_tilt.is_selected() {
                    bad_parameter = self
                        .rtf_fixed_beam_tilt
                        .get_label()
                        .unwrap_or_else(|| "null".to_string());
                    params.set_beam_tilt_option(const_tiltalign_param::FIXED_OPTION);
                    params.set_fixed_or_initial_beam_tilt(
                        self.rtf_fixed_beam_tilt
                            .get_text_boolean(do_validation)
                            .map_err(Thrown::FieldValidationFailed)?
                            .as_deref(),
                    );
                } else if self.rb_solve_for_beam_tilt.is_selected() {
                    bad_parameter = self
                        .rb_solve_for_beam_tilt
                        .get_text_void()
                        .unwrap_or_else(|| "null".to_string());
                    params.set_beam_tilt_option(const_tiltalign_param::BEAM_SEARCH_OPTION);
                    params.reset_fixed_or_initial_beam_tilt();
                }
                params
                    .set_x_tilt_automap_same_large_group(self.cb_x_tilt_automap_same.is_selected());
                Ok(())
            })();
            match inner {
                // catch (FortranInputSyntaxException except) {
                //   String message = badParameter + " " + except.getMessage();
                //   throw new FortranInputSyntaxException(message); }
                Err(Thrown::FortranInputSyntax(except)) => {
                    let message = format!(
                        "{} {}",
                        bad_parameter,
                        except.get_message().unwrap_or("null")
                    );
                    Err(Thrown::FortranInputSyntax(
                        FortranInputSyntaxException::new(&message),
                    ))
                }
                // catch (NumberFormatException except) {
                //   String message = badParameter + " " + except.getMessage();
                //   throw new NumberFormatException(message); }
                Err(Thrown::NumberFormat(except)) => {
                    let message = format!("{} {}", bad_parameter, except);
                    Err(Thrown::NumberFormat(message))
                }
                other => other,
            }
        })();
        match outer {
            Ok(()) => Ok(true),
            // catch (FieldValidationFailedException e) { return false; }
            Err(Thrown::FieldValidationFailed(_)) => Ok(false),
            Err(Thrown::FortranInputSyntax(except)) => Err(
                TiltalignParamsException::FortranInputSyntaxException(except),
            ),
            Err(Thrown::NumberFormat(message)) => {
                Err(TiltalignParamsException::NumberFormatException(message))
            }
        }
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        if self.rtf_fixed_beam_tilt.is_selected()
            && self.rtf_fixed_beam_tilt.get_text_void().as_deref() == Some("")
        {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.app_mgr),
                    &format!(
                        "{} can not be empty when it is selected.",
                        self.rtf_fixed_beam_tilt
                            .get_label()
                            .as_deref()
                            .unwrap_or("null")
                    ),
                    "Entry Error",
                )
            });
            return false;
        }
        if self.patch_tracking.get() && self.rb_dual_fiducial_surfaces.is_selected() {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
    Some(self.app_mgr),
                    &format!(
                        "Patch tracking puts fiducials only on one side.  Select \"{}\" for better results.",
                        self.rb_single_fiducial_surface
                            .get_text_void()
                            .as_deref()
                            .unwrap_or("null")
                    ),
                    "Entry Warning",
                    Some(self.axis_id),
                )
            });
            // This is just a warning so don't return false.
        }
        true
    }

    /// Java `setFirstTab()`.
    pub fn set_first_tab(&self) {
        // JTabbedPane.setSelectedComponent: select the tab holding the
        // component (jdk.rs models it through the tab index).
        let tab_pane = self.tab_pane.get_component();
        let component = self.pnl_general.get_component();
        let index = tab_pane
            .get_components()
            .iter()
            .position(|tab| Rc::ptr_eq(tab, &component));
        if let Some(index) = index {
            tab_pane.set_selected_tab(index as i32);
        }
    }

    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, state: bool) {
        self.update_advanced_beam_tilt(state);
        self.ltf_metro_factor.set_visible(state);
        self.ltf_maximum_cycles.set_visible(state);
        self.ltf_magnification_reference_view.set_visible(state);
        self.ltf_rotation_non_default_groups.set_visible(state);
        self.ltf_tilt_angle_non_default_groups.set_visible(state);
        self.ltf_magnification_non_default_groups.set_visible(state);
        self.ltf_xstretch_non_default_groups.set_visible(state);
        self.ltf_skew_non_default_groups.set_visible(state);
        self.ltf_local_rotation_non_default_groups
            .set_visible(state);
        self.ltf_local_tilt_angle_non_default_groups
            .set_visible(state);
        self.ltf_local_magnification_non_default_groups
            .set_visible(state);
        self.ltf_local_xstretch_non_default_groups
            .set_visible(state);
        self.ltf_local_skew_non_default_groups.set_visible(state);
        self.ltf_min_local_patch_size.set_visible(state);
        self.cb_fix_xyz_coordinates.set_visible(state);
    }

    /// Java private `enableLocalAlignmentDependents()`: local alignment state.
    fn enable_local_alignment_dependents(&self) {
        let state = self.cb_local_alignments.is_selected();
        self.l_local_align_validation.set_enabled(state);
        self.rb_local_align_validation_area_requirements
            .set_enabled(state);
        self.rb_local_align_validation_variables.set_enabled(state);
        self.rb_local_align_validation_both.set_enabled(state);
        self.ltf_target_measurement_ratio.set_enabled(!state);
        self.ltf_min_measurement_ratio.set_enabled(!state);
        self.rtf_target_patch_size_xand_y.set_enabled(state);
        self.rtf_n_local_patches.set_enabled(state);
        self.ltf_min_local_patch_size.set_enabled(state);
        self.ltf_min_local_fiducials.set_enabled(state);
        self.cb_fix_xyz_coordinates.set_enabled(state);
        let tab_pane = self.tab_pane.get_component();
        // JTabbedPane.indexOfComponent: the tab index holding the component.
        let component = self.pnl_local_solution.get_component();
        let index = tab_pane
            .get_components()
            .iter()
            .position(|tab| Rc::ptr_eq(tab, &component));
        if let Some(index) = index {
            tab_pane.set_enabled_at(index, state);
        }
    }

    /// Java private `enableFields()`: signal each pane to update its
    /// enabled/disabled state.
    fn enable_fields(&self) {
        // update all of the enable/disable states
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

    /// Java private `enableTiltAngleSolutionFields()`: update the
    /// enabled/disabled state of the specified solution panel.
    fn enable_tilt_angle_solution_fields(&self) {
        let state = self.rb_tilt_angle_automap.is_selected();
        self.ltf_tilt_angle_group_size.set_enabled(state);
        self.ltf_tilt_angle_non_default_groups.set_enabled(state);
    }

    /// Java private `enableMagnificationSolutionFields()`.
    fn enable_magnification_solution_fields(&self) {
        let state = self.rb_magnification_automap.is_selected();
        self.ltf_magnification_group_size.set_enabled(state);
        self.ltf_magnification_non_default_groups.set_enabled(state);
    }

    /// Java private `enableRotationSolutionFields()`.
    fn enable_rotation_solution_fields(&self) {
        let state = self.rb_rotation_automap.is_selected();
        self.ltf_rotation_group_size.set_enabled(state);
        self.ltf_rotation_non_default_groups.set_enabled(state);
        self.ltf_rotation_angle
            .set_enabled(self.rb_rotation_none.is_selected());
    }

    /// Java private `setDistortionSolutionState()`.
    fn set_distortion_solution_state(&self) {
        if self.rb_distortion_disabled.is_selected() {
            self.rb_local_distortion_disabled.set_selected_boolean(true);
        } else {
            self.rb_tilt_angle_automap.set_selected_boolean(true);
            self.enable_tilt_angle_solution_fields();
            if self.rb_distortion_full_solution.is_selected() {
                self.rb_local_distortion_full_solution
                    .set_selected_boolean(true);
            } else if self.rb_distortion_skew.is_selected() {
                self.rb_local_distortion_skew.set_selected_boolean(true);
            }
        }
        self.enable_local_distortion_solution_fields();
        self.enable_distortion_solution_fields();
    }

    /// Java private `enableDistortionSolutionFields()`.
    fn enable_distortion_solution_fields(&self) {
        let x_stretch_state = self.rb_distortion_full_solution.is_selected();
        self.ltf_xstretch_group_size.set_enabled(x_stretch_state);
        self.ltf_xstretch_non_default_groups
            .set_enabled(x_stretch_state);
        let skew_state =
            self.rb_distortion_full_solution.is_selected() || self.rb_distortion_skew.is_selected();
        self.ltf_skew_group_size.set_enabled(skew_state);
        self.ltf_skew_non_default_groups.set_enabled(skew_state);
    }

    /// Java private `enableLocalRotationSolutionFields()`.
    fn enable_local_rotation_solution_fields(&self) {
        let state = self.cb_local_rotation.is_selected();
        self.ltf_local_rotation_group_size.set_enabled(state);
        self.ltf_local_rotation_non_default_groups
            .set_enabled(state);
    }

    /// Java private `enableLocalTiltAngleSolutionFields()`.
    fn enable_local_tilt_angle_solution_fields(&self) {
        let state = self.cb_local_tilt_angle.is_selected();
        self.ltf_local_tilt_angle_group_size.set_enabled(state);
        self.ltf_local_tilt_angle_non_default_groups
            .set_enabled(state);
    }

    /// Java private `enableLocalMagnificationSolutionFields()`.
    fn enable_local_magnification_solution_fields(&self) {
        let state = self.cb_local_magnification.is_selected();
        self.ltf_local_magnification_group_size.set_enabled(state);
        self.ltf_local_magnification_non_default_groups
            .set_enabled(state);
    }

    /// Java private `enableLocalDistortionSolutionFields()`.
    fn enable_local_distortion_solution_fields(&self) {
        let x_stretch_state = self.rb_local_distortion_full_solution.is_selected();
        self.ltf_local_xstretch_group_size
            .set_enabled(x_stretch_state);
        self.ltf_local_xstretch_non_default_groups
            .set_enabled(x_stretch_state);
        let skew_state = self.rb_local_distortion_skew.is_selected()
            || self.rb_local_distortion_full_solution.is_selected();
        self.ltf_local_skew_group_size.set_enabled(skew_state);
        self.ltf_local_skew_non_default_groups
            .set_enabled(skew_state);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.tab_pane.get_component()
    }

    /// Java private `createRadioBox(JPanel, ButtonGroup, RadioButton[])`.
    fn create_radio_box_j_panel_button_group_radio_button_array(
        &self,
        panel: &Rc<JComponent>,
        group: Option<&Rc<ButtonGroup>>,
        items: &[Rc<RadioButton>],
    ) {
        self.create_radio_box_j_panel_button_group_radio_button_array_int(panel, group, items, 245);
    }

    /// Java private `createRadioBox(JPanel, ButtonGroup, RadioButton[], int)`.
    fn create_radio_box_j_panel_button_group_radio_button_array_int(
        &self,
        panel: &Rc<JComponent>,
        group: Option<&Rc<ButtonGroup>>,
        items: &[Rc<RadioButton>],
        width: i32,
    ) {
        // Swing layout: radioButtonHeight = 18 * font size adjustment;
        // radioButtonItemSize = (width * font size adjustment, height).
        let _ = width;

        // Add the items to the group and to the panel
        for item in items {
            if let Some(group) = group {
                group.add(&item.get_abstract_button());
            }
            panel.add(&item.get_abstract_button());
            // Swing layout: items[i].setPreferredSize(radioButtonItemSize).
        }
    }

    /// Java private `createGeneralTab()`: layout the general parameters tab.
    fn create_general_tab(&self) {
        let pnl_robust_fitting = JComponent::new_panel();
        let pnl_restrictalign = JComponent::new_panel();
        let pnl_measurement_ratio = JComponent::new_panel();
        let pnl_restrictalign_button = JComponent::new_panel();
        let pnl_local_align_validation = JComponent::new_panel();
        let pnl_cross_validate = JComponent::new_panel();
        // Swing layout: pnlGeneral, pnlGeneralBody BoxLayout Y_AXIS; rigid area x0_y5.

        self.pnl_general_body
            .add(&self.ltf_exclude_list.get_container());
        // Swing layout: rigid area x0_y5.

        self.pnl_general_body
            .add(&self.ltf_separate_view_groups.get_container());
        // Swing layout: rigid area x0_y10.

        // Residual reporting
        // Swing layout: pnlResidualThreshold BoxLayout Y_AXIS.
        self.pnl_residual_threshold
            .set_border(&EtchedBorder::new(Some("Residual Reporting")).get_border());
        // top panel
        let top_residual_panel = SpacedPanel::get_instance_void();
        // Swing layout: topResidualPanel.setBoxLayout(X_AXIS);
        // ltfResidualThreshold.setColumns(10).
        top_residual_panel.add_labeled_text_field(&self.ltf_residual_threshold);
        top_residual_panel.add_j_label(&JComponent::new_label("s.d."));
        self.pnl_residual_threshold
            .get_component()
            .add(&top_residual_panel.get_container());
        // bottom panel
        let bottom_residual_panel = SpacedPanel::get_instance_void();
        // Swing layout: bottomResidualPanel.setBoxLayout(X_AXIS);
        // setComponentAlignmentX(RIGHT_ALIGNMENT).
        bottom_residual_panel.add_j_label(&JComponent::new_label("Relative to"));
        // create radio button group
        let items = [
            self.rb_resid_all_views.clone(),
            self.rb_resid_neighboring.clone(),
        ];
        let pnl_rb_residual = JComponent::new_panel();
        // Swing layout: pnlRBResidual BoxLayout X_AXIS.
        self.create_radio_box_j_panel_button_group_radio_button_array_int(
            &pnl_rb_residual,
            Some(&self.bg_residual_threshold),
            &items,
            300,
        );
        bottom_residual_panel.add_j_panel(&pnl_rb_residual);
        // Swing layout: horizontal glue.

        // CrossValidate
        // Swing layout: pnlCrossValidate BoxLayout X_AXIS.
        pnl_cross_validate.add(&self.cb_cross_validate.get_component());
        // Swing layout: horizontal glue.

        // ResidualThreshold
        // Swing layout: rigid area x0_y5.
        self.pnl_residual_threshold
            .get_component()
            .add(&bottom_residual_panel.get_container());
        self.pnl_residual_threshold
            .get_component()
            .add(&pnl_cross_validate);

        self.pnl_general_body
            .add(&self.pnl_residual_threshold.get_component());
        // Swing layout: rigid area x0_y10.

        // Swing layout: pnlFiducialSurfaces BoxLayout X_AXIS.
        self.pnl_fiducial_surfaces
            .set_border(&EtchedBorder::new(Some("Analysis of Surface Angles")).get_border());

        // Need an extra panel to make border extend the appropriate width
        let pnl_rb_fiducual = JComponent::new_panel();
        // Swing layout: pnlRBFiducual BoxLayout Y_AXIS.
        let items = [
            self.rb_single_fiducial_surface.clone(),
            self.rb_dual_fiducial_surfaces.clone(),
        ];
        self.create_radio_box_j_panel_button_group_radio_button_array_int(
            &pnl_rb_fiducual,
            Some(&self.bg_fiducial_surfaces),
            &items,
            400,
        );

        self.pnl_fiducial_surfaces
            .get_component()
            .add(&pnl_rb_fiducual);
        // Swing layout: horizontal glue.
        self.pnl_general_body
            .add(&self.pnl_fiducial_surfaces.get_component());
        // Swing layout: rigid area x0_y10.

        // Swing layout: pnlVolumeParameters BoxLayout Y_AXIS.
        self.pnl_volume_parameters
            .set_border(&EtchedBorder::new(Some("Volume Position Parameters")).get_border());
        self.pnl_volume_parameters
            .get_component()
            .add(&self.ltf_tilt_angle_offset.get_container());
        // Swing layout: rigid area x0_y5.
        self.pnl_volume_parameters
            .get_component()
            .add(&self.ltf_tilt_axis_z_shift.get_container());
        // Swing layout: rigid area x0_y5.
        self.pnl_general_body
            .add(&self.pnl_volume_parameters.get_component());
        // Swing layout: rigid area x0_y10.

        // Swing layout: pnlMinimizationParams BoxLayout Y_AXIS.
        self.pnl_minimization_params
            .set_border(&EtchedBorder::new(Some("Minimization Parameters")).get_border());
        self.pnl_minimization_params
            .get_component()
            .add(&pnl_robust_fitting);
        // Swing layout: rigid area x0_y3.
        self.pnl_minimization_params
            .get_component()
            .add(&self.pnl_metro_factor);

        // RobustFitting
        // Swing layout: pnlRobustFitting BoxLayout X_AXIS.
        pnl_robust_fitting.add(
            &self
                .ctf_robust_fitting_and_k_factor_scaling
                .get_root_component(),
        );
        pnl_robust_fitting.add(&self.cb_weight_whole_tracks.get_component());
        // pnlWeightWholeTracks.add(Box.createHorizontalGlue());

        // Swing layout: pnlMetroFactor BoxLayout X_AXIS.
        self.pnl_metro_factor
            .add(&self.ltf_metro_factor.get_container());
        // Swing layout: rigid area x3_y0.
        self.pnl_metro_factor
            .add(&self.ltf_maximum_cycles.get_container());

        self.pnl_general_body
            .add(&self.pnl_minimization_params.get_component());
        // Swing layout: rigid area x0_y10.

        // local alignment
        // Swing layout: ltfMinLocalFiducials.setTextPreferredWidth(60 * font
        // size adjustment); pnlLocalParameters BoxLayout Y_AXIS.
        self.pnl_local_parameters
            .set_border(&EtchedBorder::new(Some("Local Alignment Parameters")).get_border());
        let pnl_local_alignments = JComponent::new_panel();
        // Swing layout: pnlLocalAlignments BoxLayout X_AXIS, CENTER_ALIGNMENT.
        pnl_local_alignments.add(&self.cb_local_alignments.get_component());
        // Swing layout: horizontal glue; cbLocalAlignments CENTER_ALIGNMENT.
        self.pnl_local_parameters
            .get_component()
            .add(&pnl_local_alignments);
        // Swing layout: pnlLocalPatches.setBoxLayout(Y_AXIS).
        self.pnl_local_patches
            .set_border(&EtchedBorder::new(Some("Local Patch Layout:")).get_border());
        self.pnl_local_patches
            .add_container(&self.rtf_target_patch_size_xand_y.get_container());
        self.pnl_local_patches
            .add_container(&self.rtf_n_local_patches.get_container());
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            &self.pnl_local_parameters_body,
            &self.pnl_local_patches.get_container(),
            &self.ltf_min_local_patch_size,
            Some(&self.ltf_min_local_fiducials),
            None,
            None,
            Some(&self.cb_fix_xyz_coordinates),
            None,
            Some(FixedDim::x10_y0),
        );
        self.pnl_local_parameters
            .get_component()
            .add(&self.pnl_local_parameters_body.get_component());
        self.pnl_general_body
            .add(&self.pnl_local_parameters.get_component());

        // MeasurementRatio
        // Swing layout: pnlMeasurementRatio BoxLayout X_AXIS, CENTER_ALIGNMENT.
        pnl_measurement_ratio.add(&self.ltf_target_measurement_ratio.get_component());
        // Swing layout: rigid area x10_y0.
        pnl_measurement_ratio.add(&self.ltf_min_measurement_ratio.get_component());
        // Swing layout: horizontal glue.

        // LocalAlignValidation
        // Swing layout: pnlLocalAlignValidation BoxLayout X_AXIS, CENTER_ALIGNMENT.
        pnl_local_align_validation.add(&self.l_local_align_validation);
        pnl_local_align_validation.add(
            &self
                .rb_local_align_validation_area_requirements
                .get_component(),
        );
        pnl_local_align_validation.add(&self.rb_local_align_validation_variables.get_component());
        pnl_local_align_validation.add(&self.rb_local_align_validation_both.get_component());
        // Swing layout: horizontal glue.

        // RestrictalignButton
        // Swing layout: pnlRestrictalignButton BoxLayout X_AXIS, glue around.
        pnl_restrictalign_button.add(&SwingComponent::get_component(&*self.btn_restrictalign));

        // Restrictalign
        // Swing layout: pnlRestrictalign BoxLayout Y_AXIS; EtchedBorder.
        pnl_restrictalign.set_border_title(
            EtchedBorder::new(Some("Restrict Align Variables with Cross-Validation"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_restrictalign.add(&pnl_local_align_validation);
        // Swing layout: rigid area x0_y5.
        pnl_restrictalign.add(&pnl_measurement_ratio);
        // Swing layout: rigid area x0_y5.
        pnl_restrictalign.add(&pnl_restrictalign_button);
        // Swing layout: rigid area x0_y5.
        self.pnl_general_body.add(&pnl_restrictalign);
        //
        // Swing layout: vertical glue.
        self.pnl_general.get_component().add(&self.pnl_general_body);
        self.tab_pane
            .add_tab_string_component("General", &self.pnl_general.get_component());
    }

    /// Java private `createGlobalSolutionTab()`: layout the global estimate tab.
    fn create_global_solution_tab(&self) {
        // init
        // Swing layout: btnRestrictalign.setToPreferredSize();
        // ltfTargetMeasurementRatio.setPreferredWidth(45);
        // ltfMinMeasurementRatio.setPreferredWidth(45).
        self.ltf_target_measurement_ratio
            .set_directive_def(Some(DirectiveDef::TARGET_MEASUREMENT_RATIO));
        self.ltf_min_measurement_ratio
            .set_directive_def(Some(DirectiveDef::MIN_MEASUREMENT_RATIO));
        //
        // Swing layout: pnlGlobalVariable, pnlGlobalVariableBody BoxLayout Y_AXIS.

        // Layout the global rotation variable parameters
        let pnl_rb_rotation = JComponent::new_panel();
        // Swing layout: pnlRBRotation BoxLayout Y_AXIS.
        let items = [
            self.rb_rotation_none.clone(),
            self.rb_rotation_one.clone(),
            self.rb_rotation_automap.clone(),
            self.rb_rotation_all.clone(),
        ];
        self.create_radio_box_j_panel_button_group_radio_button_array(
            &pnl_rb_rotation,
            Some(&self.bg_rotation_solution),
            &items,
        );
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            &self.pnl_rotation_solution,
            &pnl_rb_rotation,
            &self.ltf_rotation_angle,
            Some(&self.ltf_rotation_group_size),
            Some(&self.ltf_rotation_non_default_groups),
            None,
            None,
            Some("Rotation Solution Type"),
            None,
        );

        // Layout the global tilt angle estimate pane
        let pnl_rb_tilt_angle = JComponent::new_panel();
        // Swing layout: pnlRBTiltAngle BoxLayout Y_AXIS.
        let items = [
            self.rb_tilt_angle_fixed.clone(),
            self.rb_tilt_angle_automap.clone(),
            self.rb_tilt_angle_all.clone(),
        ];
        self.create_radio_box_j_panel_button_group_radio_button_array(
            &pnl_rb_tilt_angle,
            Some(&self.bg_tilt_angle_solution),
            &items,
        );
        self.create_variable_panel_etomo_panel_j_panel_labeled_text_field_labeled_text_field_string(
            &self.pnl_tilt_angle_solution,
            &pnl_rb_tilt_angle,
            &self.ltf_tilt_angle_group_size,
            Some(&self.ltf_tilt_angle_non_default_groups),
            Some("Tilt Angle Solution Type"),
        );

        // Layout the global magnification variable parameters
        let pnl_rb_magnification = JComponent::new_panel();
        // Swing layout: pnlRBMagnification BoxLayout Y_AXIS.
        let items = [
            self.rb_magnification_fixed.clone(),
            self.rb_magnification_automap.clone(),
            self.rb_magnification_all.clone(),
        ];
        self.create_radio_box_j_panel_button_group_radio_button_array(
            &pnl_rb_magnification,
            Some(&self.bg_magnification_solution),
            &items,
        );
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            &self.pnl_magnification_solution,
            &pnl_rb_magnification,
            &self.ltf_magnification_reference_view,
            Some(&self.ltf_magnification_group_size),
            Some(&self.ltf_magnification_non_default_groups),
            None,
            None,
            Some("Magnification Solution Type"),
            None,
        );

        // Layout the global distortion pane

        // Create radio box
        let items = [
            self.rb_distortion_disabled.clone(),
            self.rb_distortion_full_solution.clone(),
            self.rb_distortion_skew.clone(),
        ];
        let pnl_rb_distortion = JComponent::new_panel();
        // Swing layout: pnlRBDistortion BoxLayout Y_AXIS.
        self.create_radio_box_j_panel_button_group_radio_button_array(
            &pnl_rb_distortion,
            Some(&self.bg_distortion_solution),
            &items,
        );
        // Swing layout: ltfXstretchNonDefaultGroups.setTextPreferredWidth(
        // integer triplet width); ltfXstretchGroupSize.setTextPreferredWidth(
        // four digit width).
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            &self.pnl_distortion_solution,
            &pnl_rb_distortion,
            &self.ltf_xstretch_group_size,
            Some(&self.ltf_xstretch_non_default_groups),
            Some(&self.ltf_skew_group_size),
            Some(&self.ltf_skew_non_default_groups),
            None,
            Some("Distortion Solution Type"),
            None,
        );

        // Add the individual panes to the tab
        // Swing layout: rigid area x0_y5.
        self.pnl_global_variable_body
            .add(&self.pnl_rotation_solution.get_component());
        // Swing layout: rigid area x0_y5.
        self.pnl_global_variable_body
            .add(&self.pnl_magnification_solution.get_component());
        // Swing layout: rigid area x0_y5; vertical glue.
        self.pnl_global_variable_body
            .add(&self.pnl_tilt_angle_solution.get_component());
        // Swing layout: rigid area x0_y5; vertical glue.
        self.pnl_global_variable_body
            .add(&self.pnl_distortion_solution.get_component());
        // Swing layout: rigid area x0_y5; vertical glue.
        self.pnl_global_variable_body
            .add(&self.pnl_single_variables.get_component());
        // Swing layout: rigid area x0_y10; vertical glue.
        // single variables
        // Swing layout: pnlSingleVariables BoxLayout Y_AXIS, etched border,
        // CENTER_ALIGNMENT.
        self.pnl_single_variables.add(&self.ph_single_variables);
        // single variables body
        let opnl_single_variables_body = JComponent::new_panel();
        // Swing layout: opnlSingleVariablesBody BoxLayout X_AXIS, CENTER_ALIGNMENT.
        opnl_single_variables_body.add(&self.pnl_single_variables_body);
        // Swing layout: horizontal glue; pnlSingleVariablesBody BoxLayout Y_AXIS.
        // no beam tilt
        let pnl_no_beam_tilt = JComponent::new_panel();
        // Swing layout: pnlNoBeamTilt BoxLayout X_AXIS.
        pnl_no_beam_tilt.add(&self.rb_no_beam_tilt.get_component());
        // Swing layout: horizontal glue.
        self.bg_beam_tilt_option
            .add(&self.rb_no_beam_tilt.get_abstract_button());
        // fixed beam tilt
        let pnl_fixed_beam_tilt = JComponent::new_panel();
        // Swing layout: pnlFixedBeamTilt BoxLayout X_AXIS.
        pnl_fixed_beam_tilt.add(&self.rtf_fixed_beam_tilt.get_container());
        // Swing layout: horizontal glue.
        // solve for beam tilt
        let pnl_solve_for_beam_tilt = JComponent::new_panel();
        // Swing layout: pnlSolveForBeamTilt BoxLayout X_AXIS.
        pnl_solve_for_beam_tilt.add(&self.rb_solve_for_beam_tilt.get_component());
        // Swing layout: horizontal glue.
        self.bg_beam_tilt_option
            .add(&self.rb_solve_for_beam_tilt.get_abstract_button());

        self.pnl_single_variables
            .get_component()
            .add(&opnl_single_variables_body);
        let pnl_beam_tilt_radio_buttons = JComponent::new_panel();
        // Swing layout: pnlBeamTiltRadioButtons BoxLayout Y_AXIS, etched border,
        // LEFT_ALIGNMENT.
        pnl_beam_tilt_radio_buttons.add(&pnl_no_beam_tilt);
        pnl_beam_tilt_radio_buttons.add(&pnl_fixed_beam_tilt);
        pnl_beam_tilt_radio_buttons.add(&pnl_solve_for_beam_tilt);
        self.pnl_single_variables_body
            .add(&pnl_beam_tilt_radio_buttons);
        self.pnl_single_variables_body
            .add(&pnl_beam_tilt_radio_buttons);
        // solve for x tilt checkbox
        // Swing layout: cbXTiltAutomapSame RIGHT_ALIGNMENT.
        let pnl_x_tilt_automap_same = JComponent::new_panel();
        // Swing layout: pnlXTiltAutomapSame BoxLayout X_AXIS.
        pnl_x_tilt_automap_same.add(&self.cb_x_tilt_automap_same.get_component());
        // Swing layout: horizontal glue.
        self.pnl_single_variables_body.add(&pnl_x_tilt_automap_same);
        // projection stretch checkbox
        let pnl_projection_stretch = JComponent::new_panel();
        // Swing layout: pnlProjectionStretch BoxLayout X_AXIS.
        pnl_projection_stretch.add(&self.cb_projection_stretch.get_component());
        // Swing layout: horizontal glue.
        self.pnl_single_variables_body.add(&pnl_projection_stretch);
        // Swing layout: UIUtilities.alignComponentsX(pnlSingleVariablesBody,
        // LEFT_ALIGNMENT).
        self.pnl_global_variable
            .get_component()
            .add(&self.pnl_global_variable_body);
        self.tab_pane.add_tab_string_component(
            "Global Variables",
            &self.pnl_global_variable.get_component(),
        );
        self.pnl_global_variable_body.set_visible(false);
    }

    /// Java private `createLocalSolutionTab()`.
    fn create_local_solution_tab(&self) {
        // Construct the local solution panel
        // Swing layout: pnlLocalSolution, pnlLocalSolutionBody BoxLayout Y_AXIS.
        // pnlLocalSolution.setPreferredSize(new Dimension(400, 350));

        // Construct the rotation solution objects
        self.create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_string(
            &self.pnl_local_rotation_solution,
            &self.cb_local_rotation,
            &self.ltf_local_rotation_group_size,
            Some(&self.ltf_local_rotation_non_default_groups),
            Some("Local Rotation Solution Type"),
        );

        // Construct the tilt angle solution objects
        self.create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_string(
            &self.pnl_local_tilt_angle_solution,
            &self.cb_local_tilt_angle,
            &self.ltf_local_tilt_angle_group_size,
            Some(&self.ltf_local_tilt_angle_non_default_groups),
            Some("Local Tilt Angle Solution Type"),
        );

        // Construct the local magnification pane
        self.create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_string(
            &self.pnl_local_magnification_solution,
            &self.cb_local_magnification,
            &self.ltf_local_magnification_group_size,
            Some(&self.ltf_local_magnification_non_default_groups),
            Some("Local Magnification Solution Type"),
        );

        // Construction the local distortion pane

        // Create radio box
        let items = [
            self.rb_local_distortion_disabled.clone(),
            self.rb_local_distortion_full_solution.clone(),
            self.rb_local_distortion_skew.clone(),
        ];
        let pnl_rb_local_distortion = JComponent::new_panel();
        // Swing layout: pnlRBLocalDistortion BoxLayout Y_AXIS.
        self.create_radio_box_j_panel_button_group_radio_button_array(
            &pnl_rb_local_distortion,
            Some(&self.bg_local_distortion_solution),
            &items,
        );
        // Swing layout: ltfLocalXstretchNonDefaultGroups.setTextPreferredWidth(
        // integer triplet width); ltfLocalXstretchGroupSize
        // .setTextPreferredWidth(four digit width).
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            &self.pnl_local_distortion_solution,
            &pnl_rb_local_distortion,
            &self.ltf_local_xstretch_group_size,
            Some(&self.ltf_local_xstretch_non_default_groups),
            Some(&self.ltf_local_skew_group_size),
            Some(&self.ltf_local_skew_non_default_groups),
            None,
            Some("Local Distortion Solution Type"),
            None,
        );

        // Swing layout: vertical glue; rigid area x0_y10.
        self.pnl_local_solution_body
            .add(&self.pnl_local_rotation_solution.get_component());

        // Swing layout: vertical glue; rigid area x0_y10.
        self.pnl_local_solution_body
            .add(&self.pnl_local_magnification_solution.get_component());

        // Swing layout: vertical glue; rigid area x0_y10.
        self.pnl_local_solution_body
            .add(&self.pnl_local_tilt_angle_solution.get_component());

        // Swing layout: vertical glue; rigid area x0_y10.
        self.pnl_local_solution_body
            .add(&self.pnl_local_distortion_solution.get_component());
        self.pnl_local_solution
            .get_component()
            .add(&self.pnl_local_solution_body);
        self.tab_pane
            .add_tab_string_component("Local Variables", &self.pnl_local_solution.get_component());
        self.pnl_local_solution.get_component().set_visible(false);
    }

    /// Java private `createVariablePanel(EtomoPanel, CheckBox,
    /// LabeledTextField, LabeledTextField, String)`.
    fn create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_string(
        &self,
        panel: &Rc<EtomoPanel>,
        check_box: &Rc<CheckBox>,
        group_size: &Rc<LabeledTextField>,
        additional_groups: Option<&Rc<LabeledTextField>>,
        title: Option<&str>,
    ) {
        self.create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_labeled_text_field_string(
            panel,
            check_box,
            group_size,
            additional_groups,
            None,
            title,
        );
    }

    /// Java private `createVariablePanel(EtomoPanel, CheckBox,
    /// LabeledTextField, LabeledTextField, LabeledTextField, String)`.
    fn create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_labeled_text_field_string(
        &self,
        panel: &Rc<EtomoPanel>,
        check_box: &Rc<CheckBox>,
        field1: &Rc<LabeledTextField>,
        field2: Option<&Rc<LabeledTextField>>,
        field3: Option<&Rc<LabeledTextField>>,
        title: Option<&str>,
    ) {
        let button_panel = JComponent::new_panel();
        // Swing layout: buttonPanel BoxLayout Y_AXIS.
        button_panel.add(&check_box.get_component());
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            panel,
            &button_panel,
            field1,
            field2,
            field3,
            None,
            None,
            title,
            None,
        );
    }

    /// Java private `createVariablePanel(EtomoPanel, CheckBox,
    /// LabeledTextField, LabeledTextField, LabeledTextField, CheckBox,
    /// String)` (not called in the Java).
    #[allow(dead_code)]
    fn create_variable_panel_etomo_panel_check_box_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string(
        &self,
        panel: &Rc<EtomoPanel>,
        check_box1: &Rc<CheckBox>,
        field1: &Rc<LabeledTextField>,
        field2: Option<&Rc<LabeledTextField>>,
        field3: Option<&Rc<LabeledTextField>>,
        check_box2: Option<&Rc<CheckBox>>,
        title: Option<&str>,
    ) {
        let button_panel = JComponent::new_panel();
        // Swing layout: buttonPanel BoxLayout Y_AXIS.
        button_panel.add(&check_box1.get_component());
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            panel,
            &button_panel,
            field1,
            field2,
            field3,
            None,
            check_box2,
            title,
            None,
        );
    }

    /// Java private `createVariablePanel(EtomoPanel, JPanel, LabeledTextField,
    /// LabeledTextField, String)`: create a variable panel with an internal
    /// panel (can contain a radio button group).
    fn create_variable_panel_etomo_panel_j_panel_labeled_text_field_labeled_text_field_string(
        &self,
        panel: &Rc<EtomoPanel>,
        button_panel: &Rc<JComponent>,
        group_size: &Rc<LabeledTextField>,
        additional_groups: Option<&Rc<LabeledTextField>>,
        title: Option<&str>,
    ) {
        self.create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
            panel,
            button_panel,
            group_size,
            additional_groups,
            None,
            None,
            None,
            title,
            None,
        );
    }

    /// Java private `createVariablePanel(EtomoPanel, Container,
    /// LabeledTextField, LabeledTextField, LabeledTextField, LabeledTextField,
    /// CheckBox, String, Dimension)`.
    #[allow(clippy::too_many_arguments)]
    fn create_variable_panel_etomo_panel_container_labeled_text_field_labeled_text_field_labeled_text_field_labeled_text_field_check_box_string_dimension(
        &self,
        panel: &Rc<EtomoPanel>,
        button_panel: &Rc<JComponent>,
        field1: &Rc<LabeledTextField>,
        field2: Option<&Rc<LabeledTextField>>,
        field3: Option<&Rc<LabeledTextField>>,
        field4: Option<&Rc<LabeledTextField>>,
        check_box: Option<&Rc<CheckBox>>,
        title: Option<&str>,
        mut spacing: Option<Dimension>,
    ) {
        if spacing.is_none() {
            spacing = Some(FixedDim::x40_y0);
        }
        // Swing layout: panel BoxLayout X_AXIS.
        // panel.add(Box.createRigidArea(FixedDim.x5_y0));
        panel.get_component().add(button_panel);
        // Swing layout: rigid area of `spacing`.
        let _ = spacing;
        let field_panel = SpacedPanel::get_instance_void();
        // Swing layout: fieldPanel.setBoxLayout(Y_AXIS).
        field_panel.add_labeled_text_field(field1);
        if let Some(field2) = field2 {
            field_panel.add_labeled_text_field(field2);
        }
        if let Some(field3) = field3 {
            field_panel.add_labeled_text_field(field3);
        }
        if let Some(field4) = field4 {
            field_panel.add_labeled_text_field(field4);
        }
        if let Some(check_box) = check_box {
            let pnl_check_box = JComponent::new_panel();
            // Swing layout: pnlCheckBox BoxLayout X_AXIS, CENTER_ALIGNMENT.
            pnl_check_box.add(&check_box.get_component());
            // Swing layout: horizontal glue; checkBox RIGHT_ALIGNMENT.
            field_panel.add_j_panel(&pnl_check_box);
        }
        panel.get_component().add(&field_panel.get_container());
        if let Some(title) = title {
            panel.set_border(&EtchedBorder::new(Some(title)).get_border());
        }
    }

    /// Java private `setToolTipText()`: initialize the tooltip text for the
    /// axis panel objects.
    fn set_tool_tip_text(&self) {
        let mut section: *mut Section;
        let mut autodoc: Option<*mut Autodoc> = None;
        // SAFETY (all `unsafe` below): `AutodocFactory` owns every autodoc it
        // returns, and each autodoc owns its sections, for the life of the
        // process (the Java GC-owned singletons), so the pointers stay valid
        // for this method.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.app_mgr),
                Some(autodoc_factory::TILTALIGN),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = Some(instance),
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except)
            Err(except) => eprintln!("{except}"),
        }
        // Upstream bug fixed (TiltalignPanel.java:1942, 2133): Java calls
        // `autodoc.getAutodocName()` and `autodoc.getSection(...)` without the
        // null check it makes elsewhere, so a missing or locked tiltalign (or
        // restrictalign) autodoc throws a NullPointerException out of the
        // constructor and the Fine Alignment dialog cannot be built.  We
        // treat a null autodoc as having no name and no sections: every
        // autodoc tooltip is then null (as `EtomoAutodoc.getTooltip` returns
        // for a null autodoc or section) and the fixed tooltips are still set.
        let autodoc_ref: Option<&dyn ReadOnlyAutodoc> =
            autodoc.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        let autodoc_name: Option<String> = autodoc_ref.map(|autodoc| autodoc.get_autodoc_name());
        let autodoc_name = autodoc_name.as_deref();
        // General tab
        self.ltf_exclude_list.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc_ref, Some(const_tiltalign_param::EXCLUDE_LIST_KEY))
                .as_deref(),
        );
        self.ltf_separate_view_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::SEPARATE_GROUP_KEY),
            )
            .as_deref(),
        );
        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_tiltalign_param::RESIDUAL_REPORT_CRITERION_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        if !section.is_null() {
            let section: &dyn ReadOnlySection = unsafe { &*section };
            self.ltf_residual_threshold.set_tool_tip_text(
                etomo_autodoc::get_tooltip_add_source(autodoc_name, section, true).as_deref(),
            );
            self.rb_resid_all_views.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_enum_value_name(autodoc_name, section, Some("all"))
                    .as_deref(),
            );
            self.rb_resid_neighboring.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_enum_value_name(
                    autodoc_name,
                    section,
                    Some("neighboring"),
                )
                .as_deref(),
            );
            self.cb_cross_validate.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_enum_value_name(
                    autodoc_name,
                    section,
                    Some(const_tiltalign_param::CROSS_VALIDATE_KEY),
                )
                .as_deref(),
            );
        }
        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_tiltalign_param::SURFACES_TO_ANALYZE_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        if !section.is_null() {
            let section: &dyn ReadOnlySection = unsafe { &*section };
            // Java string concatenation: a null tooltip prints "null".
            self.rb_single_fiducial_surface
                .set_tool_tip_text_string(Some(&format!(
                    "{}{}",
                    etomo_autodoc::get_tooltip_enum_value_name(autodoc_name, section, Some("1"))
                        .unwrap_or_else(|| "null".to_string()),
                    "  Use if fiducials are on one surface or distributed in Z."
                )));
            self.rb_dual_fiducial_surfaces.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_enum_value_name(autodoc_name, section, Some("2"))
                    .as_deref(),
            );
        }
        self.ltf_tilt_angle_offset.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc_ref, Some(const_tiltalign_param::ANGLE_OFFSET_KEY))
                .as_deref(),
        );
        self.ltf_tilt_axis_z_shift.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc_ref, Some(const_tiltalign_param::AXIS_Z_SHIFT_KEY))
                .as_deref(),
        );
        self.ltf_metro_factor.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc_ref, Some(const_tiltalign_param::METRO_FACTOR_KEY))
                .as_deref(),
        );
        self.ltf_maximum_cycles.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::MAXIMUM_CYCLES_KEY),
            )
            .as_deref(),
        );
        self.cb_local_alignments.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_ALIGNMENTS_KEY),
            )
            .as_deref(),
        );
        self.rtf_target_patch_size_xand_y.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::TARGET_PATCH_SIZE_X_AND_Y_KEY),
            )
            .as_deref(),
        );

        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_tiltalign_param::NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        // Java passes a possibly null section here; `getTooltip(String,
        // ReadOnlySection, String)` catches the resulting NullPointerException
        // and returns `getTooltip(autodocName, null, true)`, which is null.
        if self.created_day_stamp_imod_5_0_1.get() {
            self.rtf_n_local_patches.set_tool_tip_text(
                if section.is_null() {
                    None
                } else {
                    etomo_autodoc::get_tooltip_enum_value_name(
                        autodoc_name,
                        unsafe { &*section },
                        Some("new"),
                    )
                }
                .as_deref(),
            );
        } else {
            self.rtf_n_local_patches.set_tool_tip_text(
                if section.is_null() {
                    None
                } else {
                    etomo_autodoc::get_tooltip_enum_value_name(
                        autodoc_name,
                        unsafe { &*section },
                        Some("old"),
                    )
                }
                .as_deref(),
            );
        }
        self.ltf_min_local_patch_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::MIN_SIZE_OR_OVERLAP_X_AND_Y_KEY),
            )
            .as_deref(),
        );
        self.ltf_min_local_fiducials.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::MIN_FIDS_TOTAL_AND_EACH_SURFACE_KEY),
            )
            .as_deref(),
        );
        self.cb_fix_xyz_coordinates.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::FIX_XYZ_COORDINATES_KEY),
            )
            .as_deref(),
        );
        self.ctf_robust_fitting_and_k_factor_scaling
            .set_check_box_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc_ref,
                    Some(const_tiltalign_param::ROBUST_FITTING_KEY),
                )
                .as_deref(),
            );
        self.ctf_robust_fitting_and_k_factor_scaling
            .set_field_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc_ref,
                    Some(const_tiltalign_param::K_FACTOR_SCALING_KEY),
                )
                .as_deref(),
            );
        self.cb_weight_whole_tracks.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::WEIGHT_WHOLE_TRACKS_KEY),
            )
            .as_deref(),
        );
        // Global variables
        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_tiltalign_param::TILT_OPTION_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        if !section.is_null() {
            let section: &dyn ReadOnlySection = unsafe { &*section };
            self.rb_tilt_angle_fixed.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::FIXED_OPTION,
                )
                .as_deref(),
            );
            self.rb_tilt_angle_all.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::TILT_ALL_OPTION,
                )
                .as_deref(),
            );
            self.rb_tilt_angle_automap.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::TILT_AUTOMAPPED_OPTION,
                )
                .as_deref(),
            );
        }
        self.ltf_tilt_angle_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::TILT_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_tilt_angle_non_default_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::TILT_NONDEFAULT_GROUP_KEY),
            )
            .as_deref(),
        );

        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_tiltalign_param::MAG_OPTION_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        if !section.is_null() {
            let section: &dyn ReadOnlySection = unsafe { &*section };
            self.rb_magnification_fixed.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::FIXED_OPTION,
                )
                .as_deref(),
            );
            self.rb_magnification_all.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::ALL_OPTION,
                )
                .as_deref(),
            );
            self.rb_magnification_automap.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::AUTOMAPPED_OPTION,
                )
                .as_deref(),
            );
        }
        self.ltf_magnification_reference_view.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::MAG_REFERENCE_VIEW_KEY),
            )
            .as_deref(),
        );
        self.ltf_magnification_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::MAG_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_magnification_non_default_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::MAG_NONDEFAULT_GROUP_KEY),
            )
            .as_deref(),
        );

        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(const_tiltalign_param::ROT_OPTION_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        if !section.is_null() {
            let section: &dyn ReadOnlySection = unsafe { &*section };
            self.rb_rotation_none.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::NONE_OPTION,
                )
                .as_deref(),
            );
            self.rb_rotation_all.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::ALL_OPTION,
                )
                .as_deref(),
            );
            self.rb_rotation_automap.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::AUTOMAPPED_OPTION,
                )
                .as_deref(),
            );
            self.rb_rotation_one.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip_int(
                    autodoc_name,
                    section,
                    const_tiltalign_param::SINGLE_OPTION,
                )
                .as_deref(),
            );
        }
        self.ltf_rotation_angle.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc_ref, Some(const_tiltalign_param::ROT_ANGLE_KEY))
                .as_deref(),
        );
        self.ltf_rotation_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::ROT_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_rotation_non_default_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::ROT_NONDEFAULT_GROUP_KEY),
            )
            .as_deref(),
        );
        self.rb_distortion_disabled.set_tool_tip_text_string(Some(
            "Do not solve for distortions in the plane of section.",
        ));
        self.rb_distortion_full_solution
            .set_tool_tip_text_string(Some(
                "Solve for X-stretch and skew in the plane of section.",
            ));
        self.rb_distortion_skew.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(autodoc_ref, Some(const_tiltalign_param::SKEW_OPTION_KEY))
                .as_deref(),
        );
        self.ltf_xstretch_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::X_STRETCH_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_xstretch_non_default_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::X_STRETCH_NONDEFAULT_GROUP_KEY),
            )
            .as_deref(),
        );
        self.ltf_skew_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::SKEW_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_skew_non_default_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::SKEW_NONDEFAULT_GROUP_KEY),
            )
            .as_deref(),
        );
        self.cb_projection_stretch.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::PROJECTION_STRETCH_KEY),
            )
            .as_deref(),
        );
        // local variables
        self.cb_local_rotation.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_ROT_OPTION_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_rotation_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_ROT_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_rotation_non_default_groups
            .set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc_ref,
                    Some(const_tiltalign_param::LOCAL_ROT_NONDEFAULT_GROUP_KEY),
                )
                .as_deref(),
            );
        self.cb_local_tilt_angle.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_TILT_OPTION_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_tilt_angle_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_TILT_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_tilt_angle_non_default_groups
            .set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc_ref,
                    Some(const_tiltalign_param::LOCAL_TILT_NONDEFAULT_GROUP_KEY),
                )
                .as_deref(),
            );
        self.cb_local_magnification.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_MAG_OPTION_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_magnification_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_MAG_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_magnification_non_default_groups
            .set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc_ref,
                    Some(const_tiltalign_param::LOCAL_MAG_NONDEFAULT_GROUP_KEY),
                )
                .as_deref(),
            );
        self.rb_local_distortion_disabled
            .set_tool_tip_text_string(Some(
                "Do not solve for local distortions in the plane of section.",
            ));
        self.rb_local_distortion_full_solution
            .set_tool_tip_text_string(Some(
                "Solve for local X-stretch and skew in the plane of section.",
            ));
        self.rb_local_distortion_skew.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_SKEW_OPTION_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_xstretch_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_X_STRETCH_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_xstretch_non_default_groups
            .set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    autodoc_ref,
                    Some(const_tiltalign_param::LOCAL_X_STRETCH_NONDEFAULT_GROUP_KEY),
                )
                .as_deref(),
            );
        self.ltf_local_skew_group_size.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_SKEW_DEFAULT_GROUPING_KEY),
            )
            .as_deref(),
        );
        self.ltf_local_skew_non_default_groups.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::LOCAL_SKEW_NONDEFAULT_GROUP_KEY),
            )
            .as_deref(),
        );
        self.rb_no_beam_tilt
            .set_tool_tip_text_string(Some("The beam axis is perpendicular to tilt axis"));
        self.rtf_fixed_beam_tilt
            .set_radio_button_tool_tip_text(Some(
                "Set the non-perpendicularity between tilt axis and beam axis.",
            ));
        self.rtf_fixed_beam_tilt.set_text_field_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(const_tiltalign_param::FIXED_OR_INITIAL_BEAM_TILT),
            )
            .as_deref(),
        );
        self.rb_solve_for_beam_tilt
            .set_tool_tip_text_string(Some(&format!(
                "{}{}",
                "Perform the minimization at a series of fixed beam tilt values and",
                " search for the value that gives the smallest error."
            )));
        self.cb_x_tilt_automap_same
            .set_tool_tip_text_string(Some(&format!(
                "{}{}",
                "Solve for an X-axis tilt for each separate view group (XTiltOption",
                " 4 with very large XTiltDefaultGrouping)"
            )));
        self.btn_restrictalign.set_tool_tip_text(Some(&format!(
            "{}{}{}",
            "Run restrictalign, which uses cross-validation to assess whether more restrictive ",
            "parameter settings give a more reliable solution with smaller leave-out ",
            "errors"
        )));
        // Restrictalign autodoc
        autodoc = None;
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.app_mgr),
                Some(autodoc_factory::RESTRICT_ALIGN),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = Some(instance),
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except)
            Err(except) => eprintln!("{except}"),
        }
        let autodoc_ref: Option<&dyn ReadOnlyAutodoc> =
            autodoc.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        self.ltf_target_measurement_ratio.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(restrictalign_param::TARGET_MEASUREMENT_RATIO_KEY),
            )
            .as_deref(),
        );
        self.ltf_min_measurement_ratio.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc_ref,
                Some(restrictalign_param::MIN_MEASUREMENT_RATIO_KEY),
            )
            .as_deref(),
        );
        section = match autodoc {
            Some(autodoc) => unsafe {
                (*autodoc).get_section(
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(restrictalign_param::LOCAL_ALIGN_VALIDATION_KEY),
                )
            },
            None => std::ptr::null_mut(),
        };
        if !section.is_null() {
            let section: &dyn ReadOnlySection = unsafe { &*section };
            // Note: the Java keeps using the tiltalign autodoc's name here.
            self.rb_local_align_validation_area_requirements
                .set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip_const_etomo_number(
                        autodoc_name,
                        section,
                        &AREA_REQUIREMENTS.get_value(),
                    )
                    .as_deref(),
                );
            self.rb_local_align_validation_variables
                .set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip_const_etomo_number(
                        autodoc_name,
                        section,
                        &VARIABLES.get_value(),
                    )
                    .as_deref(),
                );
            self.rb_local_align_validation_both
                .set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip_const_etomo_number(
                        autodoc_name,
                        section,
                        &BOTH.get_value(),
                    )
                    .as_deref(),
                );
        }
    }
}

/// Java `Expandable`.
impl Expandable for TiltalignPanel {
    /// Java `expand(GlobalExpandButton)`: the header only covers part of the
    /// advanced fields, so use the global advanced button directly to expand.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.app_mgr))
        });
    }

    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.ph_single_variables.equals_advanced_basic(button) {
            self.update_advanced_beam_tilt(button.is_expanded());
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.app_mgr))
        });
    }
}

/// Java `UIComponent`.
impl UIComponent for TiltalignPanel {
    /// Java `getUIComponent()`: `this`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        SwingComponent::get_component(self)
    }
}

/// Java `SwingComponent`.
impl SwingComponent for TiltalignPanel {
    /// Java `getComponent()`: `getContainer()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.get_container()
    }
}
