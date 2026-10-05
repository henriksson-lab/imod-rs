//! `IMOD/Etomo/src/etomo/comscript/ConstTiltalignParam.java`.
//!
//! A read only model of the parameter interface for the tiltalign program.
//!
//! **Representation.**  Java's `TiltalignParam extends ConstTiltalignParam`.  As
//! `etomo/type/script_parameter.rs` does for its superclass, the superclass is a
//! concrete struct here (`ConstTiltalignParam`, holding every field the Java class
//! declares), and `tiltalign_param.rs`'s `TiltalignParam` holds it in its `base` field
//! and reaches it through `Deref`/`DerefMut`.  The fields are `pub(crate)` because the
//! Java fields are package-private and the subclass assigns them directly.
//!
//! `ConstTiltalignParam implements CommandDetails`: that is `Command` +
//! `ProcessDetails` here, plus `Loggable` (the Java `ProcessDetails extends
//! Loggable`).  Java `String` fields hold `null` whenever the com script has the
//! keyword without a value (`ComScriptCommand.getValue`), so they are
//! `Option<String>`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::fortran_input_string::FortranInputString;
use super::param_utilities;
use super::process_details::ProcessDetails;
use super::string_list::StringList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::x_tilt_option::XTiltOption;

pub const SINGLE_OPTION: i32 = -1;
pub const FIXED_OPTION: i32 = 0;
pub const NONE_OPTION: i32 = FIXED_OPTION;
pub const ALL_OPTION: i32 = 1;
pub const TILT_ALL_OPTION: i32 = 2;
pub const BEAM_SEARCH_OPTION: i32 = 2;
pub const AUTOMAPPED_OPTION: i32 = 3;
pub const TILT_AUTOMAPPED_OPTION: i32 = 5;

pub const EXCLUDE_LIST_KEY: &str = "ExcludeList";
pub const SEPARATE_GROUP_KEY: &str = "SeparateGroup";
pub const RESIDUAL_REPORT_CRITERION_KEY: &str = "ResidualReportCriterion";
pub const SURFACES_TO_ANALYZE_KEY: &str = "SurfacesToAnalyze";
pub const ANGLE_OFFSET_KEY: &str = "AngleOffset";
pub const AXIS_Z_SHIFT_KEY: &str = "AxisZShift";
pub const METRO_FACTOR_KEY: &str = "MetroFactor";
pub const MAXIMUM_CYCLES_KEY: &str = "MaximumCycles";
pub const LOCAL_ALIGNMENTS_KEY: &str = "LocalAlignments";
pub const NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY: &str = "NumberOfLocalPatchesXandY";
pub const TARGET_PATCH_SIZE_X_AND_Y_KEY: &str = "TargetPatchSizeXandY";
pub const MIN_SIZE_OR_OVERLAP_X_AND_Y_KEY: &str = "MinSizeOrOverlapXandY";
pub const MIN_FIDS_TOTAL_AND_EACH_SURFACE_KEY: &str = "MinFidsTotalAndEachSurface";
pub const TILT_OPTION_KEY: &str = "TiltOption";
pub const TILT_DEFAULT_GROUPING_KEY: &str = "TiltDefaultGrouping";
pub const TILT_NONDEFAULT_GROUP_KEY: &str = "TiltNondefaultGroup";
pub const MAG_OPTION_KEY: &str = "MagOption";
pub const MAG_REFERENCE_VIEW_KEY: &str = "MagReferenceView";
pub const MAG_DEFAULT_GROUPING_KEY: &str = "MagDefaultGrouping";
pub const MAG_NONDEFAULT_GROUP_KEY: &str = "MagNondefaultGroup";
pub const ROT_OPTION_KEY: &str = "RotOption";
pub const ROT_ANGLE_KEY: &str = "RotationAngle";
pub const ROT_DEFAULT_GROUPING_KEY: &str = "RotDefaultGrouping";
pub const ROT_NONDEFAULT_GROUP_KEY: &str = "RotNondefaultGroup";
pub const SKEW_OPTION_KEY: &str = "SkewOption";
pub const X_STRETCH_DEFAULT_GROUPING_KEY: &str = "XStretchDefaultGrouping";
pub const X_STRETCH_NONDEFAULT_GROUP_KEY: &str = "XStretchNondefaultGroup";
pub const SKEW_DEFAULT_GROUPING_KEY: &str = "SkewDefaultGrouping";
pub const SKEW_NONDEFAULT_GROUP_KEY: &str = "SkewNondefaultGroup";
pub const LOCAL_ROT_OPTION_KEY: &str = "LocalRotOption";
pub const LOCAL_ROT_DEFAULT_GROUPING_KEY: &str = "LocalRotDefaultGrouping";
pub const LOCAL_ROT_NONDEFAULT_GROUP_KEY: &str = "LocalRotNondefaultGroup";
pub const LOCAL_TILT_OPTION_KEY: &str = "LocalTiltOption";
pub const LOCAL_TILT_DEFAULT_GROUPING_KEY: &str = "LocalTiltDefaultGrouping";
pub const LOCAL_TILT_NONDEFAULT_GROUP_KEY: &str = "LocalTiltNondefaultGroup";
pub const LOCAL_MAG_OPTION_KEY: &str = "LocalMagOption";
pub const LOCAL_MAG_DEFAULT_GROUPING_KEY: &str = "LocalMagDefaultGrouping";
pub const LOCAL_MAG_NONDEFAULT_GROUP_KEY: &str = "LocalMagNondefaultGroup";
pub const LOCAL_SKEW_OPTION_KEY: &str = "LocalSkewOption";
pub const LOCAL_X_STRETCH_DEFAULT_GROUPING_KEY: &str = "LocalXStretchDefaultGrouping";
pub const LOCAL_X_STRETCH_NONDEFAULT_GROUP_KEY: &str = "LocalXStretchNondefaultGroup";
pub const LOCAL_SKEW_DEFAULT_GROUPING_KEY: &str = "LocalSkewDefaultGrouping";
pub const LOCAL_SKEW_NONDEFAULT_GROUP_KEY: &str = "LocalSkewNondefaultGroup";
pub const PROJECTION_STRETCH_KEY: &str = "ProjectionStretch";
pub const FIX_XYZ_COORDINATES_KEY: &str = "FixXYZCoordinates";
pub(crate) const OUTPUT_X_AXIS_TILT_FILE_KEY: &str = "OutputXAxisTiltFile";
pub const BEAM_TILT_OPTION_KEY: &str = "BeamTiltOption";
pub const FIXED_OR_INITIAL_BEAM_TILT: &str = "FixedOrInitialBeamTilt";
pub const ROBUST_FITTING_KEY: &str = "RobustFitting";
pub const K_FACTOR_SCALING_KEY: &str = "KFactorScaling";
pub const WEIGHT_WHOLE_TRACKS_KEY: &str = "WeightWholeTracks";
pub const CROSS_VALIDATE_KEY: &str = "CrossValidate";
const CREATED_DAY_STAMP_IMOD_5_0_1: i32 = 1658;

pub(crate) const MODEL_FILE_STRING: &str = "ModelFile";
pub(crate) const IMAGE_FILE_STRING: &str = "ImageFile";
pub(crate) const OUTPUT_MODEL_FILE_STRING: &str = "OutputModelFile";
pub(crate) const OUTPUT_RESIDUAL_FILE_STRING: &str = "OutputResidualFile";
pub(crate) const OUTPUT_MODEL_AND_RESIDUAL_STRING: &str = "OutputModelAndResidual";
pub(crate) const OUTPUT_FID_XYZ_FILE_STRING: &str = "OutputFidXYZFile";
pub(crate) const OUTPUT_TILT_FILE_STRING: &str = "OutputTiltFile";
pub(crate) const OUTPUT_TRANSFORM_FILE_STRING: &str = "OutputTransformFile";
pub(crate) const OUTPUT_Z_FACTOR_FILE_STRING: &str = "OutputZFactorFile";
pub(crate) const INCLUDE_START_END_INC_STRING: &str = "IncludeStartEndInc";
pub(crate) const INCLUDE_LIST_STRING: &str = "IncludeList";
pub(crate) const OUTPUT_LOCAL_FILE_STRING: &str = "OutputLocalFile";
pub(crate) const LOCAL_OUTPUT_OPTIONS_STRING: &str = "LocalOutputOptions";

pub(crate) const MODEL_FILE_EXTENSION: &str = ".3dmod";
pub(crate) const RESIDUAL_FILE_EXTENSION: &str = ".resid";
pub(crate) const Z_FACTOR_FILE_EXTENSION: &str = ".zfac";
pub(crate) const LOCAL_FILE_EXTENSION: &str = "local.xf";
pub(crate) const NONDEFAULT_GROUP_INTEGER_TYPE: [bool; 3] = [true, true, true];
pub(crate) const NONDEFAULT_GROUP_SIZE: i32 = 3;

const OPTION_VALID_VALUES: [i32; 3] = [FIXED_OPTION, ALL_OPTION, AUTOMAPPED_OPTION];
const TILT_OPTION_VALID_VALUES: [i32; 3] = [FIXED_OPTION, TILT_ALL_OPTION, TILT_AUTOMAPPED_OPTION];
const DISTORTION_OPTION_VALID_VALUES: [i32; 2] = [FIXED_OPTION, AUTOMAPPED_OPTION];
const LOCAL_OPTION_VALID_VALUES: [i32; 2] = [FIXED_OPTION, AUTOMAPPED_OPTION];
const LOCAL_TILT_OPTION_VALID_VALUES: [i32; 2] = [FIXED_OPTION, TILT_AUTOMAPPED_OPTION];
const ROT_OPTION_VALID_VALUES: [i32; 4] =
    [FIXED_OPTION, ALL_OPTION, AUTOMAPPED_OPTION, SINGLE_OPTION];
const SURFACES_TO_ANALYZE_VALID_VALUES: [i32; 3] = [0, 1, 2];
const PROCESS_NAME: ProcessName = ProcessName::ALIGN;
const COMMAND_FILE_EXTENSION: &str = ".com";
pub const TARGET_PATCH_SIZE_X_AND_Y_DEFAULT: &str = "700,700";
pub const NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT: &str = "5,5";

/// Java `ConstTiltalignParam`.
pub struct ConstTiltalignParam {
    pub(crate) model_file: Option<String>,
    pub(crate) image_file: Option<String>,
    pub(crate) output_model_and_residual: Option<String>,
    pub(crate) output_model_file: Option<String>,
    pub(crate) output_residual_file: Option<String>,
    pub(crate) output_fid_xyz_file: Option<String>,
    pub(crate) output_tilt_file: Option<String>,
    pub(crate) output_transform_file: Option<String>,
    pub(crate) output_z_factor_file: Option<String>,
    pub(crate) include_start_end_inc: FortranInputString,
    pub(crate) include_list: StringList,
    pub(crate) exclude_list: StringList,
    pub(crate) rotation_angle: ScriptParameter,
    pub(crate) separate_group: StringList,
    pub(crate) tilt_angle_spec: TiltAngleSpec,
    pub(crate) angle_offset: ScriptParameter,

    // Updates done
    pub(crate) projection_stretch: EtomoBoolean2,
    pub(crate) rot_option: ScriptParameter,
    pub(crate) rot_default_grouping: ScriptParameter,
    pub(crate) rot_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) rotation_fixed_view: ScriptParameter,
    pub(crate) local_rot_option: ScriptParameter,
    pub(crate) local_rot_default_grouping: ScriptParameter,
    pub(crate) local_rot_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) tilt_option: ScriptParameter,
    pub(crate) tilt_default_grouping: ScriptParameter,
    pub(crate) tilt_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) local_tilt_option: ScriptParameter,
    pub(crate) local_tilt_default_grouping: ScriptParameter,
    pub(crate) local_tilt_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) mag_reference_view: ScriptParameter,
    pub(crate) mag_option: ScriptParameter,
    pub(crate) mag_default_grouping: ScriptParameter,
    pub(crate) mag_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) local_mag_reference_view: ScriptParameter,
    pub(crate) local_mag_option: ScriptParameter,
    pub(crate) local_mag_default_grouping: ScriptParameter,
    pub(crate) local_mag_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) x_stretch_option: ScriptParameter,
    pub(crate) x_stretch_default_grouping: ScriptParameter,
    pub(crate) x_stretch_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) local_x_stretch_option: ScriptParameter,
    pub(crate) local_x_stretch_default_grouping: ScriptParameter,
    pub(crate) local_x_stretch_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) skew_option: ScriptParameter,
    pub(crate) skew_default_grouping: ScriptParameter,
    pub(crate) skew_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) local_skew_option: ScriptParameter,
    pub(crate) local_skew_default_grouping: ScriptParameter,
    pub(crate) local_skew_nondefault_group: Option<Vec<FortranInputString>>,
    pub(crate) residual_report_criterion: ScriptParameter,
    pub(crate) surfaces_to_analyze: ScriptParameter,
    pub(crate) metro_factor: ScriptParameter,
    pub(crate) maximum_cycles: ScriptParameter,
    pub(crate) axis_z_shift: ScriptParameter,
    pub(crate) local_alignments: EtomoBoolean2,
    pub(crate) output_local_file: Option<String>,
    pub(crate) number_of_local_patches_xand_y: FortranInputString,
    pub(crate) target_patch_size_xand_y: FortranInputString,
    pub(crate) min_size_or_overlap_xand_y: FortranInputString,
    pub(crate) min_fids_total_and_each_surface: FortranInputString,
    pub(crate) fix_xyz_coordinates: EtomoBoolean2,
    pub(crate) local_output_options: FortranInputString,
    pub(crate) images_are_binned: ScriptParameter,
    pub(crate) beam_tilt_option: ScriptParameter,
    /// Behind a `Mutex` because `TiltalignParam.updateComScriptCommand` resets it
    /// (TiltalignParam.java:460-462) from a method that takes `&self` here.
    pub(crate) fixed_or_initial_beam_tilt: Mutex<ScriptParameter>,
    pub(crate) output_x_axis_tilt_file: Option<String>,
    pub(crate) robust_fitting: EtomoBoolean2,
    pub(crate) weight_whole_tracks: EtomoBoolean2,
    pub(crate) k_factor_scaling: ScriptParameter,
    pub(crate) x_tilt_option: ScriptParameter,
    pub(crate) x_tilt_default_grouping: ScriptParameter,
    pub(crate) cross_validate_deprecated: EtomoBoolean2,
    pub(crate) cross_validate: ScriptParameter,
    pub(crate) created_day_stamp: ScriptParameter,

    pub(crate) axis_id: AxisID,
    pub(crate) dataset_name: Option<String>,
    pub(crate) loaded_from_file: bool,
    pub(crate) manager: &'static dyn BaseManager,
}

impl ConstTiltalignParam {
    /// Java `ConstTiltalignParam(BaseManager, String, AxisID)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        dataset_name: Option<&str>,
        axis_id: AxisID,
    ) -> ConstTiltalignParam {
        // Field initializers, in declaration order.
        let fixed_or_initial_beam_tilt =
            ScriptParameter::new_with_type_and_name(Type::Double, "FixedOrInitialBeamTilt");
        let robust_fitting = EtomoBoolean2::new_with_name(ROBUST_FITTING_KEY);
        let weight_whole_tracks = EtomoBoolean2::new_with_name(WEIGHT_WHOLE_TRACKS_KEY);
        let k_factor_scaling =
            ScriptParameter::new_with_type_and_name(Type::Double, K_FACTOR_SCALING_KEY);
        let x_tilt_option = ScriptParameter::new_with_name("XTiltOption");
        let x_tilt_default_grouping = ScriptParameter::new_with_name("XTiltDefaultGrouping");
        let cross_validate_deprecated = EtomoBoolean2::new_with_name(CROSS_VALIDATE_KEY);
        let cross_validate = ScriptParameter::new_with_name(CROSS_VALIDATE_KEY);
        let created_day_stamp = ScriptParameter::new_with_name("CreatedDayStamp");
        // Constructor body.
        let rotation_angle = ScriptParameter::new_with_type_and_name(Type::Double, ROT_ANGLE_KEY);
        let mut tilt_angle_spec = TiltAngleSpec::new();
        tilt_angle_spec.set_range_min_key(Some("FirstTiltAngle"), Some("first"));
        tilt_angle_spec.set_range_step_key(Some("TiltIncrement"), Some("increment"));
        tilt_angle_spec.set_tilt_angle_filename_key(Some("TiltFile"), Some("tiltFile"));
        let angle_offset = ScriptParameter::new_with_type_and_name(Type::Double, ANGLE_OFFSET_KEY);
        let projection_stretch = EtomoBoolean2::new_with_name(PROJECTION_STRETCH_KEY);
        let mut rot_option = ScriptParameter::new_with_type_and_name(Type::Integer, ROT_OPTION_KEY);
        rot_option
            .set_valid_values(Some(&ROT_OPTION_VALID_VALUES[..]))
            .set_display_value_int(AUTOMAPPED_OPTION);
        let mut rot_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, ROT_DEFAULT_GROUPING_KEY);
        rot_default_grouping.set_display_value_int(3);
        let rotation_fixed_view =
            ScriptParameter::new_with_type_and_name(Type::Integer, "RotationFixedView");
        let mut local_rot_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_ROT_OPTION_KEY);
        local_rot_option
            .set_valid_values(Some(&LOCAL_OPTION_VALID_VALUES[..]))
            .set_display_value_int(AUTOMAPPED_OPTION);
        let mut local_rot_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_ROT_DEFAULT_GROUPING_KEY);
        local_rot_default_grouping.set_display_value_int(6);
        let mut tilt_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, TILT_OPTION_KEY);
        tilt_option
            .set_valid_values(Some(&TILT_OPTION_VALID_VALUES[..]))
            .set_display_value_int(TILT_ALL_OPTION);
        let mut tilt_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, TILT_DEFAULT_GROUPING_KEY);
        tilt_default_grouping.set_display_value_int(5);
        let mut local_tilt_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_TILT_OPTION_KEY);
        local_tilt_option
            .set_valid_values(Some(&LOCAL_TILT_OPTION_VALID_VALUES[..]))
            .set_display_value_int(TILT_AUTOMAPPED_OPTION);
        let mut local_tilt_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_TILT_DEFAULT_GROUPING_KEY);
        local_tilt_default_grouping.set_display_value_int(6);
        let mag_reference_view =
            ScriptParameter::new_with_type_and_name(Type::Integer, MAG_REFERENCE_VIEW_KEY);
        let mut mag_option = ScriptParameter::new_with_type_and_name(Type::Integer, MAG_OPTION_KEY);
        mag_option
            .set_valid_values(Some(&OPTION_VALID_VALUES[..]))
            .set_display_value_int(ALL_OPTION);
        let mut mag_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, MAG_DEFAULT_GROUPING_KEY);
        mag_default_grouping.set_display_value_int(4);
        let local_mag_reference_view =
            ScriptParameter::new_with_type_and_name(Type::Integer, "LocalMagReferenceView");
        let mut local_mag_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_MAG_OPTION_KEY);
        local_mag_option
            .set_valid_values(Some(&LOCAL_OPTION_VALID_VALUES[..]))
            .set_display_value_int(AUTOMAPPED_OPTION);
        let mut local_mag_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_MAG_DEFAULT_GROUPING_KEY);
        local_mag_default_grouping.set_display_value_int(7);
        let mut x_stretch_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, "XStretchOption");
        x_stretch_option
            .set_valid_values(Some(&DISTORTION_OPTION_VALID_VALUES[..]))
            .set_display_value_int(NONE_OPTION);
        let mut x_stretch_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, X_STRETCH_DEFAULT_GROUPING_KEY);
        x_stretch_default_grouping.set_display_value_int(7);
        let mut local_x_stretch_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, "LocalXStretchOption");
        local_x_stretch_option
            .set_valid_values(Some(&LOCAL_OPTION_VALID_VALUES[..]))
            .set_display_value_int(AUTOMAPPED_OPTION);
        let mut local_x_stretch_default_grouping = ScriptParameter::new_with_type_and_name(
            Type::Integer,
            LOCAL_X_STRETCH_DEFAULT_GROUPING_KEY,
        );
        local_x_stretch_default_grouping.set_display_value_int(7);
        let mut skew_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, SKEW_OPTION_KEY);
        skew_option
            .set_valid_values(Some(&DISTORTION_OPTION_VALID_VALUES[..]))
            .set_display_value_int(NONE_OPTION);
        let mut skew_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, SKEW_DEFAULT_GROUPING_KEY);
        skew_default_grouping.set_display_value_int(11);
        let mut local_skew_option =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_SKEW_OPTION_KEY);
        local_skew_option
            .set_valid_values(Some(&OPTION_VALID_VALUES[..]))
            .set_display_value_int(AUTOMAPPED_OPTION);
        let mut local_skew_default_grouping =
            ScriptParameter::new_with_type_and_name(Type::Integer, LOCAL_SKEW_DEFAULT_GROUPING_KEY);
        local_skew_default_grouping.set_display_value_int(11);
        let residual_report_criterion =
            ScriptParameter::new_with_type_and_name(Type::Double, RESIDUAL_REPORT_CRITERION_KEY);
        let mut surfaces_to_analyze =
            ScriptParameter::new_with_type_and_name(Type::Integer, SURFACES_TO_ANALYZE_KEY);
        surfaces_to_analyze.set_valid_values(Some(&SURFACES_TO_ANALYZE_VALID_VALUES[..]));
        let metro_factor = ScriptParameter::new_with_type_and_name(Type::Double, METRO_FACTOR_KEY);
        let maximum_cycles =
            ScriptParameter::new_with_type_and_name(Type::Integer, MAXIMUM_CYCLES_KEY);
        let axis_z_shift = ScriptParameter::new_with_type_and_name(Type::Double, AXIS_Z_SHIFT_KEY);
        let mut local_alignments = EtomoBoolean2::new_with_name(LOCAL_ALIGNMENTS_KEY);
        local_alignments.set_display_as_integer(true);
        let mut fix_xyz_coordinates = EtomoBoolean2::new_with_name(FIX_XYZ_COORDINATES_KEY);
        fix_xyz_coordinates.set_display_as_integer(true);
        // do not default imagesAreBinnned
        let mut images_are_binned = ScriptParameter::new_with_name("ImagesAreBinned");
        images_are_binned.set_floor(1);
        let mut beam_tilt_option = ScriptParameter::new_with_name(BEAM_TILT_OPTION_KEY);
        beam_tilt_option.set_default_int(FIXED_OPTION);
        beam_tilt_option.set_display_value_int(FIXED_OPTION);
        beam_tilt_option.set_valid_values(Some(&[FIXED_OPTION, BEAM_SEARCH_OPTION][..]));
        let mut fixed_or_initial_beam_tilt = fixed_or_initial_beam_tilt;
        fixed_or_initial_beam_tilt.set_default_int(0);
        let mut instance = ConstTiltalignParam {
            // `reset()` assigns every one of these below.
            model_file: None,
            image_file: None,
            output_model_and_residual: None,
            output_model_file: None,
            output_residual_file: None,
            output_fid_xyz_file: None,
            output_tilt_file: None,
            output_transform_file: None,
            output_z_factor_file: None,
            include_start_end_inc: FortranInputString::new(3),
            include_list: StringList::new(),
            exclude_list: StringList::new(),
            rotation_angle,
            separate_group: StringList::new(),
            tilt_angle_spec,
            angle_offset,
            projection_stretch,
            rot_option,
            rot_default_grouping,
            rot_nondefault_group: None,
            rotation_fixed_view,
            local_rot_option,
            local_rot_default_grouping,
            local_rot_nondefault_group: None,
            tilt_option,
            tilt_default_grouping,
            tilt_nondefault_group: None,
            local_tilt_option,
            local_tilt_default_grouping,
            local_tilt_nondefault_group: None,
            mag_reference_view,
            mag_option,
            mag_default_grouping,
            mag_nondefault_group: None,
            local_mag_reference_view,
            local_mag_option,
            local_mag_default_grouping,
            local_mag_nondefault_group: None,
            x_stretch_option,
            x_stretch_default_grouping,
            x_stretch_nondefault_group: None,
            local_x_stretch_option,
            local_x_stretch_default_grouping,
            local_x_stretch_nondefault_group: None,
            skew_option,
            skew_default_grouping,
            skew_nondefault_group: None,
            local_skew_option,
            local_skew_default_grouping,
            local_skew_nondefault_group: None,
            residual_report_criterion,
            surfaces_to_analyze,
            metro_factor,
            maximum_cycles,
            axis_z_shift,
            local_alignments,
            output_local_file: None,
            number_of_local_patches_xand_y: FortranInputString::new(2),
            target_patch_size_xand_y: FortranInputString::new(2),
            min_size_or_overlap_xand_y: FortranInputString::new(2),
            min_fids_total_and_each_surface: FortranInputString::new(2),
            fix_xyz_coordinates,
            local_output_options: FortranInputString::new(3),
            images_are_binned,
            beam_tilt_option,
            fixed_or_initial_beam_tilt: Mutex::new(fixed_or_initial_beam_tilt),
            output_x_axis_tilt_file: Some(String::new()),
            robust_fitting,
            weight_whole_tracks,
            k_factor_scaling,
            x_tilt_option,
            x_tilt_default_grouping,
            cross_validate_deprecated,
            cross_validate,
            created_day_stamp,
            axis_id,
            dataset_name: dataset_name.map(str::to_owned),
            loaded_from_file: false,
            manager,
        };
        instance.reset();
        instance
    }

    /// Java package-private `reset()`.
    pub(crate) fn reset(&mut self) {
        self.loaded_from_file = false;
        self.model_file = Some(String::new());
        self.image_file = Some(String::new());
        self.output_model_and_residual = Some(String::new());
        self.output_model_file = Some(String::new());
        self.output_residual_file = Some(String::new());
        self.output_fid_xyz_file = Some(String::new());
        self.output_tilt_file = Some(String::new());
        self.output_transform_file = Some(String::new());
        self.output_z_factor_file = Some(String::new());
        self.include_start_end_inc = FortranInputString::new(3);
        self.include_start_end_inc
            .set_integer_type_array(&[true, true, true]);
        self.include_list = StringList::new();
        self.exclude_list = StringList::new();
        self.rotation_angle.reset();
        self.separate_group = StringList::new();
        self.separate_group.set_key(Some(SEPARATE_GROUP_KEY));
        self.separate_group.set_successive_entries_accumulate();
        self.tilt_angle_spec.reset();
        self.angle_offset.reset();
        self.projection_stretch.reset();
        self.rot_option.reset();
        self.rot_default_grouping.reset();
        self.rot_nondefault_group = None;
        self.rotation_fixed_view.reset();
        self.local_rot_option.reset();
        self.local_rot_default_grouping.reset();
        self.local_rot_nondefault_group = None;
        self.tilt_option.reset();
        self.tilt_default_grouping.reset();
        self.tilt_nondefault_group = None;
        self.local_tilt_option.reset();
        self.local_tilt_default_grouping.reset();
        self.local_tilt_nondefault_group = None;
        self.mag_reference_view.reset();
        self.mag_option.reset();
        self.mag_default_grouping.reset();
        self.mag_nondefault_group = None;
        self.local_mag_option.reset();
        self.local_mag_default_grouping.reset();
        self.local_mag_nondefault_group = None;
        self.x_stretch_option.reset();
        self.x_stretch_default_grouping.reset();
        self.x_stretch_nondefault_group = None;
        self.local_x_stretch_option.reset();
        self.local_x_stretch_default_grouping.reset();
        self.local_x_stretch_nondefault_group = None;
        self.skew_option.reset();
        self.skew_default_grouping.reset();
        self.skew_nondefault_group = None;
        self.local_skew_option.reset();
        self.local_skew_default_grouping.reset();
        self.local_skew_nondefault_group = None;
        self.residual_report_criterion.reset();
        self.surfaces_to_analyze.reset();
        self.metro_factor.reset();
        self.maximum_cycles.reset();
        self.axis_z_shift.reset();
        self.local_alignments.reset();
        self.output_local_file = Some(String::new());
        self.number_of_local_patches_xand_y = FortranInputString::new(2);
        self.target_patch_size_xand_y = FortranInputString::new(2);
        self.number_of_local_patches_xand_y
            .set_integer_type_array(&[true, true]);
        self.target_patch_size_xand_y
            .set_integer_type_array(&[true, true]);
        self.min_size_or_overlap_xand_y = FortranInputString::new(2);
        self.min_fids_total_and_each_surface = FortranInputString::new(2);
        self.min_fids_total_and_each_surface
            .set_integer_type_array(&[true, true]);
        self.fix_xyz_coordinates.reset();
        self.local_output_options = FortranInputString::new(3);
        self.local_output_options
            .set_integer_type_array(&[true, true, true]);
        // Preserve imagesAreBinned value in case the image file is missing.
        self.beam_tilt_option.reset();
        self.fixed_or_initial_beam_tilt.get_mut().unwrap().reset();
        self.output_x_axis_tilt_file = Some(String::new());
        self.robust_fitting.reset();
        self.weight_whole_tracks.reset();
        self.k_factor_scaling.reset();
        self.x_tilt_option.reset();
        self.cross_validate_deprecated.reset();
        self.cross_validate.reset();
        self.created_day_stamp.reset();
    }

    /// Java `getImagesAreBinned`.
    pub fn get_images_are_binned(&self) -> &ConstEtomoNumber {
        &self.images_are_binned
    }

    /// Java package-private `validate`.
    pub(crate) fn validate(&self) -> String {
        let mut invalid_reason = String::new();
        if !self.rot_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.rot_option.get_description(),
                self.rot_option.get_invalid_reason()
            ));
        }
        if !self.local_rot_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.local_rot_option.get_description(),
                self.local_rot_option.get_invalid_reason()
            ));
        }
        if !self.tilt_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.tilt_option.get_description(),
                self.tilt_option.get_invalid_reason()
            ));
        }
        if !self.local_tilt_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.local_tilt_option.get_description(),
                self.local_tilt_option.get_invalid_reason()
            ));
        }
        if !self.mag_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.mag_option.get_description(),
                self.mag_option.get_invalid_reason()
            ));
        }
        if !self.local_mag_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.local_mag_option.get_description(),
                self.local_mag_option.get_invalid_reason()
            ));
        }
        if !self.x_stretch_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.x_stretch_option.get_description(),
                self.x_stretch_option.get_invalid_reason()
            ));
        }
        if !self.local_x_stretch_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.local_x_stretch_option.get_description(),
                self.local_x_stretch_option.get_invalid_reason()
            ));
        }
        if !self.skew_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.skew_option.get_description(),
                self.skew_option.get_invalid_reason()
            ));
        }
        if !self.local_skew_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.local_skew_option.get_description(),
                self.local_skew_option.get_invalid_reason()
            ));
        }
        if !self.surfaces_to_analyze.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.surfaces_to_analyze.get_description(),
                self.surfaces_to_analyze.get_invalid_reason()
            ));
        }
        if !self.beam_tilt_option.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                self.beam_tilt_option.get_description(),
                self.beam_tilt_option.get_invalid_reason()
            ));
        }
        let fixed_or_initial_beam_tilt = self.fixed_or_initial_beam_tilt.lock().unwrap();
        if !fixed_or_initial_beam_tilt.is_valid() {
            invalid_reason.push_str(&format!(
                "{}: {}\n",
                fixed_or_initial_beam_tilt.get_description(),
                fixed_or_initial_beam_tilt.get_invalid_reason()
            ));
        }
        invalid_reason
    }

    /// Java `isExcludeListAvailable`.
    pub fn is_exclude_list_available(&self) -> bool {
        (self.include_start_end_inc.is_default() || !self.include_start_end_inc.values_set())
            && self.include_list.get_n_elements() == 0
    }

    /// Java `getAngleOffset`.
    pub fn get_angle_offset(&self) -> &ConstEtomoNumber {
        &self.angle_offset
    }

    /// Java `getAxisZShift`.
    pub fn get_axis_z_shift(&self) -> &ConstEtomoNumber {
        &self.axis_z_shift
    }

    /// Java `getExcludeList`.
    pub fn get_exclude_list(&self) -> String {
        self.exclude_list.to_string()
    }

    /// Java `getFixXYZCoordinates`.
    pub fn get_fix_xyz_coordinates(&self) -> &ConstEtomoNumber {
        &self.fix_xyz_coordinates
    }

    /// Java `getImageFile`.
    pub fn get_image_file(&self) -> Option<&str> {
        self.image_file.as_deref()
    }

    /// Java `getLocalAlignments`.
    pub fn get_local_alignments(&self) -> &ConstEtomoNumber {
        &self.local_alignments
    }

    /// Java `getLocalMagDefaultGrouping`.
    pub fn get_local_mag_default_grouping(&self) -> &ConstEtomoNumber {
        &self.local_mag_default_grouping
    }

    /// Java `getLocalMagNondefaultGroup`.
    pub fn get_local_mag_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(
            self.local_mag_nondefault_group.as_deref(),
        )
    }

    /// Java `getLocalMagOption`.
    pub fn get_local_mag_option(&self) -> &ConstEtomoNumber {
        &self.local_mag_option
    }

    /// Java `getLocalRotDefaultGrouping`.
    pub fn get_local_rot_default_grouping(&self) -> &ConstEtomoNumber {
        &self.local_rot_default_grouping
    }

    /// Java `getLocalRotNondefaultGroup`.
    pub fn get_local_rot_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(
            self.local_rot_nondefault_group.as_deref(),
        )
    }

    /// Java `getLocalRotOption`.
    pub fn get_local_rot_option(&self) -> &ConstEtomoNumber {
        &self.local_rot_option
    }

    /// Java `getBeamTiltOption`.
    pub fn get_beam_tilt_option(&self) -> &ConstEtomoNumber {
        &self.beam_tilt_option
    }

    /// Java `getFixedOrInitialBeamTilt`.
    /// The value is copied out of its `Mutex` (see the field).
    pub fn get_fixed_or_initial_beam_tilt(&self) -> ConstEtomoNumber {
        self.fixed_or_initial_beam_tilt
            .lock()
            .unwrap()
            .base
            .base
            .clone()
    }

    /// Java `getLocalSkewDefaultGrouping`.
    pub fn get_local_skew_default_grouping(&self) -> &ConstEtomoNumber {
        &self.local_skew_default_grouping
    }

    /// Java `getLocalSkewNondefaultGroup`.
    pub fn get_local_skew_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(
            self.local_skew_nondefault_group.as_deref(),
        )
    }

    /// Java `getLocalSkewOption`.
    pub fn get_local_skew_option(&self) -> &ConstEtomoNumber {
        &self.local_skew_option
    }

    /// Java `getLocalTiltDefaultGrouping`.
    pub fn get_local_tilt_default_grouping(&self) -> &ConstEtomoNumber {
        &self.local_tilt_default_grouping
    }

    /// Java `getLocalTiltNondefaultGroup`.
    pub fn get_local_tilt_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(
            self.local_tilt_nondefault_group.as_deref(),
        )
    }

    /// Java `getLocalTiltOption`.
    pub fn get_local_tilt_option(&self) -> &ConstEtomoNumber {
        &self.local_tilt_option
    }

    /// Java `getLocalXStretchDefaultGrouping`.
    pub fn get_local_x_stretch_default_grouping(&self) -> &ConstEtomoNumber {
        &self.local_x_stretch_default_grouping
    }

    /// Java `getLocalXStretchNondefaultGroup`.
    pub fn get_local_x_stretch_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(
            self.local_x_stretch_nondefault_group.as_deref(),
        )
    }

    /// Java `getLocalXStretchOption`.
    pub fn get_local_x_stretch_option(&self) -> &ConstEtomoNumber {
        &self.local_x_stretch_option
    }

    /// Java `getMagDefaultGrouping`.
    pub fn get_mag_default_grouping(&self) -> &ConstEtomoNumber {
        &self.mag_default_grouping
    }

    /// Java `getMagNondefaultGroup`.
    pub fn get_mag_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(self.mag_nondefault_group.as_deref())
    }

    /// Java `getMagOption`.
    pub fn get_mag_option(&self) -> &ConstEtomoNumber {
        &self.mag_option
    }

    /// Java `getMagReferenceView`.
    pub fn get_mag_reference_view(&self) -> &ConstEtomoNumber {
        &self.mag_reference_view
    }

    /// Java `getMaximumCycles`.
    pub fn get_maximum_cycles(&self) -> &ConstEtomoNumber {
        &self.maximum_cycles
    }

    /// Java `getMetroFactor`.
    pub fn get_metro_factor(&self) -> &ConstEtomoNumber {
        &self.metro_factor
    }

    /// Java `getMinFidsTotalAndEachSurface`.
    pub fn get_min_fids_total_and_each_surface(&self) -> String {
        self.min_fids_total_and_each_surface
            .to_string_default_is_blank(true)
    }

    /// Java `getMinSizeOrOverlapXandY`.
    pub fn get_min_size_or_overlap_xand_y(&self) -> String {
        self.min_size_or_overlap_xand_y
            .to_string_default_is_blank(true)
    }

    /// Java `getModelFile`.
    pub fn get_model_file(&self) -> Option<&str> {
        self.model_file.as_deref()
    }

    /// Java `getNumberOfLocalPatchesXandY`.
    pub fn get_number_of_local_patches_xand_y(&self) -> String {
        self.number_of_local_patches_xand_y
            .to_string_default_is_blank(true)
    }

    /// Java `isTargetPatchSizeXandYEmpty`.  Returns true if the parameter was not in
    /// the .com file.
    pub fn is_target_patch_size_xand_y_empty(&self) -> bool {
        self.target_patch_size_xand_y.is_empty()
    }

    /// Java `isNumberOfLocalPatchesXandYEmpty`.  Returns true if the parameter was not
    /// in the .com file.
    pub fn is_number_of_local_patches_xand_y_empty(&self) -> bool {
        self.number_of_local_patches_xand_y.is_empty()
    }

    /// Java `getTargetPatchSizeXandY`.
    pub fn get_target_patch_size_xand_y(&self) -> String {
        self.target_patch_size_xand_y
            .to_string_default_is_blank(true)
    }

    /// Java `getOutputFidXYZFile`.
    pub fn get_output_fid_xyz_file(&self) -> Option<&str> {
        self.output_fid_xyz_file.as_deref()
    }

    /// Java `getOutputLocalFile`.
    pub fn get_output_local_file(&self) -> Option<&str> {
        self.output_local_file.as_deref()
    }

    /// Java `getOutputModelFile`.
    pub fn get_output_model_file(&self) -> Option<&str> {
        self.output_model_file.as_deref()
    }

    /// Java `getOutputResidualFile`.
    pub fn get_output_residual_file(&self) -> Option<&str> {
        self.output_residual_file.as_deref()
    }

    /// Java `getOutputTiltFile`.
    pub fn get_output_tilt_file(&self) -> Option<&str> {
        self.output_tilt_file.as_deref()
    }

    /// Java `getOutputTransformFile`.
    pub fn get_output_transform_file(&self) -> Option<&str> {
        self.output_transform_file.as_deref()
    }

    /// Java `useOutputZFactorFile`.  This must called after skewOption, or
    /// localAlignment, and localSkewOption have been set.
    pub fn use_output_z_factor_file(&self) -> bool {
        !self.skew_option.equals_int(FIXED_OPTION)
            || (self.local_alignments.is() && !self.local_skew_option.equals_int(FIXED_OPTION))
    }

    /// Java `getOutputZFactorFile`.
    pub fn get_output_z_factor_file(&self) -> Option<&str> {
        self.output_z_factor_file.as_deref()
    }

    /// Java static `getOutputZFactorFileName`.  Build an outputZFactorFile value from
    /// datasetName and axisID.  Java string concatenation writes a null name as "null".
    pub fn get_output_z_factor_file_name(dataset_name: Option<&str>, axis_id: AxisID) -> String {
        format!(
            "{}{}{}",
            dataset_name.unwrap_or("null"),
            axis_id.get_extension(),
            Z_FACTOR_FILE_EXTENSION
        )
    }

    /// Java static `getOutputLocalFileName`.
    pub fn get_output_local_file_name(dataset_name: Option<&str>, axis_id: AxisID) -> String {
        format!(
            "{}{}{}",
            dataset_name.unwrap_or("null"),
            axis_id.get_extension(),
            LOCAL_FILE_EXTENSION
        )
    }

    /// Java `getProjectionStretch`.
    pub fn get_projection_stretch(&self) -> &ConstEtomoNumber {
        &self.projection_stretch
    }

    /// Java `getResidualReportCriterion`.
    pub fn get_residual_report_criterion(&self) -> &ConstEtomoNumber {
        &self.residual_report_criterion
    }

    /// Java `isRobustFitting`.
    pub fn is_robust_fitting(&self) -> bool {
        self.robust_fitting.is()
    }

    /// Java `isWeightWholeTracks`.
    pub fn is_weight_whole_tracks(&self) -> bool {
        self.weight_whole_tracks.is()
    }

    /// Java `isCrossValidate`.
    pub fn is_cross_validate(&self) -> bool {
        self.cross_validate.is()
    }

    /// Java `isCreatedDayStampIMOD_5_0_1`.
    pub fn is_created_day_stamp_imod_5_0_1(&self) -> bool {
        self.created_day_stamp
            .ge_long(CREATED_DAY_STAMP_IMOD_5_0_1 as i64)
    }

    /// Java `getKFactorScaling`.
    pub fn get_k_factor_scaling(&self) -> String {
        self.k_factor_scaling.to_string()
    }

    /// Java `getRotationAngle`.
    pub fn get_rotation_angle(&self) -> &ConstEtomoNumber {
        &self.rotation_angle
    }

    /// Java `getRotationFixedView`.
    pub fn get_rotation_fixed_view(&self) -> &ConstEtomoNumber {
        &self.rotation_fixed_view
    }

    /// Java `getRotDefaultGrouping`.
    pub fn get_rot_default_grouping(&self) -> &ConstEtomoNumber {
        &self.rot_default_grouping
    }

    /// Java `getRotNondefaultGroup`.
    pub fn get_rot_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(self.rot_nondefault_group.as_deref())
    }

    /// Java `getRotOption`.
    pub fn get_rot_option(&self) -> &ConstEtomoNumber {
        &self.rot_option
    }

    /// Java `getSeparateGroup`.
    pub fn get_separate_group(&self) -> String {
        self.separate_group.to_string()
    }

    /// Java `getSkewDefaultGrouping`.
    pub fn get_skew_default_grouping(&self) -> &ConstEtomoNumber {
        &self.skew_default_grouping
    }

    /// Java `getSkewNondefaultGroup`.
    pub fn get_skew_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(self.skew_nondefault_group.as_deref())
    }

    /// Java `getSkewOption`.
    pub fn get_skew_option(&self) -> &ConstEtomoNumber {
        &self.skew_option
    }

    /// Java `getSurfacesToAnalyze`.
    pub fn get_surfaces_to_analyze(&self) -> &ConstEtomoNumber {
        &self.surfaces_to_analyze
    }

    /// Java `getTiltAngleSpec`.
    pub fn get_tilt_angle_spec(&self) -> &TiltAngleSpec {
        &self.tilt_angle_spec
    }

    /// Java `getTiltDefaultGrouping`.
    pub fn get_tilt_default_grouping(&self) -> &ConstEtomoNumber {
        &self.tilt_default_grouping
    }

    /// Java `getTiltNondefaultGroup`.
    pub fn get_tilt_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(self.tilt_nondefault_group.as_deref())
    }

    /// Java `getTiltOption`.
    pub fn get_tilt_option(&self) -> &ConstEtomoNumber {
        &self.tilt_option
    }

    /// Java `getXStretchDefaultGrouping`.
    pub fn get_x_stretch_default_grouping(&self) -> &ConstEtomoNumber {
        &self.x_stretch_default_grouping
    }

    /// Java `getXStretchNondefaultGroup`.
    pub fn get_x_stretch_nondefault_group(&self) -> String {
        param_utilities::value_of_fortran_input_string_array(
            self.x_stretch_nondefault_group.as_deref(),
        )
    }

    /// Java `getXStretchOption`.
    pub fn get_x_stretch_option(&self) -> &ConstEtomoNumber {
        &self.x_stretch_option
    }

    /// Java `isXTiltOptionAutomapSame`.
    pub fn is_x_tilt_option_automap_same(&self) -> bool {
        XTiltOption::get_instance(self.x_tilt_option.get_int()) == Some(XTiltOption::AUTOMAP_SAME)
    }

    /// Java `isOldVersion`.  Identifies an old version.
    pub fn is_old_version(&self) -> bool {
        self.loaded_from_file && self.images_are_binned.is_null()
    }
}

/// Java nested class `ConstTiltalignParam.Fields`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fields {
    UseOutputZFactorFile,
    LocalAlignments,
    AxisZShift,
    AngleOffset,
}

impl FieldInterface for Fields {}

impl Command for ConstTiltalignParam {
    /// Java `getAxisID`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommandMode`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        None
    }

    /// Java `getProcessName`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommand`.
    fn get_command(&self) -> Option<String> {
        Some(format!(
            "{}{}{}",
            PROCESS_NAME,
            self.axis_id.get_extension(),
            COMMAND_FILE_EXTENSION
        ))
    }

    /// Java `getCommandName`.
    fn get_command_name(&self) -> Option<String> {
        Some(PROCESS_NAME.to_string())
    }

    /// Java `getCommandLine`.
    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `getCommandArray`.  `{ getCommandLine() }`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        // A one-element Java array holding `getCommandLine()`, which is never null here.
        Some(vec![self.get_command_line().unwrap_or_default()])
    }

    /// Java `getCommandInputFile`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandOutputFile`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getOutputImageFileType` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        Some(file_type::CLASS.fiducial_3d_model.clone())
    }

    /// Java `getOutputImageFileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        Some(FileKey::clone(&file_type::CLASS.fiducial_3d_model))
    }

    /// Java `getOutputImageFileType2` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `isMessageReporter`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandProcessName`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getSubcommandDetails`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// `ConstTiltalignParam implements CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

/// Java's getters throw `IllegalArgumentException("field=" + field)` for a field they
/// do not handle; each such case returns `None` here.
impl ProcessDetails for ConstTiltalignParam {
    /// Java `getIntValue`: handles no field.
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    /// Java `getBooleanValue`.
    fn get_boolean_value(&self, field: &dyn FieldInterface) -> Option<bool> {
        match field_interface::as_field::<Fields>(field) {
            Some(Fields::UseOutputZFactorFile) => Some(self.use_output_z_factor_file()),
            Some(Fields::LocalAlignments) => Some(self.local_alignments.is()),
            _ => None,
        }
    }

    /// Java `getDoubleValue`.
    fn get_double_value(&self, field: &dyn FieldInterface) -> Option<f64> {
        match field_interface::as_field::<Fields>(field) {
            Some(Fields::AxisZShift) => Some(self.axis_z_shift.get_double()),
            Some(Fields::AngleOffset) => Some(self.angle_offset.get_double()),
            _ => None,
        }
    }

    /// Java `getHashtable`: handles no field.
    fn get_hashtable(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<super::process_details::Hashtable> {
        None
    }

    /// Java `getEtomoNumber`: handles no field.
    fn get_etomo_number(&self, _field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        None
    }

    /// Java `getIntKeyList`: handles no field.
    fn get_int_key_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::int_key_list::IntKeyList> {
        None
    }

    /// Java `getString`: handles no field.
    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    /// Java `getStringArray`: handles no field.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getIteratorElementList`: handles no field.
    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }
}

impl Loggable for ConstTiltalignParam {
    /// Java `getName`.
    fn get_name(&self) -> String {
        PROCESS_NAME.to_string()
    }

    /// Java `getLogMessage`: returns null, which is an empty message here.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}
