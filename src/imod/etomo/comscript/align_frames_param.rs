//! `IMOD/Etomo/src/etomo/comscript/AlignFramesParam.java`.
//!
//! The `alignframes` command of an align-frames com file (Tools > Align Frames).
//! Implements `CommandParam`.  Java `String` getters return `toString()` of the
//! parameter (never null); setters take Java's nullable `String` as `Option<&str>`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::eer_super_res::EERSuperRes;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::status::Status;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java private static final `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::ALIGN_FRAMES;

/// Java package-private static final `COMMAND`.
pub const COMMAND: &str = "alignframes";

/// Java private static final `METADATA_FILE`.
const METADATA_FILE: &str = "MetadataFile";
/// Java private static final `LIST_OF_INPUT_FILES`.
const LIST_OF_INPUT_FILES: &str = "ListOfInputFiles";
/// Java private static final `SELECTED_FILES`.
const SELECTED_FILES: &str = "SelectedFiles";
/// Java private static final `DIRECTORY`.
const DIRECTORY: &str = "Directory";
/// Java private static final `INPUT_FILE`.
const INPUT_FILE: &str = "InputFile";
/// Java private static final `PATH_TO_FRAMES_IN_MDOC`.
const PATH_TO_FRAMES_IN_MDOC: &str = "PathToFramesInMdoc";
/// Java private static final `CORRESPONDING_STACK`.
const CORRESPONDING_STACK: &str = "CorrespondingStack";
/// Java private static final `TILT_ANGLE_FILE`.
const TILT_ANGLE_FILE: &str = "TiltAngleFile";
/// Java private static final `AXIS_ROTATION_ANGLE`.
const AXIS_ROTATION_ANGLE: &str = "AxisRotationAngle";
/// Java private static final `REF_AND_DEFECT_FROM_TITLES`.
const REF_AND_DEFECT_FROM_TITLES: &str = "RefAndDefectFromTitles";
/// Java private static final `GAIN_REFERENCE_FILE`.
const GAIN_REFERENCE_FILE: &str = "GainReferenceFile";
/// Java private static final `ROTATION_AND_FLIP`.
const ROTATION_AND_FLIP: &str = "RotationAndFlip";
/// Java private static final `CAMERA_DEFECT_FILE`.
const CAMERA_DEFECT_FILE: &str = "CameraDefectFile";
/// Java private static final `TRUNCATE_ABOVE`.
const TRUNCATE_ABOVE: &str = "TruncateAbove";
/// Java private static final `PAIRWISE_FRAMES`.
const PAIRWISE_FRAMES: &str = "PairwiseFrames";
/// Java private static final `TARGET_ALIGN_SIZE`.
const TARGET_ALIGN_SIZE: &str = "TargetAlignSize";
/// Java private static final `TEST_BINNINGS`.
const TEST_BINNINGS: &str = "TestBinnings";
/// Java private static final `ALIGN_AND_SUM_BINNING`.
const ALIGN_AND_SUM_BINNING: &str = "AlignAndSumBinning";
/// Java private static final `VARY_FILTER`.
const VARY_FILTER: &str = "VaryFilter";
/// Java private static final `USE_HYBRID_SHIFTS`.
const USE_HYBRID_SHIFTS: &str = "UseHybridShifts";
/// Java private static final `SHIFT_LIMIT`.
const SHIFT_LIMIT: &str = "ShiftLimit";
/// Java private static final `GROUP_SIZE`.
const GROUP_SIZE: &str = "GroupSize";
/// Java private static final `REFINE_ALIGNMENT`.
const REFINE_ALIGNMENT: &str = "RefineAlignment";
/// Java private static final `REFINE_WITH_GROUP_SUMS`.
const REFINE_WITH_GROUP_SUMS: &str = "RefineWithGroupSums";
/// Java private static final `REFINE_RADIUS_2`.
const REFINE_RADIUS_2: &str = "RefineRadius2";
/// Java private static final `STOP_ITERATIONS_AT_SHIFT`.
const STOP_ITERATIONS_AT_SHIFT: &str = "StopIterationsAtShift";
/// Java private static final `MIN_FOR_SPLINE_SMOOTHING`.
const MIN_FOR_SPLINE_SMOOTHING: &str = "MinForSplineSmoothing";
/// Java private static final `STARTING_ENDING_FRAMES`.
const STARTING_ENDING_FRAMES: &str = "StartingEndingFrames";
/// Java private static final `TYPE_OF_DOSE_FILE`.
const TYPE_OF_DOSE_FILE: &str = "TypeOfDoseFile";
/// Java private static final `DOSE_WEIGHTING_FILE`.
const DOSE_WEIGHTING_FILE: &str = "DoseWeightingFile";
/// Java private static final `FIXED_TOTAL_DOSE`.
const FIXED_TOTAL_DOSE: &str = "FixedTotalDose";
/// Java private static final `NORMALIZE_DOSE_WEIGHTING`.
const NORMALIZE_DOSE_WEIGHTING: &str = "NormalizeDoseWeighting";
/// Java private static final `VOLTAGE`.
const VOLTAGE: &str = "Voltage";
/// Java private static final `OPTIMAL_DOSE_SCALING`.
const OPTIMAL_DOSE_SCALING: &str = "OptimalDoseScaling";
/// Java private static final `UNWEIGHTED_OUTPUT_FILE`.
const UNWEIGHTED_OUTPUT_FILE: &str = "UnweightedOutputFile";
/// Java public static final `UNWEIGHTED_OUTPUT_FILENAME`.
pub const UNWEIGHTED_OUTPUT_FILENAME: &str = "_af_nodw";
/// Java private static final `SCALING_OF_SUM`.
const SCALING_OF_SUM: &str = "ScalingOfSum";
/// Java private static final `MODE_TO_OUTPUT`.
const MODE_TO_OUTPUT: &str = "ModeToOutput";
/// Java private static final `SUM_ROTATION_AND_FLIP`.
const SUM_ROTATION_AND_FLIP: &str = "SumRotationAndFlip";
/// Java private static final `OUTPUT_IMAGE_FILE`.
const OUTPUT_IMAGE_FILE: &str = "OutputImageFile";
/// Java public static final `OUTPUT_IMAGE_FILE_AF`.
pub const OUTPUT_IMAGE_FILE_AF: &str = "_af";
/// Java public static final `OUTPUT_IMAGE_FILE_AF_DW`.
pub const OUTPUT_IMAGE_FILE_AF_DW: &str = "_af_dw";
/// Java private static final `USE_GPU`.
const USE_GPU: &str = "UseGPU";
/// Java private static final `ADJUST_AND_WRITE_MDOC`.
const ADJUST_AND_WRITE_MDOC: &str = "AdjustAndWriteMdoc";
/// Java public static final `ROTATION_AND_FLIP_DEFAULT`.
pub const ROTATION_AND_FLIP_DEFAULT: i32 = -1;
/// Java public static final `ROTATION_AND_FLIP_SPINNER_DEFAULT`.
pub const ROTATION_AND_FLIP_SPINNER_DEFAULT: i32 = 0;
/// Java public static final `ROTATION_AND_FLIP_SPINNER_MIN`.
pub const ROTATION_AND_FLIP_SPINNER_MIN: i32 = 0;
/// Java public static final `ROTATION_AND_FLIP_SPINNER_MAX`.
pub const ROTATION_AND_FLIP_SPINNER_MAX: i32 = 7;
/// Java public static final `PAIRWISE_FRAMES_SPINNER_DEFAULT`.
pub const PAIRWISE_FRAMES_SPINNER_DEFAULT: i32 = 7;
/// Java public static final `PAIRWISE_FRAMES_SPINNER_MIN`.
pub const PAIRWISE_FRAMES_SPINNER_MIN: i32 = 7;
/// Java public static final `PAIRWISE_FRAMES_SPINNER_MAX`.
pub const PAIRWISE_FRAMES_SPINNER_MAX: i32 = 99;
/// Java public static final `HALF_PAIRWISE_FRAMES`.
pub const HALF_PAIRWISE_FRAMES: i32 = -2;
/// Java public static final `ALL_PAIRWISE_FRAMES`.
pub const ALL_PAIRWISE_FRAMES: i32 = -1;
/// Java private static final `TEST_BINNINGS_NPARAMS`.
const TEST_BINNINGS_NPARAMS: i32 = 4;
/// Java private static final `ALIGN_AND_SUM_BINNING_NPARAMS`.
const ALIGN_AND_SUM_BINNING_NPARAMS: i32 = 2;
/// Java public static final `ALIGN_AND_SUM_BINNING_DEFAULT`.
pub const ALIGN_AND_SUM_BINNING_DEFAULT: i32 = -1;
/// Java public static final `ALIGN_AND_SUM_BINNING_SPINNER_DEFAULT`.
pub const ALIGN_AND_SUM_BINNING_SPINNER_DEFAULT: i32 = 4;
/// Java public static final `ALIGN_AND_SUM_BINNING_SPINNER_MIN`.
pub const ALIGN_AND_SUM_BINNING_SPINNER_MIN: i32 = 1;
/// Java public static final `ALIGN_AND_SUM_BINNING_SPINNER_MAX`.
pub const ALIGN_AND_SUM_BINNING_SPINNER_MAX: i32 = 16;
/// Java private static final `VARY_FILTER_NPARAMS`.
const VARY_FILTER_NPARAMS: i32 = 6;
/// Java public static final `TARGET_ALIGN_SIZE_SPINNER_DEFAULT`.
pub const TARGET_ALIGN_SIZE_SPINNER_DEFAULT: i32 = 1250;
/// Java public static final `TARGET_ALIGN_SIZE_SPINNER_MIN`.
pub const TARGET_ALIGN_SIZE_SPINNER_MIN: i32 = 250;
/// Java public static final `TARGET_ALIGN_SIZE_SPINNER_MAX`.
pub const TARGET_ALIGN_SIZE_SPINNER_MAX: i32 = 4000;
/// Java public static final `TARGET_ALIGN_SIZE_SPINNER_STEP_SIZE`.
pub const TARGET_ALIGN_SIZE_SPINNER_STEP_SIZE: i32 = 50;
/// Java public static final `VARY_FILTER_DEFAULT`.
pub const VARY_FILTER_DEFAULT: &str = "0.05,0.06,0.08,0.1";
/// Java public static final `SHIFT_LIMIT_DEFAULT`.
pub const SHIFT_LIMIT_DEFAULT: i32 = 20;
/// Java public static final `GROUP_FRAMES_SPINNER_DEFAULT`.
pub const GROUP_FRAMES_SPINNER_DEFAULT: i32 = 2;
/// Java public static final `GROUP_FRAMES_SPINNER_MIN`.
pub const GROUP_FRAMES_SPINNER_MIN: i32 = 2;
/// Java public static final `GROUP_FRAMES_SPINNER_MAX`.
pub const GROUP_FRAMES_SPINNER_MAX: i32 = 20;
/// Java public static final `REFINE_ALIGNMENT_SPINNER_DEFAULT`.
pub const REFINE_ALIGNMENT_SPINNER_DEFAULT: i32 = 5;
/// Java public static final `REFINE_ALIGNMENT_SPINNER_MIN`.
pub const REFINE_ALIGNMENT_SPINNER_MIN: i32 = 1;
/// Java public static final `REFINE_ALIGNMENT_SPINNER_MAX`.
pub const REFINE_ALIGNMENT_SPINNER_MAX: i32 = 10;
/// Java public static final `STOP_ITERATIONS_AT_SHIFT_DEFAULT`.
pub const STOP_ITERATIONS_AT_SHIFT_DEFAULT: f64 = 0.1;
/// Java public static final `PREVENT_MIN_FOR_SPLINE_SMOOTHING`.
pub const PREVENT_MIN_FOR_SPLINE_SMOOTHING: i32 = 0;
/// Java public static final `MIN_FOR_SPLINE_SMOOTHING_SPINNER_DEFAULT`.
pub const MIN_FOR_SPLINE_SMOOTHING_SPINNER_DEFAULT: i32 = 20;
/// Java public static final `MIN_FOR_SPLINE_SMOOTHING_SPINNER_MIN`.
pub const MIN_FOR_SPLINE_SMOOTHING_SPINNER_MIN: i32 = 10;
/// Java public static final `MIN_FOR_SPLINE_SMOOTHING_SPINNER_MAX`.
pub const MIN_FOR_SPLINE_SMOOTHING_SPINNER_MAX: i32 = 30;
/// Java private static final `STARTING_ENDING_FRAMES_NPARAMS`.
const STARTING_ENDING_FRAMES_NPARAMS: i32 = 2;
/// Java public static final `TYPE_OF_DOSE_FILE_VAL_2`.
pub const TYPE_OF_DOSE_FILE_VAL_2: i32 = 2;
/// Java public static final `TYPE_OF_DOSE_FILE_VAL_4`.
pub const TYPE_OF_DOSE_FILE_VAL_4: i32 = 4;
/// Java public static final `VOLTAGE_DEFAULT`.
pub const VOLTAGE_DEFAULT: i32 = 200;
/// Java public static final `OPTIMAL_DOSE_SCALING_VALIDATE_MIN`.
pub const OPTIMAL_DOSE_SCALING_VALIDATE_MIN: f64 = 0.1;
/// Java public static final `OPTIMAL_DOSE_SCALING_VALIDATE_MAX`.
pub const OPTIMAL_DOSE_SCALING_VALIDATE_MAX: f64 = 10.0;
/// Java public static final `SCALING_OF_SUM_VALIDATE_MIN`.
pub const SCALING_OF_SUM_VALIDATE_MIN: i32 = 0;
/// Java public static final `MODE_TO_OUTPUT_16BIT_INT`.
pub const MODE_TO_OUTPUT_16BIT_INT: i32 = 1;
/// Java public static final `MODE_TO_OUTPUT_FLOAT`.
pub const MODE_TO_OUTPUT_FLOAT: i32 = 2;
/// Java public static final `SUM_ROTATION_AND_FLIP_DEFAULT`.
pub const SUM_ROTATION_AND_FLIP_DEFAULT: i32 = -1;
/// Java public static final `SUM_ROTATION_AND_FLIP_SPINNER_DEFAULT`.
pub const SUM_ROTATION_AND_FLIP_SPINNER_DEFAULT: i32 = 0;
/// Java public static final `SUM_ROTATION_AND_FLIP_SPINNER_MIN`.
pub const SUM_ROTATION_AND_FLIP_SPINNER_MIN: i32 = 0;
/// Java public static final `SUM_ROTATION_AND_FLIP_SPINNER_MAX`.
pub const SUM_ROTATION_AND_FLIP_SPINNER_MAX: i32 = 7;
/// Java public static final `ALIGN_AND_SUM_BINNING_SPINNER2_DEFAULT`.
pub const ALIGN_AND_SUM_BINNING_SPINNER2_DEFAULT: i32 = 1;
/// Java public static final `ALIGN_AND_SUM_BINNING_SPINNER2_MIN`.
pub const ALIGN_AND_SUM_BINNING_SPINNER2_MIN: i32 = 1;
/// Java public static final `ALIGN_AND_SUM_BINNING_SPINNER2_MAX`.
pub const ALIGN_AND_SUM_BINNING_SPINNER2_MAX: i32 = 8;
/// Java public static final `USE_GPU_VALUE`.
pub const USE_GPU_VALUE: i32 = 0;
/// Java public static final `ADJUST_AND_WRITE_MDOC_VALUE`.
pub const ADJUST_AND_WRITE_MDOC_VALUE: i32 = 1;
/// Java public static final `EER_SUPER_RES_Z_SUM_PADDING_KEY`.
pub const EER_SUPER_RES_Z_SUM_PADDING_KEY: &str = "EERSuperResZSumPadding";
/// Java private static final `EER_SUPER_RES_INDEX`.
const EER_SUPER_RES_INDEX: i32 = 0;
/// Java private static final `EER_Z_SUM_INDEX`.
const EER_Z_SUM_INDEX: i32 = 1;
/// Java private static final `EER_Z_SUM_FRAMES_DEFAULT`.
const EER_Z_SUM_FRAMES_DEFAULT: i32 = -12;
/// Java public static final `EER_Z_SUM_SETS_DEFAULT`.
pub const EER_Z_SUM_SETS_DEFAULT: i32 = 10;
/// Java private static final `EER_Z_SUM_DEFAULT`.
const EER_Z_SUM_DEFAULT: i32 = EER_Z_SUM_FRAMES_DEFAULT;
/// Java private static final `EER_PADDING_INDEX`.
const EER_PADDING_INDEX: i32 = 2;
/// Java private static final `EER_PADDING_DEFAULT`.
const EER_PADDING_DEFAULT: i32 = -1;

/// Java `public final class AlignFramesParam implements CommandParam`.
pub struct AlignFramesParam {
    /// Java `metadataFile`.
    metadata_file: StringParameter,
    /// Java `listOfInputFiles`.
    list_of_input_files: StringParameter,
    /// Java `inputFile`.
    input_file: StringParameter,
    /// Java `pathToFramesInMdoc`.
    path_to_frames_in_mdoc: StringParameter,
    /// Java `correspondingStack`.
    corresponding_stack: StringParameter,
    /// Java `tiltAngleFile`.
    tilt_angle_file: StringParameter,
    /// Java `axisRotationAngle`.
    axis_rotation_angle: ScriptParameter,
    /// Java `refAndDefectFromTitles`.
    ref_and_defect_from_titles: EtomoBoolean2,
    /// Java `gainReferenceFile`.
    gain_reference_file: StringParameter,
    /// Java `rotationAndFlip`.
    rotation_and_flip: ScriptParameter,
    /// Java `cameraDefectFile`.
    camera_defect_file: StringParameter,
    /// Java `truncateAbove`.
    truncate_above: ScriptParameter,
    /// Java `pairwiseFrames`.
    pairwise_frames: ScriptParameter,
    /// Java `targetAlignSize`.
    target_align_size: ScriptParameter,
    /// Java `testBinnings`.
    test_binnings: FortranInputString,
    /// Java `alignAndSumBinning`.
    align_and_sum_binning: FortranInputString,
    /// Java `varyFilter`.
    vary_filter: FortranInputString,
    /// Java `useHybridShifts`.
    use_hybrid_shifts: EtomoBoolean2,
    /// Java `shiftLimit`.
    shift_limit: ScriptParameter,
    /// Java `groupSize`.
    group_size: StringParameter,
    /// Java `refineAlignment`.
    refine_alignment: ScriptParameter,
    /// Java `refineWithGroupSums`.
    refine_with_group_sums: EtomoBoolean2,
    /// Java `refineRadius2`.
    refine_radius2: ScriptParameter,
    /// Java `stopIterationsAtShift`.
    stop_iterations_at_shift: ScriptParameter,
    /// Java `minForSplineSmoothing`.
    min_for_spline_smoothing: ScriptParameter,
    /// Java `startingEndingFrames`.
    starting_ending_frames: FortranInputString,
    /// Java `fixedTotalDose`.
    fixed_total_dose: ScriptParameter,
    /// Java `typeOfDoseFile`.
    type_of_dose_file: ScriptParameter,
    /// Java `doseWeightingFile`.
    dose_weighting_file: StringParameter,
    /// Java `normalizeDoseWeighting`.
    normalize_dose_weighting: EtomoBoolean2,
    /// Java `voltage`.
    voltage: ScriptParameter,
    /// Java `optimalDoseScaling`.
    optimal_dose_scaling: ScriptParameter,
    /// Java `unweightedOutputFile`.
    unweighted_output_file: StringParameter,
    /// Java `scalingOfSum`.
    scaling_of_sum: ScriptParameter,
    /// Java `modeToOutput`.
    mode_to_output: ScriptParameter,
    /// Java `sumRotationAndFlip`.
    sum_rotation_and_flip: ScriptParameter,
    /// Java `outputImageFile`.
    output_image_file: StringParameter,
    /// Java `useGPU`.
    use_gpu: ScriptParameter,
    /// Java `adjustAndWriteMdoc`.
    adjust_and_write_mdoc: ScriptParameter,
    /// Java `eerSuperResZSumPadding`.
    eer_super_res_z_sum_padding: FortranInputString,
    /// Java final `manager`.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    /// Java final `comFilename`.
    com_filename: Option<String>,
    /// Java `commandArray`, initially null and never assigned.
    #[allow(dead_code)]
    command_array: Option<Vec<String>>,
}

impl AlignFramesParam {
    /// Java `AlignFramesParam(BaseManager, String)`.
    pub fn new(manager: &'static dyn BaseManager, com_filename: Option<&str>) -> AlignFramesParam {
        let mut this = AlignFramesParam {
            metadata_file: StringParameter::new(METADATA_FILE),
            list_of_input_files: StringParameter::new(LIST_OF_INPUT_FILES),
            input_file: StringParameter::new(INPUT_FILE),
            path_to_frames_in_mdoc: StringParameter::new(PATH_TO_FRAMES_IN_MDOC),
            corresponding_stack: StringParameter::new(CORRESPONDING_STACK),
            tilt_angle_file: StringParameter::new(TILT_ANGLE_FILE),
            axis_rotation_angle: ScriptParameter::new_with_type_and_name(
                Type::Double,
                AXIS_ROTATION_ANGLE,
            ),
            ref_and_defect_from_titles: EtomoBoolean2::new_with_name(REF_AND_DEFECT_FROM_TITLES),
            gain_reference_file: StringParameter::new(GAIN_REFERENCE_FILE),
            rotation_and_flip: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                ROTATION_AND_FLIP,
            ),
            camera_defect_file: StringParameter::new(CAMERA_DEFECT_FILE),
            truncate_above: ScriptParameter::new_with_type_and_name(Type::Double, TRUNCATE_ABOVE),
            pairwise_frames: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                PAIRWISE_FRAMES,
            ),
            target_align_size: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                TARGET_ALIGN_SIZE,
            ),
            test_binnings: FortranInputString::new_with_key(
                Some(TEST_BINNINGS),
                TEST_BINNINGS_NPARAMS,
            ),
            align_and_sum_binning: FortranInputString::new_with_key(
                Some(ALIGN_AND_SUM_BINNING),
                ALIGN_AND_SUM_BINNING_NPARAMS,
            ),
            vary_filter: FortranInputString::new_with_key(Some(VARY_FILTER), VARY_FILTER_NPARAMS),
            use_hybrid_shifts: EtomoBoolean2::new_with_name(USE_HYBRID_SHIFTS),
            shift_limit: ScriptParameter::new_with_type_and_name(Type::Integer, SHIFT_LIMIT),
            group_size: StringParameter::new(GROUP_SIZE),
            refine_alignment: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                REFINE_ALIGNMENT,
            ),
            refine_with_group_sums: EtomoBoolean2::new_with_name(REFINE_WITH_GROUP_SUMS),
            refine_radius2: ScriptParameter::new_with_type_and_name(Type::Double, REFINE_RADIUS_2),
            stop_iterations_at_shift: ScriptParameter::new_with_type_and_name(
                Type::Double,
                STOP_ITERATIONS_AT_SHIFT,
            ),
            min_for_spline_smoothing: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                MIN_FOR_SPLINE_SMOOTHING,
            ),
            starting_ending_frames: FortranInputString::new_with_key(
                Some(STARTING_ENDING_FRAMES),
                STARTING_ENDING_FRAMES_NPARAMS,
            ),
            fixed_total_dose: ScriptParameter::new_with_type_and_name(
                Type::Double,
                FIXED_TOTAL_DOSE,
            ),
            type_of_dose_file: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                TYPE_OF_DOSE_FILE,
            ),
            dose_weighting_file: StringParameter::new(DOSE_WEIGHTING_FILE),
            normalize_dose_weighting: EtomoBoolean2::new_with_name(NORMALIZE_DOSE_WEIGHTING),
            voltage: ScriptParameter::new_with_type_and_name(Type::Integer, VOLTAGE),
            optimal_dose_scaling: ScriptParameter::new_with_type_and_name(
                Type::Double,
                OPTIMAL_DOSE_SCALING,
            ),
            unweighted_output_file: StringParameter::new(UNWEIGHTED_OUTPUT_FILE),
            scaling_of_sum: ScriptParameter::new_with_type_and_name(Type::Double, SCALING_OF_SUM),
            mode_to_output: ScriptParameter::new_with_type_and_name(Type::Integer, MODE_TO_OUTPUT),
            sum_rotation_and_flip: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                SUM_ROTATION_AND_FLIP,
            ),
            output_image_file: StringParameter::new(OUTPUT_IMAGE_FILE),
            use_gpu: ScriptParameter::new_with_type_and_name(Type::Integer, USE_GPU),
            adjust_and_write_mdoc: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                ADJUST_AND_WRITE_MDOC,
            ),
            eer_super_res_z_sum_padding: FortranInputString::new_with_key(
                Some(EER_SUPER_RES_Z_SUM_PADDING_KEY),
                3,
            ),
            manager,
            com_filename: com_filename.map(str::to_owned),
            command_array: None,
        };
        this.test_binnings.set_integer_type(true);
        this.align_and_sum_binning.set_integer_type(true);
        this.starting_ending_frames.set_integer_type(true);
        this.eer_super_res_z_sum_padding.set_integer_type(true);
        this
    }
    /// Java `isMetadataFile()`.
    pub fn is_metadata_file(&self) -> bool {
        !self.metadata_file.is_empty()
    }

    /// Java `isListOfInputFiles()`.
    pub fn is_list_of_input_files(&self) -> bool {
        !self.list_of_input_files.is_empty()
    }

    /// Java `isPathToFramesInMdoc()`.
    pub fn is_path_to_frames_in_mdoc(&self) -> bool {
        !self.path_to_frames_in_mdoc.is_empty()
    }

    /// Java `isCorrespondingStack()`.
    pub fn is_corresponding_stack(&self) -> bool {
        !self.corresponding_stack.is_empty()
    }

    /// Java `isTiltAngleFile()`.
    pub fn is_tilt_angle_file(&self) -> bool {
        !self.tilt_angle_file.is_empty()
    }

    /// Java `isGainReferenceFile()`.
    pub fn is_gain_reference_file(&self) -> bool {
        !self.gain_reference_file.is_empty()
    }

    /// Java `isRotationAndFlip()`.
    pub fn is_rotation_and_flip(&self) -> bool {
        self.rotation_and_flip.is()
    }

    /// Java `isCameraDefectFile()`.
    pub fn is_camera_defect_file(&self) -> bool {
        !self.camera_defect_file.is_empty()
    }

    /// Java `isTruncateAbove()`.
    pub fn is_truncate_above(&self) -> bool {
        self.truncate_above.is()
    }

    /// Java `isPairwiseFrames()`.
    pub fn is_pairwise_frames(&self) -> bool {
        self.pairwise_frames.is()
    }

    /// Java `isTargetAlignSize()`.
    pub fn is_target_align_size(&self) -> bool {
        self.target_align_size.is()
    }

    /// Java `isTestBinnings()`.
    pub fn is_test_binnings(&self) -> bool {
        !self.test_binnings.is_empty()
    }

    /// Java `isReduceBy()`.
    pub fn is_reduce_by(&self) -> bool {
        !self.align_and_sum_binning.is_empty_index(0)
    }

    /// Java `isAlignAndSumBinningVal2()`.
    pub fn is_align_and_sum_binning_val2(&self) -> bool {
        !self.align_and_sum_binning.is_empty_index(1)
    }

    /// Java `isUseHybridShifts()`.
    pub fn is_use_hybrid_shifts(&self) -> bool {
        self.use_hybrid_shifts.is()
    }

    /// Java `isShiftLimit()`.
    pub fn is_shift_limit(&self) -> bool {
        self.shift_limit.is()
    }

    /// Java `isGroupSize()`.
    pub fn is_group_size(&self) -> bool {
        !self.group_size.is_empty()
    }

    /// Java `isRefineAlignment()`.
    pub fn is_refine_alignment(&self) -> bool {
        self.refine_alignment.is()
    }

    /// Java `isRefineWithGroupSums()`.
    pub fn is_refine_with_group_sums(&self) -> bool {
        self.refine_with_group_sums.is()
    }

    /// Java `isStopIterationsAtShift()`.
    pub fn is_stop_iterations_at_shift(&self) -> bool {
        self.stop_iterations_at_shift.is()
    }

    /// Java `isMinForSplineSmoothingSet()`.
    pub fn is_min_for_spline_smoothing_set(&self) -> bool {
        !self.min_for_spline_smoothing.is_null()
    }

    /// Java `isRefineRadius2()`.
    pub fn is_refine_radius2(&self) -> bool {
        self.refine_radius2.is()
    }

    /// Java `isStartingEndingFramesFirst()`.
    pub fn is_starting_ending_frames_first(&self) -> bool {
        !self.starting_ending_frames.is_null_index(0)
    }

    /// Java `isStartingEndingFramesSecond()`.
    pub fn is_starting_ending_frames_second(&self) -> bool {
        !self.starting_ending_frames.is_null_index(1)
    }

    /// Java `isFixedTotalDose()`.
    pub fn is_fixed_total_dose(&self) -> bool {
        self.fixed_total_dose.is()
    }

    /// Java `isTypeOfDoseFile()`.
    pub fn is_type_of_dose_file(&self) -> bool {
        self.type_of_dose_file.is()
    }

    /// Java `isDoseWeightingFile()`.
    pub fn is_dose_weighting_file(&self) -> bool {
        !self.dose_weighting_file.is_empty()
    }

    /// Java `isNormalizeDoseWeighting()`.
    pub fn is_normalize_dose_weighting(&self) -> bool {
        self.normalize_dose_weighting.is()
    }

    /// Java `isVoltage()`.
    pub fn is_voltage(&self) -> bool {
        self.voltage.is()
    }

    /// Java `isOptimalDoseScaling()`.
    pub fn is_optimal_dose_scaling(&self) -> bool {
        self.optimal_dose_scaling.is()
    }

    /// Java `isUnweightedOutputFile()`.
    pub fn is_unweighted_output_file(&self) -> bool {
        !self.unweighted_output_file.is_empty()
    }

    /// Java `isScalingOfSum()`.
    pub fn is_scaling_of_sum(&self) -> bool {
        self.scaling_of_sum.is()
    }

    /// Java `isModeToOutput()`.
    pub fn is_mode_to_output(&self) -> bool {
        self.mode_to_output.is()
    }

    /// Java `isSumRotationAndFlip()`.
    pub fn is_sum_rotation_and_flip(&self) -> bool {
        self.sum_rotation_and_flip.is()
    }

    /// Java `isUseGPU()`.
    pub fn is_use_gpu(&self) -> bool {
        self.use_gpu.is()
    }

    /// Java `isOutputImageFile()`.
    pub fn is_output_image_file(&self) -> bool {
        !self.output_image_file.is_empty()
    }

    /// Java `getMetadataFile()`.
    pub fn get_metadata_file(&self) -> String {
        self.metadata_file.to_string()
    }

    /// Java `getListOfInputFiles()`.
    pub fn get_list_of_input_files(&self) -> String {
        self.list_of_input_files.to_string()
    }

    /// Java `getInputFile()`.
    pub fn get_input_file(&self) -> String {
        self.input_file.to_string()
    }

    /// Java `getPathToFramesInMdoc()`.
    pub fn get_path_to_frames_in_mdoc(&self) -> String {
        self.path_to_frames_in_mdoc.to_string()
    }

    /// Java `getCorrespondingStack()`.
    pub fn get_corresponding_stack(&self) -> String {
        self.corresponding_stack.to_string()
    }

    /// Java `getTiltAngleFile()`.
    pub fn get_tilt_angle_file(&self) -> String {
        self.tilt_angle_file.to_string()
    }

    /// Java `getAxisRotationAngle()`.
    pub fn get_axis_rotation_angle(&self) -> String {
        self.axis_rotation_angle.to_string()
    }

    /// Java `getRefAndDefectFromTitles()`.
    pub fn get_ref_and_defect_from_titles(&self) -> bool {
        self.ref_and_defect_from_titles.is()
    }

    /// Java `getGainReferenceFile()`.
    pub fn get_gain_reference_file(&self) -> String {
        self.gain_reference_file.to_string()
    }

    /// Java `getRotationAndFlip()`.
    pub fn get_rotation_and_flip(&self) -> String {
        self.rotation_and_flip.to_string()
    }

    /// Java `getCameraDefectFile()`.
    pub fn get_camera_defect_file(&self) -> String {
        self.camera_defect_file.to_string()
    }

    /// Java `getTruncateAbove()`.
    pub fn get_truncate_above(&self) -> String {
        self.truncate_above.to_string()
    }

    /// Java `getPairwiseFrames()`.
    pub fn get_pairwise_frames(&self) -> String {
        self.pairwise_frames.to_string()
    }

    /// Java `getTargetAlignSize()`.
    pub fn get_target_align_size(&self) -> String {
        self.target_align_size.to_string()
    }

    /// Java `getTestBinnings()`.
    pub fn get_test_binnings(&self) -> String {
        self.test_binnings.to_string()
    }

    /// Java `getAlignSumAndBinning()`.
    pub fn get_align_sum_and_binning(&self) -> String {
        self.align_and_sum_binning.to_string()
    }

    /// Java `getReduceByValue()`.
    pub fn get_reduce_by_value(&self) -> i32 {
        self.align_and_sum_binning.get_int(0)
    }

    /// Java `getAlignAndSumBinningVal2()`.
    pub fn get_align_and_sum_binning_val2(&self) -> i32 {
        self.align_and_sum_binning.get_int(1)
    }

    /// Java `getVaryFilterCutoff()`.
    pub fn get_vary_filter_cutoff(&self) -> String {
        self.vary_filter.to_string()
    }

    /// Java `getShiftLimit()`.
    pub fn get_shift_limit(&self) -> String {
        self.shift_limit.to_string()
    }

    /// Java `getGroupSize()`.
    pub fn get_group_size(&self) -> String {
        self.group_size.to_string()
    }

    /// Java `getRefineAlignment()`.
    pub fn get_refine_alignment(&self) -> String {
        self.refine_alignment.to_string()
    }

    /// Java `getRefineWithGroupSums()`.
    pub fn get_refine_with_group_sums(&self) -> bool {
        self.refine_with_group_sums.is()
    }

    /// Java `getRefineRadius2()`.
    pub fn get_refine_radius2(&self) -> String {
        self.refine_radius2.to_string()
    }

    /// Java `getStopIterationsAtShift()`.
    pub fn get_stop_iterations_at_shift(&self) -> String {
        self.stop_iterations_at_shift.to_string()
    }

    /// Java `getMinForSplineSmoothing()`.
    pub fn get_min_for_spline_smoothing(&self) -> String {
        self.min_for_spline_smoothing.to_string()
    }

    /// Java `getStartingEndingFrames()`.
    pub fn get_starting_ending_frames(&self) -> String {
        self.starting_ending_frames.to_string()
    }

    /// Java `getStartingEndingFramesFirst()`.
    pub fn get_starting_ending_frames_first(&self) -> i32 {
        self.starting_ending_frames.get_int(0)
    }

    /// Java `getStartingEndingFramesSecond()`.
    pub fn get_starting_ending_frames_second(&self) -> i32 {
        self.starting_ending_frames.get_int(1)
    }

    /// Java `getFixedTotalDose()`.
    pub fn get_fixed_total_dose(&self) -> String {
        self.fixed_total_dose.to_string()
    }

    /// Java `getTypeOfDoseFile()`.
    pub fn get_type_of_dose_file(&self) -> String {
        self.type_of_dose_file.to_string()
    }

    /// Java `getNormalizeDoseWeighting()`.
    pub fn get_normalize_dose_weighting(&self) -> bool {
        self.normalize_dose_weighting.is()
    }

    /// Java `getVoltage()`.
    pub fn get_voltage(&self) -> String {
        self.voltage.to_string()
    }

    /// Java `getOptimalDoseScaling()`.
    pub fn get_optimal_dose_scaling(&self) -> String {
        self.optimal_dose_scaling.to_string()
    }

    /// Java `getUnweightedOutputFile()`.
    pub fn get_unweighted_output_file(&self) -> String {
        self.unweighted_output_file.to_string()
    }

    /// Java `getScalingOfSum()`.
    pub fn get_scaling_of_sum(&self) -> String {
        self.scaling_of_sum.to_string()
    }

    /// Java `getModeToOutput()`.
    pub fn get_mode_to_output(&self) -> String {
        self.mode_to_output.to_string()
    }

    /// Java `getSumRotationAndFlip()`.
    pub fn get_sum_rotation_and_flip(&self) -> String {
        self.sum_rotation_and_flip.to_string()
    }

    /// Java `getOutputImageFile()`.
    pub fn get_output_image_file(&self) -> String {
        self.output_image_file.to_string()
    }

    /// Java `getUseGPU()`.
    pub fn get_use_gpu(&self) -> String {
        self.use_gpu.to_string()
    }

    /// Java `setMetadataFile(String input)`.
    pub fn set_metadata_file(&mut self, input: Option<&str>) {
        self.metadata_file.set(input);
    }

    /// Java `setListOfInputFiles(String input)`.
    pub fn set_list_of_input_files(&mut self, input: Option<&str>) {
        self.list_of_input_files.set(input);
    }

    /// Java `setInputFile(String input)`.
    pub fn set_input_file(&mut self, input: Option<&str>) {
        self.input_file.set(input);
    }

    /// Java `setPathToFramesInMdoc(String input)`.
    pub fn set_path_to_frames_in_mdoc(&mut self, input: Option<&str>) {
        self.path_to_frames_in_mdoc.set(input);
    }

    /// Java `setCorrespondingStack(String input)`.
    pub fn set_corresponding_stack(&mut self, input: Option<&str>) {
        self.corresponding_stack.set(input);
    }

    /// Java `setTiltAngleFile(String input)`.
    pub fn set_tilt_angle_file(&mut self, input: Option<&str>) {
        self.tilt_angle_file.set(input);
    }

    /// Java `setAxisRotationAngle(String input)`.
    pub fn set_axis_rotation_angle(&mut self, input: Option<&str>) {
        self.axis_rotation_angle.set_string(input);
    }

    /// Java `setRefAndDefectFromTitles(boolean input)`.
    pub fn set_ref_and_defect_from_titles(&mut self, input: bool) {
        self.ref_and_defect_from_titles.set_boolean(input);
    }

    /// Java `setGainReferenceFile(String input)`.
    pub fn set_gain_reference_file(&mut self, input: Option<&str>) {
        self.gain_reference_file.set(input);
    }

    /// Java `setRotationAndFlip(String input)`.
    pub fn set_rotation_and_flip(&mut self, input: Option<&str>) {
        self.rotation_and_flip.set_string(input);
    }

    /// Java `resetRotationAndFlip()`.
    pub fn reset_rotation_and_flip(&mut self) {
        self.rotation_and_flip.reset();
    }

    /// Java `setCameraDefectFile(String input)`.
    pub fn set_camera_defect_file(&mut self, input: Option<&str>) {
        self.camera_defect_file.set(input);
    }

    /// Java `setTruncateAbove(String input)`.
    pub fn set_truncate_above(&mut self, input: Option<&str>) {
        self.truncate_above.set_string(input);
    }

    /// Java `setPairwiseFrames(String input)`.
    pub fn set_pairwise_frames(&mut self, input: Option<&str>) {
        self.pairwise_frames.set_string(input);
    }

    /// Java `setTargetAlignSize(String input)`.
    pub fn set_target_align_size(&mut self, input: Option<&str>) {
        self.target_align_size.set_string(input);
    }

    /// Java `setTestBinnings(String input)`.
    pub fn set_test_binnings(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.test_binnings.validate_and_set(input)
    }

    /// Java `setAlignSumAndBinning(String input1, String input2)`.
    pub fn set_align_sum_and_binning(
        &mut self,
        input1: Option<&str>,
        input2: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.align_and_sum_binning
            .validate_and_set_two(input1, input2)
    }

    /// Java `setVaryFilter(String input)`.
    pub fn set_vary_filter(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.vary_filter.validate_and_set(input)
    }

    /// Java `setUseHybridShifts(boolean input)`.
    pub fn set_use_hybrid_shifts(&mut self, input: bool) {
        self.use_hybrid_shifts.set_boolean(input);
    }

    /// Java `setShiftLimit(String input)`.
    pub fn set_shift_limit(&mut self, input: Option<&str>) {
        self.shift_limit.set_string(input);
    }

    /// Java `setGroupSize(String input)`.
    pub fn set_group_size(&mut self, input: Option<&str>) {
        self.group_size.set(input);
    }

    /// Java `setRefineAlignment(String input)`.
    pub fn set_refine_alignment(&mut self, input: Option<&str>) {
        self.refine_alignment.set_string(input);
    }

    /// Java `setRefineWithGroupSums(boolean input)`.
    pub fn set_refine_with_group_sums(&mut self, input: bool) {
        self.refine_with_group_sums.set_boolean(input);
    }

    /// Java `setRefineRadius2(String input)`.
    pub fn set_refine_radius2(&mut self, input: Option<&str>) {
        self.refine_radius2.set_string(input);
    }

    /// Java `setStopIterationsAtShift(String input)`.
    pub fn set_stop_iterations_at_shift(&mut self, input: Option<&str>) {
        self.stop_iterations_at_shift.set_string(input);
    }

    /// Java `setMinForSplineSmoothing(String input)`.
    pub fn set_min_for_spline_smoothing(&mut self, input: Option<&str>) {
        self.min_for_spline_smoothing.set_string(input);
    }

    /// Java `setStartingEndingFrames(String input1, String input2)`.
    pub fn set_starting_ending_frames(
        &mut self,
        input1: Option<&str>,
        input2: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.starting_ending_frames
            .validate_and_set_two(input1, input2)
    }

    /// Java `setFixedTotalDose(String input)`.
    pub fn set_fixed_total_dose(&mut self, input: Option<&str>) {
        self.fixed_total_dose.set_string(input);
    }

    /// Java `setTypeOfDoseFile(String input)`.
    pub fn set_type_of_dose_file(&mut self, input: Option<&str>) {
        self.type_of_dose_file.set_string(input);
    }

    /// Java `setDoseWeightingFile(String input)`.
    pub fn set_dose_weighting_file(&mut self, input: Option<&str>) {
        self.dose_weighting_file.set(input);
    }

    /// Java `setNormalizeDoseWeighting(boolean input)`.
    pub fn set_normalize_dose_weighting(&mut self, input: bool) {
        self.normalize_dose_weighting.set_boolean(input);
    }

    /// Java `setVoltage(String input)`.
    pub fn set_voltage(&mut self, input: Option<&str>) {
        self.voltage.set_string(input);
    }

    /// Java `setOptimalDoseScaling(String input)`.  Declares `throws FortranInputSyntaxException` and never
    /// throws it.
    pub fn set_optimal_dose_scaling(
        &mut self,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.optimal_dose_scaling.set_string(input);
        Ok(())
    }

    /// Java `setUnweightedOutputFile(String input)`.
    pub fn set_unweighted_output_file(&mut self, input: Option<&str>) {
        self.unweighted_output_file.set(input);
    }

    /// Java `setScalingOfSum(String input)`.
    pub fn set_scaling_of_sum(&mut self, input: Option<&str>) {
        self.scaling_of_sum.set_string(input);
    }

    /// Java `setModeToOutput(String input)`.
    pub fn set_mode_to_output(&mut self, input: Option<&str>) {
        self.mode_to_output.set_string(input);
    }

    /// Java `setSumRotationAndFlip(String input)`.
    pub fn set_sum_rotation_and_flip(&mut self, input: Option<&str>) {
        self.sum_rotation_and_flip.set_string(input);
    }

    /// Java `setOutputImageFile(String input)`.
    pub fn set_output_image_file(&mut self, input: Option<&str>) {
        self.output_image_file.set(input);
    }

    /// Java `setUseGPU(String input)`.
    pub fn set_use_gpu(&mut self, input: Option<&str>) {
        self.use_gpu.set_string(input);
    }

    /// Java `setAdjustAndWriteMdoc(String input)`.
    pub fn set_adjust_and_write_mdoc(&mut self, input: Option<&str>) {
        self.adjust_and_write_mdoc.set_string(input);
    }
    /// Java `getEERSuperRes()`; the `Integer` it returns is never null.
    pub fn get_eer_super_res(&self) -> i32 {
        if !self
            .eer_super_res_z_sum_padding
            .is_null_index(EER_SUPER_RES_INDEX)
        {
            self.eer_super_res_z_sum_padding
                .get_int(EER_SUPER_RES_INDEX)
        } else {
            EERSuperRes::DEFAULT.get_value().get_int()
        }
    }

    /// Java `setEERSuperRes(ConstEtomoNumber)`.
    pub fn set_eer_super_res(&mut self, value: Option<&ConstEtomoNumber>) {
        if let Some(value) = value {
            self.eer_super_res_z_sum_padding
                .set_index_const_etomo_number(EER_SUPER_RES_INDEX, value);
        } else {
            self.eer_super_res_z_sum_padding
                .set_index_const_etomo_number(
                    EER_SUPER_RES_INDEX,
                    &EERSuperRes::DEFAULT.get_value(),
                );
        }
    }

    /// Java private static `convertEERZSumFrames(Integer)`.  Converts dialog value
    /// from/to parameter value.  Parameter value is negative so it can be
    /// distinguished from Sets.
    fn convert_eer_z_sum_frames(value: Option<i32>) -> Option<i32> {
        if let Some(value) = value {
            return Some(value.wrapping_mul(-1));
        }
        None
    }

    /// Java static `getEERZSumFramesDefault()`.
    pub fn get_eer_z_sum_frames_default() -> Option<i32> {
        AlignFramesParam::convert_eer_z_sum_frames(Some(EER_Z_SUM_FRAMES_DEFAULT))
    }

    /// Java `isEERZSumFramesSet()`.  EER Z Sum can be either Frames or Sets.  Frames
    /// are negative in the comfile, and also the default.
    pub fn is_eer_z_sum_frames_set(&self) -> bool {
        if !self
            .eer_super_res_z_sum_padding
            .is_null_index(EER_Z_SUM_INDEX)
        {
            let i = self.eer_super_res_z_sum_padding.get_int(EER_Z_SUM_INDEX);
            i <= 0
        } else {
            true
        }
    }

    /// Java `getEERZSumFrames()`.  EER Z Sum can be either Frames or Sets.  Frames are
    /// negative in the comfile.  The Java `int` return unboxes an `Integer` that is
    /// never null here.
    pub fn get_eer_z_sum_frames(&self) -> i32 {
        if !self
            .eer_super_res_z_sum_padding
            .is_null_index(EER_Z_SUM_INDEX)
            && self.is_eer_z_sum_frames_set()
        {
            AlignFramesParam::convert_eer_z_sum_frames(Some(
                self.eer_super_res_z_sum_padding.get_int(EER_Z_SUM_INDEX),
            ))
            .unwrap_or_default()
        } else {
            AlignFramesParam::get_eer_z_sum_frames_default().unwrap_or_default()
        }
    }

    /// Java `getEERZSumSets()`.  EER Z Sum can be either Frames or Sets.  Frames are
    /// negative in the comfile.
    pub fn get_eer_z_sum_sets(&self) -> i32 {
        if !self
            .eer_super_res_z_sum_padding
            .is_null_index(EER_Z_SUM_INDEX)
            && !self.is_eer_z_sum_frames_set()
        {
            self.eer_super_res_z_sum_padding.get_int(EER_Z_SUM_INDEX)
        } else {
            EER_Z_SUM_SETS_DEFAULT
        }
    }

    /// Java `setEERZSumFrames(Integer)`.
    pub fn set_eer_z_sum_frames(&mut self, value: Option<i32>) {
        if value.is_some() {
            // Use the negation of the displayed value
            // (`set(int, Integer)` unboxes to `set(int, double)`).
            let converted = AlignFramesParam::convert_eer_z_sum_frames(value).unwrap_or_default();
            self.eer_super_res_z_sum_padding
                .set_index_double(EER_Z_SUM_INDEX, converted as f64);
        } else {
            self.eer_super_res_z_sum_padding
                .set_index_double(EER_Z_SUM_INDEX, EER_Z_SUM_FRAMES_DEFAULT as f64);
        }
    }

    /// Java `setEERZSumSets(Integer)`.
    pub fn set_eer_z_sum_sets(&mut self, value: Option<i32>) {
        if let Some(value) = value {
            self.eer_super_res_z_sum_padding
                .set_index_double(EER_Z_SUM_INDEX, value as f64);
        } else {
            self.eer_super_res_z_sum_padding
                .set_index_double(EER_Z_SUM_INDEX, EER_Z_SUM_SETS_DEFAULT as f64);
        }
    }

    /// Java `getAxisID()`: null.
    pub fn get_axis_id(&self) -> Option<AxisID> {
        None
    }

    /// Java `getCommandMode()`: null.
    pub fn get_command_mode(&self) -> Option<&dyn super::command_mode::CommandMode> {
        None
    }

    /// Java `getProcessName()`.
    pub fn get_process_name(&self) -> ProcessName {
        PROCESS_NAME
    }

    /// Java `getCommand()`.
    pub fn get_command(&self) -> Option<String> {
        self.com_filename.clone()
    }

    /// Java `getCommandName()`.
    pub fn get_command_name(&self) -> Option<&'static str> {
        PROCESS_NAME.get_text()
    }

    /// Java `getCommandArray()`.
    pub fn get_command_array(&self) -> Vec<Option<String>> {
        vec![self.com_filename.clone()]
    }

    /// Java `getCommandLine()`.
    pub fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }
}

impl CommandParam for AlignFramesParam {
    /// Java `parseComScriptCommand(ComScriptCommand)`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // `scriptCommand.useKeywordValue()` mutates the command; the trait hands the
        // command in by shared reference, so it is applied to a copy that the parse
        // reads from.
        let mut script_command = ComScriptCommand::new_from(script_command);
        script_command.use_keyword_value();
        let script_command = &script_command;
        self.initialize_defaults();
        self.metadata_file.parse(script_command)?;
        self.list_of_input_files.parse(script_command)?;
        self.input_file.parse(script_command)?;
        self.path_to_frames_in_mdoc.parse(script_command)?;
        self.corresponding_stack.parse(script_command)?;
        self.tilt_angle_file.parse(script_command)?;
        self.axis_rotation_angle.parse(script_command)?;
        self.ref_and_defect_from_titles.parse(script_command)?;
        self.gain_reference_file.parse(script_command)?;
        self.rotation_and_flip.parse(script_command)?;
        self.camera_defect_file.parse(script_command)?;
        self.truncate_above.parse(script_command)?;
        self.pairwise_frames.parse(script_command)?;
        self.target_align_size.parse(script_command)?;
        self.test_binnings
            .validate_and_set_com_script(script_command)?;
        self.align_and_sum_binning
            .validate_and_set_com_script(script_command)?;
        self.vary_filter
            .validate_and_set_com_script(script_command)?;
        self.use_hybrid_shifts.parse(script_command)?;
        self.shift_limit.parse(script_command)?;
        self.group_size.parse(script_command)?;
        self.refine_alignment.parse(script_command)?;
        self.refine_with_group_sums.parse(script_command)?;
        self.refine_radius2.parse(script_command)?;
        self.stop_iterations_at_shift.parse(script_command)?;
        self.min_for_spline_smoothing.parse(script_command)?;
        self.starting_ending_frames
            .validate_and_set_com_script(script_command)?;
        self.type_of_dose_file.parse(script_command)?;
        self.dose_weighting_file.parse(script_command)?;
        self.fixed_total_dose.parse(script_command)?;
        self.normalize_dose_weighting.parse(script_command)?;
        self.voltage.parse(script_command)?;
        self.optimal_dose_scaling.parse(script_command)?;
        self.unweighted_output_file.parse(script_command)?;
        self.scaling_of_sum.parse(script_command)?;
        self.mode_to_output.parse(script_command)?;
        self.sum_rotation_and_flip.parse(script_command)?;
        self.output_image_file.parse(script_command)?;
        self.use_gpu.parse(script_command)?;
        self.adjust_and_write_mdoc.parse(script_command)?;
        self.eer_super_res_z_sum_padding
            .validate_and_set_com_script(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand(ComScriptCommand)`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.metadata_file.update_com_script(script_command);
        self.list_of_input_files.update_com_script(script_command);
        self.input_file.update_com_script(script_command);
        self.path_to_frames_in_mdoc
            .update_com_script(script_command);
        self.corresponding_stack.update_com_script(script_command);
        self.tilt_angle_file.update_com_script(script_command);
        self.axis_rotation_angle.update_com_script(script_command);
        self.ref_and_defect_from_titles
            .update_com_script(script_command);
        self.gain_reference_file.update_com_script(script_command);
        self.rotation_and_flip.update_com_script(script_command);
        self.camera_defect_file.update_com_script(script_command);
        self.truncate_above.update_com_script(script_command);
        self.pairwise_frames.update_com_script(script_command);
        self.target_align_size.update_com_script(script_command);
        self.test_binnings
            .update_script_parameter_format(script_command, true, true);
        self.align_and_sum_binning
            .update_script_parameter_format(script_command, true, true);
        self.vary_filter
            .update_script_parameter_format(script_command, true, true);
        self.use_hybrid_shifts.update_com_script(script_command);
        self.shift_limit.update_com_script(script_command);
        self.group_size.update_com_script(script_command);
        self.refine_alignment.update_com_script(script_command);
        self.refine_with_group_sums
            .update_com_script(script_command);
        self.refine_radius2.update_com_script(script_command);
        self.stop_iterations_at_shift
            .update_com_script(script_command);
        self.min_for_spline_smoothing
            .update_com_script(script_command);
        self.starting_ending_frames
            .update_script_parameter_format(script_command, true, true);
        self.type_of_dose_file.update_com_script(script_command);
        self.dose_weighting_file.update_com_script(script_command);
        self.fixed_total_dose.update_com_script(script_command);
        self.normalize_dose_weighting
            .update_com_script(script_command);
        self.voltage.update_com_script(script_command);
        self.optimal_dose_scaling.update_com_script(script_command);
        self.unweighted_output_file
            .update_com_script(script_command);
        self.scaling_of_sum.update_com_script(script_command);
        self.mode_to_output.update_com_script(script_command);
        self.sum_rotation_and_flip.update_com_script(script_command);
        self.output_image_file.update_com_script(script_command);
        self.use_gpu.update_com_script(script_command);
        self.adjust_and_write_mdoc.update_com_script(script_command);
        self.eer_super_res_z_sum_padding
            .update_script_parameter_format(script_command, true, true);
        Ok(())
    }

    /// Java `initializeDefaults()`.
    fn initialize_defaults(&mut self) {
        self.metadata_file.reset();
        self.list_of_input_files.reset();
        self.input_file.reset();
        self.path_to_frames_in_mdoc.reset();
        self.corresponding_stack.reset();
        self.tilt_angle_file.reset();
        self.axis_rotation_angle.reset();
        self.ref_and_defect_from_titles.reset();
        self.gain_reference_file.reset();
        self.rotation_and_flip.reset();
        self.camera_defect_file.reset();
        self.truncate_above.reset();
        self.pairwise_frames.reset();
        self.target_align_size.reset();
        self.test_binnings.reset();
        self.align_and_sum_binning.reset();
        self.vary_filter.reset();
        self.use_hybrid_shifts.reset();
        self.shift_limit.reset();
        self.group_size.reset();
        self.refine_alignment.reset();
        self.refine_with_group_sums.reset();
        self.refine_radius2.reset();
        self.stop_iterations_at_shift.reset();
        self.min_for_spline_smoothing.reset();
        self.starting_ending_frames.set_default();
        self.type_of_dose_file.reset();
        self.dose_weighting_file.reset();
        self.fixed_total_dose.reset();
        self.normalize_dose_weighting.reset();
        self.voltage.reset();
        self.optimal_dose_scaling.reset();
        self.unweighted_output_file.reset();
        self.scaling_of_sum.reset();
        self.mode_to_output.reset();
        self.sum_rotation_and_flip.reset();
        self.output_image_file.reset();
        self.use_gpu.reset();
        self.adjust_and_write_mdoc.reset();
        self.eer_super_res_z_sum_padding.reset();
        self.eer_super_res_z_sum_padding
            .set_index_const_etomo_number(EER_SUPER_RES_INDEX, &EERSuperRes::DEFAULT.get_value());
        self.eer_super_res_z_sum_padding
            .set_index_double(EER_Z_SUM_INDEX, EER_Z_SUM_DEFAULT as f64);
        self.eer_super_res_z_sum_padding
            .set_index_double(EER_PADDING_INDEX, EER_PADDING_DEFAULT as f64);
    }
}
