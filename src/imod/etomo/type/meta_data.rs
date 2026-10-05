//! `IMOD/Etomo/src/etomo/type/MetaData.java`.
//!
//! The reconstruction dataset's state, stored in the `.edf` file under the `Setup`
//! group.  "Strings and keys must not change without provisions for backwards
//! compatibility."
//!
//! **Representation.**  `MetaData extends BaseMetaData implements ConstMetaData`.  The
//! superclass is the `BaseMetaData` trait plus the embedded `BaseMetaDataBase` field
//! block (`base_meta_data.rs`); `ConstMetaData` is the trait in `const_meta_data.rs`.
//! Every method body is an inherent method here, and the two trait impls forward to
//! them, so a caller holding a `&MetaData` needs no trait import and a caller holding a
//! `&dyn ConstMetaData` reaches the same code.
//!
//! **Threads.**  Java hands the one `MetaData` to the event dispatch thread, to process
//! threads and to monitors, and mutates it through those shared references.  Each
//! mutable field therefore carries its own lock, as `BaseMetaDataBase` does, and every
//! method takes `&self`.  A method never holds two guards of the same field at once;
//! where the source reads one field while writing another, the value read is copied to
//! a local first.
//!
//! **Returned objects.**  Java getters that return a field object (`ConstEtomoNumber`,
//! `FortranInputString`, `IntKeyList`, `TiltAngleSpec`) hand out the live object.  Here
//! they return a clone of the field's concrete type (so an `EtomoBoolean2` keeps its
//! `is()`/`toString()` overrides); a caller that mutated the returned Java object must
//! use the corresponding setter instead.
//!
//! **Parameters.**  A Java `String` parameter is `Option<&str>` (Java null is `None`);
//! a Java `File` parameter is its path; an `AxisID` parameter is passed by value
//! (every source caller passes one of the three singletons).  Overloads carry a
//! suffix naming the distinguishing parameter type.
//!
//! Java `Properties` is the deterministic `BTreeMap<String, String>` used throughout
//! the translation (`etomo/storage/storable.rs`).
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use regex::Regex;

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use super::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_double_to_string, java_lang_double_value_of,
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use super::const_meta_data::ConstMetaData;
use super::const_string_property::ConstStringProperty;
use super::data_file_type::DataFileType;
use super::data_source::DataSource;
use super::dialog_type::DialogType;
use super::enumerated_type::EnumeratedType;
use super::erase_gold::EraseGold;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::extension::{EXTENSION_DIVIDER, Extension};
use super::image_file_meta_data::ImageFileMetaData;
use super::image_filename_style::ImageFilenameStyle;
use super::int_key_list::IntKeyList;
use super::panel_id::PanelId;
use super::process_name::ProcessName;
use super::sample_type::SampleType;
use super::string_property::StringProperty;
use super::tilt_angle_spec::TiltAngleSpec;
use super::view_type::ViewType;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::combine_params::CombineParams;
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::squeezevol_param::SqueezevolParam;
use crate::imod::etomo::comscript::transferfid_param::TransferfidParam;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::tracking_method;
use crate::imod::etomo::storage::storable;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::ui::swing::trimvol_display::TrimvolDisplay;
use crate::imod::etomo::util::utilities::{
    self, java_io_file_can_read, java_io_file_can_write, java_io_file_get_absolute_path,
    java_io_file_get_name, java_io_file_get_parent, java_io_file_new, java_lang_string_split,
};

/// Java `CORRECTED_RAW_IMAGE_STACK_EXT_KEY_VERSION`.
const CORRECTED_RAW_IMAGE_STACK_EXT_KEY_VERSION: &str = "4.10.43";

/// Java `latestRevisionNumber`.
const LATEST_REVISION_NUMBER: &str = "1.12";

/// Java `newTomogramTitle`.
const NEW_TOMOGRAM_TITLE: &str = "Setup Tomogram";

/// Java `TOMO_GEN_A_TILT_PARALLEL_GROUP`.
static TOMO_GEN_A_TILT_PARALLEL_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}.Tilt.Parallel",
        DialogType::TomogramGeneration.get_storable_name(),
        AxisID::First.get_extension().to_uppercase()
    )
});

/// Java `TOMO_GEN_B_TILT_PARALLEL_GROUP`.
static TOMO_GEN_B_TILT_PARALLEL_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}.Tilt.Parallel",
        DialogType::TomogramGeneration.get_storable_name(),
        AxisID::Second.get_extension().to_uppercase()
    )
});

/// Java `FINAL_STACK_A_CTF_CORRECTION_PARALLEL_GROUP`.
static FINAL_STACK_A_CTF_CORRECTION_PARALLEL_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}.CtfCorrection.Parallel",
        DialogType::FinalAlignedStack.get_storable_name(),
        AxisID::First.get_extension().to_uppercase()
    )
});

/// Java `FINAL_STACK_B_CTF_CORRECTION_PARALLEL_GROUP`.
static FINAL_STACK_B_CTF_CORRECTION_PARALLEL_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}.CtfCorrection.Parallel",
        DialogType::FinalAlignedStack.get_storable_name(),
        AxisID::Second.get_extension().to_uppercase()
    )
});

/// Java `COMBINE_VOLCOMBINE_PARALLEL_GROUP`.
static COMBINE_VOLCOMBINE_PARALLEL_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Volcombine.Parallel",
        DialogType::TomogramCombination.get_storable_name()
    )
});

/// Java `B_STACK_PROCESSED_GROUP`.
const B_STACK_PROCESSED_GROUP: &str = "BStackProcessed";

/// Java `DEFAULT_SAMPLE_THICKNESS`.
const DEFAULT_SAMPLE_THICKNESS: i32 = 300;

/// Java `FIDUCIALESS_KEY`.
const FIDUCIALESS_KEY: &str = "Fiducialess";

/// Java `THICKNESS_KEY`.
const THICKNESS_KEY: &str = "THICKNESS";

/// Java `FINAL_STACK_BINNING_A_BACKWARD_COMPATABILITY_1_8`.
const FINAL_STACK_BINNING_A_BACKWARD_COMPATABILITY_1_8: &str = "TomoGenBinningA";

/// Java `FINAL_STACK_BINNING_B_BACKWARD_COMPATABILITY_1_8`.
const FINAL_STACK_BINNING_B_BACKWARD_COMPATABILITY_1_8: &str = "TomoGenBinningB";

/// Java `TWO_DIR_DEFAULT`.
const TWO_DIR_DEFAULT: f64 = 0.0;

/// Java `DOSE_SYM_DEFAULT`.
const DOSE_SYM_DEFAULT: f64 = 0.0;

/// Java `FIRST_AXIS_KEY`.
const FIRST_AXIS_KEY: &str = "A";

/// Java `SECOND_AXIS_KEY`.
const SECOND_AXIS_KEY: &str = "B";

/// Java `TRACK_KEY`.
const TRACK_KEY: &str = "Track";

/// Java `FINE_KEY`.
const FINE_KEY: &str = "Fine";

/// Java `POS_KEY`.
const POS_KEY: &str = "Pos";

/// Java `STACK_KEY`.
const STACK_KEY: &str = "Stack";

/// Java `GEN_KEY`.
const GEN_KEY: &str = "Gen";

/// Java `POST_KEY`.
const POST_KEY: &str = "Post";

/// Java `COARSE_KEY`.
const COARSE_KEY: &str = "Coarse";

/// Java `NEWSTACK_OR_BLENDMONT_KEY`.
const NEWSTACK_OR_BLENDMONT_KEY: &str = "NewstackOrBlendmont";

/// Java `ERASE_GOLD_KEY`.
const ERASE_GOLD_KEY: &str = "EraseGold";

/// Java `FLATTEN_KEY`.
const FLATTEN_KEY: &str = "Flatten";

/// Java `FLATTEN_WARP_KEY`.
const FLATTEN_WARP_KEY: &str = "FlattenWarp";

/// Java `RAPTOR_KEY`.
const RAPTOR_KEY: &str = "Raptor";

/// Java `TRIM_VOL_KEY`.
const TRIM_VOL_KEY: &str = "TrimVol";

/// Java `SQUEEZE_VOL_KEY`.
const SQUEEZE_VOL_KEY: &str = "SqueezeVol";

/// Java `REDUCE_FILT_VOL_KEY`.
const REDUCE_FILT_VOL_KEY: &str = "ReduceFiltVol";

/// Java `SIRT_KEY`.
const SIRT_KEY: &str = "Sirt";

/// Java `SUBTOMO_KEY`.
const SUBTOMO_KEY: &str = "Subtomo";

/// Java `ALT_TOMO_SETUP_KEY`.
const ALT_TOMO_SETUP_KEY: &str = "AltTomoSetup";

/// Java `BINNING_KEY`.
const BINNING_KEY: &str = "Binning";

/// Java `CONTOURS_ON_ONE_SURFACE_KEY`.
const CONTOURS_ON_ONE_SURFACE_KEY: &str = "ContoursOnOneSurface";

/// Java `DIAM_KEY`.
const DIAM_KEY: &str = "Diam";

/// Java `INPUT_KEY`.
const INPUT_KEY: &str = "Input";

/// Java `MARK_KEY`.
const MARK_KEY: &str = "Mark";

/// Java `MODEL_USE_FID_KEY`.
const MODEL_USE_FID_KEY: &str = "ModelUseFid";

/// Java `RAW_STACK_KEY`.
const RAW_STACK_KEY: &str = "RawStack";

/// Java `SIZE_TO_OUTPUT_IN_X_AND_Y_KEY`.
const SIZE_TO_OUTPUT_IN_X_AND_Y_KEY: &str = "SizeToOutputInXandY";

/// Java `SPACING_IN_KEY`.
const SPACING_IN_KEY: &str = "SpacingIn";

/// Java `USE_KEY`.
const USE_KEY: &str = "Use";

/// Java `X_KEY`.
const X_KEY: &str = "X";

/// Java `Y_KEY`.
const Y_KEY: &str = "Y";

/// Java `BATCHRUNTOMO_KEY`.
pub const BATCHRUNTOMO_KEY: &str = "batchruntomo";

/// Java `COMBINE_KEY`.
pub const COMBINE_KEY: &str = "Combine";

/// Java `TILT_3D_FIND_A_TILT_PARALLEL_KEY`.
static TILT_3D_FIND_A_TILT_PARALLEL_KEY: LazyLock<String> =
    LazyLock::new(|| format!("{}.A.Tilt.Parallel", STACK_KEY));

/// Java `TILT_3D_FIND_B_TILT_PARALLEL_KEY`.
static TILT_3D_FIND_B_TILT_PARALLEL_KEY: LazyLock<String> =
    LazyLock::new(|| format!("{}.B.Tilt.Parallel", STACK_KEY));

/// Java `ERASE_GOLD_MODEL_USE_FID_DEFAULT`.
const ERASE_GOLD_MODEL_USE_FID_DEFAULT: bool = false;

/// Java `EXTRA_THICKNESS_CRYO_DEFAULT`.
const EXTRA_THICKNESS_CRYO_DEFAULT: f64 = 25.0;

/// Java `CTF_SCALE_BY_CTF_POWER_DEFAULT`.
const CTF_SCALE_BY_CTF_POWER_DEFAULT: f64 = 0.5;

/// Java `BATCH_RUN_TOMO_LOG_FILE_AXIS_ID_KEY`.
const BATCH_RUN_TOMO_LOG_FILE_AXIS_ID_KEY: &str = "BatchRunTomoLog.Read.AxisID";

/// Java `INCORRECT_RAW_IMAGE_STACK_EXT_KEY`.
const INCORRECT_RAW_IMAGE_STACK_EXT_KEY: &str = "Setup.ImageStackExt";

/// Java `INCORRECT_ORIG_RAW_IMAGE_STACK_EXT_KEY`.
const INCORRECT_ORIG_RAW_IMAGE_STACK_EXT_KEY: &str = "Setup.OrigImageStackExt";

/// Java `ConstTiltalignParam.TARGET_PATCH_SIZE_X_AND_Y_KEY`
/// (ConstTiltalignParam.java:55); `TiltalignParam` has no Rust module.
const TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_KEY: &str = "TargetPatchSizeXandY";

/// Java `ConstTiltalignParam.NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY`
/// (ConstTiltalignParam.java:53-54).
const TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY: &str = "NumberOfLocalPatchesXandY";

/// Java `ConstTiltalignParam.TARGET_PATCH_SIZE_X_AND_Y_DEFAULT`
/// (ConstTiltalignParam.java:137).
const TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_DEFAULT: &str = "700,700";

/// Java `ConstTiltalignParam.NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT`
/// (ConstTiltalignParam.java:138).
const TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT: &str = "5,5";

/// Java `SubtomoSetupParam.EXTENT_OF_ZLEVELS_IN_NM_DEFAULT`
/// (SubtomoSetupParam.java:53), which is `Ctf3dSetupParam.SLAB_THICKNESS_IN_NM_DEFAULT`
/// (Ctf3dSetupParam.java:43); neither param class has a Rust module.
const SUBTOMO_SETUP_PARAM_EXTENT_OF_ZLEVELS_IN_NM_DEFAULT: i32 = 15;

/// Java `String.split("\\s+")`'s pattern.  Java's `\s` is `[ \t\n\x0B\f\r]`, narrower
/// than the `regex` crate's Unicode `\s`.
static WHITESPACE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\x0B\x0C\r]+").unwrap());

/// Java `MetaData`.
pub struct MetaData {
    /// Java superclass `BaseMetaData` state.
    base: BaseMetaDataBase,
    /// Java `private final ApplicationManager manager;`
    manager: Option<&'static ApplicationManager>,
    /// Java `private String datasetName = "";`
    dataset_name: Mutex<String>,
    /// Java `private String backupDirectory = "";`
    backup_directory: Mutex<String>,
    /// Java `private String distortionFile = null;`
    distortion_file: Mutex<Option<String>>,
    /// Java `private String magGradientFile = null;`
    mag_gradient_file: Mutex<Option<String>>,
    /// Java `private DataSource dataSource = DataSource.CCD;`
    data_source: Mutex<DataSource>,
    /// Java `private ViewType viewType = ViewType.SINGLE_VIEW;`
    view_type: Mutex<ViewType>,
    /// Java `private double pixelSize = Double.NaN;`
    pixel_size: Mutex<f64>,
    /// Java `private boolean useLocalAlignmentsA = true;`
    use_local_alignments_a: Mutex<bool>,
    /// Java `private boolean useLocalAlignmentsB = true;`
    use_local_alignments_b: Mutex<bool>,
    /// Java `private double fiducialDiameter = Double.NaN;`
    fiducial_diameter: Mutex<f64>,
    /// Java `private final EtomoNumber halfFloatModeOutput = new EtomoNumber("HalfFloatModeOutput");`
    half_float_mode_output: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber imageRotationA = new EtomoNumber(EtomoNumber.Type.DOUBLE, "ImageRotationA");`
    image_rotation_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber imageRotationB = new EtomoNumber(EtomoNumber.Type.DOUBLE, "ImageRotationB");`
    image_rotation_b: Mutex<EtomoNumber>,
    /// Java `EtomoNumber binning = new EtomoNumber(EtomoNumber.Type.DOUBLE, "Binning");`
    binning: Mutex<EtomoNumber>,
    /// Java `private boolean fiducialessAlignmentA = false;`
    fiducialess_alignment_a: Mutex<bool>,
    /// Java `private boolean fiducialessAlignmentB = false;`
    fiducialess_alignment_b: Mutex<bool>,
    /// Java `private boolean wholeTomogramSampleA = false;`
    whole_tomogram_sample_a: Mutex<bool>,
    /// Java `private boolean wholeTomogramSampleB = false;`
    whole_tomogram_sample_b: Mutex<bool>,
    /// Java `private TiltAngleSpec tiltAngleSpecA = new TiltAngleSpec();`
    tilt_angle_spec_a: Mutex<TiltAngleSpec>,
    /// Java `private String excludeProjectionsA = null;`
    exclude_projections_a: Mutex<Option<String>>,
    /// Java `private TiltAngleSpec tiltAngleSpecB = new TiltAngleSpec();`
    tilt_angle_spec_b: Mutex<TiltAngleSpec>,
    /// Java `private String excludeProjectionsB = null;`
    exclude_projections_b: Mutex<Option<String>>,
    /// Java `private final EtomoBoolean2 useZFactorsA = new EtomoBoolean2("UseZFactorsA");`
    use_z_factors_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useZFactorsB = new EtomoBoolean2("UseZFactorsB");`
    use_z_factors_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 adjustedFocusA = new EtomoBoolean2("AdjustedFocusA");`
    adjusted_focus_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 adjustedFocusB = new EtomoBoolean2("AdjustedFocusB");`
    adjusted_focus_b: Mutex<EtomoBoolean2>,
    /// Java `private boolean comScriptsCreated = false;`
    com_scripts_created: Mutex<bool>,
    /// Java `private final CombineParams combineParams;`.
    combine_params: Mutex<CombineParams>,
    /// Java `private final SqueezevolParam squeezevolParam;`.  `None` only for a
    /// meta data built without a manager (Java would pass null to the param
    /// constructors, which dereference it).
    squeezevol_param: Mutex<Option<SqueezevolParam>>,
    /// Java `private final TransferfidParam transferfidParamA;`.
    transferfid_param_a: Mutex<Option<TransferfidParam>>,
    /// Java `private final TransferfidParam transferfidParamB;`.
    transferfid_param_b: Mutex<Option<TransferfidParam>>,
    /// Java `private final EtomoBoolean2 defaultParallel = new EtomoBoolean2("DefaultParallel");`
    default_parallel: Mutex<EtomoBoolean2>,
    /// Java `private EtomoBoolean2 tomoGenTiltParallelA = null;`
    tomo_gen_tilt_parallel_a: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 tomoGenTiltParallelB = null;`
    tomo_gen_tilt_parallel_b: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 tilt3dFindTiltParallelA = null;`
    tilt_3d_find_tilt_parallel_a: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 tilt3dFindTiltParallelB = null;`
    tilt_3d_find_tilt_parallel_b: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 finalStackCtfCorrectionParallelA = null;`
    final_stack_ctf_correction_parallel_a: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 finalStackCtfCorrectionParallelB = null;`
    final_stack_ctf_correction_parallel_b: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 combineVolcombineParallel = null;`
    combine_volcombine_parallel: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 bStackProcessed = null;`
    b_stack_processed: Mutex<Option<EtomoBoolean2>>,
    /// Java `private StringBuffer message = new StringBuffer();`
    message: Mutex<String>,
    /// Java `private final EtomoNumber sampleThicknessA = new EtomoNumber(AxisID.FIRST.toString() + '.' + ProcessName.SAMPLE + '.' + THICKNESS_KEY);`
    sample_thickness_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleThicknessB = new EtomoNumber(AxisID.SECOND.toString() + '.' + ProcessName.SAMPLE + '.' + THICKNESS_KEY);`
    sample_thickness_b: Mutex<EtomoNumber>,
    /// Java `private String firstAxisPrepend = null;`
    first_axis_prepend: Mutex<Option<String>>,
    /// Java `private String secondAxisPrepend = null;`
    second_axis_prepend: Mutex<Option<String>>,
    /// Java `private final EtomoBoolean2 defaultGpuProcessing = new EtomoBoolean2("DefaultGpuProcessing");`
    default_gpu_processing: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 fiducialessA = new EtomoBoolean2("A." + FIDUCIALESS_KEY);`
    fiducialess_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 fiducialessB = new EtomoBoolean2("B." + FIDUCIALESS_KEY);`
    fiducialess_b: Mutex<EtomoBoolean2>,
    /// Java `private String targetPatchSizeXandY = TiltalignParam.TARGET_PATCH_SIZE_X_AND_Y_DEFAULT;`
    target_patch_size_x_and_y: Mutex<String>,
    /// Java `private String numberOfLocalPatchesXandY = TiltalignParam.NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT;`
    number_of_local_patches_x_and_y: Mutex<String>,
    /// Java `private final EtomoBoolean2 noBeamTiltSelectedA = new EtomoBoolean2(AxisID.FIRST.getExtension() + "." + DialogType.FINE_ALIGNMENT.getStorableName() + ".NoBeamTiltSelected");`
    no_beam_tilt_selected_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 fixedBeamTiltSelectedA = new EtomoBoolean2(AxisID.FIRST.getExtension() + "." + DialogType.FINE_ALIGNMENT.getStorableName() + ".FixedBeamTiltSelected");`
    fixed_beam_tilt_selected_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber fixedBeamTiltA = new EtomoNumber(AxisID.FIRST.getExtension() + "." + DialogType.FINE_ALIGNMENT.getStorableName() + ".FixedBeamTilt");`
    fixed_beam_tilt_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 noBeamTiltSelectedB = new EtomoBoolean2(AxisID.SECOND.getExtension() + "." + DialogType.FINE_ALIGNMENT.getStorableName() + ".NoBeamTiltSelected");`
    no_beam_tilt_selected_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 fixedBeamTiltSelectedB = new EtomoBoolean2(AxisID.SECOND.getExtension() + "." + DialogType.FINE_ALIGNMENT.getStorableName() + ".FixedBeamTiltSelected");`
    fixed_beam_tilt_selected_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber fixedBeamTiltB = new EtomoNumber(AxisID.SECOND.getExtension() + "." + DialogType.FINE_ALIGNMENT.getStorableName() + ".FixedBeamTilt");`
    fixed_beam_tilt_b: Mutex<EtomoNumber>,
    /// Java `private final FortranInputString sizeToOutputInXandYA = new FortranInputString(2);`
    size_to_output_in_x_and_y_a: Mutex<FortranInputString>,
    /// Java `private final FortranInputString sizeToOutputInXandYB = new FortranInputString(2);`
    size_to_output_in_x_and_y_b: Mutex<FortranInputString>,
    /// Java `private final StringProperty finalStackBetterRadiusA = new StringProperty(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "BetterRadius");`
    final_stack_better_radius_a: Mutex<StringProperty>,
    /// Java `private final StringProperty finalStackBetterRadiusB = new StringProperty(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "BetterRadius");`
    final_stack_better_radius_b: Mutex<StringProperty>,
    /// Java `private final EtomoNumber finalStackFiducialDiameterA = new EtomoNumber(EtomoNumber.Type.DOUBLE, DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "FiducialDiameter");`
    final_stack_fiducial_diameter_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber finalStackFiducialDiameterB = new EtomoNumber(EtomoNumber.Type.DOUBLE, DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "FiducialDiameter");`
    final_stack_fiducial_diameter_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber finalStackExpandCircleIterationsA = new EtomoNumber(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "ExpandCircleIterations");`
    final_stack_expand_circle_iterations_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber finalStackExpandCircleIterationsB = new EtomoNumber(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "ExpandCircleIterations");`
    final_stack_expand_circle_iterations_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 useFinalStackExpandCircleIterationsA = new EtomoBoolean2(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "UseExpandCircleIterations");`
    use_final_stack_expand_circle_iterations_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber useFinalStackExpandCircleIterationsB = new EtomoBoolean2(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "UseExpandCircleIterations");`
    use_final_stack_expand_circle_iterations_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber finalStackPolynomialOrderA = new EtomoNumber(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "PolynomialOrder");`
    final_stack_polynomial_order_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber finalStackPolynomialOrderB = new EtomoNumber(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "PolynomialOrder");`
    final_stack_polynomial_order_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 finalAlignedStackDialogSavedA = new EtomoBoolean2(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "DialogSaved");`
    final_aligned_stack_dialog_saved_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 finalAlignedStackDialogSavedB = new EtomoBoolean2(DialogType.FINAL_ALIGNED_STACK.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "DialogSaved");`
    final_aligned_stack_dialog_saved_b: Mutex<EtomoBoolean2>,
    /// Java `private IntKeyList tomoGenTrialTomogramNameListA = IntKeyList.getStringInstance(DialogType.TOMOGRAM_GENERATION.getStorableName() + "." + AxisID.FIRST.getExtension() + "." + "TrialTomogramName");`
    tomo_gen_trial_tomogram_name_list_a: Mutex<Arc<Mutex<IntKeyList>>>,
    /// Java `private IntKeyList tomoGenTrialTomogramNameListB = IntKeyList.getStringInstance(DialogType.TOMOGRAM_GENERATION.getStorableName() + "." + AxisID.SECOND.getExtension() + "." + "TrialTomogramName");`
    tomo_gen_trial_tomogram_name_list_b: Mutex<Arc<Mutex<IntKeyList>>>,
    /// Java `private final EtomoBoolean2 trackUseRaptorA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + "." + USE_KEY + RAPTOR_KEY);`
    track_use_raptor_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackRaptorUseRawStackA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + "." + RAPTOR_KEY + "." + USE_KEY + RAW_STACK_KEY);`
    track_raptor_use_raw_stack_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber trackRaptorMarkA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + "." + RAPTOR_KEY + "." + MARK_KEY);`
    track_raptor_mark_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackRaptorDiamA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + "." + RAPTOR_KEY + "." + DIAM_KEY);`
    track_raptor_diam_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 stackEraseGoldModelUseFidA = new EtomoBoolean2(STACK_KEY + "." + FIRST_AXIS_KEY + "." + ERASE_GOLD_KEY + "." + MODEL_USE_FID_KEY);`
    stack_erase_gold_model_use_fid_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 stackEraseGoldModelUseFidB = new EtomoBoolean2(STACK_KEY + "." + SECOND_AXIS_KEY + "." + ERASE_GOLD_KEY + "." + MODEL_USE_FID_KEY);`
    stack_erase_gold_model_use_fid_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber posBinningA = new EtomoNumber("TomoPosBinningA");`
    pos_binning_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber posBinningB = new EtomoNumber("TomoPosBinningB");`
    pos_binning_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackBinningA = new EtomoNumber("FinalStackBinningA");`
    stack_binning_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackBinningB = new EtomoNumber("FinalStackBinningB");`
    stack_binning_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stack3dFindBinningA = new EtomoNumber(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "3dFind.Binning");`
    stack_3d_find_binning_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stack3dFindBinningB = new EtomoNumber(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "3dFind.Binning");`
    stack_3d_find_binning_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 postFlattenInputTrimVol = new EtomoBoolean2(POST_KEY + "." + FLATTEN_KEY + "." + INPUT_KEY + TRIM_VOL_KEY);`
    post_flatten_input_trim_vol: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 postFlattenWarpContoursOnOneSurface = new EtomoBoolean2(POST_KEY + "." + FLATTEN_WARP_KEY + "." + CONTOURS_ON_ONE_SURFACE_KEY);`
    post_flatten_warp_contours_on_one_surface: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber postFlattenWarpSpacingInX = new EtomoNumber(EtomoNumber.Type.DOUBLE, POST_KEY + "." + FLATTEN_WARP_KEY + "." + SPACING_IN_KEY + X_KEY);`
    post_flatten_warp_spacing_in_x: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postFlattenWarpSpacingInY = new EtomoNumber(EtomoNumber.Type.DOUBLE, POST_KEY + "." + FLATTEN_WARP_KEY + "." + SPACING_IN_KEY + Y_KEY);`
    post_flatten_warp_spacing_in_y: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 postSqueezeVolInputTrimVol = new EtomoBoolean2(POST_KEY + "." + SQUEEZE_VOL_KEY + "." + INPUT_KEY + TRIM_VOL_KEY);`
    post_squeeze_vol_input_trim_vol: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber postCurTab = new EtomoNumber(POST_KEY + ".CurTab");`
    post_cur_tab: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genCurTab = new EtomoNumber(GEN_KEY + ".CurTab");`
    gen_cur_tab: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 postExists = new EtomoBoolean2(POST_KEY + ".Exists");`
    post_exists: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber lambdaForSmoothing = new EtomoNumber(EtomoNumber.Type.DOUBLE, POST_KEY + ".LambdaForSmoothing");`
    lambda_for_smoothing: Mutex<EtomoNumber>,
    /// Java `private final StringProperty lambdaForSmoothingList = new StringProperty(POST_KEY + ".LambdaForSmoothingList");`
    lambda_for_smoothing_list: Mutex<StringProperty>,
    /// Java `private final StringProperty trackOverlapOfPatchesXandYA = new StringProperty(TRACK_KEY + "." + FIRST_AXIS_KEY + ".OverlapOfPatchesXandY");`
    track_overlap_of_patches_x_and_y_a: Mutex<StringProperty>,
    /// Java `private final StringProperty trackOverlapOfPatchesXandYB = new StringProperty(TRACK_KEY + "." + SECOND_AXIS_KEY + ".OverlapOfPatchesXandY");`
    track_overlap_of_patches_x_and_y_b: Mutex<StringProperty>,
    /// Java `private final StringProperty trackNumberOfPatchesXandYA = new StringProperty(TRACK_KEY + "." + FIRST_AXIS_KEY + ".NumberOfPatchesXandY");`
    track_number_of_patches_x_and_y_a: Mutex<StringProperty>,
    /// Java `private final StringProperty trackNumberOfPatchesXandYB = new StringProperty(TRACK_KEY + "." + SECOND_AXIS_KEY + ".NumberOfPatchesXandY");`
    track_number_of_patches_x_and_y_b: Mutex<StringProperty>,
    /// Java `private final StringProperty trackLengthAndOverlapA = new StringProperty(TRACK_KEY + "." + FIRST_AXIS_KEY + ".LengthAndOverlap");`
    track_length_and_overlap_a: Mutex<StringProperty>,
    /// Java `private final StringProperty trackLengthAndOverlapB = new StringProperty(TRACK_KEY + "." + SECOND_AXIS_KEY + ".LengthAndOverlap");`
    track_length_and_overlap_b: Mutex<StringProperty>,
    /// Java `private final StringProperty trackMethodA = new StringProperty(TRACK_KEY + "." + FIRST_AXIS_KEY + ".TrackMethod");`
    track_method_a: Mutex<StringProperty>,
    /// Java `private final StringProperty trackMethodB = new StringProperty(TRACK_KEY + "." + SECOND_AXIS_KEY + ".TrackMethod");`
    track_method_b: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 fineExistsA = new EtomoBoolean2(FINE_KEY + "." + FIRST_AXIS_KEY + ".Exists");`
    fine_exists_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 fineExistsB = new EtomoBoolean2(FINE_KEY + "." + SECOND_AXIS_KEY + ".Exists");`
    fine_exists_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber genLogA = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + FIRST_AXIS_KEY + ".Log");`
    gen_log_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genLogB = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + SECOND_AXIS_KEY + ".Log");`
    gen_log_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleFactorLogA = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + FIRST_AXIS_KEY + ".Scale.Factor.Log");`
    gen_scale_factor_log_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleFactorLogB = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + SECOND_AXIS_KEY + ".Scale.Factor.Log");`
    gen_scale_factor_log_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleOffsetLogA = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + FIRST_AXIS_KEY + ".Scale.Offset.Log");`
    gen_scale_offset_log_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleOffsetLogB = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + SECOND_AXIS_KEY + ".Scale.Offset.Log");`
    gen_scale_offset_log_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleFactorLinearA = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + FIRST_AXIS_KEY + ".Scale.Factor.Linear");`
    gen_scale_factor_linear_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleFactorLinearB = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + SECOND_AXIS_KEY + ".Scale.Factor.Linear");`
    gen_scale_factor_linear_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleOffsetLinearA = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + FIRST_AXIS_KEY + ".Scale.Offset.Linear");`
    gen_scale_offset_linear_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genScaleOffsetLinearB = new EtomoNumber(EtomoNumber.Type.DOUBLE, GEN_KEY + "." + SECOND_AXIS_KEY + ".Scale.Offset.Linear");`
    gen_scale_offset_linear_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 genExistsA = new EtomoBoolean2(GEN_KEY + "." + FIRST_AXIS_KEY + ".Exists");`
    gen_exists_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genExistsB = new EtomoBoolean2(GEN_KEY + "." + SECOND_AXIS_KEY + ".Exists");`
    gen_exists_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 posExistsA = new EtomoBoolean2(POS_KEY + ".A.Exists");`
    pos_exists_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 posExistsB = new EtomoBoolean2(POS_KEY + ".B.Exists");`
    pos_exists_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genBackProjectionA = new EtomoBoolean2(GEN_KEY + "." + FIRST_AXIS_KEY + ".BackProjection");`
    gen_back_projection_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genBackProjectionB = new EtomoBoolean2(GEN_KEY + "." + SECOND_AXIS_KEY + ".BackProjection");`
    gen_back_projection_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genSubareaA = new EtomoBoolean2(GEN_KEY + "." + FIRST_AXIS_KEY + ".Subarea");`
    gen_subarea_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genSubareaB = new EtomoBoolean2(GEN_KEY + "." + SECOND_AXIS_KEY + ".Subarea");`
    gen_subarea_b: Mutex<EtomoBoolean2>,
    /// Java `private final StringProperty genSubareaSizeA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".SubareaSize");`
    gen_subarea_size_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genSubareaSizeB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".SubareaSize");`
    gen_subarea_size_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genYOffsetOfSubareaA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".YOffsetOfSubarea");`
    gen_y_offset_of_subarea_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genYOffsetOfSubareaB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".YOffsetOfSubarea");`
    gen_y_offset_of_subarea_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genRadialRadiusA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".RadialRadius");`
    gen_radial_radius_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genRadialRadiusB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".RadialRadius");`
    gen_radial_radius_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genRadialSigmaA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".RadialSigma");`
    gen_radial_sigma_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genRadialSigmaB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".RadialSigma");`
    gen_radial_sigma_b: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolXMin = new StringProperty(POST_KEY + ".Trimvol.XMin");`
    post_trimvol_x_min: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolXMinFromBatchruntomo = new StringProperty("batchruntomo.Trimvol.XMin");`
    post_trimvol_x_min_from_batchruntomo: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolXMax = new StringProperty(POST_KEY + ".Trimvol.XMax");`
    post_trimvol_x_max: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolXMaxFromBatchruntomo = new StringProperty("batchruntomo.Trimvol.XMax");`
    post_trimvol_x_max_from_batchruntomo: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolYMin = new StringProperty(POST_KEY + ".Trimvol.YMin");`
    post_trimvol_y_min: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolYMinFromBatchruntomo = new StringProperty("batchruntomo.Trimvol.YMin");`
    post_trimvol_y_min_from_batchruntomo: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolYMax = new StringProperty(POST_KEY + ".Trimvol.YMax");`
    post_trimvol_y_max: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolYMaxFromBatchruntomo = new StringProperty("batchruntomo.Trimvol.YMax");`
    post_trimvol_y_max_from_batchruntomo: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolZMin = new StringProperty(POST_KEY + ".Trimvol.ZMin");`
    post_trimvol_z_min: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolZMinFromBatchruntomo = new StringProperty("batchruntomo.Trimvol.ZMin");`
    post_trimvol_z_min_from_batchruntomo: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolZMax = new StringProperty(POST_KEY + ".Trimvol.ZMax");`
    post_trimvol_z_max: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolZMaxFromBatchruntomo = new StringProperty("batchruntomo.Trimvol.ZMax");`
    post_trimvol_z_max_from_batchruntomo: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 postTrimvolConvertToBytes = new EtomoBoolean2(POST_KEY + ".Trimvol.ConvertToBytes");`
    post_trimvol_convert_to_bytes: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 postTrimvolFixedScaling = new EtomoBoolean2(POST_KEY + ".Trimvol.FixedScaling");`
    post_trimvol_fixed_scaling: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 postTrimvolFlippedVolume = new EtomoBoolean2(POST_KEY + ".Trimvol.FlippedVolume");`
    post_trimvol_flipped_volume: Mutex<EtomoBoolean2>,
    /// Java `private final StringProperty postTrimvolSectionScaleMin = new StringProperty(POST_KEY + ".Trimvol.SectionScaleMin");`
    post_trimvol_section_scale_min: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolSectionScaleMax = new StringProperty(POST_KEY + ".Trimvol.SectionScaleMax");`
    post_trimvol_section_scale_max: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolFixedScaleMin = new StringProperty(POST_KEY + ".Trimvol.FixedScaleMin");`
    post_trimvol_fixed_scale_min: Mutex<StringProperty>,
    /// Java `private final StringProperty postTrimvolFixedScaleMax = new StringProperty(POST_KEY + ".Trimvol.FixedScaleMax");`
    post_trimvol_fixed_scale_max: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 postTrimvolSwapYZ = new EtomoBoolean2(POST_KEY + ".Trimvol.SwapYZ");`
    post_trimvol_swap_yz: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 postTrimvolRotateX = new EtomoBoolean2(POST_KEY + ".Trimvol.RotateX");`
    post_trimvol_rotate_x: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber postTrimvolScaleXMin = new EtomoNumber(POST_KEY + ".Trimvol.ScaleXMin");`
    post_trimvol_scale_x_min: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleXMinFromBatchruntomo = new EtomoNumber("batchruntomo.Trimvol.ScaleXMin");`
    post_trimvol_scale_x_min_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleXMax = new EtomoNumber(POST_KEY + ".Trimvol.ScaleXMax");`
    post_trimvol_scale_x_max: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleXMaxFromBatchruntomo = new EtomoNumber("batchruntomo.Trimvol.ScaleXMax");`
    post_trimvol_scale_x_max_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleYMin = new EtomoNumber(POST_KEY + ".Trimvol.ScaleYMin");`
    post_trimvol_scale_y_min: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleYMinFromBatchruntomo = new EtomoNumber("batchruntomo.Trimvol.ScaleYMin");`
    post_trimvol_scale_y_min_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleYMax = new EtomoNumber(POST_KEY + ".Trimvol.ScaleYMax");`
    post_trimvol_scale_y_max: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolScaleYMaxFromBatchruntomo = new EtomoNumber("batchruntomo.Trimvol.ScaleYMax");`
    post_trimvol_scale_y_max_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 eraseBeadsInitialized = new EtomoBoolean2(STACK_KEY + ".EraseBeadsInitialized");`
    erase_beads_initialized: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackSeedModelManualA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + ".SeedModel.Manual");`
    track_seed_model_manual_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackSeedModelManualB = new EtomoBoolean2(TRACK_KEY + "." + SECOND_AXIS_KEY + ".SeedModel.Manual");`
    track_seed_model_manual_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackSeedModelAutoA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + ".SeedModel.Auto");`
    track_seed_model_auto_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackSeedModelAutoB = new EtomoBoolean2(TRACK_KEY + "." + SECOND_AXIS_KEY + ".SeedModel.Auto");`
    track_seed_model_auto_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackSeedModelTransferA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + ".SeedModel.Transfer");`
    track_seed_model_transfer_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackSeedModelTransferB = new EtomoBoolean2(TRACK_KEY + "." + SECOND_AXIS_KEY + ".SeedModel.Transfer");`
    track_seed_model_transfer_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackExcludeInsideAreasA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + ".ExcludeInsideAreas");`
    track_exclude_inside_areas_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackExcludeInsideAreasB = new EtomoBoolean2(TRACK_KEY + "." + SECOND_AXIS_KEY + ".ExcludeInsideAreas");`
    track_exclude_inside_areas_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber trackJustFindShiftsNearZeroA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + ".JustFindShiftsNearZero");`
    track_just_find_shifts_near_zero_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackJustFindShiftsNearZeroB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + ".JustFindShiftsNearZero");`
    track_just_find_shifts_near_zero_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackTargetNumberOfBeadsA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + ".TargetNumberOfBeads");`
    track_target_number_of_beads_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackTargetNumberOfBeadsB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + ".TargetNumberOfBeads");`
    track_target_number_of_beads_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackTargetDensityOfBeadsA = new EtomoNumber(EtomoNumber.Type.DOUBLE, TRACK_KEY + "." + FIRST_AXIS_KEY + ".TargetDensityOfBeads");`
    track_target_density_of_beads_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackTargetDensityOfBeadsB = new EtomoNumber(EtomoNumber.Type.DOUBLE, TRACK_KEY + "." + SECOND_AXIS_KEY + ".TargetDensityOfBeads");`
    track_target_density_of_beads_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 trackClusteredPointsAllowedElongatedA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + ".ClusteredPointsAllowed.Elongated");`
    track_clustered_points_allowed_elongated_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackClusteredPointsAllowedElongatedB = new EtomoBoolean2(TRACK_KEY + "." + SECOND_AXIS_KEY + ".ClusteredPointsAllowed.Elongated");`
    track_clustered_points_allowed_elongated_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber trackClusteredPointsAllowedElongatedValueA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + ".ClusteredPointsAllowed.Elongated.Value");`
    track_clustered_points_allowed_elongated_value_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackClusteredPointsAllowedElongatedValueB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + ".ClusteredPointsAllowed.Elongated.Value");`
    track_clustered_points_allowed_elongated_value_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 trackAdvancedA = new EtomoBoolean2(TRACK_KEY + "." + FIRST_AXIS_KEY + ".Advanced");`
    track_advanced_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 trackAdvancedB = new EtomoBoolean2(TRACK_KEY + "." + SECOND_AXIS_KEY + ".Advanced");`
    track_advanced_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber stack3dFindThicknessA = new EtomoNumber(STACK_KEY + "." + FIRST_AXIS_KEY + ".3dFind.Thickness");`
    stack_3d_find_thickness_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stack3dFindThicknessB = new EtomoNumber(STACK_KEY + "." + SECOND_AXIS_KEY + ".3dFind.Thickness");`
    stack_3d_find_thickness_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 setFEIPixelSize = new EtomoBoolean2("SetFEIPixelSize");`
    set_fei_pixel_size: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 postTrimvolNewStyleZ = new EtomoBoolean2(POST_KEY + "Trimvol.NewStyleZ");`
    post_trimvol_new_style_z: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 postTrimvolScalingNewStyleZ = new EtomoBoolean2(POST_KEY + "Trimvol.Scaling.NewStyleZ");`
    post_trimvol_scaling_new_style_z: Mutex<EtomoBoolean2>,
    /// Java `private final FortranInputString stackCtfAutoFitRangeAndStepA = new FortranInputString(2);`
    stack_ctf_auto_fit_range_and_step_a: Mutex<FortranInputString>,
    /// Java `private final FortranInputString stackCtfAutoFitRangeAndStepB = new FortranInputString(2);`
    stack_ctf_auto_fit_range_and_step_b: Mutex<FortranInputString>,
    /// Java `private final StringProperty origScopeTemplate = new StringProperty("Orig.ScopeTemplate");`
    orig_scope_template: Mutex<StringProperty>,
    /// Java `private final StringProperty origSystemTemplate = new StringProperty("Orig.SystemTemplate");`
    orig_system_template: Mutex<StringProperty>,
    /// Java `private final StringProperty origUserTemplate = new StringProperty("Orig.UserTemplate");`
    orig_user_template: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 isTwodirA = new EtomoBoolean2(STACK_KEY + "." + FIRST_AXIS_KEY + ".Is.Twodir");`
    is_twodir_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 isTwodirB = new EtomoBoolean2(STACK_KEY + "." + SECOND_AXIS_KEY + ".Is.Twodir");`
    is_twodir_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber twodirA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + FIRST_AXIS_KEY + ".Twodir");`
    twodir_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber twodirB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + SECOND_AXIS_KEY + ".Twodir");`
    twodir_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 isDoseSymA = new EtomoBoolean2(STACK_KEY + "." + FIRST_AXIS_KEY + ".Is.DoseSym");`
    is_dose_sym_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 isDoseSymB = new EtomoBoolean2(STACK_KEY + "." + SECOND_AXIS_KEY + ".Is.DoseSym");`
    is_dose_sym_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber doseSymA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + FIRST_AXIS_KEY + ".DoseSym");`
    dose_sym_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber doseSymB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + SECOND_AXIS_KEY + ".DoseSym");`
    dose_sym_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber seedAndTrackTabA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + "Tab.SeedAndTrack");`
    seed_and_track_tab_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber seedAndTrackTabB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + "Tab.SeedAndTrack");`
    seed_and_track_tab_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber raptorTabA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + "Tab.Raptor");`
    raptor_tab_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber raptorTabB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + "Tab.Raptor");`
    raptor_tab_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber coarseAntialiasFilterA = new EtomoNumber(COARSE_KEY + "." + FIRST_AXIS_KEY + ".AntialiasFilter");`
    coarse_antialias_filter_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber coarseAntialiasFilterB = new EtomoNumber(COARSE_KEY + "." + SECOND_AXIS_KEY + ".AntialiasFilter");`
    coarse_antialias_filter_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackAntialiasFilterA = new EtomoNumber(STACK_KEY + "." + FIRST_AXIS_KEY + ".AntialiasFilter");`
    stack_antialias_filter_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackAntialiasFilterB = new EtomoNumber(STACK_KEY + "." + SECOND_AXIS_KEY + ".AntialiasFilter");`
    stack_antialias_filter_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackElongatedPointsAllowedA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + ".ElongatedPointsAllowed");`
    track_elongated_points_allowed_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackElongatedPointsAllowedB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + ".ElongatedPointsAllowed");`
    track_elongated_points_allowed_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackLowerTargetForClusteredA = new EtomoNumber(EtomoNumber.Type.DOUBLE, TRACK_KEY + "." + FIRST_AXIS_KEY + ".LowerTargetForClustered");`
    track_lower_target_for_clustered_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackLowerTargetForClusteredB = new EtomoNumber(EtomoNumber.Type.DOUBLE, TRACK_KEY + "." + SECOND_AXIS_KEY + ".LowerTargetForClustered");`
    track_lower_target_for_clustered_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 origViewsWithMagChangesA = new EtomoBoolean2(COARSE_KEY + ".Tiltxcorr." + FIRST_AXIS_KEY + ".Orig.ViewsWithMagChanges");`
    orig_views_with_mag_changes_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 origViewsWithMagChangesB = new EtomoBoolean2(COARSE_KEY + ".Tiltxcorr." + SECOND_AXIS_KEY + ".Orig.ViewsWithMagChanges");`
    orig_views_with_mag_changes_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 origViewsWithMagChangesSetA = new EtomoBoolean2(COARSE_KEY + ".Tiltxcorr." + FIRST_AXIS_KEY + ".Orig.ViewsWithMagChanges.Set");`
    orig_views_with_mag_changes_set_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 origViewsWithMagChangesSetB = new EtomoBoolean2(COARSE_KEY + ".Tiltxcorr." + SECOND_AXIS_KEY + ".Orig.ViewsWithMagChanges.Set");`
    orig_views_with_mag_changes_set_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 weightWholeTracksA = new EtomoBoolean2(FINE_KEY + FIRST_AXIS_KEY + ".WeightWholeTracks");`
    weight_whole_tracks_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 weightWholeTracksB = new EtomoBoolean2(FINE_KEY + SECOND_AXIS_KEY + ".WeightWholeTracks");`
    weight_whole_tracks_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber lengthOfPiecesA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + ".LengthOfPieces");`
    length_of_pieces_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber lengthOfPiecesB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + ".LengthOfPieces");`
    length_of_pieces_b: Mutex<EtomoNumber>,
    /// Java `private boolean removeIncorrectRawImageStackExtKey = false;`
    remove_incorrect_raw_image_stack_ext_key: Mutex<bool>,
    /// Java `private final StringProperty rawImageStackExt = new StringProperty("RawImageStackExt", true);`
    raw_image_stack_ext: Mutex<StringProperty>,
    /// Java `private boolean removeIncorrectOrigRawImageStackExtKey = false;`
    remove_incorrect_orig_raw_image_stack_ext_key: Mutex<bool>,
    /// Java `private final StringProperty origRawImageStackExt = new StringProperty("OrigRawImageStackExt", true);`
    orig_raw_image_stack_ext: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 origRawImageStackExtLock = new EtomoBoolean2("OrigRawImageStackExt.Lock");`
    orig_raw_image_stack_ext_lock: Mutex<EtomoBoolean2>,
    /// Java `private final StringProperty origRawImageStackExtFromBatchRunTomo = new StringProperty("batchruntomo.OrigImageStackExt");`
    orig_raw_image_stack_ext_from_batch_run_tomo: Mutex<StringProperty>,
    /// Java `private final EtomoNumber minimumOverlapA = new EtomoNumber(TRACK_KEY + "." + FIRST_AXIS_KEY + ".MinimumOverlap");`
    minimum_overlap_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber minimumOverlapB = new EtomoNumber(TRACK_KEY + "." + SECOND_AXIS_KEY + ".MinimumOverlap");`
    minimum_overlap_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber targetMeasurementRatioA = new EtomoNumber(EtomoNumber.Type.DOUBLE, FINE_KEY + "." + FIRST_AXIS_KEY + ".TargetMeasurementRatio");`
    target_measurement_ratio_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber targetMeasurementRatioB = new EtomoNumber(EtomoNumber.Type.DOUBLE, FINE_KEY + "." + SECOND_AXIS_KEY + ".TargetMeasurementRatio");`
    target_measurement_ratio_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber minMeasurementRatioA = new EtomoNumber(EtomoNumber.Type.DOUBLE, FINE_KEY + "." + FIRST_AXIS_KEY + ".MinMeasurementRatio");`
    min_measurement_ratio_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber minMeasurementRatioB = new EtomoNumber(EtomoNumber.Type.DOUBLE, FINE_KEY + "." + SECOND_AXIS_KEY + ".MinMeasurementRatio");`
    min_measurement_ratio_b: Mutex<EtomoNumber>,
    /// Java `private final StringProperty orderOfRestrictionsA = new StringProperty(FINE_KEY + "." + FIRST_AXIS_KEY + ".OrderOfRestrictions");`
    order_of_restrictions_a: Mutex<StringProperty>,
    /// Java `private final StringProperty orderOfRestrictionsB = new StringProperty(FINE_KEY + "." + SECOND_AXIS_KEY + ".OrderOfRestrictions");`
    order_of_restrictions_b: Mutex<StringProperty>,
    /// Java `private final EtomoNumber skipBeamTiltWithOneRotA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, FINE_KEY + "." + FIRST_AXIS_KEY + ".SkipBeamTiltWithOneRot");`
    skip_beam_tilt_with_one_rot_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber skipBeamTiltWithOneRotB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, FINE_KEY + "." + SECOND_AXIS_KEY + ".SkipBeamTiltWithOneRot");`
    skip_beam_tilt_with_one_rot_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber fineLocalAlignValidationA = new EtomoNumber(FINE_KEY + "." + FIRST_AXIS_KEY + ".LocalAlignValidation");`
    fine_local_align_validation_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber fineLocalAlignValidationB = new EtomoNumber(FINE_KEY + "." + SECOND_AXIS_KEY + ".LocalAlignValidation");`
    fine_local_align_validation_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleTypeA = new EtomoNumber(POS_KEY + "." + FIRST_AXIS_KEY + ".SampleType");`
    sample_type_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleTypeB = new EtomoNumber(POS_KEY + "." + SECOND_AXIS_KEY + ".SampleType");`
    sample_type_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber hasGoldBeadsA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, POS_KEY + "." + FIRST_AXIS_KEY + ".HasGoldBeads");`
    has_gold_beads_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber hasGoldBeadsB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, POS_KEY + "." + SECOND_AXIS_KEY + ".HasGoldBeads");`
    has_gold_beads_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber positioningBeadSizeA = new EtomoNumber(EtomoNumber.Type.DOUBLE, POS_KEY + "." + FIRST_AXIS_KEY + ".FiducialDiameter");`
    positioning_bead_size_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber positioningBeadSizeB = new EtomoNumber(EtomoNumber.Type.DOUBLE, POS_KEY + "." + SECOND_AXIS_KEY + ".FiducialDiameter");`
    positioning_bead_size_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber extraThicknessA = new EtomoNumber(EtomoNumber.Type.DOUBLE, POS_KEY + "." + FIRST_AXIS_KEY + ".ExtraThickness");`
    extra_thickness_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber extraThicknessB = new EtomoNumber(EtomoNumber.Type.DOUBLE, POS_KEY + "." + SECOND_AXIS_KEY + ".ExtraThickness");`
    extra_thickness_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber extraThicknessCryoA = new EtomoNumber(EtomoNumber.Type.DOUBLE, POS_KEY + "." + FIRST_AXIS_KEY + ".ExtraThickness.Cryo");`
    extra_thickness_cryo_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber extraThicknessCryoB = new EtomoNumber(EtomoNumber.Type.DOUBLE, POS_KEY + "." + SECOND_AXIS_KEY + ".ExtraThickness.Cryo");`
    extra_thickness_cryo_b: Mutex<EtomoNumber>,
    /// Java `private final StringProperty genHammingLikeFilterA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".HammingLikeFilter");`
    gen_hamming_like_filter_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genHammingLikeFilterB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".HammingLikeFilter");`
    gen_hamming_like_filter_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genFakeSIRTiterationsA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".FakeSIRTiterations");`
    gen_fake_sirt_iterations_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genFakeSIRTiterationsB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".FakeSIRTiterations");`
    gen_fake_sirt_iterations_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genExactFilterSizeA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".ExactFilterSize");`
    gen_exact_filter_size_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genExactFilterSizeB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".ExactFilterSize");`
    gen_exact_filter_size_b: Mutex<StringProperty>,
    /// Java `private final StringProperty sirtRadialRadiusA = new StringProperty(SIRT_KEY + "." + FIRST_AXIS_KEY + ".RadialRadius");`
    sirt_radial_radius_a: Mutex<StringProperty>,
    /// Java `private final StringProperty sirtRadialRadiusB = new StringProperty(SIRT_KEY + "." + SECOND_AXIS_KEY + ".RadialRadius");`
    sirt_radial_radius_b: Mutex<StringProperty>,
    /// Java `private final StringProperty sirtRadialSigmaA = new StringProperty(SIRT_KEY + "." + FIRST_AXIS_KEY + ".RadialSigma");`
    sirt_radial_sigma_a: Mutex<StringProperty>,
    /// Java `private final StringProperty sirtRadialSigmaB = new StringProperty(SIRT_KEY + "." + SECOND_AXIS_KEY + ".RadialSigma");`
    sirt_radial_sigma_b: Mutex<StringProperty>,
    /// Java `private final StringProperty batchRunTomoLogReadTimestamp = new StringProperty("BatchRunTomoLog.Read.Timestamp", true);`
    batch_run_tomo_log_read_timestamp: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 batchRunTomoLogReadFinished = new EtomoBoolean2("BatchRunTomoLog.Read.Finished");`
    batch_run_tomo_log_read_finished: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber postTrimvolSwapYZFromBatchruntomo = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.Trimvol.SwapYZ");`
    post_trimvol_swap_yz_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolRotateXFromBatchruntomo = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.Trimvol.RotateX");`
    post_trimvol_rotate_x_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolConvertToBytesFromBatchruntomo = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.Trimvol.ConvertToBytes");`
    post_trimvol_convert_to_bytes_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolSectionScaleMinFromBatchruntomo = new EtomoNumber("batchruntomo.Trimvol.ScaleSectionMin");`
    post_trimvol_section_scale_min_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postTrimvolSectionScaleMaxFromBatchruntomo = new EtomoNumber("batchruntomo.Trimvol.ScaleSectionMax");`
    post_trimvol_section_scale_max_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackAlignedStackEraseGoldFromBatchruntomo = new EtomoNumber("batchruntomo.AlignedStack.eraseGold");`
    stack_aligned_stack_erase_gold_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 positioningNewDialogA = new EtomoBoolean2(POS_KEY + "." + FIRST_AXIS_KEY + ".NewDialog");`
    positioning_new_dialog_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 positioningNewDialogB = new EtomoBoolean2(POS_KEY + "." + SECOND_AXIS_KEY + ".NewDialog");`
    positioning_new_dialog_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genFilterTrialsA = new EtomoBoolean2(GEN_KEY + "." + FIRST_AXIS_KEY + ".FilterTrials");`
    gen_filter_trials_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genFilterTrialsB = new EtomoBoolean2(GEN_KEY + "." + SECOND_AXIS_KEY + ".FilterTrials");`
    gen_filter_trials_b: Mutex<EtomoBoolean2>,
    /// Java `private AxisID batchRunTomoLogReadAxisID = null;`
    batch_run_tomo_log_read_axis_id: Mutex<Option<AxisID>>,
    /// Java `private final StringProperty genFilterTrialsFakeSIRTiterationsA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".FilterTrials.FakeSIRTiterations");`
    gen_filter_trials_fake_sirt_iterations_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsFakeSIRTiterationsB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".FilterTrials.FakeSIRTiterations");`
    gen_filter_trials_fake_sirt_iterations_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsExactObjectSizesA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".FilterTrials.ExactObjectSizes");`
    gen_filter_trials_exact_object_sizes_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsExactObjectSizesB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".FilterTrials.ExactObjectSizes");`
    gen_filter_trials_exact_object_sizes_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsGaussianCutoffsA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".FilterTrials.GaussianCutoffs");`
    gen_filter_trials_gaussian_cutoffs_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsGaussianCutoffsB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".FilterTrials.GaussianCutoffs");`
    gen_filter_trials_gaussian_cutoffs_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsGaussianFalloffsA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".FilterTrials.GaussianFalloffs");`
    gen_filter_trials_gaussian_falloffs_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsGaussianFalloffsB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".FilterTrials.GaussianFalloffs");`
    gen_filter_trials_gaussian_falloffs_b: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsHammingLikeStartsA = new StringProperty(GEN_KEY + "." + FIRST_AXIS_KEY + ".FilterTrials.HammingLikeStarts");`
    gen_filter_trials_hamming_like_starts_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genFilterTrialsHammingLikeStartsB = new StringProperty(GEN_KEY + "." + SECOND_AXIS_KEY + ".FilterTrials.HammingLikeStarts");`
    gen_filter_trials_hamming_like_starts_b: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterLowPassRadiusSigmaA = new StringProperty(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.LowPassRadiusSigma");`
    stack_mtf_filter_low_pass_radius_sigma_a: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterLowPassRadiusSigmaB = new StringProperty(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.LowPassRadiusSigma");`
    stack_mtf_filter_low_pass_radius_sigma_b: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterMtfFileA = new StringProperty(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.MtfFile");`
    stack_mtf_filter_mtf_file_a: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterMtfFileB = new StringProperty(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.MtfFile");`
    stack_mtf_filter_mtf_file_b: Mutex<StringProperty>,
    /// Java `private final EtomoNumber stackMtfFilterMaximumInverseA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.MaximumInverse");`
    stack_mtf_filter_maximum_inverse_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackMtfFilterMaximumInverseB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.MaximumInverse");`
    stack_mtf_filter_maximum_inverse_b: Mutex<EtomoNumber>,
    /// Java `private final StringProperty stackMtfFilterInverseRolloffRadiusSigmaA = new StringProperty(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.InverseRolloffRadiusSigma");`
    stack_mtf_filter_inverse_rolloff_radius_sigma_a: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterInverseRolloffRadiusSigmaB = new StringProperty(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.InverseRolloffRadiusSigma");`
    stack_mtf_filter_inverse_rolloff_radius_sigma_b: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 useStackMtfFilterFixedImageDoseA = new EtomoBoolean2(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.Use.FixedImageDose");`
    use_stack_mtf_filter_fixed_image_dose_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useStackMtfFilterFixedImageDoseB = new EtomoBoolean2(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.Use.FixedImageDose");`
    use_stack_mtf_filter_fixed_image_dose_b: Mutex<EtomoBoolean2>,
    /// Java `private final StringProperty stackMtfFilterFixedImageDoseA = new StringProperty(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.FixedImageDose");`
    stack_mtf_filter_fixed_image_dose_a: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterFixedImageDoseB = new StringProperty(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.FixedImageDose");`
    stack_mtf_filter_fixed_image_dose_b: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterDoseWeightingFileA = new StringProperty(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.DoseWeightingFile");`
    stack_mtf_filter_dose_weighting_file_a: Mutex<StringProperty>,
    /// Java `private final StringProperty stackMtfFilterDoseWeightingFileB = new StringProperty(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.DoseWeightingFile");`
    stack_mtf_filter_dose_weighting_file_b: Mutex<StringProperty>,
    /// Java `private final EtomoNumber stackMtfFilterTypeOfDoseFileA = new EtomoNumber(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.TypeOfDoseFile");`
    stack_mtf_filter_type_of_dose_file_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackMtfFilterTypeOfDoseFileB = new EtomoNumber(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.TypeOfDoseFile");`
    stack_mtf_filter_type_of_dose_file_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 stackMtfFilterVoltage200A = new EtomoBoolean2(STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.Voltage.200");`
    stack_mtf_filter_voltage_200_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 stackMtfFilterVoltage200B = new EtomoBoolean2(STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.Voltage.200");`
    stack_mtf_filter_voltage_200_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber stackMtfFilterOptimalDoseScalingA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.OptimalDoseScaling");`
    stack_mtf_filter_optimal_dose_scaling_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackMtfFilterOptimalDoseScalingB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.OptimalDoseScaling");`
    stack_mtf_filter_optimal_dose_scaling_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackMtfFilterBidirectionalNumViewsA = new EtomoNumber(EtomoNumber.Type.INTEGER, STACK_KEY + "." + FIRST_AXIS_KEY + "." + "MtfFilter.BidirectionalNumViews");`
    stack_mtf_filter_bidirectional_num_views_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackMtfFilterBidirectionalNumViewsB = new EtomoNumber(EtomoNumber.Type.INTEGER, STACK_KEY + "." + SECOND_AXIS_KEY + "." + "MtfFilter.BidirectionalNumViews");`
    stack_mtf_filter_bidirectional_num_views_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 useStackCtfPhaseFlipXAxisTiltA = new EtomoBoolean2(STACK_KEY + "." + FIRST_AXIS_KEY + ".Use." + "CtfPhaseFlip.XAxisTilt");`
    use_stack_ctf_phase_flip_x_axis_tilt_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useStackCtfPhaseFlipXAxisTiltB = new EtomoBoolean2(STACK_KEY + "." + SECOND_AXIS_KEY + ".Use." + "CtfPhaseFlip.XAxisTilt");`
    use_stack_ctf_phase_flip_x_axis_tilt_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber stackCtfPhaseFlipXAxisTiltA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + FIRST_AXIS_KEY + "." + "CtfPhaseFlip.XAxisTilt");`
    stack_ctf_phase_flip_x_axis_tilt_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackCtfPhaseFlipXAxisTiltB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + SECOND_AXIS_KEY + "." + "CtfPhaseFlip.XAxisTilt");`
    stack_ctf_phase_flip_x_axis_tilt_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackCtfPhaseFlipScaleByCtfPowerA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + FIRST_AXIS_KEY + "." + "CtfPhaseFlip.ScaleByCtfPower");`
    stack_ctf_phase_flip_scale_by_ctf_power_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackCtfPhaseFlipScaleByCtfPowerB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + SECOND_AXIS_KEY + "." + "CtfPhaseFlip.ScaleByCtfPower");`
    stack_ctf_phase_flip_scale_by_ctf_power_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genSuperSampleFactorA = new EtomoNumber(GEN_KEY + "." + FIRST_AXIS_KEY + ".SuperSampleFactor");`
    gen_super_sample_factor_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genSuperSampleFactorB = new EtomoNumber(GEN_KEY + "." + SECOND_AXIS_KEY + ".SuperSampleFactor");`
    gen_super_sample_factor_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 genExpandInputLinesA = new EtomoBoolean2(GEN_KEY + "." + FIRST_AXIS_KEY + ".ExpandInputLines");`
    gen_expand_input_lines_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genExpandInputLinesB = new EtomoBoolean2(GEN_KEY + "." + SECOND_AXIS_KEY + ".ExpandInputLines");`
    gen_expand_input_lines_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genSirtA = new EtomoBoolean2(GEN_KEY + "." + FIRST_AXIS_KEY + ".Sirt");`
    gen_sirt_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genSirtB = new EtomoBoolean2(GEN_KEY + "." + SECOND_AXIS_KEY + ".Sirt");`
    gen_sirt_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genCtf3dOldStyleXtiltingA = new EtomoBoolean2(GEN_KEY + "." + "CTF_3D." + FIRST_AXIS_KEY + ".OldStyleXtilting");`
    gen_ctf_3d_old_style_xtilting_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genCtf3dOldStyleXtiltingB = new EtomoBoolean2(GEN_KEY + "." + "CTF_3D." + SECOND_AXIS_KEY + ".OldStyleXtilting");`
    gen_ctf_3d_old_style_xtilting_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genCtf3dVerticalSlicesA = new EtomoBoolean2(GEN_KEY + "." + "CTF_3D." + FIRST_AXIS_KEY + ".VerticalSlices");`
    gen_ctf_3d_vertical_slices_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 genCtf3dVerticalSlicesB = new EtomoBoolean2(GEN_KEY + "." + "CTF_3D." + SECOND_AXIS_KEY + ".VerticalSlices");`
    gen_ctf_3d_vertical_slices_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber genCtf3dFourierReduceByFactorA = new EtomoNumber(EtomoNumber.Type.INTEGER, GEN_KEY + "." + "CTF_3D." + FIRST_AXIS_KEY + ".FourierReduceByFactor");`
    gen_ctf_3d_fourier_reduce_by_factor_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genCtf3dFourierReduceByFactorB = new EtomoNumber(EtomoNumber.Type.INTEGER, GEN_KEY + "." + "CTF_3D." + SECOND_AXIS_KEY + ".FourierReduceByFactor");`
    gen_ctf_3d_fourier_reduce_by_factor_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 subtomoReorientationTypeNone = new EtomoBoolean2(SUBTOMO_KEY + "." + "ReorientationType.None");`
    subtomo_reorientation_type_none: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 subtomoReorientationTypeFlipped = new EtomoBoolean2(SUBTOMO_KEY + "." + "ReorientationType.Flipped");`
    subtomo_reorientation_type_flipped: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 subtomoReorientationTypeRotated = new EtomoBoolean2(SUBTOMO_KEY + "." + "ReorientationType.Rotated");`
    subtomo_reorientation_type_rotated: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber subtomoMakeVolumeStacks = new EtomoNumber(EtomoNumber.Type.INTEGER, SUBTOMO_KEY + "." + "MakeVolumeStacks");`
    subtomo_make_volume_stacks: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber subtomoExtentOfZLevelsInNm = new EtomoNumber(EtomoNumber.Type.INTEGER, SUBTOMO_KEY + "." + "ExtentOfZLevelsInNm");`
    subtomo_extent_of_z_levels_in_nm: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber subtomoNewAlignedBinning = new EtomoNumber(EtomoNumber.Type.INTEGER, SUBTOMO_KEY + "." + "NewAlignedBinning");`
    subtomo_new_aligned_binning: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber subtomoFourierReduceByFactor = new EtomoNumber(EtomoNumber.Type.INTEGER, SUBTOMO_KEY + "." + "FourierReduceByFactor");`
    subtomo_fourier_reduce_by_factor: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber reduceFiltVolReductionFactor = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "ReductionFactor");`
    reduce_filt_vol_reduction_factor: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber reduceFiltVolZReductionFactor = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "ZReductionFactor");`
    reduce_filt_vol_z_reduction_factor: Mutex<EtomoNumber>,
    /// Java `private final StringProperty reduceFiltVolLowPassRadiusSigma = new StringProperty(REDUCE_FILT_VOL_KEY + "." + "LowPassRadiusSigma");`
    reduce_filt_vol_low_pass_radius_sigma: Mutex<StringProperty>,
    /// Java `private final EtomoNumber reduceFiltVolDeconvolutionStrength = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "DeconvolutionStrength");`
    reduce_filt_vol_deconvolution_strength: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber reduceFiltVolSNRFalloff = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "SNRFalloff");`
    reduce_filt_vol_snr_falloff: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber reduceFiltVolHighPassNyquist = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "HighPassNyquist");`
    reduce_filt_vol_high_pass_nyquist: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber reduceFiltVolDefocusInMicrons = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "DefocusInMicrons");`
    reduce_filt_vol_defocus_in_microns: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber reduceFiltVolPhaseShift = new EtomoNumber(EtomoNumber.Type.DOUBLE, REDUCE_FILT_VOL_KEY + "." + "PhaseShift");`
    reduce_filt_vol_phase_shift: Mutex<EtomoNumber>,
    /// Java `private final StringProperty altTomoRootnameToProcess = new StringProperty(ALT_TOMO_SETUP_KEY + "." + "RootnameToProcess");`
    alt_tomo_rootname_to_process: Mutex<StringProperty>,
    /// Java `private final EtomoBoolean2 altTomoTrimVolume = new EtomoBoolean2(ALT_TOMO_SETUP_KEY + "." + "TrimVolume");`
    alt_tomo_trim_volume: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 altTomoArchiveOrigStack = new EtomoBoolean2(ALT_TOMO_SETUP_KEY + "." + "PreprocessForExtremes");`
    alt_tomo_archive_orig_stack: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 ctf3dSetupSlabThicknessInNmSet = new EtomoBoolean2("ctf3dsetup.SlabThicknessInNmSet");`
    ctf_3d_setup_slab_thickness_in_nm_set: Mutex<EtomoBoolean2>,
}

// Safety: every field is a `Mutex` of owned data except the `&'static` references.
// `manager` is an `&'static ApplicationManager`, which is `Send + Sync` (the
// `BaseManager` trait requires it).  The one member that keeps the auto traits from
// applying is `BaseMetaDataBase`'s `Option<&'static dyn LogProperties>` (the log
// window, an EDT object): `BaseMetaDataBase` never calls through it (see its
// `store_with_created_prepend`/`load_with_created_prepend`) and nothing in this module
// reaches it, so sharing the reference across threads never touches the object
// behind it.
unsafe impl Send for MetaData {}
unsafe impl Sync for MetaData {}

impl MetaData {
    /// Java `MetaData(ApplicationManager, LogProperties, boolean)`, together with the
    /// field initialisers Java runs before its body.
    ///
    /// Upstream bug fixed in translation (MetaData.java:1250-1253): the source calls
    /// `stackCtfAutoFitRangeAndStepA.setPropertiesKey` twice, the second time with the
    /// B key, and never sets `stackCtfAutoFitRangeAndStepB`'s key.  The second call is
    /// made on B here.  See `store_with_prepend` for what that changes in the `.edf`.
    pub fn new(
        manager: Option<&'static ApplicationManager>,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
    ) -> MetaData {
        let meta_data = MetaData {
            base: BaseMetaDataBase::new_force_old_style(
                manager.map(|manager| manager as &'static dyn BaseManager),
                log_properties,
                false,
                new_dataset,
                false,
            ),
            manager,
            squeezevol_param: Mutex::new(None),
            transferfid_param_a: Mutex::new(None),
            transferfid_param_b: Mutex::new(None),
            dataset_name: Mutex::new(String::new()),
            backup_directory: Mutex::new(String::new()),
            distortion_file: Mutex::new(None),
            mag_gradient_file: Mutex::new(None),
            data_source: Mutex::new(DataSource::Ccd),
            view_type: Mutex::new(ViewType::SingleView),
            pixel_size: Mutex::new(f64::NAN),
            use_local_alignments_a: Mutex::new(true),
            use_local_alignments_b: Mutex::new(true),
            fiducial_diameter: Mutex::new(f64::NAN),
            half_float_mode_output: Mutex::new(EtomoNumber::new_with_name("HalfFloatModeOutput")),
            image_rotation_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "ImageRotationA",
            )),
            image_rotation_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "ImageRotationB",
            )),
            binning: Mutex::new(EtomoNumber::new_with_type_and_name(Type::Double, "Binning")),
            fiducialess_alignment_a: Mutex::new(false),
            fiducialess_alignment_b: Mutex::new(false),
            whole_tomogram_sample_a: Mutex::new(false),
            whole_tomogram_sample_b: Mutex::new(false),
            tilt_angle_spec_a: Mutex::new(TiltAngleSpec::new()),
            exclude_projections_a: Mutex::new(None),
            tilt_angle_spec_b: Mutex::new(TiltAngleSpec::new()),
            exclude_projections_b: Mutex::new(None),
            use_z_factors_a: Mutex::new(EtomoBoolean2::new_with_name("UseZFactorsA")),
            use_z_factors_b: Mutex::new(EtomoBoolean2::new_with_name("UseZFactorsB")),
            adjusted_focus_a: Mutex::new(EtomoBoolean2::new_with_name("AdjustedFocusA")),
            adjusted_focus_b: Mutex::new(EtomoBoolean2::new_with_name("AdjustedFocusB")),
            com_scripts_created: Mutex::new(false),
            // Java assigns `combineParams = new CombineParams(manager)` in the
            // constructor body; its constructor reads nothing from the manager.
            combine_params: Mutex::new(CombineParams::new(
                manager.map(|manager| manager as &'static dyn BaseManager),
            )),
            default_parallel: Mutex::new(EtomoBoolean2::new_with_name("DefaultParallel")),
            tomo_gen_tilt_parallel_a: Mutex::new(None),
            tomo_gen_tilt_parallel_b: Mutex::new(None),
            tilt_3d_find_tilt_parallel_a: Mutex::new(None),
            tilt_3d_find_tilt_parallel_b: Mutex::new(None),
            final_stack_ctf_correction_parallel_a: Mutex::new(None),
            final_stack_ctf_correction_parallel_b: Mutex::new(None),
            combine_volcombine_parallel: Mutex::new(None),
            b_stack_processed: Mutex::new(None),
            message: Mutex::new(String::new()),
            sample_thickness_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}",
                AxisID::First,
                ProcessName::SAMPLE,
                THICKNESS_KEY
            ))),
            sample_thickness_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}",
                AxisID::Second,
                ProcessName::SAMPLE,
                THICKNESS_KEY
            ))),
            first_axis_prepend: Mutex::new(None),
            second_axis_prepend: Mutex::new(None),
            default_gpu_processing: Mutex::new(EtomoBoolean2::new_with_name(
                "DefaultGpuProcessing",
            )),
            fiducialess_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "A.{}",
                FIDUCIALESS_KEY
            ))),
            fiducialess_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "B.{}",
                FIDUCIALESS_KEY
            ))),
            target_patch_size_x_and_y: Mutex::new(
                TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_DEFAULT.to_string(),
            ),
            number_of_local_patches_x_and_y: Mutex::new(
                TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT.to_string(),
            ),
            no_beam_tilt_selected_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.NoBeamTiltSelected",
                AxisID::First.get_extension(),
                DialogType::FineAlignment.get_storable_name()
            ))),
            fixed_beam_tilt_selected_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.FixedBeamTiltSelected",
                AxisID::First.get_extension(),
                DialogType::FineAlignment.get_storable_name()
            ))),
            fixed_beam_tilt_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.FixedBeamTilt",
                AxisID::First.get_extension(),
                DialogType::FineAlignment.get_storable_name()
            ))),
            no_beam_tilt_selected_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.NoBeamTiltSelected",
                AxisID::Second.get_extension(),
                DialogType::FineAlignment.get_storable_name()
            ))),
            fixed_beam_tilt_selected_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.FixedBeamTiltSelected",
                AxisID::Second.get_extension(),
                DialogType::FineAlignment.get_storable_name()
            ))),
            fixed_beam_tilt_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.FixedBeamTilt",
                AxisID::Second.get_extension(),
                DialogType::FineAlignment.get_storable_name()
            ))),
            size_to_output_in_x_and_y_a: Mutex::new(FortranInputString::new(2)),
            size_to_output_in_x_and_y_b: Mutex::new(FortranInputString::new(2)),
            final_stack_better_radius_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.BetterRadius",
                DialogType::FinalAlignedStack.get_storable_name(),
                AxisID::First.get_extension()
            )))),
            final_stack_better_radius_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.BetterRadius",
                DialogType::FinalAlignedStack.get_storable_name(),
                AxisID::Second.get_extension()
            )))),
            final_stack_fiducial_diameter_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.FiducialDiameter",
                    DialogType::FinalAlignedStack.get_storable_name(),
                    AxisID::First.get_extension()
                ),
            )),
            final_stack_fiducial_diameter_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.FiducialDiameter",
                    DialogType::FinalAlignedStack.get_storable_name(),
                    AxisID::Second.get_extension()
                ),
            )),
            final_stack_expand_circle_iterations_a: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.ExpandCircleIterations",
                    DialogType::FinalAlignedStack.get_storable_name(),
                    AxisID::First.get_extension()
                ),
            )),
            final_stack_expand_circle_iterations_b: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.ExpandCircleIterations",
                    DialogType::FinalAlignedStack.get_storable_name(),
                    AxisID::Second.get_extension()
                ),
            )),
            use_final_stack_expand_circle_iterations_a: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.UseExpandCircleIterations",
                    DialogType::FinalAlignedStack.get_storable_name(),
                    AxisID::First.get_extension()
                ),
            )),
            use_final_stack_expand_circle_iterations_b: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.UseExpandCircleIterations",
                    DialogType::FinalAlignedStack.get_storable_name(),
                    AxisID::Second.get_extension()
                ),
            )),
            final_stack_polynomial_order_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.PolynomialOrder",
                DialogType::FinalAlignedStack.get_storable_name(),
                AxisID::First.get_extension()
            ))),
            final_stack_polynomial_order_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.PolynomialOrder",
                DialogType::FinalAlignedStack.get_storable_name(),
                AxisID::Second.get_extension()
            ))),
            final_aligned_stack_dialog_saved_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.DialogSaved",
                DialogType::FinalAlignedStack.get_storable_name(),
                AxisID::First.get_extension()
            ))),
            final_aligned_stack_dialog_saved_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.DialogSaved",
                DialogType::FinalAlignedStack.get_storable_name(),
                AxisID::Second.get_extension()
            ))),
            tomo_gen_trial_tomogram_name_list_a: Mutex::new(Arc::new(Mutex::new(
                IntKeyList::get_string_instance_with_key(&format!(
                    "{}.{}.TrialTomogramName",
                    DialogType::TomogramGeneration.get_storable_name(),
                    AxisID::First.get_extension()
                )),
            ))),
            tomo_gen_trial_tomogram_name_list_b: Mutex::new(Arc::new(Mutex::new(
                IntKeyList::get_string_instance_with_key(&format!(
                    "{}.{}.TrialTomogramName",
                    DialogType::TomogramGeneration.get_storable_name(),
                    AxisID::Second.get_extension()
                )),
            ))),
            track_use_raptor_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}{}",
                TRACK_KEY, FIRST_AXIS_KEY, USE_KEY, RAPTOR_KEY
            ))),
            track_raptor_use_raw_stack_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}.{}{}",
                TRACK_KEY, FIRST_AXIS_KEY, RAPTOR_KEY, USE_KEY, RAW_STACK_KEY
            ))),
            track_raptor_mark_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}.{}",
                TRACK_KEY, FIRST_AXIS_KEY, RAPTOR_KEY, MARK_KEY
            ))),
            track_raptor_diam_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.{}.{}",
                TRACK_KEY, FIRST_AXIS_KEY, RAPTOR_KEY, DIAM_KEY
            ))),
            stack_erase_gold_model_use_fid_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}.{}",
                STACK_KEY, FIRST_AXIS_KEY, ERASE_GOLD_KEY, MODEL_USE_FID_KEY
            ))),
            stack_erase_gold_model_use_fid_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}.{}",
                STACK_KEY, SECOND_AXIS_KEY, ERASE_GOLD_KEY, MODEL_USE_FID_KEY
            ))),
            pos_binning_a: Mutex::new(EtomoNumber::new_with_name("TomoPosBinningA")),
            pos_binning_b: Mutex::new(EtomoNumber::new_with_name("TomoPosBinningB")),
            stack_binning_a: Mutex::new(EtomoNumber::new_with_name("FinalStackBinningA")),
            stack_binning_b: Mutex::new(EtomoNumber::new_with_name("FinalStackBinningB")),
            stack_3d_find_binning_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.3dFind.Binning",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            stack_3d_find_binning_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.3dFind.Binning",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            post_flatten_input_trim_vol: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}{}",
                POST_KEY, FLATTEN_KEY, INPUT_KEY, TRIM_VOL_KEY
            ))),
            post_flatten_warp_contours_on_one_surface: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.{}",
                    POST_KEY, FLATTEN_WARP_KEY, CONTOURS_ON_ONE_SURFACE_KEY
                ),
            )),
            post_flatten_warp_spacing_in_x: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}{}",
                    POST_KEY, FLATTEN_WARP_KEY, SPACING_IN_KEY, X_KEY
                ),
            )),
            post_flatten_warp_spacing_in_y: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}{}",
                    POST_KEY, FLATTEN_WARP_KEY, SPACING_IN_KEY, Y_KEY
                ),
            )),
            post_squeeze_vol_input_trim_vol: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.{}{}",
                POST_KEY, SQUEEZE_VOL_KEY, INPUT_KEY, TRIM_VOL_KEY
            ))),
            post_cur_tab: Mutex::new(EtomoNumber::new_with_name(&format!("{}.CurTab", POST_KEY))),
            gen_cur_tab: Mutex::new(EtomoNumber::new_with_name(&format!("{}.CurTab", GEN_KEY))),
            post_exists: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Exists",
                POST_KEY
            ))),
            lambda_for_smoothing: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.LambdaForSmoothing", POST_KEY),
            )),
            lambda_for_smoothing_list: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.LambdaForSmoothingList",
                POST_KEY
            )))),
            track_overlap_of_patches_x_and_y_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.OverlapOfPatchesXandY", TRACK_KEY, FIRST_AXIS_KEY),
            ))),
            track_overlap_of_patches_x_and_y_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.OverlapOfPatchesXandY", TRACK_KEY, SECOND_AXIS_KEY),
            ))),
            track_number_of_patches_x_and_y_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.NumberOfPatchesXandY", TRACK_KEY, FIRST_AXIS_KEY),
            ))),
            track_number_of_patches_x_and_y_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.NumberOfPatchesXandY", TRACK_KEY, SECOND_AXIS_KEY),
            ))),
            track_length_and_overlap_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.LengthAndOverlap",
                TRACK_KEY, FIRST_AXIS_KEY
            )))),
            track_length_and_overlap_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.LengthAndOverlap",
                TRACK_KEY, SECOND_AXIS_KEY
            )))),
            track_method_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.TrackMethod",
                TRACK_KEY, FIRST_AXIS_KEY
            )))),
            track_method_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.TrackMethod",
                TRACK_KEY, SECOND_AXIS_KEY
            )))),
            fine_exists_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Exists",
                FINE_KEY, FIRST_AXIS_KEY
            ))),
            fine_exists_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Exists",
                FINE_KEY, SECOND_AXIS_KEY
            ))),
            gen_log_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Log", GEN_KEY, FIRST_AXIS_KEY),
            )),
            gen_log_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Log", GEN_KEY, SECOND_AXIS_KEY),
            )),
            gen_scale_factor_log_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Factor.Log", GEN_KEY, FIRST_AXIS_KEY),
            )),
            gen_scale_factor_log_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Factor.Log", GEN_KEY, SECOND_AXIS_KEY),
            )),
            gen_scale_offset_log_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Offset.Log", GEN_KEY, FIRST_AXIS_KEY),
            )),
            gen_scale_offset_log_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Offset.Log", GEN_KEY, SECOND_AXIS_KEY),
            )),
            gen_scale_factor_linear_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Factor.Linear", GEN_KEY, FIRST_AXIS_KEY),
            )),
            gen_scale_factor_linear_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Factor.Linear", GEN_KEY, SECOND_AXIS_KEY),
            )),
            gen_scale_offset_linear_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Offset.Linear", GEN_KEY, FIRST_AXIS_KEY),
            )),
            gen_scale_offset_linear_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Scale.Offset.Linear", GEN_KEY, SECOND_AXIS_KEY),
            )),
            gen_exists_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Exists",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_exists_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Exists",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            pos_exists_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.A.Exists",
                POS_KEY
            ))),
            pos_exists_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.B.Exists",
                POS_KEY
            ))),
            gen_back_projection_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.BackProjection",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_back_projection_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.BackProjection",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_subarea_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Subarea",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_subarea_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Subarea",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_subarea_size_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.SubareaSize",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_subarea_size_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.SubareaSize",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            gen_y_offset_of_subarea_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.YOffsetOfSubarea",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_y_offset_of_subarea_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.YOffsetOfSubarea",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            gen_radial_radius_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialRadius",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_radial_radius_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialRadius",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            gen_radial_sigma_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialSigma",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_radial_sigma_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialSigma",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            post_trimvol_x_min: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.XMin",
                POST_KEY
            )))),
            post_trimvol_x_min_from_batchruntomo: Mutex::new(StringProperty::new_with_key(Some(
                "batchruntomo.Trimvol.XMin",
            ))),
            post_trimvol_x_max: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.XMax",
                POST_KEY
            )))),
            post_trimvol_x_max_from_batchruntomo: Mutex::new(StringProperty::new_with_key(Some(
                "batchruntomo.Trimvol.XMax",
            ))),
            post_trimvol_y_min: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.YMin",
                POST_KEY
            )))),
            post_trimvol_y_min_from_batchruntomo: Mutex::new(StringProperty::new_with_key(Some(
                "batchruntomo.Trimvol.YMin",
            ))),
            post_trimvol_y_max: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.YMax",
                POST_KEY
            )))),
            post_trimvol_y_max_from_batchruntomo: Mutex::new(StringProperty::new_with_key(Some(
                "batchruntomo.Trimvol.YMax",
            ))),
            post_trimvol_z_min: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.ZMin",
                POST_KEY
            )))),
            post_trimvol_z_min_from_batchruntomo: Mutex::new(StringProperty::new_with_key(Some(
                "batchruntomo.Trimvol.ZMin",
            ))),
            post_trimvol_z_max: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.ZMax",
                POST_KEY
            )))),
            post_trimvol_z_max_from_batchruntomo: Mutex::new(StringProperty::new_with_key(Some(
                "batchruntomo.Trimvol.ZMax",
            ))),
            post_trimvol_convert_to_bytes: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Trimvol.ConvertToBytes",
                POST_KEY
            ))),
            post_trimvol_fixed_scaling: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Trimvol.FixedScaling",
                POST_KEY
            ))),
            post_trimvol_flipped_volume: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Trimvol.FlippedVolume",
                POST_KEY
            ))),
            post_trimvol_section_scale_min: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.Trimvol.SectionScaleMin", POST_KEY),
            ))),
            post_trimvol_section_scale_max: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.Trimvol.SectionScaleMax", POST_KEY),
            ))),
            post_trimvol_fixed_scale_min: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.FixedScaleMin",
                POST_KEY
            )))),
            post_trimvol_fixed_scale_max: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.Trimvol.FixedScaleMax",
                POST_KEY
            )))),
            post_trimvol_swap_yz: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Trimvol.SwapYZ",
                POST_KEY
            ))),
            post_trimvol_rotate_x: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Trimvol.RotateX",
                POST_KEY
            ))),
            post_trimvol_scale_x_min: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.Trimvol.ScaleXMin",
                POST_KEY
            ))),
            post_trimvol_scale_x_min_from_batchruntomo: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.Trimvol.ScaleXMin",
            )),
            post_trimvol_scale_x_max: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.Trimvol.ScaleXMax",
                POST_KEY
            ))),
            post_trimvol_scale_x_max_from_batchruntomo: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.Trimvol.ScaleXMax",
            )),
            post_trimvol_scale_y_min: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.Trimvol.ScaleYMin",
                POST_KEY
            ))),
            post_trimvol_scale_y_min_from_batchruntomo: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.Trimvol.ScaleYMin",
            )),
            post_trimvol_scale_y_max: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.Trimvol.ScaleYMax",
                POST_KEY
            ))),
            post_trimvol_scale_y_max_from_batchruntomo: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.Trimvol.ScaleYMax",
            )),
            erase_beads_initialized: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.EraseBeadsInitialized",
                STACK_KEY
            ))),
            track_seed_model_manual_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.SeedModel.Manual",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_seed_model_manual_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.SeedModel.Manual",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_seed_model_auto_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.SeedModel.Auto",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_seed_model_auto_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.SeedModel.Auto",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_seed_model_transfer_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.SeedModel.Transfer",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_seed_model_transfer_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.SeedModel.Transfer",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_exclude_inside_areas_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.ExcludeInsideAreas",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_exclude_inside_areas_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.ExcludeInsideAreas",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_just_find_shifts_near_zero_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.JustFindShiftsNearZero",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_just_find_shifts_near_zero_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.JustFindShiftsNearZero",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_target_number_of_beads_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.TargetNumberOfBeads",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_target_number_of_beads_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.TargetNumberOfBeads",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_target_density_of_beads_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.TargetDensityOfBeads", TRACK_KEY, FIRST_AXIS_KEY),
            )),
            track_target_density_of_beads_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.TargetDensityOfBeads", TRACK_KEY, SECOND_AXIS_KEY),
            )),
            track_clustered_points_allowed_elongated_a: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.ClusteredPointsAllowed.Elongated",
                    TRACK_KEY, FIRST_AXIS_KEY
                ),
            )),
            track_clustered_points_allowed_elongated_b: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.ClusteredPointsAllowed.Elongated",
                    TRACK_KEY, SECOND_AXIS_KEY
                ),
            )),
            track_clustered_points_allowed_elongated_value_a: Mutex::new(
                EtomoNumber::new_with_name(&format!(
                    "{}.{}.ClusteredPointsAllowed.Elongated.Value",
                    TRACK_KEY, FIRST_AXIS_KEY
                )),
            ),
            track_clustered_points_allowed_elongated_value_b: Mutex::new(
                EtomoNumber::new_with_name(&format!(
                    "{}.{}.ClusteredPointsAllowed.Elongated.Value",
                    TRACK_KEY, SECOND_AXIS_KEY
                )),
            ),
            track_advanced_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Advanced",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_advanced_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Advanced",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            stack_3d_find_thickness_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.3dFind.Thickness",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            stack_3d_find_thickness_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.3dFind.Thickness",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            set_fei_pixel_size: Mutex::new(EtomoBoolean2::new_with_name("SetFEIPixelSize")),
            post_trimvol_new_style_z: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}Trimvol.NewStyleZ",
                POST_KEY
            ))),
            post_trimvol_scaling_new_style_z: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}Trimvol.Scaling.NewStyleZ",
                POST_KEY
            ))),
            stack_ctf_auto_fit_range_and_step_a: Mutex::new(FortranInputString::new(2)),
            stack_ctf_auto_fit_range_and_step_b: Mutex::new(FortranInputString::new(2)),
            orig_scope_template: Mutex::new(StringProperty::new_with_key(Some(
                "Orig.ScopeTemplate",
            ))),
            orig_system_template: Mutex::new(StringProperty::new_with_key(Some(
                "Orig.SystemTemplate",
            ))),
            orig_user_template: Mutex::new(StringProperty::new_with_key(Some("Orig.UserTemplate"))),
            is_twodir_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Is.Twodir",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            is_twodir_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Is.Twodir",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            twodir_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Twodir", STACK_KEY, FIRST_AXIS_KEY),
            )),
            twodir_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.Twodir", STACK_KEY, SECOND_AXIS_KEY),
            )),
            is_dose_sym_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Is.DoseSym",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            is_dose_sym_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Is.DoseSym",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            dose_sym_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.DoseSym", STACK_KEY, FIRST_AXIS_KEY),
            )),
            dose_sym_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.DoseSym", STACK_KEY, SECOND_AXIS_KEY),
            )),
            seed_and_track_tab_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}Tab.SeedAndTrack",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            seed_and_track_tab_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}Tab.SeedAndTrack",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            raptor_tab_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}Tab.Raptor",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            raptor_tab_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}Tab.Raptor",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            coarse_antialias_filter_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.AntialiasFilter",
                COARSE_KEY, FIRST_AXIS_KEY
            ))),
            coarse_antialias_filter_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.AntialiasFilter",
                COARSE_KEY, SECOND_AXIS_KEY
            ))),
            stack_antialias_filter_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.AntialiasFilter",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            stack_antialias_filter_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.AntialiasFilter",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            track_elongated_points_allowed_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.ElongatedPointsAllowed",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            track_elongated_points_allowed_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.ElongatedPointsAllowed",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            track_lower_target_for_clustered_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.LowerTargetForClustered", TRACK_KEY, FIRST_AXIS_KEY),
            )),
            track_lower_target_for_clustered_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.LowerTargetForClustered", TRACK_KEY, SECOND_AXIS_KEY),
            )),
            orig_views_with_mag_changes_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Tiltxcorr.{}.Orig.ViewsWithMagChanges",
                COARSE_KEY, FIRST_AXIS_KEY
            ))),
            orig_views_with_mag_changes_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Tiltxcorr.{}.Orig.ViewsWithMagChanges",
                COARSE_KEY, SECOND_AXIS_KEY
            ))),
            orig_views_with_mag_changes_set_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Tiltxcorr.{}.Orig.ViewsWithMagChanges.Set",
                COARSE_KEY, FIRST_AXIS_KEY
            ))),
            orig_views_with_mag_changes_set_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.Tiltxcorr.{}.Orig.ViewsWithMagChanges.Set",
                COARSE_KEY, SECOND_AXIS_KEY
            ))),
            weight_whole_tracks_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}{}.WeightWholeTracks",
                FINE_KEY, FIRST_AXIS_KEY
            ))),
            weight_whole_tracks_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}{}.WeightWholeTracks",
                FINE_KEY, SECOND_AXIS_KEY
            ))),
            length_of_pieces_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.LengthOfPieces",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            length_of_pieces_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.LengthOfPieces",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            remove_incorrect_raw_image_stack_ext_key: Mutex::new(false),
            raw_image_stack_ext: Mutex::new(
                StringProperty::new_with_key_and_return_null_when_empty(
                    Some("RawImageStackExt"),
                    true,
                ),
            ),
            remove_incorrect_orig_raw_image_stack_ext_key: Mutex::new(false),
            orig_raw_image_stack_ext: Mutex::new(
                StringProperty::new_with_key_and_return_null_when_empty(
                    Some("OrigRawImageStackExt"),
                    true,
                ),
            ),
            orig_raw_image_stack_ext_lock: Mutex::new(EtomoBoolean2::new_with_name(
                "OrigRawImageStackExt.Lock",
            )),
            orig_raw_image_stack_ext_from_batch_run_tomo: Mutex::new(StringProperty::new_with_key(
                Some("batchruntomo.OrigImageStackExt"),
            )),
            minimum_overlap_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.MinimumOverlap",
                TRACK_KEY, FIRST_AXIS_KEY
            ))),
            minimum_overlap_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.MinimumOverlap",
                TRACK_KEY, SECOND_AXIS_KEY
            ))),
            target_measurement_ratio_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.TargetMeasurementRatio", FINE_KEY, FIRST_AXIS_KEY),
            )),
            target_measurement_ratio_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.TargetMeasurementRatio", FINE_KEY, SECOND_AXIS_KEY),
            )),
            min_measurement_ratio_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.MinMeasurementRatio", FINE_KEY, FIRST_AXIS_KEY),
            )),
            min_measurement_ratio_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.MinMeasurementRatio", FINE_KEY, SECOND_AXIS_KEY),
            )),
            order_of_restrictions_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.OrderOfRestrictions",
                FINE_KEY, FIRST_AXIS_KEY
            )))),
            order_of_restrictions_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.OrderOfRestrictions",
                FINE_KEY, SECOND_AXIS_KEY
            )))),
            skip_beam_tilt_with_one_rot_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                &format!("{}.{}.SkipBeamTiltWithOneRot", FINE_KEY, FIRST_AXIS_KEY),
            )),
            skip_beam_tilt_with_one_rot_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                &format!("{}.{}.SkipBeamTiltWithOneRot", FINE_KEY, SECOND_AXIS_KEY),
            )),
            fine_local_align_validation_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.LocalAlignValidation",
                FINE_KEY, FIRST_AXIS_KEY
            ))),
            fine_local_align_validation_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.LocalAlignValidation",
                FINE_KEY, SECOND_AXIS_KEY
            ))),
            sample_type_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.SampleType",
                POS_KEY, FIRST_AXIS_KEY
            ))),
            sample_type_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.SampleType",
                POS_KEY, SECOND_AXIS_KEY
            ))),
            has_gold_beads_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                &format!("{}.{}.HasGoldBeads", POS_KEY, FIRST_AXIS_KEY),
            )),
            has_gold_beads_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                &format!("{}.{}.HasGoldBeads", POS_KEY, SECOND_AXIS_KEY),
            )),
            positioning_bead_size_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.FiducialDiameter", POS_KEY, FIRST_AXIS_KEY),
            )),
            positioning_bead_size_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.FiducialDiameter", POS_KEY, SECOND_AXIS_KEY),
            )),
            extra_thickness_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.ExtraThickness", POS_KEY, FIRST_AXIS_KEY),
            )),
            extra_thickness_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.ExtraThickness", POS_KEY, SECOND_AXIS_KEY),
            )),
            extra_thickness_cryo_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.ExtraThickness.Cryo", POS_KEY, FIRST_AXIS_KEY),
            )),
            extra_thickness_cryo_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.ExtraThickness.Cryo", POS_KEY, SECOND_AXIS_KEY),
            )),
            gen_hamming_like_filter_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.HammingLikeFilter",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_hamming_like_filter_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.HammingLikeFilter",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            gen_fake_sirt_iterations_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.FakeSIRTiterations",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_fake_sirt_iterations_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.FakeSIRTiterations",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            gen_exact_filter_size_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.ExactFilterSize",
                GEN_KEY, FIRST_AXIS_KEY
            )))),
            gen_exact_filter_size_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.ExactFilterSize",
                GEN_KEY, SECOND_AXIS_KEY
            )))),
            sirt_radial_radius_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialRadius",
                SIRT_KEY, FIRST_AXIS_KEY
            )))),
            sirt_radial_radius_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialRadius",
                SIRT_KEY, SECOND_AXIS_KEY
            )))),
            sirt_radial_sigma_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialSigma",
                SIRT_KEY, FIRST_AXIS_KEY
            )))),
            sirt_radial_sigma_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.RadialSigma",
                SIRT_KEY, SECOND_AXIS_KEY
            )))),
            batch_run_tomo_log_read_timestamp: Mutex::new(
                StringProperty::new_with_key_and_return_null_when_empty(
                    Some("BatchRunTomoLog.Read.Timestamp"),
                    true,
                ),
            ),
            batch_run_tomo_log_read_finished: Mutex::new(EtomoBoolean2::new_with_name(
                "BatchRunTomoLog.Read.Finished",
            )),
            post_trimvol_swap_yz_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_type_and_name(Type::Boolean, "batchruntomo.Trimvol.SwapYZ"),
            ),
            post_trimvol_rotate_x_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_type_and_name(Type::Boolean, "batchruntomo.Trimvol.RotateX"),
            ),
            post_trimvol_convert_to_bytes_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.Trimvol.ConvertToBytes",
                ),
            ),
            post_trimvol_section_scale_min_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_name("batchruntomo.Trimvol.ScaleSectionMin"),
            ),
            post_trimvol_section_scale_max_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_name("batchruntomo.Trimvol.ScaleSectionMax"),
            ),
            stack_aligned_stack_erase_gold_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_name("batchruntomo.AlignedStack.eraseGold"),
            ),
            positioning_new_dialog_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.NewDialog",
                POS_KEY, FIRST_AXIS_KEY
            ))),
            positioning_new_dialog_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.NewDialog",
                POS_KEY, SECOND_AXIS_KEY
            ))),
            gen_filter_trials_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.FilterTrials",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_filter_trials_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.FilterTrials",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            batch_run_tomo_log_read_axis_id: Mutex::new(None),
            gen_filter_trials_fake_sirt_iterations_a: Mutex::new(StringProperty::new_with_key(
                Some(&format!(
                    "{}.{}.FilterTrials.FakeSIRTiterations",
                    GEN_KEY, FIRST_AXIS_KEY
                )),
            )),
            gen_filter_trials_fake_sirt_iterations_b: Mutex::new(StringProperty::new_with_key(
                Some(&format!(
                    "{}.{}.FilterTrials.FakeSIRTiterations",
                    GEN_KEY, SECOND_AXIS_KEY
                )),
            )),
            gen_filter_trials_exact_object_sizes_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.FilterTrials.ExactObjectSizes",
                    GEN_KEY, FIRST_AXIS_KEY
                ),
            ))),
            gen_filter_trials_exact_object_sizes_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.FilterTrials.ExactObjectSizes",
                    GEN_KEY, SECOND_AXIS_KEY
                ),
            ))),
            gen_filter_trials_gaussian_cutoffs_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.FilterTrials.GaussianCutoffs",
                    GEN_KEY, FIRST_AXIS_KEY
                ),
            ))),
            gen_filter_trials_gaussian_cutoffs_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.FilterTrials.GaussianCutoffs",
                    GEN_KEY, SECOND_AXIS_KEY
                ),
            ))),
            gen_filter_trials_gaussian_falloffs_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.FilterTrials.GaussianFalloffs",
                    GEN_KEY, FIRST_AXIS_KEY
                ),
            ))),
            gen_filter_trials_gaussian_falloffs_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.FilterTrials.GaussianFalloffs",
                    GEN_KEY, SECOND_AXIS_KEY
                ),
            ))),
            gen_filter_trials_hamming_like_starts_a: Mutex::new(StringProperty::new_with_key(
                Some(&format!(
                    "{}.{}.FilterTrials.HammingLikeStarts",
                    GEN_KEY, FIRST_AXIS_KEY
                )),
            )),
            gen_filter_trials_hamming_like_starts_b: Mutex::new(StringProperty::new_with_key(
                Some(&format!(
                    "{}.{}.FilterTrials.HammingLikeStarts",
                    GEN_KEY, SECOND_AXIS_KEY
                )),
            )),
            stack_mtf_filter_low_pass_radius_sigma_a: Mutex::new(StringProperty::new_with_key(
                Some(&format!(
                    "{}.{}.MtfFilter.LowPassRadiusSigma",
                    STACK_KEY, FIRST_AXIS_KEY
                )),
            )),
            stack_mtf_filter_low_pass_radius_sigma_b: Mutex::new(StringProperty::new_with_key(
                Some(&format!(
                    "{}.{}.MtfFilter.LowPassRadiusSigma",
                    STACK_KEY, SECOND_AXIS_KEY
                )),
            )),
            stack_mtf_filter_mtf_file_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.MtfFilter.MtfFile",
                STACK_KEY, FIRST_AXIS_KEY
            )))),
            stack_mtf_filter_mtf_file_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.MtfFilter.MtfFile",
                STACK_KEY, SECOND_AXIS_KEY
            )))),
            stack_mtf_filter_maximum_inverse_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.MtfFilter.MaximumInverse", STACK_KEY, FIRST_AXIS_KEY),
            )),
            stack_mtf_filter_maximum_inverse_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.MtfFilter.MaximumInverse", STACK_KEY, SECOND_AXIS_KEY),
            )),
            stack_mtf_filter_inverse_rolloff_radius_sigma_a: Mutex::new(
                StringProperty::new_with_key(Some(&format!(
                    "{}.{}.MtfFilter.InverseRolloffRadiusSigma",
                    STACK_KEY, FIRST_AXIS_KEY
                ))),
            ),
            stack_mtf_filter_inverse_rolloff_radius_sigma_b: Mutex::new(
                StringProperty::new_with_key(Some(&format!(
                    "{}.{}.MtfFilter.InverseRolloffRadiusSigma",
                    STACK_KEY, SECOND_AXIS_KEY
                ))),
            ),
            use_stack_mtf_filter_fixed_image_dose_a: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.MtfFilter.Use.FixedImageDose",
                    STACK_KEY, FIRST_AXIS_KEY
                ),
            )),
            use_stack_mtf_filter_fixed_image_dose_b: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.MtfFilter.Use.FixedImageDose",
                    STACK_KEY, SECOND_AXIS_KEY
                ),
            )),
            stack_mtf_filter_fixed_image_dose_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.MtfFilter.FixedImageDose", STACK_KEY, FIRST_AXIS_KEY),
            ))),
            stack_mtf_filter_fixed_image_dose_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.MtfFilter.FixedImageDose", STACK_KEY, SECOND_AXIS_KEY),
            ))),
            stack_mtf_filter_dose_weighting_file_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.MtfFilter.DoseWeightingFile",
                    STACK_KEY, FIRST_AXIS_KEY
                ),
            ))),
            stack_mtf_filter_dose_weighting_file_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!(
                    "{}.{}.MtfFilter.DoseWeightingFile",
                    STACK_KEY, SECOND_AXIS_KEY
                ),
            ))),
            stack_mtf_filter_type_of_dose_file_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.MtfFilter.TypeOfDoseFile",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            stack_mtf_filter_type_of_dose_file_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.MtfFilter.TypeOfDoseFile",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            stack_mtf_filter_voltage_200_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.MtfFilter.Voltage.200",
                STACK_KEY, FIRST_AXIS_KEY
            ))),
            stack_mtf_filter_voltage_200_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.MtfFilter.Voltage.200",
                STACK_KEY, SECOND_AXIS_KEY
            ))),
            stack_mtf_filter_optimal_dose_scaling_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    &format!(
                        "{}.{}.MtfFilter.OptimalDoseScaling",
                        STACK_KEY, FIRST_AXIS_KEY
                    ),
                ),
            ),
            stack_mtf_filter_optimal_dose_scaling_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    &format!(
                        "{}.{}.MtfFilter.OptimalDoseScaling",
                        STACK_KEY, SECOND_AXIS_KEY
                    ),
                ),
            ),
            stack_mtf_filter_bidirectional_num_views_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    &format!(
                        "{}.{}.MtfFilter.BidirectionalNumViews",
                        STACK_KEY, FIRST_AXIS_KEY
                    ),
                ),
            ),
            stack_mtf_filter_bidirectional_num_views_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Integer,
                    &format!(
                        "{}.{}.MtfFilter.BidirectionalNumViews",
                        STACK_KEY, SECOND_AXIS_KEY
                    ),
                ),
            ),
            use_stack_ctf_phase_flip_x_axis_tilt_a: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.Use.CtfPhaseFlip.XAxisTilt",
                    STACK_KEY, FIRST_AXIS_KEY
                ),
            )),
            use_stack_ctf_phase_flip_x_axis_tilt_b: Mutex::new(EtomoBoolean2::new_with_name(
                &format!(
                    "{}.{}.Use.CtfPhaseFlip.XAxisTilt",
                    STACK_KEY, SECOND_AXIS_KEY
                ),
            )),
            stack_ctf_phase_flip_x_axis_tilt_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.CtfPhaseFlip.XAxisTilt", STACK_KEY, FIRST_AXIS_KEY),
            )),
            stack_ctf_phase_flip_x_axis_tilt_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.CtfPhaseFlip.XAxisTilt", STACK_KEY, SECOND_AXIS_KEY),
            )),
            stack_ctf_phase_flip_scale_by_ctf_power_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    &format!(
                        "{}.{}.CtfPhaseFlip.ScaleByCtfPower",
                        STACK_KEY, FIRST_AXIS_KEY
                    ),
                ),
            ),
            stack_ctf_phase_flip_scale_by_ctf_power_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    &format!(
                        "{}.{}.CtfPhaseFlip.ScaleByCtfPower",
                        STACK_KEY, SECOND_AXIS_KEY
                    ),
                ),
            ),
            gen_super_sample_factor_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.SuperSampleFactor",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_super_sample_factor_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.SuperSampleFactor",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_expand_input_lines_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.ExpandInputLines",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_expand_input_lines_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.ExpandInputLines",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_sirt_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Sirt",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_sirt_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.Sirt",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_ctf_3d_old_style_xtilting_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.CTF_3D.{}.OldStyleXtilting",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_ctf_3d_old_style_xtilting_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.CTF_3D.{}.OldStyleXtilting",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_ctf_3d_vertical_slices_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.CTF_3D.{}.VerticalSlices",
                GEN_KEY, FIRST_AXIS_KEY
            ))),
            gen_ctf_3d_vertical_slices_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.CTF_3D.{}.VerticalSlices",
                GEN_KEY, SECOND_AXIS_KEY
            ))),
            gen_ctf_3d_fourier_reduce_by_factor_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                &format!(
                    "{}.CTF_3D.{}.FourierReduceByFactor",
                    GEN_KEY, FIRST_AXIS_KEY
                ),
            )),
            gen_ctf_3d_fourier_reduce_by_factor_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                &format!(
                    "{}.CTF_3D.{}.FourierReduceByFactor",
                    GEN_KEY, SECOND_AXIS_KEY
                ),
            )),
            subtomo_reorientation_type_none: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.ReorientationType.None",
                SUBTOMO_KEY
            ))),
            subtomo_reorientation_type_flipped: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.ReorientationType.Flipped",
                SUBTOMO_KEY
            ))),
            subtomo_reorientation_type_rotated: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.ReorientationType.Rotated",
                SUBTOMO_KEY
            ))),
            subtomo_make_volume_stacks: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                &format!("{}.MakeVolumeStacks", SUBTOMO_KEY),
            )),
            subtomo_extent_of_z_levels_in_nm: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                &format!("{}.ExtentOfZLevelsInNm", SUBTOMO_KEY),
            )),
            subtomo_new_aligned_binning: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                &format!("{}.NewAlignedBinning", SUBTOMO_KEY),
            )),
            subtomo_fourier_reduce_by_factor: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Integer,
                &format!("{}.FourierReduceByFactor", SUBTOMO_KEY),
            )),
            reduce_filt_vol_reduction_factor: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.ReductionFactor", REDUCE_FILT_VOL_KEY),
            )),
            reduce_filt_vol_z_reduction_factor: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.ZReductionFactor", REDUCE_FILT_VOL_KEY),
            )),
            reduce_filt_vol_low_pass_radius_sigma: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.LowPassRadiusSigma", REDUCE_FILT_VOL_KEY),
            ))),
            reduce_filt_vol_deconvolution_strength: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    &format!("{}.DeconvolutionStrength", REDUCE_FILT_VOL_KEY),
                ),
            ),
            reduce_filt_vol_snr_falloff: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.SNRFalloff", REDUCE_FILT_VOL_KEY),
            )),
            reduce_filt_vol_high_pass_nyquist: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.HighPassNyquist", REDUCE_FILT_VOL_KEY),
            )),
            reduce_filt_vol_defocus_in_microns: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.DefocusInMicrons", REDUCE_FILT_VOL_KEY),
            )),
            reduce_filt_vol_phase_shift: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.PhaseShift", REDUCE_FILT_VOL_KEY),
            )),
            alt_tomo_rootname_to_process: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.RootnameToProcess",
                ALT_TOMO_SETUP_KEY
            )))),
            alt_tomo_trim_volume: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.TrimVolume",
                ALT_TOMO_SETUP_KEY
            ))),
            alt_tomo_archive_orig_stack: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.PreprocessForExtremes",
                ALT_TOMO_SETUP_KEY
            ))),
            ctf_3d_setup_slab_thickness_in_nm_set: Mutex::new(EtomoBoolean2::new_with_name(
                "ctf3dsetup.SlabThicknessInNmSet",
            )),
        };
        meta_data.binning.lock().unwrap().set_display_value_int(1);
        // `squeezevolParam = new SqueezevolParam(manager); transferfidParamA = new
        // TransferfidParam(manager, AxisID.FIRST); transferfidParamB = new
        // TransferfidParam(manager, AxisID.SECOND);`
        if let Some(manager) = manager {
            *meta_data.squeezevol_param.lock().unwrap() = Some(SqueezevolParam::new(manager));
            *meta_data.transferfid_param_a.lock().unwrap() =
                Some(TransferfidParam::new(manager, AxisID::First));
            *meta_data.transferfid_param_b.lock().unwrap() =
                Some(TransferfidParam::new(manager, AxisID::Second));
        }
        *meta_data.base.file_extension.lock().unwrap() = DataFileType::Recon
            .extension()
            .map(|extension| extension.to_string());
        meta_data
            .use_z_factors_a
            .lock()
            .unwrap()
            .set_display_value_boolean(true);
        meta_data
            .use_z_factors_b
            .lock()
            .unwrap()
            .set_display_value_boolean(true);
        meta_data
            .sample_thickness_a
            .lock()
            .unwrap()
            .set_display_value_int(DEFAULT_SAMPLE_THICKNESS);
        meta_data
            .sample_thickness_b
            .lock()
            .unwrap()
            .set_display_value_int(DEFAULT_SAMPLE_THICKNESS);
        meta_data
            .no_beam_tilt_selected_a
            .lock()
            .unwrap()
            .set_display_value_boolean(true); // backwards compatibility
        meta_data
            .no_beam_tilt_selected_b
            .lock()
            .unwrap()
            .set_display_value_boolean(true); // backwards compatibility
        meta_data
            .track_use_raptor_a
            .lock()
            .unwrap()
            .set_boolean(false);
        meta_data
            .track_raptor_use_raw_stack_a
            .lock()
            .unwrap()
            .set_boolean(false);

        meta_data
            .size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .set_integer_type_array(&[true, true]);
        meta_data
            .size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .set_integer_type_array(&[true, true]);

        meta_data
            .size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .set_properties_key(Some("A.SizeToOutputInXandY"));
        meta_data
            .size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .set_default();
        meta_data
            .size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .set_properties_key(Some("B.SizeToOutputInXandY"));
        meta_data
            .size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .set_default();

        meta_data
            .pos_binning_a
            .lock()
            .unwrap()
            .set_display_value_int(3);
        meta_data
            .pos_binning_b
            .lock()
            .unwrap()
            .set_display_value_int(3);
        meta_data
            .stack_binning_a
            .lock()
            .unwrap()
            .set_display_value_int(1);
        meta_data
            .stack_binning_b
            .lock()
            .unwrap()
            .set_display_value_int(1);

        meta_data
            .stack_erase_gold_model_use_fid_a
            .lock()
            .unwrap()
            .set_display_value_boolean(ERASE_GOLD_MODEL_USE_FID_DEFAULT);
        meta_data
            .stack_erase_gold_model_use_fid_b
            .lock()
            .unwrap()
            .set_display_value_boolean(ERASE_GOLD_MODEL_USE_FID_DEFAULT);
        meta_data
            .gen_back_projection_a
            .lock()
            .unwrap()
            .set_display_value_boolean(true);
        meta_data
            .gen_back_projection_b
            .lock()
            .unwrap()
            .set_display_value_boolean(true);
        meta_data
            .stack_ctf_auto_fit_range_and_step_a
            .lock()
            .unwrap()
            .set_properties_key(Some(&format!(
                "{}.{}.CTF.AutoFit.RangeAndStep",
                STACK_KEY, FIRST_AXIS_KEY
            )));
        // Upstream bug fixed in translation (MetaData.java:1252): the Java sets
        // the key of stackCtfAutoFitRangeAndStepA a second time (to the B key),
        // so A is stored under "Stack.B..." and B, with no key, under the bare
        // group ("Setup=-Infinity,-Infinity" in the .edf).  Each field gets its
        // own key here.
        meta_data
            .stack_ctf_auto_fit_range_and_step_b
            .lock()
            .unwrap()
            .set_properties_key(Some(&format!(
                "{}.{}.CTF.AutoFit.RangeAndStep",
                STACK_KEY, SECOND_AXIS_KEY
            )));

        meta_data
            .track_method_a
            .lock()
            .unwrap()
            .set(Some(&tracking_method::SEED.to_string()));
        meta_data
            .track_method_b
            .lock()
            .unwrap()
            .set(Some(&tracking_method::SEED.to_string()));
        meta_data
            .track_seed_model_auto_a
            .lock()
            .unwrap()
            .set_boolean(true);
        meta_data
            .track_seed_model_transfer_b
            .lock()
            .unwrap()
            .set_boolean(true);
        meta_data
            .twodir_a
            .lock()
            .unwrap()
            .set_double(TWO_DIR_DEFAULT);
        meta_data
            .twodir_b
            .lock()
            .unwrap()
            .set_double(TWO_DIR_DEFAULT);
        meta_data
            .dose_sym_a
            .lock()
            .unwrap()
            .set_double(DOSE_SYM_DEFAULT);
        meta_data
            .dose_sym_b
            .lock()
            .unwrap()
            .set_double(DOSE_SYM_DEFAULT);
        meta_data
            .extra_thickness_cryo_a
            .lock()
            .unwrap()
            .set_display_value_double(EXTRA_THICKNESS_CRYO_DEFAULT);
        meta_data
            .extra_thickness_cryo_b
            .lock()
            .unwrap()
            .set_display_value_double(EXTRA_THICKNESS_CRYO_DEFAULT);
        meta_data
            .positioning_new_dialog_a
            .lock()
            .unwrap()
            .set_boolean(true);
        meta_data
            .positioning_new_dialog_b
            .lock()
            .unwrap()
            .set_boolean(true);
        meta_data
            .post_trimvol_convert_to_bytes
            .lock()
            .unwrap()
            .set_display_value_boolean(true);
        meta_data
            .stack_ctf_phase_flip_scale_by_ctf_power_a
            .lock()
            .unwrap()
            .set_display_value_double(CTF_SCALE_BY_CTF_POWER_DEFAULT);
        meta_data
            .stack_ctf_phase_flip_scale_by_ctf_power_b
            .lock()
            .unwrap()
            .set_display_value_double(CTF_SCALE_BY_CTF_POWER_DEFAULT);
        meta_data
            .subtomo_extent_of_z_levels_in_nm
            .lock()
            .unwrap()
            .set_display_value_int(SUBTOMO_SETUP_PARAM_EXTENT_OF_ZLEVELS_IN_NM_DEFAULT);
        meta_data
    }

    /// Java `setDatasetName`.  Set the dataset name, trimming any white space from the
    /// beginning and end of the string.
    pub fn set_dataset_name(&self, file_name: &str) {
        // Trim off the path, if it exists
        *self.dataset_name.lock().unwrap() = java_io_file_get_name(file_name);
        self.fix_dataset_name();
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        utilities::manager_stamp(None, Some(&dataset_name));
    }

    /// Java private `fixDatasetName`.  Remove the ".st", "a.st", "b.st", .mrc, a.mrc, or
    /// b.mrc as appropriate to the file name.  Store the extension in
    /// origImageStackExt.
    fn fix_dataset_name(&self) {
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        let extension = Extension::get_instance(&dataset_name);
        if let Some(extension) = extension {
            self.raw_image_stack_ext
                .lock()
                .unwrap()
                .set(Some(&extension.to_string()));
            let locked = self.orig_raw_image_stack_ext_lock.lock().unwrap().is();
            if !locked {
                if extension.is_input_image_file() {
                    self.orig_raw_image_stack_ext
                        .lock()
                        .unwrap()
                        .set(Some(&extension.to_string()));
                } else {
                    // No valid extension available
                    self.orig_raw_image_stack_ext.lock().unwrap().reset();
                }
            }
        }
        let dual = *self.base.axis_type.lock().unwrap() != AxisType::SingleAxis;
        let mut err_msg = String::new();
        if !dataset_tool::is_valid_input_image_file(&dataset_name, dual, Some(&mut err_msg)) {
            // Something is wrong with the file name - report an error message.
            self.append_message(&format!("{}\n", err_msg));
        }
        // Use this file whether or not it is valid.
        *self.dataset_name.lock().unwrap() =
            dataset_tool::get_dataset_name(Some(&dataset_name), dual).unwrap_or_default();
    }

    /// Java `setLambdaForSmoothing`.
    pub fn set_lambda_for_smoothing(&self, input: Option<&str>) {
        self.lambda_for_smoothing.lock().unwrap().set_string(input);
    }

    /// Java `getLambdaForSmoothing`.
    pub fn get_lambda_for_smoothing(&self) -> String {
        self.lambda_for_smoothing.lock().unwrap().to_string()
    }

    /// Java `setLambdaForSmoothingList`.
    pub fn set_lambda_for_smoothing_list(&self, input: Option<&str>) {
        self.lambda_for_smoothing_list.lock().unwrap().set(input);
    }

    /// Java `getLambdaForSmoothingList`.
    pub fn get_lambda_for_smoothing_list(&self) -> String {
        self.lambda_for_smoothing_list.lock().unwrap().to_string()
    }

    /// Java `isLambdaForSmoothingListEmpty`.
    pub fn is_lambda_for_smoothing_list_empty(&self) -> bool {
        self.lambda_for_smoothing_list.lock().unwrap().is_empty()
    }

    /// Java `setTransferfidAFields(TransferfidParam)`.
    pub fn set_transferfid_a_fields(&self, param: &TransferfidParam) {
        if let Some(transferfid_param) = self.transferfid_param_a.lock().unwrap().as_mut() {
            transferfid_param.set_storable_fields(param);
        }
    }

    /// Java `setTransferfidBFields(TransferfidParam)`.
    pub fn set_transferfid_b_fields(&self, param: &TransferfidParam) {
        if let Some(transferfid_param) = self.transferfid_param_b.lock().unwrap().as_mut() {
            transferfid_param.set_storable_fields(param);
        }
    }

    /// Java `setPostTrimvolScaleXMin`.
    pub fn set_post_trimvol_scale_x_min(&self, input: Option<&str>) {
        self.post_trimvol_scale_x_min
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostTrimvolScaleXMax`.
    pub fn set_post_trimvol_scale_x_max(&self, input: Option<&str>) {
        self.post_trimvol_scale_x_max
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostTrimvolScaleYMin`.
    pub fn set_post_trimvol_scale_y_min(&self, input: Option<&str>) {
        self.post_trimvol_scale_y_min
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostTrimvolScaleYMax`.
    pub fn set_post_trimvol_scale_y_max(&self, input: Option<&str>) {
        self.post_trimvol_scale_y_max
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostTrimvolSectionScaleMin`.
    pub fn set_post_trimvol_section_scale_min(&self, input: Option<&str>) {
        self.post_trimvol_section_scale_min
            .lock()
            .unwrap()
            .set(input);
    }

    /// Java `setPostTrimvolSectionScaleMax`.
    pub fn set_post_trimvol_section_scale_max(&self, input: Option<&str>) {
        self.post_trimvol_section_scale_max
            .lock()
            .unwrap()
            .set(input);
    }

    /// Java `setPostTrimvolXMin`.
    pub fn set_post_trimvol_x_min(&self, input: Option<&str>) {
        self.post_trimvol_x_min.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolXMax`.
    pub fn set_post_trimvol_x_max(&self, input: Option<&str>) {
        self.post_trimvol_x_max.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolYMin`.
    pub fn set_post_trimvol_y_min(&self, input: Option<&str>) {
        self.post_trimvol_y_min.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolYMax`.
    pub fn set_post_trimvol_y_max(&self, input: Option<&str>) {
        self.post_trimvol_y_max.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolZMin`.
    pub fn set_post_trimvol_z_min(&self, input: Option<&str>) {
        self.post_trimvol_z_min.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolZMax`.
    pub fn set_post_trimvol_z_max(&self, input: Option<&str>) {
        self.post_trimvol_z_max.lock().unwrap().set(input);
    }

    /// Java `setSampleThickness`.
    pub fn set_sample_thickness(&self, axis_id: AxisID, thickness: Option<&str>) {
        if axis_id == AxisID::Second {
            self.sample_thickness_b
                .lock()
                .unwrap()
                .set_string(thickness);
        } else {
            self.sample_thickness_a
                .lock()
                .unwrap()
                .set_string(thickness);
        }
    }

    /// Java `setBackupDirectory`.  Set the backup directory, trimming any white space
    /// from the beginning and end of the string.
    pub fn set_backup_directory(&self, backup_dir: Option<&str>) {
        match backup_dir {
            None => *self.backup_directory.lock().unwrap() = String::new(),
            Some(backup_dir) => {
                *self.backup_directory.lock().unwrap() =
                    java_lang_string_trim(backup_dir).to_string()
            }
        }
    }

    /// Java `setDistortionFile`.
    pub fn set_distortion_file(&self, distortion_file: Option<&str>) {
        *self.distortion_file.lock().unwrap() = distortion_file.map(|s| s.to_string());
    }

    /// Java `setEraseBeadsInitialized`.
    pub fn set_erase_beads_initialized(&self, input: bool) {
        self.erase_beads_initialized
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setTrackSeedModelManual`.
    pub fn set_track_seed_model_manual(&self, input: bool, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_seed_model_manual_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.track_seed_model_manual_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setTrackSeedModelAuto`.
    pub fn set_track_seed_model_auto(&self, input: bool, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_seed_model_auto_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.track_seed_model_auto_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setTrackSeedModelTransfer`.
    pub fn set_track_seed_model_transfer(&self, input: bool, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_seed_model_transfer_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.track_seed_model_transfer_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setTrackExcludeInsideAreas`.
    pub fn set_track_exclude_inside_areas(&self, input: bool, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_exclude_inside_areas_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.track_exclude_inside_areas_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setTrackJustFindShiftsNearZero`.
    pub fn set_track_just_find_shifts_near_zero(&self, input: Option<&str>, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_just_find_shifts_near_zero_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.track_just_find_shifts_near_zero_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setTrackTargetNumberOfBeads`.
    pub fn set_track_target_number_of_beads(&self, input: Option<&str>, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_target_number_of_beads_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.track_target_number_of_beads_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setTrackTargetDensityOfBeads`.
    pub fn set_track_target_density_of_beads(&self, input: Option<&str>, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_target_density_of_beads_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.track_target_density_of_beads_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setTrackAdvanced`.
    pub fn set_track_advanced(&self, input: bool, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.track_advanced_b.lock().unwrap().set_boolean(input);
        } else {
            self.track_advanced_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setStack3dFindThickness`.
    pub fn set_stack_3d_find_thickness(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_3d_find_thickness_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.stack_3d_find_thickness_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setSetFEIPixelSize`.
    pub fn set_set_fei_pixel_size(&self, input: bool) {
        self.set_fei_pixel_size.lock().unwrap().set_boolean(input);
    }

    /// Java `setPostTrimvolNewStyleZ`.
    pub fn set_post_trimvol_new_style_z(&self, ui_z_min: Option<&str>, ui_z_max: Option<&str>) {
        let is_null = self.post_trimvol_new_style_z.lock().unwrap().is_null();
        if is_null || !self.post_trimvol_new_style_z.lock().unwrap().is() {
            let z_min_differs = !self.post_trimvol_z_min.lock().unwrap().equals(ui_z_min);
            let differs =
                z_min_differs || !self.post_trimvol_z_max.lock().unwrap().equals(ui_z_max);
            self.post_trimvol_new_style_z
                .lock()
                .unwrap()
                .set_boolean(differs);
        }
    }

    /// Java `setPostTrimvolScalingNewStyleZ`.
    pub fn set_post_trimvol_scaling_new_style_z(
        &self,
        ui_z_min: Option<&str>,
        ui_z_max: Option<&str>,
    ) {
        let is_null = self
            .post_trimvol_scaling_new_style_z
            .lock()
            .unwrap()
            .is_null();
        if is_null || !self.post_trimvol_scaling_new_style_z.lock().unwrap().is() {
            let min_differs = !self
                .post_trimvol_section_scale_min
                .lock()
                .unwrap()
                .equals(ui_z_min);
            let differs = min_differs
                || !self
                    .post_trimvol_section_scale_max
                    .lock()
                    .unwrap()
                    .equals(ui_z_max);
            self.post_trimvol_scaling_new_style_z
                .lock()
                .unwrap()
                .set_boolean(differs);
        }
    }

    /// Java `setMagGradientFile`.
    pub fn set_mag_gradient_file(&self, mag_gradient_file: Option<&str>) {
        *self.mag_gradient_file.lock().unwrap() = mag_gradient_file.map(|s| s.to_string());
    }

    /// Java `setAdjustedFocusA`.
    pub fn set_adjusted_focus_a(&self, adjusted_focus: bool) {
        self.adjusted_focus_a
            .lock()
            .unwrap()
            .set_boolean(adjusted_focus);
    }

    /// Java `setAdjustedFocusB`.
    pub fn set_adjusted_focus_b(&self, adjusted_focus: bool) {
        self.adjusted_focus_b
            .lock()
            .unwrap()
            .set_boolean(adjusted_focus);
    }

    /// Java `setAntialiasFilter`.
    pub fn set_antialias_filter(
        &self,
        dialog_type: DialogType,
        axis_id: AxisID,
        input: Option<&ConstEtomoNumber>,
    ) {
        if dialog_type == DialogType::CoarseAlignment {
            if axis_id == AxisID::Second {
                self.coarse_antialias_filter_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(input);
            } else {
                self.coarse_antialias_filter_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(input);
            }
        } else if dialog_type == DialogType::FinalAlignedStack {
            if axis_id == AxisID::Second {
                self.stack_antialias_filter_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(input);
            } else {
                self.stack_antialias_filter_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(input);
            }
        }
    }

    /// Java `setAxisType`.
    pub fn set_axis_type(&self, at: AxisType) {
        *self.base.axis_type.lock().unwrap() = at;
        self.set_axis_prepends();
    }

    /// Java `setOrigViewsWithMagChanges`.
    pub fn set_orig_views_with_mag_changes(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            if self.orig_views_with_mag_changes_set_b.lock().unwrap().is() {
                return;
            }
            self.orig_views_with_mag_changes_b
                .lock()
                .unwrap()
                .set_boolean(input);
            self.orig_views_with_mag_changes_set_b
                .lock()
                .unwrap()
                .set_boolean(true);
        } else {
            if self.orig_views_with_mag_changes_set_a.lock().unwrap().is() {
                return;
            }
            self.orig_views_with_mag_changes_a
                .lock()
                .unwrap()
                .set_boolean(input);
            self.orig_views_with_mag_changes_set_a
                .lock()
                .unwrap()
                .set_boolean(true);
        }
    }

    /// Java `setViewType`.
    pub fn set_view_type(&self, vt: ViewType) {
        *self.view_type.lock().unwrap() = vt;
    }

    /// Java `setPixelSize`.
    pub fn set_pixel_size_double(&self, pixel_size: f64) {
        *self.pixel_size.lock().unwrap() = pixel_size;
    }

    /// Java `setHalfFloatModeOutput`.
    pub fn set_half_float_mode_output(&self, input: Option<i32>) {
        self.half_float_mode_output
            .lock()
            .unwrap()
            .set_number(input.map(Number::Integer));
    }

    /// Java `setPixelSize(String)`.
    ///
    /// Upstream bug fixed in translation (MetaData.java:1589): `Double.parseDouble`
    /// throws `NumberFormatException` for a non-numeric string, which no caller
    /// catches.  A malformed string now sets the pixel size to NaN, the value the
    /// source gives a blank one.
    pub fn set_pixel_size_string(&self, pixel_size: Option<&str>) {
        match pixel_size {
            Some(pixel_size) if !java_lang_string_matches_whitespace(pixel_size) => {
                *self.pixel_size.lock().unwrap() =
                    java_lang_double_value_of(pixel_size).unwrap_or(f64::NAN);
            }
            _ => {
                *self.pixel_size.lock().unwrap() = f64::NAN;
            }
        }
    }

    /// Java `setUseLocalAlignments`.
    pub fn set_use_local_alignments(&self, axis_id: AxisID, state: bool) {
        if axis_id == AxisID::Second {
            *self.use_local_alignments_b.lock().unwrap() = state;
        } else {
            *self.use_local_alignments_a.lock().unwrap() = state;
        }
    }

    /// Java `setBStackProcessed(boolean)`.
    pub fn set_b_stack_processed_boolean(&self, b_stack_processed: bool) {
        let mut field = self.b_stack_processed.lock().unwrap();
        if field.is_none() {
            *field = Some(EtomoBoolean2::new_with_name(B_STACK_PROCESSED_GROUP));
        }
        field.as_mut().unwrap().set_boolean(b_stack_processed);
    }

    /// Java `setBStackProcessed(String)`.
    pub fn set_b_stack_processed_string(&self, b_stack_processed: Option<&str>) {
        let mut field = self.b_stack_processed.lock().unwrap();
        if field.is_none() {
            *field = Some(EtomoBoolean2::new_with_name(B_STACK_PROCESSED_GROUP));
        }
        field.as_mut().unwrap().set_string(b_stack_processed);
    }

    /// Java `setSizeToOutputInXandY`.
    pub fn set_size_to_output_in_x_and_y(
        &self,
        axis_id: AxisID,
        size: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        if axis_id == AxisID::Second {
            self.size_to_output_in_x_and_y_b
                .lock()
                .unwrap()
                .validate_and_set(size)?;
        } else {
            self.size_to_output_in_x_and_y_a
                .lock()
                .unwrap()
                .validate_and_set(size)?;
        }
        Ok(())
    }

    /// Java `setOrigScopeTemplate(File)`.  The `File` is its path; `getAbsolutePath()`
    /// is `java_io_file_get_absolute_path`.
    pub fn set_orig_scope_template(&self, input: Option<&str>) {
        match input {
            None => self.orig_scope_template.lock().unwrap().reset(),
            Some(input) => self
                .orig_scope_template
                .lock()
                .unwrap()
                .set(Some(&java_io_file_get_absolute_path(input))),
        }
    }

    /// Java `setOrigSystemTemplate(File)`.
    pub fn set_orig_system_template(&self, input: Option<&str>) {
        match input {
            None => self.orig_system_template.lock().unwrap().reset(),
            Some(input) => self
                .orig_system_template
                .lock()
                .unwrap()
                .set(Some(&java_io_file_get_absolute_path(input))),
        }
    }

    /// Java `setOrigUserTemplate(File)`.
    pub fn set_orig_user_template(&self, input: Option<&str>) {
        match input {
            None => self.orig_user_template.lock().unwrap().reset(),
            Some(input) => self
                .orig_user_template
                .lock()
                .unwrap()
                .set(Some(&java_io_file_get_absolute_path(input))),
        }
    }

    /// Java `setStackCtfAutoFitRangeAndStep`.
    pub fn set_stack_ctf_auto_fit_range_and_step(
        &self,
        axis_id: AxisID,
        input: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        if axis_id == AxisID::Second {
            self.stack_ctf_auto_fit_range_and_step_b
                .lock()
                .unwrap()
                .validate_and_set(input)?;
        } else {
            self.stack_ctf_auto_fit_range_and_step_a
                .lock()
                .unwrap()
                .validate_and_set(input)?;
        }
        Ok(())
    }

    /// Java `setPosBinning`.
    pub fn set_pos_binning_int(&self, axis_id: AxisID, binning: i32) {
        if axis_id == AxisID::Second {
            self.pos_binning_b.lock().unwrap().set_int(binning);
        } else {
            self.pos_binning_a.lock().unwrap().set_int(binning);
        }
    }

    /// Java `setPosBinning`.
    pub fn set_pos_binning_string(&self, axis_id: AxisID, binning: Option<&str>) {
        if axis_id == AxisID::Second {
            self.pos_binning_b.lock().unwrap().set_string(binning);
        } else {
            self.pos_binning_a.lock().unwrap().set_string(binning);
        }
    }

    /// Java `setStackBinning`.
    pub fn set_stack_binning_int(&self, axis_id: AxisID, binning: i32) {
        if axis_id == AxisID::Second {
            self.stack_binning_b.lock().unwrap().set_int(binning);
        } else {
            self.stack_binning_a.lock().unwrap().set_int(binning);
        }
    }

    /// Java `setStackBinning`.
    pub fn set_stack_binning_string(&self, axis_id: AxisID, binning: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_binning_b.lock().unwrap().set_string(binning);
        } else {
            self.stack_binning_a.lock().unwrap().set_string(binning);
        }
    }

    /// Java `setStack3dFindBinning`.
    pub fn set_stack_3d_find_binning_int(&self, axis_id: AxisID, binning: i32) {
        if axis_id == AxisID::Second {
            self.stack_3d_find_binning_b
                .lock()
                .unwrap()
                .set_int(binning);
        } else {
            self.stack_3d_find_binning_a
                .lock()
                .unwrap()
                .set_int(binning);
        }
    }

    /// Java `setStack3dFindBinning`.
    pub fn set_stack_3d_find_binning_string(&self, axis_id: AxisID, binning: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_3d_find_binning_b
                .lock()
                .unwrap()
                .set_string(binning);
        } else {
            self.stack_3d_find_binning_a
                .lock()
                .unwrap()
                .set_string(binning);
        }
    }

    /// Java `setPostCurTab`.
    pub fn set_post_cur_tab(&self, input: i32) {
        self.post_cur_tab.lock().unwrap().set_int(input);
    }

    /// Java `setGenCurTab`.
    pub fn set_gen_cur_tab(&self, input: i32) {
        self.gen_cur_tab.lock().unwrap().set_int(input);
    }

    /// Java `setPostExists`.
    pub fn set_post_exists(&self, input: bool) {
        self.post_exists.lock().unwrap().set_boolean(input);
    }

    /// Java `setCombineVolcombineParallel(boolean)`.
    pub fn set_combine_volcombine_parallel_boolean(&self, combine_volcombine_parallel: bool) {
        let mut field = self.combine_volcombine_parallel.lock().unwrap();
        if field.is_none() {
            *field = Some(EtomoBoolean2::new_with_name(
                COMBINE_VOLCOMBINE_PARALLEL_GROUP.as_str(),
            ));
        }
        field
            .as_mut()
            .unwrap()
            .set_boolean(combine_volcombine_parallel);
    }

    /// Java `setCombineVolcombineParallel(String)`.
    pub fn set_combine_volcombine_parallel_string(
        &self,
        combine_volcombine_parallel: Option<&str>,
    ) {
        let mut field = self.combine_volcombine_parallel.lock().unwrap();
        if field.is_none() {
            *field = Some(EtomoBoolean2::new_with_name(
                COMBINE_VOLCOMBINE_PARALLEL_GROUP.as_str(),
            ));
        }
        field
            .as_mut()
            .unwrap()
            .set_string(combine_volcombine_parallel);
    }

    /// Java `setTiltParallel`.
    pub fn set_tilt_parallel(&self, axis_id: AxisID, panel_id: PanelId, tilt_parallel: bool) {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                let mut field = self.tomo_gen_tilt_parallel_b.lock().unwrap();
                if field.is_none() {
                    *field = Some(EtomoBoolean2::new_with_name(
                        TOMO_GEN_B_TILT_PARALLEL_GROUP.as_str(),
                    ));
                }
                field.as_mut().unwrap().set_boolean(tilt_parallel);
            } else {
                let mut field = self.tomo_gen_tilt_parallel_a.lock().unwrap();
                if field.is_none() {
                    *field = Some(EtomoBoolean2::new_with_name(
                        TOMO_GEN_A_TILT_PARALLEL_GROUP.as_str(),
                    ));
                }
                field.as_mut().unwrap().set_boolean(tilt_parallel);
            }
        } else if panel_id == PanelId::Tilt3dFind {
            if axis_id == AxisID::Second {
                let mut field = self.tilt_3d_find_tilt_parallel_b.lock().unwrap();
                if field.is_none() {
                    *field = Some(EtomoBoolean2::new_with_name(
                        TILT_3D_FIND_B_TILT_PARALLEL_KEY.as_str(),
                    ));
                }
                field.as_mut().unwrap().set_boolean(tilt_parallel);
            } else {
                let mut field = self.tilt_3d_find_tilt_parallel_a.lock().unwrap();
                if field.is_none() {
                    *field = Some(EtomoBoolean2::new_with_name(
                        TILT_3D_FIND_A_TILT_PARALLEL_KEY.as_str(),
                    ));
                }
                field.as_mut().unwrap().set_boolean(tilt_parallel);
            }
        }
    }

    /// Java `setFinalStackCtfCorrectionParallel(AxisID, boolean)`.
    pub fn set_final_stack_ctf_correction_parallel_boolean(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            let mut field = self.final_stack_ctf_correction_parallel_b.lock().unwrap();
            if field.is_none() {
                *field = Some(EtomoBoolean2::new_with_name(
                    FINAL_STACK_B_CTF_CORRECTION_PARALLEL_GROUP.as_str(),
                ));
            }
            field.as_mut().unwrap().set_boolean(input);
        } else {
            let mut field = self.final_stack_ctf_correction_parallel_a.lock().unwrap();
            if field.is_none() {
                *field = Some(EtomoBoolean2::new_with_name(
                    FINAL_STACK_A_CTF_CORRECTION_PARALLEL_GROUP.as_str(),
                ));
            }
            field.as_mut().unwrap().set_boolean(input);
        }
    }

    /// Java `setDefaultParallel`.
    pub fn set_default_parallel(&self, default_parallel: bool) {
        self.default_parallel
            .lock()
            .unwrap()
            .set_boolean(default_parallel);
    }

    /// Java `setDefaultGpuProcessing`.
    pub fn set_default_gpu_processing(&self, input: bool) {
        self.default_gpu_processing
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java private `setTomoGenTiltParallel`.
    fn set_tomo_gen_tilt_parallel(&self, axis_id: AxisID, tomo_gen_tilt_parallel: Option<&str>) {
        if axis_id == AxisID::Second {
            let mut field = self.tomo_gen_tilt_parallel_b.lock().unwrap();
            if field.is_none() {
                *field = Some(EtomoBoolean2::new_with_name(
                    TOMO_GEN_B_TILT_PARALLEL_GROUP.as_str(),
                ));
            }
            field.as_mut().unwrap().set_string(tomo_gen_tilt_parallel);
        } else {
            let mut field = self.tomo_gen_tilt_parallel_a.lock().unwrap();
            if field.is_none() {
                *field = Some(EtomoBoolean2::new_with_name(
                    TOMO_GEN_A_TILT_PARALLEL_GROUP.as_str(),
                ));
            }
            field.as_mut().unwrap().set_string(tomo_gen_tilt_parallel);
        }
    }

    /// Java `setFinalStackCtfCorrectionParallel(AxisID, String)`.
    pub fn set_final_stack_ctf_correction_parallel_string(
        &self,
        axis_id: AxisID,
        input: Option<&str>,
    ) {
        if axis_id == AxisID::Second {
            let mut field = self.final_stack_ctf_correction_parallel_b.lock().unwrap();
            if field.is_none() {
                *field = Some(EtomoBoolean2::new_with_name(
                    FINAL_STACK_B_CTF_CORRECTION_PARALLEL_GROUP.as_str(),
                ));
            }
            field.as_mut().unwrap().set_string(input);
        } else {
            let mut field = self.final_stack_ctf_correction_parallel_a.lock().unwrap();
            if field.is_none() {
                *field = Some(EtomoBoolean2::new_with_name(
                    FINAL_STACK_A_CTF_CORRECTION_PARALLEL_GROUP.as_str(),
                ));
            }
            field.as_mut().unwrap().set_string(input);
        }
    }

    /// Java `setUseZFactors`.
    pub fn set_use_z_factors(&self, axis_id: AxisID, use_z_factors: bool) {
        if axis_id == AxisID::Second {
            self.use_z_factors_b
                .lock()
                .unwrap()
                .set_boolean(use_z_factors);
        } else {
            self.use_z_factors_a
                .lock()
                .unwrap()
                .set_boolean(use_z_factors);
        }
    }

    /// Java `setFiducialDiameter`.
    pub fn set_fiducial_diameter_double(&self, fiducial_diameter: f64) {
        *self.fiducial_diameter.lock().unwrap() = fiducial_diameter;
    }

    /// Java `setFiducialDiameter(String)`.
    ///
    /// Upstream bug fixed in translation (MetaData.java:1858): `Double.parseDouble`
    /// throws `NumberFormatException` for a non-numeric string.  A malformed string now
    /// sets NaN, the value the source gives a blank one.
    pub fn set_fiducial_diameter_string(&self, fiducial_diameter: Option<&str>) {
        match fiducial_diameter {
            Some(fiducial_diameter) if !java_lang_string_matches_whitespace(fiducial_diameter) => {
                *self.fiducial_diameter.lock().unwrap() =
                    java_lang_double_value_of(fiducial_diameter).unwrap_or(f64::NAN);
            }
            _ => {
                *self.fiducial_diameter.lock().unwrap() = f64::NAN;
            }
        }
    }

    /// Java `setImageRotation`.
    pub fn set_image_rotation(&self, rotation: Option<&str>, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.image_rotation_b.lock().unwrap().set_string(rotation);
        } else {
            self.image_rotation_a.lock().unwrap().set_string(rotation);
        }
    }

    /// Java `setBinning`.
    pub fn set_binning(&self, binning: Option<&str>) {
        self.binning.lock().unwrap().set_string(binning);
    }

    /// Java `setTiltAngleSpecA`.
    pub fn set_tilt_angle_spec_a(&self, tilt_angle_spec: TiltAngleSpec) {
        *self.tilt_angle_spec_a.lock().unwrap() = tilt_angle_spec;
    }

    /// Java `setExcludeProjections`.
    pub fn set_exclude_projections(&self, list: Option<&str>, axis_id: AxisID) {
        let list = match list {
            None => {
                if axis_id == AxisID::Second {
                    *self.exclude_projections_b.lock().unwrap() = None;
                } else {
                    *self.exclude_projections_a.lock().unwrap() = None;
                }
                return;
            }
            Some(list) => list,
        };
        // Strip whitespace.
        let array = java_lang_string_split(java_lang_string_trim(list), &WHITESPACE_PATTERN);
        if array.len() > 1 {
            let mut buffer = String::new();
            for i in 0..array.len() {
                buffer.push_str(&array[i]);
            }
            if axis_id == AxisID::Second {
                *self.exclude_projections_b.lock().unwrap() = Some(buffer);
            } else {
                *self.exclude_projections_a.lock().unwrap() = Some(buffer);
            }
        } else if axis_id == AxisID::Second {
            *self.exclude_projections_b.lock().unwrap() =
                Some(java_lang_string_trim(list).to_string());
        } else {
            *self.exclude_projections_a.lock().unwrap() =
                Some(java_lang_string_trim(list).to_string());
        }
        if axis_id == AxisID::Second {
            let mut field = self.exclude_projections_b.lock().unwrap();
            if java_lang_string_matches_whitespace(field.as_deref().unwrap()) {
                *field = None;
            }
        } else {
            let mut field = self.exclude_projections_a.lock().unwrap();
            if java_lang_string_matches_whitespace(field.as_deref().unwrap()) {
                *field = None;
            }
        }
    }

    /// Java `setTwodir`.
    pub fn set_twodir(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.twodir_b.lock().unwrap().set_string(input);
        } else {
            self.twodir_a.lock().unwrap().set_string(input);
        }
    }

    /// Java `setDoseSym`.
    pub fn set_dose_sym(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.dose_sym_b.lock().unwrap().set_string(input);
        } else {
            self.dose_sym_a.lock().unwrap().set_string(input);
        }
    }

    /// Java `setSeedAndTrackTab`.
    pub fn set_seed_and_track_tab(&self, axis_id: AxisID, input: i32) {
        if axis_id == AxisID::Second {
            self.seed_and_track_tab_b.lock().unwrap().set_int(input);
        } else {
            self.seed_and_track_tab_a.lock().unwrap().set_int(input);
        }
    }

    /// Java `setRaptorTab`.
    pub fn set_raptor_tab(&self, axis_id: AxisID, input: i32) {
        if axis_id == AxisID::Second {
            self.raptor_tab_b.lock().unwrap().set_int(input);
        } else {
            self.raptor_tab_a.lock().unwrap().set_int(input);
        }
    }

    /// Java `setIsTwodir`.
    pub fn set_is_twodir(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.is_twodir_b.lock().unwrap().set_boolean(input);
        } else {
            self.is_twodir_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setIsDoseSym`.
    pub fn set_is_dose_sym(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.is_dose_sym_b.lock().unwrap().set_boolean(input);
        } else {
            self.is_dose_sym_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setGenExists`.
    pub fn set_gen_exists(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_exists_b.lock().unwrap().set_boolean(input);
        } else {
            self.gen_exists_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setPosExists`.
    pub fn set_pos_exists(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.pos_exists_b.lock().unwrap().set_boolean(input);
        } else {
            self.pos_exists_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setGenBackProjection`.
    pub fn set_gen_back_projection(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_back_projection_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.gen_back_projection_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setGenFilterTrials`.
    pub fn set_gen_filter_trials(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_filter_trials_b.lock().unwrap().set_boolean(input);
        } else {
            self.gen_filter_trials_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setGenSirt`.
    pub fn set_gen_sirt(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_sirt_b.lock().unwrap().set_boolean(input);
        } else {
            self.gen_sirt_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setGenFilterTrialsFakeSIRTiterations`.
    pub fn set_gen_filter_trials_fake_sirt_iterations(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_filter_trials_fake_sirt_iterations_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.gen_filter_trials_fake_sirt_iterations_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setGenFilterTrialsExactObjectSizes`.
    pub fn set_gen_filter_trials_exact_object_sizes(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_filter_trials_exact_object_sizes_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.gen_filter_trials_exact_object_sizes_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setGenFilterTrialsHammingLikeStarts`.
    pub fn set_gen_filter_trials_hamming_like_starts(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_filter_trials_hamming_like_starts_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.gen_filter_trials_hamming_like_starts_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setGenCtf3dOldStyleXtilting`.
    pub fn set_gen_ctf_3d_old_style_xtilting(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_ctf_3d_old_style_xtilting_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.gen_ctf_3d_old_style_xtilting_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setGenCtf3dVerticalSlices`.
    pub fn set_gen_ctf_3d_vertical_slices(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_ctf_3d_vertical_slices_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.gen_ctf_3d_vertical_slices_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setGenCtf3dFourierReduceByFactor`.
    pub fn set_gen_ctf_3d_fourier_reduce_by_factor(&self, axis_id: AxisID, input: Option<Number>) {
        if axis_id == AxisID::Second {
            self.gen_ctf_3d_fourier_reduce_by_factor_b
                .lock()
                .unwrap()
                .set_number(input);
        } else {
            self.gen_ctf_3d_fourier_reduce_by_factor_a
                .lock()
                .unwrap()
                .set_number(input);
        }
    }

    /// Java `setCtf3dSetupSlabThicknessInNmSet`.
    pub fn set_ctf_3d_setup_slab_thickness_in_nm_set(&self, input: bool) {
        self.ctf_3d_setup_slab_thickness_in_nm_set
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isCtf3dSetupSlabThicknessInNmSet`.
    pub fn is_ctf_3d_setup_slab_thickness_in_nm_set(&self) -> bool {
        self.ctf_3d_setup_slab_thickness_in_nm_set
            .lock()
            .unwrap()
            .is()
    }

    /// Java `setStackMtfFilterLowPassRadiusSigma`.
    pub fn set_stack_mtf_filter_low_pass_radius_sigma(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_low_pass_radius_sigma_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.stack_mtf_filter_low_pass_radius_sigma_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setStackMtfFilterMtfFile`.
    pub fn set_stack_mtf_filter_mtf_file(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_mtf_file_b.lock().unwrap().set(input);
        } else {
            self.stack_mtf_filter_mtf_file_a.lock().unwrap().set(input);
        }
    }

    /// Java `setStackMtfFilterMaximumInverse`.
    pub fn set_stack_mtf_filter_maximum_inverse(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_maximum_inverse_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.stack_mtf_filter_maximum_inverse_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setStackMtfFilterOptimalDoseScaling`.
    pub fn set_stack_mtf_filter_optimal_dose_scaling(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_optimal_dose_scaling_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.stack_mtf_filter_optimal_dose_scaling_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setStackMtfFilterBidirectionalNumViews`.
    pub fn set_stack_mtf_filter_bidirectional_num_views(
        &self,
        axis_id: AxisID,
        input: Option<&str>,
    ) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_bidirectional_num_views_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.stack_mtf_filter_bidirectional_num_views_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setUseStackCtfPhaseFlipXAxisTilt`.
    pub fn set_use_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_stack_ctf_phase_flip_x_axis_tilt_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_stack_ctf_phase_flip_x_axis_tilt_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setStackCtfPhaseFlipXAxisTilt`.
    pub fn set_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_ctf_phase_flip_x_axis_tilt_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.stack_ctf_phase_flip_x_axis_tilt_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setStackCtfPhaseFlipScaleByCtfPower`.
    pub fn set_stack_ctf_phase_flip_scale_by_ctf_power(
        &self,
        axis_id: AxisID,
        input: Option<&str>,
    ) {
        if axis_id == AxisID::Second {
            self.stack_ctf_phase_flip_scale_by_ctf_power_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.stack_ctf_phase_flip_scale_by_ctf_power_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setOrigRawImageStackExtension`, overriding `BaseMetaData`.
    pub fn set_orig_raw_image_stack_extension(&self, input: Option<&Extension>) {
        if let Some(input) = input {
            let locked = self.orig_raw_image_stack_ext_lock.lock().unwrap().is();
            if !locked {
                self.orig_raw_image_stack_ext
                    .lock()
                    .unwrap()
                    .set(Some(&input.to_string()));
            }
        }
    }

    /// Java `setRawImageStackExtension`, overriding `BaseMetaData`.
    pub fn set_raw_image_stack_extension(&self, extension: Option<&Extension>) {
        match extension {
            Some(extension) => self
                .raw_image_stack_ext
                .lock()
                .unwrap()
                .set(Some(&extension.to_string())),
            None => self.raw_image_stack_ext.lock().unwrap().reset(),
        }
    }

    /// Java `getOrigRawImageStackExtension`, overriding `BaseMetaData`.  The source's
    /// `Extension.getInstance` can return null for an unrecognised stored extension.
    pub fn get_orig_raw_image_stack_extension(&self) -> Option<&'static Extension> {
        let value = {
            let field = self.orig_raw_image_stack_ext.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                field.to_string_option()
            }
        };
        if let Some(value) = value {
            return Extension::get_instance(&value);
        }
        Some(self.base.get_orig_raw_image_stack_extension())
    }

    /// Java `getRawImageStackExtension`, overriding `BaseMetaData`.  The source's
    /// `Extension.getInstance` can return null for an unrecognised stored extension.
    pub fn get_raw_image_stack_extension(&self) -> Option<&'static Extension> {
        let value = {
            let field = self.raw_image_stack_ext.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                field.to_string_option()
            }
        };
        if let Some(value) = value {
            return Extension::get_instance(&value);
        }
        Some(self.base.get_raw_image_stack_extension())
    }

    /// Java `setStackMtfFilterInverseRolloffRadiusSigma`.
    pub fn set_stack_mtf_filter_inverse_rolloff_radius_sigma(
        &self,
        axis_id: AxisID,
        input: Option<&str>,
    ) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_inverse_rolloff_radius_sigma_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.stack_mtf_filter_inverse_rolloff_radius_sigma_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setUseStackMtfFilterFixedImageDose`.
    pub fn set_use_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_stack_mtf_filter_fixed_image_dose_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_stack_mtf_filter_fixed_image_dose_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setStackMtfFilterFixedImageDose`.
    pub fn set_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_fixed_image_dose_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.stack_mtf_filter_fixed_image_dose_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setStackMtfFilterDoseWeightingFile`.
    pub fn set_stack_mtf_filter_dose_weighting_file(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_dose_weighting_file_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.stack_mtf_filter_dose_weighting_file_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setStackMtfFilterTypeOfDoseFile`.
    pub fn set_stack_mtf_filter_type_of_dose_file(
        &self,
        axis_id: AxisID,
        enumerated_type: Option<&dyn EnumeratedType>,
    ) {
        let mut value: Option<String> = None;
        if let Some(enumerated_type) = enumerated_type {
            value = Some(enumerated_type.to_string());
        }
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_type_of_dose_file_b
                .lock()
                .unwrap()
                .set_string(value.as_deref());
        } else {
            self.stack_mtf_filter_type_of_dose_file_a
                .lock()
                .unwrap()
                .set_string(value.as_deref());
        }
    }

    /// Java `setStackMtfFilterVoltage200`.
    pub fn set_stack_mtf_filter_voltage_200(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.stack_mtf_filter_voltage_200_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.stack_mtf_filter_voltage_200_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setGenFilterTrialsGaussianCutoffs`.
    pub fn set_gen_filter_trials_gaussian_cutoffs(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_filter_trials_gaussian_cutoffs_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.gen_filter_trials_gaussian_cutoffs_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setGenFilterTrialsGaussianFalloffs`.
    pub fn set_gen_filter_trials_gaussian_falloffs(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_filter_trials_gaussian_falloffs_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.gen_filter_trials_gaussian_falloffs_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `getGenFilterTrialsFakeSIRTiterations`.
    pub fn get_gen_filter_trials_fake_sirt_iterations(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_filter_trials_fake_sirt_iterations_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_filter_trials_fake_sirt_iterations_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getGenFilterTrialsExactObjectSizes`.
    pub fn get_gen_filter_trials_exact_object_sizes(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_filter_trials_exact_object_sizes_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_filter_trials_exact_object_sizes_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getGenFilterTrialsHammingLikeStarts`.
    pub fn get_gen_filter_trials_hamming_like_starts(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_filter_trials_hamming_like_starts_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_filter_trials_hamming_like_starts_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isGenCtf3dOldStyleXtilting`.
    pub fn is_gen_ctf_3d_old_style_xtilting(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_ctf_3d_old_style_xtilting_b.lock().unwrap().is();
        }
        self.gen_ctf_3d_old_style_xtilting_a.lock().unwrap().is()
    }

    /// Java `isGenCtf3dVerticalSlices`.
    pub fn is_gen_ctf_3d_vertical_slices(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_ctf_3d_vertical_slices_b.lock().unwrap().is();
        }
        self.gen_ctf_3d_vertical_slices_a.lock().unwrap().is()
    }

    /// Java `getGenCtf3dFourierReduceByFactor`.
    pub fn get_gen_ctf_3d_fourier_reduce_by_factor(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_ctf_3d_fourier_reduce_by_factor_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_ctf_3d_fourier_reduce_by_factor_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterLowPassRadiusSigma`.
    pub fn get_stack_mtf_filter_low_pass_radius_sigma(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_low_pass_radius_sigma_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_low_pass_radius_sigma_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterMtfFile`.
    pub fn get_stack_mtf_filter_mtf_file(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.stack_mtf_filter_mtf_file_b.lock().unwrap().to_string();
        }
        self.stack_mtf_filter_mtf_file_a.lock().unwrap().to_string()
    }

    /// Java `getStackMtfFilterMaximumInverse`.
    pub fn get_stack_mtf_filter_maximum_inverse(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_maximum_inverse_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_maximum_inverse_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterOptimalDoseScaling`.
    pub fn get_stack_mtf_filter_optimal_dose_scaling(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_optimal_dose_scaling_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_optimal_dose_scaling_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterBidirectionalNumViews`.
    pub fn get_stack_mtf_filter_bidirectional_num_views(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_bidirectional_num_views_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_bidirectional_num_views_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isUseStackCtfPhaseFlipXAxisTilt`.
    pub fn is_use_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .use_stack_ctf_phase_flip_x_axis_tilt_b
                .lock()
                .unwrap()
                .is();
        }
        self.use_stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .is()
    }

    /// Java `getStackCtfPhaseFlipXAxisTilt`.
    pub fn get_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_ctf_phase_flip_x_axis_tilt_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackCtfPhaseFlipScaleByCtfPower`.
    pub fn get_stack_ctf_phase_flip_scale_by_ctf_power(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_ctf_phase_flip_scale_by_ctf_power_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_ctf_phase_flip_scale_by_ctf_power_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterInverseRolloffRadiusSigma`.
    pub fn get_stack_mtf_filter_inverse_rolloff_radius_sigma(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_inverse_rolloff_radius_sigma_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isUseStackMtfFilterFixedImageDose`.
    pub fn is_use_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .use_stack_mtf_filter_fixed_image_dose_b
                .lock()
                .unwrap()
                .is();
        }
        self.use_stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .is()
    }

    /// Java `getStackMtfFilterFixedImageDose`.
    pub fn get_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_fixed_image_dose_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterDoseWeightingFile`.
    pub fn get_stack_mtf_filter_dose_weighting_file(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_dose_weighting_file_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_dose_weighting_file_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackMtfFilterTypeOfDoseFile`.
    pub fn get_stack_mtf_filter_type_of_dose_file(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_mtf_filter_type_of_dose_file_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_mtf_filter_type_of_dose_file_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isStackMtfFilterVoltage200`.
    pub fn is_stack_mtf_filter_voltage_200(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.stack_mtf_filter_voltage_200_b.lock().unwrap().is();
        }
        self.stack_mtf_filter_voltage_200_a.lock().unwrap().is()
    }

    /// Java `getGenFilterTrialsGaussianCutoffs`.
    pub fn get_gen_filter_trials_gaussian_cutoffs(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_filter_trials_gaussian_cutoffs_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_filter_trials_gaussian_cutoffs_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getGenFilterTrialsGaussianFalloffs`.
    pub fn get_gen_filter_trials_gaussian_falloffs(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_filter_trials_gaussian_falloffs_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_filter_trials_gaussian_falloffs_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isGenFilterTrials`.
    pub fn is_gen_filter_trials(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_filter_trials_b.lock().unwrap().is();
        }
        self.gen_filter_trials_a.lock().unwrap().is()
    }

    /// Java `setGenSubarea`.
    pub fn set_gen_subarea(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_subarea_b.lock().unwrap().set_boolean(input);
        } else {
            self.gen_subarea_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setGenSubareaSize`.
    pub fn set_gen_subarea_size(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_subarea_size_b.lock().unwrap().set(input);
        } else {
            self.gen_subarea_size_a.lock().unwrap().set(input);
        }
    }

    /// Java `setGenYOffsetOfSubarea`.
    pub fn set_gen_y_offset_of_subarea(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_y_offset_of_subarea_b.lock().unwrap().set(input);
        } else {
            self.gen_y_offset_of_subarea_a.lock().unwrap().set(input);
        }
    }

    /// Java `setRadialRadius`.
    pub fn set_radial_radius(&self, panel_id: PanelId, axis_id: AxisID, input: Option<&str>) {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                self.gen_radial_radius_b.lock().unwrap().set(input);
            } else {
                self.gen_radial_radius_a.lock().unwrap().set(input);
            }
        } else if panel_id == PanelId::Sirtsetup {
            if axis_id == AxisID::Second {
                self.sirt_radial_radius_b.lock().unwrap().set(input);
            } else {
                self.sirt_radial_radius_a.lock().unwrap().set(input);
            }
        }
    }

    /// Java `isGenSirt`.
    pub fn is_gen_sirt(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_sirt_b.lock().unwrap().is();
        }
        self.gen_sirt_a.lock().unwrap().is()
    }

    /// Java `setRadialSigma`.
    pub fn set_radial_sigma(&self, panel_id: PanelId, axis_id: AxisID, input: Option<&str>) {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                self.gen_radial_sigma_b.lock().unwrap().set(input);
            } else {
                self.gen_radial_sigma_a.lock().unwrap().set(input);
            }
        } else if panel_id == PanelId::Sirtsetup {
            if axis_id == AxisID::Second {
                self.sirt_radial_sigma_b.lock().unwrap().set(input);
            } else {
                self.sirt_radial_sigma_a.lock().unwrap().set(input);
            }
        }
    }

    /// Java `getBatchRunTomoLogReadTimestamp`.
    pub fn get_batch_run_tomo_log_read_timestamp(&self) -> Option<String> {
        self.batch_run_tomo_log_read_timestamp
            .lock()
            .unwrap()
            .to_string_option()
    }

    /// Java `setBatchRunTomoLogReadTimestamp`.
    pub fn set_batch_run_tomo_log_read_timestamp(&self, input: Option<&str>) {
        self.batch_run_tomo_log_read_timestamp
            .lock()
            .unwrap()
            .set(input);
    }

    /// Java `isBatchRunTomoLogReadFinished`.
    pub fn is_batch_run_tomo_log_read_finished(&self) -> bool {
        self.batch_run_tomo_log_read_finished.lock().unwrap().is()
    }

    /// Java `getBatchRunTomoLogReadAxisID`.
    pub fn get_batch_run_tomo_log_read_axis_id(&self) -> Option<AxisID> {
        *self.batch_run_tomo_log_read_axis_id.lock().unwrap()
    }

    /// Java `setBatchRunTomoLogReadAxisID`.
    pub fn set_batch_run_tomo_log_read_axis_id(&self, input: Option<AxisID>) {
        *self.batch_run_tomo_log_read_axis_id.lock().unwrap() = input;
    }

    /// Java `setBatchRunTomoLogReadFinished`.
    pub fn set_batch_run_tomo_log_read_finished(&self, input: bool) {
        self.batch_run_tomo_log_read_finished
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setHammingLikeFilter`.
    pub fn set_hamming_like_filter(&self, panel_id: PanelId, axis_id: AxisID, input: Option<&str>) {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                self.gen_hamming_like_filter_b.lock().unwrap().set(input);
            } else {
                self.gen_hamming_like_filter_a.lock().unwrap().set(input);
            }
        }
    }

    /// Java `getHammingLikeFilter`.
    pub fn get_hamming_like_filter(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                return Some(self.gen_hamming_like_filter_b.lock().unwrap().to_string());
            }
            return Some(self.gen_hamming_like_filter_a.lock().unwrap().to_string());
        }
        None
    }

    /// Java `setFakeSIRTiterations`.
    pub fn set_fake_sirt_iterations(
        &self,
        panel_id: PanelId,
        axis_id: AxisID,
        input: Option<&str>,
    ) {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                self.gen_fake_sirt_iterations_b.lock().unwrap().set(input);
            } else {
                self.gen_fake_sirt_iterations_a.lock().unwrap().set(input);
            }
        }
    }

    /// Java `getFakeSIRTiterations`.
    pub fn get_fake_sirt_iterations(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                return Some(self.gen_fake_sirt_iterations_b.lock().unwrap().to_string());
            }
            return Some(self.gen_fake_sirt_iterations_a.lock().unwrap().to_string());
        }
        None
    }

    /// Java `setExactFilterSize`.
    pub fn set_exact_filter_size(&self, panel_id: PanelId, axis_id: AxisID, input: Option<&str>) {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                self.gen_exact_filter_size_b.lock().unwrap().set(input);
            } else {
                self.gen_exact_filter_size_a.lock().unwrap().set(input);
            }
        }
    }

    /// Java `getExactFilterSize`.
    pub fn get_exact_filter_size(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                return Some(self.gen_exact_filter_size_b.lock().unwrap().to_string());
            }
            return Some(self.gen_exact_filter_size_a.lock().unwrap().to_string());
        }
        None
    }

    /// Java `setTiltAngleSpecB`.
    pub fn set_tilt_angle_spec_b(&self, tilt_angle_spec: TiltAngleSpec) {
        *self.tilt_angle_spec_b.lock().unwrap() = tilt_angle_spec;
    }

    /// Java `setComScriptCreated`.
    pub fn set_com_script_created(&self, state: bool) {
        *self.com_scripts_created.lock().unwrap() = state;
    }

    /// Java `setFiducialessAlignment`.
    pub fn set_fiducialess_alignment(&self, axis_id: AxisID, state: bool) {
        if axis_id == AxisID::Second {
            *self.fiducialess_alignment_b.lock().unwrap() = state;
        } else {
            *self.fiducialess_alignment_a.lock().unwrap() = state;
        }
    }

    /// Java `setFineExists`.
    pub fn set_fine_exists(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.fine_exists_b.lock().unwrap().set_boolean(input);
        } else {
            self.fine_exists_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setWholeTomogramSample`.
    pub fn set_whole_tomogram_sample(&self, axis_id: AxisID, state: bool) {
        if axis_id == AxisID::Second {
            *self.whole_tomogram_sample_b.lock().unwrap() = state;
        } else {
            *self.whole_tomogram_sample_a.lock().unwrap() = state;
        }
    }

    /// Java `load(Properties)`.  Get the objects attributes from the properties object.
    pub fn load(&self, props: &mut BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java private `setAxisPrepends`.  Set up axis prepends.  For dual axis, axis a
    /// prepend is "A" and axis b prepend is "B".  For single axis, axis a prepend is ""
    /// and axis b prepend doesn't exist.
    fn set_axis_prepends(&self) {
        // set firstAxis and secondAxis strings (based on AxisType)
        if *self.base.axis_type.lock().unwrap() == AxisType::DualAxis {
            *self.first_axis_prepend.lock().unwrap() =
                Some(AxisID::First.get_extension().to_uppercase());
            *self.second_axis_prepend.lock().unwrap() =
                Some(AxisID::Second.get_extension().to_uppercase());
        } else {
            *self.first_axis_prepend.lock().unwrap() = Some(String::new());
        }
    }

    /// Java `load(Properties, String)`.  Bug# 2403.
    pub fn load_with_prepend(&self, props: &mut BTreeMap<String, String>, parent_prepend: &str) {
        let base_prepend = self.create_prepend(parent_prepend);
        if self
            .base
            .load_with_created_prepend(props, base_prepend.as_deref())
        {
            self.check_image_filename_style_loaded(parent_prepend);
        }
        // `StringProperty.load` may remove a backward-compatible key from the Java
        // `Properties`; this translation's `props` is read-only, so the string properties load
        // from a copy.  None of MetaData's string properties declares such a key.
        let mut props_copy = props.clone();
        self.base.revision_number.lock().unwrap().reset();
        *self.distortion_file.lock().unwrap() = None;
        *self.mag_gradient_file.lock().unwrap() = None;
        self.binning.lock().unwrap().reset();
        *self.use_local_alignments_a.lock().unwrap() = true;
        *self.use_local_alignments_b.lock().unwrap() = true;
        self.use_z_factors_a.lock().unwrap().reset();
        self.use_z_factors_b.lock().unwrap().reset();
        if let Some(field) = self.tomo_gen_tilt_parallel_a.lock().unwrap().as_mut() {
            field.reset();
        }
        if let Some(field) = self.tomo_gen_tilt_parallel_b.lock().unwrap().as_mut() {
            field.reset();
        }
        if let Some(field) = self.tilt_3d_find_tilt_parallel_a.lock().unwrap().as_mut() {
            field.reset();
        }
        if let Some(field) = self.tilt_3d_find_tilt_parallel_b.lock().unwrap().as_mut() {
            field.reset();
        }
        if let Some(field) = self
            .final_stack_ctf_correction_parallel_a
            .lock()
            .unwrap()
            .as_mut()
        {
            field.reset();
        }
        if let Some(field) = self
            .final_stack_ctf_correction_parallel_b
            .lock()
            .unwrap()
            .as_mut()
        {
            field.reset();
        }
        if let Some(field) = self.combine_volcombine_parallel.lock().unwrap().as_mut() {
            field.reset();
        }
        self.sample_thickness_a.lock().unwrap().reset();
        self.sample_thickness_b.lock().unwrap().reset();
        *self.target_patch_size_x_and_y.lock().unwrap() =
            TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_DEFAULT.to_string(); // backwards compatibility
        *self.number_of_local_patches_x_and_y.lock().unwrap() =
            TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT.to_string();
        self.no_beam_tilt_selected_a.lock().unwrap().reset();
        self.fixed_beam_tilt_selected_a.lock().unwrap().reset();
        self.fixed_beam_tilt_a.lock().unwrap().reset();
        self.no_beam_tilt_selected_b.lock().unwrap().reset();
        self.fixed_beam_tilt_selected_b.lock().unwrap().reset();
        self.fixed_beam_tilt_b.lock().unwrap().reset();
        self.final_stack_better_radius_a.lock().unwrap().reset();
        self.final_stack_better_radius_b.lock().unwrap().reset();
        self.final_stack_fiducial_diameter_a.lock().unwrap().reset();
        self.final_stack_fiducial_diameter_b.lock().unwrap().reset();
        self.final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .reset();
        self.final_stack_expand_circle_iterations_b
            .lock()
            .unwrap()
            .reset();
        self.use_final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .reset();
        self.use_final_stack_expand_circle_iterations_b
            .lock()
            .unwrap()
            .reset();
        self.final_stack_polynomial_order_a.lock().unwrap().reset();
        self.final_stack_polynomial_order_b.lock().unwrap().reset();
        self.final_aligned_stack_dialog_saved_a
            .lock()
            .unwrap()
            .reset();
        self.final_aligned_stack_dialog_saved_b
            .lock()
            .unwrap()
            .reset();
        self.tomo_gen_trial_tomogram_name_list_a
            .lock()
            .unwrap()
            .lock()
            .unwrap()
            .reset();
        self.tomo_gen_trial_tomogram_name_list_b
            .lock()
            .unwrap()
            .lock()
            .unwrap()
            .reset();
        self.track_use_raptor_a.lock().unwrap().reset();
        self.track_raptor_use_raw_stack_a.lock().unwrap().reset();
        self.track_raptor_mark_a.lock().unwrap().reset();
        self.track_raptor_diam_a.lock().unwrap().reset();
        self.stack_erase_gold_model_use_fid_a
            .lock()
            .unwrap()
            .reset();
        self.stack_erase_gold_model_use_fid_b
            .lock()
            .unwrap()
            .reset();
        self.post_flatten_input_trim_vol.lock().unwrap().reset();
        self.post_flatten_warp_contours_on_one_surface
            .lock()
            .unwrap()
            .reset();
        self.post_flatten_warp_spacing_in_x.lock().unwrap().reset();
        self.post_flatten_warp_spacing_in_y.lock().unwrap().reset();
        self.post_squeeze_vol_input_trim_vol.lock().unwrap().reset();
        self.reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .reset();
        self.reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .reset();
        self.reduce_filt_vol_low_pass_radius_sigma
            .lock()
            .unwrap()
            .reset();
        self.reduce_filt_vol_deconvolution_strength
            .lock()
            .unwrap()
            .reset();
        self.reduce_filt_vol_snr_falloff.lock().unwrap().reset();
        self.reduce_filt_vol_high_pass_nyquist
            .lock()
            .unwrap()
            .reset();
        self.reduce_filt_vol_defocus_in_microns
            .lock()
            .unwrap()
            .reset();
        self.reduce_filt_vol_phase_shift.lock().unwrap().reset();
        self.pos_binning_a.lock().unwrap().reset();
        self.pos_binning_b.lock().unwrap().reset();
        self.stack_binning_a.lock().unwrap().reset();
        self.stack_binning_b.lock().unwrap().reset();
        self.stack_3d_find_binning_a.lock().unwrap().reset();
        self.stack_3d_find_binning_b.lock().unwrap().reset();
        self.post_cur_tab.lock().unwrap().reset();
        self.gen_cur_tab.lock().unwrap().reset();
        self.post_exists.lock().unwrap().reset();
        self.lambda_for_smoothing.lock().unwrap().reset();
        self.lambda_for_smoothing_list.lock().unwrap().reset();
        self.track_overlap_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .reset();
        self.track_overlap_of_patches_x_and_y_b
            .lock()
            .unwrap()
            .reset();
        self.track_number_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .reset();
        self.track_number_of_patches_x_and_y_b
            .lock()
            .unwrap()
            .reset();
        self.track_length_and_overlap_a.lock().unwrap().reset();
        self.track_length_and_overlap_b.lock().unwrap().reset();
        self.track_method_a
            .lock()
            .unwrap()
            .set(Some(&tracking_method::SEED.to_string()));
        self.track_method_b
            .lock()
            .unwrap()
            .set(Some(&tracking_method::SEED.to_string()));
        self.fine_exists_a.lock().unwrap().reset();
        self.fine_exists_b.lock().unwrap().reset();
        self.gen_log_a.lock().unwrap().reset();
        self.gen_log_b.lock().unwrap().reset();
        self.gen_scale_factor_log_a.lock().unwrap().reset();
        self.gen_scale_factor_log_b.lock().unwrap().reset();
        self.gen_scale_offset_log_a.lock().unwrap().reset();
        self.gen_scale_offset_log_b.lock().unwrap().reset();
        self.gen_scale_factor_linear_a.lock().unwrap().reset();
        self.gen_scale_factor_linear_b.lock().unwrap().reset();
        self.gen_scale_offset_linear_a.lock().unwrap().reset();
        self.gen_scale_offset_linear_b.lock().unwrap().reset();
        self.gen_super_sample_factor_a.lock().unwrap().reset();
        self.gen_super_sample_factor_b.lock().unwrap().reset();
        self.gen_expand_input_lines_a.lock().unwrap().reset();
        self.gen_expand_input_lines_b.lock().unwrap().reset();
        self.gen_exists_a.lock().unwrap().reset();
        self.gen_exists_b.lock().unwrap().reset();
        self.pos_exists_a.lock().unwrap().reset();
        self.pos_exists_b.lock().unwrap().reset();
        self.gen_back_projection_a.lock().unwrap().reset();
        self.gen_back_projection_b.lock().unwrap().reset();
        self.gen_subarea_a.lock().unwrap().reset();
        self.gen_subarea_b.lock().unwrap().reset();
        self.gen_subarea_size_a.lock().unwrap().reset();
        self.gen_subarea_size_b.lock().unwrap().reset();
        self.gen_y_offset_of_subarea_a.lock().unwrap().reset();
        self.gen_y_offset_of_subarea_b.lock().unwrap().reset();
        self.gen_radial_radius_a.lock().unwrap().reset();
        self.gen_radial_radius_b.lock().unwrap().reset();
        self.gen_radial_sigma_a.lock().unwrap().reset();
        self.gen_radial_sigma_b.lock().unwrap().reset();
        self.post_trimvol_x_min.lock().unwrap().reset();
        self.post_trimvol_x_max.lock().unwrap().reset();
        self.post_trimvol_y_min.lock().unwrap().reset();
        self.post_trimvol_y_max.lock().unwrap().reset();
        self.post_trimvol_z_min.lock().unwrap().reset();
        self.post_trimvol_z_max.lock().unwrap().reset();
        self.post_trimvol_convert_to_bytes.lock().unwrap().reset();
        self.post_trimvol_fixed_scaling.lock().unwrap().reset();
        self.post_trimvol_flipped_volume.lock().unwrap().reset();
        self.post_trimvol_section_scale_min.lock().unwrap().reset();
        self.post_trimvol_section_scale_max.lock().unwrap().reset();
        self.post_trimvol_fixed_scale_min.lock().unwrap().reset();
        self.post_trimvol_fixed_scale_max.lock().unwrap().reset();
        self.post_trimvol_swap_yz.lock().unwrap().reset();
        self.post_trimvol_rotate_x.lock().unwrap().reset();
        self.post_trimvol_scale_x_min.lock().unwrap().reset();
        self.post_trimvol_scale_x_max.lock().unwrap().reset();
        self.post_trimvol_scale_y_min.lock().unwrap().reset();
        self.post_trimvol_scale_y_max.lock().unwrap().reset();
        self.erase_beads_initialized.lock().unwrap().reset();
        self.track_seed_model_manual_a
            .lock()
            .unwrap()
            .set_boolean(true);
        self.track_seed_model_manual_b
            .lock()
            .unwrap()
            .set_boolean(true);
        self.track_seed_model_auto_a.lock().unwrap().reset();
        self.track_seed_model_auto_b.lock().unwrap().reset();
        self.track_seed_model_transfer_a.lock().unwrap().reset();
        self.track_seed_model_transfer_b.lock().unwrap().reset();
        self.track_exclude_inside_areas_a.lock().unwrap().reset();
        self.track_exclude_inside_areas_b.lock().unwrap().reset();
        self.track_just_find_shifts_near_zero_a
            .lock()
            .unwrap()
            .reset();
        self.track_just_find_shifts_near_zero_b
            .lock()
            .unwrap()
            .reset();
        self.track_target_number_of_beads_a.lock().unwrap().reset();
        self.track_target_number_of_beads_b.lock().unwrap().reset();
        self.track_target_density_of_beads_a.lock().unwrap().reset();
        self.track_target_density_of_beads_b.lock().unwrap().reset();
        self.track_clustered_points_allowed_elongated_a
            .lock()
            .unwrap()
            .reset();
        self.track_clustered_points_allowed_elongated_b
            .lock()
            .unwrap()
            .reset();
        self.track_clustered_points_allowed_elongated_value_a
            .lock()
            .unwrap()
            .reset();
        self.track_clustered_points_allowed_elongated_value_b
            .lock()
            .unwrap()
            .reset();
        self.track_advanced_a.lock().unwrap().reset();
        self.track_advanced_b.lock().unwrap().reset();
        self.stack_3d_find_thickness_a.lock().unwrap().reset();
        self.stack_3d_find_thickness_b.lock().unwrap().reset();
        self.set_fei_pixel_size.lock().unwrap().reset();
        self.post_trimvol_new_style_z.lock().unwrap().reset();
        self.post_trimvol_scaling_new_style_z
            .lock()
            .unwrap()
            .reset();
        self.is_twodir_a.lock().unwrap().set_double(TWO_DIR_DEFAULT);
        self.is_twodir_b.lock().unwrap().set_double(TWO_DIR_DEFAULT);
        self.is_dose_sym_a
            .lock()
            .unwrap()
            .set_double(DOSE_SYM_DEFAULT);
        self.is_dose_sym_b
            .lock()
            .unwrap()
            .set_double(DOSE_SYM_DEFAULT);
        self.twodir_a.lock().unwrap().reset();
        self.twodir_b.lock().unwrap().reset();
        self.dose_sym_a.lock().unwrap().reset();
        self.dose_sym_b.lock().unwrap().reset();
        self.seed_and_track_tab_a.lock().unwrap().reset();
        self.seed_and_track_tab_b.lock().unwrap().reset();
        self.raptor_tab_a.lock().unwrap().reset();
        self.raptor_tab_b.lock().unwrap().reset();
        self.coarse_antialias_filter_a.lock().unwrap().reset();
        self.coarse_antialias_filter_b.lock().unwrap().reset();
        self.stack_antialias_filter_a.lock().unwrap().reset();
        self.stack_antialias_filter_b.lock().unwrap().reset();
        self.track_elongated_points_allowed_a
            .lock()
            .unwrap()
            .reset();
        self.track_elongated_points_allowed_b
            .lock()
            .unwrap()
            .reset();
        self.track_lower_target_for_clustered_a
            .lock()
            .unwrap()
            .reset();
        self.track_lower_target_for_clustered_b
            .lock()
            .unwrap()
            .reset();
        self.orig_views_with_mag_changes_a.lock().unwrap().reset();
        self.orig_views_with_mag_changes_b.lock().unwrap().reset();
        self.orig_views_with_mag_changes_set_a
            .lock()
            .unwrap()
            .reset();
        self.orig_views_with_mag_changes_set_b
            .lock()
            .unwrap()
            .reset();
        self.weight_whole_tracks_a.lock().unwrap().reset();
        self.weight_whole_tracks_b.lock().unwrap().reset();
        self.length_of_pieces_a.lock().unwrap().reset();
        self.length_of_pieces_b.lock().unwrap().reset();
        self.minimum_overlap_a.lock().unwrap().reset();
        self.minimum_overlap_b.lock().unwrap().reset();
        self.target_measurement_ratio_a.lock().unwrap().reset();
        self.target_measurement_ratio_b.lock().unwrap().reset();
        self.min_measurement_ratio_a.lock().unwrap().reset();
        self.min_measurement_ratio_b.lock().unwrap().reset();
        self.order_of_restrictions_a.lock().unwrap().reset();
        self.order_of_restrictions_b.lock().unwrap().reset();
        self.skip_beam_tilt_with_one_rot_a.lock().unwrap().reset();
        self.skip_beam_tilt_with_one_rot_b.lock().unwrap().reset();
        self.fine_local_align_validation_a.lock().unwrap().reset();
        self.fine_local_align_validation_b.lock().unwrap().reset();
        self.sample_type_a.lock().unwrap().reset();
        self.sample_type_b.lock().unwrap().reset();
        self.has_gold_beads_a.lock().unwrap().reset();
        self.has_gold_beads_b.lock().unwrap().reset();
        self.positioning_bead_size_a.lock().unwrap().reset();
        self.positioning_bead_size_b.lock().unwrap().reset();
        self.extra_thickness_a.lock().unwrap().reset();
        self.extra_thickness_b.lock().unwrap().reset();
        self.extra_thickness_cryo_a.lock().unwrap().reset();
        self.extra_thickness_cryo_b.lock().unwrap().reset();
        self.gen_hamming_like_filter_a.lock().unwrap().reset();
        self.gen_hamming_like_filter_b.lock().unwrap().reset();
        self.gen_fake_sirt_iterations_a.lock().unwrap().reset();
        self.gen_fake_sirt_iterations_b.lock().unwrap().reset();
        self.gen_exact_filter_size_a.lock().unwrap().reset();
        self.gen_exact_filter_size_b.lock().unwrap().reset();
        self.sirt_radial_radius_a.lock().unwrap().reset();
        self.sirt_radial_radius_b.lock().unwrap().reset();
        self.sirt_radial_sigma_a.lock().unwrap().reset();
        self.sirt_radial_sigma_b.lock().unwrap().reset();
        self.batch_run_tomo_log_read_timestamp
            .lock()
            .unwrap()
            .reset();
        self.batch_run_tomo_log_read_finished
            .lock()
            .unwrap()
            .reset();
        self.positioning_new_dialog_a
            .lock()
            .unwrap()
            .set_boolean(true);
        self.positioning_new_dialog_b
            .lock()
            .unwrap()
            .set_boolean(true);
        *self.batch_run_tomo_log_read_axis_id.lock().unwrap() = None;
        self.gen_filter_trials_a.lock().unwrap().reset();
        self.gen_filter_trials_b.lock().unwrap().reset();
        self.gen_filter_trials_fake_sirt_iterations_a
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_fake_sirt_iterations_b
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_exact_object_sizes_a
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_exact_object_sizes_b
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_gaussian_cutoffs_a
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_gaussian_cutoffs_b
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_gaussian_falloffs_a
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_gaussian_falloffs_b
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_hamming_like_starts_a
            .lock()
            .unwrap()
            .reset();
        self.gen_filter_trials_hamming_like_starts_b
            .lock()
            .unwrap()
            .reset();
        self.gen_ctf_3d_old_style_xtilting_a.lock().unwrap().reset();
        self.gen_ctf_3d_old_style_xtilting_b.lock().unwrap().reset();
        self.gen_ctf_3d_vertical_slices_a.lock().unwrap().reset();
        self.gen_ctf_3d_vertical_slices_b.lock().unwrap().reset();
        self.gen_ctf_3d_fourier_reduce_by_factor_a
            .lock()
            .unwrap()
            .reset();
        self.gen_ctf_3d_fourier_reduce_by_factor_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_low_pass_radius_sigma_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_low_pass_radius_sigma_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_mtf_file_a.lock().unwrap().reset();
        self.stack_mtf_filter_mtf_file_b.lock().unwrap().reset();
        self.stack_mtf_filter_maximum_inverse_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_maximum_inverse_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_optimal_dose_scaling_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_optimal_dose_scaling_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_bidirectional_num_views_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_bidirectional_num_views_b
            .lock()
            .unwrap()
            .reset();
        self.use_stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .reset();
        self.use_stack_ctf_phase_flip_x_axis_tilt_b
            .lock()
            .unwrap()
            .reset();
        self.stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .reset();
        self.stack_ctf_phase_flip_x_axis_tilt_b
            .lock()
            .unwrap()
            .reset();
        self.stack_ctf_phase_flip_scale_by_ctf_power_a
            .lock()
            .unwrap()
            .reset();
        self.stack_ctf_phase_flip_scale_by_ctf_power_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_b
            .lock()
            .unwrap()
            .reset();
        self.use_stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .reset();
        self.use_stack_mtf_filter_fixed_image_dose_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_fixed_image_dose_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_dose_weighting_file_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_dose_weighting_file_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_type_of_dose_file_a
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_type_of_dose_file_b
            .lock()
            .unwrap()
            .reset();
        self.stack_mtf_filter_voltage_200_a.lock().unwrap().reset();
        self.stack_mtf_filter_voltage_200_b.lock().unwrap().reset();
        self.raw_image_stack_ext.lock().unwrap().reset();
        self.orig_raw_image_stack_ext.lock().unwrap().reset();
        self.orig_raw_image_stack_ext_lock.lock().unwrap().reset();
        self.post_trimvol_swap_yz_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_rotate_x_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_convert_to_bytes_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_section_scale_min_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_section_scale_max_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_z_min_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_z_max_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_scale_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_scale_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_scale_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_trimvol_scale_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.stack_aligned_stack_erase_gold_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.orig_raw_image_stack_ext_from_batch_run_tomo
            .lock()
            .unwrap()
            .reset();
        self.gen_sirt_a.lock().unwrap().reset();
        self.gen_sirt_b.lock().unwrap().reset();
        self.subtomo_reorientation_type_none.lock().unwrap().reset();
        self.subtomo_reorientation_type_flipped
            .lock()
            .unwrap()
            .reset();
        self.subtomo_reorientation_type_rotated
            .lock()
            .unwrap()
            .reset();
        self.subtomo_make_volume_stacks.lock().unwrap().reset();
        self.subtomo_extent_of_z_levels_in_nm
            .lock()
            .unwrap()
            .reset();
        self.subtomo_new_aligned_binning.lock().unwrap().reset();
        self.subtomo_fourier_reduce_by_factor
            .lock()
            .unwrap()
            .reset();
        self.alt_tomo_rootname_to_process.lock().unwrap().reset();
        self.alt_tomo_trim_volume.lock().unwrap().reset();
        self.alt_tomo_archive_orig_stack.lock().unwrap().reset();
        self.ctf_3d_setup_slab_thickness_in_nm_set
            .lock()
            .unwrap()
            .reset();
        // Bug# 2403
        let prepend = self
            .create_prepend(parent_prepend)
            .unwrap_or("null".to_string());
        let group = format!("{}.", prepend);
        // Upstream bug fixed in translation (MetaData.java:3017): `AxisType.fromString`
        // returns null for an unrecognised value, and the next `store` then throws
        // `NullPointerException` on `axisType.toString()`.  An unrecognised value now loads as
        // NOT_SET, which `isValid` reports exactly as it reported the null.
        *self.base.axis_type.lock().unwrap() = AxisType::from_string(
            props
                .get(&format!("{}AxisType", group))
                .map(|s| s.as_str())
                .unwrap_or("Not Set"),
        )
        .unwrap_or(AxisType::NotSet);
        self.set_axis_prepends();
        // backwards compatibility
        storable::StorableValue::load_with_prepend(
            &mut *self.base.revision_number.lock().unwrap(),
            props,
            &prepend,
        );
        let revision_le = |version: &str| {
            self.base.revision_number.lock().unwrap().le(Some(
                &EtomoVersion::get_default_instance_with_version(Some(version)),
            ))
        };
        if revision_le("1.7") {
            self.fiducialess_a.lock().unwrap().load_with_alternate_key(
                Some(props),
                Some(&prepend),
                Some(".A.Param.tilt.Fiducialess"),
            );
            self.fiducialess_b.lock().unwrap().load_with_alternate_key(
                Some(props),
                Some(&prepend),
                Some(".B.Param.tilt.Fiducialess"),
            );
        } else {
            self.fiducialess_a
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.fiducialess_b
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
        }
        if revision_le("1.8") {
            self.stack_binning_a
                .lock()
                .unwrap()
                .load_with_alternate_key(
                    Some(props),
                    Some(&prepend),
                    Some(FINAL_STACK_BINNING_A_BACKWARD_COMPATABILITY_1_8),
                );
            self.stack_binning_b
                .lock()
                .unwrap()
                .load_with_alternate_key(
                    Some(props),
                    Some(&prepend),
                    Some(FINAL_STACK_BINNING_B_BACKWARD_COMPATABILITY_1_8),
                );
        } else {
            self.stack_binning_a
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.stack_binning_b
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
        }
        if revision_le("1.9") {
            // better radius needs to be converted to final stack fiducial diameter.
            self.final_stack_better_radius_a
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.final_stack_better_radius_b
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        }
        if revision_le("1.10") {
            self.track_use_raptor_a
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            if self.track_use_raptor_a.lock().unwrap().is() {
                self.track_method_a
                    .lock()
                    .unwrap()
                    .set(Some(&tracking_method::RAPTOR.to_string()));
            }
        } else {
            self.track_method_a
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        }
        if revision_le("1.11") {
            self.post_trimvol_x_min.lock().unwrap().load_from_other_key(
                Some(&mut props_copy),
                Some(&prepend),
                Some("Trimvol.XMin"),
            );
            self.post_trimvol_x_max.lock().unwrap().load_from_other_key(
                Some(&mut props_copy),
                Some(&prepend),
                Some("Trimvol.XMax"),
            );
            // Don't use flipped data; this meta data should be the same as the screen, not
            // match
            // the param or the image.
            self.post_trimvol_y_min.lock().unwrap().load_from_other_key(
                Some(&mut props_copy),
                Some(&prepend),
                Some("Trimvol.ZMin"),
            );
            self.post_trimvol_y_max.lock().unwrap().load_from_other_key(
                Some(&mut props_copy),
                Some(&prepend),
                Some("Trimvol.ZMax"),
            );
            self.post_trimvol_z_min.lock().unwrap().load_from_other_key(
                Some(&mut props_copy),
                Some(&prepend),
                Some("Trimvol.YMin"),
            );
            self.post_trimvol_z_max.lock().unwrap().load_from_other_key(
                Some(&mut props_copy),
                Some(&prepend),
                Some("Trimvol.YMax"),
            );
            self.post_trimvol_convert_to_bytes
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.ConvertToBytes");
            self.post_trimvol_fixed_scaling
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.FixedScaling");
            self.post_trimvol_flipped_volume
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.FlippedVolume");
            self.post_trimvol_section_scale_min
                .lock()
                .unwrap()
                .load_from_other_key(
                    Some(&mut props_copy),
                    Some(&prepend),
                    Some("Trimvol.SectionScaleMin"),
                );
            self.post_trimvol_section_scale_max
                .lock()
                .unwrap()
                .load_from_other_key(
                    Some(&mut props_copy),
                    Some(&prepend),
                    Some("Trimvol.SectionScaleMax"),
                );
            self.post_trimvol_fixed_scale_min
                .lock()
                .unwrap()
                .load_from_other_key(
                    Some(&mut props_copy),
                    Some(&prepend),
                    Some("Trimvol.FixedScaleMin"),
                );
            self.post_trimvol_fixed_scale_max
                .lock()
                .unwrap()
                .load_from_other_key(
                    Some(&mut props_copy),
                    Some(&prepend),
                    Some("Trimvol.FixedScaleMax"),
                );
            self.post_trimvol_swap_yz
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.SwapYZ");
            self.post_trimvol_rotate_x
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.RotateX");
            self.post_trimvol_scale_x_min
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.ScaleXMin");
            self.post_trimvol_scale_x_max
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.ScaleXMax");
            self.post_trimvol_scale_y_min
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.ScaleYMin");
            self.post_trimvol_scale_y_max
                .lock()
                .unwrap()
                .load_from_other_key(Some(props), Some(&prepend), "Trimvol.ScaleYMax");
            if props.get(&format!("{} Trimvol.Version", group)).is_none() {
                // Handle backwards compatibility from TrimvolParam version 1.0 - the 1.0 version
                // wasn't saved.
                crate::imod::etomo::comscript::trimvol_param::TrimvolParam::convert_index_coords_to_imod_coords(
                    &mut self.post_trimvol_scale_x_min.lock().unwrap(),
                    &mut self.post_trimvol_scale_x_max.lock().unwrap(),
                    &mut self.post_trimvol_scale_y_min.lock().unwrap(),
                    &mut self.post_trimvol_scale_y_max.lock().unwrap(),
                );
            }
        } else {
            self.post_trimvol_x_min
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_x_max
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_y_min
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_y_max
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_z_min
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_z_max
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_convert_to_bytes
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_fixed_scaling
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_flipped_volume
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_section_scale_min
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_section_scale_max
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_fixed_scale_min
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_fixed_scale_max
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            self.post_trimvol_swap_yz
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_rotate_x
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_scale_x_min
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_scale_x_max
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_scale_y_min
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
            self.post_trimvol_scale_y_max
                .lock()
                .unwrap()
                .load_with_prepend(props, Some(&prepend));
        }
        self.final_stack_fiducial_diameter_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_stack_fiducial_diameter_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_stack_expand_circle_iterations_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_final_stack_expand_circle_iterations_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        // Make this true for now until the variable is present in all of the
        // data files so as to not break existing files
        // May-03-2002
        *self.com_scripts_created.lock().unwrap() = props
            .get(&format!("{}ComScriptsCreated", group))
            .map(|s| s.as_str())
            .unwrap_or("true")
            .eq_ignore_ascii_case("true");
        // Backwards compatibility with FilesetName string
        *self.dataset_name.lock().unwrap() = props
            .get(&format!("{}FilesetName", group))
            .cloned()
            .unwrap_or_default();
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        *self.dataset_name.lock().unwrap() = props
            .get(&format!("{}DatasetName", group))
            .cloned()
            .unwrap_or(dataset_name);
        *self.backup_directory.lock().unwrap() = props
            .get(&format!("{}BackupDirectory", group))
            .cloned()
            .unwrap_or_default();
        // Upstream bug fixed in translation (MetaData.java:3124): an unrecognised value makes
        // `DataSource.fromString` return null and the next `store` throw
        // `NullPointerException`.  It now loads as the property's default, CCD.
        *self.data_source.lock().unwrap() = DataSource::from_string(
            props
                .get(&format!("{}DataSource", group))
                .map(|s| s.as_str())
                .unwrap_or("CCD"),
        )
        .unwrap_or(DataSource::Ccd);
        // Upstream bug fixed in translation (MetaData.java:3125): as for DataSource, an
        // unrecognised value now loads as the property's default, Single View, instead of a
        // null that `store` dereferences.
        *self.view_type.lock().unwrap() = ViewType::from_string(
            props
                .get(&format!("{}ViewType", group))
                .map(|s| s.as_str())
                .unwrap_or("Single View"),
        )
        .unwrap_or(ViewType::SingleView);
        let mut property = props.get(&format!("{}PixelSize", group)).cloned();
        // Upstream bug fixed in translation (MetaData.java:3131): `Double.parseDouble` on a
        // malformed value throws `NumberFormatException` out of the whole load.  A malformed
        // value now loads as NaN, the value of a blank one.
        match &property {
            Some(value) if !java_lang_string_matches_whitespace(value) => {
                *self.pixel_size.lock().unwrap() =
                    java_lang_double_value_of(value).unwrap_or(f64::NAN);
            }
            _ => {
                *self.pixel_size.lock().unwrap() = f64::NAN;
            }
        }
        self.half_float_mode_output
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        *self.use_local_alignments_a.lock().unwrap() = props
            .get(&format!("{}UseLocalAlignmentsA", group))
            .map(|s| s.as_str())
            .unwrap_or("true")
            .eq_ignore_ascii_case("true");
        *self.use_local_alignments_b.lock().unwrap() = props
            .get(&format!("{}UseLocalAlignmentsB", group))
            .map(|s| s.as_str())
            .unwrap_or("true")
            .eq_ignore_ascii_case("true");
        property = props.get(&format!("{}FiducialDiameter", group)).cloned();
        // Upstream bug fixed in translation (MetaData.java:3143): as for PixelSize, a
        // malformed value now loads as NaN instead of throwing.
        match &property {
            Some(value) if !java_lang_string_matches_whitespace(value) => {
                *self.fiducial_diameter.lock().unwrap() =
                    java_lang_double_value_of(value).unwrap_or(f64::NAN);
            }
            _ => {
                *self.fiducial_diameter.lock().unwrap() = f64::NAN;
            }
        }
        // Read in the old single image rotation or the newer separate image
        // rotation for each axis
        let str_old_rotation = props
            .get(&format!("{}ImageRotation", group))
            .cloned()
            .unwrap_or("0.0".to_string());
        self.image_rotation_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        if self.image_rotation_a.lock().unwrap().is_null() {
            self.image_rotation_a
                .lock()
                .unwrap()
                .set_string(Some(&str_old_rotation));
        }
        self.image_rotation_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        if self.image_rotation_b.lock().unwrap().is_null() {
            self.image_rotation_b
                .lock()
                .unwrap()
                .set_string(Some(&str_old_rotation));
        }
        *self.exclude_projections_a.lock().unwrap() = props
            .get(&format!("{}AxisA.ExcludeProjections", group))
            .cloned();
        self.tilt_angle_spec_a
            .lock()
            .unwrap()
            .load_with_prepend(props, &format!("{}AxisA", group));
        *self.exclude_projections_b.lock().unwrap() = props
            .get(&format!("{}AxisB.ExcludeProjections", group))
            .cloned();
        self.tilt_angle_spec_b
            .lock()
            .unwrap()
            .load_with_prepend(props, &format!("{}AxisB", group));
        storable::StorableValue::load_with_prepend(
            &mut *self.combine_params.lock().unwrap(),
            props,
            &group,
        );
        *self.distortion_file.lock().unwrap() =
            props.get(&format!("{}DistortionFile", group)).cloned();
        *self.mag_gradient_file.lock().unwrap() =
            props.get(&format!("{}MagGradientFile", group)).cloned();
        self.binning
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        *self.fiducialess_alignment_a.lock().unwrap() = props
            .get(&format!("{}FiducialessAlignmentA", group))
            .map(|s| s.as_str())
            .unwrap_or("false")
            .eq_ignore_ascii_case("true");
        *self.fiducialess_alignment_b.lock().unwrap() = props
            .get(&format!("{}FiducialessAlignmentB", group))
            .map(|s| s.as_str())
            .unwrap_or("false")
            .eq_ignore_ascii_case("true");
        *self.whole_tomogram_sample_a.lock().unwrap() = props
            .get(&format!("{}WholeTomogramSampleA", group))
            .map(|s| s.as_str())
            .unwrap_or("false")
            .eq_ignore_ascii_case("true");
        *self.whole_tomogram_sample_b.lock().unwrap() = props
            .get(&format!("{}WholeTomogramSampleB", group))
            .map(|s| s.as_str())
            .unwrap_or("false")
            .eq_ignore_ascii_case("true");
        if let Some(param) = self.squeezevol_param.lock().unwrap().as_mut() {
            crate::imod::etomo::storage::storable::StorableValue::load_with_prepend(
                param, props, &prepend,
            );
        }
        self.use_z_factors_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_z_factors_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        if let Some(param) = self.transferfid_param_a.lock().unwrap().as_mut() {
            crate::imod::etomo::storage::storable::StorableValue::load_with_prepend(
                param, props, &prepend,
            );
        }
        if let Some(param) = self.transferfid_param_b.lock().unwrap().as_mut() {
            crate::imod::etomo::storage::storable::StorableValue::load_with_prepend(
                param, props, &prepend,
            );
        }
        self.size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .load(props, Some(&prepend));
        self.size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .load(props, Some(&prepend));
        let mut property_value = props
            .get(&format!(
                "{}{}",
                group,
                TOMO_GEN_A_TILT_PARALLEL_GROUP.as_str()
            ))
            .cloned();
        if property_value.is_some() {
            self.set_tomo_gen_tilt_parallel(AxisID::First, property_value.as_deref());
        }
        property_value = props
            .get(&format!(
                "{}{}",
                group,
                TOMO_GEN_B_TILT_PARALLEL_GROUP.as_str()
            ))
            .cloned();
        if property_value.is_some() {
            self.set_tomo_gen_tilt_parallel(AxisID::Second, property_value.as_deref());
        }
        property_value = props
            .get(&format!(
                "{}{}",
                group,
                FINAL_STACK_A_CTF_CORRECTION_PARALLEL_GROUP.as_str()
            ))
            .cloned();
        if property_value.is_some() {
            self.set_final_stack_ctf_correction_parallel_string(
                AxisID::First,
                property_value.as_deref(),
            );
        }
        property_value = props
            .get(&format!(
                "{}{}",
                group,
                FINAL_STACK_B_CTF_CORRECTION_PARALLEL_GROUP.as_str()
            ))
            .cloned();
        if property_value.is_some() {
            self.set_final_stack_ctf_correction_parallel_string(
                AxisID::Second,
                property_value.as_deref(),
            );
        }
        property_value = props
            .get(&format!(
                "{}{}",
                group,
                COMBINE_VOLCOMBINE_PARALLEL_GROUP.as_str()
            ))
            .cloned();
        if property_value.is_some() {
            self.set_combine_volcombine_parallel_string(property_value.as_deref());
        }
        property_value = props
            .get(&format!("{}{}", group, B_STACK_PROCESSED_GROUP))
            .cloned();
        if property_value.is_some() {
            self.set_b_stack_processed_string(property_value.as_deref());
        }
        self.default_parallel
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_thickness_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_thickness_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        // use default for backward compatibility, since this new parameter may not
        // be in any file yet
        *self.target_patch_size_x_and_y.lock().unwrap() = props
            .get(&format!(
                "{}tiltalign.{}",
                group, TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_KEY
            ))
            .cloned()
            .unwrap_or(TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_DEFAULT.to_string());
        *self.number_of_local_patches_x_and_y.lock().unwrap() = props
            .get(&format!(
                "{}tiltalign.{}",
                group, TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY
            ))
            .cloned()
            .unwrap_or(TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_DEFAULT.to_string());
        self.no_beam_tilt_selected_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_selected_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.no_beam_tilt_selected_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_selected_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_stack_polynomial_order_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_stack_polynomial_order_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_aligned_stack_dialog_saved_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.final_aligned_stack_dialog_saved_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomo_gen_trial_tomogram_name_list_a
            .lock()
            .unwrap()
            .lock()
            .unwrap()
            .load(props, &prepend);
        self.tomo_gen_trial_tomogram_name_list_b
            .lock()
            .unwrap()
            .lock()
            .unwrap()
            .load(props, &prepend);
        self.track_raptor_use_raw_stack_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_raptor_mark_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_raptor_diam_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_erase_gold_model_use_fid_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_erase_gold_model_use_fid_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_flatten_input_trim_vol
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_flatten_warp_contours_on_one_surface
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_flatten_warp_spacing_in_x
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_flatten_warp_spacing_in_y
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_squeeze_vol_input_trim_vol
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_low_pass_radius_sigma
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.reduce_filt_vol_deconvolution_strength
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_snr_falloff
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_high_pass_nyquist
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_defocus_in_microns
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_phase_shift
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.pos_binning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.pos_binning_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_3d_find_binning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_3d_find_binning_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_cur_tab
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_cur_tab
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_exists
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.lambda_for_smoothing
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.lambda_for_smoothing_list
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_overlap_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_overlap_of_patches_x_and_y_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_number_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_number_of_patches_x_and_y_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_length_and_overlap_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_length_and_overlap_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.track_method_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.fine_exists_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fine_exists_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_log_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_log_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_log_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_log_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_log_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_log_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_linear_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_linear_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_linear_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_linear_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_super_sample_factor_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_super_sample_factor_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_expand_input_lines_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_expand_input_lines_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_exists_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_exists_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.pos_exists_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.pos_exists_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_back_projection_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_back_projection_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_subarea_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_subarea_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_subarea_size_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_subarea_size_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_y_offset_of_subarea_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_y_offset_of_subarea_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_radial_radius_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_radial_radius_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_radial_sigma_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_radial_sigma_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.default_gpu_processing
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        let current = self.tilt_3d_find_tilt_parallel_a.lock().unwrap().take();
        *self.tilt_3d_find_tilt_parallel_a.lock().unwrap() = EtomoBoolean2::load_instance(
            current,
            TILT_3D_FIND_A_TILT_PARALLEL_KEY.as_str(),
            props,
            Some(&prepend),
        );
        let current = self.tilt_3d_find_tilt_parallel_b.lock().unwrap().take();
        *self.tilt_3d_find_tilt_parallel_b.lock().unwrap() = EtomoBoolean2::load_instance(
            current,
            TILT_3D_FIND_B_TILT_PARALLEL_KEY.as_str(),
            props,
            Some(&prepend),
        );
        self.erase_beads_initialized
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_seed_model_manual_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_seed_model_manual_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_seed_model_auto_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_seed_model_auto_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_seed_model_transfer_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_seed_model_transfer_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_exclude_inside_areas_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_exclude_inside_areas_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_just_find_shifts_near_zero_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_just_find_shifts_near_zero_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_target_number_of_beads_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_target_number_of_beads_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_target_density_of_beads_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_target_density_of_beads_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_clustered_points_allowed_elongated_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_clustered_points_allowed_elongated_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_clustered_points_allowed_elongated_value_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_clustered_points_allowed_elongated_value_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_advanced_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_advanced_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_3d_find_thickness_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_3d_find_thickness_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.set_fei_pixel_size
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_trimvol_new_style_z
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_trimvol_scaling_new_style_z
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_ctf_auto_fit_range_and_step_a
            .lock()
            .unwrap()
            .load(props, Some(&prepend));
        self.stack_ctf_auto_fit_range_and_step_b
            .lock()
            .unwrap()
            .load(props, Some(&prepend));
        self.orig_scope_template
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.orig_system_template
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.orig_user_template
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.is_twodir_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.is_twodir_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.twodir_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.twodir_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.is_dose_sym_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.is_dose_sym_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.dose_sym_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.dose_sym_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.seed_and_track_tab_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.seed_and_track_tab_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.raptor_tab_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.raptor_tab_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.coarse_antialias_filter_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.coarse_antialias_filter_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_antialias_filter_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_antialias_filter_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_elongated_points_allowed_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_elongated_points_allowed_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_lower_target_for_clustered_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_lower_target_for_clustered_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_set_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_set_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.weight_whole_tracks_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.weight_whole_tracks_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.length_of_pieces_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.length_of_pieces_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_type_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_type_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.has_gold_beads_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.has_gold_beads_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.positioning_bead_size_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.positioning_bead_size_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.extra_thickness_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.extra_thickness_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.extra_thickness_cryo_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.extra_thickness_cryo_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        if !self.track_length_and_overlap_a.lock().unwrap().is_empty() {
            if self.length_of_pieces_a.lock().unwrap().is_null() {
                let list = self.track_length_and_overlap_a.lock().unwrap().to_string();
                self.length_of_pieces_a
                    .lock()
                    .unwrap()
                    .set_string(utilities::get_element_from_list(Some(&list), 0).as_deref());
            }
            let list = self.track_length_and_overlap_a.lock().unwrap().to_string();
            self.minimum_overlap_a
                .lock()
                .unwrap()
                .set_string(utilities::get_element_from_list(Some(&list), 1).as_deref());
        }
        if !self.track_length_and_overlap_b.lock().unwrap().is_empty() {
            if self.length_of_pieces_b.lock().unwrap().is_null() {
                let list = self.track_length_and_overlap_b.lock().unwrap().to_string();
                self.length_of_pieces_b
                    .lock()
                    .unwrap()
                    .set_string(utilities::get_element_from_list(Some(&list), 0).as_deref());
            }
            let list = self.track_length_and_overlap_b.lock().unwrap().to_string();
            self.minimum_overlap_b
                .lock()
                .unwrap()
                .set_string(utilities::get_element_from_list(Some(&list), 1).as_deref());
        }
        // Backward compatability for raw image stack extension. The raw image stack extension
        // string and orig raw image stack extension string where give incorrect and
        // misleading property keys.
        let backward_compatability_correction = self
            .base
            .etomo_modified_version_lt(Some(CORRECTED_RAW_IMAGE_STACK_EXT_KEY_VERSION));
        if backward_compatability_correction {
            let mut key = utilities::create_property_key(
                Some(&prepend),
                Some(INCORRECT_RAW_IMAGE_STACK_EXT_KEY),
            );
            let value = key.as_ref().and_then(|key| props.get(key).cloned());
            self.raw_image_stack_ext
                .lock()
                .unwrap()
                .set(value.as_deref());
            *self
                .remove_incorrect_raw_image_stack_ext_key
                .lock()
                .unwrap() = true;
            key = utilities::create_property_key(
                Some(&prepend),
                Some(INCORRECT_ORIG_RAW_IMAGE_STACK_EXT_KEY),
            );
            let value = key.as_ref().and_then(|key| props.get(key).cloned());
            let value = self.strip_extension_divider(value.as_deref());
            self.orig_raw_image_stack_ext
                .lock()
                .unwrap()
                .set(value.as_deref());
            *self
                .remove_incorrect_orig_raw_image_stack_ext_key
                .lock()
                .unwrap() = true;
        }
        let raw_empty = self.raw_image_stack_ext.lock().unwrap().is_empty();
        if !backward_compatability_correction || raw_empty {
            self.raw_image_stack_ext
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        }
        let orig_empty = self.orig_raw_image_stack_ext.lock().unwrap().is_empty();
        if !backward_compatability_correction || orig_empty {
            self.orig_raw_image_stack_ext
                .lock()
                .unwrap()
                .load_with_prepend(Some(&mut props_copy), Some(&prepend));
            // This should not be necessary as the extension is now being saved without the ".".
            let value = self
                .orig_raw_image_stack_ext
                .lock()
                .unwrap()
                .to_string_option();
            let value = self.strip_extension_divider(value.as_deref());
            self.orig_raw_image_stack_ext
                .lock()
                .unwrap()
                .set(value.as_deref());
        }
        self.orig_raw_image_stack_ext_lock
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.target_measurement_ratio_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.target_measurement_ratio_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.min_measurement_ratio_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.min_measurement_ratio_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.order_of_restrictions_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.order_of_restrictions_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.skip_beam_tilt_with_one_rot_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.skip_beam_tilt_with_one_rot_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fine_local_align_validation_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fine_local_align_validation_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_hamming_like_filter_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_hamming_like_filter_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_fake_sirt_iterations_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_fake_sirt_iterations_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_exact_filter_size_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_exact_filter_size_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.sirt_radial_radius_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.sirt_radial_radius_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.sirt_radial_sigma_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.sirt_radial_sigma_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.batch_run_tomo_log_read_timestamp
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.batch_run_tomo_log_read_finished
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.positioning_new_dialog_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.positioning_new_dialog_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_fake_sirt_iterations_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_fake_sirt_iterations_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_exact_object_sizes_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_exact_object_sizes_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_gaussian_cutoffs_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_gaussian_cutoffs_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_gaussian_falloffs_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_gaussian_falloffs_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_hamming_like_starts_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_filter_trials_hamming_like_starts_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_ctf_3d_old_style_xtilting_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_old_style_xtilting_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_vertical_slices_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_vertical_slices_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_fourier_reduce_by_factor_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_fourier_reduce_by_factor_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_low_pass_radius_sigma_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_low_pass_radius_sigma_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_mtf_file_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_mtf_file_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_maximum_inverse_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_maximum_inverse_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_optimal_dose_scaling_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_optimal_dose_scaling_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_bidirectional_num_views_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_bidirectional_num_views_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_stack_ctf_phase_flip_x_axis_tilt_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_x_axis_tilt_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_scale_by_ctf_power_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_scale_by_ctf_power_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.use_stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_stack_mtf_filter_fixed_image_dose_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_fixed_image_dose_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_dose_weighting_file_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_dose_weighting_file_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_mtf_filter_type_of_dose_file_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_type_of_dose_file_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_voltage_200_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_voltage_200_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_sirt_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_sirt_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        *self.batch_run_tomo_log_read_axis_id.lock().unwrap() = AxisID::get_instance(
            props
                .get(&format!("{}{}", group, BATCH_RUN_TOMO_LOG_FILE_AXIS_ID_KEY))
                .map(|s| s.as_str()),
        );
        self.post_trimvol_swap_yz_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_rotate_x_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_convert_to_bytes_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_section_scale_min_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_section_scale_max_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.post_trimvol_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.post_trimvol_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.post_trimvol_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.post_trimvol_z_min_from_batchruntomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.post_trimvol_z_max_from_batchruntomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.post_trimvol_scale_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_scale_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_scale_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_trimvol_scale_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.stack_aligned_stack_erase_gold_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.orig_raw_image_stack_ext_from_batch_run_tomo
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        // Bug# 2403 - complex image filename style correct must be done in child class.
        self.check_image_filename_style_loaded(&prepend);
        self.subtomo_reorientation_type_none
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.subtomo_reorientation_type_flipped
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.subtomo_reorientation_type_rotated
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.subtomo_make_volume_stacks
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.subtomo_extent_of_z_levels_in_nm
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.subtomo_new_aligned_binning
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.subtomo_fourier_reduce_by_factor
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.alt_tomo_trim_volume
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_archive_orig_stack
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.ctf_3d_setup_slab_thickness_in_nm_set
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
    }

    // Java `repairImageFilenameStyle` overrides `BaseMetaData`; it is in the
    // `BaseMetaData` impl below.

    /// Java private `stripExtensionDivider`.  Removes "." from the start of fileExt if
    /// necessary.  For backwards compatibility.
    fn strip_extension_divider(&self, file_ext: Option<&str>) -> Option<String> {
        if let Some(file_ext) = file_ext {
            if file_ext.starts_with(EXTENSION_DIVIDER) && file_ext.encode_utf16().count() > 1 {
                return Some(file_ext[1..].to_string());
            }
        }
        file_ext.map(|s| s.to_string())
    }

    /// Java `isTrackElongatedPointsAllowedNull`.
    pub fn is_track_elongated_points_allowed_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .track_elongated_points_allowed_b
                .lock()
                .unwrap()
                .is_null();
        }
        self.track_elongated_points_allowed_a
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `getTrackElongatedPointsAllowed`.
    pub fn get_track_elongated_points_allowed(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self
                .track_elongated_points_allowed_b
                .lock()
                .unwrap()
                .clone();
        }
        self.track_elongated_points_allowed_a
            .lock()
            .unwrap()
            .clone()
    }

    /// Java `setTrackElongatedPointsAllowed`.
    pub fn set_track_elongated_points_allowed(&self, axis_id: AxisID, input: Option<Number>) {
        if axis_id == AxisID::Second {
            self.track_elongated_points_allowed_b
                .lock()
                .unwrap()
                .set_number(input);
        } else {
            self.track_elongated_points_allowed_a
                .lock()
                .unwrap()
                .set_number(input);
        }
    }

    /// Java `isTrackLowerTargetForClusteredNull`.
    pub fn is_track_lower_target_for_clustered_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .track_lower_target_for_clustered_b
                .lock()
                .unwrap()
                .is_null();
        }
        self.track_lower_target_for_clustered_a
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `getTrackLowerTargetForClustered`.
    pub fn get_track_lower_target_for_clustered(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .track_lower_target_for_clustered_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.track_lower_target_for_clustered_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setTrackLowerTargetForClustered`.
    pub fn set_track_lower_target_for_clustered(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.track_lower_target_for_clustered_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.track_lower_target_for_clustered_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setNoBeamTiltSelected`.
    pub fn set_no_beam_tilt_selected(&self, axis_id: AxisID, selected: bool) {
        if axis_id == AxisID::Second {
            self.no_beam_tilt_selected_b
                .lock()
                .unwrap()
                .set_boolean(selected);
        } else {
            self.no_beam_tilt_selected_a
                .lock()
                .unwrap()
                .set_boolean(selected);
        }
    }

    /// Java `setFixedBeamTiltSelected`.
    pub fn set_fixed_beam_tilt_selected(&self, axis_id: AxisID, selected: bool) {
        if axis_id == AxisID::Second {
            self.fixed_beam_tilt_selected_b
                .lock()
                .unwrap()
                .set_boolean(selected);
        } else {
            self.fixed_beam_tilt_selected_a
                .lock()
                .unwrap()
                .set_boolean(selected);
        }
    }

    /// Java `setGenLog`.
    pub fn set_gen_log(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_log_b.lock().unwrap().set_string(input);
        } else {
            self.gen_log_a.lock().unwrap().set_string(input);
        }
    }

    /// Java `setGenScaleFactorLog`.
    pub fn set_gen_scale_factor_log(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_scale_factor_log_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.gen_scale_factor_log_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setGenScaleOffsetLog`.
    pub fn set_gen_scale_offset_log(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_scale_offset_log_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.gen_scale_offset_log_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setGenScaleFactorLinear`.
    pub fn set_gen_scale_factor_linear(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_scale_factor_linear_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.gen_scale_factor_linear_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setGenScaleOffsetLinear`.
    pub fn set_gen_scale_offset_linear(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_scale_offset_linear_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.gen_scale_offset_linear_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setSuperSampleFactor`.
    pub fn set_super_sample_factor(&self, axis_id: AxisID, input: Option<Number>) {
        if axis_id == AxisID::Second {
            self.gen_super_sample_factor_b
                .lock()
                .unwrap()
                .set_number(input);
        } else {
            self.gen_super_sample_factor_a
                .lock()
                .unwrap()
                .set_number(input);
        }
    }

    /// Java `setExpandInputLines`.
    pub fn set_expand_input_lines(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.gen_expand_input_lines_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.gen_expand_input_lines_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setFixedBeamTilt`.
    pub fn set_fixed_beam_tilt(&self, axis_id: AxisID, fixed_beam_tilt: Option<&str>) {
        if axis_id == AxisID::Second {
            self.fixed_beam_tilt_b
                .lock()
                .unwrap()
                .set_string(fixed_beam_tilt);
        } else {
            self.fixed_beam_tilt_a
                .lock()
                .unwrap()
                .set_string(fixed_beam_tilt);
        }
    }

    /// Java `setFinalStackFiducialDiameter`.
    pub fn set_final_stack_fiducial_diameter(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.final_stack_fiducial_diameter_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.final_stack_fiducial_diameter_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setFinalStackExpandCircleIterations(AxisID, Object)`.  The source casts
    /// the `Object` to `Number` (a spinner value); the parameter is that `Number`.
    pub fn set_final_stack_expand_circle_iterations_object(
        &self,
        axis_id: AxisID,
        input: Option<Number>,
    ) {
        if axis_id == AxisID::Second {
            self.final_stack_expand_circle_iterations_b
                .lock()
                .unwrap()
                .set_number(input);
        } else {
            self.final_stack_expand_circle_iterations_a
                .lock()
                .unwrap()
                .set_number(input);
        }
    }

    /// Java `setFinalStackExpandCircleIterations`.
    pub fn set_final_stack_expand_circle_iterations_string(
        &self,
        axis_id: AxisID,
        input: Option<&str>,
    ) {
        if axis_id == AxisID::Second {
            self.final_stack_expand_circle_iterations_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.final_stack_expand_circle_iterations_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setUseFinalStackExpandCircleIterations`.
    pub fn set_use_final_stack_expand_circle_iterations(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_final_stack_expand_circle_iterations_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_final_stack_expand_circle_iterations_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setFinalStackPolynomialOrder`.
    pub fn set_final_stack_polynomial_order(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.final_stack_polynomial_order_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.final_stack_polynomial_order_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setFinalAlignedStackDialogSaved`.
    pub fn set_final_aligned_stack_dialog_saved(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.final_aligned_stack_dialog_saved_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.final_aligned_stack_dialog_saved_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setTargetPatchSizeXandY`.
    pub fn set_target_patch_size_x_and_y(&self, target_patch_size_x_and_y: Option<&str>) {
        *self.target_patch_size_x_and_y.lock().unwrap() = target_patch_size_x_and_y
            .map(|s| s.to_string())
            .unwrap_or_default();
    }

    /// Java `setNumberOfLocalPatchesXandY`.
    pub fn set_number_of_local_patches_x_and_y(
        &self,
        number_of_local_patches_x_and_y: Option<&str>,
    ) {
        *self.number_of_local_patches_x_and_y.lock().unwrap() = number_of_local_patches_x_and_y
            .map(|s| s.to_string())
            .unwrap_or_default();
    }

    /// Java `setFiducialess`.
    pub fn set_fiducialess(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.fiducialess_b.lock().unwrap().set_boolean(input);
        } else {
            self.fiducialess_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `getFirstAxisPrepend`.
    pub fn get_first_axis_prepend(&self) -> Option<String> {
        self.first_axis_prepend.lock().unwrap().clone()
    }

    /// Java `getSecondAxisPrepend`.
    pub fn get_second_axis_prepend(&self) -> Option<String> {
        self.second_axis_prepend.lock().unwrap().clone()
    }

    /// Java `getGroupKey`.
    pub fn get_group_key(&self) -> String {
        "Setup".to_string()
    }

    /// Java private `setProperty`.
    fn set_property(
        &self,
        props: &mut BTreeMap<String, String>,
        group: &str,
        key: &str,
        value: Option<&str>,
    ) {
        match value {
            None => {
                props.remove(&format!("{}{}", group, key));
            }
            Some(value) => {
                props.insert(format!("{}{}", group, key), value.to_string());
            }
        }
    }

    /// Java `store(Properties, String)`.  Insert the objects attributes into the
    /// properties object.
    ///
    /// Upstream bug fixed in translation (MetaData.java:1250-1253): the constructor sets
    /// `stackCtfAutoFitRangeAndStepA`'s properties key twice - the second time to the
    /// B key - and never sets `stackCtfAutoFitRangeAndStepB`'s.  Natively A is stored
    /// under `Setup.Stack.B.CTF.AutoFit.RangeAndStep` and B, with no key of its own,
    /// under the bare group key `Setup` (the reference `.edf` carries a
    /// `Setup=-Infinity,-Infinity` line).  The constructor now gives each its own key,
    /// so A is stored under `...Stack.A...` and B under `...Stack.B...`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let created = self.create_prepend(prepend);
        self.base
            .store_with_created_prepend(props, created.as_deref());
        let prepend = self.create_prepend(prepend).unwrap_or("null".to_string());
        let group = format!("{}.", prepend);
        // TODO is this setting the latest revision number each time??
        self.set_property(
            props,
            &group,
            "RevisionNumber",
            Some(LATEST_REVISION_NUMBER),
        );
        let value = self.com_scripts_created.lock().unwrap().to_string();
        self.set_property(props, &group, "ComScriptsCreated", Some(&value));
        let value = self.dataset_name.lock().unwrap().clone();
        self.set_property(props, &group, "DatasetName", Some(&value));
        let value = self.backup_directory.lock().unwrap().clone();
        self.set_property(props, &group, "BackupDirectory", Some(&value));
        let value = self.data_source.lock().unwrap().to_string();
        self.set_property(props, &group, "DataSource", Some(&value));
        let value = self.base.axis_type.lock().unwrap().to_string();
        self.set_property(props, &group, "AxisType", Some(&value));
        let value = self.view_type.lock().unwrap().to_string();
        self.set_property(props, &group, "ViewType", Some(&value));
        let value = java_lang_double_to_string(*self.pixel_size.lock().unwrap());
        self.set_property(props, &group, "PixelSize", Some(&value));
        self.half_float_mode_output
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        let value = self.use_local_alignments_a.lock().unwrap().to_string();
        self.set_property(props, &group, "UseLocalAlignmentsA", Some(&value));
        let value = self.use_local_alignments_b.lock().unwrap().to_string();
        self.set_property(props, &group, "UseLocalAlignmentsB", Some(&value));
        let value = java_lang_double_to_string(*self.fiducial_diameter.lock().unwrap());
        self.set_property(props, &group, "FiducialDiameter", Some(&value));
        let value = self.image_rotation_a.lock().unwrap().to_string();
        self.set_property(props, &group, "ImageRotationA", Some(&value));
        let value = self.image_rotation_b.lock().unwrap().to_string();
        self.set_property(props, &group, "ImageRotationB", Some(&value));
        self.tilt_angle_spec_a
            .lock()
            .unwrap()
            .store_with_prepend(props, &format!("{}AxisA", group));
        if self.exclude_projections_a.lock().unwrap().is_none() {
            props.remove(&format!("{}AxisA.ExcludeProjections", group));
        } else {
            let value = self.exclude_projections_a.lock().unwrap().clone().unwrap();
            self.set_property(props, &group, "AxisA.ExcludeProjections", Some(&value));
        }
        self.tilt_angle_spec_b
            .lock()
            .unwrap()
            .store_with_prepend(props, &format!("{}AxisB", group));
        if self.exclude_projections_b.lock().unwrap().is_none() {
            props.remove(&format!("{}AxisB.ExcludeProjections", group));
        } else {
            let value = self.exclude_projections_b.lock().unwrap().clone().unwrap();
            self.set_property(props, &group, "AxisB.ExcludeProjections", Some(&value));
        }
        storable::StorableValue::store_with_prepend(
            &*self.combine_params.lock().unwrap(),
            props,
            &group,
        );
        let value = self.distortion_file.lock().unwrap().clone();
        self.set_property(props, &group, "DistortionFile", value.as_deref());
        let value = self.mag_gradient_file.lock().unwrap().clone();
        self.set_property(props, &group, "MagGradientFile", value.as_deref());
        self.binning
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        let value = self.fiducialess_alignment_a.lock().unwrap().to_string();
        self.set_property(props, &group, "FiducialessAlignmentA", Some(&value));
        let value = self.fiducialess_alignment_b.lock().unwrap().to_string();
        self.set_property(props, &group, "FiducialessAlignmentB", Some(&value));
        let value = self.whole_tomogram_sample_a.lock().unwrap().to_string();
        self.set_property(props, &group, "WholeTomogramSampleA", Some(&value));
        let value = self.whole_tomogram_sample_b.lock().unwrap().to_string();
        self.set_property(props, &group, "WholeTomogramSampleB", Some(&value));
        if let Some(param) = self.squeezevol_param.lock().unwrap().as_ref() {
            crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
                param, props, &prepend,
            );
        }
        self.use_z_factors_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_z_factors_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        if let Some(param) = self.transferfid_param_a.lock().unwrap().as_ref() {
            crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
                param, props, &prepend,
            );
        }
        if let Some(param) = self.transferfid_param_b.lock().unwrap().as_ref() {
            crate::imod::etomo::storage::storable::StorableValue::store_with_prepend(
                param, props, &prepend,
            );
        }
        if let Some(field) = self.tomo_gen_tilt_parallel_a.lock().unwrap().as_ref() {
            field.store_with_prepend(props, Some(&prepend));
        }
        if let Some(field) = self.tomo_gen_tilt_parallel_b.lock().unwrap().as_ref() {
            field.store_with_prepend(props, Some(&prepend));
        }
        EtomoBoolean2::store_instance(
            self.tilt_3d_find_tilt_parallel_a.lock().unwrap().as_ref(),
            props,
            Some(&prepend),
            TILT_3D_FIND_A_TILT_PARALLEL_KEY.as_str(),
        );
        EtomoBoolean2::store_instance(
            self.tilt_3d_find_tilt_parallel_b.lock().unwrap().as_ref(),
            props,
            Some(&prepend),
            TILT_3D_FIND_B_TILT_PARALLEL_KEY.as_str(),
        );
        if let Some(field) = self
            .final_stack_ctf_correction_parallel_a
            .lock()
            .unwrap()
            .as_ref()
        {
            field.store_with_prepend(props, Some(&prepend));
        }
        if let Some(field) = self
            .final_stack_ctf_correction_parallel_b
            .lock()
            .unwrap()
            .as_ref()
        {
            field.store_with_prepend(props, Some(&prepend));
        }
        if let Some(field) = self.combine_volcombine_parallel.lock().unwrap().as_ref() {
            field.store_with_prepend(props, Some(&prepend));
        }
        if let Some(field) = self.b_stack_processed.lock().unwrap().as_ref() {
            field.store_with_prepend(props, Some(&prepend));
        }
        self.default_parallel
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_thickness_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_thickness_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fiducialess_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fiducialess_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        let value = self.target_patch_size_x_and_y.lock().unwrap().clone();
        self.set_property(
            props,
            &group,
            &format!("tiltalign.{}", TILTALIGN_TARGET_PATCH_SIZE_X_AND_Y_KEY),
            Some(&value),
        );
        let value = self.number_of_local_patches_x_and_y.lock().unwrap().clone();
        self.set_property(
            props,
            &group,
            &format!(
                "tiltalign.{}",
                TILTALIGN_NUMBER_OF_LOCAL_PATCHES_X_AND_Y_KEY
            ),
            Some(&value),
        );
        self.no_beam_tilt_selected_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_selected_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.no_beam_tilt_selected_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_selected_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fixed_beam_tilt_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .store(props, Some(&prepend));
        self.size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .store(props, Some(&prepend));
        self.final_stack_polynomial_order_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_stack_polynomial_order_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_aligned_stack_dialog_saved_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_aligned_stack_dialog_saved_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_stack_fiducial_diameter_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_stack_fiducial_diameter_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.final_stack_expand_circle_iterations_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_final_stack_expand_circle_iterations_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomo_gen_trial_tomogram_name_list_a
            .lock()
            .unwrap()
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.tomo_gen_trial_tomogram_name_list_b
            .lock()
            .unwrap()
            .lock()
            .unwrap()
            .store(props, &prepend);
        self.track_raptor_use_raw_stack_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_raptor_mark_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_raptor_diam_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_erase_gold_model_use_fid_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_erase_gold_model_use_fid_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_flatten_input_trim_vol
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_flatten_warp_contours_on_one_surface
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_flatten_warp_spacing_in_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_flatten_warp_spacing_in_y
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_squeeze_vol_input_trim_vol
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_low_pass_radius_sigma
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.reduce_filt_vol_deconvolution_strength
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_snr_falloff
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_high_pass_nyquist
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_defocus_in_microns
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_phase_shift
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.pos_binning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.pos_binning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_binning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_binning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_3d_find_binning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_3d_find_binning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_cur_tab
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_cur_tab
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_exists
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.lambda_for_smoothing
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.lambda_for_smoothing_list
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.track_overlap_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.track_overlap_of_patches_x_and_y_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.track_number_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.track_number_of_patches_x_and_y_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.track_method_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.track_method_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.fine_exists_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fine_exists_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_log_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_log_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_log_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_log_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_log_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_log_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_linear_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_factor_linear_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_linear_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_scale_offset_linear_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_super_sample_factor_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_super_sample_factor_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_expand_input_lines_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_expand_input_lines_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_exists_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_exists_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.default_gpu_processing
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.pos_exists_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.pos_exists_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_back_projection_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_back_projection_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_subarea_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_subarea_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_subarea_size_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_subarea_size_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_y_offset_of_subarea_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_y_offset_of_subarea_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_radial_radius_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_radial_radius_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_radial_sigma_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_radial_sigma_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_x_min
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_x_max
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_y_min
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_y_max
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_z_min
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_z_max
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_convert_to_bytes
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_fixed_scaling
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_flipped_volume
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_section_scale_min
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_section_scale_max
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_fixed_scale_min
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_fixed_scale_max
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.post_trimvol_swap_yz
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_rotate_x
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_scale_x_min
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_scale_x_max
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_scale_y_min
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_scale_y_max
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.erase_beads_initialized
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_seed_model_manual_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_seed_model_manual_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_seed_model_auto_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_seed_model_auto_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_seed_model_transfer_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_seed_model_transfer_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_exclude_inside_areas_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_exclude_inside_areas_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_just_find_shifts_near_zero_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_just_find_shifts_near_zero_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_target_number_of_beads_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_target_number_of_beads_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_target_density_of_beads_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_target_density_of_beads_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_advanced_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_advanced_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_3d_find_thickness_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_3d_find_thickness_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.set_fei_pixel_size
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_new_style_z
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_trimvol_scaling_new_style_z
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_ctf_auto_fit_range_and_step_a
            .lock()
            .unwrap()
            .store(props, Some(&prepend));
        self.stack_ctf_auto_fit_range_and_step_b
            .lock()
            .unwrap()
            .store(props, Some(&prepend));
        self.orig_scope_template
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.orig_system_template
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.orig_user_template
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.is_twodir_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.is_twodir_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.twodir_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.twodir_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.is_dose_sym_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.is_dose_sym_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.dose_sym_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.dose_sym_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.seed_and_track_tab_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.seed_and_track_tab_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.raptor_tab_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.raptor_tab_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.coarse_antialias_filter_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.coarse_antialias_filter_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_antialias_filter_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_antialias_filter_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_elongated_points_allowed_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_elongated_points_allowed_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_lower_target_for_clustered_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_lower_target_for_clustered_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_set_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.orig_views_with_mag_changes_set_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.weight_whole_tracks_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.weight_whole_tracks_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.length_of_pieces_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.length_of_pieces_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.raw_image_stack_ext
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        if *self
            .remove_incorrect_raw_image_stack_ext_key
            .lock()
            .unwrap()
        {
            if let Some(key) = utilities::create_property_key(
                Some(&prepend),
                Some(INCORRECT_RAW_IMAGE_STACK_EXT_KEY),
            ) {
                props.remove(&key);
            }
        }
        self.orig_raw_image_stack_ext
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.orig_raw_image_stack_ext_lock
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        if *self
            .remove_incorrect_orig_raw_image_stack_ext_key
            .lock()
            .unwrap()
        {
            if let Some(key) = utilities::create_property_key(
                Some(&prepend),
                Some(INCORRECT_ORIG_RAW_IMAGE_STACK_EXT_KEY),
            ) {
                props.remove(&key);
            }
        }
        self.target_measurement_ratio_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.target_measurement_ratio_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.min_measurement_ratio_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.min_measurement_ratio_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.order_of_restrictions_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.order_of_restrictions_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.skip_beam_tilt_with_one_rot_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.skip_beam_tilt_with_one_rot_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fine_local_align_validation_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fine_local_align_validation_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_type_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_type_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.has_gold_beads_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.has_gold_beads_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.positioning_bead_size_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.positioning_bead_size_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.extra_thickness_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.extra_thickness_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.extra_thickness_cryo_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.extra_thickness_cryo_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_hamming_like_filter_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_hamming_like_filter_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_fake_sirt_iterations_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_fake_sirt_iterations_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_exact_filter_size_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_exact_filter_size_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.sirt_radial_radius_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.sirt_radial_radius_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.sirt_radial_sigma_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.sirt_radial_sigma_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.batch_run_tomo_log_read_timestamp
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.batch_run_tomo_log_read_finished
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.positioning_new_dialog_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.positioning_new_dialog_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_fake_sirt_iterations_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_fake_sirt_iterations_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_exact_object_sizes_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_exact_object_sizes_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_hamming_like_starts_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_hamming_like_starts_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_ctf_3d_old_style_xtilting_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_old_style_xtilting_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_vertical_slices_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_vertical_slices_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_fourier_reduce_by_factor_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_ctf_3d_fourier_reduce_by_factor_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_filter_trials_gaussian_cutoffs_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_gaussian_cutoffs_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_gaussian_falloffs_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_filter_trials_gaussian_falloffs_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_low_pass_radius_sigma_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_low_pass_radius_sigma_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_mtf_file_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_mtf_file_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_maximum_inverse_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_maximum_inverse_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_optimal_dose_scaling_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_optimal_dose_scaling_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_bidirectional_num_views_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_bidirectional_num_views_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_stack_ctf_phase_flip_x_axis_tilt_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_x_axis_tilt_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_x_axis_tilt_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_scale_by_ctf_power_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_ctf_phase_flip_scale_by_ctf_power_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_inverse_rolloff_radius_sigma_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.use_stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_stack_mtf_filter_fixed_image_dose_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_fixed_image_dose_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_fixed_image_dose_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_dose_weighting_file_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_dose_weighting_file_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_mtf_filter_type_of_dose_file_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_type_of_dose_file_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_voltage_200_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_mtf_filter_voltage_200_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.raw_image_stack_ext
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_sirt_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_sirt_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        if self
            .batch_run_tomo_log_read_axis_id
            .lock()
            .unwrap()
            .is_some()
        {
            let value = self
                .batch_run_tomo_log_read_axis_id
                .lock()
                .unwrap()
                .unwrap()
                .get_extension();
            self.set_property(
                props,
                &group,
                BATCH_RUN_TOMO_LOG_FILE_AXIS_ID_KEY,
                Some(&value),
            );
        }
        self.post_trimvol_swap_yz_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_rotate_x_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_convert_to_bytes_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_section_scale_min_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_section_scale_max_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.post_trimvol_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.post_trimvol_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.post_trimvol_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.post_trimvol_z_min_from_batchruntomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.post_trimvol_z_max_from_batchruntomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.post_trimvol_scale_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_scale_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_scale_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_trimvol_scale_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.stack_aligned_stack_erase_gold_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.orig_raw_image_stack_ext_from_batch_run_tomo
            .lock()
            .unwrap()
            .store(Some(props));
        self.subtomo_reorientation_type_none
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.subtomo_reorientation_type_flipped
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.subtomo_reorientation_type_rotated
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.subtomo_make_volume_stacks
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.subtomo_extent_of_z_levels_in_nm
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.subtomo_new_aligned_binning
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.subtomo_fourier_reduce_by_factor
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.alt_tomo_trim_volume
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.alt_tomo_archive_orig_stack
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.ctf_3d_setup_slab_thickness_in_nm_set
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
    }

    /// Java `getExtraThickness`.
    pub fn get_extra_thickness(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.extra_thickness_b.lock().unwrap().to_string();
        }
        self.extra_thickness_a.lock().unwrap().to_string()
    }

    /// Java `getExtraThicknessCryo`.
    pub fn get_extra_thickness_cryo(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.extra_thickness_cryo_b.lock().unwrap().to_string();
        }
        self.extra_thickness_cryo_a.lock().unwrap().to_string()
    }

    /// Java `setExtraThicknessCryo`.
    pub fn set_extra_thickness_cryo(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.extra_thickness_cryo_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.extra_thickness_cryo_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setExtraThickness`.
    pub fn set_extra_thickness(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.extra_thickness_b.lock().unwrap().set_string(input);
        } else {
            self.extra_thickness_a.lock().unwrap().set_string(input);
        }
    }

    /// Java `isPositioningFiducialDiameterNull`.  Deprecated 2/19/2018: incorrectly
    /// named member variable - use ...BeadSize.
    pub fn is_positioning_fiducial_diameter_null(&self, axis_id: AxisID) -> bool {
        self.is_positioning_bead_size_null(axis_id)
    }

    /// Java `isPositioningBeadSizeNull`.
    pub fn is_positioning_bead_size_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.positioning_bead_size_b.lock().unwrap().is_null();
        }
        self.positioning_bead_size_a.lock().unwrap().is_null()
    }

    /// Java `isHasGoldBeadsNull`.
    pub fn is_has_gold_beads_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.has_gold_beads_b.lock().unwrap().is_null();
        }
        self.has_gold_beads_a.lock().unwrap().is_null()
    }

    /// Java `setPositioningFiducialDiameter`.  Deprecated 2/19/2018: incorrectly named
    /// member variable - use ...BeadSize.
    pub fn set_positioning_fiducial_diameter(&self, axis_id: AxisID, input: Option<&str>) {
        self.set_positioning_bead_size(axis_id, input);
    }

    /// Java `setPositioningBeadSize`.
    pub fn set_positioning_bead_size(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.positioning_bead_size_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.positioning_bead_size_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `getPositioningFiducialDiameter`.
    pub fn get_positioning_fiducial_diameter(&self, axis_id: AxisID) -> f64 {
        if axis_id == AxisID::Second {
            return self.positioning_bead_size_b.lock().unwrap().get_double();
        }
        self.positioning_bead_size_a.lock().unwrap().get_double()
    }

    /// Java `getPositioningBeadSize`.
    pub fn get_positioning_bead_size(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.positioning_bead_size_b.lock().unwrap().to_string();
        }
        self.positioning_bead_size_a.lock().unwrap().to_string()
    }

    /// Java `setHasGoldBeads`.
    pub fn set_has_gold_beads(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.has_gold_beads_b.lock().unwrap().set_boolean(input);
        } else {
            self.has_gold_beads_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `isHasGoldBeads`.
    pub fn is_has_gold_beads(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.has_gold_beads_b.lock().unwrap().is();
        }
        self.has_gold_beads_a.lock().unwrap().is()
    }

    /// Java `setSampleType(AxisID, SampleType)`.
    pub fn set_sample_type_sample_type(&self, axis_id: AxisID, input: Option<SampleType>) {
        if axis_id == AxisID::Second {
            match input {
                None => {
                    self.sample_type_b.lock().unwrap().reset();
                }
                Some(input) => {
                    let value = input.get_value();
                    self.sample_type_b
                        .lock()
                        .unwrap()
                        .set_const_etomo_number(Some(&value));
                }
            }
        } else {
            match input {
                None => {
                    self.sample_type_a.lock().unwrap().reset();
                }
                Some(input) => {
                    let value = input.get_value();
                    self.sample_type_a
                        .lock()
                        .unwrap()
                        .set_const_etomo_number(Some(&value));
                }
            }
        }
    }

    /// Java `setSampleType`.
    pub fn set_sample_type_string(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.sample_type_b.lock().unwrap().set_string(input);
        } else {
            self.sample_type_a.lock().unwrap().set_string(input);
        }
    }

    /// Java `getSampleType`.
    pub fn get_sample_type(&self, axis_id: AxisID) -> Option<SampleType> {
        if axis_id == AxisID::Second {
            let sample_type_b = self.sample_type_b.lock().unwrap().clone();
            return SampleType::get_instance(Some(&sample_type_b.base), false);
        }
        let sample_type_a = self.sample_type_a.lock().unwrap().clone();
        SampleType::get_instance(Some(&sample_type_a.base), false)
    }

    /// Java `setAutoPatchFinalSize(String)`.
    pub fn set_auto_patch_final_size(&self, input: Option<&str>) {
        self.combine_params
            .lock()
            .unwrap()
            .set_patch_size_string(true, input);
    }

    /// Java `setExtraResidualTargets(String)`.
    pub fn set_extra_residual_targets(&self, input: Option<&str>) {
        self.combine_params
            .lock()
            .unwrap()
            .set_extra_residual_targets(input);
    }

    /// Java `setWedgeReductionFraction(String)`.
    pub fn set_wedge_reduction_fraction(&self, input: Option<&str>) {
        self.combine_params
            .lock()
            .unwrap()
            .set_wedge_reduction_fraction(input);
    }

    /// Java `setLowFromBothRadius(String)`.
    pub fn set_low_from_both_radius(&self, input: Option<&str>) {
        self.combine_params
            .lock()
            .unwrap()
            .set_low_from_both_radius(input);
    }

    /// Java `setPatchTypeOrXYZ(String)`.
    pub fn set_patch_type_or_xyz(&self, input: Option<&str>) {
        self.combine_params
            .lock()
            .unwrap()
            .set_patch_size_string(false, input);
    }

    /// Java `resetBatchruntomoCombineSettings`.
    pub fn reset_batchruntomo_combine_settings(&self) {
        self.combine_params
            .lock()
            .unwrap()
            .reset_batchruntomo_combine_settings();
    }

    /// Java `moveBatchruntomoSettings`.
    pub fn move_batchruntomo_settings(&self) {
        self.combine_params
            .lock()
            .unwrap()
            .move_batchruntomo_settings();
        let erase_gold_from_batchruntomo = self
            .stack_aligned_stack_erase_gold_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !erase_gold_from_batchruntomo.is_null() {
            let fid = EraseGold::Fid.get_value();
            self.stack_erase_gold_model_use_fid_a
                .lock()
                .unwrap()
                .set_boolean(erase_gold_from_batchruntomo.equals_const_etomo_number(Some(&fid)));
            self.stack_aligned_stack_erase_gold_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let orig_from_batchruntomo = {
            let field = self
                .orig_raw_image_stack_ext_from_batch_run_tomo
                .lock()
                .unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(orig_from_batchruntomo) = orig_from_batchruntomo {
            self.orig_raw_image_stack_ext
                .lock()
                .unwrap()
                .set(Some(&orig_from_batchruntomo));
            self.orig_raw_image_stack_ext_lock
                .lock()
                .unwrap()
                .set_boolean(true);
            self.orig_raw_image_stack_ext_from_batch_run_tomo
                .lock()
                .unwrap()
                .reset();
        }
    }

    /// Java `movePostTrimvolBatchruntomoSettings`.  Move batchruntomo trimvol.
    pub fn move_post_trimvol_batchruntomo_settings(&self, display: Option<&dyn TrimvolDisplay>) {
        let from = self
            .post_trimvol_swap_yz_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_swap_yz
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&from.base));
            if let Some(display) = display {
                display.set_swap_yz(from.is());
            }
            self.post_trimvol_swap_yz_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_rotate_x_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_rotate_x
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&from.base));
            if let Some(display) = display {
                display.set_rotate_x(from.is());
            }
            self.post_trimvol_rotate_x_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_convert_to_bytes_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_convert_to_bytes
                .lock()
                .unwrap()
                .set_const_etomo_number(Some(&from.base));
            if let Some(display) = display {
                display.set_convert_to_bytes(from.is());
            }
            self.post_trimvol_convert_to_bytes_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_section_scale_min_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_section_scale_min
                .lock()
                .unwrap()
                .set(Some(&from.to_string()));
            if let Some(display) = display {
                display.set_section_scale_min(&from.to_string());
            }
            self.post_trimvol_section_scale_min_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_section_scale_max_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_section_scale_max
                .lock()
                .unwrap()
                .set(Some(&from.to_string()));
            if let Some(display) = display {
                display.set_section_scale_max(&from.to_string());
            }
            self.post_trimvol_section_scale_max_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = {
            let field = self.post_trimvol_x_min_from_batchruntomo.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(from) = from {
            self.post_trimvol_x_min.lock().unwrap().set(Some(&from));
            if let Some(display) = display {
                display.set_x_min(&from);
            }
            self.post_trimvol_x_min_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = {
            let field = self.post_trimvol_x_max_from_batchruntomo.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(from) = from {
            self.post_trimvol_x_max.lock().unwrap().set(Some(&from));
            if let Some(display) = display {
                display.set_x_max(&from);
            }
            self.post_trimvol_x_max_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = {
            let field = self.post_trimvol_y_min_from_batchruntomo.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(from) = from {
            self.post_trimvol_y_min.lock().unwrap().set(Some(&from));
            if let Some(display) = display {
                display.set_y_min(&from);
            }
            self.post_trimvol_y_min_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = {
            let field = self.post_trimvol_y_max_from_batchruntomo.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(from) = from {
            self.post_trimvol_y_max.lock().unwrap().set(Some(&from));
            if let Some(display) = display {
                display.set_y_max(&from);
            }
            self.post_trimvol_y_max_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = {
            let field = self.post_trimvol_z_min_from_batchruntomo.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(from) = from {
            self.post_trimvol_z_min.lock().unwrap().set(Some(&from));
            if let Some(display) = display {
                display.set_z_min(&from);
            }
            self.post_trimvol_z_min_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = {
            let field = self.post_trimvol_z_max_from_batchruntomo.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        if let Some(from) = from {
            self.post_trimvol_z_max.lock().unwrap().set(Some(&from));
            if let Some(display) = display {
                display.set_z_max(&from);
            }
            self.post_trimvol_z_max_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_scale_x_min_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_scale_x_min
                .lock()
                .unwrap()
                .set_string(Some(&from.to_string()));
            if let Some(display) = display {
                display.set_scale_x_min(&from.to_string());
            }
            self.post_trimvol_scale_x_min_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_scale_y_min_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_scale_y_min
                .lock()
                .unwrap()
                .set_string(Some(&from.to_string()));
            if let Some(display) = display {
                display.set_scale_y_min(&from.to_string());
            }
            self.post_trimvol_scale_y_min_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_scale_x_max_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_scale_x_max
                .lock()
                .unwrap()
                .set_string(Some(&from.to_string()));
            if let Some(display) = display {
                display.set_scale_x_max(&from.to_string());
            }
            self.post_trimvol_scale_x_max_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        let from = self
            .post_trimvol_scale_y_max_from_batchruntomo
            .lock()
            .unwrap()
            .clone();
        if !from.is_null() {
            self.post_trimvol_scale_y_max
                .lock()
                .unwrap()
                .set_string(Some(&from.to_string()));
            if let Some(display) = display {
                display.set_scale_y_max(&from.to_string());
            }
            self.post_trimvol_scale_y_max_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
    }

    /// Java `isNewBatchruntomoCombineSettings`.
    pub fn is_new_batchruntomo_combine_settings(&self) -> bool {
        self.combine_params
            .lock()
            .unwrap()
            .is_new_batchruntomo_combine_settings()
    }

    /// Java `getMatchMode`.
    pub fn get_match_mode(&self) -> Option<MatchMode> {
        self.combine_params.lock().unwrap().get_match_mode()
    }

    /// Java `isInitialVolumeMatching`.
    pub fn is_initial_volume_matching(&self) -> bool {
        self.combine_params
            .lock()
            .unwrap()
            .is_initial_volume_matching()
    }

    /// Java `getTargetMeasurementRatio`.
    pub fn get_target_measurement_ratio(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.target_measurement_ratio_b.lock().unwrap().to_string();
        }
        self.target_measurement_ratio_a.lock().unwrap().to_string()
    }

    /// Java `isTargetMeasurementRatioSet`.
    pub fn is_target_measurement_ratio_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.target_measurement_ratio_b.lock().unwrap().is_null();
        }
        !self.target_measurement_ratio_a.lock().unwrap().is_null()
    }

    /// Java `getMinMeasurementRatio`.
    pub fn get_min_measurement_ratio(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.min_measurement_ratio_b.lock().unwrap().to_string();
        }
        self.min_measurement_ratio_a.lock().unwrap().to_string()
    }

    /// Java `isMinMeasurementRatioSet`.
    pub fn is_min_measurement_ratio_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.min_measurement_ratio_b.lock().unwrap().is_null();
        }
        !self.min_measurement_ratio_a.lock().unwrap().is_null()
    }

    /// Java `getOrderOfRestrictions`.
    pub fn get_order_of_restrictions(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.order_of_restrictions_b.lock().unwrap().to_string();
        }
        self.order_of_restrictions_a.lock().unwrap().to_string()
    }

    /// Java `getSkipBeamTiltWithOneRot`.
    pub fn get_skip_beam_tilt_with_one_rot(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .skip_beam_tilt_with_one_rot_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.skip_beam_tilt_with_one_rot_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isSkipBeamTiltWithOneRot`.
    pub fn is_skip_beam_tilt_with_one_rot(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.skip_beam_tilt_with_one_rot_b.lock().unwrap().is();
        }
        self.skip_beam_tilt_with_one_rot_a.lock().unwrap().is()
    }

    /// Java `getFineLocalAlignValidation`.
    pub fn get_fine_local_align_validation(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .fine_local_align_validation_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.fine_local_align_validation_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setTargetMeasurementRatio`.
    pub fn set_target_measurement_ratio(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.target_measurement_ratio_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.target_measurement_ratio_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setMinMeasurementRatio`.
    pub fn set_min_measurement_ratio(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.min_measurement_ratio_b
                .lock()
                .unwrap()
                .set_string(input);
        } else {
            self.min_measurement_ratio_a
                .lock()
                .unwrap()
                .set_string(input);
        }
    }

    /// Java `setOrderOfRestrictions`.
    pub fn set_order_of_restrictions(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.order_of_restrictions_b.lock().unwrap().set(input);
        } else {
            self.order_of_restrictions_a.lock().unwrap().set(input);
        }
    }

    /// Java `setSkipBeamTiltWithOneRot`.
    pub fn set_skip_beam_tilt_with_one_rot(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.skip_beam_tilt_with_one_rot_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.skip_beam_tilt_with_one_rot_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setFineLocalAlignValidation`.
    pub fn set_fine_local_align_validation(
        &self,
        axis_id: AxisID,
        input: Option<&ConstEtomoNumber>,
    ) {
        if axis_id == AxisID::Second {
            self.fine_local_align_validation_b
                .lock()
                .unwrap()
                .set_const_etomo_number(input);
        } else {
            self.fine_local_align_validation_a
                .lock()
                .unwrap()
                .set_const_etomo_number(input);
        }
    }

    /// Java `getMinimumOverlap`.
    pub fn get_minimum_overlap(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.minimum_overlap_b.lock().unwrap().to_string();
        }
        self.minimum_overlap_a.lock().unwrap().to_string()
    }

    /// Java `getLengthOfPieces`.
    pub fn get_length_of_pieces(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.length_of_pieces_b.lock().unwrap().to_string();
        }
        self.length_of_pieces_a.lock().unwrap().to_string()
    }

    /// Java `setLengthOfPieces`.
    pub fn set_length_of_pieces(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.length_of_pieces_b.lock().unwrap().set_string(input);
        } else {
            self.length_of_pieces_a.lock().unwrap().set_string(input);
        }
    }

    /// Java `setWeightWholeTracks`.
    pub fn set_weight_whole_tracks(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.weight_whole_tracks_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.weight_whole_tracks_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `getWeightWholeTracks`.
    pub fn get_weight_whole_tracks(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.weight_whole_tracks_b.lock().unwrap().is();
        }
        self.weight_whole_tracks_a.lock().unwrap().is()
    }

    /// Java `getTrackRaptorUseRawStack`.
    pub fn get_track_raptor_use_raw_stack(&self) -> bool {
        self.track_raptor_use_raw_stack_a.lock().unwrap().is()
    }

    /// Java `setTrackRaptorUseRawStack`.
    pub fn set_track_raptor_use_raw_stack(&self, input: bool) {
        self.track_raptor_use_raw_stack_a
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `getTrackRaptorMark`.
    pub fn get_track_raptor_mark(&self) -> String {
        self.track_raptor_mark_a.lock().unwrap().to_string()
    }

    /// Java `setTrackRaptorMark`.
    pub fn set_track_raptor_mark(&self, input: Option<&str>) {
        self.track_raptor_mark_a.lock().unwrap().set_string(input);
    }

    /// Java `getTrackRaptorDiam`.
    pub fn get_track_raptor_diam(&self) -> EtomoNumber {
        self.track_raptor_diam_a.lock().unwrap().clone()
    }

    /// Java `setTrackRaptorDiam`.
    pub fn set_track_raptor_diam(&self, input: Option<&str>) {
        self.track_raptor_diam_a.lock().unwrap().set_string(input);
    }

    /// Java `getEraseGoldModelUseFid`.
    pub fn get_erase_gold_model_use_fid(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.stack_erase_gold_model_use_fid_b.lock().unwrap().is();
        }
        self.stack_erase_gold_model_use_fid_a.lock().unwrap().is()
    }

    /// Java `setEraseGoldModelUseFid`.
    pub fn set_erase_gold_model_use_fid_boolean(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.stack_erase_gold_model_use_fid_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.stack_erase_gold_model_use_fid_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setEraseGoldModelUseFid(AxisID, EraseGold)`.
    pub fn set_erase_gold_model_use_fid_erase_gold(
        &self,
        axis_id: AxisID,
        erase_gold: Option<EraseGold>,
    ) {
        self.set_erase_gold_model_use_fid_boolean(axis_id, erase_gold == Some(EraseGold::Fid));
    }

    /// Java `isPostFlattenWarpInputTrimVol`.
    pub fn is_post_flatten_warp_input_trim_vol(&self) -> bool {
        self.post_flatten_warp_contours_on_one_surface
            .lock()
            .unwrap()
            .is()
    }

    /// Java `isPostTrimvolRotateX`.
    pub fn is_post_trimvol_rotate_x(&self) -> bool {
        self.post_trimvol_rotate_x.lock().unwrap().is()
    }

    /// Java `setPostFlattenWarpInputTrimVol`.
    pub fn set_post_flatten_warp_input_trim_vol(&self, input: bool) {
        self.post_flatten_input_trim_vol
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isPostFlattenWarpContoursOnOneSurface`.
    pub fn is_post_flatten_warp_contours_on_one_surface(&self) -> bool {
        self.post_flatten_warp_contours_on_one_surface
            .lock()
            .unwrap()
            .is()
    }

    /// Java `setPostFlattenWarpContoursOnOneSurface`.
    pub fn set_post_flatten_warp_contours_on_one_surface(&self, input: bool) {
        self.post_flatten_warp_contours_on_one_surface
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `getPostFlattenWarpSpacingInX`.
    pub fn get_post_flatten_warp_spacing_in_x(&self) -> String {
        self.post_flatten_warp_spacing_in_x
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setPostFlattenWarpSpacingInX`.
    pub fn set_post_flatten_warp_spacing_in_x(&self, input: Option<&str>) {
        self.post_flatten_warp_spacing_in_x
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `getPostFlattenWarpSpacingInY`.
    pub fn get_post_flatten_warp_spacing_in_y(&self) -> String {
        self.post_flatten_warp_spacing_in_y
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostTrimvolFixedScaleMax`.
    pub fn get_post_trimvol_fixed_scale_max(&self) -> String {
        self.post_trimvol_fixed_scale_max
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostTrimvolFixedScaleMin`.
    pub fn get_post_trimvol_fixed_scale_min(&self) -> String {
        self.post_trimvol_fixed_scale_min
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostTrimvolScaleXMax`.
    pub fn get_post_trimvol_scale_x_max(&self) -> String {
        self.post_trimvol_scale_x_max.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolScaleXMin`.
    pub fn get_post_trimvol_scale_x_min(&self) -> String {
        self.post_trimvol_scale_x_min.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolScaleYMin`.
    pub fn get_post_trimvol_scale_y_min(&self) -> String {
        self.post_trimvol_scale_y_min.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolScaleYMax`.
    pub fn get_post_trimvol_scale_y_max(&self) -> String {
        self.post_trimvol_scale_y_max.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolSectionScaleMax`.
    pub fn get_post_trimvol_section_scale_max(&self) -> String {
        self.post_trimvol_section_scale_max
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostTrimvolSectionScaleMin`.
    pub fn get_post_trimvol_section_scale_min(&self) -> String {
        self.post_trimvol_section_scale_min
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostTrimvolXMax`.
    pub fn get_post_trimvol_x_max(&self) -> String {
        self.post_trimvol_x_max.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolXMin`.
    pub fn get_post_trimvol_x_min(&self) -> String {
        self.post_trimvol_x_min.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolYMin`.
    pub fn get_post_trimvol_y_min(&self) -> String {
        self.post_trimvol_y_min.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolYMax`.
    pub fn get_post_trimvol_y_max(&self) -> String {
        self.post_trimvol_y_max.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolZMin`.
    pub fn get_post_trimvol_z_min(&self) -> String {
        self.post_trimvol_z_min.lock().unwrap().to_string()
    }

    /// Java `getPostTrimvolZMax`.
    pub fn get_post_trimvol_z_max(&self) -> String {
        self.post_trimvol_z_max.lock().unwrap().to_string()
    }

    /// Java `setPostFlattenWarpSpacingInY`.
    pub fn set_post_flatten_warp_spacing_in_y(&self, input: Option<&str>) {
        self.post_flatten_warp_spacing_in_y
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `isPostSqueezeVolInputTrimVol`.
    pub fn is_post_squeeze_vol_input_trim_vol(&self) -> bool {
        self.post_squeeze_vol_input_trim_vol.lock().unwrap().is()
    }

    /// Java `isPostTrimvolConvertToBytes`.
    pub fn is_post_trimvol_convert_to_bytes(&self) -> bool {
        self.post_trimvol_convert_to_bytes.lock().unwrap().is()
    }

    /// Java `isPostTrimvolFixedScaling`.
    pub fn is_post_trimvol_fixed_scaling(&self) -> bool {
        self.post_trimvol_fixed_scaling.lock().unwrap().is()
    }

    /// Java `isPostTrimvolSwapYZ`.
    pub fn is_post_trimvol_swap_yz(&self) -> bool {
        self.post_trimvol_swap_yz.lock().unwrap().is()
    }

    /// Java `isTrackLengthAndOverlapSet`.
    pub fn is_track_length_and_overlap_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.track_length_and_overlap_b.lock().unwrap().is_empty();
        }
        !self.track_length_and_overlap_a.lock().unwrap().is_empty()
    }

    /// Java `isTrackOverlapOfPatchesXandYSet`.
    pub fn is_track_overlap_of_patches_x_and_y_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self
                .track_overlap_of_patches_x_and_y_b
                .lock()
                .unwrap()
                .is_empty();
        }
        !self
            .track_overlap_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .is_empty()
    }

    /// Java `isPostReduceFiltVolReductionFactor`.
    pub fn is_post_reduce_filt_vol_reduction_factor(&self) -> bool {
        !self
            .reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `getPostReduceFiltVolReductionFactor`.
    pub fn get_post_reduce_filt_vol_reduction_factor(&self) -> String {
        self.reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostReduceFiltVolReductionFactorEtomoNumber`.
    pub fn get_post_reduce_filt_vol_reduction_factor_etomo_number(&self) -> EtomoNumber {
        self.reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .clone()
    }

    /// Java `isPostReduceFiltVolZReductionFactor`.
    pub fn is_post_reduce_filt_vol_z_reduction_factor(&self) -> bool {
        !self
            .reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `getPostReduceFiltVolZReductionFactor`.
    pub fn get_post_reduce_filt_vol_z_reduction_factor(&self) -> String {
        self.reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostReduceFiltVolZReductionFactorEtomoNumber`.
    pub fn get_post_reduce_filt_vol_z_reduction_factor_etomo_number(&self) -> EtomoNumber {
        self.reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .clone()
    }

    /// Java `getPostReduceFiltVolLowPassRadiusSigma`.
    pub fn get_post_reduce_filt_vol_low_pass_radius_sigma(&self) -> String {
        self.reduce_filt_vol_low_pass_radius_sigma
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostReduceFiltVolDeconvolutionStrength`.
    pub fn get_post_reduce_filt_vol_deconvolution_strength(&self) -> String {
        self.reduce_filt_vol_deconvolution_strength
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostReduceFiltVolSNRFalloff`.
    pub fn get_post_reduce_filt_vol_snr_falloff(&self) -> String {
        self.reduce_filt_vol_snr_falloff.lock().unwrap().to_string()
    }

    /// Java `getPostReduceFiltVolHighPassNyquist`.
    pub fn get_post_reduce_filt_vol_high_pass_nyquist(&self) -> String {
        self.reduce_filt_vol_high_pass_nyquist
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostReduceFiltVolDefocusInMicrons`.
    pub fn get_post_reduce_filt_vol_defocus_in_microns(&self) -> String {
        self.reduce_filt_vol_defocus_in_microns
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPostReduceFiltVolPhaseShift`.
    pub fn get_post_reduce_filt_vol_phase_shift(&self) -> String {
        self.reduce_filt_vol_phase_shift.lock().unwrap().to_string()
    }

    /// Java `setPostSqueezeVolInputTrimVol`.
    pub fn set_post_squeeze_vol_input_trim_vol(&self, input: bool) {
        self.post_squeeze_vol_input_trim_vol
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPostReduceFiltVolReductionFactor`.
    pub fn set_post_reduce_filt_vol_reduction_factor(&self, input: Option<&str>) {
        self.reduce_filt_vol_reduction_factor
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostReduceFiltVolZReductionFactor`.
    pub fn set_post_reduce_filt_vol_z_reduction_factor(&self, input: Option<&str>) {
        self.reduce_filt_vol_z_reduction_factor
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostReduceFiltVolLowPassRadiusSigma`.
    pub fn set_post_reduce_filt_vol_low_pass_radius_sigma(&self, input: Option<&str>) {
        self.reduce_filt_vol_low_pass_radius_sigma
            .lock()
            .unwrap()
            .set(input);
    }

    /// Java `setPostReduceFiltVolDeconvolutionStrength`.
    pub fn set_post_reduce_filt_vol_deconvolution_strength(&self, input: Option<&str>) {
        self.reduce_filt_vol_deconvolution_strength
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostReduceFiltVolSNRFalloff`.
    pub fn set_post_reduce_filt_vol_snr_falloff(&self, input: Option<&str>) {
        self.reduce_filt_vol_snr_falloff
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostReduceFiltVolHighPassNyquist`.
    pub fn set_post_reduce_filt_vol_high_pass_nyquist(&self, input: Option<&str>) {
        self.reduce_filt_vol_high_pass_nyquist
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostReduceFiltVolDefocusInMicrons`.
    pub fn set_post_reduce_filt_vol_defocus_in_microns(&self, input: Option<&str>) {
        self.reduce_filt_vol_defocus_in_microns
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostReduceFiltVolPhaseShift`.
    pub fn set_post_reduce_filt_vol_phase_shift(&self, input: Option<&str>) {
        self.reduce_filt_vol_phase_shift
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `setPostTrimvolSwapYZ`.
    pub fn set_post_trimvol_swap_yz(&self, input: bool) {
        self.post_trimvol_swap_yz.lock().unwrap().set_boolean(input);
    }

    /// Java `setPostTrimvolConvertToBytes`.
    pub fn set_post_trimvol_convert_to_bytes(&self, input: bool) {
        self.post_trimvol_convert_to_bytes
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPostTrimvolFixedScaleMax`.
    pub fn set_post_trimvol_fixed_scale_max(&self, input: Option<&str>) {
        self.post_trimvol_fixed_scale_max.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolFixedScaleMin`.
    pub fn set_post_trimvol_fixed_scale_min(&self, input: Option<&str>) {
        self.post_trimvol_fixed_scale_min.lock().unwrap().set(input);
    }

    /// Java `setPostTrimvolFixedScaling`.
    pub fn set_post_trimvol_fixed_scaling(&self, input: bool) {
        self.post_trimvol_fixed_scaling
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setPostTrimvolRotateX`.
    pub fn set_post_trimvol_rotate_x(&self, input: bool) {
        self.post_trimvol_rotate_x
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `getNoBeamTiltSelected`.
    pub fn get_no_beam_tilt_selected(&self, axis_id: AxisID) -> EtomoBoolean2 {
        if axis_id == AxisID::Second {
            return self.no_beam_tilt_selected_b.lock().unwrap().clone();
        }
        self.no_beam_tilt_selected_a.lock().unwrap().clone()
    }

    /// Java `getFixedBeamTiltSelected`.
    pub fn get_fixed_beam_tilt_selected(&self, axis_id: AxisID) -> EtomoBoolean2 {
        if axis_id == AxisID::Second {
            return self.fixed_beam_tilt_selected_b.lock().unwrap().clone();
        }
        self.fixed_beam_tilt_selected_a.lock().unwrap().clone()
    }

    /// Java `getFixedBeamTilt`.
    pub fn get_fixed_beam_tilt(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.fixed_beam_tilt_b.lock().unwrap().clone();
        }
        self.fixed_beam_tilt_a.lock().unwrap().clone()
    }

    /// Java `getFinalStackFiducialDiameter`.
    pub fn get_final_stack_fiducial_diameter(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .final_stack_fiducial_diameter_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.final_stack_fiducial_diameter_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getFinalStackExpandCircleIterations`.
    pub fn get_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self
                .final_stack_expand_circle_iterations_b
                .lock()
                .unwrap()
                .get_int();
        }
        self.final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .get_int()
    }

    /// Java `isFinalStackExpandCircleIterationsSet`.
    pub fn is_final_stack_expand_circle_iterations_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self
                .final_stack_expand_circle_iterations_b
                .lock()
                .unwrap()
                .is_null();
        }
        !self
            .final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `isUseFinalStackExpandCircleIterations`.
    pub fn is_use_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .use_final_stack_expand_circle_iterations_b
                .lock()
                .unwrap()
                .is();
        }
        self.use_final_stack_expand_circle_iterations_a
            .lock()
            .unwrap()
            .is()
    }

    /// Java `getFinalStackBetterRadius`.
    pub fn get_final_stack_better_radius(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.final_stack_better_radius_b.lock().unwrap().to_string();
        }
        self.final_stack_better_radius_a.lock().unwrap().to_string()
    }

    /// Java `getFinalStackPolynomialOrder`.
    pub fn get_final_stack_polynomial_order(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self
                .final_stack_polynomial_order_b
                .lock()
                .unwrap()
                .get_int();
        }
        self.final_stack_polynomial_order_a
            .lock()
            .unwrap()
            .get_int()
    }

    /// Java `isFinalAlignedStackDialogSaved`.
    pub fn is_final_aligned_stack_dialog_saved(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.final_aligned_stack_dialog_saved_b.lock().unwrap().is();
        }
        self.final_aligned_stack_dialog_saved_a.lock().unwrap().is()
    }

    /// Java `getTomoGenTrialTomogramNameList`.
    /// The list itself is returned (a shared handle, as the Java returns its
    /// field): `TrialTiltPanel.addTrialTomogramName` adds to it directly.
    pub fn get_tomo_gen_trial_tomogram_name_list(&self, axis_id: AxisID) -> Arc<Mutex<IntKeyList>> {
        if axis_id == AxisID::Second {
            return Arc::clone(&self.tomo_gen_trial_tomogram_name_list_b.lock().unwrap());
        }
        Arc::clone(&self.tomo_gen_trial_tomogram_name_list_a.lock().unwrap())
    }

    /// Java `getTrackLengthAndOverlap`.
    pub fn get_track_length_and_overlap(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.track_length_and_overlap_b.lock().unwrap().to_string();
        }
        self.track_length_and_overlap_a.lock().unwrap().to_string()
    }

    /// Java `getTrackNumberOfPatchesXandY`.
    pub fn get_track_number_of_patches_x_and_y(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .track_number_of_patches_x_and_y_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.track_number_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getTrackOverlapOfPatchesXandY`.
    pub fn get_track_overlap_of_patches_x_and_y(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .track_overlap_of_patches_x_and_y_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.track_overlap_of_patches_x_and_y_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setTomoGenTrialTomogramNameList`.
    ///
    /// Upstream bug fixed in translation (MetaData.java:5097-5100): the source has no
    /// `else`, so a call for the second axis sets B *and then overwrites A* with the
    /// same list.  The A assignment is now the else branch, as in every other axis
    /// setter in this class.
    pub fn set_tomo_gen_trial_tomogram_name_list(
        &self,
        axis_id: AxisID,
        input: Arc<Mutex<IntKeyList>>,
    ) {
        if axis_id == AxisID::Second {
            *self.tomo_gen_trial_tomogram_name_list_b.lock().unwrap() = input;
        } else {
            *self.tomo_gen_trial_tomogram_name_list_a.lock().unwrap() = input;
        }
    }

    /// Java `setTrackLengthAndOverlap`.
    pub fn set_track_length_and_overlap(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.track_length_and_overlap_b.lock().unwrap().set(input);
        } else {
            self.track_length_and_overlap_a.lock().unwrap().set(input);
        }
    }

    /// Java `setTrackMethod`.
    pub fn set_track_method(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.track_method_b.lock().unwrap().set(input);
        } else {
            self.track_method_a.lock().unwrap().set(input);
        }
    }

    /// Java `setTrackNumberOfPatchesXandY`.
    pub fn set_track_number_of_patches_x_and_y(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.track_number_of_patches_x_and_y_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.track_number_of_patches_x_and_y_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setTrackOverlapOfPatchesXandY`.
    pub fn set_track_overlap_of_patches_x_and_y(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.track_overlap_of_patches_x_and_y_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.track_overlap_of_patches_x_and_y_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `isFinalStackFiducialDiameterNull`.
    pub fn is_final_stack_fiducial_diameter_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .final_stack_fiducial_diameter_b
                .lock()
                .unwrap()
                .is_null();
        }
        self.final_stack_fiducial_diameter_a
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `isFineExists`.
    pub fn is_fine_exists(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.fine_exists_b.lock().unwrap().is();
        }
        self.fine_exists_a.lock().unwrap().is()
    }

    /// Java `getGenLog`.
    pub fn get_gen_log(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_log_b.lock().unwrap().to_string();
        }
        self.gen_log_a.lock().unwrap().to_string()
    }

    /// Java `getGenSubareaSize`.
    pub fn get_gen_subarea_size(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_subarea_size_b.lock().unwrap().to_string();
        }
        self.gen_subarea_size_a.lock().unwrap().to_string()
    }

    /// Java `isGenSubarea`.
    pub fn is_gen_subarea(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_subarea_b.lock().unwrap().is();
        }
        self.gen_subarea_a.lock().unwrap().is()
    }

    /// Java `getRadialRadius`.
    pub fn get_radial_radius(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                return Some(self.gen_radial_radius_b.lock().unwrap().to_string());
            }
            return Some(self.gen_radial_radius_a.lock().unwrap().to_string());
        } else if panel_id == PanelId::Sirtsetup {
            if axis_id == AxisID::Second {
                return Some(self.sirt_radial_radius_b.lock().unwrap().to_string());
            }
            return Some(self.sirt_radial_radius_a.lock().unwrap().to_string());
        }
        None
    }

    /// Java `getRadialSigma`.
    pub fn get_radial_sigma(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                return Some(self.gen_radial_sigma_b.lock().unwrap().to_string());
            }
            return Some(self.gen_radial_sigma_a.lock().unwrap().to_string());
        } else if panel_id == PanelId::Sirtsetup {
            if axis_id == AxisID::Second {
                return Some(self.sirt_radial_sigma_b.lock().unwrap().to_string());
            }
            return Some(self.sirt_radial_sigma_a.lock().unwrap().to_string());
        }
        None
    }

    /// Java `getGenYOffsetOfSubarea`.
    pub fn get_gen_y_offset_of_subarea(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_y_offset_of_subarea_b.lock().unwrap().to_string();
        }
        self.gen_y_offset_of_subarea_a.lock().unwrap().to_string()
    }

    /// Java `getGenScaleFactorLog`.
    pub fn get_gen_scale_factor_log(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_scale_factor_log_b.lock().unwrap().to_string();
        }
        self.gen_scale_factor_log_a.lock().unwrap().to_string()
    }

    /// Java `getGenScaleOffsetLog`.
    pub fn get_gen_scale_offset_log(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_scale_offset_log_b.lock().unwrap().to_string();
        }
        self.gen_scale_offset_log_a.lock().unwrap().to_string()
    }

    /// Java `getGenScaleFactorLinear`.
    pub fn get_gen_scale_factor_linear(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_scale_factor_linear_b.lock().unwrap().to_string();
        }
        self.gen_scale_factor_linear_a.lock().unwrap().to_string()
    }

    /// Java `getGenScaleOffsetLinear`.
    pub fn get_gen_scale_offset_linear(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.gen_scale_offset_linear_b.lock().unwrap().to_string();
        }
        self.gen_scale_offset_linear_a.lock().unwrap().to_string()
    }

    /// Java `getGenSuperSampleFactor`.
    pub fn get_gen_super_sample_factor(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.gen_super_sample_factor_b.lock().unwrap().clone();
        }
        self.gen_super_sample_factor_a.lock().unwrap().clone()
    }

    /// Java `getGenExpandInputLines`.
    pub fn get_gen_expand_input_lines(&self, axis_id: AxisID) -> EtomoBoolean2 {
        if axis_id == AxisID::Second {
            return self.gen_expand_input_lines_b.lock().unwrap().clone();
        }
        self.gen_expand_input_lines_a.lock().unwrap().clone()
    }

    /// Java `isFinalStackBetterRadiusEmpty`.
    pub fn is_final_stack_better_radius_empty(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.final_stack_better_radius_b.lock().unwrap().is_empty();
        }
        self.final_stack_better_radius_a.lock().unwrap().is_empty()
    }

    /// Java `getTargetPatchSizeXandY`.
    pub fn get_target_patch_size_x_and_y(&self) -> String {
        self.target_patch_size_x_and_y.lock().unwrap().clone()
    }

    /// Java `getNumberOfLocalPatchesXandY`.
    pub fn get_number_of_local_patches_x_and_y(&self) -> String {
        self.number_of_local_patches_x_and_y.lock().unwrap().clone()
    }

    /// Java `isFiducialess`.
    pub fn is_fiducialess(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.fiducialess_b.lock().unwrap().is();
        }
        self.fiducialess_a.lock().unwrap().is()
    }

    /// Java `getSqueezevolParam()`: the shared `squeezevolParam` object.
    pub fn get_squeezevol_param(&self) -> MutexGuard<'_, Option<SqueezevolParam>> {
        self.squeezevol_param.lock().unwrap()
    }

    /// Java `getTransferfidAFields(TransferfidParam)`.
    pub fn get_transferfid_a_fields(&self, transferfid_param: &mut TransferfidParam) {
        if let Some(param) = self.transferfid_param_a.lock().unwrap().as_ref() {
            param.get_storable_fields(transferfid_param);
        }
    }

    /// Java `getTransferfidBFields(TransferfidParam)`.
    pub fn get_transferfid_b_fields(&self, transferfid_param: &mut TransferfidParam) {
        if let Some(param) = self.transferfid_param_b.lock().unwrap().as_ref() {
            param.get_storable_fields(transferfid_param);
        }
    }

    /// Java `getTwodir`.
    pub fn get_twodir(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.twodir_b.lock().unwrap().to_string();
        }
        self.twodir_a.lock().unwrap().to_string()
    }

    /// Java `getSeedAndTrackTab`.
    pub fn get_seed_and_track_tab(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self.seed_and_track_tab_b.lock().unwrap().get_int();
        }
        self.seed_and_track_tab_a.lock().unwrap().get_int()
    }

    /// Java `getRaptorTab`.
    pub fn get_raptor_tab(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self.raptor_tab_b.lock().unwrap().get_int();
        }
        self.raptor_tab_a.lock().unwrap().get_int()
    }

    /// Java `isTwodir`.
    pub fn is_twodir(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.is_twodir_b.lock().unwrap().is();
        }
        self.is_twodir_a.lock().unwrap().is()
    }

    /// Java `getDatasetName`.
    pub fn get_dataset_name(&self) -> String {
        self.dataset_name.lock().unwrap().clone()
    }

    // Java `toString` is the `Display` impl below.

    /// Java `getMetaDataFileName`.  The Java `datasetName + fileExtension`
    /// concatenation prints a null extension as "null".
    pub fn get_meta_data_file_name(&self) -> String {
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        if dataset_name == "" {
            return String::new();
        }
        format!(
            "{}{}",
            dataset_name,
            self.base
                .file_extension
                .lock()
                .unwrap()
                .clone()
                .unwrap_or("null".to_string())
        )
    }

    /// Java `getTrackMethod`.
    pub fn get_track_method(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.track_method_b.lock().unwrap().to_string();
        }
        self.track_method_a.lock().unwrap().to_string()
    }

    /// Java `getName`.
    pub fn get_name(&self) -> String {
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        if dataset_name == "" {
            return NEW_TOMOGRAM_TITLE.to_string();
        }
        dataset_name
    }

    /// Java static `getNewFileTitle`.
    pub fn get_new_file_title() -> &'static str {
        NEW_TOMOGRAM_TITLE
    }

    /// Java `getBackupDirectory`.
    pub fn get_backup_directory(&self) -> String {
        self.backup_directory.lock().unwrap().clone()
    }

    /// Java `getDistortionFile`.
    pub fn get_distortion_file(&self) -> String {
        let distortion_file = self.distortion_file.lock().unwrap().clone();
        match distortion_file {
            None => String::new(),
            Some(distortion_file) => distortion_file,
        }
    }

    /// Java `getSizeToOutputInXandY`.
    pub fn get_size_to_output_in_x_and_y(&self, axis_id: AxisID) -> FortranInputString {
        if axis_id == AxisID::Second {
            return self.size_to_output_in_x_and_y_b.lock().unwrap().clone();
        }
        self.size_to_output_in_x_and_y_a.lock().unwrap().clone()
    }

    /// Java `getOrigScopeTemplate`.
    pub fn get_orig_scope_template(&self) -> String {
        self.orig_scope_template.lock().unwrap().to_string()
    }

    /// Java `isOrigScopeTemplate`.
    pub fn is_orig_scope_template(&self) -> bool {
        !self.orig_scope_template.lock().unwrap().is_empty()
    }

    /// Java `getOrigSystemTemplate`.
    pub fn get_orig_system_template(&self) -> String {
        self.orig_system_template.lock().unwrap().to_string()
    }

    /// Java `isOrigSystemTemplate`.
    pub fn is_orig_system_template(&self) -> bool {
        !self.orig_system_template.lock().unwrap().is_empty()
    }

    /// Java `getOrigUserTemplate`.
    pub fn get_orig_user_template(&self) -> String {
        self.orig_user_template.lock().unwrap().to_string()
    }

    /// Java `isOrigUserTemplate`.
    pub fn is_orig_user_template(&self) -> bool {
        !self.orig_user_template.lock().unwrap().is_empty()
    }

    /// Java `getStackCtfAutoFitRangeAndStep`.
    pub fn get_stack_ctf_auto_fit_range_and_step(&self, axis_id: AxisID) -> FortranInputString {
        if axis_id == AxisID::Second {
            return self
                .stack_ctf_auto_fit_range_and_step_b
                .lock()
                .unwrap()
                .clone();
        }
        self.stack_ctf_auto_fit_range_and_step_a
            .lock()
            .unwrap()
            .clone()
    }

    /// Java `isStackCtfAutoFitRangeAndStepSet`.
    pub fn is_stack_ctf_auto_fit_range_and_step_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self
                .stack_ctf_auto_fit_range_and_step_b
                .lock()
                .unwrap()
                .is_null();
        }
        !self
            .stack_ctf_auto_fit_range_and_step_a
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `getMagGradientFile`.
    pub fn get_mag_gradient_file(&self) -> String {
        let mag_gradient_file = self.mag_gradient_file.lock().unwrap().clone();
        match mag_gradient_file {
            None => String::new(),
            Some(mag_gradient_file) if java_lang_string_matches_whitespace(&mag_gradient_file) => {
                String::new()
            }
            Some(mag_gradient_file) => mag_gradient_file,
        }
    }

    /// Java `getAdjustedFocusA`.
    pub fn get_adjusted_focus_a(&self) -> EtomoBoolean2 {
        self.adjusted_focus_a.lock().unwrap().clone()
    }

    /// Java `getAdjustedFocusB`.
    pub fn get_adjusted_focus_b(&self) -> EtomoBoolean2 {
        self.adjusted_focus_b.lock().unwrap().clone()
    }

    /// Java `getAntialiasFilter`.
    pub fn get_antialias_filter(
        &self,
        dialog_type: DialogType,
        axis_id: AxisID,
    ) -> Option<EtomoNumber> {
        if dialog_type == DialogType::CoarseAlignment {
            if axis_id == AxisID::Second {
                return Some(self.coarse_antialias_filter_b.lock().unwrap().clone());
            }
            return Some(self.coarse_antialias_filter_a.lock().unwrap().clone());
        } else if dialog_type == DialogType::FinalAlignedStack {
            if axis_id == AxisID::Second {
                return Some(self.stack_antialias_filter_b.lock().unwrap().clone());
            }
            return Some(self.stack_antialias_filter_a.lock().unwrap().clone());
        }
        None
    }

    /// Java `getDataSource`.
    pub fn get_data_source(&self) -> DataSource {
        *self.data_source.lock().unwrap()
    }

    /// Java `getViewType`.
    pub fn get_view_type(&self) -> ViewType {
        *self.view_type.lock().unwrap()
    }

    /// Java `getPixelSize`.
    ///
    /// Kept native (MetaData.java:5491): the source tests `pixelSize == Double.NaN`,
    /// which is always false, so an unset pixel size is returned as NaN.  Callers
    /// depend on that: `SetupDialogExpert.java:393` fills the field only when the value
    /// is not NaN, so returning 0.0 would show "0.0" in a new dataset's Setup dialog.
    pub fn get_pixel_size(&self) -> f64 {
        *self.pixel_size.lock().unwrap()
    }

    /// Java `getHalfFloatModeOutput`.  Java returns an `Integer`, null when unset.
    pub fn get_half_float_mode_output(&self) -> Option<i32> {
        let field = self.half_float_mode_output.lock().unwrap();
        if !field.is_null() {
            return Some(field.get_int());
        }
        None
    }

    /// Java `isHalfFloatModeOutputSet`.
    pub fn is_half_float_mode_output_set(&self) -> bool {
        !self.half_float_mode_output.lock().unwrap().is_null()
    }

    /// Java `getUseLocalAlignments`.
    pub fn get_use_local_alignments(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return *self.use_local_alignments_b.lock().unwrap();
        }
        *self.use_local_alignments_a.lock().unwrap()
    }

    /// Java `getPosBinning`.
    pub fn get_pos_binning(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self.pos_binning_b.lock().unwrap().get_defaulted_int();
        }
        self.pos_binning_a.lock().unwrap().get_defaulted_int()
    }

    /// Java `getStackBinning`.
    pub fn get_stack_binning(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self.stack_binning_b.lock().unwrap().get_defaulted_int();
        }
        self.stack_binning_a.lock().unwrap().get_defaulted_int()
    }

    /// Java `isStack3dFindBinningSet`.
    pub fn is_stack_3d_find_binning_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.stack_3d_find_binning_b.lock().unwrap().is_null();
        }
        !self.stack_3d_find_binning_a.lock().unwrap().is_null()
    }

    /// Java `getStack3dFindBinning`.
    pub fn get_stack_3d_find_binning(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self.stack_3d_find_binning_b.lock().unwrap().get_int();
        }
        self.stack_3d_find_binning_a.lock().unwrap().get_int()
    }

    /// Java `getPostCurTab`.
    pub fn get_post_cur_tab(&self) -> EtomoNumber {
        self.post_cur_tab.lock().unwrap().clone()
    }

    /// Java `getGenCurTab`.
    pub fn get_gen_cur_tab(&self) -> EtomoNumber {
        self.gen_cur_tab.lock().unwrap().clone()
    }

    /// Java `isPostExists`.
    pub fn is_post_exists(&self) -> bool {
        self.post_exists.lock().unwrap().is()
    }

    /// Java `getCombineVolcombineParallel`.
    pub fn get_combine_volcombine_parallel(&self) -> Option<EtomoBoolean2> {
        self.combine_volcombine_parallel.lock().unwrap().clone()
    }

    /// Java `getTiltParallel`.
    pub fn get_tilt_parallel(&self, axis_id: AxisID, panel_id: PanelId) -> Option<EtomoBoolean2> {
        if panel_id == PanelId::Tilt {
            if axis_id == AxisID::Second {
                return self.tomo_gen_tilt_parallel_b.lock().unwrap().clone();
            }
            return self.tomo_gen_tilt_parallel_a.lock().unwrap().clone();
        } else if panel_id == PanelId::Tilt3dFind {
            if axis_id == AxisID::Second {
                return self.tilt_3d_find_tilt_parallel_b.lock().unwrap().clone();
            }
            return self.tilt_3d_find_tilt_parallel_a.lock().unwrap().clone();
        }
        None
    }

    /// Java `getFinalStackCtfCorrectionParallel`.
    pub fn get_final_stack_ctf_correction_parallel(
        &self,
        axis_id: AxisID,
    ) -> Option<EtomoBoolean2> {
        if axis_id == AxisID::Second {
            return self
                .final_stack_ctf_correction_parallel_b
                .lock()
                .unwrap()
                .clone();
        }
        self.final_stack_ctf_correction_parallel_a
            .lock()
            .unwrap()
            .clone()
    }

    /// Java `isDefaultParallel`.
    pub fn is_default_parallel(&self) -> bool {
        self.default_parallel.lock().unwrap().is()
    }

    /// Java `isDefaultGpuProcessing`.
    pub fn is_default_gpu_processing(&self) -> bool {
        self.default_gpu_processing.lock().unwrap().is()
    }

    /// Java `getUseZFactors`.
    pub fn get_use_z_factors(&self, axis_id: AxisID) -> EtomoBoolean2 {
        if axis_id == AxisID::Second {
            return self.use_z_factors_b.lock().unwrap().clone();
        }
        self.use_z_factors_a.lock().unwrap().clone()
    }

    /// Java `getFiducialDiameter`.
    ///
    /// Kept native (MetaData.java:5612): `fiducialDiameter == Double.NaN` is always
    /// false, so an unset diameter is returned as NaN, which `SetupDialogExpert.java:399`
    /// relies on (see `get_pixel_size`).
    pub fn get_fiducial_diameter(&self) -> f64 {
        *self.fiducial_diameter.lock().unwrap()
    }

    /// Java `isFiducialDiameterAvailable`.  Returns true if fiducial diameter is not
    /// blank or zero.
    ///
    /// Upstream bug fixed in translation (MetaData.java:5623): `fiducialDiameter !=
    /// Double.NaN` is always true, so a blank (NaN) diameter was reported as
    /// available.  The test is now `!is_nan()`, as the documented contract says.
    pub fn is_fiducial_diameter_available(&self) -> bool {
        let fiducial_diameter = *self.fiducial_diameter.lock().unwrap();
        !fiducial_diameter.is_nan() && fiducial_diameter != 0.0
    }

    /// Java `isPositioningNewDialog`.
    pub fn is_positioning_new_dialog(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.positioning_new_dialog_b.lock().unwrap().is();
        }
        self.positioning_new_dialog_a.lock().unwrap().is()
    }

    /// Java `setPositioningNewDialog`.
    pub fn set_positioning_new_dialog(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.positioning_new_dialog_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.positioning_new_dialog_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `getImageRotation`.
    pub fn get_image_rotation(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.image_rotation_b.lock().unwrap().clone();
        }
        self.image_rotation_a.lock().unwrap().clone()
    }

    /// Java `getBinning`.
    pub fn get_binning(&self) -> String {
        self.binning.lock().unwrap().to_string()
    }

    /// Java `getBStackProcessed`.
    pub fn get_b_stack_processed(&self) -> Option<EtomoBoolean2> {
        self.b_stack_processed.lock().unwrap().clone()
    }

    /// Java `getSampleThickness`.
    pub fn get_sample_thickness(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.sample_thickness_b.lock().unwrap().clone();
        }
        self.sample_thickness_a.lock().unwrap().clone()
    }

    /// Java `getTiltAngleSpecA`.
    pub fn get_tilt_angle_spec_a(&self) -> TiltAngleSpec {
        self.tilt_angle_spec_a.lock().unwrap().clone()
    }

    /// Java `getTiltAngleSpec`.
    pub fn get_tilt_angle_spec(&self, axis_id: AxisID) -> TiltAngleSpec {
        if axis_id == AxisID::Second {
            return self.tilt_angle_spec_b.lock().unwrap().clone();
        }
        self.tilt_angle_spec_a.lock().unwrap().clone()
    }

    /// Java `getTiltAngleSpecB`.
    pub fn get_tilt_angle_spec_b(&self) -> TiltAngleSpec {
        self.tilt_angle_spec_b.lock().unwrap().clone()
    }

    /// Java `resetExcludeProjections`.
    pub fn reset_exclude_projections(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.exclude_projections_b.lock().unwrap() = Some(String::new());
        } else {
            *self.exclude_projections_a.lock().unwrap() = Some(String::new());
        }
    }

    /// Java `getExcludeProjectionsA`.
    pub fn get_exclude_projections_a(&self) -> String {
        let exclude_projections_a = self.exclude_projections_a.lock().unwrap().clone();
        match exclude_projections_a {
            None => String::new(),
            Some(exclude_projections_a) => exclude_projections_a,
        }
    }

    /// Java `getExcludeProjectionsB`.
    pub fn get_exclude_projections_b(&self) -> String {
        let exclude_projections_b = self.exclude_projections_b.lock().unwrap().clone();
        match exclude_projections_b {
            None => String::new(),
            Some(exclude_projections_b) => exclude_projections_b,
        }
    }

    /// Java `isGenExists`.
    pub fn is_gen_exists(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_exists_b.lock().unwrap().is();
        }
        self.gen_exists_a.lock().unwrap().is()
    }

    /// Java `isPosExists`.
    pub fn is_pos_exists(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.pos_exists_b.lock().unwrap().is();
        }
        self.pos_exists_a.lock().unwrap().is()
    }

    /// Java `isGenBackProjection`.
    pub fn is_gen_back_projection(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.gen_back_projection_b.lock().unwrap().is();
        }
        self.gen_back_projection_a.lock().unwrap().is()
    }

    /// Java `isGenScaleFactorLinearSet`.
    pub fn is_gen_scale_factor_linear_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.gen_scale_factor_linear_b.lock().unwrap().is_null();
        }
        !self.gen_scale_factor_linear_a.lock().unwrap().is_null()
    }

    /// Java `isGenScaleOffsetLinearSet`.
    pub fn is_gen_scale_offset_linear_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.gen_scale_offset_linear_b.lock().unwrap().is_null();
        }
        !self.gen_scale_offset_linear_a.lock().unwrap().is_null()
    }

    /// Java `getComScriptCreated`.
    pub fn get_com_script_created(&self) -> bool {
        *self.com_scripts_created.lock().unwrap()
    }

    /// Java `isFiducialessAlignment`.
    pub fn is_fiducialess_alignment(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return *self.fiducialess_alignment_b.lock().unwrap();
        }
        *self.fiducialess_alignment_a.lock().unwrap()
    }

    /// Java `isDistortionCorrection`.
    pub fn is_distortion_correction(&self) -> bool {
        let distortion_file = self.distortion_file.lock().unwrap().clone();
        let mag_gradient_file = self.mag_gradient_file.lock().unwrap().clone();
        (distortion_file.is_some()
            && !java_lang_string_matches_whitespace(distortion_file.as_deref().unwrap()))
            || (mag_gradient_file.is_some()
                && !java_lang_string_matches_whitespace(mag_gradient_file.as_deref().unwrap()))
    }

    /// Java `isEraseBeadsInitialized`.
    pub fn is_erase_beads_initialized(&self) -> bool {
        self.erase_beads_initialized.lock().unwrap().is()
    }

    /// Java `isTrackSeedModelManual`.
    pub fn is_track_seed_model_manual(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_seed_model_manual_b.lock().unwrap().is();
        }
        self.track_seed_model_manual_a.lock().unwrap().is()
    }

    /// Java `isTrackSeedModelAuto`.
    pub fn is_track_seed_model_auto(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_seed_model_auto_b.lock().unwrap().is();
        }
        self.track_seed_model_auto_a.lock().unwrap().is()
    }

    /// Java `isTrackSeedModelTransfer`.
    pub fn is_track_seed_model_transfer(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_seed_model_transfer_b.lock().unwrap().is();
        }
        self.track_seed_model_transfer_a.lock().unwrap().is()
    }

    /// Java `isTrackExcludeInsideAreas`.
    pub fn is_track_exclude_inside_areas(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_exclude_inside_areas_b.lock().unwrap().is();
        }
        self.track_exclude_inside_areas_a.lock().unwrap().is()
    }

    /// Java `getTrackJustFindShiftsNearZero`.
    pub fn get_track_just_find_shifts_near_zero(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .track_just_find_shifts_near_zero_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.track_just_find_shifts_near_zero_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getTrackTargetNumberOfBeads`.
    pub fn get_track_target_number_of_beads(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .track_target_number_of_beads_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.track_target_number_of_beads_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getTrackTargetDensityOfBeads`.
    pub fn get_track_target_density_of_beads(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .track_target_density_of_beads_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.track_target_density_of_beads_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isTrackClusteredPointsAllowedElongated`.
    pub fn is_track_clustered_points_allowed_elongated(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .track_clustered_points_allowed_elongated_b
                .lock()
                .unwrap()
                .is();
        }
        self.track_clustered_points_allowed_elongated_a
            .lock()
            .unwrap()
            .is()
    }

    /// Java `getTrackClusteredPointsAllowedElongatedValue`.
    pub fn get_track_clustered_points_allowed_elongated_value(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self
                .track_clustered_points_allowed_elongated_value_b
                .lock()
                .unwrap()
                .get_int();
        }
        self.track_clustered_points_allowed_elongated_value_a
            .lock()
            .unwrap()
            .get_int()
    }

    /// Java `isTrackAdvanced`.
    pub fn is_track_advanced(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_advanced_b.lock().unwrap().is();
        }
        self.track_advanced_a.lock().unwrap().is()
    }

    /// Java `isStack3dFindThicknessSet`.
    pub fn is_stack_3d_find_thickness_set(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return !self.stack_3d_find_thickness_b.lock().unwrap().is_null();
        }
        !self.stack_3d_find_thickness_a.lock().unwrap().is_null()
    }

    /// Java `isSetFEIPixelSize`.
    pub fn is_set_fei_pixel_size(&self) -> bool {
        self.set_fei_pixel_size.lock().unwrap().is()
    }

    /// Java `isPostTrimvolNewStyleZ`.
    pub fn is_post_trimvol_new_style_z(&self) -> bool {
        self.post_trimvol_new_style_z.lock().unwrap().is()
    }

    /// Java `isPostTrimvolScalingNewStyleZ`.
    pub fn is_post_trimvol_scaling_new_style_z(&self) -> bool {
        self.post_trimvol_scaling_new_style_z.lock().unwrap().is()
    }

    /// Java `getStack3dFindThickness`.
    pub fn get_stack_3d_find_thickness(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.stack_3d_find_thickness_b.lock().unwrap().to_string();
        }
        self.stack_3d_find_thickness_a.lock().unwrap().to_string()
    }

    /// Java `isWholeTomogramSample`.
    pub fn is_whole_tomogram_sample(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return *self.whole_tomogram_sample_b.lock().unwrap();
        }
        *self.whole_tomogram_sample_a.lock().unwrap()
    }

    /// Java `getConstCombineParams`.  Returns the `combineParams` field, behind its
    /// lock.
    pub fn get_const_combine_params(&self) -> MutexGuard<'_, CombineParams> {
        self.combine_params.lock().unwrap()
    }

    /// Java `getCombineParams`.  Returns the `combineParams` field, behind its lock.
    pub fn get_combine_params(&self) -> MutexGuard<'_, CombineParams> {
        self.combine_params.lock().unwrap()
    }

    // Java `isValid()` implements the abstract `BaseMetaData.isValid`; it is in the
    // `BaseMetaData` impl below.

    /// Java `isValid(boolean)`.
    pub fn is_valid_from_screen(&self, from_screen: bool) -> bool {
        self.is_valid_from_screen_param_file(from_screen, None)
    }

    /// Java `isValid(File)`.  The `File` is its path.
    pub fn is_valid_param_file(&self, param_file: Option<&str>) -> bool {
        self.is_valid_from_screen_param_file(false, param_file)
    }

    /// Java `isOrigViewsWithMagChanges`.
    pub fn is_orig_views_with_mag_changes(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.orig_views_with_mag_changes_b.lock().unwrap().is();
        }
        self.orig_views_with_mag_changes_a.lock().unwrap().is()
    }

    /// Java `isValid(boolean, File)`.
    ///
    /// The source's two NaN tests are both effectively plain comparisons:
    /// `pixelSize != Double.NaN && pixelSize <= 0.0` is `pixelSize <= 0.0` (which NaN
    /// fails anyway), and `fiducialDiameter == Double.NaN || fiducialDiameter < 0.0` is
    /// `fiducialDiameter < 0.0`.  Both are kept as written: a blank (NaN) fiducial
    /// diameter passes validation, and making it fail would be a guess about intent.
    pub fn is_valid_from_screen_param_file(
        &self,
        from_screen: bool,
        param_file: Option<&str>,
    ) -> bool {
        *self.base.invalid_reason.lock().unwrap() = String::new();

        let help_string;
        if !from_screen {
            help_string = "  Check the Etomo data file.";
        } else {
            help_string = "";
        }

        let axis_type = *self.base.axis_type.lock().unwrap();
        if axis_type == AxisType::NotSet {
            *self.base.invalid_reason.lock().unwrap() = format!(
                "Axis type should be either Dual Axis or Single Axis.{}",
                help_string
            );
            return false;
        }

        if !self.is_dataset_name_valid_param_file(param_file) {
            self.base
                .invalid_reason
                .lock()
                .unwrap()
                .push_str(help_string);
            return false;
        }

        // Is the pixel size greater than zero
        let pixel_size = *self.pixel_size.lock().unwrap();
        if from_screen && !pixel_size.is_nan() && pixel_size <= 0.0 {
            *self.base.invalid_reason.lock().unwrap() =
                "Pixel size is not greater than zero.".to_string();
            return false;
        }

        // Is the fiducial diameter greater than zero
        let fiducial_diameter = *self.fiducial_diameter.lock().unwrap();
        if from_screen && (false || fiducial_diameter < 0.0) {
            *self.base.invalid_reason.lock().unwrap() =
                "Fiducial diameter cannot be negative.".to_string();
            return false;
        }

        true
    }

    /// Java `isDatasetNameValid()`.
    pub fn is_dataset_name_valid(&self) -> bool {
        self.is_dataset_name_valid_param_file(None)
    }

    /// Java `isAntialiasFilterNull`.
    pub fn is_antialias_filter_null(&self, dialog_type: DialogType, axis_id: AxisID) -> bool {
        if dialog_type == DialogType::CoarseAlignment {
            if axis_id == AxisID::Second {
                return self.coarse_antialias_filter_b.lock().unwrap().is_null();
            }
            return self.coarse_antialias_filter_a.lock().unwrap().is_null();
        } else if dialog_type == DialogType::FinalAlignedStack {
            if axis_id == AxisID::Second {
                return self.stack_antialias_filter_b.lock().unwrap().is_null();
            }
            return self.stack_antialias_filter_a.lock().unwrap().is_null();
        }
        true
    }

    /// Java `isDatasetNameValid(File)`.  The `File` is its path.
    pub fn is_dataset_name_valid_param_file(&self, param_file: Option<&str>) -> bool {
        *self.base.invalid_reason.lock().unwrap() = String::new();
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        if dataset_name == "" {
            *self.base.invalid_reason.lock().unwrap() =
                "Dataset name has not been set.".to_string();
            return false;
        }
        match param_file {
            None => {
                // Java `manager.getPropertyUserDir()`; a Java null working directory
                // string makes `new File(null)` throw, which is not reachable from a
                // running manager.
                let property_user_dir = self
                    .manager
                    .and_then(|manager| manager.get_property_user_dir())
                    .unwrap_or_default();
                if self
                    .get_valid_dataset_directory(&property_user_dir)
                    .is_some()
                {
                    return true;
                }
            }
            Some(param_file) => {
                let parent = java_io_file_get_parent(param_file).unwrap_or_default();
                if self
                    .get_valid_dataset_directory(&java_io_file_get_absolute_path(&parent))
                    .is_some()
                {
                    return true;
                }
            }
        }
        false
    }

    /// Java `getValidDatasetDirectory`.  Returns the directory (as a path) holding the
    /// raw stack, or null with `invalidReason` set.
    ///
    /// Upstream bugs fixed in translation (MetaData.java:6041 and :6045): the source
    /// throws `IllegalStateException` when a directory exists and is readable and
    /// writable but still failed the validity test, and when the raw stack extension is
    /// empty.  Neither is caught by any caller, so a dataset file without a
    /// `RawImageStackExt` crashed validation.  Both now set `invalidReason` to the
    /// exception's message and return null, like the method's other failures.
    pub fn get_valid_dataset_directory(&self, working_dir_name: &str) -> Option<String> {
        // Does the working directory exist
        // If is doesn't then use the backup directory.
        let working_dir = working_dir_name.to_string();
        let backup_dir = self.backup_directory.lock().unwrap().clone();
        let mut current_dir;

        // find a valid directory and set directory and type
        if MetaData::is_valid_file(Some(&working_dir), true) {
            current_dir = working_dir.clone();
        } else if MetaData::is_valid_file(Some(&backup_dir), true) {
            current_dir = backup_dir.clone();
        } else {
            // can't find a valid directory, report error

            // if no directory exists then exit
            if !std::path::Path::new(&working_dir).exists()
                && !std::path::Path::new(&backup_dir).exists()
            {
                *self.base.invalid_reason.lock().unwrap() = format!(
                    "The working directory: {} and the backup directory: {} do not exist",
                    java_io_file_get_absolute_path(&working_dir),
                    java_io_file_get_absolute_path(&backup_dir)
                );
                return None;
            }

            // decide which directory to complain about:
            // complain about the working directory, if it exists
            if std::path::Path::new(&working_dir).exists() {
                current_dir = working_dir.clone();
            } else {
                current_dir = backup_dir.clone();
            }

            if !java_io_file_can_read(&current_dir) {
                *self.base.invalid_reason.lock().unwrap() = format!(
                    "Can't read {} directory",
                    java_io_file_get_absolute_path(&current_dir)
                );
                return None;
            }

            if !java_io_file_can_write(&current_dir) {
                *self.base.invalid_reason.lock().unwrap() = format!(
                    "Can't write {} directory",
                    java_io_file_get_absolute_path(&current_dir)
                );

                return None;
            }
            *self.base.invalid_reason.lock().unwrap() = format!(
                "Working directory ={},backupDir={}",
                working_dir, backup_dir
            );
            return None;
        }
        let raw_image_stack_ext = {
            let field = self.raw_image_stack_ext.lock().unwrap();
            if field.is_empty() {
                None
            } else {
                Some(field.to_string())
            }
        };
        let raw_image_stack_ext = match raw_image_stack_ext {
            None => {
                *self.base.invalid_reason.lock().unwrap() =
                    "Raw stack extension not found or invalid".to_string();
                return None;
            }
            Some(raw_image_stack_ext) => raw_image_stack_ext,
        };
        // Does the appropriate image stack exist in the working directory
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        let axis_type = *self.base.axis_type.lock().unwrap();
        let found;
        if axis_type == AxisType::DualAxis {
            found = self.find_valid_file_with_alt_dir(
                &format!(
                    "{}a{}{}",
                    dataset_name, EXTENSION_DIVIDER, raw_image_stack_ext
                ),
                &current_dir,
                &backup_dir,
            );
        } else {
            found = self.find_valid_file_with_alt_dir(
                &format!(
                    "{}{}{}",
                    dataset_name, EXTENSION_DIVIDER, raw_image_stack_ext
                ),
                &current_dir,
                &backup_dir,
            );
        }
        current_dir = match found {
            None => return None,
            Some(found) => found,
        };

        Some(current_dir)
    }

    /// Java package-private static `isValid(File, boolean)`.  Checks whether a file
    /// exists and is readable.  Optionally checks whether a file is writable.
    pub fn is_valid_file(file: Option<&str>, writeable: bool) -> bool {
        let file = match file {
            None => return false,
            Some(file) => file,
        };
        if !std::path::Path::new(file).exists() {
            return false;
        }

        java_io_file_can_read(file) && (!writeable || java_io_file_can_write(file))
    }

    /// Java private `appendMessage`.
    fn append_message(&self, string: &str) {
        self.message.lock().unwrap().push_str(string);
    }

    /// Java package-private `findValidFile(String, File, File)`.  Finds a file in
    /// either the current directory or an alternate directory.  The file must be
    /// readable.  If the function returns null, it places an error message into
    /// invalidReason.
    ///
    /// The source throws `IllegalArgumentException` for a null or invalid argument;
    /// that precondition is a programming error in the caller (the one caller,
    /// `getValidDatasetDirectory`, passes a directory it has just validated) and stays a
    /// panic.  `curDir == altDir` is a Java reference comparison between two `File`
    /// objects; it is true only after the loop has switched to the alternate
    /// directory, which is what the `switched` flag records.
    pub fn find_valid_file_with_alt_dir(
        &self,
        file_name: &str,
        cur_dir: &str,
        alt_dir: &str,
    ) -> Option<String> {
        if !MetaData::is_valid_file(Some(cur_dir), true) {
            panic!("ConstMetaData.findValidFile(String,File,File)");
        }

        // Does the appropriate image stack exist in the working or backup directory
        let mut cur_dir = cur_dir.to_string();
        let mut switched = false;
        let mut file = java_io_file_new(&cur_dir, file_name);
        while !std::path::Path::new(&file).exists() {
            if switched || !MetaData::is_valid_file(Some(alt_dir), true) {
                let mut message = self.message.lock().unwrap();
                message.push_str(&format!(
                    "{} does not exist in  {}",
                    file_name,
                    java_io_file_get_absolute_path(&cur_dir)
                ));
                *self.base.invalid_reason.lock().unwrap() = message.clone();
                *message = String::new();
                return None;
            }
            cur_dir = alt_dir.to_string();
            switched = true;
            file = java_io_file_new(&cur_dir, file_name);
        }

        if !java_io_file_can_read(&file) {
            *self.base.invalid_reason.lock().unwrap() = format!("Can't read {}", file_name);
            return None;
        }

        Some(cur_dir)
    }

    /// Java package-private `findValidFile(String, File)`.  Finds a file in the current
    /// directory.  The file must be readable.  See `find_valid_file_with_alt_dir` for
    /// the precondition panic.
    pub fn find_valid_file(&self, file_name: &str, cur_dir: &str) -> Option<String> {
        if !MetaData::is_valid_file(Some(cur_dir), true) {
            panic!("ConstMetaData.findValidFile(String,File)");
        }

        // Does the appropriate image stack exist in the working or backup directory
        let file = java_io_file_new(cur_dir, file_name);

        if !std::path::Path::new(&file).exists() {
            *self.base.invalid_reason.lock().unwrap() = format!(
                "{} does not exist in {}",
                file_name,
                java_io_file_get_absolute_path(cur_dir)
            );
            return None;
        }

        if !java_io_file_can_read(&file) {
            *self.base.invalid_reason.lock().unwrap() = format!("Can't read {}", file_name);
            return None;
        }

        Some(cur_dir.to_string())
    }

    /// Java `equals(Object)`.  The `instanceof MetaData` test is the parameter type.
    ///
    /// Upstream bug fixed in translation (MetaData.java:6195-6200): `imageRotationA`,
    /// `imageRotationB` and `binning` are compared with `==`, which for two
    /// `EtomoNumber` objects is reference identity - always false for two different
    /// `MetaData` instances, so `equals` could never be true.  They are now compared by
    /// value (`ConstEtomoNumber.equals`), as every other number field here is.  The
    /// double fields keep Java's `==` (so a NaN pixel size or fiducial diameter is never
    /// equal), which is plausibly intended.
    ///
    /// Each of `cmd`'s fields is read into a local before this instance's lock is
    /// taken, so `md.equals(&md)` cannot deadlock on a field mutex.
    pub fn equals(&self, cmd: &MetaData) -> bool {
        let other = cmd.dataset_name.lock().unwrap().clone();
        if *self.dataset_name.lock().unwrap() != other {
            return false;
        }
        let other = cmd.backup_directory.lock().unwrap().clone();
        if *self.backup_directory.lock().unwrap() != other {
            return false;
        }
        let other = cmd.distortion_file.lock().unwrap().clone();
        if *self.distortion_file.lock().unwrap() != other {
            return false;
        }
        let other = cmd.mag_gradient_file.lock().unwrap().clone();
        if *self.mag_gradient_file.lock().unwrap() != other {
            return false;
        }
        let other = *cmd.data_source.lock().unwrap();
        if *self.data_source.lock().unwrap() != other {
            return false;
        }
        let other = *cmd.base.axis_type.lock().unwrap();
        if *self.base.axis_type.lock().unwrap() != other {
            return false;
        }
        let other = *cmd.view_type.lock().unwrap();
        if *self.view_type.lock().unwrap() != other {
            return false;
        }
        let other = *cmd.pixel_size.lock().unwrap();
        if !(*self.pixel_size.lock().unwrap() == other) {
            return false;
        }
        let other = cmd.half_float_mode_output.lock().unwrap().clone();
        if !(self
            .half_float_mode_output
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base)))
        {
            return false;
        }
        let other = *cmd.use_local_alignments_a.lock().unwrap();
        if !(*self.use_local_alignments_a.lock().unwrap() == other) {
            return false;
        }
        let other = *cmd.use_local_alignments_b.lock().unwrap();
        if !(*self.use_local_alignments_b.lock().unwrap() == other) {
            return false;
        }
        let other = *cmd.fiducial_diameter.lock().unwrap();
        if !(*self.fiducial_diameter.lock().unwrap() == other) {
            return false;
        }
        let other = cmd.image_rotation_a.lock().unwrap().clone();
        if !self
            .image_rotation_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.image_rotation_b.lock().unwrap().clone();
        if !self
            .image_rotation_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.binning.lock().unwrap().clone();
        if !self
            .binning
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = *cmd.fiducialess_alignment_a.lock().unwrap();
        if !(*self.fiducialess_alignment_a.lock().unwrap() == other) {
            return false;
        }
        let other = *cmd.fiducialess_alignment_b.lock().unwrap();
        if !(*self.fiducialess_alignment_b.lock().unwrap() == other) {
            return false;
        }

        // TODO tilt angle spec needs to be more complete
        let other = cmd.get_tilt_angle_spec_a().get_type();
        if !(self.tilt_angle_spec_a.lock().unwrap().get_type() == other) {
            return false;
        }
        let other = cmd.exclude_projections_a.lock().unwrap().clone();
        let other_getter = cmd.get_exclude_projections_a();
        {
            let field = self.exclude_projections_a.lock().unwrap();
            if (field.is_none() && other.is_some())
                || (field.is_some() && *field.as_deref().unwrap() != other_getter)
            {
                return false;
            }
        }

        let other = cmd.get_tilt_angle_spec_b().get_type();
        if !(self.tilt_angle_spec_b.lock().unwrap().get_type() == other) {
            return false;
        }
        let other = cmd.exclude_projections_b.lock().unwrap().clone();
        let other_getter = cmd.get_exclude_projections_b();
        {
            let field = self.exclude_projections_b.lock().unwrap();
            if (field.is_none() && other.is_some())
                || (field.is_some() && *field.as_deref().unwrap() != other_getter)
            {
                return false;
            }
        }
        let other = cmd.get_com_script_created();
        if !(*self.com_scripts_created.lock().unwrap() == other) {
            return false;
        }
        // `if (!combineParams.equals(cmd.getConstCombineParams())) return false;`:
        // `CombineParams` does not override `Object.equals`, so this is identity, true
        // only when `cmd` is this object (BUGS.md, kept native).
        if !std::ptr::eq(&self.combine_params, &cmd.combine_params) {
            return false;
        }
        {
            let mine = self.squeezevol_param.lock().unwrap();
            let theirs = cmd.squeezevol_param.lock().unwrap();
            if let (Some(mine), Some(theirs)) = (mine.as_ref(), theirs.as_ref())
                && !mine.equals(theirs)
            {
                return false;
            }
        }
        let other = cmd.pos_binning_a.lock().unwrap().clone();
        if !self
            .pos_binning_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.pos_binning_b.lock().unwrap().clone();
        if !self
            .pos_binning_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.stack_binning_a.lock().unwrap().clone();
        if !self
            .stack_binning_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.stack_binning_b.lock().unwrap().clone();
        if !self
            .stack_binning_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.stack_3d_find_binning_a.lock().unwrap().clone();
        if !self
            .stack_3d_find_binning_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.stack_3d_find_binning_b.lock().unwrap().clone();
        if !self
            .stack_3d_find_binning_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.stack_erase_gold_model_use_fid_a.lock().unwrap().clone();
        if !self
            .stack_erase_gold_model_use_fid_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base.base.base))
        {
            return false;
        }
        let other = cmd.stack_erase_gold_model_use_fid_b.lock().unwrap().clone();
        if !self
            .stack_erase_gold_model_use_fid_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base.base.base))
        {
            return false;
        }
        let other = cmd.twodir_a.lock().unwrap().clone();
        if !self
            .twodir_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.twodir_b.lock().unwrap().clone();
        if !self
            .twodir_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.dose_sym_a.lock().unwrap().clone();
        if !self
            .dose_sym_a
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        let other = cmd.dose_sym_b.lock().unwrap().clone();
        if !self
            .dose_sym_b
            .lock()
            .unwrap()
            .equals_const_etomo_number(Some(&other.base))
        {
            return false;
        }
        true
    }

    /// Java `isSubtomoReorientationTypeNone`.
    pub fn is_subtomo_reorientation_type_none(&self) -> bool {
        self.subtomo_reorientation_type_none.lock().unwrap().is()
    }

    /// Java `setSubtomoReorientationTypeNone`.
    pub fn set_subtomo_reorientation_type_none(&self, input: bool) {
        self.subtomo_reorientation_type_none
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isSubtomoReorientationTypeFlipped`.
    pub fn is_subtomo_reorientation_type_flipped(&self) -> bool {
        self.subtomo_reorientation_type_flipped.lock().unwrap().is()
    }

    /// Java `setSubtomoReorientationTypeFlipped`.
    pub fn set_subtomo_reorientation_type_flipped(&self, input: bool) {
        self.subtomo_reorientation_type_flipped
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isSubtomoReorientationTypeRotated`.
    pub fn is_subtomo_reorientation_type_rotated(&self) -> bool {
        self.subtomo_reorientation_type_rotated.lock().unwrap().is()
    }

    /// Java `setSubtomoReorientationTypeRotated`.
    pub fn set_subtomo_reorientation_type_rotated(&self, input: bool) {
        self.subtomo_reorientation_type_rotated
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isSubtomoMakeVolumeStacks`.
    pub fn is_subtomo_make_volume_stacks(&self) -> bool {
        self.subtomo_make_volume_stacks.lock().unwrap().is()
    }

    /// Java `getSubtomoMakeVolumeStacks`.
    pub fn get_subtomo_make_volume_stacks(&self) -> String {
        self.subtomo_make_volume_stacks.lock().unwrap().to_string()
    }

    /// Java `setSubtomoMakeVolumeStacks`.
    pub fn set_subtomo_make_volume_stacks(&self, input: Option<Number>) {
        self.subtomo_make_volume_stacks
            .lock()
            .unwrap()
            .set_number(input);
    }

    /// Java `getSubtomoExtentOfZLevelsInNm`.
    pub fn get_subtomo_extent_of_z_levels_in_nm(&self) -> String {
        self.subtomo_extent_of_z_levels_in_nm
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setSubtomoExtentOfZLevelsInNm`.
    pub fn set_subtomo_extent_of_z_levels_in_nm(&self, input: Option<&str>) {
        self.subtomo_extent_of_z_levels_in_nm
            .lock()
            .unwrap()
            .set_string(input);
    }

    /// Java `getSubtomoNewAlignedBinning`.
    pub fn get_subtomo_new_aligned_binning(&self) -> String {
        self.subtomo_new_aligned_binning.lock().unwrap().to_string()
    }

    /// Java `setSubtomoNewAlignedBinning`.
    pub fn set_subtomo_new_aligned_binning(&self, input: Option<Number>) {
        self.subtomo_new_aligned_binning
            .lock()
            .unwrap()
            .set_number(input);
    }

    /// Java `getSubtomoFourierReduceByFactor`.
    pub fn get_subtomo_fourier_reduce_by_factor(&self) -> String {
        self.subtomo_fourier_reduce_by_factor
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setSubtomoFourierReduceByFactor`.
    pub fn set_subtomo_fourier_reduce_by_factor(&self, input: Option<Number>) {
        self.subtomo_fourier_reduce_by_factor
            .lock()
            .unwrap()
            .set_number(input);
    }

    /// Java `isDoseSym`.
    pub fn is_dose_sym(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.is_dose_sym_b.lock().unwrap().is();
        }
        self.is_dose_sym_a.lock().unwrap().is()
    }

    /// Java `getDoseSym`.
    pub fn get_dose_sym(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.dose_sym_b.lock().unwrap().to_string();
        }
        self.dose_sym_a.lock().unwrap().to_string()
    }

    /// Java `setAltTomoRootname`.
    pub fn set_alt_tomo_rootname(&self, input: Option<&str>) {
        self.alt_tomo_rootname_to_process.lock().unwrap().set(input);
    }

    /// Java `isAltTomoRootname`.
    pub fn is_alt_tomo_rootname(&self) -> bool {
        !self.alt_tomo_rootname_to_process.lock().unwrap().is_empty()
    }

    /// Java `getAltTomoRootname`.
    pub fn get_alt_tomo_rootname(&self) -> String {
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `setAltTomoTrimVolume`.
    pub fn set_alt_tomo_trim_volume(&self, input: bool) {
        self.alt_tomo_trim_volume.lock().unwrap().set_boolean(input);
    }

    /// Java `isAltTomoTrimVolume`.
    pub fn is_alt_tomo_trim_volume(&self) -> bool {
        self.alt_tomo_trim_volume.lock().unwrap().is()
    }

    /// Java `setAltTomoArchiveOrigStack`.
    pub fn set_alt_tomo_archive_orig_stack(&self, input: bool) {
        self.alt_tomo_archive_orig_stack
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isAltTomoArchiveOrigStack`.
    pub fn is_alt_tomo_archive_orig_stack(&self) -> bool {
        self.alt_tomo_archive_orig_stack.lock().unwrap().is()
    }

    /// Java `getLowPassRadiusSigma`.
    pub fn get_low_pass_radius_sigma(&self) -> String {
        self.reduce_filt_vol_low_pass_radius_sigma
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getDeconvolutionStrength`.
    pub fn get_deconvolution_strength(&self) -> String {
        self.reduce_filt_vol_deconvolution_strength
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getSNRFalloff`.
    pub fn get_snr_falloff(&self) -> String {
        self.reduce_filt_vol_snr_falloff.lock().unwrap().to_string()
    }

    /// Java `getHighPassNyquist`.
    pub fn get_high_pass_nyquist(&self) -> String {
        self.reduce_filt_vol_high_pass_nyquist
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getDefocusInMicrons`.
    pub fn get_defocus_in_microns(&self) -> String {
        self.reduce_filt_vol_defocus_in_microns
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getPhaseShift`.
    pub fn get_phase_shift(&self) -> String {
        self.reduce_filt_vol_phase_shift.lock().unwrap().to_string()
    }
}

impl BaseMetaData for MetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }

    /// Java `getOrigRawImageStackExtension` (override).
    fn get_orig_raw_image_stack_extension(&self) -> Option<&'static Extension> {
        MetaData::get_orig_raw_image_stack_extension(self)
    }

    /// Java `getRawImageStackExtension` (override).
    fn get_raw_image_stack_extension(&self) -> Option<&'static Extension> {
        MetaData::get_raw_image_stack_extension(self)
    }

    /// Java `getMetaDataFileName`.
    fn get_meta_data_file_name(&self) -> Option<String> {
        Some(MetaData::get_meta_data_file_name(self))
    }

    /// Java `getName`.
    fn get_name(&self) -> Option<String> {
        Some(MetaData::get_name(self))
    }

    /// Java `getDatasetName`.
    fn get_dataset_name(&self) -> Option<String> {
        Some(MetaData::get_dataset_name(self))
    }

    /// Java `isValid()`.
    fn is_valid(&self) -> bool {
        self.is_valid_from_screen_param_file(true, None)
    }

    /// Java package-private `getGroupKey`.
    fn get_group_key(&self) -> Option<String> {
        Some(MetaData::get_group_key(self))
    }

    /// Java package-private `repairImageFilenameStyle`, overriding `BaseMetaData`.
    /// Uses the dataset or the environment (as a fallback) to repair
    /// imageFilenameStyle [Bug# 2403].
    fn repair_image_filename_style(&self, parent_prepend: &str) {
        if self.base.was_image_filename_style_loaded() {
            return;
        }
        let repair_type;
        // Get the naming style from one of the datasets.
        let mut image_filename_style: Option<ImageFilenameStyle> =
            self.base.get_dataset_image_filename_style();
        if image_filename_style.is_some() {
            repair_type = "image file name style from this project's dataset";
        } else {
            // Fallback: use the environment's naming style
            image_filename_style =
                Some(ImageFileMetaData::get_temp_instance().get_image_filename_style());
            repair_type = "image file name style from the environment";
        }
        let key = self
            .get_image_filename_style_key(parent_prepend)
            .unwrap_or("null".to_string());
        if let Some(image_filename_style) = image_filename_style {
            eprintln!(
                "\nINFO: Attempting to repair the {} property, which is missing from the\ndataset file.  Using the {} and\nsetting {} to {}.  [Bug# 2403]\n",
                key, repair_type, key, image_filename_style
            );
            if self.correct_image_filename_style(parent_prepend, Some(image_filename_style)) {
                return;
            }
        }
        // The source's `repairType != null ? ... : null` is always the first arm here.
        eprintln!(
            "\nERROR: Unable to repair the {} property, which is missing from the\ndataset file.  Unable to use the {}.\n[Bug# 2403]\n",
            key, repair_type
        );
    }
}

/// Java `Storable`, implemented through `BaseMetaData`.  `store(Properties)` is
/// `BaseMetaData`'s, which stores with an empty prepend.
impl storable::Storable for MetaData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        MetaData::store_with_prepend(self, properties, "");
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        MetaData::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        MetaData::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        MetaData::load_with_prepend(self, properties, prepend);
    }
}

impl ConstMetaData for MetaData {
    fn is_ctf_3d_setup_slab_thickness_in_nm_set(&self) -> bool {
        MetaData::is_ctf_3d_setup_slab_thickness_in_nm_set(self)
    }

    fn get_raw_image_stack_extension(&self) -> Option<&'static Extension> {
        MetaData::get_raw_image_stack_extension(self)
    }

    fn get_post_cur_tab(&self) -> EtomoNumber {
        MetaData::get_post_cur_tab(self)
    }

    fn get_gen_cur_tab(&self) -> EtomoNumber {
        MetaData::get_gen_cur_tab(self)
    }

    fn get_dataset_name(&self) -> String {
        MetaData::get_dataset_name(self)
    }

    fn get_com_script_created(&self) -> bool {
        MetaData::get_com_script_created(self)
    }

    fn get_adjusted_focus_a(&self) -> EtomoBoolean2 {
        MetaData::get_adjusted_focus_a(self)
    }

    fn get_adjusted_focus_b(&self) -> EtomoBoolean2 {
        MetaData::get_adjusted_focus_b(self)
    }

    fn get_image_rotation(&self, axis_id: AxisID) -> EtomoNumber {
        MetaData::get_image_rotation(self, axis_id)
    }

    fn get_backup_directory(&self) -> String {
        MetaData::get_backup_directory(self)
    }

    fn get_binning(&self) -> String {
        MetaData::get_binning(self)
    }

    fn get_view_type(&self) -> ViewType {
        MetaData::get_view_type(self)
    }

    fn get_pixel_size(&self) -> f64 {
        MetaData::get_pixel_size(self)
    }

    fn is_half_float_mode_output_set(&self) -> bool {
        MetaData::is_half_float_mode_output_set(self)
    }

    fn get_half_float_mode_output(&self) -> Option<i32> {
        MetaData::get_half_float_mode_output(self)
    }

    fn get_data_source(&self) -> DataSource {
        MetaData::get_data_source(self)
    }

    fn get_fiducial_diameter(&self) -> f64 {
        MetaData::get_fiducial_diameter(self)
    }

    fn get_tilt_angle_spec_a(&self) -> TiltAngleSpec {
        MetaData::get_tilt_angle_spec_a(self)
    }

    fn get_exclude_projections_a(&self) -> String {
        MetaData::get_exclude_projections_a(self)
    }

    fn get_tilt_angle_spec_b(&self) -> TiltAngleSpec {
        MetaData::get_tilt_angle_spec_b(self)
    }

    fn get_distortion_file(&self) -> String {
        MetaData::get_distortion_file(self)
    }

    fn get_mag_gradient_file(&self) -> String {
        MetaData::get_mag_gradient_file(self)
    }

    fn get_exclude_projections_b(&self) -> String {
        MetaData::get_exclude_projections_b(self)
    }

    fn get_combine_volcombine_parallel(&self) -> Option<EtomoBoolean2> {
        MetaData::get_combine_volcombine_parallel(self)
    }

    fn is_default_parallel(&self) -> bool {
        MetaData::is_default_parallel(self)
    }

    fn is_default_gpu_processing(&self) -> bool {
        MetaData::is_default_gpu_processing(self)
    }

    fn get_first_axis_prepend(&self) -> Option<String> {
        MetaData::get_first_axis_prepend(self)
    }

    fn get_second_axis_prepend(&self) -> Option<String> {
        MetaData::get_second_axis_prepend(self)
    }

    fn get_target_patch_size_x_and_y(&self) -> String {
        MetaData::get_target_patch_size_x_and_y(self)
    }

    fn get_fixed_beam_tilt_selected(&self, axis_id: AxisID) -> EtomoBoolean2 {
        MetaData::get_fixed_beam_tilt_selected(self, axis_id)
    }

    fn get_number_of_local_patches_x_and_y(&self) -> String {
        MetaData::get_number_of_local_patches_x_and_y(self)
    }

    fn get_fixed_beam_tilt(&self, axis_id: AxisID) -> EtomoNumber {
        MetaData::get_fixed_beam_tilt(self, axis_id)
    }

    fn get_no_beam_tilt_selected(&self, axis_id: AxisID) -> EtomoBoolean2 {
        MetaData::get_no_beam_tilt_selected(self, axis_id)
    }

    fn get_sample_thickness(&self, axis_id: AxisID) -> EtomoNumber {
        MetaData::get_sample_thickness(self, axis_id)
    }

    fn get_size_to_output_in_x_and_y(&self, axis_id: AxisID) -> FortranInputString {
        MetaData::get_size_to_output_in_x_and_y(self, axis_id)
    }

    fn get_pos_binning(&self, axis_id: AxisID) -> i32 {
        MetaData::get_pos_binning(self, axis_id)
    }

    fn get_stack_binning(&self, axis_id: AxisID) -> i32 {
        MetaData::get_stack_binning(self, axis_id)
    }

    fn is_stack_3d_find_binning_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_stack_3d_find_binning_set(self, axis_id)
    }

    fn get_stack_3d_find_binning(&self, axis_id: AxisID) -> i32 {
        MetaData::get_stack_3d_find_binning(self, axis_id)
    }

    fn get_tilt_parallel(&self, axis_id: AxisID, panel_id: PanelId) -> Option<EtomoBoolean2> {
        MetaData::get_tilt_parallel(self, axis_id, panel_id)
    }

    fn get_final_stack_ctf_correction_parallel(&self, axis_id: AxisID) -> Option<EtomoBoolean2> {
        MetaData::get_final_stack_ctf_correction_parallel(self, axis_id)
    }

    fn is_distortion_correction(&self) -> bool {
        MetaData::is_distortion_correction(self)
    }

    fn is_final_stack_better_radius_empty(&self, axis_id: AxisID) -> bool {
        MetaData::is_final_stack_better_radius_empty(self, axis_id)
    }

    fn get_final_stack_better_radius(&self, axis_id: AxisID) -> String {
        MetaData::get_final_stack_better_radius(self, axis_id)
    }

    fn is_final_stack_fiducial_diameter_null(&self, axis_id: AxisID) -> bool {
        MetaData::is_final_stack_fiducial_diameter_null(self, axis_id)
    }

    fn get_final_stack_fiducial_diameter(&self, axis_id: AxisID) -> String {
        MetaData::get_final_stack_fiducial_diameter(self, axis_id)
    }

    fn get_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> i32 {
        MetaData::get_final_stack_expand_circle_iterations(self, axis_id)
    }

    fn is_final_stack_expand_circle_iterations_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_final_stack_expand_circle_iterations_set(self, axis_id)
    }

    fn get_final_stack_polynomial_order(&self, axis_id: AxisID) -> i32 {
        MetaData::get_final_stack_polynomial_order(self, axis_id)
    }

    fn is_final_aligned_stack_dialog_saved(&self, axis_id: AxisID) -> bool {
        MetaData::is_final_aligned_stack_dialog_saved(self, axis_id)
    }

    fn get_tomo_gen_trial_tomogram_name_list(&self, axis_id: AxisID) -> Arc<Mutex<IntKeyList>> {
        MetaData::get_tomo_gen_trial_tomogram_name_list(self, axis_id)
    }

    fn get_track_raptor_use_raw_stack(&self) -> bool {
        MetaData::get_track_raptor_use_raw_stack(self)
    }

    fn get_track_raptor_mark(&self) -> String {
        MetaData::get_track_raptor_mark(self)
    }

    fn get_track_raptor_diam(&self) -> EtomoNumber {
        MetaData::get_track_raptor_diam(self)
    }

    fn get_erase_gold_model_use_fid(&self, axis_id: AxisID) -> bool {
        MetaData::get_erase_gold_model_use_fid(self, axis_id)
    }

    fn is_post_flatten_warp_input_trim_vol(&self) -> bool {
        MetaData::is_post_flatten_warp_input_trim_vol(self)
    }

    fn is_post_flatten_warp_contours_on_one_surface(&self) -> bool {
        MetaData::is_post_flatten_warp_contours_on_one_surface(self)
    }

    fn get_post_flatten_warp_spacing_in_x(&self) -> String {
        MetaData::get_post_flatten_warp_spacing_in_x(self)
    }

    fn get_post_flatten_warp_spacing_in_y(&self) -> String {
        MetaData::get_post_flatten_warp_spacing_in_y(self)
    }

    fn is_post_squeeze_vol_input_trim_vol(&self) -> bool {
        MetaData::is_post_squeeze_vol_input_trim_vol(self)
    }

    fn is_post_trimvol_convert_to_bytes(&self) -> bool {
        MetaData::is_post_trimvol_convert_to_bytes(self)
    }

    fn is_post_trimvol_fixed_scaling(&self) -> bool {
        MetaData::is_post_trimvol_fixed_scaling(self)
    }

    fn is_post_trimvol_rotate_x(&self) -> bool {
        MetaData::is_post_trimvol_rotate_x(self)
    }

    fn is_fiducialess_alignment(&self, axis_id: AxisID) -> bool {
        MetaData::is_fiducialess_alignment(self, axis_id)
    }

    fn get_lambda_for_smoothing(&self) -> String {
        MetaData::get_lambda_for_smoothing(self)
    }

    fn get_lambda_for_smoothing_list(&self) -> String {
        MetaData::get_lambda_for_smoothing_list(self)
    }

    fn is_lambda_for_smoothing_list_empty(&self) -> bool {
        MetaData::is_lambda_for_smoothing_list_empty(self)
    }

    fn get_track_overlap_of_patches_x_and_y(&self, axis_id: AxisID) -> String {
        MetaData::get_track_overlap_of_patches_x_and_y(self, axis_id)
    }

    fn get_track_number_of_patches_x_and_y(&self, axis_id: AxisID) -> String {
        MetaData::get_track_number_of_patches_x_and_y(self, axis_id)
    }

    fn get_track_length_and_overlap(&self, axis_id: AxisID) -> String {
        MetaData::get_track_length_and_overlap(self, axis_id)
    }

    fn is_track_overlap_of_patches_x_and_y_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_overlap_of_patches_x_and_y_set(self, axis_id)
    }

    fn is_track_length_and_overlap_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_length_and_overlap_set(self, axis_id)
    }

    fn get_track_method(&self, axis_id: AxisID) -> String {
        MetaData::get_track_method(self, axis_id)
    }

    fn get_gen_log(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_log(self, axis_id)
    }

    fn get_gen_scale_factor_log(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_scale_factor_log(self, axis_id)
    }

    fn get_gen_scale_offset_log(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_scale_offset_log(self, axis_id)
    }

    fn get_gen_scale_factor_linear(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_scale_factor_linear(self, axis_id)
    }

    fn get_gen_scale_offset_linear(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_scale_offset_linear(self, axis_id)
    }

    fn is_gen_scale_factor_linear_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_scale_factor_linear_set(self, axis_id)
    }

    fn is_gen_scale_offset_linear_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_scale_offset_linear_set(self, axis_id)
    }

    fn get_gen_super_sample_factor(&self, axis_id: AxisID) -> EtomoNumber {
        MetaData::get_gen_super_sample_factor(self, axis_id)
    }

    fn get_gen_expand_input_lines(&self, axis_id: AxisID) -> EtomoBoolean2 {
        MetaData::get_gen_expand_input_lines(self, axis_id)
    }

    fn is_gen_back_projection(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_back_projection(self, axis_id)
    }

    fn is_gen_filter_trials(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_filter_trials(self, axis_id)
    }

    fn get_gen_subarea_size(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_subarea_size(self, axis_id)
    }

    fn get_gen_y_offset_of_subarea(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_y_offset_of_subarea(self, axis_id)
    }

    fn is_gen_subarea(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_subarea(self, axis_id)
    }

    fn get_radial_radius(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        MetaData::get_radial_radius(self, panel_id, axis_id)
    }

    fn get_radial_sigma(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        MetaData::get_radial_sigma(self, panel_id, axis_id)
    }

    fn is_use_final_stack_expand_circle_iterations(&self, axis_id: AxisID) -> bool {
        MetaData::is_use_final_stack_expand_circle_iterations(self, axis_id)
    }

    fn is_post_trimvol_swap_yz(&self) -> bool {
        MetaData::is_post_trimvol_swap_yz(self)
    }

    fn get_post_trimvol_fixed_scale_max(&self) -> String {
        MetaData::get_post_trimvol_fixed_scale_max(self)
    }

    fn get_post_trimvol_fixed_scale_min(&self) -> String {
        MetaData::get_post_trimvol_fixed_scale_min(self)
    }

    fn get_post_trimvol_scale_x_max(&self) -> String {
        MetaData::get_post_trimvol_scale_x_max(self)
    }

    fn get_post_trimvol_scale_x_min(&self) -> String {
        MetaData::get_post_trimvol_scale_x_min(self)
    }

    fn get_post_trimvol_scale_y_min(&self) -> String {
        MetaData::get_post_trimvol_scale_y_min(self)
    }

    fn get_post_trimvol_scale_y_max(&self) -> String {
        MetaData::get_post_trimvol_scale_y_max(self)
    }

    fn get_post_trimvol_section_scale_max(&self) -> String {
        MetaData::get_post_trimvol_section_scale_max(self)
    }

    fn get_post_trimvol_section_scale_min(&self) -> String {
        MetaData::get_post_trimvol_section_scale_min(self)
    }

    fn get_post_trimvol_x_max(&self) -> String {
        MetaData::get_post_trimvol_x_max(self)
    }

    fn get_post_trimvol_x_min(&self) -> String {
        MetaData::get_post_trimvol_x_min(self)
    }

    fn get_post_trimvol_y_min(&self) -> String {
        MetaData::get_post_trimvol_y_min(self)
    }

    fn get_post_trimvol_y_max(&self) -> String {
        MetaData::get_post_trimvol_y_max(self)
    }

    fn get_post_trimvol_z_min(&self) -> String {
        MetaData::get_post_trimvol_z_min(self)
    }

    fn get_post_trimvol_z_max(&self) -> String {
        MetaData::get_post_trimvol_z_max(self)
    }

    fn is_post_reduce_filt_vol_reduction_factor(&self) -> bool {
        MetaData::is_post_reduce_filt_vol_reduction_factor(self)
    }

    fn get_post_reduce_filt_vol_reduction_factor(&self) -> String {
        MetaData::get_post_reduce_filt_vol_reduction_factor(self)
    }

    fn get_post_reduce_filt_vol_reduction_factor_etomo_number(&self) -> EtomoNumber {
        MetaData::get_post_reduce_filt_vol_reduction_factor_etomo_number(self)
    }

    fn is_post_reduce_filt_vol_z_reduction_factor(&self) -> bool {
        MetaData::is_post_reduce_filt_vol_z_reduction_factor(self)
    }

    fn get_post_reduce_filt_vol_z_reduction_factor(&self) -> String {
        MetaData::get_post_reduce_filt_vol_z_reduction_factor(self)
    }

    fn get_post_reduce_filt_vol_z_reduction_factor_etomo_number(&self) -> EtomoNumber {
        MetaData::get_post_reduce_filt_vol_z_reduction_factor_etomo_number(self)
    }

    fn get_post_reduce_filt_vol_low_pass_radius_sigma(&self) -> String {
        MetaData::get_post_reduce_filt_vol_low_pass_radius_sigma(self)
    }

    fn get_post_reduce_filt_vol_deconvolution_strength(&self) -> String {
        MetaData::get_post_reduce_filt_vol_deconvolution_strength(self)
    }

    fn get_post_reduce_filt_vol_snr_falloff(&self) -> String {
        MetaData::get_post_reduce_filt_vol_snr_falloff(self)
    }

    fn get_post_reduce_filt_vol_high_pass_nyquist(&self) -> String {
        MetaData::get_post_reduce_filt_vol_high_pass_nyquist(self)
    }

    fn get_post_reduce_filt_vol_defocus_in_microns(&self) -> String {
        MetaData::get_post_reduce_filt_vol_defocus_in_microns(self)
    }

    fn get_post_reduce_filt_vol_phase_shift(&self) -> String {
        MetaData::get_post_reduce_filt_vol_phase_shift(self)
    }

    fn is_erase_beads_initialized(&self) -> bool {
        MetaData::is_erase_beads_initialized(self)
    }

    fn is_track_seed_model_manual(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_seed_model_manual(self, axis_id)
    }

    fn is_track_seed_model_auto(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_seed_model_auto(self, axis_id)
    }

    fn is_track_seed_model_transfer(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_seed_model_transfer(self, axis_id)
    }

    fn is_track_exclude_inside_areas(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_exclude_inside_areas(self, axis_id)
    }

    fn get_track_just_find_shifts_near_zero(&self, axis_id: AxisID) -> String {
        MetaData::get_track_just_find_shifts_near_zero(self, axis_id)
    }

    fn get_track_target_number_of_beads(&self, axis_id: AxisID) -> String {
        MetaData::get_track_target_number_of_beads(self, axis_id)
    }

    fn get_track_target_density_of_beads(&self, axis_id: AxisID) -> String {
        MetaData::get_track_target_density_of_beads(self, axis_id)
    }

    fn is_track_clustered_points_allowed_elongated(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_clustered_points_allowed_elongated(self, axis_id)
    }

    fn get_track_clustered_points_allowed_elongated_value(&self, axis_id: AxisID) -> i32 {
        MetaData::get_track_clustered_points_allowed_elongated_value(self, axis_id)
    }

    fn is_track_advanced(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_advanced(self, axis_id)
    }

    fn is_stack_3d_find_thickness_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_stack_3d_find_thickness_set(self, axis_id)
    }

    fn get_stack_3d_find_thickness(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_3d_find_thickness(self, axis_id)
    }

    fn is_set_fei_pixel_size(&self) -> bool {
        MetaData::is_set_fei_pixel_size(self)
    }

    fn is_twodir(&self, axis_id: AxisID) -> bool {
        MetaData::is_twodir(self, axis_id)
    }

    fn get_twodir(&self, axis_id: AxisID) -> String {
        MetaData::get_twodir(self, axis_id)
    }

    fn get_raptor_tab(&self, axis_id: AxisID) -> i32 {
        MetaData::get_raptor_tab(self, axis_id)
    }

    fn get_seed_and_track_tab(&self, axis_id: AxisID) -> i32 {
        MetaData::get_seed_and_track_tab(self, axis_id)
    }

    fn get_antialias_filter(
        &self,
        dialog_type: DialogType,
        axis_id: AxisID,
    ) -> Option<EtomoNumber> {
        MetaData::get_antialias_filter(self, dialog_type, axis_id)
    }

    fn is_antialias_filter_null(&self, dialog_type: DialogType, axis_id: AxisID) -> bool {
        MetaData::is_antialias_filter_null(self, dialog_type, axis_id)
    }

    fn is_track_elongated_points_allowed_null(&self, axis_id: AxisID) -> bool {
        MetaData::is_track_elongated_points_allowed_null(self, axis_id)
    }

    fn get_track_elongated_points_allowed(&self, axis_id: AxisID) -> EtomoNumber {
        MetaData::get_track_elongated_points_allowed(self, axis_id)
    }

    fn get_track_lower_target_for_clustered(&self, axis_id: AxisID) -> String {
        MetaData::get_track_lower_target_for_clustered(self, axis_id)
    }

    fn get_weight_whole_tracks(&self, axis_id: AxisID) -> bool {
        MetaData::get_weight_whole_tracks(self, axis_id)
    }

    fn get_length_of_pieces(&self, axis_id: AxisID) -> String {
        MetaData::get_length_of_pieces(self, axis_id)
    }

    fn get_minimum_overlap(&self, axis_id: AxisID) -> String {
        MetaData::get_minimum_overlap(self, axis_id)
    }

    fn get_target_measurement_ratio(&self, axis_id: AxisID) -> String {
        MetaData::get_target_measurement_ratio(self, axis_id)
    }

    fn get_min_measurement_ratio(&self, axis_id: AxisID) -> String {
        MetaData::get_min_measurement_ratio(self, axis_id)
    }

    fn is_target_measurement_ratio_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_target_measurement_ratio_set(self, axis_id)
    }

    fn is_min_measurement_ratio_set(&self, axis_id: AxisID) -> bool {
        MetaData::is_min_measurement_ratio_set(self, axis_id)
    }

    fn get_sample_type(&self, axis_id: AxisID) -> Option<SampleType> {
        MetaData::get_sample_type(self, axis_id)
    }

    fn is_has_gold_beads(&self, axis_id: AxisID) -> bool {
        MetaData::is_has_gold_beads(self, axis_id)
    }

    fn get_positioning_fiducial_diameter(&self, axis_id: AxisID) -> f64 {
        MetaData::get_positioning_fiducial_diameter(self, axis_id)
    }

    fn get_positioning_bead_size(&self, axis_id: AxisID) -> String {
        MetaData::get_positioning_bead_size(self, axis_id)
    }

    fn is_has_gold_beads_null(&self, axis_id: AxisID) -> bool {
        MetaData::is_has_gold_beads_null(self, axis_id)
    }

    fn is_positioning_fiducial_diameter_null(&self, axis_id: AxisID) -> bool {
        MetaData::is_positioning_fiducial_diameter_null(self, axis_id)
    }

    fn is_positioning_bead_size_null(&self, axis_id: AxisID) -> bool {
        MetaData::is_positioning_bead_size_null(self, axis_id)
    }

    fn get_extra_thickness(&self, axis_id: AxisID) -> String {
        MetaData::get_extra_thickness(self, axis_id)
    }

    fn get_extra_thickness_cryo(&self, axis_id: AxisID) -> String {
        MetaData::get_extra_thickness_cryo(self, axis_id)
    }

    fn get_hamming_like_filter(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        MetaData::get_hamming_like_filter(self, panel_id, axis_id)
    }

    fn get_fake_sirt_iterations(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        MetaData::get_fake_sirt_iterations(self, panel_id, axis_id)
    }

    fn get_exact_filter_size(&self, panel_id: PanelId, axis_id: AxisID) -> Option<String> {
        MetaData::get_exact_filter_size(self, panel_id, axis_id)
    }

    fn is_orig_scope_template(&self) -> bool {
        MetaData::is_orig_scope_template(self)
    }

    fn get_orig_scope_template(&self) -> String {
        MetaData::get_orig_scope_template(self)
    }

    fn is_orig_system_template(&self) -> bool {
        MetaData::is_orig_system_template(self)
    }

    fn get_orig_system_template(&self) -> String {
        MetaData::get_orig_system_template(self)
    }

    fn is_orig_user_template(&self) -> bool {
        MetaData::is_orig_user_template(self)
    }

    fn get_orig_user_template(&self) -> String {
        MetaData::get_orig_user_template(self)
    }

    fn is_fiducial_diameter_available(&self) -> bool {
        MetaData::is_fiducial_diameter_available(self)
    }

    fn is_positioning_new_dialog(&self, axis_id: AxisID) -> bool {
        MetaData::is_positioning_new_dialog(self, axis_id)
    }

    fn get_gen_filter_trials_fake_sirt_iterations(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_filter_trials_fake_sirt_iterations(self, axis_id)
    }

    fn get_gen_filter_trials_exact_object_sizes(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_filter_trials_exact_object_sizes(self, axis_id)
    }

    fn get_gen_filter_trials_gaussian_cutoffs(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_filter_trials_gaussian_cutoffs(self, axis_id)
    }

    fn get_gen_filter_trials_gaussian_falloffs(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_filter_trials_gaussian_falloffs(self, axis_id)
    }

    fn get_gen_filter_trials_hamming_like_starts(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_filter_trials_hamming_like_starts(self, axis_id)
    }

    fn get_stack_mtf_filter_low_pass_radius_sigma(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_low_pass_radius_sigma(self, axis_id)
    }

    fn get_stack_mtf_filter_mtf_file(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_mtf_file(self, axis_id)
    }

    fn get_stack_mtf_filter_maximum_inverse(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_maximum_inverse(self, axis_id)
    }

    fn get_stack_mtf_filter_inverse_rolloff_radius_sigma(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_inverse_rolloff_radius_sigma(self, axis_id)
    }

    fn is_use_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID) -> bool {
        MetaData::is_use_stack_mtf_filter_fixed_image_dose(self, axis_id)
    }

    fn get_stack_mtf_filter_fixed_image_dose(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_fixed_image_dose(self, axis_id)
    }

    fn get_stack_mtf_filter_dose_weighting_file(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_dose_weighting_file(self, axis_id)
    }

    fn get_stack_mtf_filter_type_of_dose_file(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_type_of_dose_file(self, axis_id)
    }

    fn is_stack_mtf_filter_voltage_200(&self, axis_id: AxisID) -> bool {
        MetaData::is_stack_mtf_filter_voltage_200(self, axis_id)
    }

    fn get_stack_mtf_filter_optimal_dose_scaling(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_optimal_dose_scaling(self, axis_id)
    }

    fn get_stack_mtf_filter_bidirectional_num_views(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_mtf_filter_bidirectional_num_views(self, axis_id)
    }

    fn is_use_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID) -> bool {
        MetaData::is_use_stack_ctf_phase_flip_x_axis_tilt(self, axis_id)
    }

    fn get_stack_ctf_phase_flip_x_axis_tilt(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_ctf_phase_flip_x_axis_tilt(self, axis_id)
    }

    fn get_stack_ctf_phase_flip_scale_by_ctf_power(&self, axis_id: AxisID) -> String {
        MetaData::get_stack_ctf_phase_flip_scale_by_ctf_power(self, axis_id)
    }

    fn is_gen_sirt(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_sirt(self, axis_id)
    }

    fn is_gen_ctf_3d_old_style_xtilting(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_ctf_3d_old_style_xtilting(self, axis_id)
    }

    fn is_gen_ctf_3d_vertical_slices(&self, axis_id: AxisID) -> bool {
        MetaData::is_gen_ctf_3d_vertical_slices(self, axis_id)
    }

    fn get_gen_ctf_3d_fourier_reduce_by_factor(&self, axis_id: AxisID) -> String {
        MetaData::get_gen_ctf_3d_fourier_reduce_by_factor(self, axis_id)
    }

    fn is_subtomo_reorientation_type_none(&self) -> bool {
        MetaData::is_subtomo_reorientation_type_none(self)
    }

    fn is_subtomo_reorientation_type_flipped(&self) -> bool {
        MetaData::is_subtomo_reorientation_type_flipped(self)
    }

    fn is_subtomo_reorientation_type_rotated(&self) -> bool {
        MetaData::is_subtomo_reorientation_type_rotated(self)
    }

    fn is_subtomo_make_volume_stacks(&self) -> bool {
        MetaData::is_subtomo_make_volume_stacks(self)
    }

    fn get_subtomo_make_volume_stacks(&self) -> String {
        MetaData::get_subtomo_make_volume_stacks(self)
    }

    fn get_subtomo_extent_of_z_levels_in_nm(&self) -> String {
        MetaData::get_subtomo_extent_of_z_levels_in_nm(self)
    }

    fn get_subtomo_new_aligned_binning(&self) -> String {
        MetaData::get_subtomo_new_aligned_binning(self)
    }

    fn get_subtomo_fourier_reduce_by_factor(&self) -> String {
        MetaData::get_subtomo_fourier_reduce_by_factor(self)
    }

    fn get_fine_local_align_validation(&self, axis_id: AxisID) -> String {
        MetaData::get_fine_local_align_validation(self, axis_id)
    }

    fn is_dose_sym(&self, axis_id: AxisID) -> bool {
        MetaData::is_dose_sym(self, axis_id)
    }

    fn get_dose_sym(&self, axis_id: AxisID) -> String {
        MetaData::get_dose_sym(self, axis_id)
    }

    fn get_alt_tomo_rootname(&self) -> String {
        MetaData::get_alt_tomo_rootname(self)
    }

    fn is_alt_tomo_trim_volume(&self) -> bool {
        MetaData::is_alt_tomo_trim_volume(self)
    }

    fn is_alt_tomo_archive_orig_stack(&self) -> bool {
        MetaData::is_alt_tomo_archive_orig_stack(self)
    }

    fn get_image_filename_style(&self) -> ImageFilenameStyle {
        self.base.get_image_filename_style()
    }

    fn get_axis_type(&self) -> AxisType {
        self.base.get_axis_type()
    }

    fn get_combine_params(&self) -> MutexGuard<'_, CombineParams> {
        MetaData::get_combine_params(self)
    }
}

/// Java `toString`: `"[datasetName:" + datasetName + "," + super.toString() + "]"`.
impl std::fmt::Display for MetaData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dataset_name = self.dataset_name.lock().unwrap().clone();
        write!(f, "[datasetName:{},{}]", dataset_name, self.base)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// `ts8.edf` as written by the Java eTomo reference (`java.util.Properties.store`
    /// of `MetaData.store(props, "")`, compiled from the vendored source and run
    /// headless; `/tmp/mdref/MdRef2.java` in the session that wrote this module).  The
    /// dataset is dual axis and sets a value in most of the property families.  The
    /// Java round trip (load this file into a new `MetaData`, store again) reproduces
    /// it exactly.
    const JAVA_EDF: &str = r#"#Sun Sep 27 15:14:56 CEST 2026
Setup=-Infinity,-Infinity
Setup.A.SizeToOutputInXandY=1024,1024
Setup.AltTomoSetup.RootnameToProcess=alt
Setup.AxisA.ExcludeProjections=13
Setup.AxisA.TiltAngle.RangeMin=-54.0
Setup.AxisA.TiltAngle.RangeStep=3.0
Setup.AxisA.TiltAngle.TiltAngleFilename=
Setup.AxisA.TiltAngle.Type=Range
Setup.AxisB.ExcludeProjections=7
Setup.AxisB.TiltAngle.RangeMin=-60.0
Setup.AxisB.TiltAngle.RangeStep=1.0
Setup.AxisB.TiltAngle.TiltAngleFilename=
Setup.AxisB.TiltAngle.Type=Extract
Setup.AxisType=Dual Axis
Setup.B.SizeToOutputInXandY=/
Setup.BStackProcessed=true
Setup.BackupDirectory=/tmp/backup
Setup.BatchRunTomoLog.Read.AxisID=b
Setup.BatchRunTomoLog.Read.Finished=true
Setup.Binning=2.0
Setup.ComScriptsCreated=true
Setup.Combine.FiducialMatch=BothSides
Setup.Combine.FiducialMatchListA=
Setup.Combine.FiducialMatchListB=
Setup.Combine.ManualCleanup=false
Setup.Combine.MaxPatchBoundaryZMax=0
Setup.Combine.ModelBased=false
Setup.Combine.PatchBoundaryXMax=0
Setup.Combine.PatchBoundaryXMin=0
Setup.Combine.PatchBoundaryYMax=0
Setup.Combine.PatchBoundaryYMin=0
Setup.Combine.PatchBoundaryZMax=0
Setup.Combine.PatchBoundaryZMin=0
Setup.Combine.PatchRegionModel=
Setup.Combine.PatchSize=M
Setup.Combine.RevisionNumber=1.2
Setup.Combine.TempDirectory=
Setup.Combine.Transfer=true
Setup.Combine.UseList=
Setup.Combine.Volcombine.Parallel=true
Setup.DataSource=CCD
Setup.DatasetName=ts8
Setup.DefaultParallel=true
Setup.DistortionFile=dist.idf
Setup.FiducialDiameter=10.0
Setup.FiducialessAlignmentA=false
Setup.FiducialessAlignmentB=true
Setup.FinalStack.a.FiducialDiameter=12.5
Setup.FinalStackB.CtfCorrection.Parallel=true
Setup.FinalStackBinningB=2
Setup.Fine.A.OrderOfRestrictions=1,2
Setup.Fine.B.SkipBeamTiltWithOneRot=true
Setup.Gen.A.Exists=true
Setup.Gen.A.HammingLikeFilter=0.35
Setup.Gen.A.Log=5.0
Setup.Gen.CurTab=2
Setup.HalfFloatModeOutput=1
Setup.ImageFile.ImageFilenameStyle=MRC
Setup.ImageRotationA=85.3
Setup.ImageRotationB=-4.5
Setup.OrigRawImageStackExt=st
Setup.PixelSize=2.5
Setup.Pos.A.FiducialDiameter=6.2
Setup.Pos.A.HasGoldBeads=true
Setup.Pos.A.NewDialog=true
Setup.Pos.A.SampleType=2
Setup.Pos.B.NewDialog=true
Setup.Post.Exists=true
Setup.Post.LambdaForSmoothingList=1,2,3
Setup.Post.Trimvol.ScaleXMin=3
Setup.Post.Trimvol.SwapYZ=true
Setup.Post.Trimvol.XMin=10
Setup.RawImageStackExt=st
Setup.RevisionNumber=1.12
Setup.Second.sample.THICKNESS=250
Setup.Sirt.B.RadialRadius=0.4
Setup.Stack.A.3dFind.Binning=3
Setup.Stack.A.DoseSym=0.0
Setup.Stack.A.Is.Twodir=true
Setup.Stack.A.MtfFilter.Voltage.200=true
Setup.Stack.A.Twodir=12.5
Setup.Stack.B.CTF.AutoFit.RangeAndStep=-Infinity,-Infinity
Setup.Stack.B.DoseSym=0.0
Setup.Stack.B.EraseGold.ModelUseFid=true
Setup.Stack.B.Tilt.Parallel=false
Setup.Stack.B.Twodir=0.0
Setup.Subtomo.ExtentOfZLevelsInNm=20
Setup.TomoGenA.Tilt.Parallel=true
Setup.TomoPosBinningA=4
Setup.Track.A.ElongatedPointsAllowed=3
Setup.Track.A.Raptor.UseRawStack=false
Setup.Track.A.SeedModel.Auto=true
Setup.Track.A.TargetNumberOfBeads=25
Setup.Track.A.TrackMethod=Seed
Setup.Track.B.SeedModel.Manual=true
Setup.Track.B.SeedModel.Transfer=true
Setup.Track.B.TargetDensityOfBeads=1.5
Setup.Track.B.TrackMethod=PatchTracking
Setup.UseLocalAlignmentsA=false
Setup.UseLocalAlignmentsB=true
Setup.Version.Etomo.Created=5.2.17
Setup.Version.Etomo.Modified=5.2.17
Setup.ViewType=Single View
Setup.WholeTomogramSampleA=true
Setup.WholeTomogramSampleB=false
Setup.b.FineAlign.NoBeamTiltSelected=false
Setup.ctf3dsetup.SlabThicknessInNmSet=true
Setup.tiltalign.NumberOfLocalPatchesXandY=5,5
Setup.tiltalign.TargetPatchSizeXandY=600,600
"#;

    /// Parse the `key=value` lines of a `Properties.store` file.  The reference file
    /// has no escaped characters.
    fn parse_edf(text: &str) -> BTreeMap<String, String> {
        let mut props = BTreeMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let (key, value) = line.split_once('=').unwrap();
            props.insert(key.to_string(), value.to_string());
        }
        props
    }

    /// The Java map with the differences this translation defines: the
    /// `stackCtfAutoFitRangeAndStep` properties-key fix (native writes A under the B
    /// key and B under the bare group key `Setup`).
    /// `range_and_step_a` is what A's own `RangeAndStep` key must hold: the unset
    /// value Java stores ("-Infinity,-Infinity") after the setter sequence, or the
    /// default ("/") after loading the Java file, which has no A key.
    fn expected_from_java(
        java: &BTreeMap<String, String>,
        range_and_step_a: &str,
    ) -> BTreeMap<String, String> {
        let mut expected = BTreeMap::new();
        for (key, value) in java {
            if key == "Setup" {
                continue;
            }
            expected.insert(key.clone(), value.clone());
        }
        // The Java file carries A's value under the B key and B's under the bare
        // `Setup` key (MetaData.java:1250-1253, fixed in translation).  An unset
        // FortranInputString stores as "-Infinity,-Infinity" (the `Setup=` line);
        // a missing key loads as every value the default ("/").
        expected.insert(
            "Setup.Stack.A.CTF.AutoFit.RangeAndStep".to_string(),
            range_and_step_a.to_string(),
        );
        expected.insert(
            "Setup.Stack.B.CTF.AutoFit.RangeAndStep".to_string(),
            "-Infinity,-Infinity".to_string(),
        );
        expected
    }

    fn store(meta_data: &MetaData) -> BTreeMap<String, String> {
        let mut props = BTreeMap::new();
        meta_data.store_with_prepend(&mut props, "");
        props
    }

    #[test]
    fn construct_set_store_load_round_trip() {
        let meta_data = MetaData::new(None, None, true);
        meta_data.set_axis_type(AxisType::SingleAxis);
        meta_data.set_pixel_size_double(1.25);
        meta_data.set_fiducial_diameter_string(Some("8"));
        meta_data.set_image_rotation(Some("-12.5"), AxisID::Only);
        meta_data.set_binning(Some("3"));
        meta_data.set_exclude_projections(Some(" 4 5 "), AxisID::Only);
        meta_data.set_gen_log(AxisID::First, Some("2.5"));
        meta_data.set_post_trimvol_z_max(Some("80"));
        meta_data.set_track_advanced(true, AxisID::First);
        meta_data.set_tilt_parallel(AxisID::Only, PanelId::Tilt, true);
        let mut props = store(&meta_data);
        assert_eq!(props.get("Setup.AxisType").unwrap(), "Single Axis");
        assert_eq!(props.get("Setup.PixelSize").unwrap(), "1.25");
        assert_eq!(props.get("Setup.FiducialDiameter").unwrap(), "8.0");
        assert_eq!(props.get("Setup.ImageRotationA").unwrap(), "-12.5");
        assert_eq!(props.get("Setup.ImageRotationB").unwrap(), "");
        assert_eq!(props.get("Setup.Binning").unwrap(), "3.0");
        assert_eq!(props.get("Setup.AxisA.ExcludeProjections").unwrap(), "45");
        assert_eq!(props.get("Setup.Gen.A.Log").unwrap(), "2.5");
        assert_eq!(props.get("Setup.TomoGenA.Tilt.Parallel").unwrap(), "true");
        assert_eq!(props.get("Setup.RevisionNumber").unwrap(), "1.12");
        assert!(!props.contains_key("Setup"));

        let loaded = MetaData::new(None, None, false);
        loaded.load(&mut props);
        assert_eq!(loaded.get_axis_type(), AxisType::SingleAxis);
        assert_eq!(loaded.get_pixel_size(), 1.25);
        assert_eq!(loaded.get_exclude_projections_a(), "45");
        assert!(loaded.is_track_advanced(AxisID::First));
        assert_eq!(loaded.get_first_axis_prepend().as_deref(), Some(""));
        assert!(
            loaded
                .get_tilt_parallel(AxisID::First, PanelId::Tilt)
                .unwrap()
                .is()
        );
        // MetaData.load sets an unset ImageRotationB to the old single rotation,
        // default "0.0" (MetaData.java:3150-3156), so it does not round-trip.
        let mut expected = props.clone();
        expected.insert("Setup.ImageRotationB".to_string(), "0.0".to_string());
        assert_eq!(store(&loaded), expected);
    }

    #[test]
    fn java_edf_load_store_matches_java_round_trip() {
        let mut java = parse_edf(JAVA_EDF);
        let meta_data = MetaData::new(None, None, false);
        meta_data.load(&mut java);
        assert_eq!(meta_data.get_dataset_name(), "ts8");
        assert_eq!(meta_data.get_meta_data_file_name(), "ts8.edf");
        assert_eq!(meta_data.get_axis_type(), AxisType::DualAxis);
        assert_eq!(meta_data.get_first_axis_prepend().as_deref(), Some("A"));
        assert_eq!(meta_data.get_second_axis_prepend().as_deref(), Some("B"));
        assert_eq!(
            meta_data.get_sample_type(AxisID::First),
            Some(SampleType::Cryo)
        );
        assert_eq!(meta_data.get_half_float_mode_output(), Some(1));
        assert_eq!(
            meta_data.get_tilt_angle_spec_a().get_tilt_angles(),
            "-54.0,3.0"
        );
        assert_eq!(
            meta_data.get_batch_run_tomo_log_read_axis_id(),
            Some(AxisID::Second)
        );
        let stored = store(&meta_data);
        let expected = expected_from_java(&java, "/");
        let missing: Vec<_> = expected
            .keys()
            .filter(|k| !stored.contains_key(*k))
            .collect();
        let extra: Vec<_> = stored
            .keys()
            .filter(|k| !expected.contains_key(*k))
            .collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "missing {missing:?}, extra {extra:?}"
        );
        for (key, value) in &expected {
            assert_eq!(stored.get(key).unwrap(), value, "value of {key}");
        }
    }

    #[test]
    fn java_set_sequence_stores_the_java_edf() {
        // The setter sequence the Java reference ran before `store` (MdRef2.java).
        let md = MetaData::new(None, None, true);
        md.set_axis_type(AxisType::DualAxis);
        md.set_dataset_name("ts8a.st");
        md.set_pixel_size_double(2.5);
        md.set_fiducial_diameter_double(10.0);
        md.set_image_rotation(Some("85.3"), AxisID::First);
        md.set_image_rotation(Some("-4.5"), AxisID::Second);
        md.set_binning(Some("2"));
        md.set_exclude_projections(Some("1 3"), AxisID::First);
        md.set_exclude_projections(Some("7"), AxisID::Second);
        md.set_backup_directory(Some(" /tmp/backup "));
        md.set_distortion_file(Some("dist.idf"));
        md.set_sample_thickness(AxisID::Second, Some("250"));
        md.set_track_seed_model_manual(true, AxisID::Second);
        md.set_track_target_number_of_beads(Some("25"), AxisID::First);
        md.set_track_target_density_of_beads(Some("1.5"), AxisID::Second);
        md.set_pos_binning_int(AxisID::First, 4);
        md.set_stack_binning_string(AxisID::Second, Some("2"));
        md.set_stack_3d_find_binning_int(AxisID::First, 3);
        md.set_tilt_parallel(AxisID::First, PanelId::Tilt, true);
        md.set_tilt_parallel(AxisID::Second, PanelId::Tilt3dFind, false);
        md.set_final_stack_ctf_correction_parallel_boolean(AxisID::Second, true);
        md.set_combine_volcombine_parallel_boolean(true);
        md.set_b_stack_processed_boolean(true);
        md.set_default_parallel(true);
        md.set_gen_log(AxisID::First, Some("5.0"));
        md.set_post_trimvol_x_min(Some("10"));
        md.set_post_trimvol_scale_x_min(Some("3"));
        md.set_post_trimvol_swap_yz(true);
        md.set_fixed_beam_tilt(AxisID::First, Some("0.5"));
        md.set_no_beam_tilt_selected(AxisID::Second, false);
        md.set_final_stack_fiducial_diameter(AxisID::First, Some("12.5"));
        md.set_track_method(AxisID::Second, Some("PatchTracking"));
        md.set_order_of_restrictions(AxisID::First, Some("1,2"));
        md.set_skip_beam_tilt_with_one_rot(AxisID::Second, true);
        md.set_has_gold_beads(AxisID::First, true);
        md.set_positioning_bead_size(AxisID::First, Some("6.2"));
        md.set_hamming_like_filter(PanelId::Tilt, AxisID::First, Some("0.35"));
        md.set_radial_radius(PanelId::Sirtsetup, AxisID::Second, Some("0.4"));
        md.set_batch_run_tomo_log_read_axis_id(Some(AxisID::Second));
        md.set_batch_run_tomo_log_read_finished(true);
        md.set_stack_mtf_filter_voltage_200(AxisID::First, true);
        md.set_subtomo_extent_of_z_levels_in_nm(Some("20"));
        md.set_alt_tomo_rootname(Some("alt"));
        md.set_ctf_3d_setup_slab_thickness_in_nm_set(true);
        md.set_target_patch_size_x_and_y(Some("600,600"));
        md.set_half_float_mode_output(Some(1));
        md.set_is_twodir(AxisID::First, true);
        md.set_twodir(AxisID::First, Some("12.5"));
        md.set_lambda_for_smoothing_list(Some("1,2,3"));
        md.set_fiducialess_alignment(AxisID::Second, true);
        md.set_use_local_alignments(AxisID::First, false);
        md.set_whole_tomogram_sample(AxisID::First, true);
        md.set_com_script_created(true);
        md.set_sample_type_sample_type(AxisID::First, Some(SampleType::Cryo));
        md.set_erase_gold_model_use_fid_boolean(AxisID::Second, true);
        md.set_gen_exists(AxisID::First, true);
        md.set_post_exists(true);
        md.set_gen_cur_tab(2);
        md.set_track_elongated_points_allowed(AxisID::First, Some(Number::Integer(3)));
        md.set_size_to_output_in_x_and_y(AxisID::First, Some("1024,1024"))
            .unwrap();
        let mut spec = TiltAngleSpec::new();
        spec.set_type(super::super::tilt_angle_type::TiltAngleType::Range);
        spec.set_range_min_double(-54.0);
        spec.set_range_step_double(3.0);
        md.set_tilt_angle_spec_a(spec);
        let stored = store(&md);
        let expected = expected_from_java(&parse_edf(JAVA_EDF), "-Infinity,-Infinity");
        let missing: Vec<_> = expected
            .keys()
            .filter(|k| !stored.contains_key(*k))
            .collect();
        let extra: Vec<_> = stored
            .keys()
            .filter(|k| !expected.contains_key(*k))
            .collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "missing {missing:?}, extra {extra:?}"
        );
        for (key, value) in &expected {
            assert_eq!(stored.get(key).unwrap(), value, "value of {key}");
        }
    }

    #[test]
    fn fixed_upstream_bugs() {
        let md = MetaData::new(None, None, true);
        // getPixelSize/getFiducialDiameter: kept native, an unset value is NaN.
        assert!(md.get_pixel_size().is_nan());
        assert!(md.get_fiducial_diameter().is_nan());
        assert!(!md.is_fiducial_diameter_available());
        // setTomoGenTrialTomogramNameList(SECOND) no longer overwrites A.
        let mut list_b = IntKeyList::get_string_instance_with_key("TrialTomogramName");
        list_b.add_string(Some("b1"));
        md.set_tomo_gen_trial_tomogram_name_list(AxisID::Second, Arc::new(Mutex::new(list_b)));
        {
            use super::super::const_int_key_list::ConstIntKeyList;
            assert!(
                md.get_tomo_gen_trial_tomogram_name_list(AxisID::First)
                    .lock()
                    .unwrap()
                    .is_empty()
            );
        }
        // equals ends with `combineParams.equals(...)`, which is Object.equals: two
        // distinct MetaData objects are never equal (kept native, BUGS.md).
        let other = MetaData::new(None, None, true);
        md.set_pixel_size_double(1.0);
        md.set_fiducial_diameter_double(1.0);
        other.set_pixel_size_double(1.0);
        other.set_fiducial_diameter_double(1.0);
        assert!(!md.equals(&other));
        // An unrecognised AxisType/DataSource/ViewType loads as the default instead of a
        // null the next store dereferences.
        let mut props = BTreeMap::new();
        props.insert("Setup.AxisType".to_string(), "Triple".to_string());
        props.insert("Setup.DataSource".to_string(), "Tape".to_string());
        props.insert("Setup.PixelSize".to_string(), "abc".to_string());
        md.load(&mut props);
        assert_eq!(md.get_axis_type(), AxisType::NotSet);
        assert_eq!(md.get_data_source(), DataSource::Ccd);
        assert!(!md.is_valid_from_screen(false));
        let stored = store(&md);
        assert_eq!(stored.get("Setup.AxisType").unwrap(), "Not Set");
        assert_eq!(stored.get("Setup.PixelSize").unwrap(), "NaN");
    }

    #[test]
    fn meta_data_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetaData>();
    }
}
