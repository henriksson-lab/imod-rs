//! `IMOD/Etomo/src/etomo/type/TomogramState.java`.
//!
//! What the reconstruction has actually done (as opposed to what the dialogs are set
//! to): flips, z factors, alignment offsets, file times, sizes.  Stored in the `.edf`
//! under the `ReconstructionState` group.
//!
//! **Representation.**  `TomogramState extends BaseState`; the abstract
//! `createPrepend` is the `BaseState` trait (`base_state.rs`), whose `store`/`load`
//! bodies only compute and discard a prepend.  Like `MetaData` (see its module header),
//! the object is shared across threads, so each mutable field carries its own lock and
//! every method takes `&self`; getters that return a field object return a copy, and a
//! Java `String` parameter is `Option<&str>`.
//!
//! **`prepend == ""`.**  `createPrepend` tests `prepend == ""`, a reference comparison
//! true for the interned literal that `store(Properties)`/`load(Properties)` pass; it
//! is translated as `prepend.is_empty()`.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

use super::axis_id::AxisID;
use super::base_state::BaseState;
use super::const_etomo_number::{ConstEtomoNumber, Type};
use super::const_meta_data::ConstMetaData;
use super::const_string_property::ConstStringProperty;
use super::dialog_type::DialogType;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use super::etomo_state::{self, EtomoState};
use super::file_type;
use super::match_mode::MatchMode;
use super::meta_data::BATCHRUNTOMO_KEY;
use super::pos_sample_type::PosSampleType;
use super::process_name::ProcessName;
use super::string_property::StringProperty;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities::{
    java_io_file_get_absolute_path, java_io_file_get_name, java_io_file_last_modified,
    java_io_file_new,
};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `GROUP_STRING`.
const GROUP_STRING: &str = "ReconstructionState";

/// Java `LAST_MODIFIED`.
const LAST_MODIFIED: &str = "LastModified";

/// Java `USE_FID_AS_SEED`.
const USE_FID_AS_SEED: &str = "UseFidAsSeed";

/// Java `FIXED_FIDUCIALS_KEY`.
const FIXED_FIDUCIALS_KEY: &str = "FixedFiducials";

/// Java `COMBINE_MATCH_MODE_KEY`.
static COMBINE_MATCH_MODE_KEY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.MatchMode",
        DialogType::TomogramCombination.get_storable_name()
    )
});

/// Java `COMBINE_MATCH_MODE_BACK_KEY`.
const COMBINE_MATCH_MODE_BACK_KEY: &str = "Setup.Combine.MatchBtoA";

/// Java `COMBINE_SCRIPTS_CREATED_BACK_KEY`.
const COMBINE_SCRIPTS_CREATED_BACK_KEY: &str = "Setup.ComScriptsCreated";

/// Java `SAMPLE_FIDUCIALESS_KEY`.
static SAMPLE_FIDUCIALESS_KEY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.{}.Fiducialess",
        DialogType::TomogramPositioning.get_storable_name(),
        ProcessName::TILT
    )
});

/// Java `X_AXIS_TILT_KEY`.
const X_AXIS_TILT_KEY: &str = "XAXISTILT";

/// Java `ADJUST_ORIGIN_KEY`.
const ADJUST_ORIGIN_KEY: &str = "AdjustOrigin";

/// Java `A_AXIS_KEY`.
const A_AXIS_KEY: &str = "A";

/// Java `B_AXIS_KEY`.
const B_AXIS_KEY: &str = "B";

/// Java `TOMOGRAM_SIZE_KEY`.
const TOMOGRAM_SIZE_KEY: &str = "tomogramSize";

/// Java `PRE_KEY`.
const PRE_KEY: &str = "Pre";

/// Java `TRACK_KEY`.
const TRACK_KEY: &str = "Track";

/// Java `STACK_KEY`.
const STACK_KEY: &str = "Stack";

/// Java `GEN_KEY`.
const GEN_KEY: &str = "Gen";

/// Java `ConstTiltalignParam.ANGLE_OFFSET_KEY` (ConstTiltalignParam.java:48);
/// `TiltalignParam` has no Rust module.
const TILTALIGN_ANGLE_OFFSET_KEY: &str = "AngleOffset";

/// Java `ConstTiltalignParam.AXIS_Z_SHIFT_KEY` (ConstTiltalignParam.java:49).
const TILTALIGN_AXIS_Z_SHIFT_KEY: &str = "AxisZShift";

/// Java `TomogramState`.
pub struct TomogramState {
    /// Java `EtomoState trimvolFlipped = new EtomoState("TrimvolFlipped");`
    trimvol_flipped: Mutex<EtomoState>,
    /// Java `EtomoState altTomoPreprocessForExtremes = new EtomoState("AltTomoPreprocessForExtremes");`
    alt_tomo_preprocess_for_extremes: Mutex<EtomoState>,
    /// Java `EtomoState altTomoCorrectCTF = new EtomoState("AltTomoCorrectCTF");`
    alt_tomo_correct_ctf: Mutex<EtomoState>,
    /// Java `EtomoState altTomoEraseFiducials = new EtomoState("AltTomoEraseFiducials");`
    alt_tomo_erase_fiducials: Mutex<EtomoState>,
    /// Java `EtomoState altTomoFilterIn2D = new EtomoState("AltTomoFilterIn2D");`
    alt_tomo_filter_in_2d: Mutex<EtomoState>,
    /// Java `EtomoState altTomoTrimVol = new EtomoState("AltTomoTrimVol");`
    alt_tomo_trim_vol: Mutex<EtomoState>,
    /// Java `StringProperty altTomoRootnameToProcess = new StringProperty("RootnameToProcess");`
    alt_tomo_rootname_to_process: Mutex<StringProperty>,
    /// Java `EtomoState altTomoEvenAndOddPairs = new EtomoState("AltTomoEvenAndOddPairs");`
    alt_tomo_even_and_odd_pairs: Mutex<EtomoState>,
    /// Java `StringProperty altTomoAxisToProcess = new StringProperty("AltTomoAxisToProcess");`
    alt_tomo_axis_to_process: Mutex<StringProperty>,
    /// Java `EtomoState squeezevolFlipped = new EtomoState("SqueezevolFlipped");`
    squeezevol_flipped: Mutex<EtomoState>,
    /// Java `EtomoState flattenFlipped = new EtomoState("FlattenFlipped");`
    flatten_flipped: Mutex<EtomoState>,
    /// Java `EtomoState reduceFiltVolFlipped = new EtomoState("ReduceFiltVolFlipped");`
    reduce_filt_vol_flipped: Mutex<EtomoState>,
    /// Java `EtomoState madeZFactorsA = new EtomoState("MadeZFactors" + A_AXIS_KEY);`
    made_z_factors_a: Mutex<EtomoState>,
    /// Java `EtomoState madeZFactorsB = new EtomoState("MadeZFactorsB");`
    made_z_factors_b: Mutex<EtomoState>,
    /// Java `EtomoState newstFiducialessAlignmentA = new EtomoState("NewstFiducialessAlignmentA");`
    newst_fiducialess_alignment_a: Mutex<EtomoState>,
    /// Java `EtomoState newstFiducialessAlignmentB = new EtomoState("NewstFiducialessAlignmentB");`
    newst_fiducialess_alignment_b: Mutex<EtomoState>,
    /// Java `EtomoState usedLocalAlignmentsA = new EtomoState("UsedLocalAlignments" + A_AXIS_KEY);`
    used_local_alignments_a: Mutex<EtomoState>,
    /// Java `EtomoState usedLocalAlignmentsB = new EtomoState("UsedLocalAlignments" + B_AXIS_KEY);`
    used_local_alignments_b: Mutex<EtomoState>,
    /// Java `EtomoState invalidEdgeFunctionsA = new EtomoState("InvalidEdgeFunctions" + A_AXIS_KEY);`
    invalid_edge_functions_a: Mutex<EtomoState>,
    /// Java `EtomoState invalidEdgeFunctionsB = new EtomoState("InvalidEdgeFunctions" + B_AXIS_KEY);`
    invalid_edge_functions_b: Mutex<EtomoState>,
    /// Java `private final EtomoNumber angleOffsetA = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.FIRST.getExtension() + '.' + ProcessName.ALIGN + '.' + ConstTiltalignParam.ANGLE_OFFSET_KEY);`
    angle_offset_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber angleOffsetB = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.SECOND.getExtension() + '.' + ProcessName.ALIGN + '.' + ConstTiltalignParam.ANGLE_OFFSET_KEY);`
    angle_offset_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber axisZShiftA = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.FIRST.getExtension() + '.' + ProcessName.ALIGN + '.' + ConstTiltalignParam.AXIS_Z_SHIFT_KEY);`
    axis_z_shift_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber axisZShiftB = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.SECOND.getExtension() + '.' + ProcessName.ALIGN + '.' + ConstTiltalignParam.AXIS_Z_SHIFT_KEY);`
    axis_z_shift_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleAngleOffsetA = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.FIRST.getExtension() + '.' + ProcessName.SAMPLE + '.' + ConstTiltalignParam.ANGLE_OFFSET_KEY);`
    sample_angle_offset_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleAngleOffsetB = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.SECOND.getExtension() + '.' + ProcessName.SAMPLE + '.' + ConstTiltalignParam.ANGLE_OFFSET_KEY);`
    sample_angle_offset_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleAxisZShiftA = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.FIRST.getExtension() + '.' + ProcessName.SAMPLE + '.' + ConstTiltalignParam.AXIS_Z_SHIFT_KEY);`
    sample_axis_z_shift_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleAxisZShiftB = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.SECOND.getExtension() + '.' + ProcessName.SAMPLE + '.' + ConstTiltalignParam.AXIS_Z_SHIFT_KEY);`
    sample_axis_z_shift_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleXAxisTiltA = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.FIRST.getExtension() + '.' + ProcessName.SAMPLE + '.' + X_AXIS_TILT_KEY);`
    sample_x_axis_tilt_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber sampleXAxisTiltB = new EtomoNumber(EtomoNumber.Type.DOUBLE, AxisID.SECOND.getExtension() + '.' + ProcessName.SAMPLE + '.' + X_AXIS_TILT_KEY);`
    sample_x_axis_tilt_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber fidFileLastModifiedA = new EtomoNumber(EtomoNumber.Type.LONG, AxisID.FIRST.getExtension() + '.' + ProcessName.TRACK + '.' + LAST_MODIFIED);`
    fid_file_last_modified_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber fidFileLastModifiedB = new EtomoNumber(EtomoNumber.Type.LONG, AxisID.SECOND.getExtension() + '.' + ProcessName.TRACK + '.' + LAST_MODIFIED);`
    fid_file_last_modified_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber seedFileLastModifiedA = new EtomoNumber(EtomoNumber.Type.LONG, AxisID.FIRST.getExtension() + '.' + USE_FID_AS_SEED + '.' + LAST_MODIFIED);`
    seed_file_last_modified_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber seedFileLastModifiedB = new EtomoNumber(EtomoNumber.Type.LONG, AxisID.SECOND.getExtension() + '.' + USE_FID_AS_SEED + '.' + LAST_MODIFIED);`
    seed_file_last_modified_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 fixedFiducialsA = new EtomoBoolean2(AxisID.FIRST.getExtension() + "." + FIXED_FIDUCIALS_KEY);`
    fixed_fiducials_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 fixedFiducialsB = new EtomoBoolean2(AxisID.SECOND.getExtension() + "." + FIXED_FIDUCIALS_KEY);`
    fixed_fiducials_b: Mutex<EtomoBoolean2>,
    /// Java `private MatchMode combineMatchMode = null;`
    combine_match_mode: Mutex<Option<MatchMode>>,
    /// Java `private final EtomoState combineScriptsCreated = new EtomoState(DialogType.TOMOGRAM_COMBINATION.getStorableName() + "." + "ScriptsCreated");`
    combine_scripts_created: Mutex<EtomoState>,
    /// Java `private final EtomoBoolean2 seedingDoneA = new EtomoBoolean2(AxisID.FIRST.getExtension() + '.' + "SeedingDone");`
    seeding_done_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 seedingDoneB = new EtomoBoolean2(AxisID.SECOND.getExtension() + '.' + "SeedingDone");`
    seeding_done_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 xcorrBlendmontWasRunA = new EtomoBoolean2("xcorr.blendmont.a.WasRun");`
    xcorr_blendmont_was_run_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 xcorrBlendmontWasRunB = new EtomoBoolean2("xcorr.blendmont.b.WasRun");`
    xcorr_blendmont_was_run_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber imageRotationForAliStackFromBatchruntomo = new EtomoNumber(EtomoNumber.Type.DOUBLE, MetaData.BATCHRUNTOMO_KEY + ".ImageRotationForAliStack");`
    image_rotation_for_ali_stack_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private EtomoBoolean2 sampleFiducialessA = null;`
    sample_fiducialess_a: Mutex<Option<EtomoBoolean2>>,
    /// Java `private EtomoBoolean2 sampleFiducialessB = null;`
    sample_fiducialess_b: Mutex<Option<EtomoBoolean2>>,
    /// Java `private final ApplicationManager manager;`
    manager: Option<&'static ApplicationManager>,
    /// Java `private String firstAxisGroup = null;`
    first_axis_group: Mutex<Option<String>>,
    /// Java `private String secondAxisGroup = null;`
    second_axis_group: Mutex<Option<String>>,
    /// Java `private EtomoNumber tomogramSizeA = new EtomoNumber(EtomoNumber.Type.LONG, A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY);`
    tomogram_size_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeB = new EtomoNumber(EtomoNumber.Type.LONG, B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY);`
    tomogram_size_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 adjustOriginA = new EtomoBoolean2(A_AXIS_KEY + "." + ADJUST_ORIGIN_KEY);`
    adjust_origin_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 adjustOriginB = new EtomoBoolean2(B_AXIS_KEY + "." + ADJUST_ORIGIN_KEY);`
    adjust_origin_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoNumber postProcTrimVolInputNColumns = new EtomoNumber(DialogType.POST_PROCESSING.getStorableName() + ".TrimVol.Input.NColumns");`
    post_proc_trim_vol_input_n_columns: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postProcTrimVolInputNRows = new EtomoNumber(DialogType.POST_PROCESSING.getStorableName() + ".TrimVol.Input.NRows");`
    post_proc_trim_vol_input_n_rows: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postProcTrimVolInputNSections = new EtomoNumber(DialogType.POST_PROCESSING.getStorableName() + ".TrimVol.Input.NSections");`
    post_proc_trim_vol_input_n_sections: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 stackUseLinearInterpolationA = new EtomoBoolean2(STACK_KEY + "." + A_AXIS_KEY + ".UseLinearInterpolation");`
    stack_use_linear_interpolation_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 stackUseLinearInterpolationB = new EtomoBoolean2(STACK_KEY + "." + B_AXIS_KEY + ".UseLinearInterpolation");`
    stack_use_linear_interpolation_b: Mutex<EtomoBoolean2>,
    /// Java `private final StringProperty stackUserSizeToOutputInXandYA = new StringProperty(STACK_KEY + "." + A_AXIS_KEY + ".SizeToOutputInXandY");`
    stack_user_size_to_output_in_x_and_y_a: Mutex<StringProperty>,
    /// Java `private final StringProperty stackUserSizeToOutputInXandYB = new StringProperty(STACK_KEY + "." + B_AXIS_KEY + ".SizeToOutputInXandY");`
    stack_user_size_to_output_in_x_and_y_b: Mutex<StringProperty>,
    /// Java `private final EtomoNumber stackImageRotationA = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + A_AXIS_KEY + ".ImageRotation");`
    stack_image_rotation_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber stackImageRotationB = new EtomoNumber(EtomoNumber.Type.DOUBLE, STACK_KEY + "." + B_AXIS_KEY + ".ImageRotation");`
    stack_image_rotation_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackLightBeadsA = new EtomoNumber("Track.A.LightBeads");`
    track_light_beads_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber trackLightBeadsB = new EtomoNumber("Track.B.LightBeads");`
    track_light_beads_b: Mutex<EtomoNumber>,
    /// Java `private final EtomoBoolean2 stackUsingNewstOrBlend3dFindOutputA = new EtomoBoolean2("Track.A.UsingNewstOrBlend3dFindOutput");`
    stack_using_newst_or_blend_3d_find_output_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 stackUsingNewstOrBlend3dFindOutputB = new EtomoBoolean2("Track.B.UsingNewstOrBlend3dFindOutput");`
    stack_using_newst_or_blend_3d_find_output_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useFixedStackWarningA = new EtomoBoolean2(PRE_KEY + ".A.UseFixedStack.Warning");`
    use_fixed_stack_warning_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useFixedStackWarningB = new EtomoBoolean2(PRE_KEY + ".B.UseFixedStack.Warning");`
    use_fixed_stack_warning_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useRaptorResultWarningA = new EtomoBoolean2(TRACK_KEY + ".A.UseRaptorResult.Warning");`
    use_raptor_result_warning_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useCtfCorrectionWarningA = new EtomoBoolean2(STACK_KEY + ".A.UseCtfCorrection.Warning");`
    use_ctf_correction_warning_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useCtfCorrectionWarningB = new EtomoBoolean2(STACK_KEY + ".B.UseCtfCorrection.Warning");`
    use_ctf_correction_warning_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useErasedStackWarningA = new EtomoBoolean2(STACK_KEY + ".A.UseErasedStack.Warning");`
    use_erased_stack_warning_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useErasedStackWarningB = new EtomoBoolean2(STACK_KEY + ".B.UseErasedStack.Warning");`
    use_erased_stack_warning_b: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useFilteredStackWarningA = new EtomoBoolean2(STACK_KEY + ".A.UseFilteredStack.Warning");`
    use_filtered_stack_warning_a: Mutex<EtomoBoolean2>,
    /// Java `private final EtomoBoolean2 useFilteredStackWarningB = new EtomoBoolean2(STACK_KEY + ".B.UseFilteredStack.Warning");`
    use_filtered_stack_warning_b: Mutex<EtomoBoolean2>,
    /// Java `private final StringProperty genSirtsetupSubareaSizeA = new StringProperty(GEN_KEY + "." + A_AXIS_KEY + ".SirtsetupSubareaSize");`
    gen_sirtsetup_subarea_size_a: Mutex<StringProperty>,
    /// Java `private final StringProperty genSirtsetupSubareaSizeB = new StringProperty(GEN_KEY + "." + B_AXIS_KEY + ".SirtsetupSubareaSize");`
    gen_sirtsetup_subarea_size_b: Mutex<StringProperty>,
    /// Java `private final EtomoNumber genSirtsetupyOffsetOfSubareaA = new EtomoNumber(GEN_KEY + "." + A_AXIS_KEY + ".SirtsetupyOffsetOfSubarea");`
    gen_sirtsetupy_offset_of_subarea_a: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber genSirtsetupyOffsetOfSubareaB = new EtomoNumber(GEN_KEY + "." + B_AXIS_KEY + ".SirtsetupyOffsetOfSubarea");`
    gen_sirtsetupy_offset_of_subarea_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeColumnsA = new EtomoNumber(A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Columns");`
    tomogram_size_columns_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeColumnsB = new EtomoNumber(B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Columns");`
    tomogram_size_columns_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeRowsA = new EtomoNumber(A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Rows");`
    tomogram_size_rows_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeRowsB = new EtomoNumber(B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Rows");`
    tomogram_size_rows_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeSectionsA = new EtomoNumber(A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Sections");`
    tomogram_size_sections_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeSectionsB = new EtomoNumber(B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Sections");`
    tomogram_size_sections_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeColumnsFromBatchruntomoA = new EtomoNumber(MetaData.BATCHRUNTOMO_KEY + "." + A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Columns");`
    tomogram_size_columns_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeColumnsFromBatchruntomoB = new EtomoNumber(MetaData.BATCHRUNTOMO_KEY + "." + B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Columns");`
    tomogram_size_columns_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeRowsFromBatchruntomoA = new EtomoNumber(MetaData.BATCHRUNTOMO_KEY + "." + A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Rows");`
    tomogram_size_rows_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeRowsFromBatchruntomoB = new EtomoNumber(MetaData.BATCHRUNTOMO_KEY + "." + B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Rows");`
    tomogram_size_rows_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeSectionsFromBatchruntomoA = new EtomoNumber(MetaData.BATCHRUNTOMO_KEY + "." + A_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Sections");`
    tomogram_size_sections_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber tomogramSizeSectionsFromBatchruntomoB = new EtomoNumber(MetaData.BATCHRUNTOMO_KEY + "." + B_AXIS_KEY + "." + TOMOGRAM_SIZE_KEY + ".Sections");`
    tomogram_size_sections_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber posSampleTypeA = new EtomoNumber("a.positioning.PosSampleType");`
    pos_sample_type_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber posSampleTypeB = new EtomoNumber("b.positioning.PosSampleType");`
    pos_sample_type_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber stackUseLinearInterpolationFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.Stack.A.UseLinearInterpolation");`
    stack_use_linear_interpolation_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber stackUseLinearInterpolationFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.Stack.B.UseLinearInterpolation");`
    stack_use_linear_interpolation_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber madeZFactorsFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.MadeZFactorsA");`
    made_z_factors_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber madeZFactorsFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.MadeZFactorsB");`
    made_z_factors_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber usedLocalAlignmentsFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.UsedLocalAlignmentsA");`
    used_local_alignments_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber usedLocalAlignmentsFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.UsedLocalAlignmentsB");`
    used_local_alignments_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber axisZShiftFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.a.align.AxisZShift");`
    axis_z_shift_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber axisZShiftFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.b.align.AxisZShift");`
    axis_z_shift_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber angleOffsetFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.a.align.AngleOffset");`
    angle_offset_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber angleOffsetFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.b.align.AngleOffset");`
    angle_offset_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber sampleAngleOffsetFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.a.sample.AngleOffset");`
    sample_angle_offset_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber sampleAngleOffsetFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.b.sample.AngleOffset");`
    sample_angle_offset_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber sampleAxisZShiftFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.a.sample.AxisZShift");`
    sample_axis_z_shift_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber sampleAxisZShiftFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.b.sample.AxisZShift");`
    sample_axis_z_shift_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber sampleXAxisTiltFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.a.sample.XAXISTILT");`
    sample_x_axis_tilt_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber sampleXAxisTiltFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.DOUBLE, "batchruntomo.b.sample.XAXISTILT");`
    sample_x_axis_tilt_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber seedingDoneFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.a.SeedingDone");`
    seeding_done_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber seedingDoneFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.b.SeedingDone");`
    seeding_done_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber trackLightBeadsFromBatchruntomoA = new EtomoNumber("batchruntomo.Track.A.LightBeads");`
    track_light_beads_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber trackLightBeadsFromBatchruntomoB = new EtomoNumber("batchruntomo.Track.B.LightBeads");`
    track_light_beads_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber xcorrBlendmontWasRunFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.xcorr.blendmont.a.WasRun");`
    xcorr_blendmont_was_run_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber xcorrBlendmontWasRunFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.xcorr.blendmont.b.WasRun");`
    xcorr_blendmont_was_run_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber invalidEdgeFunctionsFromBatchruntomoA = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.InvalidEdgeFunctionsA");`
    invalid_edge_functions_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber invalidEdgeFunctionsFromBatchruntomoB = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.InvalidEdgeFunctionsB");`
    invalid_edge_functions_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private StringProperty stackUserSizeToOutputInXandYFromBatchruntomoA = new StringProperty("batchruntomo.Stack.A.SizeToOutputInXandY");`
    stack_user_size_to_output_in_x_and_y_from_batchruntomo_a: Mutex<StringProperty>,
    /// Java `private StringProperty stackUserSizeToOutputInXandYFromBatchruntomoB = new StringProperty("batchruntomo.Stack.B.SizeToOutputInXandY");`
    stack_user_size_to_output_in_x_and_y_from_batchruntomo_b: Mutex<StringProperty>,
    /// Java `private EtomoNumber trimvolFlippedFromBatchruntomo = new EtomoNumber(EtomoNumber.Type.BOOLEAN, "batchruntomo.TrimvolFlipped");`
    trimvol_flipped_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber postProcTrimVolInputNRowsFromBatchruntomo = new EtomoNumber("batchruntomo.TrimVol.Input.NRows");`
    post_proc_trim_vol_input_n_rows_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postProcTrimVolInputNColumnsFromBatchruntomo = new EtomoNumber("batchruntomo.TrimVol.Input.NColumns");`
    post_proc_trim_vol_input_n_columns_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private final EtomoNumber postProcTrimVolInputNSectionsFromBatchruntomo = new EtomoNumber("batchruntomo.TrimVol.Input.NSections");`
    post_proc_trim_vol_input_n_sections_from_batchruntomo: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber posSampleTypeFromBatchruntomoA = new EtomoNumber("batchruntomo.a.positioning.PosSampleType");`
    pos_sample_type_from_batchruntomo_a: Mutex<EtomoNumber>,
    /// Java `private EtomoNumber posSampleTypeFromBatchruntomoB = new EtomoNumber("batchruntomo.b.positioning.PosSampleType");`
    pos_sample_type_from_batchruntomo_b: Mutex<EtomoNumber>,
    /// Java `private boolean newBatchruntomoCombineSettings = false;`
    new_batchruntomo_combine_settings: Mutex<bool>,
}

impl TomogramState {
    /// Java `TomogramState(ApplicationManager)`, together with the field initialisers
    /// Java runs before its body.
    pub fn new(manager: Option<&'static ApplicationManager>) -> TomogramState {
        TomogramState {
            manager,
            trimvol_flipped: Mutex::new(EtomoState::new_with_name("TrimvolFlipped")),
            alt_tomo_preprocess_for_extremes: Mutex::new(EtomoState::new_with_name(
                "AltTomoPreprocessForExtremes",
            )),
            alt_tomo_correct_ctf: Mutex::new(EtomoState::new_with_name("AltTomoCorrectCTF")),
            alt_tomo_erase_fiducials: Mutex::new(EtomoState::new_with_name(
                "AltTomoEraseFiducials",
            )),
            alt_tomo_filter_in_2d: Mutex::new(EtomoState::new_with_name("AltTomoFilterIn2D")),
            alt_tomo_trim_vol: Mutex::new(EtomoState::new_with_name("AltTomoTrimVol")),
            alt_tomo_rootname_to_process: Mutex::new(StringProperty::new_with_key(Some(
                "RootnameToProcess",
            ))),
            alt_tomo_even_and_odd_pairs: Mutex::new(EtomoState::new_with_name(
                "AltTomoEvenAndOddPairs",
            )),
            alt_tomo_axis_to_process: Mutex::new(StringProperty::new_with_key(Some(
                "AltTomoAxisToProcess",
            ))),
            squeezevol_flipped: Mutex::new(EtomoState::new_with_name("SqueezevolFlipped")),
            flatten_flipped: Mutex::new(EtomoState::new_with_name("FlattenFlipped")),
            reduce_filt_vol_flipped: Mutex::new(EtomoState::new_with_name("ReduceFiltVolFlipped")),
            made_z_factors_a: Mutex::new(EtomoState::new_with_name(&format!(
                "MadeZFactors{}",
                A_AXIS_KEY
            ))),
            made_z_factors_b: Mutex::new(EtomoState::new_with_name("MadeZFactorsB")),
            newst_fiducialess_alignment_a: Mutex::new(EtomoState::new_with_name(
                "NewstFiducialessAlignmentA",
            )),
            newst_fiducialess_alignment_b: Mutex::new(EtomoState::new_with_name(
                "NewstFiducialessAlignmentB",
            )),
            used_local_alignments_a: Mutex::new(EtomoState::new_with_name(&format!(
                "UsedLocalAlignments{}",
                A_AXIS_KEY
            ))),
            used_local_alignments_b: Mutex::new(EtomoState::new_with_name(&format!(
                "UsedLocalAlignments{}",
                B_AXIS_KEY
            ))),
            invalid_edge_functions_a: Mutex::new(EtomoState::new_with_name(&format!(
                "InvalidEdgeFunctions{}",
                A_AXIS_KEY
            ))),
            invalid_edge_functions_b: Mutex::new(EtomoState::new_with_name(&format!(
                "InvalidEdgeFunctions{}",
                B_AXIS_KEY
            ))),
            angle_offset_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    ProcessName::ALIGN,
                    TILTALIGN_ANGLE_OFFSET_KEY
                ),
            )),
            angle_offset_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    ProcessName::ALIGN,
                    TILTALIGN_ANGLE_OFFSET_KEY
                ),
            )),
            axis_z_shift_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    ProcessName::ALIGN,
                    TILTALIGN_AXIS_Z_SHIFT_KEY
                ),
            )),
            axis_z_shift_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    ProcessName::ALIGN,
                    TILTALIGN_AXIS_Z_SHIFT_KEY
                ),
            )),
            sample_angle_offset_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    ProcessName::SAMPLE,
                    TILTALIGN_ANGLE_OFFSET_KEY
                ),
            )),
            sample_angle_offset_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    ProcessName::SAMPLE,
                    TILTALIGN_ANGLE_OFFSET_KEY
                ),
            )),
            sample_axis_z_shift_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    ProcessName::SAMPLE,
                    TILTALIGN_AXIS_Z_SHIFT_KEY
                ),
            )),
            sample_axis_z_shift_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    ProcessName::SAMPLE,
                    TILTALIGN_AXIS_Z_SHIFT_KEY
                ),
            )),
            sample_x_axis_tilt_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    ProcessName::SAMPLE,
                    X_AXIS_TILT_KEY
                ),
            )),
            sample_x_axis_tilt_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    ProcessName::SAMPLE,
                    X_AXIS_TILT_KEY
                ),
            )),
            fid_file_last_modified_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Long,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    ProcessName::TRACK,
                    LAST_MODIFIED
                ),
            )),
            fid_file_last_modified_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Long,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    ProcessName::TRACK,
                    LAST_MODIFIED
                ),
            )),
            seed_file_last_modified_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Long,
                &format!(
                    "{}.{}.{}",
                    AxisID::First.get_extension(),
                    USE_FID_AS_SEED,
                    LAST_MODIFIED
                ),
            )),
            seed_file_last_modified_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Long,
                &format!(
                    "{}.{}.{}",
                    AxisID::Second.get_extension(),
                    USE_FID_AS_SEED,
                    LAST_MODIFIED
                ),
            )),
            fixed_fiducials_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}",
                AxisID::First.get_extension(),
                FIXED_FIDUCIALS_KEY
            ))),
            fixed_fiducials_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}",
                AxisID::Second.get_extension(),
                FIXED_FIDUCIALS_KEY
            ))),
            combine_match_mode: Mutex::new(None),
            combine_scripts_created: Mutex::new(EtomoState::new_with_name(&format!(
                "{}.ScriptsCreated",
                DialogType::TomogramCombination.get_storable_name()
            ))),
            seeding_done_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.SeedingDone",
                AxisID::First.get_extension()
            ))),
            seeding_done_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.SeedingDone",
                AxisID::Second.get_extension()
            ))),
            xcorr_blendmont_was_run_a: Mutex::new(EtomoBoolean2::new_with_name(
                "xcorr.blendmont.a.WasRun",
            )),
            xcorr_blendmont_was_run_b: Mutex::new(EtomoBoolean2::new_with_name(
                "xcorr.blendmont.b.WasRun",
            )),
            image_rotation_for_ali_stack_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    &format!("{}.ImageRotationForAliStack", BATCHRUNTOMO_KEY),
                ),
            ),
            sample_fiducialess_a: Mutex::new(None),
            sample_fiducialess_b: Mutex::new(None),
            first_axis_group: Mutex::new(None),
            second_axis_group: Mutex::new(None),
            tomogram_size_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Long,
                &format!("{}.{}", A_AXIS_KEY, TOMOGRAM_SIZE_KEY),
            )),
            tomogram_size_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Long,
                &format!("{}.{}", B_AXIS_KEY, TOMOGRAM_SIZE_KEY),
            )),
            adjust_origin_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}",
                A_AXIS_KEY, ADJUST_ORIGIN_KEY
            ))),
            adjust_origin_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}",
                B_AXIS_KEY, ADJUST_ORIGIN_KEY
            ))),
            post_proc_trim_vol_input_n_columns: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.TrimVol.Input.NColumns",
                DialogType::PostProcessing.get_storable_name()
            ))),
            post_proc_trim_vol_input_n_rows: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.TrimVol.Input.NRows",
                DialogType::PostProcessing.get_storable_name()
            ))),
            post_proc_trim_vol_input_n_sections: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.TrimVol.Input.NSections",
                DialogType::PostProcessing.get_storable_name()
            ))),
            stack_use_linear_interpolation_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.UseLinearInterpolation",
                STACK_KEY, A_AXIS_KEY
            ))),
            stack_use_linear_interpolation_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.{}.UseLinearInterpolation",
                STACK_KEY, B_AXIS_KEY
            ))),
            stack_user_size_to_output_in_x_and_y_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.SizeToOutputInXandY", STACK_KEY, A_AXIS_KEY),
            ))),
            stack_user_size_to_output_in_x_and_y_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!("{}.{}.SizeToOutputInXandY", STACK_KEY, B_AXIS_KEY),
            ))),
            stack_image_rotation_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.ImageRotation", STACK_KEY, A_AXIS_KEY),
            )),
            stack_image_rotation_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                &format!("{}.{}.ImageRotation", STACK_KEY, B_AXIS_KEY),
            )),
            track_light_beads_a: Mutex::new(EtomoNumber::new_with_name("Track.A.LightBeads")),
            track_light_beads_b: Mutex::new(EtomoNumber::new_with_name("Track.B.LightBeads")),
            stack_using_newst_or_blend_3d_find_output_a: Mutex::new(EtomoBoolean2::new_with_name(
                "Track.A.UsingNewstOrBlend3dFindOutput",
            )),
            stack_using_newst_or_blend_3d_find_output_b: Mutex::new(EtomoBoolean2::new_with_name(
                "Track.B.UsingNewstOrBlend3dFindOutput",
            )),
            use_fixed_stack_warning_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.A.UseFixedStack.Warning",
                PRE_KEY
            ))),
            use_fixed_stack_warning_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.B.UseFixedStack.Warning",
                PRE_KEY
            ))),
            use_raptor_result_warning_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.A.UseRaptorResult.Warning",
                TRACK_KEY
            ))),
            use_ctf_correction_warning_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.A.UseCtfCorrection.Warning",
                STACK_KEY
            ))),
            use_ctf_correction_warning_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.B.UseCtfCorrection.Warning",
                STACK_KEY
            ))),
            use_erased_stack_warning_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.A.UseErasedStack.Warning",
                STACK_KEY
            ))),
            use_erased_stack_warning_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.B.UseErasedStack.Warning",
                STACK_KEY
            ))),
            use_filtered_stack_warning_a: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.A.UseFilteredStack.Warning",
                STACK_KEY
            ))),
            use_filtered_stack_warning_b: Mutex::new(EtomoBoolean2::new_with_name(&format!(
                "{}.B.UseFilteredStack.Warning",
                STACK_KEY
            ))),
            gen_sirtsetup_subarea_size_a: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.SirtsetupSubareaSize",
                GEN_KEY, A_AXIS_KEY
            )))),
            gen_sirtsetup_subarea_size_b: Mutex::new(StringProperty::new_with_key(Some(&format!(
                "{}.{}.SirtsetupSubareaSize",
                GEN_KEY, B_AXIS_KEY
            )))),
            gen_sirtsetupy_offset_of_subarea_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.SirtsetupyOffsetOfSubarea",
                GEN_KEY, A_AXIS_KEY
            ))),
            gen_sirtsetupy_offset_of_subarea_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.SirtsetupyOffsetOfSubarea",
                GEN_KEY, B_AXIS_KEY
            ))),
            tomogram_size_columns_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.Columns",
                A_AXIS_KEY, TOMOGRAM_SIZE_KEY
            ))),
            tomogram_size_columns_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.Columns",
                B_AXIS_KEY, TOMOGRAM_SIZE_KEY
            ))),
            tomogram_size_rows_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.Rows",
                A_AXIS_KEY, TOMOGRAM_SIZE_KEY
            ))),
            tomogram_size_rows_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.Rows",
                B_AXIS_KEY, TOMOGRAM_SIZE_KEY
            ))),
            tomogram_size_sections_a: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.Sections",
                A_AXIS_KEY, TOMOGRAM_SIZE_KEY
            ))),
            tomogram_size_sections_b: Mutex::new(EtomoNumber::new_with_name(&format!(
                "{}.{}.Sections",
                B_AXIS_KEY, TOMOGRAM_SIZE_KEY
            ))),
            tomogram_size_columns_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.{}.Columns",
                    BATCHRUNTOMO_KEY, A_AXIS_KEY, TOMOGRAM_SIZE_KEY
                ),
            )),
            tomogram_size_columns_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.{}.Columns",
                    BATCHRUNTOMO_KEY, B_AXIS_KEY, TOMOGRAM_SIZE_KEY
                ),
            )),
            tomogram_size_rows_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.{}.Rows",
                    BATCHRUNTOMO_KEY, A_AXIS_KEY, TOMOGRAM_SIZE_KEY
                ),
            )),
            tomogram_size_rows_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.{}.Rows",
                    BATCHRUNTOMO_KEY, B_AXIS_KEY, TOMOGRAM_SIZE_KEY
                ),
            )),
            tomogram_size_sections_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.{}.Sections",
                    BATCHRUNTOMO_KEY, A_AXIS_KEY, TOMOGRAM_SIZE_KEY
                ),
            )),
            tomogram_size_sections_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_name(
                &format!(
                    "{}.{}.{}.Sections",
                    BATCHRUNTOMO_KEY, B_AXIS_KEY, TOMOGRAM_SIZE_KEY
                ),
            )),
            pos_sample_type_a: Mutex::new(EtomoNumber::new_with_name(
                "a.positioning.PosSampleType",
            )),
            pos_sample_type_b: Mutex::new(EtomoNumber::new_with_name(
                "b.positioning.PosSampleType",
            )),
            stack_use_linear_interpolation_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.Stack.A.UseLinearInterpolation",
                ),
            ),
            stack_use_linear_interpolation_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.Stack.B.UseLinearInterpolation",
                ),
            ),
            made_z_factors_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "batchruntomo.MadeZFactorsA",
            )),
            made_z_factors_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "batchruntomo.MadeZFactorsB",
            )),
            used_local_alignments_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.UsedLocalAlignmentsA",
                ),
            ),
            used_local_alignments_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.UsedLocalAlignmentsB",
                ),
            ),
            axis_z_shift_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "batchruntomo.a.align.AxisZShift",
            )),
            axis_z_shift_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "batchruntomo.b.align.AxisZShift",
            )),
            angle_offset_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "batchruntomo.a.align.AngleOffset",
            )),
            angle_offset_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Double,
                "batchruntomo.b.align.AngleOffset",
            )),
            sample_angle_offset_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "batchruntomo.a.sample.AngleOffset",
                ),
            ),
            sample_angle_offset_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "batchruntomo.b.sample.AngleOffset",
                ),
            ),
            sample_axis_z_shift_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "batchruntomo.a.sample.AxisZShift",
                ),
            ),
            sample_axis_z_shift_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "batchruntomo.b.sample.AxisZShift",
                ),
            ),
            sample_x_axis_tilt_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "batchruntomo.a.sample.XAXISTILT",
                ),
            ),
            sample_x_axis_tilt_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Double,
                    "batchruntomo.b.sample.XAXISTILT",
                ),
            ),
            seeding_done_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "batchruntomo.a.SeedingDone",
            )),
            seeding_done_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "batchruntomo.b.SeedingDone",
            )),
            track_light_beads_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.Track.A.LightBeads",
            )),
            track_light_beads_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.Track.B.LightBeads",
            )),
            xcorr_blendmont_was_run_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.xcorr.blendmont.a.WasRun",
                ),
            ),
            xcorr_blendmont_was_run_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.xcorr.blendmont.b.WasRun",
                ),
            ),
            invalid_edge_functions_from_batchruntomo_a: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.InvalidEdgeFunctionsA",
                ),
            ),
            invalid_edge_functions_from_batchruntomo_b: Mutex::new(
                EtomoNumber::new_with_type_and_name(
                    Type::Boolean,
                    "batchruntomo.InvalidEdgeFunctionsB",
                ),
            ),
            stack_user_size_to_output_in_x_and_y_from_batchruntomo_a: Mutex::new(
                StringProperty::new_with_key(Some("batchruntomo.Stack.A.SizeToOutputInXandY")),
            ),
            stack_user_size_to_output_in_x_and_y_from_batchruntomo_b: Mutex::new(
                StringProperty::new_with_key(Some("batchruntomo.Stack.B.SizeToOutputInXandY")),
            ),
            trimvol_flipped_from_batchruntomo: Mutex::new(EtomoNumber::new_with_type_and_name(
                Type::Boolean,
                "batchruntomo.TrimvolFlipped",
            )),
            post_proc_trim_vol_input_n_rows_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_name("batchruntomo.TrimVol.Input.NRows"),
            ),
            post_proc_trim_vol_input_n_columns_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_name("batchruntomo.TrimVol.Input.NColumns"),
            ),
            post_proc_trim_vol_input_n_sections_from_batchruntomo: Mutex::new(
                EtomoNumber::new_with_name("batchruntomo.TrimVol.Input.NSections"),
            ),
            pos_sample_type_from_batchruntomo_a: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.a.positioning.PosSampleType",
            )),
            pos_sample_type_from_batchruntomo_b: Mutex::new(EtomoNumber::new_with_name(
                "batchruntomo.b.positioning.PosSampleType",
            )),
            new_batchruntomo_combine_settings: Mutex::new(false),
        }
    }

    /// Java private `setAxisPrepends`.  Get the axis keys from meta data.  If dual axis,
    /// create a first axis group string that ends in ".".  For single axis, the first
    /// axis key is an empty string and the second axis key is null.  Java string
    /// concatenation turns a null first key into "null.".
    fn set_axis_prepends(&self, meta_data: &dyn ConstMetaData) {
        let mut first_axis_group = meta_data.get_first_axis_prepend();
        let mut second_axis_group = meta_data.get_second_axis_prepend();
        if let Some(second) = &second_axis_group {
            first_axis_group = Some(format!(
                "{}.",
                first_axis_group.as_deref().unwrap_or("null")
            ));
            second_axis_group = Some(format!("{}.", second));
        }
        *self.first_axis_group.lock().unwrap() = first_axis_group;
        *self.second_axis_group.lock().unwrap() = second_axis_group;
    }

    /// Java `initialize`.
    pub fn initialize(&self) {
        self.trimvol_flipped
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_preprocess_for_extremes
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_correct_ctf
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_erase_fiducials
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_filter_in_2d
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_trim_vol
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .set_number(Some(&etomo_state::NO_RESULT_VALUE.to_string()));
        self.alt_tomo_even_and_odd_pairs
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.alt_tomo_axis_to_process
            .lock()
            .unwrap()
            .set_number(Some(&etomo_state::NO_RESULT_VALUE.to_string()));
        self.squeezevol_flipped
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.reduce_filt_vol_flipped
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.flatten_flipped
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.made_z_factors_a
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.made_z_factors_b
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.newst_fiducialess_alignment_a
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.newst_fiducialess_alignment_b
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.used_local_alignments_a
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.used_local_alignments_b
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.invalid_edge_functions_a
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.invalid_edge_functions_b
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.combine_scripts_created
            .lock()
            .unwrap()
            .set_int(etomo_state::NO_RESULT_VALUE);
        self.adjust_origin_a.lock().unwrap().set_boolean(true);
        self.adjust_origin_b.lock().unwrap().set_boolean(true);
    }

    /// Java `resetBatchruntomoCombineSettings`.
    pub fn reset_batchruntomo_combine_settings(&self) {
        *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
    }

    /// Java `isNewBatchruntomoCombineSettings`.
    pub fn is_new_batchruntomo_combine_settings(&self) -> bool {
        *self.new_batchruntomo_combine_settings.lock().unwrap()
    }

    /// Java `moveBatchruntomoSettings`.
    pub fn move_batchruntomo_settings(&self) {
        if !self
            .image_rotation_for_ali_stack_from_batchruntomo
            .lock()
            .unwrap()
            .is_null()
        {
            let value = self
                .image_rotation_for_ali_stack_from_batchruntomo
                .lock()
                .unwrap()
                .clone();
            self.set_stack_image_rotation(AxisID::First, Some(&value.base));
            self.set_stack_image_rotation(AxisID::Second, Some(&value.base));
            self.image_rotation_for_ali_stack_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .tomogram_size_columns_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
            {
                let value = self
                    .tomogram_size_columns_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.tomogram_size_columns_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.tomogram_size_columns_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .tomogram_size_columns_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
            {
                let value = self
                    .tomogram_size_columns_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.tomogram_size_columns_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.tomogram_size_columns_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .tomogram_size_rows_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
            {
                let value = self
                    .tomogram_size_rows_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.tomogram_size_rows_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.tomogram_size_rows_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .tomogram_size_rows_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
            {
                let value = self
                    .tomogram_size_rows_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.tomogram_size_rows_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.tomogram_size_rows_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .tomogram_size_sections_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
            {
                let value = self
                    .tomogram_size_sections_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.tomogram_size_sections_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.tomogram_size_sections_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .tomogram_size_sections_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            *self.new_batchruntomo_combine_settings.lock().unwrap() = true;
            {
                let value = self
                    .tomogram_size_sections_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.tomogram_size_sections_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.tomogram_size_sections_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .stack_use_linear_interpolation_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .stack_use_linear_interpolation_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.stack_use_linear_interpolation_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.stack_use_linear_interpolation_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .stack_use_linear_interpolation_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .stack_use_linear_interpolation_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.stack_use_linear_interpolation_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.stack_use_linear_interpolation_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .made_z_factors_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .made_z_factors_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.made_z_factors_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.made_z_factors_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .made_z_factors_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .made_z_factors_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.made_z_factors_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.made_z_factors_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .used_local_alignments_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .used_local_alignments_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.used_local_alignments_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.used_local_alignments_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .used_local_alignments_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .used_local_alignments_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.used_local_alignments_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.used_local_alignments_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .axis_z_shift_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.axis_z_shift_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.axis_z_shift_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .axis_z_shift_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.axis_z_shift_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.axis_z_shift_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .angle_offset_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.angle_offset_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.angle_offset_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .angle_offset_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.angle_offset_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.angle_offset_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .sample_angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .sample_angle_offset_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.sample_angle_offset_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.sample_angle_offset_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .sample_angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .sample_angle_offset_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.sample_angle_offset_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.sample_angle_offset_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .sample_axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .sample_axis_z_shift_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.sample_axis_z_shift_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.sample_axis_z_shift_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .sample_axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .sample_axis_z_shift_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.sample_axis_z_shift_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.sample_axis_z_shift_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .sample_x_axis_tilt_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .sample_x_axis_tilt_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.sample_x_axis_tilt_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.sample_x_axis_tilt_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .sample_x_axis_tilt_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .sample_x_axis_tilt_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.sample_x_axis_tilt_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.sample_x_axis_tilt_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .seeding_done_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .seeding_done_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.seeding_done_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.seeding_done_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .seeding_done_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .seeding_done_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.seeding_done_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.seeding_done_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .track_light_beads_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .track_light_beads_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.track_light_beads_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.track_light_beads_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .track_light_beads_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .track_light_beads_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.track_light_beads_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.track_light_beads_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .xcorr_blendmont_was_run_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .xcorr_blendmont_was_run_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.xcorr_blendmont_was_run_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.xcorr_blendmont_was_run_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .xcorr_blendmont_was_run_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .xcorr_blendmont_was_run_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.xcorr_blendmont_was_run_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.xcorr_blendmont_was_run_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .invalid_edge_functions_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .invalid_edge_functions_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.invalid_edge_functions_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.invalid_edge_functions_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .invalid_edge_functions_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .invalid_edge_functions_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.invalid_edge_functions_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.invalid_edge_functions_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .stack_user_size_to_output_in_x_and_y_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_empty()
        {
            {
                let value = self
                    .stack_user_size_to_output_in_x_and_y_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .to_string();
                self.stack_user_size_to_output_in_x_and_y_a
                    .lock()
                    .unwrap()
                    .set(Some(&value));
            };
            self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .stack_user_size_to_output_in_x_and_y_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_empty()
        {
            {
                let value = self
                    .stack_user_size_to_output_in_x_and_y_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .to_string();
                self.stack_user_size_to_output_in_x_and_y_b
                    .lock()
                    .unwrap()
                    .set(Some(&value));
            };
            self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .trimvol_flipped_from_batchruntomo
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .trimvol_flipped_from_batchruntomo
                    .lock()
                    .unwrap()
                    .clone();
                self.trimvol_flipped
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.trimvol_flipped_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .post_proc_trim_vol_input_n_rows_from_batchruntomo
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .post_proc_trim_vol_input_n_rows_from_batchruntomo
                    .lock()
                    .unwrap()
                    .clone();
                self.post_proc_trim_vol_input_n_rows
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.post_proc_trim_vol_input_n_rows_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .post_proc_trim_vol_input_n_columns_from_batchruntomo
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .post_proc_trim_vol_input_n_columns_from_batchruntomo
                    .lock()
                    .unwrap()
                    .clone();
                self.post_proc_trim_vol_input_n_columns
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.post_proc_trim_vol_input_n_columns_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .post_proc_trim_vol_input_n_sections_from_batchruntomo
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .post_proc_trim_vol_input_n_sections_from_batchruntomo
                    .lock()
                    .unwrap()
                    .clone();
                self.post_proc_trim_vol_input_n_sections
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.post_proc_trim_vol_input_n_sections_from_batchruntomo
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .pos_sample_type_from_batchruntomo_a
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .pos_sample_type_from_batchruntomo_a
                    .lock()
                    .unwrap()
                    .clone();
                self.pos_sample_type_a
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.pos_sample_type_from_batchruntomo_a
                .lock()
                .unwrap()
                .reset();
        }
        if !self
            .pos_sample_type_from_batchruntomo_b
            .lock()
            .unwrap()
            .is_null()
        {
            {
                let value = self
                    .pos_sample_type_from_batchruntomo_b
                    .lock()
                    .unwrap()
                    .clone();
                self.pos_sample_type_b
                    .lock()
                    .unwrap()
                    .set_const_etomo_number(Some(&value.base));
            };
            self.pos_sample_type_from_batchruntomo_b
                .lock()
                .unwrap()
                .reset();
        }
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        // `super.store(props, prepend)`: `BaseState.store` only runs `createPrepend(prepend)`
        // and discards the result.
        let prepend = self.create_prepend(prepend);
        let group = format!("{}.", prepend);
        let _ = &group;
        self.trimvol_flipped
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_preprocess_for_extremes
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_correct_ctf
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_erase_fiducials
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_filter_in_2d
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_trim_vol
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.alt_tomo_even_and_odd_pairs
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.alt_tomo_axis_to_process
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.squeezevol_flipped
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.reduce_filt_vol_flipped
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.flatten_flipped
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.made_z_factors_a
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.made_z_factors_b
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.newst_fiducialess_alignment_a
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.newst_fiducialess_alignment_b
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.used_local_alignments_a
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.used_local_alignments_b
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.invalid_edge_functions_a
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.invalid_edge_functions_b
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        self.angle_offset_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.angle_offset_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.axis_z_shift_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.axis_z_shift_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_angle_offset_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_angle_offset_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_axis_z_shift_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_axis_z_shift_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_x_axis_tilt_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.sample_x_axis_tilt_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fid_file_last_modified_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fid_file_last_modified_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.seed_file_last_modified_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.seed_file_last_modified_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fixed_fiducials_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.fixed_fiducials_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.seeding_done_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.seeding_done_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.adjust_origin_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.adjust_origin_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_proc_trim_vol_input_n_columns
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_proc_trim_vol_input_n_rows
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.post_proc_trim_vol_input_n_sections
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_use_linear_interpolation_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_use_linear_interpolation_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_user_size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_user_size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.stack_image_rotation_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_image_rotation_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_light_beads_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.track_light_beads_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_using_newst_or_blend_3d_find_output_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.stack_using_newst_or_blend_3d_find_output_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_fixed_stack_warning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_fixed_stack_warning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_raptor_result_warning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_ctf_correction_warning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_ctf_correction_warning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_erased_stack_warning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_erased_stack_warning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_filtered_stack_warning_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.use_filtered_stack_warning_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_sirtsetup_subarea_size_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_sirtsetup_subarea_size_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(&prepend));
        self.gen_sirtsetupy_offset_of_subarea_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.gen_sirtsetupy_offset_of_subarea_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomogram_size_columns_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomogram_size_columns_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomogram_size_rows_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomogram_size_rows_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomogram_size_sections_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.tomogram_size_sections_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.xcorr_blendmont_was_run_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.xcorr_blendmont_was_run_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.pos_sample_type_a
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        self.pos_sample_type_b
            .lock()
            .unwrap()
            .store_with_prepend(props, Some(&prepend));
        // backwards compatibility
        props.remove(COMBINE_MATCH_MODE_BACK_KEY);
        if self.combine_match_mode.lock().unwrap().is_none() {
            props.remove(&format!("{}.{}", prepend, COMBINE_MATCH_MODE_KEY.as_str()));
        } else {
            let combine_match_mode = self.combine_match_mode.lock().unwrap().unwrap();
            props.insert(
                format!("{}.{}", prepend, COMBINE_MATCH_MODE_KEY.as_str()),
                combine_match_mode.to_string().to_string(),
            );
        }
        // backwards compatibility
        props.remove(COMBINE_SCRIPTS_CREATED_BACK_KEY);
        self.combine_scripts_created
            .lock()
            .unwrap()
            .store_with_prepend(props, &prepend);
        let first_axis_group = self.first_axis_group.lock().unwrap().clone();
        EtomoBoolean2::store_instance(
            self.sample_fiducialess_a.lock().unwrap().as_ref(),
            props,
            Some(&prepend),
            &format!(
                "{}{}",
                first_axis_group.as_deref().unwrap_or("null"),
                SAMPLE_FIDUCIALESS_KEY.as_str()
            ),
        );
        if self.second_axis_group.lock().unwrap().is_some() {
            let second_axis_group = self.second_axis_group.lock().unwrap().clone();
            EtomoBoolean2::store_instance(
                self.sample_fiducialess_b.lock().unwrap().as_ref(),
                props,
                Some(&prepend),
                &format!(
                    "{}{}",
                    second_axis_group.as_deref().unwrap_or("null"),
                    SAMPLE_FIDUCIALESS_KEY.as_str()
                ),
            );
        }
        self.image_rotation_for_ali_stack_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.tomogram_size_columns_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.tomogram_size_columns_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.tomogram_size_rows_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.tomogram_size_rows_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.tomogram_size_sections_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.tomogram_size_sections_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.stack_use_linear_interpolation_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.stack_use_linear_interpolation_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.made_z_factors_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.made_z_factors_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.used_local_alignments_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.used_local_alignments_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.sample_angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.sample_angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.sample_axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.sample_axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.sample_x_axis_tilt_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.sample_x_axis_tilt_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.seeding_done_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.seeding_done_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.track_light_beads_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.track_light_beads_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.xcorr_blendmont_was_run_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.xcorr_blendmont_was_run_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.invalid_edge_functions_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.invalid_edge_functions_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
        self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(Some(props));
        self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(Some(props));
        self.trimvol_flipped_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_proc_trim_vol_input_n_rows_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_proc_trim_vol_input_n_columns_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.post_proc_trim_vol_input_n_sections_from_batchruntomo
            .lock()
            .unwrap()
            .store(props);
        self.pos_sample_type_from_batchruntomo_a
            .lock()
            .unwrap()
            .store(props);
        self.pos_sample_type_from_batchruntomo_b
            .lock()
            .unwrap()
            .store(props);
    }

    // Java `createPrepend` implements the abstract `BaseState.createPrepend`; it is in
    // the `BaseState` impl below.

    /// Java `load(Properties)`.
    pub fn load(&self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &BTreeMap<String, String>, prepend: &str) {
        // `super.load(props, prepend)`: `BaseState.load` only runs `createPrepend(prepend)` and
        // discards the result.
        // `StringProperty.load` may remove a backward-compatible key from the Java
        // `Properties`; the string properties load from a copy of the read-only map.  None of
        // this class's string properties declares such a key.
        let mut props_copy = props.clone();
        self.trimvol_flipped.lock().unwrap().reset();
        self.alt_tomo_preprocess_for_extremes
            .lock()
            .unwrap()
            .reset();
        self.alt_tomo_correct_ctf.lock().unwrap().reset();
        self.alt_tomo_erase_fiducials.lock().unwrap().reset();
        self.alt_tomo_filter_in_2d.lock().unwrap().reset();
        self.alt_tomo_trim_vol.lock().unwrap().reset();
        self.alt_tomo_rootname_to_process.lock().unwrap().reset();
        self.alt_tomo_even_and_odd_pairs.lock().unwrap().reset();
        self.alt_tomo_axis_to_process.lock().unwrap().reset();
        self.squeezevol_flipped.lock().unwrap().reset();
        self.reduce_filt_vol_flipped.lock().unwrap().reset();
        self.flatten_flipped.lock().unwrap().reset();
        self.made_z_factors_a.lock().unwrap().reset();
        self.made_z_factors_b.lock().unwrap().reset();
        self.newst_fiducialess_alignment_a.lock().unwrap().reset();
        self.newst_fiducialess_alignment_b.lock().unwrap().reset();
        self.used_local_alignments_a.lock().unwrap().reset();
        self.used_local_alignments_b.lock().unwrap().reset();
        self.invalid_edge_functions_a.lock().unwrap().reset();
        self.invalid_edge_functions_b.lock().unwrap().reset();
        self.axis_z_shift_a.lock().unwrap().reset();
        self.axis_z_shift_b.lock().unwrap().reset();
        self.angle_offset_a.lock().unwrap().reset();
        self.angle_offset_b.lock().unwrap().reset();
        self.sample_axis_z_shift_a.lock().unwrap().reset();
        self.sample_axis_z_shift_b.lock().unwrap().reset();
        self.sample_angle_offset_a.lock().unwrap().reset();
        self.sample_angle_offset_b.lock().unwrap().reset();
        self.sample_x_axis_tilt_a.lock().unwrap().reset();
        self.sample_x_axis_tilt_b.lock().unwrap().reset();
        self.fid_file_last_modified_a.lock().unwrap().reset();
        self.fid_file_last_modified_b.lock().unwrap().reset();
        self.seed_file_last_modified_a.lock().unwrap().reset();
        self.seed_file_last_modified_b.lock().unwrap().reset();
        self.fixed_fiducials_a.lock().unwrap().reset();
        self.fixed_fiducials_b.lock().unwrap().reset();
        *self.combine_match_mode.lock().unwrap() = None;
        self.combine_scripts_created.lock().unwrap().reset();
        *self.sample_fiducialess_a.lock().unwrap() = None;
        *self.sample_fiducialess_b.lock().unwrap() = None;
        self.seeding_done_a.lock().unwrap().reset();
        self.seeding_done_b.lock().unwrap().reset();
        self.tomogram_size_a.lock().unwrap().reset();
        self.tomogram_size_b.lock().unwrap().reset();
        self.adjust_origin_a.lock().unwrap().reset();
        self.adjust_origin_b.lock().unwrap().reset();
        self.post_proc_trim_vol_input_n_columns
            .lock()
            .unwrap()
            .reset();
        self.post_proc_trim_vol_input_n_rows.lock().unwrap().reset();
        self.post_proc_trim_vol_input_n_sections
            .lock()
            .unwrap()
            .reset();
        self.stack_use_linear_interpolation_a
            .lock()
            .unwrap()
            .reset();
        self.stack_use_linear_interpolation_b
            .lock()
            .unwrap()
            .reset();
        self.stack_user_size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .reset();
        self.stack_user_size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .reset();
        self.stack_image_rotation_a.lock().unwrap().reset();
        self.stack_image_rotation_b.lock().unwrap().reset();
        self.track_light_beads_a.lock().unwrap().reset();
        self.track_light_beads_b.lock().unwrap().reset();
        self.stack_using_newst_or_blend_3d_find_output_a
            .lock()
            .unwrap()
            .reset();
        self.stack_using_newst_or_blend_3d_find_output_b
            .lock()
            .unwrap()
            .reset();
        self.use_fixed_stack_warning_a.lock().unwrap().reset();
        self.use_fixed_stack_warning_b.lock().unwrap().reset();
        self.use_raptor_result_warning_a.lock().unwrap().reset();
        self.use_ctf_correction_warning_a.lock().unwrap().reset();
        self.use_ctf_correction_warning_b.lock().unwrap().reset();
        self.use_erased_stack_warning_a.lock().unwrap().reset();
        self.use_erased_stack_warning_b.lock().unwrap().reset();
        self.use_filtered_stack_warning_a.lock().unwrap().reset();
        self.use_filtered_stack_warning_b.lock().unwrap().reset();
        self.gen_sirtsetup_subarea_size_a.lock().unwrap().reset();
        self.gen_sirtsetup_subarea_size_b.lock().unwrap().reset();
        self.gen_sirtsetupy_offset_of_subarea_a
            .lock()
            .unwrap()
            .reset();
        self.gen_sirtsetupy_offset_of_subarea_b
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_columns_a.lock().unwrap().reset();
        self.tomogram_size_columns_b.lock().unwrap().reset();
        self.tomogram_size_rows_a.lock().unwrap().reset();
        self.tomogram_size_rows_b.lock().unwrap().reset();
        self.tomogram_size_sections_a.lock().unwrap().reset();
        self.tomogram_size_sections_b.lock().unwrap().reset();
        self.xcorr_blendmont_was_run_a.lock().unwrap().reset();
        self.xcorr_blendmont_was_run_b.lock().unwrap().reset();
        self.pos_sample_type_a.lock().unwrap().reset();
        self.pos_sample_type_b.lock().unwrap().reset();
        self.image_rotation_for_ali_stack_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_columns_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_columns_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_rows_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_rows_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_sections_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.tomogram_size_sections_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.stack_use_linear_interpolation_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.stack_use_linear_interpolation_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.made_z_factors_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.made_z_factors_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.used_local_alignments_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.used_local_alignments_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.sample_angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.sample_angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.sample_axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.sample_axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.sample_x_axis_tilt_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.sample_x_axis_tilt_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.seeding_done_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.seeding_done_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.track_light_beads_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.track_light_beads_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.xcorr_blendmont_was_run_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.xcorr_blendmont_was_run_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.invalid_edge_functions_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.invalid_edge_functions_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        self.trimvol_flipped_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_proc_trim_vol_input_n_rows_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_proc_trim_vol_input_n_columns_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.post_proc_trim_vol_input_n_sections_from_batchruntomo
            .lock()
            .unwrap()
            .reset();
        self.pos_sample_type_from_batchruntomo_a
            .lock()
            .unwrap()
            .reset();
        self.pos_sample_type_from_batchruntomo_b
            .lock()
            .unwrap()
            .reset();
        // load
        let prepend = self.create_prepend(prepend);
        let group = format!("{}.", prepend);
        let _ = &group;
        // Upstream bug fixed in translation (TomogramState.java:751): without a manager the
        // source throws `NullPointerException` here; the axis groups are now left as they were.
        if let Some(manager) = self.manager {
            self.set_axis_prepends(manager.get_meta_data());
        }
        self.trimvol_flipped
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_preprocess_for_extremes
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_correct_ctf
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_erase_fiducials
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_filter_in_2d
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_trim_vol
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.alt_tomo_even_and_odd_pairs
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.alt_tomo_axis_to_process
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.squeezevol_flipped
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.reduce_filt_vol_flipped
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.flatten_flipped
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        if self.flatten_flipped.lock().unwrap().is_null() {
            self.flatten_flipped
                .lock()
                .unwrap()
                .set_int(etomo_state::NO_RESULT_VALUE);
        }
        self.made_z_factors_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.made_z_factors_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.newst_fiducialess_alignment_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.newst_fiducialess_alignment_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.used_local_alignments_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.used_local_alignments_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.invalid_edge_functions_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.invalid_edge_functions_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.angle_offset_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.angle_offset_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.axis_z_shift_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.axis_z_shift_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_angle_offset_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_angle_offset_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_axis_z_shift_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_axis_z_shift_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_x_axis_tilt_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.sample_x_axis_tilt_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fid_file_last_modified_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fid_file_last_modified_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.seed_file_last_modified_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.seed_file_last_modified_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fixed_fiducials_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.fixed_fiducials_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.seeding_done_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.seeding_done_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.adjust_origin_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.adjust_origin_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_proc_trim_vol_input_n_columns
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_proc_trim_vol_input_n_rows
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.post_proc_trim_vol_input_n_sections
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_use_linear_interpolation_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_use_linear_interpolation_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_user_size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_user_size_to_output_in_x_and_y_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.stack_image_rotation_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_image_rotation_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_light_beads_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.track_light_beads_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_using_newst_or_blend_3d_find_output_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.stack_using_newst_or_blend_3d_find_output_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_fixed_stack_warning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_fixed_stack_warning_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_raptor_result_warning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_ctf_correction_warning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_ctf_correction_warning_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_erased_stack_warning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_erased_stack_warning_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_filtered_stack_warning_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.use_filtered_stack_warning_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_sirtsetup_subarea_size_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_sirtsetup_subarea_size_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(&prepend));
        self.gen_sirtsetupy_offset_of_subarea_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.gen_sirtsetupy_offset_of_subarea_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_columns_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_columns_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_rows_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_rows_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_sections_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.tomogram_size_sections_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.xcorr_blendmont_was_run_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.xcorr_blendmont_was_run_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.pos_sample_type_a
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        self.pos_sample_type_b
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        *self.combine_match_mode.lock().unwrap() = MatchMode::get_instance_string(
            props
                .get(&format!("{}.{}", prepend, COMBINE_MATCH_MODE_KEY.as_str()))
                .map(|s| s.as_str()),
        );
        if self.combine_match_mode.lock().unwrap().is_none() {
            let back_combine_match_mode = props.get(COMBINE_MATCH_MODE_BACK_KEY).cloned();
            if let Some(back_combine_match_mode) = back_combine_match_mode {
                *self.combine_match_mode.lock().unwrap() =
                    Some(MatchMode::get_instance_match_b_to_a(
                        back_combine_match_mode.eq_ignore_ascii_case("true"),
                    ));
            }
        }
        self.combine_scripts_created
            .lock()
            .unwrap()
            .load_with_prepend(props, Some(&prepend));
        if self.combine_scripts_created.lock().unwrap().is_null() {
            // backwards compatibility
            let back_combine_scripts_created = props.get(COMBINE_SCRIPTS_CREATED_BACK_KEY).cloned();
            if let Some(back_combine_scripts_created) = back_combine_scripts_created {
                self.combine_scripts_created
                    .lock()
                    .unwrap()
                    .set_string(Some(&back_combine_scripts_created));
            }
        }
        let first_axis_group = self.first_axis_group.lock().unwrap().clone();
        let current = self.sample_fiducialess_a.lock().unwrap().take();
        *self.sample_fiducialess_a.lock().unwrap() = EtomoBoolean2::get_instance_from_props(
            current,
            &format!(
                "{}{}",
                first_axis_group.as_deref().unwrap_or("null"),
                SAMPLE_FIDUCIALESS_KEY.as_str()
            ),
            props,
            Some(&prepend),
        );
        if self.second_axis_group.lock().unwrap().is_some() {
            let second_axis_group = self.second_axis_group.lock().unwrap().clone();
            let current = self.sample_fiducialess_b.lock().unwrap().take();
            *self.sample_fiducialess_b.lock().unwrap() = EtomoBoolean2::get_instance_from_props(
                current,
                &format!(
                    "{}{}",
                    second_axis_group.as_deref().unwrap_or("null"),
                    SAMPLE_FIDUCIALESS_KEY.as_str()
                ),
                props,
                Some(&prepend),
            );
        }
        self.image_rotation_for_ali_stack_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.tomogram_size_columns_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.tomogram_size_columns_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.tomogram_size_rows_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.tomogram_size_rows_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.tomogram_size_sections_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.tomogram_size_sections_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.stack_use_linear_interpolation_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.stack_use_linear_interpolation_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.made_z_factors_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.made_z_factors_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.used_local_alignments_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.used_local_alignments_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.sample_angle_offset_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.sample_angle_offset_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.sample_axis_z_shift_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.sample_axis_z_shift_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.sample_x_axis_tilt_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.sample_x_axis_tilt_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.seeding_done_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.seeding_done_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.track_light_beads_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.track_light_beads_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.xcorr_blendmont_was_run_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.xcorr_blendmont_was_run_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.invalid_edge_functions_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.invalid_edge_functions_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
        self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.stack_user_size_to_output_in_x_and_y_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(Some(&mut props_copy));
        self.trimvol_flipped_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_proc_trim_vol_input_n_rows_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_proc_trim_vol_input_n_columns_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.post_proc_trim_vol_input_n_sections_from_batchruntomo
            .lock()
            .unwrap()
            .load(props);
        self.pos_sample_type_from_batchruntomo_a
            .lock()
            .unwrap()
            .load(props);
        self.pos_sample_type_from_batchruntomo_b
            .lock()
            .unwrap()
            .load(props);
    }

    /// Java `setCombineScriptsCreated`.
    pub fn set_combine_scripts_created(&self, combine_scripts_created: bool) {
        self.combine_scripts_created
            .lock()
            .unwrap()
            .set_boolean(combine_scripts_created);
    }

    /// Java `resetCombineScriptsCreated`.
    pub fn reset_combine_scripts_created(&self) {
        self.combine_scripts_created.lock().unwrap().reset();
    }

    /// Java `getCombineScriptsCreated`.
    pub fn get_combine_scripts_created(&self) -> EtomoState {
        self.combine_scripts_created.lock().unwrap().clone()
    }

    /// Java `isCombineScriptsCreated`.
    pub fn is_combine_scripts_created(&self) -> bool {
        if !self.combine_scripts_created.lock().unwrap().is_result_set() {
            return false;
        }
        self.combine_scripts_created.lock().unwrap().is()
    }

    /// Java `setCombineMatchMode`.
    pub fn set_combine_match_mode(&self, combine_match_mode: Option<MatchMode>) {
        *self.combine_match_mode.lock().unwrap() = combine_match_mode;
    }

    /// Java `getCombineMatchMode`.
    pub fn get_combine_match_mode(&self) -> Option<MatchMode> {
        *self.combine_match_mode.lock().unwrap()
    }

    /// Java `setTrimvolFlipped`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_trimvol_flipped(&self, trimvol_flipped: bool) -> EtomoState {
        let mut field = self.trimvol_flipped.lock().unwrap();
        field.set_boolean(trimvol_flipped);
        field.clone()
    }

    /// Java `setAltTomoPreprocessForExtremes`.
    pub fn set_alt_tomo_preprocess_for_extremes(&self, input: bool) {
        self.alt_tomo_preprocess_for_extremes
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setAltTomoCorrectCTF`.
    pub fn set_alt_tomo_correct_ctf(&self, input: bool) {
        self.alt_tomo_correct_ctf.lock().unwrap().set_boolean(input);
    }

    /// Java `setAltTomoEraseFiducials`.
    pub fn set_alt_tomo_erase_fiducials(&self, input: bool) {
        self.alt_tomo_erase_fiducials
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setAltTomoFilterIn2D`.
    pub fn set_alt_tomo_filter_in_2d(&self, input: bool) {
        self.alt_tomo_filter_in_2d
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setAltTomoTrimVolChecked`.
    pub fn set_alt_tomo_trim_vol_checked(&self, input: bool) {
        self.alt_tomo_trim_vol.lock().unwrap().set_boolean(input);
    }

    /// Java `setAltTomoRootnameToProcess`.
    pub fn set_alt_tomo_rootname_to_process(&self, input: Option<&str>) {
        self.alt_tomo_rootname_to_process.lock().unwrap().set(input);
    }

    /// Java `setAltTomoEvenAndOddPairs`.
    pub fn set_alt_tomo_even_and_odd_pairs(&self, input: bool) {
        self.alt_tomo_even_and_odd_pairs
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `setAltTomoAxisToProcess`.
    pub fn set_alt_tomo_axis_to_process(&self, input: Option<&str>) {
        self.alt_tomo_axis_to_process.lock().unwrap().set(input);
    }

    /// Java `setSqueezevolFlipped`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_squeezevol_flipped(&self, squeezevol_flipped: bool) -> EtomoState {
        let mut field = self.squeezevol_flipped.lock().unwrap();
        field.set_boolean(squeezevol_flipped);
        field.clone()
    }

    /// Java `setReduceFiltVolFlipped`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_reduce_filt_vol_flipped(&self, is_flipped: bool) -> EtomoState {
        let mut field = self.reduce_filt_vol_flipped.lock().unwrap();
        field.set_boolean(is_flipped);
        field.clone()
    }

    /// Java `setFlattenFlipped`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_flatten_flipped(&self, is_flipped: bool) -> EtomoState {
        let mut field = self.flatten_flipped.lock().unwrap();
        field.set_boolean(is_flipped);
        field.clone()
    }

    /// Java `setTomogramSizeColumns`.
    pub fn set_tomogram_size_columns(&self, axis_id: AxisID, input: i32) {
        if axis_id == AxisID::Second {
            self.tomogram_size_columns_b.lock().unwrap().set_int(input);
        } else {
            self.tomogram_size_columns_a.lock().unwrap().set_int(input);
        }
    }

    /// Java `setTomogramSizeRows`.
    pub fn set_tomogram_size_rows(&self, axis_id: AxisID, input: i32) {
        if axis_id == AxisID::Second {
            self.tomogram_size_rows_b.lock().unwrap().set_int(input);
        } else {
            self.tomogram_size_rows_a.lock().unwrap().set_int(input);
        }
    }

    /// Java `setTomogramSizeSections`.
    pub fn set_tomogram_size_sections(&self, axis_id: AxisID, input: i32) {
        if axis_id == AxisID::Second {
            self.tomogram_size_sections_b.lock().unwrap().set_int(input);
        } else {
            self.tomogram_size_sections_a.lock().unwrap().set_int(input);
        }
    }

    /// Java `isAdjustOrigin`.
    pub fn is_adjust_origin(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.adjust_origin_b.lock().unwrap().is();
        }
        self.adjust_origin_a.lock().unwrap().is()
    }

    /// Java `setAdjustOrigin`.
    pub fn set_adjust_origin(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.adjust_origin_b.lock().unwrap().set_boolean(input);
        } else {
            self.adjust_origin_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setPostProcTrimVolInputNColumns`.
    pub fn set_post_proc_trim_vol_input_n_columns(&self, input: i32) {
        self.post_proc_trim_vol_input_n_columns
            .lock()
            .unwrap()
            .set_int(input);
    }

    /// Java `setPostProcTrimVolInputNRows`.
    pub fn set_post_proc_trim_vol_input_n_rows(&self, input: i32) {
        self.post_proc_trim_vol_input_n_rows
            .lock()
            .unwrap()
            .set_int(input);
    }

    /// Java `setPostProcTrimVolInputNSections`.
    pub fn set_post_proc_trim_vol_input_n_sections(&self, input: i32) {
        self.post_proc_trim_vol_input_n_sections
            .lock()
            .unwrap()
            .set_int(input);
    }

    /// Java `setStackUseLinearInterpolation`.
    pub fn set_stack_use_linear_interpolation(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.stack_use_linear_interpolation_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.stack_use_linear_interpolation_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setStackUserSizeToOutputInXandY`.
    pub fn set_stack_user_size_to_output_in_x_and_y(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.stack_user_size_to_output_in_x_and_y_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.stack_user_size_to_output_in_x_and_y_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setStackImageRotation`.
    pub fn set_stack_image_rotation(&self, axis_id: AxisID, input: Option<&ConstEtomoNumber>) {
        if axis_id == AxisID::Second {
            self.stack_image_rotation_b
                .lock()
                .unwrap()
                .set_const_etomo_number(input);
        } else {
            self.stack_image_rotation_a
                .lock()
                .unwrap()
                .set_const_etomo_number(input);
        }
    }

    /// Java `setTrackLightBeads`.
    pub fn set_track_light_beads(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.track_light_beads_b.lock().unwrap().set_boolean(input);
        } else {
            self.track_light_beads_a.lock().unwrap().set_boolean(input);
        }
    }

    /// Java `setStackUsingNewstOrBlend3dFindOutput`.
    pub fn set_stack_using_newst_or_blend_3d_find_output(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.stack_using_newst_or_blend_3d_find_output_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.stack_using_newst_or_blend_3d_find_output_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setUseFixedStackWarning`.
    pub fn set_use_fixed_stack_warning(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_fixed_stack_warning_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_fixed_stack_warning_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setUseCtfCorrectionWarning`.
    pub fn set_use_ctf_correction_warning(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_ctf_correction_warning_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_ctf_correction_warning_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setUseErasedStackWarning`.
    pub fn set_use_erased_stack_warning(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_erased_stack_warning_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_erased_stack_warning_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setUseFilteredStackWarning`.
    pub fn set_use_filtered_stack_warning(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.use_filtered_stack_warning_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.use_filtered_stack_warning_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `setGenSirtsetupSubareaSize`.
    pub fn set_gen_sirtsetup_subarea_size(&self, axis_id: AxisID, input: Option<&str>) {
        if axis_id == AxisID::Second {
            self.gen_sirtsetup_subarea_size_b.lock().unwrap().set(input);
        } else {
            self.gen_sirtsetup_subarea_size_a.lock().unwrap().set(input);
        }
    }

    /// Java `setPosSampleType`.
    pub fn set_pos_sample_type(&self, axis_id: AxisID, input: Option<PosSampleType>) {
        if axis_id == AxisID::Second {
            match input {
                None => {
                    self.pos_sample_type_b.lock().unwrap().reset();
                }
                Some(input) => {
                    self.pos_sample_type_b
                        .lock()
                        .unwrap()
                        .set_int(input.get_value());
                }
            }
        } else {
            match input {
                None => {
                    self.pos_sample_type_a.lock().unwrap().reset();
                }
                Some(input) => {
                    self.pos_sample_type_a
                        .lock()
                        .unwrap()
                        .set_int(input.get_value());
                }
            }
        }
    }

    /// Java `getPosSampleType`.
    pub fn get_pos_sample_type(&self, axis_id: AxisID) -> i32 {
        if axis_id == AxisID::Second {
            return self.pos_sample_type_b.lock().unwrap().get_int();
        }
        self.pos_sample_type_a.lock().unwrap().get_int()
    }

    /// Java `setGenSirtsetupyOffsetOfSubarea`.
    pub fn set_gen_sirtsetupy_offset_of_subarea(&self, axis_id: AxisID, input: i32) {
        if axis_id == AxisID::Second {
            self.gen_sirtsetupy_offset_of_subarea_b
                .lock()
                .unwrap()
                .set_int(input);
        } else {
            self.gen_sirtsetupy_offset_of_subarea_a
                .lock()
                .unwrap()
                .set_int(input);
        }
    }

    /// Java `setUseRaptorResultWarning`.
    pub fn set_use_raptor_result_warning(&self, input: bool) {
        self.use_raptor_result_warning_a
            .lock()
            .unwrap()
            .set_boolean(input);
    }

    /// Java `isPostProcTrimVolInputNColumnsNull`.
    pub fn is_post_proc_trim_vol_input_n_columns_null(&self) -> bool {
        self.post_proc_trim_vol_input_n_columns
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `isPostProcTrimVolInputNRowsNull`.
    pub fn is_post_proc_trim_vol_input_n_rows_null(&self) -> bool {
        self.post_proc_trim_vol_input_n_rows
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `isPostProcTrimVolInputNSectionsNull`.
    pub fn is_post_proc_trim_vol_input_n_sections_null(&self) -> bool {
        self.post_proc_trim_vol_input_n_sections
            .lock()
            .unwrap()
            .is_null()
    }

    /// Java `isStackUseLinearInterpolation`.
    pub fn is_stack_use_linear_interpolation(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.stack_use_linear_interpolation_b.lock().unwrap().is();
        }
        self.stack_use_linear_interpolation_a.lock().unwrap().is()
    }

    /// Java `getStackUserSizeToOutputInXandY`.
    pub fn get_stack_user_size_to_output_in_x_and_y(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .stack_user_size_to_output_in_x_and_y_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.stack_user_size_to_output_in_x_and_y_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getStackImageRotation`.
    pub fn get_stack_image_rotation(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.stack_image_rotation_b.lock().unwrap().clone();
        }
        self.stack_image_rotation_a.lock().unwrap().clone()
    }

    /// Java `isTrackLightBeads`.
    pub fn is_track_light_beads(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_light_beads_b.lock().unwrap().is();
        }
        self.track_light_beads_a.lock().unwrap().is()
    }

    /// Java `isStackUsingNewstOrBlend3dFindOutput`.
    pub fn is_stack_using_newst_or_blend_3d_find_output(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self
                .stack_using_newst_or_blend_3d_find_output_b
                .lock()
                .unwrap()
                .is();
        }
        self.stack_using_newst_or_blend_3d_find_output_a
            .lock()
            .unwrap()
            .is()
    }

    /// Java `isUseFixedStackWarning`.
    pub fn is_use_fixed_stack_warning(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.use_fixed_stack_warning_b.lock().unwrap().is();
        }
        self.use_fixed_stack_warning_a.lock().unwrap().is()
    }

    /// Java `isUseCtfCorrectionWarning`.
    pub fn is_use_ctf_correction_warning(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.use_ctf_correction_warning_b.lock().unwrap().is();
        }
        self.use_ctf_correction_warning_a.lock().unwrap().is()
    }

    /// Java `isUseErasedStackWarning`.
    pub fn is_use_erased_stack_warning(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.use_erased_stack_warning_b.lock().unwrap().is();
        }
        self.use_erased_stack_warning_a.lock().unwrap().is()
    }

    /// Java `isUseFilteredStackWarning`.
    pub fn is_use_filtered_stack_warning(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.use_filtered_stack_warning_b.lock().unwrap().is();
        }
        self.use_filtered_stack_warning_a.lock().unwrap().is()
    }

    /// Java `getGenSirtsetupSubareaSize`.
    pub fn get_gen_sirtsetup_subarea_size(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_sirtsetup_subarea_size_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_sirtsetup_subarea_size_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `getGenSirtsetupyOffsetOfSubarea`.
    pub fn get_gen_sirtsetupy_offset_of_subarea(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self
                .gen_sirtsetupy_offset_of_subarea_b
                .lock()
                .unwrap()
                .to_string();
        }
        self.gen_sirtsetupy_offset_of_subarea_a
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isUseRaptorResultWarning`.
    pub fn is_use_raptor_result_warning(&self) -> bool {
        self.use_raptor_result_warning_a.lock().unwrap().is()
    }

    /// Java `isTrackLightBeadsNull`.
    pub fn is_track_light_beads_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.track_light_beads_b.lock().unwrap().is_null();
        }
        self.track_light_beads_a.lock().unwrap().is_null()
    }

    /// Java `getPostProcTrimVolInputNColumns`.
    pub fn get_post_proc_trim_vol_input_n_columns(&self) -> i32 {
        self.post_proc_trim_vol_input_n_columns
            .lock()
            .unwrap()
            .get_int()
    }

    /// Java `getPostProcTrimVolInputNRows`.
    pub fn get_post_proc_trim_vol_input_n_rows(&self) -> i32 {
        self.post_proc_trim_vol_input_n_rows
            .lock()
            .unwrap()
            .get_int()
    }

    /// Java `getPostProcTrimVolInputNSections`.
    pub fn get_post_proc_trim_vol_input_n_sections(&self) -> i32 {
        self.post_proc_trim_vol_input_n_sections
            .lock()
            .unwrap()
            .get_int()
    }

    /// Java `getTomogramSize`.
    pub fn get_tomogram_size(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.tomogram_size_b.lock().unwrap().clone();
        }
        self.tomogram_size_a.lock().unwrap().clone()
    }

    /// Java `getTomogramSizeColumns`.
    pub fn get_tomogram_size_columns(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.tomogram_size_columns_b.lock().unwrap().clone();
        }
        self.tomogram_size_columns_a.lock().unwrap().clone()
    }

    /// Java `getTomogramSizeRows`.
    pub fn get_tomogram_size_rows(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.tomogram_size_rows_b.lock().unwrap().clone();
        }
        self.tomogram_size_rows_a.lock().unwrap().clone()
    }

    /// Java `getTomogramSizeSections`.
    pub fn get_tomogram_size_sections(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.tomogram_size_sections_b.lock().unwrap().clone();
        }
        self.tomogram_size_sections_a.lock().unwrap().clone()
    }

    /// Java `setMadeZFactors`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_made_z_factors(&self, axis_id: AxisID, made_z_factors: bool) -> EtomoState {
        if axis_id == AxisID::Second {
            let mut field = self.made_z_factors_b.lock().unwrap();
            field.set_boolean(made_z_factors);
            return field.clone();
        }
        let mut field = self.made_z_factors_a.lock().unwrap();
        field.set_boolean(made_z_factors);
        field.clone()
    }

    /// Java `setAlignAxisZShift`.
    pub fn set_align_axis_z_shift(&self, axis_id: AxisID, axis_z_shift: f64) {
        if axis_id == AxisID::Second {
            self.axis_z_shift_b.lock().unwrap().set_double(axis_z_shift);
        } else {
            self.axis_z_shift_a.lock().unwrap().set_double(axis_z_shift);
        }
    }

    /// Java `getAlignAxisZShift`.
    pub fn get_align_axis_z_shift(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.axis_z_shift_b.lock().unwrap().clone();
        }
        self.axis_z_shift_a.lock().unwrap().clone()
    }

    /// Java `getSampleAxisZShift`.
    pub fn get_sample_axis_z_shift(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.sample_axis_z_shift_b.lock().unwrap().clone();
        }
        self.sample_axis_z_shift_a.lock().unwrap().clone()
    }

    /// Java `getSampleFiducialess`.
    pub fn get_sample_fiducialess(&self, axis_id: AxisID) -> Option<EtomoBoolean2> {
        if axis_id == AxisID::Second {
            return self.sample_fiducialess_b.lock().unwrap().clone();
        }
        self.sample_fiducialess_a.lock().unwrap().clone()
    }

    /// Java `setSampleAxisZShift`.
    pub fn set_sample_axis_z_shift_const_etomo_number(
        &self,
        axis_id: AxisID,
        axis_z_shift: Option<&ConstEtomoNumber>,
    ) {
        if axis_id == AxisID::Second {
            self.sample_axis_z_shift_b
                .lock()
                .unwrap()
                .set_const_etomo_number(axis_z_shift);
        } else {
            self.sample_axis_z_shift_a
                .lock()
                .unwrap()
                .set_const_etomo_number(axis_z_shift);
        }
    }

    /// Java `setSampleAxisZShift`.
    pub fn set_sample_axis_z_shift_double(&self, axis_id: AxisID, axis_z_shift: f64) {
        if axis_id == AxisID::Second {
            self.sample_axis_z_shift_b
                .lock()
                .unwrap()
                .set_double(axis_z_shift);
        } else {
            self.sample_axis_z_shift_a
                .lock()
                .unwrap()
                .set_double(axis_z_shift);
        }
    }

    /// Java `setSampleFiducialess`.
    pub fn set_sample_fiducialess(&self, axis_id: AxisID, sample_fiducialess: bool) {
        if axis_id == AxisID::Second {
            let second_axis_group = self.second_axis_group.lock().unwrap().clone();
            let current = self.sample_fiducialess_b.lock().unwrap().take();
            *self.sample_fiducialess_b.lock().unwrap() = EtomoBoolean2::get_instance_from_boolean(
                current,
                &format!(
                    "{}{}",
                    second_axis_group.as_deref().unwrap_or("null"),
                    SAMPLE_FIDUCIALESS_KEY.as_str()
                ),
                sample_fiducialess,
            );
        } else {
            let first_axis_group = self.first_axis_group.lock().unwrap().clone();
            let current = self.sample_fiducialess_a.lock().unwrap().take();
            *self.sample_fiducialess_a.lock().unwrap() = EtomoBoolean2::get_instance_from_boolean(
                current,
                &format!(
                    "{}{}",
                    first_axis_group.as_deref().unwrap_or("null"),
                    SAMPLE_FIDUCIALESS_KEY.as_str()
                ),
                sample_fiducialess,
            );
        }
    }

    /// Java `setFidFileLastModified`.
    pub fn set_fid_file_last_modified(&self, axis_id: AxisID, fid_file_last_modified: i64) {
        if axis_id == AxisID::Second {
            self.fid_file_last_modified_b
                .lock()
                .unwrap()
                .set_long(fid_file_last_modified);
        } else {
            self.fid_file_last_modified_a
                .lock()
                .unwrap()
                .set_long(fid_file_last_modified);
        }
    }

    /// Java `setFixedFiducials`.
    pub fn set_fixed_fiducials(&self, axis_id: AxisID, fixed_fiducials: bool) {
        if axis_id == AxisID::Second {
            self.fixed_fiducials_b
                .lock()
                .unwrap()
                .set_boolean(fixed_fiducials);
        } else {
            self.fixed_fiducials_a
                .lock()
                .unwrap()
                .set_boolean(fixed_fiducials);
        }
    }

    /// Java `setSeedFileLastModified`.
    pub fn set_seed_file_last_modified(&self, axis_id: AxisID, seed_file_last_modified: i64) {
        if axis_id == AxisID::Second {
            self.seed_file_last_modified_b
                .lock()
                .unwrap()
                .set_long(seed_file_last_modified);
        } else {
            self.seed_file_last_modified_a
                .lock()
                .unwrap()
                .set_long(seed_file_last_modified);
        }
    }

    /// Java `resetFidFileLastModified`.
    pub fn reset_fid_file_last_modified(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.fid_file_last_modified_b.lock().unwrap().reset();
        } else {
            self.fid_file_last_modified_a.lock().unwrap().reset();
        }
    }

    /// Java `resetSeedFileLastModified`.
    pub fn reset_seed_file_last_modified(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            self.seed_file_last_modified_b.lock().unwrap().reset();
        } else {
            self.seed_file_last_modified_a.lock().unwrap().reset();
        }
    }

    /// Java `setAlignAngleOffset`.
    pub fn set_align_angle_offset(&self, axis_id: AxisID, angle_offset: f64) {
        if axis_id == AxisID::Second {
            self.angle_offset_b.lock().unwrap().set_double(angle_offset);
        } else {
            self.angle_offset_a.lock().unwrap().set_double(angle_offset);
        }
    }

    /// Java `getFidFileLastModified`.
    pub fn get_fid_file_last_modified(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.fid_file_last_modified_b.lock().unwrap().clone();
        }
        self.fid_file_last_modified_a.lock().unwrap().clone()
    }

    /// Java `getSeedFileLastModified`.
    pub fn get_seed_file_last_modified(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.seed_file_last_modified_b.lock().unwrap().clone();
        }
        self.seed_file_last_modified_a.lock().unwrap().clone()
    }

    /// Java `isFixedFiducials`.
    pub fn is_fixed_fiducials(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.fixed_fiducials_b.lock().unwrap().is();
        }
        self.fixed_fiducials_a.lock().unwrap().is()
    }

    /// Java `isSeedingDone`.
    pub fn is_seeding_done(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.seeding_done_b.lock().unwrap().is();
        }
        self.seeding_done_a.lock().unwrap().is()
    }

    /// Java `isXcorrBlendmontWasRun`.
    pub fn is_xcorr_blendmont_was_run(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.xcorr_blendmont_was_run_b.lock().unwrap().is();
        }
        self.xcorr_blendmont_was_run_a.lock().unwrap().is()
    }

    /// Java `setXcorrBlendmontWasRun`.
    pub fn set_xcorr_blendmont_was_run(&self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.xcorr_blendmont_was_run_b
                .lock()
                .unwrap()
                .set_boolean(input);
        } else {
            self.xcorr_blendmont_was_run_a
                .lock()
                .unwrap()
                .set_boolean(input);
        }
    }

    /// Java `getAlignAngleOffset`.
    pub fn get_align_angle_offset(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.angle_offset_b.lock().unwrap().clone();
        }
        self.angle_offset_a.lock().unwrap().clone()
    }

    /// Java `getSampleAngleOffset`.
    pub fn get_sample_angle_offset(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.sample_angle_offset_b.lock().unwrap().clone();
        }
        self.sample_angle_offset_a.lock().unwrap().clone()
    }

    /// Java `setSampleAngleOffset`.
    pub fn set_sample_angle_offset_const_etomo_number(
        &self,
        axis_id: AxisID,
        angle_offset: Option<&ConstEtomoNumber>,
    ) {
        if axis_id == AxisID::Second {
            self.sample_angle_offset_b
                .lock()
                .unwrap()
                .set_const_etomo_number(angle_offset);
        } else {
            self.sample_angle_offset_a
                .lock()
                .unwrap()
                .set_const_etomo_number(angle_offset);
        }
    }

    /// Java `setSampleAngleOffset`.
    pub fn set_sample_angle_offset_double(&self, axis_id: AxisID, angle_offset: f64) {
        if axis_id == AxisID::Second {
            self.sample_angle_offset_b
                .lock()
                .unwrap()
                .set_double(angle_offset);
        } else {
            self.sample_angle_offset_a
                .lock()
                .unwrap()
                .set_double(angle_offset);
        }
    }

    /// Java `getSampleXAxisTilt`.
    pub fn get_sample_x_axis_tilt(&self, axis_id: AxisID) -> EtomoNumber {
        if axis_id == AxisID::Second {
            return self.sample_x_axis_tilt_b.lock().unwrap().clone();
        }
        self.sample_x_axis_tilt_a.lock().unwrap().clone()
    }

    /// Java `setSampleXAxisTilt`.
    pub fn set_sample_x_axis_tilt(&self, axis_id: AxisID, x_axis_tilt: f64) {
        if axis_id == AxisID::Second {
            self.sample_x_axis_tilt_b
                .lock()
                .unwrap()
                .set_double(x_axis_tilt);
        } else {
            self.sample_x_axis_tilt_a
                .lock()
                .unwrap()
                .set_double(x_axis_tilt);
        }
    }

    /// Java `setSeedingDone`.
    pub fn set_seeding_done(&self, axis_id: AxisID, seeding_done: bool) {
        if axis_id == AxisID::Second {
            self.seeding_done_b
                .lock()
                .unwrap()
                .set_boolean(seeding_done);
        } else {
            self.seeding_done_a
                .lock()
                .unwrap()
                .set_boolean(seeding_done);
        }
    }

    /// Java `setInvalidEdgeFunctions`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_invalid_edge_functions(
        &self,
        axis_id: AxisID,
        invalid_edge_functions: bool,
    ) -> EtomoState {
        if axis_id == AxisID::Second {
            let mut field = self.invalid_edge_functions_b.lock().unwrap();
            field.set_boolean(invalid_edge_functions);
            return field.clone();
        }
        let mut field = self.invalid_edge_functions_a.lock().unwrap();
        field.set_boolean(invalid_edge_functions);
        field.clone()
    }

    /// Java `setNewstFiducialessAlignment`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_newst_fiducialess_alignment(
        &self,
        axis_id: AxisID,
        newst_fiducialess_alignment: bool,
    ) -> EtomoState {
        if axis_id == AxisID::Second {
            let mut field = self.newst_fiducialess_alignment_b.lock().unwrap();
            field.set_boolean(newst_fiducialess_alignment);
            return field.clone();
        }
        let mut field = self.newst_fiducialess_alignment_a.lock().unwrap();
        field.set_boolean(newst_fiducialess_alignment);
        field.clone()
    }

    /// Java `setUsedLocalAlignments`.  Java returns the field itself (as `ConstEtomoNumber`); a copy is
    /// returned here.
    pub fn set_used_local_alignments(
        &self,
        axis_id: AxisID,
        used_local_alignments: bool,
    ) -> EtomoState {
        if axis_id == AxisID::Second {
            let mut field = self.used_local_alignments_b.lock().unwrap();
            field.set_boolean(used_local_alignments);
            return field.clone();
        }
        let mut field = self.used_local_alignments_a.lock().unwrap();
        field.set_boolean(used_local_alignments);
        field.clone()
    }

    /// Java `getTrimvolFlipped`.
    pub fn get_trimvol_flipped(&self) -> EtomoState {
        self.trimvol_flipped.lock().unwrap().clone()
    }

    /// Java `isAltTomoPreprocessForExtremes`.
    pub fn is_alt_tomo_preprocess_for_extremes(&self) -> bool {
        self.alt_tomo_preprocess_for_extremes.lock().unwrap().is()
    }

    /// Java `isAltTomoCorrectCTF`.
    pub fn is_alt_tomo_correct_ctf(&self) -> bool {
        self.alt_tomo_correct_ctf.lock().unwrap().is()
    }

    /// Java `isAltTomoEraseFiducials`.
    pub fn is_alt_tomo_erase_fiducials(&self) -> bool {
        self.alt_tomo_erase_fiducials.lock().unwrap().is()
    }

    /// Java `isAltTomoFilterIn2D`.
    pub fn is_alt_tomo_filter_in_2d(&self) -> bool {
        self.alt_tomo_filter_in_2d.lock().unwrap().is()
    }

    /// Java `isAltTomoTrimVolChecked`.
    pub fn is_alt_tomo_trim_vol_checked(&self) -> bool {
        self.alt_tomo_trim_vol.lock().unwrap().is()
    }

    /// Java `getAltTomoRootnameToProcess`.
    pub fn get_alt_tomo_rootname_to_process(&self) -> String {
        self.alt_tomo_rootname_to_process
            .lock()
            .unwrap()
            .to_string()
    }

    /// Java `isAltTomoEvenAndOddPairs`.
    pub fn is_alt_tomo_even_and_odd_pairs(&self) -> bool {
        self.alt_tomo_even_and_odd_pairs.lock().unwrap().is()
    }

    /// Java `getAltTomoAxisToProcess`.
    pub fn get_alt_tomo_axis_to_process(&self) -> String {
        self.alt_tomo_axis_to_process.lock().unwrap().to_string()
    }

    /// Java `getSqueezevolFlipped`.
    pub fn get_squeezevol_flipped(&self) -> EtomoState {
        self.squeezevol_flipped.lock().unwrap().clone()
    }

    /// Java `getReduceFiltVolFlipped`.
    pub fn get_reduce_filt_vol_flipped(&self) -> EtomoState {
        self.reduce_filt_vol_flipped.lock().unwrap().clone()
    }

    /// Java `isFlattenFlipped`.
    pub fn is_flatten_flipped(&self) -> bool {
        self.flatten_flipped.lock().unwrap().is()
    }

    /// Java `isResultSetFlattenFlipped`.
    pub fn is_result_set_flatten_flipped(&self) -> bool {
        self.flatten_flipped.lock().unwrap().is_result_set()
    }

    /// Java `getMadeZFactors`.
    pub fn get_made_z_factors(&self, axis_id: AxisID) -> EtomoState {
        if axis_id == AxisID::Second {
            return self.made_z_factors_b.lock().unwrap().clone();
        }
        self.made_z_factors_a.lock().unwrap().clone()
    }

    /// Java `getInvalidEdgeFunctions`.
    pub fn get_invalid_edge_functions(&self, axis_id: AxisID) -> EtomoState {
        if axis_id == AxisID::Second {
            return self.invalid_edge_functions_b.lock().unwrap().clone();
        }
        self.invalid_edge_functions_a.lock().unwrap().clone()
    }

    /// Java `getNewstFiducialessAlignment`.
    pub fn get_newst_fiducialess_alignment(&self, axis_id: AxisID) -> EtomoState {
        if axis_id == AxisID::Second {
            return self.newst_fiducialess_alignment_b.lock().unwrap().clone();
        }
        self.newst_fiducialess_alignment_a.lock().unwrap().clone()
    }

    /// Java `isNewstFiducialessAlignment`.
    pub fn is_newst_fiducialess_alignment(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.newst_fiducialess_alignment_b.lock().unwrap().is();
        }
        self.newst_fiducialess_alignment_a.lock().unwrap().is()
    }

    /// Java `getUsedLocalAlignments`.
    pub fn get_used_local_alignments(&self, axis_id: AxisID) -> EtomoState {
        if axis_id == AxisID::Second {
            return self.used_local_alignments_b.lock().unwrap().clone();
        }
        self.used_local_alignments_a.lock().unwrap().clone()
    }

    /// Java `getBackwardCompatibleTrimvolFlipped`.  Backward compatibility: decide
    /// whether trimvol is flipped based on the header.
    ///
    /// The source dereferences `manager` unconditionally; without one there is no
    /// dataset directory to look in, and the method answers false, as it does when the
    /// trimvol output does not exist.
    pub fn get_backward_compatible_trimvol_flipped(&self) -> bool {
        // If trimvol has not been run, then assume that the tomogram has not been
        // flipped.
        // `EtomoDirector etomoDirector = EtomoDirector.INSTANCE;` and `String datasetName
        // = manager.getName();` are assigned and never read.
        let manager = match self.manager {
            None => return false,
            Some(manager) => manager,
        };
        let property_user_dir = manager.get_property_user_dir().unwrap_or_default();
        // `TrimvolParam.getOutputFileName(manager, AxisID.ONLY)` is
        // `FileType.TRIM_VOL_OUTPUT.getFileName(manager, axisID)` (TrimvolParam.java:646-648).
        let trimvol_file = java_io_file_new(
            &property_user_dir,
            &file_type::CLASS
                .trim_vol_output
                .get_file_name(
                    Some(manager as &'static dyn BaseManager),
                    Some(AxisID::Only),
                )
                .unwrap_or("null".to_string()),
        );
        if !std::path::Path::new(&trimvol_file).exists() {
            return false;
        }
        let header = match MRCHeader::get_instance_in_dir(
            Some(&property_user_dir),
            Some(&java_io_file_get_absolute_path(&trimvol_file)),
            Some(AxisID::Only),
        ) {
            None => return false,
            Some(header) => header,
        };
        match header
            .borrow_mut()
            .read_with_manager(manager as &'static dyn BaseManager)
        {
            Ok(false) => return false,
            // `catch (IOException e)`; the translated `read` reports every failure as
            // an error value, which covers the source's generic `catch (Exception e)`
            // (which also printed the stack trace) as well.
            Err(_) => return false,
            Ok(true) => {}
        }
        if header.borrow().get_n_rows() < header.borrow().get_n_sections() {
            eprintln!(
                "Assuming that {} has not been flipped\nbecause the Y is less then Z in the header.",
                java_io_file_get_name(&trimvol_file)
            );
            return false;
        }
        eprintln!(
            "Assuming that {} has been flipped\nbecause the Y is greater or equal to Z in the header.",
            java_io_file_get_name(&trimvol_file)
        );
        true
    }

    /// Java `getBackwardCompatibleUsedLocalAlignments`.  Backward compatibility:
    /// decide whether tiltalign was run with local alignments based on file time.  See
    /// `get_backward_compatible_trimvol_flipped` for the missing-manager answer.
    pub fn get_backward_compatible_used_local_alignments(&self, axis_id: AxisID) -> bool {
        let manager = match self.manager {
            None => return false,
            Some(manager) => manager,
        };
        let user_dir = manager
            .get_property_user_dir()
            .unwrap_or("null".to_string());
        let dataset_name = manager.get_name().unwrap_or("null".to_string());
        let local_xf_file = java_io_file_new(
            &user_dir,
            &format!("{}{}local.xf", dataset_name, axis_id.get_extension()),
        );
        let transform_file = java_io_file_new(
            &user_dir,
            &format!("{}{}.tltxf", dataset_name, axis_id.get_extension()),
        );
        if !std::path::Path::new(&local_xf_file).exists() {
            eprintln!(
                "Assuming that local alignments where not used \nbecause {} does not exist.",
                java_io_file_get_name(&local_xf_file)
            );
            return false;
        }
        if !std::path::Path::new(&transform_file).exists() {
            eprintln!(
                "Assuming that local alignments where not used \nbecause {} does not exist.",
                java_io_file_get_name(&transform_file)
            );
            return false;
        }
        if java_io_file_last_modified(&local_xf_file) < java_io_file_last_modified(&transform_file)
        {
            eprintln!(
                "Assuming that local alignments where not used \nbecause {} was modified before {}.",
                java_io_file_get_name(&local_xf_file),
                java_io_file_get_name(&transform_file)
            );
            return false;
        }
        eprintln!(
            "Assuming that local alignments where used \nbecause {} was modified after {}.",
            java_io_file_get_name(&local_xf_file),
            java_io_file_get_name(&transform_file)
        );
        true
    }

    /// Java `getBackwardCompatibleMadeZFactors`.  Backward compatibility: decide
    /// whether z factors where made based on the relationship between .zfac file and
    /// the .tltxf file.  See `get_backward_compatible_trimvol_flipped` for the
    /// missing-manager answer.
    pub fn get_backward_compatible_made_z_factors(&self, axis_id: AxisID) -> bool {
        // `EtomoDirector etomoDirector = EtomoDirector.INSTANCE;` is never read.
        let manager = match self.manager {
            None => return false,
            Some(manager) => manager,
        };
        let user_dir = manager
            .get_property_user_dir()
            .unwrap_or("null".to_string());
        let dataset_name = manager.get_name().unwrap_or("null".to_string());
        // `TiltalignParam.getOutputZFactorFileName(datasetName, axisID)` is `datasetName +
        // axisID.getExtension() + zFactorFileExtension` with `zFactorFileExtension =
        // ".zfac"` (ConstTiltalignParam.java:118, :945-947).
        let z_factor_file = java_io_file_new(
            &user_dir,
            &format!("{}{}.zfac", dataset_name, axis_id.get_extension()),
        );
        let tltxf_file = java_io_file_new(
            &user_dir,
            &format!("{}{}.tltxf", dataset_name, axis_id.get_extension()),
        );
        if !std::path::Path::new(&z_factor_file).exists() {
            eprintln!(
                "Assuming that madeZFactors is false\nbecause {} does not exist.",
                java_io_file_get_name(&z_factor_file)
            );
            return false;
        }
        if !std::path::Path::new(&tltxf_file).exists() {
            eprintln!(
                "Assuming that madeZFactors is false\nbecause {} does not exist.",
                java_io_file_get_name(&tltxf_file)
            );
            return false;
        }
        if java_io_file_last_modified(&z_factor_file) < java_io_file_last_modified(&tltxf_file) {
            eprintln!(
                "Assuming that madeZFactors is false\nbecause {} is older then {}.",
                java_io_file_get_name(&z_factor_file),
                java_io_file_get_name(&tltxf_file)
            );
            return false;
        }
        eprintln!(
            "Assuming that madeZFactors is true\nbecause {} was modified after {}.",
            java_io_file_get_name(&z_factor_file),
            java_io_file_get_name(&tltxf_file)
        );
        true
    }
}

impl BaseState for TomogramState {
    /// Java package-private `createPrepend`, implementing `BaseState`.
    fn create_prepend(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            return GROUP_STRING.to_string();
        }
        format!("{}.{}", prepend, GROUP_STRING)
    }
}

/// Java `Storable`, implemented through `BaseState`.
impl storable::Storable for TomogramState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        TomogramState::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        TomogramState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        TomogramState::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        TomogramState::load_with_prepend(self, properties, prepend);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The key/value set the Java reference stores for the setter sequence below
    /// (`new TomogramState(null)`, `initialize()`, the setters, `store(props, "")`,
    /// run headless against the classes compiled from the vendored source).
    const JAVA_STORE: &str = r#"ReconstructionState.A.AdjustOrigin=true
ReconstructionState.A.tomogramSize.Columns=1024
ReconstructionState.AltTomoAxisToProcess=-1
ReconstructionState.AltTomoCorrectCTF=no result
ReconstructionState.AltTomoEraseFiducials=no result
ReconstructionState.AltTomoEvenAndOddPairs=no result
ReconstructionState.AltTomoFilterIn2D=no result
ReconstructionState.AltTomoPreprocessForExtremes=no result
ReconstructionState.AltTomoTrimVol=no result
ReconstructionState.B.AdjustOrigin=true
ReconstructionState.Combine.MatchMode=A_TO_B
ReconstructionState.Combine.ScriptsCreated=no result
ReconstructionState.FlattenFlipped=no result
ReconstructionState.Gen.B.SirtsetupyOffsetOfSubarea=-5
ReconstructionState.InvalidEdgeFunctionsA=no result
ReconstructionState.InvalidEdgeFunctionsB=no result
ReconstructionState.MadeZFactorsA=no result
ReconstructionState.MadeZFactorsB=false
ReconstructionState.NewstFiducialessAlignmentA=no result
ReconstructionState.NewstFiducialessAlignmentB=no result
ReconstructionState.ReduceFiltVolFlipped=no result
ReconstructionState.RootnameToProcess=root
ReconstructionState.SqueezevolFlipped=no result
ReconstructionState.Stack.B.SizeToOutputInXandY=512,512
ReconstructionState.Track.A.UseRaptorResult.Warning=true
ReconstructionState.TrimvolFlipped=true
ReconstructionState.UsedLocalAlignmentsA=no result
ReconstructionState.UsedLocalAlignmentsB=no result
ReconstructionState.a.align.AxisZShift=1.5
ReconstructionState.a.positioning.PosSampleType=1
ReconstructionState.a.track.LastModified=1234567890123"#;

    fn java_map() -> BTreeMap<String, String> {
        let mut props = BTreeMap::new();
        for line in JAVA_STORE.lines() {
            let (key, value) = line.split_once('=').unwrap();
            props.insert(key.to_string(), value.to_string());
        }
        props
    }

    fn configured() -> TomogramState {
        let ts = TomogramState::new(None);
        ts.initialize();
        ts.set_trimvol_flipped(true);
        ts.set_made_z_factors(AxisID::Second, false);
        ts.set_align_axis_z_shift(AxisID::First, 1.5);
        ts.set_fid_file_last_modified(AxisID::First, 1234567890123);
        ts.set_stack_user_size_to_output_in_x_and_y(AxisID::Second, Some("512,512"));
        ts.set_tomogram_size_columns(AxisID::First, 1024);
        ts.set_combine_match_mode(Some(MatchMode::AToB));
        ts.set_pos_sample_type(AxisID::First, Some(PosSampleType::Whole));
        ts.set_alt_tomo_rootname_to_process(Some("root"));
        ts.set_gen_sirtsetupy_offset_of_subarea(AxisID::Second, -5);
        ts.set_use_raptor_result_warning(true);
        ts
    }

    #[test]
    fn store_matches_java() {
        let mut props = BTreeMap::new();
        configured().store(&mut props);
        assert_eq!(props, java_map());
    }

    #[test]
    fn load_store_round_trip() {
        let java = java_map();
        let ts = TomogramState::new(None);
        ts.load(&java);
        assert!(ts.get_trimvol_flipped().is());
        assert!(!ts.get_made_z_factors(AxisID::Second).is());
        assert!(!ts.get_made_z_factors(AxisID::First).is_result_set());
        assert_eq!(ts.get_align_axis_z_shift(AxisID::First).get_double(), 1.5);
        assert_eq!(ts.get_combine_match_mode(), Some(MatchMode::AToB));
        assert_eq!(ts.get_pos_sample_type(AxisID::First), 1);
        assert!(ts.is_use_raptor_result_warning());
        let mut props = BTreeMap::new();
        ts.store(&mut props);
        assert_eq!(props, java);
    }

    #[test]
    fn backward_compatible_keys_are_read_and_removed() {
        let mut props = BTreeMap::new();
        props.insert("Setup.Combine.MatchBtoA".to_string(), "true".to_string());
        props.insert("Setup.ComScriptsCreated".to_string(), "true".to_string());
        let ts = TomogramState::new(None);
        ts.load(&props);
        assert_eq!(ts.get_combine_match_mode(), Some(MatchMode::BToA));
        assert!(ts.is_combine_scripts_created());
        ts.store(&mut props);
        assert!(!props.contains_key("Setup.Combine.MatchBtoA"));
        assert!(!props.contains_key("Setup.ComScriptsCreated"));
        assert_eq!(
            props.get("ReconstructionState.Combine.MatchMode").unwrap(),
            "B_TO_A"
        );
        assert_eq!(
            props
                .get("ReconstructionState.Combine.ScriptsCreated")
                .unwrap(),
            "true"
        );
    }

    #[test]
    fn tomogram_state_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TomogramState>();
    }
}
