//! `IMOD/Etomo/src/etomo/process/ImodManager.java`.
//!
//! The 3dmod launcher: it names every kind of file eTomo can open in 3dmod, maps each
//! to a key, and builds the `ImodState` that describes how to open it.
//!
//! **Representation.**  `ImodManager extends BaseImodManager`; Rust represents the
//! superclass state explicitly in `base`, followed by this class's fields.  The private
//! `new*` factory dispatch now produces translated [`ImodState`] values directly.
//!
//! **Frontier.**  `ImodState` retains the 3dmod process boundary as an explicit launch
//! error, and the five `setMetaData` overloads additionally name
//! metadata classes that have no module.  What this module *does* carry in full is the
//! key vocabulary - every `public static final String *_KEY` and every private key field
//! - which is what `etomo/type/FileType.java`'s singletons are declared with.
#![allow(dead_code)]

use std::convert::Infallible;

use crate::imod::etomo::process::base_imod_manager::BaseImodManager;
use crate::imod::etomo::process::imod_state::{ImodState, MODEL_MODE, MODEL_VIEW, MODV};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;

// public keys
/// Java `RAW_STACK_KEY`.
pub const RAW_STACK_KEY: &str = "raw stack";
/// Java `ERASED_STACK_KEY`.
pub const ERASED_STACK_KEY: &str = "erased stack";
/// Java `COARSE_ALIGNED_KEY`.
pub const COARSE_ALIGNED_KEY: &str = "coarse aligned";
/// Java `FINE_ALIGNED_KEY`.
pub const FINE_ALIGNED_KEY: &str = "fine aligned";
/// Java `SAMPLE_KEY`.
pub const SAMPLE_KEY: &str = "sample";
/// Java `FULL_VOLUME_KEY`.
pub const FULL_VOLUME_KEY: &str = "full volume";
/// Java `COMBINED_TOMOGRAM_KEY`.
pub const COMBINED_TOMOGRAM_KEY: &str = "combined tomogram";
/// Java `FIDUCIAL_MODEL_KEY`.
pub const FIDUCIAL_MODEL_KEY: &str = "fiducial model";
/// Java `SORTED_MODELS_KEY`.
pub const SORTED_MODELS_KEY: &str = "sorted models";
/// Java `TRIMMED_VOLUME_KEY`.
pub const TRIMMED_VOLUME_KEY: &str = "trimmed volume";
/// Java `PATCH_VECTOR_MODEL_KEY`.
pub const PATCH_VECTOR_MODEL_KEY: &str = "patch vector model";
/// Java `MATCH_CHECK_KEY`.
pub const MATCH_CHECK_KEY: &str = "match check";
/// Java `TRIAL_TOMOGRAM_KEY`.
pub const TRIAL_TOMOGRAM_KEY: &str = "trial tomogram";
/// Java `MTF_FILTER_KEY`.
pub const MTF_FILTER_KEY: &str = "mtf filter";
/// Java `PREVIEW_KEY`.
pub const PREVIEW_KEY: &str = "preview";
/// Java `TOMOGRAM_KEY`.
pub const TOMOGRAM_KEY: &str = "tomogram";
/// Java `JOIN_SAMPLES_KEY`.
pub const JOIN_SAMPLES_KEY: &str = "joinSamples";
/// Java `JOIN_SAMPLE_AVERAGES_KEY`.
pub const JOIN_SAMPLE_AVERAGES_KEY: &str = "joinSampleAverages";
/// Java `JOIN_KEY`.
pub const JOIN_KEY: &str = "join";
/// Java `ROT_TOMOGRAM_KEY`.
pub const ROT_TOMOGRAM_KEY: &str = "rotTomogram";
/// Java `TRIAL_JOIN_KEY`.
pub const TRIAL_JOIN_KEY: &str = "TrialJoinKey";
/// Java `SQUEEZED_VOLUME_KEY`.
pub const SQUEEZED_VOLUME_KEY: &str = "SqueezedVolume";
/// Java `REDUCED_FILTERED_VOLUME_KEY`.
pub const REDUCED_FILTERED_VOLUME_KEY: &str = "ReducedFilteredVolume";
/// Java `FLATTEN_REDUCE_FILT_VOL_KEY`.
pub const FLATTEN_REDUCE_FILT_VOL_KEY: &str = "FlattenReducedFiltVol";
/// Java `PATCH_VECTOR_CCC_MODEL_KEY`.
pub const PATCH_VECTOR_CCC_MODEL_KEY: &str = "patch vector ccc model";
/// Java `MODELED_JOIN_KEY`.
pub const MODELED_JOIN_KEY: &str = "modeled join";
/// Java `TRANSFORMED_MODEL_KEY`.
pub const TRANSFORMED_MODEL_KEY: &str = "transformed model";
/// Java `AVG_VOL_KEY`.
pub const AVG_VOL_KEY: &str = "AvgVol";
/// Java `REF_KEY`.
pub const REF_KEY: &str = "Ref";
/// Java `VOLUME_KEY`.
pub const VOLUME_KEY: &str = "Volume";
/// Java `TEST_VOLUME_KEY`.
pub const TEST_VOLUME_KEY: &str = "TestVolume";
/// Java `VARYING_K_TEST_KEY`.
pub const VARYING_K_TEST_KEY: &str = "VaryingKTest";
/// Java `VARYING_ITERATION_TEST_KEY`.
pub const VARYING_ITERATION_TEST_KEY: &str = "VaryingIterationTest";
/// Java `ANISOTROPIC_DIFFUSION_VOLUME_KEY`.
pub const ANISOTROPIC_DIFFUSION_VOLUME_KEY: &str = "AnisotropicDiffusionVolume";
/// Java `CTF_CORRECTION_KEY`.
pub const CTF_CORRECTION_KEY: &str = "CtfCorrection";
/// Java `ERASED_FIDUCIALS_KEY`.
pub const ERASED_FIDUCIALS_KEY: &str = "erased fiducials";
/// Java `FLAT_VOLUME_KEY`.
pub const FLAT_VOLUME_KEY: &str = "flattened volume";
/// Java `FINE_ALIGNED_3D_FIND_KEY`.
pub const FINE_ALIGNED_3D_FIND_KEY: &str = "fine aligned for findbeads3d";
/// Java `FULL_VOLUME_3D_FIND_KEY`.
pub const FULL_VOLUME_3D_FIND_KEY: &str = "full volume for findbeads3d";
/// Java `SMOOTHING_ASSESSMENT_KEY`.
pub const SMOOTHING_ASSESSMENT_KEY: &str = "Smoothing assessment flattenwarp output";
/// Java `FLATTEN_INPUT_KEY`.
pub const FLATTEN_INPUT_KEY: &str = "Flatten input file";
/// Java `FLATTEN_TOOL_OUTPUT_KEY`.
pub const FLATTEN_TOOL_OUTPUT_KEY: &str = "Flatten tool output file";
/// Java `SIRT_KEY`.
pub const SIRT_KEY: &str = "SIRT output files";
/// Java `PREBLEND_KEY`.
pub const PREBLEND_KEY: &str = "Preblend output file";
/// Java `ALIGNED_STACK_KEY`.
pub const ALIGNED_STACK_KEY: &str = "Aligned stack";
/// Java `BATCH_RUN_TOMO_STACK_KEY`.
pub const BATCH_RUN_TOMO_STACK_KEY: &str = "BatchRunTomo Stack";
/// Java `BATCH_RUN_TOMO_REC_KEY`.
pub const BATCH_RUN_TOMO_REC_KEY: &str = "BatchRunTomo Tomogram";
/// Java `BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY`.
pub const BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY: &str = "BatchRunTomo Trimmed Volume";
/// Java `MULTIFILT_KEY`.
pub const MULTIFILT_KEY: &str = "processchunks tilt_mulfil output";
/// Java `CTF_3D_KEY`.
pub const CTF_3D_KEY: &str = "CTF corrected tomogram";
/// Java `OPEN_OUTPUT_TILT_SERIES_KEY`.
pub const OPEN_OUTPUT_TILT_SERIES_KEY: &str = "Open output tilt series";
/// Java `SUBTOMO_SETUP_KEY`.
pub const SUBTOMO_SETUP_KEY: &str = "processchunks subtomo_setup output";
/// Java `ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY`.
pub const ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY: &str =
    "processchunks alttomosetup even and odd tomograms";
/// Java `ALT_TOMO_SETUP_EVEN_ODD_FULL_TOMOGRAM_KEY`.
pub const ALT_TOMO_SETUP_EVEN_ODD_FULL_TOMOGRAM_KEY: &str =
    "processchunks alttomosetup even and odd full tomograms";
/// Java `ALT_TOMO_SETUP_TOMOGRAM_KEY`.
pub const ALT_TOMO_SETUP_TOMOGRAM_KEY: &str = "processchunks alttomosetup tomogram";
/// Java `GENERIC_PARALLEL_PROCESS_OUTPUT_FILE_KEY`.
pub const GENERIC_PARALLEL_PROCESS_OUTPUT_FILE_KEY: &str = "Generic parallel process output file";
/// Java `GENERIC_PARALLEL_PROCESS_OUTPUT_UNKNOWN_FILE_KEY`.
pub const GENERIC_PARALLEL_PROCESS_OUTPUT_UNKNOWN_FILE_KEY: &str =
    "Generic parallel process output, unknown file";

// private keys - used with imodMap
/// Java private static field `rawStackKey = RAW_STACK_KEY`.
const RAW_STACK_KEY_FIELD: &str = RAW_STACK_KEY;
/// Java private static field `erasedStackKey = ERASED_STACK_KEY`.
const ERASED_STACK_KEY_FIELD: &str = ERASED_STACK_KEY;
/// Java private static field `coarseAlignedKey = COARSE_ALIGNED_KEY`.
const COARSE_ALIGNED_KEY_FIELD: &str = COARSE_ALIGNED_KEY;
/// Java private static field `fineAlignedKey = FINE_ALIGNED_KEY`.
const FINE_ALIGNED_KEY_FIELD: &str = FINE_ALIGNED_KEY;
/// Java private static field `sampleKey = SAMPLE_KEY`.
const SAMPLE_KEY_FIELD: &str = SAMPLE_KEY;
/// Java private static field `fullVolumeKey = FULL_VOLUME_KEY`.
const FULL_VOLUME_KEY_FIELD: &str = FULL_VOLUME_KEY;
/// Java private static field `fiducialModelKey = FIDUCIAL_MODEL_KEY`.
const FIDUCIAL_MODEL_KEY_FIELD: &str = FIDUCIAL_MODEL_KEY;
/// Java private static field `sortedModelsKey = SORTED_MODELS_KEY`.
const SORTED_MODELS_KEY_FIELD: &str = SORTED_MODELS_KEY;
/// Java private static field `trimmedVolumeKey = TRIMMED_VOLUME_KEY`.
const TRIMMED_VOLUME_KEY_FIELD: &str = TRIMMED_VOLUME_KEY;
/// Java private static field `patchVectorModelKey = PATCH_VECTOR_MODEL_KEY`.
const PATCH_VECTOR_MODEL_KEY_FIELD: &str = PATCH_VECTOR_MODEL_KEY;
/// Java private static field `matchCheckKey = MATCH_CHECK_KEY`.
const MATCH_CHECK_KEY_FIELD: &str = MATCH_CHECK_KEY;
/// Java private static field `trialTomogramKey = TRIAL_TOMOGRAM_KEY`.
const TRIAL_TOMOGRAM_KEY_FIELD: &str = TRIAL_TOMOGRAM_KEY;
/// Java private static field `mtfFilterKey = MTF_FILTER_KEY`.
const MTF_FILTER_KEY_FIELD: &str = MTF_FILTER_KEY;
/// Java private static field `previewKey = PREVIEW_KEY`.
const PREVIEW_KEY_FIELD: &str = PREVIEW_KEY;
/// Java private static field `tomogramKey = TOMOGRAM_KEY`.
const TOMOGRAM_KEY_FIELD: &str = TOMOGRAM_KEY;
/// Java private static field `joinSamplesKey = JOIN_SAMPLES_KEY`.
const JOIN_SAMPLES_KEY_FIELD: &str = JOIN_SAMPLES_KEY;
/// Java private static field `joinSampleAveragesKey = JOIN_SAMPLE_AVERAGES_KEY`.
const JOIN_SAMPLE_AVERAGES_KEY_FIELD: &str = JOIN_SAMPLE_AVERAGES_KEY;
/// Java private static field `joinKey = JOIN_KEY`.
const JOIN_KEY_FIELD: &str = JOIN_KEY;
/// Java private static field `rotTomogramKey = ROT_TOMOGRAM_KEY`.
const ROT_TOMOGRAM_KEY_FIELD: &str = ROT_TOMOGRAM_KEY;
/// Java private static field `trialJoinKey = TRIAL_JOIN_KEY`.
const TRIAL_JOIN_KEY_FIELD: &str = TRIAL_JOIN_KEY;
/// Java private static field `squeezedVolumeKey = SQUEEZED_VOLUME_KEY`.
const SQUEEZED_VOLUME_KEY_FIELD: &str = SQUEEZED_VOLUME_KEY;
/// Java private static field `reducedFilteredVolumeKey = REDUCED_FILTERED_VOLUME_KEY`.
const REDUCED_FILTERED_VOLUME_KEY_FIELD: &str = REDUCED_FILTERED_VOLUME_KEY;
/// Java private static field `flattenReduceFiltVolKey = FLATTEN_REDUCE_FILT_VOL_KEY`.
const FLATTEN_REDUCE_FILT_VOL_KEY_FIELD: &str = FLATTEN_REDUCE_FILT_VOL_KEY;
/// Java private static field `patchVectorCCCModelKey = PATCH_VECTOR_CCC_MODEL_KEY`.
const PATCH_VECTOR_C_C_C_MODEL_KEY_FIELD: &str = PATCH_VECTOR_CCC_MODEL_KEY;
/// Java private static field `modeledJoinKey = MODELED_JOIN_KEY`.
const MODELED_JOIN_KEY_FIELD: &str = MODELED_JOIN_KEY;
/// Java private static field `transformedModelKey = TRANSFORMED_MODEL_KEY`.
const TRANSFORMED_MODEL_KEY_FIELD: &str = TRANSFORMED_MODEL_KEY;
/// Java private static field `avgVolKey = AVG_VOL_KEY`.
const AVG_VOL_KEY_FIELD: &str = AVG_VOL_KEY;
/// Java private static field `refKey = REF_KEY`.
const REF_KEY_FIELD: &str = REF_KEY;
/// Java private static field `volumeKey = VOLUME_KEY`.
const VOLUME_KEY_FIELD: &str = VOLUME_KEY;
/// Java private static field `testVolumeKey = VOLUME_KEY`.  The source assigns `VOLUME_KEY`, not the constant its own name suggests.
const TEST_VOLUME_KEY_FIELD: &str = VOLUME_KEY;
/// Java private static field `varyingKTestKey = VARYING_K_TEST_KEY`.
const VARYING_K_TEST_KEY_FIELD: &str = VARYING_K_TEST_KEY;
/// Java private static field `varyingIterationTestKey = VARYING_ITERATION_TEST_KEY`.
const VARYING_ITERATION_TEST_KEY_FIELD: &str = VARYING_ITERATION_TEST_KEY;
/// Java private static field `anisotropicDiffusionVolumeKey = ANISOTROPIC_DIFFUSION_VOLUME_KEY`.
const ANISOTROPIC_DIFFUSION_VOLUME_KEY_FIELD: &str = ANISOTROPIC_DIFFUSION_VOLUME_KEY;
/// Java private static field `ctfCorrectionKey = CTF_CORRECTION_KEY`.
const CTF_CORRECTION_KEY_FIELD: &str = CTF_CORRECTION_KEY;
/// Java private static field `erasedFiducialsKey = ERASED_FIDUCIALS_KEY`.
const ERASED_FIDUCIALS_KEY_FIELD: &str = ERASED_FIDUCIALS_KEY;
/// Java private static field `flatVolumeKey = FLAT_VOLUME_KEY`.
const FLAT_VOLUME_KEY_FIELD: &str = FLAT_VOLUME_KEY;
/// Java private static field `fineAligned3dFindKey = FINE_ALIGNED_3D_FIND_KEY`.
const FINE_ALIGNED3D_FIND_KEY_FIELD: &str = FINE_ALIGNED_3D_FIND_KEY;
/// Java private static field `fullVolume3dFindKey = FULL_VOLUME_3D_FIND_KEY`.
const FULL_VOLUME3D_FIND_KEY_FIELD: &str = FULL_VOLUME_3D_FIND_KEY;
/// Java private static field `smoothingAssessmentKey = SMOOTHING_ASSESSMENT_KEY`.
const SMOOTHING_ASSESSMENT_KEY_FIELD: &str = SMOOTHING_ASSESSMENT_KEY;
/// Java private static field `flattenInputKey = FLATTEN_INPUT_KEY`.
const FLATTEN_INPUT_KEY_FIELD: &str = FLATTEN_INPUT_KEY;
/// Java private static field `flattenToolOutputKey = FLATTEN_TOOL_OUTPUT_KEY`.
const FLATTEN_TOOL_OUTPUT_KEY_FIELD: &str = FLATTEN_TOOL_OUTPUT_KEY;
/// Java private static field `sirtKey = SIRT_KEY`.
const SIRT_KEY_FIELD: &str = SIRT_KEY;
/// Java private static field `preblendKey = PREBLEND_KEY`.
const PREBLEND_KEY_FIELD: &str = PREBLEND_KEY;
/// Java private static field `alignedStackKey = ALIGNED_STACK_KEY`.
const ALIGNED_STACK_KEY_FIELD: &str = ALIGNED_STACK_KEY;
/// Java private static field `batchRunTomoStackKey = BATCH_RUN_TOMO_STACK_KEY`.
const BATCH_RUN_TOMO_STACK_KEY_FIELD: &str = BATCH_RUN_TOMO_STACK_KEY;
/// Java private static field `batchRunTomoRecKey = BATCH_RUN_TOMO_REC_KEY`.
const BATCH_RUN_TOMO_REC_KEY_FIELD: &str = BATCH_RUN_TOMO_REC_KEY;
/// Java private static field `batchRunTomoTrimmedVolumeKey = BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY`.
const BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY_FIELD: &str = BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY;
/// Java private static field `multifiltKey = MULTIFILT_KEY`.
const MULTIFILT_KEY_FIELD: &str = MULTIFILT_KEY;
/// Java private static field `ctf3dKey = CTF_3D_KEY`.
const CTF3D_KEY_FIELD: &str = CTF_3D_KEY;
/// Java private static field `subtomoSetupKey = SUBTOMO_SETUP_KEY`.
const SUBTOMO_SETUP_KEY_FIELD: &str = SUBTOMO_SETUP_KEY;

/// Java `ImodManager`.
pub struct ImodManager {
    /// Java superclass `BaseImodManager` state.
    base: BaseImodManager,
    /// Java private field `datasetName`, which defaults to "".
    dataset_name: String,
    /// Java private field `metaDataSet`, which defaults to false.
    meta_data_set: bool,
    /// Java private non-static field `combinedTomogramKey`, which `createPrivateKeys`
    /// sets from the axis type.
    combined_tomogram_key: Option<String>,
}

impl ImodManager {
    /// Java `ImodManager(BaseManager)`.
    pub fn new(
        manager: Option<&'static dyn crate::imod::etomo::base_manager::BaseManager>,
    ) -> ImodManager {
        ImodManager {
            base: BaseImodManager::new(manager),
            dataset_name: String::new(),
            meta_data_set: false,
            combined_tomogram_key: None,
        }
    }

    /// Java `setPreviewMetaData`, for running 3dmod from the SetupDialog.
    pub fn set_preview_meta_data(&mut self, meta_data: &dyn BaseMetaData) {
        if self.meta_data_set {
            return;
        }
        self.base.set_axis_type(meta_data.base().get_axis_type());
        self.dataset_name = meta_data.get_dataset_name().unwrap_or_default();
        self.create_private_keys();
    }

    /// Java `setMetaData(ConstMetaData)`.  If metaDataSet is true and the axisType is
    /// changing from dual to single, combinedTomograms will not be retrievable.  However
    /// the global isOpen() and quit() functions will work on them.
    // TODO(unit): needs etomo/type/ConstMetaData.java - the parameter type - and
    // etomo/process/BaseImodManager.java for `setAxisType`.
    pub fn set_meta_data_const(&mut self, meta_data: Option<Infallible>) {
        let _ = meta_data;
        self.meta_data_set = true;
    }

    /// Java `setMetaData(JoinMetaData)`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the parameter type - and
    // etomo/process/BaseImodManager.java for `setAxisType`.
    pub fn set_meta_data_join(&mut self, meta_data: Option<Infallible>) {
        let _ = meta_data;
        self.meta_data_set = true;
    }

    /// Java `setMetaData(SerialSectionsMetaData)`.
    // TODO(unit): needs etomo/type/SerialSectionsMetaData.java - the parameter type -
    // and etomo/process/BaseImodManager.java for `setAxisType`.
    pub fn set_meta_data_serial_sections(&mut self, meta_data: Option<Infallible>) {
        let _ = meta_data;
        self.meta_data_set = true;
    }

    /// Java `setMetaData(ConstPeetMetaData)`.
    // TODO(unit): needs etomo/type/ConstPeetMetaData.java - the parameter type - and
    // etomo/process/BaseImodManager.java for `setAxisType`.
    pub fn set_meta_data_peet(&mut self, meta_data: Option<Infallible>) {
        let _ = meta_data;
        self.meta_data_set = true;
    }

    /// Java `setMetaData(ParallelMetaData)`.
    // TODO(unit): needs etomo/type/ParallelMetaData.java - the parameter type - and
    // etomo/process/BaseImodManager.java for `setAxisType`.
    pub fn set_meta_data_parallel(&mut self, meta_data: Option<Infallible>) {
        let _ = meta_data;
        self.meta_data_set = true;
    }

    /// Java protected `newImodState(String, String, AxisID, String, File, String[],
    /// String, File[])`, overriding `BaseImodManager.newImodState`.
    // `FileType`-resolved filenames remain at that source boundary; all directly
    // represented ImodState configuration is retained in the factory values.
    #[allow(clippy::too_many_arguments)]
    fn new_imod_state(
        &self,
        key: Option<&str>,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
        file: Option<&std::path::Path>,
        file_name_array: Option<&[Option<String>]>,
        subdir_name: Option<&str>,
        file_list: Option<&[std::path::PathBuf]>,
    ) -> ImodState {
        let key = key.expect("Java newImodState dereferences key");
        if key == RAW_STACK_KEY && axis_id.is_some() {
            return self.new_raw_stack(axis_id, file);
        }
        if key == ERASED_STACK_KEY && axis_id.is_some() {
            return self.new_erased_stack(axis_id);
        }
        if key == COARSE_ALIGNED_KEY && axis_id.is_some() {
            return self.new_coarse_aligned(axis_id);
        }
        if key == FINE_ALIGNED_KEY && axis_id.is_some() {
            return self.new_fine_aligned(axis_id);
        }
        if key == SAMPLE_KEY && axis_id.is_some() {
            return self.new_sample(axis_id);
        }
        if key == FULL_VOLUME_KEY && axis_id.is_some() {
            return self.new_full_volume(axis_id);
        }
        if key == COMBINED_TOMOGRAM_KEY && self.base.equals_axis_type(AxisType::DualAxis) {
            return self.new_combined_tomogram();
        }
        if key == PATCH_VECTOR_MODEL_KEY && self.base.equals_axis_type(AxisType::DualAxis) {
            return self.new_patch_vector_model();
        }
        if key == MATCH_CHECK_KEY && self.base.equals_axis_type(AxisType::DualAxis) {
            return self.new_match_check();
        }
        if key == FIDUCIAL_MODEL_KEY && axis_id.is_some() {
            return self.new_fiducial_model(axis_id);
        }
        if key == SORTED_MODELS_KEY && axis_id.is_some() {
            return self.new_sorted_models(axis_id);
        }
        if key == TRIMMED_VOLUME_KEY {
            return self.new_trimmed_volume();
        }
        if key == TRIAL_TOMOGRAM_KEY && axis_id.is_some() && dataset_name.is_some() {
            return self.new_trial_tomogram(axis_id, dataset_name);
        }
        if key == MTF_FILTER_KEY && axis_id.is_some() {
            return self.new_mtf_filter(axis_id);
        }
        if key == PREVIEW_KEY && axis_id.is_some() {
            return self.new_preview(axis_id, file_extension);
        }
        if key == TOMOGRAM_KEY {
            return self.new_tomogram(file, axis_id);
        }
        if key == JOIN_SAMPLES_KEY {
            return self.new_join_samples();
        }
        if key == JOIN_SAMPLE_AVERAGES_KEY {
            return self.new_join_sample_averages();
        }
        if key == JOIN_KEY {
            return self.new_join();
        }
        if key == ROT_TOMOGRAM_KEY {
            return self.new_rot_tomogram(file);
        }
        if key == TRIAL_JOIN_KEY {
            return self.new_trial_join();
        }
        if key == MODELED_JOIN_KEY {
            return self.new_modeled_join();
        }
        if key == SQUEEZED_VOLUME_KEY {
            return self.new_squeezed_volume();
        }
        if key == REDUCED_FILTERED_VOLUME_KEY {
            return self.new_reduced_filtered_volume(file);
        }
        if key == FLATTEN_REDUCE_FILT_VOL_KEY {
            return self.new_flatten_reduce_filt_vol_output(file);
        }
        if key == PATCH_VECTOR_CCC_MODEL_KEY && self.base.equals_axis_type(AxisType::DualAxis) {
            return self.new_patch_vector_c_c_c_model();
        }
        if key == TRANSFORMED_MODEL_KEY {
            return self.new_transformed_model();
        }
        if key == AVG_VOL_KEY {
            return self.new_avg_vol(file_name_array);
        }
        if key == REF_KEY {
            return self.new_ref(file_name_array);
        }
        if key == VOLUME_KEY {
            return self.new_volume(file);
        }
        if key == TEST_VOLUME_KEY {
            return self.new_test_volume(file);
        }
        if key == VARYING_K_TEST_KEY {
            return self.new_varying_k_test(file_name_array, subdir_name);
        }
        if key == VARYING_ITERATION_TEST_KEY {
            return self.new_varying_iteration_test(file_name_array, subdir_name);
        }
        if key == ANISOTROPIC_DIFFUSION_VOLUME_KEY {
            return self.new_anisotropic_diffusion_volume(file);
        }
        if key == CTF_CORRECTION_KEY && axis_id.is_some() {
            return self.new_ctf_correction(axis_id);
        }
        if key == ERASED_FIDUCIALS_KEY && axis_id.is_some() {
            return self.new_erased_fiducials(axis_id);
        }
        if key == FLAT_VOLUME_KEY && axis_id.is_some() {
            return self.new_flat_volume(axis_id);
        }
        if key == FINE_ALIGNED_3D_FIND_KEY && axis_id.is_some() {
            return self.new_fine_aligned3d_find(axis_id);
        }
        if key == FULL_VOLUME_3D_FIND_KEY && axis_id.is_some() {
            return self.new_full_volume3d_find(axis_id);
        }
        if key == SMOOTHING_ASSESSMENT_KEY && axis_id.is_some() {
            return self.new_smoothing_assessment(axis_id);
        }
        if key == FLATTEN_INPUT_KEY {
            return self.new_flatten_input(file);
        }
        if key == FLATTEN_TOOL_OUTPUT_KEY && axis_id.is_some() {
            return self.new_flatten_tool_output(axis_id);
        }
        if key == SIRT_KEY && axis_id.is_some() {
            return self.new_sirt(file_list, axis_id);
        }
        if key == PREBLEND_KEY && axis_id.is_some() {
            return self.new_preblend(axis_id);
        }
        if key == ALIGNED_STACK_KEY && axis_id.is_some() {
            return self.new_aligned_stack(axis_id);
        }
        if key == BATCH_RUN_TOMO_STACK_KEY {
            return self.new_batch_run_tomo_stack(file);
        }
        if key == BATCH_RUN_TOMO_REC_KEY {
            return self.new_batch_run_tomo_rec(file);
        }
        if key == BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY {
            return self.new_batch_run_tomo_trimmed_volume(file);
        }
        if key == MULTIFILT_KEY && axis_id.is_some() {
            return self.new_multi_filt(file_list, axis_id);
        }
        if key == CTF_3D_KEY && axis_id.is_some() {
            return self.new_ctf3d(axis_id);
        }
        if key == OPEN_OUTPUT_TILT_SERIES_KEY && axis_id.is_some() {
            return self.new_open_output_tilt_series(file, axis_id);
        }
        if key == SUBTOMO_SETUP_KEY {
            return self.new_subtomo_setup(file_name_array, subdir_name);
        }
        if key == ALT_TOMO_SETUP_TOMOGRAM_KEY {
            return self.new_alt_tomo_setup(file, axis_id);
        }
        if key == ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY {
            return self.new_alt_tomo_setup_even_odd(axis_id);
        }
        if key == ALT_TOMO_SETUP_EVEN_ODD_FULL_TOMOGRAM_KEY {
            return self.new_alt_tomo_setup_even_odd_full(axis_id);
        }
        if key == GENERIC_PARALLEL_PROCESS_OUTPUT_FILE_KEY {
            return self.new_generic_parallel_process_output_file(file);
        }
        if key == GENERIC_PARALLEL_PROCESS_OUTPUT_UNKNOWN_FILE_KEY {
            return self.new_generic_parallel_process_output_unknown_file();
        }
        panic!(
            "{key} cannot be created in {} with axisID={axis_id:?}",
            self.base.get_axis_type_string()
        );
    }

    /// Java private `createPrivateKeys`.
    fn create_private_keys(&mut self) {
        self.combined_tomogram_key = Some(
            if self.base.equals_axis_type(AxisType::SingleAxis) {
                FULL_VOLUME_KEY
            } else {
                COMBINED_TOMOGRAM_KEY
            }
            .to_string(),
        );
    }

    /// Java protected `getPrivateKey`, overriding `BaseImodManager.getPrivateKey`.
    fn get_private_key(&self, public_key: &str) -> Option<String> {
        if public_key == COMBINED_TOMOGRAM_KEY {
            self.combined_tomogram_key.clone()
        } else {
            Some(public_key.to_string())
        }
    }
    /// Java private `newRawStack(final AxisID axisID, final File file)`.
    fn new_raw_stack(&self, axis_id: Option<AxisID>, file: Option<&std::path::Path>) -> ImodState {
        let axis_id = axis_id.expect("Java newRawStack dereferences axisID");
        if let Some(file) = file {
            return ImodState::new_with_axis_file_name_file_type(
                None,
                axis_id,
                Some(&file.to_string_lossy()),
                None,
            );
        }
        let mut state = ImodState::new(None, axis_id);
        state.set_load_as_integers();
        state
    }

    /// Java private `newErasedStack(final AxisID axisID)`.
    fn new_erased_stack(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newErasedStack dereferences axisID"),
        )
    }

    /// Java private `newCoarseAligned(final AxisID axisID)`.
    fn new_coarse_aligned(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newCoarseAligned dereferences axisID"),
        )
    }

    /// Java private `newFineAligned(final AxisID axisID)`.
    fn new_fine_aligned(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newFineAligned dereferences axisID"),
        )
    }

    /// Java private `newSample(final AxisID axisID)`.
    fn new_sample(&self, axis_id: Option<AxisID>) -> ImodState {
        let mut state = ImodState::new(None, axis_id.expect("Java newSample dereferences axisID"));
        state.set_initial_mode(MODEL_MODE);
        state
    }

    /// Java private `newFullVolume(final AxisID axisID)`.
    fn new_full_volume(&self, axis_id: Option<AxisID>) -> ImodState {
        let mut state = ImodState::new(
            None,
            axis_id.expect("Java newFullVolume dereferences axisID"),
        );
        state.set_allow_menu_binning_in_z(true);
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newCombinedTomogram()`.
    fn new_combined_tomogram(&self) -> ImodState {
        let mut state = ImodState::new(None, AxisID::Only);
        state.set_allow_menu_binning_in_z(true);
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newPatchVectorModel()`.
    fn new_patch_vector_model(&self) -> ImodState {
        let mut state = ImodState::new_with_model_view_type(None, MODEL_VIEW, AxisID::Only);
        state.set_initial_mode(MODEL_MODE);
        state.set_no_menu_options(true);
        state
    }

    /// Java private `newPatchVectorCCCModel()`.
    fn new_patch_vector_c_c_c_model(&self) -> ImodState {
        let mut state = ImodState::new_with_model_view_type(None, MODV, AxisID::Only);
        state.set_no_menu_options(true);
        state
    }

    /// Java private `newTransformedModel()`.
    fn new_transformed_model(&self) -> ImodState {
        let mut state = ImodState::new_with_model_view_type(None, MODV, AxisID::Only);
        state.set_no_menu_options(true);
        state
    }

    /// Java private `newMatchCheck()`.
    fn new_match_check(&self) -> ImodState {
        let mut state = ImodState::new_with_dataset_model(
            None,
            Some("matchcheck.mat"),
            Some("matchcheck.rec"),
            AxisID::Only,
        );
        state.set_allow_menu_binning_in_z(true);
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newFiducialModel(final AxisID axisID)`.
    fn new_fiducial_model(&self, axis_id: Option<AxisID>) -> ImodState {
        let mut state = ImodState::new_with_model_view_type(
            None,
            MODV,
            axis_id.expect("Java newFiducialModel dereferences axisID"),
        );
        state.set_no_menu_options(true);
        state
    }

    /// Java private `newSortedModels(final AxisID axisID)`.
    fn new_sorted_models(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new_with_model_view_type(
            None,
            MODV,
            axis_id.expect("Java newSortedModels dereferences axisID"),
        )
    }

    /// Java private `newTrimmedVolume()`.
    fn new_trimmed_volume(&self) -> ImodState {
        let mut state = ImodState::new(None, AxisID::Only);
        state.set_allow_menu_binning_in_z(true);
        state
    }

    /// Java private `newTrialTomogram(final AxisID axisID, final String fileName)`.
    fn new_trial_tomogram(&self, axis_id: Option<AxisID>, file_name: Option<&str>) -> ImodState {
        let mut state = ImodState::new_with_file_name(
            None,
            Some(file_name.expect("Java newTrialTomogram dereferences fileName")),
            axis_id.expect("Java newTrialTomogram dereferences axisID"),
        );
        state.set_allow_menu_binning_in_z(true);
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newMtfFilter(final AxisID axisID)`.
    fn new_mtf_filter(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newMtfFilter dereferences axisID"),
        )
    }

    /// Java private `newCtfCorrection(final AxisID axisID)`.
    fn new_ctf_correction(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newCtfCorrection dereferences axisID"),
        )
    }

    /// Java private `newErasedFiducials(final AxisID axisID)`.
    fn new_erased_fiducials(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newErasedFiducials dereferences axisID"),
        )
    }

    /// Java private `newFlatVolume(final AxisID axisID)`.
    fn new_flat_volume(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newFlatVolume dereferences axisID"),
        )
    }

    /// Java private `newFlattenToolOutput(final AxisID axisID)`.
    fn new_flatten_tool_output(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newFlattenToolOutput dereferences axisID"),
        )
    }

    /// Java private `newPreblend(final AxisID axisID)`.
    fn new_preblend(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(None, axis_id.expect("Java newPreblend dereferences axisID"))
    }

    /// Java private `newAlignedStack(final AxisID axisID)`.
    fn new_aligned_stack(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newAlignedStack dereferences axisID"),
        )
    }

    /// Java private `newPreview(final AxisID axisID, final String fileExtension)`.
    fn new_preview(&self, axis_id: Option<AxisID>, file_extension: Option<&str>) -> ImodState {
        let axis_id = axis_id.expect("Java newPreview dereferences axisID");
        let name = format!(
            "{}{}{}",
            self.dataset_name,
            axis_id.get_extension(),
            file_extension.unwrap_or_default()
        );
        let mut state = ImodState::new_with_file_name(None, Some(&name), axis_id);
        state.set_load_as_integers();
        state.set_suppress_save_query();
        state
    }

    /// Java private `newTomogram(final File file, final AxisID axisID)`.
    fn new_tomogram(&self, file: Option<&std::path::Path>, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new_with_file(
            None,
            file,
            axis_id.expect("Java newTomogram dereferences axisID"),
        )
    }

    /// Java private `newJoinSamples()`.
    fn new_join_samples(&self) -> ImodState {
        ImodState::new(None, AxisID::Only)
    }

    /// Java private `newJoinSampleAverages()`.
    fn new_join_sample_averages(&self) -> ImodState {
        ImodState::new(None, AxisID::Only)
    }

    /// Java private `newJoin()`.
    fn new_join(&self) -> ImodState {
        ImodState::new(None, AxisID::Only)
    }

    /// Java private `newRotTomogram(final File file)`.
    fn new_rot_tomogram(&self, file: Option<&std::path::Path>) -> ImodState {
        ImodState::new_with_file(None, file, AxisID::Only)
    }

    /// Java private `newAvgVol(final String[] fileNameArray)`.
    fn new_avg_vol(&self, file_name_array: Option<&[Option<String>]>) -> ImodState {
        let values = file_name_array.map(|files| files.iter().filter_map(Clone::clone).collect());
        let mut state = ImodState::new_with_file_name_array(None, values, AxisID::Only);
        state.set_model_view_type(MODEL_VIEW);
        state.set_open_zap();
        state
    }

    /// Java private `newRef(final String[] fileNameArray)`.
    fn new_ref(&self, file_name_array: Option<&[Option<String>]>) -> ImodState {
        ImodState::new_with_file_name_array(
            None,
            file_name_array.map(|files| files.iter().filter_map(Clone::clone).collect()),
            AxisID::Only,
        )
    }

    /// Java private `newVolume(final File file)`.
    fn new_volume(&self, file: Option<&std::path::Path>) -> ImodState {
        ImodState::new_with_file(None, file, AxisID::Only)
    }

    /// Java private `newTestVolume(final File file)`.
    fn new_test_volume(&self, file: Option<&std::path::Path>) -> ImodState {
        ImodState::new_with_file(None, file, AxisID::Only)
    }

    /// Java private `newVaryingKTest(final String[] fileNameArray, final String subdirName)`.
    fn new_varying_k_test(
        &self,
        file_name_array: Option<&[Option<String>]>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        ImodState::new_with_file_name_array_subdir(
            None,
            file_name_array.map(|files| files.iter().filter_map(Clone::clone).collect()),
            AxisID::Only,
            subdir_name,
        )
    }

    /// Java private `newSirt(final File[] fileList, final AxisID axisID)`.
    fn new_sirt(
        &self,
        file_list: Option<&[std::path::PathBuf]>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let mut state = ImodState::new_with_file_list(
            None,
            file_list.map(|files| files.to_vec()),
            axis_id.expect("Java newSirt dereferences axisID"),
        );
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newMultiFilt(final File[] fileList, final AxisID axisID)`.
    fn new_multi_filt(
        &self,
        file_list: Option<&[std::path::PathBuf]>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let mut state = ImodState::new_with_file_list(
            None,
            file_list.map(|files| files.to_vec()),
            axis_id.expect("Java newMultiFilt dereferences axisID"),
        );
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newCtf3d(final AxisID axisID)`.
    fn new_ctf3d(&self, axis_id: Option<AxisID>) -> ImodState {
        let mut state = ImodState::new(None, axis_id.expect("Java newCtf3d dereferences axisID"));
        state.set_allow_menu_binning_in_z(true);
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newVaryingIterationTest(final String[] fileNameArray, final String subdirName)`.
    fn new_varying_iteration_test(
        &self,
        file_name_array: Option<&[Option<String>]>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        ImodState::new_with_file_name_array_subdir(
            None,
            file_name_array.map(|files| files.iter().filter_map(Clone::clone).collect()),
            AxisID::Only,
            subdir_name,
        )
    }

    /// Java private `newAnisotropicDiffusionVolume(final File file)`.
    fn new_anisotropic_diffusion_volume(&self, file: Option<&std::path::Path>) -> ImodState {
        ImodState::new_with_file(None, file, AxisID::Only)
    }

    /// Java private `newModeledJoin()`.
    fn new_modeled_join(&self) -> ImodState {
        let mut state = ImodState::new(None, AxisID::Only);
        state.set_initial_mode(MODEL_MODE);
        state.set_open_contours(true);
        state
    }

    /// Java private `newTrialJoin()`.
    fn new_trial_join(&self) -> ImodState {
        ImodState::new(None, AxisID::Only)
    }

    /// Java private `newSqueezedVolume()`.
    fn new_squeezed_volume(&self) -> ImodState {
        let mut state = ImodState::new(None, AxisID::Only);
        state.set_allow_menu_binning_in_z(true);
        state
    }

    /// Java private `newReducedFilteredVolume(final File file)`.
    fn new_reduced_filtered_volume(&self, file: Option<&std::path::Path>) -> ImodState {
        let mut state = ImodState::new_with_file(None, file, AxisID::Only);
        state.set_allow_menu_binning_in_z(true);
        state
    }

    /// Java private `newFlattenReduceFiltVolOutput(final File file)`.
    fn new_flatten_reduce_filt_vol_output(&self, file: Option<&std::path::Path>) -> ImodState {
        let mut state = ImodState::new_with_file(None, file, AxisID::Only);
        state.set_allow_menu_binning_in_z(true);
        state
    }

    /// Java private `newFineAligned3dFind(final AxisID axisID)`.
    fn new_fine_aligned3d_find(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newFineAligned3dFind dereferences axisID"),
        )
    }

    /// Java private `newFullVolume3dFind(final AxisID axisID)`.
    fn new_full_volume3d_find(&self, axis_id: Option<AxisID>) -> ImodState {
        let mut state = ImodState::new(
            None,
            axis_id.expect("Java newFullVolume3dFind dereferences axisID"),
        );
        state.set_allow_menu_binning_in_z(true);
        state.set_initial_swap_yz(true);
        state
    }

    /// Java private `newSmoothingAssessment(final AxisID axisID)`.
    fn new_smoothing_assessment(&self, axis_id: Option<AxisID>) -> ImodState {
        let mut state = ImodState::new_with_model_view_type(
            None,
            MODV,
            axis_id.expect("Java newSmoothingAssessment dereferences axisID"),
        );
        state.set_no_menu_options(true);
        state
    }

    /// Java private `newFlattenInput(final File file)`.
    fn new_flatten_input(&self, file: Option<&std::path::Path>) -> ImodState {
        ImodState::new_with_file(None, file, AxisID::Only)
    }

    /// Java private `newBatchRunTomoStack(final File file)`.
    fn new_batch_run_tomo_stack(&self, file: Option<&std::path::Path>) -> ImodState {
        let mut state = ImodState::new_with_file(None, file, AxisID::Only);
        state.set_load_as_integers();
        state
    }

    /// Java private `newBatchRunTomoRec(final File file)`.
    fn new_batch_run_tomo_rec(&self, file: Option<&std::path::Path>) -> ImodState {
        let mut state = ImodState::new_with_file(None, file, AxisID::Only);
        state.set_load_as_integers();
        state.set_swap_yz(true);
        state
    }

    /// Java private `newBatchRunTomoTrimmedVolume(final File file)`.
    fn new_batch_run_tomo_trimmed_volume(&self, file: Option<&std::path::Path>) -> ImodState {
        let mut state = ImodState::new_with_file(None, file, AxisID::Only);
        state.set_load_as_integers();
        state
    }

    /// Java private `newOpenOutputTiltSeries(final File file, final AxisID axisID)`.
    fn new_open_output_tilt_series(
        &self,
        file: Option<&std::path::Path>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let mut state = ImodState::new_with_file(
            None,
            file,
            axis_id.expect("Java newOpenOutputTiltSeries dereferences axisID"),
        );
        state.set_load_as_integers();
        state
    }

    /// Java private `newSubtomoSetup(final String[] fileNameArray, final String subdirName)`.
    fn new_subtomo_setup(
        &self,
        file_name_array: Option<&[Option<String>]>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        ImodState::new_with_file_name_array_subdir(
            None,
            file_name_array.map(|files| files.iter().filter_map(Clone::clone).collect()),
            AxisID::Only,
            subdir_name,
        )
    }

    /// Java private `newAltTomoSetup(final File file, final AxisID axisID)`.
    fn new_alt_tomo_setup(
        &self,
        file: Option<&std::path::Path>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        ImodState::new_with_file(
            None,
            file,
            axis_id.expect("Java newAltTomoSetup dereferences axisID"),
        )
    }

    /// Java private `newAltTomoSetupEvenOdd(final AxisID axisID)`.
    fn new_alt_tomo_setup_even_odd(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newAltTomoSetupEvenOdd dereferences axisID"),
        )
    }

    /// Java private `newAltTomoSetupEvenOddFull(final AxisID axisID)`.
    fn new_alt_tomo_setup_even_odd_full(&self, axis_id: Option<AxisID>) -> ImodState {
        ImodState::new(
            None,
            axis_id.expect("Java newAltTomoSetupEvenOddFull dereferences axisID"),
        )
    }

    /// Java private `newGenericParallelProcessOutputFile(final File file)`.
    fn new_generic_parallel_process_output_file(
        &self,
        file: Option<&std::path::Path>,
    ) -> ImodState {
        ImodState::new_with_file(None, file, AxisID::Only)
    }

    /// Java private `newGenericParallelProcessOutputUnknownFile()`.
    fn new_generic_parallel_process_output_unknown_file(&self) -> ImodState {
        ImodState::new(None, AxisID::Only)
    }

    /// Java package-private `isPerAxis`, overriding `BaseImodManager.isPerAxis`.
    fn is_per_axis(&self, key: &str) -> bool {
        if key == COMBINED_TOMOGRAM_KEY
            || key == PATCH_VECTOR_MODEL_KEY
            || key == MATCH_CHECK_KEY
            || key == TRIMMED_VOLUME_KEY
        {
            return false;
        }
        true
    }

    /// Java package-private `isDualAxisOnly`, overriding
    /// `BaseImodManager.isDualAxisOnly`.
    fn is_dual_axis_only(&self, key: &str) -> bool {
        if key == COMBINED_TOMOGRAM_KEY || key == PATCH_VECTOR_MODEL_KEY || key == MATCH_CHECK_KEY {
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn new_imod_state_routes_raw_stack_to_translated_state() {
        let manager = ImodManager::new(None);
        let state = manager.new_imod_state(
            Some(RAW_STACK_KEY),
            None,
            Some(AxisID::First),
            None,
            Some(Path::new("rawa.st")),
            None,
            None,
            None,
        );
        assert_eq!(state.get_axis_id(), AxisID::First);
        assert_eq!(state.get_dataset_name(), Some("rawa.st".to_string()));
    }

    #[test]
    fn full_volume_factory_keeps_source_view_configuration() {
        let manager = ImodManager::new(None);
        let state = manager.new_imod_state(
            Some(FULL_VOLUME_KEY),
            None,
            Some(AxisID::Second),
            None,
            None,
            None,
            None,
            None,
        );
        assert!(state.is_swap_yz());
        assert!(state.is_initial_swap_yz());
    }

    #[test]
    fn batch_rec_factory_keeps_file_and_integer_swap_configuration() {
        let manager = ImodManager::new(None);
        let state = manager.new_imod_state(
            Some(BATCH_RUN_TOMO_REC_KEY),
            None,
            None,
            None,
            Some(Path::new("rec.mrc")),
            None,
            None,
            None,
        );
        assert_eq!(state.get_dataset_name(), Some("rec.mrc".to_string()));
        assert!(state.is_swap_yz());
    }
}
