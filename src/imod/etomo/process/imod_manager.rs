//! `IMOD/Etomo/src/etomo/process/ImodManager.java`.
//!
//! This class manages the opening, closing and sending of messages to the appropriate
//! imod processes.  This class is state based in the sense that is initialized with
//! MetaData information and uses that information to know which data sets to work with.
//!
//! **Inheritance.**  `ImodManager extends BaseImodManager`: the superclass is the
//! `base` field, reached through `Deref`, and the four methods this class overrides are
//! [`ImodManagerFields`]'s [`BaseImodManagerHooks`] implementation, which the superclass
//! calls back.  The overrides need this class's own fields, so those fields live in the
//! same shared object.
//!
//! **Upstream bugs fixed in translation:** `getPrivateKey(COMBINED_TOMOGRAM_KEY)` returns
//! the null `combinedTomogramKey` before metadata is set, and every caller then
//! dereferences it; the public key is returned instead.  `newAltTomoSetupEvenOdd*`
//! dereference a null `FileType.getFile` result; an empty path is used instead.

#![allow(dead_code)]

use super::base_imod_manager::{BaseImodManager, BaseImodManagerHooks, ImodManagerException};
use super::imod_process::WindowOpenOption;
use super::imod_state::{self, ImodState};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_file_type::ImageFileType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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

/// The fields `ImodManager` declares, which its overrides of the `BaseImodManager`
/// methods read.
pub struct ImodManagerFields {
    /// Java private field `datasetName`, which defaults to "".
    dataset_name: Mutex<String>,
    /// Java private field `metaDataSet`, which defaults to false.
    meta_data_set: Mutex<bool>,
    /// Java private non-static field `combinedTomogramKey`, which `createPrivateKeys`
    /// sets from the axis type.
    combined_tomogram_key: Mutex<Option<String>>,
}

/// Java public final `ImodManager extends BaseImodManager`.
pub struct ImodManager {
    /// Java superclass `BaseImodManager` state.
    base: BaseImodManager,
    /// This class's own fields, shared with `base` as its overrides.
    fields: Arc<ImodManagerFields>,
}

impl std::ops::Deref for ImodManager {
    type Target = BaseImodManager;

    fn deref(&self) -> &BaseImodManager {
        &self.base
    }
}

impl ImodManager {
    /// Java `ImodManager(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> ImodManager {
        let fields = Arc::new(ImodManagerFields {
            dataset_name: Mutex::new(String::new()),
            meta_data_set: Mutex::new(false),
            combined_tomogram_key: Mutex::new(None),
        });
        ImodManager {
            base: BaseImodManager::new(
                manager,
                Arc::clone(&fields) as Arc<dyn BaseImodManagerHooks>,
            ),
            fields,
        }
    }

    /// Java `setPreviewMetaData`.  for running 3dmod from the SetupDialog
    pub fn set_preview_meta_data(&self, meta_data: &dyn BaseMetaData) {
        if *self.fields.meta_data_set.lock().unwrap() {
            return;
        }
        self.base
            .set_axis_type(Some(meta_data.base().get_axis_type()));
        *self.fields.dataset_name.lock().unwrap() =
            meta_data.get_dataset_name().unwrap_or_default();
        self.fields.create_private_keys(&self.base);
    }

    /// Java `setMetaData(ConstMetaData)`.
    pub fn set_meta_data_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        // if metaDataSet is true and the axisType is changing from dual to single,
        // combinedTomograms will not be retrievable. However the global isOpen()
        // and quit() functions will work on them.
        *self.fields.meta_data_set.lock().unwrap() = true;
        self.base.set_axis_type(Some(meta_data.get_axis_type()));
        *self.fields.dataset_name.lock().unwrap() = meta_data.get_dataset_name();
        self.fields.create_private_keys(&self.base);
    }

    /// Java `setMetaData(JoinMetaData)`.
    // TODO(unit): needs etomo/type/JoinMetaData.java - the parameter's declared type;
    // the body reads only `getAxisType` and `getName`, which `BaseMetaData` declares.
    pub fn set_meta_data_join_meta_data(&self, meta_data: &dyn BaseMetaData) {
        *self.fields.meta_data_set.lock().unwrap() = true;
        self.base
            .set_axis_type(Some(meta_data.base().get_axis_type()));
        let dataset_name = meta_data.get_name().unwrap_or_default();
        *self.fields.dataset_name.lock().unwrap() = dataset_name.clone();
        if dataset_name == "" {
            eprintln!("java.lang.IllegalStateException: DatasetName is empty.");
        }
        self.fields.create_private_keys(&self.base);
    }

    /// Java `setMetaData(SerialSectionsMetaData)`.
    // TODO(unit): needs etomo/type/SerialSectionsMetaData.java - the parameter's
    // declared type; the body reads only `BaseMetaData` members.
    pub fn set_meta_data_serial_sections_meta_data(&self, meta_data: &dyn BaseMetaData) {
        *self.fields.meta_data_set.lock().unwrap() = true;
        self.base
            .set_axis_type(Some(meta_data.base().get_axis_type()));
        let dataset_name = meta_data.get_name().unwrap_or_default();
        *self.fields.dataset_name.lock().unwrap() = dataset_name.clone();
        if dataset_name == "" {
            eprintln!("java.lang.IllegalStateException: DatasetName is empty.");
        }
        self.fields.create_private_keys(&self.base);
    }

    /// Java `setMetaData(ConstPeetMetaData)`.
    // TODO(unit): needs etomo/type/ConstPeetMetaData.java - the parameter's declared
    // type; the body reads only `BaseMetaData` members.
    pub fn set_meta_data_const_peet_meta_data(&self, meta_data: &dyn BaseMetaData) {
        *self.fields.meta_data_set.lock().unwrap() = true;
        self.base
            .set_axis_type(Some(meta_data.base().get_axis_type()));
        let dataset_name = meta_data.get_name().unwrap_or_default();
        *self.fields.dataset_name.lock().unwrap() = dataset_name.clone();
        if dataset_name == "" {
            eprintln!("java.lang.IllegalStateException: DatasetName is empty.");
        }
        self.fields.create_private_keys(&self.base);
    }

    /// Java `setMetaData(ParallelMetaData)`.
    // TODO(unit): needs etomo/type/ParallelMetaData.java - the parameter's declared
    // type; the body reads only `BaseMetaData` members.
    pub fn set_meta_data_parallel_meta_data(&self, meta_data: &dyn BaseMetaData) {
        *self.fields.meta_data_set.lock().unwrap() = true;
        self.base
            .set_axis_type(Some(meta_data.base().get_axis_type()));
        let dataset_name = meta_data.get_name().unwrap_or_default();
        *self.fields.dataset_name.lock().unwrap() = dataset_name.clone();
        if dataset_name == "" {
            eprintln!("java.lang.IllegalStateException: DatasetName is empty.");
        }
        self.fields.create_private_keys(&self.base);
    }
}

impl BaseImodManagerHooks for ImodManagerFields {
    /// Java protected `newImodState(String, String, AxisID, String, File, String[],
    /// String, File[])`, overriding `BaseImodManager`'s.
    fn new_imod_state(
        &self,
        base: &BaseImodManager,
        key: &str,
        file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        dataset_name: Option<&str>,
        file: Option<&Path>,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
        file_list: Option<&[PathBuf]>,
    ) -> Result<ImodState, ImodManagerException> {
        if key == RAW_STACK_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_raw_stack(base, axis_id, file));
        }
        if key == ERASED_STACK_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_erased_stack(base, axis_id));
        }
        if key == COARSE_ALIGNED_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_coarse_aligned(base, axis_id));
        }
        if key == FINE_ALIGNED_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_fine_aligned(base, axis_id));
        }
        if key == SAMPLE_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_sample(base, axis_id));
        }
        if key == FULL_VOLUME_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_full_volume(base, axis_id));
        }
        if key == COMBINED_TOMOGRAM_KEY && base.equals_axis_type(AxisType::DualAxis) {
            return Ok(self.new_combined_tomogram(base));
        }
        if key == PATCH_VECTOR_MODEL_KEY && base.equals_axis_type(AxisType::DualAxis) {
            return Ok(self.new_patch_vector_model(base));
        }
        if key == MATCH_CHECK_KEY && base.equals_axis_type(AxisType::DualAxis) {
            return Ok(self.new_match_check(base));
        }
        if key == FIDUCIAL_MODEL_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_fiducial_model(base, axis_id));
        }
        if key == SORTED_MODELS_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_sorted_models(base, axis_id));
        }
        if key == TRIMMED_VOLUME_KEY {
            return Ok(self.new_trimmed_volume(base));
        }
        if key == TRIAL_TOMOGRAM_KEY
            && let (Some(axis_id), Some(dataset_name)) = (axis_id, dataset_name)
        {
            return Ok(self.new_trial_tomogram(base, axis_id, dataset_name));
        }
        if key == MTF_FILTER_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_mtf_filter(base, axis_id));
        }
        if key == PREVIEW_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_preview(base, axis_id, file_extension));
        }
        if key == TOMOGRAM_KEY {
            return Ok(self.new_tomogram(base, file, axis_id));
        }
        if key == JOIN_SAMPLES_KEY {
            return Ok(self.new_join_samples(base));
        }
        if key == JOIN_SAMPLE_AVERAGES_KEY {
            return Ok(self.new_join_sample_averages(base));
        }
        if key == JOIN_KEY {
            return Ok(self.new_join(base));
        }
        if key == ROT_TOMOGRAM_KEY {
            return Ok(self.new_rot_tomogram(base, file));
        }
        if key == TRIAL_JOIN_KEY {
            return Ok(self.new_trial_join(base));
        }
        if key == MODELED_JOIN_KEY {
            return Ok(self.new_modeled_join(base));
        }
        if key == SQUEEZED_VOLUME_KEY {
            return Ok(self.new_squeezed_volume(base));
        }
        if key == REDUCED_FILTERED_VOLUME_KEY {
            return Ok(self.new_reduced_filtered_volume(base, file));
        }
        if key == FLATTEN_REDUCE_FILT_VOL_KEY {
            return Ok(self.new_flatten_reduce_filt_vol_output(base, file));
        }
        if key == PATCH_VECTOR_CCC_MODEL_KEY && base.equals_axis_type(AxisType::DualAxis) {
            return Ok(self.new_patch_vector_ccc_model(base));
        }
        if key == TRANSFORMED_MODEL_KEY {
            return Ok(self.new_transformed_model(base));
        }
        if key == AVG_VOL_KEY {
            return Ok(self.new_avg_vol(base, file_name_array));
        }
        if key == REF_KEY {
            return Ok(self.new_ref(base, file_name_array));
        }
        if key == VOLUME_KEY {
            return Ok(self.new_volume(base, file));
        }
        if key == TEST_VOLUME_KEY {
            return Ok(self.new_test_volume(base, file));
        }
        if key == VARYING_K_TEST_KEY {
            return Ok(self.new_varying_k_test(base, file_name_array, subdir_name));
        }
        if key == VARYING_ITERATION_TEST_KEY {
            return Ok(self.new_varying_iteration_test(base, file_name_array, subdir_name));
        }
        if key == ANISOTROPIC_DIFFUSION_VOLUME_KEY {
            return Ok(self.new_anisotropic_diffusion_volume(base, file));
        }
        if key == CTF_CORRECTION_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_ctf_correction(base, axis_id));
        }
        if key == ERASED_FIDUCIALS_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_erased_fiducials(base, axis_id));
        }
        if key == FLAT_VOLUME_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_flat_volume(base, axis_id));
        }
        if key == FINE_ALIGNED_3D_FIND_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_fine_aligned_3d_find(base, axis_id));
        }
        if key == FULL_VOLUME_3D_FIND_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_full_volume_3d_find(base, axis_id));
        }
        if key == SMOOTHING_ASSESSMENT_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_smoothing_assessment(base, axis_id));
        }
        if key == FLATTEN_INPUT_KEY {
            return Ok(self.new_flatten_input(base, file));
        }
        if key == FLATTEN_TOOL_OUTPUT_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_flatten_tool_output(base, axis_id));
        }
        if key == SIRT_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_sirt(base, file_list, axis_id));
        }
        if key == PREBLEND_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_preblend(base, axis_id));
        }
        if key == ALIGNED_STACK_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_aligned_stack(base, axis_id));
        }
        if key == BATCH_RUN_TOMO_STACK_KEY {
            return Ok(self.new_batch_run_tomo_stack(base, file));
        }
        if key == BATCH_RUN_TOMO_REC_KEY {
            return Ok(self.new_batch_run_tomo_rec(base, file));
        }
        if key == BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY {
            return Ok(self.new_batch_run_tomo_trimmed_volume(base, file));
        }
        if key == MULTIFILT_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_multi_filt(base, file_list, axis_id));
        }
        if key == CTF_3D_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_ctf_3d(base, axis_id));
        }
        if key == OPEN_OUTPUT_TILT_SERIES_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(self.new_open_output_tilt_series(base, file, axis_id));
        }
        if key == SUBTOMO_SETUP_KEY {
            return Ok(self.new_subtomo_setup(base, file_name_array, subdir_name));
        }
        if key == ALT_TOMO_SETUP_TOMOGRAM_KEY {
            return Ok(self.new_alt_tomo_setup(base, file, axis_id));
        }
        if key == ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY {
            return Ok(self.new_alt_tomo_setup_even_odd(base, axis_id));
        }
        if key == ALT_TOMO_SETUP_EVEN_ODD_FULL_TOMOGRAM_KEY {
            return Ok(self.new_alt_tomo_setup_even_odd_full(base, axis_id));
        }
        if key == GENERIC_PARALLEL_PROCESS_OUTPUT_FILE_KEY {
            return Ok(self.new_generic_parallel_process_output_file(base, file));
        }
        if key == GENERIC_PARALLEL_PROCESS_OUTPUT_UNKNOWN_FILE_KEY {
            return Ok(self.new_generic_parallel_process_output_unknown_file(base));
        }
        Err(ImodManagerException::Runtime(format!(
            "{} cannot be created in {} with axisID={}",
            key,
            base.get_axis_type_string(),
            match axis_id {
                Some(axis_id) => axis_id.get_extension(),
                None => "null".to_string(),
            }
        )))
    }

    /// Java protected `getPrivateKey`, overriding `BaseImodManager`'s.
    ///
    /// Before `createPrivateKeys` has run, the source returns the null
    /// `combinedTomogramKey` and its callers throw on it.  Fixed in translation: the
    /// public key is returned until a private one exists.
    fn get_private_key(&self, public_key: &str) -> String {
        if public_key == COMBINED_TOMOGRAM_KEY {
            match self.combined_tomogram_key.lock().unwrap().as_ref() {
                Some(combined_tomogram_key) => combined_tomogram_key.clone(),
                None => public_key.to_string(),
            }
        } else {
            public_key.to_string()
        }
    }

    /// Java `isPerAxis`, overriding `BaseImodManager`'s.
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

    /// Java `isDualAxisOnly`, overriding `BaseImodManager`'s.
    fn is_dual_axis_only(&self, key: &str) -> bool {
        if key == COMBINED_TOMOGRAM_KEY || key == PATCH_VECTOR_MODEL_KEY || key == MATCH_CHECK_KEY {
            return true;
        }
        false
    }
}

/// The private methods of `ImodManager`.  They are on the fields object because the
/// overridden `newImodState` calls them; `base` is the superclass half (`manager`,
/// `equalsAxisType`).
impl ImodManagerFields {
    /// Java private `createPrivateKeys`.
    fn create_private_keys(&self, base: &BaseImodManager) {
        if base.equals_axis_type(AxisType::SingleAxis) {
            *self.combined_tomogram_key.lock().unwrap() = Some(FULL_VOLUME_KEY.to_string());
        } else {
            *self.combined_tomogram_key.lock().unwrap() = Some(COMBINED_TOMOGRAM_KEY.to_string());
        }
    }

    /// Java private `newRawStack`.
    fn new_raw_stack(
        &self,
        base: &BaseImodManager,
        axis_id: AxisID,
        file: Option<&Path>,
    ) -> ImodState {
        let imod_state;
        match file {
            None => {
                imod_state = ImodState::new_base_manager_file_axis_id(
                    base.manager,
                    file_type::CLASS
                        .raw_stack
                        .get_file(Some(base.manager), Some(axis_id))
                        .as_deref(),
                    Some(axis_id),
                );
                imod_state.set_load_as_integers();
            }
            Some(file) => {
                // `File.getName()`.
                let name = file
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                imod_state = ImodState::new_base_manager_axis_id_string_file_type(
                    base.manager,
                    Some(axis_id),
                    &name,
                    &file_type::CLASS.raw_stack,
                );
            }
        }
        imod_state
    }

    /// Java private `newErasedStack`.
    fn new_erased_stack(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.fixed_xrays_stack,
            Some(axis_id),
        )
    }

    /// Java private `newCoarseAligned`.
    fn new_coarse_aligned(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.prealigned_stack,
            Some(axis_id),
        )
    }

    /// Java private `newFineAligned`.
    fn new_fine_aligned(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id_file_type(
            base.manager,
            axis_id,
            &file_type::CLASS.aligned_stack,
        )
    }

    /// Java private `newSample`.
    fn new_sample(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_axis_id_file_type_file_type_file_type_string_string(
                base.manager,
                axis_id,
                &file_type::CLASS.top_sample,
                &file_type::CLASS.middle_sample,
                &file_type::CLASS.bottom_sample,
                "tomopitch",
                ".mod",
            );
        imod_state.set_initial_mode(imod_state::MODEL_MODE);
        imod_state
    }

    /// Java private `newFullVolume`.
    fn new_full_volume(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.tilt_output, /*was: datasetName + "_full.rec"*/
            Some(axis_id),
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newCombinedTomogram`.
    fn new_combined_tomogram(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.combined_volume, /*was: "sum.rec"*/
            Some(AxisID::Only),
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newPatchVectorModel`.
    fn new_patch_vector_model(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_string_int_axis_id_window_open_option_file(
            base.manager,
            dataset_files::PATCH_VECTOR_MODEL,
            imod_state::MODEL_VIEW,
            Some(AxisID::Only),
            WindowOpenOption::ImodvObjects,
            Some(dataset_files::get_patch_vector_model(base.manager).as_path()),
        );
        imod_state.set_initial_mode(imod_state::MODEL_MODE);
        imod_state.set_no_menu_options(true);
        imod_state
    }

    /// Java private `newPatchVectorCCCModel`.
    fn new_patch_vector_ccc_model(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_string_int_axis_id_window_open_option_file(
            base.manager,
            dataset_files::PATCH_VECTOR_CCC_MODEL,
            imod_state::MODV,
            Some(AxisID::Only),
            WindowOpenOption::ImodvObjects,
            Some(dataset_files::get_patch_vector_ccc_model(base.manager).as_path()),
        );
        imod_state.set_no_menu_options(true);
        imod_state
    }

    /// Java private `newTransformedModel`.
    fn new_transformed_model(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_string_int_axis_id(
            base.manager,
            &dataset_files::get_refine_aligned_model_file_name(base.manager),
            imod_state::MODV,
            Some(AxisID::Only),
        );
        imod_state.set_no_menu_options(true);
        imod_state
    }

    /// Java private `newMatchCheck`.  No need to add file name style because no new
    /// files are being created.  Deprecated: files are no longer created by IMOD.
    fn new_match_check(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_string_string_axis_id(
            base.manager,
            "matchcheck.mat",
            "matchcheck.rec",
            Some(AxisID::Only),
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newFiducialModel`.
    fn new_fiducial_model(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_int_axis_id(base.manager, imod_state::MODV, Some(axis_id));
        imod_state.set_no_menu_options(true);
        imod_state
    }

    /// Java private `newSortedModels`.
    fn new_sorted_models(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_int_axis_id(base.manager, imod_state::MODV, Some(axis_id));
        imod_state.add_window_open_option(WindowOpenOption::ModelEdit);
        imod_state
    }

    /// Java private `newTrimmedVolume`.
    fn new_trimmed_volume(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.trim_vol_output, /*was: datasetName + ".rec"*/
            Some(AxisID::Only),
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state
    }

    /// Java private `newTrialTomogram`.
    fn new_trial_tomogram(
        &self,
        base: &BaseImodManager,
        axis_id: AxisID,
        file_name: &str,
    ) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_string_axis_id(base.manager, file_name, Some(axis_id));
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newMtfFilter`.
    fn new_mtf_filter(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        // new ImodState(manager, axisID, datasetName,
        // "_filt.ali");
        ImodState::new_base_manager_axis_id_file_type(
            base.manager,
            axis_id,
            &file_type::CLASS.mtf_filtered_stack,
        )
    }

    /// Java private `newCtfCorrection`.
    fn new_ctf_correction(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id_file_type(
            base.manager,
            axis_id,
            &file_type::CLASS.ctf_corrected_stack,
        )
    }

    /// Java private `newErasedFiducials`.
    fn new_erased_fiducials(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id_file_type(
            base.manager,
            axis_id,
            &file_type::CLASS.erased_beads_stack,
        )
    }

    /// Java private `newFlatVolume`.
    fn new_flat_volume(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_file_axis_id(
            base.manager,
            ImageFileType::FlattenOutput
                .get_file(base.manager)
                .as_deref(),
            Some(axis_id),
        )
    }

    /// Java private `newFlattenToolOutput`.
    fn new_flatten_tool_output(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id_string_file_type(
            base.manager,
            Some(axis_id),
            &file_type::CLASS
                .flatten_tool_output
                .get_file_name(Some(base.manager), Some(axis_id))
                .unwrap_or_default(),
            &file_type::CLASS.flatten_tool_output,
        )
    }

    /// Java private `newPreblend`.
    fn new_preblend(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id_string_file_type(
            base.manager,
            Some(axis_id),
            &file_type::CLASS
                .preblend_output_mrc
                .get_file_name(Some(base.manager), Some(axis_id))
                .unwrap_or_default(),
            &file_type::CLASS.preblend_output_mrc,
        )
    }

    /// Java private `newAlignedStack`.
    fn new_aligned_stack(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id_string_file_type(
            base.manager,
            Some(axis_id),
            &file_type::CLASS
                .aligned_stack_mrc
                .get_file_name(Some(base.manager), Some(axis_id))
                .unwrap_or_default(),
            &file_type::CLASS.aligned_stack_mrc,
        )
    }

    /// Java private `newPreview`.  A null `fileExtension` concatenates as "null", as in
    /// the source.
    fn new_preview(
        &self,
        base: &BaseImodManager,
        axis_id: AxisID,
        file_extension: Option<&str>,
    ) -> ImodState {
        let axis_extension = axis_id.get_extension();
        if axis_extension == "ERROR" {
            // Unreachable: `AxisID` has exactly the three values that have extensions.
            unreachable!("{}", axis_id);
        }
        let imod_state = ImodState::new_base_manager_string_axis_id(
            base.manager,
            &(self.dataset_name.lock().unwrap().clone()
                + &axis_extension
                + file_extension.unwrap_or("null")),
            Some(axis_id),
        );
        imod_state.set_load_as_integers();
        imod_state.set_suppress_save_query();
        imod_state
    }

    /// Java private `newTomogram`.
    fn new_tomogram(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, axis_id)
    }

    /// Java private `newJoinSamples`.
    fn new_join_samples(&self, base: &BaseImodManager) -> ImodState {
        if *self.dataset_name.lock().unwrap() == "" {
            eprintln!("java.lang.IllegalStateException: DatasetName is empty.");
            eprintln!("manager={}", base.manager);
        }
        ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.join_sample, /*was: datasetName + ".sample"*/
            Some(AxisID::Only),
        )
    }

    /// Java private `newJoinSampleAverages`.
    fn new_join_sample_averages(&self, base: &BaseImodManager) -> ImodState {
        if *self.dataset_name.lock().unwrap() == "" {
            eprintln!("java.lang.IllegalStateException: DatasetName is empty.");
            eprintln!("manager={}", base.manager);
        }
        ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.join_sample_averages, /*was: datasetName + ".sampavg"*/
            Some(AxisID::Only),
        )
    }

    /// Java private `newJoin`.
    fn new_join(&self, base: &BaseImodManager) -> ImodState {
        ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.join, /*was: datasetName + ".join"*/
            Some(AxisID::Only),
        )
    }

    /// Java private `newRotTomogram`.
    fn new_rot_tomogram(&self, base: &BaseImodManager, file: Option<&Path>) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only))
    }

    /// Java private `newAvgVol`.
    fn new_avg_vol(&self, base: &BaseImodManager, file_name_array: Option<&[String]>) -> ImodState {
        let imod_state = ImodState::new_base_manager_string_array_axis_id(
            base.manager,
            file_name_array.map(<[String]>::to_vec),
            Some(AxisID::Only),
        );
        imod_state.set_model_view_type(imod_state::MODEL_VIEW);
        imod_state.set_open_zap();
        imod_state.add_window_open_option(WindowOpenOption::Isosurface);
        imod_state
    }

    /// Java private `newRef`.
    fn new_ref(&self, base: &BaseImodManager, file_name_array: Option<&[String]>) -> ImodState {
        ImodState::new_base_manager_string_array_axis_id(
            base.manager,
            file_name_array.map(<[String]>::to_vec),
            Some(AxisID::Only),
        )
    }

    /// Java private `newVolume`.
    fn new_volume(&self, base: &BaseImodManager, file: Option<&Path>) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only))
    }

    /// Java private `newTestVolume`.
    fn new_test_volume(&self, base: &BaseImodManager, file: Option<&Path>) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only))
    }

    /// Java private `newVaryingKTest`.
    fn new_varying_k_test(
        &self,
        base: &BaseImodManager,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        ImodState::new_base_manager_string_array_axis_id_string(
            base.manager,
            file_name_array.map(<[String]>::to_vec),
            Some(AxisID::Only),
            subdir_name,
        )
    }

    /// Java private `newSirt`.
    fn new_sirt(
        &self,
        base: &BaseImodManager,
        file_list: Option<&[PathBuf]>,
        axis_id: AxisID,
    ) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_array_axis_id(
            base.manager,
            file_list.map(<[PathBuf]>::to_vec),
            Some(axis_id),
        );
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newMultiFilt`.
    fn new_multi_filt(
        &self,
        base: &BaseImodManager,
        file_list: Option<&[PathBuf]>,
        axis_id: AxisID,
    ) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_array_axis_id(
            base.manager,
            file_list.map(<[PathBuf]>::to_vec),
            Some(axis_id),
        );
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newCtf3d`.
    fn new_ctf_3d(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state = ImodState::new_base_manager_axis_id_string_file_type(
            base.manager,
            Some(axis_id),
            &file_type::CLASS
                .ctf_3d_output
                .get_file_name(Some(base.manager), Some(axis_id))
                .unwrap_or_default(),
            &file_type::CLASS.ctf_3d_output,
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newVaryingIterationTest`.
    fn new_varying_iteration_test(
        &self,
        base: &BaseImodManager,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        ImodState::new_base_manager_string_array_axis_id_string(
            base.manager,
            file_name_array.map(<[String]>::to_vec),
            Some(AxisID::Only),
            subdir_name,
        )
    }

    /// Java private `newAnisotropicDiffusionVolume`.
    fn new_anisotropic_diffusion_volume(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
    ) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only))
    }

    /// Java private `newModeledJoin`.
    fn new_modeled_join(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.modeled_join, /*datasetName + "_modeled.join"*/
            Some(AxisID::Only),
        );
        imod_state.set_initial_mode(imod_state::MODEL_MODE);
        imod_state.set_open_contours(true);
        imod_state
    }

    /// Java private `newTrialJoin`.
    fn new_trial_join(&self, base: &BaseImodManager) -> ImodState {
        ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.trial_join, /*datasetName + "_trial.join"*/
            Some(AxisID::Only),
        )
    }

    /// Java private `newSqueezedVolume`.
    fn new_squeezed_volume(&self, base: &BaseImodManager) -> ImodState {
        let imod_state = ImodState::new_base_manager_file_type_axis_id(
            base.manager,
            &file_type::CLASS.squeeze_vol_output, /*was:datasetName + ".sqz"*/
            Some(AxisID::Only),
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state
    }

    /// Java private `newReducedFilteredVolume`.
    fn new_reduced_filtered_volume(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
    ) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only));
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state
    }

    /// Java private `newFlattenReduceFiltVolOutput`.
    fn new_flatten_reduce_filt_vol_output(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
    ) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only));
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state
    }

    /// Java private `newFineAligned3dFind`.
    fn new_fine_aligned_3d_find(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        // FileType.NEWST_3D_FIND_OUTPUT is the same as FileType.BLEND_3D_FIND_OUTPUT.
        ImodState::new_base_manager_axis_id_string_file_type(
            base.manager,
            Some(axis_id),
            &file_type::CLASS
                .newst_or_blend_3d_find_output
                .get_file_name(Some(base.manager), Some(axis_id))
                .unwrap_or_default(),
            &file_type::CLASS.newst_or_blend_3d_find_output,
        )
    }

    /// Java private `newFullVolume3dFind`.
    fn new_full_volume_3d_find(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state = ImodState::new_base_manager_axis_id_string_file_type(
            base.manager,
            Some(axis_id),
            &file_type::CLASS
                .tilt_3d_find_output
                .get_file_name(Some(base.manager), Some(axis_id))
                .unwrap_or_default(),
            &file_type::CLASS.tilt_3d_find_output,
        );
        imod_state.set_allow_menu_binning_in_z(true);
        imod_state.set_initial_swap_yz(true);
        imod_state
    }

    /// Java private `newSmoothingAssessment`.
    fn new_smoothing_assessment(&self, base: &BaseImodManager, axis_id: AxisID) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_int_axis_id(base.manager, imod_state::MODV, Some(axis_id));
        imod_state.set_no_menu_options(true);
        imod_state.add_window_open_option(WindowOpenOption::ObjectList);
        imod_state
    }

    /// Java private `newFlattenInput`.
    fn new_flatten_input(&self, base: &BaseImodManager, file: Option<&Path>) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only))
    }

    /// Java private `newBatchRunTomoStack`.
    fn new_batch_run_tomo_stack(&self, base: &BaseImodManager, file: Option<&Path>) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only));
        imod_state.set_load_as_integers();
        imod_state
    }

    /// Java private `newBatchRunTomoRec`.
    fn new_batch_run_tomo_rec(&self, base: &BaseImodManager, file: Option<&Path>) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only));
        imod_state.set_load_as_integers();
        imod_state.set_swap_yz(true);
        imod_state
    }

    /// Java private `newBatchRunTomoTrimmedVolume`.
    fn new_batch_run_tomo_trimmed_volume(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
    ) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only));
        imod_state.set_load_as_integers();
        imod_state
    }

    /// Java private `newOpenOutputTiltSeries`.
    fn new_open_output_tilt_series(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
        axis_id: AxisID,
    ) -> ImodState {
        let imod_state =
            ImodState::new_base_manager_file_axis_id(base.manager, file, Some(axis_id));
        imod_state.set_load_as_integers();
        imod_state
    }

    /// Java private `newSubtomoSetup`.
    fn new_subtomo_setup(
        &self,
        base: &BaseImodManager,
        file_name_array: Option<&[String]>,
        subdir_name: Option<&str>,
    ) -> ImodState {
        ImodState::new_base_manager_string_array_axis_id_string(
            base.manager,
            file_name_array.map(<[String]>::to_vec),
            Some(AxisID::Only),
            subdir_name,
        )
    }

    /// Java private `newAltTomoSetup`.
    fn new_alt_tomo_setup(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, axis_id)
    }

    /// Java private `newAltTomoSetupEvenOdd`.
    fn new_alt_tomo_setup_even_odd(
        &self,
        base: &BaseImodManager,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let file_list = vec![
            PathBuf::from(
                file_type::CLASS
                    .alt_stack_even_tomogram
                    .get_file(Some(base.manager), axis_id)
                    .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                    .unwrap_or_default(),
            ),
            PathBuf::from(
                file_type::CLASS
                    .alt_stack_odd_tomogram
                    .get_file(Some(base.manager), axis_id)
                    .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                    .unwrap_or_default(),
            ),
        ];
        ImodState::new_base_manager_file_array_axis_id(base.manager, Some(file_list), axis_id)
    }

    /// Java private `newAltTomoSetupEvenOddFull`.
    fn new_alt_tomo_setup_even_odd_full(
        &self,
        base: &BaseImodManager,
        axis_id: Option<AxisID>,
    ) -> ImodState {
        let file_list = vec![
            PathBuf::from(
                file_type::CLASS
                    .alt_stack_even_full_tomogram
                    .get_file(Some(base.manager), axis_id)
                    .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                    .unwrap_or_default(),
            ),
            PathBuf::from(
                file_type::CLASS
                    .alt_stack_odd_full_tomogram
                    .get_file(Some(base.manager), axis_id)
                    .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
                    .unwrap_or_default(),
            ),
        ];
        ImodState::new_base_manager_file_array_axis_id(base.manager, Some(file_list), axis_id)
    }

    /// Java private `newGenericParallelProcessOutputFile`.
    fn new_generic_parallel_process_output_file(
        &self,
        base: &BaseImodManager,
        file: Option<&Path>,
    ) -> ImodState {
        ImodState::new_base_manager_file_axis_id(base.manager, file, Some(AxisID::Only))
    }

    /// Java private `newGenericParallelProcessOutputUnknownFile`.
    fn new_generic_parallel_process_output_unknown_file(
        &self,
        base: &BaseImodManager,
    ) -> ImodState {
        ImodState::new_base_manager_axis_id(base.manager, Some(AxisID::Only))
    }
}
