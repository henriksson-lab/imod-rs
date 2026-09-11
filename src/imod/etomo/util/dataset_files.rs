//! `IMOD/Etomo/src/etomo/util/DatasetFiles.java`.
//!
//! Partially deprecated utility for building file names.  Still in use, but new file
//! names should be added to `FileType`.
//!
//! **Frontier.**  Almost every member takes a `BaseManager` and resolves the name
//! through `manager.getPropertyUserDir()` and `manager.getBaseMetaData().getName()`, or
//! through the untranslated instance half of `etomo/type/FileType.java`.  Those are
//! grouped under one `TODO(unit)` marker each below; the constants and the members that
//! stand on their own are translated.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::utilities;

/// Java `TILT_FILE_EXT`.
pub const TILT_FILE_EXT: &str = ".tlt";
/// Java `MATLAB_PARAM_FILE_EXT`.
pub const MATLAB_PARAM_FILE_EXT: &str = ".prm";
/// Java `ROTATED_TOMO_EXT` (deprecated 6/17/19).
pub const ROTATED_TOMO_EXT: &str = ".rot";
/// Java `COMSCRIPT_EXT`.
pub const COMSCRIPT_EXT: &str = ".com";
/// Java `TOMO_EXT` (deprecated 6/17/19).
pub const TOMO_EXT: &str = ".rec";
/// Java `MODEL_EXT`.
pub const MODEL_EXT: &str = ".mod";
/// Java `PATCH_VECTOR_STRING`.
const PATCH_VECTOR_STRING: &str = "patch_vector";
/// Java `PATCH_VECTOR_MODEL`.
pub const PATCH_VECTOR_MODEL: &str = "patch_vector.mod";
/// Java `PATCH_VECTOR_CCC_MODEL`.
pub const PATCH_VECTOR_CCC_MODEL: &str = "patch_vector_ccc.mod";
/// Java `LOG_EXT`.
pub const LOG_EXT: &str = ".log";
/// Java `BACKUP_CHAR`.
pub const BACKUP_CHAR: char = '~';
/// Java `TRANSFER_FID_LOG`.
pub const TRANSFER_FID_LOG: &str = "transferfid.log";
/// Java `PATCH_OUT`.
pub const PATCH_OUT: &str = "patch.out";
/// Java `VOLCOMBINE_START_LOG`:
/// `ProcessName.VOLCOMBINE.toString() + "-start" + LOG_EXT`.
pub static VOLCOMBINE_START_LOG: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| ProcessName::VOLCOMBINE.to_string() + "-start" + LOG_EXT);
/// Java `PROCESSCHUNKS_FINISH_NAME`.
pub const PROCESSCHUNKS_FINISH_NAME: &str = "-finish";
/// Java `VOLCOMBINE_FINISH_LOG`:
/// `ProcessName.VOLCOMBINE.toString() + PROCESSCHUNKS_FINISH_NAME + LOG_EXT`.
pub static VOLCOMBINE_FINISH_LOG: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    ProcessName::VOLCOMBINE.to_string() + PROCESSCHUNKS_FINISH_NAME + LOG_EXT
});
/// Java `JOIN_EXT` (deprecated 6/17/19).
pub const JOIN_EXT: &str = ".join";
/// Java `REFINE_NAME`.
pub const REFINE_NAME: &str = "_refine";
/// Java `XFJOINTOMO_LOG`.
pub const XFJOINTOMO_LOG: &str = "xfjointomo.log";
/// Java `XG_EXT`.
const XG_EXT: &str = ".xg";
/// Java `FULL_ALIGNED_EXT` (deprecated 5/9/2019).
const FULL_ALIGNED_EXT: &str = ".ali";
/// Java `CTF_PLOTTER_EXT`.
pub const CTF_PLOTTER_EXT: &str = ".defocus";
/// Java `SIMPLE_DEFOCUS_EXT`.
pub const SIMPLE_DEFOCUS_EXT: &str = "_simple.defocus";
/// Java `CTF_CORRECTION_EXT` (deprecated 5/9/2019).
const CTF_CORRECTION_EXT: &str = "_ctfcorr.ali";
/// Java `FIDUCIAL_MODEL_EXT`.
pub const FIDUCIAL_MODEL_EXT: &str = ".fid";
/// Java `ERASE_EXT` (deprecated 5/9/2019).
const ERASE_EXT: &str = "_erase";
/// Java `FLATTEN_WARP_EXT`.
const FLATTEN_WARP_EXT: &str = "_flat";
/// Java `XF_EXT`.
const XF_EXT: &str = ".xf";

/// Java private static field `calibrationDir`, initialised to null.  It is a class
/// variable the two directory getters cache into, so it carries its own lock.
static CALIBRATION_DIR: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

/// Java private static field `distortionDir`, initialised to null.
static DISTORTION_DIR: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

// Java `calibrationDir` and `distortionDir` are the caches of `getCalibrationDir` and
// `getDistortionDir`, both of which are untranslated below.

/// Java `getOriginalStack`.
pub fn get_original_stack(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> Option<std::path::PathBuf> {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    file_type::CLASS
        .original_raw_stack
        .get_file(Some(manager), axis_id)
}

/// Java `getTomogramName`.
pub fn get_tomogram_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> Option<String> {
    file_type::CLASS
        .tilt_output
        .get_file_name(Some(manager), axis_id)
}

/// Java `getTomogram`.
pub fn get_tomogram(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> Option<std::path::PathBuf> {
    file_type::CLASS
        .tilt_output
        .get_file(Some(manager), axis_id)
}

/// Java `getPrealignedStackName`.
pub fn get_prealigned_stack_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> Option<String> {
    // Java reads `manager.getBaseMetaData()` into a local it never uses.
    let _meta_data = manager.get_base_meta_data();
    file_type::CLASS
        .prealigned_stack
        .get_file_name(Some(manager), axis_id)
}

/// Java `getJoinFileName`.
pub fn get_join_file_name(trial: bool, manager: &'static dyn BaseManager) -> Option<String> {
    if trial {
        return file_type::CLASS
            .trial_join
            .get_file_name(Some(manager), Some(AxisID::Only));
    }
    file_type::CLASS
        .join
        .get_file_name(Some(manager), Some(AxisID::Only))
}

/// Java `getJoinFile`.
pub fn get_join_file(trial: bool, manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::Path::new(manager.get_property_user_dir().as_deref().unwrap_or("null"))
        .join(get_join_file_name(trial, manager).unwrap_or("null".to_string()))
}

/// Java `getModeledJoinFileName`.
pub fn get_modeled_join_file_name(manager: &'static dyn BaseManager) -> Option<String> {
    file_type::CLASS
        .modeled_join
        .get_file_name(Some(manager), Some(AxisID::Only))
}

/// Java `getModeledJoinFile(JoinManager)` (deprecated 7/13/2020, unnecessary function).
pub fn get_modeled_join_file(manager: &'static JoinManager) -> Option<std::path::PathBuf> {
    file_type::CLASS
        .modeled_join
        .get_file(Some(manager), Some(AxisID::Only))
}

// TODO(unit): needs etomo/logic/DatasetTool.java - Java `getStackName(String, AxisType,
// AxisID)` appends `DatasetTool.STANDARD_DATASET_EXT`.

/// Java `getAutodoc(File, String)`.
pub fn get_autodoc(dir: &std::path::Path, name: Option<&str>) -> std::path::PathBuf {
    dir.join(get_autodoc_name(name))
}

/// Java package-private `getAutodocName(String)`.
pub fn get_autodoc_name(name: Option<&str>) -> String {
    if autodoc_factory::ends_with_autodoc_extension(name) {
        // Java returns the field itself, so a null name reaches `new File(dir, null)`.
        return name.unwrap_or("null").to_string();
    }
    name.unwrap_or("null").to_string() + &autodoc_factory::extension::DEFAULT.to_string()
}

/// Java `isRotatedTomogram`.
pub fn is_rotated_tomogram(tomogram: &std::path::Path) -> bool {
    let tomogram_name = utilities::java_io_file_get_name(&tomogram.to_string_lossy());
    // Java's substring throws StringIndexOutOfBoundsException when there is no ".";
    // lastIndexOf returns -1 and substring(-1) is out of range.
    let index = match tomogram_name.rfind('.') {
        None => return false,
        Some(index) => index,
    };
    if &tomogram_name[index..] == ROTATED_TOMO_EXT {
        return true;
    }
    false
}

/// Java `getCommandsFileName`.
pub fn get_commands_file_name(
    subdir_name: Option<&str>,
    root_name: &str,
    suffix: Option<&str>,
) -> String {
    let commands_file_name = root_name.to_string() + suffix.unwrap_or("") + ".cmds";
    let subdir_name = match subdir_name {
        None => return commands_file_name,
        Some(subdir_name) => subdir_name,
    };
    if crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(
        subdir_name,
    ) {
        return commands_file_name;
    }
    // `new File(subdirName, commandsFileName).getPath()`
    utilities::java_io_file_new(subdir_name, &commands_file_name)
}

/// Java `correctAxisID(AxisType, AxisID)`.
pub fn correct_axis_id(axis_type: Option<AxisType>, axis_id: Option<AxisID>) -> Option<AxisID> {
    if axis_type == Some(AxisType::DualAxis) && axis_id == Some(AxisID::Only) {
        return Some(AxisID::First);
    }
    if axis_type == Some(AxisType::SingleAxis) && axis_id == Some(AxisID::First) {
        return Some(AxisID::Only);
    }
    if axis_type.is_none() || axis_type == Some(AxisType::NotSet) {
        panic!(
            "java.lang.IllegalStateException: AxisType is not set.  AxisType must be set before getting a dataset file name containing the axisID extension."
        );
    }
    axis_id
}

/// Java `getMatlabParamFile(String, String)`.
pub fn get_matlab_param_file(directory: &str, name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        directory,
        &(name.to_string() + MATLAB_PARAM_FILE_EXT),
    ))
}

/// Java `getParallelDataFileName`.
pub fn get_parallel_data_file_name(root_name: &str) -> String {
    root_name.to_string() + DataFileType::Parallel.extension().unwrap_or("null")
}

/// Java `getBatchRunTomoDataFileName`.
pub fn get_batch_run_tomo_data_file_name(root_name: &str) -> String {
    root_name.to_string() + DataFileType::BatchRunTomo.extension().unwrap_or("null")
}

/// Java `getPeetDataFileName`.
pub fn get_peet_data_file_name(root_name: &str) -> String {
    root_name.to_string() + DataFileType::Peet.extension().unwrap_or("null")
}

/// Java `getPeetRootName`.
pub fn get_peet_root_name(file_name: &str) -> String {
    let extension = DataFileType::Peet.extension().unwrap_or("null");
    // Java's substring throws StringIndexOutOfBoundsException when indexOf returns -1.
    match file_name.find(extension) {
        None => file_name.to_string(),
        Some(index) => file_name[0..index].to_string(),
    }
}

/// Java `getPeetDataFile`.
pub fn get_peet_data_file(path: &str, root_name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        path,
        &get_peet_data_file_name(root_name),
    ))
}

/// Java `getRootName`.
pub fn get_root_name(param_file: &std::path::Path) -> String {
    let root_file_name = utilities::java_io_file_get_name(&param_file.to_string_lossy());
    let extension_index = match root_file_name.find('.') {
        None => return root_file_name,
        Some(extension_index) => extension_index,
    };
    root_file_name[0..extension_index].to_string()
}

// Stacks

/// Java `getStack(BaseManager, AxisID)`.
pub fn get_stack(manager: &'static dyn BaseManager, axis_id: Option<AxisID>) -> std::path::PathBuf {
    get_stack_with_dir(
        manager,
        manager.get_property_user_dir().as_deref(),
        manager.get_base_meta_data(),
        axis_id,
    )
}

/// Java `getStack(BaseManager, String, BaseMetaData, AxisID)`.
pub fn get_stack_with_dir(
    manager: &'static dyn BaseManager,
    property_user_dir: Option<&str>,
    meta_data: Option<&dyn BaseMetaData>,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        property_user_dir.unwrap_or("null"),
        &get_stack_name_with_meta_data(manager, meta_data, axis_id).unwrap_or("null".to_string()),
    ))
}

/// Java `getStackName(BaseManager, AxisID)`.
pub fn get_stack_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> Option<String> {
    get_stack_name_with_meta_data(manager, manager.get_base_meta_data(), axis_id)
}

/// Java private `getStackName(BaseManager, BaseMetaData, AxisID)`.
fn get_stack_name_with_meta_data(
    manager: &'static dyn BaseManager,
    meta_data: Option<&dyn BaseMetaData>,
    axis_id: Option<AxisID>,
) -> Option<String> {
    // Java reads `metaData` only as a parameter it does not use.
    let _ = meta_data;
    file_type::CLASS
        .raw_stack
        .get_file_name(Some(manager), axis_id)
}

/// Java `getSeedFileName`.
pub fn get_seed_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> Option<String> {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    Some(format!(
        "{}{}.seed",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    ))
}

/// Java `getSeedFile`.
pub fn get_seed_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    let _ = axis_id;
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_seed_file_name(manager, axis_id).unwrap_or("null".to_string()),
    ))
}

// Tomograms

/// Java `getRotatedTomogram`.
pub fn get_rotated_tomogram(
    manager: &'static dyn BaseManager,
    tomogram: &std::path::Path,
) -> std::path::PathBuf {
    let tomogram_name = tomogram
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    // `tomogramName.substring(0, tomogramName.lastIndexOf('.'))`; a name with no period
    // gives -1, which Java's `substring` rejects with StringIndexOutOfBoundsException.
    let last_index_of = tomogram_name
        .rfind('.')
        .expect("java.lang.StringIndexOutOfBoundsException: begin 0, end -1");
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &format!("{}{}", &tomogram_name[..last_index_of], ROTATED_TOMO_EXT),
    ))
}

// Other dataset files

/// Java `getLogName`.
pub fn get_log_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
    process_name: Option<&ProcessName>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        match process_name {
            None => "null".to_string(),
            Some(process_name) => process_name.to_string(),
        },
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        LOG_EXT
    )
}

/// Java `getAxisOnlyComFile`.
pub fn get_axis_only_com_file(
    manager: &'static dyn BaseManager,
    process_name: Option<&ProcessName>,
) -> std::path::PathBuf {
    let meta_data = manager.get_base_meta_data();
    let _ = meta_data;
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &format!(
            "{}{}",
            match process_name {
                None => "null".to_string(),
                Some(process_name) => process_name.to_string(),
            },
            COMSCRIPT_EXT
        ),
    ))
}

/// Java `getDatasetFile` (deprecated 5/9/2019).
pub fn get_dataset_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
    file_ext: Option<&str>,
) -> std::path::PathBuf {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &format!(
            "{}{}{}",
            meta_data
                .and_then(|meta_data| meta_data.get_name())
                .unwrap_or("null".to_string()),
            axis_id
                .map(|axis_id| axis_id.get_extension())
                .unwrap_or_default(),
            file_ext.unwrap_or("null")
        ),
    ))
}

/// Java `getDatasetFileFromFileName`.
pub fn get_dataset_file_from_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
    file_name: Option<&str>,
) -> std::path::PathBuf {
    let _ = axis_id;
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        file_name.unwrap_or("null"),
    ))
}

/// Java `getRawTilt`.
pub fn get_raw_tilt(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_raw_tilt_name(manager, axis_id),
    ))
}

/// Java `getRawTiltName`.
pub fn get_raw_tilt_name(manager: &'static dyn BaseManager, axis_id: Option<AxisID>) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}.rawtlt",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    )
}

/// Java `getRawTiltFile`.
pub fn get_raw_tilt_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_raw_tilt_name(manager, axis_id),
    ))
}

/// Java `getFlattenWarpOutputName`.
pub fn get_flatten_warp_output_name(manager: &'static dyn BaseManager) -> String {
    let meta_data = manager.get_base_meta_data();
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        FLATTEN_WARP_EXT,
        XF_EXT
    )
}

/// Java `getRaptorFiducialModelName`.
pub fn get_raptor_fiducial_model_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}_raptor{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        FIDUCIAL_MODEL_EXT
    )
}

/// Java `getRaptorFiducialModel`.
pub fn get_raptor_fiducial_model(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_raptor_fiducial_model_name(manager, axis_id),
    ))
}

/// Java `getTransformFileName`.
pub fn get_transform_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}.tltxf",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    )
}

/// Java private `getTiltName`.
fn get_tilt_name(manager: &'static dyn BaseManager, axis_id: Option<AxisID>) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        TILT_FILE_EXT
    )
}

/// Java `getTiltFile`.
pub fn get_tilt_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_tilt_name(manager, axis_id),
    ))
}

/// Java `getFiducialModelName`.
pub fn get_fiducial_model_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        FIDUCIAL_MODEL_EXT
    )
}

/// Java `getXTiltFileName`.
pub fn get_x_tilt_file_name(manager: &'static dyn BaseManager, axis_id: Option<AxisID>) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}.xtilt",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    )
}

/// Java `getJoinInfoName`.
pub fn get_join_info_name(manager: &'static dyn BaseManager) -> String {
    format!("{}.info", manager.get_name().unwrap_or("null".to_string()))
}

/// Java `getFiducialModelFile`.
pub fn get_fiducial_model_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_fiducial_model_name(manager, axis_id),
    ))
}

/// Java `getPieceListFile`.
pub fn get_piece_list_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_piece_list_file_name(manager, axis_id),
    ))
}

/// Java `getPieceListFileName`.
pub fn get_piece_list_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}.pl",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    )
}

/// Java `getMagGradient`.
pub fn get_mag_gradient(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_mag_gradient_name(manager, axis_id),
    ))
}

/// Java `getMatlabParamFile(BaseManager)`.
pub fn get_matlab_param_file_from_manager(manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &format!(
            "{}{}",
            manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.get_name())
                .unwrap_or("null".to_string()),
            MATLAB_PARAM_FILE_EXT
        ),
    ))
}

/// Java `getRefineXfFileName`.
pub fn get_refine_xf_file_name(manager: &'static dyn BaseManager) -> String {
    format!(
        "{}{}{}",
        manager
            .get_base_meta_data()
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        REFINE_NAME,
        XF_EXT
    )
}

/// Java `getRefineXgFileName`.
pub fn get_refine_xg_file_name(manager: &'static dyn BaseManager) -> String {
    format!(
        "{}{}{}",
        manager
            .get_base_meta_data()
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        REFINE_NAME,
        XG_EXT
    )
}

/// Java `getRefineXgFile`.
pub fn get_refine_xg_file(manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_refine_xg_file_name(manager),
    ))
}

/// Java `getRefineJoinXgFileName`.
pub fn get_refine_join_xg_file_name(manager: &'static dyn BaseManager) -> String {
    format!(
        "{}{}join{}",
        manager
            .get_base_meta_data()
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        REFINE_NAME,
        XG_EXT
    )
}

/// Java `getRefineModelFileName`.
pub fn get_refine_model_file_name(manager: &'static dyn BaseManager) -> String {
    format!(
        "{}{}{}",
        manager
            .get_base_meta_data()
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        REFINE_NAME,
        MODEL_EXT
    )
}

/// Java `getRefineModelFile(JoinManager)`.
pub fn get_refine_model_file(manager: &'static JoinManager) -> std::path::PathBuf {
    std::path::Path::new(manager.get_property_user_dir().as_deref().unwrap_or("null"))
        .join(get_refine_model_file_name(manager))
}

/// Java `getRefineAlignedModelFileName`.
pub fn get_refine_aligned_model_file_name(manager: &'static dyn BaseManager) -> String {
    format!(
        "{}{}.alimod",
        manager
            .get_base_meta_data()
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        REFINE_NAME
    )
}

/// Java `getRefineAlignedModelFile`.
pub fn get_refine_aligned_model_file(manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_refine_aligned_model_file_name(manager),
    ))
}

/// Java private `getFullAlignedStackFileName` (deprecated 5/9/2019).
fn get_full_aligned_stack_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        FULL_ALIGNED_EXT
    )
}

/// Java private `getErasedFiducialsFileName` (deprecated 5/9/2019).
fn get_erased_fiducials_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        get_erased_fiducials_file_extension()
    )
}

/// Java private `getErasedFiducialsFileExtension` (deprecated 5/9/2019).
fn get_erased_fiducials_file_extension() -> String {
    format!("{}{}", ERASE_EXT, FULL_ALIGNED_EXT)
}

/// Java private `getCtfCorrectionFileName` (deprecated 5/9/2019).
fn get_ctf_correction_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        CTF_CORRECTION_EXT
    )
}

/// Java `getCtfCorrectionFile`.
pub fn get_ctf_correction_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_ctf_correction_file_name(manager, axis_id),
    ))
}

/// Java `getSimpleDefocusFile`.
pub fn get_simple_defocus_file(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_simple_defocus_file_name(manager, axis_id),
    ))
}

/// Java `getCtfPlotterFileName`.
pub fn get_ctf_plotter_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        CTF_PLOTTER_EXT
    )
}

/// Java `getSimpleDefocusFileName`.
pub fn get_simple_defocus_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}{}",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        SIMPLE_DEFOCUS_EXT
    )
}

/// Java `getMagGradientName`.
pub fn get_mag_gradient_name(manager: &'static dyn BaseManager, axis_id: Option<AxisID>) -> String {
    let meta_data = manager.get_base_meta_data();
    let axis_id = correct_axis_id_from_meta_data(meta_data, axis_id);
    format!(
        "{}{}.maggrad",
        meta_data
            .and_then(|meta_data| meta_data.get_name())
            .unwrap_or("null".to_string()),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    )
}

/// Java `getTransferFidCoordFileName`.
pub fn get_transfer_fid_coord_file_name() -> String {
    "transferfid.coord".to_string()
}

/// Java `getTransferFidCoordFile`.
pub fn get_transfer_fid_coord_file(manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &get_transfer_fid_coord_file_name(),
    ))
}

/// Java `getPatchVectorModel`.
pub fn get_patch_vector_model(manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        PATCH_VECTOR_MODEL,
    ))
}

/// Java `getPatchVectorCCCModel`.
pub fn get_patch_vector_ccc_model(manager: &'static dyn BaseManager) -> std::path::PathBuf {
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        PATCH_VECTOR_CCC_MODEL,
    ))
}

/// Java `getShellScript`.
pub fn get_shell_script(
    manager: &'static dyn BaseManager,
    command_name: Option<&str>,
    axis_id: Option<AxisID>,
) -> std::path::PathBuf {
    let axis_id = correct_axis_id_from_meta_data(manager.get_base_meta_data(), axis_id);
    std::path::PathBuf::from(utilities::java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &format!(
            "{}{}.csh",
            command_name.unwrap_or("null"),
            axis_id
                .map(|axis_id| axis_id.get_extension())
                .unwrap_or_default()
        ),
    ))
}

/// Java `getOutFileName`.
pub fn get_out_file_name(
    manager: &'static dyn BaseManager,
    subdir_name: Option<&str>,
    command_name: Option<&str>,
    axis_id: Option<AxisID>,
) -> String {
    let axis_id = correct_axis_id_from_meta_data(manager.get_base_meta_data(), axis_id);
    let out_file_name = format!(
        "{}{}.out",
        command_name.unwrap_or("null"),
        axis_id
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default()
    );
    let subdir_name = match subdir_name {
        None => return out_file_name,
        Some(subdir_name) => subdir_name,
    };
    // `new File(subdirName, outFileName).getPath()`
    utilities::java_io_file_new(subdir_name, &out_file_name)
}

// log files

/// Java `getTomopitchLogFileName`.
pub fn get_tomopitch_log_file_name(
    manager: &'static dyn BaseManager,
    axis_id: Option<AxisID>,
) -> String {
    format!(
        "{}{}{}",
        ProcessName::TOMOPITCH,
        correct_axis_id_from_meta_data(manager.get_base_meta_data(), axis_id)
            .map(|axis_id| axis_id.get_extension())
            .unwrap_or_default(),
        LOG_EXT
    )
}

// directories

/// Java `getCalibrationDir`.
pub fn get_calibration_dir(
    manager: Option<&'static dyn BaseManager>,
    property_user_dir: Option<&str>,
    axis_id: Option<AxisID>,
) -> Option<std::path::PathBuf> {
    let mut calibration_dir = CALIBRATION_DIR.lock().unwrap();
    if calibration_dir.is_none() {
        let calib_dir_var = environment_variable::INSTANCE.get_value(
            manager,
            property_user_dir,
            environment_variable::CALIB_DIR,
            axis_id,
        );
        if !calib_dir_var.is_empty() {
            *calibration_dir = Some(std::path::PathBuf::from(calib_dir_var));
        }
    }
    calibration_dir.clone()
}

/// Java `getDistortionDir`.
pub fn get_distortion_dir(
    manager: Option<&'static dyn BaseManager>,
    property_user_dir: Option<&str>,
    axis_id: Option<AxisID>,
) -> Option<std::path::PathBuf> {
    if CALIBRATION_DIR.lock().unwrap().is_none() {
        get_calibration_dir(manager, property_user_dir, axis_id);
    }
    let calibration_dir = CALIBRATION_DIR.lock().unwrap().clone();
    let calibration_dir = calibration_dir?;
    let mut distortion_dir = DISTORTION_DIR.lock().unwrap();
    if distortion_dir.is_none() {
        *distortion_dir = Some(std::path::PathBuf::from(utilities::java_io_file_new(
            &calibration_dir.to_string_lossy(),
            "Distortion",
        )));
    }
    distortion_dir.clone()
}

// private

/// Java private `correctAxisID(BaseMetaData, AxisID)`.
fn correct_axis_id_from_meta_data(
    meta_data: Option<&dyn BaseMetaData>,
    axis_id: Option<AxisID>,
) -> Option<AxisID> {
    correct_axis_id(
        meta_data.map(|meta_data| meta_data.base().get_axis_type()),
        axis_id,
    )
}
