//! `IMOD/Etomo/src/etomo/ToolsManager.java`.
//!
//! This is the owner of the Tools interface.  The manager's file-name collision
//! checks and alignframes list-file operations are concrete.  Its direct process,
//! comscript, and Swing panel collaborators retain explicit boundaries until their
//! respective source units are translated; this does not substitute a second process
//! implementation for Etomo's `ToolsProcessManager`.
#![allow(dead_code)]

use std::convert::Infallible;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

/// Java `private static final AxisID AXIS_ID`.
pub const AXIS_ID: AxisID = AxisID::Only;
/// Java `private static final DialogType DIALOG_TYPE`.
pub const DIALOG_TYPE: DialogType = DialogType::Tools;
/// Java `private static final int STATUS_BAR_SIZE`.
pub const STATUS_BAR_SIZE: i32 = 65;

/// Direct declared-type boundary for `etomo.type.ToolsMetaData`.
///
/// Its source owns parameter-file persistence and has not yet been translated.
pub struct ToolsMetaData {
    root_name: Mutex<Option<String>>,
}

impl ToolsMetaData {
    /// Java `setRootName(File)` / `setRootName(String)` state used by this manager.
    pub fn set_root_name(&self, root_name: Option<&str>) {
        *self.root_name.lock().unwrap() = root_name.map(str::to_owned);
    }

    /// Java `getName`.
    pub fn get_name(&self) -> Option<String> {
        self.root_name.lock().unwrap().clone()
    }
}

/// Java `ToolsManager extends BaseManager`.
pub struct ToolsManager {
    base: BaseManagerBase,
    /// Java `alignFramesTiltAngleFile`, initially false.
    align_frames_tilt_angle_file: Mutex<bool>,
    /// Java final `metaData`.
    meta_data: ToolsMetaData,
    /// Java final `toolType`.
    tool_type: ToolType,
    /// Java `mainPanel`.
    // TODO(unit): etomo/ui/swing/MainToolsPanel.java.
    main_panel: Option<Infallible>,
    /// Java `processMgr`.
    // TODO(unit): etomo/process/ToolsProcessManager.java.
    process_mgr: Option<Infallible>,
    /// Java nullable `toolsDialog`.
    // TODO(unit): etomo/ui/swing/ToolsDialog.java's panel-factory/process integration.
    tools_dialog: Mutex<Option<Infallible>>,
    /// Java `comScriptMgr`.
    // TODO(unit): etomo/comscript/ToolsComScriptManager.java.
    com_script_mgr: Option<Infallible>,
}

impl ToolsManager {
    /// Java `ToolsManager(ToolType)`.
    pub fn new(tool_type: ToolType) -> &'static Self {
        let instance = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            align_frames_tilt_angle_file: Mutex::new(false),
            meta_data: ToolsMetaData {
                root_name: Mutex::new(None),
            },
            tool_type,
            main_panel: None,
            process_mgr: None,
            tools_dialog: Mutex::new(None),
            com_script_mgr: None,
        }));
        instance.base_manager();
        instance.initialize_ui_parameters(None, Some(AXIS_ID), false);
        instance.create_state();
        instance
    }

    /// Java `initialize`.
    pub fn initialize(&self) {
        // TODO(unit): MainToolsPanel, ToolsDialog and UIHarness own these calls:
        // openProcessingPanel(); setStatusBarText(); openToolsDialog(); toFront().
    }

    /// Java `closeFrame`.
    pub fn close_frame(&self) -> bool {
        true
    }

    /// Java `isConflictingDatasetName`.
    pub fn is_conflicting_dataset_name(&self, axis_id: AxisID, file: &Path) -> bool {
        let _ = axis_id;
        let parent = match file.parent() {
            Some(value) => value,
            None => return false,
        };
        let name = match file.file_name().and_then(|value| value.to_str()) {
            Some(value) => value,
            None => return false,
        };
        match fs::read_dir(parent) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .any(|entry| ConflictFileFilter::new(name).accept(&entry.path())),
            Err(_) => false,
        }
    }

    /// Java `setName(File)`.
    pub fn set_name(&self, input_file: &Path) {
        self.set_property_user_dir(input_file.parent().and_then(Path::to_str));
        self.meta_data
            .set_root_name(input_file.file_stem().and_then(|value| value.to_str()));
        // TODO(unit): MainToolsPanel, ToolsDialog and ToolsComScriptManager.
    }

    /// Java `openToolsDialog`.
    pub fn open_tools_dialog(&self) {
        // TODO(unit): ToolsDialog.getInstance and MainToolsPanel.showProcess.
    }

    /// Java `gpuTiltTestSuceeded` (including its source spelling).
    pub fn gpu_tilt_test_suceeded(&self, output: Option<&[String]>, axis_id: AxisID) {
        let _ = (output, axis_id);
        // TODO(unit): UIHarness.openInfoMessageDialog.
    }

    /// Java `gpuTiltTest`.
    pub fn gpu_tilt_test(&self, axis_id: AxisID) {
        let _ = self.update_gpu_tilt_test(axis_id, true);
        // TODO(unit): ToolsProcessManager.gpuTiltTest and MainToolsPanel progress state.
    }

    /// Java private `updateGpuTiltTest`.
    pub fn update_gpu_tilt_test(&self, axis_id: AxisID, do_validation: bool) -> Option<Infallible> {
        let _ = (axis_id, do_validation);
        // TODO(unit): GpuTiltTestParam and ToolsDialog.getParameters.
        None
    }

    /// Java `flatten`.
    pub fn flatten(
        &self,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        run_3dmod_menu_options: Option<Infallible>,
        dialog_type: Option<DialogType>,
        axis_id: AxisID,
        display: Option<Infallible>,
    ) {
        let _ = (
            process_result_display,
            process_series,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            dialog_type,
            axis_id,
            display,
        );
        // TODO(unit): ProcessSeries, WarpVolParam, WarpVolDisplay, ToolsProcessManager.
    }

    /// Java private `updateWarpVolParam`.
    pub fn update_warp_vol_param(
        &self,
        display: Option<Infallible>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<Infallible> {
        let _ = (display, axis_id, do_validation);
        // TODO(unit): ToolsComScriptManager and WarpVolParam/WarpVolDisplay.
        None
    }

    /// Java `imodFlatten`.
    pub fn imod_flatten(&self, menu_options: Option<Infallible>, axis_id: AxisID) {
        let _ = (menu_options, axis_id);
        // TODO(unit): ImodManager.open.
    }

    /// Java `imodMakeSurfaceModel`.
    pub fn imod_make_surface_model(
        &self,
        menu_options: Option<Infallible>,
        axis_id: AxisID,
        binning: i32,
        file: Option<&Path>,
    ) {
        let _ = (menu_options, axis_id, binning, file);
        // TODO(unit): ImodManager open/configuration and FileType model naming.
    }

    /// Java private `isFlipped`.
    pub fn is_flipped(&self, mrc_file: Option<&Path>) -> bool {
        let _ = mrc_file;
        // TODO(unit): MRCHeader.read(BaseManager), whose `header` external-program
        // boundary is deliberately retained in util/mrc_header.rs.
        false
    }

    /// Java `flattenWarp`.
    pub fn flatten_warp(
        &self,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        deferred_3dmod_button: Option<Infallible>,
        run_3dmod_menu_options: Option<Infallible>,
        dialog_type: Option<DialogType>,
        axis_id: AxisID,
        display: Option<Infallible>,
    ) {
        let _ = (
            process_result_display,
            process_series,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            dialog_type,
            axis_id,
            display,
        );
        // TODO(unit): BusyStatusMediator, FlattenWarpParam/Display and ToolsProcessManager.
    }

    /// Java private `updateFlattenWarpParam`.
    pub fn update_flatten_warp_param(
        &self,
        display: Option<Infallible>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<Infallible> {
        let _ = (display, axis_id, do_validation);
        // TODO(unit): FlattenWarpParam and FlattenWarpDisplay.
        None
    }

    /// Java private `updateAlignFramesOutputParam`.
    pub fn update_align_frames_output_param(&self, display: Option<Infallible>) -> bool {
        let _ = display;
        // TODO(unit): AlignFramesDisplay, ToolsComScriptManager, LogFile backup and
        // ToolsProcessManager.touch.
        false
    }

    /// Java `getListOfInputFiles`.
    pub fn get_list_of_input_files(
        &self,
        rootname: Option<&str>,
        subdir: &Path,
        in_list: bool,
        text: Option<&str>,
    ) -> io::Result<PathBuf> {
        let rootname = rootname.unwrap_or("");
        let suffix = if in_list {
            "_inlist.txt"
        } else {
            "_templist.txt"
        };
        let path = subdir.join(format!("{rootname}{suffix}"));
        let mut list = File::create(&path)?;
        list.write_all(text.unwrap_or("").as_bytes())?;
        Ok(path)
    }

    /// Java `getOutputFileList`.
    pub fn get_output_file_list(
        &self,
        rootname: Option<&str>,
        subdir: &Path,
    ) -> io::Result<PathBuf> {
        let path = subdir.join(format!("{}_inlist.txt", rootname.unwrap_or("")));
        File::create(&path)?;
        Ok(path)
    }

    /// Java `setAlignFramesTiltAngleFileCreated`.
    pub fn set_align_frames_tilt_angle_file_created(&self, input: bool) {
        *self.align_frames_tilt_angle_file.lock().unwrap() = input;
    }

    /// Java `openAlignFramesInputComFile`.
    pub fn open_align_frames_input_com_file(
        &self,
        display: Option<Infallible>,
        com_file: Option<&Path>,
        component: Option<Infallible>,
    ) -> bool {
        let _ = (display, component);
        match com_file {
            Some(path) => path.is_file() && File::open(path).is_ok(),
            None => false,
        }
        // TODO(unit): ToolsComScriptManager.loadAlignFramesInput and AlignFramesDisplay.
    }

    /// Java `alignFrames`.
    pub fn align_frames(&self, process_series: Option<Infallible>, display: Option<Infallible>) {
        let _ = (process_series, display);
        // TODO(unit): AlignFramesParam, ToolsProcessManager and ProcessSeries.
    }

    /// Java `setRootname`.
    pub fn set_rootname(&self, rootname: Option<&str>) {
        if rootname.is_some() {
            self.meta_data.set_root_name(rootname);
        }
    }

    /// Java `sorttiltframes`.
    pub fn sorttiltframes(&self, display: Option<Infallible>) {
        let _ = display;
        // TODO(unit): SortTiltFramesParam, ProcessSeries and ToolsProcessManager.
    }

    /// Java `plotAllResults`.
    pub fn plot_all_results(&self, display: Option<Infallible>) {
        let _ = display;
        // TODO(unit): TomodataplotsParam, ProcessSeries, and BaseManager.tomodataplots.
    }

    /// Java `openOutputTiltSeries`.
    pub fn open_output_tilt_series(
        &self,
        menu_options: Option<Infallible>,
        display: Option<Infallible>,
    ) {
        let _ = (menu_options, display);
        // TODO(unit): AlignFramesDisplay and ImodManager.open.
    }

    /// Java `openTomogram`.
    pub fn open_tomogram(&self, display: Option<Infallible>) {
        let _ = display;
        // TODO(unit): AlignFramesDisplay, LocalArguments, EtomoDirector automation.
    }

    /// Java `imodViewModel`.
    pub fn imod_view_model(&self, axis_id: AxisID, model_file_type: Option<Infallible>) {
        let _ = (axis_id, model_file_type);
        // TODO(unit): FileType and ImodManager.open.
    }

    /// Java `getState`.
    pub fn get_state(&self) -> Option<Infallible> {
        None
    }

    /// Java private `createState`.
    pub fn create_state(&self) {}

    /// Java private `openProcessingPanel`.
    pub fn open_processing_panel(&self) {
        // TODO(unit): MainToolsPanel and AxisProcessData reconnect.
    }
}

impl BaseManager for ToolsManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }
    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Tools)
    }
    fn create_main_panel(&self) {
        // TODO(unit): MainToolsPanel.java.
    }
    fn get_base_meta_data(
        &self,
    ) -> Option<&dyn crate::imod::etomo::r#type::base_meta_data::BaseMetaData> {
        // TODO(unit): ToolsMetaData.java must implement BaseMetaData.
        None
    }
    fn get_main_panel(&self) -> Option<Infallible> {
        self.main_panel
    }
    fn get_process_manager(&self) -> Option<Infallible> {
        self.process_mgr
    }
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Box<dyn Storable>>> {
        let _ = offset;
        None
    }
    fn get_name(&self) -> Option<String> {
        self.meta_data.get_name()
    }
    fn create_com_script_manager(&self) {
        // TODO(unit): ToolsComScriptManager.java.
    }
    /// Java `getLogInterface`.
    fn get_log_interface(&self) -> Option<Infallible> {
        *self.tools_dialog.lock().unwrap()
    }
    /// Java `createLogWindow`.
    fn create_log_window(&self) -> Option<Infallible> {
        None
    }
    /// Java `isInManagerFrame`.
    fn is_in_manager_frame(&self) -> bool {
        false
    }
    /// Java `save`.
    fn save(&self) -> bool {
        if !self.save_super() {
            return false;
        }
        // TODO(unit): MainToolsPanel.done().
        true
    }
    /// Java `exitProgram`.
    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        if self.exit_program_super(axis_id) {
            self.end_threads();
            self.save_param_file();
            return true;
        }
        false
    }
    /// Java `startNextProcess`.
    fn start_next_process(
        &self,
        ui_component: Option<Infallible>,
        axis_id: Option<AxisID>,
        process: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        dialog_type: Option<Infallible>,
        display: Option<Infallible>,
    ) -> bool {
        let _ = (
            ui_component,
            axis_id,
            process,
            process_result_display,
            process_series,
            dialog_type,
            display,
        );
        // TODO(unit): ProcessSeries.Process and AlignFramesDisplay.  The source's
        // `Task.ALIGN_FRAMES` dispatch belongs here once those declared types exist.
        false
    }
}

/// Java private static final `ConflictFileFilter`.
pub struct ConflictFileFilter {
    compare_file_name: String,
}

impl ConflictFileFilter {
    /// Java private `ConflictFileFilter(String)`.
    pub fn new(compare_file_name: &str) -> Self {
        Self {
            compare_file_name: compare_file_name.to_owned(),
        }
    }
    /// Java `accept(File)`.
    pub fn accept(&self, file: &Path) -> bool {
        if !file.is_file() {
            return false;
        }
        let name = match file.file_name().and_then(|value| value.to_str()) {
            Some(value) => value,
            None => return false,
        };
        for extension in [".edf", ".ejf", ".epe", ".ess"] {
            if let Some(left) = name.strip_suffix(extension) {
                return left == self.compare_file_name;
            }
        }
        false
    }
    /// Java `getDescription`.
    pub fn get_description(&self) -> String {
        format!(
            "Dataset file that conflicts with {}",
            self.compare_file_name
        )
    }
}

/// Java public static final nested `Task`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    AlignFrames,
}

impl TaskInterface for Task {
    fn get_descr(&self) -> Option<String> {
        Some("align frames".to_string())
    }
    fn ok_to_drop(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflict_filter_matches_only_exclusive_dataset_extensions() {
        let dir =
            std::env::temp_dir().join(format!("imod-rs-tools-manager-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let conflict = dir.join("dataset.edf");
        let parallel = dir.join("dataset.epp");
        File::create(&conflict).unwrap();
        File::create(&parallel).unwrap();
        let filter = ConflictFileFilter::new("dataset");
        assert!(filter.accept(&conflict));
        assert!(!filter.accept(&parallel));
        assert_eq!(
            filter.get_description(),
            "Dataset file that conflicts with dataset"
        );
        fs::remove_file(conflict).unwrap();
        fs::remove_file(parallel).unwrap();
        fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn input_and_output_lists_follow_java_names() {
        let dir = std::env::temp_dir().join(format!("imod-rs-tools-lists-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let manager = ToolsManager::new(ToolType::AlignFrames);
        let input = manager
            .get_list_of_input_files(Some("set"), &dir, true, Some("a\nb\n"))
            .unwrap();
        assert_eq!(input.file_name().unwrap(), "set_inlist.txt");
        assert_eq!(fs::read_to_string(&input).unwrap(), "a\nb\n");
        let output = manager.get_output_file_list(Some("set"), &dir).unwrap();
        assert_eq!(output, input);
        fs::remove_file(output).unwrap();
        fs::remove_dir(dir).unwrap();
    }
}
