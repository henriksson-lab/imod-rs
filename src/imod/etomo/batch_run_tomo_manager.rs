//! `IMOD/Etomo/src/etomo/BatchRunTomoManager.java`.
//!
//! Source-shaped owner of the batchruntomo interface.  This deliberately retains every
//! Java field and member boundary.  The batchruntomo dialog/comscript/process units have
//! not yet been translated, therefore their Java-null references are `Option<Infallible>`:
//! this records the exact frontier without inventing runnable process or GUI behaviour.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java public static final `AXIS_ID`.
pub const AXIS_ID: AxisID = AxisID::Only;
/// Java `STACK_REFERENCE_PREFIX`, from `DataFileType.BATCH_RUN_TOMO.extension.substring(1)`.
const STACK_REFERENCE_PREFIX: &str = "brt";

/// Java private static final nested `Task`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// Java `PROCESSCHUNKS = new Task("processchunks")`.
    Processchunks,
}

impl TaskInterface for Task {
    fn get_descr(&self) -> Option<String> {
        Some("processchunks".to_string())
    }
    fn ok_to_drop(&self) -> bool {
        false
    }
}

/// Java `BatchRunTomoManager extends BaseManager`.
pub struct BatchRunTomoManager {
    base: BaseManagerBase,
    /// `BatchRunTomoMetaData.getName()` until its typed metadata is present.
    name: String,
    /// Java `tableReference`.
    // TODO(unit): etomo/type/TableReference.java.
    table_reference: Option<Infallible>,
    /// Java `comScriptManager`.
    // TODO(unit): etomo/comscript/BatchRunTomoComScriptManager.java.
    com_script_manager: Option<Infallible>,
    /// Java `screenState`.
    // TODO(unit): etomo/type/BatchRunTomoScreenState.java.
    screen_state: Option<Infallible>,
    /// Java `messagesArray`.
    // TODO(unit): etomo/process/ProcessMessages.java.
    messages_array: Mutex<Vec<Option<Infallible>>>,
    /// Java `seriesWatcherMessagesMap`.
    // TODO(unit): etomo/process/ProcessMessages.java.
    series_watcher_messages_map: Mutex<BTreeMap<String, Option<Infallible>>>,
    /// Java `datasetFileBuilder`.
    // TODO(unit): etomo/storage/DatasetFileBuilder.java.
    dataset_file_builder: Option<Infallible>,
    /// Java `cleanPrint` singleton.
    // TODO(unit): etomo/util/CleanPrint.java.
    clean_print: Option<Infallible>,
    /// Java final `metaData`.
    // TODO(unit): etomo/type/BatchRunTomoMetaData.java.
    meta_data: Option<Infallible>,
    /// Java final `processMgr`.
    // TODO(unit): etomo/process/BatchRunTomoProcessManager.java.
    process_mgr: Option<Infallible>,
    /// Java `mainPanel`.
    // TODO(unit): etomo/ui/swing/MainBatchRunTomoPanel.java.
    main_panel: Option<Infallible>,
    /// Java nullable `dialog`.
    // TODO(unit): etomo/ui/swing/BatchRunTomoDialog.java.
    dialog: Mutex<Option<Infallible>>,
    reconnect_run_a: Mutex<bool>,
    reconnect_run_b: Mutex<bool>,
    /// Java nullable `listeners`.
    // TODO(unit): etomo/ui/DialogCompleteListener.java.
    listeners: Mutex<Option<Vec<Option<Infallible>>>>,
    diagnostics: Mutex<bool>,
}

impl BatchRunTomoManager {
    /// Java `BatchRunTomoManager()`.
    pub fn new() -> &'static Self {
        Self::new_with_param_file_name(Some(""))
    }

    /// Java `BatchRunTomoManager(String paramFileName)`.
    pub fn new_with_param_file_name(param_file_name: Option<&str>) -> &'static Self {
        let instance = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            name: param_file_name
                .filter(|name| !name.is_empty())
                .unwrap_or("Batch Run Tomo")
                .to_owned(),
            table_reference: None,
            com_script_manager: None,
            screen_state: None,
            messages_array: Mutex::new(Vec::new()),
            series_watcher_messages_map: Mutex::new(BTreeMap::new()),
            dataset_file_builder: None,
            clean_print: None,
            meta_data: None,
            process_mgr: None,
            main_panel: None,
            dialog: Mutex::new(None),
            reconnect_run_a: Mutex::new(false),
            reconnect_run_b: Mutex::new(false),
            listeners: Mutex::new(None),
            diagnostics: Mutex::new(false),
        }));
        instance.base_manager();
        instance.initialize_ui_parameters_from_name(param_file_name, Some(AXIS_ID));
        instance
    }

    /// Java `getMetaData`.
    // TODO(unit): etomo/type/BatchRunTomoMetaData.java.
    pub fn get_meta_data(&self) -> Option<Infallible> {
        self.meta_data
    }

    /// Java `setNewParamFile(File, String)`.
    pub fn set_new_param_file(&self, root_dir: Option<&Path>, root_name: Option<&str>) -> bool {
        if *self.base.loaded_param_file.lock().unwrap() {
            return true;
        }
        let root_dir = match root_dir {
            Some(value) => value,
            None => return false,
        };
        if root_dir.to_string_lossy().ends_with(' ') {
            return false;
        }
        let root_name = match root_name {
            Some(value) if !value.is_empty() => value,
            _ => return false,
        };
        self.set_property_user_dir(Some(&root_dir.display().to_string()));
        let param_file = root_dir.join(format!("{root_name}.ebrt"));
        self.set_param_file_from(Some(&param_file))
    }

    /// Java private `openProcessingPanel`.
    fn open_processing_panel(&self) {
        self.set_panel();
    }
    /// Java `openBatchRunTomoDialog`.
    pub fn open_batch_run_tomo_dialog(&self) {}
    /// Java `addDialogCompleteListener(DialogCompleteListener)`.
    pub fn add_dialog_complete_listener(&self, listener: Option<Infallible>) {
        if listener.is_none() {
            return;
        }
        self.listeners
            .lock()
            .unwrap()
            .get_or_insert_with(Vec::new)
            .push(listener);
    }
    /// Java `getBatchruntomoComfileNamingStyle`.
    // TODO(unit): BatchruntomoParam / ConstEtomoNumber.
    pub fn get_batchruntomo_comfile_naming_style(&self) -> Option<Infallible> {
        None
    }
    /// Java `getDatasetImageFilenameStyle`.
    // TODO(unit): ImageFilenameStyle / BatchRunTomoDialog.
    pub fn get_dataset_image_filename_style(&self) -> Option<Infallible> {
        None
    }
    /// Java `tomosetexts(File)`.
    pub fn tomosetexts_in_dir(
        &self,
        dir: Option<&Path>,
    ) -> Option<crate::imod::etomo::process::tomosetexts_output::TomosetextsOutput> {
        let _ = dir;
        BaseManager::tomosetexts(self)
    }
    /// Java `initDialog(String, boolean)`.
    pub fn init_dialog(
        &self,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        let _ = (only_stack_id_dataset_dialog, only_advanced_dataset_dialog);
    }
    /// Java `setSeriesWatcherParameters`.
    pub fn set_series_watcher_parameters(&self) {}
    /// Java `setParameters(SeriesWatcherMetaData, String, boolean)`.
    pub fn set_parameters(
        &self,
        series_watcher_meta_data: Option<Infallible>,
        stack_id: Option<&str>,
        init: bool,
    ) {
        let _ = (series_watcher_meta_data, stack_id, init);
    }
    /// Java `isDatasetDialog(String)`.
    pub fn is_dataset_dialog(&self, stack_id: Option<&str>) -> bool {
        let _ = stack_id;
        false
    }
    /// Java `msgStatusChangerStarted(StatusChanger, boolean)`.
    pub fn msg_status_changer_started(&self, changer: Option<Infallible>, table_only: bool) {
        let _ = (changer, table_only);
    }
    /// Java private `addProcessMessagesInstance`.
    fn add_process_messages_instance(&self) {
        self.messages_array.lock().unwrap().push(None);
    }
    /// Java private `updateSplitBatch`.
    pub fn update_split_batch(&self) -> Option<Infallible> {
        None
    }
    /// Java final `processchunks(ProcessSeries, Command, String, RunType)`.
    // TODO(unit): etomo/ProcessSeries.java, etomo/comscript/Command.java,
    // etomo/comscript/ProcesschunksParam.java, etomo/type/RunType.java, and
    // etomo/ui/swing/BatchRunTomoDialog.java.
    pub fn processchunks(
        &self,
        process_series: Option<Infallible>,
        command: Option<Infallible>,
        premade_machine_list: Option<&str>,
        run_type: Option<Infallible>,
    ) {
        let _ = (process_series, command, premade_machine_list, run_type);
    }
    /// Java `findRow(String, String)`.
    pub fn find_row(&self, location: Option<&str>, root_name: Option<&str>) -> Option<String> {
        let _ = (location, root_name);
        None
    }
    /// Java `getStack(String)`.
    pub fn get_stack(&self, stack_id: Option<&str>) -> Option<PathBuf> {
        let _ = stack_id;
        None
    }
    /// Java `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed(&self, status: Option<Infallible>) {
        let _ = status;
    }
    /// Java `setPremadeMachineList(ProcessSeries, String[])`.
    pub fn set_premade_machine_list(
        &self,
        process_series: Option<Infallible>,
        std_output: Option<&[Option<&str>]>,
    ) {
        let _ = (process_series, std_output);
    }
    /// Java `splitBatch`.
    pub fn split_batch(&self) {}
    /// Java `batchruntomo(ProcessingMethod)`.
    pub fn batchruntomo(&self, processing_method: Option<Infallible>) {
        let _ = processing_method;
    }
    /// Java `startSeriesWatcherBatchMonitor(File, String)`.
    pub fn start_series_watcher_batch_monitor(
        &self,
        batch_log: Option<&Path>,
        stack_id: Option<&str>,
    ) {
        let _ = (batch_log, stack_id);
    }
    /// Java `seriesWatcher(ProcessingMethod)`.
    pub fn series_watcher(&self, processing_method: Option<Infallible>) {
        let _ = processing_method;
    }
    /// Java package-private `setupProcessMessages`.
    pub fn setup_process_messages(&self) -> Option<Infallible> {
        None
    }
    /// Java package-private `setupSeriesWatcherProcessMessages`.
    pub fn setup_series_watcher_process_messages(&self) -> Option<Infallible> {
        None
    }
    /// Java package-private `setupSeriesWatcherBatchProcessMessages(String)`.
    pub fn setup_series_watcher_batch_process_messages(
        &self,
        stack_id: Option<&str>,
    ) -> Option<Infallible> {
        let _ = stack_id;
        None
    }
    /// Java `resumeBatchruntomo(ProcessingMethod)`.
    pub fn resume_batchruntomo(&self, processing_method: Option<Infallible>) {
        let _ = processing_method;
    }
    /// Java private `retrieveScreenStateFromDialog`.
    fn retrieve_screen_state_from_dialog(&self) {}
    /// Java `saveBatchRunTomoDialog(RunType, boolean, boolean, String, boolean, boolean, boolean)`.
    pub fn save_batch_run_tomo_dialog(
        &self,
        run_type: Option<Infallible>,
        do_validation: bool,
        init: bool,
        autodoc_stack_id: Option<&str>,
        only_global_autodoc: bool,
        parallel_processing: bool,
        diagnostics: bool,
    ) -> Option<Infallible> {
        let _ = (
            run_type,
            do_validation,
            init,
            autodoc_stack_id,
            only_global_autodoc,
            parallel_processing,
            diagnostics,
        );
        None
    }
    /// Java private `updateSeriesWatcher(boolean)`.
    fn update_series_watcher(&self, do_validation: bool) -> Option<Infallible> {
        let _ = do_validation;
        None
    }
    /// Java private `validateBatchRunTomoParam`.
    fn validate_batch_run_tomo_param(&self) -> bool {
        false
    }
    /// Java private `updateBatchRunTomo(RunType, boolean, boolean, boolean)`.
    fn update_batch_run_tomo(
        &self,
        run_type: Option<Infallible>,
        do_validation: bool,
        parallel_processing: bool,
        validate_only: bool,
    ) -> Option<Infallible> {
        let _ = (run_type, do_validation, parallel_processing, validate_only);
        None
    }
    /// Java `imodStack(File, AxisID, int, File, boolean, Run3dmodMenuOptions)`.
    pub fn imod_stack(
        &self,
        stack: Option<&Path>,
        axis_id: Option<AxisID>,
        imod_index: i32,
        model_file: Option<&Path>,
        dual_axis: bool,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (stack, axis_id, model_file, dual_axis, menu_options);
        imod_index
    }
    /// Java `imodRec(File, String, AxisType, int, Run3dmodMenuOptions)`.
    pub fn imod_rec(
        &self,
        dataset_dir: Option<&Path>,
        dataset: Option<&str>,
        axis_type: Option<Infallible>,
        imod_index: i32,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (dataset_dir, dataset, axis_type, menu_options);
        imod_index
    }
    /// Java `imodTrimvol(File, String, AxisType, int, Run3dmodMenuOptions)`.
    pub fn imod_trimvol(
        &self,
        dataset_dir: Option<&Path>,
        dataset: Option<&str>,
        axis_type: Option<Infallible>,
        imod_index: i32,
        menu_options: Option<Infallible>,
    ) -> i32 {
        let _ = (dataset_dir, dataset, axis_type, menu_options);
        imod_index
    }
    /// Java `imodModel(AxisID, int, File, String, FileType, boolean)`.
    pub fn imod_model(
        &self,
        axis_id: Option<AxisID>,
        imod_index: i32,
        stack_location: Option<&Path>,
        stack_name: Option<&str>,
        model_file_type: Option<Infallible>,
        dual_axis: bool,
    ) {
        let _ = (
            axis_id,
            imod_index,
            stack_location,
            stack_name,
            model_file_type,
            dual_axis,
        );
    }
    /// Java `openLog(File)`.
    pub fn open_log(&self, log: Option<&Path>) {
        let _ = log;
    }
}

impl BaseManager for BatchRunTomoManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }
    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::BatchRunTomo)
    }
    fn create_main_panel(&self) {}
    fn get_base_meta_data(
        &self,
    ) -> Option<&dyn crate::imod::etomo::r#type::base_meta_data::BaseMetaData> {
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
        Some(self.name.clone())
    }
    /// Java `isAddGPUMachineToProcessChunks`.
    fn is_add_gpu_machine_to_process_chunks(&self) -> bool {
        true
    }
    /// Java `isPopupChunkWarnings`.
    fn is_popup_chunk_warnings(&self) -> bool {
        false
    }
    /// Java `isDualSelectionQueueTable`.
    fn is_dual_selection_queue_table(&self) -> bool {
        true
    }
    /// Java `getMessagesArray`.
    fn get_messages_array(&self) -> Option<Infallible> {
        None
    }
    /// Java `exitProgram(AxisID)`.
    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        self.exit_program_super(axis_id)
    }
    /// Java `pack`.
    fn pack(&self) {}
    /// Java `getBaseScreenState(AxisID)`.
    fn get_base_screen_state(&self, axis_id: Option<AxisID>) -> Option<Infallible> {
        let _ = axis_id;
        self.screen_state
    }
    /// Java `save`.
    fn save(&self) -> bool {
        self.save_super()
    }
    /// Java `startNextProcess(...)`.
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
        false
    }
    /// Java `sendEvent(AxisID, ProcessName, ProcessEndState, boolean)`.
    fn send_event(
        &self,
        axis_id: Option<AxisID>,
        process_name: Option<Infallible>,
        process_end_state: Option<Infallible>,
        failed: bool,
    ) {
        let _ = (axis_id, process_name, process_end_state, failed);
    }
    /// Java `updateProcessChunks(...)`.
    fn update_process_chunks(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        root_name: Option<&str>,
        subcommand_details: Option<Infallible>,
        dialog_type: Option<Infallible>,
    ) -> Option<Infallible> {
        let _ = (axis_id, param, root_name, subcommand_details, dialog_type);
        None
    }
    /// Java `createRunList(RunType)`.
    fn create_run_list(&self, run_type: Option<Infallible>) -> Option<Infallible> {
        let _ = run_type;
        None
    }
    /// Java `reconnect(ProcessData, AxisID, boolean, List<ProcessMessages>)`.
    fn reconnect(
        &self,
        process_data: Option<Infallible>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<Infallible>,
    ) -> bool {
        let _ = (process_data, multi_line_messages, messages_array);
        if self.is_reconnect_run(axis_id) {
            return false;
        }
        self.set_reconnect_run(axis_id);
        false
    }
    /// Java `reconnectToDifferentHost(ProcessData, AxisID)`.
    fn reconnect_to_different_host(
        &self,
        process_data: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let _ = (process_data, axis_id);
        true
    }
    /// Java private `isReconnectRun(AxisID)`.
    fn is_reconnect_run(&self, axis_id: Option<AxisID>) -> bool {
        if axis_id == Some(AxisID::Second) {
            *self.reconnect_run_b.lock().unwrap()
        } else {
            *self.reconnect_run_a.lock().unwrap()
        }
    }
    /// Java private `setReconnectRun(AxisID)`.
    fn set_reconnect_run(&self, axis_id: Option<AxisID>) {
        if axis_id == Some(AxisID::Second) {
            *self.reconnect_run_b.lock().unwrap() = true;
        } else {
            *self.reconnect_run_a.lock().unwrap() = true;
        }
    }
    /// Java `isSetupDone`.
    fn is_setup_done(&self) -> bool {
        self.dialog.lock().unwrap().is_some() && *self.base.loaded_param_file.lock().unwrap()
    }
    /// Java `setParamFile()`.
    fn set_param_file(&self) -> bool {
        *self.base.loaded_param_file.lock().unwrap()
    }
    // Java package-private `createMainPanel` is implemented above as the trait method.
}

impl std::fmt::Display for BatchRunTomoManager {
    /// Java inherited `toString` through `BaseManager`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BatchRunTomoManager[stackReferencePrefix:{STACK_REFERENCE_PREFIX}]"
        )
    }
}
