//! `IMOD/Etomo/src/etomo/BatchRunTomoManager.java`.
//!
//! The manager of the batchruntomo interface (`.ebt` data files).  It owns a
//! `BatchRunTomoMetaData`, a `BatchRunTomoScreenState`, a
//! `BatchRunTomoComScriptManager`, a `BatchRunTomoProcessManager`, the
//! `MainBatchRunTomoPanel` and the `BatchRunTomoDialog`.
//!
//! **Threads.**  The manager is a process-lifetime singleton shared with process
//! threads (`Send + Sync`); the main panel, the dialog, the dataset file builder and
//! the dialog-complete listeners are event dispatch thread objects, held in `EdtCell`s
//! and reached only on that thread.  The members the monitors call from their threads
//! (`findRow`, `getStack`, `msgStatusChangerStarted`, `setParameters`,
//! `startSeriesWatcherBatchMonitor`) are called by them through `invokeAndWait`;
//! `createRunList` makes that hop itself, since a monitor is constructed on either
//! thread.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use crate::imod::etomo::comscript::batch_run_tomo_com_script_manager::BatchRunTomoComScriptManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_details::CommandDetails;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::series_watcher_param::SeriesWatcherParam;
use crate::imod::etomo::comscript::split_batch_param::{self, SplitBatchParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::{AxisBusyException, BaseProcessManager};
use crate::imod::etomo::process::batch_run_tomo_process_manager::BatchRunTomoProcessManager;
use crate::imod::etomo::process::batch_run_tomo_process_monitor;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::log_feed_monitor::MessagesRef;
use crate::imod::etomo::process::process_data::ProcessData;
use crate::imod::etomo::process::process_interface::{ProcessResultDisplayRef, ProcessSeriesRef};
use crate::imod::etomo::process::process_messages::{MessagesArray, ProcessMessages};
use crate::imod::etomo::process::process_output_strings;
use crate::imod::etomo::process::series_watcher_process_monitor;
use crate::imod::etomo::process::tomosetexts_output::TomosetextsOutput;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::dataset_file_builder::DatasetFileBuilder;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::batch_run_tomo_meta_data::BatchRunTomoMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_screen_state::BatchRunTomoScreenState;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::extension_marker::ExtensionMarker;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_list::RunList;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::series_watcher_meta_data::SeriesWatcherMetaData;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_changer::StatusChanger;
use crate::imod::etomo::r#type::table_reference::TableReference;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::dialog_complete_listener::DialogCompleteListener;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;
use crate::imod::etomo::ui::swing::batch_run_tomo_dialog::{self, BatchRunTomoDialog};
use crate::imod::etomo::ui::swing::main_batch_run_tomo_panel::MainBatchRunTomoPanel;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::process_interface::ProcessInterface;
use crate::imod::etomo::ui::swing::text_page_window::TextPageWindow;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::swing::ui_parameters;
use crate::imod::etomo::util::clean_print::CleanPrint;
use crate::imod::etomo::util::event_queue::{self, EdtCell, EdtRef};
use crate::imod::etomo::util::utilities;

/// Java public static final `AXIS_ID`.
pub const AXIS_ID: AxisID = AxisID::Only;

/// Java private static final `STACK_REFERENCE_PREFIX =
/// DataFileType.BATCH_RUN_TOMO.extension.substring(1)` (".ebt" without the dot).
fn stack_reference_prefix() -> String {
    DataFileType::BatchRunTomo.extension().unwrap()[1..].to_owned()
}

/// Java private static final class `Task implements TaskInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// Java `PROCESSCHUNKS = new Task("processchunks")`.
    Processchunks,
}

impl TaskInterface for Task {
    /// Java `getDescr()`.
    fn get_descr(&self) -> Option<String> {
        Some("processchunks".to_owned())
    }

    /// Java `okToDrop()`.
    fn ok_to_drop(&self) -> bool {
        false
    }
}

/// `savePreferences(AXIS_ID, userConfig)`: `EtomoDirector`'s user configuration is
/// shared through accessors (see `etomo_director.rs`), so the `Storable` it is passed
/// as reaches it through them.
struct UserConfigStorable;

impl Storable for UserConfigStorable {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        etomo_director::INSTANCE
            .with_user_configuration(|user_config| user_config.store(properties));
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        etomo_director::INSTANCE.with_user_configuration(|user_config| {
            user_config.store_with_prepend(properties, prepend)
        });
    }
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        etomo_director::INSTANCE
            .with_user_configuration_mut(|user_config| user_config.load(properties));
    }
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        etomo_director::INSTANCE.with_user_configuration_mut(|user_config| {
            user_config.load_with_prepend(properties, prepend)
        });
    }
}

/// Java `public final class BatchRunTomoManager extends BaseManager`.
pub struct BatchRunTomoManager {
    /// Java superclass `BaseManager` state.
    base: BaseManagerBase,
    /// Java private final `tableReference = new TableReference(STACK_REFERENCE_PREFIX)`.
    table_reference: Arc<TableReference>,
    /// Java private final `comScriptManager` (needs `this`).
    com_script_manager: OnceLock<BatchRunTomoComScriptManager>,
    /// Java private final `screenState = new BatchRunTomoScreenState(AXIS_ID,
    /// AxisType.SINGLE_AXIS)`.
    screen_state: BatchRunTomoScreenState,
    /// Java private final `messagesArray`.  Reuse the message string feed because
    /// starting it is too slow for reconnect, and the reconnect finishes too fast for
    /// the string feed to complete.
    messages_array: MessagesArray,
    /// Java private final `seriesWatcherMessagesMap`.
    series_watcher_messages_map: Mutex<BTreeMap<String, MessagesRef>>,
    /// Java private final `datasetFileBuilder = new DatasetFileBuilder(this)`.
    dataset_file_builder: EdtCell<Rc<DatasetFileBuilder>>,
    /// Java private final `cleanPrint`.
    #[allow(dead_code)]
    clean_print: CleanPrint,
    /// Java private final `metaData`.
    meta_data: OnceLock<BatchRunTomoMetaData>,
    /// Java private final `processMgr`.
    process_mgr: OnceLock<&'static BatchRunTomoProcessManager>,
    /// Java private `mainPanel` (null in headless mode).
    main_panel: EdtCell<Rc<MainBatchRunTomoPanel>>,
    /// Java private `dialog`, initially null.
    dialog: EdtCell<Rc<BatchRunTomoDialog>>,
    /// Java private `reconnectRunA`: true if reconnect() has been run for the axis.
    reconnect_run_a: AtomicBool,
    /// Java private `reconnectRunB`.
    reconnect_run_b: AtomicBool,
    /// Java private `listeners`, initially null.
    listeners: EdtCell<Vec<Rc<DialogCompleteListener>>>,
    /// Java private `diagnostics = false` (never read).
    #[allow(dead_code)]
    diagnostics: AtomicBool,
}

/// Owns every `BatchRunTomoManager` this module builds (Java's owner is
/// `EtomoDirector.managerList`; the translation hands out `&'static Self`).
static INSTANCES: Mutex<Vec<&'static BatchRunTomoManager>> = Mutex::new(Vec::new());

impl BatchRunTomoManager {
    /// Java `BatchRunTomoManager()`: `this("")`.
    pub fn new() -> &'static BatchRunTomoManager {
        Self::new_with_param_file_name(Some(""))
    }

    /// Java `BatchRunTomoManager(String)`.
    pub fn new_with_param_file_name(param_file_name: Option<&str>) -> &'static BatchRunTomoManager {
        let instance: &'static BatchRunTomoManager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            table_reference: Arc::new(TableReference::new(&stack_reference_prefix())),
            com_script_manager: OnceLock::new(),
            screen_state: BatchRunTomoScreenState::new(AXIS_ID, AxisType::SingleAxis),
            messages_array: Arc::new(Mutex::new(Vec::new())),
            series_watcher_messages_map: Mutex::new(BTreeMap::new()),
            dataset_file_builder: EdtCell::new(),
            clean_print: CleanPrint::get_instance_with_label(Some("BatchRunTomoManager")),
            meta_data: OnceLock::new(),
            process_mgr: OnceLock::new(),
            main_panel: EdtCell::new(),
            dialog: EdtCell::new(),
            reconnect_run_a: AtomicBool::new(false),
            reconnect_run_b: AtomicBool::new(false),
            listeners: EdtCell::new(),
            diagnostics: AtomicBool::new(false),
        }));
        INSTANCES.lock().unwrap().push(instance);
        let _ = instance
            .com_script_manager
            .set(BatchRunTomoComScriptManager::new(instance));
        instance
            .dataset_file_builder
            .set(Some(Rc::new(DatasetFileBuilder::new(instance))));
        // super()
        instance.base_manager();
        let _ = instance.meta_data.set(BatchRunTomoMetaData::new(
            instance,
            instance.get_log_properties(),
            Arc::clone(&instance.table_reference),
            param_file_name.is_none_or(str::is_empty),
        ));
        let _ = instance
            .process_mgr
            .set(BatchRunTomoProcessManager::new(instance));
        instance.initialize_ui_parameters_from_name(param_file_name, Some(AXIS_ID));
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            instance.open_processing_panel();
            instance.set_status_bar_text();
            instance.open_batch_run_tomo_dialog();
            // Monitor is listened to by dialogs, so dialogs must exist before the
            // monitor is run.
            instance.reconnect(
                Some(
                    instance
                        .get_axis_process_data()
                        .get_saved_process_data(AXIS_ID),
                ),
                Some(AXIS_ID),
                false,
                Some(Arc::clone(&instance.messages_array)),
            );
        }
        if !instance.loaded_param_file() {
            instance.table_reference.set_new();
        }
        instance
    }

    /// The constructed manager at its final address (Java `this` inside the
    /// overrides that take `&self`).
    fn this_static(&self) -> &'static BatchRunTomoManager {
        INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|manager| std::ptr::eq(*manager, self))
            .expect("constructed BatchRunTomoManager")
    }

    fn com_script_manager(&self) -> &BatchRunTomoComScriptManager {
        self.com_script_manager.get().expect("comScriptManager")
    }

    fn process_mgr(&self) -> &'static BatchRunTomoProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    fn loaded_param_file(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    fn param_file(&self) -> Option<PathBuf> {
        self.base().param_file.lock().unwrap().clone()
    }

    fn dataset_file_builder(&self) -> Rc<DatasetFileBuilder> {
        self.dataset_file_builder
            .get()
            .expect("datasetFileBuilder is assigned by the constructor")
    }

    /// Java `mainPanel.setStatusBarText(paramFile, metaData, logWindow)`.  (Java
    /// dereferences mainPanel, which is null in headless mode; it is skipped then.)
    fn set_status_bar_text(&self) {
        if let Some(main_panel) = self.main_panel.get() {
            let param_file = self.param_file();
            let log_window = self.base().log_window.get();
            MainPanelVirtual::set_status_bar_text(
                &*main_panel,
                param_file.as_deref(),
                Some(self.get_meta_data() as &dyn BaseMetaData),
                log_window.as_ref(),
            );
        }
    }

    /// Java `uiHarness.openMessageDialog(this, message, title, axisID)`.
    fn open_message(&'static self, message: &str, title: &str, axis_id: Option<AxisID>) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                Some(self),
                message,
                title,
                axis_id,
            )
        });
    }

    /// Java `uiHarness.openMessageDialog(this, message, title)`.
    fn open_message_no_axis(&'static self, message: &str, title: &str) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string(Some(self), message, title)
        });
    }

    /// Java `uiHarness.openMessageDialog(this, String[], title, axisID)`.
    fn open_message_array(&'static self, message: &[String], title: &str, axis_id: AxisID) {
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_array_string_axis_id(
                Some(self),
                message,
                title,
                Some(axis_id),
            )
        });
    }

    /// Java `getMainPanel().stopProgressBar(AXIS_ID, ProcessEndState.FAILED)`.
    fn stop_progress_bar_failed(&self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .main_panel()
                .stop_progress_bar_axis_id_process_end_state(
                    AXIS_ID,
                    Some(ProcessEndState::Failed),
                );
        }
    }

    /// The dialog (Slint bridge and driver).
    pub fn get_dialog(&self) -> Option<Rc<BatchRunTomoDialog>> {
        self.dialog.get()
    }

    /// The screen state, as `(BatchRunTomoScreenState) getBaseScreenState(AXIS_ID)`.
    pub fn get_batch_run_tomo_screen_state(&self) -> &'static BatchRunTomoScreenState {
        &self.this_static().screen_state
    }

    /// Java `getMetaData()`.
    pub fn get_meta_data(&self) -> &'static BatchRunTomoMetaData {
        self.this_static()
            .meta_data
            .get()
            .expect("metaData is assigned by the constructor")
    }

    /// Java `setNewParamFile(File, String)`.
    pub fn set_new_param_file(
        &'static self,
        root_dir: Option<&Path>,
        root_name: Option<&str>,
    ) -> bool {
        if self.loaded_param_file() {
            return true;
        }
        // set paramFile and propertyUserDir
        let root_dir_path = utilities::java_io_file_get_absolute_path(
            &root_dir
                .map(|root_dir| root_dir.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
        if root_dir_path.ends_with(' ') {
            self.open_message(
                &format!(
                    "The directory, {root_dir_path}, cannot be used because it ends with a space."
                ),
                "Unusable Directory Name",
                Some(AxisID::Only),
            );
            return false;
        }
        self.set_property_user_dir(Some(&root_dir_path));
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("\npropertyUserDir: {root_dir_path}");
        }
        self.get_meta_data().set_root_name(root_name);
        let error_message = self.get_meta_data().validate();
        if let Some(error_message) = error_message {
            self.open_message(&error_message, "Batchruntomo Dialog error", Some(AXIS_ID));
            return false;
        }
        let meta_data_file_name = BaseMetaData::get_meta_data_file_name(self.get_meta_data());
        if !self.set_param_file_from(Some(
            &Path::new(&root_dir_path)
                .join(meta_data_file_name.unwrap_or_else(|| "null".to_owned())),
        )) {
            return false;
        }
        etomo_director::INSTANCE.rename_current_manager(self.get_meta_data().get_root_name());
        self.set_status_bar_text();
        true
    }

    /// Java private `openProcessingPanel()`.  MUST run reconnect for all axis.
    fn open_processing_panel(&'static self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_processing_panel(AxisType::SingleAxis);
        }
        self.set_panel();
    }

    /// Java `openBatchRunTomoDialog()`.
    pub fn open_batch_run_tomo_dialog(&'static self) {
        if !self.dialog.is_some() {
            let parallel_status_panel = self
                .get_main_panel()
                .and_then(|main_panel| main_panel.main_panel().get_parallel_status_panel(AXIS_ID));
            let dialog = BatchRunTomoDialog::get_instance(
                self,
                AXIS_ID,
                Arc::clone(&self.table_reference),
                parallel_status_panel,
            );
            self.dialog.set(Some(Rc::clone(&dialog)));
            self.dataset_file_builder().set_dataset_info_display(Some(
                dialog as Rc<dyn crate::imod::etomo::ui::dataset_info_display::DatasetInfoDisplay>,
            ));
        }
        let dialog = self.dialog.get().expect("dialog");
        let mut use_progress_bar = false;
        if self.param_file().is_some() {
            use_progress_bar = true;
            if let Some(main_panel) = self.main_panel.get() {
                main_panel
                    .main_panel()
                    .start_progress_bar_string_axis_id(Some("Loading Files"), AXIS_ID);
            }
        }
        self.init_dialog(None, false);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&dialog.get_container(), AXIS_ID);
        }
        ui_harness::with(|harness| harness.update_frame(Some(self)));
        let action_message =
            utilities::prepare_dialog_action_message(Some(dialog.get_dialog_type()), AXIS_ID, None);
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
        if use_progress_bar {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.main_panel().stop_progress_bar_axis_id(AXIS_ID);
            }
        }
        if let Some(listeners) = self.listeners.get() {
            if !self.is_setup_done() {
                eprintln!("ERROR: unable to notify DialogCompleteListener listeners.");
            } else {
                for listener in &listeners {
                    listener.msg_dialog_complete();
                }
            }
        }
    }

    /// Java `addDialogCompleteListener(DialogCompleteListener)`.
    pub fn add_dialog_complete_listener(&self, listener: Option<DialogCompleteListener>) {
        let Some(listener) = listener else {
            return;
        };
        if !self.listeners.is_some() {
            self.listeners.set(Some(Vec::new()));
        }
        self.listeners
            .with(|listeners| listeners.push(Rc::new(listener)));
    }

    /// Java `getBatchruntomoComfileNamingStyle()`.
    pub fn get_batchruntomo_comfile_naming_style(&self) -> Option<ConstEtomoNumber> {
        self.com_script_manager()
            .load_batch_run_tomo_root_name(AXIS_ID, Some(&self.get_meta_data().get_root_name()));
        let parallel_processing = self
            .dialog
            .get()
            .is_some_and(|dialog| dialog.is_parallel_processing());
        let param =
            self.com_script_manager()
                .get_batch_run_tomo_param(AXIS_ID, false, parallel_processing);
        Some(param.get_naming_style())
    }

    /// Java `getDatasetImageFilenameStyle()`.
    pub fn get_dataset_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        if let Some(dialog) = self.dialog.get() {
            return dialog.get_dataset_image_filename_style();
        }
        None
    }

    /// Java `tomosetexts(File)`.
    pub fn tomosetexts(&'static self, dir: Option<&Path>) -> Option<TomosetextsOutput> {
        // Java's `dir.exists()` dereferences a null dir.
        let dir = dir?;
        BaseProcessManager::tomosetexts(self, AXIS_ID, dir)
    }

    /// Java `initDialog(String, boolean)`.  `onlyStackIDDatasetDialog`: only loading
    /// the dataset dialog attached to the row with this stackID;
    /// `onlyAdvancedDatasetDialog`: only loading an advanced dataset dialog.
    pub fn init_dialog(
        &'static self,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        let init_entire_dialog =
            only_stack_id_dataset_dialog.is_none() && !only_advanced_dataset_dialog;
        let mut param: Option<BatchruntomoParam> = None;
        let param_file = self.param_file();
        if param_file.is_some() {
            self.com_script_manager().load_batch_run_tomo_root_name(
                AXIS_ID,
                Some(&self.get_meta_data().get_root_name()),
            );
            param = Some(self.com_script_manager().get_batch_run_tomo_param(
                AXIS_ID,
                false,
                dialog.is_parallel_processing(),
            ));
            // Load all parallel panel values - both tables must be open for this to
            // happen.
            let parallel_panel = self
                .get_main_panel()
                .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AXIS_ID));
            if let (Some(parallel_panel), Some(param)) = (parallel_panel, &param)
                && init_entire_dialog
            {
                parallel_panel.set_parameters(param);
            }
        }
        if init_entire_dialog {
            dialog.set_parameters_void();
            etomo_director::INSTANCE.with_user_configuration(|user_config| {
                dialog.set_parameters_user_configuration(user_config, param_file.is_none())
            });
            if param_file.is_some() {
                dialog.load_templates();
            }
        }
        dialog.update_directives(
            true,
            only_stack_id_dataset_dialog,
            only_advanced_dataset_dialog,
        );
        dialog.set_parameters_meta_data(
            self.get_meta_data(),
            only_stack_id_dataset_dialog,
            only_advanced_dataset_dialog,
            true,
        );
        // Load series watcher if it already exists.
        self.set_series_watcher_parameters();
        if param_file.is_some() {
            if init_entire_dialog && let Some(param) = &param {
                dialog.set_parameters_batchruntomo_param(param);
            }
            dialog.load_autodocs(only_stack_id_dataset_dialog, only_advanced_dataset_dialog);
        }
        if init_entire_dialog {
            dialog.msg_load_done();
        }
    }

    /// Java `setSeriesWatcherParameters()`.
    pub fn set_series_watcher_parameters(&self) {
        // Load series watcher if it already exists.
        if self
            .com_script_manager()
            .load_series_watcher(AXIS_ID, false)
        {
            let series_watcher_param = self
                .com_script_manager()
                .get_series_watcher_param(AXIS_ID, false);
            // Java dereferences dialog unguarded; every caller has one.
            if let Some(dialog) = self.dialog.get() {
                dialog.set_parameters_series_watcher_param(&series_watcher_param);
            }
        }
    }

    /// Java `setParameters(SeriesWatcherMetaData, String, boolean)`.
    pub fn set_parameters(
        &self,
        series_watcher_meta_data: &SeriesWatcherMetaData,
        stack_id: &str,
        init: bool,
    ) {
        if let Some(dialog) = self.dialog.get() {
            dialog.set_parameters_series_watcher_meta_data(
                series_watcher_meta_data,
                Some(stack_id),
                init,
            );
        }
    }

    /// Java `isDatasetDialog(String)`.
    pub fn is_dataset_dialog(&self, stack_id: &str) -> bool {
        let row_meta_data = self.get_meta_data().get_row_meta_data(stack_id);
        row_meta_data.is_dataset_dialog()
    }

    /// Java `msgStatusChangerStarted(StatusChanger, boolean)`.
    pub fn msg_status_changer_started(&self, changer: &Rc<dyn StatusChanger>, table_only: bool) {
        // Java dereferences dialog unguarded.
        if let Some(dialog) = self.dialog.get() {
            dialog.msg_status_changer_started(changer, table_only);
        }
    }

    /// Java private `updateSplitBatch()`.
    fn update_split_batch(&'static self) -> Option<SplitBatchParam> {
        let dialog = self.dialog.get()?;
        let mut param = SplitBatchParam::new(self);
        dialog.get_parameters_split_batch_param(&mut param);
        Some(param)
    }

    /// Java final package-private `processchunks(ProcessSeries, Command, String,
    /// RunType)`.  Run processchunks.
    fn processchunks_series(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        command: Option<&Arc<dyn Command + Send + Sync>>,
        premade_machine_list: Option<&str>,
        run_type: Option<RunType>,
    ) {
        let dialog = self.dialog.get();
        // `command instanceof ProcesschunksParam`, then the cast.
        let param = command.and_then(|command| {
            let command: Arc<dyn Command + Send + Sync> = Arc::clone(command);
            let any: Arc<dyn std::any::Any + Send + Sync> = command;
            any.downcast::<ProcesschunksParam>().ok()
        });
        let (Some(dialog), Some(param)) = (dialog, param) else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        param.set_premade_machine_list(premade_machine_list);
        let managed_process_data = Arc::new(Mutex::new(ProcessData::get_managed_instance(
            Some(AXIS_ID),
            Some(self),
            Some(ProcessName::PROCESSCHUNKS),
        )));
        if run_type == Some(RunType::Run) {
            let parallel_panel = self
                .get_main_panel()
                .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AXIS_ID));
            let Some(parallel_panel) = parallel_panel else {
                self.open_message(
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(AXIS_ID),
                );
                if let Some(process_series) = &process_series {
                    process_series.borrow().end_series();
                }
                return;
            };
            parallel_panel
                .get_parallel_progress_display()
                .reset_results();
        }
        <Self as BaseManager>::processchunks(
            self,
            Some(AXIS_ID),
            Some(param),
            None,
            process_series,
            true,
            Some(ProcessInterface::get_processing_method(&*dialog)),
            false,
            Some(batch_run_tomo_dialog::DIALOG_TYPE),
            run_type,
            Some(managed_process_data),
            Some(Arc::clone(&self.messages_array)),
        );
    }

    /// Java `findRow(String, String)`.
    pub fn find_row(&self, location: Option<&str>, root_name: Option<&str>) -> Option<String> {
        // Java dereferences dialog unguarded.
        self.dialog.get()?.find_row(location, root_name)
    }

    /// Java `getStack(String)`.
    pub fn get_stack(&self, stack_id: Option<&str>) -> Option<PathBuf> {
        // Java dereferences dialog unguarded.
        self.dialog.get()?.get_stack(stack_id)
    }

    /// Java `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed(&self, status: Option<BatchRunTomoStatus>) {
        if let Some(dialog) = self.dialog.get() {
            dialog.send_status_changed(status.map(StatusRef::BatchRunTomoStatus));
        }
    }

    /// Java `setPremadeMachineList(ProcessSeries, String[])`.  Create the machine
    /// list from an output string (see ProcessOutputStrings).
    pub fn set_premade_machine_list(
        &self,
        process_series: Option<ProcessSeriesRef>,
        std_output: Option<&[String]>,
    ) {
        let (Some(process_series), Some(std_output)) = (process_series, std_output) else {
            return;
        };
        let mut premade_machine_list: Option<String> = None;
        for line in std_output {
            if !line.contains(process_output_strings::SPLIT_BATCH_PROCESSCHUNKS_MACHINE_LIST_MSG_ID)
            {
                continue;
            }
            static WHITESPACE: std::sync::LazyLock<regex::Regex> =
                std::sync::LazyLock::new(|| regex::Regex::new(r"\s+").unwrap());
            let split_array = utilities::java_lang_string_split(line, &WHITESPACE);
            // The machine list is towards the end of the message so search backwards.
            for split in split_array.iter().rev() {
                // When a potentional machine list is found, check it by seeing if it is
                // preceded by a string that ends in ":"
                if premade_machine_list.is_some() {
                    if split.ends_with(':') {
                        break;
                    }
                    // Test failed. Continue to search.
                    premade_machine_list = None;
                }
                if split.contains(',') {
                    // Probably found the machine list
                    premade_machine_list = Some(split.clone());
                    continue;
                }
            }
            if premade_machine_list.is_some() {
                break;
            }
        }
        process_series
            .get()
            .borrow_mut()
            .set_next_process_parameter(premade_machine_list.as_deref());
    }

    /// Java `splitBatch()`.
    pub fn split_batch(&'static self) {
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        let process_series = ProcessSeries::new(
            self,
            AXIS_ID,
            Some(dialog.get_dialog_type()),
            Some("splitBatch"),
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .main_panel()
                .start_progress_bar_string_axis_id_process_name(
                    Some(&format!("Running {}", split_batch_param::command_name())),
                    AXIS_ID,
                    Some(&ProcessName::SPLIT_BATCH),
                );
        }
        let mut err_title: Option<String> = None;
        let mut err_msg: Option<String> = None;
        let mut process_end_state: Option<ProcessEndState> = None;
        let result = (|| -> Result<(), SplitBatchError> {
            // Save. Process chunks needs to be updated from the dialog.
            self.save_super()?;
            let batchruntomo_param = self.save_batch_run_tomo_dialog(
                Some(RunType::Run),
                true,
                false,
                None,
                false,
                true,
                false,
            );
            let mut param = None;
            let mut processchunks_param = None;
            if batchruntomo_param.is_some()
                && {
                    param = self.update_split_batch();
                    param.is_some()
                }
                && {
                    processchunks_param = self.update_process_chunks(
                        Some(AXIS_ID),
                        None,
                        Some(&self.get_meta_data().get_root_name()),
                        None,
                        Some(dialog.get_dialog_type()),
                    );
                    processchunks_param.is_some()
                }
            {
                // Setup up next process.
                process_series.borrow_mut().set_next_process_task_command(
                    Rc::new(Task::Processchunks),
                    Arc::new(processchunks_param.unwrap()),
                );
                // Run splitbatch.
                let thread_name = self.process_mgr().split_batch(
                    param.as_mut().unwrap(),
                    Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
                )?;
                self.set_thread_name(Some(&thread_name), Some(AXIS_ID));
            } else {
                process_end_state = Some(ProcessEndState::Failed);
            }
            Ok(())
        })();
        match result {
            Ok(()) | Err(SplitBatchError::LogFile(LogFileError::Lock(_))) => {}
            Err(SplitBatchError::LogFile(e)) => {
                eprintln!("{e}");
                err_title = Some("Unable to Save".to_owned());
                err_msg = Some(format!("System I/O exception.\n{}", e.get_message()));
            }
            Err(SplitBatchError::AxisBusy(e)) => {
                eprintln!("{}", e.0);
                err_title = Some("Unable to execute com script".to_owned());
                err_msg = Some(format!("Can not execute batchruntomo comfile\n{}", e.0));
            }
        }
        if process_end_state == Some(ProcessEndState::Failed) || err_msg.is_some() {
            if let Some(err_msg) = &err_msg {
                self.open_message(
                    err_msg,
                    err_title.as_deref().unwrap_or("null"),
                    Some(AXIS_ID),
                );
            }
            self.stop_progress_bar_failed();
            process_series.borrow().end_series();
        }
    }

    /// Java `batchruntomo(ProcessingMethod)`.
    pub fn batchruntomo(&'static self, processing_method: Option<ProcessingMethod>) {
        let result = (|| -> Result<(), LogFileError> {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel
                    .main_panel()
                    .set_progress_bar_string_axis_id_boolean(
                        Some(&format!("Running {}", ProcessName::BATCHRUNTOMO)),
                        AXIS_ID,
                        false,
                    );
            }
            self.save_super()?;
            let param = self.save_batch_run_tomo_dialog(
                Some(RunType::Run),
                true,
                false,
                None,
                false,
                false,
                false,
            );
            let Some(param) = param else {
                self.stop_progress_bar_failed();
                return Ok(());
            };
            // Create a processData entry for this run.
            let mut process_data = ProcessData::get_managed_instance(
                Some(AXIS_ID),
                Some(self),
                Some(ProcessName::BATCHRUNTOMO),
            );
            process_data.set_processing_method(processing_method);
            let process_data = Arc::new(Mutex::new(process_data));
            // Set up the message handler so it process and respond to messages
            // immediately.
            let thread_name = match self.process_mgr().batchruntomo(
                Some(Arc::new(param)),
                process_data,
                Some(self.setup_process_messages()),
                processing_method,
            ) {
                Ok(thread_name) => thread_name,
                Err(e) => {
                    eprintln!("{}", e.0);
                    let message = [
                        "Can not execute batchruntomo comfile".to_owned(),
                        e.0.clone(),
                    ];
                    self.open_message_array(&message, "Unable to execute com script", AXIS_ID);
                    self.stop_progress_bar_failed();
                    return Ok(());
                }
            };
            if let Some(thread_name) = thread_name {
                self.set_thread_name(Some(&thread_name), Some(AXIS_ID));
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => self.stop_progress_bar_failed(),
            Err(e) => {
                eprintln!("{e}");
                self.stop_progress_bar_failed();
            }
        }
    }

    /// Java `startSeriesWatcherBatchMonitor(File, String)`.
    pub fn start_series_watcher_batch_monitor(&'static self, batch_log: PathBuf, stack_id: &str) {
        let Some(dialog) = self.dialog.get() else {
            eprintln!("ERROR: Unable to start batchruntomo monitor.  Dialog unavailable.");
            return;
        };
        self.process_mgr().start_series_watcher_batch_submonitor(
            Some(stack_id),
            batch_log,
            Some(self.setup_series_watcher_batch_process_messages(Some(stack_id))),
            Some(ProcessInterface::get_processing_method(&*dialog)),
        );
    }

    /// Java `seriesWatcher(ProcessingMethod)`.
    pub fn series_watcher(&'static self, processing_method: Option<ProcessingMethod>) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .main_panel()
                .set_progress_bar_string_axis_id_boolean(
                    Some(&format!("Running {}", ProcessName::SERIES_WATCHER)),
                    AXIS_ID,
                    true,
                );
        }
        utilities::timestamp_full(Some("save for run"), Some("series watcher"), None, None);
        let result = (|| -> Result<(), LogFileError> {
            self.save_super()?;
            if self
                .save_batch_run_tomo_dialog(
                    Some(RunType::SeriesWatcher),
                    true,
                    false,
                    None,
                    false,
                    false,
                    false,
                )
                .is_none()
            {
                self.stop_progress_bar_failed();
                return Ok(());
            }
            let Some(param) = self.update_series_watcher(true) else {
                self.stop_progress_bar_failed();
                return Ok(());
            };
            // Create a processData entry for this run.
            let mut process_data = ProcessData::get_managed_instance(
                Some(AXIS_ID),
                Some(self),
                Some(ProcessName::SERIES_WATCHER),
            );
            process_data.set_processing_method(processing_method);
            let process_data = Arc::new(Mutex::new(process_data));
            // Set up the message handler so it process and respond to messages
            // immediately.
            let thread_name = match self.process_mgr().serieswatcher(
                Some(Arc::new(param)),
                process_data,
                Some(self.setup_series_watcher_process_messages()),
                processing_method,
                Arc::clone(&self.table_reference),
            ) {
                Ok(thread_name) => thread_name,
                Err(e) => {
                    eprintln!("{}", e.0);
                    let message = [
                        "Can not execute serieswatcher comfile".to_owned(),
                        e.0.clone(),
                    ];
                    self.open_message_array(&message, "Unable to execute com script", AXIS_ID);
                    self.stop_progress_bar_failed();
                    return Ok(());
                }
            };
            if let Some(thread_name) = thread_name {
                self.set_thread_name(Some(&thread_name), Some(AXIS_ID));
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => self.stop_progress_bar_failed(),
            Err(e) => {
                eprintln!("{e}");
                self.stop_progress_bar_failed();
            }
        }
    }

    /// Java package-private `setupProcessMessages()`.
    pub fn setup_process_messages(&'static self) -> MessagesRef {
        // Set up the message handler so it process and respond to messages
        // immediately.
        let messages = {
            let mut messages_array = self.messages_array.lock().unwrap();
            if messages_array.is_empty() {
                let messages = Arc::new(Mutex::new(
                    batch_run_tomo_process_monitor::create_process_messages_instance(self),
                ));
                messages_array.push(Some(Arc::clone(&messages)));
                messages
            } else {
                // Java's `messagesArray.get(0)`; a null element (which
                // ProcesschunksBatchRunTomoMonitor can store) is replaced here
                // rather than dereferenced.
                match &messages_array[0] {
                    Some(messages) => Arc::clone(messages),
                    None => {
                        let messages = Arc::new(Mutex::new(
                            batch_run_tomo_process_monitor::create_process_messages_instance(self),
                        ));
                        messages_array[0] = Some(Arc::clone(&messages));
                        messages
                    }
                }
            }
        };
        ProcessMessages::start_string_feed(&messages);
        messages.lock().unwrap().clear();
        messages
    }

    /// Java package-private `setupSeriesWatcherProcessMessages()`.
    pub fn setup_series_watcher_process_messages(&'static self) -> MessagesRef {
        let key = "SeriesWatcher";
        // Switch message handler to feed, so it processes and responds to messages
        // immediately.
        let messages = {
            let mut map = self.series_watcher_messages_map.lock().unwrap();
            match map.get(key) {
                Some(messages) => Arc::clone(messages),
                None => {
                    let messages = Arc::new(Mutex::new(
                        series_watcher_process_monitor::create_process_messages_instance(self),
                    ));
                    map.insert(key.to_owned(), Arc::clone(&messages));
                    messages
                }
            }
        };
        ProcessMessages::start_string_feed(&messages);
        messages.lock().unwrap().clear();
        messages
    }

    /// Java package-private `setupSeriesWatcherBatchProcessMessages(String)`.  Add the
    /// batchruntomo messages to the process message map for series watcher (which is
    /// running these batchruntomo instances).  Key them with the stack ID number from
    /// the .active version of the dataset file created by series watcher.
    pub fn setup_series_watcher_batch_process_messages(
        &'static self,
        stack_id: Option<&str>,
    ) -> MessagesRef {
        let mut messages: Option<MessagesRef> = None;
        match stack_id {
            None => {
                // The stack ID in the .active file provided by the series watcher must
                // start from 1. This is used here as the key for the process messages to
                // be used by the stack corresponding to this stackID. The messages for
                // series watcher use the 0 location of seriesWatcherMessagesMap.
                eprintln!("ERROR: Unable to store a message handler without a stackID.");
            }
            Some(stack_id) => {
                // Setting up a message handler is slow, so reuse an existing one if it
                // exists.
                messages = self
                    .series_watcher_messages_map
                    .lock()
                    .unwrap()
                    .get(stack_id)
                    .cloned();
            }
        }
        let messages = match messages {
            Some(messages) => messages,
            None => {
                let messages = Arc::new(Mutex::new(
                    batch_run_tomo_process_monitor::create_process_messages_instance(self),
                ));
                if let Some(stack_id) = stack_id {
                    self.series_watcher_messages_map
                        .lock()
                        .unwrap()
                        .insert(stack_id.to_owned(), Arc::clone(&messages));
                }
                messages
            }
        };
        // Set up message handler.
        ProcessMessages::start_string_feed(&messages);
        messages.lock().unwrap().clear();
        messages
    }

    /// Java `resumeBatchruntomo(ProcessingMethod)`.
    pub fn resume_batchruntomo(&'static self, _processing_method: Option<ProcessingMethod>) {
        // Get the saved comscript
        self.com_script_manager().load_batch_run_tomo(AXIS_ID);
        let mut param = self
            .com_script_manager()
            .get_batch_run_tomo_param(AXIS_ID, true, false);
        // Add datasets that are still checked
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        if !dialog.get_parameters_batchruntomo_param(
            &mut param,
            Some(RunType::Resume),
            true,
            true,
            false,
        ) {
            return;
        }
        self.com_script_manager()
            .save_batch_run_tomo(&param, AXIS_ID);
        //
        // Create a processData entry for this run and some of the saved data.
        let mut process_data = ProcessData::get_managed_instance(
            Some(AXIS_ID),
            Some(self),
            Some(ProcessName::BATCHRUNTOMO),
        );
        process_data.set_processing_method(_processing_method);
        let process_data = Arc::new(Mutex::new(process_data));
        // Java reads `axisProcessData.getSavedProcessData(AXIS_ID)` into an unused
        // local.
        let _saved_process_data = self.get_axis_process_data().get_saved_process_data(AXIS_ID);
        // run batchruntomo
        let thread_name = match self.process_mgr().resume_batchruntomo(
            Some(Arc::new(param)),
            process_data,
            Some(self.setup_process_messages()),
            None,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{}", e.0);
                let message = [
                    "Can not execute batchruntomo comfile".to_owned(),
                    e.0.clone(),
                ];
                self.open_message_array(&message, "Unable to execute com script", AXIS_ID);
                None
            }
        };
        if let Some(thread_name) = thread_name {
            self.set_thread_name(Some(&thread_name), Some(AXIS_ID));
        }
    }

    /// Java private `retrieveScreenStateFromDialog()`.
    fn retrieve_screen_state_from_dialog(&self) {
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        dialog.retrieve_screen_state_from_dialog(&self.this_static().screen_state);
    }

    /// Java `saveBatchRunTomoDialog(RunType, boolean, boolean, String, boolean,
    /// boolean, boolean)`.  Save the state of the dialog.  `autodocStackID`: don't
    /// save all dataset autodocs - just this one; `onlyGlobalAutodoc`: don't save any
    /// dataset autodocs.
    #[allow(clippy::too_many_arguments)]
    pub fn save_batch_run_tomo_dialog(
        &'static self,
        run_type: Option<RunType>,
        do_validation: bool,
        init: bool,
        autodoc_stack_id: Option<&str>,
        only_global_autodoc: bool,
        parallel_processing: bool,
        diagnostics: bool,
    ) -> Option<BatchruntomoParam> {
        let dialog = self.dialog.get()?;
        // Do as much validation as is possible without loading the param file. This
        // helps keep the root fields unlocked until deliveries are attempted,
        // batchruntomo starts, or the project is saved.
        let mut validate_only = true;
        if !self.loaded_param_file() && do_validation {
            if !dialog.save_autodocs(
                &self.dataset_file_builder(),
                do_validation,
                init,
                autodoc_stack_id,
                only_global_autodoc,
                validate_only,
            ) {
                return None;
            }
            self.update_batch_run_tomo(
                run_type,
                do_validation,
                parallel_processing,
                validate_only,
            )?;
        }
        validate_only = false;

        if self.param_file().is_none()
            && (!dialog.is_param_file_modifiable()
                || !self.set_new_param_file(
                    dialog.get_root_dir().as_deref(),
                    dialog.get_root_name().as_deref(),
                ))
        {
            // setting the param file failed
            return None;
        }
        *self.base().loaded_param_file.lock().unwrap() = true;
        dialog.disable_root_fields();
        ui_harness::with(|harness| harness.set_enabled_new_batch_run_tomo_menu_item(true));
        if init {
            etomo_director::INSTANCE.with_user_configuration_mut(|user_config| {
                dialog.get_parameters_user_configuration(user_config)
            });
        }
        dialog.get_parameters_meta_data(self.get_meta_data());
        dialog.retrieve_screen_state_from_dialog(&self.this_static().screen_state);
        if !dialog.save_autodocs(
            &self.dataset_file_builder(),
            do_validation,
            init,
            autodoc_stack_id,
            only_global_autodoc,
            validate_only,
        ) {
            return None;
        }
        let param = self.update_batch_run_tomo(
            run_type,
            do_validation,
            parallel_processing,
            validate_only,
        )?;

        self.save_storables(Some(AXIS_ID));
        self.save_preferences(Some(AXIS_ID), Some(&UserConfigStorable));
        if diagnostics {
            let result = (|| -> Result<bool, LogFileError> {
                LogFile::get_instance_file(
                    self.param_file().as_deref(),
                    Some(self.get_emergency_monitor(Some(AXIS_ID))),
                )?
                .copy_to_numbered_file(
                    Some(self),
                    Some(ExtensionMarker::Generic),
                    2,
                )
            })();
            match result {
                Ok(_) | Err(LogFileError::Lock(_)) => {}
                Err(e) => eprintln!("{}", e.get_message()),
            }
        }
        Some(param)
    }

    /// Java private `updateSeriesWatcher(boolean)`.
    fn update_series_watcher(&'static self, do_validation: bool) -> Option<SeriesWatcherParam> {
        if !self
            .com_script_manager()
            .load_series_watcher(AXIS_ID, false)
        {
            if !do_validation {
                // Series watcher is optional functionality, and the comfile should only
                // be created if the user wants to run it. When doValidation is off, the
                // user is just saving.
                return None;
            }
            BaseProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(
                    &file_type::CLASS
                        .series_watcher_comscript
                        .get_file(Some(self), Some(AXIS_ID))
                        .map(|file| file.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                ),
                Some(self),
            );
            self.com_script_manager().load_series_watcher(AXIS_ID, true);
        }
        let mut param = self
            .com_script_manager()
            .get_series_watcher_param(AXIS_ID, do_validation);
        let dialog = self.dialog.get()?;
        if !dialog.get_parameters_series_watcher_param(&mut param, do_validation) {
            return None;
        }
        self.com_script_manager()
            .save_series_watcher(&param, AXIS_ID);
        Some(param)
    }

    /// Java private `updateBatchRunTomo(RunType, boolean, boolean, boolean)`.
    fn update_batch_run_tomo(
        &'static self,
        run_type: Option<RunType>,
        do_validation: bool,
        parallel_processing: bool,
        validate_only: bool,
    ) -> Option<BatchruntomoParam> {
        if !validate_only && !self.com_script_manager().is_batch_run_tomo_loaded() {
            let comscript_file = self
                .dataset_file_builder()
                .build_file(Some(&file_type::CLASS.batch_run_tomo_comscript), AXIS_ID);
            BaseProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(
                    &comscript_file
                        .map(|file| file.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                ),
                Some(self),
            );
            self.com_script_manager().load_batch_run_tomo(AXIS_ID);
        }
        let mut param = self.com_script_manager().get_batch_run_tomo_param(
            AXIS_ID,
            do_validation,
            parallel_processing,
        );
        let dialog = self.dialog.get()?;
        if !dialog.get_parameters_batchruntomo_param(
            &mut param,
            run_type,
            do_validation,
            false,
            validate_only,
        ) {
            return None;
        }
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(AXIS_ID));
        if let Some(parallel_panel) = parallel_panel
            && !parallel_panel.get_parameters_batchruntomo_param_boolean_boolean(
                &mut param,
                do_validation,
                validate_only,
            )
        {
            return None;
        }
        param.set_smtp_server(
            etomo_director::INSTANCE
                .with_user_configuration(|user_config| user_config.get_smtp_server())
                .as_deref(),
        );
        if !validate_only {
            self.com_script_manager()
                .save_batch_run_tomo(&param, AXIS_ID);
        }
        if !param.is_valid() {
            return None;
        }
        Some(param)
    }

    /// Java `printStackTrace` + `openMessageDialog` for the 3dmod open exceptions:
    /// `AxisTypeException` -> "AxisType problem", `SystemProcessException` ->
    /// "Problem opening <key>", `IOException` -> "IO Exception".
    fn imod_error(
        &'static self,
        key: &str,
        result: Result<i32, ImodManagerException>,
        imod_index: i32,
        axis_id: Option<AxisID>,
    ) -> i32 {
        match result {
            Ok(index) => index,
            Err(except @ ImodManagerException::AxisType(_)) => {
                eprintln!("{except}");
                self.imod_message(&except.to_string(), "AxisType problem", axis_id);
                imod_index
            }
            Err(except @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{except}");
                self.imod_message(
                    &except.to_string(),
                    &format!("Problem opening {key}"),
                    axis_id,
                );
                imod_index
            }
            Err(e @ ImodManagerException::Io(_)) => {
                eprintln!("{e}");
                self.imod_message(&e.to_string(), "IO Exception", axis_id);
                imod_index
            }
            // An uncaught RuntimeException: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
                imod_index
            }
        }
    }

    fn imod_message(&'static self, message: &str, title: &str, axis_id: Option<AxisID>) {
        match axis_id {
            Some(axis_id) => self.open_message(message, title, Some(axis_id)),
            None => self.open_message_no_axis(message, title),
        }
    }

    /// Java private `isReconnectRun(AxisID)`: this class's own (Java private methods
    /// are not overridden, so `BaseManager`'s reconnect flags are separate).
    fn is_reconnect_run(&self, axis_id: Option<AxisID>) -> bool {
        if axis_id == Some(AxisID::Second) {
            return self.reconnect_run_b.load(Ordering::SeqCst);
        }
        self.reconnect_run_a.load(Ordering::SeqCst)
    }

    /// Java private `setReconnectRun(AxisID)`.
    fn set_reconnect_run(&self, axis_id: Option<AxisID>) {
        if axis_id == Some(AxisID::Second) {
            self.reconnect_run_b.store(true, Ordering::SeqCst);
        } else {
            self.reconnect_run_a.store(true, Ordering::SeqCst);
        }
    }

    /// Java `imodStack(File, AxisID, int, File, boolean, Run3dmodMenuOptions)`.  Open
    /// imod with an optional model file.
    pub fn imod_stack(
        &'static self,
        stack: Option<&Path>,
        axis_id: AxisID,
        mut imod_index: i32,
        model_file: Option<&Path>,
        _dual_axis: bool,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> i32 {
        let Some(stack) = stack else {
            return imod_index;
        };
        if !stack.exists() {
            self.open_message_no_axis(
                &format!(
                    "{} does not exist.",
                    utilities::java_io_file_get_absolute_path(&stack.to_string_lossy())
                ),
                "Run 3dmod failed",
            );
            return imod_index;
        }
        let key = imod_manager::BATCH_RUN_TOMO_STACK_KEY;
        // Try to set the file first. If the delivery was done since this was last
        // opened, the file has changed.
        let result = self.get_imod_manager().set_file_string_file_axis_id_int(
            key,
            Some(stack),
            Some(axis_id),
            imod_index,
        );
        imod_index = self.imod_error(key, result, imod_index, None);
        // Open the file.
        let result = self
            .get_imod_manager()
            .open_string_file_axis_id_int_file_boolean_run3dmod_menu_options(
                key,
                Some(stack),
                Some(axis_id),
                imod_index,
                model_file,
                model_file.is_some(),
                menu_options,
            );
        self.imod_error(key, result, imod_index, None)
    }

    /// Java `imodRec(File, String, AxisType, int, Run3dmodMenuOptions)`.  Opens the
    /// tomogram.
    pub fn imod_rec(
        &'static self,
        dataset_dir: Option<&Path>,
        dataset: Option<&str>,
        axis_type: AxisType,
        imod_index: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> i32 {
        let (Some(dataset_dir), Some(dataset)) = (dataset_dir, dataset) else {
            self.open_message_no_axis(
                "Unable to open - tomogram has not been built.",
                "Run 3dmod failed",
            );
            return imod_index;
        };
        let key = imod_manager::BATCH_RUN_TOMO_REC_KEY;
        let file_type: &FileType = if axis_type == AxisType::DualAxis {
            &file_type::CLASS.combined_volume
        } else {
            &file_type::CLASS.tilt_output_single
        };
        let file = file_type.get_file_in_dir(
            Some(self),
            Some(dataset_dir),
            Some(dataset),
            Some(axis_type),
            Some(AxisID::First),
        );
        let result = self
            .get_imod_manager()
            .open_string_file_int_run3dmod_menu_options(
                key,
                file.as_deref(),
                imod_index,
                menu_options,
            );
        self.imod_error(key, result, imod_index, None)
    }

    /// Java `imodTrimvol(File, String, AxisType, int, Run3dmodMenuOptions)`.  Opens the
    /// trimvol output.
    pub fn imod_trimvol(
        &'static self,
        dataset_dir: Option<&Path>,
        dataset: Option<&str>,
        axis_type: AxisType,
        imod_index: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> i32 {
        let (Some(dataset_dir), Some(dataset)) = (dataset_dir, dataset) else {
            self.open_message_no_axis(
                "Unable to open - tomogram has not been built.",
                "Run 3dmod failed",
            );
            return imod_index;
        };
        let key = imod_manager::BATCH_RUN_TOMO_TRIMMED_VOLUME_KEY;
        let file = file_type::CLASS.trim_vol_output.get_file_in_dir(
            Some(self),
            Some(dataset_dir),
            Some(dataset),
            Some(axis_type),
            Some(AxisID::First),
        );
        let result = self
            .get_imod_manager()
            .open_string_file_int_run3dmod_menu_options(
                key,
                file.as_deref(),
                imod_index,
                menu_options,
            );
        self.imod_error(key, result, imod_index, None)
    }

    /// Java `imodModel(AxisID, int, File, String, FileType, boolean)`.  Open imod model
    /// in a running imod instance.
    pub fn imod_model(
        &'static self,
        axis_id: AxisID,
        imod_index: i32,
        stack_location: Option<&Path>,
        stack_name: Option<&str>,
        model_file_type: Option<&FileType>,
        dual_axis: bool,
    ) {
        let (Some(model_file_type), Some(stack_name)) = (model_file_type, stack_name) else {
            return;
        };
        if imod_index == -1 {
            return;
        }
        let key = imod_manager::BATCH_RUN_TOMO_STACK_KEY;
        let axis_type = if dual_axis {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        };
        let dataset_name = dataset_tool::get_dataset_name(Some(stack_name), dual_axis);
        let model_file = model_file_type.get_file_in_dir(
            Some(self),
            stack_location,
            dataset_name.as_deref(),
            Some(axis_type),
            Some(axis_id),
        );
        if let Some(model_file) = model_file {
            let result = self
                .get_imod_manager()
                .open_model(
                    key,
                    Some(axis_id),
                    imod_index,
                    &utilities::escape_spaces_double(
                        &utilities::java_io_file_get_absolute_path(&model_file.to_string_lossy()),
                        true,
                    ),
                    true,
                )
                .map(|()| imod_index);
            self.imod_error(key, result, imod_index, Some(axis_id));
        }
    }

    /// Java `openLog(File)`.
    pub fn open_log(&self, log: Option<&Path>) {
        let Some(log) = log else {
            return;
        };
        let log_file_window = TextPageWindow::new(ui_parameters::DEFAULT_FONT_SIZE as i32);
        log_file_window.set_visible(log_file_window.set_file_from_file(log));
    }
}

/// The two exception families `splitBatch` catches.
enum SplitBatchError {
    LogFile(LogFileError),
    AxisBusy(AxisBusyException),
}

impl From<LogFileError> for SplitBatchError {
    fn from(e: LogFileError) -> SplitBatchError {
        SplitBatchError::LogFile(e)
    }
}

impl From<AxisBusyException> for SplitBatchError {
    fn from(e: AxisBusyException) -> SplitBatchError {
        SplitBatchError::AxisBusy(e)
    }
}

impl BaseManager for BatchRunTomoManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }

    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    /// Java `isAddGPUMachineToProcessChunks()`.
    fn is_add_gpu_machine_to_process_chunks(&self) -> bool {
        true
    }

    /// Java package-private `isPopupChunkWarnings()`.
    fn is_popup_chunk_warnings(&self) -> bool {
        false
    }

    /// Java `isDualSelectionQueueTable()`.
    fn is_dual_selection_queue_table(&self) -> bool {
        true
    }

    /// Java package-private `getMessagesArray()`.
    fn get_messages_array(&self) -> Option<MessagesArray> {
        Some(Arc::clone(&self.messages_array))
    }

    /// Java `exitProgram(AxisID)`.  Call BaseManager.exitProgram().  Call saveDialog.
    /// Return the value of BaseManager.exitProgram().  To guarantee that etomo can
    /// always exit, catch all unrecognized Exceptions and Errors and return true.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.exit_program_super(axis_id) {
                self.end_threads();
                self.save_param_file()?;
                self.process_mgr().halt_monitor_thread();
                for messages in self.messages_array.lock().unwrap().iter().flatten() {
                    ProcessMessages::stop_string_feed(messages);
                }
                for messages in self.series_watcher_messages_map.lock().unwrap().values() {
                    ProcessMessages::stop_string_feed(messages);
                }
                return Ok(true);
            }
            Ok::<bool, LogFileError>(false)
        }));
        match result {
            Ok(Ok(exit)) => exit,
            Ok(Err(e)) => {
                eprintln!("{e:?}");
                true
            }
            Err(_) => true,
        }
    }

    /// Java `pack()`.
    fn pack(&self) {
        if let Some(dialog) = self.dialog.get() {
            dialog.pack();
        }
    }

    /// Java `getBaseScreenState(AxisID)`.
    fn get_base_screen_state(&self, _axis_id: Option<AxisID>) -> Option<&'static BaseScreenState> {
        Some(&*self.this_static().screen_state)
    }

    /// Java `save() throws LogFileException, IOException, LockException`.  Save on
    /// exit.
    fn save(&'static self) -> Result<bool, LogFileError> {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .main_panel()
                .start_progress_bar_string_axis_id(Some("Saving Files"), AXIS_ID);
        }
        utilities::timestamp_full(Some("save"), Some("start"), None, None);
        self.save_super()?;
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.main_panel().done();
        }
        let parameter_store = self.get_parameter_store(Some(AXIS_ID))?;
        self.retrieve_screen_state_from_dialog();
        if let Some(parameter_store) = parameter_store {
            parameter_store
                .lock()
                .unwrap()
                .save(Some(&self.this_static().screen_state))?;
        }
        self.save_super()?;
        let parallel_processing = self
            .dialog
            .get()
            .is_some_and(|dialog| dialog.is_parallel_processing());
        self.save_batch_run_tomo_dialog(
            None,
            false,
            false,
            None,
            false,
            parallel_processing,
            false,
        );
        self.update_series_watcher(false);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.main_panel().stop_progress_bar_axis_id(AXIS_ID);
        }
        Ok(true)
    }

    /// Java `startNextProcess(UIComponent, AxisID, ProcessSeries.Process,
    /// ProcessResultDisplay, ProcessSeries, DialogType, ProcessDisplay)`.  Returns true
    /// if the process is recognized.
    fn start_next_process(
        &'static self,
        _ui_component: Option<Rc<dyn UiComponent>>,
        _axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        self.start_next_process_super(
            None,
            AXIS_ID,
            process,
            process_result_display,
            process_series,
            dialog_type,
            display,
        );
        if process.equals_task(&Task::Processchunks) {
            // After split.
            self.processchunks_series(
                Some(Rc::clone(process_series)),
                process.get_command(),
                process.get_parameter(),
                Some(RunType::Run),
            );
            return true;
        }
        false
    }

    /// Java `sendEvent(AxisID, ProcessName, ProcessEndState, boolean)`.
    fn send_event(
        &self,
        axis_id: Option<AxisID>,
        process_name: Option<ProcessName>,
        process_end_state: Option<ProcessEndState>,
        failed: bool,
    ) {
        let Some(dialog) = self.dialog.get() else {
            return;
        };
        let Some(process_name) = process_name else {
            return;
        };
        if axis_id == Some(AxisID::Second)
            || process_name == ProcessName::SPLIT_BATCH
            || process_end_state == Some(ProcessEndState::Cancelled)
            || (process_end_state.is_none() && !failed)
        {
            return;
        }
        let mut status: Option<BatchRunTomoStatus> = None;
        if process_end_state == Some(ProcessEndState::Failed)
            || process_end_state == Some(ProcessEndState::FileLockFailure)
            || failed
        {
            status = Some(BatchRunTomoStatus::Failed);
        } else if process_end_state == Some(ProcessEndState::Done) {
            status = Some(BatchRunTomoStatus::Done);
        } else if process_end_state == Some(ProcessEndState::Killed)
            || process_end_state == Some(ProcessEndState::Paused)
        {
            if process_name == ProcessName::BATCHRUNTOMO {
                status = Some(BatchRunTomoStatus::KilledOrPaused);
            } else if process_name == ProcessName::PROCESSCHUNKS {
                status = Some(BatchRunTomoStatus::KilledOrPausedProcessChunks);
            } else if process_name == ProcessName::SERIES_WATCHER {
                status = Some(BatchRunTomoStatus::KilledOrPausedSeriesWatcher);
            }
        }
        if let Some(status) = status {
            dialog.status_changed_status(Some(StatusRef::BatchRunTomoStatus(status)));
        }
    }

    /// Java package-private `updateProcessChunks(AxisID, ProcesschunksParam, String,
    /// CommandDetails, DialogType)`.
    fn update_process_chunks(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<ProcesschunksParam>,
        root_name: Option<&str>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        dialog_type: Option<DialogType>,
    ) -> Option<ProcesschunksParam> {
        let axis = axis_id.unwrap_or(AxisID::Only);
        let mut param = match param {
            Some(param) => param,
            None => ProcesschunksParam::get_instance_interface_type(
                self,
                axis,
                root_name,
                None,
                Some(InterfaceType::BatchRunTomo),
            ),
        };
        // Java dereferences dialog unguarded.
        let dialog = self.dialog.get()?;
        AbstractParallelDialog::get_parameters(&*dialog, &mut param);
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(axis));
        let Some(parallel_panel) = parallel_panel else {
            self.open_message(
                &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                "Unable to execute command",
                axis_id,
            );
            return None;
        };
        if parallel_panel.get_parameters_processchunks_param_boolean(&param, true) {
            return self.update_process_chunks_super(
                axis_id,
                Some(param),
                root_name,
                subcommand_details,
                dialog_type,
            );
        }
        None
    }

    /// Java `createRunList(RunType)`.  The list is built on the event dispatch thread
    /// (the dialog's), whichever thread asks.
    fn create_run_list(&self, run_type: Option<RunType>) -> Option<Arc<RunList>> {
        let this = self.this_static();
        event_queue::invoke_and_wait(move || {
            let dialog = this.dialog.get()?;
            Some(Arc::new(dialog.create_run_list(run_type)))
        })
    }

    /// Java `reconnect(ProcessData, AxisID, boolean, List<ProcessMessages>)`.
    /// Attempts to reconnect to a currently running process.  Only run once per axis.
    /// Only attempts one reconnect.  Returns true if a reconnect was attempted.
    fn reconnect(
        &'static self,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<MessagesArray>,
    ) -> bool {
        let axis = axis_id.unwrap_or(AxisID::Only);
        if self.reconnect_super(
            process_data.clone(),
            axis_id,
            multi_line_messages,
            messages_array.clone(),
        ) {
            return true;
        }
        if BatchRunTomoManager::is_reconnect_run(self, axis_id) {
            if let Some(process_manager) = self.get_process_manager() {
                process_manager.unblock_axis(axis);
            }
            return false;
        }
        BatchRunTomoManager::set_reconnect_run(self, axis_id);
        let Some(process_data) = process_data else {
            return false;
        };
        let (process_name, is_on_different_host) = {
            let data = process_data.lock().unwrap();
            (data.get_process_name(), data.is_on_different_host())
        };
        let Some(process_name) = process_name else {
            return false;
        };
        if is_on_different_host && !self.reconnect_to_different_host(Some(&process_data), axis_id) {
            return false;
        }
        // Build and start string feed in a ProcessMessage instance.
        let messages_array = messages_array.unwrap_or_else(|| Arc::clone(&self.messages_array));
        let messages = {
            let mut messages_array = messages_array.lock().unwrap();
            if messages_array.is_empty() {
                let messages = Arc::new(Mutex::new(
                    batch_run_tomo_process_monitor::create_process_messages_instance(self),
                ));
                messages_array.push(Some(Arc::clone(&messages)));
                messages
            } else {
                match &messages_array[0] {
                    Some(messages) => Arc::clone(messages),
                    None => {
                        let messages = Arc::new(Mutex::new(
                            batch_run_tomo_process_monitor::create_process_messages_instance(self),
                        ));
                        messages_array[0] = Some(Arc::clone(&messages));
                        messages
                    }
                }
            }
        };
        ProcessMessages::start_string_feed(&messages);
        // ProcessData will only be saved if the process was running when etomo exited.
        // Even if the process is no longer running, reconnect in order to update the
        // display.
        let debug = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
        if debug {
            eprintln!(
                "\nAttempting to reconnect in Axis {axis}\n{}",
                process_data.lock().unwrap().to_source_string()
            );
        }
        if process_name == ProcessName::BATCHRUNTOMO {
            ProcessMessages::start_string_feed(&messages);
            if debug {
                eprintln!(
                    "\nAttempting to reconnect in Axis {axis}\n{}",
                    process_data.lock().unwrap().to_source_string()
                );
            }
            if self
                .process_mgr()
                .reconnect_batchruntomo(Arc::clone(&process_data), Some(Arc::clone(&messages)))
            {
                self.set_thread_name(Some(&process_name.to_string()), axis_id);
                return true;
            }
        } else if process_name == ProcessName::SERIES_WATCHER {
            ProcessMessages::start_string_feed(&messages);
            if debug {
                eprintln!(
                    "\nAttempting to reconnect in Axis {axis}\n{}",
                    process_data.lock().unwrap().to_source_string()
                );
            }
            if self.process_mgr().reconnect_serieswatcher(
                Arc::clone(&process_data),
                Some(Arc::clone(&messages)),
                Arc::clone(&self.table_reference),
            ) {
                self.set_thread_name(Some(&process_name.to_string()), axis_id);
                return true;
            }
        } else if let Some(process_manager) = self.get_process_manager() {
            process_manager.unblock_axis(axis);
        }
        false
    }

    /// Java package-private `reconnectToDifferentHost(ProcessData, AxisID)`.
    /// Batchruntomo processes can always be reconnected via a different host.  Their
    /// monitors are all type DetachedProcessMonitor, so they don't require any
    /// connection to the original process.  And they are controlled via a file, so
    /// it's not necessary to be on the original host to kill, pause, and resume them.
    /// Batchruntomo processes also need to be updated in the dialog even after they
    /// have completed.
    fn reconnect_to_different_host(
        &'static self,
        _process_data: Option<&Arc<Mutex<ProcessData>>>,
        _axis_id: Option<AxisID>,
    ) -> bool {
        true
    }

    /// Java `isSetupDone()`.
    fn is_setup_done(&self) -> bool {
        self.dialog
            .get()
            .is_some_and(|dialog| !dialog.is_param_file_empty())
    }

    /// Java `setParamFile()`.
    fn set_param_file(&self) -> bool {
        if self.loaded_param_file() {
            return true;
        }
        let this = self.this_static();
        // Java dereferences dialog unguarded.
        let Some(dialog) = self.dialog.get() else {
            return false;
        };
        let root_name = dialog.get_root_name();
        self.get_meta_data().set_name(root_name.as_deref());
        let param_file = dialog.get_root_dir().unwrap_or_default().join(format!(
            "{}{}",
            root_name.as_deref().unwrap_or("null"),
            DataFileType::BatchRunTomo.extension().unwrap_or("null")
        ));
        *self.base().param_file.lock().unwrap() = Some(param_file.clone());
        if !self.set_param_file_from_super(Some(&param_file)) {
            return false;
        }
        if !param_file.exists() {
            BaseProcessManager::touch(
                &utilities::java_io_file_get_absolute_path(&param_file.to_string_lossy()),
                Some(this),
            );
        }
        this.initialize_ui_parameters(Some(&param_file), Some(AxisID::Only), false);
        // Update main window information and status bar
        etomo_director::INSTANCE.rename_current_manager(
            self.get_meta_data()
                .get_name()
                .unwrap_or_else(|| "null".to_owned()),
        );
        self.set_status_bar_text();
        true
    }

    /// Java package-private `createMainPanel()`.
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let this = self.this_static();
            self.main_panel.set(Some(MainBatchRunTomoPanel::new(this)));
        }
    }

    /// Java `getBaseMetaData()`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .get()
            .map(|meta_data| meta_data as &dyn BaseMetaData)
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java `getInterfaceType()`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::BatchRunTomo)
    }

    /// Java `getProcessManager()`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.meta_data
            .get()
            .and_then(|meta_data| meta_data.get_name())
    }

    /// Java package-private `getStorables(int)`.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>> {
        let this = self.this_static();
        let mut storables: Vec<Option<&'static dyn Storable>> = vec![None; (1 + offset) as usize];
        let index = offset as usize;
        storables[index] = this
            .meta_data
            .get()
            .map(|meta_data| meta_data as &'static dyn Storable);
        Some(storables)
    }
}

/// Unused import guard for `ParameterStore` (the dialog loads the screen state
/// through it).
#[allow(dead_code)]
type _ParameterStore = ParameterStore;
/// Unused alias guard for `ProcessResultDisplayHandle`.
#[allow(dead_code)]
type _ProcessResultDisplayHandle = ProcessResultDisplayHandle;
