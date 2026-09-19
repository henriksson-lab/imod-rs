//! `IMOD/Etomo/src/etomo/BaseManager.java`.
//!
//! Base class for interface managers like `ApplicationManager` and `JoinManager`.
//!
//! **Representation.**  `BaseManager` is an abstract class with seven abstract methods
//! and one implemented interface (`etomo/ui/BrowsingDirectory.java`).  Rust has no
//! inheritance, so - as in `etomo/storage/autodoc/statement.rs` - the class splits into
//! the `BaseManager` trait (every method, the abstract ones without a default body) and
//! `BaseManagerBase` (the fields the class declares), reached through `base()`.  The
//! Java constructor body is the trait's `base_manager` method, run by a subclass
//! constructor after the allocation exists, because it passes `this` to
//! `new ImodManager(this)`.
//!
//! **Lifetime.**  Java's managers are created by `EtomoDirector`, held in its manager
//! list and never collected while the program runs.  As in `etomo/ui/swing/token.rs`,
//! the translation models that by a leaked allocation, so `&'static dyn BaseManager` is
//! the reference type every unit that takes a `BaseManager` parameter uses, and
//! `Option<&'static dyn BaseManager>` carries Java's nullable one.  The trait is
//! `Send + Sync` because a manager is reachable from several threads in the source
//! (`EmergencyMonitor`, the process threads) and because Java's fields are mutated
//! through such shared references; each mutable field therefore carries its own lock,
//! the same modelling `etomo/storage/log_file.rs` uses for Java's instance monitors.
//!
//! **Frontier.**  This class names 69 Swing types and about thirty further untranslated
//! source units.  Every member that dereferences one carries a `// TODO(unit):` marker
//! naming the exact file; a field whose declared type has no module is
//! `Option<std::convert::Infallible>` - the Rust type with exactly the one inhabitant
//! Java's `null` has - as `etomo/process/emergency_monitor.rs` does, so no stub type is
//! invented and the branches that would dereference it are provably unreachable.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::manager_key::ManagerKey;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::process::imod_manager::ImodManager;
use crate::imod::etomo::process::tomosetexts_output::TomosetextsOutput;
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::util::unique_key::UniqueKey;
use crate::imod::etomo::util::utilities;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

/// Java private static `NO_PROCESS_THREAD_NAME`.
const NO_PROCESS_THREAD_NAME: &str = "none";

/// Java private static `headless`, a mutable class variable the constructor assigns.
static HEADLESS: Mutex<bool> = Mutex::new(false);

/// Java `private static final boolean DEBUG =
/// EtomoDirector.INSTANCE.getArguments().isDebug()`.  A static initialiser, so it is
/// read once, the first time the class is touched.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

/// Java public static final nested class `Task`, which implements `TaskInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// Java `RESUME`, constructed with descr "resume".
    Resume,
}

impl Task {
    /// Java field `descr`.
    fn descr(self) -> &'static str {
        match self {
            Task::Resume => "resume",
        }
    }
}

impl TaskInterface for Task {
    /// Java `getDescr`.
    fn get_descr(&self) -> Option<String> {
        Some(self.descr().to_string())
    }

    /// Java `okToDrop`.
    fn ok_to_drop(&self) -> bool {
        false
    }
}

/// Java private static final nested class `ResumeData`.
///
/// Four of its six fields have untranslated declared types, so they are
/// `Option<Infallible>`; `isNull` tests `param`, which is one of them, so it is always
/// true here and the source's resume path is never entered.
// TODO(unit): needs etomo/comscript/ProcesschunksParam.java,
// etomo/type/ProcessResultDisplay.java, etomo/ProcessSeries.java and
// etomo/type/ProcessingMethod.java - the declared types of `param`,
// `processResultDisplay`, `processSeries` and `processingMethod`.
pub struct ResumeData {
    /// Java field `param`, initialised to null.
    param: Mutex<Option<Infallible>>,
    /// Java field `processResultDisplay`, initialised to null.
    process_result_display: Mutex<Option<Infallible>>,
    /// Java field `processSeries`, initialised to null.
    process_series: Mutex<Option<Infallible>>,
    /// Java field `popupChunkWarnings`, initialised to false.
    popup_chunk_warnings: Mutex<bool>,
    /// Java field `processingMethod`, initialised to null.
    processing_method: Mutex<Option<Infallible>>,
    /// Java field `multiLineMessages`, initialised to false.
    multi_line_messages: Mutex<bool>,
}

impl ResumeData {
    /// The field initialisers of `new ResumeData()`.
    pub fn new() -> ResumeData {
        ResumeData {
            param: Mutex::new(None),
            process_result_display: Mutex::new(None),
            process_series: Mutex::new(None),
            popup_chunk_warnings: Mutex::new(false),
            processing_method: Mutex::new(None),
            multi_line_messages: Mutex::new(false),
        }
    }

    /// Java private synchronized `set`.
    fn set(
        &self,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        popup_chunk_warnings: bool,
        processing_method: Option<Infallible>,
        multi_line_messages: bool,
    ) {
        *self.param.lock().unwrap() = param;
        *self.process_result_display.lock().unwrap() = process_result_display;
        *self.process_series.lock().unwrap() = process_series;
        *self.popup_chunk_warnings.lock().unwrap() = popup_chunk_warnings;
        *self.processing_method.lock().unwrap() = processing_method;
        *self.multi_line_messages.lock().unwrap() = multi_line_messages;
    }

    /// Java private synchronized `isNull`.
    fn is_null(&self) -> bool {
        self.param.lock().unwrap().is_none()
    }

    /// Java private synchronized `reset`.
    fn reset(&self) {
        *self.param.lock().unwrap() = None;
        *self.process_result_display.lock().unwrap() = None;
        *self.process_series.lock().unwrap() = None;
        *self.popup_chunk_warnings.lock().unwrap() = false;
        *self.processing_method.lock().unwrap() = None;
        *self.multi_line_messages.lock().unwrap() = false;
    }

    /// Java private `getProcesschunksParam`.
    fn get_processchunks_param(&self) -> Option<Infallible> {
        *self.param.lock().unwrap()
    }

    /// Java private `getProcessResultDisplay`.
    fn get_process_result_display(&self) -> Option<Infallible> {
        *self.process_result_display.lock().unwrap()
    }

    /// Java private `getProcessSeries`.
    fn get_process_series(&self) -> Option<Infallible> {
        *self.process_series.lock().unwrap()
    }

    /// Java private `getProcessingMethod`.
    fn get_processing_method(&self) -> Option<Infallible> {
        *self.processing_method.lock().unwrap()
    }

    /// Java private `isMultiLineMessages`.
    fn is_multi_line_messages(&self) -> bool {
        *self.multi_line_messages.lock().unwrap()
    }

    /// Java private `isPopupChunkWarnings`.
    fn is_popup_chunk_warnings(&self) -> bool {
        *self.popup_chunk_warnings.lock().unwrap()
    }
}

impl Default for ResumeData {
    fn default() -> ResumeData {
        ResumeData::new()
    }
}

/// The fields Java's abstract `BaseManager` declares.  Every implementor embeds one and
/// returns it from `BaseManager::base`.
pub struct BaseManagerBase {
    /// Java field `busyStatusMediator`, `new BusyStatusMediator()`.
    // TODO(unit): needs etomo/logic/BusyStatusMediator.java - the field's declared type.
    busy_status_mediator: Option<Infallible>,
    /// Java field `uiHarness`, `UIHarness.INSTANCE`.
    // TODO(unit): needs etomo/ui/swing/UIHarness.java - the field's declared type.
    ui_harness: Option<Infallible>,
    /// Java field `loadedParamFile`, initialised to false.
    pub(crate) loaded_param_file: Mutex<bool>,
    /// Java field `imodManager`, `new ImodManager(this)`.  imodManager manages the
    /// opening and closing of imod(s), and message passing for loading a model.
    /// Java field `imodManager`, which the constructor builds.
    imod_manager: Mutex<Option<ImodManager>>,
    /// Java field `paramFile`, initialised to null.
    pub(crate) param_file: Mutex<Option<PathBuf>>,
    /// Java field `homeDirectory`.
    // FIXME homeDirectory may not have to be visible
    home_directory: Mutex<Option<String>>,
    /// Java field `threadNameA`, initialised to `NO_PROCESS_THREAD_NAME`.
    thread_name_a: Mutex<String>,
    /// Java field `threadNameB`, initialised to `NO_PROCESS_THREAD_NAME`.
    thread_name_b: Mutex<String>,
    /// Java field `backgroundProcessA`, initialised to false.
    background_process_a: Mutex<bool>,
    /// Java field `backgroundProcessNameA`, initialised to null.
    background_process_name_a: Mutex<Option<String>>,
    /// Java field `propertyUserDir`, the working directory for this manager.
    property_user_dir: Mutex<Option<String>>,
    /// Java field `debug`, initialised to false.
    debug: Mutex<bool>,
    /// Java field `exiting`, initialised to false.
    exiting: Mutex<bool>,
    /// Java field `initialized`, initialised to false.
    initialized: Mutex<bool>,
    /// Java field `currentDialogTypeA`, initialised to null.
    current_dialog_type_a: Mutex<Option<DialogType>>,
    /// Java field `currentDialogTypeB`, initialised to null.
    current_dialog_type_b: Mutex<Option<DialogType>>,
    /// Java field `parameterStore`, initialised to null.
    parameter_store: Mutex<Option<ParameterStore>>,
    /// Java field `reconnectRunA`.  True if `reconnect()` has been run for axis A.
    reconnect_run_a: Mutex<bool>,
    /// Java field `reconnectRunB`.  True if `reconnect()` has been run for axis B.
    reconnect_run_b: Mutex<bool>,
    /// Java field `processingMethodMediatorA`, `new ProcessingMethodMediator()`.
    // TODO(unit): needs etomo/ProcessingMethodMediator.java - the field's declared type.
    processing_method_mediator_a: Option<Infallible>,
    /// Java field `processingMethodMediatorB`, `new ProcessingMethodMediator()`.
    // TODO(unit): needs etomo/ProcessingMethodMediator.java - the field's declared type.
    processing_method_mediator_b: Option<Infallible>,
    /// Java field `managerKey`, `new ManagerKey()`.
    ///
    /// `ManagerKey` is mutable in Java, and a director and manager retain the
    /// same holder.  The `Arc` preserves that object identity while the mutex
    /// represents Java's shared mutable object across process/UI threads.
    manager_key: Arc<Mutex<ManagerKey>>,
    /// Java field `axisProcessData`, `new AxisProcessData(this)`.
    // TODO(unit): needs etomo/process/AxisProcessData.java - the field's declared type.
    axis_process_data: Option<Infallible>,
    /// Java field `resumeDataA`, `new ResumeData()`.
    resume_data_a: ResumeData,
    /// Java field `resumeDataB`, `new ResumeData()`.
    resume_data_b: ResumeData,
    /// Java field `logWindow`, `createLogWindow()`.
    // TODO(unit): needs etomo/ui/swing/LogWindow.java - the field's declared type.
    log_window: Option<Infallible>,
    /// Java field `validBrowsingDirectory`, initialised to null.
    // TODO(unit): needs etomo/util/ValidDirectory.java - the field's declared type.
    valid_browsing_directory: Mutex<Option<Infallible>>,
    /// Java field `emergencyMonitor`, initialised to null.
    emergency_monitor: Mutex<Option<Arc<EmergencyMonitor>>>,
}

impl BaseManagerBase {
    /// The field initialisers Java runs before the constructor body.
    pub fn initial() -> BaseManagerBase {
        BaseManagerBase {
            busy_status_mediator: None,
            ui_harness: None,
            loaded_param_file: Mutex::new(false),
            imod_manager: Mutex::new(None),
            param_file: Mutex::new(None),
            home_directory: Mutex::new(None),
            thread_name_a: Mutex::new(NO_PROCESS_THREAD_NAME.to_string()),
            thread_name_b: Mutex::new(NO_PROCESS_THREAD_NAME.to_string()),
            background_process_a: Mutex::new(false),
            background_process_name_a: Mutex::new(None),
            property_user_dir: Mutex::new(None),
            debug: Mutex::new(false),
            exiting: Mutex::new(false),
            initialized: Mutex::new(false),
            current_dialog_type_a: Mutex::new(None),
            current_dialog_type_b: Mutex::new(None),
            parameter_store: Mutex::new(None),
            reconnect_run_a: Mutex::new(false),
            reconnect_run_b: Mutex::new(false),
            processing_method_mediator_a: None,
            processing_method_mediator_b: None,
            manager_key: Arc::new(Mutex::new(ManagerKey::default())),
            axis_process_data: None,
            resume_data_a: ResumeData::new(),
            resume_data_b: ResumeData::new(),
            log_window: None,
            valid_browsing_directory: Mutex::new(None),
            emergency_monitor: Mutex::new(None),
        }
    }
}

impl Default for BaseManagerBase {
    fn default() -> BaseManagerBase {
        BaseManagerBase::initial()
    }
}

/// Java `BaseManager`.
pub trait BaseManager: Send + Sync {
    /// The fields Java's `BaseManager` declares.  Not a source member: it is how a Rust
    /// implementor exposes the superclass's field block, which Java reaches directly.
    fn base(&self) -> &BaseManagerBase;

    /// Java's `this` where the source passes the manager on.  Not a source member: a
    /// default trait body holds an unsized `&Self`, which Rust cannot coerce to
    /// `&dyn BaseManager`, so each implementor returns itself.
    fn this(&'static self) -> &'static dyn BaseManager;

    /// Java `BaseManager()`.  The constructor body, run by a subclass constructor once
    /// the allocation exists: `new ImodManager(this)` needs `this`.
    fn base_manager(&'static self) {
        *self.base().property_user_dir.lock().unwrap() = std::env::var("PWD").ok();
        self.create_process_track();
        self.create_com_script_manager();
        // Initialize the program settings
        *self.base().debug.lock().unwrap() = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
        *HEADLESS.lock().unwrap() = etomo_director::ARGUMENTS.lock().unwrap().is_headless();
        self.create_main_panel();
        *self.base().imod_manager.lock().unwrap() = Some(ImodManager::new(Some(self.this())));
        self.init_program();
    }

    /// Java `dumpState`.
    fn dump_state(&self) {
        if !*DEBUG {
            return;
        }
        eprintln!(
            "[headless:{},loadedParamFile:{},paramFile:",
            *HEADLESS.lock().unwrap(),
            *self.base().loaded_param_file.lock().unwrap()
        );
        if let Some(param_file) = self.base().param_file.lock().unwrap().as_ref() {
            eprintln!("{},", param_file.display());
        }
        eprintln!(
            "homeDirectory:{},threadNameA:{},\nthreadNameB:{},backgroundProcessA:{},\nbackgroundProcessNameA:{},propertyUserDir:{},\ndebug:{},exiting:{},initialized:{},\ncurrentDialogTypeA:{},\ncurrentDialogTypeB:{},reconnectRunA:{},\nreconnectRunA:{},reconnectRunB:{},\nmanagerKey:{}]",
            self.base()
                .home_directory
                .lock()
                .unwrap()
                .clone()
                .unwrap_or("null".to_string()),
            self.base().thread_name_a.lock().unwrap(),
            self.base().thread_name_b.lock().unwrap(),
            *self.base().background_process_a.lock().unwrap(),
            self.base()
                .background_process_name_a
                .lock()
                .unwrap()
                .clone()
                .unwrap_or("null".to_string()),
            self.base()
                .property_user_dir
                .lock()
                .unwrap()
                .clone()
                .unwrap_or("null".to_string()),
            *self.base().debug.lock().unwrap(),
            *self.base().exiting.lock().unwrap(),
            *self.base().initialized.lock().unwrap(),
            match *self.base().current_dialog_type_a.lock().unwrap() {
                None => "null".to_string(),
                Some(dialog_type) => dialog_type.to_string(),
            },
            match *self.base().current_dialog_type_b.lock().unwrap() {
                None => "null".to_string(),
                Some(dialog_type) => dialog_type.to_string(),
            },
            *self.base().reconnect_run_a.lock().unwrap(),
            *self.base().reconnect_run_a.lock().unwrap(),
            *self.base().reconnect_run_b.lock().unwrap(),
            // TODO(unit): needs etomo/ManagerKey.java - `managerKey` prints through its
            // own `toString`; the field is null here.
            "null"
        );
    }

    /// Java abstract `getInterfaceType`.
    fn get_interface_type(&self) -> Option<InterfaceType>;

    /// Java abstract package-private `createMainPanel`.
    // TODO(unit): needs etomo/ui/swing/MainPanel.java - every implementation of this
    // abstract method builds the manager's Swing main panel.
    fn create_main_panel(&self);

    /// Java abstract `getBaseMetaData`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData>;

    /// Java abstract `getMainPanel`.
    // TODO(unit): needs etomo/ui/swing/MainPanel.java - the abstract method's return
    // type, so an implementor can only return null here.
    fn get_main_panel(&self) -> Option<Infallible>;

    /// Java abstract `getProcessManager`.
    // TODO(unit): needs etomo/process/BaseProcessManager.java - the abstract method's
    // return type, so an implementor can only return null here.
    fn get_process_manager(&self) -> Option<Infallible>;

    /// Java abstract package-private `getStorables(int)`.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Box<dyn Storable>>>;

    /// Java abstract `getName`.
    fn get_name(&self) -> Option<String>;

    /// Java `allowProcessWatching`.
    fn allow_process_watching(&self) -> bool {
        true
    }

    /// Java `getEmergencyMonitor`.
    fn get_emergency_monitor(&'static self, axis_id: Option<AxisID>) -> Arc<EmergencyMonitor> {
        // Override for dual axis.
        let axis_id = Some(AxisID::Only);
        let _ = axis_id;
        let mut emergency_monitor = self.base().emergency_monitor.lock().unwrap();
        // The source's double-checked `if (emergencyMonitor == null) synchronized (this)
        // { if (emergencyMonitor == null) ... }`; the lock above is the instance monitor,
        // so the two tests collapse into the one the lock already serialises.
        if emergency_monitor.is_none() {
            *emergency_monitor = Some(Arc::new(EmergencyMonitor::new(
                Some(self.this()),
                Some(AxisID::Only),
            )));
        }
        emergency_monitor.clone().unwrap()
    }

    /// Java `tomosetexts`.
    // Bug# 2403
    fn tomosetexts(&self) -> Option<TomosetextsOutput> {
        let directory = self.base().property_user_dir.lock().unwrap().clone()?;
        let bin_path = get_imod_bin_path()?;
        BaseProcessManager::tomosetexts_local(
            Path::new(&directory),
            std::ffi::OsStr::new("python"),
            &Path::new(&bin_path).join("b3dtomosetexts"),
        )
    }

    /// Java `getVerticalScrollBarValue`.
    fn get_vertical_scroll_bar_value(&self, axis_id: Option<AxisID>) -> Option<i32> {
        let main_panel = self.get_main_panel();
        if main_panel.is_none() {
            return None;
        }
        // TODO(unit): needs etomo/ui/swing/MainPanel.java -
        // `mainPanel.getVerticalScrollBarValue(axisID)`.  `getMainPanel` can only return
        // null, so this line is unreachable.
        let _ = axis_id;
        None
    }

    /// Java `setVerticalScrollBarValue`.
    fn set_vertical_scroll_bar_value(&self, axis_id: Option<AxisID>, value: Option<i32>) {
        let main_panel = self.get_main_panel();
        if main_panel.is_none() {
            return;
        }
        // TODO(unit): needs etomo/ui/swing/MainPanel.java -
        // `mainPanel.setVerticalScrollBarValue(axisID, value)`.
        let _ = (axis_id, value);
    }

    /// Java `getPhysicalCores`.  Run `imodqtassist -t` and return the "physical cores"
    /// value.
    fn get_physical_cores(&self, axis_id: Option<AxisID>) -> Option<i32> {
        let process_manager = self.get_process_manager();
        if process_manager.is_none() {
            return None;
        }
        // TODO(unit): needs etomo/process/BaseProcessManager.java - the rest of the
        // method parses `processManager.imodqtassistQuery(axisID)`.  `getProcessManager`
        // can only return null, so the parse is unreachable.
        let _ = axis_id;
        None
    }

    /// Java `isBeadfixerDiameterAvailable`.
    fn is_beadfixer_diameter_available(&self) -> bool {
        false
    }

    /// Java `getBeadfixerDiameter`.
    fn get_beadfixer_diameter(&self, axis_id: Option<AxisID>) -> Option<i32> {
        let _ = axis_id;
        None
    }

    /// Java `isAddGPUMachineToProcessChunks`.
    fn is_add_gpu_machine_to_process_chunks(&self) -> bool {
        false
    }

    /// Java `addBusyStatusListener`.
    fn add_busy_status_listener(&self, listener: Option<Infallible>) {
        // TODO(unit): needs etomo/logic/BusyStatusMediator.java and
        // etomo/logic/BusyStatusListener.java -
        // `busyStatusMediator.addBusyStatusListener(listener)`.
        let _ = listener;
    }

    /// Java `removeBusyStatusListener`.
    fn remove_busy_status_listener(&self, listener: Option<Infallible>) {
        // TODO(unit): needs etomo/logic/BusyStatusMediator.java and
        // etomo/logic/BusyStatusListener.java -
        // `busyStatusMediator.removeBusyStatusListener(listener)`.
        let _ = listener;
    }

    /// Java `getBusyStatusMediator`.
    fn get_busy_status_mediator(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/logic/BusyStatusMediator.java - the return type.
        self.base().busy_status_mediator
    }

    /// Java `isDualSelectionQueueTable`.
    fn is_dual_selection_queue_table(&self) -> bool {
        false
    }

    /// Java `getBrowsingDir`, the first half of this class's
    /// `etomo/ui/BrowsingDirectory.java` implementation.
    ///
    /// Deviation: Java declares `implements BrowsingDirectory` and supplies the bodies
    /// here.  Rust cannot give a supertrait's method a default body from the subtrait,
    /// so the two methods sit on this trait; `etomo/ui/browsing_directory.rs` is the
    /// interface itself, which an implementor also implements by delegating to these.
    fn get_browsing_dir(&self) -> Option<PathBuf> {
        let valid_browsing_directory = self.base().valid_browsing_directory.lock().unwrap();
        if valid_browsing_directory.is_none() {
            // TODO(unit): needs etomo/util/ValidDirectory.java -
            // `validBrowsingDirectory = new ValidDirectory(this)` followed by
            // `validBrowsingDirectory.setToPropertyUserDir()`.
        }
        // TODO(unit): needs etomo/util/ValidDirectory.java - `validBrowsingDirectory.get()`.
        None
    }

    /// Java `setBrowsingDir(File)`, the second half of this class's
    /// `etomo/ui/BrowsingDirectory.java` implementation.  Set valid browsing directory.
    fn set_browsing_dir(&self, input: Option<&Path>) {
        let valid_browsing_directory = self.base().valid_browsing_directory.lock().unwrap();
        if valid_browsing_directory.is_none() && input.is_some() {
            // TODO(unit): needs etomo/util/ValidDirectory.java -
            // `validBrowsingDirectory = new ValidDirectory(this)`.
        }
        if valid_browsing_directory.is_some() {
            // TODO(unit): needs etomo/util/ValidDirectory.java -
            // `validBrowsingDirectory.set(input)`.
        }
    }

    /// Java package-private `setBrowsingDir(String)`.
    fn set_browsing_dir_string(&self, input: Option<&str>) {
        let mut valid_browsing_directory = self.base().valid_browsing_directory.lock().unwrap();
        if valid_browsing_directory.is_none()
            && input.is_some()
            && !utilities::EMPTY_PATTERN.is_match(input.unwrap())
            && !input.unwrap().is_empty()
        {
            // TODO(unit): needs etomo/util/ValidDirectory.java -
            // `validBrowsingDirectory = new ValidDirectory(this)`.
        }
        if valid_browsing_directory.is_some() {
            // TODO(unit): needs etomo/util/ValidDirectory.java -
            // `validBrowsingDirectory.set(input)`.
        }
        let _ = &mut valid_browsing_directory;
    }

    /// Java `getFileSubdirectoryName`.  Return the subdirectory of the dataset location
    /// where some of the files are stored.  This is necessary for the NAD manager.
    /// Return null if there is no subdirectory.
    fn get_file_subdirectory_name(&self) -> Option<String> {
        None
    }

    /// Java `getParallelProcessingDefaultNice`.
    fn get_parallel_processing_default_nice(&self) -> i32 {
        15
    }

    /// The `BaseManager` body of Java `paramString`, reached from a subclass override
    /// the way `super.paramString(...)` is.  Rust cannot call a trait method's default
    /// body from an implementation that overrides it, so the body lives here and
    /// the overridable method delegates to it; `etomo/join_manager.rs` is the
    /// caller that needs this.
    /// Java package-private `paramString`.
    fn param_string_super(&self) -> Option<String> {
        self.get_name()
    }

    /// Java `paramString`.
    fn param_string(&self) -> Option<String> {
        self.param_string_super()
    }

    /// Java `getAxisProcessData`.
    fn get_axis_process_data(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/process/AxisProcessData.java - the return type.
        self.base().axis_process_data
    }

    /// Java package-private `createLogWindow`.
    fn create_log_window(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/ui/swing/LogWindow.java - `LogWindow.getInstance(this)`.
        None
    }

    /// Java `showHideLog`.
    fn show_hide_log(&self) {
        if self.base().log_window.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogWindow.java - `logWindow.showHide()`.
        }
    }

    /// Java `getLogInterface`.
    fn get_log_interface(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/ui/swing/LogInterface.java - the return type;
        // `logWindow` is the value returned.
        self.base().log_window
    }

    /// Java `getLogProperties`.
    fn get_log_properties(&self) -> Option<&'static dyn LogProperties> {
        // TODO(unit): needs etomo/ui/swing/LogWindow.java - the source returns
        // `logWindow`, which implements `etomo/ui/LogProperties.java`; the field is null
        // here.
        None
    }

    /// Java private `initProgram`.
    fn init_program(&self) {
        if *DEBUG {
            eprintln!(
                "propertyUserDir:  {}",
                self.base()
                    .property_user_dir
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or("null".to_string())
            );
        }
    }

    /// Java `getPropertyUserDir`.
    fn get_property_user_dir(&self) -> Option<String> {
        self.base().property_user_dir.lock().unwrap().clone()
    }

    /// Java `canChangeParamFileName`.
    fn can_change_param_file_name(&self) -> bool {
        false
    }

    /// Java `canSaveDirectives`.
    fn can_save_directives(&self) -> bool {
        false
    }

    /// Java package-private `createComScriptManager`.  Empty in the base class.
    fn create_com_script_manager(&self) {}

    /// Java package-private `createProcessTrack`.  Empty in the base class.
    fn create_process_track(&self) {}

    /// Java `getViewType`.
    fn get_view_type(&self) -> ViewType {
        ViewType::DEFAULT
    }

    /// Java `getBaseScreenState`.
    fn get_base_screen_state(&self, axis_id: Option<AxisID>) -> Option<Infallible> {
        // TODO(unit): needs etomo/type/BaseScreenState.java - the return type.  The base
        // class returns null.
        let _ = axis_id;
        None
    }

    /// Java `getBaseState`.
    fn get_base_state(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/type/BaseState.java - the return type.  The base class
        // returns null.
        None
    }

    /// Java `getProcessResultDisplayFactoryInterface`.
    fn get_process_result_display_factory_interface(
        &self,
        axis_id: Option<AxisID>,
    ) -> Option<Infallible> {
        // TODO(unit): needs etomo/process/ProcessResultDisplayFactoryInterface.java - the
        // return type.  The base class returns null.
        let _ = axis_id;
        None
    }

    /// Java package-private `getProcessTrack()`.
    fn get_process_track(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/type/BaseProcessTrack.java - the return type.  The base
        // class returns null.
        None
    }

    /// Java package-private `getProcessTrack(Storable[], int)`.  Empty in the base class.
    fn get_process_track_into(&self, storable: Option<&mut [Box<dyn Storable>]>, index: i32) {
        let _ = (storable, index);
    }

    /// Java `isInManagerFrame`.
    fn is_in_manager_frame(&self) -> bool {
        false
    }

    /// Java package-private `getAutoAlignmentMetaData`.
    fn get_auto_alignment_meta_data(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/type/AutoAlignmentMetaData.java - the return type.  The
        // base class returns null.
        None
    }

    /// Java `getStatus`.
    fn get_status(&self) -> Option<String> {
        // TODO(unit): needs etomo/ui/swing/MainPanel.java - `getMainPanel().getStatus()`.
        None
    }

    /// Java `updateMetaData`.
    fn update_meta_data(
        &self,
        dialog_type: Option<DialogType>,
        axis_id: Option<AxisID>,
        do_validation: bool,
    ) -> bool {
        let _ = (dialog_type, axis_id, do_validation);
        false
    }

    /// Java `kill`.  Interrupt the currently running thread for this axis.
    fn kill(&self, axis_id: Option<AxisID>) {
        // TODO(unit): needs etomo/process/BaseProcessManager.java -
        // `getProcessManager().kill(axisID)`.
        let _ = axis_id;
    }

    /// Java `pause`.
    fn pause(&self, axis_id: Option<AxisID>) -> bool {
        // TODO(unit): needs etomo/process/BaseProcessManager.java -
        // `getProcessManager().pause(axisID)`.
        let _ = axis_id;
        false
    }

    /// Java package-private `processSeriesSucceeded`.  Empty in the base class.
    fn process_series_succeeded(&self, axis_id: Option<AxisID>, process_name: Option<Infallible>) {
        let _ = (axis_id, process_name);
    }

    /// Java `setParamFile()`.  In most managers the param file should already be set.
    fn set_param_file(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    /// Java `isSetupDone`.
    fn is_setup_done(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    /// Java `isLoadedParamFile`.
    fn is_loaded_param_file(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    /// Java `setParamFile(File)`.
    fn set_param_file_from(&self, param_file: Option<&Path>) -> bool {
        *self.base().param_file.lock().unwrap() = param_file.map(|path| path.to_path_buf());
        true
    }

    /// Java package-private `startNextProcess`.  Returns true if a process was started.
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
        // TODO(unit): needs etomo/ProcessSeries.java, etomo/ui/UIComponent.java,
        // etomo/type/ProcessResultDisplay.java, etomo/type/DialogType.java,
        // etomo/ui/swing/ProcessDisplay.java and etomo/comscript/TomodataplotsParam.java -
        // the parameter types, `process.equals(Task.RESUME)`, `process.getTask()` and the
        // `TomodataplotsParam.Task` dispatch.
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

    // Updates done

    /// Java package-private `updateDialog`.  Empty in the base class.
    fn update_dialog(&self, process_name: Option<Infallible>, axis_id: Option<AxisID>) {
        let _ = (process_name, axis_id);
    }

    /// Java `logMessagePrimaryLog`.
    fn log_message_primary_log(&self, reader: Option<Infallible>) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java and
            // etomo/storage/FileReader.java -
            // `logInterface.logMessagePrimaryLog(reader)`.
        }
        let _ = reader;
    }

    /// Java `logMessage(File)`.
    fn log_message_file(&'static self, file: Option<&Path>) {
        self.log_message_private(file, false, None);
    }

    /// Java `logSimpleMessage(File, FileWriter)`.
    fn log_simple_message_file(
        &'static self,
        file: Option<&Path>,
        secondary_log: Option<Infallible>,
    ) {
        self.log_message_private(file, true, secondary_log);
    }

    /// Java private `logMessage(File, boolean, FileWriter)`.
    fn log_message_private(
        &'static self,
        file: Option<&Path>,
        simple: bool,
        secondary_log: Option<Infallible>,
    ) {
        let file = match file {
            None => return,
            Some(file) if !file.exists() || file.is_dir() => return,
            Some(file) => file,
        };
        // `File.canRead()`; a path this process cannot open for reading is skipped.
        if std::fs::File::open(file).is_err() {
            return;
        }
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java and
            // etomo/storage/FileWriter.java - `logInterface.logMessage(file,
            // secondaryLog)` and `logInterface.logMessage(file, false, secondaryLog)`.
            let _ = (simple, secondary_log);
        } else {
            if *DEBUG {
                eprintln!("Logging from file: {}", file.display());
            }
            match crate::imod::etomo::storage::log_file::LogFile::get_instance_file(
                Some(file),
                Some(self.get_emergency_monitor(None)),
            ) {
                Err(e) => {
                    // `catch (final LogFileException | IOException e)`
                    eprintln!("{}", e);
                    if *DEBUG {
                        eprintln!("Unable to log from file.  {}", e);
                    }
                }
                Ok(log_file) => {
                    match log_file.open_reader() {
                        Err(_) => {
                            // `catch (final LockException e) {}`
                        }
                        Ok(None) => {}
                        Ok(Some(id)) => {
                            if *DEBUG {
                                while let Ok(Some(line)) = log_file.read_line(&id) {
                                    eprintln!("{}", line);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Java `logSimpleMessage(String, FileWriter)`.  Log without extra stuff.
    fn log_simple_message(&self, message: Option<&str>, secondary_log: Option<Infallible>) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.logMessage(message, false, false, secondaryLog)`.
            let _ = secondary_log;
        } else if *DEBUG {
            eprintln!("{}", message.unwrap_or("null"));
        }
    }

    /// Java `logSimpleMessage(String, boolean)`.
    fn log_simple_message_newline(&self, message: Option<&str>, newline: bool) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.logMessage(message, false, newline, null)`.
            let _ = newline;
        } else if *DEBUG {
            eprintln!("{}", message.unwrap_or("null"));
        }
    }

    /// Java `logMessage(String)`.
    fn log_message(&self, message: Option<&str>) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.logMessage(message)`.
        } else if *DEBUG {
            eprintln!("{}", message.unwrap_or("null"));
        }
    }

    /// Java `logMessage(Loggable, AxisID)`.
    fn log_message_loggable(&self, loggable: Option<Infallible>, axis_id: Option<AxisID>) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.logMessage(loggable, axisID)`.
        } else {
            // TODO(unit): needs etomo/storage/Loggable.java - the `else` branch prints
            // `loggable.getLogMessage()`.  `loggable` is `Option<Infallible>`, so the
            // branch cannot run.
            let _ = (loggable, axis_id);
        }
    }

    /// Java `getProcessingMethodMediator`.
    fn get_processing_method_mediator(&self, axis_id: Option<AxisID>) -> Option<Infallible> {
        // TODO(unit): needs etomo/ProcessingMethodMediator.java - the return type.
        if axis_id == Some(AxisID::Second) {
            return self.base().processing_method_mediator_b;
        }
        self.base().processing_method_mediator_a
    }

    /// Java `logMessage(String[], String, String, AxisID)`.
    fn log_message_array(
        &self,
        message: Option<&[String]>,
        title: Option<&str>,
        msg_id: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.logMessage(title, axisID, message, msgId)`.
            return false;
        }
        let mut retval = false;
        if *DEBUG {
            eprintln!(
                "{}\n{} - {} axis:",
                utilities::get_date_time_stamp(),
                title.unwrap_or("null"),
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                }
            );
            let message = match message {
                None => return retval,
                Some(message) => message,
            };
            for item in message {
                if !retval && (msg_id.is_none() || item.find(msg_id.unwrap()).is_some()) {
                    retval = true;
                }
                eprintln!("{}", item);
            }
        }
        retval
    }

    /// Java `logMessageUntilWithKeyword`.  Returns true when `msgId` was found.
    fn log_message_until_with_keyword(
        &self,
        file_type: Option<Infallible>,
        skip_tag: Option<&str>,
        tag: Option<&str>,
        until_tag: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        // TODO(unit): needs etomo/ui/swing/LogInterface.java - the method reads the log
        // file into an array and hands it to `logInterface.logMessage(axisID,
        // messageArray)`.  Its `fileType` parameter is `etomo/type/FileType.java`, which
        // is translated, but the whole body exists to feed the Swing log window and the
        // `else` arm is `System.err.println(line)` under the debug flag only.
        let _ = (file_type, skip_tag, tag, until_tag, axis_id);
        false
    }

    /// Java `logMessageWithKeyword(String[], String, String, AxisID)`.
    fn log_message_with_keyword(
        &self,
        message: Option<&[String]>,
        keyword: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        let log_interface = self.get_log_interface();
        let message = match message {
            None => return,
            Some(message) => message,
        };
        let mut logged_title = false;
        for item in message {
            if keyword.is_some() && item.find(keyword.unwrap()).is_some() {
                if !logged_title {
                    logged_title = true;
                    if log_interface.is_some() {
                        // TODO(unit): needs etomo/ui/swing/LogInterface.java -
                        // `logInterface.logMessage(title, axisID)`.
                    } else if *DEBUG {
                        eprintln!(
                            "{}\n{} - {} axis:",
                            utilities::get_date_time_stamp(),
                            title.unwrap_or("null"),
                            match axis_id {
                                None => "null".to_string(),
                                Some(axis_id) => axis_id.to_string(),
                            }
                        );
                    }
                }
                if log_interface.is_some() {
                    // TODO(unit): needs etomo/ui/swing/LogInterface.java -
                    // `logInterface.logMessage(message[i])`.
                } else if *DEBUG {
                    eprintln!("{}", item);
                }
            }
        }
    }

    /// Java `logMessageWithKeyword(FileType, String, String, AxisID)`.
    fn log_message_with_keyword_file_type(
        &self,
        file_type: Option<Infallible>,
        keyword: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        // TODO(unit): needs etomo/ui/swing/LogInterface.java - as with
        // `logMessageUntilWithKeyword`, the body reads the log file only to feed the
        // Swing log window; its `else` arms are debug-only prints.
        let _ = (file_type, keyword, title, axis_id);
        false
    }

    /// Java `updateDirectiveMap`.  Empty in the base class.
    fn update_directive_map(&self, directive_map: Option<Infallible>, errmsg: &mut String) {
        // TODO(unit): needs etomo/type/DirectiveMapInterface.java - the `directiveMap`
        // parameter's declared type.  The base class body is empty.
        let _ = (directive_map, errmsg);
    }

    /// Java `isAllowPrimaryLogging`.
    fn is_allow_primary_logging(&self) -> bool {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.isAllowPrimaryLogging()`.
        }
        true
    }

    /// Java `setAllowPrimaryLogging`.
    fn set_allow_primary_logging(&self, input: bool) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.setAllowPrimaryLogging(input)`.
        }
        let _ = input;
    }

    /// Java `logMessage(ArrayList<String>, String, AxisID)`.
    fn log_message_list(
        &self,
        message: Option<&Vec<String>>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java -
            // `logInterface.logMessage(title, axisID, message)`.
        } else if *DEBUG {
            eprintln!(
                "{}\n{} - {} axis:",
                utilities::get_date_time_stamp(),
                title.unwrap_or("null"),
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                }
            );
            if let Some(message) = message {
                for item in message {
                    eprintln!("{}", item);
                }
            }
        }
    }

    /// Java `saveLog`.
    fn save_log(&self) {
        let log_interface = self.get_log_interface();
        if log_interface.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogInterface.java - `logInterface.save()`.
        }
    }

    /// Java package-private `setManagerKey`.
    fn set_manager_key(&self, unique_key: Option<UniqueKey>) {
        self.base().manager_key.lock().unwrap().set_key(unique_key);
    }

    /// Java package-private `getManagerKey`.
    fn get_manager_key(&self) -> Arc<Mutex<ManagerKey>> {
        Arc::clone(&self.base().manager_key)
    }

    /// Java `getFocusComponent`.
    fn get_focus_component(&self) -> Option<Infallible> {
        // TODO(unit): needs java.awt.Component - the return type of the Swing boundary.
        // The base class returns null.
        None
    }

    /// Java `setPropertyUserDir`.
    fn set_property_user_dir(&self, property_user_dir: Option<&str>) -> Option<String> {
        // avoid empty strings
        let property_user_dir = match property_user_dir {
            Some(dir) if utilities::EMPTY_PATTERN.is_match(dir) || dir.is_empty() => None,
            other => other,
        };
        utilities::manager_stamp(property_user_dir, None);
        let mut field = self.base().property_user_dir.lock().unwrap();
        let old_property_user_dir = field.clone();
        *field = property_user_dir.map(|dir| dir.to_string());
        old_property_user_dir
    }

    /// Java `pack`.  Empty in the base class.
    fn pack(&self) {}

    /// Java package-private `initializeUIParameters(File, AxisID, boolean)`.
    fn initialize_ui_parameters(
        &self,
        data_file: Option<&Path>,
        axis_id: Option<AxisID>,
        loaded_from_a_different_file: bool,
    ) {
        if !*HEADLESS.lock().unwrap() {
            if let Some(data_file) = data_file {
                *self.base().loaded_param_file.lock().unwrap() =
                    self.load_param_file(Some(data_file), axis_id, loaded_from_a_different_file);
            }
        }
        *self.base().initialized.lock().unwrap() = true;
    }

    /// Java package-private `initializeUIParameters(String, AxisID)`.
    fn initialize_ui_parameters_from_name(
        &self,
        param_file_name: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        match param_file_name {
            None => self.initialize_ui_parameters(None, axis_id, false),
            Some(param_file_name) if param_file_name.is_empty() => {
                self.initialize_ui_parameters(None, axis_id, false)
            }
            Some(param_file_name) => {
                self.initialize_ui_parameters(Some(Path::new(param_file_name)), axis_id, false)
            }
        }
    }

    /// Java `saveStorable`.  Save storable to the data file.
    fn save_storable(&self, axis_id: Option<AxisID>, storable: Option<&dyn Storable>) {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java for the error dialog; the
        // `parameterStore.save(storable)` half needs
        // `etomo/storage/ParameterStore.java`'s manager-aware `getInstance`, which is
        // itself blocked, so `getParameterStore` cannot produce a store here.
        let _ = (axis_id, storable);
    }

    /// Java package-private `saveMetaDataToParameterStore`.
    fn save_meta_data_to_parameter_store(&self, axis_id: Option<AxisID>) -> bool {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java for the error dialog; see
        // `saveStorable` for the `ParameterStore` half.
        let _ = axis_id;
        true
    }

    /// Java `saveStorables`.  Save etomo to `parameterStore` by asking the child manager
    /// for a list of storable objects.
    fn save_storables(&self, axis_id: Option<AxisID>) {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java for the error dialogs and
        // etomo/process/BaseProcessManager.java for `getStorables`, whose two slots come
        // from `getProcessManager().getProcessData(..)`.
        let _ = axis_id;
    }

    /// Java `isNewDataset`.
    fn is_new_dataset(&self) -> bool {
        self.base().param_file.lock().unwrap().is_none()
    }

    /// Java private `getStorables()`.  Get the storable objects from the child and base
    /// manager.
    fn get_storables(&self) -> Option<Vec<Box<dyn Storable>>> {
        let storables = self.get_storables_with_offset(2);
        if storables.is_none() {
            // Manager does not have a data file.
            return None;
        }
        // TODO(unit): needs etomo/process/BaseProcessManager.java - `storables[0]` and
        // `storables[1]` are `getProcessManager().getProcessData(AxisID.FIRST/SECOND)`.
        storables
    }

    /// The `BaseManager` body of Java `save`, reached from a subclass override
    /// the way `super.save(...)` is.  Rust cannot call a trait method's default
    /// body from an implementation that overrides it, so the body lives here and
    /// the overridable method delegates to it; `etomo/join_manager.rs` is the
    /// caller that needs this.
    /// Java package-private `save`.  Save etomo to `parameterStore` by asking the child
    /// manager to save its state.
    fn save_super(&self) -> bool {
        // TODO(unit): needs etomo/process/BaseProcessManager.java -
        // `parameterStore.save(getProcessManager().getProcessData(..))` for both axes.
        if self.base().parameter_store.lock().unwrap().is_none() {
            return false;
        }
        false
    }

    /// Java `save`.
    fn save(&self) -> bool {
        self.save_super()
    }

    /// Java `saveToFile`.
    fn save_to_file(&self) -> bool {
        false
    }

    /// Java `saveAsToFile`.
    fn save_as_to_file(&self) -> bool {
        false
    }

    /// Java `closeFrame`.
    fn close_frame(&self) -> bool {
        false
    }

    /// The `BaseManager` body of Java `saveParamFile`, reached from a subclass override
    /// the way `super.saveParamFile(...)` is.  Rust cannot call a trait method's default
    /// body from an implementation that overrides it, so the body lives here and
    /// the overridable method delegates to it; `etomo/join_manager.rs` is the
    /// caller that needs this.
    /// Java `saveParamFile`.  A message asking the `ApplicationManager` to save the
    /// parameter information to a file.
    fn save_param_file_super(&self) -> bool {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java (`setMRUFileLabels`),
        // etomo/type/UserConfiguration.java (`userConfig.putDataFile`) and
        // etomo/type/BaseProcessTrack.java (`processTrack.resetModified()`); the
        // `ParameterStore` half is blocked as in `saveStorable`.
        if !self.is_setup_done() {
            return false;
        }
        self.set_param_file();
        false
    }

    /// Java `saveParamFile`.
    fn save_param_file(&self) -> bool {
        self.save_param_file_super()
    }

    /// Java `getParameterStore`.  Creates `parameterStore` if it doesn't already exist.
    /// Returns null if `paramFile` is null.
    fn get_parameter_store(&self, axis_id: Option<AxisID>) -> Option<Infallible> {
        // TODO(unit): needs the manager-aware `ParameterStore.getInstance(BaseManager,
        // AxisID, File)` in etomo/storage/ParameterStore.java, whose translated module
        // carries only the fileless and file forms.  `paramFile` is null until
        // `loadParamFile` runs, which is itself blocked.
        let _ = axis_id;
        None
    }

    /// Java package-private `endThreads`.
    fn end_threads(&self) {
        // TODO(unit): needs etomo/process/BaseImodManager.java
        // (`imodManager.stopRequestHandler()`) and etomo/ProcessingMethodMediator.java
        // (`mediator.msgExiting()`).
        if let Some(parameter_store) = self.base().parameter_store.lock().unwrap().as_mut() {
            parameter_store.set_auto_store(false);
        }
    }

    /// Java `progressBarDone`.
    fn progress_bar_done(&self, axis_id: Option<AxisID>, process_end_state: Option<Infallible>) {
        // TODO(unit): needs etomo/ui/swing/MainPanel.java and
        // etomo/type/ProcessEndState.java - `getMainPanel().stopProgressBar(axisID,
        // processEndState)`.
        let _ = (axis_id, process_end_state);
    }

    /// Java private `checkNextProcess`.
    fn check_next_process(&self, axis_id: Option<AxisID>) -> bool {
        // TODO(unit): needs etomo/process/BaseProcessManager.java,
        // etomo/process/AxisProcessData.java, etomo/type/ConstProcessSeries.java and
        // etomo/ui/swing/UIHarness.java - the whole body runs only when
        // `getProcessManager()` is non-null, which it cannot be here.
        let _ = axis_id;
        true
    }

    /// Java `renameImageFile(FileType, FileType, AxisID)`.  Renames an image file.
    fn rename_image_file(
        &self,
        from_file_type: Option<Infallible>,
        to_file_type: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java and etomo/ui/swing/MainPanel.java
        // - the error dialog and the three `setProgressBarValue` calls.  The
        // `FileType.getFile(this, axisID)` half is translated but the method cannot run
        // without the progress bar.
        let _ = (from_file_type, to_file_type, axis_id);
    }

    /// Java package-private `renameImageFile(FileKey, File, FileType, AxisID, boolean)`.
    fn rename_image_file_from_key(
        &self,
        from_file_key: Option<Infallible>,
        from_file: Option<&Path>,
        to_file_type: Option<Infallible>,
        axis_id: Option<AxisID>,
        use_file_name_in_close: bool,
    ) {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java - the error dialog - and
        // etomo/process/BaseImodManager.java through `closeImod`.
        let _ = (
            from_file_key,
            from_file,
            to_file_type,
            axis_id,
            use_file_name_in_close,
        );
    }

    /// Java `backupImageFile`.  Renames an image file to `image_file_name~`.
    fn backup_image_file(&self, file_type: Option<Infallible>, axis_id: Option<AxisID>) {
        // TODO(unit): needs etomo/process/BaseImodManager.java through `closeImod`.  The
        // `Utilities.backupFile` half is translated.
        let _ = (file_type, axis_id);
    }

    /// Java `closeStaleFile(FileKey, AxisID)`.  Asks to close a stale file.
    fn close_stale_file(&self, file_key: Option<Infallible>, axis_id: Option<AxisID>) {
        // TODO(unit): needs etomo/process/BaseImodManager.java through `closeImod`.
        let _ = (file_key, axis_id);
    }

    /// Java `closeStaleFile(FileType, AxisID)`.  Deprecated 6/18/19.
    fn close_stale_file_from_file_type(
        &self,
        file_type: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) {
        // TODO(unit): needs etomo/process/BaseImodManager.java through `closeImod`.
        let _ = (file_type, axis_id);
    }

    /// Java `closeImod(FileKey, AxisID, boolean)`.  Ask to close all 3dmods associated
    /// with this file type.
    fn close_imod_file_key(
        &self,
        file_key: Option<Infallible>,
        axis_id: Option<AxisID>,
        warn_once: bool,
    ) {
        // TODO(unit): needs etomo/process/BaseImodManager.java - every `closeImod` overload
        // dereferences `imodManager`.
        let _ = (file_key, axis_id, warn_once);
    }

    /// Java private `closeImod(FileKey, File, AxisID, boolean)`.
    fn close_imod_file_key_and_file(
        &self,
        file_key: Option<Infallible>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        warn_once: bool,
    ) {
        // TODO(unit): needs etomo/process/BaseImodManager.java.
        let _ = (file_key, file, axis_id, warn_once);
    }

    /// Java `closeImod(String, File, AxisID, String, boolean)`.
    fn close_imod_key_file(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        warn_once: bool,
    ) {
        if key.is_none() {
            return;
        }
        self.close_imod_with_key_and_file(key, file, axis_id, description, warn_once, false);
    }

    /// Java `closeImod(String, File, AxisID, String, boolean, boolean)`.
    fn close_imod_key_file_move(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        warn_once: bool,
        file_move: bool,
    ) {
        if key.is_none() {
            return;
        }
        self.close_imod_with_key_and_file(key, file, axis_id, description, warn_once, file_move);
    }

    /// Java package-private `closeImod(FileKey, String, AxisID, boolean)`.
    fn close_imod_file_key_and_name(
        &self,
        file_key: Option<Infallible>,
        file_name: Option<&str>,
        axis_id: Option<AxisID>,
        warn_once: bool,
    ) {
        // TODO(unit): needs etomo/process/BaseImodManager.java.
        let _ = (file_key, file_name, axis_id, warn_once);
    }

    /// Java `closeImod(String, AxisID, String, boolean)`.
    fn close_imod(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        warn_once: bool,
    ) -> bool {
        self.close_imod_with_message(key, axis_id, description, None, warn_once)
    }

    /// Java `closeImods`.  Returns a group of up to three files; true if files were
    /// closed or did not need to be closed.
    fn close_imods(
        &self,
        key1: Option<&str>,
        key2: Option<&str>,
        key3: Option<&str>,
        axis_id: Option<AxisID>,
        descr: Option<&str>,
        message: Option<&str>,
        question: Option<&str>,
    ) -> bool {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImods:key1:{}key2:{}key3:{},axisID:{},message:{}",
                key1.unwrap_or("null"),
                key2.unwrap_or("null"),
                key3.unwrap_or("null"),
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                },
                message.unwrap_or("null")
            );
        }
        if key1.is_none() && key2.is_none() && key3.is_none() {
            return true;
        }
        // TODO(unit): needs etomo/process/BaseImodManager.java (`imodManager.isOpen`/`quit`)
        // and etomo/ui/swing/UIHarness.java (`openYesNoDialog`).  With `imodManager`
        // null the source throws a NullPointerException, which is not a behaviour to
        // reproduce; the translated method stops here.
        let _ = (descr, question);
        false
    }

    /// Java `closeImod(String, AxisID, String, String, boolean)`.  Close the 3dmod
    /// instance denoted by key and axisID if either the `--autoclose3dmod` param was
    /// passed to etomo, or the user wants the 3dmod to be closed.
    fn close_imod_with_message(
        &self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        message: Option<&str>,
        warn_once: bool,
    ) -> bool {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImod:key:{},axisID:{},description:{},message:{}",
                key.unwrap_or("null"),
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                },
                description.unwrap_or("null"),
                message.unwrap_or("null")
            );
        }
        if key.is_none() {
            return true;
        }
        // TODO(unit): needs etomo/process/BaseImodManager.java and
        // etomo/ui/swing/UIHarness.java - see `closeImods`.
        let _ = warn_once;
        false
    }

    /// Java private `closeImod(String, String, AxisID, String, boolean)`.
    fn close_imod_with_file_name(
        &self,
        key: Option<&str>,
        file_name: Option<&str>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        stale: bool,
    ) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImod:key:{},fileName:{},axisID:{},description:{},stale:{}",
                key.unwrap_or("null"),
                file_name.unwrap_or("null"),
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                },
                description.unwrap_or("null"),
                stale
            );
        }
        if key.is_none() {
            return;
        }
        // TODO(unit): needs etomo/process/BaseImodManager.java and
        // etomo/ui/swing/UIHarness.java - see `closeImods`.
    }

    /// Java private `closeImodWithKeyAndFile`.
    fn close_imod_with_key_and_file(
        &self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        warn_once: bool,
        file_move: bool,
    ) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImod:key:{},fileName:{},axisID:{},description:{},warnOnce:{}",
                key.unwrap_or("null"),
                match file {
                    None => "null".to_string(),
                    Some(file) => file
                        .file_name()
                        .map(|name| name.to_string_lossy().to_string())
                        .unwrap_or_default(),
                },
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                },
                description.unwrap_or("null"),
                warn_once
            );
        }
        if key.is_none() {
            return;
        }
        // TODO(unit): needs etomo/process/BaseImodManager.java and
        // etomo/ui/swing/UIHarness.java - see `closeImods`.
        let _ = file_move;
    }

    /// Java package-private `getFileLockMessage`.
    fn get_file_lock_message(&self, spacer: Option<&str>) -> String {
        if utilities::is_windows_os() || etomo_director::EtomoDirector::is_simulate_windows() {
            let spacer = spacer.unwrap_or("");
            return format!(
                "{}You will need to close this file in order to proceed.",
                spacer
            );
        }
        "".to_string()
    }

    /// Java package-private `releaseFile`.
    fn release_file(&self) {
        if !utilities::is_windows_os() {
            // Nothing to do
            return;
        }
        // Give Windows a chance to release control of the file.
        if *self.base().debug.lock().unwrap() {
            eprintln!("Waiting for Windows file lock to be released.");
        }
        std::thread::sleep(std::time::Duration::from_millis(3000));
    }

    /// Java private `close3dmods`.
    fn close3dmods(&self, axis_id: Option<AxisID>) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() && *DEBUG {
            eprintln!(
                "close3dmods:axisID:{}",
                match axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                }
            );
        }
        // Should we close the 3dmod windows
        // TODO(unit): needs etomo/process/BaseImodManager.java (`imodManager.isOpen()` and
        // `quit()`) and etomo/ui/swing/UIHarness.java (`openYesNoDialog`).
    }

    /// Java private `disconnect3dmods`.
    fn disconnect3dmods(&self) {
        // TODO(unit): needs etomo/process/BaseImodManager.java -
        // `imodManager.disconnect()`, wrapped in `catch (Throwable e)`.
    }

    /// Java package-private `close`.
    fn close(&self, axis_id: Option<AxisID>) -> bool {
        if !self.check_next_process(axis_id) {
            return false;
        }
        self.close3dmods(axis_id);
        self.disconnect3dmods();
        true
    }

    /// The `BaseManager` body of Java `exitProgram`, reached from a subclass override
    /// the way `super.exitProgram(...)` is.  Rust cannot call a trait method's default
    /// body from an implementation that overrides it, so the body lives here and
    /// the overridable method delegates to it; `etomo/join_manager.rs` is the
    /// caller that needs this.
    /// Java package-private `exitProgram`.  Exit the program.  To guarantee that etomo
    /// can always exit, catch all unrecognized Exceptions and Errors and return true.
    fn exit_program_super(&self, axis_id: Option<AxisID>) -> bool {
        *self.base().exiting.lock().unwrap() = true;
        // Check for processes that will die if etomo exits
        // TODO(unit): needs etomo/process/BaseProcessManager.java,
        // etomo/process/AxisProcessData.java, etomo/ui/swing/UIHarness.java and
        // etomo/process/ImodqtassistProcess.java - the running-process check, the warning
        // dialog and `ImodqtassistProcess.INSTANCE.quit()`.  `getProcessManager()` is
        // null here, so the check is skipped exactly as the source skips it.
        if !self.check_next_process(axis_id) {
            return false;
        }
        self.close3dmods(axis_id);
        // Do this even if everything else fails
        self.disconnect3dmods();
        true
    }

    /// Java `exitProgram`.
    fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        self.exit_program_super(axis_id)
    }

    /// Java private `checkUnidentifiedProcess`.
    fn check_unidentified_process(&self, axis_id: Option<AxisID>) -> bool {
        // TODO(unit): needs etomo/process/AxisProcessData.java,
        // etomo/process/ProcessInterface.java, etomo/process/ProcessData.java and
        // etomo/ui/swing/UIHarness.java - `axisProcessData.getThread(axisID)` is null
        // here, which is the source's early `return true`.
        let _ = axis_id;
        true
    }

    /// Java `isDualAxis`.  Check if the current data set is a dual axis data set.
    fn is_dual_axis(&self) -> bool {
        // The source dereferences `getBaseMetaData()` without a null check.
        !matches!(
            self.get_base_meta_data()
                .map(|meta_data| meta_data.base().get_axis_type()),
            Some(AxisType::SingleAxis)
        )
    }

    /// Java `isExiting`.
    fn is_exiting(&self) -> bool {
        *self.base().exiting.lock().unwrap()
    }

    /// Java `imodGetRubberbandCoordinates`.
    fn imod_get_rubberband_coordinates(
        &self,
        imod_key: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Option<Vec<String>> {
        // TODO(unit): needs etomo/process/BaseImodManager.java
        // (`imodManager.getRubberbandCoordinates`) and etomo/ui/swing/UIHarness.java for
        // the three catch arms.
        let _ = (imod_key, axis_id);
        None
    }

    /// Java package-private `setPanel`.
    fn set_panel(&self) {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java, etomo/ui/swing/MainPanel.java
        // and etomo/type/UserConfiguration.java - `uiHarness.pack/doLayout/validate`,
        // `getMainPanel().setSize/setDividerLocation` and the stored window dimensions.
    }

    /// Java `isValid`.
    fn is_valid(&self) -> bool {
        true
    }

    /// Java `imodOpen(String, int, File, int, Run3dmodMenuOptions)`.  Open or raise a
    /// specific 3dmod to view a file with binning, or open a new 3dmod.  Returns the
    /// index of the 3dmod opened or raised.
    fn imod_open_with_binning(
        &self,
        imod_key: Option<&str>,
        imod_index: i32,
        file: Option<&Path>,
        binning: i32,
        menu_options: Option<Infallible>,
    ) -> i32 {
        // TODO(unit): needs etomo/process/BaseImodManager.java,
        // etomo/type/Run3dmodMenuOptions.java and etomo/ui/swing/UIHarness.java.
        let _ = (imod_key, file, binning, menu_options);
        imod_index
    }

    /// Java `imodOpen(String, int, String, String, Run3dmodMenuOptions)`.  Open or raise
    /// a specific 3dmod to view a file with a model, or open a new 3dmod.
    fn imod_open_with_model(
        &self,
        imod_key: Option<&str>,
        imod_index: i32,
        absolute_file_path: Option<&str>,
        absolute_model_path: Option<&str>,
        menu_options: Option<Infallible>,
    ) -> i32 {
        // TODO(unit): needs etomo/process/BaseImodManager.java,
        // etomo/type/Run3dmodMenuOptions.java and etomo/ui/swing/UIHarness.java.
        let _ = (
            imod_key,
            absolute_file_path,
            absolute_model_path,
            menu_options,
        );
        imod_index
    }

    /// Java `imodOpen(String, Run3dmodMenuOptions)`.  Open 3dmod.
    fn imod_open(&self, imod_key: Option<&str>, menu_options: Option<Infallible>) {
        // TODO(unit): needs etomo/process/BaseImodManager.java,
        // etomo/type/Run3dmodMenuOptions.java and etomo/ui/swing/UIHarness.java.
        let _ = (imod_key, menu_options);
    }

    /// Java `imodOpen(AxisID, String, String, Run3dmodMenuOptions, boolean)`.
    fn imod_open_axis(
        &self,
        axis_id: Option<AxisID>,
        imod_key: Option<&str>,
        model: Option<&str>,
        menu_options: Option<Infallible>,
        model_mode: bool,
    ) {
        // TODO(unit): needs etomo/process/BaseImodManager.java,
        // etomo/type/Run3dmodMenuOptions.java and etomo/ui/swing/UIHarness.java.
        let _ = (axis_id, imod_key, model, menu_options, model_mode);
    }

    /// Java `getParamFile`.  Return the parameter file as a `File` object.
    fn get_param_file(&self) -> Option<PathBuf> {
        self.base().param_file.lock().unwrap().clone()
    }

    /// Java package-private `loadParamFile`.  Loads storables, sets the param file, and
    /// sets up the `ImodManager`.
    fn load_param_file(
        &self,
        param_file: Option<&Path>,
        axis_id: Option<AxisID>,
        loaded_from_a_different_file: bool,
    ) -> bool {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java for the four error dialogs and
        // etomo/type/UserConfiguration.java for `userConfig.putDataFile`; the
        // `ParameterStore` half is blocked as in `saveStorable`.
        let _ = (param_file, axis_id, loaded_from_a_different_file);
        false
    }

    /// Java package-private `backupFile`.
    fn backup_file(&self, file: Option<&Path>, axis_id: Option<AxisID>) -> bool {
        // TODO(unit): needs etomo/ui/swing/UIHarness.java for the failure dialog; the
        // `Utilities.renameFile(this, axisID, file, backupFile, false, false, false)`
        // half needs the manager-aware overload of that method.
        let _ = (file, axis_id);
        true
    }

    /// Java `processDone(String, int, ProcessName, AxisID, ProcessEndState, boolean,
    /// ProcessResultDisplay, ProcessSeries, boolean)`.  Stop progress bar and start next
    /// process.
    #[allow(clippy::too_many_arguments)]
    fn process_done_end_state(
        &self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<Infallible>,
        axis_id: Option<AxisID>,
        end_state: Option<Infallible>,
        failed: bool,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        non_blocking: bool,
    ) {
        if *self.base().debug.lock().unwrap() {
            // TODO(unit): needs etomo/type/ProcessName.java's instance half and
            // etomo/type/ProcessEndState.java - the trace prints both through
            // `toString`.
            eprintln!(
                "BaseProcessManager.processDone:exitValue:{},processName:null,endState:null",
                exit_value
            );
        }
        self.process_done(
            thread_name,
            exit_value,
            process_name,
            axis_id,
            false,
            end_state,
            None,
            failed,
            process_result_display,
            process_series,
            non_blocking,
        );
    }

    /// Java `processDone(String, int, ProcessName, AxisID, boolean, ProcessEndState,
    /// boolean, ProcessResultDisplay, ProcessSeries, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn process_done_force(
        &self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<Infallible>,
        axis_id: Option<AxisID>,
        force_next_process: bool,
        end_state: Option<Infallible>,
        failed: bool,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        non_blocking: bool,
    ) {
        self.process_done(
            thread_name,
            exit_value,
            process_name,
            axis_id,
            force_next_process,
            end_state,
            None,
            failed,
            process_result_display,
            process_series,
            non_blocking,
        );
    }

    /// Java `processDone(String, int, ProcessName, AxisID, boolean, ProcessEndState,
    /// String, boolean, ProcessResultDisplay, ProcessSeries, boolean)`.  Notification
    /// message that a background process is done.
    #[allow(clippy::too_many_arguments)]
    fn process_done(
        &self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<Infallible>,
        axis_id: Option<AxisID>,
        force_next_process: bool,
        end_state: Option<Infallible>,
        status_string: Option<&str>,
        failed: bool,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        non_blocking: bool,
    ) {
        // TODO(unit): needs etomo/ui/swing/MainPanel.java, etomo/ui/swing/ParallelPanel.java,
        // etomo/ui/swing/UIHarness.java, etomo/process/BaseProcessManager.java,
        // etomo/ProcessSeries.java, etomo/type/ProcessEndState.java and
        // etomo/logic/BusyStatusMediator.java - every statement after the thread-name
        // bookkeeping dereferences one of them.
        if *self.base().debug.lock().unwrap() {
            eprintln!(
                "BaseManager.processDone:exitValue:{},processName:null,endState:null,processSeries:null",
                exit_value
            );
        }
        let thread_name = thread_name.unwrap_or("null");
        if thread_name == *self.base().thread_name_a.lock().unwrap() {
            *self.base().thread_name_a.lock().unwrap() = NO_PROCESS_THREAD_NAME.to_string();
            *self.base().background_process_a.lock().unwrap() = false;
            *self.base().background_process_name_a.lock().unwrap() = None;
        } else if thread_name == *self.base().thread_name_b.lock().unwrap() {
            *self.base().thread_name_b.lock().unwrap() = NO_PROCESS_THREAD_NAME.to_string();
        } else if !non_blocking {
            // TODO(unit): needs etomo/ui/swing/UIHarness.java - "Unknown thread
            // finished!!!" dialog.
        }
        self.update_dialog(process_name, axis_id);
        self.set_pause_process(axis_id, end_state, process_series);
        let _ = (
            force_next_process,
            status_string,
            failed,
            process_result_display,
        );
        self.send_event(axis_id, process_name, end_state, failed);
    }

    /// Java `sendEvent`.  Empty in the base class.
    fn send_event(
        &self,
        axis_id: Option<AxisID>,
        process_name: Option<Infallible>,
        process_end_state: Option<Infallible>,
        failed: bool,
    ) {
        let _ = (axis_id, process_name, process_end_state, failed);
    }

    /// Java private `isReconnectRun`.  Should remain private.
    fn is_reconnect_run(&self, axis_id: Option<AxisID>) -> bool {
        if axis_id == Some(AxisID::Second) {
            return *self.base().reconnect_run_b.lock().unwrap();
        }
        *self.base().reconnect_run_a.lock().unwrap()
    }

    /// Java private `setReconnectRun`.  Should remain private.
    fn set_reconnect_run(&self, axis_id: Option<AxisID>) {
        if axis_id == Some(AxisID::Second) {
            *self.base().reconnect_run_b.lock().unwrap() = true;
        } else {
            *self.base().reconnect_run_a.lock().unwrap() = true;
        }
    }

    /// Java `saveAll`.  Save param file and open dialogs.  Returns a timestamp, or null
    /// if this functionality is not implemented.
    fn save_all(&self, errmsg: &mut String) -> Option<String> {
        let _ = errmsg;
        None
    }

    /// Java `doAutomation`.
    fn do_automation(&self, local_arguments: Option<Infallible>) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_exit() {
            // TODO(unit): needs etomo/ui/swing/UIHarness.java - `uiHarness.exit(AxisID.ONLY,
            // 0)`.
        }
        // TODO(unit): needs etomo/EtomoDirector.java's nested `LocalArguments` - the
        // parameter's declared type.
        let _ = local_arguments;
    }

    /// Java package-private `reconnectToDifferentHost`.  The manager default is that it
    /// cannot connect to a different host.
    fn reconnect_to_different_host(
        &self,
        process_data: Option<Infallible>,
        axis_id: Option<AxisID>,
    ) -> bool {
        // TODO(unit): needs etomo/process/ProcessData.java and
        // etomo/ui/swing/UIHarness.java - the whole body runs only for a non-null
        // `processData`, which the parameter's type makes impossible.
        let _ = (process_data, axis_id);
        false
    }

    /// Java `reconnect`.  Attempts to reconnect to a currently running process.
    fn reconnect(
        &self,
        process_data: Option<Infallible>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<Infallible>,
    ) -> bool {
        // TODO(unit): needs etomo/process/BaseProcessManager.java (`unblockAxis`) and
        // etomo/process/ProcessData.java.
        if self.is_reconnect_run(axis_id) {
            // Just in case
            return false;
        }
        self.set_reconnect_run(axis_id);
        let _ = (process_data, multi_line_messages, messages_array);
        false
    }

    /// Java `reconnectProcesschunks`.
    fn reconnect_processchunks(
        &self,
        process_data: Option<Infallible>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<Infallible>,
    ) -> bool {
        // TODO(unit): needs etomo/process/ProcessData.java,
        // etomo/process/BaseProcessManager.java, etomo/ProcessSeries.java,
        // etomo/ui/swing/MainPanel.java and etomo/type/ProcessResultDisplay.java.
        let _ = (process_data, axis_id, multi_line_messages, messages_array);
        false
    }

    /// Java package-private `isPopupChunkWarnings`.
    fn is_popup_chunk_warnings(&self) -> bool {
        true
    }

    /// Java `tomodataplots`.
    fn tomodataplots(
        &self,
        task: Option<&dyn TaskInterface>,
        axis_id: Option<AxisID>,
        process_series: Option<Infallible>,
        alternative_input_file_absolute_path: Option<&str>,
    ) {
        if self.can_run_tomodataplots(task, axis_id) {
            // TODO(unit): needs etomo/comscript/TomodataplotsParam.java and
            // etomo/process/BaseProcessManager.java - `new TomodataplotsParam()` and
            // `processManager.tomodataplots(param, axisID)`.
            let _ = alternative_input_file_absolute_path;
        }
        if process_series.is_some() {
            // TODO(unit): needs etomo/type/ConstProcessSeries.java -
            // `processSeries.startNextProcess(AxisID.ONLY, null)`.
        }
    }

    /// Java `canRunTomodataplots`.
    fn can_run_tomodataplots(
        &self,
        task: Option<&dyn TaskInterface>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let _ = (task, axis_id);
        true
    }

    /// Java `processchunks`.  Run processchunks.
    #[allow(clippy::too_many_arguments)]
    fn processchunks(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        popup_chunk_warnings: bool,
        processing_method: Option<Infallible>,
        multi_line_messages: bool,
        dialog_type: Option<Infallible>,
        run_type: Option<Infallible>,
        managed_process_data: Option<Infallible>,
        messages_array: Option<Infallible>,
    ) -> bool {
        // TODO(unit): needs etomo/ui/swing/ParallelPanel.java, etomo/ui/swing/MainPanel.java,
        // etomo/ui/swing/UIHarness.java, etomo/ui/SharedStrings.java,
        // etomo/comscript/ProcesschunksParam.java, etomo/ProcessSeries.java and
        // etomo/process/BaseProcessManager.java.
        let _ = (
            axis_id,
            param,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
            dialog_type,
            run_type,
            managed_process_data,
            messages_array,
        );
        false
    }

    /// Java package-private `processDone(AxisID, ProcessResultDisplay,
    /// ConstProcessSeries)`.  A process-done function for processes which are completed
    /// while the original manager function waits and do not use the process manager.
    fn process_done_secondary(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
    ) {
        // TODO(unit): needs etomo/ui/swing/ParallelPanel.java, etomo/ui/swing/MainPanel.java,
        // etomo/type/ConstProcessSeries.java and etomo/logic/BusyStatusMediator.java.
        let _ = (axis_id, process_result_display);
        if process_series.is_none() {
            utilities::timestamp_marker(Some("processDone"));
        }
    }

    /// Java `sendMsgProcessStarting`.
    fn send_msg_process_starting(&self, process_result_display: Option<Infallible>) {
        // TODO(unit): needs etomo/type/ProcessResultDisplay.java -
        // `processResultDisplay.msgProcessStarting()`.
        let _ = process_result_display;
    }

    /// Java package-private `sendMsgProcessFailedToStart`.
    fn send_msg_process_failed_to_start(&self, process_result_display: Option<Infallible>) {
        // TODO(unit): needs etomo/type/ProcessResultDisplay.java -
        // `processResultDisplay.msgProcessFailedToStart()`.
        let _ = process_result_display;
    }

    /// Java package-private `sendMsgProcessSucceeded`.
    fn send_msg_process_succeeded(&self, process_result_display: Option<Infallible>) {
        // TODO(unit): needs etomo/type/ProcessResultDisplay.java -
        // `processResultDisplay.msgProcessSucceeded()`.
        let _ = process_result_display;
    }

    /// Java package-private `sendMsgProcessFailed`.
    fn send_msg_process_failed(&self, process_result_display: Option<Infallible>) {
        // TODO(unit): needs etomo/type/ProcessResultDisplay.java -
        // `processResultDisplay.msgProcessFailed()`.
        let _ = process_result_display;
    }

    /// Java `setCurrentDialogType`.  Set the current dialog type.  Returns the action
    /// message.
    fn set_current_dialog_type(
        &self,
        dialog_type: Option<DialogType>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        let action_message;
        if axis_id == Some(AxisID::Second) {
            action_message = utilities::prepare_dialog_action_message(
                dialog_type,
                axis_id.unwrap_or(AxisID::Only),
                *self.base().current_dialog_type_b.lock().unwrap(),
            );
            *self.base().current_dialog_type_b.lock().unwrap() = dialog_type;
        } else {
            action_message = utilities::prepare_dialog_action_message(
                dialog_type,
                axis_id.unwrap_or(AxisID::Only),
                *self.base().current_dialog_type_a.lock().unwrap(),
            );
            *self.base().current_dialog_type_a.lock().unwrap() = dialog_type;
        }
        action_message
    }

    /// Java package-private `getCurrentDialogType`.  Gets the current dialog type.
    fn get_current_dialog_type(&self, axis_id: Option<AxisID>) -> Option<DialogType> {
        if axis_id == Some(AxisID::Second) {
            return *self.base().current_dialog_type_b.lock().unwrap();
        }
        *self.base().current_dialog_type_a.lock().unwrap()
    }

    /// The `BaseManager` body of Java `setDebug`, reached from a subclass override
    /// the way `super.setDebug(...)` is.  Rust cannot call a trait method's default
    /// body from an implementation that overrides it, so the body lives here and
    /// the overridable method delegates to it; `etomo/join_manager.rs` is the
    /// caller that needs this.
    /// Java `setDebug`.
    fn set_debug_super(&self, debug: bool) {
        *self.base().debug.lock().unwrap() = debug;
    }

    /// Java `setDebug`.
    fn set_debug(&self, debug: bool) {
        self.set_debug_super(debug)
    }

    /// Java `startProgressBar`.  Start generic progress bar.
    fn start_progress_bar(
        &self,
        label: Option<&str>,
        axis_id: Option<AxisID>,
        process_name: Option<Infallible>,
    ) {
        // TODO(unit): needs etomo/ui/swing/MainPanel.java -
        // `getMainPanel().startProgressBar(label, axisID, processName)`.
        let _ = (label, axis_id, process_name);
    }

    /// Java `stopProgressBar`.
    fn stop_progress_bar(&self, axis_id: Option<AxisID>) {
        // TODO(unit): needs etomo/ui/swing/MainPanel.java -
        // `getMainPanel().stopProgressBar(axisID)`.
        let _ = axis_id;
    }

    /// Java `startLoad`.
    fn start_load(&self, param: Option<Infallible>, monitor: Option<Infallible>) {
        // TODO(unit): needs etomo/process/BaseProcessManager.java,
        // etomo/comscript/IntermittentCommand.java and etomo/process/LoadMonitor.java.
        let _ = (param, monitor);
    }

    /// Java `endLoad`.
    fn end_load(&self, param: Option<Infallible>, monitor: Option<Infallible>) {
        // TODO(unit): needs etomo/process/BaseProcessManager.java,
        // etomo/comscript/IntermittentCommand.java and etomo/process/LoadMonitor.java.
        let _ = (param, monitor);
    }

    /// Java `stopLoad`.
    fn stop_load(&self, param: Option<Infallible>, monitor: Option<Infallible>) {
        // TODO(unit): needs etomo/process/BaseProcessManager.java,
        // etomo/comscript/IntermittentCommand.java and etomo/process/LoadMonitor.java.
        let _ = (param, monitor);
    }

    /// Java `msgCurrentManagerChanged`.  Called when the manager's interface is either
    /// displayed or hidden.
    fn msg_current_manager_changed(&self, current: bool) {
        if current {
            self.make_property_user_dir_local();
        }
        if self.base().log_window.is_some() {
            // TODO(unit): needs etomo/ui/swing/LogWindow.java -
            // `logWindow.msgCurrentManagerChanged(current, isStartupPopupOpen())`.
        }
    }

    /// Java `isStartupPopupOpen`.  Startup popup dialogs are used to set the dataset
    /// name and location in some interfaces.
    fn is_startup_popup_open(&self) -> bool {
        // Most interfaces don't use a startup popup.
        false
    }

    /// Java `makePropertyUserDirLocal`.
    fn make_property_user_dir_local(&self) {
        // make the manager's directory the local directory
        let property_user_dir = self.base().property_user_dir.lock().unwrap().clone();
        match property_user_dir {
            None => {
                // TODO(unit): needs etomo/EtomoDirector.java's `makeOriginalDirLocal`,
                // which is part of that class's untranslated manager-window half.
            }
            Some(property_user_dir) => {
                // `System.setProperty("user.dir", propertyUserDir)`.  The JVM's `user.dir`
                // is a process-wide property that does not change the working directory;
                // `PWD` is the environment variable this translation reads for it.
                unsafe { std::env::set_var("PWD", property_user_dir) };
            }
        }
    }

    /// Java `savePreferences`.
    fn save_preferences(&self, axis_id: Option<AxisID>, storable: Option<&dyn Storable>) {
        if storable.is_none() {
            return;
        }
        let _main_panel = self.get_main_panel();
        // TODO(unit): needs etomo/ui/swing/UIHarness.java for the failure dialog and
        // `EtomoDirector.INSTANCE.getParameterStore()`, which is part of that class's
        // untranslated half.
        let _ = axis_id;
    }

    /// Java `setThreadName`.  Map the thread name to the correct axis.
    fn set_thread_name(&self, name: Option<&str>, axis_id: Option<AxisID>) {
        let name = name.unwrap_or(NO_PROCESS_THREAD_NAME).to_string();
        if axis_id == Some(AxisID::Second) {
            *self.base().thread_name_b.lock().unwrap() = name;
        } else {
            *self.base().thread_name_a.lock().unwrap() = name;
        }
        // Are there processes covered by this and not covered by AxisProcessData?
        // busyStatusMediator.msgThreadChanged(axisID,
        // name != null && !name.equals(NO_PROCESS_THREAD_NAME));
    }

    /// Java `resetCurrentProcesschunks`.
    fn reset_current_processchunks(&self, axis_id: Option<AxisID>) {
        if let Some(meta_data) = self.get_base_meta_data() {
            meta_data
                .base()
                .reset_current_processchunks_root_name(axis_id);
            meta_data
                .base()
                .reset_current_processchunks_subdir_name(axis_id);
        }
    }

    /// Java private `setPauseProcess`.
    fn set_pause_process(
        &self,
        axis_id: Option<AxisID>,
        end_state: Option<Infallible>,
        process_series: Option<Infallible>,
    ) {
        // TODO(unit): needs etomo/ProcessSeries.java and etomo/type/ProcessEndState.java -
        // `processSeries.setPauseProcess(Task.RESUME)` when the end state is `PAUSED`.
        let _ = (axis_id, end_state, process_series);
    }

    /// Java private `saveResume`.
    #[allow(clippy::too_many_arguments)]
    fn save_resume(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        popup_chunk_warnings: bool,
        processing_method: Option<Infallible>,
        multi_line_messages: bool,
    ) {
        // TODO(unit): needs etomo/process/AxisProcessData.java -
        // `axisProcessData.isPausing(axisID)` guards the whole body and
        // `axisProcessData.setWillResume(axisID)` ends it.
        let resume_data = if axis_id == Some(AxisID::Second) {
            &self.base().resume_data_b
        } else {
            &self.base().resume_data_a
        };
        let _ = resume_data;
        let _ = (
            param,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
        );
    }

    /// Java private `resume(AxisID)`.
    fn resume_axis(&self, axis_id: Option<AxisID>) {
        let resume_data = if axis_id == Some(AxisID::Second) {
            &self.base().resume_data_b
        } else {
            &self.base().resume_data_a
        };
        if !resume_data.is_null() {
            std::thread::sleep(std::time::Duration::from_millis(3000));
            self.resume_private(
                axis_id,
                resume_data.get_processchunks_param(),
                resume_data.get_process_result_display(),
                resume_data.get_process_series(),
                resume_data.is_popup_chunk_warnings(),
                resume_data.get_processing_method(),
                resume_data.is_multi_line_messages(),
            );
        }
        resume_data.reset();
    }

    /// Java package-private `updateProcessChunks`.  Pass in param to allow override
    /// functions to modify it.  Constructs (if necessary), modifies, and returns param.
    fn update_process_chunks(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        root_name: Option<&str>,
        subcommand_details: Option<Infallible>,
        dialog_type: Option<Infallible>,
    ) -> Option<Infallible> {
        // TODO(unit): needs etomo/comscript/ProcesschunksParam.java,
        // etomo/comscript/CommandDetails.java and etomo/type/DialogType.java.
        let _ = (axis_id, root_name, subcommand_details, dialog_type);
        param
    }

    /// Java `resume(AxisID, ProcesschunksParam, ProcessResultDisplay, ProcessSeries,
    /// CommandDetails, boolean, ProcessingMethod, boolean, DialogType)`.  Get the
    /// current processchunks root name from meta data; if it exists, attempt to resume
    /// processchunks.
    #[allow(clippy::too_many_arguments)]
    fn resume(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        subcommand_details: Option<Infallible>,
        popup_chunk_warnings: bool,
        processing_method: Option<Infallible>,
        multi_line_messages: bool,
        dialog_type: Option<Infallible>,
    ) {
        // TODO(unit): needs etomo/ProcessSeries.java, etomo/process/BaseProcessManager.java
        // (`inUse`), etomo/ui/swing/ParallelPanel.java (`getResumeParameters`) and
        // etomo/ui/swing/UIHarness.java.
        let _ = (
            axis_id,
            param,
            process_result_display,
            process_series,
            subcommand_details,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
            dialog_type,
        );
    }

    /// Java package-private `createRunList`.
    fn create_run_list(&self, run_type: Option<Infallible>) -> Option<Infallible> {
        // TODO(unit): needs etomo/type/RunList.java and etomo/type/RunType.java - the
        // return and parameter types.  The base class returns null.
        let _ = run_type;
        None
    }

    /// Java package-private `getMessagesArray`.
    fn get_messages_array(&self) -> Option<Infallible> {
        // TODO(unit): needs etomo/process/ProcessMessages.java - the return type.  The
        // base class returns null.
        None
    }

    /// Java private `resume(AxisID, ProcesschunksParam, ProcessResultDisplay,
    /// ProcessSeries, boolean, ProcessingMethod, boolean)`.  Resume processchunks.
    #[allow(clippy::too_many_arguments)]
    fn resume_private(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Infallible>,
        process_result_display: Option<Infallible>,
        process_series: Option<Infallible>,
        popup_chunk_warnings: bool,
        processing_method: Option<Infallible>,
        multi_line_messages: bool,
    ) {
        // TODO(unit): needs etomo/comscript/ProcesschunksParam.java,
        // etomo/ui/swing/ParallelPanel.java, etomo/ui/swing/MainPanel.java,
        // etomo/ui/swing/UIHarness.java, etomo/process/ProcessData.java and
        // etomo/process/BaseProcessManager.java.
        let _ = (
            axis_id,
            param,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
        );
    }

    /// Java `tomosnapshot`.
    fn tomosnapshot(&self, axis_id: Option<AxisID>) {
        // TODO(unit): needs etomo/process/BaseProcessManager.java
        // (`processManager.tomosnapshot(axisID, isTomosnapshotThumbnail())`) and
        // etomo/ui/swing/UIHarness.java for the `else` dialog.
        let _ = axis_id;
    }

    /// Java package-private `isTomosnapshotThumbnail`.
    fn is_tomosnapshot_thumbnail(&self) -> bool {
        false
    }
}

/// Java `toString`.  Returns `"[" + paramString() + "]"`.
impl std::fmt::Display for dyn BaseManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.param_string().unwrap_or("null".to_string()))
    }
}

/// Java `getIMODBinPath`, a static method.  Return the absolute IMOD bin path.
pub fn get_imod_bin_path() -> Option<String> {
    etomo_director::INSTANCE
        .lock()
        .unwrap()
        .get_imod_directory()
        .map(|directory| {
            format!(
                "{}{}bin{}",
                directory.display(),
                std::path::MAIN_SEPARATOR,
                std::path::MAIN_SEPARATOR
            )
        })
}

/// Java `chunkComscriptAction`, a static method.
pub fn chunk_comscript_action(root: Option<Infallible>) -> Option<PathBuf> {
    // TODO(unit): needs etomo/ui/swing/FileChooser.java, etomo/ui/swing/FixedDim.java and
    // etomo/storage/ChunkComscriptFileFilter.java - the whole body is a Swing
    // `JFileChooser` dialog.
    let _ = root;
    None
}
