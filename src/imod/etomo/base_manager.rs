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
//! `new ImodManager(this)`.  A Java `super.x(...)` call from a subclass override reaches
//! the `BaseManager` body through the trait's `x_super` method, which holds that body;
//! the overridable `x` delegates to it.
//!
//! **Lifetime.**  Java's managers are created by `EtomoDirector`, held in its manager
//! list and never collected while the program runs.  As in `etomo/ui/swing/token.rs`,
//! the translation models that by a leaked allocation, so `&'static dyn BaseManager` is
//! the reference type every unit that takes a `BaseManager` parameter uses, and
//! `Option<&'static dyn BaseManager>` carries Java's nullable one.  A method whose body
//! passes `this` on (to `UIHarness`, `ProcessSeries`, `FileType`, `ParameterStore`, ...)
//! therefore takes `&'static self`.  The trait is `Send + Sync` because a manager is
//! reachable from several threads in the source (`EmergencyMonitor`, the process
//! threads) and because Java's fields are mutated through such shared references; each
//! mutable field therefore carries its own lock, the same modelling
//! `etomo/storage/log_file.rs` uses for Java's instance monitors.  The Swing objects the
//! class holds (`logWindow`, the two `ProcessingMethodMediator`s) live on the event
//! dispatch thread (`util/event_queue.rs`).
#![allow(dead_code)]

use crate::imod::etomo::comscript::command_details::CommandDetails;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::tomodataplots_param::{self, TomodataplotsParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::logic::busy_status_mediator::{BusyStatusListener, BusyStatusMediator};
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::manager_key::ManagerKey;
use crate::imod::etomo::process::axis_process_data::AxisProcessData;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::{AxisBusyException, BaseProcessManager};
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::process::imod_manager::ImodManager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::imodqtassist_process;
use crate::imod::etomo::process::load_monitor::LoadMonitor;
use crate::imod::etomo::process::process_data::ProcessData;
use crate::imod::etomo::process::process_interface::{ProcessResultDisplayRef, ProcessSeriesRef};
use crate::imod::etomo::process::process_messages::MessagesArray;
use crate::imod::etomo::process::process_messages::ProcessMessages;
use crate::imod::etomo::process::tomosetexts_output::TomosetextsOutput;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::file_reader::FileReaderRef;
use crate::imod::etomo::storage::file_writer::FileWriterRef;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError, ReaderId};
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::storage::parameter_store::ParameterStore;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::auto_alignment_meta_data::AutoAlignmentMetaData;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::base_process_track::BaseProcessTrack;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::base_state::BaseState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::directive_map_interface::DirectiveMapInterface;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::log_properties::LogProperties;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use crate::imod::etomo::ui::swing::log_interface::LogInterface;
use crate::imod::etomo::ui::swing::log_window::LogWindow;
use crate::imod::etomo::ui::swing::main_panel::{MainPanel, MainPanelVirtual};
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::process_result_display_factory::ProcessResultDisplayFactory;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::event_queue::{EdtCell, EdtRef};
use crate::imod::etomo::util::unique_key::UniqueKey;
use crate::imod::etomo::util::utilities;
use crate::imod::etomo::util::valid_directory::ValidDirectory;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

/// Java string concatenation of a nullable reference: `toString()` or "null".
fn display_or_null<T: std::fmt::Display>(value: Option<&T>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| value.to_string())
}

/// Java private static `NO_PROCESS_THREAD_NAME`.
const NO_PROCESS_THREAD_NAME: &str = "none";

/// Java private static `headless`, a mutable class variable the constructor assigns.
static HEADLESS: Mutex<bool> = Mutex::new(false);

/// Java `private static final boolean DEBUG =
/// EtomoDirector.INSTANCE.getArguments().isDebug()`.  A static initialiser, so it is
/// read once, the first time the class is touched.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

// Java package-private static `userConfig =
// EtomoDirector.INSTANCE.getUserConfiguration()`: the director's one object, reached
// through `etomo_director::INSTANCE.with_user_configuration(_mut)` at each use.

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
/// The display and the series are event-dispatch-thread objects held by a manager that
/// other threads share, so they are kept as the process layer's thread-crossing
/// references (`ProcessResultDisplayRef`, `ProcessSeriesRef`).
pub struct ResumeData {
    /// Java field `param`, initialised to null.
    param: Mutex<Option<Arc<ProcesschunksParam>>>,
    /// Java field `processResultDisplay`, initialised to null.
    process_result_display: Mutex<Option<ProcessResultDisplayRef>>,
    /// Java field `processSeries`, initialised to null.
    process_series: Mutex<Option<ProcessSeriesRef>>,
    /// Java field `popupChunkWarnings`, initialised to false.
    popup_chunk_warnings: Mutex<bool>,
    /// Java field `processingMethod`, initialised to null.
    processing_method: Mutex<Option<ProcessingMethod>>,
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
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
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
    fn get_processchunks_param(&self) -> Option<Arc<ProcesschunksParam>> {
        self.param.lock().unwrap().clone()
    }

    /// Java private `getProcessResultDisplay`.
    fn get_process_result_display(&self) -> Option<ProcessResultDisplayRef> {
        self.process_result_display.lock().unwrap().clone()
    }

    /// Java private `getProcessSeries`.
    fn get_process_series(&self) -> Option<ProcessSeriesRef> {
        self.process_series.lock().unwrap().clone()
    }

    /// Java private `getProcessingMethod`.
    fn get_processing_method(&self) -> Option<ProcessingMethod> {
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
/// returns it from `BaseManager::base`.  The package-visible Java fields are
/// `pub(crate)`, as subclasses read them directly.
pub struct BaseManagerBase {
    /// Java field `busyStatusMediator`, `new BusyStatusMediator()`.
    pub(crate) busy_status_mediator: Arc<BusyStatusMediator>,
    // Java field `uiHarness`, `UIHarness.INSTANCE`: always the singleton, which is an
    // event-dispatch-thread object (`ui_harness::INSTANCE`, a `thread_local!`); each use
    // below reads the singleton where the source reads the field.
    /// Java field `loadedParamFile`, initialised to false.
    pub(crate) loaded_param_file: Mutex<bool>,
    /// Java field `imodManager`, `new ImodManager(this)`.  imodManager manages the
    /// opening and closing closing of imod(s), message passing for loading model.
    imod_manager: OnceLock<ImodManager>,
    /// Java field `paramFile`, initialised to null.
    pub(crate) param_file: Mutex<Option<PathBuf>>,
    /// Java field `homeDirectory`.
    // FIXME homeDirectory may not have to be visible
    pub(crate) home_directory: Mutex<Option<String>>,
    /// Java field `threadNameA`, initialised to `NO_PROCESS_THREAD_NAME`.
    pub(crate) thread_name_a: Mutex<String>,
    /// Java field `threadNameB`, initialised to `NO_PROCESS_THREAD_NAME`.
    pub(crate) thread_name_b: Mutex<String>,
    /// Java field `backgroundProcessA`, initialised to false.
    pub(crate) background_process_a: Mutex<bool>,
    /// Java field `backgroundProcessNameA`, initialised to null.
    pub(crate) background_process_name_a: Mutex<Option<String>>,
    /// Java field `propertyUserDir`, the working directory for this manager.
    pub(crate) property_user_dir: Mutex<Option<String>>,
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
    /// Java field `parameterStore`, initialised to null.  The store is one object shared
    /// by reference in Java (`getParameterStore` hands it out), hence the `Arc`.
    parameter_store: Mutex<Option<Arc<Mutex<ParameterStore>>>>,
    /// Java field `reconnectRunA`.  True if `reconnect()` has been run for axis A.
    reconnect_run_a: Mutex<bool>,
    /// Java field `reconnectRunB`.  True if `reconnect()` has been run for axis B.
    reconnect_run_b: Mutex<bool>,
    /// Java field `processingMethodMediatorA`, `new ProcessingMethodMediator()`.  The
    /// mediator is a Swing-side object, usable on the thread that built the manager.
    processing_method_mediator_a: EdtRef<ProcessingMethodMediator>,
    /// Java field `processingMethodMediatorB`, `new ProcessingMethodMediator()`.
    processing_method_mediator_b: EdtRef<ProcessingMethodMediator>,
    /// Java field `managerKey`, `new ManagerKey()`.
    ///
    /// `ManagerKey` is mutable in Java, and a director and manager retain the
    /// same holder.  The `Arc` preserves that object identity while the mutex
    /// represents Java's shared mutable object across process/UI threads.
    manager_key: Arc<Mutex<ManagerKey>>,
    /// Java field `axisProcessData`, `new AxisProcessData(this)`: set by
    /// `base_manager`, which is where `this` exists.
    pub(crate) axis_process_data: OnceLock<Arc<AxisProcessData>>,
    /// Java field `resumeDataA`, `new ResumeData()`.
    resume_data_a: ResumeData,
    /// Java field `resumeDataB`, `new ResumeData()`.
    resume_data_b: ResumeData,
    /// Java field `logWindow`, `createLogWindow()`: set by `base_manager`, since
    /// `createLogWindow` is virtual and takes `this`.  An event-dispatch-thread object.
    pub(crate) log_window: EdtCell<Rc<LogWindow>>,
    /// Java field `validBrowsingDirectory`, initialised to null.
    valid_browsing_directory: Mutex<Option<ValidDirectory>>,
    /// Java field `emergencyMonitor`, initialised to null.
    emergency_monitor: Mutex<Option<Arc<EmergencyMonitor>>>,
}

impl BaseManagerBase {
    /// The field initialisers Java runs before the constructor body (those that do not
    /// need `this`; `base_manager` runs the rest).
    pub fn initial() -> BaseManagerBase {
        BaseManagerBase {
            busy_status_mediator: Arc::new(BusyStatusMediator::new()),
            loaded_param_file: Mutex::new(false),
            imod_manager: OnceLock::new(),
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
            processing_method_mediator_a: EdtRef::new(ProcessingMethodMediator::new()),
            processing_method_mediator_b: EdtRef::new(ProcessingMethodMediator::new()),
            manager_key: Arc::new(Mutex::new(ManagerKey::default())),
            axis_process_data: OnceLock::new(),
            resume_data_a: ResumeData::new(),
            resume_data_b: ResumeData::new(),
            log_window: EdtCell::new(),
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
        // Field initialiser `axisProcessData = new AxisProcessData(this)`.
        let _ = self
            .base()
            .axis_process_data
            .set(Arc::new(AxisProcessData::new(
                self.this(),
                Arc::clone(&self.base().busy_status_mediator),
            )));
        // Field initialiser `logWindow = createLogWindow()`.  A null window (headless,
        // or an override returning null) leaves the field null; a real one is created
        // on the event dispatch thread, where every manager with a window is built.
        if let Some(log_window) = self.create_log_window() {
            self.base().log_window.set(Some(log_window));
        }
        *self.base().property_user_dir.lock().unwrap() = std::env::var("PWD").ok();
        self.create_process_track();
        self.create_com_script_manager();
        // Initialize the program settings
        *self.base().debug.lock().unwrap() = etomo_director::ARGUMENTS.lock().unwrap().is_debug();
        *HEADLESS.lock().unwrap() = etomo_director::ARGUMENTS.lock().unwrap().is_headless();
        self.create_main_panel();
        let _ = self.base().imod_manager.set(ImodManager::new(self.this()));
        // The request-handler half of `BaseImodManager`'s constructor, which needs the
        // manager at its final address.
        self.get_imod_manager().start_request_handler();
        self.init_program();
    }

    /// Java `imodManager` field access (`imodManager.xxx(...)`).  Set by
    /// `base_manager`, which every manager constructor runs.
    fn get_imod_manager(&self) -> &ImodManager {
        self.base()
            .imod_manager
            .get()
            .expect("imodManager is created by the BaseManager constructor")
    }

    /// Java `logWindow` field access, which subclasses make directly.  Not a source
    /// member.  The window is an event-dispatch-thread object: off that thread (a
    /// process or monitor thread logging a message) it cannot be reached and reads as
    /// null, so those callers take the source's no-window path.
    fn get_log_window(&self) -> Option<Rc<LogWindow>> {
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            return None;
        }
        self.base().log_window.get()
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
            eprintln!(
                "{},",
                utilities::java_io_file_get_absolute_path(&param_file.to_string_lossy())
            );
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
            display_or_null(self.base().current_dialog_type_a.lock().unwrap().as_ref()),
            display_or_null(self.base().current_dialog_type_b.lock().unwrap().as_ref()),
            *self.base().reconnect_run_a.lock().unwrap(),
            *self.base().reconnect_run_a.lock().unwrap(),
            *self.base().reconnect_run_b.lock().unwrap(),
            self.base().manager_key.lock().unwrap()
        );
    }

    /// Java abstract `getInterfaceType`.
    fn get_interface_type(&self) -> Option<InterfaceType>;

    /// Rust-only (for `ui/swing/slint_bridge.rs`, which draws the tool panel the
    /// manager built): `ToolsManager`'s `toolType`; `None` for every other manager.
    fn get_tool_type(&self) -> Option<crate::imod::etomo::ui::swing::etomo_menu::ToolType> {
        None
    }

    /// Java abstract package-private `createMainPanel`.
    fn create_main_panel(&self);

    /// Java abstract `getBaseMetaData`.
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData>;

    /// Java abstract `getMainPanel`.  Event dispatch thread only.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>>;

    /// Java abstract `getProcessManager`.
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager>;

    /// Runs `f` on this manager's `MainPanel` (Java `getMainPanel()`), on the
    /// event dispatch thread where the panel lives.  Rust-only access path:
    /// the Java returns the panel and calls it from any thread.
    fn with_main_panel(&self, f: &mut dyn FnMut(&MainPanel)) {
        if let Some(main_panel) = self.get_main_panel() {
            f(main_panel.main_panel());
        }
    }

    /// `getMainPanel().<call>(...)` from a process or monitor thread: posted
    /// to the event dispatch thread (`util/event_queue.rs`).
    fn post_main_panel(&'static self, f: Box<dyn FnOnce(&MainPanel) + Send>) {
        let this = self.this();
        crate::imod::etomo::util::event_queue::invoke_later(move || {
            let mut f = Some(f);
            this.with_main_panel(&mut |panel| {
                if let Some(f) = f.take() {
                    f(panel);
                }
            });
        });
    }

    /// Java abstract package-private `getStorables(int)`.  Slots before `offset` are
    /// null, as in the Java array.
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>>;

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
        let mut emergency_monitor = self.base().emergency_monitor.lock().unwrap();
        // The source's double-checked `if (emergencyMonitor == null) synchronized (this)
        // { if (emergencyMonitor == null) ... }`; the lock above is the instance monitor,
        // so the two tests collapse into the one the lock already serialises.
        if emergency_monitor.is_none() {
            *emergency_monitor = Some(Arc::new(EmergencyMonitor::new(Some(self.this()), axis_id)));
        }
        emergency_monitor.clone().unwrap()
    }

    /// Java `tomosetexts`.
    // Bug# 2403
    fn tomosetexts(&'static self) -> Option<TomosetextsOutput> {
        let directory = self.base().property_user_dir.lock().unwrap().clone()?;
        BaseProcessManager::tomosetexts(self.this(), AxisID::Only, Path::new(&directory))
    }

    /// Java `getVerticalScrollBarValue`.
    fn get_vertical_scroll_bar_value(&self, axis_id: Option<AxisID>) -> Option<i32> {
        let main_panel = self.get_main_panel();
        let main_panel = main_panel?;
        // MainPanel maps every axis but SECOND to A, as a null axis is in the source.
        main_panel
            .main_panel()
            .get_vertical_scroll_bar_value(axis_id.unwrap_or(AxisID::Only))
    }

    /// Java `setVerticalScrollBarValue`.
    fn set_vertical_scroll_bar_value(&self, axis_id: Option<AxisID>, value: Option<i32>) {
        let main_panel = self.get_main_panel();
        let Some(main_panel) = main_panel else {
            return;
        };
        main_panel
            .main_panel()
            .set_vertical_scroll_bar_value(axis_id.unwrap_or(AxisID::Only), value);
    }

    /// Java `getPhysicalCores`.  Run "imodqtassist -t" and return the "physical cores"
    /// value.
    fn get_physical_cores(&self, axis_id: Option<AxisID>) -> Option<i32> {
        let process_manager = self.get_process_manager();
        let process_manager = process_manager?;
        // Find the physical cores setting in the standard output.
        // Looking for an output line like this:
        // "Qt ideal thread count = 20 physical cores = 10 logical processors = 20"
        let mut exception: Option<String> = None;
        // Java passes the (possibly null) `axisID` on; `SystemProgram` keeps it.
        let stdout = process_manager.imodqtassist_query(axis_id.unwrap_or(AxisID::Only))?;
        let delimiter = "=";
        let key = "physical cores";
        // Java `split("\\s*" + delimiter + "\\s*")` and `split("\\s+")`.
        let key_pattern = regex::Regex::new(&format!("\\s*{delimiter}\\s*")).unwrap();
        let value_pattern = regex::Regex::new("\\s+").unwrap();
        for line in &stdout {
            if !line.contains(delimiter) {
                continue;
            }
            // Break up by the equals sign in order to find the key and value.
            // Java `String.split` drops trailing empty strings.
            let mut key_array: Vec<&str> = key_pattern.split(line).collect();
            while key_array.last() == Some(&"") {
                key_array.pop();
            }
            // KeyArray should look like this:
            // "Qt ideal thread count", "20 physical cores", "10 logical processors", "20"
            // (Java's `keyArray == null` test cannot be true: `split` never returns null.)
            let mut found = false;
            for element in &key_array {
                if !found {
                    if element.contains(key) {
                        // Found the key in a string like this:
                        // "20 physical cores"
                        found = true;
                    }
                    continue;
                }
                // Return the value from a string like this:
                // "10 logical processors"
                found = false;
                // Return the value.
                let mut value_array: Vec<&str> = value_pattern.split(element).collect();
                while value_array.last() == Some(&"") {
                    value_array.pop();
                }
                // ValueArray should look like this:
                // "10", "logical", "processors"
                if value_array.is_empty() {
                    exception = Some(format!("Warning: unable to parse {line}"));
                    continue;
                }
                let value = converter::to_integer(Some(value_array[0]));
                if value.is_none() {
                    exception = Some(format!("Warning: unable to parse {line}"));
                    continue;
                }
                return value;
            }
        }
        if let Some(exception) = exception {
            // `exception.printStackTrace()`.
            eprintln!("java.lang.IllegalStateException: {exception}");
        }
        None
    }

    /// Java `isBeadfixerDiameterAvailable`.
    fn is_beadfixer_diameter_available(&self) -> bool {
        false
    }

    /// Java `getBeadfixerDiameter`.
    fn get_beadfixer_diameter(&'static self, axis_id: Option<AxisID>) -> Option<i32> {
        let _ = axis_id;
        None
    }

    /// Java `isAddGPUMachineToProcessChunks`.
    fn is_add_gpu_machine_to_process_chunks(&self) -> bool {
        false
    }

    /// Java `addBusyStatusListener`.
    fn add_busy_status_listener(&self, listener: Option<Arc<EdtRef<dyn BusyStatusListener>>>) {
        self.base()
            .busy_status_mediator
            .add_busy_status_listener(listener);
    }

    /// Java `removeBusyStatusListener`.
    fn remove_busy_status_listener(&self, listener: Option<&Arc<EdtRef<dyn BusyStatusListener>>>) {
        self.base()
            .busy_status_mediator
            .remove_busy_status_listener(listener);
    }

    /// Java `getBusyStatusMediator`.
    fn get_busy_status_mediator(&self) -> Arc<BusyStatusMediator> {
        Arc::clone(&self.base().busy_status_mediator)
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
    fn get_browsing_dir(&'static self) -> Option<PathBuf> {
        let mut valid_browsing_directory = self.base().valid_browsing_directory.lock().unwrap();
        if valid_browsing_directory.is_none() {
            let mut directory = ValidDirectory::new(Some(self.this()));
            directory.set_to_property_user_dir();
            *valid_browsing_directory = Some(directory);
        }
        valid_browsing_directory.as_ref().unwrap().get_void()
    }

    /// Java `setBrowsingDir(File)`, the second half of this class's
    /// `etomo/ui/BrowsingDirectory.java` implementation.  Set valid browsing directory.
    fn set_browsing_dir(&'static self, input: Option<&Path>) {
        let mut valid_browsing_directory = self.base().valid_browsing_directory.lock().unwrap();
        if valid_browsing_directory.is_none() && input.is_some() {
            *valid_browsing_directory = Some(ValidDirectory::new(Some(self.this())));
        }
        if let Some(directory) = valid_browsing_directory.as_mut() {
            directory.set_file(input);
        }
    }

    /// Java package-private `setBrowsingDir(String)`.
    fn set_browsing_dir_string(&'static self, input: Option<&str>) {
        let mut valid_browsing_directory = self.base().valid_browsing_directory.lock().unwrap();
        // Java `input.matches("\\s*")`: every character is Java whitespace (or none).
        if valid_browsing_directory.is_none()
            && input.is_some()
            && !input
                .unwrap()
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
        {
            *valid_browsing_directory = Some(ValidDirectory::new(Some(self.this())));
        }
        if let Some(directory) = valid_browsing_directory.as_mut() {
            directory.set_string(input);
        }
    }

    /// Java `getFileSubdirectoryName`.  Return the subdirectory of the dataset location
    /// where some of the files are stored.  This is necessary the NAD manager.  Return
    /// null if there is no subdirectory.
    fn get_file_subdirectory_name(&self) -> Option<String> {
        None
    }

    /// Java `getParallelProcessingDefaultNice`.
    fn get_parallel_processing_default_nice(&self) -> i32 {
        15
    }

    /// `BaseManager.paramString` itself, for the overrides' `super` call.
    /// Java package-private `paramString`.
    fn param_string_super(&self) -> Option<String> {
        self.get_name()
    }

    /// Java package-private `paramString`.
    fn param_string(&self) -> Option<String> {
        self.param_string_super()
    }

    /// Java `getAxisProcessData`.
    fn get_axis_process_data(&self) -> Arc<AxisProcessData> {
        Arc::clone(
            self.base()
                .axis_process_data
                .get()
                .expect("BaseManager constructor body has run"),
        )
    }

    /// Java package-private `createLogWindow`.
    fn create_log_window(&'static self) -> Option<Rc<LogWindow>> {
        LogWindow::get_instance(Some(self.this()))
    }

    /// Java `showHideLog`.
    fn show_hide_log(&self) {
        if let Some(log_window) = self.get_log_window() {
            log_window.show_hide();
        }
    }

    /// Java `getLogInterface`.
    fn get_log_interface(&self) -> Option<Rc<dyn LogInterface>> {
        self.get_log_window()
            .map(|log_window| log_window as Rc<dyn LogInterface>)
    }

    /// Java `getLogProperties`.  Returns `logWindow`.
    ///
    /// `BaseMetaData` keeps the log properties as a `&'static dyn LogProperties` (the
    /// window, like the manager, lives until the program exits), so one strong count of
    /// the window is given up to make that reference; the window is never freed while
    /// the manager holds it anyway.
    fn get_log_properties(&self) -> Option<&'static dyn LogProperties> {
        self.get_log_window().map(|log_window| {
            // SAFETY: `Rc::into_raw` leaks one strong count, so the allocation lives
            // for the rest of the program and the reference stays valid.
            let log_window: &'static LogWindow = unsafe { &*Rc::into_raw(log_window) };
            log_window as &'static dyn LogProperties
        })
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
    fn get_base_screen_state(&self, axis_id: Option<AxisID>) -> Option<&'static BaseScreenState> {
        let _ = axis_id;
        None
    }

    /// Java `getBaseState`.
    fn get_base_state(&self) -> Option<&'static dyn BaseState> {
        None
    }

    /// Java `getProcessResultDisplayFactoryInterface`.  The Java return type is the
    /// `ProcessResultDisplayFactoryInterface`; `ApplicationManager`, the one override,
    /// returns its `ProcessResultDisplayFactory`, which is the type carried here.
    fn get_process_result_display_factory_interface(
        &self,
        axis_id: Option<AxisID>,
    ) -> Option<Rc<ProcessResultDisplayFactory>> {
        let _ = axis_id;
        None
    }

    /// Java package-private `getProcessTrack()`.
    fn get_process_track(&self) -> Option<&'static dyn BaseProcessTrack> {
        None
    }

    /// Java package-private `getProcessTrack(Storable[], int)`.  Empty in the base class.
    fn get_process_track_into(
        &self,
        storable: Option<&mut [Option<&'static dyn Storable>]>,
        index: i32,
    ) {
        let _ = (storable, index);
    }

    /// Java `isInManagerFrame`.
    fn is_in_manager_frame(&self) -> bool {
        false
    }

    /// Java package-private `getAutoAlignmentMetaData`.  The object is shared and
    /// mutated (the auto-alignment panel writes it, `XfalignParam` reads it), so it is
    /// handed out with its lock.
    fn get_auto_alignment_meta_data(
        &self,
    ) -> Option<&'static std::sync::Mutex<AutoAlignmentMetaData>> {
        None
    }

    /// Java `getStatus`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:460): `getMainPanel()` is null
    /// in a headless run and the source throws a NullPointerException; here it returns
    /// null.
    fn get_status(&self) -> Option<String> {
        self.get_main_panel()
            .map(|main_panel| main_panel.main_panel().get_status())
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

    /// `BaseManager.kill` itself, for the overrides' `super` call.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:474): a manager with no
    /// process manager throws a NullPointerException; here nothing is killed.
    fn kill_super(&self, axis_id: Option<AxisID>) {
        if let Some(process_manager) = self.get_process_manager() {
            process_manager.kill(axis_id.unwrap_or(AxisID::Only));
        }
    }

    /// Java `kill`.  Interrupt the currently running thread for this axis.
    fn kill(&self, axis_id: Option<AxisID>) {
        self.kill_super(axis_id)
    }

    /// Java `pause`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:478): a null process manager
    /// throws a NullPointerException; here the pause fails (false).
    fn pause(&self, axis_id: Option<AxisID>) -> bool {
        match self.get_process_manager() {
            Some(process_manager) => process_manager.pause(axis_id.unwrap_or(AxisID::Only)),
            None => false,
        }
    }

    /// Java package-private `processSeriesSucceeded`.  Empty in the base class.
    fn process_series_succeeded(&self, axis_id: Option<AxisID>, process_name: Option<ProcessName>) {
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

    /// Java final `isLoadedParamFile`.
    fn is_loaded_param_file(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    /// `BaseManager.setParamFile(File)` itself, for the overrides' `super` call.
    fn set_param_file_from_super(&self, param_file: Option<&Path>) -> bool {
        *self.base().param_file.lock().unwrap() = param_file.map(|path| path.to_path_buf());
        true
    }

    /// Java `setParamFile(File)`.
    fn set_param_file_from(&self, param_file: Option<&Path>) -> bool {
        self.set_param_file_from_super(param_file)
    }

    /// Java package-private `startNextProcess`.  Returns true if a process was started
    /// (true if the process is recognized).
    #[allow(clippy::too_many_arguments)]
    fn start_next_process(
        &'static self,
        ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        self.start_next_process_super(
            ui_component,
            axis_id,
            process,
            process_result_display,
            process_series,
            dialog_type,
            display,
        )
    }

    /// `BaseManager.startNextProcess` itself, for the overrides' `super` call.
    #[allow(clippy::too_many_arguments)]
    fn start_next_process_super(
        &'static self,
        ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        let _ = (ui_component, process_result_display, dialog_type, display);
        // Java `ResumeData resumeData = axisID == AxisID.SECOND ? resumeDataB :
        // resumeDataA;` - assigned and never read.
        if process.equals_task(&Task::Resume) {
            self.resume_axis(Some(axis_id));
            return true;
        }
        let task = process.get_task();
        if let Some(task) = task
            && (&**task as &dyn std::any::Any)
                .downcast_ref::<tomodataplots_param::Task>()
                .is_some()
        {
            self.tomodataplots(
                Some(&**task),
                Some(axis_id),
                Some(process_series),
                process.get_parameter(),
            );
            return true;
        }
        false
    }

    // Updates done

    /// Java package-private `updateDialog`.  Empty in the base class.
    fn update_dialog(&'static self, process_name: Option<ProcessName>, axis_id: Option<AxisID>) {
        let _ = (process_name, axis_id);
    }

    /// Java `logMessagePrimaryLog`.
    fn log_message_primary_log(&self, reader: Option<FileReaderRef>) {
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.log_message_primary_log(reader);
        }
    }

    /// Java `logMessage(File)`.
    fn log_message_file(&'static self, file: Option<&Path>) {
        self.log_message_private(file, false, None);
    }

    /// Java `logSimpleMessage(File, FileWriter)`.  Off the event dispatch thread the
    /// call is posted to it, where the log window lives (Java's `EtomoLogger` posts
    /// each append the same way).
    fn log_simple_message_file(
        &'static self,
        file: Option<&Path>,
        secondary_log: Option<FileWriterRef>,
    ) {
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let manager = self.this();
            let file = file.map(Path::to_path_buf);
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                manager.log_simple_message_file(file.as_deref(), secondary_log);
            });
            return;
        }
        self.log_message_private(file, true, secondary_log);
    }

    /// Java private `logMessage(File, boolean, FileWriter)`.
    fn log_message_private(
        &'static self,
        file: Option<&Path>,
        simple: bool,
        secondary_log: Option<FileWriterRef>,
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
        if let Some(log_interface) = log_interface {
            if !simple {
                log_interface.log_message_file_file_writer(Some(file), secondary_log);
            } else {
                log_interface.log_message_file_boolean_file_writer(
                    Some(file),
                    false,
                    secondary_log,
                );
            }
        } else {
            if *DEBUG {
                eprintln!(
                    "Logging from file: {}",
                    utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                );
            }
            match LogFile::get_instance_file(Some(file), Some(self.get_emergency_monitor(None))) {
                Err(LogFileError::Lock(_)) => {
                    // `catch (final LockException e) {}`
                }
                Err(e) => {
                    // `catch (final LogFileException | IOException e)`
                    eprintln!("{e}");
                    if *DEBUG {
                        eprintln!("Unable to log from file.  {}", e.get_message());
                    }
                }
                Ok(log_file) => match log_file.open_reader() {
                    Err(LogFileError::Lock(_)) => {
                        // `catch (final LockException e) {}`
                    }
                    Err(e) => {
                        eprintln!("{e}");
                        if *DEBUG {
                            eprintln!("Unable to log from file.  {}", e.get_message());
                        }
                    }
                    Ok(id) => {
                        if *DEBUG {
                            // Upstream bug fixed in translation (BaseManager.java:575): a
                            // null reader id throws a NullPointerException in `readLine`;
                            // here nothing is read.
                            if let Some(id) = id {
                                loop {
                                    match log_file.read_line(&id) {
                                        Ok(Some(line)) => eprintln!("{line}"),
                                        Ok(None) => break,
                                        Err(LogFileError::Lock(_)) => break,
                                        Err(e) => {
                                            eprintln!("{e}");
                                            if *DEBUG {
                                                eprintln!(
                                                    "Unable to log from file.  {}",
                                                    e.get_message()
                                                );
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            }
        }
    }

    /// Java `logSimpleMessage(String, FileWriter)`.  Log without extra stuff.  Off the
    /// event dispatch thread the call is posted to it, where the log window lives
    /// (Java's `EtomoLogger` posts each append the same way).
    fn log_simple_message(
        &'static self,
        message: Option<&str>,
        secondary_log: Option<FileWriterRef>,
    ) {
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let manager = self.this();
            let message = message.map(str::to_owned);
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                manager.log_simple_message(message.as_deref(), secondary_log);
            });
            return;
        }
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.log_message_string_boolean_boolean_file_writer(
                message,
                false,
                false,
                secondary_log,
            );
        } else if *DEBUG {
            eprintln!("{}", message.unwrap_or("null"));
        }
    }

    /// Java `logSimpleMessage(String, boolean)`.  Off the event dispatch
    /// thread the call is posted to it, as `logSimpleMessage(String,
    /// FileWriter)` is: the log window lives there, and `ProcessManager`
    /// calls this from a process's completion thread
    /// (`postProcess(String, AxisID)`, the alternate-stack message).
    fn log_simple_message_newline(&'static self, message: Option<&str>, newline: bool) {
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let manager = self.this();
            let message = message.map(str::to_owned);
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                manager.log_simple_message_newline(message.as_deref(), newline);
            });
            return;
        }
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface
                .log_message_string_boolean_boolean_file_writer(message, false, newline, None);
        } else if *DEBUG {
            eprintln!("{}", message.unwrap_or("null"));
        }
    }

    /// Java `logMessage(String)`.
    fn log_message(&self, message: Option<&str>) {
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.log_message_string(message);
        } else if *DEBUG {
            eprintln!("{}", message.unwrap_or("null"));
        }
    }

    /// Java `logMessage(Loggable, AxisID)`.
    fn log_message_loggable(&self, loggable: Option<&dyn Loggable>, axis_id: Option<AxisID>) {
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.log_message_loggable_axis_id(loggable, axis_id);
        } else {
            // Upstream bug fixed in translation (BaseManager.java:633): a null
            // `loggable` throws NullPointerException; here nothing is logged.
            let Some(loggable) = loggable else {
                return;
            };
            match loggable.get_log_message() {
                Ok(message_list) => {
                    if *DEBUG {
                        eprintln!("{}", utilities::get_date_time_stamp());
                        for message in &message_list {
                            eprintln!("{}", message.as_deref().unwrap_or("null"));
                        }
                    }
                }
                Err(LoggableException::Lock(_)) => {}
                Err(e) => {
                    // `catch (final LogFileException | IOException e)`
                    eprintln!("{:?}", e);
                }
            }
        }
    }

    /// Java `getProcessingMethodMediator`.  The mediators are Swing-side objects:
    /// callable on the thread that built the manager (the event dispatch thread).
    fn get_processing_method_mediator(
        &self,
        axis_id: Option<AxisID>,
    ) -> Option<Rc<ProcessingMethodMediator>> {
        // Off the thread that built the manager (a monitor thread) the Swing-side
        // mediator cannot be reached and reads as null.
        let mediator = if axis_id == Some(AxisID::Second) {
            &self.base().processing_method_mediator_b
        } else {
            &self.base().processing_method_mediator_a
        };
        if !mediator.is_owner_thread() {
            return None;
        }
        Some(Rc::clone(mediator.get()))
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
        if let Some(log_interface) = log_interface {
            return log_interface
                .log_message_string_axis_id_string_array_string(title, axis_id, message, msg_id);
        }
        let mut retval = false;
        if *DEBUG {
            eprintln!(
                "{}\n{} - {} axis:",
                utilities::get_date_time_stamp(),
                title.unwrap_or("null"),
                display_or_null(axis_id.as_ref())
            );
            // Upstream bug fixed in translation (BaseManager.java:665): a null `message`
            // throws a NullPointerException; here nothing is printed.
            let message = match message {
                None => return retval,
                Some(message) => message,
            };
            for item in message {
                if !retval && (msg_id.is_none() || item.contains(msg_id.unwrap())) {
                    retval = true;
                }
                eprintln!("{}", item);
            }
        }
        retval
    }

    /// Java `logMessageUntilWithKeyword`.  Returns true when `msgId` was found.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:699-704): a null `untilTag`,
    /// `tag` or `skipTag` throws a NullPointerException in `startsWith`/`indexOf`; here a
    /// null tag never matches.
    fn log_message_until_with_keyword(
        &'static self,
        file_type: Option<&FileType>,
        skip_tag: Option<&str>,
        tag: Option<&str>,
        until_tag: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let log_interface = self.get_log_interface();
        let mut log_file: Option<Arc<Handle>> = None;
        let mut id: Option<ReaderId> = None;
        let result = (|| -> Result<bool, LogFileError> {
            // Upstream bug fixed in translation (BaseManager.java:689): a null
            // `fileType` throws a NullPointerException; here nothing is logged.
            let Some(file_type) = file_type else {
                return Ok(false);
            };
            let handle = LogFile::get_instance_user_dir(
                &self.get_property_user_dir().unwrap_or_default(),
                &file_type
                    .get_file_name(Some(self.this()), axis_id)
                    .unwrap_or_default(),
                Some(self.get_emergency_monitor(axis_id)),
            )?;
            log_file = Some(Arc::clone(&handle));
            id = handle.open_reader()?;
            let Some(reader_id) = id.as_ref() else {
                return Ok(false);
            };
            if reader_id.is_empty() {
                return Ok(false);
            }
            let mut tag_found = false;
            let mut message_array: Vec<String> = Vec::new();
            let mut line = handle.read_line(reader_id)?;
            while let Some(current) = line.as_deref() {
                if until_tag.is_some_and(|until_tag| current.starts_with(until_tag)) {
                    break;
                }
                if tag.is_some_and(|tag| current.starts_with(tag)) {
                    tag_found = true;
                }
                if tag_found && !skip_tag.is_some_and(|skip_tag| current.contains(skip_tag)) {
                    message_array.push(current.to_string());
                }
                line = handle.read_line(reader_id)?;
            }
            if log_interface.is_some() && !message_array.is_empty() {
                log_interface
                    .as_ref()
                    .unwrap()
                    .log_message_axis_id_array_list(axis_id, Some(&message_array));
            } else if *DEBUG {
                eprintln!("{}", line.as_deref().unwrap_or("null"));
            }
            handle.close_id(id.as_ref().map(|id| &**id));
            Ok(true)
        })();
        match result {
            Ok(retval) => return retval,
            Err(LogFileError::LogFile(e)) => {
                eprintln!("{e:?}");
            }
            // `catch (final IOException | LockException e) {}`; the other `LogFile`
            // exceptions are `LogFileException` subclasses in the source.
            Err(LogFileError::Io(_)) | Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                eprintln!("{e}");
            }
        }
        if let Some(log_file) = &log_file {
            log_file.close_id(id.as_ref().map(|id| &**id));
        }
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
            // Upstream bug fixed in translation (BaseManager.java:739): a null `keyword`
            // throws a NullPointerException in `indexOf`; here it never matches.
            if keyword.is_some_and(|keyword| item.contains(keyword)) {
                if !logged_title {
                    logged_title = true;
                    if let Some(log_interface) = &log_interface {
                        log_interface.log_message_string_axis_id(title, axis_id);
                    } else if *DEBUG {
                        eprintln!(
                            "{}\n{} - {} axis:",
                            utilities::get_date_time_stamp(),
                            title.unwrap_or("null"),
                            display_or_null(axis_id.as_ref())
                        );
                    }
                }
                if let Some(log_interface) = &log_interface {
                    log_interface.log_message_string(Some(item));
                } else if *DEBUG {
                    eprintln!("{}", item);
                }
            }
        }
    }

    /// Java `logMessageWithKeyword(FileType, String, String, AxisID)`.
    fn log_message_with_keyword_file_type(
        &'static self,
        file_type: Option<&FileType>,
        keyword: Option<&str>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let log_interface = self.get_log_interface();
        let Some(file_type) = file_type else {
            return false;
        };
        let mut log_file: Option<Arc<Handle>> = None;
        let mut id: Option<ReaderId> = None;
        let result = (|| -> Result<bool, LogFileError> {
            let handle = LogFile::get_instance_user_dir(
                &self.get_property_user_dir().unwrap_or_default(),
                &file_type
                    .get_file_name(Some(self.this()), axis_id)
                    .unwrap_or_default(),
                Some(self.get_emergency_monitor(axis_id)),
            )?;
            log_file = Some(Arc::clone(&handle));
            id = handle.open_reader()?;
            let Some(reader_id) = id.as_ref() else {
                return Ok(false);
            };
            if reader_id.is_empty() {
                return Ok(false);
            }
            let mut logged_title = false;
            let mut line = handle.read_line(reader_id)?;
            while let Some(current) = line.as_deref() {
                // A null `keyword` never matches (the source throws a
                // NullPointerException in `indexOf`; see `log_message_with_keyword`).
                if keyword.is_some_and(|keyword| current.contains(keyword)) {
                    if !logged_title {
                        logged_title = true;
                        if log_interface.is_some() && title.is_some() {
                            log_interface
                                .as_ref()
                                .unwrap()
                                .log_message_string_axis_id(title, axis_id);
                        } else if *DEBUG {
                            eprintln!(
                                "{}\n{} - {} axis:",
                                utilities::get_date_time_stamp(),
                                title.unwrap_or("null"),
                                display_or_null(axis_id.as_ref())
                            );
                        }
                    }
                    if let Some(log_interface) = &log_interface {
                        log_interface.log_message_string(Some(current));
                    } else if *DEBUG {
                        eprintln!("{}", current);
                    }
                }
                line = handle.read_line(reader_id)?;
            }
            handle.close_id(id.as_ref().map(|id| &**id));
            Ok(true)
        })();
        match result {
            Ok(retval) => return retval,
            Err(LogFileError::Unlocked(e)) => {
                eprintln!("{e:?}");
            }
            // `catch (final LogFileException | IOException | LockException e) {}`
            Err(_) => {}
        }
        if let Some(log_file) = &log_file {
            log_file.close_id(id.as_ref().map(|id| &**id));
        }
        false
    }

    /// Java `updateDirectiveMap(DirectiveMap, StringBuffer)`, the overload that
    /// `ApplicationManager` and `DirectiveEditorManager` declare.
    ///
    /// Upstream bug fixed in translation (DirectiveEditorBuilder.java:149,
    /// DirectiveEditorManager.java:102): both callers hold the manager as a
    /// `BaseManager`, so Java resolves the call at compile time to
    /// `BaseManager.updateDirectiveMap(DirectiveMapInterface, ...)`, which is empty,
    /// and the dataset's setupset/runtime values never reach the directive editor (the
    /// `DirectiveMap` overloads have no other caller).  Here the call dispatches to
    /// the `DirectiveMap` overload, as the builder's documentation says it should;
    /// managers without one fall back to the base class's empty method.
    fn update_directive_map_directive_map(
        &'static self,
        directive_map: &crate::imod::etomo::storage::directive_map::DirectiveMap,
        errmsg: &mut String,
    ) {
        self.update_directive_map(Some(directive_map), errmsg);
    }

    /// Java `updateDirectiveMap`.  Empty in the base class.
    fn update_directive_map(
        &self,
        directive_map: Option<&dyn DirectiveMapInterface>,
        errmsg: &mut String,
    ) {
        let _ = (directive_map, errmsg);
    }

    /// Java `isAllowPrimaryLogging`.
    fn is_allow_primary_logging(&self) -> bool {
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            return log_interface.is_allow_primary_logging();
        }
        true
    }

    /// Java `setAllowPrimaryLogging`.
    fn set_allow_primary_logging(&self, input: bool) {
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.set_allow_primary_logging(input);
        }
    }

    /// Java `logMessage(ArrayList<String>, String, AxisID)`.
    fn log_message_list(
        &self,
        message: Option<&Vec<String>>,
        title: Option<&str>,
        axis_id: Option<AxisID>,
    ) {
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.log_message_string_axis_id_array_list(
                title,
                axis_id,
                message.map(|message| message.as_slice()),
            );
        } else if *DEBUG {
            eprintln!(
                "{}\n{} - {} axis:",
                utilities::get_date_time_stamp(),
                title.unwrap_or("null"),
                display_or_null(axis_id.as_ref())
            );
            // Upstream bug fixed in translation (BaseManager.java:839): a null `message`
            // throws a NullPointerException; here nothing more is printed.
            if let Some(message) = message {
                for item in message {
                    eprintln!("{}", item);
                }
            }
        }
    }

    /// Java `saveLog`.
    fn save_log(&'static self) {
        // The Java calls this from the `UtilityThread` (`EtomoDirector.saveLogs`);
        // the log window is an event-dispatch-thread object, so the call is
        // posted there.
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let this = self.this();
            crate::imod::etomo::util::event_queue::invoke_later(move || this.save_log());
            return;
        }
        let log_interface = self.get_log_interface();
        if let Some(log_interface) = log_interface {
            log_interface.save();
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

    /// Java `getFocusComponent`.  `java.awt.Component` is the Swing stand-in node.
    fn get_focus_component(&self) -> Option<Rc<JComponent>> {
        None
    }

    /// Java `setPropertyUserDir`.
    fn set_property_user_dir(&self, property_user_dir: Option<&str>) -> Option<String> {
        // avoid empty strings
        // Upstream bug fixed in translation (BaseManager.java:867): a null argument
        // throws a NullPointerException in `matches`; here it is kept as null.
        let property_user_dir = match property_user_dir {
            // Java `matches("\\s*")`: every character is Java whitespace (or none).
            Some(dir)
                if dir
                    .chars()
                    .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r')) =>
            {
                None
            }
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
        &'static self,
        data_file: Option<&Path>,
        axis_id: Option<AxisID>,
        loaded_from_a_different_file: bool,
    ) {
        if !*HEADLESS.lock().unwrap() {
            if let Some(data_file) = data_file {
                let loaded =
                    self.load_param_file(Some(data_file), axis_id, loaded_from_a_different_file);
                *self.base().loaded_param_file.lock().unwrap() = loaded;
            }
        }
        *self.base().initialized.lock().unwrap() = true;
    }

    /// Java package-private `initializeUIParameters(String, AxisID)`.
    fn initialize_ui_parameters_from_name(
        &'static self,
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
    fn save_storable(&'static self, axis_id: Option<AxisID>, storable: Option<&dyn Storable>) {
        let result = (|| -> Result<(), LogFileError> {
            let parameter_store = self.get_parameter_store(axis_id)?;
            let Some(parameter_store) = parameter_store else {
                return Ok(());
            };
            let mut parameter_store = parameter_store.lock().unwrap();
            parameter_store.set_auto_store(true);
            parameter_store.save(storable)?;
            Ok(())
        })();
        match result {
            Ok(()) | Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &format!(
                            "Unable to save or write to properties.  {}",
                            e.get_message()
                        ),
                        "Etomo Error",
                        axis_id,
                    )
                });
            }
        }
    }

    /// Java package-private `saveMetaDataToParameterStore`.
    fn save_meta_data_to_parameter_store(&'static self, axis_id: Option<AxisID>) -> bool {
        let result = (|| -> Result<bool, LogFileError> {
            let parameter_store = self.get_parameter_store(axis_id)?;
            let Some(parameter_store) = parameter_store else {
                return Ok(false);
            };
            parameter_store.lock().unwrap().save(
                self.get_base_meta_data()
                    .map(|meta_data| meta_data as &dyn Storable),
            )?;
            Ok(true)
        })();
        match result {
            Ok(false) => return false,
            Ok(true) | Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string(
                        Some(self.this()),
                        &format!("Cannot save or write to metaData.\n{}", e.get_message()),
                        "Etomo Error",
                    )
                });
            }
        }
        true
    }

    /// Java final `saveStorables`.  Save etomo to parametersState by asking the child
    /// manager for a list of storable objects.  This is used when storable objects may
    /// have been changes and nothing has been saved (update comscript functions).  This
    /// will often do unnecessary saves but it guarentees that everything will be saved.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:967): when `getParameterStore`
    /// throws before the store exists, the source reaches
    /// `parameterStore.setAutoStore(true)` with a null store and throws a
    /// NullPointerException; here the method ends there.
    fn save_storables(&'static self, axis_id: Option<AxisID>) {
        // `Ok(false)` is one of the source's `return`s out of the `try` block, which
        // leave the method; `Ok(true)` falls through to the code after the `catch`es.
        let result = (|| -> Result<bool, LogFileError> {
            let parameter_store = self.get_parameter_store(axis_id)?;
            let Some(parameter_store) = parameter_store else {
                return Ok(false);
            };
            parameter_store.lock().unwrap().set_auto_store(false);
            let storables = self.get_storables();
            let Some(storables) = storables else {
                return Ok(false);
            };
            for storable in &storables {
                parameter_store.lock().unwrap().save(storable.as_deref())?;
            }
            Ok(true)
        })();
        match result {
            Ok(false) => return,
            Ok(true) | Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &format!(
                            "Unable to save or write to properties.  {}",
                            e.get_message()
                        ),
                        "Etomo Error",
                        axis_id,
                    )
                });
            }
        }
        let parameter_store = self.base().parameter_store.lock().unwrap().clone();
        let Some(parameter_store) = parameter_store else {
            return;
        };
        parameter_store.lock().unwrap().set_auto_store(true);
        let store_result = parameter_store.lock().unwrap().store_properties();
        if let Err(e) = store_result {
            // `catch (final LogFileException | IOException e)`
            eprintln!("{e}");
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "Unable to save or write to {}.  {}",
                        self.base()
                            .param_file
                            .lock()
                            .unwrap()
                            .as_ref()
                            .map(|param_file| utilities::java_io_file_get_absolute_path(
                                &param_file.to_string_lossy()
                            ))
                            .unwrap_or("null".to_string()),
                        e
                    ),
                    "Etomo Error",
                    axis_id,
                )
            });
        }
    }

    /// Java `isNewDataset`.
    fn is_new_dataset(&self) -> bool {
        self.base().param_file.lock().unwrap().is_none()
    }

    /// Java private `getStorables()`.  Get the storable objects from the child and base
    /// manager.
    ///
    /// Java's `Storable[]` holds references of two lifetimes here: the child's storables
    /// are `&'static` (the manager's own objects), while slots 0 and 1 are the process
    /// manager's current `ProcessData` - the shared `Arc<Mutex<ProcessData>>` a running
    /// process thread may also hold, whose `Mutex` is its `Storable`.  A boxed
    /// `Storable` is either (`storage/storable.rs` forwards `&T` and `Arc<T>`).
    ///
    /// Upstream bug fixed in translation (BaseManager.java:995-996): a manager with no
    /// process manager throws a NullPointerException here; its two process data slots
    /// stay null instead (as `save` does, BaseManager.java:1011).
    fn get_storables(&self) -> Option<Vec<Option<Box<dyn Storable>>>> {
        let storables = self.get_storables_with_offset(2);
        let Some(storables) = storables else {
            // Manager does not have a data file.
            return None;
        };
        let mut storables: Vec<Option<Box<dyn Storable>>> = storables
            .into_iter()
            .map(|storable| storable.map(|storable| Box::new(storable) as Box<dyn Storable>))
            .collect();
        if let Some(process_manager) = self.get_process_manager() {
            storables[0] = Some(Box::new(process_manager.get_process_data(AxisID::First)));
            storables[1] = Some(Box::new(process_manager.get_process_data(AxisID::Second)));
        }
        Some(storables)
    }

    /// `BaseManager.save` itself, for the overrides' `super` call.
    /// Java package-private `save() throws LogFileException, IOException, LockException`.
    /// Save etomo to parameterStore by asking the child manager to save its state.  This
    /// is used when using the done functionality of the dialogs (file save and exit).
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1011): a manager with no
    /// process manager throws a NullPointerException; here the process data are not
    /// saved.
    fn save_super(&self) -> Result<bool, LogFileError> {
        let parameter_store = self.base().parameter_store.lock().unwrap().clone();
        let Some(parameter_store) = parameter_store else {
            return Ok(false);
        };
        let mut parameter_store = parameter_store.lock().unwrap();
        parameter_store.set_auto_store(false);
        if let Some(process_manager) = self.get_process_manager() {
            // The shared `ProcessData` is its `Mutex` (see `storage/storable.rs`).
            parameter_store.save(Some(&*process_manager.get_process_data(AxisID::First)))?;
            parameter_store.save(Some(&*process_manager.get_process_data(AxisID::Second)))?;
        }
        Ok(true)
    }

    /// Java package-private `save() throws LogFileException, IOException, LockException`.
    fn save(&'static self) -> Result<bool, LogFileError> {
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

    /// `BaseManager.saveParamFile` itself, for the overrides' `super` call.
    /// Java `saveParamFile() throws LogFileException, IOException, LockException`.  A
    /// message asking the ApplicationManager to save the parameter information to a
    /// file.
    fn save_param_file_super(&'static self) -> Result<bool, LogFileError> {
        if !self.is_setup_done() {
            return Ok(false);
        }
        self.set_param_file();
        if self.get_parameter_store(Some(AxisID::Only))?.is_none() {
            return Ok(false);
        }
        if !etomo_director::INSTANCE.is_memory_available() {
            return Ok(true);
        }
        self.save()?;
        let parameter_store = self.base().parameter_store.lock().unwrap().clone();
        if let Some(parameter_store) = parameter_store {
            let mut parameter_store = parameter_store.lock().unwrap();
            parameter_store.set_auto_store(true);
            parameter_store.store_properties()?;
        }
        self.save_storables(Some(AxisID::Only));
        // Update the MRU test data filename list
        let param_file = self.base().param_file.lock().unwrap().clone();
        etomo_director::INSTANCE.with_user_configuration_mut(|user_config| {
            user_config.put_data_file(
                param_file
                    .as_ref()
                    .map(|param_file| {
                        utilities::java_io_file_get_absolute_path(&param_file.to_string_lossy())
                    })
                    .as_deref(),
            )
        });
        // A null MRU entry (an unused slot) is shown as the empty label that hides it;
        // the source's `EtomoMenu.setMRUFileLabels` would throw a NullPointerException.
        let mru_file_list: Vec<String> = etomo_director::INSTANCE
            .with_user_configuration_mut(|user_config| user_config.get_mru_file_list())
            .into_iter()
            .map(|file| file.unwrap_or_default())
            .collect();
        ui_harness::INSTANCE.with(|ui_harness| ui_harness.set_mru_file_labels(&mru_file_list));
        // Reset the process track flag, if it exists
        let process_track = self.get_process_track();
        if let Some(process_track) = process_track {
            process_track.reset_modified();
        }
        Ok(true)
    }

    /// Java `saveParamFile() throws LogFileException, IOException, LockException`.
    fn save_param_file(&'static self) -> Result<bool, LogFileError> {
        self.save_param_file_super()
    }

    /// Java `getParameterStore`.  Creates parameterStore if it doesn't already exist.
    /// Return null if paramFile is null.
    fn get_parameter_store(
        &'static self,
        axis_id: Option<AxisID>,
    ) -> Result<Option<Arc<Mutex<ParameterStore>>>, LogFileError> {
        // `synchronized (paramFile)`: the field's own lock is held for the block.
        let param_file = self.base().param_file.lock().unwrap();
        let Some(param_file) = param_file.as_ref() else {
            return Ok(None);
        };
        let mut parameter_store = self.base().parameter_store.lock().unwrap();
        if parameter_store.is_some() {
            return Ok(parameter_store.clone());
        }
        *parameter_store = ParameterStore::get_instance_manager(
            Some(self.this()),
            axis_id,
            Some(param_file.clone()),
        )?
        .map(|store| Arc::new(Mutex::new(store)));
        Ok(parameter_store.clone())
    }

    /// Java package-private `endThreads`.
    fn end_threads(&self) {
        self.get_imod_manager().stop_request_handler();
        let mediator = self.get_processing_method_mediator(Some(AxisID::First));
        if let Some(mediator) = mediator {
            mediator.msg_exiting();
        }
        let mediator = self.get_processing_method_mediator(Some(AxisID::Second));
        if let Some(mediator) = mediator {
            mediator.msg_exiting();
        }
        let parameter_store = self.base().parameter_store.lock().unwrap().clone();
        if let Some(parameter_store) = parameter_store {
            parameter_store.lock().unwrap().set_auto_store(false);
        }
    }

    /// Java `progressBarDone`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1100): a null main panel
    /// (headless) throws a NullPointerException; here nothing happens.
    fn progress_bar_done(
        &'static self,
        axis_id: Option<AxisID>,
        process_end_state: Option<ProcessEndState>,
    ) {
        // The Java monitors call this from their own threads; the main panel
        // is an event-dispatch-thread object, so the call is posted there.
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let this = self.this();
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                this.progress_bar_done(axis_id, process_end_state)
            });
            return;
        }
        if let Some(main_panel) = self.get_main_panel() {
            main_panel
                .main_panel()
                .stop_progress_bar_axis_id_process_end_state(
                    axis_id.unwrap_or(AxisID::Only),
                    process_end_state,
                );
        }
    }

    /// Java private `checkNextProcess`.
    fn check_next_process(&'static self, axis_id: Option<AxisID>) -> bool {
        let process_manager = self.get_process_manager();
        if let Some(process_manager) = process_manager {
            let axis_process_data = self.get_axis_process_data();
            let process_a = axis_process_data.get_thread(AxisID::First);
            let process_b = axis_process_data.get_thread(AxisID::Second);
            // Check to see if next processes have to be done
            // (Java allocates an unused `ArrayList messageArray` here.)
            let mut dropped_process_a = false;
            let mut dropped_process_b = false;
            if let Some(process_a) = &process_a
                && let Some(process_series_a) = process_a.get_process_series()
            {
                dropped_process_a = process_series_a
                    .get()
                    .borrow()
                    .will_process_be_dropped(Some(
                        &*process_manager
                            .get_process_data(AxisID::First)
                            .lock()
                            .unwrap(),
                    ));
            }
            if let Some(process_b) = &process_b
                && let Some(process_series_b) = process_b.get_process_series()
            {
                dropped_process_b = process_series_b
                    .get()
                    .borrow()
                    .will_process_be_dropped(Some(
                        &*process_manager
                            .get_process_data(AxisID::Second)
                            .lock()
                            .unwrap(),
                    ));
            }
            if dropped_process_a || dropped_process_b {
                let mut message = String::from(
                    "WARNING!!!\nIf you exit now then not all processes will complete.",
                );
                if *self.base().exiting.lock().unwrap() {
                    message.push_str("Do you still wish to exit the program?");
                } else {
                    message.push_str("Do you still wish to close this interface?");
                }
                if !ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_yes_no_warning_dialog(Some(self.this()), &message, axis_id)
                }) {
                    *self.base().exiting.lock().unwrap() = false;
                    return false;
                }
            }
            if !self.check_unidentified_process(Some(AxisID::First))
                || !self.check_unidentified_process(Some(AxisID::Second))
            {
                *self.base().exiting.lock().unwrap() = false;
                return false;
            }
        }
        true
    }

    /// Java `renameImageFile(FileType, FileType, AxisID) throws IOException,
    /// LockException, LogFileException`.  Renames an image file.  Pops up an error
    /// message and returns if the from file doesn't exist.  If either the from file type
    /// or the to file type have imod manager key(s), calls closeImod to tell the user to
    /// close the file(s) before renaming them.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1170-1178): with a null main
    /// panel (headless) the three progress-bar calls throw a NullPointerException; here
    /// they are skipped.
    fn rename_image_file(
        &'static self,
        from_file_type: Option<&FileType>,
        to_file_type: Option<&FileType>,
        axis_id: Option<AxisID>,
    ) -> Result<(), LogFileError> {
        let (Some(from_file_type), Some(to_file_type)) = (from_file_type, to_file_type) else {
            return Ok(());
        };
        if !from_file_type
            .get_file(Some(self.this()), axis_id)
            .is_some_and(|file| file.exists())
        {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "Unable to rename file.  {} doesn't exist.",
                        from_file_type
                            .get_file(Some(self.this()), axis_id)
                            .and_then(|file| file
                                .file_name()
                                .map(|name| name.to_string_lossy().into_owned()))
                            .unwrap_or("null".to_string())
                    ),
                    "Entry Error",
                    axis_id,
                )
            });
            return Ok(());
        }
        self.close_imod_file_key(Some(&**from_file_type), axis_id, true);
        self.close_imod_file_key(Some(&**to_file_type), axis_id, true);
        let from_file = from_file_type.get_file(Some(self.this()), axis_id);
        let to_file = to_file_type.get_file(Some(self.this()), axis_id);
        let axis = axis_id.unwrap_or(AxisID::Only);
        if let Some(main_panel) = self.get_main_panel() {
            main_panel
                .main_panel()
                .set_progress_bar_value_int_standard_bar_string_file_file_axis_id(
                    0,
                    Some(StandardBarString::Renaming),
                    from_file.as_deref(),
                    to_file.as_deref(),
                    axis,
                );
        }
        match utilities::rename_file(
            Some(self.this()),
            axis_id,
            from_file.as_deref(),
            to_file.as_deref(),
            false,
            false,
            false,
        ) {
            Ok(_) => {
                if let Some(main_panel) = self.get_main_panel() {
                    main_panel
                        .main_panel()
                        .set_progress_bar_value_int_standard_bar_string_file_file_boolean_boolean_axis_id(
                            0,
                            Some(StandardBarString::Renaming),
                            from_file.as_deref(),
                            to_file.as_deref(),
                            true,
                            false,
                            axis,
                        );
                }
                Ok(())
            }
            // `catch (final IOException e)`: mark the bar failed and rethrow.
            Err(e @ LogFileError::Io(_)) => {
                if let Some(main_panel) = self.get_main_panel() {
                    main_panel
                        .main_panel()
                        .set_progress_bar_value_int_standard_bar_string_file_file_boolean_boolean_axis_id(
                            0,
                            Some(StandardBarString::Renaming),
                            from_file.as_deref(),
                            to_file.as_deref(),
                            false,
                            true,
                            axis,
                        );
                }
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Java package-private `renameImageFile(FileKey, File, FileType, AxisID, boolean)
    /// throws IOException, LockException, LogFileException`.  Renames an image file
    /// where the from file type doesn't contain the file name.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1200): a null `fromFile`
    /// throws a NullPointerException; here it is treated as a file that does not exist.
    fn rename_image_file_from_key(
        &'static self,
        from_file_key: Option<&FileKey>,
        from_file: Option<&Path>,
        to_file_type: Option<&FileType>,
        axis_id: Option<AxisID>,
        use_file_name_in_close: bool,
    ) -> Result<(), LogFileError> {
        let (Some(from_file_key), Some(to_file_type)) = (from_file_key, to_file_type) else {
            return Ok(());
        };
        if !from_file.is_some_and(|file| file.exists()) {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "Unable to rename file.  {} doesn't exist.",
                        from_file
                            .and_then(|file| file.file_name())
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or("null".to_string())
                    ),
                    "Entry Error",
                    axis_id,
                )
            });
            return Ok(());
        }
        if use_file_name_in_close {
            self.close_imod_file_key_and_file(Some(from_file_key), from_file, axis_id, true);
        } else {
            self.close_imod_file_key(Some(from_file_key), axis_id, true);
        }
        self.close_imod_file_key(Some(&**to_file_type), axis_id, true);
        utilities::rename_file(
            Some(self.this()),
            axis_id,
            from_file,
            to_file_type.get_file(Some(self.this()), axis_id).as_deref(),
            false,
            false,
            false,
        )?;
        Ok(())
    }

    /// Java `backupImageFile(FileType, AxisID) throws IOException, LogFileException,
    /// LockException`.  Renames an image file to image_file_name~.  Does nothing if the
    /// file doesn't exist.  If the file type has imod manager key(s), calls closeImod to
    /// tell the user to close the file before renaming it.
    fn backup_image_file(
        &'static self,
        file_type: Option<&FileType>,
        axis_id: Option<AxisID>,
    ) -> Result<(), LogFileError> {
        let Some(file_type) = file_type else {
            return Ok(());
        };
        if !file_type
            .get_file(Some(self.this()), axis_id)
            .is_some_and(|file| file.exists())
        {
            return Ok(());
        }
        self.close_imod_file_key(Some(&**file_type), axis_id, true);
        utilities::backup_file(file_type.get_file(Some(self.this()), axis_id).as_deref())
    }

    /// Java `closeStaleFile(FileKey, AxisID)`.  Asks to close a stale file.
    fn close_stale_file(&'static self, file_key: Option<FileKey>, axis_id: Option<AxisID>) {
        self.close_imod_file_key(file_key.as_ref(), axis_id, true);
    }

    /// Java `closeStaleFile(FileType, AxisID)`.  Deprecated 6/18/19.
    fn close_stale_file_from_file_type(
        &'static self,
        file_type: Option<&FileType>,
        axis_id: Option<AxisID>,
    ) {
        self.close_imod_file_key(file_type.map(|file_type| &**file_type), axis_id, true);
    }

    /// Java `closeImod(FileKey, AxisID, boolean)`.  Ask to close all 3dmods associated
    /// with this file type.  A file type may have multiple 3dmod keys.  `warnOnce` -
    /// when true, only one warning is give for the open 3dmod instance.
    fn close_imod_file_key(
        &'static self,
        file_key: Option<&FileKey>,
        axis_id: Option<AxisID>,
        warn_once: bool,
    ) {
        let Some(file_key) = file_key else {
            return;
        };
        self.close_imod(
            file_key.get_imod_manager_key(),
            axis_id,
            file_key
                .get_file_name(Some(self.this()), axis_id)
                .as_deref(),
            warn_once,
        );
        self.close_imod(
            file_key.get_imod_manager_key2(),
            axis_id,
            file_key
                .get_file_name(Some(self.this()), axis_id)
                .as_deref(),
            warn_once,
        );
    }

    /// Java private `closeImod(FileKey, File, AxisID, boolean)`.  Ask to close all 3dmods
    /// associated with this file type and file.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1286): a null `file` throws a
    /// NullPointerException in `getName`; here the file name is null.
    fn close_imod_file_key_and_file(
        &'static self,
        file_key: Option<&FileKey>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        warn_once: bool,
    ) {
        let Some(file_key) = file_key else {
            return;
        };
        let file_name = file
            .and_then(|file| file.file_name())
            .map(|name| name.to_string_lossy().into_owned());
        self.close_imod_with_file_name(
            file_key.get_imod_manager_key(),
            file_name.as_deref(),
            axis_id,
            file_key.get_description().as_deref(),
            warn_once,
        );
        self.close_imod_with_file_name(
            file_key.get_imod_manager_key2(),
            file_name.as_deref(),
            axis_id,
            file_key.get_imod_manager_key2(),
            warn_once,
        );
    }

    /// Java `closeImod(String, File, AxisID, String, boolean)`.
    fn close_imod_key_file(
        &'static self,
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
        &'static self,
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

    /// Java package-private `closeImod(FileKey, String, AxisID, boolean)`.  Ask to close
    /// all 3dmods associated with this file type and file.
    fn close_imod_file_key_and_name(
        &'static self,
        file_key: Option<&FileKey>,
        file_name: Option<&str>,
        axis_id: Option<AxisID>,
        warn_once: bool,
    ) {
        let Some(file_key) = file_key else {
            return;
        };
        self.close_imod_with_file_name(
            file_key.get_imod_manager_key(),
            file_name,
            axis_id,
            file_key.get_description().as_deref(),
            warn_once,
        );
        self.close_imod_with_file_name(
            file_key.get_imod_manager_key2(),
            file_name,
            axis_id,
            file_key.get_imod_manager_key2(),
            warn_once,
        );
    }

    /// Java `closeImod(String, AxisID, String, boolean)`.
    fn close_imod(
        &'static self,
        key: Option<&str>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        warn_once: bool,
    ) -> bool {
        self.close_imod_with_message(key, axis_id, description, None, warn_once)
    }

    /// Java `closeImods`.  Returns a group of up to three files; true if files where
    /// closed or files did not need to be closed.
    #[allow(clippy::too_many_arguments)]
    fn close_imods(
        &'static self,
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
                display_or_null(axis_id.as_ref()),
                message.unwrap_or("null")
            );
        }
        if key1.is_none() && key2.is_none() && key3.is_none() {
            return true;
        }
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<bool, ImodManagerException> {
            if (key1.is_some() && imod_manager.is_open_string_axis_id(key1.unwrap(), axis_id)?)
                || (key2.is_some()
                    && imod_manager.is_open_string_axis_id(key2.unwrap(), axis_id)?)
                || (key3.is_some()
                    && imod_manager.is_open_string_axis_id(key3.unwrap(), axis_id)?)
            {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    let descr = descr.unwrap_or("Files");
                    if ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_yes_no_dialog_base_manager_string_axis_id(
                            None,
                            &format!(
                                "{}{}{}\n\n{}",
                                descr,
                                message.unwrap_or("null"),
                                self.get_file_lock_message(Some("\n\n")),
                                question.unwrap_or("null")
                            ),
                            axis_id,
                        )
                    }) {
                        if let Some(key1) = key1 {
                            imod_manager.quit_string_axis_id(key1, axis_id)?;
                        }
                        if let Some(key2) = key2 {
                            imod_manager.quit_string_axis_id(key2, axis_id)?;
                        }
                        if let Some(key3) = key3 {
                            imod_manager.quit_string_axis_id(key3, axis_id)?;
                        }
                        self.release_file();
                        return Ok(true);
                    }
                } else {
                    if let Some(key1) = key1 {
                        imod_manager.quit_string_axis_id(key1, axis_id)?;
                    }
                    if let Some(key2) = key2 {
                        imod_manager.quit_string_axis_id(key2, axis_id)?;
                    }
                    if let Some(key3) = key3 {
                        imod_manager.quit_string_axis_id(key3, axis_id)?;
                    }
                    self.release_file();
                    return Ok(true);
                }
            } else {
                return Ok(true);
            }
            Ok(false)
        })();
        match result {
            Ok(retval) => return retval,
            Err(e) => {
                {
                    // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                    // each arm prints the stack trace and opens a dialog with its own title.
                    // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                    // not caught by the source; it is printed here and the call fails.
                    eprintln!("{e}");
                    let title = match &e {
                        ImodManagerException::AxisType(_) => Some("AxisType problem"),
                        ImodManagerException::Io(_) => Some("IO Exception"),
                        ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                        ImodManagerException::Runtime(_) => None,
                    };
                    if let Some(title) = title {
                        ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(self.this()),
                                &e.to_string(),
                                title,
                                axis_id,
                            )
                        });
                    }
                }
            }
        }
        false
    }

    /// Java `closeImod(String, AxisID, String, String, boolean)`.  Close the 3dmod
    /// instance denoted by key and axisID if either the --autoclose3dmod param was
    /// passed to etomo, or the user wants the 3dmod to be closed.  If stale is true then
    /// the popup can only be display once for this instance of 3dmod.  Returns true if
    /// file was closed or file did not need to be closed.
    fn close_imod_with_message(
        &'static self,
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
                display_or_null(axis_id.as_ref()),
                description.unwrap_or("null"),
                message.unwrap_or("null")
            );
        }
        let Some(key) = key else {
            return true;
        };
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<bool, ImodManagerException> {
            if imod_manager.is_open_string_axis_id(key, axis_id)? {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    if !warn_once || (warn_once && imod_manager.warn_stale_file(key, axis_id)?) {
                        if ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_yes_no_dialog_base_manager_string_axis_id(
                                None,
                                &format!(
                                    "{} is open in 3dmod.{}{}{}\n\nShould file be closed?",
                                    description.unwrap_or("null"),
                                    if warn_once {
                                        "  This 3dmod instance will display an out of date version of this file.  "
                                    } else {
                                        ""
                                    },
                                    match message {
                                        None => "".to_string(),
                                        Some(message) => format!("  {message}  "),
                                    },
                                    self.get_file_lock_message(Some("\n\n"))
                                ),
                                axis_id,
                            )
                        }) {
                            imod_manager.quit_string_axis_id(key, axis_id)?;
                            self.release_file();
                            return Ok(true);
                        }
                    }
                } else {
                    imod_manager.quit_string_axis_id(key, axis_id)?;
                    self.release_file();
                    return Ok(true);
                }
            } else {
                return Ok(true);
            }
            Ok(false)
        })();
        match result {
            Ok(retval) => return retval,
            Err(e) => {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            axis_id,
                        )
                    });
                }
            }
        }
        false
    }

    /// Java private `closeImod(String, String, AxisID, String, boolean)`.  Close the
    /// 3dmod instance denoted by key and axisID if either the --autoclose3dmod param was
    /// passed to etomo, or the user wants the 3dmod to be closed.
    fn close_imod_with_file_name(
        &'static self,
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
                display_or_null(axis_id.as_ref()),
                description.unwrap_or("null"),
                stale
            );
        }
        let Some(key) = key else {
            return;
        };
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerException> {
            if imod_manager.is_open_string_axis_id_string(
                Some(key),
                axis_id,
                file_name.unwrap_or_default(),
            )? {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    if !stale || (stale && imod_manager.warn_stale_file(key, axis_id)?) {
                        let message = vec![
                            format!(
                                "{} {} is open in 3dmod.{}{}",
                                description.unwrap_or("null"),
                                file_name.unwrap_or("null"),
                                if stale {
                                    "  This 3dmod instance will display an out of date version of this file.  "
                                } else {
                                    ""
                                },
                                self.get_file_lock_message(Some("\n\n"))
                            ),
                            "Should it be closed?".to_string(),
                        ];
                        if ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                                None, &message, axis_id,
                            )
                        }) {
                            imod_manager.quit_string_axis_id_string(
                                key,
                                axis_id,
                                file_name.unwrap_or_default(),
                            )?;
                            self.release_file();
                        }
                    }
                } else {
                    imod_manager.quit_string_axis_id_string(
                        key,
                        axis_id,
                        file_name.unwrap_or_default(),
                    )?;
                    self.release_file();
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            axis_id,
                        )
                    });
                }
            }
        }
    }

    /// Java private `closeImodWithKeyAndFile`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1537, :1553): a null `file`
    /// throws a NullPointerException in `getName`; here the name is null.
    fn close_imod_with_key_and_file(
        &'static self,
        key: Option<&str>,
        file: Option<&Path>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
        warn_once: bool,
        file_move: bool,
    ) {
        let file_name = file
            .and_then(|file| file.file_name())
            .map(|name| name.to_string_lossy().into_owned());
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImod:key:{},fileName:{},axisID:{},description:{},warnOnce:{}",
                key.unwrap_or("null"),
                file_name.as_deref().unwrap_or("null"),
                display_or_null(axis_id.as_ref()),
                description.unwrap_or("null"),
                warn_once
            );
        }
        let Some(key) = key else {
            return;
        };
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerException> {
            if imod_manager.is_open_string_axis_id_file(Some(key), axis_id, file)? {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    if !warn_once || (warn_once && imod_manager.warn_stale_file(key, axis_id)?) {
                        // Use the first part of the message with and without the custom
                        // message.
                        let mut builder = String::new();
                        builder.push_str(description.unwrap_or("null"));
                        builder.push(' ');
                        builder.push_str(file_name.as_deref().unwrap_or("null"));
                        builder.push_str(" is open in 3dmod.");
                        let message_0 = if !file_move {
                            format!(
                                "{}{}{}",
                                builder,
                                if warn_once {
                                    "  This 3dmod instance will display an out of date version of this file.  "
                                } else {
                                    ""
                                },
                                self.get_file_lock_message(Some("\n\n"))
                            )
                        } else {
                            builder.push_str(
                                "\n\nPlease close this file so it can be moved to a new location.",
                            );
                            builder.clone()
                        };
                        let message = vec![message_0, "\nShould it be closed?".to_string()];
                        if ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                                None, &message, axis_id,
                            )
                        }) {
                            imod_manager.quit_string_axis_id_file(key, axis_id, file)?;
                            self.release_file();
                        }
                    }
                } else {
                    imod_manager.quit_string_axis_id_file(key, axis_id, file)?;
                    self.release_file();
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            axis_id,
                        )
                    });
                }
            }
        }
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
    fn close3dmods(&'static self, axis_id: Option<AxisID>) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() && *DEBUG {
            eprintln!("close3dmods:axisID:{}", display_or_null(axis_id.as_ref()));
        }
        // Should we close the 3dmod windows
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerException> {
            if imod_manager.is_open()? {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    // Java `new String[3]` with `message[2]` never set (null).
                    let message = vec![
                        "There are still 3dmod programs running.".to_string(),
                        "Do you wish to end these programs?".to_string(),
                    ];
                    if ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                            None, &message, axis_id,
                        )
                    }) {
                        imod_manager.quit()?;
                        self.release_file();
                    }
                } else {
                    imod_manager.quit()?;
                    self.release_file();
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(e @ ImodManagerException::AxisType(_)) => {
                {
                    // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                    // each arm prints the stack trace and opens a dialog with its own title.
                    // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                    // not caught by the source; it is printed here and the call fails.
                    eprintln!("{e}");
                    let title = match &e {
                        ImodManagerException::AxisType(_) => Some("AxisType problem"),
                        ImodManagerException::Io(_) => Some("IO Exception"),
                        ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                        ImodManagerException::Runtime(_) => None,
                    };
                    if let Some(title) = title {
                        ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(self.this()),
                                &e.to_string(),
                                title,
                                axis_id,
                            )
                        });
                    }
                }
            }
            // `catch (IOException e)` and `catch (SystemProcessException e)`: stack trace
            // only.
            Err(e) => eprintln!("{e}"),
        }
    }

    /// Java private `disconnect3dmods`.
    fn disconnect3dmods(&self) {
        // `catch (Throwable e) { e.printStackTrace(); }`
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.get_imod_manager().disconnect();
        })) {
            eprintln!("{e:?}");
        }
    }

    /// Java package-private `close`.
    fn close(&'static self, axis_id: Option<AxisID>) -> bool {
        if !self.check_next_process(axis_id) {
            return false;
        }
        self.close3dmods(axis_id);
        self.disconnect3dmods();
        true
    }

    /// `BaseManager.exitProgram` itself, for the overrides' `super` call.
    /// Java package-private `exitProgram`.  Exit the program.  To guarantee that etomo
    /// can always exit, catch all unrecognized Exceptions and Errors and return true.
    fn exit_program_super(&'static self, axis_id: Option<AxisID>) -> bool {
        *self.base().exiting.lock().unwrap() = true;
        // `catch (Throwable e) { e.printStackTrace(); }`
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Check for processes that will die if etomo exits
            let process_manager = self.get_process_manager();
            if process_manager.is_some() {
                let axis_process_data = self.get_axis_process_data();
                let process_a = axis_process_data.get_thread(AxisID::First);
                let process_b = axis_process_data.get_thread(AxisID::Second);
                let nohup_a = process_a.as_ref().is_none_or(|process| process.is_nohup());
                let nohup_b = process_b.as_ref().is_none_or(|process| process.is_nohup());
                if !nohup_a || !nohup_b {
                    if !ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_yes_no_warning_dialog(
                            Some(self.this()),
                            "WARNING!!\nThere is process running which will stop if Etomo exits.\nDo you still wish to exit the program?",
                            axis_id,
                        )
                    }) {
                        *self.base().exiting.lock().unwrap() = false;
                        return Some(false);
                    }
                }
            }
            if !self.check_next_process(axis_id) {
                return Some(false);
            }
            self.close3dmods(axis_id);
            imodqtassist_process::INSTANCE.quit();
            None
        }));
        match result {
            Ok(Some(retval)) => return retval,
            Ok(None) => {}
            Err(e) => eprintln!("{e:?}"),
        }
        // Do this even if everything else fails
        self.disconnect3dmods();
        true
    }

    /// Java package-private `exitProgram`.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        self.exit_program_super(axis_id)
    }

    /// Java private `checkUnidentifiedProcess`.
    fn check_unidentified_process(&'static self, axis_id: Option<AxisID>) -> bool {
        let thread = self
            .get_axis_process_data()
            .get_thread(axis_id.unwrap_or(AxisID::Only));
        let Some(thread) = thread else {
            return true;
        };
        let process_data = thread.get_process_data();
        if process_data
            .as_ref()
            .is_some_and(|process_data| process_data.lock().unwrap().is_empty())
        {
            if !ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_yes_no_warning_dialog(
                    Some(self.this()),
                    "There currently is an unidentified process running.\nPlease wait a few seconds while it is identified.\n\nExit without waiting?",
                    axis_id,
                )
            }) {
                return false;
            }
        }
        true
    }

    /// Java `isDualAxis`.  Check if the current data set is a dual axis data set.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1735): a null
    /// `getBaseMetaData()` throws a NullPointerException; here it counts as dual axis
    /// (not `SINGLE_AXIS`).
    fn is_dual_axis(&self) -> bool {
        if self
            .get_base_meta_data()
            .map(|meta_data| meta_data.base().get_axis_type())
            == Some(AxisType::SingleAxis)
        {
            false
        } else {
            true
        }
    }

    /// Java `isExiting`.
    fn is_exiting(&self) -> bool {
        *self.base().exiting.lock().unwrap()
    }

    /// Java `imodGetRubberbandCoordinates`.
    fn imod_get_rubberband_coordinates(
        &'static self,
        imod_key: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Option<Vec<String>> {
        let mut results = None;
        // Upstream bug fixed in translation: a null key throws a NullPointerException in
        // `BaseImodManager`; here there are no results.
        let Some(imod_key) = imod_key else {
            return results;
        };
        match self.get_imod_manager().get_rubberband_coordinates(imod_key) {
            Ok(coordinates) => results = coordinates,
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                // This arm names `AxisID.ONLY`, not `axisID`.
                {
                    // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                    // each arm prints the stack trace and opens a dialog with its own title.
                    // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                    // not caught by the source; it is printed here and the call fails.
                    eprintln!("{e}");
                    let title = match &e {
                        ImodManagerException::AxisType(_) => Some("AxisType problem"),
                        ImodManagerException::Io(_) => Some("IO Exception"),
                        ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                        ImodManagerException::Runtime(_) => None,
                    };
                    if let Some(title) = title {
                        ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                                Some(self.this()),
                                &e.to_string(),
                                title,
                                Some(AxisID::Only),
                            )
                        });
                    }
                }
            }
            Err(e) => {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            axis_id,
                        )
                    });
                }
            }
        }
        results
    }

    /// Java package-private `setPanel`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:1771): a null main panel
    /// (headless) throws a NullPointerException; here the panel calls are skipped.
    fn set_panel(&'static self) {
        ui_harness::INSTANCE.with(|ui_harness| ui_harness.pack_base_manager(Some(self.this())));
        // Resize to the users preferrred window dimensions
        // Swing layout: getMainPanel().setSize(new Dimension(
        // userConfig.getMainWindowWidth(), userConfig.getMainWindowHeight())).
        let _ = etomo_director::INSTANCE.with_user_configuration(|user_config| {
            (
                user_config.get_main_window_width(),
                user_config.get_main_window_height(),
            )
        });
        ui_harness::INSTANCE.with(|ui_harness| ui_harness.do_layout(Some(self.this())));
        ui_harness::INSTANCE.with(|ui_harness| ui_harness.validate(Some(self.this())));
        if self.is_dual_axis() {
            if let Some(main_panel) = self.get_main_panel() {
                main_panel.main_panel().set_divider_location(0.51);
            }
        }
    }

    // get functions

    /// Java `imodOpen(String, int, File, int, Run3dmodMenuOptions)`.  Open or raise a
    /// specific 3dmod to view a file with binning.  Or open a new 3dmod.  Return the
    /// index of the 3dmod opened or raised.
    fn imod_open_with_binning(
        &'static self,
        imod_key: Option<&str>,
        mut imod_index: i32,
        file: Option<&Path>,
        binning: i32,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> i32 {
        let imod_manager = self.get_imod_manager();
        let key = imod_key.unwrap_or_default();
        let result = (|| -> Result<(), ImodManagerException> {
            if imod_index == -1 {
                imod_index = imod_manager.new_imod_string_file(key, file)?;
            } else {
                imod_manager.update_imod(key, imod_index, file)?;
            }
            imod_manager.set_binning_xy_string_int_int(key, imod_index, binning)?;
            imod_manager.open_string_int_run3dmod_menu_options(key, imod_index, menu_options)?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &e.to_string(),
                        &format!(
                            "Can't open {} 3dmod with imodIndex={}",
                            imod_key.unwrap_or("null"),
                            imod_index
                        ),
                        Some(AxisID::Only),
                    )
                });
            }
            Err(e) => {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            Some(AxisID::Only),
                        )
                    });
                }
            }
        }
        imod_index
    }

    /// Java `imodOpen(String, int, String, String, Run3dmodMenuOptions)`.  Open or raise
    /// a specific 3dmod to view a file with a model.  Or open a new 3dmod.  Return the
    /// index of the 3dmod opened or raised.
    fn imod_open_with_model(
        &'static self,
        imod_key: Option<&str>,
        mut imod_index: i32,
        absolute_file_path: Option<&str>,
        absolute_model_path: Option<&str>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) -> i32 {
        // Upstream bug fixed in translation (BaseManager.java:1831): `new File(null)`
        // throws a NullPointerException; here a null path is a null file.
        let file = absolute_file_path.map(PathBuf::from);
        let imod_manager = self.get_imod_manager();
        let key = imod_key.unwrap_or_default();
        let result = (|| -> Result<(), ImodManagerException> {
            if imod_index == -1 {
                imod_index = imod_manager.new_imod_string_file(key, file.as_deref())?;
            } else {
                imod_manager.update_imod(key, imod_index, file.as_deref())?;
            }
            imod_manager.open_string_int_string_boolean_run3dmod_menu_options(
                key,
                imod_index,
                absolute_model_path,
                true,
                menu_options,
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &e.to_string(),
                        &format!(
                            "Can't open {} 3dmod with imodIndex={}",
                            imod_key.unwrap_or("null"),
                            imod_index
                        ),
                        Some(AxisID::Only),
                    )
                });
            }
            Err(e) => {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            Some(AxisID::Only),
                        )
                    });
                }
            }
        }
        imod_index
    }

    /// Java `imodOpen(String, Run3dmodMenuOptions)`.  Open 3dmod.
    fn imod_open(&'static self, imod_key: Option<&str>, menu_options: Option<Run3dmodMenuOptions>) {
        let result = self
            .get_imod_manager()
            .open_string_run3dmod_menu_options(imod_key.unwrap_or_default(), menu_options);
        match result {
            Ok(()) => {}
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &e.to_string(),
                        &format!("Can't open {} in 3dmod ", imod_key.unwrap_or("null")),
                        Some(AxisID::Only),
                    )
                });
            }
            Err(e) => {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            Some(AxisID::Only),
                        )
                    });
                }
            }
        }
    }

    /// Java `imodOpen(AxisID, String, String, Run3dmodMenuOptions, boolean)`.
    fn imod_open_axis(
        &'static self,
        axis_id: Option<AxisID>,
        imod_key: Option<&str>,
        model: Option<&str>,
        menu_options: Option<Run3dmodMenuOptions>,
        model_mode: bool,
    ) {
        let result = self
            .get_imod_manager()
            .open_string_axis_id_string_boolean_run3dmod_menu_options(
                imod_key.unwrap_or_default(),
                axis_id,
                model,
                model_mode,
                menu_options,
            );
        match result {
            Ok(()) => {}
            Err(e @ ImodManagerException::SystemProcess(_)) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &e.to_string(),
                        &format!("Can't open {} in 3dmod ", imod_key.unwrap_or("null")),
                        axis_id,
                    )
                });
            }
            Err(e) => {
                // `catch (AxisTypeException | IOException | SystemProcessException e)`:
                // each arm prints the stack trace and opens a dialog with its own title.
                // An unchecked exception (`IllegalArgumentException` for an unknown key) is
                // not caught by the source; it is printed here and the call fails.
                eprintln!("{e}");
                let title = match &e {
                    ImodManagerException::AxisType(_) => Some("AxisType problem"),
                    ImodManagerException::Io(_) => Some("IO Exception"),
                    ImodManagerException::SystemProcess(_) => Some("System Process Exception"),
                    ImodManagerException::Runtime(_) => None,
                };
                if let Some(title) = title {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &e.to_string(),
                            title,
                            axis_id,
                        )
                    });
                }
            }
        }
    }

    /// Java `getParamFile`.  Return the parameter file as a File object.
    fn get_param_file(&self) -> Option<PathBuf> {
        self.base().param_file.lock().unwrap().clone()
    }

    /// Java package-private `loadParamFile`.  Loads storables, sets the param file, and
    /// sets up the ImodManager.  Loads the meta data object first, and then loads all
    /// the storables.  Set loadedFromADIfferentFile to true if duplicating or extracting
    /// data from another param file to create this project.  In this case storables will
    /// not be loaded.
    fn load_param_file(
        &'static self,
        param_file: Option<&Path>,
        axis_id: Option<AxisID>,
        loaded_from_a_different_file: bool,
    ) -> bool {
        // Upstream bug fixed in translation (BaseManager.java:1931): a null `paramFile`
        // throws a NullPointerException; here nothing is loaded.
        let Some(param_file) = param_file else {
            return false;
        };
        // Set the current working directory for the application, this is the
        // path to the EDF or EJF file. The working directory is defined by the
        // current
        // user.dir system property.
        // Uggh, stupid JAVA bug, getParent() only returns the parent if the File
        // was created with the full path
        let param_file = PathBuf::from(utilities::java_io_file_get_absolute_path(
            &param_file.to_string_lossy(),
        ));
        // Upstream bug fixed in translation (BaseManager.java:1933): a root path has a
        // null parent and `endsWith` throws a NullPointerException; here the parent is
        // the empty string.
        let param_file_parent = param_file
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned())
            .unwrap_or_default();
        if param_file_parent.ends_with(' ') {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "The directory, {param_file_parent}, cannot be used because it ends with a space."
                    ),
                    "Unusable Directory Name",
                    Some(AxisID::Only),
                )
            });
            return false;
        }
        *self.base().property_user_dir.lock().unwrap() = Some(param_file_parent);
        let mut invalid_reason = String::new();
        if !utilities::is_valid_file(
            Some(param_file.as_path()),
            Some("Parameter file"),
            &mut invalid_reason,
            true,
            true,
            true,
            false,
        ) {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &invalid_reason,
                    "File Error",
                    axis_id,
                )
            });
            return false;
        }
        *self.base().param_file.lock().unwrap() = Some(param_file.clone());
        if !loaded_from_a_different_file {
            // Read in the test parameter data file
            let result = (|| -> Result<bool, LogFileError> {
                let parameter_store = self.get_parameter_store(axis_id)?;
                let Some(parameter_store) = parameter_store else {
                    return Ok(false);
                };
                // must load meta data before other storables can be constructed
                //
                // Upstream bug fixed in translation (BaseManager.java:1956,1960):
                // `ParameterStore.load` calls `storable.load(properties)` on whatever it
                // is given, so a manager with no meta data, or a null slot in the
                // storable array (FrontPageManager.getStorables fills one of its three
                // slots; ApplicationManager's are all set), throws a
                // NullPointerException.  Here a null storable is skipped, as
                // `ParameterStore.save` skips one.
                if let Some(base_meta_data) = self.get_base_meta_data() {
                    parameter_store.lock().unwrap().load(base_meta_data);
                }
                let storables = self.get_storables();
                if let Some(storables) = storables {
                    for storable in &storables {
                        if let Some(storable) = storable.as_deref() {
                            parameter_store.lock().unwrap().load(storable);
                        }
                    }
                }
                Ok(true)
            })();
            match result {
                Ok(true) => {}
                Ok(false) => return false,
                Err(LogFileError::Lock(_)) => return false,
                // `catch (final LogFile.FileException | IOException except)`
                Err(except) => {
                    eprintln!("{except}");
                    let error_message = vec![
                        "Test parameter file read error".to_string(),
                        "Could not find the test parameter data file:".to_string(),
                        except.get_message(),
                    ];
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self.this()),
                            &error_message,
                            "Etomo Error",
                            axis_id,
                        )
                    });
                    return false;
                }
            }
        }
        // Update the MRU test data filename list
        etomo_director::INSTANCE.with_user_configuration_mut(|user_config| {
            user_config.put_data_file(Some(&utilities::java_io_file_get_absolute_path(
                &param_file.to_string_lossy(),
            )))
        });
        true
    }

    /// Java package-private `backupFile`.
    fn backup_file(&'static self, file: Option<&Path>, axis_id: Option<AxisID>) -> bool {
        if let Some(file) = file
            && file.exists()
        {
            let absolute_path = utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
            let backup_file = PathBuf::from(format!("{absolute_path}~"));
            match utilities::rename_file(
                Some(self.this()),
                axis_id,
                Some(file),
                Some(&backup_file),
                false,
                false,
                false,
            ) {
                Ok(_) => {}
                Err(LogFileError::Lock(_)) => return false,
                // `catch (final IOException | LogFileException except)`
                Err(except) => {
                    eprintln!(
                        "Unable to backup file: {} to {}",
                        absolute_path,
                        utilities::java_io_file_get_absolute_path(&backup_file.to_string_lossy())
                    );
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.this()),
                            &except.get_message(),
                            "File Rename Error (9)",
                            axis_id,
                        )
                    });
                    return false;
                }
            }
        }
        true
    }

    /// Java final `processDone(String, int, ProcessName, AxisID, ProcessEndState,
    /// boolean, ProcessResultDisplay, ProcessSeries, boolean)`.  Stop progress bar and
    /// start next process.
    #[allow(clippy::too_many_arguments)]
    fn process_done_end_state(
        &'static self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<ProcessName>,
        axis_id: Option<AxisID>,
        end_state: Option<ProcessEndState>,
        failed: bool,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        non_blocking: bool,
    ) {
        if *self.base().debug.lock().unwrap() {
            eprintln!(
                "BaseProcessManager.processDone:exitValue:{exit_value},processName:{},endState:{}",
                display_or_null(process_name.as_ref()),
                display_or_null(end_state.as_ref())
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

    /// Java final `processDone(String, int, ProcessName, AxisID, boolean,
    /// ProcessEndState, boolean, ProcessResultDisplay, ProcessSeries, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn process_done_force(
        &'static self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<ProcessName>,
        axis_id: Option<AxisID>,
        force_next_process: bool,
        end_state: Option<ProcessEndState>,
        failed: bool,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
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

    /// Java final `processDone(String, int, ProcessName, AxisID, boolean,
    /// ProcessEndState, String, boolean, ProcessResultDisplay, ProcessSeries, boolean)`.
    /// Notification message that a background process is done.  Runs on the event
    /// dispatch thread (`BaseProcessManager` posts it there).
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2046, :2053, :2060): a null
    /// `threadName` throws a NullPointerException in `equals`, and a manager with no
    /// process manager throws one in `unblockAxis`; here a null name matches neither
    /// axis and the unblock is skipped without a process manager.
    #[allow(clippy::too_many_arguments)]
    fn process_done(
        &'static self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<ProcessName>,
        axis_id: Option<AxisID>,
        force_next_process: bool,
        end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
        failed: bool,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        non_blocking: bool,
    ) {
        if *self.base().debug.lock().unwrap() {
            eprintln!(
                "BaseManager.processDone:exitValue:{exit_value},processName:{},endState:{},processSeries:{}",
                display_or_null(process_name.as_ref()),
                display_or_null(end_state.as_ref()),
                if process_series.is_some() {
                    "set"
                } else {
                    "null"
                }
            );
        }
        let main_panel = self.get_main_panel();
        let thread_name_a = self.base().thread_name_a.lock().unwrap().clone();
        let thread_name_b = self.base().thread_name_b.lock().unwrap().clone();
        if thread_name == Some(thread_name_a.as_str()) {
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state_string(
                    AxisID::First,
                    end_state,
                    status_string,
                );
            }
            *self.base().thread_name_a.lock().unwrap() = NO_PROCESS_THREAD_NAME.to_string();
            *self.base().background_process_a.lock().unwrap() = false;
            *self.base().background_process_name_a.lock().unwrap() = None;
            if let Some(process_manager) = self.get_process_manager() {
                process_manager.unblock_axis(AxisID::First);
            }
        } else if thread_name == Some(thread_name_b.as_str()) {
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state_string(
                    AxisID::Second,
                    end_state,
                    status_string,
                );
            }
            *self.base().thread_name_b.lock().unwrap() = NO_PROCESS_THREAD_NAME.to_string();
            if let Some(process_manager) = self.get_process_manager() {
                process_manager.unblock_axis(AxisID::Second);
            }
        } else if !non_blocking {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "Unknown thread finished!!!\nThread name: {}",
                        thread_name.unwrap_or("null")
                    ),
                    "Unknown Thread",
                    axis_id,
                )
            });
        }
        let mut parallel_panel = None;
        if let Some(main_panel) = &main_panel {
            parallel_panel = main_panel
                .main_panel()
                .get_parallel_panel(axis_id.unwrap_or(AxisID::Only));
        }
        if let Some(parallel_panel) = &parallel_panel {
            parallel_panel.msg_process_done();
        }
        let axis = axis_id.unwrap_or(AxisID::Only);
        self.update_dialog(process_name, axis_id);
        self.set_pause_process(axis_id, end_state, process_series.as_ref());
        // Try to start the next process if the process succeeded, or if
        // forceNextProcess is true (unless the user killed the process, it makes
        // the
        // nextProcess execute even when the current process failed).
        // If the process is
        if end_state != Some(ProcessEndState::Killed) && (exit_value == 0 || force_next_process) {
            if process_series.as_ref().is_none_or(|series| {
                !ProcessSeries::start_next_process_display(
                    series.get(),
                    axis,
                    process_result_display.clone(),
                )
            }) {
                self.send_msg_process_succeeded(process_result_display.as_ref());
                self.process_series_succeeded(axis_id, process_name);
            }
        } else if end_state == Some(ProcessEndState::Paused) {
            if process_series.as_ref().is_none_or(|series| {
                !ProcessSeries::start_pause_process(
                    series.get(),
                    axis,
                    process_result_display.clone(),
                )
            }) {
                self.send_msg_process_succeeded(process_result_display.as_ref());
                self.process_series_succeeded(axis_id, process_name);
            }
        } else {
            // ProcessSeries gets thrown away after it fails or the processes are used
            // up, so the processes don't have to be cleared as they did when next
            // processes where managed by BaseManager.
            if failed {
                self.send_msg_process_failed(process_result_display.as_ref());
                if let Some(series) = &process_series {
                    ProcessSeries::start_fail_process_display(
                        series.get(),
                        axis,
                        process_result_display.clone(),
                    );
                }
            } else if end_state == Some(ProcessEndState::Killed)
                && let Some(series) = &process_series
            {
                series
                    .get()
                    .borrow_mut()
                    .kill_series(axis, process_result_display.clone());
            }
        }
        self.send_event(axis_id, process_name, end_state, failed);
        self.base().busy_status_mediator.msg_process_done(axis);
    }

    /// Java `sendEvent`.  Empty in the base class.
    fn send_event(
        &self,
        axis_id: Option<AxisID>,
        process_name: Option<ProcessName>,
        process_end_state: Option<ProcessEndState>,
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

    /// Java `isValid`.
    fn is_valid(&self) -> bool {
        true
    }

    /// Java private `setReconnectRun`.  Should remail private.
    fn set_reconnect_run(&self, axis_id: Option<AxisID>) {
        if axis_id == Some(AxisID::Second) {
            *self.base().reconnect_run_b.lock().unwrap() = true;
        } else {
            *self.base().reconnect_run_a.lock().unwrap() = true;
        }
    }

    /// Java `saveAll`.  Save param file and open dialogs.  `errmsg` - should not be
    /// null.  Returns a timestamp or null if this functionality is not implemented.
    fn save_all(&'static self, errmsg: &mut String) -> Option<String> {
        let _ = errmsg;
        None
    }

    /// `BaseManager.doAutomation` itself, for the overrides' `super` call.
    fn do_automation_super(&self, local_arguments: Option<&LocalArguments>) {
        let _ = local_arguments;
        if etomo_director::ARGUMENTS.lock().unwrap().is_exit() {
            ui_harness::INSTANCE.with(|ui_harness| ui_harness.exit(Some(AxisID::Only), 0));
        }
    }

    /// Java `doAutomation`.
    fn do_automation(&self, local_arguments: Option<&LocalArguments>) {
        self.do_automation_super(local_arguments)
    }

    /// Java package-private `reconnectToDifferentHost`.  The manager default is that it
    /// cannot connect to a different host.
    fn reconnect_to_different_host(
        &'static self,
        process_data: Option<&Arc<Mutex<ProcessData>>>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let Some(process_data) = process_data else {
            return false;
        };
        let (host_name, is_running, is_ssh_failed, process_name) = {
            let mut process_data = process_data.lock().unwrap();
            (
                process_data.get_host_name(),
                process_data.is_running(),
                process_data.is_ssh_failed(),
                display_or_null(process_data.get_process_name().as_ref()),
            )
        };
        if is_running {
            // Handles the case where ssh hostname ps finds the pid of this process.
            if ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_yes_no_dialog_base_manager_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "WARNING:  Cannot connect to {process_name}.  The process is running on {host_name}.  Please exit Etomo, run xhost {host_name}, ssh to {host_name}, and run etomo in order to connect to this process.  Exit Etomo Y/N?"
                    ),
                    axis_id,
                )
            }) {
                // Exit from etomo.
                ui_harness::INSTANCE.with(|ui_harness| ui_harness.exit(Some(AxisID::Only), 0));
            }
        } else if is_ssh_failed {
            // Handles the case where the ssh fails.
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &format!(
                        "WARNING:  Cannot connect to {process_name}.  This process may be running on {host_name}.  Unable to connect to {host_name} to find out.  If {process_name} is still running on {host_name}, please exit Etomo, run xhost {host_name}, ssh to {host_name}, and run etomo in order to connect to this process."
                    ),
                    "Reconnect Warning",
                    axis_id,
                )
            });
        }
        false
    }

    /// Java `reconnect`.  IMPORTANT: Must turn off the blocking in BaseProcessManager
    /// when it is first run.  If it doesn't then no process that blocks an axis can
    /// run.  Attempts to reconnect to a currently running process.  Only run once per
    /// axis.  Only attempts one reconnect.  Returns true if a reconnect was attempted.
    /// Throws a RuntimeException if any Throwable is caught.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2216-2244): each
    /// `getProcessManager().unblockAxis(axisID)` throws a NullPointerException for a
    /// manager with no process manager (which the source then wraps and rethrows); here
    /// the unblock is skipped.
    fn reconnect(
        &'static self,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<MessagesArray>,
    ) -> bool {
        self.reconnect_super(process_data, axis_id, multi_line_messages, messages_array)
    }

    /// The base class's `reconnect` body (`super.reconnect(...)` in an override,
    /// `BatchRunTomoManager`).
    fn reconnect_super(
        &'static self,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<MessagesArray>,
    ) -> bool {
        let axis = axis_id.unwrap_or(AxisID::Only);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<Option<bool>, LogFileError> {
                if self.is_reconnect_run(axis_id) {
                    // Just in case
                    if let Some(process_manager) = self.get_process_manager() {
                        process_manager.unblock_axis(axis);
                    }
                    return Ok(Some(false));
                }
                self.set_reconnect_run(axis_id);
                let Some(process_data) = &process_data else {
                    if let Some(process_manager) = self.get_process_manager() {
                        process_manager.unblock_axis(axis);
                    }
                    return Ok(Some(false));
                };
                let (process_name, is_on_different_host) = {
                    let data = process_data.lock().unwrap();
                    (data.get_process_name(), data.is_on_different_host())
                };
                if process_name == Some(ProcessName::PROCESSCHUNKS) {
                    if is_on_different_host
                        && !self.reconnect_to_different_host(Some(process_data), axis_id)
                    {
                        if let Some(process_manager) = self.get_process_manager() {
                            process_manager.unblock_axis(axis);
                        }
                        return Ok(Some(false));
                    }
                    let is_running = process_data.lock().unwrap().is_running();
                    if is_on_different_host || is_running {
                        eprintln!(
                            "\nAttempting to reconnect in Axis {}\n{}",
                            display_or_null(axis_id.as_ref()),
                            process_data.lock().unwrap().to_source_string()
                        );
                        if !self.reconnect_processchunks(
                            Some(Arc::clone(process_data)),
                            axis_id,
                            multi_line_messages,
                            messages_array.clone(),
                        )? {
                            eprintln!(
                                "\nReconnect in Axis{} failed",
                                display_or_null(axis_id.as_ref())
                            );
                        }
                        if let Some(process_manager) = self.get_process_manager() {
                            process_manager.unblock_axis(axis);
                        }
                        return Ok(Some(true));
                    }
                }
                Ok(None)
            },
        ));
        match result {
            Ok(Ok(Some(retval))) => return retval,
            Ok(Ok(None)) => {}
            // `catch (Throwable t) { t.printStackTrace(); unblockAxis; throw new
            // RuntimeException(t); }`: the source deliberately escalates.
            Ok(Err(t)) => {
                eprintln!("{t}");
                if let Some(process_manager) = self.get_process_manager() {
                    process_manager.unblock_axis(axis);
                }
                panic!("{t}");
            }
            Err(t) => {
                eprintln!("{t:?}");
                if let Some(process_manager) = self.get_process_manager() {
                    process_manager.unblock_axis(axis);
                }
                std::panic::resume_unwind(t);
            }
        }
        let process_manager = self.get_process_manager();
        if let Some(process_manager) = process_manager {
            process_manager.unblock_axis(axis);
        }
        false
    }

    /// Java `reconnectProcesschunks(ProcessData, AxisID, boolean, List<ProcessMessages>)
    /// throws LockException`.
    fn reconnect_processchunks(
        &'static self,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: Option<AxisID>,
        multi_line_messages: bool,
        messages_array: Option<MessagesArray>,
    ) -> Result<bool, LogFileError> {
        // Upstream bug fixed in translation (BaseManager.java:2261): a null
        // `processData` throws a NullPointerException; here nothing is reconnected.
        let Some(process_data) = process_data else {
            return Ok(false);
        };
        let axis = axis_id.unwrap_or(AxisID::Only);
        let mut display: Option<ProcessResultDisplayHandle> = None;
        let factory = self.get_process_result_display_factory_interface(axis_id);
        if let Some(factory) = factory {
            let (display_id, factory_id) = {
                let data = process_data.lock().unwrap();
                (data.get_display_id(), data.get_factory_id())
            };
            display = factory
                .get_process_result_display(display_id, factory_id.as_deref().unwrap_or("null"));
        }
        let display_ref: Option<ProcessResultDisplayRef> = display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        if display_ref.is_some() {
            self.send_msg_process_starting(display_ref.as_ref());
        }
        let main_panel = self.get_main_panel();
        // FIXME - null pointer getPArallelPanel
        let _ = main_panel;
        let (last_process, dialog_type, process_name) = {
            let data = process_data.lock().unwrap();
            (
                data.get_last_process(),
                data.get_dialog_type(),
                data.get_process_name(),
            )
        };
        let process_series = ProcessSeries::new(
            self.this(),
            axis,
            dialog_type,
            Some("reconnectProcesschunks"),
        );
        if let Some(last_process) = &last_process {
            process_series
                .borrow_mut()
                .set_last_process(Some(last_process.as_str()));
        }
        // Fixed in translation (BaseManager.java:2275): a manager without a
        // process manager throws a NullPointerException; nothing is reconnected.
        let Some(process_manager) = self.get_process_manager() else {
            return Ok(false);
        };
        let ret = process_manager.reconnect_processchunks(
            axis,
            process_data,
            display_ref,
            Some(Arc::new(EdtRef::new(process_series))),
            multi_line_messages,
            self.is_popup_chunk_warnings(),
            messages_array,
        )?;
        self.set_thread_name(Some(&display_or_null(process_name.as_ref())), axis_id);
        Ok(ret)
    }

    /// Java package-private `isPopupChunkWarnings`.
    fn is_popup_chunk_warnings(&self) -> bool {
        true
    }

    /// Java final `tomodataplots`.
    fn tomodataplots(
        &'static self,
        task: Option<&dyn TaskInterface>,
        axis_id: Option<AxisID>,
        process_series: Option<&ProcessSeriesHandle>,
        alternative_input_file_absolute_path: Option<&str>,
    ) {
        if self.can_run_tomodataplots(task, axis_id) {
            let mut param = TomodataplotsParam::new();
            param.set_task(task);
            param.set_alternative_input_file_absolute_path(alternative_input_file_absolute_path);
            let process_manager = self.get_process_manager();
            if let Some(process_manager) = process_manager {
                let axis = axis_id.unwrap_or(AxisID::Only);
                process_manager.tomodataplots(param.get_command_array(self.this(), axis), axis);
            }
        }
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process_display(process_series, AxisID::Only, None);
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

    /// Java final `processchunks`.  Run processchunks.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2320, :2331, :2336): a null
    /// main panel (headless), a null `param`, a null `getBaseMetaData()` or a null
    /// process manager throws a NullPointerException; here a null main panel is treated
    /// as a null parallel panel, and the others end the call as a failure to start.
    // TODO
    #[allow(clippy::too_many_arguments)]
    fn processchunks(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
        dialog_type: Option<DialogType>,
        run_type: Option<RunType>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
        messages_array: Option<MessagesArray>,
    ) -> bool {
        let axis = axis_id.unwrap_or(AxisID::Only);
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self.this(), axis, dialog_type, Some("processchunks")),
        };
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(axis));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    axis_id,
                )
            });
            if let Some(process_result_display) = &process_result_display {
                process_result_display.msg_process_failed_to_start();
            }
            return false;
        };
        let (Some(param), Some(meta_data)) = (param, self.get_base_meta_data()) else {
            if let Some(process_result_display) = &process_result_display {
                process_result_display.msg_process_failed_to_start();
            }
            return false;
        };
        meta_data
            .base()
            .set_current_processchunks_root_name(axis_id, param.get_root_name().as_deref());
        meta_data
            .base()
            .set_current_processchunks_subdir_name(axis_id, param.get_subdir_name().as_deref());
        self.save_storable(axis_id, Some(meta_data as &dyn Storable));
        let display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        let started: Option<Result<String, AxisBusyException>> =
            self.get_process_manager().map(|process_manager| {
                process_manager.processchunks(
                    axis,
                    Arc::clone(&param),
                    &*parallel_panel.get_parallel_progress_display(),
                    display_ref,
                    Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
                    popup_chunk_warnings,
                    processing_method,
                    multi_line_messages,
                    run_type,
                    managed_process_data,
                    messages_array,
                )
            });
        let thread_name = match started {
            None => {
                if let Some(process_result_display) = &process_result_display {
                    process_result_display.msg_process_failed_to_start();
                }
                return false;
            }
            Some(Err(e)) => {
                eprintln!("{}", e.0);
                let message = vec![
                    format!("Can not execute {}", ProcessName::PROCESSCHUNKS),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self.this()),
                        &message,
                        "Unable to execute command",
                        axis_id,
                    )
                });
                if let Some(process_result_display) = &process_result_display {
                    process_result_display.msg_process_failed_to_start();
                }
                return false;
            }
            Some(Ok(thread_name)) => thread_name,
        };
        // set param in parallel panel so it can do a resume
        parallel_panel.set_process_info(Some(param), process_result_display);
        self.set_thread_name(Some(&thread_name), axis_id);
        true
    }

    /// Java final package-private `processDone(AxisID, ProcessResultDisplay,
    /// ConstProcessSeries)`.  This is a process done function for processes which are
    /// completed while the original manager function waits and do not use the process
    /// manager.  It always assumes success because the secondary process won't run if
    /// the proceeding process failed.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2376): a null main panel
    /// (headless) throws a NullPointerException; here there is no parallel panel.
    fn process_done_secondary(
        &self,
        axis_id: Option<AxisID>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<&ProcessSeriesHandle>,
    ) {
        let axis = axis_id.unwrap_or(AxisID::Only);
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(axis));
        if let Some(parallel_panel) = parallel_panel {
            parallel_panel.msg_process_done();
        }
        if process_series.is_none_or(|series| {
            !ProcessSeries::start_next_process_display(series, axis, process_result_display.clone())
        }) {
            self.send_msg_process_succeeded(process_result_display.as_ref());
        }
        self.base().busy_status_mediator.msg_process_done(axis);
        if process_series.is_none() {
            utilities::timestamp_marker(Some("processDone"));
        }
    }

    /// Java `sendMsgProcessStarting`.
    fn send_msg_process_starting(&self, process_result_display: Option<&ProcessResultDisplayRef>) {
        let Some(process_result_display) = process_result_display else {
            return;
        };
        process_result_display.get().msg_process_starting();
    }

    /// Java package-private `sendMsgProcessFailedToStart`.
    fn send_msg_process_failed_to_start(
        &self,
        process_result_display: Option<&ProcessResultDisplayRef>,
    ) {
        let Some(process_result_display) = process_result_display else {
            return;
        };
        process_result_display.get().msg_process_failed_to_start();
    }

    /// Java package-private `sendMsgProcessSucceeded`.
    fn send_msg_process_succeeded(&self, process_result_display: Option<&ProcessResultDisplayRef>) {
        let Some(process_result_display) = process_result_display else {
            return;
        };
        process_result_display.get().msg_process_succeeded();
    }

    /// Java package-private `sendMsgProcessFailed`.
    fn send_msg_process_failed(&self, process_result_display: Option<&ProcessResultDisplayRef>) {
        let Some(process_result_display) = process_result_display else {
            return;
        };
        process_result_display.get().msg_process_failed();
    }

    /// Java `setCurrentDialogType`.  Set the current dialog type.  This function is
    /// called from open functions and from showBlankPRocess().  It allows Etomo to call
    /// the done function when the user switches to another dialog.  Returns the action
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
                AxisID::Second,
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

    /// `BaseManager.setDebug` itself, for the overrides' `super` call.
    fn set_debug_super(&self, debug: bool) {
        *self.base().debug.lock().unwrap() = debug;
    }

    /// Java `setDebug`.
    fn set_debug(&self, debug: bool) {
        self.set_debug_super(debug)
    }

    /// Java final `startProgressBar`.  Start generic progress bar.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2467): a null main panel
    /// (headless) throws a NullPointerException; here nothing happens.
    fn start_progress_bar(
        &'static self,
        label: Option<&str>,
        axis_id: Option<AxisID>,
        process_name: Option<ProcessName>,
    ) {
        // The Java monitors call this from their own threads
        // (`CombineProcessMonitor.initializeProgressBar`/`startProgressBar`);
        // the main panel is an event-dispatch-thread object, so the call is
        // posted there, as `progressBarDone` does.
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let this = self.this();
            let label = label.map(str::to_owned);
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                this.start_progress_bar(label.as_deref(), axis_id, process_name)
            });
            return;
        }
        if let Some(main_panel) = self.get_main_panel() {
            main_panel
                .main_panel()
                .start_progress_bar_string_axis_id_process_name(
                    label,
                    axis_id.unwrap_or(AxisID::Only),
                    process_name.as_ref(),
                );
        }
    }

    /// Java final `stopProgressBar`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2471): a null main panel
    /// (headless) throws a NullPointerException; here nothing happens.
    fn stop_progress_bar(&'static self, axis_id: Option<AxisID>) {
        // Posted to the event dispatch thread from another thread, as
        // `startProgressBar`.
        if !crate::imod::etomo::util::event_queue::is_dispatch_thread() {
            let this = self.this();
            crate::imod::etomo::util::event_queue::invoke_later(move || {
                this.stop_progress_bar(axis_id)
            });
            return;
        }
        if let Some(main_panel) = self.get_main_panel() {
            main_panel
                .main_panel()
                .stop_progress_bar_axis_id(axis_id.unwrap_or(AxisID::Only));
        }
    }

    /// Java final `startLoad`.
    fn start_load(
        &self,
        param: Option<Arc<dyn IntermittentCommand>>,
        monitor: Option<Arc<dyn LoadMonitor>>,
    ) {
        // `getProcessManager().startLoad(param, monitor)`; a manager without one
        // (Java NullPointerException) does nothing.
        if let Some(process_manager) = self.get_process_manager() {
            process_manager.start_load(param, monitor);
        }
    }

    /// Java final `endLoad`.
    fn end_load(
        &self,
        param: Option<Arc<dyn IntermittentCommand>>,
        monitor: Option<Arc<dyn LoadMonitor>>,
    ) {
        // `getProcessManager().endLoad(param, monitor)`; a manager without one
        // (Java NullPointerException) does nothing.
        if let Some(process_manager) = self.get_process_manager() {
            process_manager.end_load(param, monitor);
        }
    }

    /// Java final `stopLoad`.
    fn stop_load(
        &self,
        param: Option<Arc<dyn IntermittentCommand>>,
        monitor: Option<Arc<dyn LoadMonitor>>,
    ) {
        // `getProcessManager().stopLoad(param, monitor)`; a manager without one
        // (Java NullPointerException) does nothing.
        if let Some(process_manager) = self.get_process_manager() {
            process_manager.stop_load(param, monitor);
        }
    }

    /// Java final `msgCurrentManagerChanged`.  Called when the manager's interface is
    /// either displayed or hidden.  `current` - true when interface is displayed.
    fn msg_current_manager_changed(&self, current: bool) {
        if current {
            self.make_property_user_dir_local();
        }
        if let Some(log_window) = self.get_log_window() {
            log_window.msg_current_manager_changed(current, self.is_startup_popup_open());
        }
    }

    /// Java `isStartupPopupOpen`.  Startup popup dialogs are used to set the dataset
    /// name and location in some interfaces.
    fn is_startup_popup_open(&self) -> bool {
        // Most interfaces don't use a startup popup.
        false
    }

    /// Java final `makePropertyUserDirLocal`.
    fn make_property_user_dir_local(&self) {
        // make the manager's directory the local directory
        let property_user_dir = self.base().property_user_dir.lock().unwrap().clone();
        match property_user_dir {
            None => {
                etomo_director::INSTANCE.make_original_dir_local();
            }
            Some(property_user_dir) => {
                // `System.setProperty("user.dir", propertyUserDir)`.  The JVM's `user.dir`
                // is a process-wide property that does not change the working directory;
                // `PWD` is the environment variable this translation reads for it.
                unsafe { std::env::set_var("PWD", property_user_dir) };
            }
        }
    }

    /// Java final `savePreferences`.
    fn save_preferences(&'static self, axis_id: Option<AxisID>, storable: Option<&dyn Storable>) {
        let Some(storable) = storable else {
            return;
        };
        // Java reads `getMainPanel()` into an unused local here.
        let result = (|| -> Result<(), LogFileError> {
            let mut local_parameter_store = etomo_director::INSTANCE.get_parameter_store();
            // Upstream bug fixed in translation (BaseManager.java:2526): a director
            // without a preference store throws a NullPointerException; here nothing
            // is saved.
            if let Some(local_parameter_store) = local_parameter_store.as_mut() {
                local_parameter_store.save(Some(storable))?;
            }
            Ok(())
        })();
        match result {
            Ok(()) | Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.this()),
                        &format!("Unable to save preferences.\n{}", e.get_message()),
                        "Etomo Error",
                        axis_id,
                    )
                });
            }
        }
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

    /// Java final `resetCurrentProcesschunks`.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2576): a null
    /// `getBaseMetaData()` throws a NullPointerException; here nothing is reset.
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
        end_state: Option<ProcessEndState>,
        process_series: Option<&ProcessSeriesRef>,
    ) {
        let resume_data = if axis_id == Some(AxisID::Second) {
            &self.base().resume_data_b
        } else {
            &self.base().resume_data_a
        };
        if let Some(process_series) = process_series
            && end_state == Some(ProcessEndState::Paused)
            && !resume_data.is_null()
        {
            process_series
                .get()
                .borrow_mut()
                .set_pause_process(Rc::new(Task::Resume));
        }
    }

    /// Java private `saveResume`.
    #[allow(clippy::too_many_arguments)]
    fn save_resume(
        &self,
        axis_id: Option<AxisID>,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
    ) {
        let axis_process_data = self.get_axis_process_data();
        let axis = axis_id.unwrap_or(AxisID::Only);
        if axis_process_data.is_pausing(axis) {
            let resume_data = if axis_id == Some(AxisID::Second) {
                &self.base().resume_data_b
            } else {
                &self.base().resume_data_a
            };
            resume_data.set(
                param,
                process_result_display
                    .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
                process_series.map(|series| Arc::new(EdtRef::new(series))),
                popup_chunk_warnings,
                processing_method,
                multi_line_messages,
            );
            axis_process_data.set_will_resume(axis);
        }
    }

    /// Java private `resume(AxisID)`.
    fn resume_axis(&'static self, axis_id: Option<AxisID>) {
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
                resume_data
                    .get_process_result_display()
                    .map(|display| Rc::clone(display.get())),
                resume_data
                    .get_process_series()
                    .map(|series| Rc::clone(series.get())),
                resume_data.is_popup_chunk_warnings(),
                resume_data.get_processing_method(),
                resume_data.is_multi_line_messages(),
            );
        }
        resume_data.reset();
    }

    /// `BaseManager.updateProcessChunks` itself, for the overrides' `super` call.
    /// Java package-private `updateProcessChunks`.  Pass in param to allow override
    /// functions to modify it.  Constructs (if necessary), modifies, and returns param.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2629): a null
    /// `getBaseMetaData()` throws a NullPointerException; here the subdirectory name is
    /// left unset.
    fn update_process_chunks_super(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<ProcesschunksParam>,
        root_name: Option<&str>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        dialog_type: Option<DialogType>,
    ) -> Option<ProcesschunksParam> {
        let mut param = match param {
            Some(param) => param,
            None => ProcesschunksParam::get_instance_dialog_type(
                self.this(),
                axis_id.unwrap_or(AxisID::Only),
                root_name,
                None,
                dialog_type,
            ),
        };
        param.set_subcommand_details(subcommand_details);
        let meta_data = self.get_base_meta_data();
        if let Some(meta_data) = meta_data
            && meta_data
                .base()
                .is_current_processchunks_subdir_name_set(axis_id)
        {
            param.set_subdir_name(
                meta_data
                    .base()
                    .get_current_processchunks_subdir_name(axis_id)
                    .as_deref(),
            );
        }
        Some(param)
    }

    /// Java package-private `updateProcessChunks`.  An override may return null.
    fn update_process_chunks(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<ProcesschunksParam>,
        root_name: Option<&str>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        dialog_type: Option<DialogType>,
    ) -> Option<ProcesschunksParam> {
        self.update_process_chunks_super(axis_id, param, root_name, subcommand_details, dialog_type)
    }

    /// Java `resume(AxisID, ProcesschunksParam, ProcessResultDisplay, ProcessSeries,
    /// CommandDetails, boolean, ProcessingMethod, boolean, DialogType)`, the virtual
    /// method (`PeetManager` overrides it); the base body is [`Self::resume_super`].
    #[allow(clippy::too_many_arguments)]
    fn resume(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
        dialog_type: Option<DialogType>,
    ) {
        self.resume_super(
            axis_id,
            param,
            process_result_display,
            process_series,
            subcommand_details,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
            dialog_type,
        )
    }

    /// `BaseManager.resume` itself, for the overrides' `super` call.  Java
    /// `resume(AxisID, ProcesschunksParam, ProcessResultDisplay, ProcessSeries,
    /// CommandDetails, boolean, ProcessingMethod, boolean, DialogType)`.  Get the
    /// current processchunks root name from meta data.  If it exists, attempt to resume
    /// processchunks.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2653, :2663, :2678): a null
    /// process manager, `getBaseMetaData()` or main panel throws a NullPointerException;
    /// here a null process manager leaves the axis not in use, a null meta data has no
    /// root name, and a null main panel has no parallel panel.
    #[allow(clippy::too_many_arguments)]
    fn resume_super(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
        dialog_type: Option<DialogType>,
    ) {
        let axis = axis_id.unwrap_or(AxisID::Only);
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self.this(), axis, dialog_type, Some("resume")),
        };
        let mut process_result_display = process_result_display;
        let axis_in_use = self.get_process_manager().is_some_and(|process_manager| {
            process_manager.in_use(
                axis,
                process_result_display.as_ref().map(|display| {
                    Arc::new(EdtRef::new(Rc::clone(display))) as ProcessResultDisplayRef
                }),
                false,
            )
        });
        let mut orig_process_result_display: Option<ProcessResultDisplayHandle> = None;
        if axis_in_use {
            // When the axis is in use, store the resume information. The result display
            // should not change.
            orig_process_result_display = process_result_display.take();
        }
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let meta_data = self.get_base_meta_data();
        let param = match param {
            Some(param) => param,
            None => {
                let root_name = meta_data.and_then(|meta_data| {
                    meta_data
                        .base()
                        .get_current_processchunks_root_name(axis_id)
                });
                if let Some(root_name) = root_name
                    .as_deref()
                    // Java `!rootName.matches("\\s*")`.
                    .filter(|root_name| {
                        !root_name
                            .chars()
                            .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
                    })
                {
                    match self.update_process_chunks(
                        axis_id,
                        None,
                        Some(root_name),
                        subcommand_details,
                        dialog_type,
                    ) {
                        Some(param) => Arc::new(param),
                        // An override (BatchRunTomoManager) returns null when its
                        // parameters cannot be read; the source then throws a
                        // NullPointerException in `getResumeParameters(param, true)`.
                        // Fixed in translation: the resume fails to start.
                        None => {
                            self.send_msg_process_failed_to_start(
                                process_result_display_ref.as_ref(),
                            );
                            process_series.borrow().end_series();
                            return;
                        }
                    }
                } else {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string(
                            Some(self.this()),
                            "No command to resume",
                            "Resume Failed",
                        )
                    });
                    self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                    process_series.borrow().end_series();
                    return;
                }
            }
        };
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(axis));
        let Some(parallel_panel) = parallel_panel else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    axis_id,
                )
            });
            return;
        };
        if !parallel_panel.get_resume_parameters(&param, true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if axis_in_use {
            self.save_resume(
                axis_id,
                Some(param),
                orig_process_result_display,
                Some(process_series),
                popup_chunk_warnings,
                processing_method,
                multi_line_messages,
            );
            return;
        }
        self.resume_private(
            axis_id,
            Some(param),
            process_result_display,
            Some(process_series),
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
        );
    }

    /// Java package-private `createRunList(RunType)`.  The base class returns null.
    /// The list is handed to monitor threads, hence the `Arc`.
    fn create_run_list(
        &self,
        run_type: Option<RunType>,
    ) -> Option<std::sync::Arc<crate::imod::etomo::r#type::run_list::RunList>> {
        let _ = run_type;
        None
    }

    /// Java package-private `getMessagesArray`.  The list is one object shared with the
    /// process layer, hence the `Arc<Mutex<..>>`.
    fn get_messages_array(&self) -> Option<MessagesArray> {
        None
    }

    /// Java private final `resume(AxisID, ProcesschunksParam, ProcessResultDisplay,
    /// ProcessSeries, boolean, ProcessingMethod, boolean)`.  Resume processchunks.
    ///
    /// Upstream bug fixed in translation (BaseManager.java:2728, :2731): a null `param`
    /// or `getBaseMetaData()`, or a null main panel, throws a NullPointerException; here
    /// the missing meta data is skipped, a null param ends the resume, and a null main
    /// panel has no parallel panel.
    #[allow(clippy::too_many_arguments)]
    fn resume_private(
        &'static self,
        axis_id: Option<AxisID>,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
    ) {
        let Some(param) = param else {
            return;
        };
        let axis = axis_id.unwrap_or(AxisID::Only);
        let meta_data = self.get_base_meta_data();
        if let Some(meta_data) = meta_data {
            meta_data
                .base()
                .set_current_processchunks_root_name(axis_id, param.get_root_name().as_deref());
            meta_data
                .base()
                .set_current_processchunks_subdir_name(axis_id, param.get_subdir_name().as_deref());
        }
        self.save_storable(
            axis_id,
            meta_data.map(|meta_data| meta_data as &dyn Storable),
        );
        let parallel_panel = self
            .get_main_panel()
            .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(axis));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    axis_id,
                )
            });
            return;
        };
        let managed_process_data = Arc::new(Mutex::new(ProcessData::get_managed_instance(
            axis_id,
            Some(self.this()),
            Some(ProcessName::PROCESSCHUNKS),
        )));
        let display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        let started: Option<Result<String, AxisBusyException>> =
            self.get_process_manager().map(|process_manager| {
                process_manager.processchunks(
                    axis,
                    Arc::clone(&param),
                    &*parallel_panel.get_parallel_progress_display(),
                    display_ref,
                    process_series.map(|process_series| {
                        Arc::new(EdtRef::new(process_series)) as ProcessSeriesRef
                    }),
                    popup_chunk_warnings,
                    processing_method,
                    multi_line_messages,
                    Some(RunType::ResumeProcessChunks),
                    Some(managed_process_data),
                    self.get_messages_array(),
                )
            });
        let thread_name = match started {
            None => return,
            Some(Err(e)) => {
                eprintln!("{}", e.0);
                let message = vec![
                    format!("Can not execute {}", ProcessName::PROCESSCHUNKS),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self.this()),
                        &message,
                        "Unable to execute command",
                        axis_id,
                    )
                });
                return;
            }
            Some(Ok(thread_name)) => thread_name,
        };
        self.set_thread_name(Some(&thread_name), axis_id);
    }

    /// Java final `tomosnapshot`.
    fn tomosnapshot(&'static self, axis_id: Option<AxisID>) {
        let process_manager = self.get_process_manager();
        if let Some(process_manager) = process_manager {
            process_manager.tomosnapshot(
                axis_id.unwrap_or(AxisID::Only),
                self.is_tomosnapshot_thumbnail(),
            );
        } else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.this()),
                    "No processes can be run in this interface.",
                    "Unable to run tomosnapshot",
                    axis_id,
                )
            });
        }
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
        .get_imod_directory()
        .map(|directory| {
            format!(
                "{}{}bin{}",
                utilities::java_io_file_get_absolute_path(&directory.to_string_lossy()),
                std::path::MAIN_SEPARATOR,
                std::path::MAIN_SEPARATOR
            )
        })
}

/// Java `chunkComscriptAction(Container)`, a static method.
pub fn chunk_comscript_action(root: Option<Rc<JComponent>>) -> Option<PathBuf> {
    // Open up the file chooser in the working directory
    let chooser = crate::imod::etomo::ui::swing::file_chooser::FileChooser::new_base_manager_string(
        None,
        etomo_director::INSTANCE.get_original_user_dir().as_deref(),
    );
    let filter =
        crate::imod::etomo::storage::chunk_comscript_file_filter::ChunkComscriptFileFilter::new();
    chooser.set_file_filter(Some(Rc::new(filter)));
    // Swing layout: chooser.setPreferredSize(FixedDim.fileChooser).
    chooser.set_file_selection_mode(crate::imod::etomo::ui::swing::file_chooser::FILES_ONLY);
    let return_val = chooser.show_open_dialog(root.as_ref());
    if return_val == crate::imod::etomo::ui::swing::file_chooser::APPROVE_OPTION {
        return chooser.get_selected_file();
    }
    None
}

/// Java `BaseManager implements BrowsingDirectory`: a manager handed to a widget as
/// its `BrowsingDirectory` (`setBrowsingDirectory(manager)`).  The two interface
/// methods are `BaseManager::get_browsing_dir`/`set_browsing_dir`.
pub struct ManagerBrowsingDirectory(pub &'static dyn BaseManager);

impl crate::imod::etomo::ui::browsing_directory::BrowsingDirectory for ManagerBrowsingDirectory {
    fn get_browsing_dir(&self) -> Option<PathBuf> {
        self.0.get_browsing_dir()
    }

    fn set_browsing_dir(&self, file: Option<&Path>) {
        self.0.set_browsing_dir(file);
    }
}
