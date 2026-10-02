//! `IMOD/Etomo/src/etomo/EtomoDirector.java`.
//!
//! Directs ApplicationManager, JoinManager and the other managers through
//! `BaseManager`: it owns the manager list and the current manager, the user
//! configuration (`$HOME/.etomo`) and its settings dialog, the IMOD and IMOD
//! calibration directories, and the program's startup (`main`, `setup`,
//! `initialize`) and shutdown (`exitProgram`).
//!
//! **Shape.**  Java's `INSTANCE` is a plain object whose methods call each
//! other re-entrantly: a director method builds a manager whose constructor
//! calls back into the director, a `UIHarness` popup reads the current manager,
//! a settings-dialog button calls `getSettingsParameters`.  A lock around the
//! whole director would deadlock on that, so [`INSTANCE`] is a bare
//! `LazyLock<EtomoDirector>`, every method takes `&self`, and each mutable
//! field carries its own interior mutability: a `Mutex` or an atomic for the
//! fields other threads touch (the utility thread saves logs and checks memory;
//! process threads read the directories and the Python script path), and an
//! [`EdtCell`] for the Swing-side settings dialog.  A `Mutex` guard is never
//! held across a call out of this module.  The two `synchronized` Java methods
//! hold `monitor`, a lock the owning thread may take again, as a Java monitor
//! is.
//!
//! **User configuration.**  Java's `private static final UserConfiguration
//! USER_CONFIG` is [`USER_CONFIG`] here.  `UserConfiguration` is not `Clone`
//! (it holds `Mutex`es) and Java hands out the shared object, so callers reach
//! it through closures: [`EtomoDirector::with_user_configuration`] for reads
//! and [`EtomoDirector::with_user_configuration_mut`] for writes (Java
//! `getUserConfiguration()` followed by the getter or setter).  The object is
//! guarded by a re-entrant lock plus a `RefCell`, so a read nested inside a
//! read on the same thread (the settings dialog's `setParameters` reaches
//! `Network`, whose `Node` reads the configuration again) works, while a
//! write nested inside a read panics by name instead of deadlocking.
//!
//! **JVM facts with no Rust counterpart.**  `setup()` prints the JVM's system
//! properties and the listing of the JVM's `lib` directory; `setLookAndFeel`
//! prints the Swing look-and-feel defaults; `setUIFont`, `setLookAndFeel` and
//! the `ToolTipManager` delays configure Swing itself.  Those are properties of
//! the Java runtime, not of the program (CLAUDE.md, "etomo's Java startup
//! dump"); each site says what the Java did.  `System.getProperty("user.dir")`
//! is the working directory at startup, and `System.setProperty("user.dir",
//! ...)` sets `PWD`, as `BaseManager.makePropertyUserDirLocal` does in this
//! translation.  `-D` JVM properties (`IMOD_DIR`, `IMOD_CALIB_DIR`) never
//! exist.  `Runtime.maxMemory/totalMemory/freeMemory` describe the JVM heap;
//! the host's physical memory stands in for the heap (see
//! [`EtomoDirector::get_available_memory`]).
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use super::application_manager::ApplicationManager;
use super::arguments::Arguments;
use super::base_manager::BaseManager;
use super::batch_run_tomo_manager::BatchRunTomoManager;
use super::directive_editor_manager::DirectiveEditorManager;
use super::front_page_manager::FrontPageManager;
use super::join_manager::JoinManager;
use super::local_arguments::LocalArguments;
use super::logic::version_control;
use super::manager_key::ManagerKey;
use super::parallel_manager::ParallelManager;
use super::process::intermittent_background_process::IntermittentBackgroundProcess;
use super::process::process_messages::ProcessMessages;
use super::process::process_restarter::ProcessRestarter;
use super::serial_sections_manager::SerialSectionsManager;
use super::storage::batch_run_tomo_file_filter::BatchRunTomoFileFilter;
use super::storage::etomo_file_filter::EtomoFileFilter;
use super::storage::join_file_filter::JoinFileFilter;
use super::storage::parallel_file_filter::ParallelFileFilter;
use super::storage::parameter_store::ParameterStore;
use super::storage::peet_file_filter::PeetFileFilter;
use super::storage::serial_sections_file_filter::SerialSectionsFileFilter;
use super::tools_manager::ToolsManager;
use super::r#type::axis_id::AxisID;
use super::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use super::r#type::data_file_type::DataFileType;
use super::r#type::dialog_type::DialogType;
use super::r#type::directive_file_type::DirectiveFileType;
use super::r#type::etomo_number::EtomoNumber;
use super::r#type::image_filename_style::ImageFilenameStyle;
use super::r#type::imod_version;
use super::r#type::interface_type::InterfaceType;
use super::r#type::meta_data::MetaData;
use super::r#type::parallel_meta_data;
use super::r#type::user_configuration::UserConfiguration;
use super::ui::swing::etomo_menu::ToolType;
use super::ui::swing::main_frame;
use super::ui::swing::settings_dialog::SettingsDialog;
use super::ui::swing::ui_harness;
use super::ui::swing::ui_parameters::UIParameters;
use super::ui::ui_component::UIComponent;
use super::ui_tester::UITester;
use super::util::clean_print::CleanPrint;
use super::util::environment_variable;
use super::util::event_queue::{self, EdtCell, ReentrantLock};
use super::util::redactor;
use super::util::unique_hashed_array::UniqueHashedArray;
use super::util::unique_key::UniqueKey;
use super::util::utilities;
use crate::imod::libcfshr::b3dutil;

/// Java `SCALE_IMAGES_ABOVE_FONT_SIZE`.
pub const SCALE_IMAGES_ABOVE_FONT_SIZE: i32 = 14;
/// Java `SCALE_IMAGES_BELOW_FONT_SIZE`.
pub const SCALE_IMAGES_BELOW_FONT_SIZE: i32 = 11;

/// Java `INSTANCE` (`EtomoDirector.java:80`):
/// `public static final EtomoDirector INSTANCE = new EtomoDirector();`.
/// See the module comment for why this is not behind a lock.
pub static INSTANCE: LazyLock<EtomoDirector> = LazyLock::new(EtomoDirector::new);

/// Java `ARGUMENTS` (`EtomoDirector.java:81`):
/// `public static final Arguments ARGUMENTS = new Arguments();`.  This is the object
/// `EtomoDirector.INSTANCE.getArguments()` returns (`EtomoDirector.java:1415`).
pub static ARGUMENTS: std::sync::LazyLock<std::sync::Mutex<Arguments>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Arguments::new()));

/// Java private static final `USER_CONFIG` (`EtomoDirector.java:82`).  See the
/// module comment.
static USER_CONFIG: LazyLock<UserConfigurationCell> = LazyLock::new(|| UserConfigurationCell {
    lock: ReentrantLock::new(),
    value: RefCell::new(UserConfiguration::new()),
});

/// Rust-only holder of [`USER_CONFIG`]: the configuration and the re-entrant
/// lock every access holds.
struct UserConfigurationCell {
    lock: ReentrantLock,
    value: RefCell<UserConfiguration>,
}

// SAFETY: `value` is only borrowed while the calling thread holds `lock`
// (`with_user_configuration`, `with_user_configuration_mut`,
// `is_user_preference_loaded`, `get_user_font_size`), and the guard is not
// `Send`, so the `RefCell` is never touched by two threads at once.
unsafe impl Sync for UserConfigurationCell {}

/// Java `DEFAULT_TO_STANDARD_IMAGE_FILE_NAMES`.
pub const DEFAULT_TO_STANDARD_IMAGE_FILE_NAMES: bool = true;
/// Java `USER_CONFIG_FILE_EXT`.
pub const USER_CONFIG_FILE_EXT: &str = ".etomo";
/// Java `IMOD_DIR_ENV_VAR`.
pub const IMOD_DIR_ENV_VAR: &str = "IMOD_DIR";
/// Java `SOURCE_ENV_VAR`.
pub const SOURCE_ENV_VAR: &str = "IMOD_UITEST_SOURCE";
/// Java `NEW_MESSAGE_PARSER`.
pub const NEW_MESSAGE_PARSER: bool = true;

/// Java private static final `TO_BYTES`.
const TO_BYTES: i32 = 1024;
/// Java `MIN_AVAILABLE_MEMORY_REQUIRED = 2 * TO_BYTES * TO_BYTES`.
pub const MIN_AVAILABLE_MEMORY_REQUIRED: f64 = (2 * TO_BYTES * TO_BYTES) as f64;
/// Java `NUMBER_STORABLES`.
pub const NUMBER_STORABLES: i32 = 2;
/// Java private static final `JAVA_MEMORY_LIMIT_ENV_VAR`.
const JAVA_MEMORY_LIMIT_ENV_VAR: &str = "ETOMO_MEM_LIM";
/// Java `FILE_INFO_CLEAN_PRINT_LABEL` (`EtomoDirector.java:94`).
pub const FILE_INFO_CLEAN_PRINT_LABEL: &str = "File Info";

// Testing and diagnostics
/// Java private static final `FILE_INFO_DIAGNOSTICS` (`EtomoDirector.java:97`).
pub const FILE_INFO_DIAGNOSTICS: bool = false;
/// Java private static final `SIMULATE_WINDOWS`.
const SIMULATE_WINDOWS: bool = false;

// TODO(unit): needs etomo/type/JoinMetaData.java - `getNewFileTitle()` returns the
// private `newJoinTitle` (JoinMetaData.java:167).
const JOIN_META_DATA_NEW_FILE_TITLE: &str = "New Join";
// TODO(unit): needs etomo/type/BatchRunTomoMetaData.java - `NEW_TITLE`
// (BatchRunTomoMetaData.java:25).
const BATCH_RUN_TOMO_META_DATA_NEW_TITLE: &str = "Batch Run Tomo";
// TODO(unit): needs etomo/type/PeetMetaData.java - `NEW_TITLE` (PeetMetaData.java:119).
const PEET_META_DATA_NEW_TITLE: &str = "PEET";
// TODO(unit): needs etomo/type/SerialSectionsMetaData.java - `NEW_TITLE`
// (SerialSectionsMetaData.java:29).
const SERIAL_SECTIONS_META_DATA_NEW_TITLE: &str = "Serial Sections";

/// Java private field `testFailed`, kept outside the director so it can be read
/// by code that must not reach the director (see [`is_test_failed`]).
static TEST_FAILED: AtomicBool = AtomicBool::new(false);

/// Java `EtomoDirector.INSTANCE.isTestFailed()`, as a free function for callers
/// that only need the flag.
pub fn is_test_failed() -> bool {
    TEST_FAILED.load(Ordering::SeqCst)
}

/// Java `public final class EtomoDirector`.
pub struct EtomoDirector {
    /// Java private final `homeDirectory`.
    home_directory: String,

    /// Java private `settingsDialog`, initially null.
    settings_dialog: EdtCell<std::rc::Rc<SettingsDialog>>,
    /// Java private `outOfMemoryMessage`, initially false.
    out_of_memory_message: AtomicBool,
    /// Java private `originalUserDir`, initially null.
    original_user_dir: Mutex<Option<String>>,
    // Java private `testFailed` is [`TEST_FAILED`].
    /// Java private final `javaMemoryLimit = new EtomoNumber(EtomoNumber.Type.LONG)`.
    java_memory_limit: Mutex<EtomoNumber>,
    /// Java private `imodBriefHeader`, initially false.
    imod_brief_header: AtomicBool,
    /// Java private `numberOfProcessorsWindows`, initially null.
    number_of_processors_windows: Mutex<Option<EtomoNumber>>,
    /// Java private `uiTester`, initially null.
    ui_tester: Mutex<Option<Arc<dyn UITester + Send + Sync>>>,

    // state
    /// Java private `currentManagerKey`, initially null.  It is the manager's
    /// own `ManagerKey` holder (`manager.getManagerKey()`), shared, so that
    /// `renameCurrentManager` renames the manager's key too.
    current_manager_key: Mutex<Option<Arc<Mutex<ManagerKey>>>>,
    /// Java private `defaultWindow`, initially false.
    default_window: AtomicBool,
    /// Java private `isAdvanced`, initially false.  Advanced dialog state for
    /// this instance, this gets set upon startup from the user configuration
    /// and can be modified for this instance by either the option or advanced
    /// menu items.
    is_advanced: AtomicBool,
    /// Java private `debug`, initially false.
    debug: AtomicBool,

    // Initialized in initialize() or in function called by initialize().
    /// Java private `parameterStore`.
    parameter_store: Mutex<Option<ParameterStore>>,
    /// Java private `managerList`.
    manager_list: Mutex<Option<UniqueHashedArray<&'static dyn BaseManager>>>,
    /// Java private `IMODDirectory`.
    imod_directory: Mutex<Option<PathBuf>>,
    /// Java private `IMODCalibDirectory`.
    imod_calib_directory: Mutex<Option<PathBuf>>,
    /// Java private `utilityThread`.
    utility_thread: Mutex<Option<Arc<UtilityThread>>>,
    /// Rust-only: the join handle of the thread running `utilityThread`, so
    /// `main` can wait for it as the JVM waits for a non-daemon thread.
    utility_thread_handle: Mutex<Option<std::thread::JoinHandle<()>>>,

    /// Java private `pythonScriptPath`, initially null.
    python_script_path: Mutex<Option<String>>,

    /// Rust-only: the monitor of the two `synchronized` methods.
    monitor: ReentrantLock,
}

impl EtomoDirector {
    /// Java private `EtomoDirector()`.
    fn new() -> EtomoDirector {
        EtomoDirector {
            // System.getProperty("user.home")
            home_directory: std::env::var("HOME").unwrap_or_default(),
            settings_dialog: EdtCell::new(),
            out_of_memory_message: AtomicBool::new(false),
            original_user_dir: Mutex::new(None),
            java_memory_limit: Mutex::new(EtomoNumber::new_with_type(Some(Type::Long))),
            imod_brief_header: AtomicBool::new(false),
            number_of_processors_windows: Mutex::new(None),
            ui_tester: Mutex::new(None),
            current_manager_key: Mutex::new(None),
            default_window: AtomicBool::new(false),
            is_advanced: AtomicBool::new(false),
            debug: AtomicBool::new(false),
            parameter_store: Mutex::new(None),
            manager_list: Mutex::new(None),
            imod_directory: Mutex::new(None),
            imod_calib_directory: Mutex::new(None),
            utility_thread: Mutex::new(None),
            utility_thread_handle: Mutex::new(None),
            python_script_path: Mutex::new(None),
            monitor: ReentrantLock::new(),
        }
    }

    /// Java static `main(String[])` (`EtomoDirector.java:133`).
    ///
    /// Java queues `new Setup()` with `SwingUtilities.invokeLater` when not
    /// headless, then returns; the JVM stays up while the event dispatch
    /// thread and the non-daemon `UtilityThread` run, until `UIHarness.exit`
    /// calls `System.exit`.  Rust ends the process when `main` returns, so:
    /// * with the `gui` feature the Slint event loop is the event dispatch
    ///   thread; `setup` runs on this thread (which `UIHarness.createMainFrame`
    ///   makes the EDT) and then the event loop runs here;
    /// * without it the setup is queued on the headless EDT and this thread
    ///   parks for good (the JVM's wait for its event dispatch thread);
    /// * headless, after `setup` this thread waits for the utility thread, the
    ///   JVM's wait for that non-daemon thread.
    pub fn main(args: &[String]) {
        EtomoDirector::main_body(args, false);
    }

    /// Rust-only variant of [`EtomoDirector::main`] for the click driver
    /// (`driver.rs`): exactly Java's `main` as the JVM runs it, which queues
    /// `new Setup()` on the event dispatch thread and *returns*, so the
    /// caller (the driver, as `EtomoDriver.java` on the JVM's main thread)
    /// keeps running.  The setup always goes to the headless
    /// `AWT-EventQueue-0` thread, with or without the `gui` feature, and the
    /// non-daemon wait of the JVM is the caller's business.
    pub fn main_headless_edt(args: &[String]) {
        EtomoDirector::main_body(args, true);
    }

    /// The body of Java `main(String[])`; with `return_after_queue` the
    /// non-headless setup is queued on the headless event dispatch thread and
    /// this returns (see [`EtomoDirector::main_headless_edt`]).
    fn main_body(args: &[String], return_after_queue: bool) {
        eprintln!("Running Etomo...");
        eprintln!(
            "Etomo Version:  {} {}\n",
            imod_version::CURRENT_VERSION,
            version_control::TIME_STAMP
        );
        eprintln!("Arguments:");
        for arg in args {
            eprint!("{} ", arg);
        }
        eprintln!("\n");

        ARGUMENTS.lock().unwrap().parse(args);
        let debug = ARGUMENTS.lock().unwrap().is_debug();
        INSTANCE.debug.store(debug, Ordering::SeqCst);

        let stress_test = ARGUMENTS.lock().unwrap().is_stress_test();
        if FILE_INFO_DIAGNOSTICS || stress_test {
            CleanPrint::add_allowed_label(Some(FILE_INFO_CLEAN_PRINT_LABEL));
        }

        INSTANCE.load_user_configuration();

        let grab_it = ARGUMENTS.lock().unwrap().is_grab_it();
        if !grab_it {
            let headless = ARGUMENTS.lock().unwrap().is_headless();
            if headless {
                EtomoDirector::setup();
                // The JVM waits for the non-daemon UtilityThread.
                let handle = INSTANCE.utility_thread_handle.lock().unwrap().take();
                if let Some(handle) = handle {
                    let _ = handle.join();
                }
            } else if return_after_queue {
                // SwingUtilities.invokeLater(new Setup()); then main returns.
                event_queue::invoke_later(EtomoDirector::setup);
            } else {
                // SwingUtilities.invokeLater(new Setup());
                #[cfg(feature = "gui")]
                {
                    // Setup.run()
                    EtomoDirector::setup();
                    let result = ui_harness::with(|harness| harness.run());
                    if let Err(error) = result {
                        eprintln!("{}", error);
                        b3dutil::exit(1);
                    }
                }
                #[cfg(not(feature = "gui"))]
                {
                    // Setup.run()
                    event_queue::invoke_later(EtomoDirector::setup);
                    loop {
                        std::thread::park();
                    }
                }
            }
        } else {
            // ProcessMessages.grabIt(): reads each argument file; the parsed
            // messages are returned to this caller, which has no use for them.
            let arguments = ARGUMENTS.lock().unwrap().clone();
            if let Err(error) = ProcessMessages::grab_it(&arguments) {
                eprintln!("{}", error);
            }
        }
    }

    /// Rust-only entry point of the `etomo-gui` command: Java `main(String[])`.
    /// A Slint failure is reported and exits inside `main`, so this returns
    /// `Ok` when `main` returns.
    #[cfg(feature = "gui")]
    pub fn main_gui(arguments: &[String]) -> Result<(), String> {
        EtomoDirector::main(arguments);
        Ok(())
    }

    /// Java static `isSimulateWindows()`.
    pub fn is_simulate_windows() -> bool {
        SIMULATE_WINDOWS
    }

    /// Java static `isUserPreferenceLoaded()`.
    pub fn is_user_preference_loaded() -> bool {
        let _guard = USER_CONFIG.lock.lock();
        USER_CONFIG.value.borrow().is_loaded()
    }

    /// Java static `getUserFontSize()`.
    pub fn get_user_font_size() -> i32 {
        let _guard = USER_CONFIG.lock.lock();
        USER_CONFIG.value.borrow().get_font_size()
    }

    /// Java `pack(BaseManager)`.
    pub fn pack(&self, manager: Option<&'static dyn BaseManager>) {
        ui_harness::with(|harness| harness.pack_base_manager(manager));
    }

    /// Java private `loadUserConfiguration()`.  Set the user preferences.
    fn load_user_configuration(&self) {
        let debug = self.debug.load(Ordering::SeqCst);
        // Create a File object specifying the user configuration file
        // create the user config file
        // (new File(homeDirectory, USER_CONFIG_FILE_EXT) cannot throw here.)
        let user_config_file = PathBuf::from(utilities::java_io_file_new(
            &self.home_directory,
            USER_CONFIG_FILE_EXT,
        ));
        // create config file it if it doesn't exist
        if let Err(except) = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&user_config_file)
            && except.kind() != std::io::ErrorKind::AlreadyExists
        {
            eprintln!("java.io.IOException: {}", except);
            if debug {
                eprintln!(
                    "Could not create file:{}",
                    utilities::java_io_file_get_absolute_path(&user_config_file.to_string_lossy())
                );
                eprintln!("{}", except);
            }
            b3dutil::exit(1);
        }
        let ignore_settings = ARGUMENTS.lock().unwrap().is_ignore_settings();
        let parameter_store = if !ignore_settings {
            ParameterStore::get_instance(Some(user_config_file))
        } else {
            ParameterStore::get_fileless_instance().map(Some)
        };
        match parameter_store {
            Ok(parameter_store) => {
                let mut store = self.parameter_store.lock().unwrap();
                *store = parameter_store;
                if let Some(store) = store.as_ref() {
                    let _guard = USER_CONFIG.lock.lock();
                    store.load(&USER_CONFIG.value);
                }
            }
            Err(except) => {
                // catch (LogFile.FileException | IOException except)
                if debug {
                    eprintln!("{}", except);
                }
                eprintln!("java.io.IOException: {}", except);
                let message = format!("Can't load user configuration.\n{}", except);
                let current_manager = self.get_current_manager();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        current_manager,
                        &message,
                        "Etomo Error",
                    )
                });
            }
        }
        // catch (LockException except) {}: the translated ParameterStore takes
        // no lock.
        self.set_user_preferences();
    }

    // Updates done

    /// Java `setMaintainEtomo(UITester)`.
    pub fn set_maintain_etomo(&self, ui_tester: Option<Arc<dyn UITester + Send + Sync>>) {
        *self.ui_tester.lock().unwrap() = ui_tester;
    }

    /// Java `stopMaintainEtomo()`.
    pub fn stop_maintain_etomo(&self) {
        let ui_tester = self.ui_tester.lock().unwrap().clone();
        if let Some(ui_tester) = ui_tester {
            ui_tester.set_maintain_etomo(false);
        }
    }

    /// Java `isUnitTest()`.
    pub fn is_unit_test(&self) -> bool {
        let arguments = ARGUMENTS.lock().unwrap();
        arguments.is_test() && arguments.is_headless() && arguments.is_self_test()
    }

    /// Java `isTest()`.
    pub fn is_test(&self) -> bool {
        ARGUMENTS.lock().unwrap().is_test()
    }

    /// Java private static `setup()`.
    ///
    /// The Java catches `OutOfMemoryError`, `Exception` and `Error`; a Rust
    /// panic out of `initialize`/`doAutomation` takes the `Exception` arm.
    fn setup() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // First and last lines of functions stored in the environment. Avoid adding
            // them to the Environment variables list.  (java.util.regex `\s` is
            // [ \t\n\x0B\f\r]; `matches` is a whole-string match.)
            let first_line =
                regex::Regex::new(r"^(?:[ \t\n\x0B\x0C\r]*?\(\)[ \t\n\x0B\x0C\r]*?\{.*)$").unwrap();
            let last_line = regex::Regex::new(r"^(?:.*\}[ \t\n\x0B\x0C\r]*)$").unwrap();
            // String.split("\\R")
            let line_break =
                regex::Regex::new("\r\n|[\n\x0B\x0C\r\u{85}\u{2028}\u{2029}]").unwrap();
            // Environment variables.
            eprintln!("\nEnvironment variables:\n");
            let is_test = ARGUMENTS.lock().unwrap().is_test();
            if !is_test {
                for (key, value) in std::env::vars_os() {
                    let key = key.to_string_lossy().into_owned();
                    let value = value.to_string_lossy().into_owned();
                    // value.split("\\R"): a split with no match is the whole
                    // input; otherwise trailing empty strings are dropped.
                    let mut array: Vec<&str> = line_break.split(&value).collect();
                    if array.len() > 1 {
                        while array.last().is_some_and(|last| last.is_empty()) {
                            array.pop();
                        }
                    }
                    if !array.is_empty()
                        && (!first_line.is_match(array[0])
                            || !last_line.is_match(array[array.len() - 1]))
                    {
                        let redacted_key = redactor::INSTANCE.redact_name(Some(&key));
                        if let Some(redacted_key) = redacted_key {
                            eprintln!(
                                "{}:  {}",
                                redacted_key,
                                redactor::INSTANCE
                                    .redact_value(Some(&value))
                                    .unwrap_or_else(|| "null".to_owned())
                            );
                        }
                    }
                }
                eprintln!();
            }

            // Functions stored in the environment.
            // (Commented out in the Java: functions don't seem to carry useful
            // information, and they would be hard to redact.)

            // Java properties: `System.getProperties()` lists the JVM's system
            // properties (redacted).  JVM-only; see the module comment.

            // Java library: the listing of `${java.home}/lib`.  JVM-only.

            eprintln!();
            utilities::date_time_stamp();
            utilities::set_start_time();
            INSTANCE.initialize();
            // automation must be done last in main, otherwise initialization may not
            // complete normally.
            INSTANCE.do_automation_void();
        }));
        if let Err(payload) = result {
            // catch (final Exception e)
            let message = payload
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "null".to_owned());
            if INSTANCE.debug.load(Ordering::SeqCst) {
                eprintln!("{}", message);
            }
            eprintln!("{}", message);
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(None, &message, "Exception")
            });
            let headless = INSTANCE.get_arguments().is_headless();
            if headless {
                ui_harness::with(|harness| harness.exit(Some(AxisID::Only), 1));
            }
        }
    }

    /// Java private `setAdvanced(boolean)`.
    fn set_advanced(&self, state: bool) {
        self.is_advanced.store(state, Ordering::SeqCst);
    }

    /// Java `doAutomation()`.
    pub fn do_automation_void(&self) {
        if self.manager_list.lock().unwrap().is_none() {
            return;
        }
        let mut manager: Option<&'static dyn BaseManager> = None;
        let current_key = self
            .current_manager_key
            .lock()
            .unwrap()
            .as_ref()
            .map(|manager_key| manager_key.lock().unwrap().get_key().cloned());
        if let Some(current_key) = current_key {
            manager = self.get_manager(current_key.as_ref());
        }
        if let Some(manager) = manager {
            manager.do_automation(None);
        }
    }

    /// Java private `doAutomation(ManagerKey, LocalArguments)`.  Do automation
    /// in a non-default manager.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:408-417): with no
    /// manager list the Java reports "Unable to open interface." and then, when
    /// not headless (where `exit` does not return), dereferences the null list
    /// (`NullPointerException`).  Here it returns after the report.
    fn do_automation_manager_key_local_arguments(
        &self,
        manager_key: Option<Arc<Mutex<ManagerKey>>>,
        local_arguments: Option<&LocalArguments>,
    ) {
        if self.manager_list.lock().unwrap().is_none() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    None,
                    "Unable to open interface.",
                    "Interface Failed",
                )
            });
            let headless = ARGUMENTS.lock().unwrap().is_headless();
            if headless {
                ui_harness::with(|harness| harness.exit(Some(AxisID::Only), 1));
            }
            return;
        }
        let mut manager: Option<&'static dyn BaseManager> = None;
        if let Some(manager_key) = manager_key {
            let key = manager_key.lock().unwrap().get_key().cloned();
            manager = self.get_manager(key.as_ref());
        }
        if let Some(manager) = manager {
            manager.do_automation(local_arguments);
        }
    }

    /// Java private `initialize()`.
    fn initialize(&self) {
        // System.getProperty("user.dir")
        *self.original_user_dir.lock().unwrap() = std::env::current_dir()
            .ok()
            .map(|directory| directory.to_string_lossy().into_owned());
        // Get the HOME directory environment variable to find the program
        // configuration file
        if self.home_directory.is_empty() {
            let message = [
                "Can not find home directory! Unable to load user preferences".to_owned(),
                "Set HOME environment variable and restart program to fix this problem".to_owned(),
            ];
            let current_manager = self.get_current_manager();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_array_string_axis_id(
                    current_manager,
                    &message,
                    "Program Initialization Error",
                    Some(AxisID::Only),
                )
            });
            b3dutil::exit(1);
        }
        self.imod_brief_header.store(
            environment_variable::INSTANCE.exists(
                None,
                Some(self.home_directory.as_str()),
                "IMOD_BRIEF_HEADER",
                Some(AxisID::Only),
            ),
            Ordering::SeqCst,
        );
        if utilities::is_windows_os()
            && environment_variable::INSTANCE.exists(
                None,
                Some(self.home_directory.as_str()),
                "NUMBER_OF_PROCESSORS",
                Some(AxisID::Only),
            )
        {
            let mut number_of_processors_windows = EtomoNumber::new();
            number_of_processors_windows.set_string(Some(
                environment_variable::INSTANCE
                    .get_value(
                        None,
                        Some(self.home_directory.as_str()),
                        "NUMBER_OF_PROCESSORS",
                        Some(AxisID::Only),
                    )
                    .as_str(),
            ));
            let is_test = ARGUMENTS.lock().unwrap().is_test();
            if !is_test {
                eprintln!("NUMBER_OF_PROCESSORS:{}", number_of_processors_windows);
            }
            *self.number_of_processors_windows.lock().unwrap() = Some(number_of_processors_windows);
        }

        let param_file_name_list: Vec<String> = ARGUMENTS
            .lock()
            .unwrap()
            .get_param_file_name_list()
            .to_vec();
        let is_help = ARGUMENTS.lock().unwrap().is_help();
        if is_help {
            self.print_usage_message();
            return;
        }
        ui_harness::with(|harness| harness.create_main_frame());
        // ARGUMENTS.validate(UIHarness.INSTANCE.getMainFrame()): the translated
        // validate reports its errors itself.
        let valid = ARGUMENTS.lock().unwrap().validate();
        if !valid {
            ui_harness::with(|harness| harness.exit(Some(AxisID::Only), 1));
            return;
        }
        self.init_imod_directory();
        let param_file_name_list_size = param_file_name_list.len();
        let mut param_file_name: Option<&str>;
        *self.manager_list.lock().unwrap() = Some(UniqueHashedArray::new());
        // if no param file is found bring up Parallel manager
        self.setup_imod_calib_dir();
        if param_file_name_list_size == 0 {
            self.default_window.store(true, Ordering::SeqCst);
        }
        let mut save_key: Option<Arc<Mutex<ManagerKey>>> = None;
        let mut manager_key: Option<Arc<Mutex<ManagerKey>>>;
        for i in 0..param_file_name_list_size {
            param_file_name = Some(&param_file_name_list[i]);
            let name = param_file_name.unwrap();
            manager_key = None;
            if name.ends_with(DataFileType::Recon.extension().unwrap()) {
                manager_key = self.open_tomogram_string_boolean_axis_id(
                    Some(name),
                    false,
                    Some(AxisID::Only),
                );
            } else if name.ends_with(DataFileType::Join.extension().unwrap()) {
                manager_key =
                    self.open_join_string_boolean_axis_id(Some(name), false, Some(AxisID::Only));
            } else if name.ends_with(DataFileType::Parallel.extension().unwrap()) {
                manager_key = self.open_parallel_string_boolean_axis_id(
                    Some(name),
                    false,
                    Some(AxisID::Only),
                );
            } else if name.ends_with(DataFileType::BatchRunTomo.extension().unwrap()) {
                manager_key = self.open_batch_run_tomo_string_boolean_axis_id(
                    Some(name),
                    false,
                    Some(AxisID::Only),
                );
            } else if name.ends_with(DataFileType::Peet.extension().unwrap()) {
                manager_key =
                    self.open_peet_string_boolean_axis_id(Some(name), false, Some(AxisID::Only));
            } else if name.ends_with(DataFileType::SerialSections.extension().unwrap()) {
                manager_key = self.open_serial_sections_string_boolean_axis_id(
                    Some(name),
                    false,
                    Some(AxisID::Only),
                );
            }
            if i == 0 {
                save_key = manager_key;
            }
        }
        if save_key.is_none() {
            manager_key = self.open_front_page_boolean_axis_id(true, Some(AxisID::Only));
            save_key = manager_key;
        }
        *self.current_manager_key.lock().unwrap() = save_key.clone();
        self.init_program();
        let mut manager: Option<&'static dyn BaseManager> = None;
        let current_key = save_key
            .as_ref()
            .and_then(|current_manager_key| current_manager_key.lock().unwrap().get_key().cloned());
        if save_key.is_some() {
            manager = self.get_manager(current_key.as_ref());
        }
        if let Some(manager) = manager {
            ui_harness::with(|harness| {
                harness.set_current_manager_base_manager_unique_key_boolean_boolean(
                    Some(manager),
                    current_key.as_ref(),
                    true,
                    true,
                )
            });
        }
        if save_key.is_some() {
            ui_harness::with(|harness| {
                harness.select_window_menu_item_unique_key(current_key.as_ref())
            });
        }
        self.set_current_manager_manager_key_boolean_boolean(save_key, false, false);
        // UIHarness.INSTANCE.setMRUFileLabels(USER_CONFIG.getMRUFileList()).
        // Upstream bug fixed in translation (EtomoMenu.java:406): a null entry
        // in the list makes `setMRUFileLabels` throw NullPointerException; a
        // null entry is passed as "" (a hidden menu item).
        let mru_file_list: Vec<String> = self
            .with_user_configuration_mut(|user_config| user_config.get_mru_file_list())
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect();
        ui_harness::with(|harness| harness.set_mru_file_labels(&mru_file_list));
        ui_harness::with(|harness| harness.pack_base_manager(manager));
        if manager.is_none() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    None,
                    "Invalid dataset file",
                    "Unable to Open Dataset",
                )
            });
        }
        ui_harness::with(|harness| harness.set_visible(manager, true));
        if manager.is_none() {
            ui_harness::with(|harness| harness.set_title(None, main_frame::ETOMO_TITLE));
        }
    }

    /// Java private `setupImodCalibDir()`.
    fn setup_imod_calib_dir(&self) {
        // Get the IMOD calibration directory so we know where to find documentation
        // Check to see if is defined on the command line first with -D
        // Otherwise check to see if we can get it from the environment
        // System.getProperty(CALIB_DIR): there are no JVM -D properties.
        let mut imod_calib_directory_name: Option<String> = None;
        if imod_calib_directory_name.is_none() {
            let value = environment_variable::INSTANCE.get_value(
                None,
                None,
                environment_variable::CALIB_DIR,
                Some(AxisID::Only),
            );
            if value != "" {
                eprintln!("{} (env): {}", environment_variable::CALIB_DIR, value);
            } else {
                eprintln!(
                    "WARNING:\nThe environment variable {}is not set.\n\
                     Several Etomo functions will not be available:\n\
                     Image distortion field files, Mag gradient correction, \
                     and parallel processing.\n",
                    environment_variable::CALIB_DIR
                );
            }
            imod_calib_directory_name = Some(value);
        } else {
            eprintln!(
                "{} (-D): {}",
                environment_variable::CALIB_DIR,
                imod_calib_directory_name.as_deref().unwrap()
            );
        }
        *self.imod_calib_directory.lock().unwrap() = imod_calib_directory_name.map(PathBuf::from);
    }

    /// Java private `initProgram()`.
    fn init_program(&self) {
        let debug = self.debug.load(Ordering::SeqCst);
        let mut printed;
        let is_test = ARGUMENTS.lock().unwrap().is_test();
        if !is_test {
            // GraphicsEnvironment.isHeadless(): the JVM's headless mode, which
            // etomo runs in exactly when -headless is given.
            let headless = ARGUMENTS.lock().unwrap().is_headless();
            eprintln!("\nGraphicsEnvironment.isHeadless()={}\n", headless);
        }
        printed = true;
        // print versions
        // VersionControl.getImodInfo(null): the translated function takes a
        // non-null axis, used only for its process messages.
        let imod_info = version_control::get_imod_info(AxisID::Only);
        if let Some(imod_info) = &imod_info
            && !imod_info.is_empty()
        {
            eprintln!("IMOD Version: {}", imod_info[0]);
            printed = true;
        }
        let version = version_control::get_peet_version();
        if version.is_some() {
            eprintln!(
                "PEET Version: {}",
                version_control::get_peet_version().unwrap_or_else(|| "null".to_owned())
            );
            printed = true;
        }
        if printed {
            eprintln!();
        }
        let utility_thread = Arc::new(UtilityThread::new());
        *self.utility_thread.lock().unwrap() = Some(Arc::clone(&utility_thread));
        // new Thread(utilityThread).start();
        let handle = std::thread::spawn(move || utility_thread.run());
        *self.utility_thread_handle.lock().unwrap() = Some(handle);
        // get the java memory limit
        // check it before complaining about having too little memory available
        // SGI seems to go very low on the available memory, but its fine as long
        // as long as it does't get near the java memory limit.
        let original_user_dir = self.original_user_dir.lock().unwrap().clone();
        let mut s_java_memory_limit: Option<String> =
            Some(environment_variable::INSTANCE.get_value(
                None,
                original_user_dir.as_deref(),
                JAVA_MEMORY_LIMIT_ENV_VAR,
                Some(AxisID::Only),
            ));
        if let Some(value) = s_java_memory_limit.as_mut() {
            let mut conversion_number: i32 = 1;
            if value.ends_with('k') || value.ends_with('K') {
                conversion_number = TO_BYTES;
                value.pop();
            } else if value.ends_with('m') || value.ends_with('M') {
                conversion_number = TO_BYTES * TO_BYTES;
                value.pop();
            }
            let mut java_memory_limit = self.java_memory_limit.lock().unwrap();
            java_memory_limit.set_string(Some(value.as_str()));
            let limit = java_memory_limit
                .get_long()
                .wrapping_mul(conversion_number as i64);
            java_memory_limit.set_long(limit);
            if debug {
                eprintln!("{}={}", JAVA_MEMORY_LIMIT_ENV_VAR, *java_memory_limit);
                eprintln!(
                    "MIN_AVAILABLE_MEMORY_REQUIRED={:?}",
                    MIN_AVAILABLE_MEMORY_REQUIRED
                );
            }
        }
    }

    /// Java private `printProperties(String)`.  Lists the Swing `UIManager`
    /// defaults whose key contains `type`; JVM-only (module comment).  No caller
    /// in the Java either.
    fn print_properties(&self, r#type: Option<&str>) {
        if r#type.is_none() {
            return;
        }
        // UIManager.getDefaults() keys containing type, printed as key=value,
        // then a blank line if any was printed: no Swing defaults exist here.
    }

    /// Java private `initIMODDirectory()`.
    pub fn init_imod_directory(&self) {
        // Get the IMOD directory so we know where to find documentation
        // Check to see if is defined on the command line first with -D
        // Otherwise check to see if we can get it from the environment
        // System.getProperty(IMOD_DIR_ENV_VAR): there are no JVM -D properties.
        let mut imod_directory_name: Option<String> = None;
        if imod_directory_name.is_none() {
            let value = environment_variable::INSTANCE.get_value(
                None,
                None,
                IMOD_DIR_ENV_VAR,
                Some(AxisID::Only),
            );
            if value == "" {
                // String[3] with two entries set: the third is null.
                let message = [
                    "Can not find IMOD directory!".to_owned(),
                    "Set IMOD_DIR environment variable and restart program to fix this problem"
                        .to_owned(),
                ];
                let current_manager = self.get_current_manager();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        current_manager,
                        &message,
                        "Program Initialization Error",
                        Some(AxisID::Only),
                    )
                });
                b3dutil::exit(1);
            } else {
                eprintln!("IMOD_DIR (env): {}", value);
            }
            imod_directory_name = Some(value);
        } else {
            eprintln!("IMOD_DIR (-D): {}", imod_directory_name.as_deref().unwrap());
        }
        *self.imod_directory.lock().unwrap() = imod_directory_name.map(PathBuf::from);
    }

    /// Java `getManager(UniqueKey)`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:651): before
    /// `initialize` the manager list is null and the Java throws
    /// NullPointerException; here there is no manager.
    pub fn get_manager(&self, key: Option<&UniqueKey>) -> Option<&'static dyn BaseManager> {
        let key = key?;
        self.manager_list
            .lock()
            .unwrap()
            .as_ref()?
            .get(key)
            .copied()
    }

    /// Java private `getCurrentManager()`.  Gets the current manager.  It is
    /// important that this function remain package level because the current
    /// manager changes when the user switches tabs, which would change the
    /// functionality of code executed after a process is finished.
    pub(crate) fn get_current_manager(&self) -> Option<&'static dyn BaseManager> {
        let current_key = self
            .current_manager_key
            .lock()
            .unwrap()
            .as_ref()
            .map(|manager_key| manager_key.lock().unwrap().get_key().cloned())?;
        self.get_manager(current_key.as_ref())
    }

    /// The current manager, for the click driver (`driver.rs`), which reads
    /// the current manager's log window as Java's `Window.getWindows()` would
    /// list it.  Rust-only; Java `getCurrentManager()` is package-private.
    pub fn get_current_manager_for_driver(&self) -> Option<&'static dyn BaseManager> {
        self.get_current_manager()
    }

    /// Java `getCurrentManagerForTest()`.  The Java throws
    /// `IllegalStateException("Illegal use of getCurrentManagerForTest")`
    /// outside a test; that is the `Err`.
    pub fn get_current_manager_for_test(&self) -> Result<Option<&'static dyn BaseManager>, String> {
        let is_test = ARGUMENTS.lock().unwrap().is_test();
        if !is_test {
            return Err("Illegal use of getCurrentManagerForTest".to_owned());
        }
        let current_key = self
            .current_manager_key
            .lock()
            .unwrap()
            .as_ref()
            .map(|manager_key| manager_key.lock().unwrap().get_key().cloned());
        let Some(current_key) = current_key else {
            return Ok(None);
        };
        Ok(self.get_manager(current_key.as_ref()))
    }

    /// Java `setCurrentPropertyUserDir(String)`, for testing.  Returns the old
    /// property user dir.  The Java throws `IllegalStateException("test-only
    /// function")` outside a test; that is the `Err`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:690): a current
    /// manager key whose manager is no longer listed makes the Java throw
    /// NullPointerException; here the key is treated as absent.
    pub fn set_current_property_user_dir(
        &self,
        property_user_dir: &str,
    ) -> Result<Option<String>, String> {
        let is_test = ARGUMENTS.lock().unwrap().is_test();
        if !is_test {
            return Err("test-only function".to_owned());
        }
        let current_manager = self.get_current_manager();
        match current_manager {
            None => {
                // System.setProperty("user.dir", propertyUserDir)
                let old = std::env::var("PWD").ok();
                unsafe { std::env::set_var("PWD", property_user_dir) };
                Ok(old)
            }
            Some(current_manager) => {
                Ok(current_manager.set_property_user_dir(Some(property_user_dir)))
            }
        }
    }

    /// Java `makeOriginalDirLocal()`.  Called when the propertyUserDir is empty,
    /// or there is no manager.
    pub fn make_original_dir_local(&self) {
        let original_user_dir = self.original_user_dir.lock().unwrap().clone();
        // System.setProperty("user.dir", originalUserDir)
        match &original_user_dir {
            Some(original_user_dir) => unsafe { std::env::set_var("PWD", original_user_dir) },
            // System.setProperty with a null value throws NullPointerException
            // (fixed in translation: nothing to set before initialize).
            None => {}
        }
        utilities::manager_stamp(original_user_dir.as_deref(), None);
    }

    /// Java `getOriginalUserDir()`.
    pub fn get_original_user_dir(&self) -> Option<String> {
        self.original_user_dir.lock().unwrap().clone()
    }

    /// Java `setCurrentManager(ManagerKey)`.
    pub fn set_current_manager_manager_key(&self, manager_key: Option<Arc<Mutex<ManagerKey>>>) {
        self.set_current_manager_manager_key_boolean_boolean(manager_key, false, true);
    }

    /// Java `setCurrentManager(UniqueKey)`.
    pub fn set_current_manager_unique_key(&self, key: Option<&UniqueKey>) -> bool {
        self.set_current_manager_unique_key_boolean(key, false)
    }

    /// Java synchronized `setCurrentManager(ManagerKey, boolean, boolean)`.
    /// Gets a manager from managerList with managerKey.
    pub fn set_current_manager_manager_key_boolean_boolean(
        &self,
        manager_key: Option<Arc<Mutex<ManagerKey>>>,
        new_window: bool,
        manager_stamp: bool,
    ) {
        let _synchronized = self.monitor.lock();
        let Some(manager_key) = manager_key else {
            return;
        };
        let key = manager_key.lock().unwrap().get_key().cloned();
        let Some(key) = key else {
            return;
        };
        let manager = self.get_manager(Some(&key));
        self.set_current_manager_base_manager_manager_key_boolean_boolean(
            manager,
            Some(manager_key),
            new_window,
            manager_stamp,
        );
    }

    /// Java synchronized `setCurrentManager(UniqueKey, boolean)`.  Gets a
    /// manager from managerList with key.
    pub fn set_current_manager_unique_key_boolean(
        &self,
        key: Option<&UniqueKey>,
        new_window: bool,
    ) -> bool {
        let _synchronized = self.monitor.lock();
        let Some(key) = key else {
            return false;
        };
        let new_current_manager = self.get_manager(Some(key));
        let Some(new_current_manager) = new_current_manager else {
            return false;
        };
        self.set_current_manager_base_manager_manager_key_boolean_boolean(
            Some(new_current_manager),
            Some(new_current_manager.get_manager_key()),
            new_window,
            true,
        );
        true
    }

    /// Java private `setCurrentManager(BaseManager, ManagerKey, boolean,
    /// boolean)`.  Checks newCurrentManager, sets currentManagerKey, and sets
    /// the current manager in UIHarness.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:755-756): a key
    /// with no listed manager makes the Java throw
    /// `NullPointerException("managerKey=" + managerKey)` out to the event
    /// dispatch thread.  Here the message is printed and the current manager
    /// is left unchanged.
    fn set_current_manager_base_manager_manager_key_boolean_boolean(
        &self,
        new_current_manager: Option<&'static dyn BaseManager>,
        manager_key: Option<Arc<Mutex<ManagerKey>>>,
        new_window: bool,
        manager_stamp: bool,
    ) {
        let Some(new_current_manager) = new_current_manager else {
            eprintln!(
                "java.lang.NullPointerException: managerKey={}",
                manager_key
                    .as_ref()
                    .map_or_else(|| "null".to_owned(), |key| key.lock().unwrap().to_string())
            );
            return;
        };
        *self.current_manager_key.lock().unwrap() = manager_key.clone();
        let key = manager_key
            .as_ref()
            .and_then(|manager_key| manager_key.lock().unwrap().get_key().cloned());
        ui_harness::with(|harness| {
            harness.set_current_manager_base_manager_unique_key_boolean_boolean(
                Some(new_current_manager),
                key.as_ref(),
                new_window,
                manager_stamp,
            )
        });
    }

    /// Java private `openTomogram(String, boolean, AxisID)`.  Creates
    /// ApplicationManager and adds it to ManagerList.
    fn open_tomogram_string_boolean_axis_id(
        &self,
        etomo_data_file_name: Option<&str>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let manager: &'static ApplicationManager;
        if etomo_data_file_name.is_none()
            || etomo_data_file_name == Some(MetaData::get_new_file_title())
        {
            manager = ApplicationManager::new(Some(""), axis_id);
            ui_harness::with(|harness| harness.set_enabled_new_tomogram_menu_item(false));
        } else {
            manager = ApplicationManager::new(etomo_data_file_name, axis_id);
        }
        self.set_manager(manager, axis_id, make_current)
    }

    /// Java `openJoin(boolean, AxisID)`.
    pub fn open_join_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_join_string_boolean_axis_id(
            Some(JOIN_META_DATA_NEW_FILE_TITLE),
            make_current,
            axis_id,
        )
    }

    /// Java `openParallel(boolean, AxisID)`.
    pub fn open_parallel_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_parallel_string_boolean_axis_id(None, make_current, axis_id)
    }

    /// Java `openGenericParallel(boolean, AxisID)`.
    pub fn open_generic_parallel(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_parallel_string_boolean_axis_id(
            Some(parallel_meta_data::NEW_GENERIC_PARALLEL_PROCESS_TITLE),
            make_current,
            axis_id,
        )
    }

    /// Java `openAnisotropicDiffusion(boolean, AxisID)`.
    pub fn open_anisotropic_diffusion(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_parallel_string_boolean_axis_id(
            Some(parallel_meta_data::NEW_ANISOTROPIC_DIFFUSION_TITLE),
            make_current,
            axis_id,
        )
    }

    /// Java `openBatchRunTomo(boolean, AxisID)`.
    pub fn open_batch_run_tomo_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_batch_run_tomo_string_boolean_axis_id(
            Some(BATCH_RUN_TOMO_META_DATA_NEW_TITLE),
            make_current,
            axis_id,
        )
    }

    /// Java `openToolInSeparateFrame(ToolType)`.  Build, save, and display a
    /// ToolsManager instance in a separate frame.
    pub fn open_tool_in_separate_frame(&self, tool_type: ToolType) -> &'static dyn BaseManager {
        let manager = ToolsManager::new(tool_type);
        ui_harness::with(|harness| harness.add_frame(manager, false));
        manager.initialize();
        utilities::manager_stamp(
            manager.get_property_user_dir().as_deref(),
            manager.get_name().as_deref(),
        );
        manager
    }

    /// Java `openTool(boolean, ToolType)`.  Build, save, and display a
    /// ToolsManager instance.
    pub fn open_tool(
        &self,
        make_current: bool,
        tool_type: ToolType,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(Some(AxisID::First));
        let manager = ToolsManager::new(tool_type);
        manager.initialize();
        utilities::manager_stamp(
            manager.get_property_user_dir().as_deref(),
            manager.get_name().as_deref(),
        );
        self.set_manager(manager, Some(AxisID::First), make_current)
    }

    /// Java `openDirectiveEditor(DirectiveFileType, BaseManager, String,
    /// StringBuffer)`.  Build, save, and display a DirectiveEditorManager
    /// instance.
    pub fn open_directive_editor(
        &self,
        directive_file_type: Option<DirectiveFileType>,
        data_source: Option<&'static dyn BaseManager>,
        timestamp: Option<&str>,
        errmsg: Option<&str>,
    ) {
        let manager =
            DirectiveEditorManager::new(directive_file_type, data_source, timestamp, errmsg);
        ui_harness::with(|harness| harness.add_frame(manager, true));
        manager.initialize();
        utilities::manager_stamp(
            manager.get_property_user_dir().as_deref(),
            manager.get_name().as_deref(),
        );
    }

    /// Java `openPeet(boolean, AxisID)`.
    pub fn open_peet_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_peet_string_boolean_axis_id(Some(PEET_META_DATA_NEW_TITLE), make_current, axis_id)
    }

    /// Java `openSerialSections(boolean, AxisID)`.
    pub fn open_serial_sections_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        self.open_serial_sections_string_boolean_axis_id(
            Some(SERIAL_SECTIONS_META_DATA_NEW_TITLE),
            make_current,
            axis_id,
        )
    }

    /// Java private `openJoin(File, boolean, AxisID)`.
    fn open_join_file_boolean_axis_id(
        &self,
        etomo_join_file: Option<&Path>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let Some(etomo_join_file) = etomo_join_file else {
            return self.open_join_boolean_axis_id(make_current, axis_id);
        };
        self.open_join_string_boolean_axis_id(
            Some(&utilities::java_io_file_get_absolute_path(
                &etomo_join_file.to_string_lossy(),
            )),
            make_current,
            axis_id,
        )
    }

    /// Java `openFrontPage(boolean, AxisID)`.
    pub fn open_front_page_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        let manager = FrontPageManager::new();
        self.set_manager(manager, axis_id, make_current)
    }

    /// Java `openFrontPage(boolean, AxisID, ImageFilenameStyle)`.
    pub fn open_front_page_boolean_axis_id_image_filename_style(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
        image_filename_style: Option<ImageFilenameStyle>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        self.close_default_window(axis_id);
        let manager = FrontPageManager::new_image_filename_style(image_filename_style);
        self.set_manager(manager, axis_id, make_current)
    }

    /// Java private `openParallel(File, boolean, AxisID)`.
    fn open_parallel_file_boolean_axis_id(
        &self,
        etomo_parallel_file: Option<&Path>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let Some(etomo_parallel_file) = etomo_parallel_file else {
            return self.open_parallel_boolean_axis_id(make_current, axis_id);
        };
        self.open_parallel_string_boolean_axis_id(
            Some(&utilities::java_io_file_get_absolute_path(
                &etomo_parallel_file.to_string_lossy(),
            )),
            make_current,
            axis_id,
        )
    }

    /// Java private `openBatchRunTomo(File, boolean, AxisID)`.
    fn open_batch_run_tomo_file_boolean_axis_id(
        &self,
        etomo_batch_run_tomo_file: Option<&Path>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let Some(etomo_batch_run_tomo_file) = etomo_batch_run_tomo_file else {
            return self.open_batch_run_tomo_boolean_axis_id(make_current, axis_id);
        };
        self.open_batch_run_tomo_string_boolean_axis_id(
            Some(&utilities::java_io_file_get_absolute_path(
                &etomo_batch_run_tomo_file.to_string_lossy(),
            )),
            make_current,
            axis_id,
        )
    }

    /// Java private `openPeet(File, boolean, AxisID)`.
    fn open_peet_file_boolean_axis_id(
        &self,
        etomo_peet_file: Option<&Path>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let Some(etomo_peet_file) = etomo_peet_file else {
            return self.open_peet_boolean_axis_id(make_current, axis_id);
        };
        self.open_peet_string_boolean_axis_id(
            Some(&utilities::java_io_file_get_absolute_path(
                &etomo_peet_file.to_string_lossy(),
            )),
            make_current,
            axis_id,
        )
    }

    /// Java private `openSerialSections(File, boolean, AxisID)`.
    fn open_serial_sections_file_boolean_axis_id(
        &self,
        etomo_serial_sections_file: Option<&Path>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let Some(etomo_serial_sections_file) = etomo_serial_sections_file else {
            return self.open_serial_sections_boolean_axis_id(make_current, axis_id);
        };
        self.open_serial_sections_string_boolean_axis_id(
            Some(&utilities::java_io_file_get_absolute_path(
                &etomo_serial_sections_file.to_string_lossy(),
            )),
            make_current,
            axis_id,
        )
    }

    /// Java private `openJoin(String, boolean, AxisID)`.
    fn open_join_string_boolean_axis_id(
        &self,
        etomo_join_file_name: Option<&str>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let manager: &'static JoinManager;
        if etomo_join_file_name.is_none()
            || etomo_join_file_name == Some(JOIN_META_DATA_NEW_FILE_TITLE)
        {
            manager = JoinManager::new(Some(""), axis_id);
            ui_harness::with(|harness| harness.set_enabled_new_join_menu_item(false));
        } else {
            manager = JoinManager::new(etomo_join_file_name, axis_id);
        }
        self.set_manager(manager, axis_id, make_current)
    }

    /// Java private `openParallel(String, boolean, AxisID)`.
    fn open_parallel_string_boolean_axis_id(
        &self,
        parallel_file_name: Option<&str>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let manager: &'static ParallelManager;
        if parallel_file_name.is_none() {
            manager = ParallelManager::new();
        } else if parallel_file_name == Some(parallel_meta_data::NEW_GENERIC_PARALLEL_PROCESS_TITLE)
        {
            manager = ParallelManager::new_with_dialog_type(DialogType::Parallel);
            ui_harness::with(|harness| harness.set_enabled_new_generic_parallel_menu_item(false));
        } else if parallel_file_name == Some(parallel_meta_data::NEW_ANISOTROPIC_DIFFUSION_TITLE) {
            manager = ParallelManager::new_with_dialog_type(DialogType::AnisotropicDiffusion);
            ui_harness::with(|harness| {
                harness.set_enabled_new_anisotropic_diffusion_menu_item(false)
            });
        } else {
            manager = ParallelManager::new_with_param_file(parallel_file_name);
        }
        self.set_manager(manager, axis_id, make_current)
    }

    /// Java private `openBatchRunTomo(String, boolean, AxisID)`.
    fn open_batch_run_tomo_string_boolean_axis_id(
        &self,
        batch_run_tomo_file_name: Option<&str>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let manager: &'static BatchRunTomoManager;
        if batch_run_tomo_file_name.is_none()
            || batch_run_tomo_file_name == Some(BATCH_RUN_TOMO_META_DATA_NEW_TITLE)
        {
            manager = BatchRunTomoManager::new();
            ui_harness::with(|harness| harness.set_enabled_new_batch_run_tomo_menu_item(false));
        } else {
            manager = BatchRunTomoManager::new_with_param_file_name(batch_run_tomo_file_name);
        }
        let key = self.set_manager(manager, axis_id, make_current);
        if !BaseManager::is_valid(manager) {
            self.close_current_manager(Some(AxisID::Only), false);
            return None;
        }
        key
    }

    /// Java private `openPeet(String, boolean, AxisID)`.
    fn open_peet_string_boolean_axis_id(
        &self,
        peet_file_name: Option<&str>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        // TODO(unit): needs etomo/PeetManager.java - the body is
        //   final PeetManager manager;
        //   if (peetFileName == null || peetFileName.equals(PeetMetaData.NEW_TITLE)) {
        //     manager = PeetManager.getInstance();
        //     UIHarness.INSTANCE.setEnabledNewPeetMenuItem(false);
        //   }
        //   else {
        //     manager = PeetManager.getInstance(peetFileName);
        //   }
        //   ManagerKey key = setManager(manager, axisID, makeCurrent);
        //   manager.display();
        //   if (!manager.isValid()) {
        //     closeCurrentManager(AxisID.ONLY, false);
        //     return null;
        //   }
        //   return key;
        // Without the manager no PEET interface is opened.
        let _ = (peet_file_name, make_current, axis_id);
        None
    }

    /// Java private `openSerialSections(String, boolean, AxisID)`.
    fn open_serial_sections_string_boolean_axis_id(
        &self,
        serial_sections_file_name: Option<&str>,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let manager: &'static SerialSectionsManager;
        if serial_sections_file_name.is_none()
            || serial_sections_file_name == Some(SERIAL_SECTIONS_META_DATA_NEW_TITLE)
        {
            manager = SerialSectionsManager::get_instance();
            ui_harness::with(|harness| harness.set_enabled_new_serial_sections_menu_item(false));
        } else {
            manager =
                SerialSectionsManager::get_instance_with_param_file_name(serial_sections_file_name);
        }
        let key = self.set_manager(manager, axis_id, make_current);
        manager.display();
        if !BaseManager::is_valid(manager) {
            self.close_current_manager(Some(AxisID::Only), false);
            return None;
        }
        key
    }

    /// Java private `setManager(BaseManager, AxisID, boolean)`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1010): before
    /// `initialize` the manager list is null and the Java throws
    /// NullPointerException; here the manager is not registered and null is
    /// returned.
    fn set_manager(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        make_current: bool,
    ) -> Option<Arc<Mutex<ManagerKey>>> {
        let unique_key: UniqueKey;
        {
            let mut manager_list = self.manager_list.lock().unwrap();
            let manager_list = manager_list.as_mut()?;
            unique_key =
                manager_list.add_with_name(manager.get_name().unwrap_or_default(), manager);
        }
        manager.set_manager_key(Some(unique_key));
        let manager_key = manager.get_manager_key();
        let key = manager_key.lock().unwrap().get_key().cloned();
        ui_harness::with(|harness| harness.add_window(Some(manager), axis_id, key.as_ref()));
        if make_current {
            ui_harness::with(|harness| {
                harness.select_window_menu_item_unique_key_boolean(key.as_ref(), true)
            });
            self.set_current_manager_manager_key_boolean_boolean(
                Some(Arc::clone(&manager_key)),
                true,
                true,
            );
        }
        Some(manager_key)
    }

    /// Java `openTomogram(boolean, AxisID)`.
    pub fn open_tomogram_boolean_axis_id(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
    ) -> Option<UniqueKey> {
        self.close_default_window(axis_id);
        let manager_key = self.open_tomogram_string_boolean_axis_id(
            Some(MetaData::get_new_file_title()),
            make_current,
            axis_id,
        );
        let manager_key = manager_key?;
        let key = manager_key.lock().unwrap().get_key().cloned();
        key
    }

    /// Java `openTomogramAndDoAutomation(boolean, AxisID, LocalArguments)`.
    pub fn open_tomogram_and_do_automation(
        &self,
        make_current: bool,
        axis_id: Option<AxisID>,
        local_arguments: Option<&LocalArguments>,
    ) {
        self.close_default_window(axis_id);
        let manager_key = self.open_tomogram_string_boolean_axis_id(
            Some(MetaData::get_new_file_title()),
            make_current,
            axis_id,
        );
        self.do_automation_manager_key_local_arguments(manager_key, local_arguments);
    }

    /// Java private `closeDefaultWindow(AxisID)`.  When etomo is run with no
    /// data file, it automatically opens a Setup Tomogram window.  This window
    /// should be closed if the user opens another window without adding data to
    /// the Setup Tomogram fields.  This is only true if the Setup Tomogram was
    /// opened as the default window.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1045-1046): with no
    /// manager list (before `initialize`) or no current manager the Java throws
    /// NullPointerException; here there is no default window to close then.
    fn close_default_window(&self, axis_id: Option<AxisID>) {
        let size = self
            .manager_list
            .lock()
            .unwrap()
            .as_ref()
            .map(UniqueHashedArray::size);
        if self.default_window.load(Ordering::SeqCst) && size == Some(1) {
            let manager = self.get_current_manager();
            if let Some(manager) = manager
                && manager.get_interface_type() == Some(InterfaceType::FrontPage)
            {
                // manager instanceof FrontPageManager: only FrontPageManager has
                // the FRONT_PAGE interface type.
                self.default_window.store(false, Ordering::SeqCst);
                self.close_current_manager(axis_id, false);
            }
        }
    }

    /// Java private `saveLogs()`.
    fn save_logs(&self) {
        let managers: Option<Vec<&'static dyn BaseManager>> = self
            .manager_list
            .lock()
            .unwrap()
            .as_ref()
            .map(|manager_list| {
                (0..manager_list.size())
                    .filter_map(|i| manager_list.get_at(i).copied())
                    .collect()
            });
        if let Some(managers) = managers {
            for manager in managers {
                manager.save_log();
            }
        }
    }

    /// Java `openManager(File, boolean, AxisID, UIComponent)`.
    ///
    /// The Java throws `IllegalStateException("null dataFile")` for a null file
    /// (a `&Path` cannot be null), and `IllegalStateException("unknown
    /// dataFile")` after reporting an unrecognised file; that is the `Err`.
    pub fn open_manager(
        &self,
        data_file: &Path,
        make_current: bool,
        axis_id: Option<AxisID>,
        ui_component: Option<&dyn UIComponent>,
    ) -> Result<(), String> {
        self.close_default_window(axis_id);
        let etomo_file_filter = EtomoFileFilter;
        if etomo_file_filter.accept(data_file) {
            self.open_tomogram_file_boolean_axis_id_ui_component(
                Some(data_file),
                make_current,
                axis_id,
                ui_component,
            );
            return Ok(());
        }
        let join_file_filter = JoinFileFilter::new();
        if join_file_filter.accept(data_file) {
            self.open_join_file_boolean_axis_id(Some(data_file), make_current, axis_id);
            return Ok(());
        }
        let parallel_file_filter = ParallelFileFilter::new();
        if parallel_file_filter.accept(data_file) {
            self.open_parallel_file_boolean_axis_id(Some(data_file), make_current, axis_id);
            return Ok(());
        }
        let batch_run_tomo_file_filter = BatchRunTomoFileFilter::new();
        if batch_run_tomo_file_filter.accept(data_file) {
            self.open_batch_run_tomo_file_boolean_axis_id(Some(data_file), make_current, axis_id);
            return Ok(());
        }
        let peet_file_filter = PeetFileFilter::new();
        if peet_file_filter.accept(data_file) {
            self.open_peet_file_boolean_axis_id(Some(data_file), make_current, axis_id);
            return Ok(());
        }
        let serial_sections_file_filter = SerialSectionsFileFilter::new();
        if serial_sections_file_filter.accept(data_file) {
            self.open_serial_sections_file_boolean_axis_id(Some(data_file), make_current, axis_id);
            return Ok(());
        }
        let message = format!(
            "Unknown file type {}.",
            utilities::java_io_file_get_name(&data_file.to_string_lossy())
        );
        let current_manager = self.get_current_manager();
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                current_manager,
                &message,
                "Unknown File Type",
                axis_id,
            )
        });
        Err("unknown dataFile".to_owned())
    }

    /// Java `openTomogram(File, boolean, AxisID, UIComponent)`.
    pub fn open_tomogram_file_boolean_axis_id_ui_component(
        &self,
        etomo_data_file: Option<&Path>,
        make_current: bool,
        axis_id: Option<AxisID>,
        ui_component: Option<&dyn UIComponent>,
    ) -> Option<UniqueKey> {
        match etomo_data_file {
            None => {
                return self.open_tomogram_boolean_axis_id(make_current, axis_id);
            }
            Some(etomo_data_file) => {
                let absolute_path =
                    utilities::java_io_file_get_absolute_path(&etomo_data_file.to_string_lossy());
                if etomo_data_file.exists() {
                    let manager_key = self.open_tomogram_string_boolean_axis_id(
                        Some(&absolute_path),
                        make_current,
                        axis_id,
                    );
                    let manager_key = manager_key?;
                    let key = manager_key.lock().unwrap().get_key().cloned();
                    return key;
                } else {
                    let message = format!("Dataset file {} does not exist.", absolute_path);
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                            None,
                            ui_component,
                            &message,
                            "Open Dataset Failed",
                            None,
                        )
                    });
                }
            }
        }
        None
    }

    /// Java `isOpen(UniqueKey)`.
    pub fn is_open(&self, manager_unique_key: Option<&UniqueKey>) -> bool {
        let manager_list = self.manager_list.lock().unwrap();
        let Some(manager_list) = manager_list.as_ref() else {
            return false;
        };
        manager_list.contains(manager_unique_key)
    }

    /// Java `closeManagers(List<UniqueKey>)`.
    pub fn close_managers(&self, manager_unique_key_list: Option<&[UniqueKey]>) {
        let Some(manager_unique_key_list) = manager_unique_key_list else {
            return;
        };
        let size = manager_unique_key_list.len();
        for i in 0..size {
            if self.set_current_manager_unique_key(Some(&manager_unique_key_list[i])) {
                self.close_current_manager(None, false);
            }
        }
    }

    /// Java `closeManager(AxisID, UniqueKey)`.  Close the manager indicated by
    /// managerUniqueKey.  Then go back to the current manager.
    pub fn close_manager(&self, axis_id: Option<AxisID>, manager_unique_key: Option<&UniqueKey>) {
        let Some(manager_unique_key) = manager_unique_key else {
            return;
        };
        let mut saved_current_manager_key: Option<Arc<Mutex<ManagerKey>>> = None;
        // Set the manager to be closed to the current manager.
        let current_manager_key = self.current_manager_key.lock().unwrap().clone();
        if current_manager_key.is_none()
            || !current_manager_key
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .equals_unique_key(Some(manager_unique_key))
        {
            saved_current_manager_key = current_manager_key;
            self.set_current_manager_unique_key(Some(manager_unique_key));
        }
        // Close the managerUniqueKey manager.
        self.close_current_manager(axis_id, false);
        // Got back to current manager.
        if saved_current_manager_key.is_some() {
            self.set_current_manager_manager_key_boolean_boolean(
                saved_current_manager_key,
                false,
                true,
            );
        }
    }

    /// Java `closeCurrentManager(AxisID, boolean)`.
    pub fn close_current_manager(&self, axis_id: Option<AxisID>, exiting: bool) -> bool {
        let current_manager = self.get_current_manager();
        let Some(current_manager) = current_manager else {
            return true;
        };
        if exiting {
            if !current_manager.exit_program(axis_id) {
                return false;
            }
        } else if !current_manager.close(axis_id) {
            return false;
        }
        let current_key = self
            .current_manager_key
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|manager_key| manager_key.lock().unwrap().get_key().cloned());
        if let Some(current_key) = &current_key
            && let Some(manager_list) = self.manager_list.lock().unwrap().as_mut()
        {
            manager_list.remove(current_key);
        }
        self.enable_open_manager_menu_item();
        ui_harness::with(|harness| harness.remove_window(current_key.as_ref()));
        *self.current_manager_key.lock().unwrap() = None;
        let (size, first_key) = {
            let manager_list = self.manager_list.lock().unwrap();
            match manager_list.as_ref() {
                Some(manager_list) => (manager_list.size(), manager_list.get_key(0).cloned()),
                None => (0, None),
            }
        };
        if size == 0 {
            ui_harness::with(|harness| harness.remove_window(None));
            // mainFrame.setWindowMenuLabels(controllerList);
            ui_harness::with(|harness| {
                harness.set_current_manager_base_manager_unique_key(None, None)
            });
            ui_harness::with(|harness| harness.select_window_menu_item_unique_key(None));
            self.set_current_manager_manager_key_boolean_boolean(None, false, true);
            return true;
        }
        self.set_current_manager_unique_key(first_key.as_ref());
        ui_harness::with(|harness| harness.select_window_menu_item_unique_key(first_key.as_ref()));
        true
    }

    /// Java private `enableOpenManagerMenuItem()`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1202): with no
    /// current manager key (`renameCurrentManager` with no manager) the Java
    /// throws NullPointerException; here no menu item changes.
    fn enable_open_manager_menu_item(&self) {
        let key = self
            .current_manager_key
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|manager_key| manager_key.lock().unwrap().get_key().cloned());
        let Some(key) = key else {
            return;
        };
        if key.get_name() == MetaData::get_new_file_title() {
            ui_harness::with(|harness| harness.set_enabled_new_tomogram_menu_item(true));
        } else if key.get_name() == JOIN_META_DATA_NEW_FILE_TITLE {
            ui_harness::with(|harness| harness.set_enabled_new_join_menu_item(true));
        } else if key.get_name() == parallel_meta_data::NEW_GENERIC_PARALLEL_PROCESS_TITLE {
            ui_harness::with(|harness| harness.set_enabled_new_generic_parallel_menu_item(true));
        } else if key.get_name() == parallel_meta_data::NEW_ANISOTROPIC_DIFFUSION_TITLE {
            ui_harness::with(|harness| {
                harness.set_enabled_new_anisotropic_diffusion_menu_item(true)
            });
        } else if key.get_name() == BATCH_RUN_TOMO_META_DATA_NEW_TITLE {
            ui_harness::with(|harness| harness.set_enabled_new_batch_run_tomo_menu_item(true));
        } else if key.get_name() == PEET_META_DATA_NEW_TITLE {
            ui_harness::with(|harness| harness.set_enabled_new_peet_menu_item(true));
        } else if key.get_name() == SERIAL_SECTIONS_META_DATA_NEW_TITLE {
            ui_harness::with(|harness| harness.set_enabled_new_serial_sections_menu_item(true));
        }
    }

    /// Java `setTestFailed(boolean)`.  Set failure while doing UI testing to
    /// prevent popups as the test tries to end.  The Java throws
    /// `IllegalStateException("test=" + ARGUMENTS.isTest())` outside a test;
    /// that is the `Err`.
    pub fn set_test_failed(&self, input: bool) -> Result<(), String> {
        let is_test = ARGUMENTS.lock().unwrap().is_test();
        if !is_test {
            return Err(format!("test={}", is_test));
        }
        TEST_FAILED.store(input, Ordering::SeqCst);
        Ok(())
    }

    /// Java `isTestFailed()`.  Returns true if this is a UI test and it has
    /// failed.
    pub fn is_test_failed(&self) -> bool {
        TEST_FAILED.load(Ordering::SeqCst)
    }

    /// Java `exitProgram(AxisID)`.  Close all managers, unless the user
    /// prevents it.  If all managers are closed, stop threads, save data and
    /// return true.  To guarantee that etomo can exit during a failure, catch
    /// all uncaught Exceptions and Errors and return true.  Returns true if
    /// etomo can exit, false if the user prevented exit.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1256-1259): with
    /// managers listed but no current manager, `closeCurrentManager` returns
    /// true without closing anything and the loop never ends.  Here the first
    /// listed manager is made current first; if none can be, the loop stops.
    pub fn exit_program(&self, axis_id: Option<AxisID>) -> bool {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.save_logs();
            let has_manager_list = self.manager_list.lock().unwrap().is_some();
            if has_manager_list {
                loop {
                    let (size, first_key) = {
                        let manager_list = self.manager_list.lock().unwrap();
                        let manager_list = manager_list.as_ref().unwrap();
                        (manager_list.size(), manager_list.get_key(0).cloned())
                    };
                    if size == 0 {
                        break;
                    }
                    if self.get_current_manager().is_none()
                        && !self.set_current_manager_unique_key(first_key.as_ref())
                    {
                        break;
                    }
                    if !self.close_current_manager(axis_id, true) {
                        return false;
                    }
                }
            }
            let utility_thread = self.utility_thread.lock().unwrap().clone();
            if let Some(utility_thread) = utility_thread {
                utility_thread.stop();
            }
            ProcessRestarter::stop();
            IntermittentBackgroundProcess::stop();
            if self.is_memory_available() {
                // Should we close the 3dmod windows
                // Save the current window size to the user config
                let size = ui_harness::with(|harness| harness.get_size(None));
                self.with_user_configuration_mut(|user_config| {
                    user_config.set_main_window_width(size.width);
                    user_config.set_main_window_height(size.height);
                });
                // Write out the user configuration data
                let mut parameter_store = self.parameter_store.lock().unwrap();
                if let Some(parameter_store) = parameter_store.as_mut() {
                    let _guard = USER_CONFIG.lock.lock();
                    if let Err(e) = parameter_store.save(Some(&USER_CONFIG.value)) {
                        eprintln!("Exception upon exiting.  {}", e);
                        eprintln!("{}", e);
                    }
                }
                return true;
            }
            true
        }));
        match result {
            Ok(result) => result,
            Err(payload) => {
                // catch (Throwable e)
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "null".to_owned());
                eprintln!("Exception upon exiting.  {}", message);
                eprintln!("{}", message);
                true
            }
        }
    }

    /// Java `getParameterStore()`.  The store is behind the director's lock:
    /// hold the guard only for the statement that uses it (the Java object is
    /// null before `loadUserConfiguration`).
    pub fn get_parameter_store(&self) -> MutexGuard<'_, Option<ParameterStore>> {
        self.parameter_store.lock().unwrap()
    }

    /// Java `renameCurrentManager(String)`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1293-1294): with no
    /// current manager the Java throws NullPointerException; here nothing is
    /// renamed.
    pub fn rename_current_manager(&self, manager_name: impl Into<String>) {
        self.enable_open_manager_menu_item();
        let current_manager_key = self.current_manager_key.lock().unwrap().clone();
        let Some(current_manager_key) = current_manager_key else {
            return;
        };
        let old_manager_key = current_manager_key.lock().unwrap().get_key().cloned();
        let Some(old_key) = old_manager_key.clone() else {
            return;
        };
        let new_key = self
            .manager_list
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|manager_list| manager_list.rekey_with_name(&old_key, manager_name.into()));
        current_manager_key.lock().unwrap().set_key(new_key.clone());
        ui_harness::with(|harness| {
            harness.rename_window(old_manager_key.as_ref(), new_key.as_ref())
        });
    }

    /// Java `getUserConfiguration()` followed by a read of the configuration.
    /// See the module comment.
    pub fn with_user_configuration<R>(&self, f: impl FnOnce(&UserConfiguration) -> R) -> R {
        let _guard = USER_CONFIG.lock.lock();
        let user_config = USER_CONFIG.value.borrow();
        f(&user_config)
    }

    /// Java `getUserConfiguration()` followed by a change to the
    /// configuration.  See the module comment.
    pub fn with_user_configuration_mut<R>(&self, f: impl FnOnce(&mut UserConfiguration) -> R) -> R {
        let _guard = USER_CONFIG.lock.lock();
        let mut user_config = USER_CONFIG.value.borrow_mut();
        f(&mut user_config)
    }

    /// Java private final `printUsageMessage()`.
    fn print_usage_message(&self) {
        println!("Usage: etomo [options] [data files]\n");
        Arguments::print_help_message();
    }

    /// Java private `setUserPreferences()`.  Set the user preferences.
    fn set_user_preferences(&self) {
        let headless = ARGUMENTS.lock().unwrap().is_headless();
        if headless {
            return;
        }
        // ToolTipManager.sharedInstance().setInitialDelay(
        //   USER_CONFIG.getToolTipsInitialDelay()) and .setDismissDelay(
        //   USER_CONFIG.getToolTipsDismissDelay()): Swing's tooltip timing (module
        //   comment).
        let (font_family, font_size, native_look_and_feel, advanced_dialogs) = self
            .with_user_configuration(|user_config| {
                (
                    user_config.get_font_family(),
                    user_config.get_font_size(),
                    user_config.get_native_look_and_feel(),
                    user_config.get_advanced_dialogs(),
                )
            });
        self.set_ui_font(font_family.as_deref(), font_size);
        self.set_look_and_feel(native_look_and_feel);
        self.is_advanced.store(advanced_dialogs, Ordering::SeqCst);
        UIParameters::create_instance(font_size as f64);
        // CpuAdoc.INSTANCE.setUserConfig(userConfig.getParallelProcessing(), userConfig
        // .getCpus());
    }

    /// Java private `setLookAndFeel(boolean)`.  Sets the look and feel for the
    /// program: the host os look and feel when `native_look_and_feel`, the
    /// Metal look and feel otherwise.
    fn set_look_and_feel(&self, native_look_and_feel: bool) {
        let debug = self.debug.load(Ordering::SeqCst);
        let look_and_feel_class_name: &str;

        // UIManager.LookAndFeelInfo plaf[] = UIManager.getInstalledLookAndFeels();
        // for(int i = 0; i < plaf.length; i++) {
        // System.err.println(plaf[i].getClassName());
        // }
        if debug {
            eprintln!();
        }
        let os_name = utilities::java_lang_system_get_property_os_name();
        if debug {
            eprintln!("os.name: {}", os_name);
        }
        if native_look_and_feel {
            if os_name.starts_with("Mac OS X") {
                look_and_feel_class_name = "apple.laf.AquaLookAndFeel";
                if debug {
                    eprintln!("Setting AquaLookAndFeel");
                }
            } else if os_name.starts_with("Windows") {
                look_and_feel_class_name = "com.sun.java.swing.plaf.windows.WindowsLookAndFeel";
                if debug {
                    eprintln!("Setting WindowsLookAndFeel");
                }
            } else {
                look_and_feel_class_name = "com.sun.java.swing.plaf.motif.MotifLookAndFeel";
                if debug {
                    eprintln!("Setting MotifLookAndFeel");
                }
            }
        } else {
            // UIManager.getCrossPlatformLookAndFeelClassName()
            look_and_feel_class_name = "javax.swing.plaf.metal.MetalLookAndFeel";
            if debug {
                eprintln!("Setting MetalLookAndFeel");
            }
        }
        // UIManager.setLookAndFeel(lookAndFeelClassName), reporting
        // "Could not set <name> look and feel" on failure: Swing only (module
        // comment); the Slint frontend has its own style.
        let _ = look_and_feel_class_name;

        // print look and feel info: "\nLook and feel defaults:" and every
        // UIManager.getLookAndFeelDefaults() string entry.  JVM-only.
    }

    /// Java private `setUIFont(String, int)`.  Sets the default font for all
    /// Swing components: every `FontUIResource` in the `UIManager` defaults is
    /// replaced with one of this family and size, keeping its style.  Swing
    /// only (module comment); the settings dialog reads the family and size
    /// back from the user configuration.
    fn set_ui_font(&self, font_family: Option<&str>, font_size: i32) {
        // ex.
        // setUIFont (new javax.swing.plaf.FontUIResource("Serif",Font.ITALIC,12));
        // Taken from: http://www.rgagnon.com/javadetails/java-0335.html
        let _ = (font_family, font_size);
    }

    /// Java `getArguments()`.  Java returns the `ARGUMENTS` static; hold the
    /// guard only for the statement that uses it.
    pub fn get_arguments(&self) -> MutexGuard<'static, Arguments> {
        ARGUMENTS.lock().unwrap()
    }

    /// Java `getIMODDirectory()`.  Return the IMOD directory (a copy, as an
    /// absolute path).
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1424): before
    /// `initialize` the directory is null and the Java throws
    /// NullPointerException; here it is `None`.
    pub fn get_imod_directory(&self) -> Option<PathBuf> {
        let imod_directory = self.imod_directory.lock().unwrap().clone()?;
        // Return a copy of the IMODDirectory object
        Some(PathBuf::from(utilities::java_io_file_get_absolute_path(
            &imod_directory.to_string_lossy(),
        )))
    }

    /// Java `getIMODBinPath()`.  `None` before `initialize` (see
    /// `getIMODDirectory`).
    pub fn get_imod_bin_path(&self) -> Option<String> {
        let imod_directory = self.get_imod_directory()?;
        Some(format!(
            "{}{}bin{}",
            utilities::java_io_file_get_absolute_path(&imod_directory.to_string_lossy()),
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR
        ))
    }

    /// Java `getPythonScriptPath()`.  If the python installation is python 3
    /// running under cygwin, returns the IMOD bin path converted to a
    /// cygwin-style path.  Otherwise returns IMOD bin path as is.  `None`
    /// before `initialize` (see `getIMODDirectory`).
    pub fn get_python_script_path(&self) -> Option<String> {
        let python_script_path = self.python_script_path.lock().unwrap().clone();
        if python_script_path.is_some() {
            return python_script_path;
        }
        let imod_bin_path = self.get_imod_bin_path()?;
        if !utilities::is_python3() {
            return Some(imod_bin_path);
        }
        let python_script_path =
            utilities::convert_windows_abs_file_path_to_cygdrive_path(Some(&imod_bin_path));
        *self.python_script_path.lock().unwrap() = python_script_path.clone();
        if python_script_path.is_none() {
            return Some(imod_bin_path);
        }
        python_script_path
    }

    /// Java `getPythonScriptPath(String)`.  If the python installation is
    /// python 3 running under cygwin, returns the IMOD bin path converted to a
    /// cygwin-style path.  Otherwise returns imodBinPath.
    pub fn get_python_script_path_string(&self, imod_bin_path: &str) -> String {
        if !utilities::is_python3() {
            return imod_bin_path.to_owned();
        }
        let python_script_path =
            utilities::convert_windows_abs_file_path_to_cygdrive_path(Some(imod_bin_path));
        if let Some(python_script_path) = python_script_path {
            return python_script_path;
        }
        imod_bin_path.to_owned()
    }

    /// Java `getNumberOfProcessorsWindows()`.
    pub fn get_number_of_processors_windows(&self) -> Option<ConstEtomoNumber> {
        self.number_of_processors_windows
            .lock()
            .unwrap()
            .as_ref()
            .map(|number| number.base.clone())
    }

    /// Java `getIMODCalibDirectory()`.  Return the IMOD calibration directory
    /// (a copy, as an absolute path).  `None` before `initialize` (fixed
    /// NullPointerException, as `getIMODDirectory`).
    pub fn get_imod_calib_directory(&self) -> Option<PathBuf> {
        let imod_calib_directory = self.imod_calib_directory.lock().unwrap().clone()?;
        // Return a copy of the IMODDirectory object
        Some(PathBuf::from(utilities::java_io_file_get_absolute_path(
            &imod_calib_directory.to_string_lossy(),
        )))
    }

    /// Java `getAdvanced()`.  Get the current advanced state.
    pub fn get_advanced(&self) -> bool {
        self.is_advanced.load(Ordering::SeqCst)
    }

    /// Java `getSettingsParameters()`.
    pub fn get_settings_parameters(&self) {
        let settings_dialog = self.settings_dialog.get();
        if let Some(settings_dialog) = settings_dialog {
            let appearance_setting_changed = self.with_user_configuration(|user_config| {
                settings_dialog.is_appearance_setting_changed(user_config)
            });
            if appearance_setting_changed {
                let current_manager = self.get_current_manager();
                ui_harness::with(|harness| {
                    harness.open_info_message_dialog_base_manager_string_string_axis_id(
                        current_manager,
                        "You must exit from Etomo and re-run it for this change to fully take \
                         effect.",
                        "Settings",
                        Some(AxisID::First),
                    )
                });
            }
            self.with_user_configuration_mut(|user_config| {
                settings_dialog.get_parameters(user_config)
            });
            self.set_user_preferences();
            ui_harness::with(|harness| harness.repaint_window(None, Some(AxisID::First)));
        }
    }

    /// Java private `getPropertyUserDir()`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1506): a current
    /// manager key whose manager is no longer listed makes the Java throw
    /// NullPointerException; here the property user dir is then null.
    fn get_property_user_dir(&self) -> Option<String> {
        let has_current_manager_key = self.current_manager_key.lock().unwrap().is_some();
        if !has_current_manager_key {
            return self.original_user_dir.lock().unwrap().clone();
        }
        self.get_current_manager()
            .and_then(|current_manager| current_manager.get_property_user_dir())
    }

    /// Java `openSettingsDialog()`.  Open up the settings dialog box.
    ///
    /// Upstream bug fixed in translation (SettingsDialog.java:371): with no
    /// current manager the Java builds the dialog with a null manager and
    /// `setParameters` dereferences it (NullPointerException) whenever
    /// parallel processing is not selected.  The translated dialog requires a
    /// manager (its `TemplatePanel` does), so with no current manager no
    /// dialog is opened.
    pub fn open_settings_dialog(&self) {
        // Open the dialog in the appropriate mode for the current state of
        // processing
        if !self.settings_dialog.is_some() {
            let Some(current_manager) = self.get_current_manager() else {
                return;
            };
            let property_user_dir = self.get_property_user_dir();
            let settings_dialog =
                SettingsDialog::get_instance(current_manager, property_user_dir.as_deref());
            self.settings_dialog
                .set(Some(std::rc::Rc::clone(&settings_dialog)));
            self.with_user_configuration(|user_config| settings_dialog.set_parameters(user_config));
            let current_manager = self.get_current_manager();
            // Dimension frmSize = UIHarness.INSTANCE.getSize(getCurrentManager()) (unused)
            let _frm_size = ui_harness::with(|harness| harness.get_size(current_manager));
            let loc = ui_harness::with(|harness| harness.get_location(current_manager));
            settings_dialog.set_location(loc.x, loc.y + 20);
            settings_dialog.set_modal(false);
        }
        if let Some(settings_dialog) = self.settings_dialog.get() {
            settings_dialog.set_visible(true);
        }
    }

    /// Java `saveSettingsDialog()`.
    ///
    /// Upstream bug fixed in translation (EtomoDirector.java:1529): before the
    /// user configuration is loaded the parameter store is null and the Java
    /// throws NullPointerException; here nothing is saved.
    pub fn save_settings_dialog(&self) {
        let debug = self.debug.load(Ordering::SeqCst);
        let error = {
            let mut parameter_store = self.parameter_store.lock().unwrap();
            let Some(parameter_store) = parameter_store.as_mut() else {
                return;
            };
            let _guard = USER_CONFIG.lock.lock();
            parameter_store
                .save(Some(&USER_CONFIG.value))
                .err()
                .map(|e| {
                    (
                        e,
                        parameter_store
                            .get_absolute_path()
                            .unwrap_or_else(|| "null".to_owned()),
                    )
                })
        };
        if let Some((e, absolute_path)) = error {
            // catch (LogFileException e) / catch (IOException e)
            if debug {
                eprintln!("{}", e);
            }
            eprintln!("java.io.IOException: {}", e);
            let message = format!(
                "Unable to save or write preferences to {}.\n{}",
                absolute_path, e
            );
            let current_manager = self.get_current_manager();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    current_manager,
                    &message,
                    "Etomo Error",
                )
            });
        }
        // catch (LockException e) {}: the translated ParameterStore takes no lock.
    }

    /// Java `closeSettingsDialog()`.
    pub fn close_settings_dialog(&self) {
        if let Some(settings_dialog) = self.settings_dialog.get() {
            settings_dialog.dispose();
        }
    }

    /// Java `getAvailableMemory()`.
    ///
    /// The Java is the JVM heap's `maxMemory() - totalMemory() + freeMemory()`,
    /// the room left before the heap limit.  A native program has no heap
    /// limit; the host's available physical memory is the room it has.
    pub fn get_available_memory(&self) -> i64 {
        // System.err.println("max= " + Runtime.getRuntime().maxMemory());
        // System.err.println("total=" + Runtime.getRuntime().totalMemory());
        // System.err.println("free= " + Runtime.getRuntime().freeMemory());
        #[cfg(unix)]
        {
            // SAFETY: `sysconf` reads a system constant.
            let pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if pages <= 0 || page_size <= 0 {
                0
            } else {
                (pages as i64).saturating_mul(page_size as i64)
            }
        }
        #[cfg(not(unix))]
        {
            0
        }
    }

    /// Java `isImodBriefHeader()`.
    pub fn is_imod_brief_header(&self) -> bool {
        self.imod_brief_header.load(Ordering::SeqCst)
    }

    /// Java `isMemoryAvailable()`.
    ///
    /// The Java opens its warning with `UIHarness` on whatever thread calls it
    /// (the utility thread among them); off the event dispatch thread the
    /// warning is posted to it.
    pub fn is_memory_available(&self) -> bool {
        let available_memory = self.get_available_memory();
        // Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory():
        // the physical memory in use (see getAvailableMemory).
        #[cfg(unix)]
        let used_memory: i64 = {
            // SAFETY: `sysconf` reads a system constant.
            let physical_pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
            let available_pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            ((physical_pages - available_pages) as i64).saturating_mul(page_size as i64)
        };
        #[cfg(not(unix))]
        let used_memory: i64 = 0;
        // System.out.println();
        // System.out.println("Available memory = " + availableMemory);
        // System.out.println("Memory in use = " + usedMemory);
        // System.out.println();
        let display_memory = ARGUMENTS.lock().unwrap().is_display_memory();
        if display_memory {
            eprintln!("Available memory = {}", available_memory);
            eprintln!("Memory in use    = {}", used_memory);
        }
        // Check to see if the memory has been made available up to the memory limit.
        // SGI doesn't make all the memory available up to the memory limit until it
        // needs to.
        // Memory limit is adjusted down because availableMemory never matches
        // javaMemoryLimit.
        // Old code for SGI
        /*
         * if (javaMemoryLimit.isNull() || availableMemory + usedMemory >=
         * javaMemoryLimit.getLong() - (MIN_AVAILABLE_MEMORY_REQUIRED * 3)) { //Check
         * available memory if (availableMemory < MIN_AVAILABLE_MEMORY_REQUIRED) { //send
         * message once per memory problem if (!outOfMemoryMessage) {
         * UIHarness.INSTANCE.openMessageDialog(
         * "WARNING:  Ran out of memory.  Changes to the .edf file and/or" +
         * " comscript files may not be saved." +
         * "\nPlease close open windows or exit Etomo.", "Out of Memory"); }
         * outOfMemoryMessage = true; return false; } }
         */
        if available_memory as f64 <= MIN_AVAILABLE_MEMORY_REQUIRED {
            if !self.out_of_memory_message.load(Ordering::SeqCst) {
                let message = "WARNING:  Ran out of memory.  Changes to the .edf file and/or \
                               comscript files may not be saved.\nPlease close open windows \
                               or exit Etomo.";
                let current_manager = self.get_current_manager();
                if event_queue::is_dispatch_thread() {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            current_manager,
                            message,
                            "Out of Memory",
                        )
                    });
                } else {
                    ui_harness::post_message_dialog(
                        current_manager,
                        message.to_owned(),
                        "Out of Memory".to_owned(),
                        None,
                    );
                }
                self.out_of_memory_message.store(true, Ordering::SeqCst);
            }
            return false;
        }
        // memory problem is gone - reset message
        self.out_of_memory_message.store(false, Ordering::SeqCst);
        true
    }

    /// Java `getHomeDirectory()`.  Return the users home directory environment
    /// variable HOME or an empty string if it doesn't exist.
    pub fn get_home_directory(&self) -> &str {
        &self.home_directory
    }
}

// Java private static final class `Setup implements Runnable`: its `run()` is
// `setup()`; `main` queues `EtomoDirector::setup` directly.

/// Java private final inner class `UtilityThread implements Runnable`.  Saves
/// the managers' logs every `autoSaveLogInterval` minutes and, with
/// `-memory`, reports the memory every `displayMemoryInterval` minutes.  Its
/// outer instance is [`INSTANCE`].
struct UtilityThread {
    /// Java private final `autoSaveLogInterval = 1` (in minutes).
    auto_save_log_interval: i32,
    /// Java private final `autoSaveLogEvery`.
    auto_save_log_every: i32,
    /// Java private final `displayMemory`.
    display_memory: bool,
    /// Java private final `displayMemoryInterval`.
    display_memory_interval: i32,
    /// Java private final `displayMemoryEvery`.
    display_memory_every: i32,
    /// Java private final `sleep` (in minutes).
    sleep: i32,

    /// Java private `stop`, initially false.
    stop: AtomicBool,
    /// Java private `utilityThread`, initially null.
    utility_thread: Mutex<Option<std::thread::Thread>>,
    /// Rust-only: the interrupt flag `Thread.interrupt()` sets and
    /// `Thread.sleep` consumes.
    interrupted: AtomicBool,
}

impl UtilityThread {
    /// Java private `UtilityThread()`.
    fn new() -> UtilityThread {
        let auto_save_log_interval = 1;
        let display_memory_interval = ARGUMENTS.lock().unwrap().get_display_memory_interval();
        let display_memory =
            ARGUMENTS.lock().unwrap().is_display_memory() && display_memory_interval >= 1;
        let sleep;
        let auto_save_log_every;
        let display_memory_every;
        if !display_memory || display_memory_interval == auto_save_log_interval {
            // sleep for five minutes
            sleep = auto_save_log_interval;
            // run autosave after every sleep
            auto_save_log_every = 1;
            display_memory_every = 1; // ignored if !displayMemory
        } else {
            // This could be more complicated to allow longer sleeps, but I don't
            // think its worth it.
            sleep = 1;
            // run autosave after 5 one-minute sleeps
            auto_save_log_every = auto_save_log_interval;
            display_memory_every = display_memory_interval;
        }
        UtilityThread {
            auto_save_log_interval,
            auto_save_log_every,
            display_memory,
            display_memory_interval,
            display_memory_every,
            sleep,
            stop: AtomicBool::new(false),
            utility_thread: Mutex::new(None),
            interrupted: AtomicBool::new(false),
        }
    }

    /// Java `run()`.
    fn run(&self) {
        let debug = INSTANCE.debug.load(Ordering::SeqCst);
        *self.utility_thread.lock().unwrap() = Some(std::thread::current());
        if self.display_memory {
            if debug {
                eprintln!(
                    "{}",
                    utilities::java_util_date_to_string(
                        utilities::java_lang_system_current_time_millis()
                    )
                );
            }
            INSTANCE.is_memory_available();
        }
        let mut auto_save_log_count = 0;
        let mut display_memory_count = 0;
        while !self.stop.load(Ordering::SeqCst) {
            // Thread.sleep(1000 * 60 * sleep), which an interrupt ends with
            // InterruptedException.
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(1000 * 60 * self.sleep as u64);
            loop {
                if self.interrupted.swap(false, Ordering::SeqCst) {
                    if debug {
                        eprintln!("sleep interrupted");
                    }
                    eprintln!("java.lang.InterruptedException: sleep interrupted");
                    break;
                }
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                std::thread::park_timeout(deadline - now);
            }
            if self.display_memory {
                display_memory_count += 1;
                if display_memory_count == self.display_memory_every {
                    display_memory_count = 0;
                    eprintln!(
                        "{}",
                        utilities::java_util_date_to_string(
                            utilities::java_lang_system_current_time_millis()
                        )
                    );
                    INSTANCE.is_memory_available();
                }
            }
            auto_save_log_count += 1;
            if auto_save_log_count == self.auto_save_log_every {
                auto_save_log_count = 0;
                INSTANCE.save_logs();
            }
        }
        *self.utility_thread.lock().unwrap() = None;
    }

    /// Java private final `stop()`.
    fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let utility_thread = self.utility_thread.lock().unwrap().clone();
        if let Some(utility_thread) = utility_thread {
            // utilityThread.interrupt()
            self.interrupted.store(true, Ordering::SeqCst);
            utility_thread.unpark();
        }
    }
}
