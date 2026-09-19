//! Translation boundary for `IMOD/Etomo/src/etomo/EtomoDirector.java`.
//!
//! Swing window/manager methods remain explicit JVM GUI boundaries.  This unit
//! maps the director's launcher, environment, IMOD-directory, calibration, and
//! memory-limit logic used before batch/headless automation reaches that boundary.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::application_manager::ApplicationManager;
use super::arguments::Arguments;
use super::base_manager::BaseManager;
use super::batch_run_tomo_manager::BatchRunTomoManager;
use super::directive_editor_manager::DirectiveEditorManager;
use super::front_page_manager::FrontPageManager;
use super::join_manager::JoinManager;
use super::manager_key::ManagerKey;
use super::parallel_manager::ParallelManager;
use super::serial_sections_manager::SerialSectionsManager;
use super::storage::etomo_file_filter::EtomoFileFilter;
use super::storage::join_file_filter::JoinFileFilter;
use super::storage::parameter_store::ParameterStore;
use super::storage::peet_file_filter::PeetFileFilter;
use super::storage::serial_sections_file_filter::SerialSectionsFileFilter;
use super::storage::storable::Storable;
use super::tools_manager::ToolsManager;
use super::r#type::axis_id::AxisID;
use super::r#type::data_file_type::DataFileType;
use super::r#type::dialog_type::DialogType;
use super::r#type::directive_file_type::DirectiveFileType;
use super::ui::swing::etomo_menu::ToolType;
use super::ui::swing::settings_dialog::{SettingsDialog, UserConfigurationValues};
use super::util::unique_hashed_array::UniqueHashedArray;
use super::util::unique_key::UniqueKey;

pub const SCALE_IMAGES_ABOVE_FONT_SIZE: i32 = 14;
pub const SCALE_IMAGES_BELOW_FONT_SIZE: i32 = 11;
pub const DEFAULT_TO_STANDARD_IMAGE_FILE_NAMES: bool = true;
pub const USER_CONFIG_FILE_EXT: &str = ".etomo";
pub const IMOD_DIR_ENV_VAR: &str = "IMOD_DIR";
pub const SOURCE_ENV_VAR: &str = "IMOD_UITEST_SOURCE";
pub const NEW_MESSAGE_PARSER: bool = true;
pub const MIN_AVAILABLE_MEMORY_REQUIRED: f64 = 2. * 1024. * 1024.;
pub const NUMBER_STORABLES: i32 = 2;

/// Java `INSTANCE` (`EtomoDirector.java:80`):
/// `public static final EtomoDirector INSTANCE = new EtomoDirector();`.
///
/// Java initialises the singleton in a class initialiser; Rust needs a lock because the
/// director's fields are mutable and `Arguments::is_debug` takes `&mut self`.  The Java
/// field is not synchronised either, so the lock is a Rust representation requirement,
/// not added behaviour.
pub static INSTANCE: std::sync::LazyLock<std::sync::Mutex<EtomoDirector>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(EtomoDirector::new()));

/// Java `ARGUMENTS` (`EtomoDirector.java:81`):
/// `public static final Arguments ARGUMENTS = new Arguments();`.  This is the object
/// `EtomoDirector.INSTANCE.getArguments()` returns (`EtomoDirector.java:1415`).
pub static ARGUMENTS: std::sync::LazyLock<std::sync::Mutex<Arguments>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Arguments::new()));

/// Java `FILE_INFO_DIAGNOSTICS` (`EtomoDirector.java:97`).
pub const FILE_INFO_DIAGNOSTICS: bool = false;

/// Java `FILE_INFO_CLEAN_PRINT_LABEL` (`EtomoDirector.java:94`).
pub const FILE_INFO_CLEAN_PRINT_LABEL: &str = "File Info";

/// Fields of Java `EtomoDirector` (`EtomoDirector.java:76`).
#[derive(Default)]
pub struct EtomoDirector {
    pub home_directory: String,
    pub original_user_dir: Option<String>,
    /// Native boundary for Java's singleton `UserConfiguration`: its concrete
    /// typed settings model is translated at the settings/storage layer, while
    /// the director owns lifecycle, location and the widely-used appearance bits.
    user_preference_loaded: bool,
    user_font_size: i32,
    user_configuration_path: Option<PathBuf>,
    parameter_store: Option<ParameterStore>,
    user_configuration: UserConfigurationValues,
    settings_dialog: Option<SettingsDialog>,
    native_look_and_feel: bool,
    ui_font_family: String,
    maintain_etomo: bool,
    pub test_failed: bool,
    pub java_memory_limit: i64,
    pub imod_brief_header: bool,
    pub number_of_processors_windows: Option<i64>,
    pub is_advanced: bool,
    pub debug: bool,
    pub imod_directory: Option<PathBuf>,
    pub imod_calib_directory: Option<PathBuf>,
    pub python_script_path: Option<String>,
    /// Java state field `currentManagerKey`, initially null.
    current_manager_key: Option<Arc<Mutex<ManagerKey>>>,
    /// Java field `defaultWindow`, initially false.
    default_window: bool,
    /// Java field `managerList`, initialized in `initialize()`.
    manager_list: Option<UniqueHashedArray<&'static dyn BaseManager>>,
    pub headless: bool,
    pub test: bool,
    pub help: bool,
    pub directive: bool,
    pub self_test: bool,
}

impl EtomoDirector {
    /// Matches Java constructor `EtomoDirector()` (`EtomoDirector.java:129`).
    pub fn new() -> Self {
        Self {
            home_directory: std::env::var("HOME").unwrap_or_default(),
            user_font_size: 12,
            ..Self::default()
        }
    }

    /// Matches Java `main(String[])` (`EtomoDirector.java:133`) on the non-GUI path.
    pub fn main(&mut self, arguments: &[String]) -> Result<(), String> {
        let mut parsed_arguments = Arguments::new();
        parsed_arguments.parse(arguments);
        self.debug = parsed_arguments.is_debug();
        self.headless = parsed_arguments.is_headless();
        self.test = parsed_arguments.is_test();
        self.help = parsed_arguments.is_help();
        self.directive = parsed_arguments.is_directive();
        self.self_test = parsed_arguments.is_self_test();
        eprintln!("Running Etomo...");
        eprintln!("Arguments:");
        for argument in arguments {
            eprint!("{argument} ");
        }
        eprintln!("\n");
        if self.headless { self.setup() } else { Ok(()) }
    }

    /// GUI invocation of Java `main(String[])` (`EtomoDirector.java:133-159`).
    ///
    /// Java queues `Setup` with `SwingUtilities.invokeLater` for a non-headless
    /// invocation.  Slint has no separate queued setup object: this method runs
    /// the same setup synchronously, makes the translated main frame visible,
    /// then enters Slint's event loop.  This method is only compiled for the
    /// optional `etomo-gui` command; `main` above deliberately retains the
    /// existing non-GUI `etomo` behaviour.
    #[cfg(feature = "gui")]
    pub fn main_gui(&mut self, arguments: &[String]) -> Result<(), String> {
        let mut parsed_arguments = Arguments::new();
        parsed_arguments.parse(arguments);
        self.debug = parsed_arguments.is_debug();
        self.headless = parsed_arguments.is_headless();
        self.test = parsed_arguments.is_test();
        self.help = parsed_arguments.is_help();
        self.directive = parsed_arguments.is_directive();
        self.self_test = parsed_arguments.is_self_test();
        eprintln!("Running Etomo...");
        eprintln!("Arguments:");
        for argument in arguments {
            eprint!("{argument} ");
        }
        eprintln!("\n");

        self.setup()?;
        super::ui::swing::ui_harness::INSTANCE.with(|ui_harness| {
            let ui_harness = ui_harness.borrow();
            ui_harness
                .set_visible(true)
                .and_then(|()| ui_harness.run())
                .map_err(|error| error.to_string())
        })
    }

    /// Java `getArguments()` (`EtomoDirector.java:1415`).  Java returns the
    /// `ARGUMENTS` static; a caller that needs to mutate it locks the static directly.
    pub fn get_arguments(&self) -> std::sync::MutexGuard<'static, Arguments> {
        ARGUMENTS.lock().unwrap()
    }

    /// Matches Java `isSimulateWindows()` (`EtomoDirector.java:167`).
    pub fn is_simulate_windows() -> bool {
        false
    }
    /// Matches Java `isUnitTest()` (`EtomoDirector.java:246`).
    pub fn is_unit_test(&self) -> bool {
        self.self_test
    }
    /// Matches Java `isTest()` (`EtomoDirector.java:250`).
    pub fn is_test(&self) -> bool {
        self.test
    }
    /// Java static `isUserPreferenceLoaded()`.
    pub fn is_user_preference_loaded(&self) -> bool {
        self.user_preference_loaded
    }
    /// Java static `getUserFontSize()`.
    pub fn get_user_font_size(&self) -> i32 {
        self.user_font_size
    }
    /// Native replacement for Java's `loadUserConfiguration` lifecycle entry.
    /// `ParameterStore` remains responsible for decoding the typed settings;
    /// this establishes the same user-owned file and records a successful load.
    pub fn load_user_configuration(&mut self) -> Result<&Path, String> {
        if self.home_directory.is_empty() {
            return Err("Can not find home directory! Unable to load user preferences".into());
        }
        let path = PathBuf::from(&self.home_directory).join(USER_CONFIG_FILE_EXT);
        if !path.exists() {
            std::fs::File::create(&path).map_err(|error| error.to_string())?;
        }
        let mut parameter_store = ParameterStore::get_instance(Some(path.clone()))
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "unable to create user parameter store".to_owned())?;
        parameter_store.load(&mut self.user_configuration);
        self.user_configuration_path = Some(path);
        self.parameter_store = Some(parameter_store);
        self.user_preference_loaded = true;
        Ok(self.user_configuration_path.as_deref().unwrap())
    }
    pub fn get_user_configuration_path(&self) -> Option<&Path> {
        self.user_configuration_path.as_deref()
    }
    /// Java `getUserConfiguration()`, represented by the fully-owned native
    /// values rather than a JVM singleton.
    pub fn get_user_configuration(&self) -> &UserConfigurationValues {
        &self.user_configuration
    }
    /// Java `getParameterStore()`: the user-preference store is established
    /// during setup/load and remains the single persistence owner used by the
    /// settings dialog.
    pub fn get_parameter_store(&self) -> Option<&ParameterStore> {
        self.parameter_store.as_ref()
    }
    /// Java `setMaintainEtomo(UITester)` at the director state boundary.
    pub fn set_maintain_etomo(&mut self, maintain: bool) {
        self.maintain_etomo = maintain;
    }
    pub fn stop_maintain_etomo(&mut self) {
        self.maintain_etomo = false;
    }
    pub fn is_maintaining_etomo(&self) -> bool {
        self.maintain_etomo
    }
    /// Java private `setAdvanced(boolean)`, made an explicit native state update.
    pub fn set_advanced(&mut self, state: bool) {
        self.is_advanced = state;
    }
    /// Rust's diagnostic equivalent of Java `printProperties`; it never exposes
    /// environment values because those can hold credentials.
    pub fn print_properties(&self) -> String {
        format!(
            "EtomoDirector{{home:{}, headless:{}, advanced:{}, preferencesLoaded:{}}}",
            self.home_directory, self.headless, self.is_advanced, self.user_preference_loaded
        )
    }

    /// Matches Java `setup()` (`EtomoDirector.java:254`) excluding JVM property/Swing output.
    pub fn setup(&mut self) -> Result<(), String> {
        eprintln!("\nEnvironment variables:\n");
        if !self.test {
            for (key, value) in std::env::vars() {
                let lines = value.lines().collect::<Vec<_>>();
                let function_value = lines
                    .first()
                    .is_some_and(|line| line.trim_start().starts_with("()") && line.contains('{'))
                    && lines
                        .last()
                        .is_some_and(|line| line.trim_end().ends_with('}'));
                if !lines.is_empty() && !function_value {
                    let redacted = if key.to_ascii_lowercase().contains("pass")
                        || key.to_ascii_lowercase().contains("token")
                    {
                        "<redacted>"
                    } else {
                        &value
                    };
                    eprintln!("{key}:  {redacted}");
                }
            }
        }
        self.load_user_configuration()?;
        self.initialize()?;
        self.do_automation();
        Ok(())
    }

    /// Matches Java `doAutomation()` (`EtomoDirector.java:389`).
    pub fn do_automation(&self) {
        if let Some(manager) = self.get_current_manager() {
            manager.do_automation(None);
        }
    }

    /// Matches Java `initialize()` (`EtomoDirector.java:424`) before manager/UI opening.
    pub fn initialize(&mut self) -> Result<(), String> {
        self.original_user_dir = Some(
            std::env::current_dir()
                .map_err(|error| error.to_string())?
                .to_string_lossy()
                .into_owned(),
        );
        if self.home_directory.is_empty() {
            return Err("Can not find home directory! Unable to load user preferences".to_owned());
        }
        self.imod_brief_header = std::env::var_os("IMOD_BRIEF_HEADER").is_some();
        if cfg!(windows) {
            self.number_of_processors_windows = std::env::var("NUMBER_OF_PROCESSORS")
                .ok()
                .and_then(|value| value.parse().ok());
        }
        if self.help {
            return Ok(());
        }
        #[cfg(feature = "gui")]
        super::ui::swing::ui_harness::INSTANCE.with(|ui_harness| {
            ui_harness
                .borrow_mut()
                .create_main_frame(self.headless)
                .map_err(|error| error.to_string())
        })?;
        self.init_imod_directory()?;
        // Java `initialize()` creates this just before it opens the initial
        // front page or dataset manager.  The concrete manager constructors
        // remain untranslated, but the ownership list exists at the same
        // lifecycle point for their eventual callers.
        self.manager_list = Some(UniqueHashedArray::new());
        self.setup_imod_calib_dir();
        self.init_program();
        Ok(())
    }

    /// Matches Java `setupImodCalibDir()` (`EtomoDirector.java:525`).
    pub fn setup_imod_calib_dir(&mut self) {
        let directory = std::env::var("IMOD_CALIB_DIR").unwrap_or_default();
        if directory.is_empty() {
            eprintln!(
                "WARNING:\nThe environment variable IMOD_CALIB_DIRis not set.\nSeveral Etomo functions will not be available:"
            );
        }
        self.imod_calib_directory = Some(PathBuf::from(directory));
    }

    /// Matches Java `initProgram()` (`EtomoDirector.java:555`) non-GUI state setup.
    pub fn init_program(&mut self) {
        if !self.test {
            eprintln!("\nGraphicsEnvironment.isHeadless()={}\n", self.headless);
        }
        let mut value = std::env::var("ETOMO_MEM_LIM").unwrap_or_default();
        if value.is_empty() {
            return;
        }
        let mut multiplier = 1i64;
        if value.ends_with(['k', 'K']) {
            multiplier = 1024;
            value.pop();
        } else if value.ends_with(['m', 'M']) {
            multiplier = 1024 * 1024;
            value.pop();
        }
        self.java_memory_limit = value.parse::<i64>().unwrap_or(0) * multiplier;
    }

    /// Matches Java `initIMODDirectory()` (`EtomoDirector.java:623`).
    pub fn init_imod_directory(&mut self) -> Result<(), String> {
        let directory = std::env::var(IMOD_DIR_ENV_VAR).map_err(|_| "Can not find IMOD directory! Set IMOD_DIR environment variable and restart program to fix this problem".to_owned())?;
        eprintln!("IMOD_DIR (env): {directory}");
        self.imod_directory = Some(PathBuf::from(directory));
        Ok(())
    }

    /// Matches Java `getIMODDirectory()` (`EtomoDirector.java:1422`).
    pub fn get_imod_directory(&self) -> Option<&Path> {
        self.imod_directory.as_deref()
    }
    /// Matches Java `getIMODBinPath()` (`EtomoDirector.java:1427`).
    pub fn get_imod_bin_path(&self) -> Option<String> {
        self.imod_directory.as_ref().map(|directory| {
            let mut path = directory.join("bin").to_string_lossy().into_owned();
            if !path.ends_with(std::path::MAIN_SEPARATOR) {
                path.push(std::path::MAIN_SEPARATOR);
            }
            path
        })
    }
    /// Java `getPythonScriptPath()`.  Cygwin conversion is intentionally owned
    /// by the platform process runner; the director caches the canonical IMOD
    /// bin path that all native runners consume.
    pub fn get_python_script_path(&mut self) -> Option<&str> {
        if self.python_script_path.is_none() {
            self.python_script_path = self.get_imod_bin_path();
        }
        self.python_script_path.as_deref()
    }
    /// Java overload `getPythonScriptPath(String)`.
    pub fn get_python_script_path_for(&self, imod_bin_path: impl Into<String>) -> String {
        imod_bin_path.into()
    }
    pub fn get_number_of_processors_windows(&self) -> Option<i64> {
        self.number_of_processors_windows
    }
    /// Matches Java `getIMODCalibDirectory()` (`EtomoDirector.java:1476`).
    pub fn get_imod_calib_directory(&self) -> Option<&Path> {
        self.imod_calib_directory.as_deref()
    }
    /// Matches Java `getAdvanced()` (`EtomoDirector.java:1484`).
    pub fn get_advanced(&self) -> bool {
        self.is_advanced
    }
    /// Matches Java `getAvailableMemory()` (`EtomoDirector.java:1559`).
    pub fn get_available_memory(&self) -> i64 {
        #[cfg(unix)]
        {
            // `sysconf` is the direct native analogue of the JVM runtime's
            // available-memory query.  Saturate rather than wrapping on a
            // large-memory host.
            let pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if pages <= 0 || page_size <= 0 {
                0
            } else {
                pages.saturating_mul(page_size)
            }
        }
        #[cfg(not(unix))]
        {
            0
        }
    }
    /// Matches Java `isImodBriefHeader()` (`EtomoDirector.java:1567`).
    pub fn is_imod_brief_header(&self) -> bool {
        self.imod_brief_header
    }
    /// Matches Java `isMemoryAvailable()` (`EtomoDirector.java:1571`).
    pub fn is_memory_available(&self) -> bool {
        self.get_available_memory() as f64
            > MIN_AVAILABLE_MEMORY_REQUIRED + self.java_memory_limit as f64
    }
    /// Matches Java `getHomeDirectory()` (`EtomoDirector.java:1620`).
    pub fn get_home_directory(&self) -> &str {
        &self.home_directory
    }
    /// Java test-only `setCurrentPropertyUserDir`; manager-specific properties
    /// are an application-manager boundary, so without one this updates the
    /// director's original working directory state.
    pub fn set_current_property_user_dir(&mut self, directory: impl Into<String>) -> String {
        let directory = directory.into();
        if let Some(manager) = self.get_current_manager() {
            return manager
                .set_property_user_dir(Some(&directory))
                .unwrap_or_default();
        }
        let previous = self.original_user_dir.clone().unwrap_or_default();
        self.original_user_dir = Some(directory);
        previous
    }
    /// Java `makeOriginalDirLocal`, represented as an explicit process cwd change.
    pub fn make_original_dir_local(&self) -> Result<(), String> {
        let directory = self
            .original_user_dir
            .as_deref()
            .ok_or("original user directory has not been initialized")?;
        std::env::set_current_dir(directory).map_err(|error| error.to_string())
    }
    pub fn get_original_user_dir(&self) -> Option<&str> {
        self.original_user_dir.as_deref()
    }
    /// Java `getPropertyUserDir`: a current manager owns its project
    /// directory; otherwise the director keeps the process's original user
    /// directory captured during setup.
    pub fn get_property_user_dir(&self) -> Option<String> {
        self.get_current_manager()
            .and_then(BaseManager::get_property_user_dir)
            .or_else(|| self.original_user_dir.clone())
    }
    /// Java `setTestFailed`.  The source rejects this outside UI-test mode so
    /// a normal shutdown cannot accidentally suppress an error dialog.
    pub fn set_test_failed(&mut self, input: bool) -> Result<(), String> {
        if !self.test {
            return Err(format!("test={}", self.test));
        }
        self.test_failed = input;
        Ok(())
    }
    /// Java `isTestFailed`.
    pub const fn is_test_failed(&self) -> bool {
        self.test_failed
    }
    /// Java private `printUsageMessage`, returned for the command/UI frontend
    /// to emit using its own output sink.
    pub fn print_usage_message(&self) -> String {
        "Usage: etomo [options] [data files]\n".to_owned()
    }
    /// Java `pack(BaseManager)`.  Geometry is presentation-owned, but the
    /// director performs the exact UIHarness dispatch rather than recording a
    /// request for a future frontend to interpret.
    #[cfg(feature = "gui")]
    pub fn pack(&self, manager: &dyn BaseManager) {
        let manager_name = manager.get_name();
        super::ui::swing::ui_harness::INSTANCE.with(|ui_harness| {
            ui_harness
                .borrow_mut()
                .pack_manager(manager_name.as_deref());
        });
    }
    /// Java private `setUserPreferences`.  The native frontend consumes the
    /// retained appearance values; tooltip timing and advanced-dialog state
    /// are already represented in `UserConfigurationValues`.
    pub fn set_user_preferences(&mut self) {
        if self.headless {
            return;
        }
        let config = self.user_configuration.clone();
        self.set_ui_font(&config.font_family, config.font_size);
        self.set_look_and_feel(config.native_laf);
        self.is_advanced = config.advanced_dialogs;
    }

    /// Java private `setLookAndFeel`.  Selecting a Slint style is owned by
    /// the GUI bootstrap, while this director preserves the source setting
    /// and makes it observable before that bootstrap runs.
    pub fn set_look_and_feel(&mut self, native_look_and_feel: bool) {
        self.native_look_and_feel = native_look_and_feel;
    }

    /// Java private `setUIFont`.  Font discovery/application is a frontend
    /// concern, but the selected family and size are director-owned user
    /// preferences and must survive opening/saving Settings.
    pub fn set_ui_font(&mut self, font_family: &str, font_size: i32) {
        self.ui_font_family = font_family.to_owned();
        self.user_font_size = font_size;
    }

    /// Java `getSettingsParameters`.  Returns whether the selected appearance
    /// needs a restart, which is the source's informational-dialog condition.
    pub fn get_settings_parameters(&mut self) -> Result<bool, String> {
        let dialog = self
            .settings_dialog
            .as_ref()
            .ok_or_else(|| "settingsDialog is null".to_owned())?;
        let appearance_changed = dialog.is_appearance_setting_changed(&self.user_configuration);
        dialog.get_parameters(&mut self.user_configuration)?;
        self.set_user_preferences();
        Ok(appearance_changed)
    }

    /// Java `openSettingsDialog`: one non-modal dialog is retained and reused
    /// for the active manager, seeded from the currently loaded preferences.
    pub fn open_settings_dialog(&mut self) -> Result<(), String> {
        if self.settings_dialog.is_none() {
            let manager = self
                .get_current_manager()
                .ok_or_else(|| "current manager is required for settings".to_owned())?;
            let property_user_dir = self.get_property_user_dir().unwrap_or_default();
            let mut dialog = SettingsDialog::get_instance(
                manager,
                property_user_dir,
                &["Dialog".to_owned()],
                false,
            );
            dialog.set_parameters(&self.user_configuration);
            self.settings_dialog = Some(dialog);
        }
        self.settings_dialog.as_mut().unwrap().visible = true;
        Ok(())
    }

    /// Java `saveSettingsDialog`: persist the current configuration through
    /// the user `.etomo` parameter store.
    pub fn save_settings_dialog(&mut self) -> Result<(), String> {
        let parameter_store = self
            .parameter_store
            .as_mut()
            .ok_or_else(|| "user parameter store has not been initialized".to_owned())?;
        parameter_store
            .save(Some(&self.user_configuration))
            .map_err(|error| error.to_string())
    }

    /// Java `closeSettingsDialog` disposes its presentation.  The native
    /// dialog state is retained so a later open preserves its source-owned
    /// singleton lifecycle while becoming visible again.
    pub fn close_settings_dialog(&mut self) {
        if let Some(dialog) = &mut self.settings_dialog {
            dialog.visible = false;
            dialog.closed = true;
        }
    }
    /// Java `renameCurrentManager`.  The native UI observes the resulting
    /// `ManagerKey`; this director-owned portion rekeys the ordered manager
    /// list and updates the shared current-key holder atomically with respect
    /// to subsequent manager lookups.
    pub fn rename_current_manager(
        &mut self,
        manager_name: impl Into<String>,
    ) -> Result<(), String> {
        let manager_key = self
            .current_manager_key
            .as_ref()
            .ok_or_else(|| "currentManagerKey is null".to_owned())?
            .clone();
        let old_key = manager_key
            .lock()
            .unwrap()
            .get_key()
            .cloned()
            .ok_or_else(|| "currentManagerKey.key is null".to_owned())?;
        let new_key = self
            .manager_list
            .as_mut()
            .ok_or_else(|| "managerList is null".to_owned())?
            .rekey_with_name(&old_key, manager_name.into())
            .ok_or_else(|| format!("manager key {old_key} is not open"))?;
        manager_key.lock().unwrap().set_key(Some(new_key));
        Ok(())
    }

    /// Java private `enableOpenManagerMenuItem`.  New-project menu entries
    /// are disabled while their unique untitled manager exists and restored
    /// when that manager is closed or renamed.
    pub fn enable_open_manager_menu_item(&self, key: &UniqueKey) {
        #[cfg(feature = "gui")]
        super::ui::swing::ui_harness::INSTANCE.with(|ui_harness| {
            let mut ui_harness = ui_harness.borrow_mut();
            match key.get_name() {
                "Setup Tomogram" => ui_harness.set_enabled_new_tomogram_menu_item(true),
                "New Join" => ui_harness.set_enabled_new_join_menu_item(true),
                "Parallel Processing" => {
                    ui_harness.set_enabled_new_generic_parallel_menu_item(true)
                }
                "Nonlinear Anisotropic Diffusion" => {
                    ui_harness.set_enabled_new_anisotropic_diffusion_menu_item(true)
                }
                "Batch Run Tomo" => ui_harness.set_enabled_new_batch_run_tomo_menu_item(true),
                "PEET" => ui_harness.set_enabled_new_peet_menu_item(true),
                "Serial Sections" => ui_harness.set_enabled_new_serial_sections_menu_item(true),
                _ => {}
            }
        });
        #[cfg(not(feature = "gui"))]
        let _ = key;
    }

    /// Java `getManager(UniqueKey)` (`EtomoDirector.java:650`).
    pub(crate) fn get_manager(&self, key: Option<&UniqueKey>) -> Option<&'static dyn BaseManager> {
        self.manager_list.as_ref()?.get(key?).copied()
    }

    /// Java package-private `getCurrentManager()` (`EtomoDirector.java:661`).
    ///
    /// Keeping this package-visible matters: Java deliberately keeps it below
    /// public visibility because UI tab changes can alter the current manager.
    pub(crate) fn get_current_manager(&self) -> Option<&'static dyn BaseManager> {
        let manager_key = self.current_manager_key.as_ref()?;
        let key = manager_key.lock().unwrap().get_key()?.clone();
        self.get_manager(Some(&key))
    }

    /// Java `getCurrentManagerForTest()` (`EtomoDirector.java:668`).
    pub(crate) fn get_current_manager_for_test(
        &self,
    ) -> Result<Option<&'static dyn BaseManager>, String> {
        if !self.test {
            return Err("Illegal use of getCurrentManagerForTest".to_owned());
        }
        Ok(self.get_current_manager())
    }

    /// Java private `setManager(BaseManager, AxisID, boolean)`
    /// (`EtomoDirector.java:1007`).
    ///
    /// The `UIHarness.addWindow` and menu selection calls remain at the
    /// `WindowSwitch` frontier.  This method deliberately performs the
    /// director-owned portion only: allocate the unique key, attach its exact
    /// mutable `ManagerKey` holder to the manager, retain the manager, and make
    /// it current when requested.
    pub(crate) fn set_manager(
        &mut self,
        manager: &'static dyn BaseManager,
        make_current: bool,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        let name = manager
            .get_name()
            .ok_or_else(|| "BaseManager.getName() returned null".to_owned())?;
        let manager_list = self
            .manager_list
            .as_mut()
            .ok_or_else(|| "managerList is null".to_owned())?;
        let unique_key = manager_list.add_with_name(name, manager);
        manager.set_manager_key(Some(unique_key));
        let manager_key = manager.get_manager_key();
        if make_current {
            self.set_current_manager_manager_key(Some(Arc::clone(&manager_key)), true, true)?;
        }
        Ok(manager_key)
    }

    /// Java `openJoin`, using the existing native manager constructor.
    pub fn open_join(
        &mut self,
        param_file_name: Option<&str>,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(
            JoinManager::new(param_file_name, Some(axis_id)),
            make_current,
        )
    }
    /// Java `openParallel` for a saved parallel project or a generic one.
    pub fn open_parallel(
        &mut self,
        param_file_name: Option<&str>,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        let manager = match param_file_name {
            Some(name) => ParallelManager::new_with_param_file(Some(name)),
            None => ParallelManager::new(),
        };
        self.set_manager(manager, make_current)
    }
    /// Java `openGenericParallel`.
    pub fn open_generic_parallel(
        &mut self,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(
            ParallelManager::new_with_dialog_type(DialogType::Parallel),
            make_current,
        )
    }
    /// Java `openAnisotropicDiffusion`.
    pub fn open_anisotropic_diffusion(
        &mut self,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(
            ParallelManager::new_with_dialog_type(DialogType::AnisotropicDiffusion),
            make_current,
        )
    }
    /// Java `openBatchRunTomo`.
    pub fn open_batch_run_tomo(
        &mut self,
        param_file_name: Option<&str>,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(
            BatchRunTomoManager::new_with_param_file_name(param_file_name),
            make_current,
        )
    }
    /// Java `openSerialSections`.
    pub fn open_serial_sections(
        &mut self,
        param_file_name: Option<&str>,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(
            SerialSectionsManager::get_instance_with_param_file_name(param_file_name),
            make_current,
        )
    }
    /// Java `openFrontPage`.  This is the director's ordinary default-window
    /// path: the concrete FrontPageManager is registered exactly like the
    /// other dataset managers and can be made current immediately.
    pub fn open_front_page(
        &mut self,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(FrontPageManager::new(), make_current)
    }
    pub fn open_tomogram(
        &mut self,
        param_file_name: Option<&str>,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        self.set_manager(
            ApplicationManager::new(param_file_name, axis_id),
            make_current,
        )
    }
    /// Java `openManager(File, boolean, AxisID, UIComponent)`.  Dispatch only
    /// reaches concrete Rust managers; recognised formats whose source manager
    /// is still absent fail explicitly instead of being opened as another type.
    pub fn open_manager(
        &mut self,
        data_file: &Path,
        make_current: bool,
        axis_id: AxisID,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(axis_id));
        if JoinFileFilter::new().accept(data_file) {
            return self.open_join(data_file.to_str(), make_current, axis_id);
        }
        if data_file
            .to_str()
            .is_some_and(|name| name.ends_with(DataFileType::Parallel.extension().unwrap()))
        {
            return self.open_parallel(data_file.to_str(), make_current, axis_id);
        }
        if data_file
            .to_str()
            .is_some_and(|name| name.ends_with(DataFileType::BatchRunTomo.extension().unwrap()))
        {
            return self.open_batch_run_tomo(data_file.to_str(), make_current, axis_id);
        }
        if SerialSectionsFileFilter::new().accept(data_file) {
            return self.open_serial_sections(data_file.to_str(), make_current, axis_id);
        }
        if EtomoFileFilter.accept(data_file) {
            return self.open_tomogram(data_file.to_str(), make_current, axis_id);
        }
        if PeetFileFilter::new().accept(data_file) {
            return Err("PEET manager is not translated yet".to_owned());
        }
        Err(format!("unknown dataFile {}", data_file.display()))
    }
    /// Java `openTool`.  The director owns replacement of its default window,
    /// construction/initialization of the typed tools manager, and manager-list
    /// registration; the Slint frontend observes that registered manager to add
    /// its concrete window.
    pub fn open_tool(
        &mut self,
        make_current: bool,
        tool_type: ToolType,
    ) -> Result<Arc<Mutex<ManagerKey>>, String> {
        self.close_default_window(Some(AxisID::First));
        let manager = ToolsManager::new(tool_type);
        manager.initialize();
        self.set_manager(manager, make_current)
    }
    /// Java `openToolInSeparateFrame`.  Unlike `openTool`, this manager is not
    /// placed in the director's document-manager list; it is returned to the
    /// UI owner that creates the independent frame.
    pub fn open_tool_in_separate_frame(&self, tool_type: ToolType) -> &'static ToolsManager {
        let manager = ToolsManager::new(tool_type);
        manager.initialize();
        manager
    }
    /// Java `openDirectiveEditor`.  Directive editing uses its own frame and
    /// intentionally does not become a dataset manager-list entry; retain the
    /// source manager construction, initialization, data-source link and
    /// timestamp/error-message inputs for the native frontend to present.
    pub fn open_directive_editor(
        &self,
        directive_file_type: Option<DirectiveFileType>,
        data_source: Option<&'static dyn BaseManager>,
        timestamp: Option<&str>,
        errmsg: Option<&str>,
    ) -> &'static DirectiveEditorManager {
        let manager =
            DirectiveEditorManager::new(directive_file_type, data_source, timestamp, errmsg);
        manager.initialize();
        manager
    }

    /// Java `setCurrentManager(ManagerKey, boolean, boolean)`
    /// (`EtomoDirector.java:719`).
    pub(crate) fn set_current_manager_manager_key(
        &mut self,
        manager_key: Option<Arc<Mutex<ManagerKey>>>,
        _new_window: bool,
        _manager_stamp: bool,
    ) -> Result<(), String> {
        let Some(manager_key) = manager_key else {
            return Ok(());
        };
        let Some(key) = manager_key.lock().unwrap().get_key().cloned() else {
            return Ok(());
        };
        if self.get_manager(Some(&key)).is_none() {
            // Java routes this to the private overload, which throws its
            // `NullPointerException("managerKey=" + managerKey)`.
            return Err(format!("managerKey={}", manager_key.lock().unwrap()));
        }
        self.current_manager_key = Some(manager_key);
        Ok(())
    }

    /// Java `setCurrentManager(UniqueKey, boolean)` (`EtomoDirector.java:733`).
    pub(crate) fn set_current_manager_unique_key(
        &mut self,
        key: Option<&UniqueKey>,
        new_window: bool,
    ) -> bool {
        let Some(manager) = self.get_manager(key) else {
            return false;
        };
        self.set_current_manager_manager_key(Some(manager.get_manager_key()), new_window, true)
            .is_ok()
    }

    /// Java `isOpen(UniqueKey)` (`EtomoDirector.java:1127`).
    pub(crate) fn is_open(&self, manager_unique_key: Option<&UniqueKey>) -> bool {
        self.manager_list
            .as_ref()
            .is_some_and(|manager_list| manager_list.contains(manager_unique_key))
    }

    /// Java `closeManagers(List<UniqueKey>)` (`EtomoDirector.java:1134`).
    pub(crate) fn close_managers(&mut self, manager_unique_key_list: Option<&[UniqueKey]>) {
        let Some(manager_unique_key_list) = manager_unique_key_list else {
            return;
        };
        for manager_unique_key in manager_unique_key_list {
            if self.set_current_manager_unique_key(Some(manager_unique_key), false) {
                let _ = self.close_current_manager(None, false);
            }
        }
    }

    /// Java `closeDefaultWindow`.  Concrete front-page creation belongs to the
    /// native UI factory; when it marks its sole window as default, opening a
    /// real manager closes it through this director-owned lifecycle.
    pub(crate) fn set_default_window(&mut self, value: bool) {
        self.default_window = value;
    }
    pub(crate) fn close_default_window(
        &mut self,
        axis_id: Option<super::r#type::axis_id::AxisID>,
    ) -> bool {
        if !self.default_window
            || self
                .manager_list
                .as_ref()
                .map_or(0, UniqueHashedArray::size)
                != 1
        {
            return true;
        }
        self.default_window = false;
        self.close_current_manager(axis_id, false)
    }

    /// Java private `saveLogs`: asks each open manager to persist its log.
    pub fn save_logs(&self) {
        if let Some(managers) = &self.manager_list {
            for index in 0..managers.size() {
                if let Some(manager) = managers.get_at(index) {
                    manager.save_log();
                }
            }
        }
    }

    /// Java `exitProgram(AxisID)`, excluding process-global JVM shutdown.
    /// Each manager can veto closure; successful close releases it from the
    /// director list before the next manager is selected.
    pub fn exit_program(&mut self, axis_id: Option<super::r#type::axis_id::AxisID>) -> bool {
        self.save_logs();
        loop {
            let next_key = self
                .manager_list
                .as_ref()
                .and_then(|managers| managers.get_key(0).cloned());
            let Some(next_key) = next_key else { break };
            if !self.set_current_manager_unique_key(Some(&next_key), false)
                || !self.close_current_manager(axis_id, true)
            {
                return false;
            }
        }
        self.stop_maintain_etomo();
        true
    }

    /// Java `closeManager(AxisID, UniqueKey)` (`EtomoDirector.java:1151`).
    pub(crate) fn close_manager(
        &mut self,
        manager_unique_key: Option<&UniqueKey>,
        exiting: bool,
    ) -> bool {
        let Some(manager_unique_key) = manager_unique_key else {
            return true;
        };
        let saved_current_manager_key = match self.current_manager_key.as_ref() {
            Some(current_manager_key)
                if current_manager_key
                    .lock()
                    .unwrap()
                    .equals_unique_key(Some(manager_unique_key)) =>
            {
                None
            }
            _ => self.current_manager_key.clone(),
        };
        if saved_current_manager_key.is_some()
            && !self.set_current_manager_unique_key(Some(manager_unique_key), false)
        {
            return true;
        }
        if !self.close_current_manager(None, exiting) {
            return false;
        }
        if let Some(saved_current_manager_key) = saved_current_manager_key {
            let _ =
                self.set_current_manager_manager_key(Some(saved_current_manager_key), false, true);
        }
        true
    }

    /// Java `closeCurrentManager(AxisID, boolean)` (`EtomoDirector.java:1169`).
    pub(crate) fn close_current_manager(
        &mut self,
        axis_id: Option<super::r#type::axis_id::AxisID>,
        exiting: bool,
    ) -> bool {
        let Some(current_manager) = self.get_current_manager() else {
            return true;
        };
        if exiting {
            if !current_manager.exit_program(axis_id) {
                return false;
            }
        } else if !current_manager.close(axis_id) {
            return false;
        }
        let Some(current_manager_key) = self.current_manager_key.as_ref() else {
            return true;
        };
        let Some(key) = current_manager_key.lock().unwrap().get_key().cloned() else {
            return true;
        };
        self.enable_open_manager_menu_item(&key);
        let Some(manager_list) = self.manager_list.as_mut() else {
            return true;
        };
        manager_list.remove(&key);
        self.current_manager_key = None;
        let first_key = manager_list.get_key(0).cloned();
        if let Some(first_key) = first_key {
            let _ = self.set_current_manager_unique_key(Some(&first_key), false);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::convert::Infallible;
    use std::rc::Rc;

    use super::{BaseManager, EtomoDirector};
    use crate::imod::etomo::base_manager::BaseManagerBase;
    use crate::imod::etomo::storage::storable::Storable;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
    use crate::imod::etomo::r#type::interface_type::InterfaceType;
    use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
    use crate::imod::etomo::ui::swing::window_switch::{
        WindowMainPanel, WindowManager, WindowSwitch,
    };

    /// A test-only concrete subclass of Java's abstract `BaseManager`.
    /// Runtime GUI wiring never fabricates a manager: the source's concrete
    /// manager constructors are the producers of this input.
    struct Manager {
        base: BaseManagerBase,
        name: String,
    }

    impl BrowsingDirectory for Manager {
        fn get_browsing_dir(&self) -> Option<std::path::PathBuf> {
            BaseManager::get_browsing_dir(self)
        }

        fn set_browsing_dir(&self, file: Option<&std::path::Path>) {
            BaseManager::set_browsing_dir(self, file)
        }
    }

    impl BaseManager for Manager {
        fn base(&self) -> &BaseManagerBase {
            &self.base
        }

        fn this(&'static self) -> &'static dyn BaseManager {
            self
        }

        fn get_interface_type(&self) -> Option<InterfaceType> {
            None
        }

        fn create_main_panel(&self) {}

        fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
            None
        }

        fn get_main_panel(&self) -> Option<Infallible> {
            None
        }

        fn get_process_manager(&self) -> Option<Infallible> {
            None
        }

        fn get_storables_with_offset(&self, _offset: i32) -> Option<Vec<Box<dyn Storable>>> {
            None
        }

        fn get_name(&self) -> Option<String> {
            Some(self.name.clone())
        }
    }

    struct Panel;

    impl WindowMainPanel for Panel {
        fn save_display_state(&mut self) {}
    }

    struct WindowManagerAdapter(Option<Panel>);

    impl WindowManager<Panel> for WindowManagerAdapter {
        fn get_main_panel(&mut self) -> Option<Panel> {
            self.0.take()
        }
    }

    fn manager(name: &str) -> &'static Manager {
        Box::leak(Box::new(Manager {
            base: BaseManagerBase::initial(),
            name: name.to_owned(),
        }))
    }

    #[test]
    fn parses_headless_director_path_without_jvm() {
        let mut director = EtomoDirector::new();
        let result = director.main(&["-headless".to_owned(), "-test".to_owned()]);
        assert!(result.is_err() || director.headless);
    }

    #[test]
    fn window_switch_callbacks_select_the_directors_owned_manager_key() {
        // This is the direct Java boundary in WindowSwitch.menuAction and
        // WindowSwitch.tabChanged: a UI selection calls
        // EtomoDirector.setCurrentManager(UniqueKey).  The WindowSwitch does
        // not retain a second current-manager value.
        let director = Rc::new(RefCell::new(EtomoDirector::new()));
        let first_manager = manager("one.edf");
        let second_manager = manager("two.edf");
        let (first, second) = {
            let mut director = director.borrow_mut();
            director.manager_list = Some(super::UniqueHashedArray::new());
            let first = director.set_manager(first_manager, false).unwrap();
            let second = director.set_manager(second_manager, false).unwrap();
            (first, second)
        };

        let first_key = first.lock().unwrap().get_key().cloned().unwrap();
        let second_key = second.lock().unwrap().get_key().cloned().unwrap();
        let mut window_switch = WindowSwitch::new();
        let callback_director = Rc::clone(&director);
        window_switch.set_current_manager_listener(Box::new(move |key| {
            assert!(
                callback_director
                    .borrow_mut()
                    .set_current_manager_unique_key(Some(&key), false)
            );
        }));
        window_switch.add(
            &mut WindowManagerAdapter(Some(Panel)),
            crate::imod::etomo::r#type::axis_id::AxisID::Only,
            Some(first_key),
        );
        window_switch.add(
            &mut WindowManagerAdapter(Some(Panel)),
            crate::imod::etomo::r#type::axis_id::AxisID::Only,
            Some(second_key),
        );

        window_switch.menu_action("2: two.edf");
        assert!(std::ptr::eq(
            director.borrow().get_current_manager().unwrap(),
            second_manager as &'static dyn BaseManager,
        ));

        window_switch.tab_changed(Some(0));
        assert!(std::ptr::eq(
            director.borrow().get_current_manager().unwrap(),
            first_manager as &'static dyn BaseManager,
        ));
    }

    #[test]
    fn property_user_dir_falls_back_to_director_without_current_manager() {
        let mut director = EtomoDirector::new();
        director.original_user_dir = Some("before".into());
        assert_eq!(director.set_current_property_user_dir("after"), "before");
        assert_eq!(director.get_original_user_dir(), Some("after"));
    }

    #[test]
    fn settings_dialog_is_retained_applied_and_closed_by_the_director() {
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        director.set_manager(manager("settings.edf"), true).unwrap();

        director.open_settings_dialog().unwrap();
        let dialog = director.settings_dialog.as_mut().unwrap();
        assert!(dialog.visible);
        dialog.font_size = "16".to_owned();
        dialog.native_laf = true;
        dialog.advanced_dialogs = true;

        assert!(director.get_settings_parameters().unwrap());
        assert_eq!(director.get_user_font_size(), 16);
        assert!(director.native_look_and_feel);
        assert!(director.is_advanced);

        director.close_settings_dialog();
        let dialog = director.settings_dialog.as_ref().unwrap();
        assert!(!dialog.visible && dialog.closed);
    }

    #[test]
    fn front_page_opening_registers_the_default_manager_lifecycle() {
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director.open_front_page(true, AxisID::Only).unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            "Front Page"
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::FrontPage)
        );
    }

    #[test]
    fn open_manager_dispatches_a_join_file_to_its_concrete_manager() {
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director
            .open_manager(std::path::Path::new("joined.ejf"), true, AxisID::Only)
            .unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            "joined.ejf"
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::Join)
        );
    }

    #[test]
    fn open_manager_dispatches_serial_sections_to_its_concrete_manager() {
        let root = std::env::temp_dir().join(format!(
            "imod_rs_director_serial_dispatch_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let data_file = root.join("serial.ess");
        std::fs::write(&data_file, []).unwrap();
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director
            .open_manager(&data_file, true, AxisID::Only)
            .unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            data_file.to_string_lossy()
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::SerialSections)
        );
        std::fs::remove_file(data_file).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn generic_parallel_opening_registers_its_source_identity() {
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director.open_generic_parallel(true, AxisID::Only).unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            "Parallel Processing"
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::Pp)
        );
    }

    #[test]
    fn tomogram_opening_registers_new_dataset_identity() {
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director.open_tomogram(None, true, AxisID::Only).unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            "Setup Tomogram"
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::Recon)
        );
    }

    #[test]
    fn open_manager_dispatches_existing_parallel_file_to_parallel_manager() {
        let root = std::env::temp_dir().join(format!(
            "imod_rs_director_parallel_dispatch_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let data_file = root.join("parallel.epp");
        std::fs::write(&data_file, []).unwrap();
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director
            .open_manager(&data_file, true, AxisID::Only)
            .unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            data_file.to_string_lossy()
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::Pp)
        );
        std::fs::remove_file(data_file).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn open_manager_dispatches_existing_batch_file_to_batch_manager() {
        let root = std::env::temp_dir().join(format!(
            "imod_rs_director_batch_dispatch_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let data_file = root.join("batch.ebt");
        std::fs::write(&data_file, []).unwrap();
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director
            .open_manager(&data_file, true, AxisID::Only)
            .unwrap();
        assert_eq!(
            key.lock().unwrap().get_key().unwrap().get_name(),
            data_file.to_string_lossy()
        );
        assert_eq!(
            director.get_current_manager().unwrap().get_interface_type(),
            Some(InterfaceType::BatchRunTomo)
        );
        std::fs::remove_file(data_file).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn test_failure_and_usage_retain_director_only_contracts() {
        let mut director = EtomoDirector::new();
        assert_eq!(director.set_test_failed(true), Err("test=false".to_owned()));
        director.test = true;
        director.set_test_failed(true).unwrap();
        assert!(director.is_test_failed());
        assert_eq!(
            director.print_usage_message(),
            "Usage: etomo [options] [data files]\n"
        );
        director.original_user_dir = Some("original".to_owned());
        assert_eq!(
            director.get_property_user_dir(),
            Some("original".to_owned())
        );
    }

    #[test]
    fn tool_opening_constructs_the_translated_tools_manager_in_its_source_lifetime() {
        use crate::imod::etomo::ui::swing::etomo_menu::ToolType;

        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let key = director.open_tool(true, ToolType::AlignFrames).unwrap();
        assert!(director.is_open(key.lock().unwrap().get_key()));
        let separate = director.open_tool_in_separate_frame(ToolType::FlattenVolume);
        assert_eq!(separate.get_name(), Some("Flatten Volume".to_owned()));
    }

    #[test]
    fn directive_editor_opening_uses_its_separate_manager_lifetime() {
        let director = EtomoDirector::new();
        let manager = director.open_directive_editor(None, None, Some("stamp"), Some("error"));
        assert_eq!(
            manager.get_interface_type(),
            Some(crate::imod::etomo::r#type::interface_type::InterfaceType::Tools)
        );
    }

    #[test]
    fn rename_current_manager_rekeys_the_shared_manager_key() {
        let mut director = EtomoDirector::new();
        director.manager_list = Some(super::UniqueHashedArray::new());
        let current = director.set_manager(manager("old.edf"), true).unwrap();
        director.rename_current_manager("renamed.edf").unwrap();
        let key = current.lock().unwrap().get_key().cloned().unwrap();
        assert_eq!(key.get_name(), "renamed.edf");
        assert!(director.is_open(Some(&key)));
    }
}
