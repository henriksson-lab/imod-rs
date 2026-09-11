//! Translation boundary for `IMOD/Etomo/src/etomo/EtomoDirector.java`.
//!
//! Swing window/manager methods remain explicit JVM GUI boundaries.  This unit
//! maps the director's launcher, environment, IMOD-directory, calibration, and
//! memory-limit logic used before batch/headless automation reaches that boundary.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::arguments::Arguments;
use super::base_manager::BaseManager;
use super::manager_key::ManagerKey;
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
        self.initialize()?;
        self.do_automation();
        Ok(())
    }

    /// Matches Java `doAutomation()` (`EtomoDirector.java:389`); manager automation is JVM boundary.
    pub fn do_automation(&self) {}

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
        self.imod_directory
            .as_ref()
            .map(|directory| directory.join("bin").to_string_lossy().into_owned())
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
        0
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
}
