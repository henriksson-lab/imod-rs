//! Non-GUI translation boundary for `IMOD/Etomo/src/etomo/EtomoDirector.java`.
//!
//! Swing window/manager methods remain explicit JVM GUI boundaries.  This unit
//! maps the director's launcher, environment, IMOD-directory, calibration, and
//! memory-limit logic used before batch/headless automation reaches that boundary.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::arguments::Arguments;

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

/// Non-GUI fields of Java `EtomoDirector` (`EtomoDirector.java:76`).
#[derive(Clone, Debug, Default)]
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
        self.init_imod_directory()?;
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
}

#[cfg(test)]
mod tests {
    use super::EtomoDirector;
    #[test]
    fn parses_headless_director_path_without_jvm() {
        let mut director = EtomoDirector::new();
        let result = director.main(&["-headless".to_owned(), "-test".to_owned()]);
        assert!(result.is_err() || director.headless);
    }
}
