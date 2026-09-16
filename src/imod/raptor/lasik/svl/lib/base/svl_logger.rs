//! Owned translation of `svlLogger.{h,cpp}`.
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SvlLogLevel {
    Fatal = 0,
    Error = 1,
    Warning = 2,
    Message = 3,
    Verbose = 4,
    Debug = 5,
}
struct Logger {
    level: SvlLogLevel,
    path: Option<String>,
}
static LOGGER: OnceLock<Mutex<Logger>> = OnceLock::new();
fn logger() -> &'static Mutex<Logger> {
    LOGGER.get_or_init(|| {
        Mutex::new(Logger {
            level: SvlLogLevel::Message,
            path: None,
        })
    })
}
pub fn set_log_level(level: SvlLogLevel) {
    logger().lock().unwrap().level = level
}
pub fn get_log_level() -> SvlLogLevel {
    logger().lock().unwrap().level
}
pub fn initialize(
    filename: Option<&str>,
    overwrite: bool,
    level: Option<SvlLogLevel>,
) -> std::io::Result<()> {
    let mut state = logger().lock().unwrap();
    if let Some(level) = level {
        state.level = level;
    }
    state.path = filename.filter(|s| !s.is_empty()).map(str::to_owned);
    if let Some(path) = &state.path {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(!overwrite)
            .truncate(overwrite)
            .open(path)?;
        writeln!(file, "--- log opened --- ")?;
    }
    Ok(())
}
pub fn log_message(level: SvlLogLevel, message: &str) -> Result<(), String> {
    let state = logger().lock().unwrap();
    if level > state.level {
        return Ok(());
    }
    let prefix = match level {
        SvlLogLevel::Fatal => "-*-",
        SvlLogLevel::Error => "-E-",
        SvlLogLevel::Warning => "-W-",
        SvlLogLevel::Debug => "-D-",
        _ => "---",
    };
    if let Some(path) = &state.path {
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        writeln!(file, "{prefix} {message}").map_err(|e| e.to_string())?;
    }
    if level == SvlLogLevel::Fatal {
        Err(message.into())
    } else {
        Ok(())
    }
}
pub fn set_configuration(name: &str, value: &str) -> Result<(), String> {
    match name {
        "logLevel" => {
            let level = match value.to_ascii_uppercase().as_str() {
                "ERROR" => SvlLogLevel::Error,
                "WARNING" => SvlLogLevel::Warning,
                "MESSAGE" => SvlLogLevel::Message,
                "VERBOSE" => SvlLogLevel::Verbose,
                "DEBUG" => SvlLogLevel::Debug,
                _ => return Err("invalid configuration value for logLevel".into()),
            };
            set_log_level(level);
            Ok(())
        }
        "logFile" => initialize(Some(value), false, None).map_err(|e| e.to_string()),
        _ => Err(format!("unknown configuration option {name}")),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filter_and_config() {
        set_log_level(SvlLogLevel::Warning);
        assert!(log_message(SvlLogLevel::Debug, "x").is_ok());
        assert!(set_configuration("logLevel", "debug").is_ok());
        assert_eq!(get_log_level(), SvlLogLevel::Debug);
    }
}
