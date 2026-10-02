//! Translation of `IMOD/raptor/lasik/svl/lib/base/svlLogger.h` and
//! `svlLogger.cpp`, the parts `MarkersCorrespond` reaches.
//!
//! The log file (`svlLogger::_log`) is only opened by `initialize`, which the
//! program never calls, and the four display callbacks are never set, so a
//! message goes to `cerr` (fatal, error, warning) or `cout` (the rest).

use crate::imod::cxx_stream::{cerr, cout};

/// `svlLogLevel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SvlLogLevel {
    Fatal = 0,
    Error,
    Warning,
    Message,
    Verbose,
    Debug,
}

/// `svlLogger::_logLevel` (`svlLogger.cpp:56`); only the configuration
/// manager (not reached) changes it.
const LOG_LEVEL: SvlLogLevel = SvlLogLevel::Message;

/// `svlLogger::getLogLevel()`.
pub fn get_log_level() -> SvlLogLevel {
    LOG_LEVEL
}

/// The `SVL_LOG(L, M)` macro (`svlLogger.h:46`): `msg` is the streamed
/// message `M`; `file` and `line` are what `__FILE__`/`__LINE__` expand to at
/// the call site, used only for a fatal message.
pub fn svl_log(level: SvlLogLevel, file: &str, line: u32, msg: &str) {
    if level > get_log_level() {
        return;
    }
    let mut s = String::new();
    if level == SvlLogLevel::Fatal {
        s.push_str(&format!("({file}, {line}) "));
    }
    s.push_str(msg);
    log_message(level, &s);
}

/// `svlLogger::logMessage(level, msg)` (`svlLogger.cpp:95`).
pub fn log_message(level: SvlLogLevel, msg: &str) {
    if level > LOG_LEVEL {
        return;
    }

    let mut prefix = *b"---";
    match level {
        SvlLogLevel::Fatal => prefix[1] = b'*',
        SvlLogLevel::Error => prefix[1] = b'E',
        SvlLogLevel::Warning => prefix[1] = b'W',
        SvlLogLevel::Message => prefix[1] = b'-',
        SvlLogLevel::Verbose => prefix[1] = b'-',
        SvlLogLevel::Debug => prefix[1] = b'D',
    }
    let prefix = String::from_utf8_lossy(&prefix).into_owned();

    match level {
        SvlLogLevel::Fatal | SvlLogLevel::Error | SvlLogLevel::Warning => {
            cerr(&format!("{prefix} {msg}\n"));
        }
        _ => {
            cout(&format!("{prefix} {msg}\n"));
        }
    }

    if level == SvlLogLevel::Fatal {
        std::process::abort();
    }
}
