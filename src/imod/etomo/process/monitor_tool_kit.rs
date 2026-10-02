//! `IMOD/Etomo/src/etomo/process/MonitorToolKit.java`.
//!
//! Progress bar settings.  Can be used by any class, with the monitor set to
//! null.  In that situation it will be missing a small amount of
//! functionality, which would probably be useless to anything but a monitor
//! class.
//!
//! The monitor that owns a tool kit also owns the tool kit, so the Java
//! `this` reference handed to the constructor is a `Weak` here (the monitor
//! is built with `Arc::new_cyclic`).
//!
//! This module also carries the Rust stand-ins that every monitor in this
//! directory uses for `Thread.sleep`/`InterruptedException`
//! ([`sleep`], [`InterruptedException`]; see `monitor.rs`) and for the
//! `String.split("\\s+")` pattern ([`WHITESPACE`]).

use super::monitor::Monitor;
use super::renaming_status::RenamingStatus;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::Handle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::ui::standard_bar_string::StandardBarString;
use regex::Regex;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Weak};
use std::time::{Duration, Instant};

/// `java.lang.InterruptedException`, thrown by [`sleep`] when the monitor was
/// interrupted ([`Monitor::interrupt`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterruptedException;

impl std::fmt::Display for InterruptedException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("java.lang.InterruptedException: sleep interrupted")
    }
}

impl std::error::Error for InterruptedException {}

/// `Thread.sleep(millis)` on a monitor thread.  `interrupted` is the monitor's
/// interrupt flag (set by [`Monitor::interrupt`]).  As in Java, a pending
/// interrupt ends the sleep at once, an interrupt during the sleep ends it
/// early, and throwing clears the flag.
pub fn sleep(interrupted: &AtomicBool, millis: u64) -> Result<(), InterruptedException> {
    let deadline = Instant::now() + Duration::from_millis(millis);
    loop {
        if interrupted.swap(false, Ordering::SeqCst) {
            return Err(InterruptedException);
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(());
        }
        std::thread::sleep((deadline - now).min(Duration::from_millis(10)));
    }
}

/// Java's `"\\s+"` split pattern (`\s` is `[ \t\n\x0B\f\r]`), for
/// `utilities::java_lang_string_split`.
pub static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap());

/// Java `MonitorToolKit`.
pub struct MonitorToolKit {
    /// Java field `renamingStatus`.
    renaming_status: RenamingStatus,
    /// Java field `manager`.
    manager: &'static dyn BaseManager,
    /// Java field `axisID`.
    axis_id: AxisID,
    /// Java field `monitor` - optional.
    pub monitor: Option<Weak<dyn Monitor>>,
}

impl MonitorToolKit {
    /// Java `MonitorToolKit(BaseManager, AxisID, Monitor)`.  `monitor` is
    /// optional.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        monitor: Option<Weak<dyn Monitor>>,
    ) -> MonitorToolKit {
        MonitorToolKit {
            renaming_status: RenamingStatus::new(),
            manager,
            axis_id,
            monitor,
        }
    }

    /// Java `initializeProgressBar(String, int, boolean)`.
    pub fn initialize_progress_bar_n_sections(
        &self,
        label: &str,
        n_sections: i32,
        indeterminate_mode: bool,
    ) {
        let monitor: Option<Arc<dyn Monitor>> = self.monitor.as_ref().and_then(Weak::upgrade);
        // Fixed in translation: MonitorToolKit.java:45 calls
        // `monitor.getProcessEndState()` before any null check, although the
        // class documents `monitor` as optional and tests it for null two lines
        // later; with a null monitor Java throws a NullPointerException.  The
        // translation skips the file-lock test when there is no monitor.
        if let Some(monitor) = &monitor {
            if monitor.get_process_end_state() == Some(ProcessEndState::FileLockFailure) {
                return;
            }
        }
        let axis_id = self.axis_id;
        if n_sections == i32::MIN {
            if let Some(monitor) = &monitor {
                if !monitor.has_progress_bar_access() {
                    return;
                }
            }
            let label = label.to_string();
            self.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_string_int_boolean_axis_id(Some(&label), 1, indeterminate_mode, axis_id);
            }));
            let standard_bar_string =
                if self.renaming_status.is_renaming() || self.renaming_status.is_failed() {
                    StandardBarString::BackingUp
                } else if monitor
                    .as_ref()
                    .is_some_and(|monitor| monitor.is_reconnect())
                {
                    StandardBarString::Reconnecting
                } else {
                    StandardBarString::Starting
                };
            let file_name = self.renaming_status.get_file_name();
            let renamed = self.renaming_status.is_renamed();
            let failed = self.renaming_status.is_failed();
            self.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_value_int_standard_bar_string_string_boolean_boolean_axis_id(
                    0,
                    Some(standard_bar_string),
                    file_name.as_deref(),
                    renamed,
                    failed,
                    axis_id,
                );
            }));
        } else {
            let label = label.to_string();
            self.manager.post_main_panel(Box::new(move |panel| {
                panel.set_progress_bar_string_int_boolean_axis_id(Some(&label), n_sections, false, axis_id);
            }));
        }
    }

    /// Java `initializeProgressBar(String, boolean)`.
    pub fn initialize_progress_bar(&self, title: &str, indeterminate_mode: bool) {
        self.initialize_progress_bar_n_sections(title, i32::MIN, indeterminate_mode);
    }

    /// Java `msgLogFileRenaming(File)`.
    pub fn msg_log_file_renaming_file(&self, file: &Path) {
        self.renaming_status.set_renaming_file(Some(file));
        self.initialize_progress_bar(" ", false);
    }

    /// Java `msgLogFileRenaming(LogFile.Handle, boolean)`.
    pub fn msg_log_file_renaming_handle(
        &self,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) {
        self.renaming_status.set_renaming_handle(log_file);
        self.initialize_progress_bar(" ", indeterminate_mode);
    }

    /// Java `msgLogFileRenaming(String)`.
    pub fn msg_log_file_renaming_name(&self, file_name: &str) {
        self.renaming_status.set_renaming_name(Some(file_name));
        self.initialize_progress_bar(" ", false);
    }

    /// Java `msgLogFileRenamed`.
    pub fn msg_log_file_renamed(&self, indeterminate_mode: bool) {
        self.renaming_status.renamed();
        self.initialize_progress_bar(" ", indeterminate_mode);
    }

    /// Java `msgLogFileRenamingFailed`.
    pub fn msg_log_file_renaming_failed(&self) {
        self.renaming_status.set_failed();
        self.initialize_progress_bar(" ", false);
    }

    /// Java `reset`.
    pub fn reset(&self) {
        self.renaming_status.reset();
    }

    /// Java `isRenamed`.
    pub fn is_renamed(&self) -> bool {
        self.renaming_status.is_renamed()
    }
}
