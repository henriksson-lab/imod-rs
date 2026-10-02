//! `IMOD/Etomo/src/etomo/process/Monitor.java`, `ProcessMonitor.java`,
//! `DetachedProcessMonitor.java` and `OutfileProcessMonitor.java`.
//!
//! The four interfaces are one small hierarchy; they share a module because
//! each adds a handful of methods to the one before.  A monitor is a
//! `Runnable` run on its own thread (`BaseProcessManager.startComScriptMonitor`)
//! and shared with the process it watches, hence `Send + Sync` and `&self`.
//!
//! Java stops a monitor thread with `Thread.interrupt()`, which ends its next
//! `Thread.sleep` with an `InterruptedException`.  Rust threads cannot be
//! interrupted, so [`Monitor::interrupt`] is that call: the monitor records it
//! and its sleeps observe it.

use super::process_interface::SystemProcessInterface;
use super::process_messages::ProcessMessages;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::storage::log_file::{Handle, LogFileError};
use std::path::Path;
use std::sync::{Arc, MutexGuard};

/// Java `Monitor extends Runnable`.
pub trait Monitor: Send + Sync {
    /// Java `run`, the monitor thread's body.
    fn run(&self);
    /// Java `isRunning`: true if monitor is running.
    fn is_running(&self) -> bool;
    /// Java `isPausing`.
    fn is_pausing(&self) -> bool;
    /// Java `setWillResume`.
    fn set_will_resume(&self);
    /// Java `halt`: halt the monitor as quickly as possible with a valid
    /// state, but without running end-of-monitor or end-of-process
    /// functionality.
    fn halt(&self);
    /// Java `hasProgressBarAccess`.
    fn has_progress_bar_access(&self) -> bool;
    /// Java `isReconnect`.
    fn is_reconnect(&self) -> bool;
    /// Java `getProcessEndState`.
    fn get_process_end_state(&self) -> Option<ProcessEndState>;
    /// `Thread.interrupt()` on the monitor's thread; see the module comment.
    fn interrupt(&self);
}

/// Java `ProcessMonitor extends Monitor`.
pub trait ProcessMonitor: Monitor {
    /// Java `setProcessEndState`.
    fn set_process_end_state(&self, end_state: ProcessEndState);
    /// Java `kill`.
    fn kill(&self, process: &dyn SystemProcessInterface, axis_id: AxisID);
    /// Java `pause`: return true if able to pause.
    fn pause(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool;
    /// Java `getStatusString`.
    fn get_status_string(&self) -> Option<String>;
    /// Java `getProcessMessages`.
    fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>>;
    /// Java `stop`: stop the monitor.  Doesn't effect the process.
    fn stop(&self);
    /// Java `useMessageReporter`: use MessageReporter instance in the monitor
    /// thread.
    fn use_message_reporter(&self);
    /// Java `dumpState`.
    fn dump_state(&self);
    /// Java `msgLogFileRenaming(LogFile.Handle, boolean)`; the one-argument
    /// overload passes false.
    fn msg_log_file_renaming_handle(&self, log_file: Option<&Arc<Handle>>, indeterminate_mode: bool);
    /// Java `msgLogFileRenaming(String)`.
    fn msg_log_file_renaming_name(&self, file_name: &str);
    /// Java `msgLogFileRenaming(File)`.
    fn msg_log_file_renaming_file(&self, file: &Path);
    /// Java `msgLogFileRenamed`.
    fn msg_log_file_renamed(&self);
    /// Java `msgLogFileRenamingFailed`.
    fn msg_log_file_renaming_failed(&self);
}

/// Java `DetachedProcessMonitor extends ProcessMonitor`.
pub trait DetachedProcessMonitor: ProcessMonitor {
    /// Java `isProcessRunning`.
    fn is_process_running(&self) -> bool;
    /// Java `getProcessOutputFileName`.
    fn get_process_output_file_name(&self) -> Result<String, LogFileError>;
    /// Java `setProcess`.
    fn set_process(&self, process: Arc<dyn SystemProcessInterface>);
}

/// Java `OutfileProcessMonitor extends DetachedProcessMonitor`.
pub trait OutfileProcessMonitor: DetachedProcessMonitor {
    /// Java `getPid`.
    fn get_pid(&self) -> Option<String>;
    /// Java `endMonitor`.
    fn end_monitor(&self, end_state: ProcessEndState);
    /// Java `getSubProcessName`.
    fn get_sub_process_name(&self) -> Option<String>;
}
