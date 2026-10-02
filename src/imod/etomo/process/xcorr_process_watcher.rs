//! `IMOD/Etomo/src/etomo/process/XcorrProcessWatcher.java`.
//!
//! Runs a `BlendmontProcessMonitor` (when blendmont runs first) and then a
//! `TiltxcorrProcessWatcher` as child monitors, each on its own thread.  A
//! child of either class is held as
//! `Arc<LogFileProcessMonitorOf<dyn LogFileProcessMonitorImpl>>` (Java's
//! `LogFileProcessMonitor curChildMonitor`); Java's `Thread.interrupt()` on a
//! child's thread is that child's `Monitor::interrupt`.

use super::blendmont_process_monitor::BlendmontProcessMonitor;
use super::log_file_process_monitor::{LogFileProcessMonitorImpl, LogFileProcessMonitorOf};
use super::monitor::{Monitor, ProcessMonitor};
use super::monitor_tool_kit::{self, MonitorToolKit};
use super::process_interface::SystemProcessInterface;
use super::process_messages::ProcessMessages;
use super::tiltxcorr_process_watcher::TiltxcorrProcessWatcher;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::Mode;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::storage::log_file::Handle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

/// A child monitor of any `LogFileProcessMonitor` subclass.
type ChildMonitor = Arc<LogFileProcessMonitorOf<dyn LogFileProcessMonitorImpl>>;

/// Java `XcorrProcessWatcher`.
pub struct XcorrProcessWatcher {
    /// Java field `busyStatusMediator`.
    busy_status_mediator: Arc<BusyStatusMediator>,
    /// Java field `toolKit`.
    tool_kit: MonitorToolKit,

    /// Java field `curChildMonitor`; the mutex is Java's `curChildMonitorLock`.
    cur_child_monitor: Mutex<Option<ChildMonitor>>,

    /// Java field `manager`.
    manager: &'static dyn BaseManager,
    /// Java field `axisID`.
    axis_id: AxisID,
    /// Java field `blendmont`.
    blendmont: bool,
    /// Java field `endState`; the mutex stands for the `synchronized` of
    /// `setProcessEndState`/`getProcessEndState`.
    end_state: Mutex<Option<ProcessEndState>>,
    /// Java field `stop`.
    stop: AtomicBool,
    /// Java `volatile` field `running`.
    running: AtomicBool,
    /// `Thread.interrupt()` on this monitor's thread.
    interrupted: AtomicBool,
}

impl XcorrProcessWatcher {
    /// Java `XcorrProcessWatcher(BaseManager, AxisID, boolean)`.  Construct a
    /// xcorr process watcher.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        blendmont: bool,
    ) -> Arc<XcorrProcessWatcher> {
        Arc::new_cyclic(|this: &Weak<XcorrProcessWatcher>| {
            let monitor: Weak<dyn Monitor> = this.clone();
            let tool_kit = MonitorToolKit::new(manager, axis_id, Some(monitor));
            let busy_status_mediator = manager.get_busy_status_mediator();
            busy_status_mediator.msg_monitor_constructed(axis_id);
            XcorrProcessWatcher {
                busy_status_mediator,
                tool_kit,
                cur_child_monitor: Mutex::new(None),
                manager,
                axis_id,
                blendmont,
                end_state: Mutex::new(None),
                stop: AtomicBool::new(false),
                running: AtomicBool::new(false),
                interrupted: AtomicBool::new(false),
            }
        })
    }

    /// Java `getProcessName`.
    pub fn get_process_name(&self) -> Option<ProcessName> {
        None
    }

    /// Java private `endChildMonitor`.  `child_thread` is true when Java passes
    /// the child's (non-null) thread.
    fn end_child_monitor(
        &self,
        child_monitor: Option<&ChildMonitor>,
        child_thread: bool,
        cur_end_state: ProcessEndState,
    ) {
        if let Some(child_monitor) = child_monitor {
            child_monitor.halt_process(child_thread, Some(cur_end_state));
        }
    }

    /// The body of Java `run`'s `try` block.
    fn run_children(&self) {
        self.set_running(true);
        if self.blendmont {
            let cur_child_monitor: ChildMonitor = {
                let mut lock = self.cur_child_monitor.lock().unwrap();
                let child: ChildMonitor =
                    BlendmontProcessMonitor::new(self.manager, self.axis_id, Mode::Xcorr);
                *lock = Some(child.clone());
                child
            };
            cur_child_monitor.set_last_process(false);
            let blendmont_thread = cur_child_monitor.clone();
            std::thread::spawn(move || blendmont_thread.run());
            let _ = monitor_tool_kit::sleep(&self.interrupted, 10);
            while self.is_running()
                && cur_child_monitor.is_running()
                && !cur_child_monitor.is_done()
                && !self.stop.load(Ordering::SeqCst)
            {
                if let Err(e) = monitor_tool_kit::sleep(&self.interrupted, 100) {
                    let end_state = cur_child_monitor
                        .get_process_end_state()
                        .unwrap_or(ProcessEndState::Done);
                    self.set_process_end_state(end_state);
                    // not expecting any exception here
                    eprintln!("{e}");
                    // send an interrupt to the monitor so it can clean up
                    cur_child_monitor.interrupt();
                    return;
                }
            }
            let end_state = cur_child_monitor
                .get_process_end_state()
                .unwrap_or(ProcessEndState::Done);
            self.end_child_monitor(Some(&cur_child_monitor), true, end_state);
            *self.cur_child_monitor.lock().unwrap() = None;
        }
        let cur_child_monitor: ChildMonitor = {
            let mut lock = self.cur_child_monitor.lock().unwrap();
            let child: ChildMonitor = TiltxcorrProcessWatcher::new_blendmont_ran(
                self.manager,
                self.axis_id,
                self.blendmont,
            );
            *lock = Some(child.clone());
            child
        };
        let tiltxcorr_thread = cur_child_monitor.clone();
        std::thread::spawn(move || tiltxcorr_thread.run());
        let _ = monitor_tool_kit::sleep(&self.interrupted, 10);
        while self.is_running()
            && cur_child_monitor.is_running()
            && !cur_child_monitor.is_done()
            && !self.stop.load(Ordering::SeqCst)
        {
            if monitor_tool_kit::sleep(&self.interrupted, 100).is_err() {
                // only expecting interrupt here
                // send an interrupt to the monitor so it can clean up
                cur_child_monitor.interrupt();
            }
        }
        let end_state = cur_child_monitor
            .get_process_end_state()
            .unwrap_or(ProcessEndState::Done);
        self.end_child_monitor(Some(&cur_child_monitor), true, end_state);
        *self.cur_child_monitor.lock().unwrap() = None;
        self.set_process_end_state(end_state);
    }

    /// Java private synchronized `setRunning`.
    fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
    }
}

impl Monitor for XcorrProcessWatcher {
    /// Java `run`.
    fn run(&self) {
        self.run_children();
        // finally
        self.busy_status_mediator.msg_monitor_stopped(self.axis_id);
        self.set_running(false);
    }

    /// Java `isRunning`.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Java `isPausing`.
    fn is_pausing(&self) -> bool {
        false
    }

    /// Java `setWillResume`.
    fn set_will_resume(&self) {}

    /// Java `halt`.
    fn halt(&self) {}

    /// Java `hasProgressBarAccess`.
    fn has_progress_bar_access(&self) -> bool {
        true
    }

    /// Java `isReconnect`.
    fn is_reconnect(&self) -> bool {
        false
    }

    /// Java `synchronized final getProcessEndState`.
    fn get_process_end_state(&self) -> Option<ProcessEndState> {
        *self.end_state.lock().unwrap()
    }

    fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
    }
}

impl ProcessMonitor for XcorrProcessWatcher {
    /// Java `synchronized final setProcessEndState`.
    fn set_process_end_state(&self, end_state: ProcessEndState) {
        let mut this_end_state = self.end_state.lock().unwrap();
        *this_end_state = Some(match *this_end_state {
            None => end_state,
            Some(existing) => ProcessEndState::precedence(existing, end_state),
        });
    }

    /// Java `kill`.
    fn kill(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) {
        *self.end_state.lock().unwrap() = Some(ProcessEndState::Killed);
        process.signal_kill(axis_id);
    }

    /// Java `pause`.
    fn pause(&self, process: &dyn SystemProcessInterface, axis_id: AxisID) -> bool {
        let _ = (process, axis_id);
        // Fixed in translation: XcorrProcessWatcher.java:186 throws an unchecked
        // IllegalStateException ("pause illegal in this monitor") out of the
        // caller.  The translation reports it and answers "not able to pause".
        eprintln!("java.lang.IllegalStateException: pause illegal in this monitor");
        false
    }

    /// Java `getStatusString`.
    fn get_status_string(&self) -> Option<String> {
        None
    }

    /// Java `getProcessMessages`.
    fn get_process_messages(&self) -> Option<MutexGuard<'_, ProcessMessages>> {
        None
    }

    /// Java `stop`.
    fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Java `useMessageReporter`.
    fn use_message_reporter(&self) {}

    /// Java `dumpState`.
    fn dump_state(&self) {}

    /// Java final `msgLogFileRenaming(LogFile.Handle, boolean)` and its
    /// one-argument overload.
    fn msg_log_file_renaming_handle(
        &self,
        log_file: Option<&Arc<Handle>>,
        indeterminate_mode: bool,
    ) {
        let cur_child_monitor = self.cur_child_monitor.lock().unwrap();
        if let Some(cur_child_monitor) = &*cur_child_monitor {
            cur_child_monitor.msg_log_file_renaming_handle(log_file, indeterminate_mode);
        }
    }

    /// Java final `msgLogFileRenaming(String)`.
    fn msg_log_file_renaming_name(&self, file_name: &str) {
        let cur_child_monitor = self.cur_child_monitor.lock().unwrap();
        if let Some(cur_child_monitor) = &*cur_child_monitor {
            cur_child_monitor.msg_log_file_renaming_name(file_name);
        }
    }

    /// Java final `msgLogFileRenaming(File)`.
    fn msg_log_file_renaming_file(&self, file: &Path) {
        let cur_child_monitor = self.cur_child_monitor.lock().unwrap();
        if let Some(cur_child_monitor) = &*cur_child_monitor {
            cur_child_monitor.msg_log_file_renaming_file(file);
        }
    }

    /// Java final `msgLogFileRenamed`.
    fn msg_log_file_renamed(&self) {
        let cur_child_monitor = self.cur_child_monitor.lock().unwrap();
        if let Some(cur_child_monitor) = &*cur_child_monitor {
            cur_child_monitor.msg_log_file_renamed();
        }
    }

    /// Java final `msgLogFileRenamingFailed`.
    fn msg_log_file_renaming_failed(&self) {
        let cur_child_monitor = self.cur_child_monitor.lock().unwrap();
        if let Some(cur_child_monitor) = &*cur_child_monitor {
            cur_child_monitor.msg_log_file_renaming_failed();
        }
    }
}
