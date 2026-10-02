//! `IMOD/Etomo/src/etomo/process/AxisProcessData.java`.
//!
//! Per-axis bookkeeping of the running process and its monitor.  The Java
//! fields are unsynchronized and touched from the process, monitor and UI
//! threads; they sit behind one `Mutex` here, which is released before the
//! bounded wait in `waitForMonitor` so a finishing monitor can still call in.
//!
//! A monitor "thread" is its [`Monitor`] (which `interrupt()` reaches, see
//! `monitor.rs`) plus the `JoinHandle` that `join()` needs.

use super::monitor::Monitor;
use super::process_data::ProcessData;
use super::process_interface::{ProcessInterface, same_process};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusMediator;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::clean_print::{CLEAN_PRINT, TEST_FLAG};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A Java `Thread` running a monitor.
pub struct MonitorThread {
    pub handle: Option<JoinHandle<()>>,
    pub monitor: Arc<dyn Monitor>,
}

impl MonitorThread {
    /// `Thread.interrupt()`.
    fn interrupt(&self) {
        self.monitor.interrupt();
    }
}

#[derive(Default)]
struct State {
    killed_list: HashMap<String, String>,
    thread_axis_a: Option<Arc<dyn ProcessInterface>>,
    thread_axis_b: Option<Arc<dyn ProcessInterface>>,
    process_monitor_thread_a: Option<MonitorThread>,
    process_monitor_thread_b: Option<MonitorThread>,
    subprocess_monitor_thread_array_a: Option<Vec<MonitorThread>>,
    block_axis_a: bool,
    block_axis_b: bool,
    monitor_a: Option<Arc<dyn Monitor>>,
    monitor_b: Option<Arc<dyn Monitor>>,
    /// Java `submonitorArrayA`; nothing in the source ever assigns it, so it
    /// stays empty.
    submonitor_array_a: Option<Vec<Arc<dyn Monitor>>>,
}

/// Java final class `AxisProcessData`.
pub struct AxisProcessData {
    state: Mutex<State>,
    saved_process_data_a: Arc<Mutex<ProcessData>>,
    saved_process_data_b: Arc<Mutex<ProcessData>>,
    busy_status_mediator: Arc<BusyStatusMediator>,
}

/// Which `clearThread` overload: they differ in how much monitor state they
/// clear.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ClearKind {
    /// `clearThread(ComScriptProcess)`.
    ComScript,
    /// `clearThread(ReconnectProcess)`.
    Reconnect,
    /// `clearThread(DetachedProcess)`, `clearThread(SimpleProcess)` and
    /// `clearThread(BackgroundProcess)`.
    Other,
}

impl AxisProcessData {
    /// Java `AxisProcessData(BaseManager)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        busy_status_mediator: Arc<BusyStatusMediator>,
    ) -> AxisProcessData {
        AxisProcessData {
            state: Mutex::new(State {
                block_axis_a: true,
                block_axis_b: true,
                ..State::default()
            }),
            saved_process_data_a: Arc::new(Mutex::new(ProcessData::new(
                Some(AxisID::First),
                Some(manager),
            ))),
            saved_process_data_b: Arc::new(Mutex::new(ProcessData::new(
                Some(AxisID::Second),
                Some(manager),
            ))),
            busy_status_mediator,
        }
    }

    /// Java `dumpState`.
    pub fn dump_state(&self) {
        eprintln!("{}", self.to_source_string());
    }

    /// Java `toString`.
    pub fn to_source_string(&self) -> String {
        let state = self.state.lock().unwrap();
        format!(
            "[threadAxisA={},threadAxisB={},\nprocessMonitorThreadA={},processMonitorThreadB={},\nsubprocessMonitorThreadArrayA:{},\nkilledList={:?},blockAxisA:{},blockAxisB:{}]",
            state.thread_axis_a.is_some(),
            state.thread_axis_b.is_some(),
            state.process_monitor_thread_a.is_some(),
            state.process_monitor_thread_b.is_some(),
            state
                .subprocess_monitor_thread_array_a
                .as_ref()
                .map_or(0, Vec::len),
            state.killed_list,
            state.block_axis_a,
            state.block_axis_b
        )
    }

    /// Java `isThreadAxisNull`.
    pub fn is_thread_axis_null(&self, axis_id: AxisID) -> bool {
        let state = self.state.lock().unwrap();
        if axis_id == AxisID::Second {
            return state.thread_axis_b.is_none();
        }
        state.thread_axis_a.is_none()
    }

    /// Java `isPausing`.
    pub fn is_pausing(&self, axis_id: AxisID) -> bool {
        let state = self.state.lock().unwrap();
        if axis_id == AxisID::Second
            && let Some(monitor_b) = &state.monitor_b
        {
            return monitor_b.is_pausing();
        }
        if let Some(monitor_a) = &state.monitor_a {
            return monitor_a.is_pausing();
        }
        false
    }

    /// Java `setWillResume`.
    pub fn set_will_resume(&self, axis_id: AxisID) {
        let state = self.state.lock().unwrap();
        if axis_id == AxisID::Second {
            if let Some(monitor_b) = &state.monitor_b {
                monitor_b.set_will_resume();
            }
        } else {
            if let Some(monitor_a) = &state.monitor_a {
                monitor_a.set_will_resume();
            }
            if let Some(submonitors) = &state.submonitor_array_a {
                for monitor in submonitors {
                    monitor.set_will_resume();
                }
            }
        }
    }

    /// Java `mapAxisThread`: save the process thread reference for the
    /// appropriate axis.
    pub fn map_axis_thread(&self, thread: Option<Arc<dyn ProcessInterface>>, axis_id: AxisID) {
        let busy = thread.is_some();
        {
            let mut state = self.state.lock().unwrap();
            if axis_id == AxisID::Second {
                state.thread_axis_b = thread;
            } else {
                state.thread_axis_a = thread;
            }
        }
        self.busy_status_mediator.msg_thread_changed(axis_id, busy);
    }

    /// Java `getThread`.
    pub fn get_thread(&self, axis_id: AxisID) -> Option<Arc<dyn ProcessInterface>> {
        let state = self.state.lock().unwrap();
        if axis_id == AxisID::Second {
            state.thread_axis_b.clone()
        } else {
            state.thread_axis_a.clone()
        }
    }

    /// Java `mapAxisProcessMonitor`: save the process monitor thread
    /// reference for the appropriate axis.
    pub fn map_axis_process_monitor(
        &self,
        process_monitor_thread: Option<JoinHandle<()>>,
        monitor: Option<Arc<dyn Monitor>>,
        axis_id: AxisID,
    ) {
        let mut state = self.state.lock().unwrap();
        let thread = monitor.clone().map(|monitor| MonitorThread {
            handle: process_monitor_thread,
            monitor,
        });
        if axis_id == AxisID::Second {
            state.process_monitor_thread_b = thread;
            state.monitor_b = monitor;
        } else {
            state.process_monitor_thread_a = thread;
            state.monitor_a = monitor;
            // Assume that the submonitors subordinate to or created by the monitor.
            if let Some(array) = &mut state.subprocess_monitor_thread_array_a {
                array.clear();
            }
            if let Some(array) = &mut state.submonitor_array_a {
                array.clear();
            }
        }
    }

    /// Java `mapAxisSubprocessMonitor`.
    pub fn map_axis_subprocess_monitor(
        &self,
        subprocess_monitor_thread: Option<JoinHandle<()>>,
        submonitor: Arc<dyn Monitor>,
        axis_id: AxisID,
    ) {
        if axis_id != AxisID::Second {
            let mut state = self.state.lock().unwrap();
            state
                .subprocess_monitor_thread_array_a
                .get_or_insert_with(Vec::new)
                .push(MonitorThread {
                    handle: subprocess_monitor_thread,
                    monitor: submonitor,
                });
        }
    }

    /// Java `haltMonitorThread`: tells the monitor to halt and waits for the
    /// monitor thread to end.
    pub fn halt_monitor_thread(&self, axis_id: AxisID) {
        let (monitor, thread, submonitors, subthreads) = {
            let mut state = self.state.lock().unwrap();
            if axis_id == AxisID::Second {
                (
                    state.monitor_b.clone(),
                    state
                        .process_monitor_thread_b
                        .as_mut()
                        .and_then(|thread| thread.handle.take()),
                    None,
                    None,
                )
            } else {
                (
                    state.monitor_a.clone(),
                    state
                        .process_monitor_thread_a
                        .as_mut()
                        .and_then(|thread| thread.handle.take()),
                    state.submonitor_array_a.clone(),
                    state.subprocess_monitor_thread_array_a.as_mut().map(|array| {
                        array
                            .iter_mut()
                            .filter_map(|thread| thread.handle.take())
                            .collect::<Vec<_>>()
                    }),
                )
            }
        };
        let Some(monitor) = monitor else {
            return;
        };
        monitor.halt();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
        if let Some(submonitors) = submonitors {
            for monitor in submonitors {
                monitor.halt();
            }
        }
        if let Some(subthreads) = subthreads {
            for thread in subthreads {
                let _ = thread.join();
            }
        }
    }

    /// Java private `waitForMonitor`: let the monitor finish up before
    /// unlocking the axis.
    fn wait_for_monitor(monitor: Option<&Arc<dyn Monitor>>, axis_id: AxisID) {
        let thread_id = format!("{:?}", std::thread::current().id());
        CLEAN_PRINT.print_with_flags(
            Some(TEST_FLAG),
            Some(&format!(
                "Waiting for monitor {} to end. (Thread {thread_id})",
                axis_id.get_upper_case_extension()
            )),
        );
        let Some(monitor) = monitor else {
            return;
        };
        for _ in 0..2000 {
            if !monitor.is_running() {
                CLEAN_PRINT.print_with_flags(
                    Some(TEST_FLAG),
                    Some(&format!(
                        "Monitor {} finished. (Thread {thread_id})",
                        axis_id.get_upper_case_extension()
                    )),
                );
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        CLEAN_PRINT.print_with_flags(
            Some(TEST_FLAG),
            Some(&format!(
                "Warning:Monitor {}: wait timed out. (Thread {thread_id})",
                axis_id.get_upper_case_extension()
            )),
        );
    }

    /// Java `clearThread(ComScriptProcess)`, `clearThread(ReconnectProcess)`,
    /// `clearThread(DetachedProcess)`, `clearThread(SimpleProcess)` and
    /// `clearThread(BackgroundProcess)`, selected by `kind`.
    ///
    /// The `ComScriptProcess` and `ReconnectProcess` overloads interrupt the
    /// subprocess monitor threads in a loop that removes element `i` while
    /// counting `i` up to the original size (`AxisProcessData.java:307-316`,
    /// `:346-353`); with two or more such threads that skips every other one
    /// and then reads past the end of the list (`IndexOutOfBoundsException`).
    /// Fixed in translation (`BUGS.md`): every thread is interrupted and the
    /// list is cleared, which is what the loop and the `clear()` after it
    /// intend.
    pub fn clear_thread(&self, script: &dyn ProcessInterface, kind: ClearKind) {
        // Null out the correct thread
        let is_a = {
            let state = self.state.lock().unwrap();
            state
                .thread_axis_a
                .as_deref()
                .is_some_and(|thread| same_process(thread, script))
        };
        if is_a {
            let monitor = {
                let mut state = self.state.lock().unwrap();
                if kind != ClearKind::Other
                    && let Some(thread) = state.process_monitor_thread_a.take()
                {
                    thread.interrupt();
                }
                state.monitor_a.clone()
            };
            AxisProcessData::wait_for_monitor(monitor.as_ref(), AxisID::First);
            {
                let mut state = self.state.lock().unwrap();
                state.monitor_a = None;
                state.thread_axis_a = None;
            }
            self.busy_status_mediator
                .msg_thread_changed(AxisID::Only, false);
            if kind != ClearKind::Other {
                let mut state = self.state.lock().unwrap();
                if let Some(array) = &mut state.subprocess_monitor_thread_array_a {
                    for thread in array.iter() {
                        thread.interrupt();
                    }
                    array.clear();
                }
                if kind == ClearKind::ComScript
                    && let Some(array) = &mut state.submonitor_array_a
                {
                    array.clear();
                }
            }
        }
        let is_b = {
            let state = self.state.lock().unwrap();
            state
                .thread_axis_b
                .as_deref()
                .is_some_and(|thread| same_process(thread, script))
        };
        if is_b {
            let monitor = {
                let mut state = self.state.lock().unwrap();
                if kind != ClearKind::Other
                    && let Some(thread) = state.process_monitor_thread_b.take()
                {
                    thread.interrupt();
                }
                state.monitor_b.clone()
            };
            AxisProcessData::wait_for_monitor(monitor.as_ref(), AxisID::Second);
            {
                let mut state = self.state.lock().unwrap();
                state.monitor_b = None;
                state.thread_axis_b = None;
            }
            self.busy_status_mediator
                .msg_thread_changed(AxisID::Second, false);
        }
    }

    /// Java `getSavedProcessData`.
    pub fn get_saved_process_data(&self, axis_id: AxisID) -> Arc<Mutex<ProcessData>> {
        if axis_id == AxisID::Second {
            return Arc::clone(&self.saved_process_data_b);
        }
        Arc::clone(&self.saved_process_data_a)
    }

    /// Java `putKilledList`.
    pub fn put_killed_list(&self, process_id: &str) {
        self.state
            .lock()
            .unwrap()
            .killed_list
            .insert(process_id.to_owned(), String::new());
    }

    /// Java `containsKeyKilledList`.
    pub fn contains_key_killed_list(&self, pid_field: &str) -> bool {
        self.state
            .lock()
            .unwrap()
            .killed_list
            .contains_key(pid_field)
    }

    /// Java `isBlockAxis`.
    pub fn is_block_axis(&self, axis_id: AxisID) -> bool {
        let state = self.state.lock().unwrap();
        if axis_id == AxisID::Second {
            return state.block_axis_b;
        }
        state.block_axis_a
    }

    /// Java `setBlockAxis`.
    pub fn set_block_axis(&self, axis_id: AxisID, input: bool) {
        let mut state = self.state.lock().unwrap();
        if axis_id == AxisID::Second {
            state.block_axis_b = input;
        } else {
            state.block_axis_a = input;
        }
    }
}
