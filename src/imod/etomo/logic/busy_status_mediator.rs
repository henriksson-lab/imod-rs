//! `IMOD/Etomo/src/etomo/logic/BusyStatusMediator.java` and
//! `IMOD/Etomo/src/etomo/logic/BusyStatusListener.java`.
//!
//! Every Java method is `synchronized`; the whole state sits behind one
//! `Mutex`.  The listeners are Swing panels, so they are held as
//! [`EdtRef`]s and `sendMessage` posts each notification to the event
//! dispatch thread, where the Java called them on whatever thread reported
//! the change (usually a process thread).

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::event_queue::{self, EdtRef};
use crate::imod::etomo::util::event_queue::{ReentrantGuard, ReentrantLock};
use std::sync::{Arc, Mutex};

/// Java `BusyStatusListener`.
pub trait BusyStatusListener {
    fn msg_busy_status_changed(&self, axis_id: AxisID, busy_status: bool);
}

/// Java `DELAY`.
pub const DELAY: u64 = 15;

#[derive(Default)]
struct State {
    listeners: Vec<Arc<EdtRef<dyn BusyStatusListener>>>,
    busy_status_a: bool,
    busy_status_b: bool,
    process_series_a: bool,
    process_series_b: bool,
    thread_a: bool,
    thread_b: bool,
    monitor_a: bool,
    monitor_b: bool,
    process_run_a: bool,
    process_run_b: bool,
    kill_process_button_a: bool,
    kill_process_button_b: bool,
}

/// Java final class `BusyStatusMediator`.
#[derive(Default)]
pub struct BusyStatusMediator {
    /// The object's Java monitor: every synchronized method takes it, and so does
    /// a caller's `synchronized (busyStatusMediator) { ... }` block, which may then
    /// call those methods (Java monitors are re-entrant).
    monitor: ReentrantLock,
    state: Mutex<State>,
}

impl BusyStatusMediator {
    /// Java `BusyStatusMediator()`.
    pub fn new() -> BusyStatusMediator {
        BusyStatusMediator::default()
    }

    /// Java `addBusyStatusListener`.
    pub fn add_busy_status_listener(&self, listener: Option<Arc<EdtRef<dyn BusyStatusListener>>>) {
        let _monitor = self.monitor.lock();
        if let Some(listener) = listener {
            self.state.lock().unwrap().listeners.push(listener);
        }
    }

    /// Java `removeBusyStatusListener`.
    pub fn remove_busy_status_listener(
        &self,
        listener: Option<&Arc<EdtRef<dyn BusyStatusListener>>>,
    ) {
        let _monitor = self.monitor.lock();
        if let Some(listener) = listener {
            let mut state = self.state.lock().unwrap();
            if let Some(index) = state.listeners.iter().position(|l| l.ptr_eq(listener)) {
                state.listeners.remove(index);
            }
        }
    }

    /// Java `msgProcessSeriesConstructed`.
    pub fn msg_process_series_constructed(&self, axis_id: AxisID) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_process_series(&mut state, axis_id, true);
    }

    /// Java `msgProcessSeriesDone`.
    pub fn msg_process_series_done(&self, axis_id: AxisID) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_process_series(&mut state, axis_id, false);
    }

    /// Java `msgProcessConstructed`.
    pub fn msg_process_constructed(&self, axis_id: AxisID) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_process_run(&mut state, axis_id, true);
    }

    /// Java `msgProcessDone`.
    pub fn msg_process_done(&self, axis_id: AxisID) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_process_run(&mut state, axis_id, false);
    }

    /// Java `msgThreadChanged`.
    pub fn msg_thread_changed(&self, axis_id: AxisID, busy: bool) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        if busy == map_thread(&state, axis_id) {
            return;
        }
        set_thread(&mut state, axis_id, busy);
        update_busy_status(&mut state, axis_id);
    }

    /// Java `msgMonitorConstructed`.
    pub fn msg_monitor_constructed(&self, axis_id: AxisID) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_monitor(&mut state, axis_id, true);
    }

    /// Java `msgKillProcessButton`.
    pub fn msg_kill_process_button(&self, axis_id: AxisID, enabled: bool) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_kill_process_button(&mut state, axis_id, enabled);
    }

    /// Java `msgMonitorStopped`.
    pub fn msg_monitor_stopped(&self, axis_id: AxisID) {
        let _monitor = self.monitor.lock();
        let mut state = self.state.lock().unwrap();
        update_monitor(&mut state, axis_id, false);
    }

    /// Java `synchronized (busyStatusMediator)`: holds the object's monitor for
    /// the guard's life.
    pub fn synchronized(&self) -> ReentrantGuard<'_> {
        self.monitor.lock()
    }

    /// The busy status the mediator last sent for `axis_id` (Java private
    /// `mapBusyStatus`, read by the translation's tests).
    pub fn is_busy(&self, axis_id: AxisID) -> bool {
        map_busy_status(&self.state.lock().unwrap(), axis_id)
    }
}

/// Java private `updateMonitor`.
fn update_monitor(state: &mut State, axis_id: AxisID, busy: bool) {
    if busy == map_monitor(state, axis_id) {
        return;
    }
    set_monitor(state, axis_id, busy);
    update_busy_status(state, axis_id);
}

/// Java private `updateKillProcessButton`.
fn update_kill_process_button(state: &mut State, axis_id: AxisID, enabled: bool) {
    if enabled == map_kill_process_button(state, axis_id) {
        return;
    }
    set_kill_process_button(state, axis_id, enabled);
    update_busy_status(state, axis_id);
}

/// Java private `updateProcessSeries`.
fn update_process_series(state: &mut State, axis_id: AxisID, busy: bool) {
    if busy != map_process_series(state, axis_id) {
        set_process_series(state, axis_id, busy);
        update_busy_status(state, axis_id);
    }
}

/// Java private `updateProcessRun`.
fn update_process_run(state: &mut State, axis_id: AxisID, busy: bool) {
    if busy == map_process_run(state, axis_id) {
        return;
    }
    set_process_run(state, axis_id, busy);
    update_busy_status(state, axis_id);
}

/// Java private `calcBusyStatus`.
fn calc_busy_status(state: &State, axis_id: AxisID) -> bool {
    map_process_series(state, axis_id)
        || map_thread(state, axis_id)
        || map_monitor(state, axis_id)
        || map_process_run(state, axis_id)
        || map_kill_process_button(state, axis_id)
}

/// Java private `updateBusyStatus`.
fn update_busy_status(state: &mut State, axis_id: AxisID) {
    // If at least one of processSeries, Thread, Monitor, etc is on, then the busy
    // status is true. If none of them on, then the busy status is false.
    let busy_status = calc_busy_status(state, axis_id);
    if busy_status != map_busy_status(state, axis_id) {
        set_busy_status(state, axis_id, busy_status);
        send_message(state, axis_id, busy_status);
    }
}

/// Java private `sendMessage`.
fn send_message(state: &State, axis_id: AxisID, busy_status: bool) {
    let mut sleep = 0;
    // waits needed for ending busy status.
    if !busy_status {
        // Short wait to protect a process that's finishing.
        sleep = DELAY;
    }
    if sleep > 0 {
        std::thread::sleep(std::time::Duration::from_millis(sleep));
    }
    for listener in &state.listeners {
        // Java calls the listener on this thread; the listener is a Swing panel.
        let listener = Arc::clone(listener);
        event_queue::invoke_later(move || {
            listener.get().msg_busy_status_changed(axis_id, busy_status);
        });
    }
}

fn map_process_series(state: &State, axis_id: AxisID) -> bool {
    if axis_id == AxisID::Second {
        return state.process_series_b;
    }
    state.process_series_a
}

fn map_process_run(state: &State, axis_id: AxisID) -> bool {
    if axis_id == AxisID::Second {
        return state.process_run_b;
    }
    state.process_run_a
}

fn set_process_series(state: &mut State, axis_id: AxisID, busy: bool) {
    if axis_id == AxisID::Second {
        state.process_series_b = busy;
    } else {
        state.process_series_a = busy;
    }
}

fn set_process_run(state: &mut State, axis_id: AxisID, busy: bool) {
    if axis_id == AxisID::Second {
        state.process_run_b = busy;
    } else {
        state.process_run_a = busy;
    }
}

fn map_monitor(state: &State, axis_id: AxisID) -> bool {
    if axis_id == AxisID::Second {
        return state.monitor_b;
    }
    state.monitor_a
}

fn map_kill_process_button(state: &State, axis_id: AxisID) -> bool {
    if axis_id == AxisID::Second {
        return state.kill_process_button_b;
    }
    state.kill_process_button_a
}

fn set_monitor(state: &mut State, axis_id: AxisID, busy: bool) {
    if axis_id == AxisID::Second {
        state.monitor_b = busy;
    } else {
        state.monitor_a = busy;
    }
}

fn set_kill_process_button(state: &mut State, axis_id: AxisID, enabled: bool) {
    if axis_id == AxisID::Second {
        state.kill_process_button_b = enabled;
    } else {
        state.kill_process_button_a = enabled;
    }
}

fn map_thread(state: &State, axis_id: AxisID) -> bool {
    if axis_id == AxisID::Second {
        return state.thread_b;
    }
    state.thread_a
}

fn set_thread(state: &mut State, axis_id: AxisID, busy: bool) {
    if axis_id == AxisID::Second {
        state.thread_b = busy;
    } else {
        state.thread_a = busy;
    }
}

fn map_busy_status(state: &State, axis_id: AxisID) -> bool {
    if axis_id == AxisID::Second {
        return state.busy_status_b;
    }
    state.busy_status_a
}

fn set_busy_status(state: &mut State, axis_id: AxisID, busy: bool) {
    if axis_id == AxisID::Second {
        state.busy_status_b = busy;
    } else {
        state.busy_status_a = busy;
    }
}
