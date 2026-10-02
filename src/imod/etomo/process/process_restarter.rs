//! `IMOD/Etomo/src/etomo/process/ProcessRestarter.java`.
//!
//! A singleton that restarts failed intermittent background processes on its own
//! thread.  Java's `synchronized (this)` blocks guard `processList` and `running`;
//! each field sits behind its own lock here (no block touches both), and `stop` is an
//! atomic, as the unsynchronized Java field is read from several threads.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use crate::imod::etomo::process::intermittent_background_process::IntermittentBackgroundProcess;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java static final `INSTANCE`.
pub static INSTANCE: LazyLock<ProcessRestarter> = LazyLock::new(ProcessRestarter::new);

/// Java final class `ProcessRestarter implements Runnable`.
pub struct ProcessRestarter {
    /// Java private final `processList`, `new ArrayList()`.
    process_list: Mutex<Vec<Arc<IntermittentBackgroundProcess>>>,
    /// Java private `stop`, initialised to false.
    stop: AtomicBool,
    /// Java private `running`, initialised to false.
    running: Mutex<bool>,
}

impl ProcessRestarter {
    /// Java private `ProcessRestarter()`.
    fn new() -> ProcessRestarter {
        ProcessRestarter {
            process_list: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            running: Mutex::new(false),
        }
    }

    /// Java synchronized package-private `addProcess(IntermittentBackgroundProcess)`.
    /// Adds a process to processList if the process isn't already on the processList.
    /// `List.contains` uses `equals`, which the process does not override: identity.
    pub fn add_process(&self, process: Arc<IntermittentBackgroundProcess>) {
        let mut process_list = self.process_list.lock().unwrap();
        if !process_list
            .iter()
            .any(|listed| Arc::ptr_eq(listed, &process))
        {
            process_list.push(process);
        }
    }

    /// Java package-private `restart()`: `new Thread(INSTANCE).start()`.
    pub fn restart(&self) {
        if self.stop.load(Ordering::SeqCst) {
            return;
        }
        std::thread::spawn(|| INSTANCE.run());
    }

    /// Java `run()`.  Try to restart all the processes in processList.
    pub fn run(&self) {
        if self.stop.load(Ordering::SeqCst) {
            return;
        }
        // Only one thread can at a time can set running to true and continue.
        {
            let mut running = self.running.lock().unwrap();
            if *running {
                return;
            }
            *running = true;
        }
        // Work with the existing part of the list and ignore new entries created by
        // other failed processes.
        let size: usize;
        {
            size = self.process_list.lock().unwrap().len();
        }
        for i in 0..size {
            if self.stop.load(Ordering::SeqCst) {
                break;
            }
            let process: Option<Arc<IntermittentBackgroundProcess>>;
            {
                // Not checking for out-of-bounds exception because no other thread
                // should be able to remove entries from the list.
                process = self.process_list.lock().unwrap().get(i).cloned();
            }
            // This should not be synchronized, since it is calling a function that is
            // synchronized on another mutex.
            if let Some(process) = process
                && !self.stop.load(Ordering::SeqCst)
            {
                process.restart_all();
            }
        }
        if !self.stop.load(Ordering::SeqCst) {
            // After the processes are restarted, remove them. Do this separately to
            // prevent the processes that fail fast from being re-added to processList
            // while the thread is still doing restarts.
            let mut i = size as i64 - 1;
            while i >= 0 {
                if self.stop.load(Ordering::SeqCst) {
                    break;
                }
                {
                    // This must be the only place that a process can be removed from
                    // the processList.
                    self.process_list.lock().unwrap().remove(i as usize);
                }
                i -= 1;
            }
        }
        {
            *self.running.lock().unwrap() = false;
        }
    }

    /// Java public static `stop()`.  Stops run().
    pub fn stop() {
        if INSTANCE.stop.load(Ordering::SeqCst) == true {
            return;
        }
        INSTANCE.stop.store(true, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
