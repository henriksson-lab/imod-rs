//! `IMOD/Etomo/src/etomo/process/ImodRequestHandler.java`.
//!
//! Watches 3dmod stderr for requests.  Processes requests.
//!
//! The source only creates a handler on Windows (`getInstance` returns null
//! elsewhere), so on this platform `get_instance` always returns `None`; the class is
//! translated whole regardless.

use super::base_imod_manager::BaseImodManager;
use crate::imod::etomo::util::utilities;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Java static `instances`: one instance per ImodManager, keyed by the manager's
/// identity.
static INSTANCES: Mutex<Vec<(usize, Arc<ImodRequestHandler>)>> = Mutex::new(Vec::new());

/// Java package-private `ImodRequestHandler implements Runnable`.
pub struct ImodRequestHandler {
    imod_manager: &'static BaseImodManager,
    stop: AtomicBool,
    running: AtomicBool,
}

impl ImodRequestHandler {
    /// Java private `ImodRequestHandler(BaseImodManager)`.
    fn new(imod_manager: &'static BaseImodManager) -> ImodRequestHandler {
        ImodRequestHandler {
            imod_manager,
            stop: AtomicBool::new(false),
            running: AtomicBool::new(true),
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(imod_manager: &'static BaseImodManager) -> Option<Arc<ImodRequestHandler>> {
        if !utilities::is_windows_os() {
            return None;
        }
        let key = imod_manager as *const BaseImodManager as usize;
        let mut instances = INSTANCES.lock().unwrap();
        if let Some((_, handler)) = instances.iter().find(|(manager, _)| *manager == key) {
            return Some(Arc::clone(handler));
        }
        let handler = Arc::new(ImodRequestHandler::new(imod_manager));
        instances.push((key, Arc::clone(&handler)));
        let thread_handler = Arc::clone(&handler);
        std::thread::spawn(move || thread_handler.run());
        Some(handler)
    }

    /// Java `run`.
    pub fn run(&self) {
        // System.out.println("run");
        while !self.stop.load(Ordering::SeqCst) {
            // System.out.println("!stop");
            std::thread::sleep(Duration::from_millis(100));
            if let Err(e) = self.imod_manager.process_request() {
                eprintln!("{e}");
            }
        }
        self.running.store(false, Ordering::SeqCst);
        // System.out.println("end run");
    }

    /// Java `stop`.  stop the worker thread; this request times out after 1 seconds.
    pub fn stop(&self) {
        // System.out.println("stop");
        self.stop.store(true, Ordering::SeqCst);
        // give the worker thread 1 second to clean up
        for _i in 0..20 {
            // System.out.println("i="+i);
            std::thread::sleep(Duration::from_millis(50));
            if !self.running.load(Ordering::SeqCst) {
                break;
            }
        }
        // System.out.println("end stop");
    }
}
