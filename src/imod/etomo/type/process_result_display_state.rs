//! `IMOD/Etomo/src/etomo/type/ProcessResultDisplayState.java`.
//!
//! The state and message handling shared by every `ProcessResultDisplay`
//! (in the Java tree only `MultiLineButton` constructs one).  It is an EDT
//! object: it lives inside its display, every method takes `&self`, and the
//! mutable fields are `Cell`/`RefCell`, never borrowed across a call into a
//! display (a display's `setProcessDone` fires Swing listeners that may call
//! back into this state).
//!
//! `display` is the Java `this` the owning display passes to the constructor.
//! The display owns this state, so the back pointer is a `Weak`; it is always
//! alive while a method of this state runs, because only the display calls
//! them.

use std::cell::{Cell, RefCell};
use std::rc::Weak;

use super::process_end_state::ProcessEndState;
use super::process_result::ProcessResult;
use super::process_result_display::{ProcessResultDisplay, ProcessResultDisplayHandle};

/// Java `ProcessResultDisplayState`.
pub struct ProcessResultDisplayState {
    /// Java final `display`.
    display: Weak<dyn ProcessResultDisplay>,

    /// Java `dependentDisplayList` (a raw `Vector`, null until the first add).
    dependent_display_list: RefCell<Option<Vec<ProcessResultDisplayHandle>>>,
    /// Java `failureDisplayList`.
    failure_display_list: RefCell<Option<Vec<ProcessResultDisplayHandle>>>,
    /// Java `successDisplayList`.
    success_display_list: RefCell<Option<Vec<ProcessResultDisplayHandle>>>,
    /// Java `displayID`.
    display_id: Cell<i32>,
    /// Java `factoryID`.
    factory_id: RefCell<Option<String>>,
    /// Java `debug`.
    debug: Cell<bool>,

    /// Java `originalState`: will go back to the original state if the process
    /// failed to run.
    original_state: Cell<bool>,
    /// Java `secondaryProcess`: tells which process is current being run.
    secondary_process: Cell<bool>,
    /// Java `processRunning`: will ignore most messages when the process is
    /// not running.
    process_running: Cell<bool>,
    /// Java `useGlobalDependencyList`.
    use_global_dependency_list: Cell<bool>,
    /// Java `next`.
    next: RefCell<Option<ProcessResultDisplayHandle>>,
}

impl ProcessResultDisplayState {
    /// Java `ProcessResultDisplayState(ProcessResultDisplay)`.
    pub fn new(display: Weak<dyn ProcessResultDisplay>) -> ProcessResultDisplayState {
        ProcessResultDisplayState {
            display,
            dependent_display_list: RefCell::new(None),
            failure_display_list: RefCell::new(None),
            success_display_list: RefCell::new(None),
            display_id: Cell::new(-1),
            factory_id: RefCell::new(None),
            debug: Cell::new(false),
            original_state: Cell::new(false),
            secondary_process: Cell::new(false),
            process_running: Cell::new(false),
            use_global_dependency_list: Cell::new(true),
            next: RefCell::new(None),
        }
    }

    /// The Java field `display`.  The display owns this state and is the only
    /// caller of its methods, so it is alive whenever this is reached.
    fn display(&self) -> ProcessResultDisplayHandle {
        self.display
            .upgrade()
            .expect("ProcessResultDisplayState used after its display was dropped")
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `setOriginalState(boolean)`.
    pub fn set_original_state(&self, original_state: bool) {
        self.original_state.set(original_state);
    }

    /// Java `synchronized msgProcessStarting()`.  The lock has no Rust
    /// counterpart: this state is only touched on the EDT.
    pub fn msg_process_starting(&self) {
        if !self.process_running.get() {
            let display = self.display();
            self.original_state.set(display.get_original_state());
            // on average the process will complete, so set process done to true at the
            // beginning in case the user exits etomo.
            display.set_process_done(true);
            self.secondary_process.set(false);
        }
        self.process_running.set(true);
    }

    /// Java `msg(ProcessResult)`.
    pub fn msg_process_result(&self, process_result: Option<ProcessResult>) {
        if process_result == Some(ProcessResult::SUCCEEDED) {
            self.msg_process_succeeded();
        } else if process_result == Some(ProcessResult::FAILED) {
            self.msg_process_failed();
        } else if process_result == Some(ProcessResult::FAILED_TO_START) {
            self.msg_process_failed_to_start();
        }
    }

    /// Java `msg(ProcessEndState)`.
    pub fn msg_process_end_state(&self, end_state: Option<ProcessEndState>) {
        if end_state == Some(ProcessEndState::Done)
            || end_state == Some(ProcessEndState::Killed)
            || end_state == Some(ProcessEndState::Paused)
        {
            self.msg_process_succeeded();
        }
        if end_state == Some(ProcessEndState::Cancelled) {
            self.msg_process_failed_to_start();
        }
        if end_state == Some(ProcessEndState::Failed) {
            self.msg_process_failed();
        }
    }

    /// Java `setUseGlobalDependencyList(boolean)`.
    pub fn set_use_global_dependency_list(&self, use_: bool) {
        self.use_global_dependency_list.set(use_);
    }

    /// Java `msgProcessSucceeded()`.
    pub fn msg_process_succeeded(&self) {
        if !self.process_running.get() {
            return;
        }
        let display = self.display();
        display.set_process_done(true);
        self.set_process_done_boolean_process_result_display(false, &*display);
        let dependent = self.dependent_display_list.borrow().clone();
        self.set_process_done_boolean_vector(false, dependent.as_deref());
        let success = self.success_display_list.borrow().clone();
        self.set_process_done_boolean_vector(true, success.as_deref());
        self.process_running.set(false);
    }

    /// Java `msgProcessFailed()`.
    pub fn msg_process_failed(&self) {
        if !self.process_running.get() {
            return;
        }
        let display = self.display();
        display.set_process_done(false);
        self.set_process_done_boolean_process_result_display(false, &*display);
        let dependent = self.dependent_display_list.borrow().clone();
        self.set_process_done_boolean_vector(false, dependent.as_deref());
        let failure = self.failure_display_list.borrow().clone();
        self.set_process_done_boolean_vector(false, failure.as_deref());
        self.process_running.set(false);
    }

    /// Java private `setProcessDone(boolean, ProcessResultDisplay)`.
    fn set_process_done_boolean_process_result_display(
        &self,
        done: bool,
        display: &dyn ProcessResultDisplay,
    ) {
        if self.use_global_dependency_list.get() {
            // Go through the general depend display list
            let mut current = display.get_next();
            while let Some(display) = current {
                display.set_process_done(done);
                display.set_original_state(done);
                current = display.get_next();
            }
        }
    }

    /// Java private `setProcessDone(boolean, Vector)`.  The caller passes a
    /// snapshot of the list, so no borrow is held while displays react.
    fn set_process_done_boolean_vector(
        &self,
        done: bool,
        display_list: Option<&[ProcessResultDisplayHandle]>,
    ) {
        let Some(display_list) = display_list else {
            return;
        };
        for display in display_list {
            display.set_process_done(done);
            display.set_original_state(done);
        }
    }

    /// Java `msgProcessFailedToStart()`.
    pub fn msg_process_failed_to_start(&self) {
        if !self.process_running.get() {
            return;
        }
        if self.secondary_process.get() {
            self.msg_process_failed();
        } else {
            self.display().set_process_done(self.original_state.get());
            self.process_running.set(false);
        }
    }

    /// Java `setID(int, String)`.
    pub fn set_id(&self, display_id: i32, factory_id: Option<String>) {
        self.display_id.set(display_id);
        *self.factory_id.borrow_mut() = factory_id;
    }

    /// Java `equalsID(int, String)`.
    pub fn equals_id(&self, display_id: i32, factory_id: Option<&str>) -> bool {
        // Display ID is required for equalsID to return true.
        if self.display_id.get() < 0 || self.display_id.get() != display_id {
            return false;
        }
        // Include factoryID if it is set.
        match &*self.factory_id.borrow() {
            None => true,
            // Java `this.factoryID.equals(factoryID)`: false for a null argument.
            Some(own) => factory_id == Some(own.as_str()),
        }
    }

    /// Java `getDisplayID()`.
    pub fn get_display_id(&self) -> i32 {
        self.display_id.get()
    }

    /// Java `setFactoryID(String)`.
    pub fn set_factory_id(&self, input: Option<String>) {
        *self.factory_id.borrow_mut() = input;
    }

    /// Java `getFactoryID()`.
    pub fn get_factory_id(&self) -> Option<String> {
        self.factory_id.borrow().clone()
    }

    /// Java `getNext()`.
    pub fn get_next(&self) -> Option<ProcessResultDisplayHandle> {
        self.next.borrow().clone()
    }

    /// Java `setNext(ProcessResultDisplay)`.
    pub fn set_next(&self, display: Option<ProcessResultDisplayHandle>) {
        *self.next.borrow_mut() = display;
    }

    /// Java `addDependentDisplay(ProcessResultDisplay)`.
    pub fn add_dependent_display(&self, dependent_display: Option<ProcessResultDisplayHandle>) {
        let Some(dependent_display) = dependent_display else {
            return;
        };
        self.dependent_display_list
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(dependent_display);
    }

    /// Java `addFailureDisplay(ProcessResultDisplay)`.
    pub fn add_failure_display(&self, failure_display: Option<ProcessResultDisplayHandle>) {
        let Some(failure_display) = failure_display else {
            return;
        };
        self.failure_display_list
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(failure_display);
    }

    /// Java `addSuccessDisplay(ProcessResultDisplay)`.
    pub fn add_success_display(&self, success_display: Option<ProcessResultDisplayHandle>) {
        let Some(success_display) = success_display else {
            return;
        };
        self.success_display_list
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(success_display);
    }

    /// Java `msgSecondaryProcess()`.
    pub fn msg_secondary_process(&self) {
        self.secondary_process.set(true);
    }

    /// Java package-private `isOriginalState()`.
    pub fn is_original_state(&self) -> bool {
        self.original_state.get()
    }

    /// Java package-private `isProcessRunning()`.
    pub fn is_process_running(&self) -> bool {
        self.process_running.get()
    }

    /// Java package-private `isSecondaryProcess()`.
    pub fn is_secondary_process(&self) -> bool {
        self.secondary_process.get()
    }
}
