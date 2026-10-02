//! `IMOD/Etomo/src/etomo/ProcessingMethodMediator.java`.
//!
//! Mediates the processing method (local, parallel CPU/GPU, queue) between the
//! registered process interface (a dialog or panel), the axis process panel,
//! the parallel panel, a reconnecting process and a running parallel process
//! monitor.
//!
//! The mediator is an EDT object (`Rc`, `&self` methods), one per axis, held by
//! the manager.  Java `synchronized` has no counterpart: every method runs on
//! the event dispatch thread.  A process or monitor thread that registers or
//! deregisters itself (`ReconnectProcess`, `ParallelProcessMonitor`) posts that
//! call to the EDT.
//!
//! Java overloads follow the suffix rule: `register(ReconnectProcess)` is
//! `register_reconnect_process`, `setMethod(ProcessInterface, ProcessingMethod)`
//! is `set_method_process_interface_processing_method`, and so on.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crate::imod::etomo::jdk::{ActionEvent, ActionListener};
use crate::imod::etomo::process::parallel_process_monitor::ParallelProcessMonitor;
use crate::imod::etomo::process::reconnect_process::ReconnectProcess;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::ui::swing::axis_process_panel::AxisProcessPanel;
use crate::imod::etomo::ui::swing::button_component::ButtonComponent;
use crate::imod::etomo::ui::swing::parallel_panel::ParallelPanel;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;
use crate::imod::etomo::ui::swing::process_interface::ProcessInterface;

/// Java `public final class ProcessingMethodMediator`.
pub struct ProcessingMethodMediator {
    /// Java `reconnectProcess`: Only one.  May or may not register before a
    /// dialog is created.  Has priority over everything else.  Must deregister
    /// when its process ends.
    reconnect_process: RefCell<Option<Arc<ReconnectProcess>>>,
    /// Java `processInterface`: Dialogs, tab panels, panels, or ancestors of
    /// panels.  One may register at a time.  Must deregister when not displayed,
    /// unless it is immediately replaces by another interface.
    process_interface: RefCell<Option<Rc<dyn ProcessInterface>>>,
    /// Java `parallelProcessMonitor`: Parallel process monitor instances.
    /// Registered while running.  Inhibits axisProcessPanel.  Overriden by
    /// reconnectProcess.
    parallel_process_monitor: RefCell<Option<Arc<dyn ParallelProcessMonitor>>>,
    /// Java `axisProcessPanel`: Only one.  Manages the parallel panel.  First
    /// thing to register.  Does not have to deregister because it is not
    /// destroyed until the program exits.  Takes direction from reconnectProcess
    /// and processInterface.
    axis_process_panel: RefCell<Option<Rc<AxisProcessPanel>>>,
    /// Java `parallelPanel`: Parallel table panel.  Only one.  Does not have to
    /// deregister because it is not destroyed until the program exits  Takes
    /// direction from reconnectProcess and processInterface.  Tells
    /// processInterface if the QUEUE ProcessingMethod has been selected.
    parallel_panel: RefCell<Option<Rc<ParallelPanel>>>,
    /// Java `private ButtonComponent[] gpuComponent = new ButtonComponent[] {}`.
    gpu_component: RefCell<Option<Vec<Rc<dyn ButtonComponent>>>>,
}

impl Default for ProcessingMethodMediator {
    fn default() -> ProcessingMethodMediator {
        ProcessingMethodMediator {
            reconnect_process: RefCell::new(None),
            process_interface: RefCell::new(None),
            parallel_process_monitor: RefCell::new(None),
            axis_process_panel: RefCell::new(None),
            parallel_panel: RefCell::new(None),
            gpu_component: RefCell::new(Some(Vec::new())),
        }
    }
}

/// Java reference identity (`==`) on two interface handles.
fn same_process_interface(
    a: Option<&Rc<dyn ProcessInterface>>,
    b: Option<&Rc<dyn ProcessInterface>>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::ptr::addr_eq(Rc::as_ptr(a), Rc::as_ptr(b)),
        _ => false,
    }
}

impl ProcessingMethodMediator {
    /// Java implicit constructor.
    pub fn new() -> Rc<ProcessingMethodMediator> {
        Rc::new(ProcessingMethodMediator::default())
    }

    fn reconnect_process(&self) -> Option<Arc<ReconnectProcess>> {
        self.reconnect_process.borrow().clone()
    }

    fn process_interface(&self) -> Option<Rc<dyn ProcessInterface>> {
        self.process_interface.borrow().clone()
    }

    fn axis_process_panel(&self) -> Option<Rc<AxisProcessPanel>> {
        self.axis_process_panel.borrow().clone()
    }

    fn parallel_panel(&self) -> Option<Rc<ParallelPanel>> {
        self.parallel_panel.borrow().clone()
    }

    /// Java synchronized `register(ReconnectProcess)`.  ReconnectProcess prevents
    /// processInterface and parallelPanel from changing processing method.
    pub fn register_reconnect_process(&self, origin: Option<&Arc<ReconnectProcess>>) {
        let Some(origin) = origin else {
            // can't use this function to deregister a process
            return;
        };
        if self.reconnect_process.borrow().is_some() {
            // must deregister old process to register new one
            return;
        }
        // Show the reconnect process method
        let method = origin.get_processing_method();
        if let Some(axis_process_panel) = self.axis_process_panel() {
            if method.is_some_and(|method| !method.is_local()) {
                // Does the monitor start before the process? Maybe, so force the
                // dumpState("REG", origin, method, null, true, false);
                axis_process_panel.force_show_parallel_panel(true);
                if let Some(parallel_panel) = self.parallel_panel() {
                    parallel_panel.set_visible(true);
                }
            } else {
                // dumpState("REG", origin, method, null, null, false);
                axis_process_panel.show_parallel_panel_boolean(false);
            }
        }
        // lock everything while the reconnect process is running.
        if let Some(parallel_panel) = self.parallel_panel() {
            // The queues checkbox is not being loaded.
            if method == Some(ProcessingMethod::Queue) {
                parallel_panel.set_queue(true, Some(&**origin), method);
            }
            parallel_panel.set_processing_method(method);
            parallel_panel.lock_processing_method(true);
        }
        if let Some(process_interface) = self.process_interface() {
            process_interface.lock_processing_method(true);
        }
        *self.reconnect_process.borrow_mut() = Some(origin.clone());
    }

    /// Java synchronized `deregister(ReconnectProcess)`.  Releases control of
    /// reconnectProcess and reinstates the dialog method.
    pub fn deregister_reconnect_process(&self, origin: Option<&Arc<ReconnectProcess>>) {
        let same = match (origin, self.reconnect_process.borrow().as_ref()) {
            (None, None) => true,
            (Some(origin), Some(reconnect_process)) => Arc::ptr_eq(origin, reconnect_process),
            _ => false,
        };
        if !same {
            return;
        }
        *self.reconnect_process.borrow_mut() = None;
        // Going back to the processInterface method
        let mut method: Option<ProcessingMethod> = None;
        if let Some(process_interface) = self.process_interface() {
            method = Some(process_interface.get_processing_method());
        }
        let method = method.unwrap_or(ProcessingMethod::DEFAULT);
        // Unlock after the reconnect process is done and have the processInterface
        // take over.
        if let Some(axis_process_panel) = self.axis_process_panel() {
            // dumpState("DEREG", origin, method, null, null, false);
            axis_process_panel.show_parallel_panel_boolean(!method.is_local());
        }
        let mut queue_method = false;
        if let Some(parallel_panel) = self.parallel_panel() {
            parallel_panel.lock_processing_method(false);
            parallel_panel.set_processing_method(Some(method));
            queue_method = parallel_panel.get_processing_method() == Some(ProcessingMethod::Queue);
        }
        if let Some(process_interface) = self.process_interface() {
            process_interface.lock_processing_method(false);
            process_interface.update_gpu(queue_method);
        }
    }

    /// Java synchronized `register(ProcessInterface)`.  Sets the interface
    /// processing method.  When another processInterface is register, the new
    /// one replaces it.
    pub fn register_process_interface(&self, origin: Rc<dyn ProcessInterface>) {
        // `if (origin == null) return;`: the Rust handle is never null.
        *self.process_interface.borrow_mut() = Some(origin.clone());
        if self.reconnect_process.borrow().is_some() {
            // nothing to do - reconnectProcess locks everything
            return;
        }
        let process_interface = origin.clone();
        if let Some(parallel_panel) = self.parallel_panel() {
            process_interface
                .add_queue_table_listener(parallel_panel.clone() as Rc<dyn QueueTableListener>);
            parallel_panel.add_queue_table_listener(Some(
                process_interface.clone() as Rc<dyn QueueTableListener>
            ));
        }
        if let Some(parallel_panel) = self.parallel_panel() {
            origin.set_use_queue_check_box(Some(parallel_panel.get_use_queue_checkbox()));
        }
        // dumpState("REG", origin, processInterface.getProcessingMethod(), null, null,
        // false);
        // Set interface method - needs to be called twice because of the
        // interdependency of process interface and the parallel panel.
        self.set_interface_method(Some(process_interface.get_processing_method()));
        self.set_interface_method(Some(process_interface.get_processing_method()));
        self.set_secondary_interface_method(process_interface.get_secondary_processing_method());
    }

    /// Java synchronized `deregister(ProcessInterface)`.  Sets the default
    /// interface processing method.  To prevent the parallel panel from blinking
    /// on and off, only call this function when another interface isn't
    /// immediately available to replace it (helps with tabbing).
    pub fn deregister_process_interface(&self, origin: &Rc<dyn ProcessInterface>) {
        if !same_process_interface(Some(origin), self.process_interface.borrow().as_ref()) {
            return;
        }
        // dumpState("DEREG", origin, ProcessingMethod.DEFAULT, null, null, false);
        if let (Some(process_interface), Some(parallel_panel)) =
            (self.process_interface(), self.parallel_panel())
        {
            let parallel_panel_listener: Rc<dyn QueueTableListener> = parallel_panel.clone();
            process_interface.remove_queue_table_listener(&parallel_panel_listener);
            let process_interface_listener: Rc<dyn QueueTableListener> = process_interface;
            parallel_panel.remove_queue_table_listener(Some(&process_interface_listener));
        }
        *self.process_interface.borrow_mut() = None;
        // hide parallel panel
        if let Some(axis_process_panel) = self.axis_process_panel() {
            axis_process_panel.show_parallel_panel_boolean(false);
        }
        if let Some(parallel_panel) = self.parallel_panel() {
            parallel_panel.set_processing_method(Some(ProcessingMethod::DEFAULT));
            parallel_panel.set_secondary_processing_method(None);
        }

        *self.gpu_component.borrow_mut() = None;
    }

    /// Java synchronized `register(ParallelProcessMonitor)`.  Prevents
    /// axisProcessPanel from hiding the parallel panel while a parallel process
    /// is running.  When a parallel process is done, reinstates the normal
    /// display.
    pub fn register_parallel_process_monitor(
        &self,
        origin: Option<&Arc<dyn ParallelProcessMonitor>>,
    ) {
        let Some(origin) = origin else {
            // can't use this function to deregister an interface
            return;
        };
        if self.parallel_process_monitor.borrow().is_some() {
            // must deregister old interface to register new one
            return;
        }
        *self.parallel_process_monitor.borrow_mut() = Some(origin.clone());
        if let Some(axis_process_panel) = self.axis_process_panel() {
            axis_process_panel.lock_processing_method(true);
        }
    }

    /// Java synchronized `deregister(ParallelProcessMonitor)`.
    pub fn deregister_parallel_process_monitor(
        &self,
        origin: Option<&Arc<dyn ParallelProcessMonitor>>,
    ) {
        let same = match (origin, self.parallel_process_monitor.borrow().as_ref()) {
            (None, None) => true,
            (Some(origin), Some(parallel_process_monitor)) => {
                std::ptr::addr_eq(Arc::as_ptr(origin), Arc::as_ptr(parallel_process_monitor))
            }
            _ => false,
        };
        if !same {
            return;
        }
        *self.parallel_process_monitor.borrow_mut() = None;
        if let Some(axis_process_panel) = self.axis_process_panel() {
            axis_process_panel.lock_processing_method(false);
        }
        // do nothing if reconnectProcess is running - its deregistration will take
        // care of any changes.
        if self.reconnect_process.borrow().is_none() {
            let mut method = ProcessingMethod::DEFAULT;
            if let Some(process_interface) = self.process_interface() {
                method = process_interface.get_processing_method();
            }
            // dumpState("DEREG", origin, method, null, null, false);
            self.set_interface_method(Some(method));
        }
    }

    /// Java synchronized `register(AxisProcessPanel)`.
    pub fn register_axis_process_panel(&self, origin: &Rc<AxisProcessPanel>) {
        // `if (origin == null) return;`: the Rust handle is never null.
        if self.axis_process_panel.borrow().is_some() {
            // must deregister old panel to register new one
            return;
        }
        *self.axis_process_panel.borrow_mut() = Some(origin.clone());
        // The axisProcessPanel should be created very early, so there wouldn't be
        // anything available for it to get information from.
    }

    /// Java synchronized `register(ParallelPanel)`.
    pub fn register_parallel_panel(&self, origin: Rc<ParallelPanel>) {
        // `if (origin == null) return;`: the Rust handle is never null.
        if self.parallel_panel.borrow().is_some() {
            // must deregister old panel to register new one
            // ? ParallelPanel cannot be deregistered.
            return;
        }
        *self.parallel_panel.borrow_mut() = Some(origin.clone());
        let parallel_panel = origin;
        if let Some(process_interface) = self.process_interface() {
            process_interface
                .add_queue_table_listener(parallel_panel.clone() as Rc<dyn QueueTableListener>);
            parallel_panel.add_queue_table_listener(Some(
                process_interface.clone() as Rc<dyn QueueTableListener>
            ));
        }
        // Parallel panel is create in response to the existance of a reconnect
        // process or an interface. Setting its method should be taken care of
        // by the process or interface.
        let gpu_component = self.gpu_component.borrow().clone();
        if let Some(gpu_component) = gpu_component {
            self.add_gpu_listener(gpu_component);
        }
        if let Some(process_interface) = self.process_interface() {
            process_interface
                .set_use_queue_check_box(Some(parallel_panel.get_use_queue_checkbox()));
        }
    }

    /// Java private `setInterfaceMethod(ProcessingMethod)`.  The the method from
    /// processInterface in axisProcessPanel and parallelPanel.  Also disable GPU
    /// in processInterface if the queue check box is checked.
    fn set_interface_method(&self, method: Option<ProcessingMethod>) {
        let method = method.unwrap_or(ProcessingMethod::DEFAULT);
        if let Some(axis_process_panel) = self.axis_process_panel() {
            axis_process_panel.show_parallel_panel_boolean(!method.is_local());
        }
        let mut queue_method = false;
        if let Some(parallel_panel) = self.parallel_panel() {
            parallel_panel.set_processing_method(Some(method));
            queue_method = parallel_panel.get_processing_method() == Some(ProcessingMethod::Queue);
        }
        if let Some(process_interface) = self.process_interface() {
            process_interface.update_gpu(queue_method);
        }
    }

    /// Java private `setSecondaryInterfaceMethod(ProcessingMethod)`.
    fn set_secondary_interface_method(&self, method: Option<ProcessingMethod>) {
        if let Some(parallel_panel) = self.parallel_panel() {
            parallel_panel.set_secondary_processing_method(method);
        }
    }

    /// Java synchronized `setMethod(ProcessInterface, ProcessingMethod)`.  Tell
    /// the parallel panel about the change in ProcessInterface method.
    pub fn set_method_process_interface_processing_method(
        &self,
        origin: &Rc<dyn ProcessInterface>,
        method: ProcessingMethod,
    ) {
        // Ignore an unregistered process interface
        // Don't change processing method while reconnect process exists
        if !same_process_interface(Some(origin), self.process_interface.borrow().as_ref())
            || self.reconnect_process.borrow().is_some()
        {
            return;
        }
        // dumpState("SET", origin, method, null, null, false);
        self.set_interface_method(Some(method));
    }

    /// Java synchronized `setMethod(ProcessInterface, ProcessingMethod,
    /// ProcessingMethod, boolean)`.
    pub fn set_method_process_interface_processing_method_processing_method_boolean(
        &self,
        origin: &Rc<dyn ProcessInterface>,
        method: Option<ProcessingMethod>,
        secondary_method: Option<ProcessingMethod>,
        visible: bool,
    ) {
        if !same_process_interface(Some(origin), self.process_interface.borrow().as_ref()) {
            return;
        }
        // Don't change processing method while reconnect process exists. Except
        // change the secondary method because the reconnect process does not use
        // that.
        // dumpState("SET", origin, method, visible, null, false);
        if self.reconnect_process.borrow().is_none() {
            self.set_interface_method(method);
        }
        self.set_secondary_interface_method(secondary_method);
        if self.reconnect_process.borrow().is_none() {
            // Upstream bug fixed in translation (ProcessingMethodMediator.java:
            // 363): Java calls `parallelPanel.setVisible(visible)` without the null
            // check every other use makes, and throws NullPointerException before
            // the parallel panel is registered.  Here nothing is done then.
            if let Some(parallel_panel) = self.parallel_panel() {
                parallel_panel.set_visible(visible);
            }
        }
    }

    /// Java synchronized `setMethod(ParallelPanel, ProcessingMethod)`.  Tell the
    /// processInterface when to disable the GPU check box.
    pub fn set_method_parallel_panel_processing_method(
        &self,
        origin: &Rc<ParallelPanel>,
        method: Option<ProcessingMethod>,
    ) {
        // Ignore an unregistered parallel panel
        let registered = self
            .parallel_panel
            .borrow()
            .as_ref()
            .is_some_and(|parallel_panel| Rc::ptr_eq(origin, parallel_panel));
        if !registered {
            return;
        }
        if let Some(process_interface) = self.process_interface() {
            // check for gpu selector and add gpu action listener
            let gpu_component = self.gpu_component.borrow().clone();
            if let Some(gpu_component) = gpu_component {
                self.add_gpu_listener(gpu_component);
            }
            process_interface.update_gpu(method == Some(ProcessingMethod::Queue));
            if method == Some(ProcessingMethod::Queue)
                && let Some(parallel_panel) = self.parallel_panel()
            {
                process_interface
                    .set_use_queue_check_box(Some(parallel_panel.get_use_queue_checkbox()));
            }
        }
    }

    /// Java private `dumpState(String, Object, ProcessingMethod, Boolean, Boolean,
    /// boolean)`.  Only referenced from commented-out calls.  `origin` is the
    /// origin's `toString()`.
    fn dump_state(
        &self,
        id: &str,
        origin: &str,
        method: Option<ProcessingMethod>,
        visible: Option<bool>,
        force: Option<bool>,
        dump_stack: bool,
    ) {
        println!(
            "\n{} {},\nmethod:{}{}{},\nreconnectProcess:{}",
            id,
            origin,
            method.map_or_else(|| "null".to_owned(), |method| method.to_string()),
            visible.map_or_else(String::new, |visible| format!(",visible:{}", visible)),
            force.map_or_else(String::new, |force| format!(",force:{}", force)),
            if self.reconnect_process.borrow().is_some() {
                "ReconnectProcess"
            } else {
                "null"
            }
        );
        if dump_stack {
            // Thread.dumpStack()
            eprintln!("{}", std::backtrace::Backtrace::force_capture());
        }
    }

    /// Java `getRunMethodForParallelPanel(ProcessingMethod)`.  Get the
    /// processing method for resume and for turning off queue check box.
    /// `parallel_panel_method` - may be null.
    pub fn get_run_method_for_parallel_panel(
        &self,
        parallel_panel_method: Option<ProcessingMethod>,
    ) -> Option<ProcessingMethod> {
        if let Some(process_interface) = self.process_interface() {
            let process_interface_method = process_interface.get_processing_method();
            if !process_interface_method.is_local()
                && parallel_panel_method == Some(ProcessingMethod::Queue)
            {
                return parallel_panel_method;
            }
            return Some(process_interface_method);
        }
        // OK to return the correct method during a reconnect because the resume
        // button has no effect and the parallel panel method is locked.
        parallel_panel_method
    }

    /// Java `getSecondaryRunMethodForParallelPanel(ProcessingMethod)`.
    pub fn get_secondary_run_method_for_parallel_panel(
        &self,
        parallel_panel_method: Option<ProcessingMethod>,
    ) -> Option<ProcessingMethod> {
        if let Some(process_interface) = self.process_interface() {
            let process_interface_method = process_interface.get_secondary_processing_method();
            if process_interface_method.is_some_and(|method| !method.is_local())
                && parallel_panel_method == Some(ProcessingMethod::Queue)
            {
                return parallel_panel_method;
            }
            return process_interface_method;
        }
        // OK to return the correct method during a reconnect because the resume
        // button has no effect and the parallel panel method is locked.
        parallel_panel_method
    }

    /// Java `getRunMethodForProcessInterface(ProcessingMethod)`.  Get the
    /// processing method for running a process from the interface.
    pub fn get_run_method_for_process_interface(
        &self,
        process_interface_method: ProcessingMethod,
    ) -> ProcessingMethod {
        if let Some(parallel_panel) = self.parallel_panel() {
            if !parallel_panel.is_runnable() {
                // If the parallel panel is not runnable, then this mediator is
                // primarily being used to set processing method parameters, not
                // actually run a process. When the run method is asked for, send the
                // default.
                return ProcessingMethod::DEFAULT;
            }
            let parallel_panel_method = parallel_panel.get_processing_method();
            if !process_interface_method.is_local()
                && parallel_panel_method == Some(ProcessingMethod::Queue)
            {
                return ProcessingMethod::Queue;
            }
        }
        process_interface_method
    }

    /// Java `msgExiting()`.  Tells parallelPanel to stop its thread for exit.
    pub fn msg_exiting(&self) {
        if let Some(parallel_panel) = self.parallel_panel() {
            parallel_panel.end_table();
        }
    }

    /// Java `getParallelProgressDisplay()`.  Provides a simple way to request the
    /// ParallelProgressDisplay from parallelPanel if it has already been
    /// displayed.  Displaying parallePanel should already have been taken care
    /// of by the register classes.  Does not change anything.  Returns null if
    /// not available.
    pub fn get_parallel_progress_display(&self) -> Option<Rc<dyn ParallelProgressDisplay>> {
        let parallel_panel = self.parallel_panel()?;
        Some(parallel_panel.get_parallel_progress_display())
    }

    /// Java `isUseGpu()`.
    pub fn is_use_gpu(&self) -> bool {
        if let Some(process_interface) = self.process_interface() {
            return process_interface.is_use_gpu();
        }
        false
    }

    /// Java `addGpuListener(ButtonComponent[])`.  This function is used to
    /// inform the parallel panel when the 'Use the GPU' checkbox changes its
    /// state.
    pub fn add_gpu_listener(&self, gpu_component: Vec<Rc<dyn ButtonComponent>>) {
        // `if (gpuComponent == null) return;`: the Rust array is never null.
        if let Some(parallel_panel) = self.parallel_panel() {
            let queue_table = parallel_panel.get_table(Some(ProcessingMethod::Queue));
            if let Some(queue_table) = queue_table {
                parallel_panel.add_queue_listener(&gpu_component);
                for component in &gpu_component {
                    // gpuComponent[i].addActionListener(queueTable): the table's
                    // ActionListener.actionPerformed.
                    let queue_table = queue_table.clone();
                    let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                        ParallelProgressDisplay::action_performed(&*queue_table, event);
                    });
                    component.add_action_listener(listener);
                }
            }
        } else {
            *self.gpu_component.borrow_mut() = Some(gpu_component);
        }
    }

    /// Java `addQueueListenerOnSwitchDialog()`.  There is a call to
    /// addQueueListener() in setMethod() and setInterfaceMethod() functions of
    /// this class. Despite that, this method is added to catch the first
    /// queueCheckbox event after dialog/tab switching takes place. This is when
    /// processing method is not yet set to QUEUE.
    pub fn add_queue_listener_on_switch_dialog(&self) {
        if let (Some(process_interface), Some(parallel_panel)) =
            (self.process_interface(), self.parallel_panel())
        {
            process_interface
                .set_use_queue_check_box(Some(parallel_panel.get_use_queue_checkbox()));
        }
    }
}
