//! `IMOD/Etomo/src/etomo/process/IntermittentBackgroundProcess.java`.
//!
//! Runs a command which sends another command through standard in at intervals
//! (the load-average `w` over a long-lived `bash`/`ssh` session).  The original command
//! is halted by a stop function.
//!
//! This class saves the commands it receives in a static storage.  A new instance of
//! this class is created only when a command that is not already stored is sent to it.
//! The monitors (`IntermittentProcessMonitor`) that process standard out are added to
//! an instance-level list; a monitor added while the command is running is "hooked
//! into" the existing command via the standard out.  So only one command per instance
//! can be run at a time.  Any previous commands will stop as soon as they see that they
//! are not the most recent command.
//!
//! - `stop(monitor)` drops the monitor from its output and tells the monitor to stop
//!   monitoring it; the process stops only if it is no longer being monitored.
//! - `end(monitor)` calls `stop(monitor)` and then removes the monitor from its list.
//!   It is used when a manager exits.
//! - `fail()` drops all of its monitors from its output and stops.  The monitors keep
//!   running, but ignore the stopped process.
//! - `restartAll()` is called by `ProcessRestarter` to restart the process.
//!
//! **Shape.**  An instance is shared by its own `run` thread, the monitors, the
//! restarter and the manager, so it lives in an `Arc` (the static `instances` table
//! keeps every one alive for the life of the process, as Java's static `Hashtable`
//! does), the mutable fields are atomics or sit behind locks, and it keeps a `Weak` to
//! itself for `new Thread(this)` and `monitor.setProcess(this)`.
//!
//! Java's `synchronized` instance methods share one reentrant monitor, and `restartAll`
//! and `end` call the synchronized `start`/`stop` while holding it.  Rust's `Mutex` is
//! not reentrant, so `start` and `stop(IntermittentProcessMonitor)` take the held
//! guard as a parameter and their callers lock.  The static synchronized methods
//! (`createInstance`, `stop()`) share `CLASS_LOCK`, the class monitor.
//!
//! **Identity.**  Java's `instances` is keyed by the `IntermittentCommand` object and
//! `monitors` by the monitor object, both by identity (neither class overrides
//! `equals`).  Here the keys are the objects' addresses; so a caller must pass the same
//! command object (the same `Arc`) every time, as Java passes the same param instance.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Weak};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::process::failure_reason::FailureReason;
use crate::imod::etomo::process::intermittent_process_monitor::IntermittentProcessMonitor;
use crate::imod::etomo::process::intermittent_system_program::IntermittentSystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::hashed_array::HashedArray;
use crate::imod::etomo::util::remote_path;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static `instances`, `new Hashtable()`: one instance per
/// IntermittentCommand instance, keyed by the command object's address.
static INSTANCES: LazyLock<Mutex<HashMap<usize, Arc<IntermittentBackgroundProcess>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The class monitor of Java's `static synchronized` methods.
static CLASS_LOCK: Mutex<()> = Mutex::new(());

/// Java final class `IntermittentBackgroundProcess implements Runnable`.
pub struct IntermittentBackgroundProcess {
    /// Java private `stopped`, initialised to true.  stopped: means that the program
    /// needs to stop.
    stopped: AtomicBool,
    /// Java private `canRestart`, initialised to true.
    can_restart: AtomicBool,
    /// Java private `monitors`, `new HashedArray()`: keyed by the monitor object's
    /// address, valued by the monitor (Java's `add(monitor)` is `add(monitor,
    /// monitor)`).
    monitors: HashedArray<usize, Arc<dyn IntermittentProcessMonitor>>,
    /// Java private final `command`.
    command: Arc<dyn IntermittentCommand>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private `program`, initialised to null.
    program: Mutex<Option<Arc<IntermittentSystemProgram>>>,
    /// Java private `outputKeyPhrase`, initialised to null: string to look for in the
    /// standard output. Assumes all monitors use the same key phrase for a single
    /// instance of IntermittentCommand. This is important for intermittent commands
    /// because there is a lot more standard output then there is for a single command,
    /// and processing it can slow Etomo down.
    output_key_phrase: Option<String>,
    /// Java private `failureReason`: null unless the process fails.
    failure_reason: Mutex<Option<&'static FailureReason>>,
    /// The monitor of Java's `synchronized` instance methods.
    synchronized: Mutex<()>,
    /// Java `this`.
    this: Weak<IntermittentBackgroundProcess>,
}

impl IntermittentBackgroundProcess {
    /// Java static package-private `startInstance(BaseManager, IntermittentCommand,
    /// IntermittentProcessMonitor)`.
    pub fn start_instance(
        manager: &'static dyn BaseManager,
        command: Arc<dyn IntermittentCommand>,
        monitor: Arc<dyn IntermittentProcessMonitor>,
    ) {
        let instance =
            IntermittentBackgroundProcess::get_instance_base_manager_intermittent_command_intermittent_process_monitor(
                manager, command, &*monitor,
            );
        let synchronized = instance.synchronized.lock().unwrap();
        instance.start(&synchronized, monitor);
    }

    /// Java static package-private `endInstance(BaseManager, IntermittentCommand,
    /// IntermittentProcessMonitor)`.
    pub fn end_instance(
        _manager: &'static dyn BaseManager,
        command: &dyn IntermittentCommand,
        monitor: &dyn IntermittentProcessMonitor,
    ) {
        let intermittent_background_process =
            IntermittentBackgroundProcess::get_instance_intermittent_command(command);
        if let Some(intermittent_background_process) = intermittent_background_process {
            intermittent_background_process.end(monitor);
        }
    }

    /// Java static package-private `stopInstance(BaseManager, IntermittentCommand,
    /// IntermittentProcessMonitor)`.
    pub fn stop_instance(
        _manager: &'static dyn BaseManager,
        command: &dyn IntermittentCommand,
        monitor: &dyn IntermittentProcessMonitor,
    ) {
        let intermittent_background_process =
            IntermittentBackgroundProcess::get_instance_intermittent_command(command);
        if let Some(intermittent_background_process) = intermittent_background_process {
            let synchronized = intermittent_background_process.synchronized.lock().unwrap();
            intermittent_background_process
                .stop_intermittent_process_monitor(&synchronized, monitor);
        }
    }

    /// Java private static `getInstance(BaseManager, IntermittentCommand,
    /// IntermittentProcessMonitor)`.
    fn get_instance_base_manager_intermittent_command_intermittent_process_monitor(
        manager: &'static dyn BaseManager,
        command: Arc<dyn IntermittentCommand>,
        monitor: &dyn IntermittentProcessMonitor,
    ) -> Arc<IntermittentBackgroundProcess> {
        let intermittent_background_process =
            IntermittentBackgroundProcess::get_instance_intermittent_command(&*command);
        match intermittent_background_process {
            None => IntermittentBackgroundProcess::create_instance(manager, command, monitor),
            Some(intermittent_background_process) => intermittent_background_process,
        }
    }

    /// Java private static `getInstance(IntermittentCommand)`: `instances.get(command)`.
    fn get_instance_intermittent_command(
        command: &dyn IntermittentCommand,
    ) -> Option<Arc<IntermittentBackgroundProcess>> {
        let key = command as *const dyn IntermittentCommand as *const () as usize;
        INSTANCES.lock().unwrap().get(&key).cloned()
    }

    /// Java private static synchronized `createInstance(BaseManager,
    /// IntermittentCommand, IntermittentProcessMonitor)`.
    fn create_instance(
        manager: &'static dyn BaseManager,
        command: Arc<dyn IntermittentCommand>,
        monitor: &dyn IntermittentProcessMonitor,
    ) -> Arc<IntermittentBackgroundProcess> {
        let _synchronized = CLASS_LOCK.lock().unwrap();
        let mut intermittent_background_process =
            IntermittentBackgroundProcess::get_instance_intermittent_command(&*command);
        if intermittent_background_process.is_none() {
            let created = Arc::new_cyclic(|this| {
                IntermittentBackgroundProcess::new(manager, command, monitor, this.clone())
            });
            let key = &*created.command as *const dyn IntermittentCommand as *const () as usize;
            INSTANCES.lock().unwrap().insert(key, Arc::clone(&created));
            intermittent_background_process = Some(created);
        }
        intermittent_background_process.unwrap()
    }

    /// Java private `IntermittentBackgroundProcess(BaseManager, IntermittentCommand,
    /// IntermittentProcessMonitor)`.  `outputKeyPhrase` is always null on entry, so it
    /// takes the monitor's.
    fn new(
        manager: &'static dyn BaseManager,
        command: Arc<dyn IntermittentCommand>,
        monitor: &dyn IntermittentProcessMonitor,
        this: Weak<IntermittentBackgroundProcess>,
    ) -> IntermittentBackgroundProcess {
        let mut output_key_phrase: Option<String> = None;
        if output_key_phrase.is_none() {
            output_key_phrase = monitor.get_output_key_phrase();
        }
        IntermittentBackgroundProcess {
            stopped: AtomicBool::new(true),
            can_restart: AtomicBool::new(true),
            monitors: HashedArray::new(),
            command,
            manager,
            program: Mutex::new(None),
            output_key_phrase,
            failure_reason: Mutex::new(None),
            synchronized: Mutex::new(()),
            this,
        }
    }

    /// Java private synchronized `start(IntermittentProcessMonitor)`.  The caller holds
    /// this object's monitor and passes the guard.
    fn start(
        &self,
        _synchronized: &MutexGuard<'_, ()>,
        monitor: Arc<dyn IntermittentProcessMonitor>,
    ) {
        let _new_monitor = false;
        let Some(this) = self.this.upgrade() else {
            return;
        };
        // run the instance, if it is not running
        // this is the only place that stopped should be set to false
        if self.stopped.load(Ordering::SeqCst) {
            self.stopped.store(false, Ordering::SeqCst);
            self.can_restart.store(true, Ordering::SeqCst);
            let runner = Arc::clone(&this);
            std::thread::spawn(move || runner.run());
        }
        // Once the thread is started, add the monitor if it is new, make sure not to
        // add it more then once
        let key = Arc::as_ptr(&monitor) as *const () as usize;
        if !self.monitors.contains_key(&key) {
            self.monitors.add_object_object(key, Arc::clone(&monitor));
        }
        monitor.set_process(this);
    }

    /// Java package-private synchronized `stop(IntermittentProcessMonitor)`.  Ask the
    /// monitor to stop (it will only stop if this is its only running process).  Drop
    /// the monitor from the program.  Check whether all the monitors associated with
    /// this process are stopped.  If so, stop.  This is used when a parallel processing
    /// panel is hidden.  The caller holds this object's monitor and passes the guard.
    pub fn stop_intermittent_process_monitor(
        &self,
        _synchronized: &MutexGuard<'_, ()>,
        monitor: &dyn IntermittentProcessMonitor,
    ) {
        monitor.stop_monitoring(self);
        let program = self.program.lock().unwrap().clone();
        if let Some(program) = program {
            program.msg_dropped_monitor(monitor);
        }
        // Set monitorsStopped is this process is no longer being monitored by any
        // monitor.
        let mut monitors_stopped = true;
        for i in 0..self.monitors.size() {
            if let Some(listed) = self.monitors.get_int(i)
                && listed.is_monitoring(self)
            {
                monitors_stopped = false;
                break;
            }
        }
        // If this process is not being monitored, then stop it and prevent it from
        // being restarted by ProcessRestarter.
        if monitors_stopped {
            self.can_restart.store(false, Ordering::SeqCst);
            self.stopped.store(true, Ordering::SeqCst);
        }
    }

    /// Java package-private synchronized `restartAll()`.
    pub fn restart_all(&self) {
        let synchronized = self.synchronized.lock().unwrap();
        if !self.can_restart.load(Ordering::SeqCst) {
            return;
        }
        for i in 0..self.monitors.size() {
            if let Some(monitor) = self.monitors.get_int(i) {
                self.start(&synchronized, monitor);
            }
        }
    }

    /// Java package-private synchronized `end(IntermittentProcessMonitor)`.  Call
    /// stop(monitor) and then remove the monitor.  This is used when a manager exits.
    pub fn end(&self, monitor: &dyn IntermittentProcessMonitor) {
        let synchronized = self.synchronized.lock().unwrap();
        self.stop_intermittent_process_monitor(&synchronized, monitor);
        let key = monitor as *const dyn IntermittentProcessMonitor as *const () as usize;
        self.monitors.remove(&key);
    }

    /// Java public static synchronized `stop()`.  Stops every instance and waits for
    /// their programs to end.
    ///
    /// Upstream bugs fixed (IntermittentBackgroundProcess.java:232-258):
    /// - `program.isDone()` is called on every instance, and `program` is null for an
    ///   instance whose thread has not yet run or found no command to run, so Java
    ///   throws a NullPointerException out of `EtomoDirector`'s exit path.  An instance
    ///   with no program has nothing to wait for and counts as done.
    /// - The second "are the programs done?" loop reuses the enumeration the first loop
    ///   exhausted, so it never runs and "Error:  processes haven't stopped." can never
    ///   be printed.  It re-enumerates the instances, as the comment intends.
    pub fn stop() {
        let _synchronized = CLASS_LOCK.lock().unwrap();
        let instances: Vec<Arc<IntermittentBackgroundProcess>> =
            INSTANCES.lock().unwrap().values().cloned().collect();
        for instance in instances.iter() {
            instance.stop_all();
        }
        // wait while the processes are ending
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Are the programs done?
        let instances: Vec<Arc<IntermittentBackgroundProcess>> =
            INSTANCES.lock().unwrap().values().cloned().collect();
        let mut done = true;
        for instance in instances.iter() {
            let program = instance.program.lock().unwrap().clone();
            if let Some(program) = program
                && !program.is_done()
            {
                done = false;
            }
        }
        if done {
            return;
        }
        eprintln!("Waiting for processes to stop.");
        // make sure programs have ended
        std::thread::sleep(std::time::Duration::from_millis(1000));
        let instances: Vec<Arc<IntermittentBackgroundProcess>> =
            INSTANCES.lock().unwrap().values().cloned().collect();
        done = true;
        for instance in instances.iter() {
            let program = instance.program.lock().unwrap().clone();
            if let Some(program) = program
                && !program.is_done()
            {
                done = false;
            }
        }
        if !done {
            eprintln!("Error:  processes haven't stopped.");
        }
    }

    /// Java private synchronized `stopAll()`.
    fn stop_all(&self) {
        let _synchronized = self.synchronized.lock().unwrap();
        let program = self.program.lock().unwrap().clone();
        if let Some(program) = program {
            for i in 0..self.monitors.size() {
                if let Some(monitor) = self.monitors.get_int(i) {
                    program.msg_dropped_monitor(&*monitor);
                }
            }
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Java package-private synchronized `fail()`.  Drop all the monitors from the
    /// program.  Stop this process.  This is used when the process fails.
    pub fn fail(&self) {
        let _synchronized = self.synchronized.lock().unwrap();
        let program = self.program.lock().unwrap().clone();
        if let Some(program) = program {
            for i in 0..self.monitors.size() {
                if let Some(monitor) = self.monitors.get_int(i) {
                    program.msg_dropped_monitor(&*monitor);
                }
            }
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Java package-private final `isStopped()`.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Java `run()`.
    ///
    /// Java catches `InterruptedException` and `OutOfMemoryError` around the loop and
    /// prints their stack traces; `thread::sleep` cannot be interrupted and a Rust
    /// allocation failure aborts, so neither arm exists here.
    pub fn run(&self) {
        *self.failure_reason.lock().unwrap() = None;
        // use a local SystemProgram because stops and starts may overlap
        let mut local_program: Option<Arc<IntermittentSystemProgram>> = None;
        let local_start_command = self.command.get_local_start_command();
        let remote_start_command = self.command.get_remote_start_command();
        let local_section = remote_path::INSTANCE.is_local_section(
            self.command.get_computer().as_deref(),
            self.manager,
            AxisID::Only,
        );
        let intermittent_command = self.command.get_intermittent_command();
        if local_section && local_start_command.is_some() {
            local_program = Some(Arc::new(IntermittentSystemProgram::get_start_instance(
                self.manager,
                self.manager.get_property_user_dir(),
                local_start_command.unwrap(),
                AxisID::Only,
                self.output_key_phrase.clone(),
            )));
        } else if !local_section && remote_start_command.is_some() {
            // Java calls `command.getRemoteStartCommand()` again here.
            local_program = Some(Arc::new(IntermittentSystemProgram::get_start_instance(
                self.manager,
                self.manager.get_property_user_dir(),
                self.command
                    .get_remote_start_command()
                    .unwrap_or_else(|| remote_start_command.unwrap()),
                AxisID::Only,
                self.output_key_phrase.clone(),
            )));
        } else if let Some(intermittent_command) = &intermittent_command {
            local_program = Some(Arc::new(
                IntermittentSystemProgram::get_intermittent_instance(
                    self.manager,
                    self.manager.get_property_user_dir(),
                    intermittent_command,
                    AxisID::Only,
                    self.output_key_phrase.clone(),
                ),
            ));
        }
        // place the most recent local SystemProgram in the member SystemProgram
        // non-local request (getting and setting standard input and output) will go
        // to the most recent local SystemProgram.
        *self.program.lock().unwrap() = local_program.clone();
        // Commented out localProgram.useStartCommand because this comparison was only
        // being executed when localProgram was null, so it was meaningless.
        if let Some(local_program) = &local_program {
            local_program.set_accept_input_while_running(true);
            local_program.start();
        }
        let interval = self.command.get_interval();
        let result: std::io::Result<()> = (|| {
            // see load average requests while the program is not stopped and this
            // program is the same as the most recent program run
            loop {
                let Some(local_program) = &local_program else {
                    break;
                };
                if self.stopped.load(Ordering::SeqCst) {
                    break;
                }
                let same_program = match &*self.program.lock().unwrap() {
                    Some(program) => Arc::ptr_eq(program, local_program),
                    None => false,
                };
                if !same_program {
                    break;
                }
                if local_program.use_start_command() {
                    // Upstream bug fixed (IntermittentBackgroundProcess.java:314): a
                    // command with a start command but no intermittent command passes
                    // null to `BufferedWriter.write`, whose NullPointerException ends
                    // the thread without sending the end command.  Nothing is sent
                    // instead.
                    if let Some(intermittent_command) = &intermittent_command {
                        local_program.set_current_std_input(intermittent_command)?;
                    }
                } else if local_program.is_done() || !local_program.is_started() {
                    local_program.start();
                }
                if self.command.notify_sent_intermittent_command() {
                    for i in 0..self.monitors.size() {
                        if let Some(monitor) = self.monitors.get_int(i) {
                            monitor.msg_sent_intermittent_command(&*self.command);
                        }
                    }
                }
                // Upstream bug fixed (IntermittentBackgroundProcess.java:326):
                // `Thread.sleep` throws an uncaught IllegalArgumentException for a
                // negative interval, ending the thread without sending the end
                // command.  A negative interval sleeps for zero milliseconds.
                std::thread::sleep(std::time::Duration::from_millis(interval.max(0) as u64));
            }
            Ok(())
        })();
        if result.is_err() {
            self.stopped.store(true, Ordering::SeqCst);
            for i in 0..self.monitors.size() {
                if let Some(monitor) = self.monitors.get_int(i) {
                    monitor.msg_intermittent_command_failed(&*self.command);
                }
            }
        }
        let failure_reason = *self.failure_reason.lock().unwrap();
        let result: std::io::Result<()> = (|| {
            // Upstream bug fixed (IntermittentBackgroundProcess.java:350): Java tests
            // `failureReason.equals("")`, comparing a FailureReason with a String, which
            // is always false (the class does not override `equals`), so the destroy
            // branch could never run.  The comment below states the intent - an
            // unidentified problem, the reason "" of `FailureReason.UNKOWN` - and that
            // is what is tested.
            if let Some(failure_reason) = failure_reason
                && failure_reason.get_reason() == ""
            {
                // If there was a problem and we don't know what it is (see
                // LoadAverageMonitor.ProgramState.getFailureReason()), then destroy the
                // process. Processes with identified problems should fail naturally
                // because we are setting the PreferredAuthentications option in ssh.
                if let Some(local_program) = &local_program {
                    local_program.destroy();
                }
            } else {
                let end_command = self.command.get_end_command();
                if let (Some(end_command), Some(local_program)) = (&end_command, &local_program) {
                    // Java calls `command.getEndCommand()` again here.
                    local_program.set_current_std_input(
                        &self
                            .command
                            .get_end_command()
                            .unwrap_or_else(|| end_command.clone()),
                    )?;
                }
            }
            Ok(())
        })();
        if result.is_err()
            && let Some(local_program) = &local_program
        {
            local_program.destroy();
        }
    }

    /// Java package-private `getCommand()`.
    pub fn get_command(&self) -> Arc<dyn IntermittentCommand> {
        Arc::clone(&self.command)
    }

    /// Java package-private `getFailureReason()`.
    pub fn get_failure_reason(&self) -> Option<&'static FailureReason> {
        *self.failure_reason.lock().unwrap()
    }

    /// Java package-private `setFailureReason(FailureReason)`.
    pub fn set_failure_reason(&self, input: &'static FailureReason) {
        *self.failure_reason.lock().unwrap() = Some(input);
    }

    /// Java package-private `clearStdError()`.
    pub fn clear_std_error(&self) {
        let program = self.program.lock().unwrap().clone();
        let Some(program) = program else {
            return;
        };
        program.clear_std_error();
    }

    /// Java package-private `getStdOutput(IntermittentProcessMonitor)`.
    pub fn get_std_output(&self, monitor: &dyn IntermittentProcessMonitor) -> Option<Vec<String>> {
        // don't get output for a stopped monitor because this would make
        // OutputBufferManager start saving output for the monitor
        let program = self.program.lock().unwrap().clone();
        let key = monitor as *const dyn IntermittentProcessMonitor as *const () as usize;
        match program {
            Some(program) if self.monitors.contains_key(&key) => program.get_std_output(monitor),
            _ => None,
        }
    }

    /// Java package-private `getStdError()`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        // don't get error for a stopped monitor because this would make
        // OutputBufferManager start saving output for the monitor
        let program = self.program.lock().unwrap().clone();
        let program = program?;
        program.get_std_error()
    }

    /// Java package-private `setCurrentStdInput(String)`.  Java prints the
    /// `IOException`'s stack trace; its message is printed here.
    pub fn set_current_std_input(&self, input: &str) {
        let program = self.program.lock().unwrap().clone();
        let program = match program {
            Some(program) if !self.stopped.load(Ordering::SeqCst) => program,
            _ => return,
        };
        if let Err(e) = program.set_current_std_input(input) {
            eprintln!("{e}");
        }
    }
}

/// Java `toString()`: `"[stopped=" + stopped + "," + super.toString() + "]"`.
/// `Object.toString()` is the class name and identity hash; the object's address
/// stands in for the hash.
impl std::fmt::Display for IntermittentBackgroundProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[stopped={},etomo.process.IntermittentBackgroundProcess@{:x}]",
            self.stopped.load(Ordering::SeqCst),
            self as *const IntermittentBackgroundProcess as usize
        )
    }
}
