//! `IMOD/Etomo/src/etomo/process/LoadMonitor.java`.
//!
//! The common part of `LoadAverageMonitor` and `QueuechunkLoadMonitor`: watches the
//! intermittent background processes (one per computer) that report a computer's load,
//! and passes their output to a `LoadDisplay` (the processor table).
//!
//! **Shape.**  Java's `abstract class LoadMonitor implements IntermittentProcessMonitor,
//! Runnable` is split in two:
//! - `LoadMonitorBase` holds the fields the class declares and implements its concrete
//!   methods; each subclass embeds one as `base` (and derefs to it);
//! - the trait `LoadMonitor` is the polymorphic face the rest of the tree holds
//!   (`Arc<dyn LoadMonitor>`, e.g. `ProcessorTable.loadMonitor`): it carries the
//!   abstract `processData` and the public `restart`, and has
//!   `IntermittentProcessMonitor` as a supertrait, whose methods each subclass
//!   implements by delegating to `LoadMonitorBase`.
//!
//! A monitor is shared between the thread it runs on (`new Thread(this).start()`), the
//! intermittent processes, and the event dispatch thread, so it is `Send + Sync`: the
//! mutable fields are atomics or sit behind locks, and every method takes `&self`.
//! `LoadMonitorBase` keeps a `Weak` to the subclass object (`this`), set by the
//! subclass constructor, to start the monitor thread and to dispatch `processData`.
//!
//! The display is a Swing object (`Arc<EdtRef<dyn LoadDisplay>>`); a call on it from
//! any thread other than the event dispatch thread is posted there with
//! `invoke_later`, where Java called Swing directly from the monitor thread.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::load_display::LoadDisplay;
use crate::imod::etomo::util::event_queue::{EdtRef, invoke_later};
// TODO(unit): needs etomo/process/FailureReason.java - the static instances UNKOWN,
// COMPUTER_DOWN and LOGIN_FAILED with getReason() and getTooltip().
use crate::imod::etomo::process::failure_reason::{self, FailureReason};
// TODO(unit): needs etomo/process/IntermittentBackgroundProcess.java - the monitored
// program: getCommand, isStopped, fail, getStdError, getStdOutput, clearStdError,
// getFailureReason, setFailureReason, toString.
use crate::imod::etomo::process::intermittent_background_process::IntermittentBackgroundProcess;
// TODO(unit): needs etomo/process/IntermittentProcessMonitor.java - the interface this
// class implements.
use crate::imod::etomo::process::intermittent_process_monitor::IntermittentProcessMonitor;
// TODO(unit): needs etomo/process/ProcessRestarter.java - INSTANCE.restart() and
// INSTANCE.addProcess(IntermittentBackgroundProcess).
use crate::imod::etomo::process::process_restarter;
// TODO(unit): needs etomo/util/HashedArray.java - the synchronized keyed list of
// program states.
use crate::imod::etomo::util::hashed_array::HashedArray;

/// Java `LoadMonitor`, the polymorphic face (see the module comment).
pub trait LoadMonitor: IntermittentProcessMonitor + Send + Sync {
    /// The fields and concrete methods of Java's `LoadMonitor` (not a source member).
    fn load_monitor_base(&self) -> &LoadMonitorBase;

    /// Java abstract package-private `processData(ProgramState)`.
    fn process_data(&self, program_state: &ProgramState);

    /// Java `restart()`.
    fn restart(&self) {
        process_restarter::INSTANCE.restart();
    }
}

/// The fields Java's abstract `LoadMonitor` declares, and its concrete methods.
pub struct LoadMonitorBase {
    /// Java package-private final `display`.
    pub display: Arc<EdtRef<dyn LoadDisplay>>,
    /// Java package-private final `usersColumn`.
    pub users_column: bool,
    /// Java private `programs`, `new HashedArray()`: keyed by computer name.
    programs: HashedArray<Option<String>, Arc<ProgramState>>,
    /// Java private `stopped`, initialised to true.  True when the run() is not
    /// executing.  Set at the end of the run program.  Also set externally to stop the
    /// run() program.
    stopped: AtomicBool,
    /// Java `synchronized` on `setProcess`.
    set_process_lock: Mutex<()>,
    /// Java `this`, seen as the subclass (set once by the subclass constructor).
    this: OnceLock<Weak<dyn LoadMonitor>>,
}

impl LoadMonitorBase {
    /// Java package-private `LoadMonitor(LoadDisplay, AxisID, BaseManager)`.  The
    /// subclass constructor must then call `set_this`.
    pub fn new(
        display: Arc<EdtRef<dyn LoadDisplay>>,
        _axis_id: AxisID,
        _manager: &'static dyn BaseManager,
    ) -> LoadMonitorBase {
        LoadMonitorBase {
            display,
            users_column: cpu_adoc::INSTANCE.is_users_column(),
            programs: HashedArray::new(),
            stopped: AtomicBool::new(true),
            set_process_lock: Mutex::new(()),
            this: OnceLock::new(),
        }
    }

    /// Records the subclass object as `this` (not a source member: Java's `this` is
    /// implicit).
    pub fn set_this(&self, this: Weak<dyn LoadMonitor>) {
        let _ = self.this.set(this);
    }

    /// Java `run()`.
    ///
    /// Java catches `OutOfMemoryError` and `InterruptedException` around the loop and
    /// prints their stack traces; a Rust allocation failure aborts and
    /// `thread::sleep` cannot be interrupted, so neither arm exists here.
    pub fn run(&self) {
        let Some(this) = self.this.get().and_then(Weak::upgrade) else {
            self.stopped.store(true, Ordering::SeqCst);
            return;
        };
        while !self.stopped.load(Ordering::SeqCst) {
            let mut programs_stopped = true;
            // update the output on the display from each of the running programs.
            let mut i = 0;
            while i < self.programs.size() {
                let program_state = self.programs.get_int(i);
                if let Some(program_state) = program_state
                    && !self.stopped.load(Ordering::SeqCst)
                    && !program_state.is_stopped()
                {
                    programs_stopped = false;
                    this.process_data(&program_state);
                    if program_state.get_wait_for_command() > 12 {
                        program_state.set_wait_for_command(0);
                        self.msg_intermittent_command_failed(&*program_state.get_command());
                    }
                }
                i += 1;
            }
            if programs_stopped {
                self.stopped.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Java `stop()`.  Set stopped to true.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Java `stopMonitoring(IntermittentBackgroundProcess)`.  Sets the stopping member
    /// variable in the program state.  This just sets a flag in ProgramState and
    /// doesn't affect the program.  Check all the program states.  If they are all
    /// stopping or the program is stopped, then set stopped to true.
    pub fn stop_monitoring(&self, program: &IntermittentBackgroundProcess) {
        let program_state = self
            .programs
            .get_object(&program.get_command().get_computer());
        let Some(program_state) = program_state else {
            return;
        };
        program_state.set_stop_monitoring(true);
        let mut programs_stopped = true;
        let mut i = 0;
        while i < self.programs.size() {
            if let Some(program_state) = self.programs.get_int(i)
                && !program_state.is_stop_monitoring()
                && !program_state.is_stopped()
            {
                programs_stopped = false;
                break;
            }
            i += 1;
        }
        if programs_stopped {
            self.stopped.store(true, Ordering::SeqCst);
        }
    }

    /// Java `isMonitoring(IntermittentBackgroundProcess)`.  Returns false if the
    /// monitor is stopped, the program is unknown, or if it has received a stop
    /// monitoring command from the program.
    ///
    /// Upstream bug fixed (LoadMonitor.java:147): a computer with no program state
    /// dereferences the null `programs.get(key)` and throws a NullPointerException;
    /// the method's own contract says an unknown program is not monitored, so false is
    /// returned.
    pub fn is_monitoring(&self, program: &IntermittentBackgroundProcess) -> bool {
        if self.stopped.load(Ordering::SeqCst) {
            return false;
        }
        let key = program.get_command().get_computer();
        if key.is_none() {
            return false;
        }
        let Some(program_state) = self.programs.get_object(&key) else {
            return false;
        };
        !program_state.is_stop_monitoring()
    }

    /// Java synchronized `setProcess(IntermittentBackgroundProcess)`.
    pub fn set_process(&self, program: Arc<IntermittentBackgroundProcess>) {
        let _synchronized = self.set_process_lock.lock().unwrap();
        let key = program.get_command().get_computer();
        let program_state = self.programs.get_object(&key);
        match program_state {
            None => {
                self.programs
                    .add_object_object(key.clone(), Arc::new(ProgramState::new(program)));
            }
            Some(program_state) => {
                program_state.set_stop_monitoring(false);
                program_state.set_wait_for_command(0);
            }
        }
        if self.stopped.load(Ordering::SeqCst) {
            self.stopped.store(false, Ordering::SeqCst);
            // `new Thread(this).start()`.
            if let Some(this) = self.this.get().and_then(Weak::upgrade) {
                std::thread::spawn(move || this.load_monitor_base().run());
            }
        }
        let reason1 = failure_reason::COMPUTER_DOWN.get_reason().to_string();
        let reason2 = failure_reason::LOGIN_FAILED.get_reason().to_string();
        if self.display.is_owner_thread() {
            self.display
                .get()
                .msg_starting_process(key.as_deref(), Some(&reason1), Some(&reason2));
        } else {
            let display = self.display.clone();
            invoke_later(move || {
                display
                    .get()
                    .msg_starting_process(key.as_deref(), Some(&reason1), Some(&reason2));
            });
        }
    }

    /// Java `msgIntermittentCommandFailed(IntermittentCommand)`.
    pub fn msg_intermittent_command_failed(&self, command: &dyn IntermittentCommand) {
        let key = command.get_computer();
        if self.programs.contains_key(&key) {
            let Some(program) = self.programs.get_object(&key) else {
                return;
            };
            let failure_reason = program.get_failure_reason();
            let reason = failure_reason.get_reason().to_string();
            let tooltip = failure_reason.get_tooltip().to_string();
            if self.display.is_owner_thread() {
                self.display
                    .get()
                    .msg_load_failed(key.as_deref(), Some(&reason), Some(&tooltip));
            } else {
                let display = self.display.clone();
                let key = key.clone();
                invoke_later(move || {
                    display
                        .get()
                        .msg_load_failed(key.as_deref(), Some(&reason), Some(&tooltip));
                });
            }
            program.fail();
            program.add_to_restarter();
        }
    }

    /// Java `msgSentIntermittentCommand(IntermittentCommand)`.
    pub fn msg_sent_intermittent_command(&self, command: &dyn IntermittentCommand) {
        let program_state = self.programs.get_object(&command.get_computer());
        let Some(program_state) = program_state else {
            return;
        };
        program_state.increment_wait_for_command();
    }
}

/// Java static final package-private nested class `LoadMonitor.ProgramState`.  Shared
/// by the monitor thread and the processes' threads: counters are atomics, the user
/// lists sit behind a lock.
pub struct ProgramState {
    /// Java private final `program`.
    program: Arc<IntermittentBackgroundProcess>,
    /// Java private final `userMap`: convenience variable for counting the number of
    /// different users logged into a computer.  (A `HashMap` with null values.)
    user_map: Mutex<HashMap<String, ()>>,
    /// Java private final `userList`.
    user_list: Mutex<Vec<String>>,
    /// Java private `waitForCommand`, initialised to 0.
    wait_for_command: AtomicI32,
    /// Java private `stopMonitoring`, initialised to false.
    stop_monitoring: AtomicBool,
    /// Java private `receivedData`, initialised to false.
    received_data: AtomicBool,
    /// Java `synchronized` on `getFailureReason`.
    failure_reason_lock: Mutex<()>,
}

impl ProgramState {
    /// Java private `ProgramState(IntermittentBackgroundProcess)`.
    fn new(program: Arc<IntermittentBackgroundProcess>) -> ProgramState {
        ProgramState {
            program,
            user_map: Mutex::new(HashMap::new()),
            user_list: Mutex::new(Vec::new()),
            wait_for_command: AtomicI32::new(0),
            stop_monitoring: AtomicBool::new(false),
            received_data: AtomicBool::new(false),
            failure_reason_lock: Mutex::new(()),
        }
    }

    /// Java private `getStdError()`.
    fn get_std_error(&self) -> Option<Vec<String>> {
        self.program.get_std_error()
    }

    /// Java package-private `isStopped()`.
    pub fn is_stopped(&self) -> bool {
        self.program.is_stopped()
    }

    /// Java package-private `setStopMonitoring(boolean)`.
    pub fn set_stop_monitoring(&self, stop_monitoring: bool) {
        self.stop_monitoring
            .store(stop_monitoring, Ordering::SeqCst);
    }

    /// Java package-private `isStopMonitoring()`.
    pub fn is_stop_monitoring(&self) -> bool {
        self.stop_monitoring.load(Ordering::SeqCst)
    }

    /// Java package-private `fail()`.
    pub fn fail(&self) {
        self.program.fail();
    }

    /// Java package-private `getWaitForCommand()`.
    pub fn get_wait_for_command(&self) -> i32 {
        self.wait_for_command.load(Ordering::SeqCst)
    }

    /// Java package-private `incrementWaitForCommand()`.
    pub fn increment_wait_for_command(&self) {
        self.wait_for_command.fetch_add(1, Ordering::SeqCst);
    }

    /// Java package-private `setWaitForCommand(int)`.
    pub fn set_wait_for_command(&self, wait_for_command: i32) {
        self.wait_for_command
            .store(wait_for_command, Ordering::SeqCst);
    }

    /// Java package-private `getCommand()`.
    pub fn get_command(&self) -> Arc<dyn IntermittentCommand> {
        self.program.get_command()
    }

    /// Java package-private `addToRestarter()`.
    pub fn add_to_restarter(&self) {
        process_restarter::INSTANCE.add_process(self.program.clone());
    }

    /// Java package-private `getStdOutput(IntermittentProcessMonitor)`.
    pub fn get_std_output(&self, monitor: &dyn IntermittentProcessMonitor) -> Option<Vec<String>> {
        self.program.get_std_output(monitor)
    }

    /// Java package-private `msgReceivedData()`.  Sets receivedData to true, resets
    /// failureReason, and clears stderr.
    pub fn msg_received_data(&self) {
        if self.received_data.load(Ordering::SeqCst) {
            return;
        }
        self.received_data.store(true, Ordering::SeqCst);
        self.clear_std_error();
    }

    /// Java package-private `clearStdError()`.
    pub fn clear_std_error(&self) {
        self.program.clear_std_error();
    }

    /// Java private synchronized `getFailureReason()`.  Provide a failure reason if the
    /// state of stderr shows that the computer being ssh'ed to is down or the
    /// authentication failed.  Sets the failure reason to non-null.
    ///
    /// Java catches `OutOfMemoryError` around the stderr scan; a Rust allocation
    /// failure aborts, so that arm does not exist here.
    fn get_failure_reason(&self) -> &'static FailureReason {
        let _synchronized = self.failure_reason_lock.lock().unwrap();
        // There was a failure, so failureReason must not be null
        let failure_reason = self.program.get_failure_reason();
        if failure_reason.is_none() {
            self.program.set_failure_reason(&failure_reason::UNKOWN);
        }
        // If data has already been received, this is an unknown error
        if self.received_data.load(Ordering::SeqCst) {
            self.program.set_failure_reason(&failure_reason::UNKOWN);
            return &failure_reason::UNKOWN;
        }
        // If stderr is empty but receivedData is false, return the existing failure
        // reason
        let stderr = self.get_std_error();
        let stderr = match stderr {
            Some(stderr) if !stderr.is_empty() => stderr,
            // The reason was made non-null above.
            _ => {
                return self
                    .program
                    .get_failure_reason()
                    .unwrap_or(&failure_reason::UNKOWN);
            }
        };
        // Try to set a failure reason from the information in stderr
        let mut connection_succeeded = false;
        let mut failure_reason: &'static FailureReason = &failure_reason::UNKOWN;
        for i in 0..stderr.len() {
            if !connection_succeeded {
                // `String.toLowerCase()` (default locale).
                let line = stderr[i].to_lowercase();
                if line.contains("connecting to") {
                    failure_reason = &failure_reason::COMPUTER_DOWN;
                } else if line.contains("next authentication") {
                    failure_reason = &failure_reason::LOGIN_FAILED;
                } else if line.contains("authentication succeeded") {
                    // Not a connection failure. Don't know why this failed.
                    connection_succeeded = true;
                    failure_reason = &failure_reason::UNKOWN;
                }
            }
        }
        self.program.set_failure_reason(failure_reason);
        failure_reason
    }

    /// Java package-private `clearUsers()`.
    pub fn clear_users(&self) {
        self.user_map.lock().unwrap().clear();
        self.user_list.lock().unwrap().clear();
    }

    /// Java package-private `containsUser(String)`.
    pub fn contains_user(&self, user: &str) -> bool {
        self.user_map.lock().unwrap().contains_key(user)
    }

    /// Java package-private `addUser(String)`.
    pub fn add_user(&self, user: &str) {
        self.user_map.lock().unwrap().insert(user.to_string(), ());
        self.user_list.lock().unwrap().push(user.to_string());
    }

    /// Java package-private `getUserList()`.
    pub fn get_user_list(&self) -> Option<String> {
        let user_list = self.user_list.lock().unwrap();
        if user_list.is_empty() {
            return None;
        }
        let mut list = user_list[0].clone();
        for i in 1..user_list.len() {
            list.push_str(&format!(",{}", user_list[i]));
        }
        Some(list)
    }
}

/// Java `ProgramState.toString()`.  `super.toString()` is `Object.toString()`, the
/// class name and identity hash; the object's address stands in for the hash.
impl std::fmt::Display for ProgramState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[program={},\nwaitForCommand={},etomo.process.LoadMonitor$ProgramState@{:x}]",
            self.program,
            self.wait_for_command.load(Ordering::SeqCst),
            self as *const ProgramState as usize
        )
    }
}
