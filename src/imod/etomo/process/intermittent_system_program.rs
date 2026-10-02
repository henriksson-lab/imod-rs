//! `IMOD/Etomo/src/etomo/process/IntermittentSystemProgram.java`.
//!
//! Runs a program that stays open (a `bash`/`ssh` session) and sends it a string
//! through standard input at intervals, or reruns a one-shot command.  Wraps a
//! `SystemProgram`, which starts the child through `system_program::runtime_exec`
//! like the rest of the process layer.
//!
//! Shared between the `IntermittentBackgroundProcess` thread that drives it, the
//! `SystemProgram::run` thread, and the monitors reading its output, so it is held in
//! an `Arc` and every method takes `&self`.
//!
//! A monitor is the Java listener key of the output buffers (object identity); here the
//! key is the monitor object's address, formatted as text (see
//! `intermittent_process_monitor`).

use std::sync::Arc;

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::intermittent_process_monitor::IntermittentProcessMonitor;
use crate::imod::etomo::process::output_buffer_manager::OutputBufferManager;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::utilities;

/// Java package-private class `IntermittentSystemProgram`.
pub struct IntermittentSystemProgram {
    /// Java private final `outputKeyPhrase`.  Read only by `newOutputBufferManager`,
    /// which nothing calls (the class does not extend `SystemProgram`, so it overrides
    /// nothing): in the source too, the key phrase never reaches the output buffers.
    output_key_phrase: Option<String>,
    /// Java private final `program`.
    program: Arc<SystemProgram>,
    /// Java private final `useStartCommand`.
    use_start_command: bool,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private `debug`, initialised to false.
    debug: bool,
}

impl IntermittentSystemProgram {
    /// Java private `IntermittentSystemProgram(BaseManager, String, String[], AxisID,
    /// String, boolean)`.
    fn new(
        manager: &'static dyn BaseManager,
        property_user_dir: Option<String>,
        cmd_array: Vec<String>,
        axis_id: AxisID,
        output_key_phrase: Option<String>,
        use_start_command: bool,
    ) -> IntermittentSystemProgram {
        let program = Arc::new(SystemProgram::new_array(
            Some(manager),
            property_user_dir,
            Some(cmd_array),
            axis_id,
        ));
        program.set_collect_output(false);
        IntermittentSystemProgram {
            output_key_phrase,
            program,
            use_start_command,
            axis_id,
            debug: false,
        }
    }

    /// Java static package-private `getStartInstance(BaseManager, String, String[],
    /// AxisID, String)`.
    pub fn get_start_instance(
        manager: &'static dyn BaseManager,
        property_user_dir: Option<String>,
        start_cmd_array: Vec<String>,
        axis_id: AxisID,
        output_key_phrase: Option<String>,
    ) -> IntermittentSystemProgram {
        IntermittentSystemProgram::new(
            manager,
            property_user_dir,
            start_cmd_array,
            axis_id,
            output_key_phrase,
            true,
        )
    }

    /// Java static package-private `getIntermittentInstance(BaseManager, String, String,
    /// AxisID, String)`.  IntermittentCommand will be split on whitespace.  It must not
    /// contain any whitespace in the directory path.
    ///
    /// `split("\\s+")`: Java's `\s` is the ASCII set `[ \t\n\x0B\f\r]`, not Rust's
    /// Unicode `\s`.
    pub fn get_intermittent_instance(
        manager: &'static dyn BaseManager,
        property_user_dir: Option<String>,
        intermittent_command: &str,
        axis_id: AxisID,
        output_key_phrase: Option<String>,
    ) -> IntermittentSystemProgram {
        let pattern = Regex::new("[ \\t\\n\\x0B\\x0C\\r]+").unwrap();
        IntermittentSystemProgram::new(
            manager,
            property_user_dir,
            utilities::java_lang_string_split(intermittent_command, &pattern),
            axis_id,
            output_key_phrase,
            false,
        )
    }

    /// Java package-private `useStartCommand()`.
    pub fn use_start_command(&self) -> bool {
        self.use_start_command
    }

    /// Java package-private `newOutputBufferManager(BufferedReader)`.  The Rust
    /// `OutputBufferManager` is fed by `SystemProgram`'s reader thread, so it takes no
    /// reader.  Nothing calls this (see `output_key_phrase`).
    pub fn new_output_buffer_manager(&self) -> OutputBufferManager {
        // `new OutputBufferManager(cmdBuffer, outputKeyPhrase)`: a null key phrase is
        // no key phrase.
        let mut buffer_manager = match &self.output_key_phrase {
            Some(output_key_phrase) => OutputBufferManager::with_key_phrase(output_key_phrase),
            None => OutputBufferManager::new(),
        };
        buffer_manager.set_collect_output(false);
        buffer_manager
    }

    /// Java package-private `newErrorBufferManager(BufferedReader)`.  Nothing calls
    /// this (see `new_output_buffer_manager`).
    pub fn new_error_buffer_manager(&self) -> OutputBufferManager {
        let mut buffer_manager = OutputBufferManager::new();
        buffer_manager.set_collect_output(false);
        buffer_manager
    }

    /// Java package-private `clearStdError()`.  Clear stderr.  Because
    /// stderr.collectionOutput is false, this can be done by running stderr.get();
    /// (`program` is final and never null, so Java's null test always passes.)
    pub fn clear_std_error(&self) {
        self.program.clear_std_error();
    }

    /// Java package-private `isDone()`.
    pub fn is_done(&self) -> bool {
        self.program.is_done()
    }

    /// Java package-private `isStarted()`.
    pub fn is_started(&self) -> bool {
        self.program.is_started()
    }

    /// Java package-private `destroy()`.
    pub fn destroy(&self) {
        self.program.destroy();
    }

    /// Java package-private `getStdError()`.
    pub fn get_std_error(&self) -> Option<Vec<String>> {
        self.program.get_std_error()
    }

    /// Java package-private `setAcceptInputWhileRunning(boolean)`.
    pub fn set_accept_input_while_running(&self, accept_input_while_running: bool) {
        self.program
            .set_accept_input_while_running(accept_input_while_running);
    }

    /// Java package-private `setCurrentStdInput(String) throws IOException`.
    pub fn set_current_std_input(&self, input: &str) -> std::io::Result<()> {
        self.program.set_current_std_input(input)
    }

    /// Java package-private `start()`: `new Thread(program).start()`.
    pub fn start(&self) {
        let program = Arc::clone(&self.program);
        std::thread::spawn(move || program.run());
    }

    /// Java package-private `getStdOutput(IntermittentProcessMonitor)`.  Get the
    /// standard output from the execution of the program.  Returns an array of strings
    /// containing the standard output from the program.  Each line of standard out is
    /// stored in a String.
    pub fn get_std_output(&self, monitor: &dyn IntermittentProcessMonitor) -> Option<Vec<String>> {
        let listener_key = format!(
            "{:p}",
            monitor as *const dyn IntermittentProcessMonitor as *const ()
        );
        self.program.get_std_output_listener(&listener_key)
    }

    /// Java package-private `getStdError(IntermittentProcessMonitor)`.  Get the
    /// standard error from the execution of the program.  Returns an array of strings
    /// containing the standard error from the program.  Each line of standard err is
    /// stored in a String.
    pub fn get_std_error_intermittent_process_monitor(
        &self,
        monitor: &dyn IntermittentProcessMonitor,
    ) -> Option<Vec<String>> {
        let listener_key = format!(
            "{:p}",
            monitor as *const dyn IntermittentProcessMonitor as *const ()
        );
        self.program.get_std_error_listener(&listener_key)
    }

    /// Java package-private `msgDroppedMonitor(IntermittentProcessMonitor)`.
    pub fn msg_dropped_monitor(&self, monitor: &dyn IntermittentProcessMonitor) {
        let listener_key = format!(
            "{:p}",
            monitor as *const dyn IntermittentProcessMonitor as *const ()
        );
        self.program.drop_std_output_listener(&listener_key);
    }
}
