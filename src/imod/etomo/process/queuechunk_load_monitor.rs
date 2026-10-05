//! `IMOD/Etomo/src/etomo/process/QueuechunkLoadMonitor.java`.
//!
//! `QueuechunkLoadMonitor extends LoadMonitor`: has mostly the same functionality as
//! LoadAverageMonitor, except for the output (a comma-separated load array from
//! `queuechunk`) and how it is used.  See `load_monitor.rs` for how the abstract
//! superclass is split into `LoadMonitorBase` (embedded as `base`) and the
//! `LoadMonitor` trait.

use std::sync::{Arc, Weak};

use regex::Regex;

use super::load_monitor::{LoadMonitor, LoadMonitorBase, ProgramState};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::load_display::LoadDisplay;
use crate::imod::etomo::util::event_queue::{EdtRef, invoke_later};
use crate::imod::etomo::util::utilities::java_lang_string_split;
use crate::imod::etomo::process::intermittent_background_process::IntermittentBackgroundProcess;
use crate::imod::etomo::process::intermittent_process_monitor::IntermittentProcessMonitor;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `QueuechunkLoadMonitor`.
pub struct QueuechunkLoadMonitor {
    /// Java superclass `LoadMonitor` state.
    pub base: LoadMonitorBase,
}

impl std::ops::Deref for QueuechunkLoadMonitor {
    type Target = LoadMonitorBase;

    fn deref(&self) -> &LoadMonitorBase {
        &self.base
    }
}

impl QueuechunkLoadMonitor {
    /// Java `QueuechunkLoadMonitor(LoadDisplay, AxisID, BaseManager)`.
    pub fn new(
        display: Arc<EdtRef<dyn LoadDisplay>>,
        axis_id: AxisID,
        manager: &'static dyn BaseManager,
    ) -> Arc<QueuechunkLoadMonitor> {
        let instance = Arc::new(QueuechunkLoadMonitor {
            base: LoadMonitorBase::new(display, axis_id, manager),
        });
        let this: Arc<dyn LoadMonitor> = instance.clone();
        let this: Weak<dyn LoadMonitor> = Arc::downgrade(&this);
        instance.base.set_this(this);
        instance
    }
}

impl LoadMonitor for QueuechunkLoadMonitor {
    fn load_monitor_base(&self) -> &LoadMonitorBase {
        &self.base
    }

    /// Java package-private `processData(ProgramState)`.
    fn process_data(&self, program_state: &ProgramState) {
        // process standard out
        let stdout = program_state.get_std_output(self);
        let stdout = match stdout {
            Some(stdout) if !stdout.is_empty() => stdout,
            _ => return,
        };
        program_state.msg_received_data();
        program_state.clear_users();
        // `stdout[0].split("\\s*\\,\\s*")`, with Java's `\s` = [ \t\n\x0B\f\r].
        let load_array = java_lang_string_split(
            &stdout[0],
            &Regex::new(r"[ \t\n\x0B\x0C\r]*,[ \t\n\x0B\x0C\r]*").unwrap(),
        );
        if load_array.is_empty() {
            return;
        }
        let computer = program_state.get_command().get_computer();
        if self.display.is_owner_thread() {
            self.display
                .get()
                .set_load_string_string_array(computer.as_deref(), &load_array);
        } else {
            let display = self.display.clone();
            invoke_later(move || {
                display
                    .get()
                    .set_load_string_string_array(computer.as_deref(), &load_array);
            });
        }
    }
}

/// Java `implements IntermittentProcessMonitor`, inherited from `LoadMonitor` except
/// for `getOutputKeyPhrase`.
impl IntermittentProcessMonitor for QueuechunkLoadMonitor {
    fn set_process(&self, program: Arc<IntermittentBackgroundProcess>) {
        self.base.set_process(program);
    }

    fn msg_intermittent_command_failed(&self, command: &dyn IntermittentCommand) {
        self.base.msg_intermittent_command_failed(command);
    }

    fn msg_sent_intermittent_command(&self, command: &dyn IntermittentCommand) {
        self.base.msg_sent_intermittent_command(command);
    }

    /// Java final `getOutputKeyPhrase()`.
    fn get_output_key_phrase(&self) -> Option<String> {
        None
    }

    fn stop(&self) {
        self.base.stop();
    }

    fn is_monitoring(&self, program: &IntermittentBackgroundProcess) -> bool {
        self.base.is_monitoring(program)
    }

    fn stop_monitoring(&self, program: &IntermittentBackgroundProcess) {
        self.base.stop_monitoring(program);
    }
}
