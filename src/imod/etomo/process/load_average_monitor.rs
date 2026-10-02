//! `IMOD/Etomo/src/etomo/process/LoadAverageMonitor.java`.
//!
//! `LoadAverageMonitor extends LoadMonitor`: reads the output of `w`/`uptime` (or the
//! Windows CPU usage report) from each computer's intermittent process and shows the
//! load averages, and the users logged in, on the load display.  See
//! `load_monitor.rs` for how the abstract superclass is split into `LoadMonitorBase`
//! (embedded as `base`) and the `LoadMonitor` trait.

use std::sync::{Arc, Weak};

use regex::Regex;

use super::load_monitor::{LoadMonitor, LoadMonitorBase, ProgramState};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_double_value_of, java_lang_string_trim,
};
use crate::imod::etomo::ui::swing::load_display::LoadDisplay;
use crate::imod::etomo::util::event_queue::{EdtRef, invoke_later};
use crate::imod::etomo::util::utilities::{self, java_lang_string_split};
// TODO(unit): needs etomo/process/IntermittentBackgroundProcess.java.
use crate::imod::etomo::process::intermittent_background_process::IntermittentBackgroundProcess;
// TODO(unit): needs etomo/process/IntermittentProcessMonitor.java.
use crate::imod::etomo::process::intermittent_process_monitor::IntermittentProcessMonitor;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java private static final `OUTPUT_KEY_PHRASE`.
const OUTPUT_KEY_PHRASE: &str = "load average";
/// Java private static final `OUTPUT_KEY_PHRASE_WINDOWS`.
const OUTPUT_KEY_PHRASE_WINDOWS: &str = "Percent CPU usage";

/// Java `LoadAverageMonitor`.
pub struct LoadAverageMonitor {
    /// Java superclass `LoadMonitor` state.
    pub base: LoadMonitorBase,
    /// Java private final `numberOfProcessorsWindows`,
    /// `EtomoDirector.INSTANCE.getNumberOfProcessorsWindows()`.
    number_of_processors_windows: Option<ConstEtomoNumber>,
}

impl std::ops::Deref for LoadAverageMonitor {
    type Target = LoadMonitorBase;

    fn deref(&self) -> &LoadMonitorBase {
        &self.base
    }
}

impl LoadAverageMonitor {
    /// Java `LoadAverageMonitor(LoadDisplay, AxisID, BaseManager)`.
    pub fn new(
        display: Arc<EdtRef<dyn LoadDisplay>>,
        axis_id: AxisID,
        manager: &'static dyn BaseManager,
    ) -> Arc<LoadAverageMonitor> {
        let instance = Arc::new(LoadAverageMonitor {
            base: LoadMonitorBase::new(display, axis_id, manager),
            number_of_processors_windows: etomo_director::INSTANCE
                .get_number_of_processors_windows(),
        });
        let this: Arc<dyn LoadMonitor> = instance.clone();
        let this: Weak<dyn LoadMonitor> = Arc::downgrade(&this);
        instance.base.set_this(this);
        instance
    }

    /// Java private `getLoad(String)`.  `Err` is the `NumberFormatException` of
    /// `Double.parseDouble` (or the `StringIndexOutOfBoundsException` of an empty
    /// string).
    fn get_load(&self, load: &str) -> Result<f64, String> {
        let load = java_lang_string_trim(load);
        if load.is_empty() {
            return Err(format!("String index out of range: {}", -1));
        }
        if load.ends_with(',') {
            return java_lang_double_value_of(&load[..load.len() - 1]);
        }
        java_lang_double_value_of(load)
    }
}

impl LoadMonitor for LoadAverageMonitor {
    fn load_monitor_base(&self) -> &LoadMonitorBase {
        &self.base
    }

    /// Java package-private `processData(ProgramState)`.  Processes the output of
    /// programState.program.  Sets the results in display.  This function is meant to
    /// be called over and over while programState.program is running.
    ///
    /// Upstream bug fixed (LoadAverageMonitor.java:62, 68-69, 116-118): a key line whose
    /// load fields are missing (fewer than three words) or not numbers throws an
    /// `ArrayIndexOutOfBoundsException`/`NumberFormatException` on the monitor thread,
    /// which ends `run()` without resetting `stopped`, so monitoring stops for good and
    /// can never be restarted.  Such a line is ignored instead, as if it were not a key
    /// line.
    fn process_data(&self, program_state: &ProgramState) {
        // process standard out
        let stdout = program_state.get_std_output(self);
        let stdout = match stdout {
            Some(stdout) if !stdout.is_empty() => stdout,
            _ => return,
        };
        program_state.msg_received_data();
        let mut cpu_usage: f64 = -1.0;
        let mut load1: f64 = -1.0;
        let mut load5: f64 = -1.0;
        program_state.clear_users();
        let mut users: i32 = 0;
        let mut header_line_found = false;
        // `split("\\s+")`, with Java's `\s` = [ \t\n\x0B\f\r].
        let whitespace = Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap();
        for i in 0..stdout.len() {
            if utilities::is_windows_os() {
                if stdout[i].contains(OUTPUT_KEY_PHRASE_WINDOWS) {
                    program_state.set_wait_for_command(0);
                    let array =
                        java_lang_string_split(java_lang_string_trim(&stdout[i]), &whitespace);
                    let Some(last) = array.last() else {
                        continue;
                    };
                    let Ok(value) = self.get_load(last) else {
                        continue;
                    };
                    cpu_usage = value;
                }
            } else if stdout[i].contains(OUTPUT_KEY_PHRASE) {
                program_state.set_wait_for_command(0);
                let array = java_lang_string_split(java_lang_string_trim(&stdout[i]), &whitespace);
                if array.len() < 3 {
                    continue;
                }
                let (Ok(value1), Ok(value5)) = (
                    self.get_load(&array[array.len() - 3]),
                    self.get_load(&array[array.len() - 2]),
                ) else {
                    continue;
                };
                load1 = value1;
                load5 = value5;
            }
            // no need to total users when the usersColumn is not being displayed
            else if self.users_column {
                if !header_line_found {
                    // ignore the header line
                    header_line_found = true;
                } else {
                    // count users
                    let array =
                        java_lang_string_split(java_lang_string_trim(&stdout[i]), &whitespace);
                    // A blank line splits to one empty word in Java.
                    let user = array.first().map(String::as_str).unwrap_or("");
                    if user != "root" && !program_state.contains_user(user) {
                        program_state.add_user(user);
                        users += 1;
                    }
                }
            }
        }
        let computer = program_state.get_command().get_computer();
        if utilities::is_windows_os() {
            if cpu_usage == -1.0 {
                return;
            }
            let number_of_processors = self.number_of_processors_windows.clone();
            if self.display.is_owner_thread() {
                self.display.get().set_cpu_usage(
                    computer.as_deref(),
                    cpu_usage,
                    number_of_processors.as_ref(),
                );
            } else {
                let display = self.display.clone();
                invoke_later(move || {
                    display.get().set_cpu_usage(
                        computer.as_deref(),
                        cpu_usage,
                        number_of_processors.as_ref(),
                    );
                });
            }
        } else {
            if load1 == -1.0 {
                return;
            }
            let user_list = program_state.get_user_list();
            if self.display.is_owner_thread() {
                self.display.get().set_load_string_double_double_int_string(
                    computer.as_deref(),
                    load1,
                    load5,
                    users,
                    user_list.as_deref(),
                );
            } else {
                let display = self.display.clone();
                invoke_later(move || {
                    display.get().set_load_string_double_double_int_string(
                        computer.as_deref(),
                        load1,
                        load5,
                        users,
                        user_list.as_deref(),
                    );
                });
            }
        }
    }
}

/// Java `implements IntermittentProcessMonitor`, inherited from `LoadMonitor` except
/// for `getOutputKeyPhrase`.
impl IntermittentProcessMonitor for LoadAverageMonitor {
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
        if utilities::is_windows_os() {
            return Some(OUTPUT_KEY_PHRASE_WINDOWS.to_string());
        }
        // need to get users for linux systems, so don't use the output key phrase to
        // limit process output
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
