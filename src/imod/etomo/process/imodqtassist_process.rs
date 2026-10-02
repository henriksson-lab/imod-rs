//! `IMOD/Etomo/src/etomo/process/ImodqtassistProcess.java`.
//!
//! Runs imodqtassist and sends commands to it via its stdin (help mode), or runs
//! `imodqtassist -t` and returns its output (query mode).  Quits imodqtassist when
//! etomo exits.  `imodqtassist` is started through `system_program::runtime_exec`, so
//! it resolves like every other IMOD program.

use super::interactive_system_program::InteractiveSystemProgram;
use super::system_program::SystemProgram;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::ui_harness;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

/// Java `INSTANCE`, the help-mode instance.
pub static INSTANCE: LazyLock<ImodqtassistProcess> =
    LazyLock::new(|| ImodqtassistProcess::new(Mode::Help));

/// Java private static final nested `Mode`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Help,
    Query,
}

/// Java field `program`, declared `Runnable`: a `SystemProgram` in query mode, an
/// `InteractiveSystemProgram` in help mode.
#[derive(Clone)]
enum Program {
    System(Arc<SystemProgram>),
    Interactive(Arc<InteractiveSystemProgram>),
}

/// Java `ImodqtassistProcess`.
pub struct ImodqtassistProcess {
    mode: Mode,
    program: Mutex<Option<Program>>,
}

impl ImodqtassistProcess {
    /// Java private `ImodqtassistProcess(Mode)`.
    fn new(mode: Mode) -> ImodqtassistProcess {
        ImodqtassistProcess {
            mode,
            program: Mutex::new(None),
        }
    }

    /// Java static `getQueryInstance`.
    pub fn get_query_instance() -> ImodqtassistProcess {
        ImodqtassistProcess::new(Mode::Query)
    }

    /// Java `open`.
    ///
    /// Upstream NPE fixed in translation (ImodqtassistProcess.java:105 and
    /// InteractiveSystemProgram.java:281): `run` dereferences a null manager
    /// for its user directory (in query mode directly, in help mode on the
    /// program's own thread, so imodqtassist never starts).  With no manager
    /// the program is not run; `send` then reports that it cannot send.
    pub fn open(&self, manager: Option<&'static dyn BaseManager>, action: &str, axis_id: AxisID) {
        if self.mode == Mode::Query {
            if let Some(manager) = manager {
                self.run(manager, axis_id);
            }
        } else {
            // Help mode
            let program_is_null = self.program.lock().unwrap().is_none();
            if program_is_null && let Some(manager) = manager {
                self.run(manager, axis_id);
            }
            if self.send(manager, action, axis_id).is_err() {
                // try running the program again, in case it died
                if let Some(manager) = manager {
                    self.run(manager, axis_id);
                }
                if let Err(e1) = self.send(manager, action, axis_id) {
                    eprintln!("{e1}");
                }
            }
        }
    }

    /// Java private `send`.
    fn send(
        &self,
        manager: Option<&'static dyn BaseManager>,
        action: &str,
        axis_id: AxisID,
    ) -> std::io::Result<()> {
        let program = self.program.lock().unwrap().clone();
        let interactive_program = match (&program, self.mode) {
            (Some(Program::Interactive(interactive_program)), Mode::Help) => {
                Arc::clone(interactive_program)
            }
            _ => {
                println!("Warning: unable to send {action} command to imodqtassist");
                return Ok(());
            }
        };
        interactive_program.set_current_std_input(action)?;
        std::thread::sleep(Duration::from_millis(500));
        let _buffer = String::new();
        while let Some(line) = interactive_program.read_stderr() {
            if line.starts_with("ERROR:") || line.starts_with("WARNING:") {
                ui_harness::post_message_dialog(
                    manager,
                    line,
                    "Problem Displaying Help Topic".to_string(),
                    Some(axis_id),
                );
            }
        }
        Ok(())
    }

    /// Java package-private `getStandardOutput`.
    pub fn get_standard_output(&self) -> Option<Vec<String>> {
        let program = self.program.lock().unwrap().clone()?;
        if self.mode == Mode::Query
            && let Program::System(program) = &program
        {
            return program.get_std_output();
        }
        // Help mode
        if let Program::Interactive(program) = &program {
            return program.get_std_output();
        }
        None
    }

    /// Java package-private `run`.
    pub fn run(&self, manager: &'static dyn BaseManager, axis_id: AxisID) {
        let program = {
            let mut program = self.program.lock().unwrap();
            if program.is_none() {
                let mut command_list: Vec<String> = Vec::new();
                command_list.push("imodqtassist".to_string());
                if self.mode == Mode::Query {
                    command_list.push("-t".to_string());
                    *program = Some(Program::System(Arc::new(SystemProgram::new_array(
                        Some(manager),
                        manager.get_property_user_dir(),
                        Some(command_list),
                        axis_id,
                    ))));
                } else {
                    // Help mode
                    // Construct the interactive system program.
                    command_list.push("-p".to_string());
                    command_list.push("IMOD.adp".to_string());
                    command_list.push("-k".to_string());
                    command_list.push("html".to_string());
                    *program = Some(Program::Interactive(Arc::new(
                        InteractiveSystemProgram::new(manager, Some(command_list), axis_id),
                    )));
                }
            }
            program.clone().unwrap()
        };
        // run program
        match program {
            Program::System(program) => {
                std::thread::spawn(move || program.run());
            }
            Program::Interactive(program) => {
                std::thread::spawn(move || program.run());
            }
        }
        // Wait while the imodqtassist starts
        std::thread::sleep(Duration::from_millis(500));
    }

    /// Java `quit`.
    pub fn quit(&self) {
        let Some(program) = self.program.lock().unwrap().clone() else {
            return;
        };
        if self.mode == Mode::Help
            && let Program::Interactive(program) = &program
        {
            // program is probably already dead
            let _ = program.set_current_std_input("q");
        }
        std::thread::sleep(Duration::from_millis(50));
        *self.program.lock().unwrap() = None;
    }
}
