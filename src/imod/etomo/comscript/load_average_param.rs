//! `IMOD/Etomo/src/etomo/comscript/LoadAverageParam.java`.
//!
//! The intermittent command that reads a computer's load average (`w`, or
//! `imodwincpu` on Windows) through a shell started locally or over ssh.  One
//! instance per computer.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use super::intermittent_command::IntermittentCommand;
use super::ssh_param;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static `instances`: one instance per computer.  Instances live
/// for the rest of the program, as they do in the Java `Hashtable`.
static INSTANCES: LazyLock<Mutex<HashMap<String, std::sync::Arc<LoadAverageParam>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The lazily built command fields.
struct Commands {
    /// Java private field `localStartCommandArray`.
    local_start_command_array: Option<Vec<String>>,
    /// Java private field `remoteStartCommandArray`.
    remote_start_command_array: Option<Vec<String>>,
    /// Java private field `intermittentCommand`.
    intermittent_command: Option<String>,
    /// Java private field `endCommand`.
    end_command: Option<String>,
}

/// Java `LoadAverageParam`.
pub struct LoadAverageParam {
    /// Java private final field `computer`.
    computer: String,
    /// Java private final field `manager`.
    manager: &'static dyn BaseManager,
    /// The lazily built fields; instances are shared between the monitor threads.
    commands: Mutex<Commands>,
    /// Java private field `debug`.
    debug: DebugLevel,
}

impl LoadAverageParam {
    /// Java public final static `getInstance(String, BaseManager)`.
    pub fn get_instance(
        computer: &str,
        manager: &'static dyn BaseManager,
    ) -> std::sync::Arc<LoadAverageParam> {
        // Java checks the table once unsynchronized and again inside
        // `synchronized (instances)`; the lock covers both here.
        let mut instances = INSTANCES.lock().unwrap();
        if let Some(load_average_param) = instances.get(computer) {
            return std::sync::Arc::clone(load_average_param);
        }
        let load_average_param = std::sync::Arc::new(LoadAverageParam::new(computer, manager));
        instances.insert(
            computer.to_string(),
            std::sync::Arc::clone(&load_average_param),
        );
        load_average_param
    }

    /// Java private `LoadAverageParam(String, BaseManager)`.
    fn new(computer: &str, manager: &'static dyn BaseManager) -> LoadAverageParam {
        LoadAverageParam {
            computer: computer.to_string(),
            manager,
            commands: Mutex::new(Commands {
                local_start_command_array: None,
                remote_start_command_array: None,
                intermittent_command: None,
                end_command: None,
            }),
            debug: etomo_director::ARGUMENTS.lock().unwrap().get_debug_level(),
        }
    }

    /// Java private final `buildLocalStartCommand`.
    fn build_local_start_command(&self, commands: &mut Commands) {
        let mut command: Vec<String> = Vec::new();
        if utilities::is_windows_os() {
            command.push("cmd".to_string());
        } else {
            // If the user is a bash user, a bad .cshrc might cause local load average to
            // fail without causing any other symptoms. So its safer to use the bash
            // shell for a bash user.
            // Use bash as the default. Use tcsh only when it is set in $SHELL.
            let tcsh_shell = "tcsh";
            let shell = environment_variable::INSTANCE.get_value(
                Some(self.manager),
                self.manager.get_property_user_dir().as_deref(),
                "SHELL",
                Some(AxisID::Only),
            );
            if shell.contains(tcsh_shell) {
                command.push(tcsh_shell.to_string());
            } else {
                command.push("bash".to_string());
            }
        }
        let command_size = command.len();
        let mut local_start_command_array = Vec::with_capacity(command_size);
        if self.debug.is_verbose() {
            eprint!("local start command:");
        }
        for i in 0..command_size {
            local_start_command_array.push(command[i].clone());
            if self.debug.is_verbose() {
                eprint!("{} ", command[i]);
            }
        }
        if self.debug.is_verbose() && command_size > 0 {
            eprintln!();
        }
        commands.local_start_command_array = Some(local_start_command_array);
    }

    /// Java private final `buildRemoteStartCommand`.
    fn build_remote_start_command(&self, commands: &mut Commands) {
        let command = ssh_param::INSTANCE.get_command(self.manager, false, Some(&self.computer));
        let command_size = command.len();
        let mut remote_start_command_array = Vec::with_capacity(command_size);
        for i in 0..command_size {
            remote_start_command_array.push(command[i].clone());
        }
        commands.remote_start_command_array = Some(remote_start_command_array);
    }

    /// Java private final `buildIntermittentCommand`.
    fn build_intermittent_command(&self, commands: &mut Commands) {
        if utilities::is_windows_os() {
            commands.intermittent_command = Some("imodwincpu".to_string());
        } else {
            commands.intermittent_command = Some("w".to_string());
        }
    }

    /// Java private final `buildEndCommand`.
    fn build_end_command(&self, commands: &mut Commands) {
        commands.end_command = Some("exit".to_string());
    }
}

impl IntermittentCommand for LoadAverageParam {
    /// Java public final `getLocalStartCommand`.
    fn get_local_start_command(&self) -> Option<Vec<String>> {
        let mut commands = self.commands.lock().unwrap();
        if commands.local_start_command_array.is_none() {
            self.build_local_start_command(&mut commands);
        }
        commands.local_start_command_array.clone()
    }

    /// Java public final `getRemoteStartCommand`.
    fn get_remote_start_command(&self) -> Option<Vec<String>> {
        let mut commands = self.commands.lock().unwrap();
        if commands.remote_start_command_array.is_none() {
            self.build_remote_start_command(&mut commands);
        }
        commands.remote_start_command_array.clone()
    }

    /// Java `getIntermittentCommand`.
    fn get_intermittent_command(&self) -> Option<String> {
        let mut commands = self.commands.lock().unwrap();
        if commands.intermittent_command.is_none() {
            self.build_intermittent_command(&mut commands);
        }
        commands.intermittent_command.clone()
    }

    /// Java `getEndCommand`.
    fn get_end_command(&self) -> Option<String> {
        let mut commands = self.commands.lock().unwrap();
        if commands.end_command.is_none() {
            self.build_end_command(&mut commands);
        }
        commands.end_command.clone()
    }

    /// Java `getInterval`.
    fn get_interval(&self) -> i32 {
        5000
    }

    /// Java `notifySentIntermittentCommand`.
    fn notify_sent_intermittent_command(&self) -> bool {
        true
    }

    /// Java public final `getComputer`.
    fn get_computer(&self) -> Option<String> {
        Some(self.computer.clone())
    }
}
