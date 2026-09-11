//! `IMOD/Etomo/src/etomo/util/EnvironmentVariable.java`.
//!
//! Description:
//!
//! Copyright: Copyright 2006 - 2024 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **`etomo.process.SystemProgram` is an external-process boundary.**  Both of the
//! class's methods run `env` (or `cmd.exe /C echo %VAR%` on Windows) as a *separate
//! process* and read its stdout - reading this process's own environment instead would
//! be a different program, because the source sees the environment of a freshly forked
//! `env`.  `etomo/process/SystemProgram.java` has no module of its own; what the two
//! methods below need of it is exactly `Runtime.getRuntime().exec(commandArray, null,
//! null)` plus the two `OutputBufferManager` threads that turn the child's stdout and
//! stderr into `String[]` of `BufferedReader.readLine()` lines
//! (`SystemProgram.java:300-345`).  With a null `BaseManager`, a null `commandAction`
//! and debug off, `run()` prints nothing of its own, so that is the whole observable
//! boundary and it is spawned here rather than emulated.
#![allow(dead_code)]

use super::utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Java `CALIB_DIR`.
pub const CALIB_DIR: &str = "IMOD_CALIB_DIR";
/// Java `PARTICLE_DIR`.
pub const PARTICLE_DIR: &str = "PARTICLE_DIR";

/// Java `EnvironmentVariable`.  The class is final and its only constructor is private,
/// so `INSTANCE` is the one instance that exists.
pub struct EnvironmentVariable {
    /// Java field `variableFoundList`, a `HashMap<String, Boolean>`.  A Java `HashMap`
    /// value may be null; nothing puts null into this one.
    variable_found_list: Mutex<HashMap<String, bool>>,
    /// Java field `variableList`, a `HashMap<String, String>`.  `getValue` distinguishes
    /// a present-but-null value from an absent key, so the value is an `Option`.
    variable_list: Mutex<HashMap<String, Option<String>>>,
}

/// Java `INSTANCE`.
pub static INSTANCE: LazyLock<EnvironmentVariable> = LazyLock::new(EnvironmentVariable::new);

impl EnvironmentVariable {
    /// Java `EnvironmentVariable()`, the private no-argument constructor.  The two
    /// `HashMap` fields are initialised at their declarations.
    fn new() -> EnvironmentVariable {
        EnvironmentVariable {
            variable_found_list: Mutex::new(HashMap::new()),
            variable_list: Mutex::new(HashMap::new()),
        }
    }

    /// Java `getValue(BaseManager, String, String, AxisID)`.
    ///
    /// Return an environment variable value.
    ///
    /// Java declares this `synchronized`; the two `Mutex` fields carry that here.
    pub fn get_value(
        &self,
        _manager: Option<&'static dyn BaseManager>,
        _property_user_dir: Option<&str>,
        var_name: &str,
        _axis_id: Option<AxisID>,
    ) -> String {
        let mut value = "".to_string();
        // prevent multiple reads and writes at the same time
        let mut variable_list = self.variable_list.lock().unwrap();
        if let Some(mapped) = variable_list.get(var_name) {
            match mapped {
                None => return "".to_string(),
                Some(mapped) => return mapped.clone(),
            }
        }
        // There is not a real good way to access the system environment variables
        // since the primary method was deprecated
        let read_env_var;
        if utilities::is_windows_os() {
            let var = "%".to_string() + var_name + "%";
            read_env_var = std::process::Command::new("cmd.exe")
                .args(["/C", "echo", &var])
                .output();
            let read_env_var = match read_env_var {
                Err(excep) => {
                    // `excep.printStackTrace()`; see etomo/util/stack_trace.rs.
                    eprintln!("{}", excep);
                    eprintln!("{}", excep);
                    eprintln!(
                        "Unable to run cmd command to find {} environment variable",
                        var_name
                    );
                    return "".to_string();
                }
                Ok(read_env_var) => read_env_var,
            };
            let stderr: Vec<String> = String::from_utf8_lossy(&read_env_var.stderr)
                .lines()
                .map(|line| line.to_string())
                .collect();
            if !stderr.is_empty() {
                eprintln!("Error running 'cmd.exe' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }
            // Return the first line from the command
            let stdout: Vec<String> = String::from_utf8_lossy(&read_env_var.stdout)
                .lines()
                .map(|line| line.to_string())
                .collect();
            if !stdout.is_empty() {
                // if the variable isn't set, echo will return the string sent to it
                if stdout[0] != var {
                    value = stdout[0].clone();
                }
            }
        }
        // Non windows environment
        else {
            read_env_var = std::process::Command::new("env").output();
            let read_env_var = match read_env_var {
                Err(excep) => {
                    // `excep.printStackTrace()`; see etomo/util/stack_trace.rs.
                    eprintln!("{}", excep);
                    eprintln!("{}", excep);
                    eprintln!(
                        "Unable to run env command to find {} environment variable",
                        var_name
                    );
                    return "".to_string();
                }
                Ok(read_env_var) => read_env_var,
            };
            let stderr: Vec<String> = String::from_utf8_lossy(&read_env_var.stderr)
                .lines()
                .map(|line| line.to_string())
                .collect();
            if !stderr.is_empty() {
                eprintln!("Error running 'env' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }

            // Search through the evironment string array to find the request
            // environment variable
            let search_string = var_name.to_string() + "=";
            let n_char = search_string.len();
            let stdout: Vec<String> = String::from_utf8_lossy(&read_env_var.stdout)
                .lines()
                .map(|line| line.to_string())
                .collect();
            for line in stdout.iter() {
                if line.find(&search_string) == Some(0) {
                    value = line[n_char..].to_string();
                    break;
                }
            }
        }
        variable_list.insert(var_name.to_string(), Some(value.clone()));
        value
    }

    /// Java `exists(BaseManager, String, String, AxisID)`.
    ///
    /// Return true if an environment variable value exists.  Doesn't use `variableList`
    /// because `getValue` doesn't distinguish between an empty env var and a
    /// non-existant one.  If unable to check, returns false.
    pub fn exists(
        &self,
        _manager: Option<&'static dyn BaseManager>,
        _property_user_dir: Option<&str>,
        var_name: &str,
        _axis_id: Option<AxisID>,
    ) -> bool {
        let mut variable_found_list = self.variable_found_list.lock().unwrap();
        if let Some(found) = variable_found_list.get(var_name) {
            return *found;
        }
        // There is not a real good way to access the system environment variables
        // since the primary method was deprecated
        let read_env_var;
        if utilities::is_windows_os() {
            let var = "%".to_string() + var_name + "%";
            read_env_var = std::process::Command::new("cmd.exe")
                .args(["/C", "echo", &var])
                .output();
            let read_env_var = match read_env_var {
                Err(excep) => {
                    // `excep.printStackTrace()`; see etomo/util/stack_trace.rs.
                    eprintln!("{}", excep);
                    eprintln!("{}", excep);
                    eprintln!(
                        "Unable to run cmd command to find {} environment variable",
                        var_name
                    );
                    variable_found_list.insert(var_name.to_string(), false);
                    return false;
                }
                Ok(read_env_var) => read_env_var,
            };
            let stderr: Vec<String> = String::from_utf8_lossy(&read_env_var.stderr)
                .lines()
                .map(|line| line.to_string())
                .collect();
            if !stderr.is_empty() {
                eprintln!("Error running 'cmd.exe' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }
            // Return the first line from the command
            let stdout: Vec<String> = String::from_utf8_lossy(&read_env_var.stdout)
                .lines()
                .map(|line| line.to_string())
                .collect();
            if !stdout.is_empty() {
                // if the variable isn't set, echo will return the string sent to it
                if stdout[0] != var {
                    variable_found_list.insert(var_name.to_string(), true);
                    return true;
                }
            }
        }
        // Non windows environment
        else {
            read_env_var = std::process::Command::new("env").output();
            let read_env_var = match read_env_var {
                Err(excep) => {
                    // `excep.printStackTrace()`; see etomo/util/stack_trace.rs.
                    eprintln!("{}", excep);
                    eprintln!("{}", excep);
                    eprintln!(
                        "Unable to run env command to find {} environment variable",
                        var_name
                    );
                    variable_found_list.insert(var_name.to_string(), false);
                    return false;
                }
                Ok(read_env_var) => read_env_var,
            };
            let stderr: Vec<String> = String::from_utf8_lossy(&read_env_var.stderr)
                .lines()
                .map(|line| line.to_string())
                .collect();
            if !stderr.is_empty() {
                eprintln!("Error running 'env' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }

            // Search through the evironment string array to find the request
            // environment variable
            let search_string = var_name.to_string() + "=";
            let stdout: Vec<String> = String::from_utf8_lossy(&read_env_var.stdout)
                .lines()
                .map(|line| line.to_string())
                .collect();
            for line in stdout.iter() {
                if line.find(&search_string) == Some(0) {
                    variable_found_list.insert(var_name.to_string(), true);
                    return true;
                }
            }
        }
        variable_found_list.insert(var_name.to_string(), false);
        false
    }
}
