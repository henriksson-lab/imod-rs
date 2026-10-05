//! `IMOD/Etomo/src/etomo/util/EnvironmentVariable.java`.
//!
//! Description:
//!
//! Copyright: Copyright 2006 - 2024 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Both methods run `env` (or `cmd.exe /C echo %VAR%` on Windows) through a
//! `SystemProgram` and read its standard output, as the source does: the values come
//! from the environment of a freshly forked `env`, not from this process.
//! `SystemProgram.run` reports a failure to start through its exit value rather than
//! by throwing, so the source's `catch (Exception)` arms cannot be reached.
#![allow(dead_code)]

use super::utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::system_program::SystemProgram;
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
    /// `SystemProgram` never reads its axis, so a null axis is passed as `ONLY`.
    pub fn get_value(
        &self,
        manager: Option<&'static dyn BaseManager>,
        property_user_dir: Option<&str>,
        var_name: &str,
        axis_id: Option<AxisID>,
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
            read_env_var = SystemProgram::new_array(
                manager,
                property_user_dir.map(str::to_owned),
                Some(vec![
                    "cmd.exe".to_owned(),
                    "/C".to_owned(),
                    "echo".to_owned(),
                    var.clone(),
                ]),
                axis_id.unwrap_or(AxisID::Only),
            );
            read_env_var.run();
            let stderr = read_env_var.get_std_error();
            if let Some(stderr) = stderr.filter(|stderr| !stderr.is_empty()) {
                eprintln!("Error running 'cmd.exe' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }
            // Return the first line from the command
            let stdout = read_env_var.get_std_output();
            if let Some(stdout) = stdout.filter(|stdout| !stdout.is_empty()) {
                // if the variable isn't set, echo will return the string sent to it
                if stdout[0] != var {
                    value = stdout[0].clone();
                }
            }
        }
        // Non windows environment
        else {
            read_env_var = SystemProgram::new_array(
                manager,
                property_user_dir.map(str::to_owned),
                Some(vec!["env".to_owned()]),
                axis_id.unwrap_or(AxisID::Only),
            );
            read_env_var.run();
            let stderr = read_env_var.get_std_error().unwrap_or_default();
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
            let stdout = read_env_var.get_std_output().unwrap_or_default();
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
        manager: Option<&'static dyn BaseManager>,
        property_user_dir: Option<&str>,
        var_name: &str,
        axis_id: Option<AxisID>,
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
            read_env_var = SystemProgram::new_array(
                manager,
                property_user_dir.map(str::to_owned),
                Some(vec![
                    "cmd.exe".to_owned(),
                    "/C".to_owned(),
                    "echo".to_owned(),
                    var.clone(),
                ]),
                axis_id.unwrap_or(AxisID::Only),
            );
            read_env_var.run();
            let stderr = read_env_var.get_std_error();
            if let Some(stderr) = stderr.filter(|stderr| !stderr.is_empty()) {
                eprintln!("Error running 'cmd.exe' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }
            // Return the first line from the command
            let stdout = read_env_var.get_std_output();
            if let Some(stdout) = stdout.filter(|stdout| !stdout.is_empty()) {
                // if the variable isn't set, echo will return the string sent to it
                if stdout[0] != var {
                    variable_found_list.insert(var_name.to_string(), true);
                    return true;
                }
            }
        }
        // Non windows environment
        else {
            read_env_var = SystemProgram::new_array(
                manager,
                property_user_dir.map(str::to_owned),
                Some(vec!["env".to_owned()]),
                axis_id.unwrap_or(AxisID::Only),
            );
            read_env_var.run();
            let stderr = read_env_var.get_std_error().unwrap_or_default();
            if !stderr.is_empty() {
                eprintln!("Error running 'env' command");
                for line in stderr.iter() {
                    eprintln!("{}", line);
                }
            }

            // Search through the evironment string array to find the request
            // environment variable
            let search_string = var_name.to_string() + "=";
            let stdout = read_env_var.get_std_output().unwrap_or_default();
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
