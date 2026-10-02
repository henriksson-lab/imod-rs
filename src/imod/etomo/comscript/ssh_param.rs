//! `IMOD/Etomo/src/etomo/comscript/SshParam.java`.
//!
//! Builds the `ssh` command line used to reach another computer, and decides
//! once per process whether the installed OpenSSH understands `ConnectTimeout`.

use std::sync::{LazyLock, Mutex};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_version::EtomoVersion;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private static final `INSTANCE`.
pub static INSTANCE: LazyLock<SshParam> = LazyLock::new(SshParam::new);

/// Java package-private final `SshParam`.
pub struct SshParam {
    /// Java private field `timeoutAvailable`, initialised to null.  The `Mutex`
    /// carries the `synchronized` on `isTimeoutAvailable`.
    timeout_available: Mutex<Option<EtomoBoolean2>>,
}

impl SshParam {
    /// Java's implicit constructor.
    fn new() -> SshParam {
        SshParam {
            timeout_available: Mutex::new(None),
        }
    }

    /// Java package-private final `getCommand(BaseManager, boolean, String)`.
    pub fn get_command(
        &self,
        manager: &'static dyn BaseManager,
        use_timeout_if_possible: bool,
        computer: Option<&str>,
    ) -> Vec<String> {
        let mut command = Vec::new();
        command.push("ssh".to_string());
        command.push("-x".to_string());
        // prevents ssh from waiting for an answer when connecting to a computer for
        // the first time
        // see man ssh_config
        command.push("-o".to_string());
        command.push("StrictHostKeyChecking=no".to_string());
        if use_timeout_if_possible && self.is_timeout_available(manager) {
            // Timeout doesn't work with older versions of Redhat (see bug# 1043).
            // maximum connection timeout for a down computer
            command.push("-o".to_string());
            command.push("ConnectTimeout=5".to_string());
        }
        command.push("-o".to_string());
        // prevents password prompts when the publickey authentication fails
        command.push("PreferredAuthentications=publickey".to_string());
        command.push("-v".to_string());
        // Java adds a null computer as a null element; `ProcessBuilder` would then
        // reject the command.  "null" is what Java string use of it prints.
        command.push(computer.unwrap_or("null").to_string());
        command
    }

    /// Java package-private synchronized `isTimeoutAvailable(BaseManager)`.
    /// Sets timeoutAvailable based on the result of running "ssh -v".  Only sets
    /// timeoutAvailable once.  If running ssh -v fails, timeoutAvailable is set to
    /// false.
    pub fn is_timeout_available(&self, manager: &'static dyn BaseManager) -> bool {
        let mut timeout_available = self.timeout_available.lock().unwrap();
        if let Some(timeout_available) = timeout_available.as_ref() {
            return timeout_available.is();
        }
        // Set timeoutAvailable from "ssh -v". OpenSSH with a version of 3.9 or
        // greater will understand the ConnectTimeout option.
        let timeout_available = timeout_available.insert(EtomoBoolean2::new());
        // Run ssh -v.  `System.getProperty("user.dir")`: `PWD` is the variable this
        // translation keeps it in (see `BaseManager::make_property_user_dir_local`).
        let system_program = SystemProgram::new_array(
            Some(manager),
            std::env::var("PWD").ok(),
            Some(vec!["ssh".to_string(), "-V".to_string()]),
            AxisID::Only,
        );
        system_program.run();
        // Find and parse the OpenSSH version.
        eprintln!("For isTimeoutAvailable, ssh -V response:");
        system_program.print_std_error();
        system_program.print_std_output();
        let stderr = system_program.get_std_error();
        if let Some(stderr) = stderr
            && !stderr.is_empty()
        {
            let app_string = "openssh";
            let mut found = false;
            let mut i = 0;
            while i < stderr.len() {
                if stderr[i].to_lowercase().contains(app_string) {
                    found = true;
                    break;
                }
                i += 1;
            }
            if found {
                // Find and store the version of OpenSSH (OpenSSH_version, ...).
                // Java `split("[_,\\s]+")` drops trailing empty strings; a leading
                // empty string only occurs for a leading separator, which `trim()`
                // has removed unless it is '_' or ','.
                let lowered = stderr[i].to_lowercase();
                let trimmed = lowered.trim_matches(|c: char| c <= ' ');
                let mut version_info_array: Vec<&str> = trimmed
                    .split(|c: char| {
                        matches!(c, '_' | ',' | ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r')
                    })
                    .collect();
                // Merge the empty fields runs of separators leave (`[...]+`), keeping a
                // leading empty field as Java does.
                let mut merged: Vec<&str> = Vec::new();
                for (index, element) in version_info_array.drain(..).enumerate() {
                    if element.is_empty() && index > 0 {
                        continue;
                    }
                    merged.push(element);
                }
                while merged.len() > 1 && merged.last().is_some_and(|last| last.is_empty()) {
                    merged.pop();
                }
                let version_info_array = merged;
                if !version_info_array.is_empty() {
                    i = 0;
                    // SshParam.java:95 reads `versionInfoArray.length < i + 1 && ...`,
                    // which is false on entry for any non-empty array, so the loop never
                    // runs and the program name itself ("openssh") is parsed as the
                    // version - `ConnectTimeout` was never used.  Fixed in translation:
                    // the loop advances past the element naming OpenSSH, as its comment
                    // and the following `i < length` test intend.
                    loop {
                        if !(version_info_array.len() > i + 1) {
                            break;
                        }
                        let element = version_info_array[i];
                        i += 1;
                        if element.to_lowercase().contains(app_string) {
                            break;
                        }
                    }
                    if i < version_info_array.len() {
                        let open_ssh_version = EtomoVersion::get_default_instance_with_version(
                            Some(version_info_array[i]),
                        );
                        if open_ssh_version.ge(Some(
                            &EtomoVersion::get_default_instance_with_version(Some("3.9")),
                        )) {
                            timeout_available.set_boolean(true);
                            return true;
                        }
                    }
                }
            }
        }
        timeout_available.set_boolean(false);
        false
    }
}
