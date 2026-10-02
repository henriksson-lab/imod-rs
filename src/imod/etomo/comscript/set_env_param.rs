//! `IMOD/Etomo/src/etomo/comscript/SetEnvParam.java`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::r#type::imod_output_format::ImodOutputFormat;

/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "setenv";
/// Java private `NAME_INDEX`.
const NAME_INDEX: usize = 0;
/// Java private `VALUE_INDEX`.
const VALUE_INDEX: usize = 1;

/// Java final `SetEnvParam implements CommandParam`.
#[derive(Clone, Debug)]
pub struct SetEnvParam {
    /// Java final field `name`.  Fixed - command needs to match.
    name: Option<String>,
    /// Java field `valid`, initialised to true.
    valid: bool,
    /// Java field `value`, initialised to null.
    value: Option<String>,
    /// Java field `command` (a raw `List`), initialised to null.
    command: Option<Vec<Option<String>>>,
}

impl SetEnvParam {
    /// Java `SetEnvParam(String)`.
    pub fn new(name: Option<&str>) -> SetEnvParam {
        SetEnvParam {
            name: name.map(|name| name.to_string()),
            valid: true,
            value: None,
            command: None,
        }
    }

    /// Java `getCommandLine`.
    pub fn get_command_line(&mut self) -> String {
        if self.command.is_none() {
            self.build_command();
        }
        let command = self.command.as_ref().unwrap();
        let mut command_line = String::new();
        for i in 0..command.len() {
            command_line.push_str(&(command[i].as_deref().unwrap_or("null").to_string() + " "));
        }
        command_line
    }

    /// Java private `buildCommand`.
    fn build_command(&mut self) {
        match &mut self.command {
            None => self.command = Some(Vec::new()),
            Some(command) => command.clear(),
        }
        let command = self.command.as_mut().unwrap();
        command.push(Some(COMMAND_NAME.to_string()));
        command.push(self.name.clone());
        command.push(self.value.clone());
    }

    /// Java `getImodOutputFormatValue`.  `ImodOutputFormat.getInstance(null)` compares
    /// every singleton's value against null and returns null.
    pub fn get_imod_output_format_value(&self) -> Option<ImodOutputFormat> {
        match &self.value {
            None => None,
            Some(value) => ImodOutputFormat::get_instance(value),
        }
    }

    /// Java `setValue`.
    pub fn set_value(&mut self, value: Option<&str>) {
        self.value = value.map(|value| value.to_string());
    }
}

/// Java `toString`.
impl std::fmt::Display for SetEnvParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[name:{},value:{},valid:{}]",
            self.name.as_deref().unwrap_or("null"),
            self.value.as_deref().unwrap_or("null"),
            self.valid
        )
    }
}

impl CommandParam for SetEnvParam {
    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {
        self.valid = true;
        self.value = None;
    }

    /// Java `parseComScriptCommand`.  An empty argument list throws
    /// `ArrayIndexOutOfBoundsException`, which `ComScriptUtil.initialize` catches as an
    /// `Exception`; it is returned as an `InvalidParameterException` carrying the JVM's
    /// message.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // TODO error checking - throw exceptions for bad syntax
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        self.initialize_defaults();
        if cmd_line_args.len() <= NAME_INDEX {
            return Err(InvalidParameterException::new(Some(&format!(
                "Index {} out of bounds for length {}",
                NAME_INDEX,
                cmd_line_args.len()
            )))
            .into());
        }
        if cmd_line_args[NAME_INDEX].is_some() && cmd_line_args[NAME_INDEX] != self.name {
            self.valid = false;
        }
        if cmd_line_args.len() > VALUE_INDEX {
            self.value = cmd_line_args[VALUE_INDEX].clone();
        }
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let mut cmd_line_args: Vec<Option<String>> = Vec::with_capacity(VALUE_INDEX + 1);
        cmd_line_args.push(self.name.clone());
        cmd_line_args.push(self.value.clone());
        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }
}
