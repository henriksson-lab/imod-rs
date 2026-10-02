//! `IMOD/Etomo/src/etomo/type/StringParameter.java`.
//!
//! Copyright: Copyright 2008 - 2022 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Class to update and read strings in a comscript.  The string is set to null when it
//! is empty.

use super::const_etomo_number::java_lang_string_matches_whitespace;
use super::const_string_parameter::ConstStringParameter;
use crate::imod::etomo::comscript::com_script_command::ComScriptCommand;
use crate::imod::etomo::comscript::invalid_parameter_exception::InvalidParameterException;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `StringParameter`.
#[derive(Clone, Debug)]
pub struct StringParameter {
    /// Java field `name`.
    name: String,
    /// Java field `value`, initialised to null.
    value: Option<String>,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

impl StringParameter {
    /// Java `StringParameter(String)`.
    pub fn new(name: &str) -> StringParameter {
        StringParameter {
            name: name.to_string(),
            value: None,
            debug: false,
        }
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.value = None;
    }

    /// Java `equals(String)`.
    pub fn equals(&self, comparee: Option<&str>) -> bool {
        match &self.value {
            None => StringParameter::is_empty_string(comparee),
            Some(value) => Some(value.as_str()) == comparee,
        }
    }

    /// Java `endsWith(String)`.
    ///
    /// Java `value.endsWith(null)` throws a NullPointerException
    /// (StringParameter.java:61).  Fixed in translation: a null comparee does not end a
    /// non-null value, so this returns false.
    pub fn ends_with(&self, comparee: Option<&str>) -> bool {
        match &self.value {
            None => StringParameter::is_empty_string(comparee),
            Some(value) => match comparee {
                None => false,
                Some(comparee) => value.ends_with(comparee),
            },
        }
    }

    /// Java `parse(ComScriptCommand)`.
    pub fn parse(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), InvalidParameterException> {
        if !script_command.has_keyword(Some(&self.name))? {
            self.reset();
        } else {
            self.value = script_command.get_value(Some(&self.name))?;
        }
        Ok(())
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        match &self.value {
            None => true,
            Some(value) => java_lang_string_matches_whitespace(value),
        }
    }

    /// Java private `isEmpty(String)`.
    fn is_empty_string(string: Option<&str>) -> bool {
        match string {
            None => true,
            Some(string) => java_lang_string_matches_whitespace(string),
        }
    }

    /// Java `updateComScript`.  If `isEmpty()`, delete name from scriptCommand.  Else
    /// set name and value in scriptCommand.
    pub fn update_com_script(&self, script_command: &mut ComScriptCommand) {
        if self.debug {
            script_command.set_debug(true);
        }
        if self.is_empty() {
            script_command.delete_key(Some(&self.name));
        } else {
            script_command.set_value(Some(&self.name), self.value.as_deref());
        }
    }

    /// Java `deleteFromComScript`.
    pub fn delete_from_com_script(&self, script_command: &mut ComScriptCommand) {
        if self.debug {
            script_command.set_debug(true);
        }
        script_command.delete_key(Some(&self.name));
    }

    /// Java `set(String)`.
    pub fn set(&mut self, input: Option<&str>) {
        if StringParameter::is_empty_string(input) {
            self.reset();
        } else {
            self.value = input.map(|input| input.to_string());
        }
    }

    /// Java `set(File)`.
    pub fn set_file(&mut self, input: Option<&std::path::Path>) {
        match input {
            None => self.reset(),
            Some(input) => {
                self.value = Some(java_io_file_get_absolute_path(&input.to_string_lossy()));
            }
        }
    }

    /// Java `getName`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
}

/// Java `toString`: the empty string when the value is null.
impl std::fmt::Display for StringParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            None => f.write_str(""),
            Some(value) => f.write_str(value),
        }
    }
}

/// Java `StringParameter implements ConstStringParameter`.
impl ConstStringParameter for StringParameter {
    fn equals(&self, input: Option<&str>) -> bool {
        StringParameter::equals(self, input)
    }

    fn ends_with(&self, input: Option<&str>) -> bool {
        StringParameter::ends_with(self, input)
    }

    fn is_empty(&self) -> bool {
        StringParameter::is_empty(self)
    }
}
