//! `IMOD/Etomo/src/etomo/type/ScriptParameter.java`.
//!
//! Java's `ScriptParameter extends EtomoNumber`.  Rust has no inheritance, so - as
//! `etomo/type/etomo_number.rs` does for its own superclass - the superclass state is
//! held in the `base` field and reached through `Deref`/`DerefMut`; every inherited
//! member is therefore callable on a `ScriptParameter` exactly as in Java.
#![allow(dead_code)]

use std::collections::HashMap;

use super::const_etomo_number::{ConstEtomoNumber, Type};
use super::etomo_autodoc;
use super::etomo_boolean2::EtomoBoolean2;
use super::etomo_number::EtomoNumber;
use crate::imod::etomo::comscript::com_script_command::ComScriptCommand;
use crate::imod::etomo::comscript::invalid_parameter_exception::InvalidParameterException;

/// Java `ScriptParameter`.
#[derive(Clone, Debug)]
pub struct ScriptParameter {
    /// Java superclass `EtomoNumber` state.
    pub base: EtomoNumber,
    /// Java package-private field `shortName`, which defaults to null.
    pub(crate) short_name: Option<String>,
    /// Java private field `active`, which defaults to null.  An inactive value is not
    /// placed in the comscript.
    active: Option<Box<EtomoBoolean2>>,
}

/// Java inheritance: every `EtomoNumber` member is reachable on a `ScriptParameter`.
impl std::ops::Deref for ScriptParameter {
    type Target = EtomoNumber;

    fn deref(&self) -> &EtomoNumber {
        &self.base
    }
}

impl std::ops::DerefMut for ScriptParameter {
    fn deref_mut(&mut self) -> &mut EtomoNumber {
        &mut self.base
    }
}

impl ScriptParameter {
    /// Java `ScriptParameter(String)`.  Construct a ScriptParameter with
    /// type = INTEGER_TYPE; the parameter is the name of the instance.
    pub fn new_with_name(name: &str) -> ScriptParameter {
        ScriptParameter {
            base: EtomoNumber::new_with_name(name),
            short_name: None,
            active: None,
        }
    }

    /// Java `ScriptParameter(Type)`.
    pub fn new_with_type(r#type: Option<Type>) -> ScriptParameter {
        ScriptParameter {
            base: EtomoNumber::new_with_type(r#type),
            short_name: None,
            active: None,
        }
    }

    /// Java `ScriptParameter(Type, String)`.
    pub fn new_with_type_and_name(r#type: Type, name: &str) -> ScriptParameter {
        ScriptParameter {
            base: EtomoNumber::new_with_type_and_name(r#type, name),
            short_name: None,
            active: None,
        }
    }

    /// Java `ScriptParameter(Type, String, HashMap)`.
    pub fn new_with_required_map(
        r#type: Type,
        name: &str,
        required_map: Option<&HashMap<String, String>>,
    ) -> ScriptParameter {
        let mut instance = ScriptParameter {
            base: EtomoNumber::new_with_type_and_name(r#type, name),
            short_name: None,
            active: None,
        };
        instance.set_required(required_map);
        instance
    }

    /// Java `ScriptParameter(Type, String, String)`.
    pub fn new_with_short_name(r#type: Type, name: &str, short_name: &str) -> ScriptParameter {
        ScriptParameter {
            base: EtomoNumber::new_with_type_and_name(r#type, name),
            short_name: Some(short_name.to_string()),
            active: None,
        }
    }

    /// Java `ScriptParameter(ConstEtomoNumber)`.
    pub fn new_from_instance(that: Option<&ConstEtomoNumber>) -> ScriptParameter {
        ScriptParameter {
            base: EtomoNumber::new_from_instance(that),
            short_name: None,
            active: None,
        }
    }

    /// Java package-private `paramString`.
    pub(crate) fn param_string(&self) -> String {
        self.base.base.param_string()
            + ",\nactive="
            + &match &self.active {
                None => "null".to_string(),
                Some(active) => active.to_string(),
            }
    }

    /// Java private `setRequired(HashMap)`.  Sets nullIsValid based on requiredMap.  It
    /// would be ok to make this public or override it.
    fn set_required(
        &mut self,
        required_map: Option<&HashMap<String, String>>,
    ) -> &mut ConstEtomoNumber {
        let required_map = match required_map {
            None => return &mut self.base.base,
            Some(required_map) => required_map,
        };
        let mut required = EtomoNumber::new();
        required.set_string(required_map.get(&self.base.base.name).map(|x| x.as_str()));
        if required.base.equals_int(etomo_autodoc::REQUIRED_TRUE_VALUE) {
            self.base.base.null_is_valid = false;
        }
        &mut self.base.base
    }

    /// Java `updateComScript(ComScriptCommand)`.
    pub fn update_com_script(&self, script_command: &mut ComScriptCommand) {
        self.update_com_script_when_defaulted(script_command, false);
    }

    /// Java `updateComScript(ComScriptCommand, boolean)`.  Remove currentValue from or
    /// place currentValue in the comscript command depending on whether this instance is
    /// active, null, or defaulted.  An inactive value is always removed.  If
    /// includeWhenDefaulted is false, then the value is removed when it is null or
    /// defaulted.  If includeWhenDefaulted is true, then the value is remove when it is
    /// null.
    pub fn update_com_script_when_defaulted(
        &self,
        script_command: &mut ComScriptCommand,
        include_when_defaulted: bool,
    ) {
        if self.is_active()
            && ((include_when_defaulted && !self.base.base.is_null())
                || (!include_when_defaulted && self.is_not_null_and_not_default()))
        {
            if self.base.base.r#type != Type::Boolean || self.base.base.is_display_as_number() {
                let value = self.base.base.get_value();
                script_command.set_value(
                    Some(&self.base.base.name),
                    Some(&self.base.base.to_string_number(Some(value))),
                );
            } else if self.base.base.is() {
                // When not displayAsNumber, boolean parameters are included without a
                // value when they are true
                script_command.set_value(Some(&self.base.base.name), Some(""));
            } else {
                // When not displayAsNumber, boolean parameters are removed when they are
                // false
                script_command.delete_key(Some(&self.base.base.name));
            }
        } else {
            script_command.delete_key(Some(&self.base.base.name));
        }
    }

    /// Java `deleteFromComScript`.
    pub fn delete_from_com_script(&self, script_command: &mut ComScriptCommand) {
        script_command.delete_key(Some(&self.base.base.name));
    }

    /// Java `isNotNullAndNotDefault`.  Returns true if value is null or default.
    pub fn is_not_null_and_not_default(&self) -> bool {
        let value = self.base.base.get_value();
        if !self.base.base.is_null_number(Some(value))
            && !self.base.base.is_default_number(Some(value))
        {
            return true;
        }
        false
    }

    /// Java `parse(ComScriptCommand)`.
    pub fn parse(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<&mut ConstEtomoNumber, InvalidParameterException> {
        self.parse_set_active(script_command, false)
    }

    /// Java `parse(ComScriptCommand, boolean)`.  Parse scriptCommand for name and
    /// shortName.  If keyword is not found, call reset().  If name or shortName is
    /// found, call set with the string value found in scriptCommand.
    pub fn parse_set_active(
        &mut self,
        script_command: &ComScriptCommand,
        set_active: bool,
    ) -> Result<&mut ConstEtomoNumber, InvalidParameterException> {
        let mut found = false;
        let name = self.base.base.name.clone();
        if !script_command.has_keyword(Some(&name))? {
            let short_name = self.short_name.clone();
            if short_name.is_none() || !script_command.has_keyword(short_name.as_deref())? {
                if self.base.base.r#type == Type::Boolean && !self.base.base.is_display_as_number()
                {
                    // Missing boolean parameters equal false unless the boolean is
                    // displayed as a number
                    self.base.set_int(0);
                } else {
                    self.base.reset();
                }
            } else {
                found = true;
                let value = script_command.get_value(short_name.as_deref())?;
                self.base.set_string(value.as_deref());
            }
        } else {
            found = true;
            let value = script_command.get_value(Some(&name))?;
            self.base.set_string(value.as_deref());
        }
        if found
            && self.base.base.r#type == Type::Boolean
            && !self.base.base.is_display_as_number()
            && self.base.base.is_null()
        {
            // Boolean parameters do not have a value unless the boolean is displayed as
            // a number
            self.base.set_int(1);
        }
        if set_active {
            self.set_active(found);
        }
        Ok(&mut self.base.base)
    }

    /// Java `setActive`.
    pub fn set_active(&mut self, active: bool) {
        if self.active.is_none() {
            self.active = Some(Box::new(EtomoBoolean2::new()));
        }
        self.active.as_mut().unwrap().set_boolean(active);
    }

    /// Java `isActive`.
    pub fn is_active(&self) -> bool {
        match &self.active {
            None => true,
            Some(active) => active.base.base.base.is(),
        }
    }
}

/// Java `toString()`, inherited from `ConstEtomoNumber`.
impl std::fmt::Display for ScriptParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.base.base.get_value();
        f.write_str(&self.base.base.to_string_number(Some(value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `paramString` is package-private in Java, so the reference harness that captured
    /// these lines ran inside `etomo.type`; `pub(crate)` puts it out of reach of an
    /// integration test, which is why this check lives in the module.
    #[test]
    fn jvm_verified_param_string() {
        let mut sp = ScriptParameter::new_with_type_and_name(Type::Integer, "thickness");
        assert_eq!(
            sp.param_string(),
            ",\ntype=Integer,\nname=thickness,\ndescription=thickness,\ninvalidReason=null,\
             \ncurrentValue=-2147483648,\ndisplayValue=-2147483648,\
             \nceilingValue=-2147483648,\nfloorValue=-2147483648,\nnullIsValid=true,\
             \nvalidValues=null,\nactive=null"
        );
        sp.base.set_int(100);
        sp.set_active(false);
        assert_eq!(
            sp.param_string(),
            ",\ntype=Integer,\nname=thickness,\ndescription=thickness,\ninvalidReason=null,\
             \ncurrentValue=100,\ndisplayValue=-2147483648,\nceilingValue=-2147483648,\
             \nfloorValue=-2147483648,\nnullIsValid=true,\nvalidValues=null,\nactive=false"
        );
        let x = EtomoNumber::new_with_name("x");
        let sp5 = ScriptParameter::new_from_instance(Some(&x.base));
        assert_eq!(
            sp5.param_string(),
            ",\ntype=Integer,\nname=x,\ndescription=x,\ninvalidReason=null,\
             \ncurrentValue=-2147483648,\ndisplayValue=-2147483648,\
             \nceilingValue=-2147483648,\nfloorValue=-2147483648,\nnullIsValid=true,\
             \nvalidValues=null,\nactive=null"
        );
    }
}
