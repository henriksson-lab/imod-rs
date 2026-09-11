//! `IMOD/Etomo/src/etomo/type/EtomoBoolean2.java`.
//!
//! Java's `EtomoBoolean2 extends ScriptParameter extends EtomoNumber extends
//! ConstEtomoNumber`.  Rust has no inheritance, so - as `etomo/type/etomo_number.rs`
//! and `etomo/type/script_parameter.rs` do - the superclass state is held in the `base`
//! field and reached through `Deref`/`DerefMut`.
//!
//! **Virtual dispatch.**  This is the first subclass in the `ConstEtomoNumber`
//! hierarchy that overrides methods the *superclass bodies* call: `newNumber(String,
//! StringBuffer)` (reached from `EtomoNumber.java:331` and from
//! `ConstEtomoNumber.java:912`, `:920`, `:928`, `:1009`), `setInvalidReason()` (reached
//! from `EtomoNumber.java:340`, `:353`, `:447` and `ConstEtomoNumber.java:605`, `:629`,
//! `:643`) and `toString(Number)`.  A `Deref` to the superclass struct cannot dispatch
//! back down, so the four inherited setters this class's own bodies call -
//! `set(String)`, `set(int)`, `set(boolean)` and `set(ConstEtomoNumber)` - are written
//! out here as the inherited body with those virtual calls resolved to this class's
//! overrides, each marked in place.  So are `reset()`, `setValidValues`,
//! `setNullIsValid` and `setValidFloor`, which reach `setInvalidReason()`, and `is()`
//! and `getDefaultedValue()`, which reach the `isNull()` override, and `store` (both
//! overloads) and `toDefaultedString()`, which reach the `toString(Number)` one.  That is a deviation in form, not in behaviour: a
//! caller holding an `EtomoBoolean2` reaches exactly the methods Java's dispatch
//! reaches, because Rust resolves an inherent method before a `Deref`ed one.  Reaching
//! the superclass setter through `.base` instead skips the overrides, as calling
//! `super.set(...)` would in Java.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use super::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_string_matches_whitespace,
};
use super::script_parameter::ScriptParameter;
use crate::imod::etomo::comscript::com_script_command::ComScriptCommand;
use crate::imod::etomo::comscript::invalid_parameter_exception::InvalidParameterException;

/// Java `DEFAULT_FALSE_VALUE`.
pub const DEFAULT_FALSE_VALUE: i32 = 0;
/// Java `DEFAULT_TRUE_VALUE`.
pub const DEFAULT_TRUE_VALUE: i32 = 1;

/// Java private static `defaultFalseString`.
const DEFAULT_FALSE_STRING: &str = "false";
/// Java private static `defaultFalseStrings`.
const DEFAULT_FALSE_STRINGS: &[&str] = &["f", "no"];
/// Java private static `defaultTrueString`.
const DEFAULT_TRUE_STRING: &str = "true";
/// Java private static `defaultTrueStrings`.
const DEFAULT_TRUE_STRINGS: &[&str] = &["t", "yes"];

/// Java `EtomoBoolean2`.
#[derive(Clone, Debug)]
pub struct EtomoBoolean2 {
    /// Java superclass `ScriptParameter` state.
    pub base: ScriptParameter,
    /// Java private field `falseValue`, which defaults to `DEFAULT_FALSE_VALUE`.
    false_value: i32,
    /// Java private field `trueValue`, which defaults to `DEFAULT_TRUE_VALUE`.
    true_value: i32,
    /// Java private field `falseString`, which defaults to `defaultFalseString`.
    false_string: Option<String>,
    /// Java private field `trueString`, which defaults to `defaultTrueString`.
    true_string: Option<String>,
    /// Java private field `falseStrings`, which defaults to `defaultFalseStrings`.
    false_strings: Option<Vec<String>>,
    /// Java private field `trueStrings`, which defaults to `defaultTrueStrings`.
    true_strings: Option<Vec<String>>,
    /// Java private field `displayAsInteger`, which defaults to false.
    display_as_integer: bool,
}

/// Java inheritance: every `ScriptParameter` member is reachable on an `EtomoBoolean2`.
impl std::ops::Deref for EtomoBoolean2 {
    type Target = ScriptParameter;

    fn deref(&self) -> &ScriptParameter {
        &self.base
    }
}

impl std::ops::DerefMut for EtomoBoolean2 {
    fn deref_mut(&mut self) -> &mut ScriptParameter {
        &mut self.base
    }
}

impl EtomoBoolean2 {
    /// Java `EtomoBoolean2()`.
    pub fn new() -> EtomoBoolean2 {
        let mut instance = EtomoBoolean2 {
            base: ScriptParameter::new_with_type(Some(Type::Integer)),
            false_value: DEFAULT_FALSE_VALUE,
            true_value: DEFAULT_TRUE_VALUE,
            false_string: Some(DEFAULT_FALSE_STRING.to_string()),
            true_string: Some(DEFAULT_TRUE_STRING.to_string()),
            false_strings: Some(
                DEFAULT_FALSE_STRINGS
                    .iter()
                    .map(|x| x.to_string())
                    .collect(),
            ),
            true_strings: Some(DEFAULT_TRUE_STRINGS.iter().map(|x| x.to_string()).collect()),
            display_as_integer: false,
        };
        instance.set_valid_values(Some(&[instance.false_value, instance.true_value]));
        instance
            .base
            .base
            .base
            .set_display_value_int(instance.false_value);
        instance
    }

    /// Java `EtomoBoolean2(String)`.
    pub fn new_with_name(name: &str) -> EtomoBoolean2 {
        let mut instance = EtomoBoolean2 {
            base: ScriptParameter::new_with_type_and_name(Type::Integer, name),
            false_value: DEFAULT_FALSE_VALUE,
            true_value: DEFAULT_TRUE_VALUE,
            false_string: Some(DEFAULT_FALSE_STRING.to_string()),
            true_string: Some(DEFAULT_TRUE_STRING.to_string()),
            false_strings: Some(
                DEFAULT_FALSE_STRINGS
                    .iter()
                    .map(|x| x.to_string())
                    .collect(),
            ),
            true_strings: Some(DEFAULT_TRUE_STRINGS.iter().map(|x| x.to_string()).collect()),
            display_as_integer: false,
        };
        instance.set_valid_values(Some(&[instance.false_value, instance.true_value]));
        instance
            .base
            .base
            .base
            .set_display_value_int(instance.false_value);
        instance
    }

    /// Java `EtomoBoolean2(String, HashMap)`.
    pub fn new_with_required_map(
        name: &str,
        required_map: Option<&HashMap<String, String>>,
    ) -> EtomoBoolean2 {
        let mut instance = EtomoBoolean2 {
            base: ScriptParameter::new_with_required_map(Type::Integer, name, required_map),
            false_value: DEFAULT_FALSE_VALUE,
            true_value: DEFAULT_TRUE_VALUE,
            false_string: Some(DEFAULT_FALSE_STRING.to_string()),
            true_string: Some(DEFAULT_TRUE_STRING.to_string()),
            false_strings: Some(
                DEFAULT_FALSE_STRINGS
                    .iter()
                    .map(|x| x.to_string())
                    .collect(),
            ),
            true_strings: Some(DEFAULT_TRUE_STRINGS.iter().map(|x| x.to_string()).collect()),
            display_as_integer: false,
        };
        instance.set_valid_values(Some(&[instance.false_value, instance.true_value]));
        instance
            .base
            .base
            .base
            .set_display_value_int(instance.false_value);
        instance
    }

    /// Java `EtomoBoolean2(String, int, int)`.
    pub fn new_with_on_off_values(name: &str, on_value: i32, off_value: i32) -> EtomoBoolean2 {
        let mut instance = EtomoBoolean2 {
            base: ScriptParameter::new_with_type_and_name(Type::Integer, name),
            false_value: DEFAULT_FALSE_VALUE,
            true_value: DEFAULT_TRUE_VALUE,
            false_string: Some(DEFAULT_FALSE_STRING.to_string()),
            true_string: Some(DEFAULT_TRUE_STRING.to_string()),
            false_strings: Some(
                DEFAULT_FALSE_STRINGS
                    .iter()
                    .map(|x| x.to_string())
                    .collect(),
            ),
            true_strings: Some(DEFAULT_TRUE_STRINGS.iter().map(|x| x.to_string()).collect()),
            display_as_integer: false,
        };
        instance.true_value = on_value;
        instance.false_value = off_value;
        instance.false_string = None;
        instance.true_string = None;
        instance.false_strings = None;
        instance.true_strings = None;
        instance.set_valid_values(Some(&[instance.false_value, instance.true_value]));
        instance
            .base
            .base
            .base
            .set_display_value_int(instance.false_value);
        instance
    }

    /// Java static `store(EtomoBoolean2, Properties, String, String)`.
    pub fn store_instance(
        instance: Option<&EtomoBoolean2>,
        props: &mut BTreeMap<String, String>,
        prepend: Option<&str>,
        name: &str,
    ) {
        match instance {
            None => {
                // Java `props.remove(prepend + "." + name)`, whose concatenation prints a
                // null prepend as "null".
                props.remove(&format!("{}.{}", prepend.unwrap_or("null"), name));
            }
            Some(instance) => {
                instance.store_with_prepend(props, prepend);
            }
        }
    }

    /// Java static `load(EtomoBoolean2, String, Properties, String)`.  Attempt to get
    /// the property specified by prepend and name.  If it doesn't exist, return null.
    /// If it does, set it in instance (create instance if it doesn't exist).  Return the
    /// instance.
    pub fn load_instance(
        instance: Option<EtomoBoolean2>,
        name: &str,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> Option<EtomoBoolean2> {
        let key;
        let prepend_blank = match prepend {
            None => true,
            Some(prepend) => java_lang_string_matches_whitespace(prepend),
        };
        if prepend_blank {
            key = name.to_string();
        } else if prepend.unwrap().ends_with('.') {
            key = format!("{}{}", prepend.unwrap(), name);
        } else {
            key = format!("{}.{}", prepend.unwrap(), name);
        }
        let value = props.get(&key);
        let value = match value {
            None => return None,
            Some(value) => value.clone(),
        };
        let mut instance = match instance {
            None => EtomoBoolean2::new_with_name(name),
            Some(instance) => instance,
        };
        instance.set_string(Some(&value));
        Some(instance)
    }

    /// Java static `remove(String, Properties, String)`.
    pub fn remove(name: &str, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let key;
        let prepend_blank = match prepend {
            None => true,
            Some(prepend) => java_lang_string_matches_whitespace(prepend),
        };
        if prepend_blank {
            key = name.to_string();
        } else if prepend.unwrap().ends_with('.') {
            key = format!("{}{}", prepend.unwrap(), name);
        } else {
            key = format!("{}.{}", prepend.unwrap(), name);
        }
        props.remove(&key);
    }

    /// Java static `equals(EtomoBoolean2, EtomoBoolean2)`.
    pub fn equals_instances(
        instance1: Option<&EtomoBoolean2>,
        instance2: Option<&EtomoBoolean2>,
    ) -> bool {
        // Java's `instance1 == instance2` is reference identity, which is true for two
        // nulls and for the same object; `std::ptr::eq` is the same test.
        match (instance1, instance2) {
            (None, None) => return true,
            (Some(instance1), Some(instance2)) if std::ptr::eq(instance1, instance2) => {
                return true;
            }
            _ => {}
        }
        let instance1 = match instance1 {
            None => return false,
            Some(instance1) => instance1,
        };
        instance1
            .base
            .base
            .base
            .equals_const_etomo_number(instance2.map(|instance2| &instance2.base.base.base))
    }

    /// Java static `set(EtomoBoolean2, ConstEtomoNumber, String)`.
    pub fn set_instance_const_etomo_number(
        instance: Option<EtomoBoolean2>,
        value: Option<&ConstEtomoNumber>,
        name: &str,
    ) -> Option<EtomoBoolean2> {
        let mut instance = instance;
        if instance.is_none() && value.is_some() {
            instance = Some(EtomoBoolean2::new_with_name(name));
        }
        let mut instance = match instance {
            None => return None,
            Some(instance) => instance,
        };
        instance.set_const_etomo_number(value);
        Some(instance)
    }

    /// Java static `set(EtomoBoolean2, boolean, String)`.
    pub fn set_instance_boolean(
        instance: Option<EtomoBoolean2>,
        value: bool,
        name: &str,
    ) -> Option<EtomoBoolean2> {
        let mut instance = match instance {
            None => EtomoBoolean2::new_with_name(name),
            Some(instance) => instance,
        };
        instance.set_boolean(value);
        Some(instance)
    }

    /// Java static `getInstance(EtomoBoolean2, String, Properties, String)`.  If
    /// instance exists, load it and return it.  If instance doesn't exist, only create
    /// and return it if the value in props exists.  If there is no value in props with
    /// the key prepend + '.' + key, then return null.
    pub fn get_instance_from_props(
        instance: Option<EtomoBoolean2>,
        key: &str,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> Option<EtomoBoolean2> {
        if let Some(mut instance) = instance {
            instance.load_with_prepend(props, prepend);
            return Some(instance);
        }
        let value = props.get(&format!("{}.{}", prepend.unwrap_or("null"), key));
        let value = match value {
            None => return None,
            Some(value) => value.clone(),
        };
        // Java's `if (instance == null)` is always true here: the non-null case returned
        // above.
        let mut instance = EtomoBoolean2::new_with_name(key);
        instance.set_string(Some(&value));
        Some(instance)
    }

    /// Java static `getInstance(EtomoBoolean2, String, boolean)`.
    pub fn get_instance_from_boolean(
        instance: Option<EtomoBoolean2>,
        key: &str,
        value: bool,
    ) -> Option<EtomoBoolean2> {
        let mut instance = match instance {
            None => EtomoBoolean2::new_with_name(key),
            Some(instance) => instance,
        };
        instance.set_boolean(value);
        Some(instance)
    }

    /// Java `isNull()`, overriding `ConstEtomoNumber.isNull()` to prevent EtomoBoolean2
    /// from being null.  Checks the display value, which is always set, when checking
    /// for null.  So it never tests as null.  This could be overriden by setting the
    /// display value to null.
    pub fn is_null(&self) -> bool {
        let value = self.base.base.base.get_value();
        self.base.base.base.is_null_number(Some(value))
    }

    /// Java package-private `setInvalidReason()`, overriding
    /// `ConstEtomoNumber.setInvalidReason()`.  To prevent values not in validValues from
    /// being used: call super.setInvalidReason(), then throw an exception when
    /// invalidReason is set.
    ///
    /// `super.setInvalidReason()` is written out here rather than delegated, because its
    /// body calls `toString(Number)` - directly, and through the private
    /// `toString(Vector)` - and those calls dispatch to this class's override.  See the
    /// module header.
    pub(crate) fn set_invalid_reason(&mut self) {
        // `super.setInvalidReason()`; the source's `return`s leave that body, not this
        // method, so the throw below still runs.
        'super_set_invalid_reason: {
            // Pass when there are no validation settings
            if self.base.base.base.null_is_valid
                && self.base.base.base.valid_values.is_none()
                && self
                    .base
                    .base
                    .base
                    .is_null_number(Some(self.base.base.base.valid_floor))
            {
                break 'super_set_invalid_reason;
            }
            // Catch illegal null values
            if self
                .base
                .base
                .base
                .is_null_number(Some(self.base.base.base.current_value))
            {
                if self.base.base.base.null_is_valid {
                    break 'super_set_invalid_reason;
                }
                self.base
                    .base
                    .base
                    .add_invalid_reason(Some("This field cannot be empty."));
            }
            // Validate against validValues, overrides validFloor
            else if self.base.base.base.valid_values.is_some() {
                let valid_values = self.base.base.base.valid_values.clone().unwrap();
                for i in 0..valid_values.len() {
                    if self.base.base.base.equals_numbers(
                        Some(self.base.base.base.current_value),
                        Some(valid_values[i]),
                    ) {
                        break 'super_set_invalid_reason;
                    }
                }
                // Java concatenates a null `toString(Number)` as "null".
                let message = format!(
                    "{} is not a valid value.",
                    self.to_string_number(Some(self.base.base.base.current_value))
                        .unwrap_or("null".to_string())
                );
                self.base.base.base.add_invalid_reason(Some(&message));
                // `toString(Vector)` (ConstEtomoNumber.java:1124), whose
                // `new StringBuffer(toString(numberVector.get(0)))` throws a
                // NullPointerException when this class's override returns null - which it
                // does for an instance built by `EtomoBoolean2(String, int, int)`.
                let vector_string = if valid_values.is_empty() {
                    String::new()
                } else {
                    let mut buffer = match self.to_string_number(Some(valid_values[0])) {
                        None => panic!("Cannot invoke \"String.length()\" because \"str\" is null"),
                        Some(first) => first,
                    };
                    for i in 1..valid_values.len() {
                        buffer.push_str(&format!(
                            ",{}",
                            self.to_string_number(Some(valid_values[i]))
                                .unwrap_or("null".to_string())
                        ));
                    }
                    buffer
                };
                let message = format!("Valid values are {}.", vector_string);
                self.base.base.base.add_invalid_reason(Some(&message));
                break 'super_set_invalid_reason;
            }
            // If validValues is not set, validate against validFloor
            else if !self
                .base
                .base
                .base
                .is_null_number(Some(self.base.base.base.valid_floor))
            {
                if self.base.base.base.ge_numbers(
                    Some(self.base.base.base.current_value),
                    Some(self.base.base.base.valid_floor),
                ) {
                    break 'super_set_invalid_reason;
                }
                let message = format!(
                    "{} is not a valid value.",
                    self.to_string_number(Some(self.base.base.base.current_value))
                        .unwrap_or("null".to_string())
                );
                self.base.base.base.add_invalid_reason(Some(&message));
                let message = format!(
                    "Valid values are greater or equal to {}.",
                    self.to_string_number(Some(self.base.base.base.valid_floor))
                        .unwrap_or("null".to_string())
                );
                self.base.base.base.add_invalid_reason(Some(&message));
            }
        }
        if let Some(invalid_reason) = self.base.base.base.invalid_reason.clone() {
            // Java `throw new IllegalArgumentException(invalidReason.toString())`, an
            // unchecked exception no caller in the source catches.
            panic!("{}", invalid_reason);
        }
    }

    /// Java package-private `toString(Number)`, overriding
    /// `ConstEtomoNumber.toString(Number)`.  Return false if null or 0, otherwise return
    /// true.
    /// `trueString` and `falseString` are null for an instance built by
    /// `EtomoBoolean2(String, int, int)`, so the Java return type carries null; the
    /// translation returns `Option<String>` rather than flattening it to "".
    pub(crate) fn to_string_number(&self, value: Option<Number>) -> Option<String> {
        if self.display_as_integer {
            return Some(self.base.base.base.to_string_number(value));
        }
        let true_number = self.base.base.base.new_number_from_int(self.true_value);
        if self.base.base.base.equals_numbers(value, Some(true_number)) {
            return self.true_string.clone();
        }
        self.false_string.clone()
    }

    /// Java `setDisplayAsInteger`.
    pub fn set_display_as_integer(&mut self, display_as_integer: bool) -> &mut ConstEtomoNumber {
        if !display_as_integer && (self.true_string.is_none() || self.false_string.is_none()) {
            // Java `throw new IllegalStateException(...)`.
            panic!(
                "Must display {}as an integer, since it has no string equivalent.",
                self.base.base.base.name
            );
        }
        self.display_as_integer = display_as_integer;
        &mut self.base.base.base
    }

    /// Java `parse(ComScriptCommand)`, overriding `ScriptParameter.parse` to handle a
    /// boolean being true when it has no value in the script.
    pub fn parse(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<&mut ConstEtomoNumber, InvalidParameterException> {
        let name = self.base.base.base.name.clone();
        let name_in_script = script_command.has_keyword(Some(&name))?;
        let short_name = self.base.short_name.clone();
        if !name_in_script
            && (short_name.is_none() || !script_command.has_keyword(short_name.as_deref())?)
        {
            let false_value = self.false_value;
            return Ok(self.set_int(false_value));
        }
        let script_value;
        if name_in_script {
            script_value = script_command.get_value(Some(&name))?;
        } else {
            script_value = script_command.get_value(short_name.as_deref())?;
        }
        let blank = match &script_value {
            None => true,
            Some(script_value) => java_lang_string_matches_whitespace(script_value),
        };
        if blank {
            let true_value = self.true_value;
            return Ok(self.set_int(true_value));
        }
        Ok(self.set_string(script_value.as_deref()))
    }

    /// Java `updateComScript(ComScriptCommand)`, overriding
    /// `ScriptParameter.updateComScript` to handle writing the boolean as an integer in
    /// the script (use super.toString()).  Also handle writing the boolean as a
    /// parameter without a value.
    pub fn update_com_script(&self, script_command: &mut ComScriptCommand) {
        if !self.is_use_in_script() {
            return;
        }
        if !self.display_as_integer && !self.base.base.base.is() {
            script_command.delete_key(Some(&self.base.base.base.name));
        } else if self.display_as_integer {
            // Java `super.toString()`, which is `ConstEtomoNumber.toString()`, not this
            // class's `toString(Number)` override.
            let value = self.base.base.base.get_value();
            script_command.set_value(
                Some(&self.base.base.base.name),
                Some(&self.base.base.base.to_string_number(Some(value))),
            );
        } else {
            script_command.set_value(Some(&self.base.base.base.name), Some(""));
        }
    }

    /// Java `setOn`.
    pub fn set_on(&mut self) -> &mut ConstEtomoNumber {
        let true_value = self.true_value;
        self.set_int(true_value)
    }

    /// Java `setOff`.
    pub fn set_off(&mut self) -> &mut ConstEtomoNumber {
        let false_value = self.false_value;
        self.set_int(false_value)
    }

    /// Java `isUseInScript`.
    pub fn is_use_in_script(&self) -> bool {
        true
    }

    /// Java `equals(boolean)`.
    pub fn equals_boolean(&self, value: bool) -> bool {
        if value {
            return self.base.base.base.equals_int(self.true_value);
        }
        self.base.base.base.equals_int(self.false_value)
    }

    /// Java package-private `newNumber(String, StringBuffer)`, overriding
    /// `ConstEtomoNumber.newNumber(String, StringBuffer)`.  To convert from strings such
    /// as "false": convert from a trimmed, case adjusted character string to a value,
    /// then call super.newNumber(String, StringBuffer) to handle strings such as "0".
    pub(crate) fn new_number_from_string(
        &self,
        value: Option<&str>,
        invalid_buffer: &mut String,
    ) -> Number {
        // Convert from character string to integer value
        let trimmed_value = value.unwrap_or_default().trim().to_lowercase();
        if let Some(false_string) = &self.false_string {
            if &trimmed_value == false_string {
                return self.base.base.base.new_number_from_int(self.false_value);
            }
        }
        if let Some(true_string) = &self.true_string {
            if &trimmed_value == true_string {
                return self.base.base.base.new_number_from_int(self.true_value);
            }
        }
        if let Some(false_strings) = &self.false_strings {
            for false_string in false_strings.iter() {
                if &trimmed_value == false_string {
                    return self.base.base.base.new_number_from_int(self.false_value);
                }
            }
        }
        if let Some(true_strings) = &self.true_strings {
            for true_string in true_strings.iter() {
                if &trimmed_value == true_string {
                    return self.base.base.base.new_number_from_int(self.true_value);
                }
            }
        }
        self.base
            .base
            .base
            .new_number_from_string(value, invalid_buffer)
    }

    /// Java inherited `ConstEtomoNumber.store(Properties)`, whose virtual
    /// `toString(Number)` call resolves to this class's override - so a true instance
    /// stores "true", not "1".  See the module header.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        if self
            .base
            .base
            .base
            .is_null_number(Some(self.base.base.base.current_value))
        {
            self.base.base.base.remove(props);
            return;
        }
        props.insert(
            self.base.base.base.name.clone(),
            self.to_string_number(Some(self.base.base.base.current_value))
                .unwrap_or("null".to_string()),
        );
    }

    /// Java inherited `ConstEtomoNumber.store(Properties, String)`, whose virtual
    /// `toString(Number)` call resolves to this class's override.  See the module header.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        if self
            .base
            .base
            .base
            .is_null_number(Some(self.base.base.base.current_value))
        {
            self.base.base.base.remove_with_prepend(props, prepend);
            return;
        }
        match prepend {
            None => self.store(props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.store(props),
            Some(prepend) => {
                props.insert(
                    format!("{}.{}", prepend, self.base.base.base.name),
                    self.to_string_number(Some(self.base.base.base.current_value))
                        .unwrap_or("null".to_string()),
                );
            }
        }
    }

    /// Java inherited `ConstEtomoNumber.toDefaultedString()`, whose virtual
    /// `toString(Number)` call resolves to this class's override.  See the module header.
    pub fn to_defaulted_string(&self) -> Option<String> {
        self.to_string_number(Some(self.get_defaulted_value()))
    }

    /// Java inherited `ConstEtomoNumber.is()`, whose virtual `isNull()` call resolves to
    /// this class's override - which is why a fresh instance built by
    /// `EtomoBoolean2(String, int, int)` with a non-zero off value is already true.  See
    /// the module header.
    pub fn is(&self) -> bool {
        if self.is_null() || self.base.base.base.equals_int(0) {
            return false;
        }
        true
    }

    /// Java inherited package-private `ConstEtomoNumber.getDefaultedValue()`, whose
    /// virtual `isNull()` call resolves to this class's override.  See the module header.
    pub(crate) fn get_defaulted_value(&self) -> Number {
        if self.base.base.base.is_default_set() && self.is_null() {
            return self.base.base.base.default_value;
        }
        self.base.base.base.get_value()
    }

    /// Java inherited `ConstEtomoNumber.setValidValues(int[])`, whose virtual
    /// `setInvalidReason()` call resolves to this class's override - which is why every
    /// constructor of this class throws for a `requiredMap` that makes `nullIsValid`
    /// false.  See the module header.
    pub fn set_valid_values(&mut self, valid_values: Option<&[i32]>) -> &mut ConstEtomoNumber {
        self.base.base.base.reset_state();
        match valid_values {
            None => self.base.base.base.valid_values = None,
            Some(valid_values) if valid_values.is_empty() => {
                self.base.base.base.valid_values = None
            }
            Some(valid_values) => {
                let mut list = Vec::with_capacity(valid_values.len());
                for i in 0..valid_values.len() {
                    let valid_value = valid_values[i];
                    if !self.base.base.base.is_null_int(valid_value) {
                        list.push(self.base.base.base.new_number_from_int(valid_values[i]));
                    }
                }
                self.base.base.base.valid_values = Some(list);
            }
        }
        self.set_invalid_reason();
        &mut self.base.base.base
    }

    /// Java inherited `ConstEtomoNumber.setNullIsValid(boolean)`, whose virtual
    /// `setInvalidReason()` call resolves to this class's override.  See the module
    /// header.
    pub fn set_null_is_valid(&mut self, null_is_valid: bool) -> &mut ConstEtomoNumber {
        self.base.base.base.reset_state();
        self.base.base.base.null_is_valid = null_is_valid;
        self.set_invalid_reason();
        &mut self.base.base.base
    }

    /// Java inherited `ConstEtomoNumber.setValidFloor(int)`, whose virtual
    /// `setInvalidReason()` call resolves to this class's override.  See the module
    /// header.
    pub fn set_valid_floor(&mut self, valid_floor: i32) -> &mut ConstEtomoNumber {
        self.base.base.base.reset_state();
        self.base.base.base.valid_floor = self.base.base.base.new_number_from_int(valid_floor);
        self.set_invalid_reason();
        &mut self.base.base.base
    }

    /// Java inherited `EtomoNumber.set(String)`, whose two virtual calls -
    /// `newNumber(String, StringBuffer)` and `setInvalidReason()` - resolve to this
    /// class's overrides.  See the module header.
    pub fn set_string(&mut self, value: Option<&str>) -> &mut ConstEtomoNumber {
        if self.base.base.base.is_debug() {
            println!(
                "value={}",
                match value {
                    None => "null",
                    Some(value) => value,
                }
            );
        }
        self.base.base.base.reset_state();
        let blank = match value {
            None => true,
            Some(value) => java_lang_string_matches_whitespace(value),
        };
        if blank {
            self.base.base.base.current_value = self.base.base.base.new_number();
        } else {
            let mut invalid_buffer = String::new();
            let number = self.new_number_from_string(value, &mut invalid_buffer);
            let number = self.base.base.base.apply_floor_value(Some(number));
            let number = self.base.base.base.apply_ceiling_value(number);
            self.base.base.base.current_value = self.base.base.base.new_number_from_number(number);
            if self.base.base.base.is_debug() {
                println!(
                    "currentValue={},invalidBuffer={}",
                    self.base.base.base.current_value, invalid_buffer
                );
            }
            if !invalid_buffer.is_empty() {
                self.base
                    .base
                    .base
                    .add_invalid_reason(Some(&invalid_buffer.clone()));
            } else {
                self.set_invalid_reason();
            }
        }
        &mut self.base.base.base
    }

    /// Java inherited `EtomoNumber.set(Number)`, whose virtual `setInvalidReason()` call
    /// resolves to this class's override.  See the module header.
    pub fn set_number(&mut self, value: Option<Number>) -> &mut ConstEtomoNumber {
        self.base.base.base.reset_state();
        let number = self.base.base.base.apply_floor_value(value);
        let number = self.base.base.base.apply_ceiling_value(number);
        self.base.base.base.current_value = self.base.base.base.new_number_from_number(number);
        self.set_invalid_reason();
        &mut self.base.base.base
    }

    /// Java inherited `EtomoNumber.set(int)`, whose virtual `setInvalidReason()` call
    /// resolves to this class's override.  See the module header.
    pub fn set_int(&mut self, value: i32) -> &mut ConstEtomoNumber {
        let number = self.base.base.base.new_number_from_int(value);
        self.set_number(Some(number))
    }

    /// Java inherited `EtomoNumber.set(boolean)`, whose virtual `setInvalidReason()`
    /// call resolves to this class's override.  See the module header.
    pub fn set_boolean(&mut self, value: bool) -> &mut ConstEtomoNumber {
        let number = self.base.base.base.new_number_from_boolean(value);
        self.set_number(Some(number))
    }

    /// Java inherited `EtomoNumber.set(ConstEtomoNumber)`, whose virtual
    /// `setInvalidReason()` call resolves to this class's override.  See the module
    /// header.
    pub fn set_const_etomo_number(
        &mut self,
        number: Option<&ConstEtomoNumber>,
    ) -> &mut ConstEtomoNumber {
        match number {
            None => {
                self.reset();
            }
            Some(number) => {
                self.set_number(Some(number.get_value()));
            }
        }
        &mut self.base.base.base
    }

    /// Java inherited `EtomoNumber.reset()`, whose virtual `setInvalidReason()` call
    /// resolves to this class's override.  See the module header.
    pub fn reset(&mut self) -> &mut ConstEtomoNumber {
        self.base.base.base.reset_state();
        self.base.base.base.current_value = self.base.base.base.new_number();
        self.set_invalid_reason();
        &mut self.base.base.base
    }

    /// Java inherited `EtomoNumber.load(Properties)`, whose virtual `set(String)` call
    /// resolves to this class's override.  See the module header.
    pub fn load(&mut self, props: &BTreeMap<String, String>) {
        let value = props.get(&self.base.base.base.name).cloned();
        self.set_string(value.as_deref());
    }

    /// Java inherited `EtomoNumber.load(Properties, String)`, whose virtual `set(String)`
    /// calls resolve to this class's override.  See the module header.
    pub fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        match prepend {
            None => self.load(props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.load(props),
            Some(prepend) if prepend.ends_with('.') => {
                let value = props
                    .get(&format!("{}{}", prepend, self.base.base.base.name))
                    .cloned();
                self.set_string(value.as_deref());
            }
            Some(prepend) => {
                let value = props
                    .get(&format!("{}.{}", prepend, self.base.base.base.name))
                    .cloned();
                self.set_string(value.as_deref());
            }
        }
    }
}

/// Java `toString()`, inherited from `ConstEtomoNumber`, which calls this class's
/// `toString(Number)` override.  That override returns null for an instance built by
/// `EtomoBoolean2(String, int, int)`; Java's `String.valueOf` - what every `println` and
/// string concatenation of such an instance goes through - turns that into "null".
impl std::fmt::Display for EtomoBoolean2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.base.base.base.get_value();
        f.write_str(
            &self
                .to_string_number(Some(value))
                .unwrap_or("null".to_string()),
        )
    }
}
