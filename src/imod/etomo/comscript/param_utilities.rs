//! `IMOD/Etomo/src/etomo/comscript/ParamUtilities.java`.
//!
//! A collection of static utility functions to work with Com script parameter objects.
//! There are four classes of functions:
//!
//! 1. setParamIfPresent returns value of the specified parameter from the com script if
//!    it is present in the comscript, otherwise it returns the "not present value"
//!    specified in the argument list.  The purpose of these functions is to assist in
//!    reading com scripts into parameter objects
//!
//! 2. updateScriptParameter updates the ComScript object with specified parameter,
//!    deleting the keyword if the specified parameter equals the default value.  The
//!    purpose of these function is to assist in writing out new com scripts
//!
//! 3. valueOf returns the String representation of the parameter handling the not
//!    present or default case correctly
//!
//! 4. parseType parses a string into the specfied type returning the correct not
//!    present or default value if the String is null or empty or white space
//!
//! Copyright: Copyright(c) 2002, 2003, 2004
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado
//!
//! **Unchecked exceptions.**  `Integer.parseInt`/`Double.parseDouble` throw the
//! unchecked `NumberFormatException`.  Where the Java method declares no checked
//! exception (`parseInt`, `parseDouble`, `set(String, FortranInputString, int)`), the
//! translation returns `Err(message)` with the JVM's `NumberFormatException` message.
//! Where it already declares `InvalidParameterException` (`setParamIfPresent` for
//! `int`/`double`), the `NumberFormatException` is returned as an
//! `InvalidParameterException` with the same message: both reach the same
//! `catch (Exception)` in `ComScriptUtil.initialize`, and only the exception class name
//! in its dialog differs.
//!
//! A null `key` (`NullPointerException`) and a blank `key`
//! (`IllegalArgumentException`) are programming errors - every caller passes a
//! non-blank constant - and stay panics.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::ParseComScriptError;
use super::fortran_input_string::FortranInputString;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::invalid_parameter_exception::InvalidParameterException;
use super::string_list::StringList;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_double_value_of, java_lang_float_to_string,
    java_lang_integer_parse_int, java_lang_string_matches_whitespace,
};

/// Java package-private `INT_NOT_SET`.
pub const INT_NOT_SET: i32 = i32::MIN;
/// Java private `zeroOrMoreWhiteSpace` (`"\\s*"`); applied through
/// `java_lang_string_matches_whitespace`.
const ZERO_OR_MORE_WHITE_SPACE: &str = "\\s*";

/// Java `String.matches("\\S+")`: one or more characters, none of them Java `\s`
/// (`[ \t\n\x0B\f\r]`), anchored at both ends.  A JDK shim, the counterpart of
/// `const_etomo_number::java_lang_string_matches_whitespace`, not a translated unit.
fn java_lang_string_matches_non_white_space(value: &str) -> bool {
    !value.is_empty()
        && !value
            .chars()
            .any(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
}

/// Java `isEmpty(String)`.  Returns true if value is null, has a zero length, or
/// contains only whitespace.
pub fn is_empty(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(value) => !java_lang_string_matches_non_white_space(value),
    }
}

/// Java `valueOf(int)`.  Return the string representation of the int value or an
/// empty string if the value is the not present value.
pub fn value_of_int(value: i32) -> String {
    if value == i32::MIN {
        return String::new();
    }
    value.to_string()
}

/// Java `valueOf(float)`.  Return the string representation of the float value or an
/// empty string if the value is the not present value.
pub fn value_of_float(value: f32) -> String {
    if value.is_nan() {
        return String::new();
    }
    java_lang_float_to_string(value)
}

/// Java `valueOf(double)`.  Return the string representation of the double value or
/// an empty string if the value is the not present value.
pub fn value_of_double(value: f64) -> String {
    if value.is_nan() {
        return String::new();
    }
    java_lang_double_to_string(value)
}

/// Java `valueOf(double[])`.
pub fn value_of_double_array(values: &[f64]) -> Vec<String> {
    let mut strings = vec![String::new(); if values.is_empty() { 1 } else { values.len() }];
    if values.is_empty() {
        strings[0] = String::new();
    }
    for i in 0..values.len() {
        strings[i] = value_of_double(values[i]);
    }
    strings
}

/// Java `valueOf(FortranInputString[])`.
pub fn value_of_fortran_input_string_array(value_array: Option<&[FortranInputString]>) -> String {
    let value_array = match value_array {
        None => return String::new(),
        Some(value_array) if value_array.is_empty() => return String::new(),
        Some(value_array) => value_array,
    };
    let mut buffer = value_array[0].to_string_default_is_blank(true);
    for i in 1..value_array.len() {
        buffer.push_str(&(" ".to_string() + &value_array[i].to_string_default_is_blank(true)));
    }
    buffer
}

/// Java `parseInt(String)`.  Parse an integer value from a string, returning the
/// default value if the string is white space.  `Err` carries the
/// `NumberFormatException` message.
pub fn parse_int(value: Option<&str>) -> Result<i32, String> {
    let value = match value {
        None => return Ok(i32::MIN),
        Some(value) if !java_lang_string_matches_non_white_space(value) => return Ok(i32::MIN),
        Some(value) => value,
    };
    java_lang_integer_parse_int(value)
}

/// Java `parse(String, boolean, int)`.  Parse a FortranInputString array from a string.
/// Use StringList to split the string at whitespace.
pub fn parse_string(
    value: Option<&str>,
    integer_type: bool,
    input_string_size: i32,
) -> Result<Option<Vec<FortranInputString>>, FortranInputSyntaxException> {
    let value = match value {
        None => return Ok(None),
        Some(value) if java_lang_string_matches_whitespace(value) => return Ok(None),
        Some(value) => value,
    };
    let mut string_list = StringList::new();
    string_list.parse_string(Some(value));
    let mut integer_type_array = vec![false; input_string_size as usize];
    for i in 0..integer_type_array.len() {
        integer_type_array[i] = integer_type;
    }
    parse_string_list(Some(&string_list), &integer_type_array, input_string_size)
}

/// Java `parse(StringList, boolean[], int)`.  Parse a FortranInputString array from a
/// StringList.
///
/// A null element of the list reaches `validateAndSet(null)`, as in the source.
pub fn parse_string_list(
    string_list: Option<&StringList>,
    integer_type: &[bool],
    input_string_size: i32,
) -> Result<Option<Vec<FortranInputString>>, FortranInputSyntaxException> {
    if let Some(string_list) = string_list {
        if string_list.get_n_elements() > 0 {
            let string_list_size = string_list.get_n_elements();
            let mut input_string_array: Vec<FortranInputString> =
                Vec::with_capacity(string_list_size as usize);
            for i in 0..string_list_size {
                let mut input_string = FortranInputString::new(input_string_size);
                input_string.set_integer_type_array(integer_type);
                input_string.validate_and_set(string_list.get(i))?;
                input_string_array.push(input_string);
            }
            return Ok(Some(input_string_array));
        }
    }
    Ok(None)
}

/// Java `get(int, int)`.  Returns value or defaultValue if value isn't set.
pub fn get(value: i32, default_value: i32) -> i32 {
    if value == INT_NOT_SET {
        return default_value;
    }
    value
}

/// Java `parseDouble(String)`.  Parse a double value from a string returning the
/// default value the string is white space.  `Err` carries the
/// `NumberFormatException` message.
pub fn parse_double(value: Option<&str>) -> Result<f64, String> {
    let value = match value {
        None => return Ok(f64::NAN),
        Some(value) if !java_lang_string_matches_non_white_space(value) => return Ok(f64::NAN),
        Some(value) => value,
    };
    java_lang_double_value_of(value)
}

/// Java `set(String, FortranInputString)`.  Sets a FortranInputString from a string.
pub fn set_fortran_input_string(
    value: Option<&str>,
    target: &mut FortranInputString,
) -> Result<(), FortranInputSyntaxException> {
    match value {
        None => target.set_default(),
        Some(value) => target.validate_and_set(Some(value))?,
    }
    Ok(())
}

/// Java `set(String, FortranInputString, int)`.  `Err` carries the
/// `NumberFormatException` message of `Double.parseDouble`.
pub fn set_fortran_input_string_index(
    value: Option<&str>,
    target: &mut FortranInputString,
    index: i32,
) -> Result<(), String> {
    match value {
        None => target.set_default_index(index),
        Some(value) if !java_lang_string_matches_non_white_space(value) => {
            target.set_default_index(index)
        }
        Some(value) => target.set_index_double(index, java_lang_double_value_of(value)?),
    }
    Ok(())
}

/// Java `setParamIfPresent(ComScriptCommand, String, int, boolean[])`.
pub fn set_param_if_present_fortran_input_string_array(
    script_command: &ComScriptCommand,
    keyword: &str,
    size: i32,
    integer_type: &[bool],
) -> Result<Option<Vec<FortranInputString>>, ParseComScriptError> {
    if script_command.has_keyword(Some(keyword))? {
        let values = script_command.get_values(Some(keyword));
        return Ok(parse_string_list(
            Some(&StringList::new_from_array(Some(&values))),
            integer_type,
            size,
        )?);
    }
    Ok(None)
}

/// Java `setParamIfPresent(ComScriptCommand, String, boolean)`.  Return the boolean
/// parameter if it is present in the com script command object, otherwise return the
/// notPresentValue.
pub fn set_param_if_present_boolean(
    script_command: &ComScriptCommand,
    keyword: &str,
    not_present_value: bool,
) -> Result<bool, InvalidParameterException> {
    Ok(if script_command.has_keyword(Some(keyword))? {
        true
    } else {
        not_present_value
    })
}

/// Java `setParamIfPresent(ComScriptCommand, String, String)`.  Return the string
/// parameter if it is present in the com script command object, otherwise return the
/// notPresentValue.
pub fn set_param_if_present_string(
    script_command: &ComScriptCommand,
    keyword: &str,
    not_present_value: Option<&str>,
) -> Result<Option<String>, InvalidParameterException> {
    Ok(if script_command.has_keyword(Some(keyword))? {
        script_command.get_value(Some(keyword))?
    } else {
        not_present_value.map(|value| value.to_string())
    })
}

/// Java `setParamIfPresent(ComScriptCommand, String, int)`.  Return the int parameter
/// if it is present in the com script command object, otherwise return the
/// notPresentValue.  `NumberFormatException` is returned as an
/// `InvalidParameterException` (see the module comment).
pub fn set_param_if_present_int(
    script_command: &ComScriptCommand,
    keyword: &str,
    not_present_value: i32,
) -> Result<i32, InvalidParameterException> {
    if script_command.has_keyword(Some(keyword))? {
        return match script_command.get_value(Some(keyword))? {
            None => Err(InvalidParameterException::new(Some(
                "Cannot parse null string: null",
            ))),
            Some(value) => java_lang_integer_parse_int(&value)
                .map_err(|message| InvalidParameterException::new(Some(&message))),
        };
    }
    Ok(not_present_value)
}

/// Java `setParamIfPresent(ComScriptCommand, String, double)`.  Return the double
/// parameter if it is present in the com script command object, otherwise return the
/// notPresentValue.  `NumberFormatException` (and the `NullPointerException`
/// `Double.parseDouble(null)` throws) is returned as an `InvalidParameterException`
/// (see the module comment).
pub fn set_param_if_present_double(
    script_command: &ComScriptCommand,
    keyword: &str,
    not_present_value: f64,
) -> Result<f64, InvalidParameterException> {
    if script_command.has_keyword(Some(keyword))? {
        return match script_command.get_value(Some(keyword))? {
            None => Err(InvalidParameterException::new(None)),
            Some(value) => java_lang_double_value_of(&value)
                .map_err(|message| InvalidParameterException::new(Some(&message))),
        };
    }
    Ok(not_present_value)
}

/// Java `setParamIfPresent(ComScriptCommand, String, FortranInputString)`.  Set the
/// FortranInputString parameter if it is present in the com script command object.
pub fn set_param_if_present_fortran_input_string(
    script_command: &ComScriptCommand,
    keyword: &str,
    fis_parameter: &mut FortranInputString,
) -> Result<(), ParseComScriptError> {
    if script_command.has_keyword(Some(keyword))? {
        let value = script_command.get_value(Some(keyword))?;
        fis_parameter.validate_and_set(value.as_deref())?;
    }
    Ok(())
}

/// Java `setParamIfPresent(ComScriptCommand, String, StringList)`.  Set the StringList
/// parameter if it is present in the com script command object.
pub fn set_param_if_present_string_list(
    script_command: &ComScriptCommand,
    keyword: &str,
    string_list: Option<StringList>,
) -> Result<Option<StringList>, InvalidParameterException> {
    let mut string_list = string_list;
    if script_command.has_keyword(Some(keyword))? {
        if string_list.is_none() {
            string_list = Some(StringList::new());
        }
        let value = script_command.get_value(Some(keyword))?;
        string_list.as_mut().unwrap().parse_string(value.as_deref());
    }
    Ok(string_list)
}

/// Java `updateScriptParameter(ComScriptCommand, String, String)`.  Update the
/// specified com script parameter with the supplied value.
pub fn update_script_parameter_string(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: Option<&str>,
) -> Result<(), BadComScriptException> {
    update_script_parameter_string_required(script_command, key, value, false)
}

/// Java `updateScriptParameter(ComScriptCommand, String, String, boolean)`.
pub fn update_script_parameter_string_required(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: Option<&str>,
    required: bool,
) -> Result<(), BadComScriptException> {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if java_lang_string_matches_whitespace(key) {
        panic!("java.lang.IllegalArgumentException");
    }
    if value.is_some() && !java_lang_string_matches_whitespace(value.unwrap()) {
        script_command.set_value(Some(key), value);
    } else {
        script_command.delete_key(Some(key));
        if required {
            return Err(BadComScriptException::new(&format!(
                "{} missing required parameter: {}.",
                script_command.get_command().unwrap_or("null"),
                key
            )));
        }
    }
    Ok(())
}

/// Java `updateScriptParameter(ComScriptCommand, String, StringList)`.
pub fn update_script_parameter_string_list(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: Option<&StringList>,
) -> Result<(), BadComScriptException> {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if java_lang_string_matches_whitespace(key) {
        panic!("java.lang.IllegalArgumentException");
    }
    match value {
        Some(value) if value.get_n_elements() > 0 => {
            script_command.set_value(Some(key), Some(&value.to_string()));
        }
        _ => {
            script_command.delete_key(Some(key));
        }
    }
    Ok(())
}

/// Java `updateScriptParameter(ComScriptCommand, String, int)`.
pub fn update_script_parameter_int(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: i32,
) {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if !(value == i32::MIN) {
        script_command.set_value(Some(key), Some(&value.to_string()));
    } else {
        script_command.delete_key(Some(key));
    }
}

/// Java `updateScriptParameter(ComScriptCommand, String, double)`.
pub fn update_script_parameter_double(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: f64,
) {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if !value.is_nan() {
        script_command.set_value(Some(key), Some(&java_lang_double_to_string(value)));
    } else {
        script_command.delete_key(Some(key));
    }
}

/// Java `updateScriptParameter(ComScriptCommand, String, FortranInputString)`.
pub fn update_script_parameter_fortran_input_string(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: &FortranInputString,
) {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if !value.is_active() {
        script_command.delete_key(Some(key));
        return;
    }
    if value.values_set() && !value.is_default() {
        script_command.set_value(Some(key), Some(&value.to_string()));
    } else {
        script_command.delete_key(Some(key));
    }
}

/// Java `updateScriptParameter(ComScriptCommand, String, FortranInputString, boolean,
/// boolean)`.
pub fn update_script_parameter_fortran_input_string_format(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: &FortranInputString,
    default_is_blank: bool,
    strip_value_ends_char: bool,
) {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if !value.is_active() {
        script_command.delete_key(Some(key));
        return;
    }
    if value.values_set() && !value.is_default() {
        script_command.set_value(
            Some(key),
            Some(&value.to_string_strip(default_is_blank, strip_value_ends_char)),
        );
    } else {
        script_command.delete_key(Some(key));
    }
}

/// Java `updateScriptParameter(ComScriptCommand, String, FortranInputString[])`.
pub fn update_script_parameter_fortran_input_string_array(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    value: Option<&[FortranInputString]>,
) {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if java_lang_string_matches_whitespace(key) {
        panic!("java.lang.IllegalArgumentException");
    }
    match value {
        None => {
            script_command.delete_key(Some(key));
        }
        Some(value) if value.is_empty() => {
            script_command.delete_key(Some(key));
        }
        Some(value) => {
            let mut buffer = value[0].to_string();
            for i in 1..value.len() {
                buffer.push_str(&(" ".to_string() + &value[i].to_string()));
            }
            script_command.set_value(Some(key), Some(&buffer));
        }
    }
}

/// Java `updateScriptParameter(ComScriptCommand, String, double[])`.
pub fn update_script_parameter_double_array(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    values: Option<&[f64]>,
) -> Result<(), BadComScriptException> {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    let values = match values {
        None => {
            script_command.delete_key_all(Some(key));
            return Ok(());
        }
        Some(values) if values.is_empty() => {
            script_command.delete_key_all(Some(key));
            return Ok(());
        }
        Some(values) => values,
    };
    let mut buffer = String::new();
    for i in 0..values.len() {
        buffer.push_str(&java_lang_double_to_string(values[i]));
        if i < values.len() - 1 {
            buffer.push(',');
        }
    }
    script_command.set_value(Some(key), Some(&buffer));
    Ok(())
}

/// Java `updateScriptParameter(ComScriptCommand, String, boolean)`.
pub fn update_script_parameter_boolean(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    set: bool,
) {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    if set {
        script_command.set_value(Some(key), Some(""));
    } else {
        script_command.delete_key(Some(key));
    }
}

/// Java `updateScriptParameterStrings(ComScriptCommand, String, Vector)`.
pub fn update_script_parameter_strings(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    strings: Option<&[Option<String>]>,
) -> Result<(), BadComScriptException> {
    update_script_parameter_strings_required(script_command, key, strings, false)
}

/// Java `updateScriptParameterStrings(ComScriptCommand, String, Vector, boolean)`.
pub fn update_script_parameter_strings_required(
    script_command: &mut ComScriptCommand,
    key: Option<&str>,
    strings: Option<&[Option<String>]>,
    required: bool,
) -> Result<(), BadComScriptException> {
    let key = match key {
        None => panic!("java.lang.NullPointerException"),
        Some(key) => key,
    };
    let strings = match strings {
        Some(strings) if !strings.is_empty() => strings,
        _ => {
            script_command.delete_key_all(Some(key));
            if required {
                return Err(BadComScriptException::new(&format!(
                    "{} missing required parameter: {}.",
                    script_command.get_command().unwrap_or("null"),
                    key
                )));
            }
            return Ok(());
        }
    };
    script_command.set_values(Some(key), strings);
    Ok(())
}
