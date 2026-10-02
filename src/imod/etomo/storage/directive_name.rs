//! `IMOD/Etomo/src/etomo/storage/DirectiveName.java`.
//!
//! Handles the left side of a directive set.  Directive set:
//! - 1 directive with no axisID information or
//! - 3 directives: A, B, and both axes.
//!
//! This class can return a key, which is identical to the directive name for both axes.
//! It can also return a directive name for a specific axisID.
//!
//! Overloads carry the suffix of their parameter types: static `equals(String,
//! DirectiveType)` is `equals_string_directive_type` and instance `equals(DirectiveType)`
//! is `equals_directive_type`; `setKey(DirectiveDescr)` / `setKey(String)` are
//! `set_key_directive_descr` / `set_key_string`; instance `getType()` is `get_type` and
//! the private static `getType(String[])` is `get_type_string_array`.

use regex::Regex;

use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::storage::directive_def::RUN_TIME_ANY_AXIS_TAG;
use crate::imod::etomo::storage::directive_descr::DirectiveDescr;
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private `TYPE_INDEX`.
const TYPE_INDEX: usize = 0;
/// Java private `COM_FILE_NAME_INDEX`.
const COM_FILE_NAME_INDEX: usize = 1;
/// Java private `PROGRAM_INDEX`.
const PROGRAM_INDEX: usize = 2;
/// Java private `PARAMETER_NAME_INDEX`.
const PARAMETER_NAME_INDEX: usize = 3;
/// Java private `RUNTIME_AXIS_INDEX`.
const RUNTIME_AXIS_INDEX: usize = 2;

/// Java final `DirectiveName`.  A Java `String[]` produced by `String.split` holds no
/// null element, so `key` is `Option<Vec<String>>` with non-null elements.
#[derive(Clone, Debug, Default)]
pub struct DirectiveName {
    /// Java private field `key`, initialised to null.
    key: Option<Vec<String>>,
    /// Java private field `type`, initialised to null.
    r#type: Option<DirectiveType>,
}

impl DirectiveName {
    /// Java `DirectiveName()`.
    pub fn new() -> DirectiveName {
        DirectiveName {
            key: None,
            r#type: None,
        }
    }

    /// Java static `equals(String, DirectiveType)`.
    pub fn equals_string_directive_type(key: Option<&str>, input: Option<DirectiveType>) -> bool {
        let (key, input) = match (key, input) {
            (Some(key), Some(input)) => (key, input),
            _ => return false,
        };
        key.starts_with(&format!("{}{}", input, SEPARATOR_CHAR))
    }

    /// Java `equals(DirectiveType)`.
    pub fn equals_directive_type(&self, input: Option<DirectiveType>) -> bool {
        if self.is_null() {
            return false;
        }
        self.r#type == input
    }

    /// Java `getKey()`.  Returns the name with no axis ID.
    pub fn get_key(&self) -> Option<String> {
        Self::convert_key_to_string(self.key.as_deref())
    }

    /// Java `getKeyDescription()`.
    pub fn get_key_description(&self) -> Option<String> {
        let key = self.get_key();
        if self.r#type == Some(DirectiveType::RUN_TIME) {
            // A RUN_TIME type implies a non-null key (`getType(String[])` returns null
            // for a null key).
            return key.map(|key| {
                key.replace(&format!("{}{}", SEPARATOR_CHAR, RUN_TIME_ANY_AXIS_TAG), "")
            });
        }
        key
    }

    /// Java `getComFileName()`.
    pub fn get_com_file_name(&self) -> Option<String> {
        if self.is_null() {
            return None;
        }
        let key = self.key.as_ref().unwrap();
        // Only comparam directives have a comfile name. Missing comfile name.
        if self.r#type != Some(DirectiveType::COM_PARAM) || key.len() <= COM_FILE_NAME_INDEX {
            return None;
        }
        Some(key[COM_FILE_NAME_INDEX].clone())
    }

    /// Java `getParameterName()`.
    pub fn get_parameter_name(&self) -> Option<String> {
        if self.is_null() {
            return None;
        }
        let key = self.key.as_ref().unwrap();
        if self.r#type == Some(DirectiveType::SETUP_SET) {
            if key.len() < 2 {
                return None;
            }
            let copyarg = DirectiveType::COPY_ARG.equals(Some(&key[1]));
            if copyarg {
                if key.len() > 2 {
                    return Some(key[2].clone());
                }
            } else {
                return Some(key[1].clone());
            }
        } else if (self.r#type == Some(DirectiveType::COM_PARAM)
            || self.r#type == Some(DirectiveType::RUN_TIME))
            && key.len() > PARAMETER_NAME_INDEX
        {
            return Some(key[PARAMETER_NAME_INDEX].clone());
        }
        None
    }

    /// Java `getProgramName()`.
    pub fn get_program_name(&self) -> Option<String> {
        if self.is_null() {
            return None;
        }
        let key = self.key.as_ref().unwrap();
        // Only comparam directives have a program name. Missing program name.
        if self.r#type != Some(DirectiveType::COM_PARAM) || key.len() <= PROGRAM_INDEX {
            return None;
        }
        Some(key[PROGRAM_INDEX].clone())
    }

    /// Java `getTitle()`.
    pub fn get_title(&self) -> Option<String> {
        if self.is_null() {
            return None;
        }
        if self.r#type == Some(DirectiveType::COM_PARAM) {
            // Java string concatenation prints a null part as "null".
            Some(format!(
                "{}.{}",
                self.get_com_file_name()
                    .unwrap_or_else(|| "null".to_string()),
                self.get_parameter_name()
                    .unwrap_or_else(|| "null".to_string())
            ))
        } else {
            self.get_parameter_name()
        }
    }

    /// Java `getType()`.
    pub fn get_type(&self) -> Option<DirectiveType> {
        self.r#type
    }

    /// Java package-private `isCopyArg()`.
    pub(crate) fn is_copy_arg(&self) -> bool {
        // A SETUP_SET type implies a non-null key.
        self.r#type == Some(DirectiveType::SETUP_SET)
            && self.key.as_ref().unwrap().len() > 1
            && self.key.as_ref().unwrap()[1] == "copyarg"
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        if self.is_null() {
            return false;
        }
        self.r#type.is_some() && self.key.as_ref().unwrap().len() > 1
    }

    /// Java package-private `deepCopy(DirectiveName)`.  Copies the data rather then
    /// pointers to mutable objects in the parameter.
    pub(crate) fn deep_copy(&mut self, directive_name: &DirectiveName) {
        match &directive_name.key {
            None => self.key = None,
            Some(source) => {
                let mut key = Vec::with_capacity(source.len());
                for i in 0..source.len() {
                    key.push(source[i].clone());
                }
                self.key = Some(key);
            }
        }
        self.r#type = directive_name.r#type;
    }

    /// Java package-private `setKey(DirectiveDescr)`.
    pub(crate) fn set_key_directive_descr(&mut self, descr: &dyn DirectiveDescr) {
        // The a and b axes are not included for comparam and runtime directives in the
        // directive.csv file, so no need to remove them.
        self.key = Self::split_key(descr.get_name().as_deref());
        self.r#type = Self::get_type_string_array(self.key.as_deref());
    }

    /// Java package-private static `makeKey(String)`.
    pub(crate) fn make_key(input: Option<&str>) -> Option<String> {
        let mut static_key = Self::split_key(input);
        let static_type = Self::get_type_string_array(static_key.as_deref());
        Self::strip_axis(static_key.as_deref_mut(), static_type);
        Self::convert_key_to_string(static_key.as_deref())
    }

    /// Java private static `splitKey(String)`.
    fn split_key(input: Option<&str>) -> Option<Vec<String>> {
        if let Some(input) = input {
            if !java_lang_string_matches_whitespace(input) {
                return Some(java_lang_string_split(
                    input,
                    &Regex::new(&regex::escape(SEPARATOR_CHAR)).unwrap(),
                ));
            }
        }
        None
    }

    /// Java private static `getType(String[])`.
    fn get_type_string_array(key: Option<&[String]>) -> Option<DirectiveType> {
        if let Some(key) = key {
            if key.len() > TYPE_INDEX {
                return DirectiveType::get_first_section_instance(Some(&key[TYPE_INDEX]));
            }
        }
        None
    }

    /// Java private static `stripAxis(String[], DirectiveType)`.
    fn strip_axis(key: Option<&mut [String]>, r#type: Option<DirectiveType>) -> Option<AxisID> {
        if r#type != Some(DirectiveType::COM_PARAM) && r#type != Some(DirectiveType::RUN_TIME) {
            return None;
        }
        // A COM_PARAM or RUN_TIME type implies a non-null key.
        let key = key.unwrap();
        // Remove axisID from the name to create a key. Standardize the key to the Any form
        // of the directive name, and return the axisID that was found.
        let mut axis_id = None;
        let first = AxisID::First.get_extension();
        let second = AxisID::Second.get_extension();
        for i in 0..key.len() {
            if r#type == Some(DirectiveType::COM_PARAM)
                && i == COM_FILE_NAME_INDEX
                && (key[i].ends_with(&first) || key[i].ends_with(&second))
            {
                if key[i].ends_with(&first) {
                    axis_id = Some(AxisID::First);
                } else {
                    axis_id = Some(AxisID::Second);
                }
                // Strip off the a or b
                let length = key[i].len();
                key[i] = key[i][..length - 1].to_string();
            } else if r#type == Some(DirectiveType::RUN_TIME)
                && i == RUNTIME_AXIS_INDEX
                && (key[i] == first || key[i] == second)
            {
                if key[i] == first {
                    axis_id = Some(AxisID::First);
                } else {
                    axis_id = Some(AxisID::Second);
                }
                // Replace with "any".
                key[i] = RUN_TIME_ANY_AXIS_TAG.to_string();
            }
        }
        axis_id
    }

    /// Java private static `convertKeyToString(String[])`.
    fn convert_key_to_string(key: Option<&[String]>) -> Option<String> {
        let key = match key {
            Some(key) if !key.is_empty() => key,
            _ => return None,
        };
        let mut buffer = String::new();
        for i in 0..key.len() {
            buffer.push_str(if i > 0 { "." } else { "" });
            buffer.push_str(&key[i]);
        }
        Some(buffer)
    }

    /// Java `setKey(String)`.  Strips axis information and saves a key containing the
    /// "any" form of the directive name.  For a directive with no axis information or an
    /// "any" directive name, the key is the same as the input string, and null is
    /// returned.  Returns the axisID that was removed from the name (or null for "any").
    pub fn set_key_string(&mut self, input: Option<&str>) -> Option<AxisID> {
        self.key = Self::split_key(input);
        self.r#type = Self::get_type_string_array(self.key.as_deref());
        Self::strip_axis(self.key.as_deref_mut(), self.r#type)
    }

    /// Java private `isNull()`.
    fn is_null(&self) -> bool {
        match &self.key {
            None => true,
            Some(key) => key.is_empty(),
        }
    }

    /// Java private static `mayContainAxisID(String)`.  Unused in the source.
    fn may_contain_axis_id(name: Option<&str>) -> bool {
        let name = match name {
            None => return false,
            Some(name) => name,
        };
        (name.starts_with(&format!("{}{}", DirectiveType::RUN_TIME, SEPARATOR_CHAR))
            || name.starts_with(&format!("{}{}", DirectiveType::COM_PARAM, SEPARATOR_CHAR)))
            && (name
                .find(&format!(
                    "{}{}",
                    AxisID::First.get_extension(),
                    SEPARATOR_CHAR
                ))
                .is_some()
                || name
                    .find(&format!(
                        "{}{}",
                        AxisID::Second.get_extension(),
                        SEPARATOR_CHAR
                    ))
                    .is_some())
    }

    /// Java `getName()`.  Returns the directive name for the axisID specified.
    pub fn get_name(&self) -> Option<String> {
        let key = self.key.as_ref()?;
        // Set ext to the correct form
        let mut ext = "";
        if self.r#type == Some(DirectiveType::RUN_TIME) {
            ext = RUN_TIME_ANY_AXIS_TAG;
        }
        // Create a string version of the directive name with the correct axisID string
        let mut buffer = String::new();
        for i in 0..key.len() {
            let separator = if i > 0 { SEPARATOR_CHAR } else { "" };
            if self.r#type == Some(DirectiveType::COM_PARAM) && i == COM_FILE_NAME_INDEX {
                buffer.push_str(separator);
                buffer.push_str(&key[i]);
                buffer.push_str(ext);
            } else if self.r#type == Some(DirectiveType::RUN_TIME) && i == RUNTIME_AXIS_INDEX {
                buffer.push_str(separator);
                buffer.push_str(ext);
            } else {
                buffer.push_str(separator);
                buffer.push_str(&key[i]);
            }
        }
        Some(buffer)
    }
}

/// Java `toString()`.
impl std::fmt::Display for DirectiveName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = String::new();
        buffer.push_str("[key:");
        if let Some(key) = &self.key {
            for i in 0..key.len() {
                buffer.push_str(&key[i]);
                buffer.push(' ');
            }
        }
        match self.r#type {
            None => write!(f, "{},type:null]", buffer),
            Some(r#type) => write!(f, "{},type:{}]", buffer, r#type),
        }
    }
}
