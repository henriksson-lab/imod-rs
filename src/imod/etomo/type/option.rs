//! `IMOD/Etomo/src/etomo/type/Option.java`.
//!
//! The Java class is named `Option`, so this module refers to the standard library's
//! type as `std::option::Option` throughout.  Importers should rename it
//! (`use crate::imod::etomo::r#type::option::Option as TypeOption;`).  Java `equals(Object)`
//! is only ever given a `String` (or null) by its callers; it is
//! `equals_object(Option<&str>)` here.

/// Java private `DELIMITER`.
const DELIMITER: char = ':';

/// Java final `Option`.
#[derive(Clone, Debug)]
pub struct Option {
    /// Java private final field `value`.
    value: std::option::Option<String>,
    /// Java private final field `descr`.
    descr: std::option::Option<String>,
    /// Java private field `includeValue`.
    include_value: bool,
}

/// `String.equalsIgnoreCase(String)`: false for a null argument; otherwise equal
/// lengths and each char pair equal after upper- or lower-casing.
fn java_lang_string_equals_ignore_case(string: &str, another: std::option::Option<&str>) -> bool {
    let another = match another {
        None => return false,
        Some(another) => another,
    };
    let a: Vec<char> = string.chars().collect();
    let b: Vec<char> = another.chars().collect();
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        let (c1, c2) = (a[i], b[i]);
        if c1 == c2 {
            continue;
        }
        let u1 = c1.to_uppercase().next().unwrap_or(c1);
        let u2 = c2.to_uppercase().next().unwrap_or(c2);
        if u1 == u2 {
            continue;
        }
        if u1.to_lowercase().next().unwrap_or(u1) == u2.to_lowercase().next().unwrap_or(u2) {
            continue;
        }
        return false;
    }
    true
}

impl Option {
    /// Java `Option(String[])`.
    pub fn new_string_array(array: std::option::Option<&[String]>) -> Option {
        let (value, descr) = match array {
            None => (None, None),
            Some(array) if array.is_empty() => (None, None),
            Some(array) if array.len() == 1 => (Some(array[0].clone()), Some(array[0].clone())),
            Some(array) => (Some(array[0].clone()), Some(array[1].clone())),
        };
        Option {
            value,
            descr,
            include_value: false,
        }
    }

    /// Java `Option(String, String)`.
    pub fn new_string_string(
        value: std::option::Option<&str>,
        descr: std::option::Option<&str>,
    ) -> Option {
        Option {
            value: value.map(str::to_string),
            descr: descr.map(str::to_string),
            include_value: false,
        }
    }

    /// Java `Option(Option)`.  `includeValue` is not copied (it keeps its declared
    /// initial value, false).
    pub fn new_option(option: &Option) -> Option {
        Option {
            value: option.value.clone(),
            descr: option.descr.clone(),
            include_value: false,
        }
    }

    /// Java `setIncludeValue(boolean)`.
    pub fn set_include_value(&mut self, include_value: bool) {
        self.include_value = include_value;
    }

    /// Java `equals(String)`.
    pub fn equals_string(&self, string: std::option::Option<&str>) -> bool {
        let string = string.map(|string| string.trim_matches(|c: char| c <= ' '));
        if self.equals_object(string) {
            return true;
        }
        // Option.java:58 calls `string.equalsIgnoreCase(...)` unconditionally, so a
        // null `string` that `equals(Object)` did not accept throws a
        // NullPointerException.  Fixed in translation: a null string matches nothing
        // more, and false is returned.
        let string = match string {
            None => return false,
            Some(string) => string,
        };
        // Check value first
        if java_lang_string_equals_ignore_case(string, self.value.as_deref())
            || java_lang_string_equals_ignore_case(string, self.descr.as_deref())
        {
            return true;
        }
        false
    }

    /// Java `equals(Object)`, for a `String` (or null) argument.
    pub fn equals_object(&self, object: std::option::Option<&str>) -> bool {
        if object.is_none() && self.value.is_none() && self.descr.is_none() {
            return true;
        }
        let object = match object {
            None => return false,
            Some(object) => object,
        };
        let display_string = self.get_display_string();
        if display_string.is_some() && display_string.as_deref() == Some(object) {
            return true;
        }
        // Check value first
        if self.value.as_deref() == Some(object) || self.descr.as_deref() == Some(object) {
            return true;
        }
        false
    }

    /// Java `getDisplayString()`.
    pub fn get_display_string(&self) -> std::option::Option<String> {
        let descr = match &self.descr {
            None => return self.value.clone(),
            Some(descr) => descr,
        };
        if self.value.is_none() || !self.include_value {
            return Some(descr.clone());
        }
        // IncludeValue is true and value and descr are present
        Some(format!(
            "{}{} {}",
            self.value.as_deref().unwrap(),
            DELIMITER,
            descr
        ))
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> std::option::Option<&str> {
        self.value.as_deref()
    }
}

/// Java `toString()`: `getDisplayString()` (a null display string prints as "null"
/// wherever Java concatenates it).
impl std::fmt::Display for Option {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.get_display_string() {
            None => f.write_str("null"),
            Some(display_string) => f.write_str(&display_string),
        }
    }
}
