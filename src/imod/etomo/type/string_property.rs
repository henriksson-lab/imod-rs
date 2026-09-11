//! `IMOD/Etomo/src/etomo/type/StringProperty.java`.
//!
//! A class to make storing, loading, and removing strings from `Properties` easier.
//! Java `Properties` is modelled by a deterministic `BTreeMap<String, String>`
//! throughout this translation, as `etomo/storage/storable.rs` does.
//!
//! `toString()` can return Java `null` (when `returnNullWhenEmpty` is set and the value
//! is empty), which `std::fmt::Display` cannot express, so the `Display` body prints
//! the four characters `null` - what Java string concatenation produces for that return
//! - and `to_string_option` is the method callers use when they need to see the null.
#![allow(dead_code)]

use super::const_string_property::ConstStringProperty;
use crate::imod::etomo::util::utilities::EMPTY_PATTERN;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

/// The literal `"\\*"` that Java `equals(String)` hands `String.matches`: a single
/// escaped asterisk, so the pattern matches the one-character string `*` and nothing
/// else.  `String.matches` anchors the whole input, which `\A...\z` reproduces.
static ASTERISK_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new("\\A\\*\\z").unwrap());

/// Java `StringProperty`.
pub struct StringProperty {
    /// Java field `key`.
    key: Option<String>,
    /// Java field `returnNullWhenEmpty`.
    return_null_when_empty: bool,
    /// Java field `debug`, initialised to false.
    debug: bool,
    /// Java field `string`, initialised to null.
    string: Option<String>,
    /// Java field `backwardCompatibleKey`, initialised to null.
    backward_compatible_key: Option<String>,
    /// Java field `displayValue`, initialised to null.
    display_value: Option<String>,
}

impl StringProperty {
    /// Java `StringProperty()`.
    pub fn new() -> StringProperty {
        StringProperty {
            key: None,
            return_null_when_empty: false,
            debug: false,
            string: None,
            backward_compatible_key: None,
            display_value: None,
        }
    }

    /// Java `StringProperty(String)`.
    pub fn new_with_key(key: Option<&str>) -> StringProperty {
        StringProperty {
            key: key.map(|key| key.to_string()),
            return_null_when_empty: false,
            debug: false,
            string: None,
            backward_compatible_key: None,
            display_value: None,
        }
    }

    /// Java `StringProperty(String, boolean)`.
    pub fn new_with_key_and_return_null_when_empty(
        key: Option<&str>,
        return_null_when_empty: bool,
    ) -> StringProperty {
        StringProperty {
            key: key.map(|key| key.to_string()),
            return_null_when_empty,
            debug: false,
            string: None,
            backward_compatible_key: None,
            display_value: None,
        }
    }

    /// Java `getKey`.
    pub fn get_key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Java `setDisplayValue`.
    pub fn set_display_value(&mut self, display_value: Option<&str>) {
        self.display_value = display_value.map(|display_value| display_value.to_string());
    }

    /// Java `toString()` with its Java `null` return preserved.
    pub fn to_string_option(&self) -> Option<String> {
        if self.is_empty() {
            if let Some(display_value) = &self.display_value {
                return Some(display_value.clone());
            }
            if self.return_null_when_empty {
                return None;
            }
            return Some(String::new());
        }
        self.string.clone()
    }

    /// Java `set(String)`.  Sets string to input.  If string is null, empty, or
    /// contains only whitespace, sets string to null.
    pub fn set(&mut self, input: Option<&str>) {
        if Self::is_empty_string(input) {
            self.reset();
        } else {
            self.string = input.map(|input| input.to_string());
        }
    }

    /// Java `set(Number)`.  The Rust parameter is the already-formatted
    /// `Number.toString()`, which is what the source passes on; `None` is Java's null
    /// `Number`.
    pub fn set_number(&mut self, input: Option<&str>) {
        match input {
            None => self.set(None),
            Some(input) => self.set(Some(input)),
        }
    }

    /// Java package-private `length`.
    pub fn length(&self) -> i32 {
        if self.is_empty() {
            return 0;
        }
        // Java `String.length()` counts UTF-16 code units.
        self.string
            .as_ref()
            .map(|string| string.encode_utf16().count() as i32)
            .unwrap_or(0)
    }

    /// Java `set(StringProperty)`.
    pub fn set_string_property(&mut self, input: &StringProperty) {
        self.string = input.string.clone();
    }

    /// Java private `isEmpty(String)`.
    fn is_empty_string(string: Option<&str>) -> bool {
        match string {
            None => true,
            Some(string) => EMPTY_PATTERN.is_match(string) || string.is_empty(),
        }
    }

    /// Java `equals(String)`.
    pub fn equals(&self, string: Option<&str>) -> bool {
        let string = match string {
            None => return Self::is_empty_string(self.string.as_deref()),
            Some(string) => string,
        };
        if ASTERISK_PATTERN.is_match(string) {
            return Self::is_empty_string(self.string.as_deref());
        }
        if Self::is_empty_string(self.string.as_deref()) {
            return false;
        }
        self.string.as_deref() == Some(string)
    }

    /// Java `equals(StringProperty)`.
    pub fn equals_string_property(&self, string_property: &StringProperty) -> bool {
        match &self.string {
            None => Self::is_empty_string(string_property.string.as_deref()),
            Some(string) => Some(string.as_str()) == string_property.string.as_deref(),
        }
    }

    /// Java `load(Properties)`.
    pub fn load(&mut self, props: Option<&mut BTreeMap<String, String>>) {
        let key = self.key.clone();
        self.load_private(props, Some(""), key.as_deref(), None);
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(
        &mut self,
        props: Option<&mut BTreeMap<String, String>>,
        prepend: Option<&str>,
    ) {
        let key = self.key.clone();
        self.load_private(props, prepend, key.as_deref(), None);
    }

    /// Java `load(Properties, String, String)`.
    pub fn load_with_default(
        &mut self,
        props: Option<&mut BTreeMap<String, String>>,
        prepend: Option<&str>,
        default_string: Option<&str>,
    ) {
        let key = self.key.clone();
        self.load_private(props, prepend, key.as_deref(), default_string);
    }

    /// Java package-private `loadFromOtherKey`.
    pub fn load_from_other_key(
        &mut self,
        props: Option<&mut BTreeMap<String, String>>,
        prepend: Option<&str>,
        key: Option<&str>,
    ) {
        self.load_private(props, prepend, key, None);
    }

    /// Java private `load(Properties, String, String, String)`.  Load string.
    /// Retrieves with `backwardCompatibleKey` if the default retrieve returns null.
    /// Removes the entry containing `backwardCompatibleKey`.
    fn load_private(
        &mut self,
        props: Option<&mut BTreeMap<String, String>>,
        prepend: Option<&str>,
        key: Option<&str>,
        default_string: Option<&str>,
    ) {
        let props = match props {
            None => {
                match default_string {
                    None => self.reset(),
                    Some(default_string) => self.set(Some(default_string)),
                }
                return;
            }
            Some(props) => props,
        };
        let mut current_key = Self::create_key_with_key(prepend, key);
        self.string = match &current_key {
            None => None,
            Some(current_key) => props.get(current_key).cloned(),
        };
        // Use the backward compatible key if regular key did not work. Remove the
        // backward compatible key.
        if let Some(backward_compatible_key) = self.backward_compatible_key.clone() {
            current_key = Self::create_key_with_key(prepend, Some(&backward_compatible_key));
            if let Some(current_key) = &current_key {
                if props.contains_key(current_key) {
                    if self.string.is_none() {
                        self.string = props.get(current_key).cloned();
                    }
                    props.remove(current_key);
                }
            }
        }
        // Apply the default string
        if let Some(default_string) = default_string {
            if self.string.is_none() {
                self.string = Some(default_string.to_string());
            }
        }
    }

    /// Java `setBackwardCompatibleKey`.
    pub fn set_backward_compatible_key(&mut self, input: Option<&str>) {
        self.backward_compatible_key = input.map(|input| input.to_string());
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: Option<&mut BTreeMap<String, String>>) {
        self.store_with_prepend(props, Some(""));
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(
        &self,
        props: Option<&mut BTreeMap<String, String>>,
        prepend: Option<&str>,
    ) {
        let props = match props {
            None => return,
            Some(props) => props,
        };
        let current_key = self.create_key(prepend);
        // Java `Properties.remove(null)`/`setProperty(null, ..)` would throw; the source
        // only reaches this with a key, and a `None` key has no map entry to touch.
        let current_key = match current_key {
            None => return,
            Some(current_key) => current_key,
        };
        if self.is_empty() {
            props.remove(&current_key);
        } else {
            props.insert(current_key, self.string.clone().unwrap());
        }
    }

    /// Java `remove(Properties, String)`.
    pub fn remove(&self, props: Option<&mut BTreeMap<String, String>>, prepend: Option<&str>) {
        let props = match props {
            None => return,
            Some(props) => props,
        };
        let current_key = self.create_key(prepend);
        if let Some(current_key) = current_key {
            props.remove(&current_key);
        }
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.string = None;
    }

    /// Java private `createKey(String)`.
    fn create_key(&self, prepend: Option<&str>) -> Option<String> {
        Self::create_key_with_key(prepend, self.key.as_deref())
    }

    /// Java private `createKey(String, String)`.
    fn create_key_with_key(prepend: Option<&str>, key: Option<&str>) -> Option<String> {
        if Self::is_empty_string(prepend) {
            return key.map(|key| key.to_string());
        }
        Some(format!("{}.{}", prepend.unwrap(), key.unwrap_or("null")))
    }
}

impl Default for StringProperty {
    fn default() -> StringProperty {
        StringProperty::new()
    }
}

/// Java `isEmpty()`, declared by `ConstStringProperty`.
impl ConstStringProperty for StringProperty {
    fn is_empty(&self) -> bool {
        match &self.string {
            None => true,
            Some(string) => string.is_empty(),
        }
    }
}

/// Java `toString`.  See the module header for the `null` return.
impl std::fmt::Display for StringProperty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.to_string_option() {
            None => f.write_str("null"),
            Some(string) => f.write_str(&string),
        }
    }
}
