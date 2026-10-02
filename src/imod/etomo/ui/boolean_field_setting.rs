//! `IMOD/Etomo/src/etomo/ui/BooleanFieldSetting.java`.
//!
//! Three state boolean setting (not set, on, and off).  NOT thread-safe.  The set
//! member variable is turned on when set() is called.  Handles string values by
//! translating them into a boolean, and also storing the original string in a
//! TextFieldSetting instance.  Currently doesn't handle string types other then string
//! and integer.
//!
//! A plain value: the fields that hold one keep it in a cell and mutate it through
//! `&mut self`.

use super::field_setting_interface::FieldSettingInterface;
use super::text_field_setting::TextFieldSetting;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_trim,
};

/// Java `BooleanFieldSetting`.
#[derive(Clone, Debug, Default)]
pub struct BooleanFieldSetting {
    /// Java private `set`, initialised to false.
    set: bool,
    /// Java private `value`, initialised to false.
    value: bool,
    /// Java private `textSetting`, initialised to null.
    text_setting: Option<TextFieldSetting>,
}

impl BooleanFieldSetting {
    /// Java `BooleanFieldSetting()`.
    pub fn new() -> BooleanFieldSetting {
        BooleanFieldSetting {
            set: false,
            value: false,
            text_setting: None,
        }
    }

    /// Java `equals(boolean)`.
    pub fn equals_boolean(&self, input: bool) -> bool {
        self.set && self.value == input
    }

    /// Java `equals(String)`.  Compares input with textSetting member variable if set.
    /// Otherwise does a comparison with value.  If set is false, always returns false.
    pub fn equals_string(&self, input: Option<&str>) -> bool {
        if !self.set {
            return false;
        }
        if let Some(text_setting) = &self.text_setting
            && text_setting.is_set()
        {
            return text_setting.equals_string(input);
        }
        self.value == BooleanFieldSetting::string_to_boolean(input)
    }

    /// Java `set(boolean)`.  Changes set to true and sets value member variable.
    /// TextSetting is reset.
    pub fn set_boolean(&mut self, setting: bool) {
        self.set = true;
        self.value = setting;
        if let Some(text_setting) = self.text_setting.as_mut() {
            text_setting.reset();
        }
    }

    /// Java `set(String)`.  Changes set to true and sets value member variable.
    /// TextSetting is set to preserve string for comparison.
    pub fn set_string(&mut self, setting: Option<&str>) {
        self.set = true;
        self.value = BooleanFieldSetting::string_to_boolean(setting);
        if self.text_setting.is_none() {
            self.text_setting = Some(TextFieldSetting::new());
        }
        self.text_setting.as_mut().unwrap().set_string(setting);
    }

    /// Java `set(boolean, String)`.  For when the setting string does not translate to
    /// the right boolean value.  Changes set to true and sets value member variable to
    /// the boolean setting.  TextSetting is set to settingString, in order to preserve
    /// it for comparison.
    pub fn set_boolean_string(&mut self, setting: bool, setting_string: Option<&str>) {
        self.set = true;
        self.value = setting;
        if self.text_setting.is_none() {
            self.text_setting = Some(TextFieldSetting::new());
        }
        self.text_setting
            .as_mut()
            .unwrap()
            .set_string(setting_string);
    }

    /// Java `reset()`.  Changes set to false.  Resets value and textSetting.
    pub fn reset(&mut self) {
        self.set = false;
        self.value = false;
        if let Some(text_setting) = self.text_setting.as_mut() {
            text_setting.reset();
        }
    }

    /// Java `copy(FieldSettingInterface)`.  Resets and then copies set, value, and
    /// textFieldSetting.
    pub fn copy(&mut self, input: Option<&dyn FieldSettingInterface>) {
        self.reset();
        let mut setting: Option<BooleanFieldSetting> = None;
        if let Some(input) = input {
            setting = input.get_boolean_setting();
        }
        if let Some(setting) = setting {
            self.set = setting.set;
            self.value = setting.value;
            if let Some(setting_text_setting) = &setting.text_setting
                && setting_text_setting.is_set()
            {
                if self.text_setting.is_none() {
                    self.text_setting = Some(TextFieldSetting::new());
                }
                let setting_text_setting: &dyn FieldSettingInterface = setting_text_setting;
                self.text_setting
                    .as_mut()
                    .unwrap()
                    .copy(Some(setting_text_setting));
            }
        }
    }

    /// Java `isValue()`.
    pub fn is_value(&self) -> bool {
        self.value
    }

    /// Java static `stringToBoolean(String)`.  Translates a string to a boolean as best
    /// it can:
    /// - Null:  false (an empty non-boolean directive is being overridden?)
    /// - Empty:  true (an empty attribute or parameter is true (may have to
    ///   intentionally pass an empty string to get this)
    /// - Zero:  false
    /// - False string:  false (f, false, n, no, na, off)
    /// - Any other string:  true (the parameter, directive, or attribute exists and has
    ///   a value)
    pub fn string_to_boolean(string: Option<&str>) -> bool {
        let Some(string) = string else {
            return false;
        };
        let string = java_lang_string_trim(string);
        if string == "" {
            return true;
        }
        // `try { if (Integer.parseInt(string) == 0) return false; }
        // catch (NumberFormatException e) {}`
        if let Ok(number) = java_lang_integer_parse_int(string) {
            if number == 0 {
                return false;
            }
        }
        // `String.compareToIgnoreCase(other) == 0`: the two strings have the same number
        // of chars, and each pair of chars is equal, or equal after
        // `Character.toUpperCase`, or equal after `Character.toLowerCase` of those.
        let compare_to_ignore_case_is_zero = |other: &str| -> bool {
            let upper = |c: char| -> char {
                let mut mapped = c.to_uppercase();
                match (mapped.next(), mapped.next()) {
                    (Some(single), None) => single,
                    _ => c,
                }
            };
            let lower = |c: char| -> char {
                let mut mapped = c.to_lowercase();
                match (mapped.next(), mapped.next()) {
                    (Some(single), None) => single,
                    _ => c,
                }
            };
            let this: Vec<char> = string.chars().collect();
            let other: Vec<char> = other.chars().collect();
            if this.len() != other.len() {
                return false;
            }
            for i in 0..this.len() {
                let c1 = this[i];
                let c2 = other[i];
                if c1 != c2 {
                    let u1 = upper(c1);
                    let u2 = upper(c2);
                    if u1 != u2 && lower(u1) != lower(u2) {
                        return false;
                    }
                }
            }
            true
        };
        if compare_to_ignore_case_is_zero("f")
            || compare_to_ignore_case_is_zero("false")
            || compare_to_ignore_case_is_zero("n")
            || compare_to_ignore_case_is_zero("na")
            || compare_to_ignore_case_is_zero("no")
            || compare_to_ignore_case_is_zero("off")
        {
            return false;
        }
        true
    }
}

impl FieldSettingInterface for BooleanFieldSetting {
    /// Java `getBooleanSetting()`: `this`.
    fn get_boolean_setting(&self) -> Option<BooleanFieldSetting> {
        Some(self.clone())
    }

    /// Java `getTextSetting()`.
    fn get_text_setting(&self) -> Option<TextFieldSetting> {
        if let Some(text_setting) = &self.text_setting
            && text_setting.is_set()
        {
            return Some(text_setting.clone());
        }
        None
    }

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        true
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        self.text_setting
            .as_ref()
            .is_some_and(|text_setting| text_setting.is_set())
    }

    /// Java `isSet()`.
    fn is_set(&self) -> bool {
        self.set
    }
}

/// Java `toString()`.
impl std::fmt::Display for BooleanFieldSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[set:{},value:{}]", self.set, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_to_boolean_follows_the_source_table() {
        assert!(!BooleanFieldSetting::string_to_boolean(None));
        assert!(BooleanFieldSetting::string_to_boolean(Some("  ")));
        assert!(!BooleanFieldSetting::string_to_boolean(Some(" 0 ")));
        assert!(!BooleanFieldSetting::string_to_boolean(Some("OFF")));
        assert!(!BooleanFieldSetting::string_to_boolean(Some("No")));
        assert!(BooleanFieldSetting::string_to_boolean(Some("1")));
        assert!(BooleanFieldSetting::string_to_boolean(Some("yes")));
        let mut setting = BooleanFieldSetting::new();
        assert!(!setting.equals_boolean(false));
        setting.set_string(Some("off"));
        assert!(setting.equals_string(Some("off")));
        assert!(!setting.equals_string(Some("no")));
        let mut copy = BooleanFieldSetting::new();
        copy.copy(Some(&setting));
        assert!(copy.is_set() && !copy.is_value() && copy.is_text());
        assert_eq!(copy.to_string(), "[set:true,value:false]");
    }
}
