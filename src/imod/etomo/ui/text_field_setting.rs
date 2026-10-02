//! `IMOD/Etomo/src/etomo/ui/TextFieldSetting.java`.
//!
//! Text setting that can be turned on and off.  Not thread safe.  A null value can be
//! a valid setting.  The set member variable is turned on when set() is called.
//!
//! Held as `Rc<TextFieldSetting>` by the fields that hand it out as their checkpoint
//! or highlight (Java shares the object), so its state is interior-mutable and every
//! method takes `&self`.

use std::cell::{Cell, RefCell};

use super::boolean_field_setting::BooleanFieldSetting;
use super::field_setting_interface::FieldSettingInterface;
use super::field_type::FieldType;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_double_to_string, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java `TextFieldSetting`.
#[derive(Clone, Debug)]
pub struct TextFieldSetting {
    /// Java private `set`, initialised to false.
    set: Cell<bool>,
    /// Java private `value`, initialised to null.
    value: RefCell<Option<String>>,
    /// Java private final `type`.
    r#type: Option<Type>,
}

impl TextFieldSetting {
    /// Java package-private `TextFieldSetting()`.
    pub fn new() -> TextFieldSetting {
        TextFieldSetting {
            set: Cell::new(false),
            value: RefCell::new(None),
            r#type: None,
        }
    }

    /// Java `TextFieldSetting(FieldType)`.
    pub fn new_field_type(field_type: FieldType) -> TextFieldSetting {
        TextFieldSetting {
            set: Cell::new(false),
            value: RefCell::new(None),
            r#type: field_type.get_numeric_type(),
        }
    }

    /// Java `TextFieldSetting(EtomoNumber.Type)`.
    pub fn new_type(r#type: Type) -> TextFieldSetting {
        TextFieldSetting {
            set: Cell::new(false),
            value: RefCell::new(None),
            r#type: Some(r#type),
        }
    }

    /// Java `equals(String)`.
    pub fn equals_string(&self, input: Option<&str>) -> bool {
        if !self.set.get() {
            return false;
        }
        let value = self.value.borrow().clone();
        if value.is_none() && input.is_none() {
            // Treating two nulls as equal
            return true;
        }
        let (Some(value), Some(input)) = (value.as_deref(), input) else {
            // One is null - not equal
            return false;
        };
        // Ignore whitespace
        let input = java_lang_string_trim(input);
        if value == input {
            // Strings are identical - equal
            return true;
        }
        // Compare as a number if both are numbers
        let n_value = self.create_number(value);
        if let Some(n_value) = n_value {
            let n_input = self.create_number(input);
            let Some(n_input) = n_input else {
                // One is numeric and the other is not
                return false;
            };
            // Treating two nulls as equal (value and input may not have the same type)
            if n_value.is_null() && n_input.is_null() {
                return true;
            }
            // One is null - not equal
            if n_value.is_null() || n_input.is_null() {
                return false;
            }
            // Do a numeric comparison
            return n_value.equals_const_etomo_number(Some(&n_input));
        }
        // Not numeric and not the same string - equals
        false
    }

    /// Java `equals(Number)`.
    pub fn equals_number(&self, input: Option<Number>) -> bool {
        if !self.set.get() {
            return false;
        }
        let value = self.value.borrow().clone();
        // Treating two nulls as equal
        if value.is_none() && input.is_none() {
            return true;
        }
        // One is null - not equal
        let (Some(value), Some(_)) = (value.as_deref(), input) else {
            return false;
        };
        // Compare as a number if value is a number
        let n_value = self.create_number(value);
        if let Some(n_value) = n_value {
            // Treating two nulls as equal (value and input may not have the same type)
            if n_value.is_null() && ConstEtomoNumber::is_number_null(input) {
                return true;
            }
            // One is null - not equal
            if n_value.is_null() || ConstEtomoNumber::is_number_null(input) {
                return false;
            }
            // Do a numeric comparison
            return n_value.equals_number(input);
        }
        // Value is not a number, and input is - not equal
        false
    }

    /// Java private `createNumber(String)`.  Returns a valid etomoNumber or null.  The
    /// type with be from the parameter, unless the string has a decimal point and type
    /// is long or integer.  In that case the type will be double.
    fn create_number(&self, string: &str) -> Option<ConstEtomoNumber> {
        let mut number = EtomoNumber::new_with_type(self.r#type);
        number.set_string(Some(string));
        if number.is_valid() {
            return Some(number.base);
        }
        if self.r#type != Some(Type::Double) {
            number = EtomoNumber::new_with_type(Some(Type::Double));
            number.set_string(Some(string));
            if number.is_valid() {
                return Some(number.base);
            }
        }
        None
    }

    /// Java `set(String)`.  Sets checkoint (trims whitespace).
    pub fn set_string(&self, input: Option<&str>) {
        self.set.set(true);
        *self.value.borrow_mut() = input.map(|input| input.to_string());
        // Ignore whitespace
        let value = self.value.borrow().clone();
        if let Some(value) = value {
            *self.value.borrow_mut() = Some(java_lang_string_trim(&value).to_string());
        }
    }

    /// Java `set(int)`.
    pub fn set_int(&self, input: i32) {
        self.set_string(Some(&input.to_string()));
    }

    /// Java `set(double)`: `String.valueOf(double)`.
    pub fn set_double(&self, input: f64) {
        self.set_string(Some(&java_lang_double_to_string(input)));
    }

    /// Java `set(ConstEtomoNumber)`.
    pub fn set_const_etomo_number(&self, input: Option<&ConstEtomoNumber>) {
        match input {
            None => {
                self.set.set(true);
                *self.value.borrow_mut() = None;
            }
            Some(input) => self.set_string(Some(&input.to_string())),
        }
    }

    /// Java `set(Number)`.
    pub fn set_number(&self, input: Option<Number>) {
        match input {
            None => {
                self.set.set(true);
                *self.value.borrow_mut() = None;
            }
            Some(input) => self.set_string(Some(&input.to_string())),
        }
    }

    /// Java `reset()`.
    pub fn reset(&self) {
        self.set.set(false);
        *self.value.borrow_mut() = None;
    }

    /// Java `copy(FieldSettingInterface)`.  Resets and then copies member variables.
    pub fn copy(&self, input: Option<&dyn FieldSettingInterface>) {
        self.reset();
        let mut setting: Option<TextFieldSetting> = None;
        if let Some(input) = input {
            setting = input.get_text_setting();
        }
        if let Some(setting) = setting {
            self.set.set(setting.set.get());
            *self.value.borrow_mut() = setting.value.into_inner();
        }
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Option<String> {
        self.value.borrow().clone()
    }
}

impl FieldSettingInterface for TextFieldSetting {
    /// Java `getBooleanSetting()`.
    fn get_boolean_setting(&self) -> Option<BooleanFieldSetting> {
        None
    }

    /// Java `getTextSetting()`: `this`.
    fn get_text_setting(&self) -> Option<TextFieldSetting> {
        Some(self.clone())
    }

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        false
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        true
    }

    /// Java `isSet()`.
    fn is_set(&self) -> bool {
        self.set.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equals_compares_strings_then_numbers() {
        let setting = TextFieldSetting::new_type(Type::Integer);
        assert!(!setting.equals_string(None));
        setting.set_string(Some(" 3 "));
        assert_eq!(setting.get_value().as_deref(), Some("3"));
        assert!(setting.equals_string(Some("3")));
        assert!(setting.equals_string(Some("3.0")));
        assert!(!setting.equals_string(Some("abc")));
        assert!(setting.equals_number(Some(Number::Integer(3))));
        let copy = TextFieldSetting::new();
        copy.copy(Some(&setting));
        assert!(copy.is_set());
        assert_eq!(copy.get_value().as_deref(), Some("3"));
    }
}
