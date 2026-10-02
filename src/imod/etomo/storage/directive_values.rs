//! `IMOD/Etomo/src/etomo/storage/DirectiveValues.java`.
//!
//! The value and default value of one `Directive`.  Overloads carry the suffix of their
//! parameter types (`setValue(int)` is `set_value_int`, ...).  `Value` and its
//! subclasses are `storage::directive::Value`.

use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::directive::{Value, ValueFactory};
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::const_string_parameter::ConstStringParameter;
use crate::imod::etomo::r#type::debug_level::DebugLevel;

/// Java final `DirectiveValues`.
pub struct DirectiveValues {
    /// Java package-private final field `valueType`.
    pub(crate) value_type: Option<DirectiveValueType>,
    /// Java private field `defaultValue`, initialised to null.
    default_value: Option<Value>,
    /// Java private field `value`, initialised to null.
    value: Option<Value>,
    /// Java private field `debug`, initialised from the arguments' debug level.
    debug: DebugLevel,
}

impl DirectiveValues {
    /// Java package-private `DirectiveValues(DirectiveValueType)`.
    pub(crate) fn new(value_type: Option<DirectiveValueType>) -> DirectiveValues {
        DirectiveValues {
            value_type,
            default_value: None,
            value: None,
            debug: etomo_director::ARGUMENTS.lock().unwrap().get_debug_level(),
        }
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Java `getDefaultValue()`.
    pub fn get_default_value(&self) -> Option<&Value> {
        self.default_value.as_ref()
    }

    /// Java `isSet()`.
    pub fn is_set(&self) -> bool {
        self.value.is_some()
    }

    /// Java `isChanged()`.  Returns true if any of the values when changed from their
    /// coresponding default values.  If an A or B value has not been set, compare the
    /// value to non-axis default value.
    pub fn is_changed(&self) -> bool {
        self.is_changed_value_value(self.value.as_ref(), self.default_value.as_ref())
    }

    /// Java `equals(Value, Value)`.  Assumes that the two values have the same type.
    pub fn equals(&self, value1: Option<&Value>, value2: Option<&Value>) -> bool {
        let (mut value1, mut value2) = (value1, value2);
        if self.debug.is_extra() {
            eprintln!(
                "equals:value1:{},value2:{}",
                match value1 {
                    None => "null".to_string(),
                    Some(value) => value.to_string(),
                },
                match value2 {
                    None => "null".to_string(),
                    Some(value) => value.to_string(),
                }
            );
            // DirectiveValues.java:73 calls `value1.setDebug(debug)` before the null
            // checks, a NullPointerException when value1 is null.  Fixed in
            // translation: the debug level is set only on a non-null value1.
            if let Some(value1) = value1 {
                value1.set_debug(self.debug);
            }
        }
        if value1.is_none() && value2.is_none() {
            return true;
        }
        if value1.is_none() {
            // Value2 is not null - swap them to do the comparison.
            value1 = value2;
            value2 = None;
        }
        let value1 = value1.unwrap();
        // The casts cannot fail: both values were made by `ValueFactory` for this
        // `valueType`.
        if self.value_type == Some(DirectiveValueType::Boolean) {
            if let Value::Boolean(value1) = value1 {
                return value1.equals_boolean_value(match value2 {
                    Some(Value::Boolean(value2)) => Some(value2),
                    _ => None,
                });
            }
        }
        if self.value_type == Some(DirectiveValueType::FloatingPoint)
            || self.value_type == Some(DirectiveValueType::Integer)
        {
            if let Value::Numeric(value1) = value1 {
                return value1.equals_numeric_value(match value2 {
                    Some(Value::Numeric(value2)) => Some(value2),
                    _ => None,
                });
            }
        }
        if self.value_type == Some(DirectiveValueType::FloatingPointPair)
            || self.value_type == Some(DirectiveValueType::IntegerPair)
        {
            if let Value::NumericPair(value1) = value1 {
                return value1.equals_numeric_pair_value(match value2 {
                    Some(Value::NumericPair(value2)) => Some(value2),
                    _ => None,
                });
            }
        }
        if self.value_type == Some(DirectiveValueType::List)
            || self.value_type == Some(DirectiveValueType::String)
        {
            if let Value::String(value1) = value1 {
                return value1.equals_string_value(match value2 {
                    Some(Value::String(value2)) => Some(value2),
                    _ => None,
                });
            }
        }
        // UNKNOWN and FILE (and a null valueType) hold a StringValue but are not
        // compared: the source returns false, so such a set value always reads as
        // changed.
        false
    }

    /// Java private `isChanged(Value, Value)`.  Returns false if value is null, because
    /// that means it hasn't been set.  Otherwise returns !equals.
    fn is_changed_value_value(&self, value: Option<&Value>, default_value: Option<&Value>) -> bool {
        if self.debug.is_extra() {
            eprintln!(
                "isChanged:value:{},defaultValue:{}",
                match value {
                    None => "null".to_string(),
                    Some(value) => value.to_string(),
                },
                match default_value {
                    None => "null".to_string(),
                    Some(value) => value.to_string(),
                }
            );
        }
        let value = match value {
            None => return false,
            Some(value) => value,
        };
        if self.debug.is_extra() {
            value.set_debug(self.debug);
        }
        !self.equals(Some(value), default_value)
    }

    /// Java `setDebug(DebugLevel)`.
    pub fn set_debug(&mut self, input: DebugLevel) {
        self.debug = input;
    }

    /// Java `resetDebug()`.
    pub fn reset_debug(&mut self) {
        self.debug = etomo_director::ARGUMENTS.lock().unwrap().get_debug_level();
    }

    /// Java package-private `setDefaultValue(boolean)`.
    pub(crate) fn set_default_value_boolean(&mut self, input: bool) {
        self.create_default_value();
        self.default_value.as_mut().unwrap().set_boolean(input);
    }

    /// Java package-private `setDefaultValue(String)`.
    pub(crate) fn set_default_value_string(&mut self, input: Option<&str>) {
        // DirectiveValues.java:134 tests `input.matches("\\ss*")` (one whitespace
        // character followed by any number of 's'), a typo for the `"\\s*"` every other
        // setter here uses: native stores an empty or blank default as a set value.
        // Fixed in translation: an empty or blank default is null, as in
        // `setValue(String)`.
        match input {
            Some(input) if !java_lang_string_matches_whitespace(input) => {
                self.create_default_value();
                self.default_value.as_mut().unwrap().set_string(Some(input));
            }
            _ => self.default_value = None,
        }
    }

    /// Java package-private `setDefaultValue(int)`.
    pub(crate) fn set_default_value_int(&mut self, input: i32) {
        if input == INTEGER_NULL_VALUE {
            self.default_value = None;
        } else {
            self.create_default_value();
            self.default_value.as_mut().unwrap().set_int(input);
        }
    }

    /// Java package-private `setDefaultValue(ConstEtomoNumber)`.
    pub(crate) fn set_default_value_const_etomo_number(
        &mut self,
        input: Option<&ConstEtomoNumber>,
    ) {
        // DirectiveValues.java:154 calls `input.isNull()` on a possibly null input
        // (NullPointerException).  Fixed in translation: a null input is treated like a
        // null number, as `setValue(ConstEtomoNumber)` does.
        match input {
            Some(input) if !input.is_null() => {
                self.create_default_value();
                self.default_value
                    .as_mut()
                    .unwrap()
                    .set_const_etomo_number(Some(input));
            }
            _ => self.default_value = None,
        }
    }

    /// Java package-private `resetValue()`.
    pub(crate) fn reset_value(&mut self) {
        self.value = None;
    }

    /// Java package-private `setValue(boolean)`.
    pub(crate) fn set_value_boolean(&mut self, input: bool) {
        self.create_value();
        self.value.as_mut().unwrap().set_boolean(input);
    }

    /// Java package-private `setValue(ConstEtomoNumber)`.
    pub(crate) fn set_value_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        match input {
            Some(input) if !input.is_null() => {
                self.create_value();
                self.value
                    .as_mut()
                    .unwrap()
                    .set_const_etomo_number(Some(input));
            }
            _ => self.value = None,
        }
    }

    /// Java package-private `setValue(ConstStringParameter)`.
    pub(crate) fn set_value_const_string_parameter(
        &mut self,
        input: Option<&dyn ConstStringParameter>,
    ) {
        match input {
            Some(input) if !input.is_empty() => {
                self.create_value();
                self.value
                    .as_mut()
                    .unwrap()
                    .set_const_string_parameter(Some(input));
            }
            _ => self.value = None,
        }
    }

    /// Java package-private `setValue(double)`.
    pub(crate) fn set_value_double(&mut self, input: f64) {
        // DirectiveValues.java:193 tests `input == EtomoNumber.DOUBLE_NULL_VALUE`, which
        // is NaN and never equal, so native creates a value from a null double.  Fixed
        // in translation: a null (NaN) double unsets the value, as the source intends.
        if input.is_nan() {
            self.value = None;
        } else {
            self.create_value();
            self.value.as_mut().unwrap().set_double(input);
        }
    }

    /// Java package-private `setValue(double[])`.
    pub(crate) fn set_value_double_array(&mut self, input: Option<&[f64]>) {
        match input {
            None => self.value = None,
            Some(input) => {
                self.create_value();
                self.value.as_mut().unwrap().set_double_array(Some(input));
            }
        }
    }

    /// Java package-private `setValue(FortranInputString)`.
    pub(crate) fn set_value_fortran_input_string(&mut self, input: Option<&FortranInputString>) {
        match input {
            None => self.value = None,
            Some(input) => {
                self.create_value();
                self.value
                    .as_mut()
                    .unwrap()
                    .set_fortran_input_string(Some(input));
            }
        }
    }

    /// Java package-private `setValue(int)`.
    pub(crate) fn set_value_int(&mut self, input: i32) {
        // DirectiveValues.java:222-227 sets `value = null` for the null integer and then,
        // with no `else`, creates the value again and sets it, so the null never sticks
        // (every other setter here has the `else`).  Fixed in translation: the null
        // integer unsets the value.
        if input == INTEGER_NULL_VALUE {
            self.value = None;
        } else {
            self.create_value();
            self.value.as_mut().unwrap().set_int(input);
        }
    }

    /// Java package-private `setValue(String)`.
    pub(crate) fn set_value_string(&mut self, input: Option<&str>) {
        match input {
            Some(input) if !java_lang_string_matches_whitespace(input) => {
                self.create_value();
                self.value.as_mut().unwrap().set_string(Some(input));
            }
            _ => self.value = None,
        }
    }

    /// Java private `createDefaultValue()`.
    fn create_default_value(&mut self) {
        if self.default_value.is_none() {
            self.default_value = Some(ValueFactory::get_value(self.value_type));
        }
    }

    /// Java private `createValue()`.
    fn create_value(&mut self) {
        if self.value.is_none() {
            self.value = Some(ValueFactory::get_value(self.value_type));
        }
    }
}
