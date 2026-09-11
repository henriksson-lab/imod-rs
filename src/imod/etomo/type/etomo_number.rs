//! `IMOD/Etomo/src/etomo/type/EtomoNumber.java`.
//!
//! Java's `EtomoNumber extends ConstEtomoNumber`.  Rust has no inheritance, so the
//! superclass state is held in the `base` field and reached through `Deref`/`DerefMut`;
//! every inherited member is therefore callable on an `EtomoNumber` exactly as in Java.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::storage::storable::Storable;

/// Java `EtomoNumber`.
#[derive(Clone, Debug)]
pub struct EtomoNumber {
    /// Java superclass `ConstEtomoNumber` state.
    pub base: ConstEtomoNumber,
}

/// Java inheritance: every `ConstEtomoNumber` member is reachable on an `EtomoNumber`.
impl std::ops::Deref for EtomoNumber {
    type Target = ConstEtomoNumber;

    fn deref(&self) -> &ConstEtomoNumber {
        &self.base
    }
}

impl std::ops::DerefMut for EtomoNumber {
    fn deref_mut(&mut self) -> &mut ConstEtomoNumber {
        &mut self.base
    }
}

impl EtomoNumber {
    /// Java `EtomoNumber()`.  Construct an EtomoNumber with type = INTEGER_TYPE.
    pub fn new() -> EtomoNumber {
        EtomoNumber {
            base: ConstEtomoNumber::new(),
        }
    }

    /// Java `EtomoNumber(Number)`.  Set the type and value from number.
    pub fn new_from_number(number: Option<Number>) -> EtomoNumber {
        let mut instance = EtomoNumber {
            base: ConstEtomoNumber::new_with_type(Type::get_type(number)),
        };
        instance.set_number(number);
        instance
    }

    /// Java `EtomoNumber(String)`.  Construct a ConstEtomoNumber with
    /// type = INTEGER_TYPE; the parameter is the name of the instance.
    pub fn new_with_name(name: &str) -> EtomoNumber {
        EtomoNumber {
            base: ConstEtomoNumber::new_with_name(name),
        }
    }

    /// Java `EtomoNumber(Type)`.
    pub fn new_with_type(r#type: Option<Type>) -> EtomoNumber {
        EtomoNumber {
            base: ConstEtomoNumber::new_with_type(r#type),
        }
    }

    /// Java `EtomoNumber(Type, String)`.
    pub fn new_with_type_and_name(r#type: Type, name: &str) -> EtomoNumber {
        EtomoNumber {
            base: ConstEtomoNumber::new_with_type_and_name(r#type, name),
        }
    }

    /// Java `EtomoNumber(ConstEtomoNumber)`.
    pub fn new_from_instance(that: Option<&ConstEtomoNumber>) -> EtomoNumber {
        EtomoNumber {
            base: ConstEtomoNumber::new_from_instance(that),
        }
    }

    /// Java `loadWithAlternateKey`.
    pub fn load_with_alternate_key(
        &mut self,
        props: Option<&BTreeMap<String, String>>,
        prepend: Option<&str>,
        key: Option<&str>,
    ) {
        let props = match props {
            None => {
                self.reset();
                return;
            }
            Some(props) => props,
        };
        let prepend_blank = match prepend {
            None => true,
            Some(prepend) => java_lang_string_matches_whitespace(prepend),
        };
        let key_blank = match key {
            None => true,
            Some(key) => java_lang_string_matches_whitespace(key),
        };
        if prepend_blank && key_blank {
            self.load(props);
        } else if prepend_blank {
            let value = props.get(key.unwrap()).cloned();
            self.set_string(value.as_deref());
        } else if key_blank {
            self.load_with_prepend(props, prepend);
        } else {
            let value = props
                .get(&format!("{}.{}", prepend.unwrap(), key.unwrap()))
                .cloned();
            self.set_string(value.as_deref());
        }
    }

    /// Java `load(Properties)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>) {
        let value = props.get(&self.base.name).cloned();
        self.set_string(value.as_deref());
    }

    /// Java `createKey`.
    fn create_key(&self, prepend: Option<&str>, key: &str) -> String {
        match prepend {
            None => key.to_string(),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => key.to_string(),
            Some(prepend) if prepend.ends_with('.') => format!("{}{}", prepend, key),
            Some(prepend) => format!("{}.{}", prepend, key),
        }
    }

    /// Java `loadFromOtherKey`.
    pub(crate) fn load_from_other_key(
        &mut self,
        props: Option<&BTreeMap<String, String>>,
        prepend: Option<&str>,
        key: &str,
    ) {
        match props {
            None => {
                self.reset();
            }
            Some(props) => {
                let value = props.get(&self.create_key(prepend, key)).cloned();
                self.set_string(value.as_deref());
            }
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        match prepend {
            None => self.load(props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.load(props),
            Some(prepend) if prepend.ends_with('.') => {
                let value = props
                    .get(&format!("{}{}", prepend, self.base.name))
                    .cloned();
                self.set_string(value.as_deref());
            }
            Some(prepend) => {
                let value = props
                    .get(&format!("{}.{}", prepend, self.base.name))
                    .cloned();
                self.set_string(value.as_deref());
            }
        }
    }

    /// Java `load(Properties, String, int)`.
    pub fn load_with_default_int(
        &mut self,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
        default_value: i32,
    ) {
        if self.load_if_present(props, prepend) {
            return;
        }
        self.set_int(default_value);
    }

    /// Java `load(Properties, String, boolean)`.
    pub fn load_with_default_boolean(
        &mut self,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
        default_value: bool,
    ) {
        if self.load_if_present(props, prepend) {
            return;
        }
        self.set_boolean(default_value);
    }

    /// Java `isKeyPresent`.
    pub fn is_key_present(&self, props: &BTreeMap<String, String>, prepend: Option<&str>) -> bool {
        let key = match prepend {
            None => self.base.name.clone(),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.base.name.clone(),
            Some(prepend) => format!("{}.{}", prepend, self.base.name),
        };
        if props.get(&key).is_none() {
            return false;
        }
        true
    }

    /// Java `loadIfPresent`.
    pub fn load_if_present(
        &mut self,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> bool {
        let key = match prepend {
            None => self.base.name.clone(),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.base.name.clone(),
            Some(prepend) => format!("{}.{}", prepend, self.base.name),
        };
        if props.get(&key).is_none() {
            return false;
        }
        self.load_with_prepend(props, prepend);
        true
    }

    /// Java `load(EtomoNumber, Type, String, Properties, String)`, the static overload.
    /// Attempt to get the property specified by prepend and name.  If it doesn't exist,
    /// return null.  If it does, set it in instance (create instance if it doesn't
    /// exist).
    pub fn load_instance_with_type(
        instance: Option<EtomoNumber>,
        r#type: Type,
        name: &str,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> Option<EtomoNumber> {
        let key = match prepend {
            None => name.to_string(),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => name.to_string(),
            Some(prepend) => format!("{}.{}", prepend, name),
        };
        let value = match props.get(&key) {
            None => return None,
            Some(value) => value.clone(),
        };
        let mut instance = match instance {
            None => EtomoNumber::new_with_type_and_name(r#type, name),
            Some(instance) => instance,
        };
        instance.set_string(Some(&value));
        Some(instance)
    }

    /// Java `load(EtomoNumber, String, Properties, String)`, the static overload.
    pub fn load_instance(
        instance: Option<EtomoNumber>,
        name: &str,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> Option<EtomoNumber> {
        let key = match prepend {
            None => name.to_string(),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => name.to_string(),
            Some(prepend) => format!("{}.{}", prepend, name),
        };
        let value = match props.get(&key) {
            None => return None,
            Some(value) => value.clone(),
        };
        let mut instance = match instance {
            None => EtomoNumber::new_with_name(name),
            Some(instance) => instance,
        };
        instance.set_string(Some(&value));
        Some(instance)
    }

    /// Java `set(String)`.  Converts a string to a Number of the correct type.  If the
    /// string is empty, currentValue will be set to `newNumber()`.  If the string is not
    /// a valid number of the correct type, currentValue will be set to `newNumber()`.
    /// If ceilingValue is not null and the new value exceeds ceilingValue, currentValue
    /// will be set to ceilingValue.
    pub fn set_string(&mut self, value: Option<&str>) -> &mut EtomoNumber {
        if self.base.is_debug() {
            println!(
                "value={}",
                match value {
                    None => "null",
                    Some(value) => value,
                }
            );
        }
        self.base.reset_state();
        let blank = match value {
            None => true,
            Some(value) => java_lang_string_matches_whitespace(value),
        };
        if blank {
            self.base.current_value = self.base.new_number();
        } else {
            let mut invalid_buffer = String::new();
            let number = self.base.new_number_from_string(value, &mut invalid_buffer);
            let number = self.base.apply_floor_value(Some(number));
            let number = self.base.apply_ceiling_value(number);
            self.base.current_value = self.base.new_number_from_number(number);
            if self.base.is_debug() {
                println!(
                    "currentValue={},invalidBuffer={}",
                    self.base.current_value, invalid_buffer
                );
            }
            if !invalid_buffer.is_empty() {
                self.base.add_invalid_reason(Some(&invalid_buffer.clone()));
            } else {
                self.base.set_invalid_reason();
            }
        }
        self
    }

    /// Java `setToDefault`.
    pub fn set_to_default(&mut self) {
        let default_value = self.base.default_value;
        self.set_number(Some(default_value));
    }

    /// Java `set(Number)`.
    pub fn set_number(&mut self, value: Option<Number>) -> &mut EtomoNumber {
        self.base.reset_state();
        let number = self.base.apply_floor_value(value);
        let number = self.base.apply_ceiling_value(number);
        self.base.current_value = self.base.new_number_from_number(number);
        self.base.set_invalid_reason();
        self
    }

    /// Java `set(ConstEtomoNumber)`.  Set currentValue from `number.getValue()`.  Take
    /// the currentValueSet value from number.
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
        &mut self.base
    }

    /// Java `add(ConstEtomoNumber)`.
    pub fn add_const_etomo_number(&mut self, number: Option<&ConstEtomoNumber>) {
        let number = match number {
            None => return,
            Some(number) => number,
        };
        let sum = self
            .base
            .add(Some(self.base.get_value()), Some(number.get_value()));
        self.set_number(Some(sum));
    }

    /// Java `add(int)`.
    pub fn add_int(&mut self, i: i32) {
        let sum = self.base.add(
            Some(self.base.get_value()),
            Some(self.base.new_number_from_int(i)),
        );
        self.set_number(Some(sum));
    }

    /// Java `add(long)`.
    pub fn add_long(&mut self, l: i64) {
        let sum = self.base.add(
            Some(self.base.get_value()),
            Some(self.base.new_number_from_long(l)),
        );
        self.set_number(Some(sum));
    }

    /// Java `increment`.
    pub fn increment(&mut self) {
        let sum = self.base.add(
            Some(self.base.get_value()),
            Some(self.base.new_number_from_int(1)),
        );
        self.set_number(Some(sum));
    }

    /// Java `decrement`.
    pub fn decrement(&mut self) {
        let difference = self.base.subtract(
            Some(self.base.get_value()),
            Some(self.base.new_number_from_int(1)),
        );
        self.set_number(Some(difference));
    }

    /// Java `multiply(int)`.  Multiply the current value by i and store the result as
    /// the current value.
    pub fn multiply_int(&mut self, i: i32) {
        let product = self.base.multiply(
            Some(self.base.get_value()),
            Some(self.base.new_number_from_int(i)),
        );
        self.set_number(Some(product));
    }

    /// Java `divideBy(int)`.  Divide the current value by i and store the result as the
    /// current value.
    pub fn divide_by_int(&mut self, i: i32) {
        let quotient = self.base.divide_by(
            Some(self.base.get_value()),
            Some(self.base.new_number_from_int(i)),
        );
        self.set_number(Some(quotient));
    }

    /// Java `set(int)`.
    pub fn set_int(&mut self, value: i32) -> &mut EtomoNumber {
        let number = self.base.new_number_from_int(value);
        self.set_number(Some(number))
    }

    /// Java `set(boolean)`.
    pub fn set_boolean(&mut self, value: bool) -> &mut ConstEtomoNumber {
        let number = self.base.new_number_from_boolean(value);
        self.set_number(Some(number));
        &mut self.base
    }

    /// Java `set(long)`.
    pub fn set_long(&mut self, value: i64) -> &mut EtomoNumber {
        let number = self.base.new_number_from_long(value);
        self.set_number(Some(number))
    }

    /// Java `set(double)`.
    pub fn set_double(&mut self, value: f64) -> &mut EtomoNumber {
        let number = self.base.new_number_from_double(value);
        self.set_number(Some(number))
    }

    /// Java `set(FortranInputString, int)`.
    pub fn set_fortran_input_string(
        &mut self,
        fortran_input_string: &FortranInputString,
        index: i32,
    ) -> &mut EtomoNumber {
        if fortran_input_string.is_empty_index(index)
            || fortran_input_string.is_default_index(index)
        {
            let number = self.base.new_number();
            return self.set_number(Some(number));
        }
        if fortran_input_string.is_integer_type(index) {
            let number = self
                .base
                .new_number_from_int(fortran_input_string.get_int(index));
            return self.set_number(Some(number));
        }
        let number = self
            .base
            .new_number_from_double(fortran_input_string.get_double_index(index));
        self.set_number(Some(number))
    }

    /// Java `reset`.  Sets currentValue to null.
    pub fn reset(&mut self) -> &mut EtomoNumber {
        self.base.reset_state();
        self.base.current_value = self.base.new_number();
        self.base.set_invalid_reason();
        self
    }
}

/// Java `EtomoNumber implements Storable` (through `ConstEtomoNumber`): `store` and
/// `store(prepend)` are inherited from the superclass, `load` and `load(prepend)` are
/// declared here.
impl Storable for EtomoNumber {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.base.store(properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.store_with_prepend(properties, Some(prepend));
    }

    fn load(&mut self, properties: &BTreeMap<String, String>) {
        EtomoNumber::load(self, properties);
    }

    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        EtomoNumber::load_with_prepend(self, properties, Some(prepend));
    }
}

/// Java `toString()`, inherited from `ConstEtomoNumber`.
impl std::fmt::Display for EtomoNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.base, f)
    }
}
