//! `IMOD/Etomo/src/etomo/type/EtomoState.java`.
//!
//! A tri-state flag - no result, false, true - stored as "no result"/"false"/"true".
//!
//! Java's `EtomoState extends EtomoNumber`.  As in `etomo_boolean2.rs`, the superclass
//! state is the `base` field reached through `Deref`/`DerefMut`, and the inherited
//! methods whose bodies make a virtual call this class overrides are written out here
//! with that call resolved to the override: `set(String)` and `load` (which reach the
//! `newNumber(String, StringBuffer)` and `setInvalidReason()` overrides), the numeric
//! `set` overloads (which reach `setInvalidReason()`), and `toString()` (which reaches
//! `toString(Number)`).  Calling a setter through `.base` skips the overrides, as
//! `super.set(...)` would.
//!
//! **`setInvalidReason()`.**  The Java override throws `IllegalArgumentException` for
//! any value outside {-1, 0, 1}.  No caller catches it, so a data file holding, say,
//! `ReconstructionState.MadeZFactorsA=2` aborted the whole load.  Upstream bug fixed in
//! translation (EtomoState.java:84-89): the message is printed to standard error and
//! the value is reset to null (unset), which every caller already handles.

use std::collections::BTreeMap;

use super::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, Number, java_lang_string_matches_whitespace,
    java_lang_string_trim,
};
use super::etomo_number::EtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java `NO_RESULT_VALUE`.
pub const NO_RESULT_VALUE: i32 = -1;
/// Java `FALSE_VALUE`.
pub const FALSE_VALUE: i32 = 0;
/// Java `TRUE_VALUE`.
pub const TRUE_VALUE: i32 = 1;
/// Java `NO_RESULT_STRING`.
pub const NO_RESULT_STRING: &str = "no result";
/// Java private static `nullString`.
const NULL_STRING: &str = "null";
/// Java private static `falseString`.
const FALSE_STRING: &str = "false";
/// Java private static `trueString`.
const TRUE_STRING: &str = "true";

/// Java `EtomoState`.
#[derive(Clone, Debug)]
pub struct EtomoState {
    /// Java superclass `EtomoNumber` state.
    pub base: EtomoNumber,
}

/// Java inheritance: every `EtomoNumber` member is reachable on an `EtomoState`.
impl std::ops::Deref for EtomoState {
    type Target = EtomoNumber;

    fn deref(&self) -> &EtomoNumber {
        &self.base
    }
}

impl std::ops::DerefMut for EtomoState {
    fn deref_mut(&mut self) -> &mut EtomoNumber {
        &mut self.base
    }
}

impl EtomoState {
    /// Java `EtomoState(String)`.  `setValidValues` makes a virtual
    /// `setInvalidReason()` call; the current value is null at construction and null is
    /// valid, so the override has nothing to report and the superclass's body is used.
    pub fn new_with_name(name: &str) -> EtomoState {
        let mut instance = EtomoState {
            base: EtomoNumber::new_with_name(name),
        };
        instance
            .base
            .set_valid_values(Some(&[NO_RESULT_VALUE, FALSE_VALUE, TRUE_VALUE]));
        instance
    }

    /// Java `EtomoState()`.
    pub fn new() -> EtomoState {
        let mut instance = EtomoState {
            base: EtomoNumber::new(),
        };
        instance
            .base
            .set_valid_values(Some(&[NO_RESULT_VALUE, FALSE_VALUE, TRUE_VALUE]));
        instance
    }

    /// Java package-private `setInvalidReason()`, overriding `ConstEtomoNumber`:
    /// `super.setInvalidReason()`, then throw if a reason was found.  See the module
    /// header for the fixed throw.
    pub(crate) fn set_invalid_reason(&mut self) {
        self.base.base.set_invalid_reason();
        if let Some(invalid_reason) = self.base.base.invalid_reason.clone() {
            eprintln!("java.lang.IllegalArgumentException: {}", invalid_reason);
            self.base.reset();
        }
    }

    /// Java `is()`, overriding `ConstEtomoNumber.is()`.  Convert to true/false.
    pub fn is(&self) -> bool {
        let int_value = self.base.base.get_value().int_value();
        if int_value == NO_RESULT_VALUE {
            return false;
        }
        self.base.base.is()
    }

    /// Java `isResultSet`.
    pub fn is_result_set(&self) -> bool {
        self.base.base.current_value.int_value() != NO_RESULT_VALUE
    }

    /// Java package-private `toString(Number)`, overriding `ConstEtomoNumber`.
    pub(crate) fn to_string_number(&self, value: Number) -> String {
        let int_value = value.int_value();
        match int_value {
            INTEGER_NULL_VALUE => NULL_STRING.to_string(),
            NO_RESULT_VALUE => NO_RESULT_STRING.to_string(),
            FALSE_VALUE => FALSE_STRING.to_string(),
            TRUE_VALUE => TRUE_STRING.to_string(),
            _ => self.base.base.to_string_number(Some(value)),
        }
    }

    /// Java package-private `newNumber(String, StringBuffer)`, overriding
    /// `ConstEtomoNumber`.  Convert from a character string to the value.
    ///
    /// EtomoState.java:122 falls through to `return newNumber(value, invalidBuffer)`,
    /// which calls this same override, so any string other than the four keywords
    /// recurses until `StackOverflowError`.  Fixed in translation: the fall-through
    /// calls the superclass `newNumber(String, StringBuffer)`, which is what the
    /// override plainly means to delegate to.
    pub(crate) fn new_number_from_string(
        &self,
        value: Option<&str>,
        invalid_buffer: &mut String,
    ) -> Number {
        if let Some(value) = value {
            if !java_lang_string_matches_whitespace(value) {
                // Convert from character string to integer value
                let trimmed_value = java_lang_string_trim(value).to_lowercase();
                if trimmed_value == NULL_STRING {
                    return self.base.base.new_number_from_int(INTEGER_NULL_VALUE);
                }
                if trimmed_value == NO_RESULT_STRING {
                    return self.base.base.new_number_from_int(NO_RESULT_VALUE);
                }
                if trimmed_value == FALSE_STRING {
                    return self.base.base.new_number_from_int(FALSE_VALUE);
                }
                if trimmed_value == TRUE_STRING {
                    return self.base.base.new_number_from_int(TRUE_VALUE);
                }
            }
        }
        self.base.base.new_number_from_string(value, invalid_buffer)
    }

    /// Java inherited `EtomoNumber.set(String)`, whose `newNumber(String,
    /// StringBuffer)` and `setInvalidReason()` calls resolve to this class's overrides.
    pub fn set_string(&mut self, value: Option<&str>) -> &mut EtomoState {
        if self.base.base.is_debug() {
            println!("value={}", value.unwrap_or("null"));
        }
        self.base.base.reset_state();
        let blank = match value {
            None => true,
            Some(value) => java_lang_string_matches_whitespace(value),
        };
        if blank {
            self.base.base.current_value = self.base.base.new_number();
        } else {
            let mut invalid_buffer = String::new();
            let number = self.new_number_from_string(value, &mut invalid_buffer);
            let number = self.base.base.apply_floor_value(Some(number));
            let number = self.base.base.apply_ceiling_value(number);
            self.base.base.current_value = self.base.base.new_number_from_number(number);
            if self.base.base.is_debug() {
                println!(
                    "currentValue={},invalidBuffer={}",
                    self.base.base.current_value, invalid_buffer
                );
            }
            if !invalid_buffer.is_empty() {
                self.base
                    .base
                    .add_invalid_reason(Some(&invalid_buffer.clone()));
            } else {
                self.set_invalid_reason();
            }
        }
        self
    }

    /// Java inherited `EtomoNumber.set(Number)`, whose `setInvalidReason()` call
    /// resolves to this class's override.
    pub fn set_number(&mut self, value: Option<Number>) -> &mut EtomoState {
        self.base.set_number(value);
        // `setInvalidReason()` override: the superclass body ran inside the inherited
        // setter; this is the override's throw (see the module header).
        if let Some(invalid_reason) = self.base.base.invalid_reason.clone() {
            eprintln!("java.lang.IllegalArgumentException: {}", invalid_reason);
            self.base.reset();
        }
        self
    }

    /// Java inherited `EtomoNumber.set(int)`.
    pub fn set_int(&mut self, value: i32) -> &mut EtomoState {
        self.base.set_int(value);
        // `setInvalidReason()` override: the superclass body ran inside the inherited
        // setter; this is the override's throw (see the module header).
        if let Some(invalid_reason) = self.base.base.invalid_reason.clone() {
            eprintln!("java.lang.IllegalArgumentException: {}", invalid_reason);
            self.base.reset();
        }
        self
    }

    /// Java inherited `EtomoNumber.set(boolean)`.
    pub fn set_boolean(&mut self, value: bool) -> &mut EtomoState {
        self.base.set_boolean(value);
        // `setInvalidReason()` override: the superclass body ran inside the inherited
        // setter; this is the override's throw (see the module header).
        if let Some(invalid_reason) = self.base.base.invalid_reason.clone() {
            eprintln!("java.lang.IllegalArgumentException: {}", invalid_reason);
            self.base.reset();
        }
        self
    }

    /// Java inherited `EtomoNumber.set(ConstEtomoNumber)`.
    pub fn set_const_etomo_number(&mut self, number: Option<&ConstEtomoNumber>) -> &mut EtomoState {
        self.base.set_const_etomo_number(number);
        // `setInvalidReason()` override: the superclass body ran inside the inherited
        // setter; this is the override's throw (see the module header).
        if let Some(invalid_reason) = self.base.base.invalid_reason.clone() {
            eprintln!("java.lang.IllegalArgumentException: {}", invalid_reason);
            self.base.reset();
        }
        self
    }

    /// Java inherited `EtomoNumber.load(Properties)`, whose `set(String)` call
    /// resolves to this class's override.
    pub fn load(&mut self, props: &BTreeMap<String, String>) {
        let value = props.get(&self.base.base.name).cloned();
        self.set_string(value.as_deref());
    }

    /// Java inherited `EtomoNumber.load(Properties, String)`, whose `set(String)` calls
    /// resolve to this class's override.
    pub fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        match prepend {
            None => self.load(props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.load(props),
            Some(prepend) if prepend.ends_with('.') => {
                let value = props
                    .get(&format!("{}{}", prepend, self.base.base.name))
                    .cloned();
                self.set_string(value.as_deref());
            }
            Some(prepend) => {
                let value = props
                    .get(&format!("{}.{}", prepend, self.base.base.name))
                    .cloned();
                self.set_string(value.as_deref());
            }
        }
    }

    /// Java `store(Properties)`, overriding `ConstEtomoNumber`.  An unset state is
    /// stored as "null", not removed.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        props.insert(
            self.base.base.name.clone(),
            self.to_string_number(self.base.base.current_value),
        );
    }

    /// Java `store(Properties, String)`, overriding `ConstEtomoNumber`.  The key is
    /// `prepend + "." + name` whatever the prepend.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        props.insert(
            format!("{}.{}", prepend, self.base.base.name),
            self.to_string_number(self.base.base.current_value),
        );
    }
}

impl Default for EtomoState {
    fn default() -> EtomoState {
        EtomoState::new()
    }
}

/// Java inherited `ConstEtomoNumber.toString()`: `toString(getValue())`, whose
/// `toString(Number)` call resolves to this class's override.
impl std::fmt::Display for EtomoState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_number(self.base.base.get_value()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_round_trip() {
        let mut state = EtomoState::new_with_name("TrimvolFlipped");
        assert_eq!(state.to_string(), "null");
        let mut props = BTreeMap::new();
        state.store_with_prepend(&mut props, "ReconstructionState");
        assert_eq!(
            props.get("ReconstructionState.TrimvolFlipped").unwrap(),
            "null"
        );
        state.set_int(NO_RESULT_VALUE);
        assert!(!state.is());
        assert!(!state.is_result_set());
        state.store_with_prepend(&mut props, "ReconstructionState");
        assert_eq!(
            props.get("ReconstructionState.TrimvolFlipped").unwrap(),
            "no result"
        );
        let mut loaded = EtomoState::new_with_name("TrimvolFlipped");
        props.insert(
            "ReconstructionState.TrimvolFlipped".to_string(),
            " TRUE ".to_string(),
        );
        loaded.load_with_prepend(&props, Some("ReconstructionState"));
        assert!(loaded.is());
        assert!(loaded.is_result_set());
        assert_eq!(loaded.to_string(), "true");
    }

    #[test]
    fn invalid_value_resets_instead_of_throwing() {
        let mut state = EtomoState::new_with_name("X");
        state.set_int(2);
        assert!(state.is_null());
        state.set_string(Some("7"));
        assert!(state.is_null());
    }
}
