//! `IMOD/Etomo/src/etomo/logic/ValidationSet.java`.
//!
//! This is deliberately independent of the widget implementation: Java keeps the
//! validation data in `logic`, then shares it between `TextField` implementations.

use crate::imod::etomo::r#type::const_etomo_number::{
    Type, java_lang_double_value_of, java_lang_integer_parse_int, java_lang_long_parse_long,
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::ui::field_type::FieldType;

/// Java final `ValidationSet`.  Bounds are represented as doubles because the Java
/// `EtomoNumber` comparisons coerce their string operand to the stored numeric type.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationSet {
    pub field_type: Option<FieldType>,
    pub numeric_type: Option<Type>,
    pub descr: Option<String>,
    pub zero: Option<f64>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub parsable_string: bool,
}

impl ValidationSet {
    pub fn new(field_type: Option<FieldType>, descr: Option<String>) -> Self {
        Self {
            numeric_type: field_type.and_then(FieldType::get_numeric_type),
            field_type,
            descr,
            ..Self::default()
        }
    }
    pub fn new_numeric(numeric_type: Type) -> Self {
        Self {
            numeric_type: Some(numeric_type),
            ..Self::default()
        }
    }
    pub fn clear(&mut self) {
        self.zero = None;
        self.minimum = None;
        self.maximum = None;
        self.parsable_string = false;
    }
    /// Java `copy`: scalar Rust storage has value semantics, yielding the same
    /// observable result as Java's `EtomoNumber.set`/shared first assignment.
    pub fn copy(&mut self, input: Option<&Self>) {
        if let Some(input) = input {
            self.zero = input.zero;
            self.minimum = input.minimum;
            self.maximum = input.maximum;
            self.parsable_string = input.parsable_string;
        } else {
            self.clear();
        }
    }
    pub fn validate_static(
        text: Option<&str>,
        field_type: Option<FieldType>,
        numeric_type: Option<Type>,
    ) -> Option<String> {
        let text = text?;
        if java_lang_string_matches_whitespace(text) {
            return None;
        }
        Self::validate_number(text, field_type, numeric_type)
    }
    pub fn validate(&self, text: Option<&str>) -> Option<String> {
        let text = text?;
        if java_lang_string_matches_whitespace(text) {
            return None;
        }
        if let Some(error) = Self::validate_number(text, self.field_type, self.numeric_type) {
            return Some(error);
        }
        let value = match self.numeric_type {
            Some(Type::Long) => java_lang_long_parse_long(java_lang_string_trim(text))
                .ok()
                .map(|v| v as f64),
            Some(Type::Double) => java_lang_double_value_of(java_lang_string_trim(text)).ok(),
            Some(_) => java_lang_integer_parse_int(java_lang_string_trim(text))
                .ok()
                .map(|v| v as f64),
            None => None,
        };
        if let Some(value) = value {
            if self.zero.is_some_and(|zero| value <= zero) {
                return Some("must be a positive number".into());
            }
            if let Some(minimum) = self.minimum.filter(|minimum| value < *minimum) {
                return Some(format!("must be greater than or equal to {minimum}"));
            }
            if let Some(maximum) = self.maximum.filter(|maximum| value > *maximum) {
                return Some(format!("must be less than or equal to {maximum}"));
            }
        }
        // ParsedList is a separate Java unit.  Preserve the source's important
        // distinction: an ordinary unquoted string remains valid; bracketed or
        // divider-containing malformed input is rejected at that boundary.
        if self.numeric_type.is_none() && self.parsable_string {
            let trimmed = java_lang_string_trim(text);
            let bracketed = trimmed.starts_with('[') || trimmed.ends_with(']');
            if bracketed && (!trimmed.starts_with('[') || !trimmed.ends_with(']')) {
                return Some("invalid array".into());
            }
        }
        None
    }
    fn validate_number(
        text: &str,
        field_type: Option<FieldType>,
        numeric_type: Option<Type>,
    ) -> Option<String> {
        if numeric_type.is_none()
            && field_type.is_some_and(|field| field.get_numeric_type().is_none())
        {
            return None;
        }
        let text = java_lang_string_trim(text);
        let numeric_type = numeric_type.unwrap_or(Type::Integer);
        let valid = match numeric_type {
            Type::Long => java_lang_long_parse_long(text).is_ok(),
            Type::Double => java_lang_double_value_of(text).is_ok(),
            Type::Integer | Type::Boolean => java_lang_integer_parse_int(text).is_ok(),
        };
        (!valid).then(|| format!("must be of type {numeric_type}"))
    }
    pub fn set_number_must_be_positive(&mut self, value: bool) {
        self.zero = value.then_some(0.);
    }
    pub fn set_minimum(&mut self, value: f64) {
        self.minimum = Some(value);
    }
    pub fn set_maximum(&mut self, value: f64) {
        self.maximum = Some(value);
    }
    pub fn set_parsable_string(&mut self, value: bool) {
        self.parsable_string = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_type_and_bounds_follow_source() {
        let mut set = ValidationSet::new(Some(FieldType::Integer), None);
        set.set_number_must_be_positive(true);
        set.set_minimum(2.);
        set.set_maximum(5.);
        assert_eq!(
            set.validate(Some("0")),
            Some("must be a positive number".into())
        );
        assert_eq!(
            set.validate(Some("1")),
            Some("must be greater than or equal to 2".into())
        );
        assert_eq!(
            set.validate(Some("6")),
            Some("must be less than or equal to 5".into())
        );
        assert_eq!(set.validate(Some("3")), None);
        assert_eq!(
            set.validate(Some("x")),
            Some("must be of type Integer".into())
        );
    }
}
