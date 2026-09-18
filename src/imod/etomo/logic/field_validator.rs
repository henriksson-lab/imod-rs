//! `IMOD/Etomo/src/etomo/logic/FieldValidator.java`.

use super::validation_set::ValidationSet;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_integer_parse_int, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::ui::field_type::{CollectionType, FieldType};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Java's process-wide `DEBUG` flag.  The Rust UI chooses the initial value;
/// keeping the override here preserves `setDebug`/`resetDebug` semantics
/// without coupling validation to a particular GUI toolkit.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// The non-Swing portion of Java `FieldValidator`.  Popup dispatch and
/// `FieldDisplayer` selection stay at the GUI boundary; callers receive exactly the
/// source error text that would be placed in `FieldValidationFailedException`.
pub struct FieldValidator;
impl FieldValidator {
    /// Java `setDebug`.
    pub fn set_debug(debug: bool) {
        DEBUG.store(debug, Ordering::Relaxed);
    }

    /// Java `resetDebug`, with the current application default supplied by
    /// the Rust application host instead of `EtomoDirector.INSTANCE`.
    pub fn reset_debug(application_debug: bool) {
        Self::set_debug(application_debug);
    }

    /// Current Java `DEBUG` equivalent, for a GUI host deciding whether to
    /// render the source's diagnostic stack trace.
    pub fn debug_enabled() -> bool {
        DEBUG.load(Ordering::Relaxed)
    }

    /// Java `handleValidation`'s non-Swing decision.  Rendering the message
    /// and making a field visible remain the caller's UI responsibility.
    pub fn handle_validation(error: Option<&str>) -> bool {
        error.is_none()
    }

    /// Java `fail`'s stable diagnostic payload.  Popup and stack-trace work
    /// are intentionally outside the pure validator, while callers retain
    /// the metadata necessary to present the native message.
    pub fn fail(
        error: &str,
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        validation_type: Option<&str>,
        description: Option<&str>,
    ) -> String {
        let mut result = error.to_owned();
        result.push_str(",descr:");
        result.push_str(description.unwrap_or("null"));
        result.push_str(",fieldText:");
        result.push_str(field_text.unwrap_or("null"));
        if let Some(field_type) = field_type {
            result.push_str(",fieldType:");
            result.push_str(&field_type.to_string());
        }
        if let Some(validation_type) = validation_type {
            result.push_str(",validationType:");
            result.push_str(validation_type);
        }
        result
    }

    /// Java `validateNumber`, factored so scalar and collection validation
    /// share exactly the same validation-set, empty-text, numeric, and
    /// positive-value rules.
    pub fn validate_number(
        text: &str,
        field_type: FieldType,
        validation_set: Option<&ValidationSet>,
        must_be_positive: bool,
    ) -> Result<(), String> {
        if let Some(set) = validation_set {
            if let Some(error) = set.validate(Some(text)) {
                return Err(error);
            }
        }
        if text.is_empty() {
            return Ok(());
        }
        let valid = if field_type.validation_type().integer() {
            java_lang_integer_parse_int(text).is_ok()
        } else {
            text.parse::<f64>().is_ok()
        };
        if !valid {
            return Err(format!(
                "wrong type - should be {}",
                field_type.validation_type()
            ));
        }
        if must_be_positive && text.parse::<f64>().is_ok_and(|number| number <= 0.) {
            return Err("requires a postive number".into());
        }
        Ok(())
    }
    /// Java `isTextValid`; GUI popup/display work is intentionally absent from
    /// this boolean convenience path just as callers use it in unit tests.
    pub fn is_text_valid(
        text: Option<&str>,
        field_type: Option<FieldType>,
        required: bool,
    ) -> bool {
        Self::validate_text(text, field_type, None, required, false, false, None, false).is_ok()
    }
    pub fn validate_text(
        text: Option<&str>,
        field_type: Option<FieldType>,
        max_array_size: Option<usize>,
        required: bool,
        file_must_exist: bool,
        file_only: bool,
        validation_set: Option<&ValidationSet>,
        must_be_positive: bool,
    ) -> Result<Option<String>, String> {
        if required && text.is_none_or(java_lang_string_matches_whitespace) {
            return Err("an entry is required".into());
        }
        let Some(text) = text else {
            return Ok(None);
        };
        let Some(field_type) = field_type else {
            return Ok(Some(text.into()));
        };
        if !field_type.validation_type().numeric() {
            if let Some(set) = validation_set {
                if let Some(error) = set.validate(Some(text)) {
                    return Err(error);
                }
            }
            if field_type == FieldType::File
                && (file_must_exist || file_only)
                && !java_lang_string_matches_whitespace(text)
            {
                let path = Path::new(text);
                if file_must_exist && !path.exists() {
                    return Err(format!(
                        "must contain a file {}that exists",
                        if file_only { "" } else { "or directory " }
                    ));
                }
                if file_only && !path.is_file() {
                    return Err("must contain a file".into());
                }
            }
            return Ok(Some(text.into()));
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Some(String::new()));
        }
        let element_list = ElementList::new(field_type, trimmed);
        if field_type.has_required_size()
            && !element_list.equals_n_elements(field_type.required_size() as usize)
        {
            return Err(format!(
                "wrong number of elements - should have {} elements",
                field_type.required_size()
            ));
        }
        if let Some(max) = max_array_size.filter(|max| !element_list.gt_n_elements(*max)) {
            return Err(format!(
                "Too many elements.  The maximum number of elements allowed for this field is {max}"
            ));
        }
        for element in if field_type.is_collection() {
            element_list.iter().collect()
        } else {
            vec![trimmed]
        } {
            Self::validate_number(element, field_type, validation_set, must_be_positive)?;
        }
        Ok(Some(trimmed.into()))
    }
    pub fn equals(
        field_type: Option<FieldType>,
        first: Option<&str>,
        second: Option<&str>,
    ) -> bool {
        let (Some(first), Some(second)) = (first, second) else {
            return first.is_none() && second.is_none();
        };
        match field_type {
            Some(FieldType::Integer) => first.parse::<i32>().ok() == second.parse::<i32>().ok(),
            Some(FieldType::FloatingPoint) => {
                first.parse::<f64>().ok() == second.parse::<f64>().ok()
            }
            Some(FieldType::File) => Path::new(first) == Path::new(second),
            Some(FieldType::IntegerList) => {
                first.split_whitespace().collect::<String>()
                    == second.split_whitespace().collect::<String>()
            }
            Some(field) if field.is_collection() => {
                split(field, first)
                    .iter()
                    .zip(split(field, second).iter())
                    .all(|(a, b)| {
                        Self::equals(
                            Some(if field.validation_type().integer() {
                                FieldType::Integer
                            } else {
                                FieldType::FloatingPoint
                            }),
                            Some(a),
                            Some(b),
                        )
                    })
                    && split(field, first).len() == split(field, second).len()
            }
            _ => first.trim() == second.trim(),
        }
    }

    /// Java `validatePairedArrays`.  `UIComponent` and the popup exception
    /// route are GUI boundaries; this returns the exact source failure text.
    pub fn validate_paired_arrays(
        text1: Option<&str>,
        _descr1: &str,
        text2: Option<&str>,
        _descr2: &str,
    ) -> Result<(), String> {
        let count1 = text1.map_or(0, |text| split(FieldType::IntegerArray, text).len());
        let count2 = text2.map_or(0, |text| split(FieldType::IntegerArray, text).len());
        if count1 == 0 && count2 == 0 {
            return Err("required - at least one field must be filled in".into());
        }
        if count1 > 1 && count2 > 1 && count1 != count2 {
            return Err("the number of elements in the two fields are unequal. Either use an equal number of elements in both fields, use only one field, or put a single element in one field".into());
        }
        Ok(())
    }
}

/// Java `FieldValidator.ElementList`.  Java lazily caches its split array;
/// this Rust form caches the same result on construction while retaining its
/// unusual comma-counting rule in [`Self::get_n_elements`].
#[derive(Clone, Debug)]
pub struct ElementList {
    field_type: FieldType,
    field_text: String,
    list: Vec<String>,
}
impl ElementList {
    pub fn new(field_type: FieldType, field_text: impl Into<String>) -> Self {
        let field_text = field_text.into();
        let list = split(field_type, &field_text)
            .into_iter()
            .map(str::to_owned)
            .collect();
        Self {
            field_type,
            field_text,
            list,
        }
    }
    /// Java `equalsNElements`.
    pub fn equals_n_elements(&self, input: usize) -> bool {
        self.get_n_elements() == input
    }
    /// Java `gtNElements`.  Despite its historical name, Java returns true
    /// when the count is *less* than the supplied maximum.
    pub fn gt_n_elements(&self, input: usize) -> bool {
        self.get_n_elements() < input
    }
    /// Java `getNElements`.
    pub fn get_n_elements(&self) -> usize {
        count_elements(
            &self.field_text,
            &self.list.iter().map(String::as_str).collect::<Vec<_>>(),
        )
    }
    /// Java `iterator`.
    pub fn iter(&self) -> ElementListIterator<'_> {
        let _ = self.field_type;
        ElementListIterator {
            list: &self.list,
            index: 0,
        }
    }
}

/// Java `FieldValidator.ElementListIterator`.
pub struct ElementListIterator<'a> {
    list: &'a [String],
    index: usize,
}
impl<'a> ElementListIterator<'a> {
    pub fn has_next(&self) -> bool {
        self.index < self.list.len()
    }
    pub fn next(&mut self) -> Option<&'a str> {
        let value = self.list.get(self.index)?;
        self.index += 1;
        Some(value)
    }
}
impl<'a> Iterator for ElementListIterator<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        ElementListIterator::next(self)
    }
}
fn split(field_type: FieldType, value: &str) -> Vec<&str> {
    match field_type.get_collection_type() {
        Some(CollectionType::Array) => value
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(CollectionType::List) => value
            .split(|c: char| c == ',' || c.is_ascii_whitespace() || c == '-')
            .filter(|s| !s.is_empty())
            .collect(),
        Some(CollectionType::MatlabArray) => value
            .split(|c: char| c == ',' || c.is_ascii_whitespace() || c == ':')
            .filter(|s| !s.is_empty())
            .collect(),
        None => vec![value],
    }
}

/// Java `ElementList.countElements`: `String.split` discards trailing empty
/// elements, then the source adds one for every additional trailing comma.
fn count_elements(original: &str, elements: &[&str]) -> usize {
    let compact: String = original
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    let extra_trailing = compact.strip_suffix(',').map_or(0, |without_last| {
        without_last.chars().rev().take_while(|c| *c == ',').count()
    });
    elements.len() + extra_trailing
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collections_require_shape_and_numeric_values() {
        assert!(
            FieldValidator::validate_text(
                Some("1, 2"),
                Some(FieldType::IntegerPair),
                None,
                false,
                false,
                false,
                None,
                false
            )
            .is_ok()
        );
        assert_eq!(
            FieldValidator::validate_text(
                Some("1"),
                Some(FieldType::IntegerPair),
                None,
                false,
                false,
                false,
                None,
                false
            ),
            Err("wrong number of elements - should have 2 elements".into())
        );
        assert!(
            FieldValidator::validate_text(
                Some("1,,"),
                Some(FieldType::IntegerPair),
                None,
                false,
                false,
                false,
                None,
                false
            )
            .is_ok()
        );
        assert!(FieldValidator::equals(
            Some(FieldType::FloatingPointArray),
            Some("1, 2.0"),
            Some("1.0 2")
        ));
        assert!(FieldValidator::is_text_valid(
            Some("7"),
            Some(FieldType::Integer),
            true
        ));
        assert_eq!(
            FieldValidator::validate_paired_arrays(None, "first", None, "second"),
            Err("required - at least one field must be filled in".into())
        );
        assert!(
            FieldValidator::validate_paired_arrays(Some("1,2"), "first", Some("3,4"), "second")
                .is_ok()
        );
    }
    #[test]
    fn diagnostic_and_number_helpers_follow_java_validation_rules() {
        FieldValidator::set_debug(true);
        assert!(FieldValidator::debug_enabled());
        FieldValidator::reset_debug(false);
        assert!(!FieldValidator::debug_enabled());
        assert!(FieldValidator::handle_validation(None));
        assert!(!FieldValidator::handle_validation(Some("bad")));
        assert_eq!(
            FieldValidator::fail(
                "bad",
                Some("x"),
                Some(FieldType::Integer),
                Some("number"),
                Some("field"),
            ),
            "bad,descr:field,fieldText:x,fieldType:[validationType:an integer,collectionType:null,requiredSize:-1,validationType:number"
        );
        assert!(
            FieldValidator::validate_number("2147483647", FieldType::Integer, None, false).is_ok()
        );
        assert!(
            FieldValidator::validate_number("2147483648", FieldType::Integer, None, false).is_err()
        );
    }
    #[test]
    fn element_list_preserves_java_counting_and_iteration() {
        let list = ElementList::new(FieldType::IntegerArray, "1,,");
        // Java String.split drops trailing empties, then ElementList restores
        // each additional trailing comma for required-size validation.
        assert_eq!(list.get_n_elements(), 2);
        assert!(list.gt_n_elements(3));
        assert!(!list.gt_n_elements(2));
        let mut iter = list.iter();
        assert!(iter.has_next());
        assert_eq!(iter.next(), Some("1"));
        assert!(!iter.has_next());
        assert_eq!(iter.next(), None);
    }
}
