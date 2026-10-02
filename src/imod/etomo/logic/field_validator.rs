//! `IMOD/Etomo/src/etomo/logic/FieldValidator.java`.

use super::validation_set::ValidationSet;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::const_etomo_number::{
    Type, java_lang_double_value_of, java_lang_integer_parse_int, java_lang_long_parse_long,
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::swing::validation_extension::ValidationExtension;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;
use regex::Regex;
use std::path::Path;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Java `TITLE`.
const TITLE: &str = "Field Validation Failed";

/// Java `private static boolean DEBUG =
/// EtomoDirector.INSTANCE.getArguments().isDebug()`.
static DEBUG: LazyLock<AtomicBool> =
    LazyLock::new(|| AtomicBool::new(etomo_director::ARGUMENTS.lock().unwrap().is_debug()));

/// Java `EtomoDirector.INSTANCE.isUnitTest()`.
fn director_is_unit_test() -> bool {
    etomo_director::INSTANCE.is_unit_test()
}

/// Java `FieldValidator`.
pub struct FieldValidator;
impl FieldValidator {
    /// Java `setDebug`.
    pub fn set_debug(debug: bool) {
        DEBUG.store(debug, Ordering::Relaxed);
    }

    /// Java `resetDebug`.
    pub fn reset_debug() {
        DEBUG.store(
            etomo_director::ARGUMENTS.lock().unwrap().is_debug(),
            Ordering::Relaxed,
        );
    }

    /// Java `isTextValid`.  Validates without popping up an error message.
    /// Returns true if valid.
    pub fn is_text_valid(
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        component: Option<&dyn UIComponent>,
        validation_extension: Option<&ValidationExtension>,
        debug: bool,
    ) -> bool {
        let mut required = false;
        if let Some(validation_extension) = validation_extension {
            required = validation_extension.is_required();
        }
        match Self::validate_text_private(
            field_text,
            field_type,
            -1,
            component,
            None,
            required,
            false,
            false,
            None,
            validation_extension,
            None,
            None,
            false,
            debug,
        ) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Java `handleValidation`.  Returns false if invalid.
    pub fn handle_validation(
        err_msg: Option<&str>,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
    ) -> bool {
        let Some(err_msg) = err_msg else {
            return true;
        };
        ui_harness::INSTANCE.with(|harness| {
            harness.open_message_dialog_base_manager_ui_component_string_string_field_displayer_field_displayer(
                None,
                component,
                &format!(
                    "Validation failure in {}: {}.",
                    descr.unwrap_or("null"),
                    err_msg
                ),
                TITLE,
                field_displayer1,
                field_displayer2,
            )
        });
        false
    }

    /// Java `validateText(String, FieldType, UIComponent, String, boolean, boolean,
    /// boolean, ValidationSet, FieldDisplayer, FieldDisplayer)`.  Pops up an error
    /// message and throws on failure; returns the (trimmed, when numeric) text.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_text_string_field_type_ui_component_string_boolean_boolean_boolean_validation_set_field_displayer_field_displayer(
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        required: bool,
        file_must_exist: bool,
        file_only: bool,
        validation_set: Option<&ValidationSet>,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Self::validate_text_private(
            field_text,
            field_type,
            -1,
            component,
            descr,
            required,
            file_must_exist,
            file_only,
            validation_set,
            None,
            field_displayer1,
            field_displayer2,
            true,
            false,
        )
    }

    /// Java `validateText(String, FieldType, int, UIComponent, String, boolean,
    /// boolean, boolean, ValidationSet, FieldDisplayer, FieldDisplayer)`.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_text_string_field_type_int_ui_component_string_boolean_boolean_boolean_validation_set_field_displayer_field_displayer(
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        max_array_size: i32,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        required: bool,
        file_must_exist: bool,
        file_only: bool,
        validation_set: Option<&ValidationSet>,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Self::validate_text_private(
            field_text,
            field_type,
            max_array_size,
            component,
            descr,
            required,
            file_must_exist,
            file_only,
            validation_set,
            None,
            field_displayer1,
            field_displayer2,
            true,
            false,
        )
    }

    /// Java `validateText(String, FieldType, UIComponent, String,
    /// ValidationExtension, FieldDisplayer, FieldDisplayer, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_text_string_field_type_ui_component_string_validation_extension_field_displayer_field_displayer_boolean(
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        validation_extension: Option<&ValidationExtension>,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
        debug: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut required = false;
        if let Some(validation_extension) = validation_extension {
            required = validation_extension.is_required();
        }
        Self::validate_text_private(
            field_text,
            field_type,
            -1,
            component,
            descr,
            required,
            false,
            false,
            None,
            validation_extension,
            field_displayer1,
            field_displayer2,
            true,
            debug,
        )
    }

    /// Java private `fail`.  Java always throws; the Rust form returns the
    /// exception for the caller to return as `Err`.
    #[allow(clippy::too_many_arguments)]
    fn fail(
        err_msg: &str,
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        validation_type: Option<&str>,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
        popup_message: bool,
        debug: bool,
    ) -> FieldValidationFailedException {
        if popup_message {
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_ui_component_string_string_field_displayer_field_displayer(
                    None,
                    component,
                    &format!(
                        "Validation failure in {}: {}.",
                        descr.unwrap_or("null"),
                        err_msg
                    ),
                    TITLE,
                    field_displayer1,
                    field_displayer2,
                )
            });
        }
        let exception = FieldValidationFailedException::new(Some(&format!(
            "{},descr:{},fieldText:{}{}{}",
            err_msg,
            descr.unwrap_or("null"),
            field_text.unwrap_or("null"),
            match field_type {
                Some(field_type) => format!(",fieldType:{}", field_type),
                None => String::new(),
            },
            match validation_type {
                Some(validation_type) => format!(",validationType:{}", validation_type),
                None => String::new(),
            }
        )));
        if debug
            || DEBUG.load(Ordering::Relaxed)
            || {
                let mut arguments = etomo_director::ARGUMENTS.lock().unwrap();
                arguments.is_test() || arguments.is_debug()
            }
            || director_is_unit_test()
        {
            // exception.printStackTrace(): the stack frames are the JVM's.
            eprintln!("{}", exception);
        }
        exception
    }

    /// Java private `validateText(String, FieldType, int, UIComponent, String,
    /// boolean, boolean, boolean, ValidationSet, ValidationExtension,
    /// FieldDisplayer, FieldDisplayer, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn validate_text_private(
        field_text: Option<&str>,
        field_type: Option<FieldType>,
        max_array_size: i32,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        required: bool,
        mut file_must_exist: bool,
        mut file_only: bool,
        validation_set: Option<&ValidationSet>,
        validation_extension: Option<&ValidationExtension>,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
        popup_message: bool,
        debug: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        if let Some(validation_extension) = validation_extension {
            file_must_exist = validation_extension.is_file_must_exist();
            file_only = validation_extension.is_file_only();
        }
        // validate required (field level)
        if required && field_text.is_none_or(java_lang_string_matches_whitespace) {
            return Err(Self::fail(
                "an entry is required",
                field_text,
                field_type,
                None,
                component,
                descr,
                field_displayer1,
                field_displayer2,
                popup_message,
                debug,
            ));
        }
        let Some(field_text) = field_text else {
            return Ok(None);
        };
        let Some(field_type) = field_type else {
            return Ok(Some(field_text.to_string()));
        };
        // Don't trim strings or file names.
        if !field_type.validation_type().numeric() {
            let mut errmsg = None;
            if let Some(validation_set) = validation_set {
                errmsg = validation_set.validate(Some(field_text));
            }
            if let Some(errmsg) = errmsg {
                return Err(Self::fail(
                    &errmsg,
                    Some(field_text),
                    Some(field_type),
                    None,
                    component,
                    descr,
                    field_displayer1,
                    field_displayer2,
                    popup_message,
                    false,
                ));
            }
            // File validation only if at least one file validation boolean is set.
            if field_type == FieldType::File
                && (file_must_exist || file_only)
                && !java_lang_string_matches_whitespace(field_text)
            {
                let file = Path::new(field_text);
                if file_must_exist && !file.exists() {
                    return Err(Self::fail(
                        &format!(
                            "must contain a file {}that exists",
                            if file_only { "" } else { "or directory " }
                        ),
                        Some(field_text),
                        Some(field_type),
                        None,
                        component,
                        descr,
                        field_displayer1,
                        field_displayer2,
                        popup_message,
                        debug,
                    ));
                }
                if file_only && !file.is_file() {
                    return Err(Self::fail(
                        "must contain a file",
                        Some(field_text),
                        Some(field_type),
                        None,
                        component,
                        descr,
                        field_displayer1,
                        field_displayer2,
                        popup_message,
                        debug,
                    ));
                }
            }
            return Ok(Some(field_text.to_string()));
        }
        // Trim because Number.parse... will fail on external spaces.
        let text = java_lang_string_trim(field_text);
        if field_type.is_collection() {
            // Empty collections are valid
            if text.is_empty() {
                return Ok(Some(text.to_string()));
            }
            // Validate arrays and lists.
            let mut element_list = ElementList::new(field_type, Some(text));
            if field_type.has_required_size() {
                // Validate the number of elements
                if !element_list.equals_n_elements(field_type.required_size()) {
                    // Wrong number of elements
                    return Err(Self::fail(
                        &format!(
                            "wrong number of elements - should have {} elements",
                            field_type.required_size()
                        ),
                        Some(field_text),
                        Some(field_type),
                        None,
                        component,
                        descr,
                        field_displayer1,
                        field_displayer2,
                        popup_message,
                        debug,
                    ));
                }
            } else if max_array_size != -1 {
                // Validate the number of elements
                // Kept native (BUGS.md): gtNElements is "count < max", so a field
                // holding exactly maxArraySize elements is rejected.  No caller
                // constructs a LabeledTextField with a maximum, so it is unreachable.
                if !element_list.gt_n_elements(max_array_size) {
                    // Too many elements
                    return Err(Self::fail(
                        &format!(
                            "Too many elements.  The maximum number of elements allowed for this field is {}",
                            max_array_size
                        ),
                        Some(field_text),
                        Some(field_type),
                        None,
                        component,
                        descr,
                        field_displayer1,
                        field_displayer2,
                        popup_message,
                        debug,
                    ));
                }
            }
            // Validate integers or floating point numbers in the array or list.
            let mut iterator = element_list.iterator();
            while iterator.has_next() {
                Self::validate_number(
                    validation_extension,
                    validation_set,
                    iterator.next(),
                    component,
                    descr,
                    field_type,
                    field_displayer1,
                    field_displayer2,
                    popup_message,
                    debug,
                )?;
            }
        } else {
            // Validate integers and floating point numbers.
            Self::validate_number(
                validation_extension,
                validation_set,
                Some(text),
                component,
                descr,
                field_type,
                field_displayer1,
                field_displayer2,
                popup_message,
                debug,
            )?;
        }
        // Validation succeeded - return original trimmed field text.
        Ok(Some(java_lang_string_trim(field_text).to_string()))
    }

    /// Java `validatePairedArrays`.
    pub fn validate_paired_arrays(
        text1: Option<&str>,
        component: Option<&dyn UIComponent>,
        descr1: &str,
        text2: Option<&str>,
        descr2: &str,
    ) -> Result<(), FieldValidationFailedException> {
        let validation_type = "comma-separated paired arrays";
        let field_text = format!(
            "[text1:{},text2:{}]",
            text1.unwrap_or("null"),
            text2.unwrap_or("null")
        );
        let descr = format!("{} and {}", descr1, descr2);
        // At least one must have a valid array
        let num1 = utilities::get_number_elements(text1);
        let num2 = utilities::get_number_elements(text2);
        if num1 == 0 && num2 == 0 {
            return Err(Self::fail(
                "required - at least one field must be filled in",
                Some(&field_text),
                None,
                Some(validation_type),
                component,
                Some(&descr),
                None,
                None,
                true,
                false,
            ));
        }
        // If both have a value they must have an equal number of array elements, or one
        // must have only one array element
        if num1 > 1 && num2 > 1 && num1 != num2 {
            return Err(Self::fail(
                "the number of elements in the two fields are unequal.  Either use an equal \
                 number of elements in both fields, use only one field, or put a single element \
                 in one field",
                Some(&field_text),
                None,
                Some(validation_type),
                component,
                Some(&descr),
                None,
                None,
                true,
                false,
            ));
        }
        Ok(())
    }

    /// Java private `validateNumber`.
    #[allow(clippy::too_many_arguments)]
    fn validate_number(
        validation_extension: Option<&ValidationExtension>,
        validation_set: Option<&ValidationSet>,
        text: Option<&str>,
        component: Option<&dyn UIComponent>,
        descr: Option<&str>,
        field_type: FieldType,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
        popup_message: bool,
        _debug: bool,
    ) -> Result<(), FieldValidationFailedException> {
        let mut errmsg = None;
        if let Some(validation_set) = validation_set {
            errmsg = validation_set.validate(text);
        }
        if let Some(errmsg) = errmsg {
            return Err(Self::fail(
                &errmsg,
                text,
                Some(field_type),
                None,
                component,
                descr,
                field_displayer1,
                field_displayer2,
                popup_message,
                false,
            ));
        }
        let Some(text) = text.filter(|text| !text.is_empty()) else {
            return Ok(());
        };
        let mut must_be_positive = false;
        if validation_extension.is_some_and(|extension| extension.is_must_be_positive()) {
            must_be_positive = true;
        }
        let r#type = field_type.get_numeric_type();
        let mut failed = false;
        let parsed: Result<(), String> = (|| {
            if r#type == Some(Type::Long) {
                let l = java_lang_long_parse_long(text)?;
                if must_be_positive && l <= 0 {
                    failed = true;
                }
            } else if r#type == Some(Type::Double) {
                let d = java_lang_double_value_of(text)?;
                if must_be_positive && d <= 0. {
                    failed = true;
                }
            } else {
                let i = java_lang_integer_parse_int(text)?;
                if must_be_positive && i <= 0 {
                    failed = true;
                }
            }
            Ok(())
        })();
        match parsed {
            Ok(()) => {
                if failed {
                    return Err(Self::fail(
                        "requires a postive number",
                        Some(text),
                        Some(field_type),
                        None,
                        component,
                        descr,
                        field_displayer1,
                        field_displayer2,
                        popup_message,
                        false,
                    ));
                }
                Ok(())
            }
            // catch (NumberFormatException e)
            Err(_) => Err(Self::fail(
                &format!("wrong type - should be {}", field_type.validation_type()),
                Some(text),
                Some(field_type),
                None,
                component,
                descr,
                field_displayer1,
                field_displayer2,
                popup_message,
                false,
            )),
        }
    }

    /// Java `equals(FieldType, String, String)`.
    pub fn equals(
        field_type: Option<FieldType>,
        field_text1: Option<&str>,
        field_text2: Option<&str>,
    ) -> bool {
        let (Some(field_text1), Some(field_text2)) = (field_text1, field_text2) else {
            if field_text1.is_none() && field_text2.is_none() {
                return true;
            }
            return false;
        };
        if field_type == Some(FieldType::Integer) {
            let mut number1 = EtomoNumber::new();
            let mut number2 = EtomoNumber::new();
            number1.set_string(Some(field_text1));
            number2.set_string(Some(field_text2));
            return number1.equals_const_etomo_number(Some(&number2));
        }
        if field_type == Some(FieldType::FloatingPoint) {
            let mut number1 = EtomoNumber::new_with_type(Some(Type::Double));
            let mut number2 = EtomoNumber::new_with_type(Some(Type::Double));
            number1.set_string(Some(field_text1));
            number2.set_string(Some(field_text2));
            return number1.equals_const_etomo_number(Some(&number2));
        }
        if field_type.is_none() || field_type == Some(FieldType::String) {
            return java_lang_string_trim(field_text1) == java_lang_string_trim(field_text2);
        }
        if field_type == Some(FieldType::File) {
            // new File(a).equals(new File(b)): java.io.UnixFileSystem.normalize
            // collapses repeated '/' and drops a trailing '/', then the path
            // strings are compared.
            let normalize = |path: &str| -> String {
                let mut out = String::with_capacity(path.len());
                let mut previous_slash = false;
                for c in path.chars() {
                    if c == '/' && previous_slash {
                        continue;
                    }
                    previous_slash = c == '/';
                    out.push(c);
                }
                if out.len() > 1 && out.ends_with('/') {
                    out.pop();
                }
                out
            };
            return normalize(field_text1) == normalize(field_text2);
        }
        if field_type == Some(FieldType::IntegerList) {
            // Lists are too complicated to parse (they contain things like "1 - 3").
            // Remove all spaces and compare as strings.
            let whitespace = Regex::new(r"(?-u:\s)+").unwrap();
            return whitespace.replace_all(field_text1, "")
                == whitespace.replace_all(field_text2, "");
        }
        // Handle arrays
        let mut number_field_type = None;
        if field_type == Some(FieldType::IntegerArray)
            || field_type == Some(FieldType::IntegerPair)
            || field_type == Some(FieldType::IntegerTriple)
        {
            number_field_type = Some(FieldType::Integer);
        }
        if field_type == Some(FieldType::FloatingPointArray)
            || field_type == Some(FieldType::FloatingPointPair)
        {
            number_field_type = Some(FieldType::FloatingPoint);
        }
        let field_type = field_type.unwrap();
        let mut element_list1 = ElementList::new(field_type, Some(field_text1));
        let mut element_list2 = ElementList::new(field_type, Some(field_text2));
        if element_list1.get_n_elements() != element_list2.get_n_elements() {
            return false;
        }
        let mut iterator1 = element_list1.iterator();
        let mut iterator2 = element_list2.iterator();
        while iterator1.has_next() {
            // Call this function with the field type for either float or integer.
            if !Self::equals(number_field_type, iterator1.next(), iterator2.next()) {
                return false;
            }
        }
        true
    }
}

/// Java `FieldValidator.ElementList`.
struct ElementList {
    field_type: FieldType,
    field_text: Option<String>,
    list: Option<Vec<String>>,
    n_elements: i32,
}
impl ElementList {
    fn new(field_type: FieldType, field_text: Option<&str>) -> ElementList {
        ElementList {
            field_type,
            field_text: field_text.map(str::to_string),
            list: None,
            n_elements: -1,
        }
    }

    /// Java `equalsNElements`.
    fn equals_n_elements(&mut self, input: i32) -> bool {
        self.get_n_elements() == input
    }

    /// Java `gtNElements`: true when the element count is less than `input`.
    fn gt_n_elements(&mut self, input: i32) -> bool {
        self.get_n_elements() < input
    }

    /// Java `iterator`.
    fn iterator(&mut self) -> ElementListIterator {
        if self.list.is_none() {
            self.split();
        }
        ElementListIterator::new(self.list.clone())
    }

    /// Java `getNElements`.
    fn get_n_elements(&mut self) -> i32 {
        if self.list.is_none() {
            self.split();
        }
        if self.n_elements == -1 {
            self.count_elements();
        }
        self.n_elements
    }

    /// Java `split`: `fieldText.split(fieldType.getSplitter())`.
    fn split(&mut self) {
        let splitter = Regex::new(&format!("(?-u){}", self.field_type.get_splitter())).unwrap();
        self.list = Some(utilities::java_lang_string_split(
            self.field_text
                .as_deref()
                .expect("FieldValidator.ElementList.fieldText is null"),
            &splitter,
        ));
    }

    /// Java `countElements`.
    fn count_elements(&mut self) {
        self.n_elements = 0;
        let Some(field_text) = &self.field_text else {
            return;
        };
        let mut text = field_text.clone();
        // The commas at the end of the collection count as elements. Splitting will
        // eliminate these commas, so they have to be counted.
        if text.ends_with(',') {
            // Remove all whitespace
            text = Regex::new(r"(?-u:\s)+")
                .unwrap()
                .replace_all(&text, "")
                .into_owned();
            // Count ending commas. The last comma doesn't count because an extra comma
            // is necessary to add an element onto the end.
            let bytes = text.as_bytes();
            let mut i = bytes.len() as i32 - 2;
            while i >= 0 {
                if bytes[i as usize] == b',' {
                    self.n_elements += 1;
                } else {
                    break;
                }
                i -= 1;
            }
        }
        if let Some(list) = &self.list {
            self.n_elements += list.len() as i32;
        }
    }
}

/// Java `FieldValidator.ElementListIterator`.
struct ElementListIterator {
    list: Option<Vec<String>>,
    index: i32,
}
impl ElementListIterator {
    fn new(list: Option<Vec<String>>) -> ElementListIterator {
        ElementListIterator { list, index: 0 }
    }

    /// Java `hasNext`.
    fn has_next(&self) -> bool {
        match &self.list {
            Some(list) if self.index >= 0 && (self.index as usize) < list.len() => true,
            _ => false,
        }
    }

    /// Java `next`.
    fn next(&mut self) -> Option<&str> {
        if !self.has_next() {
            return None;
        }
        let index = self.index as usize;
        self.index += 1;
        self.list.as_ref().map(|list| list[index].as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collections_require_shape_and_numeric_values() {
        let validate = |text: &str, field_type: FieldType| {
            FieldValidator::validate_text_private(
                Some(text),
                Some(field_type),
                -1,
                None,
                None,
                false,
                false,
                false,
                None,
                None,
                None,
                None,
                false,
                false,
            )
        };
        assert!(validate("1, 2", FieldType::IntegerPair).is_ok());
        assert_eq!(
            validate("1", FieldType::IntegerPair)
                .unwrap_err()
                .get_message(),
            Some(
                "wrong number of elements - should have 2 elements,descr:null,fieldText:1,\
                 fieldType:[validationType:an integer,collectionType:null,requiredSize:2"
            )
        );
        assert!(validate("1,,", FieldType::IntegerPair).is_ok());
        assert!(validate("1 - 3,7", FieldType::IntegerList).is_ok());
        assert!(validate("1.5", FieldType::Integer).is_err());
        assert!(FieldValidator::equals(
            Some(FieldType::FloatingPointArray),
            Some("1, 2.0"),
            Some("1.0 2")
        ));
        assert!(FieldValidator::equals(
            Some(FieldType::File),
            Some("a//b/"),
            Some("a/b")
        ));
        assert!(!FieldValidator::equals(
            Some(FieldType::File),
            Some("a/./b"),
            Some("a/b")
        ));
        assert!(FieldValidator::is_text_valid(
            Some("7"),
            Some(FieldType::Integer),
            None,
            None,
            false
        ));
    }
}
