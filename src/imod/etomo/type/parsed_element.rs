//! `IMOD/Etomo/src/etomo/type/ParsedElement.java`.
//!
//! The abstract base of the parsed Matlab/non-Matlab values (`ParsedNumber`,
//! `ParsedArray`, `ParsedList`, `ParsedDescriptor`, `ParsedQuotedString`, ...).
//!
//! **Shape.**  Java's abstract class becomes the trait `ParsedElement`; the fields the
//! class declares sit in `ParsedElementBase`, which every implementor embeds and returns
//! from `parsed_element_base`/`parsed_element_base_mut`.  The abstract methods are
//! required trait methods, the overridable concrete methods are default methods, and
//! the `final` methods are default methods an implementor must not override.  Parsed
//! elements are plain values owned by their parameter object, so mutators take
//! `&mut self`.  Java's overloaded members carry the parameter-type suffix
//! (`getRawString()` -> `get_raw_string_void`).

use super::const_etomo_number::Number;
use super::etomo_number::EtomoNumber;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::ui::swing::token::Token;
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// The fields Java's abstract `ParsedElement` declares.
#[derive(Clone, Debug)]
pub struct ParsedElementBase {
    /// Java private `failed`, initialised to false.
    failed: bool,
    /// Java private `missingAttribute`, initialised to false.
    missing_attribute: bool,
    /// Java private `lineNum`, initialised to -1.
    line_num: i32,
    /// Java private `errorMessage`, initialised to null.
    error_message: Option<String>,
    /// Java package-private final `descr`.
    pub descr: Option<String>,
}

impl ParsedElementBase {
    /// Java package-private `ParsedElement(String)`.
    pub fn new(descr: Option<&str>) -> ParsedElementBase {
        ParsedElementBase {
            failed: false,
            missing_attribute: false,
            line_num: -1,
            error_message: None,
            descr: descr.map(|descr| descr.to_string()),
        }
    }
}

/// Java `ParsedElement`.
pub trait ParsedElement {
    /// The fields Java's `ParsedElement` declares (not a source member).
    fn parsed_element_base(&self) -> &ParsedElementBase;

    /// Mutable access to the fields Java's `ParsedElement` declares (not a source
    /// member).
    fn parsed_element_base_mut(&mut self) -> &mut ParsedElementBase;

    /// Java abstract `getRawString()`.
    fn get_raw_string_void(&self) -> Option<String>;

    /// Java abstract `getRawString(int)`.
    fn get_raw_string_int(&self, index: i32) -> Option<String>;

    /// Java abstract `setDefault(int)`.
    fn set_default_int(&mut self, input: i32);

    /// Java abstract `setDebug(boolean)`.
    fn set_debug(&mut self, input: bool);

    /// Java abstract `equals(int)`.
    fn equals(&self, number: i32) -> bool;

    /// Java abstract package-private `setRawString(String, int)`.
    fn set_raw_string_string_int(&mut self, number: Option<&str>, line_num: i32);

    /// Java abstract package-private `getElement(int)`.
    fn get_element(&self, index: i32) -> Option<&dyn ParsedElement>;

    /// Java abstract package-private `setRawString(int, double)`.
    fn set_raw_string_int_double(&mut self, index: i32, number: f64);

    /// Java abstract package-private `setRawString(int, String)`.
    fn set_raw_string_int_string(&mut self, index: i32, string: Option<&str>);

    /// Java abstract package-private `parse(Token, PrimativeTokenizer, int)`.
    fn parse(
        &mut self,
        token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
        line_num: i32,
    ) -> Option<Box<Token>>;

    /// Java abstract package-private `size()`.
    fn size(&self) -> i32;

    /// Java abstract package-private `getParsableString()`.
    fn get_parsable_string(&self) -> Option<String>;

    /// Java abstract package-private `isCollection()`.
    fn is_collection(&self) -> bool;

    /// Java abstract package-private `isDescriptor()`.
    fn is_descriptor(&self) -> bool;

    /// Java abstract package-private `setDefault(EtomoNumber)`.
    fn set_default_etomo_number(&mut self, input: Option<&EtomoNumber>);

    /// Java abstract package-private `removeElement(int)`.
    fn remove_element(&mut self, index: i32);

    /// Java abstract package-private `ge(int)`.
    fn ge(&self, number: i32) -> bool;

    /// Java abstract `clear()`.
    fn clear(&mut self);

    // TODO(unit): needs etomo/type/ParsedElementList.java - Java abstract
    // package-private `ParsedElementList getParsedNumberExpandedArray(ParsedElementList
    // parsedNumberExpandedArray)`: "Append non-null ParsedNumbers to
    // parsedNumberExpandedArray.  Create parsedNumberExpandedArray if
    // parsedNumberExpandedArray == null.  Returns parsedNumberExpandedArray."  Declared
    // here as `fn get_parsed_number_expanded_array(&self, Option<ParsedElementList>) ->
    // Option<ParsedElementList>` once the list type exists; nothing in this class calls
    // it.

    /// Java final `setRawString(String)`.
    fn set_raw_string_string(&mut self, string: Option<&str>) {
        let line_num = self.parsed_element_base().line_num;
        self.set_raw_string_string_int(string, line_num);
    }

    /// Java `isQuoteFound()`.
    fn is_quote_found(&self) -> bool {
        false
    }

    /// Java final `isValid()`.
    fn is_valid(&self) -> bool {
        self.validate().is_none()
    }

    /// Java `validate()`.  Returns the first parse error.  If failed and there's no
    /// error message, returns a generic one.  Otherwise return null.
    fn validate(&self) -> Option<String> {
        let base = self.parsed_element_base();
        if let Some(error_message) = &base.error_message {
            return Some(error_message.clone());
        }
        if base.failed {
            return Some(format!(
                "{}: Unable to parse.{}",
                base.descr.as_deref().unwrap_or(""),
                if base.line_num > 0 {
                    format!("  Line# {}", base.line_num)
                } else {
                    String::new()
                }
            ));
        }
        None
    }

    /// Java final package-private `getErrorMessage()`.
    fn get_error_message(&self) -> Option<String> {
        self.parsed_element_base().error_message.clone()
    }

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        if self.size() == 0 {
            return true;
        }
        for i in 0..self.size() {
            let element = self.get_element(i);
            if let Some(element) = element
                && !element.is_empty()
            {
                return false;
            }
        }
        true
    }

    /// Java `getRawNumber()`.
    fn get_raw_number(&self) -> Option<Number> {
        let element = self.get_element(0);
        if let Some(element) = element {
            return element.get_raw_number();
        }
        None
    }

    /// Java final package-private `resetLineNum()`.
    fn reset_line_num(&mut self) {
        self.parsed_element_base_mut().line_num = -1;
    }

    /// Java final package-private `setLineNum(int)`.
    fn set_line_num(&mut self, line_num: i32) {
        self.parsed_element_base_mut().line_num = line_num;
    }

    /// Java final package-private `getLineNum()`.
    fn get_line_num(&self) -> i32 {
        self.parsed_element_base().line_num
    }

    /// Java package-private `isDefaultedEmpty()`.
    fn is_defaulted_empty(&self) -> bool {
        if self.size() == 0 {
            return true;
        }
        for i in 0..self.size() {
            let element = self.get_element(i);
            if let Some(element) = element
                && !element.is_defaulted_empty()
            {
                return false;
            }
        }
        true
    }

    /// Java final package-private `createTokenizer(String)`.  `PrimativeTokenizer`
    /// reads a null string as an empty one (`PrimativeTokenizer.java:327-331`), so a
    /// null value is passed as "".
    fn create_tokenizer(&mut self, value: Option<&str>) -> PrimativeTokenizer {
        let mut tokenizer = PrimativeTokenizer::get_string_instance(value.unwrap_or(""), false);
        match tokenizer.initialize() {
            Ok(()) => {}
            // `catch (final LockException e) { fail(e.getMessage()); }`
            Err(e @ LogFileError::Lock(_)) => {
                self.fail(Some(&e.get_message()));
            }
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                self.fail(Some(&e.get_message()));
            }
        }
        tokenizer
    }

    /// Java final package-private `fail(String)`.  Set failed and store the first error
    /// message.
    fn fail(&mut self, message: Option<&str>) {
        let base = self.parsed_element_base_mut();
        base.failed = true;
        if let Some(message) = message
            && base.error_message.is_none()
        {
            base.error_message = Some(format!(
                "{}  {}{}",
                base.descr.as_deref().unwrap_or(""),
                message,
                if base.line_num > 0 {
                    format!("  Line# {}", base.line_num)
                } else {
                    String::new()
                }
            ));
        }
    }

    /// Java final `isMissingAttribute()`.
    fn is_missing_attribute(&self) -> bool {
        self.parsed_element_base().missing_attribute
    }

    /// Java final package-private `setMissingAttribute()`.
    fn set_missing_attribute(&mut self) {
        self.parsed_element_base_mut().missing_attribute = true;
    }

    /// Java final package-private `resetFailed()`.
    fn reset_failed(&mut self) {
        let base = self.parsed_element_base_mut();
        base.failed = false;
        base.error_message = None;
    }

    /// Java final package-private `isFailed()`.
    fn is_failed(&self) -> bool {
        self.parsed_element_base().failed
    }
}
