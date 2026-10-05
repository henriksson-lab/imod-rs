//! `IMOD/Etomo/src/etomo/type/ParsedNumber.java`.
//!
//! A number parsed from a Matlab-style or plain value.  `ParsedNumber extends
//! ParsedElement`: the superclass fields are the embedded `ParsedElementBase`.
//!
//! The source wraps every `tokenizer.next` call in `catch (IOException e)`; the
//! translated `PrimativeTokenizer.next` reads from memory and cannot fail, so those
//! handlers have nothing to catch and are not reproduced.

use super::const_etomo_number::{ConstEtomoNumber, Number, Type};
use super::etomo_number::EtomoNumber;
use super::parsed_array;
use super::parsed_descriptor;
use super::parsed_element::{ParsedElement, ParsedElementBase};
use super::parsed_element_list::ParsedElementList;
use super::parsed_element_type::{self, ParsedElementType};
use super::parsed_list;
use super::parsed_quoted_string;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// Java `public final class ParsedNumber extends ParsedElement`.
#[derive(Clone)]
pub struct ParsedNumber {
    /// Java superclass `ParsedElement` fields.
    base: ParsedElementBase,
    /// Java private final `rawNumber`.
    raw_number: EtomoNumber,
    /// Java private final `etomoNumberType`.
    etomo_number_type: Option<Type>,
    /// Java private final `NON_ELEMENT_SYMBOLS`.
    non_element_symbols: String,
    /// Java private final `type`.
    r#type: &'static ParsedElementType,
    /// Java private `defaultValue`, initially null and never assigned by the source.
    default_value: Option<EtomoNumber>,
    /// Java private `debug`, initially false.
    debug: bool,
}

impl ParsedNumber {
    /// Java private `ParsedNumber(ParsedElementType, EtomoNumber.Type, boolean,
    /// EtomoNumber, String)`.
    fn new(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedNumber {
        let mut raw_number = EtomoNumber::new_with_type(etomo_number_type);
        raw_number.set_default_const_etomo_number(default_value.map(|d| &**d));
        let non_element_symbols = format!(
            "{}{}{}{}{}{}{}{}",
            parsed_list::OPEN_SYMBOL,
            parsed_list::CLOSE_SYMBOL,
            parsed_array::OPEN_SYMBOL,
            parsed_array::CLOSE_SYMBOL,
            parsed_quoted_string::DELIMITER_SYMBOL,
            parsed_list::DIVIDER_SYMBOL,
            parsed_array::DIVIDER_SYMBOL,
            parsed_descriptor::get_divider_symbol(r#type)
        );
        let mut instance = ParsedNumber {
            base: ParsedElementBase::new(descr),
            raw_number,
            etomo_number_type,
            non_element_symbols,
            r#type,
            default_value: None,
            debug: false,
        };
        instance.set_debug(debug);
        instance
    }

    /// Java `getMatlabInstance(String)`.
    pub fn get_matlab_instance(descr: Option<&str>) -> ParsedNumber {
        ParsedNumber::new(
            &parsed_element_type::MATLAB_NUMBER,
            None,
            false,
            None,
            descr,
        )
    }

    /// Java `getMatlabInstance(EtomoNumber.Type, String)`.
    pub fn get_matlab_instance_type(
        etomo_number_type: Option<Type>,
        descr: Option<&str>,
    ) -> ParsedNumber {
        ParsedNumber::new(
            &parsed_element_type::MATLAB_NUMBER,
            etomo_number_type,
            false,
            None,
            descr,
        )
    }

    /// Java package-private static `getInstance(ParsedElementType, EtomoNumber.Type,
    /// boolean, EtomoNumber, String)`.
    pub fn get_instance(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedNumber {
        ParsedNumber::new(r#type, etomo_number_type, debug, default_value, descr)
    }

    /// Java `parse(ReadOnlyAttribute)`.
    pub fn parse_attribute(&mut self, attribute: Option<&dyn ReadOnlyAttribute>) {
        self.clear();
        self.reset_failed();
        let Some(attribute) = attribute else {
            self.set_missing_attribute();
            return;
        };
        self.set_line_num(attribute.get_line_num());
        let mut tokenizer = self.create_tokenizer(attribute.get_value().as_deref());
        let mut token = None;
        tokenizer.next(&mut token);
        self.parse(token, &mut tokenizer, attribute.get_line_num());
    }

    /// Java `getRawBoolean()`.
    pub fn get_raw_boolean(&self) -> bool {
        self.raw_number.get_defaulted_boolean()
    }

    /// Java `getEtomoNumber()`.
    pub fn get_etomo_number(&self) -> &ConstEtomoNumber {
        &self.raw_number
    }

    /// Java `getNegatedRawNumber()`.
    pub fn get_negated_raw_number(&self) -> Number {
        self.raw_number.get_negated_defaulted_number()
    }

    /// Java `setRawString(boolean)`.
    pub fn set_raw_string_boolean(&mut self, bool: bool) {
        self.raw_number.set_boolean(bool);
    }

    /// Java `setElement(ParsedElement)`.
    pub fn set_element(&mut self, element: Option<&dyn ParsedElement>) {
        self.clear();
        self.reset_failed();
        if let Some(element) = element {
            let raw_string = element.get_raw_string_void();
            if raw_string.as_deref() != Some("NaN") {
                self.raw_number.set_string(raw_string.as_deref());
            }
        }
    }

    /// Java package-private `setRawString(BaseManager, String, String)`.
    pub fn set_raw_string_base_manager_string_string(
        &mut self,
        manager: Option<&'static dyn BaseManager>,
        number: &str,
        field_description: Option<&str>,
    ) {
        self.clear();
        self.reset_failed();
        if number != "NaN" {
            self.raw_number.set_string(Some(number));
        }
        if let Some(field_description) = field_description {
            let error_message = self.raw_number.validate(Some(field_description));
            if let Some(error_message) = error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        manager,
                        &error_message,
                        "Entry Error",
                    )
                });
            }
        }
    }

    /// Java package-private `setRawString(double)`.
    pub fn set_raw_string_double(&mut self, number: f64) {
        self.raw_number.set_double(number);
    }

    /// Java `equals(ParsedNumber)`.
    pub fn equals_parsed_number(&self, input: &ParsedNumber) -> bool {
        self.raw_number
            .equals_const_etomo_number(Some(&input.raw_number))
    }

    /// Java package-private `isPositive()`.
    pub fn is_positive(&self) -> bool {
        self.raw_number.is_positive()
    }

    /// Java package-private `isNegative()`.
    pub fn is_negative(&self) -> bool {
        self.raw_number.is_negative()
    }

    /// Java package-private `le(ParsedNumber)`.
    pub fn le(&self, element: &ParsedNumber) -> bool {
        self.raw_number
            .lt_const_etomo_number(Some(&element.raw_number))
            || self
                .raw_number
                .equals_const_etomo_number(Some(&element.raw_number))
    }

    /// Java package-private `lt(ParsedNumber)`.
    pub fn lt(&self, element: &ParsedNumber) -> bool {
        self.raw_number
            .lt_const_etomo_number(Some(&element.raw_number))
    }

    /// Java package-private `ge(ParsedNumber)`.
    pub fn ge_parsed_number(&self, element: &ParsedNumber) -> bool {
        self.raw_number
            .gt_const_etomo_number(Some(&element.raw_number))
            || self
                .raw_number
                .equals_const_etomo_number(Some(&element.raw_number))
    }

    /// Java package-private `gt(ParsedNumber)`.
    pub fn gt(&self, element: &ParsedNumber) -> bool {
        self.raw_number
            .gt_const_etomo_number(Some(&element.raw_number))
    }

    /// Java `setRawString(Number)`.
    pub fn set_raw_string_number(&mut self, number: Option<Number>) {
        self.raw_number.set_number(number);
    }

    /// Java package-private `plus(ConstEtomoNumber)`.
    pub fn plus(&mut self, number: &ConstEtomoNumber) {
        self.raw_number.add_const_etomo_number(Some(number));
    }

    /// Java `setDefault(boolean)`.
    pub fn set_default_boolean(&mut self, input: bool) {
        self.raw_number.set_default_boolean(input);
    }

    /// Java `setFloor(int)`.
    pub fn set_floor(&mut self, input: i32) {
        self.raw_number.set_floor(input);
    }

    /// Java private `parseElement(Token, PrimativeTokenizer)`.
    fn parse_element(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
    ) -> Option<Box<Token>> {
        token.as_ref()?;
        // Loop until whitespace, EOL, EOF, or a recognized symbol is found; that
        // should be the end of the number.
        let mut buffer = String::new();
        while !self.is_failed()
            && let Some(current) = token.as_ref()
            && !current.is(TokenType::Whitespace)
            && !current.is(TokenType::Eol)
            && !current.is(TokenType::Eof)
            && (!current.is(TokenType::Symbol)
                || !self
                    .non_element_symbols
                    .encode_utf16()
                    .any(|symbol| symbol == current.get_char()))
        {
            // build the number
            buffer.push_str(current.get_value().unwrap_or("null"));
            if self.debug {
                println!("ParsedNumber.parseElement:buffer={buffer}");
            }
            tokenizer.next(&mut token);
        }
        if buffer != "NaN" {
            self.raw_number.set_string(Some(&buffer));
        } else {
            self.clear();
        }
        token
    }
}

impl ParsedElement for ParsedNumber {
    fn parsed_element_base(&self) -> &ParsedElementBase {
        &self.base
    }

    fn parsed_element_base_mut(&mut self) -> &mut ParsedElementBase {
        &mut self.base
    }

    /// Java `getRawString()`.
    fn get_raw_string_void(&self) -> Option<String> {
        Some(self.raw_number.to_defaulted_string())
    }

    /// Java `getRawString(int)`.  When an index is passed, treat ParsedNumber as an
    /// array of 1.
    fn get_raw_string_int(&self, index: i32) -> Option<String> {
        if index == 0 {
            return self.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java `setDefault(int)`.
    fn set_default_int(&mut self, input: i32) {
        self.raw_number.set_default_int(input);
    }

    /// Java `setDebug(boolean)`.
    fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.raw_number.set_debug(input);
    }

    /// Java `equals(int)`.
    fn equals(&self, number: i32) -> bool {
        self.raw_number.equals_int(number)
    }

    /// Java `setRawString(String, int)`.
    fn set_raw_string_string_int(&mut self, number: Option<&str>, line_num: i32) {
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        // Fixed in translation: Java's `number.equals("NaN")` throws
        // NullPointerException for a null number; a null number is set (an empty
        // value) here.
        if number != Some("NaN") {
            self.raw_number.set_string(number);
        }
        if self.raw_number.is_valid() {
            self.reset_failed();
        } else {
            let reason = self.raw_number.get_invalid_reason();
            self.fail(Some(&reason));
        }
    }

    /// Java `getElement(int)`.  When an index is passed, treat ParsedNumber as an
    /// array of 1.  Return null for any other index.
    fn get_element(&self, index: i32) -> Option<&dyn ParsedElement> {
        if index == 0 {
            return Some(self);
        }
        None
    }

    /// Java package-private `setRawString(int, double)`.
    fn set_raw_string_int_double(&mut self, index: i32, number: f64) {
        if index != 0 {
            return;
        }
        self.raw_number.set_double(number);
    }

    /// Java package-private `setRawString(int, String)`.
    fn set_raw_string_int_string(&mut self, index: i32, string: Option<&str>) {
        if index != 0 {
            return;
        }
        let line_num = self.get_line_num();
        self.set_raw_string_string_int(string, line_num);
    }

    /// Java package-private `parse(Token, PrimativeTokenizer, int)`.  Parse the
    /// number including delimiters.  Returns the token that is current when the number
    /// is parsed.
    fn parse(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
        line_num: i32,
    ) -> Option<Box<Token>> {
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        token.as_ref()?;
        let mut close_symbol: Option<char> = None;
        if !self.r#type.is_array() {
            if token.as_ref().unwrap().is(TokenType::Whitespace) {
                tokenizer.next(&mut token);
            }
            let current = token.as_ref()?;
            // If the number is not in an array, it may still have delimiters
            // (either [] or ''). Find opening delimiter.
            if current.equals_type_and_char(TokenType::Symbol, parsed_array::OPEN_SYMBOL as u16) {
                close_symbol = Some(parsed_array::CLOSE_SYMBOL);
                tokenizer.next(&mut token);
            } else if current.equals_type_and_char(
                TokenType::Symbol,
                parsed_quoted_string::DELIMITER_SYMBOL as u16,
            ) {
                close_symbol = Some(parsed_quoted_string::DELIMITER_SYMBOL);
                tokenizer.next(&mut token);
            }
        }
        // Remove any whitespace before the element.
        if token
            .as_ref()
            .is_some_and(|token| token.is(TokenType::Whitespace))
        {
            tokenizer.next(&mut token);
        }
        token = self.parse_element(token, tokenizer);
        if self.debug {
            println!("ParsedNumber.parse:rawNumber={}", self.raw_number);
        }
        if self.is_failed() {
            return token;
        }
        // Find closing delimiter
        if let Some(close_symbol) = close_symbol {
            if token
                .as_ref()
                .is_some_and(|token| token.is(TokenType::Whitespace))
            {
                tokenizer.next(&mut token);
            }
            let Some(current) = token.as_ref() else {
                self.fail(Some(&format!(
                    "End of value.  Closing delimiter, {close_symbol}, was not found."
                )));
                return token;
            };
            // If the number is not in an array, it may have delimiters
            // (either [] or '').
            if current.equals_type_and_char(TokenType::Symbol, close_symbol as u16) {
                tokenizer.next(&mut token);
            } else {
                self.fail(Some(&format!(
                    "Closing delimiter, {close_symbol}, was not found."
                )));
                return token;
            }
        }
        token
    }

    /// Java package-private `size()`.
    fn size(&self) -> i32 {
        1
    }

    /// Java `getParsableString()`.  Return the raw number.  If the type is double,
    /// return it as an int or long if there is no decimal value.
    fn get_parsable_string(&self) -> Option<String> {
        if self.raw_number.is_defaulted_null() {
            if self.r#type.is_matlab() {
                // Empty strings cannot be parsed by MatLab. If this instance is a
                // Matlab syntax instance, return NaN (unless it is an array descriptor
                // because they can't contain NaN).
                if std::ptr::eq(self.r#type, &parsed_element_type::MATLAB_ARRAY_DESCRIPTOR) {
                    return Some(String::new());
                }
                return Some("NaN".to_owned());
            } else {
                return Some(String::new());
            }
        }
        let number = self.raw_number.get_defaulted_number();
        // Remove unnecessary decimal points.
        if self.etomo_number_type == Some(Type::Double) {
            let double_number = number.double_value();
            // Java `Math.round(doubleNumber) == doubleNumber`: only an integral value
            // compares equal to its rounded long.
            if ((double_number + 0.5).floor() as i64) as f64 == double_number {
                return Some(number.int_value().to_string());
            }
        }
        Some(number.to_string())
    }

    /// Java package-private `isCollection()`.
    fn is_collection(&self) -> bool {
        false
    }

    /// Java package-private `isDescriptor()`.
    fn is_descriptor(&self) -> bool {
        false
    }

    /// Java package-private `setDefault(EtomoNumber)`.
    fn set_default_etomo_number(&mut self, input: Option<&EtomoNumber>) {
        self.raw_number
            .set_default_const_etomo_number(input.map(|input| &**input));
    }

    /// Java package-private `removeElement(int)`.
    fn remove_element(&mut self, index: i32) {
        if index == 0 {
            self.clear();
            self.reset_failed();
        }
    }

    /// Java package-private `ge(int)`.
    fn ge(&self, number: i32) -> bool {
        self.raw_number.gt_int(number) || self.raw_number.equals_int(number)
    }

    /// Java `clear()`.
    fn clear(&mut self) {
        self.raw_number.reset();
        self.reset_line_num();
    }

    /// Java package-private `getParsedNumberExpandedArray(ParsedElementList)`.  If
    /// rawNumber is not null append this to parsedNumberExpandedArray.
    fn get_parsed_number_expanded_array(
        &self,
        parsed_number_expanded_array: Option<ParsedElementList>,
    ) -> ParsedElementList {
        let mut parsed_number_expanded_array = match parsed_number_expanded_array {
            Some(list) => list,
            None => ParsedElementList::new(
                self.r#type,
                self.etomo_number_type,
                self.debug,
                self.default_value.as_ref(),
                self.base.descr.as_deref(),
            ),
        };
        if self.raw_number.is_null() {
            return parsed_number_expanded_array;
        }
        parsed_number_expanded_array.add(Some(self.clone_element()));
        parsed_number_expanded_array
    }

    fn clone_element(&self) -> Box<dyn ParsedElement> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Java `getRawNumber()`.
    fn get_raw_number(&self) -> Option<Number> {
        Some(self.raw_number.get_defaulted_number())
    }

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        self.raw_number.is_null()
    }

    /// Java `validate()`.  Returns the first parse error or the invalid reason of the
    /// number.  If failed and there's no error messages, returns a generic one.
    /// Otherwise return null.
    fn validate(&self) -> Option<String> {
        let error_message = self.get_error_message();
        if error_message.is_some() {
            return error_message;
        }
        if !self.raw_number.is_valid() {
            let invalid_reason = self.raw_number.get_invalid_reason();
            return Some(format!(
                "{}: {}",
                self.base.descr.as_deref().unwrap_or(""),
                invalid_reason
            ));
        }
        self.base.validate()
    }

    /// Java package-private `isDefaultedEmpty()`.
    fn is_defaulted_empty(&self) -> bool {
        self.raw_number.is_defaulted_null()
    }
}

/// Java `toString()`.
impl std::fmt::Display for ParsedNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[rawNumber:{}]", self.raw_number)
    }
}
