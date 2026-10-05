//! `IMOD/Etomo/src/etomo/type/ParsedQuotedString.java`.
//!
//! A single-quoted Matlab string.  `ParsedQuotedString extends ParsedElement
//! implements Parsable`.
//!
//! The source wraps every `tokenizer.next` call in `catch (IOException e)`; the
//! translated `PrimativeTokenizer.next` reads from memory and cannot fail, so those
//! handlers have nothing to catch and are not reproduced.

use super::const_etomo_number::Number;
use super::etomo_number::EtomoNumber;
use super::parsable::Parsable;
use super::parsed_element::{ParsedElement, ParsedElementBase};
use super::parsed_element_list::ParsedElementList;
use super::parsed_element_type::{self, ParsedElementType};
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;
use crate::imod::etomo::util::quote::{self, Quote};

/// Java `public static final Character DELIMITER_SYMBOL = new Character('\'')`.
pub const DELIMITER_SYMBOL: char = '\'';
/// Java private static final `type = ParsedElementType.STRING`.
static TYPE: &ParsedElementType = &parsed_element_type::STRING;

/// Java `public final class ParsedQuotedString extends ParsedElement implements
/// Parsable`.
#[derive(Clone)]
pub struct ParsedQuotedString {
    /// Java superclass `ParsedElement` fields.
    base: ParsedElementBase,
    /// Java private `rawString`, initially "".
    raw_string: Option<String>,
    /// Java private `debug`, initially false.
    debug: bool,
    /// Java private `quoteFound`, initially false.
    quote_found: bool,
}

impl ParsedQuotedString {
    /// Java private `ParsedQuotedString(boolean, String)`.
    fn new(debug: bool, descr: Option<&str>) -> ParsedQuotedString {
        let mut instance = ParsedQuotedString {
            base: ParsedElementBase::new(descr),
            raw_string: Some(String::new()),
            debug: false,
            quote_found: false,
        };
        instance.set_debug(debug);
        instance
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(descr: Option<&str>) -> ParsedQuotedString {
        ParsedQuotedString::new(false, descr)
    }

    /// Java package-private static `getInstance(boolean, String)`.
    pub fn get_instance_boolean(debug: bool, descr: Option<&str>) -> ParsedQuotedString {
        ParsedQuotedString::new(debug, descr)
    }

    /// Java static `isQuotedString(ReadOnlyAttribute)`.  This is a quoted string only
    /// if starts with a quote (ignoring whitespace).
    pub fn is_quoted_string_attribute(attribute: Option<&dyn ReadOnlyAttribute>) -> bool {
        let Some(attribute) = attribute else {
            return false;
        };
        ParsedQuotedString::is_quoted_string(attribute.get_value().as_deref())
    }

    /// Java static `isQuotedString(String)`.  This is a quoted string only if starts
    /// with a quote (ignoring whitespace).
    pub fn is_quoted_string(value: Option<&str>) -> bool {
        let Some(value) = value else {
            return false;
        };
        // Java `value.trim().charAt(0)` throws StringIndexOutOfBoundsException on a
        // blank value.  Fixed in translation: a blank value is not a quoted string.
        value
            .trim_matches(|c: char| (c as u32) <= 0x20)
            .starts_with(DELIMITER_SYMBOL)
    }

    /// Java private `parse(String, int)`.
    fn parse_string_int(&mut self, parsable_string: Option<&str>, line_num: i32) {
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        let Some(parsable_string) = parsable_string else {
            self.set_missing_attribute();
            return;
        };
        let mut tokenizer = self.create_tokenizer(Some(parsable_string));
        let mut token = None;
        tokenizer.next(&mut token);
        self.parse(token, &mut tokenizer, line_num);
    }

    /// Java package-private `setRawString(int, String, int)`.
    pub fn set_raw_string_int_string_int(
        &mut self,
        index: i32,
        string: Option<&str>,
        _line_num: i32,
    ) {
        if index != 0 {
            return;
        }
        self.raw_string = string.map(str::to_owned);
    }

    /// Java `setElement(ParsedElement)`.
    pub fn set_element(&mut self, element: Option<&dyn ParsedElement>) {
        match element {
            Some(element) => self.raw_string = element.get_raw_string_void(),
            None => self.clear(),
        }
    }

    /// Java package-private `moveElement(int, int)`: empty.
    pub fn move_element(&mut self, _from_index: i32, _to_index: i32) {}

    /// Java private `parseElement(Token, PrimativeTokenizer, int)`.
    fn parse_element(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
        line_num: i32,
    ) -> Option<Box<Token>> {
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        token.as_ref()?;
        // Loop until DELIMITER_SYMBOL, EOL, EOF is found; that should be the end of
        // the string.
        let mut buffer = String::new();
        while !self.is_failed()
            && let Some(current) = token.as_ref()
            && !current.equals_type_and_char(TokenType::Symbol, DELIMITER_SYMBOL as u16)
            && !current.is(TokenType::Eol)
            && !current.is(TokenType::Eof)
        {
            // build the string
            buffer.push_str(current.get_value().unwrap_or("null"));
            tokenizer.next(&mut token);
        }
        self.raw_string = Some(buffer);
        token
    }

    /// The quote mark a token is, for the wrong-quote checks
    /// (`Quote.getInstance(token.getChar())`).
    fn token_quote(token: &Token) -> Option<&'static Quote> {
        Quote::get_instance_character(char::from_u32(token.get_char() as u32))
    }
}

impl ParsedElement for ParsedQuotedString {
    fn parsed_element_base(&self) -> &ParsedElementBase {
        &self.base
    }

    fn parsed_element_base_mut(&mut self) -> &mut ParsedElementBase {
        &mut self.base
    }

    /// Java `getRawString()`.
    fn get_raw_string_void(&self) -> Option<String> {
        self.raw_string.clone()
    }

    /// Java `getRawString(int)`.
    fn get_raw_string_int(&self, index: i32) -> Option<String> {
        if index == 0 {
            return self.raw_string.clone();
        }
        ParsedQuotedString::new(self.debug, self.base.descr.as_deref()).get_raw_string_void()
    }

    /// Java `setDefault(int)`: empty.
    fn set_default_int(&mut self, _input: i32) {}

    /// Java `setDebug(boolean)`.
    fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java `equals(int)`.
    fn equals(&self, _number: i32) -> bool {
        false
    }

    /// Java `setRawString(String, int)`.
    fn set_raw_string_string_int(&mut self, string: Option<&str>, _line_num: i32) {
        self.raw_string = string.map(str::to_owned);
    }

    /// Java package-private `getElement(int)`.
    fn get_element(&self, index: i32) -> Option<&dyn ParsedElement> {
        if index == 0 {
            return Some(self);
        }
        None
    }

    /// Java package-private `setRawString(int, double)`.
    fn set_raw_string_int_double(&mut self, index: i32, number: f64) {
        if index > 0 {
            return;
        }
        self.raw_string = Some(
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(number),
        );
    }

    /// Java package-private `setRawString(int, String)`.
    fn set_raw_string_int_string(&mut self, index: i32, string: Option<&str>) {
        if index != 0 {
            return;
        }
        let line_num = self.get_line_num();
        self.set_raw_string_string_int(string, line_num);
    }

    /// Java package-private `parse(Token, PrimativeTokenizer, int)`.
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
        if token.as_ref().unwrap().is(TokenType::Whitespace) {
            tokenizer.next(&mut token);
        }
        if !token.as_ref().is_some_and(|token| {
            token.equals_type_and_char(TokenType::Symbol, DELIMITER_SYMBOL as u16)
        }) {
            let wrong_quote = token
                .as_ref()
                .is_some_and(|token| ParsedQuotedString::token_quote(token).is_some());
            if wrong_quote {
                self.quote_found = true;
                self.fail(Some(&format!(
                    "Wrong type of quote.  Use {}",
                    quote::SINGLE.get_descr()
                )));
                return token;
            }
            self.fail(Some(&format!(
                "String requires single quotes ( {DELIMITER_SYMBOL} )"
            )));
            return token;
        }
        self.quote_found = true;
        tokenizer.next(&mut token);
        // everything within DELIMITER symbols is part of the rawString.
        token = self.parse_element(token, tokenizer, line_num);
        if self.is_failed() {
            return token;
        }
        if !token.as_ref().is_some_and(|token| {
            token.equals_type_and_char(TokenType::Symbol, DELIMITER_SYMBOL as u16)
        }) {
            let wrong_quote = token
                .as_ref()
                .is_some_and(|token| ParsedQuotedString::token_quote(token).is_some());
            if wrong_quote {
                self.fail(Some(&format!(
                    "Mismatched quotes.  Use {}",
                    quote::SINGLE.get_descr()
                )));
                return token;
            }
            self.fail(Some(&format!(
                "Unclosed quotation mark ( {DELIMITER_SYMBOL} )"
            )));
            return token;
        }
        tokenizer.next(&mut token);
        token
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        1
    }

    /// Java `getParsableString()`.
    fn get_parsable_string(&self) -> Option<String> {
        let mut buffer = DELIMITER_SYMBOL.to_string();
        buffer.push_str(self.raw_string.as_deref().unwrap_or("null"));
        buffer.push(DELIMITER_SYMBOL);
        Some(buffer)
    }

    /// Java package-private `isCollection()`.
    fn is_collection(&self) -> bool {
        false
    }

    /// Java package-private `isDescriptor()`.
    fn is_descriptor(&self) -> bool {
        false
    }

    /// Java package-private `setDefault(EtomoNumber)`: empty.
    fn set_default_etomo_number(&mut self, _input: Option<&EtomoNumber>) {}

    /// Java package-private `removeElement(int)`.
    fn remove_element(&mut self, index: i32) {
        if index == 0 {
            self.clear();
        }
    }

    /// Java package-private `ge(int)`.
    fn ge(&self, _number: i32) -> bool {
        false
    }

    /// Java `clear()`.
    fn clear(&mut self) {
        self.raw_string = Some(String::new());
        self.quote_found = false;
        self.reset_line_num();
    }

    /// Java package-private `getParsedNumberExpandedArray(ParsedElementList)`.
    fn get_parsed_number_expanded_array(
        &self,
        parsed_number_expanded_array: Option<ParsedElementList>,
    ) -> ParsedElementList {
        match parsed_number_expanded_array {
            Some(list) => list,
            None => {
                ParsedElementList::new(TYPE, None, self.debug, None, self.base.descr.as_deref())
            }
        }
    }

    fn clone_element(&self) -> Box<dyn ParsedElement> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        self.raw_string.as_deref().is_none_or(str::is_empty)
    }

    /// Java `getRawNumber()`.
    fn get_raw_number(&self) -> Option<Number> {
        let mut etomo_number = EtomoNumber::new();
        etomo_number.set_string(self.raw_string.as_deref());
        Some(etomo_number.get_number())
    }

    /// Java `validate()`.
    fn validate(&self) -> Option<String> {
        let error_message = self.get_error_message();
        if error_message.is_some() {
            return error_message;
        }
        self.base.validate()
    }

    /// Java `isQuoteFound()`.
    fn is_quote_found(&self) -> bool {
        self.quote_found
    }

    /// Java package-private `isDefaultedEmpty()`.
    fn is_defaulted_empty(&self) -> bool {
        ParsedElement::is_empty(self)
    }
}

impl Parsable for ParsedQuotedString {
    fn clear_parsable(&mut self) {
        ParsedElement::clear(self);
    }

    /// Java `parse(String)`.
    fn parse_string(&mut self, parsable_string: Option<&str>) {
        self.parse_string_int(parsable_string, -1);
    }

    fn validate_parsable(&self) -> Option<String> {
        ParsedElement::validate(self)
    }

    fn get_parsable_string_parsable(&self) -> Option<String> {
        ParsedElement::get_parsable_string(self)
    }

    /// Java `parse(ReadOnlyAttribute)`.  Parse attribute value and set the list
    /// member variable.
    fn parse_attribute(&mut self, attribute: Option<&dyn ReadOnlyAttribute>) {
        let Some(attribute) = attribute else {
            self.set_missing_attribute();
            return;
        };
        self.parse_string_int(attribute.get_value().as_deref(), attribute.get_line_num());
    }

    fn is_empty_parsable(&self) -> bool {
        ParsedElement::is_empty(self)
    }

    fn size_parsable(&self) -> i32 {
        ParsedElement::size(self)
    }
}

/// Java `toString()`.
impl std::fmt::Display for ParsedQuotedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[rawString:{}]",
            self.raw_string.as_deref().unwrap_or("null")
        )
    }
}
