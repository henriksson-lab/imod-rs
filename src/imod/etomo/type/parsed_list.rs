//! `IMOD/Etomo/src/etomo/type/ParsedList.java`.
//!
//! A cell array in Matlab.  An expandable array of elements.  The array does not
//! have to be fully populated.  Use parse and toString functions to load and
//! retrieve parsable data.  The add, set, and get functions refer to raw data.
//! When the STRING type is set, it only parses quoted strings; otherwise it only
//! parses numbers and arrays.  (The source's long Matlab syntax comment is in
//! `ParsedList.java:19-62`.)
//!
//! The source wraps every `tokenizer.next` call in `catch (IOException e)`; the
//! translated `PrimativeTokenizer.next` reads from memory and cannot fail, so those
//! handlers have nothing to catch and are not reproduced.

use super::const_etomo_number::Type;
use super::etomo_number::EtomoNumber;
use super::parsable::Parsable;
use super::parsed_array::{self, ParsedArray};
use super::parsed_descriptor::{self, ParsedDescriptor, matches_whitespace};
use super::parsed_element::ParsedElement;
use super::parsed_element_list::ParsedElementList;
use super::parsed_element_type::{self, ParsedElementType};
use super::parsed_number::ParsedNumber;
use super::parsed_quoted_string::ParsedQuotedString;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::util::bracket::{self, Bracket};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// Java `public static final Character OPEN_SYMBOL = new Character('{')`.
pub const OPEN_SYMBOL: char = '{';
/// Java `public static final Character CLOSE_SYMBOL = new Character('}')`.
pub const CLOSE_SYMBOL: char = '}';
/// Java `public static final Character DIVIDER_SYMBOL = new Character(',')`.
pub const DIVIDER_SYMBOL: char = ',';

/// Java `public final class ParsedList implements Parsable`.
pub struct ParsedList {
    /// Java private final `type`.
    r#type: &'static ParsedElementType,
    /// Java private final `list`.
    list: ParsedElementList,
    /// Java private final `etomoNumberType`.
    etomo_number_type: Option<Type>,
    /// Java private final `descr`.
    descr: Option<String>,
    /// Java private final `optionalBracket`.
    optional_bracket: bool,
    /// Java private `defaultValue`, initially null.
    default_value: Option<EtomoNumber>,
    /// Java private `failed`, initially false.
    failed: bool,
    /// Java private `debug`, initially false.
    debug: bool,
    /// Java private `bracketFound`, initially false.
    bracket_found: bool,
    /// Java private `quoteFound`, initially false.
    quote_found: bool,
    /// Java private `lineNum`, initially -1.
    line_num: i32,
    /// Java private `errorMessage`, initially null.
    error_message: Option<String>,
}

impl ParsedList {
    /// Java private `ParsedList(ParsedElementType, EtomoNumber.Type, String, boolean)`.
    fn new(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        descr: Option<&str>,
        optional_bracket: bool,
    ) -> ParsedList {
        ParsedList {
            r#type,
            // `debug` and `defaultValue` still hold their initial values here.
            list: ParsedElementList::new(r#type, etomo_number_type, false, None, descr),
            etomo_number_type,
            descr: descr.map(str::to_owned),
            optional_bracket,
            default_value: None,
            failed: false,
            debug: false,
            bracket_found: false,
            quote_found: false,
            line_num: -1,
            error_message: None,
        }
    }

    /// Java `getMatlabInstance(String)`.
    pub fn get_matlab_instance(descr: Option<&str>) -> ParsedList {
        ParsedList::new(&parsed_element_type::MATLAB_NUMBER, None, descr, false)
    }

    /// Java `getMatlabInstance(EtomoNumber.Type, String)`.
    pub fn get_matlab_instance_type(
        etomo_number_type: Option<Type>,
        descr: Option<&str>,
    ) -> ParsedList {
        ParsedList::new(
            &parsed_element_type::MATLAB_NUMBER,
            etomo_number_type,
            descr,
            false,
        )
    }

    /// Java `getStringInstance(String)`.
    pub fn get_string_instance(descr: Option<&str>) -> ParsedList {
        ParsedList::new(&parsed_element_type::STRING, None, descr, false)
    }

    /// Java `getOptionalBracketStringInstance(String)`.
    pub fn get_optional_bracket_string_instance(descr: Option<&str>) -> ParsedList {
        ParsedList::new(&parsed_element_type::STRING, None, descr, true)
    }

    /// Java static `isList(ReadOnlyAttribute)`.  This is a list only if starts with
    /// "{" (ignoring whitespace).
    pub fn is_list(attribute: Option<&dyn ReadOnlyAttribute>) -> bool {
        let Some(attribute) = attribute else {
            return false;
        };
        let Some(value) = attribute.get_value() else {
            return false;
        };
        // Java `value.trim().charAt(0)` throws on a blank value.  Fixed in
        // translation: a blank value is not a list.
        value
            .trim_matches(|c: char| (c as u32) <= 0x20)
            .starts_with(OPEN_SYMBOL)
    }

    /// Java static `isStringList(ReadOnlyAttribute)`.  This is a string list only if
    /// starts with "{'" (ignoring whitespace).
    pub fn is_string_list(attribute: Option<&dyn ReadOnlyAttribute>) -> bool {
        if ParsedList::is_list(attribute) {
            // String off the list character and check for the string character
            let value = attribute.unwrap().get_value().unwrap();
            let trimmed = value.trim_matches(|c: char| (c as u32) <= 0x20);
            return ParsedQuotedString::is_quoted_string(Some(&trimmed[1..]));
        }
        false
    }

    /// Java static `containsDivider(String)`.  Returns true if it contains commas or
    /// embedded whitespace.  This means that it could be an array or list.
    pub fn contains_divider(text: &str) -> bool {
        let text = text.trim_matches(|c: char| (c as u32) <= 0x20);
        if text.contains(DIVIDER_SYMBOL) {
            return true;
        }
        // Java `Character.isWhitespace`.
        text.chars().any(|c| {
            c.is_whitespace() && !matches!(c, '\u{00A0}' | '\u{2007}' | '\u{202F}')
                || matches!(c, '\u{1C}'..='\u{1F}')
        })
    }

    /// Java `setDefault(int)`.
    pub fn set_default(&mut self, input: i32) {
        if self.default_value.is_none() {
            self.default_value = Some(EtomoNumber::new_with_type(self.etomo_number_type));
        }
        self.default_value.as_mut().unwrap().set_int(input);
        self.list.set_default(self.default_value.as_ref());
        let default_value = self.default_value.clone();
        for i in 0..self.list.size() {
            if let Some(element) = self.list.get_mut(i) {
                element.set_default_etomo_number(default_value.as_ref());
            }
        }
    }

    /// Java `getElement(int)`.
    pub fn get_element(&self, index: i32) -> Option<&dyn ParsedElement> {
        self.list.get(index)
    }

    /// Java `getRawString(int)`.
    pub fn get_raw_string_int(&self, index: i32) -> Option<String> {
        if let Some(element) = self.list.get(index) {
            return element.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java `getRawString()`.
    pub fn get_raw_string(&self) -> String {
        let mut builder = String::new();
        let size = self.list.size();
        for i in 0..size {
            if i > 0 {
                builder.push(DIVIDER_SYMBOL);
            }
            builder.push_str(
                &self
                    .get_raw_string_int(i)
                    .unwrap_or_else(|| "null".to_owned()),
            );
        }
        builder
    }

    /// Java `addElement(ParsedElement)`.
    pub fn add_element(&mut self, mut element: Box<dyn ParsedElement>) {
        element.set_debug(self.debug);
        element.set_default_etomo_number(self.default_value.as_ref());
        self.list.add(Some(element));
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.list.set_debug(input);
        for i in 0..self.list.size() {
            if let Some(element) = self.list.get_mut(i) {
                element.set_debug(input);
            }
        }
    }

    /// Java private `parse(String, int)`.  Parse parsableString and set the list
    /// member variable.
    fn parse_string_int(&mut self, parsable_string: Option<&str>, line_num: i32) {
        self.clear_parsable();
        self.reset_failed();
        self.line_num = line_num;
        let Some(parsable_string) = parsable_string else {
            return;
        };
        let mut tokenizer = self.create_tokenizer(parsable_string);
        let mut token: Option<Box<Token>> = None;
        tokenizer.next(&mut token);
        if token.is_none() {
            return;
        }
        if token.as_ref().unwrap().is(TokenType::Whitespace) {
            tokenizer.next(&mut token);
        }
        self.bracket_found = false;
        let token_char = |token: &Token| char::from_u32(token.get_char() as u32);
        if !token
            .as_ref()
            .is_some_and(|token| token.equals_type_and_char(TokenType::Symbol, OPEN_SYMBOL as u16))
        {
            if !self.optional_bracket {
                self.fail(Some(&format!("Missing bracket: '{OPEN_SYMBOL}'")));
                return;
            }
            if token.as_ref().is_some_and(|token| {
                Bracket::get_open_instance_character(token_char(token)).is_some()
            }) {
                self.bracket_found = true;
                self.fail(Some(&format!(
                    "Wrong type of bracket.  Use {}",
                    bracket::CURLY.get_descr()
                )));
                return;
            }
        } else {
            self.bracket_found = true;
            tokenizer.next(&mut token);
        }
        // remove any whitespace before the first element
        if token
            .as_ref()
            .is_some_and(|token| token.is(TokenType::Whitespace))
        {
            tokenizer.next(&mut token);
        }
        token = self.parse_list(token, &mut tokenizer);
        if self.is_failed() {
            return;
        }
        if token
            .as_ref()
            .is_some_and(|token| token.is(TokenType::Whitespace))
        {
            tokenizer.next(&mut token);
        }
        // if the close symbol wasn't found, fail
        if !token
            .as_ref()
            .is_some_and(|token| token.equals_type_and_char(TokenType::Symbol, CLOSE_SYMBOL as u16))
        {
            let wrong_bracket = token.as_ref().is_some_and(|token| {
                Bracket::get_close_instance_character(token_char(token)).is_some()
            });
            if !self.optional_bracket || self.bracket_found {
                if wrong_bracket {
                    self.bracket_found = true;
                    self.fail(Some(&format!(
                        "Mismatched brackets.  Use {}",
                        bracket::CURLY.get_descr()
                    )));
                    return;
                }
                self.fail(Some(&format!(
                    "Unclosed bracket.  '{CLOSE_SYMBOL}' not found"
                )));
                return;
            }
            if wrong_bracket {
                self.bracket_found = true;
                self.fail(Some(&format!(
                    "Wrong closing bracket used.  Use {}",
                    bracket::CURLY.get_descr()
                )));
            }
        } else if !self.bracket_found {
            self.fail(Some(&format!(
                "Mismatched brackets.  Use {}",
                bracket::CURLY.get_descr()
            )));
        }
    }

    /// Java `isBracketFound()`.
    pub fn is_bracket_found(&self) -> bool {
        self.bracket_found
    }

    /// Java `isQuoteFound()`.
    pub fn is_quote_found(&self) -> bool {
        self.quote_found
    }

    /// Java private `createTokenizer(String)`.
    fn create_tokenizer(&mut self, value: &str) -> PrimativeTokenizer {
        let mut tokenizer = PrimativeTokenizer::get_string_instance(value, self.debug);
        match tokenizer.initialize() {
            Ok(()) => {}
            Err(e @ LogFileError::Lock(_)) => self.fail(Some(&e.get_message())),
            Err(e) => {
                eprintln!("{e}");
                self.fail(Some(&e.get_message()));
            }
        }
        tokenizer
    }

    /// Java private `parseList(Token, PrimativeTokenizer)`.
    fn parse_list(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
    ) -> Option<Box<Token>> {
        token.as_ref()?;
        let mut divider_found = true;
        // loop until the end of the array
        // can't just check dividerFound, because whitespace can act as a divider,
        // but isn't always the divider
        while divider_found
            && !self.is_failed()
            && let Some(current) = token.as_ref()
            && !current.is(TokenType::Eol)
            && !current.is(TokenType::Eof)
            && !current.equals_type_and_char(TokenType::Symbol, CLOSE_SYMBOL as u16)
        {
            // parse an element
            token = self.parse_element(token, tokenizer);
            // Find the divider.
            // Whitespace may be used as a divider or the divider may be preceded by
            // whitespace.
            divider_found = false;
            if token.as_ref().is_some_and(|token| {
                token.is(TokenType::Whitespace)
                    || token.equals_type_and_char(TokenType::Symbol, DIVIDER_SYMBOL as u16)
            }) {
                divider_found = true;
                tokenizer.next(&mut token);
            }
            if divider_found {
                // If whitespace was found, it may precede the divider.
                if token.as_ref().is_some_and(|token| {
                    token.equals_type_and_char(TokenType::Symbol, DIVIDER_SYMBOL as u16)
                }) {
                    tokenizer.next(&mut token);
                }
            }
            // Don't worry about whitespace after the divider. It should be handled
            // by the element.
        }
        token
    }

    /// Java private `parseElement(Token, PrimativeTokenizer)`.
    fn parse_element(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
    ) -> Option<Box<Token>> {
        if token
            .as_ref()
            .is_some_and(|token| token.is(TokenType::Whitespace))
        {
            tokenizer.next(&mut token);
        }
        if token.as_ref().is_some_and(|token| {
            token.equals_type_and_char(TokenType::Symbol, DIVIDER_SYMBOL as u16)
        }) {
            // Found an empty element.
            self.list.add(Some(Box::new(ParsedNumber::get_instance(
                self.r#type,
                self.etomo_number_type,
                self.is_debug(),
                self.default_value.as_ref(),
                self.descr.as_deref(),
            ))));
            return token;
        }
        // May have found an element.
        let element: Option<Box<dyn ParsedElement>>;
        if std::ptr::eq(self.r#type, &parsed_element_type::STRING) {
            let mut string =
                ParsedQuotedString::get_instance_boolean(self.is_debug(), self.descr.as_deref());
            token = string.parse(token, tokenizer, self.line_num);
            if !self.quote_found && string.is_quote_found() {
                self.quote_found = true;
            }
            element = Some(Box::new(string));
        } else if ParsedArray::is_array(token.as_deref()) {
            let mut array = ParsedArray::get_instance_full(
                self.r#type,
                self.etomo_number_type,
                self.is_debug(),
                self.default_value.as_ref(),
                self.descr.as_deref(),
            );
            token = array.parse(token, tokenizer, self.line_num);
            element = Some(Box::new(array));
        } else {
            // Array descriptors don't have their own open and close symbols, so they
            // look like numbers until to you get to the first divider (":"or "-").
            // Java dereferences the descriptor unchecked; it is null only for a
            // non-Matlab, non-string type, which none of the four factories
            // create.  Such an element would be parsed as a number here.
            match parsed_descriptor::get_instance(
                self.r#type,
                self.etomo_number_type,
                self.is_debug(),
                self.default_value.as_ref(),
                self.descr.as_deref(),
            ) {
                Some(mut descriptor) => {
                    token = descriptor.parse(token, tokenizer, self.line_num);
                    // create the correct type of element
                    if descriptor.is_empty() {
                        // There's nothing there, so its an empty element
                        self.list.add_empty_element();
                        return token;
                    } else if descriptor.was_divider_parsed() {
                        element = Some(Box::new(descriptor));
                    } else {
                        // If the divider was not found then it is not a descriptor.
                        element = descriptor.descriptor_base_mut().descriptor.remove(0);
                    }
                }
                None => {
                    let mut number = ParsedNumber::get_instance(
                        self.r#type,
                        self.etomo_number_type,
                        self.is_debug(),
                        self.default_value.as_ref(),
                        self.descr.as_deref(),
                    );
                    token = number.parse(token, tokenizer, self.line_num);
                    element = Some(Box::new(number));
                }
            }
        }
        self.list.add(element);
        token
    }

    /// Java private `isDebug()`.
    fn is_debug(&self) -> bool {
        self.debug
    }

    /// Java package-private `isFailed()`.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Java package-private `resetFailed()`.
    pub fn reset_failed(&mut self) {
        self.failed = false;
        self.error_message = None;
    }

    /// Java package-private `fail(String)`.  Set failed and store the first error
    /// message.
    pub fn fail(&mut self, message: Option<&str>) {
        self.failed = true;
        if let Some(message) = message
            && self.error_message.is_none()
        {
            self.error_message = Some(format!(
                "{}  {}{}",
                self.descr.as_deref().unwrap_or(""),
                message,
                if self.line_num > 0 {
                    format!("  Line# {}", self.line_num)
                } else {
                    String::new()
                }
            ));
        }
    }
}

impl Parsable for ParsedList {
    /// Java `clear()`.
    fn clear_parsable(&mut self) {
        self.list.clear();
        self.bracket_found = false;
        self.quote_found = false;
        self.line_num = -1;
    }

    /// Java `parse(String)`.
    fn parse_string(&mut self, parsable_string: Option<&str>) {
        self.parse_string_int(parsable_string, -1);
    }

    /// Java `validate()`.  Returns the first parse error or the first parse error of
    /// an element.  (Java assigns an element's error to the `errorMessage` field;
    /// that is the same field `fail` fills, so a later validate returns it.  The
    /// translation's `validate` takes `&self` and does not store it.)
    fn validate_parsable(&self) -> Option<String> {
        if self.error_message.is_some() {
            return self.error_message.clone();
        }
        let mut error_message = None;
        for i in 0..self.list.size() {
            if let Some(element) = self.list.get(i) {
                error_message = element.validate();
            }
            if error_message.is_some() {
                return error_message;
            }
        }
        if self.failed {
            return Some(format!(
                "{}: Unable to parse.{}",
                self.descr.as_deref().unwrap_or(""),
                if self.line_num > 0 {
                    format!("  Line# {}", self.line_num)
                } else {
                    String::new()
                }
            ));
        }
        None
    }

    /// Java `getParsableString()`.
    fn get_parsable_string_parsable(&self) -> Option<String> {
        let mut buffer = String::new();
        buffer.push(OPEN_SYMBOL);
        for i in 0..self.list.size() {
            if let Some(element) = self.list.get(i) {
                let string = element
                    .get_parsable_string()
                    .unwrap_or_else(|| "null".to_owned());
                if buffer.chars().count() > 1 {
                    buffer.push_str(&format!("{DIVIDER_SYMBOL} "));
                }
                if matches_whitespace(&string) {
                    buffer.push(parsed_array::OPEN_SYMBOL);
                    buffer.push(parsed_array::CLOSE_SYMBOL);
                } else {
                    buffer.push_str(&string);
                }
            }
        }
        buffer.push(CLOSE_SYMBOL);
        Some(buffer)
    }

    /// Java `parse(ReadOnlyAttribute)`.
    fn parse_attribute(&mut self, attribute: Option<&dyn ReadOnlyAttribute>) {
        let Some(attribute) = attribute else {
            return;
        };
        self.parse_string_int(attribute.get_value().as_deref(), attribute.get_line_num());
    }

    /// Java `isEmpty()`.
    fn is_empty_parsable(&self) -> bool {
        self.list.is_empty()
    }

    /// Java `size()`.
    fn size_parsable(&self) -> i32 {
        self.list.size()
    }
}

/// Java private static final nested class `Type` (`NUMERIC`, `STRING`), which
/// nothing in the source reads.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ListType {
    /// Java `NUMERIC`.
    Numeric,
    /// Java `STRING`.
    String,
}

/// Java `toString()`.
impl std::fmt::Display for ParsedList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[list:{}]", self.list)
    }
}
