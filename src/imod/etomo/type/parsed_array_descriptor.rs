//! `IMOD/Etomo/src/etomo/type/ParsedArrayDescriptor.java`.
//!
//! A Matlab array descriptor: `start:end` or `start:increment:end`.
//! `ParsedArrayDescriptor extends ParsedDescriptor`; the superclass fields are the
//! embedded `ParsedDescriptorBase` and its members are the `ParsedDescriptor` trait's
//! default methods (see `parsed_descriptor.rs`).

use super::const_etomo_number::{Number, Type};
use super::etomo_number::EtomoNumber;
use super::parsed_array;
use super::parsed_descriptor::{ParsedDescriptor, ParsedDescriptorBase};
use super::parsed_element::{ParsedElement, ParsedElementBase};
use super::parsed_element_list::ParsedElementList;
use super::parsed_element_type;
use super::parsed_list;
use super::parsed_number::ParsedNumber;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// Java package-private static final `DIVIDER_SYMBOL = new Character(':')`.
pub const DIVIDER_SYMBOL: char = ':';
/// Java private static final `START_INDEX`.
const START_INDEX: i32 = 0;
/// Java private static final `INCREMENT_INDEX`.
const INCREMENT_INDEX: i32 = 1;
/// Java private static final `END_INDEX`.
const END_INDEX: i32 = 2;
/// Java private static final `NO_INCREMENT_SIZE`.
const NO_INCREMENT_SIZE: i32 = 2;
/// Java private static final `MAX_SIZE`.
const MAX_SIZE: i32 = 3;

/// Java `public final class ParsedArrayDescriptor extends ParsedDescriptor`.
#[derive(Clone)]
pub struct ParsedArrayDescriptor {
    /// Java superclass `ParsedDescriptor` fields.
    base: ParsedDescriptorBase,
    /// Java private `debug`, initially false.
    debug: bool,
}

impl ParsedArrayDescriptor {
    /// Java package-private `ParsedArrayDescriptor(EtomoNumber.Type, boolean,
    /// EtomoNumber, String)`.  (The superclass constructor's virtual `setDebug(debug)`
    /// runs before this class's field initializer resets `debug` to false; the
    /// constructor body then calls `setDebug(debug)` again, which is the state kept.)
    pub fn new(
        etomo_number_type: Option<Type>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedArrayDescriptor {
        let mut instance = ParsedArrayDescriptor {
            base: ParsedDescriptorBase::new(
                &parsed_element_type::MATLAB_ARRAY_DESCRIPTOR,
                etomo_number_type,
                debug,
                default_value,
                descr,
            ),
            debug: false,
        };
        instance.set_debug(debug);
        instance
    }

    /// Java `getInstance(EtomoNumber.Type, String)`.
    pub fn get_instance(
        etomo_number_type: Option<Type>,
        descr: Option<&str>,
    ) -> ParsedArrayDescriptor {
        ParsedArrayDescriptor::new(etomo_number_type, false, None, descr)
    }

    /// Java `setRawStringEnd(String)`.
    pub fn set_raw_string_end(&mut self, input: Option<&str>) {
        self.set_raw_string_int_string(END_INDEX, input);
    }

    /// Java `setRawStringStart(String)`.
    pub fn set_raw_string_start(&mut self, input: Option<&str>) {
        self.set_raw_string_int_string(START_INDEX, input);
    }

    /// Java `setRawStringIncrement(String)`.
    pub fn set_raw_string_increment(&mut self, input: Option<&str>) {
        self.set_raw_string_int_string(INCREMENT_INDEX, input);
    }

    /// Java `getRawStringEnd()`.
    pub fn get_raw_string_end(&self) -> Option<String> {
        if let Some(element) = self.base.descriptor.get(END_INDEX) {
            return element.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java package-private `getRawStringStart()`.
    pub fn get_raw_string_start(&self) -> Option<String> {
        if let Some(element) = self.base.descriptor.get(START_INDEX) {
            return element.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java `getRawStringIncrement()`.
    pub fn get_raw_string_increment(&self) -> Option<String> {
        if let Some(element) = self.base.descriptor.get(INCREMENT_INDEX) {
            return element.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java `setMinArraySize(int)`.  No effect.  The array size is always three.
    pub fn set_min_array_size(&mut self, _input: i32) {}

    /// Java `getStart()`.
    pub fn get_start(&self) -> Option<&dyn ParsedElement> {
        let size = self.base.descriptor.size();
        if size > 0 {
            return self.base.descriptor.get(START_INDEX);
        }
        None
    }

    /// Java `getEnd()`.
    pub fn get_end(&self) -> Option<&dyn ParsedElement> {
        let size = self.base.descriptor.size();
        if size == NO_INCREMENT_SIZE {
            return self.base.descriptor.get(INCREMENT_INDEX);
        }
        if size == MAX_SIZE {
            return self.base.descriptor.get(END_INDEX);
        }
        None
    }
}

impl ParsedDescriptor for ParsedArrayDescriptor {
    fn descriptor_base(&self) -> &ParsedDescriptorBase {
        &self.base
    }

    fn descriptor_base_mut(&mut self) -> &mut ParsedDescriptorBase {
        &mut self.base
    }

    /// Java package-private `getDividerSymbol()`.
    fn get_divider_symbol_void(&self) -> char {
        DIVIDER_SYMBOL
    }

    /// Java package-private `isDebug()`.
    fn is_debug(&self) -> bool {
        self.debug
    }

    /// Java package-private `getString(boolean)`.  Return a two or three element
    /// array descriptor.  Returns two elements if increment is not set.
    fn get_string(&self, parsable: bool) -> String {
        if self.base.descriptor.is_empty() {
            return String::new();
        }
        let start = self.base.descriptor.get(START_INDEX);
        let increment = self.base.descriptor.get(INCREMENT_INDEX);
        let end = self.base.descriptor.get(END_INDEX);
        let mut start_string = String::new();
        let mut increment_string = String::new();
        let mut end_string = String::new();
        let text = |element: &dyn ParsedElement| {
            if parsable {
                element.get_parsable_string()
            } else {
                element.get_raw_string_void()
            }
            .unwrap_or_else(|| "null".to_owned())
        };
        if let Some(start) = start {
            start_string = text(start);
        }
        if let Some(increment) = increment {
            increment_string = text(increment);
        }
        if let Some(end) = end {
            end_string = text(end);
        }
        let mut buffer = String::new();
        buffer.push_str(&format!("{start_string}{DIVIDER_SYMBOL}"));
        // Never use an empty or zero increment element.
        if increment.is_some_and(|increment| !increment.is_empty()) {
            buffer.push_str(&format!("{increment_string}{DIVIDER_SYMBOL}"));
        }
        buffer.push_str(&end_string);
        buffer
    }
}

impl ParsedElement for ParsedArrayDescriptor {
    fn parsed_element_base(&self) -> &ParsedElementBase {
        &self.base.base
    }

    fn parsed_element_base_mut(&mut self) -> &mut ParsedElementBase {
        &mut self.base.base
    }

    /// Java final `getRawString()` (ParsedDescriptor).
    fn get_raw_string_void(&self) -> Option<String> {
        self.get_raw_string_void_descriptor()
    }

    /// Java final `getRawString(int)` (ParsedDescriptor).
    fn get_raw_string_int(&self, index: i32) -> Option<String> {
        self.get_raw_string_int_descriptor(index)
    }

    /// Java final `setDefault(int)` (ParsedDescriptor).
    fn set_default_int(&mut self, input: i32) {
        self.set_default_int_descriptor(input);
    }

    /// Java `setDebug(boolean)`.
    fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.base.descriptor.set_debug(input);
        for i in 0..self.size() {
            if let Some(element) = self.base.descriptor.get_mut(i) {
                element.set_debug(input);
            }
        }
    }

    /// Java final `equals(int)` (ParsedDescriptor).
    fn equals(&self, number: i32) -> bool {
        self.equals_descriptor(number)
    }

    /// Java final `setRawString(String, int)` (ParsedDescriptor).
    fn set_raw_string_string_int(&mut self, number: Option<&str>, line_num: i32) {
        self.set_raw_string_string_int_descriptor(number, line_num);
    }

    /// Java `getElement(int)` (ParsedDescriptor).
    fn get_element(&self, index: i32) -> Option<&dyn ParsedElement> {
        self.get_element_descriptor(index)
    }

    /// Java package-private `setRawString(int, double)`.  Set number at index if
    /// index between 0 and 2.
    fn set_raw_string_int_double(&mut self, index: i32, number: f64) {
        if index < 0 {
            return;
        }
        if index > END_INDEX {
            // `new IllegalStateException(...).printStackTrace()`: the string
            // concatenation appends the digits ("Unable to add element 31.").
            eprintln!(
                "java.lang.IllegalStateException: Unable to add element {index}1.  No more then {END_INDEX}1 elements are allowed in an array descriptor."
            );
            return;
        }
        self.set_raw_string_int_double_descriptor(index, number);
    }

    /// Java package-private `setRawString(int, String)`.  Set string at index if index
    /// between 0 and 2.
    fn set_raw_string_int_string(&mut self, index: i32, string: Option<&str>) {
        if index < 0 {
            return;
        }
        if index > END_INDEX {
            self.fail(Some(&format!(
                "Unable to add element {index}1.  No more then {END_INDEX}1 elements are allowed in an array descriptor."
            )));
            return;
        }
        self.set_raw_string_int_string_descriptor(index, string);
    }

    /// Java package-private `parse(Token, PrimativeTokenizer, int)`.  Parse the array
    /// descriptor.  Also figures out if this actually is an array descriptor.  If the
    /// descriptor only contains two elements move the second one; treat it as the end
    /// element rather then the increment element.
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
        let mut divider_found = true;
        // loop until the end of the array descriptor. Whitespace is not allowed in
        // an array descriptor.
        while divider_found
            && !self.is_failed()
            && let Some(current) = token.as_ref()
            && !current.is(TokenType::Eol)
            && !current.is(TokenType::Eof)
            && !current.is(TokenType::Whitespace)
            && !current.equals_type_and_char(TokenType::Symbol, parsed_list::CLOSE_SYMBOL as u16)
            && !current.equals_type_and_char(TokenType::Symbol, parsed_array::CLOSE_SYMBOL as u16)
        {
            // parse an element
            token = self.parse_element_descriptor(token, tokenizer);
            // Find the divider.
            divider_found = false;
            if token.as_ref().is_some_and(|token| {
                token.equals_type_and_char(TokenType::Symbol, DIVIDER_SYMBOL as u16)
            }) {
                // This confirms that this is a descriptor.
                self.set_divider_parsed();
                divider_found = true;
                tokenizer.next(&mut token);
            }
        }
        // If there are 2 elements, then assume that the increment is one and put the
        // second element in the end slot.
        if self.base.descriptor.size() == END_INDEX {
            let increment = self
                .get_element(INCREMENT_INDEX)
                .map(|increment| increment.get_raw_string_void());
            if let Some(increment_raw_string) = increment {
                self.set_raw_string_int_string(END_INDEX, increment_raw_string.as_deref());
                if let Some(increment) = self.base.descriptor.get_mut(INCREMENT_INDEX) {
                    increment.set_raw_string_string_int(Some("1"), line_num);
                }
            }
        }
        token
    }

    /// Java package-private `size()`.
    fn size(&self) -> i32 {
        self.base.descriptor.size()
    }

    /// Java final package-private `getParsableString()` (ParsedDescriptor).
    fn get_parsable_string(&self) -> Option<String> {
        self.get_parsable_string_descriptor()
    }

    /// Java final package-private `isCollection()` (ParsedDescriptor).
    fn is_collection(&self) -> bool {
        true
    }

    /// Java final package-private `isDescriptor()` (ParsedDescriptor).
    fn is_descriptor(&self) -> bool {
        true
    }

    /// Java final package-private `setDefault(EtomoNumber)` (ParsedDescriptor).
    fn set_default_etomo_number(&mut self, input: Option<&EtomoNumber>) {
        self.set_default_etomo_number_descriptor(input);
    }

    /// Java final package-private `removeElement(int)` (ParsedDescriptor).
    fn remove_element(&mut self, index: i32) {
        self.remove_element_descriptor(index);
    }

    /// Java final package-private `ge(int)` (ParsedDescriptor).
    fn ge(&self, number: i32) -> bool {
        self.ge_descriptor(number)
    }

    /// Java final `clear()` (ParsedDescriptor).
    fn clear(&mut self) {
        self.clear_descriptor();
    }

    /// Java package-private `getParsedNumberExpandedArray(ParsedElementList)`.  Append
    /// an array of non-empty ParsedNumbers described by this.descriptor.  Construct
    /// parsedNumberExpandedArray if it is null.
    fn get_parsed_number_expanded_array(
        &self,
        parsed_number_expanded_array: Option<ParsedElementList>,
    ) -> ParsedElementList {
        let mut parsed_number_expanded_array = match parsed_number_expanded_array {
            Some(list) => list,
            None => ParsedElementList::new(
                self.get_type(),
                self.get_etomo_number_type(),
                self.debug,
                self.get_default(),
                self.base.base.descr.as_deref(),
            ),
        };
        let as_number = |element: Option<&dyn ParsedElement>| -> Option<ParsedNumber> {
            element
                .and_then(|element| element.as_any().downcast_ref::<ParsedNumber>())
                .cloned()
        };
        let start = as_number(self.base.descriptor.get(START_INDEX));
        let end = as_number(self.base.descriptor.get(END_INDEX));
        let (Some(start), Some(end)) = (start, end) else {
            // Invalid descriptor.
            return parsed_number_expanded_array;
        };
        if start.is_empty() || end.is_empty() {
            // Invalid descriptor.
            return parsed_number_expanded_array;
        }
        let mut increment = EtomoNumber::new_with_type(self.get_etomo_number_type());
        if let Some(element) = self.base.descriptor.get(INCREMENT_INDEX) {
            increment.set_number(element.get_raw_number());
        }
        // A missing increment is the same as an increment equals to 1.
        if increment.is_null() {
            increment.set_int(1);
        }
        if increment.is_negative() && start.lt(&end) {
            // Empty descriptor
            return parsed_number_expanded_array;
        }
        if !increment.is_negative() && end.lt(&start) {
            // Empty descriptor
            return parsed_number_expanded_array;
        }
        // Add the first element to the array.
        parsed_number_expanded_array.add(Some(start.clone_element()));
        if start.equals_parsed_number(&end) {
            // Descriptor with one element.
            return parsed_number_expanded_array;
        }
        // Add elements to the expanded array.
        let mut current = EtomoNumber::new_with_type(self.get_etomo_number_type());
        current.set_number(start.get_raw_number());
        current.add_const_etomo_number(Some(&increment));
        let mut last = EtomoNumber::new_with_type(self.get_etomo_number_type());
        last.set_number(end.get_raw_number());
        let increasing = !increment.is_negative();
        while (increasing && current.lt_const_etomo_number(Some(&last)))
            || (!increasing && current.gt_const_etomo_number(Some(&last)))
        {
            let mut parsed_current = ParsedNumber::get_instance(
                self.get_type(),
                self.get_etomo_number_type(),
                self.debug,
                self.get_default(),
                self.base.base.descr.as_deref(),
            );
            let value: Number = current.get_number();
            parsed_current.set_raw_string_number(Some(value));
            parsed_number_expanded_array.add(Some(Box::new(parsed_current)));
            // `current = new EtomoNumber(current)`.
            current = EtomoNumber::new_from_instance(Some(&current));
            current.add_const_etomo_number(Some(&increment));
        }
        // Add the last element to the array.
        parsed_number_expanded_array.add(Some(end.clone_element()));
        parsed_number_expanded_array
    }

    fn clone_element(&self) -> Box<dyn ParsedElement> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Java `validate()`.  Returns the first parse error or the first parse error of
    /// an element.
    fn validate(&self) -> Option<String> {
        let error_message = self.get_error_message();
        if error_message.is_some() {
            return error_message;
        }
        let size = self.base.descriptor.size();
        if !(NO_INCREMENT_SIZE..=MAX_SIZE).contains(&size) {
            return Some(format!(
                "Array descriptors can contain either {NO_INCREMENT_SIZE} or {MAX_SIZE} elements."
            ));
        }
        let start = self.base.descriptor.get(START_INDEX);
        let end = if size > NO_INCREMENT_SIZE {
            self.base.descriptor.get(END_INDEX)
        } else {
            self.base.descriptor.get(INCREMENT_INDEX)
        };
        if start.is_none_or(|start| start.is_empty()) || end.is_none_or(|end| end.is_empty()) {
            return Some("Array descriptors must contain at least a start and an end.".to_owned());
        }
        self.validate_descriptor()
    }
}

/// Java final `toString()` (ParsedDescriptor).
impl std::fmt::Display for ParsedArrayDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_descriptor())
    }
}
