//! `IMOD/Etomo/src/etomo/type/ParsedArray.java`.
//!
//! A bracketed array of numbers and array descriptors (`[1, 2, 4:6]`), or a plain
//! comma/space separated list of numbers.  `ParsedArray extends ParsedElement`: the
//! superclass fields are the embedded `ParsedElementBase`.
//!
//! The source wraps every `tokenizer.next` call in `catch (IOException e)`; the
//! translated `PrimativeTokenizer.next` reads from memory and cannot fail, so those
//! handlers have nothing to catch and are not reproduced.

use std::collections::BTreeMap;

use super::const_etomo_number::Type;
use super::etomo_number::EtomoNumber;
use super::parsed_array_descriptor::ParsedArrayDescriptor;
use super::parsed_descriptor::{self, ParsedDescriptor, matches_whitespace};
use super::parsed_element::{ParsedElement, ParsedElementBase};
use super::parsed_element_list::ParsedElementList;
use super::parsed_element_type::{self, ParsedElementType};
use super::parsed_list;
use super::parsed_number::ParsedNumber;
use super::string_property::StringProperty;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// Java package-private static final `OPEN_SYMBOL = new Character('[')`.
pub const OPEN_SYMBOL: char = '[';
/// Java package-private static final `CLOSE_SYMBOL = new Character(']')`.
pub const CLOSE_SYMBOL: char = ']';
/// Java package-private static final `DIVIDER_SYMBOL = ParsedList.DIVIDER_SYMBOL`.
pub const DIVIDER_SYMBOL: char = parsed_list::DIVIDER_SYMBOL;

/// Java `public final class ParsedArray extends ParsedElement`.
#[derive(Clone)]
pub struct ParsedArray {
    /// Java superclass `ParsedElement` fields.
    base: ParsedElementBase,
    /// Java private final `array`.
    array: ParsedElementList,
    /// Java private final `type`.
    r#type: &'static ParsedElementType,
    /// Java private final `etomoNumberType`.
    etomo_number_type: Option<Type>,
    /// Java private final `key`.
    key: Option<String>,
    /// Java private `defaultValue`, initially null.
    default_value: Option<EtomoNumber>,
    /// Java private `debug`, initially false.
    debug: bool,
    /// Java private `backwardCompatibleNullKey`, initially false.
    backward_compatible_null_key: bool,
}

impl ParsedArray {
    /// Java private `ParsedArray(ParsedElementType, EtomoNumber.Type, String,
    /// boolean, EtomoNumber, String)`.
    fn new(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        key: Option<&str>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedArray {
        let mut instance = ParsedArray {
            base: ParsedElementBase::new(descr),
            // The list keeps the element type it was given (not the array instance).
            array: ParsedElementList::new(r#type, etomo_number_type, debug, default_value, descr),
            r#type: r#type.to_array_instance(),
            etomo_number_type,
            key: key.map(str::to_owned),
            default_value: default_value.cloned(),
            debug,
            backward_compatible_null_key: false,
        };
        instance.set_debug(debug);
        instance
    }

    /// Java `getInstance(ParsedElementType, String)`.
    pub fn get_instance_type_descr(
        r#type: &'static ParsedElementType,
        descr: Option<&str>,
    ) -> ParsedArray {
        ParsedArray::new(r#type, None, None, false, None, descr)
    }

    /// Java `getMatlabInstance(String)`.
    pub fn get_matlab_instance(descr: Option<&str>) -> ParsedArray {
        ParsedArray::new(
            &parsed_element_type::MATLAB_ARRAY,
            None,
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
    ) -> ParsedArray {
        ParsedArray::new(
            &parsed_element_type::MATLAB_ARRAY,
            etomo_number_type,
            None,
            false,
            None,
            descr,
        )
    }

    /// Java `getInstance(EtomoNumber.Type, String, String)`.
    pub fn get_instance(
        etomo_number_type: Option<Type>,
        key: Option<&str>,
        descr: Option<&str>,
    ) -> ParsedArray {
        ParsedArray::new(
            &parsed_element_type::NON_MATLAB_ARRAY,
            etomo_number_type,
            key,
            false,
            None,
            descr,
        )
    }

    /// Java package-private static `getInstance(ParsedElementType, EtomoNumber.Type,
    /// boolean, EtomoNumber, String)`.
    pub fn get_instance_full(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedArray {
        ParsedArray::new(r#type, etomo_number_type, None, debug, default_value, descr)
    }

    /// Java `setBackwardCompatibleNullKey()`.  A null key was used because the
    /// description was set instead of the key.  Create a backwards compatible key set
    /// to "null".
    pub fn set_backward_compatible_null_key(&mut self) {
        self.backward_compatible_null_key = true;
    }

    /// Java `getPaddedStringExpandedArray()`.  Get the array stored in this.array,
    /// excluding empty ParsedNumbers and expanding array descriptors.  Pad numbers
    /// with zeros.
    pub fn get_padded_string_expanded_array(&self) -> Vec<String> {
        let expanded_array = self.get_parsed_number_expanded_array(None);
        if expanded_array.size() == 0 {
            return Vec::new();
        }
        let mut max_digits = 0;
        let mut buffer_array: Vec<String> = Vec::with_capacity(expanded_array.size() as usize);
        for i in 0..expanded_array.size() {
            let mut buffer = String::new();
            if let Some(number) = expanded_array.get(i) {
                buffer.push_str(&number.get_parsable_string().unwrap_or_default());
                max_digits = max_digits.max(buffer.encode_utf16().count());
            }
            buffer_array.push(buffer);
        }
        let mut return_array = Vec::with_capacity(buffer_array.len());
        for mut buffer in buffer_array {
            let padding = max_digits - buffer.encode_utf16().count();
            for _ in 0..padding {
                buffer.insert(0, '0');
            }
            return_array.push(buffer);
        }
        return_array
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

    /// Java `isEmpty(int)`.
    pub fn is_empty_int(&self, index: i32) -> bool {
        if let Some(element) = self.array.get(index) {
            return element.is_empty();
        }
        true
    }

    /// Java `setRawStringStart(String)`.
    pub fn set_raw_string_start(&mut self, string: Option<&str>) {
        if let Some(descriptor) = self.get_add_first_array_descriptor(string) {
            descriptor.set_raw_string_start(string);
        }
    }

    /// Java `setRawStringEnd(String)`.
    pub fn set_raw_string_end(&mut self, string: Option<&str>) {
        if let Some(descriptor) = self.get_add_first_array_descriptor(string) {
            descriptor.set_raw_string_end(string);
        }
    }

    /// Java `setRawStringIncrement(String)`.
    pub fn set_raw_string_increment(&mut self, string: Option<&str>) {
        if let Some(descriptor) = self.get_add_first_array_descriptor(string) {
            descriptor.set_raw_string_increment(string);
        }
    }

    /// Java private `getAddFirstArrayDescriptor(String)`.  Gets the first array
    /// descriptor in the array.  If there is no array descriptor, it adds one, but
    /// only if addIfNumber is a number.
    fn get_add_first_array_descriptor(
        &mut self,
        add_if_number: Option<&str>,
    ) -> Option<&mut ParsedArrayDescriptor> {
        let index = match self.get_first_array_descriptor_index() {
            -1 => {
                let mut number = ParsedNumber::get_instance(
                    self.r#type,
                    self.etomo_number_type,
                    self.debug,
                    self.default_value.as_ref(),
                    self.base.descr.as_deref(),
                );
                // If the string doesn't have a number in it, don't bother to create
                // the descriptor. Descriptors can't use "NaN", so there is no point
                // to creating an empty one.
                number.set_raw_string_string_int(add_if_number, self.get_line_num());
                if number.is_empty() {
                    return None;
                }
                let descriptor = ParsedArrayDescriptor::new(
                    self.etomo_number_type,
                    self.debug,
                    self.default_value.as_ref(),
                    self.base.descr.as_deref(),
                );
                let index = self.array.size();
                self.array.add(Some(Box::new(descriptor)));
                index
            }
            index => index,
        };
        self.array.get_mut(index).and_then(|element| {
            (element.as_mut() as &mut dyn std::any::Any).downcast_mut::<ParsedArrayDescriptor>()
        })
    }

    /// Java `getRawStringStart()`.
    pub fn get_raw_string_start(&self) -> Option<String> {
        if let Some(descriptor) = self.get_first_array_descriptor() {
            return descriptor.get_raw_string_start();
        }
        Some(String::new())
    }

    /// Java `getRawStringIncrement()`.
    pub fn get_raw_string_increment(&self) -> Option<String> {
        if let Some(descriptor) = self.get_first_array_descriptor() {
            return descriptor.get_raw_string_increment();
        }
        Some(String::new())
    }

    /// Java `getRawStringEnd()`.
    pub fn get_raw_string_end(&self) -> Option<String> {
        if let Some(descriptor) = self.get_first_array_descriptor() {
            return descriptor.get_raw_string_end();
        }
        Some(String::new())
    }

    /// Java `setRawStrings(String)`.  Parse one or more elements in an array.
    pub fn set_raw_strings(&mut self, input: Option<&str>) {
        let mut tokenizer = self.create_tokenizer(input);
        let mut token = None;
        tokenizer.next(&mut token);
        // raw strings shouldn't have brackets so start with parseArray, not parse.
        self.parse_array(token, &mut tokenizer);
    }

    /// Java private `getElement(int, int)`.
    #[allow(dead_code)]
    fn get_element_int_int(
        &self,
        array_index: i32,
        descriptor_index: i32,
    ) -> Option<&dyn ParsedElement> {
        if let Some(element) = self.array.get(array_index) {
            return element.get_element(descriptor_index);
        }
        None
    }

    /// Java `getRawStringsExceptFirstArrayDescriptor()`.
    pub fn get_raw_strings_except_first_array_descriptor(&self) -> String {
        let first_array_descriptor_index = self.get_first_array_descriptor_index();
        self.get_string(false, 0, first_array_descriptor_index)
    }

    /// Java private `getFirstArrayDescriptor()`.
    fn get_first_array_descriptor(&self) -> Option<&ParsedArrayDescriptor> {
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get(i)
                && element.is_descriptor()
            {
                return element.as_any().downcast_ref::<ParsedArrayDescriptor>();
            }
        }
        None
    }

    /// Java private `getFirstArrayDescriptorIndex()`.
    fn get_first_array_descriptor_index(&self) -> i32 {
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get(i)
                && element.is_descriptor()
            {
                return i;
            }
        }
        -1
    }

    /// Java `getRawString(int, int)`.
    pub fn get_raw_string_int_int(
        &self,
        array_index: i32,
        descriptor_index: i32,
    ) -> Option<String> {
        if let Some(element) = self.array.get(array_index) {
            return element.get_raw_string_int(descriptor_index);
        }
        Some(String::new())
    }

    /// Java `set(ParsedElement)`.  Clear the array member variable, and add each
    /// element of the ParsedElement to the array.  (Java sets the input's debug and
    /// default before reading it; the translation sets them on the copy it reads.)
    pub fn set(&mut self, input: Option<&dyn ParsedElement>) {
        self.clear();
        let Some(input) = input else {
            return;
        };
        let mut input = input.clone_element();
        input.set_debug(self.debug);
        input.set_default_etomo_number(self.default_value.as_ref());
        for i in 0..input.size() {
            self.array
                .add(input.get_element(i).map(|element| element.clone_element()));
        }
    }

    /// Java package-private static `isArray(Token)`.  This is a array only if starts
    /// with "[" (strip whitespace before calling).
    pub fn is_array(token: Option<&Token>) -> bool {
        let Some(token) = token else {
            return false;
        };
        if token.equals_type_and_char(TokenType::Symbol, OPEN_SYMBOL as u16) {
            return true;
        }
        false
    }

    /// Java package-private `setElement(int, ParsedElement)`.
    pub fn set_element(&mut self, index: i32, mut element: Box<dyn ParsedElement>) {
        element.set_debug(self.debug);
        element.set_default_etomo_number(self.default_value.as_ref());
        self.array.set(index, element);
    }

    /// Java `addElement(ParsedElement)`.
    pub fn add_element(&mut self, mut element: Box<dyn ParsedElement>) {
        element.set_debug(self.debug);
        element.set_default_etomo_number(self.default_value.as_ref());
        self.array.add(Some(element));
    }

    /// Java package-private `store(Properties, String)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        let mut property = StringProperty::new_with_key(self.key.as_deref());
        property.set(self.get_raw_string_void().as_deref());
        property.store_with_prepend(Some(props), prepend);
    }

    /// Java package-private `load(Properties, String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: Option<&str>) {
        let mut property = StringProperty::new_with_key(self.key.as_deref());
        if self.backward_compatible_null_key {
            property.set_backward_compatible_key(Some("null"));
        }
        let mut props = props.clone();
        property.load_with_prepend(Some(&mut props), prepend);
        let line_num = self.get_line_num();
        self.set_raw_string_string_int(Some(&property.to_string()), line_num);
    }

    /// Java private `getString(boolean, int, int)`.  Returns a raw string or a
    /// parsable string.  Will return the whole string when startIndex is equal to -1
    /// or 0.  Otherwise returns a partial string starting at startIndex.
    fn get_string(&self, parsable: bool, mut start_index: i32, exclusion_index: i32) -> String {
        if start_index == -1 {
            start_index = 0;
        }
        let mut buffer = String::new();
        for i in start_index..self.array.size() {
            if exclusion_index != i {
                let element = self.array.get(i);
                if self.debug {
                    println!(
                        "i={i},element={}",
                        element.map_or("null".to_owned(), |element| element.to_string())
                    );
                }
                if let Some(element) = element {
                    let string = if parsable {
                        element.get_parsable_string()
                    } else {
                        element.get_raw_string_void()
                    }
                    .unwrap_or_else(|| "null".to_owned());
                    if !buffer.is_empty() {
                        buffer.push_str(&format!("{DIVIDER_SYMBOL} "));
                    }
                    if parsable && matches_whitespace(&string) {
                        buffer.push_str("NaN");
                    } else {
                        buffer.push_str(&string);
                    }
                }
            }
        }
        buffer
    }

    /// Java private `parseArray(Token, PrimativeTokenizer)`.
    fn parse_array(
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
            if self.debug {
                println!("ParsedArray.parseArray:while");
            }
            // parse an element
            token = self.parse_element_token(token, tokenizer, -1);
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

    /// Java private `parseElement(Token, PrimativeTokenizer, int)`.  Adds or sets
    /// either a ParsedArrayDescriptor or a ParsedNumber.  Adds the element when index
    /// is -1.  (Java's two-argument `parseElement(Token, PrimativeTokenizer)` is this
    /// with index -1.)
    fn parse_element_token(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
        index: i32,
    ) -> Option<Box<Token>> {
        // parse element
        // Array descriptors don't have their own open and close symbols, so they
        // look like numbers until to you get to the first divider (":"or "-").
        if self.debug {
            println!(
                "ParsedArray.parseElement:token={},type={:?},index:{index}",
                token
                    .as_ref()
                    .map_or("null".to_owned(), |token| token.to_string()),
                self.r#type
            );
        }
        let element: Option<Box<dyn ParsedElement>>;
        // First assume that there might be an array descriptor.
        let descriptor = parsed_descriptor::get_instance(
            self.r#type,
            self.etomo_number_type,
            self.debug,
            self.default_value.as_ref(),
            self.base.descr.as_deref(),
        );
        if let Some(mut descriptor) = descriptor {
            descriptor.set_debug(self.debug);
            token = descriptor.parse(token, tokenizer, self.get_line_num());
            // create the correct type of element
            if descriptor.is_empty() {
                // There's nothing there, so its an empty element
                if index == -1 {
                    self.array.add_empty_element();
                } else {
                    self.array.set_empty_element(index);
                }
                return token;
            } else if descriptor.was_divider_parsed() {
                element = Some(Box::new(descriptor));
            } else {
                // If the divider was not found then it is not a descriptor.
                element = descriptor.descriptor_base_mut().descriptor.remove(0);
            }
        } else {
            // ParsedDescriptor would not return an instance so the type is not a type
            // that can have an array descriptor or iterator.
            let mut number = ParsedNumber::get_instance(
                self.r#type,
                self.etomo_number_type,
                self.debug,
                self.default_value.as_ref(),
                self.base.descr.as_deref(),
            );
            token = number.parse(token, tokenizer, self.get_line_num());
            element = Some(Box::new(number));
        }
        if index == -1 {
            self.array.add(element);
        } else if let Some(element) = element {
            self.array.set(index, element);
        }
        token
    }
}

impl ParsedElement for ParsedArray {
    fn parsed_element_base(&self) -> &ParsedElementBase {
        &self.base
    }

    fn parsed_element_base_mut(&mut self) -> &mut ParsedElementBase {
        &mut self.base
    }

    /// Java `getRawString()`.  Raw strings go to the screen.
    fn get_raw_string_void(&self) -> Option<String> {
        Some(self.get_string(false, -1, -1))
    }

    /// Java `getRawString(int)`.
    fn get_raw_string_int(&self, index: i32) -> Option<String> {
        if let Some(element) = self.array.get(index) {
            return element.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java `setDefault(int)`.
    fn set_default_int(&mut self, input: i32) {
        if self.default_value.is_none() {
            self.default_value = Some(EtomoNumber::new_with_type(self.etomo_number_type));
        }
        self.default_value.as_mut().unwrap().set_int(input);
        self.array.set_default(self.default_value.as_ref());
        let default_value = self.default_value.clone();
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get_mut(i) {
                element.set_default_etomo_number(default_value.as_ref());
            }
        }
    }

    /// Java `setDebug(boolean)`.
    fn set_debug(&mut self, input: bool) {
        self.debug = input;
        self.array.set_debug(input);
        for i in 0..self.size() {
            if let Some(element) = self.array.get_mut(i) {
                element.set_debug(input);
            }
        }
    }

    /// Java `equals(int)`.  Returns true only when all numbers in the array are equal
    /// to the number parameter.
    fn equals(&self, number: i32) -> bool {
        let mut equal = true;
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get(i)
                && !element.equals(number)
            {
                equal = false;
                break;
            }
        }
        equal
    }

    /// Java package-private `setRawString(String, int)`.  Raw strings come from the
    /// screen.  Input may be a collection, since it is not indexed, so treat it as a
    /// semi-raw string (it may have commas).
    fn set_raw_string_string_int(&mut self, input: Option<&str>, line_num: i32) {
        if self.debug {
            println!("ParsedArray.setRawString:input={}", input.unwrap_or("null"));
        }
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        let Some(input) = input else {
            return;
        };
        let mut tokenizer = self.create_tokenizer(Some(input));
        let mut token = None;
        tokenizer.next(&mut token);
        // raw strings shouldn't have brackets
        // place input into array starting from the beginning of the list
        self.parse_array(token, &mut tokenizer);
    }

    /// Java `getElement(int)`.
    fn get_element(&self, index: i32) -> Option<&dyn ParsedElement> {
        self.array.get(index)
    }

    /// Java package-private `setRawString(int, double)`.
    fn set_raw_string_int_double(&mut self, index: i32, number: f64) {
        let mut element = ParsedNumber::get_instance(
            self.r#type,
            self.etomo_number_type,
            self.debug,
            self.default_value.as_ref(),
            self.base.descr.as_deref(),
        );
        element.set_raw_string_double(number);
        self.array.set(index, Box::new(element));
    }

    /// Java `setRawString(int, String)`.
    fn set_raw_string_int_string(&mut self, index: i32, string: Option<&str>) {
        let mut tokenizer = self.create_tokenizer(string);
        let mut token = None;
        tokenizer.next(&mut token);
        self.parse_element_token(token, &mut tokenizer, index);
    }

    /// Java package-private `parse(Token, PrimativeTokenizer, int)`.  Parse the entire
    /// array.  Returns the token that is current when the array is parsed.
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
        let current = token.as_ref()?;
        if !current.equals_type_and_char(TokenType::Symbol, OPEN_SYMBOL as u16) {
            self.fail(Some(&format!("Missing delimiter: '{OPEN_SYMBOL}'")));
            return token;
        }
        tokenizer.next(&mut token);
        // Remove any whitespace before the first element.
        if token
            .as_ref()
            .is_some_and(|token| token.is(TokenType::Whitespace))
        {
            tokenizer.next(&mut token);
        }
        token = self.parse_array(token, tokenizer);
        if self.is_failed() {
            return token;
        }
        if token
            .as_ref()
            .is_some_and(|token| token.is(TokenType::Whitespace))
        {
            tokenizer.next(&mut token);
        }
        if !token
            .as_ref()
            .is_some_and(|token| token.equals_type_and_char(TokenType::Symbol, CLOSE_SYMBOL as u16))
        {
            self.fail(Some(&format!("Missing delimiter: '{CLOSE_SYMBOL}'")));
            return token;
        }
        tokenizer.next(&mut token);
        token
    }

    /// Java `size()`.
    fn size(&self) -> i32 {
        self.array.size()
    }

    /// Java `getParsableString()`.  Parsable strings are saved to the .prm file.
    /// Returns [] if the list is empty.  Returns a list surrounded by brackets.
    fn get_parsable_string(&self) -> Option<String> {
        let mut buffer = OPEN_SYMBOL.to_string();
        buffer.push_str(&self.get_string(true, -1, -1));
        buffer.push(CLOSE_SYMBOL);
        Some(buffer)
    }

    /// Java package-private `isCollection()`.
    fn is_collection(&self) -> bool {
        true
    }

    /// Java package-private `isDescriptor()`.
    fn is_descriptor(&self) -> bool {
        false
    }

    /// Java package-private `setDefault(EtomoNumber)`.  Sets defaultValue and calls
    /// setDefaultValue() for each element in the array.
    fn set_default_etomo_number(&mut self, input: Option<&EtomoNumber>) {
        self.default_value = input.cloned();
        self.array.set_default(input);
        let default_value = self.default_value.clone();
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get_mut(i) {
                element.set_default_etomo_number(default_value.as_ref());
            }
        }
    }

    /// Java package-private `removeElement(int)`.
    fn remove_element(&mut self, index: i32) {
        self.array.remove(index);
    }

    /// Java `ge(int)`.  Returns true only when all numbers in the array are greater
    /// then or equal to the number parameter.
    fn ge(&self, number: i32) -> bool {
        let mut greater_or_equal = true;
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get(i)
                && !element.ge(number)
            {
                greater_or_equal = false;
                break;
            }
        }
        greater_or_equal
    }

    /// Java `clear()`.
    fn clear(&mut self) {
        self.array.clear();
        self.reset_line_num();
    }

    /// Java `getParsedNumberExpandedArray(ParsedElementList)`.  Create a list of
    /// non-null ParsedNumber based on this.array.  Expand the array descriptors to
    /// create the entire array.
    fn get_parsed_number_expanded_array(
        &self,
        mut parsed_number_expanded_array: Option<ParsedElementList>,
    ) -> ParsedElementList {
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get(i) {
                parsed_number_expanded_array =
                    Some(element.get_parsed_number_expanded_array(parsed_number_expanded_array));
            }
        }
        match parsed_number_expanded_array {
            Some(list) => list,
            None => ParsedElementList::new(
                self.r#type,
                self.etomo_number_type,
                self.debug,
                self.default_value.as_ref(),
                self.base.descr.as_deref(),
            ),
        }
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
        let mut error_message = self.get_error_message();
        if error_message.is_some() {
            return error_message;
        }
        for i in 0..self.array.size() {
            if let Some(element) = self.array.get(i) {
                error_message = element.validate();
            }
            if error_message.is_some() {
                return error_message;
            }
        }
        self.base.validate()
    }
}

/// Java `toString()`.
impl std::fmt::Display for ParsedArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[array:{}]", self.array)
    }
}
