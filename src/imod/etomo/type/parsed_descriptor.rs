//! `IMOD/Etomo/src/etomo/type/ParsedDescriptor.java`.
//!
//! The package-private abstract parent of the array descriptors (`j:k`, `j:i:k`).
//! `ParsedArrayDescriptor` is its only subclass in the source.
//!
//! **Shape.**  The fields the class declares sit in [`ParsedDescriptorBase`], which the
//! subclass embeds; the class's members are the default methods of the trait
//! [`ParsedDescriptor`] (`abstract getDividerSymbol`/`isDebug` are its required
//! methods).  The members that override `ParsedElement` methods carry a
//! `_descriptor` suffix and the subclass's `ParsedElement` impl calls them, which is
//! how the Java dispatch reaches them; calls the Java makes virtually (`parse`,
//! `getString`, `getParsedNumberExpandedArray`) go through `self`, so they reach the
//! subclass's overrides as in Java.
//!
//! The source wraps every `tokenizer.next` call in `catch (IOException e)`; the
//! translated `PrimativeTokenizer.next` reads from memory and cannot fail, so those
//! handlers have nothing to catch and are not reproduced.

use super::const_etomo_number::Type;
use super::etomo_number::EtomoNumber;
use super::parsed_array;
use super::parsed_array_descriptor::{self, ParsedArrayDescriptor};
use super::parsed_element::{ParsedElement, ParsedElementBase};
use super::parsed_element_list::ParsedElementList;
use super::parsed_element_type::ParsedElementType;
use super::parsed_list;
use super::parsed_number::ParsedNumber;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// The fields Java's abstract `ParsedDescriptor` declares (with its superclass's).
#[derive(Clone)]
pub struct ParsedDescriptorBase {
    /// Java superclass `ParsedElement` fields.
    pub base: ParsedElementBase,
    /// Java private final `etomoNumberType`.
    etomo_number_type: Option<Type>,
    /// Java package-private final `descriptor`, for use by children classes.
    pub descriptor: ParsedElementList,
    /// Java private final `type`.
    r#type: &'static ParsedElementType,
    /// Java private `dividerParsed`, initially false.
    divider_parsed: bool,
    /// Java private `defaultValue`, initially null.
    default_value: Option<EtomoNumber>,
}

impl ParsedDescriptorBase {
    /// The field part of Java `ParsedDescriptor(ParsedElementType, EtomoNumber.Type,
    /// boolean, EtomoNumber, String)`: `super(descr)` and the assignments.  The
    /// constructor's closing `setDebug(debug)` is virtual; the subclass constructor
    /// makes it.
    pub fn new(
        r#type: &'static ParsedElementType,
        etomo_number_type: Option<Type>,
        debug: bool,
        default_value: Option<&EtomoNumber>,
        descr: Option<&str>,
    ) -> ParsedDescriptorBase {
        ParsedDescriptorBase {
            base: ParsedElementBase::new(descr),
            etomo_number_type,
            descriptor: ParsedElementList::new(
                r#type,
                etomo_number_type,
                debug,
                default_value,
                descr,
            ),
            r#type,
            divider_parsed: false,
            default_value: default_value.cloned(),
        }
    }
}

/// Java package-private static `getDividerSymbol(ParsedElementType)`.
pub fn get_divider_symbol(_type: &ParsedElementType) -> char {
    parsed_array_descriptor::DIVIDER_SYMBOL
}

/// Java package-private static `getInstance(ParsedElementType, EtomoNumber.Type,
/// boolean, EtomoNumber, String)`.  If the type is any kind of matlab number, return
/// an instance of ParsedArrayDescriptor.  These are the only types where an array
/// descriptor of some kind is valid.  If the type is anything else, return null.
/// (The declared return type is `ParsedDescriptor`; its only subclass is returned.)
pub fn get_instance(
    r#type: &'static ParsedElementType,
    etomo_number_type: Option<Type>,
    debug: bool,
    default_value: Option<&EtomoNumber>,
    descr: Option<&str>,
) -> Option<ParsedArrayDescriptor> {
    if debug {
        println!("ParsedDescriptor.getInstance");
    }
    if r#type.is_matlab() {
        return Some(ParsedArrayDescriptor::new(
            etomo_number_type,
            debug,
            default_value,
            descr,
        ));
    }
    None
}

/// Java abstract class `ParsedDescriptor extends ParsedElement`.
pub trait ParsedDescriptor: ParsedElement {
    /// The fields Java's `ParsedDescriptor` declares (not a source member).
    fn descriptor_base(&self) -> &ParsedDescriptorBase;

    /// Mutable access to the fields Java's `ParsedDescriptor` declares (not a source
    /// member).
    fn descriptor_base_mut(&mut self) -> &mut ParsedDescriptorBase;

    /// Java abstract package-private `getDividerSymbol()`.
    fn get_divider_symbol_void(&self) -> char;

    /// Java abstract package-private `isDebug()`.
    fn is_debug(&self) -> bool;

    /// Java `toString()` (final): `"[descriptor:" + descriptor + "]"`.
    fn to_string_descriptor(&self) -> String {
        format!("[descriptor:{}]", self.descriptor_base().descriptor)
    }

    /// Java final package-private `removeElement(int)`.
    fn remove_element_descriptor(&mut self, index: i32) {
        self.descriptor_base_mut().descriptor.remove(index);
    }

    /// Java final `clear()`.
    fn clear_descriptor(&mut self) {
        self.descriptor_base_mut().descriptor.clear();
        self.reset_line_num();
    }

    /// Java final package-private `wasDividerParsed()`.  Returns true if the
    /// tokenizer passed to parse() contained a divider.
    fn was_divider_parsed(&self) -> bool {
        self.descriptor_base().divider_parsed
    }

    /// Java final package-private `setDividerParsed()`.
    fn set_divider_parsed(&mut self) {
        self.descriptor_base_mut().divider_parsed = true;
    }

    /// Java package-private `parse(Token, PrimativeTokenizer, int)`.  Parse the array
    /// descriptor.  Also figures out if this actually is an array.  Whitespace are not
    /// allowed around the divider in an array descriptor.  (Overridden by the only
    /// subclass; kept as the class member.)
    fn parse_descriptor(
        &mut self,
        mut token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
        line_num: i32,
    ) -> Option<Box<Token>> {
        if self.is_debug() {
            println!(
                "ParsedDescriptor.parse:token={}",
                token
                    .as_ref()
                    .map_or("null".to_owned(), |token| token.to_string())
            );
        }
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        token.as_ref()?;
        if token.as_ref().unwrap().is(TokenType::Whitespace) {
            tokenizer.next(&mut token);
        }
        let mut divider_found = true;
        // loop until the end of the array descriptor.
        while divider_found
            && !self.is_failed()
            && let Some(current) = token.as_ref()
            && !current.is(TokenType::Eol)
            && !current.is(TokenType::Eof)
            && !current.equals_type_and_char(TokenType::Symbol, parsed_list::CLOSE_SYMBOL as u16)
            && !current.equals_type_and_char(TokenType::Symbol, parsed_array::CLOSE_SYMBOL as u16)
        {
            // parse an element
            token = self.parse_element_descriptor(token, tokenizer);
            if self.is_debug() {
                println!(
                    "ParsedDescriptor.parse:descriptor={}",
                    self.descriptor_base().descriptor
                );
            }
            // Find the divider.
            divider_found = false;
            if token.as_ref().is_some_and(|token| {
                token.equals_type_and_char(TokenType::Symbol, self.get_divider_symbol_void() as u16)
            }) {
                // Until the first divider is found this may not be a descriptor.
                self.set_divider_parsed();
                divider_found = true;
                tokenizer.next(&mut token);
            }
            // Don't worry about whitespace after the divider. It should be handled
            // by the element.
        }
        token
    }

    /// Java final package-private `setDefault(EtomoNumber)`.  Sets defaultValue and
    /// calls setDefaultValue() for each element in the array.
    fn set_default_etomo_number_descriptor(&mut self, input: Option<&EtomoNumber>) {
        let base = self.descriptor_base_mut();
        base.default_value = input.cloned();
        base.descriptor.set_default(input);
        for i in 0..base.descriptor.size() {
            let default_value = base.default_value.clone();
            if let Some(element) = base.descriptor.get_mut(i) {
                element.set_default_etomo_number(default_value.as_ref());
            }
        }
    }

    /// Java final `setDefault(int)`.
    fn set_default_int_descriptor(&mut self, input: i32) {
        let base = self.descriptor_base_mut();
        if base.default_value.is_none() {
            base.default_value = Some(EtomoNumber::new_with_type(base.etomo_number_type));
        }
        base.default_value.as_mut().unwrap().set_int(input);
        let default_value = base.default_value.clone();
        base.descriptor.set_default(default_value.as_ref());
        for i in 0..base.descriptor.size() {
            if let Some(element) = base.descriptor.get_mut(i) {
                element.set_default_etomo_number(default_value.as_ref());
            }
        }
    }

    /// Java final package-private `parseElement(Token, PrimativeTokenizer)`.  Parse a
    /// number.
    fn parse_element_descriptor(
        &mut self,
        token: Option<Box<Token>>,
        tokenizer: &mut PrimativeTokenizer,
    ) -> Option<Box<Token>> {
        let debug = self.is_debug();
        let line_num = self.get_line_num();
        let base = self.descriptor_base();
        let mut element = ParsedNumber::get_instance(
            base.r#type,
            base.etomo_number_type,
            debug,
            base.default_value.as_ref(),
            base.base.descr.as_deref(),
        );
        let default_value = base.default_value.clone();
        element.set_debug(debug);
        element.set_default_etomo_number(default_value.as_ref());
        let token = element.parse(token, tokenizer, line_num);
        if debug {
            println!("ParsedDescriptor.parse:element={element}");
        }
        self.descriptor_base_mut()
            .descriptor
            .add(Some(Box::new(element)));
        token
    }

    /// Java final `set(ParsedElement)`.  Clear the descriptor and add the elements in
    /// the descriptor to the elements in the input.  If an element is a collection,
    /// set the elements in the collection element, because a descriptor cannot
    /// contain elements that are collections.  (Java sets the input's debug and
    /// default before reading it; the translation sets them on the copy it reads.)
    fn set(&mut self, input: Option<&dyn ParsedElement>) {
        self.clear();
        let Some(input) = input else {
            return;
        };
        let mut input = input.clone_element();
        input.set_debug(self.is_debug());
        let default_value = self.descriptor_base().default_value.clone();
        input.set_default_etomo_number(default_value.as_ref());
        self.append(Some(input.as_ref()));
    }

    /// Java private `append(ParsedElement)`.  Add the elements in the descriptor to
    /// the elements in the input.  If an element is a collection, append the elements
    /// in the collection element.
    fn append(&mut self, input: Option<&dyn ParsedElement>) {
        let Some(input) = input else {
            return;
        };
        let mut input_index = 0;
        while input_index < input.size() {
            let element = input.get_element(input_index);
            input_index += 1;
            // Fixed in translation: Java dereferences a null element of a sparse
            // collection (NullPointerException); a missing element is skipped.
            let Some(element) = element else {
                continue;
            };
            if element.is_collection() {
                self.append(Some(element));
            } else {
                self.descriptor_base_mut()
                    .descriptor
                    .add(Some(element.clone_element()));
            }
        }
    }

    /// Java package-private `setRawString(int, String)`.
    fn set_raw_string_int_string_descriptor(&mut self, index: i32, string: Option<&str>) {
        if index < 0 {
            return;
        }
        let debug = self.is_debug();
        let line_num = self.get_line_num();
        let base = self.descriptor_base();
        let mut element = ParsedNumber::get_instance(
            base.r#type,
            base.etomo_number_type,
            debug,
            base.default_value.as_ref(),
            base.base.descr.as_deref(),
        );
        element.set_raw_string_string_int(string, line_num);
        self.descriptor_base_mut()
            .descriptor
            .set(index, Box::new(element));
    }

    /// Java package-private `setRawString(int, double)`.
    fn set_raw_string_int_double_descriptor(&mut self, index: i32, number: f64) {
        let debug = self.is_debug();
        let base = self.descriptor_base();
        let mut element = ParsedNumber::get_instance(
            base.r#type,
            base.etomo_number_type,
            debug,
            base.default_value.as_ref(),
            base.base.descr.as_deref(),
        );
        element.set_raw_string_double(number);
        self.descriptor_base_mut()
            .descriptor
            .set(index, Box::new(element));
    }

    /// Java package-private `size()`.
    fn size_descriptor(&self) -> i32 {
        self.descriptor_base().descriptor.size()
    }

    /// Java `validate()`.  Returns the first parse error or the first parse error of
    /// an element.
    fn validate_descriptor(&self) -> Option<String> {
        let mut error_message = self.get_error_message();
        if error_message.is_some() {
            return error_message;
        }
        let descriptor = &self.descriptor_base().descriptor;
        for i in 0..descriptor.size() {
            if let Some(element) = descriptor.get(i) {
                error_message = element.validate();
            }
            if error_message.is_some() {
                return error_message;
            }
        }
        // super.validate()
        self.descriptor_base().base.validate()
    }

    /// Java final package-private `getEtomoNumberType()`.
    fn get_etomo_number_type(&self) -> Option<Type> {
        self.descriptor_base().etomo_number_type
    }

    /// Java final package-private `getType()`.
    fn get_type(&self) -> &'static ParsedElementType {
        self.descriptor_base().r#type
    }

    /// Java final package-private `getDefault()`.
    fn get_default(&self) -> Option<&EtomoNumber> {
        self.descriptor_base().default_value.as_ref()
    }

    /// Java final package-private `setRawString(String, int)`.  Input is a
    /// collection, since it is not indexed, so it is a semi-raw string - it has :'s or
    /// -'s.
    fn set_raw_string_string_int_descriptor(&mut self, input: Option<&str>, line_num: i32) {
        self.clear();
        self.reset_failed();
        self.set_line_num(line_num);
        let Some(input) = input else {
            return;
        };
        let mut tokenizer = self.create_tokenizer(Some(input));
        let mut token = None;
        tokenizer.next(&mut token);
        self.parse(token, &mut tokenizer, line_num);
    }

    /// Java package-private `getParsedNumberExpandedArray(ParsedElementList)`.  Append
    /// an array of non-empty ParsedNumbers described by this.descriptor.  (Overridden
    /// by the only subclass; kept as the class member.)
    fn get_parsed_number_expanded_array_descriptor(
        &self,
        parsed_number_expanded_array: Option<ParsedElementList>,
    ) -> ParsedElementList {
        let base = self.descriptor_base();
        let debug = self.is_debug();
        let descr = base.base.descr.as_deref();
        let mut parsed_number_expanded_array = match parsed_number_expanded_array {
            Some(list) => list,
            None => ParsedElementList::new(
                base.r#type,
                base.etomo_number_type,
                debug,
                base.default_value.as_ref(),
                descr,
            ),
        };
        if base.descriptor.size() == 0 {
            return parsed_number_expanded_array;
        }
        // exclude empty descriptor numbers
        let mut list = ParsedElementList::new(
            base.r#type,
            base.etomo_number_type,
            debug,
            base.default_value.as_ref(),
            descr,
        );
        for i in 0..base.descriptor.size() {
            if let Some(element) = base.descriptor.get(i)
                && !element.is_empty()
            {
                list.add(Some(element.clone_element()));
            }
        }
        if list.size() == 0 {
            return parsed_number_expanded_array;
        }
        let as_number = |element: Option<&dyn ParsedElement>| -> ParsedNumber {
            element
                .and_then(|element| element.as_any().downcast_ref::<ParsedNumber>())
                .expect("a descriptor holds ParsedNumbers")
                .clone()
        };
        // the first number in the descriptor is the first number in the array
        parsed_number_expanded_array.add(list.get(0).map(|element| element.clone_element()));
        // if there is only one number, then we are done
        if list.size() == 1 {
            return parsed_number_expanded_array;
        }
        // two or three numbers means that it is a descriptor, so expand the descriptor
        // into the list of numbers
        let mut increment = EtomoNumber::new_with_type(base.etomo_number_type);
        let last_number: ParsedNumber;
        if list.size() == 2 {
            let first = as_number(list.get(0));
            let second = as_number(list.get(1));
            // If there are two numbers and they are the same, then the the array
            // expands into the one number
            if first.equals_parsed_number(&second) {
                return parsed_number_expanded_array;
            } else {
                increment.set_int(self.get_increment(&first, &second));
            }
            last_number = second;
        } else {
            increment.set_number(list.get(1).and_then(|element| element.get_raw_number()));
            last_number = as_number(list.get(2));
        }
        // if the increment is 0, return the first and last number
        // not sure if this is right.
        if increment.equals_int(0) {
            parsed_number_expanded_array.add(Some(Box::new(last_number)));
            return parsed_number_expanded_array;
        }
        // increment the number and save the result until you get to the last number
        // the increment can be positive or negative
        let mut cur_number = as_number(list.get(0));
        let mut done = false;
        while !done {
            let prev_number = cur_number;
            cur_number = ParsedNumber::get_instance(
                base.r#type,
                base.etomo_number_type,
                debug,
                base.default_value.as_ref(),
                descr,
            );
            cur_number.set_raw_string_number(prev_number.get_raw_number());
            cur_number.plus(&increment);
            if (increment.is_positive() && cur_number.le(&last_number))
                || (increment.is_negative() && cur_number.ge_parsed_number(&last_number))
            {
                parsed_number_expanded_array.add(Some(cur_number.clone_element()));
            } else {
                done = true;
            }
        }
        // Matlab parser ignores the last number, so don't add it. With the
        // iterator descriptor the increment is always 1 and only integers are
        // allowed, so that the last number will automatically be added.
        parsed_number_expanded_array
    }

    /// Java package-private `getString(boolean)`.  Return the elements of the
    /// collection separated by dividers.  (Overridden by the only subclass; kept as
    /// the class member.)
    fn get_string(&self, parsable: bool) -> String {
        let mut buffer = String::new();
        let descriptor = &self.descriptor_base().descriptor;
        for i in 0..descriptor.size() {
            if let Some(element) = descriptor.get(i) {
                let string = if parsable {
                    element.get_parsable_string()
                } else {
                    element.get_raw_string_void()
                }
                .unwrap_or_else(|| "null".to_owned());
                if !buffer.is_empty() && !matches_whitespace(&string) {
                    buffer.push(self.get_divider_symbol_void());
                }
                buffer.push_str(&string);
            }
        }
        buffer
    }

    /// Java final package-private `getIncrement(ParsedNumber, ParsedNumber)`.
    fn get_increment(&self, first: &ParsedNumber, last: &ParsedNumber) -> i32 {
        if first.gt(last) {
            return -1;
        }
        1
    }

    /// Java final `getRawString()`.
    fn get_raw_string_void_descriptor(&self) -> Option<String> {
        Some(self.get_string(false))
    }

    /// Java final `getRawString(int)`.
    fn get_raw_string_int_descriptor(&self, index: i32) -> Option<String> {
        if let Some(element) = self.get_element(index) {
            return element.get_raw_string_void();
        }
        Some(String::new())
    }

    /// Java `getElement(int)`.
    fn get_element_descriptor(&self, index: i32) -> Option<&dyn ParsedElement> {
        self.descriptor_base().descriptor.get(index)
    }

    /// Java final package-private `getParsableString()`.
    fn get_parsable_string_descriptor(&self) -> Option<String> {
        Some(self.get_string(true))
    }

    /// Java final package-private `ge(int)`.  Returns true only when all numbers in
    /// the array are greater then or equal to the number parameter.  Expands any array
    /// descriptors into the arrays they represent.
    fn ge_descriptor(&self, number: i32) -> bool {
        let expanded_array = self.get_parsed_number_expanded_array(None);
        let mut greater_or_equal = true;
        for i in 0..expanded_array.size() {
            if let Some(element) = expanded_array.get(i)
                && !element.ge(number)
            {
                greater_or_equal = false;
                break;
            }
        }
        greater_or_equal
    }

    /// Java final `equals(int)`.  Returns true only when all numbers in the array are
    /// equal to the number parameter.
    fn equals_descriptor(&self, number: i32) -> bool {
        let expanded_array = self.get_parsed_number_expanded_array(None);
        let mut equal = true;
        for i in 0..expanded_array.size() {
            if let Some(element) = expanded_array.get(i)
                && !element.equals(number)
            {
                equal = false;
                break;
            }
        }
        equal
    }
}

/// Java `s.matches("\\s*")`.
pub(crate) fn matches_whitespace(s: &str) -> bool {
    s.chars()
        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
}
