//! `IMOD/Etomo/src/etomo/storage/autodoc/NameValuePair.java`.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::read_only_attribute::ReadOnlyAttribute;

use super::attribute::Attribute;
use super::autodoc;
use super::autodoc_tokenizer;
use super::read_only_statement::ReadOnlyStatement;
use super::section::Section;
use super::statement::{Statement, StatementBase, Type};
use super::writable_statement::WritableStatement;
use super::write_only_statement_list::WriteOnlyStatementList;
use crate::imod::etomo::ui::swing::token::{self, Token};
use crate::imod::etomo::util::utilities;

/// Java package-private final `NameValuePair extends Statement`.
pub struct NameValuePair {
    /// The fields Java inherits from `Statement`.
    statement: StatementBase,
    /// Java field `name`, a `Vector` of `Attribute`s: the name is made of attributes.
    name: Vec<*mut Attribute>,
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyStatementList,
    /// The parsed token list is borrowed from the tokenizer.  Values constructed by
    /// the writable API stay owned by this statement instead of being turned into a
    /// leaked raw allocation.
    value: Option<Value>,
    /// Java field `newDelimiter`, initialised to null.
    new_delimiter: *mut Token,
}

/// Storage for a name/value-pair value.  Parser token lists remain borrowed until
/// the tokenizer graph is made owned; tokens made by this statement are owned here.
enum Value {
    Parsed(*mut Token),
    Generated(Box<Token>),
}

/// Java `TYPE`: `Statement.Type.NAME_VALUE_PAIR`.
const TYPE: Type = Type::NameValuePair;

impl NameValuePair {
    /// Java `NameValuePair(WriteOnlyStatementList, Statement, int)`.
    ///
    /// # Safety
    /// `parent` must point to a live statement list and `previous_statement` must be
    /// null or point to a live statement.
    pub unsafe fn new(
        parent: *mut dyn WriteOnlyStatementList,
        previous_statement: *mut dyn Statement,
        line_num: i32,
    ) -> *mut NameValuePair {
        let this = Box::into_raw(Box::new(NameValuePair {
            statement: StatementBase::initial(),
            name: Vec::new(),
            parent,
            value: None,
            new_delimiter: std::ptr::null_mut(),
        }));
        unsafe { StatementBase::statement(this, previous_statement, line_num) };
        this
    }

    /// Java package-private `setDelimiterChange(Token)`.
    ///
    /// # Safety
    /// `new_delimiter` must be null or point to a live `Token` link list.
    pub unsafe fn set_delimiter_change(&mut self, new_delimiter: *mut Token) {
        self.new_delimiter = new_delimiter;
    }

    /// Java package-private `addAttribute(Attribute)`.
    ///
    /// each attribute is added as it is found
    ///
    /// # Safety
    /// `attribute` must point to a live `Attribute`.
    pub unsafe fn add_attribute(&mut self, attribute: *mut Attribute) {
        self.name.push(attribute);
    }

    /// Java package-private `addValue(Token)`.
    ///
    /// The value is added after all the attributes are added.  This name/value pair
    /// to the last attribute added.
    ///
    /// # Safety
    /// `value` must be null or point to a live `Token` link list, and the name must
    /// hold at least one live attribute - Java's `name.get(name.size() - 1)` throws
    /// `ArrayIndexOutOfBoundsException` on an empty name.
    pub unsafe fn add_value(&mut self, value: *mut Token) {
        self.value = if value.is_null() {
            None
        } else {
            Some(Value::Parsed(value))
        };
        let this: *mut NameValuePair = self;
        let last = self.name[self.name.len() - 1];
        unsafe { (*last).add_name_value_pair(this) };
    }

    /// Java package-private `setValue(Token)`.
    ///
    /// # Safety
    /// `value` is retained by the pair.
    pub fn set_value(&mut self, value: Box<Token>) {
        self.value = Some(Value::Generated(value));
    }

    /// Java `getValue()`.
    ///
    /// # Safety
    /// `value` must be null or point to a live `Token` link list.
    pub unsafe fn get_value(&self) -> Option<String> {
        match &self.value {
            None => None,
            Some(Value::Parsed(value)) => Some(unsafe { (**value).get_values() }),
            Some(Value::Generated(value)) => Some(unsafe { value.get_values() }),
        }
    }

    /// Java `getTokenValue()`.
    pub fn get_token_value(&self) -> *mut Token {
        match &self.value {
            None => std::ptr::null_mut(),
            Some(Value::Parsed(value)) => *value,
            Some(Value::Generated(value)) => value.as_ref() as *const Token as *mut Token,
        }
    }

    /// Java `toString()`, which returns `getString()`.
    ///
    /// # Safety
    /// Every attribute in the name must be live.
    pub unsafe fn to_string(&self) -> String {
        ReadOnlyStatement::get_string(self)
    }
}

impl Statement for NameValuePair {
    fn statement(&self) -> &StatementBase {
        &self.statement
    }

    fn statement_mut(&mut self) -> &mut StatementBase {
        &mut self.statement
    }

    /// Java `wrapValue(String, String, String, String, int, int)`.
    ///
    /// `value.getMultiLineValues()` cannot return null, so the source's
    /// `valueString == null` guard is dead; a null `noWrapPrefix` would throw a
    /// `NullPointerException` in `startsWith`, which no caller produces.
    unsafe fn wrap_value(
        &mut self,
        no_wrap_prefix: Option<&str>,
        wrap_prefix: Option<&str>,
        divider: Option<&str>,
        default_divider: Option<&str>,
        min_length: i32,
        wrap_length: i32,
    ) {
        let value_string = match &self.value {
            None => return,
            Some(Value::Parsed(value)) => unsafe { (**value).get_multi_line_values() },
            Some(Value::Generated(value)) => unsafe { value.get_multi_line_values() },
        };
        if value_string.starts_with(match no_wrap_prefix {
            None => panic!("java.lang.NullPointerException"),
            Some(no_wrap_prefix) => no_wrap_prefix,
        }) {
            return;
        }
        let matched_divider: Option<&str> = if value_string.starts_with(match wrap_prefix {
            None => panic!("java.lang.NullPointerException"),
            Some(wrap_prefix) => wrap_prefix,
        }) {
            divider
        } else {
            default_divider
        };
        if utilities::can_wrap(
            Some(&value_string),
            matched_divider,
            min_length,
            wrap_length,
            0,
        ) {
            let mut value = Box::new(Token::new());
            value.set_type_and_string(
                token::Type::Anything,
                &utilities::wrap(
                    Some(&value_string),
                    matched_divider,
                    min_length,
                    wrap_length,
                    0,
                )
                .unwrap(),
            );
            self.value = Some(Value::Generated(value));
        }
    }

    /// Java `write(LogFile.Handle, LogFile.WriterId)`.
    unsafe fn write(
        &self,
        file: &std::sync::Arc<log_file::Handle>,
        writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError> {
        for i in 0..self.name.len() {
            unsafe { (*self.name[i]).write(file, writer_id)? };
            if i < self.name.len() - 1 {
                file.write(Some(autodoc_tokenizer::SEPARATOR_CHAR), writer_id)?;
            }
        }
        file.write(
            Some(&format!(" {} ", unsafe {
                (*self.parent).get_current_delimiter()
            })),
            writer_id,
        )?;
        match &self.value {
            None => {}
            Some(Value::Parsed(value)) => unsafe { (**value).write(file, writer_id)? },
            Some(Value::Generated(value)) => unsafe { value.write(file, writer_id)? },
        }
        file.new_line(writer_id)?;
        if !self.new_delimiter.is_null() {
            unsafe { (*self.parent).set_current_delimiter(self.new_delimiter) };
        }
        Ok(())
    }

    /// Java `print(int)`.
    unsafe fn print(&self, level: i32) {
        autodoc::print_indent(level);
        for i in 0..self.name.len() {
            print!("{}", unsafe {
                match (*self.name[i]).get_value() {
                    None => "null".to_string(),
                    Some(value) => value,
                }
            });
            if i < self.name.len() - 1 {
                print!(".");
            }
        }
        print!(" {} ", unsafe { (*self.parent).get_current_delimiter() });
        match &self.value {
            None => println!(),
            Some(Value::Parsed(value)) => println!("{}", unsafe { (**value).get_values() }),
            Some(Value::Generated(value)) => println!("{}", unsafe { value.get_values() }),
        }
        if !self.new_delimiter.is_null() {
            unsafe { (*self.parent).set_current_delimiter(self.new_delimiter) };
        }
    }
}

impl WritableStatement for NameValuePair {
    /// Java `remove()`.
    ///
    /// Remove an occurrence from each attribute in the name.  Remove this instance
    /// from the last attribute.  Remove the instance from the Statement link list.
    /// Returns `Statement.previous`.
    unsafe fn remove(&mut self) -> *mut dyn Statement {
        let this: *mut NameValuePair = self;
        let mut attribute: *mut Attribute;
        for i in 0..self.name.len() {
            attribute = self.name[i];
            unsafe { (*attribute).remove() };
            if i == self.name.len() - 1 {
                unsafe { (*attribute).remove_name_value_pair(this) };
            }
        }
        unsafe { self.statement.remove() }
    }
}

impl ReadOnlyStatement for NameValuePair {
    /// Java `getType()`.
    fn get_type(&self) -> Type {
        TYPE
    }

    /// Java `sizeLeftSide()`.
    fn size_left_side(&self) -> i32 {
        self.name.len() as i32
    }

    /// Java `getLeftSide()`.
    fn get_left_side(&self) -> Option<String> {
        let mut buffer = String::new();
        let size = self.size_left_side();
        for i in 0..size {
            buffer.push_str(&format!(
                "{}{}",
                if i > 0 {
                    autodoc_tokenizer::SEPARATOR_CHAR
                } else {
                    ""
                },
                match self.get_left_side_at(i) {
                    None => "null".to_string(),
                    Some(left_side) => left_side,
                }
            ));
        }
        Some(buffer)
    }

    /// Java `getLeftSide(int)`.
    fn get_left_side_at(&self, index: i32) -> Option<String> {
        if index < 0 || index >= self.name.len() as i32 {
            return None;
        }
        Some(unsafe { (*self.name[index as usize]).get_name() })
    }

    /// Java `getRightSide()`.
    fn get_right_side(&self) -> Option<String> {
        match &self.value {
            None => None,
            Some(Value::Parsed(value)) => Some(unsafe { (**value).get_values() }),
            Some(Value::Generated(value)) => Some(unsafe { value.get_values() }),
        }
    }

    /// Java `getSubsection()`.
    fn get_subsection(&self) -> *mut Section {
        std::ptr::null_mut()
    }

    /// Java `getString()`.
    ///
    /// Get something equivalent to the original statement.  Not guarenteed to be
    /// exactly the same.
    fn get_string(&self) -> String {
        let mut buffer = String::new();
        if !self.name.is_empty() {
            buffer.push_str(&unsafe { (*self.name[0]).get_name() });
        }
        for i in 1..self.name.len() {
            buffer.push_str(autodoc_tokenizer::SEPARATOR_CHAR);
            buffer.push_str(&unsafe { (*self.name[i]).get_name() });
        }
        buffer.push_str(&format!(" {} ", autodoc_tokenizer::DEFAULT_DELIMITER));
        match &self.value {
            None => {}
            Some(Value::Parsed(value)) => buffer.push_str(&unsafe { (**value).get_values() }),
            Some(Value::Generated(value)) => buffer.push_str(&unsafe { value.get_values() }),
        }
        buffer
    }

    /// Java inherits `Statement.getLineNum()`.
    fn get_line_num(&self) -> i32 {
        self.statement.get_line_num()
    }
}
