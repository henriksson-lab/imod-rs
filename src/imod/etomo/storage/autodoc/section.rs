//! `IMOD/Etomo/src/etomo/storage/autodoc/Section.java`.
//!
//! `@notthreadsafe`
//!
//! **Ownership.**  A parent section list owns each `Section` in a `Box`; maps and
//! subsection statements borrow its stable address.  Each section owns its attribute
//! list directly.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::attribute::Attribute;
use super::attribute_list::AttributeList;
use super::autodoc;
use super::autodoc_tokenizer;
use super::comment::Comment;
use super::empty_line::EmptyLine;
use super::name_value_pair::NameValuePair;
use super::read_only_section::ReadOnlySection;
use super::read_only_section_list::ReadOnlySectionList;
use super::read_only_statement_list::ReadOnlyStatementList;
use super::section_location::SectionLocation;
use super::statement::{EmptyStatement, Statement};
use super::statement_location::StatementLocation;
use super::subsection::Subsection;
use super::write_only_attribute_list::WriteOnlyAttributeList;
use super::write_only_statement_list::WriteOnlyStatementList;
use crate::imod::etomo::ui::swing::token::{self, Token};
use std::collections::HashMap;

/// Java package-private final `Section extends WriteOnlyStatementList implements
/// ReadOnlySection`.
pub struct Section {
    /// Java field `statementList`.
    statement_list: Vec<Box<dyn Statement>>,
    /// Java field `key`.
    key: Option<String>,
    /// Java field `type`.
    r#type: *mut Token,
    /// Java field `name`.
    name: *mut Token,
    /// Java field `attributeList`.
    attribute_list: Option<Box<AttributeList>>,
    /// Java field `sectionList`.
    section_list: Vec<Box<Section>>,
    /// Java field `subSectionMap`.
    sub_section_map: HashMap<String, std::ptr::NonNull<Section>>,
    /// Java field `subsection`, initialised to false.
    subsection: bool,
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyStatementList,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

/// Java `public static getKey(Token type, Token name)`.
///
pub fn get_key_of_tokens(r#type: Option<&Token>, name: Option<&Token>) -> Option<String> {
    match (r#type, name) {
        (None, None) => None,
        (None, Some(name)) => Some(unsafe { name.get_key() }),
        (Some(r#type), None) => Some(unsafe { r#type.get_key() }),
        (Some(r#type), Some(name)) => {
            Some(unsafe { r#type.get_key() } + &unsafe { name.get_key() })
        }
    }
}

/// Java `public static getKey(String type, String name)`.
pub fn get_key_of_strings(r#type: Option<&str>, name: Option<&str>) -> Option<String> {
    if r#type.is_none() && name.is_none() {
        return None;
    }
    if r#type.is_none() {
        return Some(token::convert_to_key(name.unwrap()));
    }
    if name.is_none() {
        return Some(token::convert_to_key(r#type.unwrap()));
    }
    Some(token::convert_to_key(r#type.unwrap()) + &token::convert_to_key(name.unwrap()))
}

impl Section {
    /// Java package-private `Section(Token, Token, WriteOnlyStatementList)`.  Java's
    /// `new AttributeList(this)` needs the object, which Rust cannot produce before the
    /// allocation, so the attribute list is assigned immediately after it.
    ///
    /// # Safety
    /// `type` and `name` must be null or point to live `Token` link lists and `parent`
    /// must point to a live statement list.
    pub unsafe fn new(
        r#type: *mut Token,
        name: *mut Token,
        parent: *mut dyn WriteOnlyStatementList,
    ) -> *mut Section {
        let this = Box::into_raw(Box::new(Section {
            statement_list: Vec::new(),
            key: get_key_of_tokens(unsafe { r#type.as_ref() }, unsafe { name.as_ref() }),
            r#type,
            name,
            attribute_list: None,
            section_list: Vec::new(),
            sub_section_map: HashMap::new(),
            subsection: false,
            parent,
            debug: false,
        }));
        unsafe { (*this).attribute_list = Some(Box::new(AttributeList::new(this))) };
        this
    }

    /// Java `toString()`.
    ///
    /// # Safety
    /// The type token, name token and attribute list must be live.
    pub unsafe fn to_string(&self) -> String {
        format!(
            "etomo.storage.autodoc.Section[key={},type={},name={},\nattributeList={}]",
            match &self.key {
                None => "null".to_string(),
                Some(key) => key.clone(),
            },
            if self.r#type.is_null() {
                "null".to_string()
            } else {
                unsafe { (*self.r#type).to_string() }
            },
            if self.name.is_null() {
                "null".to_string()
            } else {
                unsafe { (*self.name).to_string() }
            },
            unsafe { self.attribute_list.as_deref().unwrap().to_string() }
        )
    }

    /// Java package-private `equalsType(String)`.
    ///
    /// # Safety
    /// The type token must be live when `type` is non-null.
    pub unsafe fn equals_type(&self, r#type: Option<&str>) -> bool {
        if r#type.is_none() {
            // For a section location with a null type, all sections are returned.
            return true;
        }
        unsafe { (*self.r#type).get_key() == token::convert_to_key(r#type.unwrap()) }
    }

    /// Java package-private `getTypeToken()`.
    pub fn get_type_token(&self) -> &Token {
        if self.r#type.is_null() {
            panic!("java.lang.IllegalStateException: type is required");
        }
        unsafe { &*self.r#type }
    }

    /// Java package-private `getKey()`.
    pub fn get_key(&self) -> Option<String> {
        self.key.clone()
    }

    /// Java package-private `getNameToken()`.
    pub fn get_name_token(&self) -> &Token {
        if self.name.is_null() {
            panic!("java.lang.IllegalStateException: name is required");
        }
        unsafe { &*self.name }
    }

    /// Java `hashCode()`, which is `key.hashCode()` - see `Attribute.hashCode`.
    pub fn hash_code(&self) -> i32 {
        let mut h: i32 = 0;
        for unit in match &self.key {
            None => panic!("java.lang.NullPointerException"),
            Some(key) => key,
        }
        .encode_utf16()
        {
            h = h.wrapping_mul(31).wrapping_add(unit as i32);
        }
        h
    }

    /// Java package-private `write(LogFile.Handle, LogFile.WriterId)`.
    ///
    /// # Safety
    /// Every token, statement and subsection this section points at must be live.
    pub unsafe fn write(
        &self,
        file: &std::sync::Arc<log_file::Handle>,
        writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError> {
        // write section header
        file.write_character(Some(autodoc_tokenizer::OPEN_CHAR), writer_id)?;
        if self.subsection {
            file.write_character(Some(autodoc_tokenizer::OPEN_CHAR), writer_id)?;
        }
        unsafe { (*self.r#type).write(file, writer_id)? };
        file.write(
            Some(&format!(" {} ", unsafe {
                (*self.parent).get_current_delimiter()
            })),
            writer_id,
        )?;
        unsafe { (*self.name).write(file, writer_id)? };
        file.write_character(Some(autodoc_tokenizer::CLOSE_CHAR), writer_id)?;
        if self.subsection {
            file.write_character(Some(autodoc_tokenizer::CLOSE_CHAR), writer_id)?;
        }
        file.new_line(writer_id)?;
        for statement in self.statement_list.iter() {
            unsafe { statement.write(file, writer_id)? };
        }
        // if subsection, write subsection footer
        if self.subsection {
            file.write(
                Some(&format!(
                    "{}{}{}{}",
                    autodoc_tokenizer::OPEN_CHAR,
                    autodoc_tokenizer::OPEN_CHAR,
                    autodoc_tokenizer::CLOSE_CHAR,
                    autodoc_tokenizer::CLOSE_CHAR
                )),
                writer_id,
            )?;
            file.new_line(writer_id)?;
        }
        Ok(())
    }

    /// Java package-private `print(int)`.
    ///
    /// # Safety
    /// Every token, statement and attribute in the section must be live.
    pub unsafe fn print(&self, level: i32) {
        if level > 0 {
            autodoc::print_indent(level);
            print!("[");
        } else {
            println!();
        }
        print!(
            "[{} = {}]",
            unsafe { (*self.r#type).get_values() },
            unsafe { (*self.name).get_values() }
        );
        if level > 0 {
            println!("]");
        } else {
            println!();
        }
        // name value pair list
        autodoc::print_indent(level);
        println!("Statements:");
        for i in 0..self.statement_list.len() {
            let statement =
                self.statement_list[i].as_ref() as *const dyn Statement as *mut dyn Statement;
            unsafe { (*statement).print(level) };
        }
        autodoc::print_indent(level);
        println!("Attributes:");
        unsafe { self.attribute_list.as_deref().unwrap().print(level) };
    }

    /// Java private `getMostRecentStatement()`.
    fn get_most_recent_statement(&self) -> *mut dyn Statement {
        if self.statement_list.is_empty() {
            return std::ptr::null_mut::<EmptyStatement>();
        }
        self.statement_list[self.statement_list.len() - 1].as_ref() as *const dyn Statement
            as *mut dyn Statement
    }
}

impl WriteOnlyAttributeList for Section {
    /// Java `addAttribute(Token, int)`.
    unsafe fn add_attribute(&mut self, name: *mut Token, line_num: i32) -> *mut Attribute {
        unsafe {
            self.attribute_list
                .as_deref_mut()
                .unwrap()
                .add_attribute(name, line_num)
        }
    }

    /// Java `isGlobal()`.
    fn is_global(&self) -> bool {
        false
    }

    /// Java `isAttribute()`.
    fn is_attribute(&self) -> bool {
        false
    }
}

impl WriteOnlyStatementList for Section {
    /// Java `addNameValuePair(int)`.
    unsafe fn add_name_value_pair(&mut self, line_num: i32) -> *mut NameValuePair {
        let this: *mut Section = self;
        let pair = unsafe { NameValuePair::new(this, self.get_most_recent_statement(), line_num) };
        self.statement_list.push(unsafe { Box::from_raw(pair) });
        pair
    }

    /// Java `addSection(Token, Token, int)`.  Adds a subsection to a section.
    unsafe fn add_section(
        &mut self,
        r#type: *mut Token,
        name: *mut Token,
        line_num: i32,
    ) -> *mut Section {
        let this: *mut Section = self;
        let section = unsafe { Section::new(r#type, name, this) };
        unsafe { (*section).subsection = true };
        let subsection =
            unsafe { Subsection::new(section, this, self.get_most_recent_statement(), line_num) };
        self.statement_list
            .push(unsafe { Box::from_raw(subsection) });
        self.section_list.push(unsafe { Box::from_raw(section) });
        self.sub_section_map.insert(
            match unsafe { (*section).get_key() } {
                None => panic!("java.lang.NullPointerException"),
                Some(key) => key,
            },
            std::ptr::NonNull::new(section).expect("new section is non-null"),
        );
        section
    }

    /// Java `addComment(Token, int)`.
    unsafe fn add_comment(&mut self, comment: *mut Token, line_num: i32) {
        let this: *mut Section = self;
        let statement =
            unsafe { Comment::new(comment, this, self.get_most_recent_statement(), line_num) };
        self.statement_list
            .push(unsafe { Box::from_raw(statement) });
    }

    /// Java `addEmptyLine(int)`.
    unsafe fn add_empty_line(&mut self, line_num: i32) {
        let this: *mut Section = self;
        let statement = unsafe { EmptyLine::new(this, self.get_most_recent_statement(), line_num) };
        self.statement_list
            .push(unsafe { Box::from_raw(statement) });
    }

    /// Java `setCurrentDelimiter(Token)`.
    unsafe fn set_current_delimiter(&mut self, new_delimiter: *mut Token) {
        unsafe { (*self.parent).set_current_delimiter(new_delimiter) };
    }

    /// Java `getCurrentDelimiter()`.
    fn get_current_delimiter(&self) -> String {
        unsafe { (*self.parent).get_current_delimiter() }
    }
}

impl ReadOnlyStatementList for Section {
    /// Java `getString()`.
    fn get_string(&self) -> String {
        let mut buffer = String::new();
        buffer.push(autodoc_tokenizer::OPEN_CHAR);
        if self.subsection {
            buffer.push(autodoc_tokenizer::OPEN_CHAR);
        }
        buffer.push_str(&format!(
            "{} {} {}{}",
            unsafe { (*self.r#type).get_values() },
            autodoc_tokenizer::DEFAULT_DELIMITER,
            unsafe { (*self.name).get_values() },
            autodoc_tokenizer::CLOSE_CHAR
        ));
        if self.subsection {
            buffer.push(autodoc_tokenizer::CLOSE_CHAR);
        }
        buffer
    }

    /// Java `getStatementLocation()`.
    fn get_statement_location(&self) -> Option<StatementLocation> {
        Some(StatementLocation::new())
    }

    /// Java `nextStatement(StatementLocation)`.
    unsafe fn next_statement(
        &self,
        location: Option<&mut StatementLocation>,
    ) -> *mut dyn Statement {
        let location = match location {
            None => return std::ptr::null_mut::<EmptyStatement>(),
            Some(location) => location,
        };
        if location.is_out_of_range(Some(&self.statement_list)) {
            return std::ptr::null_mut::<EmptyStatement>();
        }
        let statement = self.statement_list[location.get_index()].as_ref() as *const dyn Statement
            as *mut dyn Statement;
        location.increment();
        statement
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        if self.name.is_null() {
            panic!("java.lang.IllegalStateException: name is required");
        }
        Some(unsafe { (*self.name).get_values() })
    }
}

impl ReadOnlySectionList for Section {
    /// Java `getSection(String, String)`.  Gets a subsection.
    unsafe fn get_section(
        &self,
        sub_section_type: Option<&str>,
        sub_section_name: Option<&str>,
    ) -> *mut Section {
        match get_key_of_strings(sub_section_type, sub_section_name) {
            None => std::ptr::null_mut(),
            Some(key) => self
                .sub_section_map
                .get(&key)
                .map_or(std::ptr::null_mut(), |section| section.as_ptr()),
        }
    }

    /// Java `getSectionLocation(String)`.
    ///
    /// Sets a SectionLocation index to the first section with the type the same as
    /// the type parameter.
    unsafe fn get_section_location_by_type(&self, r#type: Option<&str>) -> Option<SectionLocation> {
        let mut section: *mut Section;
        for i in 0..self.section_list.len() {
            section = self.section_list[i].as_ref() as *const Section as *mut Section;
            if unsafe { (*section).equals_type(r#type) } {
                return Some(SectionLocation::new(
                    r#type.map(|r#type| r#type.to_string()),
                    i as i32,
                ));
            }
        }
        None
    }

    /// Java `getSectionLocation()`.
    fn get_section_location(&self) -> Option<SectionLocation> {
        if !self.section_list.is_empty() {
            return Some(SectionLocation::new_with_index(0));
        }
        None
    }

    /// Java `nextSection(SectionLocation)`.
    ///
    /// Starts with the section that location is pointing to returns the first
    /// section which the same type as location.  Increments location.
    unsafe fn next_section(&self, location: Option<&mut SectionLocation>) -> *mut Section {
        let location = match location {
            None => return std::ptr::null_mut(),
            Some(location) => location,
        };
        let mut section: *mut Section;
        for i in location.get_index()..self.section_list.len() as i32 {
            section = self.section_list[i as usize].as_ref() as *const Section as *mut Section;
            if unsafe { (*section).equals_type(location.get_type()) } {
                location.set_index(i + 1);
                return section;
            }
        }
        std::ptr::null_mut()
    }

    /// Java `getString()`, the single body that satisfies both interface declarations.
    fn get_string(&self) -> String {
        ReadOnlyStatementList::get_string(self)
    }

    /// Java `setDebug()`.
    fn set_debug(&mut self) {
        self.debug = true;
    }

    /// Java `getName()`, the single body that satisfies both interface declarations.
    fn get_name(&self) -> Option<String> {
        ReadOnlyStatementList::get_name(self)
    }
}

impl ReadOnlySection for Section {
    /// Java `getAttribute(String)`.
    unsafe fn get_attribute(&self, name: Option<&str>) -> *mut Attribute {
        unsafe { self.attribute_list.as_deref().unwrap().get_attribute(name) }
    }

    /// Java `getType()`.
    fn get_type(&self) -> String {
        if self.r#type.is_null() {
            panic!("java.lang.IllegalStateException: type is required");
        }
        unsafe { (*self.r#type).get_values() }
    }
}

#[cfg(test)]
mod tests {
    use super::{get_key_of_strings, get_key_of_tokens};
    use crate::imod::etomo::ui::swing::token::{self, Token};

    /// Both `getKey` overloads lower-case each half and concatenate them, and a null
    /// half drops out of the key entirely.
    #[test]
    fn section_keys_match_the_source_overloads() {
        let mut r#type = Token::new();
        r#type.set_type_and_string(token::Type::Anything, "Field");
        let mut name = Token::new();
        name.set_type_and_string(token::Type::Anything, "Name");
        assert_eq!(
            get_key_of_tokens(Some(&r#type), Some(&name)).as_deref(),
            Some("fieldname")
        );
        assert_eq!(
            get_key_of_tokens(Some(&r#type), None).as_deref(),
            Some("field")
        );
        assert_eq!(
            get_key_of_tokens(None, Some(&name)).as_deref(),
            Some("name")
        );
        assert_eq!(get_key_of_tokens(None, None), None);
        assert_eq!(
            get_key_of_strings(Some("FiElD"), Some("NaMe")).as_deref(),
            Some("fieldname")
        );
        assert_eq!(
            get_key_of_strings(None, Some("NaMe")).as_deref(),
            Some("name")
        );
        assert_eq!(get_key_of_strings(None, None), None);
    }
}
