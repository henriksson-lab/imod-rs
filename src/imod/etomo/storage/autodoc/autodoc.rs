//! `IMOD/Etomo/src/etomo/storage/autodoc/Autodoc.java`.
//!
//! Description:  Data storage for an autodoc file.
//!
//! Versions:
//! 1.0
//! 1.1:  Added the break character "^".  When a value is formatted, the "^" is
//! replaced with a "\n".
//! 1.2:  Handling duplicate attribute names.  Duplicate attributes are
//! attributes with the same parentage (section and parent attribute names) and
//! the same name.  Before version 1.2 the last value assigned to a duplicate
//! attribute could be retrieved by using the name as a key.  In version 1.2
//! the first value assigned to a duplicate attribute can be retrieved by using the
//! name as a key.  When attributes of the same parentage are retrieved as an
//! ordered list, each duplicate attribute, and their different values, will be
//! included in the list.
//!
//! 1.3:  Fixed handling duplicate attribute names (see 1.2).  The rule for
//! autodoc is that is keeps the last value.  So it needs to keep acting like it
//! is doing this even though it is actually keeping  all the values.
//!
//! `@notthreadsafe`
//!
//! `BaseManager` has no module; every caller that reaches an `initialize...` member
//! passes a null one, so the parameter is typed `Option<Infallible>`.  Java's
//! `initialize...` members are instance methods that assign `this.parser`; a Rust
//! `&mut self` would alias the `*mut Autodoc` the parser stores, so each takes the raw
//! pointer instead.
#![allow(dead_code)]

use super::writable_statement::WritableStatement;
use crate::imod::etomo::base_manager::BaseManager;

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::log_file::{self, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::utilities;

use super::attribute::Attribute;
use super::attribute_list::AttributeList;
use super::autodoc_parser::AutodocParser;
use super::autodoc_tokenizer;
use super::comment::Comment;
use super::empty_line::EmptyLine;
use super::name_value_pair::NameValuePair;
use super::read_only_autodoc::ReadOnlyAutodoc;
use super::read_only_section::ReadOnlySection;
use super::read_only_section_list::ReadOnlySectionList;
use super::read_only_statement::ReadOnlyStatement;
use super::read_only_statement_list::ReadOnlyStatementList;
use super::section::{self, Section};
use super::section_location::SectionLocation;
use super::statement::{EmptyStatement, Statement};
use super::statement_location::StatementLocation;
use super::writable_autodoc::WritableAutodoc;
use super::write_only_attribute_list::WriteOnlyAttributeList;
use super::write_only_statement_list::WriteOnlyStatementList;
use crate::imod::etomo::ui::swing::token::{self, Token};
use std::collections::HashMap;

/// Java's nested `static final class InternalTestType`.  Each constant is a distinct
/// object compared with `==`, so the typesafe enum is a Rust enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InternalTestType {
    /// Java `STREAM_TOKENIZER`.
    StreamTokenizer,
    /// Java `PRIMATIVE_TOKENIZER`.
    PrimativeTokenizer,
    /// Java `AUTODOC_TOKENIZER`.
    AutodocTokenizer,
    /// Java `PREPROCESSOR`.
    Preprocessor,
    /// Java `PARSER`.
    Parser,
}

/// Java static package-private `printIndent(int)`.
pub fn print_indent(level: i32) {
    if level == 0 {
        return;
    }
    for _i in 0..level {
        print!("  ");
    }
}

/// Java public final `Autodoc extends WriteOnlyStatementList implements
/// WritableAutodoc`.
pub struct Autodoc {
    /// Java field `autodocName`: the autodoc file name, excluding the extension.
    autodoc_name: String,
    /// Java field `parser`, initialised to null.
    parser: *mut AutodocParser,
    // data
    /// Java field `sectionList`.
    section_list: Vec<*mut Section>,
    /// Java field `sectionMap`.
    section_map: HashMap<String, *mut Section>,
    /// Java field `statementList`.
    statement_list: Vec<*mut dyn Statement>,
    /// Java field `attributeList`.
    attribute_list: *mut AttributeList,
    /// Java field `currentDelimiter`, initialised to
    /// `AutodocTokenizer.DEFAULT_DELIMITER`.
    current_delimiter: String,
    /// Java field `debug`, initialised to false.
    debug: bool,
    /// Java field `writable`, initialised to false.
    writable: bool,
    /// Java field `errMsg`, a caller-owned `StringBuilder` that the parser appends to.
    err_msg: *mut String,
}

impl Autodoc {
    /// Java package-private `Autodoc(String, StringBuilder)`.  Java's
    /// `new AttributeList(this)` needs the object, which Rust cannot produce before the
    /// allocation, so the attribute list is assigned immediately after it.
    ///
    /// # Safety
    /// `err_msg` must be null or point to a live `String`.
    pub unsafe fn new(autodoc_name: Option<&str>, err_msg: *mut String) -> *mut Autodoc {
        let this = Box::into_raw(Box::new(Autodoc {
            autodoc_name: match autodoc_name {
                None => String::new(),
                Some(autodoc_name) => autodoc_name.to_string(),
            },
            parser: std::ptr::null_mut(),
            section_list: Vec::new(),
            section_map: HashMap::new(),
            statement_list: Vec::new(),
            attribute_list: std::ptr::null_mut(),
            current_delimiter: autodoc_tokenizer::DEFAULT_DELIMITER.to_string(),
            debug: false,
            writable: false,
            err_msg,
        }));
        unsafe { (*this).attribute_list = Box::into_raw(Box::new(AttributeList::new(this))) };
        this
    }

    /// Java `write()`.
    ///
    /// # Safety
    /// The parser, every statement and every section must be live.
    pub unsafe fn write(&self) -> Result<(), LogFileError> {
        if !self.writable {
            // `new IllegalStateException("Not a writable autodoc.").printStackTrace()`;
            // see etomo/util/stack_trace.rs.
            eprintln!("java.lang.IllegalStateException: Not a writable autodoc.");
            return Ok(());
        }
        let autodoc_file = unsafe { (*self.parser).get_log_file() }.unwrap();
        let writer_id = autodoc_file.open_writer()?;
        for statement in self.statement_list.iter() {
            unsafe { (**statement).write(&autodoc_file, &writer_id)? };
        }
        for section in self.section_list.iter() {
            unsafe { (**section).write(&autodoc_file, &writer_id)? };
        }
        autodoc_file.close_id(Some(&writer_id));
        Ok(())
    }

    /// Java `getLogFile()`.
    ///
    /// # Safety
    /// The parser must be live.
    pub unsafe fn get_log_file(&self) -> Option<std::sync::Arc<log_file::Handle>> {
        unsafe { (*self.parser).get_log_file() }
    }

    /// Java private `getDir(BaseManager, String, String, AxisID)`.
    fn get_dir(
        &self,
        manager: Option<&'static dyn BaseManager>,
        env_variable: Option<&str>,
        dir_name: Option<&str>,
        axis_id: AxisID,
    ) -> Option<std::path::PathBuf> {
        let parent_dir = utilities::get_existing_dir(manager, env_variable, axis_id)?;
        let dir = parent_dir.join(dir_name.unwrap_or("null"));
        if !utilities::check_existing_dir(&dir, env_variable) {
            return None;
        }
        Some(dir)
    }

    /// Java package-private `initializeGenericInstance(BaseManager, String, String,
    /// String, AxisID, boolean)`.
    ///
    /// # Safety
    /// `this` must point to a live `Autodoc`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn initialize_generic_instance_env_var(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        env_var: Option<&str>,
        subdir_name: Option<&str>,
        name: Option<&str>,
        axis_id: AxisID,
        writable: bool,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance_env_var(
                this,
                false,
                true,
                false,
                env_var,
                subdir_name,
                name,
                manager,
                axis_id,
                None,
                debug,
                writable,
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java package-private `initializeGenericInstance(BaseManager, File, AxisID,
    /// boolean, StringBuilder)`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_generic_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        autodoc_file: &std::path::Path,
        axis_id: AxisID,
        writable: bool,
        err_msg: *mut String,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).writable = writable;
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance(
                this,
                false,
                false,
                false,
                Some(autodoc_file),
                manager,
                axis_id,
                None,
                debug,
                writable,
                err_msg,
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            if autodoc_file.exists() {
                (*(*this).parser).initialize()?;
                (*(*this).parser).parse();
            }
        }
        Ok(())
    }

    /// Java package-private `initializeMatlabInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_matlab_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        file: &std::path::Path,
        writable: bool,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).writable = writable;
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance(
                this,
                true,
                false,
                true,
                Some(file),
                manager,
                AxisID::Only,
                None,
                debug,
                true,
                std::ptr::null_mut(),
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java package-private `initializeWritableInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_writable_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        file: &std::path::Path,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).writable = true;
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance(
                this,
                false,
                false,
                false,
                Some(file),
                manager,
                AxisID::Only,
                None,
                debug,
                true,
                std::ptr::null_mut(),
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java package-private `initializeEmptyWritableInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_empty_writable_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        file: &std::path::Path,
    ) {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).writable = true;
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance(
                this,
                false,
                false,
                false,
                Some(file),
                manager,
                AxisID::Only,
                None,
                debug,
                true,
                std::ptr::null_mut(),
            )));
        }
    }

    /// Java package-private `initializeEmptyMatlapInstance`.  Initializes a writable peet
    /// autodoc.  Parse is not initialized.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_empty_matlap_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        file: &std::path::Path,
    ) {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).writable = true;
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance(
                this,
                true,
                false,
                true,
                Some(file),
                manager,
                AxisID::Only,
                None,
                debug,
                true,
                std::ptr::null_mut(),
            )));
        }
    }

    /// Java package-private `initializeAutodocInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_autodoc_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
        axis_id: AxisID,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_autodoc_instance(
                this, false, true, false, name, manager, axis_id, None, debug,
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java package-private `initializeUnmanagedAutodocInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_unmanaged_autodoc_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
        autodoc_file: Option<&std::path::Path>,
        axis_id: AxisID,
    ) -> Result<(), LogFileError> {
        unsafe {
            (*this).parser =
                Box::into_raw(Box::new(AutodocParser::get_unmanaged_autodoc_instance(
                    this,
                    false,
                    true,
                    false,
                    name,
                    autodoc_file,
                    manager,
                    axis_id,
                    None,
                )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java package-private `initializeUITestInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_ui_test_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
        axis_id: AxisID,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance_env_var(
                this,
                false,
                true,
                false,
                Some(etomo_director::SOURCE_ENV_VAR),
                None,
                name,
                manager,
                axis_id,
                None,
                debug,
                false,
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java package-private `initializeCpuInstance`.
    ///
    /// # Safety
    /// See `initialize_generic_instance_env_var`.
    pub unsafe fn initialize_cpu_instance(
        this: *mut Autodoc,
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
        axis_id: AxisID,
    ) -> Result<(), LogFileError> {
        let debug = unsafe { (*this).debug };
        unsafe {
            (*this).parser = Box::into_raw(Box::new(AutodocParser::get_generic_instance_env_var(
                this,
                false,
                true,
                false,
                Some(environment_variable::CALIB_DIR),
                None,
                name,
                manager,
                axis_id,
                Some(
                    "Info:  No local calibration information is available.  There is no \
                     cpu.adoc file.  Parallel processing on multiple machines will not \
                     be available unless it has been enabled in the Etomo Settings \
                     dialog (under the Options menu).",
                ),
                debug,
                false,
            )));
            // To test comment out initialize and parse and uncomment runInternalTest.
            // runInternalTest(InternalTestType.STREAM_TOKENIZER, true, false);
            (*(*this).parser).initialize()?;
            (*(*this).parser).parse();
        }
        Ok(())
    }

    /// Java private `getMostRecentStatement()`.
    fn get_most_recent_statement(&self) -> *mut dyn Statement {
        if self.statement_list.is_empty() {
            return std::ptr::null_mut::<EmptyStatement>();
        }
        self.statement_list[self.statement_list.len() - 1]
    }

    /// Java private `getAttributeValues(String, String, boolean)`.
    ///
    /// Returns a HashMap containing a list of attribute values, keyed by
    /// sectionName.  The elements in each list is using section type and attribute
    /// name.
    ///
    /// The source wraps the body in `catch (NullPointerException)` with the comment
    /// "An attribute with attributeName doesn't exist"; the null attribute is tested
    /// directly here.
    ///
    /// # Safety
    /// Every section and attribute in the autodoc must be live.
    unsafe fn get_attribute_values_internal(
        &self,
        section_type: Option<&str>,
        attribute_name: Option<&str>,
        multi_line: bool,
    ) -> Option<HashMap<String, Option<String>>> {
        if section_type.is_none() || attribute_name.is_none() {
            return None;
        }
        // Create attributeValues
        let mut attribute_values: HashMap<String, Option<String>> = HashMap::new();
        let mut section_location = unsafe { self.get_section_location_by_type(section_type) };
        let mut section = unsafe { self.next_section(section_location.as_mut()) };
        while !section.is_null() {
            let section_name = unsafe { ReadOnlyStatementList::get_name(&*section) };
            let attribute = unsafe { ReadOnlySection::get_attribute(&*section, attribute_name) };
            if section_name.is_some() && !attribute.is_null() {
                if multi_line {
                    attribute_values.insert(section_name.unwrap(), unsafe {
                        crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute::get_multi_line_value(&*attribute)
                    });
                } else {
                    attribute_values.insert(section_name.unwrap(), unsafe {
                        crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute::get_value(&*attribute)
                    });
                }
            }
            // Go to next section
            section = unsafe { self.next_section(section_location.as_mut()) };
        }
        Some(attribute_values)
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        if self.parser.is_null() {
            return "".to_string();
        }
        unsafe { (*self.parser).get_absolute_path() }
    }
}

impl WriteOnlyAttributeList for Autodoc {
    /// Java `addAttribute(Token, int)`.
    unsafe fn add_attribute(&mut self, name: *mut Token, line_num: i32) -> *mut Attribute {
        unsafe { (*self.attribute_list).add_attribute(name, line_num) }
    }

    /// Java `isGlobal()`.
    fn is_global(&self) -> bool {
        true
    }

    /// Java `isAttribute()`.
    fn is_attribute(&self) -> bool {
        false
    }
}

impl WriteOnlyStatementList for Autodoc {
    /// Java `addSection(Token, Token, int)`.
    unsafe fn add_section(
        &mut self,
        r#type: *mut Token,
        name: *mut Token,
        line_num: i32,
    ) -> *mut Section {
        let _ = line_num;
        let this: *mut Autodoc = self;
        let existing_section: *mut Section;
        let key = unsafe { section::get_key_of_tokens(r#type, name) };
        existing_section = match &key {
            None => std::ptr::null_mut(),
            Some(key) => *self.section_map.get(key).unwrap_or(&std::ptr::null_mut()),
        };
        if existing_section.is_null() {
            let new_section = unsafe { Section::new(r#type, name, this) };
            self.section_list.push(new_section);
            self.section_map.insert(
                match unsafe { (*new_section).get_key() } {
                    None => panic!("java.lang.NullPointerException"),
                    Some(key) => key,
                },
                new_section,
            );
            return new_section;
        }
        existing_section
    }

    /// Java `addNameValuePair(int)`.
    unsafe fn add_name_value_pair(&mut self, line_num: i32) -> *mut NameValuePair {
        let this: *mut Autodoc = self;
        let pair = unsafe { NameValuePair::new(this, self.get_most_recent_statement(), line_num) };
        self.statement_list.push(pair);
        pair
    }

    /// Java `addComment(Token, int)`.
    unsafe fn add_comment(&mut self, comment: *mut Token, line_num: i32) {
        let this: *mut Autodoc = self;
        let statement =
            unsafe { Comment::new(comment, this, self.get_most_recent_statement(), line_num) };
        self.statement_list.push(statement);
    }

    /// Java `addEmptyLine(int)`.
    unsafe fn add_empty_line(&mut self, line_num: i32) {
        let this: *mut Autodoc = self;
        let statement = unsafe { EmptyLine::new(this, self.get_most_recent_statement(), line_num) };
        self.statement_list.push(statement);
    }

    /// Java `setCurrentDelimiter(Token)`.
    unsafe fn set_current_delimiter(&mut self, new_delimiter: *mut Token) {
        self.current_delimiter = unsafe { (*new_delimiter).get_values() };
    }

    /// Java `getCurrentDelimiter()`.
    fn get_current_delimiter(&self) -> String {
        self.current_delimiter.clone()
    }
}

impl ReadOnlyStatementList for Autodoc {
    /// Java `getString()`.
    fn get_string(&self) -> String {
        if self.parser.is_null() {
            return "".to_string();
        }
        unsafe { (*self.parser).get_absolute_path() }
    }

    /// Java `getStatementLocation()`.  The source's `statementList == null` guard cannot
    /// fail; the field is final and assigned at its declaration.
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
        let statement = self.statement_list[location.get_index()];
        location.increment();
        statement
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        if !self.parser.is_null() {
            return Some(unsafe { (*self.parser).get_file_name() });
        }
        None
    }
}

impl ReadOnlySectionList for Autodoc {
    /// Java `getSection(String, String)`.
    unsafe fn get_section(&self, r#type: Option<&str>, name: Option<&str>) -> *mut Section {
        if self.debug {
            println!(
                "Autodoc.getSection:type={},name={}",
                r#type.unwrap_or("null"),
                name.unwrap_or("null")
            );
        }
        // The source's `sectionMap == null` guard cannot fail; the field is final.
        let key = section::get_key_of_strings(r#type, name);
        if self.debug {
            println!(
                "Autodoc.getSection:key={}",
                match &key {
                    None => "null".to_string(),
                    Some(key) => key.clone(),
                }
            );
        }
        let section: *mut Section = match &key {
            None => std::ptr::null_mut(),
            Some(key) => *self.section_map.get(key).unwrap_or(&std::ptr::null_mut()),
        };
        if self.debug {
            println!(
                "Autodoc.getSection:section={}",
                if section.is_null() {
                    "null".to_string()
                } else {
                    unsafe { (*section).to_string() }
                }
            );
        }
        section
    }

    /// Java `getSectionLocation(String)`.
    ///
    /// Sets a SectionLocation index to the first section with the type the same as
    /// the type parameter.  Returns the SectionLocation index.
    unsafe fn get_section_location_by_type(&self, r#type: Option<&str>) -> Option<SectionLocation> {
        let mut section: *mut Section;
        for i in 0..self.section_list.len() {
            section = self.section_list[i];
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
            section = self.section_list[i as usize];
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

impl ReadOnlyAutodoc for Autodoc {
    /// Java `getAttributeValues(String, String)`.
    unsafe fn get_attribute_values(
        &self,
        section_type: Option<&str>,
        attribute_name: Option<&str>,
    ) -> Option<HashMap<String, Option<String>>> {
        unsafe { self.get_attribute_values_internal(section_type, attribute_name, false) }
    }

    /// Java `getAttributeMultiLineValues(String, String)`.
    unsafe fn get_attribute_multi_line_values(
        &self,
        section_type: Option<&str>,
        attribute_name: Option<&str>,
    ) -> Option<HashMap<String, Option<String>>> {
        unsafe { self.get_attribute_values_internal(section_type, attribute_name, true) }
    }

    /// Java `isError()`.
    fn is_error(&self) -> bool {
        if self.parser.is_null() {
            return true;
        }
        unsafe { (*self.parser).is_error() }
    }

    /// Java `printStoredData()`.
    unsafe fn print_stored_data(&self) {
        println!("Printing stored data:");
        // name value pair list
        println!("LIST:");
        // The source's `statementList != null` guard cannot fail; the field is final.
        let mut statement: *mut dyn Statement;
        for i in 0..self.statement_list.len() {
            statement = self.statement_list[i];
            unsafe { (*statement).print(0) };
        }
        // attribute map
        println!("Attributes:");
        unsafe { (*self.attribute_list).print(0) };
        // section list
        for i in 0..self.section_list.len() {
            let section = self.section_list[i];
            unsafe { (*section).print(0) };
        }
    }

    /// Java `sectionExists(String)`.
    unsafe fn section_exists(&self, r#type: Option<&str>) -> bool {
        unsafe { self.get_section_location_by_type(r#type) }.is_some()
    }

    /// Java `getAttribute(String)`.
    unsafe fn get_attribute(&self, name: Option<&str>) -> *mut Attribute {
        unsafe { (*self.attribute_list).get_attribute(name) }
    }

    /// Java `runInternalTest(InternalTestType, boolean, boolean)`.
    ///
    /// Initializes parser and prints parsing data instead of storing it.
    fn run_internal_test(
        &mut self,
        r#type: InternalTestType,
        show_tokens: bool,
        show_details: bool,
    ) {
        println!("runInternalTest");
        if r#type == InternalTestType::StreamTokenizer {
            unsafe { (*self.parser).test_stream_tokenizer(show_tokens, show_details) };
        } else if r#type == InternalTestType::PrimativeTokenizer {
            unsafe { (*self.parser).test_primative_tokenizer(show_tokens) };
        } else if r#type == InternalTestType::AutodocTokenizer {
            unsafe { (*self.parser).test_autodoc_tokenizer(show_tokens) };
        } else if r#type == InternalTestType::Preprocessor {
            unsafe { (*self.parser).test_preprocessor(show_tokens) };
        } else if r#type == InternalTestType::Parser {
            unsafe { (*self.parser).test(show_tokens, show_details) };
        }
    }

    /// Java `isDebug()`.
    fn is_debug(&self) -> bool {
        self.debug
    }

    /// Java `setDebug(boolean)`.
    fn set_debug_to(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java `getAutodocName()`.
    fn get_autodoc_name(&self) -> String {
        self.autodoc_name.clone()
    }

    /// Java `exists()`.  Java throws a `NullPointerException` with a null parser.
    fn exists(&self) -> bool {
        if self.parser.is_null() {
            panic!("java.lang.NullPointerException");
        }
        unsafe { (*self.parser).exists() }
    }

    /// Java `getChildren()`.
    fn get_children(&self) -> *mut AttributeList {
        self.attribute_list
    }
}

impl WritableAutodoc for Autodoc {
    /// Java `addNameValuePairAttribute(String, String)`.
    unsafe fn add_name_value_pair_attribute(&mut self, name: Option<&str>, value: Option<&str>) {
        unsafe { self.add_name_value_pair_attribute_with_line_num(name, value, 0) };
    }

    /// Java `addNameValuePairAttribute(String, String, int)`.
    ///
    /// add a name/value pair with a name containing one attribute
    unsafe fn add_name_value_pair_attribute_with_line_num(
        &mut self,
        name: Option<&str>,
        value: Option<&str>,
        line_num: i32,
    ) {
        let name = match name {
            None => return,
            // `String.trim()` strips code units <= ' ', which is not Rust's
            // `str::trim`: U+00A0 is Unicode whitespace but is above ' '.
            Some(name) => {
                let units: Vec<u16> = name.encode_utf16().collect();
                let mut start = 0usize;
                let mut end = units.len();
                while start < end && units[start] <= ' ' as u16 {
                    start += 1;
                }
                while start < end && units[end - 1] <= ' ' as u16 {
                    end -= 1;
                }
                String::from_utf16_lossy(&units[start..end])
            }
        };
        if !name.contains(autodoc_tokenizer::SEPARATOR_CHAR) {
            // Not dot separators - entire attribute is saved at this level.
            // add attribute
            let name_token = Box::into_raw(Box::new(Token::new()));
            unsafe { (*name_token).set_type_and_string(token::Type::Anything, &name) };
            unsafe { (*self.attribute_list).add_attribute(name_token, line_num) };
            // add value to attribute
            let attribute = unsafe { (*self.attribute_list).get_attribute(Some(&name)) };
            let value_token = Box::into_raw(Box::new(Token::new()));
            unsafe {
                (*value_token).set_type_and_string(
                    token::Type::Anything,
                    match value {
                        None => panic!("java.lang.NullPointerException"),
                        Some(value) => value,
                    },
                )
            };
            // add name/value pair
            let pair = unsafe { self.add_name_value_pair(line_num) };
            // add attribute and value to pair
            unsafe { (*pair).add_attribute(attribute) };
            unsafe { (*pair).add_value(value_token) };
        } else {
            // Multiple-level attribute
            // Java splits on the quoted separator, which `String.split` returns without
            // trailing empty strings.
            let mut parts: Vec<Option<String>> = name
                .split(autodoc_tokenizer::SEPARATOR_CHAR)
                .map(|part| Some(part.to_string()))
                .collect();
            while !parts.is_empty() && parts[parts.len() - 1].as_deref() == Some("") {
                parts.pop();
            }
            let pair = unsafe { self.add_name_value_pair(line_num) };
            unsafe {
                (*self.attribute_list).add_attribute_multi(0, Some(&parts), line_num, value, pair)
            };
        }
    }

    /// Java `getWritableAttribute(String)`.
    unsafe fn get_writable_attribute(&self, name: Option<&str>) -> *mut Attribute {
        unsafe { (*self.attribute_list).get_attribute(name) }
    }

    /// Java `addComment(Token, int)`, the single body that also satisfies
    /// `WriteOnlyStatementList`.
    unsafe fn add_comment(&mut self, comment: *mut Token, line_num: i32) {
        unsafe { WriteOnlyStatementList::add_comment(self, comment, line_num) };
    }

    /// Java `addEmptyLine(int)`, the single body that also satisfies
    /// `WriteOnlyStatementList`.
    unsafe fn add_empty_line(&mut self, line_num: i32) {
        unsafe { WriteOnlyStatementList::add_empty_line(self, line_num) };
    }

    /// Java `addComment(String, int)`.
    unsafe fn add_comment_string(&mut self, comment: Option<&str>, line_num: i32) {
        let token = Box::into_raw(Box::new(Token::new()));
        unsafe {
            (*token).set_type_and_string(
                token::Type::Anything,
                match comment {
                    None => panic!("java.lang.NullPointerException"),
                    Some(comment) => comment,
                },
            )
        };
        unsafe { WriteOnlyStatementList::add_comment(self, token, line_num) };
    }

    /// Java `removeNameValuePair(String)`.
    ///
    /// Removes a simple (single attribute) name/value pair.  Removes the occurrance
    /// of the attribute in the name/value pair.  Returns the previous statement in
    /// statementList.
    unsafe fn remove_name_value_pair(&mut self, name: Option<&str>) -> *mut dyn Statement {
        let attribute = unsafe { (*self.attribute_list).get_attribute(name) };
        if attribute.is_null() {
            // unable to find an attribute with this name
            return std::ptr::null_mut::<EmptyStatement>();
        }
        let pair = unsafe { (*attribute).get_name_value_pair() };
        if let Some(index) = self
            .statement_list
            .iter()
            .position(|statement| std::ptr::addr_eq(*statement, pair))
        {
            self.statement_list.remove(index);
        }
        unsafe { (*pair).remove() }
    }

    /// Java `removeStatement(WritableStatement)`.  Returns the previous statement in
    /// statementList.
    unsafe fn remove_statement(&mut self, statement: *mut dyn Statement) -> *mut dyn Statement {
        if let Some(index) = self
            .statement_list
            .iter()
            .position(|entry| std::ptr::addr_eq(*entry, statement))
        {
            self.statement_list.remove(index);
        }
        unsafe { (*statement).remove() }
    }

    /// Java `printStatementList()`.  `"statementList=" + statementList` renders the
    /// `ArrayList` through each statement's `toString`; only `NameValuePair` overrides
    /// it, so the other three print `java.lang.Object.toString`'s identity hash, which
    /// is not reproducible.
    fn print_statement_list(&self) {
        let mut buffer = String::from("statementList=[");
        for i in 0..self.statement_list.len() {
            if i > 0 {
                buffer.push_str(", ");
            }
            buffer.push_str(&unsafe { (*self.statement_list[i]).get_string() });
        }
        buffer.push(']');
        println!("{}", buffer);
    }

    /// Java `wrapAttributeValues(String, String, String, String, int, int)`.
    ///
    /// Wraps all global attribute leaf values in the autodoc, except values starting
    /// with noWrapTag.  If a value starts with wrapTag, uses divider, otherwise uses
    /// defaultDivider.  Not wrapping non-global attributes as this is not currently
    /// required.
    unsafe fn wrap_attribute_values(
        &mut self,
        no_wrap_prefix: Option<&str>,
        wrap_prefix: Option<&str>,
        divider: Option<&str>,
        default_divider: Option<&str>,
        min_length: i32,
        wrap_length: i32,
    ) {
        let size = self.statement_list.len();
        for i in 0..size {
            unsafe {
                (*self.statement_list[i]).wrap_value(
                    no_wrap_prefix,
                    wrap_prefix,
                    divider,
                    default_divider,
                    min_length,
                    wrap_length,
                )
            };
        }
        if !self.section_list.is_empty() {
            eprintln!("Info: Non-global sections will not be wrapped.");
        }
    }
}

/// Keeps `ReadOnlyStatement` in scope for the statements this autodoc prints.
const _: Option<&dyn ReadOnlyStatement> = None;
