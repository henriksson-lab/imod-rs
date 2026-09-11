//! `IMOD/Etomo/src/etomo/storage/autodoc/AutodocParser.java`.
//!
//! Description:
//! Parses an autodoc file.  Finds and saves autodoc elements in an Autodoc
//! object.
//!
//! AutodocParser is not case sensitive.  It stores all text in the original case.
//! It retains the original whitespace, except for end of line.  It substitute one
//! space for each end of line character in a multi-line value.  Comments and
//! empty lines are stored.  Messages about syntax errors are sent to System.err.
//! It stores attributes in a tree structure and as individual ordered name/value
//! pairs.  When there are duplicate attributes in a section, retrieving the
//! value from the tree structure retrieves the last value.
//!
//! ```text
//! Language definition for the parser:
//!
//! Autodoc => { emptyLine | comment | pair | section } EOF
//!
//! section => OPEN sectionHeader CLOSE -WHITESPACE- ( EOL | EOF )
//!            { emptyLine | comment | pair | subsection }
//!
//! sectionHeader => sectionType -WHITESPACE- DELIMITER -WHITESPACE- sectionName -WHITESPACE-
//!
//! sectionType => ( WORD | KEYWORD )
//!
//! sectionName => [ \CLOSE & SUBCLOSE & WHITESPACE & EOL & EOF\ ]
//!
//! subsection => SUBOPEN sectionHeader SUBCLOSE -WHITESPACE- ( EOL | EOF )
//!               { emptyLine | comment | pair }
//!               subsectionClose
//!
//! subsectionClose => SUBOPEN -WHITESPACE- SUBCLOSE (EOL | EOF )
//!
//! pair => name -WHITESPACE- DELIMITER -WHITESPACE- -QUOTE#- value
//!
//! name => base-attribute { SEPARATOR attribute }
//!
//! base-attribute => ( WORD | KEYWORD | QUOTE ) -attribute-
//!
//! attribute => [ WORD | KEYWORD | COMMENT | QUOTE ]
//!
//! value => { \EOL & EOF\ } ( EOL | EOF ) { (!quoted & valueline ) | ( quoted & ( quotedValueLine | #QUOTE ) ) }
//!
//! valueLine => !emptyLine !DelimiterInLine !comment { \DELIMITER & SUBOPEN & EOL & EOF\ } ( EOL | EOF )
//!
//! quotedValueLine => !emptyLine { \EOL & EOF\ } #QUOTE -WHITESPACE- ( EOL | EOF )
//!
//! comment => COMMENT { \EOL\ } ( EOL | EOF )
//!
//! emptyLine => ^ -WHITESPACE- ( EOL | EOF )
//! ```
//!
//! `BaseManager` has no module; every caller that reaches the constructor passes a null
//! one, so the parameter is typed `Option<Infallible>`.
#![allow(dead_code)]

use super::attribute::Attribute;
use super::autodoc::Autodoc;
use super::autodoc_tokenizer::{self, AutodocTokenizer, Location};
use super::name_value_pair::NameValuePair;
use super::read_only_attribute::ReadOnlyAttribute;
use super::section::Section;
use super::write_only_attribute_list::WriteOnlyAttributeList;
use super::write_only_statement_list::WriteOnlyStatementList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{self, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::token::{self, Token};

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`, read once when the
/// class is initialised.
static DEBUG: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    crate::imod::etomo::etomo_director::ARGUMENTS
        .lock()
        .unwrap()
        .is_debug()
});

/// Java's nested `private final static class LinkList`.
struct LinkList {
    /// Java field `head`.
    head: *mut Token,
    /// Java field `tail`.
    tail: *mut Token,
    /// Java field `size`.
    size: i32,
    /// Java field `done`, initialised to false.
    done: bool,
}

impl LinkList {
    /// Java package-private `LinkList(Token)`.
    fn new(start: *mut Token) -> LinkList {
        LinkList {
            head: start,
            tail: start,
            size: 1,
            done: false,
        }
    }

    /// Java private `setDone()`.
    fn set_done(&mut self) {
        self.done = true;
    }

    /// Java private `isDone()`.
    fn is_done(&self) -> bool {
        self.done
    }

    /// Java package-private `append(Token)`.
    ///
    /// # Safety
    /// `token` and the list must be live.
    unsafe fn append(&mut self, token: *mut Token) {
        if self.tail.is_null() {
            self.head = token;
            self.tail = token;
            self.size = 1;
        } else {
            self.tail = unsafe { (*self.tail).set_next(token) };
            self.size += 1;
        }
    }

    /// Java package-private `size()`.
    fn size(&self) -> i32 {
        self.size
    }

    /// Java package-private `isFirstElement(Token.Type)`.
    ///
    /// # Safety
    /// The list must be live.
    unsafe fn is_first_element(&self, compare_type: token::Type) -> bool {
        if self.head.is_null() {
            return false;
        }
        unsafe { (*self.head).is(compare_type) }
    }

    /// Java package-private `isLastElement(Token.Type)`.
    ///
    /// # Safety
    /// The list must be live.
    unsafe fn is_last_element(&self, compare_type: token::Type) -> bool {
        if self.tail.is_null() {
            return false;
        }
        unsafe { (*self.tail).is(compare_type) }
    }

    /// Java package-private `dropFirstElement()`.
    ///
    /// # Safety
    /// The list must be live.
    unsafe fn drop_first_element(&mut self) {
        if self.head.is_null() {
            return;
        }
        self.head = unsafe { (*self.head).drop_from_list() };
        if self.head.is_null() {
            self.tail = std::ptr::null_mut();
            self.size = 0;
        } else {
            self.size -= 1;
        }
    }

    /// Java package-private `lastElementEquals(Token)`.
    ///
    /// # Safety
    /// The list and `token` must be live.
    unsafe fn last_element_equals(&self, token: *mut Token) -> bool {
        if self.tail.is_null() {
            return token.is_null();
        }
        if token.is_null() {
            // Java's `Token.equals(Token)` dereferences its argument.
            panic!("java.lang.NullPointerException");
        }
        unsafe { (*self.tail).equals_token(&*token) }
    }

    /// Java package-private `dropLastElement()`.
    ///
    /// # Safety
    /// The list must be live.
    unsafe fn drop_last_element(&mut self) {
        if self.tail.is_null() {
            return;
        }
        self.tail = unsafe { (*self.tail).drop_from_list() };
        if self.tail.is_null() {
            self.head = std::ptr::null_mut();
            self.size = 0;
        } else {
            self.size -= 1;
        }
    }

    /// Java package-private `getHead()`.
    fn get_head(&self) -> *mut Token {
        self.head
    }

    /// Java `toString()`.
    ///
    /// # Safety
    /// The list must be live.
    unsafe fn to_string(&self) -> String {
        unsafe { (*self.get_head()).to_string() }
    }
}

/// Java package-private final `AutodocParser`.
pub struct AutodocParser {
    /// Java field `line`, a `Vector` of tokens.
    line: Vec<*mut Token>,
    /// Java field `peetVariant`.
    peet_variant: bool,
    /// Java field `tokenizer`, cleared to null by `parse()`.
    tokenizer: *mut AutodocTokenizer,
    /// Java field `logFile`, a `LogFile.Handle`.
    log_file: Option<std::sync::Arc<log_file::Handle>>,
    /// Java field `errMsg`, a caller-owned `StringBuilder`.
    err_msg: *mut String,
    /// Java field `tokenIndex`, initialised to 0.
    token_index: i32,
    /// Java field `autodoc`, initialised to null.
    autodoc: *mut dyn WriteOnlyStatementList,
    /// Java field `token`, initialised to null.
    token: *mut Token,
    /// Java field `prevToken`, initialised to null.
    prev_token: *mut Token,
    /// Java field `prevPrevToken`, initialised to null.
    prev_prev_token: *mut Token,
    /// Java field `parsed`, initialised to false.
    parsed: bool,
    // error flags
    /// Java field `errorsFound`, initialised to false.
    errors_found: bool,
    /// Java field `error`, initialised to false.
    error: bool,
    // testing flags
    /// Java field `test`, initialised to false.
    test: bool,
    /// Java field `detailedTest`, initialised to false.
    detailed_test: bool,
    /// Java field `testWithTokens`, initialised to false.
    test_with_tokens: bool,
    /// Java field `lastTokenType`, initialised to `Token.Type.NULL`.
    last_token_type: token::Type,
    /// Java field `testIndent`, initialised to -1.
    test_indent: i32,
    // Preprocessor flags
    /// Java field `delimiterInLine`, initialised to false.
    delimiter_in_line: bool,
    /// Java field `lineNum`, initialised to 0.
    line_num: i32,
    /// Java field `lastLinePrinted`, initialised to 0.
    last_line_printed: i32,
    // Postprocessor flags
    /// Java field `versionFound`, initialised to false.
    version_found: bool,
    /// Java field `pipFound`, initialised to false.
    pip_found: bool,
    /// Java field `versionRequired`, initialised to true.
    version_required: bool,
    /// Java field `debug`, initialised to false.
    debug: bool,
    /// Java field `debugLocal`, initialised to false.
    debug_local: bool,
}

impl AutodocParser {
    /// Java private `AutodocParser(Autodoc, boolean, boolean, boolean, Location, String,
    /// String, String, File, BaseManager, AxisID, String, boolean, boolean,
    /// StringBuilder)`.
    ///
    /// # Safety
    /// `autodoc` must be null or point to a live `Autodoc`, and `err_msg` null or to a
    /// live `String`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn new(
        autodoc: *mut Autodoc,
        allow_alt_comment: bool,
        version_required: bool,
        peet_variant: bool,
        location: Option<Location>,
        env_var: Option<&str>,
        subdir_name: Option<&str>,
        name: Option<&str>,
        autodoc_file: Option<&std::path::Path>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        debug: bool,
        writable: bool,
        err_msg: *mut String,
    ) -> AutodocParser {
        if autodoc.is_null() {
            panic!("java.lang.IllegalArgumentException: autodoc is null.");
        }
        let tokenizer = Box::into_raw(Box::new(AutodocTokenizer::new(
            allow_alt_comment,
            location,
            env_var,
            subdir_name,
            name,
            autodoc_file,
            manager,
            axis_id,
            not_found_message,
            debug,
            writable,
        )));
        let log_file = unsafe { (*tokenizer).get_log_file() };
        AutodocParser {
            line: Vec::new(),
            peet_variant,
            tokenizer,
            log_file,
            err_msg,
            token_index: 0,
            autodoc,
            token: std::ptr::null_mut(),
            prev_token: std::ptr::null_mut(),
            prev_prev_token: std::ptr::null_mut(),
            parsed: false,
            errors_found: false,
            error: false,
            test: false,
            detailed_test: false,
            test_with_tokens: false,
            last_token_type: token::Type::Null,
            test_indent: -1,
            delimiter_in_line: false,
            line_num: 0,
            last_line_printed: 0,
            version_found: false,
            pip_found: false,
            version_required,
            debug,
            debug_local: false,
        }
    }

    /// Java package-private static `getAutodocInstance`.
    ///
    /// # Safety
    /// See `new`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn get_autodoc_instance(
        autodoc: *mut Autodoc,
        allow_alt_comment: bool,
        version_required: bool,
        peet_variant: bool,
        name: Option<&str>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        debug: bool,
    ) -> AutodocParser {
        unsafe {
            AutodocParser::new(
                autodoc,
                allow_alt_comment,
                version_required,
                peet_variant,
                Some(Location::Autodoc),
                None,
                None,
                name,
                None,
                manager,
                axis_id,
                not_found_message,
                debug,
                false,
                std::ptr::null_mut(),
            )
        }
    }

    /// Java package-private static `getUnmanagedAutodocInstance`.
    ///
    /// # Safety
    /// See `new`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn get_unmanaged_autodoc_instance(
        autodoc: *mut Autodoc,
        allow_alt_comment: bool,
        version_required: bool,
        peet_variant: bool,
        name: Option<&str>,
        autodoc_file: Option<&std::path::Path>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
    ) -> AutodocParser {
        unsafe {
            AutodocParser::new(
                autodoc,
                allow_alt_comment,
                version_required,
                peet_variant,
                Some(Location::Autodoc),
                None,
                None,
                name,
                autodoc_file,
                manager,
                axis_id,
                not_found_message,
                false,
                false,
                std::ptr::null_mut(),
            )
        }
    }

    /// Java package-private static `getGenericInstance(Autodoc, boolean, boolean,
    /// boolean, String, String, String, BaseManager, AxisID, String, boolean, boolean)`.
    ///
    /// # Safety
    /// See `new`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn get_generic_instance_env_var(
        autodoc: *mut Autodoc,
        allow_alt_comment: bool,
        version_required: bool,
        peet_variant: bool,
        env_var: Option<&str>,
        subdir_name: Option<&str>,
        name: Option<&str>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        debug: bool,
        writable: bool,
    ) -> AutodocParser {
        unsafe {
            AutodocParser::new(
                autodoc,
                allow_alt_comment,
                version_required,
                peet_variant,
                None,
                env_var,
                subdir_name,
                name,
                None,
                manager,
                axis_id,
                not_found_message,
                debug,
                writable,
                std::ptr::null_mut(),
            )
        }
    }

    /// Java package-private static `getGenericInstance(Autodoc, boolean, boolean,
    /// boolean, File, BaseManager, AxisID, String, boolean, boolean, StringBuilder)`.
    ///
    /// # Safety
    /// See `new`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn get_generic_instance(
        autodoc: *mut Autodoc,
        allow_alt_comment: bool,
        version_required: bool,
        peet_variant: bool,
        autodoc_file: Option<&std::path::Path>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        debug: bool,
        writable: bool,
        err_msg: *mut String,
    ) -> AutodocParser {
        unsafe {
            AutodocParser::new(
                autodoc,
                allow_alt_comment,
                version_required,
                peet_variant,
                None,
                None,
                None,
                None,
                autodoc_file,
                manager,
                axis_id,
                not_found_message,
                debug,
                writable,
                err_msg,
            )
        }
    }

    /// Java package-private `initialize()`.
    pub fn initialize(&mut self) -> Result<(), LogFileError> {
        unsafe { (*self.tokenizer).initialize() }
    }

    /// Java package-private `isError()`.
    pub fn is_error(&self) -> bool {
        self.error
    }

    /// Java package-private `getFileName()`.
    pub fn get_file_name(&self) -> String {
        match &self.log_file {
            None => "".to_string(),
            Some(log_file) => log_file.get_name(),
        }
    }

    /// Java package-private `exists()`.
    pub fn exists(&self) -> bool {
        match &self.log_file {
            None => false,
            Some(log_file) => log_file.exists(),
        }
    }

    /// Java package-private `getAbsolutePath()`.
    pub fn get_absolute_path(&self) -> String {
        match &self.log_file {
            None => "".to_string(),
            Some(log_file) => log_file.get_absolute_path(),
        }
    }

    /// Java package-private `getLogFile()`.
    pub fn get_log_file(&self) -> Option<std::sync::Arc<log_file::Handle>> {
        self.log_file.clone()
    }

    /// Java package-private `parse()`.
    ///
    /// Parses an autodoc file.  This function can be run only once per instance of the
    /// object.
    ///
    /// # Safety
    /// The tokenizer and the autodoc must be live.
    pub unsafe fn parse(&mut self) {
        unsafe { self.autodoc_unit() };
        self.tokenizer = std::ptr::null_mut();
    }

    /// Java private `autodoc()`.
    ///
    /// `Autodoc => { emptyLine | comment | pair | section } EOF`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn autodoc_unit(&mut self) {
        if self.parsed {
            return;
        }
        self.parsed = true;
        unsafe { self.next_token() };
        let mut global_section = true;
        while !unsafe { (*self.token).is(token::Type::Eof) } {
            let autodoc = self.autodoc;
            if !unsafe { self.empty_line(autodoc) } && !unsafe { self.comment(autodoc) } {
                if global_section {
                    if unsafe { self.section(false) } {
                        global_section = false;
                    } else {
                        unsafe { self.pair(autodoc, true) };
                    }
                } else {
                    // Sections can contain empty lines. They only end when another
                    // section starts.  Once the sections start there are no more
                    // autodoc-level pairs.
                    unsafe { self.section(true) };
                }
            }
        }
    }

    /// Java private `emptyLine(WriteOnlyStatementList)`.
    ///
    /// `emptyLine => ^ -WHITESPACE- ( EOL | EOF )`
    ///
    /// Eats up an empty line, starting from the beginning of the line.  Writes the
    /// empty line to the autodoc, section, or subsection is in.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn empty_line(&mut self, list: *mut dyn WriteOnlyStatementList) -> bool {
        if !self.is_beginning_of_line() {
            // empty lines must start at the beginning of the line
            return false;
        }
        // Eat up the whitespace. This helps identify the empty line. But if its not
        // an empty line, it gets rid of the white space at the beginning of the line
        while !unsafe { self.match_token(token::Type::Whitespace) }.is_null() {}
        if unsafe { (*self.token).is(token::Type::Eol) } {
            self.test_start_function("emptyline");
            // found empty line
            unsafe { (*list).add_empty_line(self.line_num) };
            unsafe { self.next_token() };
            self.test_end_function("emptyline", true);
            return true;
        }
        false
    }

    /// Java private `comment(WriteOnlyStatementList)`.
    ///
    /// `comment => COMMENT { \EOL\ } ( EOL | EOF )`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn comment(&mut self, list: *mut dyn WriteOnlyStatementList) -> bool {
        if unsafe { self.match_token(token::Type::Comment) }.is_null() {
            // not a comment
            return false;
        }
        self.test_start_function("comment");
        // comments can be made of multiple tokens, so use a link list. A comment entry
        // should not contain the EOL or EOF
        let mut comment_link_list: Option<LinkList> = None;
        if !unsafe { (*self.token).is(token::Type::Eol) }
            && !unsafe { (*self.token).is(token::Type::Eof) }
        {
            // Non-empty comment
            comment_link_list = Some(LinkList::new(self.token));
        }
        // Add comment
        match &comment_link_list {
            Some(comment_link_list) => {
                // Add non-empty comment
                unsafe { (*list).add_comment(comment_link_list.get_head(), self.line_num) };
            }
            None => {
                // Add empty comment
                unsafe { (*list).add_comment(std::ptr::null_mut(), self.line_num) };
            }
        }
        // add tokens to the non-empty comment
        while !unsafe { (*self.token).is(token::Type::Eol) }
            && !unsafe { (*self.token).is(token::Type::Eof) }
        {
            unsafe { comment_link_list.as_mut().unwrap().append(self.token) };
            unsafe { self.next_token() };
        }
        // eat up the EOL
        unsafe { self.match_token(token::Type::Eol) };
        self.test_end_function("comment", true);
        true
    }

    /// Java private `section(boolean)`.
    ///
    /// `section => OPEN sectionHeader CLOSE -WHITESPACE- ( EOL | EOF )
    /// { emptyLine | comment | pair | subsection }`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn section(&mut self, required: bool) -> bool {
        if unsafe { self.match_token(token::Type::Open) }.is_null() {
            // not a section
            if required {
                unsafe {
                    self.report_error(Some(&format!(
                        "Unknown statement.  Expecting a section (missing '{}').",
                        token::Type::Open.get_descr()
                    )))
                };
            }
            return false;
        }
        self.test_start_function("section");
        // save the new section in the autodoc
        #[allow(unused_assignments)]
        let mut section: *mut Section = std::ptr::null_mut();
        // sectionHeader saves the section
        section = unsafe { self.section_header(self.autodoc) };
        if section.is_null() {
            // bad section
            self.test_end_function("section", false);
            return false;
        }
        if unsafe { self.match_token(token::Type::Close) }.is_null() {
            // bad section
            unsafe {
                self.report_error(Some(&format!(
                    "A section header must end with '{}'.",
                    token::Type::Close.get_descr()
                )))
            };
            self.test_end_function("section", false);
            return false;
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        if unsafe { self.match_token(token::Type::Eol) }.is_null()
            && unsafe { self.match_token(token::Type::Eof) }.is_null()
        {
            // bad section
            unsafe {
                self.report_error(Some(&format!(
                    "A section header must end with '{}'.",
                    token::Type::Close.get_descr()
                )))
            };
            self.test_end_function("section", false);
            return false;
        }
        while !unsafe { (*self.token).is(token::Type::Eof) } {
            // look for elements in the section
            if !unsafe { self.empty_line(section) }
                && !unsafe { self.comment(section) }
                && !unsafe { self.pair(section, false) }
                && !unsafe { self.subsection(section) }
            {
                // end of section
                self.test_end_function("section", true);
                return true;
            }
        }
        // empty section
        self.test_end_function("section", true);
        true
    }

    /// Java private `subsection(Section)`.
    ///
    /// `subsection => SUBOPEN sectionHeader SUBCLOSE ( EOL | EOF )
    /// { emptyLine | comment | pair } subsectionClose`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn subsection(&mut self, section: *mut Section) -> bool {
        // use look ahead because this could be a section
        if !unsafe { (*self.token).is(token::Type::Subopen) } {
            // not a subsection
            return false;
        }
        self.test_start_function("subsection");
        // its a subsection so eat up SUBOPEN
        unsafe { self.next_token() };
        // save the new subsection in the section
        #[allow(unused_assignments)]
        let mut subsection: *mut Section = std::ptr::null_mut();
        subsection = unsafe { self.section_header(section) };
        if subsection.is_null() {
            // bad subsection
            self.test_end_function("subsection", false);
            return false;
        }
        if unsafe { self.match_token(token::Type::Subclose) }.is_null() {
            // bad subsection
            unsafe {
                self.report_error(Some(&format!(
                    "A subsection header must end with '{}'.",
                    token::Type::Subclose.get_descr()
                )))
            };
            self.test_end_function("subsection", false);
            return false;
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        if unsafe { self.match_token(token::Type::Eol) }.is_null()
            && unsafe { self.match_token(token::Type::Eof) }.is_null()
        {
            // bad section
            unsafe {
                self.report_error(Some(&format!(
                    "A section header must end with '{}'.",
                    token::Type::Subclose.get_descr()
                )))
            };
            self.test_end_function("section", false);
            return false;
        }
        let mut done = false;
        let mut closed = false;
        while !done {
            // look for elements in the subsection
            if !unsafe { self.empty_line(subsection) }
                && !unsafe { self.comment(subsection) }
                && !unsafe { self.pair(subsection, false) }
            {
                closed = unsafe { self.subsection_close(subsection) };
                done = true;
            }
        }
        // end subsection
        self.test_end_function("subsection", closed);
        closed
    }

    /// Java private `subsectionClose(WriteOnlyStatementList)`.
    ///
    /// `subsectionClose -> SUBOPEN -WHITESPACE- SUBCLOSE (EOL | EOF )`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn subsection_close(&mut self, _subsection: *mut dyn WriteOnlyStatementList) -> bool {
        if !unsafe { (*self.token).is(token::Type::Subopen) } {
            // not a subsectionClose
            unsafe {
                self.report_error(Some(&format!(
                    "The subsection must be closed - missing '{}{}'.",
                    token::Type::Subopen.get_descr(),
                    token::Type::Subclose.get_descr()
                )))
            };
            return false;
        }
        self.test_start_function("subsectionClose");
        // its a subsectionClose so eat up SUBOPEN
        unsafe { self.next_token() };
        unsafe { self.match_token(token::Type::Whitespace) };
        // eat up SUBCLOSE
        if unsafe { self.match_token(token::Type::Subclose) }.is_null() {
            // bad subsection
            unsafe {
                self.report_error(Some(&format!(
                    "The subsection must be closed - '{}{}'.",
                    token::Type::Subopen.get_descr(),
                    token::Type::Subclose.get_descr()
                )))
            };
            self.test_end_function("subsectionClose", false);
            return false;
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        if unsafe { self.match_token(token::Type::Eol) }.is_null()
            && unsafe { self.match_token(token::Type::Eof) }.is_null()
        {
            // bad subsection
            unsafe {
                self.report_error(Some(&format!(
                    "A subsection close must end with '{}'.",
                    token::Type::Subclose.get_descr()
                )))
            };
            self.test_end_function("section", false);
            return false;
        }
        self.test_end_function("subsectionClose", true);
        true
    }

    /// Java private `matchToken(Token.Type)`.
    ///
    /// calls nextToken() and returns the matched token, if token is tokenType.
    /// If token is not tokenType, returns null and does not call nextToken().
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn match_token(&mut self, token_type: token::Type) -> *mut Token {
        if unsafe { (*self.token).is(token_type) } {
            let matched_token = self.token;
            unsafe { self.next_token() };
            return matched_token;
        }
        std::ptr::null_mut()
    }

    /// Java private `sectionHeader(WriteOnlyStatementList)`.
    ///
    /// `sectionHeader => sectionType -WHITESPACE- DELIMITER -WHITESPACE- sectionName
    /// -WHITESPACE-`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn section_header(
        &mut self,
        name_value_pair_list: *mut dyn WriteOnlyStatementList,
    ) -> *mut Section {
        self.test_start_function("sectionHeader");
        unsafe { self.match_token(token::Type::Whitespace) };
        // get the section type
        let r#type = unsafe { self.section_type() };
        if r#type.is_null() {
            // bad section header
            self.test_end_function("sectionHeader", false);
            return std::ptr::null_mut();
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        if unsafe { self.match_token(token::Type::Delimiter) }.is_null() {
            // bad section header
            unsafe {
                let delimiter = (*self.tokenizer).get_delimiter_string();
                self.report_error(Some(&format!(
                    "A section header must contain a delimiter - missing '{}'.",
                    delimiter
                )))
            };
            self.test_end_function("sectionHeader", false);
            return std::ptr::null_mut();
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        // get the section name
        let name_link_list = unsafe { self.section_name() };
        // Java dereferences the returned link list unconditionally; `sectionName`
        // returns null on the same condition it tests here, so the source throws a
        // `NullPointerException` rather than reporting a bad section header.
        let name_link_list = match name_link_list {
            None => panic!("java.lang.NullPointerException"),
            Some(name_link_list) => name_link_list,
        };
        if name_link_list.size() == 0 {
            // bad section header
            self.test_end_function("sectionHeader", false);
            return std::ptr::null_mut();
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        self.test_end_function("sectionHeader", true);
        // assume that this is a good section and save it now
        unsafe {
            (*name_value_pair_list).add_section(r#type, name_link_list.get_head(), self.line_num)
        }
    }

    /// Java private `sectionType()`.
    ///
    /// `sectionType => ( WORD | KEYWORD )`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn section_type(&mut self) -> *mut Token {
        self.test_start_function("sectionType");
        if !unsafe { self.match_token(token::Type::Word) }.is_null()
            || !unsafe { self.match_token(token::Type::Keyword) }.is_null()
        {
            // return the section type
            self.test_end_function("sectionType", true);
            return self.prev_token;
        }
        // did not find section type
        unsafe {
            let delimiter = (*self.tokenizer).get_delimiter_string();
            self.report_error(Some(&format!(
                "Missing section type (right side of the '{}' missing).",
                delimiter
            )))
        };
        self.test_end_function("sectionType", false);
        std::ptr::null_mut()
    }

    /// Java private `sectionName()`.
    ///
    /// `sectionName => [ \CLOSE & SUBCLOSE & WHITESPACE & EOL & EOF\ ]`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn section_name(&mut self) -> Option<LinkList> {
        self.test_start_function("sectionName");
        // section name may contain multiple tokens
        let mut name_link_list = LinkList::new(self.token);
        // link the section name together
        while !unsafe { (*self.token).is(token::Type::Close) }
            && !unsafe { (*self.token).is(token::Type::Subclose) }
            && !unsafe { (*self.token).is(token::Type::Whitespace) }
            && !unsafe { (*self.token).is(token::Type::Eol) }
            && !unsafe { (*self.token).is(token::Type::Eof) }
        {
            unsafe { name_link_list.append(self.token) };
            unsafe { self.next_token() };
        }
        if name_link_list.size() == 0 {
            // bad section name
            unsafe {
                let delimiter = (*self.tokenizer).get_delimiter_string();
                self.report_error(Some(&format!(
                    "Missing section name (left side of the '{}' missing).",
                    delimiter
                )))
            };
            self.test_end_function("sectionName", false);
            return None;
        }
        self.test_end_function("sectionName", true);
        Some(name_link_list)
    }

    /// Java private `pair(WriteOnlyStatementList, boolean)`.
    ///
    /// `pair => name -WHITESPACE- DELIMITER -WHITESPACE- -QUOTE#- value`
    ///
    /// PEET variant: `pair => name -WHITESPACE- DELIMITER -WHITESPACE- value`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn pair(&mut self, list: *mut dyn WriteOnlyStatementList, required: bool) -> bool {
        if !unsafe { (*self.token).is(token::Type::Word) }
            && !unsafe { (*self.token).is(token::Type::Keyword) }
            && !unsafe { (*self.token).is(token::Type::Quote) }
        {
            if required {
                unsafe {
                    let delimiter = (*self.tokenizer).get_delimiter_string();
                    self.report_error(Some(&format!(
                        "Unknown statement.  Expecting a name/value pair (name {} value).",
                        delimiter
                    )))
                };
            }
            return false;
        }
        self.test_start_function("pair");
        let pair = unsafe { (*list).add_name_value_pair(self.line_num) };
        let leaf_attribute = unsafe { self.name(list, pair) };
        if leaf_attribute.is_null() {
            // bad pair
            self.test_end_function("pair", false);
            return false;
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        if unsafe { self.match_token(token::Type::Delimiter) }.is_null() {
            // bad pair
            unsafe {
                let delimiter = (*self.tokenizer).get_delimiter_string();
                self.report_error(Some(&format!(
                    "Missing '{}'.  Invalid name/value pair (name {} value).",
                    delimiter, delimiter
                )))
            };
            self.test_end_function("pair", false);
            return false;
        }
        unsafe { self.match_token(token::Type::Whitespace) };
        // attach the value to the last attribute
        let close_quote = if self.peet_variant {
            std::ptr::null_mut()
        } else {
            unsafe { self.match_token(token::Type::Quote) }
        };
        unsafe { self.value(list, leaf_attribute, pair, close_quote) };
        self.test_end_function("pair", true);
        true
    }

    /// Java private `name(WriteOnlyStatementList, NameValuePair)`.
    ///
    /// `name => base-attribute { SEPARATOR attribute }`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn name(
        &mut self,
        list: *mut dyn WriteOnlyStatementList,
        pair: *mut NameValuePair,
    ) -> *mut Attribute {
        let function = "name";
        self.test_start_function(function);
        let mut attribute = unsafe { self.base_attribute(list, pair) };
        if attribute.is_null() {
            // bad name
            self.test_end_function(function, false);
            return std::ptr::null_mut();
        }
        while !unsafe { self.match_token(token::Type::Separator) }.is_null() {
            // add the attribute to the map, point to the child attribute
            attribute = unsafe { self.attribute(attribute, pair) };
            if attribute.is_null() {
                // bad name
                self.test_end_function(function, false);
                return std::ptr::null_mut();
            }
        }
        self.test_end_function(function, true);
        attribute
    }

    /// Java private `baseAttribute(WriteOnlyAttributeList, NameValuePair)`.
    ///
    /// `base-attribute => ( WORD | KEYWORD | QUOTE ) -attribute-`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn base_attribute(
        &mut self,
        attribute_list: *mut dyn WriteOnlyAttributeList,
        pair: *mut NameValuePair,
    ) -> *mut Attribute {
        let attribute = unsafe { self.build_attribute(attribute_list, pair, true) };
        if attribute.is_null() {
            unsafe { self.report_error(Some("Missing attribute.")) };
        }
        attribute
    }

    /// Java private `attribute(WriteOnlyAttributeList, NameValuePair)`.
    ///
    /// `attribute => [ WORD | KEYWORD | COMMENT | QUOTE ]`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn attribute(
        &mut self,
        attribute_list: *mut dyn WriteOnlyAttributeList,
        pair: *mut NameValuePair,
    ) -> *mut Attribute {
        let attribute = unsafe { self.build_attribute(attribute_list, pair, false) };
        if attribute.is_null() {
            // bad name
            unsafe {
                self.report_error(Some(&format!(
                    "Another attribute must follow the '{}' (attribute{}attribute{}attribute...).",
                    token::Type::Separator.get_descr(),
                    token::Type::Separator.get_descr(),
                    token::Type::Separator.get_descr()
                )))
            };
        }
        attribute
    }

    /// Java private `buildAttribute(WriteOnlyAttributeList, NameValuePair, boolean)`.
    ///
    /// Builds base-attribute and attribute.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn build_attribute(
        &mut self,
        attribute_list: *mut dyn WriteOnlyAttributeList,
        pair: *mut NameValuePair,
        base: bool,
    ) -> *mut Attribute {
        let function: &str = if base { "base-attribute" } else { "attribute" };
        if base {
            self.test_start_function(function);
        }
        if !unsafe { (*self.token).is(token::Type::Word) }
            && !unsafe { (*self.token).is(token::Type::Keyword) }
            && !unsafe { (*self.token).is(token::Type::Quote) }
            && (base || !unsafe { (*self.token).is(token::Type::Comment) })
        {
            if base {
                self.test_end_function(function, false);
            }
            return std::ptr::null_mut();
        }
        if !base {
            self.test_start_function(function);
        }
        let mut value_link_list = LinkList::new(self.token);
        unsafe { value_link_list.append(self.token) };
        unsafe { self.next_token() };
        while unsafe { (*self.token).is(token::Type::Word) }
            || unsafe { (*self.token).is(token::Type::Keyword) }
            || unsafe { (*self.token).is(token::Type::Quote) }
            || unsafe { (*self.token).is(token::Type::Comment) }
        {
            unsafe { value_link_list.append(self.token) };
            unsafe { self.next_token() };
        }
        self.test_end_function(function, true);
        // add and return the new attribute
        let attribute =
            unsafe { (*attribute_list).add_attribute(value_link_list.get_head(), self.line_num) };
        unsafe { (*pair).add_attribute(attribute) };
        attribute
    }

    /// Java private `value(WriteOnlyStatementList, Attribute, NameValuePair, Token)`.
    ///
    /// `value => { \EOL & EOF\ } ( EOL | EOF ) { (!quoted & valueline ) | ( quoted &
    /// ( quotedValueLine | #QUOTE ) ) }`
    ///
    /// sets the value in the attribute, if the value exists
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn value(
        &mut self,
        parent: *mut dyn WriteOnlyStatementList,
        attribute: *mut Attribute,
        pair: *mut NameValuePair,
        close_quote: *mut Token,
    ) {
        self.test_start_function("value");
        // values can be made of multiple tokens, so use a link list
        let mut value_link_list = LinkList::new(self.token);
        let mut found_close_quote = false;
        while !unsafe { (*self.token).is(token::Type::Eol) }
            && !unsafe { (*self.token).is(token::Type::Eof) }
        {
            // add the token to the value link list
            unsafe { value_link_list.append(self.token) };
            // look for the close quote
            if !close_quote.is_null() {
                if unsafe { (*self.token).equals_token(&*close_quote) } {
                    found_close_quote = true;
                }
                // check the tokens that follow a possible close quote
                else if found_close_quote {
                    // ignore embedded quotes
                    if !unsafe { (*self.token).is(token::Type::Whitespace) } {
                        found_close_quote = false;
                    }
                }
            }
            unsafe { self.next_token() };
        }
        // check for keywords
        unsafe { self.process_meta_data(attribute) };
        // Finished the first line of the value (excluding the EOL) if this is
        // delimiter reassignment, it must be set now, or the following pair will be
        // mistaken for part of this value.
        if unsafe { self.is_delimiter_change(attribute) } {
            let values = unsafe { (*value_link_list.get_head()).get_values() };
            unsafe { (*self.tokenizer).set_delimiter_string(Some(&values)) };
            unsafe { (*pair).set_delimiter_change(value_link_list.get_head()) };
        }
        // grab the EOL in case the value continues in the value line
        if unsafe { (*self.token).is(token::Type::Eol) } {
            unsafe { value_link_list.append(self.token) };
        }
        unsafe { self.next_token() };
        if !found_close_quote {
            if close_quote.is_null() {
                while !self.error && unsafe { self.value_line(parent, &mut value_link_list) } {}
            } else {
                while !self.error
                    && unsafe { self.quoted_value_line(parent, &mut value_link_list, close_quote) }
                {
                }
            }
        }
        // Strip non-embedded EOL, EOF, and WHITESPACE at the start and end of the value
        while value_link_list.size() > 0
            && (unsafe { value_link_list.is_first_element(token::Type::Whitespace) }
                || unsafe { value_link_list.is_first_element(token::Type::Eol) }
                || unsafe { value_link_list.is_first_element(token::Type::Eof) })
        {
            unsafe { value_link_list.drop_first_element() };
        }
        while value_link_list.size() > 0
            && (unsafe { value_link_list.is_last_element(token::Type::Whitespace) }
                || unsafe { value_link_list.is_last_element(token::Type::Eol) }
                || unsafe { value_link_list.is_last_element(token::Type::Eof) })
        {
            unsafe { value_link_list.drop_last_element() };
        }
        // Remove the closing quote
        if !close_quote.is_null() && unsafe { value_link_list.last_element_equals(close_quote) } {
            unsafe { value_link_list.drop_last_element() };
        }
        // Save the value to the name/value pair even if it doesn't exist. This is
        // how the name/value pair knows that its name is complete and can assign itself
        // to the last attribute
        unsafe { (*pair).add_value(value_link_list.get_head()) };
        self.test_end_function("value", true);
    }

    /// Java private `valueLine(WriteOnlyStatementList, LinkList)`.
    ///
    /// `valueLine =>  !emptyLine !DelimiterInLine !comment { \DELIMITER & SUBOPEN & EOL
    /// & EOF\ } ( EOL | EOF )`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn value_line(
        &mut self,
        parent: *mut dyn WriteOnlyStatementList,
        value_link_list: &mut LinkList,
    ) -> bool {
        if unsafe { self.empty_line(parent) }
            || self.delimiter_in_line
            || unsafe { (*self.token).is(token::Type::Comment) }
            || unsafe { (*self.token).is(token::Type::Eol) }
            || unsafe { (*self.token).is(token::Type::Eof) }
            || unsafe { (*self.token).is(token::Type::Subopen) }
        {
            // not a value line
            return false;
        }
        self.test_start_function("valueLine");
        while !unsafe { (*self.token).is(token::Type::Delimiter) }
            && !unsafe { (*self.token).is(token::Type::Eol) }
            && !unsafe { (*self.token).is(token::Type::Eof) }
        {
            // add the token to the value link list
            unsafe { value_link_list.append(self.token) };
            unsafe { self.next_token() };
        }
        if unsafe { (*self.token).is(token::Type::Delimiter) } {
            // really bad error - preprocessor is wrong
            // bad value line
            unsafe {
                let delimiter = (*self.tokenizer).get_delimiter_string();
                self.report_error(Some(&format!(
                    "A value line cannot contain the delimiter string (\"{}\").",
                    delimiter
                )))
            };
            self.test_end_function("valueLine", false);
            return false;
        }
        // grab the EOL in case another value line follows
        if unsafe { (*self.token).is(token::Type::Eol) } {
            unsafe { value_link_list.append(self.token) };
        }
        unsafe { self.next_token() };
        self.test_end_function("valueLine", true);
        true
    }

    /// Java private `quotedValueLine(WriteOnlyStatementList, LinkList, Token)`.
    ///
    /// `quotedValueLine => !emptyLine { \EOL & EOF\ } #QUOTE -WHITESPACE- ( EOL | EOF )`
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn quoted_value_line(
        &mut self,
        parent: *mut dyn WriteOnlyStatementList,
        value_link_list: &mut LinkList,
        close_quote: *mut Token,
    ) -> bool {
        if value_link_list.is_done() {
            // quoted value was completed in the previous line
            return false;
        }
        if unsafe { self.empty_line(parent) } || unsafe { (*self.token).is(token::Type::Eol) } {
            // illegal empty line inside a quoted value
            unsafe {
                let quote = String::from_utf16_lossy(&[(*close_quote).get_char()]);
                self.report_error(Some(&format!(
                    "An empty line cannot be embedded in a quoted value (\"{}\").",
                    quote
                )))
            };
            return false;
        }
        if unsafe { (*self.token).is(token::Type::Eof) } {
            // incomplete quoted value
            unsafe {
                let quote = String::from_utf16_lossy(&[(*close_quote).get_char()]);
                self.report_error(Some(&format!("Close quote not found (\"{}\").", quote)))
            };
            return false;
        }
        self.test_start_function("valueLine");
        let mut found_close_quote = false;
        while !unsafe { (*self.token).is(token::Type::Eol) }
            && !unsafe { (*self.token).is(token::Type::Eof) }
        {
            // add the token to the value link list
            unsafe { value_link_list.append(self.token) };
            // look for the close quote
            if !close_quote.is_null() {
                if unsafe { (*self.token).equals_token(&*close_quote) } {
                    found_close_quote = true;
                }
                // check the tokens that follow a possible close quote
                else if found_close_quote {
                    // ignore embedded quotes
                    if !unsafe { (*self.token).is(token::Type::Whitespace) } {
                        found_close_quote = false;
                    }
                }
            }
            unsafe { self.next_token() };
        }
        // grab the EOL in case another value line follows
        if unsafe { (*self.token).is(token::Type::Eol) } {
            unsafe { value_link_list.append(self.token) };
        }
        if found_close_quote {
            // close quote has been found - added a done flag to the value link list
            value_link_list.set_done();
        } else if unsafe { (*self.token).is(token::Type::Eof) } {
            // incomplete quoted value
            unsafe {
                let quote = String::from_utf16_lossy(&[(*close_quote).get_char()]);
                self.report_error(Some(&format!("Close quote not found (\"{}\").", quote)))
            };
            self.test_end_function("valueLine", false);
            return false;
        }
        unsafe { self.next_token() };
        self.test_end_function("valueLine", true);
        true
    }

    /// Java private `reportError(String)`.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn report_error(&mut self, message: Option<&str>) {
        let message = match message {
            None => "Unknown error.",
            Some(message) => message,
        };
        self.error = true;
        let error_message = format!("Line# {}: {}", self.line_num, message);
        let error_index = self.token_index - 1;
        let size = self.line.len() as i32;
        let mut token: *mut Token;
        let mut value: String;
        let mut error_position: i32 = 0;
        let mut index: i32;
        let mut print_line = String::new();
        let mut carat = String::new();
        index = 0;
        while index <= error_index && index < size - 1 {
            token = self.line[index as usize];
            value = unsafe {
                match (*token).get_value() {
                    None => "null".to_string(),
                    Some(value) => value.to_string(),
                }
            };
            if index < error_index {
                error_position += value.encode_utf16().count() as i32;
            }
            print_line.push_str(&value);
            index += 1;
        }
        index = error_index + 1;
        while index < size - 1 {
            token = self.line[index as usize];
            print_line.push_str(&unsafe {
                match (*token).get_value() {
                    None => "null".to_string(),
                    Some(value) => value.to_string(),
                }
            });
            index += 1;
        }
        index = 0;
        while index < error_position {
            carat.push(' ');
            index += 1;
        }
        carat.push('^');
        if !self.errors_found {
            self.errors_found = true;
            if *DEBUG {
                // `logFile` is permanently null; the source's other arm prints
                // "Errors in " + logFile.getName().
                eprintln!("Errors found");
                eprintln!();
            }
        }
        if *DEBUG {
            eprintln!("{}", error_message);
            eprintln!("{}", print_line);
            eprintln!("{}", carat);
            eprintln!();
            if !self.err_msg.is_null() {
                unsafe {
                    (*self.err_msg).push_str(&format!("{}\n{}\n\n", error_message, print_line))
                };
            }
        }
        if self.detailed_test {
            panic!("java.lang.IllegalStateException");
        }
        self.token_index = self.line.len() as i32;
        unsafe { self.next_token() };
    }

    /// Java private `nextToken()`.
    ///
    /// get the next token; set prevToken and prevPrevToken
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn next_token(&mut self) {
        if !self.token.is_null() && unsafe { (*self.token).is(token::Type::Eof) } {
            // It may try nextToken() a few times at the end of file before it figures
            // out that its done, but that's OK
            return;
        }
        // The source's `line == null` guard cannot fail; the field is final.
        if self.token_index == self.line.len() as i32 {
            unsafe { self.preprocess() };
            self.token_index = 0;
        }
        self.prev_prev_token = self.prev_token;
        self.prev_token = self.token;
        self.token = self.line[self.token_index as usize];
        self.token_index += 1;
        if self.detailed_test && *DEBUG {
            eprintln!(
                "{}:{},delimiterInLine={}:{}",
                self.token_index,
                unsafe { (*self.token).to_string() },
                self.delimiter_in_line,
                unsafe { (*self.tokenizer).get_delimiter_string() }
            );
        }
    }

    /// Java package-private `isBeginningOfLine()`.
    pub fn is_beginning_of_line(&self) -> bool {
        self.token_index == 1
    }

    /// Java private `preprocess()`.
    ///
    /// Set line level preprocessor flag delimiterInLine.  Also set the line number.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn preprocess(&mut self) {
        let mut token: *mut Token;
        self.line_num += 1;
        self.line.clear();
        self.delimiter_in_line = false;
        loop {
            token = unsafe { (*self.tokenizer).next() };
            if unsafe { (*token).is(token::Type::Delimiter) } {
                self.delimiter_in_line = true;
            }
            self.line.push(token);
            if unsafe { (*token).is(token::Type::Eol) } || unsafe { (*token).is(token::Type::Eof) }
            {
                break;
            }
        }
    }

    // postprocessor

    /// Java private `processMetaData(Attribute)`.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn process_meta_data(&mut self, attribute: *mut Attribute) {
        if !unsafe { (*attribute).is_base() } {
            return;
        }
        let name = unsafe { (*attribute).get_name_token() };
        if unsafe { (*attribute).is_global() } {
            if unsafe {
                (*name).equals_type_and_string(
                    token::Type::Keyword,
                    Some(autodoc_tokenizer::VERSION_KEYWORD),
                )
            } {
                self.version_found = true;
                return;
            }
            if unsafe {
                (*name).equals_type_and_string(
                    token::Type::Keyword,
                    Some(autodoc_tokenizer::PIP_KEYWORD),
                )
            } {
                self.pip_found = true;
            }
        }
    }

    /// Java private `isDelimiterChange(Attribute)`.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn is_delimiter_change(&self, attribute: *mut Attribute) -> bool {
        unsafe {
            (*(*attribute).get_name_token()).equals_type_and_string(
                token::Type::Keyword,
                Some(autodoc_tokenizer::DELIMITER_KEYWORD),
            )
        }
    }

    /// Java package-private `testStreamTokenizer(boolean, boolean)`.
    pub fn test_stream_tokenizer(&mut self, tokens: bool, details: bool) {
        unsafe { (*self.tokenizer).test_stream_tokenizer(tokens, details) };
    }

    /// Java package-private `testPrimativeTokenizer(boolean)`.
    pub fn test_primative_tokenizer(&mut self, tokens: bool) {
        unsafe { (*self.tokenizer).test_primative_tokenizer(tokens) };
    }

    /// Java package-private `testAutodocTokenizer(boolean)`.
    pub fn test_autodoc_tokenizer(&mut self, tokens: bool) {
        unsafe { (*self.tokenizer).test(tokens) };
    }

    /// Java package-private `testPreprocessor(boolean)`.
    pub fn test_preprocessor(&mut self, tokens: bool) {
        if tokens {
            eprintln!("(type,value):delimiterInLine");
        }
        unsafe { (*self.tokenizer).initialize() };
        loop {
            unsafe { self.next_token() };
            if tokens {
                eprintln!(
                    "{}:{}",
                    unsafe { (*self.token).to_string() },
                    self.delimiter_in_line
                );
            } else if unsafe { (*self.token).is(token::Type::Eol) } {
                eprintln!();
            } else if !unsafe { (*self.token).is(token::Type::Eof) } {
                eprint!("{}", unsafe {
                    match (*self.token).get_value() {
                        None => "null".to_string(),
                        Some(value) => value.to_string(),
                    }
                });
            }
            if unsafe { (*self.token).is(token::Type::Eof) } {
                break;
            }
        }
    }

    /// Java package-private `test(boolean, boolean)`.
    ///
    /// `tokens`: display tokens rather then text.
    /// `details`: display more information and throw an exception as the first error.
    pub fn test(&mut self, tokens: bool, details: bool) {
        self.detailed_test = details;
        self.test = true;
        self.test_with_tokens = tokens;
        self.initialize();
        unsafe { self.parse() };
    }

    /// Java private `testStartFunction(String)`.
    fn test_start_function(&mut self, function_name: &str) {
        if self.test {
            self.test_indent += 1;
            self.print_test_indent();
            eprintln!("{} {{", function_name);
        }
    }

    /// Java private `testEndFunction(String, boolean)`.
    fn test_end_function(&mut self, function_name: &str, result: bool) -> bool {
        if self.test {
            self.print_test_indent();
            if result {
                eprintln!("{} }} succeeded", function_name);
            } else {
                eprintln!("{} }} failed", function_name);
            }
            self.test_indent -= 1;
        }
        result
    }

    /// Java private `printTestLine()`.
    ///
    /// # Safety
    /// See `parse`.
    unsafe fn print_test_line(&mut self) {
        if self.last_line_printed == self.line_num {
            return;
        }
        self.last_line_printed = self.line_num;
        let mut token: *mut Token;
        let size = self.line.len();
        self.print_test_indent();
        eprint!("Line# {}: ", self.line_num);
        for index in 0..size {
            token = self.line[index];
            if self.test_with_tokens {
                eprint!("{}", unsafe { (*token).to_string() });
            } else if !unsafe { (*token).is(token::Type::Eol) }
                && !unsafe { (*token).is(token::Type::Eof) }
            {
                eprint!("{}", unsafe {
                    match (*token).get_value() {
                        None => "null".to_string(),
                        Some(value) => value.to_string(),
                    }
                });
            }
        }
        eprintln!();
    }

    /// Java private `printTestIndent()`.
    fn print_test_indent(&self) {
        for _i in 0..self.test_indent {
            eprint!("\t");
        }
    }
}

/// Keeps `ReadOnlyAttribute` in scope, the interface the attributes this parser builds
/// are read through.
const _: Option<&dyn ReadOnlyAttribute> = None;
