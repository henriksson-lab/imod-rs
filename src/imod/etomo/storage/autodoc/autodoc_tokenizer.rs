//! `IMOD/Etomo/src/etomo/storage/autodoc/AutodocTokenizer.java`.
//!
//! Description:
//! Creates the tokens required for autodoc functionality.  It can recognize the
//! following tokens:  EOF, EOL, WHITESPACE, COMMENT, SEPARATOR, OPEN, CLOSE,
//! DELIMITER, WORD, and KEYWORD.  It is not case sensitive, but it does preserve
//! original case and whitespace.
//!
//! To Use:
//! construct with a file.
//! call initialize().
//! call next() to get the next token, until the end of file is reached.
//!
//! Current token characters and strings:
//! COMMENT:  #
//! ALT_COMMENT:  %
//! SEPARATOR:  .
//! OPEN:  \[
//! CLOSE:  \]
//! QUOTE: " or ' or `
//! Default DELIMITER:  =
//! Keywords:  Version, Pip, KeyValueDelimiter
//!
//! The constructor is the package's file boundary: both `PrimativeTokenizer` factories
//! it calls open the autodoc as a `LogFile.Handle`.  `BaseManager` has no module; every
//! caller that reaches here passes a null one, so the parameter is typed
//! `Option<Infallible>`.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::{self, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::token::{self, Token};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

// special characters
/// Java `COMMENT_CHAR`.
pub const COMMENT_CHAR: char = '#';
/// Java `ALT_COMMENT_CHAR`.
pub const ALT_COMMENT_CHAR: char = '%';
/// Java `SEPARATOR_CHAR`, a `String`.
pub const SEPARATOR_CHAR: &str = ".";
/// Java `OPEN_CHAR`, a `Character`.
pub const OPEN_CHAR: char = '[';
/// Java `CLOSE_CHAR`, a `Character`.
pub const CLOSE_CHAR: char = ']';
/// Java `DEFAULT_DELIMITER`.
pub const DEFAULT_DELIMITER: &str = "=";
// keywords - keywords may not contain special characters
/// Java package-private `VERSION_KEYWORD`.
pub const VERSION_KEYWORD: &str = "Version";
/// Java package-private `PIP_KEYWORD`.
pub const PIP_KEYWORD: &str = "Pip";
/// Java package-private `DELIMITER_KEYWORD`.
pub const DELIMITER_KEYWORD: &str = "KeyValueDelimiter";
/// Java private `QUOTE_LIST`.
const QUOTE_LIST: [u16; 3] = ['"' as u16, '\'' as u16, '`' as u16];

/// Java's nested `static final class Location`, which has one constant and is compared
/// with `==`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Location {
    /// Java `Location.AUTODOC`.
    Autodoc,
}

/// Java public final `AutodocTokenizer`.
pub struct AutodocTokenizer {
    /// Java field `allowAltComment`.
    allow_alt_comment: bool,
    /// Java field `restrictedSymbols`.
    restricted_symbols: String,
    /// Java field `delimiterString`, initialised to `DEFAULT_DELIMITER`.
    delimiter_string: String,
    /// Java field `primativeTokenizer`, initialised to null.
    primative_tokenizer: PrimativeTokenizer,
    /// Java field `primativeToken`, initialised to null.
    primative_token: *mut Token,
    /// Java field `autodocToken`, initialised to null.
    autodoc_token: *mut Token,
    /// Java field `token`, initialised to `new Token()`.
    token: Token,
    /// Java field `nextToken`, initialised to `new Token()`.
    next_token: Token,
    /// Java field `useNextToken`, initialised to false.
    use_next_token: bool,
    /// Java field `wordBuffer`, initialised to null.
    word_buffer: Option<String>,
    /// Java field `debug`, initialised to false.
    debug: bool,
    /// Java field `lookAhead`, initialised to false.
    look_ahead: bool,
}

impl AutodocTokenizer {
    /// Java package-private `AutodocTokenizer(boolean, Location, String, String, String,
    /// File, BaseManager, AxisID, String, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        allow_alt_comment: bool,
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
    ) -> AutodocTokenizer {
        let primative_tokenizer = if location == Some(Location::Autodoc) {
            PrimativeTokenizer::get_autodoc_instance(
                name,
                manager,
                axis_id,
                not_found_message,
                debug,
            )
        } else {
            PrimativeTokenizer::get_generic_instance(
                env_var,
                subdir_name,
                name,
                autodoc_file,
                manager,
                axis_id,
                not_found_message,
                debug,
                writable,
            )
        };
        let mut restricted_symbols = format!(
            "{}{}{}{}",
            COMMENT_CHAR, SEPARATOR_CHAR, OPEN_CHAR, CLOSE_CHAR
        );
        if allow_alt_comment {
            restricted_symbols.push(ALT_COMMENT_CHAR);
        }
        for quote in QUOTE_LIST.iter() {
            restricted_symbols.push_str(&String::from_utf16_lossy(&[*quote]));
        }
        AutodocTokenizer {
            allow_alt_comment,
            restricted_symbols,
            delimiter_string: DEFAULT_DELIMITER.to_string(),
            primative_tokenizer,
            primative_token: std::ptr::null_mut(),
            autodoc_token: std::ptr::null_mut(),
            token: Token::new(),
            next_token: Token::new(),
            use_next_token: false,
            word_buffer: None,
            debug,
            look_ahead: false,
        }
    }

    /// Java package-private `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java package-private `initialize()`.
    pub fn initialize(&mut self) -> Result<(), LogFileError> {
        self.primative_tokenizer.initialize()
    }

    /// Java package-private `getLogFile()`.
    pub fn get_log_file(&self) -> Option<std::sync::Arc<log_file::Handle>> {
        self.primative_tokenizer.get_log_file_handle()
    }

    /// Java package-private `getToken()`.
    pub fn get_token(&self) -> *mut Token {
        self.autodoc_token
    }

    /// Java package-private `getDelimiterString()`.
    pub fn get_delimiter_string(&self) -> String {
        self.delimiter_string.clone()
    }

    /// Java package-private `setDelimiterString(String)`.
    pub fn set_delimiter_string(&mut self, delimiter_string: Option<&str>) -> bool {
        let delimiter_string = match delimiter_string {
            None => return false,
            Some(delimiter_string) => delimiter_string,
        };
        let mut character: u16;
        let symbols = self.primative_tokenizer.get_symbols().to_string();
        let units: Vec<u16> = delimiter_string.encode_utf16().collect();
        for i in 0..units.len() {
            character = units[i];
            if self
                .restricted_symbols
                .encode_utf16()
                .any(|unit| unit == character)
            {
                return false;
            }
            if !symbols.encode_utf16().any(|unit| unit == character) {
                return false;
            }
        }
        self.delimiter_string = delimiter_string.to_string();
        true
    }

    /// Java package-private `next()`.
    ///
    /// # Safety
    /// The tokenizer's tokens must be live.
    pub unsafe fn next(&mut self) -> *mut Token {
        if self.use_next_token {
            self.use_next_token = false;
            self.autodoc_token = Box::into_raw(Box::new(Token::new_from_token(&self.next_token)));
            return self.autodoc_token;
        }
        if !self.look_ahead {
            self.primative_token = unsafe { self.primative_tokenizer.next(self.primative_token) };
        }
        let found = unsafe { self.find_token() };
        self.autodoc_token = Box::into_raw(Box::new(Token::new_from_token(unsafe { &*found })));
        self.autodoc_token
    }

    /// Java package-private `test(boolean)`.
    ///
    /// # Safety
    /// See `next`.
    pub unsafe fn test(&mut self, tokens: bool) {
        self.initialize();
        let mut token: *mut Token;
        loop {
            token = unsafe { self.next() };
            if tokens {
                println!("{}", unsafe { (*token).to_string() });
            } else if unsafe { (*token).is(token::Type::Eol) } {
                println!();
            } else if !unsafe { (*token).is(token::Type::Eof) } {
                print!("{}", unsafe {
                    match (*token).get_value() {
                        None => "null".to_string(),
                        Some(value) => value.to_string(),
                    }
                });
            }
            if unsafe { (*token).is(token::Type::Eof) } {
                break;
            }
        }
    }

    /// Java package-private `testPrimativeTokenizer(boolean)`.
    pub fn test_primative_tokenizer(&mut self, tokens: bool) {
        self.primative_tokenizer.test(tokens);
    }

    /// Java package-private `testStreamTokenizer(boolean, boolean)`.
    pub fn test_stream_tokenizer(&mut self, tokens: bool, details: bool) {
        self.primative_tokenizer
            .test_stream_tokenizer(tokens, details);
    }

    /// Java private `findToken()`.
    ///
    /// # Safety
    /// See `next`.
    unsafe fn find_token(&mut self) -> *mut Token {
        let mut building_word = false;
        self.look_ahead = false;
        loop {
            if unsafe { self.find_simple_token() } || unsafe { self.find_look_ahead_token() } {
                if building_word {
                    #[allow(unused_assignments)]
                    {
                        building_word = false;
                    }
                    self.make_word();
                    return &mut self.token;
                }
                return &mut self.token;
            }
            if unsafe { (*self.primative_token).is(token::Type::Alphanum) } {
                building_word = true;
                self.build_word();
                self.primative_token =
                    unsafe { self.primative_tokenizer.next(self.primative_token) };
            } else {
                if unsafe { self.find_delimiter() } {
                    if building_word {
                        #[allow(unused_assignments)]
                        {
                            building_word = false;
                        }
                        self.make_word();
                        return &mut self.token;
                    }
                    return &mut self.token;
                }
                // Don't have to call buildWord() here, because findDelimiter handles
                // building a word when it fails.
                building_word = true;
            }
            if !building_word {
                break;
            }
        }
        panic!("java.lang.IllegalStateException");
    }

    /// Java private `findSimpleToken()`.
    ///
    /// # Safety
    /// See `next`.
    ///
    /// Recognizes primative tokens that are also used by autodoc.
    /// Makes one-character tokens out of primative SYMBOL tokens.
    unsafe fn find_simple_token(&mut self) -> bool {
        let primative_token: *const Token = self.primative_token;
        if unsafe {
            (*primative_token).is(token::Type::Eof)
                || (*primative_token).is(token::Type::Eol)
                || (*primative_token).is(token::Type::Whitespace)
        } {
            self.token.copy(unsafe { &*primative_token });
        } else if unsafe {
            (*primative_token).equals_type_and_char(token::Type::Symbol, COMMENT_CHAR as u16)
        } {
            self.token
                .set_type_and_char(token::Type::Comment, COMMENT_CHAR as u16);
        } else if self.allow_alt_comment
            && unsafe {
                (*primative_token)
                    .equals_type_and_char(token::Type::Symbol, ALT_COMMENT_CHAR as u16)
            }
        {
            self.token
                .set_type_and_char(token::Type::Comment, ALT_COMMENT_CHAR as u16);
        } else if unsafe {
            (*primative_token).equals_type_and_string(token::Type::Symbol, Some(SEPARATOR_CHAR))
        } {
            self.token
                .set_type_and_string(token::Type::Separator, SEPARATOR_CHAR);
        } else if unsafe {
            (*primative_token)
                .equals_type_and_string(token::Type::Symbol, Some(&self.delimiter_string))
        } {
            // Found a one character DELIMITER.
            let delimiter_string = self.delimiter_string.clone();
            self.token
                .set_type_and_string(token::Type::Delimiter, &delimiter_string);
        } else if unsafe {
            (*primative_token).equals_type_and_char_list(token::Type::Symbol, Some(&QUOTE_LIST))
        } {
            let character = unsafe { (*primative_token).get_char() };
            self.token.set_type_and_char(token::Type::Quote, character);
        } else {
            return false;
        }
        true
    }

    /// Java private `findLookAheadToken()`.
    ///
    /// # Safety
    /// See `next`.
    unsafe fn find_look_ahead_token(&mut self) -> bool {
        if unsafe {
            (*self.primative_token)
                .equals_type_and_character(token::Type::Symbol, Some(OPEN_CHAR as u16))
        } {
            if unsafe { self.match_with_look_ahead(token::Type::Symbol, Some(OPEN_CHAR as u16)) } {
                self.token.set_type_and_string(
                    token::Type::Subopen,
                    &format!("{}{}", OPEN_CHAR, OPEN_CHAR),
                );
            } else {
                self.token
                    .set_type_and_character(token::Type::Open, Some(OPEN_CHAR as u16));
            }
        } else if unsafe {
            (*self.primative_token)
                .equals_type_and_character(token::Type::Symbol, Some(CLOSE_CHAR as u16))
        } {
            if unsafe { self.match_with_look_ahead(token::Type::Symbol, Some(CLOSE_CHAR as u16)) } {
                self.token.set_type_and_string(
                    token::Type::Subclose,
                    &format!("{}{}", CLOSE_CHAR, CLOSE_CHAR),
                );
            } else {
                self.token
                    .set_type_and_character(token::Type::Close, Some(CLOSE_CHAR as u16));
            }
        } else {
            return false;
        }
        true
    }

    /// Java private `matchWithLookAhead(Token.Type, Character)`.
    ///
    /// Looks ahead to match a type and character.  If the match fails, set
    /// lookAhead to true.
    ///
    /// # Safety
    /// See `next`.
    unsafe fn match_with_look_ahead(
        &mut self,
        match_type: token::Type,
        match_char: Option<u16>,
    ) -> bool {
        self.primative_token = unsafe { self.primative_tokenizer.next(self.primative_token) };
        if unsafe { (*self.primative_token).equals_type_and_character(match_type, match_char) } {
            return true;
        }
        self.look_ahead = true;
        false
    }

    /// Java private `findDelimiter()`.
    ///
    /// Tries to build a multi-character delimiter.
    /// Assumes that findSimpleToken() has already been called.
    /// The delimiter string can only contain symbols that are not found by
    /// findSimpleToken() and findLookAheadToken().
    ///
    /// # Safety
    /// See `next`.
    unsafe fn find_delimiter(&mut self) -> bool {
        let delimiter_units: Vec<u16> = self.delimiter_string.encode_utf16().collect();
        let length = delimiter_units.len() as i32;
        let mut index: i32 = 0;
        let mut delimiter_buffer: Option<String> = None;
        let mut success = false;
        let mut symbol = unsafe { (*self.primative_token).get_char() };
        // attempt to build a delimiter that matches delimiterString
        while !success
            && unsafe { (*self.primative_token).is(token::Type::Symbol) }
            && index < length
            && delimiter_units[index as usize] == symbol
        {
            if delimiter_buffer.is_none() {
                delimiter_buffer = Some(String::new());
            }
            delimiter_buffer
                .as_mut()
                .unwrap()
                .push_str(&String::from_utf16_lossy(&[symbol]));
            if index == length - 1 {
                // found the whole delimiterString - succeed
                success = true;
            } else {
                // haven't matched the entire delimiter string - get next primative token
                index += 1;
                self.primative_token =
                    unsafe { self.primative_tokenizer.next(self.primative_token) };
            }
            symbol = unsafe { (*self.primative_token).get_char() };
        }
        if success {
            let delimiter_buffer = delimiter_buffer.unwrap();
            self.token
                .set_type_and_string_buffer(token::Type::Delimiter, &delimiter_buffer);
            return true;
        }
        // delimiter match failed - build a word
        match delimiter_buffer {
            None => {
                // never went into delimiter string recongnition loop - build a word from
                // the current primativeToken
                self.build_word();
                self.primative_token =
                    unsafe { self.primative_tokenizer.next(self.primative_token) };
            }
            Some(delimiter_buffer) => self.build_word_from_buffer(&delimiter_buffer),
        }
        false
    }

    /// Java private `buildWord()`.  Start or add to a word.
    fn build_word(&mut self) {
        let value = unsafe {
            match (*self.primative_token).get_value() {
                None => "null".to_string(),
                Some(value) => value.to_string(),
            }
        };
        match &mut self.word_buffer {
            None => self.word_buffer = Some(value),
            Some(word_buffer) => word_buffer.push_str(&value),
        }
    }

    /// Java private `buildWord(StringBuffer)`.  Start or add to a word.
    fn build_word_from_buffer(&mut self, buffer: &str) {
        match &mut self.word_buffer {
            None => self.word_buffer = Some(buffer.to_string()),
            Some(word_buffer) => word_buffer.push_str(buffer),
        }
    }

    /// Java private `makeWord()`.
    ///
    /// Make a WORD token out of wordBuffer.  Calls findKeyword().
    fn make_word(&mut self) {
        // The entire word was found - the current token will have to wait until
        // the next time next() is called.
        self.next_token.copy(&self.token);
        self.use_next_token = true;
        // Make the WORD token
        let word_buffer = self.word_buffer.take().unwrap();
        self.token
            .set_type_and_string_buffer(token::Type::Word, &word_buffer);
        // Convert the token to a KEYWORD token if necessary
        self.find_keyword();
    }

    /// Java private `findKeyword()`.  Converts a WORD token to a KEYWORD token.
    fn find_keyword(&mut self) {
        if self.token.is(token::Type::Word)
            && (self.token.equals_string(Some(VERSION_KEYWORD))
                || self.token.equals_string(Some(PIP_KEYWORD))
                || self.token.equals_string(Some(DELIMITER_KEYWORD)))
        {
            self.token.set_type(token::Type::Keyword);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ALT_COMMENT_CHAR, CLOSE_CHAR, COMMENT_CHAR, DEFAULT_DELIMITER, DELIMITER_KEYWORD,
        OPEN_CHAR, PIP_KEYWORD, SEPARATOR_CHAR, VERSION_KEYWORD,
    };

    /// The special characters and keywords the rest of the package builds its strings
    /// from.
    #[test]
    fn source_special_characters_and_keywords() {
        assert_eq!(COMMENT_CHAR, '#');
        assert_eq!(ALT_COMMENT_CHAR, '%');
        assert_eq!(SEPARATOR_CHAR, ".");
        assert_eq!(OPEN_CHAR, '[');
        assert_eq!(CLOSE_CHAR, ']');
        assert_eq!(DEFAULT_DELIMITER, "=");
        assert_eq!(VERSION_KEYWORD, "Version");
        assert_eq!(PIP_KEYWORD, "Pip");
        assert_eq!(DELIMITER_KEYWORD, "KeyValueDelimiter");
    }
}
