//! `IMOD/Etomo/src/etomo/ui/swing/Token.java`.
//!
//! Description:
//! A class to encapsulate a token type and value.  Provides tools for comparing
//! tokens, saving tokens with a key, and for values made of multiple tokens.
//!
//! Token type:
//! Token types are integers. Use typeToString() to get name of the type.
//! Unknown token types are allowed.  See Possible Upgrades.
//!
//! Token value:
//! Tokens with a NULL, EOF, and EOL type always have a null token value.
//!
//! Comparing tokens:
//! Tokens can be compared with an is(int type) function and various equals()
//! functions.  All token value comparisons are done using the result of the
//! static function getKey(String value).
//!
//! Saving tokens with a key:
//! The getKey() functions are public and can be used to create standard keys for
//! saving and retrieving tokens.
//!
//! Values made up of multiple tokens:
//! Link list:
//! This class can be used to make a linked list of tokens.  The next token can be
//! set.  A token can be removed from the list (see dropFromList()).  The next
//! token can be retrieved.
//! Values:
//! Values and keys made of multiple tokens can be retrieved.  See
//! getValue(boolean includeNext) and getKey(boolean includeNext).  When
//! retrieving a string made of multiple tokens, one space with be appended to the
//! string for each null value.
//!
//! Inheritance:
//! This class is not designed to be inherited.
//!
//! Possible Upgrades:
//! This class could be upgraded to allow the addition or an
//! Vector of new token types.  The existing token numbers are all negative, so
//! they wouldn't have to change.
//!
//! Copyright: Copyright 2002 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Representation.**  Despite the `etomo.ui.swing` package name this unit imports
//! nothing from Swing; it is the lexer token shared by `etomo.util.PrimativeTokenizer`
//! and `etomo.storage.autodoc`.
//!
//! `next` and `previous` are Java object references into a mutable doubly linked list,
//! which every list operation aliases.  They are `*mut Token` here - the C-shaped
//! representation CLAUDE.md sanctions - so that each method keeps the source's
//! signature and the source's aliasing.  Ownership of the pointed-to tokens stays with
//! whoever allocated them, exactly as in the source, where the GC owns them.
//!
//! `char` is a UTF-16 code unit in Java, so `getChar`, `numberOf`, `length`, `split` and
//! the `char`-taking `set`/`equals` overloads index and count UTF-16 code units, and the
//! `char` type is `u16`.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

/// Java `Token`.  The class is final.
pub struct Token {
    /// Java field `type`, initialised to `Type.NULL`.
    r#type: Type,
    /// Java field `value`, initialised to null.
    value: Option<String>,
    /// Java field `key`, initialised to null.
    key: Option<String>,
    /// Java field `next`, initialised to null.
    next: *mut Token,
    /// Java field `previous`, initialised to null.
    previous: *mut Token,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

/// Java `convertToKey`.
///
/// Converts a string into a standard key for saving and retrieving tokens.
/// This function is unnecessary when using Token.equals() functions because it
/// is used internally.
///
/// Converts values to lower case.
///
/// `String.toLowerCase()` with the default locale; the Turkish locale's dotless-i rule
/// is the one place Rust's `to_lowercase` and Java's differ, and the JVM this crate is
/// verified against runs in the C locale.
pub fn convert_to_key(value: &str) -> String {
    value.to_lowercase()
}

impl Token {
    /// Java `Token()`.
    pub fn new() -> Token {
        Token {
            r#type: Type::Null,
            value: None,
            key: None,
            next: std::ptr::null_mut(),
            previous: std::ptr::null_mut(),
            debug: false,
        }
    }

    /// Java `Token(Token)`.
    ///
    /// Makes a deep copy of a token.  The field initialisers run first, so `next`,
    /// `previous` and `debug` keep their defaults and only `copy` runs.
    pub fn new_from_token(token: &Token) -> Token {
        let mut new_token = Token::new();
        new_token.copy(token);
        new_token
    }

    /// Java `getType`.
    pub fn get_type(&self) -> Type {
        self.r#type
    }

    /// Java `getValue`.
    pub fn get_value(&self) -> Option<&str> {
        self.value.as_deref()
    }

    /// Java `getChar`.
    pub fn get_char(&self) -> u16 {
        match &self.value {
            None => b' ' as u16,
            Some(value) => value.encode_utf16().next().unwrap_or_else(|| {
                // `"".charAt(0)` throws StringIndexOutOfBoundsException.
                panic!("java.lang.StringIndexOutOfBoundsException: index 0, length 0")
            }),
        }
    }

    /// Java `getValues`.
    ///
    /// Returns a string containing all values in the
    /// token link list concatenated together.  Null values are converted to ' '.
    ///
    /// # Safety
    /// Every `next` pointer reachable from this token must be null or point to a live
    /// `Token`.
    pub unsafe fn get_values(&self) -> String {
        let mut token: *const Token = self;
        let mut buffer = String::new();
        while !token.is_null() {
            match unsafe { &(*token).value } {
                None => buffer.push(' '),
                Some(value) => buffer.push_str(value),
            }
            token = unsafe { (*token).next };
        }
        buffer
    }

    /// Java `getMultiLineValues`.
    ///
    /// Returns a string containing all values in the
    /// token link list concatenated together.  Null values are converted to ' '.
    /// Preserves EOL by concatenating a "\n".
    ///
    /// # Safety
    /// See `get_values`.
    pub unsafe fn get_multi_line_values(&self) -> String {
        let mut token: *const Token = self;
        let mut buffer = String::new();
        while !token.is_null() {
            if unsafe { (*token).r#type } == Type::Eol {
                buffer.push('\n');
            } else {
                match unsafe { &(*token).value } {
                    None => buffer.push(' '),
                    Some(value) => buffer.push_str(value),
                }
            }
            token = unsafe { (*token).next };
        }
        buffer
    }

    // TODO(unit): needs etomo/process/EmergencyMonitor.java - Java
    // `write(LogFile.Handle, LogFile.WriterId)` writes each token's value through
    // `LogFile.Handle.write`/`newLine`, and `etomo/storage/log_file.rs` cannot construct
    // a `Handle` because `Handle`'s constructor takes an `EmergencyMonitor`.

    /// Java `write(LogFile.Handle, LogFile.WriterId)`.
    ///
    /// # Safety
    /// Every token in this token's `next` list must be live.
    pub unsafe fn write(
        &self,
        file: &std::sync::Arc<log_file::Handle>,
        writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError> {
        if self.r#type == Type::Eol {
            file.new_line(writer_id)?;
        } else {
            file.write(self.value.as_deref(), writer_id)?;
        }
        // Avoid using recursion here because token list may be large enough to cause a
        // stack overflow
        let mut pointer = self.next;
        while !pointer.is_null() {
            if unsafe { (*pointer).r#type } == Type::Eol {
                file.new_line(writer_id)?;
            } else {
                file.write(unsafe { (*pointer).value.as_deref() }, writer_id)?;
            }
            pointer = unsafe { (*pointer).next };
        }
        Ok(())
    }

    /// Java `getKey`.
    ///
    /// Returns a string containing all keys in the token
    /// link list concatenated together Null keys are converted to ' '.
    ///
    /// # Safety
    /// See `get_values`.
    pub unsafe fn get_key(&self) -> String {
        let mut buffer = String::new();
        match &self.key {
            None => buffer.push(' '),
            Some(key) => buffer.push_str(key),
        }
        let mut token: *const Token = self.next;
        while !token.is_null() {
            match unsafe { &(*token).key } {
                None => buffer.push(' '),
                Some(key) => buffer.push_str(key),
            }
            token = unsafe { (*token).next };
        }
        buffer
    }

    /// Java `numberOf`.
    ///
    /// Finds the number of contiguous searchChars, start from fromIndex.  Note that the
    /// source declares `fromIndex` and never reads it: the loop starts at 0.
    pub fn number_of(&self, search_char: u16, _from_index: i32) -> i32 {
        let value = match &self.value {
            None => return 0,
            Some(value) => value,
        };
        let value: Vec<u16> = value.encode_utf16().collect();
        let mut found = false;
        let mut number_found = 0;
        for i in 0..value.len() {
            if !found {
                if value[i] == search_char {
                    found = true;
                    number_found += 1;
                }
            } else if value[i] == search_char {
                number_found += 1;
            } else {
                return number_found;
            }
        }
        number_found
    }

    /// Java `length`.
    ///
    /// Returns the number of characters in the value.
    pub fn length(&self) -> i32 {
        match &self.value {
            None => 0,
            Some(value) => value.encode_utf16().count() as i32,
        }
    }

    /// Java `split`.
    ///
    /// Split off a new token from this token.  Set the new token type to the type
    /// parameter.  Set the new token value to a substring of the value in this
    /// token, starting from startIndex and going for size characters.  Remove this
    /// substring from this token.
    ///
    /// The source's `IndexOutOfBoundsException` message concatenates `startIndex` and
    /// `size` into the already-`String` expression rather than adding them, so it reads
    /// e.g. `startIndex + size, 12, must be ...` for `startIndex` 1 and `size` 2; and
    /// when `value` is null the message's own `value.length()` throws a
    /// `NullPointerException` before the `IndexOutOfBoundsException` is constructed.
    ///
    /// The source returns a reference to a heap `Token` that it also links into this
    /// token's `previous`, so the new token is boxed and returned as `*mut Token`;
    /// ownership passes to the caller, where the source's owner is the GC.
    ///
    /// # Safety
    /// `previous` and `next` must be null or point to live `Token`s.  The returned
    /// pointer must eventually be reclaimed with `Box::from_raw`.
    pub unsafe fn split(&mut self, r#type: Type, start_index: i32, size: i32) -> *mut Token {
        let value: Vec<u16> = match &self.value {
            None => panic!("java.lang.NullPointerException"),
            Some(value) => value.encode_utf16().collect(),
        };
        if (value.len() as i32) <= start_index + size {
            panic!(
                "java.lang.IndexOutOfBoundsException: startIndex + size, {}{}, must be less then value.length,{}.",
                start_index,
                size,
                value.len()
            );
        }
        let new_token = Box::into_raw(Box::new(Token::new()));
        unsafe {
            (*new_token).set_type_and_string(
                r#type,
                &String::from_utf16_lossy(
                    &value[start_index as usize..(start_index + size) as usize],
                ),
            )
        };
        self.value = Some(String::from_utf16_lossy(&value[size as usize..]));
        if !self.next.is_null() || !self.previous.is_null() {
            unsafe { (*new_token).next = self };
            if !self.previous.is_null() {
                unsafe { (*new_token).previous = self.previous };
            }
            self.previous = new_token;
        }
        new_token
    }

    /// Java `reset`.
    ///
    /// Sets the type of the token to NULL.
    pub fn reset(&mut self) {
        self.r#type = Type::Null;
        self.value = None;
        self.key = None;
    }

    /// Java `copy`.
    ///
    /// Makes a deep copy of another token.  `next`, `previous` and `debug` are not
    /// copied.  The source's `new String(token.key)` throws a NullPointerException if
    /// `token.value` is non-null while `token.key` is null, which no path in the class
    /// produces.
    pub fn copy(&mut self, token: &Token) {
        self.r#type = token.r#type;
        if token.value.is_none() {
            self.value = None;
            self.key = None;
        } else {
            self.value = token.value.clone();
            self.key = Some(match &token.key {
                None => panic!("java.lang.NullPointerException"),
                Some(key) => key.clone(),
            });
        }
    }

    /// Java `set(Type, String)`.
    ///
    /// Sets the type and value of the token.
    pub fn set_type_and_string(&mut self, r#type: Type, value: &str) {
        self.set_type_field(r#type);
        self.set_string(value);
    }

    /// Java `set(Type, double)`.
    ///
    /// Sets the type and value of the token.
    pub fn set_type_and_double(&mut self, r#type: Type, value: f64) {
        self.set_type_field(r#type);
        // `String.valueOf(double)` is `Double.toString(double)`.
        self.set_string(
            &crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(value),
        );
    }

    /// Java `set(Type, char)`.
    ///
    /// Sets the type and value of the token.
    pub fn set_type_and_char(&mut self, r#type: Type, value: u16) {
        self.set_type_field(r#type);
        self.set_string(&String::from_utf16_lossy(&[value]));
    }

    /// Java `set(Type, Character)`.  A Java `Character` may be null, in which case
    /// `value.charValue()` throws a NullPointerException.
    pub fn set_type_and_character(&mut self, r#type: Type, value: Option<u16>) {
        match value {
            None => panic!("java.lang.NullPointerException"),
            Some(value) => self.set_type_and_char(r#type, value),
        }
    }

    /// Java `set(Type, StringBuffer)`.
    ///
    /// Sets the type and value of the token.
    pub fn set_type_and_string_buffer(&mut self, r#type: Type, value_buffer: &str) {
        self.set_type_and_string(r#type, value_buffer);
    }

    /// Java `set(Type)`.
    ///
    /// Sets the type of the token.  Does not reset the value, unless the type is
    /// NULL, EOF, or EOL.
    pub fn set_type(&mut self, r#type: Type) {
        self.set_type_field(r#type);
        if r#type == Type::Null || r#type == Type::Eof || r#type == Type::Eol {
            self.value = None;
            self.key = None;
        }
    }

    /// Java `set(String)`.
    ///
    /// Sets the value of the token.
    ///
    /// **The NULL/EOF/EOL branch assigns the *parameter*, not the field.**  The source
    /// writes `value = null; key = null;` there, and `value` names the method parameter
    /// while `key` names the field, so `this.value` survives unchanged and only `key` is
    /// cleared.  See CLAUDE.md's "a by-value parameter the source assigns to".
    pub fn set_string(&mut self, value: &str) {
        if self.r#type == Type::Null || self.r#type == Type::Eof || self.r#type == Type::Eol {
            // `value = null;` assigns the parameter, which is never read again; the
            // field `this.value` is deliberately left alone.
            self.key = None;
        } else {
            self.value = Some(value.to_string());
            self.key = Some(convert_to_key(self.value.as_ref().unwrap()));
        }
    }

    /// Java private `setType`.
    fn set_type_field(&mut self, r#type: Type) {
        self.r#type = r#type;
    }

    /// Java `is`.  Returns true if type equals the token type.
    pub fn is(&self, r#type: Type) -> bool {
        self.r#type == r#type
    }

    /// Java `equals(Token)`.  Returns true if the type and key of this token are the
    /// same.
    pub fn equals_token(&self, token: &Token) -> bool {
        self.r#type == token.r#type && self.equals_string(token.value.as_deref())
    }

    /// Java `equals(Type, String)`.  Returns true if the type and key of this token are
    /// the same as the type and getKey(value).
    pub fn equals_type_and_string(&self, r#type: Type, value: Option<&str>) -> bool {
        self.r#type == r#type && self.equals_string(value)
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `equals(Type, char)`.  Returns true if the type and key of this token are
    /// the same as the type and getKey(value).
    pub fn equals_type_and_char(&self, r#type: Type, value: u16) -> bool {
        if self.r#type != r#type {
            return false;
        }
        if self.value.is_none() {
            return false;
        }
        let key: Vec<u16> = match &self.key {
            None => panic!("java.lang.NullPointerException"),
            Some(key) => key.encode_utf16().collect(),
        };
        key.len() == 1 && key[0] == value
    }

    /// Java `equals(Type, char[])`.  Returns true if the type and key of this token are
    /// the same as the type and getKey(an element of valueList).
    pub fn equals_type_and_char_list(&self, r#type: Type, value_list: Option<&[u16]>) -> bool {
        if self.r#type != r#type {
            return false;
        }
        if self.value.is_none() && value_list.is_none() {
            return true;
        }
        if self.value.is_none() || value_list.is_none() {
            return false;
        }
        let value_list = value_list.unwrap();
        let key: Vec<u16> = match &self.key {
            None => panic!("java.lang.NullPointerException"),
            Some(key) => key.encode_utf16().collect(),
        };
        if key.len() == 1 {
            let c_key = key[0];
            for item in value_list.iter() {
                if c_key == *item {
                    return true;
                }
            }
        }
        false
    }

    /// Java `equals(Type, Character)`.
    pub fn equals_type_and_character(&self, r#type: Type, value: Option<u16>) -> bool {
        match value {
            None => panic!("java.lang.NullPointerException"),
            Some(value) => self.equals_type_and_char(r#type, value),
        }
    }

    /// Java `equals(String)`.  Returns true if the key of this token is the same as
    /// getKey(value).
    ///
    /// The source's opening `this.value == value` is a reference comparison; it is true
    /// when both are null, and for two distinct non-null `String` objects it is false
    /// and the `key.equals(convertToKey(value))` test decides.  Rust owns its strings,
    /// so the reference test is reproduced only for the both-null case, which is the
    /// only one any caller reaches.
    pub fn equals_string(&self, value: Option<&str>) -> bool {
        if self.value.is_none() && value.is_none() {
            return true;
        }
        if self.value.is_none() || value.is_none() {
            return false;
        }
        let key = match &self.key {
            None => panic!("java.lang.NullPointerException"),
            Some(key) => key,
        };
        if *key == convert_to_key(value.unwrap()) {
            return true;
        }
        false
    }

    /// Java `getString`.
    pub fn get_string(&self) -> String {
        match &self.value {
            None => format!("({})", self.r#type),
            Some(value) => format!("({},{})", self.r#type, value),
        }
    }

    /// Java `setNext`.
    ///
    /// Sets the next token in the link list.  Returns the next token.
    ///
    /// # Safety
    /// `token` must be null or point to a live `Token`.
    pub unsafe fn set_next(&mut self, token: *mut Token) -> *mut Token {
        if std::ptr::eq(token, self) {
            return token;
        }
        self.next = token;
        if !token.is_null() {
            unsafe { (*token).previous = self };
        }
        token
    }

    /// Java `next`.  Returns the next token.
    pub fn next(&self) -> *mut Token {
        self.next
    }

    /// Java `removeListFromHead`.
    ///
    /// Removes the list of tokens pointed to by next from a token.  Token must be
    /// the head of the list.  Note the source's `previos` spelling in the message.
    ///
    /// # Safety
    /// `previous` and `next` must be null or point to live `Token`s.
    pub unsafe fn remove_list_from_head(&mut self) {
        if !self.previous.is_null() {
            // error - not the head of the list
            panic!(
                "java.lang.IllegalStateException: Must be the head of the list:  this={},previos={},next={}",
                self,
                unsafe { Token::to_string_of_reference(self.previous) },
                unsafe { Token::to_string_of_reference(self.next) }
            );
        }
        self.next = std::ptr::null_mut();
    }

    /// Java's `"" + aTokenReference`, which is `"null"` for a null reference and
    /// `toString()` otherwise.  Used only by the two messages that print one.
    ///
    /// # Safety
    /// `token` must be null or point to a live `Token`.
    unsafe fn to_string_of_reference(token: *const Token) -> String {
        if token.is_null() {
            "null".to_string()
        } else {
            unsafe { (*token).to_string() }
        }
    }

    /// Java `dropFromList`.
    ///
    /// Drops the token from the link list.  Returns the previous token on the list, if
    /// it exists.  If not, returns the next token on the list.
    ///
    /// # Safety
    /// `previous` and `next` must be null or point to live `Token`s.
    pub unsafe fn drop_from_list(&mut self) -> *mut Token {
        let list: *mut Token = if self.previous.is_null() {
            self.next
        } else {
            self.previous
        };
        if !self.previous.is_null() {
            unsafe { (*self.previous).next = self.next };
        }
        if !self.next.is_null() {
            unsafe { (*self.next).previous = self.previous };
        }
        self.previous = std::ptr::null_mut();
        self.next = std::ptr::null_mut();
        list
    }
}

impl Default for Token {
    fn default() -> Token {
        Token::new()
    }
}

/// Java `toString`.  `type + " " + value`, and Java renders a null `value` as `null`.
impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            self.r#type,
            match &self.value {
                None => "null",
                Some(value) => value,
            }
        )
    }
}

/// Java's nested `public static final class Type`.  Each constant is a distinct object
/// and every comparison in the class is `==`, so the typesafe enum is a Rust enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
    /// Java `NULL`.
    Null,
    /// Java `EOF`.
    Eof,
    /// Java `EOL`.
    Eol,
    /// Java `ALPHANUM`.
    Alphanum,
    /// Java `SYMBOL`.
    Symbol,
    /// Java `WHITESPACE`.
    Whitespace,
    /// Java `COMMENT`.
    Comment,
    /// Java `SEPARATOR`, whose descr is `AutodocTokenizer.SEPARATOR_CHAR.toString()`.
    Separator,
    /// Java `OPEN`, whose descr is `AutodocTokenizer.OPEN_CHAR.toString()`.
    Open,
    /// Java `CLOSE`, whose descr is `AutodocTokenizer.CLOSE_CHAR.toString()`.
    Close,
    /// Java `DELIMITER`.
    Delimiter,
    /// Java `WORD`.
    Word,
    /// Java `KEYWORD`.
    Keyword,
    /// Java `ANYTHING`.
    Anything,
    /// Java `SUBOPEN`, whose descr is `OPEN_CHAR.toString() + OPEN_CHAR.toString()`.
    Subopen,
    /// Java `SUBCLOSE`, whose descr is `CLOSE_CHAR.toString() + CLOSE_CHAR.toString()`.
    Subclose,
    /// Java `NUMERIC`.
    Numeric,
    /// Java `ALPHABETIC`.
    Alphabetic,
    /// Java `QUOTE`.
    Quote,
}

impl Type {
    /// Java field `descr`, a `private final String` that is null for every constant
    /// built with the no-argument constructor.  The five that are not are built from
    /// `AutodocTokenizer.SEPARATOR_CHAR` (`"."`), `OPEN_CHAR` (`'['`) and `CLOSE_CHAR`
    /// (`']'`) - `IMOD/Etomo/src/etomo/storage/autodoc/AutodocTokenizer.java:72-74`.
    /// `AutodocTokenizer.java` has no module of its own yet; these are its three
    /// declared constants, quoted at their declaration.
    fn descr(self) -> Option<&'static str> {
        match self {
            Self::Separator => Some("."),
            Self::Open => Some("["),
            Self::Close => Some("]"),
            Self::Subopen => Some("[["),
            Self::Subclose => Some("]]"),
            _ => None,
        }
    }

    /// Java `getDescr`.
    pub fn get_descr(self) -> String {
        match self.descr() {
            None => self.to_string(),
            Some(descr) => descr.to_string(),
        }
    }
}

/// Java `Type.toString`.  Note that `ALPHABETIC` prints `ALPHABETICAL`, and that the
/// final `return "UNKNOWN"` is unreachable for every declared constant.
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Null => "NULL",
            Self::Eof => "EOF",
            Self::Eol => "EOL",
            Self::Alphanum => "ALPHANUM",
            Self::Symbol => "SYMBOL",
            Self::Whitespace => "WHITESPACE",
            Self::Comment => "COMMENT",
            Self::Separator => "SEPARATOR",
            Self::Open => "OPEN",
            Self::Close => "CLOSE",
            Self::Delimiter => "DELIMITER",
            Self::Word => "WORD",
            Self::Keyword => "KEYWORD",
            Self::Anything => "ANYTHING",
            Self::Subopen => "SUBOPEN",
            Self::Subclose => "SUBCLOSE",
            Self::Numeric => "NUMERIC",
            Self::Alphabetic => "ALPHABETICAL",
            Self::Quote => "QUOTE",
        })
    }
}
