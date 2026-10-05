//! `IMOD/Etomo/src/etomo/type/IteratorParser.java`.
//!
//! IterationList may contain array descriptors in the form start-end, for example
//! "2,4 - 9,10".
//!
//! ```text
//! iterator =>  -WHITESPACE- -( element { -WHITESPACE- DIVIDER -WHITESPACE- element } -WHITESPACE- )- EOF
//! element => NUMBER -(-WHITESPACE- DASH -WHITESPACE- NUMBER )-
//! ```
//!
//! This parser fails on the first error.  It can return the list as written (for
//! nad_eed_3d) or can expand the ranges (for 3dmod).
//!
//! The list being filled is passed to each private method rather than kept in a
//! field (Java's `iteratorElementList` member), because the list owns this parser.
//! Java's `tokenizer` and `token` fields are reset by every `parse` and read only
//! during it; they live in a [`Cursor`] for the duration of the parse (a `Token`
//! carries a raw link pointer, so a parser holding one could not be shared with
//! process threads as the params that own it are).
//! The source's `catch (IOException e)` around `tokenizer.next` has nothing to catch
//! in the translation (`PrimativeTokenizer.next` reads from memory).

use super::axis_id::AxisID;
use super::iterator_element::IteratorElement;
use super::iterator_element_list::IteratorElementList;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// Java private static final `DIVIDER`.
const DIVIDER: &str = ",";
/// Java private static final `DASH`.
const DASH: &str = "-";

/// Java's `tokenizer` and `token` fields while a `parse` runs.
struct Cursor {
    /// Java private `tokenizer`.
    tokenizer: Option<PrimativeTokenizer>,
    /// Java private `token`.
    token: Option<Box<Token>>,
}

/// Java `public final class IteratorParser`.
pub struct IteratorParser {
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final `description`.
    description: Option<String>,
    /// Java private `prevPrevPrevValue`, initially null.
    prev_prev_prev_value: Option<String>,
    /// Java private `prevPrevValue`, initially null.
    prev_prev_value: Option<String>,
    /// Java private `prevValue`, initially null.
    prev_value: Option<String>,
    /// Java private `valid`, initially true.
    valid: bool,
}

impl IteratorParser {
    /// Java `IteratorParser(BaseManager, AxisID, String)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        description: Option<&str>,
    ) -> IteratorParser {
        IteratorParser {
            manager,
            axis_id,
            description: description.map(str::to_owned),
            prev_prev_prev_value: None,
            prev_prev_value: None,
            prev_value: None,
            valid: true,
        }
    }

    /// Java `parse(String, IteratorElementList)`.  Parses an iterator.
    pub fn parse(
        &mut self,
        input: Option<&str>,
        iterator_element_list: Option<&mut IteratorElementList>,
    ) {
        // Reset parser.
        self.valid = true;
        let mut own_list;
        let iterator_element_list = match iterator_element_list {
            Some(list) => list,
            None => {
                own_list = IteratorElementList::new(
                    self.manager,
                    self.axis_id,
                    self.description.as_deref(),
                );
                &mut own_list
            }
        };
        let mut tokenizer =
            PrimativeTokenizer::get_numeric_string_instance(input.unwrap_or(""), false);
        self.valid = false;
        match tokenizer.initialize() {
            Ok(()) => self.valid = true,
            Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                eprintln!("{e}");
                // `catch (LogFileException e)` / `catch (IOException e)`.
                let kind = match e {
                    LogFileError::Io(_) => "IOException",
                    _ => "FileNotFoundException",
                };
                let message = format!(
                    "Unable to parse {}.  {kind}: {}",
                    self.description.as_deref().unwrap_or("null"),
                    e.get_message()
                );
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        self.manager,
                        &message,
                        "Etomo Error",
                        self.axis_id,
                    )
                });
            }
        }
        let mut cursor = Cursor {
            tokenizer: Some(tokenizer),
            token: None,
        };
        self.iterator(&mut cursor, iterator_element_list);
    }

    /// Java private `iterator()`.
    fn iterator(&mut self, cursor: &mut Cursor, list: &mut IteratorElementList) {
        self.next_token(cursor);
        if Self::token_is(cursor, TokenType::Whitespace) {
            self.next_token(cursor);
        }
        if Self::token_is(cursor, TokenType::Eof) {
            return;
        }
        if !self.element(cursor, list) {
            return;
        }
        if Self::token_is(cursor, TokenType::Whitespace) {
            self.next_token(cursor);
        }
        while !Self::token_is(cursor, TokenType::Eof) {
            if !cursor
                .token
                .as_ref()
                .is_some_and(|token| token.equals_type_and_string(TokenType::Symbol, Some(DIVIDER)))
            {
                self.report_error(cursor, DIVIDER);
                return;
            }
            self.next_token(cursor);
            if Self::token_is(cursor, TokenType::Whitespace) {
                self.next_token(cursor);
            }
            if !self.element(cursor, list) {
                return;
            }
            if Self::token_is(cursor, TokenType::Whitespace) {
                self.next_token(cursor);
            }
        }
    }

    /// Java private `element()`.  Returns true if successful.
    fn element(&mut self, cursor: &mut Cursor, list: &mut IteratorElementList) -> bool {
        if !Self::token_is(cursor, TokenType::Numeric) {
            self.report_error(cursor, &TokenType::Numeric.to_string());
            return false;
        }
        // Found an element - save it.
        let first = Self::token_value(cursor);
        self.next_token(cursor);
        if Self::token_is(cursor, TokenType::Whitespace) {
            self.next_token(cursor);
        }
        if !cursor
            .token
            .as_ref()
            .is_some_and(|token| token.equals_type_and_string(TokenType::Symbol, Some(DASH)))
        {
            // Save numeric element
            list.add(IteratorElement::new_string(first.as_deref()));
            return true;
        }
        self.next_token(cursor);
        if Self::token_is(cursor, TokenType::Whitespace) {
            self.next_token(cursor);
        }
        if !Self::token_is(cursor, TokenType::Numeric) {
            self.report_error(cursor, &TokenType::Numeric.to_string());
            return false;
        }
        // Save range element
        list.add(IteratorElement::new_string_string(
            first.as_deref(),
            Self::token_value(cursor).as_deref(),
        ));
        self.next_token(cursor);
        true
    }

    /// Java private `reportError(String)`.
    fn report_error(&mut self, cursor: &Cursor, expected: &str) {
        self.valid = false;
        let mut buffer = String::new();
        let mut index: i32 = -1;
        for value in [
            &self.prev_prev_prev_value,
            &self.prev_prev_value,
            &self.prev_value,
        ]
        .into_iter()
        .flatten()
        {
            buffer.push_str(value);
            index = buffer.encode_utf16().count() as i32;
        }
        if let Some(token) = &cursor.token {
            index += 1;
            buffer.push_str(token.get_value().unwrap_or("null"));
        }
        let mut caret = String::new();
        if index > 0 {
            // `String.format("%1$" + index + "s", "^")`: right-aligned in index columns.
            caret = format!("{:>width$}", "^", width = index as usize);
        }
        let message = format!(
            "In {}, iterator syntax error.\n{}\n{}\nExpected \"{}\".  Syntax example:  2,4-9,10",
            self.description.as_deref().unwrap_or("null"),
            buffer,
            caret,
            expected
        );
        ui_harness::with(|harness| {
            harness.open_message_dialog_base_manager_string_string_axis_id(
                self.manager,
                &message,
                "Entry Error",
                self.axis_id,
            )
        });
    }

    /// Java private `nextToken()`.  Save the last two values for error messages.
    /// Place the result of tokenizer.next() into token.
    fn next_token(&mut self, cursor: &mut Cursor) {
        self.prev_prev_prev_value = self.prev_prev_value.take();
        self.prev_prev_value = self.prev_value.take();
        self.prev_value = cursor
            .token
            .as_ref()
            .and_then(|token| token.get_value().map(str::to_owned));
        match cursor.tokenizer.as_mut() {
            Some(tokenizer) => tokenizer.next(&mut cursor.token),
            // Java sets an EOL token and then dereferences the null tokenizer
            // (unreachable: `parse` always sets it).
            None => {
                let mut token = Token::new();
                token.set_type(TokenType::Eol);
                cursor.token = Some(Box::new(token));
            }
        }
    }

    /// `token.is(type)` on the current token.
    fn token_is(cursor: &Cursor, r#type: TokenType) -> bool {
        cursor.token.as_ref().is_some_and(|token| token.is(r#type))
    }

    /// `token.getValue()` on the current token.
    fn token_value(cursor: &Cursor) -> Option<String> {
        cursor
            .token
            .as_ref()
            .and_then(|token| token.get_value().map(str::to_owned))
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}
