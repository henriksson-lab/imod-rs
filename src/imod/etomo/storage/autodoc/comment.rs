//! `IMOD/Etomo/src/etomo/storage/autodoc/Comment.java`.
//!
//! Description: Represents a comment in an autodoc.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::autodoc_tokenizer;
use super::read_only_statement::ReadOnlyStatement;
use super::section::Section;
use super::statement::{Statement, StatementBase, Type};
use super::writable_statement::WritableStatement;
use super::write_only_statement_list::WriteOnlyStatementList;
use crate::imod::etomo::ui::swing::token::Token;

/// Java package-private final `Comment extends Statement`.
pub struct Comment {
    /// The fields Java inherits from `Statement`.
    statement: StatementBase,
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyStatementList,
    /// Java field `comment`.
    comment: *mut Token,
}

/// Java `TYPE`: `Statement.Type.COMMENT`.
const TYPE: Type = Type::Comment;

impl Comment {
    /// Java `Comment(Token, WriteOnlyStatementList, Statement, int)`.
    ///
    /// # Safety
    /// `comment` must be null or point to a live `Token` link list, `parent` must point
    /// to a live statement list, and `previous_statement` must be null or point to a
    /// live statement.
    pub unsafe fn new(
        comment: *mut Token,
        parent: *mut dyn WriteOnlyStatementList,
        previous_statement: *mut dyn Statement,
        line_num: i32,
    ) -> *mut Comment {
        let this = Box::into_raw(Box::new(Comment {
            statement: StatementBase::initial(),
            parent,
            comment,
        }));
        unsafe { StatementBase::statement(this, previous_statement, line_num) };
        this
    }
}

impl Statement for Comment {
    fn statement(&self) -> &StatementBase {
        &self.statement
    }

    fn statement_mut(&mut self) -> &mut StatementBase {
        &mut self.statement
    }

    /// Java `wrapValue(String, String, String, String, int, int)`, whose body is empty.
    unsafe fn wrap_value(
        &mut self,
        _no_wrap_prefix: Option<&str>,
        _wrap_prefix: Option<&str>,
        _divider: Option<&str>,
        _default_divider: Option<&str>,
        _min_length: i32,
        _wrap_length: i32,
    ) {
    }

    /// Java `write(LogFile.Handle, LogFile.WriterId)`.
    unsafe fn write(
        &self,
        file: &std::sync::Arc<log_file::Handle>,
        writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError> {
        file.write_character(Some(autodoc_tokenizer::COMMENT_CHAR), writer_id)?;
        if !self.comment.is_null() {
            unsafe { (*self.comment).write(file, writer_id)? };
        }
        file.new_line(writer_id)
    }

    /// Java `print(int)`.  Note the source ignores `level` here.
    unsafe fn print(&self, _level: i32) {
        if !self.comment.is_null() {
            println!("<comment> {}", unsafe { (*self.comment).get_values() });
        }
    }
}

impl WritableStatement for Comment {
    /// Java inherits `Statement.remove()` unchanged.
    unsafe fn remove(&mut self) -> *mut dyn Statement {
        unsafe { self.statement.remove() }
    }
}

impl ReadOnlyStatement for Comment {
    /// Java `getType()`.
    fn get_type(&self) -> Type {
        TYPE
    }

    /// Java `getString()`.
    fn get_string(&self) -> String {
        if !self.comment.is_null() {
            return format!("{} {}", autodoc_tokenizer::COMMENT_CHAR, unsafe {
                (*self.comment).get_values()
            });
        }
        format!("{}", autodoc_tokenizer::COMMENT_CHAR)
    }

    /// Java `sizeLeftSide()`.
    fn size_left_side(&self) -> i32 {
        0
    }

    /// Java `getLeftSide()`.
    fn get_left_side(&self) -> Option<String> {
        None
    }

    /// Java `getLeftSide(int)`.
    fn get_left_side_at(&self, _index: i32) -> Option<String> {
        None
    }

    /// Java `getRightSide()`.
    fn get_right_side(&self) -> Option<String> {
        if !self.comment.is_null() {
            return Some(unsafe { (*self.comment).get_values() });
        }
        Some("".to_string())
    }

    /// Java `getSubsection()`.
    fn get_subsection(&self) -> *mut Section {
        std::ptr::null_mut()
    }

    /// Java inherits `Statement.getLineNum()`.
    fn get_line_num(&self) -> i32 {
        self.statement.get_line_num()
    }
}
