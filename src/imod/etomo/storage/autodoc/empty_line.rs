//! `IMOD/Etomo/src/etomo/storage/autodoc/EmptyLine.java`.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::autodoc;
use super::read_only_statement::ReadOnlyStatement;
use super::section::Section;
use super::statement::{Statement, StatementBase, Type};
use super::writable_statement::WritableStatement;
use super::write_only_statement_list::WriteOnlyStatementList;

/// Java package-private final `EmptyLine extends Statement`.
pub struct EmptyLine {
    /// The fields Java inherits from `Statement`.
    statement: StatementBase,
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyStatementList,
}

/// Java `TYPE`: `Statement.Type.EMPTY_LINE`.
const TYPE: Type = Type::EmptyLine;

impl EmptyLine {
    /// Java `EmptyLine(WriteOnlyStatementList, Statement, int)`.  The allocation is
    /// leaked, as `statement.rs`'s header describes.
    ///
    /// # Safety
    /// `parent` must point to a live statement list and `previous_statement` must be
    /// null or point to a live statement.
    pub unsafe fn new(
        parent: *mut dyn WriteOnlyStatementList,
        previous_statement: *mut dyn Statement,
        line_num: i32,
    ) -> *mut EmptyLine {
        let this = Box::into_raw(Box::new(EmptyLine {
            statement: StatementBase::initial(),
            parent,
        }));
        unsafe { StatementBase::statement(this, previous_statement, line_num) };
        this
    }
}

impl Statement for EmptyLine {
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
        file.new_line(writer_id)
    }

    /// Java `print(int)`.
    unsafe fn print(&self, level: i32) {
        autodoc::print_indent(level);
        println!("<empty-line>");
    }
}

impl WritableStatement for EmptyLine {
    /// Java inherits `Statement.remove()` unchanged.
    unsafe fn remove(&mut self) -> *mut dyn Statement {
        unsafe { self.statement.remove() }
    }
}

impl ReadOnlyStatement for EmptyLine {
    /// Java `getType()`.
    fn get_type(&self) -> Type {
        TYPE
    }

    /// Java `getString()`.
    fn get_string(&self) -> String {
        "".to_string()
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
