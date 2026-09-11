//! `IMOD/Etomo/src/etomo/storage/autodoc/Subsection.java`.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::autodoc;
use super::read_only_statement::ReadOnlyStatement;
use super::read_only_statement_list::ReadOnlyStatementList;
use super::section::Section;
use super::statement::{Statement, StatementBase, Type};
use super::writable_statement::WritableStatement;
use super::write_only_statement_list::WriteOnlyStatementList;

/// Java public final `Subsection extends Statement`.
pub struct Subsection {
    /// The fields Java inherits from `Statement`.
    statement: StatementBase,
    /// Java field `parent`.
    parent: *mut dyn WriteOnlyStatementList,
    /// Java field `subsection`.
    subsection: *mut Section,
}

/// Java `TYPE`: `Statement.Type.SUBSECTION`.
const TYPE: Type = Type::Subsection;

impl Subsection {
    /// Java package-private `Subsection(Section, WriteOnlyStatementList, Statement,
    /// int)`.
    ///
    /// # Safety
    /// `subsection` must point to a live `Section`, `parent` to a live statement list,
    /// and `previous_statement` must be null or point to a live statement.
    pub unsafe fn new(
        subsection: *mut Section,
        parent: *mut dyn WriteOnlyStatementList,
        previous_statement: *mut dyn Statement,
        line_num: i32,
    ) -> *mut Subsection {
        let this = Box::into_raw(Box::new(Subsection {
            statement: StatementBase::initial(),
            parent,
            subsection,
        }));
        unsafe { StatementBase::statement(this, previous_statement, line_num) };
        this
    }
}

impl Statement for Subsection {
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
        unsafe { (*self.subsection).write(file, writer_id) }
    }

    /// Java `print(int)`.
    unsafe fn print(&self, level: i32) {
        autodoc::print_indent(level);
        unsafe { (*self.subsection).print(level) };
    }
}

impl WritableStatement for Subsection {
    /// Java inherits `Statement.remove()` unchanged.
    unsafe fn remove(&mut self) -> *mut dyn Statement {
        unsafe { self.statement.remove() }
    }
}

impl ReadOnlyStatement for Subsection {
    /// Java `getType()`.
    fn get_type(&self) -> Type {
        TYPE
    }

    /// Java `sizeLeftSide()`.
    fn size_left_side(&self) -> i32 {
        1
    }

    /// Java `getString()`.
    fn get_string(&self) -> String {
        unsafe { ReadOnlyStatementList::get_string(&*self.subsection) }
    }

    /// Java `getLeftSide()`.
    fn get_left_side(&self) -> Option<String> {
        Some(unsafe { (*(*self.subsection).get_type_token()).get_values() })
    }

    /// Java `getLeftSide(int)`.
    fn get_left_side_at(&self, index: i32) -> Option<String> {
        if index > 0 {
            return None;
        }
        Some(unsafe { (*(*self.subsection).get_type_token()).get_values() })
    }

    /// Java `getRightSide()`.
    fn get_right_side(&self) -> Option<String> {
        unsafe { ReadOnlyStatementList::get_name(&*self.subsection) }
    }

    /// Java `getSubsection()`.
    fn get_subsection(&self) -> *mut Section {
        self.subsection
    }

    /// Java inherits `Statement.getLineNum()`.
    fn get_line_num(&self) -> i32 {
        self.statement.get_line_num()
    }
}
