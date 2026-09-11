//! `IMOD/Etomo/src/etomo/storage/autodoc/Statement.java`.
//!
//! **Representation.**  `Statement` is an abstract class with four concrete
//! subclasses (`NameValuePair`, `Subsection`, `Comment`, `EmptyLine`) held
//! polymorphically in `Autodoc`'s and `Section`'s statement lists.  Rust has no
//! inheritance, so the class splits into the `Statement` trait - the virtual calls -
//! and `StatementBase` - the three fields `Statement` itself declares, embedded in
//! every subclass and reached through `statement()`/`statement_mut()`.
//!
//! Java's constructor assigns `previous.next = this`, which needs the object to
//! exist; Rust cannot produce `this` before the allocation, so each subclass
//! constructor boxes the object first and then runs `StatementBase::statement`, the
//! superclass constructor body.  The only observable difference is that the
//! subclass's own fields are already assigned when `previous.next` is set, and no
//! source path reads them from there.
//!
//! **Ownership.**  Java's statements are owned by the garbage collector and aliased
//! by the statement link list, the statement lists of `Autodoc`/`Section`, and (for
//! `NameValuePair`) by `Attribute.nameValuePairList`.  As in `etomo/ui/swing/token.rs`,
//! the Rust translation keeps the source's aliasing with raw pointers: each
//! constructor leaks a `Box`, and nothing in the package reclaims it.
#![allow(dead_code)]

use crate::imod::etomo::storage::log_file;

use super::writable_statement::WritableStatement;

/// Java's nested `public static final class Type`.  Each constant is a distinct object
/// and every comparison against it is `==`, so the typesafe enum is a Rust enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
    /// Java `NAME_VALUE_PAIR`.
    NameValuePair,
    /// Java `SUBSECTION`.
    Subsection,
    /// Java `COMMENT`.
    Comment,
    /// Java `EMPTY_LINE`.
    EmptyLine,
}

/// Java `Type.toString()`, which returns the `string` each constant was built with.
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NameValuePair => "NAME_VALUE_PAIR",
            Self::Subsection => "SUBSECTION",
            Self::Comment => "COMMENT",
            Self::EmptyLine => "EMPTY_LINE",
        })
    }
}

/// The three fields Java's abstract `Statement` declares, plus the two methods it
/// implements.  Every subclass embeds one.
pub struct StatementBase {
    /// Java field `lineNum`.
    line_num: i32,
    /// Java field `previous`, initialised to null.
    previous: *mut dyn Statement,
    /// Java field `next`, initialised to null.
    next: *mut dyn Statement,
}

impl StatementBase {
    /// Java's field initialisers: `previous = null` and `next = null`, with `lineNum`
    /// left for the constructor.  A null `*mut dyn Statement` still needs a vtable, so
    /// the null is built from a null `*mut EmptyStatement` - a type that exists only to
    /// name one - which is never dereferenced.
    pub fn initial() -> StatementBase {
        StatementBase {
            line_num: 0,
            previous: std::ptr::null_mut::<EmptyStatement>(),
            next: std::ptr::null_mut::<EmptyStatement>(),
        }
    }

    /// Java `Statement(Statement previousStatement, final int lineNum)`.  `this` is the
    /// subclass's own allocation; see the module header.
    ///
    /// # Safety
    /// `this` must point to a live statement whose `StatementBase` is the one being
    /// initialised, and `previous_statement` must be null or point to a live statement.
    pub unsafe fn statement(
        this: *mut dyn Statement,
        previous_statement: *mut dyn Statement,
        line_num: i32,
    ) {
        unsafe {
            (*this).statement_mut().line_num = line_num;
            if !previous_statement.is_null() {
                // set up link list
                (*this).statement_mut().previous = previous_statement;
                (*(*this).statement_mut().previous).statement_mut().next = this;
            }
        }
    }

    /// Java `remove()`.
    ///
    /// Update the previous.next and next.previous to remove this instance from the
    /// link list.  Returns previous.
    ///
    /// # Safety
    /// `previous` and `next` must be null or point to live statements.
    pub unsafe fn remove(&mut self) -> *mut dyn Statement {
        unsafe {
            // remove this instance from the link list
            if !self.previous.is_null() {
                (*self.previous).statement_mut().next = self.next;
            }
            if !self.next.is_null() {
                (*self.next).statement_mut().previous = self.previous;
            }
        }
        self.previous
    }

    /// Java `getLineNum()`.
    pub fn get_line_num(&self) -> i32 {
        self.line_num
    }
}

/// Java's abstract `Statement`.  The trait carries the class's three abstract methods
/// and the accessors that stand for Java's inherited field access.
pub trait Statement: WritableStatement {
    /// Java's implicit access to the inherited `Statement` fields.
    fn statement(&self) -> &StatementBase;
    /// Java's implicit access to the inherited `Statement` fields.
    fn statement_mut(&mut self) -> &mut StatementBase;

    /// Java abstract `write(LogFile.Handle, LogFile.WriterId)`.
    ///
    /// # Safety
    /// Every token, attribute and section this statement points at must be live.
    unsafe fn write(
        &self,
        file: &std::sync::Arc<log_file::Handle>,
        writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError>;

    /// Java abstract `print(int)`.
    ///
    /// # Safety
    /// Every token, attribute and section this statement points at must be live.
    unsafe fn print(&self, level: i32);

    /// Java abstract `wrapValue(String, String, String, String, int, int)`.
    ///
    /// # Safety
    /// See `print`.
    unsafe fn wrap_value(
        &mut self,
        no_wrap_prefix: Option<&str>,
        wrap_prefix: Option<&str>,
        divider: Option<&str>,
        default_divider: Option<&str>,
        min_length: i32,
        wrap_length: i32,
    );
}

/// Not a source type.  `*mut dyn Statement` is a fat pointer, so a null one needs a
/// concrete type to take its vtable from; this names that type and is never
/// instantiated, dereferenced or reachable from any source path.
pub enum EmptyStatement {}

impl super::read_only_statement::ReadOnlyStatement for EmptyStatement {
    fn get_type(&self) -> Type {
        match *self {}
    }
    fn get_string(&self) -> String {
        match *self {}
    }
    fn size_left_side(&self) -> i32 {
        match *self {}
    }
    fn get_left_side(&self) -> Option<String> {
        match *self {}
    }
    fn get_left_side_at(&self, _index: i32) -> Option<String> {
        match *self {}
    }
    fn get_right_side(&self) -> Option<String> {
        match *self {}
    }
    fn get_subsection(&self) -> *mut super::section::Section {
        match *self {}
    }
    fn get_line_num(&self) -> i32 {
        match *self {}
    }
}
impl WritableStatement for EmptyStatement {
    unsafe fn remove(&mut self) -> *mut dyn Statement {
        match *self {}
    }
}
impl Statement for EmptyStatement {
    fn statement(&self) -> &StatementBase {
        match *self {}
    }
    unsafe fn write(
        &self,
        _file: &std::sync::Arc<log_file::Handle>,
        _writer_id: &log_file::WriterId,
    ) -> Result<(), log_file::LogFileError> {
        match *self {}
    }
    fn statement_mut(&mut self) -> &mut StatementBase {
        match *self {}
    }
    unsafe fn print(&self, _level: i32) {
        match *self {}
    }
    unsafe fn wrap_value(
        &mut self,
        _no_wrap_prefix: Option<&str>,
        _wrap_prefix: Option<&str>,
        _divider: Option<&str>,
        _default_divider: Option<&str>,
        _min_length: i32,
        _wrap_length: i32,
    ) {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::super::autodoc::Autodoc;
    use super::super::read_only_statement_list::ReadOnlyStatementList;
    use super::super::writable_autodoc::WritableAutodoc;

    /// The superclass constructor links each new statement behind the most recent one,
    /// and `remove()` returns the previous statement after unlinking.
    #[test]
    fn statement_link_list_is_built_and_unlinked_as_the_source_does() {
        unsafe {
            let autodoc = Autodoc::new(Some("link"), std::ptr::null_mut());
            (*autodoc).add_name_value_pair_attribute(Some("one"), Some("1"));
            (*autodoc).add_name_value_pair_attribute(Some("two"), Some("2"));
            (*autodoc).add_name_value_pair_attribute(Some("three"), Some("3"));
            let mut location = (*autodoc).get_statement_location();
            let first = ReadOnlyStatementList::next_statement(&*autodoc, location.as_mut());
            let second = ReadOnlyStatementList::next_statement(&*autodoc, location.as_mut());
            let third = ReadOnlyStatementList::next_statement(&*autodoc, location.as_mut());
            assert_eq!((*first).get_string(), "one = 1");
            let previous = (*second).remove();
            assert!(std::ptr::addr_eq(previous, first));
            assert!(std::ptr::addr_eq((*first).statement().next, third));
            assert!(std::ptr::addr_eq((*third).statement().previous, first));
        }
    }
}
