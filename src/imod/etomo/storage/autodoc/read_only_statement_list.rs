//! `IMOD/Etomo/src/etomo/storage/autodoc/ReadOnlyStatementList.java`.
#![allow(dead_code)]

use super::statement::Statement;
use super::statement_location::StatementLocation;

/// Source `ReadOnlyStatementList` interface.
pub trait ReadOnlyStatementList {
    /// Java `getString()`.
    fn get_string(&self) -> String;
    /// Java `getStatementLocation()`.  `Autodoc` returns null when its statement list
    /// is null, so the return is an `Option`.
    fn get_statement_location(&self) -> Option<StatementLocation>;
    /// Java `nextStatement(StatementLocation)`.  Java's parameter may be null.
    ///
    /// # Safety
    /// Every statement in the list must be live.
    unsafe fn next_statement(&self, location: Option<&mut StatementLocation>)
    -> *mut dyn Statement;
    /// Java `getName()`.
    fn get_name(&self) -> Option<String>;
}
