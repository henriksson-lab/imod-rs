//! `IMOD/Etomo/src/etomo/storage/autodoc/WritableStatement.java`.
#![allow(dead_code)]

use super::read_only_statement::ReadOnlyStatement;
use super::statement::Statement;

/// Java's abstract `WritableStatement`.  It declares `remove()` and redeclares
/// `getType()`, `getString()`, `sizeLeftSide()`, `getLeftSide(int)` and
/// `getRightSide()` as abstract with `@Override`; those are the same
/// `ReadOnlyStatement` methods, so they stay on the supertrait.
pub trait WritableStatement: ReadOnlyStatement {
    /// Java package-private abstract `remove()`.  Returns a `WritableStatement`, which
    /// every implementation reaches through `Statement`, so the returned reference is a
    /// `Statement` trait object here.
    ///
    /// # Safety
    /// Every `previous`/`next` link reachable from this statement must be null or point
    /// to a live statement.
    unsafe fn remove(&mut self) -> *mut dyn Statement;
}
