//! `IMOD/Etomo/src/etomo/storage/autodoc/WritableStatement.java`.

use super::read_only_statement::ReadOnlyStatement;

/// Complete writable extension of Java `WritableStatement`.
pub trait WritableStatement: ReadOnlyStatement {
    fn remove(&mut self) -> Option<&mut Self>;
}
