//! `IMOD/Etomo/src/etomo/ui/TableField.java`.
//!
//! An interface for enums used to identify fields in a table.

use std::any::Any;

/// Java `TableField`: a marker interface with no members.  Java callers compare a
/// `TableField` against enum constants with `==`; the `Any` supertrait lets a
/// `&dyn TableField` be upcast to `&dyn Any` and downcast to the concrete enum for
/// that comparison (see `logic/processor_table_state.rs`).
pub trait TableField: Any {}
