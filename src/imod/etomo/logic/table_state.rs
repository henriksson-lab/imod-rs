//! `IMOD/Etomo/src/etomo/logic/TableState.java`.
//!
//! Collects and distributes information about the state of a table.  Implementers are
//! EDT objects (`Rc`, `&self` methods).

use crate::imod::etomo::ui::table_field::TableField;

/// Java `TableState.DEFAULT_GRIDWIDTH`.
pub const DEFAULT_GRIDWIDTH: i32 = 1;

/// Java `TableState`.
pub trait TableState {
    /// Java `isDisplay(TableField)`.
    fn is_display(&self, table_field: Option<&dyn TableField>) -> bool;

    /// Java `getGridwidth(TableField)`.
    fn get_gridwidth(&self, table_field: Option<&dyn TableField>) -> i32;
}
