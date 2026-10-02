//! `IMOD/Etomo/src/etomo/ui/swing/Column.java`.
//!
//! A table column: a list of cells that are enabled and disabled together.

use std::cell::{Cell as StdCell, RefCell};
use std::rc::Rc;

use super::cell::CellVirtual;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `final class Column`.
pub struct Column {
    /// Java final `list` (an `ArrayList` of `Cell`).
    list: RefCell<Vec<Rc<dyn CellVirtual>>>,
    /// Java `enabled`.
    enabled: StdCell<bool>,
}

impl Default for Column {
    fn default() -> Self {
        Column::new()
    }
}

impl Column {
    /// Java implicit `Column()`.
    pub fn new() -> Column {
        Column {
            list: RefCell::new(Vec::new()),
            enabled: StdCell::new(true),
        }
    }

    /// Java `synchronized add(Cell)`.  (The lock has no Rust counterpart: cells live
    /// on the EDT.)
    pub fn add(&self, cell: Rc<dyn CellVirtual>) {
        cell.set_enabled(self.enabled.get());
        self.list.borrow_mut().push(cell);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enable: bool) {
        self.enabled.set(enable);
        let list = self.list.borrow().clone();
        for cell in list.iter() {
            cell.set_enabled(enable);
        }
    }
}
