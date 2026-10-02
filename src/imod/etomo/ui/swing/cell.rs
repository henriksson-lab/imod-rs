//! `IMOD/Etomo/src/etomo/ui/swing/Cell.java`.
//!
//! Java `abstract class Cell`, the base of the table cells.  A subclass embeds
//! [`Cell`] as its `base` field (with `Deref<Target = Cell>`) and implements
//! [`CellVirtual`] for the abstract methods.  `Cell` itself never calls them, so it
//! keeps no pointer to its subclass; code that holds cells polymorphically (the Java
//! `Cell` type) holds `Rc<dyn CellVirtual>`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::logic::table_state::{self, TableState};
use crate::imod::etomo::ui::table_field::TableField;

/// The abstract methods of Java `Cell`.
pub trait CellVirtual {
    /// The embedded `Cell` (Java `this` seen as a `Cell`).
    fn cell(&self) -> &Cell;

    /// Java abstract `setEnabled(boolean)`.
    fn set_enabled(&self, enable: bool);

    /// Java abstract `msgLabelChanged()`.  Message from row header or column header
    /// that their label has changed.
    fn msg_label_changed(&self);

    /// Java abstract `add(JPanel, GridBagLayout, GridBagConstraints)`.  The layout and
    /// constraints are layout only and are not modelled.
    fn add(&self, panel: &Rc<JComponent>);
}

/// Java `Cell`.
#[derive(Default)]
pub struct Cell {
    /// Java `tableField`.
    table_field: RefCell<Option<Rc<dyn TableField>>>,
    /// Java `tableState`.
    table_state: RefCell<Option<Rc<dyn TableState>>>,
}

impl Cell {
    /// Java implicit constructor.
    pub fn new() -> Cell {
        Cell::default()
    }

    /// Java `setTableState(TableField, TableState)`.
    pub fn set_table_state(
        &self,
        table_field: Option<Rc<dyn TableField>>,
        table_state: Option<Rc<dyn TableState>>,
    ) {
        *self.table_field.borrow_mut() = table_field;
        *self.table_state.borrow_mut() = table_state;
    }

    /// Java `isDisplay()`.
    pub fn is_display(&self) -> bool {
        let table_state = self.table_state.borrow().clone();
        let Some(table_state) = table_state else {
            return true;
        };
        let table_field = self.table_field.borrow().clone();
        table_state.is_display(table_field.as_deref())
    }

    /// Java `getGridwidth()`.
    pub fn get_gridwidth(&self) -> i32 {
        let table_state = self.table_state.borrow().clone();
        let Some(table_state) = table_state else {
            return table_state::DEFAULT_GRIDWIDTH;
        };
        let table_field = self.table_field.borrow().clone();
        table_state.get_gridwidth(table_field.as_deref())
    }
}
