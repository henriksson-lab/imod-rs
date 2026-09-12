//! `IMOD/Etomo/src/etomo/ui/swing/Column.java`.
//!
//! Java's `Cell` is an abstract package-private base class.  The heterogeneous
//! Java `List<Cell>` is represented by boxed `Cell` trait objects; native Swing
//! widget enablement remains owned by each concrete cell source unit.
#![allow(dead_code)]

use super::cell::{Cell, TableField, TableState};

/// Java package-private final `Column`.
///
/// The Java monitor on `add` maps to Rust's exclusive `&mut self` access.  A
/// `Column` owns the same heterogeneous collection of abstract Java cells and
/// replays its stored enabled state when a cell is registered.
pub struct Column<F: TableField, S: TableState<F>> {
    list: Vec<Box<dyn Cell<F, S>>>,
    enabled: bool,
}

impl<F: TableField, S: TableState<F>> Default for Column<F, S> {
    /// Java's implicit no-argument constructor: `list` is empty and `enabled`
    /// has Java's explicit initializer value `true`.
    fn default() -> Self {
        Self {
            list: Vec::new(),
            enabled: true,
        }
    }
}

impl<F: TableField, S: TableState<F>> Column<F, S> {
    /// Java's implicit no-argument constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Java synchronized `add(Cell)`.
    pub fn add(&mut self, mut cell: Box<dyn Cell<F, S>>) {
        cell.set_enabled(self.enabled);
        self.list.push(cell);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enable: bool) {
        self.enabled = enable;
        for cell in &mut self.list {
            cell.set_enabled(enable);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::cell::{
        CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary, CellState,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestField;
    impl TableField for TestField {}

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestTableState;
    impl TableState<TestField> for TestTableState {
        fn is_display(&self, _table_field: Option<&TestField>) -> bool {
            true
        }

        fn get_gridwidth(&self, _table_field: Option<&TestField>) -> i32 {
            Self::DEFAULT_GRIDWIDTH
        }
    }

    struct TestCell {
        state: CellState<TestField, TestTableState>,
        enabled: Rc<RefCell<Vec<bool>>>,
    }
    impl Cell<TestField, TestTableState> for TestCell {
        fn set_enabled(&mut self, enable: bool) {
            self.enabled.borrow_mut().push(enable);
        }

        fn msg_label_changed(&mut self) {}

        fn add(
            &mut self,
            _panel: &mut CellPanelBoundary,
            _layout: &mut CellGridBagLayoutBoundary,
            _constraints: &mut CellGridBagConstraintsBoundary,
        ) {
        }

        fn cell_state(&self) -> &CellState<TestField, TestTableState> {
            &self.state
        }

        fn cell_state_mut(&mut self) -> &mut CellState<TestField, TestTableState> {
            &mut self.state
        }
    }

    #[test]
    fn source_enabled_initializer_is_applied_to_added_cell() {
        let mut column = Column::<TestField, TestTableState>::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        column.add(Box::new(TestCell {
            state: CellState::default(),
            enabled: calls.clone(),
        }));

        assert_eq!(*calls.borrow(), vec![true]);
    }

    #[test]
    fn disabling_before_add_sets_new_cell_to_stored_state() {
        let mut column = Column::<TestField, TestTableState>::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        column.set_enabled(false);
        column.add(Box::new(TestCell {
            state: CellState::default(),
            enabled: calls.clone(),
        }));

        assert!(!column.enabled);
        assert_eq!(*calls.borrow(), vec![false]);
    }

    #[test]
    fn set_enabled_replays_to_every_registered_cell() {
        let mut column = Column::<TestField, TestTableState>::new();
        let first_calls = Rc::new(RefCell::new(Vec::new()));
        let second_calls = Rc::new(RefCell::new(Vec::new()));
        column.add(Box::new(TestCell {
            state: CellState::default(),
            enabled: first_calls.clone(),
        }));
        column.add(Box::new(TestCell {
            state: CellState::default(),
            enabled: second_calls.clone(),
        }));
        column.set_enabled(false);

        assert!(!column.enabled);
        assert_eq!(column.list.len(), 2);
        assert_eq!(*first_calls.borrow(), vec![true, false]);
        assert_eq!(*second_calls.borrow(), vec![true, false]);
    }
}
