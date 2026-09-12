//! `IMOD/Etomo/src/etomo/ui/swing/Cell.java`.
//!
//! `JPanel`, `GridBagLayout`, and `GridBagConstraints` are native GUI
//! presentation objects.  This unit preserves the abstract Cell contract and
//! its table-state dispatch at that boundary; the Java `TableField` and
//! `TableState` interfaces are traits because their concrete implementations
//! belong to their respective source units.
#![allow(dead_code)]

/// Java `etomo.ui.TableField`, including Java's permitted `null` field value.
pub trait TableField {}

/// Java `etomo.logic.TableState`.
pub trait TableState<F: TableField> {
    /// Java `TableState.DEFAULT_GRIDWIDTH`.
    const DEFAULT_GRIDWIDTH: i32 = 1;

    /// Java `isDisplay(TableField)`.
    fn is_display(&self, table_field: Option<&F>) -> bool;

    /// Java `getGridwidth(TableField)`.
    fn get_gridwidth(&self, table_field: Option<&F>) -> i32;
}

/// Native `JPanel` handle used by Java `Cell.add`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CellPanelBoundary;

/// Native `GridBagLayout` handle used by Java `Cell.add`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CellGridBagLayoutBoundary;

/// Native `GridBagConstraints` handle used by Java `Cell.add`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CellGridBagConstraintsBoundary;

/// Java private `Cell.tableField` and `Cell.tableState` state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellState<F, S> {
    pub table_field: Option<F>,
    pub table_state: Option<S>,
}

impl<F, S> Default for CellState<F, S> {
    fn default() -> Self {
        Self {
            table_field: None,
            table_state: None,
        }
    }
}

/// Java package-private abstract `Cell`.
///
/// Rust models its abstract Java methods as required trait methods.  Concrete
/// cell source units retain a `CellState` and expose it through the two state
/// accessors, making the non-abstract methods below direct translations.
pub trait Cell<F: TableField, S: TableState<F>> {
    /// Java abstract `setEnabled(boolean)`.
    fn set_enabled(&mut self, enable: bool);

    /// Java abstract `msgLabelChanged()`.
    fn msg_label_changed(&mut self);

    /// Java abstract `add(JPanel, GridBagLayout, GridBagConstraints)`.
    fn add(
        &mut self,
        panel: &mut CellPanelBoundary,
        layout: &mut CellGridBagLayoutBoundary,
        constraints: &mut CellGridBagConstraintsBoundary,
    );

    /// Storage access corresponding to Java's private superclass fields.
    fn cell_state(&self) -> &CellState<F, S>;

    /// Mutable storage access corresponding to Java's private superclass fields.
    fn cell_state_mut(&mut self) -> &mut CellState<F, S>;

    /// Java `setTableState(TableField, TableState)`.
    fn set_table_state(&mut self, table_field: Option<F>, table_state: Option<S>) {
        let cell_state = self.cell_state_mut();
        cell_state.table_field = table_field;
        cell_state.table_state = table_state;
    }

    /// Java `isDisplay()`.
    fn is_display(&self) -> bool {
        let cell_state = self.cell_state();
        match cell_state.table_state.as_ref() {
            None => true,
            Some(table_state) => table_state.is_display(cell_state.table_field.as_ref()),
        }
    }

    /// Java `getGridwidth()`.
    fn get_gridwidth(&self) -> i32 {
        let cell_state = self.cell_state();
        match cell_state.table_state.as_ref() {
            None => S::DEFAULT_GRIDWIDTH,
            Some(table_state) => table_state.get_gridwidth(cell_state.table_field.as_ref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestField(i32);
    impl TableField for TestField {}

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestTableState;
    impl TableState<TestField> for TestTableState {
        fn is_display(&self, table_field: Option<&TestField>) -> bool {
            table_field.is_some_and(|field| field.0 > 0)
        }

        fn get_gridwidth(&self, table_field: Option<&TestField>) -> i32 {
            table_field.map_or(-1, |field| field.0 * 2)
        }
    }

    #[derive(Clone, Debug, Default, Eq, PartialEq)]
    struct TestCell {
        state: CellState<TestField, TestTableState>,
        enabled: bool,
        label_change_count: usize,
        add_count: usize,
    }
    impl Cell<TestField, TestTableState> for TestCell {
        fn set_enabled(&mut self, enable: bool) {
            self.enabled = enable;
        }

        fn msg_label_changed(&mut self) {
            self.label_change_count += 1;
        }

        fn add(
            &mut self,
            _panel: &mut CellPanelBoundary,
            _layout: &mut CellGridBagLayoutBoundary,
            _constraints: &mut CellGridBagConstraintsBoundary,
        ) {
            self.add_count += 1;
        }

        fn cell_state(&self) -> &CellState<TestField, TestTableState> {
            &self.state
        }

        fn cell_state_mut(&mut self) -> &mut CellState<TestField, TestTableState> {
            &mut self.state
        }
    }

    #[test]
    fn unset_table_state_uses_source_defaults() {
        let cell = TestCell::default();

        assert!(Cell::<TestField, TestTableState>::is_display(&cell));
        assert_eq!(Cell::<TestField, TestTableState>::get_gridwidth(&cell), 1);
    }

    #[test]
    fn table_state_receives_the_source_table_field() {
        let mut cell = TestCell::default();
        Cell::<TestField, TestTableState>::set_table_state(
            &mut cell,
            Some(TestField(3)),
            Some(TestTableState),
        );

        assert!(Cell::<TestField, TestTableState>::is_display(&cell));
        assert_eq!(Cell::<TestField, TestTableState>::get_gridwidth(&cell), 6);
    }

    #[test]
    fn table_state_observes_java_null_table_field() {
        let mut cell = TestCell::default();
        Cell::<TestField, TestTableState>::set_table_state(&mut cell, None, Some(TestTableState));

        assert!(!Cell::<TestField, TestTableState>::is_display(&cell));
        assert_eq!(Cell::<TestField, TestTableState>::get_gridwidth(&cell), -1);
    }
}
