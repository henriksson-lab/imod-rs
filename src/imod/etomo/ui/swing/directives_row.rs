//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesRow.java`.
//!
//! `JPanel`, `GridBagLayout`, and `GridBagConstraints` are the native Swing
//! boundary.  This source unit is only the common row contract: it deliberately
//! does not add row state or dispatch policy beyond the three Java methods.

use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::cell::{CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary};

/// Java package-private `DirectivesRow`.
///
/// `display` receives the three native Swing objects used by the Java
/// interface.  Rust's mutable receiver records the same fact as Java's
/// component insertion/removal operations: a row may change its source-owned
/// display state while it is added to the table.
pub trait DirectivesRow {
    /// Java `display(JPanel, GridBagLayout, GridBagConstraints)`.
    fn display(
        &mut self,
        pnl_table: &mut CellPanelBoundary,
        layout: &mut CellGridBagLayoutBoundary,
        constraints: &mut CellGridBagConstraintsBoundary,
    );

    /// Java `remove()`.
    fn remove(&mut self);

    /// Java `statusChanged(BatchRunTomoStatus)`.
    fn status_changed(&mut self, status: BatchRunTomoStatus);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Row {
        displayed: bool,
        removed: bool,
        status: Option<BatchRunTomoStatus>,
    }

    impl DirectivesRow for Row {
        fn display(
            &mut self,
            _pnl_table: &mut CellPanelBoundary,
            _layout: &mut CellGridBagLayoutBoundary,
            _constraints: &mut CellGridBagConstraintsBoundary,
        ) {
            self.displayed = true;
        }

        fn remove(&mut self) {
            self.removed = true;
        }

        fn status_changed(&mut self, status: BatchRunTomoStatus) {
            self.status = Some(status);
        }
    }

    #[test]
    fn contract_preserves_display_remove_and_status_arguments() {
        let mut row = Row::default();
        let mut panel = CellPanelBoundary;
        let mut layout = CellGridBagLayoutBoundary;
        let mut constraints = CellGridBagConstraintsBoundary;

        row.display(&mut panel, &mut layout, &mut constraints);
        row.status_changed(BatchRunTomoStatus::Running);
        row.remove();

        assert!(row.displayed);
        assert!(row.removed);
        assert_eq!(row.status, Some(BatchRunTomoStatus::Running));
    }
}
