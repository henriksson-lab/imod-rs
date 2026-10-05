//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesRow.java`.
//!
//! A row of the directives table (`DirectivesTable`): a section header or a directive.

use std::rc::Rc;

use crate::imod::etomo::jdk::{GridBagConstraints, GridBagLayout, JComponent};
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;

/// Java package-private `interface DirectivesRow`.
pub trait DirectivesRow {
    /// Java `display(JPanel, GridBagLayout, GridBagConstraints)`.
    fn display(
        &self,
        pnl_table: &Rc<JComponent>,
        layout: &GridBagLayout,
        constraints: &mut GridBagConstraints,
    );

    /// Java `remove()`.
    fn remove(&self);

    /// Java `statusChanged(BatchRunTomoStatus)`.
    fn status_changed(&self, status: Option<BatchRunTomoStatus>);
}
