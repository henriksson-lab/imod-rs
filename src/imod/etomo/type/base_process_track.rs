//! `IMOD/Etomo/src/etomo/type/BaseProcessTrack.java`.

use super::axis_id::AxisID;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java interface `BaseProcessTrack extends Storable`.
pub trait BaseProcessTrack: Storable {
    /// Java `getRevisionNumber`.
    fn get_revision_number(&self) -> String;

    /// Java `isModified`.
    fn is_modified(&self) -> bool;

    /// Java `resetModified`.
    fn reset_modified(&self);

    /// Java `setState(ProcessState, AxisID, AbstractParallelDialog)`.
    fn set_state(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        parallel_dialog: &dyn AbstractParallelDialog,
    );
}
