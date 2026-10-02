//! `IMOD/Etomo/src/etomo/type/DialogExitState.java`.
//!
//! How a process dialog was left.  A Java class with four static instances compared by
//! identity, which a Rust enum expresses exactly.
//!
//! Note: the staged `ui/swing/process_dialog.rs` currently declares an identical
//! `DialogExitState` enum of its own; that module should re-export this one
//! (`pub use crate::imod::etomo::r#type::dialog_exit_state::DialogExitState;`).

/// Java `DialogExitState.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `DialogExitState.CANCEL_LABEL`.
pub const CANCEL_LABEL: &str = "Cancel";

/// Java `DialogExitState`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DialogExitState {
    /// Java `CANCEL`, named `CANCEL_LABEL`.
    Cancel,
    /// Java `POSTPONE`, named "Postpone".
    Postpone,
    /// Java `EXECUTE`, named "Execute".
    Execute,
    /// Java `SAVE`, named "Save".
    Save,
}

impl DialogExitState {
    /// Java `DialogExitState.CANCEL_LABEL`, also reachable through the type as the
    /// staged `process_dialog.rs` spells it.
    pub const CANCEL_LABEL: &'static str = CANCEL_LABEL;
}

/// Java `toString()`: the instance's name.
impl std::fmt::Display for DialogExitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DialogExitState::Cancel => CANCEL_LABEL,
            DialogExitState::Postpone => "Postpone",
            DialogExitState::Execute => "Execute",
            DialogExitState::Save => "Save",
        })
    }
}
