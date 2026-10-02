//! `IMOD/Etomo/src/etomo/type/ActionElement.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ActionElement`.
pub trait ActionElement {
    /// Java `getActionCommand()`.
    fn get_action_command(&self) -> Option<String>;
}
