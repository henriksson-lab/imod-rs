//! `IMOD/Etomo/src/etomo/comscript/DetachedCommandDetails.java`.

use super::command_details::CommandDetails;

/// Java `DetachedCommandDetails extends CommandDetails`.  Its redeclared
/// `getCommandArray` is `Command`'s.
pub trait DetachedCommandDetails: CommandDetails {
    /// Java `getCommandString`.
    fn get_command_string(&self) -> Option<String>;
    /// Java `isValid`.
    fn is_valid(&self) -> bool;
    /// Java `isCommandNiced`.
    fn is_command_niced(&self) -> bool;
    /// Java `getNiceCommand`.
    fn get_nice_command(&self) -> Option<String>;
}
