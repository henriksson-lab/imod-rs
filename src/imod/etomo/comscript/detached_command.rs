//! `IMOD/Etomo/src/etomo/comscript/DetachedCommand.java`.
//!
//! Command interface for a detached command.  Can create a safe command string
//! that can go into a run file.

use super::command::Command;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `DetachedCommand extends Command`.
pub trait DetachedCommand: Command {
    /// Java `getCommandString`.  Returns the command in a string which works,
    /// even if it contains directory paths with embedded spaces.
    fn get_command_string(&self) -> Option<String>;
    /// Java `isValid`.
    fn is_valid(&self) -> bool;
    /// Java `isSecondCommandLine`.
    fn is_second_command_line(&self) -> bool;
    /// Java `getSecondCommandLine`.
    fn get_second_command_line(&self) -> Option<String>;
}
