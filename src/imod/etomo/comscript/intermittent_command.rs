//! `IMOD/Etomo/src/etomo/comscript/IntermittentCommand.java`.
//!
//! Interface for a param which can be used by `IntermittentSystemProgram`: a
//! connection is started (locally or on a remote computer), a command is sent
//! at an interval, and a generic end command closes the connection.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `IntermittentCommand`.  Java's `String[]`/`String` returns may be null;
/// they are `Option` here.
pub trait IntermittentCommand: Send + Sync {
    /// Java `getLocalStartCommand`.
    fn get_local_start_command(&self) -> Option<Vec<String>>;
    /// Java `getRemoteStartCommand`.
    fn get_remote_start_command(&self) -> Option<Vec<String>>;
    /// Java `getIntermittentCommand`.
    fn get_intermittent_command(&self) -> Option<String>;
    /// Java `getEndCommand`.
    fn get_end_command(&self) -> Option<String>;
    /// Java `getInterval`.
    fn get_interval(&self) -> i32;
    /// Java `getComputer`.
    fn get_computer(&self) -> Option<String>;
    /// Java `notifySentIntermittentCommand`.
    fn notify_sent_intermittent_command(&self) -> bool;
}
