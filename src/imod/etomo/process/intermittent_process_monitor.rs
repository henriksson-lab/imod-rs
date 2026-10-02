//! `IMOD/Etomo/src/etomo/process/IntermittentProcessMonitor.java`.
//!
//! The monitor side of the many-to-many relationship with
//! `IntermittentBackgroundProcess`: a monitor reads the standard output of one or more
//! intermittent processes, and a process feeds one or more monitors.  Monitors are
//! shared between the process threads, their own thread and the event dispatch thread,
//! so the trait is `Send + Sync` and every method takes `&self`.
//!
//! Java identifies a monitor by object identity (it is the key of the process's
//! `monitors` list and the listener key of the output buffer); here that identity is
//! the address of the monitor object, `&*monitor as *const dyn
//! IntermittentProcessMonitor as *const ()` - the same for an `Arc` and for a `&self`
//! reference to the object inside it.

use std::sync::Arc;

use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::process::intermittent_background_process::IntermittentBackgroundProcess;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private interface `IntermittentProcessMonitor`.
pub trait IntermittentProcessMonitor: Send + Sync {
    /// Java `setProcess(IntermittentBackgroundProcess)`.
    fn set_process(&self, intermittent_background_process: Arc<IntermittentBackgroundProcess>);
    /// Java `msgIntermittentCommandFailed(IntermittentCommand)`.
    fn msg_intermittent_command_failed(&self, command: &dyn IntermittentCommand);
    /// Java `msgSentIntermittentCommand(IntermittentCommand)`.
    fn msg_sent_intermittent_command(&self, command: &dyn IntermittentCommand);
    /// Java `getOutputKeyPhrase()`.
    fn get_output_key_phrase(&self) -> Option<String>;
    /// Java `stop()`.
    fn stop(&self);
    /// Java `isMonitoring(IntermittentBackgroundProcess)`.
    fn is_monitoring(&self, program: &IntermittentBackgroundProcess) -> bool;
    /// Java `stopMonitoring(IntermittentBackgroundProcess)`.
    fn stop_monitoring(&self, program: &IntermittentBackgroundProcess);
}
