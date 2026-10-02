//! `IMOD/Etomo/src/etomo/ui/swing/ParallelProgressDisplay.java`.
//!
//! Java `public interface ParallelProgressDisplay extends ActionListener`: the way a
//! parallel process monitor talks to the parallel processing table without knowing
//! anything about the GUI.  Implemented by `ProcessorTable` (every concrete table).
//!
//! The display is a Swing object living on the event dispatch thread.  A monitor
//! holds it as `Arc<EdtRef<dyn ParallelProgressDisplay>>` and posts its calls to the
//! EDT; every method takes `&self`.

use std::collections::HashMap;

use crate::imod::etomo::jdk::ActionEvent;

/// Java `ParallelProgressDisplay`.
pub trait ParallelProgressDisplay {
    /// Java `actionPerformed(ActionEvent)`, inherited from `ActionListener`.
    fn action_performed(&self, event: &ActionEvent);

    /// Java `msgDropped(String, String)`.
    fn msg_dropped(&self, computer: Option<&str>, reason: Option<&str>);

    /// Java `addSuccess(String)`.
    fn add_success(&self, computer: Option<&str>);

    /// Java `addRestart(String)`.
    fn add_restart(&self, computer: Option<&str>);

    /// Java `msgKillingProcess()`.
    fn msg_killing_process(&self);

    /// Java `msgPausingProcess()`.
    fn msg_pausing_process(&self);

    /// Java `msgStartingProcessOnSelectedComputers()`.
    fn msg_starting_process_on_selected_computers(&self);

    /// Java `msgEndingProcess()`.
    fn msg_ending_process(&self);

    /// Java `resetResults()`.
    fn reset_results(&self);

    /// Java `setComputerMap(Map<String, String>)`.  Used by reconnect.  Sets the
    /// computers and CPUs that where is use when the parallel process was last being
    /// tracked by Etomo.
    fn set_computer_map(&self, computer_map: Option<&HashMap<String, String>>);

    /// Java `setSecondaryQueue(String)`.
    fn set_secondary_queue(&self, secondary_queue: Option<&str>);

    /// Java `msgProcessStarted()`.
    fn msg_process_started(&self);

    /// Java `isSecondary()`.
    fn is_secondary(&self) -> bool;

    /// Java `isRunnable()`.
    fn is_runnable(&self) -> bool;

    /// Java `isLimited()`.
    fn is_limited(&self) -> bool;
}
