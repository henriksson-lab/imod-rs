//! `IMOD/Etomo/src/etomo/ui/swing/ParallelProgressDisplay.java`.
#![allow(dead_code)]

use std::collections::BTreeMap;

/// Java `ParallelProgressDisplay`, including its inherited `ActionListener`
/// callback.  The event payload is not read by Java `ProcessorTable`, so the
/// translated action callback retains that exact no-payload surface.
pub trait ParallelProgressDisplay {
    /// Java `ActionListener.actionPerformed(ActionEvent)`.
    fn action_performed(&mut self);
    fn msg_dropped(&mut self, computer: &str, reason: &str);
    fn add_success(&mut self, computer: &str);
    fn add_restart(&mut self, computer: &str);
    fn msg_killing_process(&mut self);
    fn msg_pausing_process(&mut self);
    fn msg_starting_process_on_selected_computers(&mut self);
    fn msg_ending_process(&mut self);
    fn reset_results(&mut self);
    fn set_computer_map(&mut self, computer_map: &BTreeMap<String, String>);
    fn set_secondary_queue(&mut self, secondary_queue: Option<&str>);
    fn msg_process_started(&mut self);
    fn is_secondary(&self) -> bool;
    fn is_runnable(&self) -> bool;
    fn is_limited(&self) -> bool;
}
