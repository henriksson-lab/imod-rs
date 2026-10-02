//! `IMOD/Etomo/src/etomo/process/ProcessInterface.java` and
//! `IMOD/Etomo/src/etomo/process/SystemProcessInterface.java`.
//!
//! A process object is shared by the thread running it, its monitor, the
//! manager and `AxisProcessData`, so the translation holds it as
//! `Arc<dyn SystemProcessInterface>` and every method takes `&self`.  Java
//! compares these references with `==`; [`same_process`] is that identity.

use super::process_data::ProcessData;
use crate::imod::etomo::process_series::ProcessSeries;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::util::event_queue::EdtRef;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A `ProcessResultDisplay` (a Swing button) as a process carries it: shared
/// with any thread through the `Arc`, dereferenced only on the event dispatch
/// thread through the `EdtRef`.
pub type ProcessResultDisplayRef = Arc<EdtRef<dyn ProcessResultDisplay>>;

/// A `ProcessSeries` as a process carries it; it lives on the event dispatch
/// thread with the dialogs it drives.
pub type ProcessSeriesRef = Arc<EdtRef<RefCell<ProcessSeries>>>;

/// Java `ProcessInterface`.
pub trait ProcessInterface: Send + Sync {
    /// Java `getProcessSeries`.
    fn get_process_series(&self) -> Option<ProcessSeriesRef>;
    /// Java `isNohup`.
    fn is_nohup(&self) -> bool;
    /// Java `getProcessData`.
    fn get_process_data(&self) -> Option<Arc<Mutex<ProcessData>>>;
    /// Java `pause`: return true if able to pause.
    fn pause(&self, axis_id: AxisID) -> bool;
    /// Java `kill`.
    fn kill(&self, axis_id: AxisID);
}

/// Java `SystemProcessInterface extends ProcessInterface`.
pub trait SystemProcessInterface: ProcessInterface {
    /// Java `toString`.
    fn to_source_string(&self) -> String;
    /// Java `getStdOutput`.
    fn get_std_output(&self) -> Option<Vec<String>>;
    /// Java `getStdError`.
    fn get_std_error(&self) -> Option<Vec<String>>;
    /// Java `isStarted`.
    fn is_started(&self) -> bool;
    /// Java `isDone`.
    fn is_done(&self) -> bool;
    /// Java `getShellProcessID`.
    fn get_shell_process_id(&self) -> String;
    /// Java `notifyKilled`.
    fn notify_killed(&self);
    /// Java `setProcessEndState`.
    fn set_process_end_state(&self, end_state: ProcessEndState);
    /// Java `signalKill`.
    fn signal_kill(&self, axis_id: AxisID);
    /// Java `setProcessResultDisplay`.
    fn set_process_result_display(&self, process_result_display: Option<ProcessResultDisplayRef>);
    /// Java `setComputerMap`.
    fn set_computer_map(&self, computer_map: Option<BTreeMap<String, String>>);
    /// Java `setSecondaryQueue`.
    fn set_secondary_queue(&self, secondary_queue: Option<&str>);
    /// Java `setProcessingMethod`.
    fn set_processing_method(&self, processing_method: Option<ProcessingMethod>);
    /// Java `resetProcessData`.
    fn reset_process_data(&self);
}

/// Java reference equality (`thread == script`) between two process objects.
pub fn same_process(a: &dyn ProcessInterface, b: &dyn ProcessInterface) -> bool {
    std::ptr::addr_eq(a as *const dyn ProcessInterface, b as *const dyn ProcessInterface)
}
