//! `IMOD/Etomo/src/etomo/ui/swing/LoadDisplay.java`.
//!
//! Java `public interface LoadDisplay`: a display that can communicate with a load
//! monitor (`LoadAverageMonitor`, `QueuechunkLoadMonitor`) and show the load.
//! Implemented by `ProcessorTable` (every concrete table).  The display lives on
//! the event dispatch thread; every method takes `&self`.

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `LoadDisplay`.
pub trait LoadDisplay {
    /// Java `setLoad(String, double, double, int, String)`.
    fn set_load_string_double_double_int_string(
        &self,
        computer: Option<&str>,
        load1: f64,
        load5: f64,
        users: i32,
        users_tooltip: Option<&str>,
    );

    /// Java `msgLoadFailed(String, String, String)`.
    fn msg_load_failed(&self, computer: Option<&str>, reason: Option<&str>, tooltip: Option<&str>);

    /// Java `msgStartingProcess(String, String, String)`.
    fn msg_starting_process(
        &self,
        computer: Option<&str>,
        failure_reason1: Option<&str>,
        failure_reason2: Option<&str>,
    );

    /// Java `setCPUUsage(String, double, ConstEtomoNumber)`.
    fn set_cpu_usage(
        &self,
        computer: Option<&str>,
        cpu_usage: f64,
        number_of_processors: Option<&ConstEtomoNumber>,
    );

    /// Java `startLoad()`.
    fn start_load(&self);

    /// Java `stopLoad()`.
    fn stop_load(&self);

    /// Java `endLoad()`.
    fn end_load(&self);

    /// Java `setLoad(String, String[])`.
    fn set_load_string_string_array(&self, computer: Option<&str>, load_array: &[String]);
}
