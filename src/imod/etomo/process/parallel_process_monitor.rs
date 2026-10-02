//! `IMOD/Etomo/src/etomo/process/ParallelProcessMonitor.java`.
//!
//! Interface for ProcesschunksProcessMonitor.  Allows another class to tell
//! monitor to drop a computer.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ParallelProcessMonitor`.
pub trait ParallelProcessMonitor: Send + Sync {
    /// Java `drop`.
    fn drop(&self, computer: &str);
}
