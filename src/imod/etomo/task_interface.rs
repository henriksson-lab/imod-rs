//! `IMOD/Etomo/src/etomo/TaskInterface.java`.
//!
//! A Java interface with two methods.
#![allow(dead_code)]

/// Java `TaskInterface`.
pub trait TaskInterface {
    /// Java `okToDrop`.  If true, then the user will be warned if they exit before the
    /// task is started.
    fn ok_to_drop(&self) -> bool;

    /// Java `getDescr`.
    fn get_descr(&self) -> Option<String>;
}
