//! `IMOD/Etomo/src/etomo/ui/swing/RadialParent.java`.
#![allow(dead_code)]

/// Java `RadialParent`.
pub trait RadialParent {
    /// Java `isMultifilt`.
    fn is_multifilt(&self) -> bool;

    /// Java `isCtf3d`.
    fn is_ctf3d(&self) -> bool;

    /// Java `isAdvanced`.
    fn is_advanced(&self) -> bool;
}
