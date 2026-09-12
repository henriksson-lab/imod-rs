//! `IMOD/Etomo/src/etomo/ui/swing/IterationParent.java`.
#![allow(dead_code)]

/// Java `IterationParent`.
pub trait IterationParent {
    /// Java `updateDisplay(boolean)`.
    fn update_display(&mut self, init: bool);

    /// Java `isSampleSphere`.
    fn is_sample_sphere(&self) -> bool;
}
