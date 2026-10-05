//! `IMOD/Etomo/src/etomo/ui/swing/IterationParent.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the methods take `&self`.

/// Java package-private `interface IterationParent`.
pub trait IterationParent {
    /// Java `updateDisplay(boolean)`.
    fn update_display(&self, init: bool);

    /// Java `isSampleSphere()`.
    fn is_sample_sphere(&self) -> bool;
}
