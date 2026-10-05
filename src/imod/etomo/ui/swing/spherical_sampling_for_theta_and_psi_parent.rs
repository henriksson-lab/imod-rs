//! `IMOD/Etomo/src/etomo/ui/swing/SphericalSamplingForThetaAndPsiParent.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the method takes `&self`.

/// Java package-private `interface SphericalSamplingForThetaAndPsiParent`.
pub trait SphericalSamplingForThetaAndPsiParent {
    /// Java `updateDisplay(boolean)`.
    fn update_display(&self, init: bool);
}
