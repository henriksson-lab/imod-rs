//! `IMOD/Etomo/src/etomo/ui/swing/MissingWedgeCompensationParent.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the methods take `&self`.

/// Java package-private `interface MissingWedgeCompensationParent`.
pub trait MissingWedgeCompensationParent {
    /// Java `isVolumeTableEmpty()`.
    fn is_volume_table_empty(&self) -> bool;

    /// Java `isReferenceParticleSelected()`.
    fn is_reference_particle_selected(&self) -> bool;

    /// Java `updateDisplay(boolean)`.
    fn update_display(&self, init: bool);
}
