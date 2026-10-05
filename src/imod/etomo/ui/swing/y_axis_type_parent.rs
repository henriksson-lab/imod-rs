//! `IMOD/Etomo/src/etomo/ui/swing/YAxisTypeParent.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the methods take `&self`.

/// Java package-private `interface YAxisTypeParent`.
pub trait YAxisTypeParent {
    /// Java `updateDisplay(boolean)`.
    fn update_display(&self, init: bool);

    /// Java `isVolumeTableEmpty()`.
    fn is_volume_table_empty(&self) -> bool;
}
