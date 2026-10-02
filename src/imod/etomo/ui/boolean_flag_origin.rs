//! `IMOD/Etomo/src/etomo/ui/BooleanFlagOrigin.java`.

/// Java `BooleanFlagOrigin`: a boolean field which generates a flagged state
/// (see `BooleanFlagExtension`).
pub trait BooleanFlagOrigin {
    /// Java `isSelected()`.
    fn is_selected(&self) -> bool;
}
