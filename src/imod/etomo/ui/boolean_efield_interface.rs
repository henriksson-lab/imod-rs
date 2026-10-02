//! `IMOD/Etomo/src/etomo/ui/BooleanEfieldInterface.java`.
//!
//! Interface for a toggle field.  Implementers are EDT objects (`Rc`, `&self`
//! methods).

/// Java `BooleanEfieldInterface`.
pub trait BooleanEfieldInterface {
    /// Java `isSelected()`.
    fn is_selected(&self) -> bool;

    /// Java `setSelected(boolean)`.
    fn set_selected(&self, selected: bool);
}
