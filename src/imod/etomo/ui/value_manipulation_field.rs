//! `IMOD/Etomo/src/etomo/ui/ValueManipulationField.java`.
//!
//! Interface for extensible fields that need their value modified.

use std::rc::Rc;

use super::value_manipulation_listener::ValueManipulationListener;

/// Java `ValueManipulationField`.  Implementers are EDT objects (`Rc`, `&self`
/// methods).
pub trait ValueManipulationField {
    /// Java `addValueManipulationListener(ValueManipulationListener)`.
    fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>);

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool;

    /// Java `setText(String)`.
    fn set_text(&self, text: Option<&str>);
}
