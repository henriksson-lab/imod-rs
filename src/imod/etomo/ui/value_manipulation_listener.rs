//! `IMOD/Etomo/src/etomo/ui/ValueManipulationListener.java`.
//!
//! Interface for a value manipulator extension.

/// Java `ValueManipulationListener extends FocusListener`.  No implementer reads the
/// `FocusEvent`, so the inherited focus methods take no event; the field delivering
/// it calls the one the event's kind selects.
pub trait ValueManipulationListener {
    /// Java `FocusListener.focusGained(FocusEvent)`.
    fn focus_gained(&self);

    /// Java `FocusListener.focusLost(FocusEvent)`.
    fn focus_lost(&self);
}
