//! `IMOD/Etomo/src/etomo/ui/swing/MenuButtonContainer.java`.
#![allow(dead_code)]

use super::menu_button::ActionElement;

/// Java `MenuButtonContainer`.  The caller owns the concrete mutable receiver,
/// so this trait passes the source command and element across an explicit GUI
/// boundary rather than retaining an invalid Rust borrow.
pub trait MenuButtonContainer {
    /// Java `action(String, ActionElement)`.
    fn action(&mut self, command: &str, action_element: &ActionElement);
}
