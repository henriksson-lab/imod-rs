//! `IMOD/Etomo/src/etomo/ui/swing/MenuButtonContainer.java`.

use std::rc::Rc;

use crate::imod::etomo::r#type::action_element::ActionElement;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `interface MenuButtonContainer`.
pub trait MenuButtonContainer {
    /// Java `action(String, ActionElement)`.
    fn action(&self, command: Option<&str>, action_element: &Rc<dyn ActionElement>);
}
