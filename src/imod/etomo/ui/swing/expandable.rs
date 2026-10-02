//! `IMOD/Etomo/src/etomo/ui/swing/Expandable.java`.
//!
//! A panel or dialog that shows or hides fields when an `ExpandButton` or a
//! `GlobalExpandButton` is pressed.  Buttons hold it as a
//! `Weak<dyn Expandable>`: the expandable owns its buttons, and usually hands
//! itself over while it is still being constructed.

use std::rc::Rc;

use super::expand_button::ExpandButton;
use super::global_expand_button::GlobalExpandButton;

/// Java `Expandable`.
pub trait Expandable {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>);

    /// Java `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>);
}
