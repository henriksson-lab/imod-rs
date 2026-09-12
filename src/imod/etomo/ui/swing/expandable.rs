//! `IMOD/Etomo/src/etomo/ui/swing/Expandable.java`.
//!
//! The implementor owns the expanded content.  The borrowed button/global
//! button arguments keep Java's event boundary explicit without making a
//! Swing callback registry part of the translated model.
#![allow(dead_code)]

use super::expand_button::ExpandButton;
use super::process_dialog::GlobalExpandButton;

/// Java `Expandable`.
pub trait Expandable {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&mut self, button: &ExpandButton);

    /// Java `expand(GlobalExpandButton)`.
    fn expand_global_button(&mut self, button: &GlobalExpandButton);
}
