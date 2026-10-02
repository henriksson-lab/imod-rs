//! `IMOD/Etomo/src/etomo/ui/swing/GenericMouseAdapter.java`.
//!
//! Responds to pressed (not clicked) right mouse button events by asking the
//! adaptee to pop up its context menu.

use std::rc::{Rc, Weak};

use super::context_menu::ContextMenu;
use crate::imod::etomo::jdk::{self, MouseEvent, MouseListener};

/// Java `public final class GenericMouseAdapter implements MouseListener`.
pub struct GenericMouseAdapter {
    /// Java `private final ContextMenu adaptee`.  The adaptee owns the component
    /// this listener is registered on, so the back reference is weak.
    adaptee: Weak<dyn ContextMenu>,
}

impl GenericMouseAdapter {
    /// Java `GenericMouseAdapter(ContextMenu)`.
    pub fn new(adaptee: Weak<dyn ContextMenu>) -> Rc<GenericMouseAdapter> {
        Rc::new(GenericMouseAdapter { adaptee })
    }
}

impl MouseListener for GenericMouseAdapter {
    /// Java `mouseClicked(MouseEvent)`: empty.
    fn mouse_clicked(&self, _event: &MouseEvent) {}

    /// Java `mousePressed(MouseEvent)`.
    fn mouse_pressed(&self, event: &MouseEvent) {
        if jdk::is_right_mouse_button(event) {
            if let Some(adaptee) = self.adaptee.upgrade() {
                adaptee.pop_up_context_menu(event);
            }
        }
    }

    /// Java `mouseReleased(MouseEvent)`: empty.
    fn mouse_released(&self, _event: &MouseEvent) {}

    /// Java `mouseEntered(MouseEvent)`: empty.
    fn mouse_entered(&self, _event: &MouseEvent) {}

    /// Java `mouseExited(MouseEvent)`: empty.
    fn mouse_exited(&self, _event: &MouseEvent) {}
}
