//! `IMOD/Etomo/src/etomo/ui/swing/ContextMenu.java`.
//!
//! Defines the popUpContextMenu interface for right mouse button events.

use crate::imod::etomo::jdk::MouseEvent;

/// Java `public interface ContextMenu`.
pub trait ContextMenu {
    /// Java `popUpContextMenu(MouseEvent)`.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent);
}
