//! `IMOD/Etomo/src/etomo/ui/swing/ContextMenu.java`.
//!
//! The Java source is a one-method interface for right-mouse-button popup
//! requests. The native event remains a GUI boundary: this contract only
//! carries it to the translated receiver and does not synthesize an event or
//! a popup implementation.
#![allow(dead_code)]

pub use super::context_popup::MouseEvent;

/// Java `ContextMenu`.
///
/// Implementors receive the exact mouse event passed by Swing's mouse adapter.
/// Showing the resulting popup remains the concrete GUI implementation's
/// responsibility.
pub trait ContextMenu {
    /// Java `popUpContextMenu(MouseEvent)`.
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent);
}

#[cfg(test)]
mod tests {
    use super::{ContextMenu, MouseEvent};

    #[derive(Default)]
    struct TestContextMenu {
        received: Option<MouseEvent>,
    }

    impl ContextMenu for TestContextMenu {
        fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
            self.received = Some(mouse_event);
        }
    }

    #[test]
    fn popup_request_passes_the_swing_event_unchanged() {
        let mouse_event = MouseEvent { x: 17, y: 29 };
        let mut menu = TestContextMenu::default();

        menu.pop_up_context_menu(mouse_event);

        assert_eq!(menu.received, Some(mouse_event));
    }
}
