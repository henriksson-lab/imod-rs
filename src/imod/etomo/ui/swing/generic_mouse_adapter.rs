//! `IMOD/Etomo/src/etomo/ui/swing/GenericMouseAdapter.java`.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use super::context_menu::ContextMenu;
use super::context_popup::MouseEvent;

/// Java final `GenericMouseAdapter`.
pub struct GenericMouseAdapter {
    pub adaptee: Rc<RefCell<dyn ContextMenu>>,
}

impl GenericMouseAdapter {
    /// Java `GenericMouseAdapter(ContextMenu)`.
    pub fn new(adaptee: Rc<RefCell<dyn ContextMenu>>) -> Self {
        Self { adaptee }
    }

    /// Java `mouseClicked(MouseEvent)`.
    pub fn mouse_clicked(&self, _event: MouseEvent) {}

    /// Java `mousePressed(MouseEvent)`.
    pub fn mouse_pressed(&self, event: MouseEvent) {
        if event.right_mouse_button {
            self.adaptee.borrow_mut().pop_up_context_menu(event);
        }
    }

    /// Java `mouseReleased(MouseEvent)`.
    pub fn mouse_released(&self, _event: MouseEvent) {}

    /// Java `mouseEntered(MouseEvent)`.
    pub fn mouse_entered(&self, _event: MouseEvent) {}

    /// Java `mouseExited(MouseEvent)`.
    pub fn mouse_exited(&self, _event: MouseEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Menu(Option<MouseEvent>);
    impl ContextMenu for Menu {
        fn pop_up_context_menu(&mut self, event: MouseEvent) {
            self.0 = Some(event);
        }
    }

    #[test]
    fn only_right_press_reaches_the_adaptee() {
        let menu = Rc::new(RefCell::new(Menu::default()));
        let adapter = GenericMouseAdapter::new(menu.clone());
        adapter.mouse_pressed(MouseEvent {
            x: 1,
            y: 2,
            right_mouse_button: false,
        });
        assert_eq!(menu.borrow().0, None);
        adapter.mouse_pressed(MouseEvent {
            x: 1,
            y: 2,
            right_mouse_button: true,
        });
        assert_eq!(
            menu.borrow().0.map(|event| (event.x, event.y)),
            Some((1, 2))
        );
    }
}
