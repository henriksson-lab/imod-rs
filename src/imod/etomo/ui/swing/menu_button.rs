//! `IMOD/Etomo/src/etomo/ui/swing/MenuButton.java`.
//!
//! A `MultiLineButton` with a right-click "Run To" menu: one item per action element,
//! each reported to a `MenuButtonContainer`.  (Nothing in the Java constructs one
//! outside this class; it is translated whole.)
//!
//! Java `final class MenuButton extends MultiLineButton implements ContextMenu`: the
//! superclass is embedded as `base` (with `Deref`), and construction follows
//! `MultiLineButton`'s split (`new_fields`, then `construct`).  The popup is shown by
//! the Swing stand-in's popup layer (`JComponent::show`).

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::context_menu::ContextMenu;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::menu_button_container::MenuButtonContainer;
use super::menu_item::MenuItem;
use super::multi_line_button::{MultiLineButton, MultiLineButtonVirtual};
use crate::imod::etomo::jdk::{ActionEvent, JComponent, MouseEvent};
use crate::imod::etomo::r#type::action_element::ActionElement;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `MENU_STRING`.
const MENU_STRING: &str = "Run To";

/// Java package-private `final class MenuButton extends MultiLineButton implements
/// ContextMenu`.
pub struct MenuButton {
    /// Java superclass `MultiLineButton`.
    base: MultiLineButton,
    /// This object, for the listeners.
    self_ref: RefCell<Weak<MenuButton>>,
    /// Java `container`.
    container: RefCell<Option<Rc<dyn MenuButtonContainer>>>,
    /// Java `contextMenu`.
    context_menu: RefCell<Option<Rc<JComponent>>>,
    /// Java `menuItemArray`.
    menu_item_array: RefCell<Option<Vec<Rc<MenuItem>>>>,
    /// Java `actionElementArray`.
    action_element_array: RefCell<Option<Vec<Rc<dyn ActionElement>>>>,
    // Java `listener = new MenuActionListener(this)`: the closure registered on each
    // menu item in `add_menu`.
}

impl Deref for MenuButton {
    type Target = MultiLineButton;
    fn deref(&self) -> &MultiLineButton {
        &self.base
    }
}

impl MultiLineButtonVirtual for MenuButton {
    fn get_multi_line_button(&self) -> &MultiLineButton {
        &self.base
    }
}

impl MenuButton {
    /// Java `MenuButton(String, boolean, DialogType)`.  Creates the right click menu.
    pub fn new(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
    ) -> Rc<MenuButton> {
        // super(label, toggleButton, dialogType, false, false, false, null)
        let instance = Rc::new(MenuButton {
            base: MultiLineButton::new_fields(
                label,
                toggle_button,
                dialog_type,
                false,
                false,
                None,
            ),
            self_ref: RefCell::new(Weak::new()),
            container: RefCell::new(None),
            context_menu: RefCell::new(None),
            menu_item_array: RefCell::new(None),
            action_element_array: RefCell::new(None),
        });
        MultiLineButton::construct(&instance, false);
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        instance
    }

    /// Java static final `getToggleMenuButtonInstance(String, DialogType)`.
    pub fn get_toggle_menu_button_instance(
        label: Option<&str>,
        dialog_type: Option<DialogType>,
    ) -> Rc<MenuButton> {
        MenuButton::new(label, true, dialog_type)
    }

    /// Java `addMenu(MenuButtonContainer, ActionElement[])`.
    ///
    /// Upstream bugs fixed in translation (`MenuButton.java:77-93`): the Java never
    /// creates `contextMenu` (it stays null), so `contextMenu.add(...)` throws a
    /// NullPointerException for the first action element; here the popup menu is
    /// created on first use.  The Java also stores the caller's array in the field
    /// before replacing a null array with an empty one, so a null array made every
    /// later `action` throw; the empty array is stored instead.
    pub fn add_menu(
        &self,
        menu_button_container: Option<Rc<dyn MenuButtonContainer>>,
        action_element_array: Option<Vec<Rc<dyn ActionElement>>>,
    ) {
        if self.container.borrow().is_some() {
            return;
        }
        // addMouseListener(new GenericMouseAdapter(this))
        let this = self.self_ref.borrow().clone();
        self.get_component()
            .add_mouse_listener(GenericMouseAdapter::new(
                this.clone() as Weak<dyn ContextMenu>
            ));
        *self.container.borrow_mut() = menu_button_container;
        let action_element_array = action_element_array.unwrap_or_default();
        *self.action_element_array.borrow_mut() = Some(action_element_array.clone());
        if self.context_menu.borrow().is_none() {
            *self.context_menu.borrow_mut() = Some(JComponent::new_popup_menu(""));
        }
        let context_menu = self.context_menu.borrow().clone().unwrap();
        let mut menu_item_array = Vec::with_capacity(action_element_array.len());
        for action_element in action_element_array.iter() {
            let menu_item = MenuItem::new_string(&format!(
                "{} {}",
                MENU_STRING,
                action_element
                    .get_action_command()
                    .as_deref()
                    .unwrap_or("null")
            ));
            context_menu.add(&menu_item.get_component());
            // menuItemArray[i].addActionListener(listener): MenuActionListener
            let adaptee = this.clone();
            menu_item
                .get_component()
                .add_action_listener(Rc::new(move |event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event);
                    }
                }));
            menu_item_array.push(menu_item);
        }
        *self.menu_item_array.borrow_mut() = Some(menu_item_array);
    }

    /// Java public final `@Override popUpContextMenu(MouseEvent)`.
    ///
    /// Upstream bug fixed in translation (`MenuButton.java:97`): with no menu added
    /// `contextMenu` is null and the Java throws a NullPointerException; here nothing
    /// is shown.
    pub fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let context_menu = self.context_menu.borrow().clone();
        if let Some(context_menu) = context_menu {
            context_menu.show(&self.get_component(), mouse_event.x, mouse_event.y);
            context_menu.set_visible(true);
        }
    }

    /// Java final `action(ActionEvent)`.
    ///
    /// Upstream bug fixed in translation (`MenuButton.java:108`): the Java compares the
    /// menu item's command (its text, `"Run To " + element command`) with the bare
    /// element command, which never matches, so no menu choice reached the
    /// container.  The evident intent is to find the element whose menu item was
    /// chosen, so the command is compared with that item's text.
    pub fn action(&self, event: &ActionEvent) {
        let container = self.container.borrow().clone();
        if let Some(container) = container {
            let command = event.get_action_command();
            // Find the action element that matches the menu item and pass it to the
            // container.
            let action_element_array = self
                .action_element_array
                .borrow()
                .clone()
                .unwrap_or_default();
            for action_element in action_element_array.iter() {
                let menu_text = format!(
                    "{} {}",
                    MENU_STRING,
                    action_element
                        .get_action_command()
                        .as_deref()
                        .unwrap_or("null")
                );
                if command == Some(menu_text.as_str()) {
                    container.action(self.get_action_command().as_deref(), action_element);
                }
            }
        }
    }
}

impl ContextMenu for MenuButton {
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        MenuButton::pop_up_context_menu(self, mouse_event);
    }
}
