//! `IMOD/Etomo/src/etomo/ui/swing/MenuButton.java`.
#![allow(dead_code)]

use super::{
    context_menu::{ContextMenu, MouseEvent},
    menu_button_container::MenuButtonContainer,
    menu_item::MenuItem,
    multi_line_button::MultiLineButton,
};
use crate::imod::etomo::r#type::dialog_type::DialogType;

pub const MENU_STRING: &str = "Run To";

/// Direct `etomo.type.ActionElement` dependency boundary.  Its action command
/// is the only member reached by this Java source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionElement {
    pub action_command: String,
}
impl ActionElement {
    pub fn new(action_command: &str) -> Self {
        Self {
            action_command: action_command.into(),
        }
    }
    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> &str {
        &self.action_command
    }
}

/// Source-observable `JPopupMenu` state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JPopupMenuBoundary {
    pub visible: bool,
    pub position: Option<(i32, i32)>,
    pub items: Vec<MenuItem>,
}

/// Java package-private final `MenuButton`.
#[derive(Clone, Debug, PartialEq)]
pub struct MenuButton {
    pub multi_line_button: MultiLineButton,
    pub container_attached: bool,
    pub context_menu: Option<JPopupMenuBoundary>,
    pub menu_item_array: Option<Vec<MenuItem>>,
    pub action_element_array: Option<Vec<ActionElement>>,
    pub generic_mouse_adapter_attached: bool,
    pub menu_action_listener_attached: bool,
    pub last_container_action: Option<(String, ActionElement)>,
}

impl MenuButton {
    /// Java `MenuButton(String, boolean, DialogType)`.
    pub fn new(label: &str, toggle_button: bool, dialog_type: Option<DialogType>) -> Self {
        Self {
            multi_line_button: MultiLineButton::new_full(
                Some(label),
                toggle_button,
                dialog_type,
                false,
                false,
                false,
                None,
            ),
            container_attached: false,
            // Java declares this `null`; retaining that direct source state is
            // important because `addMenu` has no constructor for it.
            context_menu: None,
            menu_item_array: None,
            action_element_array: None,
            generic_mouse_adapter_attached: false,
            menu_action_listener_attached: true,
            last_container_action: None,
        }
    }

    /// Java static `getToggleMenuButtonInstance(String, DialogType)`.
    pub fn get_toggle_menu_button_instance(label: &str, dialog_type: Option<DialogType>) -> Self {
        Self::new(label, true, dialog_type)
    }

    /// Java `addMenu(MenuButtonContainer, ActionElement[])`.
    pub fn add_menu(&mut self, action_element_array: Option<Vec<ActionElement>>) {
        if self.container_attached {
            return;
        }
        self.multi_line_button.add_mouse_listener();
        self.generic_mouse_adapter_attached = true;
        self.container_attached = true;
        self.action_element_array = Some(action_element_array.unwrap_or_default());
        let elements = self.action_element_array.as_ref().unwrap();
        let mut items = Vec::with_capacity(elements.len());
        for element in elements {
            let mut item =
                MenuItem::new(&format!("{MENU_STRING} {}", element.get_action_command()));
            item.add_action_listener();
            // Java immediately invokes `contextMenu.add`.  Its field has no
            // source initializer, so a native frontend must install the popup
            // boundary before this method; preserve that direct null failure
            // instead of creating an invisible replacement popup.
            self.context_menu
                .as_mut()
                .expect("MenuButton.contextMenu is null")
                .items
                .push(item.clone());
            items.push(item);
        }
        self.menu_item_array = Some(items);
    }

    /// Native frontend installation of Java's otherwise-null `JPopupMenu` field.
    pub fn set_context_menu_boundary(&mut self, context_menu: Option<JPopupMenuBoundary>) {
        self.context_menu = context_menu;
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        if let Some(context_menu) = &mut self.context_menu {
            context_menu.position = Some((mouse_event.x, mouse_event.y));
            context_menu.visible = true;
        }
    }

    /// Java `action(ActionEvent)`, represented by its source action command.
    pub fn action(&mut self, command: &str) {
        if !self.container_attached {
            return;
        }
        if let Some(elements) = &self.action_element_array {
            for element in elements {
                if element.get_action_command() == command {
                    self.last_container_action = Some((
                        self.multi_line_button
                            .get_action_command()
                            .unwrap_or_default()
                            .into(),
                        element.clone(),
                    ));
                }
            }
        }
    }

    /// Java private `MenuActionListener.actionPerformed(ActionEvent)`.
    pub fn menu_action_listener_action_performed(&mut self, command: &str) {
        self.action(command);
    }

    /// Explicit native container invocation after Java's listener crosses the
    /// ownership boundary.
    pub fn dispatch_container_action(&mut self, container: &mut dyn MenuButtonContainer) {
        if let Some((command, element)) = self.last_container_action.take() {
            container.action(&command, &element);
        }
    }
}

impl ContextMenu for MenuButton {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        MenuButton::pop_up_context_menu(self, mouse_event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Target(Option<(String, String)>);
    impl MenuButtonContainer for Target {
        fn action(&mut self, command: &str, action_element: &ActionElement) {
            self.0 = Some((command.into(), action_element.action_command.clone()));
        }
    }
    #[test]
    fn menu_items_and_action_keep_source_commands() {
        let mut button = MenuButton::get_toggle_menu_button_instance("Run", None);
        button.set_context_menu_boundary(Some(JPopupMenuBoundary::default()));
        button.add_menu(Some(vec![ActionElement::new("Next")]));
        assert_eq!(
            button.context_menu.as_ref().unwrap().items[0]
                .text
                .as_deref(),
            Some("Run To Next")
        );
        button.menu_action_listener_action_performed("Next");
        let mut target = Target(None);
        button.dispatch_container_action(&mut target);
        assert_eq!(target.0, Some(("Run".into(), "Next".into())));
    }
    #[test]
    fn popup_uses_exact_mouse_position() {
        let mut button = MenuButton::new("Run", false, None);
        button.set_context_menu_boundary(Some(JPopupMenuBoundary::default()));
        button.pop_up_context_menu(MouseEvent {
            x: 4,
            y: 9,
            right_mouse_button: true,
        });
        assert_eq!(button.context_menu.unwrap().position, Some((4, 9)));
    }
}
