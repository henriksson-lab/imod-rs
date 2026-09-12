//! `IMOD/Etomo/src/etomo/ui/swing/GlobalExpandButton.java`.
#![allow(dead_code)]

use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::multi_line_button::ButtonBoundary;
use super::panel::Dimension;
use super::single_line_button::SingleLineButton;
use std::{cell::RefCell, rc::Rc};

/// Java final `GlobalExpandButton`.  The `Rc<RefCell<_>>` entries retain the
/// mutable object references in the Java registration lists at the GUI boundary.
pub struct GlobalExpandButton {
    pub button: SingleLineButton,
    pub contracted_label: String,
    pub expanded_label: String,
    pub expand_button_list: Option<Vec<Rc<RefCell<ExpandButton>>>>,
    pub expandable_list: Option<Vec<Rc<RefCell<dyn Expandable>>>>,
    pub expanded: bool,
}

impl GlobalExpandButton {
    /// Java private `GlobalExpandButton(String, String)`.
    fn new(contracted_label: &str, expanded_label: &str) -> Self {
        Self {
            button: SingleLineButton::new_with_label(Some(contracted_label)),
            contracted_label: contracted_label.to_owned(),
            expanded_label: expanded_label.to_owned(),
            expand_button_list: None,
            expandable_list: None,
            expanded: false,
        }
    }
    /// Java static `getInstance(String, String)`.
    pub fn get_instance(contracted_label: &str, expanded_label: &str) -> Self {
        let mut instance = Self::new(contracted_label, expanded_label);
        instance.set_listeners();
        instance
    }
    /// Java private `setListeners()`.
    fn set_listeners(&mut self) {
        self.button.add_action_listener();
    }
    /// Java `register(ExpandButton)`.
    pub fn register_expand_button(&mut self, expand_button: Rc<RefCell<ExpandButton>>) {
        if self.expand_button_list.is_none() {
            self.expand_button_list = Some(Vec::new());
        }
        self.expand_button_list
            .as_mut()
            .unwrap()
            .push(expand_button);
    }
    /// Java `register(Expandable)`.
    pub fn register_expandable(&mut self, expandable: Rc<RefCell<dyn Expandable>>) {
        if self.expandable_list.is_none() {
            self.expandable_list = Some(Vec::new());
        }
        self.expandable_list.as_mut().unwrap().push(expandable);
    }
    /// Java `deregister(Expandable)`.
    pub fn deregister(&mut self, expandable: &Rc<RefCell<dyn Expandable>>) {
        if let Some(list) = self.expandable_list.as_mut() {
            list.retain(|registered| !Rc::ptr_eq(registered, expandable));
        }
    }
    /// Java `getComponent()`.
    pub fn get_component(&self) -> &ButtonBoundary {
        self.button.get_component()
    }
    /// Java `getPreferredSize()`.
    pub fn get_preferred_size(&self) -> Dimension {
        self.button.get_preferred_size()
    }
    /// Java `display()`.
    pub fn display(&mut self) {
        if !self.is_expanded() {
            self.button.do_click();
            self.action();
        }
    }
    /// Java `isExpanded()`.
    pub fn is_expanded(&self) -> bool {
        self.expanded
    }
    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.button.set_visible(visible);
    }
    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, tooltip: &str) {
        self.button.set_tool_tip_text(Some(tooltip));
    }
    /// Java `msgExpandButtonAction(ExpandButton, boolean)`.
    pub fn msg_expand_button_action(
        &mut self,
        active: &Rc<RefCell<ExpandButton>>,
        is_expanded: bool,
    ) {
        if self.expanded == is_expanded
            || self.expandable_list.is_some()
            || self.expand_button_list.is_none()
        {
            return;
        }
        for expand_button in self.expand_button_list.as_ref().unwrap() {
            if !Rc::ptr_eq(expand_button, active)
                && expand_button.borrow().is_expanded() == self.expanded
            {
                return;
            }
        }
        self.toggle_state();
    }
    /// Java private `toggleState()`.
    fn toggle_state(&mut self) {
        if self.expanded {
            self.expanded = false;
            self.button.set_text(&self.contracted_label);
        } else {
            self.expanded = true;
            self.button.set_text(&self.expanded_label);
        }
    }
    /// Java `changeState(boolean)`.
    pub fn change_state(&mut self, expanded: bool) {
        self.expanded = expanded;
        if expanded {
            self.button.set_text(&self.expanded_label);
        } else {
            self.button.set_text(&self.contracted_label);
        }
    }
    /// Java private `action()`.
    pub fn action(&mut self) {
        self.toggle_state();
        if let Some(list) = self.expandable_list.as_ref() {
            for expandable in list {
                expandable.borrow_mut().expand_global_button(self);
            }
        }
        if let Some(list) = self.expand_button_list.as_ref() {
            for expand_button in list {
                expand_button.borrow_mut().set_expanded(self.expanded);
            }
        }
    }
    /// Java inner listener `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        self.action();
    }
    /// Java `display(UIComponent)`.
    pub fn display_ui_component(&mut self) {
        self.display();
    }
}

#[cfg(test)]
mod tests {
    use super::super::expand_button::ExpandButtonType;
    use super::*;
    #[test]
    fn display_clicks_only_when_contracted() {
        let mut button = GlobalExpandButton::get_instance("Advanced", "Basic");
        button.display();
        assert!(button.expanded);
        assert_eq!(button.button.multi_line_button.get_text(), Some("Basic"));
        button.display();
        assert_eq!(button.button.multi_line_button.button.click_count, 1);
    }
    #[test]
    fn action_updates_registered_expand_buttons_in_order() {
        let first = Rc::new(RefCell::new(ExpandButton::new(
            ExpandButtonType::Advanced,
            false,
            false,
        )));
        let second = Rc::new(RefCell::new(ExpandButton::new(
            ExpandButtonType::Open,
            false,
            false,
        )));
        let mut button = GlobalExpandButton::get_instance("Advanced", "Basic");
        button.register_expand_button(first.clone());
        button.register_expand_button(second.clone());
        button.action_performed();
        assert!(first.borrow().is_expanded());
        assert!(second.borrow().is_expanded());
    }
}
