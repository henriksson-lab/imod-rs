//! `IMOD/Etomo/src/etomo/ui/swing/GlobalExpandButton.java`.
//!
//! A single-line button that expands or contracts a whole dialog: it tells
//! its registered `Expandable`s and `ExpandButton`s, and follows them when
//! every expand button has been toggled to the other state by hand.
//!
//! The registered expand buttons and expandables are held as `Weak`
//! references (each `ExpandButton` holds this button strongly, and an
//! expandable owns this button); a dropped one is skipped.  `getPreferredSize`
//! is layout and has no Rust counterpart.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::ui_component::UIComponent;

use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::single_line_button::SingleLineButton;

/// Java public final `GlobalExpandButton`.
pub struct GlobalExpandButton {
    /// This object, for the Java calls that pass `this`.
    self_ref: RefCell<Weak<GlobalExpandButton>>,
    /// Java final `button`.
    button: Rc<SingleLineButton>,
    /// Java final `contractedLabel`.
    contracted_label: Option<String>,
    /// Java final `expandedLabel`.
    expanded_label: Option<String>,
    /// Java `expandButtonList`.
    expand_button_list: RefCell<Option<Vec<Weak<ExpandButton>>>>,
    /// Java `expandableList`.
    expandable_list: RefCell<Option<Vec<Weak<dyn Expandable>>>>,
    /// Java `expanded`.
    expanded: Cell<bool>,
}

impl GlobalExpandButton {
    /// Java private `GlobalExpandButton(String, String)`.
    fn new(contracted_label: Option<&str>, expanded_label: Option<&str>) -> Rc<GlobalExpandButton> {
        let instance = Rc::new(GlobalExpandButton {
            self_ref: RefCell::new(Weak::new()),
            button: SingleLineButton::new_string(contracted_label),
            contracted_label: contracted_label.map(str::to_owned),
            expanded_label: expanded_label.map(str::to_owned),
            expand_button_list: RefCell::new(None),
            expandable_list: RefCell::new(None),
            expanded: Cell::new(false),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        instance
    }

    /// Java static `getInstance(String, String)`.
    pub fn get_instance(
        contracted_label: Option<&str>,
        expanded_label: Option<&str>,
    ) -> Rc<GlobalExpandButton> {
        let instance = GlobalExpandButton::new(contracted_label, expanded_label);
        instance.set_listeners();
        instance
    }

    /// Java private `setListeners()`.
    fn set_listeners(&self) {
        // Java `button.addActionListener(new GlobalExpandButtonActionListener(this))`.
        let adaptee = self.self_ref.borrow().clone();
        self.button.add_action_listener(Rc::new(move |_event| {
            if let Some(adaptee) = adaptee.upgrade() {
                // GlobalExpandButtonActionListener.actionPerformed
                adaptee.action();
            }
        }));
    }

    /// Java `register(ExpandButton)`.
    pub fn register_expand_button(&self, expand_button: &Rc<ExpandButton>) {
        self.expand_button_list
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(Rc::downgrade(expand_button));
    }

    /// Java `register(Expandable)`.
    pub fn register_expandable(&self, expandable: Weak<dyn Expandable>) {
        self.expandable_list
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(expandable);
    }

    /// Java `deregister(Expandable)`: `List.remove(Object)` removes the first
    /// occurrence.
    pub fn deregister(&self, expandable: &Weak<dyn Expandable>) {
        if let Some(expandable_list) = self.expandable_list.borrow_mut().as_mut() {
            if let Some(index) = expandable_list
                .iter()
                .position(|member| Weak::ptr_eq(member, expandable))
            {
                expandable_list.remove(index);
            }
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.get_button()
    }

    /// Java `display()`.
    pub fn display_void(&self) {
        if !self.is_expanded() {
            self.button.do_click();
        }
    }

    /// Java `isExpanded()`.
    pub fn is_expanded(&self) -> bool {
        self.expanded.get()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.button.set_visible(visible);
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        self.button.set_tool_tip_text(tooltip);
    }

    /// Java `msgExpandButtonAction(ExpandButton, boolean)`.
    pub fn msg_expand_button_action(&self, active_expand_button: &Rc<ExpandButton>, is_expanded: bool) {
        if self.expanded.get() == is_expanded
            || self.expandable_list.borrow().is_some()
            || self.expand_button_list.borrow().is_none()
        {
            // Same state as this button - nothing to do; or some advanced items only
            // appear when the global button is used, so the dialog can't be completely
            // expanded using ExpandButtons.
            return;
        }
        let expand_button_list = self.expand_button_list.borrow().clone().unwrap_or_default();
        for expand_button in expand_button_list.iter().filter_map(Weak::upgrade) {
            if !Rc::ptr_eq(&expand_button, active_expand_button) {
                if expand_button.is_expanded() == self.expanded.get() {
                    // Not all the expand buttons are a different state as this button
                    // - don't change this button.
                    return;
                }
            }
        }
        // All expand buttons have the state that is opposite to this button's state.
        // And all expandable fields in the dialog are controlled by expand buttons.
        // So change this button.
        self.toggle_state();
    }

    /// Java private `toggleState()`.
    fn toggle_state(&self) {
        if self.expanded.get() {
            self.expanded.set(false);
            self.button.set_text(self.contracted_label.as_deref());
        } else {
            self.expanded.set(true);
            self.button.set_text(self.expanded_label.as_deref());
        }
    }

    /// Java `changeState(boolean)`.
    pub fn change_state(&self, expanded: bool) {
        self.expanded.set(expanded);
        if expanded {
            self.button.set_text(self.expanded_label.as_deref());
        } else {
            self.button.set_text(self.contracted_label.as_deref());
        }
    }

    /// Java private `action()`.
    fn action(&self) {
        self.toggle_state();
        // This button was pressed.
        // Tell objects controlled by this button to expand.
        let expandable_list = self.expandable_list.borrow().clone();
        if let Some(expandable_list) = expandable_list {
            let this = self
                .self_ref
                .borrow()
                .upgrade()
                .expect("GlobalExpandButton used after drop");
            for expandable in expandable_list.iter().filter_map(Weak::upgrade) {
                expandable.expand_global_expand_button(&this);
            }
        }
        // Tell registered expand buttons to expand.
        let expand_button_list = self.expand_button_list.borrow().clone();
        if let Some(expand_button_list) = expand_button_list {
            for expand_button in expand_button_list.iter().filter_map(Weak::upgrade) {
                expand_button.set_expanded(self.expanded.get());
            }
        }
    }

    /// Java `display(UIComponent)`.
    pub fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        self.display_void();
    }
}

/// Java `implements FieldDisplayer`.
impl FieldDisplayer for GlobalExpandButton {
    fn display_void(&self) {
        GlobalExpandButton::display_void(self)
    }
    fn display_ui_component(&self, ui_component: Option<&dyn UIComponent>) {
        GlobalExpandButton::display_ui_component(self, ui_component)
    }
}
