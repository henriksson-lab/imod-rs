//! `IMOD/Etomo/src/etomo/ui/swing/Label.java`.
//!
//! Java `final class Label extends JLabel`: the `JLabel` is [`Label::get_component`];
//! inherited `JLabel` members are called on it.

use std::cell::Cell;
use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Java `Label`.
pub struct Label {
    /// The `JLabel` this class extends.
    component: Rc<JComponent>,
    /// Java `debug`.
    debug: Cell<bool>,
}

impl Label {
    /// Java private `Label()`.
    fn new_void() -> Rc<Label> {
        Rc::new(Label {
            component: JComponent::new_label(""),
            debug: Cell::new(false),
        })
    }

    /// Java `Label(String)`.
    pub fn new_string(text: Option<&str>) -> Rc<Label> {
        let label = Label::new_void();
        if text.is_some() {
            label.set_name(text);
        }
        // Java `setText(text)`; the stand-in's text is never null.
        label.component.set_text(text.unwrap_or(""));
        label
    }

    /// Java `Label(String, String)`.
    pub fn new_string_string(name: Option<&str>, text: Option<&str>) -> Rc<Label> {
        let label = Label::new_void();
        if name.is_some() {
            label.set_name(name);
        }
        label.component.set_text(text.unwrap_or(""));
        label
    }

    /// Java static `getNamedInstance(String)`.
    pub fn get_named_instance(name: Option<&str>) -> Rc<Label> {
        let instance = Label::new_void();
        if name.is_some() {
            instance.set_name(name);
        }
        instance
    }

    /// The `JLabel` this class extends.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `setVisible(boolean)` (override that only calls super).
    pub fn set_visible(&self, visible: bool) {
        self.component.set_visible(visible);
    }

    /// Java `setName(String)` (override).
    pub fn set_name(&self, name: Option<&str>) {
        let Some(name) = name else {
            self.component.set_name(Some(""));
            return;
        };
        let field_type = &ui_test_field_type::LABEL;
        // build name
        let Some(name) = utilities::convert_label_to_name(Some(name), field_type.is_unlimited_segments())
        else {
            return;
        };
        self.component
            .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.component.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }
}
