//! `IMOD/Etomo/src/etomo/ui/swing/SimpleToggleButton.java`.
//!
//! A self-naming `JToggleButton` (bug# 1102).  Field names carry the `bn.` prefix
//! (bug# 1282).
//!
//! Java `extends JToggleButton`: the Swing button is held as `button`, and the two
//! overridden members (`setText`, `setName`) are methods here.  Everything else Java
//! callers invoke on the inherited `JToggleButton` goes through
//! [`SimpleToggleButton::get_component`].

use std::rc::Rc;

use super::swing_component::SwingComponent;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Java `final class SimpleToggleButton extends JToggleButton`.
pub struct SimpleToggleButton {
    /// The `JToggleButton` this class extends.
    button: Rc<JComponent>,
}

impl SimpleToggleButton {
    /// Java `SimpleToggleButton()`.  (`JToggleButton()` sets no text, so the
    /// overridden `setText` is not reached.)
    pub fn new_void() -> Rc<SimpleToggleButton> {
        Rc::new(SimpleToggleButton {
            button: JComponent::new_toggle_button(""),
        })
    }

    /// Java `SimpleToggleButton(String)`.  `super(text)` runs `AbstractButton.init`,
    /// which calls the overridden `setText` for a non-null text; then the constructor
    /// calls `setName(text)` itself.
    pub fn new_string(text: Option<&str>) -> Rc<SimpleToggleButton> {
        let instance = Rc::new(SimpleToggleButton {
            button: JComponent::new_toggle_button(""),
        });
        if let Some(text) = text {
            instance.set_text(text);
        }
        instance.set_name(text);
        instance
    }

    /// Java `@Override setText(String)`.
    pub fn set_text(&self, text: &str) {
        self.button.set_text(text);
        self.set_name(Some(text));
    }

    /// Java `@Override setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = &ui_test_field_type::BUTTON;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        self.button.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.button.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// The `JToggleButton` itself (Java uses `this` as the Component).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }
}

impl SwingComponent for SimpleToggleButton {
    fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }
}
