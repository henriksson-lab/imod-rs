//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxMenuItem.java`: a self-naming
//! `JCheckBoxMenuItem`.
//!
//! `final class CheckBoxMenuItem extends JCheckBoxMenuItem` overrides `setText`
//! and `setName`.  The Swing superclass is the jdk stand-in [`JComponent`]
//! (`ComponentKind::CheckBoxMenuItem`), held in `component`; the inherited
//! Swing members are reached through [`CheckBoxMenuItem::get_component`].

use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::util::utilities;

/// Java `CheckBoxMenuItem`.
pub struct CheckBoxMenuItem {
    /// The `JCheckBoxMenuItem` this class extends.
    component: Rc<JComponent>,
}

impl CheckBoxMenuItem {
    /// Java `CheckBoxMenuItem()`: `super()`.
    pub fn new_void() -> Rc<CheckBoxMenuItem> {
        Rc::new(CheckBoxMenuItem {
            component: JComponent::new_check_box_menu_item(""),
        })
    }

    /// Java `CheckBoxMenuItem(String text)`: `super(text)`.  Swing's
    /// `AbstractButton.init` calls the (overridden) `setText` for a non-null text,
    /// so the item is named here.
    pub fn new_string(text: Option<&str>) -> Rc<CheckBoxMenuItem> {
        let instance = Rc::new(CheckBoxMenuItem {
            component: JComponent::new_check_box_menu_item(""),
        });
        if text.is_some() {
            instance.set_text(text);
        }
        instance
    }

    /// The Swing `JCheckBoxMenuItem` itself (Java `this` as a component).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java `setText(String)` (override).
    pub fn set_text(&self, text: Option<&str>) {
        self.component.set_text(text.unwrap_or(""));
        self.set_name(text);
    }

    /// Java `setName(String)` (override).
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = &ui_test_field_type::CHECK_BOX_MENU_ITEM;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        self.component.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.component.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }
}
