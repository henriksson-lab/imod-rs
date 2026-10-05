//! `IMOD/Etomo/src/etomo/ui/swing/Menu.java`.
//!
//! A self-naming `JMenu`.  `final class Menu extends JMenu`: the Swing part is
//! the `JComponent` node in `component`, reached through `Deref`; the two
//! overrides (`setText`, `setName`) are inherent methods.

use std::ops::Deref;
use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::utilities;

/// Java package-private `final class Menu extends JMenu`.
pub struct Menu {
    component: Rc<JComponent>,
}

impl Deref for Menu {
    type Target = Rc<JComponent>;
    fn deref(&self) -> &Rc<JComponent> {
        &self.component
    }
}

impl Menu {
    /// Java `Menu(String)`: `super(s)`.  `JMenuItem.init` calls the overridden
    /// `setText`, which names the menu.
    pub fn new(s: &str) -> Rc<Menu> {
        let menu = Rc::new(Menu {
            component: JComponent::new_menu(""),
        });
        menu.set_text(s);
        menu
    }

    /// The `JMenu` this class extends, as a `java.awt.Component`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java overridden `setText(String)`.
    pub fn set_text(&self, text: &str) {
        self.component.set_text(text);
        self.set_name(Some(text));
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = UITestFieldType::MENU_ITEM;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation renders a null name as "null".
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
