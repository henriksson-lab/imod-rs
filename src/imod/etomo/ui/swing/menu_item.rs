//! `IMOD/Etomo/src/etomo/ui/swing/MenuItem.java`.
//!
//! A self-naming `JMenuItem`.  `final class MenuItem extends JMenuItem`: the
//! Swing part is the `JComponent` node in `component`, reached through `Deref`
//! (so the inherited `JMenuItem` members — `setEnabled`, `addActionListener`,
//! `getActionCommand`, `doClick`, ... — resolve), while the two overrides
//! (`setText`, `setName`) are inherent methods and win method resolution.

use std::ops::Deref;
use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Java `UITestFieldType.MENU_ITEM.toString()`.
// TODO(unit): needs etomo/type/UITestFieldType.java - `UITestFieldType.MENU_ITEM`
// ("mn", unlimitedSegments true) is written out here.
const MENU_ITEM_FIELD_TYPE: &str = "mn";
/// Java `UITestFieldType.MENU_ITEM.isUnlimitedSegments()`.
const MENU_ITEM_UNLIMITED_SEGMENTS: bool = true;

/// Java package-private `final class MenuItem extends JMenuItem`.
pub struct MenuItem {
    component: Rc<JComponent>,
}

impl Deref for MenuItem {
    type Target = Rc<JComponent>;
    fn deref(&self) -> &Rc<JComponent> {
        &self.component
    }
}

impl MenuItem {
    /// Java `MenuItem()`: `super()`.  `JMenuItem.init` calls `setText` only for
    /// non-null text, so no name is set.
    pub fn new_void() -> Rc<MenuItem> {
        Rc::new(MenuItem {
            component: JComponent::new_menu_item(""),
        })
    }

    /// Java `MenuItem(String)`: `super(text)`.  `JMenuItem.init` calls the
    /// overridden `setText`, which names the item.
    pub fn new_string(text: &str) -> Rc<MenuItem> {
        let item = Rc::new(MenuItem {
            component: JComponent::new_menu_item(""),
        });
        item.set_text(text);
        item
    }

    /// Java `MenuItem(String, int)`: `super(text, mnemonic)`.
    pub fn new_string_int(text: &str, mnemonic: i32) -> Rc<MenuItem> {
        let item = Rc::new(MenuItem {
            component: JComponent::new_menu_item(""),
        });
        item.set_text(text);
        // Swing key binding: `setMnemonic(mnemonic)`; key events are not modelled.
        let _ = mnemonic;
        item
    }

    /// The `JMenuItem` this class extends, as a `java.awt.Component`.
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
        let name = utilities::convert_label_to_name(text, MENU_ITEM_UNLIMITED_SEGMENTS);
        // Java string concatenation renders a null name as "null".
        self.component.set_name(Some(&format!(
            "{}{}{}",
            MENU_ITEM_FIELD_TYPE,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_names_the_item() {
        let item = MenuItem::new_string("Open...");
        assert_eq!(item.get_text(), "Open...");
        assert!(item.get_name().unwrap().starts_with("mn."));
        let empty = MenuItem::new_void();
        assert_eq!(empty.get_name(), None);
    }
}
