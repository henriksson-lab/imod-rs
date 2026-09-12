//! `IMOD/Etomo/src/etomo/ui/swing/Menu.java`.
#![allow(dead_code)]

use crate::imod::etomo::{
    etomo_director::ARGUMENTS,
    storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR},
    util::utilities,
};

/// Source-observable `JMenu` state.  Native painting and menu hierarchy remain
/// at the Swing boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JMenuBoundary {
    pub text: Option<String>,
    pub name: Option<String>,
}

/// Java package-private final `Menu`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Menu {
    pub menu: JMenuBoundary,
}

impl Menu {
    /// Java `Menu(String)`.
    pub fn new(text: &str) -> Self {
        let mut value = Self::default();
        value.set_text(text);
        value
    }

    /// Java overridden `setText(String)`.
    pub fn set_text(&mut self, text: &str) {
        self.menu.text = Some(text.into());
        self.set_name(text);
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, text: &str) {
        let name = utilities::convert_label_to_name(Some(text), true).unwrap_or_default();
        self.menu.name = Some(format!("mn{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.menu.name.as_deref().unwrap_or_default()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_and_explicit_name_use_menu_item_test_name() {
        let mut menu = Menu::new("Old Menu");
        assert_eq!(menu.menu.name.as_deref(), Some("mn.old-menu"));
        menu.set_name("New Menu");
        assert_eq!(menu.menu.text.as_deref(), Some("Old Menu"));
        assert_eq!(menu.menu.name.as_deref(), Some("mn.new-menu"));
    }
}
