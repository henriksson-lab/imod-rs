//! `IMOD/Etomo/src/etomo/ui/swing/MenuItem.java`.
#![allow(dead_code)]

use crate::imod::etomo::{
    etomo_director::ARGUMENTS,
    storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR},
    util::utilities,
};

/// Java package-private final `MenuItem` plus source-observable inherited
/// `JMenuItem` state.  Swing dispatch remains a native GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuItem {
    pub action_command: String,
    pub enabled: bool,
    pub visible: bool,
    pub selected: bool,
    pub text: Option<String>,
    pub name: Option<String>,
    pub mnemonic: Option<i32>,
    pub action_listener_count: usize,
}

impl MenuItem {
    /// Java `MenuItem()`.
    pub fn empty() -> Self {
        Self {
            action_command: String::new(),
            enabled: true,
            visible: true,
            selected: false,
            text: None,
            name: None,
            mnemonic: None,
            action_listener_count: 0,
        }
    }

    /// Java `MenuItem(String)`.
    pub fn new(text: &str) -> Self {
        let mut value = Self::empty();
        value.set_text(text);
        value
    }

    /// Java `MenuItem(String, int)`.
    pub fn with_mnemonic(text: &str, mnemonic: i32) -> Self {
        let mut value = Self::new(text);
        value.mnemonic = Some(mnemonic);
        value
    }

    /// Java overridden `setText(String)`.
    pub fn set_text(&mut self, text: &str) {
        self.text = Some(text.into());
        self.action_command = text.into();
        self.set_name(text);
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, text: &str) {
        let name = utilities::convert_label_to_name(Some(text), true).unwrap_or_default();
        self.name = Some(format!("mn{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.name.as_deref().unwrap_or_default()
            );
        }
    }

    /// Native `JMenuItem.addActionListener` boundary used by source consumers.
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
}

impl Default for MenuItem {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_and_rename_match_source_wrappers() {
        let mut item = MenuItem::with_mnemonic("Open File", 79);
        assert_eq!(item.action_command, "Open File");
        assert_eq!(item.name.as_deref(), Some("mn.open-file"));
        assert_eq!(item.mnemonic, Some(79));
        item.set_name("Different");
        assert_eq!(item.text.as_deref(), Some("Open File"));
        assert_eq!(item.name.as_deref(), Some("mn.different"));
    }
}
