//! `IMOD/Etomo/src/etomo/ui/swing/Label.java`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Java package-private final `Label` and its native `JLabel` state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Label {
    pub debug: bool,
    pub name: String,
    pub text: Option<String>,
    pub visible: bool,
}

impl Label {
    /// Java private `Label()`.
    fn empty() -> Self {
        Self {
            visible: true,
            ..Self::default()
        }
    }

    /// Java `Label(String)`.
    pub fn new(text: Option<&str>) -> Self {
        let mut value = Self::empty();
        if let Some(text) = text {
            value.set_name(Some(text));
        }
        value.text = text.map(str::to_owned);
        value
    }

    /// Java `Label(String, String)`.
    pub fn new_with_name(name: Option<&str>, text: Option<&str>) -> Self {
        let mut value = Self::empty();
        if let Some(name) = name {
            value.set_name(Some(name));
        }
        value.text = text.map(str::to_owned);
        value
    }

    /// Java static `getNamedInstance(String)`.
    pub fn get_named_instance(name: Option<&str>) -> Self {
        let mut value = Self::empty();
        if let Some(name) = name {
            value.set_name(Some(name));
        }
        value
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Java `setName(String)`.
    pub fn set_name(&mut self, name: Option<&str>) {
        let Some(name) = name else {
            self.name.clear();
            return;
        };
        let Some(name) = utilities::convert_label_to_name(Some(name), true) else {
            return;
        };
        self.name = format!("label{SEPARATOR_CHAR}{name}");
        if ARGUMENTS.lock().expect("arguments lock").is_print_names() {
            println!("{} {} ", self.name, DEFAULT_DELIMITER);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn name_and_text_follow_the_distinct_constructor_arguments() {
        let label = Label::new_with_name(Some("Rate:"), Some("Visible rate"));
        assert_eq!(label.name, "label.rate");
        assert_eq!(label.text.as_deref(), Some("Visible rate"));
        assert_eq!(Label::get_named_instance(None).name, "");
    }
}
