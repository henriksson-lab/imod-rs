//! `IMOD/Etomo/src/etomo/ui/swing/SimpleToggleButton.java`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

use super::multi_line_button::ButtonBoundary;

/// Java final package-private `SimpleToggleButton` with its `JToggleButton`
/// boundary state.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleToggleButton {
    pub button: ButtonBoundary,
}

impl SimpleToggleButton {
    /// Java `SimpleToggleButton()`.
    pub fn new() -> Self {
        Self {
            button: ButtonBoundary::default(),
        }
    }

    /// Java `SimpleToggleButton(String)`.
    pub fn new_with_text(text: Option<&str>) -> Self {
        let mut value = Self::new();
        value.button.text = text.map(str::to_owned);
        value.set_name(text);
        value
    }

    /// Java overridden `setText(String)`.
    pub fn set_text(&mut self, text: Option<&str>) {
        self.button.text = text.map(str::to_owned);
        self.set_name(text);
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, text: Option<&str>) {
        let name = utilities::convert_label_to_name(text, true);
        self.button.name = name.map(|name| format!("bn{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.button.name.as_deref().unwrap_or_default(),
                DEFAULT_DELIMITER
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_assignment_rebuilds_button_ui_test_name() {
        let mut button = SimpleToggleButton::new();
        button.set_text(Some("Fine Alignment"));
        assert_eq!(button.button.name.as_deref(), Some("bn.fine-alignment"));
    }
}
