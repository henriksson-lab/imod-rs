//! `IMOD/Etomo/src/etomo/ui/swing/SimpleButton.java`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

use super::multi_line_button::ButtonBoundary;
use super::ui_utilities::{Icon, UiUtilities};

/// Java final package-private `SimpleButton` with its native `JButton` boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleButton {
    pub button: ButtonBoundary,
    /// Java `ScaledImage.getImage(this)` / `ImageIcon` construction remains at
    /// the native image-presentation boundary.  This records its source image.
    pub scaled_image: Option<String>,
}

impl SimpleButton {
    /// Java `SimpleButton()`.
    pub fn new() -> Self {
        Self {
            button: ButtonBoundary::default(),
            scaled_image: None,
        }
    }

    /// Java `SimpleButton(String)`.
    pub fn new_with_text(text: Option<&str>) -> Self {
        let mut value = Self::new();
        value.button.text = text.map(str::to_owned);
        value.set_name(text);
        value
    }

    /// Java `SimpleButton(Icon)`.
    pub fn new_with_icon(icon: Option<Icon>) -> Self {
        let mut value = Self::new();
        value.button.icon = icon;
        value.button.abstract_button.icon = icon;
        value
    }

    /// Java `SimpleButton(ScaledImage)`.
    pub fn new_with_scaled_image(scaled_image: Option<&str>) -> Self {
        Self {
            button: ButtonBoundary::default(),
            scaled_image: scaled_image.map(str::to_owned),
        }
    }

    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        UiUtilities::get_preferred_width_button(
            &self.button.abstract_button,
            self.button.text.as_deref(),
        )
    }

    /// Java `setToPreferredSize()`.
    pub fn set_to_preferred_size(&mut self) {
        let size = UiUtilities::get_preferred_size(
            &self.button.abstract_button,
            self.button.text.as_deref(),
        );
        self.button.abstract_button.preferred_size = Some(size);
        self.button.abstract_button.maximum_size = Some(size);
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
    fn text_constructor_and_set_text_self_name_as_button() {
        let mut button = SimpleButton::new_with_text(Some("Open File"));
        assert_eq!(button.button.name.as_deref(), Some("bn.open-file"));
        button.set_text(Some("Run Process"));
        assert_eq!(button.button.name.as_deref(), Some("bn.run-process"));
    }

    #[test]
    fn preferred_size_is_applied_to_both_swing_constraints() {
        let mut button = SimpleButton::new_with_text(Some("Run"));
        button.set_to_preferred_size();
        assert_eq!(
            button.button.abstract_button.preferred_size,
            button.button.abstract_button.maximum_size
        );
    }
}
