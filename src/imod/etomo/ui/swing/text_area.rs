//! `IMOD/Etomo/src/etomo/ui/swing/TextArea.java`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Java `UITestFieldType.TEXT_AREA.toString()`.
pub const TEXT_AREA_FIELD_TYPE: &str = "ta";

/// Java `TextArea`, including the `JTextArea(rows, columns)` state used here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextArea {
    pub rows: i32,
    pub columns: i32,
    pub name: String,
}

impl TextArea {
    /// Java package-private `TextArea(String, int, int)`.
    pub fn new(reference: &str, rows: i32, columns: i32) -> Self {
        let mut value = Self {
            rows,
            columns,
            name: String::new(),
        };
        value.set_name(reference);
        value
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, reference: &str) {
        let name = utilities::convert_label_to_name(Some(reference), false).unwrap_or_default();
        self.name = format!("{TEXT_AREA_FIELD_TYPE}{SEPARATOR_CHAR}{name}");
        if ARGUMENTS.lock().expect("arguments lock").is_print_names() {
            println!("{} {DEFAULT_DELIMITER} ", self.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_uses_text_area_name_and_dimensions() {
        let text_area = TextArea::new("Long description:", 4, 50);
        assert_eq!(text_area.rows, 4);
        assert_eq!(text_area.columns, 50);
        assert_eq!(text_area.name, "ta.long-description");
    }

    #[test]
    fn set_name_replaces_the_jtext_area_name() {
        let mut text_area = TextArea::new("Old", 1, 1);
        text_area.set_name("New label");
        assert_eq!(text_area.name, "ta.new-label");
    }
}
