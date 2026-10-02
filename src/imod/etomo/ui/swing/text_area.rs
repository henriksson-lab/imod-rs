//! `IMOD/Etomo/src/etomo/ui/swing/TextArea.java`.
//!
//! Java `public class TextArea extends JTextArea`: a text area that names itself
//! (uitest `ta.`) from a reference label.  The `JTextArea` is
//! [`TextArea::get_component`]; the inherited `JTextArea` members the callers use
//! are called on it.  The rows/columns of `super(rows, columns)` size the area
//! (layout, not modelled) and are kept as fields.

use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::utilities;

/// Java `public class TextArea extends JTextArea`.
pub struct TextArea {
    /// The `JTextArea` this class extends.
    component: Rc<JComponent>,
    /// `JTextArea(rows, columns)`: the rows.
    pub rows: i32,
    /// `JTextArea(rows, columns)`: the columns.
    pub columns: i32,
}

impl TextArea {
    /// Java package-private `TextArea(String, int, int)`.
    pub fn new(reference: Option<&str>, rows: i32, columns: i32) -> Rc<TextArea> {
        // super(rows, columns)
        let instance = Rc::new(TextArea {
            component: JComponent::new_text_area(),
            rows,
            columns,
        });
        instance.set_name(reference);
        instance
    }

    /// The `JTextArea`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java `setName(String)` (override).
    pub fn set_name(&self, reference: Option<&str>) {
        let field_type = UITestFieldType::TEXT_AREA;
        let name = utilities::convert_label_to_name(reference, field_type.is_unlimited_segments());
        self.component.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().expect("arguments lock").is_print_names() {
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
    fn constructor_names_the_text_area() {
        let text_area = TextArea::new(Some("Long description:"), 4, 50);
        assert_eq!(text_area.rows, 4);
        assert_eq!(text_area.columns, 50);
        assert_eq!(
            text_area.get_component().get_name().as_deref(),
            Some("ta.long-description")
        );
    }
}
