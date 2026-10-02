//! `IMOD/Etomo/src/etomo/ui/swing/SimpleButton.java`.
//!
//! A `JButton` whose name follows its text (uitest name `bn.<label>`).
//!
//! Java `extends JButton`: the button is the `component` field.  Icons,
//! preferred sizes and widths are not modelled by `jdk.rs`, so
//! `getPreferredWidth` and `setToPreferredSize` have no Rust counterpart and
//! the icon constructors build a plain button.

use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::utilities;

use super::scaled_image::ScaledImage;

/// Java package-private final `SimpleButton`.
pub struct SimpleButton {
    /// The Java `JButton` this class extends.
    component: Rc<JComponent>,
}

impl SimpleButton {
    /// Java `SimpleButton()`: `super()`.
    pub fn new_void() -> Rc<SimpleButton> {
        Rc::new(SimpleButton {
            component: JComponent::new_button(""),
        })
    }

    /// Java `SimpleButton(String)`.
    pub fn new_string(text: Option<&str>) -> Rc<SimpleButton> {
        let instance = Rc::new(SimpleButton {
            component: JComponent::new_button(""),
        });
        // Java `super(text)`: `AbstractButton.init` calls the overridden
        // `setText(text)` when the text is not null, which also sets the name.
        if text.is_some() {
            instance.set_text(text);
        }
        instance.set_name(text);
        instance
    }

    /// Java `SimpleButton(Icon)`: `super(icon)`.  Icons are painting, not
    /// modelled.
    pub fn new_icon() -> Rc<SimpleButton> {
        Rc::new(SimpleButton {
            component: JComponent::new_button(""),
        })
    }

    /// Java `SimpleButton(ScaledImage)`.
    pub fn new_scaled_image(scaled_image: Option<&ScaledImage>) -> Rc<SimpleButton> {
        let instance = Rc::new(SimpleButton {
            component: JComponent::new_button(""),
        });
        if scaled_image.is_some() {
            // Swing painting: setIcon(new ImageIcon(scaledImage.getImage(this))).
        }
        instance
    }

    /// The Java `JButton` itself.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java package-private `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        super::ui_utilities::get_preferred_width_abstract_button_string(
            &self.component,
            Some(&self.component.get_text()),
        )
    }

    /// Java `setToPreferredSize()`: the preferred and maximum sizes are layout
    /// hints the stand-in does not keep.
    pub fn set_to_preferred_size(&self) {
        let _size = super::ui_utilities::get_preferred_size(&self.component, None);
    }

    /// Java `setText(String)` (overrides `AbstractButton.setText`).
    pub fn set_text(&self, text: Option<&str>) {
        // Java `super.setText(text)`; the stand-in holds no null text.
        self.component.set_text(text.unwrap_or(""));
        self.set_name(text);
    }

    /// Java `setName(String)` (overrides `Component.setName`).
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = UITestFieldType::BUTTON;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation writes a null name as "null".
        self.component.set_name(Some(&format!(
            "{}{}{}",
            field_type.to_string(),
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        // Java `EtomoDirector.INSTANCE.getArguments()` is the `ARGUMENTS` static.
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.component.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `getName()` (inherited from `Component`).
    pub fn get_name(&self) -> Option<String> {
        self.component.get_name()
    }
}
