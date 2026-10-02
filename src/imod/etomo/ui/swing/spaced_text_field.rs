//! `IMOD/Etomo/src/etomo/ui/swing/SpacedTextField.java`.
//!
//! A labelled text field with spacing around it.  Box layouts and rigid areas are
//! Swing layout and recorded as comments.

use std::rc::Rc;

use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::logic::field_validator::FieldValidator;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `SpacedTextField`.
pub struct SpacedTextField {
    /// Java `textField`.
    text_field: Rc<JComponent>,
    /// Java `fieldPanel`.
    field_panel: Rc<JComponent>,
    /// Java `yAxisPanel`.
    y_axis_panel: Rc<JComponent>,
    /// Java `label`.
    label: Rc<JComponent>,
    /// Java `fieldType`.
    field_type: FieldType,
}

impl SpacedTextField {
    /// Java `SpacedTextField(FieldType, String)`.
    pub fn new(field_type: FieldType, label: &str) -> Rc<SpacedTextField> {
        let text_field = JComponent::new_text_field();
        let uitest_field_type = UITestFieldType::TEXT_FIELD;
        // set name
        let name = utilities::convert_label_to_name(
            Some(label),
            uitest_field_type.is_unlimited_segments(),
        );
        text_field.set_name(Some(&format!(
            "{}{}{}",
            uitest_field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                text_field.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
        let label = crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(label);
        let label = JComponent::new_label(label);
        // panels
        let y_axis_panel = JComponent::new_panel();
        let field_panel = JComponent::new_panel();
        // Swing layout: yAxisPanel BoxLayout Y_AXIS, fieldPanel BoxLayout X_AXIS.
        // fieldPanel: rigid area, label, rigid area, text field, rigid area.
        field_panel.add(&label);
        field_panel.add(&text_field);
        // yPanel: fieldPanel, rigid area.
        y_axis_panel.add(&field_panel);
        Rc::new(SpacedTextField {
            text_field,
            field_panel,
            y_axis_panel,
            label,
            field_type,
        })
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        let tooltip = tooltip_formatter::INSTANCE.format(text);
        self.text_field.set_tool_tip_text(tooltip.as_deref());
        self.label.set_tool_tip_text(tooltip.as_deref());
        self.field_panel.set_tool_tip_text(tooltip.as_deref());
        self.y_axis_panel.set_tool_tip_text(tooltip.as_deref());
    }

    /// Java `getContainer()`.  `yAxisPanel` is final and never null.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.y_axis_panel.clone()
    }

    /// Java `setText(ConstEtomoNumber)`.
    pub fn set_text_const_etomo_number(&self, number: &ConstEtomoNumber) {
        self.text_field.set_text(&number.to_string());
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        self.text_field.set_text(text.unwrap_or(""));
    }

    /// Java `setText(int)`.
    pub fn set_text_int(&self, value: i32) {
        self.text_field.set_text(&value.to_string());
    }

    /// Java `setText(double)`.
    pub fn set_text_double(&self, value: f64) {
        self.text_field.set_text(
            &crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(value),
        );
    }

    /// Java `setText(long)`.
    pub fn set_text_long(&self, value: i64) {
        self.text_field.set_text(&value.to_string());
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> String {
        self.label.get_text()
    }

    /// Java `getText(boolean)`.
    pub fn get_text_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut text = Some(self.text_field.get_text());
        if do_validation && self.text_field.is_enabled() {
            let descr = self.get_quoted_label();
            text = FieldValidator::validate_text_string_field_type_ui_component_string_boolean_boolean_boolean_validation_set_field_displayer_field_displayer(
                text.as_deref(),
                Some(self.field_type),
                Some(self),
                descr.as_deref(),
                false,
                false,
                false,
                None,
                None,
                None,
            )?;
        }
        Ok(text)
    }

    /// Java `getText()`.
    pub fn get_text_void(&self) -> String {
        self.text_field.get_text()
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(Some(&self.label.get_text()))
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_container().set_visible(visible);
    }

    /// Java `setAlignmentX(float)`: layout only.
    pub fn set_alignment_x(&self, _alignment_x: f32) {}

    /// Java `addKeyListener(KeyListener)`: key events are not modelled.
    pub fn add_key_listener<T>(&self, _listener: T) {}
}

impl SwingComponent for SpacedTextField {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.y_axis_panel.clone()
    }
}

impl UIComponent for SpacedTextField {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.y_axis_panel.clone()
    }
}
