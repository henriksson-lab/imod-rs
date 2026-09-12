//! `IMOD/Etomo/src/etomo/ui/swing/SpacedTextField.java`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

use super::fixed_dim::FixedDim;
use super::labeled_text_field::FieldValidationFailedException;

/// Java package-private final `SpacedTextField`, including its three widgets.
#[derive(Clone, Debug, PartialEq)]
pub struct SpacedTextField {
    pub text: String,
    pub text_field_name: String,
    pub text_field_enabled: bool,
    pub text_field_alignment_x: f32,
    pub label: String,
    pub label_alignment_x: f32,
    pub field_type: FieldType,
    pub text_field_tooltip: Option<String>,
    pub label_tooltip: Option<String>,
    pub field_panel_tooltip: Option<String>,
    pub y_axis_panel_tooltip: Option<String>,
    pub field_panel_alignment_x: f32,
    pub y_axis_panel_alignment_x: f32,
    pub y_axis_panel_visible: bool,
    pub field_panel_rigid_areas: [(i32, i32); 3],
    pub y_axis_rigid_area: (i32, i32),
    pub key_listener_count: usize,
}

impl SpacedTextField {
    /// Java `SpacedTextField(FieldType, String)`.
    pub fn new(field_type: FieldType, label: &str) -> Self {
        let name = utilities::convert_label_to_name(Some(label), true).unwrap_or_default();
        let text_field_name = format!("tf{SEPARATOR_CHAR}{name}");
        if ARGUMENTS.lock().expect("arguments lock").is_print_names() {
            println!("{text_field_name} {} ", DEFAULT_DELIMITER);
        }
        Self {
            text: String::new(),
            text_field_name,
            text_field_enabled: true,
            text_field_alignment_x: 0.5,
            label: label.trim().to_owned(),
            label_alignment_x: 0.5,
            field_type,
            text_field_tooltip: None,
            label_tooltip: None,
            field_panel_tooltip: None,
            y_axis_panel_tooltip: None,
            field_panel_alignment_x: 0.5,
            y_axis_panel_alignment_x: 0.5,
            y_axis_panel_visible: true,
            field_panel_rigid_areas: [
                (FixedDim::x5_y0.width, FixedDim::x5_y0.height),
                (FixedDim::x5_y0.width, FixedDim::x5_y0.height),
                (FixedDim::x5_y0.width, FixedDim::x5_y0.height),
            ],
            y_axis_rigid_area: (FixedDim::x0_y5.width, FixedDim::x0_y5.height),
            key_listener_count: 0,
        }
    }

    /// Java `setToolTipText(String)`; tooltip formatting remains the formatter boundary.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        let tooltip = text.map(str::to_owned);
        self.text_field_tooltip = tooltip.clone();
        self.label_tooltip = tooltip.clone();
        self.field_panel_tooltip = tooltip.clone();
        self.y_axis_panel_tooltip = tooltip;
    }

    /// Java `getContainer()`.
    pub fn get_container_is_y_axis_panel(&self) -> bool {
        true
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> &Self {
        self
    }

    /// Java `setText(ConstEtomoNumber)`.
    pub fn set_text_number(&mut self, number: &ConstEtomoNumber) {
        self.text = number.to_string();
    }
    /// Java overloaded `setText(String)`.
    pub fn set_text(&mut self, text: Option<&str>) {
        self.text = text.unwrap_or_default().to_owned();
    }
    /// Java overloaded `setText(int)`.
    pub fn set_text_int(&mut self, value: i32) {
        self.text = value.to_string();
    }
    /// Java overloaded `setText(double)`.
    pub fn set_text_double(&mut self, value: f64) {
        self.text = value.to_string();
    }
    /// Java overloaded `setText(long)`.
    pub fn set_text_long(&mut self, value: i64) {
        self.text = value.to_string();
    }
    /// Java `getLabel()`.
    pub fn get_label(&self) -> &str {
        &self.label
    }

    /// Java `getText(boolean)`; the `FieldValidator` call remains its direct
    /// validation boundary until that source unit is translated.
    pub fn get_text_validated(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        if do_validation && self.text_field_enabled {
            return Ok(self.text.clone());
        }
        Ok(self.text.clone())
    }
    /// Java overloaded `getText()`.
    pub fn get_text(&self) -> &str {
        &self.text
    }
    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> String {
        utilities::quote_label(Some(&self.label)).unwrap_or_default()
    }
    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.y_axis_panel_visible = visible;
    }
    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&mut self, alignment_x: f32) {
        self.text_field_alignment_x = alignment_x;
        self.label_alignment_x = alignment_x;
        self.field_panel_alignment_x = alignment_x;
        self.y_axis_panel_alignment_x = alignment_x;
    }
    /// Java `addKeyListener(KeyListener)` at the native listener boundary.
    pub fn add_key_listener(&mut self) {
        self.key_listener_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_names_raw_label_but_displays_trimmed_label() {
        let mut field = SpacedTextField::new(FieldType::Integer, "  Frames: ");
        field.set_text_int(3);
        field.set_alignment_x(0.0);
        assert_eq!(field.text_field_name, "tf.frames");
        assert_eq!(field.label, "Frames:");
        assert_eq!(field.get_text(), "3");
        assert_eq!(field.y_axis_panel_alignment_x, 0.0);
    }
}
