//! `IMOD/Etomo/src/etomo/ui/swing/RadioButtonCell.java`.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use crate::imod::etomo::util::utilities;

use super::{
    cell::{CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary},
    check_box::Color,
    input_cell::InputCell,
    radio_button::{RadioButton, RadioButtonGroup},
    toggle_cell::ToggleCell,
};

/// Java final package-private `RadioButtonCell`.
#[derive(Clone, Debug)]
pub struct RadioButtonCell {
    pub input_cell: InputCell,
    pub radio_button: RadioButton,
    pub unformatted_label: String,
    pub background_refresh_count: usize,
    pub border_bottom: i32,
}

impl RadioButtonCell {
    /// Java private `RadioButtonCell(ButtonGroup,String)`.
    fn new(button_group: Option<Rc<RefCell<RadioButtonGroup>>>, reference: Option<&str>) -> Self {
        let mut radio_button = match button_group {
            Some(group) => RadioButton::new_in_group("", group),
            None => RadioButton::new(""),
        };
        radio_button.set_border_painted(true);
        radio_button.set_border(true);
        let mut value = Self {
            input_cell: InputCell::new(),
            radio_button,
            unformatted_label: String::new(),
            background_refresh_count: 0,
            border_bottom: 0,
        };
        value.set_background();
        value.set_foreground();
        value.set_font();
        if let Some(reference) = reference {
            value.set_name(reference);
        }
        value
    }
    /// Java `getInstance(ButtonGroup)`.
    pub fn get_instance(button_group: Option<Rc<RefCell<RadioButtonGroup>>>) -> Self {
        Self::new(button_group, None)
    }
    /// Java `getNamedInstance(ButtonGroup,String)`.
    pub fn get_named_instance(
        button_group: Option<Rc<RefCell<RadioButtonGroup>>>,
        header_label: &str,
    ) -> Self {
        Self::new(button_group, Some(header_label))
    }
    /// Java three-string `getNamedInstance` overload.
    pub fn get_named_three_instance(
        button_group: Option<Rc<RefCell<RadioButtonGroup>>>,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) -> Self {
        let reference = utilities::concatenate(reference1, reference2, reference3, Some(" "));
        Self::new(button_group, reference.as_deref())
    }
    /// Java overload `setName(String,String,String)`.
    pub fn set_name_three(
        &mut self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        if let Some(reference) =
            utilities::concatenate(reference1, reference2, reference3, Some(" "))
        {
            self.set_name(&reference);
        }
    }
    /// Java private `setName(String)`.
    pub fn set_name(&mut self, reference: &str) {
        self.radio_button.set_name(reference);
    }
    pub fn get_name(&self) -> Option<&str> {
        self.radio_button.get_name()
    }
    /// Java private `setForeground()`.
    fn set_foreground(&mut self) {
        let color = Color(0, 0, 0);
        self.radio_button.set_foreground(color);
        self.set_html_label(color);
    }
    /// Java private `setHtmlLabel(ColorUIResource)`.
    fn set_html_label(&mut self, color: Color) {
        self.radio_button.radio_button.text = format!(
            "<html><P style=\"font-weight:normal; color:rgb({},{},{})\">{}</style>",
            color.0, color.1, color.2, self.unformatted_label
        );
    }
    /// Java inherited `setBackground()`.
    fn set_background(&mut self) {
        self.background_refresh_count += 1;
    }
    /// Java inherited `setFont()`.
    fn set_font(&mut self) {
        self.input_cell.set_font("plain", "italic");
    }
    pub fn get_width(&self) -> i32 {
        self.radio_button
            .radio_button
            .preferred_size
            .map(|value| value.0)
            .unwrap_or(0)
    }
    pub fn get_text(&self) -> &str {
        &self.unformatted_label
    }
    pub fn get_height(&self) -> i32 {
        self.radio_button
            .radio_button
            .preferred_size
            .map(|value| value.1)
            .unwrap_or(0)
            + self.border_bottom
            - 1
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.radio_button.set_tool_tip_text(text);
    }
    pub fn get_field_type(&self) -> &'static str {
        "rb"
    }
    pub fn set_locked(&mut self, locked: bool) {
        self.radio_button.set_locked(locked);
        self.set_background();
    }
    pub fn is_locked(&self) -> bool {
        self.radio_button.is_locked()
    }
    pub fn set_editable(&mut self, editable: bool) {
        self.radio_button.set_editable(editable);
        self.set_background();
    }
    pub fn is_editable(&self) -> bool {
        self.radio_button.is_editable()
    }
}

impl ToggleCell for RadioButtonCell {
    fn get_label(&self) -> &str {
        &self.unformatted_label
    }
    fn set_label(&mut self, label: &str) {
        self.unformatted_label = label.into();
        self.set_foreground();
    }
    fn set_selected(&mut self, selected: bool) {
        self.radio_button.set_selected(selected);
    }
    fn add_action_listener(&mut self) {
        self.radio_button.add_action_listener();
    }
    fn add(
        &mut self,
        _: &mut CellPanelBoundary,
        _: &mut CellGridBagLayoutBoundary,
        _: &mut CellGridBagConstraintsBoundary,
    ) {
        self.input_cell.add();
    }
    fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }
    fn get_height(&self) -> i32 {
        RadioButtonCell::get_height(self)
    }
    fn get_width(&self) -> i32 {
        RadioButtonCell::get_width(self)
    }
    fn set_warning(&mut self, warning: bool) {
        self.input_cell.warning = warning;
        self.set_background();
    }
    fn add_change_listener(&mut self) {
        self.radio_button.add_change_listener();
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.radio_button.set_enabled(enabled);
        self.set_background();
    }
    fn is_enabled(&self) -> bool {
        self.radio_button.is_enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn label_is_html_colored_but_text_remains_unformatted() {
        let mut value = RadioButtonCell::get_instance(None);
        value.set_label("Queue");
        assert_eq!(value.get_text(), "Queue");
        assert!(value.radio_button.get_text().contains("Queue"));
    }
}
