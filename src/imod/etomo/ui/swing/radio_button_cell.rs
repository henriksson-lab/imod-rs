//! `IMOD/Etomo/src/etomo/ui/swing/RadioButtonCell.java`.
//!
//! Java `final class RadioButtonCell extends InputCell implements ToggleCell`: a
//! table cell holding a radio button.  Every Java method body is an inherent
//! method; the trait impls at the end bind `CellVirtual`, `InputCellVirtual` and
//! `ToggleCell` to them.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::cell::{Cell, CellVirtual};
use super::colors::{self, ColorUIResource};
use super::input_cell::{InputCell, InputCellVirtual};
use super::radio_button::RadioButton;
use super::toggle_cell::ToggleCell;
use super::tooltip_formatter;
use crate::imod::etomo::jdk::{ActionListener, ButtonGroup, ChangeListener, JComponent};
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};
use crate::imod::etomo::util::utilities;

/// Java `RadioButtonCell`.
pub struct RadioButtonCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// Java `radioButton`.
    radio_button: Rc<RadioButton>,
    /// Java `unformattedLabel` (initialized to "" by its field initializer, which
    /// runs before the constructor body).
    unformatted_label: RefCell<Option<String>>,
}

impl Deref for RadioButtonCell {
    type Target = InputCell;

    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl RadioButtonCell {
    /// Java private `RadioButtonCell(ButtonGroup, String)`.
    fn new(button_group: Option<&Rc<ButtonGroup>>, reference: Option<&str>) -> Rc<RadioButtonCell> {
        // super(): InputCell()
        let radio_button = RadioButton::new_button_group(button_group);
        let instance = Rc::new(RadioButtonCell {
            base: InputCell::new_void(),
            radio_button,
            unformatted_label: RefCell::new(Some(String::new())),
        });
        let weak: Weak<RadioButtonCell> = Rc::downgrade(&instance);
        instance.base.set_this(weak);
        // Swing layout: radioButton.setBorderPainted(true);
        // radioButton.setBorder(BorderFactory.createEtchedBorder()).
        instance.set_background_void();
        instance.set_foreground();
        instance.set_font();
        if reference.is_some() {
            instance.set_name_string(reference);
        }
        instance
    }

    /// Java static `getInstance(ButtonGroup)`.
    pub fn get_instance(button_group: Option<&Rc<ButtonGroup>>) -> Rc<RadioButtonCell> {
        RadioButtonCell::new(button_group, None)
    }

    /// Java static `getNamedInstance(ButtonGroup, String)`.
    pub fn get_named_instance_button_group_string(
        button_group: Option<&Rc<ButtonGroup>>,
        header_label: Option<&str>,
    ) -> Rc<RadioButtonCell> {
        RadioButtonCell::new(button_group, header_label)
    }

    /// Java static `getNamedInstance(ButtonGroup, String, String, String)`.
    pub fn get_named_instance_button_group_string_string_string(
        button_group: Option<&Rc<ButtonGroup>>,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) -> Rc<RadioButtonCell> {
        RadioButtonCell::new(
            button_group,
            utilities::concatenate(reference1, reference2, reference3, Some(" ")).as_deref(),
        )
    }

    /// Java `setName(String, String, String)` (implements `InputCell.setName`).
    pub fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        self.set_name_string(
            utilities::concatenate(reference1, reference2, reference3, Some(" ")).as_deref(),
        );
    }

    /// Java private `setName(String)`.
    fn set_name_string(&self, reference: Option<&str>) {
        self.radio_button.set_name(reference);
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.radio_button.get_name()
    }

    /// Java private `setForeground()`.
    fn set_foreground(&self) {
        self.radio_button.set_foreground(Some(colors::CELL_FOREGROUND));
        self.set_html_label(colors::CELL_FOREGROUND);
    }

    /// Java private `setHtmlLabel(ColorUIResource)`.
    fn set_html_label(&self, color: ColorUIResource) {
        let text = format!(
            "<html><P style=\"font-weight:normal; color:rgb({},{},{})\">{}</style>",
            color.0,
            color.1,
            color.2,
            self.unformatted_label.borrow().as_deref().unwrap_or("null")
        );
        self.radio_button.set_text(Some(&text));
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.radio_button.set_selected_boolean(selected);
    }

    /// Java `getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: radioButton.getWidth().  Sizes are not modelled by the jdk
        // stand-in.
        0
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> Option<String> {
        self.unformatted_label.borrow().clone()
    }

    /// Java `getHeight()`.
    pub fn get_height(&self) -> i32 {
        // Swing geometry: radioButton.getHeight() + the border's bottom inset - 1.
        // Sizes are not modelled by the jdk stand-in.
        0
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&self, label: Option<&str>) {
        *self.unformatted_label.borrow_mut() = label.map(str::to_owned);
        self.set_foreground();
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.radio_button
            .set_tool_tip_text_string(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.radio_button.add_action_listener(action_listener);
    }

    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&self, listener: ChangeListener) {
        self.radio_button.add_change_listener(listener);
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.radio_button.get_component()
    }

    /// Java `getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::RADIO_BUTTON
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        self.radio_button.set_locked(locked);
        self.set_background_void();
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.radio_button.is_locked()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.radio_button.set_enabled(enabled); // handles enabled vs editable
        self.set_background_void();
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.radio_button.is_enabled()
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.radio_button.set_editable(editable);
        self.set_background_void();
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.radio_button.is_editable()
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        self.unformatted_label.borrow().clone()
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }
}

impl CellVirtual for RadioButtonCell {
    fn cell(&self) -> &Cell {
        &self.base
    }

    fn set_enabled(&self, enable: bool) {
        RadioButtonCell::set_enabled(self, enable);
    }

    fn msg_label_changed(&self) {
        self.base.msg_label_changed();
    }

    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }
}

impl InputCellVirtual for RadioButtonCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }

    fn get_component(&self) -> Rc<JComponent> {
        RadioButtonCell::get_component(self)
    }

    fn get_field_type(&self) -> &'static UITestFieldType {
        RadioButtonCell::get_field_type(self)
    }

    fn get_width(&self) -> i32 {
        RadioButtonCell::get_width(self)
    }

    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        RadioButtonCell::set_tool_tip_text(self, tool_tip_text);
    }

    fn get_text(&self) -> Option<String> {
        RadioButtonCell::get_text(self)
    }

    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        RadioButtonCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }

    fn get_name(&self) -> Option<String> {
        RadioButtonCell::get_name(self)
    }

    fn set_locked(&self, locked: bool) {
        RadioButtonCell::set_locked(self, locked);
    }

    fn set_editable(&self, editable: bool) {
        RadioButtonCell::set_editable(self, editable);
    }

    fn is_locked(&self) -> bool {
        RadioButtonCell::is_locked(self)
    }

    fn is_editable(&self) -> bool {
        RadioButtonCell::is_editable(self)
    }

    fn is_enabled(&self) -> bool {
        RadioButtonCell::is_enabled(self)
    }
}

impl ToggleCell for RadioButtonCell {
    fn get_label(&self) -> Option<String> {
        RadioButtonCell::get_label(self)
    }

    fn set_label(&self, label: Option<&str>) {
        RadioButtonCell::set_label(self, label);
    }

    fn set_selected(&self, selected: bool) {
        RadioButtonCell::set_selected(self, selected);
    }

    fn add_action_listener(&self, action_listener: ActionListener) {
        RadioButtonCell::add_action_listener(self, action_listener);
    }

    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }

    fn is_selected(&self) -> bool {
        RadioButtonCell::is_selected(self)
    }

    fn get_height(&self) -> i32 {
        RadioButtonCell::get_height(self)
    }

    fn get_width(&self) -> i32 {
        RadioButtonCell::get_width(self)
    }

    fn set_warning(&self, warning: bool) {
        self.base.set_warning_boolean(warning);
    }

    fn add_change_listener(&self, listener: ChangeListener) {
        RadioButtonCell::add_change_listener(self, listener);
    }

    fn set_enabled(&self, enabled: bool) {
        RadioButtonCell::set_enabled(self, enabled);
    }

    fn is_enabled(&self) -> bool {
        RadioButtonCell::is_enabled(self)
    }
}
