//! `IMOD/Etomo/src/etomo/ui/swing/ButtonCell.java`.
//!
//! A table cell holding a raised `JButton` (with an icon) or `JToggleButton` (with a
//! title).
//!
//! Java `final class ButtonCell extends InputCell implements TableComponent`: every
//! Java method body is an inherent method; the trait impls at the end bind
//! `CellVirtual` and `InputCellVirtual` to them.  Icons are represented by their
//! image names (see `complete_icon.rs`); borders and sizes are layout.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::cell::{Cell as TableCell, CellVirtual};
use super::field_lock_controller::FieldLockController;
use super::input_cell::{InputCell, InputCellVirtual};
use super::tooltip_formatter;
use super::ui_utilities;
use crate::imod::etomo::jdk::{ActionListener, JComponent};
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};

/// Java package-private `final class ButtonCell extends InputCell`.
pub struct ButtonCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// Java final `button` (a `JButton` or a `JToggleButton`).
    button: Rc<JComponent>,
    /// Java final `fieldLockController`.
    field_lock_controller: Rc<FieldLockController>,
}

impl Deref for ButtonCell {
    type Target = InputCell;
    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl ButtonCell {
    /// Java private `ButtonCell(Icon, String, boolean)`.
    fn new(icon: Option<&str>, title: Option<&str>, toggle: bool) -> Rc<ButtonCell> {
        // super(): InputCell()
        let (button, field_lock_controller) = if !toggle {
            let button = JComponent::new_button("");
            let field_lock_controller = FieldLockController::get_button_instance(&button);
            (button, field_lock_controller)
        } else {
            let button = JComponent::new_toggle_button("");
            let field_lock_controller =
                FieldLockController::get_toggle_button_instance_j_toggle_button(&button);
            (button, field_lock_controller)
        };
        if icon.is_some() {
            // Swing painting: button.setIcon(icon).
        } else if let Some(title) = title {
            button.set_text(title);
        }
        // Swing layout: button.setBorder(BorderFactory.createBevelBorder(
        // BevelBorder.RAISED)); button.setPreferredSize(UIUtilities.getPreferredSize(
        // button, title)).
        let instance = Rc::new(ButtonCell {
            base: InputCell::new_void(),
            button,
            field_lock_controller,
        });
        instance
            .base
            .set_this(Rc::downgrade(&instance) as Weak<dyn InputCellVirtual>);
        instance
    }

    /// Java static `getInstance(Icon)`.
    pub fn get_instance(icon: Option<&str>) -> Rc<ButtonCell> {
        ButtonCell::new(icon, None, false)
    }

    /// Java static `getToggleInstance(String)`.
    pub fn get_toggle_instance(title: Option<&str>) -> Rc<ButtonCell> {
        ButtonCell::new(None, title, true)
    }

    /// Java `@Override setName(String, String, String)`.  Not implemented at this
    /// time. See HeaderCell.setName to implement.
    pub fn set_name_string_string_string(
        &self,
        _reference1: Option<&str>,
        _reference2: Option<&str>,
        _reference3: Option<&str>,
    ) {
    }

    /// Java `@Override getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.button.get_name()
    }

    /// Java `getUniqueActionCommand()`: the class name and identity hash (Java
    /// `Object.toString`).
    pub fn get_unique_action_command(&self) -> String {
        format!(
            "etomo.ui.swing.ButtonCell@{:x}",
            self as *const ButtonCell as usize
        )
    }

    /// Java `@Override getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }

    /// Java `@Override getText()`.
    pub fn get_text(&self) -> Option<String> {
        Some(self.button.get_text())
    }

    /// Java `@Override getPreferredWidth()` (`TableComponent`).
    pub fn get_preferred_width(&self) -> i32 {
        ui_utilities::get_preferred_width_abstract_button_string(
            &self.button,
            Some(&self.button.get_text()),
        )
    }

    /// Java `@Override getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::BUTTON
    }

    /// Java `@Override getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: button.getWidth().  Sizes are not modelled by the jdk
        // stand-in.
        0
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.button.set_selected(selected);
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.button.is_selected()
    }

    /// Java `setActionCommand(String)`.
    pub fn set_action_command(&self, input: Option<&str>) {
        self.button.set_action_command(input);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.button.get_action_command()
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.button.add_action_listener(listener);
    }

    /// Java `@Override setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.base.set_background_void();
        }
    }

    /// Java `@Override setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.base.set_background_void();
        }
    }

    /// Java `@Override setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
    }

    /// Java `@Override isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `@Override isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `@Override isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `setDisabledIcon(Icon)`.
    pub fn set_disabled_icon(&self, _icon: Option<&str>) {
        // Swing painting: button.setDisabledIcon(icon).
    }

    /// Java `@Override setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.button
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }
}

impl CellVirtual for ButtonCell {
    fn cell(&self) -> &TableCell {
        &self.base
    }
    fn set_enabled(&self, enable: bool) {
        ButtonCell::set_enabled(self, enable);
    }
    fn msg_label_changed(&self) {
        self.base.msg_label_changed();
    }
    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }
}

impl InputCellVirtual for ButtonCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }
    fn get_component(&self) -> Rc<JComponent> {
        ButtonCell::get_component(self)
    }
    fn get_field_type(&self) -> &'static UITestFieldType {
        ButtonCell::get_field_type(self)
    }
    fn get_width(&self) -> i32 {
        ButtonCell::get_width(self)
    }
    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        ButtonCell::set_tool_tip_text(self, tool_tip_text);
    }
    fn get_text(&self) -> Option<String> {
        ButtonCell::get_text(self)
    }
    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        ButtonCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }
    fn get_name(&self) -> Option<String> {
        ButtonCell::get_name(self)
    }
    fn set_locked(&self, locked: bool) {
        ButtonCell::set_locked(self, locked);
    }
    fn set_editable(&self, editable: bool) {
        ButtonCell::set_editable(self, editable);
    }
    fn is_locked(&self) -> bool {
        ButtonCell::is_locked(self)
    }
    fn is_editable(&self) -> bool {
        ButtonCell::is_editable(self)
    }
    fn is_enabled(&self) -> bool {
        ButtonCell::is_enabled(self)
    }
}

/// The `etomo.ui.TableComponent` interface the Java class implements.
impl crate::imod::etomo::ui::table_component::TableComponent for ButtonCell {
    fn get_preferred_width(&self) -> i32 {
        ButtonCell::get_preferred_width(self)
    }
}
