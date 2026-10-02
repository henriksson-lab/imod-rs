//! `IMOD/Etomo/src/etomo/ui/swing/SpinnerCell.java`.
//!
//! Java `class SpinnerCell extends InputCell`: a table cell with an integer
//! `JSpinner`.
//!
//! The spinner's editor text field (`JSpinner.DefaultEditor.getTextField()`) is not a
//! separate component in the jdk stand-in: its text is the spinner's value, and its
//! tooltip, alignment and colours are painting.  Every Java method body is an
//! inherent method; the trait impls at the end bind `CellVirtual` and
//! `InputCellVirtual` to them.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::cell::{Cell, CellVirtual};
use super::colors::{self, ColorUIResource};
use super::field_lock_controller::FieldLockController;
use super::input_cell::{InputCell, InputCellVirtual};
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ChangeListener, FocusListener, JComponent, SpinnerNumberModel};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::const_etomo_number::{Number, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};
use crate::imod::etomo::util::utilities;

/// Java `SpinnerCell`.
pub struct SpinnerCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// Java `disabledValue`.
    disabled_value: RefCell<EtomoNumber>,
    /// Java `savedValue`.
    saved_value: RefCell<EtomoNumber>,
    /// Java `spinner`.
    spinner: Rc<JComponent>,
    /// Java `type`.
    r#type: Type,
    /// Java `minimumValue`.
    minimum_value: i32,
    /// Java `fieldLockController`.
    field_lock_controller: Rc<FieldLockController>,
}

impl Deref for SpinnerCell {
    type Target = InputCell;

    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl SpinnerCell {
    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        // getTextField().getText(): the editor's text is the integer value.
        (self.spinner.get_spinner_value() as i32).to_string()
    }

    /// Java static `getIntInstance(int, int)`.
    pub fn get_int_instance(min: i32, max: i32) -> Rc<SpinnerCell> {
        SpinnerCell::new(min, max, None)
    }

    /// Java static `getNamedIntInstance(int, int, String, String, String)`.
    pub fn get_named_int_instance(
        min: i32,
        max: i32,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) -> Rc<SpinnerCell> {
        SpinnerCell::new(
            min,
            max,
            utilities::concatenate(reference1, reference2, reference3, Some(" ")).as_deref(),
        )
    }

    /// Java `setEnabled(boolean)`.  disable - the buttons shouldn't work
    pub fn set_enabled(&self, enabled: bool) {
        let was_enabled = self.is_enabled();
        self.field_lock_controller.set_enabled(enabled);
        // If disabledValue is set, then enabling/disabling changes the value back and
        // forth from a disabled value and the value it was most recently set to when it
        // was enabled.
        if !self.disabled_value.borrow().is_null() {
            if enabled {
                let current = Number::Integer(self.spinner.get_spinner_value() as i32);
                if self.disabled_value.borrow().equals_number(Some(current))
                    && !self.saved_value.borrow().is_null()
                {
                    let saved = self.saved_value.borrow().get_number();
                    self.spinner.set_spinner_value(saved.double_value());
                }
            } else {
                if was_enabled {
                    // Hang onto the enabled value.
                    let current = Number::Integer(self.spinner.get_spinner_value() as i32);
                    self.saved_value.borrow_mut().set_number(Some(current));
                    // Out of range values prevent the spinner from working, so make sure
                    // they are not used when the spinner is enabled.
                    if self.saved_value.borrow().lt_int(self.minimum_value) {
                        self.saved_value.borrow_mut().set_int(self.minimum_value);
                    }
                }
                let disabled = self.disabled_value.borrow().get_number();
                self.spinner.set_spinner_value(disabled.double_value());
            }
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        self.field_lock_controller.set_locked(locked);
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `setEditable(boolean)`.  The buttons should still work
    pub fn set_editable(&self, editable: bool) {
        self.field_lock_controller.set_editable(editable);
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `setDisabledValue(int)`.
    pub fn set_disabled_value(&self, disabled_value: i32) {
        self.disabled_value.borrow_mut().set_int(disabled_value);
    }

    /// Java final `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.spinner.clone()
    }

    /// Java final `setValue(int)`.
    pub fn set_value_int(&self, value: i32) {
        let number = {
            let mut etomo_number = EtomoNumber::new_with_type(Some(self.r#type));
            etomo_number.set_int(value);
            etomo_number.get_number()
        };
        self.set_value_number(number);
    }

    /// Java final `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        let number = {
            let mut etomo_number = EtomoNumber::new_with_type(Some(self.r#type));
            etomo_number.set_string(value);
            etomo_number.get_number()
        };
        self.set_value_number(number);
    }

    /// Java `getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::SPINNER
    }

    /// Java final `getIntValue()`.
    pub fn get_int_value(&self) -> i32 {
        // ((Integer) spinner.getValue()).intValue(): the model is built from ints.
        self.spinner.get_spinner_value() as i32
    }

    /// Java final `getStringValue()`.
    pub fn get_string_value(&self) -> Option<String> {
        if self.r#type == Type::Integer {
            return Some(self.get_int_value().to_string());
        }
        None
    }

    /// Java final `getText()`.
    pub fn get_text(&self) -> Option<String> {
        self.get_string_value()
    }

    /// Java final `getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: spinner.getWidth().  Sizes are not modelled by the jdk
        // stand-in.
        0
    }

    /// Java final `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&self, change_listener: ChangeListener) {
        self.spinner.add_change_listener(change_listener);
    }

    /// Java final `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, focus_listener: FocusListener) {
        self.spinner.add_focus_listener(focus_listener);
    }

    /// Java final `removeChangeListener(ChangeListener)`.
    pub fn remove_change_listener(&self, change_listener: &ChangeListener) {
        self.spinner.remove_change_listener(change_listener);
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        let tooltip = tooltip_formatter::INSTANCE.format(text);
        self.spinner.set_tool_tip_text(tooltip.as_deref());
        // getTextField().setToolTipText(tooltip): the editor is not a separate
        // component in the stand-in.
    }

    /// Java final `setBackground(ColorUIResource)` (overrides `InputCell`).  This
    /// probably doesn't work.  Should use something like
    /// UIUtilities.highlightJTextComponents.
    pub fn set_background_color_ui_resource(&self, _color: ColorUIResource) {
        // Swing painting: getTextField().setBackground(color).
    }

    /// Java final `setForeground()`.
    pub fn set_foreground(&self) {
        // JFormattedTextField textField = getTextField();
        // textField.setForeground(Colors.CELL_FOREGROUND): the editor's colour is the
        // spinner's in the stand-in.
        self.spinner.set_foreground(Some(colors::CELL_FOREGROUND));
        // Swing painting: textField.setDisabledTextColor(Colors.CELL_FOREGROUND).
    }

    /// Java private `SpinnerCell(int, int, String)`.
    fn new(min: i32, max: i32, reference: Option<&str>) -> Rc<SpinnerCell> {
        // super(): InputCell()
        let spinner = JComponent::new_spinner(SpinnerNumberModel::new_int(min, min, max, 1));
        let field_lock_controller = FieldLockController::get_spinner_instance(&spinner);
        let instance = Rc::new(SpinnerCell {
            base: InputCell::new_void(),
            r#type: Type::Integer,
            disabled_value: RefCell::new(EtomoNumber::new()),
            saved_value: RefCell::new(EtomoNumber::new()),
            minimum_value: min,
            spinner,
            field_lock_controller,
        });
        let weak: Weak<SpinnerCell> = Rc::downgrade(&instance);
        instance.base.set_this(weak);
        // Swing layout: spinner.setBorder(BorderFactory.createEtchedBorder());
        // getTextField().setHorizontalAlignment(JTextField.LEFT).
        instance.set_background_void();
        instance.set_foreground();
        instance.set_font();
        if reference.is_some() {
            instance.set_name_string(reference);
        }
        instance
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

    /// Java `setName(String)`.
    pub fn set_name_string(&self, reference: Option<&str>) {
        let field_type = &ui_test_field_type::SPINNER;
        // Java string concatenation of a null name gives "null".
        let name = format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            utilities::convert_label_to_name(reference, field_type.is_unlimited_segments())
                .as_deref()
                .unwrap_or("null")
        );
        self.spinner.set_name(Some(&name));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.spinner.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.spinner.get_name()
    }

    // Java private final `getTextField()`: ((JSpinner.DefaultEditor)
    // spinner.getEditor()).getTextField().  The editor is not a separate component in
    // the stand-in; its callers above act on the spinner or say why they do not.

    /// Java private final `setValue(Number)`.
    fn set_value_number(&self, value: Number) {
        self.saved_value.borrow_mut().set_number(Some(value));
        if self.is_enabled() || self.disabled_value.borrow().is_null() {
            self.spinner.set_spinner_value(value.double_value());
        }
    }
}

impl CellVirtual for SpinnerCell {
    fn cell(&self) -> &Cell {
        &self.base
    }

    fn set_enabled(&self, enable: bool) {
        SpinnerCell::set_enabled(self, enable);
    }

    fn msg_label_changed(&self) {
        self.base.msg_label_changed();
    }

    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }
}

impl InputCellVirtual for SpinnerCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }

    fn get_component(&self) -> Rc<JComponent> {
        SpinnerCell::get_component(self)
    }

    fn get_field_type(&self) -> &'static UITestFieldType {
        SpinnerCell::get_field_type(self)
    }

    fn get_width(&self) -> i32 {
        SpinnerCell::get_width(self)
    }

    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        SpinnerCell::set_tool_tip_text(self, tool_tip_text);
    }

    fn get_text(&self) -> Option<String> {
        SpinnerCell::get_text(self)
    }

    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        SpinnerCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }

    fn get_name(&self) -> Option<String> {
        SpinnerCell::get_name(self)
    }

    fn set_locked(&self, locked: bool) {
        SpinnerCell::set_locked(self, locked);
    }

    fn set_editable(&self, editable: bool) {
        SpinnerCell::set_editable(self, editable);
    }

    fn is_locked(&self) -> bool {
        SpinnerCell::is_locked(self)
    }

    fn is_editable(&self) -> bool {
        SpinnerCell::is_editable(self)
    }

    fn is_enabled(&self) -> bool {
        SpinnerCell::is_enabled(self)
    }

    fn set_background_color_ui_resource(&self, color: ColorUIResource) {
        SpinnerCell::set_background_color_ui_resource(self, color);
    }
}
