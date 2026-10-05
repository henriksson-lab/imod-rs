//! `IMOD/Etomo/src/etomo/ui/swing/FieldCell.java`.
//!
//! Java `final class FieldCell extends InputCell implements ActionTarget,
//! TableComponent, TextFieldInterface, UIComponent, SwingComponent`: a table cell
//! holding a text field.
//!
//! Every Java method body is an inherent method here (overloads carry the
//! parameter-type suffix of `ui.md`); the trait impls at the end bind
//! `CellVirtual`, `InputCellVirtual`, `UIComponent` and `SwingComponent` to those
//! bodies.  `ActionTarget`, `TableComponent` and `TextFieldInterface` are
//! implemented by the inherent methods of the same names (see the report's NEEDS).

use std::cell::{Cell as StdCell, RefCell};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::cell::{Cell, CellVirtual};
use super::colors::{self, ColorUIResource};
use super::field_lock_controller::FieldLockController;
use super::input_cell::{InputCell, InputCellVirtual};
use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::logic::text_field_state::TextFieldState;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    self, ConstEtomoNumber, Type, java_lang_double_to_string, java_lang_double_value_of,
    java_lang_integer_parse_int, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::parsed_element_type::{self, ParsedElementType};
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `FieldCell`.
pub struct FieldCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// Java `textField`.
    text_field: Rc<JComponent>,
    /// Java `parsedElementType`.
    parsed_element_type: Option<&'static ParsedElementType>,
    /// Java `fieldLockController`.
    field_lock_controller: Rc<FieldLockController>,
    /// Java `inUse`.
    in_use: StdCell<bool>,
    // Java `fontMetrics`: font metrics are painting and are not modelled (see
    // `get_preferred_width`).
    /// Java `directiveDef`.
    directive_def: RefCell<Option<DirectiveDef>>,
    /// Java `unformattedTooltip`.
    unformatted_tooltip: RefCell<Option<String>>,
    /// Java `state`.
    state: RefCell<Option<TextFieldState>>,
}

impl Deref for FieldCell {
    type Target = InputCell;

    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl FieldCell {
    /// Allocates the cell and connects `InputCell`'s `this` (the part of Java object
    /// construction that happens before any constructor body runs).
    fn allocate(
        text_field: Rc<JComponent>,
        parsed_element_type: Option<&'static ParsedElementType>,
        field_lock_controller: Rc<FieldLockController>,
        state: TextFieldState,
    ) -> Rc<FieldCell> {
        let instance = Rc::new(FieldCell {
            base: InputCell::new_void(),
            text_field,
            parsed_element_type,
            field_lock_controller,
            in_use: StdCell::new(true),
            directive_def: RefCell::new(None),
            unformatted_tooltip: RefCell::new(None),
            state: RefCell::new(Some(state)),
        });
        let weak: Weak<FieldCell> = Rc::downgrade(&instance);
        instance.base.set_this(weak);
        instance
    }

    /// Java private `FieldCell(boolean, ParsedElementType, String, String, String)`.
    fn new_boolean_parsed_element_type_string_string_string(
        editable: bool,
        parsed_element_type: Option<&'static ParsedElementType>,
        root_dir: Option<&str>,
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<FieldCell> {
        // super(): InputCell() - see `allocate`.
        let state = TextFieldState::new_boolean_parsed_element_type_string(
            editable,
            parsed_element_type,
            root_dir,
        );
        // construction
        let text_field = JComponent::new_text_field();
        let field_lock_controller =
            FieldLockController::get_text_component_instance_j_text_component_boolean(
                &text_field,
                !editable,
            );
        let instance = FieldCell::allocate(
            text_field,
            parsed_element_type,
            field_lock_controller,
            state,
        );
        let header_label = utilities::concatenate(header_label1, header_label2, None, Some(" "));
        if header_label.is_some() {
            instance.set_name_string(header_label.as_deref());
        }
        // field
        // Swing layout: textField.setBorder(BorderFactory.createEtchedBorder()).
        // color
        instance.set_background_void();
        instance.set_foreground();
        instance.set_font();
        instance.set_expanded();
        instance
    }

    /// Java private `FieldCell(TextFieldState)`.
    fn new_text_field_state(state: &TextFieldState) -> Rc<FieldCell> {
        let state = TextFieldState::new_text_field_state(state);
        // construction
        let text_field = JComponent::new_text_field();
        let field_lock_controller =
            FieldLockController::get_text_component_instance_j_text_component(&text_field);
        let instance = FieldCell::allocate(text_field, None, field_lock_controller, state);
        // field
        // Swing layout: textField.setBorder(BorderFactory.createEtchedBorder()).
        // color
        instance.set_background_void();
        instance.set_foreground();
        instance.set_font();
        instance.set_expanded();
        instance
    }

    /// Java static `getInstance(FieldCell)`.
    pub fn get_instance(field_cell: &FieldCell) -> Rc<FieldCell> {
        let instance = {
            let state = field_cell.state.borrow();
            FieldCell::new_text_field_state(
                state
                    .as_ref()
                    .expect("FieldCell: state is set by every constructor"),
            )
        };
        instance.in_use.set(field_cell.in_use.get());
        instance.set_value_string(field_cell.get_expanded_value().as_deref());
        instance.add_listeners();
        instance.set_tool_tip_text(field_cell.text_field.get_tool_tip_text().as_deref());
        instance
    }

    /// Java static `getEditableInstance()`.
    pub fn get_editable_instance() -> Rc<FieldCell> {
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            true,
            Some(&parsed_element_type::NON_MATLAB_NUMBER),
            None,
            None,
            None,
        );
        instance.add_listeners();
        instance
    }

    /// Java static `getEditableMatlabInstance()`.
    pub fn get_editable_matlab_instance() -> Rc<FieldCell> {
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            true,
            Some(&parsed_element_type::MATLAB_NUMBER),
            None,
            None,
            None,
        );
        instance.add_listeners();
        instance
    }

    /// Java static `getIneditableInstance()`.
    pub fn get_ineditable_instance() -> Rc<FieldCell> {
        let editable = false;
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            editable,
            Some(&parsed_element_type::NON_MATLAB_NUMBER),
            None,
            None,
            None,
        );
        instance.set_editable(editable);
        instance.add_listeners();
        instance
    }

    /// Java static `getNamedIneditableInstance(String)`.
    pub fn get_named_ineditable_instance_string(header_label: Option<&str>) -> Rc<FieldCell> {
        let editable = false;
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            editable,
            Some(&parsed_element_type::NON_MATLAB_NUMBER),
            None,
            header_label,
            None,
        );
        instance.set_editable(editable);
        instance.add_listeners();
        instance
    }

    /// Java static `getNamedIneditableInstance(String, String)`.
    pub fn get_named_ineditable_instance_string_string(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<FieldCell> {
        let editable = false;
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            editable,
            Some(&parsed_element_type::NON_MATLAB_NUMBER),
            None,
            header_label1,
            header_label2,
        );
        instance.set_editable(editable);
        instance.add_listeners();
        instance
    }

    /// Java static `getExpandableInstance(String)`.
    pub fn get_expandable_instance(root_dir: Option<&str>) -> Rc<FieldCell> {
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            true,
            Some(&parsed_element_type::NON_MATLAB_NUMBER),
            root_dir,
            None,
            None,
        );
        instance.add_listeners();
        instance
    }

    /// Java static `getExpandableIneditableInstance(String)`.
    pub fn get_expandable_ineditable_instance(root_dir: Option<&str>) -> Rc<FieldCell> {
        let editable = false;
        let instance = FieldCell::new_boolean_parsed_element_type_string_string_string(
            editable,
            Some(&parsed_element_type::NON_MATLAB_NUMBER),
            root_dir,
            None,
            None,
        );
        instance.set_editable(editable);
        instance.add_listeners();
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

    /// Java private `setName(String)`.  Create a much simpler name and rely on the
    /// index to handle the multiple rows.
    fn set_name_string(&self, reference: Option<&str>) {
        let field_type = &ui_test_field_type::TEXT_FIELD;
        let name = utilities::convert_label_to_name(reference, field_type.is_unlimited_segments());
        if let Some(name) = name {
            self.text_field
                .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {} ",
                    self.text_field.get_name().as_deref().unwrap_or("null"),
                    DEFAULT_DELIMITER
                );
            }
        }
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.text_field.get_name()
    }

    /// Java `isText()`.
    pub fn is_text(&self) -> bool {
        true
    }

    /// Java `isBoolean()`.
    pub fn is_boolean(&self) -> bool {
        false
    }

    /// Java `setDebug(boolean)` (overrides `InputCell.setDebug`).
    pub fn set_debug(&self, input: bool) {
        if let Some(state) = self.state.borrow_mut().as_mut() {
            state.set_debug(input);
        }
        self.base.set_debug_super(input);
    }

    /// Java `setRootDir(String)`.
    pub fn set_root_dir(&self, input: Option<&str>) {
        let state = TextFieldState::new_boolean_parsed_element_type_string(
            self.is_editable(),
            self.parsed_element_type,
            input,
        );
        *self.state.borrow_mut() = Some(state);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // TextFieldFocusListener: textField.addFocusListener(new
        // TextFieldFocusListener(textField)).  Focus events are not modelled by the
        // jdk stand-in; the listener only selects all of the text on focus gained and
        // clears the selection on focus lost (text selection, not state).
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        self.text_field.get_text()
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        *self.directive_def.borrow_mut() = directive_def;
    }

    /// Java `getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.borrow().clone()
    }

    // Java `getPreferredSize()` returns `textField.getPreferredSize()` and
    // `getRightBorder()` returns the text field border's right inset: Swing geometry,
    // not modelled by the jdk stand-in.

    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        // Swing painting: `UIUtilities.getPreferredWidth(textField.getText(),
        // fontMetrics)` measures the text in the field's font.  Fonts are not modelled
        // by the jdk stand-in, so there is no width to report.
        0
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.text_field.set_visible(visible);
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        self.field_lock_controller.set_locked(locked);
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.field_lock_controller.set_editable(editable);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        if self.field_lock_controller.set_enabled(enabled) {
            self.set_background_void();
            if self.is_enabled() && self.is_editable() {
                self.set_foreground();
            } else {
                self.text_field
                    .set_foreground(Some(colors::CELL_DISABLED_FOREGROUND));
                // Swing painting: textField.setDisabledTextColor(CELL_DISABLED_FOREGROUND).
            }
        }
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `setInUse(boolean)`.
    pub fn set_in_use(&self, in_use: bool) {
        self.in_use.set(in_use);
        self.set_foreground();
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        let value = self.text_field.get_text();
        java_lang_string_matches_whitespace(&value)
    }

    /// Java `backup()`.
    pub fn backup(&self) {
        eprintln!("Warning: Backup has not been implemented in FieldCell.");
    }

    /// Java `restoreFromBackup()`.
    pub fn restore_from_backup(&self) {}

    /// Java `checkpoint()`.
    pub fn checkpoint(&self) {
        eprintln!("Warning: Checkpoint has not been implemented in FieldCell.");
    }

    /// Java `getCheckpoint()`.
    pub fn get_checkpoint(&self) -> Option<Box<dyn FieldSettingInterface>> {
        None
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, _always_check: bool) -> bool {
        false
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    pub fn set_checkpoint(&self, _input: Option<&dyn FieldSettingInterface>) {
        eprintln!("Warning: Checkpoint has not been implemented in FieldCell.");
    }

    /// Java `equalsDefaultValue()`.
    pub fn equals_default_value_void(&self) -> bool {
        false
    }

    /// Java `equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, _value: Option<&str>) -> bool {
        false
    }

    /// Java `useDefaultValue()`.
    pub fn use_default_value(&self) {
        eprintln!("Warning: Default value has not been implemented in FieldCell.");
    }

    /// Java `isFieldHighlightSet()`.
    pub fn is_field_highlight_set(&self) -> bool {
        false
    }

    /// Java `clearFieldHighlight()`.
    pub fn clear_field_highlight(&self) {
        eprintln!("Warning: Field highlight has not been implemented in FieldCell.");
    }

    /// Java `equalsFieldHighlight()`.
    pub fn equals_field_highlight_void(&self) -> bool {
        false
    }

    /// Java `equalsFieldHighlight(String)`.
    pub fn equals_field_highlight_string(&self, _value: Option<&str>) -> bool {
        false
    }

    /// Java `getFieldHighlight()`.
    pub fn get_field_highlight(&self) -> Option<Box<dyn FieldSettingInterface>> {
        None
    }

    /// Java `setFieldHighlight(boolean)`.
    pub fn set_field_highlight_boolean(&self, _value: bool) {}

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    pub fn set_field_highlight_field_setting_interface(
        &self,
        _setting_interface: Option<&dyn FieldSettingInterface>,
    ) {
        eprintln!("Warning: Field highlight has not been implemented in FieldCell.");
    }

    /// Java `setFieldHighlight(String)`.
    pub fn set_field_highlight_string(&self, _value: Option<&str>) {
        eprintln!("Warning: Field highlight has not been implemented in FieldCell.");
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }

    /// Java `setTargetFile(File)` (`ActionTarget`).
    pub fn set_target_file(&self, file: Option<&Path>) {
        self.set_file(file);
    }

    /// Java `setFile(File)`.
    pub fn set_file(&self, file: Option<&Path>) {
        self.set_value_file(file);
    }

    /// Java `setValue(File)`.
    pub fn set_value_file(&self, file: Option<&Path>) {
        let text = self
            .state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .convert_to_field_text_file(file);
        // Swing setText(null) shows an empty field.
        self.text_field.set_text(text.as_deref().unwrap_or(""));
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        let text = self
            .state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .convert_to_field_text_string(value);
        self.text_field.set_text(&text);
    }

    /// Java `setValue(String, boolean)`.
    pub fn set_value_string_boolean(&self, value: Option<&str>, allow_empty: bool) {
        if allow_empty || value.is_some_and(|value| !value.is_empty()) {
            self.set_value_string(value);
        }
    }

    /// Java `setValue(Field)`.
    pub fn set_value_field(&self, input: Option<&dyn Field>) {
        match input {
            None => self.clear(),
            Some(input) => self.set_value_string(input.get_text_void().as_deref()),
        }
    }

    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&self, _value: bool) {}

    /// Java `getContractedValue()`.
    pub fn get_contracted_value(&self) -> Option<String> {
        let text = self.text_field.get_text();
        self.state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .convert_to_contracted_string(Some(&text))
    }

    /// Java `getExpandedValue()`.
    pub fn get_expanded_value(&self) -> Option<String> {
        let text = self.text_field.get_text();
        Some(
            self.state
                .borrow_mut()
                .as_mut()
                .expect("FieldCell: state is set by every constructor")
                .convert_to_expanded_string(Some(&text)),
        )
    }

    /// Java `getFile()`.
    pub fn get_file(&self) -> Option<PathBuf> {
        let expanded_value = self.get_expanded_value();
        if let Some(expanded_value) = expanded_value
            && !expanded_value.is_empty()
        {
            return Some(PathBuf::from(expanded_value));
        }
        None
    }

    /// Java `expand(boolean)`.
    pub fn expand(&self, expand: bool) {
        let text = self.text_field.get_text();
        let text = self
            .state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .expand_field_text(expand, Some(&text));
        // Swing setText(null) shows an empty field.
        self.text_field.set_text(text.as_deref().unwrap_or(""));
    }

    /// Java `setHorizontalAlignment(int)`.
    pub fn set_horizontal_alignment(&self, _alignment: i32) {
        // Swing layout: textField.setHorizontalAlignment(JTextField.CENTER) (the
        // parameter is ignored by the source).
    }

    /// Java `setExpanded()`.
    pub fn set_expanded(&self) {
        let text = self.text_field.get_text();
        let text = self
            .state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .apply_expanded_to_field_text(Some(&text));
        // Swing setText(null) shows an empty field.
        self.text_field.set_text(text.as_deref().unwrap_or(""));
    }

    /// Java `setValue(ConstEtomoNumber)`.
    pub fn set_value_const_etomo_number(&self, value: &ConstEtomoNumber) {
        self.set_value_string(Some(&value.to_string()));
    }

    /// Java `setRangeValue(int, int)`.
    pub fn set_range_value(&self, start: i32, end: i32) {
        let text = self
            .state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .convert_range_to_field_text(start, end);
        self.text_field.set_text(&text);
    }

    /// Java `setValue()`.
    pub fn set_value_void(&self) {
        self.state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .msg_resetting_field_text();
        self.text_field.set_text("");
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        self.state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .msg_resetting_field_text();
        self.text_field.set_text("");
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(&self, value: i32) {
        self.set_value_string(Some(&value.to_string()));
    }

    /// Java `setValue(double)`.
    pub fn set_value_double(&self, value: f64) {
        self.set_value_string(Some(&java_lang_double_to_string(value)));
    }

    /// Java `setValue(long)`.
    pub fn set_value_long(&self, value: i64) {
        self.set_value_string(Some(&value.to_string()));
    }

    /// Java `getEndValue()`.
    pub fn get_end_value(&self) -> i32 {
        let text = self.text_field.get_text();
        self.state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .extract_end_value(Some(&text))
    }

    /// Java `getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::TEXT_FIELD
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<String> {
        Some(String::new())
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        eprintln!("Warning: A label cannot be set in FieldCell.");
        None
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> Option<String> {
        Some(self.text_field.get_text())
    }

    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        false
    }

    /// Java `getText(boolean, FieldDisplayer)`.  Field validation is currently not
    /// available for this class.
    pub fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, field_displayer1, None)
    }

    /// Java `getText(boolean, FieldDisplayer, FieldDisplayer)`.
    pub fn get_text_boolean_field_displayer_field_displayer(
        &self,
        _do_validation: bool,
        _field_displayer1: Option<&dyn FieldDisplayer>,
        _field_displayer2: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(Some(self.text_field.get_text()))
    }

    /// Java `getText()`.
    pub fn get_text_void(&self) -> Option<String> {
        Some(self.text_field.get_text())
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        false
    }

    /// Java `getIntValue()`.
    pub fn get_int_value(&self) -> i32 {
        match java_lang_integer_parse_int(&self.text_field.get_text()) {
            Ok(value) => value,
            // catch (NumberFormatException e)
            Err(_) => const_etomo_number::INTEGER_NULL_VALUE,
        }
    }

    /// Java `getDoubleValue()`.
    pub fn get_double_value(&self) -> f64 {
        match java_lang_double_value_of(&self.text_field.get_text()) {
            Ok(value) => value,
            // catch (NumberFormatException e)
            Err(_) => const_etomo_number::DOUBLE_NULL_VALUE,
        }
    }

    /// Java `getEtomoNumber()`.
    pub fn get_etomo_number_void(&self) -> ConstEtomoNumber {
        let text = self.text_field.get_text();
        self.state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .convert_to_etomo_number_string(Some(&text))
    }

    /// Java `getEtomoNumber(EtomoNumber.Type)`.
    pub fn get_etomo_number_type(&self, r#type: Type) -> ConstEtomoNumber {
        let text = self.text_field.get_text();
        self.state
            .borrow_mut()
            .as_mut()
            .expect("FieldCell: state is set by every constructor")
            .convert_to_etomo_number_type_string(Some(r#type), Some(&text))
    }

    /// Java `setForeground()`.
    pub fn set_foreground(&self) {
        if self.in_use.get() {
            self.text_field
                .set_foreground(Some(colors::CELL_FOREGROUND));
            // Swing painting: textField.setDisabledTextColor(Colors.CELL_FOREGROUND).
        } else {
            self.text_field
                .set_foreground(Some(colors::CELL_NOT_IN_USE_FOREGROUND));
            // Swing painting: textField.setDisabledTextColor(CELL_NOT_IN_USE_FOREGROUND).
        }
    }

    /// Java `getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: textField.getWidth().  Sizes are not modelled by the jdk
        // stand-in.
        0
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.text_field
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setTooltip(Field)`.
    pub fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            self.text_field
                .set_tool_tip_text(field.get_tooltip().as_deref());
        }
    }

    /// Java `setUnformattedTooltip(String)`.
    pub fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        *self.unformatted_tooltip.borrow_mut() = text.map(str::to_owned);
        self.unformatted_tooltip.borrow().clone()
    }

    /// Java `hasUnformattedTooltip()`.
    pub fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.borrow().is_some()
    }

    /// Java synchronized `useUnformattedTooltip(String, String)`.  Use
    /// unformattedTooltip to build a tooltip, and then delete unformattedTooltip.
    pub fn use_unformatted_tooltip(
        &self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        let unformatted_tooltip = self.unformatted_tooltip.borrow().clone();
        self.set_tool_tip_text(
            tooltip_formatter::INSTANCE
                .build_tooltip(unformatted_tooltip.as_deref(), param_descr, directive_descr)
                .as_deref(),
        );
        *self.unformatted_tooltip.borrow_mut() = None;
    }

    /// Java `getTooltip()`.
    pub fn get_tooltip(&self) -> Option<String> {
        self.text_field.get_tool_tip_text()
    }

    /// Java `equals(String)`.
    pub fn equals(&self, comp: Option<&str>) -> bool {
        // `getValue().equals(comp)`: getValue() is never null (the text field's
        // text), and `equals(null)` is false.
        self.get_value().as_deref() == comp && comp.is_some()
    }
}

impl CellVirtual for FieldCell {
    fn cell(&self) -> &Cell {
        &self.base
    }

    fn set_enabled(&self, enable: bool) {
        FieldCell::set_enabled(self, enable);
    }

    fn msg_label_changed(&self) {
        self.base.msg_label_changed();
    }

    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }
}

impl InputCellVirtual for FieldCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }

    fn get_component(&self) -> Rc<JComponent> {
        FieldCell::get_component(self)
    }

    fn get_field_type(&self) -> &'static UITestFieldType {
        FieldCell::get_field_type(self)
    }

    fn get_width(&self) -> i32 {
        FieldCell::get_width(self)
    }

    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        FieldCell::set_tool_tip_text(self, tool_tip_text);
    }

    fn get_text(&self) -> Option<String> {
        self.get_text_void()
    }

    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        FieldCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }

    fn get_name(&self) -> Option<String> {
        FieldCell::get_name(self)
    }

    fn set_locked(&self, locked: bool) {
        FieldCell::set_locked(self, locked);
    }

    fn set_editable(&self, editable: bool) {
        FieldCell::set_editable(self, editable);
    }

    fn is_locked(&self) -> bool {
        FieldCell::is_locked(self)
    }

    fn is_editable(&self) -> bool {
        FieldCell::is_editable(self)
    }

    fn is_enabled(&self) -> bool {
        FieldCell::is_enabled(self)
    }

    fn set_debug(&self, input: bool) {
        FieldCell::set_debug(self, input);
    }

    fn set_background_color_ui_resource(&self, color: ColorUIResource) {
        self.base.set_background_color_ui_resource_super(color);
    }
}

impl UIComponent for FieldCell {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }
}

impl SwingComponent for FieldCell {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }
}

/// Java `FieldCell implements ActionTarget`.
impl super::action_target::ActionTarget for FieldCell {
    fn set_target_file(&self, file: Option<&Path>) {
        FieldCell::set_target_file(self, file)
    }

    fn get_expanded_value(&self) -> Option<String> {
        FieldCell::get_expanded_value(self)
    }
}

/// The `etomo.ui.Field` interface the Java class implements.
impl Field for FieldCell {
    fn is_debug(&self) -> bool {
        InputCellVirtual::is_debug(self)
    }
    fn get_name(&self) -> Option<String> {
        FieldCell::get_name(self)
    }
    fn is_boolean(&self) -> bool {
        FieldCell::is_boolean(self)
    }
    fn is_text(&self) -> bool {
        FieldCell::is_text(self)
    }
    fn get_quoted_label(&self) -> Option<String> {
        FieldCell::get_quoted_label(self)
    }
    fn is_enabled(&self) -> bool {
        FieldCell::is_enabled(self)
    }
    fn clear(&self) {
        FieldCell::clear(self)
    }
    fn set_value_field(&self, from: Option<&dyn Field>) {
        FieldCell::set_value_field(self, from)
    }
    fn set_value_string(&self, text: Option<&str>) {
        FieldCell::set_value_string(self, text)
    }
    fn set_value_boolean(&self, bool_: bool) {
        FieldCell::set_value_boolean(self, bool_)
    }
    fn is_empty(&self) -> bool {
        FieldCell::is_empty(self)
    }
    fn is_selected(&self) -> bool {
        FieldCell::is_selected(self)
    }
    fn is_required(&self) -> bool {
        FieldCell::is_required(self)
    }
    fn get_text_void(&self) -> Option<String> {
        FieldCell::get_text_void(self)
    }
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        FieldCell::get_text_boolean_field_displayer(self, do_validation, field_displayer1.as_deref())
    }
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        FieldCell::get_text_boolean_field_displayer_field_displayer(
            self,
            do_validation,
            field_displayer1.as_deref(),
            field_displayer2.as_deref(),
        )
    }
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        FieldCell::get_directive_def(self)
    }
    fn use_default_value(&self) {
        FieldCell::use_default_value(self)
    }
    fn equals_default_value_void(&self) -> bool {
        FieldCell::equals_default_value_void(self)
    }
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        FieldCell::equals_default_value_string(self, value)
    }
    fn backup(&self) {
        FieldCell::backup(self)
    }
    fn restore_from_backup(&self) {
        FieldCell::restore_from_backup(self)
    }
    fn checkpoint(&self) {
        FieldCell::checkpoint(self)
    }
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        FieldCell::set_checkpoint(self, input)
    }
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        FieldCell::get_checkpoint(self).map(Rc::from)
    }
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        FieldCell::is_different_from_checkpoint(self, always_check)
    }
    fn is_field_highlight_set(&self) -> bool {
        FieldCell::is_field_highlight_set(self)
    }
    fn clear_field_highlight(&self) {
        FieldCell::clear_field_highlight(self)
    }
    fn set_field_highlight_field_setting_interface(&self, input: Option<&dyn FieldSettingInterface>) {
        FieldCell::set_field_highlight_field_setting_interface(self, input)
    }
    fn set_field_highlight_string(&self, input: Option<&str>) {
        FieldCell::set_field_highlight_string(self, input)
    }
    fn set_field_highlight_boolean(&self, input: bool) {
        FieldCell::set_field_highlight_boolean(self, input)
    }
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        FieldCell::get_field_highlight(self).map(Rc::from)
    }
    fn equals_field_highlight_void(&self) -> bool {
        FieldCell::equals_field_highlight_void(self)
    }
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        FieldCell::equals_field_highlight_string(self, value)
    }
    fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        FieldCell::set_tool_tip_text(self, tooltip)
    }
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        FieldCell::set_tooltip(self, field)
    }
    fn get_tooltip(&self) -> Option<String> {
        FieldCell::get_tooltip(self)
    }
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        InputCellVirtual::equals_selected_string_value(self, value)
    }
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        FieldCell::set_directive_def(self, directive_def)
    }
    fn get_description(&self) -> String {
        FieldCell::get_description(self)
            .unwrap_or_else(|| "null".to_owned())
    }
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        FieldCell::set_unformatted_tooltip(self, text)
    }
    fn has_unformatted_tooltip(&self) -> bool {
        FieldCell::has_unformatted_tooltip(self)
    }
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        FieldCell::use_unformatted_tooltip(self, param_descr, directive_descr)
    }
}

impl crate::imod::etomo::ui::text_field_interface::TextFieldInterface for FieldCell {}

/// The `etomo.ui.TableComponent` interface the Java class implements.
impl crate::imod::etomo::ui::table_component::TableComponent for FieldCell {
    fn get_preferred_width(&self) -> i32 {
        FieldCell::get_preferred_width(self)
    }
}
