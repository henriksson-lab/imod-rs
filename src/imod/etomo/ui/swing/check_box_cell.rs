//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxCell.java`.
//!
//! Java `final class CheckBoxCell extends InputCell implements ToggleCell,
//! ActionListener, BooleanFieldInterface, UIComponent, SwingComponent,
//! FieldSettings`: a table cell holding a check box.
//!
//! Every Java method body is an inherent method here (overloads carry the
//! parameter-type suffix of `ui.md`); the trait impls at the end bind
//! `CellVirtual`, `InputCellVirtual`, `ToggleCell`, `UIComponent` and
//! `SwingComponent` to those bodies.  `BooleanFieldInterface` and `FieldSettings`
//! are implemented by the inherent methods of the same names (see the report's
//! NEEDS).  The Java object registers itself as an `ActionListener` on its check box
//! (field highlight); that is a closure holding a `Weak` to the cell.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::cell::{Cell, CellVirtual};
use super::colors::{self, ColorUIResource};
use super::field_lock_controller::FieldLockController;
use super::input_cell::{InputCell, InputCellVirtual};
use super::swing_component::SwingComponent;
use super::toggle_cell::ToggleCell;
use super::toggle_coordinator::ToggleCoordinator;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ChangeListener, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};
use crate::imod::etomo::ui::boolean_field_setting::BooleanFieldSetting;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `CheckBoxCell`.
pub struct CheckBoxCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// Java `this`, for the places the source passes itself.
    this: Weak<CheckBoxCell>,
    /// Java `checkBox`.
    check_box: Rc<JComponent>,
    /// Java `fieldLockController`.
    field_lock_controller: Rc<FieldLockController>,
    /// Java `toggleCoordinator` (set once by the constructor, null when the cell does
    /// not have a header background).
    toggle_coordinator: RefCell<Option<Rc<ToggleCoordinator>>>,
    /// Java `unformattedLabel`: from JCheckBox.getText().  Updated in setLabel().
    unformatted_label: RefCell<Option<String>>,
    /// Java `checkpoint`.
    checkpoint: RefCell<Option<BooleanFieldSetting>>,
    /// Java `backupValue`.
    backup_value: std::cell::Cell<bool>,
    /// Java `fieldIsBackedUp`.
    field_is_backed_up: std::cell::Cell<bool>,
    /// Java `fieldHighlight`.
    field_highlight: RefCell<Option<BooleanFieldSetting>>,
    /// Java `directiveDef`.
    directive_def: RefCell<Option<DirectiveDef>>,
    /// Java `selectedStringValue`.
    selected_string_value: RefCell<Option<String>>,
    /// Java `unformattedTooltip`.
    unformatted_tooltip: RefCell<Option<String>>,
}

impl Deref for CheckBoxCell {
    type Target = InputCell;

    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl CheckBoxCell {
    /// Java private `CheckBoxCell(String, boolean, boolean)`.
    fn new(header_label: Option<&str>, header_background: bool, debug: bool) -> Rc<CheckBoxCell> {
        // super(headerBackground, debug)
        let check_box = JComponent::new_check_box("");
        let field_lock_controller =
            FieldLockController::get_toggle_button_instance_j_toggle_button_boolean(
                &check_box, debug,
            );
        let instance = Rc::new_cyclic(|this: &Weak<CheckBoxCell>| CheckBoxCell {
            base: InputCell::new_boolean_boolean(header_background, debug),
            this: this.clone(),
            check_box,
            field_lock_controller,
            toggle_coordinator: RefCell::new(None),
            unformatted_label: RefCell::new(Some(String::new())),
            checkpoint: RefCell::new(None),
            backup_value: std::cell::Cell::new(false),
            field_is_backed_up: std::cell::Cell::new(false),
            field_highlight: RefCell::new(None),
            directive_def: RefCell::new(None),
            selected_string_value: RefCell::new(None),
            unformatted_tooltip: RefCell::new(None),
        });
        let weak: Weak<CheckBoxCell> = Rc::downgrade(&instance);
        instance.base.set_this(weak);
        if header_background {
            *instance.toggle_coordinator.borrow_mut() =
                Some(ToggleCoordinator::new(Some(&instance)));
        } else {
            *instance.toggle_coordinator.borrow_mut() = None;
        }
        // Swing layout: checkBox.setBorderPainted(true);
        // checkBox.setBorder(BorderFactory.createEtchedBorder()).
        instance.set_background_void();
        instance.set_foreground();
        instance.set_font();
        if header_label.is_some() {
            instance.set_name_string(header_label);
        }
        instance
    }

    /// Java static `getInstance()`.
    pub fn get_instance() -> Rc<CheckBoxCell> {
        CheckBoxCell::new(None, false, false)
    }

    /// Java static `getHeaderBackgroundNamedInstance(String, String)`.
    pub fn get_header_background_named_instance(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
    ) -> Rc<CheckBoxCell> {
        CheckBoxCell::new(
            utilities::concatenate(header_label1, Some(" "), header_label2, None).as_deref(),
            true,
            false,
        )
    }

    /// Java static `getNamedInstance(String)`.
    pub fn get_named_instance_string(header_label: Option<&str>) -> Rc<CheckBoxCell> {
        CheckBoxCell::new(header_label, false, false)
    }

    /// Java static `getNamedInstance(String, boolean)`.
    pub fn get_named_instance_string_boolean(
        header_label: Option<&str>,
        debug: bool,
    ) -> Rc<CheckBoxCell> {
        CheckBoxCell::new(header_label, false, debug)
    }

    /// Java static `getNamedInstance(String, String, String)`.
    pub fn get_named_instance_string_string_string(
        header_label1: Option<&str>,
        header_label2: Option<&str>,
        header_label3: Option<&str>,
    ) -> Rc<CheckBoxCell> {
        CheckBoxCell::new(
            utilities::concatenate(header_label1, header_label2, header_label3, Some(" "))
                .as_deref(),
            false,
            false,
        )
    }

    /// Java final `addTarget(CheckBoxCell)`.
    pub fn add_target(&self, cbc_target: &Rc<CheckBoxCell>) {
        let toggle_coordinator = self.toggle_coordinator.borrow().clone();
        if let Some(toggle_coordinator) = toggle_coordinator {
            toggle_coordinator.add_target(Some(cbc_target));
        }
    }

    /// Java final `deleteTarget(CheckBoxCell)`.
    pub fn delete_target(&self, cbc_target: &Rc<CheckBoxCell>) {
        let toggle_coordinator = self.toggle_coordinator.borrow().clone();
        if let Some(toggle_coordinator) = toggle_coordinator {
            toggle_coordinator.delete_target(Some(cbc_target));
        }
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
        let field_type = &ui_test_field_type::CHECK_BOX;
        let name = utilities::convert_label_to_name(reference, field_type.is_unlimited_segments());
        if let Some(name) = name {
            self.check_box
                .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {} ",
                    self.check_box.get_name().as_deref().unwrap_or("null"),
                    DEFAULT_DELIMITER
                );
            }
        }
    }

    /// Java `isDebug()` (overrides `InputCell.isDebug`).
    pub fn is_debug(&self) -> bool {
        ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `getUniqueActionCommand()`: `getClass().getName() + "@" +
    /// Integer.toHexString(hashCode())`.  The identity hash code is the object's
    /// address here.
    pub fn get_unique_action_command(&self) -> Option<String> {
        Some(format!(
            "etomo.ui.swing.CheckBoxCell@{:x}",
            (self as *const CheckBoxCell as usize) as u32
        ))
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> String {
        format!(
            "[name:{},unformattedLabel:{},selected:{}]",
            self.check_box.get_name().as_deref().unwrap_or("null"),
            self.unformatted_label.borrow().as_deref().unwrap_or("null"),
            self.check_box.is_selected()
        )
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.check_box.get_name()
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.check_box.clone()
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::CHECK_BOX
    }

    /// Java `isText()`.
    pub fn is_text(&self) -> bool {
        false
    }

    /// Java `isBoolean()`.
    pub fn is_boolean(&self) -> bool {
        true
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Java `setSelectedStringValue(String)`.
    pub fn set_selected_string_value(&self, value: Option<&str>) {
        *self.selected_string_value.borrow_mut() = value.map(str::to_owned);
    }

    /// Java `equalsSelectedStringValue(String)` (overrides `InputCell`).
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        let selected_string_value = self.selected_string_value.borrow().clone();
        match selected_string_value {
            None => value.is_some_and(|value| !value.is_empty()),
            Some(selected_string_value) => Some(selected_string_value.as_str()) == value,
        }
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.  alwaysCheck - check for
    /// difference even when the field is disables or invisible.  Returns true if
    /// different from checkpoint or checkpoint is null.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.check_box.is_visible()) {
            return false;
        }
        let selected = self.is_selected();
        match self.checkpoint.borrow().as_ref() {
            None => true,
            Some(checkpoint) => !checkpoint.equals_boolean(selected),
        }
    }

    /// Java `backup()`.
    pub fn backup(&self) {
        self.backup_value.set(self.is_selected());
        self.field_is_backed_up.set(true);
    }

    /// Java `restoreFromBackup()`.  If the field was backed up, make the backup value
    /// the displayed value, and turn off the back up.
    pub fn restore_from_backup(&self) {
        if self.field_is_backed_up.get() {
            self.set_selected_boolean(self.backup_value.get());
            self.field_is_backed_up.set(false);
        }
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        self.set_selected_boolean(false);
    }

    /// Java `checkpoint()`.  Constructs savedValue (if it doesn't exist).  Saves the
    /// current setting.
    pub fn checkpoint(&self) {
        let selected = self.is_selected();
        let mut checkpoint = self.checkpoint.borrow_mut();
        if checkpoint.is_none() {
            *checkpoint = Some(BooleanFieldSetting::new());
        }
        checkpoint.as_mut().unwrap().set_boolean(selected);
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    pub fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        let mut checkpoint = self.checkpoint.borrow_mut();
        if checkpoint.is_none() && input.is_some_and(|input| input.is_set() && input.is_boolean()) {
            *checkpoint = Some(BooleanFieldSetting::new());
        }
        if let Some(checkpoint) = checkpoint.as_mut() {
            checkpoint.copy(input);
        }
    }

    /// Java `getCheckpoint()`.  (A copy of the setting.)
    pub fn get_checkpoint(&self) -> Option<BooleanFieldSetting> {
        self.checkpoint.borrow().clone()
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.set_background_void();
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.set_background_void();
        }
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        if self.field_lock_controller.set_enabled(enabled) {
            self.set_background_void();
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

    /// Java `setLabel(String)`.
    pub fn set_label(&self, label: Option<&str>) {
        *self.unformatted_label.borrow_mut() = label.map(str::to_owned);
        self.set_foreground();
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
        self.check_box.set_text(&text);
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        self.unformatted_label.borrow().clone()
    }

    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        false
    }

    /// Java `getText(boolean, FieldDisplayer)`.  The checkbox label is not validated.
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
        Ok(self.unformatted_label.borrow().clone())
    }

    /// Java `getText()`.
    pub fn get_text_void(&self) -> Option<String> {
        self.unformatted_label.borrow().clone()
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        let unformatted_label = self.unformatted_label.borrow().clone();
        utilities::quote_label(unformatted_label.as_deref())
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        let mut etomo_boolean = EtomoBoolean2::new();
        etomo_boolean.set_string(value);
        let selected = etomo_boolean.is();
        self.check_box.set_selected(selected);
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected_boolean(&self, selected: bool) {
        self.check_box.set_selected(selected);
    }

    /// Java `setSelected(ConstEtomoNumber, boolean)`.  Prevents a field from being
    /// overridden by an empty value.  allowEmpty: allow field to be overridden by
    /// null.
    ///
    /// Upstream bug fixed (CheckBoxCell.java:365-372): the source's second test is a
    /// plain `if (allowEmpty)`, so with `allowEmpty` a non-null value is set and then
    /// immediately cleared, and a non-null `true` can never be set that way.  Per the
    /// method's documentation only an empty value may override the field when
    /// `allowEmpty` is set, so the second test applies only when `selected` is empty.
    pub fn set_selected_const_etomo_number_boolean(
        &self,
        selected: Option<&ConstEtomoNumber>,
        allow_empty: bool,
    ) {
        if let Some(selected) = selected
            && !selected.is_null()
        {
            self.set_selected_boolean(selected.is());
        } else if allow_empty {
            self.set_selected_boolean(false);
        }
    }

    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&self, selected: bool) {
        self.check_box.set_selected(selected);
    }

    /// Java `setValue(Field)`.
    pub fn set_value_field(&self, input: Option<&dyn Field>) {
        match input {
            None => self.clear(),
            Some(input) => self.set_selected_boolean(input.is_selected()),
        }
    }

    /// Java `setActionCommand(String)`.
    pub fn set_action_command(&self, input: Option<&str>) {
        self.check_box.set_action_command(input);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.check_box.get_action_command()
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.check_box.add_action_listener(action_listener);
    }

    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&self, listener: ChangeListener) {
        self.check_box.add_change_listener(listener);
    }

    /// Java private `setForeground()`.
    fn set_foreground(&self) {
        self.check_box.set_foreground(Some(colors::CELL_FOREGROUND));
        self.set_html_label(colors::CELL_FOREGROUND);
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        *self.directive_def.borrow_mut() = directive_def;
    }

    /// Java `getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.borrow().clone()
    }

    /// Java `isFieldHighlightSet()`.
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.is_set())
    }

    /// The cell as the check box's `ActionListener` (Java `this`).
    fn self_action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(this) = this.upgrade() {
                this.action_performed(event);
            }
        })
    }

    /// Java `setFieldHighlight(boolean)`.
    pub fn set_field_highlight_boolean(&self, value: bool) {
        if self.field_highlight.borrow().is_none() {
            *self.field_highlight.borrow_mut() = Some(BooleanFieldSetting::new());
            self.check_box
                .add_action_listener(self.self_action_listener());
        }
        self.field_highlight
            .borrow_mut()
            .as_mut()
            .unwrap()
            .set_boolean(value);
        self.update_field_highlight();
    }

    /// Java `setFieldHighlight(String)`.
    pub fn set_field_highlight_string(&self, _value: Option<&str>) {}

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    pub fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        if self.field_highlight.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_boolean())
        {
            *self.field_highlight.borrow_mut() = Some(BooleanFieldSetting::new());
            self.check_box
                .add_action_listener(self.self_action_listener());
        }
        let exists = match self.field_highlight.borrow_mut().as_mut() {
            Some(field_highlight) => {
                field_highlight.copy(input);
                true
            }
            None => false,
        };
        if exists {
            self.update_field_highlight();
        }
    }

    /// Java `equalsFieldHighlight()`.
    pub fn equals_field_highlight_void(&self) -> bool {
        let selected = self.is_selected();
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| {
                field_highlight.is_set() && field_highlight.equals_boolean(selected)
            })
    }

    /// Java `equalsFieldHighlight(String)`.
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        let value = value.is_some_and(|value| !java_lang_string_matches_whitespace(value));
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| {
                field_highlight.is_set() && field_highlight.equals_boolean(value)
            })
    }

    /// Java `clearFieldHighlight()`.
    pub fn clear_field_highlight(&self) {
        let cleared = match self.field_highlight.borrow_mut().as_mut() {
            Some(field_highlight) if field_highlight.is_set() => {
                field_highlight.reset();
                true
            }
            _ => false,
        };
        if cleared {
            // Turn off field highlight - parameter doesn't matter since field
            // highlight is off.
            self.update_field_highlight();
        }
    }

    /// Java `getFieldHighlight()`.  (A copy of the setting.)
    pub fn get_field_highlight(&self) -> Option<BooleanFieldSetting> {
        self.field_highlight.borrow().clone()
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _e: &ActionEvent) {
        self.update_field_highlight();
        self.field_lock_controller
            .apply_toggle_button_selection_state();
    }

    /// Java `updateFieldHighlight()`.
    pub fn update_field_highlight(&self) {
        let selected = self.is_selected();
        let highlight = match self.field_highlight.borrow().as_ref() {
            Some(field_highlight) if field_highlight.is_set() => {
                Some(field_highlight.is_value() == selected)
            }
            _ => None,
        };
        match highlight {
            Some(true) => self.check_box.set_foreground(Some(colors::FIELD_HIGHLIGHT)),
            Some(false) => self.check_box.set_foreground(Some(colors::CELL_FOREGROUND)),
            None => {}
        }
    }

    /// Java `useDefaultValue()`.
    pub fn use_default_value(&self) {
        eprintln!("Warning: CheckBoxCell.useDefaultValue has not been implemented");
    }

    /// Java `equalsDefaultValue()`.
    pub fn equals_default_value_void(&self) -> bool {
        false
    }

    /// Java `equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, _value: Option<&str>) -> bool {
        false
    }

    /// Java `getHeight()`.
    pub fn get_height(&self) -> i32 {
        // Swing geometry: checkBox.getHeight() + the border's bottom inset - 1.  Sizes
        // are not modelled by the jdk stand-in.
        0
    }

    /// Java `getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: checkBox.getWidth().  Not modelled by the jdk stand-in.
        0
    }

    // Java `getLeftBorder()` returns the check box border's left inset: Swing
    // geometry, not modelled by the jdk stand-in.

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.check_box
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setTooltip(Field)`.
    pub fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            self.check_box
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
        self.check_box.get_tool_tip_text()
    }

    /// Java `addTooltip(String)`.
    pub fn add_tooltip(&self, text: Option<&str>) {
        let Some(text) = text else {
            return;
        };
        let tooltip = self.check_box.get_tool_tip_text();
        match tooltip {
            None => self.set_tool_tip_text(Some(text)),
            Some(tooltip) => {
                let formatted = tooltip_formatter::INSTANCE.format(Some(text));
                self.check_box.set_tool_tip_text(Some(&format!(
                    "{} & {}",
                    tooltip,
                    formatted.as_deref().unwrap_or("null")
                )));
            }
        }
    }
}

impl CellVirtual for CheckBoxCell {
    fn cell(&self) -> &Cell {
        &self.base
    }

    fn set_enabled(&self, enable: bool) {
        CheckBoxCell::set_enabled(self, enable);
    }

    fn msg_label_changed(&self) {
        self.base.msg_label_changed();
    }

    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }
}

impl InputCellVirtual for CheckBoxCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }

    fn get_component(&self) -> Rc<JComponent> {
        CheckBoxCell::get_component(self)
    }

    fn get_field_type(&self) -> &'static UITestFieldType {
        CheckBoxCell::get_field_type(self)
    }

    fn get_width(&self) -> i32 {
        CheckBoxCell::get_width(self)
    }

    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        CheckBoxCell::set_tool_tip_text(self, tool_tip_text);
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
        CheckBoxCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }

    fn get_name(&self) -> Option<String> {
        CheckBoxCell::get_name(self)
    }

    fn set_locked(&self, locked: bool) {
        CheckBoxCell::set_locked(self, locked);
    }

    fn set_editable(&self, editable: bool) {
        CheckBoxCell::set_editable(self, editable);
    }

    fn is_locked(&self) -> bool {
        CheckBoxCell::is_locked(self)
    }

    fn is_editable(&self) -> bool {
        CheckBoxCell::is_editable(self)
    }

    fn is_enabled(&self) -> bool {
        CheckBoxCell::is_enabled(self)
    }

    fn is_debug(&self) -> bool {
        CheckBoxCell::is_debug(self)
    }

    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        CheckBoxCell::equals_selected_string_value(self, value)
    }
}

impl ToggleCell for CheckBoxCell {
    fn get_label(&self) -> Option<String> {
        CheckBoxCell::get_label(self)
    }

    fn set_label(&self, label: Option<&str>) {
        CheckBoxCell::set_label(self, label);
    }

    fn set_selected(&self, selected: bool) {
        self.set_selected_boolean(selected);
    }

    fn add_action_listener(&self, action_listener: ActionListener) {
        CheckBoxCell::add_action_listener(self, action_listener);
    }

    fn add(&self, panel: &Rc<JComponent>) {
        self.base.add(panel);
    }

    fn is_selected(&self) -> bool {
        CheckBoxCell::is_selected(self)
    }

    fn get_height(&self) -> i32 {
        CheckBoxCell::get_height(self)
    }

    fn get_width(&self) -> i32 {
        CheckBoxCell::get_width(self)
    }

    fn set_warning(&self, warning: bool) {
        self.base.set_warning_boolean(warning);
    }

    fn add_change_listener(&self, listener: ChangeListener) {
        CheckBoxCell::add_change_listener(self, listener);
    }

    fn set_enabled(&self, enabled: bool) {
        CheckBoxCell::set_enabled(self, enabled);
    }

    fn is_enabled(&self) -> bool {
        CheckBoxCell::is_enabled(self)
    }
}

impl UIComponent for CheckBoxCell {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.check_box.clone()
    }
}

impl SwingComponent for CheckBoxCell {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.check_box.clone()
    }
}

/// The `etomo.ui.Field` interface the Java class implements.
impl Field for CheckBoxCell {
    fn is_debug(&self) -> bool {
        CheckBoxCell::is_debug(self)
    }
    fn get_name(&self) -> Option<String> {
        CheckBoxCell::get_name(self)
    }
    fn is_boolean(&self) -> bool {
        CheckBoxCell::is_boolean(self)
    }
    fn is_text(&self) -> bool {
        CheckBoxCell::is_text(self)
    }
    fn get_quoted_label(&self) -> Option<String> {
        CheckBoxCell::get_quoted_label(self)
    }
    fn is_enabled(&self) -> bool {
        CheckBoxCell::is_enabled(self)
    }
    fn clear(&self) {
        CheckBoxCell::clear(self)
    }
    fn set_value_field(&self, from: Option<&dyn Field>) {
        CheckBoxCell::set_value_field(self, from)
    }
    fn set_value_string(&self, text: Option<&str>) {
        CheckBoxCell::set_value_string(self, text)
    }
    fn set_value_boolean(&self, bool_: bool) {
        CheckBoxCell::set_value_boolean(self, bool_)
    }
    fn is_empty(&self) -> bool {
        CheckBoxCell::is_empty(self)
    }
    fn is_selected(&self) -> bool {
        CheckBoxCell::is_selected(self)
    }
    fn is_required(&self) -> bool {
        CheckBoxCell::is_required(self)
    }
    fn get_text_void(&self) -> Option<String> {
        CheckBoxCell::get_text_void(self)
    }
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        CheckBoxCell::get_text_boolean_field_displayer(self, do_validation, field_displayer1.as_deref())
    }
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        CheckBoxCell::get_text_boolean_field_displayer_field_displayer(
            self,
            do_validation,
            field_displayer1.as_deref(),
            field_displayer2.as_deref(),
        )
    }
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        CheckBoxCell::get_directive_def(self)
    }
    fn use_default_value(&self) {
        CheckBoxCell::use_default_value(self)
    }
    fn equals_default_value_void(&self) -> bool {
        CheckBoxCell::equals_default_value_void(self)
    }
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        CheckBoxCell::equals_default_value_string(self, value)
    }
    fn backup(&self) {
        CheckBoxCell::backup(self)
    }
    fn restore_from_backup(&self) {
        CheckBoxCell::restore_from_backup(self)
    }
    fn checkpoint(&self) {
        CheckBoxCell::checkpoint(self)
    }
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        CheckBoxCell::set_checkpoint(self, input)
    }
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        CheckBoxCell::get_checkpoint(self)
            .map(|setting| Rc::new(setting) as Rc<dyn FieldSettingInterface>)
    }
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        CheckBoxCell::is_different_from_checkpoint(self, always_check)
    }
    fn is_field_highlight_set(&self) -> bool {
        CheckBoxCell::is_field_highlight_set(self)
    }
    fn clear_field_highlight(&self) {
        CheckBoxCell::clear_field_highlight(self)
    }
    fn set_field_highlight_field_setting_interface(&self, input: Option<&dyn FieldSettingInterface>) {
        CheckBoxCell::set_field_highlight_field_setting_interface(self, input)
    }
    fn set_field_highlight_string(&self, input: Option<&str>) {
        CheckBoxCell::set_field_highlight_string(self, input)
    }
    fn set_field_highlight_boolean(&self, input: bool) {
        CheckBoxCell::set_field_highlight_boolean(self, input)
    }
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        CheckBoxCell::get_field_highlight(self)
            .map(|setting| Box::new(setting) as Box<dyn FieldSettingInterface>)
            .map(Rc::from)
    }
    fn equals_field_highlight_void(&self) -> bool {
        CheckBoxCell::equals_field_highlight_void(self)
    }
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        CheckBoxCell::equals_field_highlight_string(self, value)
    }
    fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        CheckBoxCell::set_tool_tip_text(self, tooltip)
    }
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        CheckBoxCell::set_tooltip(self, field)
    }
    fn get_tooltip(&self) -> Option<String> {
        CheckBoxCell::get_tooltip(self)
    }
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        CheckBoxCell::equals_selected_string_value(self, value)
    }
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        CheckBoxCell::set_directive_def(self, directive_def)
    }
    fn get_description(&self) -> String {
        CheckBoxCell::get_description(self)
            .unwrap_or_else(|| "null".to_owned())
    }
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        CheckBoxCell::set_unformatted_tooltip(self, text)
    }
    fn has_unformatted_tooltip(&self) -> bool {
        CheckBoxCell::has_unformatted_tooltip(self)
    }
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        CheckBoxCell::use_unformatted_tooltip(self, param_descr, directive_descr)
    }
}

impl crate::imod::etomo::ui::boolean_field_interface::BooleanFieldInterface for CheckBoxCell {}

/// The `etomo.type.FieldSettings` interface the Java class implements.
impl crate::imod::etomo::r#type::field_settings::FieldSettings for CheckBoxCell {
    fn set_selected(&self, selected: bool) {
        CheckBoxCell::set_selected_boolean(self, selected)
    }
    fn is_selected(&self) -> bool {
        CheckBoxCell::is_selected(self)
    }
    fn is_editable(&self) -> bool {
        CheckBoxCell::is_editable(self)
    }
    fn set_editable(&self, editable: bool) {
        CheckBoxCell::set_editable(self, editable)
    }
    fn set_enabled(&self, enabled: bool) {
        CheckBoxCell::set_enabled(self, enabled)
    }
    fn is_enabled(&self) -> bool {
        CheckBoxCell::is_enabled(self)
    }
}
