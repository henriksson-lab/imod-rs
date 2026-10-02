//! `IMOD/Etomo/src/etomo/ui/swing/CheckTextField.java`.
//!
//! A check box followed by a text field that is enabled only when the check box is
//! checked.  Implements StateChangeSource with its state equal to whether it has
//! changed since it was checkpointed.
//!
//! The `CheckBox` and `TextField` members that come from the `Field` interface are
//! called through the trait (`Field::...`), the others as their own methods.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::swing_component::SwingComponent;
use super::text_field::TextField;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionListener, Dimension, DocumentListener, JComponent};
use crate::imod::etomo::logic::field_validator::FieldValidator;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type as EtomoNumberType, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::ui::boolean_field_interface::BooleanFieldInterface;
use crate::imod::etomo::ui::boolean_text_field_interface::BooleanTextFieldInterface;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_bundle::FieldSettingBundle;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::text_field_interface::TextFieldInterface;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `public final class CheckTextField implements UIComponent, SwingComponent,
/// BooleanTextFieldInterface`.
pub struct CheckTextField {
    /// This object, for the listener and for `FieldValidator`'s `this`.
    self_ref: RefCell<Weak<CheckTextField>>,
    /// Java final `pnlRoot` (`new JPanel()`).
    pnl_root: Rc<JComponent>,
    /// Java final `checkBox`.
    check_box: Rc<CheckBox>,
    /// Java final `textField`.
    text_field: Rc<TextField>,
    /// Java final `label`.
    label: String,
    /// Java final `numericType` (stored, never read, in the Java too).
    #[allow(dead_code)]
    numeric_type: Option<EtomoNumberType>,
    /// Java final `fieldType`.
    field_type: FieldType,
    /// Java `required`.
    required: Cell<bool>,
    /// Java `directiveDef`.
    directive_def: Cell<Option<DirectiveDef>>,
    /// Java `debug`.
    debug: Cell<bool>,
}

impl CheckTextField {
    /// Java private `CheckTextField(FieldType, String, EtomoNumber.Type)`.
    fn new(
        field_type: FieldType,
        label: &str,
        numeric_type: Option<EtomoNumberType>,
    ) -> Rc<CheckTextField> {
        let text_field = TextField::new(field_type, Some(label), None);
        let check_box = CheckBox::new_void();
        let instance = Rc::new(CheckTextField {
            self_ref: RefCell::new(Weak::new()),
            pnl_root: JComponent::new_panel(),
            check_box,
            text_field,
            label: label.to_owned(),
            numeric_type,
            field_type,
            required: Cell::new(false),
            directive_def: Cell::new(None),
            debug: Cell::new(false),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        instance.set_label(label);
        instance
    }

    /// Java public static `getInstance(FieldType, String)`.
    pub fn get_instance(field_type: FieldType, label: &str) -> Rc<CheckTextField> {
        let instance = CheckTextField::new(field_type, label, None);
        instance.create_panel();
        instance.update_display();
        instance.add_listeners();
        instance
    }

    /// Java public static `getNumericInstance(FieldType, String, EtomoNumber.Type)`.
    pub fn get_numeric_instance(
        field_type: FieldType,
        tf_label: &str,
        numeric_type: Option<EtomoNumberType>,
    ) -> Rc<CheckTextField> {
        let instance = CheckTextField::new(field_type, tf_label, numeric_type);
        instance.create_panel();
        instance.update_display();
        instance.add_listeners();
        instance
    }

    /// Java `@Override getName()`.
    pub fn get_name(&self) -> Option<String> {
        Field::get_name(&*self.check_box)
    }

    /// Java `@Override equals(Object)`: true when `object` is this field's check box
    /// or text field.
    pub fn equals_object(&self, object: &dyn Any) -> bool {
        let object = object as *const dyn Any as *const ();
        std::ptr::eq(object, Rc::as_ptr(&self.check_box) as *const ())
            || std::ptr::eq(object, Rc::as_ptr(&self.text_field) as *const ())
    }

    /// Java public `equals(Document)`.
    pub fn equals_document(&self, document: &Rc<JComponent>) -> bool {
        Rc::ptr_eq(&self.text_field.get_document(), document)
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&self, label: &str) {
        self.check_box.set_text(Some(label));
        self.text_field.set_name(Some(label));
    }

    /// Java public `setAlternateLabel(String)`.
    pub fn set_alternate_label(&self, label: Option<&str>) {
        self.check_box.set_alternate_text(label);
        self.text_field.set_name(label);
    }

    /// Java public `switchLabels(boolean)`.
    pub fn switch_labels(&self, alternate: bool) {
        self.check_box.switch_text(alternate);
        // Set text field name to newly set text
        self.text_field
            .set_name(Field::get_text_void(&*self.check_box).as_deref());
    }

    /// Java `checkpoint(boolean, String)`.  Checkpoints checkbox from checkboxValue.
    /// Saves textValue as the text field checkpoint.
    pub fn checkpoint_boolean_string(&self, checkbox_value: bool, text_value: Option<&str>) {
        self.check_box.checkpoint_boolean(checkbox_value);
        self.text_field.checkpoint_string(text_value);
    }

    /// Java `@Override checkpoint()`.
    pub fn checkpoint_void(&self) {
        Field::checkpoint(&*self.check_box);
        Field::checkpoint(&*self.text_field);
    }

    /// Java `resetToCheckpoint()`.  Resets to checkpointValue if checkpointValue has
    /// been set.  Otherwise has no effect.
    pub fn reset_to_checkpoint(&self) {
        self.check_box.reset_to_checkpoint();
        self.text_field.reset_to_checkpoint();
    }

    /// Java `setColumns()`.
    pub fn set_columns(&self) {
        // Java: `if (fieldType != null)`; the field type is never null here.
        self.text_field.set_columns_void();
    }

    /// Java `isDifferentFromCheckpoint()`.  True if checkBox is visible, enabled, and
    /// different from checkpoint or text field is visible, enabled, and different
    /// from checkpoint.
    pub fn is_different_from_checkpoint_void(&self) -> bool {
        Field::is_different_from_checkpoint(&*self.check_box, false)
            || Field::is_different_from_checkpoint(&*self.text_field, false)
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
        self.text_field.set_debug(input);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enable: bool) {
        self.check_box.set_enabled(enable);
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        // checkBox can handle enabled versus editable
        self.text_field.set_enabled(
            Field::is_enabled(&*self.check_box) && Field::is_selected(&*self.check_box),
        );
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        // checkBox can handle enabled versus editable
        self.check_box.set_editable(editable);
        self.text_field.set_editable(editable);
    }

    /// Java `@Override isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        Field::is_enabled(&*self.check_box)
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // root panel
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.X_AXIS)).
        self.pnl_root.add(&self.check_box.get_component());
        self.pnl_root.add(&self.text_field.get_component());
    }

    /// Java `@Override setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.text_field, text);
        Field::set_tool_tip_text(&*self.check_box, text);
    }

    /// Java `@Override setUnformattedTooltip(String)`.
    pub fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        Field::set_unformatted_tooltip(&*self.text_field, text);
        Field::set_unformatted_tooltip(&*self.check_box, text)
    }

    /// Java `@Override hasUnformattedTooltip()`.
    pub fn has_unformatted_tooltip(&self) -> bool {
        Field::has_unformatted_tooltip(&*self.text_field)
            || Field::has_unformatted_tooltip(&*self.check_box)
    }

    /// Java `@Override synchronized useUnformattedTooltip(String, String)`.  Use
    /// unformattedTooltip to build a tooltip, and then delete unformattedTooltip.
    /// (The lock has no Rust counterpart: fields live on the EDT.)
    pub fn use_unformatted_tooltip(
        &self,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) {
        Field::use_unformatted_tooltip(&*self.text_field, param_descr, directive_descr);
        Field::use_unformatted_tooltip(&*self.check_box, param_descr, directive_descr);
    }

    /// Java public `setCheckBoxUnformattedTooltip(String)`.
    pub fn set_check_box_unformatted_tooltip(&self, text: Option<&str>) {
        Field::set_unformatted_tooltip(&*self.check_box, text);
    }

    /// Java public `setFieldUnformattedTooltip(String)`.
    pub fn set_field_unformatted_tooltip(&self, text: Option<&str>) {
        Field::set_unformatted_tooltip(&*self.text_field, text);
    }

    /// Java public `setAlternateTooltipText(String)`.  Stores a second tooltip.  Does
    /// not switch to the second tooltip (see `switchTooltips`).
    pub fn set_alternate_tooltip_text(&self, text: Option<&str>) {
        self.text_field.set_alternate_tooltip_text(text);
        self.check_box.set_alternate_tooltip_text(text);
    }

    /// Java public `switchTooltips(boolean)`.  Switch to/from the alternate tooltip.
    pub fn switch_tooltips(&self, alternate: bool) {
        self.text_field.switch_tooltips(alternate);
        self.check_box.switch_tooltips(alternate);
    }

    /// Java public `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java `@Override setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java `@Override backup()`.
    pub fn backup(&self) {
        Field::backup(&*self.check_box);
        Field::backup(&*self.text_field);
    }

    /// Java `@Override clear()`.
    pub fn clear(&self) {
        Field::clear(&*self.check_box);
        Field::clear(&*self.text_field);
        self.update_display();
    }

    /// Java `@Override clearFieldHighlight()`.
    pub fn clear_field_highlight(&self) {
        Field::clear_field_highlight(&*self.check_box);
        Field::clear_field_highlight(&*self.text_field);
    }

    /// Java `@Override equalsDefaultValue()`.
    pub fn equals_default_value_void(&self) -> bool {
        Field::equals_default_value_void(&*self.check_box)
            && Field::equals_default_value_void(&*self.text_field)
    }

    /// Java `@Override equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        Field::equals_default_value_string(&*self.check_box, value)
            && Field::equals_default_value_string(&*self.text_field, value)
    }

    /// Java `@Override equalsFieldHighlight()`.
    pub fn equals_field_highlight_void(&self) -> bool {
        Field::equals_field_highlight_void(&*self.check_box)
            && Field::equals_field_highlight_void(&*self.text_field)
    }

    /// Java `@Override equalsFieldHighlight(String)`.
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        Field::equals_field_highlight_string(&*self.check_box, value)
            && Field::equals_field_highlight_string(&*self.text_field, value)
    }

    /// Java `@Override equalsSelectedStringValue(String)`.
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        Field::equals_selected_string_value(&*self.check_box, value)
    }

    /// Java `@Override getTooltip()`.
    pub fn get_tooltip(&self) -> Option<String> {
        Field::get_tooltip(&*self.text_field)
    }

    /// Java `@Override isBoolean()`.
    pub fn is_boolean(&self) -> bool {
        true
    }

    /// Java `@Override isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug.get() || ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `@Override isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint_boolean(&self, always_check: bool) -> bool {
        Field::is_different_from_checkpoint(&*self.check_box, always_check)
            || Field::is_different_from_checkpoint(&*self.text_field, always_check)
    }

    /// Java `@Override getCheckpoint()`.
    pub fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        let bundle = FieldSettingBundle::new();
        bundle.add_boolean_setting(Field::get_checkpoint(&*self.check_box).as_deref());
        bundle.add_text_setting(Field::get_checkpoint(&*self.text_field).as_deref());
        Some(Rc::new(bundle))
    }

    /// Java `@Override getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `@Override getFieldHighlight()`.
    pub fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        let bundle = FieldSettingBundle::new();
        bundle.add_boolean_setting(Field::get_field_highlight(&*self.check_box).as_deref());
        bundle.add_text_setting(Field::get_field_highlight(&*self.text_field).as_deref());
        Some(Rc::new(bundle))
    }

    /// Java `@Override isEmpty()`.
    pub fn is_empty(&self) -> bool {
        let text = Field::get_text_void(&*self.text_field);
        text.as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `@Override isFieldHighlightSet()`.
    pub fn is_field_highlight_set(&self) -> bool {
        Field::is_field_highlight_set(&*self.check_box)
            || Field::is_field_highlight_set(&*self.text_field)
    }

    /// Java `@Override isRequired()`.
    pub fn is_required(&self) -> bool {
        Field::is_required(&*self.text_field)
    }

    /// Java `@Override isText()`.
    pub fn is_text(&self) -> bool {
        true
    }

    /// Java `@Override restoreFromBackup()`.  If a field was backed up, make the
    /// backup value the displayed value, and turn off the back up.  This has no
    /// effect on a check box with a backupValue of false, other then to turn off the
    /// backup.
    pub fn restore_from_backup(&self) {
        Field::restore_from_backup(&*self.check_box);
        Field::restore_from_backup(&*self.text_field);
        self.update_display();
    }

    /// Java `@Override setCheckpoint(FieldSettingInterface)`.
    pub fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        Field::set_checkpoint(&*self.check_box, input);
        Field::set_checkpoint(&*self.text_field, input);
    }

    /// Java `@Override setFieldHighlight(boolean)`.
    pub fn set_field_highlight_boolean(&self, bool_: bool) {
        Field::set_field_highlight_boolean(&*self.check_box, bool_);
    }

    /// Java `@Override setFieldHighlight(FieldSettingInterface)`.
    pub fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        Field::set_field_highlight_field_setting_interface(&*self.check_box, input);
        Field::set_field_highlight_field_setting_interface(&*self.text_field, input);
    }

    /// Java `@Override setFieldHighlight(String)`.
    pub fn set_field_highlight_string(&self, text: Option<&str>) {
        Field::set_field_highlight_string(&*self.text_field, text);
    }

    /// Java `@Override setTooltip(Field)`.
    pub fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            let tooltip = field.get_tooltip();
            self.check_box.set_preformatted_tooltip(tooltip.as_deref());
            self.text_field.set_preformatted_tooltip(tooltip.as_deref());
        }
    }

    /// Java `@Override setValue(boolean)`.
    pub fn set_value_boolean(&self, value: bool) {
        Field::set_value_boolean(&*self.check_box, value);
        self.update_display();
    }

    /// Java `@Override setValue(Field)`.
    pub fn set_value_field(&self, input: Option<&dyn Field>) {
        Field::set_value_field(&*self.check_box, input);
        Field::set_value_field(&*self.text_field, input);
        self.update_display();
    }

    /// Java `@Override setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        Field::set_value_string(&*self.text_field, value);
    }

    /// Java `@Override useDefaultValue()`.
    pub fn use_default_value(&self) {
        Field::use_default_value(&*self.check_box);
        Field::use_default_value(&*self.text_field);
        self.update_display();
    }

    /// Java `setCheckBoxToolTipText(String)`.
    pub fn set_check_box_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.check_box, text);
    }

    /// Java `setFieldToolTipText(String)`.
    pub fn set_field_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.text_field, text);
    }

    /// Java `getRootComponent()`.
    pub fn get_root_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // checkBox.addActionListener(new CheckTextFieldActionListener(this))
        let check_text_field = self.self_ref.borrow().clone();
        self.check_box
            .add_action_listener(Some(Rc::new(move |_action_event| {
                // CheckTextFieldActionListener.actionPerformed
                if let Some(check_text_field) = check_text_field.upgrade() {
                    check_text_field.action();
                }
            })));
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.check_box.add_action_listener(Some(action_listener));
    }

    /// Java `addDocumentListener(DocumentListener)`.
    pub fn add_document_listener(&self, listener: DocumentListener) {
        self.text_field
            .get_document()
            .add_document_listener(listener);
    }

    /// Java public `setText(String)`.
    pub fn set_text_string(&self, input: Option<&str>) {
        self.text_field.set_text_string(input);
    }

    /// Java `setText(String, boolean)`.
    pub fn set_text_string_boolean(&self, input: Option<&str>, nonempty: bool) {
        if !nonempty || input.is_some_and(|input| !input.is_empty()) {
            self.set_text_string(input);
        }
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.check_box.get_action_command()
    }

    /// Java public `getLabel()`.
    pub fn get_label(&self) -> &str {
        &self.label
    }

    /// Java public `setSelected(boolean)`.
    pub fn set_selected_boolean(&self, selected: bool) {
        self.check_box.set_selected_boolean(selected);
        self.update_display();
    }

    /// Java `setSelected(ConstEtomoNumber, boolean)`.
    ///
    /// Upstream bug fixed in translation (`CheckTextField.java:481`): with
    /// `allowEmpty` true and a null `selected`, the Java calls `selected.is()` and
    /// throws a NullPointerException; here a null `selected` leaves the field
    /// unchanged, as it does when `allowEmpty` is false.
    pub fn set_selected_const_etomo_number_boolean(
        &self,
        selected: Option<&ConstEtomoNumber>,
        allow_empty: bool,
    ) {
        if let Some(selected) = selected {
            if allow_empty || !selected.is_null() {
                self.set_selected_boolean(selected.is());
            }
        }
    }

    /// Java `@Override isSelected()`.
    pub fn is_selected(&self) -> bool {
        Field::is_selected(&*self.check_box)
    }

    /// Java `@Override getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `setRequired(boolean)`.
    pub fn set_required(&self, required: bool) {
        self.required.set(required);
    }

    /// Java public `getText(boolean)`.
    pub fn get_text_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer(do_validation, None)
    }

    /// Java `@Override getText(boolean, FieldDisplayer)`.
    pub fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, field_displayer1, None)
    }

    /// Java `@Override getText(boolean, FieldDisplayer, FieldDisplayer)`.
    pub fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut text = Field::get_text_void(&*self.text_field);
        if do_validation && Field::is_enabled(&*self.text_field) {
            text = FieldValidator::validate_text_string_field_type_ui_component_string_boolean_boolean_boolean_validation_set_field_displayer_field_displayer(
                text.as_deref(),
                Some(self.field_type),
                Some(self),
                Some(&self.get_description()),
                self.required.get(),
                false,
                false,
                None,
                field_displayer1.as_deref(),
                field_displayer2.as_deref(),
            )?;
        }
        Ok(text)
    }

    /// Java `@Override getDescription()`.
    pub fn get_description(&self) -> String {
        // getQuotedLabel(): never null, since a check box's text is never null.
        self.get_quoted_label().unwrap_or_default()
    }

    /// Java `@Override getText()`.  Get text without validation.
    pub fn get_text_void(&self) -> Option<String> {
        Field::get_text_void(&*self.text_field)
    }

    /// Java `@Override getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(Field::get_text_void(&*self.check_box).as_deref())
    }

    /// Java `getSize()`.
    pub fn get_size(&self) -> Dimension {
        self.text_field.get_size()
    }

    /// Java `setTextPreferredWidth(int)`.
    pub fn set_text_preferred_width(&self, width: i32) {
        self.text_field.set_preferred_width(width);
    }

    /// Java public `setTextFieldVisible(boolean)`.
    pub fn set_text_field_visible(&self, visible: bool) {
        self.text_field.set_visible(visible);
    }

    /// Java private `action()`.
    fn action(&self) {
        self.update_display();
    }
}

impl UIComponent for CheckTextField {
    /// Java `@Override getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        CheckTextField::get_component(self)
    }
}

impl SwingComponent for CheckTextField {
    fn get_component(&self) -> Rc<JComponent> {
        CheckTextField::get_component(self)
    }
}

impl Field for CheckTextField {
    fn is_debug(&self) -> bool {
        CheckTextField::is_debug(self)
    }
    fn get_name(&self) -> Option<String> {
        CheckTextField::get_name(self)
    }
    fn is_boolean(&self) -> bool {
        CheckTextField::is_boolean(self)
    }
    fn is_text(&self) -> bool {
        CheckTextField::is_text(self)
    }
    fn get_quoted_label(&self) -> Option<String> {
        CheckTextField::get_quoted_label(self)
    }
    fn is_enabled(&self) -> bool {
        CheckTextField::is_enabled(self)
    }
    fn clear(&self) {
        CheckTextField::clear(self)
    }
    fn set_value_field(&self, from: Option<&dyn Field>) {
        CheckTextField::set_value_field(self, from)
    }
    fn set_value_string(&self, text: Option<&str>) {
        CheckTextField::set_value_string(self, text)
    }
    fn set_value_boolean(&self, bool_: bool) {
        CheckTextField::set_value_boolean(self, bool_)
    }
    fn is_empty(&self) -> bool {
        CheckTextField::is_empty(self)
    }
    fn is_selected(&self) -> bool {
        CheckTextField::is_selected(self)
    }
    fn is_required(&self) -> bool {
        CheckTextField::is_required(self)
    }
    fn get_text_void(&self) -> Option<String> {
        CheckTextField::get_text_void(self)
    }
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        CheckTextField::get_text_boolean_field_displayer(self, do_validation, field_displayer1)
    }
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        CheckTextField::get_text_boolean_field_displayer_field_displayer(
            self,
            do_validation,
            field_displayer1,
            field_displayer2,
        )
    }
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        CheckTextField::get_directive_def(self)
    }
    fn use_default_value(&self) {
        CheckTextField::use_default_value(self)
    }
    fn equals_default_value_void(&self) -> bool {
        CheckTextField::equals_default_value_void(self)
    }
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        CheckTextField::equals_default_value_string(self, value)
    }
    fn backup(&self) {
        CheckTextField::backup(self)
    }
    fn restore_from_backup(&self) {
        CheckTextField::restore_from_backup(self)
    }
    fn checkpoint(&self) {
        CheckTextField::checkpoint_void(self)
    }
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        CheckTextField::set_checkpoint(self, input)
    }
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        CheckTextField::get_checkpoint(self)
    }
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        CheckTextField::is_different_from_checkpoint_boolean(self, always_check)
    }
    fn is_field_highlight_set(&self) -> bool {
        CheckTextField::is_field_highlight_set(self)
    }
    fn clear_field_highlight(&self) {
        CheckTextField::clear_field_highlight(self)
    }
    fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        CheckTextField::set_field_highlight_field_setting_interface(self, input)
    }
    fn set_field_highlight_string(&self, input: Option<&str>) {
        CheckTextField::set_field_highlight_string(self, input)
    }
    fn set_field_highlight_boolean(&self, input: bool) {
        CheckTextField::set_field_highlight_boolean(self, input)
    }
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        CheckTextField::get_field_highlight(self)
    }
    fn equals_field_highlight_void(&self) -> bool {
        CheckTextField::equals_field_highlight_void(self)
    }
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        CheckTextField::equals_field_highlight_string(self, value)
    }
    fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        CheckTextField::set_tool_tip_text(self, tooltip)
    }
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        CheckTextField::set_tooltip(self, field)
    }
    fn get_tooltip(&self) -> Option<String> {
        CheckTextField::get_tooltip(self)
    }
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        CheckTextField::equals_selected_string_value(self, value)
    }
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        CheckTextField::set_directive_def(self, directive_def)
    }
    fn get_description(&self) -> String {
        CheckTextField::get_description(self)
    }
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        CheckTextField::set_unformatted_tooltip(self, text)
    }
    fn has_unformatted_tooltip(&self) -> bool {
        CheckTextField::has_unformatted_tooltip(self)
    }
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        CheckTextField::use_unformatted_tooltip(self, param_descr, directive_descr)
    }
}

impl BooleanFieldInterface for CheckTextField {}
impl TextFieldInterface for CheckTextField {}
impl BooleanTextFieldInterface for CheckTextField {}
