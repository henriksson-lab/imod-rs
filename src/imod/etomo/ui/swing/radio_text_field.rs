//! `IMOD/Etomo/src/etomo/ui/swing/RadioTextField.java`.
//!
//! A radio button followed by a text field that is enabled only while the radio button
//! is selected.  The radio button's model reports selection changes back to this object
//! (Java `new RadioButton.RadioButtonModel(this)`), so it holds a `Weak` to it.

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_button_interface::RadioButtonInterface;
use super::text_field::TextField;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionListener, ButtonGroup, Dimension, JComponent};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_double_to_string, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::parsed_element::ParsedElement;
use crate::imod::etomo::ui::boolean_field_interface::BooleanFieldInterface;
use crate::imod::etomo::ui::boolean_text_field_interface::BooleanTextFieldInterface;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_bundle::FieldSettingBundle;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::text_field_interface::TextFieldInterface;
use crate::imod::etomo::util::utilities;

/// Java `RadioTextField`.
pub struct RadioTextField {
    /// Java `rootPanel`.
    root_panel: Rc<JComponent>,
    /// Java `radioButton`.
    radio_button: Rc<RadioButton>,
    /// Java `textField`.
    text_field: Rc<TextField>,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `directiveDef`.
    directive_def: Cell<Option<DirectiveDef>>,
    /// Java `enabled`.
    enabled: Cell<bool>,
    /// Java `editable`.
    editable: Cell<bool>,
    /// Java `this`, for `getField()`.
    this: Weak<RadioTextField>,
}

impl RadioTextField {
    /// Java static `getInstance(FieldType, String, ButtonGroup)`.  Constructs local
    /// instance, adds listener, and returns.
    pub fn get_instance_field_type_string_button_group(
        field_type: FieldType,
        label: Option<&str>,
        group: Option<&Rc<ButtonGroup>>,
    ) -> Rc<RadioTextField> {
        RadioTextField::new(field_type, label, None, group, None, None)
    }

    /// Java static `getInstanceWithAlternateLabel(FieldType, String, ButtonGroup,
    /// String)`.
    pub fn get_instance_with_alternate_label(
        field_type: FieldType,
        label: Option<&str>,
        group: Option<&Rc<ButtonGroup>>,
        alternate_label: Option<&str>,
    ) -> Rc<RadioTextField> {
        RadioTextField::new(field_type, label, None, group, None, alternate_label)
    }

    /// Java static `getInstance(FieldType, EnumeratedType, ButtonGroup)`.
    pub fn get_instance_field_type_enumerated_type_button_group(
        field_type: FieldType,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<&Rc<ButtonGroup>>,
    ) -> Rc<RadioTextField> {
        RadioTextField::new(field_type, None, enumerated_type, group, None, None)
    }

    /// Java static `getInstance(FieldType, String, ButtonGroup, String)`.
    pub fn get_instance_field_type_string_button_group_string(
        field_type: FieldType,
        label: Option<&str>,
        group: Option<&Rc<ButtonGroup>>,
        location_descr: Option<&str>,
    ) -> Rc<RadioTextField> {
        RadioTextField::new(field_type, label, None, group, location_descr, None)
    }

    /// Java static `getInstance(FieldType, EnumeratedType, ButtonGroup, String)`.
    ///
    /// Fixed in translation (`RadioTextField.java:96-97`): Java passes `null` instead of
    /// its `enumeratedType` parameter to the constructor, so the radio button got no
    /// label, no enumerated type and no default selection.  The parameter is passed.
    pub fn get_instance_field_type_enumerated_type_button_group_string(
        field_type: FieldType,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<&Rc<ButtonGroup>>,
        location_descr: Option<&str>,
    ) -> Rc<RadioTextField> {
        RadioTextField::new(
            field_type,
            None,
            enumerated_type,
            group,
            location_descr,
            None,
        )
    }

    /// Java private `RadioTextField(FieldType, String, EnumeratedType, ButtonGroup,
    /// String, String)`.
    ///
    /// Fixed in translation: when the enumerated type is the default one, Java's
    /// `RadioButton` constructor selects the button, the model calls back
    /// `msgSelected`, and `updateDisplay` dereferences the still-null `textField`
    /// (NullPointerException).  Here the callback cannot reach the object under
    /// construction; `init` runs `updateDisplay` once both fields exist.
    fn new(
        field_type: FieldType,
        label: Option<&str>,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<&Rc<ButtonGroup>>,
        location_descr: Option<&str>,
        alternate_label: Option<&str>,
    ) -> Rc<RadioTextField> {
        let instance = Rc::new_cyclic(|this: &Weak<RadioTextField>| {
            let radio_button =
                RadioButton::new_string_enumerated_type_button_group_radio_button_model(
                    label,
                    enumerated_type,
                    group,
                    Some(RadioButtonModel::new(Some(
                        this.clone() as Weak<dyn RadioButtonInterface>
                    ))),
                );
            let text_field = TextField::new(field_type, label, location_descr);
            RadioTextField {
                root_panel: JComponent::new_panel(),
                radio_button,
                text_field,
                debug: Cell::new(false),
                directive_def: Cell::new(None),
                enabled: Cell::new(true),
                editable: Cell::new(true),
                this: this.clone(),
            }
        });
        if alternate_label.is_some() {
            instance.radio_button.set_name(alternate_label);
            instance.text_field.set_name(alternate_label);
        }
        instance.init();
        instance
    }

    /// Java private `init()`.
    fn init(&self) {
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel, BoxLayout.X_AXIS)).
        self.root_panel.add(&self.radio_button.get_component());
        self.root_panel.add(&self.text_field.get_component());
        self.update_display();
    }

    /// Java `setTextPreferredWidth(double)`.
    pub fn set_text_preferred_width(&self, min_width: f64) {
        let mut pref_size: Dimension = self.text_field.get_preferred_size();
        // Java Dimension.setSize(double, double) rounds up.
        pref_size.width = min_width.ceil() as i32;
        self.text_field.set_text_preferred_size(pref_size);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java `setText(int)`.
    pub fn set_text_int(&self, value: i32) {
        self.text_field.set_text_string(Some(&value.to_string()));
    }

    /// Java `setText(long)`.
    pub fn set_text_long(&self, value: i64) {
        self.text_field.set_text_string(Some(&value.to_string()));
    }

    /// Java `setText(double)`.
    pub fn set_text_double(&self, value: f64) {
        self.text_field
            .set_text_string(Some(&java_lang_double_to_string(value)));
    }

    /// Java `setText(ParsedElement)`.
    pub fn set_text_parsed_element(&self, value: Option<&dyn ParsedElement>) {
        match value {
            None => self.text_field.set_text_string(Some("")),
            Some(value) => self
                .text_field
                .set_text_string(value.get_raw_string_void().as_deref()),
        }
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        self.text_field.set_text_string(text);
    }

    /// Java `setText(String, boolean)`.
    pub fn set_text_string_boolean(&self, text: Option<&str>, allow_empty: bool) {
        if allow_empty || text.is_some_and(|text| !text.is_empty()) {
            self.set_text_string(text);
        }
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
        self.radio_button.set_debug(debug);
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&self, label: Option<&str>) {
        self.radio_button.set_text(label);
        self.text_field.set_reference(label);
    }

    /// Java `setText(ConstEtomoNumber)`.
    pub fn set_text_const_etomo_number(&self, text: &ConstEtomoNumber) {
        self.text_field.set_text_string(Some(&text.to_string()));
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        self.radio_button.get_text_void()
    }

    /// Java `setRequired(boolean)`.
    pub fn set_required(&self, required: bool) {
        self.text_field.set_required(required);
    }

    /// Java `getText(boolean)`.
    pub fn get_text_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer(do_validation, None)
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
        // Only visually enabled if both enabled and editable
        self.radio_button
            .set_enabled(enabled && self.editable.get());
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.text_field
            .set_enabled(self.enabled.get() && self.radio_button.is_selected());
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.editable.set(editable);
        // Editable has no visible effect if the button is disabled.
        if self.enabled.get() {
            self.radio_button.set_enabled(editable);
            self.text_field.set_editable(editable);
        }
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.radio_button.set_visible(visible);
        self.text_field.set_visible(visible);
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected_boolean(&self, selected: bool) {
        self.radio_button.set_selected_boolean(selected);
        self.update_display();
    }

    /// Java `setSelected(ConstEtomoNumber, boolean)`.  Only sets when selected is not
    /// null or empty.
    ///
    /// Fixed in translation (`RadioTextField.java:442-445`): with `allowEmpty` true and
    /// a null `selected`, Java calls `selected.is()` and throws a NullPointerException;
    /// nothing is set instead.
    pub fn set_selected_const_etomo_number_boolean(
        &self,
        selected: Option<&ConstEtomoNumber>,
        allow_empty: bool,
    ) {
        if allow_empty || selected.is_some_and(|selected| !selected.is_null()) {
            if let Some(selected) = selected {
                self.set_selected_boolean(selected.is());
            }
        }
    }

    /// Java `setTextFieldUnformattedTooltip(String)`.
    pub fn set_text_field_unformatted_tooltip(&self, text: Option<&str>) {
        self.text_field.set_unformatted_tooltip(text);
    }

    /// Java `setRadioButtonToolTipText(String)`.
    pub fn set_radio_button_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.radio_button, text);
    }

    /// Java `setTextFieldToolTipText(String)`.
    pub fn set_text_field_tool_tip_text(&self, text: Option<&str>) {
        self.text_field.set_tool_tip_text(text);
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.radio_button.add_action_listener(action_listener);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.radio_button.get_action_command()
    }

    /// Java `validate()`.  Returns null if instance is in a valid state.
    pub fn validate(&self) -> Option<String> {
        let radio_button_name = Field::get_name(&*self.radio_button).unwrap_or_default();
        let text_field_name = Field::get_name(&*self.text_field).unwrap_or_default();
        // Java textField.getName().substring(2): the name always starts "tf".
        let suffix: String = text_field_name.chars().skip(2).collect();
        if !radio_button_name.ends_with(&suffix) {
            return Some("Fields should have the same name, except for the prefix".to_string());
        }
        if !self.enabled.get() && Field::is_enabled(&*self.text_field) {
            return Some("Fields should enable and disable together".to_string());
        }
        if !self.radio_button.is_selected() && Field::is_enabled(&*self.text_field) {
            return Some(
                "Text field should be disabled when radio button is not selected".to_string(),
            );
        }
        if self.enabled.get()
            && self.radio_button.is_selected()
            && !Field::is_enabled(&*self.text_field)
        {
            return Some("text field should be enabled when radio button is selected".to_string());
        }
        None
    }
}

impl RadioButtonInterface for RadioTextField {
    /// Java `msgSelected()`.
    fn msg_selected(&self) {
        self.update_display();
    }

    /// Java `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef> {
        self.radio_button.get_enumerated_type()
    }

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `getField()`.
    fn get_field(&self) -> Option<Rc<dyn Field>> {
        self.this.upgrade().map(|this| this as Rc<dyn Field>)
    }
}

impl Field for RadioTextField {
    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        Field::get_name(&*self.radio_button)
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        true
    }

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        true
    }

    /// Java `isDebug()`.
    fn is_debug(&self) -> bool {
        self.debug.get() || ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `equalsSelectedStringValue(String)`.
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        self.radio_button.equals_selected_string_value(value)
    }

    /// Java `backup()`.
    fn backup(&self) {
        self.radio_button.backup();
        self.text_field.backup();
    }

    /// Java `restoreFromBackup()`.  If a field was backed up, make the backup value the
    /// displayed value, and turn off the back up.
    fn restore_from_backup(&self) {
        self.radio_button.restore_from_backup();
        self.text_field.restore_from_backup();
        self.update_display();
    }

    /// Java `clear()`.
    fn clear(&self) {
        self.radio_button.clear();
        Field::clear(&*self.text_field);
        self.update_display();
    }

    /// Java `setValue(Field)`.
    fn set_value_field(&self, input: Option<&dyn Field>) {
        self.radio_button.set_value_field(input);
        self.text_field.set_value_field(input);
        self.update_display();
    }

    /// Java `setValue(String)`.
    fn set_value_string(&self, value: Option<&str>) {
        self.text_field.set_value_string(value);
    }

    /// Java `setValue(boolean)`.
    fn set_value_boolean(&self, value: bool) {
        self.radio_button.set_value_boolean(value);
        self.update_display();
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java `useDefaultValue()`.
    fn use_default_value(&self) {
        self.radio_button.use_default_value();
        self.text_field.use_default_value();
        self.update_display();
    }

    /// Java `equalsDefaultValue()`.
    fn equals_default_value_void(&self) -> bool {
        self.radio_button.equals_default_value_void() && self.text_field.equals_default_value_void()
    }

    /// Java `equalsDefaultValue(String)`.
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        self.radio_button.equals_default_value_string(value)
            && self.text_field.equals_default_value_string(value)
    }

    /// Java `getDirectiveDef()`.
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `isFieldHighlightSet()`.
    fn is_field_highlight_set(&self) -> bool {
        self.radio_button.is_field_highlight_set() || self.text_field.is_field_highlight_set()
    }

    /// Java `setFieldHighlight(String)`.
    fn set_field_highlight_string(&self, text: Option<&str>) {
        self.text_field.set_field_highlight_string(text);
    }

    /// Java `setFieldHighlight(boolean)`.
    fn set_field_highlight_boolean(&self, bool_: bool) {
        self.radio_button.set_field_highlight_boolean(bool_);
    }

    /// Java `getFieldHighlight()`.
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        let bundle = FieldSettingBundle::new();
        bundle.add_boolean_setting(self.radio_button.get_field_highlight().as_deref());
        bundle.add_text_setting(self.text_field.get_field_highlight().as_deref());
        Some(Rc::new(bundle) as Rc<dyn FieldSettingInterface>)
    }

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        self.radio_button
            .set_field_highlight_field_setting_interface(input);
        self.text_field
            .set_field_highlight_field_setting_interface(input);
    }

    /// Java `clearFieldHighlight()`.
    fn clear_field_highlight(&self) {
        self.text_field.clear_field_highlight();
        self.radio_button.clear_field_highlight();
    }

    /// Java `equalsFieldHighlight()`.
    fn equals_field_highlight_void(&self) -> bool {
        self.text_field.equals_field_highlight_void()
            && self.radio_button.equals_field_highlight_void()
    }

    /// Java `equalsFieldHighlight(String)`.
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.text_field.equals_field_highlight_string(value)
            && self.radio_button.equals_field_highlight_string(value)
    }

    /// Java `checkpoint()`.
    fn checkpoint(&self) {
        Field::checkpoint(&*self.radio_button);
        Field::checkpoint(&*self.text_field);
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        self.radio_button.set_checkpoint(input);
        self.text_field.set_checkpoint(input);
    }

    /// Java `getCheckpoint()`.
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        let bundle = FieldSettingBundle::new();
        bundle.add_boolean_setting(self.radio_button.get_checkpoint().as_deref());
        bundle.add_text_setting(self.text_field.get_checkpoint().as_deref());
        Some(Rc::new(bundle) as Rc<dyn FieldSettingInterface>)
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        self.radio_button.is_different_from_checkpoint(always_check)
            || self.text_field.is_different_from_checkpoint(always_check)
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> String {
        self.get_quoted_label()
            .unwrap_or_else(|| "null".to_string())
    }

    /// Java `getQuotedLabel()`.
    fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(self.get_label().as_deref())
    }

    /// Java `isRequired()`.
    fn is_required(&self) -> bool {
        self.text_field.is_required()
    }

    /// Java `getText(boolean, FieldDisplayer)`.
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, field_displayer1, None)
    }

    /// Java `getText(boolean, FieldDisplayer, FieldDisplayer)`.
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let text = self
            .text_field
            .get_text_boolean_field_displayer_field_displayer(
                do_validation,
                field_displayer1,
                field_displayer2,
            )?;
        if text
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
        {
            return Ok(Some(String::new()));
        }
        Ok(text)
    }

    /// Java `getText()`: return text without validation.
    fn get_text_void(&self) -> Option<String> {
        let text = self.text_field.get_text_void();
        if text
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
        {
            return Some(String::new());
        }
        text
    }

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        let text = self.text_field.get_text_void();
        text.as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `setToolTipText(String)`.
    fn set_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.radio_button, text);
        Field::set_tool_tip_text(&*self.text_field, text);
    }

    /// Java `setUnformattedTooltip(String)`.
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        self.radio_button.set_unformatted_tooltip(text);
        self.text_field.set_unformatted_tooltip(text)
    }

    /// Java `hasUnformattedTooltip()`.
    fn has_unformatted_tooltip(&self) -> bool {
        self.radio_button.has_unformatted_tooltip() || self.text_field.has_unformatted_tooltip()
    }

    /// Java synchronized `useUnformattedTooltip(String, String)`.
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        self.radio_button
            .use_unformatted_tooltip(param_descr, directive_descr);
        self.text_field
            .use_unformatted_tooltip(param_descr, directive_descr);
    }

    /// Java `setTooltip(Field)`.
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            let tooltip = field.get_tooltip();
            self.radio_button
                .set_preformatted_tooltip(tooltip.as_deref());
            self.text_field.set_preformatted_tooltip(tooltip.as_deref());
        }
    }

    /// Java `getTooltip()`.
    fn get_tooltip(&self) -> Option<String> {
        self.text_field.get_tooltip()
    }
}

impl BooleanFieldInterface for RadioTextField {}
impl TextFieldInterface for RadioTextField {}
impl BooleanTextFieldInterface for RadioTextField {}
