//! `IMOD/Etomo/src/etomo/ui/swing/LabeledTextField.java`.
//!
//! A `JLabel` and a `JTextField` in a horizontal panel.  The text field names itself
//! from the label (uitest `tf.` names); the field validates its text and keeps backup,
//! default, checkpoint and field-highlight settings.  Sizes, fonts, alignment,
//! backgrounds and mouse listeners are Swing layout/painting/events and are not
//! modelled (`// Swing ...:` comments).  Java registers this object as the text field's
//! `FocusListener` when a field highlight is first used; [`LabeledTextField::focus_lost`]
//! is what that listener runs.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use super::colors;
use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use super::ui_utilities;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{
    ActionListener, Color, Dimension, DocumentListener, FocusEvent, FocusListener, JComponent,
};
use crate::imod::etomo::logic::autodoc_attribute_retriever;
use crate::imod::etomo::logic::field_validator::FieldValidator;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::text_field_interface::TextFieldInterface;
use crate::imod::etomo::ui::text_field_setting::TextFieldSetting;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `LabeledTextField`.
pub struct LabeledTextField {
    /// Java `panel`.
    panel: Rc<JComponent>,
    /// Java `label`.
    label: Rc<JComponent>,
    /// Java `textField`.
    text_field: Rc<JComponent>,
    /// Java `fieldType`.
    field_type: FieldType,
    /// Java `locationDescr`.
    location_descr: Option<String>,
    /// Java `maxArraySize`.  Only for validating fields with a field type of
    /// FieldType.FLOATING_POINT_ARRAY and FieldType.INTEGER_ARRAY.
    max_array_size: i32,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `origTextForeground`.
    orig_text_foreground: Cell<Option<Color>>,
    /// Java `origLabelForeground`.
    orig_label_foreground: Cell<Option<Color>>,
    /// Java `directiveDef`.
    directive_def: Cell<Option<DirectiveDef>>,
    /// Java `maxDecimalPlaces`.
    max_decimal_places: RefCell<Option<EtomoNumber>>,
    // Never reassign TextFieldSetting to null. If null means that they have never been
    // used, less updating when checking the value of TextFieldSetting variables is
    // required.
    /// Java `backup`.
    backup: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `defaultValue`.
    default_value: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `fieldHighlight`.
    field_highlight: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `checkpoint`.
    checkpoint: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `required`.
    required: Cell<bool>,
    /// Java `validationSet`.  Java keeps the caller's object (shared); Rust keeps a copy.
    validation_set: RefCell<Option<ValidationSet>>,
    /// Java `overridableFieldDisplayer1`.
    overridable_field_displayer1: RefCell<Option<Rc<dyn FieldDisplayer>>>,
    /// Java `overridableFieldDisplayer2`.
    overridable_field_displayer2: RefCell<Option<Rc<dyn FieldDisplayer>>>,
    /// Java `unformattedTooltip`.
    unformatted_tooltip: RefCell<Option<String>>,
    /// `this`, for registering this object as the text field's `FocusListener`.
    self_ref: std::rc::Weak<LabeledTextField>,
}

impl std::fmt::Display for LabeledTextField {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[label:{}]", self.get_label())
    }
}

impl LabeledTextField {
    /// Java private `paramString()` (no caller in the Java either).
    #[allow(dead_code)]
    fn param_string(&self) -> String {
        format!(
            "label={},textField={}",
            self.label.get_text(),
            self.text_field.get_text()
        )
    }

    /// Java `equals(Object)`: true when the object is the text field.
    pub fn equals_object(&self, object: &Rc<JComponent>) -> bool {
        Rc::ptr_eq(object, &self.text_field)
    }

    /// Java `equals(Document)`.  The stand-in's document is the text field.
    pub fn equals_document(&self, document: &Rc<JComponent>) -> bool {
        Rc::ptr_eq(document, &self.text_field)
    }

    /// Java private `LabeledTextField(FieldType, int, String, int, String)`.
    fn new_field_type_int_string_int_string(
        field_type: FieldType,
        max_array_size: i32,
        tf_label: Option<&str>,
        hgap: i32,
        location_descr: Option<&str>,
    ) -> Rc<LabeledTextField> {
        let field = Rc::new_cyclic(|self_ref| LabeledTextField {
            self_ref: self_ref.clone(),
            panel: JComponent::new_panel(),
            label: JComponent::new_label(""),
            text_field: JComponent::new_text_field(),
            field_type,
            location_descr: location_descr.map(str::to_owned),
            max_array_size,
            debug: Cell::new(false),
            orig_text_foreground: Cell::new(None),
            orig_label_foreground: Cell::new(None),
            directive_def: Cell::new(None),
            max_decimal_places: RefCell::new(None),
            backup: RefCell::new(None),
            default_value: RefCell::new(None),
            field_highlight: RefCell::new(None),
            checkpoint: RefCell::new(None),
            required: Cell::new(false),
            validation_set: RefCell::new(None),
            overridable_field_displayer1: RefCell::new(None),
            overridable_field_displayer2: RefCell::new(None),
            unformatted_tooltip: RefCell::new(None),
        });
        // set label
        field.set_label(tf_label);
        // Swing layout: panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS)).
        field.panel.add(&field.label);
        if hgap > 0 {
            // Swing layout: panel.add(Box.createRigidArea(new Dimension(hgap, 0))).
        }
        field.panel.add(&field.text_field);
        // Use the label as the action command. The <Enter> key triggers an action.
        field.text_field.set_action_command(tf_label);
        // Swing layout: set the maximum height of the text field box to twice the font
        // size (the larger of the label's and the text field's), since it is not set by
        // default.
        if field_type == FieldType::File {
            // Swing layout: textField.setHorizontalAlignment(JTextField.RIGHT).
        }
        field
    }

    /// Java `LabeledTextField(FieldType, String)`.
    pub fn new_field_type_string(
        field_type: FieldType,
        tf_label: Option<&str>,
    ) -> Rc<LabeledTextField> {
        LabeledTextField::new_field_type_int_string_int_string(field_type, -1, tf_label, 0, None)
    }

    /// Java `LabeledTextField(FieldType, int, String)`.
    pub fn new_field_type_int_string(
        field_type: FieldType,
        max_array_size: i32,
        tf_label: Option<&str>,
    ) -> Rc<LabeledTextField> {
        LabeledTextField::new_field_type_int_string_int_string(
            field_type,
            max_array_size,
            tf_label,
            0,
            None,
        )
    }

    /// Java `LabeledTextField(FieldType, String, String)`.
    pub fn new_field_type_string_string(
        field_type: FieldType,
        tf_label: Option<&str>,
        location_descr: Option<&str>,
    ) -> Rc<LabeledTextField> {
        LabeledTextField::new_field_type_int_string_int_string(
            field_type,
            -1,
            tf_label,
            0,
            location_descr,
        )
    }

    /// Java `LabeledTextField(FieldType, String, int)`.
    pub fn new_field_type_string_int(
        field_type: FieldType,
        tf_label: Option<&str>,
        hgap: i32,
    ) -> Rc<LabeledTextField> {
        LabeledTextField::new_field_type_int_string_int_string(field_type, -1, tf_label, hgap, None)
    }

    /// Java static `getNumericInstance(String, EtomoNumber.Type)`.
    pub fn get_numeric_instance_string_type(
        tf_label: Option<&str>,
        numeric_type: Type,
    ) -> Rc<LabeledTextField> {
        let mut field_type = FieldType::Integer;
        if numeric_type == Type::Double {
            field_type = FieldType::FloatingPoint;
        }
        LabeledTextField::new_field_type_int_string_int_string(field_type, -1, tf_label, 0, None)
    }

    /// Java static `getNumericInstance(String)`.
    pub fn get_numeric_instance_string(tf_label: Option<&str>) -> Rc<LabeledTextField> {
        LabeledTextField::get_numeric_instance_string_type(tf_label, Type::Integer)
    }

    /// Java `setMaxDecimalPlaces(int)`.
    pub fn set_max_decimal_places(&self, digits: i32) {
        let mut max_decimal_places = self.max_decimal_places.borrow_mut();
        if max_decimal_places.is_none() {
            *max_decimal_places = Some(EtomoNumber::new());
        }
        max_decimal_places.as_mut().unwrap().set_int(digits);
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, tf_label: Option<&str>) {
        let field_type = &ui_test_field_type::TEXT_FIELD;
        let name = utilities::convert_label_to_name(tf_label, field_type.is_unlimited_segments());
        self.text_field.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.text_field.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `checkpoint()`: saves the current text as the checkpoint.
    pub fn checkpoint_void(&self) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_string(self.get_text_void().as_deref());
    }

    /// Java `checkpoint(int)`: saves value as the checkpoint.
    pub fn checkpoint_int(&self, value: i32) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_int(value);
    }

    /// Java `checkpoint(ConstEtomoNumber)`: saves value as the checkpoint.
    pub fn checkpoint_const_etomo_number(&self, value: Option<&ConstEtomoNumber>) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_const_etomo_number(value);
    }

    /// Java `checkpoint(double)`: saves value as the checkpoint.
    pub fn checkpoint_double(&self, value: f64) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_double(value);
    }

    /// Java `checkpoint(String)`: saves value as the checkpoint.
    pub fn checkpoint_string(&self, value: Option<&str>) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_string(value);
    }

    /// Java `resetToCheckpoint()`.  Resets to the checkpoint value if it has been set.
    pub fn reset_to_checkpoint(&self) {
        let checkpoint = self.checkpoint.borrow().clone();
        let Some(checkpoint) = checkpoint.filter(|checkpoint| checkpoint.is_set()) else {
            return;
        };
        self.set_text_string(checkpoint.get_value().as_deref());
    }

    /// Java `getFieldType()`.
    pub fn get_field_type(&self) -> FieldType {
        self.field_type
    }

    // Java `getFont()`: fonts are not modelled.

    /// Java `focusGained(FocusEvent)`.
    pub fn focus_gained(&self) {}

    /// Java `focusLost(FocusEvent)`.
    pub fn focus_lost(&self) {
        self.update_field_highlight();
    }

    /// Java `updateFieldHighlight()`.  If the field highlight is in use, use the field
    /// highlight color on the foreground of the label and text field when the value
    /// equals the field highlight value, saving the original foregrounds; otherwise
    /// restore them.
    pub fn update_field_highlight(&self) {
        // To avoid constantly updating the foreground color, assuming that
        // fieldHighlight is never reassigned to null
        let field_highlight = self.field_highlight.borrow().clone();
        let Some(field_highlight) = field_highlight else {
            return;
        };
        if !self.text_field.is_enabled() {
            return;
        }
        if field_highlight.is_set()
            && field_highlight.equals_string(Some(&self.text_field.get_text()))
        {
            // save the original color
            if self.orig_text_foreground.get().is_none() {
                let mut orig = self.text_field.get_foreground();
                if orig.is_none() {
                    // Color.BLACK
                    orig = Some((0, 0, 0));
                }
                self.orig_text_foreground.set(orig);
            }
            if self.orig_label_foreground.get().is_none() {
                let mut orig = self.label.get_foreground();
                if orig.is_none() {
                    // Color.BLACK
                    orig = Some((0, 0, 0));
                }
                self.orig_label_foreground.set(orig);
            }
            self.label.set_foreground(Some(colors::FIELD_HIGHLIGHT));
            self.text_field
                .set_foreground(Some(colors::FIELD_HIGHLIGHT));
        } else {
            if let Some(orig) = self.orig_text_foreground.get() {
                self.text_field.set_foreground(Some(orig));
            }
            if let Some(orig) = self.orig_label_foreground.get() {
                self.label.set_foreground(Some(orig));
            }
        }
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> String {
        self.label.get_text()
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.text_field.add_action_listener(listener);
    }

    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.text_field.add_focus_listener(listener);
    }

    /// Java `removeFocusListener(FocusListener)`.
    pub fn remove_focus_listener(&self, listener: &FocusListener) {
        self.text_field.remove_focus_listener(listener);
    }

    /// `this` as a `FocusListener` (Java `textField.addFocusListener(this)`).
    fn this_focus_listener(&self) -> FocusListener {
        let this = self.self_ref.clone();
        Rc::new(move |event: &FocusEvent| {
            if let Some(this) = this.upgrade() {
                if event.gained {
                    this.focus_gained();
                } else {
                    this.focus_lost();
                }
            }
        })
    }

    /// Java `isDifferentFromCheckpoint()`.  If the field is disabled then return false
    /// because its value doesn't matter.  Returns true if the checkpoint has not been
    /// done, otherwise compares the text with the checkpoint.
    pub fn is_different_from_checkpoint_void(&self) -> bool {
        self.is_different_from_checkpoint_boolean(false)
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.  When alwaysCheck is false, return
    /// false when the field is disabled or invisible.
    pub fn is_different_from_checkpoint_boolean(&self, always_check: bool) -> bool {
        if !always_check && (!self.text_field.is_enabled() || !self.text_field.is_visible()) {
            return false;
        }
        let checkpoint = self.checkpoint.borrow().clone();
        match checkpoint {
            None => true,
            Some(checkpoint) => !checkpoint.equals_string(self.get_text_void().as_deref()),
        }
    }

    /// Java `equals(String)`.
    pub fn equals_string(&self, that_text: Option<&str>) -> bool {
        let text = self.get_text_void();
        let Some(text) = text else {
            return that_text.is_none();
        };
        let Some(that_text) = that_text else {
            return false;
        };
        java_lang_string_trim(&text) == java_lang_string_trim(that_text)
    }

    /// Java `setHighlight(boolean)`.
    pub fn set_highlight(&self, highlight: bool) {
        if highlight {
            // Swing painting: textField.setBackground(Colors.HIGHLIGHT_BACKGROUND).
        } else {
            // Swing painting: textField.setBackground(Colors.BACKGROUND).
        }
    }

    /// Java `getField()`.
    pub fn get_field(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> String {
        self.label.get_text()
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&self, label: Option<&str>) {
        self.label.set_text(label.unwrap_or(""));
        self.set_name(label);
    }

    /// Java `setRequired(boolean)`.
    pub fn set_required(&self, required: bool) {
        self.required.set(required);
    }

    /// Java `setValidationSet(ValidationSet)`.
    pub fn set_validation_set(&self, input: Option<&ValidationSet>) {
        let mut validation_set = self.validation_set.borrow_mut();
        match validation_set.as_mut() {
            None => *validation_set = input.cloned(),
            Some(validation_set) => {
                if input.is_some() {
                    validation_set.copy(input);
                } else {
                    validation_set.clear();
                }
            }
        }
    }

    /// Java `setNumberMustBePositive(boolean)`.
    pub fn set_number_must_be_positive(&self, input: bool) {
        let mut validation_set = self.validation_set.borrow_mut();
        if validation_set.is_none() {
            *validation_set = Some(ValidationSet::new(Some(self.field_type), None));
        }
        validation_set
            .as_mut()
            .unwrap()
            .set_number_must_be_positive(input);
    }

    /// Java `setMinimum(double)`.
    pub fn set_minimum(&self, input: f64) {
        let mut validation_set = self.validation_set.borrow_mut();
        if validation_set.is_none() {
            *validation_set = Some(ValidationSet::new(Some(self.field_type), None));
        }
        validation_set.as_mut().unwrap().set_minimum(input);
    }

    /// Java `setMaximum(double)`.
    pub fn set_maximum(&self, input: f64) {
        let mut validation_set = self.validation_set.borrow_mut();
        if validation_set.is_none() {
            *validation_set = Some(ValidationSet::new(Some(self.field_type), None));
        }
        validation_set.as_mut().unwrap().set_maximum(input);
    }

    /// Java `setParsableString(boolean)`.
    pub fn set_parsable_string(&self, parsable_string: bool) {
        let mut validation_set = self.validation_set.borrow_mut();
        if validation_set.is_none() {
            *validation_set = Some(ValidationSet::new(Some(self.field_type), None));
        }
        validation_set
            .as_mut()
            .unwrap()
            .set_parsable_string(parsable_string);
    }

    /// Java `getText(boolean)`.
    pub fn get_text_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, None, None)
    }

    /// Java `setOverridableFieldDisplayers(FieldDisplayer, FieldDisplayer)`.
    pub fn set_overridable_field_displayers(
        &self,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) {
        *self.overridable_field_displayer1.borrow_mut() = field_displayer1;
        *self.overridable_field_displayer2.borrow_mut() = field_displayer2;
    }

    /// Java `handleValidation(String, FieldDisplayer, FieldDisplayer)`.  Returns false
    /// if invalid.
    pub fn handle_validation(
        &self,
        errmsg: Option<&str>,
        mut field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        mut field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> bool {
        if field_displayer1.is_none() {
            field_displayer1 = self.overridable_field_displayer1.borrow().clone();
        }
        if field_displayer2.is_none() {
            field_displayer2 = self.overridable_field_displayer2.borrow().clone();
        }
        FieldValidator::handle_validation(
            errmsg,
            Some(self),
            Some(&self.get_description()),
            field_displayer1.as_deref(),
            field_displayer2.as_deref(),
        )
    }

    /// Java `setText(File)`.
    pub fn set_text_file(&self, file: &Path) {
        let absolute_path = std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf());
        self.text_field.set_text(&absolute_path.to_string_lossy());
    }

    /// Java `setText(ConstEtomoNumber)`.
    pub fn set_text_const_etomo_number(&self, text: Option<&ConstEtomoNumber>) {
        match text {
            None => self.text_field.set_text(""),
            Some(text) => {
                let max_decimal_places = self.max_decimal_places.borrow();
                self.text_field.set_text(
                    &utilities::round_to_max_decimal_places(
                        Some(text),
                        max_decimal_places.as_deref(),
                    )
                    .unwrap_or_else(|| "null".to_string()),
                );
            }
        }
    }

    /// Java `setText(Number)`.
    pub fn set_text_number(&self, input: Option<Number>) {
        match input {
            None => self.text_field.set_text(""),
            Some(input) => {
                let text = utilities::round_to_max_decimal_places_number(
                    Some(input),
                    self.max_decimal_places.borrow().as_deref(),
                );
                self.text_field
                    .set_text(&text.unwrap_or_else(|| "null".to_string()));
            }
        }
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        let text = utilities::round_to_max_decimal_places_string(
            text,
            self.max_decimal_places.borrow().as_deref(),
        );
        // JTextField.setText(null) empties the field.
        self.text_field.set_text(text.as_deref().unwrap_or(""));
    }

    /// Java `setNonEmptyText(String)`.  Set text if parameter is not empty.  Prevents a
    /// value from being overridden by nothing.
    pub fn set_non_empty_text(&self, text: Option<&str>) {
        if text.is_some_and(|text| !text.is_empty()) {
            self.set_text_string(text);
        }
    }

    /// Java `setText(String, boolean)`.
    pub fn set_text_string_boolean(&self, text: Option<&str>, allow_empty: bool) {
        if allow_empty || text.is_some_and(|text| !text.is_empty()) {
            self.set_text_string(text);
        }
    }

    /// Java `setText(int)`.
    pub fn set_text_int(&self, value: i32) {
        // textField.setText(String.valueOf(value));
        let text = utilities::round_to_max_decimal_places_number(
            Some(Number::Integer(value)),
            self.max_decimal_places.borrow().as_deref(),
        );
        self.text_field
            .set_text(&text.unwrap_or_else(|| "null".to_string()));
    }

    /// Java `setText(long)`.
    pub fn set_text_long(&self, value: i64) {
        // textField.setText(String.valueOf(value));
        let text = utilities::round_to_max_decimal_places_number(
            Some(Number::Long(value)),
            self.max_decimal_places.borrow().as_deref(),
        );
        self.text_field
            .set_text(&text.unwrap_or_else(|| "null".to_string()));
    }

    /// Java `setText(double)`.
    pub fn set_text_double(&self, value: f64) {
        // textField.setText(Double.toString(value));
        let text = utilities::round_to_max_decimal_places_number(
            Some(Number::Double(value)),
            self.max_decimal_places.borrow().as_deref(),
        );
        self.text_field
            .set_text(&text.unwrap_or_else(|| "null".to_string()));
    }

    /// Java `setTextFieldEnabled(boolean)`.
    pub fn set_text_field_enabled(&self, enabled: bool) {
        self.text_field.set_enabled(enabled);
        if enabled {
            self.update_field_highlight();
        }
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.text_field.set_enabled(enabled);
        self.label.set_enabled(enabled);
        if enabled {
            self.update_field_highlight();
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        // Label is not changed by editable status
        self.text_field.set_editable(editable);
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.text_field.is_editable()
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.panel.is_visible()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, is_visible: bool) {
        self.panel.set_visible(is_visible);
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `addDocumentListener(DocumentListener)`.
    pub fn add_document_listener(&self, listener: DocumentListener) {
        self.text_field.add_document_listener(listener);
    }

    /// Java `setTextPreferredSize(Dimension)`.
    pub fn set_text_preferred_size(&self, _size: Dimension) {
        // Swing layout: textField.setPreferredSize(size); textField.setMaximumSize(size).
    }

    /// Java `setPreferredWidth(int)`.
    pub fn set_preferred_width(&self, width: i32) {
        // Swing layout: textField.getPreferredSize() is not modelled.
        let _dim = ui_utilities::calc_new_text_field_size(None, width, true);
        // Swing layout: textField.setPreferredSize(dim); textField.setMaximumSize(dim).
    }

    /// Java `setTextPreferredWidth(double)`.
    pub fn set_text_preferred_width(&self, _min_width: f64) {
        // Swing layout: prefSize = textField.getPreferredSize(); prefSize.setSize(minWidth,
        // prefSize.getHeight()); textField.setPreferredSize(prefSize).
    }

    /// Java `setMinimumWidth(double)`.
    pub fn set_minimum_width(&self, _min_width: f64) {
        // Swing layout: prefSize = textField.getPreferredSize(); prefSize.setSize(minWidth,
        // prefSize.getHeight()); textField.setMinimumSize(prefSize).
    }

    // Java: `setPreferredSize(Dimension)` and `setMaximumSize(Dimension)` are commented
    // out in the source.

    /// Java `getLabelPreferredSize()`.  Sizes are not modelled; zero is returned.
    pub fn get_label_preferred_size(&self) -> Dimension {
        // Swing layout: label.getPreferredSize().
        Dimension {
            width: 0,
            height: 0,
        }
    }

    /// Java `setColumns(int)`.
    pub fn set_columns(&self, _columns: i32) {
        // Swing layout: textField.setColumns(columns).
    }

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, _alignment: f32) {
        // Swing layout: panel.setAlignmentX(alignment).
    }

    /// Java `addMouseListener(MouseListener)`.
    pub fn add_mouse_listener(&self) {
        // Swing mouse: panel, label and textField.addMouseListener(listener) - mouse
        // events are not modelled.
    }
}

impl Field for LabeledTextField {
    /// Java `isDebug()`.
    fn is_debug(&self) -> bool {
        self.debug.get() || ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.text_field.get_name()
    }

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        false
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        true
    }

    /// Java `equalsSelectedStringValue(String)`.  No true value is implemented so all
    /// non-empty values are true.
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }

    /// Java `checkpoint()`.
    fn checkpoint(&self) {
        self.checkpoint_void();
    }

    /// Java `backup()`.  Saves the current text in backup.
    fn backup(&self) {
        if self.backup.borrow().is_none() {
            *self.backup.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let backup = self.backup.borrow().clone().unwrap();
        backup.set_string(self.get_text_void().as_deref());
    }

    /// Java `restoreFromBackup()`.  If the field was backed up, make the backup value
    /// the displayed value, and turn off the back up.
    fn restore_from_backup(&self) {
        let backup = self.backup.borrow().clone();
        if let Some(backup) = backup.filter(|backup| backup.is_set()) {
            self.set_text_string(backup.get_value().as_deref());
            backup.reset();
        }
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java `getDirectiveDef()`.
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `useDefaultValue()`.
    fn use_default_value(&self) {
        let Some(directive_def) = self.directive_def.get() else {
            let default_value = self.default_value.borrow().clone();
            if let Some(default_value) =
                default_value.filter(|default_value| default_value.is_set())
            {
                default_value.reset();
            }
            return;
        };
        // only search for default value once
        if self.default_value.borrow().is_none() {
            let default_value = Rc::new(TextFieldSetting::new_field_type(self.field_type));
            // TODO(unit): needs etomo/logic/AutodocAttributeRetriever.java -
            // INSTANCE.getDefaultValue(DirectiveDef).
            let value = autodoc_attribute_retriever::INSTANCE
                .get_default_value(Some(directive_def.clone()));
            if let Some(value) = value {
                // if default value has been found, set it in the field setting
                default_value.set_string(Some(&value));
            }
            *self.default_value.borrow_mut() = Some(default_value);
        }
        let default_value = self.default_value.borrow().clone().unwrap();
        if default_value.is_set() {
            self.set_text_string(default_value.get_value().as_deref());
        }
    }

    /// Java `equalsDefaultValue()`.
    fn equals_default_value_void(&self) -> bool {
        let default_value = self.default_value.borrow().clone();
        default_value.is_some_and(|default_value| {
            default_value.is_set() && default_value.equals_string(self.get_text_void().as_deref())
        })
    }

    /// Java `equalsDefaultValue(String)`.
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        let default_value = self.default_value.borrow().clone();
        default_value.is_some_and(|default_value| {
            default_value.is_set() && default_value.equals_string(value)
        })
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        if self.checkpoint.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_text())
        {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone();
        if let Some(checkpoint) = checkpoint {
            checkpoint.copy(input);
        }
    }

    /// Java `getCheckpoint()`.
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        self.checkpoint
            .borrow()
            .clone()
            .map(|checkpoint| checkpoint as Rc<dyn FieldSettingInterface>)
    }

    /// Java `isFieldHighlightSet()`.
    fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.is_set())
    }

    /// Java `setFieldHighlight(String)`.
    fn set_field_highlight_string(&self, value: Option<&str>) {
        if self.field_highlight.borrow().is_none() {
            *self.field_highlight.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
            self.text_field
                .add_focus_listener(self.this_focus_listener());
        }
        let field_highlight = self.field_highlight.borrow().clone().unwrap();
        field_highlight.set_string(value);
        self.update_field_highlight();
    }

    /// Java `setFieldHighlight(boolean)`.
    fn set_field_highlight_boolean(&self, _value: bool) {}

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        if self.field_highlight.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_text())
        {
            *self.field_highlight.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
            self.text_field
                .add_focus_listener(self.this_focus_listener());
        }
        let field_highlight = self.field_highlight.borrow().clone();
        if let Some(field_highlight) = field_highlight {
            field_highlight.copy(input);
            self.update_field_highlight();
        }
    }

    /// Java `clearFieldHighlight()`.
    fn clear_field_highlight(&self) {
        let field_highlight = self.field_highlight.borrow().clone();
        if let Some(field_highlight) =
            field_highlight.filter(|field_highlight| field_highlight.is_set())
        {
            field_highlight.reset();
            self.update_field_highlight();
        }
    }

    /// Java `getFieldHighlight()`.
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        self.field_highlight
            .borrow()
            .clone()
            .map(|field_highlight| field_highlight as Rc<dyn FieldSettingInterface>)
    }

    /// Java `equalsFieldHighlight()`.
    fn equals_field_highlight_void(&self) -> bool {
        let field_highlight = self.field_highlight.borrow().clone();
        field_highlight.is_some_and(|field_highlight| {
            field_highlight.is_set()
                && field_highlight.equals_string(self.get_text_void().as_deref())
        })
    }

    /// Java `equalsFieldHighlight(String)`.
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        let field_highlight = self.field_highlight.borrow().clone();
        field_highlight.is_some_and(|field_highlight| {
            field_highlight.is_set() && field_highlight.equals_string(value)
        })
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        self.is_different_from_checkpoint_boolean(always_check)
    }

    /// Java `clear()`.
    fn clear(&self) {
        self.text_field.set_text("");
    }

    /// Java `setValue(Field)`.
    fn set_value_field(&self, input: Option<&dyn Field>) {
        match input {
            None => self.clear(),
            Some(input) => self.set_text_string(input.get_text_void().as_deref()),
        }
    }

    /// Java `setValue(String)`.
    fn set_value_string(&self, value: Option<&str>) {
        self.set_text_string(value);
    }

    /// Java `setValue(boolean)`.
    fn set_value_boolean(&self, _value: bool) {}

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool {
        false
    }

    /// Java `getQuotedLabel()`.
    fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(Some(&self.label.get_text()))
    }

    /// Java `isRequired()`.
    fn is_required(&self) -> bool {
        self.required.get() && self.text_field.is_enabled()
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
        mut field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        mut field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut text = Some(self.text_field.get_text());
        if self.debug.get() {
            println!(
                "doValidation:{},text:{},required:{}",
                do_validation,
                text.as_deref().unwrap_or("null"),
                self.required.get()
            );
        }
        if do_validation && self.text_field.is_enabled() {
            if field_displayer1.is_none() {
                field_displayer1 = self.overridable_field_displayer1.borrow().clone();
            }
            if field_displayer2.is_none() {
                field_displayer2 = self.overridable_field_displayer2.borrow().clone();
            }
            let validation_set = self.validation_set.borrow().clone();
            text = FieldValidator::validate_text_string_field_type_int_ui_component_string_boolean_boolean_boolean_validation_set_field_displayer_field_displayer(
                text.as_deref(),
                Some(self.field_type),
                self.max_array_size,
                Some(self),
                Some(&self.get_description()),
                self.required.get(),
                false,
                false,
                validation_set.as_ref(),
                field_displayer1.as_deref(),
                field_displayer2.as_deref(),
            )?;
        }
        Ok(text)
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> String {
        self.get_quoted_label()
            .unwrap_or_else(|| "null".to_string())
            + &match &self.location_descr {
                None => String::new(),
                Some(location_descr) => " in ".to_string() + location_descr,
            }
    }

    /// Java `getText()`: return text without validation.
    fn get_text_void(&self) -> Option<String> {
        self.get_text_boolean_field_displayer_field_displayer(false, None, None)
            .unwrap_or(None)
    }

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        java_lang_string_matches_whitespace(&self.text_field.get_text())
    }

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool {
        self.text_field.is_enabled()
    }

    /// Java `setToolTipText(String)`.
    fn set_tool_tip_text(&self, text: Option<&str>) {
        let set_debug = self.debug.get() && !tooltip_formatter::INSTANCE.is_debug();
        if set_debug {
            tooltip_formatter::INSTANCE.set_debug(self.debug.get());
        }
        let tooltip = tooltip_formatter::INSTANCE.format(text);
        if set_debug {
            tooltip_formatter::INSTANCE.set_debug(false);
        }
        self.panel.set_tool_tip_text(tooltip.as_deref());
        self.text_field.set_tool_tip_text(tooltip.as_deref());
        self.label.set_tool_tip_text(tooltip.as_deref());
    }

    /// Java `setUnformattedTooltip(String)`.
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        *self.unformatted_tooltip.borrow_mut() = text.map(str::to_owned);
        self.unformatted_tooltip.borrow().clone()
    }

    /// Java `hasUnformattedTooltip()`.
    fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.borrow().is_some()
    }

    /// Java synchronized `useUnformattedTooltip(String, String)`.
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        let unformatted_tooltip = self.unformatted_tooltip.borrow().clone();
        self.set_tool_tip_text(
            tooltip_formatter::INSTANCE
                .build_tooltip(unformatted_tooltip.as_deref(), param_descr, directive_descr)
                .as_deref(),
        );
        *self.unformatted_tooltip.borrow_mut() = None;
    }

    /// Java `setTooltip(Field)`.
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            let tooltip = field.get_tooltip();
            self.panel.set_tool_tip_text(tooltip.as_deref());
            self.text_field.set_tool_tip_text(tooltip.as_deref());
            self.label.set_tool_tip_text(tooltip.as_deref());
        }
    }

    /// Java `getTooltip()`.
    fn get_tooltip(&self) -> Option<String> {
        self.text_field.get_tool_tip_text()
    }
}

impl TextFieldInterface for LabeledTextField {}

impl UIComponent for LabeledTextField {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.panel.clone()
    }
}

impl SwingComponent for LabeledTextField {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.panel.clone()
    }
}
