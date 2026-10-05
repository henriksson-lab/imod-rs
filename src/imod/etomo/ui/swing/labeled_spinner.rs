//! `IMOD/Etomo/src/etomo/ui/swing/LabeledSpinner.java`: a label and an integer
//! spinner in one panel, usable as a text `Field`.
//!
//! `final class LabeledSpinner implements TextFieldInterface, ChangeListener,
//! FocusListener`.  The `JPanel`, `JLabel` and `JSpinner` are jdk stand-in
//! [`JComponent`]s; the spinner's `SpinnerNumberModel` lives inside the spinner
//! component.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{
    ChangeEvent, ChangeListener, FocusEvent, FocusListener, JComponent, SpinnerNumberModel,
};
use crate::imod::etomo::logic::autodoc_attribute_retriever;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, Number, Type, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::swing::colors;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::ui::text_field_interface::TextFieldInterface;
use crate::imod::etomo::ui::text_field_setting::TextFieldSetting;
use crate::imod::etomo::util::utilities;

/// `java.awt.Color` as the jdk stand-in carries it (RGB).
type Rgb = (u8, u8, u8);

/// Java `LabeledSpinner`.
pub struct LabeledSpinner {
    /// Java `this`, for `spinner.addChangeListener(this)`.
    this: Weak<LabeledSpinner>,
    panel: Rc<JComponent>,
    label: Rc<JComponent>,
    spinner: Rc<JComponent>,

    default_value: i32,
    // Java field `model`: the spinner's `SpinnerNumberModel`, held by the jdk
    // spinner component (`get_spinner_model` / `set_spinner_model`).
    minimum: Cell<i32>,
    maximum: Cell<i32>,

    /// Java `origLabelForeground` / `origTextForeground`: outer `None` is Java
    /// null; the inner value is what `getForeground()` returned (the jdk's
    /// `None` is the look-and-feel default, a real colour in Swing, so Java's
    /// `Color.BLACK` fallback never applies).
    orig_label_foreground: Cell<Option<Option<Rgb>>>,
    orig_text_foreground: Cell<Option<Option<Rgb>>>,
    directive_def: Cell<Option<DirectiveDef>>,
    backup: RefCell<Option<TextFieldSetting>>,
    default_value_setting: RefCell<Option<TextFieldSetting>>,
    checkpoint: RefCell<Option<TextFieldSetting>>,
    field_highlight: RefCell<Option<TextFieldSetting>>,
    enabled: Cell<bool>,
    editable: Cell<bool>,
    unformatted_tooltip: RefCell<Option<String>>,
}

impl LabeledSpinner {
    /// Java `LabeledSpinner(String spinLabel, int value, int minimum, int maximum,
    /// int stepSize, int defaultValue, int hgap)`.
    fn new(
        spin_label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
        default_value: i32,
        hgap: i32,
    ) -> Rc<LabeledSpinner> {
        let instance = Rc::new_cyclic(|this| LabeledSpinner {
            this: this.clone(),
            panel: JComponent::new_panel(),
            label: JComponent::new_label(""),
            // `new JSpinner()`: Swing's default model is
            // `new SpinnerNumberModel()` (Integer 0, no bounds, step 1).
            spinner: JComponent::new_spinner(SpinnerNumberModel {
                value: 0.0,
                minimum: None,
                maximum: None,
                step_size: 1.0,
                integer: true,
            }),
            default_value,
            minimum: Cell::new(minimum),
            maximum: Cell::new(maximum),
            orig_label_foreground: Cell::new(None),
            orig_text_foreground: Cell::new(None),
            directive_def: Cell::new(None),
            backup: RefCell::new(None),
            default_value_setting: RefCell::new(None),
            checkpoint: RefCell::new(None),
            field_highlight: RefCell::new(None),
            enabled: Cell::new(true),
            editable: Cell::new(true),
            unformatted_tooltip: RefCell::new(None),
        });
        let model = SpinnerNumberModel::new_int(value, minimum, maximum, step_size);
        // set name
        let field_type = &ui_test_field_type::SPINNER;
        let name = utilities::convert_label_to_name(spin_label, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        instance.spinner.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                instance.spinner.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
        // set label
        instance.label.set_text(spin_label.unwrap_or(""));
        // Swing layout: panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS)).
        instance.panel.add(&instance.label);
        if hgap > 0 {
            // Swing layout: panel.add(Box.createRigidArea(new Dimension(hgap, 0))).
        }
        instance.panel.add(&instance.spinner);
        instance.spinner.set_spinner_model(model);
        // Swing layout: set the maximum height of the text field box to twice the
        // font size (of the label if larger, else of the spinner) since it is not
        // set by default - spinner.setMaximumSize(maxSize).
        instance
    }

    /// Java `getDefaultedInstance(String, int value, int minimum, int maximum,
    /// int stepSize, int defaultValue)`.
    pub fn get_defaulted_instance(
        spin_label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
        default_value: i32,
    ) -> Rc<LabeledSpinner> {
        LabeledSpinner::new(
            spin_label,
            value,
            minimum,
            maximum,
            step_size,
            default_value,
            0,
        )
    }

    /// Java `getInstance(String, int value, int minimum, int maximum, int stepSize)`.
    pub fn get_instance_string_int_int_int_int(
        spin_label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
    ) -> Rc<LabeledSpinner> {
        LabeledSpinner::new(spin_label, value, minimum, maximum, step_size, value, 0)
    }

    /// Java `getInstance(String, int value, int minimum, int maximum, int stepSize,
    /// int hgap)`.
    pub fn get_instance_string_int_int_int_int_int(
        spin_label: Option<&str>,
        value: i32,
        minimum: i32,
        maximum: i32,
        step_size: i32,
        hgap: i32,
    ) -> Rc<LabeledSpinner> {
        LabeledSpinner::new(spin_label, value, minimum, maximum, step_size, value, hgap)
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.spinner.get_name()
    }

    /// Java `isText()`.
    pub fn is_text(&self) -> bool {
        true
    }

    /// Java `isBoolean()`.
    pub fn is_boolean(&self) -> bool {
        false
    }

    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `equalsSelectedStringValue(String)`.  Not true value is implemented so
    /// all non-empty values are true.
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }

    /// Java `setMax(int)`.
    pub fn set_max(&self, max: i32) {
        self.maximum.set(max);
        // model.setMaximum((Integer) max)
        if let Some(mut model) = self.spinner.get_spinner_model() {
            model.maximum = Some(max as f64);
            self.spinner.set_spinner_model(model);
        }
    }

    /// Java `setModel(int value, int minimum, int maximum, int stepSize)`.
    pub fn set_model(&self, value: i32, minimum: i32, maximum: i32, step_size: i32) {
        self.minimum.set(minimum);
        self.maximum.set(maximum);
        let model = SpinnerNumberModel::new_int(value, minimum, maximum, step_size);
        self.spinner.set_spinner_model(model);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        Some(self.label.get_text())
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(self.get_label().as_deref())
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&self) {
        let text = self.get_text_void();
        let mut checkpoint = self.checkpoint.borrow_mut();
        if checkpoint.is_none() {
            *checkpoint = Some(TextFieldSetting::new_type(Type::Integer));
        }
        checkpoint.as_mut().unwrap().set_string(text.as_deref());
    }

    /// Java `getCheckpoint()`.  (A copy: the live setting cannot be lent out of
    /// its cell.)
    pub fn get_checkpoint(&self) -> Option<TextFieldSetting> {
        self.checkpoint.borrow().clone()
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    pub fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        let mut checkpoint = self.checkpoint.borrow_mut();
        if checkpoint.is_none() && input.is_some_and(|input| input.is_set() && input.is_text()) {
            *checkpoint = Some(TextFieldSetting::new_type(Type::Integer));
        }
        if let Some(checkpoint) = checkpoint.as_mut() {
            checkpoint.copy(input);
        }
    }

    /// Java `backup()`.
    pub fn backup(&self) {
        let value = self.get_value();
        let mut backup = self.backup.borrow_mut();
        if backup.is_none() {
            *backup = Some(TextFieldSetting::new_type(Type::Integer));
        }
        backup.as_mut().unwrap().set_number(Some(value));
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java `getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `equalsDefaultValue()`.
    pub fn equals_default_value_void(&self) -> bool {
        let text = self.get_text_void();
        self.default_value_setting
            .borrow()
            .as_ref()
            .is_some_and(|setting| setting.is_set() && setting.equals_string(text.as_deref()))
    }

    /// Java `equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        self.default_value_setting
            .borrow()
            .as_ref()
            .is_some_and(|setting| setting.is_set() && setting.equals_string(value))
    }

    /// Java `useDefaultValue()`.
    pub fn use_default_value(&self) {
        let Some(directive_def) = self.directive_def.get() else {
            if let Some(setting) = self.default_value_setting.borrow_mut().as_mut() {
                if setting.is_set() {
                    setting.reset();
                }
            }
            return;
        };
        // only search for default value once
        if self.default_value_setting.borrow().is_none() {
            let mut setting = TextFieldSetting::new_type(Type::Integer);
            let value =
                autodoc_attribute_retriever::INSTANCE.get_default_value(Some(directive_def));
            if let Some(value) = value {
                // if default value has been found, set it in the field setting
                setting.set_string(Some(&value));
            }
            *self.default_value_setting.borrow_mut() = Some(setting);
        }
        let value = {
            let setting = self.default_value_setting.borrow();
            let setting = setting.as_ref().unwrap();
            setting.is_set().then(|| setting.get_value())
        };
        if let Some(value) = value {
            self.set_text(value.as_deref());
        }
    }

    /// Java `restoreFromBackup()`.  If the field was backed up, make the backup
    /// value the displayed value, and turn off the back up.
    pub fn restore_from_backup(&self) {
        let value = match self.backup.borrow().as_ref() {
            Some(backup) if backup.is_set() => backup.get_value(),
            _ => return,
        };
        self.set_value_string(value.as_deref());
        if let Some(backup) = self.backup.borrow_mut().as_mut() {
            backup.reset();
        }
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        self.spinner.set_spinner_value(self.minimum.get() as f64);
    }

    /// Java `setValue(Field)`.
    pub fn set_value_field(&self, input: Option<&dyn Field>) {
        match input {
            None => self.clear(),
            Some(input) => self.set_text(input.get_text_void().as_deref()),
        }
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        false
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        let text = self.get_text_void();
        text.as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        false
    }

    /// Java `getText(boolean doValidation, FieldDisplayer)`.  No validation
    /// available for spinner.
    pub fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, field_displayer1, None)
    }

    /// Java `getText(boolean doValidation, FieldDisplayer, FieldDisplayer)`.
    pub fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let _ = (do_validation, field_displayer1, field_displayer2);
        Ok(Some(self.get_value().to_string()))
    }

    /// Java `getText()`.
    pub fn get_text_void(&self) -> Option<String> {
        Some(self.get_value().to_string())
    }

    /// Java `isFieldHighlightSet()`.
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.is_set())
    }

    /// Java `equalsFieldHighlight()`.
    pub fn equals_field_highlight_void(&self) -> bool {
        let text = self.get_text_void();
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.equals_string(text.as_deref()))
    }

    /// Java `equalsFieldHighlight(String)`.
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.equals_string(value))
    }

    /// Java `setFieldHighlight(String)`.  Creates and sets the field highlight.
    pub fn set_field_highlight_string(&self, value: Option<&str>) {
        if self.field_highlight.borrow().is_none() && value.is_some() {
            *self.field_highlight.borrow_mut() = Some(TextFieldSetting::new_type(Type::Integer));
            // spinner.addChangeListener(this)
            let this = self.this.clone();
            let listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
                if let Some(this) = this.upgrade() {
                    this.state_changed(event);
                }
            });
            self.spinner.add_change_listener(listener);
            // spinner.addFocusListener(this)
            let this = self.this.clone();
            let listener: FocusListener = Rc::new(move |event: &FocusEvent| {
                if let Some(this) = this.upgrade() {
                    if event.gained {
                        this.focus_gained();
                    } else {
                        this.focus_lost();
                    }
                }
            });
            self.spinner.add_focus_listener(listener);
        }
        // Upstream bug fixed (LabeledSpinner.java:354-362): with no field highlight
        // yet and a null value the Java calls `fieldHighlight.set(value)` on null and
        // throws a NullPointerException.  Nothing is set in that case here.
        let exists = match self.field_highlight.borrow_mut().as_mut() {
            Some(field_highlight) => {
                field_highlight.set_string(value);
                true
            }
            None => false,
        };
        if exists {
            self.update_field_highlight();
        }
    }

    /// Java `setFieldHighlight(boolean)`.
    pub fn set_field_highlight_boolean(&self, value: bool) {
        let _ = value;
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
            self.update_field_highlight();
        }
    }

    /// Java `getFieldHighlight()`.  (A copy; see `get_checkpoint`.)
    pub fn get_field_highlight(&self) -> Option<TextFieldSetting> {
        self.field_highlight.borrow().clone()
    }

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    pub fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        if self.field_highlight.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_text())
        {
            *self.field_highlight.borrow_mut() = Some(TextFieldSetting::new_type(Type::Integer));
            // spinner.addChangeListener(this)
            let this = self.this.clone();
            let listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
                if let Some(this) = this.upgrade() {
                    this.state_changed(event);
                }
            });
            self.spinner.add_change_listener(listener);
            // spinner.addFocusListener(this)
            let this = self.this.clone();
            let listener: FocusListener = Rc::new(move |event: &FocusEvent| {
                if let Some(this) = this.upgrade() {
                    if event.gained {
                        this.focus_gained();
                    } else {
                        this.focus_lost();
                    }
                }
            });
            self.spinner.add_focus_listener(listener);
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

    /// Java `clearFieldHighlightValue()`.
    pub fn clear_field_highlight_value(&self) {
        // Upstream bug fixed (LabeledSpinner.java:400-403): the Java resets
        // `fieldHighlight` unguarded and throws a NullPointerException when no
        // field highlight was ever set; there is nothing to clear then.
        if let Some(field_highlight) = self.field_highlight.borrow_mut().as_mut() {
            field_highlight.reset();
        }
        self.update_field_highlight();
    }

    /// Java `stateChanged(ChangeEvent)`.
    pub fn state_changed(&self, e: &ChangeEvent) {
        let _ = e;
        self.update_field_highlight();
    }

    /// Java `focusGained(FocusEvent)`.
    pub fn focus_gained(&self) {}

    /// Java `focusLost(FocusEvent)`.
    pub fn focus_lost(&self) {
        self.update_field_highlight();
    }

    /// Java `updateFieldHighlight()`.  If the field highlight is in use, use the
    /// field highlight color on the foreground of the text field if the value of
    /// the text field equals the field highlight value.  Save the original
    /// foreground.  Otherwise restore the original foreground.  Assumes that
    /// field highlight is not used when the field is disabled.
    pub fn update_field_highlight(&self) {
        // To avoid constantly updating the foreground color, assuming that
        // fieldHighlight is never reassigned to null
        if self.field_highlight.borrow().is_none() || !self.is_enabled() {
            return;
        }
        let value = self.get_value();
        let matches = self
            .field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| {
                field_highlight.is_set() && field_highlight.equals_number(Some(value))
            });
        if matches {
            // save the original color
            if self.orig_text_foreground.get().is_none() {
                self.orig_text_foreground
                    .set(Some(self.spinner.get_foreground()));
            }
            if self.orig_label_foreground.get().is_none() {
                self.orig_label_foreground
                    .set(Some(self.label.get_foreground()));
            }
            self.label.set_foreground(Some(colors::FIELD_HIGHLIGHT));
            self.spinner.set_foreground(Some(colors::FIELD_HIGHLIGHT));
        } else {
            if let Some(orig_text_foreground) = self.orig_text_foreground.get() {
                self.spinner.set_foreground(orig_text_foreground);
            }
            if let Some(orig_label_foreground) = self.orig_label_foreground.get() {
                self.label.set_foreground(orig_label_foreground);
            }
        }
    }

    /// Java `resetToCheckpoint()`.  Resets to checkpointValue if checkpointValue
    /// has been set.  Otherwise has no effect.
    pub fn reset_to_checkpoint(&self) {
        let value = match self.checkpoint.borrow().as_ref() {
            Some(checkpoint) if checkpoint.is_set() => checkpoint.get_value(),
            _ => return,
        };
        self.set_text(value.as_deref());
    }

    /// Java `isDifferentFromCheckpoint(boolean alwaysCheck)`: check for difference
    /// even when the field is disabled or invisible.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.is_visible()) {
            return false;
        }
        let value = self.get_value();
        self.checkpoint
            .borrow()
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals_number(Some(value)))
    }

    /// Java `getValue()`: the spinner's value (an `Integer`, the model being
    /// built from `int`s).
    pub fn get_value(&self) -> Number {
        let value = self.spinner.get_spinner_value();
        if self
            .spinner
            .get_spinner_model()
            .is_none_or(|model| model.integer)
        {
            Number::Integer(value as i32)
        } else {
            Number::Double(value)
        }
    }

    /// Java `isInRange(ConstEtomoNumber)`: true if value empty or >= min and <= max
    /// (if max set).
    pub fn is_in_range(&self, number: Option<&ConstEtomoNumber>) -> bool {
        let Some(number) = number.filter(|number| !number.is_null()) else {
            return true;
        };
        // `number.ge(minimum)` resolves to `ge(long)`; `number.le(maximum)` to
        // `le(int)`.
        number.ge_long(self.minimum.get() as i64)
            && (self.maximum.get() <= self.minimum.get() || number.le_int(self.maximum.get()))
    }

    /// Java `setValue(ConstEtomoNumber)`.
    pub fn set_value_const_etomo_number(&self, value: &ConstEtomoNumber) {
        if value.is_null() {
            self.spinner.set_spinner_value(self.default_value as f64);
        } else {
            self.spinner
                .set_spinner_value(value.get_number().double_value());
        }
    }

    /// Java `setText(String)`.
    pub fn set_text(&self, value: Option<&str>) {
        self.set_value_string(value);
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        match value {
            Some(value) if !java_lang_string_matches_whitespace(value) => {
                let mut number = EtomoNumber::new();
                number.set_string(Some(value));
                self.set_value_const_etomo_number(&number);
            }
            _ => self.spinner.set_spinner_value(self.default_value as f64),
        }
    }

    /// Java `setValue(String, boolean nonempty)`.
    pub fn set_value_string_boolean(&self, value: Option<&str>, nonempty: bool) {
        if !nonempty || value.is_some_and(|value| !value.is_empty()) {
            self.set_value_string(value);
        }
    }

    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&self, value: bool) {
        let _ = value;
    }

    /// Java `setValue(int)`.
    pub fn set_value_int(&self, value: i32) {
        if value == INTEGER_NULL_VALUE {
            self.spinner.set_spinner_value(self.default_value as f64);
        } else {
            self.spinner.set_spinner_value(value as f64);
        }
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
        // Only visually enabled if both enabled and editable
        self.spinner.set_enabled(enabled && self.editable.get());
        self.label.set_enabled(enabled && self.editable.get());
        if enabled && self.editable.get() {
            self.update_field_highlight();
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.editable.set(editable);
        // Editable has no visible effect if the button is disabled.
        if self.enabled.get() {
            // leave the label enabled for uneditable
            self.spinner.set_enabled(editable);
        }
        if self.enabled.get() && editable {
            self.update_field_highlight();
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.panel.is_visible()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, is_visible: bool) {
        self.panel.set_visible(is_visible);
    }

    /// Java `setHighlight(boolean)`.
    pub fn set_highlight(&self, highlight: bool) {
        // Swing painting: the spinner editor's text field background becomes
        // Colors.HIGHLIGHT_BACKGROUND when highlighting, else Colors.BACKGROUND
        // (backgrounds and the editor's internal text field are not modelled).
        let _ = highlight;
    }

    // Java `getTextField()`: the spinner editor's internal `JFormattedTextField`
    // (not modelled by the jdk stand-in).

    /// Java `setTextPreferredSize(Dimension)`.  Set the absolute preferred size of
    /// the text field.
    pub fn set_text_preferred_size(&self) {
        // Swing layout: spinner.setPreferredSize(size).
    }

    /// Java `setTextMaxmimumSize(Dimension)`.  Set the absolute maximum size of the
    /// text field.
    pub fn set_text_maxmimum_size(&self) {
        // Swing layout: spinner.setMaximumSize(size).
    }

    /// Java `setPreferredWidth(int)`.
    pub fn set_preferred_width(&self, width: i32) {
        // Swing layout: the spinner's preferred and maximum width become
        // width * round(UIParameters.getFontSizeAdjustment()).
        let _ = width;
    }

    /// Java `setMaximumSize(Dimension)`.  Set the absolute maximum size of the
    /// panel.
    pub fn set_maximum_size(&self) {
        // Swing layout: panel.setMaximumSize(size).
    }

    // Java `getLabelPreferredSize()`: `label.getPreferredSize()` (Swing layout,
    // not modelled).

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, alignment: f32) {
        // Swing layout: panel.setAlignmentX(alignment).
        let _ = alignment;
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

    /// Java `useUnformattedTooltip(String, String)` (`synchronized`; EDT-only
    /// here).  Use unformattedTooltip to build a tooltip, and then delete
    /// unformattedTooltip.
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

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        let tooltip = tooltip_formatter::INSTANCE.format(text);
        self.panel.set_tool_tip_text(tooltip.as_deref());
        self.spinner.set_tool_tip_text(tooltip.as_deref());
        // getTextField().setToolTipText(tooltip): the editor's internal text field
        // is not modelled.
        self.label.set_tool_tip_text(tooltip.as_deref());
    }

    /// Java `setTooltip(Field)`.
    pub fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            let tooltip = field.get_tooltip();
            self.panel.set_tool_tip_text(tooltip.as_deref());
            self.spinner.set_tool_tip_text(tooltip.as_deref());
            // getTextField().setToolTipText(tooltip): not modelled.
            self.label.set_tool_tip_text(tooltip.as_deref());
        }
    }

    /// Java `getTooltip()`.
    pub fn get_tooltip(&self) -> Option<String> {
        self.spinner.get_tool_tip_text()
    }

    /// Java `addMouseListener(MouseListener)`.
    pub fn add_mouse_listener(&self, listener: Rc<dyn crate::imod::etomo::jdk::MouseListener>) {
        self.panel.add_mouse_listener(listener.clone());
        self.label.add_mouse_listener(listener.clone());
        self.spinner.add_mouse_listener(listener);
    }

    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&self, listener: ChangeListener) {
        self.spinner.add_change_listener(listener);
    }
}

// ---- interface bindings (each forwards to the method above) ----

impl Field for LabeledSpinner {
    fn is_debug(&self) -> bool {
        LabeledSpinner::is_debug(self)
    }
    fn get_name(&self) -> Option<String> {
        LabeledSpinner::get_name(self)
    }
    fn is_boolean(&self) -> bool {
        LabeledSpinner::is_boolean(self)
    }
    fn is_text(&self) -> bool {
        LabeledSpinner::is_text(self)
    }
    fn get_quoted_label(&self) -> Option<String> {
        LabeledSpinner::get_quoted_label(self)
    }
    fn is_enabled(&self) -> bool {
        LabeledSpinner::is_enabled(self)
    }
    fn clear(&self) {
        LabeledSpinner::clear(self)
    }
    fn set_value_field(&self, from: Option<&dyn Field>) {
        LabeledSpinner::set_value_field(self, from)
    }
    fn set_value_string(&self, text: Option<&str>) {
        LabeledSpinner::set_value_string(self, text)
    }
    fn set_value_boolean(&self, bool_: bool) {
        LabeledSpinner::set_value_boolean(self, bool_)
    }
    fn is_empty(&self) -> bool {
        LabeledSpinner::is_empty(self)
    }
    fn is_selected(&self) -> bool {
        LabeledSpinner::is_selected(self)
    }
    fn is_required(&self) -> bool {
        LabeledSpinner::is_required(self)
    }
    fn get_text_void(&self) -> Option<String> {
        LabeledSpinner::get_text_void(self)
    }
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        LabeledSpinner::get_text_boolean_field_displayer(
            self,
            do_validation,
            field_displayer1.as_deref(),
        )
    }
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        LabeledSpinner::get_text_boolean_field_displayer_field_displayer(
            self,
            do_validation,
            field_displayer1.as_deref(),
            field_displayer2.as_deref(),
        )
    }
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        LabeledSpinner::get_directive_def(self)
    }
    fn use_default_value(&self) {
        LabeledSpinner::use_default_value(self)
    }
    fn equals_default_value_void(&self) -> bool {
        LabeledSpinner::equals_default_value_void(self)
    }
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        LabeledSpinner::equals_default_value_string(self, value)
    }
    fn backup(&self) {
        LabeledSpinner::backup(self)
    }
    fn restore_from_backup(&self) {
        LabeledSpinner::restore_from_backup(self)
    }
    fn checkpoint(&self) {
        LabeledSpinner::checkpoint(self)
    }
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        LabeledSpinner::set_checkpoint(self, input)
    }
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        LabeledSpinner::get_checkpoint(self)
            .map(|setting| Box::new(setting) as Box<dyn FieldSettingInterface>)
            .map(Rc::from)
    }
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        LabeledSpinner::is_different_from_checkpoint(self, always_check)
    }
    fn is_field_highlight_set(&self) -> bool {
        LabeledSpinner::is_field_highlight_set(self)
    }
    fn clear_field_highlight(&self) {
        LabeledSpinner::clear_field_highlight(self)
    }
    fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        LabeledSpinner::set_field_highlight_field_setting_interface(self, input)
    }
    fn set_field_highlight_string(&self, input: Option<&str>) {
        LabeledSpinner::set_field_highlight_string(self, input)
    }
    fn set_field_highlight_boolean(&self, input: bool) {
        LabeledSpinner::set_field_highlight_boolean(self, input)
    }
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        LabeledSpinner::get_field_highlight(self)
            .map(|setting| Box::new(setting) as Box<dyn FieldSettingInterface>)
            .map(Rc::from)
    }
    fn equals_field_highlight_void(&self) -> bool {
        LabeledSpinner::equals_field_highlight_void(self)
    }
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        LabeledSpinner::equals_field_highlight_string(self, value)
    }
    fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        LabeledSpinner::set_tool_tip_text(self, tooltip)
    }
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        LabeledSpinner::set_tooltip(self, field)
    }
    fn get_tooltip(&self) -> Option<String> {
        LabeledSpinner::get_tooltip(self)
    }
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        LabeledSpinner::equals_selected_string_value(self, value)
    }
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        LabeledSpinner::set_directive_def(self, directive_def)
    }
    fn get_description(&self) -> String {
        LabeledSpinner::get_description(self).unwrap_or_else(|| "null".to_owned())
    }
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        LabeledSpinner::set_unformatted_tooltip(self, text)
    }
    fn has_unformatted_tooltip(&self) -> bool {
        LabeledSpinner::has_unformatted_tooltip(self)
    }
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        LabeledSpinner::use_unformatted_tooltip(self, param_descr, directive_descr)
    }
}

impl TextFieldInterface for LabeledSpinner {}
