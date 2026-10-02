//! `IMOD/Etomo/src/etomo/ui/swing/CheckBox.java`: a self-naming check box.
//!
//! `public final class CheckBox implements BooleanFieldInterface, BooleanFlagOrigin,
//! FlagDisplay, ActionListener, UIComponent, SwingComponent, ButtonComponent`.
//!
//! The wrapped `JCheckBox` is a jdk stand-in [`JComponent`].  The Java object
//! registers *itself* as an `ActionListener` (two-label switching, field
//! highlight, warning flag); that is one closure, created once and kept in
//! `self_action_listener`, so the source's "avoid adding duplicate listeners"
//! test (`listener.equals(list[i])`) still recognises it.
//!
//! Every Java method body is an inherent method here (overloads carry the
//! parameter-type suffix of `ui.md`); the interface impls at the end bind the
//! interfaces' methods to those bodies.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::logic::autodoc_attribute_retriever;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::boolean_field_interface::BooleanFieldInterface;
use crate::imod::etomo::ui::boolean_field_setting::BooleanFieldSetting;
use crate::imod::etomo::ui::boolean_flag_extension::BooleanFlagExtension;
use crate::imod::etomo::ui::boolean_flag_origin::BooleanFlagOrigin;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::ui::swing::button_component::ButtonComponent;
use crate::imod::etomo::ui::swing::colors;
use crate::imod::etomo::ui::swing::component_style_extension;
use crate::imod::etomo::ui::swing::swing_component::SwingComponent;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// `java.awt.Color` as the jdk stand-in carries it (RGB); `None` inside is the
/// look-and-feel default foreground.
type Rgb = (u8, u8, u8);

/// Java `CheckBox`.
pub struct CheckBox {
    /// Java `this`, for the places the source passes itself.
    this: Weak<CheckBox>,
    check_box: Rc<JComponent>,
    /// The Java object as an `ActionListener` (created on first use).
    self_action_listener: RefCell<Option<ActionListener>>,

    debug: Cell<bool>,
    /// Java `origForeground`: outer `None` is Java null; the inner value is
    /// what `getForeground()` returned (the jdk's `None` is the look-and-feel
    /// default, which Swing reports as a non-null colour).
    orig_foreground: Cell<Option<Option<Rgb>>>,
    directive_def: Cell<Option<DirectiveDef>>,
    // Field settings should not be reset to null.
    backup: RefCell<Option<BooleanFieldSetting>>,
    default_value: RefCell<Option<BooleanFieldSetting>>,
    checkpoint: RefCell<Option<BooleanFieldSetting>>,
    field_highlight: RefCell<Option<BooleanFieldSetting>>,

    enabled: Cell<bool>,
    editable: Cell<bool>,
    selected_string_value: RefCell<Option<String>>,
    tooltip: RefCell<Option<String>>,
    alternate_tooltip: RefCell<Option<String>>,
    text: RefCell<Option<String>>,
    alternate_text: RefCell<Option<String>>,

    flag_extension: RefCell<Option<Rc<BooleanFlagExtension>>>,
    /// Java `defaultBackground`.  Backgrounds are not modelled by the jdk
    /// stand-in (painting), so this stays `None`; it is still handed to
    /// `ComponentStyleExtension.updateAppearance` where the Java passes it.
    default_background: Cell<Option<Rgb>>,
    flag_type: Cell<Option<&'static FlagType>>,
    unformatted_tooltip: RefCell<Option<String>>,
    label_when_false: RefCell<Option<String>>,
    label_when_true: RefCell<Option<String>>,
    action_command_when_false: RefCell<Option<String>>,
    action_command_when_true: RefCell<Option<String>>,
}

impl CheckBox {
    fn construct(this: &Weak<CheckBox>, check_box: Rc<JComponent>) -> CheckBox {
        CheckBox {
            this: this.clone(),
            check_box,
            self_action_listener: RefCell::new(None),
            debug: Cell::new(false),
            orig_foreground: Cell::new(None),
            directive_def: Cell::new(None),
            backup: RefCell::new(None),
            default_value: RefCell::new(None),
            checkpoint: RefCell::new(None),
            field_highlight: RefCell::new(None),
            enabled: Cell::new(true),
            editable: Cell::new(true),
            selected_string_value: RefCell::new(None),
            tooltip: RefCell::new(None),
            alternate_tooltip: RefCell::new(None),
            text: RefCell::new(None),
            alternate_text: RefCell::new(None),
            flag_extension: RefCell::new(None),
            default_background: Cell::new(None),
            flag_type: Cell::new(None),
            unformatted_tooltip: RefCell::new(None),
            label_when_false: RefCell::new(None),
            label_when_true: RefCell::new(None),
            action_command_when_false: RefCell::new(None),
            action_command_when_true: RefCell::new(None),
        }
    }

    /// Java package-private `CheckBox()`.
    pub fn new_void() -> Rc<CheckBox> {
        Rc::new_cyclic(|this| CheckBox::construct(this, JComponent::new_check_box("")))
    }

    /// Java `CheckBox(String text)`.
    pub fn new_string(text: Option<&str>) -> Rc<CheckBox> {
        let instance = Rc::new_cyclic(|this| {
            CheckBox::construct(this, JComponent::new_check_box(text.unwrap_or("")))
        });
        instance.set_name(text);
        instance
    }

    /// Java `CheckBox(String textFalse, String textTrue)`.
    ///
    /// If both parameters are not null, it changes the label based on whether the
    /// checkbox is checked or not.  If only one parameter is not null, it uses that
    /// parameter as the label.
    pub fn new_string_string(text_false: Option<&str>, text_true: Option<&str>) -> Rc<CheckBox> {
        let instance =
            Rc::new_cyclic(|this| CheckBox::construct(this, JComponent::new_check_box("")));
        if text_false.is_none() && text_true.is_none() {
            return instance;
        }
        if text_false.is_none() || text_true.is_none() {
            // Single label checkbox:
            if let Some(text_false) = text_false {
                instance.check_box.set_text(text_false);
                instance.set_name(Some(text_false));
                return instance;
            }
            instance.check_box.set_text(text_true.unwrap_or(""));
            instance.set_name(text_true);
            // Upstream bug fixed (CheckBox.java:185-188): the Java has no `return`
            // here, so a check box built with only `textTrue` falls through into the
            // two-label code, which sets its text to the null `textFalse` (the box
            // ends up unlabeled and named "cb.null").  The evident intent, stated in
            // the javadoc and done for `textFalse`, is a single-label check box.
            return instance;
        }
        // Two label checkbox:
        *instance.label_when_false.borrow_mut() = text_false.map(str::to_owned);
        *instance.label_when_true.borrow_mut() = text_true.map(str::to_owned);
        // TextFalse is set as the default, but need the action command for both labels.
        instance.check_box.set_text(text_true.unwrap_or(""));
        *instance.action_command_when_true.borrow_mut() = instance.check_box.get_action_command();
        instance.check_box.set_text(text_false.unwrap_or(""));
        *instance.action_command_when_false.borrow_mut() =
            instance.check_box.get_action_command();
        instance.set_name(text_false);
        instance.add_action_listener(Some(instance.get_self_action_listener()));
        instance
    }

    /// The Java object used as its own `ActionListener` (`addActionListener(this)`):
    /// one closure per instance, so the duplicate test in `addActionListener` sees
    /// the same listener each time.
    fn get_self_action_listener(&self) -> ActionListener {
        if let Some(listener) = self.self_action_listener.borrow().as_ref() {
            return listener.clone();
        }
        let this = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(this) = this.upgrade() {
                this.action_performed(event);
            }
        });
        *self.self_action_listener.borrow_mut() = Some(listener.clone());
        listener
    }

    /// Java `doClick(boolean allowHeadless)`: arm and press the model, paint, wait
    /// 68 ms, release (which toggles the box and fires the action listeners).
    pub fn do_click(&self, allow_headless: bool) {
        // Swing painting: getSize, paintImmediately (and its HeadlessException,
        // rethrown unless allowHeadless) are not modelled.
        let _ = allow_headless;
        std::thread::sleep(std::time::Duration::from_millis(68));
        // model.setArmed(true); setPressed(true); ... setPressed(false);
        // setArmed(false): the release of an armed, pressed toggle-button model
        // toggles the selection and fires the action listeners, which is the
        // jdk stand-in's `doClick` (which also does nothing when disabled, as the
        // disabled model ignores setPressed).
        self.check_box.do_click();
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.check_box.get_name()
    }

    /// Java `isBoolean()`.
    pub fn is_boolean(&self) -> bool {
        true
    }

    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug.get() || ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `isText()`.
    pub fn is_text(&self) -> bool {
        false
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.check_box.is_visible()
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Java `equals(Object)`: identity.
    pub fn equals_object(&self, object: &CheckBox) -> bool {
        std::ptr::eq(object, self)
    }

    /// Java `equals(Document)`.  (The document is represented by its text
    /// component in the jdk stand-in.)
    pub fn equals_document(&self, document: &Rc<JComponent>) -> bool {
        let _ = document;
        false
    }

    /// Java `setSelectedStringValue(String)`.
    pub fn set_selected_string_value(&self, value: Option<&str>) {
        *self.selected_string_value.borrow_mut() = value.map(str::to_owned);
    }

    /// Java `equalsSelectedStringValue(String)`.
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        match self.selected_string_value.borrow().as_deref() {
            None => value.is_some_and(|value| !value.is_empty()),
            Some(selected_string_value) => Some(selected_string_value) == value,
        }
    }

    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        false
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.check_box.is_selected()
    }

    /// Java `getText()`.
    pub fn get_text_void(&self) -> Option<String> {
        Some(self.check_box.get_text())
    }

    /// Java `getText(boolean doValidation, FieldDisplayer)`.  Validation is not
    /// available for the label.
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
        Ok(self.get_text_void())
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        let mut label = self.get_text_void();
        if label
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
        {
            label = self.check_box.get_name();
        }
        utilities::quote_label(label.as_deref())
    }

    /// Java `setText(String)`.  Sets the label.  Handles alternate text.  Overrides
    /// label change based on selection state.
    pub fn set_text(&self, text: Option<&str>) {
        self.check_box.set_text(text.unwrap_or(""));
        self.set_name(text);
        *self.text.borrow_mut() = text.map(str::to_owned);
        *self.label_when_false.borrow_mut() = None;
        *self.label_when_true.borrow_mut() = None;
    }

    /// Java `setAlternateText(String)`.  Stores a second text.  Does not switch to
    /// the second text.
    pub fn set_alternate_text(&self, text: Option<&str>) {
        *self.alternate_text.borrow_mut() = text.map(str::to_owned);
    }

    /// Java `switchText(boolean)`.  Switch to/from the alternate text.
    pub fn switch_text(&self, alternate: bool) {
        let alternate_text = self.alternate_text.borrow().clone();
        if alternate && alternate_text.is_some() {
            self.check_box.set_text(alternate_text.as_deref().unwrap_or(""));
        } else {
            let text = self.text.borrow().clone();
            self.check_box.set_text(text.as_deref().unwrap_or(""));
        }
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected_boolean(&self, input: bool) {
        self.check_box.set_selected(input);
        if self.label_when_false.borrow().is_some() {
            if input {
                let label = self.label_when_true.borrow().clone();
                self.check_box.set_text(label.as_deref().unwrap_or(""));
                self.set_name(label.as_deref());
            } else {
                let label = self.label_when_false.borrow().clone();
                self.check_box.set_text(label.as_deref().unwrap_or(""));
                self.set_name(label.as_deref());
            }
        }
        self.update_field_highlight();
    }

    /// Java `setSelected(ConstEtomoNumber, boolean allowEmpty)`.
    pub fn set_selected_const_etomo_number_boolean(
        &self,
        input: Option<&ConstEtomoNumber>,
        allow_empty: bool,
    ) {
        if allow_empty || input.is_some_and(|input| !input.is_null()) {
            // Upstream bug fixed (CheckBox.java:337-341): with allowEmpty true and a
            // null input the Java calls `input.is()` and throws a
            // NullPointerException.  A null number is left alone here.
            if let Some(input) = input {
                self.set_selected_boolean(input.is());
            }
        }
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = &ui_test_field_type::CHECK_BOX;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        self.check_box.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.check_box.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `backup()`.
    pub fn backup(&self) {
        let selected = self.is_selected();
        let mut backup = self.backup.borrow_mut();
        if backup.is_none() {
            *backup = Some(BooleanFieldSetting::new());
        }
        backup.as_mut().unwrap().set_boolean(selected);
    }

    /// Java `restoreFromBackup()`.  If the field was backed up, make the backup
    /// value the displayed value, and turn off the back up.
    pub fn restore_from_backup(&self) {
        let value = match self.backup.borrow().as_ref() {
            Some(backup) if backup.is_set() => Some(backup.is_value()),
            _ => None,
        };
        if let Some(value) = value {
            self.set_selected_boolean(value);
        }
    }

    /// Java `setActionCommand(String)`.
    pub fn set_action_command(&self, input: Option<&str>) {
        self.check_box.set_action_command(input);
    }

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, input: f32) {
        // Swing layout: checkBox.setAlignmentX(input).
        let _ = input;
    }

    /// Java `setBackground(Color)`.
    pub fn set_background(&self, color: Option<Rgb>) {
        // Swing painting: checkBox.setBackground(color) (backgrounds are not
        // modelled).
        let _ = color;
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        self.set_selected_boolean(false);
    }

    /// Java `setValue(Field)`.  Copy the value, checkpoint, and field highlight
    /// settings.
    pub fn set_value_field(&self, input: Option<&dyn Field>) {
        match input {
            None => self.clear(),
            Some(input) => self.set_selected_boolean(input.is_selected()),
        }
    }

    /// Java `setValue(String)`.
    pub fn set_value_string(&self, value: Option<&str>) {
        let _ = value;
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, input: bool) {
        self.check_box.set_visible(input);
    }

    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&self, value: bool) {
        self.set_selected_boolean(value);
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
        if let Some(default_value) = self.default_value.borrow_mut().as_mut() {
            default_value.reset();
        }
    }

    /// Java `getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `useDefaultValue()`.
    pub fn use_default_value(&self) {
        let Some(directive_def) = self.directive_def.get() else {
            if let Some(default_value) = self.default_value.borrow_mut().as_mut() {
                if default_value.is_set() {
                    default_value.reset();
                }
            }
            return;
        };
        // only search for default value once for this directiveDef
        if self.default_value.borrow().is_none() {
            let mut default_value = BooleanFieldSetting::new();
            let value = autodoc_attribute_retriever::INSTANCE.get_default_value(Some(directive_def));
            if let Some(value) = value {
                // if default value has been found, set it in the field setting
                default_value.set_string(Some(&value));
            }
            *self.default_value.borrow_mut() = Some(default_value);
        }
        let value = {
            let default_value = self.default_value.borrow();
            let default_value = default_value.as_ref().unwrap();
            default_value.is_set().then(|| default_value.is_value())
        };
        if let Some(value) = value {
            self.set_selected_boolean(value);
        }
    }

    /// Java `equalsDefaultValue()`.
    pub fn equals_default_value_void(&self) -> bool {
        let selected = self.is_selected();
        self.default_value
            .borrow()
            .as_ref()
            .is_some_and(|default_value| default_value.equals_boolean(selected))
    }

    /// Java `equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        self.default_value
            .borrow()
            .as_ref()
            .is_some_and(|default_value| default_value.equals_string(value))
    }

    /// Java `checkpoint()`.
    pub fn checkpoint_void(&self) {
        let selected = self.is_selected();
        let mut checkpoint = self.checkpoint.borrow_mut();
        if checkpoint.is_none() {
            *checkpoint = Some(BooleanFieldSetting::new());
        }
        checkpoint.as_mut().unwrap().set_boolean(selected);
    }

    /// Java `checkpoint(boolean)`.
    pub fn checkpoint_boolean(&self, value: bool) {
        let mut checkpoint = self.checkpoint.borrow_mut();
        if checkpoint.is_none() {
            *checkpoint = Some(BooleanFieldSetting::new());
        }
        checkpoint.as_mut().unwrap().set_boolean(value);
    }

    /// Java `getCheckpoint()`.  (The live setting cannot be lent out of the cell;
    /// a copy is returned.)
    pub fn get_checkpoint(&self) -> Option<Box<dyn FieldSettingInterface>> {
        self.checkpoint
            .borrow()
            .as_ref()
            .map(|checkpoint| Box::new(checkpoint.clone()) as Box<dyn FieldSettingInterface>)
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> Option<Rc<dyn SwingComponent>> {
        self.this.upgrade().map(|this| this as Rc<dyn SwingComponent>)
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.check_box.clone()
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

    /// Java `resetToCheckpoint()`.  Resets to checkpointValue if checkpointValue
    /// has been set.  Otherwise has no effect.
    pub fn reset_to_checkpoint(&self) {
        let value = match self.checkpoint.borrow().as_ref() {
            Some(checkpoint) if checkpoint.is_set() => checkpoint.is_value(),
            _ => return,
        };
        self.set_selected_boolean(value);
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java `setFieldHighlight(boolean)`.
    pub fn set_field_highlight_boolean(&self, value: bool) {
        if self.field_highlight.borrow().is_none() {
            *self.field_highlight.borrow_mut() = Some(BooleanFieldSetting::new());
            self.add_action_listener(Some(self.get_self_action_listener()));
        }
        self.field_highlight
            .borrow_mut()
            .as_mut()
            .unwrap()
            .set_boolean(value);
        self.update_field_highlight();
    }

    /// Java `setFieldHighlight(String)`.
    pub fn set_field_highlight_string(&self, input: Option<&str>) {
        if self.field_highlight.borrow().is_none() && input.is_some() {
            *self.field_highlight.borrow_mut() = Some(BooleanFieldSetting::new());
            self.add_action_listener(Some(self.get_self_action_listener()));
        }
        let exists = match self.field_highlight.borrow_mut().as_mut() {
            Some(field_highlight) => {
                field_highlight.set_string(input);
                true
            }
            None => false,
        };
        if exists {
            self.update_field_highlight();
        }
    }

    /// Java `getFieldHighlight()`.  (A copy; see `get_checkpoint`.)
    pub fn get_field_highlight(&self) -> Option<BooleanFieldSetting> {
        self.field_highlight.borrow().clone()
    }

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    pub fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        if self.field_highlight.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_boolean())
        {
            *self.field_highlight.borrow_mut() = Some(BooleanFieldSetting::new());
            self.add_action_listener(Some(self.get_self_action_listener()));
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

    /// Java `isFieldHighlightSet()`.
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.is_set())
    }

    /// Java `equalsFieldHighlight()`.
    pub fn equals_field_highlight_void(&self) -> bool {
        let selected = self.is_selected();
        self.field_highlight.borrow().as_ref().is_some_and(|field_highlight| {
            field_highlight.is_set() && field_highlight.equals_boolean(selected)
        })
    }

    /// Java `equalsFieldHighlight(String)`.
    pub fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.field_highlight.borrow().as_ref().is_some_and(|field_highlight| {
            field_highlight.is_set() && field_highlight.equals_string(value)
        })
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.check_box.get_action_command()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
        // Only visually enabled if both enabled and editable
        self.check_box.set_enabled(enabled && self.editable.get());
        if enabled && self.editable.get() {
            self.update_field_highlight();
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.editable.set(editable);
        // Editable has no visible effect if the button is disabled.
        if self.enabled.get() {
            self.check_box.set_enabled(editable);
        }
        if self.enabled.get() && editable {
            self.update_field_highlight();
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.editable.get()
    }

    /// Java `equalsActionCommand(String)`.
    pub fn equals_action_command(&self, action_command: Option<&str>) -> bool {
        let Some(action_command) = action_command else {
            return false;
        };
        if Some(action_command) == self.check_box.get_action_command().as_deref() {
            return true;
        }
        if self.action_command_when_false.borrow().is_some() {
            return Some(action_command) == self.action_command_when_false.borrow().as_deref()
                || Some(action_command) == self.action_command_when_true.borrow().as_deref();
        }
        false
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, e: &ActionEvent) {
        let _ = e;
        if self.label_when_false.borrow().is_some() {
            if self.check_box.is_selected() {
                let label = self.label_when_true.borrow().clone();
                self.check_box.set_text(label.as_deref().unwrap_or(""));
                self.set_name(label.as_deref());
            } else {
                let label = self.label_when_false.borrow().clone();
                self.check_box.set_text(label.as_deref().unwrap_or(""));
                self.set_name(label.as_deref());
            }
        }
        self.update_field_highlight();
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: Option<ActionListener>) {
        let Some(listener) = listener else {
            return;
        };
        // Avoid adding duplicate listeners.
        let list = self.check_box.get_action_listeners();
        for item in &list {
            if Rc::ptr_eq(&listener, item) {
                return;
            }
        }
        self.check_box.add_action_listener(listener);
    }

    /// Java `removeActionListener(ActionListener)`.
    pub fn remove_action_listener(&self, listener: &ActionListener) {
        self.check_box.remove_action_listener(listener);
    }

    /// Java `updateFieldHighlight()`.
    fn update_field_highlight(&self) {
        let selected = self.is_selected();
        let matches = self
            .field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.equals_boolean(selected));
        if matches {
            if self.orig_foreground.get().is_none() {
                // origForeground must be set if the foreground is going to be changed
                // (Java: `if (origForeground == null) origForeground = Color.black`;
                // the jdk's `None` is the look-and-feel default, a real colour.)
                self.orig_foreground.set(Some(self.check_box.get_foreground()));
            }
            self.check_box.set_foreground(Some(colors::FIELD_HIGHLIGHT));
            return;
        }
        if let Some(orig_foreground) = self.orig_foreground.get() {
            // Field highlight value currently doesn't match the field text, or field
            // highlight was removed.
            self.check_box.set_foreground(orig_foreground);
        }
    }

    /// Java `isDifferentFromCheckpoint()`.  If the field is disabled or not
    /// visible then return false because its value doesn't matter.  It returns
    /// true if the checkpoint has not been done.
    pub fn is_different_from_checkpoint_void(&self) -> bool {
        self.is_different_from_checkpoint_boolean(false)
    }

    /// Java `isDifferentFromCheckpoint(boolean alwaysCheck)`: check for difference
    /// even when the field is disabled or invisible.
    pub fn is_different_from_checkpoint_boolean(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.check_box.is_visible()) {
            return false;
        }
        let selected = self.is_selected();
        self.checkpoint
            .borrow()
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals_boolean(selected))
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text_string(&self, text: Option<&str>) {
        self.set_preformatted_tooltip(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setToolTipText(String autodocName, ReadOnlySection, String enumValue)`.
    pub fn set_tool_tip_text_string_read_only_section_string(
        &self,
        autodoc_name: Option<&str>,
        section: &dyn ReadOnlySection,
        enum_value: Option<&str>,
    ) {
        self.set_tool_tip_text_string(
            etomo_autodoc::get_tooltip_enum_value_name(autodoc_name, section, enum_value)
                .as_deref(),
        );
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
    pub fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        let unformatted_tooltip = self.unformatted_tooltip.borrow().clone();
        self.set_tool_tip_text_string(
            tooltip_formatter::INSTANCE
                .build_tooltip(unformatted_tooltip.as_deref(), param_descr, directive_descr)
                .as_deref(),
        );
        *self.unformatted_tooltip.borrow_mut() = None;
    }

    /// Java `setTooltip(Field)`.
    pub fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            self.set_preformatted_tooltip(field.get_tooltip().as_deref());
        }
    }

    /// Java `setPreformattedTooltip(String)`.  Sets a preformatted tooltip.
    pub fn set_preformatted_tooltip(&self, tooltip: Option<&str>) {
        self.check_box.set_tool_tip_text(tooltip);
        *self.tooltip.borrow_mut() = tooltip.map(str::to_owned);
    }

    /// Java `setAlternateTooltipText(String)`.  Stores a second tooltip.  Does not
    /// switch to the second tooltip.
    pub fn set_alternate_tooltip_text(&self, text: Option<&str>) {
        *self.alternate_tooltip.borrow_mut() = tooltip_formatter::INSTANCE.format(text);
    }

    /// Java `switchTooltips(boolean)`.  Switch to/from the alternate tooltip.
    pub fn switch_tooltips(&self, alternate: bool) {
        let alternate_tooltip = self.alternate_tooltip.borrow().clone();
        if alternate && alternate_tooltip.is_some() {
            self.check_box.set_tool_tip_text(alternate_tooltip.as_deref());
        } else {
            let tooltip = self.tooltip.borrow().clone();
            self.check_box.set_tool_tip_text(tooltip.as_deref());
        }
    }

    /// Java `getTooltip()`.  Gets the tooltip that is currently in use.
    pub fn get_tooltip(&self) -> Option<String> {
        self.check_box.get_tool_tip_text()
    }

    /// Java `printInfo()`.
    pub fn print_info_void(&self) {
        println!("{}", self.check_box.get_name().as_deref().unwrap_or("null"));
        self.print_info_container(self.check_box.get_parent());
    }

    /// Java `printInfo(Container)`.
    fn print_info_container(&self, parent: Option<Rc<JComponent>>) {
        // Java prints `parent.toString()` (Swing's class/bounds/flags dump, not
        // modelled); the component kind stands in for it.
        match &parent {
            None => println!("null"),
            Some(parent) => println!("{:?}", parent.kind()),
        }
        if let Some(parent) = parent {
            let parent_name = parent.get_name();
            if let Some(parent_name) = parent_name {
                println!("{parent_name}");
            } else {
                self.print_info_container(parent.get_parent());
            }
        }
    }

    /// Java `enableWarning(boolean)`.
    pub fn enable_warning(&self, value: bool) {
        // Give the button a generic component style for displaying the warning.
        // Swing painting: `defaultBackground = checkBox.getBackground()` (or
        // Color.GRAY when null) - backgrounds are not modelled.
        // Use flag extension to set warning flag.
        if self.flag_extension.borrow().is_none() {
            self.add_action_listener(Some(self.get_self_action_listener()));
            let this = self.this.upgrade().expect("CheckBox used after it was dropped");
            let flag_extension = BooleanFlagExtension::new(this.clone() as Rc<dyn BooleanFlagOrigin>);
            *self.flag_extension.borrow_mut() = Some(flag_extension.clone());
            flag_extension.add_flag_display(Some(this as Rc<dyn FlagDisplay>));
        }
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.enable_warning(value);
        flag_extension.update();
    }

    /// Java `disableWarning()`.
    pub fn disable_warning(&self) {
        let Some(flag_extension) = self.flag_extension.borrow().clone() else {
            return;
        };
        flag_extension.disable_warning();
        flag_extension.update();
    }

    /// Java `setFlag(FlagType)`.
    pub fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        component_style_extension::INSTANCE.update_appearance(
            Some(&self.check_box),
            flag_type,
            None,
            self.default_background.get(),
        );
        self.flag_type.set(flag_type);
    }
}

/// Java `toString()`.
impl fmt::Display for CheckBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[text:{}]", self.get_text_void().as_deref().unwrap_or("null"))
    }
}

// ---- interface bindings (each forwards to the method above) ----

impl Field for CheckBox {
    fn is_debug(&self) -> bool {
        CheckBox::is_debug(self)
    }
    fn get_name(&self) -> Option<String> {
        CheckBox::get_name(self)
    }
    fn is_boolean(&self) -> bool {
        CheckBox::is_boolean(self)
    }
    fn is_text(&self) -> bool {
        CheckBox::is_text(self)
    }
    fn get_quoted_label(&self) -> Option<String> {
        CheckBox::get_quoted_label(self)
    }
    fn is_enabled(&self) -> bool {
        CheckBox::is_enabled(self)
    }
    fn clear(&self) {
        CheckBox::clear(self)
    }
    fn set_value_field(&self, from: Option<&dyn Field>) {
        CheckBox::set_value_field(self, from)
    }
    fn set_value_string(&self, text: Option<&str>) {
        CheckBox::set_value_string(self, text)
    }
    fn set_value_boolean(&self, bool_: bool) {
        CheckBox::set_value_boolean(self, bool_)
    }
    fn is_empty(&self) -> bool {
        CheckBox::is_empty(self)
    }
    fn is_selected(&self) -> bool {
        CheckBox::is_selected(self)
    }
    fn is_required(&self) -> bool {
        CheckBox::is_required(self)
    }
    fn get_text_void(&self) -> Option<String> {
        CheckBox::get_text_void(self)
    }
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        CheckBox::get_text_boolean_field_displayer(self, do_validation, field_displayer1.as_deref())
    }
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        CheckBox::get_text_boolean_field_displayer_field_displayer(
            self,
            do_validation,
            field_displayer1.as_deref(),
            field_displayer2.as_deref(),
        )
    }
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        CheckBox::get_directive_def(self)
    }
    fn use_default_value(&self) {
        CheckBox::use_default_value(self)
    }
    fn equals_default_value_void(&self) -> bool {
        CheckBox::equals_default_value_void(self)
    }
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        CheckBox::equals_default_value_string(self, value)
    }
    fn backup(&self) {
        CheckBox::backup(self)
    }
    fn restore_from_backup(&self) {
        CheckBox::restore_from_backup(self)
    }
    fn checkpoint(&self) {
        CheckBox::checkpoint_void(self)
    }
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        CheckBox::set_checkpoint(self, input)
    }
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        CheckBox::get_checkpoint(self)
            .map(Rc::from)
    }
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        CheckBox::is_different_from_checkpoint_boolean(self, always_check)
    }
    fn is_field_highlight_set(&self) -> bool {
        CheckBox::is_field_highlight_set(self)
    }
    fn clear_field_highlight(&self) {
        CheckBox::clear_field_highlight(self)
    }
    fn set_field_highlight_field_setting_interface(&self, input: Option<&dyn FieldSettingInterface>) {
        CheckBox::set_field_highlight_field_setting_interface(self, input)
    }
    fn set_field_highlight_string(&self, input: Option<&str>) {
        CheckBox::set_field_highlight_string(self, input)
    }
    fn set_field_highlight_boolean(&self, input: bool) {
        CheckBox::set_field_highlight_boolean(self, input)
    }
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        CheckBox::get_field_highlight(self)
            .map(|setting| Box::new(setting) as Box<dyn FieldSettingInterface>)
            .map(Rc::from)
    }
    fn equals_field_highlight_void(&self) -> bool {
        CheckBox::equals_field_highlight_void(self)
    }
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        CheckBox::equals_field_highlight_string(self, value)
    }
    fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        CheckBox::set_tool_tip_text_string(self, tooltip)
    }
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        CheckBox::set_tooltip(self, field)
    }
    fn get_tooltip(&self) -> Option<String> {
        CheckBox::get_tooltip(self)
    }
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        CheckBox::equals_selected_string_value(self, value)
    }
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        CheckBox::set_directive_def(self, directive_def)
    }
    fn get_description(&self) -> String {
        CheckBox::get_description(self)
            .unwrap_or_else(|| "null".to_owned())
    }
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        CheckBox::set_unformatted_tooltip(self, text)
    }
    fn has_unformatted_tooltip(&self) -> bool {
        CheckBox::has_unformatted_tooltip(self)
    }
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        CheckBox::use_unformatted_tooltip(self, param_descr, directive_descr)
    }
}

impl BooleanFieldInterface for CheckBox {}

impl BooleanFlagOrigin for CheckBox {
    fn is_selected(&self) -> bool {
        CheckBox::is_selected(self)
    }
}

impl FlagDisplay for CheckBox {
    fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        CheckBox::set_flag(self, flag_type)
    }
}

impl UIComponent for CheckBox {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        CheckBox::get_component(self)
    }
}

impl SwingComponent for CheckBox {
    fn get_component(&self) -> Rc<JComponent> {
        CheckBox::get_component(self)
    }
}

impl ButtonComponent for CheckBox {
    fn add_action_listener(&self, listener: ActionListener) {
        CheckBox::add_action_listener(self, Some(listener))
    }
    fn is_selected(&self) -> bool {
        CheckBox::is_selected(self)
    }
    fn get_action_command(&self) -> Option<String> {
        CheckBox::get_action_command(self)
    }
    fn is_enabled(&self) -> bool {
        CheckBox::is_enabled(self)
    }
}
