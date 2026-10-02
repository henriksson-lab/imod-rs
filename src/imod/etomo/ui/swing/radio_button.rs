//! `IMOD/Etomo/src/etomo/ui/swing/RadioButton.java`: a self-naming radio button.
//!
//! `public final class RadioButton implements RadioButtonInterface,
//! BooleanFieldInterface, ItemListener, UIComponent, SwingComponent,
//! ButtonComponent`.
//!
//! The wrapped `JRadioButton` is a jdk stand-in [`JComponent`] whose button
//! model is a [`RadioButtonModel`] (the Java nested class) reporting back
//! through [`RadioButtonInterface`].  The Java object registers itself as an
//! `ItemListener` on every member of its group for the field highlight; that is
//! one closure, kept in `self_item_listener`.
//!
//! Every Java method body is an inherent method here (overloads carry the
//! parameter-type suffix of `ui.md`); the interface impls at the end bind the
//! interfaces' methods to those bodies.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{
    ActionListener, ButtonGroup, ButtonModel, ChangeListener, ItemEvent, ItemListener, JComponent,
};
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
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::swing::abstract_radio_button_model::AbstractRadioButtonModel;
use crate::imod::etomo::ui::swing::button_component::ButtonComponent;
use crate::imod::etomo::ui::swing::colors;
use crate::imod::etomo::ui::swing::field_lock_controller::FieldLockController;
use crate::imod::etomo::ui::swing::radio_button_interface::{
    EnumeratedTypeRef, RadioButtonInterface,
};
use crate::imod::etomo::ui::swing::swing_component::SwingComponent;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// `java.awt.Color` as the jdk stand-in carries it (RGB).
type Rgb = (u8, u8, u8);

/// Java `RadioButton`.
pub struct RadioButton {
    /// Java `this`, for the places the source passes itself.
    this: Weak<RadioButton>,
    radio_button: Rc<JComponent>,
    enumerated_type: Option<EnumeratedTypeRef>,
    group: Option<Rc<ButtonGroup>>,
    field_lock_controller: Rc<FieldLockController>,
    /// The Java object as an `ItemListener` (created on first use).
    self_item_listener: RefCell<Option<ItemListener>>,

    debug: Cell<bool>,
    /// Java `origForeground`: outer `None` is Java null; the inner value is
    /// what `getForeground()` returned (the jdk's `None` is the look-and-feel
    /// default, which Swing reports as a non-null colour, so Java's
    /// `Color.black` fallback never applies).
    orig_foreground: Cell<Option<Option<Rgb>>>,
    directive_def: Cell<Option<DirectiveDef>>,
    backup: RefCell<Option<BooleanFieldSetting>>,
    checkpoint: RefCell<Option<BooleanFieldSetting>>,
    default_value: RefCell<Option<BooleanFieldSetting>>,
    field_highlight: RefCell<Option<BooleanFieldSetting>>,
    // Directive value associated with this instance being selected.
    selected_string_value: RefCell<Option<String>>,
    unformatted_tooltip: RefCell<Option<String>>,
}

impl RadioButton {
    /// Java `RadioButton(String text)`.
    pub fn new_string(text: Option<&str>) -> Rc<RadioButton> {
        Self::new_string_enumerated_type_button_group_radio_button_model(text, None, None, None)
    }

    /// Java `RadioButton(String text, String tflabel)`.
    pub fn new_string_string(text: Option<&str>, tflabel: Option<&str>) -> Rc<RadioButton> {
        Self::new_string_string_enumerated_type_button_group_radio_button_model(
            text, tflabel, None, None, None,
        )
    }

    /// Java `RadioButton(String text, ButtonGroup group)`.
    pub fn new_string_button_group(
        text: Option<&str>,
        group: Option<&Rc<ButtonGroup>>,
    ) -> Rc<RadioButton> {
        Self::new_string_enumerated_type_button_group_radio_button_model(text, None, group, None)
    }

    /// Java `RadioButton(String text, String tflabel, ButtonGroup group)`.
    pub fn new_string_string_button_group(
        text: Option<&str>,
        tflabel: Option<&str>,
        group: Option<&Rc<ButtonGroup>>,
    ) -> Rc<RadioButton> {
        Self::new_string_string_enumerated_type_button_group_radio_button_model(
            text, tflabel, None, group, None,
        )
    }

    /// Java `RadioButton(String text, ButtonGroup group, RadioButtonModel model)`.
    pub fn new_string_button_group_radio_button_model(
        text: Option<&str>,
        group: Option<&Rc<ButtonGroup>>,
        model: Option<Rc<RadioButtonModel>>,
    ) -> Rc<RadioButton> {
        Self::new_string_enumerated_type_button_group_radio_button_model(text, None, group, model)
    }

    /// Java `RadioButton(ButtonGroup group)`.
    pub fn new_button_group(group: Option<&Rc<ButtonGroup>>) -> Rc<RadioButton> {
        Self::new_string_enumerated_type_button_group_radio_button_model(Some(""), None, group, None)
    }

    /// Java `RadioButton(String text, EnumeratedType enumeratedType)`.
    pub fn new_string_enumerated_type(
        text: Option<&str>,
        enumerated_type: Option<EnumeratedTypeRef>,
    ) -> Rc<RadioButton> {
        Self::new_string_enumerated_type_button_group_radio_button_model(
            text,
            enumerated_type,
            None,
            None,
        )
    }

    /// Java `RadioButton(String text, EnumeratedType enumeratedType, ButtonGroup group)`.
    pub fn new_string_enumerated_type_button_group(
        text: Option<&str>,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<&Rc<ButtonGroup>>,
    ) -> Rc<RadioButton> {
        Self::new_string_enumerated_type_button_group_radio_button_model(
            text,
            enumerated_type,
            group,
            None,
        )
    }

    /// The field values every constructor starts from.
    fn construct(
        this: &Weak<RadioButton>,
        radio_button: Rc<JComponent>,
        field_lock_controller: Rc<FieldLockController>,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<Rc<ButtonGroup>>,
    ) -> RadioButton {
        RadioButton {
            this: this.clone(),
            radio_button,
            enumerated_type,
            group,
            field_lock_controller,
            self_item_listener: RefCell::new(None),
            debug: Cell::new(false),
            orig_foreground: Cell::new(None),
            directive_def: Cell::new(None),
            backup: RefCell::new(None),
            checkpoint: RefCell::new(None),
            default_value: RefCell::new(None),
            field_highlight: RefCell::new(None),
            selected_string_value: RefCell::new(None),
            unformatted_tooltip: RefCell::new(None),
        }
    }

    /// Java `RadioButton(String text, EnumeratedType enumeratedType, ButtonGroup group,
    /// RadioButtonModel model)`.
    pub fn new_string_enumerated_type_button_group_radio_button_model(
        text: Option<&str>,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<&Rc<ButtonGroup>>,
        model: Option<Rc<RadioButtonModel>>,
    ) -> Rc<RadioButton> {
        let mut text = text.map(str::to_owned);
        if text.is_none() {
            if let Some(enumerated_type) = &enumerated_type {
                text = enumerated_type.get_label();
            }
        }
        let instance = Rc::new_cyclic(|this: &Weak<RadioButton>| {
            let radio_button = JComponent::new_radio_button(text.as_deref().unwrap_or(""));
            let field_lock_controller = FieldLockController::get_toggle_button_instance_j_toggle_button(&radio_button);
            // Swing focus: radioButton.setFocusable(false) (focus is not modelled).
            let model = match model {
                Some(model) => model,
                None => {
                    let button: Weak<dyn RadioButtonInterface> = this.clone();
                    RadioButtonModel::new(Some(button))
                }
            };
            radio_button.set_model(Some(model as Rc<dyn ButtonModel>));
            RadioButton::construct(
                this,
                radio_button,
                field_lock_controller,
                enumerated_type.clone(),
                group.cloned(),
            )
        });
        instance.set_name(text.as_deref());
        // this.enumeratedType = enumeratedType (set in construct).
        if let Some(group) = group {
            group.add(&instance.radio_button);
        }
        if let Some(enumerated_type) = &instance.enumerated_type {
            // An enum-constructed radio button selects itself when its type is the
            // default (RadioButton.java:114-116 and the three copies).
            if enumerated_type.is_default() {
                instance.radio_button.set_selected(true);
            }
            // Java: `if (number != null)`; the Rust `getValue` always returns one.
            let number: ConstEtomoNumber = enumerated_type.get_value();
            *instance.selected_string_value.borrow_mut() = Some(number.to_string());
        }
        instance
    }

    /// Java `RadioButton(String text, String tflabel, EnumeratedType enumeratedType,
    /// ButtonGroup group, RadioButtonModel model)`.
    pub fn new_string_string_enumerated_type_button_group_radio_button_model(
        text: Option<&str>,
        tflabel: Option<&str>,
        enumerated_type: Option<EnumeratedTypeRef>,
        group: Option<&Rc<ButtonGroup>>,
        model: Option<Rc<RadioButtonModel>>,
    ) -> Rc<RadioButton> {
        let mut text = text.map(str::to_owned);
        if text.is_none() {
            if let Some(enumerated_type) = &enumerated_type {
                text = enumerated_type.get_label();
            }
        }
        let instance = Rc::new_cyclic(|this: &Weak<RadioButton>| {
            let radio_button = JComponent::new_radio_button(text.as_deref().unwrap_or(""));
            let field_lock_controller = FieldLockController::get_toggle_button_instance_j_toggle_button(&radio_button);
            // Swing focus: radioButton.setFocusable(false) (focus is not modelled).
            let model = match model {
                Some(model) => model,
                None => {
                    let button: Weak<dyn RadioButtonInterface> = this.clone();
                    RadioButtonModel::new(Some(button))
                }
            };
            radio_button.set_model(Some(model as Rc<dyn ButtonModel>));
            RadioButton::construct(
                this,
                radio_button,
                field_lock_controller,
                enumerated_type.clone(),
                group.cloned(),
            )
        });
        instance.set_name(tflabel);
        // this.enumeratedType = enumeratedType (set in construct).
        if let Some(group) = group {
            group.add(&instance.radio_button);
        }
        if let Some(enumerated_type) = &instance.enumerated_type {
            // An enum-constructed radio button selects itself when its type is the
            // default (RadioButton.java:114-116 and the three copies).
            if enumerated_type.is_default() {
                instance.radio_button.set_selected(true);
            }
            // Java: `if (number != null)`; the Rust `getValue` always returns one.
            let number: ConstEtomoNumber = enumerated_type.get_value();
            *instance.selected_string_value.borrow_mut() = Some(number.to_string());
        }
        instance
    }

    /// Java `RadioButton(EnumeratedType enumeratedType, ButtonGroup group)`.
    pub fn new_enumerated_type_button_group(
        enumerated_type: EnumeratedTypeRef,
        group: Option<&Rc<ButtonGroup>>,
    ) -> Rc<RadioButton> {
        // Java string conversion of a null label gives "null".
        let text = enumerated_type
            .get_label()
            .unwrap_or_else(|| "null".to_owned());
        let instance = Rc::new_cyclic(|this: &Weak<RadioButton>| {
            let radio_button = JComponent::new_radio_button(&text);
            let field_lock_controller = FieldLockController::get_toggle_button_instance_j_toggle_button(&radio_button);
            let button: Weak<dyn RadioButtonInterface> = this.clone();
            radio_button.set_model(Some(RadioButtonModel::new(Some(button)) as Rc<dyn ButtonModel>));
            RadioButton::construct(
                this,
                radio_button,
                field_lock_controller,
                Some(enumerated_type.clone()),
                group.cloned(),
            )
        });
        instance.set_name(Some(&text));
        if let Some(group) = group {
            group.add(&instance.radio_button);
        }
        if let Some(enumerated_type) = &instance.enumerated_type {
            // An enum-constructed radio button selects itself when its type is the
            // default (RadioButton.java:114-116 and the three copies).
            if enumerated_type.is_default() {
                instance.radio_button.set_selected(true);
            }
            // Java: `if (number != null)`; the Rust `getValue` always returns one.
            let number: ConstEtomoNumber = enumerated_type.get_value();
            *instance.selected_string_value.borrow_mut() = Some(number.to_string());
        }
        instance
    }

    /// Java `RadioButton(EnumeratedType enumeratedType, ButtonGroup group,
    /// String addToLabel)`.
    pub fn new_enumerated_type_button_group_string(
        enumerated_type: EnumeratedTypeRef,
        group: Option<&Rc<ButtonGroup>>,
        add_to_label: Option<&str>,
    ) -> Rc<RadioButton> {
        let text = enumerated_type
            .get_label()
            .unwrap_or_else(|| "null".to_owned())
            + add_to_label.unwrap_or("");
        let instance = Rc::new_cyclic(|this: &Weak<RadioButton>| {
            let radio_button = JComponent::new_radio_button(&text);
            let field_lock_controller = FieldLockController::get_toggle_button_instance_j_toggle_button(&radio_button);
            let button: Weak<dyn RadioButtonInterface> = this.clone();
            radio_button.set_model(Some(RadioButtonModel::new(Some(button)) as Rc<dyn ButtonModel>));
            RadioButton::construct(
                this,
                radio_button,
                field_lock_controller,
                Some(enumerated_type.clone()),
                group.cloned(),
            )
        });
        instance.set_name(Some(&text));
        if let Some(group) = group {
            group.add(&instance.radio_button);
        }
        if let Some(enumerated_type) = &instance.enumerated_type {
            // An enum-constructed radio button selects itself when its type is the
            // default (RadioButton.java:114-116 and the three copies).
            if enumerated_type.is_default() {
                instance.radio_button.set_selected(true);
            }
            // Java: `if (number != null)`; the Rust `getValue` always returns one.
            let number: ConstEtomoNumber = enumerated_type.get_value();
            *instance.selected_string_value.borrow_mut() = Some(number.to_string());
        }
        instance
    }

    /// Java `getField()`.
    pub fn get_field(&self) -> Option<Rc<dyn Field>> {
        self.this.upgrade().map(|this| this as Rc<dyn Field>)
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
        self.radio_button.is_visible()
    }

    /// Java `equalsSelectedStringValue(String)`.
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        match self.selected_string_value.borrow().as_deref() {
            None => value.is_some_and(|value| !value.is_empty()),
            Some(selected_string_value) => Some(selected_string_value) == value,
        }
    }

    /// Java `getCheckpoint()`.  (A copy: the live setting cannot be lent out of
    /// its cell.)
    pub fn get_checkpoint(&self) -> Option<Box<dyn FieldSettingInterface>> {
        self.checkpoint
            .borrow()
            .as_ref()
            .map(|checkpoint| Box::new(checkpoint.clone()) as Box<dyn FieldSettingInterface>)
    }

    /// Java `doClick()`.
    pub fn do_click(&self) {
        self.radio_button.do_click();
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
    /// value the displayed value if possible, and turn off the back up.  Its
    /// impossible to turn off a radio button, so this only works if the
    /// backupValue is true.  This relies on the other radio buttons in the group
    /// also being backed up.
    pub fn restore_from_backup(&self) {
        let value = match self.backup.borrow().as_ref() {
            Some(backup) if backup.is_set() => backup.is_value(),
            _ => return,
        };
        self.set_selected_boolean(value);
        if let Some(backup) = self.backup.borrow_mut().as_mut() {
            backup.reset();
        }
    }

    /// Java `setValue(Field)`.
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

    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&self, value: bool) {
        self.set_selected_boolean(value);
    }

    /// Java `clear()`.  No way to clear a radio button.
    pub fn clear(&self) {}

    /// Java `setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
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
        // only search for default value once
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
        self.default_value.borrow().as_ref().is_some_and(|default_value| {
            default_value.is_set() && default_value.equals_boolean(selected)
        })
    }

    /// Java `equalsDefaultValue(String)`.
    pub fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        self.default_value.borrow().as_ref().is_some_and(|default_value| {
            default_value.is_set()
                && default_value.equals_boolean(
                    value.is_some_and(|value| !java_lang_string_matches_whitespace(value)),
                )
        })
    }

    /// Java `equalsDefaultValue(boolean)`.
    pub fn equals_default_value_boolean(&self, input: bool) -> bool {
        self.default_value.borrow().as_ref().is_some_and(|default_value| {
            default_value.is_set() && default_value.equals_boolean(input)
        })
    }

    /// Java `isCheckpointValue()`.
    pub fn is_checkpoint_value(&self) -> bool {
        match self.checkpoint.borrow().as_ref() {
            None => false,
            Some(checkpoint) => checkpoint.is_value(),
        }
    }

    /// Java `isDifferentFromCheckpoint(boolean alwaysCheck)`: check for difference
    /// even when the field is disabled or invisible.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.radio_button.is_visible()) {
            return false;
        }
        let selected = self.is_selected();
        self.checkpoint
            .borrow()
            .as_ref()
            .is_none_or(|checkpoint| !checkpoint.equals_boolean(selected))
    }

    /// Java `setText(String)`.
    pub fn set_text(&self, text: Option<&str>) {
        self.radio_button.set_text(text.unwrap_or(""));
        self.set_name(text);
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<String> {
        self.get_quoted_label()
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(self.get_text_void().as_deref())
    }

    /// Java `setBorderPainted(boolean)`.
    pub fn set_border_painted(&self, b: bool) {
        // Swing painting: radioButton.setBorderPainted(b).
        let _ = b;
    }

    // Java `setBorder(Border)`, `getFont()`, `getWidth()`, `getHeight()`,
    // `getBorder()` and `setPreferredSize(Dimension)`: Swing painting and layout
    // accessors of the JRadioButton (borders, fonts, sizes are not modelled by
    // the jdk stand-in), so they have no Rust counterpart.

    /// Java `setForeground(Color)`.
    pub fn set_foreground(&self, fg: Option<Rgb>) {
        self.radio_button.set_foreground(fg);
    }

    /// Java `setName(String reference)`.
    pub fn set_name(&self, reference: Option<&str>) {
        let field_type = &ui_test_field_type::RADIO_BUTTON;
        let name = utilities::convert_label_to_name(reference, field_type.is_unlimited_segments());
        if let Some(name) = name {
            self.radio_button
                .set_name(Some(&format!("{}{}{}", field_type, SEPARATOR_CHAR, name)));
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {} ",
                    self.radio_button.get_name().as_deref().unwrap_or("null"),
                    DEFAULT_DELIMITER
                );
            }
        }
    }

    /// Java `equals(EnumeratedType)`: reference identity of the enumerated type.
    pub fn equals(&self, enumerated_type: Option<&EnumeratedTypeRef>) -> bool {
        self.enumerated_type.as_ref() == enumerated_type
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java `isFieldHighlightSet()`.
    pub fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.is_set())
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
            self.add_field_highlight_action_listeners();
        }
        let exists = match self.field_highlight.borrow_mut().as_mut() {
            Some(field_highlight) => {
                field_highlight.copy(input);
                true
            }
            None => false,
        };
        if exists {
            self.update_field_highlight(self.is_selected());
        }
    }

    /// The Java object as an `ItemListener`.
    fn get_self_item_listener(&self) -> ItemListener {
        if let Some(listener) = self.self_item_listener.borrow().as_ref() {
            return listener.clone();
        }
        let this = self.this.clone();
        let listener: ItemListener = Rc::new(move |event: &ItemEvent| {
            if let Some(this) = this.upgrade() {
                this.item_state_changed(event);
            }
        });
        *self.self_item_listener.borrow_mut() = Some(listener.clone());
        listener
    }

    /// Java `addFieldHighlightActionListeners()`.
    fn add_field_highlight_action_listeners(&self) {
        // Radio buttons turn off when another button in the group is turned on. So
        // listen to all of the radio buttons in the group.
        let mut listener_added = false;
        if let Some(group) = &self.group {
            for element in group.get_elements() {
                listener_added = true;
                // enumeration.nextElement().addActionListener(this);
                element.add_item_listener(self.get_self_item_listener());
            }
        }
        if !listener_added {
            // radioButton.addActionListener(this);
            self.radio_button.add_item_listener(self.get_self_item_listener());
        }
    }

    /// Java `setFieldHighlight(boolean)`.
    pub fn set_field_highlight_boolean(&self, value: bool) {
        if self.field_highlight.borrow().is_none() {
            *self.field_highlight.borrow_mut() = Some(BooleanFieldSetting::new());
            self.add_field_highlight_action_listeners();
        }
        self.field_highlight
            .borrow_mut()
            .as_mut()
            .unwrap()
            .set_boolean(value);
        self.update_field_highlight(self.is_selected());
    }

    /// Java `setFieldHighlight(String)`.
    pub fn set_field_highlight_string(&self, value: Option<&str>) {
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
            // Turn off field highlight - parameter doesn't matter since field
            // highlight is off.
            self.update_field_highlight(false);
        }
    }

    /// Java `getFieldHighlight()`.  (A copy; see `get_checkpoint`.)
    pub fn get_field_highlight(&self) -> Option<Box<dyn FieldSettingInterface>> {
        self.field_highlight
            .borrow()
            .as_ref()
            .map(|setting| Box::new(setting.clone()) as Box<dyn FieldSettingInterface>)
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
            field_highlight.is_set()
                && field_highlight.equals_boolean(
                    value.is_some_and(|value| !java_lang_string_matches_whitespace(value)),
                )
        })
    }

    /// Java `equalsFieldHighlight(boolean)`.
    pub fn equals_field_highlight_boolean(&self, input: bool) -> bool {
        self.field_highlight.borrow().as_ref().is_some_and(|field_highlight| {
            field_highlight.is_set() && field_highlight.equals_boolean(input)
        })
    }

    /// Java `itemStateChanged(ItemEvent)`.
    pub fn item_state_changed(&self, item_event: &ItemEvent) {
        let _ = item_event;
        let mut selected;
        // Radio buttons cannot be turned off directly. When another button in the
        // group is clicked, this button will turn off if it was on. This response
        // doesn't happen instantly, but its accurate to assume that this button is
        // off when another button was clicked.
        // if (!event.getActionCommand().equals(radioButton.getActionCommand())) {
        selected = false;
        // }
        // else {
        selected = self.is_selected();
        // }
        self.update_field_highlight(selected);
    }

    /// Java `updateFieldHighlight(boolean isSelected)`.
    pub fn update_field_highlight(&self, is_selected: bool) {
        let matches = self.field_highlight.borrow().as_ref().is_some_and(|field_highlight| {
            field_highlight.is_set() && field_highlight.is_value() == is_selected
        });
        if matches {
            if self.orig_foreground.get().is_none() {
                self.orig_foreground
                    .set(Some(self.radio_button.get_foreground()));
            }
            self.radio_button.set_foreground(Some(colors::FIELD_HIGHLIGHT));
            return;
        }
        if let Some(orig_foreground) = self.orig_foreground.get() {
            self.radio_button.set_foreground(orig_foreground);
        }
    }

    /// Java `setToolTipText(String autodocName, ReadOnlySection section)`.  Sets a
    /// tooltip from a section using the enumeratedType, if it exists.
    pub fn set_tool_tip_text_string_read_only_section(
        &self,
        autodoc_name: Option<&str>,
        section: &dyn ReadOnlySection,
    ) {
        let text = match &self.enumerated_type {
            None => etomo_autodoc::get_tooltip_add_source(autodoc_name, section, true),
            Some(enumerated_type) => etomo_autodoc::get_tooltip_enum_value_name(
                autodoc_name,
                section,
                Some(&enumerated_type.to_string()),
            ),
        };
        self.set_tool_tip_text_string(text.as_deref());
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text_string(&self, text: Option<&str>) {
        self.radio_button
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
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
            self.radio_button.set_tool_tip_text(field.get_tooltip().as_deref());
        }
    }

    /// Java `getTooltip()`.
    pub fn get_tooltip(&self) -> Option<String> {
        self.radio_button.get_tool_tip_text()
    }

    /// Java `setPreformattedTooltip(String)`.
    pub fn set_preformatted_tooltip(&self, tooltip: Option<&str>) {
        self.radio_button.set_tool_tip_text(tooltip);
    }

    /// Java `addTooltip(String)`.
    pub fn add_tooltip(&self, text: Option<&str>) {
        if text.is_none() {
            return;
        }
        let tooltip = self.radio_button.get_tool_tip_text();
        match tooltip {
            None => self.set_tool_tip_text_string(text),
            Some(tooltip) => self.radio_button.set_tool_tip_text(Some(&format!(
                "{} & {}",
                tooltip,
                tooltip_formatter::INSTANCE
                    .format(text)
                    .as_deref()
                    .unwrap_or("null")
            ))),
        }
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.radio_button.set_visible(visible);
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.radio_button.add_action_listener(action_listener);
    }

    /// Java `removeActionListener(ActionListener)`.
    pub fn remove_action_listener(&self, action_listener: &ActionListener) {
        self.radio_button.remove_action_listener(action_listener);
    }

    /// Java `addChangeListener(ChangeListener)`.
    pub fn add_change_listener(&self, listener: ChangeListener) {
        self.radio_button.add_change_listener(listener);
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected_boolean(&self, selected: bool) {
        self.radio_button.set_selected(selected);
        if self.field_highlight.borrow().is_some() {
            self.update_field_highlight(self.is_selected());
        }
    }

    /// Java `setSelected(ConstEtomoNumber, boolean allowEmpty)`.
    pub fn set_selected_const_etomo_number_boolean(
        &self,
        selected: Option<&ConstEtomoNumber>,
        allow_empty: bool,
    ) {
        if allow_empty || selected.is_some_and(|selected| !selected.is_null()) {
            // Upstream bug fixed (RadioButton.java:664-668): with allowEmpty true and
            // a null number the Java calls `selected.is()` and throws a
            // NullPointerException.  A null number is left alone here.
            if let Some(selected) = selected {
                self.set_selected_boolean(selected.is());
            }
        }
    }

    /// Java `msgSelected()`.
    pub fn msg_selected(&self) {}

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Java `getAbstractButton()`.
    pub fn get_abstract_button(&self) -> Rc<JComponent> {
        self.radio_button.clone()
    }

    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        false
    }

    /// Java `getText(boolean doValidation, FieldDisplayer)`.  Returns button label.
    /// No validation available.
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
        Ok(Some(self.radio_button.get_text()))
    }

    /// Java `getText()`.
    pub fn get_text_void(&self) -> Option<String> {
        Some(self.radio_button.get_text())
    }

    /// Java `setModel(ButtonModel)`.
    pub fn set_model(&self, new_model: Option<Rc<dyn ButtonModel>>) {
        self.radio_button.set_model(new_model);
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.radio_button.get_name()
    }

    /// Java `getEnumeratedType()`.
    pub fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef> {
        self.enumerated_type.clone()
    }

    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> Option<Rc<dyn SwingComponent>> {
        self.this.upgrade().map(|this| this as Rc<dyn SwingComponent>)
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.radio_button.clone()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
        if self.is_enabled() && self.is_editable() && !self.is_locked() {
            self.update_field_highlight(self.is_selected());
        }
    }

    /// Java `setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        self.field_lock_controller.set_locked(locked);
        if self.is_enabled() && self.is_editable() && !self.is_locked() {
            self.update_field_highlight(self.is_selected());
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.field_lock_controller.set_editable(editable);
        if self.is_enabled() && self.is_editable() && !self.is_locked() {
            self.update_field_highlight(self.is_selected());
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.radio_button.get_action_command()
    }

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, alignment_x: f32) {
        // Swing layout: radioButton.setAlignmentX(alignmentX).
        let _ = alignment_x;
    }

    /// Java `getSelectedObjects()` (`AbstractButton.getSelectedObjects`: the
    /// label when selected, else null).
    pub fn get_selected_objects(&self) -> Option<Vec<String>> {
        if !self.radio_button.is_selected() {
            return None;
        }
        Some(vec![self.radio_button.get_text()])
    }
}

/// Java `toString()`.
impl fmt::Display for RadioButton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}",
            self.radio_button.get_text(),
            if self.radio_button.is_selected() {
                "On"
            } else {
                "Off"
            }
        )
    }
}

/// Java `RadioButton.RadioButtonModel`
/// (`public static final class RadioButtonModel extends AbstractRadioButtonModel`).
pub struct RadioButtonModel {
    /// Java `radioButton`.  Weak: the button owns the component that owns this
    /// model.
    radio_button: Option<Weak<dyn RadioButtonInterface>>,
}

impl RadioButtonModel {
    /// Java `RadioButtonModel(RadioButtonInterface)`.
    pub fn new(radio_button: Option<Weak<dyn RadioButtonInterface>>) -> Rc<RadioButtonModel> {
        // super();
        Rc::new(RadioButtonModel { radio_button })
    }

    /// Java `getButton()`.
    pub fn get_button(&self) -> Option<Rc<dyn RadioButtonInterface>> {
        self.radio_button.as_ref().and_then(Weak::upgrade)
    }

    /// Java `getField()`.
    pub fn get_field(&self) -> Option<Rc<dyn Field>> {
        // Java dereferences `radioButton` unguarded; a model without a (live)
        // button answers null.
        self.get_button().and_then(|radio_button| radio_button.get_field())
    }
}

impl ButtonModel for RadioButtonModel {
    /// Java `setSelected(boolean)`: `super.setSelected(selected)` has been done by
    /// the component; then notify the button.
    fn set_selected(&self, selected: bool) {
        let _ = selected;
        if let Some(radio_button) = self.get_button() {
            radio_button.msg_selected();
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl AbstractRadioButtonModel for RadioButtonModel {
    /// Java `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef> {
        // Java dereferences `radioButton` unguarded; see `get_field`.
        self.get_button()
            .and_then(|radio_button| radio_button.get_enumerated_type())
    }
}

// ---- interface bindings (each forwards to the method above) ----

impl RadioButtonInterface for RadioButton {
    fn msg_selected(&self) {
        RadioButton::msg_selected(self)
    }
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef> {
        RadioButton::get_enumerated_type(self)
    }
    fn is_enabled(&self) -> bool {
        RadioButton::is_enabled(self)
    }
    fn get_field(&self) -> Option<Rc<dyn Field>> {
        RadioButton::get_field(self)
    }
}

impl Field for RadioButton {
    fn is_debug(&self) -> bool {
        RadioButton::is_debug(self)
    }
    fn get_name(&self) -> Option<String> {
        RadioButton::get_name(self)
    }
    fn is_boolean(&self) -> bool {
        RadioButton::is_boolean(self)
    }
    fn is_text(&self) -> bool {
        RadioButton::is_text(self)
    }
    fn get_quoted_label(&self) -> Option<String> {
        RadioButton::get_quoted_label(self)
    }
    fn is_enabled(&self) -> bool {
        RadioButton::is_enabled(self)
    }
    fn clear(&self) {
        RadioButton::clear(self)
    }
    fn set_value_field(&self, from: Option<&dyn Field>) {
        RadioButton::set_value_field(self, from)
    }
    fn set_value_string(&self, text: Option<&str>) {
        RadioButton::set_value_string(self, text)
    }
    fn set_value_boolean(&self, bool_: bool) {
        RadioButton::set_value_boolean(self, bool_)
    }
    fn is_empty(&self) -> bool {
        RadioButton::is_empty(self)
    }
    fn is_selected(&self) -> bool {
        RadioButton::is_selected(self)
    }
    fn is_required(&self) -> bool {
        RadioButton::is_required(self)
    }
    fn get_text_void(&self) -> Option<String> {
        RadioButton::get_text_void(self)
    }
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        RadioButton::get_text_boolean_field_displayer(self, do_validation, field_displayer1.as_deref())
    }
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        RadioButton::get_text_boolean_field_displayer_field_displayer(
            self,
            do_validation,
            field_displayer1.as_deref(),
            field_displayer2.as_deref(),
        )
    }
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        RadioButton::get_directive_def(self)
    }
    fn use_default_value(&self) {
        RadioButton::use_default_value(self)
    }
    fn equals_default_value_void(&self) -> bool {
        RadioButton::equals_default_value_void(self)
    }
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        RadioButton::equals_default_value_string(self, value)
    }
    fn backup(&self) {
        RadioButton::backup(self)
    }
    fn restore_from_backup(&self) {
        RadioButton::restore_from_backup(self)
    }
    fn checkpoint(&self) {
        RadioButton::checkpoint_void(self)
    }
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        RadioButton::set_checkpoint(self, input)
    }
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        RadioButton::get_checkpoint(self)
            .map(Rc::from)
    }
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        RadioButton::is_different_from_checkpoint(self, always_check)
    }
    fn is_field_highlight_set(&self) -> bool {
        RadioButton::is_field_highlight_set(self)
    }
    fn clear_field_highlight(&self) {
        RadioButton::clear_field_highlight(self)
    }
    fn set_field_highlight_field_setting_interface(&self, input: Option<&dyn FieldSettingInterface>) {
        RadioButton::set_field_highlight_field_setting_interface(self, input)
    }
    fn set_field_highlight_string(&self, input: Option<&str>) {
        RadioButton::set_field_highlight_string(self, input)
    }
    fn set_field_highlight_boolean(&self, input: bool) {
        RadioButton::set_field_highlight_boolean(self, input)
    }
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        RadioButton::get_field_highlight(self)
            .map(Rc::from)
    }
    fn equals_field_highlight_void(&self) -> bool {
        RadioButton::equals_field_highlight_void(self)
    }
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        RadioButton::equals_field_highlight_string(self, value)
    }
    fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        RadioButton::set_tool_tip_text_string(self, tooltip)
    }
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        RadioButton::set_tooltip(self, field)
    }
    fn get_tooltip(&self) -> Option<String> {
        RadioButton::get_tooltip(self)
    }
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        RadioButton::equals_selected_string_value(self, value)
    }
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        RadioButton::set_directive_def(self, directive_def)
    }
    fn get_description(&self) -> String {
        RadioButton::get_description(self)
            .unwrap_or_else(|| "null".to_owned())
    }
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        RadioButton::set_unformatted_tooltip(self, text)
    }
    fn has_unformatted_tooltip(&self) -> bool {
        RadioButton::has_unformatted_tooltip(self)
    }
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        RadioButton::use_unformatted_tooltip(self, param_descr, directive_descr)
    }
}

impl BooleanFieldInterface for RadioButton {}

impl UIComponent for RadioButton {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        RadioButton::get_component(self)
    }
}

impl SwingComponent for RadioButton {
    fn get_component(&self) -> Rc<JComponent> {
        RadioButton::get_component(self)
    }
}

impl ButtonComponent for RadioButton {
    fn add_action_listener(&self, listener: ActionListener) {
        RadioButton::add_action_listener(self, listener)
    }
    fn is_selected(&self) -> bool {
        RadioButton::is_selected(self)
    }
    fn get_action_command(&self) -> Option<String> {
        RadioButton::get_action_command(self)
    }
    fn is_enabled(&self) -> bool {
        RadioButton::is_enabled(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::mirror_in_x::MirrorInX;

    #[test]
    fn enum_constructed_default_selects_itself() {
        let group = ButtonGroup::new();
        let always = RadioButton::new_enumerated_type_button_group(
            EnumeratedTypeRef::new(MirrorInX::ALWAYS),
            Some(&group),
        );
        let default = RadioButton::new_enumerated_type_button_group(
            EnumeratedTypeRef::new(MirrorInX::DEFAULT),
            Some(&group),
        );
        assert!(!always.is_selected());
        assert!(default.is_selected());
        assert!(default.equals(Some(&EnumeratedTypeRef::new(MirrorInX::ASSESS_BOTH))));
        assert!(default.get_name().is_some_and(|name| name.starts_with("rb.")));
    }
}
