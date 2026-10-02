//! `IMOD/Etomo/src/etomo/ui/swing/RadioEbutton.java`.
//!
//! Extensible radio button: an optional enumerated type (whose label wins
//! over the given label, and whose default constant starts selected), a
//! checkpoint (`BooleanStateExtension`), and a warning flag
//! (`BooleanFlagExtension`) shown through `ComponentStyleExtension`.
//!
//! Backgrounds are painting and not modelled by `jdk.rs`: `defaultBackground`
//! keeps Java's fallback value and is only handed on to
//! `ComponentStyleExtension`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::radio_button_interface::EnumeratedTypeRef;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::boolean_efield_interface::BooleanEfieldInterface;
use crate::imod::etomo::ui::boolean_flag_extension::BooleanFlagExtension;
use crate::imod::etomo::ui::boolean_flag_origin::BooleanFlagOrigin;
use crate::imod::etomo::ui::boolean_state_extension::BooleanStateExtension;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::util::utilities;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::component_style_extension::{self, ComponentStyleExtension};
use super::etomo_button_group::EtomoButtonGroup;
use super::tooltip_formatter::{self, TooltipFormatter};

/// Java `Color.GRAY`.
const GRAY: (u8, u8, u8) = (128, 128, 128);

/// Java package-private final `RadioEbutton`.
pub struct RadioEbutton {
    /// This object, for the Java calls that pass `this`.
    self_ref: RefCell<Weak<RadioEbutton>>,
    /// Java final `button`.
    button: Rc<JComponent>,
    /// Java final `enumeratedType`.
    enumerated_type: Option<EnumeratedTypeRef>,
    /// Java `stateExtension`.
    state_extension: RefCell<Option<Rc<BooleanStateExtension>>>,
    /// Java `flagExtension`.
    flag_extension: RefCell<Option<Rc<BooleanFlagExtension>>>,
    /// Java `defaultBackground`.  Currently only using a flag that alters
    /// background color so not saving foreground color.
    default_background: Cell<Option<(u8, u8, u8)>>,
    /// Java `flagType`.
    flag_type: Cell<Option<&'static FlagType>>,
}

impl RadioEbutton {
    /// Java private
    /// `RadioEbutton(EnumeratedType, String, EtomoButtonGroup, ComponentStyleExtension)`.
    /// EnumeratedType takes precence over label - only when it contains a
    /// non-null label.  (`componentStyle` is unused in the Java body.)
    fn new(
        enumerated_type: Option<EnumeratedTypeRef>,
        label: Option<&str>,
        button_group: Option<&EtomoButtonGroup>,
        _component_style: Option<&ComponentStyleExtension>,
    ) -> Rc<RadioEbutton> {
        let mut label: Option<String> = label.map(str::to_owned);
        if let Some(enumerated_type) = &enumerated_type {
            let temp_label = enumerated_type.get_label();
            if temp_label.is_some() {
                label = temp_label;
            }
        }
        let button = match &label {
            Some(label) => JComponent::new_radio_button(label),
            None => JComponent::new_radio_button(""),
        };
        let instance = Rc::new(RadioEbutton {
            self_ref: RefCell::new(Weak::new()),
            button,
            enumerated_type,
            state_extension: RefCell::new(None),
            flag_extension: RefCell::new(None),
            default_background: Cell::new(None),
            flag_type: Cell::new(None),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        if label.is_some() {
            instance.set_name(label.as_deref());
        }
        // Java `button.setModel(new RadioEButtonModel(this))`: the stand-in
        // keeps selection on the component; the model answers
        // getEnumeratedType, for the button group and for the callers that
        // read `((AbstractRadioButtonModel) group.getSelection())`.
        let radio_model = Rc::new(RadioEButtonModel::new(Rc::downgrade(&instance)));
        instance.button.set_model(Some(
            radio_model.clone() as Rc<dyn crate::imod::etomo::jdk::ButtonModel>
        ));
        let model: Rc<dyn AbstractRadioButtonModel> = radio_model;
        if let Some(button_group) = button_group {
            button_group.add(&instance.button, Some(model));
        }
        if let Some(enumerated_type) = &instance.enumerated_type {
            if enumerated_type.is_default() {
                instance.button.set_selected(true);
            }
        }
        instance
    }

    /// This object as an `Rc` (Java `this` passed to another object).
    fn this(&self) -> Rc<RadioEbutton> {
        self.self_ref
            .borrow()
            .upgrade()
            .expect("RadioEbutton used before construction or after drop")
    }

    /// Java static `getEnumInstance(EnumeratedType, EtomoButtonGroup)`.
    pub fn get_enum_instance(
        enumerated_type: Option<EnumeratedTypeRef>,
        button_group: Option<&EtomoButtonGroup>,
    ) -> Rc<RadioEbutton> {
        RadioEbutton::new(enumerated_type, None, button_group, None)
    }

    /// Java static `getInstance(String, EtomoButtonGroup)`.
    pub fn get_instance(
        label: Option<&str>,
        button_group: Option<&EtomoButtonGroup>,
    ) -> Rc<RadioEbutton> {
        RadioEbutton::new(None, label, button_group, None)
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&self, label: Option<&str>) {
        // Java `button.setText(label)`; the stand-in holds no null text.
        self.button.set_text(label.unwrap_or(""));
        self.set_name(label);
    }

    /// Java private `setName(String)`.
    fn set_name(&self, label: Option<&str>) {
        let field_type = UITestFieldType::RADIO_BUTTON;
        let name = utilities::convert_label_to_name(label, field_type.is_unlimited_segments());
        if let Some(name) = name {
            self.button.set_name(Some(&format!(
                "{}{}{}",
                field_type.to_string(),
                SEPARATOR_CHAR,
                name
            )));
            // Java `EtomoDirector.INSTANCE.getArguments()` is the `ARGUMENTS` static.
            if ARGUMENTS.lock().unwrap().is_print_names() {
                println!(
                    "{} {} ",
                    self.button.get_name().as_deref().unwrap_or("null"),
                    DEFAULT_DELIMITER
                );
            }
        }
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.button.add_action_listener(listener);
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.clone()
    }

    /// Java `actionPerformed(ActionEvent)` (`implements ActionListener`).
    pub fn action_performed(&self, _event: &ActionEvent) {
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            flag_extension.update();
        }
    }

    /// Java private `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef> {
        self.enumerated_type.clone()
    }

    /// Java public `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.get_component().is_enabled()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.button.set_enabled(enabled);
        if self.flag_extension.borrow().is_some() {
            // A flag may have altered the field's appearance
            component_style_extension::INSTANCE.update_appearance(
                Some(&self.button),
                self.flag_type.get(),
                None,
                self.default_background.get(),
            );
        }
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.button.is_selected()
    }

    /// Java public `equals(boolean)`.
    pub fn equals(&self, value: bool) -> bool {
        self.button.is_selected() == value
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.button.set_selected(selected);
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            flag_extension.update();
        }
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> String {
        self.button.get_text()
    }

    /// Java final `checkpoint()`.
    pub fn checkpoint(&self) {
        if self.state_extension.borrow().is_none() {
            let field: Rc<dyn BooleanEfieldInterface> = self.this();
            *self.state_extension.borrow_mut() = Some(BooleanStateExtension::new(field));
        }
        let state_extension = self.state_extension.borrow().clone().unwrap();
        state_extension.checkpoint();
    }

    /// Java final `isCheckpointValue()`: the checkpoint value.
    pub fn is_checkpoint_value(&self) -> bool {
        let state_extension = self.state_extension.borrow().clone();
        let Some(state_extension) = state_extension else {
            return false;
        };
        state_extension.is_checkpoint_value()
    }

    /// Java final `enableWarning(boolean)`.
    pub fn enable_warning(&self, value: bool) {
        // Give the button a generic component style for displaying the warning.
        if self.default_background.get().is_none() {
            // Swing painting: defaultBackground = button.getBackground(),
            // which the stand-in does not model; Java falls back to
            // Color.GRAY when it is null.
            self.default_background.set(Some(GRAY));
        }
        // Use flag extension to set warning flag.
        if self.flag_extension.borrow().is_none() {
            // Java `button.addActionListener(this)`.
            let this = self.self_ref.borrow().clone();
            self.button.add_action_listener(Rc::new(move |event| {
                if let Some(this) = this.upgrade() {
                    this.action_performed(event);
                }
            }));
            let flag_origin: Rc<dyn BooleanFlagOrigin> = self.this();
            let flag_extension = BooleanFlagExtension::new(flag_origin);
            let flag_display: Rc<dyn FlagDisplay> = self.this();
            flag_extension.add_flag_display(Some(flag_display));
            *self.flag_extension.borrow_mut() = Some(flag_extension);
        }
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.enable_warning(value);
        flag_extension.update();
    }

    /// Java final `disableWarning()`.
    pub fn disable_warning(&self) {
        let flag_extension = self.flag_extension.borrow().clone();
        let Some(flag_extension) = flag_extension else {
            return;
        };
        flag_extension.disable_warning();
        flag_extension.update();
    }

    /// Java public final `setFlag(FlagType)`: update the appearance based on
    /// the flag.
    pub fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        component_style_extension::INSTANCE.update_appearance(
            Some(&self.button),
            flag_type,
            None,
            self.default_background.get(),
        );
        self.flag_type.set(flag_type);
    }

    /// Java `setFormattedTooltip(String)`.
    pub fn set_formatted_tooltip(&self, formatted_tooltip: Option<&str>) {
        self.button.set_tool_tip_text(formatted_tooltip);
    }

    /// Java final `setTooltip(String)`.
    pub fn set_tooltip_string(&self, text: Option<&str>) {
        self.button
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setTooltip(String, ReadOnlySection)`: sets a tooltip from a
    /// section using the enumeratedType, if it exists.
    pub fn set_tooltip_string_read_only_section(
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
        self.set_tooltip_string(text.as_deref());
    }
}

/// Java `implements BooleanEfieldInterface`.
impl BooleanEfieldInterface for RadioEbutton {
    fn is_selected(&self) -> bool {
        RadioEbutton::is_selected(self)
    }
    fn set_selected(&self, selected: bool) {
        RadioEbutton::set_selected(self, selected)
    }
}

/// Java `implements BooleanFlagOrigin`.
impl BooleanFlagOrigin for RadioEbutton {
    fn is_selected(&self) -> bool {
        RadioEbutton::is_selected(self)
    }
}

/// Java `implements FlagDisplay`.
impl FlagDisplay for RadioEbutton {
    fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        RadioEbutton::set_flag(self, flag_type)
    }
}

/// Java public static final nested class `RadioEbutton.RadioEButtonModel`
/// (`extends AbstractRadioButtonModel`).  It refers back to its radio button,
/// which owns it, so the reference is a `Weak`.
pub struct RadioEButtonModel {
    /// Java final `radioButton`.
    radio_button: Weak<RadioEbutton>,
}

impl RadioEButtonModel {
    /// Java private `RadioEButtonModel(RadioEbutton)`.
    fn new(radio_button: Weak<RadioEbutton>) -> RadioEButtonModel {
        RadioEButtonModel { radio_button }
    }
}

impl crate::imod::etomo::jdk::ButtonModel for RadioEButtonModel {
    /// Java inherits `ToggleButtonModel.setSelected`; nothing more happens here.
    fn set_selected(&self, _selected: bool) {}
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AbstractRadioButtonModel for RadioEButtonModel {
    /// Java `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef> {
        self.radio_button.upgrade()?.get_enumerated_type()
    }
}
