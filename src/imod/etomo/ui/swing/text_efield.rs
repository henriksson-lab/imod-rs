//! `IMOD/Etomo/src/etomo/ui/swing/TextEfield.java`: a self-naming text field
//! with optional label, control component, appearance, flag, value
//! manipulation, validation, state and grid-bag extensions.
//!
//! `class TextEfield implements TextEfieldInterface, UIComponent, SwingComponent,
//! TextFlagOrigin, ValueManipulationField, ControlTarget`.
//!
//! `TextEfield` is extended by `ButtonControlTextEfield`, which overrides
//! `setDebug`, `setTooltip`, `createAppearanceExtension` and
//! `createFlagExtension`.  The last two are called from inside this class, so
//! they dispatch through [`TextEfieldVirtual`], the trait every concrete text
//! Efield implements (with the Java bodies as its default methods, found here
//! as `default_*`).  The Java object passes itself to its extensions and to the
//! file-chooser buttons as a `ControlTarget`, `TextFlagOrigin`,
//! `ValueManipulationField`, `TextEfieldInterface` and `UIComponent`; `this`
//! is the outermost object as a `dyn TextEfieldVirtual`, which has all of those
//! as supertraits.  It is handed in by the outermost constructor, which builds
//! the object with `Rc::new_cyclic`, so it cannot be upgraded while the
//! constructor runs.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{FocusEvent, FocusListener, JComponent};
use crate::imod::etomo::logic::field_validator::FieldValidator;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_origin_listener::FlagOriginListener;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::ui::swing::control_component_module::ControlComponentModule;
use crate::imod::etomo::ui::swing::control_listener::ControlListener;
use crate::imod::etomo::ui::swing::control_state::ControlState;
use crate::imod::etomo::ui::swing::control_target::ControlTarget;
use crate::imod::etomo::ui::swing::efield_container::EfieldContainer;
use crate::imod::etomo::ui::swing::grid_bag_extension::GridBagExtension;
use crate::imod::etomo::ui::swing::swing_component::SwingComponent;
use crate::imod::etomo::ui::swing::text_component_appearance_extension::TextComponentAppearanceExtension;
use crate::imod::etomo::ui::swing::text_efield_interface::TextEfieldInterface;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::ui::swing::validation_extension::ValidationExtension;
use crate::imod::etomo::ui::swing::value_manipulation_extension::ValueManipulationExtension;
use crate::imod::etomo::ui::text_flag_extension::TextFlagExtension;
use crate::imod::etomo::ui::text_flag_origin::TextFlagOrigin;
use crate::imod::etomo::ui::text_state_extension::TextStateExtension;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::ui::value_manipulation_field::ValueManipulationField;
use crate::imod::etomo::ui::value_manipulation_listener::ValueManipulationListener;
use crate::imod::etomo::util::utilities;

/// The `TextEfield` methods a subclass overrides and `TextEfield` itself calls,
/// plus the interfaces the Java object passes itself as.  The default bodies
/// are `TextEfield`'s own.
pub trait TextEfieldVirtual:
    ControlTarget
    + TextFlagOrigin
    + ValueManipulationField
    + TextEfieldInterface
    + UIComponent
    + SwingComponent
{
    /// The `TextEfield` part of the object.
    fn text_efield(&self) -> &TextEfield;

    /// Java `setTooltip(String)`.
    fn set_tooltip(&self, text: Option<&str>) {
        self.text_efield().default_set_tooltip(text);
    }

    /// Java `createAppearanceExtension(boolean enabledField,
    /// boolean editableComponent)`.
    fn create_appearance_extension(&self, enabled_field: bool, editable_component: bool) {
        self.text_efield()
            .default_create_appearance_extension(enabled_field, editable_component);
    }

    /// Java `createFlagExtension()`.  Creates flagExtension if it is null;
    /// returns true if flagExtension was created.
    fn create_flag_extension(&self) -> bool {
        self.text_efield().default_create_flag_extension()
    }
}

/// Java `TextEfield`.
pub struct TextEfield {
    /// Java `this`: the outermost object (see the module documentation).
    this: Weak<dyn TextEfieldVirtual>,
    text_field: Rc<JComponent>,

    field_type: Option<FieldType>,
    label_text: Option<String>,
    control_component: Option<Rc<ControlComponentModule>>,
    /// Java package field `container`.
    pub(crate) container: EfieldContainer,
    label: Option<Rc<JComponent>>,

    directive_def: Cell<Option<DirectiveDef>>,
    state_extension: RefCell<Option<Rc<TextStateExtension>>>,
    grid_bag_extension: RefCell<Option<Rc<GridBagExtension>>>,
    /// Java package field `appearanceExtension`.
    pub(crate) appearance_extension: RefCell<Option<Rc<TextComponentAppearanceExtension>>>,
    flag_extension: RefCell<Option<Rc<TextFlagExtension>>>,
    value_manipulation_extension: RefCell<Option<Rc<ValueManipulationExtension>>>,
    validation_extension: RefCell<Option<Rc<ValidationExtension>>>,
    debug: Cell<bool>,
    enable_control_state: Cell<Option<&'static ControlState>>,
    control_listeners: RefCell<Option<Vec<Rc<dyn ControlListener>>>>,
}

impl TextEfield {
    /// Java `TextEfield(String labelText, FieldType fieldType, boolean useLabel,
    /// boolean useControlComponent, boolean enabledField, boolean editableComponent,
    /// boolean useGridBag, boolean defaultToFileName)`.
    ///
    /// `this` is the outermost object, from the caller's `Rc::new_cyclic`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        this: Weak<dyn TextEfieldVirtual>,
        label_text: Option<&str>,
        field_type: Option<FieldType>,
        use_label: bool,
        use_control_component: bool,
        enabled_field: bool,
        editable_component: bool,
        use_grid_bag: bool,
        default_to_file_name: bool,
    ) -> TextEfield {
        let text_field = JComponent::new_text_field();
        // Add a label and control component as needed.
        let label = if use_label {
            Some(JComponent::new_label(label_text.unwrap_or("")))
        } else {
            None
        };
        let control_component = if use_control_component {
            Some(ControlComponentModule::new())
        } else {
            None
        };
        // Java builds the container after setName() and createAppearanceExtension;
        // it depends only on the label, the text field and the control component,
        // so building it first (the Rust field is immutable) changes nothing.
        let container = EfieldContainer::new(
            false,
            use_grid_bag,
            label.as_ref(),
            &text_field,
            control_component
                .as_ref()
                .map(|control_component| control_component.get_component())
                .as_ref(),
        );
        let instance = TextEfield {
            this: this.clone(),
            text_field,
            field_type,
            label_text: label_text.map(str::to_owned),
            control_component,
            container,
            label,
            directive_def: Cell::new(None),
            state_extension: RefCell::new(None),
            grid_bag_extension: RefCell::new(None),
            appearance_extension: RefCell::new(None),
            flag_extension: RefCell::new(None),
            value_manipulation_extension: RefCell::new(None),
            validation_extension: RefCell::new(None),
            debug: Cell::new(false),
            enable_control_state: Cell::new(None),
            control_listeners: RefCell::new(None),
        };
        instance.set_name();
        // Use the appearance extension if controlling the appearance characteristics
        // will not be simple.
        if !enabled_field || !editable_component {
            // Java calls the (virtual) createAppearanceExtension here, before the
            // subclass constructor has run.  `this` cannot be upgraded yet, so the
            // TextEfield body runs; the only override (ButtonControlTextEfield)
            // additionally sets the child controllers, which are null at this point
            // and are set again by its constructor.
            instance.default_create_appearance_extension(enabled_field, editable_component);
        }
        if default_to_file_name {
            let field: Weak<dyn ValueManipulationField> = this;
            let value_manipulation_extension =
                ValueManipulationExtension::new(field, &instance, instance.debug.get());
            value_manipulation_extension.set_default_to_filename(default_to_file_name);
            *instance.value_manipulation_extension.borrow_mut() =
                Some(value_manipulation_extension);
        }
        instance
    }

    /// The outermost object; panics only if used during its own construction.
    fn this(&self) -> Rc<dyn TextEfieldVirtual> {
        self.this
            .upgrade()
            .expect("TextEfield used during construction or after it was dropped")
    }

    /// Java `getInstance(String labelText, FieldType)`.
    pub fn get_instance(label_text: Option<&str>, field_type: Option<FieldType>) -> Rc<TextEfield> {
        Rc::new_cyclic(|this: &Weak<TextEfield>| {
            TextEfield::new(
                this.clone(),
                label_text,
                field_type,
                false,
                false,
                true,
                true,
                false,
                false,
            )
        })
    }

    /// Java `getLabeledInstance(String labelText, FieldType)`.
    pub fn get_labeled_instance(
        label_text: Option<&str>,
        field_type: Option<FieldType>,
    ) -> Rc<TextEfield> {
        Rc::new_cyclic(|this: &Weak<TextEfield>| {
            TextEfield::new(
                this.clone(),
                label_text,
                field_type,
                true,
                false,
                true,
                true,
                false,
                false,
            )
        })
    }

    /// Java `getOverrideInstance(String labelText, DirectiveValueType)`.
    pub fn get_override_instance(
        label_text: Option<&str>,
        value_type: Option<DirectiveValueType>,
    ) -> Rc<TextEfield> {
        Rc::new_cyclic(|this: &Weak<TextEfield>| {
            TextEfield::new(
                this.clone(),
                label_text,
                FieldType::get_instance(value_type),
                false,
                true,
                true,
                true,
                false,
                false,
            )
        })
    }

    /// Java `getDisabledInstance(String labelText, FieldType)`.
    pub fn get_disabled_instance(
        label_text: Option<&str>,
        field_type: Option<FieldType>,
    ) -> Rc<TextEfield> {
        Rc::new_cyclic(|this: &Weak<TextEfield>| {
            TextEfield::new(
                this.clone(),
                label_text,
                field_type,
                false,
                false,
                false,
                true,
                false,
                false,
            )
        })
    }

    /// Java `setName()`.
    fn set_name(&self) {
        let field_type = &ui_test_field_type::TEXT_FIELD;
        let name = utilities::convert_label_to_name(
            self.label_text.as_deref(),
            field_type.is_unlimited_segments(),
        );
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
        if let Some(control_component) = &self.control_component {
            control_component.set_to_container_name(self.label_text.as_deref());
        }
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        self.label_text.clone()
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> Option<String> {
        self.label_text.clone()
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
        if let Some(appearance_extension) = self.appearance_extension.borrow().as_ref() {
            appearance_extension.set_debug(debug);
        }
    }

    /// Java `getUIComponent()` (final).
    pub fn get_ui_component(&self) -> Option<Rc<dyn SwingComponent>> {
        self.this
            .upgrade()
            .map(|this| this as Rc<dyn SwingComponent>)
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.container.get_component()
    }

    /// Java `setColumns()` (final).
    pub fn set_columns(&self) {
        if let Some(field_type) = self.field_type {
            // Swing layout: textField.setColumns(fieldType.getColumns()).
            let _ = field_type.get_columns();
        }
    }

    // Java (commented out in the source) `setPreferredWidth` via
    // UIParameters.adjustSize.

    /// Java `setPreferredWidth(int)`.
    pub fn set_preferred_width(&self, new_width: i32) {
        // Swing layout: textField.setPreferredSize(UIUtilities.calcNewTextFieldSize(
        // textField.getPreferredSize(), newWidth, true)).
        let _ = new_width;
    }

    // Java `getPreferredSize()`: `textField.getPreferredSize()` (Swing layout, not
    // modelled).

    /// Java `getFile()` (final).
    pub fn get_file(&self) -> Option<PathBuf> {
        let text = self.get_text_void();
        if let Some(text) = text {
            if !text.is_empty() {
                return Some(PathBuf::from(text));
            }
        }
        None
    }

    /// Java `getText()` (final).
    pub fn get_text_void(&self) -> Option<String> {
        if !self.is_override() {
            let text = self.text_field.get_text();
            if let Some(value_manipulation_extension) =
                self.value_manipulation_extension.borrow().as_ref()
            {
                return value_manipulation_extension.get_full_file_path(Some(&text));
            }
            return Some(text);
        }
        Some(String::new())
    }

    /// Java `isEmpty()` (final).
    pub fn is_empty(&self) -> bool {
        let text = self.get_text_void();
        text.as_deref()
            .is_some_and(java_lang_string_matches_whitespace)
    }

    /// Java `addFocusListener(FocusListener)` (final).
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.text_field.add_focus_listener(listener);
    }

    /// Java `removeFocusListener(FocusListener)` (final).
    pub fn remove_focus_listener(&self, listener: &FocusListener) {
        self.text_field.remove_focus_listener(listener);
    }

    /// Java `equals(String)` (final).
    pub fn equals(&self, compare_text: Option<&str>) -> bool {
        let text = self.get_text_void();
        match (text.as_deref(), compare_text) {
            (None, None) => true,
            (None, _) | (_, None) => false,
            (Some(text), Some(compare_text)) => {
                if text.is_empty() && compare_text.is_empty() {
                    return true;
                }
                text.trim() == compare_text.trim()
            }
        }
    }

    /// Java `setTooltip(String)` (virtual; see [`TextEfieldVirtual`]).
    pub fn set_tooltip(&self, text: Option<&str>) {
        match self.this.upgrade() {
            Some(this) => this.set_tooltip(text),
            None => self.default_set_tooltip(text),
        }
    }

    /// `TextEfield`'s own `setTooltip(String)` body.
    pub fn default_set_tooltip(&self, text: Option<&str>) {
        self.text_field
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
        if let Some(label) = &self.label {
            label.set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
        }
        if let Some(control_component) = &self.control_component {
            control_component.set_tooltip(text);
        }
    }

    /// Java `setVisible(boolean)` (final).
    pub fn set_visible(&self, visible: bool) {
        self.get_component().set_visible(visible);
    }

    /// Java `isVisible()` (final).
    pub fn is_visible(&self) -> bool {
        self.get_component().is_visible()
    }

    /// Java `setFile(String filePath)`.
    pub fn set_file(&self, file_path: Option<&str>) {
        match file_path {
            Some(file_path) if !java_lang_string_matches_whitespace(file_path) => {
                self.set_text_file(Some(Path::new(file_path)));
            }
            _ => self.clear(),
        }
    }

    /// Java `setText(File)` (final).
    pub fn set_text_file(&self, file: Option<&Path>) {
        let string: Option<String>;
        if let Some(value_manipulation_extension) =
            self.value_manipulation_extension.borrow().clone()
        {
            string = value_manipulation_extension.create_displayed_file_path_file(file);
        } else if let Some(file) = file {
            // `absolutePath != null ? absolutePath : file.getName()`: never null.
            let absolute_path = utilities::java_io_file_get_absolute_path(&file.to_string_lossy());
            string = Some(absolute_path);
        } else {
            string = None;
        }
        self.set_text_internal(string.as_deref());
    }

    /// Java `setText(int)` (final).
    pub fn set_text_int(&self, i_text: i32) {
        self.set_text_string(Some(&i_text.to_string()));
    }

    /// Java `setText(String)` (final).
    pub fn set_text_string(&self, string: Option<&str>) {
        let mut string = string.map(str::to_owned);
        if let Some(value_manipulation_extension) =
            self.value_manipulation_extension.borrow().clone()
        {
            string = value_manipulation_extension
                .create_displayed_file_path_string_field_type(string.as_deref(), self.field_type);
        }
        self.set_text_internal(string.as_deref());
    }

    /// Java `setTextInternal(String)` (private final).
    fn set_text_internal(&self, text: Option<&str>) {
        match text {
            None | Some("") => self.clear(),
            Some(text) => {
                self.text_field.set_text(text);
                self.update_flag_extension();
            }
        }
    }

    /// Java `getDirectiveDef()` (final).
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `setDirectiveDef(DirectiveDef)` (final).
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java `isTemplateValue()`.
    pub fn is_template_value(&self) -> bool {
        let flag_type = self.get_flag_type();
        flag_type.is_some_and(|flag_type| flag_type.is_template())
    }

    // / control

    /// Java `isOverride()`.
    pub fn is_override(&self) -> bool {
        match &self.control_component {
            None => false,
            Some(control_component) => control_component.is_override(),
        }
    }

    /// Java `setComponentControl(boolean, ControlState)` (final).
    pub fn set_component_control(
        &self,
        control: bool,
        control_state: Option<&'static ControlState>,
    ) {
        let Some(control_component) = &self.control_component else {
            return;
        };
        self.text_field
            .set_visible(!control_component.set_component_control(control, control_state));
    }

    /// Java `addControlListener(ControlListener)`.
    pub fn add_control_listener(&self, listener: Option<Rc<dyn ControlListener>>) {
        let Some(listener) = listener else {
            return;
        };
        let mut control_listeners = self.control_listeners.borrow_mut();
        if control_listeners.is_none() {
            *control_listeners = Some(Vec::new());
        }
        control_listeners.as_mut().unwrap().push(listener);
    }

    /// Java `sendControlEvent()`.
    pub fn send_control_event(&self) {
        let control_listeners = self.control_listeners.borrow().clone();
        if let Some(control_listeners) = control_listeners {
            for listener in control_listeners.iter() {
                listener.control_event();
            }
        }
    }

    /// Java `setEnableControl(boolean, ControlState)` (final).
    pub fn set_enable_control(
        &self,
        mut control: bool,
        control_state: Option<&'static ControlState>,
    ) {
        if control_state.is_none() {
            control = false;
        }
        if self.appearance_extension.borrow().is_none() && control {
            self.create_appearance_extension(true, true);
        }
        if control {
            self.enable_control_state.set(control_state);
        } else {
            self.enable_control_state.set(None);
        }
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_enable_control_state(self.enable_control_state.get());
        }
    }

    // / valueManipulationExtension

    /// Java `clear()` (final).
    pub fn clear(&self) {
        self.text_field.set_text("");
        let value_manipulation_extension = self.value_manipulation_extension.borrow().clone();
        if let Some(value_manipulation_extension) = value_manipulation_extension {
            value_manipulation_extension.clear_full_file_path();
            value_manipulation_extension.substitute();
        }
        self.update_flag_extension();
    }

    /// Java `setSubstituteText(String)` (final).
    pub fn set_substitute_text(&self, text: Option<&str>) {
        self.text_field.set_text(text.unwrap_or(""));
    }

    /// Java `addValueManipulationListener(ValueManipulationListener)` (final).
    pub fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>) {
        self.text_field
            .add_focus_listener(Rc::new(move |event: &FocusEvent| {
                if event.gained {
                    listener.focus_gained();
                } else {
                    listener.focus_lost();
                }
            }));
    }

    // / validation & ValidationExtension

    /// Java `setRequired(boolean)` (final).
    pub fn set_required(&self, required: bool) {
        if required && self.validation_extension.borrow().is_none() {
            *self.validation_extension.borrow_mut() = Some(ValidationExtension::new());
        }
        if let Some(validation_extension) = self.validation_extension.borrow().as_ref() {
            validation_extension.set_required(required);
        }
    }

    /// Java `setLocationDescr(String)` (final).
    pub fn set_location_descr(&self, location_descr: Option<&str>) {
        if location_descr.is_some() && self.validation_extension.borrow().is_none() {
            *self.validation_extension.borrow_mut() = Some(ValidationExtension::new());
        }
        if let Some(validation_extension) = self.validation_extension.borrow().as_ref() {
            validation_extension.set_location_descr(location_descr);
        }
    }

    /// Java `setFileMustExist(boolean)` (final).
    pub fn set_file_must_exist(&self, file_must_exist: bool) {
        if file_must_exist && self.validation_extension.borrow().is_none() {
            *self.validation_extension.borrow_mut() = Some(ValidationExtension::new());
        }
        if let Some(validation_extension) = self.validation_extension.borrow().as_ref() {
            validation_extension.set_file_must_exist(file_must_exist);
        }
    }

    /// Java `setFileOnly(boolean)` (final).
    pub fn set_file_only(&self, file_only: bool) {
        if file_only && self.validation_extension.borrow().is_none() {
            *self.validation_extension.borrow_mut() = Some(ValidationExtension::new());
        }
        if let Some(validation_extension) = self.validation_extension.borrow().as_ref() {
            validation_extension.set_file_only(file_only);
        }
    }

    /// Java `setMustBePositive(boolean)` (final).
    pub fn set_must_be_positive(&self, must_be_positive: bool) {
        if must_be_positive && self.validation_extension.borrow().is_none() {
            *self.validation_extension.borrow_mut() = Some(ValidationExtension::new());
        }
        if let Some(validation_extension) = self.validation_extension.borrow().as_ref() {
            validation_extension.set_must_be_positive(must_be_positive);
        }
    }

    /// Java `getText(boolean doValidation)` (final).
    pub fn get_text_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, None, None)
    }

    /// Java `getText(boolean doValidation, FieldDisplayer, FieldDisplayer)`
    /// (final).  For non standard validation only member variables
    /// fileNustExist, fileOnly, positiveNumberOnly.
    pub fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<&dyn FieldDisplayer>,
        field_displayer2: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut text = self.get_text_void();
        if !self.is_override() && do_validation && self.is_enabled() {
            let validation_extension = self.validation_extension.borrow().clone();
            // Java string concatenation: a null quoted label reads "null".
            let descr = format!(
                "{}{}",
                utilities::quote_label(self.label_text.as_deref())
                    .as_deref()
                    .unwrap_or("null"),
                match &validation_extension {
                    Some(validation_extension) => validation_extension.get_location_addon(),
                    None => String::new(),
                }
            );
            let this = self.this();
            text = FieldValidator::validate_text_string_field_type_ui_component_string_validation_extension_field_displayer_field_displayer_boolean(
                text.as_deref(),
                self.field_type,
                Some(&*this as &dyn UIComponent),
                Some(&descr),
                validation_extension.as_deref(),
                field_displayer1,
                field_displayer2,
                self.debug.get(),
            )?;
        }
        Ok(text)
    }

    /// Java `getText(boolean doValidation, FieldDisplayer)` (final).
    pub fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<&dyn FieldDisplayer>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut text = self.get_text_void();
        if !self.is_override() && do_validation && self.is_enabled() {
            let validation_extension = self.validation_extension.borrow().clone();
            // Java string concatenation: a null quoted label reads "null".
            let descr = format!(
                "{}{}",
                utilities::quote_label(self.label_text.as_deref())
                    .as_deref()
                    .unwrap_or("null"),
                match &validation_extension {
                    Some(validation_extension) => validation_extension.get_location_addon(),
                    None => String::new(),
                }
            );
            let this = self.this();
            text = FieldValidator::validate_text_string_field_type_ui_component_string_validation_extension_field_displayer_field_displayer_boolean(
                text.as_deref(),
                self.field_type,
                Some(&*this as &dyn UIComponent),
                Some(&descr),
                validation_extension.as_deref(),
                field_displayer1,
                None,
                self.debug.get(),
            )?;
        }
        Ok(text)
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        if self.is_override() {
            return true;
        }
        let validation_extension = self.validation_extension.borrow().clone();
        let this = self.this();
        FieldValidator::is_text_valid(
            self.get_text_void().as_deref(),
            self.field_type,
            Some(&*this as &dyn UIComponent),
            validation_extension.as_deref(),
            self.debug.get(),
        )
    }

    // / appearanceExtension

    /// Java `createAppearanceExtension(boolean, boolean)` (virtual; see
    /// [`TextEfieldVirtual`]).
    pub fn create_appearance_extension(&self, enabled_field: bool, editable_component: bool) {
        match self.this.upgrade() {
            Some(this) => this.create_appearance_extension(enabled_field, editable_component),
            None => self.default_create_appearance_extension(enabled_field, editable_component),
        }
    }

    /// `TextEfield`'s own `createAppearanceExtension(boolean, boolean)` body.
    pub fn default_create_appearance_extension(
        &self,
        enabled_field: bool,
        editable_component: bool,
    ) {
        if self.appearance_extension.borrow().is_none() {
            // Using the text component ability to set the field to be editable
            let appearance_extension = TextComponentAppearanceExtension::new(
                &self.text_field,
                enabled_field,
                editable_component,
            );
            *self.appearance_extension.borrow_mut() = Some(appearance_extension.clone());
            appearance_extension.set_debug(self.debug.get());
        }
        // In this class the appearanceExtension acts as the field's flag display for
        // all types of flags.
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            let appearance_extension = self.appearance_extension.borrow().clone();
            flag_extension.add_flag_display(
                appearance_extension
                    .map(|appearance_extension| appearance_extension as Rc<dyn FlagDisplay>),
            );
        }
    }

    /// Java `setEnabled(boolean)` (final).
    pub fn set_enabled(&self, enabled: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_enabled(enabled);
        } else {
            self.text_field.set_enabled(enabled);
        }
        if let Some(label) = &self.label {
            label.set_enabled(self.is_enabled());
        }
    }

    /// Java `isEnabled()` (final).
    pub fn is_enabled(&self) -> bool {
        if let Some(appearance_extension) = self.appearance_extension.borrow().as_ref() {
            return appearance_extension.is_enabled();
        }
        self.text_field.is_enabled()
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_editable(editable);
        } else {
            self.text_field.set_editable(editable);
        }
    }

    /// Java `isEditable()` (final).
    pub fn is_editable(&self) -> bool {
        if let Some(appearance_extension) = self.appearance_extension.borrow().as_ref() {
            return appearance_extension.is_editable();
        }
        self.text_field.is_editable()
    }

    /// Java `setLimitDisplayedFilePath(int)`: `maxFilePathSize` > 0 to turn on
    /// limit, <= 0 to turn it off.
    pub fn set_limit_displayed_file_path(&self, max_file_path_size: i32) {
        if max_file_path_size > 0 && self.value_manipulation_extension.borrow().is_none() {
            let field: Weak<dyn ValueManipulationField> = self.this.clone();
            *self.value_manipulation_extension.borrow_mut() = Some(
                ValueManipulationExtension::new(field, self, self.debug.get()),
            );
        }
        let value_manipulation_extension = self.value_manipulation_extension.borrow().clone();
        if let Some(value_manipulation_extension) = value_manipulation_extension {
            if value_manipulation_extension
                .set_limit_displayed_file_path(max_file_path_size, self.field_type)
            {
                // Redisplay text if necessary
                self.set_text_string(
                    value_manipulation_extension
                        .create_displayed_file_path_string_field_type(
                            self.get_text_void().as_deref(),
                            self.field_type,
                        )
                        .as_deref(),
                );
            }
        }
    }

    // / flagExtension

    /// Java `updateFlagExtension()`.
    pub fn update_flag_extension(&self) {
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            flag_extension.update_void();
        }
    }

    /// Java `createFlagExtension()` (virtual; see [`TextEfieldVirtual`]).  Creates
    /// flagExtension if it is null; returns true if flagExtension was created.
    pub fn create_flag_extension(&self) -> bool {
        match self.this.upgrade() {
            Some(this) => this.create_flag_extension(),
            None => self.default_create_flag_extension(),
        }
    }

    /// `TextEfield`'s own `createFlagExtension()` body.
    pub fn default_create_flag_extension(&self) -> bool {
        if self.flag_extension.borrow().is_none() {
            let this = self.this();
            *self.flag_extension.borrow_mut() =
                Some(TextFlagExtension::new(this as Rc<dyn TextFlagOrigin>));
            self.create_appearance_extension(true, true);
            return true;
        }
        false
    }

    /// Java `getFlagType()`.
    pub fn get_flag_type(&self) -> Option<&'static FlagType> {
        if let Some(appearance_extension) = self.appearance_extension.borrow().as_ref() {
            return appearance_extension.get_flag_type();
        }
        None
    }

    /// Java `addFlagOriginListener(FlagOriginListener)` (final).  Allow flags to
    /// listen for changes that they need to react to.
    pub fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>) {
        self.text_field
            .add_focus_listener(Rc::new(move |event: &FocusEvent| {
                if event.gained {
                    listener.focus_gained();
                } else {
                    listener.focus_lost();
                }
            }));
    }

    /// Java `flagTemplate(String)` (final).  Change appearance of the flag displays
    /// when value matches the template value.  Automatically adds its appearance
    /// extension as a flag display.
    pub fn flag_template(&self, template_value: Option<&str>) {
        self.create_flag_extension();
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.flag_template(template_value);
        flag_extension.update_void();
        if self.value_manipulation_extension.borrow().is_none() {
            let field: Weak<dyn ValueManipulationField> = self.this.clone();
            *self.value_manipulation_extension.borrow_mut() = Some(
                ValueManipulationExtension::new(field, self, self.debug.get()),
            );
        }
        let value_manipulation_extension =
            self.value_manipulation_extension.borrow().clone().unwrap();
        value_manipulation_extension.set_prevent_blank(true, template_value);
    }

    /// Java `setFlagErrors()` (final).
    pub fn set_flag_errors(&self) {
        self.create_flag_extension();
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.set_flag_errors();
        flag_extension.update_void();
    }

    /// Java `clearTemplateValue()` (final).
    pub fn clear_template_value(&self) {
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            flag_extension.clear_flags();
        }
        let value_manipulation_extension = self.value_manipulation_extension.borrow().clone();
        if let Some(value_manipulation_extension) = value_manipulation_extension {
            value_manipulation_extension.clear_prevent_blank();
        }
    }

    /// Java `setTemplateValue()` (final).
    pub fn set_template_value(&self) {
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            self.set_text_string(flag_extension.get_flagged_template_value().as_deref());
        }
    }

    /// Java `addFlagDisplay(FlagDisplay)` (final).
    pub fn add_flag_display(&self, flag_display: Option<Rc<dyn FlagDisplay>>) {
        self.create_flag_extension();
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.add_flag_display(flag_display);
        flag_extension.update_void();
    }

    /// Java `addFinalFlagDisplay(FlagDisplay)` (final).
    pub fn add_final_flag_display(&self, flag_display: Option<Rc<dyn FlagDisplay>>) {
        self.create_flag_extension();
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.add_final_flag_display(flag_display);
        flag_extension.update_void();
    }

    /// Java `setFieldHighlight(String)` (final).
    pub fn set_field_highlight(&self, value: Option<&str>) {
        self.flag_template(value);
    }

    // / stateExtension

    /// Java `backup()` (final).
    pub fn backup(&self) {
        if self.state_extension.borrow().is_none() {
            let this = self.this();
            *self.state_extension.borrow_mut() =
                Some(TextStateExtension::new(this as Rc<dyn TextEfieldInterface>));
        }
        let state_extension = self.state_extension.borrow().clone().unwrap();
        state_extension.backup();
    }

    /// Java `checkpoint()` (final).
    pub fn checkpoint(&self) {
        if self.state_extension.borrow().is_none() {
            let this = self.this();
            *self.state_extension.borrow_mut() =
                Some(TextStateExtension::new(this as Rc<dyn TextEfieldInterface>));
        }
        let state_extension = self.state_extension.borrow().clone().unwrap();
        state_extension.checkpoint();
    }

    /// Java `isDifferentFromCheckpoint(boolean alwaysCheck)` (final).
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        let state_extension = self.state_extension.borrow().clone();
        match state_extension {
            None => false,
            Some(state_extension) => state_extension.is_different_from_checkpoint(always_check),
        }
    }

    /// Java `restoreFromBackup()` (final).
    pub fn restore_from_backup(&self) {
        let state_extension = self.state_extension.borrow().clone();
        if let Some(state_extension) = state_extension {
            state_extension.restore_from_backup();
        }
    }

    // / Calls to gridBagExtension

    /// Java `remove()` (final).
    pub fn remove(&self) {
        let grid_bag_extension = self.grid_bag_extension.borrow().clone();
        if let Some(grid_bag_extension) = grid_bag_extension {
            grid_bag_extension.remove(&self.get_component());
        }
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)` (final).  The layout
    /// and constraints are Swing layout (not modelled); the panel is the parent.
    pub fn add(&self, panel: &Rc<JComponent>) {
        if self.grid_bag_extension.borrow().is_none() {
            *self.grid_bag_extension.borrow_mut() = Some(GridBagExtension::new());
        }
        let grid_bag_extension = self.grid_bag_extension.borrow().clone().unwrap();
        grid_bag_extension.add(&self.get_component(), panel);
    }

    /// Java `setText(File[])`.  (Empty in the source: "TODO Auto-generated method
    /// stub".)
    pub fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        let _ = files;
    }

    /// Java `isLocalDir(String)`.  (A stub in the source.)
    pub fn is_local_dir(&self, current_directory: Option<&str>) -> bool {
        let _ = current_directory;
        false
    }
}

impl TextEfieldVirtual for TextEfield {
    fn text_efield(&self) -> &TextEfield {
        self
    }
}

// ---- interface bindings (each forwards to the method above) ----

impl TextEfieldInterface for TextEfield {
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        TextEfield::get_directive_def(self)
    }
    fn is_enabled(&self) -> bool {
        TextEfield::is_enabled(self)
    }
    fn is_visible(&self) -> bool {
        TextEfield::is_visible(self)
    }
    fn get_text(&self) -> Option<String> {
        TextEfield::get_text_void(self)
    }
    fn set_text(&self, text: Option<&str>) {
        TextEfield::set_text_string(self, text)
    }
    fn set_field_highlight(&self, text: Option<&str>) {
        TextEfield::set_field_highlight(self, text)
    }
    fn set_template_value(&self) {
        TextEfield::set_template_value(self)
    }
    fn equals(&self, string: Option<&str>) -> bool {
        TextEfield::equals(self, string)
    }
    fn set_debug(&self, debug: bool) {
        TextEfield::set_debug(self, debug)
    }
}

impl UIComponent for TextEfield {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        TextEfield::get_component(self)
    }
}

impl SwingComponent for TextEfield {
    fn get_component(&self) -> Rc<JComponent> {
        TextEfield::get_component(self)
    }
}

impl TextFlagOrigin for TextEfield {
    fn equals(&self, value: Option<&str>) -> bool {
        TextEfield::equals(self, value)
    }
    fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>) {
        TextEfield::add_flag_origin_listener(self, listener)
    }
    fn is_valid(&self) -> bool {
        TextEfield::is_valid(self)
    }
}

impl ValueManipulationField for TextEfield {
    fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>) {
        TextEfield::add_value_manipulation_listener(self, listener)
    }
    fn is_empty(&self) -> bool {
        TextEfield::is_empty(self)
    }
    fn set_text(&self, text: Option<&str>) {
        TextEfield::set_text_string(self, text)
    }
}

impl ControlTarget for TextEfield {
    fn clear(&self) {
        TextEfield::clear(self)
    }
    fn set_text_file(&self, file: Option<&Path>) {
        TextEfield::set_text_file(self, file)
    }
    fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        TextEfield::set_text_file_array(self, files)
    }
    fn get_label(&self) -> Option<String> {
        TextEfield::get_label(self)
    }
    fn set_component_control(&self, control: bool, state: Option<&'static ControlState>) {
        TextEfield::set_component_control(self, control, state)
    }
    fn set_enable_control(&self, control: bool, state: Option<&'static ControlState>) {
        TextEfield::set_enable_control(self, control, state)
    }
    fn send_control_event(&self) {
        TextEfield::send_control_event(self)
    }
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool {
        TextEfield::is_local_dir(self, current_directory)
    }
}
