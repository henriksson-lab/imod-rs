//! `IMOD/Etomo/src/etomo/ui/swing/ComboBoxEfield.java`.
//!
//! A self-naming `JComboBox` field with optional control component, appearance,
//! flag, value manipulation, state and grid-bag extensions.  Its items are `null`
//! (the empty choice), `etomo.type.Option` choices and plain strings.
//!
//! `class ComboBoxEfield implements SwingComponent, UIComponent, TextFlagOrigin,
//! TextEfieldInterface, ValueManipulationField, ControlTarget`.  It is extended by
//! `BooleanComboBoxEfield`, which overrides nothing; but the Java object passes
//! itself to its extensions, so, as in `TextEfield`, the outermost object is handed
//! in as `this` (a `dyn ComboBoxEfieldVirtual`, which has all of those interfaces as
//! supertraits) by a constructor built with `Rc::new_cyclic`.  `isControlled`, the non-final
//! member it calls, is a default method of that trait.
//!
//! **Item objects.**  The Swing stand-in's combo box holds display strings only.
//! The Java items are objects (`null`, `Option`, `String`) that the class reads
//! back with `getSelectedItem`/`getItemAt` and compares with `equals`; so the
//! objects are kept here in `items`, index for index with the stand-in's list,
//! whose strings are each object's `toString()` (`""` for the null item).  Every
//! `comboBox.addItem` adds to both.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::appearance_extension::AppearanceExtension;
use super::control_component_module::ControlComponentModule;
use super::control_state::ControlState;
use super::control_target::ControlTarget;
use super::efield_container::EfieldContainer;
use super::grid_bag_extension::GridBagExtension;
use super::swing_component::SwingComponent;
use super::text_efield_interface::TextEfieldInterface;
use super::tooltip_formatter;
use super::value_manipulation_extension::ValueManipulationExtension;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{FocusEvent, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_descr_choice_list::DirectiveDescrChoiceList;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::option::Option as TypeOption;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_origin_listener::FlagOriginListener;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::ui::text_flag_extension::TextFlagExtension;
use crate::imod::etomo::ui::text_flag_origin::TextFlagOrigin;
use crate::imod::etomo::ui::text_state_extension::TextStateExtension;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::ui::value_manipulation_field::ValueManipulationField;
use crate::imod::etomo::ui::value_manipulation_listener::ValueManipulationListener;
use crate::imod::etomo::util::utilities;

/// One `JComboBox` item object (see the module documentation).
#[derive(Clone, Debug)]
pub enum ComboBoxItem {
    /// `comboBox.addItem(null)`.
    Null,
    /// An `etomo.type.Option`.
    Option(TypeOption),
    /// A `String`.
    Text(String),
}

impl ComboBoxItem {
    /// Java `item.toString()`, as the combo box displays it (`""` for null).
    fn display_string(&self) -> String {
        match self {
            ComboBoxItem::Null => String::new(),
            ComboBoxItem::Option(option) => option.to_string(),
            ComboBoxItem::Text(text) => text.clone(),
        }
    }

    /// Java `item.equals(Object)` for a `String` (or null) argument.
    fn equals(&self, string: Option<&str>) -> bool {
        match self {
            // Never called on the null item (the Java tests `item != null` first).
            ComboBoxItem::Null => false,
            ComboBoxItem::Option(option) => option.equals_object(string),
            ComboBoxItem::Text(text) => Some(text.as_str()) == string,
        }
    }
}

/// The non-final members of `ComboBoxEfield` and the interfaces the Java object
/// passes itself as.  The default bodies are `ComboBoxEfield`'s own.
pub trait ComboBoxEfieldVirtual:
    SwingComponent
    + UIComponent
    + TextFlagOrigin
    + TextEfieldInterface
    + ValueManipulationField
    + ControlTarget
{
    /// The `ComboBoxEfield` part of the object.
    fn combo_box_efield(&self) -> &ComboBoxEfield;

    /// Java `isControlled()`.
    fn is_controlled(&self) -> bool {
        false
    }
}

/// Java package-private `class ComboBoxEfield`.
pub struct ComboBoxEfield {
    /// Java `this`: the outermost object (see the module documentation).
    this: Weak<dyn ComboBoxEfieldVirtual>,
    /// Java final `comboBox` (`new JComboBox()`).
    combo_box: Rc<JComponent>,
    /// The combo box's item objects, index for index with `combo_box`'s list.
    items: RefCell<Vec<ComboBoxItem>>,
    /// Java final `label`.
    label: Option<String>,
    /// Java final `includeValue`.
    include_value: bool,
    /// Java final `controlComponent`.
    control_component: Option<Rc<ControlComponentModule>>,
    /// Java final `container`.
    container: EfieldContainer,
    /// Java `stateExtension`.
    state_extension: RefCell<Option<Rc<TextStateExtension>>>,
    /// Java `appearanceExtension`.
    appearance_extension: RefCell<Option<Rc<AppearanceExtension>>>,
    /// Java `flagExtension`.
    flag_extension: RefCell<Option<Rc<TextFlagExtension>>>,
    /// Java `gridBagExtension`.
    grid_bag_extension: RefCell<Option<Rc<GridBagExtension>>>,
    /// Java `choiceListSet`.
    choice_list_set: Cell<bool>,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `valueManipulationExtension`.
    value_manipulation_extension: RefCell<Option<Rc<ValueManipulationExtension>>>,
    /// Java `emptyIndex`.
    empty_index: Cell<i32>,
    /// Java `directiveDef`.
    directive_def: Cell<Option<DirectiveDef>>,
}

impl ComboBoxEfield {
    /// Java `ComboBoxEfield(String label, boolean includeValue, boolean
    /// includeControlComponent)`.  `includeValue`: include the value along with the
    /// description for each option in the pulldown list.
    ///
    /// `this` is the outermost object, from the caller's `Rc::new_cyclic`.
    pub fn new(
        this: Weak<dyn ComboBoxEfieldVirtual>,
        label: Option<&str>,
        include_value: bool,
        include_control_component: bool,
    ) -> ComboBoxEfield {
        let combo_box = JComponent::new_combo_box();
        let control_component = if include_control_component {
            Some(ControlComponentModule::new())
        } else {
            None
        };
        // Java builds the container last; it depends only on the combo box and the
        // control component, so building it first (the Rust field is immutable)
        // changes nothing.
        let container = EfieldContainer::new(
            false,
            false,
            None,
            &combo_box,
            control_component
                .as_ref()
                .map(|control_component| control_component.get_component())
                .as_ref(),
        );
        let instance = ComboBoxEfield {
            this,
            combo_box,
            items: RefCell::new(Vec::new()),
            label: label.map(str::to_owned),
            include_value,
            control_component,
            container,
            state_extension: RefCell::new(None),
            appearance_extension: RefCell::new(None),
            flag_extension: RefCell::new(None),
            grid_bag_extension: RefCell::new(None),
            choice_list_set: Cell::new(false),
            debug: Cell::new(false),
            value_manipulation_extension: RefCell::new(None),
            empty_index: Cell::new(-1),
            directive_def: Cell::new(None),
        };
        instance.empty_index.set(0);
        // setName(label).  Java calls it before the final `controlComponent` is
        // assigned, so its `controlComponent != null` branch does not run here.
        let field_type = UITestFieldType::COMBO_BOX;
        let name = utilities::convert_label_to_name(label, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        instance.combo_box.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                instance.combo_box.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
        // comboBox.addItem(null)
        instance.items.borrow_mut().push(ComboBoxItem::Null);
        instance.combo_box.add_item("");
        instance
    }

    /// Java static `getInstance(String, boolean)`.
    pub fn get_instance(label: Option<&str>, include_value: bool) -> Rc<ComboBoxEfield> {
        Rc::new_cyclic(|this: &Weak<ComboBoxEfield>| {
            ComboBoxEfield::new(this.clone(), label, include_value, false)
        })
    }

    /// Java static `getOverrideInstance(String, boolean)`.
    pub fn get_override_instance(label: Option<&str>, include_value: bool) -> Rc<ComboBoxEfield> {
        Rc::new_cyclic(|this: &Weak<ComboBoxEfield>| {
            ComboBoxEfield::new(this.clone(), label, include_value, true)
        })
    }

    /// The outermost object; panics only if used during its own construction.
    fn this(&self) -> Rc<dyn ComboBoxEfieldVirtual> {
        self.this
            .upgrade()
            .expect("ComboBoxEfield used during construction or after it was dropped")
    }

    /// Java `comboBox.addItem(Object)`: adds the object and its display string.
    fn combo_box_add_item(&self, item: ComboBoxItem) {
        let display_string = item.display_string();
        self.items.borrow_mut().push(item);
        self.combo_box.add_item(&display_string);
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = UITestFieldType::COMBO_BOX;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        self.combo_box.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.combo_box.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
        if let Some(control_component) = &self.control_component {
            control_component.set_to_container_name(text);
        }
    }

    /// Java `@Override getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        self.label.clone()
    }

    /// Java `@Override setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            flag_extension.set_debug(debug);
        }
    }

    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug.get()
    }

    /// Java final `setChoiceList(DirectiveDescrChoiceList)`.
    pub fn set_choice_list(&self, choice_list: &DirectiveDescrChoiceList) {
        self.choice_list_set.set(true);
        for choice in choice_list.iterator() {
            // Java: `if (choice != null)`; the list holds no null choices.
            let mut local_choice = TypeOption::new_option(choice);
            local_choice.set_include_value(self.include_value);
            self.combo_box_add_item(ComboBoxItem::Option(local_choice));
        }
    }

    /// Java final `getUIComponent()`.
    pub fn get_ui_component(&self) -> Rc<dyn SwingComponent> {
        self.this() as Rc<dyn SwingComponent>
    }

    /// Java `@Override getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.container.get_component()
    }

    /// Java final `setChoiceListSet(boolean)`.
    pub fn set_choice_list_set(&self, set: bool) {
        self.choice_list_set.set(set);
    }

    /// Java final `setSelectedIndex(int)`.
    pub fn set_selected_index(&self, index: i32) {
        self.combo_box.set_selected_index(index);
    }

    /// Java final `setText(File)`.
    pub fn set_text_file(&self, file: Option<&Path>) {
        match file {
            None => self.set_text_string(Some("")),
            Some(file) => self.set_text_string(Some(&utilities::java_io_file_get_absolute_path(
                &file.to_string_lossy(),
            ))),
        }
    }

    /// Java final `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        let text = match text {
            Some(text) if !java_lang_string_matches_whitespace(text) => text,
            _ => {
                self.clear();
                return;
            }
        };
        // Set an existing item
        let size = self.combo_box.get_item_count() as i32;
        let mut i = self.empty_index.get() + 1;
        while i < size {
            let item = self.items.borrow().get(i as usize).cloned();
            if let Some(item) = item {
                if !matches!(item, ComboBoxItem::Null) && item.equals(Some(text)) {
                    self.set_selected_index(i);
                    self.update_flag_extension();
                    return;
                }
            }
            i += 1;
        }
        // Add and select new item. This is invalid if the choice list was set.
        self.combo_box_add_item(ComboBoxItem::Text(text.to_owned()));
        self.set_selected_index(size);
        if self.choice_list_set.get() {
            self.create_flag_extension();
            let flag_extension = self.flag_extension.borrow().clone().unwrap();
            flag_extension.set_flag_errors();
        }
        self.update_flag_extension();
    }

    /// Java final `getText()`.
    pub fn get_text(&self) -> Option<String> {
        if self.this().is_controlled() {
            return Some(String::new());
        }
        // comboBox.getSelectedItem()
        let index = self.combo_box.get_selected_index();
        if index < 0 {
            return None;
        }
        let item = self.items.borrow().get(index as usize).cloned();
        match item {
            None | Some(ComboBoxItem::Null) => None,
            Some(ComboBoxItem::Option(option)) => option.get_value().map(str::to_owned),
            Some(ComboBoxItem::Text(text)) => Some(text),
        }
    }

    /// Java final `getSelectedIndex()`.
    pub fn get_selected_index(&self) -> i32 {
        self.combo_box.get_selected_index()
    }

    /// Java final `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        let index = self.get_selected_index();
        index == -1 || self.empty_index.get() != -1 && index == self.empty_index.get()
    }

    /// Java final `equals(String)`.
    pub fn equals(&self, compare_text: Option<&str>) -> bool {
        // comboBox.getSelectedItem()
        let index = self.combo_box.get_selected_index();
        let item = if index < 0 {
            None
        } else {
            self.items.borrow().get(index as usize).cloned()
        };
        match item {
            None | Some(ComboBoxItem::Null) => compare_text.is_none(),
            Some(item) => item.equals(compare_text),
        }
    }

    /// Java final `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.combo_box
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java final `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_component().set_visible(visible);
    }

    /// Java final `setComboBoxVisible(boolean)`.
    pub fn set_combo_box_visible(&self, visible: bool) {
        self.combo_box.set_visible(visible);
    }

    /// Java final `addItem(Option)`.
    pub fn add_item(&self, option: Option<TypeOption>) {
        match option {
            None => self.combo_box_add_item(ComboBoxItem::Null),
            Some(option) => self.combo_box_add_item(ComboBoxItem::Option(option)),
        }
    }

    /// Java final `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.get_component().is_visible()
    }

    /// Java final `getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java final `setDirectiveDef(DirectiveDef)`.
    pub fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java final `isTemplateValue()`.
    pub fn is_template_value(&self) -> bool {
        let flag_type = self.get_flag_type();
        flag_type.is_some_and(|flag_type| flag_type.is_template())
    }

    // / validation

    /// Java `@Override isValid()`.
    pub fn is_valid(&self) -> bool {
        if !self.choice_list_set.get() || self.this().is_controlled() {
            return true;
        }
        // comboBox.getSelectedItem()
        let index = self.combo_box.get_selected_index();
        let item = if index < 0 {
            None
        } else {
            self.items.borrow().get(index as usize).cloned()
        };
        match item {
            None | Some(ComboBoxItem::Null) => true,
            // When the choice list was set, anything added using setText is invalid.
            Some(ComboBoxItem::Text(_)) => false,
            Some(ComboBoxItem::Option(_)) => true,
        }
    }

    // / valueManipulationExtension

    /// Java final `clear()`.
    pub fn clear(&self) {
        if self.empty_index.get() != -1 {
            self.set_selected_index(self.empty_index.get());
        } else {
            let value_manipulation_extension = self.value_manipulation_extension.borrow().clone();
            if let Some(value_manipulation_extension) = value_manipulation_extension {
                value_manipulation_extension.substitute();
            }
        }
        self.update_flag_extension();
    }

    /// Java final `addValueManipulationListener(ValueManipulationListener)`.  Not
    /// necessary for a comboBox with an ineditable text field.
    pub fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>) {
        self.combo_box
            .add_focus_listener(Rc::new(move |event: &FocusEvent| {
                if event.gained {
                    listener.focus_gained();
                } else {
                    listener.focus_lost();
                }
            }));
    }

    // / control

    /// Java `isOverride()`.
    pub fn is_override(&self) -> bool {
        match &self.control_component {
            None => false,
            Some(control_component) => control_component.is_override(),
        }
    }

    /// Java final `setComponentControl(boolean, ControlState)`.
    pub fn set_component_control(
        &self,
        control: bool,
        control_state: Option<&'static ControlState>,
    ) {
        let Some(control_component) = &self.control_component else {
            return;
        };
        self.combo_box
            .set_visible(!control_component.set_component_control(control, control_state));
    }

    /// Java final `setEnableControl(boolean, ControlState)` (empty).
    pub fn set_enable_control(
        &self,
        _control: bool,
        _control_state: Option<&'static ControlState>,
    ) {
    }

    // / appearanceExtension

    /// Java private final `createAppearanceExtension()`.
    fn create_appearance_extension(&self) {
        if self.appearance_extension.borrow().is_none() {
            let appearance_extension = AppearanceExtension::new_component(&self.combo_box);
            appearance_extension.set_allow_foreground_change_on_error(false);
            *self.appearance_extension.borrow_mut() = Some(appearance_extension.clone());
            // In this class the appearanceExtension acts as the field's flag display
            // for all types of flags.
            let flag_extension = self.flag_extension.borrow().clone();
            if let Some(flag_extension) = flag_extension {
                flag_extension.add_flag_display(Some(appearance_extension as Rc<dyn FlagDisplay>));
            }
        }
    }

    /// Java final `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.create_appearance_extension();
        let appearance_extension = self.appearance_extension.borrow().clone().unwrap();
        appearance_extension.set_editable(editable);
    }

    /// Java final `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            None => self.combo_box.is_enabled(),
            Some(appearance_extension) => appearance_extension.is_enabled(),
        }
    }

    /// Java final `isEditable()`.
    pub fn is_editable(&self) -> bool {
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            None => false,
            Some(appearance_extension) => appearance_extension.is_editable(),
        }
    }

    /// Java final `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            None => self.combo_box.set_enabled(enabled),
            Some(appearance_extension) => appearance_extension.set_enabled(enabled),
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

    /// Java `createFlagExtension()`.  Creates flagExtension if it is null; returns
    /// true if flagExtension was created.
    pub fn create_flag_extension(&self) -> bool {
        if self.flag_extension.borrow().is_none() {
            let flag_extension = TextFlagExtension::new(self.this() as Rc<dyn TextFlagOrigin>);
            *self.flag_extension.borrow_mut() = Some(flag_extension.clone());
            let appearance_extension = self.appearance_extension.borrow().clone();
            if let Some(appearance_extension) = appearance_extension {
                flag_extension.add_flag_display(Some(appearance_extension as Rc<dyn FlagDisplay>));
            }
            return true;
        }
        false
    }

    /// Java `getFlagType()`.
    pub fn get_flag_type(&self) -> Option<&'static FlagType> {
        let appearance_extension = self.appearance_extension.borrow().clone();
        match appearance_extension {
            Some(appearance_extension) => appearance_extension.get_flag_type(),
            None => None,
        }
    }

    /// Java final `addFlagOriginListener(FlagOriginListener)`.  Allow flags to listen
    /// for changes that they need to react to.
    pub fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>) {
        // comboBox.addItemListener(listener)
        self.combo_box
            .add_item_listener(Rc::new(move |event| listener.item_state_changed(event)));
    }

    /// Java private final `flagTemplate(String)`.  Change appearance when value
    /// matches the template value.  Also set up value manipulation to prevent a blank
    /// value.
    fn flag_template(&self, template_value: Option<&str>) {
        if template_value.is_some() {
            self.create_flag_extension();
            let flag_extension = self.flag_extension.borrow().clone().unwrap();
            flag_extension.flag_template(template_value);
            flag_extension.update_void();
            if self.value_manipulation_extension.borrow().is_none() {
                let field = self.this() as Rc<dyn ValueManipulationField>;
                *self.value_manipulation_extension.borrow_mut() = Some(
                    ValueManipulationExtension::new(Rc::downgrade(&field), self, self.debug.get()),
                );
            }
            let value_manipulation_extension =
                self.value_manipulation_extension.borrow().clone().unwrap();
            value_manipulation_extension.set_prevent_blank(true, template_value);
        }
    }

    /// Java final `clearTemplateValue()`.
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

    /// Java final `setTemplateValue()`.
    pub fn set_template_value(&self) {
        let flag_extension = self.flag_extension.borrow().clone();
        if let Some(flag_extension) = flag_extension {
            self.set_text_string(flag_extension.get_flagged_template_value().as_deref());
        }
    }

    /// Java final `addFlagDisplay(FlagDisplay)`.
    pub fn add_flag_display(&self, flag_display: Option<Rc<dyn FlagDisplay>>) {
        self.create_flag_extension();
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.add_flag_display(flag_display);
        flag_extension.update_void();
    }

    /// Java final `addFinalFlagDisplay(FlagDisplay)`.
    pub fn add_final_flag_display(&self, flag_display: Option<Rc<dyn FlagDisplay>>) {
        self.create_flag_extension();
        let flag_extension = self.flag_extension.borrow().clone().unwrap();
        flag_extension.add_final_flag_display(flag_display);
        flag_extension.update_void();
    }

    /// Java final `setFieldHighlight(String)`.
    pub fn set_field_highlight(&self, value: Option<&str>) {
        self.flag_template(value);
    }

    // / stateExtension

    /// Java private final `createStateExtension()`.
    fn create_state_extension(&self) {
        if self.state_extension.borrow().is_none() {
            *self.state_extension.borrow_mut() = Some(TextStateExtension::new(
                self.this() as Rc<dyn TextEfieldInterface>
            ));
        }
    }

    /// Java final `backup()`.
    pub fn backup(&self) {
        self.create_state_extension();
        let state_extension = self.state_extension.borrow().clone().unwrap();
        state_extension.backup();
    }

    /// Java final `checkpoint()`.
    pub fn checkpoint(&self) {
        self.create_state_extension();
        let state_extension = self.state_extension.borrow().clone().unwrap();
        state_extension.checkpoint();
    }

    /// Java final `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        let state_extension = self.state_extension.borrow().clone();
        match state_extension {
            None => false,
            Some(state_extension) => state_extension.is_different_from_checkpoint(always_check),
        }
    }

    /// Java final `restoreFromBackup()`.
    pub fn restore_from_backup(&self) {
        let state_extension = self.state_extension.borrow().clone();
        if let Some(state_extension) = state_extension {
            state_extension.restore_from_backup();
        }
    }

    // / gridBagExtension

    /// Java final `remove()`.
    pub fn remove(&self) {
        let grid_bag_extension = self.grid_bag_extension.borrow().clone();
        if let Some(grid_bag_extension) = grid_bag_extension {
            grid_bag_extension.remove(&self.get_component());
        }
    }

    /// Java final `add(JPanel, GridBagLayout, GridBagConstraints)`.  The layout and
    /// constraints are not modelled.
    pub fn add(&self, panel: &Rc<JComponent>) {
        if self.grid_bag_extension.borrow().is_none() {
            *self.grid_bag_extension.borrow_mut() = Some(GridBagExtension::new());
        }
        let grid_bag_extension = self.grid_bag_extension.borrow().clone().unwrap();
        grid_bag_extension.add(&self.get_component(), panel);
    }

    /// Java `@Override setText(File[])`.  (Empty in the source: "TODO Auto-generated
    /// method stub".)
    pub fn set_text_file_array(&self, _files: Option<&[PathBuf]>) {}

    /// Java `@Override isLocalDir(String)`.  (A stub in the source.)
    pub fn is_local_dir(&self, _current_directory: Option<&str>) -> bool {
        false
    }
}

impl ComboBoxEfieldVirtual for ComboBoxEfield {
    fn combo_box_efield(&self) -> &ComboBoxEfield {
        self
    }
}

// ---- interface bindings (each forwards to the method above) ----

impl SwingComponent for ComboBoxEfield {
    fn get_component(&self) -> Rc<JComponent> {
        ComboBoxEfield::get_component(self)
    }
}

impl UIComponent for ComboBoxEfield {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        ComboBoxEfield::get_component(self)
    }
}

impl TextFlagOrigin for ComboBoxEfield {
    fn equals(&self, value: Option<&str>) -> bool {
        ComboBoxEfield::equals(self, value)
    }
    fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>) {
        ComboBoxEfield::add_flag_origin_listener(self, listener)
    }
    fn is_valid(&self) -> bool {
        ComboBoxEfield::is_valid(self)
    }
}

impl TextEfieldInterface for ComboBoxEfield {
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        ComboBoxEfield::get_directive_def(self)
    }
    fn is_enabled(&self) -> bool {
        ComboBoxEfield::is_enabled(self)
    }
    fn is_visible(&self) -> bool {
        ComboBoxEfield::is_visible(self)
    }
    fn get_text(&self) -> Option<String> {
        ComboBoxEfield::get_text(self)
    }
    fn set_text(&self, text: Option<&str>) {
        ComboBoxEfield::set_text_string(self, text)
    }
    fn set_field_highlight(&self, text: Option<&str>) {
        ComboBoxEfield::set_field_highlight(self, text)
    }
    fn set_template_value(&self) {
        ComboBoxEfield::set_template_value(self)
    }
    fn equals(&self, string: Option<&str>) -> bool {
        ComboBoxEfield::equals(self, string)
    }
    fn set_debug(&self, debug: bool) {
        ComboBoxEfield::set_debug(self, debug)
    }
}

impl ValueManipulationField for ComboBoxEfield {
    fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>) {
        ComboBoxEfield::add_value_manipulation_listener(self, listener)
    }
    fn is_empty(&self) -> bool {
        ComboBoxEfield::is_empty(self)
    }
    fn set_text(&self, text: Option<&str>) {
        ComboBoxEfield::set_text_string(self, text)
    }
}

impl ControlTarget for ComboBoxEfield {
    fn clear(&self) {
        ComboBoxEfield::clear(self)
    }
    fn set_text_file(&self, file: Option<&Path>) {
        ComboBoxEfield::set_text_file(self, file)
    }
    fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        ComboBoxEfield::set_text_file_array(self, files)
    }
    fn get_label(&self) -> Option<String> {
        ComboBoxEfield::get_label(self)
    }
    fn set_component_control(&self, control: bool, state: Option<&'static ControlState>) {
        ComboBoxEfield::set_component_control(self, control, state)
    }
    fn set_enable_control(&self, control: bool, state: Option<&'static ControlState>) {
        ComboBoxEfield::set_enable_control(self, control, state)
    }
    /// Java `@Override sendControlEvent()` (empty).
    fn send_control_event(&self) {}
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool {
        ComboBoxEfield::is_local_dir(self, current_directory)
    }
}
