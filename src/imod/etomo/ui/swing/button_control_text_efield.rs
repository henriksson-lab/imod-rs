//! `IMOD/Etomo/src/etomo/ui/swing/ButtonControlTextEfield.java`.
//!
//! A `TextEfield` with an optional select-file button and an optional clear button,
//! both `Ebutton`s that control this field.
//!
//! Java `final class ButtonControlTextEfield extends TextEfield`: the superclass is
//! embedded as `base` (with `Deref`), and the overridden members `setDebug`,
//! `setTooltip(String)`, `createAppearanceExtension` and `createFlagExtension` are
//! in the [`TextEfieldVirtual`] / [`TextEfieldInterface`] implementations.
//!
//! The Java constructor passes `this` to the buttons as their `ControlTarget`, and
//! the buttons read the target's label for their uitest names while they are being
//! constructed.  A `Weak` handed out from inside `Rc::new_cyclic` cannot be upgraded
//! until the constructor returns, so the buttons would lose the label from their
//! names; they are therefore created right after the object is allocated, and the
//! two Java `final` button fields are `OnceCell`s set there, before the constructor
//! returns - which is everything the Java constructor does after `super(...)`.

use std::cell::OnceCell;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::control_state::ControlState;
use super::control_target::ControlTarget;
use super::controller::Controller;
use super::ebutton::Ebutton;
use super::select_file_extension::SelectFileExtension;
use super::swing_component::SwingComponent;
use super::text_efield::{TextEfield, TextEfieldVirtual};
use super::text_efield_interface::TextEfieldInterface;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_origin_listener::FlagOriginListener;
use crate::imod::etomo::ui::text_flag_origin::TextFlagOrigin;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::ui::value_manipulation_field::ValueManipulationField;
use crate::imod::etomo::ui::value_manipulation_listener::ValueManipulationListener;

/// Java package-private `final class ButtonControlTextEfield extends TextEfield`.
pub struct ButtonControlTextEfield {
    base: TextEfield,
    /// Java final `selectFileButton` (null when not included).
    select_file_button: OnceCell<Option<Rc<Ebutton>>>,
    /// Java final `clearButton` (null when not included).
    clear_button: OnceCell<Option<Rc<Ebutton>>>,
}

impl Deref for ButtonControlTextEfield {
    type Target = TextEfield;
    fn deref(&self) -> &TextEfield {
        &self.base
    }
}

impl ButtonControlTextEfield {
    /// Java private `ButtonControlTextEfield(String label, FieldType fieldType, boolean
    /// useLabelComponent, boolean useControlComponent, boolean enabledField, boolean
    /// editableComponent, SelectFileExtension sharedSelectFileExtension, boolean
    /// includeSelectFileButton, boolean includeClearButton, boolean useGridBag,
    /// boolean selectMultipleFiles, boolean defaultToFileName, boolean debug)`.
    ///
    /// - `label`: identifier and label - for labeled and unlabeled fields
    /// - `field_type`: for validation
    /// - `use_label_component`: for fields with labels
    /// - `use_control_component`: for stateful control
    /// - `enabled_field`: when false field cannot be enabled
    /// - `editable_component`: when false field cannot be made editable
    /// - `shared_select_file_extension`: a shared extension for building a file
    ///   chooser
    /// - `include_select_file_button`: add a file chooser button
    /// - `include_clear_button`: add a clear button
    #[allow(clippy::too_many_arguments)]
    fn new(
        label: Option<&str>,
        field_type: Option<FieldType>,
        use_label_component: bool,
        use_control_component: bool,
        enabled_field: bool,
        editable_component: bool,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
        include_select_file_button: bool,
        include_clear_button: bool,
        use_grid_bag: bool,
        select_multiple_files: bool,
        default_to_file_name: bool,
        debug: bool,
    ) -> Rc<ButtonControlTextEfield> {
        // super(label, fieldType, useLabelComponent, useControlComponent, enabledField,
        // editableComponent, useGridBag, defaultToFileName)
        let instance =
            Rc::new_cyclic(
                |this: &Weak<ButtonControlTextEfield>| ButtonControlTextEfield {
                    base: TextEfield::new(
                        this.clone() as Weak<dyn TextEfieldVirtual>,
                        label,
                        field_type,
                        use_label_component,
                        use_control_component,
                        enabled_field,
                        editable_component,
                        use_grid_bag,
                        default_to_file_name,
                    ),
                    select_file_button: OnceCell::new(),
                    clear_button: OnceCell::new(),
                },
            );
        let target = Rc::downgrade(&instance) as Weak<dyn ControlTarget>;
        if include_select_file_button {
            let select_file_button = if select_multiple_files {
                Ebutton::get_select_multiple_files_instance_control_target_select_file_extension(
                    Some(target.clone()),
                    shared_select_file_extension,
                )
            } else {
                Ebutton::get_select_file_instance_control_target_select_file_extension_boolean(
                    Some(target.clone()),
                    shared_select_file_extension,
                    debug,
                )
            };
            instance
                .base
                .container
                .add(Some(&select_file_button.get_component()));
            let _ = instance.select_file_button.set(Some(select_file_button));
        } else {
            let _ = instance.select_file_button.set(None);
        }
        if include_clear_button {
            let clear_button = Ebutton::get_clear_instance(Some(target));
            instance
                .base
                .container
                .add(Some(&clear_button.get_component()));
            let _ = instance.clear_button.set(Some(clear_button));
        } else {
            let _ = instance.clear_button.set(None);
        }
        let appearance_extension = instance.base.appearance_extension.borrow().clone();
        if let Some(appearance_extension) = appearance_extension {
            appearance_extension.set_child_controllers(instance.create_child_controllers());
        }
        instance
    }

    /// Java static `getFileOverrideInstance(String, SelectFileExtension, boolean)`.
    /// `sharedSelectFileExtension` (optional) is for a group of fields that use the
    /// same file chooser definition.
    pub fn get_file_override_instance(
        label: Option<&str>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
        debug: bool,
    ) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            false,
            true,
            true,
            false,
            shared_select_file_extension,
            true,
            true,
            false,
            false,
            true,
            debug,
        )
    }

    /// Java static `getFileInstance(String, SelectFileExtension)`.
    pub fn get_file_instance_string_select_file_extension(
        label: Option<&str>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
    ) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            false,
            false,
            true,
            false,
            shared_select_file_extension,
            true,
            true,
            false,
            false,
            true,
            false,
        )
    }

    /// Java static `getFileInstance(String, SelectFileExtension, boolean, boolean)`.
    pub fn get_file_instance_string_select_file_extension_boolean_boolean(
        label: Option<&str>,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
        include_clear_button: bool,
        editable_component: bool,
    ) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            false,
            false,
            true,
            editable_component,
            shared_select_file_extension,
            true,
            include_clear_button,
            false,
            false,
            true,
            false,
        )
    }

    /// Java static `getFileInstance(String)`.
    pub fn get_file_instance_string(label: Option<&str>) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            false,
            false,
            true,
            false,
            None,
            true,
            true,
            false,
            false,
            true,
            false,
        )
    }

    /// Java static `getLabeledFileInstance(String)`.
    pub fn get_labeled_file_instance_string(label: Option<&str>) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            true,
            false,
            true,
            false,
            None,
            true,
            false,
            false,
            false,
            true,
            false,
        )
    }

    /// Java static `getLabeledFileInstance(String, boolean, boolean)`.
    pub fn get_labeled_file_instance_string_boolean_boolean(
        label: Option<&str>,
        include_clear_button: bool,
        use_grid_bag: bool,
    ) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            true,
            false,
            true,
            false,
            None,
            true,
            include_clear_button,
            use_grid_bag,
            false,
            true,
            false,
        )
    }

    /// Java static `getLabeledFileInstance(String, boolean, boolean,
    /// SelectFileExtension)`.
    pub fn get_labeled_file_instance_string_boolean_boolean_select_file_extension(
        label: Option<&str>,
        include_clear_button: bool,
        use_grid_bag: bool,
        shared_select_file_extension: Option<Rc<SelectFileExtension>>,
    ) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            true,
            false,
            true,
            false,
            shared_select_file_extension,
            true,
            include_clear_button,
            use_grid_bag,
            false,
            true,
            false,
        )
    }

    /// Java static `getLabeledMultipleFilesInstance(String, boolean, boolean)`.
    /// ALIGN_FRAMES.
    pub fn get_labeled_multiple_files_instance(
        label: Option<&str>,
        include_clear_button: bool,
        use_grid_bag: bool,
    ) -> Rc<ButtonControlTextEfield> {
        ButtonControlTextEfield::new(
            label,
            Some(FieldType::File),
            true,
            false,
            true,
            false,
            None,
            true,
            include_clear_button,
            use_grid_bag,
            true,
            true,
            false,
        )
    }

    /// Java public `@Override setDebug(boolean)`.  (It does not call
    /// `super.setDebug`, so the field's own debug flag is left unchanged; kept as in
    /// the Java.)
    pub fn set_debug(&self, debug: bool) {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_debug(debug);
        }
    }

    /// Java final `@Override setTooltip(String)`.
    pub fn set_tooltip_string(&self, text: Option<&str>) {
        self.base.default_set_tooltip(text);
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_tooltip(text);
        }
        if let Some(clear_button) = self.clear_button.get().cloned().flatten() {
            clear_button.set_tooltip(text);
        }
    }

    /// Java final `setTooltip(String, String)`.
    pub fn set_tooltip_string_string(&self, text: Option<&str>, select_file_text: Option<&str>) {
        // super.setTooltip(text)
        self.base.default_set_tooltip(text);
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_tooltip(select_file_text);
        }
        if let Some(clear_button) = self.clear_button.get().cloned().flatten() {
            clear_button.set_tooltip(text);
        }
    }

    /// Java `setOverrideFileOpenDirectory(File)`.
    pub fn set_override_file_open_directory(&self, override_file_open_directory: Option<PathBuf>) {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_override_file_open_directory(override_file_open_directory);
        }
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&self, file_filter: Option<Rc<dyn FileFilter>>) {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_file_filter(file_filter);
        }
    }

    /// Java `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&self, file_selection_mode: i32) {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_file_selection_mode(file_selection_mode);
        }
    }

    /// Java `setSelectFileDir(String)`.
    pub fn set_select_file_dir(&self, dir: Option<&str>) {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_select_file_dir(dir);
        }
    }

    /// Java `getSelectFileExtension()`.
    pub fn get_select_file_extension(&self) -> Option<Rc<SelectFileExtension>> {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            return select_file_button.get_select_file_extension();
        }
        None
    }

    // / appearanceExtension

    /// Java `@Override createAppearanceExtension(boolean, boolean)`.
    pub fn create_appearance_extension(&self, enabled_field: bool, editable_component: bool) {
        let create = self.base.appearance_extension.borrow().is_none();
        self.base
            .default_create_appearance_extension(enabled_field, editable_component);
        if create {
            let appearance_extension = self.base.appearance_extension.borrow().clone().unwrap();
            appearance_extension.set_child_controllers(self.create_child_controllers());
        }
    }

    /// Java private `createChildControllers()`.
    fn create_child_controllers(&self) -> Option<Vec<Option<Rc<dyn Controller>>>> {
        let select_file_button = self.select_file_button.get().cloned().flatten();
        let clear_button = self.clear_button.get().cloned().flatten();
        let mut size = 0;
        if select_file_button.is_some() {
            size += 1;
        }
        if clear_button.is_some() {
            size += 1;
        }
        if size == 0 {
            return None;
        }
        let mut controllers: Vec<Option<Rc<dyn Controller>>> = vec![None; size];
        let mut index = 0;
        if let Some(select_file_button) = select_file_button {
            controllers[index] = Some(select_file_button as Rc<dyn Controller>);
            index += 1;
        }
        if let Some(clear_button) = clear_button {
            controllers[index] = Some(clear_button as Rc<dyn Controller>);
        }
        Some(controllers)
    }

    // / flagExtension

    /// Java `@Override createFlagExtension()`.  Adds buttons as flag displays to the
    /// flag extension, if the flag extension was created.  Returns true if
    /// flagExtension was created.
    pub fn create_flag_extension(&self) -> bool {
        if self.base.default_create_flag_extension() {
            if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
                self.base
                    .add_flag_display(Some(select_file_button as Rc<dyn FlagDisplay>));
            }
            if let Some(clear_button) = self.clear_button.get().cloned().flatten() {
                self.base
                    .add_flag_display(Some(clear_button as Rc<dyn FlagDisplay>));
            }
            return true;
        }
        false
    }

    /// Java public `setAltBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_alt_browsing_directory(
        &self,
        browsing_directory: Option<Rc<dyn BrowsingDirectory>>,
    ) {
        if let Some(select_file_button) = self.select_file_button.get().cloned().flatten() {
            select_file_button.set_alt_browsing_directory(browsing_directory);
        }
    }
}

impl TextEfieldVirtual for ButtonControlTextEfield {
    fn text_efield(&self) -> &TextEfield {
        &self.base
    }
    fn set_tooltip(&self, text: Option<&str>) {
        ButtonControlTextEfield::set_tooltip_string(self, text)
    }
    fn create_appearance_extension(&self, enabled_field: bool, editable_component: bool) {
        ButtonControlTextEfield::create_appearance_extension(
            self,
            enabled_field,
            editable_component,
        )
    }
    fn create_flag_extension(&self) -> bool {
        ButtonControlTextEfield::create_flag_extension(self)
    }
}

// ---- interface bindings (inherited from TextEfield, except setDebug) ----

impl TextEfieldInterface for ButtonControlTextEfield {
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.base.get_directive_def()
    }
    fn is_enabled(&self) -> bool {
        self.base.is_enabled()
    }
    fn is_visible(&self) -> bool {
        self.base.is_visible()
    }
    fn get_text(&self) -> Option<String> {
        self.base.get_text_void()
    }
    fn set_text(&self, text: Option<&str>) {
        self.base.set_text_string(text)
    }
    fn set_field_highlight(&self, text: Option<&str>) {
        self.base.set_field_highlight(text)
    }
    fn set_template_value(&self) {
        self.base.set_template_value()
    }
    fn equals(&self, string: Option<&str>) -> bool {
        self.base.equals(string)
    }
    fn set_debug(&self, debug: bool) {
        ButtonControlTextEfield::set_debug(self, debug)
    }
}

impl UIComponent for ButtonControlTextEfield {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }
}

impl SwingComponent for ButtonControlTextEfield {
    fn get_component(&self) -> Rc<JComponent> {
        self.base.get_component()
    }
}

impl TextFlagOrigin for ButtonControlTextEfield {
    fn equals(&self, value: Option<&str>) -> bool {
        self.base.equals(value)
    }
    fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>) {
        self.base.add_flag_origin_listener(listener)
    }
    fn is_valid(&self) -> bool {
        self.base.is_valid()
    }
}

impl ValueManipulationField for ButtonControlTextEfield {
    fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>) {
        self.base.add_value_manipulation_listener(listener)
    }
    fn is_empty(&self) -> bool {
        self.base.is_empty()
    }
    fn set_text(&self, text: Option<&str>) {
        self.base.set_text_string(text)
    }
}

impl ControlTarget for ButtonControlTextEfield {
    fn clear(&self) {
        self.base.clear()
    }
    fn set_text_file(&self, file: Option<&Path>) {
        self.base.set_text_file(file)
    }
    fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        self.base.set_text_file_array(files)
    }
    fn get_label(&self) -> Option<String> {
        self.base.get_label()
    }
    fn set_component_control(&self, control: bool, state: Option<&'static ControlState>) {
        self.base.set_component_control(control, state)
    }
    fn set_enable_control(&self, control: bool, state: Option<&'static ControlState>) {
        self.base.set_enable_control(control, state)
    }
    fn send_control_event(&self) {
        self.base.send_control_event()
    }
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool {
        self.base.is_local_dir(current_directory)
    }
}
