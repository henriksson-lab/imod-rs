//! `IMOD/Etomo/src/etomo/ui/swing/ButtonControlTextEfield.java`.
//!
//! `Ebutton`, `SelectFileExtension`, Swing's `FileFilter`, and the Swing
//! container are distinct Java source units.  Their state is deliberately
//! retained as narrow boundaries here: this unit owns the construction and
//! forwarding decisions made by `ButtonControlTextEfield`, without inventing
//! a second file-chooser implementation.
#![allow(dead_code)]

use std::{cell::RefCell, path::PathBuf, rc::Rc};

use crate::imod::etomo::ui::{browsing_directory::BrowsingDirectory, field_type::FieldType};

use super::text_efield::TextEfield;

/// Java `javax.swing.filechooser.FileFilter`, whose selection predicate stays
/// at the Swing boundary.
pub trait FileFilter {}

/// State from the separately sourced Java `SelectFileExtension` consumed by
/// this source unit.
pub struct SelectFileExtension {
    pub dir: Option<String>,
    pub alt_browsing_directory: Option<Rc<dyn BrowsingDirectory>>,
    pub file_filter: Option<Rc<dyn FileFilter>>,
    pub file_selection_mode: i32,
}

impl Default for SelectFileExtension {
    fn default() -> Self {
        Self {
            dir: None,
            alt_browsing_directory: None,
            file_filter: None,
            file_selection_mode: -1,
        }
    }
}

impl SelectFileExtension {
    /// Java package-private `SelectFileExtension()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Java `setDir(String)`.
    pub fn set_dir(&mut self, dir: impl Into<String>) {
        self.dir = Some(dir.into());
    }

    /// Java `setAltBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_alt_browsing_directory(&mut self, browsing_directory: Rc<dyn BrowsingDirectory>) {
        self.alt_browsing_directory = Some(browsing_directory);
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&mut self, file_filter: Rc<dyn FileFilter>) {
        self.file_filter = Some(file_filter);
    }

    /// Java `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&mut self, file_selection_mode: i32) {
        self.file_selection_mode = file_selection_mode;
    }
}

/// The subset of Java `Ebutton` reached by this source unit.  The physical
/// `JButton` and its file chooser are GUI boundaries.
pub struct ButtonControlTextEfieldButtonBoundary {
    pub select_multiple_files: bool,
    pub debug: bool,
    pub tooltip: Option<String>,
    pub override_file_open_directory: Option<PathBuf>,
    pub select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
    pub enabled: bool,
    pub editable: bool,
    pub flag_display: bool,
}

impl ButtonControlTextEfieldButtonBoundary {
    /// Java `Ebutton.getSelectFileInstance` / `getSelectMultipleFilesInstance`.
    fn get_select_file_instance(
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        select_multiple_files: bool,
        debug: bool,
    ) -> Self {
        Self {
            select_multiple_files,
            debug,
            tooltip: None,
            override_file_open_directory: None,
            select_file_extension: Some(
                shared_select_file_extension
                    .unwrap_or_else(|| Rc::new(RefCell::new(SelectFileExtension::new()))),
            ),
            enabled: true,
            editable: true,
            flag_display: false,
        }
    }

    /// Java `Ebutton.getClearInstance`.
    fn get_clear_instance() -> Self {
        Self {
            select_multiple_files: false,
            debug: false,
            tooltip: None,
            override_file_open_directory: None,
            select_file_extension: None,
            enabled: true,
            editable: true,
            flag_display: false,
        }
    }

    /// Java `Ebutton.setDebug(boolean)`.
    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `Ebutton.setTooltip(String)`.
    fn set_tooltip(&mut self, text: impl Into<String>) {
        self.tooltip = Some(text.into());
    }

    /// Java `Ebutton.setOverrideFileOpenDirectory(File)`.
    fn set_override_file_open_directory(&mut self, override_file_open_directory: PathBuf) {
        self.override_file_open_directory = Some(override_file_open_directory);
    }

    /// Java `Ebutton.setFileFilter(FileFilter)`.
    fn set_file_filter(&mut self, file_filter: Rc<dyn FileFilter>) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension
                .borrow_mut()
                .set_file_filter(file_filter);
        }
    }

    /// Java `Ebutton.setFileSelectionMode(int)`.
    fn set_file_selection_mode(&mut self, file_selection_mode: i32) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension
                .borrow_mut()
                .set_file_selection_mode(file_selection_mode);
        }
    }

    /// Java `Ebutton.setSelectFileDir(String)`.
    fn set_select_file_dir(&mut self, dir: impl Into<String>) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension.borrow_mut().set_dir(dir);
        }
    }

    /// Java `Ebutton.getSelectFileExtension()`.
    fn get_select_file_extension(&self) -> Option<Rc<RefCell<SelectFileExtension>>> {
        self.select_file_extension.clone()
    }

    /// Java `Ebutton.setAltBrowsingDirectory(BrowsingDirectory)`.
    fn set_alt_browsing_directory(&mut self, browsing_directory: Rc<dyn BrowsingDirectory>) {
        if let Some(select_file_extension) = &self.select_file_extension {
            select_file_extension
                .borrow_mut()
                .set_alt_browsing_directory(browsing_directory);
        }
    }
}

/// Java package-private final `ButtonControlTextEfield`.
pub struct ButtonControlTextEfield {
    /// Java superclass `TextEfield`.
    pub text_efield: TextEfield,
    /// Java final `selectFileButton`.
    pub select_file_button: Option<Rc<RefCell<ButtonControlTextEfieldButtonBoundary>>>,
    /// Java final `clearButton`.
    pub clear_button: Option<Rc<RefCell<ButtonControlTextEfieldButtonBoundary>>>,
    /// Components added by this constructor to the inherited `container`.
    pub added_container_component_count: usize,
    /// Java inherited nullable `appearanceExtension`, represented at the GUI boundary.
    pub appearance_extension_present: bool,
    /// Java `appearanceExtension.setChildControllers(createChildControllers())`.
    pub appearance_child_controllers:
        Option<Vec<Option<Rc<RefCell<ButtonControlTextEfieldButtonBoundary>>>>>,
}

impl ButtonControlTextEfield {
    #[allow(clippy::too_many_arguments)]
    fn new(
        label: impl Into<String>,
        field_type: FieldType,
        use_label_component: bool,
        use_control_component: bool,
        enabled_field: bool,
        editable_component: bool,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        include_select_file_button: bool,
        include_clear_button: bool,
        use_grid_bag: bool,
        select_multiple_files: bool,
        default_to_file_name: bool,
        debug: bool,
    ) -> Self {
        let appearance_extension_present = !enabled_field || !editable_component;
        let text_efield = TextEfield::new(
            label,
            Some(field_type),
            use_label_component,
            use_control_component,
            enabled_field,
            editable_component,
            use_grid_bag,
            default_to_file_name,
        );
        let select_file_button = include_select_file_button.then(|| {
            Rc::new(RefCell::new(
                ButtonControlTextEfieldButtonBoundary::get_select_file_instance(
                    shared_select_file_extension,
                    select_multiple_files,
                    debug,
                ),
            ))
        });
        let clear_button = include_clear_button.then(|| {
            Rc::new(RefCell::new(
                ButtonControlTextEfieldButtonBoundary::get_clear_instance(),
            ))
        });
        let mut instance = Self {
            text_efield,
            select_file_button,
            clear_button,
            added_container_component_count: usize::from(include_select_file_button)
                + usize::from(include_clear_button),
            appearance_extension_present,
            appearance_child_controllers: None,
        };
        if instance.appearance_extension_present {
            instance.appearance_child_controllers = instance.create_child_controllers();
        }
        instance
    }

    /// Java `getFileOverrideInstance(String, SelectFileExtension, boolean)`.
    pub fn get_file_override_instance(
        label: impl Into<String>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        debug: bool,
    ) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getFileInstance(String, SelectFileExtension)`.
    pub fn get_file_instance_with_extension(
        label: impl Into<String>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
    ) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getFileInstance(String, SelectFileExtension, boolean, boolean)`.
    pub fn get_file_instance_with_options(
        label: impl Into<String>,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
        include_clear_button: bool,
        editable_component: bool,
    ) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getFileInstance(String)`.
    pub fn get_file_instance(label: impl Into<String>) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getLabeledFileInstance(String)`.
    pub fn get_labeled_file_instance(label: impl Into<String>) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getLabeledFileInstance(String, boolean, boolean)`.
    pub fn get_labeled_file_instance_with_options(
        label: impl Into<String>,
        include_clear_button: bool,
        use_grid_bag: bool,
    ) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getLabeledFileInstance(String, boolean, boolean, SelectFileExtension)`.
    pub fn get_labeled_file_instance_with_extension(
        label: impl Into<String>,
        include_clear_button: bool,
        use_grid_bag: bool,
        shared_select_file_extension: Option<Rc<RefCell<SelectFileExtension>>>,
    ) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java `getLabeledMultipleFilesInstance(String, boolean, boolean)`.
    pub fn get_labeled_multiple_files_instance(
        label: impl Into<String>,
        include_clear_button: bool,
        use_grid_bag: bool,
    ) -> Self {
        Self::new(
            label,
            FieldType::File,
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

    /// Java overridden `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.text_efield.set_debug(debug);
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button.borrow_mut().set_debug(debug);
        }
    }

    /// Java overridden `setTooltip(String)`.
    pub fn set_tooltip(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.text_efield.set_tooltip(text.clone());
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button.borrow_mut().set_tooltip(text.clone());
        }
        if let Some(clear_button) = &self.clear_button {
            clear_button.borrow_mut().set_tooltip(text);
        }
    }

    /// Java `setTooltip(String, String)`.
    pub fn set_tooltip_with_select_file_text(
        &mut self,
        text: impl Into<String>,
        select_file_text: impl Into<String>,
    ) {
        let text = text.into();
        self.text_efield.set_tooltip(text.clone());
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button
                .borrow_mut()
                .set_tooltip(select_file_text);
        }
        if let Some(clear_button) = &self.clear_button {
            clear_button.borrow_mut().set_tooltip(text);
        }
    }

    /// Java `setOverrideFileOpenDirectory(File)`.
    pub fn set_override_file_open_directory(&mut self, override_file_open_directory: PathBuf) {
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button
                .borrow_mut()
                .set_override_file_open_directory(override_file_open_directory);
        }
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&mut self, file_filter: Rc<dyn FileFilter>) {
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button.borrow_mut().set_file_filter(file_filter);
        }
    }

    /// Java `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&mut self, file_selection_mode: i32) {
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button
                .borrow_mut()
                .set_file_selection_mode(file_selection_mode);
        }
    }

    /// Java `setSelectFileDir(String)`.
    pub fn set_select_file_dir(&mut self, dir: impl Into<String>) {
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button.borrow_mut().set_select_file_dir(dir);
        }
    }

    /// Java `getSelectFileExtension()`.
    pub fn get_select_file_extension(&self) -> Option<Rc<RefCell<SelectFileExtension>>> {
        self.select_file_button
            .as_ref()
            .and_then(|button| button.borrow().get_select_file_extension())
    }

    /// Java overridden `createAppearanceExtension(boolean, boolean)`.
    pub fn create_appearance_extension(&mut self, enabled_field: bool, editable_component: bool) {
        let create = !self.appearance_extension_present;
        self.text_efield
            .create_appearance_extension(enabled_field, editable_component);
        if create {
            self.appearance_extension_present = true;
            self.appearance_child_controllers = self.create_child_controllers();
        }
    }

    /// Java private `createChildControllers()`.
    fn create_child_controllers(
        &self,
    ) -> Option<Vec<Option<Rc<RefCell<ButtonControlTextEfieldButtonBoundary>>>>> {
        let mut size = 0;
        if self.select_file_button.is_some() {
            size += 1;
        }
        if self.clear_button.is_some() {
            size += 1;
        }
        if size == 0 {
            return None;
        }
        let mut controllers = Vec::with_capacity(size);
        if let Some(select_file_button) = &self.select_file_button {
            controllers.push(Some(select_file_button.clone()));
        }
        if let Some(clear_button) = &self.clear_button {
            controllers.push(Some(clear_button.clone()));
        }
        Some(controllers)
    }

    /// Java overridden `createFlagExtension()`.
    pub fn create_flag_extension(&mut self) -> bool {
        if self.text_efield.create_flag_extension() {
            if let Some(select_file_button) = &self.select_file_button {
                select_file_button.borrow_mut().flag_display = true;
            }
            if let Some(clear_button) = &self.clear_button {
                clear_button.borrow_mut().flag_display = true;
            }
            return true;
        }
        false
    }

    /// Java `setAltBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_alt_browsing_directory(&mut self, browsing_directory: Rc<dyn BrowsingDirectory>) {
        if let Some(select_file_button) = &self.select_file_button {
            select_file_button
                .borrow_mut()
                .set_alt_browsing_directory(browsing_directory);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factories_preserve_the_source_button_and_text_field_configuration() {
        let file = ButtonControlTextEfield::get_file_instance("Input file");
        assert_eq!(file.text_efield.field_type, Some(FieldType::File));
        assert!(!file.text_efield.use_label);
        assert!(!file.text_efield.use_control_component);
        assert!(!file.text_efield.text_field.editable);
        assert_eq!(file.added_container_component_count, 2);
        assert!(file.select_file_button.is_some());
        assert!(file.clear_button.is_some());
        assert!(file.appearance_extension_present);
        assert_eq!(file.appearance_child_controllers.as_ref().unwrap().len(), 2);

        let multiple =
            ButtonControlTextEfield::get_labeled_multiple_files_instance("Frames", false, true);
        assert!(multiple.text_efield.use_label);
        assert!(multiple.text_efield.in_grid_bag);
        assert!(
            multiple
                .select_file_button
                .as_ref()
                .unwrap()
                .borrow()
                .select_multiple_files
        );
        assert!(multiple.clear_button.is_none());
    }

    #[test]
    fn tooltips_and_file_chooser_configuration_are_forwarded_only_to_select_button() {
        let mut field = ButtonControlTextEfield::get_file_instance("Input");
        field.set_tooltip_with_select_file_text("Field help", "Choose a file");
        field.set_debug(true);
        field.set_override_file_open_directory(PathBuf::from("/tmp/input"));
        field.set_file_selection_mode(1);
        field.set_select_file_dir("/work");

        assert_eq!(
            field.text_efield.text_field.tooltip.as_deref(),
            Some("Field help")
        );
        let select = field.select_file_button.as_ref().unwrap().borrow();
        assert_eq!(select.tooltip.as_deref(), Some("Choose a file"));
        assert!(select.debug);
        assert_eq!(
            select.override_file_open_directory,
            Some(PathBuf::from("/tmp/input"))
        );
        let extension = select.select_file_extension.as_ref().unwrap().borrow();
        assert_eq!(extension.file_selection_mode, 1);
        assert_eq!(extension.dir.as_deref(), Some("/work"));
        drop(extension);
        assert_eq!(
            field
                .clear_button
                .as_ref()
                .unwrap()
                .borrow()
                .tooltip
                .as_deref(),
            Some("Field help")
        );
    }

    #[test]
    fn flag_and_appearance_creation_follow_super_result_and_child_order() {
        let mut field =
            ButtonControlTextEfield::get_file_instance_with_options("Input", None, false, true);
        assert!(!field.appearance_extension_present);
        field.create_appearance_extension(true, false);
        assert!(field.appearance_extension_present);
        assert_eq!(
            field.appearance_child_controllers.as_ref().unwrap().len(),
            1
        );

        assert!(field.create_flag_extension());
        assert!(
            field
                .select_file_button
                .as_ref()
                .unwrap()
                .borrow()
                .flag_display
        );
        assert!(!field.create_flag_extension());
    }
}
