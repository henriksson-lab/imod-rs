//! `IMOD/Etomo/src/etomo/ui/swing/FileButtonCell.java`.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::utilities;

use super::action_target::ActionTarget;
use super::field_lock_controller::{FieldLockController, JButton};
use super::file_chooser::{FileChooser, FileChooserReturnValue, FileChooserSelectionMode};
use super::file_text_field_interface::FileFilter;
use super::panel::Dimension;

/// Java `ScaledImage` choice used by this source unit's `SimpleButton`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileButtonScaledImage {
    OpenFilePeet,
    OpenFileFool,
}

/// `SimpleButton` state and native Swing boundary used by `FileButtonCell`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FileButtonBoundary {
    pub image: FileButtonScaledImage,
    pub border: Option<String>,
    pub preferred_size: Dimension,
    pub size: Dimension,
    pub name: Option<String>,
    pub tooltip: Option<String>,
    pub enabled: bool,
    pub action_listener_count: usize,
}

/// Java final package-private `FileButtonCell`.
pub struct FileButtonCell {
    /// Source retains a `BaseManager`; no FileButtonCell method reads it except
    /// while creating `FileChooser`, where the manager only supplies its user
    /// directory.  The native boundary consequently records the resolved
    /// manager directory without retaining a borrowed manager lifetime.
    pub manager_browsing_dir: Option<PathBuf>,
    pub action_target: Option<Rc<RefCell<dyn ActionTarget>>>,
    pub label: Option<String>,
    pub file_filter: Option<Rc<dyn FileFilter>>,
    pub browsing_dir: Option<Rc<dyn BrowsingDirectory>>,
    pub button: FileButtonBoundary,
    pub field_lock_controller: FieldLockController,
    pub background_refresh_count: usize,
    pub table_header: Option<String>,
    pub row_header: Option<String>,
    pub column_header: Option<String>,
}

impl FileButtonCell {
    /// Java private `FileButtonCell(BaseManager)`.
    fn new(manager: &dyn BaseManager) -> Self {
        let image = if *utilities::APRIL_FOOLS {
            FileButtonScaledImage::OpenFileFool
        } else {
            FileButtonScaledImage::OpenFilePeet
        };
        let preferred_size = Dimension {
            width: 22,
            height: 22,
        };
        let mut size = preferred_size;
        if size.width < size.height {
            size.width = size.height;
        }
        Self {
            manager_browsing_dir: manager.get_property_user_dir().map(PathBuf::from),
            action_target: None,
            label: None,
            file_filter: None,
            browsing_dir: None,
            button: FileButtonBoundary {
                image,
                border: Some("BevelBorder.RAISED".to_owned()),
                preferred_size,
                size,
                name: None,
                tooltip: None,
                enabled: true,
                action_listener_count: 0,
            },
            field_lock_controller: FieldLockController::get_button_instance(JButton {
                enabled: true,
            }),
            background_refresh_count: 0,
            table_header: None,
            row_header: None,
            column_header: None,
        }
    }

    /// Java static `getInstance(FileButtonCell)`.
    pub fn get_instance(file_button_cell: &FileButtonCell, manager: &dyn BaseManager) -> Self {
        let mut instance = Self::new(manager);
        instance.browsing_dir = file_button_cell.browsing_dir.clone();
        instance.label = file_button_cell.label.clone();
        instance.file_filter = file_button_cell.file_filter.clone();
        instance.add_listeners();
        instance
    }

    /// Java static `getInstance(BaseManager)`.
    pub fn get_instance_with_base_manager(manager: &dyn BaseManager) -> Self {
        let mut instance = Self::new(manager);
        instance.add_listeners();
        instance
    }

    /// Java `getText`.
    pub fn get_text(&self) -> Option<&str> {
        None
    }

    /// Java `setBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_browsing_directory(&mut self, input: Option<Rc<dyn BrowsingDirectory>>) {
        self.browsing_dir = input;
    }

    /// Java overridden `add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&self, constraints_weightx: f64) -> f64 {
        let _old_weightx = constraints_weightx;
        0.0
    }

    /// Java private `addListeners`.
    fn add_listeners(&mut self) {
        self.button.action_listener_count += 1;
    }

    /// Java `setActionTarget(ActionTarget)`.
    pub fn set_action_target(&mut self, input: Option<Rc<RefCell<dyn ActionTarget>>>) {
        self.action_target = input;
    }

    /// Java overridden `setHeaders(String, HeaderCell, HeaderCell)`.
    pub fn set_headers(
        &mut self,
        table_header: Option<&str>,
        row_header: Option<&str>,
        column_header: Option<&str>,
    ) {
        if self.label.is_none() {
            self.label = column_header.map(str::to_owned);
        }
        self.table_header = table_header.map(str::to_owned);
        self.row_header = row_header.map(str::to_owned);
        self.column_header = column_header.map(str::to_owned);
    }

    /// Java overridden `setName(String, String, String)`, deliberately empty
    /// in the original source.
    pub fn set_name_three(
        &mut self,
        _reference1: Option<&str>,
        _reference2: Option<&str>,
        _reference3: Option<&str>,
    ) {
    }

    /// Java overridden `setName()`.
    pub fn set_name(&mut self) {
        let name = utilities::convert_label_to_name(self.label.as_deref(), true);
        self.button.name = name.map(|name| format!("bn{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.button.name.as_deref().unwrap_or_default(),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `getName`.
    pub fn get_name(&self) -> Option<&str> {
        self.button.name.as_deref()
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&mut self, input: Option<&str>) {
        self.label = input.map(str::to_owned);
    }

    /// Java `setFileFilter(FileFilter)` and `setFileFilter(ExtensibleFileFilter)`.
    pub fn set_file_filter(&mut self, input: Option<Rc<dyn FileFilter>>) {
        self.file_filter = input;
    }

    /// Java `getFileFilter`.
    pub fn get_file_filter(&self) -> Option<&Rc<dyn FileFilter>> {
        self.file_filter.as_ref()
    }

    /// Java overridden `getComponent`.
    pub fn get_component(&self) -> &FileButtonBoundary {
        &self.button
    }

    /// Java overridden `getFieldType` (`UITestFieldType.BUTTON`).
    pub fn get_field_type(&self) -> &'static str {
        "bn"
    }

    /// Java overridden `getWidth`.
    pub fn get_width(&self) -> i32 {
        self.button.size.width
    }

    /// Java overridden `setLocked`.
    pub fn set_locked(&mut self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.button.enabled = self
                .field_lock_controller
                .button
                .as_ref()
                .expect("FileButtonCell owns Java JButton")
                .enabled;
            self.set_background();
        }
    }

    /// Java overridden `setEditable`.
    pub fn set_editable(&mut self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.button.enabled = self
                .field_lock_controller
                .button
                .as_ref()
                .expect("FileButtonCell owns Java JButton")
                .enabled;
            self.set_background();
        }
    }

    /// Java overridden `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
        self.button.enabled = self
            .field_lock_controller
            .button
            .as_ref()
            .expect("FileButtonCell owns Java JButton")
            .enabled;
    }

    /// Java `isLocked`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `isEditable`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java inherited `setBackground`, at the separately translated InputCell
    /// presentation boundary.
    pub fn set_background(&mut self) {
        self.background_refresh_count += 1;
    }

    /// Java private `action`, with `showOpenDialog` supplied by the native GUI
    /// adapter as its selected file.  The same chooser construction, filter,
    /// title, file-only mode, target assignment, and browsing-directory update
    /// order are retained.
    pub fn action(&mut self, selected_file: Option<&Path>) -> FileChooser {
        let target_value = self
            .action_target
            .as_ref()
            .map(|target| target.borrow().get_expanded_value());
        let mut chooser = FileChooser::new_with_browsing_dir(target_value.as_deref());
        if chooser.current_directory.is_none() {
            chooser.current_directory = self
                .browsing_dir
                .as_ref()
                .and_then(|browsing_dir| browsing_dir.get_browsing_dir())
                .or_else(|| self.manager_browsing_dir.clone());
        }
        chooser.set_dialog_title(Some(self.label.as_deref().unwrap_or("Open File")));
        chooser.set_preferred_size(Dimension {
            width: 0,
            height: 0,
        });
        chooser.set_file_selection_mode(FileChooserSelectionMode::FilesOnly);
        chooser.set_file_filter(self.file_filter.clone());
        if chooser.show_open_dialog(selected_file) == FileChooserReturnValue::ApproveOption {
            let file = chooser.get_selected_file();
            if let Some(action_target) = &self.action_target {
                action_target.borrow_mut().set_target_file(file.as_deref());
            }
            if let (Some(browsing_dir), Some(file)) = (&self.browsing_dir, file.as_ref()) {
                browsing_dir.set_browsing_dir(file.parent());
            }
        }
        chooser
    }

    /// Java overridden `setToolTipText(String)`; formatter rendering remains
    /// the native presentation boundary.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.button.tooltip = text.map(str::to_owned);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
    use crate::imod::etomo::storage::storable::Storable;
    use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
    use crate::imod::etomo::r#type::interface_type::InterfaceType;
    use std::convert::Infallible;

    struct Manager(BaseManagerBase);
    impl BaseManager for Manager {
        fn base(&self) -> &BaseManagerBase {
            &self.0
        }
        fn this(&'static self) -> &'static dyn BaseManager {
            self
        }
        fn get_interface_type(&self) -> Option<InterfaceType> {
            None
        }
        fn create_main_panel(&self) {}
        fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
            None
        }
        fn get_main_panel(&self) -> Option<Infallible> {
            None
        }
        fn get_process_manager(&self) -> Option<Infallible> {
            None
        }
        fn get_storables_with_offset(&self, _: i32) -> Option<Vec<Box<dyn Storable>>> {
            None
        }
        fn get_name(&self) -> Option<String> {
            None
        }
    }
    struct Target(Option<PathBuf>);
    impl ActionTarget for Target {
        fn set_target_file(&mut self, file: Option<&Path>) {
            self.0 = file.map(Path::to_path_buf);
        }
        fn get_expanded_value(&self) -> String {
            self.0
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string())
        }
    }
    struct Browsing(Mutex<Option<PathBuf>>);
    impl BrowsingDirectory for Browsing {
        fn get_browsing_dir(&self) -> Option<PathBuf> {
            self.0.lock().unwrap().clone()
        }
        fn set_browsing_dir(&self, file: Option<&Path>) {
            *self.0.lock().unwrap() = file.map(Path::to_path_buf);
        }
    }

    #[test]
    fn factory_and_headers_preserve_listener_icon_border_and_label_rules() {
        let manager = Manager(BaseManagerBase::initial());
        let mut cell = FileButtonCell::get_instance_with_base_manager(&manager);
        cell.set_headers(Some("Table"), Some("Row"), Some("Column"));
        cell.set_name();
        assert_eq!(cell.button.action_listener_count, 1);
        assert_eq!(cell.button.border.as_deref(), Some("BevelBorder.RAISED"));
        assert_eq!(cell.label.as_deref(), Some("Column"));
        assert_eq!(cell.get_name(), Some("bn.column"));
        assert_eq!(cell.get_text(), None);
        assert_eq!(cell.add(8.0), 0.0);
    }

    #[test]
    fn action_assigns_only_approved_file_and_updates_browsing_parent_after_target() {
        let manager = Manager(BaseManagerBase::initial());
        let mut cell = FileButtonCell::get_instance_with_base_manager(&manager);
        let target = Rc::new(RefCell::new(Target(None)));
        let browsing = Rc::new(Browsing(Mutex::new(None)));
        cell.set_action_target(Some(target.clone()));
        cell.set_browsing_directory(Some(browsing.clone()));
        cell.set_label(Some("Open reconstruction"));
        let chooser = cell.action(Some(Path::new("/tmp/a/rec.mrc")));
        assert_eq!(chooser.dialog_title.as_deref(), Some("Open reconstruction"));
        assert_eq!(
            chooser.file_selection_mode,
            FileChooserSelectionMode::FilesOnly
        );
        assert_eq!(target.borrow().0, Some(PathBuf::from("/tmp/a/rec.mrc")));
        assert_eq!(browsing.get_browsing_dir(), Some(PathBuf::from("/tmp/a")));
        cell.action(None);
        assert_eq!(target.borrow().0, Some(PathBuf::from("/tmp/a/rec.mrc")));
    }

    #[test]
    fn lock_editable_and_enabled_routes_follow_field_lock_controller() {
        let manager = Manager(BaseManagerBase::initial());
        let mut cell = FileButtonCell::get_instance_with_base_manager(&manager);
        cell.set_locked(true);
        assert!(cell.is_locked());
        assert!(!cell.button.enabled);
        assert_eq!(cell.background_refresh_count, 1);
        cell.set_enabled(false);
        assert!(!cell.is_enabled());
        assert!(!cell.button.enabled);
    }
}
