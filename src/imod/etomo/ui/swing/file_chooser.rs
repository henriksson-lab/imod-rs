//! `IMOD/Etomo/src/etomo/ui/swing/FileChooser.java`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::DEFAULT_DELIMITER;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::utilities;

use super::file_text_field_interface::FileFilter;
use super::panel::Dimension;

/// Java `FileChooser.DEFAULT_TITLE`.
pub const DEFAULT_TITLE: &str = "Open";

/// Native frontend result of `JFileChooser.show*Dialog`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileChooserReturnValue {
    ApproveOption,
    CancelOption,
}

/// Java `JFileChooser` dialog mode retained at the native widget boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileChooserDialogType {
    OpenDialog,
    SaveDialog,
}

/// Java `JFileChooser` selection mode retained at the native widget boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileChooserSelectionMode {
    FilesOnly,
    DirectoriesOnly,
    FilesAndDirectories,
}

/// Java final `FileChooser`.
///
/// The inherited native chooser is represented by its source-observed state.
/// `BaseManager` and `AxisID` are stored in Java but are not read by any method
/// in this source unit after construction, so the Rust boundary retains the
/// manager-presence fact and resolved initial directory rather than a borrowed
/// manager lifetime.
pub struct FileChooser {
    pub manager_present: bool,
    pub axis_id_present: bool,
    pub current_directory: Option<PathBuf>,
    pub dialog_title: Option<String>,
    pub name: Option<String>,
    pub dialog_type: FileChooserDialogType,
    pub preferred_size: Option<Dimension>,
    pub file_selection_mode: FileChooserSelectionMode,
    pub file_filter: Option<Rc<dyn FileFilter>>,
    pub selected_file: Option<PathBuf>,
}

impl FileChooser {
    /// Java `FileChooser()`.
    pub fn new() -> Self {
        Self::new_with_manager(None, None, None, None)
    }

    /// Java `FileChooser(BaseManager)`.
    pub fn new_with_base_manager(manager: &dyn BaseManager) -> Self {
        Self::new_with_manager(Some(manager), None, None, None)
    }

    /// Java `FileChooser(String)`.
    pub fn new_with_browsing_dir(browsing_dir: Option<&str>) -> Self {
        Self::new_with_manager(None, None, browsing_dir, None)
    }

    /// Java `FileChooser(File)`.
    pub fn new_with_browsing_file(browsing_dir: Option<&Path>) -> Self {
        Self::new_with_browsing_dir(browsing_dir.and_then(Path::to_str))
    }

    /// Java `FileChooser(BaseManager, File)`.
    pub fn new_with_base_manager_and_browsing_file(
        manager: &dyn BaseManager,
        browsing_dir: Option<&Path>,
    ) -> Self {
        Self::new_with_manager(
            Some(manager),
            None,
            browsing_dir.and_then(Path::to_str),
            None,
        )
    }

    /// Java `FileChooser(BaseManager, String)`.
    pub fn new_with_base_manager_and_browsing_dir(
        manager: &dyn BaseManager,
        browsing_dir: Option<&str>,
    ) -> Self {
        Self::new_with_manager(Some(manager), None, browsing_dir, None)
    }

    /// Java `FileChooser(BaseManager, File, BrowsingDirectory)`.
    pub fn new_with_base_manager_browsing_file_and_alt_browsing_dir(
        manager: &dyn BaseManager,
        browsing_dir: Option<&Path>,
        alt_browsing_dir: Option<&dyn BrowsingDirectory>,
    ) -> Self {
        Self::new_with_manager(
            Some(manager),
            None,
            browsing_dir.and_then(Path::to_str),
            alt_browsing_dir,
        )
    }

    /// Java `FileChooser(BrowsingDirectory)`.
    pub fn new_with_alt_browsing_dir(alt_browsing_dir: Option<&dyn BrowsingDirectory>) -> Self {
        Self::new_with_manager(None, None, None, alt_browsing_dir)
    }

    /// Java `FileChooser(BaseManager, AxisID, String, BrowsingDirectory)`.
    pub fn new_with_manager(
        manager: Option<&dyn BaseManager>,
        axis_id: Option<()>,
        browsing_dir: Option<&str>,
        alt_browsing_dir: Option<&dyn BrowsingDirectory>,
    ) -> Self {
        let mut chooser = Self {
            manager_present: manager.is_some(),
            axis_id_present: axis_id.is_some(),
            current_directory: Self::get_browsing_dir(manager, browsing_dir, alt_browsing_dir),
            dialog_title: None,
            name: None,
            dialog_type: FileChooserDialogType::OpenDialog,
            preferred_size: None,
            file_selection_mode: FileChooserSelectionMode::FilesAndDirectories,
            file_filter: None,
            selected_file: None,
        };
        chooser.set_name(DEFAULT_TITLE);
        chooser
    }

    /// Java private static `getBrowsingDir(BaseManager, String, BrowsingDirectory)`.
    pub fn get_browsing_dir(
        manager: Option<&dyn BaseManager>,
        browsing_dir: Option<&str>,
        alt_browsing_dir: Option<&dyn BrowsingDirectory>,
    ) -> Option<PathBuf> {
        if let Some(browsing_dir) = browsing_dir.filter(|value| !value.trim().is_empty()) {
            let directory = PathBuf::from(browsing_dir);
            let directory = if directory.is_dir() {
                Some(directory)
            } else {
                directory.parent().map(Path::to_path_buf)
            };
            if let Some(directory) = directory.filter(|directory| directory.is_dir()) {
                return Some(directory);
            }
        }
        if let Some(alt_browsing_dir) = alt_browsing_dir {
            return alt_browsing_dir.get_browsing_dir();
        }
        manager
            .and_then(BaseManager::get_property_user_dir)
            .map(PathBuf::from)
    }

    /// Java overridden `setDialogTitle(String)`.
    pub fn set_dialog_title(&mut self, dialog_title: Option<&str>) {
        self.dialog_title = dialog_title.map(str::to_owned);
        self.set_name(dialog_title.unwrap_or_default());
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, text: &str) {
        self.name = utilities::convert_label_to_name(Some(text), true);
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.name.as_deref().unwrap_or_default(),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java inherited `setDialogType(int)`.
    pub fn set_dialog_type(&mut self, dialog_type: FileChooserDialogType) {
        self.dialog_type = dialog_type;
    }

    /// Java inherited `setPreferredSize(Dimension)`.
    pub fn set_preferred_size(&mut self, preferred_size: Dimension) {
        self.preferred_size = Some(preferred_size);
    }

    /// Java inherited `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&mut self, file_selection_mode: FileChooserSelectionMode) {
        self.file_selection_mode = file_selection_mode;
    }

    /// Java inherited `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&mut self, file_filter: Option<Rc<dyn FileFilter>>) {
        self.file_filter = file_filter;
    }

    /// Java inherited `setSelectedFile(File)`.
    pub fn set_selected_file(&mut self, selected_file: Option<&Path>) {
        self.selected_file = selected_file.map(Path::to_path_buf);
    }

    /// Java inherited `getSelectedFile()`.
    pub fn get_selected_file(&self) -> Option<PathBuf> {
        self.selected_file.clone()
    }

    /// Java inherited `showOpenDialog(Component)`, at the native dialog boundary.
    pub fn show_open_dialog(&mut self, selected_file: Option<&Path>) -> FileChooserReturnValue {
        self.selected_file = selected_file.map(Path::to_path_buf);
        if self.selected_file.is_some() {
            FileChooserReturnValue::ApproveOption
        } else {
            FileChooserReturnValue::CancelOption
        }
    }

    /// Java inherited `showSaveDialog(Component)`, at the native dialog boundary.
    pub fn show_save_dialog(&mut self, selected_file: Option<&Path>) -> FileChooserReturnValue {
        self.selected_file = selected_file.map(Path::to_path_buf);
        if self.selected_file.is_some() {
            FileChooserReturnValue::ApproveOption
        } else {
            FileChooserReturnValue::CancelOption
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

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
    fn constructors_and_name_override_preserve_source_default_and_title_paths() {
        let mut chooser = FileChooser::new();
        assert_eq!(chooser.name.as_deref(), Some("open"));
        assert_eq!(chooser.dialog_type, FileChooserDialogType::OpenDialog);
        assert_eq!(
            chooser.file_selection_mode,
            FileChooserSelectionMode::FilesAndDirectories
        );
        chooser.set_dialog_title(Some("Open reference volume"));
        assert_eq!(
            chooser.dialog_title.as_deref(),
            Some("Open reference volume")
        );
        assert_eq!(chooser.name.as_deref(), Some("open-reference-volume"));
    }

    #[test]
    fn browsing_directory_precedence_matches_java_get_browsing_dir() {
        let alt = Browsing(Mutex::new(Some(PathBuf::from("/tmp"))));
        assert_eq!(
            FileChooser::get_browsing_dir(None, Some(" \t"), Some(&alt)),
            Some(PathBuf::from("/tmp"))
        );
        assert_eq!(
            FileChooser::get_browsing_dir(None, Some("/definitely/not/a/directory"), Some(&alt)),
            Some(PathBuf::from("/tmp"))
        );
    }

    #[test]
    fn native_open_and_save_boundaries_retain_selected_file_and_result() {
        let mut chooser = FileChooser::new();
        assert_eq!(
            chooser.show_open_dialog(Some(Path::new("/tmp/input.mrc"))),
            FileChooserReturnValue::ApproveOption
        );
        assert_eq!(
            chooser.get_selected_file(),
            Some(PathBuf::from("/tmp/input.mrc"))
        );
        assert_eq!(
            chooser.show_save_dialog(None),
            FileChooserReturnValue::CancelOption
        );
        assert_eq!(chooser.get_selected_file(), None);
    }
}
