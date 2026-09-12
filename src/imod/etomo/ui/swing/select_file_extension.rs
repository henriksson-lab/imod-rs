//! `IMOD/Etomo/src/etomo/ui/swing/SelectFileExtension.java`.
#![allow(dead_code)]
use super::file_text_field_interface::FileFilter;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

/// Java final `SelectFileExtension`.  The chooser's native dialog is supplied by
/// the frontend as its selected-file result; this source unit owns selection setup and
/// directory updates, exactly as Java does after `showOpenDialog` returns.
pub struct SelectFileExtension {
    pub dir: Option<String>,
    pub alt_browsing_directory: Option<Rc<dyn BrowsingDirectory>>,
    pub file_filter: Option<Rc<dyn FileFilter>>,
    pub file_selection_mode: i32,
    pub file_chooser_title: Option<String>,
    pub last_chooser_directory: Option<PathBuf>,
    pub multiple_selection_enabled: bool,
}
impl Default for SelectFileExtension {
    fn default() -> Self {
        Self {
            dir: None,
            alt_browsing_directory: None,
            file_filter: None,
            file_selection_mode: -1,
            file_chooser_title: None,
            last_chooser_directory: None,
            multiple_selection_enabled: false,
        }
    }
}
impl SelectFileExtension {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set_dir(&mut self, dir: impl Into<String>) {
        self.dir = Some(dir.into());
    }
    pub fn set_alt_browsing_directory(&mut self, value: Rc<dyn BrowsingDirectory>) {
        self.alt_browsing_directory = Some(value);
    }
    pub fn set_file_filter(&mut self, value: Rc<dyn FileFilter>) {
        self.file_filter = Some(value);
    }
    pub fn set_file_selection_mode(&mut self, value: i32) {
        self.file_selection_mode = value;
    }
    pub fn set_file_chooser_title(&mut self, value: impl Into<String>) {
        self.file_chooser_title = Some(value.into());
    }
    /// Java private `getDirectory(File)`.
    pub fn get_directory(&self, override_file_open_directory: Option<&Path>) -> Option<PathBuf> {
        if let Some(path) = override_file_open_directory.filter(|path| path.is_dir()) {
            return Some(path.to_path_buf());
        }
        if let Some(directory) = &self.alt_browsing_directory {
            if let Some(path) = directory.get_browsing_dir().filter(|path| path.is_dir()) {
                return Some(path);
            }
        }
        self.dir.as_deref().map(PathBuf::from)
    }
    /// Java `selectFile(Component, File)`, with `selected_file` supplied by native GUI.
    pub fn select_file(
        &mut self,
        override_file_open_directory: Option<&Path>,
        selected_file: Option<PathBuf>,
    ) -> Option<PathBuf> {
        self.last_chooser_directory = self.get_directory(override_file_open_directory);
        self.multiple_selection_enabled = false;
        let selected_file = selected_file?;
        if let Some(parent) = selected_file.parent() {
            self.set_dir(parent.to_string_lossy());
            if let Some(directory) = &self.alt_browsing_directory {
                directory.set_browsing_dir(Some(parent));
            }
        }
        Some(selected_file)
    }
    /// Java `selectMultipleFiles(Component, File)`, with native selected values explicit.
    pub fn select_multiple_files(
        &mut self,
        override_file_open_directory: Option<&Path>,
        selected_files: Option<Vec<PathBuf>>,
    ) -> Option<Vec<PathBuf>> {
        self.last_chooser_directory = self.get_directory(override_file_open_directory);
        self.multiple_selection_enabled = true;
        let selected_files = selected_files.filter(|value| !value.is_empty())?;
        if let Some(parent) = selected_files[0].parent() {
            self.set_dir(parent.to_string_lossy());
            if let Some(directory) = &self.alt_browsing_directory {
                directory.set_browsing_dir(Some(parent));
            }
        }
        Some(selected_files)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    struct Browsing(Mutex<Option<PathBuf>>);
    impl BrowsingDirectory for Browsing {
        fn get_browsing_dir(&self) -> Option<PathBuf> {
            self.0.lock().unwrap().clone()
        }
        fn set_browsing_dir(&self, path: Option<&Path>) {
            *self.0.lock().unwrap() = path.map(Path::to_path_buf)
        }
    }
    #[test]
    fn selected_file_updates_source_directory_and_shared_browsing_directory() {
        let browsing = Rc::new(Browsing(Mutex::new(None)));
        let mut value = SelectFileExtension::new();
        value.set_alt_browsing_directory(browsing.clone());
        let selected = value
            .select_file(None, Some(PathBuf::from("/tmp/a.mrc")))
            .unwrap();
        assert_eq!(selected, PathBuf::from("/tmp/a.mrc"));
        assert_eq!(value.dir.as_deref(), Some("/tmp"));
        assert_eq!(browsing.get_browsing_dir(), Some(PathBuf::from("/tmp")));
    }
}
