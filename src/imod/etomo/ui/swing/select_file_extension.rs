//! `IMOD/Etomo/src/etomo/ui/swing/SelectFileExtension.java`.
//!
//! The file-selection settings of an efield and the file chooser it opens.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

use super::file_chooser::FileChooser;
use super::ui_parameters::UIParameters;
use crate::imod::etomo::jdk::{FileFilter, JComponent, JFileChooser};
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::valid_directory::ValidDirectory;

/// Java `SelectFileExtension`.
pub struct SelectFileExtension {
    /// Java `dir`.
    dir: RefCell<Option<String>>,
    /// Java `altBrowsingDirectory`.
    alt_browsing_directory: RefCell<Option<Rc<dyn BrowsingDirectory>>>,
    /// Java `fileFilter`.
    file_filter: RefCell<Option<Rc<dyn FileFilter>>>,
    /// Java `fileSelectionMode`.
    file_selection_mode: Cell<i32>,
    /// Java `fileChooserTitle`.
    file_chooser_title: RefCell<Option<String>>,
}

impl SelectFileExtension {
    /// Java `SelectFileExtension()`.
    pub fn new() -> Rc<SelectFileExtension> {
        Rc::new(SelectFileExtension {
            dir: RefCell::new(None),
            alt_browsing_directory: RefCell::new(None),
            file_filter: RefCell::new(None),
            file_selection_mode: Cell::new(-1),
            file_chooser_title: RefCell::new(None),
        })
    }

    /// Java `setDir(String)`.
    pub fn set_dir(&self, dir: Option<&str>) {
        *self.dir.borrow_mut() = dir.map(str::to_owned);
    }

    /// Java `setAltBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_alt_browsing_directory(
        &self,
        alt_browsing_directory: Option<Rc<dyn BrowsingDirectory>>,
    ) {
        *self.alt_browsing_directory.borrow_mut() = alt_browsing_directory;
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&self, file_filter: Option<Rc<dyn FileFilter>>) {
        *self.file_filter.borrow_mut() = file_filter;
    }

    /// Java `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&self, file_selection_mode: i32) {
        self.file_selection_mode.set(file_selection_mode);
    }

    /// Java `selectFile(Component, File)`.
    pub fn select_file(
        &self,
        component: Option<&Rc<JComponent>>,
        override_file_open_directory: Option<PathBuf>,
    ) -> Option<PathBuf> {
        let chooser = FileChooser::new_file(self.get_directory(override_file_open_directory).as_deref());
        let file_chooser_title = self.file_chooser_title.borrow().clone();
        if let Some(file_chooser_title) = file_chooser_title.as_deref() {
            chooser.set_dialog_title(Some(file_chooser_title));
            chooser.set_name(Some(file_chooser_title));
        } else {
            chooser.set_dialog_title(Some("Select File"));
        }
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        let _ = UIParameters::get_instance_void().get_file_chooser_dimension();
        if self.file_selection_mode.get() != -1 {
            chooser.set_file_selection_mode(self.file_selection_mode.get());
        }
        let file_filter = self.file_filter.borrow().clone();
        if file_filter.is_some() {
            chooser.set_file_filter(file_filter);
        }
        if chooser.show_open_dialog(component) == JFileChooser::APPROVE_OPTION {
            let selected_file = chooser.get_selected_file()?;
            // Java selectedFile.getParentFile().getAbsolutePath(): a chooser selection
            // is absolute, so it has a parent.
            let parent = selected_file.parent().map(|parent| parent.to_path_buf());
            self.set_dir(
                parent
                    .as_ref()
                    .map(|parent| std::path::absolute(parent).unwrap_or(parent.clone()))
                    .as_deref()
                    .and_then(|parent| parent.to_str()),
            );
            let alt_browsing_directory = self.alt_browsing_directory.borrow().clone();
            if let Some(alt_browsing_directory) = alt_browsing_directory {
                alt_browsing_directory.set_browsing_dir(parent.as_deref());
            }
            return Some(selected_file);
        }
        None
    }

    /// Java `selectMultipleFiles(Component, File)` (ALIGN_FRAMES).
    pub fn select_multiple_files(
        &self,
        component: Option<&Rc<JComponent>>,
        override_file_open_directory: Option<PathBuf>,
    ) -> Option<Vec<PathBuf>> {
        let chooser = JFileChooser::new_file(self.get_directory(override_file_open_directory).as_deref());
        let file_chooser_title = self.file_chooser_title.borrow().clone();
        if let Some(file_chooser_title) = file_chooser_title.as_deref() {
            chooser.set_dialog_title(Some(file_chooser_title));
        } else {
            chooser.set_dialog_title(Some("Select File"));
        }
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        let _ = UIParameters::get_instance_void().get_file_chooser_dimension();
        chooser.set_multi_selection_enabled(true);
        if self.file_selection_mode.get() != -1 {
            chooser.set_file_selection_mode(self.file_selection_mode.get());
        }
        let file_filter = self.file_filter.borrow().clone();
        if file_filter.is_some() {
            chooser.set_file_filter(file_filter);
        }
        if chooser.show_open_dialog(component) == JFileChooser::APPROVE_OPTION {
            let selected_files = chooser.get_selected_files();
            if !selected_files.is_empty() {
                let parent = selected_files[0].parent().map(|parent| parent.to_path_buf());
                self.set_dir(
                    parent
                        .as_ref()
                        .map(|parent| std::path::absolute(parent).unwrap_or(parent.clone()))
                        .as_deref()
                        .and_then(|parent| parent.to_str()),
                );
                let alt_browsing_directory = self.alt_browsing_directory.borrow().clone();
                if let Some(alt_browsing_directory) = alt_browsing_directory {
                    alt_browsing_directory.set_browsing_dir(parent.as_deref());
                }
                return Some(selected_files);
            }
        }
        None
    }

    /// Java private `getDirectory(File)`.
    fn get_directory(&self, override_file_open_directory: Option<PathBuf>) -> Option<PathBuf> {
        // TODO(unit): needs etomo/util/ValidDirectory.java - static isValid(File),
        // isValid(BrowsingDirectory), get(File), get(BrowsingDirectory).
        if ValidDirectory::is_valid_file(override_file_open_directory.as_deref()) {
            return ValidDirectory::get_file(override_file_open_directory.as_deref());
        }
        let alt_browsing_directory = self.alt_browsing_directory.borrow().clone();
        if ValidDirectory::is_valid_browsing_directory(alt_browsing_directory.as_deref()) {
            return ValidDirectory::get_browsing_directory(alt_browsing_directory.as_deref());
        }
        if let Some(dir) = self.dir.borrow().as_deref() {
            return Some(PathBuf::from(dir));
        }
        None
    }

    /// Java `setFileChooserTitle(String)`.
    pub fn set_file_chooser_title(&self, input: Option<&str>) {
        *self.file_chooser_title.borrow_mut() = input.map(str::to_owned);
    }
}
