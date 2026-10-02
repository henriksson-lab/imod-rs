//! `IMOD/Etomo/src/etomo/ui/swing/FileChooser.java`.
//!
//! A self-naming `JFileChooser`.  The default title of a file chooser is
//! "Open", so instances are named "open" by default.
//!
//! `JFileChooser` is not a `jdk::JComponent` kind: this struct carries the
//! chooser state eTomo reads and writes (current directory, dialog title and
//! type, selection mode, filters, the selected file(s), the name) and the
//! dialog itself is a UI boundary.  `showOpenDialog`/`showSaveDialog` hand the
//! chooser to the responder installed with [`set_dialog_responder`] (the Slint
//! frontend, a test, or the click driver), which sets the selected file and
//! calls [`FileChooser::approve_selection`] or [`FileChooser::cancel_selection`],
//! exactly as the Swing dialog's buttons do.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::DEFAULT_DELIMITER;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
// TODO(unit): needs etomo/util/ValidDirectory.java - `new ValidDirectory(manager)`,
// `set(String)`, `isNull()`, `get()` in `getBrowsingDir`.
use crate::imod::etomo::util::utilities;
use crate::imod::etomo::util::valid_directory::ValidDirectory;

/// Java package-private `static final String DEFAULT_TITLE = "Open"`.
pub const DEFAULT_TITLE: &str = "Open";

/// Java `UITestFieldType.FILE_CHOOSER.isUnlimitedSegments()`.
// TODO(unit): needs etomo/type/UITestFieldType.java - `FILE_CHOOSER` is written out.
const FILE_CHOOSER_UNLIMITED_SEGMENTS: bool = true;

/// Java `JFileChooser.CANCEL_OPTION`.
pub const CANCEL_OPTION: i32 = 1;
/// Java `JFileChooser.APPROVE_OPTION`.
pub const APPROVE_OPTION: i32 = 0;
/// Java `JFileChooser.ERROR_OPTION`.
pub const ERROR_OPTION: i32 = -1;
/// Java `JFileChooser.OPEN_DIALOG`.
pub const OPEN_DIALOG: i32 = 0;
/// Java `JFileChooser.SAVE_DIALOG`.
pub const SAVE_DIALOG: i32 = 1;
/// Java `JFileChooser.CUSTOM_DIALOG`.
pub const CUSTOM_DIALOG: i32 = 2;
/// Java `JFileChooser.FILES_ONLY`.
pub const FILES_ONLY: i32 = 0;
/// Java `JFileChooser.DIRECTORIES_ONLY`.
pub const DIRECTORIES_ONLY: i32 = 1;
/// Java `JFileChooser.FILES_AND_DIRECTORIES`.
pub const FILES_AND_DIRECTORIES: i32 = 2;

/// What answers a shown chooser: it may set the selected file(s) and must call
/// `approve_selection` or `cancel_selection`; returning without either is the
/// window being closed (`CANCEL_OPTION`).
pub type DialogResponder = Rc<dyn Fn(&FileChooser, Option<&Rc<JComponent>>)>;

thread_local! {
    /// The installed dialog boundary (EDT-local, like the Swing dialog).
    static DIALOG_RESPONDER: RefCell<Option<DialogResponder>> = const { RefCell::new(None) };
}

/// Installs (or, with `None`, removes) the boundary that answers
/// `showOpenDialog`/`showSaveDialog`.  Not a Java member.
// TODO(unit): the file dialog is a UI boundary with no Java unit behind it;
// UIHarness / the Slint bridge should install the real dialog here.
pub fn set_dialog_responder(responder: Option<DialogResponder>) {
    DIALOG_RESPONDER.with(|slot| *slot.borrow_mut() = responder);
}

/// Java `public final class FileChooser extends JFileChooser`.
pub struct FileChooser {
    /// Java `private final BaseManager manager` (stored, not read again).
    manager: Option<&'static dyn BaseManager>,
    /// Java `private final AxisID axisID` (stored, not read again).
    axis_id: Option<AxisID>,
    // --- JFileChooser state ---
    name: RefCell<Option<String>>,
    current_directory: RefCell<Option<PathBuf>>,
    dialog_title: RefCell<Option<String>>,
    dialog_type: Cell<i32>,
    file_selection_mode: Cell<i32>,
    multi_selection_enabled: Cell<bool>,
    file_hiding_enabled: Cell<bool>,
    control_buttons_are_shown: Cell<bool>,
    file_filter: RefCell<Option<Rc<dyn FileFilter>>>,
    choosable_file_filters: RefCell<Vec<Rc<dyn FileFilter>>>,
    selected_file: RefCell<Option<PathBuf>>,
    selected_files: RefCell<Vec<PathBuf>>,
    tool_tip_text: RefCell<Option<String>>,
    return_value: Cell<i32>,
    /// The chooser as a Swing `Component`, for where the Java adds the
    /// `JFileChooser` itself to a container (`CleanupPanel`).  Carries the
    /// chooser's name.
    component: Rc<JComponent>,
}

impl FileChooser {
    /// Java `JFileChooser.APPROVE_OPTION`.
    pub const APPROVE_OPTION: i32 = APPROVE_OPTION;
    /// Java `JFileChooser.CANCEL_OPTION`.
    pub const CANCEL_OPTION: i32 = CANCEL_OPTION;

    /// Java `FileChooser()`.
    pub fn new_void() -> Rc<FileChooser> {
        Self::new_base_manager_axis_id_string_browsing_directory(None, None, None, None)
    }

    /// Java `FileChooser(BaseManager)`.
    pub fn new_base_manager(manager: Option<&'static dyn BaseManager>) -> Rc<FileChooser> {
        Self::new_base_manager_axis_id_string_browsing_directory(manager, None, None, None)
    }

    /// Java `FileChooser(String)`.
    pub fn new_string(browsing_dir: Option<&str>) -> Rc<FileChooser> {
        Self::new_base_manager_axis_id_string_browsing_directory(None, None, browsing_dir, None)
    }

    /// Java `FileChooser(File)`.
    pub fn new_file(browsing_dir: Option<&Path>) -> Rc<FileChooser> {
        // `browsingDir != null ? browsingDir.getAbsolutePath() : null`
        let path = browsing_dir.map(|dir| {
            std::path::absolute(dir)
                .unwrap_or_else(|_| dir.to_path_buf())
                .to_string_lossy()
                .into_owned()
        });
        Self::new_base_manager_axis_id_string_browsing_directory(None, None, path.as_deref(), None)
    }

    /// Java `FileChooser(BaseManager, File)`.
    pub fn new_base_manager_file(
        manager: Option<&'static dyn BaseManager>,
        browsing_dir: Option<&Path>,
    ) -> Rc<FileChooser> {
        let path = browsing_dir.map(|dir| {
            std::path::absolute(dir)
                .unwrap_or_else(|_| dir.to_path_buf())
                .to_string_lossy()
                .into_owned()
        });
        Self::new_base_manager_axis_id_string_browsing_directory(
            manager,
            None,
            path.as_deref(),
            None,
        )
    }

    /// Java `FileChooser(BaseManager, String)`.
    pub fn new_base_manager_string(
        manager: Option<&'static dyn BaseManager>,
        browsing_dir: Option<&str>,
    ) -> Rc<FileChooser> {
        Self::new_base_manager_axis_id_string_browsing_directory(manager, None, browsing_dir, None)
    }

    /// Java `FileChooser(BaseManager, File, BrowsingDirectory)`.
    pub fn new_base_manager_file_browsing_directory(
        manager: Option<&'static dyn BaseManager>,
        browsing_dir: Option<&Path>,
        alt_browsing_dir: Option<&dyn BrowsingDirectory>,
    ) -> Rc<FileChooser> {
        let path = browsing_dir.map(|dir| {
            std::path::absolute(dir)
                .unwrap_or_else(|_| dir.to_path_buf())
                .to_string_lossy()
                .into_owned()
        });
        Self::new_base_manager_axis_id_string_browsing_directory(
            manager,
            None,
            path.as_deref(),
            alt_browsing_dir,
        )
    }

    /// Java `FileChooser(BrowsingDirectory)`.
    pub fn new_browsing_directory(browsing_dir: Option<&dyn BrowsingDirectory>) -> Rc<FileChooser> {
        Self::new_base_manager_axis_id_string_browsing_directory(None, None, None, browsing_dir)
    }

    /// Java `FileChooser(BaseManager, AxisID, String, BrowsingDirectory)`.
    pub fn new_base_manager_axis_id_string_browsing_directory(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        browsing_dir: Option<&str>,
        alt_browsing_dir: Option<&dyn BrowsingDirectory>,
    ) -> Rc<FileChooser> {
        // `super(getBrowsingDir(manager, browsingDir, altBrowsingDir))`: the
        // `JFileChooser(File currentDirectory)` constructor.
        let current_directory = Self::get_browsing_dir(manager, browsing_dir, alt_browsing_dir);
        let chooser = Rc::new(FileChooser {
            manager,
            axis_id,
            name: RefCell::new(None),
            current_directory: RefCell::new(None),
            dialog_title: RefCell::new(None),
            dialog_type: Cell::new(OPEN_DIALOG),
            file_selection_mode: Cell::new(FILES_ONLY),
            multi_selection_enabled: Cell::new(false),
            file_hiding_enabled: Cell::new(true),
            control_buttons_are_shown: Cell::new(true),
            file_filter: RefCell::new(None),
            choosable_file_filters: RefCell::new(Vec::new()),
            selected_file: RefCell::new(None),
            selected_files: RefCell::new(Vec::new()),
            tool_tip_text: RefCell::new(None),
            return_value: Cell::new(ERROR_OPTION),
            component: JComponent::new_other(),
        });
        chooser.set_current_directory(current_directory.as_deref());
        chooser.set_name(Some(DEFAULT_TITLE));
        chooser
    }

    /// Java private static `getBrowsingDir(BaseManager, String, BrowsingDirectory)`.
    fn get_browsing_dir(
        manager: Option<&'static dyn BaseManager>,
        browsing_dir: Option<&str>,
        alt_browsing_dir: Option<&dyn BrowsingDirectory>,
    ) -> Option<PathBuf> {
        // `!browsingDir.matches("\\s*")`: Java `\s` is [ \t\n\x0B\f\r].
        if let Some(browsing_dir) = browsing_dir
            && !browsing_dir
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        {
            let mut dir = ValidDirectory::new(manager);
            dir.set_string(Some(browsing_dir));
            if !dir.is_null() {
                return dir.get_void();
            }
        }
        if let Some(alt_browsing_dir) = alt_browsing_dir {
            return alt_browsing_dir.get_browsing_dir();
        }
        if let Some(manager) = manager {
            let user_dir = manager.get_property_user_dir();
            if let Some(user_dir) = user_dir {
                return Some(PathBuf::from(user_dir));
            }
        }
        None
    }

    /// Java overridden `setDialogTitle(String)`.
    pub fn set_dialog_title(&self, dialog_title: Option<&str>) {
        // super.setDialogTitle(dialogTitle)
        *self.dialog_title.borrow_mut() = dialog_title.map(str::to_owned);
        self.set_name(dialog_title);
    }

    /// The `JFileChooser` used as a `Component` (`container.add(chooser)`).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let name = utilities::convert_label_to_name(text, FILE_CHOOSER_UNLIMITED_SEGMENTS);
        // super.setName(name)
        self.component.set_name(name.as_deref());
        *self.name.borrow_mut() = name;
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    // --- inherited JFileChooser / Component members eTomo uses ---

    /// Java inherited `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.name.borrow().clone()
    }

    /// Java inherited `getDialogTitle()`.
    pub fn get_dialog_title(&self) -> Option<String> {
        self.dialog_title.borrow().clone()
    }

    /// Java inherited `setCurrentDirectory(File)`: a missing directory keeps the
    /// current one; `null` means the default (home) directory; a file that is
    /// not a directory is replaced by the nearest directory above it.
    pub fn set_current_directory(&self, dir: Option<&Path>) {
        let mut dir = dir.map(Path::to_path_buf);
        if dir.as_ref().is_some_and(|dir| !dir.exists()) {
            dir = self.current_directory.borrow().clone();
        }
        if dir.is_none() {
            dir = std::env::var_os("HOME").map(PathBuf::from);
        }
        let Some(mut dir) = dir else {
            return;
        };
        if self.current_directory.borrow().as_ref() == Some(&dir) {
            return;
        }
        while !dir.is_dir() {
            let Some(parent) = dir.parent() else {
                break;
            };
            dir = parent.to_path_buf();
        }
        *self.current_directory.borrow_mut() = Some(dir);
    }

    /// Java inherited `getCurrentDirectory()`.
    pub fn get_current_directory(&self) -> Option<PathBuf> {
        self.current_directory.borrow().clone()
    }

    /// Java inherited `rescanCurrentDirectory()`: refreshes the dialog's list;
    /// nothing is cached here.
    pub fn rescan_current_directory(&self) {}

    /// Java inherited `setDialogType(int)`.
    pub fn set_dialog_type(&self, dialog_type: i32) {
        self.dialog_type.set(dialog_type);
    }

    /// Java inherited `getDialogType()`.
    pub fn get_dialog_type(&self) -> i32 {
        self.dialog_type.get()
    }

    /// Java inherited `setFileSelectionMode(int)`.
    pub fn set_file_selection_mode(&self, mode: i32) {
        self.file_selection_mode.set(mode);
    }

    /// Java inherited `getFileSelectionMode()`.
    pub fn get_file_selection_mode(&self) -> i32 {
        self.file_selection_mode.get()
    }

    /// Java inherited `setMultiSelectionEnabled(boolean)`.
    pub fn set_multi_selection_enabled(&self, enabled: bool) {
        self.multi_selection_enabled.set(enabled);
    }

    /// Java inherited `isMultiSelectionEnabled()`.
    pub fn is_multi_selection_enabled(&self) -> bool {
        self.multi_selection_enabled.get()
    }

    /// Java inherited `setFileHidingEnabled(boolean)`.
    pub fn set_file_hiding_enabled(&self, enabled: bool) {
        self.file_hiding_enabled.set(enabled);
    }

    /// Java inherited `isFileHidingEnabled()`.
    pub fn is_file_hiding_enabled(&self) -> bool {
        self.file_hiding_enabled.get()
    }

    /// Java inherited `setControlButtonsAreShown(boolean)`.
    pub fn set_control_buttons_are_shown(&self, shown: bool) {
        self.control_buttons_are_shown.set(shown);
    }

    /// Java inherited `getControlButtonsAreShown()`.
    pub fn get_control_buttons_are_shown(&self) -> bool {
        self.control_buttons_are_shown.get()
    }

    /// Java inherited `setFileFilter(FileFilter)`; Swing also adds it to the
    /// choosable filters.
    pub fn set_file_filter(&self, filter: Option<Rc<dyn FileFilter>>) {
        if let Some(filter) = &filter {
            let known = self
                .choosable_file_filters
                .borrow()
                .iter()
                .any(|f| Rc::ptr_eq(f, filter));
            if !known {
                self.choosable_file_filters
                    .borrow_mut()
                    .push(filter.clone());
            }
        }
        *self.file_filter.borrow_mut() = filter;
    }

    /// Java inherited `getFileFilter()`.
    pub fn get_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        self.file_filter.borrow().clone()
    }

    /// Java inherited `addChoosableFileFilter(FileFilter)`; Swing makes the
    /// added filter the current one.
    pub fn add_choosable_file_filter(&self, filter: Rc<dyn FileFilter>) {
        self.set_file_filter(Some(filter));
    }

    /// Java inherited `resetChoosableFileFilters()`: removes the added filters
    /// (the look and feel's "All Files" filter is not modelled).
    pub fn reset_choosable_file_filters(&self) {
        self.choosable_file_filters.borrow_mut().clear();
        *self.file_filter.borrow_mut() = None;
    }

    /// Java inherited `getChoosableFileFilters()`.
    pub fn get_choosable_file_filters(&self) -> Vec<Rc<dyn FileFilter>> {
        self.choosable_file_filters.borrow().clone()
    }

    /// Java inherited `setSelectedFile(File)`: an absolute file outside the
    /// current directory moves the current directory to its parent.
    pub fn set_selected_file(&self, file: Option<&Path>) {
        *self.selected_file.borrow_mut() = file.map(Path::to_path_buf);
        if let Some(file) = file
            && file.is_absolute()
            && self.current_directory.borrow().as_deref() != file.parent()
        {
            self.set_current_directory(file.parent());
        }
    }

    /// Java inherited `getSelectedFile()`.
    pub fn get_selected_file(&self) -> Option<PathBuf> {
        self.selected_file.borrow().clone()
    }

    /// Java inherited `setSelectedFiles(File[])`: also selects the first file.
    pub fn set_selected_files(&self, files: &[PathBuf]) {
        *self.selected_files.borrow_mut() = files.to_vec();
        self.set_selected_file(files.first().map(PathBuf::as_path));
    }

    /// Java inherited `getSelectedFiles()`.
    pub fn get_selected_files(&self) -> Vec<PathBuf> {
        self.selected_files.borrow().clone()
    }

    /// Java inherited `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        *self.tool_tip_text.borrow_mut() = text.map(str::to_owned);
    }

    /// Java inherited `getToolTipText()`.
    pub fn get_tool_tip_text(&self) -> Option<String> {
        self.tool_tip_text.borrow().clone()
    }

    /// Java inherited `approveSelection()`: the Approve button.
    pub fn approve_selection(&self) {
        self.return_value.set(APPROVE_OPTION);
    }

    /// Java inherited `cancelSelection()`: the Cancel button.
    pub fn cancel_selection(&self) {
        self.return_value.set(CANCEL_OPTION);
    }

    /// Java inherited `showOpenDialog(Component)`.
    pub fn show_open_dialog(&self, parent: Option<&Rc<JComponent>>) -> i32 {
        self.set_dialog_type(OPEN_DIALOG);
        self.show_dialog(parent)
    }

    /// Java inherited `showSaveDialog(Component)`.
    pub fn show_save_dialog(&self, parent: Option<&Rc<JComponent>>) -> i32 {
        self.set_dialog_type(SAVE_DIALOG);
        self.show_dialog(parent)
    }

    /// Java inherited `showDialog(Component, String)` without an approve-button
    /// label: `returnValue = ERROR_OPTION`, show the modal dialog, return
    /// `returnValue`.  Closing the window is `CANCEL_OPTION`; with no dialog
    /// boundary installed the dialog is closed at once.
    pub fn show_dialog(&self, parent: Option<&Rc<JComponent>>) -> i32 {
        self.return_value.set(ERROR_OPTION);
        let responder = DIALOG_RESPONDER.with(|slot| slot.borrow().clone());
        if let Some(responder) = responder {
            responder(self, parent);
        }
        if self.return_value.get() == ERROR_OPTION {
            self.return_value.set(CANCEL_OPTION);
        }
        self.return_value.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_open_and_answered_by_the_responder() {
        let chooser = FileChooser::new_void();
        assert_eq!(chooser.get_name().as_deref(), Some("open"));
        assert_eq!(chooser.show_open_dialog(None), CANCEL_OPTION);
        set_dialog_responder(Some(Rc::new(|chooser: &FileChooser, _| {
            chooser.set_selected_file(Some(Path::new("/tmp/in.mrc")));
            chooser.approve_selection();
        })));
        assert_eq!(chooser.show_open_dialog(None), APPROVE_OPTION);
        assert_eq!(
            chooser.get_selected_file(),
            Some(PathBuf::from("/tmp/in.mrc"))
        );
        set_dialog_responder(None);
    }
}
