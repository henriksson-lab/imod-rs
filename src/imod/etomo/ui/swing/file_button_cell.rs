//! `IMOD/Etomo/src/etomo/ui/swing/FileButtonCell.java`.
//!
//! A table cell holding a folder-icon button that opens a file chooser and hands
//! the chosen file to an `ActionTarget` (the field this button is associated with).
//!
//! Java `final class FileButtonCell extends InputCell`: every Java method body is an
//! inherent method; the trait impls at the end bind `CellVirtual` and
//! `InputCellVirtual` to them.  `FileButtonCell` also overrides `setName()` and
//! `setHeaders(...)`, which `InputCell` calls on itself; see NEEDS in the report
//! (`InputCellVirtual` must dispatch them).  Sizes and borders are layout.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::action_target::ActionTarget;
use super::cell::{Cell as TableCell, CellVirtual};
use super::field_lock_controller::FieldLockController;
use super::file_chooser::{self, FileChooser};
use super::header_cell::HeaderCell;
use super::input_cell::{InputCell, InputCellVirtual};
use super::scaled_image;
use super::simple_button::SimpleButton;
use super::tooltip_formatter;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::DEFAULT_DELIMITER;
use crate::imod::etomo::r#type::ui_test_field_type::{self, UITestFieldType};
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::utilities;

/// Java package-private `final class FileButtonCell extends InputCell`.
pub struct FileButtonCell {
    /// Java superclass `InputCell`.
    base: InputCell,
    /// Java final `manager`.
    manager: &'static dyn BaseManager,
    /// Java `actionTarget`: the field that this button is associated with.
    action_target: RefCell<Option<Rc<dyn ActionTarget>>>,
    /// Java `label`.
    label: RefCell<Option<String>>,
    /// Java `fileFilter`.
    file_filter: RefCell<Option<Rc<dyn FileFilter>>>,
    /// Java `browsingDir`.
    browsing_dir: RefCell<Option<Rc<dyn BrowsingDirectory>>>,
    /// Java final `button`.
    button: Rc<SimpleButton>,
    /// Java final `fieldLockController`.
    field_lock_controller: Rc<FieldLockController>,
}

impl Deref for FileButtonCell {
    type Target = InputCell;
    fn deref(&self) -> &InputCell {
        &self.base
    }
}

impl FileButtonCell {
    /// Java private `FileButtonCell(BaseManager)`.
    fn new(manager: &'static dyn BaseManager) -> Rc<FileButtonCell> {
        // super(): InputCell(); then the field initialisers.
        let button = SimpleButton::new_scaled_image(Some(if !*utilities::APRIL_FOOLS {
            &scaled_image::OPEN_FILE_PEET
        } else {
            &scaled_image::OPEN_FILE_FOOL
        }));
        let field_lock_controller =
            FieldLockController::get_button_instance(&button.get_component());
        let instance = Rc::new(FileButtonCell {
            base: InputCell::new_void(),
            manager,
            action_target: RefCell::new(None),
            label: RefCell::new(None),
            file_filter: RefCell::new(None),
            browsing_dir: RefCell::new(None),
            button,
            field_lock_controller,
        });
        instance
            .base
            .set_this(Rc::downgrade(&instance) as Weak<dyn InputCellVirtual>);
        // Swing layout: button.setBorder(BorderFactory.createBevelBorder(
        // BevelBorder.RAISED)); (a commented-out etched border); size =
        // button.getPreferredSize(); if (size.width < size.height) size.width =
        // size.height; button.setSize(size).
        instance
    }

    /// Java static `getInstance(FileButtonCell)`.
    pub fn get_instance_file_button_cell(file_button_cell: &FileButtonCell) -> Rc<FileButtonCell> {
        let instance = FileButtonCell::new(file_button_cell.manager);
        *instance.browsing_dir.borrow_mut() = file_button_cell.browsing_dir.borrow().clone();
        *instance.label.borrow_mut() = file_button_cell.label.borrow().clone();
        *instance.file_filter.borrow_mut() = file_button_cell.file_filter.borrow().clone();
        instance.add_listeners(&instance);
        instance
    }

    /// Java static `getInstance(BaseManager)`.
    pub fn get_instance_base_manager(manager: &'static dyn BaseManager) -> Rc<FileButtonCell> {
        let instance = FileButtonCell::new(manager);
        instance.add_listeners(&instance);
        instance
    }

    /// Java `@Override getText()`.
    pub fn get_text(&self) -> Option<String> {
        None
    }

    /// Java `setBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_browsing_directory(&self, input: Option<Rc<dyn BrowsingDirectory>>) {
        *self.browsing_dir.borrow_mut() = input;
    }

    /// Java `@Override add(JPanel, GridBagLayout, GridBagConstraints)`.  The
    /// constraints' `weightx` is set to 0 for the button and restored after (layout).
    pub fn add(&self, panel: &Rc<JComponent>) {
        // Swing layout: oldWeightx = constraints.weightx; constraints.weightx = 0.0.
        self.base.add(panel);
        // Swing layout: constraints.weightx = oldWeightx.
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self, this: &Rc<FileButtonCell>) {
        // button.addActionListener(new FileButtonActionListener(this))
        let file_button_cell = Rc::downgrade(this);
        self.button
            .get_component()
            .add_action_listener(Rc::new(move |_event| {
                // FileButtonActionListener.actionPerformed
                if let Some(file_button_cell) = file_button_cell.upgrade() {
                    file_button_cell.action();
                }
            }));
    }

    /// Java `setActionTarget(ActionTarget)`.
    pub fn set_action_target(&self, input: Option<Rc<dyn ActionTarget>>) {
        *self.action_target.borrow_mut() = input;
    }

    /// Java `@Override setHeaders(String, HeaderCell, HeaderCell)`.
    pub fn set_headers(
        &self,
        table_header: Option<&str>,
        row_header: &Rc<HeaderCell>,
        column_header: &Rc<HeaderCell>,
    ) {
        if self.label.borrow().is_none() {
            *self.label.borrow_mut() = column_header.get_text();
        }
        self.base
            .set_headers(table_header, row_header, column_header);
    }

    /// Java `@Override setName(String, String, String)`.  Not implemented at this
    /// time.
    pub fn set_name_string_string_string(
        &self,
        _reference1: Option<&str>,
        _reference2: Option<&str>,
        _reference3: Option<&str>,
    ) {
    }

    /// Java `@Override setName()`.
    pub fn set_name_void(&self) {
        let name = self
            .base
            .convert_label_to_name(self.get_field_type().is_unlimited_segments());
        self.button.set_name(name.as_deref());
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.get_component().get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `@Override getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.button.get_name()
    }

    /// Java `setLabel(String)`.
    pub fn set_label(&self, input: Option<&str>) {
        *self.label.borrow_mut() = input.map(str::to_owned);
    }

    /// Java `setFileFilter(FileFilter)`.
    pub fn set_file_filter_file_filter(&self, input: Option<Rc<dyn FileFilter>>) {
        *self.file_filter.borrow_mut() = input;
    }

    /// Java `setFileFilter(ExtensibleFileFilter)`.  `TomogramFileFilter` is the only
    /// concrete `ExtensibleFileFilter`.
    pub fn set_file_filter_extensible_file_filter(
        &self,
        input: Option<Rc<crate::imod::etomo::storage::tomogram_file_filter::TomogramFileFilter>>,
    ) {
        *self.file_filter.borrow_mut() = input.map(|input| input as Rc<dyn FileFilter>);
    }

    /// Java `getFileFilter()`.
    pub fn get_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        self.file_filter.borrow().clone()
    }

    /// Java `@Override getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.button.get_component()
    }

    /// Java `@Override getFieldType()`.
    pub fn get_field_type(&self) -> &'static UITestFieldType {
        &ui_test_field_type::BUTTON
    }

    /// Java `@Override getWidth()`.
    pub fn get_width(&self) -> i32 {
        // Swing geometry: button.getSize().width.  Sizes are not modelled by the jdk
        // stand-in.
        0
    }

    /// Java `@Override setLocked(boolean)`.
    pub fn set_locked(&self, locked: bool) {
        if self.field_lock_controller.set_locked(locked) {
            self.base.set_background_void();
        }
    }

    /// Java `@Override setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if self.field_lock_controller.set_editable(editable) {
            self.base.set_background_void();
        }
    }

    /// Java `@Override setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.field_lock_controller.set_enabled(enabled);
    }

    /// Java `@Override isLocked()`.
    pub fn is_locked(&self) -> bool {
        self.field_lock_controller.is_locked()
    }

    /// Java `@Override isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field_lock_controller.is_editable()
    }

    /// Java `@Override isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.field_lock_controller.is_enabled()
    }

    /// Java private `action()`.
    fn action(&self) {
        let action_target = self.action_target.borrow().clone();
        let expanded_value = action_target
            .as_ref()
            .and_then(|action_target| action_target.get_expanded_value());
        let browsing_dir = self.browsing_dir.borrow().clone();
        let chooser = FileChooser::new_base_manager_axis_id_string_browsing_directory(
            Some(self.manager),
            None,
            expanded_value.as_deref(),
            browsing_dir.as_deref(),
        );
        let label = self.label.borrow().clone();
        chooser.set_dialog_title(Some(label.as_deref().unwrap_or("Open File")));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        let file_filter = self.file_filter.borrow().clone();
        if file_filter.is_some() {
            chooser.set_file_filter(file_filter);
        }
        let return_val =
            chooser.show_open_dialog(self.button.get_component().get_parent().as_ref());
        if return_val == file_chooser::APPROVE_OPTION {
            let file = chooser.get_selected_file();
            if let Some(action_target) = &action_target {
                action_target.set_target_file(file.as_deref());
            }
            if let (Some(browsing_dir), Some(file)) = (&browsing_dir, &file) {
                browsing_dir.set_browsing_dir(file.parent());
            }
        }
    }

    /// Java `@Override setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.button
            .get_component()
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }
}

impl CellVirtual for FileButtonCell {
    fn cell(&self) -> &TableCell {
        &self.base
    }
    fn set_enabled(&self, enable: bool) {
        FileButtonCell::set_enabled(self, enable);
    }
    /// Java `InputCell.msgLabelChanged()`: `setName()`, which this class overrides.
    fn msg_label_changed(&self) {
        FileButtonCell::set_name_void(self);
    }
    fn add(&self, panel: &Rc<JComponent>) {
        FileButtonCell::add(self, panel);
    }
}

impl InputCellVirtual for FileButtonCell {
    fn input_cell(&self) -> &InputCell {
        &self.base
    }
    fn get_component(&self) -> Rc<JComponent> {
        FileButtonCell::get_component(self)
    }
    fn get_field_type(&self) -> &'static UITestFieldType {
        FileButtonCell::get_field_type(self)
    }
    fn get_width(&self) -> i32 {
        FileButtonCell::get_width(self)
    }
    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        FileButtonCell::set_tool_tip_text(self, tool_tip_text);
    }
    fn get_text(&self) -> Option<String> {
        FileButtonCell::get_text(self)
    }
    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) {
        FileButtonCell::set_name_string_string_string(self, reference1, reference2, reference3);
    }
    fn get_name(&self) -> Option<String> {
        FileButtonCell::get_name(self)
    }
    fn set_locked(&self, locked: bool) {
        FileButtonCell::set_locked(self, locked);
    }
    fn set_editable(&self, editable: bool) {
        FileButtonCell::set_editable(self, editable);
    }
    fn is_locked(&self) -> bool {
        FileButtonCell::is_locked(self)
    }
    fn is_editable(&self) -> bool {
        FileButtonCell::is_editable(self)
    }
    fn is_enabled(&self) -> bool {
        FileButtonCell::is_enabled(self)
    }
}
