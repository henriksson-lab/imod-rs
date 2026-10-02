//! `IMOD/Etomo/src/etomo/ui/swing/FileTextField.java`.
//!
//! A text field with a folder button that opens a file chooser.  GridBag layout,
//! sizes and insets are Swing layout and recorded as comments.

use crate::imod::etomo::ui::field::Field;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::file_chooser::{self, FileChooser};
use super::file_text_field_interface::FileTextFieldInterface;
use super::scaled_image;
use super::simple_button::SimpleButton;
use super::text_field::TextField;
use super::tooltip_formatter;
use super::ui_utilities;
use crate::imod::etomo::jdk::{ActionListener, JComponent};
use crate::imod::etomo::r#type::const_string_parameter::ConstStringParameter;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::utilities;

/// Java private static `FIELD_TYPE`: assuming the field type is always non-numeric.
const FIELD_TYPE: FieldType = FieldType::String;

/// Java `FileTextField`.
pub struct FileTextField {
    /// Java `button`.
    button: Rc<SimpleButton>,
    /// Java `panel`.
    panel: Rc<JComponent>,
    /// Java `field`.
    field: Rc<TextField>,
    /// Java `label`.
    label: Option<Rc<JComponent>>,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `file`.  Must have file because the field may display a shortened name.
    /// Keep file up to date.
    file: RefCell<Option<PathBuf>>,
    /// Java `propertyUserDir`.
    property_user_dir: RefCell<Option<String>>,
    /// Java `parent`.
    parent: RefCell<Option<Rc<JComponent>>>,
    /// Java `fileSelectionMode`.
    file_selection_mode: Cell<i32>,
    /// Java `checkpointValue`.
    checkpoint_value: RefCell<Option<String>>,
    /// Java `prevChooserDir`.
    prev_chooser_dir: RefCell<Option<String>>,
    /// Java `usePrevChooserDir`.
    use_prev_chooser_dir: Cell<bool>,
    /// Java `textPreferredWidth` (an `Integer`).
    text_preferred_width: Cell<Option<i32>>,
    /// Java `showPartialPath`.
    show_partial_path: bool,
    this: Weak<FileTextField>,
}

impl FileTextField {
    /// Java `FileTextField(String)`.
    pub fn new(label: &str) -> Rc<FileTextField> {
        Self::new_private(label, true, None, false)
    }

    /// Java private `FileTextField(String, boolean, String, boolean)`.
    fn new_private(
        label: &str,
        labeled: bool,
        property_user_dir: Option<&str>,
        show_partial_path: bool,
    ) -> Rc<FileTextField> {
        let button = SimpleButton::new_scaled_image(Some(if !*utilities::APRIL_FOOLS {
            &scaled_image::OPEN_FILE
        } else {
            &scaled_image::OPEN_FILE_FOOL
        }));
        let panel = JComponent::new_panel();
        // Swing layout: GridBagLayout, fill BOTH, weights 0, grid 1x1.
        let label_component = if labeled {
            let label_component = JComponent::new_label(label);
            panel.add(&label_component);
            Some(label_component)
        } else {
            None
        };
        let field = TextField::new(FIELD_TYPE, Some(label), None);
        let folder_button_size = ui_utilities::get_scaled_folder_button_dimension();
        let text_preferred_width = 250
            * utilities::java_lang_math_round(
                super::ui_parameters::UIParameters::get_instance_void().get_font_size_adjustment(),
            ) as i32;
        field.set_text_preferred_size(crate::imod::etomo::jdk::Dimension {
            width: text_preferred_width,
            height: folder_button_size.height,
        });
        // Swing layout: insets (0, 0, 0, -1).
        panel.add(&field.get_component());
        button.get_component().set_action_command(Some(label));
        button.set_name(Some(label));
        // Swing layout: insets (0, -1, 0, 0); button preferred and maximum size.
        panel.add(&button.get_component());
        // showPartialPath allows the class to hide the whole absolute path of a file
        // to save display space. Permanantly make the field ineditable.
        if show_partial_path {
            field.set_editable(false);
        }
        Rc::new_cyclic(|this| FileTextField {
            button,
            panel,
            field,
            label: label_component,
            debug: Cell::new(false),
            file: RefCell::new(None),
            property_user_dir: RefCell::new(property_user_dir.map(str::to_owned)),
            parent: RefCell::new(None),
            file_selection_mode: Cell::new(-1),
            checkpoint_value: RefCell::new(None),
            prev_chooser_dir: RefCell::new(None),
            use_prev_chooser_dir: Cell::new(false),
            text_preferred_width: Cell::new(Some(text_preferred_width)),
            show_partial_path,
            this: this.clone(),
        })
    }

    /// Java `setTextPreferredWidth(int)`.
    pub fn set_text_preferred_width(&self, width: i32) {
        let width = ui_utilities::scale_by_font_size_int(width);
        self.text_preferred_width.set(Some(width));
        let height = self.field.get_preferred_size().height;
        self.field
            .set_text_preferred_size(crate::imod::etomo::jdk::Dimension { width, height });
        if self.show_partial_path {
            let file = self.file.borrow().clone();
            self.fill_with_partial_path(file.as_deref());
        }
    }

    /// Java `getUnlabeledInstance(String)`.
    pub fn get_unlabeled_instance(action_command: &str) -> Rc<FileTextField> {
        Self::new_private(action_command, false, None, false)
    }

    /// Java `getUnlabeledPartialPathInstance(String)`.
    pub fn get_unlabeled_partial_path_instance(action_command: &str) -> Rc<FileTextField> {
        Self::new_private(action_command, false, None, true)
    }

    /// Java `getPartialPathInstance(String)`.
    pub fn get_partial_path_instance(label: &str) -> Rc<FileTextField> {
        Self::new_private(label, true, None, true)
    }

    /// Java `setAlignmentX(float)`: layout only.
    pub fn set_alignment_x(&self, _alignment: f32) {}

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.button.get_component().get_action_command()
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.button
            .get_component()
            .add_action_listener(action_listener);
    }

    /// Java `addAction(String, Component, int)`.
    pub fn add_action(
        &self,
        property_user_dir: Option<&str>,
        parent: Option<Rc<JComponent>>,
        file_selection_mode: i32,
    ) {
        *self.property_user_dir.borrow_mut() = property_user_dir.map(str::to_owned);
        *self.parent.borrow_mut() = parent;
        self.file_selection_mode.set(file_selection_mode);
        // Java `new FileTextFieldActionListener(this)`.
        let adaptee = self.this.clone();
        self.button
            .get_component()
            .add_action_listener(Rc::new(move |_event| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action();
                }
            }));
    }

    /// Java `setUsePrevChooserDir(boolean)`.
    pub fn set_use_prev_chooser_dir(&self, use_prev_chooser_dir: bool) {
        self.use_prev_chooser_dir.set(use_prev_chooser_dir);
    }

    /// Java private `action()`.
    fn action(&self) {
        // Open up the file chooser in the current working directory
        let dir = if self.use_prev_chooser_dir.get() && self.prev_chooser_dir.borrow().is_some() {
            self.prev_chooser_dir.borrow().clone()
        } else {
            self.property_user_dir.borrow().clone()
        };
        let chooser = FileChooser::new_string(dir.as_deref());
        // Swing layout: chooser preferred size from UIParameters.
        if self.file_selection_mode.get() != -1 {
            chooser.set_file_selection_mode(self.file_selection_mode.get());
        }
        let parent = self.parent.borrow().clone();
        let return_val = chooser.show_open_dialog(parent.as_ref());
        if return_val == file_chooser::APPROVE_OPTION
            && let Some(file) = chooser.get_selected_file()
        {
            self.set_text_string(Some(&utilities::java_io_file_get_absolute_path(
                &file.to_string_lossy(),
            )));
            *self.prev_chooser_dir.borrow_mut() =
                utilities::java_io_file_get_parent(&file.to_string_lossy());
        }
    }

    /// Java `clear()`.
    pub fn clear(&self) {
        *self.file.borrow_mut() = None;
        self.field.set_text_string(Some(""));
        *self.prev_chooser_dir.borrow_mut() = None;
    }

    /// Java `checkpoint()`.
    pub fn checkpoint(&self) {
        *self.checkpoint_value.borrow_mut() = self.get_text();
    }

    /// Java `resetToCheckpoint()`.
    pub fn reset_to_checkpoint(&self) {
        let value = self.checkpoint_value.borrow().clone();
        let Some(value) = value else {
            return;
        };
        self.set_text_string(Some(&value));
    }

    /// Java `setFieldEditable(boolean)`.
    pub fn set_field_editable(&self, editable: bool) {
        if self.show_partial_path {
            // Cannot make the field editable if showPartialPath is on.
            return;
        }
        self.field.set_editable(editable);
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if !self.show_partial_path {
            // Cannot make the field editable if showPartialPath is on.
            self.field.set_editable(editable);
        }
        self.button.get_component().set_enabled(editable);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.field.set_enabled(enabled);
        self.button.get_component().set_enabled(enabled);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.panel.set_visible(visible);
    }

    /// Java `setButtonEnabled(boolean)`.
    pub fn set_button_enabled(&self, enabled: bool) {
        self.button.get_component().set_enabled(enabled);
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        let text = self.field.get_text_void();
        text.as_deref()
            .unwrap_or("")
            .trim_matches(char::is_whitespace)
            .is_empty()
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.field.is_editable()
    }

    /// Java `exists()`.
    pub fn exists(&self) -> bool {
        self.update_internal_values_void();
        match self.file.borrow().as_ref() {
            None => false,
            Some(file) => file.exists(),
        }
    }

    /// Java `setRequired(boolean)`.
    pub fn set_required(&self, required: bool) {
        self.field.set_required(required);
    }

    /// Java `setFileMustExist(boolean)`.
    pub fn set_file_must_exist(&self, file_must_exist: bool) {
        self.field.set_file_must_exist(file_must_exist);
    }

    /// Java `getFile(boolean)`.
    pub fn get_file_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<PathBuf>, FieldValidationFailedException> {
        if !self.update_internal_values_boolean(do_validation)? {
            // partial paths in use. Values do not need to be updated. Validation not done.
            self.field.get_text_boolean(do_validation)?;
        }
        Ok(self.file.borrow().clone())
    }

    /// Java `getFileName()`.  Java throws NullPointerException with no file; the
    /// translation returns None.
    pub fn get_file_name(&self) -> Option<String> {
        self.update_internal_values_void();
        self.file
            .borrow()
            .as_ref()
            .map(|file| utilities::java_io_file_get_name(&file.to_string_lossy()))
    }

    /// Java `getFileAbsolutePath()`.  Java throws NullPointerException with no
    /// file; the translation returns None.
    pub fn get_file_absolute_path(&self) -> Option<String> {
        self.update_internal_values_void();
        self.file
            .borrow()
            .as_ref()
            .map(|file| utilities::java_io_file_get_absolute_path(&file.to_string_lossy()))
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        if let Some(text) = text
            && !text.trim_matches(char::is_whitespace).is_empty()
        {
            self.set_internal_values(Some(Path::new(text)));
        } else {
            self.set_internal_values(None);
        }
    }

    /// Java `setText(ConstStringParameter)`.
    pub fn set_text_const_string_parameter(&self, input: &dyn ConstStringParameter) {
        self.set_text_string(Some(&input.to_string()));
    }

    /// Java private `updateInternalValues(boolean)`.
    fn update_internal_values_boolean(
        &self,
        do_validation: bool,
    ) -> Result<bool, FieldValidationFailedException> {
        if self.show_partial_path {
            return Ok(false);
        }
        let text = self.field.get_text_boolean(do_validation)?;
        self.update_internal_values_string(text.as_deref());
        Ok(true)
    }

    /// Java private `updateInternalValues()`.
    fn update_internal_values_void(&self) {
        if self.show_partial_path {
            return;
        }
        let text = self.field.get_text_void();
        self.update_internal_values_string(text.as_deref());
    }

    /// Java private `updateInternalValues(String)`.
    fn update_internal_values_string(&self, text: Option<&str>) {
        if self.show_partial_path {
            return;
        }
        match text {
            Some(text) if !text.trim_matches(char::is_whitespace).is_empty() => {
                *self.file.borrow_mut() = Some(PathBuf::from(text));
            }
            _ => *self.file.borrow_mut() = None,
        }
    }

    /// Java private `setInternalValues(File)`.
    fn set_internal_values(&self, input_file: Option<&Path>) {
        *self.file.borrow_mut() = input_file.map(Path::to_path_buf);
        let Some(input_file) = input_file else {
            self.field.set_text_string(Some(""));
            *self.prev_chooser_dir.borrow_mut() = None;
            return;
        };
        *self.prev_chooser_dir.borrow_mut() =
            utilities::java_io_file_get_parent(&input_file.to_string_lossy());
        if !self.show_partial_path || !self.fill_with_partial_path(Some(input_file)) {
            self.field
                .set_text_string(Some(&utilities::java_io_file_get_absolute_path(
                    &input_file.to_string_lossy(),
                )));
        }
    }

    /// Java private `fillWithPartialPath(File)`.
    fn fill_with_partial_path(&self, input_file: Option<&Path>) -> bool {
        if !self.show_partial_path {
            return false;
        }
        let Some(input_file) = input_file else {
            self.field.set_text_string(Some(""));
            return true;
        };
        let font_metrics =
            ui_utilities::get_font_metrics_abstract_button(&self.button.get_component())
                .expect("font metrics");
        let ellipsis = "...";
        let ellipsis_width = font_metrics.string_width(ellipsis);
        let separator_width = font_metrics.string_width("/");
        let absolute_path =
            utilities::java_io_file_get_absolute_path(&input_file.to_string_lossy());
        // `absolutePath.split("\\Q/\\E")`: Java split drops trailing empty strings.
        let mut path_array: Vec<&str> = absolute_path.split('/').collect();
        while path_array.last().is_some_and(|last| last.is_empty()) {
            path_array.pop();
        }
        if path_array.is_empty() {
            self.field.set_text_string(Some(""));
            return true;
        }
        if path_array.len() == 1 {
            self.field.set_text_string(Some(&absolute_path));
            return true;
        }
        // If there is no preferred width, or if the field is very narrow, the default
        // partial path is the parent directory and the file.
        let default_index = path_array.len() as i32 - 2;
        // Build the partial path until no more elements can fit.
        let mut builder = String::new();
        let mut whole_path = false;
        let mut i = path_array.len() as i32 - 1;
        while i >= 0 {
            let element = path_array[i as usize];
            if element.is_empty() {
                if i == 0 {
                    // The absolute path fits. Use that.
                    whole_path = true;
                    break;
                }
                i -= 1;
                continue;
            }
            // Separator is added after the element.
            let use_separator = i < path_array.len() as i32 - 1;
            // Keep going if haven't reached the default index. Keep going when there's
            // more space available in the text field.  Java compares the Integer
            // `textPreferredWidth`, unboxing it (NullPointerException when null); the
            // null case is handled below the comparison in Java and reached here as
            // "does not fit".
            let fits = self.text_preferred_width.get().is_some_and(|width| {
                width
                    >= (if i > 0 { ellipsis_width } else { 0 })
                        + font_metrics.string_width(element)
                        + (if use_separator { separator_width } else { 0 })
                        + font_metrics.string_width(&builder)
            });
            if i >= default_index || fits {
                if i == 0 {
                    // The absolute path fits. Use that.
                    whole_path = true;
                    break;
                }
                builder.insert_str(
                    0,
                    &format!("{}{}", element, if use_separator { "/" } else { "" }),
                );
                // If the width was never set, use the default partial path.
                if self.text_preferred_width.get().is_none() && i == default_index {
                    break;
                }
            } else {
                break;
            }
            i -= 1;
        }
        if whole_path {
            self.field.set_text_string(Some(&absolute_path));
        } else {
            self.field
                .set_text_string(Some(&format!("{ellipsis}{builder}")));
        }
        true
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> Option<String> {
        self.field.get_text_void()
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.field.set_tool_tip_text(text);
        self.panel.set_tool_tip_text(text);
        let text = tooltip_formatter::INSTANCE.format(text);
        self.button
            .get_component()
            .set_tool_tip_text(text.as_deref());
        if let Some(label) = &self.label {
            label.set_tool_tip_text(text.as_deref());
        }
    }

    /// Java `setFieldToolTipText(String)`.
    pub fn set_field_tool_tip_text(&self, text: Option<&str>) {
        self.field.set_tool_tip_text(text);
        self.panel.set_tool_tip_text(text);
        let text = tooltip_formatter::INSTANCE.format(text);
        if let Some(label) = &self.label {
            label.set_tool_tip_text(text.as_deref());
        }
    }

    /// Java `setButtonToolTipText(String)`.
    pub fn set_button_tool_tip_text(&self, text: Option<&str>) {
        self.button.get_component().set_tool_tip_text(text);
    }
}

impl FileTextFieldInterface for FileTextField {
    /// Java `setFile(File)`.
    fn set_file(&self, file: Option<PathBuf>) {
        self.set_internal_values(file.as_deref());
    }

    /// Java `getFile()`.
    fn get_file(&self) -> Option<PathBuf> {
        self.update_internal_values_void();
        self.file.borrow().clone()
    }

    /// Java `getFileFilter()`.
    fn get_file_filter(&self) -> Option<Rc<dyn crate::imod::etomo::jdk::FileFilter>> {
        None
    }
}
