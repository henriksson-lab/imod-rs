//! `IMOD/Etomo/src/etomo/ui/swing/FileTextField2.java`.
//!
//! Like `FileTextField` but handles relative paths.
//!
//! Java `public final class FileTextField2 implements FileTextFieldInterface,
//! TextFieldInterface, ActionListener, UIComponent, SwingComponent`.  An EDT object:
//! created as `Rc<Self>` by the static factories, every method takes `&self`, mutable
//! fields are cells.  The Java `ActionListener` implementation (the folder button's
//! listener) is a closure holding a `Weak` to the field that calls
//! [`FileTextField2::action_performed`].  GridBag/Box layout, preferred sizes, insets
//! and background colour are Swing layout/painting and appear as comments.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::file_chooser::{self, FileChooser};
use super::file_text_field_interface::FileTextFieldInterface;
use super::result_listener::ResultListener;
use super::scaled_image;
use super::simple_button::SimpleButton;
use super::swing_component::SwingComponent;
use super::text_field::TextField;
use super::tooltip_formatter;
use super::ui_parameters::UIParameters;
use super::ui_utilities;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::{self, ARGUMENTS};
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, Dimension, FileFilter, FocusListener, JComponent,
};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::text_field_interface::TextFieldInterface;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::file_path::FilePath;
use crate::imod::etomo::util::utilities;
use crate::imod::etomo::util::valid_directory::ValidDirectory;

/// Java `public final class FileTextField2`.
pub struct FileTextField2 {
    /// Java `this` (handed to the button's action listener and to result listeners).
    this: Weak<FileTextField2>,
    /// Java private final `STRING_FIELD_TYPE`: assuming the field type is always
    /// non-numeric.
    string_field_type: FieldType,

    /// Java private final `panel = new JPanel()`.
    panel: Rc<JComponent>,

    /// Java private final `button`.
    button: Rc<SimpleButton>,
    /// Java private final `field`.
    field: Rc<TextField>,
    /// Java private final `label` (a `JLabel`).
    label: Rc<JComponent>,
    /// Java private final `labeled`.
    labeled: bool,
    /// Java private final `manager` (may be null).
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID` (may be null).
    axis_id: Option<AxisID>,
    /// Java package-private final `alternateLayout`.
    pub alternate_layout: bool,
    // Java private final `layout` (GridBagLayout) and `constraints`
    // (GridBagConstraints): Swing layout, null when alternateLayout.
    /// Java private `resultListenerList`, initially null.
    result_listener_list: RefCell<Option<Vec<Rc<RefCell<dyn ResultListener>>>>>,
    /// Java private `fileSelectionMode`, initially -1.
    file_selection_mode: Cell<i32>,
    /// Java private `fileFilter`, initially null.
    file_filter: RefCell<Option<Rc<dyn FileFilter>>>,
    /// Java private `absolutePath`, initially false.
    absolute_path: Cell<bool>,
    /// Java private `useTextAsFileChooserDir`, initially false.
    use_text_as_file_chooser_dir: Cell<bool>,
    /// Java private `turnOffFileHiding`, initially false.
    turn_off_file_hiding: Cell<bool>,
    /// Java private `originReference`: overrides origin.
    origin_reference: RefCell<Option<Rc<FileTextField2>>>,
    /// Java private `origin`: if valid, it overrides originEtomoRunDir.
    origin: RefCell<Option<PathBuf>>,
    /// Java private `originEtomoRunDir`: if true, then the origin directory of the
    /// file is the directory in which etomo was run.  Useful when a dataset location
    /// has not been set.
    origin_etomo_run_dir: Cell<bool>,
    // Java private `directiveDef` (never read; the field's own directiveDef is used)
    // and `fontMetrics` (never read): no state carried.
    /// Java private `textEntryPolicy`, initially true.
    text_entry_policy: Cell<bool>,
    /// Java private `enabled`, initially true.
    enabled: Cell<bool>,
    /// Java private `editable`, initially true.
    editable: Cell<bool>,
    /// Java private `browsingDir`, initially null.
    browsing_dir: RefCell<Option<Rc<dyn BrowsingDirectory>>>,
    /// Java private `unformattedTooltip`, initially null.
    unformatted_tooltip: RefCell<Option<String>>,
    /// Java private `debug`, initially false.
    debug: Cell<bool>,
}

impl std::fmt::Display for FileTextField2 {
    /// Java `toString()`: `super.toString() + ":[text:" + field.getText() + ",label:" +
    /// label.getText()`.  `Object.toString()` is the class name and identity hash; the
    /// address stands in for the hash.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.ui.swing.FileTextField2@{:x}:[text:{},label:{}",
            self as *const FileTextField2 as usize,
            self.field.get_text_void().as_deref().unwrap_or("null"),
            self.label.get_text()
        )
    }
}

impl FileTextField2 {
    /// Java private constructor `FileTextField2(BaseManager, AxisID, String, boolean,
    /// boolean, boolean)`.
    fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        label: Option<&str>,
        labeled: bool,
        peet: bool,
        alternate_layout: bool,
    ) -> Rc<FileTextField2> {
        let string_field_type = FieldType::String;
        let panel = JComponent::new_panel();
        let button = if !peet {
            SimpleButton::new_scaled_image(Some(if !*utilities::APRIL_FOOLS {
                &scaled_image::OPEN_FILE
            } else {
                &scaled_image::OPEN_FILE_FOOL
            }))
        } else {
            SimpleButton::new_scaled_image(Some(if !*utilities::APRIL_FOOLS {
                &scaled_image::OPEN_FILE_PEET
            } else {
                &scaled_image::OPEN_FILE_FOOL
            }))
        };
        button.set_name(label);
        let field = TextField::new(string_field_type, label, None);
        // Java `new JLabel(label)`: a null label displays no text.
        let label_component = JComponent::new_label(label.unwrap_or(""));
        // if (!alternateLayout) { layout = new GridBagLayout(); constraints = new
        // GridBagConstraints(); } else { layout = null; constraints = null; }
        // Swing layout: no state carried.
        Rc::new_cyclic(|this| FileTextField2 {
            this: this.clone(),
            string_field_type,
            panel,
            button,
            field,
            label: label_component,
            labeled,
            manager,
            axis_id,
            alternate_layout,
            result_listener_list: RefCell::new(None),
            file_selection_mode: Cell::new(-1),
            file_filter: RefCell::new(None),
            absolute_path: Cell::new(false),
            use_text_as_file_chooser_dir: Cell::new(false),
            turn_off_file_hiding: Cell::new(false),
            origin_reference: RefCell::new(None),
            origin: RefCell::new(None),
            origin_etomo_run_dir: Cell::new(false),
            text_entry_policy: Cell::new(true),
            enabled: Cell::new(true),
            editable: Cell::new(true),
            browsing_dir: RefCell::new(None),
            unformatted_tooltip: RefCell::new(None),
            debug: Cell::new(false),
        })
    }

    /// Java static `getUnlabeledPeetInstance(BaseManager, String)`: get an unlabeled
    /// instance with a PEET-style button.  The starting directory for the file chooser
    /// and the origin of relative files is the manager's property user directory.
    pub fn get_unlabeled_peet_instance(
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
    ) -> Rc<FileTextField2> {
        let instance = FileTextField2::new(manager, None, name, false, true, false);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java static `getUnlabeledAltLayoutInstance(BaseManager, String)`.
    pub fn get_unlabeled_alt_layout_instance(
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
    ) -> Rc<FileTextField2> {
        let instance = FileTextField2::new(manager, None, name, false, false, true);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java static `getUnlabeledInstance(BaseManager, String)`.
    pub fn get_unlabeled_instance(
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
    ) -> Rc<FileTextField2> {
        let instance = FileTextField2::new(manager, None, name, false, false, false);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java static `getPeetInstance(BaseManager, AxisID, String)`: get a labeled
    /// instance with a PEET-style button.  The starting directory for the file chooser
    /// and the origin of relative files is the manager's property user directory.
    pub fn get_peet_instance(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        name: Option<&str>,
    ) -> Rc<FileTextField2> {
        let instance = FileTextField2::new(manager, axis_id, name, true, true, false);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java public static `getInstance(BaseManager, String)`.
    pub fn get_instance(
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
    ) -> Rc<FileTextField2> {
        let instance = FileTextField2::new(manager, None, name, true, false, false);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java static `getAltLayoutInstance(BaseManager, String)`.
    pub fn get_alt_layout_instance(
        manager: Option<&'static dyn BaseManager>,
        name: Option<&str>,
    ) -> Rc<FileTextField2> {
        let instance = FileTextField2::new(manager, None, name, true, false, true);
        instance.create_panel();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        let folder_button_dim = ui_utilities::get_scaled_folder_button_dimension();
        self.field.set_text_preferred_size(Dimension {
            width: 250
                * utilities::java_lang_math_round(
                    UIParameters::get_instance_void().get_font_size_adjustment(),
                ) as i32,
            height: folder_button_dim.height,
        });
        self.button.set_name(Some(&self.label.get_text()));
        // Swing layout: button.setPreferredSize(folderButtonDim);
        // button.setMaximumSize(folderButtonDim).
        if !self.alternate_layout {
            // panel
            // Swing layout: panel GridBagLayout; constraints fill BOTH, weights 0.0,
            // gridheight/gridwidth 1.
            if self.labeled {
                self.panel.add(&self.label);
            }
            // Swing layout: insets (0, 0, 0, -1).
            self.panel.add(&self.field.get_component());
            // Swing layout: insets (0, -1, 0, 0).
            self.panel.add(&self.button.get_component());
        } else {
            // Swing layout: panel BoxLayout X_AXIS.
            // Java `labeled && label != null`: label is final and never null.
            if self.labeled {
                self.panel.add(&self.label);
            }
            self.panel.add(&self.field.get_component());
            self.panel.add(&self.button.get_component());
            // Swing layout: panel.add(Box.createHorizontalGlue()).
        }
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.panel.set_visible(visible);
    }

    /// Java package-private `setBackground(Color)`.
    pub fn set_background(&self, _color: Option<(u8, u8, u8)>) {
        // Swing painting: panel.setBackground(color).
    }

    /// Java package-private `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        ui_utilities::get_preferred_width_j_label_string(&self.label, Some(&self.label.get_text()))
            + self.field.get_preferred_width()
            + self.button.get_preferred_width()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Java `button.addActionListener(this)`.
        let adaptee = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action_performed(event);
            }
        });
        self.button.get_component().add_action_listener(listener);
    }

    /// Java package-private `addResultListener(ResultListener)`: adds a result listener
    /// to a list of result listeners.  A null listener has no effect.
    pub fn add_result_listener(&self, listener: Option<Rc<RefCell<dyn ResultListener>>>) {
        let Some(listener) = listener else {
            return;
        };
        let mut result_listener_list = self.result_listener_list.borrow_mut();
        if result_listener_list.is_none() {
            *result_listener_list = Some(Vec::new());
        }
        result_listener_list.as_mut().unwrap().push(listener);
    }

    /// Java package-private `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.field.add_focus_listener(listener);
    }

    /// Java public `getRootPanel()`.
    pub fn get_root_panel(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `actionPerformed(ActionEvent)`: opens a file chooser and notifies the
    /// result listener list.
    pub fn action_performed(&self, _e: &ActionEvent) {
        let file_chooser_location = self.get_file_chooser_location();
        let browsing_dir = self.browsing_dir.borrow().clone();
        let chooser = FileChooser::new_base_manager_axis_id_string_browsing_directory(
            self.manager,
            self.axis_id,
            file_chooser_location.as_deref(),
            browsing_dir.as_deref(),
        );
        chooser.set_dialog_title(Some(&utilities::clean_up_label(&self.label.get_text())));
        if self.file_selection_mode.get() != -1 {
            chooser.set_file_selection_mode(self.file_selection_mode.get());
        }
        let file_filter = self.file_filter.borrow().clone();
        if file_filter.is_some() {
            chooser.set_file_filter(file_filter);
        }
        chooser.set_file_hiding_enabled(!self.turn_off_file_hiding.get());
        // Swing layout: chooser.setPreferredSize(
        // UIParameters.getInstance().getFileChooserDimension()).
        let return_val = chooser.show_open_dialog(Some(&self.panel));
        if return_val == file_chooser::APPROVE_OPTION {
            let file = chooser.get_selected_file();
            FileTextFieldInterface::set_file(self, file.clone());
            if let (Some(browsing_dir), Some(file)) = (&browsing_dir, &file) {
                browsing_dir.set_browsing_dir(
                    utilities::java_io_file_get_parent(&file.to_string_lossy())
                        .map(PathBuf::from)
                        .as_deref(),
                );
            }
        }
        let result_listener_list = self.result_listener_list.borrow().clone();
        if let Some(result_listener_list) = result_listener_list {
            if !result_listener_list.is_empty() {
                let this = self.this.upgrade();
                for listener in result_listener_list.iter() {
                    let origin: &dyn Any = match &this {
                        Some(this) => &**this,
                        None => self,
                    };
                    listener.borrow_mut().process_result(origin, false);
                }
            }
        }
    }

    /// Java package-private `setAdjustedFieldWidth(double)`: sets field width with
    /// font adjustment.
    pub fn set_adjusted_field_width(&self, width: f64) {
        self.field.set_text_preferred_width(
            width * UIParameters::get_instance_void().get_font_size_adjustment(),
        );
    }

    /// Java package-private `setAbsolutePath(boolean)`.
    pub fn set_absolute_path(&self, input: bool) {
        self.absolute_path.set(input);
    }

    /// Java package-private `setOriginEtomoRunDir(boolean)`.
    ///
    /// WARNING: The origin is also used to create relative file paths.  If the origin
    /// is being changed to something other then the run directory, turn on
    /// absolutePath.  The run directory is the dataset directory, except in the
    /// BatchRunTomo interface.
    pub fn set_origin_etomo_run_dir(&self, input: bool) {
        self.origin_etomo_run_dir.set(input);
    }

    /// Java package-private `setColumns(int)`.
    pub fn set_columns(&self, columns: i32) {
        self.field.set_columns_int(columns);
    }

    /// Java package-private `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java package-private `setPreferredWidth(double)`.
    pub fn set_preferred_width(&self, width: f64) {
        self.field.set_text_preferred_width(width);
    }

    /// Java package-private `setOrigin(File)`: sets the origin member variable which
    /// overrides the originEtomoRunDir member variable and the propertyUserDir when it
    /// is a valid directory.
    ///
    /// WARNING: The origin is also used to create relative file paths.  If the origin
    /// is being changed to something other then the run directory, turn on
    /// absolutePath.  The run directory is the dataset directory, except in the
    /// BatchRunTomo interface.
    pub fn set_origin_file(&self, input: Option<&Path>) {
        *self.origin.borrow_mut() = input.map(Path::to_path_buf);
    }

    /// Java package-private `setOrigin(String)`.
    ///
    /// WARNING: The origin is also used to create relative file paths.  If the origin
    /// is being changed to something other then the run directory, turn on
    /// absolutePath.  The run directory is the dataset directory, except in the
    /// BatchRunTomo interface.
    pub fn set_origin_string(&self, input: Option<&str>) {
        if let Some(input) = input {
            *self.origin.borrow_mut() = Some(PathBuf::from(input));
        }
    }

    /// Java package-private `setOriginReference(FileTextField2)`: sets the
    /// originReference member variable, which is first choice for where the file
    /// chooser should open.  It is checked each time the file chooser opens.  If
    /// originReference is null or empty, the fallback is the origin member variable.
    ///
    /// WARNING: The origin is also used to create relative file paths.  If the origin
    /// is being changed to something other then the run directory, turn on
    /// absolutePath.  The run directory is the dataset directory, except in the
    /// BatchRunTomo interface.
    pub fn set_origin_reference(&self, input: Option<Rc<FileTextField2>>) {
        *self.origin_reference.borrow_mut() = input;
    }

    /// Java package-private `setUseTextAsFileChooserDir(boolean)`: if true, the text in
    /// the text field with be where the file chooser opens, if the text field contains
    /// a directory.
    pub fn set_use_text_as_file_chooser_dir(&self, input: bool) {
        self.use_text_as_file_chooser_dir.set(input);
    }

    /// Java package-private `exists()`.
    pub fn exists(&self) -> bool {
        if !Field::is_empty(self) {
            return FileTextFieldInterface::get_file(self).is_some_and(|file| file.exists());
        }
        false
    }

    /// Java public `getFile(boolean, FieldDisplayer) throws
    /// FieldValidationFailedException`.
    pub fn get_file_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<PathBuf>, FieldValidationFailedException> {
        let text = self
            .field
            .get_text_boolean_field_displayer(do_validation, field_displayer)?;
        if let Some(text) = text {
            if !java_lang_string_matches_whitespace(&text) {
                return Ok(Some(FilePath::build_absolute_file_string_string(
                    self.get_origin_dir().as_deref(),
                    &text,
                )));
            }
        }
        Ok(None)
    }

    /// Java public `equals(FileTextField2)`.
    pub fn equals(&self, input: Option<&FileTextField2>) -> bool {
        let Some(input) = input else {
            return false;
        };
        let file = FileTextFieldInterface::get_file(self);
        let input_file = FileTextFieldInterface::get_file(input);
        match file {
            None => input_file.is_none(),
            Some(file) => input_file.is_some_and(|input_file| file == input_file),
        }
    }

    /// Java package-private `setBrowsingDirectory(BrowsingDirectory)`.
    pub fn set_browsing_directory(&self, browsing_dir: Option<Rc<dyn BrowsingDirectory>>) {
        *self.browsing_dir.borrow_mut() = browsing_dir;
    }

    /// Java private `getOriginDir()`: gets the origin directory.
    fn get_origin_dir(&self) -> Option<String> {
        let dir = self.get_origin_for_file_chooser();
        if dir.is_some() {
            return dir;
        }
        if let Some(manager) = self.manager {
            return manager.get_property_user_dir();
        }
        None
    }

    /// Java private `getOriginForFileChooser()`.
    fn get_origin_for_file_chooser(&self) -> Option<String> {
        let origin_reference = self.origin_reference.borrow().clone();
        if let Some(origin_reference) = origin_reference {
            if !Field::is_empty(&*origin_reference) {
                let dir = FileTextFieldInterface::get_file(&*origin_reference);
                if let Some(dir) = dir {
                    if dir.is_dir() {
                        return Some(utilities::java_io_file_get_absolute_path(
                            &dir.to_string_lossy(),
                        ));
                    } else {
                        return utilities::java_io_file_get_parent(&dir.to_string_lossy());
                    }
                }
            }
        }
        let origin = self.origin.borrow().clone();
        if let Some(origin) = origin {
            if origin.exists() && origin.is_dir() {
                return Some(utilities::java_io_file_get_absolute_path(
                    &origin.to_string_lossy(),
                ));
            }
        }
        if self.manager.is_none() || self.origin_etomo_run_dir.get() {
            return etomo_director::INSTANCE.get_original_user_dir();
        }
        None
    }

    /// Java private `getFileChooserLocation()`.
    fn get_file_chooser_location(&self) -> Option<String> {
        if self.use_text_as_file_chooser_dir.get() {
            let file = FileTextFieldInterface::get_file(self);
            if let Some(file) = file {
                let mut dir = ValidDirectory::new(self.manager);
                dir.set_file(Some(&file));
                if !dir.is_null() {
                    return dir.get_void().map(|dir| {
                        utilities::java_io_file_get_absolute_path(&dir.to_string_lossy())
                    });
                }
            }
        }
        self.get_origin_for_file_chooser()
    }

    /// Java package-private `setFileSelectionMode(int)`: sets the file selection mode
    /// to be used in the file chooser.
    pub fn set_file_selection_mode(&self, input: i32) {
        if input != file_chooser::FILES_ONLY
            && input != file_chooser::DIRECTORIES_ONLY
            && input != file_chooser::FILES_AND_DIRECTORIES
        {
            eprintln!("WARNING: Incorrect file chooser file selection mode: {input}");
            return;
        }
        self.file_selection_mode.set(input);
    }

    /// Java package-private `setTurnOffFileHiding(boolean)`.
    pub fn set_turn_off_file_hiding(&self, input: bool) {
        self.turn_off_file_hiding.set(input);
    }

    /// Java public `setFileFilter(FileFilter)`.
    pub fn set_file_filter(&self, input: Option<Rc<dyn FileFilter>>) {
        *self.file_filter.borrow_mut() = input;
    }

    /// Java package-private `setRequired(boolean)`.
    pub fn set_required(&self, required: bool) {
        self.field.set_required(required);
    }

    /// Java package-private `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        self.field.set_text_string(text);
    }

    /// Java package-private `setText(String, boolean)`.
    pub fn set_text_string_boolean(&self, text: Option<&str>, allow_empty: bool) {
        if allow_empty || text.is_some_and(|text| !text.is_empty()) {
            self.set_text_string(text);
        }
    }

    /// Java package-private `setTextEntryPolicy(boolean)`.
    pub fn set_text_entry_policy(&self, input: bool) {
        if !input {
            self.field.set_editable(false);
            self.text_entry_policy.set(false);
        } else {
            self.text_entry_policy.set(true);
        }
    }

    /// Java package-private `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
        self.field.set_enabled(enabled); // field handles enabled versus editable
        // Only visually enabled if both enabled and editable
        self.button
            .get_component()
            .set_enabled(enabled && self.editable.get());
    }

    /// Java package-private `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if self.text_entry_policy.get() {
            self.editable.set(editable);
            self.field.set_editable(editable); // field handles enabled versus editable
        }
        // Editable has no visible effect if the field is disabled.
        if self.enabled.get() {
            self.button.get_component().set_enabled(editable);
        }
    }

    /// Java package-private `setFieldToolTipText(String)`.
    pub fn set_field_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.field, text);
    }

    /// Java package-private `setButtonToolTipText(String)`.
    pub fn set_button_tool_tip_text(&self, text: Option<&str>) {
        self.button
            .get_component()
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }
}

impl Field for FileTextField2 {
    /// Java `isDebug()`.
    fn is_debug(&self) -> bool {
        ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        Field::get_name(&*self.field)
    }

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        false
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        true
    }

    /// Java `getQuotedLabel()`: a label suitable for a message - in single quotes and
    /// truncated at the colon.
    fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(Some(&self.label.get_text()))
    }

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `clear()`.
    fn clear(&self) {
        self.field.set_text_string(Some(""));
    }

    /// Java `setValue(Field)`.
    fn set_value_field(&self, input: Option<&dyn Field>) {
        self.field.set_value_field(input);
    }

    /// Java `setValue(String)`.
    fn set_value_string(&self, input: Option<&str>) {
        self.field.set_value_string(input);
    }

    /// Java `setValue(boolean)`: empty.
    fn set_value_boolean(&self, _input: bool) {}

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        let text = self.field.get_text_void();
        match text {
            None => true,
            Some(text) => java_lang_string_matches_whitespace(&text),
        }
    }

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool {
        false
    }

    /// Java `isRequired()`.
    fn is_required(&self) -> bool {
        self.field.is_required()
    }

    /// Java `getText()`.
    fn get_text_void(&self) -> Option<String> {
        self.field.get_text_void()
    }

    /// Java `getText(boolean, FieldDisplayer) throws FieldValidationFailedException`.
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, field_displayer1, None)
    }

    /// Java `getText(boolean, FieldDisplayer, FieldDisplayer) throws
    /// FieldValidationFailedException`.
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.field.get_text_boolean_field_displayer_field_displayer(
            do_validation,
            field_displayer1,
            field_displayer2,
        )
    }

    /// Java `getDirectiveDef()`.
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.field.get_directive_def()
    }

    /// Java `useDefaultValue()`.
    fn use_default_value(&self) {
        self.field.use_default_value();
    }

    /// Java `equalsDefaultValue()`.
    fn equals_default_value_void(&self) -> bool {
        self.field.equals_default_value_void()
    }

    /// Java `equalsDefaultValue(String)`.
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        self.field.equals_default_value_string(value)
    }

    /// Java `backup()`.
    fn backup(&self) {
        Field::backup(&*self.field);
    }

    /// Java `restoreFromBackup()`: if the field was backed up, make the backup value
    /// the displayed value, and turn off the back up.
    fn restore_from_backup(&self) {
        Field::restore_from_backup(&*self.field);
    }

    /// Java `checkpoint()`: saves the current text as the checkpoint.
    fn checkpoint(&self) {
        self.field.checkpoint_void();
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        Field::set_checkpoint(&*self.field, input);
    }

    /// Java `getCheckpoint()`.
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        Field::get_checkpoint(&*self.field)
    }

    /// Java `isDifferentFromCheckpoint(boolean)`: `alwaysCheck` - check for difference
    /// even when the field is disabled or invisible.
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        Field::is_different_from_checkpoint(&*self.field, always_check)
    }

    /// Java `isFieldHighlightSet()`.
    fn is_field_highlight_set(&self) -> bool {
        self.field.is_field_highlight_set()
    }

    /// Java `clearFieldHighlight()`.
    fn clear_field_highlight(&self) {
        self.field.clear_field_highlight();
    }

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    fn set_field_highlight_field_setting_interface(
        &self,
        setting_interface: Option<&dyn FieldSettingInterface>,
    ) {
        self.field
            .set_field_highlight_field_setting_interface(setting_interface);
    }

    /// Java `setFieldHighlight(String)`.
    fn set_field_highlight_string(&self, value: Option<&str>) {
        self.field.set_field_highlight_string(value);
    }

    /// Java `setFieldHighlight(boolean)`: empty.
    fn set_field_highlight_boolean(&self, _value: bool) {}

    /// Java `getFieldHighlight()` (a `TextFieldSetting`).
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        Field::get_field_highlight(&*self.field)
    }

    /// Java `equalsFieldHighlight()`.
    fn equals_field_highlight_void(&self) -> bool {
        self.field.equals_field_highlight_void()
    }

    /// Java `equalsFieldHighlight(String)`.
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        self.field.equals_field_highlight_string(value)
    }

    /// Java `setToolTipText(String)`.
    fn set_tool_tip_text(&self, text: Option<&str>) {
        Field::set_tool_tip_text(&*self.field, text);
        let text = tooltip_formatter::INSTANCE.format(text);
        self.panel.set_tool_tip_text(text.as_deref());
        self.button
            .get_component()
            .set_tool_tip_text(text.as_deref());
    }

    /// Java `setTooltip(Field)`.
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            let tooltip = field.get_tooltip();
            self.field.set_preformatted_tooltip(tooltip.as_deref());
            self.panel.set_tool_tip_text(tooltip.as_deref());
            self.button
                .get_component()
                .set_tool_tip_text(tooltip.as_deref());
        }
    }

    /// Java `getTooltip()`.
    fn get_tooltip(&self) -> Option<String> {
        Field::get_tooltip(&*self.field)
    }

    /// Java `equalsSelectedStringValue(String)`: not true value is implemented so all
    /// non-empty values are true.
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.field.set_directive_def(directive_def);
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> String {
        Field::get_description(&*self.field)
    }

    /// Java `setUnformattedTooltip(String)`.
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        *self.unformatted_tooltip.borrow_mut() = text.map(str::to_owned);
        self.field.set_unformatted_tooltip(text);
        self.unformatted_tooltip.borrow().clone()
    }

    /// Java `hasUnformattedTooltip()`.
    fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.borrow().is_some() || self.field.has_unformatted_tooltip()
    }

    /// Java synchronized `useUnformattedTooltip(String, String)`: use
    /// unformattedTooltip to build a tooltip, and then delete unformattedTooltip.
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        let unformatted_tooltip = self.unformatted_tooltip.borrow().clone();
        let tooltip = tooltip_formatter::INSTANCE.build_tooltip(
            unformatted_tooltip.as_deref(),
            param_descr,
            directive_descr,
        );
        Field::set_tool_tip_text(self, tooltip.as_deref());
        *self.unformatted_tooltip.borrow_mut() = None;
        self.field
            .use_unformatted_tooltip(param_descr, directive_descr);
    }
}

impl TextFieldInterface for FileTextField2 {}

impl FileTextFieldInterface for FileTextField2 {
    /// Java `getFile()`.
    fn get_file(&self) -> Option<PathBuf> {
        if !Field::is_empty(self) {
            return Some(FilePath::build_absolute_file_string_string(
                self.get_origin_dir().as_deref(),
                &self.field.get_text_void().unwrap_or_default(),
            ));
        }
        None
    }

    /// Java `setFile(File)`: adds the text of the file path to the field.  The file
    /// path will be either absolute or relative depending on the member variable
    /// absolutePath.  The directory will be set to propertyUserDir, unless the member
    /// variable originEtomoRunDir is true.  The directory will be used as the origin
    /// when building a relative file, or when building an absolute file out of a
    /// relative file.
    ///
    /// Upstream bug fixed in translation (FileTextField2.java:666-667): with
    /// absolutePath on, a null file reaches `FilePath.buildAbsoluteFile(String, File)`,
    /// which dereferences it (NullPointerException).  Here a null file clears the text
    /// in that branch.  The relative branch keeps the source's behaviour
    /// (`getRelativePath` of a null file returns the origin directory).
    fn set_file(&self, file: Option<PathBuf>) {
        if self.absolute_path.get() {
            match file {
                Some(file) => {
                    let path = FilePath::build_absolute_file_string_file(
                        self.get_origin_dir().as_deref(),
                        &file,
                    );
                    self.field.set_text_string(Some(&path.to_string_lossy()));
                }
                None => self.field.set_text_string(Some("")),
            }
        } else {
            let text =
                FilePath::get_relative_path(self.get_origin_dir().as_deref(), file.as_deref());
            self.field.set_text_string(text.as_deref());
        }
    }

    /// Java `getFileFilter()`.
    fn get_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        self.file_filter.borrow().clone()
    }
}

impl UIComponent for FileTextField2 {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.panel.clone()
    }
}

impl SwingComponent for FileTextField2 {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.panel.clone()
    }
}
