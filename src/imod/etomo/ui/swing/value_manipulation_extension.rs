//! `IMOD/Etomo/src/etomo/ui/swing/ValueManipulationExtension.java`.
//!
//! Changes the value a field displays: shortening a long file path, or substituting a
//! value for a blank entry when the field loses the focus.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::value_manipulation_field::ValueManipulationField;
use crate::imod::etomo::ui::value_manipulation_listener::ValueManipulationListener;

/// Java private `PREFIX`.
const PREFIX: &str = "...";
/// Java private `EXCLUDE_POST_PREFIX_SEPARATOR` (unused in the Java too).
#[allow(dead_code)]
const EXCLUDE_POST_PREFIX_SEPARATOR: i32 = 1;
/// Java `File.separatorChar`.
const SEPARATOR_CHAR: char = std::path::MAIN_SEPARATOR;

/// Java `ValueManipulationExtension`.
pub struct ValueManipulationExtension {
    /// Java `field`: the field that owns this extension (so a `Weak`).
    field: Weak<dyn ValueManipulationField>,
    /// Java `preventBlank`.
    prevent_blank: Cell<bool>,
    /// Java `substituteValue`.
    substitute_value: RefCell<Option<String>>,
    /// Java `limitDisplayedFilePath`.
    limit_displayed_file_path: Cell<bool>,
    /// Java `maxFilePathSize`.
    max_file_path_size: Cell<i32>,
    /// Java `fullFilePath`.
    full_file_path: RefCell<Option<String>>,
    /// Java `debug`.
    #[allow(dead_code)]
    debug: Cell<bool>,
    /// Java `defaultToFilename`.
    default_to_filename: Cell<bool>,
}

impl ValueManipulationExtension {
    /// Java `ValueManipulationExtension(ValueManipulationField, boolean)`.
    ///
    /// `field_ref` is the same object as `field`, reached directly: Java's
    /// constructor calls `field.addValueManipulationListener(this)`, which the
    /// translation must also do while the field is still being built (when
    /// `field` cannot be upgraded yet).
    pub fn new(
        field: Weak<dyn ValueManipulationField>,
        field_ref: &dyn ValueManipulationField,
        debug: bool,
    ) -> Rc<ValueManipulationExtension> {
        let extension = Rc::new(ValueManipulationExtension {
            field,
            prevent_blank: Cell::new(false),
            substitute_value: RefCell::new(None),
            limit_displayed_file_path: Cell::new(false),
            max_file_path_size: Cell::new(-1),
            full_file_path: RefCell::new(None),
            debug: Cell::new(debug),
            default_to_filename: Cell::new(false),
        });
        field_ref.add_value_manipulation_listener(extension.clone() as Rc<dyn ValueManipulationListener>);
        extension
    }

    /// Java `setLimitDisplayedFilePath(int, FieldType)`.  Limit the size of the display
    /// value.  Truncates only on the system file separator.  Returns true if the
    /// displayed text might have to be shortened or lengthened.
    pub fn set_limit_displayed_file_path(
        &self,
        max_file_path_size: i32,
        field_type: Option<FieldType>,
    ) -> bool {
        let old_limit_displayed_file_path = self.limit_displayed_file_path.get();
        let old_max_file_path_size = self.max_file_path_size.get();
        if field_type != Some(FieldType::File) {
            self.max_file_path_size.set(-1);
            self.limit_displayed_file_path.set(false);
        } else {
            self.max_file_path_size.set(max_file_path_size);
            self.limit_displayed_file_path.set(max_file_path_size > 0);
        }
        old_limit_displayed_file_path != self.limit_displayed_file_path.get()
            || old_max_file_path_size != self.max_file_path_size.get()
    }

    /// Java `createDisplayedFilePath(File)`.
    pub fn create_displayed_file_path_file(&self, file: Option<&Path>) -> Option<String> {
        let file = file?;
        let absolute_path = std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf());
        self.create_displayed_file_path_string_field_type(
            Some(&absolute_path.to_string_lossy()),
            Some(FieldType::File),
        )
    }

    /// Java `createDisplayedFilePath(String, FieldType)`.  If limitDisplayedFilePath is
    /// set, returns a limited size version of a file's absolute path (truncated on the
    /// file separator), saving the original string in fullFilePath.
    pub fn create_displayed_file_path_string_field_type(
        &self,
        string: Option<&str>,
        field_type: Option<FieldType>,
    ) -> Option<String> {
        let string = string?;
        // Java indexes by UTF-16 unit; chars are used here.
        let chars: Vec<char> = string.chars().collect();
        let string_len = chars.len() as i32;
        // See if the possibleFilePath should be returned without truncation.
        // Truncation is only done on the file separator.
        if field_type != Some(FieldType::File)
            || string_len <= self.max_file_path_size.get()
            || !chars.contains(&SEPARATOR_CHAR)
        {
            *self.full_file_path.borrow_mut() = None;
            return Some(string.to_string());
        }
        // Check local here only is defaulttoFilename is true
        if self.default_to_filename.get() {
            let curr_file = PathBuf::from(string);
            // Fixed in translation (`ValueManipulationExtension.java:117`): Java calls
            // getParentFile().getAbsolutePath() and throws a NullPointerException for a
            // path with no parent; such a path is treated as not in the current
            // directory.
            let parent = curr_file
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty());
            let user_dir = std::env::current_dir().ok();
            if let (Some(parent), Some(user_dir)) = (parent, user_dir) {
                let parent = std::path::absolute(parent).unwrap_or_else(|_| parent.to_path_buf());
                if parent == user_dir {
                    *self.full_file_path.borrow_mut() = Some(string.to_string());
                    return Some(
                        curr_file
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    );
                }
            }
        }
        if !self.limit_displayed_file_path.get() {
            *self.full_file_path.borrow_mut() = None;
            Some(string.to_string())
        } else {
            // The displayed string will be shortened so save the full string.
            *self.full_file_path.borrow_mut() = Some(string.to_string());
            // Find the first separator within the area that can be retained.
            let from = (string_len - (self.max_file_path_size.get() - PREFIX.len() as i32)).max(0);
            let separator_index = chars
                .iter()
                .skip(from as usize)
                .position(|&c| c == SEPARATOR_CHAR)
                .map(|i| i + from as usize);
            let Some(separator_index) = separator_index else {
                // File name is longer then the area that can be retained. Return the file
                // name.
                let last = chars.iter().rposition(|&c| c == SEPARATOR_CHAR).unwrap();
                return Some(PREFIX.to_string() + &chars[last..].iter().collect::<String>());
            };
            // Return a shortened file path.
            Some(PREFIX.to_string() + &chars[separator_index..].iter().collect::<String>())
        }
    }

    /// Java `clearFullFilePath()`.
    pub fn clear_full_file_path(&self) {
        *self.full_file_path.borrow_mut() = None;
    }

    /// Java `getFullFilePath(String)`.  Returns the saved filePath if
    /// limitDisplayedFilePath is on and the filePath is not null.  Otherwise returns
    /// displayedString.
    pub fn get_full_file_path(&self, displayed_string: Option<&str>) -> Option<String> {
        let full_file_path = self.full_file_path.borrow().clone();
        if !self.limit_displayed_file_path.get() || full_file_path.is_none() {
            return displayed_string.map(str::to_owned);
        }
        full_file_path
    }

    /// Java `setPreventBlank(boolean, String)`.
    pub fn set_prevent_blank(&self, prevent_blank: bool, substitute_value: Option<&str>) {
        self.prevent_blank.set(prevent_blank);
        *self.substitute_value.borrow_mut() = substitute_value.map(str::to_owned);
    }

    /// Java `clearPreventBlank()`.
    pub fn clear_prevent_blank(&self) {
        self.prevent_blank.set(false);
        *self.substitute_value.borrow_mut() = None;
    }

    /// Java `substitute()`.  Sets substituteValue in field if preventBlank is true and
    /// field is empty.
    pub fn substitute(&self) {
        let Some(field) = self.field.upgrade() else {
            return;
        };
        let substitute_value = self.substitute_value.borrow().clone();
        if self.prevent_blank.get() && field.is_empty() {
            if let Some(substitute_value) = substitute_value {
                field.set_text(Some(&substitute_value));
            }
        }
    }

    /// Java `setDefaultToFilename(boolean)`.
    pub fn set_default_to_filename(&self, default_to_filename: bool) {
        self.default_to_filename.set(default_to_filename);
    }
}

impl ValueManipulationListener for ValueManipulationExtension {
    /// Java `focusLost(FocusEvent)`.
    fn focus_lost(&self) {
        self.substitute();
    }

    /// Java `focusGained(FocusEvent)`.
    fn focus_gained(&self) {}
}
