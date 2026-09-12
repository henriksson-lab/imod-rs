//! `IMOD/Etomo/src/etomo/ui/swing/ValueManipulationExtension.java`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use crate::imod::etomo::ui::field_type::FieldType;

/// Java `ValueManipulationField`, at this source unit's interface boundary.
pub trait ValueManipulationField {
    fn is_empty(&self) -> bool;
    fn set_text(&mut self, text: String);
    fn add_value_manipulation_listener(&mut self);
}

/// Java package-private final `ValueManipulationExtension`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueManipulationExtension {
    prevent_blank: bool,
    substitute_value: Option<String>,
    limit_displayed_file_path: bool,
    max_file_path_size: i32,
    full_file_path: Option<String>,
    debug: bool,
    default_to_filename: bool,
}

impl ValueManipulationExtension {
    pub const PREFIX: &'static str = "...";
    pub const EXCLUDE_POST_PREFIX_SEPARATOR: i32 = 1;

    /// Java `ValueManipulationExtension(ValueManipulationField, boolean)`.
    pub fn new<F: ValueManipulationField>(field: &mut F, debug: bool) -> Self {
        field.add_value_manipulation_listener();
        Self {
            prevent_blank: false,
            substitute_value: None,
            limit_displayed_file_path: false,
            max_file_path_size: -1,
            full_file_path: None,
            debug,
            default_to_filename: false,
        }
    }

    /// Java `setLimitDisplayedFilePath(int, FieldType)`.
    pub fn set_limit_displayed_file_path(
        &mut self,
        max_file_path_size: i32,
        field_type: Option<FieldType>,
    ) -> bool {
        let old_limit_displayed_file_path = self.limit_displayed_file_path;
        let old_max_file_path_size = self.max_file_path_size;
        if field_type != Some(FieldType::File) {
            self.max_file_path_size = -1;
            self.limit_displayed_file_path = false;
        } else {
            self.max_file_path_size = max_file_path_size;
            self.limit_displayed_file_path = max_file_path_size > 0;
        }
        old_limit_displayed_file_path != self.limit_displayed_file_path
            || old_max_file_path_size != self.max_file_path_size
    }

    /// Java overloaded `createDisplayedFilePath(File)`.
    pub fn create_displayed_file_path_file(&mut self, file: Option<&Path>) -> Option<String> {
        let file = file?;
        let absolute_path = if file.is_absolute() {
            file.to_string_lossy().into_owned()
        } else {
            std::env::current_dir()
                .ok()?
                .join(file)
                .to_string_lossy()
                .into_owned()
        };
        self.create_displayed_file_path(Some(&absolute_path), Some(FieldType::File))
    }

    /// Java overloaded `createDisplayedFilePath(String, FieldType)`.
    pub fn create_displayed_file_path(
        &mut self,
        string: Option<&str>,
        field_type: Option<FieldType>,
    ) -> Option<String> {
        let string = string?;
        let string_len = string.len() as i32;
        let separator = std::path::MAIN_SEPARATOR;
        if field_type != Some(FieldType::File)
            || string_len <= self.max_file_path_size
            || !string.contains(separator)
        {
            self.full_file_path = None;
            return Some(string.to_owned());
        }
        if self.default_to_filename {
            let current_file = Path::new(string);
            if current_file
                .parent()
                .and_then(|parent| parent.canonicalize().ok())
                == std::env::current_dir().ok()
            {
                self.full_file_path = Some(string.to_owned());
                return current_file
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned());
            }
        }
        if !self.limit_displayed_file_path {
            self.full_file_path = None;
            return Some(string.to_owned());
        }
        self.full_file_path = Some(string.to_owned());
        let start =
            (string_len - (self.max_file_path_size - Self::PREFIX.len() as i32)).max(0) as usize;
        let separator_index = string[start..].find(separator).map(|index| start + index);
        if separator_index.is_none() {
            return string
                .rfind(separator)
                .map(|index| format!("{}{}", Self::PREFIX, &string[index..]));
        }
        Some(format!(
            "{}{}",
            Self::PREFIX,
            &string[separator_index.unwrap()..]
        ))
    }

    /// Java `clearFullFilePath()`.
    pub fn clear_full_file_path(&mut self) {
        self.full_file_path = None;
    }

    /// Java `getFullFilePath(String)`.
    pub fn get_full_file_path(&self, displayed_string: String) -> String {
        if !self.limit_displayed_file_path || self.full_file_path.is_none() {
            displayed_string
        } else {
            self.full_file_path.clone().unwrap()
        }
    }

    /// Java `setPreventBlank(boolean, String)`.
    pub fn set_prevent_blank(&mut self, prevent_blank: bool, substitute_value: Option<String>) {
        self.prevent_blank = prevent_blank;
        self.substitute_value = substitute_value;
    }

    /// Java `clearPreventBlank()`.
    pub fn clear_prevent_blank(&mut self) {
        self.prevent_blank = false;
        self.substitute_value = None;
    }

    /// Java `substitute()`; field ownership is passed explicitly in Rust.
    pub fn substitute<F: ValueManipulationField>(&self, field: &mut F) {
        if self.prevent_blank && field.is_empty() {
            if let Some(value) = &self.substitute_value {
                field.set_text(value.clone());
            }
        }
    }

    /// Java `focusLost(FocusEvent)`; native event installation is a GUI boundary.
    pub fn focus_lost<F: ValueManipulationField>(&self, field: &mut F) {
        self.substitute(field);
    }

    /// Java `focusGained(FocusEvent)`.
    pub fn focus_gained(&self) {}

    /// Java `setDefaultToFilename(boolean)`.
    pub fn set_default_to_filename(&mut self, default_to_filename: bool) {
        self.default_to_filename = default_to_filename;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Field {
        text: String,
        listener: bool,
    }
    impl ValueManipulationField for Field {
        fn is_empty(&self) -> bool {
            self.text.is_empty()
        }
        fn set_text(&mut self, text: String) {
            self.text = text;
        }
        fn add_value_manipulation_listener(&mut self) {
            self.listener = true;
        }
    }
    #[test]
    fn source_path_limit_retains_full_path() {
        let mut field = Field::default();
        let mut extension = ValueManipulationExtension::new(&mut field, false);
        extension.set_limit_displayed_file_path(4, Some(FieldType::File));
        assert_eq!(
            extension
                .create_displayed_file_path(Some("/one/two/file.mrc"), Some(FieldType::File))
                .as_deref(),
            Some(".../file.mrc")
        );
        assert_eq!(
            extension.get_full_file_path(".../file.mrc".to_owned()),
            "/one/two/file.mrc"
        );
    }
    #[test]
    fn source_substitute_runs_on_focus_loss() {
        let mut field = Field::default();
        let mut extension = ValueManipulationExtension::new(&mut field, false);
        extension.set_prevent_blank(true, Some("template".to_owned()));
        extension.focus_lost(&mut field);
        assert!(field.listener);
        assert_eq!(field.text, "template");
    }
}
