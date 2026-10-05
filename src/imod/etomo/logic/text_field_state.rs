//! `IMOD/Etomo/src/etomo/logic/TextFieldState.java`.
//!
//! The expand/contract state of a text field that can hold a file path.  A plain value
//! held by its field (`FieldCell`) in a cell; mutating methods take `&mut self`.

use std::path::{Path, PathBuf};

use regex::Regex;

use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, INTEGER_NULL_VALUE, Type, java_lang_integer_parse_int,
    java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::parsed_element_type::ParsedElementType;
use crate::imod::etomo::util::file_path::FilePath;
use crate::imod::etomo::util::utilities::{
    java_io_file_get_absolute_path, java_io_file_get_parent, java_io_file_new,
};

/// Java `TextFieldState`.
#[derive(Clone, Debug)]
pub struct TextFieldState {
    /// Java private final `editableField`.
    editable_field: bool,
    /// Java private final `parsedElementType`.
    parsed_element_type: Option<&'static ParsedElementType>,
    /// Java private final `rootDir`.
    root_dir: Option<String>,
    /// Java private `debug`, initialised to false.
    debug: bool,
    /// Java private `expanded`, initialised to true.
    expanded: bool,
    /// Java private `parent`, initialised to null.
    parent: Option<PathBuf>,
}

impl TextFieldState {
    /// Java `TextFieldState(boolean, ParsedElementType, String)`.
    pub fn new_boolean_parsed_element_type_string(
        editable_field: bool,
        parsed_element_type: Option<&'static ParsedElementType>,
        root_dir: Option<&str>,
    ) -> TextFieldState {
        TextFieldState {
            editable_field,
            parsed_element_type,
            root_dir: root_dir.map(|root_dir| root_dir.to_string()),
            debug: false,
            expanded: true,
            parent: None,
        }
    }

    /// Java `TextFieldState(TextFieldState)`.  (`debug` is not copied.)
    pub fn new_text_field_state(text_field_state: &TextFieldState) -> TextFieldState {
        TextFieldState {
            editable_field: text_field_state.editable_field,
            parsed_element_type: text_field_state.parsed_element_type,
            root_dir: text_field_state.root_dir.clone(),
            debug: false,
            expanded: text_field_state.expanded,
            parent: text_field_state.parent.clone(),
        }
    }

    /// Java `isEditableField()`.
    pub fn is_editable_field(&self) -> bool {
        self.editable_field
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java `expandFieldText(boolean, String)`.  Applies the expand state to the text
    /// and returns it.
    pub fn expand_field_text(&mut self, expand: bool, text: Option<&str>) -> Option<String> {
        if self.expanded == expand {
            return text.map(|text| text.to_string());
        }
        self.expanded = expand;
        self.apply_expanded_to_field_text(text)
    }

    /// Java `applyExpandedToFieldText(String)`.  When expanded, return the file with
    /// its path.  When contracted, return the file name.  The only change that needs to
    /// be handled is if a path was added to the text while it was contracted.  Paths
    /// entered by hand are not modified.  Paths entered using the button are made
    /// relative to the rootDir.
    ///
    /// Upstream bug fixed (TextFieldState.java:83-87): expanding a null text while a
    /// parent is saved calls `new File(parent, null)`, which throws a
    /// NullPointerException.  A null text has no file name to put back under the
    /// parent, so it takes the "nothing is saved" path: the parent is dropped and null
    /// is returned.
    pub fn apply_expanded_to_field_text(&mut self, text: Option<&str>) -> Option<String> {
        if !self.expanded {
            // Contract
            // Save the path when contracting.
            self.parent = FilePath::get_file_parent(text);
            // Set the contracted form of the text
            return FilePath::get_file_name(text);
        }
        // Expand
        if !FilePath::is_path(text)
            && self.parent.is_some()
            && let Some(text) = text
        {
            // The text has changed, but is doesn't include a path, so replace the old
            // file name with the new one in the path.
            return Some(java_io_file_new(
                &self.parent.as_ref().unwrap().to_string_lossy(),
                text,
            ));
        }
        // Nothing is saved while the field is expanded.
        self.parent = None;
        text.map(|text| text.to_string())
    }

    /// Java `msgResettingFieldText()`.
    pub fn msg_resetting_field_text(&mut self) {
        self.parent = None;
    }

    /// Java `convertToFieldText(File)`.  Sets the absolute path of the file if the root
    /// directory is empty, otherwise sets the relative path from the root directory to
    /// the file.
    pub fn convert_to_field_text_file(&mut self, file: Option<&Path>) -> Option<String> {
        let file = file?;
        if self.root_dir.is_none() {
            return Some(
                self.convert_to_field_text_string(Some(&java_io_file_get_absolute_path(
                    &file.to_string_lossy(),
                ))),
            );
        }
        let relative_path = FilePath::get_relative_path(self.root_dir.as_deref(), Some(file));
        Some(self.convert_to_field_text_string(relative_path.as_deref()))
    }

    /// Java `convertToFieldText(String)`.  Returns string, except when string is a file
    /// path and the state is contracted.  In that case it returns the string's file
    /// name.  When the instance is contracted the strings parent path is stored in the
    /// parent member variable.
    pub fn convert_to_field_text_string(&mut self, string: Option<&str>) -> String {
        let Some(string) = string else {
            return String::new();
        };
        if self.expanded || !FilePath::is_path(Some(string)) {
            // Set value as is, unless the field is contracted and its a file path.
            self.parent = None;
            return string.to_string();
        }
        // Set a contracted file path.
        self.parent = java_io_file_get_parent(string).map(PathBuf::from);
        FilePath::get_file_name(Some(string)).unwrap_or_default()
    }

    /// Java `convertRangeToFieldText(int, int)`.
    pub fn convert_range_to_field_text(&mut self, start: i32, end: i32) -> String {
        self.convert_to_field_text_string(Some(&format!("{} - {}", start, end)))
    }

    /// Java `convertToContractedString(String)`.
    pub fn convert_to_contracted_string(&self, text: Option<&str>) -> Option<String> {
        if !self.expanded {
            return text.map(|text| text.to_string());
        }
        FilePath::get_file_name(text)
    }

    /// Java `convertToExpandedString(String)`.  Return text.  Or if text is contracted
    /// and is not a path, and the parent exists, return the parent plus the text.
    pub fn convert_to_expanded_string(&mut self, text: Option<&str>) -> String {
        let text = match text {
            Some(text) if !java_lang_string_matches_whitespace(text) => text,
            _ => {
                self.parent = None;
                return String::new();
            }
        };
        if self.expanded || self.parent.is_none() {
            return text.to_string();
        }
        if FilePath::is_path(Some(text)) {
            return text.to_string();
        }
        java_io_file_new(&self.parent.as_ref().unwrap().to_string_lossy(), text)
    }

    /// Java `extractEndValue(String)`.  Parse and return the second number in an
    /// "n - m" string.  Return the null value from EtomoNumber if the format is wrong.
    pub fn extract_end_value(&self, text: Option<&str>) -> i32 {
        let Some(text) = text else {
            return INTEGER_NULL_VALUE;
        };
        let text = java_lang_string_trim(text);
        // `text.matches("\\S+\\s*-\\s*\\S+")`, with Java's `\s` = [ \t\n\x0B\f\r].
        let range = Regex::new(
            r"^[^ \t\n\x0B\x0C\r]+[ \t\n\x0B\x0C\r]*-[ \t\n\x0B\x0C\r]*[^ \t\n\x0B\x0C\r]+$",
        )
        .unwrap();
        if text.chars().count() <= 1 || !range.is_match(text) {
            return INTEGER_NULL_VALUE;
        }
        let mut end_value = EtomoNumber::new();
        // Avoid getting the "-" from a negative number by searching from the second
        // character.  (The pattern guarantees a '-' after the first character.)
        let index = text
            .char_indices()
            .skip(1)
            .find(|(_, c)| *c == '-')
            .map(|(index, _)| index)
            .unwrap();
        end_value.set_string(Some(&text[index + 1..]));
        // Returns null if the string set was not a valid integer.
        end_value.get_int()
    }

    /// Java `convertToEtomoNumber(String)`.
    pub fn convert_to_etomo_number_string(&self, text: Option<&str>) -> ConstEtomoNumber {
        let mut number = EtomoNumber::new();
        number.set_string(text);
        number.base
    }

    /// Java `convertToEtomoNumber(EtomoNumber.Type, String)`.
    pub fn convert_to_etomo_number_type_string(
        &self,
        r#type: Option<Type>,
        text: Option<&str>,
    ) -> ConstEtomoNumber {
        let mut number = EtomoNumber::new_with_type(r#type);
        number.set_string(text);
        number.base
    }

    /// Java `convertToInt(String)`: `new Integer(text).intValue()`, 0 on a
    /// NumberFormatException (which a null text also raises).
    pub fn convert_to_int(&self, text: Option<&str>) -> i32 {
        let Some(text) = text else {
            return 0;
        };
        match java_lang_integer_parse_int(text) {
            Ok(value) => value,
            Err(_) => 0,
        }
    }
}
