//! `IMOD/Etomo/src/etomo/storage/DirectiveDescrElement.java`.
//!
//! One line of the directives description file (`$IMOD_DIR/com/directives.csv`), split
//! into its columns.  Java's static `String[]` overloads carry the suffix
//! `_from_line_array`; the instance methods keep the plain names.  A Java `String[]`
//! produced by `String.split` never holds a null element, so the array is
//! `Option<Vec<String>>` (null array) with non-null elements.
#![allow(dead_code)]

use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_descr::DirectiveDescr;
use crate::imod::etomo::storage::directive_descr_choice_list::DirectiveDescrChoiceList;
use crate::imod::etomo::storage::directive_descr_etomo_column::DirectiveDescrEtomoColumn;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;

/// Java private `COMMENT`.
const COMMENT: &str = "#";
/// Java private `HEADER_STRING`.
const HEADER_STRING: &str = "Directives";
/// Java private `NAME_COLUMN_INDEX`.
const NAME_COLUMN_INDEX: usize = 0;
/// Java private `DESCR_COLUMN_INDEX`.
const DESCR_COLUMN_INDEX: usize = 1;
/// Java private `VALUE_TYPE_COLUMN_INDEX`.
const VALUE_TYPE_COLUMN_INDEX: usize = 2;
/// Java private `BATCH_COLUMN_INDEX`.
const BATCH_COLUMN_INDEX: usize = 3;
/// Java private `TEMPLATE_COLUMN_INDEX`.
const TEMPLATE_COLUMN_INDEX: usize = 4;
/// Java private `ETOMO_COLUMN_INDEX`.
const ETOMO_COLUMN_INDEX: usize = 5;
/// Java private `NOTE_COLUMN_INDEX`.
const NOTE_COLUMN_INDEX: usize = 6;
/// Java private `LABEL_COLUMN_INDEX`.
const LABEL_COLUMN_INDEX: usize = 7;
/// Java private `CHOICES_COLUMN_INDEX`.
const CHOICES_COLUMN_INDEX: usize = 8;

/// Java `DirectiveDescrElement`.
#[derive(Clone, Debug, Default)]
pub struct DirectiveDescrElement {
    /// Java private field `lineArray`, initialised to null.
    line_array: Option<Vec<String>>,
}

/// `String.compareToIgnoreCase("Y") == 0`.  Only a one-character string can compare
/// equal to "Y", and no character other than `y`/`Y` upper- or lower-cases to `Y`/`y`.
fn java_compare_to_ignore_case_y(value: &str) -> bool {
    value.eq_ignore_ascii_case("Y")
}

impl DirectiveDescrElement {
    /// Java package-private `DirectiveDescrElement()`.
    pub(crate) fn new() -> DirectiveDescrElement {
        DirectiveDescrElement { line_array: None }
    }

    /// Java `DirectiveDescrElement(String[])`.
    pub fn new_with_line_array(line_array: Option<Vec<String>>) -> DirectiveDescrElement {
        DirectiveDescrElement { line_array }
    }

    /// Java package-private `getLineArray`.
    pub(crate) fn get_line_array(&self) -> Option<Vec<String>> {
        self.line_array.clone()
    }

    /// Java package-private `setLineArray`.
    pub(crate) fn set_line_array(&mut self, line_array: Option<Vec<String>>) {
        self.line_array = line_array;
    }

    /// Java static `getDirectiveDef(String[], DirectiveDef)`.
    pub fn get_directive_def(
        line_array: Option<&[String]>,
        prev_directive_def: Option<DirectiveDef>,
    ) -> Option<DirectiveDef> {
        if DirectiveDescrElement::is_directive_from_line_array(line_array) {
            return DirectiveDef::get_instance_from_csv(
                Some(&line_array.unwrap()[NAME_COLUMN_INDEX]),
                prev_directive_def,
            );
        }
        None
    }

    /// Java static `getNote(String[])`.
    pub fn get_note(line_array: Option<&[String]>) -> Option<String> {
        if let Some(line_array) = line_array
            && line_array.len() > NOTE_COLUMN_INDEX
        {
            return Some(line_array[NOTE_COLUMN_INDEX].clone());
        }
        None
    }

    /// Java static `getDescription(String[])`.
    pub fn get_description_from_line_array(line_array: Option<&[String]>) -> Option<String> {
        if let Some(line_array) = line_array
            && line_array.len() > DESCR_COLUMN_INDEX
        {
            return Some(line_array[DESCR_COLUMN_INDEX].clone());
        }
        None
    }

    /// Java `getDescription()` (DirectiveDescr).
    pub fn get_description(&self) -> Option<String> {
        DirectiveDescrElement::get_description_from_line_array(self.line_array.as_deref())
    }

    /// Java static `isLabel(String[])`.
    pub fn is_label(line_array: Option<&[String]>) -> bool {
        line_array.is_some_and(|line_array| {
            line_array.len() > LABEL_COLUMN_INDEX && line_array[LABEL_COLUMN_INDEX].len() > 0
        })
    }

    /// Java static `getLabel(String[])`.
    pub fn get_label_from_line_array(line_array: Option<&[String]>) -> Option<String> {
        if DirectiveDescrElement::is_label(line_array) {
            Some(line_array.unwrap()[LABEL_COLUMN_INDEX].clone())
        } else {
            None
        }
    }

    /// Java `getLabel()` (DirectiveDescr).
    pub fn get_label(&self) -> Option<String> {
        if let Some(line_array) = &self.line_array
            && line_array.len() > LABEL_COLUMN_INDEX
        {
            return Some(line_array[LABEL_COLUMN_INDEX].clone());
        }
        None
    }

    /// Java static `isChoiceList(String[])`.
    pub fn is_choice_list(line_array: Option<&[String]>) -> bool {
        line_array.is_some_and(|line_array| {
            line_array.len() > CHOICES_COLUMN_INDEX && line_array[CHOICES_COLUMN_INDEX].len() > 0
        })
    }

    /// Java static `getChoiceList(String[])`.
    pub fn get_choice_list_from_line_array(
        line_array: Option<&[String]>,
    ) -> Option<DirectiveDescrChoiceList> {
        if DirectiveDescrElement::is_choice_list(line_array) {
            return DirectiveDescrChoiceList::get_instance(Some(
                &line_array.unwrap()[CHOICES_COLUMN_INDEX],
            ));
        }
        None
    }

    /// Java `getChoiceList()` (DirectiveDescr).
    pub fn get_choice_list(&self) -> Option<DirectiveDescrChoiceList> {
        DirectiveDescrElement::get_choice_list_from_line_array(self.line_array.as_deref())
    }

    /// Java static `getEtomoColumn(String[])`.
    pub fn get_etomo_column_from_line_array(
        line_array: Option<&[String]>,
    ) -> Option<DirectiveDescrEtomoColumn> {
        if let Some(line_array) = line_array
            && line_array.len() > ETOMO_COLUMN_INDEX
        {
            return DirectiveDescrEtomoColumn::get_instance(&line_array[ETOMO_COLUMN_INDEX]);
        }
        None
    }

    /// Java `getEtomoColumn()` (DirectiveDescr).
    pub fn get_etomo_column(&self) -> Option<DirectiveDescrEtomoColumn> {
        DirectiveDescrElement::get_etomo_column_from_line_array(self.line_array.as_deref())
    }

    /// Java static `getName(String[])`.
    pub fn get_name_from_line_array(line_array: Option<&[String]>) -> Option<String> {
        if let Some(line_array) = line_array
            && line_array.len() > NAME_COLUMN_INDEX
        {
            return Some(line_array[NAME_COLUMN_INDEX].clone());
        }
        None
    }

    /// Java `getName()` (DirectiveDescr).
    pub fn get_name(&self) -> Option<String> {
        DirectiveDescrElement::get_name_from_line_array(self.line_array.as_deref())
    }

    /// Java `getSectionHeader()`.
    pub fn get_section_header(&self) -> Option<String> {
        DirectiveDescrElement::get_section_header_from_line_array(self.line_array.as_deref())
    }

    /// Java static `getSectionHeader(String[])`.
    pub fn get_section_header_from_line_array(line_array: Option<&[String]>) -> Option<String> {
        if let Some(line_array) = line_array
            && line_array.len() > NAME_COLUMN_INDEX
        {
            return Some(line_array[NAME_COLUMN_INDEX].clone());
        }
        None
    }

    /// Java static `getValueType(String[])`.
    pub fn get_value_type_from_line_array(line_array: Option<&[String]>) -> DirectiveValueType {
        if let Some(line_array) = line_array
            && line_array.len() > VALUE_TYPE_COLUMN_INDEX
        {
            return DirectiveValueType::get_instance(Some(&line_array[VALUE_TYPE_COLUMN_INDEX]));
        }
        DirectiveValueType::Unknown
    }

    /// Java `getValueType()` (DirectiveDescr).  The Java method never returns null; the
    /// `Option` is the shape `DirectiveDef.loadDirectiveDescr`'s translation reads, and
    /// it is always `Some`.
    pub fn get_value_type(&self) -> Option<DirectiveValueType> {
        Some(DirectiveDescrElement::get_value_type_from_line_array(
            self.line_array.as_deref(),
        ))
    }

    /// Java static `isIncluded(String[], DirectiveFileType)`.
    pub fn is_included(
        line_array: Option<&[String]>,
        directive_file_type: Option<DirectiveFileType>,
    ) -> bool {
        let directive_file_type = match directive_file_type {
            None => return true,
            Some(directive_file_type) => directive_file_type,
        };
        if directive_file_type.is_batch()
            && DirectiveDescrElement::is_batch_from_line_array(line_array)
        {
            return true;
        }
        if directive_file_type.is_template()
            && DirectiveDescrElement::is_template_from_line_array(line_array)
        {
            return true;
        }
        false
    }

    /// Java static `isBatch(String[])`.
    pub fn is_batch_from_line_array(line_array: Option<&[String]>) -> bool {
        if let Some(line_array) = line_array
            && line_array.len() > BATCH_COLUMN_INDEX
        {
            return java_compare_to_ignore_case_y(&line_array[BATCH_COLUMN_INDEX]);
        }
        false
    }

    /// Java `isBatch()` (DirectiveDescr).
    pub fn is_batch(&self) -> bool {
        DirectiveDescrElement::is_batch_from_line_array(self.line_array.as_deref())
    }

    /// Java `isDirective()`.
    pub fn is_directive(&self) -> bool {
        DirectiveDescrElement::is_directive_from_line_array(self.line_array.as_deref())
    }

    /// Java static `isDirective(String[])`.
    pub fn is_directive_from_line_array(line_array: Option<&[String]>) -> bool {
        line_array.is_some_and(|line_array| {
            line_array.len() >= 3 && line_array[NAME_COLUMN_INDEX].contains(SEPARATOR_CHAR)
        })
    }

    /// Java `isSection()`.
    pub fn is_section(&self) -> bool {
        DirectiveDescrElement::is_section_from_line_array(self.line_array.as_deref())
    }

    /// Java static `isSection(String[])`.
    pub fn is_section_from_line_array(line_array: Option<&[String]>) -> bool {
        line_array.is_some_and(|line_array| {
            line_array.len() == 1
                && !line_array[0].starts_with(COMMENT)
                && !line_array[0].contains(HEADER_STRING)
        })
    }

    /// Java static `isTemplate(String[])`.
    pub fn is_template_from_line_array(line_array: Option<&[String]>) -> bool {
        if let Some(line_array) = line_array
            && line_array.len() > TEMPLATE_COLUMN_INDEX
        {
            return java_compare_to_ignore_case_y(&line_array[TEMPLATE_COLUMN_INDEX]);
        }
        false
    }

    /// Java `isTemplate()` (DirectiveDescr).
    pub fn is_template(&self) -> bool {
        DirectiveDescrElement::is_template_from_line_array(self.line_array.as_deref())
    }
}

/// Java `implements DirectiveDescr`: the interface methods are the inherent ones.
impl DirectiveDescr for DirectiveDescrElement {
    fn get_name(&self) -> Option<String> {
        DirectiveDescrElement::get_name(self)
    }

    fn get_description(&self) -> Option<String> {
        DirectiveDescrElement::get_description(self)
    }

    fn get_value_type(&self) -> Option<DirectiveValueType> {
        DirectiveDescrElement::get_value_type(self)
    }

    fn is_batch(&self) -> bool {
        DirectiveDescrElement::is_batch(self)
    }

    fn is_template(&self) -> bool {
        DirectiveDescrElement::is_template(self)
    }

    fn get_etomo_column(&self) -> Option<DirectiveDescrEtomoColumn> {
        DirectiveDescrElement::get_etomo_column(self)
    }

    fn get_label(&self) -> Option<String> {
        DirectiveDescrElement::get_label(self)
    }

    fn get_choice_list(&self) -> Option<DirectiveDescrChoiceList> {
        DirectiveDescrElement::get_choice_list(self)
    }
}

/// Java `toString`.
impl std::fmt::Display for DirectiveDescrElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line_array) = &self.line_array
            && line_array.len() > 0
        {
            return write!(f, "[{}]", line_array[0]);
        }
        f.write_str("[]")
    }
}
