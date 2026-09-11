//! `IMOD/Etomo/src/etomo/type/NumberedFileType.java`.
//!
//! A file type where the last two elements of the left side is an underscore followed by
//! an integer with a fixed number of digits.
//!
//! `NumberedFileType extends FileType`; the superclass is held as the `file_type` field
//! and reached through `Deref`, the same shape `FileType` itself uses for `FileKey`.
#![allow(dead_code)]

use std::sync::Arc;

use super::extension::Extension;
use super::extension_marker::ExtensionMarker;
use super::file_type::{FileType, PatternElement, Variable};
use super::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::util::utilities;
use regex::Regex;
use std::sync::LazyLock;

/// The `SEPARATOR` literal as `String.split` takes it: `"%"` is not a regex
/// metacharacter, so it matches itself.
static SEPARATOR_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(SEPARATOR).unwrap());

/// `Extension.STANDARDIZATION_DIVIDER` as `String.split` takes it.
static STANDARDIZATION_DIVIDER_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(crate::imod::etomo::r#type::extension::STANDARDIZATION_DIVIDER).unwrap()
});

/// Java `SEPARATOR`.
const SEPARATOR: &str = "%";

/// Java `NumberedFileType`.
#[derive(Debug)]
pub struct NumberedFileType {
    /// The `FileType` superclass state.
    file_type: Arc<FileType>,
    /// Java field `maxInteger`.
    max_integer: Option<i32>,
}

impl std::ops::Deref for NumberedFileType {
    type Target = FileType;
    fn deref(&self) -> &FileType {
        &self.file_type
    }
}

impl NumberedFileType {
    /// Java `NumberedFileType(Object[], Integer)`.
    pub fn new(
        file_name_pattern: Option<Vec<PatternElement>>,
        max_integer: Option<i32>,
    ) -> NumberedFileType {
        NumberedFileType {
            file_type: FileType::new_with_pattern_for_subclass(
                None,
                file_name_pattern,
                None,
                None,
                None,
                None,
            ),
            max_integer,
        }
    }

    /// Java `contructInstance` [sic].  Use the file name to make a pattern, adding an
    /// integer to the end of the left side.
    ///
    /// * `file_name` - required
    /// * `extension_marker` - default is GENERIC
    /// * `num_digits` - valid numbers: 1 - 4.  Anything else is treated as a 1.
    pub fn contruct_instance(
        file_name: Option<&str>,
        extension_marker: Option<ExtensionMarker>,
        num_digits: i32,
    ) -> Option<NumberedFileType> {
        if utilities::is_empty(file_name) {
            return None;
        }
        let extension_marker = match extension_marker {
            None => ExtensionMarker::Generic,
            Some(extension_marker) => extension_marker,
        };
        // Get an extension - either real, or built based on the extension of this file
        let extension = Extension::get_literal_instance(file_name, Some(extension_marker), false);
        let left_side = utilities::get_stripped_file_name(file_name);
        let mut integer_variable = Variable::select_integer_instance(num_digits);
        if integer_variable.is_none() {
            integer_variable = Some(Variable::one_digit_integer());
        }
        let integer_variable = integer_variable.unwrap();
        let max_integer = integer_variable.get_max_integer();
        if let Some(extension) = extension {
            // The pattern holds the `Extension` instance itself; the pattern element type
            // borrows the stored singletons, and `getLiteralInstance` can also return an
            // instance it built for an unrecognised extension, so the built one is leaked
            // to give it the 'static lifetime the pattern needs.  Java simply holds the
            // reference; the object lives as long as the pattern does either way.
            let extension: &'static Extension = Box::leak(Box::new(extension));
            return Some(NumberedFileType::new(
                Some(vec![
                    match &left_side {
                        None => PatternElement::Null,
                        Some(left_side) => PatternElement::Str(left_side.clone()),
                    },
                    PatternElement::Str(SEPARATOR.to_string()),
                    PatternElement::Variable(integer_variable),
                    PatternElement::ExtensionMarker(extension_marker),
                    PatternElement::Extension(extension),
                ]),
                max_integer,
            ));
        }
        Some(NumberedFileType::new(
            Some(vec![
                match &left_side {
                    None => PatternElement::Null,
                    Some(left_side) => PatternElement::Str(left_side.clone()),
                },
                PatternElement::Str(SEPARATOR.to_string()),
                PatternElement::Variable(integer_variable),
            ]),
            max_integer,
        ))
    }

    /// Java `getFile(BaseManager, String, Number)`.  The `numeric1` parameter is the
    /// already-formatted number; see the note on `Variable::to_formatted_string` in
    /// `etomo/type/file_type.rs` for why the `Number` transfer is left out.
    pub fn get_file(
        &self,
        manager: Option<&'static dyn BaseManager>,
        property_use_dir: Option<&str>,
        numeric1: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        self.file_type.get_file_full(
            manager,
            None,
            None,
            None,
            None,
            None,
            property_use_dir,
            None,
            None,
            numeric1,
            None,
            None,
        )
    }

    /// Java `getFile(ImageFilenameStyle, String, Number)`.
    pub fn get_file_with_image_filename_style(
        &self,
        override_image_filename_style: Option<ImageFilenameStyle>,
        property_use_dir: Option<&str>,
        numeric1: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        self.file_type.get_file_full(
            None,
            None,
            override_image_filename_style,
            None,
            None,
            None,
            property_use_dir,
            None,
            None,
            numeric1,
            None,
            None,
        )
    }

    /// Java `searchDirectory`.
    pub fn search_directory(
        &self,
        manager: Option<&'static dyn BaseManager>,
        dir: Option<&std::path::Path>,
    ) -> SearchResults {
        let mut results = SearchResults::new();
        results.search(self, manager, dir);
        results
    }
}

/// Java's nested `public class SearchResults`.  It is an inner class, so it reads
/// `NumberedFileType.this`; the Rust translation passes that as `search`'s first
/// parameter and as `get_next_unused_number`'s, which is where the source reads
/// `maxInteger`.
pub struct SearchResults {
    /// Java field `numberSet`, initialised to null.  A `TreeSet<Integer>`, so it is
    /// sorted and holds no duplicates.
    number_set: Option<std::collections::BTreeSet<i32>>,
    /// Java field `fileMap`, initialised to null.  A `TreeMap<Long, File>` keyed by the
    /// file's modification time.
    file_map: Option<std::collections::BTreeMap<i64, std::path::PathBuf>>,
}

impl SearchResults {
    /// Java private `SearchResults()`.
    fn new() -> SearchResults {
        SearchResults {
            number_set: None,
            file_map: None,
        }
    }

    /// Java private `search`.  Get a list of all files that match this
    /// `NumberedFileType` instance and store them.
    fn search(
        &mut self,
        numbered_file_type: &NumberedFileType,
        manager: Option<&'static dyn BaseManager>,
        dir: Option<&std::path::Path>,
    ) {
        self.number_set = None;
        self.file_map = None;
        let dir = match dir {
            None => return,
            Some(dir) => dir,
        };
        // Get file list.
        let mut file_filter = ExtensionFileFilter::get_instance(manager);
        file_filter.setup(
            None,
            None,
            None,
            None,
            None,
            Some(&[Some(&numbered_file_type.file_type)]),
        );
        // `dir.listFiles((FilenameFilter) fileFilter)`, which is null for a path that is
        // not a directory or cannot be read.
        let file_list = match std::fs::read_dir(dir) {
            Err(_) => return,
            Ok(entries) => {
                let mut file_list: Vec<std::path::PathBuf> = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if file_filter.accept_in_dir(Some(dir), Some(&name)) {
                        file_list.push(entry.path());
                    }
                }
                file_list
            }
        };
        if file_list.is_empty() {
            return;
        }
        // Store file, modified time, and number.
        for file in &file_list {
            if self.file_map.is_none() {
                self.file_map = Some(std::collections::BTreeMap::new());
            }
            self.file_map.as_mut().unwrap().insert(
                utilities::java_io_file_last_modified(&file.to_string_lossy()),
                file.clone(),
            );
            let file_name = utilities::java_io_file_get_name(&file.to_string_lossy());
            let left_side = utilities::get_stripped_file_name(Some(&file_name));
            if !utilities::is_empty(left_side.as_deref()) {
                // Use the number that was added to the end of the file name as the key.
                let array = utilities::java_lang_string_split(
                    left_side.as_deref().unwrap(),
                    &SEPARATOR_PATTERN,
                );
                if !array.is_empty() {
                    let suffix = array[array.len() - 1].clone();
                    let array = utilities::java_lang_string_split(
                        &suffix,
                        &STANDARDIZATION_DIVIDER_PATTERN,
                    );
                    if !array.is_empty() {
                        let number = converter::to_integer(Some(&array[0]));
                        if let Some(number) = number {
                            if self.number_set.is_none() {
                                self.number_set = Some(std::collections::BTreeSet::new());
                            }
                            self.number_set.as_mut().unwrap().insert(number);
                        }
                    }
                }
            }
        }
    }

    /// Java `getNextUnusedNumber`.
    pub fn get_next_unused_number(&self, numbered_file_type: &NumberedFileType) -> Option<i32> {
        if let Some(number_set) = &self.number_set {
            // `numberSet.last()` throws NoSuchElementException on an empty set, which
            // `search` never stores.
            let highest = *number_set.iter().next_back().unwrap();
            // The source dereferences `maxInteger` without a null check.
            if highest < numbered_file_type.max_integer.unwrap() {
                return Some(highest + 1);
            } else {
                return None;
            }
        }
        // No files found.
        Some(0)
    }

    /// Java `getOldestFile`.
    pub fn get_oldest_file(&self) -> Option<std::path::PathBuf> {
        if let Some(file_map) = &self.file_map {
            // `fileMap.get(fileMap.firstKey())`; `firstKey` throws on an empty map,
            // which `search` never stores.
            return file_map.values().next().cloned();
        }
        None
    }
}
