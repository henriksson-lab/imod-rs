//! `IMOD/Etomo/src/etomo/storage/ExtensionFileFilter.java`.
//!
//! `ExtensionFileFilter extends javax.swing.filechooser.FileFilter implements
//! java.io.FileFilter, FilenameFilter`.  The Swing superclass contributes only the two
//! abstract methods this class implements (`accept(File)` and `getDescription()`), so
//! the whole unit translates; the three interfaces become inherent methods here, and
//! `accept(File, String)` carries the `FilenameFilter` name.
//!
//! The nested `SearchCollection` uses a `HashMap`/`HashSet`; `accept` iterates them and
//! returns on the first hit, so the iteration order is observable when more than one
//! element matches, but every arm of the test returns the same `true`.  The Rust
//! translation keeps the same containers with deterministic ordering (`BTreeMap` /
//! `Vec`), which cannot change the result.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use regex::Regex;
use std::collections::BTreeMap;
use std::path::Path;

/// Java `ExtensionFileFilter`.
pub struct ExtensionFileFilter {
    /// Java field `acceptDirectory`.
    accept_directory: bool,
    /// Java field `useAllFileStyles`.
    use_all_file_styles: bool,
    /// Java field `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java field `searchCollection`, initialised to null.
    search_collection: Option<SearchCollection>,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

impl ExtensionFileFilter {
    /// Java `ExtensionFileFilter(BaseManager, boolean, boolean, boolean)`.  Call `setup`
    /// to add search elements.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        accept_directory: bool,
        use_all_file_styles: bool,
        debug: bool,
    ) -> ExtensionFileFilter {
        ExtensionFileFilter {
            accept_directory,
            use_all_file_styles,
            manager,
            search_collection: None,
            debug,
        }
    }

    /// Java `getInstance`.
    pub fn get_instance(manager: Option<&'static dyn BaseManager>) -> ExtensionFileFilter {
        ExtensionFileFilter::new(manager, false, false, false)
    }

    /// Java package-private `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
        if let Some(search_collection) = self.search_collection.as_mut() {
            search_collection.set_debug(debug);
        }
    }

    /// Java `setup`.  Call before running `accept`.  If the standard extension is
    /// included in the search, use `Extension` and `FileType` to allow unwanted
    /// standardized files to be excluded even though they have the same suffix as the
    /// standard extension.
    ///
    /// * `suffixes` - will be compared with `String.endsWith`
    /// * `extensions` - `getSuffix` result will be compared with `String.endsWith`
    /// * `file_types` - `getRegexp` result will be compared against the file name with
    ///   `String.matches`
    pub fn setup(
        &mut self,
        dataset_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        suffixes: Option<&[String]>,
        extensions: Option<&[&'static Extension]>,
        file_types: Option<&[Option<&FileType>]>,
    ) {
        // reset
        if let Some(search_collection) = self.search_collection.as_mut() {
            search_collection.clear();
        }
        // Add search elements
        if suffixes.map(|s| !s.is_empty()).unwrap_or(false)
            || extensions.map(|e| !e.is_empty()).unwrap_or(false)
            || file_types.map(|f| !f.is_empty()).unwrap_or(false)
        {
            let mut search_collection = SearchCollection::new(self.manager);
            search_collection.add_suffixes(suffixes);
            search_collection.add_extensions(extensions, self.use_all_file_styles);
            search_collection.add_file_types(
                file_types,
                dataset_name,
                axis_type,
                axis_id,
                self.use_all_file_styles,
            );
            self.search_collection = Some(search_collection);
        }
    }

    /// Java package-private `add(String)`.
    pub fn add(&mut self, suffix: Option<&str>) {
        let suffix = match suffix {
            None => return,
            Some(suffix) if suffix.is_empty() => return,
            Some(suffix) => suffix,
        };
        if self.search_collection.is_none() {
            self.search_collection = Some(SearchCollection::new(self.manager));
        }
        self.search_collection.as_mut().unwrap().add(Some(suffix));
    }

    /// Java `accept(File, String)`, the `FilenameFilter` method.
    pub fn accept_in_dir(&self, dir: Option<&Path>, name: Option<&str>) -> bool {
        let name = match name {
            None => return false,
            Some(name) => name,
        };
        let file = std::path::PathBuf::from(crate::imod::etomo::util::utilities::java_io_file_new(
            &dir.map(|dir| dir.to_string_lossy().to_string())
                .unwrap_or("null".to_string()),
            name,
        ));
        if self.accept_directory && file.is_dir() {
            return true;
        }
        self.accept(Some(&file))
    }

    /// Java `accept(File)`, the `FileFilter` method.
    pub fn accept(&self, file: Option<&Path>) -> bool {
        let file = match file {
            None => return false,
            Some(file) => file,
        };
        if self.accept_directory && file.is_dir() {
            return true;
        }
        let search_collection = match &self.search_collection {
            None => return false,
            Some(search_collection) if search_collection.is_empty() => return false,
            Some(search_collection) => search_collection,
        };
        search_collection.accept(&crate::imod::etomo::util::utilities::java_io_file_get_name(
            &file.to_string_lossy(),
        ))
    }

    /// Java `getDescription`.
    pub fn get_description(&self) -> &'static str {
        ""
    }
}

/// Java private static final nested class `SearchCollection`.
struct SearchCollection {
    /// Java field `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java field `imageFilenameStyle`.
    image_filename_style: Option<ImageFilenameStyle>,
    /// Java field `fileTypePatterns`, initialised to null.  The key is the pattern's
    /// own string, so a `BTreeMap` keeps the same one-entry-per-regex behaviour.
    file_type_patterns: Option<BTreeMap<String, Regex>>,
    /// Java field `suffixes`, initialised to null.
    suffixes: Option<Vec<String>>,
    /// Java field `extensions`, initialised to null.
    extensions: Option<Vec<&'static Extension>>,
    /// Java field `debug`, initialised to false.
    debug: bool,
}

impl SearchCollection {
    /// Java private `SearchCollection(BaseManager)`.
    fn new(manager: Option<&'static dyn BaseManager>) -> SearchCollection {
        // The source dereferences `manager.getBaseMetaData()` without a null check.
        let image_filename_style = manager
            .and_then(|manager| manager.get_base_meta_data())
            .map(|meta_data| meta_data.base().get_image_filename_style());
        SearchCollection {
            manager,
            image_filename_style,
            file_type_patterns: None,
            suffixes: None,
            extensions: None,
            debug: false,
        }
    }

    /// Java private `accept(String)`.
    fn accept(&self, file_name: &str) -> bool {
        if let Some(file_type_patterns) = &self.file_type_patterns {
            if !file_type_patterns.is_empty() {
                for pattern in file_type_patterns.values() {
                    // `Matcher.matches()` anchors the whole input.
                    if pattern
                        .find(file_name)
                        .map(|found| found.start() == 0 && found.end() == file_name.len())
                        .unwrap_or(false)
                    {
                        return true;
                    }
                }
            }
        }
        if let Some(suffixes) = &self.suffixes {
            if !suffixes.is_empty() {
                for suffix in suffixes {
                    if file_name.ends_with(suffix) {
                        return true;
                    }
                }
            }
        }
        if let Some(extensions) = &self.extensions {
            let extension = Extension::get_instance_with_style(
                self.manager,
                Some(file_name),
                self.image_filename_style,
            );
            if extensions.iter().any(|stored| match extension {
                None => false,
                Some(extension) => std::ptr::eq(*stored, extension),
            }) {
                return true;
            }
        }
        false
    }

    /// Java private `add(FileType[], String, AxisType, AxisID, boolean)`.
    fn add_file_types(
        &mut self,
        file_types: Option<&[Option<&FileType>]>,
        dataset_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        use_all_file_styles: bool,
    ) {
        let file_types = match file_types {
            None => return,
            Some(file_types) if file_types.is_empty() => return,
            Some(file_types) => file_types,
        };
        // The source dereferences `imageFilenameStyle` without a null check.
        let style_index = self.image_filename_style.unwrap().get_index();
        // Either loop through all styles or just process this dataset's style.
        let mut i_style = if use_all_file_styles { 0 } else { style_index };
        let limit = if use_all_file_styles {
            ImageFilenameStyle::TOTAL
        } else {
            style_index + 1
        };
        while i_style < limit {
            let cur_image_filename_style = ImageFilenameStyle::get_instance_from_index(i_style);
            for file_type in file_types.iter() {
                let file_type = match file_type {
                    None => continue,
                    Some(file_type) => file_type,
                };
                let regex = file_type.get_regex(
                    dataset_name,
                    axis_type,
                    axis_id,
                    Some(cur_image_filename_style),
                    self.manager
                        .and_then(|manager| manager.get_base_meta_data())
                        .map(|meta_data| meta_data.base().get_raw_image_stack_extension()),
                );
                let regex = match regex {
                    None => continue,
                    Some(regex) => regex,
                };
                // `Pattern.compile(regex)` throws for a malformed pattern rather than
                // returning null, which is what the source's own null check implies.
                let pattern = match Regex::new(&regex) {
                    Err(_) => continue,
                    Ok(pattern) => pattern,
                };
                // `pattern.toString()` is the regex the pattern was compiled from.
                let key = regex.clone();
                match self.file_type_patterns.as_mut() {
                    None => {
                        let mut map = BTreeMap::new();
                        map.insert(key, pattern);
                        self.file_type_patterns = Some(map);
                    }
                    Some(map) => {
                        if map.contains_key(&key) {
                            continue;
                        }
                        map.insert(key, pattern);
                    }
                }
            }
            i_style += 1;
        }
    }

    /// Java private `add(String[])`.
    fn add_suffixes(&mut self, input_suffixes: Option<&[String]>) {
        let input_suffixes = match input_suffixes {
            None => return,
            Some(input_suffixes) if input_suffixes.is_empty() => return,
            Some(input_suffixes) => input_suffixes,
        };
        for input_suffix in input_suffixes {
            if input_suffix.is_empty() {
                continue;
            }
            match self.suffixes.as_mut() {
                None => self.suffixes = Some(vec![input_suffix.clone()]),
                Some(suffixes) => {
                    if suffixes.contains(input_suffix) {
                        continue;
                    }
                    suffixes.push(input_suffix.clone());
                }
            }
        }
    }

    /// Java private `add(String)`.
    fn add(&mut self, suffix: Option<&str>) {
        let suffix = match suffix {
            None => return,
            Some(suffix) if suffix.is_empty() => return,
            Some(suffix) => suffix,
        };
        match self.suffixes.as_mut() {
            None => self.suffixes = Some(vec![suffix.to_string()]),
            Some(suffixes) => {
                if suffixes.iter().any(|stored| stored == suffix) {
                    return;
                }
                suffixes.push(suffix.to_string());
            }
        }
    }

    /// Java private `add(Extension[], boolean)`.  Note that the source ignores its
    /// `useAllFileStyles` parameter.
    fn add_extensions(
        &mut self,
        input_extensions: Option<&[&'static Extension]>,
        use_all_file_styles: bool,
    ) {
        let _ = use_all_file_styles;
        let input_extensions = match input_extensions {
            None => return,
            Some(input_extensions) if input_extensions.is_empty() => return,
            Some(input_extensions) => input_extensions,
        };
        for input_extension in input_extensions.iter() {
            match self.extensions.as_mut() {
                None => self.extensions = Some(vec![input_extension]),
                Some(extensions) => {
                    if extensions
                        .iter()
                        .any(|stored| std::ptr::eq(*stored, *input_extension))
                    {
                        continue;
                    }
                    // The extension is new - add it.
                    extensions.push(input_extension);
                }
            }
        }
    }

    /// Java private `clear`.
    fn clear(&mut self) {
        if let Some(file_type_patterns) = self.file_type_patterns.as_mut() {
            file_type_patterns.clear();
        }
        if let Some(suffixes) = self.suffixes.as_mut() {
            suffixes.clear();
        }
        if let Some(extensions) = self.extensions.as_mut() {
            extensions.clear();
        }
    }

    /// Java private `isEmpty`.
    fn is_empty(&self) -> bool {
        if let Some(file_type_patterns) = &self.file_type_patterns {
            if !file_type_patterns.is_empty() {
                return false;
            }
        }
        if let Some(suffixes) = &self.suffixes {
            if !suffixes.is_empty() {
                return false;
            }
        }
        if let Some(extensions) = &self.extensions {
            if !extensions.is_empty() {
                return false;
            }
        }
        true
    }

    /// Java private `setDebug`.
    fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

/// Java `SearchCollection.toString`.
impl std::fmt::Display for SearchCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            match self.image_filename_style {
                None => "null".to_string(),
                Some(image_filename_style) => image_filename_style.to_string(),
            }
        )
    }
}
