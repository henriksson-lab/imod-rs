//! `IMOD/Etomo/src/etomo/storage/AutodocFilter.java`.
//!
//! Description: Default autodoc extension file filter
//!
//! Copyright: Copyright 2005 - 2015 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Java extends `javax.swing.filechooser.FileFilter` and implements
//! `java.io.FileFilter`; neither supertype contributes state or behaviour to the two
//! members below.
#![allow(dead_code)]

use crate::imod::etomo::storage::autodoc::autodoc_factory::extension;
use crate::imod::etomo::util::utilities;

/// Java public `AutodocFilter`.
pub struct AutodocFilter {
    /// Java private final field `excludeHidden`.
    exclude_hidden: bool,
}

impl AutodocFilter {
    /// Java `AutodocFilter()`.
    pub fn new() -> AutodocFilter {
        AutodocFilter {
            exclude_hidden: false,
        }
    }

    /// Java `AutodocFilter(boolean)`.
    pub fn new_exclude_hidden(exclude_hidden: bool) -> AutodocFilter {
        AutodocFilter { exclude_hidden }
    }

    /// Java `accept(File)`.
    pub fn accept(&self, f: &std::path::Path) -> bool {
        if !f.exists() {
            eprintln!(
                "Warning: {} does not exist",
                utilities::java_io_file_get_absolute_path(&f.to_string_lossy())
            );
            return false;
        }
        if f.is_file() {
            let name = utilities::java_io_file_get_name(&f.to_string_lossy());
            return name.ends_with(&extension::DEFAULT.to_string())
                && (!self.exclude_hidden || !name.starts_with('.'));
        }
        true
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> String {
        "Autodoc file".to_string()
    }
}

impl Default for AutodocFilter {
    fn default() -> AutodocFilter {
        AutodocFilter::new()
    }
}
