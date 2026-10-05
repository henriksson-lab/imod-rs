//! `IMOD/Etomo/src/etomo/storage/ComFileFilter.java`.
//!
//! A `java.io.FileFilter` (for `File.listFiles`) accepting the `<rootName>-*.com`
//! files.  It also carries `getDescription()`, so it implements `jdk::FileFilter`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class ComFileFilter implements FileFilter`.
#[derive(Clone, Debug)]
pub struct ComFileFilter {
    /// Java private final `rootName`.
    root_name: Option<String>,
}

impl ComFileFilter {
    /// Java `ComFileFilter(String)`.
    pub fn new(root_name: Option<&str>) -> ComFileFilter {
        ComFileFilter {
            root_name: root_name.map(str::to_owned),
        }
    }
}

impl FileFilter for ComFileFilter {
    /// Java `accept(File)`.
    fn accept(&self, file: &Path) -> bool {
        if file.is_dir() {
            return false;
        }
        let name = utilities::java_io_file_get_name(&file.to_string_lossy());
        if !name.starts_with(&format!(
            "{}-",
            self.root_name.as_deref().unwrap_or("null")
        )) || !name.ends_with(".com")
        {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String> {
        Some("Comscripts (.com)".to_owned())
    }
}
