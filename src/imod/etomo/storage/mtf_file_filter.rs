//! `IMOD/Etomo/src/etomo/storage/MtfFileFilter.java`.
//!
//! `MtfFileFilter extends javax.swing.filechooser.FileFilter`.  The two
//! abstract methods are inherent methods (as in the other storage file
//! filters) and are also the `jdk::FileFilter` implementation, so the filter
//! can be handed to a file chooser as `Rc<dyn FileFilter>`.

use std::path::Path;

use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java `public class MtfFileFilter extends FileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MtfFileFilter {}

impl MtfFileFilter {
    /// Java's implicit `MtfFileFilter()`.
    pub fn new() -> MtfFileFilter {
        MtfFileFilter {}
    }

    /// Java `accept(File)`.
    pub fn accept(&self, f: &Path) -> bool {
        // If this is a file test its extension, all others should return true
        if f.is_file() && !java_io_file_get_absolute_path(&f.to_string_lossy()).ends_with(".mtf") {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "MTF File"
    }
}

impl FileFilter for MtfFileFilter {
    fn accept(&self, file: &Path) -> bool {
        MtfFileFilter::accept(self, file)
    }

    fn get_description(&self) -> Option<String> {
        Some(MtfFileFilter::get_description(self).to_string())
    }
}
