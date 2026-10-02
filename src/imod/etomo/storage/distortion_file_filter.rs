//! `IMOD/Etomo/src/etomo/storage/DistortionFileFilter.java`.
//!
//! `DistortionFileFilter extends javax.swing.filechooser.FileFilter`.  The Swing superclass contributes
//! only the two abstract methods this class implements, so they are inherent methods
//! here.

use std::path::Path;

use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `rcsid`.
pub const RCSID: &str = "$$Id$$";

/// Java `DistortionFileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DistortionFileFilter {}

impl DistortionFileFilter {
    /// Java's implicit `DistortionFileFilter()`.
    pub fn new() -> DistortionFileFilter {
        DistortionFileFilter {}
    }

    /// Java `accept(File)`.
    pub fn accept(&self, f: &Path) -> bool {
        // If this is a file test its extension, all others should return true
        if f.is_file() && !java_io_file_get_absolute_path(&f.to_string_lossy()).ends_with(".idf") {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "Image Distortion Field File"
    }
}

impl crate::imod::etomo::jdk::FileFilter for DistortionFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        DistortionFileFilter::accept(self, file)
    }

    fn get_description(&self) -> Option<String> {
        Some(DistortionFileFilter::get_description(self).to_string())
    }
}
