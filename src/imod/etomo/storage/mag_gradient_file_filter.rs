//! `IMOD/Etomo/src/etomo/storage/MagGradientFileFilter.java`.
//!
//! `MagGradientFileFilter extends javax.swing.filechooser.FileFilter`.  The Swing superclass contributes
//! only the two abstract methods this class implements, so they are inherent methods
//! here.

use std::path::Path;

use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `MagGradientFileFilter`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MagGradientFileFilter {}

impl MagGradientFileFilter {
    /// Java's implicit `MagGradientFileFilter()`.
    pub fn new() -> MagGradientFileFilter {
        MagGradientFileFilter {}
    }

    /// Java `accept(File)`.
    pub fn accept(&self, f: &Path) -> bool {
        // If this is a file test its extension, all others should return true
        if f.is_file() && !java_io_file_get_absolute_path(&f.to_string_lossy()).ends_with(".mgt") {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "Mag Gradients Correction File"
    }
}

impl crate::imod::etomo::jdk::FileFilter for MagGradientFileFilter {
    fn accept(&self, file: &std::path::Path) -> bool {
        MagGradientFileFilter::accept(self, file)
    }

    fn get_description(&self) -> Option<String> {
        Some(MagGradientFileFilter::get_description(self).to_string())
    }
}
