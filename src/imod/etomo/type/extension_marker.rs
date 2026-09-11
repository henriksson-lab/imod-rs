//! `IMOD/Etomo/src/etomo/type/ExtensionMarker.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `static final` singleton.
#![allow(dead_code)]

use super::image_filename_style::ImageFilenameStyle;

/// Java `ExtensionMarker`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtensionMarker {
    /// Java `IMAGE`.  Standard IMAGE extensions: mrc, hdf.  Old to new filename
    /// conversion: filename.oldext => filename_oldext.stdext.  Stdext is from
    /// FilenameSettings.Settings.imageFilenameStyle when it is not 0 (old style).
    Image,
    /// Java `INPUT_IMAGE`.  Standard INPUT_IMAGE extensions: mrc, hdf, st, tif, tiff.
    /// Old to new filename conversion: filename.oldext => filename.stdext.  Stdext is
    /// from FilenameSettings.Settings.rawStackExtension.
    InputImage,
    /// Java `GENERIC`.  GENERIC allows all extensions.
    Generic,
}

impl ExtensionMarker {
    /// Java `usesStandardExtension`.
    pub fn uses_standard_extension(self, image_filename_style: Option<ImageFilenameStyle>) -> bool {
        if image_filename_style.is_none() || image_filename_style == Some(ImageFilenameStyle::Old) {
            return false;
        }
        self != Self::Generic
    }

    /// Java `isCompatible`.  Look for the first `Extension` following this extension
    /// marker in the pattern.  See if it is compatible with this extension marker.
    pub fn is_compatible(
        self,
        marker_index: i32,
        pattern: Option<&[super::file_type::PatternElement]>,
    ) -> bool {
        if self == Self::Generic {
            return true;
        }
        // If this isn't a pattern containing this Extension Marker and an Extension then
        // consider it non-compatible.
        let pattern = match pattern {
            None => return false,
            Some(pattern) => pattern,
        };
        if marker_index < 0
            || marker_index >= pattern.len() as i32
            || !matches!(
                pattern[marker_index as usize],
                super::file_type::PatternElement::ExtensionMarker(marker) if marker == self
            )
            || marker_index + 1 >= pattern.len() as i32
        {
            return false;
        }
        for i in (marker_index + 1) as usize..pattern.len() {
            if let super::file_type::PatternElement::Extension(extension) = &pattern[i] {
                return extension.is_compatible(Some(self));
            }
        }
        // No Extension was found. Consider this pattern to be non-compatible.
        false
    }
}

/// Java `toString`.
impl std::fmt::Display for ExtensionMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::Image {
            return f.write_str("[IMAGE]");
        }
        if *self == Self::InputImage {
            return f.write_str("[INPUT_IMAGE]");
        }
        if *self == Self::Generic {
            return f.write_str("[GENERIC]");
        }
        // Java falls back to Object.toString(); unreachable for the three singletons.
        f.write_str("")
    }
}
