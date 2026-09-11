//! `IMOD/Etomo/src/etomo/type/ImodOutputFormat.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton, matching the shape used by
//! `etomo::r#type::image_filename_style`.  The private `value` field is the string the
//! constructor was handed, so it is recovered from the variant rather than stored.
//!
//! Note that `TIF` and `TIFF` are two distinct singletons carrying two distinct
//! strings, and the source compares them by identity elsewhere
//! (`ImageOutputFormat.getInstance`), so they must stay separate variants.
#![allow(dead_code)]

/// Java `ENV_VAR`.
pub const ENV_VAR: &str = "IMOD_OUTPUT_FORMAT";

/// Java `ImodOutputFormat`.
// Possible environment variable values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImodOutputFormat {
    /// Java `MRC`, constructed with value "MRC".
    Mrc,
    /// Java `HDF`, constructed with value "HDF".
    Hdf,
    /// Java `TIF`, constructed with value "TIF".
    Tif,
    /// Java `TIFF`, constructed with value "TIFF".
    Tiff,
    /// Java `JPG`, constructed with value "JPG".
    Jpg,
    /// Java `JPEG`, constructed with value "JPEG".
    Jpeg,
}

impl ImodOutputFormat {
    /// Java field `value`, a `private final String` set by
    /// `ImodOutputFormat(final String value)`.
    fn value(self) -> &'static str {
        match self {
            Self::Mrc => "MRC",
            Self::Hdf => "HDF",
            Self::Tif => "TIF",
            Self::Tiff => "TIFF",
            Self::Jpg => "JPG",
            Self::Jpeg => "JPEG",
        }
    }

    /// Java `getInstance(String)`.  The source's `MRC.value.equals(value)` chain
    /// returns null when nothing matches.
    pub fn get_instance(value: &str) -> Option<ImodOutputFormat> {
        if Self::Mrc.value() == value {
            return Some(Self::Mrc);
        }
        if Self::Hdf.value() == value {
            return Some(Self::Hdf);
        }
        if Self::Tif.value() == value {
            return Some(Self::Tif);
        }
        if Self::Tiff.value() == value {
            return Some(Self::Tiff);
        }
        if Self::Jpg.value() == value {
            return Some(Self::Jpg);
        }
        if Self::Jpeg.value() == value {
            return Some(Self::Jpeg);
        }
        None
    }

    // TODO(unit): needs etomo/type/ImageOutputFormat.java - Java
    // `getInstance(ImageOutputFormat)` returns HDF when the argument is identically
    // `ImageOutputFormat.HDF` and MRC otherwise, and `ImageOutputFormat.java` has no
    // module.

    /// Java `equals(String)`.  The source's `this.value.equals(value)` is false for a
    /// null argument; a Rust `&str` cannot be null.
    pub fn equals(self, value: &str) -> bool {
        self.value() == value
    }

    /// Java `getValue`.
    pub fn get_value(self) -> &'static str {
        self.value()
    }
}

/// Java `toString`.  Returns `value`.
impl std::fmt::Display for ImodOutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value())
    }
}
