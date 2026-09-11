//! `IMOD/Etomo/src/etomo/type/ImageOutputFormat.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton, matching the shape used by
//! `etomo::r#type::image_filename_style` and `etomo::r#type::imod_output_format`.  The
//! private `propertyValue` field is the string the constructor was handed, so it is
//! recovered from the variant rather than stored.
//!
//! Deviation, as in `etomo/type/view_type.rs`: the Java class implements
//! `EnumeratedType`, which is not translated, so its four methods are inherent methods
//! here rather than a trait implementation.
#![allow(dead_code)]

use super::const_etomo_number::ConstEtomoNumber;
use super::image_filename_style::ImageFilenameStyle;
use super::imod_output_format::ImodOutputFormat;
use crate::imod::etomo::util::utilities::create_property_key;
use std::collections::BTreeMap;

/// Java `ImageOutputFormat`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageOutputFormat {
    /// Java `MRC`, constructed with propertyValue "MRC".
    Mrc,
    /// Java `HDF`, constructed with propertyValue "HDF".
    Hdf,
    /// Java `TIFF`, constructed with propertyValue "TIFF".
    Tiff,
}

impl ImageOutputFormat {
    /// Java `DEFAULT`.
    pub const DEFAULT: ImageOutputFormat = ImageOutputFormat::Mrc;

    /// Java field `propertyValue`.  Warning: do not change propertyValue - required
    /// for backward compatibility.
    fn property_value(self) -> &'static str {
        match self {
            Self::Mrc => "MRC",
            Self::Hdf => "HDF",
            Self::Tiff => "TIFF",
        }
    }

    /// Java `getInstance(ImodOutputFormat)`.  Returns the instance based on
    /// `ImodOutputFormat`.
    pub fn get_instance_from_imod_output_format(
        imod_output_format: Option<ImodOutputFormat>,
    ) -> ImageOutputFormat {
        if imod_output_format == Some(ImodOutputFormat::Mrc) {
            return Self::Mrc;
        }
        if imod_output_format == Some(ImodOutputFormat::Hdf) {
            return Self::Hdf;
        }
        if imod_output_format == Some(ImodOutputFormat::Tiff)
            || imod_output_format == Some(ImodOutputFormat::Tif)
        {
            return Self::Tiff;
        }
        Self::DEFAULT
    }

    /// Java `getInstance(String)`.  Returns the instance based on propertyValue.
    /// Returns default value when no match.
    pub fn get_instance(property_value: Option<&str>) -> ImageOutputFormat {
        if Some(Self::Mrc.property_value()) == property_value {
            return Self::Mrc;
        }
        if Some(Self::Hdf.property_value()) == property_value {
            return Self::Hdf;
        }
        if Some(Self::Tiff.property_value()) == property_value {
            return Self::Tiff;
        }
        Self::DEFAULT
    }

    /// Java `getInstance(ImageFilenameStyle)`.  Returns the instance based on
    /// `imageFilenameStyle`.
    pub fn get_instance_from_image_filename_style(
        image_filename_style: Option<ImageFilenameStyle>,
    ) -> ImageOutputFormat {
        if image_filename_style == Some(ImageFilenameStyle::Mrc) {
            return Self::Mrc;
        }
        if image_filename_style == Some(ImageFilenameStyle::Hdf) {
            return Self::Hdf;
        }
        Self::DEFAULT
    }

    /// Java `getInstanceFromPropertyValue`.
    pub fn get_instance_from_property_value(
        property_value: Option<&str>,
    ) -> Option<ImageOutputFormat> {
        if Some(Self::Mrc.property_value()) == property_value {
            return Some(Self::Mrc);
        }
        if Some(Self::Hdf.property_value()) == property_value {
            return Some(Self::Hdf);
        }
        if Some(Self::Tiff.property_value()) == property_value {
            return Some(Self::Tiff);
        }
        None
    }

    /// Java `isDefault`.
    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }

    /// Java `getValue`.  Returns null.
    pub fn get_value(self) -> Option<ConstEtomoNumber> {
        None
    }

    /// Java `getLabel`.
    pub fn get_label(self) -> &'static str {
        self.property_value()
    }

    /// Java `load(Properties, String, String)`.
    pub fn load(
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
        key: Option<&str>,
    ) -> Option<ImageOutputFormat> {
        let property_key = create_property_key(prepend, key);
        let value = match property_key {
            None => None,
            Some(property_key) => props.get(&property_key).cloned(),
        };
        Self::get_instance_from_property_value(value.as_deref())
    }

    /// Java `store(Properties, String, String)`.
    pub fn store(
        self,
        props: &mut BTreeMap<String, String>,
        prepend: Option<&str>,
        key: Option<&str>,
    ) {
        let property_key = create_property_key(prepend, key);
        if let Some(property_key) = property_key {
            props.insert(property_key, self.property_value().to_string());
        }
    }
}

/// Java `toString`.  Returns the string version of the value.
impl std::fmt::Display for ImageOutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.property_value())
    }
}
