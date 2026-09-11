//! `IMOD/Etomo/src/etomo/type/ImageFilenameStyle.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton.
//!
//! Boundaries: the `value` field is the source's `EtomoNumber`;
//! because a Rust enum variant carries no per-instance storage, `value()` builds the
//! field's value on demand and `get_value` returns it by value rather than by reference.
#![allow(dead_code)]

use super::const_etomo_number::ConstEtomoNumber;
use super::etomo_number::EtomoNumber;
use super::extension::{self, Extension};
use super::imod_output_format::ImodOutputFormat;
use crate::imod::etomo::etomo_director::DEFAULT_TO_STANDARD_IMAGE_FILE_NAMES;

/// Java `ImageFilenameStyle`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFilenameStyle {
    /// Java `OLD`: value 0, imodOutputFormat null, defaultRawImageStackExtension
    /// `Extension.ST`, index 0, propertyValue "OLD".
    Old,
    /// Java `MRC`: value 1, imodOutputFormat `ImodOutputFormat.MRC`,
    /// defaultRawImageStackExtension `Extension.MRC`, index 1, propertyValue "MRC".
    Mrc,
    /// Java `HDF`: value 2, imodOutputFormat `ImodOutputFormat.HDF`,
    /// defaultRawImageStackExtension `Extension.HDF`, index 2, propertyValue "HDF".
    Hdf,
}

impl ImageFilenameStyle {
    /// Java `TOTAL`, the value of the `INDEX` counter after the three singletons are
    /// constructed.
    pub const TOTAL: i32 = 3;

    /// Java `DEFAULT`.
    pub const DEFAULT: ImageFilenameStyle = if DEFAULT_TO_STANDARD_IMAGE_FILE_NAMES {
        ImageFilenameStyle::Mrc
    } else {
        // 2206 Change to MRC when we decide to
        ImageFilenameStyle::Old
    };

    /// Java `BACKWARDS_COMPATIBLE`.
    pub const BACKWARDS_COMPATIBLE: ImageFilenameStyle = ImageFilenameStyle::Old;

    /// Java `ENV_VAR`.
    pub const ENV_VAR: &'static str = "ETOMO_NAMING_STYLE";

    /// Java field `value`, a `private final EtomoNumber` the constructor fills in with
    /// `this.value.set(value)`.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::Old => 0,
            Self::Mrc => 1,
            Self::Hdf => 2,
        });
        value
    }

    /// Java field `index`.
    fn index(self) -> i32 {
        match self {
            Self::Old => 0,
            Self::Mrc => 1,
            Self::Hdf => 2,
        }
    }

    /// Java field `propertyValue`.  Warning: do not change propertyValue - required
    /// for backward compatibility.  Don't allow to be null.
    fn property_value(self) -> &'static str {
        match self {
            Self::Old => "OLD",
            Self::Mrc => "MRC",
            Self::Hdf => "HDF",
        }
    }

    /// Java package-private field `defaultRawImageStackExtension`.
    fn default_raw_image_stack_extension_field(self) -> &'static Extension {
        match self {
            Self::Old => &extension::CLASS.st,
            Self::Mrc => &extension::CLASS.mrc,
            Self::Hdf => &extension::CLASS.hdf,
        }
    }

    /// Java `private final ImodOutputFormat imodOutputFormat`, set by the constructor.
    /// `OLD` is constructed with null.
    fn imod_output_format(self) -> Option<ImodOutputFormat> {
        match self {
            Self::Old => None,
            Self::Mrc => Some(ImodOutputFormat::Mrc),
            Self::Hdf => Some(ImodOutputFormat::Hdf),
        }
    }

    /// Java `getInstance` (ImodOutputFormat overload).  Returns default when neither
    /// MRC's nor HDF's field is identically the argument.  The source's `==` is
    /// reference identity between typesafe-enum singletons, which the Rust enum's
    /// `PartialEq` reproduces.
    pub fn get_instance_from_imod_output_format(
        imod_output_format: Option<ImodOutputFormat>,
    ) -> ImageFilenameStyle {
        if Self::Mrc.imod_output_format() == imod_output_format {
            return Self::Mrc;
        }
        if Self::Hdf.imod_output_format() == imod_output_format {
            return Self::Hdf;
        }
        Self::DEFAULT
    }

    /// Java `equals` (ImodOutputFormat overload).
    pub fn equals_imod_output_format(self, imod_output_format: Option<ImodOutputFormat>) -> bool {
        self.imod_output_format() == imod_output_format
    }

    /// Java `load(Properties, String, String)`.
    pub fn load(
        props: &std::collections::BTreeMap<String, String>,
        prepend: Option<&str>,
        key: Option<&str>,
    ) -> Option<ImageFilenameStyle> {
        let property_key = crate::imod::etomo::util::utilities::create_property_key(prepend, key);
        let value = match property_key {
            None => None,
            Some(property_key) => props.get(&property_key).cloned(),
        };
        match value {
            // `getInstanceFromPropertyValue(null)` returns null through its own guard.
            None => None,
            Some(value) => Self::get_instance_from_property_value(&value),
        }
    }

    /// Java `store(Properties, String, String)`.
    pub fn store(
        self,
        props: &mut std::collections::BTreeMap<String, String>,
        prepend: Option<&str>,
        key: Option<&str>,
    ) {
        let property_key = crate::imod::etomo::util::utilities::create_property_key(prepend, key);
        if let Some(property_key) = property_key {
            props.insert(property_key, self.property_value().to_string());
        }
    }

    /// Java `getInstance` (String, boolean overload).  Returns default for bad value.
    pub fn get_instance(string: &str, allow_default: bool) -> Option<ImageFilenameStyle> {
        if Self::Old.equals(string) {
            return Some(Self::Old);
        }
        if Self::Mrc.equals(string) {
            return Some(Self::Mrc);
        }
        if Self::Hdf.equals(string) {
            return Some(Self::Hdf);
        }
        if allow_default {
            return Some(Self::DEFAULT);
        }
        None
    }

    /// Java `getInstance` (ConstEtomoNumber overload).  Returns null for bad value.
    pub fn get_instance_from_naming_style(
        naming_style: Option<&ConstEtomoNumber>,
    ) -> Option<ImageFilenameStyle> {
        let naming_style = match naming_style {
            None => return None,
            Some(naming_style) => naming_style,
        };
        if Self::Old
            .value()
            .equals_const_etomo_number(Some(naming_style))
        {
            return Some(Self::Old);
        }
        if Self::Mrc
            .value()
            .equals_const_etomo_number(Some(naming_style))
        {
            return Some(Self::Mrc);
        }
        if Self::Hdf
            .value()
            .equals_const_etomo_number(Some(naming_style))
        {
            return Some(Self::Hdf);
        }
        None
    }

    /// Java `isValidValue`.
    pub fn is_valid_value(naming_style: Option<&ConstEtomoNumber>) -> bool {
        let naming_style = match naming_style {
            None => return false,
            Some(naming_style) => naming_style,
        };
        if Self::Old
            .value()
            .equals_const_etomo_number(Some(naming_style))
        {
            return true;
        }
        if Self::Mrc
            .value()
            .equals_const_etomo_number(Some(naming_style))
        {
            return true;
        }
        if Self::Hdf
            .value()
            .equals_const_etomo_number(Some(naming_style))
        {
            return true;
        }
        false
    }

    /// Java `getInstanceFromIndex`.  Returns default for bad index.
    pub fn get_instance_from_index(index: i32) -> ImageFilenameStyle {
        if Self::Old.index() == index {
            return Self::Old;
        }
        if Self::Mrc.index() == index {
            return Self::Mrc;
        }
        if Self::Hdf.index() == index {
            return Self::Hdf;
        }
        Self::DEFAULT
    }

    /// Java `getInstanceFromPropertyValue`.
    pub fn get_instance_from_property_value(property_value: &str) -> Option<ImageFilenameStyle> {
        if Self::Old.property_value() == property_value {
            return Some(Self::Old);
        }
        if Self::Mrc.property_value() == property_value {
            return Some(Self::Mrc);
        }
        if Self::Hdf.property_value() == property_value {
            return Some(Self::Hdf);
        }
        None
    }

    /// Java `getIndex`.  index and TOTAL are for enumerating through all instances.
    pub fn get_index(self) -> i32 {
        self.index()
    }

    /// Java `isStandard`.
    pub fn is_standard(self) -> bool {
        // Standard styles have a standard output format.
        self.imod_output_format().is_some()
    }

    /// Java `getValue`.
    pub fn get_value(self) -> ConstEtomoNumber {
        self.value().base
    }

    /// Java `getPropertyValue`.
    pub fn get_property_value(self) -> &'static str {
        self.property_value()
    }

    /// Java `equals` (String overload).  The source's leading
    /// `if (string == null) return false;` guard has no counterpart: a Rust `&str`
    /// cannot be null.
    pub fn equals(self, string: &str) -> bool {
        if self.value().equals_string(Some(string))
            || self
                .default_raw_image_stack_extension_field()
                .equals_string(Some(string))
            || self.property_value().eq_ignore_ascii_case(string)
        {
            return true;
        }
        false
    }

    /// Java `getDefaultRawImageStackExtension`.
    pub fn get_default_raw_image_stack_extension(self) -> &'static Extension {
        self.default_raw_image_stack_extension_field()
    }
}

/// Java `toString`.  Returns `value.toString()`.
impl std::fmt::Display for ImageFilenameStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Java package-private inner class `ImageFilenameStyleException`.
#[derive(Clone, Debug)]
pub struct ImageFilenameStyleException {
    message: String,
}

impl ImageFilenameStyleException {
    /// Java `ImageFilenameStyleException(String)`.
    pub fn new(message: &str) -> ImageFilenameStyleException {
        ImageFilenameStyleException {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for ImageFilenameStyleException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ImageFilenameStyleException {}
