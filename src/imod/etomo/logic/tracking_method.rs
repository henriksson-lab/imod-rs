//! `IMOD/Etomo/src/etomo/logic/TrackingMethod.java`.
//!
//! The fiducial tracking method of a dataset: its batchruntomo directive value
//! (`runtime.Fiducials.any.trackingMethod`, 0-2) and its metadata string.
//! The three Java singletons are constants here; `value` is the `EtomoNumber`
//! the source holds, built on demand.

use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java final `TrackingMethod implements EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrackingMethod {
    is_default: bool,
    value: i32,
    string: &'static str,
}

/// Java `SEED`.
pub const SEED: TrackingMethod = TrackingMethod {
    is_default: true,
    value: 0,
    string: "Seed",
};
/// Java `PATCH_TRACKING`.
pub const PATCH_TRACKING: TrackingMethod = TrackingMethod {
    is_default: false,
    value: 1,
    string: "PatchTracking",
};
/// Java `RAPTOR`.
pub const RAPTOR: TrackingMethod = TrackingMethod {
    is_default: false,
    value: 2,
    string: "Raptor",
};

/// Java `NUM`.
pub const NUM: i32 = 3;

impl TrackingMethod {
    /// Java field `value`, an `EtomoNumber` set to the directive value.
    fn value_number(&self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(self.value);
        value
    }

    /// Java static `getInstance(String)`: matches the metadata string or the
    /// directive value (`ConstEtomoNumber.equals(String)`, a numeric test).
    pub fn get_instance(string: Option<&str>) -> Option<TrackingMethod> {
        let string = string?;
        for method in [SEED, PATCH_TRACKING, RAPTOR] {
            if method.string == string || method.value_number().equals_string(Some(string)) {
                return Some(method);
            }
        }
        None
    }

    /// Java static `toMetaDataValue(String)`.
    pub fn to_meta_data_value(from_directive_value: Option<&str>) -> Option<&'static str> {
        let from_directive_value = from_directive_value?;
        for method in [SEED, PATCH_TRACKING, RAPTOR] {
            if from_directive_value == method.value_number().to_string() {
                return Some(method.string);
            }
        }
        None
    }

    /// Java static `toDirectiveValue(String)`.
    pub fn to_directive_value(from_meta_data_value: Option<&str>) -> Option<EtomoNumber> {
        let from_meta_data_value = from_meta_data_value?;
        for method in [SEED, PATCH_TRACKING, RAPTOR] {
            if from_meta_data_value == method.string {
                return Some(method.value_number());
            }
        }
        None
    }

    /// Java `isDefault()`.
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<&'static str> {
        None
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> EtomoNumber {
        self.value_number()
    }
}

/// Java `implements EnumeratedType`.
impl EnumeratedType for TrackingMethod {
    fn is_default(&self) -> bool {
        self.is_default
    }
    fn get_value(&self) -> ConstEtomoNumber {
        self.value_number().base
    }
    fn get_label(&self) -> Option<String> {
        None
    }
}

/// Java `toString()`.
impl std::fmt::Display for TrackingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_and_metadata_values_map_both_ways() {
        assert_eq!(TrackingMethod::get_instance(Some("2")), Some(RAPTOR));
        assert_eq!(TrackingMethod::get_instance(Some("Raptor")), Some(RAPTOR));
        assert_eq!(TrackingMethod::get_instance(Some("0")), Some(SEED));
        assert_eq!(TrackingMethod::get_instance(Some("raptor")), None);
        assert_eq!(TrackingMethod::get_instance(None), None);
        assert_eq!(
            TrackingMethod::to_meta_data_value(Some("1")),
            Some("PatchTracking")
        );
        assert_eq!(
            TrackingMethod::to_directive_value(Some("Raptor")).map(|v| v.to_string()),
            Some("2".to_owned())
        );
        assert!(SEED.is_default() && !RAPTOR.is_default());
    }
}
