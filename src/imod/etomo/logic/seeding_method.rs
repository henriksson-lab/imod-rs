//! `IMOD/Etomo/src/etomo/logic/SeedingMethod.java`.
//!
//! The batchruntomo `seedingMethod` directive value (0-3).
//!
//! The singleton's `EtomoNumber value` field is rebuilt from the constructor's string
//! on each access.  `implements EnumeratedType`: the three interface methods are
//! inherent here (`getValue` returns the `EtomoNumber` by value); the trait itself also
//! requires `Display`, and Java does not override `toString` (it would print
//! `Object.toString()`), so the trait is not implemented.

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java final `SeedingMethod implements EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeedingMethod {
    value: &'static str,
}

/// Java `MANUAL`.
pub const MANUAL: SeedingMethod = SeedingMethod { value: "0" };
/// Java `AUTO_FID_SEED`.
pub const AUTO_FID_SEED: SeedingMethod = SeedingMethod { value: "1" };
/// Java `TRANSFER_FID`.
pub const TRANSFER_FID: SeedingMethod = SeedingMethod { value: "2" };
/// Java `BOTH`.
pub const BOTH: SeedingMethod = SeedingMethod { value: "3" };

impl SeedingMethod {
    /// Java field `value`, an `EtomoNumber` set from the string.
    fn value_number(&self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_string(Some(self.value));
        value
    }

    /// Java static `getInstance(String)`: the value is trimmed and compared
    /// numerically (`ConstEtomoNumber.equals(String)`).
    pub fn get_instance(value: Option<&str>) -> Option<SeedingMethod> {
        // `String.trim()` strips code units <= ' '
        let value = value?.trim_matches(|c: char| c <= ' ');
        if MANUAL.value_number().equals_string(Some(value)) {
            return Some(MANUAL);
        }
        if AUTO_FID_SEED.value_number().equals_string(Some(value)) {
            return Some(AUTO_FID_SEED);
        }
        if TRANSFER_FID.value_number().equals_string(Some(value)) {
            return Some(TRANSFER_FID);
        }
        if BOTH.value_number().equals_string(Some(value)) {
            return Some(BOTH);
        }
        None
    }

    /// Java `getValue()`.
    pub fn get_value(&self) -> EtomoNumber {
        self.value_number()
    }

    /// Java `isDefault()`.
    pub fn is_default(&self) -> bool {
        false
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<&'static str> {
        None
    }

    /// Java static `toDirectiveValue(MetaData, AxisID)`.
    pub fn to_directive_value(meta_data: &MetaData, axis_id: AxisID) -> Option<String> {
        if meta_data.is_track_seed_model_manual(axis_id) {
            return Some(MANUAL.value_number().to_string());
        }
        if meta_data.is_track_seed_model_auto(axis_id) {
            return Some(AUTO_FID_SEED.value_number().to_string());
        }
        if meta_data.is_track_seed_model_transfer(axis_id) {
            return Some(TRANSFER_FID.value_number().to_string());
        }
        None
    }
}
