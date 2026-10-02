//! `IMOD/Etomo/src/etomo/comscript/ConstSetParam.java`.
//!
//! Java `ConstSetParam` is a class, not an interface: `SetParam` extends it and reads
//! and writes its package-private fields.  It is translated as a struct that
//! `SetParam` embeds (`SetParam.base`, reached through `Deref`).

use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "set";
/// Java `COMBINEFFT_REDUCTION_FACTOR_NAME`.
pub const COMBINEFFT_REDUCTION_FACTOR_NAME: &str = "combinefft_reduce";
/// Java package-private `COMBINEFFT_REDUCTION_FACTOR_TYPE`.
pub const COMBINEFFT_REDUCTION_FACTOR_TYPE: Type = Type::Double;
/// Java `COMBINEFFT_LOW_FROM_BOTH_RADIUS_NAME`.
pub const COMBINEFFT_LOW_FROM_BOTH_RADIUS_NAME: &str = "combinefft_lowboth";
/// Java `COMBINEFFT_LOW_FROM_BOTH_RADIUS_TYPE`.
pub const COMBINEFFT_LOW_FROM_BOTH_RADIUS_TYPE: Type = Type::Double;

/// Java package-private `delimiter`.
pub const DELIMITER: &str = "=";

/// Java `ConstSetParam`.
#[derive(Clone, Debug)]
pub struct ConstSetParam {
    /// Java field `expectedName`, initialised to null.
    pub expected_name: Option<String>,
    /// Java field `name`, initialised to "".
    pub name: Option<String>,
    /// Java field `type`.
    pub r#type: Type,
    /// Java field `numericValue`.
    pub numeric_value: EtomoNumber,
    /// Java field `value`.
    pub value: Option<String>,
    /// Java field `numeric`, initialised to false.
    pub numeric: bool,
    /// Java field `valid`, initialised to true.
    pub valid: bool,
}

impl ConstSetParam {
    /// Java package-private `ConstSetParam(String, EtomoNumber.Type)`.
    pub fn new(expected_name: &str, r#type: Type) -> ConstSetParam {
        let mut param = ConstSetParam {
            expected_name: None,
            name: Some(String::new()),
            r#type,
            numeric_value: EtomoNumber::new_with_type_and_name(r#type, expected_name),
            value: None,
            numeric: false,
            valid: true,
        };
        param.numeric_value.set_display_value_int(0);
        param.reset();
        param.numeric = true;

        param.expected_name = Some(expected_name.to_string());
        param
    }

    /// Java package-private `reset`.
    pub fn reset(&mut self) {
        self.name = Some(String::new());
        self.value = Some(String::new());
        self.numeric_value.reset();
        self.valid = true;
    }

    /// Java `isValid`.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Java `getValue`.
    pub fn get_value(&self) -> Option<String> {
        if self.numeric {
            return Some(self.numeric_value.to_string());
        }
        self.value.clone()
    }
}
