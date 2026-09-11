//! `IMOD/Etomo/src/etomo/type/ValidationType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton.
#![allow(dead_code)]

use super::const_etomo_number::Type;

/// Java `ValidationType`.  Instances correspond to the startingStep and endingStep
/// parameters in batchruntomo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationType {
    /// Java `STRING`: numeric false, integer false, numericType null, descr
    /// "a string".
    String,
    /// Java `INTEGER`: numeric true, integer true, numericType
    /// `EtomoNumber.Type.INTEGER`, descr "an integer".
    Integer,
    /// Java `FLOATING_POINT`: numeric true, integer false, numericType
    /// `EtomoNumber.Type.DOUBLE`, descr "a floating point number".
    FloatingPoint,
}

impl ValidationType {
    /// Java public field `numeric`.
    pub fn numeric(self) -> bool {
        match self {
            Self::String => false,
            Self::Integer => true,
            Self::FloatingPoint => true,
        }
    }

    /// Java public field `integer`.
    pub fn integer(self) -> bool {
        match self {
            Self::String => false,
            Self::Integer => true,
            Self::FloatingPoint => false,
        }
    }

    /// Java field `descr`.
    fn descr(self) -> &'static str {
        match self {
            Self::String => "a string",
            Self::Integer => "an integer",
            Self::FloatingPoint => "a floating point number",
        }
    }

    /// Java field `numericType`.
    fn numeric_type(self) -> Option<Type> {
        match self {
            Self::String => None,
            Self::Integer => Some(Type::Integer),
            Self::FloatingPoint => Some(Type::Double),
        }
    }

    /// Java `getNumericType`.
    pub fn get_numeric_type(self) -> Option<Type> {
        self.numeric_type()
    }

    /// Java `isNumeric`.
    pub fn is_numeric(self) -> bool {
        self.numeric()
    }

    /// Java `isInteger`.
    pub fn is_integer(self) -> bool {
        self.integer()
    }
}

/// Java `toString`.
impl std::fmt::Display for ValidationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.descr())
    }
}
