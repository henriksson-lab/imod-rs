//! `IMOD/Etomo/src/etomo/type/EERSuperRes.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; the singletons are also associated constants
//! under their Java names.  The Java `value` field is an `EtomoNumber` set from the
//! constructor's `int`; `value()` builds it on demand (see `mirror_in_x.rs`).

use super::const_etomo_number::ConstEtomoNumber;
use super::enumerated_type::EnumeratedType;
use super::etomo_number::EtomoNumber;

/// Java `public class EERSuperRes implements EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EERSuperRes {
    /// Java `NONE = new EERSuperRes(0, "4K", ...)`.
    None,
    /// Java `TWO_X = new EERSuperRes(1, "8K", ...)`.
    TwoX,
    /// Java `FOUR_X = new EERSuperRes(2, "16K", ...)`.
    FourX,
}

impl EERSuperRes {
    /// Java `NONE`.
    pub const NONE: EERSuperRes = EERSuperRes::None;
    /// Java `TWO_X`.
    pub const TWO_X: EERSuperRes = EERSuperRes::TwoX;
    /// Java `FOUR_X`.
    pub const FOUR_X: EERSuperRes = EERSuperRes::FourX;
    /// Java `DEFAULT = TWO_X`.
    pub const DEFAULT: EERSuperRes = EERSuperRes::TWO_X;

    /// Java field `value`: `new EtomoNumber()` then `value.set(int)`.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::None => 0,
            Self::TwoX => 1,
            Self::FourX => 2,
        });
        value
    }

    /// Java `getInstance(Integer)`.  `EtomoNumber.equals(Number)` is false for null.
    pub fn get_instance(value: Option<i32>) -> Option<EERSuperRes> {
        let value = value?;
        if Self::NONE.value().base.equals_int(value) {
            return Some(Self::NONE);
        }
        if Self::TWO_X.value().base.equals_int(value) {
            return Some(Self::TWO_X);
        }
        if Self::FOUR_X.value().base.equals_int(value) {
            return Some(Self::FOUR_X);
        }
        None
    }

    /// Java `isDefault`.
    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }

    /// Java `getValue`.
    pub fn get_value(self) -> ConstEtomoNumber {
        self.value().base
    }

    /// Java `getLabel`.
    pub fn get_label(self) -> String {
        match self {
            Self::None => "4K",
            Self::TwoX => "8K",
            Self::FourX => "16K",
        }
        .to_string()
    }

    /// Java `getTooltip`.
    pub fn get_tooltip(self) -> String {
        match self {
            Self::None => "Read in frames for processing with anti-aliased reduction to 4K by 4K.",
            Self::TwoX => "Read in frames for processing with anti-aliased reduction to 8K by 8K.",
            Self::FourX => "Read in 16K frames at full 4x super-resolution.",
        }
        .to_string()
    }
}

/// Java `toString`: `value.toString()`.
impl std::fmt::Display for EERSuperRes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.value(), f)
    }
}

impl EnumeratedType for EERSuperRes {
    fn is_default(&self) -> bool {
        EERSuperRes::is_default(*self)
    }

    fn get_value(&self) -> ConstEtomoNumber {
        EERSuperRes::get_value(*self)
    }

    fn get_label(&self) -> Option<String> {
        Some(EERSuperRes::get_label(*self))
    }
}
