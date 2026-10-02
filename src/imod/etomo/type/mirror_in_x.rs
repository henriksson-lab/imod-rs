//! `IMOD/Etomo/src/etomo/type/MirrorInX.java`.
//!
//! Transferfid parameter MirrorInX.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; the singletons are also associated constants
//! under their Java names.  The Java `value` field is an `EtomoNumber` set from the
//! constructor's `int`; a Rust enum variant carries no storage, so `value()` builds
//! that `EtomoNumber` on demand and `get_value` returns it by value (as
//! `EnumeratedType::get_value` documents).
#![allow(dead_code)]

use super::const_etomo_number::ConstEtomoNumber;
use super::enumerated_type::EnumeratedType;
use super::etomo_number::EtomoNumber;

/// Java `MirrorInX`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MirrorInX {
    /// Java `ASSESS_BOTH = new MirrorInX(0)`.
    AssessBoth,
    /// Java `ALWAYS = new MirrorInX(1)`.
    Always,
    /// Java `NEVER = new MirrorInX(-1)`.
    Never,
}

impl MirrorInX {
    /// Java `ASSESS_BOTH`.
    pub const ASSESS_BOTH: MirrorInX = MirrorInX::AssessBoth;
    /// Java `ALWAYS`.
    pub const ALWAYS: MirrorInX = MirrorInX::Always;
    /// Java `NEVER`.
    pub const NEVER: MirrorInX = MirrorInX::Never;
    /// Java `DEFAULT = ASSESS_BOTH`.
    pub const DEFAULT: MirrorInX = MirrorInX::ASSESS_BOTH;

    /// Java field `value`: `new EtomoNumber()` then `value.set(int)` in the private
    /// constructor.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::AssessBoth => 0,
            Self::Always => 1,
            Self::Never => -1,
        });
        value
    }

    /// Java `getInstance(ConstEtomoNumber)`.
    pub fn get_instance(input: Option<&ConstEtomoNumber>) -> MirrorInX {
        let input = match input {
            None => return Self::DEFAULT,
            Some(input) => input,
        };
        if input.equals_const_etomo_number(Some(&Self::ASSESS_BOTH.value().base)) {
            return Self::ASSESS_BOTH;
        }
        if input.equals_const_etomo_number(Some(&Self::ALWAYS.value().base)) {
            return Self::ALWAYS;
        }
        if input.equals_const_etomo_number(Some(&Self::NEVER.value().base)) {
            return Self::NEVER;
        }
        Self::DEFAULT
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
    pub fn get_label(self) -> Option<String> {
        if self == Self::ASSESS_BOTH {
            return Some("Try with and without".to_string());
        }
        if self == Self::ALWAYS {
            return Some("Use mirroring".to_string());
        }
        if self == Self::NEVER {
            return Some("Do not use mirroring".to_string());
        }
        None
    }
}

/// Java `toString`: `value.toString()`.
impl std::fmt::Display for MirrorInX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.value(), f)
    }
}

/// Java `MirrorInX implements EnumeratedType`.
impl EnumeratedType for MirrorInX {
    fn is_default(&self) -> bool {
        MirrorInX::is_default(*self)
    }

    fn get_value(&self) -> ConstEtomoNumber {
        MirrorInX::get_value(*self)
    }

    fn get_label(&self) -> Option<String> {
        MirrorInX::get_label(*self)
    }
}
