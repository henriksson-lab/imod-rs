//! `IMOD/Etomo/src/etomo/type/EnumeratedType.java`.
//!
//! The Java interface of typesafe enums whose value is a number.  `toString()` is part of
//! the interface, so the trait requires `Display`.

use super::const_etomo_number::ConstEtomoNumber;

/// Java `EnumeratedType`.
pub trait EnumeratedType: std::fmt::Display {
    /// Java `isDefault`.
    fn is_default(&self) -> bool;

    /// Java `getValue`.  Java returns the instance's own `EtomoNumber`; a Rust enum
    /// variant carries no storage, so the value is built and returned by value.
    fn get_value(&self) -> ConstEtomoNumber;

    /// Java `getLabel`.
    fn get_label(&self) -> Option<String>;
}
