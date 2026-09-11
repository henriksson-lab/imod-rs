//! `IMOD/Etomo/src/etomo/type/ConstStringProperty.java`.
//!
//! A Java interface declaring one constant and two methods, one of which is
//! `toString`.  The Rust trait carries `is_empty`; `toString` is the `Display`
//! implementation each implementor writes, which is the project's standing mapping for
//! `toString` and is what the coverage audit recognises.
#![allow(dead_code)]

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstStringProperty`.
pub trait ConstStringProperty: std::fmt::Display {
    /// Java `toString()`.  Declared by the interface; every implementor supplies it as
    /// its `std::fmt::Display` body, so the trait only requires that bound.
    ///
    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool;
}
