//! `IMOD/Etomo/src/etomo/type/ConstStringParameter.java`.
//!
//! Copyright: Copyright 2008
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java interface `ConstStringParameter`.  Its `toString` is the `Display` supertrait.
pub trait ConstStringParameter: std::fmt::Display {
    /// Java `equals(String)`.
    fn equals(&self, input: Option<&str>) -> bool;

    /// Java `endsWith(String)`.
    fn ends_with(&self, input: Option<&str>) -> bool;

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool;
}
