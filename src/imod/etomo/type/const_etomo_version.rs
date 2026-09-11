//! `IMOD/Etomo/src/etomo/type/ConstEtomoVersion.java`.
//!
//! A Java interface declaring one constant and three methods.  The two `ge` overloads
//! and `lt(String)` are implemented by `etomo/type/etomo_version.rs`; the trait carries
//! them under the systematic overload-suffixed names that module uses.
#![allow(dead_code)]

use super::etomo_version::EtomoVersion;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstEtomoVersion`.
pub trait ConstEtomoVersion {
    /// Java `ge(String)`.
    fn ge_string(&self, version: Option<&str>) -> bool;

    /// Java `ge(EtomoVersion)`.
    fn ge(&self, version: Option<&EtomoVersion>) -> bool;

    /// Java `lt(String)`.
    fn lt_string(&self, version: Option<&str>) -> bool;
}
