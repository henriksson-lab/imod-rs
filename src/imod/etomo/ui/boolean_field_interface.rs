//! `IMOD/Etomo/src/etomo/ui/BooleanFieldInterface.java`.
//!
//! Interface for distinguishing boolean fields.

use super::field::Field;

/// Java `BooleanFieldInterface extends Field`: a marker with no members of its own.
pub trait BooleanFieldInterface: Field {}
