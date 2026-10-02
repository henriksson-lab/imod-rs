//! `IMOD/Etomo/src/etomo/ui/TextFieldInterface.java`.
//!
//! Interface for distinguishing text fields.

use super::field::Field;

/// Java `TextFieldInterface extends Field`: a marker with no members of its own.
pub trait TextFieldInterface: Field {}
