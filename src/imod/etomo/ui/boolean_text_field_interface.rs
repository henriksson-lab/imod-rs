//! `IMOD/Etomo/src/etomo/ui/BooleanTextFieldInterface.java`.
//!
//! Interface for distinguishing boolean text fields.

use super::boolean_field_interface::BooleanFieldInterface;
use super::text_field_interface::TextFieldInterface;

/// Java `BooleanTextFieldInterface extends BooleanFieldInterface, TextFieldInterface`:
/// a marker with no members of its own.
pub trait BooleanTextFieldInterface: BooleanFieldInterface + TextFieldInterface {}
