//! `IMOD/Etomo/src/etomo/ui/swing/RadioButtonInterface.java`.
#![allow(dead_code)]

use super::radio_button::EnumeratedTypeBoundary;

/// Java package-private `RadioButtonInterface`.
///
/// The concrete field object remains owned by each implementer, as it does in
/// Java; the trait preserves the group selection notification contract.
pub trait RadioButtonInterface {
    /// Java `msgSelected()`.
    fn msg_selected(&mut self);

    /// Java `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<&EnumeratedTypeBoundary>;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;
}
