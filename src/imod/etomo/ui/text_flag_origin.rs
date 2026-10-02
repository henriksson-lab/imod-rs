//! `IMOD/Etomo/src/etomo/ui/TextFlagOrigin.java`.
//!
//! An interface for a field which generates a flagged state.  Replaces FlagOrigin.

use std::rc::Rc;

use super::flag_origin_listener::FlagOriginListener;

/// Java `TextFlagOrigin`.  Implementers are EDT objects (`Rc`, `&self` methods).
pub trait TextFlagOrigin {
    /// Java `equals(String)`.
    fn equals(&self, value: Option<&str>) -> bool;

    /// Java `addFlagOriginListener(FlagOriginListener)`.
    fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>);

    /// Java `isValid()`.
    fn is_valid(&self) -> bool;
}
