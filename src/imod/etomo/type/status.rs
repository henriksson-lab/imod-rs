//! `IMOD/Etomo/src/etomo/type/Status.java`.
#![allow(dead_code)]

/// Java `Status`.  Works with StatusChangeListener.
pub trait Status {
    /// Java `getText`.
    fn get_text(&self) -> Option<&'static str>;
}
