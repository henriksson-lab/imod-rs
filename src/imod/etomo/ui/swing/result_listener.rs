//! `IMOD/Etomo/src/etomo/ui/swing/ResultListener.java`.
#![allow(dead_code)]

use std::any::Any;

/// Java `ResultListener`.  `Object` is deliberately represented by `Any`: the
/// callback receives the originating source object, not a file-field-specific
/// substitute.
pub trait ResultListener {
    /// Java `processResult(Object resultOrigin, boolean init)`.
    fn process_result(&mut self, result_origin: &dyn Any, init: bool);
}
