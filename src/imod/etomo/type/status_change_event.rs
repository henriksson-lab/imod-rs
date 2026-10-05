//! `IMOD/Etomo/src/etomo/type/StatusChangeEvent.java`.
//!
//! The implementors (`StatusChangeBooleanEvent`, `StatusChangeRowEvent`,
//! `StatusChangeTaggedEvent`) are told apart by listeners with `instanceof`, so the
//! trait exposes `as_any` for the downcast.  Events are built on monitor threads and
//! delivered on the event dispatch thread, so they are `Send + Sync`.

use std::any::Any;

use super::status::StatusRef;

/// Java `public interface StatusChangeEvent`.
pub trait StatusChangeEvent: Any + Send + Sync {
    /// Java `getStatus()`.
    fn get_status(&self) -> Option<StatusRef>;

    /// The Java `instanceof`/cast to the concrete event class.
    fn as_any(&self) -> &dyn Any;
}
