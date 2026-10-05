//! `IMOD/Etomo/src/etomo/type/StatusChangeListener.java`.
//!
//! The implementors are event dispatch thread objects (`Rc`, `&self` methods).  Java's
//! two `statusChanged` overloads carry the parameter-type suffix.

use super::status::StatusRef;
use super::status_change_event::StatusChangeEvent;

/// Java `public interface StatusChangeListener`.
pub trait StatusChangeListener {
    /// Java `statusChanged(Status)`.
    fn status_changed_status(&self, status: Option<StatusRef>);

    /// Java `statusChanged(StatusChangeEvent)`.
    fn status_changed_event(&self, status_change_event: Option<&dyn StatusChangeEvent>);

    /// Java `startOver()`.
    fn start_over(&self);
}
