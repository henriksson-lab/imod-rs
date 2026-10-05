//! `IMOD/Etomo/src/etomo/type/StatusChangeBooleanEvent.java`.

use std::any::Any;

use super::status::StatusRef;
use super::status_change_event::StatusChangeEvent;

/// Java `public class StatusChangeBooleanEvent implements StatusChangeEvent`.
pub struct StatusChangeBooleanEvent {
    /// Java private final `bool`.
    bool_: bool,
    /// Java private final `status`.
    status: Option<StatusRef>,
}

impl StatusChangeBooleanEvent {
    /// Java `StatusChangeBooleanEvent(boolean, Status)`.
    pub fn new(bool_: bool, status: Option<StatusRef>) -> StatusChangeBooleanEvent {
        StatusChangeBooleanEvent { bool_, status }
    }

    /// Java `is()`.
    pub fn is(&self) -> bool {
        self.bool_
    }
}

impl StatusChangeEvent for StatusChangeBooleanEvent {
    /// Java `getStatus()`.
    fn get_status(&self) -> Option<StatusRef> {
        self.status
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Java `toString()` (the source leaves the bracket open).
impl std::fmt::Display for StatusChangeBooleanEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[StatusChangeBooleanEvent:bool:{},\nstatus:{}",
            self.bool_,
            self.status
                .map_or("null".to_owned(), |status| status.to_string())
        )
    }
}
