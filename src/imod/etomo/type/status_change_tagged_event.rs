//! `IMOD/Etomo/src/etomo/type/StatusChangeTaggedEvent.java`.

use std::any::Any;
use std::sync::Mutex;

use super::status::StatusRef;
use super::status_change_event::StatusChangeEvent;

/// Java `public class StatusChangeTaggedEvent implements StatusChangeEvent`.
pub struct StatusChangeTaggedEvent {
    /// Java private final `tag`.
    tag: Option<String>,
    /// Java private final `string`.
    string: Option<String>,
    /// Java private `status`, initially null.
    status: Mutex<Option<StatusRef>>,
}

impl StatusChangeTaggedEvent {
    /// Java `StatusChangeTaggedEvent(String, Status)`.
    pub fn new(tag: Option<&str>, status: Option<StatusRef>) -> StatusChangeTaggedEvent {
        StatusChangeTaggedEvent {
            tag: tag.map(str::to_owned),
            string: None,
            status: Mutex::new(status),
        }
    }

    /// Java `StatusChangeTaggedEvent(String, String, Status)`.
    pub fn new_with_string(
        tag: Option<&str>,
        string: Option<&str>,
        status: Option<StatusRef>,
    ) -> StatusChangeTaggedEvent {
        StatusChangeTaggedEvent {
            tag: tag.map(str::to_owned),
            string: string.map(str::to_owned),
            status: Mutex::new(status),
        }
    }

    /// Java `equals(String)`.
    pub fn equals(&self, input: Option<&str>) -> bool {
        (self.tag.is_none() && input.is_none())
            || (self.tag.is_some() && self.tag.as_deref() == input)
    }

    /// Java `setStatus(Status)`.
    pub fn set_status(&self, status: Option<StatusRef>) {
        *self.status.lock().unwrap() = status;
    }

    /// Java `getString()`.
    pub fn get_string(&self) -> Option<&str> {
        self.string.as_deref()
    }
}

impl StatusChangeEvent for StatusChangeTaggedEvent {
    /// Java `getStatus()`.
    fn get_status(&self) -> Option<StatusRef> {
        *self.status.lock().unwrap()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Java `toString()`.
impl std::fmt::Display for StatusChangeTaggedEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[tag:{},string:{},status:{}]",
            self.tag.as_deref().unwrap_or("null"),
            self.string.as_deref().unwrap_or("null"),
            self.get_status()
                .map_or("null".to_owned(), |status| status.to_string())
        )
    }
}
