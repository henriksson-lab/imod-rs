//! `IMOD/Etomo/src/etomo/type/StatusChangeRowEvent.java`.

use std::any::Any;
use std::sync::Mutex;

use super::axis_id::AxisID;
use super::status::StatusRef;
use super::status_change_event::StatusChangeEvent;
use crate::imod::etomo::util::utilities::{to_string_if_set, to_string_if_set_axis_id};

/// Java `public class StatusChangeRowEvent implements StatusChangeEvent`.
pub struct StatusChangeRowEvent {
    /// Java private final `stackID`.
    stack_id: Option<String>,
    /// Java private final `curAxisID`.
    cur_axis_id: Option<AxisID>,
    /// Java private final `fileString`.
    file_string: Option<String>,
    /// Java private `status`, initially null.
    status: Mutex<Option<StatusRef>>,
}

impl StatusChangeRowEvent {
    /// Java `StatusChangeRowEvent(String, AxisID, Status, String)`.
    pub fn new_with_file_string(
        stack_id: Option<&str>,
        cur_axis_id: Option<AxisID>,
        status: Option<StatusRef>,
        file_string: Option<&str>,
    ) -> StatusChangeRowEvent {
        StatusChangeRowEvent {
            stack_id: stack_id.map(str::to_owned),
            cur_axis_id,
            file_string: file_string.map(str::to_owned),
            status: Mutex::new(status),
        }
    }

    /// Java `StatusChangeRowEvent(String, AxisID, Status)`.
    pub fn new(
        stack_id: Option<&str>,
        cur_axis_id: Option<AxisID>,
        status: Option<StatusRef>,
    ) -> StatusChangeRowEvent {
        Self::new_with_file_string(stack_id, cur_axis_id, status, None)
    }

    /// Java `getStackID()`.
    pub fn get_stack_id(&self) -> Option<&str> {
        self.stack_id.as_deref()
    }

    /// Java `equalsStackID(String)`.
    pub fn equals_stack_id(&self, input_stack_id: Option<&str>) -> bool {
        (self.stack_id.is_none() && input_stack_id.is_none())
            || (self.stack_id.is_some() && self.stack_id.as_deref() == input_stack_id)
    }

    /// Java `setStatus(Status)`.
    pub fn set_status(&self, status: Option<StatusRef>) {
        *self.status.lock().unwrap() = status;
    }

    /// Java `getCurAxisID()`.
    pub fn get_cur_axis_id(&self) -> Option<AxisID> {
        self.cur_axis_id
    }

    /// Java `getFileString()`.
    pub fn get_file_string(&self) -> Option<&str> {
        self.file_string.as_deref()
    }
}

impl StatusChangeEvent for StatusChangeRowEvent {
    /// Java `getStatus()`.
    fn get_status(&self) -> Option<StatusRef> {
        *self.status.lock().unwrap()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Java `toString()`.
impl std::fmt::Display for StatusChangeRowEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[StatusChangeRowEvent{}{}{},\nstatus:{}]",
            to_string_if_set(Some(":stackID:"), self.stack_id.as_deref()),
            to_string_if_set_axis_id(Some(",curAxisID:"), self.cur_axis_id),
            to_string_if_set(Some(",fileString:"), self.file_string.as_deref()),
            self.get_status()
                .map_or("null".to_owned(), |status| status.to_string())
        )
    }
}
