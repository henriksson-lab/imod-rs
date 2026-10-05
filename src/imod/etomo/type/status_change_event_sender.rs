//! `IMOD/Etomo/src/etomo/type/StatusChangeEventSender.java`.
//!
//! A `Runnable` that the batchruntomo monitors post with `SwingUtilities.invokeLater`:
//! it calls each listener with the status or the event.  It is built on the monitor
//! thread and run on the event dispatch thread; the listeners are event dispatch
//! thread objects, carried as `EdtRef`s.  Java hands the sender the monitor's own
//! `Vector` (not a copy), so a listener added after the post is still called; the
//! listener list here is shared the same way (`Arc<Mutex<..>>`).

use std::sync::{Arc, Mutex};

use super::status::StatusRef;
use super::status_change_event::StatusChangeEvent;
use super::status_change_listener::StatusChangeListener;
use crate::imod::etomo::util::event_queue::EdtRef;

/// A monitor's `Vector<StatusChangeListener>` (null until the first listener).
pub type StatusChangeListeners = Arc<Mutex<Option<Vec<EdtRef<dyn StatusChangeListener>>>>>;

/// Java `public final class StatusChangeEventSender implements Runnable`.
pub struct StatusChangeEventSender {
    /// Java private final `listeners`: the monitor's vector as it was when the
    /// sender was built (null if the monitor had no listener then).
    listeners: Option<StatusChangeListeners>,
    /// Java private final `status`.
    status: Option<StatusRef>,
    /// Java private final `event`.
    event: Option<Arc<dyn StatusChangeEvent>>,
}

impl StatusChangeEventSender {
    /// Java `StatusChangeEventSender(Vector<StatusChangeListener>, Status)`.
    pub fn new_status(
        listeners: StatusChangeListeners,
        status: Option<StatusRef>,
    ) -> StatusChangeEventSender {
        StatusChangeEventSender {
            listeners: Self::reference(listeners),
            status,
            event: None,
        }
    }

    /// Java `StatusChangeEventSender(Vector<StatusChangeListener>, StatusChangeEvent)`.
    pub fn new_event(
        listeners: StatusChangeListeners,
        event: Option<Arc<dyn StatusChangeEvent>>,
    ) -> StatusChangeEventSender {
        StatusChangeEventSender {
            listeners: Self::reference(listeners),
            status: None,
            event,
        }
    }

    /// Java passes the monitor's `listeners` reference, which is null until the
    /// monitor's first `addStatusChangeListener`.
    fn reference(listeners: StatusChangeListeners) -> Option<StatusChangeListeners> {
        if listeners.lock().unwrap().is_none() {
            return None;
        }
        Some(listeners)
    }

    /// Java `run()`.  Runs on the event dispatch thread.
    pub fn run(&self) {
        let Some(listeners) = &self.listeners else {
            return;
        };
        let listeners = listeners.lock().unwrap().clone().unwrap_or_default();
        let size = listeners.len();
        if self.status.is_some() {
            for listener in listeners.iter().take(size) {
                listener.get().status_changed_status(self.status);
            }
        } else if let Some(event) = &self.event {
            for listener in listeners.iter().take(size) {
                listener.get().status_changed_event(Some(&**event));
            }
        }
    }
}
