//! `IMOD/Etomo/src/etomo/ui/QueueTableListener.java`.
#![allow(dead_code)]

use super::queue_table_event::QueueTableEvent;

/// Java `QueueTableListener`.
pub trait QueueTableListener {
    /// Java `queueTableEventAction(QueueTableEvent)`.
    fn queue_table_event_action(&mut self, event: QueueTableEvent);
}
