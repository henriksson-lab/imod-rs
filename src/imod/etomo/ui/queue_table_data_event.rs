//! `IMOD/Etomo/src/etomo/ui/QueueTableDataEvent.java`.
#![allow(dead_code)]

use super::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::swing::processor_table_row::{QueueMode, QueueType};

/// Java final `QueueTableDataEvent` source unit.
///
/// Its Java superclass is represented by the canonical `QueueTableEvent` enum,
/// whose data-bearing cases retain this class's inherited listener identity.
pub struct QueueTableDataEvent;

impl QueueTableDataEvent {
    /// Java `getQueueSelectedInstance`.
    pub fn get_queue_selected_instance(
        queue_mode: QueueMode,
        maximum: Option<String>,
    ) -> QueueTableEvent {
        QueueTableEvent::QueueSelected {
            queue_mode,
            maximum,
        }
    }

    /// Java `getNumberJobsChangedInstance`.
    pub fn get_number_jobs_changed_instance(jobs: impl Into<String>) -> QueueTableEvent {
        QueueTableEvent::NumberJobsChanged(jobs.into())
    }

    /// Java `getOnlyQueueTypeInstance`.
    pub fn get_only_queue_type_instance(queue_type: QueueType) -> QueueTableEvent {
        QueueTableEvent::OnlyQueueType(queue_type)
    }

    #[allow(non_snake_case)]
    pub fn toString(event: &QueueTableEvent) -> String {
        format!("{event:?}")
    }
    #[allow(non_snake_case)]
    pub fn getType(event: &QueueTableEvent) -> &'static str {
        match event {
            QueueTableEvent::QueueSelected { .. } => "QUEUE_SELECTED",
            QueueTableEvent::NumberJobsChanged(_) => "NUMBER_JOBS_CHANGED",
            QueueTableEvent::OnlyQueueType(_) => "ONLY_QUEUE_TYPE",
            _ => "OTHER",
        }
    }
    #[allow(non_snake_case)]
    pub fn getQueueMode(event: &QueueTableEvent) -> Option<QueueMode> {
        match event {
            QueueTableEvent::QueueSelected { queue_mode, .. } => Some(*queue_mode),
            _ => None,
        }
    }
    #[allow(non_snake_case)]
    pub fn getQueueType(event: &QueueTableEvent) -> Option<QueueType> {
        match event {
            QueueTableEvent::OnlyQueueType(queue_type) => Some(*queue_type),
            _ => None,
        }
    }
    #[allow(non_snake_case)]
    pub fn getMaximum(event: &QueueTableEvent) -> Option<&str> {
        match event {
            QueueTableEvent::QueueSelected { maximum, .. } => maximum.as_deref(),
            _ => None,
        }
    }
    #[allow(non_snake_case)]
    pub fn getJobs(event: &QueueTableEvent) -> Option<&str> {
        match event {
            QueueTableEvent::NumberJobsChanged(jobs) => Some(jobs),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_factories_preserve_data_event_kinds() {
        assert_eq!(
            QueueTableDataEvent::get_queue_selected_instance(QueueMode::Node, Some("12".into())),
            QueueTableEvent::QueueSelected {
                queue_mode: QueueMode::Node,
                maximum: Some("12".into()),
            }
        );
        assert_eq!(
            QueueTableDataEvent::get_number_jobs_changed_instance("4"),
            QueueTableEvent::NumberJobsChanged("4".into())
        );
    }
}
