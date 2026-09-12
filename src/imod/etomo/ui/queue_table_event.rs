//! `IMOD/Etomo/src/etomo/ui/QueueTableEvent.java`.
#![allow(dead_code)]

use crate::imod::etomo::ui::swing::processor_table_row::{QueueMode, QueueType};

/// Java `QueueTableEvent` and its `QueueTableDataEvent` subclass.
///
/// Java's data event inherits the base event, so one listener can receive both.
/// Rust models that sealed inheritance relationship as data-bearing enum cases;
/// the names and data remain the direct source event protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueueTableEvent {
    AllowDisplay,
    Display,
    Displayed,
    DisableSecondaryQueue,
    EnableSecondaryQueue,
    Hidden,
    PreventDisplay,
    QueueSelected {
        queue_mode: QueueMode,
        maximum: Option<String>,
    },
    NumberJobsChanged(String),
    OnlyQueueType(QueueType),
}

impl QueueTableEvent {
    /// Java `getLabel`.
    pub fn get_label(&self) -> &'static str {
        match self {
            Self::AllowDisplay => "ALLOW_DISPLAY",
            Self::Display => "DISPLAY",
            Self::Displayed => "DISPLAYED",
            Self::DisableSecondaryQueue => "DISABLE_SECONDARY_QUEUE",
            Self::EnableSecondaryQueue => "ENABLE_SECONDARY_QUEUE",
            Self::Hidden => "HIDDEN",
            Self::PreventDisplay => "PREVENT_DISPLAY",
            Self::QueueSelected { .. } => "QUEUE_SELECTED",
            Self::NumberJobsChanged(_) => "NUMBER_JOBS_CHANGED",
            Self::OnlyQueueType(_) => "ONLY_QUEUE_TYPE",
        }
    }
}

impl std::fmt::Display for QueueTableEvent {
    /// Java `toString`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueSelected {
                queue_mode,
                maximum,
            } => write!(
                formatter,
                "[type:[label:QUEUE_SELECTED],queueMode:{queue_mode:?},jobs:null,maximum:{}]",
                maximum.as_deref().unwrap_or("null")
            ),
            Self::NumberJobsChanged(jobs) => write!(
                formatter,
                "[type:[label:NUMBER_JOBS_CHANGED],queueMode:null,jobs:{jobs},maximum:null]"
            ),
            Self::OnlyQueueType(_) => write!(
                formatter,
                "[type:[label:ONLY_QUEUE_TYPE],queueMode:null,jobs:null,maximum:null]"
            ),
            _ => write!(formatter, "[label:{}]", self.get_label()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QueueTableEvent;

    #[test]
    fn source_labels_and_to_string_are_exact() {
        assert_eq!(QueueTableEvent::AllowDisplay.get_label(), "ALLOW_DISPLAY");
        assert_eq!(
            QueueTableEvent::EnableSecondaryQueue.to_string(),
            "[label:ENABLE_SECONDARY_QUEUE]"
        );
    }
}
