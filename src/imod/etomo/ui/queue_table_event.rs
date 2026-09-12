//! `IMOD/Etomo/src/etomo/ui/QueueTableEvent.java`.
#![allow(dead_code)]

/// Java final static `QueueTableEvent` instances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueTableEvent {
    AllowDisplay,
    Display,
    Displayed,
    DisableSecondaryQueue,
    EnableSecondaryQueue,
    Hidden,
    PreventDisplay,
}

impl QueueTableEvent {
    /// Java `getLabel`.
    pub fn get_label(self) -> &'static str {
        match self {
            Self::AllowDisplay => "ALLOW_DISPLAY",
            Self::Display => "DISPLAY",
            Self::Displayed => "DISPLAYED",
            Self::DisableSecondaryQueue => "DISABLE_SECONDARY_QUEUE",
            Self::EnableSecondaryQueue => "ENABLE_SECONDARY_QUEUE",
            Self::Hidden => "HIDDEN",
            Self::PreventDisplay => "PREVENT_DISPLAY",
        }
    }
}

impl std::fmt::Display for QueueTableEvent {
    /// Java `toString`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "[label:{}]", self.get_label())
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
