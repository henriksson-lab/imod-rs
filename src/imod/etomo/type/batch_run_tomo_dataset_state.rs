//! `IMOD/Etomo/src/etomo/type/BatchRunTomoDatasetState.java`.
//!
//! The state of a dataset in the BatchRunTomo interface.  This status is passed by
//! BatchRunTomoMonitor (in StatusChangeIndexedEvent).  BatchRunTomoRow responds to
//! it.  The Java singletons are an enum; the static `TEXT_LIST` each constructor
//! appends to is the list of the instances' texts in declaration order.

use std::rc::Rc;

use super::status::Status;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::ui::swing::ui_utilities;
use crate::imod::etomo::util::utilities::to_string_if_set;

/// Java `public final class BatchRunTomoDatasetState implements Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BatchRunTomoDatasetState {
    /// Java `DONE = new BatchRunTomoDatasetState("Done", false, false, "Done")`.
    Done,
    /// Java `FAILED = new BatchRunTomoDatasetState("Failed", false, true, "Failed")`.
    Failed,
    /// Java `FAILING = new BatchRunTomoDatasetState("Running", true, true, "Failing")`.
    Failing,
    /// Java `KILLED = new BatchRunTomoDatasetState("Killed", false, false, "Killed")`.
    Killed,
    /// Java `RUNNING = new BatchRunTomoDatasetState("Running", true, false, "Running")`.
    Running,
    /// Java `STARTING = new BatchRunTomoDatasetState("Running", true, false, "Starting")`.
    Starting,
    /// Java `STOPPED = new BatchRunTomoDatasetState("Stopped", false, false, "Stopped")`.
    Stopped,
    /// Java `A_DONE = new BatchRunTomoDatasetState("A done", false, false, "ADone")`.  A
    /// axis only.
    ADone,
}

/// The instances in declaration order (the order Java's `TEXT_LIST` is filled in).
const DECLARED: [BatchRunTomoDatasetState; 8] = [
    BatchRunTomoDatasetState::Done,
    BatchRunTomoDatasetState::Failed,
    BatchRunTomoDatasetState::Failing,
    BatchRunTomoDatasetState::Killed,
    BatchRunTomoDatasetState::Running,
    BatchRunTomoDatasetState::Starting,
    BatchRunTomoDatasetState::Stopped,
    BatchRunTomoDatasetState::ADone,
];

impl BatchRunTomoDatasetState {
    /// Java private final `text`.
    fn text(self) -> &'static str {
        match self {
            BatchRunTomoDatasetState::Done => "Done",
            BatchRunTomoDatasetState::Failed => "Failed",
            BatchRunTomoDatasetState::Failing => "Running",
            BatchRunTomoDatasetState::Killed => "Killed",
            BatchRunTomoDatasetState::Running => "Running",
            BatchRunTomoDatasetState::Starting => "Running",
            BatchRunTomoDatasetState::Stopped => "Stopped",
            BatchRunTomoDatasetState::ADone => "A done",
        }
    }

    /// Java private final `active`.
    fn active(self) -> bool {
        matches!(
            self,
            BatchRunTomoDatasetState::Failing
                | BatchRunTomoDatasetState::Running
                | BatchRunTomoDatasetState::Starting
        )
    }

    /// Java private final `error`.
    fn error(self) -> bool {
        matches!(
            self,
            BatchRunTomoDatasetState::Failed | BatchRunTomoDatasetState::Failing
        )
    }

    /// Java private final `key`.
    fn key(self) -> &'static str {
        match self {
            BatchRunTomoDatasetState::Done => "Done",
            BatchRunTomoDatasetState::Failed => "Failed",
            BatchRunTomoDatasetState::Failing => "Failing",
            BatchRunTomoDatasetState::Killed => "Killed",
            BatchRunTomoDatasetState::Running => "Running",
            BatchRunTomoDatasetState::Starting => "Starting",
            BatchRunTomoDatasetState::Stopped => "Stopped",
            BatchRunTomoDatasetState::ADone => "ADone",
        }
    }

    /// Java static `getInstance(String)`.  (`A_DONE` is not recognized by the source.)
    pub fn get_instance(key: Option<&str>) -> Option<BatchRunTomoDatasetState> {
        let key = key?;
        for state in [
            BatchRunTomoDatasetState::Done,
            BatchRunTomoDatasetState::Failed,
            BatchRunTomoDatasetState::Failing,
            BatchRunTomoDatasetState::Killed,
            BatchRunTomoDatasetState::Running,
            BatchRunTomoDatasetState::Starting,
            BatchRunTomoDatasetState::Stopped,
        ] {
            if state.key() == key {
                return Some(state);
            }
        }
        None
    }

    /// Java static `setPreferredWidth(AbstractButton)`.  The widths are computed as in
    /// Java; `setPreferredSize` is layout, which the Slint side does.
    pub fn set_preferred_width(button: Option<&Rc<JComponent>>) {
        let Some(button) = button else {
            return;
        };
        let mut size = ui_utilities::get_preferred_size(button, Some(&button.get_text()));
        for state in DECLARED {
            let width = ui_utilities::get_preferred_width_abstract_button_string(
                button,
                Some(state.text()),
            );
            if width > size.width {
                size.width = width;
            }
        }
        let _ = size;
    }

    /// Java `isActive()`.
    pub fn is_active(self) -> bool {
        self.active()
    }

    /// Java `isError()`.
    pub fn is_error(self) -> bool {
        self.error()
    }

    /// Java `getKey()`.
    pub fn get_key(self) -> &'static str {
        self.key()
    }

    /// Java `equals(String)`.
    pub fn equals(self, input: Option<&str>) -> bool {
        Some(self.text()) == input
    }
}

impl Status for BatchRunTomoDatasetState {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        Some(self.text())
    }
}

/// Java `toString()`.
impl std::fmt::Display for BatchRunTomoDatasetState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[BatchRunTomoDatasetState{},active:{}{}]",
            to_string_if_set(Some(":text:"), Some(self.text())),
            self.active(),
            to_string_if_set(Some(",key:"), Some(self.key()))
        )
    }
}
