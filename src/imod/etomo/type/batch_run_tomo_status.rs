//! `IMOD/Etomo/src/etomo/type/BatchRunTomoStatus.java`.
//!
//! The states of the BatchRunTomo interface.  This status is passed by
//! BatchRunTomoMonitor and BatchRunTomoDialog.  BatchRunTomoDialog,
//! BatchRunTomoStepPanel, BatchRunTomoTable, BatchRunTomoTable.RowList, and
//! BatchRunTomoRow respond to it.  The Java singletons are an enum.

use super::status::Status;
use crate::imod::etomo::util::utilities::to_string_if_set;

/// Java `public final class BatchRunTomoStatus implements Status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BatchRunTomoStatus {
    /// Java `OPEN = new BatchRunTomoStatus("Open", false, false)`.
    Open,
    /// Java `RUNNING = new BatchRunTomoStatus("Running", false, false)`.
    Running,
    /// Java `PAUSING = new BatchRunTomoStatus("Pausing", false, false)`.
    Pausing,
    /// Java `KILLING = new BatchRunTomoStatus("Killing", false, false)`.
    Killing,
    /// Java `DONE = new BatchRunTomoStatus("Done", true, false)`.
    Done,
    /// Java `KILLED_OR_PAUSED = new BatchRunTomoStatus("Killed/Paused", true, false)`.
    KilledOrPaused,
    /// Java `KILLED_OR_PAUSED_PROCESS_CHUNKS = new
    /// BatchRunTomoStatus("Killed/Paused-Processchunks", true, false)`.
    KilledOrPausedProcessChunks,
    /// Java `KILLED_OR_PAUSED_SERIES_WATCHER = new
    /// BatchRunTomoStatus("Killed/Paused-SeriesWatcher", true, false)`.
    KilledOrPausedSeriesWatcher,
    /// Java `STOPPED = new BatchRunTomoStatus("Stopped", true, false)`.
    Stopped,
    /// Java `FAILED = new BatchRunTomoStatus("Failed", true, true)`.
    Failed,
}

/// Java `DEFAULT = OPEN`.
pub const DEFAULT: BatchRunTomoStatus = BatchRunTomoStatus::Open;

impl BatchRunTomoStatus {
    /// Java private final `text`.
    fn text(self) -> &'static str {
        match self {
            BatchRunTomoStatus::Open => "Open",
            BatchRunTomoStatus::Running => "Running",
            BatchRunTomoStatus::Pausing => "Pausing",
            BatchRunTomoStatus::Killing => "Killing",
            BatchRunTomoStatus::Done => "Done",
            BatchRunTomoStatus::KilledOrPaused => "Killed/Paused",
            BatchRunTomoStatus::KilledOrPausedProcessChunks => "Killed/Paused-Processchunks",
            BatchRunTomoStatus::KilledOrPausedSeriesWatcher => "Killed/Paused-SeriesWatcher",
            BatchRunTomoStatus::Stopped => "Stopped",
            BatchRunTomoStatus::Failed => "Failed",
        }
    }

    /// Java private final `endStatus`.
    fn end_status(self) -> bool {
        !matches!(
            self,
            BatchRunTomoStatus::Open
                | BatchRunTomoStatus::Running
                | BatchRunTomoStatus::Pausing
                | BatchRunTomoStatus::Killing
        )
    }

    /// Java private final `errorStatus`.
    fn error_status(self) -> bool {
        self == BatchRunTomoStatus::Failed
    }

    /// Java static `getInstance(BatchRunTomoStatus, BatchRunTomoStatus)`.  Returns the
    /// new status but prevents an end status from being overridden by an error status.
    /// This is necessary because batchruntomo treats a kill or pause as a failure.  But a
    /// failure status comes at the end of a run, so it's an error status and an end
    /// status.
    pub fn get_instance_from_statuses(
        cur_status: Option<BatchRunTomoStatus>,
        new_status: Option<BatchRunTomoStatus>,
    ) -> Option<BatchRunTomoStatus> {
        // If both statuses aren't end status, then there is no precidence issue. Allow
        // the new status to come into play.
        let (Some(cur_status), Some(new_status_value)) = (cur_status, new_status) else {
            return new_status;
        };
        if !cur_status.end_status() || !new_status_value.end_status() {
            return new_status;
        }
        // If there's an end status that's not a failure, use that instead of the failure
        // status.
        if new_status_value.error_status() {
            return Some(cur_status);
        }
        new_status
    }

    /// Java static `getKilledOrPausedInstance(boolean)`.
    pub fn get_killed_or_paused_instance(is_process_chunks: bool) -> BatchRunTomoStatus {
        if is_process_chunks {
            return BatchRunTomoStatus::KilledOrPausedProcessChunks;
        }
        BatchRunTomoStatus::KilledOrPaused
    }

    /// Java static `getInstance(String)`.  (`KILLED_OR_PAUSED_SERIES_WATCHER` is not
    /// recognized by the source; its text yields `DEFAULT`.)
    pub fn get_instance(text: Option<&str>) -> BatchRunTomoStatus {
        let Some(text) = text else {
            return DEFAULT;
        };
        for status in [
            BatchRunTomoStatus::Open,
            BatchRunTomoStatus::Running,
            BatchRunTomoStatus::Pausing,
            BatchRunTomoStatus::Killing,
            BatchRunTomoStatus::Done,
            BatchRunTomoStatus::KilledOrPaused,
            BatchRunTomoStatus::KilledOrPausedProcessChunks,
            BatchRunTomoStatus::Stopped,
            BatchRunTomoStatus::Failed,
        ] {
            if status.text() == text {
                return status;
            }
        }
        DEFAULT
    }

    /// Java `isEndStatus()`.
    pub fn is_end_status(self) -> bool {
        self.end_status()
    }
}

impl Status for BatchRunTomoStatus {
    /// Java `getText()`.
    fn get_text(&self) -> Option<&'static str> {
        Some(self.text())
    }
}

/// Java `toString()`.
impl std::fmt::Display for BatchRunTomoStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[BatchRunTomoStatus{},endStatus:{},errorStatus:{}]",
            to_string_if_set(Some(":text:"), Some(self.text())),
            self.end_status(),
            self.error_status()
        )
    }
}
