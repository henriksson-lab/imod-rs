//! `IMOD/Etomo/src/etomo/ui/BatchRunTomoState.java`.

use crate::imod::etomo::r#type::batch_run_tomo_row_meta_data::StateProperties;
use crate::imod::etomo::r#type::batch_run_tomo_status::{self, BatchRunTomoStatus};
use crate::imod::etomo::r#type::status::StatusRef;

/// Java `public class BatchRunTomoState`.
pub struct BatchRunTomoState {
    /// Java private final `dialogLevel`.
    dialog_level: bool,
    /// Java private `batchRunTomoStatus = BatchRunTomoStatus.DEFAULT`.
    batch_run_tomo_status: Option<BatchRunTomoStatus>,
    /// Java private `resumeEnabled`, initially false.
    resume_enabled: bool,
    /// Java private `processchunkResumeEnabled`, initially false.
    processchunk_resume_enabled: bool,
    /// Java private `stackDumped`, initially false.
    stack_dumped: bool,
}

impl BatchRunTomoState {
    /// Java `BatchRunTomoState(boolean)`.
    pub fn new(dialog_level: bool) -> BatchRunTomoState {
        BatchRunTomoState {
            dialog_level,
            batch_run_tomo_status: Some(batch_run_tomo_status::DEFAULT),
            resume_enabled: false,
            processchunk_resume_enabled: false,
            stack_dumped: false,
        }
    }

    /// Java `handleStatusEvent(Status)`.
    pub fn handle_status_event(&mut self, status: Option<StatusRef>) {
        if let Some(StatusRef::BatchRunTomoStatus(status)) = status {
            // An end status BatchRunTomoStatus can't replace another end status
            // BatchRunTomoStatus.
            if self.batch_run_tomo_status.is_none_or(|current| !current.is_end_status())
                || !status.is_end_status()
            {
                self.batch_run_tomo_status = Some(status);
            }
            // Resume is left enabled until the user cancels it with the Reset button.
            if self.batch_run_tomo_status == Some(BatchRunTomoStatus::Open) {
                self.resume_enabled = false;
                self.processchunk_resume_enabled = false;
            } else if self.batch_run_tomo_status == Some(BatchRunTomoStatus::KilledOrPaused) {
                self.resume_enabled = true;
            } else if self.batch_run_tomo_status
                == Some(BatchRunTomoStatus::KilledOrPausedProcessChunks)
            {
                self.processchunk_resume_enabled = true;
            }
        }
    }

    /// Java `copyFrom(BatchRunTomoRowMetaData.StateProperties)`.
    pub fn copy_from(&mut self, state_properties: Option<&StateProperties>) {
        let Some(state_properties) = state_properties else {
            return;
        };
        let temp_status = BatchRunTomoStatus::get_instance(Some(&state_properties.get_status()));
        self.batch_run_tomo_status = Some(temp_status);
    }

    /// Java `copyTo(BatchRunTomoRowMetaData.StateProperties)`.
    pub fn copy_to(&self, state_properties: Option<&mut StateProperties>) {
        let Some(state_properties) = state_properties else {
            return;
        };
        state_properties.set_status(self.batch_run_tomo_status);
    }

    /// Java `getBatchRunTomoStatus()`.
    pub fn get_batch_run_tomo_status(&self) -> Option<BatchRunTomoStatus> {
        self.batch_run_tomo_status
    }

    /// Java package-private `equalsBatchRunTomoStatus(Status)`.
    pub fn equals_batch_run_tomo_status(&self, status: Option<StatusRef>) -> bool {
        self.batch_run_tomo_status.map(StatusRef::from) == status
    }

    /// Java `isResumeEnabled()`.
    pub fn is_resume_enabled(&self) -> bool {
        self.resume_enabled
    }

    /// Java `isProcesschunkResumeEnabled()`.
    pub fn is_processchunk_resume_enabled(&self) -> bool {
        self.processchunk_resume_enabled
    }
}

/// Java `toString()`.
impl std::fmt::Display for BatchRunTomoState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[batchRunTomoStatus:{},processchunkResumeEnabled:{}]",
            self.batch_run_tomo_status
                .map_or("null".to_owned(), |status| status.to_string()),
            self.processchunk_resume_enabled
        )
    }
}
