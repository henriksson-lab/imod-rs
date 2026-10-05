//! `IMOD/Etomo/src/etomo/ui/BatchRunTomoRowState.java`.
//!
//! Provides selected settings from the Batchruntomo dataset table row.  An event
//! dispatch thread object owned by its `BatchRunTomoRow`.

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_dataset_state::BatchRunTomoDatasetState;
use crate::imod::etomo::r#type::batch_run_tomo_row_meta_data::StateProperties;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::ending_step::EndingStep;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::run_status::RunStatus;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::status::{Status, StatusRef};
use crate::imod::etomo::r#type::step::Step;
use crate::imod::etomo::ui::batch_run_tomo_state::BatchRunTomoState;
use crate::imod::etomo::util::clean_print::CleanPrint;

/// Java `public final class BatchRunTomoRowState`.
pub struct BatchRunTomoRowState {
    /// Java private final `cleanPrint = CleanPrint.getInstance(new String[] { "ebt1" },
    /// "BatchRunTomoRowState")`.
    clean_print: CleanPrint,

    // Updates done
    /// Java private final `stackID`.
    stack_id: Option<String>,

    // State variables
    /// Java private `batchRunTomoState = new BatchRunTomoState(false)`.
    batch_run_tomo_state: BatchRunTomoState,
    /// Java public `datasetState`, initially null.
    pub dataset_state: Option<BatchRunTomoDatasetState>,
    /// Java private `datasetStateInit`, initially false.
    dataset_state_init: bool,
    /// Java private `step`, initially null.
    step: Option<Step>,
    /// Java private `endingStep`, initially null.
    ending_step: Option<EndingStep>,
    /// Java private `endingStepAxisID`, initially null.
    ending_step_axis_id: Option<AxisID>,
    /// Java private `processName`, initially null.
    process_name: Option<ProcessName>,
    /// Java private `curReconStep`, initially null.
    cur_recon_step: Option<StatusRef>,
    /// Java private `print`, initially false.
    print: bool,
    /// Java package-private `printMsg`, initially null.
    print_msg: Option<String>,
}

impl BatchRunTomoRowState {
    /// Java `BatchRunTomoRowState(String)`.
    pub fn new(stack_id: Option<&str>) -> BatchRunTomoRowState {
        BatchRunTomoRowState {
            clean_print: CleanPrint::get_instance_with_message_labels(
                Some(&["ebt1".to_owned()]),
                Some("BatchRunTomoRowState"),
            ),
            stack_id: stack_id.map(str::to_owned),
            batch_run_tomo_state: BatchRunTomoState::new(false),
            dataset_state: None,
            dataset_state_init: false,
            step: None,
            ending_step: None,
            ending_step_axis_id: None,
            process_name: None,
            cur_recon_step: None,
            print: stack_id == Some("ebt1"),
            print_msg: None,
        }
    }

    /// Java `printState(String)`.
    pub fn print_state(&self, request_stack_id: Option<&str>) {
        if request_stack_id.is_some() && request_stack_id == self.stack_id.as_deref() {
            self.clean_print
                .print_labelled(self.stack_id.as_deref(), Some(&format!("A:{}", self)));
        }
    }

    /// Java `handleStatusEvent(Status, AxisID, boolean)`.  Updates state variables
    /// according to status.  Status's other than BatchRunTomoStatus,
    /// BatchRunTomoDatasetState, EndingStep, Step, and ProcessName are ignored.
    /// EventAxisID is ignored unless the status is an EndingStep.
    pub fn handle_status_event(
        &mut self,
        status: Option<StatusRef>,
        event_axis_id: Option<AxisID>,
        init: bool,
    ) {
        if let Some(StatusRef::EndingStep(ending_step)) = status {
            // Save EndingStep information.
            self.ending_step = Some(ending_step);
            self.cur_recon_step = Some(StatusRef::EndingStep(ending_step));
            self.ending_step_axis_id = event_axis_id;
        } else {
            self.handle_status_event_init(status, init);
        }
    }

    /// Java `isProcesschunkResumeEnabled()`.
    pub fn is_processchunk_resume_enabled(&self) -> bool {
        self.batch_run_tomo_state.is_processchunk_resume_enabled()
    }

    /// Java private `handleStatusEvent(Status, boolean)`.  Updates state variables
    /// according to status.  Status's other than BatchRunTomoStatus,
    /// BatchRunTomoDatasetState, Step, and ProcessName are ignored.
    fn handle_status_event_init(&mut self, status: Option<StatusRef>, init: bool) {
        match status {
            Some(StatusRef::BatchRunTomoStatus(_)) => {
                self.batch_run_tomo_state.handle_status_event(status);
            }
            Some(StatusRef::BatchRunTomoDatasetState(new_dataset_state)) => {
                // Inactive dataset states have limitations in terms of what inactive
                // states they can replace.
                if !new_dataset_state.is_active()
                    && self.dataset_state.is_none_or(|state| !state.is_active())
                {
                    // Failed: A failed state can't be replaced by an inactive state,
                    // because that would hide the failure.
                    if self.dataset_state == Some(BatchRunTomoDatasetState::Failed) {
                        return;
                    }
                    // Killed: A killed state can't replace an inactive DatasetState. If
                    // the process is already inactive then it wasn't killed.
                    if new_dataset_state == BatchRunTomoDatasetState::Killed {
                        return;
                    }
                }
                // Done or Running: Hide/delete EndingStep data.
                if new_dataset_state == BatchRunTomoDatasetState::Running
                    || new_dataset_state == BatchRunTomoDatasetState::ADone
                    || new_dataset_state == BatchRunTomoDatasetState::Done
                {
                    self.ending_step = None;
                    // A_DONE: delete the endingStep, but not the endingStep axisID
                    if new_dataset_state != BatchRunTomoDatasetState::ADone {
                        self.ending_step_axis_id = None;
                    }
                }
                // BatchRunTomoDatasetState can be saved.
                self.dataset_state = Some(new_dataset_state);
                self.dataset_state_init = init;
            }
            Some(StatusRef::Step(step)) => {
                self.step = Some(step);
                self.cur_recon_step = Some(StatusRef::Step(step));
            }
            Some(StatusRef::ProcessName(process_name)) => {
                self.process_name = Some(process_name);
                self.cur_recon_step = Some(StatusRef::ProcessName(process_name));
            }
            _ => {}
        }
    }

    /// Java `copyFrom(BatchRunTomoRowMetaData.StateProperties)`.  Set all state
    /// variables in this class from rowProperties.  Use a deep copy or the equivalent.
    /// Reset any state variable where the corresponding property is missing.
    ///
    /// CopyFrom copies (deeply or by value) from RowProperties into this RowState and
    /// does not retain references to metadata internals.
    pub fn copy_from(&mut self, state_properties: Option<&StateProperties>) {
        let Some(state_properties) = state_properties else {
            return;
        };
        self.batch_run_tomo_state.copy_from(Some(state_properties));
        let temp_dataset_state =
            BatchRunTomoDatasetState::get_instance(Some(&state_properties.get_dataset_state()));
        if temp_dataset_state.is_some() {
            self.dataset_state = temp_dataset_state;
        }
        let temp_dataset_state_init = state_properties.get_dataset_state_init();
        if let Some(temp_dataset_state_init) = temp_dataset_state_init {
            self.dataset_state_init = temp_dataset_state_init;
        }
        let temp_step = Step::get_instance(Some(&state_properties.get_step()));
        if let Some(temp_step) = temp_step {
            self.step = Some(temp_step);
            if state_properties.is_step_current() {
                self.cur_recon_step = Some(StatusRef::Step(temp_step));
            }
        }
        let temp_ending_step = EndingStep::get_instance(Some(state_properties.get_ending_step()));
        if let Some(temp_ending_step) = temp_ending_step {
            self.ending_step = Some(temp_ending_step);
            if state_properties.is_ending_step_current() {
                self.cur_recon_step = Some(StatusRef::EndingStep(temp_ending_step));
            }
        }
        let temp_ending_step_axis_id =
            AxisID::get_instance_from_key(&state_properties.get_ending_step_axis_id());
        if temp_ending_step_axis_id.is_some() {
            self.ending_step_axis_id = temp_ending_step_axis_id;
        }

        let temp_process_name =
            ProcessName::get_instance(Some(&state_properties.get_process_name()));
        if let Some(temp_process_name) = temp_process_name {
            self.process_name = Some(temp_process_name);
            if state_properties.is_process_name_current() {
                self.cur_recon_step = Some(StatusRef::ProcessName(temp_process_name));
            }
        }
    }

    /// Java `copyTo(BatchRunTomoRowMetaData.StateProperties)`.  Set all row properties
    /// from the state variables in this class from rowProperties.  Use a deep copy or
    /// the equivalent.  RowProperties functions must handle nulls.
    ///
    /// copyTo writes current state values into the supplied RowProperties.   It does
    /// not pass references to mutable instances.
    pub fn copy_to(&self, state_properties: Option<&mut StateProperties>) {
        let Some(state_properties) = state_properties else {
            // `new IllegalArgumentException(...).printStackTrace()`.
            eprintln!(
                "java.lang.IllegalArgumentException: Warning: missing rowProperties.  Row state not transfered."
            );
            return;
        };
        self.batch_run_tomo_state.copy_to(Some(state_properties));
        state_properties.set_dataset_state(self.dataset_state);
        state_properties.set_dataset_state_init(self.dataset_state_init);
        state_properties.set_step(self.step);
        state_properties.set_ending_step(self.ending_step);
        state_properties.set_ending_step_axis_id(self.ending_step_axis_id);
        state_properties.set_process_name(self.process_name);
        state_properties.set_cur_recon_step(self.cur_recon_step);
    }

    /// Java `equalsBatchRunTomoStatus(Status)`.
    pub fn equals_batch_run_tomo_status(&self, status: Option<StatusRef>) -> bool {
        self.batch_run_tomo_state.equals_batch_run_tomo_status(status)
    }

    /// Java `equalsBatchRunTomoDatasetState(Status)`.
    pub fn equals_batch_run_tomo_dataset_state(&self, dataset_state: Option<StatusRef>) -> bool {
        self.dataset_state.map(StatusRef::from) == dataset_state
    }

    /// Java `equalsStep(Status)`.
    pub fn equals_step(&self, step: Option<StatusRef>) -> bool {
        self.step.map(StatusRef::from) == step
    }

    /// Java `equalsEndingStep(Status)`.
    pub fn equals_ending_step(&self, ending_step: Option<StatusRef>) -> bool {
        self.ending_step.map(StatusRef::from) == ending_step
    }

    /// Java `isRunHighlight()`.
    pub fn is_run_highlight(&self) -> bool {
        self.dataset_state
            .is_some_and(|state| state.is_active() && !state.is_error())
    }

    /// Java `isActive()`.
    pub fn is_active(&self) -> bool {
        self.dataset_state.is_some_and(|state| state.is_active())
    }

    /// Java `isError()`.
    pub fn is_error(&self) -> bool {
        self.dataset_state.is_some_and(|state| state.is_error())
    }

    /// Java `getRunStatus(RunStatus, RunType, boolean, Boolean)`.
    pub fn get_run_status(
        &self,
        cur_run_status: Option<RunStatus>,
        run_type: Option<RunType>,
        run: bool,
        validate_only: Option<bool>,
    ) -> Option<RunStatus> {
        if self.dataset_state.is_none()
            || self.dataset_state_init
            || (run_type == Some(RunType::Run) && run && validate_only == Some(false))
        {
            return cur_run_status;
        }
        if self
            .batch_run_tomo_state
            .equals_batch_run_tomo_status(Some(StatusRef::BatchRunTomoStatus(
                BatchRunTomoStatus::Open,
            )))
        {
            return None;
        }
        let dataset_state = self.dataset_state;
        if dataset_state == Some(BatchRunTomoDatasetState::ADone)
            || dataset_state == Some(BatchRunTomoDatasetState::Failing)
            || dataset_state == Some(BatchRunTomoDatasetState::Running)
            || dataset_state == Some(BatchRunTomoDatasetState::Starting)
        {
            return Some(RunStatus::ToRun);
        }
        if dataset_state == Some(BatchRunTomoDatasetState::Done)
            || dataset_state == Some(BatchRunTomoDatasetState::Stopped)
        {
            return Some(RunStatus::Ran);
        }
        if dataset_state == Some(BatchRunTomoDatasetState::Failed) {
            return Some(RunStatus::Failed);
        }
        if dataset_state == Some(BatchRunTomoDatasetState::Killed) {
            return Some(RunStatus::Killed);
        }
        None
    }

    /// Java `getDatasetState()`.
    pub fn get_dataset_state(&self) -> Option<BatchRunTomoDatasetState> {
        self.dataset_state
    }

    /// Java `getDatasetStateValue()`.
    pub fn get_dataset_state_value(&self) -> String {
        match self.dataset_state {
            None => String::new(),
            Some(state) => state.get_text().unwrap_or("").to_owned(),
        }
    }

    /// Java `getEndingStep()`.
    pub fn get_ending_step(&self) -> Option<EndingStep> {
        self.ending_step
    }

    /// Java `getEndingStepValue()`.
    pub fn get_ending_step_value(&self) -> String {
        match self.ending_step {
            None => String::new(),
            Some(ending_step) => ending_step.get_text().unwrap_or("").to_owned(),
        }
    }

    /// Java `getEndingStepAxisID()`.
    pub fn get_ending_step_axis_id(&self) -> Option<AxisID> {
        self.ending_step_axis_id
    }

    /// Java `getEndingStepAxisLetter()`.
    pub fn get_ending_step_axis_letter(&self) -> String {
        match self.ending_step_axis_id {
            None => String::new(),
            Some(axis_id) => axis_id.get_upper_case_extension(),
        }
    }

    /// Java `getCurReconStep()`.
    pub fn get_cur_recon_step(&self) -> Option<StatusRef> {
        self.cur_recon_step
    }
}

/// Java `toString()`.
impl std::fmt::Display for BatchRunTomoRowState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opt = |value: Option<String>| value.unwrap_or("null".to_owned());
        write!(
            f,
            "[stackID:{},\nbatchRunTomoState:{},\ndatasetState:{},datasetStateInit:{},\nstep:{},\nendingStep:{},endingStepAxisID:{},\nprocessName:{},curReconStep:{}]",
            self.stack_id.as_deref().unwrap_or("null"),
            self.batch_run_tomo_state,
            opt(self.dataset_state.map(|value| value.to_string())),
            self.dataset_state_init,
            opt(self.step.map(|value| value.to_string())),
            opt(self.ending_step.map(|value| value.to_string())),
            opt(self.ending_step_axis_id.map(|value| value.to_string())),
            opt(self.process_name.map(|value| value.to_string())),
            opt(self.cur_recon_step.map(|value| value.to_string()))
        )
    }
}
