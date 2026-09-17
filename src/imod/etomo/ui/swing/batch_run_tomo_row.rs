//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoRow.java`.
//!
//! Swing widgets, Etomo's manager, 3dmod, directive/autodoc I/O, and process
//! monitoring are deliberately represented as explicit boundaries.  The row's
//! source-owned table state and decisions remain here; callers perform the
//! boundary operations recorded in [`BatchRunTomoRowActions`].
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::batch_run_tomo_table::{
    DATASET_LABEL1, DATASET_LABEL2, LOG_LABEL1, LOG_LABEL2, REC_LABEL1, REC_LABEL2,
};
use super::minibutton_cell::MinibuttonCell;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;

pub const SURFACES_TO_ANALYZE_TWO: &str = "2";
pub const SURFACES_TO_ANALYZE_ONE: &str = "1";
pub const EDIT_DATASET_VALUE: &str = "   Set";
pub const STEP_PAIRS: usize = 12;

/// Java `RunType`, restricted to values dispatched by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunType {
    Run,
    Resume,
    ResumeProcessChunks,
    Reconnect,
}
/// Java `RunStatus` stored by the row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunStatus {
    ToRun,
    Killed,
    Failed,
    Ran,
}
/// Java `BatchRunTomoDatasetState` values which affect this row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchRunTomoDatasetState {
    Running,
    Done,
    ADone,
    Failed,
    Failing,
    Killed,
}
/// Java `BatchRunTomoStatus` values sent by table controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchRunTomoRowStatus {
    Default,
    Open,
    Run,
}
/// Java `BatchRunTomoDatasetStatus` status events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchRunTomoDatasetStatus {
    Renamed,
    Delivered,
}
/// Java `EndingStep`; its index is used by the dual-axis first-step guard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndingStep {
    pub index: usize,
    pub text: String,
}
/// Java status event payload as received from `StatusChangeRowEvent`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchRunTomoRowEvent {
    RowStatus(BatchRunTomoRowStatus),
    DatasetState(BatchRunTomoDatasetState),
    DatasetStatus(BatchRunTomoDatasetStatus, String),
    EndingStep(EndingStep),
    Setup,
    Reconstruction,
    Volcombine,
    Trimvol,
}

/// One Swing cell's observable state.  Rendering and listener wiring are GUI boundaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchRunTomoRowCell {
    pub value: String,
    pub selected: bool,
    pub enabled: bool,
    pub editable: bool,
    pub locked: bool,
    pub highlight: bool,
    pub error: bool,
    pub tooltip: String,
}
impl Default for BatchRunTomoRowCell {
    fn default() -> Self {
        Self {
            value: String::new(),
            selected: false,
            enabled: true,
            editable: true,
            locked: false,
            highlight: false,
            error: false,
            tooltip: String::new(),
        }
    }
}

/// Persistent subset of Java `BatchRunTomoRowState` used to compute the display.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoRowState {
    pub dataset_state: Option<BatchRunTomoDatasetState>,
    pub ending_step: Option<EndingStep>,
    pub ending_step_axis_id: Option<AxisID>,
    pub processchunk_resume_enabled: bool,
    pub active: bool,
    pub cur_recon_step: Option<String>,
    pub run_highlight: bool,
    pub error: bool,
}

/// Operations that Java performs through managers, dialogs, `BatchTool`, and Swing.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoRowActions {
    pub open_stack: Vec<(AxisID, Option<PathBuf>)>,
    pub open_dataset: Option<PathBuf>,
    pub open_tomogram: bool,
    pub open_trimvol: bool,
    pub open_log: Vec<PathBuf>,
    pub close_stack: Vec<AxisID>,
    pub save_row: bool,
    pub initialize_dialog: bool,
    pub table_montage: Option<bool>,
    pub status_notifications: Vec<BatchRunTomoRowStatus>,
}

/// Java package-private final `BatchRunTomoRow`.
pub struct BatchRunTomoRow {
    pub fc_stack: BatchRunTomoRowCell,
    pub cbc_boundary_model: BatchRunTomoRowCell,
    pub cbc_dual: BatchRunTomoRowCell,
    pub cbc_montage: BatchRunTomoRowCell,
    pub fc_skip: BatchRunTomoRowCell,
    pub fc_bskip: BatchRunTomoRowCell,
    pub cbc_surfaces_to_analyze2: BatchRunTomoRowCell,
    pub fc_edit_dataset: BatchRunTomoRowCell,
    pub fc_dataset_state: BatchRunTomoRowCell,
    pub fc_ending_step: BatchRunTomoRowCell,
    pub fc_cur_axis_letter: BatchRunTomoRowCell,
    pub cbc_run: BatchRunTomoRowCell,
    pub mbc_open_dataset: MinibuttonCell,
    pub mbc_image_stack_a: MinibuttonCell,
    pub mbc_image_stack_b: MinibuttonCell,
    pub hc_number: BatchRunTomoRowCell,
    pub bc_edit_dataset: BatchRunTomoRowCell,
    pub mbc_tomogram: MinibuttonCell,
    pub mbc_proj_log: MinibuttonCell,
    pub mbc_brt_log: MinibuttonCell,
    pub first_steps: [bool; STEP_PAIRS],
    pub stack_id: Option<String>,
    pub stack: Option<PathBuf>,
    pub row_state: BatchRunTomoRowState,
    pub imod_index_a: i32,
    pub imod_index_b: i32,
    pub imod_rec: i32,
    pub imod_trim_vol: i32,
    pub status: BatchRunTomoRowStatus,
    pub orig_stack: Option<PathBuf>,
    pub dataset_state: Option<BatchRunTomoDatasetState>,
    pub tomogram_done: bool,
    pub trimvol_done: bool,
    pub log: Option<PathBuf>,
    pub run_status: Option<RunStatus>,
    pub cur_ending_step_a: Option<EndingStep>,
    pub cur_ending_step: Option<EndingStep>,
    pub cur_axis_id: Option<AxisID>,
    pub table_parallel_processing: bool,
    pub table_series_watcher_on: bool,
    pub dialog_deliver: bool,
    pub dataset_dialog_exists: bool,
    pub highlighted: bool,
    pub actions: BatchRunTomoRowActions,
}

impl BatchRunTomoRow {
    /// Java private `BatchRunTomoRow(...)` constructor.
    pub fn new(
        number: i32,
        stack: Option<PathBuf>,
        stack_id: Option<String>,
        axis_type: Option<AxisType>,
        new_row: bool,
    ) -> Self {
        let mut value = Self {
            fc_stack: BatchRunTomoRowCell {
                value: stack
                    .as_ref()
                    .map_or_else(String::new, |path| path.display().to_string()),
                ..Default::default()
            },
            cbc_boundary_model: Default::default(),
            cbc_dual: Default::default(),
            cbc_montage: Default::default(),
            fc_skip: Default::default(),
            fc_bskip: Default::default(),
            cbc_surfaces_to_analyze2: Default::default(),
            fc_edit_dataset: Default::default(),
            fc_dataset_state: Default::default(),
            fc_ending_step: Default::default(),
            fc_cur_axis_letter: Default::default(),
            cbc_run: Default::default(),
            mbc_open_dataset: MinibuttonCell::get_named_etomo_instance(
                Some(DATASET_LABEL1),
                Some(DATASET_LABEL2),
            ),
            mbc_image_stack_a: MinibuttonCell::get_run_3dmod_instance(),
            mbc_image_stack_b: MinibuttonCell::get_run_3dmod_instance(),
            hc_number: Default::default(),
            bc_edit_dataset: Default::default(),
            mbc_tomogram: MinibuttonCell::get_named_run_3dmod_instance(
                Some(REC_LABEL1),
                Some(REC_LABEL2),
            ),
            mbc_proj_log: MinibuttonCell::get_named_etomo_log_instance(
                Some("Proj"),
                Some(LOG_LABEL2),
            ),
            mbc_brt_log: MinibuttonCell::get_named_brt_log_instance(
                Some(LOG_LABEL1),
                Some(LOG_LABEL2),
            ),
            first_steps: [false; STEP_PAIRS],
            stack_id,
            stack,
            row_state: Default::default(),
            imod_index_a: -1,
            imod_index_b: -1,
            imod_rec: -1,
            imod_trim_vol: -1,
            status: BatchRunTomoRowStatus::Default,
            orig_stack: None,
            dataset_state: None,
            tomogram_done: false,
            trimvol_done: false,
            log: None,
            run_status: None,
            cur_ending_step_a: None,
            cur_ending_step: None,
            cur_axis_id: None,
            table_parallel_processing: false,
            table_series_watcher_on: false,
            dialog_deliver: false,
            dataset_dialog_exists: false,
            highlighted: false,
            actions: Default::default(),
        };
        value.hc_number.value = number.to_string();
        value.cbc_run.selected = true;
        if axis_type == Some(AxisType::DualAxis) {
            value.cbc_dual.selected = true;
        }
        if new_row {
            value.mbc_open_dataset.set_editable(false);
            value.mbc_tomogram.set_editable(false);
            value.mbc_proj_log.set_editable(false);
            value.mbc_brt_log.set_editable(false);
        }
        value.set_log();
        value.set_tooltips();
        value.update_display(None, None);
        value
    }
    /// Java `getInstance`.
    pub fn get_instance(
        number: i32,
        stack: PathBuf,
        stack_id: String,
        axis_type: Option<AxisType>,
        new_row: bool,
    ) -> Self {
        let mut value = Self::new(number, Some(stack), Some(stack_id), axis_type, new_row);
        value.add_listeners();
        value
    }
    /// Java `getDefaultsInstance`.
    pub fn get_defaults_instance() -> Self {
        let mut value = Self::new(-1, None, None, None, false);
        value.add_listeners();
        value
    }
    /// Java `getSeriesWatcherInstance`.
    pub fn get_series_watcher_instance(
        number: i32,
        stack_id: String,
        axis_type: AxisType,
        surfaces_to_analyze2: Option<bool>,
    ) -> Self {
        let mut row = Self::new(number, None, Some(stack_id), Some(axis_type), true);
        row.add_listeners();
        if let Some(selected) = surfaces_to_analyze2 {
            row.cbc_surfaces_to_analyze2.selected = selected;
        }
        row
    }

    /// Java private `addListeners`; non-Mini controls remain their respective
    /// translated widget boundaries, while each Minibutton receives the same
    /// unique command and action listener setup here.
    fn add_listeners(&mut self) {
        self.mbc_image_stack_a
            .set_action_command(Some(&self.mbc_image_stack_a.get_unique_action_command()));
        self.mbc_image_stack_b
            .set_action_command(Some(&self.mbc_image_stack_b.get_unique_action_command()));
        self.mbc_open_dataset
            .set_action_command(Some(&self.mbc_open_dataset.get_unique_action_command()));
        self.mbc_tomogram
            .set_action_command(Some(&self.mbc_tomogram.get_unique_action_command()));
        self.mbc_proj_log
            .set_action_command(Some(&self.mbc_proj_log.get_unique_action_command()));
        self.mbc_brt_log
            .set_action_command(Some(&self.mbc_brt_log.get_unique_action_command()));
        self.mbc_image_stack_a.add_action_listener();
        self.mbc_image_stack_b.add_action_listener();
        self.mbc_open_dataset.add_action_listener();
        self.mbc_tomogram.add_action_listener();
        self.mbc_proj_log.add_action_listener();
        self.mbc_brt_log.add_action_listener();
    }
    /// Java `copy(BatchRunTomoRow)`.
    pub fn copy(&mut self, previous: Option<&Self>) {
        if let Some(previous) = previous {
            self.cbc_dual.selected = previous.cbc_dual.selected;
            self.cbc_montage.selected = previous.cbc_montage.selected;
            self.cbc_surfaces_to_analyze2.selected = previous.cbc_surfaces_to_analyze2.selected;
        }
        self.update_display(None, None);
    }
    #[allow(non_snake_case)]
    pub fn setupField(cell: &mut BatchRunTomoRowCell, value: impl Into<String>, editable: bool) {
        cell.value = value.into();
        cell.editable = editable;
    }
    #[allow(non_snake_case)]
    pub fn getAxisType(&self) -> AxisType {
        if self.is_dual() {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        }
    }
    #[allow(non_snake_case)]
    pub fn statusChangedOldVersion(&mut self, state: BatchRunTomoDatasetState) {
        self.status_changed(BatchRunTomoRowEvent::DatasetState(state), false);
    }
    #[allow(non_snake_case)]
    pub fn sendStatusChange(&mut self, status: BatchRunTomoRowStatus) {
        self.actions.status_notifications.push(status);
    }
    pub fn validate(
        &self,
        dataset_dialog_valid: Option<bool>,
        global_dialog_valid: Option<bool>,
    ) -> bool {
        dataset_dialog_valid.or(global_dialog_valid).unwrap_or(true)
    }
    pub fn is_parallel_processing(&self) -> bool {
        self.table_parallel_processing
    }
    pub fn is_dual(&self) -> bool {
        self.cbc_dual.selected
    }
    pub fn is_run(&self) -> bool {
        self.cbc_run.enabled && self.cbc_run.selected
    }
    pub fn is_montage(&self) -> bool {
        self.cbc_montage.selected
    }
    pub fn get_ending_step(&self) -> Option<&EndingStep> {
        if self.row_state.dataset_state == Some(BatchRunTomoDatasetState::Done)
            || (self.is_dual() && self.row_state.ending_step_axis_id != Some(AxisID::Second))
        {
            None
        } else {
            self.row_state.ending_step.as_ref()
        }
    }
    pub fn get_dataset_state(&self) -> Option<BatchRunTomoDatasetState> {
        self.dataset_state
    }
    pub fn get_run_status(&self) -> Option<RunStatus> {
        self.run_status
    }
    pub fn get_stack_id(&self) -> Option<&str> {
        self.stack_id.as_deref()
    }
    pub fn get_stack(&self) -> Option<&Path> {
        self.stack.as_deref()
    }
    pub fn get_stack_path(&self) -> Option<&Path> {
        self.stack.as_deref().and_then(Path::parent)
    }
    /// Java `imodStack(FileType)`: the model-name build is supplied at the file boundary.
    pub fn imod_stack(&mut self, model_file: Option<PathBuf>) -> Option<PathBuf> {
        self.imod_stack_axis(model_file.clone(), AxisID::First, self.is_dual());
        model_file
    }
    pub fn imod_stack_axis(&mut self, model_file: Option<PathBuf>, axis_id: AxisID, _dual: bool) {
        self.actions.open_stack.push((axis_id, model_file));
        if axis_id == AxisID::Second {
            self.imod_index_b += 1;
        } else {
            self.imod_index_a += 1;
        }
    }
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action(&mut self, action_command: &str) {
        match action_command {
            "dual" => self.update_display(None, None),
            "image_stack_a" => {
                let model = if self.cbc_boundary_model.selected {
                    self.boundary_model_file()
                } else {
                    None
                };
                self.imod_stack(model);
            }
            "image_stack_b" => self.imod_stack_axis(None, AxisID::Second, self.is_dual()),
            "open_dataset" => self.actions.open_dataset = self.get_dataset_file(),
            "tomogram" => {
                if self.trimvol_done {
                    self.actions.open_trimvol = true;
                } else {
                    self.actions.open_tomogram = true;
                }
            }
            "brt_log" => {
                if let Some(log) = &self.log {
                    self.actions.open_log.push(log.clone());
                }
            }
            "proj_log" => {
                if let Some(log) = self.get_project_log() {
                    self.actions.open_log.push(log);
                }
            }
            "boundary_model" if self.cbc_boundary_model.selected && self.imod_index_a != -1 => {
                self.imod_stack(self.boundary_model_file());
            }
            "edit_dataset" => {
                self.dataset_dialog_exists = true;
                self.actions.initialize_dialog = true;
                self.fc_edit_dataset.value = EDIT_DATASET_VALUE.into();
                self.bc_edit_dataset.selected = true;
            }
            "run" => self
                .actions
                .status_notifications
                .push(BatchRunTomoRowStatus::Run),
            "montage" => self.actions.table_montage = Some(self.cbc_montage.selected),
            _ => {}
        }
    }
    pub fn get_dataset_file(&self) -> Option<PathBuf> {
        self.stack.as_ref().map(|stack| {
            let mut file = stack.clone();
            file.set_extension(if self.is_dual() { "edf" } else { "edf" });
            file
        })
    }
    pub fn remove(&mut self) {
        self.cbc_run.enabled = false;
        self.mbc_open_dataset.set_enabled(false);
        self.mbc_tomogram.set_enabled(false);
    }
    pub fn delete(&mut self) {
        self.delete_dataset();
    }
    pub fn delete_dataset(&mut self) {
        if self.dataset_dialog_exists {
            self.dataset_dialog_exists = false;
            self.bc_edit_dataset.selected = false;
            self.fc_edit_dataset.value.clear();
            self.actions.save_row = true;
        }
    }
    pub fn is_edit_dataset(&self) -> bool {
        self.bc_edit_dataset.selected
    }
    /// Java `updateDisplay(RunType, Boolean)`.
    pub fn update_display(&mut self, run_type: Option<RunType>, validate_only: Option<bool>) {
        let dual = self.is_dual();
        self.fc_bskip.enabled = dual;
        self.mbc_image_stack_b.set_enabled(dual);
        self.cbc_run.editable =
            !self.table_series_watcher_on && !self.row_state.processchunk_resume_enabled;
        self.fc_dataset_state.value = self
            .dataset_state
            .map_or_else(String::new, |s| format!("{s:?}"));
        self.fc_dataset_state.highlight = self.row_state.run_highlight;
        self.fc_dataset_state.error = self.row_state.error;
        self.fc_ending_step.value = self
            .row_state
            .ending_step
            .as_ref()
            .map_or_else(String::new, |step| step.text.clone());
        self.fc_cur_axis_letter.value = self
            .cur_axis_id
            .map_or_else(String::new, |axis| axis.key().to_owned());
        let directory_set = self.row_state.cur_recon_step.is_some();
        self.cbc_dual.editable = !directory_set;
        if self.dialog_deliver
            && run_type.is_some()
            && validate_only == Some(false)
            && !directory_set
        {
            if self.mbc_image_stack_a.is_editable() {
                self.mbc_image_stack_a.set_editable(false);
                self.actions
                    .close_stack
                    .push(if dual { AxisID::First } else { AxisID::Only });
            }
            if dual && self.mbc_image_stack_b.is_editable() {
                self.mbc_image_stack_b.set_editable(false);
                self.actions.close_stack.push(AxisID::Second);
            }
        } else {
            self.mbc_image_stack_a.set_editable(true);
            self.mbc_image_stack_b.set_editable(true);
        }
        self.mbc_open_dataset.set_editable(directory_set);
        let running_without_directory =
            !directory_set && self.dataset_state == Some(BatchRunTomoDatasetState::Running);
        self.mbc_proj_log
            .set_editable(!running_without_directory && directory_set);
        self.mbc_brt_log
            .set_editable(self.mbc_proj_log.is_editable());
        self.mbc_tomogram.set_editable(
            self.row_state.cur_recon_step.as_deref() == Some("RECONSTRUCTION")
                && (!dual || self.tomogram_done)
                || matches!(
                    self.row_state.cur_recon_step.as_deref(),
                    Some("VOLCOMBINE") | Some("TRIMVOL")
                ),
        );
        self.mbc_open_dataset.set_locked(self.row_state.active);
        self.mbc_tomogram.set_locked(self.row_state.active);
        for cell in [
            &mut self.cbc_dual,
            &mut self.cbc_run,
            &mut self.cbc_boundary_model,
            &mut self.cbc_montage,
            &mut self.fc_skip,
            &mut self.fc_bskip,
            &mut self.cbc_surfaces_to_analyze2,
            &mut self.bc_edit_dataset,
        ] {
            cell.locked = self.row_state.active;
        }
    }
    pub fn start_over(&mut self) {
        self.status_changed(
            BatchRunTomoRowEvent::RowStatus(BatchRunTomoRowStatus::Open),
            false,
        );
        self.run_status = None;
    }
    pub fn status_changed(&mut self, event: BatchRunTomoRowEvent, init: bool) {
        match event {
            BatchRunTomoRowEvent::RowStatus(status) => {
                self.status = status;
                self.update_display(None, None);
            }
            BatchRunTomoRowEvent::DatasetState(state) => {
                self.dataset_state = Some(state);
                self.row_state.dataset_state = Some(state);
                self.row_state.active = matches!(
                    state,
                    BatchRunTomoDatasetState::Running | BatchRunTomoDatasetState::Failing
                );
                self.row_state.error = state == BatchRunTomoDatasetState::Failing;
                self.row_state.run_highlight = state == BatchRunTomoDatasetState::Running;
                if state == BatchRunTomoDatasetState::Done {
                    self.cbc_run.selected = false;
                }
                if !init && state == BatchRunTomoDatasetState::Failed {
                    self.run_status = Some(RunStatus::Failed);
                }
                if !init && state == BatchRunTomoDatasetState::Killed {
                    self.run_status = Some(RunStatus::Killed);
                }
                self.update_display(None, None);
            }
            BatchRunTomoRowEvent::DatasetStatus(BatchRunTomoDatasetStatus::Renamed, file) => {
                if let Some(stack) = &self.stack {
                    self.stack = Some(stack.parent().unwrap_or(Path::new("")).join(file));
                }
            }
            BatchRunTomoRowEvent::DatasetStatus(BatchRunTomoDatasetStatus::Delivered, file) => {
                self.orig_stack
                    .get_or_insert_with(|| self.stack.clone().unwrap_or_default());
                self.stack = Some(PathBuf::from(file));
                self.set_log();
                self.cbc_dual.locked = true;
                self.update_display(None, None);
            }
            BatchRunTomoRowEvent::EndingStep(step) => {
                if self.is_dual() && step.index < STEP_PAIRS && !self.first_steps[step.index] {
                    self.first_steps[step.index] = true;
                    self.cur_axis_id = Some(AxisID::First);
                } else if self.cur_axis_id.is_none() {
                    self.cur_axis_id = Some(if self.is_dual() {
                        AxisID::Second
                    } else {
                        AxisID::Only
                    });
                }
                if self.cur_axis_id == Some(AxisID::First) {
                    self.cur_ending_step_a = Some(step.clone());
                } else {
                    self.cur_ending_step = Some(step.clone());
                }
                self.row_state.ending_step = Some(step);
                self.row_state.ending_step_axis_id = self.cur_axis_id;
                self.update_display(None, None);
            }
            BatchRunTomoRowEvent::Setup => {
                self.mbc_open_dataset.set_enabled(true);
                self.mbc_tomogram.set_enabled(true);
                self.mbc_proj_log.set_enabled(true);
                self.mbc_brt_log.set_enabled(true);
            }
            BatchRunTomoRowEvent::Reconstruction if !self.is_dual() => self.tomogram_done = true,
            BatchRunTomoRowEvent::Volcombine => {
                if self.is_dual() {
                    self.tomogram_done = true;
                }
                self.fc_cur_axis_letter.value.clear();
            }
            BatchRunTomoRowEvent::Trimvol => {
                self.trimvol_done = true;
                self.fc_cur_axis_letter.value.clear();
            }
            _ => {}
        }
    }
    pub fn reset_ending_step(&mut self) {
        self.fc_ending_step.value.clear();
        if self.cur_axis_id == Some(AxisID::First) {
            self.cur_ending_step_a = None;
        } else {
            self.cur_ending_step = None;
        }
    }
    pub fn set_ending_step(&mut self, ending_step: Option<EndingStep>) {
        if ending_step.is_none() {
            self.reset_ending_step();
            return;
        }
        if self.cur_axis_id == Some(AxisID::First) {
            self.cur_ending_step_a = ending_step;
        } else {
            self.cur_ending_step = ending_step;
        }
    }
    pub fn set_cur_axis_id(&mut self, axis_id: Option<AxisID>) {
        if (self.is_dual() && axis_id != Some(AxisID::Only))
            || (!self.is_dual() && axis_id == Some(AxisID::Only))
        {
            self.cur_axis_id = axis_id;
        }
    }
    pub fn set_cur_axis_id_from_boolean(&mut self, set_axis_id: bool) {
        if set_axis_id {
            self.cur_axis_id = Some(if self.is_dual() {
                AxisID::Second
            } else {
                AxisID::Only
            });
        }
    }
    pub fn set_cur_axis_letter(&mut self, axis_id: Option<AxisID>) {
        if let Some(axis_id) = axis_id {
            self.fc_cur_axis_letter.value = axis_id.key().to_owned();
        }
    }
    pub fn add_status_change_listener(&mut self) { /* Java listener identity is a controller boundary. */
    }
    pub fn display(&mut self, in_viewport: bool, _tab: &str) {
        if !in_viewport {
            return;
        }
    }
    pub fn expand_stack(&mut self, _expanded: bool) {}
    pub fn highlight(&mut self, highlight: bool) {
        self.highlighted = highlight;
        for cell in [
            &mut self.fc_stack,
            &mut self.cbc_boundary_model,
            &mut self.cbc_dual,
            &mut self.cbc_montage,
            &mut self.fc_skip,
            &mut self.fc_bskip,
            &mut self.cbc_surfaces_to_analyze2,
            &mut self.fc_edit_dataset,
            &mut self.fc_dataset_state,
            &mut self.fc_ending_step,
            &mut self.fc_cur_axis_letter,
            &mut self.cbc_run,
        ] {
            cell.highlight = highlight;
        }
    }
    pub fn set_error(&mut self, error: bool) {
        for cell in [
            &mut self.cbc_boundary_model,
            &mut self.cbc_dual,
            &mut self.cbc_montage,
            &mut self.fc_skip,
            &mut self.fc_bskip,
            &mut self.cbc_surfaces_to_analyze2,
            &mut self.fc_edit_dataset,
            &mut self.fc_dataset_state,
            &mut self.fc_ending_step,
            &mut self.fc_cur_axis_letter,
            &mut self.cbc_run,
        ] {
            cell.error = error;
        }
    }
    pub fn equals_stack_id(&self, stack_id: &str) -> bool {
        self.stack_id.as_deref() == Some(stack_id)
    }
    pub fn get_parameters(&self) -> BatchRunTomoRowMetadata {
        BatchRunTomoRowMetadata {
            row_number: self.hc_number.value.clone(),
            dual: self.cbc_dual.selected,
            run: self.cbc_run.selected,
            bskip: self.fc_bskip.value.clone(),
            dataset_dialog: self.dataset_dialog_exists,
            orig_stack: self.orig_stack.clone(),
            tomogram_done: self.tomogram_done,
            trimvol_done: self.trimvol_done,
            dataset_state: self.dataset_state,
            ending_step_a: self.cur_ending_step_a.clone(),
            ending_step: self.cur_ending_step.clone(),
            cur_axis_letter: self.fc_cur_axis_letter.value.clone(),
            run_status: self.run_status,
        }
    }
    pub fn set_parameters(&mut self, metadata: &BatchRunTomoRowMetadata, init: bool) {
        self.hc_number.value = metadata.row_number.clone();
        self.cbc_dual.selected = metadata.dual;
        self.cbc_run.selected = metadata.run;
        self.fc_bskip.value = metadata.bskip.clone();
        self.dataset_dialog_exists = metadata.dataset_dialog;
        self.orig_stack = metadata.orig_stack.clone();
        self.tomogram_done = metadata.tomogram_done;
        self.trimvol_done = metadata.trimvol_done;
        self.dataset_state = metadata.dataset_state;
        self.cur_ending_step_a = metadata.ending_step_a.clone();
        self.cur_ending_step = metadata.ending_step.clone();
        self.fc_cur_axis_letter.value = metadata.cur_axis_letter.clone();
        self.run_status = metadata.run_status;
        if init {
            self.update_display(None, None);
        }
    }
    pub fn equals(&self, stack_id: &str) -> bool {
        self.equals_stack_id(stack_id)
    }
    pub fn equals_location_root_name(&self, location: &Path, root_name: &str) -> bool {
        let Some(stack) = &self.stack else {
            return false;
        };
        stack.parent() == Some(location)
            && stack.file_stem().and_then(|s| s.to_str()) == Some(root_name)
    }
    /// Java `setupRun(RunType, boolean)`.
    pub fn setup_run(&mut self, run_type: RunType, validate_only: bool) -> bool {
        let run = self.is_run();
        let mut temporary = self.run_status;
        let result = match run_type {
            RunType::Run => {
                temporary = run.then_some(RunStatus::ToRun);
                run
            }
            RunType::Resume | RunType::ResumeProcessChunks => {
                if run {
                    if matches!(temporary, None | Some(RunStatus::Killed)) {
                        temporary = Some(RunStatus::ToRun);
                    }
                    if run_type == RunType::Resume {
                        temporary == Some(RunStatus::ToRun)
                    } else {
                        temporary.is_some()
                    }
                } else {
                    temporary = None;
                    false
                }
            }
            RunType::Reconnect => temporary.is_some(),
        };
        if !validate_only {
            self.run_status = temporary;
        }
        self.update_display(Some(run_type), Some(validate_only));
        result
    }
    pub fn get_parameters_for_run(&mut self, run_type: RunType, validate_only: bool) -> bool {
        self.setup_run(run_type, validate_only)
    }
    pub fn get_autodoc_file(&self) -> Option<PathBuf> {
        let stack = self.orig_stack.as_ref().or(self.stack.as_ref())?;
        let parent = stack.parent()?;
        Some(parent.join(format!(
            "batchruntomo_{}.adoc",
            stack.file_stem()?.to_string_lossy()
        )))
    }
    pub fn get_project_log(&self) -> Option<PathBuf> {
        let stack = self.stack.as_ref()?;
        let candidate = stack.parent()?.join("project.log");
        candidate.exists().then_some(candidate)
    }
    pub fn load_autodoc(&mut self, _only_load_dataset: bool, _only_advanced_dataset_dialog: bool) {}
    pub fn save_autodoc(&mut self, do_validation: bool) -> bool {
        !do_validation || self.stack.as_ref().and_then(|s| s.extension()).is_some()
    }
    pub fn is_highlighted(&self) -> bool {
        self.highlighted
    }
    pub fn select_highlight_button(&mut self) {
        self.highlighted = true;
    }
    pub fn backup_if_changed(
        &mut self,
        _only_dataset_dialog: bool,
        _only_advanced_dataset_dialog: bool,
    ) -> bool {
        false
    }
    pub fn apply_values(
        &mut self,
        _init: bool,
        _retain_user_values: bool,
        _only_dataset_dialog: bool,
        _only_advanced_dataset_dialog: bool,
    ) {
        self.update_display(None, None);
    }
    pub fn set_number(&mut self, input: i32) {
        self.hc_number.value = input.to_string();
    }
    pub fn set_values(
        &mut self,
        dual: Option<bool>,
        montage: Option<bool>,
        surfaces: Option<bool>,
    ) {
        if let Some(value) = dual {
            self.cbc_dual.selected = value;
        }
        if let Some(value) = montage {
            self.cbc_montage.selected = value;
        }
        if let Some(value) = surfaces {
            self.cbc_surfaces_to_analyze2.selected = value;
        }
    }
    pub fn set_values_user_configuration(&mut self, single_axis: bool, montage: bool) {
        if !self.table_series_watcher_on && single_axis {
            self.cbc_dual.selected = false;
        }
        self.cbc_montage.selected = montage;
        self.update_display(None, None);
    }
    fn set_log(&mut self) {
        self.log = self
            .stack
            .as_ref()
            .and_then(|stack| stack.parent())
            .map(|parent| parent.join("batchruntomo.log"));
    }
    fn boundary_model_file(&self) -> Option<PathBuf> {
        self.stack
            .as_ref()
            .and_then(|stack| stack.parent())
            .map(|parent| parent.join("boundary.mod"))
    }
    fn set_tooltips(&mut self) {
        self.cbc_dual.tooltip = "Dual axis dataset".into();
        self.cbc_montage.tooltip = "Montage".into();
        self.cbc_run.tooltip = "This dataset will be included in the batchruntomo run".into();
    }
}

/// Native event adapter for Java `BatchRunTomoRowActionListener`.
pub struct BatchRunTomoRowActionListener;
impl BatchRunTomoRowActionListener {
    #[allow(non_snake_case)]
    pub fn actionPerformed(row: &mut BatchRunTomoRow, command: &str) {
        row.action(command);
    }
}

/// Java `BatchRunTomoRowMetaData` members saved and restored by this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoRowMetadata {
    pub row_number: String,
    pub dual: bool,
    pub run: bool,
    pub bskip: String,
    pub dataset_dialog: bool,
    pub orig_stack: Option<PathBuf>,
    pub tomogram_done: bool,
    pub trimvol_done: bool,
    pub dataset_state: Option<BatchRunTomoDatasetState>,
    pub ending_step_a: Option<EndingStep>,
    pub ending_step: Option<EndingStep>,
    pub cur_axis_letter: String,
    pub run_status: Option<RunStatus>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_copies_axis_and_new_row_editability() {
        let row = BatchRunTomoRow::get_instance(
            3,
            PathBuf::from("/d/stacka.mrc"),
            "id".into(),
            Some(AxisType::DualAxis),
            true,
        );
        assert!(row.is_dual());
        assert!(row.is_run());
        assert!(!row.mbc_open_dataset.is_editable());
        assert_eq!(row.hc_number.value, "3");
    }
    #[test]
    fn dual_axis_ending_step_waits_for_second_axis() {
        let mut row = BatchRunTomoRow::get_instance(
            1,
            PathBuf::from("x.mrc"),
            "x".into(),
            Some(AxisType::DualAxis),
            false,
        );
        row.status_changed(
            BatchRunTomoRowEvent::EndingStep(EndingStep {
                index: 2,
                text: "Align".into(),
            }),
            false,
        );
        assert_eq!(row.cur_axis_id, Some(AxisID::First));
        assert_eq!(row.get_ending_step(), None);
        // `StatusChangeRowEvent.getCurAxisID()` supplies the B axis on the
        // second report; model that event boundary explicitly here.
        row.set_cur_axis_id(Some(AxisID::Second));
        row.status_changed(
            BatchRunTomoRowEvent::EndingStep(EndingStep {
                index: 2,
                text: "Align".into(),
            }),
            false,
        );
        assert_eq!(row.cur_axis_id, Some(AxisID::Second));
        assert!(row.get_ending_step().is_some());
    }
    #[test]
    fn setup_run_obeys_resume_and_validate_only() {
        let mut row = BatchRunTomoRow::get_defaults_instance();
        row.cbc_run.selected = true;
        assert!(row.setup_run(RunType::Run, true));
        assert_eq!(row.run_status, None);
        assert!(row.setup_run(RunType::Run, false));
        assert_eq!(row.run_status, Some(RunStatus::ToRun));
        row.cbc_run.selected = false;
        assert!(!row.setup_run(RunType::Resume, false));
        assert_eq!(row.run_status, None);
    }
}
