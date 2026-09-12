//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoStepPanel.java`.
//!
//! Swing layout, listener dispatch, the table, and the series-watcher parent
//! stay explicit boundaries.  The selection and enablement rules are retained
//! here because they are the behaviour owned by `BatchRunTomoStepPanel`.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use super::check_box::CheckBox;
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

pub const STEP_PAIRS: usize = 5;

/// `EndingStep`, at the not-yet-translated `etomo.type` boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndingStep {
    BeadTracking,
    FineAlignment,
    Positioning,
    GoldDetection3d,
    TwoDFiltering,
}

impl EndingStep {
    pub fn get_instance(index: usize) -> Option<Self> {
        [
            Self::BeadTracking,
            Self::FineAlignment,
            Self::Positioning,
            Self::GoldDetection3d,
            Self::TwoDFiltering,
        ]
        .get(index)
        .copied()
    }
    pub fn get_index(self) -> usize {
        match self {
            Self::BeadTracking => 0,
            Self::FineAlignment => 1,
            Self::Positioning => 2,
            Self::GoldDetection3d => 3,
            Self::TwoDFiltering => 4,
        }
    }
    pub fn is_default(self) -> bool {
        self == Self::GoldDetection3d
    }
    pub fn get_label(self) -> &'static str {
        match self {
            Self::BeadTracking => "Fiducial model generation",
            Self::FineAlignment => "Fine alignment",
            Self::Positioning => "Tomogram positioning",
            Self::GoldDetection3d => "CTF estimation and gold detection in 3D",
            Self::TwoDFiltering => "2D filtering",
        }
    }
    pub fn get_tooltip(self) -> &'static str {
        match self {
            Self::BeadTracking => "Stop after fiducial model generation or patch tracking.",
            Self::FineAlignment => "Stop after fine alignment with Tiltalign.",
            Self::Positioning => "Stop after tomogram positioning (if any).",
            Self::GoldDetection3d => {
                "Stop after CTF estimation and detection of gold in 3D (if any)."
            }
            Self::TwoDFiltering => "Stop when all steps on the aligned stack are completed.",
        }
    }
}

/// `StartingStep`, at the not-yet-translated `etomo.type` boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartingStep {
    FineAlignment,
    Positioning,
    AlignedStackGeneration,
    CtfCorrection,
    Reconstruction,
}

impl StartingStep {
    pub fn get_instance(index: usize) -> Option<Self> {
        [
            Self::FineAlignment,
            Self::Positioning,
            Self::AlignedStackGeneration,
            Self::CtfCorrection,
            Self::Reconstruction,
        ]
        .get(index)
        .copied()
    }
    pub fn get_index(self) -> usize {
        match self {
            Self::FineAlignment => 0,
            Self::Positioning => 1,
            Self::AlignedStackGeneration => 2,
            Self::CtfCorrection => 3,
            Self::Reconstruction => 4,
        }
    }
    pub fn is_default(self) -> bool {
        self == Self::CtfCorrection
    }
    pub fn get_label(self) -> &'static str {
        match self {
            Self::FineAlignment => "Fine alignment",
            Self::Positioning => "Tomogram positioning",
            Self::AlignedStackGeneration => "Aligned stack generation",
            Self::CtfCorrection => "CTF correction",
            Self::Reconstruction => "Reconstruction",
        }
    }
    pub fn get_tooltip(self) -> &'static str {
        match self {
            Self::FineAlignment => "Start from fine alignment with Tiltalign.",
            Self::Positioning => "Start with tomogram positioning (if any).",
            Self::AlignedStackGeneration => {
                "Start with generating the aligned stack from the raw stack."
            }
            Self::CtfCorrection => "Start with correcting the CTF then erasing the gold (if any).",
            Self::Reconstruction => "Start with making the reconstruction.",
        }
    }
}

/// `BatchRunTomoStatus`, at the `etomo.type` boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchRunTomoStatus {
    Open,
    Running,
    Pausing,
    Killing,
    Done,
    KilledOrPaused,
    KilledOrPausedProcessChunks,
    KilledOrPausedSeriesWatcher,
    Stopped,
    Failed,
}

impl BatchRunTomoStatus {
    pub const DEFAULT: Self = Self::Open;
    pub fn is_end_status(self) -> bool {
        matches!(
            self,
            Self::Done
                | Self::KilledOrPaused
                | Self::KilledOrPausedProcessChunks
                | Self::KilledOrPausedSeriesWatcher
                | Self::Stopped
                | Self::Failed
        )
    }
    pub fn is_error_status(self) -> bool {
        self == Self::Failed
    }
    /// Java `BatchRunTomoStatus.getInstance(curStatus, newStatus)`.
    pub fn get_instance(current: Option<Self>, new_status: Option<Self>) -> Option<Self> {
        if current.is_none()
            || new_status.is_none()
            || !current.is_some_and(Self::is_end_status)
            || !new_status.is_some_and(Self::is_end_status)
        {
            return new_status;
        }
        if new_status.is_some_and(Self::is_error_status) {
            current
        } else {
            new_status
        }
    }
}

/// Narrow `BatchRunTomoTable` boundary used by this source unit.
#[derive(Clone, Debug, Default)]
pub struct BatchRunTomoTableBoundary {
    pub earliest_run_ending_step: Option<EndingStep>,
    pub row_list_status_listener_count: usize,
    pub rows_status_listener_count: usize,
}

impl BatchRunTomoTableBoundary {
    pub fn get_earliest_run_ending_step(&self) -> Option<EndingStep> {
        self.earliest_run_ending_step
    }
    pub fn add_status_change_listener_to_row_list(&mut self) {
        self.row_list_status_listener_count += 1;
    }
    pub fn add_status_change_listener_to_rows(&mut self) {
        self.rows_status_listener_count += 1;
    }
}

/// Narrow `SeriesWatcherParent` boundary used by this source unit.
pub trait SeriesWatcherParent {
    fn is_series_watcher_on(&self) -> bool;
    fn equals_series_watcher_action_command(&self, action_command: &str) -> bool;
}

/// `BatchRunTomoMetaData` fields read/written by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoMetaDataBoundary {
    pub use_ending_step: bool,
    pub ending_step: Option<EndingStep>,
    pub use_starting_step: bool,
    pub starting_step: Option<StartingStep>,
    pub enable_starting_step: bool,
    pub status: Option<BatchRunTomoStatus>,
    pub earliest_run_ending_step: Option<EndingStep>,
}

/// `BatchruntomoParam` subset read/written by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchruntomoParamBoundary {
    pub ending_step: Option<usize>,
    pub starting_step: Option<usize>,
}

/// Java package-private `BatchRunTomoStepPanel`.
pub struct BatchRunTomoStepPanel<'a> {
    pub pnl_root_layout: Vec<&'static str>,
    pub cb_ending_step: CheckBox,
    pub bg_ending_step: Rc<RefCell<RadioButtonGroup>>,
    pub rb_ending_step: [RadioButton; STEP_PAIRS],
    pub cb_starting_step: CheckBox,
    pub bg_starting_step: Rc<RefCell<RadioButtonGroup>>,
    pub rb_starting_step: [RadioButton; STEP_PAIRS],
    pub cb_enable_starting_step: CheckBox,
    pub manager: &'a dyn BaseManager,
    pub axis_id: AxisID,
    pub table: &'a RefCell<BatchRunTomoTableBoundary>,
    pub series_watcher_parent: &'a dyn SeriesWatcherParent,
    pub status: Option<BatchRunTomoStatus>,
}

impl<'a> BatchRunTomoStepPanel<'a> {
    fn new(
        manager: &'a dyn BaseManager,
        axis_id: AxisID,
        table: &'a RefCell<BatchRunTomoTableBoundary>,
        series_watcher_parent: &'a dyn SeriesWatcherParent,
    ) -> Self {
        let bg_ending_step = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let bg_starting_step = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root_layout: Vec::new(),
            cb_ending_step: CheckBox::new_with_text("Stop after"),
            bg_ending_step: bg_ending_step.clone(),
            rb_ending_step: std::array::from_fn(|index| {
                let step = EndingStep::get_instance(index).unwrap();
                RadioButton::new_with_enumerated_type(
                    Some(step.get_label().into()),
                    EnumeratedTypeBoundary {
                        label: step.get_label().into(),
                        default: step.is_default(),
                        value: Some(step.get_index().to_string()),
                    },
                    Some(bg_ending_step.clone()),
                )
            }),
            cb_starting_step: CheckBox::new_with_text("Start from"),
            bg_starting_step: bg_starting_step.clone(),
            rb_starting_step: std::array::from_fn(|index| {
                let step = StartingStep::get_instance(index).unwrap();
                RadioButton::new_with_enumerated_type(
                    Some(step.get_label().into()),
                    EnumeratedTypeBoundary {
                        label: step.get_label().into(),
                        default: step.is_default(),
                        value: Some(step.get_index().to_string()),
                    },
                    Some(bg_starting_step.clone()),
                )
            }),
            cb_enable_starting_step: CheckBox::new_with_text("Enable starting from any step"),
            manager,
            axis_id,
            table,
            series_watcher_parent,
            status: Some(BatchRunTomoStatus::DEFAULT),
        }
    }

    /// Java `getInstance`.
    pub fn get_instance(
        manager: &'a dyn BaseManager,
        axis_id: AxisID,
        table: &'a RefCell<BatchRunTomoTableBoundary>,
        series_watcher_parent: &'a dyn SeriesWatcherParent,
    ) -> Self {
        let mut instance = Self::new(manager, axis_id, table, series_watcher_parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java `createPanel`; Swing component construction is retained as ordered layout state.
    fn create_panel(&mut self) {
        self.pnl_root_layout = vec![
            "Y_AXIS",
            "vertical-strut:8",
            "body",
            "vertical-glue",
            "body:Y_AXIS:etched:Subset of Steps to Run",
            "step:X_AXIS",
            "ending:Y_AXIS",
            "starting:Y_AXIS",
            "enable-starting:X_AXIS",
            "horizontal-glue",
            "align-left",
            "shrink-wrap-horizontal",
        ];
        for index in 0..STEP_PAIRS {
            self.rb_ending_step[index]
                .set_tooltip(Some(EndingStep::get_instance(index).unwrap().get_tooltip()));
            self.rb_starting_step[index].set_tooltip(Some(
                StartingStep::get_instance(index).unwrap().get_tooltip(),
            ));
        }
        self.update_display();
    }

    fn add_listeners(&mut self) {
        self.cb_ending_step.add_action_listener();
        self.cb_starting_step.add_action_listener();
        for index in 0..STEP_PAIRS {
            self.rb_ending_step[index].add_action_listener();
            self.rb_starting_step[index].add_action_listener();
        }
        self.cb_enable_starting_step.add_action_listener();
        let mut table = self.table.borrow_mut();
        table.add_status_change_listener_to_row_list();
        table.add_status_change_listener_to_rows();
    }

    pub fn action_performed(&mut self) {
        self.update_display();
    }
    pub fn start_over(&mut self) {
        self.status_changed(Some(BatchRunTomoStatus::Open));
    }
    pub fn status_changed(&mut self, new_status: Option<BatchRunTomoStatus>) {
        if let Some(new_status) = new_status {
            self.status = BatchRunTomoStatus::get_instance(self.status, Some(new_status));
            let editable = self.status.is_none()
                || self.status == Some(BatchRunTomoStatus::Open)
                || (self.status.is_some_and(BatchRunTomoStatus::is_end_status)
                    && self.status != Some(BatchRunTomoStatus::KilledOrPaused));
            self.update_display();
            self.cb_starting_step.set_editable(editable);
            for index in 0..STEP_PAIRS {
                self.rb_starting_step[index].set_editable(editable);
            }
            self.cb_ending_step.set_editable(editable);
            for index in 0..STEP_PAIRS {
                self.rb_ending_step[index].set_editable(editable);
            }
            self.cb_enable_starting_step.set_editable(editable);
        }
        self.update_display();
    }
    pub fn status_changed_event(&mut self) {
        self.update_display();
    }

    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        if !self.cb_starting_step.is_editable() {
            return;
        }
        let mut starting_step_selection_changed = false;
        let enabled_starting_step = self.cb_enable_starting_step.is_selected();
        let series_watcher_on = self.series_watcher_parent.is_series_watcher_on();
        let earliest_run_ending_step = self.table.borrow().get_earliest_run_ending_step();
        self.cb_enable_starting_step.set_enabled(!series_watcher_on);
        self.cb_starting_step.set_enabled(
            (enabled_starting_step || earliest_run_ending_step.is_some()) && !series_watcher_on,
        );
        let starting_step_selected =
            self.cb_starting_step.is_enabled() && self.cb_starting_step.is_selected();
        if !starting_step_selected {
            for index in 0..STEP_PAIRS {
                self.rb_starting_step[index].set_enabled(false);
            }
        } else if enabled_starting_step {
            for index in 0..STEP_PAIRS {
                self.rb_starting_step[index].set_enabled(true);
            }
        } else {
            let max_enabled = earliest_run_ending_step.map_or(0, |step| step.get_index() + 1);
            for index in 0..max_enabled {
                self.rb_starting_step[index].set_enabled(true);
            }
            for index in max_enabled..STEP_PAIRS {
                self.rb_starting_step[index].set_enabled(false);
            }
            if max_enabled < STEP_PAIRS && max_enabled > 0 {
                if self
                    .rb_starting_step
                    .iter()
                    .position(RadioButton::is_selected)
                    .is_some_and(|index| !self.rb_starting_step[index].is_enabled())
                {
                    self.rb_starting_step[max_enabled - 1].set_selected(true);
                    starting_step_selection_changed = true;
                }
            }
        }
        let enable_ending_step =
            self.cb_ending_step.is_enabled() && self.cb_ending_step.is_selected();
        if !enable_ending_step {
            for index in 0..STEP_PAIRS {
                self.rb_ending_step[index].set_enabled(false);
            }
        } else {
            let mut enabled_start_index = 0;
            if starting_step_selected {
                if starting_step_selection_changed
                    || self.rb_starting_step.iter().any(RadioButton::is_selected)
                {
                    if let Some(index) = self
                        .rb_starting_step
                        .iter()
                        .position(|button| button.is_selected() && button.is_enabled())
                    {
                        enabled_start_index = index + 1;
                    }
                }
            }
            for index in 0..enabled_start_index {
                self.rb_ending_step[index].set_enabled(false);
            }
            for index in enabled_start_index..STEP_PAIRS {
                self.rb_ending_step[index].set_enabled(true);
            }
            if enabled_start_index > 0 && enabled_start_index < STEP_PAIRS {
                if self
                    .rb_ending_step
                    .iter()
                    .position(RadioButton::is_selected)
                    .is_some_and(|index| !self.rb_ending_step[index].is_enabled())
                {
                    self.rb_ending_step[enabled_start_index].set_selected(true);
                }
            }
        }
    }

    pub fn get_parameters_meta_data(&self, meta_data: &mut BatchRunTomoMetaDataBoundary) {
        meta_data.use_ending_step = self.cb_ending_step.is_selected();
        meta_data.ending_step = self
            .rb_ending_step
            .iter()
            .position(RadioButton::is_selected)
            .and_then(EndingStep::get_instance);
        meta_data.use_starting_step = self.cb_starting_step.is_selected();
        meta_data.starting_step = self
            .rb_starting_step
            .iter()
            .position(RadioButton::is_selected)
            .and_then(StartingStep::get_instance);
        meta_data.enable_starting_step = self.cb_enable_starting_step.is_selected();
    }
    pub fn set_parameters_meta_data(&mut self, meta_data: &BatchRunTomoMetaDataBoundary) {
        self.cb_ending_step.set_selected(meta_data.use_ending_step);
        if let Some(step) = meta_data.ending_step {
            self.rb_ending_step[step.get_index()].set_selected(true);
        }
        self.cb_starting_step
            .set_selected(meta_data.use_starting_step);
        if let Some(step) = meta_data.starting_step {
            self.rb_starting_step[step.get_index()].set_selected(true);
        }
        self.cb_enable_starting_step
            .set_selected(meta_data.enable_starting_step);
        self.status_changed(meta_data.status);
        self.update_display();
    }
    pub fn set_parameters_batchruntomo(&mut self, param: &BatchruntomoParamBoundary) {
        if let Some(step) = param.ending_step.and_then(EndingStep::get_instance) {
            self.rb_ending_step[step.get_index()].set_selected(true);
        }
        if let Some(step) = param.starting_step.and_then(StartingStep::get_instance) {
            self.rb_starting_step[step.get_index()].set_selected(true);
        }
        self.update_display();
    }
    pub fn get_parameters_batchruntomo(
        &self,
        param: &mut BatchruntomoParamBoundary,
        _validate_only: bool,
    ) {
        param.ending_step = None;
        if self.cb_ending_step.is_enabled() && self.cb_ending_step.is_selected() {
            param.ending_step = self
                .rb_ending_step
                .iter()
                .position(|button| button.is_selected() && button.is_enabled());
        }
        param.starting_step = None;
        if self.cb_starting_step.is_enabled() && self.cb_starting_step.is_selected() {
            param.starting_step = self
                .rb_starting_step
                .iter()
                .position(|button| button.is_selected() && button.is_enabled());
        }
    }
    fn set_tooltips(&mut self) {
        self.cb_ending_step
            .set_tooltip(Some("Process all datasets through the selected step."));
        self.cb_starting_step
            .set_tooltip(Some("Start all datasets from the selected step."));
        self.cb_enable_starting_step.set_tooltip(Some(
            "Allow 'Start from' to be set past the point reached by all datasets.",
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    struct Parent(bool);
    impl SeriesWatcherParent for Parent {
        fn is_series_watcher_on(&self) -> bool {
            self.0
        }
        fn equals_series_watcher_action_command(&self, _: &str) -> bool {
            false
        }
    }
    #[test]
    fn source_defaults_and_listeners_are_constructed() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let table = RefCell::new(BatchRunTomoTableBoundary::default());
        let parent = Parent(false);
        let panel = BatchRunTomoStepPanel::get_instance(manager, AxisID::Only, &table, &parent);
        assert!(panel.rb_ending_step[3].is_selected());
        assert!(panel.rb_starting_step[3].is_selected());
        assert_eq!(table.borrow().row_list_status_listener_count, 1);
        assert_eq!(table.borrow().rows_status_listener_count, 1);
    }
    #[test]
    fn start_from_cannot_pass_earliest_run_stop() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let table = RefCell::new(BatchRunTomoTableBoundary {
            earliest_run_ending_step: Some(EndingStep::Positioning),
            ..Default::default()
        });
        let parent = Parent(false);
        let mut panel = BatchRunTomoStepPanel::get_instance(manager, AxisID::Only, &table, &parent);
        panel.cb_starting_step.set_selected(true);
        panel.rb_starting_step[4].set_selected(true);
        panel.update_display();
        assert!(panel.rb_starting_step[0].is_enabled());
        assert!(panel.rb_starting_step[2].is_enabled());
        assert!(!panel.rb_starting_step[3].is_enabled());
        assert!(panel.rb_starting_step[2].is_selected());
    }
    #[test]
    fn series_watcher_disables_start_controls_and_param_uses_enabled_selection() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let table = RefCell::new(BatchRunTomoTableBoundary {
            earliest_run_ending_step: Some(EndingStep::TwoDFiltering),
            ..Default::default()
        });
        let parent = Parent(true);
        let mut panel = BatchRunTomoStepPanel::get_instance(manager, AxisID::Only, &table, &parent);
        panel.cb_starting_step.set_selected(true);
        panel.cb_ending_step.set_selected(true);
        panel.update_display();
        let mut param = BatchruntomoParamBoundary::default();
        panel.get_parameters_batchruntomo(&mut param, false);
        assert!(!panel.cb_enable_starting_step.is_enabled());
        assert!(!panel.cb_starting_step.is_enabled());
        assert_eq!(param.starting_step, None);
        assert_eq!(param.ending_step, Some(3));
    }
}
