//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoStepPanel.java`.
//!
//! Batchruntomo StartingStep and EndingStep: the "Subset of Steps to Run" panel of the
//! batchruntomo dialog's Run tab.  An event dispatch thread object, created as
//! `Rc<Self>` by [`BatchRunTomoStepPanel::get_instance`].

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::batch_run_tomo_table::BatchRunTomoTable;
use super::check_box::CheckBox;
use super::etched_border::EtchedBorder;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::series_watcher_parent::SeriesWatcherParent;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use super::abstract_radio_button_model::AbstractRadioButtonModel;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_meta_data::BatchRunTomoMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_status::{self, BatchRunTomoStatus};
use crate::imod::etomo::r#type::ending_step::EndingStep;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::starting_step::StartingStep;
use crate::imod::etomo::r#type::status::StatusRef;
use crate::imod::etomo::r#type::status_change_event::StatusChangeEvent;
use crate::imod::etomo::r#type::status_change_listener::StatusChangeListener;

/// Java package-private static final `STEP_PAIRS`.
pub const STEP_PAIRS: usize = 5;

/// Java `final class BatchRunTomoStepPanel implements ActionListener,
/// StatusChangeListener`.
pub struct BatchRunTomoStepPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `cbEndingStep`.
    cb_ending_step: Rc<CheckBox>,
    /// Java private final `bgEndingStep`.
    bg_ending_step: Rc<ButtonGroup>,
    /// Java private final `rbEndingStep[]`.
    rb_ending_step: Vec<Rc<RadioButton>>,
    /// Java private final `cbStartingStep`.
    cb_starting_step: Rc<CheckBox>,
    /// Java private final `bgStartingStep`.
    bg_starting_step: Rc<ButtonGroup>,
    /// Java private final `rbStartingStep[]`.
    rb_starting_step: Vec<Rc<RadioButton>>,
    /// Java private final `cbEnableStartingStep`.
    cb_enable_starting_step: Rc<CheckBox>,

    /// Java private final `manager`.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    #[allow(dead_code)]
    axis_id: AxisID,
    /// Java private final `table`.
    table: Weak<BatchRunTomoTable>,
    /// Java private final `seriesWatcherParent`.
    series_watcher_parent: Weak<dyn SeriesWatcherParent>,

    /// Java private `status`, initially `BatchRunTomoStatus.DEFAULT`.
    status: Cell<Option<BatchRunTomoStatus>>,
    /// Java `this`.
    this: Weak<BatchRunTomoStepPanel>,
}

impl BatchRunTomoStepPanel {
    /// Java private `BatchRunTomoStepPanel(BaseManager, AxisID, BatchRunTomoTable,
    /// SeriesWatcherParent)`, with the field initialisers.  The radio buttons, which
    /// Java creates in `createPanel`, are created here with the same arguments and in the
    /// same order.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        table: Weak<BatchRunTomoTable>,
        series_watcher_parent: Weak<dyn SeriesWatcherParent>,
    ) -> Rc<BatchRunTomoStepPanel> {
        let bg_ending_step = ButtonGroup::new();
        let bg_starting_step = ButtonGroup::new();
        let cb_ending_step = CheckBox::new_string(Some("Stop after"));
        let cb_starting_step = CheckBox::new_string(Some("Start from"));
        let cb_enable_starting_step =
            CheckBox::new_string(Some("Enable starting from any step"));
        // createPanel: Step
        let mut rb_ending_step = Vec::with_capacity(STEP_PAIRS);
        let mut rb_starting_step = Vec::with_capacity(STEP_PAIRS);
        for i in 0..STEP_PAIRS {
            if let Some(ending_step) = EndingStep::get_instance(Some(i as i32)) {
                // init
                let rb = RadioButton::new_string_enumerated_type_button_group(
                    ending_step.get_label().as_deref(),
                    Some(EnumeratedTypeRef::new(ending_step)),
                    Some(&bg_ending_step),
                );
                if ending_step.is_default() {
                    rb.set_selected_boolean(true);
                }
                rb.set_tool_tip_text_string(Some(ending_step.get_tooltip()));
                rb_ending_step.push(rb);
            }
            if let Some(starting_step) = StartingStep::get_instance(i as i32) {
                let rb = RadioButton::new_string_enumerated_type_button_group(
                    starting_step.get_label().as_deref(),
                    Some(EnumeratedTypeRef::new(starting_step)),
                    Some(&bg_starting_step),
                );
                if starting_step.is_default() {
                    rb.set_selected_boolean(true);
                }
                rb.set_tool_tip_text_string(Some(starting_step.get_tooltip()));
                rb_starting_step.push(rb);
            }
        }
        Rc::new_cyclic(|this| BatchRunTomoStepPanel {
            pnl_root: JComponent::new_panel(),
            cb_ending_step,
            bg_ending_step,
            rb_ending_step,
            cb_starting_step,
            bg_starting_step,
            rb_starting_step,
            cb_enable_starting_step,
            manager,
            axis_id,
            table,
            series_watcher_parent,
            status: Cell::new(Some(batch_run_tomo_status::DEFAULT)),
            this: this.clone(),
        })
    }

    /// Java package-private static `getInstance(BaseManager, AxisID, BatchRunTomoTable,
    /// SeriesWatcherParent)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        table: Weak<BatchRunTomoTable>,
        series_watcher_parent: Weak<dyn SeriesWatcherParent>,
    ) -> Rc<BatchRunTomoStepPanel> {
        let instance = BatchRunTomoStepPanel::new(manager, axis_id, table, series_watcher_parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    fn table(&self) -> Rc<BatchRunTomoTable> {
        self.table.upgrade().expect("the dialog owns its table")
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_ending_step = JComponent::new_panel();
        let pnl_starting_step = JComponent::new_panel();
        let pnl_step = JComponent::new_panel();
        let pnl_body = JComponent::new_panel();
        let pnl_enable_starting_step = JComponent::new_panel();
        // Root
        self.pnl_root.add(&pnl_body);
        // Body
        pnl_body.set_border_title(
            EtchedBorder::new(Some("Subset of Steps to Run"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_body.add(&pnl_step);
        pnl_body.add(&pnl_enable_starting_step);
        // EnableStartingStep
        pnl_enable_starting_step.add(&self.cb_enable_starting_step.get_component());
        // Step
        pnl_step.add(&pnl_ending_step);
        pnl_step.add(&pnl_starting_step);
        // EndingStep
        pnl_ending_step.add(&self.cb_ending_step.get_component());
        // StartingStep
        pnl_starting_step.add(&self.cb_starting_step.get_component());
        // Step: the buttons were built by the constructor; add the pairs.
        for i in 0..STEP_PAIRS {
            if let (Some(rb_ending), Some(rb_starting)) =
                (self.rb_ending_step.get(i), self.rb_starting_step.get(i))
            {
                // EndingStep
                pnl_ending_step.add(&rb_ending.get_component());
                // StartingStep
                pnl_starting_step.add(&rb_starting.get_component());
            }
        }
        // update
        self.update_display();
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let this = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(panel) = this.upgrade() {
                panel.action_performed(event);
            }
        });
        self.cb_ending_step.add_action_listener(Some(listener.clone()));
        self.cb_starting_step.add_action_listener(Some(listener.clone()));
        for i in 0..STEP_PAIRS {
            self.rb_ending_step[i].add_action_listener(listener.clone());
            self.rb_starting_step[i].add_action_listener(listener.clone());
        }
        self.cb_enable_starting_step
            .add_action_listener(Some(listener.clone()));
        if let Some(this) = self.this.upgrade() {
            let table = self.table();
            table.add_status_change_listener_to_row_list(Some(this.clone()));
            table.add_status_change_listener_to_rows(Some(this));
        }
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _e: &ActionEvent) {
        self.update_display();
    }

    /// The selected button's model, Java
    /// `(RadioButton.RadioButtonModel) group.getSelection()`, with the model's enabled
    /// state (the component's).
    fn selection(group: &ButtonGroup) -> Option<(Rc<JComponent>, Option<EnumeratedTypeRef>)> {
        let selection = group.get_selection()?;
        let enumerated_type = selection.get_model().and_then(|model| {
            model
                .as_any()
                .downcast_ref::<RadioButtonModel>()
                .and_then(AbstractRadioButtonModel::get_enumerated_type)
        });
        Some((selection, enumerated_type))
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        // Don't update the display while it is ineditable during the run.
        if !self.cb_starting_step.is_editable() {
            return;
        }
        // Starting Step - dependent on defined stop point
        let mut starting_step_selection_changed = false;
        let mut starting_step_model: Option<(Rc<JComponent>, Option<EnumeratedTypeRef>)> = None;
        let enabled_starting_step = self.cb_enable_starting_step.is_selected();
        let series_watcher_on = self
            .series_watcher_parent
            .upgrade()
            .is_some_and(|parent| parent.is_series_watcher_on());
        let earliest_run_ending_step = self
            .table
            .upgrade()
            .and_then(|table| table.get_earliest_run_ending_step());
        // Disable starting step if none of the radio buttons are available.
        self.cb_enable_starting_step.set_enabled(!series_watcher_on);
        self.cb_starting_step.set_enabled(
            (enabled_starting_step || earliest_run_ending_step.is_some()) && !series_watcher_on,
        );
        let starting_step_selected =
            self.cb_starting_step.is_enabled() && self.cb_starting_step.is_selected();
        if !starting_step_selected {
            // Checkbox is not checked - yeay - just disable everything
            for i in 0..STEP_PAIRS {
                self.rb_starting_step[i].set_enabled(false);
            }
        } else if enabled_starting_step {
            for i in 0..STEP_PAIRS {
                self.rb_starting_step[i].set_enabled(true);
            }
        } else {
            // Rule:
            // Disable past the defined stop point (paired with run to)
            // Don't jump ahead when restarting
            // defined stop point:
            // earliest completion status of all run checked datasets (disabled or enabled)
            let mut max_enabled = 0;
            // earliestRunStep is the defined stop point
            if let Some(earliest_run_ending_step) = earliest_run_ending_step {
                max_enabled = (earliest_run_ending_step.get_index() + 1) as usize;
            }
            for i in 0..max_enabled.min(STEP_PAIRS) {
                self.rb_starting_step[i].set_enabled(true);
            }
            for i in max_enabled..STEP_PAIRS {
                self.rb_starting_step[i].set_enabled(false);
            }
            if max_enabled < STEP_PAIRS && max_enabled > 0 {
                // At least one radio button was disabled
                // Rule:
                // Move the selection to the first enabled one
                // Move it only if the selected button is disabled
                starting_step_model = Self::selection(&self.bg_starting_step);
                if let Some((selection, _)) = &starting_step_model
                    && !selection.is_enabled()
                {
                    // The selected radio button is now disabled - move it
                    self.rb_starting_step[max_enabled - 1].set_selected_boolean(true);
                    starting_step_selection_changed = true;
                }
            }
        }
        // Ending Step - dependent on Starting Step
        let enable_ending_step =
            self.cb_ending_step.is_enabled() && self.cb_ending_step.is_selected();
        if !enable_ending_step {
            // Checkbox is not checked - yeay - just disable everything
            for i in 0..STEP_PAIRS {
                self.rb_ending_step[i].set_enabled(false);
            }
        } else {
            // Rule:
            // disable up to checked & enabled start from (can't go backwards)
            // End has to be at least one more then start
            let mut enabled_start_index: usize = 0;
            if starting_step_selected {
                // Get the starting step model if it has changed or wasn't already
                // retrieved
                if starting_step_model.is_none() || starting_step_selection_changed {
                    starting_step_model = Self::selection(&self.bg_starting_step);
                }
                if let Some((selection, enumerated_type)) = &starting_step_model
                    && selection.is_enabled()
                {
                    // Java casts the enumerated type to StartingStep.
                    if let Some(starting_step) = enumerated_type
                        .as_ref()
                        .and_then(|enumerated_type| enumerated_type.downcast_ref::<StartingStep>())
                    {
                        enabled_start_index = (starting_step.get_index() + 1) as usize;
                    }
                }
            }
            for i in 0..enabled_start_index.min(STEP_PAIRS) {
                self.rb_ending_step[i].set_enabled(false);
            }
            for i in enabled_start_index..STEP_PAIRS {
                self.rb_ending_step[i].set_enabled(true);
            }
            if enabled_start_index > 0 && enabled_start_index < STEP_PAIRS {
                // some radio buttons where disabled
                // Move the selection to the first enabled one
                // I think this should read "last enabled one"
                // Move it only if the selected button is disabled
                if let Some((selection, _)) = Self::selection(&self.bg_ending_step)
                    && !selection.is_enabled()
                {
                    // The selected radio button is now disabled - move it
                    self.rb_ending_step[enabled_start_index].set_selected_boolean(true);
                }
            }
        }
    }

    /// Java package-private `getParameters(BatchRunTomoMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        meta_data.set_use_ending_step(self.cb_ending_step.is_selected());
        if let Some((_, enumerated_type)) = Self::selection(&self.bg_ending_step)
            && let Some(ending_step) = enumerated_type
                .as_ref()
                .and_then(|enumerated_type| enumerated_type.downcast_ref::<EndingStep>())
        {
            meta_data.set_ending_step(Some(*ending_step));
        }
        meta_data.set_use_starting_step(self.cb_starting_step.is_selected());
        if let Some((_, enumerated_type)) = Self::selection(&self.bg_starting_step)
            && let Some(starting_step) = enumerated_type
                .as_ref()
                .and_then(|enumerated_type| enumerated_type.downcast_ref::<StartingStep>())
        {
            meta_data.set_starting_step(Some(*starting_step));
        }
        meta_data.set_enable_starting_step(self.cb_enable_starting_step.is_selected());
    }

    /// Java package-private `setParameters(BatchRunTomoMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        self.cb_ending_step
            .set_selected_boolean(meta_data.is_use_ending_step());
        if let Some(ending_step) = meta_data.get_ending_step() {
            self.rb_ending_step[ending_step.get_index() as usize].set_selected_boolean(true);
        }
        self.cb_starting_step
            .set_selected_boolean(meta_data.is_use_starting_step());
        if let Some(starting_step) = meta_data.get_starting_step() {
            self.rb_starting_step[starting_step.get_index() as usize].set_selected_boolean(true);
        }
        self.cb_enable_starting_step
            .set_selected_boolean(meta_data.is_enable_starting_step());
        self.status_changed_status(meta_data.get_status().map(StatusRef::BatchRunTomoStatus));
        self.status_changed_status(
            meta_data
                .get_earliest_run_ending_step()
                .map(StatusRef::EndingStep),
        );
    }

    /// Java package-private `setParameters(BatchruntomoParam)`.
    pub fn set_parameters_param(&self, param: &BatchruntomoParam) {
        if let Some(ending_step) =
            EndingStep::get_instance_from_step_value(Some(&param.get_ending_step()))
        {
            self.rb_ending_step[ending_step.get_index() as usize].set_selected_boolean(true);
        }
        if let Some(starting_step) =
            StartingStep::get_instance_from_step_value(Some(&param.get_starting_step()))
        {
            self.rb_starting_step[starting_step.get_index() as usize].set_selected_boolean(true);
        }
        self.update_display();
    }

    /// Java package-private `getParameters(BatchruntomoParam, boolean)`.
    pub fn get_parameters_param(&self, param: &mut BatchruntomoParam, _validate_only: bool) {
        param.reset_ending_step();
        if self.cb_ending_step.is_enabled()
            && self.cb_ending_step.is_selected()
            && let Some((selection, enumerated_type)) = Self::selection(&self.bg_ending_step)
        {
            // `model.getButton().isEnabled()`: the button's own enabled state.
            if selection.is_enabled()
                && let Some(enumerated_type) = enumerated_type
            {
                param.set_ending_step(Some(&enumerated_type.get_value()));
            }
        }
        param.reset_starting_step();
        if self.cb_starting_step.is_enabled()
            && self.cb_starting_step.is_selected()
            && let Some((selection, enumerated_type)) = Self::selection(&self.bg_starting_step)
        {
            if selection.is_enabled()
                && let Some(enumerated_type) = enumerated_type
            {
                param.set_starting_step(Some(&enumerated_type.get_value()));
            }
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.cb_ending_step
            .set_tool_tip_text_string(Some("Process all datasets through the selected step."));
        self.cb_starting_step
            .set_tool_tip_text_string(Some("Start all datasets from the selected step."));
        self.cb_enable_starting_step.set_tool_tip_text_string(Some(
            "Allow 'Start from' to be set past the point reached by all datasets.",
        ));
    }
}

impl StatusChangeListener for BatchRunTomoStepPanel {
    /// Java `statusChanged(Status)`.
    fn status_changed_status(&self, new_status: Option<StatusRef>) {
        if let Some(StatusRef::BatchRunTomoStatus(new_status)) = new_status {
            // Avoid overriding an end state with an error state.
            let status = BatchRunTomoStatus::get_instance_from_statuses(
                self.status.get(),
                Some(new_status),
            );
            self.status.set(status);
            let editable = match status {
                None => true,
                Some(status) => {
                    status == BatchRunTomoStatus::Open
                        || (status.is_end_status() && status != BatchRunTomoStatus::KilledOrPaused)
                }
            };
            self.update_display();
            self.cb_starting_step.set_editable(editable);
            for i in 0..self.rb_starting_step.len() {
                self.rb_starting_step[i].set_editable(editable);
            }
            self.cb_ending_step.set_editable(editable);
            for i in 0..self.rb_starting_step.len() {
                self.rb_ending_step[i].set_editable(editable);
            }
            self.cb_enable_starting_step.set_editable(editable);
        }
        self.update_display();
    }

    /// Java `statusChanged(StatusChangeEvent)`.
    fn status_changed_event(&self, _status_change_event: Option<&dyn StatusChangeEvent>) {
        self.update_display();
    }

    /// Java `startOver()`.
    fn start_over(&self) {
        self.status_changed_status(Some(StatusRef::BatchRunTomoStatus(BatchRunTomoStatus::Open)));
    }
}

