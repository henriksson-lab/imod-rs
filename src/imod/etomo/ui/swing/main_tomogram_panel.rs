//! `IMOD/Etomo/src/etomo/ui/swing/MainTomogramPanel.java`.
//!
//! The main panel of the reconstruction interface (`ApplicationManager`):
//! one or two `TomogramProcessPanel`s, or the setup dialog.  Extends
//! [`MainPanel`] (held as `base`, dereffed to) and implements
//! [`MainPanelVirtual`].

use crate::imod::etomo::base_manager::BaseManager;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanelVirtual;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::{MainPanel, MainPanelVirtual};
use super::setup_dialog_expert::SetupDialogExpert;
use super::tomogram_process_panel::TomogramProcessPanel;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::jdk::{FileFilter, JComponent};
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::etomo_file_filter::EtomoFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_track::ProcessTrack;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java public class `MainTomogramPanel extends MainPanel`.
pub struct MainTomogramPanel {
    /// The Java superclass part.
    base: Rc<MainPanel>,
    /// Java `(ApplicationManager) manager`, the constructor's argument.
    application_manager: &'static ApplicationManager,
    axis_panel_a: RefCell<Option<Rc<TomogramProcessPanel>>>,
    axis_panel_b: RefCell<Option<Rc<TomogramProcessPanel>>>,
}

impl Deref for MainTomogramPanel {
    type Target = MainPanel;
    fn deref(&self) -> &MainPanel {
        &self.base
    }
}

impl MainTomogramPanel {
    /// Java constructor `MainTomogramPanel(ApplicationManager)`.
    pub fn new(app_manager: &'static ApplicationManager) -> Rc<MainTomogramPanel> {
        let this = Rc::new(MainTomogramPanel {
            base: MainPanel::new(app_manager),
            application_manager: app_manager,
            axis_panel_a: RefCell::new(None),
            axis_panel_b: RefCell::new(None),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn MainPanelVirtual>);
        this
    }

    /// Java `updateAllProcessingStates(ProcessTrack)`.  Update the state of all
    /// the process control panels.
    pub fn update_all_processing_states(&self, process_track: &ProcessTrack) {
        let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() else {
            return;
        };
        axis_panel_a.set_pre_proc_state(process_track.get_pre_processing_state(AxisID::Only));
        axis_panel_a.set_coarse_align_state(process_track.get_coarse_alignment_state(AxisID::Only));
        axis_panel_a.set_fiducial_model_state(process_track.get_fiducial_model_state(AxisID::Only));
        axis_panel_a.set_fine_alignment_state(process_track.get_fine_alignment_state(AxisID::Only));
        axis_panel_a.set_tomogram_positioning_state(
            process_track.get_tomogram_positioning_state(AxisID::Only),
        );
        axis_panel_a.set_final_aligned_stack_state(
            process_track.get_final_aligned_stack_state(AxisID::Only),
        );
        axis_panel_a.set_tomogram_generation_state(
            process_track.get_tomogram_generation_state(AxisID::Only),
        );
        axis_panel_a.set_tomogram_combination_state(process_track.get_tomogram_combination_state());
        if self.base.manager.is_dual_axis() {
            // Upstream: axisPanelB is dereferenced unchecked in the Java.
            if let Some(axis_panel_b) = self.axis_panel_b.borrow().clone() {
                axis_panel_b
                    .set_pre_proc_state(process_track.get_pre_processing_state(AxisID::Second));
                axis_panel_b.set_coarse_align_state(
                    process_track.get_coarse_alignment_state(AxisID::Second),
                );
                axis_panel_b.set_fiducial_model_state(
                    process_track.get_fiducial_model_state(AxisID::Second),
                );
                axis_panel_b.set_fine_alignment_state(
                    process_track.get_fine_alignment_state(AxisID::Second),
                );
                axis_panel_b.set_tomogram_positioning_state(
                    process_track.get_tomogram_positioning_state(AxisID::Second),
                );
                axis_panel_b.set_final_aligned_stack_state(
                    process_track.get_final_aligned_stack_state(AxisID::Second),
                );
                axis_panel_b.set_tomogram_generation_state(
                    process_track.get_tomogram_generation_state(AxisID::Second),
                );
            }
        }
        axis_panel_a.set_post_processing_state(process_track.get_post_processing_state());
        axis_panel_a.set_clean_up_state(process_track.get_clean_up_state());
    }

    /// Java `openSetupPanel(SetupDialogExpert)`.  Open the setup panel.
    pub fn open_setup_panel(&self, setup_dialog_expert: &SetupDialogExpert) {
        self.base.set_showing_setup(true);
        self.base.panel_center.remove_all();
        let container: Rc<JComponent> = setup_dialog_expert.get_container();
        self.base.panel_center.add(&container);
        // Swing layout: revalidate().
        ui_harness::INSTANCE.with(|harness| harness.pack_base_manager(Some(self.base.manager)));
    }

    /// Java `selectButton(AxisID, String)`.  Set the specified button as
    /// selected.
    pub fn select_button(&self, axis_id: AxisID, name: &str) {
        // Upstream: the Java dereferences mapAxis's result unchecked.
        if let Some(axis_panel) = self.map_axis(axis_id) {
            axis_panel.select_button(name);
        }
    }

    /// Java final `setState(ProcessState, AxisID, DialogType)`.
    pub fn set_state_process_state_axis_id_dialog_type(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        if dialog_type == DialogType::CleanUp {
            self.set_clean_up_state(process_state);
        } else if dialog_type == DialogType::CoarseAlignment {
            self.set_coarse_align_state(process_state, axis_id);
        } else if dialog_type == DialogType::FiducialModel {
            self.set_fiducial_model_state(process_state, axis_id);
        } else if dialog_type == DialogType::FineAlignment {
            self.set_fine_alignment_state(process_state, axis_id);
        } else if dialog_type == DialogType::PostProcessing {
            self.set_post_processing_state(process_state);
        } else if dialog_type == DialogType::PreProcessing {
            self.set_pre_processing_state(process_state, axis_id);
        } else if dialog_type == DialogType::TomogramCombination {
            self.set_tomogram_combination_state(process_state);
        } else if dialog_type == DialogType::FinalAlignedStack {
            self.set_final_aligned_stack_state(process_state, axis_id);
        } else if dialog_type == DialogType::TomogramGeneration {
            self.set_tomogram_generation_state(process_state, axis_id);
        } else if dialog_type == DialogType::TomogramPositioning {
            self.set_tomogram_positioning_state(process_state, axis_id);
        }
    }

    // In the following per-process setters the Java dereferences mapAxis's
    // result (or axisPanelA) unchecked; a null panel is skipped here.

    /// Java `setPreProcessingState(ProcessState, AxisID)`.
    pub fn set_pre_processing_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_pre_proc_state(state);
        }
    }

    /// Java `setCoarseAlignState(ProcessState, AxisID)`.
    pub fn set_coarse_align_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_coarse_align_state(state);
        }
    }

    /// Java `setFiducialModelState(ProcessState, AxisID)`.
    pub fn set_fiducial_model_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_fiducial_model_state(state);
        }
    }

    /// Java `setFineAlignmentState(ProcessState, AxisID)`.
    pub fn set_fine_alignment_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_fine_alignment_state(state);
        }
    }

    /// Java `setTomogramPositioningState(ProcessState, AxisID)`.
    pub fn set_tomogram_positioning_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_tomogram_positioning_state(state);
        }
    }

    /// Java `setFinalAlignedStackState(ProcessState, AxisID)`.
    pub fn set_final_aligned_stack_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_final_aligned_stack_state(state);
        }
    }

    /// Java `setTomogramGenerationState(ProcessState, AxisID)`.
    pub fn set_tomogram_generation_state(&self, state: ProcessState, axis_id: AxisID) {
        let axis_panel = self.map_axis(axis_id);
        if let Some(axis_panel) = axis_panel {
            axis_panel.set_tomogram_generation_state(state);
        }
    }

    /// Java `setTomogramCombinationState(ProcessState)`.
    pub fn set_tomogram_combination_state(&self, state: ProcessState) {
        if let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() {
            axis_panel_a.set_tomogram_combination_state(state);
        }
    }

    /// Java `setPostProcessingState(ProcessState)`.
    pub fn set_post_processing_state(&self, state: ProcessState) {
        if let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() {
            axis_panel_a.set_post_processing_state(state);
        }
    }

    /// Java `setCleanUpState(ProcessState)`.
    pub fn set_clean_up_state(&self, state: ProcessState) {
        if let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() {
            axis_panel_a.set_clean_up_state(state);
        }
    }

    /// Java `getCPUsSelectedInt(AxisID, boolean)`.
    pub fn get_cpus_selected_int(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<i32, FieldValidationFailedException> {
        if axis_id == AxisID::Second {
            if let Some(axis_panel_b) = self.axis_panel_b.borrow().clone() {
                return axis_panel_b.get_cpus_selected_int(do_validation);
            }
            return Ok(0);
        }
        if let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() {
            return axis_panel_a.get_cpus_selected_int(do_validation);
        }
        Ok(0)
    }

    /// Java private `mapAxis(AxisID)`.  Convenience function to return a
    /// reference to the correct AxisProcessPanel.
    fn map_axis(&self, axis_id: AxisID) -> Option<Rc<TomogramProcessPanel>> {
        if axis_id == AxisID::Second {
            return self.axis_panel_b.borrow().clone();
        }
        self.axis_panel_a.borrow().clone()
    }
}

impl MainPanelVirtual for MainTomogramPanel {
    fn main_panel(&self) -> &MainPanel {
        &self.base
    }

    /// Java `getDataFileFilter()`.
    fn get_data_file_filter(&self) -> Option<Rc<dyn FileFilter>> {
        Some(Rc::new(EtomoFileFilter))
    }

    /// Java `saveDisplayState()`.
    fn save_display_state(&self) {
        if let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() {
            axis_panel_a.save_display_state();
        }
        if let Some(axis_panel_b) = self.axis_panel_b.borrow().clone() {
            axis_panel_b.save_display_state();
        }
    }

    /// Java `showAxisA()` override.
    fn show_axis_a(&self) {
        if self.base.is_showing_setup() || self.base.axis_type.get() == AxisType::SingleAxis {
            ui_harness::INSTANCE
                .with(|harness| harness.pack_boolean_base_manager(true, Some(self.base.manager)));
        } else if let Some(axis_panel_a) = self.axis_panel_a.borrow().clone() {
            axis_panel_a.show_axis_a_void();
            self.base.show_axis_a_super();
        }
    }

    /// Java `showAxisB()` override.
    fn show_axis_b(&self) {
        // Upstream: axisPanelB is dereferenced unchecked in the Java.
        if let Some(axis_panel_b) = self.axis_panel_b.borrow().clone() {
            axis_panel_b.show_axis_b_void();
        }
        self.base.show_axis_b_super();
    }

    /// Java `showBothAxis()` override.
    fn show_both_axis(&self) -> Option<Rc<JComponent>> {
        // Upstream: axisPanelB is dereferenced unchecked in the Java.
        if let Some(axis_panel_b) = self.axis_panel_b.borrow().clone() {
            axis_panel_b.show_axis_b_void();
        }
        self.base.show_both_axis_super()
    }

    /// Java `showProcessingPanel(AxisType)` override.
    fn show_processing_panel(&self, axis_type: AxisType) {
        self.base.set_showing_setup(false);
        self.base.show_processing_panel_super(axis_type);
    }

    /// Java final `setState(ProcessState, AxisID, AbstractParallelDialog)`.
    fn set_state(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        parallel_dialog: &dyn AbstractParallelDialog,
    ) {
        self.set_state_process_state_axis_id_dialog_type(
            process_state,
            axis_id,
            parallel_dialog.get_dialog_type(),
        );
    }

    /// Java `createAxisPanelA(AxisID, AxisProgressPanel)`.
    fn create_axis_panel_a(&self, axis_id: AxisID, axis_progress_panel: Rc<AxisProgressPanel>) {
        let panel =
            TomogramProcessPanel::new(self.application_manager, axis_id, axis_progress_panel);
        *self.axis_panel_a.borrow_mut() = Some(panel);
    }

    /// Java `createAxisPanelB(AxisProgressPanel)`.
    fn create_axis_panel_b(&self, axis_progress_panel: Rc<AxisProgressPanel>) {
        let panel = TomogramProcessPanel::new(
            self.application_manager,
            AxisID::Second,
            axis_progress_panel,
        );
        *self.axis_panel_b.borrow_mut() = Some(panel);
    }

    /// Java `resetAxisPanels()`.
    fn reset_axis_panels(&self) {
        *self.axis_panel_a.borrow_mut() = None;
        *self.axis_panel_b.borrow_mut() = None;
    }

    /// Java `addAxisPanelA()`.
    fn add_axis_panel_a(&self) {
        let axis_panel_a = self.axis_panel_a.borrow().clone();
        if let (Some(scroll_a), Some(axis_panel_a)) = (self.base.get_scroll_a(), axis_panel_a) {
            scroll_a.add(&axis_panel_a.get_container());
        }
    }

    /// Java `addAxisPanelB()`.
    fn add_axis_panel_b(&self) {
        let axis_panel_b = self.axis_panel_b.borrow().clone();
        if let (Some(scroll_b), Some(axis_panel_b)) = (self.base.get_scroll_b(), axis_panel_b) {
            scroll_b.add(&axis_panel_b.get_container());
        }
    }

    /// Java `isAxisPanelANull()`.
    fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.borrow().is_none()
    }

    /// Java `isAxisPanelBNull()`.
    fn is_axis_panel_b_null(&self) -> bool {
        self.axis_panel_b.borrow().is_none()
    }

    /// Java `getAxisPanelA()`.
    fn get_axis_panel_a(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        self.axis_panel_a
            .borrow()
            .clone()
            .map(|panel| panel as Rc<dyn AxisProcessPanelVirtual>)
    }

    /// Java `getAxisPanelB()`.
    fn get_axis_panel_b(&self) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        self.axis_panel_b
            .borrow()
            .clone()
            .map(|panel| panel as Rc<dyn AxisProcessPanelVirtual>)
    }

    /// Java `hideAxisPanelA()`.
    fn hide_axis_panel_a(&self) -> bool {
        self.axis_panel_a
            .borrow()
            .clone()
            .is_some_and(|panel| panel.hide())
    }

    /// Java `hideAxisPanelB()`.
    fn hide_axis_panel_b(&self) -> bool {
        self.axis_panel_b
            .borrow()
            .clone()
            .is_some_and(|panel| panel.hide())
    }

    /// Java `showAxisPanelA()`.
    fn show_axis_panel_a(&self) {
        if let Some(panel) = self.axis_panel_a.borrow().clone() {
            panel.show();
        }
    }

    /// Java `showAxisPanelB()`.
    fn show_axis_panel_b(&self) {
        if let Some(panel) = self.axis_panel_b.borrow().clone() {
            panel.show();
        }
    }

    /// Java `showBlankProcess(AxisID)` override.  Show a blank processing
    /// panel.
    fn show_blank_process(&self, axis_id: AxisID) {
        // Java `((ApplicationManager) manager).setCurrentDialogType(null, axisID)`.
        self.application_manager
            .set_current_dialog_type(None, Some(axis_id));
        self.base.show_blank_process_super(axis_id);
    }

    /// Java `mapBaseAxisProcessPanel(AxisID)`.
    fn map_base_axis_process_panel(
        &self,
        axis_id: AxisID,
    ) -> Option<Rc<dyn AxisProcessPanelVirtual>> {
        self.map_axis(axis_id)
            .map(|panel| panel as Rc<dyn AxisProcessPanelVirtual>)
    }

    /// Java `mapAxisProgressPanel(AxisID)`.
    fn map_axis_progress_panel(&self, axis_id: AxisID) -> Option<Rc<AxisProgressPanel>> {
        Some(self.base.get_progress_panel(axis_id))
    }

    /// Java `stopProgressBar(AxisID, ProcessEndState, String)` override (calls
    /// the superclass body).
    fn stop_progress_bar_axis_id_process_end_state_string(
        &self,
        axis_id: AxisID,
        process_end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
    ) {
        self.base
            .stop_progress_bar_super(axis_id, process_end_state, status_string);
    }
}
