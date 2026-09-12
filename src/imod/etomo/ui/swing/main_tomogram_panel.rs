//! `IMOD/Etomo/src/etomo/ui/swing/MainTomogramPanel.java`.
//!
//! Java inheritance is represented by the owned `main_panel` superclass
//! state.  `ApplicationManager`, `UIHarness`, `SetupDialogExpert`, and
//! `ProcessTrack` remain explicit direct boundaries, rather than being
//! replaced with alternate GUI or process-management implementations.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::MainPanel;
use super::tomogram_process_panel::TomogramProcessPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::etomo_file_filter::EtomoFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Direct `ProcessTrack` calls made by `updateAllProcessingStates`.
pub trait MainTomogramPanelProcessTrack {
    fn get_pre_processing_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_coarse_alignment_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_fiducial_model_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_fine_alignment_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_tomogram_positioning_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_final_aligned_stack_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_tomogram_generation_state(&self, axis_id: AxisID) -> ProcessState;
    fn get_tomogram_combination_state(&self) -> ProcessState;
    fn get_post_processing_state(&self) -> ProcessState;
    fn get_clean_up_state(&self) -> ProcessState;
}

/// Direct `UIHarness.pack` calls in this source unit.
pub trait MainTomogramPanelUiHarness {
    fn pack(&mut self, manager: &'static dyn BaseManager);
    fn pack_force(&mut self, force: bool, manager: &'static dyn BaseManager);
}

/// Direct `SetupDialogExpert.getContainer` call in `openSetupPanel`.
pub trait MainTomogramPanelSetupDialogExpert {
    fn get_container(&self) -> String;
}

/// Direct `ApplicationManager.setCurrentDialogType` call in
/// `showBlankProcess`.
pub trait MainTomogramPanelApplicationManager {
    fn set_current_dialog_type(&mut self, dialog_type: Option<DialogType>, axis_id: AxisID);
}

/// Java `MainTomogramPanel`, including its `MainPanel` superclass state.
pub struct MainTomogramPanel {
    pub main_panel: MainPanel,
    pub axis_panel_a: Option<TomogramProcessPanel>,
    pub axis_panel_b: Option<TomogramProcessPanel>,
    /// Native `panelCenter` has heterogeneous Swing children.  This preserves
    /// the source child placed there by `openSetupPanel`.
    pub panel_center_setup_container: Option<String>,
    /// Java `revalidate()` presentation-boundary calls.
    pub revalidate_count: u64,
    /// `compactDisplay` is read from `EtomoDirector.INSTANCE` by the child
    /// constructor; it remains an explicit global-boundary input here.
    pub compact_display: bool,
}

impl MainTomogramPanel {
    /// `MainTomogramPanel(ApplicationManager)`.
    pub fn new(manager: &'static dyn BaseManager, compact_display: bool) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            axis_panel_a: None,
            axis_panel_b: None,
            panel_center_setup_container: None,
            revalidate_count: 0,
            compact_display,
        }
    }

    /// Java override `getDataFileFilter()`.
    pub fn get_data_file_filter(&self) -> EtomoFileFilter {
        EtomoFileFilter
    }

    /// Java override `saveDisplayState()`.
    pub fn save_display_state(&mut self) {
        if let Some(axis_panel_a) = &mut self.axis_panel_a {
            axis_panel_a.axis_process_panel.save_display_state();
        }
        if let Some(axis_panel_b) = &mut self.axis_panel_b {
            axis_panel_b.axis_process_panel.save_display_state();
        }
    }

    /// Java override `showAxisA()`.
    pub fn show_axis_a<U: MainTomogramPanelUiHarness>(&mut self, ui_harness: &mut U) {
        if self.main_panel.is_showing_setup() || self.main_panel.axis_type == AxisType::SingleAxis {
            ui_harness.pack_force(true, self.main_panel.manager);
        } else if let Some(axis_panel_a) = &mut self.axis_panel_a {
            axis_panel_a.show_axis_a();
            self.main_panel.show_axis_a();
        }
    }

    /// Java override `showAxisB()`.
    pub fn show_axis_b(&mut self) {
        self.axis_panel_b
            .as_mut()
            .expect("MainTomogramPanel.showAxisB requires axisPanelB")
            .show_axis_b();
        self.main_panel.show_axis_b();
    }

    /// Java override `showBothAxis()`.
    pub fn show_both_axis(&mut self) -> Option<i32> {
        self.axis_panel_b
            .as_mut()
            .expect("MainTomogramPanel.showBothAxis requires axisPanelB")
            .show_axis_b();
        self.main_panel.show_both_axis()
    }

    /// `updateAllProcessingStates(ProcessTrack)`.
    pub fn update_all_processing_states<T: MainTomogramPanelProcessTrack>(
        &mut self,
        process_track: &T,
    ) {
        if self.axis_panel_a.is_none() {
            return;
        }
        let axis_panel_a = self.axis_panel_a.as_mut().unwrap();
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
        if self.main_panel.manager.is_dual_axis() {
            let axis_panel_b = self.axis_panel_b.as_mut().expect(
                "MainTomogramPanel.updateAllProcessingStates requires axisPanelB for dual axis",
            );
            axis_panel_b.set_pre_proc_state(process_track.get_pre_processing_state(AxisID::Second));
            axis_panel_b
                .set_coarse_align_state(process_track.get_coarse_alignment_state(AxisID::Second));
            axis_panel_b
                .set_fiducial_model_state(process_track.get_fiducial_model_state(AxisID::Second));
            axis_panel_b
                .set_fine_alignment_state(process_track.get_fine_alignment_state(AxisID::Second));
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
        let axis_panel_a = self.axis_panel_a.as_mut().unwrap();
        axis_panel_a.set_post_processing_state(process_track.get_post_processing_state());
        axis_panel_a.set_clean_up_state(process_track.get_clean_up_state());
    }

    /// Java override `showProcessingPanel(AxisType)` plus its direct
    /// `MainPanel.showProcessingPanel` superclass call.
    pub fn show_processing_panel(&mut self, axis_type: AxisType) {
        self.main_panel.set_showing_setup(false);
        self.reset_axis_panels();
        self.main_panel.axis_type = axis_type;
        self.main_panel.panel_center.clear();
        if axis_type == AxisType::SingleAxis {
            let axis_id = AxisID::Only;
            self.create_axis_panel_a(
                axis_id,
                AxisProgressPanel::get_instance(Some(axis_id), self.main_panel.manager),
            );
            self.main_panel.scroll_a = Some(super::scroll_panel::ScrollPanel::new());
            self.main_panel.scroll_pane_a = Some(0);
            self.add_axis_panel_a();
            self.main_panel.panel_center.push(axis_id);
        } else {
            let axis_id = AxisID::First;
            self.create_axis_panel_a(
                axis_id,
                AxisProgressPanel::get_instance(Some(axis_id), self.main_panel.manager),
            );
            self.main_panel.scroll_a = Some(super::scroll_panel::ScrollPanel::new());
            self.main_panel.scroll_pane_a = Some(0);
            self.add_axis_panel_a();
            let axis_id = AxisID::Second;
            self.create_axis_panel_b(AxisProgressPanel::get_instance(
                Some(axis_id),
                self.main_panel.manager,
            ));
            self.main_panel.scroll_b = Some(super::scroll_panel::ScrollPanel::new());
            self.main_panel.scroll_pane_b = Some(0);
            self.add_axis_panel_b();
            self.main_panel.show_axis_a();
        }
    }

    /// `openSetupPanel(SetupDialogExpert)`.
    pub fn open_setup_panel<
        U: MainTomogramPanelUiHarness,
        S: MainTomogramPanelSetupDialogExpert,
    >(
        &mut self,
        setup_dialog_expert: &S,
        ui_harness: &mut U,
    ) {
        self.main_panel.set_showing_setup(true);
        self.main_panel.panel_center.clear();
        self.panel_center_setup_container = Some(setup_dialog_expert.get_container());
        self.revalidate_count += 1;
        ui_harness.pack(self.main_panel.manager);
    }

    /// `selectButton(AxisID, String)`.
    pub fn select_button(&mut self, axis_id: AxisID, name: &str) {
        self.map_axis(axis_id).select_button(name);
    }

    /// `setState(ProcessState, AxisID, AbstractParallelDialog)` after the
    /// dialog's direct `getDialogType` boundary has supplied `dialog_type`.
    pub fn set_state_parallel_dialog(
        &mut self,
        process_state: ProcessState,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        self.set_state(process_state, axis_id, dialog_type);
    }

    /// `setState(ProcessState, AxisID, DialogType)`.
    pub fn set_state(
        &mut self,
        process_state: ProcessState,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        match dialog_type {
            DialogType::CleanUp => self.set_clean_up_state(process_state),
            DialogType::CoarseAlignment => self.set_coarse_align_state(process_state, axis_id),
            DialogType::FiducialModel => self.set_fiducial_model_state(process_state, axis_id),
            DialogType::FineAlignment => self.set_fine_alignment_state(process_state, axis_id),
            DialogType::PostProcessing => self.set_post_processing_state(process_state),
            DialogType::PreProcessing => self.set_pre_processing_state(process_state, axis_id),
            DialogType::TomogramCombination => self.set_tomogram_combination_state(process_state),
            DialogType::FinalAlignedStack => {
                self.set_final_aligned_stack_state(process_state, axis_id)
            }
            DialogType::TomogramGeneration => {
                self.set_tomogram_generation_state(process_state, axis_id)
            }
            DialogType::TomogramPositioning => {
                self.set_tomogram_positioning_state(process_state, axis_id)
            }
            _ => {}
        }
    }

    /// `setPreProcessingState(ProcessState, AxisID)`.
    pub fn set_pre_processing_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_pre_proc_state(state);
    }
    /// `setCoarseAlignState(ProcessState, AxisID)`.
    pub fn set_coarse_align_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_coarse_align_state(state);
    }
    /// `setFiducialModelState(ProcessState, AxisID)`.
    pub fn set_fiducial_model_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_fiducial_model_state(state);
    }
    /// `setFineAlignmentState(ProcessState, AxisID)`.
    pub fn set_fine_alignment_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_fine_alignment_state(state);
    }
    /// `setTomogramPositioningState(ProcessState, AxisID)`.
    pub fn set_tomogram_positioning_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_tomogram_positioning_state(state);
    }
    /// `setFinalAlignedStackState(ProcessState, AxisID)`.
    pub fn set_final_aligned_stack_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_final_aligned_stack_state(state);
    }
    /// `setTomogramGenerationState(ProcessState, AxisID)`.
    pub fn set_tomogram_generation_state(&mut self, state: ProcessState, axis_id: AxisID) {
        self.map_axis(axis_id).set_tomogram_generation_state(state);
    }
    /// `setTomogramCombinationState(ProcessState)`.
    pub fn set_tomogram_combination_state(&mut self, state: ProcessState) {
        self.axis_panel_a
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelA")
            .set_tomogram_combination_state(state);
    }
    /// `setPostProcessingState(ProcessState)`.
    pub fn set_post_processing_state(&mut self, state: ProcessState) {
        self.axis_panel_a
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelA")
            .set_post_processing_state(state);
    }
    /// `setCleanUpState(ProcessState)`.
    pub fn set_clean_up_state(&mut self, state: ProcessState) {
        self.axis_panel_a
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelA")
            .set_clean_up_state(state);
    }

    /// `getCPUsSelectedInt(AxisID, boolean)`.
    pub fn get_cpus_selected_int(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<i32, String> {
        if axis_id == AxisID::Second {
            return self.axis_panel_b.as_ref().map_or(Ok(0), |panel| {
                panel
                    .axis_process_panel
                    .get_cpus_selected_int(do_validation)
            });
        }
        self.axis_panel_a.as_ref().map_or(Ok(0), |panel| {
            panel
                .axis_process_panel
                .get_cpus_selected_int(do_validation)
        })
    }

    /// Java override `createAxisPanelA(AxisID, AxisProgressPanel)`.
    pub fn create_axis_panel_a(&mut self, axis_id: AxisID, axis_progress_panel: AxisProgressPanel) {
        self.axis_panel_a = Some(TomogramProcessPanel::new(
            self.main_panel.manager,
            axis_id,
            axis_progress_panel,
            self.compact_display,
        ));
    }
    /// Java override `createAxisPanelB(AxisProgressPanel)`.
    pub fn create_axis_panel_b(&mut self, axis_progress_panel: AxisProgressPanel) {
        self.axis_panel_b = Some(TomogramProcessPanel::new(
            self.main_panel.manager,
            AxisID::Second,
            axis_progress_panel,
            self.compact_display,
        ));
    }
    /// Java override `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
        self.axis_panel_b = None;
    }
    /// Java override `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        let _ = self
            .axis_panel_a
            .as_ref()
            .expect("MainTomogramPanel requires axisPanelA")
            .axis_process_panel
            .get_container();
    }
    /// Java override `addAxisPanelB()`.
    pub fn add_axis_panel_b(&mut self) {
        let _ = self
            .axis_panel_b
            .as_ref()
            .expect("MainTomogramPanel requires axisPanelB")
            .axis_process_panel
            .get_container();
    }
    /// Java override `isAxisPanelANull()`.
    pub fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.is_none()
    }
    /// Java override `isAxisPanelBNull()`.
    pub fn is_axis_panel_b_null(&self) -> bool {
        self.axis_panel_b.is_none()
    }
    /// Java override `getAxisPanelA()`.
    pub fn get_axis_panel_a(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_a
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }
    /// Java override `getAxisPanelB()`.
    pub fn get_axis_panel_b(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_b
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }
    /// Java override `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelA")
            .axis_process_panel
            .hide()
    }
    /// Java override `hideAxisPanelB()`.
    pub fn hide_axis_panel_b(&mut self) -> bool {
        self.axis_panel_b
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelB")
            .axis_process_panel
            .hide()
    }
    /// Java override `showAxisPanelA()`.
    pub fn show_axis_panel_a(&mut self) {
        self.axis_panel_a
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelA")
            .axis_process_panel
            .show();
    }
    /// Java override `showAxisPanelB()`.
    pub fn show_axis_panel_b(&mut self) {
        self.axis_panel_b
            .as_mut()
            .expect("MainTomogramPanel requires axisPanelB")
            .axis_process_panel
            .show();
    }

    /// Java override `showBlankProcess(AxisID)`.
    pub fn show_blank_process<M: MainTomogramPanelApplicationManager>(
        &mut self,
        application_manager: &mut M,
        axis_id: AxisID,
    ) {
        application_manager.set_current_dialog_type(None, axis_id);
        self.map_base_axis_process_panel(axis_id)
            .erase_dialog_panel();
    }

    /// Java private `mapAxis(AxisID)`.
    fn map_axis(&mut self, axis_id: AxisID) -> &mut TomogramProcessPanel {
        if axis_id == AxisID::Second {
            self.axis_panel_b
                .as_mut()
                .expect("MainTomogramPanel.mapAxis requires axisPanelB")
        } else {
            self.axis_panel_a
                .as_mut()
                .expect("MainTomogramPanel.mapAxis requires axisPanelA")
        }
    }
    /// Java override `mapBaseAxisProcessPanel(AxisID)`.
    pub fn map_base_axis_process_panel(&mut self, axis_id: AxisID) -> &mut AxisProcessPanel {
        &mut self.map_axis(axis_id).axis_process_panel
    }
    /// Java override `mapAxisProgressPanel(AxisID)`.
    pub fn map_axis_progress_panel(&mut self, axis_id: AxisID) -> &mut AxisProgressPanel {
        &mut self
            .map_axis(axis_id)
            .axis_process_panel
            .axis_progress_panel
    }
    /// Java override `stopProgressBar(AxisID, ProcessEndState, String)`.
    pub fn stop_progress_bar(
        &mut self,
        axis_id: AxisID,
        process_end_state: &str,
        status_string: Option<&str>,
    ) {
        self.main_panel
            .stop_progress_bar(axis_id, process_end_state, status_string);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    struct Harness {
        packs: Vec<bool>,
    }
    impl MainTomogramPanelUiHarness for Harness {
        fn pack(&mut self, _: &'static dyn BaseManager) {
            self.packs.push(false);
        }
        fn pack_force(&mut self, force: bool, _: &'static dyn BaseManager) {
            self.packs.push(force);
        }
    }
    struct Setup;
    impl MainTomogramPanelSetupDialogExpert for Setup {
        fn get_container(&self) -> String {
            "setup-container".into()
        }
    }
    struct Application {
        current: Option<(Option<DialogType>, AxisID)>,
    }
    impl MainTomogramPanelApplicationManager for Application {
        fn set_current_dialog_type(&mut self, dialog_type: Option<DialogType>, axis_id: AxisID) {
            self.current = Some((dialog_type, axis_id));
        }
    }

    fn panel() -> MainTomogramPanel {
        MainTomogramPanel::new(DirectiveEditorManager::new(None, None, None, None), false)
    }

    #[test]
    fn processing_panels_follow_single_and_dual_axis_source_construction() {
        let mut panel = panel();
        panel.show_processing_panel(AxisType::SingleAxis);
        assert_eq!(
            panel
                .axis_panel_a
                .as_ref()
                .unwrap()
                .axis_process_panel
                .axis_id,
            AxisID::Only
        );
        assert!(panel.axis_panel_b.is_none());
        panel.show_processing_panel(AxisType::DualAxis);
        assert_eq!(
            panel
                .axis_panel_a
                .as_ref()
                .unwrap()
                .axis_process_panel
                .axis_id,
            AxisID::First
        );
        assert_eq!(
            panel
                .axis_panel_b
                .as_ref()
                .unwrap()
                .axis_process_panel
                .axis_id,
            AxisID::Second
        );
    }

    #[test]
    fn setup_and_blank_process_preserve_manager_and_presentation_boundaries() {
        let mut panel = panel();
        let mut harness = Harness { packs: vec![] };
        panel.open_setup_panel(&Setup, &mut harness);
        assert!(panel.main_panel.is_showing_setup());
        assert_eq!(
            panel.panel_center_setup_container.as_deref(),
            Some("setup-container")
        );
        assert_eq!(harness.packs, vec![false]);
        panel.show_processing_panel(AxisType::SingleAxis);
        let mut application = Application {
            current: Some((Some(DialogType::CleanUp), AxisID::Only)),
        };
        panel.show_blank_process(&mut application, AxisID::Only);
        assert_eq!(application.current, Some((None, AxisID::Only)));
        assert!(
            panel
                .axis_panel_a
                .unwrap()
                .axis_process_panel
                .panel_dialog
                .is_none()
        );
    }

    #[test]
    fn state_dispatches_to_the_matching_tomogram_control() {
        let mut panel = panel();
        panel.show_processing_panel(AxisType::SingleAxis);
        panel.set_state(
            ProcessState::Complete,
            AxisID::Only,
            DialogType::FineAlignment,
        );
        assert_eq!(
            panel
                .axis_panel_a
                .unwrap()
                .proc_ctl_fine_alignment
                .selected_state,
            2
        );
    }
}
