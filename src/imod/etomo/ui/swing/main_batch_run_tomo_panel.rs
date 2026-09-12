//! `IMOD/Etomo/src/etomo/ui/swing/MainBatchRunTomoPanel.java`.
//!
//! The Java `MainPanel` superclass is represented by the owned `main_panel`
//! field.  `ScrollPanel.add` remains an explicit GUI boundary: Rust records
//! that the source container was supplied to that parent without substituting
//! a new widget/layout system.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::batch_run_tomo_process_panel::BatchRunTomoProcessPanel;
use super::main_panel::MainPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java final `MainBatchRunTomoPanel`, including inherited `MainPanel` state.
pub struct MainBatchRunTomoPanel {
    pub main_panel: MainPanel,
    /// Java final `manager`.
    pub manager: &'static dyn BaseManager,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<BatchRunTomoProcessPanel>,
    /// Native `getScrollA().add(axisPanelA.getContainer())` boundary.
    pub axis_panel_a_added_to_scroll: bool,
}

impl MainBatchRunTomoPanel {
    /// Java `MainBatchRunTomoPanel(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            manager,
            axis_panel_a: None,
            axis_panel_a_added_to_scroll: false,
        }
    }

    /// Java `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        let _scroll_a = self
            .main_panel
            .get_scroll_a()
            .expect("MainPanel.scrollA is null");
        let _container = self
            .axis_panel_a
            .as_ref()
            .expect("MainBatchRunTomoPanel.axisPanelA is null")
            .axis_process_panel
            .get_container();
        self.axis_panel_a_added_to_scroll = true;
    }

    /// Java `addAxisPanelB()`, whose body is empty.
    pub fn add_axis_panel_b(&mut self) {}

    /// Java `createAxisPanelA(AxisID, AxisProgressPanel)`.
    pub fn create_axis_panel_a(
        &mut self,
        _axis_id: AxisID,
        axis_progress_panel: AxisProgressPanel,
    ) {
        self.axis_panel_a = Some(BatchRunTomoProcessPanel::new(
            self.manager,
            InterfaceType::BatchRunTomo,
            axis_progress_panel,
        ));
    }

    /// Java `setState(ProcessState, AxisID, AbstractParallelDialog)`, empty in source.
    pub fn set_state(
        &mut self,
        _process_state: ProcessState,
        _axis_id: AxisID,
        _batch_run_tomo_dialog: &dyn AbstractParallelDialog,
    ) {
    }

    /// Java `createAxisPanelB(AxisProgressPanel)`, whose body is empty.
    pub fn create_axis_panel_b(&mut self, _axis_progress_panel: AxisProgressPanel) {}

    /// Java `getAxisPanelA()`.
    pub fn get_axis_panel_a(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_a
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }

    /// Java `getAxisPanelB()`, returning null.
    pub fn get_axis_panel_b(&mut self) -> Option<&mut AxisProcessPanel> {
        None
    }

    /// Java `getDataFileFilter()`, returning null.
    pub fn get_data_file_filter(&self) -> Option<DataFileFilter> {
        None
    }

    /// Java `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainBatchRunTomoPanel.axisPanelA is null")
            .axis_process_panel
            .hide()
    }

    /// Java `hideAxisPanelB()`.
    pub fn hide_axis_panel_b(&mut self) -> bool {
        true
    }

    /// Java `isAxisPanelANull()`.
    pub fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.is_none()
    }

    /// Java `isAxisPanelBNull()`.
    pub fn is_axis_panel_b_null(&self) -> bool {
        true
    }

    /// Java `mapBaseAxisProcessPanel(AxisID)`.
    pub fn map_base_axis_process_panel(
        &mut self,
        axis_id: AxisID,
    ) -> Option<&mut AxisProcessPanel> {
        if axis_id == AxisID::Second {
            return None;
        }
        self.get_axis_panel_a()
    }

    /// Java `mapAxisProgressPanel(AxisID)`.
    pub fn map_axis_progress_panel(&mut self, axis_id: AxisID) -> Option<&mut AxisProgressPanel> {
        if axis_id == AxisID::Second {
            return None;
        }
        Some(self.main_panel.get_progress_panel(axis_id))
    }

    /// Java `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
        self.axis_panel_a_added_to_scroll = false;
    }

    /// Java `saveDisplayState()`, whose body is empty.
    pub fn save_display_state(&mut self) {}

    /// Java `showAxisPanelA()`.
    pub fn show_axis_panel_a(&mut self) {
        self.axis_panel_a
            .as_mut()
            .expect("MainBatchRunTomoPanel.axisPanelA is null")
            .axis_process_panel
            .show();
    }

    /// Java `showAxisPanelB()`, whose body is empty.
    pub fn show_axis_panel_b(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
    use crate::imod::etomo::comscript::parallel_param::ParallelParam;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::scroll_panel::ScrollPanel;

    struct Dialog;

    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

        fn get_dialog_type(&self) -> DialogType {
            DialogType::BatchRunTomo
        }
    }

    fn panel() -> MainBatchRunTomoPanel {
        MainBatchRunTomoPanel::new(BatchRunTomoManager::new())
    }

    #[test]
    fn axis_a_creation_maps_batch_run_tomo_process_panel_and_scroll_container() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::First, progress);
        assert!(!panel.is_axis_panel_a_null());
        let axis_panel = panel.get_axis_panel_a().unwrap();
        assert_eq!(axis_panel.axis_id, AxisID::Only);
        assert_eq!(axis_panel.interface_type, InterfaceType::BatchRunTomo);
        assert!(axis_panel.popup_chunk_warnings);
        assert!(!axis_panel.runnable_parallel);
        assert!(axis_panel.alt_parallel_loc);
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());

        panel.main_panel.scroll_a = Some(ScrollPanel::new());
        panel.add_axis_panel_a();
        assert!(panel.axis_panel_a_added_to_scroll);
    }

    #[test]
    fn only_axis_progress_and_null_b_axis_overrides_match_source() {
        let mut panel = panel();
        assert_eq!(
            panel
                .map_axis_progress_panel(AxisID::First)
                .unwrap()
                .axis_id,
            AxisID::First
        );
        assert!(panel.map_axis_progress_panel(AxisID::Second).is_none());
        assert!(panel.is_axis_panel_b_null());
        assert!(panel.get_axis_panel_b().is_none());
        assert!(panel.get_data_file_filter().is_none());
        assert!(panel.hide_axis_panel_b());
        panel.add_axis_panel_b();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_b(progress);
        panel.show_axis_panel_b();
    }

    #[test]
    fn visibility_reset_and_empty_source_overrides_are_preserved() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::Only, progress);
        assert!(panel.hide_axis_panel_a());
        assert!(!panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.show_axis_panel_a();
        assert!(panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.set_state(ProcessState::Complete, AxisID::Only, &Dialog);
        panel.save_display_state();
        panel.reset_axis_panels();
        assert!(panel.is_axis_panel_a_null());
        assert!(!panel.axis_panel_a_added_to_scroll);
    }
}
