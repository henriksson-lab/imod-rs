//! `IMOD/Etomo/src/etomo/ui/swing/MainParallelPanel.java`.
//!
//! The Java superclass relationship is represented by the owned `main_panel`
//! field.  The Swing `getScrollA().add(axisPanelA.getContainer())` call remains
//! an explicit GUI boundary; it is not substituted with a different layout
//! abstraction.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::MainPanel;
use super::parallel_process_panel::ParallelProcessPanel;
use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java final `MainParallelPanel`, including its `MainPanel` superclass state.
pub struct MainParallelPanel {
    pub main_panel: MainPanel,
    /// Java private final `manager`.
    pub manager: &'static ParallelManager,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<ParallelProcessPanel>,
    /// Native `getScrollA().add(axisPanelA.getContainer())` boundary.
    pub axis_panel_a_added_to_scroll: bool,
}

impl MainParallelPanel {
    /// `MainParallelPanel(ParallelManager)`.
    pub fn new(manager: &'static ParallelManager) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            manager,
            axis_panel_a: None,
            axis_panel_a_added_to_scroll: false,
        }
    }

    /// Java override `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        let _scroll_a = self
            .main_panel
            .get_scroll_a()
            .expect("MainPanel.scrollA is null");
        let _container = self
            .axis_panel_a
            .as_ref()
            .expect("MainParallelPanel.axisPanelA is null")
            .axis_process_panel
            .get_container();
        self.axis_panel_a_added_to_scroll = true;
    }

    /// Java override `addAxisPanelB()`, empty in the source.
    pub fn add_axis_panel_b(&mut self) {}

    /// Java override `isAxisPanelANull()`.
    pub fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.is_none()
    }

    /// Java override `isAxisPanelBNull()`.
    pub fn is_axis_panel_b_null(&self) -> bool {
        true
    }

    /// Java override `createAxisPanelA(AxisID, AxisProgressPanel)`.
    pub fn create_axis_panel_a(
        &mut self,
        _axis_id: AxisID,
        axis_progress_panel: AxisProgressPanel,
    ) {
        self.axis_panel_a = Some(ParallelProcessPanel::new(self.manager, axis_progress_panel));
    }

    /// Java override `createAxisPanelB(AxisProgressPanel)`, empty in source.
    pub fn create_axis_panel_b(&mut self, _axis_progress_panel: AxisProgressPanel) {}

    /// Java override `getAxisPanelA()`.
    pub fn get_axis_panel_a(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_a
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }

    /// Java override `getAxisPanelB()`, returning null.
    pub fn get_axis_panel_b(&mut self) -> Option<&mut AxisProcessPanel> {
        None
    }

    /// Java override `getDataFileFilter()`, returning null.
    pub fn get_data_file_filter(&self) -> Option<DataFileFilter> {
        None
    }

    /// Java override `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainParallelPanel.axisPanelA is null")
            .axis_process_panel
            .hide()
    }

    /// Java override `hideAxisPanelB()`.
    pub fn hide_axis_panel_b(&mut self) -> bool {
        true
    }

    /// Java override `mapBaseAxisProcessPanel(AxisID)`.
    pub fn map_base_axis_process_panel(
        &mut self,
        axis_id: AxisID,
    ) -> Option<&mut AxisProcessPanel> {
        if axis_id == AxisID::Second {
            return None;
        }
        self.get_axis_panel_a()
    }

    /// Java override `mapAxisProgressPanel(AxisID)`.
    pub fn map_axis_progress_panel(&mut self, axis_id: AxisID) -> Option<&mut AxisProgressPanel> {
        if axis_id == AxisID::Second {
            return None;
        }
        Some(self.main_panel.get_progress_panel(axis_id))
    }

    /// Java override `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
        self.axis_panel_a_added_to_scroll = false;
    }

    /// Java override `saveDisplayState()`, empty in source.
    pub fn save_display_state(&mut self) {}

    /// Java override `setState(ProcessState, AxisID, AbstractParallelDialog)`, empty in source.
    pub fn set_state(
        &mut self,
        _process_state: ProcessState,
        _axis_id: AxisID,
        _parallel_dialog: &dyn AbstractParallelDialog,
    ) {
    }

    /// Java override `showAxisPanelA()`.
    pub fn show_axis_panel_a(&mut self) {
        self.axis_panel_a
            .as_mut()
            .expect("MainParallelPanel.axisPanelA is null")
            .axis_process_panel
            .show();
    }

    /// Java override `showAxisPanelB()`, empty in source.
    pub fn show_axis_panel_b(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::comscript::parallel_param::ParallelParam;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::scroll_panel::ScrollPanel;

    struct Dialog;

    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

        fn get_dialog_type(&self) -> DialogType {
            DialogType::Parallel
        }
    }

    fn panel() -> MainParallelPanel {
        MainParallelPanel::new(ParallelManager::new())
    }

    #[test]
    fn axis_a_creation_keeps_parallel_manager_and_process_panel() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::First, progress);
        assert!(!panel.is_axis_panel_a_null());
        let axis_panel = panel.get_axis_panel_a().unwrap();
        assert_eq!(axis_panel.axis_id, AxisID::Only);
        assert!(axis_panel.popup_chunk_warnings);
        assert!(axis_panel.runnable_parallel);
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
    }

    #[test]
    fn source_scroll_visibility_and_reset_calls_are_delegated_to_axis_a() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::Only, progress);
        panel.main_panel.scroll_a = Some(ScrollPanel::new());
        panel.add_axis_panel_a();
        assert!(panel.axis_panel_a_added_to_scroll);
        assert!(panel.hide_axis_panel_a());
        assert!(!panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.show_axis_panel_a();
        assert!(panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.reset_axis_panels();
        assert!(panel.is_axis_panel_a_null());
        assert!(!panel.axis_panel_a_added_to_scroll);
    }

    #[test]
    fn null_b_axis_filter_and_empty_source_overrides_are_preserved() {
        let mut panel = panel();
        assert!(panel.is_axis_panel_b_null());
        assert!(panel.get_axis_panel_b().is_none());
        assert!(panel.get_data_file_filter().is_none());
        assert!(panel.hide_axis_panel_b());
        assert!(panel.map_axis_progress_panel(AxisID::Second).is_none());
        assert_eq!(
            panel
                .map_axis_progress_panel(AxisID::First)
                .unwrap()
                .axis_id,
            AxisID::First
        );
        panel.set_state(ProcessState::Complete, AxisID::Only, &Dialog);
        panel.save_display_state();
        panel.add_axis_panel_b();
        panel.show_axis_panel_b();
    }
}
