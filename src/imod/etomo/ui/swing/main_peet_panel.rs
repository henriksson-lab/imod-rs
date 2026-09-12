//! `IMOD/Etomo/src/etomo/ui/swing/MainPeetPanel.java`.
//!
//! The `JPanel.add` operation in `addAxisPanelA` remains an explicit native
//! GUI boundary: the source-shaped boolean records that the Peet process-panel
//! container was handed to its `ScrollPanel` parent.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::MainPanel;
use super::peet_process_panel::PeetProcessPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::peet_file_filter::PeetFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java final `MainPeetPanel`, including the inherited `MainPanel` state.
pub struct MainPeetPanel {
    pub main_panel: MainPanel,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<PeetProcessPanel>,
    /// Native JPanel/ScrollPanel add boundary used by `addAxisPanelA`.
    pub axis_panel_a_added_to_scroll: bool,
}

impl MainPeetPanel {
    /// Java `MainPeetPanel(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            axis_panel_a: None,
            axis_panel_a_added_to_scroll: false,
        }
    }

    /// Java `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        // `getScrollA().add(axisPanelA.getContainer())`: construction of the
        // actual native ScrollPanel lives in MainPanel/UI harness.  Preserve
        // both Java dereferences and record the native add boundary.
        let _scroll_a = self
            .main_panel
            .get_scroll_a()
            .expect("MainPanel.scrollA is null");
        let axis_panel_a = self
            .axis_panel_a
            .as_ref()
            .expect("MainPeetPanel.axisPanelA is null");
        let _container = axis_panel_a.axis_process_panel.get_container();
        self.axis_panel_a_added_to_scroll = true;
    }

    /// Java `addAxisPanelB()`, whose body is empty.
    pub fn add_axis_panel_b(&mut self) {}

    /// Java `isAxisPanelANull()`.
    pub fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.is_none()
    }

    /// Java `isAxisPanelBNull()`.
    pub fn is_axis_panel_b_null(&self) -> bool {
        true
    }

    /// Java `createAxisPanelA(AxisID, AxisProgressPanel)`.
    pub fn create_axis_panel_a(
        &mut self,
        _axis_id: AxisID,
        axis_progress_panel: AxisProgressPanel,
    ) {
        self.axis_panel_a = Some(PeetProcessPanel::new(
            self.main_panel.manager,
            axis_progress_panel,
        ));
    }

    /// Java `createAxisPanelB(AxisProgressPanel)`, whose body is empty.
    pub fn create_axis_panel_b(&mut self, _axis_progress_panel: AxisProgressPanel) {}

    /// Java `getAxisPanelA()`.
    pub fn get_axis_panel_a(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_a
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }

    /// Java `getAxisPanelB()`, which returns null.
    pub fn get_axis_panel_b(&mut self) -> Option<&mut AxisProcessPanel> {
        None
    }

    /// Java `getDataFileFilter()`.
    pub fn get_data_file_filter(&self) -> PeetFileFilter {
        PeetFileFilter::new()
    }

    /// Java `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainPeetPanel.axisPanelA is null")
            .axis_process_panel
            .hide()
    }

    /// Java `hideAxisPanelB()`.
    pub fn hide_axis_panel_b(&mut self) -> bool {
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
        if self.main_panel.axis_progress_panel_a.is_none() {
            self.main_panel.axis_progress_panel_a = Some(AxisProgressPanel::get_instance(
                Some(axis_id),
                self.main_panel.manager,
            ));
        }
        self.main_panel.axis_progress_panel_a.as_mut()
    }

    /// Java `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
        self.axis_panel_a_added_to_scroll = false;
    }

    /// Java `saveDisplayState()`, whose body is empty.
    pub fn save_display_state(&mut self) {}

    /// Java `setState(ProcessState, AxisID, AbstractParallelDialog)`, whose body is empty.
    pub fn set_state(
        &mut self,
        _process_state: ProcessState,
        _axis_id: AxisID,
        _parallel_dialog: &dyn AbstractParallelDialog,
    ) {
    }

    /// Java `showAxisPanelA()`.
    pub fn show_axis_panel_a(&mut self) {
        self.axis_panel_a
            .as_mut()
            .expect("MainPeetPanel.axisPanelA is null")
            .axis_process_panel
            .show();
    }

    /// Java `showAxisPanelB()`, whose body is empty.
    pub fn show_axis_panel_b(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::comscript::parallel_param::ParallelParam;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    use crate::imod::etomo::r#type::dialog_type::DialogType;

    struct Dialog;
    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}
        fn get_dialog_type(&self) -> DialogType {
            DialogType::Peet
        }
    }

    fn panel() -> MainPeetPanel {
        MainPeetPanel::new(DirectiveEditorManager::new(None, None, None, None))
    }

    #[test]
    fn a_axis_creation_and_mapping_keep_the_peet_process_panel() {
        let mut panel = panel();
        let progress =
            AxisProgressPanel::get_instance(Some(AxisID::Only), panel.main_panel.manager);
        panel.create_axis_panel_a(AxisID::First, progress);
        assert!(!panel.is_axis_panel_a_null());
        assert_eq!(panel.get_axis_panel_a().unwrap().axis_id, AxisID::Only);
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
        panel.main_panel.scroll_a = Some(super::super::scroll_panel::ScrollPanel::new());
        panel.add_axis_panel_a();
        assert!(panel.axis_panel_a_added_to_scroll);
    }

    #[test]
    fn b_axis_methods_remain_empty_or_null_and_filter_is_peet_specific() {
        let mut panel = panel();
        assert!(panel.is_axis_panel_b_null());
        assert!(panel.get_axis_panel_b().is_none());
        assert!(panel.map_axis_progress_panel(AxisID::Second).is_none());
        assert!(panel.hide_axis_panel_b());
        assert!(
            panel
                .get_data_file_filter()
                .accept(std::path::Path::new("x.epe"))
        );
    }

    #[test]
    fn source_empty_set_state_and_display_methods_do_not_change_panel() {
        let mut panel = panel();
        let dialog = Dialog;
        panel.set_state(ProcessState::Complete, AxisID::Only, &dialog);
        panel.save_display_state();
        panel.add_axis_panel_b();
        panel.show_axis_panel_b();
        assert!(panel.is_axis_panel_a_null());
    }
}
