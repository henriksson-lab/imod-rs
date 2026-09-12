//! `IMOD/Etomo/src/etomo/ui/swing/MainToolsPanel.java`.
//!
//! Java inheritance is represented by the owned `main_panel` field.  The one
//! Swing `ScrollPanel.add` operation remains an explicit presentation boundary:
//! its source container is retained in `scroll_a_components` without inventing
//! a replacement GUI layout layer.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::MainPanel;
use super::tools_process_panel::ToolsProcessPanel;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java final `MainToolsPanel`, including its inherited `MainPanel` state.
pub struct MainToolsPanel {
    pub main_panel: MainPanel,
    /// Java final `manager`.
    pub manager: &'static ToolsManager,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<ToolsProcessPanel>,
    /// Native `getScrollA().add(axisPanelA.getContainer())` boundary.
    pub scroll_a_components: Vec<bool>,
}

impl MainToolsPanel {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java `MainToolsPanel(ToolsManager)`.
    pub fn new(manager: &'static ToolsManager) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            manager,
            axis_panel_a: None,
            scroll_a_components: Vec::new(),
        }
    }

    /// Java `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        let _scroll_a = self
            .main_panel
            .get_scroll_a()
            .expect("MainPanel.scrollA is null");
        self.scroll_a_components.push(
            self.axis_panel_a
                .as_ref()
                .expect("MainToolsPanel.axisPanelA is null")
                .axis_process_panel
                .get_container(),
        );
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
        self.axis_panel_a = Some(ToolsProcessPanel::new(self.manager, axis_progress_panel));
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

    /// Java `getDataFileFilter()`, which returns null.
    pub fn get_data_file_filter(&self) -> Option<DataFileFilter> {
        None
    }

    /// Java `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainToolsPanel.axisPanelA is null")
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
        Some(self.main_panel.get_progress_panel(axis_id))
    }

    /// Java `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
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
            .expect("MainToolsPanel.axisPanelA is null")
            .axis_process_panel
            .show();
    }

    /// Java `showAxisPanelB()`, whose body is empty.
    pub fn show_axis_panel_b(&mut self) {}

    /// Java `setStatusBarText(String, int)`.
    pub fn set_status_bar_text(&mut self, directory: Option<&str>, max_title_length: i32) {
        self.main_panel
            .set_status_bar_text_to_directory(directory, max_title_length as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::comscript::parallel_param::ParallelParam;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::etomo_menu::ToolType;
    use crate::imod::etomo::ui::swing::scroll_panel::ScrollPanel;

    struct Dialog;
    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

        fn get_dialog_type(&self) -> DialogType {
            DialogType::Tools
        }
    }

    fn panel() -> MainToolsPanel {
        MainToolsPanel::new(ToolsManager::new(ToolType::FlattenVolume))
    }

    #[test]
    fn a_axis_creation_mapping_and_scroll_addition_preserve_tools_panel() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::First, progress);
        assert!(!panel.is_axis_panel_a_null());
        assert_eq!(panel.get_axis_panel_a().unwrap().axis_id, AxisID::Only);
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
        panel.main_panel.scroll_a = Some(ScrollPanel::new());
        panel.add_axis_panel_a();
        assert_eq!(panel.scroll_a_components, vec![true]);
    }

    #[test]
    fn only_axis_progress_is_lazy_and_b_axis_stays_null_or_empty() {
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
        panel.create_axis_panel_b(AxisProgressPanel::get_instance(
            Some(AxisID::Only),
            panel.manager,
        ));
        panel.show_axis_panel_b();
    }

    #[test]
    fn visibility_reset_empty_overrides_and_status_delegate_match_java() {
        let mut panel = panel();
        panel.create_axis_panel_a(
            AxisID::Only,
            AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager),
        );
        assert!(panel.hide_axis_panel_a());
        assert!(!panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.show_axis_panel_a();
        assert!(panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.set_status_bar_text(Some("/a/very-long-directory"), 8);
        assert_eq!(panel.main_panel.get_status_bar_text(), "...irectory");
        panel.set_status_bar_text(None, 8);
        assert_eq!(panel.main_panel.get_status_bar_text(), "");
        panel.set_state(ProcessState::Complete, AxisID::Only, &Dialog);
        panel.save_display_state();
        panel.reset_axis_panels();
        assert!(panel.is_axis_panel_a_null());
        assert_eq!(MainToolsPanel::RCSID, "$Id$");
    }
}
