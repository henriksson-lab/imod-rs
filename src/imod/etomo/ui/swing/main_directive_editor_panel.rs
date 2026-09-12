//! `IMOD/Etomo/src/etomo/ui/swing/MainDirectiveEditorPanel.java`.
//!
//! Java inheritance is represented by the owned `main_panel` field.  The
//! `ScrollPanel.add` call stays an explicit native GUI boundary, while the
//! directive-editor process panel and manager stay their separate source units.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::directive_editor_process_panel::DirectiveEditorProcessPanel;
use super::main_panel::MainPanel;
use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java final `MainDirectiveEditorPanel`, including inherited `MainPanel` state.
pub struct MainDirectiveEditorPanel {
    pub main_panel: MainPanel,
    /// Java final `manager`.
    pub manager: &'static DirectiveEditorManager,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<DirectiveEditorProcessPanel>,
    /// Native `getScrollA().add(axisPanelA.getContainer())` presentation boundary.
    pub axis_panel_a_added_to_scroll: bool,
}

impl MainDirectiveEditorPanel {
    /// Java `MainDirectiveEditorPanel(DirectiveEditorManager)`.
    pub fn new(manager: &'static DirectiveEditorManager) -> Self {
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
            .expect("MainDirectiveEditorPanel.axisPanelA is null")
            .axis_process_panel
            .get_container();
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
        self.axis_panel_a = Some(DirectiveEditorProcessPanel::new(
            self.manager,
            InterfaceType::DirectiveEditor,
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

    /// Java `getDataFileFilter()`, which returns null.
    pub fn get_data_file_filter(&self) -> Option<DataFileFilter> {
        None
    }

    /// Java `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainDirectiveEditorPanel.axisPanelA is null")
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
            .expect("MainDirectiveEditorPanel.axisPanelA is null")
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
    use crate::imod::etomo::ui::swing::scroll_panel::ScrollPanel;

    struct Dialog;

    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

        fn get_dialog_type(&self) -> DialogType {
            DialogType::Tools
        }
    }

    #[test]
    fn a_axis_creation_mapping_and_scroll_addition_preserve_directive_editor_panel() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let mut panel = MainDirectiveEditorPanel::new(manager);
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        panel.create_axis_panel_a(AxisID::First, progress);
        assert!(!panel.is_axis_panel_a_null());
        assert_eq!(panel.get_axis_panel_a().unwrap().axis_id, AxisID::Only);
        assert_eq!(
            panel.get_axis_panel_a().unwrap().interface_type,
            InterfaceType::DirectiveEditor
        );
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
        panel.main_panel.scroll_a = Some(ScrollPanel::new());
        panel.add_axis_panel_a();
        assert!(panel.axis_panel_a_added_to_scroll);
    }

    #[test]
    fn only_axis_progress_is_mapped_and_b_axis_source_nulls_are_preserved() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let mut panel = MainDirectiveEditorPanel::new(manager);
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
        panel.create_axis_panel_b(AxisProgressPanel::get_instance(Some(AxisID::Only), manager));
        panel.show_axis_panel_b();
    }

    #[test]
    fn visibility_reset_empty_overrides_and_status_delegate_match_java() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let mut panel = MainDirectiveEditorPanel::new(manager);
        panel.create_axis_panel_a(
            AxisID::Only,
            AxisProgressPanel::get_instance(Some(AxisID::Only), manager),
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
    }
}
