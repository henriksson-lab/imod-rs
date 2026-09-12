//! `IMOD/Etomo/src/etomo/ui/swing/MainSerialSectionsPanel.java`.
//!
//! Java inheritance is represented by the owned `main_panel` field. Swing
//! `JScrollPane.add` stays an explicit presentation boundary in
//! `scroll_a_components`, rather than being replaced by a layout abstraction.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::main_panel::MainPanel;
use super::serial_sections_process_panel::SerialSectionsProcessPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::serial_sections_file_filter::SerialSectionsFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java final `MainSerialSectionsPanel`, including its `MainPanel` superclass.
pub struct MainSerialSectionsPanel {
    pub main_panel: MainPanel,
    pub axis_panel_a: Option<SerialSectionsProcessPanel>,
    /// Native `getScrollA().add(axisPanelA.getContainer())` call boundary.
    pub scroll_a_components: Vec<bool>,
}

impl MainSerialSectionsPanel {
    /// `MainSerialSectionsPanel(BaseManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            axis_panel_a: None,
            scroll_a_components: Vec::new(),
        }
    }
    /// Java override `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        self.scroll_a_components.push(
            self.axis_panel_a
                .as_ref()
                .unwrap()
                .axis_process_panel
                .get_container(),
        );
    }
    /// Java override `addAxisPanelB()`, empty in the source.
    pub fn add_axis_panel_b(&mut self) {}
    /// Java override `createAxisPanelA(AxisID, AxisProgressPanel)`.
    pub fn create_axis_panel_a(
        &mut self,
        _axis_id: AxisID,
        axis_progress_panel: AxisProgressPanel,
    ) {
        self.axis_panel_a = Some(SerialSectionsProcessPanel::new(
            self.main_panel.manager,
            axis_progress_panel,
        ));
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
    /// Java override `getDataFileFilter()`.
    pub fn get_data_file_filter(&self) -> SerialSectionsFileFilter {
        SerialSectionsFileFilter::new()
    }
    /// Java override `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .unwrap()
            .axis_process_panel
            .hide()
    }
    /// Java override `hideAxisPanelB()`.
    pub fn hide_axis_panel_b(&mut self) -> bool {
        true
    }
    /// Java override `isAxisPanelANull()`.
    pub fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.is_none()
    }
    /// Java override `isAxisPanelBNull()`.
    pub fn is_axis_panel_b_null(&self) -> bool {
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
        self.main_panel.axis_progress_panel_a.as_mut()
    }
    /// Java override `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
    }
    /// Java override `saveDisplayState()`, empty in the source.
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
            .unwrap()
            .axis_process_panel
            .show();
    }
    /// Java override `showAxisPanelB()`, empty in the source.
    pub fn show_axis_panel_b(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    struct ParallelDialog;
    impl AbstractParallelDialog for ParallelDialog {
        fn get_parameters(
            &self,
            _param: &mut dyn crate::imod::etomo::comscript::parallel_param::ParallelParam,
        ) {
        }
        fn get_dialog_type(&self) -> DialogType {
            DialogType::SerialSections
        }
    }
    #[test]
    fn only_axis_creates_serial_sections_process_panel_and_maps_non_b_axis() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let mut panel = MainSerialSectionsPanel::new(manager);
        assert!(panel.is_axis_panel_a_null());
        assert!(panel.is_axis_panel_b_null());
        panel.create_axis_panel_a(
            AxisID::First,
            AxisProgressPanel::get_instance(Some(AxisID::Only), manager),
        );
        assert!(!panel.is_axis_panel_a_null());
        assert_eq!(panel.get_axis_panel_a().unwrap().axis_id, AxisID::Only);
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
        assert!(panel.get_axis_panel_b().is_none());
    }
    #[test]
    fn source_visibility_and_scroll_addition_are_delegated_to_axis_a() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let mut panel = MainSerialSectionsPanel::new(manager);
        panel.create_axis_panel_a(
            AxisID::Only,
            AxisProgressPanel::get_instance(Some(AxisID::Only), manager),
        );
        panel.add_axis_panel_a();
        assert_eq!(panel.scroll_a_components, vec![true]);
        assert!(panel.hide_axis_panel_a());
        assert!(!panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.show_axis_panel_a();
        assert!(panel.get_axis_panel_a().unwrap().panel_root_visible);
        assert!(panel.hide_axis_panel_b());
    }
    #[test]
    fn data_filter_progress_mapping_reset_and_empty_overrides_match_source() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let mut panel = MainSerialSectionsPanel::new(manager);
        panel.main_panel.axis_progress_panel_a =
            Some(AxisProgressPanel::get_instance(Some(AxisID::Only), manager));
        assert_eq!(
            panel
                .map_axis_progress_panel(AxisID::First)
                .unwrap()
                .axis_id,
            AxisID::Only
        );
        assert!(panel.map_axis_progress_panel(AxisID::Second).is_none());
        assert_eq!(
            panel.get_data_file_filter().get_description(),
            "Serial sections data file (.ess)"
        );
        panel.create_axis_panel_a(
            AxisID::Only,
            AxisProgressPanel::get_instance(Some(AxisID::Only), manager),
        );
        panel.set_state(ProcessState::Complete, AxisID::Only, &ParallelDialog);
        panel.save_display_state();
        panel.reset_axis_panels();
        assert!(panel.is_axis_panel_a_null());
    }
}
