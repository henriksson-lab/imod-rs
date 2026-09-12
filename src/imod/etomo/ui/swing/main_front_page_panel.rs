//! `IMOD/Etomo/src/etomo/ui/swing/MainFrontPagePanel.java`.
//!
//! The Java superclass relationship is represented by the owned `main_panel`
//! field. `FrontPageManager` is not yet a Rust source unit, so its declared
//! concrete reference is retained as its actual inherited `BaseManager`
//! interface at the constructor boundary.  Swing's `ScrollPanel.add` remains
//! explicit in `scroll_a_components` rather than being replaced by a layout
//! abstraction.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::front_page_process_panel::FrontPageProcessPanel;
use super::main_panel::MainPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::data_file_filter::DataFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use std::path::Path;

/// Java final `MainFrontPagePanel`, including its inherited `MainPanel` state.
pub struct MainFrontPagePanel {
    pub main_panel: MainPanel,
    /// Java final `manager`; `FrontPageManager.java` remains the concrete
    /// application-manager boundary.
    pub manager: &'static dyn BaseManager,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<FrontPageProcessPanel>,
    /// Native `getScrollA().add(axisPanelA.getContainer())` boundary.
    pub scroll_a_components: Vec<bool>,
}

impl MainFrontPagePanel {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java `MainFrontPagePanel(FrontPageManager)`.
    pub fn new(manager: &'static dyn BaseManager) -> Self {
        Self {
            main_panel: MainPanel::new(manager),
            manager,
            axis_panel_a: None,
            scroll_a_components: Vec::new(),
        }
    }

    /// Java override `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        let _scroll_a = self
            .main_panel
            .get_scroll_a()
            .expect("MainPanel.scrollA is null");
        self.scroll_a_components.push(
            self.axis_panel_a
                .as_ref()
                .expect("MainFrontPagePanel.axisPanelA is null")
                .axis_process_panel
                .get_container(),
        );
    }

    /// Java override `addAxisPanelB()`, whose body is empty.
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
        self.axis_panel_a = Some(FrontPageProcessPanel::new(
            self.manager,
            axis_progress_panel,
        ));
    }

    /// Java override `createAxisPanelB(AxisProgressPanel)`, whose body is empty.
    pub fn create_axis_panel_b(&mut self, _axis_progress_panel: AxisProgressPanel) {}

    /// Java override `getAxisPanelA()`.
    pub fn get_axis_panel_a(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_a
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }

    /// Java override `getAxisPanelB()`, which returns null.
    pub fn get_axis_panel_b(&mut self) -> Option<&mut AxisProcessPanel> {
        None
    }

    /// Java override `getDataFileFilter()`, which returns null.
    pub fn get_data_file_filter(&self) -> Option<DataFileFilter> {
        None
    }

    /// Java override `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainFrontPagePanel.axisPanelA is null")
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
    }

    /// Java override `saveDisplayState()`, whose body is empty.
    pub fn save_display_state(&mut self) {}

    /// Java override `setState(ProcessState, AxisID, AbstractParallelDialog)`, whose body is empty.
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
            .expect("MainFrontPagePanel.axisPanelA is null")
            .axis_process_panel
            .show();
    }

    /// Java override `showAxisPanelB()`, whose body is empty.
    pub fn show_axis_panel_b(&mut self) {}

    /// Java override `setStatusBarText(File, BaseMetaData, LogWindow)`.
    /// `param_file` and `LogWindow` are declared by the override but neither is
    /// read in its source body. A null Java metadata name maps to the empty
    /// native status string because this state field cannot itself be null.
    pub fn set_status_bar_text(
        &mut self,
        _param_file: Option<&Path>,
        meta_data: &dyn BaseMetaData,
    ) {
        self.main_panel.status_bar = meta_data.get_name().unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::comscript::parallel_param::ParallelParam;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    use crate::imod::etomo::storage::storable::Storable;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::r#type::directive_editor_meta_data::DirectiveEditorMetaData;
    use crate::imod::etomo::ui::swing::scroll_panel::ScrollPanel;
    use std::collections::BTreeMap;
    use std::path::Path;

    struct Dialog;
    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

        fn get_dialog_type(&self) -> DialogType {
            DialogType::Tools
        }
    }

    fn panel() -> MainFrontPagePanel {
        MainFrontPagePanel::new(DirectiveEditorManager::new(None, None, None, None))
    }

    #[test]
    fn creates_maps_and_adds_only_the_source_a_axis_panel() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::First, progress);
        assert!(!panel.is_axis_panel_a_null());
        assert_eq!(panel.get_axis_panel_a().unwrap().axis_id, AxisID::Only);
        assert_eq!(
            panel.get_axis_panel_a().unwrap().interface_type,
            crate::imod::etomo::r#type::interface_type::InterfaceType::FrontPage
        );
        assert!(panel.map_base_axis_process_panel(AxisID::First).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
        panel.main_panel.scroll_a = Some(ScrollPanel::new());
        panel.add_axis_panel_a();
        assert_eq!(panel.scroll_a_components, vec![true]);
    }

    #[test]
    fn b_axis_null_empty_overrides_and_visibility_match_source() {
        let mut panel = panel();
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), panel.manager);
        panel.create_axis_panel_a(AxisID::Only, progress);
        assert!(panel.hide_axis_panel_a());
        assert!(!panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.show_axis_panel_a();
        assert!(panel.get_axis_panel_a().unwrap().panel_root_visible);
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
        panel.add_axis_panel_b();
        panel.create_axis_panel_b(AxisProgressPanel::get_instance(
            Some(AxisID::Only),
            panel.manager,
        ));
        panel.show_axis_panel_b();
    }

    #[test]
    fn status_uses_metadata_name_and_empty_overrides_preserve_state() {
        let mut panel = panel();
        let metadata = DirectiveEditorMetaData::new(None, None, None, true);
        metadata.set_root_name(Some(Path::new("front-page.ejf")));
        panel.set_status_bar_text(None, &metadata);
        assert_eq!(panel.main_panel.get_status_bar_text(), "front-page.ejf");
        panel.set_state(ProcessState::Complete, AxisID::Only, &Dialog);
        panel.save_display_state();
        panel.reset_axis_panels();
        assert!(panel.is_axis_panel_a_null());
        assert_eq!(MainFrontPagePanel::RCSID, "$Id$");
        let mut properties = BTreeMap::new();
        metadata.store(&mut properties);
    }
}
