//! `IMOD/Etomo/src/etomo/ui/swing/MainJoinPanel.java`.
//!
//! Java inheritance is represented by the owned `main_panel` superclass
//! state.  Swing `JPanel.add`/`revalidate` and `UIHarness.pack` remain direct
//! presentation boundaries; `JoinManager` remains the application boundary.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use super::join_process_panel::JoinProcessPanel;
use super::main_panel::MainPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::join_file_filter::JoinFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use std::path::Path;

/// Direct `UIHarness.INSTANCE.pack(manager)` call in `openPanel`.
pub trait MainJoinPanelUiHarness {
    fn pack(&mut self, manager: &'static dyn BaseManager);
}

/// Java `MainJoinPanel`, including its `MainPanel` superclass state.
pub struct MainJoinPanel {
    pub main_panel: MainPanel,
    /// The Java constructor's concrete `JoinManager` argument.  MainPanel
    /// stores it through its declared `BaseManager` superclass type; this
    /// field retains the concrete constructor reference for the source's
    /// `(JoinManager) manager` cast in `createAxisPanelA`.
    pub join_manager: &'static JoinManager,
    /// Java `axisPanelA`, null before `createAxisPanelA`.
    pub axis_panel_a: Option<JoinProcessPanel>,
    /// Java `axisPanelB`, which this class never constructs but whose direct
    /// override methods still dereference when invoked.
    pub axis_panel_b: Option<JoinProcessPanel>,
    /// `getScrollA().add` native presentation-boundary children, in source
    /// insertion order.  A `String` stands for the supplied Java `JPanel`.
    pub scroll_a_components: Vec<String>,
    /// `getScrollB().add` native presentation-boundary children.
    pub scroll_b_components: Vec<String>,
    /// Java `revalidate()` calls in `openPanel`.
    pub revalidate_count: u64,
}

impl MainJoinPanel {
    /// Java `MainJoinPanel(JoinManager)`.
    pub fn new(join_manager: &'static JoinManager) -> Self {
        Self {
            main_panel: MainPanel::new(join_manager),
            join_manager,
            axis_panel_a: None,
            axis_panel_b: None,
            scroll_a_components: Vec::new(),
            scroll_b_components: Vec::new(),
            revalidate_count: 0,
        }
    }

    /// Java override `saveDisplayState()`, whose body is empty.
    pub fn save_display_state(&mut self) {}

    /// Java override `getDataFileFilter()`.
    pub fn get_data_file_filter(&self) -> JoinFileFilter {
        JoinFileFilter::new()
    }

    /// Java override `createAxisPanelA(AxisID, AxisProgressPanel)`.
    pub fn create_axis_panel_a(&mut self, axis_id: AxisID, axis_progress_panel: AxisProgressPanel) {
        self.axis_panel_a = Some(JoinProcessPanel::new(
            self.join_manager,
            axis_id,
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

    /// Java override `getAxisPanelB()`, returning null before any impossible
    /// external state mutation.
    pub fn get_axis_panel_b(&mut self) -> Option<&mut AxisProcessPanel> {
        self.axis_panel_b
            .as_mut()
            .map(|panel| &mut panel.axis_process_panel)
    }

    /// Java `openPanel(JPanel)`: the `JPanel` is represented by its native
    /// presentation-boundary identifier.
    pub fn open_panel<U: MainJoinPanelUiHarness>(&mut self, panel: String, ui_harness: &mut U) {
        self.scroll_a_components.push(panel);
        self.revalidate_count += 1;
        ui_harness.pack(self.main_panel.manager);
    }

    /// Java override `setStatusBarText(File, BaseMetaData, LogWindow)`.
    /// `LogWindow` is forwarded only by the superclass implementation and is
    /// not read by either translated body, so it has no Rust state parameter.
    pub fn set_status_bar_text(
        &mut self,
        param_file: Option<&Path>,
        meta_data: Option<&dyn BaseMetaData>,
    ) {
        if meta_data.is_none_or(|meta_data| !meta_data.is_valid()) {
            // Source constructs an empty `StringBuffer`, unlike MainPanel's
            // no-data default status text.
            self.main_panel.status_bar.clear();
        } else {
            self.main_panel.set_status_bar_text(param_file, true);
        }
    }

    /// Java override `resetAxisPanels()`.
    pub fn reset_axis_panels(&mut self) {
        self.axis_panel_a = None;
        self.axis_panel_b = None;
    }

    /// Java override `addAxisPanelA()`.
    pub fn add_axis_panel_a(&mut self) {
        self.scroll_a_components.push(
            self.axis_panel_a
                .as_ref()
                .expect("MainJoinPanel.axisPanelA is null")
                .axis_process_panel
                .get_container()
                .then_some("axis-panel-a".to_owned())
                .expect("AxisProcessPanel container is absent"),
        );
    }

    /// Java override `addAxisPanelB()`.
    pub fn add_axis_panel_b(&mut self) {
        self.scroll_b_components.push(
            self.axis_panel_b
                .as_ref()
                .expect("MainJoinPanel.axisPanelB is null")
                .axis_process_panel
                .get_container()
                .then_some("axis-panel-b".to_owned())
                .expect("AxisProcessPanel container is absent"),
        );
    }

    /// Java override `isAxisPanelANull()`.
    pub fn is_axis_panel_a_null(&self) -> bool {
        self.axis_panel_a.is_none()
    }

    /// Java override `isAxisPanelBNull()`.
    pub fn is_axis_panel_b_null(&self) -> bool {
        self.axis_panel_b.is_none()
    }

    /// Java override `hideAxisPanelA()`.
    pub fn hide_axis_panel_a(&mut self) -> bool {
        self.axis_panel_a
            .as_mut()
            .expect("MainJoinPanel.axisPanelA is null")
            .axis_process_panel
            .hide()
    }

    /// Java override `hideAxisPanelB()`.
    pub fn hide_axis_panel_b(&mut self) -> bool {
        self.axis_panel_b
            .as_mut()
            .expect("MainJoinPanel.axisPanelB is null")
            .axis_process_panel
            .hide()
    }

    /// Java override `showAxisPanelA()`.
    pub fn show_axis_panel_a(&mut self) {
        self.axis_panel_a
            .as_mut()
            .expect("MainJoinPanel.axisPanelA is null")
            .axis_process_panel
            .show();
    }

    /// Java override `showAxisPanelB()`.
    pub fn show_axis_panel_b(&mut self) {
        self.axis_panel_b
            .as_mut()
            .expect("MainJoinPanel.axisPanelB is null")
            .axis_process_panel
            .show();
    }

    /// Java override `mapBaseAxisProcessPanel(AxisID)`.
    pub fn map_base_axis_process_panel(
        &mut self,
        axis_id: AxisID,
    ) -> Option<&mut AxisProcessPanel> {
        if axis_id == AxisID::Second {
            self.get_axis_panel_b()
        } else {
            self.get_axis_panel_a()
        }
    }

    /// Java override `mapAxisProgressPanel(AxisID)` through the source
    /// `MainPanel.getProgressPanel` lazy construction path.
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

    /// Java override `setState(ProcessState, AxisID, AbstractParallelDialog)`,
    /// whose body is empty.
    pub fn set_state(
        &mut self,
        _process_state: ProcessState,
        _axis_id: AxisID,
        _parallel_dialog: &dyn AbstractParallelDialog,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::comscript::parallel_param::ParallelParam;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::r#type::directive_editor_meta_data::DirectiveEditorMetaData;
    use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;

    struct Harness {
        packed: usize,
    }
    impl MainJoinPanelUiHarness for Harness {
        fn pack(&mut self, _manager: &'static dyn BaseManager) {
            self.packed += 1;
        }
    }
    struct Dialog;
    impl AbstractParallelDialog for Dialog {
        fn get_parameters(&self, _param: &mut dyn ParallelParam) {}
        fn get_dialog_type(&self) -> DialogType {
            DialogType::Join
        }
    }

    fn panel() -> MainJoinPanel {
        MainJoinPanel::new(JoinManager::new(None, Some(AxisID::Only)))
    }

    #[test]
    fn creates_and_maps_only_the_source_a_axis_join_panel() {
        let mut panel = panel();
        assert!(panel.is_axis_panel_a_null());
        assert!(panel.is_axis_panel_b_null());
        let progress =
            AxisProgressPanel::get_instance(Some(AxisID::Only), panel.main_panel.manager);
        panel.create_axis_panel_a(AxisID::Only, progress);
        assert!(!panel.is_axis_panel_a_null());
        assert_eq!(panel.get_axis_panel_a().unwrap().axis_id, AxisID::Only);
        assert!(panel.map_base_axis_process_panel(AxisID::Only).is_some());
        assert!(panel.map_base_axis_process_panel(AxisID::Second).is_none());
        assert!(panel.map_axis_progress_panel(AxisID::Second).is_none());
    }

    #[test]
    fn add_visibility_reset_and_empty_state_follow_the_source_overrides() {
        let mut panel = panel();
        let progress =
            AxisProgressPanel::get_instance(Some(AxisID::Only), panel.main_panel.manager);
        panel.create_axis_panel_a(AxisID::Only, progress);
        panel.add_axis_panel_a();
        assert_eq!(panel.scroll_a_components, vec!["axis-panel-a"]);
        assert!(panel.hide_axis_panel_a());
        assert!(!panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.show_axis_panel_a();
        assert!(panel.get_axis_panel_a().unwrap().panel_root_visible);
        panel.set_state(ProcessState::Complete, AxisID::Only, &Dialog);
        panel.save_display_state();
        panel.reset_axis_panels();
        assert!(panel.is_axis_panel_a_null());
        assert!(panel.is_axis_panel_b_null());
    }

    #[test]
    fn open_panel_status_and_filter_keep_join_specific_behavior() {
        let mut panel = panel();
        let mut harness = Harness { packed: 0 };
        panel.open_panel("initial-join".to_owned(), &mut harness);
        assert_eq!(panel.scroll_a_components, vec!["initial-join"]);
        assert_eq!(panel.revalidate_count, 1);
        assert_eq!(harness.packed, 1);
        panel.set_status_bar_text(Some(Path::new("joined.ejf")), None);
        assert_eq!(panel.main_panel.status_bar, "");
        let meta_data =
            DirectiveEditorMetaData::new(None, Some(DirectiveFileType::User), None, false);
        meta_data.set_root_name(Some(Path::new("joined")));
        panel.set_status_bar_text(Some(Path::new("joined.ejf")), Some(&meta_data));
        assert_eq!(panel.main_panel.status_bar, "Data file: joined.ejf");
        assert_eq!(
            panel.get_data_file_filter().get_description(),
            "Join data file"
        );
    }
}
