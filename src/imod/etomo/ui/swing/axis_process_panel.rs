//! `IMOD/Etomo/src/etomo/ui/swing/AxisProcessPanel.java`.
//!
//! This source class is abstract.  The concrete process dialogs, ParallelPanel,
//! ContextPopup, and processing-method mediator are retained as named boundaries;
//! their state transitions initiated by AxisProcessPanel are represented here.
#![allow(dead_code)]

use super::axis_progress_panel::AxisProgressPanel;
use super::parallel_panel::ParallelPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Fields and concrete method bodies of Java's abstract `AxisProcessPanel`.
pub struct AxisProcessPanel {
    pub panel_root_visible: bool,
    pub panel_process_info_has_status: bool,
    pub outer_status_panel_present: bool,
    pub panel_dialog: Option<String>,
    pub parallel_status_panel_visible: bool,
    /// Java `EtomoNumber lastWidth`, null is `None`.
    pub last_width: Option<i32>,
    pub panel_process_select_axis_label: Option<String>,
    pub manager: &'static dyn BaseManager,
    pub axis_id: AxisID,
    pub interface_type: InterfaceType,
    pub popup_chunk_warnings: bool,
    pub alt_parallel_loc: bool,
    pub runnable_parallel: bool,
    pub axis_progress_panel: AxisProgressPanel,
    pub parallel_showing: bool,
    pub processing_method_locked: bool,
    pub parallel_panel: Option<ParallelPanel>,
    pub parallel_pause_enabled: Option<bool>,
    pub parallel_load_running: bool,
    pub visible_width: i32,
    pub background: Option<String>,
}
impl AxisProcessPanel {
    /// `AxisProcessPanel(AxisID, BaseManager, boolean, boolean, InterfaceType,
    /// boolean, AxisProgressPanel)`.
    pub fn new(
        axis_id: AxisID,
        manager: &'static dyn BaseManager,
        popup_chunk_warnings: bool,
        runnable_parallel: bool,
        interface_type: InterfaceType,
        alt_parallel_loc: bool,
        axis_progress_panel: AxisProgressPanel,
    ) -> Self {
        Self {
            panel_root_visible: true,
            panel_process_info_has_status: false,
            outer_status_panel_present: true,
            panel_dialog: None,
            parallel_status_panel_visible: false,
            last_width: None,
            panel_process_select_axis_label: None,
            manager,
            axis_id,
            interface_type,
            popup_chunk_warnings,
            alt_parallel_loc,
            runnable_parallel,
            axis_progress_panel,
            parallel_showing: false,
            processing_method_locked: false,
            parallel_panel: None,
            parallel_pause_enabled: None,
            parallel_load_running: false,
            visible_width: 0,
            background: None,
        }
    }
    /// `showBothAxis()`, empty in Java.
    pub fn show_both_axis(&mut self) {}
    /// `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.panel_root_visible = visible;
    }
    /// `setBackground(Color)`.
    pub fn set_background(&mut self, color: impl Into<String>) {
        let color = color.into();
        self.background = Some(color.clone());
        self.axis_progress_panel.set_background(color);
    }
    /// `initializePanels()`. `getProcessManager` is null at its current source boundary,
    /// so callers supply whether that branch is available.
    pub fn initialize_panels(&mut self, has_process_manager: bool) {
        self.panel_process_info_has_status =
            has_process_manager && self.manager.allow_process_watching();
    }
    /// `hide()`.
    pub fn hide(&mut self) -> bool {
        if self.get_width() != 0 {
            return false;
        }
        self.panel_root_visible = false;
        true
    }
    /// `lockProcessingMethod(boolean)`.
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.processing_method_locked = lock;
    }
    /// `showParallelPanel(boolean)`.
    pub fn show_parallel_panel(&mut self, show: bool) {
        self.show_parallel_panel_force(show, false);
    }
    /// `forceShowParallelPanel(boolean)`.
    pub fn force_show_parallel_panel(&mut self, show: bool) {
        self.show_parallel_panel_force(show, true);
    }
    /// `getCPUsSelectedInt(boolean)`: CPU validation belongs to ParallelPanel.
    pub fn get_cpus_selected_int(&self, do_validation: bool) -> Result<i32, String> {
        self.parallel_panel
            .as_ref()
            .map_or(Ok(0), |panel| panel.get_cpus_selected_int(do_validation))
    }
    /// `buildParallelPanel()`.
    pub fn build_parallel_panel(&mut self) {
        if self.parallel_panel.is_none() {
            self.parallel_panel = Some(ParallelPanel::get_instance(
                self.manager,
                self.axis_id,
                self,
                self.popup_chunk_warnings,
                self.runnable_parallel,
                self.interface_type,
                false,
                false,
            ));
        }
    }
    /// private `showParallelPanel(boolean, boolean)`.
    fn show_parallel_panel_force(&mut self, show: bool, force: bool) {
        if self.processing_method_locked && !force {
            return;
        }
        if !show {
            if self.parallel_panel.is_some() && self.parallel_showing {
                self.parallel_showing = false;
                self.parallel_status_panel_visible = false;
            }
        } else {
            self.build_parallel_panel();
            if !self.parallel_showing {
                self.parallel_showing = true;
                self.parallel_status_panel_visible = true;
            }
        }
    }
    /// private `startParallelPanel()`.
    pub fn start_parallel_panel(&mut self) {
        self.parallel_showing = true;
        self.parallel_load_running = true;
        self.parallel_status_panel_visible = true;
    }
    /// private `stopParallelPanel()`.
    pub fn stop_parallel_panel(&mut self) {
        self.parallel_showing = false;
        self.parallel_load_running = false;
        self.parallel_status_panel_visible = false;
    }
    /// `getParallelPanel()`.
    pub fn get_parallel_panel(&self) -> Option<&ParallelPanel> {
        self.parallel_panel.as_ref()
    }
    /// `getParallelStatusPanel()`.
    pub fn get_parallel_status_panel(&self) -> bool {
        self.parallel_status_panel_visible
    }
    /// `done()`; `ParallelPanel.getHeaderState` is its direct boundary.
    pub fn done(&mut self) {}
    /// `show()`.
    pub fn show(&mut self) {
        self.panel_root_visible = true;
    }
    /// `saveDisplayState()`.
    pub fn save_display_state(&mut self) {
        self.last_width = Some(self.get_width());
    }
    /// `getWidth()`.
    pub fn get_width(&mut self) -> i32 {
        if let Some(width) = self.last_width.take() {
            width
        } else {
            self.visible_width
        }
    }
    /// `getContainer()`; native JPanel is represented by its visibility.
    pub fn get_container(&self) -> bool {
        self.panel_root_visible
    }
    /// `replaceDialogPanel(Container)`.
    pub fn replace_dialog_panel(&mut self, dialog: impl Into<String>) {
        self.panel_dialog = Some(dialog.into());
    }
    /// `eraseDialogPanel()`.
    pub fn erase_dialog_panel(&mut self) {
        self.panel_dialog = None;
    }
    /// `setPauseEnabled(boolean)`.
    pub fn set_pause_enabled(&mut self, enable_pause: bool) {
        if let Some(panel) = &mut self.parallel_panel {
            panel.set_pause_enabled(enable_pause);
            self.parallel_pause_enabled = Some(enable_pause);
        }
    }
    /// `popUpContextMenu(MouseEvent)`; ContextPopup is the concrete GUI boundary.
    pub fn pop_up_context_menu(&self) -> Result<(), String> {
        Err("ContextPopup.java is not yet translated".into())
    }
    /// `createProcessControlPanel()`.
    pub fn create_process_control_panel(&mut self) {
        self.panel_process_select_axis_label = match self.axis_id {
            AxisID::First => Some("Axis A:".into()),
            AxisID::Second => Some("Axis B:".into()),
            AxisID::Only => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn panel() -> AxisProcessPanel {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        AxisProcessPanel::new(
            AxisID::First,
            manager,
            false,
            true,
            InterfaceType::DirectiveEditor,
            false,
            AxisProgressPanel::get_instance(Some(AxisID::First), manager),
        )
    }
    #[test]
    fn locked_method_prevents_non_forced_parallel_visibility_change() {
        let mut p = panel();
        p.lock_processing_method(true);
        p.show_parallel_panel(true);
        assert!(p.parallel_panel.is_none());
        p.force_show_parallel_panel(true);
        assert!(
            p.parallel_panel.is_some() && p.parallel_showing && p.parallel_status_panel_visible
        );
    }
    #[test]
    fn saved_width_is_consumed_once() {
        let mut p = panel();
        p.visible_width = 50;
        p.save_display_state();
        p.visible_width = 0;
        assert_eq!(p.get_width(), 50);
        assert_eq!(p.get_width(), 0);
    }
    #[test]
    fn process_control_axis_label_matches_source() {
        let mut p = panel();
        p.create_process_control_panel();
        assert_eq!(
            p.panel_process_select_axis_label.as_deref(),
            Some("Axis A:")
        );
    }
}
