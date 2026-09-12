//! `IMOD/Etomo/src/etomo/ui/swing/FrontPageProcessPanel.java`.
//!
//! `FrontPageManager.java` is the declared concrete constructor type.  Until
//! that manager source unit crosses the Rust boundary, its inherited
//! `BaseManager` reference is preserved exactly at this direct constructor
//! boundary; no substitute front-page manager is introduced here.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private final `FrontPageProcessPanel`, including its
/// `AxisProcessPanel` superclass state.
pub struct FrontPageProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl FrontPageProcessPanel {
    /// Java `FrontPageProcessPanel(FrontPageManager, AxisProgressPanel)`.
    pub fn new(manager: &'static dyn BaseManager, axis_progress_panel: AxisProgressPanel) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            true,
            true,
            InterfaceType::FrontPage,
            false,
            axis_progress_panel,
        );
        let mut instance = Self { axis_process_panel };
        // `FrontPageManager.allowProcessWatching()` returns false.  Its lazy
        // `getProcessManager()` creation is an adjacent manager boundary, but
        // this source call consequently leaves the process-info status absent.
        instance.axis_process_panel.initialize_panels(false);
        instance
    }

    /// Java override `createProcessControlPanel()`, whose body is empty.
    pub fn create_process_control_panel(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[test]
    fn constructor_preserves_front_page_axis_process_panel_arguments() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let mut panel = FrontPageProcessPanel::new(manager, progress);
        assert_eq!(panel.axis_process_panel.axis_id, AxisID::Only);
        assert!(panel.axis_process_panel.popup_chunk_warnings);
        assert!(panel.axis_process_panel.runnable_parallel);
        assert_eq!(
            panel.axis_process_panel.interface_type,
            InterfaceType::FrontPage
        );
        assert!(!panel.axis_process_panel.alt_parallel_loc);
        assert!(!panel.axis_process_panel.panel_process_info_has_status);
        panel.create_process_control_panel();
        assert_eq!(
            panel.axis_process_panel.panel_process_select_axis_label,
            None
        );
    }
}
