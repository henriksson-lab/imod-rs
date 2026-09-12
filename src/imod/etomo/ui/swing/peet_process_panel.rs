//! `IMOD/Etomo/src/etomo/ui/swing/PeetProcessPanel.java`.
//!
//! `PeetManager.java` and its `PeetProcessManager` are separate source units.
//! This unit only passes the manager through to its `AxisProcessPanel`
//! superclass, so the declared manager remains the inherited `BaseManager`
//! boundary here rather than being replaced with a panel-local substitute.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `PeetProcessPanel`, including its `AxisProcessPanel`
/// superclass state.
pub struct PeetProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl PeetProcessPanel {
    /// `PeetProcessPanel(BaseManager, AxisProgressPanel)`.
    ///
    /// The Java parameter is concretely `PeetManager`; the unported manager
    /// source unit contributes only the inherited `BaseManager` contract used
    /// by this three-call constructor.
    pub fn new(manager: &'static dyn BaseManager, axis_progress_panel: AxisProgressPanel) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            false,
            true,
            InterfaceType::Peet,
            false,
            axis_progress_panel,
        );
        let mut instance = Self { axis_process_panel };
        instance.axis_process_panel.create_process_control_panel();
        // Java's no-argument `initializePanels()` queries the Peet process
        // manager.  The translated `AxisProcessPanel` makes that source
        // boundary explicit with this boolean; a Peet manager owns one.
        instance.axis_process_panel.initialize_panels(true);
        instance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[test]
    fn constructor_preserves_peet_superclass_arguments() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = PeetProcessPanel::new(manager, axis_progress_panel);
        let superclass = &panel.axis_process_panel;

        assert_eq!(superclass.axis_id, AxisID::Only);
        assert!(!superclass.popup_chunk_warnings);
        assert!(superclass.runnable_parallel);
        assert_eq!(superclass.interface_type, InterfaceType::Peet);
        assert!(!superclass.alt_parallel_loc);
    }

    #[test]
    fn constructor_creates_control_panel_then_initializes_panels() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = PeetProcessPanel::new(manager, axis_progress_panel);

        assert_eq!(
            panel.axis_process_panel.panel_process_select_axis_label,
            None
        );
        assert!(panel.axis_process_panel.panel_process_info_has_status);
    }
}
