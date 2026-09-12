//! `IMOD/Etomo/src/etomo/ui/swing/ToolsProcessPanel.java`.
//!
//! `ToolsManager.java` is the declared manager type in the Java constructor.
//! Its complete source unit has not yet crossed the Rust boundary, so the
//! inherited `BaseManager` surface is retained here rather than inventing a
//! stand-in manager only for this three-method source unit.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java final `ToolsProcessPanel`, including its `AxisProcessPanel` superclass.
pub struct ToolsProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl ToolsProcessPanel {
    /// `ToolsProcessPanel(ToolsManager, AxisProgressPanel)`.
    pub fn new(manager: &'static dyn BaseManager, axis_progress_panel: AxisProgressPanel) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            true,
            true,
            InterfaceType::Tools,
            false,
            axis_progress_panel,
        );
        let mut instance = Self { axis_process_panel };
        instance.axis_process_panel.create_process_control_panel();
        instance.show_both_axis();
        instance.axis_process_panel.initialize_panels(true);
        instance
    }

    /// `showBothAxis()`.
    pub fn show_both_axis(&mut self) {
        // `Colors.getBackgroundTools()`: the `Color` Swing boundary is reduced
        // to its exact RGB value, including the source's April Fools branch.
        if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            self.axis_process_panel.set_background("rgb(52,130,218)");
        } else {
            self.axis_process_panel.set_background("rgb(173,212,224)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[test]
    fn constructor_preserves_tools_superclass_arguments_and_initialization_order() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = ToolsProcessPanel::new(manager, axis_progress_panel);
        let superclass = &panel.axis_process_panel;
        assert_eq!(superclass.axis_id, AxisID::Only);
        assert!(superclass.popup_chunk_warnings);
        assert!(superclass.runnable_parallel);
        assert_eq!(superclass.interface_type, InterfaceType::Tools);
        assert!(!superclass.alt_parallel_loc);
        assert_eq!(superclass.panel_process_select_axis_label, None);
        assert!(superclass.panel_process_info_has_status);
    }

    #[test]
    fn show_both_axis_uses_the_source_tools_background() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = ToolsProcessPanel::new(manager, axis_progress_panel);
        let expected = if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            "rgb(52,130,218)"
        } else {
            "rgb(173,212,224)"
        };
        assert_eq!(
            panel.axis_process_panel.background.as_deref(),
            Some(expected)
        );
    }
}
