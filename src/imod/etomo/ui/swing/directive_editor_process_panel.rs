//! `IMOD/Etomo/src/etomo/ui/swing/DirectiveEditorProcessPanel.java`.
//!
//! This package-private source unit keeps its `AxisProcessPanel` superclass
//! explicit.  `Colors.getBackgroundTools()` remains a Swing colour boundary;
//! its source RGB result is retained instead of introducing a replacement
//! theme abstraction.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private final `DirectiveEditorProcessPanel`.
pub struct DirectiveEditorProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl DirectiveEditorProcessPanel {
    /// Java `DirectiveEditorProcessPanel(DirectiveEditorManager, InterfaceType,
    /// AxisProgressPanel)`.
    pub fn new(
        manager: &'static DirectiveEditorManager,
        interface_type: InterfaceType,
        axis_progress_panel: AxisProgressPanel,
    ) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            true,
            true,
            interface_type,
            false,
            axis_progress_panel,
        );
        let mut instance = Self { axis_process_panel };
        instance.axis_process_panel.create_process_control_panel();
        instance.show_both_axis();
        instance.axis_process_panel.initialize_panels(true);
        instance
    }

    /// Java override `showBothAxis()`.
    pub fn show_both_axis(&mut self) {
        // `Colors.getBackgroundTools()` returns this cached source colour on
        // ordinary dates and its separate April Fools colour on April 1.
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

    #[test]
    fn constructor_preserves_superclass_arguments_initialization_and_tools_background() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel =
            DirectiveEditorProcessPanel::new(manager, InterfaceType::DirectiveEditor, progress);

        assert_eq!(panel.axis_process_panel.axis_id, AxisID::Only);
        assert!(panel.axis_process_panel.popup_chunk_warnings);
        assert!(panel.axis_process_panel.runnable_parallel);
        assert_eq!(
            panel.axis_process_panel.interface_type,
            InterfaceType::DirectiveEditor
        );
        assert!(!panel.axis_process_panel.alt_parallel_loc);
        assert!(panel.axis_process_panel.panel_process_info_has_status);
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
