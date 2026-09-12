//! `IMOD/Etomo/src/etomo/ui/swing/ParallelProcessPanel.java`.
//!
//! `ParallelManager.java` has not yet been translated.  Its only use in this
//! source unit is as the `BaseManager` argument passed unchanged to
//! `AxisProcessPanel`, so the inherited Rust manager reference retains that
//! source boundary without inventing a stand-in manager.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java `ParallelProcessPanel`, including its `AxisProcessPanel` superclass.
pub struct ParallelProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl ParallelProcessPanel {
    /// `ParallelProcessPanel(ParallelManager, AxisProgressPanel)`.
    pub fn new(manager: &'static dyn BaseManager, axis_progress_panel: AxisProgressPanel) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            true,
            true,
            InterfaceType::Pp,
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
        // `Colors.getBackgroundParallel()`: 186, 224, 173 when APRIL_FOOLS is
        // false.  The Colors unit is still an independent source boundary, so
        // retain the concrete color passed to the inherited Swing operation.
        self.axis_process_panel.set_background("rgb(186,224,173)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[test]
    fn constructor_preserves_parallel_superclass_arguments_and_initialization_order() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = ParallelProcessPanel::new(manager, axis_progress_panel);
        let superclass = &panel.axis_process_panel;
        assert_eq!(superclass.axis_id, AxisID::Only);
        assert!(superclass.popup_chunk_warnings);
        assert!(superclass.runnable_parallel);
        assert_eq!(superclass.interface_type, InterfaceType::Pp);
        assert!(!superclass.alt_parallel_loc);
        assert_eq!(superclass.background.as_deref(), Some("rgb(186,224,173)"));
        assert!(superclass.panel_process_info_has_status);
    }
}
