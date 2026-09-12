//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoProcessPanel.java`.
//!
//! Java inheritance is represented by the owned `axis_process_panel` field.
//! The `Colors.getBackgroundBatchruntomo()` call stays at the presentation
//! boundary as its exact RGB result, including the source's April Fools path.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `BatchRunTomoProcessPanel`, including its
/// `AxisProcessPanel` superclass state.
pub struct BatchRunTomoProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl BatchRunTomoProcessPanel {
    /// Java `BatchRunTomoProcessPanel(BaseManager, InterfaceType, AxisProgressPanel)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        interface_type: InterfaceType,
        axis_progress_panel: AxisProgressPanel,
    ) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            true,
            false,
            interface_type,
            true,
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
        if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            self.axis_process_panel.set_background("rgb(255,239,192)");
        } else {
            self.axis_process_panel.set_background("rgb(199,173,224)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[test]
    fn constructor_preserves_batch_run_tomo_superclass_arguments_and_order() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = BatchRunTomoProcessPanel::new(manager, InterfaceType::BatchRunTomo, progress);
        let superclass = &panel.axis_process_panel;

        assert_eq!(superclass.axis_id, AxisID::Only);
        assert!(superclass.popup_chunk_warnings);
        assert!(!superclass.runnable_parallel);
        assert_eq!(superclass.interface_type, InterfaceType::BatchRunTomo);
        assert!(superclass.alt_parallel_loc);
        assert_eq!(superclass.panel_process_select_axis_label, None);
        assert!(superclass.panel_process_info_has_status);
    }

    #[test]
    fn show_both_axis_uses_source_batchruntomo_color() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let progress = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = BatchRunTomoProcessPanel::new(manager, InterfaceType::BatchRunTomo, progress);
        let expected = if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            "rgb(255,239,192)"
        } else {
            "rgb(199,173,224)"
        };
        assert_eq!(
            panel.axis_process_panel.background.as_deref(),
            Some(expected)
        );
        assert_eq!(
            panel
                .axis_process_panel
                .axis_progress_panel
                .background
                .as_deref(),
            Some(expected)
        );
    }
}
