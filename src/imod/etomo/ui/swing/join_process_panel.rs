//! `IMOD/Etomo/src/etomo/ui/swing/JoinProcessPanel.java`.
//!
//! Java inheritance is represented by the owned `axis_process_panel` field.
//! `JoinManager` and the Swing color are direct application/presentation
//! boundaries; this small source unit contributes only its constructor call
//! sequence and `showBothAxis` override.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `JoinProcessPanel`, including its `AxisProcessPanel`
/// superclass state.
pub struct JoinProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl JoinProcessPanel {
    /// `JoinProcessPanel(JoinManager, AxisID, AxisProgressPanel)`.
    pub fn new(
        join_manager: &'static JoinManager,
        axis: AxisID,
        axis_progress_panel: AxisProgressPanel,
    ) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            axis,
            join_manager,
            true,
            true,
            InterfaceType::Join,
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
        // `Colors.getBackgroundJoin()` evaluates this same April Fools branch
        // once before caching its `Color` instance.
        if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            self.axis_process_panel.set_background("rgb(162,167,255)");
        } else {
            self.axis_process_panel.set_background("rgb(199,173,224)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_preserves_join_superclass_arguments_and_initialization_order() {
        let manager = JoinManager::new(None, Some(AxisID::Only));
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = JoinProcessPanel::new(manager, AxisID::Only, axis_progress_panel);
        let superclass = &panel.axis_process_panel;
        assert_eq!(superclass.axis_id, AxisID::Only);
        assert!(superclass.popup_chunk_warnings);
        assert!(superclass.runnable_parallel);
        assert_eq!(superclass.interface_type, InterfaceType::Join);
        assert!(!superclass.alt_parallel_loc);
        assert_eq!(superclass.panel_process_select_axis_label, None);
        assert!(superclass.panel_process_info_has_status);
    }

    #[test]
    fn show_both_axis_uses_the_source_join_background() {
        let manager = JoinManager::new(None, Some(AxisID::First));
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::First), manager);
        let panel = JoinProcessPanel::new(manager, AxisID::First, axis_progress_panel);
        let expected = if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            "rgb(162,167,255)"
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
