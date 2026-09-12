//! `IMOD/Etomo/src/etomo/ui/swing/SerialSectionsProcessPanel.java`.
//!
//! Java inheritance is represented by the owned `axis_process_panel` field.
//! The color returned by `Colors.getBackgroundSerialSections()` is retained at
//! the Swing presentation boundary as its source RGB value.
#![allow(dead_code)]

use super::axis_process_panel::AxisProcessPanel;
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `SerialSectionsProcessPanel`, including its
/// `AxisProcessPanel` superclass state.
pub struct SerialSectionsProcessPanel {
    pub axis_process_panel: AxisProcessPanel,
}

impl SerialSectionsProcessPanel {
    /// `SerialSectionsProcessPanel(BaseManager, AxisProgressPanel)`.
    pub fn new(manager: &'static dyn BaseManager, axis_progress_panel: AxisProgressPanel) -> Self {
        let axis_process_panel = AxisProcessPanel::new(
            AxisID::Only,
            manager,
            false,
            true,
            InterfaceType::SerialSections,
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
        // `Colors.getBackgroundSerialSections()` caches this color after its
        // first evaluation.  The value is 218,232,250 normally and 194,247,159
        // on the source's April Fools branch.
        if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            self.axis_process_panel.set_background("rgb(194,247,159)");
        } else {
            self.axis_process_panel.set_background("rgb(218,232,250)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[test]
    fn constructor_preserves_source_superclass_arguments_and_call_order() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = SerialSectionsProcessPanel::new(manager, axis_progress_panel);
        let superclass = &panel.axis_process_panel;

        assert_eq!(superclass.axis_id, AxisID::Only);
        assert!(!superclass.popup_chunk_warnings);
        assert!(superclass.runnable_parallel);
        assert_eq!(superclass.interface_type, InterfaceType::SerialSections);
        assert!(!superclass.alt_parallel_loc);
        assert_eq!(superclass.panel_process_select_axis_label, None);
        assert!(superclass.panel_process_info_has_status);
    }

    #[test]
    fn show_both_axis_uses_serial_sections_colors_value() {
        let manager = DirectiveEditorManager::new(None, None, None, None);
        let axis_progress_panel = AxisProgressPanel::get_instance(Some(AxisID::Only), manager);
        let panel = SerialSectionsProcessPanel::new(manager, axis_progress_panel);
        let expected = if *crate::imod::etomo::util::utilities::APRIL_FOOLS {
            "rgb(194,247,159)"
        } else {
            "rgb(218,232,250)"
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
