//! `IMOD/Etomo/src/etomo/ui/swing/FrameInterface.java`.
#![allow(dead_code)]

use crate::imod::etomo::r#type::axis_id::AxisID;

use super::etomo_frame::{ActionEvent, EtomoFrame};

/// Java package-private `FrameInterface`.
pub trait FrameInterface {
    /// Java `menuFileAction(ActionEvent)`.
    fn menu_file_action(&mut self, action_event: &ActionEvent);
    /// Java `menuToolsAction(ActionEvent)`.
    fn menu_tools_action(&mut self, action_event: &ActionEvent);
    /// Java `menuViewAction(ActionEvent)`.
    fn menu_view_action(&mut self, action_event: &ActionEvent);
    /// Java `menuOptionsAction(ActionEvent)`.
    fn menu_options_action(&mut self, action_event: &ActionEvent);
    /// Java `menuHelpAction(ActionEvent)`.
    fn menu_help_action(&mut self, action_event: &ActionEvent);
    /// Java `repaint()`.
    fn repaint(&mut self);
    /// Java `pack(boolean)`.
    fn pack(&mut self, force: bool);
    /// Java `repaint(AxisID)`.
    fn repaint_axis(&mut self, axis_id: AxisID);
    /// Java `pack(AxisID)`.
    fn pack_axis(&mut self, axis_id: AxisID);
}

impl FrameInterface for EtomoFrame {
    fn menu_file_action(&mut self, action_event: &ActionEvent) {
        let _ = EtomoFrame::menu_file_action(self, action_event);
    }
    fn menu_tools_action(&mut self, action_event: &ActionEvent) {
        let _ = EtomoFrame::menu_tools_action(self, action_event);
    }
    fn menu_view_action(&mut self, action_event: &ActionEvent) {
        let _ = EtomoFrame::menu_view_action(self, action_event);
    }
    fn menu_options_action(&mut self, action_event: &ActionEvent) {
        let _ = EtomoFrame::menu_options_action(self, action_event);
    }
    fn menu_help_action(&mut self, action_event: &ActionEvent) {
        let _ = EtomoFrame::menu_help_action(self, action_event);
    }
    fn repaint(&mut self) {
        EtomoFrame::repaint(self, AxisID::Only);
    }
    fn pack(&mut self, _force: bool) {
        EtomoFrame::pack(self);
    }
    fn repaint_axis(&mut self, axis_id: AxisID) {
        EtomoFrame::repaint(self, axis_id);
    }
    fn pack_axis(&mut self, axis_id: AxisID) {
        EtomoFrame::pack_axis(self, axis_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etomo_frame_implements_frame_interface_presentation_calls() {
        let mut frame = EtomoFrame::new();
        FrameInterface::repaint(&mut frame);
        FrameInterface::repaint_axis(&mut frame, AxisID::First);
        FrameInterface::pack(&mut frame, true);
        FrameInterface::pack_axis(&mut frame, AxisID::Second);
        assert_eq!(frame.presentation.repaint_count, 2);
        assert!(frame.presentation.packed);
    }
}
