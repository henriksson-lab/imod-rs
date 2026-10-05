//! `IMOD/Etomo/src/etomo/ui/swing/FrameInterface.java`.
//!
//! A package-private interface no class in the Java implements; translated as the
//! declaration only.

use crate::imod::etomo::jdk::ActionEvent;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `interface FrameInterface`.
pub trait FrameInterface {
    /// Java `menuFileAction(ActionEvent)`.
    fn menu_file_action(&self, action_event: Option<&ActionEvent>);
    /// Java `menuToolsAction(ActionEvent)`.
    fn menu_tools_action(&self, action_event: Option<&ActionEvent>);
    /// Java `menuViewAction(ActionEvent)`.
    fn menu_view_action(&self, action_event: Option<&ActionEvent>);
    /// Java `menuOptionsAction(ActionEvent)`.
    fn menu_options_action(&self, action_event: Option<&ActionEvent>);
    /// Java `menuHelpAction(ActionEvent)`.
    fn menu_help_action(&self, action_event: Option<&ActionEvent>);
    /// Java `repaint()`.
    fn repaint_void(&self);
    /// Java `pack(boolean)`.
    fn pack_boolean(&self, force: bool);
    /// Java `repaint(AxisID)`.
    fn repaint_axis_id(&self, axis_id: Option<AxisID>);
    /// Java `pack(AxisID)`.
    fn pack_axis_id(&self, axis_id: Option<AxisID>);
}
