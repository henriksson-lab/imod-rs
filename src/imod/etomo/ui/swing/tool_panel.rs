//! `IMOD/Etomo/src/etomo/ui/swing/ToolPanel.java`.
#![allow(dead_code)]

use super::abstract_frame::ComponentState;

/// Java package-private `ToolPanel` interface.
pub trait ToolPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> &ComponentState;
}
