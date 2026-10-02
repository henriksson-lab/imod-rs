//! `IMOD/Etomo/src/etomo/ui/swing/ToolPanel.java`.
//!
//! Java package-private `interface ToolPanel`: a panel the Tools dialog
//! (`ToolsDialog`) can show.  Implemented by `FlattenVolumePanel`,
//! `GpuTiltTestPanel` and `AlignFramesPanel`.

use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `interface ToolPanel`.
pub trait ToolPanel {
    /// Java `Component getComponent()`.
    fn get_component(&self) -> Rc<JComponent>;
}
