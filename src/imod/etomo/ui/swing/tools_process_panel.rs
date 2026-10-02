//! `IMOD/Etomo/src/etomo/ui/swing/ToolsProcessPanel.java`.
//!
//! The axis process panel of the Tools interface (`ToolsManager`).  Extends
//! [`AxisProcessPanel`] (held as `base`, dereffed to) and implements
//! [`AxisProcessPanelVirtual`]; its only override is `showBothAxis`, which
//! sets the Tools background colour.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `final class ToolsProcessPanel extends AxisProcessPanel`.
pub struct ToolsProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for ToolsProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl ToolsProcessPanel {
    /// Java package-private constructor
    /// `ToolsProcessPanel(ToolsManager, AxisProgressPanel)`.
    pub fn new(
        manager: &'static ToolsManager,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<ToolsProcessPanel> {
        // super(AxisID.ONLY, manager, true, true, InterfaceType.TOOLS, false,
        //   axisProgressPanel)
        let this = Rc::new(ToolsProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                true,
                true,
                InterfaceType::Tools,
                false,
                axis_progress_panel,
            ),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn AxisProcessPanelVirtual>);
        this.base.create_process_control_panel();
        this.base.show_both_axis();
        this.base.initialize_panels();
        this
    }
}

impl AxisProcessPanelVirtual for ToolsProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `showBothAxis()` override.
    fn show_both_axis(&self) {
        // Swing painting: setBackground(Colors.getBackgroundTools()).
    }
}
