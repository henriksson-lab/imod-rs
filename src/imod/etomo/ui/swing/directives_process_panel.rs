//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesProcessPanel.java`.
//!
//! An axis process panel for a directives interface.  Extends [`AxisProcessPanel`]
//! (held as `base`, dereffed to) and implements [`AxisProcessPanelVirtual`] with no
//! overrides.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `final class DirectivesProcessPanel extends AxisProcessPanel`.
pub struct DirectivesProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for DirectivesProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl DirectivesProcessPanel {
    /// Java package-private constructor `DirectivesProcessPanel(BaseManager,
    /// InterfaceType, AxisProgressPanel)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        interface_type: InterfaceType,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<DirectivesProcessPanel> {
        // super(AxisID.ONLY, manager, true, true, interfaceType, false,
        //   axisProgressPanel)
        let this = Rc::new(DirectivesProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                true,
                true,
                interface_type,
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

impl AxisProcessPanelVirtual for DirectivesProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }
}
