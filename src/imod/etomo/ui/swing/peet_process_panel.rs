//! `IMOD/Etomo/src/etomo/ui/swing/PeetProcessPanel.java`.
//!
//! The axis process panel of the PEET interface.  Extends [`AxisProcessPanel`]
//! (held as `base`, dereffed to) and implements [`AxisProcessPanelVirtual`]; it
//! constructs the superclass with popupChunkWarnings false and overrides nothing.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `class PeetProcessPanel extends AxisProcessPanel`.
pub struct PeetProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for PeetProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl PeetProcessPanel {
    /// Java package-private `PeetProcessPanel(BaseManager, AxisProgressPanel)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<PeetProcessPanel> {
        // super(AxisID.ONLY, manager, false, true, InterfaceType.PEET, false,
        //   axisProgressPanel)
        let this = Rc::new(PeetProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                false,
                true,
                InterfaceType::Peet,
                false,
                axis_progress_panel,
            ),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn AxisProcessPanelVirtual>);
        this.base.create_process_control_panel();
        this.base.initialize_panels();
        this
    }
}

impl AxisProcessPanelVirtual for PeetProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }
}
