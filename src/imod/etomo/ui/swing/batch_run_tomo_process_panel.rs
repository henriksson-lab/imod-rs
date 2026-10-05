//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoProcessPanel.java`.
//!
//! The axis process panel of the batchruntomo interface (`BatchRunTomoManager`).
//! Extends [`AxisProcessPanel`] (held as `base`, dereffed to) and implements
//! [`AxisProcessPanelVirtual`]; its only override is `showBothAxis`, which sets the
//! batchruntomo background colour.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `final class BatchRunTomoProcessPanel extends
/// AxisProcessPanel`.
pub struct BatchRunTomoProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for BatchRunTomoProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl BatchRunTomoProcessPanel {
    /// Java package-private constructor `BatchRunTomoProcessPanel(BaseManager,
    /// InterfaceType, AxisProgressPanel)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        interface_type: InterfaceType,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<BatchRunTomoProcessPanel> {
        // super(AxisID.ONLY, manager, true, false, interfaceType, true,
        //   axisProgressPanel)
        let this = Rc::new(BatchRunTomoProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                true,
                false,
                interface_type,
                true,
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

impl AxisProcessPanelVirtual for BatchRunTomoProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `showBothAxis()` override.
    fn show_both_axis(&self) {
        // Swing painting: setBackground(Colors.getBackgroundBatchruntomo()).
    }
}
