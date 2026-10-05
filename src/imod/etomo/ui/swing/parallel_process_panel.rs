//! `IMOD/Etomo/src/etomo/ui/swing/ParallelProcessPanel.java`.
//!
//! The axis process panel of the generic parallel process and anisotropic
//! diffusion interfaces (`ParallelManager`).  Extends [`AxisProcessPanel`] (held as
//! `base`, dereffed to) and implements [`AxisProcessPanelVirtual`]; its only
//! override is `showBothAxis`, which sets the parallel background colour.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `final class ParallelProcessPanel extends AxisProcessPanel`.
pub struct ParallelProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for ParallelProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl ParallelProcessPanel {
    /// Java package-private constructor
    /// `ParallelProcessPanel(ParallelManager, AxisProgressPanel)`.
    pub fn new(
        manager: &'static ParallelManager,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<ParallelProcessPanel> {
        // super(AxisID.ONLY, manager, true, true, InterfaceType.PP, false,
        //   axisProgressPanel)
        let this = Rc::new(ParallelProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                true,
                true,
                InterfaceType::Pp,
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

impl AxisProcessPanelVirtual for ParallelProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `showBothAxis()` override.
    fn show_both_axis(&self) {
        // Swing painting: setBackground(Colors.getBackgroundParallel()).
    }
}
