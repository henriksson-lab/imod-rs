//! `IMOD/Etomo/src/etomo/ui/swing/SerialSectionsProcessPanel.java`.
//!
//! The axis process panel of the Serial Sections interface (`SerialSectionsManager`).
//! Extends [`AxisProcessPanel`] (held as `base`, dereffed to) and implements
//! [`AxisProcessPanelVirtual`]; its only override is `showBothAxis`, which sets the
//! serial sections background colour.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `final class SerialSectionsProcessPanel extends
/// AxisProcessPanel`.
pub struct SerialSectionsProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for SerialSectionsProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl SerialSectionsProcessPanel {
    /// Java package-private constructor
    /// `SerialSectionsProcessPanel(BaseManager, AxisProgressPanel)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<SerialSectionsProcessPanel> {
        // super(AxisID.ONLY, manager, false, true, InterfaceType.SERIAL_SECTIONS,
        //   false, axisProgressPanel)
        let this = Rc::new(SerialSectionsProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                false,
                true,
                InterfaceType::SerialSections,
                false,
                axis_progress_panel,
            ),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn AxisProcessPanelVirtual>);
        this.base.create_process_control_panel();
        this.show_both_axis();
        this.base.initialize_panels();
        this
    }
}

impl AxisProcessPanelVirtual for SerialSectionsProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `showBothAxis()` override.
    fn show_both_axis(&self) {
        // Swing painting: setBackground(Colors.getBackgroundSerialSections()).
    }
}
