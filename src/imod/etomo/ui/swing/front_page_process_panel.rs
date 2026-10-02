//! `IMOD/Etomo/src/etomo/ui/swing/FrontPageProcessPanel.java`.
//!
//! The axis process panel of the front page (`FrontPageManager`).  Extends
//! [`AxisProcessPanel`] (held as `base`, dereffed to) and implements
//! [`AxisProcessPanelVirtual`]; its only override is an empty
//! `createProcessControlPanel`.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::front_page_manager::FrontPageManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private final class `FrontPageProcessPanel extends
/// AxisProcessPanel`.
pub struct FrontPageProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for FrontPageProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl FrontPageProcessPanel {
    /// Java package-private constructor
    /// `FrontPageProcessPanel(FrontPageManager, AxisProgressPanel)`.
    pub fn new(
        manager: &'static FrontPageManager,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<FrontPageProcessPanel> {
        // super(AxisID.ONLY, manager, true, true, InterfaceType.FRONT_PAGE, false,
        //   axisProgressPanel)
        let this = Rc::new(FrontPageProcessPanel {
            base: AxisProcessPanel::new(
                AxisID::Only,
                manager,
                true,
                true,
                InterfaceType::FrontPage,
                false,
                axis_progress_panel,
            ),
        });
        this.base
            .set_this(Rc::downgrade(&this) as Weak<dyn AxisProcessPanelVirtual>);
        this.base.initialize_panels();
        this
    }
}

impl AxisProcessPanelVirtual for FrontPageProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `createProcessControlPanel()` override: empty.
    fn create_process_control_panel(&self) {}
}
