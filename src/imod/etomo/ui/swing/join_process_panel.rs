//! `IMOD/Etomo/src/etomo/ui/swing/JoinProcessPanel.java`.
//!
//! The axis process panel of the Join interface (`JoinManager`).  Extends
//! [`AxisProcessPanel`] (held as `base`, dereffed to) and implements
//! [`AxisProcessPanelVirtual`]; its only override is `showBothAxis`, which
//! sets the Join background colour.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `class JoinProcessPanel extends AxisProcessPanel`.
pub struct JoinProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for JoinProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl JoinProcessPanel {
    /// Java package-private constructor
    /// `JoinProcessPanel(JoinManager, AxisID, AxisProgressPanel)`.
    pub fn new(
        join_manager: &'static JoinManager,
        axis: AxisID,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<JoinProcessPanel> {
        // super(axis, joinManager, true, true, InterfaceType.JOIN, false,
        //   axisProgressPanel)
        let this = Rc::new(JoinProcessPanel {
            base: AxisProcessPanel::new(
                axis,
                join_manager,
                true,
                true,
                InterfaceType::Join,
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

impl AxisProcessPanelVirtual for JoinProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `showBothAxis()` override.
    fn show_both_axis(&self) {
        // Swing painting: setBackground(Colors.getBackgroundJoin()).
    }
}
