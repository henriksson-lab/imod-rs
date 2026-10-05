//! `IMOD/Etomo/src/etomo/ui/swing/DirectiveEditorProcessPanel.java`.
//!
//! The axis process panel of the directive editor (`DirectiveEditorManager`).  Extends
//! [`AxisProcessPanel`] (held as `base`, dereffed to) and implements
//! [`AxisProcessPanelVirtual`]; its only override is `showBothAxis`, which sets the
//! tools background colour.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::axis_process_panel::{AxisProcessPanel, AxisProcessPanelVirtual};
use super::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;

/// Java package-private `final class DirectiveEditorProcessPanel extends
/// AxisProcessPanel`.
pub struct DirectiveEditorProcessPanel {
    /// The Java superclass part.
    base: Rc<AxisProcessPanel>,
}

impl Deref for DirectiveEditorProcessPanel {
    type Target = AxisProcessPanel;
    fn deref(&self) -> &AxisProcessPanel {
        &self.base
    }
}

impl DirectiveEditorProcessPanel {
    /// Java package-private constructor `DirectiveEditorProcessPanel(
    /// DirectiveEditorManager, InterfaceType, AxisProgressPanel)`.
    pub fn new(
        manager: &'static DirectiveEditorManager,
        interface_type: InterfaceType,
        axis_progress_panel: Rc<AxisProgressPanel>,
    ) -> Rc<DirectiveEditorProcessPanel> {
        // super(AxisID.ONLY, manager, true, true, interfaceType, false,
        //   axisProgressPanel)
        let this = Rc::new(DirectiveEditorProcessPanel {
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

impl AxisProcessPanelVirtual for DirectiveEditorProcessPanel {
    fn axis_process_panel(&self) -> &AxisProcessPanel {
        &self.base
    }

    /// Java package-private `showBothAxis()` override.
    fn show_both_axis(&self) {
        // Swing painting: setBackground(Colors.getBackgroundTools()).
    }
}
