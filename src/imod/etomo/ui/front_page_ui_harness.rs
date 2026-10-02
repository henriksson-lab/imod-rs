//! `IMOD/Etomo/src/etomo/ui/FrontPageUIHarness.java`.
//!
//! The `FrontPageManager`'s dialog expert: opens the `FrontPageDialog` and
//! runs the reconstruction automation.
//!
//! The manager holds this object (Java `dialogExpert`), and a manager is
//! shared across threads, so the struct is `Send + Sync`; the dialog it holds
//! is a Swing object and lives in an [`EdtCell`] (event dispatch thread
//! only), as a manager's own dialogs do.

use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::front_page_dialog::FrontPageDialog;
use crate::imod::etomo::util::event_queue::EdtCell;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java public final class `FrontPageUIHarness`.
pub struct FrontPageUIHarness {
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private `dialog`, initialised to null.
    dialog: EdtCell<Rc<FrontPageDialog>>,
}

impl FrontPageUIHarness {
    /// Java `FrontPageUIHarness(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> FrontPageUIHarness {
        FrontPageUIHarness {
            manager,
            axis_id,
            dialog: EdtCell::new(),
        }
    }

    /// Java `openDialog()`.
    pub fn open_dialog(&self) {
        if !self.dialog.is_some() && !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            let dialog = FrontPageDialog::get_instance(self.manager, self.axis_id);
            self.dialog.set(Some(dialog.clone()));
            dialog.show();
        }
    }

    /// Java `reconActionForAutomation()`.
    pub fn recon_action_for_automation(&self) {
        etomo_director::INSTANCE.open_tomogram_and_do_automation(true, Some(self.axis_id), None);
    }
}
