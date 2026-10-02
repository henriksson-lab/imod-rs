//! `IMOD/Etomo/src/etomo/PeetManager.java` (partial).
//!
//! TODO(unit): the rest of `PeetManager` (its constructor, dialog handling and
//! the PEET process launches, PeetManager.java:1-949) is not translated yet, and
//! neither are `PeetDialog` and `PeetMetaData`; only the static interface check
//! the front page and the menu call is here.

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::environment_variable;

/// Java final `PeetManager`: only its static members exist so far.
pub struct PeetManager;

impl PeetManager {
    /// Java static `isInterfaceAvailable()` (PeetManager.java:171).
    pub fn is_interface_available() -> bool {
        if !environment_variable::INSTANCE.exists(
            None,
            etomo_director::INSTANCE.get_original_user_dir().as_deref(),
            environment_variable::PARTICLE_DIR,
            Some(AxisID::Only),
        ) {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string(
                    None,
                    "PEET is an optional package for particle averaging, which has not been installed and correctly configured.  See the PEET link under Other Programs at http://bio3d.colorado.edu/.",
                    "Interface Unavailable",
                )
            });
            return false;
        }
        true
    }
}
