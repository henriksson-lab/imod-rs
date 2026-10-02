//! `IMOD/Etomo/src/etomo/ui/swing/SirtsetupDisplay.java`.

use crate::imod::etomo::comscript::sirtsetup_param::SirtsetupParam;

/// Java `SirtsetupDisplay` (does not extend `ProcessDisplay`).
pub trait SirtsetupDisplay {
    /// Java `getParameters(SirtsetupParam, boolean)`.
    fn get_parameters(&self, param: &mut SirtsetupParam, do_validation: bool) -> bool;
}

// TODO(unit): SirtPanel.java implements SirtsetupDisplay (the Rust tree had it
// on `RadialPanel` through a boundary param); the impl waits for SirtPanel.
