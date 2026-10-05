//! `IMOD/Etomo/src/etomo/ui/swing/FindBeads3dDisplay.java`.

use crate::imod::etomo::comscript::find_beads3d_param::FindBeads3dParam;

/// Java `FindBeads3dDisplay` (does not extend `ProcessDisplay`).
pub trait FindBeads3dDisplay {
    /// Java `getParameters(FindBeads3dParam, boolean)`.
    fn get_parameters(&self, param: &mut FindBeads3dParam, do_validation: bool) -> bool;

    /// Java `isFiducialess()`.
    fn is_fiducialess(&self) -> bool;
}
