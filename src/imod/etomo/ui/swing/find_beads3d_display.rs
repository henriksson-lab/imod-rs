//! `IMOD/Etomo/src/etomo/ui/swing/FindBeads3dDisplay.java`.

use crate::imod::etomo::comscript::find_beads3d_param::FindBeads3dParam;

/// Java `FindBeads3dDisplay` (does not extend `ProcessDisplay`).
pub trait FindBeads3dDisplay {
    /// Java `getParameters(FindBeads3dParam, boolean)`.
    fn get_parameters(&self, param: &mut FindBeads3dParam, do_validation: bool) -> bool;

    /// Java `isFiducialess()`.
    fn is_fiducialess(&self) -> bool;
}

// TODO(unit): FindBeads3dPanel.java implements FindBeads3dDisplay; the Rust
// `find_beads3d_panel.rs` takes manager/parent arguments and a boundary param,
// so the impl waits for that unit.
