//! `IMOD/Etomo/src/etomo/ui/swing/FlattenWarpDisplay.java`.

use crate::imod::etomo::comscript::flatten_warp_param::FlattenWarpParam;

/// Java `FlattenWarpDisplay` (does not extend `ProcessDisplay`).
pub trait FlattenWarpDisplay {
    /// Java `getParameters(FlattenWarpParam, boolean)`.
    fn get_parameters(&self, param: &mut FlattenWarpParam, do_validation: bool) -> bool;
}
