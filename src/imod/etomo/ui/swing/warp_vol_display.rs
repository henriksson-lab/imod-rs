//! `IMOD/Etomo/src/etomo/ui/swing/WarpVolDisplay.java`.

use crate::imod::etomo::comscript::warp_vol_param::WarpVolParam;

/// Java `WarpVolDisplay` (does not extend `ProcessDisplay`).
pub trait WarpVolDisplay {
    /// Java `getParameters(WarpVolParam, boolean)`.
    fn get_parameters(&self, param: &mut WarpVolParam, do_validation: bool) -> bool;
}
