//! `IMOD/Etomo/src/etomo/ui/swing/SubtomoSetupDisplay.java`.

use crate::imod::etomo::comscript::subtomo_setup_param::SubtomoSetupParam;

/// Java `SubtomoSetupDisplay` (does not extend `ProcessDisplay`).
pub trait SubtomoSetupDisplay {
    /// Java `getParameters(SubtomoSetupParam, boolean)`.
    fn get_parameters(&self, param: &mut SubtomoSetupParam, do_validation: bool) -> bool;
}
