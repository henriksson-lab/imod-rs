//! `IMOD/Etomo/src/etomo/ui/swing/ReduceFiltVolDisplay.java`.

use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::reduce_filt_vol_param::ReduceFiltVolParam;

/// Java `ReduceFiltVolDisplay` (does not extend `ProcessDisplay`).
pub trait ReduceFiltVolDisplay {
    /// Java `getParameters(ReduceFiltVolParam, boolean) throws
    /// FortranInputSyntaxException`.
    fn get_parameters(
        &self,
        param: &mut ReduceFiltVolParam,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException>;
}
