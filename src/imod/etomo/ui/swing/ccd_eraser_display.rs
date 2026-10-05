//! `IMOD/Etomo/src/etomo/ui/swing/CcdEraserDisplay.java`.

use super::process_display::ProcessDisplay;
use crate::imod::etomo::comscript::ccd_eraser_param::CCDEraserParam;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;

/// Java `CcdEraserDisplay extends ProcessDisplay`.
pub trait CcdEraserDisplay: ProcessDisplay {
    /// Java `getParameters(CCDEraserParam, boolean)`.
    fn get_parameters(&self, ccd_eraser_params: &mut CCDEraserParam, do_validation: bool) -> bool;

    /// Java `getParameters(MakecomfileParam, boolean)`.
    fn get_parameters_makecomfile(&self, param: &mut MakecomfileParam, do_validation: bool)
    -> bool;
}
