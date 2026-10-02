//! `IMOD/Etomo/src/etomo/ui/swing/AltStackDisplay.java`.
//!
//! Java `public interface AltStackDisplay extends ProcessDisplay`: what the
//! manager reads from the Post Processing dialog's Alt Stack tab
//! (`AltStackPanel`) to update tilt.com and alttomosetup.com.
//!
//! The interface declares two `getParameters` overloads, so both carry the
//! parameter-type suffix (`ui.md` naming rule).

use super::process_display::ProcessDisplay;
use crate::imod::etomo::comscript::alt_tomo_setup_param::AltTomoSetupParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `interface AltStackDisplay extends ProcessDisplay`.
pub trait AltStackDisplay: ProcessDisplay {
    /// Java `getParameters(TiltParam)`.
    fn get_parameters_tilt_param(&self, tilt_param: &mut TiltParam) -> bool;

    /// Java `getAxisID()`: "return null for both axes option" (`None`).
    fn get_axis_id(&self) -> Option<AxisID>;

    /// Java `getParameters(AltTomoSetupParam, boolean)`.
    fn get_parameters_alt_tomo_setup_param_boolean(
        &self,
        param: &mut AltTomoSetupParam,
        do_validation: bool,
    ) -> bool;
}
