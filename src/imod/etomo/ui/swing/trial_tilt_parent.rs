//! `IMOD/Etomo/src/etomo/ui/swing/TrialTiltParent.java`.
//!
//! Java package-private `interface TrialTiltParent`: what `TrialTiltPanel`
//! asks of the tilt panel that owns it (`AbstractTiltPanel`).  An EDT
//! interface: every method takes `&self`; `TrialTiltPanel` holds its parent
//! as a `Weak<dyn TrialTiltParent>`.

use super::tilt_display::TiltDisplayException;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `interface TrialTiltParent`.
pub trait TrialTiltParent {
    /// Java `getParameters(TiltParam, boolean) throws NumberFormatException,
    /// InvalidParameterException, IOException`.  The three Java exceptions
    /// are the variants of [`TiltDisplayException`].
    fn get_parameters_tilt_param_boolean(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException>;

    /// Java `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt_param_boolean(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool;

    /// Java `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod;
}
