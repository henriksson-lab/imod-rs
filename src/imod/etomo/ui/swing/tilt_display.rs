//! `IMOD/Etomo/src/etomo/ui/swing/TiltDisplay.java`.

use std::io;

use super::process_display::ProcessDisplay;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// Failures declared by Java `TiltDisplay.getParameters(TiltParam, boolean)`:
/// `NumberFormatException, InvalidParameterException, IOException`.
#[derive(Debug)]
pub enum TiltDisplayException {
    /// Java `NumberFormatException` (its message).
    NumberFormat(String),
    /// Java `etomo.util.InvalidParameterException`.
    InvalidParameter(InvalidParameterException),
    /// Java `IOException`.
    Io(io::Error),
}
impl std::fmt::Display for TiltDisplayException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumberFormat(value) => f.write_str(value),
            Self::InvalidParameter(value) => value.fmt(f),
            Self::Io(value) => value.fmt(f),
        }
    }
}
impl std::error::Error for TiltDisplayException {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidParameter(value) => Some(value),
            Self::Io(value) => Some(value),
            Self::NumberFormat(_) => None,
        }
    }
}

/// Java `TiltDisplay extends ProcessDisplay`.  Methods take `&self`;
/// implementing panels keep their state interior-mutable.
pub trait TiltDisplay: ProcessDisplay {
    /// Java `getParameters(TiltParam, boolean)`.
    fn get_parameters(
        &self,
        param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException>;

    /// Java `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool;

    /// Java `@Deprecated allowTiltComSave()`.
    fn allow_tilt_com_save(&self) -> bool;

    /// Java `setDebug(boolean)`.
    fn set_debug(&self, debug: bool);
}
