//! `IMOD/Etomo/src/etomo/ui/swing/BlendmontDisplay.java`.

use std::io;

use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// The checked exceptions declared by Java `BlendmontDisplay.getParameters`.
#[derive(Debug)]
pub enum BlendmontDisplayException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `etomo.util.InvalidParameterException`.
    InvalidParameterException(InvalidParameterException),
    /// Java `IOException`.
    Io(io::Error),
}

impl std::fmt::Display for BlendmontDisplayException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FortranInputSyntaxException(exception) => exception.fmt(formatter),
            Self::InvalidParameterException(exception) => exception.fmt(formatter),
            Self::Io(exception) => exception.fmt(formatter),
        }
    }
}

impl std::error::Error for BlendmontDisplayException {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FortranInputSyntaxException(exception) => Some(exception),
            Self::InvalidParameterException(exception) => Some(exception),
            Self::Io(exception) => Some(exception),
        }
    }
}

/// Java `BlendmontDisplay` (does not extend `ProcessDisplay`).  Methods take
/// `&self`; implementing panels keep their widgets interior-mutable.
pub trait BlendmontDisplay {
    /// Java `getParameters(BlendmontParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`.
    fn get_parameters(
        &self,
        blendmont_param: &mut BlendmontParam,
        do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException>;

    /// Java `setParameters(BlendmontParam)`.
    fn set_parameters(&self, param: &BlendmontParam);

    /// Java `validate()`.
    fn validate(&self) -> bool;

    /// Java `isFiducialess()`.
    fn is_fiducialess(&self) -> bool;
}
