//! `IMOD/Etomo/src/etomo/ui/swing/NewstackDisplay.java`.

use std::io;

use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// The checked exceptions declared by Java `NewstackDisplay.getParameters`.
#[derive(Debug)]
pub enum NewstackDisplayException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `etomo.util.InvalidParameterException`.
    InvalidParameterException(InvalidParameterException),
    /// Java `IOException`.
    Io(io::Error),
}

impl std::fmt::Display for NewstackDisplayException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FortranInputSyntaxException(exception) => exception.fmt(formatter),
            Self::InvalidParameterException(exception) => exception.fmt(formatter),
            Self::Io(exception) => exception.fmt(formatter),
        }
    }
}

impl std::error::Error for NewstackDisplayException {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FortranInputSyntaxException(exception) => Some(exception),
            Self::InvalidParameterException(exception) => Some(exception),
            Self::Io(exception) => Some(exception),
        }
    }
}

/// Java `NewstackDisplay` (does not extend `ProcessDisplay`).  Methods take
/// `&self`; implementing panels keep their widgets interior-mutable.
pub trait NewstackDisplay {
    /// Java `getParameters(NewstParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`.
    fn get_parameters(
        &self,
        newst_param: &mut NewstParam,
        do_validation: bool,
    ) -> Result<bool, NewstackDisplayException>;

    /// Java `setParameters(ConstNewstParam)`.
    fn set_parameters(&self, param: &dyn ConstNewstParam);

    /// Java `validate()`.
    fn validate(&self) -> bool;

    /// Java `isFiducialess()`.
    fn is_fiducialess(&self) -> bool;
}

// TODO(unit): Newstack3dFindPanel.java and NewstackOrBlendmontPanel.java
// implement NewstackDisplay (the Rust tree had it on `NewstackPanel` through a
// local `NewstParam` boundary); the impls wait for faithful panel units.
