//! `IMOD/Etomo/src/etomo/ui/swing/BeadTrackDisplay.java`.

use crate::imod::etomo::comscript::beadtrack_param::BeadtrackParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::r#type::invalid_etomo_number_exception::InvalidEtomoNumberException;

/// The two checked exceptions on Java `BeadTrackDisplay.getParameters`.
#[derive(Debug)]
pub enum BeadTrackDisplayException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `InvalidEtomoNumberException`.
    InvalidEtomoNumberException(InvalidEtomoNumberException),
}

impl std::fmt::Display for BeadTrackDisplayException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FortranInputSyntaxException(exception) => exception.fmt(formatter),
            Self::InvalidEtomoNumberException(exception) => exception.fmt(formatter),
        }
    }
}

impl std::error::Error for BeadTrackDisplayException {}

/// Java `BeadTrackDisplay` (does not extend `ProcessDisplay`).
pub trait BeadTrackDisplay {
    /// Java `getParameters(BeadtrackParam, boolean) throws
    /// FortranInputSyntaxException, InvalidEtomoNumberException`.
    fn get_parameters(
        &self,
        beadtrack_params: &mut BeadtrackParam,
        do_validation: bool,
    ) -> Result<bool, BeadTrackDisplayException>;
}

// TODO(unit): BeadtrackPanel.java implements BeadTrackDisplay; the Rust
// `beadtrack_panel.rs` takes a boundary param and a manager argument, so the
// impl waits for that unit.
