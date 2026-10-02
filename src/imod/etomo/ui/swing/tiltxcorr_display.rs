//! `IMOD/Etomo/src/etomo/ui/swing/TiltXcorrDisplay.java`.

use super::process_display::ProcessDisplay;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::imodchopconts_param::ImodchopcontsParam;
use crate::imod::etomo::comscript::tiltxcorr_param::TiltxcorrParam;
use crate::imod::etomo::r#type::panel_id::PanelId;

/// Java `TiltXcorrDisplay extends ProcessDisplay`.
pub trait TiltXcorrDisplay: ProcessDisplay {
    /// Java `getParameters(TiltxcorrParam, boolean) throws
    /// FortranInputSyntaxException`.
    fn get_parameters(
        &self,
        tilt_xcorr_params: &mut TiltxcorrParam,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException>;

    /// Java `getPanelId()`.
    fn get_panel_id(&self) -> PanelId;

    /// Java `getParameters(ImodchopcontsParam, boolean)`.
    fn get_parameters_imodchopconts(
        &self,
        param: &mut ImodchopcontsParam,
        do_validation: bool,
    ) -> bool;
}

// TODO(unit): TiltxcorrPanel.java implements TiltXcorrDisplay; the Rust
// `tiltxcorr_panel.rs` works through boundary traits, so the impl waits.
