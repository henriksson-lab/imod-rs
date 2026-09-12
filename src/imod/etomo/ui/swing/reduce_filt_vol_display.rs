//! `IMOD/Etomo/src/etomo/ui/swing/ReduceFiltVolDisplay.java`.
#![allow(dead_code)]

use super::squeeze_vol_panel::ReduceFiltVolParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;

/// Java `ReduceFiltVolDisplay`.
///
/// `SqueezeVolPanel` retains its `ApplicationManager` as an explicit method
/// input, so it cannot implement this narrower Java call without inventing
/// stored manager state.  The contract keeps the checked syntax failure exact.
pub trait ReduceFiltVolDisplay<P: ReduceFiltVolParam> {
    fn get_parameters(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException>;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parameter;
    impl ReduceFiltVolParam for Parameter {
        fn set_input_file(&mut self, _: String, _: bool) {}
        fn set_reduction_factor(&mut self, _: String) {}
        fn reset_reduction_factor(&mut self) {}
        fn set_z_reduction_factor(&mut self, _: String) {}
        fn reset_z_reduction_factor(&mut self) {}
        fn set_low_pass_radius_sigma(&mut self, _: String, _: bool) -> Option<String> {
            None
        }
        fn reset_low_pass_radius_sigma(&mut self) {}
        fn set_deconvolution_strength(&mut self, _: String) {}
        fn reset_deconvolution_strength(&mut self) {}
        fn set_snr_falloff(&mut self, _: String) {}
        fn reset_snr_falloff(&mut self) {}
        fn set_high_pass_nyquist(&mut self, _: String) {}
        fn reset_high_pass_nyquist(&mut self) {}
        fn set_defocus_in_microns(&mut self, _: String) {}
        fn reset_defocus_in_microns(&mut self) {}
        fn set_phase_shift(&mut self, _: String) {}
        fn reset_phase_shift(&mut self) {}
        fn set_mode_to_output(&mut self, _: String) {}
        fn set_setup_chunks_if_memory_error(&mut self, _: bool) {}
        fn set_output_file(&mut self, _: String) {}
    }
    struct Display;
    impl ReduceFiltVolDisplay<Parameter> for Display {
        fn get_parameters(
            &self,
            _: &mut Parameter,
            validation: bool,
        ) -> Result<bool, FortranInputSyntaxException> {
            Ok(validation)
        }
    }
    #[test]
    fn syntax_channel_and_validation_are_preserved() {
        assert!(Display.get_parameters(&mut Parameter, true).unwrap());
    }
}
