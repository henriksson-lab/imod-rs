//! `IMOD/Etomo/src/etomo/ui/swing/TiltXcorrDisplay.java`.
#![allow(dead_code)]

use super::{
    process_display::ProcessDisplay,
    tiltxcorr_panel::{ImodchopcontsParam, PanelId, TiltxcorrParam},
};
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;

/// Java `TiltXcorrDisplay`.
///
/// The current `TiltxcorrPanel` carries its metadata read boundary explicitly,
/// so no implementation is added until that source dependency is owned by it.
pub trait TiltXcorrDisplay<T: TiltxcorrParam, I: ImodchopcontsParam>: ProcessDisplay {
    fn get_parameters(
        &self,
        param: &mut T,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException>;
    fn get_panel_id(&self) -> PanelId;
    fn get_imodchopconts_parameters(&self, param: &mut I, do_validation: bool) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Tilt;
    impl TiltxcorrParam for Tilt {
        fn set_value(&mut self, _: &str, _: String) -> Result<(), String> {
            Ok(())
        }
        fn reset_value(&mut self, _: &str) {}
        fn set_flag(&mut self, _: &str, _: bool) {}
        fn set_iterate_correlations(&mut self, _: i32) -> Option<String> {
            None
        }
    }
    struct Chop;
    impl ImodchopcontsParam for Chop {
        fn minimum_overlap(&self) -> Option<String> {
            None
        }
        fn length_of_pieces_is_null(&self) -> bool {
            true
        }
        fn length_of_pieces_is_default(&self) -> bool {
            false
        }
        fn length_of_pieces(&self) -> Option<String> {
            None
        }
        fn set_minimum_overlap(&mut self, _: String) {}
        fn set_length_of_pieces_default(&mut self) {}
        fn set_length_of_pieces(&mut self, _: String) {}
        fn reset_length_of_pieces(&mut self) {}
    }
    struct Display;
    impl ProcessDisplay for Display {}
    impl TiltXcorrDisplay<Tilt, Chop> for Display {
        fn get_parameters(
            &self,
            _: &mut Tilt,
            v: bool,
        ) -> Result<bool, FortranInputSyntaxException> {
            Ok(v)
        }
        fn get_panel_id(&self) -> PanelId {
            PanelId::CrossCorrelation
        }
        fn get_imodchopconts_parameters(&self, _: &mut Chop, v: bool) -> bool {
            v
        }
    }
    #[test]
    fn contract_retains_both_parameter_types_and_panel_identity() {
        assert!(Display.get_parameters(&mut Tilt, true).unwrap());
        assert!(Display.get_imodchopconts_parameters(&mut Chop, true));
        assert_eq!(Display.get_panel_id(), PanelId::CrossCorrelation);
    }
}
