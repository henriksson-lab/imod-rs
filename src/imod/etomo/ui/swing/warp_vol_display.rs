//! `IMOD/Etomo/src/etomo/ui/swing/WarpVolDisplay.java`.
#![allow(dead_code)]

use super::flatten_volume_panel::WarpVolParamBoundary;

/// Java `WarpVolDisplay`.
///
/// `FlattenVolumePanel` keeps its manager boundary explicit for this call, so
/// it is not falsely adapted to this manager-free contract.
pub trait WarpVolDisplay {
    fn get_parameters(&self, param: &mut WarpVolParamBoundary, do_validation: bool) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Display;
    impl WarpVolDisplay for Display {
        fn get_parameters(&self, param: &mut WarpVolParamBoundary, validation: bool) -> bool {
            param.output_size_z = "4".into();
            validation
        }
    }
    #[test]
    fn parameter_and_validation_are_preserved() {
        let mut param = WarpVolParamBoundary::default();
        assert!(Display.get_parameters(&mut param, true));
        assert_eq!(param.output_size_z, "4");
    }
}
