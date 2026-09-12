//! `IMOD/Etomo/src/etomo/ui/swing/CoarseAlignDisplay.java`.
#![allow(dead_code)]

use super::{
    coarse_align_dialog::{CoarseAlignDialog, CoarseAlignMidasParam, CoarseAlignPrenewstPanel},
    process_display::ProcessDisplay,
};

/// Java `CoarseAlignDisplay`.
pub trait CoarseAlignDisplay<M: CoarseAlignMidasParam>: ProcessDisplay {
    /// Java `getCoarseAlignParameters(MidasParam)`.
    fn get_coarse_align_parameters(&self, param: &mut M);
}

impl<P: CoarseAlignPrenewstPanel> ProcessDisplay for CoarseAlignDialog<P> {}

impl<P: CoarseAlignPrenewstPanel, M: CoarseAlignMidasParam> CoarseAlignDisplay<M>
    for CoarseAlignDialog<P>
{
    fn get_coarse_align_parameters(&self, param: &mut M) {
        CoarseAlignDialog::get_coarse_align_parameters(self, param);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Display;
    impl ProcessDisplay for Display {}
    impl CoarseAlignMidasParam for i32 {
        fn set_binning(&mut self, value: i32) {
            *self = value;
        }
    }
    impl CoarseAlignDisplay<i32> for Display {
        fn get_coarse_align_parameters(&self, param: &mut i32) {
            *param = 4;
        }
    }

    #[test]
    fn coarse_align_contract_passes_the_original_parameter_by_mutable_reference() {
        let mut parameter = 0;
        Display.get_coarse_align_parameters(&mut parameter);
        assert_eq!(parameter, 4);
    }
}
