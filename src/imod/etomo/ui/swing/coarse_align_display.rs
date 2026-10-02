//! `IMOD/Etomo/src/etomo/ui/swing/CoarseAlignDisplay.java`.

use super::process_display::ProcessDisplay;
use crate::imod::etomo::comscript::midas_param::MidasParam;

/// Java `CoarseAlignDisplay extends ProcessDisplay`.
pub trait CoarseAlignDisplay: ProcessDisplay {
    /// Java `getCoarseAlignParameters(MidasParam)`.
    fn get_coarse_align_parameters(&self, param: &mut MidasParam);
}

