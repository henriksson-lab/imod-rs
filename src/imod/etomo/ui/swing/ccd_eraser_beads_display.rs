//! `IMOD/Etomo/src/etomo/ui/swing/CcdEraserBeadsDisplay.java`.
//!
//! No class in the vendored tree implements this interface.

use super::process_display::ProcessDisplay;
use crate::imod::etomo::comscript::ccd_eraser_param::CCDEraserParam;

/// Java `CcdEraserBeadsDisplay extends ProcessDisplay`.
pub trait CcdEraserBeadsDisplay: ProcessDisplay {
    /// Java `getParameters(CCDEraserParam)`.
    fn get_parameters(&self, ccd_eraser_params: &mut CCDEraserParam) -> bool;
}
