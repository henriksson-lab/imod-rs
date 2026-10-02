//! `IMOD/Etomo/src/etomo/ui/swing/NewstackAndBlendmontParamParent.java`.
#![allow(dead_code)]

use super::newstack_and_blendmont_param_panel::NewstackAndBlendmontParamPanel;

/// Java `NewstackAndBlendmontParamParent.rcsid`.
pub const RCSID: &str = "$Id$";
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;

/// Java package-private `NewstackAndBlendmontParamParent`.
pub trait NewstackAndBlendmontParamParent {
    /// Java `getMainInstance()`.
    fn get_main_instance(&self) -> &NewstackAndBlendmontParamPanel;
    /// Java `getUnbinnedBeadPixels()`.
    fn get_unbinned_bead_pixels(&self) -> ConstEtomoNumber;
    /// Java `validate()`.
    fn validate(&self) -> bool;
}
