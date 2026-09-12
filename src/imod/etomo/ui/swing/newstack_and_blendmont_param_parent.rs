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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::{
        r#type::{axis_id::AxisID, dialog_type::DialogType},
        r#type::{etomo_number::EtomoNumber, view_type::ViewType},
    };
    struct Parent {
        panel: NewstackAndBlendmontParamPanel,
    }
    impl NewstackAndBlendmontParamParent for Parent {
        fn get_main_instance(&self) -> &NewstackAndBlendmontParamPanel {
            &self.panel
        }
        fn get_unbinned_bead_pixels(&self) -> ConstEtomoNumber {
            EtomoNumber::new().base
        }
        fn validate(&self) -> bool {
            true
        }
    }
    #[test]
    fn source_contract_keeps_main_instance_and_validation() {
        let parent = Parent {
            panel: NewstackAndBlendmontParamPanel::new(
                AxisID::Only,
                DialogType::FinalAlignedStack,
                ViewType::SingleView,
            ),
        };
        assert!(parent.validate());
        assert_eq!(parent.get_main_instance().axis_id, AxisID::Only);
        assert!(parent.get_unbinned_bead_pixels().is_null());
    }
}
