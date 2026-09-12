//! `IMOD/Etomo/src/etomo/ui/swing/SmoothingAssessmentParent.java`.
#![allow(dead_code)]

use super::labeled_text_field::FieldValidationFailedException;

/// Java `SmoothingAssessmentParent.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `SmoothingAssessmentParent`.
pub trait SmoothingAssessmentParent {
    /// Java `isOneSurface()`.
    fn is_one_surface(&self) -> bool;
    /// Java `getWarpSpacingX(boolean)`.
    fn get_warp_spacing_x(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
    /// Java `getWarpSpacingY(boolean)`.
    fn get_warp_spacing_y(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent;
    impl SmoothingAssessmentParent for Parent {
        fn is_one_surface(&self) -> bool {
            true
        }
        fn get_warp_spacing_x(&self, _: bool) -> Result<String, FieldValidationFailedException> {
            Ok("8".into())
        }
        fn get_warp_spacing_y(&self, _: bool) -> Result<String, FieldValidationFailedException> {
            Ok("9".into())
        }
    }
    #[test]
    fn source_contract_retains_validation_argument() {
        let parent = Parent;
        assert!(parent.is_one_surface());
        assert_eq!(parent.get_warp_spacing_x(true).unwrap(), "8");
        assert_eq!(parent.get_warp_spacing_y(false).unwrap(), "9");
    }
}
