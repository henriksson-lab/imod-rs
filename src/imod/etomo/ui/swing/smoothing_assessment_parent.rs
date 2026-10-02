//! `IMOD/Etomo/src/etomo/ui/swing/SmoothingAssessmentParent.java`.
//!
//! Java package-private `interface SmoothingAssessmentParent`: what
//! `SmoothingAssessmentPanel` reads from the panel that contains it
//! (`FlattenVolumePanel`).

use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `interface SmoothingAssessmentParent`.
pub trait SmoothingAssessmentParent {
    /// Java `isOneSurface()`.
    fn is_one_surface(&self) -> bool;

    /// Java `getWarpSpacingX(boolean) throws FieldValidationFailedException`.
    /// `None` is Java null.
    fn get_warp_spacing_x(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getWarpSpacingY(boolean) throws FieldValidationFailedException`.
    /// `None` is Java null.
    fn get_warp_spacing_y(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;
}
