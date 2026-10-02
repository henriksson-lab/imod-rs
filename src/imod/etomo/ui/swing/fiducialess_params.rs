//! `IMOD/Etomo/src/etomo/ui/swing/FiducialessParams.java`.
//!
//! Specifies the fiducialess UI parameters.  Implemented by
//! `CoarseAlignDialog` and `NewstackAndBlendmontParamPanel`.

use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public interface FiducialessParams`.  Implementations are Swing-side
/// objects (`Rc`, interior mutability), so the methods take `&self`.
pub trait FiducialessParams {
    /// Java `isFiducialess()` (FiducialessParams.java:41).
    fn is_fiducialess(&self) -> bool;

    /// Java `getImageRotation(boolean) throws FieldValidationFailedException`
    /// (FiducialessParams.java:43-44).
    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
}
