//! `IMOD/Etomo/src/etomo/ui/swing/FiducialessParams.java`.
//!
//! This source interface is the shared boundary through which eTomo obtains
//! the coarse-alignment-only state and its validated image rotation.

use super::labeled_text_field::FieldValidationFailedException;

/// Java `FiducialessParams`.
pub trait FiducialessParams {
    /// Java `isFiducialess`.
    fn is_fiducialess(&self) -> bool;

    /// Java `getImageRotation(boolean)`.
    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Params;

    impl FiducialessParams for Params {
        fn is_fiducialess(&self) -> bool {
            true
        }

        fn get_image_rotation(
            &self,
            do_validation: bool,
        ) -> Result<String, FieldValidationFailedException> {
            if do_validation {
                Ok("12.5".into())
            } else {
                Ok(String::new())
            }
        }
    }

    #[test]
    fn interface_keeps_both_java_operations() {
        let params: &dyn FiducialessParams = &Params;
        assert!(params.is_fiducialess());
        assert_eq!(params.get_image_rotation(true).unwrap(), "12.5");
    }
}
