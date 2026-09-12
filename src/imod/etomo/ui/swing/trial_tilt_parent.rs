//! `IMOD/Etomo/src/etomo/ui/swing/TrialTiltParent.java`.
#![allow(dead_code)]

use super::trial_tilt_panel::{SplittiltParam, TrialTiltParam};

/// Java `TrialTiltParent.rcsid`.
pub const RCSID: &str = "$Id$";
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java package-private `TrialTiltParent`.
///
/// The Java checked exceptions are represented by the existing panel's boolean
/// parameter-boundary result; the two source overloads remain separate names.
pub trait TrialTiltParent {
    /// Java `getParameters(TiltParam, boolean)`.
    fn get_parameters_tilt(&self, tilt_param: &mut TrialTiltParam, do_validation: bool) -> bool;
    /// Java `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool;
    /// Java `getProcessingMethod()`.
    fn get_processing_method(&self) -> ProcessingMethod;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent;
    impl TrialTiltParent for Parent {
        fn get_parameters_tilt(&self, _: &mut TrialTiltParam, _: bool) -> bool {
            true
        }
        fn get_parameters_splittilt(&self, _: &mut SplittiltParam, _: bool) -> bool {
            true
        }
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::LocalCpu
        }
    }
    #[test]
    fn overloads_remain_separate_source_contract_methods() {
        let parent = Parent;
        assert!(parent.get_parameters_tilt(&mut TrialTiltParam::default(), true));
        assert!(parent.get_parameters_splittilt(&mut SplittiltParam, false));
        assert_eq!(parent.get_processing_method(), ProcessingMethod::LocalCpu);
    }
}
