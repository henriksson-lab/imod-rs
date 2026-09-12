//! `IMOD/Etomo/src/etomo/ui/swing/FilterFullVolumeParent.java`.
//!
//! `ProcessInterface` has not yet become a separate Rust source unit.  Its
//! source-used processing-method query is retained directly in this parent
//! contract, rather than duplicating the interface inside its child panel.
#![allow(dead_code)]

use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java `FilterFullVolumeParent`, including its inherited
/// `ProcessInterface.getProcessingMethod` call made by
/// `FilterFullVolumePanel`.
pub trait FilterFullVolumeParent {
    /// Java `cleanUp`.
    fn clean_up(&mut self);

    /// Java `getVolume`.
    fn get_volume(&self) -> String;

    /// Java `initSubdir`.
    fn init_subdir(&mut self) -> bool;

    /// Java `isLoadWithFlipping`.
    fn is_load_with_flipping(&self) -> bool;

    /// Inherited Java `ProcessInterface.getProcessingMethod`.
    fn get_processing_method(&self) -> ProcessingMethod;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Parent;

    impl FilterFullVolumeParent for Parent {
        fn clean_up(&mut self) {}
        fn get_volume(&self) -> String {
            "volume.rec".into()
        }
        fn init_subdir(&mut self) -> bool {
            true
        }
        fn is_load_with_flipping(&self) -> bool {
            false
        }
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::PpCpu
        }
    }

    #[test]
    fn preserves_parent_and_inherited_process_contract() {
        let mut parent = Parent;
        assert!(parent.init_subdir());
        assert_eq!(parent.get_volume(), "volume.rec");
        assert!(!parent.is_load_with_flipping());
        assert_eq!(parent.get_processing_method(), ProcessingMethod::PpCpu);
        parent.clean_up();
    }
}
