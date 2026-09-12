//! `IMOD/Etomo/src/etomo/ui/swing/Tilt3dFindParent.java`.
#![allow(dead_code)]

use super::{
    beads3d_find_panel::{Deferred3dmodButton, ProcessResultDisplay},
    tomogram_generation_parent::TomogramGenerationParent,
};

/// Java `Tilt3dFindParent.rcsid`.
pub const RCSID: &str = "$Id$";
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions, r#type::processing_method::ProcessingMethod,
};

/// Java package-private `Tilt3dFindParent`.
pub trait Tilt3dFindParent: TomogramGenerationParent {
    /// Java `tilt3dFindAction(ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions, ProcessingMethod)`.
    fn tilt_3d_find_action(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        processing_method: ProcessingMethod,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent;
    impl TomogramGenerationParent for Parent {
        fn is_ctf3d(&self) -> bool {
            false
        }
        fn is_method_plugin(&self) -> bool {
            false
        }
        fn is_multifilt(&self) -> bool {
            false
        }
        fn is_back_projection(&self) -> bool {
            true
        }
        fn is_sirt(&self) -> bool {
            false
        }
    }
    impl Tilt3dFindParent for Parent {
        fn tilt_3d_find_action(
            &mut self,
            _: &ProcessResultDisplay,
            _: Option<&Deferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: ProcessingMethod,
        ) {
        }
    }
    #[test]
    fn source_contract_retains_tomogram_generation_superinterface() {
        let mut parent = Parent;
        assert!(parent.is_back_projection());
        parent.tilt_3d_find_action(
            &ProcessResultDisplay,
            None,
            None,
            ProcessingMethod::LocalCpu,
        );
    }
}
