//! `IMOD/Etomo/src/etomo/ui/swing/Tilt3dFindParent.java`.
//!
//! Java package-private `interface Tilt3dFindParent extends
//! TomogramGenerationParent`: the owner of a `Tilt3dFindPanel`
//! (`Beads3dFindPanel`), which runs tilt_3dfind for it.

use std::rc::Rc;

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::tomogram_generation_parent::TomogramGenerationParent;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `interface Tilt3dFindParent extends TomogramGenerationParent`.
pub trait Tilt3dFindParent: TomogramGenerationParent {
    /// Java `tilt3dFindAction(ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions, ProcessingMethod)`.
    fn tilt3d_find_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        processing_method: ProcessingMethod,
    );
}
