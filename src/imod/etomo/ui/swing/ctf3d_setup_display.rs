//! `IMOD/Etomo/src/etomo/ui/swing/Ctf3dSetupDisplay.java`.

use crate::imod::etomo::comscript::ctf3d_setup_param::Ctf3dSetupParam;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java `Ctf3dSetupDisplay` (does not extend `ProcessDisplay`).
pub trait Ctf3dSetupDisplay {
    /// Java `getParameters(Ctf3dSetupParam, boolean)`.
    fn get_parameters(&self, param: &mut Ctf3dSetupParam, do_validation: bool) -> bool;

    /// Java `isRunSlabsInParallel()`.
    fn is_run_slabs_in_parallel(&self) -> bool;

    /// Java `getCtfCorrectionUIComponent()`.
    fn get_ctf_correction_ui_component(&self) -> Option<&dyn UIComponent>;

    /// Java `getEraseFiducialsUIComponent()`.
    fn get_erase_fiducials_ui_component(&self) -> Option<&dyn UIComponent>;

    /// Java `isEraseFiducials()`.
    fn is_erase_fiducials(&self) -> bool;

    /// Java `getFilterIn2DUIComponent()`.
    fn get_filter_in_2d_ui_component(&self) -> Option<&dyn UIComponent>;

    /// Java `isFilterIn2D()`.
    fn is_filter_in_2d(&self) -> bool;

    /// Java `isUseUnalignedImages()`.
    fn is_use_unaligned_images(&self) -> bool;

    /// Java `getUseUnalignedImagesUIComponent()`.
    fn get_use_unaligned_images_ui_component(&self) -> Option<&dyn UIComponent>;
}
