//! `IMOD/Etomo/src/etomo/ui/swing/Ctf3dSetupDisplay.java`.
//!
//! The Java interface is the boundary used by `ApplicationManager` while it
//! writes and runs `ctf3dsetup.com`.  `UIComponent` remains a frontend
//! boundary: it deliberately exposes identity rather than manufacturing a
//! replacement widget hierarchy.  The translated `Ctf3dPanel` keeps its Java
//! parent, pixel-size, and manager dependencies as explicit Rust arguments;
//! those values were fields of the Java panel rather than arguments of this
//! interface.
#![allow(dead_code)]

use super::check_box::CheckBox;
use super::ctf3d_panel::{
    Ctf3dPanel, Ctf3dPanelApplicationManager, Ctf3dPanelParent,
    Ctf3dSetupParam as Ctf3dSetupParamBoundary,
};
use crate::imod::etomo::ui::UiComponent;

/// Java `Ctf3dSetupDisplay`.
///
/// `P`, `T`, and `M` make the dependencies retained by the translated
/// `Ctf3dPanel` explicit.  They correspond respectively to Java
/// `Ctf3dSetupParam`, the dialog parent, and `ApplicationManager`.
pub trait Ctf3dSetupDisplay<P, T, M>
where
    P: Ctf3dSetupParamBoundary,
    T: Ctf3dPanelParent,
    M: Ctf3dPanelApplicationManager,
{
    /// Java `getParameters(Ctf3dSetupParam, boolean)`.
    fn get_parameters(
        &mut self,
        param: &mut P,
        do_validation: bool,
        parent: &T,
        pixel_size: Option<f64>,
        manager: &mut M,
    ) -> bool;

    /// Java `isRunSlabsInParallel()`.
    fn is_run_slabs_in_parallel(&self) -> bool;

    /// Java `getCtfCorrectionUIComponent()`.
    fn get_ctf_correction_ui_component(&self) -> &dyn UiComponent;

    /// Java `getEraseFiducialsUIComponent()`.
    fn get_erase_fiducials_ui_component(&self) -> &dyn UiComponent;

    /// Java `isEraseFiducials()`.
    fn is_erase_fiducials(&self) -> bool;

    /// Java `getFilterIn2DUIComponent()`.
    fn get_filter_in_2d_ui_component(&self) -> &dyn UiComponent;

    /// Java `isFilterIn2D()`.
    fn is_filter_in_2d(&self) -> bool;

    /// Java `isUseUnalignedImages()`.
    fn is_use_unaligned_images(&self) -> bool;

    /// Java `getUseUnalignedImagesUIComponent()`.
    fn get_use_unaligned_images_ui_component(&self) -> &dyn UiComponent;
}

impl UiComponent for Ctf3dPanel {}
impl UiComponent for CheckBox {}

impl<P, T, M> Ctf3dSetupDisplay<P, T, M> for Ctf3dPanel
where
    P: Ctf3dSetupParamBoundary,
    T: Ctf3dPanelParent,
    M: Ctf3dPanelApplicationManager,
{
    fn get_parameters(
        &mut self,
        param: &mut P,
        do_validation: bool,
        parent: &T,
        pixel_size: Option<f64>,
        manager: &mut M,
    ) -> bool {
        Ctf3dPanel::get_parameters(self, param, do_validation, parent, pixel_size, manager)
    }

    fn is_run_slabs_in_parallel(&self) -> bool {
        Ctf3dPanel::is_run_slabs_in_parallel(self)
    }

    fn get_ctf_correction_ui_component(&self) -> &dyn UiComponent {
        self
    }

    fn get_erase_fiducials_ui_component(&self) -> &dyn UiComponent {
        &self.cb_erase_fiducials
    }

    fn is_erase_fiducials(&self) -> bool {
        Ctf3dPanel::is_erase_fiducials(self)
    }

    fn get_filter_in_2d_ui_component(&self) -> &dyn UiComponent {
        &self.cb_filter_in_2d
    }

    fn is_filter_in_2d(&self) -> bool {
        Ctf3dPanel::is_filter_in_2d(self)
    }

    fn is_use_unaligned_images(&self) -> bool {
        Ctf3dPanel::is_use_unaligned_images(self)
    }

    fn get_use_unaligned_images_ui_component(&self) -> &dyn UiComponent {
        &self.cb_use_unaligned_images
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::axis_id::AxisID;

    #[derive(Default)]
    struct Param {
        slab_thickness: Option<String>,
        temporary_directory: Option<String>,
        run_slabs_in_parallel: bool,
        erase_fiducials: bool,
        filter_in_2d: bool,
        use_unaligned_images: bool,
        adjust_for_align_z_shift: bool,
        fourier_reduce_by_factor: i32,
        vertical_slices: bool,
        old_style_x_tilting: bool,
    }
    impl Ctf3dSetupParamBoundary for Param {
        fn slab_thickness_in_nm(&self) -> Option<String> {
            self.slab_thickness.clone()
        }
        fn set_slab_thickness_in_nm(&mut self, value: String) -> Result<(), String> {
            self.slab_thickness = Some(value);
            Ok(())
        }
        fn temporary_directory(&self) -> Option<String> {
            self.temporary_directory.clone()
        }
        fn set_temporary_directory(&mut self, value: String) -> Result<(), String> {
            self.temporary_directory = Some(value);
            Ok(())
        }
        fn run_slabs_in_parallel(&self) -> bool {
            self.run_slabs_in_parallel
        }
        fn set_run_slabs_in_parallel(&mut self, value: bool) {
            self.run_slabs_in_parallel = value;
        }
        fn erase_fiducials(&self) -> bool {
            self.erase_fiducials
        }
        fn set_erase_fiducials(&mut self, value: bool) {
            self.erase_fiducials = value;
        }
        fn filter_in_2d(&self) -> bool {
            self.filter_in_2d
        }
        fn set_filter_in_2d(&mut self, value: bool) {
            self.filter_in_2d = value;
        }
        fn use_unaligned_images(&self) -> bool {
            self.use_unaligned_images
        }
        fn set_use_unaligned_images(&mut self, value: bool) {
            self.use_unaligned_images = value;
        }
        fn adjust_for_align_z_shift(&self) -> bool {
            self.adjust_for_align_z_shift
        }
        fn set_adjust_for_align_z_shift(&mut self, value: bool) {
            self.adjust_for_align_z_shift = value;
        }
        fn fourier_reduce_by_factor(&self) -> i32 {
            self.fourier_reduce_by_factor
        }
        fn set_fourier_reduce_by_factor(&mut self, value: i32) {
            self.fourier_reduce_by_factor = value;
        }
        fn vertical_slices(&self) -> bool {
            self.vertical_slices
        }
        fn set_vertical_slices(&mut self, value: bool) {
            self.vertical_slices = value;
        }
        fn old_style_x_tilting(&self) -> bool {
            self.old_style_x_tilting
        }
        fn set_old_style_x_tilting(&mut self, value: bool) {
            self.old_style_x_tilting = value;
        }
    }

    struct Parent;
    impl Ctf3dPanelParent for Parent {
        fn tomo_thickness(&self) -> Option<i64> {
            Some(1_000)
        }
        fn x_axis_tilt(&self) -> Option<String> {
            Some("0".into())
        }
        fn is_use_local_alignment(&self) -> bool {
            false
        }
        fn is_use_z_factors(&self) -> bool {
            false
        }
        fn is_ctf3d(&self) -> bool {
            true
        }
    }
    #[derive(Default)]
    struct Manager {
        messages: Vec<String>,
    }
    impl Ctf3dPanelApplicationManager for Manager {
        fn ctf3d_setup(
            &mut self,
            _: AxisID,
            _: Option<crate::imod::etomo::process::imod_process::Run3dmodMenuOptions>,
        ) {
        }
        fn open_ctf3d(
            &mut self,
            _: AxisID,
            _: Option<crate::imod::etomo::process::imod_process::Run3dmodMenuOptions>,
        ) {
        }
        fn use_ctf3d(&mut self, _: AxisID) {}
        fn open_message_dialog(&mut self, message: String, _: &str, _: AxisID) {
            self.messages.push(message);
        }
        fn pack(&mut self, _: AxisID) {}
    }

    #[test]
    fn ctf3d_panel_implements_the_full_display_contract() {
        let mut panel = Ctf3dPanel::get_instance(AxisID::Only);
        panel.tf_slab_thickness_in_nm.set_text("100");
        panel.temporary_directory.set_text("temporary");
        panel.cb_run_slabs_in_parallel.set_selected(true);
        panel.cb_erase_fiducials.set_selected(true);
        panel.cb_filter_in_2d.set_selected(true);
        panel.cb_use_unaligned_images.set_selected(true);
        let mut param = Param::default();
        let mut manager = Manager::default();
        let display: &mut dyn Ctf3dSetupDisplay<Param, Parent, Manager> = &mut panel;

        assert!(display.get_parameters(&mut param, false, &Parent, Some(1.0), &mut manager));
        assert_eq!(param.slab_thickness.as_deref(), Some("100"));
        assert_eq!(param.temporary_directory.as_deref(), Some("temporary"));
        assert!(display.is_run_slabs_in_parallel());
        assert!(display.is_erase_fiducials());
        assert!(display.is_filter_in_2d());
        assert!(display.is_use_unaligned_images());
        let _: &dyn UiComponent = display.get_ctf_correction_ui_component();
        let _: &dyn UiComponent = display.get_erase_fiducials_ui_component();
        let _: &dyn UiComponent = display.get_filter_in_2d_ui_component();
        let _: &dyn UiComponent = display.get_use_unaligned_images_ui_component();
    }
}
