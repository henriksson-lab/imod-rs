//! `IMOD/Etomo/src/etomo/ui/swing/PreProcessingDialog.java`.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::ccd_eraser_display::CcdEraserDisplay;
use super::ccd_eraser_xrays_panel::{self, CcdEraserXRaysPanel};
use super::check_box::CheckBox;
use super::etomo_panel::EtomoPanel;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::const_ccd_eraser_param::ConstCCDEraserParam;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class PreProcessingDialog extends ProcessDialog`.
pub struct PreProcessingDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,
    text_dm2mrc: Rc<JComponent>,
    pnl_dm_convert: Rc<EtomoPanel>,
    cb_unique_headers: Rc<CheckBox>,
    pnl_eraser: Rc<EtomoPanel>,

    ccd_eraser_x_rays_panel: Rc<CcdEraserXRaysPanel>,
}

impl Deref for PreProcessingDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl PreProcessingDialog {
    /// Java `PreProcessingDialog(ApplicationManager, AxisID)`
    /// (PreProcessingDialog.java:212-243).
    pub fn new(
        app_manager: &'static ApplicationManager,
        axis_id: AxisID,
    ) -> Rc<PreProcessingDialog> {
        let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
            app_manager,
            axis_id,
            DialogType::PreProcessing,
        );
        // Field initializers (PreProcessingDialog.java:204-208).
        let text_dm2mrc = JComponent::new_label("No digital micrograph files detected:  ");
        let pnl_dm_convert = EtomoPanel::new();
        let cb_unique_headers =
            CheckBox::new_string(Some("Digital micrograph files have unique headers"));
        let pnl_eraser = EtomoPanel::new();
        let ccd_eraser_x_rays_panel = CcdEraserXRaysPanel::get_instance(
            app_manager,
            axis_id,
            base.dialog_type,
            &base.btn_advanced,
        );
        let dialog = Rc::new(PreProcessingDialog {
            base,
            text_dm2mrc,
            pnl_dm_convert,
            cb_unique_headers,
            pnl_eraser,
            ccd_eraser_x_rays_panel,
        });
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&(dialog.clone() as Rc<dyn ProcessDialogVirtual>));
        dialog.base.set_this(this);

        // Swing layout: rootPanel BoxLayout Y_AXIS.

        // Build the digital micrograph panel
        // Swing layout: pnlDMConvert BoxLayout Y_AXIS.
        dialog
            .pnl_dm_convert
            .set_border(&BeveledBorder::new(Some("Digital Micrograph Conversion")).get_border());
        let dm_convert = dialog.pnl_dm_convert.get_component();
        dm_convert.add(&dialog.text_dm2mrc);
        // Swing layout: rigid area x20_y0.
        dm_convert.add(&dialog.cb_unique_headers.get_component());
        // applicationManager.isDigitalMicrographData();
        dialog.disable_dm2mrc();

        // Build the base panel
        dialog.btn_execute.set_text(Some("Done"));
        // Swing layout: pnlDMConvert center alignment.
        dialog.root_panel.get_component().add(&dm_convert);

        // Swing layout: pnlEraser BoxLayout Y_AXIS.
        dialog
            .pnl_eraser
            .set_border(&BeveledBorder::new(Some("CCD Eraser")).get_border());
        dialog
            .pnl_eraser
            .get_component()
            .add(&dialog.ccd_eraser_x_rays_panel.get_container());

        dialog
            .root_panel
            .get_component()
            .add(&dialog.pnl_eraser.get_component());
        dialog.add_exit_buttons();

        // Set the default advanced state for the window, this also executes
        dialog.update_advanced();
        dialog
    }

    /// Java `static getUseFixedStackLabel()` (PreProcessingDialog.java:245-247).
    pub fn get_use_fixed_stack_label() -> &'static str {
        ccd_eraser_xrays_panel::USE_FIXED_STACK_LABEL
    }

    /// Java `setCCDEraserParams(ConstCCDEraserParam)` (PreProcessingDialog.java:252-254).
    /// Set the parameters for the specified CCD eraser panel.
    pub fn set_ccd_eraser_params(&self, ccd_eraser_params: &ConstCCDEraserParam) {
        self.ccd_eraser_x_rays_panel
            .set_parameters_const_ccd_eraser_param(ccd_eraser_params);
    }

    /// Java `setParameters(BaseScreenState)` (PreProcessingDialog.java:256-258).
    pub fn set_parameters(&self, screen_state: &BaseScreenState) {
        self.ccd_eraser_x_rays_panel
            .set_parameters_base_screen_state(screen_state);
    }

    /// Java `getParameters(BaseScreenState)` (PreProcessingDialog.java:260-262).
    pub fn get_parameters(&self, screen_state: &BaseScreenState) {
        self.ccd_eraser_x_rays_panel
            .get_parameters_base_screen_state(screen_state);
    }

    /// Java `getCCDEraserDisplay()` (PreProcessingDialog.java:264-266).
    pub fn get_ccd_eraser_display(&self) -> Rc<dyn CcdEraserDisplay> {
        self.ccd_eraser_x_rays_panel.clone()
    }

    /// Java private `updateAdvanced()` (PreProcessingDialog.java:271-274).
    /// Update the dialog with the current advanced state.
    fn update_advanced(&self) {
        self.ccd_eraser_x_rays_panel
            .update_advanced(self.is_advanced());
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.application_manager))
        });
    }

    /// Java private `disableDM2MRC()` (PreProcessingDialog.java:276-280).
    fn disable_dm2mrc(&self) {
        self.cb_unique_headers.set_enabled(false);
        self.cb_unique_headers.set_selected_boolean(false);
        self.pnl_dm_convert.get_component().set_visible(false);
    }
}

impl ProcessDialogVirtual for PreProcessingDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java `done()` (PreProcessingDialog.java:282-287).
    fn done(&self) {
        self.application_manager.done_pre_proc_dialog(self.axis_id);
        self.ccd_eraser_x_rays_panel.done();
        self.set_displayed(false);
    }
}
