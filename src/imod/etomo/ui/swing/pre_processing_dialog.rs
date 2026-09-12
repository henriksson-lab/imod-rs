//! `IMOD/Etomo/src/etomo/ui/swing/PreProcessingDialog.java`.
//!
//! The native component hierarchy and `UIHarness.INSTANCE.pack` are direct GUI
//! boundaries.  This module retains the source dialog's state, construction
//! order, parameter forwarding, advanced-state update, and teardown order.
#![allow(dead_code)]

use super::{
    beveled_border::BeveledBorder,
    ccd_eraser_xrays_panel::{
        CcdEraserScreenState, CcdEraserXRaysApplicationManager, CcdEraserXRaysPanel,
        ConstCcdEraserParam, USE_FIXED_STACK_LABEL,
    },
    check_box::CheckBox,
    process_dialog::{ProcessDialog, ProcessDialogApplicationManager},
};
use crate::imod::etomo::r#type::{axis_id::AxisID, dialog_type::DialogType};

/// Direct `ApplicationManager.donePreProcDialog` and UI-packing boundary.
pub trait PreProcessingDialogApplicationManager:
    CcdEraserXRaysApplicationManager + ProcessDialogApplicationManager
{
    fn done_pre_proc_dialog(&mut self, axis_id: AxisID);
    fn pack(&self, axis_id: AxisID);
}

/// The `JLabel` source state passed to the native Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreProcessingLabel {
    pub text: String,
}

/// Java `PreProcessingDialog` fields and methods.
pub struct PreProcessingDialog<'a> {
    /// Java superclass `ProcessDialog` state.
    pub process_dialog: ProcessDialog<'a>,
    pub text_dm2mrc: PreProcessingLabel,
    pub cb_unique_headers: CheckBox,
    pub ccd_eraser_xrays_panel: CcdEraserXRaysPanel,
    /// Source `BoxLayout.Y_AXIS` assignment on root, DM-conversion, and eraser panels.
    pub box_layout_y_axis: [bool; 3],
    /// Source `pnlDMConvert` and `pnlEraser` titled-border boundary state.
    pub panel_borders: [BeveledBorder; 2],
    /// Exact source child insertion order at the widget boundary.
    pub dm_convert_component_order: Vec<&'static str>,
    pub root_component_order: Vec<&'static str>,
    /// Source `disableDM2MRC` result.
    pub dm_convert_visible: bool,
}

impl<'a> PreProcessingDialog<'a> {
    /// Java `PreProcessingDialog(ApplicationManager, AxisID)`.
    pub fn new<M: PreProcessingDialogApplicationManager>(
        application_manager: &'a M,
        axis_id: AxisID,
    ) -> Self {
        let mut process_dialog = ProcessDialog::new(
            application_manager,
            axis_id,
            DialogType::PreProcessing,
            Box::new(|| {}),
        );
        process_dialog.btn_execute.set_text("Done");
        process_dialog.add_exit_buttons();
        let mut dialog = Self {
            process_dialog,
            text_dm2mrc: PreProcessingLabel {
                text: "No digital micrograph files detected:  ".into(),
            },
            cb_unique_headers: CheckBox::new_with_text(
                "Digital micrograph files have unique headers",
            ),
            ccd_eraser_xrays_panel: CcdEraserXRaysPanel::get_instance(
                axis_id,
                DialogType::PreProcessing,
                application_manager.is_advanced(DialogType::PreProcessing, axis_id),
            ),
            box_layout_y_axis: [true; 3],
            panel_borders: [
                BeveledBorder::new("Digital Micrograph Conversion"),
                BeveledBorder::new("CCD Eraser"),
            ],
            dm_convert_component_order: vec![
                "no-digital-micrograph-files-label",
                "rigid-area-x20-y0",
                "unique-headers-checkbox",
            ],
            root_component_order: vec![
                "digital-micrograph-conversion-panel",
                "ccd-eraser-panel",
                "exit-buttons",
            ],
            dm_convert_visible: true,
        };
        dialog.disable_dm2mrc();
        dialog.update_advanced(application_manager);
        dialog
    }

    /// Java static `getUseFixedStackLabel()`.
    pub fn get_use_fixed_stack_label() -> &'static str {
        USE_FIXED_STACK_LABEL
    }

    /// Java `setCCDEraserParams(ConstCCDEraserParam)`.
    pub fn set_ccd_eraser_params<P: ConstCcdEraserParam>(&mut self, ccd_eraser_params: &P) {
        self.ccd_eraser_xrays_panel
            .set_parameters(ccd_eraser_params);
    }

    /// Java `setParameters(BaseScreenState)`.
    pub fn set_parameters(&mut self, screen_state: &CcdEraserScreenState) {
        self.ccd_eraser_xrays_panel.set_screen_state(screen_state);
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters(&self, screen_state: &mut CcdEraserScreenState) {
        self.ccd_eraser_xrays_panel.get_screen_state(screen_state);
    }

    /// Java `getCCDEraserDisplay()`.
    pub fn get_ccd_eraser_display(&self) -> &CcdEraserXRaysPanel {
        &self.ccd_eraser_xrays_panel
    }

    /// Java private `updateAdvanced()`.
    pub fn update_advanced<M: PreProcessingDialogApplicationManager>(
        &mut self,
        application_manager: &M,
    ) {
        self.ccd_eraser_xrays_panel
            .update_advanced(self.process_dialog.is_advanced());
        application_manager.pack(self.process_dialog.axis_id);
    }

    /// Java private `disableDM2MRC()`.
    pub fn disable_dm2mrc(&mut self) {
        self.cb_unique_headers.set_enabled(false);
        self.cb_unique_headers.set_selected(false);
        self.dm_convert_visible = false;
    }

    /// Java override `done()`.
    pub fn done<M: PreProcessingDialogApplicationManager>(&mut self, application_manager: &mut M) {
        application_manager.done_pre_proc_dialog(self.process_dialog.axis_id);
        self.ccd_eraser_xrays_panel.done();
        self.process_dialog.set_displayed(false);
    }
}
