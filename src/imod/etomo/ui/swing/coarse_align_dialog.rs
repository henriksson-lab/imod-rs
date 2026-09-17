//! `IMOD/Etomo/src/etomo/ui/swing/CoarseAlignDialog.java`.
//!
//! Native widget painting, filesystem probing, and application process launches
//! are explicit boundaries.  This preserves the dialog-owned construction,
//! delegation, context menu, action dispatch, and listener teardown.
#![allow(dead_code)]

use super::{
    check_box::CheckBox,
    context_popup::{ContextPopup, GraphTask, MouseEvent},
    fiducialess_params::FiducialessParams,
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    multi_line_button::MultiLineButton,
    spinner::Spinner,
    tiltxcorr_panel::{ConstTiltxcorrParam, TiltxcorrPanel, TiltxcorrScreenState},
};
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{axis_id::AxisID, dialog_type::DialogType, view_type::ViewType},
    ui::field_type::FieldType,
};
use std::path::Path;

pub const COARSE_ALIGNMENT_BORDER: &str = "Coarse Alignment";
pub const COARSE_ALIGNMENT_ONLY: &str = "Coarse alignment only";
pub const TILT_AXIS_ROTATION: &str = "Tilt axis rotation:";
pub const BINNING_IN_MIDAS: &str = "Binning in Midas: ";

/// Java `PrenewstPanel` calls at the self-parented dialog boundary.  The standalone
/// Rust panel owns its manager and parent by value, whereas Java passes this dialog
/// as parent, so representing that here avoids an invalid self-reference.
pub trait CoarseAlignPrenewstPanel {
    fn set_parameters_newst<T>(&mut self, param: &T);
    fn set_parameters_blendmont<T>(&mut self, param: &T);
    fn set_parameters_screen_state<T>(&mut self, state: &T);
    fn set_parameters_meta_data<T>(&mut self, data: &T);
    fn get_parameters_screen_state<T>(&self, state: &mut T);
    fn get_parameters_meta_data<T>(&self, data: &mut T);
    fn update_advanced(&mut self, state: bool);
    fn done(&mut self);
}
/// Direct `ApplicationManager` calls and source metadata reads.
pub trait CoarseAlignDialogApplicationManager {
    fn view_type(&self) -> ViewType;
    fn distortion_correction(&self) -> bool;
    fn property_user_dir(&self) -> &Path;
    fn distortion_corrected_file_exists(&self, axis_id: AxisID) -> bool;
    fn midas_raw_stack(&mut self, axis_id: AxisID, button: &MultiLineButton);
    fn midas_fix_edges(&mut self, axis_id: AxisID, button: &MultiLineButton);
    fn make_distortion_corrected_stack(&mut self, axis_id: AxisID, button: &MultiLineButton);
    fn done_coarse_align_dialog(&mut self, axis_id: AxisID);
    fn pack(&mut self, axis_id: AxisID);
}
/// Source-visible JPanel/BoxLayout/border/add-order state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoarseAlignDialogLayout {
    pub pnl_coarse_align_box_layout_y_axis: bool,
    pub pnl_coarse_align_border: Option<String>,
    pub pnl_fiducialess_box_layout_y_axis: bool,
    pub root_panel_box_layout_y_axis: bool,
    pub coarse_align_order: Vec<String>,
    pub root_order: Vec<String>,
    pub fix_edges_present: bool,
    pub fix_edges_order: Vec<String>,
    pub component_alignment_x: f32,
    pub mouse_listener_count: usize,
    pub button_size_set: bool,
}
/// Java final `CoarseAlignDialog` state.
pub struct CoarseAlignDialog<P: CoarseAlignPrenewstPanel> {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub pnl_coarse_align: CoarseAlignDialogLayout,
    pub cb_fiducialess: CheckBox,
    pub ltf_rotation: LabeledTextField,
    pub btn_midas: MultiLineButton,
    pub tiltxcorr_panel: TiltxcorrPanel,
    pub pnl_prenewst: P,
    pub btn_fix_edges_midas: MultiLineButton,
    pub btn_distortion_corrected_stack: MultiLineButton,
    pub sp_midas_binning: Spinner,
    pub advanced: bool,
    pub displayed: bool,
    pub execute_button_text: String,
    pub action_listener_count: usize,
    pub context_popup: Option<ContextPopup>,
}
impl<P: CoarseAlignPrenewstPanel> CoarseAlignDialog<P> {
    /// Java private constructor; process-factory buttons remain factory-owned inputs.
    pub fn new<M: CoarseAlignDialogApplicationManager>(
        manager: &M,
        axis_id: AxisID,
        mag_changes_mode: bool,
        pnl_prenewst: P,
        btn_distortion_corrected_stack: MultiLineButton,
        btn_fix_edges_midas: MultiLineButton,
        btn_midas: MultiLineButton,
        advanced: bool,
    ) -> Self {
        let montage = manager.view_type() == ViewType::Montage;
        let mut dialog = Self {
            axis_id,
            dialog_type: DialogType::CoarseAlignment,
            pnl_coarse_align: CoarseAlignDialogLayout {
                pnl_coarse_align_box_layout_y_axis: true,
                pnl_coarse_align_border: Some(COARSE_ALIGNMENT_BORDER.into()),
                pnl_fiducialess_box_layout_y_axis: true,
                root_panel_box_layout_y_axis: true,
                coarse_align_order: if montage {
                    vec![
                        "tiltxcorrPanel".into(),
                        "pnlFixEdges".into(),
                        "pnlPrenewst".into(),
                        "pnlFiducialess".into(),
                        "spMidasBinning".into(),
                        "btnMidas".into(),
                    ]
                } else {
                    vec![
                        "tiltxcorrPanel".into(),
                        "pnlPrenewst".into(),
                        "pnlFiducialess".into(),
                        "spMidasBinning".into(),
                        "btnMidas".into(),
                    ]
                },
                root_order: vec!["pnlCoarseAlign".into(), "exitButtons".into()],
                fix_edges_present: montage,
                fix_edges_order: if montage {
                    vec![
                        "btnDistortionCorrectedStack".into(),
                        "btnFixEdgesMidas".into(),
                    ]
                } else {
                    vec![]
                },
                component_alignment_x: 0.5,
                button_size_set: true,
                ..Default::default()
            },
            cb_fiducialess: CheckBox::new_with_text(COARSE_ALIGNMENT_ONLY),
            ltf_rotation: LabeledTextField::new(FieldType::FloatingPoint, TILT_AXIS_ROTATION),
            btn_midas,
            tiltxcorr_panel: TiltxcorrPanel::get_cross_correlation_instance(
                axis_id,
                DialogType::CoarseAlignment,
                mag_changes_mode,
            ),
            pnl_prenewst,
            btn_fix_edges_midas,
            btn_distortion_corrected_stack,
            sp_midas_binning: Spinner::get_labeled_instance(BINNING_IN_MIDAS, 1, 1, 8, 1),
            advanced,
            displayed: true,
            execute_button_text: "Done".into(),
            action_listener_count: 0,
            context_popup: None,
        };
        dialog.set_tool_tip_text();
        if montage && !manager.distortion_correction() {
            dialog.btn_distortion_corrected_stack.set_enabled(false);
        }
        dialog.set_enabled_fix_edges_midas_button(manager);
        dialog.update_advanced_without_manager();
        dialog
    }
    /// Java static `getInstance`.
    pub fn get_instance<M: CoarseAlignDialogApplicationManager>(
        manager: &M,
        axis_id: AxisID,
        mag_changes_mode: bool,
        pnl_prenewst: P,
        btn_distortion_corrected_stack: MultiLineButton,
        btn_fix_edges_midas: MultiLineButton,
        btn_midas: MultiLineButton,
        advanced: bool,
    ) -> Self {
        let mut dialog = Self::new(
            manager,
            axis_id,
            mag_changes_mode,
            pnl_prenewst,
            btn_distortion_corrected_stack,
            btn_fix_edges_midas,
            btn_midas,
            advanced,
        );
        dialog.add_listeners();
        dialog
    }
    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_midas.add_action_listener();
        self.btn_fix_edges_midas.add_action_listener();
        self.btn_distortion_corrected_stack.add_action_listener();
        self.action_listener_count = 3;
        self.pnl_coarse_align.mouse_listener_count += 1;
    }
    /// Java `setEnabledFixEdgesMidasButton`.
    pub fn set_enabled_fix_edges_midas_button<M: CoarseAlignDialogApplicationManager>(
        &mut self,
        manager: &M,
    ) {
        self.btn_fix_edges_midas.set_enabled(
            !self.btn_distortion_corrected_stack.is_enabled()
                || manager.distortion_corrected_file_exists(self.axis_id),
        );
    }
    pub fn set_cross_correlation_params<T: ConstTiltxcorrParam>(&mut self, param: &T) {
        self.tiltxcorr_panel.set_parameters_tiltxcorr(param);
    }
    pub fn get_tiltxcorr_display(&self) -> &TiltxcorrPanel {
        &self.tiltxcorr_panel
    }
    pub fn get_newstack_display(&self) -> &P {
        &self.pnl_prenewst
    }
    pub fn set_prenewst_params<T>(&mut self, value: &T) {
        self.pnl_prenewst.set_parameters_newst(value);
    }
    pub fn set_params<T>(&mut self, value: &T) {
        self.pnl_prenewst.set_parameters_blendmont(value);
    }
    pub fn set_parameters_screen_state<S: TiltxcorrScreenState>(&mut self, value: &mut S) {
        self.tiltxcorr_panel.set_parameters_screen_state(value);
        self.pnl_prenewst.set_parameters_screen_state(value);
    }
    pub fn set_parameters_meta_data<M>(&mut self, value: &M) {
        self.pnl_prenewst.set_parameters_meta_data(value);
    }
    pub fn get_parameters_screen_state<S: TiltxcorrScreenState>(&self, value: &mut S) {
        self.tiltxcorr_panel.get_parameters_screen_state(value);
        self.pnl_prenewst.get_parameters_screen_state(value);
    }
    pub fn get_parameters_meta_data<M>(&self, value: &mut M) {
        self.pnl_prenewst.get_parameters_meta_data(value);
    }
    pub fn set_fiducialess_alignment(&mut self, state: bool) {
        self.cb_fiducialess.set_selected(state);
    }
    pub fn is_fiducialess(&self) -> bool {
        self.cb_fiducialess.is_selected()
    }
    pub fn set_image_rotation(&mut self, rotation: &str) {
        self.ltf_rotation.set_text(rotation);
    }
    pub fn get_image_rotation(
        &self,
        validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_rotation.get_text_validated(validation)
    }
    fn update_advanced_without_manager(&mut self) {
        self.tiltxcorr_panel.update_advanced(self.advanced);
        self.pnl_prenewst.update_advanced(self.advanced);
    }
    /// Java `updateAdvanced` and `UIHarness.pack` boundary.
    pub fn update_advanced<M: CoarseAlignDialogApplicationManager>(&mut self, manager: &mut M) {
        self.update_advanced_without_manager();
        manager.pack(self.axis_id);
    }
    /// Java `popUpContextMenu` construction inputs.
    pub fn pop_up_context_menu<M: CoarseAlignDialogApplicationManager>(
        &mut self,
        manager: &M,
        mouse_event: MouseEvent,
        raw_stack_is_one_by: bool,
        xcorr_blendmont_was_run: bool,
    ) -> Result<(), String> {
        let montage = manager.view_type() == ViewType::Montage;
        let (align_label, align_manpage, log_label, log_file) = if montage {
            ("Blendmont", "blendmont", "Preblend", "preblend")
        } else {
            ("Newstack", "newstack", "Prenewst", "prenewst")
        };
        let graph = if montage && !raw_stack_is_one_by && xcorr_blendmont_was_run {
            vec![GraphTask {
                description: "COARSE_MEAN_MAX".into(),
                available: true,
                input_file: None,
            }]
        } else {
            vec![]
        };
        let labels = vec![
            "Tiltxcorr".into(),
            "Xftoxg".into(),
            align_label.into(),
            "3dmod".into(),
            "Midas".into(),
        ];
        let pages = vec![
            "tiltxcorr.html".into(),
            "xftoxg.html".into(),
            format!("{align_manpage}.html"),
            "3dmod.html".into(),
            "midas.html".into(),
        ];
        let logfile_labels = vec!["Xcorr".into(), log_label.into()];
        let e = self.axis_id.get_extension();
        let logfiles = vec![format!("xcorr{e}.log"), format!("{log_file}{e}.log")];
        self.context_popup = Some(ContextPopup::new_graphs(
            mouse_event,
            Some("COARSE ALIGNMENT"),
            super::context_popup::TOMO_GUIDE,
            &labels,
            &pages,
            &logfile_labels,
            &logfiles,
            &graph,
            self.axis_id,
            false,
        )?);
        Ok(())
    }
    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.cb_fiducialess.set_tool_tip_text(Some(
            "Enable or disable the processing flow using cross-correlation alignment only.",
        ));
        self.btn_midas
            .set_tool_tip_text(Some("Use Midas to adjust bad alignments."));
        self.ltf_rotation.set_tool_tip_text(Some(
            "Initial rotation angle of tilt axis when viewing images in Midas.",
        ));
        self.btn_distortion_corrected_stack.set_tool_tip_text(Some("Create a stack to use in Midas that incorporates the corrections from the image distortion field file and/or the magnification gradients file."));
        self.btn_fix_edges_midas.set_tool_tip_text(Some(
            "Use Midas to adjust the alignment of the montage frames.",
        ));
    }
    /// Java `action`.
    pub fn action<M: CoarseAlignDialogApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        _deferred: Option<()>,
        _options: Option<Run3dmodMenuOptions>,
    ) {
        if self.btn_midas.get_action_command() == Some(command) {
            manager.midas_raw_stack(self.axis_id, &self.btn_midas);
        } else if self.btn_fix_edges_midas.get_action_command() == Some(command) {
            manager.midas_fix_edges(self.axis_id, &self.btn_fix_edges_midas);
        } else if self.btn_distortion_corrected_stack.get_action_command() == Some(command) {
            manager.make_distortion_corrected_stack(
                self.axis_id,
                &self.btn_distortion_corrected_stack,
            );
        }
    }
    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<M: CoarseAlignDialogApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
    ) {
        self.action(manager, command, None, None);
    }
    /// Java `done`.
    pub fn done<M: CoarseAlignDialogApplicationManager>(&mut self, manager: &mut M) {
        manager.done_coarse_align_dialog(self.axis_id);
        self.tiltxcorr_panel.done();
        self.pnl_prenewst.done();
        self.btn_distortion_corrected_stack.remove_action_listener();
        self.btn_fix_edges_midas.remove_action_listener();
        self.btn_midas.remove_action_listener();
        self.action_listener_count = 0;
        self.displayed = false;
    }
    /// Java `getCoarseAlignParameters(MidasParam)`.
    pub fn get_coarse_align_parameters<T: CoarseAlignMidasParam>(&self, param: &mut T) {
        param.set_binning(self.sp_midas_binning.get_value());
    }
}
/// Java `MidasParam.setBinning` boundary.
pub trait CoarseAlignMidasParam {
    fn set_binning(&mut self, binning: i32);
}

impl<P: CoarseAlignPrenewstPanel> FiducialessParams for CoarseAlignDialog<P> {
    fn is_fiducialess(&self) -> bool {
        Self::is_fiducialess(self)
    }

    fn get_image_rotation(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        Self::get_image_rotation(self, do_validation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Prenewst {
        advanced: bool,
        done: bool,
    }
    impl CoarseAlignPrenewstPanel for Prenewst {
        fn set_parameters_newst<T>(&mut self, _: &T) {}
        fn set_parameters_blendmont<T>(&mut self, _: &T) {}
        fn set_parameters_screen_state<T>(&mut self, _: &T) {}
        fn set_parameters_meta_data<T>(&mut self, _: &T) {}
        fn get_parameters_screen_state<T>(&self, _: &mut T) {}
        fn get_parameters_meta_data<T>(&self, _: &mut T) {}
        fn update_advanced(&mut self, state: bool) {
            self.advanced = state;
        }
        fn done(&mut self) {
            self.done = true;
        }
    }
    struct Manager {
        montage: bool,
        dc: bool,
        file: bool,
        calls: Vec<String>,
    }
    impl CoarseAlignDialogApplicationManager for Manager {
        fn view_type(&self) -> ViewType {
            if self.montage {
                ViewType::Montage
            } else {
                ViewType::SingleView
            }
        }
        fn distortion_correction(&self) -> bool {
            self.dc
        }
        fn property_user_dir(&self) -> &Path {
            Path::new(".")
        }
        fn distortion_corrected_file_exists(&self, _: AxisID) -> bool {
            self.file
        }
        fn midas_raw_stack(&mut self, _: AxisID, _: &MultiLineButton) {
            self.calls.push("midas".into())
        }
        fn midas_fix_edges(&mut self, _: AxisID, _: &MultiLineButton) {
            self.calls.push("edges".into())
        }
        fn make_distortion_corrected_stack(&mut self, _: AxisID, _: &MultiLineButton) {
            self.calls.push("dcst".into())
        }
        fn done_coarse_align_dialog(&mut self, _: AxisID) {
            self.calls.push("done".into())
        }
        fn pack(&mut self, _: AxisID) {
            self.calls.push("pack".into())
        }
    }
    fn button(label: &str) -> MultiLineButton {
        let mut button = MultiLineButton::new_with_label(Some(label));
        // `ProcessResultDisplayFactory` supplies action commands on these buttons.
        button.set_action_command(Some(label));
        button
    }
    fn dialog(manager: &Manager) -> CoarseAlignDialog<Prenewst> {
        CoarseAlignDialog::get_instance(
            manager,
            AxisID::Only,
            false,
            Prenewst::default(),
            button("DC"),
            button("Edges"),
            button("Midas"),
            true,
        )
    }
    #[test]
    fn montage_disables_distortion_and_enables_edges_without_file() {
        let manager = Manager {
            montage: true,
            dc: false,
            file: false,
            calls: vec![],
        };
        let dialog = dialog(&manager);
        assert!(dialog.pnl_coarse_align.fix_edges_present);
        assert!(!dialog.btn_distortion_corrected_stack.is_enabled());
        assert!(dialog.btn_fix_edges_midas.is_enabled());
    }
    #[test]
    fn action_routes_source_commands() {
        let mut manager = Manager {
            montage: false,
            dc: true,
            file: false,
            calls: vec![],
        };
        let mut dialog = dialog(&manager);
        let command = dialog.btn_midas.get_action_command().unwrap().to_owned();
        dialog.action(&mut manager, &command, None, None);
        assert_eq!(manager.calls, ["midas"]);
    }
    #[test]
    fn done_tears_down_children_and_hides_dialog() {
        let mut manager = Manager {
            montage: false,
            dc: true,
            file: false,
            calls: vec![],
        };
        let mut dialog = dialog(&manager);
        dialog.done(&mut manager);
        assert!(!dialog.displayed);
        assert!(dialog.pnl_prenewst.done);
        assert_eq!(manager.calls, ["done"]);
    }

    #[test]
    fn implements_canonical_fiducialess_params() {
        let manager = Manager {
            montage: false,
            dc: true,
            file: false,
            calls: vec![],
        };
        let mut dialog = dialog(&manager);
        dialog.set_fiducialess_alignment(true);
        dialog.set_image_rotation("4.25");
        let params: &dyn FiducialessParams = &dialog;
        assert!(params.is_fiducialess());
        assert_eq!(params.get_image_rotation(true).unwrap(), "4.25");
    }
}
