//! `IMOD/Etomo/src/etomo/ui/swing/AlignmentEstimationDialog.java`.
//!
//! Swing construction and `ApplicationManager` process calls stay at explicit
//! boundaries.  This unit retains the dialog-owned layout, button/action state,
//! tiltalign delegation, log-tab selection, and action routing.
#![allow(dead_code)]

use std::path::Path;

use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::process::imod_process::{BeadFixerMode, Run3dmodMenuOptions};
use crate::imod::etomo::r#type::axis_id::AxisID;

use super::context_popup::{ContextPopup, GraphTask, MouseEvent};
use super::multi_line_button::MultiLineButton;
use super::tiltalign_panel::{TiltalignPanel, TiltalignParameter};

pub const FINE_ALIGNMENT_BORDER: &str = "Fine Alignment";
pub const COMPUTE_ALIGNMENT: &str = "Compute Alignment";
pub const VIEW_EDIT_FIDUCIAL_MODEL: &str = "View/Edit Fiducial Model";
pub const VIEW_3D_MODEL: &str = "View 3D Model";
pub const VIEW_RESIDUAL_VECTORS: &str = "View Residual Vectors";

/// Direct `ApplicationManager` calls made by `AlignmentEstimationDialog`.
pub trait AlignmentEstimationDialogApplicationManager {
    fn fine_alignment(&mut self, axis_id: AxisID, button: &MultiLineButton);
    fn imod_view_model(&mut self, axis_id: AxisID);
    fn imod_fix_fiducials(
        &mut self,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
        mode: BeadFixerMode,
    );
    fn imod_view_residuals(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn done_alignment_estimation_dialog(&mut self, axis_id: AxisID);
    fn pack(&mut self, axis_id: AxisID);
}

/// `BaseScreenState` subset delegated unchanged to `TiltalignPanel`.
pub trait AlignmentEstimationScreenState: TiltalignParameter {}
impl<T: TiltalignParameter> AlignmentEstimationScreenState for T {}
/// Java `ConstMetaData` and `MetaData` at this dialog's boundary.
pub trait AlignmentEstimationMetaData: TiltalignParameter {}
impl<T: TiltalignParameter> AlignmentEstimationMetaData for T {}
/// Java `RestrictalignParam`, `TiltalignParam`, and `MakecomfileParam` boundary.
pub trait AlignmentEstimationParameter: TiltalignParameter {}
impl<T: TiltalignParameter> AlignmentEstimationParameter for T {}

/// Java's two `Run3dmodButton`s at the viewer boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run3dmodButton {
    pub label: String,
    pub action_command: String,
    pub tooltip: Option<String>,
    pub action_listener_count: usize,
}
impl Run3dmodButton {
    pub fn get_3dmod_instance(label: &str) -> Self {
        Self {
            label: label.into(),
            action_command: label.into(),
            tooltip: None,
            action_listener_count: 0,
        }
    }
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
    pub fn remove_action_listener(&mut self) {
        self.action_listener_count = self.action_listener_count.saturating_sub(1);
    }
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.tooltip = Some(text.into());
    }
}

/// Source-owned JPanel / BoxLayout construction state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AlignmentEstimationDialogLayout {
    pub panel_button_box_layout_y_axis: bool,
    pub alignment_panel_box_layout_y_axis: bool,
    pub alignment_panel_border: Option<String>,
    pub root_panel_box_layout_y_axis: bool,
    pub root_panel_order: Vec<String>,
    pub panel_button_order: Vec<String>,
    pub top_button_order: Vec<String>,
    pub bottom_button_order: Vec<String>,
    pub mouse_listener_count: usize,
}

/// Java final `AlignmentEstimationDialog` state.
pub struct AlignmentEstimationDialog {
    pub axis_id: AxisID,
    pub pnl_tiltalign: TiltalignPanel,
    pub layout: AlignmentEstimationDialogLayout,
    pub btn_compute_alignment: MultiLineButton,
    pub btn_imod: Run3dmodButton,
    pub btn_view_3d_model: MultiLineButton,
    pub btn_view_residuals: Run3dmodButton,
    pub action_listener_present: bool,
    pub patch_tracking: bool,
    pub advanced: bool,
    pub displayed: bool,
    pub execute_button_text: String,
    pub context_popup: Option<ContextPopup>,
}

impl AlignmentEstimationDialog {
    /// Java constructor.  The process-result factory is represented by its returned
    /// `btnComputeAlignment`, preserving factory ownership outside this dialog.
    pub fn new(
        axis_id: AxisID,
        mut btn_compute_alignment: MultiLineButton,
        advanced: bool,
    ) -> Self {
        btn_compute_alignment.set_text(COMPUTE_ALIGNMENT);
        if btn_compute_alignment.get_action_command().is_none() {
            btn_compute_alignment.set_action_command(Some(COMPUTE_ALIGNMENT));
        }
        let mut dialog = Self {
            axis_id,
            pnl_tiltalign: TiltalignPanel::get_instance(axis_id),
            layout: AlignmentEstimationDialogLayout {
                panel_button_box_layout_y_axis: true,
                alignment_panel_box_layout_y_axis: true,
                alignment_panel_border: Some(FINE_ALIGNMENT_BORDER.into()),
                root_panel_box_layout_y_axis: true,
                root_panel_order: vec!["pnlAlignEst".into(), "exitButtons".into()],
                panel_button_order: vec![
                    "topButtonPanel".into(),
                    "rigidArea(x0,y10)".into(),
                    "bottomButtonPanel".into(),
                ],
                top_button_order: vec!["btnComputeAlignment".into(), "btnImod".into()],
                bottom_button_order: vec!["btnView3DModel".into(), "btnViewResiduals".into()],
                mouse_listener_count: 2,
            },
            btn_compute_alignment,
            btn_imod: Run3dmodButton::get_3dmod_instance(VIEW_EDIT_FIDUCIAL_MODEL),
            btn_view_3d_model: MultiLineButton::new_with_label(Some(VIEW_3D_MODEL)),
            btn_view_residuals: Run3dmodButton::get_3dmod_instance(VIEW_RESIDUAL_VECTORS),
            action_listener_present: true,
            patch_tracking: false,
            advanced,
            displayed: true,
            execute_button_text: "Done".into(),
            context_popup: None,
        };
        dialog
            .btn_view_3d_model
            .set_action_command(Some(VIEW_3D_MODEL));
        dialog.btn_compute_alignment.add_action_listener();
        dialog.btn_view_3d_model.add_action_listener();
        dialog.btn_view_residuals.add_action_listener();
        dialog.btn_imod.add_action_listener();
        dialog.update_advanced_without_manager();
        dialog.pnl_tiltalign.set_first_tab();
        dialog.set_tool_tip_text();
        dialog
    }

    pub fn set_parameters<P: AlignmentEstimationScreenState>(&mut self, screen_state: &P) {
        self.pnl_tiltalign.set_parameters(screen_state);
    }
    pub fn get_restrictalign_parameters<P: AlignmentEstimationParameter>(
        &mut self,
        param: &mut P,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException> {
        self.pnl_tiltalign
            .get_parameters(param, do_validation)
            .map_err(|e| FortranInputSyntaxException::new(&e.to_string()))
    }
    pub fn set_patch_tracking(&mut self, input: bool) {
        self.patch_tracking = input;
        self.pnl_tiltalign.set_patch_tracking(input);
    }
    pub fn set_surfaces_to_analyze(&mut self, surfaces_to_analyze: i32) {
        self.pnl_tiltalign
            .set_surfaces_to_analyze(surfaces_to_analyze);
    }
    pub fn get_parameters<P: AlignmentEstimationScreenState>(&mut self, screen_state: &mut P) {
        let _ = self.pnl_tiltalign.get_parameters(screen_state, false);
    }
    pub fn set_default_parameters(&mut self) {
        self.pnl_tiltalign.set_default_parameters();
    }
    pub fn set_metadata_parameters<P: AlignmentEstimationMetaData>(&mut self, meta_data: &P) {
        self.pnl_tiltalign.set_parameters(meta_data);
    }
    pub fn get_metadata_parameters<P: AlignmentEstimationMetaData>(&mut self, meta_data: &mut P) {
        let _ = self.pnl_tiltalign.get_parameters(meta_data, false);
    }
    pub fn set_tiltalign_params<P: AlignmentEstimationParameter>(&mut self, param: &P) {
        self.pnl_tiltalign.set_parameters(param);
    }
    pub fn set_restrictalign_params<P: AlignmentEstimationParameter>(&mut self, param: &P) {
        self.pnl_tiltalign.set_parameters(param);
    }
    pub fn get_tiltalign_params<P: AlignmentEstimationParameter>(
        &mut self,
        param: &mut P,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException> {
        self.pnl_tiltalign
            .get_parameters(param, do_validation)
            .map_err(|except| {
                FortranInputSyntaxException::new(&format!(
                    "Axis: {}{}",
                    self.axis_id.get_extension(),
                    except
                ))
            })
    }
    pub fn get_makecomfile_parameters<P: AlignmentEstimationParameter>(
        &mut self,
        param: &mut P,
        do_validation: bool,
    ) -> Result<bool, FortranInputSyntaxException> {
        self.pnl_tiltalign
            .get_parameters(param, do_validation)
            .map_err(|e| FortranInputSyntaxException::new(&e.to_string()))
    }
    pub fn is_valid(&mut self) -> bool {
        self.pnl_tiltalign.is_valid()
    }

    /// Java private `addLogFileTab`.
    fn add_log_file_tab(
        &self,
        log_file_name: &str,
        label: &str,
        log_file_list: &mut Vec<String>,
        label_list: &mut Vec<String>,
        user_dir: &Path,
    ) {
        let name = format!("{}{}.log", log_file_name, self.axis_id.get_extension());
        if user_dir
            .join(&name)
            .metadata()
            .map(|metadata| metadata.len() > 10)
            .unwrap_or(false)
        {
            log_file_list.push(name);
            label_list.push(label.into());
        }
    }

    /// Java `popUpContextMenu`; `ContextPopup` retains the native popup boundary.
    pub fn pop_up_context_menu(
        &mut self,
        mouse_event: MouseEvent,
        user_dir: &Path,
    ) -> Result<(), String> {
        let man_page_label = vec!["Tiltalign".into(), "Restrict Align".into(), "3dmod".into()];
        let man_page = vec![
            "tiltalign.html".into(),
            "restrictalign.html".into(),
            "3dmod.html".into(),
        ];
        let window_label = if self.axis_id == AxisID::Only {
            "Align".into()
        } else {
            format!("Align Axis:{}", self.axis_id.get_extension())
        };
        let mut log_file_set = Vec::new();
        let mut align_labels = Vec::new();
        for (name, label) in [
            ("taRobust", "Robust"),
            ("taError", "Errors"),
            ("taSolution", "Solution"),
            ("taAngles", "Surface Angles"),
            ("taLocals", "Locals"),
            ("taResiduals", "Large Residual"),
            ("taMappings", "Mappings"),
            ("taCoordinates", "Coordinates"),
            ("taBeamtilt", "Beam Tilt"),
            ("align", "Complete Log"),
        ] {
            self.add_log_file_tab(name, label, &mut log_file_set, &mut align_labels, user_dir);
        }
        let graph = [
            "Rotation",
            "Tilt Skew",
            "Magnification",
            "X Stretch",
            "Residuals",
            "Average Residual",
        ]
        .into_iter()
        .map(|description| GraphTask {
            description: description.into(),
            available: true,
            input_file: None,
        })
        .collect::<Vec<_>>();
        self.context_popup = Some(ContextPopup::new_tabbed_log_files(
            mouse_event,
            Some("FINAL ALIGNMENT"),
            &man_page_label,
            &man_page,
            &[window_label.clone()],
            &[align_labels],
            &[log_file_set],
            &["Restrict Align".into()],
            &[format!("restrictalign{}.log", self.axis_id.get_extension())],
            &graph,
            Some(&window_label),
            self.axis_id,
        )?);
        Ok(())
    }
    pub fn done<M: AlignmentEstimationDialogApplicationManager>(&mut self, manager: &mut M) {
        manager.done_alignment_estimation_dialog(self.axis_id);
        self.btn_compute_alignment.remove_action_listener();
        self.displayed = false;
    }
    fn update_advanced_without_manager(&mut self) {
        self.pnl_tiltalign.update_advanced(self.advanced);
    }
    pub fn update_advanced<M: AlignmentEstimationDialogApplicationManager>(
        &mut self,
        manager: &mut M,
    ) {
        self.update_advanced_without_manager();
        manager.pack(self.axis_id);
    }
    pub fn action<M: AlignmentEstimationDialogApplicationManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if self.btn_compute_alignment.get_action_command() == Some(command) {
            manager.fine_alignment(self.axis_id, &self.btn_compute_alignment);
        } else if self.btn_view_3d_model.get_action_command() == Some(command) {
            manager.imod_view_model(self.axis_id);
        } else if self.btn_imod.action_command == command {
            manager.imod_fix_fiducials(
                self.axis_id,
                options,
                if self.patch_tracking {
                    BeadFixerMode::PatchTrackingResidualMode
                } else {
                    BeadFixerMode::ResidualMode
                },
            );
        } else if self.btn_view_residuals.action_command == command {
            manager.imod_view_residuals(self.axis_id, options);
        }
    }
    fn set_tool_tip_text(&mut self) {
        self.btn_compute_alignment
            .set_tool_tip_text(Some("Run Tiltalign with current parameters."));
        self.btn_imod
            .set_tool_tip_text("View fiducial model on the image stack in 3dmod.");
        self.btn_view_3d_model.set_tool_tip_text(Some(
            "View model of solved 3D locations of fiducial points in 3dmodv.",
        ));
        self.btn_view_residuals.set_tool_tip_text(
            "Show model of residual vectors (exaggerated 10x) on the image stack.",
        );
    }
}

/// Java private `AlignmentEstimationActionListner`.
pub struct AlignmentEstimationActionListener;
impl AlignmentEstimationActionListener {
    pub fn action_performed<M: AlignmentEstimationDialogApplicationManager>(
        dialog: &mut AlignmentEstimationDialog,
        manager: &mut M,
        action_command: &str,
    ) {
        dialog.action(manager, action_command, None);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tiltalign_panel::TiltalignPanelParameters;
    use super::*;
    #[derive(Default)]
    struct Manager {
        calls: Vec<String>,
    }
    impl AlignmentEstimationDialogApplicationManager for Manager {
        fn fine_alignment(&mut self, axis: AxisID, _: &MultiLineButton) {
            self.calls.push(format!("fine{}", axis.get_extension()));
        }
        fn imod_view_model(&mut self, _: AxisID) {
            self.calls.push("model".into());
        }
        fn imod_fix_fiducials(
            &mut self,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            mode: BeadFixerMode,
        ) {
            self.calls.push(format!("fix{:?}", mode));
        }
        fn imod_view_residuals(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.calls.push("residuals".into());
        }
        fn done_alignment_estimation_dialog(&mut self, _: AxisID) {
            self.calls.push("done".into());
        }
        fn pack(&mut self, _: AxisID) {
            self.calls.push("pack".into());
        }
    }
    #[test]
    fn constructor_preserves_layout_tooltips_and_tiltalign_initialization() {
        let dialog = AlignmentEstimationDialog::new(
            AxisID::First,
            MultiLineButton::new_with_label(Some(COMPUTE_ALIGNMENT)),
            true,
        );
        assert_eq!(
            dialog.layout.alignment_panel_border.as_deref(),
            Some(FINE_ALIGNMENT_BORDER)
        );
        assert_eq!(dialog.pnl_tiltalign.current_tab.index(), 0);
        assert!(dialog.pnl_tiltalign.advanced);
        assert!(
            dialog
                .btn_imod
                .tooltip
                .as_ref()
                .unwrap()
                .contains("fiducial")
        );
    }
    #[test]
    fn actions_preserve_source_routes_and_patch_tracking_mode() {
        let mut dialog = AlignmentEstimationDialog::new(
            AxisID::Only,
            MultiLineButton::new_with_label(Some(COMPUTE_ALIGNMENT)),
            false,
        );
        let mut manager = Manager::default();
        for command in [
            COMPUTE_ALIGNMENT,
            VIEW_3D_MODEL,
            VIEW_EDIT_FIDUCIAL_MODEL,
            VIEW_RESIDUAL_VECTORS,
        ] {
            dialog.action(&mut manager, command, None);
        }
        dialog.set_patch_tracking(true);
        dialog.action(&mut manager, VIEW_EDIT_FIDUCIAL_MODEL, None);
        assert_eq!(
            manager.calls,
            [
                "fine",
                "model",
                "fixResidualMode",
                "residuals",
                "fixPatchTrackingResidualMode"
            ]
        );
    }
    #[test]
    fn tiltalign_parameter_delegation_and_done_are_retained() {
        let mut dialog = AlignmentEstimationDialog::new(
            AxisID::Second,
            MultiLineButton::new_with_label(Some(COMPUTE_ALIGNMENT)),
            false,
        );
        let mut params = TiltalignPanelParameters::default();
        params
            .values
            .insert("residual_report_criterion".into(), "0".into());
        dialog.set_tiltalign_params(&params);
        assert_eq!(dialog.pnl_tiltalign.ltf_residual_threshold.get_text(), "0");
        let mut manager = Manager::default();
        dialog.update_advanced(&mut manager);
        dialog.done(&mut manager);
        assert!(!dialog.displayed);
        assert_eq!(dialog.btn_compute_alignment.button.action_listener_count, 0);
        assert_eq!(manager.calls, ["pack", "done"]);
    }
}
