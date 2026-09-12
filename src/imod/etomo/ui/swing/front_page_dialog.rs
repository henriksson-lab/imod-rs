//! `IMOD/Etomo/src/etomo/ui/swing/FrontPageDialog.java`.
//!
//! `JPanel`, `BoxLayout`, `GridLayout`, `JLabel`, `Box`, and action-listener
//! delivery remain native GUI boundaries.  The application opening calls on
//! `EtomoDirector.INSTANCE`, `PeetManager.isInterfaceAvailable`, and
//! `BaseManager.getMainPanel().showProcess` remain direct application
//! boundaries until those concrete GUI/director paths are represented here.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

use super::etomo_frame::ActionEvent;
use super::etomo_menu::{
    ALIGN_FRAMES_LABEL, BATCH_RUN_TOMO_LABEL, FLATTEN_VOLUME_LABEL, GENERIC_LABEL,
    GPU_TILT_TEST_LABEL, JOIN_LABEL, NAD_LABEL, PEET_LABEL, RECON_LABEL, SERIAL_SECTIONS_LABEL,
    ToolType,
};
use super::multi_line_button::MultiLineButton;

/// Direct `BaseManager.getMainPanel().showProcess(JPanel, AxisID)` boundary.
pub trait FrontPageDialogMainPanel {
    fn show_process(&mut self, pnl_root: &FrontPageDialogLayout, axis_id: AxisID);
}

/// Direct `EtomoDirector.INSTANCE` calls made by `action`.
pub trait FrontPageDialogDirector {
    fn open_tomogram(&mut self, new_window: bool, axis_id: AxisID);
    fn open_join(&mut self, new_window: bool, axis_id: AxisID);
    fn open_anisotropic_diffusion(&mut self, new_window: bool, axis_id: AxisID);
    fn open_batch_run_tomo(&mut self, new_window: bool, axis_id: AxisID);
    fn open_generic_parallel(&mut self, new_window: bool, axis_id: AxisID);
    fn open_peet(&mut self, new_window: bool, axis_id: AxisID);
    fn open_serial_sections(&mut self, new_window: bool, axis_id: AxisID);
    fn open_tool(&mut self, new_window: bool, tool_type: ToolType);
}

/// Direct static `PeetManager.isInterfaceAvailable()` boundary.
pub trait FrontPageDialogPeetManager {
    fn is_interface_available(&self) -> bool;
}

/// Source-visible `JPanel` hierarchy and ordered Swing additions from
/// `createPanel`.  Widget painting remains the native GUI boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrontPageDialogLayout {
    pub pnl_root_layout: Option<&'static str>,
    pub pnl_root_alignment_x: f32,
    pub pnl_root_order: Vec<&'static str>,
    pub pnl_project_label_layout: Option<&'static str>,
    pub pnl_project_label_order: Vec<&'static str>,
    pub pnl_projects_grid: Option<(i32, i32, i32, i32)>,
    pub pnl_projects_order: Vec<&'static str>,
    pub pnl_tool_label_layout: Option<&'static str>,
    pub pnl_tool_label_order: Vec<&'static str>,
    pub pnl_tools_grid: Option<(i32, i32, i32, i32)>,
    pub pnl_tools_order: Vec<&'static str>,
    pub pnl_batch_run_tomo_layout: Option<&'static str>,
    pub pnl_batch_run_tomo_order: Vec<&'static str>,
    pub pnl_align_frames_layout: Option<&'static str>,
    pub pnl_align_frames_order: Vec<&'static str>,
}

/// Java final `FrontPageDialog`.
pub struct FrontPageDialog {
    pub pnl_root: FrontPageDialogLayout,
    pub btn_recon: MultiLineButton,
    pub btn_join: MultiLineButton,
    pub btn_nad: MultiLineButton,
    pub btn_batch_run_tomo: MultiLineButton,
    pub btn_generic: MultiLineButton,
    pub btn_peet: MultiLineButton,
    pub btn_serial_sections: MultiLineButton,
    pub btn_flatten_volume: MultiLineButton,
    pub btn_gpu_tilt_test: MultiLineButton,
    pub btn_align_frames: MultiLineButton,
    pub manager: &'static dyn BaseManager,
    pub axis_id: AxisID,
}

impl FrontPageDialog {
    /// Java private `FrontPageDialog(BaseManager, AxisID)`.
    fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> Self {
        Self {
            pnl_root: FrontPageDialogLayout::default(),
            btn_recon: MultiLineButton::new_with_label(Some(RECON_LABEL)),
            btn_join: MultiLineButton::new_with_label(Some(JOIN_LABEL)),
            btn_nad: MultiLineButton::new_with_label(Some(NAD_LABEL)),
            btn_batch_run_tomo: MultiLineButton::new_with_label(Some(BATCH_RUN_TOMO_LABEL)),
            btn_generic: MultiLineButton::new_with_label(Some(GENERIC_LABEL)),
            btn_peet: MultiLineButton::new_with_label(Some(PEET_LABEL)),
            btn_serial_sections: MultiLineButton::new_with_label(Some(SERIAL_SECTIONS_LABEL)),
            btn_flatten_volume: MultiLineButton::new_with_label(Some(FLATTEN_VOLUME_LABEL)),
            btn_gpu_tilt_test: MultiLineButton::new_with_label(Some(GPU_TILT_TEST_LABEL)),
            btn_align_frames: MultiLineButton::new_with_label(Some(ALIGN_FRAMES_LABEL)),
            manager,
            axis_id,
        }
    }

    /// Java static `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> Self {
        let mut instance = Self::new(manager, axis_id);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.pnl_root.pnl_root_layout = Some("BoxLayout.Y_AXIS");
        self.pnl_root.pnl_root_alignment_x = 0.0;
        self.pnl_root.pnl_root_order = vec![
            "pnlProjectLabel",
            "FixedDim.x0_y3",
            "pnlProjects",
            "FixedDim.x0_y7",
            "pnlBatchRunTomo",
            "FixedDim.x0_y10",
            "pnlToolLabel",
            "FixedDim.x0_y3",
            "pnlTools",
            "FixedDim.x0_y7",
            "pnlAlignFrames",
        ];
        self.pnl_root.pnl_project_label_layout = Some("BoxLayout.X_AXIS");
        self.pnl_root.pnl_project_label_order = vec!["JLabel(New project:)", "HorizontalGlue"];
        self.pnl_root.pnl_projects_grid = Some((3, 2, 7, 7));
        self.pnl_root.pnl_projects_order = vec![
            "btnRecon",
            "btnJoin",
            "btnPeet",
            "btnSerialSections",
            "btnNad",
            "btnGeneric",
        ];
        self.pnl_root.pnl_tool_label_layout = Some("BoxLayout.X_AXIS");
        self.pnl_root.pnl_tool_label_order = vec!["JLabel(Tools:)", "HorizontalGlue"];
        self.pnl_root.pnl_tools_grid = Some((1, 2, 7, 7));
        self.pnl_root.pnl_tools_order = vec!["btnFlattenVolume", "btnGpuTiltTest"];
        self.pnl_root.pnl_batch_run_tomo_layout = Some("BoxLayout.X_AXIS");
        self.pnl_root.pnl_batch_run_tomo_order =
            vec!["HorizontalGlue", "btnBatchRunTomo", "HorizontalGlue"];
        self.pnl_root.pnl_align_frames_layout = Some("BoxLayout.X_AXIS");
        self.pnl_root.pnl_align_frames_order =
            vec!["HorizontalGlue", "btnAlignFrames", "HorizontalGlue"];
    }

    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        self.btn_recon.add_action_listener();
        self.btn_join.add_action_listener();
        self.btn_nad.add_action_listener();
        self.btn_batch_run_tomo.add_action_listener();
        self.btn_generic.add_action_listener();
        self.btn_peet.add_action_listener();
        self.btn_serial_sections.add_action_listener();
        self.btn_flatten_volume.add_action_listener();
        self.btn_gpu_tilt_test.add_action_listener();
        self.btn_align_frames.add_action_listener();
    }

    /// Java `show()`.
    pub fn show(&self, main_panel: &mut dyn FrontPageDialogMainPanel) {
        main_panel.show_process(&self.pnl_root, self.axis_id);
    }

    /// Java private `action(ActionEvent)`.
    fn action<D: FrontPageDialogDirector + ?Sized, P: FrontPageDialogPeetManager + ?Sized>(
        &self,
        action_event: &ActionEvent,
        director: &mut D,
        peet_manager: &P,
    ) {
        let action_command = &action_event.action_command;
        if Some(action_command.as_str()) == self.btn_recon.get_action_command() {
            director.open_tomogram(true, self.axis_id);
        } else if Some(action_command.as_str()) == self.btn_join.get_action_command() {
            director.open_join(true, self.axis_id);
        } else if Some(action_command.as_str()) == self.btn_nad.get_action_command() {
            director.open_anisotropic_diffusion(true, self.axis_id);
        } else if Some(action_command.as_str()) == self.btn_batch_run_tomo.get_action_command() {
            director.open_batch_run_tomo(true, self.axis_id);
        } else if Some(action_command.as_str()) == self.btn_generic.get_action_command() {
            director.open_generic_parallel(true, self.axis_id);
        } else if Some(action_command.as_str()) == self.btn_peet.get_action_command() {
            if peet_manager.is_interface_available() {
                director.open_peet(true, self.axis_id);
            }
        } else if Some(action_command.as_str()) == self.btn_serial_sections.get_action_command() {
            director.open_serial_sections(true, self.axis_id);
        } else if Some(action_command.as_str()) == self.btn_flatten_volume.get_action_command() {
            director.open_tool(true, ToolType::FlattenVolume);
        } else if Some(action_command.as_str()) == self.btn_gpu_tilt_test.get_action_command() {
            director.open_tool(true, ToolType::GpuTiltTest);
        } else if Some(action_command.as_str()) == self.btn_align_frames.get_action_command() {
            director.open_tool(true, ToolType::AlignFrames);
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&mut self) {
        self.btn_recon
            .set_tool_tip_text(Some("Start a new tomographic reconstruction."));
        self.btn_join.set_tool_tip_text(Some("Stack tomograms."));
        self.btn_nad.set_tool_tip_text(Some(
            "Run a nonlinear anisotropic diffusion process on a tomogram.",
        ));
        self.btn_batch_run_tomo.set_tool_tip_text(Some(
            "Use batchruntomo to create or start one or more tomograms.",
        ));
        self.btn_generic
            .set_tool_tip_text(Some("Run a generic parallel process."));
        self.btn_peet.set_tool_tip_text(Some(
            "Start the interface for the PEET particle averaging package.",
        ));
        self.btn_gpu_tilt_test.set_tool_tip_text(Some(
            "Test the reliability of GPU with repeated runs of the Tilt program",
        ));
    }
}

/// Java private static final `FrontPageActionListener`.
///
/// Swing retains this listener after `addListeners`; the native widget boundary
/// owns that retention, while this source-visible value retains Java's final
/// `listenee` reference for explicit native event delivery.
pub struct FrontPageActionListener<'a> {
    pub listenee: &'a FrontPageDialog,
}

impl<'a> FrontPageActionListener<'a> {
    /// Java private `FrontPageActionListener(FrontPageDialog)`.
    pub fn new(listenee: &'a FrontPageDialog) -> Self {
        Self { listenee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed<
        D: FrontPageDialogDirector + ?Sized,
        P: FrontPageDialogPeetManager + ?Sized,
    >(
        &self,
        action_event: &ActionEvent,
        director: &mut D,
        peet_manager: &P,
    ) {
        self.listenee.action(action_event, director, peet_manager);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;

    #[derive(Default)]
    struct MainPanel {
        shown: Option<(FrontPageDialogLayout, AxisID)>,
    }

    impl FrontPageDialogMainPanel for MainPanel {
        fn show_process(&mut self, pnl_root: &FrontPageDialogLayout, axis_id: AxisID) {
            self.shown = Some((pnl_root.clone(), axis_id));
        }
    }

    #[derive(Default)]
    struct Director {
        calls: Vec<String>,
    }

    impl FrontPageDialogDirector for Director {
        fn open_tomogram(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls
                .push(format!("tomogram:{new_window}:{axis_id:?}"));
        }
        fn open_join(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls.push(format!("join:{new_window}:{axis_id:?}"));
        }
        fn open_anisotropic_diffusion(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls.push(format!("nad:{new_window}:{axis_id:?}"));
        }
        fn open_batch_run_tomo(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls.push(format!("batch:{new_window}:{axis_id:?}"));
        }
        fn open_generic_parallel(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls.push(format!("generic:{new_window}:{axis_id:?}"));
        }
        fn open_peet(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls.push(format!("peet:{new_window}:{axis_id:?}"));
        }
        fn open_serial_sections(&mut self, new_window: bool, axis_id: AxisID) {
            self.calls.push(format!("serial:{new_window}:{axis_id:?}"));
        }
        fn open_tool(&mut self, new_window: bool, tool_type: ToolType) {
            self.calls.push(format!("tool:{new_window}:{tool_type:?}"));
        }
    }

    struct PeetManager(bool);
    impl FrontPageDialogPeetManager for PeetManager {
        fn is_interface_available(&self) -> bool {
            self.0
        }
    }

    fn dialog() -> FrontPageDialog {
        FrontPageDialog::get_instance(
            DirectiveEditorManager::new(None, None, None, None),
            AxisID::Second,
        )
    }

    #[test]
    fn source_construction_preserves_layout_tooltips_and_all_listeners() {
        let dialog = dialog();
        assert_eq!(dialog.pnl_root.pnl_root_layout, Some("BoxLayout.Y_AXIS"));
        assert_eq!(dialog.pnl_root.pnl_root_alignment_x, 0.0);
        assert_eq!(
            dialog.pnl_root.pnl_root_order,
            [
                "pnlProjectLabel",
                "FixedDim.x0_y3",
                "pnlProjects",
                "FixedDim.x0_y7",
                "pnlBatchRunTomo",
                "FixedDim.x0_y10",
                "pnlToolLabel",
                "FixedDim.x0_y3",
                "pnlTools",
                "FixedDim.x0_y7",
                "pnlAlignFrames",
            ]
        );
        assert_eq!(dialog.pnl_root.pnl_projects_grid, Some((3, 2, 7, 7)));
        assert_eq!(
            dialog.pnl_root.pnl_projects_order,
            [
                "btnRecon",
                "btnJoin",
                "btnPeet",
                "btnSerialSections",
                "btnNad",
                "btnGeneric",
            ]
        );
        assert_eq!(dialog.pnl_root.pnl_tools_grid, Some((1, 2, 7, 7)));
        assert_eq!(dialog.btn_recon.button.action_listener_count, 1);
        assert_eq!(dialog.btn_align_frames.button.action_listener_count, 1);
        assert_eq!(
            dialog.btn_nad.button.tooltip.as_deref(),
            Some("Run a nonlinear anisotropic diffusion process on a tomogram.")
        );
        assert!(dialog.btn_flatten_volume.button.tooltip.is_none());
    }

    #[test]
    fn source_show_and_actions_dispatch_every_button_and_gate_peet() {
        let dialog = dialog();
        let mut main_panel = MainPanel::default();
        dialog.show(&mut main_panel);
        assert_eq!(main_panel.shown.unwrap().1, AxisID::Second);

        let mut director = Director::default();
        let unavailable = PeetManager(false);
        let action_listener = FrontPageActionListener::new(&dialog);
        for command in [
            RECON_LABEL,
            JOIN_LABEL,
            NAD_LABEL,
            BATCH_RUN_TOMO_LABEL,
            GENERIC_LABEL,
            PEET_LABEL,
            SERIAL_SECTIONS_LABEL,
            FLATTEN_VOLUME_LABEL,
            GPU_TILT_TEST_LABEL,
            ALIGN_FRAMES_LABEL,
            "ignored",
        ] {
            action_listener.action_performed(
                &ActionEvent::new(command),
                &mut director,
                &unavailable,
            );
        }
        assert_eq!(
            director.calls,
            [
                "tomogram:true:Second",
                "join:true:Second",
                "nad:true:Second",
                "batch:true:Second",
                "generic:true:Second",
                "serial:true:Second",
                "tool:true:FlattenVolume",
                "tool:true:GpuTiltTest",
                "tool:true:AlignFrames",
            ]
        );

        action_listener.action_performed(
            &ActionEvent::new(PEET_LABEL),
            &mut director,
            &PeetManager(true),
        );
        assert_eq!(director.calls.last(), Some(&"peet:true:Second".into()));
    }
}
