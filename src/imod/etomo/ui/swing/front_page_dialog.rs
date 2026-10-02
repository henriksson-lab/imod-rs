//! `IMOD/Etomo/src/etomo/ui/swing/FrontPageDialog.java`.
//!
//! Default display: the top-level dialog of `FrontPageManager`.  Contains
//! buttons for choosing one of the interfaces (reconstruction, join, PEET,
//! serial sections, nonlinear anisotropic diffusion, generic parallel, batch)
//! and the tools (flatten volume, GPU test, align frames).  Each button asks
//! `EtomoDirector` to open the matching manager; the director closes the
//! front page (its default window) first.
//!
//! The buttons are `MultiLineButton`s, which name themselves from their
//! labels (`bn.build-tomogram`, `bn.join-serial-tomograms`, ...), so a driver
//! finds them by those names under [`FrontPageDialog::get_root`].

use std::rc::{Rc, Weak};

use super::etomo_menu::{self, ToolType};
use super::multi_line_button::MultiLineButton;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::peet_manager::PeetManager;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java public final class `FrontPageDialog`.
pub struct FrontPageDialog {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java `btnRecon = new MultiLineButton(EtomoMenu.RECON_LABEL)`.
    btn_recon: Rc<MultiLineButton>,
    /// Java `btnJoin = new MultiLineButton(EtomoMenu.JOIN_LABEL)`.
    btn_join: Rc<MultiLineButton>,
    /// Java `btnNad = new MultiLineButton(EtomoMenu.NAD_LABEL)`.
    btn_nad: Rc<MultiLineButton>,
    /// Java `btnBatchRunTomo = new MultiLineButton(EtomoMenu.BATCH_RUN_TOMO_LABEL)`.
    btn_batch_run_tomo: Rc<MultiLineButton>,
    /// Java `btnGeneric = new MultiLineButton(EtomoMenu.GENERIC_LABEL)`.
    btn_generic: Rc<MultiLineButton>,
    /// Java `btnPeet = new MultiLineButton(EtomoMenu.PEET_LABEL)`.
    btn_peet: Rc<MultiLineButton>,
    /// Java `btnSerialSections = new MultiLineButton(EtomoMenu.SERIAL_SECTIONS_LABEL)`.
    btn_serial_sections: Rc<MultiLineButton>,
    /// Java `btnFlattenVolume = new MultiLineButton(EtomoMenu.FLATTEN_VOLUME_LABEL)`.
    btn_flatten_volume: Rc<MultiLineButton>,
    /// Java `btnGpuTiltTest = new MultiLineButton(EtomoMenu.GPU_TILT_TEST_LABEL)`.
    btn_gpu_tilt_test: Rc<MultiLineButton>,
    /// Java `btnAlignFrames = new MultiLineButton(EtomoMenu.ALIGN_FRAMES_LABEL)`.
    btn_align_frames: Rc<MultiLineButton>,

    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
}

impl FrontPageDialog {
    /// Java private constructor `FrontPageDialog(BaseManager, AxisID)`,
    /// together with the field initialisers.
    fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> FrontPageDialog {
        FrontPageDialog {
            pnl_root: JComponent::new_panel(),
            btn_recon: MultiLineButton::new_string(Some(etomo_menu::RECON_LABEL)),
            btn_join: MultiLineButton::new_string(Some(etomo_menu::JOIN_LABEL)),
            btn_nad: MultiLineButton::new_string(Some(etomo_menu::NAD_LABEL)),
            btn_batch_run_tomo: MultiLineButton::new_string(Some(etomo_menu::BATCH_RUN_TOMO_LABEL)),
            btn_generic: MultiLineButton::new_string(Some(etomo_menu::GENERIC_LABEL)),
            btn_peet: MultiLineButton::new_string(Some(etomo_menu::PEET_LABEL)),
            btn_serial_sections: MultiLineButton::new_string(Some(
                etomo_menu::SERIAL_SECTIONS_LABEL,
            )),
            btn_flatten_volume: MultiLineButton::new_string(Some(etomo_menu::FLATTEN_VOLUME_LABEL)),
            btn_gpu_tilt_test: MultiLineButton::new_string(Some(etomo_menu::GPU_TILT_TEST_LABEL)),
            btn_align_frames: MultiLineButton::new_string(Some(etomo_menu::ALIGN_FRAMES_LABEL)),
            manager,
            axis_id,
        }
    }

    /// Java `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> Rc<FrontPageDialog> {
        let instance = Rc::new(FrontPageDialog::new(manager, axis_id));
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// The root panel (Java `pnlRoot`), for a driver searching the dialog's
    /// components by name.  Rust-only accessor; the Java hands `pnlRoot` to
    /// `MainPanel.showProcess` in `show`.
    pub fn get_root(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_project_label = JComponent::new_panel();
        let pnl_projects = JComponent::new_panel();
        let pnl_tool_label = JComponent::new_panel();
        let pnl_tools = JComponent::new_panel();
        let pnl_batch_run_tomo = JComponent::new_panel();
        let pnl_align_frames = JComponent::new_panel();
        // root panel
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS));
        // pnlRoot.setAlignmentX(Box.LEFT_ALIGNMENT).  The rigid areas between the
        // children (FixedDim.x0_y3, x0_y7, x0_y10, x0_y3, x0_y7) are omitted.
        self.pnl_root.add(&pnl_project_label);
        self.pnl_root.add(&pnl_projects);
        self.pnl_root.add(&pnl_batch_run_tomo);
        self.pnl_root.add(&pnl_tool_label);
        self.pnl_root.add(&pnl_tools);
        self.pnl_root.add(&pnl_align_frames);
        // project label
        // Swing layout: pnlProjectLabel.setLayout(new BoxLayout(pnlProjectLabel,
        // BoxLayout.X_AXIS)).
        pnl_project_label.add(&JComponent::new_label("New project:"));
        // Swing layout: pnlProjectLabel.add(Box.createHorizontalGlue()).
        // projects
        // Swing layout: pnlProjects.setLayout(new GridLayout(3, 2, 7, 7)).
        pnl_projects.add(&self.btn_recon.get_component());
        pnl_projects.add(&self.btn_join.get_component());
        pnl_projects.add(&self.btn_peet.get_component());
        pnl_projects.add(&self.btn_serial_sections.get_component());
        pnl_projects.add(&self.btn_nad.get_component());
        pnl_projects.add(&self.btn_generic.get_component());
        // tool label
        // Swing layout: pnlToolLabel.setLayout(new BoxLayout(pnlToolLabel,
        // BoxLayout.X_AXIS)).
        pnl_tool_label.add(&JComponent::new_label("Tools:"));
        // Swing layout: pnlToolLabel.add(Box.createHorizontalGlue()).
        // tools
        // Swing layout: pnlTools.setLayout(new GridLayout(1, 2, 7, 7)).
        pnl_tools.add(&self.btn_flatten_volume.get_component());
        pnl_tools.add(&self.btn_gpu_tilt_test.get_component());
        // BatchRunTomo
        // Swing layout: pnlBatchRunTomo.setLayout(new BoxLayout(pnlBatchRunTomo,
        // BoxLayout.X_AXIS)); horizontal glue on both sides of the button.
        pnl_batch_run_tomo.add(&self.btn_batch_run_tomo.get_component());
        // AlignFrames
        // Swing layout: pnlAlignFrames.setLayout(new BoxLayout(pnlAlignFrames,
        // BoxLayout.X_AXIS)); horizontal glue on both sides of the button.
        pnl_align_frames.add(&self.btn_align_frames.get_component());
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let action_listener = Rc::new(FrontPageActionListener::new(Rc::downgrade(self)));
        for button in [
            &self.btn_recon,
            &self.btn_join,
            &self.btn_nad,
            &self.btn_batch_run_tomo,
            &self.btn_generic,
            &self.btn_peet,
            &self.btn_serial_sections,
            &self.btn_flatten_volume,
            &self.btn_gpu_tilt_test,
            &self.btn_align_frames,
        ] {
            let action_listener = action_listener.clone();
            button.add_action_listener(Rc::new(move |action_event: &ActionEvent| {
                action_listener.action_performed(action_event)
            }));
        }
    }

    /// Java `show()`.
    pub fn show(&self) {
        // Upstream bug fixed in translation (FrontPageDialog.java:171): Java
        // dereferences manager.getMainPanel() unchecked (null when headless);
        // here nothing is shown then.
        if let Some(main_panel) = self.manager.get_main_panel() {
            main_panel
                .main_panel()
                .show_process(&self.pnl_root, self.axis_id);
        }
        // Test: a commented-out block of Popup test instances follows in the
        // Java; it is not code.
    }

    /// Java private `action(ActionEvent)`.
    fn action(&self, action_event: &ActionEvent) {
        // Upstream bug fixed in translation (FrontPageDialog.java:200): Java calls
        // `actionCommand.equals(...)` on a possibly null command; a null command
        // matches no button here.
        let Some(action_command) = action_event.get_action_command() else {
            return;
        };
        let action_command = Some(action_command.to_owned());
        if action_command == self.btn_recon.get_action_command() {
            let _ =
                etomo_director::INSTANCE.open_tomogram_boolean_axis_id(true, Some(self.axis_id));
        } else if action_command == self.btn_join.get_action_command() {
            let _ = etomo_director::INSTANCE.open_join_boolean_axis_id(true, Some(self.axis_id));
        } else if action_command == self.btn_nad.get_action_command() {
            let _ = etomo_director::INSTANCE.open_anisotropic_diffusion(true, Some(self.axis_id));
        } else if action_command == self.btn_batch_run_tomo.get_action_command() {
            let _ = etomo_director::INSTANCE
                .open_batch_run_tomo_boolean_axis_id(true, Some(self.axis_id));
        } else if action_command == self.btn_generic.get_action_command() {
            let _ = etomo_director::INSTANCE.open_generic_parallel(true, Some(self.axis_id));
        } else if action_command == self.btn_peet.get_action_command() {
            // NEEDS etomo/PeetManager.java's static `isInterfaceAvailable()` as
            // `PeetManager::is_interface_available() -> bool` (peet_manager.rs).
            if PeetManager::is_interface_available() {
                let _ =
                    etomo_director::INSTANCE.open_peet_boolean_axis_id(true, Some(self.axis_id));
            }
        } else if action_command == self.btn_serial_sections.get_action_command() {
            let _ = etomo_director::INSTANCE
                .open_serial_sections_boolean_axis_id(true, Some(self.axis_id));
        } else if action_command == self.btn_flatten_volume.get_action_command() {
            let _ = etomo_director::INSTANCE.open_tool(true, ToolType::FlattenVolume);
        } else if action_command == self.btn_gpu_tilt_test.get_action_command() {
            let _ = etomo_director::INSTANCE.open_tool(true, ToolType::GpuTiltTest);
        } else if action_command == self.btn_align_frames.get_action_command() {
            let _ = etomo_director::INSTANCE.open_tool(true, ToolType::AlignFrames);
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
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

/// Java private static final class `FrontPageActionListener implements
/// ActionListener`.
struct FrontPageActionListener {
    /// Java private final `listenee`.  Weak: the dialog owns its buttons,
    /// which own this listener.
    listenee: Weak<FrontPageDialog>,
}

impl FrontPageActionListener {
    /// Java private constructor `FrontPageActionListener(FrontPageDialog)`.
    fn new(listenee: Weak<FrontPageDialog>) -> FrontPageActionListener {
        FrontPageActionListener { listenee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, action_event: &ActionEvent) {
        let Some(listenee) = self.listenee.upgrade() else {
            return;
        };
        listenee.action(action_event);
    }
}
