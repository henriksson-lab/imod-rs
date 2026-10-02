//! `IMOD/Etomo/src/etomo/ui/swing/AlignmentEstimationDialog.java`.
//!
//! Java `public final class AlignmentEstimationDialog extends ProcessDialog
//! implements ContextMenu, Run3dmodButtonContainer`: the Fine Alignment
//! dialog.  An EDT object: created as `Rc<Self>` by
//! [`AlignmentEstimationDialog::new`]; every method takes `&self`; the
//! `ProcessDialog` superclass is the embedded `base` (reached through
//! `Deref`), and the overridden `done()` is `ProcessDialogVirtual::done`.  The
//! inner listener class `AlignmentEstimationActionListner` is a closure holding
//! a weak reference to the dialog.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::context_menu::ContextMenu;
use super::context_popup::ContextPopup;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::multi_line_button::MultiLineButton;
use super::process_dialog::{ProcessDialog, ProcessDialogVirtual};
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{SpacedPanel, X_AXIS};
use super::tiltalign_panel::{TiltalignPanel, TiltalignParamsException};
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::restrictalign_param::RestrictalignParam;
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::comscript::tomodataplots_param::Task;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent, MouseListener};
use crate::imod::etomo::process::imod_process::{BeadFixerMode, Run3dmodMenuOptions};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;

/// Java `public final class AlignmentEstimationDialog extends ProcessDialog
/// implements ContextMenu, Run3dmodButtonContainer`.
pub struct AlignmentEstimationDialog {
    /// The `ProcessDialog` superclass.
    base: Rc<ProcessDialog>,
    /// Java private `pnlAlignEst = new EtomoPanel()`.
    pnl_align_est: Rc<EtomoPanel>,
    /// Java private `border = new BeveledBorder("Fine Alignment")`.
    border: BeveledBorder,
    /// Java private `pnlTiltalign`.
    pnl_tiltalign: Rc<TiltalignPanel>,
    /// Java private `panelButton = new JPanel()`.
    panel_button: Rc<JComponent>,
    /// Java private final `btnComputeAlignment`.
    btn_compute_alignment: Rc<MultiLineButton>,
    /// Java private `btnImod`.
    btn_imod: Rc<Run3dmodButton>,
    /// Java private `btnView3DModel`.
    btn_view_3d_model: Rc<MultiLineButton>,
    /// Java private `btnViewResiduals`.
    btn_view_residuals: Rc<Run3dmodButton>,
    /// Java private final `actionListener` (an
    /// `AlignmentEstimationActionListner`).
    action_listener: ActionListener,

    /// Java private `patchTracking = false`.
    patch_tracking: std::cell::Cell<bool>,
}

impl Deref for AlignmentEstimationDialog {
    type Target = ProcessDialog;
    fn deref(&self) -> &ProcessDialog {
        &self.base
    }
}

impl AlignmentEstimationDialog {
    /// Java public constructor `AlignmentEstimationDialog(ApplicationManager,
    /// AxisID)`.
    pub fn new(
        app_mgr: &'static ApplicationManager,
        axis_id: AxisID,
    ) -> Rc<AlignmentEstimationDialog> {
        let dialog = Rc::new_cyclic(|this: &Weak<AlignmentEstimationDialog>| {
            // super(appMgr, axisID, DialogType.FINE_ALIGNMENT)
            let base = ProcessDialog::new_application_manager_axis_id_dialog_type(
                app_mgr,
                axis_id,
                DialogType::FineAlignment,
            );
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            // Field initializers, in declaration order.
            let pnl_align_est = EtomoPanel::new();
            let border = BeveledBorder::new(Some("Fine Alignment"));
            let panel_button = JComponent::new_panel();
            let btn_imod = Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                Some("View/Edit Fiducial Model"),
                Some(container.clone()),
            );
            let btn_view_3d_model = MultiLineButton::new_string(Some("View 3D Model"));
            let btn_view_residuals =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Residual Vectors"),
                    Some(container.clone()),
                );

            // Constructor body.
            // Java casts `(MultiLineButton) ...getComputeAlignment()`; the
            // factory returns the concrete button.
            let btn_compute_alignment = app_mgr
                .get_process_result_display_factory(axis_id)
                .get_compute_alignment();
            let pnl_tiltalign = TiltalignPanel::get_instance(axis_id, app_mgr, &base.btn_advanced);
            base.btn_execute.set_text(Some("Done"));

            // Create the first tiltalign panel
            // Swing layout: panelButton BoxLayout Y_AXIS.

            let top_button_panel = SpacedPanel::get_instance_void();
            top_button_panel.set_box_layout(X_AXIS);
            top_button_panel.add_multi_line_button(&btn_compute_alignment);
            top_button_panel.add_multi_line_button(&btn_imod);
            panel_button.add(&top_button_panel.get_container());
            // Swing layout: panelButton.add(Box.createRigidArea(FixedDim.x0_y10)).
            let bottom_button_panel = SpacedPanel::get_instance_void();
            bottom_button_panel.set_box_layout(X_AXIS);
            bottom_button_panel.add_multi_line_button(&btn_view_3d_model);
            // panelButton.add(Box.createRigidArea(FixedDim.x10_y0));
            bottom_button_panel.add_multi_line_button(&btn_view_residuals);
            panel_button.add(&bottom_button_panel.get_container());

            // Swing layout: pnlAlignEst BoxLayout Y_AXIS.
            pnl_align_est.set_border(&border.get_border());

            let align_est = pnl_align_est.get_component();
            align_est.add(&pnl_tiltalign.get_container());
            // Swing layout: pnlAlignEst.add(Box.createRigidArea(FixedDim.x5_y0)).
            align_est.add(&panel_button);

            // Construct the main panel from the alignment panel and exist buttons
            // Swing layout: rootPanel BoxLayout Y_AXIS.
            // Java `new JScrollPane(pnlAlignEst)` is never used: the next statement
            // moves pnlAlignEst out of its viewport into rootPanel.
            let _scroll_pane = JComponent::new_scroll_pane(Some(&align_est));
            // rootPanel.add(pnlAlignEst, BorderLayout.CENTER)
            base.root_panel.get_component().add(&align_est);
            base.add_exit_buttons();

            // Bind the action listeners to the buttons
            // Java `actionListener = new AlignmentEstimationActionListner(this)`:
            // its `actionPerformed` calls `adaptee.action(event.getActionCommand(),
            // null, null)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });

            AlignmentEstimationDialog {
                base,
                pnl_align_est,
                border,
                pnl_tiltalign,
                panel_button,
                btn_compute_alignment,
                btn_imod,
                btn_view_3d_model,
                btn_view_residuals,
                action_listener,
                patch_tracking: std::cell::Cell::new(false),
            }
        });
        // Java `this` as the ProcessDialog subclass (for the virtual `done()`).
        let this: Weak<dyn ProcessDialogVirtual> =
            Rc::downgrade(&dialog) as Weak<dyn ProcessDialogVirtual>;
        dialog.base.set_this(this);

        // The rest of the Java constructor, which needs the constructed dialog.
        dialog
            .btn_compute_alignment
            .add_action_listener(dialog.action_listener.clone());
        dialog
            .btn_view_3d_model
            .add_action_listener(dialog.action_listener.clone());
        dialog
            .btn_view_residuals
            .add_action_listener(dialog.action_listener.clone());
        dialog
            .btn_imod
            .add_action_listener(dialog.action_listener.clone());

        // Mouse adapter for context menu
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(&dialog) as Weak<dyn ContextMenu>;
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        dialog
            .root_panel
            .get_component()
            .add_mouse_listener(mouse_adapter.clone());
        dialog
            .pnl_tiltalign
            .get_container()
            .add_mouse_listener(mouse_adapter);

        // Set the default advanced state
        dialog.update_advanced();
        dialog.pnl_tiltalign.set_first_tab();
        dialog.set_tool_tip_text();
        dialog
    }

    /// Java `setParameters(BaseScreenState)`.
    pub fn set_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.pnl_tiltalign
            .set_parameters_base_screen_state(screen_state);
    }

    /// Java `getParameters(RestrictalignParam, boolean)`.
    pub fn get_parameters_restrictalign_param_boolean(
        &self,
        param: &mut RestrictalignParam,
        do_validation: bool,
    ) -> bool {
        self.pnl_tiltalign
            .get_parameters_restrictalign_param_boolean(param, do_validation)
    }

    /// Java `setPatchTracking(boolean)`.
    pub fn set_patch_tracking(&self, input: bool) {
        self.patch_tracking.set(input);
        self.pnl_tiltalign.set_patch_tracking(input);
    }

    /// Java `setSurfacesToAnalyze(int)`.
    pub fn set_surfaces_to_analyze(&self, surfaces_to_analyze: i32) {
        self.pnl_tiltalign
            .set_surfaces_to_analyze(surfaces_to_analyze);
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.pnl_tiltalign
            .get_parameters_base_screen_state(screen_state);
    }

    /// Java `setDefaultParameters()`.
    pub fn set_default_parameters(&self) {
        self.pnl_tiltalign.set_default_parameters();
    }

    /// Java `setParameters(ConstMetaData)`.  `TiltalignPanel.setParameters`
    /// takes the concrete metadata in the translation.
    pub fn set_parameters_const_meta_data(&self, meta_data: &MetaData) {
        self.pnl_tiltalign.set_parameters_const_meta_data(meta_data);
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        self.pnl_tiltalign.get_parameters_meta_data(meta_data);
    }

    /// Java `setTiltalignParams(TiltalignParam)`.
    pub fn set_tiltalign_params(&self, tiltalign_param: &TiltalignParam) {
        self.pnl_tiltalign
            .set_parameters_const_tiltalign_param(tiltalign_param);
    }

    /// Java `setRestrictalignParams(RestrictalignParam)`.
    pub fn set_restrictalign_params(&self, restrictalign_param: &RestrictalignParam) {
        self.pnl_tiltalign
            .set_parameters_restrictalign_param(restrictalign_param);
    }

    /// Java `getTiltalignParams(TiltalignParam, boolean) throws
    /// FortranInputSyntaxException`.
    ///
    /// The panel's unchecked `NumberFormatException` is not caught by the Java
    /// and propagates; it is passed through unchanged here.
    pub fn get_tiltalign_params(
        &self,
        tiltalign_param: &mut TiltalignParam,
        do_validation: bool,
    ) -> Result<bool, TiltalignParamsException> {
        match self
            .pnl_tiltalign
            .get_parameters_tiltalign_param_boolean(tiltalign_param, do_validation)
        {
            Ok(result) => {
                if !result {
                    return Ok(false);
                }
            }
            Err(TiltalignParamsException::FortranInputSyntaxException(except)) => {
                let message = format!(
                    "Axis: {}{}",
                    self.axis_id.get_extension(),
                    except.get_message().unwrap_or("null")
                );
                return Err(TiltalignParamsException::FortranInputSyntaxException(
                    FortranInputSyntaxException::new(&message),
                ));
            }
            Err(except) => return Err(except),
        }
        Ok(true)
    }

    /// Java `getParameters(MakecomfileParam, boolean)`.
    pub fn get_parameters_makecomfile_param_boolean(
        &self,
        param: &mut MakecomfileParam,
        do_validation: bool,
    ) -> bool {
        if !self
            .pnl_tiltalign
            .get_parameters_makecomfile_param_boolean(param, do_validation)
        {
            return false;
        }
        true
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.pnl_tiltalign.is_valid()
    }

    /// Java private `addLogFileTab(String, String, List<String>, List<String>)`.
    /// Adds a log file to logFileList and labelList, if the log file is not
    /// empty.
    fn add_log_file_tab(
        &self,
        log_file_name: &str,
        label: &str,
        log_file_list: &mut Vec<String>,
        label_list: &mut Vec<String>,
    ) {
        let name = format!("{}{}.log", log_file_name, self.axis_id.get_extension());
        // Java `new File(applicationManager.getPropertyUserDir(), name)`: a null
        // parent is the name alone; `File.length()` is 0 for a missing file.
        let log = match self.application_manager.get_property_user_dir() {
            Some(user_dir) => std::path::Path::new(&user_dir).join(&name),
            None => std::path::PathBuf::from(&name),
        };
        let length = std::fs::metadata(&log).map(|m| m.len()).unwrap_or(0);
        if length > 10 {
            log_file_list.push(name);
            label_list.push(label.to_string());
        }
    }

    /// Java private `updateAdvanced()`.  This is a separate function so it can
    /// be called at initialization time as well as from the button action
    /// above.
    fn update_advanced(&self) {
        self.pnl_tiltalign.update_advanced(self.is_advanced());
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text for the
    /// axis panel objects.
    fn set_tool_tip_text(&self) {
        self.btn_compute_alignment
            .set_tool_tip_text(Some("Run Tiltalign with current parameters."));
        self.btn_imod
            .set_tool_tip_text(Some("View fiducial model on the image stack in 3dmod."));
        self.btn_view_3d_model.set_tool_tip_text(Some(
            "View model of solved 3D locations of fiducial points in 3dmodv.",
        ));
        self.btn_view_residuals.set_tool_tip_text(Some(
            "Show model of residual vectors (exaggerated 10x) on the image stack.",
        ));
    }
}

impl ProcessDialogVirtual for AlignmentEstimationDialog {
    fn process_dialog(&self) -> &ProcessDialog {
        &self.base
    }

    /// Java `done()`.
    fn done(&self) {
        self.application_manager
            .done_alignment_estimation_dialog(self.axis_id);
        self.btn_compute_alignment
            .remove_action_listener(&self.action_listener);
        self.set_displayed(false);
    }
}

impl ContextMenu for AlignmentEstimationDialog {
    /// Java `popUpContextMenu(MouseEvent)`: right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel: Vec<String> = ["Tiltalign", "Restrict Align", "3dmod"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let man_page: Vec<String> = ["tiltalign.html", "restrictalign.html", "3dmod.html"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut log_file_set_window_label: Vec<String> = vec!["Align".to_string()];
        let log_file_label: Vec<String> = vec!["Restrict Align".to_string()];
        let log_file: Vec<String> =
            vec![format!("restrictalign{}.log", self.axis_id.get_extension())];

        if self.axis_id != AxisID::Only {
            log_file_set_window_label[0] = format!("Align Axis:{}", self.axis_id.get_extension());
        }
        let align_command_name = log_file_set_window_label[0].clone();
        // Add tabs
        let mut log_file_set_list: Vec<String> = Vec::new();
        let mut align_labels: Vec<String> = Vec::new();
        self.add_log_file_tab(
            "taRobust",
            "Robust",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taError",
            "Errors",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taSolution",
            "Solution",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taAngles",
            "Surface Angles",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taLocals",
            "Locals",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taResiduals",
            "Large Residual",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taMappings",
            "Mappings",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taCoordinates",
            "Coordinates",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "taBeamtilt",
            "Beam Tilt",
            &mut log_file_set_list,
            &mut align_labels,
        );
        self.add_log_file_tab(
            "align",
            "Complete Log",
            &mut log_file_set_list,
            &mut align_labels,
        );

        // Java `Vector logFileSet` / `logFileSetLabel`, one array each.
        let log_file_set: Vec<Vec<String>> = vec![log_file_set_list];
        let log_file_set_label: Vec<Vec<String>> = vec![align_labels];

        let graph = [
            Task::Rotation,
            Task::TiltSkew,
            Task::Mag,
            Task::Xstretch,
            Task::Resid,
            Task::AverResid,
        ];

        // Java `new ContextPopup(...)`; its constructor throws
        // IllegalArgumentException on mismatched lengths, which these arrays
        // cannot have.
        if let Err(message) = ContextPopup::new_component_mouse_event_string_string_array_string_array_string_array_vector_vector_string_array_string_array_task_array_file_array_application_manager_string_axis_id(
            &self.root_panel.get_component(),
            mouse_event,
            Some("FINAL ALIGNMENT"),
            &man_pagelabel,
            &man_page,
            &log_file_set_window_label,
            &log_file_set_label,
            &log_file_set,
            Some(log_file_label.as_slice()),
            Some(log_file.as_slice()),
            Some(&graph[..]),
            None,
            self.application_manager,
            &align_command_name,
            self.axis_id,
        ) {
            eprintln!("java.lang.IllegalArgumentException: {message}");
        }
    }
}

impl Run3dmodButtonContainer for AlignmentEstimationDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`: event
    /// handler for panel buttons.
    ///
    /// The action listener passes a null `Run3dmodMenuOptions`; the manager
    /// takes the options by value, and a default instance (every option off)
    /// is what the 3dmod layer makes of null.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let menu_options = run_3dmod_menu_options.unwrap_or_default();
        if Some(command) == self.btn_compute_alignment.get_action_command().as_deref() {
            self.application_manager.fine_alignment(
                self.axis_id,
                Some(self.btn_compute_alignment.clone() as ProcessResultDisplayHandle),
                None,
            );
        } else if Some(command) == self.btn_view_3d_model.get_action_command().as_deref() {
            self.application_manager
                .imod_view_model(self.axis_id, &file_type::CLASS.fiducial_3d_model);
        } else if Some(command) == self.btn_imod.get_action_command().as_deref() {
            self.application_manager.imod_fix_fiducials(
                self.axis_id,
                menu_options,
                None,
                if self.patch_tracking.get() {
                    BeadFixerMode::PatchTrackingResidualMode
                } else {
                    BeadFixerMode::ResidualMode
                },
                None,
            );
        } else if Some(command) == self.btn_view_residuals.get_action_command().as_deref() {
            self.application_manager
                .imod_view_residuals(self.axis_id, menu_options);
        }
    }
}
