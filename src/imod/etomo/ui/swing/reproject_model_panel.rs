//! `IMOD/Etomo/src/etomo/ui/swing/ReprojectModelPanel.java`.
//!
//! Java `final class ReprojectModelPanel implements TiltDisplay,
//! Run3dmodButtonContainer`: the Reproject Model buttons of the erase-gold
//! (findbeads3d) panel.  An EDT object created as `Rc<Self>` by
//! [`ReprojectModelPanel::get_instance`]; every method takes `&self`.  The
//! inner listener class `ReprojectModelPanelActionListener` is a closure
//! holding a weak reference to the panel.

use std::rc::{Rc, Weak};

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::process_display::ProcessDisplay;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::tilt_display::{TiltDisplay, TiltDisplayException};
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private static final `REPROJECT_MODEL_LABEL`.
pub const REPROJECT_MODEL_LABEL: &str = "Reproject Model";

/// Java `final class ReprojectModelPanel implements TiltDisplay,
/// Run3dmodButtonContainer`.
pub struct ReprojectModelPanel {
    /// Java private final `pnlRoot = SpacedPanel.getInstance(true)`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `actionListener`
    /// (`ReprojectModelPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `btn3dmodReprojectModel`.
    btn_3dmod_reproject_model: Rc<Run3dmodButton>,

    /// Java private final `btnReprojectModel`.
    btn_reproject_model: Rc<Run3dmodButton>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java `this`, for the containers the constructor hands out.
    this: Weak<ReprojectModelPanel>,
}

impl ReprojectModelPanel {
    /// Java private constructor `ReprojectModelPanel(ApplicationManager,
    /// AxisID, DialogType)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<ReprojectModelPanel> {
        Rc::new_cyclic(|this: &Weak<ReprojectModelPanel>| {
            // Field initializers.
            let pnl_root = SpacedPanel::get_instance_boolean(true);
            // ReprojectModelPanelActionListener
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_reproject_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View 2D Model on Aligned Stack"),
                    Some(container),
                );
            // Constructor body.  Java casts `(Run3dmodButton)
            // manager.getProcessResultDisplayFactory(axisID).getReprojectModel()`;
            // the factory returns the concrete button.
            let btn_reproject_model = manager
                .get_process_result_display_factory(axis_id)
                .get_reproject_model();
            ReprojectModelPanel {
                pnl_root,
                action_listener,
                btn_3dmod_reproject_model,
                btn_reproject_model,
                manager,
                axis_id,
                dialog_type,
                this: this.clone(),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<ReprojectModelPanel> {
        let instance = ReprojectModelPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_reproject_model
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_reproject_model
            .add_action_listener(self.action_listener.clone());
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Initialize
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_reproject_model.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_reproject_model.clone();
        self.btn_reproject_model
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        // Root panel
        // Swing layout: pnlRoot.setBoxLayout(BoxLayout.X_AXIS).
        self.pnl_root
            .add_component(&self.btn_reproject_model.get_component());
        self.pnl_root
            .add_component(&self.btn_3dmod_reproject_model.get_component());
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java `msgTiltComSaved()`; empty.
    pub fn msg_tilt_com_saved(&self) {}

    /// Java `done()`.
    pub fn done(&self) {
        self.btn_reproject_model
            .remove_action_listener(&self.action_listener);
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters(&self, screen_state: &ReconScreenState) {
        self.btn_reproject_model.set_button_state(
            screen_state
                .get_button_state(self.btn_reproject_model.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.btn_reproject_model
            .set_tool_tip_text(Some("Run tilt to reproject the model."));
        self.btn_3dmod_reproject_model
            .set_tool_tip_text(Some("View model of gold particles."));
    }
}

impl ProcessDisplay for ReprojectModelPanel {
    /// Java cast `(TiltDisplay) display`: this panel is one (the process
    /// series hands it back to `tilt3dFindAction` / the reprojection).
    fn as_tilt_display(&self) -> Option<&dyn super::tilt_display::TiltDisplay> {
        Some(self)
    }
}

impl TiltDisplay for ReprojectModelPanel {
    /// Java `getParameters(TiltParam, boolean)`.
    fn get_parameters(
        &self,
        _param: &mut TiltParam,
        _do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        Ok(true)
    }

    /// Java `getParameters(SplittiltParam, boolean)`.  Don't need parallel
    /// processing for reprojection of a model.
    fn get_parameters_splittilt(&self, _param: &mut SplittiltParam, _do_validation: bool) -> bool {
        false
    }

    /// Java `@Deprecated allowTiltComSave()` (8/3/2018 See TiltDisplay).
    fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java `setDebug(boolean)`; empty.
    fn set_debug(&self, _debug: bool) {}
}

impl Run3dmodButtonContainer for ReprojectModelPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_reproject_model.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_reproject_model.clone();
            // Java passes a possibly-null Run3dmodMenuOptions; the manager takes
            // the value, so null is the default (no options set).
            self.manager.reproject_model_action(
                Some(display),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options.unwrap_or_default(),
                Some(self as &dyn TiltDisplay),
                self.axis_id,
                self.dialog_type,
            );
        } else if Some(command)
            == self
                .btn_3dmod_reproject_model
                .get_action_command()
                .as_deref()
        {
            self.manager
                .imod_reproject_model(self.axis_id, run_3dmod_menu_options.unwrap_or_default());
        }
    }
}
