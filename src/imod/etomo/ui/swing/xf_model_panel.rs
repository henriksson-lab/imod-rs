//! `IMOD/Etomo/src/etomo/ui/swing/XfModelPanel.java`.
//!
//! Java `final class XfModelPanel implements Run3dmodButtonContainer`: the
//! "use the existing fiducial model" part of the Erase Gold tab (runs
//! xfmodel.com and views the transformed `_erase.fid` model).
//!
//! An EDT object (`Rc<Self>`, `&self` methods).  The inner listener class
//! `XfModelPanelActionListener` is a closure holding a weak reference to the
//! panel.

use std::rc::{Rc, Weak};

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `BoxLayout.X_AXIS` (for `SpacedPanel.setBoxLayout`).
const X_AXIS: i32 = 0;

/// Java `final class XfModelPanel implements Run3dmodButtonContainer`.
pub struct XfModelPanel {
    /// Java `this`.
    this: Weak<XfModelPanel>,
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `actionListener` (`XfModelPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `btn3dmodXfModel`.
    btn_3dmod_xf_model: Rc<Run3dmodButton>,

    /// Java private final `btnXfModel`.
    btn_xf_model: Rc<Run3dmodButton>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
}

impl XfModelPanel {
    /// Java private constructor `XfModelPanel(ApplicationManager, AxisID,
    /// DialogType)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<XfModelPanel> {
        Rc::new_cyclic(|this: &Weak<XfModelPanel>| {
            // Field initializers, in declaration order.
            let pnl_root = SpacedPanel::get_instance_void();
            // Java `new XfModelPanelActionListener(this)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_xf_model =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Transformed Model"),
                    Some(container),
                );
            // Constructor body.  Java casts `(Run3dmodButton) ...getXfModel()`; the
            // factory returns the concrete button.
            let btn_xf_model = manager
                .get_process_result_display_factory(axis_id)
                .get_xf_model();
            XfModelPanel {
                this: this.clone(),
                pnl_root,
                action_listener,
                btn_3dmod_xf_model,
                btn_xf_model,
                manager,
                axis_id,
                dialog_type,
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<XfModelPanel> {
        let instance = XfModelPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_xf_model
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_xf_model
            .add_action_listener(self.action_listener.clone());
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Initialize
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_xf_model.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_xf_model.clone();
        self.btn_xf_model
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        // Root panel
        self.pnl_root.set_box_layout(X_AXIS);
        self.pnl_root
            .add_component(&self.btn_xf_model.get_component());
        self.pnl_root
            .add_component(&self.btn_3dmod_xf_model.get_component());
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_xf_model
            .remove_action_listener(&self.action_listener);
    }

    /// Java package-private `setParameters(ReconScreenState)`.
    pub fn set_parameters(&self, screen_state: &ReconScreenState) {
        self.btn_xf_model.set_button_state(
            screen_state.get_button_state(self.btn_xf_model.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.btn_xf_model.set_tool_tip_text(Some(
            "Transform .fid mode built on prealigned stack to _erase.fid model that fits the \
             aligned stack.",
        ));
        self.btn_3dmod_xf_model
            .set_tool_tip_text(Some("View the _erase.fid model on the aligned stack."));
    }
}

impl Run3dmodButtonContainer for XfModelPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_xf_model.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_xf_model.clone();
            self.manager
                .xfmodel_process_result_display_process_series_deferred3dmod_button_run3dmod_menu_options_axis_id_dialog_type(
                    Some(display),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options,
                    self.axis_id,
                    self.dialog_type,
                );
        } else if Some(command) == self.btn_3dmod_xf_model.get_action_command().as_deref() {
            self.manager.seed_erase_fiducial_model(
                run_3dmod_menu_options,
                self.axis_id,
                self.dialog_type,
            );
        }
    }
}
