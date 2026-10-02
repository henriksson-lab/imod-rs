//! `IMOD/Etomo/src/etomo/ui/swing/NewstackOrBlendmontPanel.java`.
//!
//! The abstract base of `NewstackPanel` and `BlendmontPanel` (the full aligned
//! stack panels of the final aligned stack dialog).
//!
//! **Representation.**  A subclass embeds this struct as its `base` field
//! (reached through `Deref`) and implements [`NewstackOrBlendmontPanelVirtual`]
//! for the members Java leaves abstract.  Java passes `this` (the subclass
//! object) as the `Expandable` of the panel header and as the
//! `Run3dmodButtonContainer` of the two buttons, so a subclass also implements
//! `Expandable` by delegating to `base.expand_*` and `Run3dmodButtonContainer`
//! with its `action` override; the trait bounds below say so.
//!
//! Java calls the abstract `getHeaderTitle()` from this constructor
//! (NewstackOrBlendmontPanel.java:83), before the subclass object exists.  A
//! Rust object cannot dispatch to itself before it is built, so the subclass
//! passes the value its override returns (both overrides return a constant:
//! `"Newstack"`, `"Blendmont"`).

use std::cell::RefCell;
use std::io;
use std::rc::{Rc, Weak};

use super::blendmont_display::{BlendmontDisplay, BlendmontDisplayException};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::fiducialess_params::FiducialessParams;
use super::global_expand_button::GlobalExpandButton;
use super::newstack_and_blendmont_param_panel::NewstackAndBlendmontParamPanel;
use super::newstack_display::{NewstackDisplay, NewstackDisplayException};
use super::panel_header::PanelHeader;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;

/// Java `static final String RUN_BUTTON_LABEL`.
pub const RUN_BUTTON_LABEL: &str = "Create Full Aligned Stack";

/// The members `NewstackOrBlendmontPanel` declares abstract and calls on
/// `this`: `getHeaderTitle()` and `action(String, Deferred3dmodButton,
/// Run3dmodMenuOptions)` (the latter is `Run3dmodButtonContainer.action`).
pub trait NewstackOrBlendmontPanelVirtual: Run3dmodButtonContainer + Expandable {
    /// Java `abstract String getHeaderTitle()` (NewstackOrBlendmontPanel.java:91).
    fn get_header_title(&self) -> String;
}

/// Java `abstract class NewstackOrBlendmontPanel implements
/// Run3dmodButtonContainer, Expandable, NewstackDisplay, BlendmontDisplay`.
pub struct NewstackOrBlendmontPanel {
    pnl_root: Rc<JComponent>,

    /// Java `actionListener` (a `NewstackOrBlendmontPanelActionListener`).
    action_listener: ActionListener,
    header: Option<Rc<PanelHeader>>,
    pnl_body: Rc<SpacedPanel>,
    btn_3dmod_full: Rc<Run3dmodButton>,

    newstack_and_blendmont_param_panel: Rc<NewstackAndBlendmontParamPanel>,
    btn_run_process: Rc<Run3dmodButton>,
    pub axis_id: AxisID,
    pub manager: &'static ApplicationManager,
    pub dialog_type: DialogType,
    /// The subclass object (Java `this`).
    pub this: RefCell<Weak<dyn NewstackOrBlendmontPanelVirtual>>,
}

impl NewstackOrBlendmontPanel {
    /// Java `NewstackOrBlendmontPanel(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton)` (NewstackOrBlendmontPanel.java:78-89).  `this` is
    /// the subclass being built (`Rc::new_cyclic`'s weak), `header_title` the
    /// value of its `getHeaderTitle()`.
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
        this: Weak<dyn NewstackOrBlendmontPanelVirtual>,
        header_title: &str,
    ) -> NewstackOrBlendmontPanel {
        // Field initializers (NewstackOrBlendmontPanel.java:63-70).
        let pnl_root = JComponent::new_panel();
        // Java `new NewstackOrBlendmontPanelActionListener(this)`; the class is
        // at NewstackOrBlendmontPanel.java:257-270.  `adaptee.action` is the
        // subclass override.
        let adaptee = this.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            let Some(adaptee) = adaptee.upgrade() else {
                return;
            };
            adaptee.action(event.get_action_command().unwrap_or(""), None, None);
        });
        let pnl_body = SpacedPanel::get_instance_void();
        let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
        let btn_3dmod_full = Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
            Some("View Full Aligned Stack"),
            Some(container),
        );

        // Constructor body.
        let expandable: Weak<dyn Expandable> = this.clone();
        let header = PanelHeader::get_advanced_basic_only_instance(
            Some(header_title),
            Some(expandable),
            Some(dialog_type),
            Some(global_advanced_button.clone()),
            false,
        );
        let newstack_and_blendmont_param_panel =
            NewstackAndBlendmontParamPanel::get_instance(manager, axis_id, dialog_type);
        let btn_run_process = manager
            .get_process_result_display_factory(axis_id)
            .get_full_aligned_stack();
        NewstackOrBlendmontPanel {
            pnl_root,
            action_listener,
            header: Some(header),
            pnl_body,
            btn_3dmod_full,
            newstack_and_blendmont_param_panel,
            btn_run_process,
            axis_id,
            manager,
            dialog_type,
            this: RefCell::new(this),
        }
    }

    /// Java `addListeners()` (NewstackOrBlendmontPanel.java:93-96).
    pub fn add_listeners(&self) {
        self.btn_run_process
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_full
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `getComponent()` (NewstackOrBlendmontPanel.java:98-100).
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `createPanel()` (NewstackOrBlendmontPanel.java:102-123).
    pub fn create_panel(&self) {
        // Initialize
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.borrow().clone();
        self.btn_run_process.set_container(Some(container));
        self.btn_run_process
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_3dmod_full.clone() as Rc<dyn Deferred3dmodButton>
            ));
        // Local panels
        let pnl_buttons = SpacedPanel::get_instance_void();
        // Root panel
        // Swing layout: pnlRoot BoxLayout Y_AXIS, center aligned, untitled
        // etched border.
        self.pnl_root
            .add(&self.header.as_ref().unwrap().get_container());
        self.pnl_root.add(&self.pnl_body.get_container());
        // Swing layout: left align pnlRoot's components.
        // Body Panel
        // Swing layout: pnlBody.setBoxLayout(BoxLayout.Y_AXIS).
        self.pnl_body
            .add_component(&self.newstack_and_blendmont_param_panel.get_component());
        self.pnl_body.add_spaced_panel(&pnl_buttons);
        // Button panel
        // Swing layout: pnlButtons.setBoxLayout(BoxLayout.X_AXIS).
        pnl_buttons.add_component(&self.btn_run_process.get_component());
        pnl_buttons.add_component(&self.btn_3dmod_full.get_component());
    }

    /// Java private `setVisible(boolean)` (NewstackOrBlendmontPanel.java:130-132).
    fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java `done()` (NewstackOrBlendmontPanel.java:134-136).
    pub fn done(&self) {
        self.btn_run_process
            .remove_action_listener(&self.action_listener);
    }

    /// Java `getFiducialessParams()` (NewstackOrBlendmontPanel.java:138-140).
    pub fn get_fiducialess_params(&self) -> Rc<dyn FiducialessParams> {
        self.newstack_and_blendmont_param_panel.clone()
    }

    /// Java `final setParameters(ReconScreenState)` (NewstackOrBlendmontPanel.java:142-146).
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .as_ref()
            .unwrap()
            .set_state(Some(screen_state.get_newst_header_state()));
        self.btn_run_process.set_button_state(
            screen_state.get_button_state(self.btn_run_process.get_button_state_key().as_deref()),
        );
    }

    /// Java `final getParameters(ReconScreenState)` (NewstackOrBlendmontPanel.java:148-150).
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .as_ref()
            .unwrap()
            .get_state(Some(screen_state.get_newst_header_state()));
    }

    /// Java `getParameters(MetaData) throws FortranInputSyntaxException`
    /// (NewstackOrBlendmontPanel.java:183-185).  The Metadata values that are
    /// from the setup dialog should not be overrided by this dialog unless the
    /// Metadata values are empty.  Must save data from the two instances under
    /// separate keys.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        self.newstack_and_blendmont_param_panel
            .get_parameters_meta_data(meta_data)
    }

    /// Java `setParameters(ConstMetaData)` (NewstackOrBlendmontPanel.java:191-193).
    /// Must save data from the two instances under separate keys.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.newstack_and_blendmont_param_panel
            .set_parameters_const_meta_data(meta_data);
    }

    /// Java `setFiducialessAlignment(boolean)` (NewstackOrBlendmontPanel.java:195-197).
    pub fn set_fiducialess_alignment(&self, input: bool) {
        self.newstack_and_blendmont_param_panel
            .set_fiducialess_alignment(input);
    }

    /// Java `setImageRotation(String)` (NewstackOrBlendmontPanel.java:199-201).
    pub fn set_image_rotation(&self, input: Option<&str>) {
        self.newstack_and_blendmont_param_panel
            .set_image_rotation(input);
    }

    /// Java `getRunProcessButtonActionCommand()` (NewstackOrBlendmontPanel.java:208-210).
    pub fn get_run_process_button_action_command(&self) -> Option<String> {
        self.btn_run_process.get_action_command()
    }

    /// Java `get3dmodFullButtonActionCommand()` (NewstackOrBlendmontPanel.java:212-214).
    pub fn get3dmod_full_button_action_command(&self) -> Option<String> {
        self.btn_3dmod_full.get_action_command()
    }

    /// Java `getRunProcessResultDisplay()` (NewstackOrBlendmontPanel.java:216-218).
    pub fn get_run_process_result_display(&self) -> ProcessResultDisplayHandle {
        self.btn_run_process.clone()
    }

    /// Java `expand(GlobalExpandButton)` (NewstackOrBlendmontPanel.java:234-235).
    /// A subclass's `Expandable` implementation delegates here.
    pub fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)` (NewstackOrBlendmontPanel.java:237-245).
    /// A subclass's `Expandable` implementation delegates here.
    pub fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if let Some(header) = &self.header {
            if header.equals_advanced_basic(button) {
                self.newstack_and_blendmont_param_panel
                    .update_advanced(button.is_expanded());
            }
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }

    /// Java `updateAdvanced(boolean)` (NewstackOrBlendmontPanel.java:247-249).
    pub fn update_advanced(&self, advanced: bool) {
        self.newstack_and_blendmont_param_panel
            .update_advanced(advanced);
    }

    /// Java `setToolTipText()` (NewstackOrBlendmontPanel.java:251-255).
    pub fn set_tool_tip_text(&self) {
        self.btn_run_process.set_tool_tip_text(Some(
            &*("Generate the complete aligned stack for input into the ".to_string()
                + "tilt process."),
        ));
        self.btn_3dmod_full
            .set_tool_tip_text(Some("Open the complete aligned stack in 3dmod"));
    }
}

impl NewstackDisplay for NewstackOrBlendmontPanel {
    /// Java `final getParameters(NewstParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (NewstackOrBlendmontPanel.java:157-162).  Copy the newstack parameters
    /// from the GUI to the NewstParam object.
    fn get_parameters(
        &self,
        newst_param: &mut NewstParam,
        do_validation: bool,
    ) -> Result<bool, NewstackDisplayException> {
        self.newstack_and_blendmont_param_panel
            .get_parameters_newst_param_boolean(newst_param, do_validation)
    }

    /// Java `final setParameters(ConstNewstParam)` (NewstackOrBlendmontPanel.java:152-155).
    fn set_parameters(&self, newst_param: &dyn ConstNewstParam) {
        self.newstack_and_blendmont_param_panel
            .set_parameters_const_newst_param(newst_param);
    }

    /// Java `validate()` (NewstackOrBlendmontPanel.java:203-206).
    fn validate(&self) -> bool {
        true
    }

    /// Java `isFiducialess()` (NewstackOrBlendmontPanel.java:125-128).
    fn is_fiducialess(&self) -> bool {
        self.newstack_and_blendmont_param_panel.is_fiducialess()
    }
}

impl BlendmontDisplay for NewstackOrBlendmontPanel {
    /// Java `final getParameters(BlendmontParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (NewstackOrBlendmontPanel.java:169-174).  Copy the newstack parameters
    /// from the GUI to the NewstParam object.
    fn get_parameters(
        &self,
        param: &mut BlendmontParam,
        do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException> {
        self.newstack_and_blendmont_param_panel
            .get_parameters_blendmont_param_boolean(param, do_validation)
    }

    /// Java `final setParameters(BlendmontParam)` (NewstackOrBlendmontPanel.java:164-167).
    fn set_parameters(&self, param: &BlendmontParam) {
        self.newstack_and_blendmont_param_panel
            .set_parameters_blendmont_param(param);
    }

    /// Java `validate()` (NewstackOrBlendmontPanel.java:203-206).
    fn validate(&self) -> bool {
        true
    }

    /// Java `isFiducialess()` (NewstackOrBlendmontPanel.java:125-128).
    fn is_fiducialess(&self) -> bool {
        self.newstack_and_blendmont_param_panel.is_fiducialess()
    }
}

// Keep `io` referenced for the exception types' documentation: both display
// traits' exception enums carry `io::Error` for Java's `IOException`.
#[allow(unused_imports)]
use io as _;
