//! `IMOD/Etomo/src/etomo/ui/swing/TrialTiltPanel.java`.
//!
//! Java `final class TrialTiltPanel implements Expandable,
//! Run3dmodButtonContainer, TrialTiltDisplay`: the trial tomogram panel of
//! the tilt panels.  An EDT object: created as `Rc<Self>` by
//! [`TrialTiltPanel::get_instance`] (with `Rc::new_cyclic`, since the Java
//! hands `this` to the panel header, the 3dmod button and the action
//! listener while constructing its fields); every method takes `&self`.  The
//! inner listener class `TrialTiltActionListener` is a closure holding a weak
//! reference to the panel.  The parent (`TrialTiltParent`, the owning tilt
//! panel) is held weakly.
//!
//! **Trial tomogram list.**  Java keeps a reference to the `IntKeyList` inside
//! `MetaData` (`metaData.getTomoGenTrialTomogramNameList(axisID)`), so
//! `addTrialTomogramName` changes the metadata's list directly.  The Rust
//! `MetaData` hands out a copy, so the panel keeps the copy and
//! `getParameters(MetaData)` stores it back with
//! `setTomoGenTrialTomogramNameList`, which the Java also calls: the list the
//! metadata holds after a save is the same.

use crate::imod::etomo::base_manager::BaseManager;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::combo_box::ComboBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::process_display::ProcessDisplay;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::tilt_display::{TiltDisplay, TiltDisplayException};
use super::tooltip_formatter;
use super::trial_tilt_display::TrialTiltDisplay;
use super::trial_tilt_parent::TrialTiltParent;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::tilt_param::{self, TiltParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_int_key_list::ConstIntKeyList;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;

/// Java `final class TrialTiltPanel implements Expandable,
/// Run3dmodButtonContainer, TrialTiltDisplay`.
pub struct TrialTiltPanel {
    /// Java private final `actionListener` (`TrialTiltActionListener`).
    action_listener: ActionListener,
    /// Java private final `pnlRoot = new EtomoPanel()`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `pnlBody = SpacedPanel.getInstance()`.
    pnl_body: Rc<SpacedPanel>,
    /// Java private final `cmboTrialTomogramName`.
    cmbo_trial_tomogram_name: Rc<ComboBox>,
    /// Java private final `btnTrial`.
    btn_trial: Rc<MultiLineButton>,
    /// Java private final `btn3dmodTrial`.
    btn_3dmod_trial: Rc<Run3dmodButton>,
    /// Java private final `btnUseTrial`.
    btn_use_trial: Rc<MultiLineButton>,

    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `parent` (held weakly: the parent owns this panel).
    parent: Weak<dyn TrialTiltParent>,

    /// Java private `trialTomogramList = null`.  A way to know what items are
    /// currently in the trial tomogram combo box.  It is set from MetaData,
    /// which is assumed to be not null.
    trial_tomogram_list: RefCell<Option<IntKeyList>>,
}

impl TrialTiltPanel {
    /// Java private constructor `TrialTiltPanel(ApplicationManager, AxisID,
    /// DialogType, TrialTiltParent)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn TrialTiltParent>,
    ) -> Rc<TrialTiltPanel> {
        Rc::new_cyclic(|this: &Weak<TrialTiltPanel>| {
            // Field initializers.
            // TrialTiltActionListener
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            let pnl_root = EtomoPanel::new();
            let pnl_body = SpacedPanel::get_instance_void();
            let cmbo_trial_tomogram_name =
                ComboBox::get_editable_instance(Some("Trial tomogram filename: "));
            let btn_trial = MultiLineButton::new_string(Some("Generate Trial Tomogram"));
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_trial =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Trial in 3dmod"),
                    Some(container),
                );
            // Constructor body.
            let expandable: Weak<dyn Expandable> = this.clone();
            let header =
                PanelHeader::get_instance(Some("Trial Tilt"), Some(expandable), Some(dialog_type));
            let display_factory = manager.get_process_result_display_factory(axis_id);
            // Java casts `(MultiLineButton) displayFactory.getUseTrialTomogram()`;
            // the factory returns the concrete button.
            let btn_use_trial = display_factory.get_use_trial_tomogram();
            TrialTiltPanel {
                action_listener,
                pnl_root,
                pnl_body,
                cmbo_trial_tomogram_name,
                btn_trial,
                btn_3dmod_trial,
                btn_use_trial,
                header,
                manager,
                axis_id,
                dialog_type,
                parent,
                trial_tomogram_list: RefCell::new(None),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// TrialTiltParent)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn TrialTiltParent>,
    ) -> Rc<TrialTiltPanel> {
        let instance = TrialTiltPanel::new(manager, axis_id, dialog_type, parent);
        instance.create_panel();
        instance.update_display();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.  Layout the trial tomogram panel.
    fn create_panel(&self) {
        // Local panels
        let north_panel = SpacedPanel::get_instance_void();
        let button_panel = SpacedPanel::get_instance_void();
        // Root panel
        // Swing layout: pnlRoot BoxLayout Y_AXIS, untitled etched border.
        self.pnl_root.add(&self.header);
        self.pnl_root
            .get_component()
            .add(&self.pnl_body.get_container());
        // Body panel
        self.pnl_body.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_body.add_rigid_area_void();
        self.pnl_body.add_spaced_panel(&north_panel);
        self.pnl_body.add_spaced_panel(&button_panel);
        // North panel
        north_panel.set_box_layout(spaced_panel::X_AXIS);
        north_panel.add_component(&self.cmbo_trial_tomogram_name.get_component());
        // Button panel
        button_panel.set_box_layout(spaced_panel::X_AXIS);
        button_panel.add_multi_line_button(&self.btn_trial);
        button_panel.add_multi_line_button(&self.btn_3dmod_trial);
        button_panel.add_multi_line_button(&self.btn_use_trial);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_trial
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_trial
            .add_action_listener(self.action_listener.clone());
        self.btn_use_trial
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `msgTiltComSaved()`; empty.
    pub fn msg_tilt_com_saved(&self) {}

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.btn_use_trial
            .remove_action_listener(&self.action_listener);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.get_component().set_visible(visible);
    }

    /// Java `@Deprecated final setResume(boolean)`; empty.
    pub fn set_resume(&self, _resume: bool) {}

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.cmbo_trial_tomogram_name.set_enabled(true);
        self.btn_trial.set_enabled(true);
        self.btn_3dmod_trial.set_enabled(true);
        self.btn_use_trial.set_enabled(true);
    }

    /// Java `setTrialTomogramNameList(ConstIntKeyList)`.
    pub fn set_trial_tomogram_name_list(&self, input: &dyn ConstIntKeyList) {
        let mut walker = input.get_walker();
        while walker.has_next() {
            self.cmbo_trial_tomogram_name
                .add_item(walker.next_string().as_deref());
        }
    }

    /// Java `addToTrialTomogramName(String)`.
    pub fn add_to_trial_tomogram_name(&self, trial_tomogram_name: Option<&str>) {
        self.cmbo_trial_tomogram_name.add_item(trial_tomogram_name);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .get_state(Some(screen_state.get_tomo_gen_trial_tilt_header_state()));
    }

    /// Java `getParameters(MetaData) throws FortranInputSyntaxException`.
    ///
    /// Upstream bug fixed in translation (TrialTiltPanel.java:209): the Java
    /// passes `trialTomogramList` even when `setParameters(ConstMetaData)` was
    /// never called, storing null into the metadata (whose setter then throws
    /// a NullPointerException copying it).  Here the metadata is left as it
    /// is in that case.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        let trial_tomogram_list = self.trial_tomogram_list.borrow().clone();
        if let Some(trial_tomogram_list) = trial_tomogram_list {
            meta_data.set_tomo_gen_trial_tomogram_name_list(self.axis_id, trial_tomogram_list);
        }
        Ok(())
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        let trial_tomogram_list = meta_data.get_tomo_gen_trial_tomogram_name_list(self.axis_id);
        // `setTrialTomogramNameList(trialTomogramList)`: read from the copy
        // before storing it, so no borrow of the field is held across the call.
        self.set_trial_tomogram_name_list(&trial_tomogram_list);
        *self.trial_tomogram_list.borrow_mut() = Some(trial_tomogram_list);
    }

    /// Java final `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .set_state(Some(screen_state.get_tomo_gen_trial_tilt_header_state()));
        self.btn_use_trial.set_button_state(
            screen_state.get_button_state(self.btn_use_trial.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text for the
    /// axis panel objects.
    fn set_tool_tip_text(&self) {
        self.cmbo_trial_tomogram_name.set_tool_tip_text(
            tooltip_formatter::INSTANCE
                .format(Some(
                    "Current name of trial tomogram, which will be generated, viewed, or \
                     used by the buttons below.",
                ))
                .as_deref(),
        );
        self.btn_trial.set_tool_tip_text(Some(
            "Compute a trial tomogram with the current parameters, using the \
             filename in the \" Trial tomogram filename \" box.",
        ));
        self.btn_3dmod_trial.set_tool_tip_text(Some(
            "View the trial tomogram whose name is shown in \"Trial \
             tomogram filename\" box.",
        ));
        self.btn_use_trial.set_tool_tip_text(Some(
            "Rename the trial tomogram whose name is shown in the \"Trial \
             tomogram filename\" box to be the final tomogram.",
        ));
    }
}

impl ProcessDisplay for TrialTiltPanel {
    fn as_tilt_display(&self) -> Option<&dyn super::tilt_display::TiltDisplay> {
        Some(self)
    }
}

impl TiltDisplay for TrialTiltPanel {
    /// Java override `getParameters(TiltParam, boolean) throws
    /// NumberFormatException, InvalidParameterException, IOException`.
    fn get_parameters(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> Result<bool, TiltDisplayException> {
        tilt_param.set_command_mode(tilt_param::Mode::TrialTilt);
        // The parent owns this panel, so it outlives it.
        match self.parent.upgrade() {
            Some(parent) => parent.get_parameters_tilt_param_boolean(tilt_param, do_validation),
            None => Ok(false),
        }
    }

    /// Java override `getParameters(SplittiltParam, boolean)`.
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool {
        match self.parent.upgrade() {
            Some(parent) => parent.get_parameters_splittilt_param_boolean(param, do_validation),
            None => false,
        }
    }

    /// Java `@Deprecated allowTiltComSave()` (8/3/2018 See TiltDisplay).
    fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java override `setDebug(boolean)`; empty.
    fn set_debug(&self, _debug: bool) {}
}

impl TrialTiltDisplay for TrialTiltPanel {
    /// Java override `getTrialTomogramName()`.  Return the selected trial
    /// tomogram name.
    fn get_trial_tomogram_name(&self) -> Option<String> {
        let mut trial_tomogram_name = self.cmbo_trial_tomogram_name.get_selected_item();
        if trial_tomogram_name.is_none() {
            trial_tomogram_name = Some(String::new());
        }
        trial_tomogram_name
    }

    /// Java override `containsTrialTomogramName(String)`.
    ///
    /// Upstream bug fixed in translation (TrialTiltPanel.java:183): the Java
    /// dereferences `trialTomogramList`, which is null until
    /// `setParameters(ConstMetaData)` has run (NullPointerException).  Here a
    /// missing list contains nothing.
    fn contains_trial_tomogram_name(&self, trial_tomogram_name: Option<&str>) -> bool {
        self.trial_tomogram_list
            .borrow()
            .as_ref()
            .is_some_and(|list| list.contains_value(trial_tomogram_name))
    }

    /// Java override `addTrialTomogramName(String)`.
    ///
    /// Upstream bug fixed in translation (TrialTiltPanel.java:177): the Java
    /// dereferences `trialTomogramList`, which is null until
    /// `setParameters(ConstMetaData)` has run (NullPointerException before
    /// the combo box is updated).  Here the list is skipped when missing and
    /// the combo box still gets the name.
    fn add_trial_tomogram_name(&self, trial_tomogram_name: Option<&str>) {
        if let Some(list) = self.trial_tomogram_list.borrow_mut().as_mut() {
            list.add_string(trial_tomogram_name);
        }
        self.add_to_trial_tomogram_name(trial_tomogram_name);
    }
}

impl Expandable for TrialTiltPanel {
    /// Java override `expand(GlobalExpandButton)`; empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        }
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }
}

impl Run3dmodButtonContainer for TrialTiltPanel {
    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    /// Executes the action associated with command.  Deferred3dmodButton is
    /// null if it comes from the dialog's ActionListener.  Otherwise is comes
    /// from a Run3dmodButton which called action(Run3dmodButton,
    /// Run3dmoMenuOptions).  In that case it will be null unless it was set
    /// in the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_trial.get_action_command().as_deref() {
            // The parent owns this panel, so it outlives it.
            let Some(parent) = self.parent.upgrade() else {
                return;
            };
            let parent_method = parent.get_processing_method();
            // `manager.getProcessingMethodMediator(axisID)
            // .getRunMethodForProcessInterface(ProcessingMethod)`.
            let tilt_processing_method = self
                .manager
                .get_processing_method_mediator(Some(self.axis_id))
                .map(|mediator| mediator.get_run_method_for_process_interface(parent_method))
                .unwrap_or(parent_method);
            let trial: ProcessResultDisplayHandle = self.btn_trial.clone();
            self.manager.trial_action(
                Some(trial),
                None,
                Some(self),
                self.axis_id,
                self.dialog_type,
                tilt_processing_method,
            );
        } else if Some(command) == self.btn_use_trial.get_action_command().as_deref() {
            let use_trial: ProcessResultDisplayHandle = self.btn_use_trial.clone();
            self.manager
                .commit_test_volume_process_result_display_axis_id_trial_tilt_display(
                    Some(use_trial),
                    self.axis_id,
                    Some(self),
                );
        } else if Some(command) == self.btn_3dmod_trial.get_action_command().as_deref() {
            // Java passes `run3dmodMenuOptions` through (null from the action
            // listener); the staged manager takes the options by value, so a
            // null becomes the default (empty) options.  See NEEDS.
            self.manager
                .imod_test_volume_run3dmod_menu_options_axis_id_trial_tilt_display(
                    run_3dmod_menu_options.unwrap_or_default(),
                    self.axis_id,
                    Some(self),
                );
        }
    }
}
