//! `IMOD/Etomo/src/etomo/ui/swing/ReprojectModelPanel.java`.
//!
//! Swing construction, the process-display factory, and `ApplicationManager`
//! dispatch are explicit boundaries.  This module retains the source panel's
//! button linkage, screen-state transfer, `TiltDisplay` parameter answers, and
//! two-command dispatch.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

use super::beads3d_find_panel::{Beads3dFindScreenState, Deferred3dmodButton};
use super::multi_line_button::MultiLineButton;

pub const REPROJECT_MODEL_LABEL: &str = "Reproject Model";
pub const VIEW_2D_MODEL_ON_ALIGNED_STACK_LABEL: &str = "View 2D Model on Aligned Stack";

/// Java `SplittiltParam` dependency of the `TiltDisplay` overload.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SplittiltParam;

/// Java `TiltParam` dependency of the `TiltDisplay` overload.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltParam;

/// Java `ReconScreenState` calls made by this source unit.
pub trait ReprojectModelPanelReconScreenState {
    fn get_button_state(&self, button_state_key: &str) -> bool;
}

/// Native in-memory form of the source unit's required `ReconScreenState`
/// button state.  The actual screen-state owner remains a separate boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReprojectModelPanelScreenState {
    pub button_states: BTreeMap<String, bool>,
}

impl ReprojectModelPanelReconScreenState for ReprojectModelPanelScreenState {
    fn get_button_state(&self, button_state_key: &str) -> bool {
        self.button_states
            .get(button_state_key)
            .copied()
            .unwrap_or(false)
    }
}

/// Direct `ApplicationManager` calls made by `ReprojectModelPanel.java`.
pub trait ReprojectModelPanelApplicationManager {
    fn reproject_model_action(
        &mut self,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
    );
    fn imod_reproject_model(
        &mut self,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// Java `ProcessResultDisplayFactory.getReprojectModel` boundary.
pub trait ReprojectModelPanelProcessResultDisplayFactory {
    fn get_reproject_model(&self) -> MultiLineButton;
}

/// Source-visible `SpacedPanel pnlRoot` construction state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReprojectModelPanelLayout {
    pub box_layout_x_axis: bool,
    pub component_order: Vec<String>,
}

/// Java final `ReprojectModelPanel`.
pub struct ReprojectModelPanel {
    pub pnl_root: ReprojectModelPanelLayout,
    pub btn_3dmod_reproject_model: MultiLineButton,
    pub btn_reproject_model: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    /// Java `btnReprojectModel.setContainer(this)` relationship.
    pub reproject_model_container_set: bool,
    /// Java `setDeferred3dmodButton(btn3dmodReprojectModel)` relationship.
    pub reproject_model_deferred_3dmod_button_set: bool,
    /// The private Java listener is represented by its registrations on both
    /// buttons; the GUI event system remains the presentation boundary.
    pub action_listener_registered: bool,
}

impl ReprojectModelPanel {
    /// Java private `ReprojectModelPanel(ApplicationManager, AxisID, DialogType)`.
    pub fn new(
        mut btn_reproject_model: MultiLineButton,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        // Swing's default action command is its displayed text.  The generic
        // button boundary has no AWT defaulting event, so retain the source
        // command on the factory-owned button when it has not supplied one.
        if btn_reproject_model.get_action_command().is_none() {
            btn_reproject_model.set_action_command(Some(REPROJECT_MODEL_LABEL));
        }
        let mut btn_3dmod_reproject_model =
            MultiLineButton::new_with_label(Some(VIEW_2D_MODEL_ON_ALIGNED_STACK_LABEL));
        btn_3dmod_reproject_model.set_action_command(Some(VIEW_2D_MODEL_ON_ALIGNED_STACK_LABEL));
        Self {
            pnl_root: ReprojectModelPanelLayout::default(),
            btn_3dmod_reproject_model,
            btn_reproject_model,
            axis_id,
            dialog_type,
            reproject_model_container_set: false,
            reproject_model_deferred_3dmod_button_set: false,
            action_listener_registered: false,
        }
    }

    /// Java static `getInstance`; factory selection is kept at its source
    /// boundary so this unit receives the exact shared process-result button.
    pub fn get_instance<F: ReprojectModelPanelProcessResultDisplayFactory>(
        factory: &F,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        let mut instance = Self::new(factory.get_reproject_model(), axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java `setDebug(final boolean)`, intentionally empty.
    pub fn set_debug(&mut self, _debug: bool) {}

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_reproject_model.add_action_listener();
        self.btn_3dmod_reproject_model.add_action_listener();
        self.action_listener_registered = true;
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.reproject_model_container_set = true;
        self.reproject_model_deferred_3dmod_button_set = true;
        self.pnl_root.box_layout_x_axis = true;
        self.pnl_root.component_order =
            vec!["btnReprojectModel".into(), "btn3dmodReprojectModel".into()];
    }

    /// Java `getComponent`; concrete widget realization is a GUI boundary.
    pub fn get_component(&self) -> &ReprojectModelPanelLayout {
        &self.pnl_root
    }

    /// Java deprecated `allowTiltComSave`.
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }

    /// Java `msgTiltComSaved`, intentionally empty.
    pub fn msg_tilt_com_saved(&mut self) {}

    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_reproject_model.remove_action_listener();
        self.action_listener_registered = false;
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters<S: ReprojectModelPanelReconScreenState>(&mut self, screen_state: &S) {
        let button_state_key = self
            .btn_reproject_model
            .get_button_state_key()
            .unwrap_or_default();
        self.btn_reproject_model
            .set_button_state(screen_state.get_button_state(&button_state_key));
    }

    /// Java `getParameters(SplittiltParam, boolean)`.
    pub fn get_parameters_splittilt(
        &self,
        _param: &mut SplittiltParam,
        _do_validation: bool,
    ) -> bool {
        false
    }

    /// Java `getParameters(TiltParam, boolean)`.
    pub fn get_parameters_tilt(&self, _param: &mut TiltParam, _do_validation: bool) -> bool {
        true
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: ReprojectModelPanelApplicationManager>(
        &mut self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if self.btn_reproject_model.get_action_command() == Some(command) {
            manager.reproject_model_action(
                &self.btn_reproject_model,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                self.axis_id,
                self.dialog_type,
            );
        } else if self.btn_3dmod_reproject_model.get_action_command() == Some(command) {
            manager.imod_reproject_model(self.axis_id, run_3dmod_menu_options);
        }
    }

    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.btn_reproject_model
            .set_tool_tip_text(Some("Run tilt to reproject the model."));
        self.btn_3dmod_reproject_model
            .set_tool_tip_text(Some("View model of gold particles."));
    }

    /// Java inner `ReprojectModelPanelActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed<M: ReprojectModelPanelApplicationManager>(
        &mut self,
        command: &str,
        manager: &mut M,
    ) {
        self.action(command, None, None, manager);
    }
}

impl super::beads3d_find_panel::ReprojectModelPanel for ReprojectModelPanel {
    fn done(&mut self) {
        Self::done(self);
    }

    /// `Beads3dFindPanel` currently exposes only a parameter-set count rather
    /// than the Java `ReconScreenState` button map.  Its existing boundary
    /// cannot supply this source method's required key/value lookup.
    fn set_parameters_screen_state(&mut self, _screen_state: &Beads3dFindScreenState) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Factory;
    impl ReprojectModelPanelProcessResultDisplayFactory for Factory {
        fn get_reproject_model(&self) -> MultiLineButton {
            MultiLineButton::new_full(
                Some(REPROJECT_MODEL_LABEL),
                true,
                Some(DialogType::FinalAlignedStack),
                false,
                false,
                false,
                None,
            )
        }
    }

    #[derive(Default)]
    struct Manager {
        reproject: Option<(AxisID, DialogType)>,
        imod: Option<AxisID>,
    }
    impl ReprojectModelPanelApplicationManager for Manager {
        fn reproject_model_action(
            &mut self,
            _button: &MultiLineButton,
            _deferred_3dmod_button: Option<&Deferred3dmodButton>,
            _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
            axis_id: AxisID,
            dialog_type: DialogType,
        ) {
            self.reproject = Some((axis_id, dialog_type));
        }
        fn imod_reproject_model(
            &mut self,
            axis_id: AxisID,
            _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        ) {
            self.imod = Some(axis_id);
        }
    }

    #[test]
    fn instance_creates_the_source_button_linkage_and_layout() {
        let panel = ReprojectModelPanel::get_instance(
            &Factory,
            AxisID::First,
            DialogType::FinalAlignedStack,
        );
        assert!(panel.pnl_root.box_layout_x_axis);
        assert_eq!(
            panel.pnl_root.component_order,
            ["btnReprojectModel", "btn3dmodReprojectModel"]
        );
        assert!(panel.reproject_model_container_set);
        assert!(panel.reproject_model_deferred_3dmod_button_set);
        assert_eq!(panel.btn_reproject_model.button.action_listener_count, 1);
        assert_eq!(
            panel.btn_3dmod_reproject_model.button.action_listener_count,
            1
        );
    }

    #[test]
    fn action_routes_each_source_command_to_its_manager_call() {
        let mut panel = ReprojectModelPanel::get_instance(
            &Factory,
            AxisID::Second,
            DialogType::FinalAlignedStack,
        );
        let mut manager = Manager::default();
        let reproject = panel
            .btn_reproject_model
            .get_action_command()
            .unwrap()
            .to_owned();
        panel.action(&reproject, None, None, &mut manager);
        assert_eq!(
            manager.reproject,
            Some((AxisID::Second, DialogType::FinalAlignedStack))
        );
        let imod = panel
            .btn_3dmod_reproject_model
            .get_action_command()
            .unwrap()
            .to_owned();
        panel.action(&imod, None, None, &mut manager);
        assert_eq!(manager.imod, Some(AxisID::Second));
    }

    #[test]
    fn screen_state_and_tilt_parameter_answers_match_source() {
        let mut panel = ReprojectModelPanel::get_instance(
            &Factory,
            AxisID::Only,
            DialogType::FinalAlignedStack,
        );
        let key = panel.btn_reproject_model.get_button_state_key().unwrap();
        let mut state = ReprojectModelPanelScreenState::default();
        state.button_states.insert(key, true);
        panel.set_parameters(&state);
        assert!(panel.btn_reproject_model.get_button_state());
        assert!(!panel.get_parameters_splittilt(&mut SplittiltParam, true));
        assert!(panel.get_parameters_tilt(&mut TiltParam, true));
        panel.done();
        assert_eq!(panel.btn_reproject_model.button.action_listener_count, 0);
        assert_eq!(
            panel.btn_3dmod_reproject_model.button.action_listener_count,
            1
        );
    }
}
