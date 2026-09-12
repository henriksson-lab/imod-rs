//! `IMOD/Etomo/src/etomo/ui/swing/XfModelPanel.java`.
//!
//! Swing widgets, the process-result-display factory, and `ApplicationManager`
//! dispatch are explicit boundaries.  The translated unit keeps the Java
//! button ownership/linkage, ordered root panel, screen-state transfer, and
//! both command routes without creating an alternate process controller.
#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

use super::multi_line_button::MultiLineButton;
use super::tilt_panel::Deferred3dmodButton;

pub const VIEW_TRANSFORMED_MODEL_LABEL: &str = "View Transformed Model";
pub const TRANSFORM_FIDUCIAL_MODEL_LABEL: &str = "Transform Fiducial Model";

/// Java `ReconScreenState` call made by `XfModelPanel`.
pub trait XfModelPanelReconScreenState {
    fn get_button_state(&self, button_state_key: Option<&str>) -> bool;
}

/// Native in-memory form of the source unit's required screen-state button map.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct XfModelPanelScreenState {
    pub button_states: BTreeMap<String, bool>,
}

impl XfModelPanelReconScreenState for XfModelPanelScreenState {
    fn get_button_state(&self, button_state_key: Option<&str>) -> bool {
        button_state_key
            .and_then(|key| self.button_states.get(key))
            .copied()
            .unwrap_or(false)
    }
}

/// Direct `ApplicationManager` calls made by `XfModelPanel.java`.
pub trait XfModelPanelApplicationManager {
    fn xfmodel(
        &mut self,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
    );
    fn seed_erase_fiducial_model(
        &mut self,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
    );
}

/// Java `ProcessResultDisplayFactory.getXfModel` boundary.
pub trait XfModelPanelProcessResultDisplayFactory {
    fn get_xf_model(&self) -> MultiLineButton;
}

impl XfModelPanelProcessResultDisplayFactory
    for super::process_result_display_factory::ProcessResultDisplayFactory
{
    fn get_xf_model(&self) -> MultiLineButton {
        super::process_result_display_factory::ProcessResultDisplayFactory::get_xf_model(self)
            .clone()
    }
}

/// Source-visible `SpacedPanel pnlRoot` construction state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct XfModelPanelLayout {
    pub box_layout_x_axis: bool,
    pub component_order: Vec<String>,
    pub visible: bool,
}

/// Java final `XfModelPanel`.
#[derive(Clone, Debug, PartialEq)]
pub struct XfModelPanel {
    pub pnl_root: XfModelPanelLayout,
    pub btn_3dmod_xf_model: MultiLineButton,
    pub btn_xf_model: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    /// Java `btnXfModel.setContainer(this)` relationship.
    pub xf_model_container_set: bool,
    /// Java `btnXfModel.setDeferred3dmodButton(btn3dmodXfModel)` relationship.
    pub xf_model_deferred_3dmod_button_set: bool,
    /// Java private listener registration on the two buttons.
    pub action_listener_registered: bool,
}

impl XfModelPanel {
    /// Java private `XfModelPanel(ApplicationManager, AxisID, DialogType)`.
    pub fn new(
        mut btn_xf_model: MultiLineButton,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        if btn_xf_model.get_action_command().is_none() {
            btn_xf_model.set_action_command(Some(TRANSFORM_FIDUCIAL_MODEL_LABEL));
        }
        let mut btn_3dmod_xf_model =
            MultiLineButton::new_with_label(Some(VIEW_TRANSFORMED_MODEL_LABEL));
        btn_3dmod_xf_model.set_action_command(Some(VIEW_TRANSFORMED_MODEL_LABEL));
        Self {
            pnl_root: XfModelPanelLayout {
                visible: true,
                ..Default::default()
            },
            btn_3dmod_xf_model,
            btn_xf_model,
            axis_id,
            dialog_type,
            xf_model_container_set: false,
            xf_model_deferred_3dmod_button_set: false,
            action_listener_registered: false,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance<F: XfModelPanelProcessResultDisplayFactory>(
        factory: &F,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        let mut instance = Self::new(factory.get_xf_model(), axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_xf_model.add_action_listener();
        self.btn_3dmod_xf_model.add_action_listener();
        self.action_listener_registered = true;
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.xf_model_container_set = true;
        self.xf_model_deferred_3dmod_button_set = true;
        self.pnl_root.box_layout_x_axis = true;
        self.pnl_root.component_order = vec!["btnXfModel".into(), "btn3dmodXfModel".into()];
    }

    /// Java `getComponent`; concrete Swing component realization is a GUI boundary.
    pub fn get_component(&self) -> &XfModelPanelLayout {
        &self.pnl_root
    }

    /// Java `setVisible`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.visible = visible;
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_xf_model.remove_action_listener();
        self.action_listener_registered = false;
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters<S: XfModelPanelReconScreenState>(&mut self, screen_state: &S) {
        let button_state_key = self.btn_xf_model.get_button_state_key();
        self.btn_xf_model
            .set_button_state(screen_state.get_button_state(button_state_key.as_deref()));
    }

    /// Java `Run3dmodButtonContainer.action`.
    pub fn action<M: XfModelPanelApplicationManager>(
        &mut self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if self.btn_xf_model.get_action_command() == Some(command) {
            manager.xfmodel(
                &self.btn_xf_model,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                self.axis_id,
                self.dialog_type,
            );
        } else if self.btn_3dmod_xf_model.get_action_command() == Some(command) {
            manager.seed_erase_fiducial_model(
                run_3dmod_menu_options,
                self.axis_id,
                self.dialog_type,
            );
        }
    }

    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.btn_xf_model.set_tool_tip_text(Some(
            "Transform .fid mode built on prealigned stack to _erase.fid model that fits the aligned stack.",
        ));
        self.btn_3dmod_xf_model
            .set_tool_tip_text(Some("View the _erase.fid model on the aligned stack."));
    }

    /// Java inner `XfModelPanelActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed<M: XfModelPanelApplicationManager>(
        &mut self,
        command: &str,
        manager: &mut M,
    ) {
        self.action(command, None, None, manager);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Factory;
    impl XfModelPanelProcessResultDisplayFactory for Factory {
        fn get_xf_model(&self) -> MultiLineButton {
            MultiLineButton::new_full(
                Some(TRANSFORM_FIDUCIAL_MODEL_LABEL),
                false,
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
        xfmodel: Option<(AxisID, DialogType)>,
        seed_erase_fiducial_model: Option<(AxisID, DialogType)>,
    }
    impl XfModelPanelApplicationManager for Manager {
        fn xfmodel(
            &mut self,
            _button: &MultiLineButton,
            _deferred_3dmod_button: Option<&Deferred3dmodButton>,
            _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
            axis_id: AxisID,
            dialog_type: DialogType,
        ) {
            self.xfmodel = Some((axis_id, dialog_type));
        }
        fn seed_erase_fiducial_model(
            &mut self,
            _run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
            axis_id: AxisID,
            dialog_type: DialogType,
        ) {
            self.seed_erase_fiducial_model = Some((axis_id, dialog_type));
        }
    }

    #[test]
    fn instance_preserves_source_button_linkage_layout_and_tooltips() {
        let panel =
            XfModelPanel::get_instance(&Factory, AxisID::First, DialogType::FinalAlignedStack);
        assert!(panel.pnl_root.box_layout_x_axis);
        assert_eq!(
            panel.pnl_root.component_order,
            ["btnXfModel", "btn3dmodXfModel"]
        );
        assert!(panel.xf_model_container_set);
        assert!(panel.xf_model_deferred_3dmod_button_set);
        assert_eq!(panel.btn_xf_model.button.action_listener_count, 1);
        assert_eq!(panel.btn_3dmod_xf_model.button.action_listener_count, 1);
        assert_eq!(
            panel.btn_xf_model.button.tooltip.as_deref(),
            Some(
                "Transform .fid mode built on prealigned stack to _erase.fid model that fits the aligned stack."
            )
        );
    }

    #[test]
    fn commands_route_to_the_two_source_manager_calls() {
        let mut panel =
            XfModelPanel::get_instance(&Factory, AxisID::Second, DialogType::FinalAlignedStack);
        let mut manager = Manager::default();
        let xfmodel = panel.btn_xf_model.get_action_command().unwrap().to_owned();
        panel.action(&xfmodel, None, None, &mut manager);
        let viewer = panel
            .btn_3dmod_xf_model
            .get_action_command()
            .unwrap()
            .to_owned();
        panel.action(&viewer, None, None, &mut manager);
        assert_eq!(
            manager.xfmodel,
            Some((AxisID::Second, DialogType::FinalAlignedStack))
        );
        assert_eq!(
            manager.seed_erase_fiducial_model,
            Some((AxisID::Second, DialogType::FinalAlignedStack))
        );
    }

    #[test]
    fn screen_state_visibility_and_done_follow_the_source() {
        let mut panel =
            XfModelPanel::get_instance(&Factory, AxisID::Only, DialogType::FinalAlignedStack);
        let mut state = XfModelPanelScreenState::default();
        let key = panel.btn_xf_model.get_button_state_key().unwrap();
        state.button_states.insert(key, true);
        panel.set_parameters(&state);
        panel.set_visible(false);
        panel.done();
        assert!(panel.btn_xf_model.button.selected);
        assert!(!panel.pnl_root.visible);
        assert_eq!(panel.btn_xf_model.button.action_listener_count, 0);
        assert_eq!(panel.btn_3dmod_xf_model.button.action_listener_count, 1);
    }
}
