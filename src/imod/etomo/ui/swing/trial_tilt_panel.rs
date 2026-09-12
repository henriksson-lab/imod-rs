//! `IMOD/Etomo/src/etomo/ui/swing/TrialTiltPanel.java`.
//!
//! The Swing widgets, `ApplicationManager`, process-result factory, and
//! `UIHarness` pack call are kept as direct boundaries.  The source unit's
//! panel state, list handling, parameter delegation, and three action routes
//! are retained here.
#![allow(dead_code)]

use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{axis_id::AxisID, dialog_type::DialogType, processing_method::ProcessingMethod},
};

use super::{
    beads3d_find_panel::Deferred3dmodButton,
    multi_line_button::MultiLineButton,
    panel_header::{ExpandButton, Expandable, PanelHeader, PanelHeaderState},
};

pub const TRIAL_TILT_LABEL: &str = "Trial Tilt";
pub const GENERATE_TRIAL_TOMOGRAM_LABEL: &str = "Generate Trial Tomogram";
pub const VIEW_TRIAL_IN_3DMOD_LABEL: &str = "View Trial in 3dmod";

/// Java editable `ComboBox` at this unit's Swing boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrialTomogramNameComboBox {
    pub label: String,
    pub items: Vec<String>,
    pub selected_item: Option<String>,
    pub enabled: bool,
    pub tooltip: Option<String>,
}

/// Java `IntKeyList` dependency as used by this class.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IntKeyList {
    pub values: Vec<String>,
}

impl IntKeyList {
    /// Java `add(String)`.
    pub fn add(&mut self, value: impl Into<String>) {
        self.values.push(value.into());
    }
    /// Java `containsValue(String)`.
    pub fn contains_value(&self, value: &str) -> bool {
        self.values.iter().any(|item| item == value)
    }
}

/// Source-facing `TiltParam.Mode` value written by this panel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TiltParamMode {
    #[default]
    Default,
    TrialTilt,
}

/// Java `TiltParam` call surface used by `TrialTiltPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrialTiltParam {
    pub command_mode: TiltParamMode,
}

/// Java `SplittiltParam` direct boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SplittiltParam;

/// Java `ReconScreenState` fields accessed by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrialTiltScreenState {
    pub tomo_gen_trial_tilt_header_state: PanelHeaderState,
    pub button_states: std::collections::BTreeMap<String, bool>,
}

/// Java `ConstMetaData`/`MetaData` trial-name-list access.
pub trait TrialTiltMetaData {
    fn get_tomo_gen_trial_tomogram_name_list(&self, axis_id: AxisID) -> IntKeyList;
    fn set_tomo_gen_trial_tomogram_name_list(&mut self, axis_id: AxisID, list: IntKeyList);
}

/// Java `TrialTiltParent` calls made by this class.
pub trait TrialTiltParent {
    fn get_parameters_tilt(&self, tilt_param: &mut TrialTiltParam, do_validation: bool) -> bool;
    fn get_parameters_splittilt(&self, param: &mut SplittiltParam, do_validation: bool) -> bool;
    fn get_processing_method(&self) -> ProcessingMethod;
}

/// Direct `ApplicationManager` / `ProcessingMethodMediator` calls.
pub trait TrialTiltPanelApplicationManager {
    fn get_use_trial_tomogram(&self, axis_id: AxisID) -> MultiLineButton;
    fn get_run_method_for_process_interface(
        &self,
        axis_id: AxisID,
        method: ProcessingMethod,
    ) -> ProcessingMethod;
    fn trial_action(
        &mut self,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
    );
    fn commit_test_volume(&mut self, button: &MultiLineButton, axis_id: AxisID);
    fn imod_test_volume(&mut self, options: Option<Run3dmodMenuOptions>, axis_id: AxisID);
    fn pack(&mut self, axis_id: AxisID);
}

/// Java `EtomoPanel pnlRoot`, `SpacedPanel pnlBody`, and local layout order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrialTiltPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_etched_border: bool,
    pub root_visible: bool,
    pub body_box_layout_y_axis: bool,
    pub body_visible: bool,
    pub root_component_order: Vec<String>,
    pub body_component_order: Vec<String>,
    pub north_component_order: Vec<String>,
    pub button_component_order: Vec<String>,
}

/// Java final `TrialTiltPanel`.
pub struct TrialTiltPanel<P: TrialTiltParent> {
    pub action_listener_registered: bool,
    pub pnl_root: TrialTiltPanelLayout,
    pub cmbo_trial_tomogram_name: TrialTomogramNameComboBox,
    pub btn_trial: MultiLineButton,
    pub btn_3dmod_trial: MultiLineButton,
    pub btn_use_trial: MultiLineButton,
    pub header: PanelHeader,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub parent: P,
    pub trial_tomogram_list: Option<IntKeyList>,
}

impl<P: TrialTiltParent> TrialTiltPanel<P> {
    /// Java private constructor, with the manager factory boundary supplied by
    /// `get_instance`.
    pub fn new(
        parent: P,
        axis_id: AxisID,
        dialog_type: DialogType,
        btn_use_trial: MultiLineButton,
    ) -> Self {
        let mut btn_trial = MultiLineButton::new_with_label(Some(GENERATE_TRIAL_TOMOGRAM_LABEL));
        btn_trial.set_action_command(Some(GENERATE_TRIAL_TOMOGRAM_LABEL));
        let mut btn_3dmod_trial = MultiLineButton::new_with_label(Some(VIEW_TRIAL_IN_3DMOD_LABEL));
        btn_3dmod_trial.set_action_command(Some(VIEW_TRIAL_IN_3DMOD_LABEL));
        Self {
            action_listener_registered: false,
            pnl_root: TrialTiltPanelLayout::default(),
            cmbo_trial_tomogram_name: TrialTomogramNameComboBox {
                label: "Trial tomogram filename: ".into(),
                enabled: true,
                ..Default::default()
            },
            btn_trial,
            btn_3dmod_trial,
            btn_use_trial,
            // Java calls `PanelHeader.getInstance`; its Rust constructor keeps
            // exactly the resulting `openClose=true` construction choices
            // without manufacturing a second owner solely for borrowing.
            header: PanelHeader::new(
                TRIAL_TILT_LABEL,
                false,
                false,
                dialog_type,
                true,
                false,
                true,
                false,
                true,
            ),
            axis_id,
            dialog_type,
            parent,
            trial_tomogram_list: None,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance<M: TrialTiltPanelApplicationManager>(
        manager: &M,
        parent: P,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        let mut instance = Self::new(
            parent,
            axis_id,
            dialog_type,
            manager.get_use_trial_tomogram(axis_id),
        );
        instance.create_panel();
        instance.update_display();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java `setDebug`, intentionally empty.
    pub fn set_debug(&mut self, _debug: bool) {}

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_etched_border = true;
        self.pnl_root.root_component_order = vec!["header".into(), "pnlBody".into()];
        self.pnl_root.body_box_layout_y_axis = true;
        self.pnl_root.body_component_order = vec![
            "rigidArea".into(),
            "northPanel".into(),
            "buttonPanel".into(),
        ];
        self.pnl_root.north_component_order = vec!["cmboTrialTomogramName".into()];
        self.pnl_root.button_component_order = vec![
            "btnTrial".into(),
            "btn3dmodTrial".into(),
            "btnUseTrial".into(),
        ];
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_trial.add_action_listener();
        self.btn_3dmod_trial.add_action_listener();
        self.btn_use_trial.add_action_listener();
        self.action_listener_registered = true;
    }
    /// Java deprecated `allowTiltComSave`.
    pub fn allow_tilt_com_save(&self) -> bool {
        true
    }
    /// Java `msgTiltComSaved`, intentionally empty.
    pub fn msg_tilt_com_saved(&mut self) {}
    /// Java `getComponent` boundary.
    pub fn get_component(&self) -> &TrialTiltPanelLayout {
        &self.pnl_root
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_use_trial.remove_action_listener();
    }
    /// Java `setVisible`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.root_visible = visible;
    }
    /// Java deprecated `setResume`, intentionally empty.
    pub fn set_resume(&mut self, _resume: bool) {}
    /// Java private `updateDisplay`.
    pub fn update_display(&mut self) {
        self.cmbo_trial_tomogram_name.enabled = true;
        self.btn_trial.set_enabled(true);
        self.btn_3dmod_trial.set_enabled(true);
        self.btn_use_trial.set_enabled(true);
    }
    /// Java `setTrialTomogramNameList`.
    pub fn set_trial_tomogram_name_list(&mut self, input: &IntKeyList) {
        for item in &input.values {
            self.cmbo_trial_tomogram_name.items.push(item.clone());
        }
    }
    /// Java `addToTrialTomogramName`.
    pub fn add_to_trial_tomogram_name(&mut self, name: impl Into<String>) {
        self.cmbo_trial_tomogram_name.items.push(name.into());
    }
    /// Java `addTrialTomogramName`.
    pub fn add_trial_tomogram_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.trial_tomogram_list
            .as_mut()
            .expect("MetaData must be set before addTrialTomogramName")
            .add(name.clone());
        self.add_to_trial_tomogram_name(name);
    }
    /// Java `containsTrialTomogramName`.
    pub fn contains_trial_tomogram_name(&self, name: &str) -> bool {
        self.trial_tomogram_list
            .as_ref()
            .expect("MetaData must be set before containsTrialTomogramName")
            .contains_value(name)
    }
    /// Java `getTrialTomogramName`.
    pub fn get_trial_tomogram_name(&self) -> String {
        self.cmbo_trial_tomogram_name
            .selected_item
            .clone()
            .unwrap_or_default()
    }
    /// Java `getParameters(TiltParam, boolean)`.
    pub fn get_parameters_tilt(&self, param: &mut TrialTiltParam, do_validation: bool) -> bool {
        param.command_mode = TiltParamMode::TrialTilt;
        self.parent.get_parameters_tilt(param, do_validation)
    }
    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_screen_state(&self, screen_state: &mut TrialTiltScreenState) {
        self.header
            .get_state(Some(&mut screen_state.tomo_gen_trial_tilt_header_state));
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data<M: TrialTiltMetaData>(&self, meta_data: &mut M) {
        meta_data.set_tomo_gen_trial_tomogram_name_list(
            self.axis_id,
            self.trial_tomogram_list
                .clone()
                .expect("MetaData must be set before getParameters"),
        );
    }
    /// Java `getParameters(SplittiltParam, boolean)`.
    pub fn get_parameters_splittilt(
        &self,
        param: &mut SplittiltParam,
        do_validation: bool,
    ) -> bool {
        self.parent.get_parameters_splittilt(param, do_validation)
    }
    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_meta_data<M: TrialTiltMetaData>(&mut self, meta_data: &M) {
        self.trial_tomogram_list =
            Some(meta_data.get_tomo_gen_trial_tomogram_name_list(self.axis_id));
        let list = self.trial_tomogram_list.clone().expect("assigned above");
        self.set_trial_tomogram_name_list(&list);
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_screen_state(&mut self, screen_state: &TrialTiltScreenState) {
        self.header
            .set_state(Some(&screen_state.tomo_gen_trial_tilt_header_state));
        let key = self
            .btn_use_trial
            .get_button_state_key()
            .unwrap_or_default();
        self.btn_use_trial.set_button_state(
            screen_state
                .button_states
                .get(&key)
                .copied()
                .unwrap_or(false),
        );
    }
    /// Java `expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global(&mut self) {}
    /// Java `expand(ExpandButton)`.
    pub fn expand<M: TrialTiltPanelApplicationManager>(
        &mut self,
        button: &ExpandButton,
        manager: &mut M,
    ) {
        if self.header.equals_open_close(button) {
            self.pnl_root.body_visible = button.is_expanded();
        }
        manager.pack(self.axis_id);
    }
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: TrialTiltPanelApplicationManager>(
        &mut self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if self.btn_trial.get_action_command() == Some(command) {
            let method = manager.get_run_method_for_process_interface(
                self.axis_id,
                self.parent.get_processing_method(),
            );
            manager.trial_action(
                &self.btn_trial,
                deferred_3dmod_button,
                self.axis_id,
                self.dialog_type,
                method,
            );
        } else if self.btn_use_trial.get_action_command() == Some(command) {
            manager.commit_test_volume(&self.btn_use_trial, self.axis_id);
        } else if self.btn_3dmod_trial.get_action_command() == Some(command) {
            manager.imod_test_volume(options, self.axis_id);
        }
    }
    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.cmbo_trial_tomogram_name.tooltip = Some("Current name of trial tomogram, which will be generated, viewed, or used by the buttons below.".into());
        self.btn_trial.set_tool_tip_text(Some("Compute a trial tomogram with the current parameters, using the filename in the \" Trial tomogram filename \" box."));
        self.btn_3dmod_trial.set_tool_tip_text(Some(
            "View the trial tomogram whose name is shown in \"Trial tomogram filename\" box.",
        ));
        self.btn_use_trial.set_tool_tip_text(Some("Rename the trial tomogram whose name is shown in the \"Trial tomogram filename\" box to be the final tomogram."));
    }
    /// Java inner `TrialTiltActionListener.actionPerformed`.
    pub fn action_performed<M: TrialTiltPanelApplicationManager>(
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
    #[derive(Default)]
    struct Parent;
    impl TrialTiltParent for Parent {
        fn get_parameters_tilt(&self, _: &mut TrialTiltParam, _: bool) -> bool {
            true
        }
        fn get_parameters_splittilt(&self, _: &mut SplittiltParam, _: bool) -> bool {
            true
        }
        fn get_processing_method(&self) -> ProcessingMethod {
            ProcessingMethod::Queue
        }
    }
    #[derive(Default)]
    struct Manager {
        calls: Vec<String>,
    }
    impl TrialTiltPanelApplicationManager for Manager {
        fn get_use_trial_tomogram(&self, _: AxisID) -> MultiLineButton {
            let mut b = MultiLineButton::new_with_label(Some("Use Trial"));
            b.set_action_command(Some("Use Trial"));
            b
        }
        fn get_run_method_for_process_interface(
            &self,
            _: AxisID,
            x: ProcessingMethod,
        ) -> ProcessingMethod {
            x
        }
        fn trial_action(
            &mut self,
            _: &MultiLineButton,
            _: Option<&Deferred3dmodButton>,
            _: AxisID,
            _: DialogType,
            _: ProcessingMethod,
        ) {
            self.calls.push("trial".into());
        }
        fn commit_test_volume(&mut self, _: &MultiLineButton, _: AxisID) {
            self.calls.push("use".into());
        }
        fn imod_test_volume(&mut self, _: Option<Run3dmodMenuOptions>, _: AxisID) {
            self.calls.push("view".into());
        }
        fn pack(&mut self, _: AxisID) {
            self.calls.push("pack".into());
        }
    }
    #[derive(Default)]
    struct Meta(IntKeyList);
    impl TrialTiltMetaData for Meta {
        fn get_tomo_gen_trial_tomogram_name_list(&self, _: AxisID) -> IntKeyList {
            self.0.clone()
        }
        fn set_tomo_gen_trial_tomogram_name_list(&mut self, _: AxisID, x: IntKeyList) {
            self.0 = x;
        }
    }
    #[test]
    fn list_parameters_and_actions_follow_source() {
        let manager = Manager::default();
        let mut p = TrialTiltPanel::get_instance(
            &manager,
            Parent,
            AxisID::Only,
            DialogType::TomogramGeneration,
        );
        p.set_parameters_meta_data(&Meta(IntKeyList {
            values: vec!["trial.rec".into()],
        }));
        p.add_trial_tomogram_name("other.rec");
        assert!(p.contains_trial_tomogram_name("other.rec"));
        assert_eq!(p.cmbo_trial_tomogram_name.items.len(), 2);
        let mut param = TrialTiltParam::default();
        assert!(p.get_parameters_tilt(&mut param, true));
        assert_eq!(param.command_mode, TiltParamMode::TrialTilt);
        let mut manager = manager;
        p.action(GENERATE_TRIAL_TOMOGRAM_LABEL, None, None, &mut manager);
        p.action("Use Trial", None, None, &mut manager);
        p.action(VIEW_TRIAL_IN_3DMOD_LABEL, None, None, &mut manager);
        assert_eq!(manager.calls, ["trial", "use", "view"]);
    }
    #[test]
    fn expansion_and_screen_state_follow_source() {
        let mut manager = Manager::default();
        let mut p = TrialTiltPanel::get_instance(
            &manager,
            Parent,
            AxisID::First,
            DialogType::TomogramGeneration,
        );
        let button = p.header.btn_open_close.clone().unwrap();
        p.expand(&button, &mut manager);
        assert!(p.pnl_root.body_visible);
        assert_eq!(manager.calls, ["pack"]);
        let mut screen = TrialTiltScreenState::default();
        p.get_parameters_screen_state(&mut screen);
        p.set_parameters_screen_state(&screen);
        p.done();
        assert_eq!(p.btn_use_trial.button.action_listener_count, 0);
    }
}
