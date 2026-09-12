//! `IMOD/Etomo/src/etomo/ui/swing/FindBeads3dPanel.java`.
//!
//! Swing component construction, autodoc access, and `ApplicationManager`
//! process dispatch remain direct GUI/application boundaries.  The panel's
//! source-owned field values, selection rules, validation order, display state,
//! and exact action split are retained here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::panel_header::{ExpandButton, PanelHeader, PanelHeaderState};
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};

pub const BEAD_SIZE_LABEL: &str = "Bead diameter";
pub const SOME_BELOW_STORAGE_THRESHOLD: i32 = 0;
pub const ONLY_ABOVE_STORAGE_THRESHOLD: i32 = -1;

/// Java `NewstackOrBlendmont3dFindParent` calls from this source unit.
pub trait NewstackOrBlendmont3dFindParent {
    fn get_bead_size(&self) -> String;
    fn is_fiducialess(&self) -> bool;
}

/// Java `ConstFindBeads3dParam` reads.
pub trait ConstFindBeads3dParam {
    fn value(&self, field: FindBeads3dField) -> Option<String>;
    fn storage_threshold(&self) -> Option<i32>;
}

/// Java `FindBeads3dParam` writes.
pub trait FindBeads3dParam: ConstFindBeads3dParam {
    fn set_value(&mut self, field: FindBeads3dField, value: String) -> Result<(), String>;
    fn set_input_file(&mut self, value: &str);
    fn set_output_file(&mut self, value: String);
    fn set_storage_threshold_number(&mut self, value: i32);
}

/// Every `FindBeads3dParam` property used by `FindBeads3dPanel.java`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FindBeads3dField {
    BeadSize,
    MinSpacing,
    GuessNumBeads,
    MinRelativeStrength,
    ThresholdForAveraging,
    StorageThreshold,
    MaxNumBeads,
}

/// Direct `ApplicationManager` calls made by this source unit.
pub trait FindBeads3dPanelApplicationManager {
    fn calc_unbinned_bead_diameter_pixels(&self) -> String;
    fn find_beads3d_output_model_file_name(&self, axis_id: AxisID) -> String;
    fn find_beads3d(
        &mut self,
        process_button: &FindBeads3dProcessButton,
        deferred_3dmod_button: Option<&FindBeads3dViewButton>,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
    );
    fn imod_find_beads3d(
        &mut self,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
        image_file: &str,
        model_file: String,
        dialog_type: DialogType,
    );
    fn pack(&mut self, axis_id: AxisID);
}

/// The Java `Run3dmodButton` owned by `ProcessResultDisplayFactory`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FindBeads3dProcessButton {
    pub action_command: String,
    pub state_key: String,
    pub selected: bool,
    pub container_set: bool,
    pub deferred_view_set: bool,
    pub action_listener_count: usize,
    pub tooltip: Option<String>,
}

impl Default for FindBeads3dProcessButton {
    fn default() -> Self {
        Self {
            action_command: "Run Findbeads3d".into(),
            state_key: "findBeads3d".into(),
            selected: false,
            container_set: false,
            deferred_view_set: false,
            action_listener_count: 0,
            tooltip: None,
        }
    }
}

/// The Java `Run3dmodButton.get3dmodInstance` state at the GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FindBeads3dViewButton {
    pub action_command: String,
    pub action_listener_count: usize,
    pub tooltip: Option<String>,
}

impl Default for FindBeads3dViewButton {
    fn default() -> Self {
        Self {
            action_command: "View 3D Model on Tomogram".into(),
            action_listener_count: 0,
            tooltip: None,
        }
    }
}

/// Java `RadioTextField` state used by this source unit.
#[derive(Clone, Debug)]
pub struct RadioTextField {
    pub selected: bool,
    pub field: LabeledTextField,
}

impl RadioTextField {
    pub fn new(field_type: FieldType, label: &str) -> Self {
        Self {
            selected: false,
            field: LabeledTextField::new(field_type, label),
        }
    }
    pub fn is_selected(&self) -> bool {
        self.selected
    }
    pub fn set_text_number(&mut self, value: impl std::fmt::Display) {
        self.field.set_text_number(value);
    }
    pub fn get_text(&self, validation: bool) -> Result<String, FieldValidationFailedException> {
        self.field.get_text_validated(validation)
    }
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.field.set_tool_tip_text(text);
    }
}

/// Java `ReconScreenState` members read and written here.
pub trait FindBeads3dScreenState {
    fn stack_find_beads3d_header_state(&mut self) -> &mut PanelHeaderState;
    fn get_button_state(&self, key: &str) -> bool;
    fn set_button_state(&mut self, key: String, value: bool);
}

/// Source-shaped in-memory screen state for native frontends and tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FindBeads3dPanelScreenState {
    pub header_state: PanelHeaderState,
    pub button_states: std::collections::BTreeMap<String, bool>,
}

impl FindBeads3dScreenState for FindBeads3dPanelScreenState {
    fn stack_find_beads3d_header_state(&mut self) -> &mut PanelHeaderState {
        &mut self.header_state
    }
    fn get_button_state(&self, key: &str) -> bool {
        self.button_states.get(key).copied().unwrap_or(false)
    }
    fn set_button_state(&mut self, key: String, value: bool) {
        self.button_states.insert(key, value);
    }
}

/// Source-visible `JPanel` hierarchy and listener state created by `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FindBeads3dPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_etched_border: bool,
    pub root_component_order: Vec<String>,
    pub body_box_layout_y_axis: bool,
    pub body_component_order: Vec<String>,
    pub panel_a_box_layout_x_axis: bool,
    pub panel_b_box_layout_x_axis: bool,
    pub storage_threshold_grid: Option<(i32, i32, i32, i32)>,
    pub storage_threshold_border: Option<String>,
    pub storage_threshold_component_order: Vec<String>,
    pub buttons_box_layout_x_axis: bool,
    pub buttons_component_order: Vec<String>,
    pub body_visible: bool,
    pub tooltip_initialized: bool,
}

/// Java final `FindBeads3dPanel`.
pub struct FindBeads3dPanel {
    pub pnl_root: FindBeads3dPanelLayout,
    pub ltf_bead_size: LabeledTextField,
    pub ltf_min_spacing: LabeledTextField,
    pub ltf_guess_num_beads: LabeledTextField,
    pub ltf_min_relative_strength: LabeledTextField,
    pub ltf_threshold_for_averaging: LabeledTextField,
    pub bg_storage_threshold: Rc<RefCell<RadioButtonGroup>>,
    pub rb_storage_threshold_some_below: RadioButton,
    pub rb_storage_threshold_only_above: RadioButton,
    pub rtf_storage_threshold: RadioTextField,
    pub ltf_max_num_beads: LabeledTextField,
    pub btn_3dmod_find_beads3d: FindBeads3dViewButton,
    pub btn_find_beads3d: FindBeads3dProcessButton,
    pub header: PanelHeader,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
}

impl FindBeads3dPanel {
    /// Java private constructor `FindBeads3dPanel(...)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let bg_storage_threshold = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root: FindBeads3dPanelLayout {
                body_visible: true,
                ..Default::default()
            },
            ltf_bead_size: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Bead diameter (pixels): ",
            ),
            ltf_min_spacing: LabeledTextField::new(FieldType::FloatingPoint, "Minimum spacing: "),
            ltf_guess_num_beads: LabeledTextField::new(
                FieldType::Integer,
                "Estimated number of beads: ",
            ),
            ltf_min_relative_strength: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Minimum peak strength: ",
            ),
            ltf_threshold_for_averaging: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Threshold for averaging: ",
            ),
            bg_storage_threshold: bg_storage_threshold.clone(),
            rb_storage_threshold_some_below: RadioButton::new_with_enumerated_type(
                Some("Store some points below threshold".into()),
                EnumeratedTypeBoundary {
                    label: String::new(),
                    default: true,
                    value: Some(SOME_BELOW_STORAGE_THRESHOLD.to_string()),
                },
                Some(bg_storage_threshold.clone()),
            ),
            rb_storage_threshold_only_above: RadioButton::new_with_enumerated_type(
                Some("Store only points above threshold".into()),
                EnumeratedTypeBoundary {
                    label: String::new(),
                    default: false,
                    value: Some(ONLY_ABOVE_STORAGE_THRESHOLD.to_string()),
                },
                Some(bg_storage_threshold),
            ),
            rtf_storage_threshold: RadioTextField::new(
                FieldType::FloatingPoint,
                "Set threshold for storing: ",
            ),
            ltf_max_num_beads: LabeledTextField::new(FieldType::Integer, "Max points to analyze: "),
            btn_3dmod_find_beads3d: FindBeads3dViewButton::default(),
            btn_find_beads3d: FindBeads3dProcessButton::default(),
            header: PanelHeader::new(
                "Find Beads 3d",
                true,
                false,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            axis_id,
            dialog_type,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_find_beads3d.action_listener_count += 1;
        self.btn_3dmod_find_beads3d.action_listener_count += 1;
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_find_beads3d.action_listener_count = self
            .btn_find_beads3d
            .action_listener_count
            .saturating_sub(1);
    }

    /// Java `createPanel`.
    pub fn create_panel(&mut self) {
        self.btn_find_beads3d.container_set = true;
        self.btn_find_beads3d.deferred_view_set = true;
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_etched_border = true;
        self.pnl_root.root_component_order = vec!["header".into(), "pnlBody".into()];
        self.pnl_root.body_box_layout_y_axis = true;
        self.pnl_root.body_component_order = vec![
            "ltfBeadSize".into(),
            "pnlA".into(),
            "pnlB".into(),
            "pnlStorageThreshold".into(),
            "ltfMaxNumBeads".into(),
            "pnlButtons".into(),
        ];
        self.pnl_root.panel_a_box_layout_x_axis = true;
        self.pnl_root.panel_b_box_layout_x_axis = true;
        self.pnl_root.storage_threshold_grid = Some((3, 2, 3, 3));
        self.pnl_root.storage_threshold_border = Some("Storage Threshold".into());
        self.pnl_root.storage_threshold_component_order = vec![
            "rbStorageThresholdSomeBelow".into(),
            "rbStorageThresholdOnlyAbove".into(),
            "rtfStorageThreshold".into(),
        ];
        self.pnl_root.buttons_box_layout_x_axis = true;
        self.pnl_root.buttons_component_order =
            vec!["btnFindBeads3d".into(), "btn3dmodFindBeads3d".into()];
    }

    /// Java `isAdvanced`.
    pub fn is_advanced(&self) -> bool {
        self.header.is_advanced()
    }

    /// Java `expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global(&mut self) {}

    /// Java `expand(ExpandButton)`.
    pub fn expand<M: FindBeads3dPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        button: &ExpandButton,
    ) {
        if self.header.equals_open_close(button) {
            self.pnl_root.body_visible = button.is_expanded();
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        manager.pack(self.axis_id);
    }

    /// Java `updateAdvanced`.
    pub fn update_advanced(&mut self, advanced: bool) {
        self.ltf_min_spacing.set_visible(advanced);
        self.ltf_guess_num_beads.set_visible(advanced);
        self.ltf_min_relative_strength.set_visible(advanced);
        self.ltf_threshold_for_averaging.set_visible(advanced);
        self.ltf_max_num_beads.set_visible(advanced);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_screen_state<S: FindBeads3dScreenState>(&self, screen_state: &mut S) {
        self.header
            .get_state(Some(screen_state.stack_find_beads3d_header_state()));
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_screen_state<S: FindBeads3dScreenState>(&mut self, screen_state: &mut S) {
        self.header
            .set_state(Some(screen_state.stack_find_beads3d_header_state()));
        self.btn_find_beads3d.selected =
            screen_state.get_button_state(&self.btn_find_beads3d.state_key);
    }

    /// Java `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters<P: ConstFindBeads3dParam, M: FindBeads3dPanelApplicationManager>(
        &mut self,
        manager: &M,
        param: &P,
        initialize: bool,
    ) {
        if initialize {
            self.ltf_bead_size
                .set_text(&manager.calc_unbinned_bead_diameter_pixels());
            self.ltf_min_spacing.set_text_number(0.9);
            self.ltf_min_relative_strength.set_text_number(0.05);
        } else {
            self.ltf_bead_size
                .set_text(&param.value(FindBeads3dField::BeadSize).unwrap_or_default());
            self.ltf_min_spacing.set_text(
                &param
                    .value(FindBeads3dField::MinSpacing)
                    .unwrap_or_default(),
            );
            self.ltf_guess_num_beads.set_text(
                &param
                    .value(FindBeads3dField::GuessNumBeads)
                    .unwrap_or_default(),
            );
            self.ltf_min_relative_strength.set_text(
                &param
                    .value(FindBeads3dField::MinRelativeStrength)
                    .unwrap_or_default(),
            );
            self.ltf_threshold_for_averaging.set_text(
                &param
                    .value(FindBeads3dField::ThresholdForAveraging)
                    .unwrap_or_default(),
            );
            match param.storage_threshold() {
                Some(SOME_BELOW_STORAGE_THRESHOLD) => {
                    self.rb_storage_threshold_some_below.set_selected(true)
                }
                Some(ONLY_ABOVE_STORAGE_THRESHOLD) => {
                    self.rb_storage_threshold_only_above.set_selected(true)
                }
                Some(value) => self.rtf_storage_threshold.set_text_number(value),
                None => self.rtf_storage_threshold.set_text_number(""),
            }
            self.ltf_max_num_beads.set_text(
                &param
                    .value(FindBeads3dField::MaxNumBeads)
                    .unwrap_or_default(),
            );
        }
    }

    /// Java `getParameters(FindBeads3dParam, boolean)`.
    pub fn get_parameters<P: FindBeads3dParam, M: FindBeads3dPanelApplicationManager>(
        &self,
        manager: &M,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            param.set_input_file("TILT_3D_FIND_OUTPUT");
            param.set_output_file(manager.find_beads3d_output_model_file_name(self.axis_id));
            param
                .set_value(
                    FindBeads3dField::BeadSize,
                    self.ltf_bead_size
                        .get_text_validated(do_validation)
                        .map_err(|error| error)?,
                )
                .map_err(FieldValidationFailedException)?;
            param
                .set_value(
                    FindBeads3dField::MinSpacing,
                    self.ltf_min_spacing.get_text_validated(do_validation)?,
                )
                .map_err(FieldValidationFailedException)?;
            param
                .set_value(
                    FindBeads3dField::GuessNumBeads,
                    self.ltf_guess_num_beads.get_text_validated(do_validation)?,
                )
                .map_err(FieldValidationFailedException)?;
            param
                .set_value(
                    FindBeads3dField::MinRelativeStrength,
                    self.ltf_min_relative_strength
                        .get_text_validated(do_validation)?,
                )
                .map_err(FieldValidationFailedException)?;
            param
                .set_value(
                    FindBeads3dField::ThresholdForAveraging,
                    self.ltf_threshold_for_averaging
                        .get_text_validated(do_validation)?,
                )
                .map_err(FieldValidationFailedException)?;
            if !self.rtf_storage_threshold.is_selected() {
                param.set_storage_threshold_number(
                    if self.rb_storage_threshold_some_below.is_selected() {
                        SOME_BELOW_STORAGE_THRESHOLD
                    } else {
                        ONLY_ABOVE_STORAGE_THRESHOLD
                    },
                );
            } else {
                param
                    .set_value(
                        FindBeads3dField::StorageThreshold,
                        self.rtf_storage_threshold.get_text(do_validation)?,
                    )
                    .map_err(FieldValidationFailedException)?;
            }
            param
                .set_value(
                    FindBeads3dField::MaxNumBeads,
                    self.ltf_max_num_beads.get_text_validated(do_validation)?,
                )
                .map_err(FieldValidationFailedException)?;
            Ok(())
        })();
        result.is_ok()
    }

    /// Java `isFiducialess`.
    pub fn is_fiducialess<P: NewstackOrBlendmont3dFindParent>(&self, parent: &P) -> bool {
        parent.is_fiducialess()
    }

    /// Java `getBeadSize`.
    pub fn get_bead_size(&self) -> String {
        self.ltf_bead_size.get_text()
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: FindBeads3dPanelApplicationManager>(
        &self,
        manager: &mut M,
        command: &str,
        deferred_3dmod_button: Option<&FindBeads3dViewButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if command == self.btn_find_beads3d.action_command {
            manager.find_beads3d(
                &self.btn_find_beads3d,
                deferred_3dmod_button,
                self.axis_id,
                options,
                self.dialog_type,
            );
        } else if command == self.btn_3dmod_find_beads3d.action_command {
            manager.imod_find_beads3d(
                self.axis_id,
                options,
                "TILT_3D_FIND_OUTPUT",
                manager.find_beads3d_output_model_file_name(self.axis_id),
                self.dialog_type,
            );
        }
    }

    /// Java `setToolTipText`; autodoc lookup is a storage boundary.
    pub fn set_tool_tip_text(&mut self) {
        self.ltf_bead_size
            .set_tool_tip_text(Some("Size of beads in unbinned pixels."));
        self.rb_storage_threshold_some_below.set_tool_tip_text(Some("Model will include some points that are probably not beads, because their relative peak strengths are below the threshold between beads and non-beads"));
        self.rb_storage_threshold_only_above.set_tool_tip_text(Some("Model will include only the points with relative peak strengths above the threshold between beads and non-beads"));
        self.rtf_storage_threshold.set_tool_tip_text(Some(
            "Threshold relative peak strength (between 0 and 1) for storing peaks in model",
        ));
        self.btn_find_beads3d.tooltip =
            Some("Run findbeads3d to find gold particles in the tomogram.".into());
        self.btn_3dmod_find_beads3d.tooltip = Some("View model of gold particles.".into());
        self.pnl_root.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Param {
        values: BTreeMap<FindBeads3dField, String>,
        storage: Option<i32>,
        input: String,
        output: String,
    }
    impl ConstFindBeads3dParam for Param {
        fn value(&self, field: FindBeads3dField) -> Option<String> {
            self.values.get(&field).cloned()
        }
        fn storage_threshold(&self) -> Option<i32> {
            self.storage
        }
    }
    impl FindBeads3dParam for Param {
        fn set_value(&mut self, field: FindBeads3dField, value: String) -> Result<(), String> {
            self.values.insert(field, value);
            Ok(())
        }
        fn set_input_file(&mut self, value: &str) {
            self.input = value.into();
        }
        fn set_output_file(&mut self, value: String) {
            self.output = value;
        }
        fn set_storage_threshold_number(&mut self, value: i32) {
            self.storage = Some(value);
        }
    }
    #[derive(Default)]
    struct Manager {
        packed: Vec<AxisID>,
        action: Option<String>,
    }
    impl FindBeads3dPanelApplicationManager for Manager {
        fn calc_unbinned_bead_diameter_pixels(&self) -> String {
            "12.5".into()
        }
        fn find_beads3d_output_model_file_name(&self, axis_id: AxisID) -> String {
            format!("dataset{axis_id}_3dfind.mod")
        }
        fn find_beads3d(
            &mut self,
            _: &FindBeads3dProcessButton,
            _: Option<&FindBeads3dViewButton>,
            axis: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
        ) {
            self.action = Some(format!("find:{axis}"));
        }
        fn imod_find_beads3d(
            &mut self,
            axis: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: &str,
            _: String,
            _: DialogType,
        ) {
            self.action = Some(format!("view:{axis}"));
        }
        fn pack(&mut self, axis: AxisID) {
            self.packed.push(axis);
        }
    }
    #[test]
    fn source_panel_creation_and_advanced_visibility_are_preserved() {
        let mut panel =
            FindBeads3dPanel::get_instance(AxisID::First, DialogType::FinalAlignedStack);
        assert_eq!(panel.pnl_root.root_component_order, ["header", "pnlBody"]);
        assert_eq!(panel.pnl_root.storage_threshold_grid, Some((3, 2, 3, 3)));
        panel.update_advanced(false);
        assert!(!panel.ltf_min_spacing.is_visible());
        assert!(!panel.ltf_max_num_beads.is_visible());
        assert!(panel.btn_find_beads3d.container_set);
    }
    #[test]
    fn parameter_transfer_uses_source_storage_threshold_branch() {
        let panel = FindBeads3dPanel::get_instance(AxisID::First, DialogType::FinalAlignedStack);
        let mut param = Param::default();
        let manager = Manager::default();
        assert!(panel.get_parameters(&manager, &mut param, true));
        assert_eq!(param.input, "TILT_3D_FIND_OUTPUT");
        assert_eq!(param.storage, Some(SOME_BELOW_STORAGE_THRESHOLD));
        assert_eq!(param.output, "datasetFirst_3dfind.mod");
    }
    #[test]
    fn source_action_commands_dispatch_the_two_manager_calls() {
        let panel = FindBeads3dPanel::get_instance(AxisID::Second, DialogType::FinalAlignedStack);
        let mut manager = Manager::default();
        panel.action(&mut manager, "Run Findbeads3d", None, None);
        assert_eq!(manager.action.as_deref(), Some("find:Second"));
        panel.action(&mut manager, "View 3D Model on Tomogram", None, None);
        assert_eq!(manager.action.as_deref(), Some("view:Second"));
    }
}
