//! `IMOD/Etomo/src/etomo/ui/swing/NewstackOrBlendmontPanel.java`.
//!
//! Swing components and the parameter/com-script implementations remain their
//! direct boundaries.  This unit keeps the panel's source-owned control,
//! listener, parameter, advanced-state, and packing state so subclasses retain
//! the same call sequence as eTomo.
#![allow(dead_code)]

pub use super::global_expand_button::GlobalExpandButton;
use super::multi_line_button::MultiLineButton;
use super::newstack_and_blendmont_param_panel::NewstackAndBlendmontParamPanel;
use super::panel_header::{ExpandButton, PanelHeaderState};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

pub const RUN_BUTTON_LABEL: &str = "Create Full Aligned Stack";
pub const VIEW_FULL_ALIGNED_STACK_LABEL: &str = "View Full Aligned Stack";
pub const CREATE_FULL_ALIGNED_STACK_TOOLTIP: &str =
    "Generate the complete aligned stack for input into the tilt process.";
pub const VIEW_FULL_ALIGNED_STACK_TOOLTIP: &str = "Open the complete aligned stack in 3dmod";

/// Java `ConstNewstParam` / `NewstParam` boundary values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NewstParam {
    pub values: Vec<String>,
    pub bin_by_factor: Option<i32>,
    pub linear_interpolation: bool,
    pub antialias_filter: Option<f64>,
    pub size_to_output_in_x_and_y: Option<String>,
}

/// Java `BlendmontParam` boundary values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlendmontParam {
    pub values: Vec<String>,
    pub bin_by_factor: Option<i32>,
    pub linear_interpolation: bool,
    pub size_to_output_in_x_and_y: Option<String>,
    pub fiducialess: bool,
}

/// Java `ConstMetaData` / `MetaData` boundary values.
#[derive(Clone, Debug, PartialEq)]
pub struct MetaData {
    pub values: Vec<String>,
    /// Java `stack3dFindBinning`, indexed by `AxisID`.
    pub stack_3d_find_binning: [Option<i32>; 3],
    pub stack_binning: [i32; 3],
    pub size_to_output_in_x_and_y: [String; 3],
    pub antialias_filter: [Option<f64>; 3],
    pub ctf3d_setup_slab_thickness_in_nm_set: bool,
}

impl Default for MetaData {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            stack_3d_find_binning: [None; 3],
            stack_binning: [1; 3],
            size_to_output_in_x_and_y: std::array::from_fn(|_| String::new()),
            antialias_filter: [None; 3],
            ctf3d_setup_slab_thickness_in_nm_set: false,
        }
    }
}

impl MetaData {
    /// Java `setStack3dFindBinning`.
    pub fn set_stack_3d_find_binning(&mut self, axis_id: AxisID, binning: i32) {
        self.stack_3d_find_binning[axis_id.get_axis_of_extension() as usize] = Some(binning);
    }

    /// Java `isStack3dFindBinningSet`.
    pub fn is_stack_3d_find_binning_set(&self, axis_id: AxisID) -> bool {
        self.stack_3d_find_binning[axis_id.get_axis_of_extension() as usize].is_some()
    }

    /// Java `getStack3dFindBinning`.
    pub fn get_stack_3d_find_binning(&self, axis_id: AxisID) -> i32 {
        self.stack_3d_find_binning[axis_id.get_axis_of_extension() as usize]
            .expect("stack3dFindBinning must be set before getStack3dFindBinning")
    }
}

/// Compatibility name for the Java `FiducialessParams` interface implemented
/// by this exact source unit.
pub type FiducialessParams = NewstackAndBlendmontParamPanel;

/// Java `ReconScreenState` members accessed by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReconScreenState {
    pub newst_header_state: PanelHeaderState,
    pub button_state: bool,
}

/// Source-visible `JPanel` state of `pnlRoot` and its insertion order.
#[derive(Clone, Debug, PartialEq)]
pub struct NewstackPanelRoot {
    pub box_layout_y_axis: bool,
    pub alignment_x: f32,
    pub etched_border: bool,
    pub header_added: bool,
    pub body_added: bool,
    pub visible: bool,
}

impl Default for NewstackPanelRoot {
    fn default() -> Self {
        Self {
            box_layout_y_axis: false,
            alignment_x: 0.0,
            etched_border: false,
            header_added: false,
            body_added: false,
            visible: true,
        }
    }
}

/// Complete local state of Java `NewstackOrBlendmontPanel`.
#[derive(Clone, Debug, PartialEq)]
pub struct NewstackOrBlendmontPanel {
    pub pnl_root: NewstackPanelRoot,
    pub header_title: String,
    pub header_state: PanelHeaderState,
    pub pnl_body_contains_parameter_panel: bool,
    pub pnl_buttons_box_layout_x_axis: bool,
    pub pnl_buttons_contains_run: bool,
    pub pnl_buttons_contains_3dmod_full: bool,
    pub btn_3dmod_full: MultiLineButton,
    pub newstack_and_blendmont_param_panel: FiducialessParams,
    pub btn_run_process: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub action_listener_present: bool,
    pub deferred_3dmod_button_set: bool,
    pub packed_count: u32,
}

impl NewstackOrBlendmontPanel {
    /// Java package-private constructor.  The `ApplicationManager` and Swing
    /// `PanelHeader` construction are direct owner boundaries; all values read
    /// or subsequently changed here are retained locally.
    pub fn new(axis_id: AxisID, dialog_type: DialogType, header_title: impl Into<String>) -> Self {
        let header_title = header_title.into();
        let mut btn_3dmod_full = MultiLineButton::get_toggle_button_instance_with_label(Some(
            VIEW_FULL_ALIGNED_STACK_LABEL,
        ));
        btn_3dmod_full.set_action_command(Some(VIEW_FULL_ALIGNED_STACK_LABEL));
        let mut btn_run_process =
            MultiLineButton::get_toggle_button_instance_with_label(Some(RUN_BUTTON_LABEL));
        btn_run_process.set_action_command(Some(RUN_BUTTON_LABEL));
        Self {
            pnl_root: NewstackPanelRoot::default(),
            header_title,
            header_state: PanelHeaderState::default(),
            pnl_body_contains_parameter_panel: false,
            pnl_buttons_box_layout_x_axis: false,
            pnl_buttons_contains_run: false,
            pnl_buttons_contains_3dmod_full: false,
            btn_3dmod_full,
            newstack_and_blendmont_param_panel: NewstackAndBlendmontParamPanel::new(
                axis_id,
                dialog_type,
                crate::imod::etomo::r#type::view_type::ViewType::SingleView,
            ),
            btn_run_process,
            axis_id,
            dialog_type,
            action_listener_present: false,
            deferred_3dmod_button_set: false,
            packed_count: 0,
        }
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_run_process.add_action_listener();
        self.btn_3dmod_full.add_action_listener();
        self.action_listener_present = true;
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &NewstackPanelRoot {
        &self.pnl_root
    }

    /// Java `createPanel`.
    pub fn create_panel(&mut self) {
        self.deferred_3dmod_button_set = true;
        self.pnl_root.box_layout_y_axis = true;
        self.pnl_root.alignment_x = 0.5;
        self.pnl_root.etched_border = true;
        self.pnl_root.header_added = true;
        self.pnl_root.body_added = true;
        self.pnl_body_contains_parameter_panel = true;
        self.pnl_buttons_box_layout_x_axis = true;
        self.pnl_buttons_contains_run = true;
        self.pnl_buttons_contains_3dmod_full = true;
    }

    /// Java `isFiducialess`.
    pub fn is_fiducialess(&self) -> bool {
        self.newstack_and_blendmont_param_panel.is_fiducialess()
    }

    /// Java private `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.visible = visible;
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.action_listener_present = false;
        self.btn_run_process.button.action_listener_count = 0;
        self.btn_3dmod_full.button.action_listener_count = 0;
    }

    /// Java `getFiducialessParams`.
    pub fn get_fiducialess_params(&self) -> &FiducialessParams {
        &self.newstack_and_blendmont_param_panel
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_recon_screen_state(&mut self, screen_state: &ReconScreenState) {
        self.header_state = screen_state.newst_header_state.clone();
        self.btn_run_process
            .set_button_state(screen_state.button_state);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_recon_screen_state(&self, screen_state: &mut ReconScreenState) {
        screen_state.newst_header_state = self.header_state.clone();
        screen_state.button_state = self.btn_run_process.get_button_state();
    }

    /// Java `setParameters(ConstNewstParam)`.
    pub fn set_newst_parameters(&mut self, newst_param: &NewstParam) {
        self.newstack_and_blendmont_param_panel
            .set_newst_parameters(newst_param);
    }

    /// Java `getParameters(NewstParam, boolean)`.
    pub fn get_newst_parameters(&self, newst_param: &mut NewstParam, do_validation: bool) -> bool {
        self.newstack_and_blendmont_param_panel
            .get_newst_parameters(newst_param, do_validation)
    }

    /// Java `setParameters(BlendmontParam)`.
    pub fn set_blendmont_parameters(&mut self, param: &BlendmontParam) {
        self.newstack_and_blendmont_param_panel
            .set_blendmont_parameters(param);
    }

    /// Java `getParameters(BlendmontParam, boolean)`.
    pub fn get_blendmont_parameters(
        &self,
        param: &mut BlendmontParam,
        do_validation: bool,
    ) -> bool {
        self.newstack_and_blendmont_param_panel
            .get_blendmont_parameters(param, do_validation)
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_meta_data_parameters(&self, meta_data: &mut MetaData) {
        self.newstack_and_blendmont_param_panel
            .get_meta_data_parameters(meta_data);
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_meta_data_parameters(&mut self, meta_data: &MetaData) {
        self.newstack_and_blendmont_param_panel
            .set_meta_data_parameters(meta_data);
    }

    /// Java `setFiducialessAlignment`.
    pub fn set_fiducialess_alignment(&mut self, input: bool) {
        self.newstack_and_blendmont_param_panel
            .set_fiducialess_alignment(input);
    }

    /// Java `setImageRotation`.
    pub fn set_image_rotation(&mut self, input: impl Into<String>) {
        self.newstack_and_blendmont_param_panel
            .set_image_rotation(input);
    }

    /// Java `validate`.
    pub fn validate(&self) -> bool {
        true
    }

    /// Java `getRunProcessButtonActionCommand`.
    pub fn get_run_process_button_action_command(&self) -> &str {
        self.btn_run_process
            .get_action_command()
            .unwrap_or_default()
    }

    /// Java `get3dmodFullButtonActionCommand`.
    pub fn get_3dmod_full_button_action_command(&self) -> &str {
        self.btn_3dmod_full.get_action_command().unwrap_or_default()
    }

    /// Java `getRunProcessResultDisplay`.
    pub fn get_run_process_result_display(&self) -> &MultiLineButton {
        &self.btn_run_process
    }

    /// Java `expand(GlobalExpandButton)`, deliberately empty.
    pub fn expand_global(&mut self, _button: &GlobalExpandButton) {}

    /// Java `expand(ExpandButton)`, including the `UIHarness.pack` boundary.
    pub fn expand(&mut self, button: &ExpandButton) {
        if button.button_type == super::panel_header::ExpandButtonType::Advanced {
            self.newstack_and_blendmont_param_panel
                .update_advanced(button.is_expanded());
        }
        self.packed_count += 1;
    }

    /// Java `updateAdvanced`.
    pub fn update_advanced(&mut self, advanced: bool) {
        self.newstack_and_blendmont_param_panel
            .update_advanced(advanced);
    }

    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.btn_run_process
            .set_tool_tip_text(Some(CREATE_FULL_ALIGNED_STACK_TOOLTIP));
        self.btn_3dmod_full
            .set_tool_tip_text(Some(VIEW_FULL_ALIGNED_STACK_TOOLTIP));
    }

    /// Java private listener `actionPerformed`, reduced only to its exact
    /// source arguments; subclass action dispatch is its owner boundary.
    pub fn action_performed<'a>(
        &self,
        action_command: &'a str,
    ) -> (&'a str, Option<()>, Option<()>) {
        (action_command, None, None)
    }
}
