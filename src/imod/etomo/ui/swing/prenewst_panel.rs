//! `IMOD/Etomo/src/etomo/ui/swing/PrenewstPanel.java`.
//!
//! Swing widgets, `ApplicationManager`, the process-result factory, context
//! popup, and `UIHarness` are direct boundaries.  This unit owns exactly the
//! panel's construction state, parameter transfers, expander transitions, and
//! two process action routes.
#![allow(dead_code)]

use super::{
    check_box::CheckBox,
    labeled_spinner::LabeledSpinner,
    panel_header::{ExpandButton, ExpandButtonType, PanelHeader, PanelHeaderState},
};
use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{axis_id::AxisID, dialog_type::DialogType, view_type::ViewType},
};

pub const DATA_MODE_BYTE: i32 = 0;
pub const DATA_MODE_DEFAULT: i32 = -1;
pub const FLOAT_DENSITIES_MEAN: i32 = 1;
pub const FLOAT_DENSITIES_DEFAULT: i32 = 0;
pub const PREALIGNED: &str = "PREALIGNED";

/// Java `NewstParam` values read and written by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrenewstParam {
    pub bin_by_factor: i32,
    pub antialias_filter: Option<String>,
    pub mode_to_output: i32,
    pub float_densities: i32,
    pub command_mode: Option<String>,
}

impl Default for PrenewstParam {
    fn default() -> Self {
        Self {
            bin_by_factor: 1,
            antialias_filter: None,
            mode_to_output: DATA_MODE_DEFAULT,
            float_densities: FLOAT_DENSITIES_DEFAULT,
            command_mode: None,
        }
    }
}

/// Java `BlendmontParam` values used by this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrenewstBlendmontParam {
    pub bin_by_factor: Option<i32>,
}

/// Java `ConstMetaData` / `MetaData` antialias value scoped by dialog and axis.
pub trait PrenewstMetaData {
    fn antialias_filter(&self, dialog_type: DialogType, axis_id: AxisID) -> Option<String>;
    fn set_antialias_filter(
        &mut self,
        dialog_type: DialogType,
        axis_id: AxisID,
        value: Option<String>,
    );
}

/// Java `BaseScreenState` calls made by this unit.
pub trait PrenewstScreenState {
    fn get_header_state(&self) -> PanelHeaderState;
    fn set_header_state(&mut self, state: PanelHeaderState);
}

/// Java `CoarseAlignDialog.isFiducialess` boundary.
pub trait PrenewstParent {
    fn is_fiducialess(&self) -> bool;
}

/// Java `ApplicationManager` and process-result-factory calls made by this unit.
pub trait PrenewstApplicationManager {
    type MetaData: PrenewstMetaData;
    fn view_type(&self) -> ViewType;
    fn meta_data(&self) -> &Self::MetaData;
    fn coarse_align(
        &mut self,
        axis_id: AxisID,
        deferred_3dmod_button: Option<&PrenewstDeferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
    );
    fn imod_coarse_align(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn pack(&mut self, axis_id: AxisID);
}

/// Java `Deferred3dmodButton` boundary, passed unmodified to `coarseAlign`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrenewstDeferred3dmodButton;

/// Source-visible state of `pnlPrenewst`, `pnlBody`, and local Swing layout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrenewstPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_etched_border: bool,
    pub root_alignment_x: f32,
    pub root_mouse_listener_count: usize,
    pub body_box_layout_y_axis: bool,
    pub body_visible: bool,
    pub check_boxes_box_layout_y_axis: bool,
    pub antialias_panel_present: bool,
    pub byte_mode_panel_present: bool,
    pub buttons: Vec<String>,
    pub root_components: Vec<String>,
}

/// Complete Java `PrenewstPanel` state.
pub struct PrenewstPanel<M: PrenewstApplicationManager, P: PrenewstParent> {
    pub pnl_prenewst: PrenewstPanelLayout,
    pub cb_byte_mode_to_output: CheckBox,
    pub cb_mean_float_densities: CheckBox,
    pub cb_antialias_filter: Option<CheckBox>,
    pub antialias_filter_value: Option<String>,
    pub spin_binning: LabeledSpinner,
    pub header: PanelHeader,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub manager: M,
    pub parent: P,
    pub montage: bool,
    pub btn_coarse_align_action_command: String,
    pub btn_imod_action_command: String,
    pub btn_coarse_align_listener_count: usize,
    pub btn_imod_listener_count: usize,
    pub deferred_3dmod_button_set: bool,
    pub btn_coarse_align_tooltip: Option<String>,
    pub btn_imod_tooltip: Option<String>,
}

impl<M: PrenewstApplicationManager, P: PrenewstParent> PrenewstPanel<M, P> {
    /// Java package-private constructor and exact initialization sequence.
    pub fn new(manager: M, axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        let montage = manager.view_type() == ViewType::Montage;
        let mut panel = Self {
            pnl_prenewst: PrenewstPanelLayout {
                root_box_layout_y_axis: true,
                root_etched_border: true,
                body_box_layout_y_axis: true,
                body_visible: true,
                check_boxes_box_layout_y_axis: true,
                antialias_panel_present: !montage,
                byte_mode_panel_present: !montage,
                buttons: vec!["btnCoarseAlign".into(), "btnImod".into()],
                root_components: vec!["header".into(), "pnlBody".into()],
                ..Default::default()
            },
            cb_byte_mode_to_output: CheckBox::new_with_text("Convert to bytes"),
            cb_mean_float_densities: CheckBox::new_with_text("Float intensities to mean"),
            cb_antialias_filter: (!montage)
                .then(|| CheckBox::new_with_text("Reduce size with antialiasing filter")),
            antialias_filter_value: None,
            spin_binning: LabeledSpinner::get_instance(
                "Coarse aligned image stack binning ",
                1,
                1,
                8,
                1,
            ),
            header: PanelHeader::new(
                if montage { "Blendmont" } else { "Newstack" },
                false,
                true,
                dialog_type,
                false,
                true,
                false,
                false,
                false,
            ),
            axis_id,
            dialog_type,
            manager,
            parent,
            montage,
            btn_coarse_align_action_command: "Coarse Align".into(),
            btn_imod_action_command: "View Aligned Stack In 3dmod".into(),
            btn_coarse_align_listener_count: 0,
            btn_imod_listener_count: 0,
            deferred_3dmod_button_set: true,
            btn_coarse_align_tooltip: None,
            btn_imod_tooltip: None,
        };
        panel.add_listeners();
        panel.set_tool_tip_text();
        panel
    }

    /// Java `expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global(&mut self) {}

    /// Java `expand(ExpandButton)`.
    pub fn expand(&mut self, button: &ExpandButton) {
        if self.header.equals_open_close(button) {
            self.pnl_prenewst.body_visible = button.is_expanded();
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        self.manager.pack(self.axis_id);
    }

    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&mut self, state: bool) {
        self.spin_binning.set_visible(state);
        if let Some(cb) = &mut self.cb_antialias_filter {
            cb.set_visible(state);
        }
        self.cb_byte_mode_to_output.set_visible(state);
        self.cb_mean_float_densities.set_visible(state);
    }

    /// Java private `updateEnabled`.
    pub fn update_enabled(&mut self) {
        if let Some(cb) = &mut self.cb_antialias_filter {
            cb.set_enabled(self.spin_binning.get_value() > 1);
        }
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_coarse_align_listener_count = 0;
    }
    /// Java `getPanel` boundary representation.
    pub fn get_panel(&self) -> &PrenewstPanelLayout {
        &self.pnl_prenewst
    }
    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&mut self, align: f32) {
        self.pnl_prenewst.root_alignment_x = align;
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_meta_data_parameters(&mut self) {
        if let Some(value) = self
            .manager
            .meta_data()
            .antialias_filter(self.dialog_type, self.axis_id)
        {
            self.antialias_filter_value = Some(value);
        }
    }
    /// Java `setParameters(ConstNewstParam)`.
    pub fn set_newst_parameters(&mut self, input: &PrenewstParam) {
        if input.bin_by_factor > 1 {
            self.spin_binning.set_value_int(input.bin_by_factor);
        }
        let antialias = input.antialias_filter.is_some();
        if let Some(cb) = &mut self.cb_antialias_filter {
            cb.set_selected(antialias);
        }
        if antialias {
            self.antialias_filter_value = input.antialias_filter.clone();
        }
        self.cb_byte_mode_to_output
            .set_selected(input.mode_to_output == DATA_MODE_BYTE);
        self.cb_mean_float_densities
            .set_selected(input.float_densities == FLOAT_DENSITIES_MEAN);
        self.update_enabled();
    }
    /// Java `setParameters(BaseScreenState)`.
    pub fn set_screen_state_parameters<S: PrenewstScreenState>(&mut self, input: &S) {
        let state = input.get_header_state();
        self.header.set_state(Some(&state));
    }
    /// Java `getParameters(BaseScreenState)`.
    pub fn get_screen_state_parameters<S: PrenewstScreenState>(&self, output: &mut S) {
        let mut state = PanelHeaderState::default();
        self.header.get_state(Some(&mut state));
        output.set_header_state(state);
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_meta_data_parameters<D: PrenewstMetaData>(&self, output: &mut D) {
        output.set_antialias_filter(
            self.dialog_type,
            self.axis_id,
            self.antialias_filter_value.clone(),
        );
    }
    /// Java `setParameters(BlendmontParam)`.
    pub fn set_blendmont_parameters(&mut self, input: &PrenewstBlendmontParam) {
        if let Some(value) = input.bin_by_factor {
            self.spin_binning.set_value_int(value);
        }
    }
    /// Java `getParameters(NewstParam, boolean)`.
    pub fn get_newst_parameters(&self, output: &mut PrenewstParam, _do_validation: bool) -> bool {
        output.command_mode = Some(PREALIGNED.into());
        output.bin_by_factor = if self.spin_binning.get_value() > 1 {
            self.spin_binning.get_value()
        } else {
            i32::MIN
        };
        if let Some(cb) = &self.cb_antialias_filter {
            output.antialias_filter = cb
                .is_selected()
                .then(|| self.antialias_filter_value.clone().unwrap_or_default());
        }
        output.mode_to_output = if self.cb_byte_mode_to_output.is_selected() {
            DATA_MODE_BYTE
        } else {
            DATA_MODE_DEFAULT
        };
        output.float_densities = if self.cb_mean_float_densities.is_selected() {
            FLOAT_DENSITIES_MEAN
        } else {
            FLOAT_DENSITIES_DEFAULT
        };
        true
    }
    /// Java `getProcessName` result.
    pub fn get_process_name(&self) -> &'static str {
        if self.montage { "preblend" } else { "prenewst" }
    }
    /// Java `isFiducialess`.
    pub fn is_fiducialess(&self) -> bool {
        self.parent.is_fiducialess()
    }
    /// Java `getParameters(BlendmontParam, boolean)`.
    pub fn get_blendmont_parameters(
        &self,
        output: &mut PrenewstBlendmontParam,
        _do_validation: bool,
    ) -> bool {
        output.bin_by_factor = Some(self.spin_binning.get_value());
        true
    }
    /// Java `popUpContextMenu`, retaining exactly the constructed popup inputs.
    pub fn pop_up_context_menu(&self) -> PrenewstContextPopup {
        PrenewstContextPopup {
            manual_page_label: "Newstack".into(),
            manual_page: "newstack.html".into(),
            log_file_label: "Prenewst".into(),
            log_file: format!("prenewst{}.log", self.axis_id.get_extension()),
            axis_id: self.axis_id,
        }
    }
    /// Java `validate`.
    pub fn validate(&self) -> bool {
        true
    }
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action(
        &mut self,
        command: &str,
        deferred: Option<&PrenewstDeferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if command == self.btn_coarse_align_action_command {
            self.manager
                .coarse_align(self.axis_id, deferred, options, self.dialog_type);
        } else if command == self.btn_imod_action_command {
            self.manager.imod_coarse_align(self.axis_id, options);
        }
    }
    /// Java `PrenewstPanelActionListener.actionPerformed`.
    pub fn action_performed(&mut self, command: &str) {
        self.action(command, None, None);
    }
    /// Java `PrenewstBinningChangeListener.stateChanged`.
    pub fn state_changed(&mut self) {
        self.update_enabled();
    }
    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_coarse_align_listener_count += 1;
        self.btn_imod_listener_count += 1;
        self.spin_binning.add_change_listener();
        self.pnl_prenewst.root_mouse_listener_count += 1;
    }
    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.spin_binning.set_tool_tip_text(Some(
            "Binning for the image stack used to generate and fix the fiducial model.",
        ));
        self.cb_byte_mode_to_output.set_tool_tip_text(Some("Set the storage mode of the output file to bytes.  When unchecked the storage mode is the same as that of the first input file.  This option should be turned off when the dynamic range is still too poor after X ray removal.  Command:  -mode 0"));
        self.cb_mean_float_densities.set_tool_tip_text(Some("Adjust densities of sections individually.  Scale sections to common mean and standard deviation.  Command:  -float 1"));
        self.btn_coarse_align_tooltip =
            Some("Use transformations to produce stack of aligned images.".into());
        self.btn_imod_tooltip = Some("Use 3dmod to view the coarsely aligned images.".into());
        if let Some(cb) = &mut self.cb_antialias_filter {
            cb.set_tool_tip_text(Some("Use antialiased image reduction instead binning with the default filter in Newstack; useful for data from direct detection cameras."));
        }
    }
}

/// Java `ContextPopup` constructor values made by `popUpContextMenu`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrenewstContextPopup {
    pub manual_page_label: String,
    pub manual_page: String,
    pub log_file_label: String,
    pub log_file: String,
    pub axis_id: AxisID,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Meta {
        antialias: Option<String>,
    }
    impl PrenewstMetaData for Meta {
        fn antialias_filter(&self, _: DialogType, _: AxisID) -> Option<String> {
            self.antialias.clone()
        }
        fn set_antialias_filter(&mut self, _: DialogType, _: AxisID, value: Option<String>) {
            self.antialias = value;
        }
    }
    struct Manager {
        view: ViewType,
        meta: Meta,
        calls: Vec<String>,
    }
    impl PrenewstApplicationManager for Manager {
        type MetaData = Meta;
        fn view_type(&self) -> ViewType {
            self.view
        }
        fn meta_data(&self) -> &Meta {
            &self.meta
        }
        fn coarse_align(
            &mut self,
            _: AxisID,
            _: Option<&PrenewstDeferred3dmodButton>,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
        ) {
            self.calls.push("coarse".into());
        }
        fn imod_coarse_align(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.calls.push("imod".into());
        }
        fn pack(&mut self, _: AxisID) {
            self.calls.push("pack".into());
        }
    }
    struct Parent(bool);
    impl PrenewstParent for Parent {
        fn is_fiducialess(&self) -> bool {
            self.0
        }
    }
    fn panel(montage: bool) -> PrenewstPanel<Manager, Parent> {
        PrenewstPanel::new(
            Manager {
                view: if montage {
                    ViewType::Montage
                } else {
                    ViewType::SingleView
                },
                meta: Meta::default(),
                calls: vec![],
            },
            AxisID::First,
            DialogType::CoarseAlignment,
            Parent(true),
        )
    }
    #[test]
    fn newst_round_trip_and_action_routes_match_source() {
        let mut panel = panel(false);
        panel.set_newst_parameters(&PrenewstParam {
            bin_by_factor: 4,
            antialias_filter: Some("0.7".into()),
            mode_to_output: DATA_MODE_BYTE,
            float_densities: FLOAT_DENSITIES_MEAN,
            command_mode: None,
        });
        let mut output = PrenewstParam::default();
        assert!(panel.get_newst_parameters(&mut output, true));
        assert_eq!(output.bin_by_factor, 4);
        assert_eq!(output.command_mode.as_deref(), Some(PREALIGNED));
        assert_eq!(output.antialias_filter.as_deref(), Some("0.7"));
        panel.action_performed("Coarse Align");
        panel.action_performed("View Aligned Stack In 3dmod");
        assert_eq!(panel.manager.calls, ["coarse", "imod"]);
        assert!(panel.is_fiducialess());
    }
    #[test]
    fn montage_is_preblend_and_omits_antialias_control() {
        let mut panel = panel(true);
        assert!(panel.cb_antialias_filter.is_none());
        assert_eq!(panel.get_process_name(), "preblend");
        panel.update_advanced(false);
        assert!(!panel.spin_binning.panel_visible);
        assert_eq!(panel.pop_up_context_menu().log_file, "prenewsta.log");
    }
    #[test]
    fn spinner_changes_enable_antialias_only_above_one() {
        let mut panel = panel(false);
        panel.spin_binning.set_value_int(1);
        panel.state_changed();
        assert!(
            !panel
                .cb_antialias_filter
                .as_ref()
                .unwrap()
                .check_box
                .enabled
        );
        panel.spin_binning.set_value_int(2);
        panel.state_changed();
        assert!(
            panel
                .cb_antialias_filter
                .as_ref()
                .unwrap()
                .check_box
                .enabled
        );
    }
}
