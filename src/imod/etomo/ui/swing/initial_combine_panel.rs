//! `IMOD/Etomo/src/etomo/ui/swing/InitialCombinePanel.java`.
//!
//! Swing construction, MRC-header reads, deferred 3dmod ownership, packing,
//! and concrete manager/dialog calls stay at their original integration
//! boundaries.  The fields, parameter routing, expansion state, and restart
//! dispatch are retained by this source-shaped Rust unit.
#![allow(dead_code)]

use crate::imod::etomo::{
    process::imod_process::Run3dmodMenuOptions,
    r#type::{axis_id::AxisID, dialog_type::DialogType, processing_method::ProcessingMethod},
    ui::field_type::FieldType,
};

use super::{
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    multi_line_button::MultiLineButton,
    panel_header::{ExpandButton, PanelHeader},
    solvematch_panel::{
        DualvolmatchParameters, FiducialMatch, SolvematchPanel, SolvematchPanelScreenState,
        SolvematchParameters,
    },
};

/// Java `MatchMode` used only to choose the two MRC headers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchMode {
    AToB,
    BToA,
}

/// Java `MatchvolParam` calls from this source unit.
pub trait MatchvolParameters {
    fn output_size_y(&self) -> String;
    fn set_output_size_y(&mut self, value: String);
}

/// The two MRC-header reads in Java `setMatchMode`.
pub trait InitialCombineMrcHeaders {
    fn tilt_output_n_rows(&mut self, axis: AxisID) -> Result<Option<i32>, String>;
}

/// Direct `TomogramCombinationDialog` calls from this source unit.
pub trait InitialCombinePanelParent {
    fn initial_tab_enabled(&self) -> bool;
    fn synchronize_initial(&mut self);
    fn imod_combined_button(&self) -> Option<&MultiLineButton>;
    fn run_processing_method(&self) -> ProcessingMethod;
    fn is_parallel(&self) -> bool;
    fn is_run_volcombine(&self) -> bool;
}

/// Direct `ApplicationManager` invocation from `action`.
pub trait InitialCombinePanelApplicationManager {
    fn matchvol1_combine(
        &mut self,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&MultiLineButton>,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
        parallel: bool,
        not_run_volcombine: bool,
    );
}

/// `ReconScreenState` restart-button slot in addition to Solvematch's slots.
pub trait InitialCombinePanelScreenState: SolvematchPanelScreenState {
    fn matchvol1_restart_button_state(&self) -> bool;
    fn set_matchvol1_restart_button_state(&mut self, state: bool);
}

/// Exact source-visible component hierarchy at the GUI renderer boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InitialCombinePanelLayout {
    pub root_visible: bool,
    pub solvematch_present: bool,
    pub matchvol1_present: bool,
    pub matchvol1_body_visible: bool,
    pub output_size_y_info_visible: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub output_size_y_info: String,
    pub component_order: Vec<&'static str>,
}

/// Java `ContextPopup` constructor arguments produced by `popUpContextMenu`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitialCombineContextPopup {
    pub title: &'static str,
    pub manual_page_labels: [&'static str; 2],
    pub manual_pages: [&'static str; 2],
    pub log_file_labels: [&'static str; 2],
    pub log_files: [&'static str; 2],
}

/// Java `InitialCombinePanel`.
pub struct InitialCombinePanel {
    pub pnl_root: InitialCombinePanelLayout,
    pub dialog_type: DialogType,
    pub matchvol1_header: PanelHeader,
    pub btn_matchvol_restart: MultiLineButton,
    pub ltf_output_size_y: LabeledTextField,
    pub pnl_solvematch: SolvematchPanel,
    pub match_mode: Option<MatchMode>,
}

impl InitialCombinePanel {
    /// Java private constructor `InitialCombinePanel(...)`.
    pub fn new(dialog_type: DialogType) -> Self {
        Self {
            pnl_root: InitialCombinePanelLayout {
                root_visible: true,
                solvematch_present: true,
                matchvol1_present: true,
                matchvol1_body_visible: true,
                output_size_y_info_visible: true,
                component_order: vec!["solvematch", "matchvol1"],
                ..Default::default()
            },
            dialog_type,
            matchvol1_header: PanelHeader::new(
                "Matchvol1",
                true,
                true,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            btn_matchvol_restart: MultiLineButton::new_with_label(Some("Restart at Matchvol1")),
            ltf_output_size_y: LabeledTextField::new(FieldType::Integer, "Initial match size: "),
            pnl_solvematch: SolvematchPanel::get_instance(
                "Initial",
                "COMBINE_INITIAL_SOLVEMATCH_HEADER_GROUP",
                dialog_type,
                false,
            ),
            match_mode: None,
        }
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(dialog_type: DialogType) -> Self {
        let mut instance = Self::new(dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `createPanel()`.
    pub fn create_panel(&mut self) {
        self.pnl_root.matchvol1_body_visible = true;
    }

    /// Java private `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.pnl_root.listener_count = 2;
    }

    /// Java `removeListeners()`.
    pub fn remove_listeners(&mut self) {
        self.pnl_root.listener_count = 1;
    }

    /// Java `setDeferred3dmodButtons()`; actual launch ownership is a UI boundary.
    pub fn set_deferred_3dmod_buttons(&mut self, parent_button: Option<&MultiLineButton>) {
        if parent_button.is_some() {}
        self.pnl_solvematch
            .set_deferred_3dmod_buttons(parent_button);
    }

    /// Java `setMatchMode`, with `MRCHeader` I/O supplied by its direct boundary.
    pub fn set_match_mode<H: InitialCombineMrcHeaders>(
        &mut self,
        match_mode: MatchMode,
        headers: &mut H,
    ) {
        if self.match_mode == Some(match_mode) {
            return;
        }
        self.match_mode = Some(match_mode);
        let (to_axis, from_axis) = match match_mode {
            MatchMode::AToB => (AxisID::Second, AxisID::First),
            MatchMode::BToA => (AxisID::First, AxisID::Second),
        };
        let Some(to_y) = headers.tilt_output_n_rows(to_axis).ok().flatten() else {
            return;
        };
        let Some(from_y) = headers.tilt_output_n_rows(from_axis).ok().flatten() else {
            return;
        };
        self.pnl_root.output_size_y_info = format!(
            "Original {} size is {}.  Final size will be {}",
            from_axis.get_upper_case_extension(),
            from_y,
            to_y
        );
    }

    /// Java `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        ProcessingMethod::LocalCpu
    }
    /// Java `getMatchMode()`: initial tab never modifies it.
    pub fn get_match_mode(&self) -> Option<MatchMode> {
        None
    }
    pub fn is_use_corresponding_points(&self) -> bool {
        self.pnl_solvematch.is_use_corresponding_points()
    }
    pub fn set_use_corresponding_points(&mut self, use_points: bool) {
        self.pnl_solvematch.set_use_corresponding_points(use_points);
    }
    pub fn update_advanced(&mut self, state: bool) {
        self.update_matchvol1_advanced(state);
    }
    pub fn update_display(&mut self) {
        self.pnl_solvematch.update_display();
    }
    pub fn update_matchvol1_advanced(&mut self, advanced: bool) {
        self.ltf_output_size_y.set_visible(advanced);
        self.pnl_root.output_size_y_info_visible = advanced;
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_solvematch.set_visible(visible);
    }
    /// Java `getContainer()`.
    pub fn get_container(&self) -> &InitialCombinePanelLayout {
        &self.pnl_root
    }
    pub fn is_enabled<P: InitialCombinePanelParent>(&self, parent: &P) -> bool {
        parent.initial_tab_enabled()
    }
    pub fn is_initial_volume_matching(&self) -> bool {
        self.pnl_solvematch.is_initial_volume_matching()
    }
    pub fn set_initial_volume_matching(&mut self, input: bool) {
        self.pnl_solvematch.set_initial_volume_matching(input);
    }

    /// Java `getParameters(MatchvolParam,boolean)`.
    pub fn get_matchvol_parameters<P: MatchvolParameters>(
        &self,
        param: &mut P,
        validate: bool,
    ) -> bool {
        match self.ltf_output_size_y.get_text_validated(validate) {
            Ok(value) => {
                param.set_output_size_y(value);
                true
            }
            Err(_) => false,
        }
    }
    pub fn set_matchvol_parameters<P: MatchvolParameters>(&mut self, param: &P) {
        self.ltf_output_size_y.set_text(&param.output_size_y());
    }
    pub fn expand_global(&mut self, advanced: bool) {
        self.update_advanced(advanced);
    }
    /// Java `expand(ExpandButton)`.
    pub fn expand(&mut self, button: &ExpandButton) {
        if self.matchvol1_header.equals_open_close(button) {
            self.pnl_root.matchvol1_body_visible = button.is_expanded();
        } else if self.matchvol1_header.equals_advanced_basic(button) {
            self.update_matchvol1_advanced(button.is_expanded());
        }
    }
    pub fn set_solvematch_parameters<P: SolvematchParameters>(&mut self, param: &P) {
        self.pnl_solvematch.set_solvematch_parameters(param);
    }
    pub fn set_dualvolmatch_parameters<P: DualvolmatchParameters>(&mut self, param: &P) {
        self.pnl_solvematch.set_dualvolmatch_parameters(param);
    }
    pub fn get_solvematch_parameters<P: SolvematchParameters>(
        &self,
        param: &mut P,
        validate: bool,
    ) -> bool {
        self.pnl_solvematch
            .get_solvematch_parameters(param, validate)
    }
    pub fn get_dualvolmatch_parameters<P: DualvolmatchParameters>(
        &self,
        param: &mut P,
        validate: bool,
    ) -> bool {
        self.pnl_solvematch
            .get_dualvolmatch_parameters(param, validate)
    }
    pub fn get_screen_state<S: InitialCombinePanelScreenState>(&self, state: &mut S) {
        self.pnl_solvematch.get_screen_state(state);
        state.set_matchvol1_restart_button_state(self.btn_matchvol_restart.get_button_state());
    }
    pub fn set_screen_state<S: InitialCombinePanelScreenState>(&mut self, state: &S) {
        self.pnl_solvematch.set_screen_state(state);
        self.btn_matchvol_restart
            .set_button_state(state.matchvol1_restart_button_state());
    }
    pub fn get_surfaces_or_models(&self) -> FiducialMatch {
        self.pnl_solvematch.get_surfaces_or_models()
    }
    pub fn set_surfaces_or_models(&mut self, value: FiducialMatch) {
        self.pnl_solvematch.set_surfaces_or_models(value);
    }
    pub fn is_bin_by_2(&self) -> bool {
        self.pnl_solvematch.is_bin_by_2()
    }
    pub fn set_bin_by_2(&mut self, value: bool) {
        self.pnl_solvematch.set_bin_by_2(value);
    }
    pub fn set_use_list(&mut self, value: &str) {
        self.pnl_solvematch.set_use_list(value);
    }
    pub fn get_use_list(&self, validate: bool) -> Result<String, FieldValidationFailedException> {
        self.pnl_solvematch.get_use_list(validate)
    }
    pub fn set_fiducial_match_list_a(&mut self, value: &str) {
        self.pnl_solvematch.set_fiducial_match_list_a(value);
    }
    pub fn get_fiducial_match_list_a(
        &self,
        validate: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.pnl_solvematch.get_fiducial_match_list_a(validate)
    }
    pub fn set_fiducial_match_list_b(&mut self, value: &str) {
        self.pnl_solvematch.set_fiducial_match_list_b(value);
    }
    pub fn get_fiducial_match_list_b(
        &self,
        validate: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.pnl_solvematch.get_fiducial_match_list_b(validate)
    }

    /// Java `popUpContextMenu(MouseEvent)`; presentation of this request is a
    /// concrete frontend boundary.
    pub fn pop_up_context_menu(&self) -> InitialCombineContextPopup {
        InitialCombineContextPopup {
            title: "Initial Problems in Combining",
            manual_page_labels: ["Solvematch", "Matchshifts"],
            manual_pages: ["solvematch.html", "matchshifts.html"],
            log_file_labels: ["Transferfid", "Solvematch"],
            log_files: ["transferfid.log", "solvematch.log"],
        }
    }

    /// Java `action(String,Deferred3dmodButton,Run3dmodMenuOptions)`.
    pub fn action<M: InitialCombinePanelApplicationManager, P: InitialCombinePanelParent>(
        &mut self,
        manager: &mut M,
        parent: &mut P,
        command: &str,
        deferred: Option<&MultiLineButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        parent.synchronize_initial();
        if Some(command) == self.btn_matchvol_restart.get_action_command() {
            manager.matchvol1_combine(
                &self.btn_matchvol_restart,
                deferred,
                options,
                self.dialog_type,
                parent.run_processing_method(),
                parent.is_parallel(),
                !parent.is_run_volcombine(),
            );
        }
    }
    /// Java private `setToolTipText()`.
    pub fn set_tool_tip_text(&mut self) {
        let text = "Thickness to make initial matching volume, which may need to be thicker than the final matching volume to contain all the material needed for patch correlations.";
        self.ltf_output_size_y.set_tool_tip_text(Some(text));
        self.btn_matchvol_restart.set_tool_tip_text(Some(
            "Resume and make first matching volume, despite a small displacement between the match check volumes",
        ));
        self.pnl_root.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Headers {
        rows: [Option<i32>; 2],
    }
    impl InitialCombineMrcHeaders for Headers {
        fn tilt_output_n_rows(&mut self, axis: AxisID) -> Result<Option<i32>, String> {
            Ok(self.rows[if axis == AxisID::First { 0 } else { 1 }])
        }
    }
    #[test]
    fn match_mode_reads_b_to_a_headers_and_formats_info() {
        let mut panel = InitialCombinePanel::get_instance(DialogType::TomogramCombination);
        let mut headers = Headers {
            rows: [Some(200), Some(300)],
        };
        panel.set_match_mode(MatchMode::BToA, &mut headers);
        assert_eq!(
            panel.pnl_root.output_size_y_info,
            "Original B size is 300.  Final size will be 200"
        );
        assert_eq!(panel.get_match_mode(), None);
    }
    #[test]
    fn expansion_controls_matchvol_body_and_advanced_fields() {
        let mut panel = InitialCombinePanel::get_instance(DialogType::TomogramCombination);
        panel.expand_global(false);
        assert!(!panel.ltf_output_size_y.visible);
        panel.update_matchvol1_advanced(true);
        assert!(panel.pnl_root.output_size_y_info_visible);
    }
    #[test]
    fn solvematch_passthrough_keeps_initial_volume_matching() {
        let mut panel = InitialCombinePanel::get_instance(DialogType::TomogramCombination);
        panel.set_initial_volume_matching(true);
        assert!(panel.is_initial_volume_matching());
        panel.set_use_list("1,3");
        assert_eq!(panel.get_use_list(false).unwrap(), "1,3");
    }
    #[test]
    fn context_popup_retains_the_source_manual_and_log_targets() {
        let panel = InitialCombinePanel::get_instance(DialogType::TomogramCombination);
        assert_eq!(panel.pop_up_context_menu().log_files[1], "solvematch.log");
        assert!(panel.pnl_root.tooltip_initialized);
    }
}
