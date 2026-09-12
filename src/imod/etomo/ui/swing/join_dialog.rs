//! `IMOD/Etomo/src/etomo/ui/swing/JoinDialog.java`.
//!
//! `JoinDialog` is the five-tab serial-section join dialog.  Swing construction,
//! file chooser presentation, autodoc lookup, and the still-untranslated
//! `JoinManager` process graph are explicit boundaries.  The source-owned dialog
//! state, tab/mode policy, metadata transfer, spinner ranges, and action routing
//! live here rather than in a replacement UI abstraction.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::section_table_panel::{
    JoinMetaData as SectionJoinMetaData, SectionTablePanel, Tab as SectionTab,
};

pub const SETUP_MODE: i32 = -1;
pub const SAMPLE_NOT_PRODUCED_MODE: i32 = -2;
pub const SAMPLE_PRODUCED_MODE: i32 = -3;
pub const CHANGING_SAMPLE_MODE: i32 = -4;
pub const FINISH_JOIN_TEXT: &str = "Finish Join";
pub const WORKING_DIRECTORY_TEXT: &str = "Working directory";
pub const GET_MAX_SIZE_TEXT: &str = "Get Max Size and Shift";
pub const TRIAL_JOIN_TEXT: &str = "Trial Join";
pub const REJOIN_TEXT: &str = "Rejoin";
pub const TRIAL_REJOIN_TEXT: &str = "Trial Rejoin";
const REFINE_JOIN_TEXT: &str = "Refine Join";
const OPEN_IN_3DMOD: &str = "Open in 3dmod";

/// Java private static final inner `Tab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Setup,
    Align,
    Join,
    Model,
    Rejoin,
}
impl Tab {
    pub fn get_index(self) -> usize {
        match self {
            Self::Setup => 0,
            Self::Align => 1,
            Self::Join => 2,
            Self::Model => 3,
            Self::Rejoin => 4,
        }
    }
    pub fn get_instance(index: usize) -> Self {
        match index {
            0 => Self::Setup,
            1 => Self::Align,
            2 => Self::Join,
            3 => Self::Model,
            4 => Self::Rejoin,
            _ => Self::Setup,
        }
    }
    fn section_tab(self) -> SectionTab {
        match self {
            Self::Setup => SectionTab::Setup,
            Self::Align => SectionTab::Align,
            Self::Join => SectionTab::Join,
            Self::Model => SectionTab::Model,
            Self::Rejoin => SectionTab::Rejoin,
        }
    }
}

/// Source fields transferred by `getMetaData`/`setMetaData`, apart from the
/// canonical `SectionTablePanel` rows which remain in `section_table`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JoinDialogMetaData {
    pub dataset_name: String,
    pub density_ref_section: i32,
    pub use_alignment_ref_section: bool,
    pub alignment_ref_section: i32,
    pub size_in_x: String,
    pub size_in_y: String,
    pub shift_in_x: String,
    pub shift_in_y: String,
    pub local_fits: bool,
    pub use_every_n_slices: i32,
    pub rejoin_use_every_n_slices: i32,
    pub trial_binning: i32,
    pub midas_limit: String,
    pub model_transform: String,
    pub boundaries_to_analyze: String,
    pub objects_to_include: String,
    pub gap: bool,
    pub gap_start: String,
    pub gap_end: String,
    pub gap_inc: String,
    pub points_to_fit_min: String,
    pub points_to_fit_max: String,
    pub rejoin_trial_binning: i32,
    pub section_table: SectionJoinMetaData,
}

/// Direct `JoinState` reads/writes in this source file.
pub trait JoinDialogState {
    fn is_refine_start_list_empty(&self) -> bool;
    fn join_trial_use_every_n_slices(&self) -> Option<i32>;
    fn refine_trial_use_every_n_slices(&self) -> Option<i32>;
    fn is_sample_produced(&self) -> bool;
    fn set_revert_state(&mut self, value: bool);
    fn revert(&mut self);
}

/// Direct `JoinManager` action calls.  Its concrete source unit remains the
/// boundary; this trait intentionally retains each Java action independently.
pub trait JoinDialogManager {
    fn make_join_com(&mut self);
    fn finish_join(&mut self, mode: JoinFinishMode, label: &str);
    fn start_refine(&mut self);
    fn xfjointomo(&mut self);
    fn xfmodel(&mut self, input: &str, output: &str);
    fn imod_open(&mut self, key: &str, binning: i32, model: Option<&str>);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JoinFinishMode {
    FinishJoin,
    MaxSize,
    Trial,
    Rejoin,
    TrialRejoin,
    SuppressExecution,
}

/// Java `XfjointomoParam` direct calls.
pub trait XfjointomoParam {
    fn set_transform(&mut self, value: String);
    fn set_boundaries_to_analyze(&mut self, value: String);
    fn set_objects_to_include(&mut self, value: String);
    fn set_points_to_fit(&mut self, min: String, max: String);
    fn set_gap_start_end_inc(&mut self, start: String, end: String, increment: String);
}
/// Java `Joinwarp2modelParam` direct calls.
pub trait Joinwarp2modelParam {
    fn set_binning_of_join(&mut self, value: String);
    fn set_offset_in_x_and_y(&mut self, value: String);
    fn set_chunk_sizes(&mut self, value: String);
}

/// The source dialog. `*_enabled`, `tabs_enabled`, and `mounted_tab` are the
/// corresponding native Swing widget state at the GUI boundary.
#[derive(Debug)]
pub struct JoinDialog {
    pub root_name: String,
    pub working_dir: String,
    pub model_file: String,
    pub transformed_model: String,
    pub size_in_x: String,
    pub size_in_y: String,
    pub shift_in_x: String,
    pub shift_in_y: String,
    pub midas_limit: String,
    pub boundaries_to_analyze: String,
    pub objects_to_include: String,
    pub gap_start: String,
    pub gap_end: String,
    pub gap_inc: String,
    pub points_to_fit_min: String,
    pub points_to_fit_max: String,
    pub density_ref_section: i32,
    pub alignment_ref_section: i32,
    pub trial_binning: i32,
    pub rejoin_trial_binning: i32,
    pub use_every_n_slices: i32,
    pub rejoin_use_every_n_slices: i32,
    pub local_fits: bool,
    pub use_alignment_ref_section: bool,
    pub gap: bool,
    pub refine_with_trial: bool,
    pub refining_join: bool,
    pub num_sections: usize,
    pub cur_tab: Tab,
    pub invalid_reason: Option<String>,
    pub section_table: SectionTablePanel,
    pub default_x_size: i32,
    pub default_y_size: i32,
    pub root_name_editable: bool,
    pub working_dir_editable: bool,
    pub tabs_enabled: [bool; 5],
    pub midas_limit_enabled: bool,
    pub density_ref_section_enabled: bool,
    pub change_setup_enabled: bool,
    pub revert_to_last_setup_enabled: bool,
    pub make_samples_enabled: bool,
    pub gap_fields_enabled: bool,
    pub transform_model_enabled: bool,
    pub refine_with_trial_enabled: bool,
    pub refine_join_enabled: bool,
    pub local_fits_enabled: bool,
    pub alignment_ref_section_enabled: bool,
    pub mounted_tab: Option<Tab>,
    pub action_listeners_installed: bool,
    /// Source Swing insertion order for each of the five tab panels.  Actual
    /// `JPanel`/BoxLayout creation remains the native GUI adapter boundary.
    pub tab_components: [Vec<String>; 5],
    pub model_panel_created: bool,
    pub rejoin_panel_created: bool,
    /// Java `setToolTipText` assignments keyed by the source field name.
    pub tooltips: Vec<(String, String)>,
    pub messages: Vec<(String, String)>,
    pub last_action: Option<String>,
}

impl JoinDialog {
    /// Java private constructor plus both `getInstance` overloads' shared construction.
    pub fn new(
        working_dir_name: Option<&str>,
        metadata: &JoinDialogMetaData,
        refining_join: bool,
    ) -> Self {
        let mut dialog = Self {
            root_name: String::new(),
            working_dir: working_dir_name.unwrap_or_default().into(),
            model_file: String::new(),
            transformed_model: String::new(),
            size_in_x: String::new(),
            size_in_y: String::new(),
            shift_in_x: String::new(),
            shift_in_y: String::new(),
            midas_limit: String::new(),
            boundaries_to_analyze: String::new(),
            objects_to_include: String::new(),
            gap_start: String::new(),
            gap_end: String::new(),
            gap_inc: String::new(),
            points_to_fit_min: String::new(),
            points_to_fit_max: String::new(),
            density_ref_section: 1,
            alignment_ref_section: 1,
            trial_binning: 1,
            rejoin_trial_binning: 1,
            use_every_n_slices: 1,
            rejoin_use_every_n_slices: 1,
            local_fits: false,
            use_alignment_ref_section: false,
            gap: false,
            refine_with_trial: false,
            refining_join,
            num_sections: 0,
            cur_tab: Tab::Setup,
            invalid_reason: None,
            section_table: SectionTablePanel::new(SectionTab::Setup, 0),
            default_x_size: 0,
            default_y_size: 0,
            root_name_editable: true,
            working_dir_editable: true,
            tabs_enabled: [true; 5],
            midas_limit_enabled: true,
            density_ref_section_enabled: true,
            change_setup_enabled: false,
            revert_to_last_setup_enabled: false,
            make_samples_enabled: true,
            gap_fields_enabled: false,
            transform_model_enabled: false,
            refine_with_trial_enabled: false,
            refine_join_enabled: false,
            local_fits_enabled: true,
            alignment_ref_section_enabled: true,
            mounted_tab: None,
            action_listeners_installed: false,
            tab_components: std::array::from_fn(|_| Vec::new()),
            model_panel_created: false,
            rejoin_panel_created: false,
            tooltips: Vec::new(),
            messages: Vec::new(),
            last_action: None,
        };
        dialog.create_root_panel();
        dialog.set_meta_data(metadata);
        dialog.init();
        dialog.set_tool_tip_text();
        dialog
    }
    /// Java `getInstance(JoinManager, ConstJoinMetaData, JoinState)`.
    pub fn get_instance(metadata: &JoinDialogMetaData, refining_join: bool) -> Self {
        let mut value = Self::new(None, metadata, refining_join);
        value.add_listeners();
        value
    }
    /// Java `getInstance(JoinManager, String, ConstJoinMetaData, JoinState)`.
    pub fn get_instance_with_working_dir(
        working_dir: &str,
        metadata: &JoinDialogMetaData,
        refining_join: bool,
    ) -> Self {
        let mut value = Self::new(Some(working_dir), metadata, refining_join);
        value.add_listeners();
        value
    }
    pub fn get_focus_component(&self) -> Option<Tab> {
        Some(self.cur_tab)
    }
    /// Java `setAutoAlignmentController`; controller ownership remains the
    /// `AutoAlignmentPanel`/manager boundary.
    pub fn set_auto_alignment_controller(&mut self) {
        self.last_action = Some("autoAlignmentPanel.setController".into());
    }
    pub fn get_axis_id(&self) -> &'static str {
        "ONLY"
    }
    pub fn get_dialog_type(&self) -> &'static str {
        "JOIN"
    }
    pub fn param_string(&self) -> String {
        format!(
            "ltfRootName={},ltfSizeInX={},\nltfSizeInY={},\nltfShiftInX={},\nltfShiftInY={},\nltfMidasLimit={},\nspinDensityRefSection={},\nspinTrialBinning={},\nspinUseEveryNSlices={},\nnumSections{},curTab={:?},invalidReason{:?},\naxisID=ONLY",
            self.root_name,
            self.size_in_x,
            self.size_in_y,
            self.shift_in_x,
            self.shift_in_y,
            self.midas_limit,
            self.density_ref_section,
            self.trial_binning,
            self.use_every_n_slices,
            self.num_sections,
            self.cur_tab,
            self.invalid_reason
        )
    }
    pub fn get_model_tab_j_component(&self) -> Tab {
        Tab::Model
    }
    pub fn get_rejoin_tab_j_component(&self) -> Tab {
        Tab::Rejoin
    }
    pub fn get_setup_tab_j_component(&self) -> Tab {
        Tab::Setup
    }
    pub fn get_align_tab_j_component(&self) -> Tab {
        Tab::Align
    }
    pub fn get_join_tab_j_component(&self) -> Tab {
        Tab::Join
    }
    fn create_root_panel(&mut self) {
        self.create_tab_pane();
    }
    fn add_listeners(&mut self) {
        self.action_listeners_installed = true;
    }
    fn create_tab_pane(&mut self) {
        self.create_setup_panel();
        self.create_align_panel();
        self.create_join_panel();
        self.create_model_panel();
        self.create_rejoin_panel();
        self.add_panel_components(Tab::Setup);
        self.update_display(false, false, None);
    }
    pub fn update_display(
        &mut self,
        join_file_exists: bool,
        trial_join_file_exists: bool,
        state: Option<&dyn JoinDialogState>,
    ) {
        self.tabs_enabled[Tab::Model.get_index()] = self.refining_join;
        self.tabs_enabled[Tab::Rejoin.get_index()] = self.refining_join;
        self.gap_fields_enabled = self.gap;
        self.transform_model_enabled =
            state.is_some_and(|state| !state.is_refine_start_list_empty());
        let refinable = trial_join_file_exists
            && state.and_then(JoinDialogState::join_trial_use_every_n_slices) == Some(1);
        self.refine_with_trial_enabled = refinable;
        if !refinable {
            self.refine_with_trial = false;
        }
        self.refine_join_enabled = join_file_exists || refinable;
        self.local_fits_enabled = !self.use_alignment_ref_section;
        self.alignment_ref_section_enabled = !self.local_fits || !self.local_fits_enabled;
    }
    fn add_panel_components(&mut self, tab: Tab) {
        match tab {
            Tab::Setup => self.add_setup_panel_components(),
            Tab::Align => self.add_align_panel_components(),
            Tab::Join => self.add_join_panel_components(),
            Tab::Model => self.add_model_panel_components(),
            Tab::Rejoin => self.add_rejoin_panel_components(),
        }
        self.mounted_tab = Some(tab);
        self.section_table.current_tab = tab.section_tab();
        self.section_table.display_cur_tab();
    }
    /// Java private `setSizeAndShift(Vector)`.  `coordinate` is the direct
    /// `ConstJoinMetaData.getCoordinate(EtomoNumber, JoinState)` boundary and
    /// `new_shift` is the corresponding `JoinState` calculation.  The parser
    /// deliberately preserves the source's scan for the last rubber-band tag.
    pub fn set_size_and_shift(
        &mut self,
        coordinates: Option<&[String]>,
        coordinate: impl Fn(&str) -> i32,
        new_shift: impl Fn(i32, i32, bool) -> i32,
    ) -> Result<(), String> {
        let Some(coordinates) = coordinates else {
            return Ok(());
        };
        if coordinates.is_empty() {
            return Ok(());
        }
        let mut x = None;
        let mut y = None;
        let mut index = 0;
        while index < coordinates.len() {
            if coordinates[index] == "Rubberband" {
                if index + 4 >= coordinates.len() {
                    break;
                }
                x = Some((
                    coordinates[index + 1].clone(),
                    coordinates[index + 3].clone(),
                ));
                y = Some((
                    coordinates[index + 2].clone(),
                    coordinates[index + 4].clone(),
                ));
                index += 5;
            } else {
                index += 1;
            }
        }
        if let Some((minimum, maximum)) = x {
            let minimum = coordinate(&minimum);
            let maximum = coordinate(&maximum);
            self.size_in_x = (maximum - minimum + 1).to_string();
            self.shift_in_x = new_shift(minimum, maximum, true).to_string();
        }
        if let Some((minimum, maximum)) = y {
            let minimum = coordinate(&minimum);
            let maximum = coordinate(&maximum);
            self.size_in_y = (maximum - minimum + 1).to_string();
            self.shift_in_y = new_shift(minimum, maximum, false).to_string();
        }
        Ok(())
    }
    fn remove_panel_components(&mut self, _tab: Tab) {
        self.mounted_tab = None;
    }
    pub fn is_setup_tab(&self) -> bool {
        self.cur_tab == Tab::Setup
    }
    pub fn is_align_tab(&self) -> bool {
        self.cur_tab == Tab::Align
    }
    pub fn is_join_tab(&self) -> bool {
        self.cur_tab == Tab::Join
    }
    pub fn is_model_tab(&self) -> bool {
        self.cur_tab == Tab::Model
    }
    pub fn is_rejoin_tab(&self) -> bool {
        self.cur_tab == Tab::Rejoin
    }
    pub fn get_tab(&self) -> Tab {
        self.cur_tab
    }
    pub fn get_section_table_size(&self) -> usize {
        self.section_table.size()
    }
    pub fn change_tab(&mut self, selected_index: usize) {
        let previous = self.cur_tab;
        self.remove_panel_components(previous);
        self.cur_tab = Tab::get_instance(selected_index);
        self.synchronize(Some(previous));
        self.add_panel_components(self.cur_tab);
    }
    pub fn set_inverted(&mut self, inverted: &[bool]) {
        self.section_table.set_inverted(inverted);
    }
    pub fn set_mode<S: JoinDialogState>(&mut self, mode: i32, state: &mut S) {
        self.root_name_editable = mode == SETUP_MODE;
        self.working_dir_editable = mode == SETUP_MODE;
        match mode {
            SETUP_MODE | SAMPLE_NOT_PRODUCED_MODE => {
                self.tabs_enabled[1] = false;
                self.tabs_enabled[2] = false;
                self.midas_limit_enabled = true;
                self.density_ref_section_enabled = true;
                self.change_setup_enabled = false;
                self.set_revert_state(false, state);
                self.make_samples_enabled = true;
            }
            SAMPLE_PRODUCED_MODE => {
                self.tabs_enabled[1] = true;
                self.tabs_enabled[2] = true;
                self.midas_limit_enabled = false;
                self.density_ref_section_enabled = false;
                self.change_setup_enabled = true;
                self.set_revert_state(false, state);
                self.make_samples_enabled = false;
            }
            CHANGING_SAMPLE_MODE => {
                self.tabs_enabled[1] = false;
                self.tabs_enabled[2] = false;
                self.midas_limit_enabled = true;
                self.density_ref_section_enabled = true;
                self.change_setup_enabled = false;
                self.set_revert_state(true, state);
                self.make_samples_enabled = true;
            }
            _ => panic!("mode={mode}"),
        };
        // `SectionTablePanel` retains its own source constants rather than the
        // dialog's public mode constants.  Java passes the semantic mode through;
        // preserve that correspondence at the already-translated unit boundary.
        self.section_table.set_mode_to(match mode {
            SETUP_MODE => super::section_table_panel::SETUP_MODE,
            SAMPLE_NOT_PRODUCED_MODE => super::section_table_panel::SAMPLE_NOT_PRODUCED_MODE,
            SAMPLE_PRODUCED_MODE => super::section_table_panel::SAMPLE_PRODUCED_MODE,
            CHANGING_SAMPLE_MODE => super::section_table_panel::CHANGING_SAMPLE_MODE,
            _ => unreachable!(),
        });
    }
    fn set_revert_state<S: JoinDialogState>(&mut self, enabled: bool, state: &mut S) {
        self.revert_to_last_setup_enabled = enabled;
        state.set_revert_state(enabled);
    }
    /// Java `createSetupPanel`: these are constructed before the tab is first
    /// mounted; `addSetupPanelComponents` supplies the source insertion order.
    fn create_setup_panel(&mut self) {
        self.tab_components[Tab::Setup.get_index()].clear();
        self.working_dir_editable = true;
    }
    /// Java `addSetupPanelComponents`.
    fn add_setup_panel_components(&mut self) {
        self.tab_components[Tab::Setup.get_index()] = vec![
            "ftfWorkingDir".into(),
            "ltfRootName".into(),
            "pnlSectionTable.rootPanel".into(),
            "pnlMidasLimit(ltfMidasLimit,lblMidasLimit)".into(),
            "spinDensityRefSection".into(),
            "setupPanel2(btnChangeSetup,btnRevertToLastSetup)".into(),
            "btnMakeSamples".into(),
        ];
    }
    /// Java `createAlignPanel`.
    fn create_align_panel(&mut self) {
        self.tab_components[Tab::Align.get_index()].clear();
    }
    /// Java `addAlignPanelComponents`.
    fn add_align_panel_components(&mut self) {
        self.tab_components[Tab::Align.get_index()] = vec![
            "pnlSectionTable.rootPanel".into(),
            "alignPanel1(btnOpenSample,btnOpenSampleAverages)".into(),
            "autoAlignmentPanel.rootComponent".into(),
        ];
    }
    /// Java lazy `createModelPanel`.
    fn create_model_panel(&mut self) {
        if self.model_panel_created {
            return;
        }
        self.model_panel_created = true;
    }
    /// Java `addModelPanelComponents`.
    fn add_model_panel_components(&mut self) {
        self.create_model_panel();
        self.tab_components[Tab::Model.get_index()] = vec![
            "btnMakeRefiningModel".into(), "boundaryTable.container".into(),
            "pnlTransformations(tcModel,ltfBoundariesToAnalyze,ltfObjectsToInclude,cbGap,ltfGapStart,ltfGapEnd,ltfGapInc,ltfPointsToFitMin,ltfPointsToFitMax,btnXfjointomo,btnTransformAndViewModel)".into(),
        ];
    }
    /// Java lazy `createRejoinPanel`.
    fn create_rejoin_panel(&mut self) {
        if self.rejoin_panel_created {
            return;
        }
        self.rejoin_panel_created = true;
    }
    /// Java `addRejoinPanelComponents`.
    fn add_rejoin_panel_components(&mut self) {
        self.create_rejoin_panel();
        self.tab_components[Tab::Rejoin.get_index()] = vec![
            "pnlTables(pnlSectionTable.container,boundaryTable.container)".into(),
            "pnlRejoin(trialRejoin,useEvery,rejoin,transformModel)".into(),
        ];
    }
    /// Java `createJoinPanel`.
    fn create_join_panel(&mut self) {
        self.create_finish_join_panel();
    }
    /// Java `addJoinPanelComponents`.
    fn add_join_panel_components(&mut self) {
        self.tab_components[Tab::Join.get_index()] =
            vec!["pnlSectionTable.rootPanel".into(), "pnlFinishJoin".into()];
    }
    /// Java `createFinishJoinPanel`.
    fn create_finish_join_panel(&mut self) {
        self.create_trial_join_panel();
    }
    /// Java `createTrialJoinPanel`; its controls retain their initial source
    /// values in the dialog's spinner fields.
    fn create_trial_join_panel(&mut self) {
        let default = self.section_table.get_z_max().clamp(1, 10);
        self.use_every_n_slices = default;
    }
    pub fn msg_row_change(&mut self) {
        self.last_action = Some("boundaryTable.msgRowChange".into());
    }
    pub fn set_num_sections(&mut self, num_sections: usize, init: bool) {
        self.num_sections = num_sections;
        let z_max = self.section_table.get_z_max();
        let maximum = num_sections.max(1) as i32;
        self.density_ref_section = self.density_ref_section.clamp(1, maximum);
        self.alignment_ref_section = self.alignment_ref_section.clamp(1, maximum);
        let default = if z_max < 1 { 1 } else { z_max.min(10) };
        if z_max == 0 {
            self.use_every_n_slices = 1;
        } else {
            self.use_every_n_slices = self.use_every_n_slices.min(z_max).max(1);
        }
        if init && z_max > 0 && self.use_every_n_slices == 1 {
            self.use_every_n_slices = default;
        }
        self.rejoin_use_every_n_slices = self.rejoin_use_every_n_slices.min(z_max.max(1)).max(1);
        if init && z_max > 0 && self.rejoin_use_every_n_slices == 1 {
            self.rejoin_use_every_n_slices = default;
        }
        self.default_size_in_xy();
    }
    fn init(&mut self) {
        self.default_x_size = self.section_table.get_x_max();
        self.default_y_size = self.section_table.get_y_max();
    }
    pub fn default_size_in_xy(&mut self) {
        let x_max = self.section_table.get_x_max();
        let y_max = self.section_table.get_y_max();
        if x_max == self.default_x_size && y_max == self.default_y_size {
            return;
        }
        self.default_x_size = x_max;
        self.default_y_size = y_max;
        self.size_in_x = x_max.to_string();
        self.size_in_y = y_max.to_string();
    }
    pub fn set_size_in_x(&mut self, value: impl ToString) {
        self.size_in_x = value.to_string();
    }
    pub fn set_size_in_y(&mut self, value: impl ToString) {
        self.size_in_y = value.to_string();
    }
    pub fn set_shift_in_x(&mut self, value: i32) {
        self.shift_in_x = value.to_string();
    }
    pub fn set_shift_in_y(&mut self, value: i32) {
        self.shift_in_y = value.to_string();
    }
    pub fn get_invalid_reason(&self) -> Option<String> {
        self.invalid_reason
            .clone()
            .or_else(|| self.section_table.get_invalid_reason())
    }
    pub fn get_mode(&self) -> i32 {
        self.section_table.get_mode()
    }
    pub fn get_parameters_xfjointomo<P: XfjointomoParam>(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        if do_validation
            && [
                &self.boundaries_to_analyze,
                &self.objects_to_include,
                &self.points_to_fit_min,
                &self.points_to_fit_max,
            ]
            .iter()
            .any(|text| text.contains('\n'))
        {
            return false;
        }
        param.set_transform("search".into());
        param.set_boundaries_to_analyze(self.boundaries_to_analyze.clone());
        param.set_objects_to_include(self.objects_to_include.clone());
        param.set_points_to_fit(
            self.points_to_fit_min.clone(),
            self.points_to_fit_max.clone(),
        );
        if self.gap {
            param.set_gap_start_end_inc(
                self.gap_start.clone(),
                self.gap_end.clone(),
                self.gap_inc.clone(),
            );
        }
        true
    }
    pub fn get_parameters_joinwarp2model<P: Joinwarp2modelParam>(
        &self,
        param: &mut P,
        _do_validation: bool,
    ) -> bool {
        let (Ok(x), Ok(y)) = (
            self.shift_in_x.parse::<i32>(),
            self.shift_in_y.parse::<i32>(),
        ) else {
            return false;
        };
        param.set_binning_of_join(self.trial_binning.to_string());
        param.set_offset_in_x_and_y(format!("{}, {}", -x, -y));
        param.set_chunk_sizes(self.section_table.get_chunk_sizes());
        true
    }
    /// Java `getAutoAlignmentParameters(MidasParam)`.  `MidasParam` is owned
    /// by JoinManager's comscript graph, so the precise forwarding boundary is
    /// represented by this source-named call until that graph is translated.
    pub fn get_auto_alignment_parameters_midas(&mut self) {
        self.last_action = Some("manager.getParameters(MidasParam,ONLY)".into());
    }
    /// Java `getAutoAlignmentParameters(XfalignParam,boolean)`.
    pub fn get_auto_alignment_parameters_xfalign(&mut self, _do_validation: bool) -> bool {
        self.last_action = Some("manager.getParameters(XfalignParam,ONLY)".into());
        true
    }
    /// Java `setXfjointomoResult`; log parsing is retained by `BoundaryTable`.
    pub fn set_xfjointomo_result(&mut self) {
        self.last_action = Some("boundaryTable.setXfjointomoResult".into());
    }
    pub fn get_meta_data(
        &mut self,
        metadata: &mut JoinDialogMetaData,
        do_validation: bool,
    ) -> bool {
        self.synchronize(None);
        if do_validation && self.root_name.contains('\n') {
            return false;
        }
        metadata.dataset_name = self.root_name.clone();
        metadata.density_ref_section = self.density_ref_section;
        metadata.use_alignment_ref_section = self.use_alignment_ref_section;
        metadata.alignment_ref_section = self.alignment_ref_section;
        metadata.size_in_x = self.size_in_x.clone();
        metadata.size_in_y = self.size_in_y.clone();
        metadata.shift_in_x = self.shift_in_x.clone();
        metadata.shift_in_y = self.shift_in_y.clone();
        metadata.local_fits = self.local_fits;
        metadata.use_every_n_slices = self.use_every_n_slices;
        metadata.rejoin_use_every_n_slices = self.rejoin_use_every_n_slices;
        metadata.trial_binning = self.trial_binning;
        metadata.midas_limit = self.midas_limit.clone();
        metadata.model_transform = "search".into();
        metadata.boundaries_to_analyze = self.boundaries_to_analyze.clone();
        metadata.objects_to_include = self.objects_to_include.clone();
        metadata.gap = self.gap;
        metadata.gap_start = self.gap_start.clone();
        metadata.gap_end = self.gap_end.clone();
        metadata.gap_inc = self.gap_inc.clone();
        metadata.points_to_fit_min = self.points_to_fit_min.clone();
        metadata.points_to_fit_max = self.points_to_fit_max.clone();
        metadata.rejoin_trial_binning = self.rejoin_trial_binning;
        self.section_table
            .get_meta_data(&mut metadata.section_table)
    }
    pub fn set_meta_data(&mut self, metadata: &JoinDialogMetaData) {
        self.root_name = metadata.dataset_name.clone();
        self.density_ref_section = metadata.density_ref_section;
        self.use_alignment_ref_section = metadata.use_alignment_ref_section;
        self.alignment_ref_section = metadata.alignment_ref_section;
        self.shift_in_x = metadata.shift_in_x.clone();
        self.shift_in_y = metadata.shift_in_y.clone();
        self.use_every_n_slices = metadata.use_every_n_slices;
        self.rejoin_use_every_n_slices = metadata.rejoin_use_every_n_slices;
        self.trial_binning = metadata.trial_binning;
        self.midas_limit = metadata.midas_limit.clone();
        self.section_table.set_meta_data(&metadata.section_table);
        self.size_in_x = metadata.size_in_x.clone();
        self.size_in_y = metadata.size_in_y.clone();
        self.local_fits = metadata.local_fits;
        self.boundaries_to_analyze = metadata.boundaries_to_analyze.clone();
        self.objects_to_include = metadata.objects_to_include.clone();
        self.gap = metadata.gap;
        self.gap_start = metadata.gap_start.clone();
        self.gap_end = metadata.gap_end.clone();
        self.gap_inc = metadata.gap_inc.clone();
        self.points_to_fit_min = metadata.points_to_fit_min.clone();
        self.points_to_fit_max = metadata.points_to_fit_max.clone();
        self.rejoin_trial_binning = metadata.rejoin_trial_binning;
        self.update_display(false, false, None);
    }
    pub fn set_screen_state(&mut self, refine_with_trial: bool) {
        if self.refine_with_trial_enabled {
            self.refine_with_trial = refine_with_trial;
        }
    }
    /// Java `getScreenState` including the boundary-table delegation.
    pub fn get_screen_state(&mut self, refine_with_trial: &mut bool) {
        *refine_with_trial = self.refine_with_trial;
        self.last_action = Some("boundaryTable.getScreenState".into());
    }
    pub fn is_refine_with_trial(&self) -> bool {
        self.refine_with_trial
    }
    pub fn get_container(&self) -> bool {
        true
    }
    pub fn validate_makejoincom(&mut self) -> bool {
        self.section_table.validate_makejoincom()
    }
    pub fn validate_finishjoin(&mut self) -> bool {
        self.section_table.validate_finishjoin()
    }
    pub fn get_working_dir_name(&self) -> &str {
        &self.working_dir
    }
    pub fn get_working_dir(&mut self) -> Option<PathBuf> {
        if self.working_dir.trim().is_empty() {
            return None;
        }
        if self.working_dir.ends_with(' ') {
            self.messages.push((
                "Unusable Directory Name".into(),
                format!(
                    "The directory, {}, cannot be used because it ends with a space.",
                    self.working_dir
                ),
            ));
            return None;
        }
        Some(PathBuf::from(&self.working_dir))
    }
    pub fn get_root_name(&self) -> &str {
        &self.root_name
    }
    pub fn abort_add_section(&mut self) {
        self.section_table.enable_add_section();
    }
    pub fn msg_process_ended(&mut self) {
        self.last_action = Some("autoAlignmentPanel.msgProcessChange(true)".into());
    }
    pub fn equals(&self, metadata: &JoinDialogMetaData) -> bool {
        self.root_name == metadata.dataset_name
            && self.density_ref_section == metadata.density_ref_section
            && self.use_alignment_ref_section == metadata.use_alignment_ref_section
            && self.alignment_ref_section == metadata.alignment_ref_section
            && self.size_in_x == metadata.size_in_x
            && self.size_in_y == metadata.size_in_y
            && self.shift_in_x == metadata.shift_in_x
            && self.shift_in_y == metadata.shift_in_y
            && self.use_every_n_slices == metadata.use_every_n_slices
            && self.rejoin_use_every_n_slices == metadata.rejoin_use_every_n_slices
            && self.trial_binning == metadata.trial_binning
            && self.section_table.equals(&metadata.section_table)
    }
    pub fn equals_sample(&self, metadata: &JoinDialogMetaData) -> bool {
        self.root_name == metadata.dataset_name
            && self.density_ref_section == metadata.density_ref_section
            && self.section_table.equals_sample(&metadata.section_table)
    }
    pub fn add_section(&mut self, tomogram: impl Into<PathBuf>) {
        self.section_table.add_section(tomogram);
    }
    pub fn pop_up_context_menu(&self) -> Option<(&'static str, Vec<&'static str>)> {
        match self.cur_tab {
            Tab::Setup => Some(("Setup", vec!["3dmod", "startjoin"])),
            Tab::Align => Some(("Align", vec!["Xfalign", "Midas", "3dmod"])),
            Tab::Join => Some(("Joining", vec!["Finishjoin", "3dmod"])),
            Tab::Model => Some(("Joining", vec!["Xfjointomo", "3dmod"])),
            Tab::Rejoin => None,
        }
    }
    pub fn action<M: JoinDialogManager>(&mut self, command: &str, manager: &mut M) {
        match command {
            "Make Samples" => manager.make_join_com(),
            FINISH_JOIN_TEXT => manager.finish_join(JoinFinishMode::FinishJoin, FINISH_JOIN_TEXT),
            GET_MAX_SIZE_TEXT => manager.finish_join(JoinFinishMode::MaxSize, GET_MAX_SIZE_TEXT),
            TRIAL_JOIN_TEXT => manager.finish_join(JoinFinishMode::Trial, TRIAL_JOIN_TEXT),
            REJOIN_TEXT => manager.finish_join(JoinFinishMode::Rejoin, REJOIN_TEXT),
            TRIAL_REJOIN_TEXT => {
                manager.finish_join(JoinFinishMode::TrialRejoin, TRIAL_REJOIN_TEXT)
            }
            REFINE_JOIN_TEXT => manager.start_refine(),
            "Find Transformations" => manager.xfjointomo(),
            "Transform Model" => manager.xfmodel(&self.model_file, &self.transformed_model),
            "Transform & View Model" => {
                manager.finish_join(JoinFinishMode::SuppressExecution, REJOIN_TEXT)
            }
            "Open Sample in 3dmod" => manager.imod_open("join-samples", 1, None),
            "Open Sample Averages in 3dmod" => manager.imod_open("join-sample-averages", 1, None),
            OPEN_IN_3DMOD => manager.imod_open("join", 1, None),
            "Open Trial in 3dmod" => manager.imod_open("trial-join", 1, None),
            "Open Rejoin in 3dmod" => manager.imod_open("join", 1, Some("refine-aligned-model")),
            "Open Trial Rejoin in 3dmod" => manager.imod_open("trial-join", 1, None),
            "Open Rejoin with Transformed Model" => {
                manager.imod_open("join", 1, Some(&self.transformed_model))
            }
            "Change Setup" => {
                // `JoinManager.getJoinMetaData`/ParameterStore save are the
                // untranslated persistence boundary; setMode itself is kept
                // as its source-named public operation above.
                self.last_action = Some(
                    "manager.getJoinMetaData; parameterStore.save; setMode(CHANGING_SAMPLE_MODE)"
                        .into(),
                );
            }
            "Revert to Last Setup" => {
                self.last_action = Some("manager.getConstMetaData; sectionTable.deleteSections; state.revert; setMode(SAMPLE_PRODUCED_MODE)".into());
            }
            "Get Subarea Size And Shift" => {
                self.last_action =
                    Some("manager.imodGetRubberbandCoordinates; setSizeAndShift".into());
            }
            _ => self.update_display(false, false, None),
        };
        self.last_action = Some(command.into());
    }
    pub fn set_refine_data_highlight(&mut self, highlight: bool) {
        self.section_table.set_join_final_start_highlight(highlight);
        self.section_table.set_join_final_end_highlight(highlight);
    }
    pub fn get_section_table_meta_data(&mut self, metadata: &mut JoinDialogMetaData) -> bool {
        self.section_table
            .get_meta_data(&mut metadata.section_table)
    }
    /// Java package-private `getSectionTable`.
    pub fn get_section_table(&mut self) -> &mut SectionTablePanel {
        &mut self.section_table
    }
    pub fn get_button_name(&self, use_trial: bool) -> &'static str {
        if use_trial {
            TRIAL_JOIN_TEXT
        } else {
            FINISH_JOIN_TEXT
        }
    }
    pub fn set_refining_join(&mut self, input: bool) {
        self.refining_join = input;
    }
    pub fn working_dir_action(&mut self, selected: Option<&Path>) {
        if let Some(selected) = selected {
            self.working_dir = selected.to_string_lossy().into();
        }
    }
    pub fn model_file_action(&mut self, selected: Option<&Path>) {
        if let Some(selected) = selected {
            self.model_file = selected.to_string_lossy().into();
        }
    }
    pub fn synchronize(&mut self, previous: Option<Tab>) {
        if let Some(previous) = previous {
            self.section_table
                .synchronize(previous.section_tab(), self.cur_tab.section_tab());
        }
    }
    /// Java `setToolTipText`. Autodoc-owned xfjointomo text is intentionally a
    /// storage boundary; the source's static tooltip assignments are retained.
    fn set_tool_tip_text(&mut self) {
        self.tooltips = vec![
            ("ftfWorkingDir".into(), "Enter the directory where you wish to place the joined tomogram.".into()),
            ("ltfRootName".into(), "Enter the root name for the joined tomogram.".into()),
            ("ltfMidasLimit".into(), "The size to which samples will be squeezed if they are bigger (default 1024).".into()),
            ("spinDensityRefSection".into(), "Select a section to use as a reference for density scaling.".into()),
            ("btnChangeSetup".into(), "Press to redo an existing sample.".into()),
            ("btnRevertToLastSetup".into(), "Press to go back to the existing sample.".into()),
            ("btnMakeSamples".into(), "Press to make a sample.".into()),
            ("btnGetMaxSize".into(), "Compute the maximum size and offsets needed to contain the transformed images from all of the sections, given the current transformations.".into()),
            ("cbLocalFits".into(), "When running Xftoxg(1) on the primary alignment transforms, run the program in its default mode, which does local fits to 7 adjacent sections.".into()),
            ("spinUseEveryNSlices".into(), "Slices to use when creating the trial joined tomogram.".into()),
            ("spinTrialBinning".into(), "The binning to use when creating the trial joined tomogram.".into()),
            ("btnTrialJoin".into(), "Press to make a trial version of the joined tomogram.".into()),
            ("btnFinishJoin".into(), "Press to make the joined tomogram.".into()),
            ("btnRefineJoin".into(), "Press to refine the serial section join using a refining model.".into()),
            ("btnXfjointomo".into(), "Press to run xfjointomo, which computes transforms for aligning tomograms of serial sections from features modeled on an initial joined tomogram.".into()),
            ("btnTransformAndViewModel".into(), "Press to apply tranformations to the refining model and view the result.".into()),
            ("btnRejoin".into(), "Press to make the joined tomogram using the adjusted end and start values.".into()),
            ("btnTrialRejoin".into(), "Press to make a trial version of the joined tomogram using the adjusted end and start values.".into()),
            ("ftfModelFile".into(), "The model to transform.".into()),
            ("btnTransformModel".into(), "Press to transform the model".into()),
        ];
    }
}

impl std::fmt::Display for JoinDialog {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "JoinDialog[{}]", self.param_string())
    }
}

/// Java private inner `JoinActionListener`. Native `ActionEvent` construction
/// is a GUI boundary; this adapter preserves its source forwarding rule.
pub struct JoinActionListener;
impl JoinActionListener {
    pub fn new() -> Self {
        Self
    }
    pub fn action_performed<M: JoinDialogManager>(
        &self,
        dialog: &mut JoinDialog,
        command: &str,
        manager: &mut M,
    ) {
        dialog.action(command, manager);
    }
}
/// Java inner `WorkingDirActionListener`.
pub struct WorkingDirActionListener;
impl WorkingDirActionListener {
    pub fn new() -> Self {
        Self
    }
    pub fn action_performed(&self, dialog: &mut JoinDialog, selected: Option<&Path>) {
        dialog.working_dir_action(selected);
    }
}
/// Java private inner `ModelFileActionListener`.
pub struct ModelFileActionListener;
impl ModelFileActionListener {
    pub fn new() -> Self {
        Self
    }
    pub fn action_performed(&self, dialog: &mut JoinDialog, selected: Option<&Path>) {
        dialog.model_file_action(selected);
    }
}
/// Java inner `TabChangeListener`.
pub struct TabChangeListener;
impl TabChangeListener {
    pub fn new() -> Self {
        Self
    }
    pub fn state_changed(&self, dialog: &mut JoinDialog, selected_index: usize) {
        dialog.change_tab(selected_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct State {
        revert: bool,
        sample: bool,
        trial: Option<i32>,
        list_empty: bool,
    }
    impl JoinDialogState for State {
        fn is_refine_start_list_empty(&self) -> bool {
            self.list_empty
        }
        fn join_trial_use_every_n_slices(&self) -> Option<i32> {
            self.trial
        }
        fn refine_trial_use_every_n_slices(&self) -> Option<i32> {
            self.trial
        }
        fn is_sample_produced(&self) -> bool {
            self.sample
        }
        fn set_revert_state(&mut self, v: bool) {
            self.revert = v
        }
        fn revert(&mut self) {}
    }
    #[derive(Default)]
    struct Manager(Vec<String>);
    impl JoinDialogManager for Manager {
        fn make_join_com(&mut self) {
            self.0.push("makejoincom".into())
        }
        fn finish_join(&mut self, _: JoinFinishMode, label: &str) {
            self.0.push(label.into())
        }
        fn start_refine(&mut self) {
            self.0.push("refine".into())
        }
        fn xfjointomo(&mut self) {
            self.0.push("xfjointomo".into())
        }
        fn xfmodel(&mut self, _: &str, _: &str) {
            self.0.push("xfmodel".into())
        }
        fn imod_open(&mut self, key: &str, _: i32, _: Option<&str>) {
            self.0.push(key.into())
        }
    }
    #[test]
    fn source_mode_policy_and_trial_refinement_are_retained() {
        let metadata = JoinDialogMetaData::default();
        let mut dialog = JoinDialog::get_instance(&metadata, true);
        let mut state = State {
            trial: Some(1),
            ..Default::default()
        };
        dialog.update_display(false, true, Some(&state));
        assert!(dialog.refine_with_trial_enabled);
        dialog.set_mode(SAMPLE_PRODUCED_MODE, &mut state);
        assert!(!dialog.root_name_editable);
        assert!(dialog.tabs_enabled[1]);
        dialog.set_mode(CHANGING_SAMPLE_MODE, &mut state);
        assert!(state.revert);
        assert!(dialog.revert_to_last_setup_enabled);
    }
    #[test]
    fn action_routes_source_labels_and_metadata_round_trips() {
        let mut metadata = JoinDialogMetaData {
            dataset_name: "join".into(),
            shift_in_x: "3".into(),
            shift_in_y: "-2".into(),
            ..Default::default()
        };
        let mut dialog = JoinDialog::get_instance(&metadata, false);
        let mut manager = Manager::default();
        dialog.action(TRIAL_JOIN_TEXT, &mut manager);
        dialog.action("Find Transformations", &mut manager);
        assert_eq!(manager.0, vec![TRIAL_JOIN_TEXT, "xfjointomo"]);
        assert!(dialog.get_meta_data(&mut metadata, false));
        assert_eq!(metadata.dataset_name, "join");
    }
    #[test]
    fn blank_and_trailing_space_work_directories_follow_source() {
        let mut dialog = JoinDialog::get_instance(&JoinDialogMetaData::default(), false);
        assert_eq!(dialog.get_working_dir(), None);
        dialog.working_dir = "bad ".into();
        assert_eq!(dialog.get_working_dir(), None);
        assert_eq!(dialog.messages[0].0, "Unusable Directory Name");
    }
}
