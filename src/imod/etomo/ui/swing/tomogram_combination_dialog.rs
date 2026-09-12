//! `IMOD/Etomo/src/etomo/ui/swing/TomogramCombinationDialog.java`.
//!
//! Panel rendering, process launch, and processing-method mediation remain direct
//! boundaries.  This unit keeps the three-tab dialog's selection, synchronization,
//! enablement, visibility, and context-popup decisions.
#![allow(dead_code)]

use super::context_menu::{ContextMenu, MouseEvent};
use crate::imod::etomo::r#type::{
    axis_id::AxisID, dialog_type::DialogType, match_mode::MatchMode,
    processing_method::ProcessingMethod,
};

pub const SETUP_INDEX: usize = 0;
pub const INITIAL_INDEX: usize = 1;
pub const FINAL_INDEX: usize = 2;
pub const LBL_SETUP: &str = "Setup";
pub const LBL_INITIAL: &str = "Initial Match";
pub const LBL_FINAL: &str = "Final Match";
pub const ALL_FIELDS: i32 = 10;

/// Java `CombineProcessType` values consumed by `showPane`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CombineProcessType {
    Solvematch,
    Dualvolmatch,
    Matchvol1,
    Patchcorr,
    Matchorwarp,
    Volcombine,
}

/// Direct `ProcessingMethodMediator` and `UIHarness` calls from this source unit.
pub trait TomogramCombinationDialogApplicationManager {
    fn register_processing_method(&mut self);
    fn deregister_processing_method(&mut self);
    fn set_processing_method(&mut self, method: ProcessingMethod);
    fn done_tomogram_combination_dialog(&mut self);
    fn pack(&mut self, axis_id: AxisID);
    fn move_sub_frame(&mut self);
}

/// The source fields shared by initial and setup panels.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InitialCombineFields {
    pub enabled: bool,
    pub surfaces_or_models: String,
    pub bin_by_2: bool,
    pub fiducial_match_list_a: String,
    pub fiducial_match_list_b: String,
    pub use_corresponding_points: bool,
    pub use_list: bool,
    pub match_mode: Option<MatchMode>,
    pub initial_volume_matching: bool,
}
/// The source fields shared by final and setup panels.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FinalCombineFields {
    pub enabled: bool,
    pub use_patch_region_model: bool,
    pub x_min: String,
    pub x_max: String,
    pub y_min: String,
    pub y_max: String,
    pub z_min: String,
    pub z_max: String,
    pub parallel: bool,
    pub parallel_enabled: bool,
    pub no_volcombine: bool,
}
/// Java `ContextPopup` constructor inputs in `popUpContextMenu`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TomogramCombinationContextPopup {
    pub mouse_event: MouseEvent,
    pub title: &'static str,
    pub guide: &'static str,
    pub man_page_labels: [&'static str; 4],
    pub man_pages: [&'static str; 4],
    pub log_file_labels: [&'static str; 3],
}

/// Java `TomogramCombinationDialog` fields, with panel internals remaining at their
/// already translated source-unit boundaries.
pub struct TomogramCombinationDialog {
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub pnl_setup: InitialCombineFields,
    pub pnl_initial: InitialCombineFields,
    pub pnl_final: FinalCombineFields,
    pub combine_panel_enabled: bool,
    pub parallel_panel_container_present: bool,
    pub tab_enabled: [bool; 3],
    pub selected_tab_index: usize,
    pub idx_last_tab: usize,
    pub parallel_process_check_box_text: String,
    pub constructed: bool,
    pub displayed: bool,
    pub setup_visible: bool,
    pub initial_visible: bool,
    pub final_visible: bool,
    pub advanced: bool,
    pub initial_advanced: bool,
    pub final_advanced: bool,
    pub setup_processing_method: ProcessingMethod,
    pub initial_processing_method: ProcessingMethod,
    pub final_processing_method: ProcessingMethod,
    pub z_min: String,
    pub z_max: String,
    pub binning_warning: bool,
    pub run_volcombine: bool,
    pub context_popup: Option<TomogramCombinationContextPopup>,
    pub listeners_removed: bool,
}

impl TomogramCombinationDialog {
    /// Java `TomogramCombinationDialog(ApplicationManager)`.
    pub fn new(max_volcombine_cpus: Option<i32>) -> Self {
        let parallel_process_check_box_text = match max_volcombine_cpus {
            Some(value) => format!("Parallel processing (maximum CPUs: {value})"),
            None => "Parallel processing".into(),
        };
        let mut dialog = Self {
            axis_id: AxisID::First,
            dialog_type: DialogType::TomogramCombination,
            pnl_setup: InitialCombineFields {
                enabled: true,
                ..Default::default()
            },
            pnl_initial: InitialCombineFields {
                enabled: true,
                ..Default::default()
            },
            pnl_final: FinalCombineFields {
                enabled: true,
                parallel_enabled: true,
                ..Default::default()
            },
            combine_panel_enabled: true,
            parallel_panel_container_present: true,
            tab_enabled: [true; 3],
            selected_tab_index: SETUP_INDEX,
            idx_last_tab: SETUP_INDEX,
            parallel_process_check_box_text,
            constructed: false,
            displayed: false,
            setup_visible: true,
            initial_visible: false,
            final_visible: false,
            advanced: false,
            initial_advanced: false,
            final_advanced: false,
            setup_processing_method: ProcessingMethod::LocalCpu,
            initial_processing_method: ProcessingMethod::LocalCpu,
            final_processing_method: ProcessingMethod::LocalCpu,
            z_min: String::new(),
            z_max: String::new(),
            binning_warning: false,
            run_volcombine: true,
            context_popup: None,
            listeners_removed: false,
        };
        dialog.update_advanced();
        dialog.constructed = true;
        dialog.update_display();
        dialog
    }
    pub fn to_string(&self) -> String {
        format!("TomogramCombinationDialog[{}]\n", self.param_string())
    }
    pub fn param_string(&self) -> String {
        format!(
            "pnlSetup={:?},\npnlInitial={:?},\npnlFinal={:?},\ncombinePanelEnabled={},\nparallelProcessCheckBoxText={},\nidxLastTab={}",
            self.pnl_setup,
            self.pnl_initial,
            self.pnl_final,
            self.combine_panel_enabled,
            self.parallel_process_check_box_text,
            self.idx_last_tab
        )
    }
    pub fn remove_listeners(&mut self) {
        self.listeners_removed = true;
    }
    /// Java `setCombineParams` source state entry.
    pub fn set_combine_params(
        &mut self,
        setup: InitialCombineFields,
        final_fields: FinalCombineFields,
        _init: bool,
    ) {
        self.pnl_setup = setup;
        self.pnl_final = final_fields;
    }
    pub fn set_z_min(&mut self, z_min: impl Into<String>) {
        self.z_min = z_min.into();
        self.pnl_final.z_min = self.z_min.clone();
    }
    pub fn set_z_max(&mut self, z_max: impl Into<String>) {
        self.z_max = z_max.into();
        self.pnl_final.z_max = self.z_max.clone();
    }
    pub fn get_combine_params(
        &self,
        setup: &mut InitialCombineFields,
        _do_validation: bool,
    ) -> bool {
        *setup = self.pnl_setup.clone();
        true
    }
    pub fn get_imod_combined_button(&self) -> &'static str {
        "Open Combined Tomogram"
    }
    pub fn set_solvematch_params(&mut self, fields: InitialCombineFields) {
        self.pnl_initial = fields;
    }
    pub fn set_dualvolmatch_params(&mut self, fields: InitialCombineFields) {
        self.pnl_initial = fields;
    }
    pub fn set_parameters_matchvol(&mut self, fields: InitialCombineFields) {
        self.pnl_initial = fields;
    }
    pub fn set_parameters_recon_screen_state(
        &mut self,
        setup: InitialCombineFields,
        initial: InitialCombineFields,
        final_fields: FinalCombineFields,
    ) {
        self.pnl_setup = setup;
        self.pnl_initial = initial;
        self.pnl_final = final_fields;
    }
    pub fn get_parameters_recon_screen_state(
        &self,
    ) -> (
        InitialCombineFields,
        InitialCombineFields,
        FinalCombineFields,
    ) {
        (
            self.pnl_setup.clone(),
            self.pnl_initial.clone(),
            self.pnl_final.clone(),
        )
    }
    pub fn show_pane(&mut self, combine_process_type: Option<CombineProcessType>) {
        if let Some(kind) = combine_process_type {
            self.selected_tab_index = match kind {
                CombineProcessType::Solvematch
                | CombineProcessType::Dualvolmatch
                | CombineProcessType::Matchvol1 => INITIAL_INDEX,
                _ => FINAL_INDEX,
            };
        }
    }
    pub fn get_parameters_metadata(&mut self, setup: &mut InitialCombineFields) {
        self.synchronize(LBL_SETUP, false);
        *setup = self.pnl_setup.clone();
    }
    pub fn show<M: TomogramCombinationDialogApplicationManager>(&mut self, manager: &mut M) {
        manager.register_processing_method();
        manager.set_processing_method(ProcessingMethod::LocalCpu);
        self.displayed = true;
    }
    pub fn set_parameters_metadata(&mut self, setup: InitialCombineFields) {
        self.pnl_setup = setup;
        self.synchronize(LBL_SETUP, true);
    }
    pub fn get_solvematch_params(
        &self,
        initial: &mut InitialCombineFields,
        _do_validation: bool,
    ) -> bool {
        *initial = self.pnl_initial.clone();
        true
    }
    pub fn get_parameters_dualvolmatch(
        &self,
        initial: &mut InitialCombineFields,
        _do_validation: bool,
    ) -> bool {
        *initial = self.pnl_initial.clone();
        true
    }
    pub fn get_parameters_matchvol(
        &self,
        initial: &mut InitialCombineFields,
        _do_validation: bool,
    ) -> bool {
        *initial = self.pnl_initial.clone();
        true
    }
    pub fn set_patchcrawl_3d_params(&mut self, final_fields: FinalCombineFields) {
        self.pnl_final = final_fields;
    }
    pub fn set_reduction_factor_params(&mut self, enabled: bool) {
        self.pnl_final.parallel_enabled = enabled;
    }
    pub fn set_low_from_both_radius_params(&mut self, enabled: bool) {
        self.pnl_final.use_patch_region_model = enabled;
    }
    pub fn get_patchcrawl_3d_params(
        &self,
        final_fields: &mut FinalCombineFields,
        _do_validation: bool,
    ) -> bool {
        *final_fields = self.pnl_final.clone();
        true
    }
    pub fn get_reduction_factor_param(
        &self,
        final_fields: &mut FinalCombineFields,
        do_validation: bool,
    ) -> bool {
        self.get_patchcrawl_3d_params(final_fields, do_validation)
    }
    pub fn get_low_from_both_radius_param(
        &self,
        final_fields: &mut FinalCombineFields,
        do_validation: bool,
    ) -> bool {
        self.get_patchcrawl_3d_params(final_fields, do_validation)
    }
    pub fn enable_reduction_factor(&mut self, enable: bool) {
        self.pnl_final.parallel_enabled = enable;
    }
    pub fn enable_low_from_both_radius(&mut self, enable: bool) {
        self.pnl_final.use_patch_region_model = enable;
    }
    pub fn get_match_mode(&self) -> Option<MatchMode> {
        self.pnl_setup.match_mode
    }
    pub fn set_matchorwarp_params(&mut self, final_fields: FinalCombineFields) {
        self.pnl_final = final_fields;
    }
    pub fn update_gpu(&mut self, _disable: bool) {}
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.pnl_setup.enabled = !lock;
        self.pnl_final.parallel_enabled = self.pnl_setup.enabled;
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        match self.selected_tab_index {
            SETUP_INDEX => self.setup_processing_method,
            INITIAL_INDEX => self.initial_processing_method,
            FINAL_INDEX => self.final_processing_method,
            _ => ProcessingMethod::LocalCpu,
        }
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    pub fn get_run_processing_method(&self) -> ProcessingMethod {
        if self.selected_tab_index == INITIAL_INDEX {
            self.final_processing_method
        } else {
            self.get_processing_method()
        }
    }
    /// Java `synchronize(String, boolean)`.
    pub fn synchronize(&mut self, tab_title: &str, copy_from_tab: bool) {
        if tab_title == LBL_SETUP {
            if copy_from_tab {
                self.synchronize_initial(self.pnl_setup.clone());
                self.synchronize_final(self.pnl_final.clone());
            } else {
                self.pnl_setup = self.pnl_initial.clone();
                self.synchronize_final(self.pnl_final.clone());
            }
        } else if tab_title == LBL_INITIAL {
            if copy_from_tab {
                self.pnl_setup = self.pnl_initial.clone();
            } else {
                self.pnl_initial = self.pnl_setup.clone();
            }
        } else if tab_title == LBL_FINAL {
            if copy_from_tab {
                self.synchronize_final(self.pnl_final.clone());
            } else {
                self.pnl_final = self.final_from_setup();
            }
        }
        self.update_display();
    }
    /// Java private `synchronize(InitialCombineFields, InitialCombineFields)`.
    pub fn synchronize_initial(&mut self, from_panel: InitialCombineFields) {
        if from_panel.enabled && self.pnl_initial.enabled {
            self.pnl_initial = from_panel;
        }
    }
    /// Java private `synchronize(FinalCombineFields, FinalCombineFields)`.
    pub fn synchronize_final(&mut self, from_panel: FinalCombineFields) {
        if from_panel.enabled && self.pnl_final.enabled {
            self.pnl_final = from_panel;
        }
    }
    pub fn final_from_setup(&self) -> FinalCombineFields {
        self.pnl_final.clone()
    }
    pub fn is_changed(
        &self,
        combine_scripts_created: bool,
        script_match_mode: Option<MatchMode>,
    ) -> bool {
        !combine_scripts_created || script_match_mode != self.pnl_setup.match_mode
    }
    pub fn update_display(&mut self) {
        if self.constructed {
            let enable_tabs = self.combine_panel_enabled;
            self.tab_enabled[INITIAL_INDEX] = enable_tabs;
            self.tab_enabled[FINAL_INDEX] = enable_tabs;
        }
    }
    pub fn update_patch_vector_model_display(&mut self) {}
    pub fn is_run_volcombine(&self) -> bool {
        self.run_volcombine
    }
    pub fn set_run_volcombine(&mut self, run_volcombine: bool) {
        self.run_volcombine = run_volcombine;
    }
    pub fn is_parallel(&self) -> bool {
        self.pnl_final.parallel
    }
    pub fn set_binning_warning(&mut self, binning_warning: bool) {
        self.binning_warning = binning_warning;
    }
    pub fn get_matchorwarp_params(
        &self,
        final_fields: &mut FinalCombineFields,
        do_validation: bool,
    ) -> bool {
        self.get_patchcrawl_3d_params(final_fields, do_validation)
    }
    pub fn synchronize_from_current_tab(&mut self) {
        let title = [LBL_SETUP, LBL_INITIAL, LBL_FINAL]
            .get(self.idx_last_tab)
            .copied()
            .unwrap_or(LBL_SETUP);
        self.synchronize(title, true);
    }
    pub fn done<M: TomogramCombinationDialogApplicationManager>(&mut self, manager: &mut M) {
        manager.done_tomogram_combination_dialog();
        self.displayed = false;
        manager.deregister_processing_method();
    }
    pub fn update_advanced(&mut self) {
        self.initial_advanced = self.advanced;
        self.final_advanced = self.advanced;
    }
    pub fn is_tab_enabled(&self, tab_label: &str) -> Result<bool, String> {
        match tab_label {
            LBL_SETUP => Ok(self.tab_enabled[SETUP_INDEX]),
            LBL_INITIAL => Ok(self.tab_enabled[INITIAL_INDEX]),
            LBL_FINAL => Ok(self.tab_enabled[FINAL_INDEX]),
            _ => Err(format!("tabLabel={tab_label}")),
        }
    }
    pub fn tab_state_change<M: TomogramCombinationDialogApplicationManager>(
        &mut self,
        manager: &mut M,
    ) {
        let last = [LBL_SETUP, LBL_INITIAL, LBL_FINAL]
            .get(self.idx_last_tab)
            .copied()
            .unwrap_or(LBL_SETUP);
        self.synchronize(last, true);
        let title = [LBL_SETUP, LBL_INITIAL, LBL_FINAL]
            .get(self.selected_tab_index)
            .copied()
            .unwrap_or(LBL_SETUP);
        self.set_visible(manager, title);
        self.idx_last_tab = self.selected_tab_index;
        manager.set_processing_method(self.get_processing_method());
    }
    pub fn set_visible<M: TomogramCombinationDialogApplicationManager>(
        &mut self,
        manager: &mut M,
        show_tab_title: &str,
    ) {
        self.setup_visible = show_tab_title == LBL_SETUP;
        self.initial_visible = show_tab_title == LBL_INITIAL;
        self.final_visible = show_tab_title == LBL_FINAL;
        manager.pack(AxisID::Only);
        manager.move_sub_frame();
    }
    pub fn set_method<M: TomogramCombinationDialogApplicationManager>(
        &self,
        manager: &mut M,
        processing_method: ProcessingMethod,
    ) {
        manager.set_processing_method(processing_method);
    }
    pub fn is_use_gpu(&self) -> bool {
        false
    }
    pub fn set_use_queue_check_box(&mut self) {}
}
impl ContextMenu for TomogramCombinationDialog {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.context_popup = Some(TomogramCombinationContextPopup {
            mouse_event,
            title: "TOMOGRAM COMBINATION",
            guide: "tomoguide.html",
            man_page_labels: ["Solvematch", "Matchshifts", "Patchcrawl3d", "Matchorwarp"],
            man_pages: [
                "solvematch.html",
                "matchshifts.html",
                "patchcrawl3d.html",
                "matchorwarp.html",
            ],
            log_file_labels: ["Solvematch.log", "Patchcorr.log", "Matchorwarp.log"],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_keeps_source_tab_and_parallel_defaults() {
        let dialog = TomogramCombinationDialog::new(Some(16));
        assert_eq!(dialog.selected_tab_index, SETUP_INDEX);
        assert!(dialog.constructed);
        assert!(dialog.parallel_process_check_box_text.contains("16"));
    }
    #[test]
    fn initial_tab_run_method_uses_final_panel_method() {
        let mut dialog = TomogramCombinationDialog::new(None);
        dialog.selected_tab_index = INITIAL_INDEX;
        dialog.final_processing_method = ProcessingMethod::PpCpu;
        assert_eq!(dialog.get_run_processing_method(), ProcessingMethod::PpCpu);
    }
    #[test]
    fn popup_has_source_manual_and_log_values() {
        let mut dialog = TomogramCombinationDialog::new(None);
        dialog.pop_up_context_menu(MouseEvent {
            x: 1,
            y: 2,
            right_mouse_button: true,
        });
        let popup = dialog.context_popup.unwrap();
        assert_eq!(popup.man_pages[2], "patchcrawl3d.html");
        assert_eq!(popup.log_file_labels[1], "Patchcorr.log");
    }
}
