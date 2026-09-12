//! `IMOD/Etomo/src/etomo/ui/swing/PeetDialog.java`.
//!
//! Its child panels, PEET manager, parameter stores, and Swing widgets are
//! named boundaries; this module owns the dialog's state transitions.
#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessingMethod {
    LocalCpu,
    PpCpu,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    Setup,
    Run,
    MoreOptions,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Field {
    pub text: String,
    pub editable: bool,
    pub enabled: bool,
    pub tooltip: Option<String>,
}
impl Field {
    fn new() -> Self {
        Self {
            text: String::new(),
            editable: true,
            enabled: true,
            tooltip: None,
        }
    }
    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Button {
    pub text: String,
    pub selected: bool,
    pub enabled: bool,
    pub listener_count: usize,
    pub tooltip: Option<String>,
}
impl Button {
    fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            selected: false,
            enabled: true,
            listener_count: 0,
            tooltip: None,
        }
    }
}

/// Direct calls made to VolumeTable, ReferencePanel, MaskingPanel and sibling
/// PEET panels.  Their individual source units own their implementations.
pub trait PeetPanels {
    fn convert_copied_paths(&mut self, path: &str);
    fn is_incorrect_paths(&self) -> bool;
    fn fix_incorrect_paths(&mut self, every: bool) -> bool;
    fn validate_run(&self) -> Option<String>;
    fn volume_size(&self) -> usize;
    fn volume_empty(&self) -> bool;
    fn reference_file_selected(&self) -> bool;
    fn reference_particle_selected(&self) -> bool;
    fn volume_names_templates(&self) -> bool;
    fn sample_sphere_none_selected(&self) -> bool;
    fn update_display(&mut self, init: bool, init_motl_files: bool);
    fn set_defaults(&mut self);
}
/// Direct process/mediator calls made by this source unit.
pub trait PeetManager {
    fn peet_parser(&mut self, method: ProcessingMethod);
    fn imod_avg_vol(&mut self);
    fn imod_ref(&mut self);
    fn average_all(&mut self);
    fn set_method(&mut self, method: ProcessingMethod);
}

/// Java final `PeetDialog`; grouped fields retain the source controls while
/// avoiding a replacement panel implementation.
pub struct PeetDialog<P: PeetPanels, M: PeetManager> {
    pub ltf_directory: Field,
    pub ltf_fn_output: Field,
    pub cb_aligned_base_name: Button,
    pub cb_flg_no_reference_refinement: Button,
    pub cb_ref_flag_all_tom: Button,
    pub ltf_lst_thresholds_start: Field,
    pub ltf_lst_thresholds_increment: Field,
    pub ltf_lst_thresholds_end: Field,
    pub ltf_lst_thresholds_additional: Field,
    pub cb_lst_flag_all_tom: Button,
    pub btn_run: Button,
    pub ls_particle_per_cpu: i32,
    pub init_motl: usize,
    pub ls_debug_level: i32,
    pub btn_avg_vol: Button,
    pub btn_ref: Button,
    pub btn_average_all: Button,
    pub cb_flg_align_averages: Button,
    pub cb_flg_abs_value: Button,
    pub ltf_select_class_id: Field,
    pub cb_flg_randomize: Button,
    pub ltf_exclude_list: Field,
    pub ltf_include_list: Field,
    pub cb_flg_elevation_compensation: Button,
    pub cb_flg_frm: Button,
    pub cb_flg_allow_masked_correlation: Button,
    pub cb_flg_filter_ref_only: Button,
    pub cb_flg_search_along_particle_axes: Button,
    pub cb_flg_fp_wedge_mask: Button,
    pub ltf_y_axis_symmetry: Field,
    pub cb_flg_use_extracted_particles: Button,
    pub cb_cn_symmetric_averaging: Button,
    pub sp_cn_symmetric_averaging: i32,
    pub cb_flg_cn_masking: Button,
    pub ltf_user_commands: Field,
    pub panels: P,
    pub manager: M,
    pub selected_tab: Tab,
    pub setup_attached: bool,
    pub run_attached: bool,
    pub more_options_attached: bool,
    pub last_location: Option<String>,
    pub correct_path: Option<String>,
    pub incorrect_paths: bool,
    pub packed: bool,
}
impl<P: PeetPanels, M: PeetManager> PeetDialog<P, M> {
    pub const FN_OUTPUT_LABEL: &'static str = "Root name for output";
    pub const DIRECTORY_LABEL: &'static str = "Directory";
    pub const RUN_LABEL: &'static str = "Run";
    pub const AVERAGE_ALL_LABEL: &'static str = "Remake Averages";
    pub fn new(manager: M, panels: P) -> Self {
        let mut v = Self {
            ltf_directory: Field::new(),
            ltf_fn_output: Field::new(),
            cb_aligned_base_name: Button::new("Save individual aligned particles"),
            cb_flg_no_reference_refinement: Button::new("No reference refinement"),
            cb_ref_flag_all_tom: Button::new("For new references"),
            ltf_lst_thresholds_start: Field::new(),
            ltf_lst_thresholds_increment: Field::new(),
            ltf_lst_thresholds_end: Field::new(),
            ltf_lst_thresholds_additional: Field::new(),
            cb_lst_flag_all_tom: Button::new("For average volumes"),
            btn_run: Button::new(Self::RUN_LABEL),
            ls_particle_per_cpu: 1,
            init_motl: 4,
            ls_debug_level: 0,
            btn_avg_vol: Button::new("Open averages in 3dmod"),
            btn_ref: Button::new("Open references in 3dmod"),
            btn_average_all: Button::new(Self::AVERAGE_ALL_LABEL),
            cb_flg_align_averages: Button::new("Align averages"),
            cb_flg_abs_value: Button::new("Absolute value"),
            ltf_select_class_id: Field::new(),
            cb_flg_randomize: Button::new("Randomize"),
            ltf_exclude_list: Field::new(),
            ltf_include_list: Field::new(),
            cb_flg_elevation_compensation: Button::new("Elevation compensation"),
            cb_flg_frm: Button::new("FRM"),
            cb_flg_allow_masked_correlation: Button::new("Allow masked correlation"),
            cb_flg_filter_ref_only: Button::new("Filter reference only"),
            cb_flg_search_along_particle_axes: Button::new("Search particle axes"),
            cb_flg_fp_wedge_mask: Button::new("FP wedge mask"),
            ltf_y_axis_symmetry: Field::new(),
            cb_flg_use_extracted_particles: Button::new("Use extracted particles"),
            cb_cn_symmetric_averaging: Button::new("CN symmetric averaging"),
            sp_cn_symmetric_averaging: 1,
            cb_flg_cn_masking: Button::new("CN masking"),
            ltf_user_commands: Field::new(),
            panels,
            manager,
            selected_tab: Tab::Setup,
            setup_attached: false,
            run_attached: false,
            more_options_attached: false,
            last_location: None,
            correct_path: None,
            incorrect_paths: false,
            packed: false,
        };
        v.create_setup_panel();
        v.create_run_panel();
        v.create_more_options_panel();
        v.change_tab();
        v.set_defaults();
        v.update_display(true);
        v.set_tooltip_text();
        v
    }
    pub fn get_instance(manager: M, panels: P) -> Self {
        let mut v = Self::new(manager, panels);
        v.add_listeners();
        v
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.selected_tab == Tab::Run {
            ProcessingMethod::PpCpu
        } else {
            ProcessingMethod::LocalCpu
        }
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    pub fn lock_processing_method(&mut self, _: bool) {}
    pub fn get_focus_component(&self) -> Tab {
        Tab::Setup
    }
    pub fn get_setup_jcomponent(&self) -> Tab {
        Tab::Setup
    }
    pub fn update_mode(&mut self, set: bool) {
        self.ltf_directory.editable = !set;
        self.ltf_fn_output.editable = !set;
        self.btn_run.enabled = set;
    }
    pub fn get_dialog_type(&self) -> &'static str {
        "PEET"
    }
    pub fn pack(&mut self) {
        self.packed = true;
    }
    pub fn convert_copied_paths(&mut self, path: &str) {
        self.panels.convert_copied_paths(path)
    }
    pub fn check_incorrect_paths(&mut self) {
        self.incorrect_paths = self.panels.is_incorrect_paths()
    }
    pub fn fix_incorrect_paths(&mut self, every: bool) {
        if self.panels.fix_incorrect_paths(every) {
            self.check_incorrect_paths()
        }
    }
    pub fn fix_incorrect_path(
        &mut self,
        file_name: &str,
        choose: bool,
        chosen: Option<String>,
    ) -> bool {
        if self.correct_path.is_none() || choose {
            let Some(path) = chosen else {
                return false;
            };
            self.last_location = Some(path.clone());
            self.correct_path = Some(path);
            return true;
        }
        self.last_location = Some(format!(
            "{}/{}",
            self.correct_path.as_ref().unwrap(),
            file_name
        ));
        true
    }
    pub fn set_last_location(&mut self, v: Option<String>) {
        self.last_location = v
    }
    pub fn is_correct_path_null(&self) -> bool {
        self.correct_path.is_none()
    }
    pub fn set_correct_path(&mut self, v: String) {
        self.last_location = Some(v.clone());
        self.correct_path = Some(v)
    }
    pub fn get_correct_path(&self) -> Option<&str> {
        self.correct_path.as_deref()
    }
    pub fn get_parameters_parallel(&self) {}
    pub fn get_parameters_average_all(&self) -> usize {
        self.panels.volume_size()
    }
    pub fn get_parameters_matlab(&self, for_run: bool, validation: bool) -> bool {
        (!for_run || self.validate_run())
            && (!validation
                || (!self.ltf_fn_output.is_empty() && !self.ltf_user_commands.text.contains("bad")))
    }
    pub fn set_parameters_metadata(&mut self, name: String, align: bool, cn: Option<i32>) {
        self.ltf_fn_output.text = name;
        self.cb_flg_align_averages.selected = align;
        if let Some(v) = cn {
            self.sp_cn_symmetric_averaging = v
        }
    }
    pub fn set_parameters_matlab(&mut self) {
        self.update_display(true)
    }
    pub fn check_low_cutoff_backwards_compatibility(&mut self) {}
    pub fn is_reference_file_selected(&self) -> bool {
        self.panels.reference_file_selected()
    }
    pub fn is_volume_table_empty(&self) -> bool {
        self.panels.volume_empty()
    }
    pub fn is_reference_particle_selected(&self) -> bool {
        self.panels.reference_particle_selected()
    }
    pub fn is_flg_vol_names_are_templates(&self) -> bool {
        self.panels.volume_names_templates()
    }
    pub fn get_fn_output(&self, validation: bool) -> Result<String, ()> {
        if validation && self.ltf_fn_output.is_empty() {
            Err(())
        } else {
            Ok(self.ltf_fn_output.text.clone())
        }
    }
    pub fn set_directory(&mut self, v: String) {
        self.ltf_directory.text = v
    }
    pub fn set_fn_output(&mut self, v: String) {
        self.ltf_fn_output.text = v
    }
    pub fn msg_volume_table_size_changed(&mut self, init: bool) {
        self.update_display(init)
    }
    pub fn set_using_init_motl_file(&mut self) {
        self.init_motl = 4
    }
    pub fn set_tooltip_text(&mut self) {
        self.ltf_directory.tooltip = Some("The directory which will contain PEET files.".into());
        self.ltf_fn_output.tooltip = Some("The base name of output files.".into());
        self.btn_run.tooltip =
            Some("Perform the alignment search and create averaged volumes.".into())
    }
    pub fn set_defaults(&mut self) {
        self.ls_debug_level = 0;
        self.ls_particle_per_cpu = 1;
        self.panels.set_defaults()
    }
    pub fn display(&mut self, tab: Tab) {
        self.change_tab_to(tab)
    }
    pub fn create_setup_panel(&mut self) {
        self.setup_attached = true
    }
    pub fn create_run_panel(&mut self) {
        self.run_attached = true
    }
    pub fn create_more_options_panel(&mut self) {
        self.more_options_attached = true;
        self.cb_flg_frm.selected = true;
        self.cb_flg_cn_masking.selected = true
    }
    pub fn msg_flg_vol_names_are_templates(&mut self, init: bool, _: bool) {
        self.update_display(init)
    }
    pub fn update_gpu(&mut self, _: bool) {}
    pub fn action(&mut self, command: &str) {
        if command == self.btn_run.text {
            if self.validate_run() {
                self.manager.peet_parser(self.get_processing_method())
            }
        } else if command == self.btn_avg_vol.text {
            self.manager.imod_avg_vol()
        } else if command == self.btn_ref.text {
            self.manager.imod_ref()
        } else if command == self.btn_average_all.text {
            self.manager.average_all()
        } else if command == self.cb_cn_symmetric_averaging.text {
            if !self.cb_cn_symmetric_averaging.selected {
                self.sp_cn_symmetric_averaging = 0
            }
        } else {
            self.update_display(false)
        }
    }
    pub fn validate_run(&self) -> bool {
        if self.ltf_directory.is_empty()
            || self.ltf_fn_output.is_empty()
            || self.panels.validate_run().is_some()
        {
            return false;
        }
        let (s, i, e, a) = (
            self.ltf_lst_thresholds_start.is_empty(),
            self.ltf_lst_thresholds_increment.is_empty(),
            self.ltf_lst_thresholds_end.is_empty(),
            self.ltf_lst_thresholds_additional.is_empty(),
        );
        if s && i {
            return !a && e;
        }
        !(s || e)
    }
    pub fn goto_setup_tab(&mut self) {
        self.change_tab_to(Tab::Setup)
    }
    pub fn change_tab_to(&mut self, tab: Tab) {
        self.selected_tab = tab;
        self.change_tab()
    }
    pub fn change_tab(&mut self) {
        self.setup_attached = self.selected_tab == Tab::Setup;
        self.run_attached = self.selected_tab == Tab::Run;
        self.more_options_attached = self.selected_tab == Tab::MoreOptions;
        self.manager.set_method(self.get_processing_method());
        self.packed = true
    }
    pub fn get_volume_table_size(&self) -> usize {
        self.panels.volume_size()
    }
    pub fn update_display(&mut self, init: bool) {
        self.panels.update_display(init, self.init_motl == 4);
        if !self.cb_cn_symmetric_averaging.selected {
            self.sp_cn_symmetric_averaging = 0
        }
    }
    pub fn is_sample_sphere(&self) -> bool {
        !self.panels.sample_sphere_none_selected()
    }
    pub fn add_listeners(&mut self) {
        self.btn_run.listener_count += 1;
        self.btn_avg_vol.listener_count += 1;
        self.btn_ref.listener_count += 1;
        self.btn_average_all.listener_count += 1;
        self.cb_cn_symmetric_averaging.listener_count += 1
    }
    pub fn set_iteration_rows(&mut self) {}
    pub fn set_method(&mut self, m: ProcessingMethod) {
        self.manager.set_method(m)
    }
    pub fn is_use_gpu(&self) -> bool {
        false
    }
    pub fn queue_table_event_action(&mut self) {}
    pub fn set_use_queue_check_box(&mut self) {}
    pub fn add_queue_table_listener(&mut self) {}
    pub fn remove_queue_table_listener(&mut self) {}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct P;
    impl PeetPanels for P {
        fn convert_copied_paths(&mut self, _: &str) {}
        fn is_incorrect_paths(&self) -> bool {
            false
        }
        fn fix_incorrect_paths(&mut self, _: bool) -> bool {
            true
        }
        fn validate_run(&self) -> Option<String> {
            None
        }
        fn volume_size(&self) -> usize {
            2
        }
        fn volume_empty(&self) -> bool {
            false
        }
        fn reference_file_selected(&self) -> bool {
            true
        }
        fn reference_particle_selected(&self) -> bool {
            true
        }
        fn volume_names_templates(&self) -> bool {
            false
        }
        fn sample_sphere_none_selected(&self) -> bool {
            true
        }
        fn update_display(&mut self, _: bool, _: bool) {}
        fn set_defaults(&mut self) {}
    }
    #[derive(Default)]
    struct M;
    impl PeetManager for M {
        fn peet_parser(&mut self, _: ProcessingMethod) {}
        fn imod_avg_vol(&mut self) {}
        fn imod_ref(&mut self) {}
        fn average_all(&mut self) {}
        fn set_method(&mut self, _: ProcessingMethod) {}
    }
    #[test]
    fn validation_uses_setup_and_threshold_source_rules() {
        let mut d = PeetDialog::get_instance(M, P);
        assert!(!d.validate_run());
        d.set_directory("d".into());
        d.set_fn_output("o".into());
        d.ltf_lst_thresholds_additional.text = "2".into();
        assert!(d.validate_run())
    }
    #[test]
    fn run_tab_selects_parallel_cpu() {
        let mut d = PeetDialog::get_instance(M, P);
        d.change_tab_to(Tab::Run);
        assert_eq!(d.get_processing_method(), ProcessingMethod::PpCpu)
    }
}
