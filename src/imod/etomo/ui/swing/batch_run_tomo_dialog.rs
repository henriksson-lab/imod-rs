//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoDialog.java`.
//!
//! Swing construction, the manager, batchruntomo/processchunks invocation,
//! autodoc persistence, and the processor table are explicit boundaries.  This
//! module keeps the dialog-owned selection, validation, tab, queue, and status
//! rules in source order.
#![allow(dead_code)]

use std::path::PathBuf;

use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::check_box::CheckBox;
use super::radio_button::{RadioButton, RadioButtonGroup};

pub const DELIVER_TO_DIRECTORY_LABEL: &str = "Move all stacks to dataset directories under: ";
pub const MAX_GPUS_LABEL: &str = "Max # of GPUs to use by one job: ";
pub const SPLIT_BATCH_DEFAULT_LABEL: &str = "Run multiple batch jobs in parallel";
pub const SPLIT_BATCH_CLUSTER_ONLY_LABEL: &str = "Run multiple batch jobs on cluster";
pub const DEFAULT_CORES: i32 = 4;
pub const MINIMUM_CORES: i32 = 2;
pub const NUMBER_OF_JOBS_TO_MAKE_LABEL: &str = "Run up to ";
pub const DUAL_SELECTION_MAX_DIVISOR: i32 = 4;
pub const USE_SERIES_WATCHER_LABEL: &str = "Watch for stacks to run";
pub const WATCH_DIRECTORY_LABEL: &str = " in: ";
pub const TABLE_LABEL: &str = "Datasets";

/// Java `BatchRunTomoTab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchRunTomoTab {
    Batch,
    Stacks,
    Dataset,
    Run,
}
impl BatchRunTomoTab {
    pub const SIZE: usize = 4;
    pub const DEFAULT: Self = Self::Batch;
    pub fn get_index(self) -> usize {
        match self {
            Self::Batch => 0,
            Self::Stacks => 1,
            Self::Dataset => 2,
            Self::Run => 3,
        }
    }
    pub fn get_instance(index: usize) -> Option<Self> {
        [Self::Batch, Self::Stacks, Self::Dataset, Self::Run]
            .get(index)
            .copied()
    }
}

/// Java `QueueType` / selected queue mode needed by this dialog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueType {
    Queue,
    Node,
    NodeWithoutGpu,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueueTableEvent {
    PreventDisplay,
    AllowDisplay,
    Display,
    Displayed,
    Hidden,
    EnableSecondaryQueue,
    DisableSecondaryQueue,
    OnlyQueueType(QueueType),
    NumberJobsChanged(String),
    QueueSelected {
        queue_type: QueueType,
        maximum: Option<String>,
    },
}

/// Calls crossing from this source unit to Java manager/table/autodoc code.
pub trait BatchRunTomoDialogBoundary {
    fn batchruntomo(&mut self, _method: ProcessingMethod) {}
    fn split_batch(&mut self) {}
    fn resume_batchruntomo(&mut self, _method: ProcessingMethod) {}
    fn pause(&mut self) -> bool {
        true
    }
    fn series_watcher(&mut self, _method: ProcessingMethod) {}
    fn validate_dataset_dialog(&mut self) -> bool {
        true
    }
    fn validate_table(&mut self) -> bool {
        true
    }
    fn update_directives(&mut self, _init: bool, _retain_user_values: bool) {}
    fn tab_changed(&mut self, _tab: BatchRunTomoTab) {}
    fn set_processing_method(
        &mut self,
        _primary: ProcessingMethod,
        _secondary: Option<ProcessingMethod>,
        _run_tab: bool,
    ) {
    }
}

/// Rust metadata exchange equivalent to `BatchRunTomoMetaData`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchRunTomoDialogMetaData {
    pub root_name: String,
    pub root_dir: Option<PathBuf>,
    pub deliver_to_directory: Option<PathBuf>,
    pub input_directive_file: Option<PathBuf>,
    pub use_series_watcher: bool,
    pub watch_directory: Option<PathBuf>,
    pub split_batch: bool,
    pub max_gpus_for_one_job: String,
    pub number_of_jobs_to_make: String,
    pub queue_number_of_jobs_to_make: String,
    pub use_cpu_machine_list: bool,
    pub gpu_machine_list_parallel: Option<bool>,
    pub queue_type: Option<QueueType>,
    pub use_secondary_queue: bool,
    pub status: BatchRunTomoStatus,
}

/// Java `UserConfiguration` fields owned by this dialog.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoUserConfiguration {
    pub use_email_address: bool,
    pub email_address: String,
}

/// Java `BatchruntomoParam` exchange at the command-file boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchruntomoParamBoundary {
    pub deliver_to_directory: Option<PathBuf>,
    pub make_sub_directory: bool,
    pub cpu_machine_list: Option<String>,
    pub gpu_machine_list: Option<String>,
    pub email_address: Option<String>,
    pub multi_proc: i32,
    pub max_gpus_for_one_job: Option<i32>,
}

/// Java `SeriesWatcherParam` exchange at the command-file boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SeriesWatcherParamBoundary {
    pub watch_directory: Option<PathBuf>,
    pub etomo_project_root: String,
    pub parallel_runs: Option<i32>,
}

/// The source's deferred Swing layout and listener registrations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoDialogLayout {
    pub root_created: bool,
    pub batch_created: bool,
    pub stacks_created: bool,
    pub dataset_created: bool,
    pub run_created: bool,
    pub listeners_added: bool,
    pub dataset_table_body_visible: bool,
    pub run_table_body_visible: bool,
    pub pack_count: usize,
    pub context_anchor: Option<String>,
    pub tooltip_count: usize,
}

impl Default for BatchRunTomoDialogMetaData {
    fn default() -> Self {
        Self {
            root_name: String::new(),
            root_dir: None,
            deliver_to_directory: None,
            input_directive_file: None,
            use_series_watcher: false,
            watch_directory: None,
            split_batch: false,
            max_gpus_for_one_job: String::new(),
            number_of_jobs_to_make: String::new(),
            queue_number_of_jobs_to_make: String::new(),
            use_cpu_machine_list: false,
            gpu_machine_list_parallel: None,
            queue_type: None,
            use_secondary_queue: false,
            status: BatchRunTomoStatus::DEFAULT,
        }
    }
}

/// Java `BatchRunTomoDialog` source-visible state, with Swing widgets represented
/// by their model values.
#[derive(Clone, Debug)]
pub struct BatchRunTomoDialog {
    pub layout: BatchRunTomoDialogLayout,
    pub root_name: String,
    pub root_dir: Option<PathBuf>,
    pub input_directive_file: Option<PathBuf>,
    pub deliver_to_directory: Option<PathBuf>,
    pub watch_directory: Option<PathBuf>,
    pub email_address: String,
    pub rb_deliver_off: RadioButton,
    pub rb_deliver_to_directory: RadioButton,
    pub rb_deliver_make_sub_directory: RadioButton,
    pub cb_cpu_machine_list: CheckBox,
    pub rb_gpu_machine_list_off: RadioButton,
    pub rb_gpu_machine_list_local: RadioButton,
    pub rb_gpu_machine_list: RadioButton,
    pub cb_split_batch: CheckBox,
    pub cb_use_series_watcher: CheckBox,
    pub queue_type_queue: bool,
    pub queue_type_node: bool,
    pub queue_secondary_queue: bool,
    pub total_cpus: i32,
    pub total_gpus: i32,
    pub number_queue_cpus_available: i32,
    pub queues_available: bool,
    pub local_gpu_available: bool,
    pub gpu_available: bool,
    pub queue_type_queue_enabled: bool,
    pub queue_type_node_without_gpu_available: bool,
    pub queue_type_node_with_gpu_available: bool,
    pub secondary_queues: bool,
    pub number_of_jobs_to_make: i32,
    pub queue_number_of_jobs_to_make: i32,
    pub queue_number_of_jobs_maximum: i32,
    pub max_gpus_for_one_job: i32,
    pub cur_tab: Option<BatchRunTomoTab>,
    pub tab_displayed: [bool; BatchRunTomoTab::SIZE],
    pub queue_table_displayed: bool,
    pub queue_mode: Option<QueueType>,
    pub status: BatchRunTomoStatus,
    pub killed_paused: bool,
    pub dataset_table_visible: bool,
    pub run_buttons_visible: bool,
    pub series_watcher_run_buttons_visible: bool,
    pub pause_enabled: bool,
    pub resume_enabled: bool,
    pub parallel_resume_enabled: bool,
    pub reset_enabled: bool,
    pub run_enabled: bool,
    pub start_series_watcher_enabled: bool,
    pub finish_series_watcher_enabled: bool,
    pub input_directive_file_checkpoint: Option<PathBuf>,
    pub queue_table_events: Vec<QueueTableEvent>,
    pub root_fields_editable: bool,
    pub delivery_fields_editable: bool,
    pub advanced_starting_batch: Option<Vec<(String, String)>>,
    pub status_listener_count: usize,
    pub queue_table_listener_count: usize,
}

impl BatchRunTomoDialog {
    /// Java private constructor plus `getInstance` construction defaults.
    pub fn new(total_cpus: i32, total_gpus: i32, queues_available: bool) -> Self {
        let deliver = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let gpu = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let mut rb_deliver_off =
            RadioButton::new_in_group("Stacks are already in dataset directories", deliver.clone());
        rb_deliver_off.set_selected(true);
        let mut rb_gpu_machine_list = RadioButton::new_in_group("Parallel GPUs", gpu.clone());
        rb_gpu_machine_list.set_selected(true);
        let mut value = Self {
            layout: BatchRunTomoDialogLayout::default(),
            root_name: String::new(),
            root_dir: None,
            input_directive_file: None,
            deliver_to_directory: None,
            watch_directory: None,
            email_address: String::new(),
            rb_deliver_off,
            rb_deliver_to_directory: RadioButton::new_in_group(
                DELIVER_TO_DIRECTORY_LABEL,
                deliver.clone(),
            ),
            rb_deliver_make_sub_directory: RadioButton::new_in_group(
                "Move stacks to dataset directories under their current locations",
                deliver,
            ),
            cb_cpu_machine_list: CheckBox::new_with_text("Use multiple cores"),
            rb_gpu_machine_list_off: RadioButton::new_in_group("No GPU", gpu.clone()),
            rb_gpu_machine_list_local: RadioButton::new_in_group("Local GPU", gpu),
            rb_gpu_machine_list,
            cb_split_batch: CheckBox::new_with_text(SPLIT_BATCH_DEFAULT_LABEL),
            cb_use_series_watcher: CheckBox::new_with_text(USE_SERIES_WATCHER_LABEL),
            queue_type_queue: queues_available,
            queue_type_node: false,
            queue_secondary_queue: false,
            total_cpus,
            total_gpus,
            number_queue_cpus_available: 0,
            queues_available,
            local_gpu_available: total_gpus > 0,
            gpu_available: total_gpus > 0,
            queue_type_queue_enabled: queues_available,
            queue_type_node_without_gpu_available: false,
            queue_type_node_with_gpu_available: false,
            secondary_queues: false,
            number_of_jobs_to_make: DEFAULT_CORES.min(total_cpus),
            queue_number_of_jobs_to_make: DEFAULT_CORES,
            queue_number_of_jobs_maximum: DEFAULT_CORES,
            max_gpus_for_one_job: DEFAULT_CORES.min(total_gpus),
            cur_tab: None,
            tab_displayed: [false; BatchRunTomoTab::SIZE],
            queue_table_displayed: false,
            queue_mode: None,
            status: BatchRunTomoStatus::DEFAULT,
            killed_paused: false,
            dataset_table_visible: true,
            run_buttons_visible: true,
            series_watcher_run_buttons_visible: false,
            pause_enabled: false,
            resume_enabled: false,
            parallel_resume_enabled: false,
            reset_enabled: true,
            run_enabled: true,
            start_series_watcher_enabled: true,
            finish_series_watcher_enabled: false,
            input_directive_file_checkpoint: None,
            queue_table_events: Vec::new(),
            root_fields_editable: true,
            delivery_fields_editable: true,
            advanced_starting_batch: None,
            status_listener_count: 0,
            queue_table_listener_count: 0,
        };
        value.cb_cpu_machine_list.set_selected(true);
        value.create_panel();
        value
    }

    pub fn get_dialog_type(&self) -> &'static str {
        "BATCH_RUN_TOMO"
    }
    pub fn create_panel(&mut self) {
        self.layout.root_created = true;
        self.layout.batch_created = true;
        self.layout.stacks_created = true;
        self.layout.dataset_created = true;
        self.layout.run_created = true;
        self.state_changed(None);
        self.update_display(false);
    }
    pub fn retrieve_screen_state_from_dialog(&self) -> (bool, bool) {
        (self.dataset_table_visible, self.queue_table_displayed)
    }
    pub fn apply_screen_state_to_dialog(&mut self, state: (bool, bool)) {
        self.dataset_table_visible = state.0;
        self.queue_table_displayed = state.1;
    }
    pub fn build_run_settings_panel(&mut self) {
        self.series_watcher_run_buttons_visible = self.cb_use_series_watcher.is_selected();
        self.run_buttons_visible = !self.series_watcher_run_buttons_visible;
    }
    pub fn add_listeners(&mut self) {}
    pub fn add_status_change_listener(&mut self, listener_present: bool) {
        if listener_present {
            self.status_listener_count += 1;
        }
    }
    pub fn pop_up_context_menu(&mut self) {
        self.layout.context_anchor = self.cur_tab.map(|tab| match tab {
            BatchRunTomoTab::Batch => "BatchSetup".into(),
            BatchRunTomoTab::Stacks => "Stacks".into(),
            BatchRunTomoTab::Dataset => "SetValues".into(),
            BatchRunTomoTab::Run => "Run".into(),
        });
    }
    pub fn get_global_directives_dialog(&self) -> bool {
        true
    }
    pub fn msg_status_changer_started(&mut self, table_only: bool) {
        if !table_only {
            self.status_listener_count += 2;
        }
    }
    pub fn get_container(&self) -> &BatchRunTomoDialogLayout {
        &self.layout
    }
    pub fn msg_load_done(&mut self) {
        self.add_listeners();
        self.layout.listeners_added = true;
    }
    pub fn get_browsing_dir(&self) -> Option<PathBuf> {
        if self.rb_deliver_off.is_selected() {
            self.root_dir
                .clone()
                .and_then(|path| path.parent().map(PathBuf::from))
        } else {
            self.root_dir.clone()
        }
    }
    pub fn set_browsing_dir(&mut self, input: Option<PathBuf>) {
        self.root_dir = input;
    }
    pub fn set_browsing_dir_string(&mut self, input: Option<&str>) {
        if let Some(input) = input.filter(|input| !input.trim().is_empty()) {
            self.root_dir = Some(PathBuf::from(input));
        }
    }
    pub fn set_parameters(&mut self, metadata: &BatchRunTomoDialogMetaData) {
        self.root_name = metadata.root_name.clone();
        self.root_dir = metadata.root_dir.clone();
        self.deliver_to_directory = metadata.deliver_to_directory.clone();
        self.input_directive_file = metadata.input_directive_file.clone();
        self.cb_use_series_watcher
            .set_selected(metadata.use_series_watcher);
        self.watch_directory = metadata.watch_directory.clone();
        self.cb_split_batch.set_selected(metadata.split_batch);
        self.max_gpus_for_one_job = metadata
            .max_gpus_for_one_job
            .parse()
            .unwrap_or(self.max_gpus_for_one_job);
        self.number_of_jobs_to_make = metadata
            .number_of_jobs_to_make
            .parse()
            .unwrap_or(self.number_of_jobs_to_make);
        self.queue_number_of_jobs_to_make = metadata
            .queue_number_of_jobs_to_make
            .parse()
            .unwrap_or(self.queue_number_of_jobs_to_make);
        self.cb_cpu_machine_list
            .set_selected(metadata.use_cpu_machine_list);
        self.rb_gpu_machine_list_off
            .set_selected(metadata.gpu_machine_list_parallel.is_none());
        self.rb_gpu_machine_list_local
            .set_selected(metadata.gpu_machine_list_parallel == Some(false));
        self.rb_gpu_machine_list
            .set_selected(metadata.gpu_machine_list_parallel == Some(true));
        self.queue_type_queue = metadata.queue_type == Some(QueueType::Queue);
        self.queue_type_node = metadata.queue_type == Some(QueueType::Node);
        self.queue_secondary_queue = metadata.use_secondary_queue;
        self.status_changed(metadata.status);
        self.build_run_settings_panel();
        self.update_display(false);
    }
    pub fn get_parameters(&self) -> BatchRunTomoDialogMetaData {
        BatchRunTomoDialogMetaData {
            root_name: self.root_name.clone(),
            root_dir: self.root_dir.clone(),
            deliver_to_directory: self.deliver_to_directory.clone(),
            input_directive_file: self.input_directive_file.clone(),
            use_series_watcher: self.cb_use_series_watcher.is_selected(),
            watch_directory: self.watch_directory.clone(),
            split_batch: self.cb_split_batch.is_selected(),
            max_gpus_for_one_job: self.max_gpus_for_one_job.to_string(),
            number_of_jobs_to_make: self.number_of_jobs_to_make.to_string(),
            queue_number_of_jobs_to_make: self.queue_number_of_jobs_to_make.to_string(),
            use_cpu_machine_list: self.cb_cpu_machine_list.is_selected(),
            gpu_machine_list_parallel: if self.rb_gpu_machine_list_local.is_selected() {
                Some(false)
            } else if self.rb_gpu_machine_list.is_selected() {
                Some(true)
            } else {
                None
            },
            queue_type: if self.queue_type_queue {
                Some(QueueType::Queue)
            } else if self.queue_type_node {
                Some(QueueType::Node)
            } else {
                None
            },
            use_secondary_queue: self.queue_secondary_queue,
            status: self.status,
        }
    }
    pub fn is_param_file_modifiable(&self) -> bool {
        self.root_fields_editable
    }
    pub fn is_param_file_empty(&self) -> bool {
        self.root_name.is_empty() || self.root_dir.is_none()
    }
    pub fn disable_root_fields(&mut self) {
        self.root_fields_editable = false;
    }
    pub fn disable_delivery_fields(&mut self) {
        self.delivery_fields_editable = false;
    }
    pub fn get_parameters_user_configuration(
        &self,
        user_configuration: &mut BatchRunTomoUserConfiguration,
    ) {
        user_configuration.use_email_address = !self.email_address.is_empty();
        user_configuration.email_address = self.email_address.clone();
    }
    pub fn set_parameters_environment(&mut self, parallel: bool, gpu: bool) {
        self.cb_cpu_machine_list.set_selected(parallel);
        self.rb_gpu_machine_list_local.set_selected(gpu);
        self.rb_gpu_machine_list_off.set_selected(!gpu);
    }
    pub fn set_parameters_user_configuration(
        &mut self,
        user_configuration: &BatchRunTomoUserConfiguration,
    ) {
        self.email_address = if user_configuration.use_email_address {
            user_configuration.email_address.clone()
        } else {
            String::new()
        };
    }
    pub fn get_parameters_parallel(&self, param: &mut BatchruntomoParamBoundary) {
        param.multi_proc = if self.cb_split_batch.is_selected() {
            if self.queue_table_displayed {
                self.queue_number_of_jobs_to_make
            } else {
                self.number_of_jobs_to_make
            }
        } else {
            1
        };
        param.gpu_machine_list = if self.rb_gpu_machine_list_off.is_selected() {
            None
        } else if self.rb_gpu_machine_list_local.is_selected() {
            Some("local".into())
        } else {
            param.gpu_machine_list.clone()
        };
    }
    pub fn get_parameters_split_batch(&self, param: &mut BatchruntomoParamBoundary) {
        param.max_gpus_for_one_job = if self.rb_gpu_machine_list.is_selected() {
            Some(self.max_gpus_for_one_job)
        } else if self.rb_gpu_machine_list_local.is_selected() {
            Some(1)
        } else {
            None
        };
    }
    pub fn set_parameters_batchruntomo(&mut self, param: &BatchruntomoParamBoundary) {
        self.rb_deliver_off.set_selected(true);
        self.rb_deliver_to_directory
            .set_selected(param.deliver_to_directory.is_some());
        self.deliver_to_directory = param.deliver_to_directory.clone();
        self.rb_deliver_make_sub_directory
            .set_selected(param.make_sub_directory);
        self.cb_cpu_machine_list
            .set_selected(param.cpu_machine_list.is_none());
        self.rb_gpu_machine_list_off
            .set_selected(param.gpu_machine_list.is_none());
        self.rb_gpu_machine_list_local
            .set_selected(param.gpu_machine_list.as_deref() == Some("local"));
        self.rb_gpu_machine_list.set_selected(
            param
                .gpu_machine_list
                .as_deref()
                .is_some_and(|value| value != "local"),
        );
        self.email_address = param.email_address.clone().unwrap_or_default();
        self.update_display(false);
    }
    pub fn set_parameters_series_watcher(&mut self, param: &SeriesWatcherParamBoundary) {
        self.watch_directory = param.watch_directory.clone();
    }
    pub fn is_parallel_processing(&self) -> bool {
        self.cb_split_batch.is_selected() && self.total_cpus > 1
    }
    pub fn validate_batch_run_tomo_param(&self) -> bool {
        !self.rb_deliver_to_directory.is_selected() || self.deliver_to_directory.is_some()
    }
    pub fn validate(&self) -> bool {
        self.validate_batch_run_tomo_param()
    }
    pub fn get_parameters_batchruntomo(
        &mut self,
        param: &mut BatchruntomoParamBoundary,
        do_validation: bool,
        for_update: bool,
        validate_only: bool,
    ) -> bool {
        if do_validation && !self.validate_batch_run_tomo_param() {
            return false;
        }
        if !for_update {
            param.deliver_to_directory = if self.rb_deliver_to_directory.is_selected() {
                self.deliver_to_directory.clone()
            } else {
                None
            };
            param.make_sub_directory = self.rb_deliver_make_sub_directory.is_selected();
            param.email_address =
                (!self.email_address.is_empty()).then(|| self.email_address.clone());
        }
        if !self.cb_cpu_machine_list.is_selected() {
            param.cpu_machine_list = Some("local".into());
        }
        if self.rb_gpu_machine_list_off.is_selected() {
            param.gpu_machine_list = None;
        } else if self.rb_gpu_machine_list_local.is_selected() {
            param.gpu_machine_list = Some("local".into());
        }
        if do_validation && !validate_only {
            self.disable_delivery_fields();
        }
        true
    }
    pub fn get_parameters_series_watcher(
        &self,
        param: &mut SeriesWatcherParamBoundary,
        do_validation: bool,
    ) -> bool {
        if do_validation && self.watch_directory.is_none() {
            return false;
        }
        param.watch_directory = self.watch_directory.clone();
        param.etomo_project_root = self.root_name.clone();
        param.parallel_runs = (self.cb_use_series_watcher.is_selected()
            && self.cb_split_batch.is_selected())
        .then_some(if self.queue_table_displayed {
            self.queue_number_of_jobs_to_make
        } else {
            self.number_of_jobs_to_make
        });
        true
    }
    pub fn load_templates(&mut self) {}
    pub fn load_autodocs(
        &mut self,
        _only_stack_id_dataset_dialog: Option<&str>,
        _only_advanced_dataset_dialog: bool,
    ) {
    }
    pub fn save_autodocs(
        &mut self,
        _do_validation: bool,
        _init: bool,
        _autodoc_stack_id: Option<&str>,
        _only_global_autodoc: bool,
        _validate_only: bool,
    ) -> bool {
        true
    }
    pub fn get_input_directive_autodoc(&self) -> Option<&PathBuf> {
        self.input_directive_file
            .as_ref()
            .filter(|file| file.exists())
    }
    pub fn validate_input_directive_file(&self) -> bool {
        self.input_directive_file
            .as_ref()
            .is_none_or(|file| file.exists())
    }
    pub fn get_advanced_starting_batch(&self) -> Option<&Vec<(String, String)>> {
        self.advanced_starting_batch.as_ref()
    }
    pub fn get_dataset_image_filename_style(&self) -> Option<&'static str> {
        None
    }
    pub fn update_directives<B: BatchRunTomoDialogBoundary>(
        &mut self,
        boundary: &mut B,
        init: bool,
        retain_user_values: bool,
    ) {
        boundary.update_directives(init, retain_user_values);
    }
    pub fn action_performed<B: BatchRunTomoDialogBoundary>(
        &mut self,
        boundary: &mut B,
        action_command: &str,
    ) {
        match action_command {
            "Run" => {
                if self.validate()
                    && boundary.validate_dataset_dialog()
                    && boundary.validate_table()
                {
                    if self.is_parallel_processing() {
                        boundary.split_batch();
                    } else {
                        boundary.batchruntomo(self.get_processing_method());
                    }
                }
            }
            "Resume" => boundary.resume_batchruntomo(self.get_processing_method()),
            "Reset" | "Reset Watching" => self.start_over(),
            "Pause" | "Finish" => {
                self.update_display(!boundary.pause());
            }
            "Start Watching" => boundary.series_watcher(self.get_processing_method()),
            "Clear" => {
                self.input_directive_file = None;
                self.input_directive_file_checkpoint = None;
            }
            _ => {
                self.update_display(false);
                boundary.set_processing_method(
                    self.get_processing_method(),
                    self.get_secondary_processing_method(),
                    self.cur_tab == Some(BatchRunTomoTab::Run),
                );
            }
        }
    }
    pub fn equals_series_watcher_action_command(&self, action_command: Option<&str>) -> bool {
        action_command == Some(USE_SERIES_WATCHER_LABEL)
    }
    pub fn get_split_batch_queue_table_event(&self) -> QueueTableEvent {
        if self.cb_split_batch.is_selected() {
            if self.cb_cpu_machine_list.is_selected() {
                QueueTableEvent::AllowDisplay
            } else {
                QueueTableEvent::Display
            }
        } else {
            QueueTableEvent::PreventDisplay
        }
    }
    pub fn get_only_queue_type_queue_table_event(&self) -> Option<QueueTableEvent> {
        if self.queue_type_queue {
            Some(QueueTableEvent::OnlyQueueType(QueueType::Queue))
        } else if self.queue_type_node {
            Some(QueueTableEvent::OnlyQueueType(
                if self.queue_secondary_queue {
                    QueueType::NodeWithoutGpu
                } else {
                    QueueType::Node
                },
            ))
        } else {
            None
        }
    }
    pub fn get_secondary_queue_table_event(&self) -> Option<QueueTableEvent> {
        self.queues_available
            .then_some(if self.queue_secondary_queue {
                QueueTableEvent::EnableSecondaryQueue
            } else {
                QueueTableEvent::DisableSecondaryQueue
            })
    }
    pub fn get_number_jobs_changed_queue_table_event(&self) -> QueueTableEvent {
        QueueTableEvent::NumberJobsChanged(self.queue_number_of_jobs_to_make.to_string())
    }
    pub fn send_queue_table_event(&mut self, event: Option<QueueTableEvent>) {
        if let Some(event) = event {
            self.queue_table_events.push(event);
        }
    }
    pub fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        match event {
            QueueTableEvent::QueueSelected {
                queue_type,
                maximum,
            } => {
                self.queue_mode = Some(queue_type);
                if let Some(maximum) = maximum.and_then(|s| s.parse::<i32>().ok()) {
                    self.queue_number_of_jobs_maximum = if queue_type == QueueType::Queue {
                        maximum / DUAL_SELECTION_MAX_DIVISOR
                    } else {
                        maximum
                    };
                    self.queue_number_of_jobs_to_make = self
                        .queue_number_of_jobs_to_make
                        .min(self.queue_number_of_jobs_maximum)
                        .max(1);
                }
            }
            QueueTableEvent::Displayed => self.queue_table_displayed = true,
            QueueTableEvent::Hidden => self.queue_table_displayed = false,
            _ => {}
        }
        self.update_display(false);
    }
    pub fn start_over(&mut self) {
        self.status_changed(BatchRunTomoStatus::Open);
    }
    pub fn create_run_list(&self, run_type: Option<&str>) -> Vec<String> {
        run_type
            .map(|value| vec![value.to_owned()])
            .unwrap_or_default()
    }
    pub fn find_row(&self, _location: &str, _root_name: &str) -> Option<String> {
        None
    }
    pub fn get_stack(&self, _stack_id: &str) -> Option<PathBuf> {
        None
    }
    pub fn lock_processing_method(&mut self, _lock: bool) {}
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.queue_table_displayed {
            ProcessingMethod::Queue
        } else if (self.cb_split_batch.is_selected() && self.cb_split_batch.is_enabled())
            || (self.cb_cpu_machine_list.is_selected() && self.cb_cpu_machine_list.is_enabled())
        {
            ProcessingMethod::PpCpu
        } else if self.rb_gpu_machine_list.is_selected() && self.rb_gpu_machine_list.is_enabled() {
            ProcessingMethod::PpGpu
        } else if self.rb_gpu_machine_list_local.is_selected()
            && self.rb_gpu_machine_list_local.is_enabled()
        {
            ProcessingMethod::LocalGpu
        } else {
            ProcessingMethod::LocalCpu
        }
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        if self.get_processing_method() != ProcessingMethod::PpCpu {
            None
        } else if self.rb_gpu_machine_list.is_selected() && self.rb_gpu_machine_list.is_enabled() {
            Some(ProcessingMethod::PpGpu)
        } else if self.rb_gpu_machine_list_local.is_selected()
            && self.rb_gpu_machine_list_local.is_enabled()
        {
            Some(ProcessingMethod::LocalGpu)
        } else {
            None
        }
    }
    pub fn get_root_name(&self) -> &str {
        &self.root_name
    }
    pub fn get_dataset_name(&self) -> &str {
        &self.root_name
    }
    pub fn get_root_dir(&self) -> Option<&PathBuf> {
        self.root_dir.as_ref()
    }
    pub fn get_dataset_absolute_path(&self) -> Option<String> {
        self.root_dir
            .as_ref()
            .map(|path| path.display().to_string())
    }
    pub fn is_series_watcher_on(&self) -> bool {
        self.cb_use_series_watcher.is_selected()
    }
    pub fn process_result(&mut self, input_directive_changed: bool) {
        if input_directive_changed
            && self.input_directive_file != self.input_directive_file_checkpoint
            && self.validate_input_directive_file()
        {
            self.input_directive_file_checkpoint = self.input_directive_file.clone();
        }
    }
    pub fn display(&mut self) {
        self.display_tab(BatchRunTomoTab::Batch);
    }
    pub fn display_tab(&mut self, tab: BatchRunTomoTab) {
        if self.cur_tab != Some(tab) {
            self.cur_tab = Some(tab);
            self.state_changed(None);
        }
    }
    pub fn set_dataset_table_visible(&mut self, visible: bool) {
        self.dataset_table_visible = visible;
    }
    pub fn expand_dataset_table(&mut self, expanded: bool) {
        self.layout.dataset_table_body_visible = expanded;
    }
    pub fn expand_run_table(&mut self, expanded: bool) {
        self.layout.run_table_body_visible = expanded;
    }
    pub fn pack(&mut self) {
        self.layout.pack_count += 1;
    }
    pub fn expand_global(&mut self) {}
    pub fn display_component(&mut self) {
        self.display();
    }
    pub fn focus_gained(&mut self) {}
    pub fn focus_lost(&mut self, queue_spinner: bool) {
        if queue_spinner {
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
        }
    }
    pub fn state_changed(&mut self, spinner_changed: Option<bool>) {
        if spinner_changed == Some(true) {
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
            return;
        }
        let tab = self.cur_tab.unwrap_or(BatchRunTomoTab::DEFAULT);
        self.cur_tab = Some(tab);
        self.tab_displayed[tab.get_index()] = true;
    }
    pub fn update_display(&mut self, paused_failed: bool) {
        let series_watcher = self.cb_use_series_watcher.is_selected();
        if series_watcher {
            self.cb_use_series_watcher
                .set_text(Some("Watch for stacks to run in: "));
        } else {
            self.cb_use_series_watcher
                .set_text(Some(USE_SERIES_WATCHER_LABEL));
        }
        self.run_buttons_visible = !series_watcher;
        self.series_watcher_run_buttons_visible = series_watcher;
        let split = self.cb_split_batch.is_selected()
            && (self.total_cpus > 1
                || self.queues_available && self.number_queue_cpus_available > 1);
        self.cb_split_batch.set_enabled(
            self.total_cpus > 1 || self.queues_available && self.number_queue_cpus_available > 1,
        );
        self.cb_split_batch
            .set_text(Some(if self.queue_table_displayed {
                SPLIT_BATCH_CLUSTER_ONLY_LABEL
            } else {
                SPLIT_BATCH_DEFAULT_LABEL
            }));
        self.cb_cpu_machine_list.set_enabled(!split);
        self.rb_gpu_machine_list_off
            .set_enabled(!self.queue_table_displayed);
        self.rb_gpu_machine_list_local
            .set_enabled(!self.queue_table_displayed && self.local_gpu_available);
        self.rb_gpu_machine_list
            .set_enabled(!self.queue_table_displayed && self.gpu_available);
        let run = matches!(
            self.status,
            BatchRunTomoStatus::Open
                | BatchRunTomoStatus::Done
                | BatchRunTomoStatus::Stopped
                | BatchRunTomoStatus::Failed
        );
        self.run_enabled = run;
        self.start_series_watcher_enabled = run;
        self.pause_enabled = self.status == BatchRunTomoStatus::Running;
        self.finish_series_watcher_enabled = self.pause_enabled;
        self.resume_enabled = matches!(
            self.status,
            BatchRunTomoStatus::KilledOrPaused | BatchRunTomoStatus::KilledOrPausedSeriesWatcher
        );
        self.parallel_resume_enabled =
            self.status == BatchRunTomoStatus::KilledOrPausedProcessChunks;
        self.reset_enabled =
            self.status == BatchRunTomoStatus::Open || self.status.is_end_status() || paused_failed;
    }
    pub fn status_changed(&mut self, status: BatchRunTomoStatus) {
        self.status =
            BatchRunTomoStatus::get_instance(Some(self.status), Some(status)).unwrap_or(status);
        if matches!(
            self.status,
            BatchRunTomoStatus::Running | BatchRunTomoStatus::Open
        ) {
            self.killed_paused = false;
        }
        if !self.killed_paused {
            self.killed_paused = matches!(
                self.status,
                BatchRunTomoStatus::KilledOrPaused
                    | BatchRunTomoStatus::KilledOrPausedProcessChunks
            );
        }
        self.update_display(false);
    }
    pub fn status_changed_event(&mut self, status: Option<BatchRunTomoStatus>) {
        if let Some(status) = status {
            self.status_changed(status);
        }
    }
    pub fn send_status_changed(&mut self, status: BatchRunTomoStatus) {
        self.status_changed(status);
    }
    pub fn set_tooltips(&mut self) {
        self.layout.tooltip_count = 17;
    }
    pub fn set_method<B: BatchRunTomoDialogBoundary>(
        &self,
        boundary: &mut B,
        processing_method: ProcessingMethod,
    ) {
        boundary.set_processing_method(processing_method, None, false);
    }
    pub fn is_use_gpu(&self) -> bool {
        true
    }
    pub fn set_use_queue_check_box(&mut self, _use_queue_check_box_present: bool) {}
    pub fn add_queue_table_listener(&mut self, listener_present: bool) {
        if listener_present {
            self.queue_table_listener_count += 1;
            self.send_queue_table_event(Some(self.get_split_batch_queue_table_event()));
            self.send_queue_table_event(self.get_only_queue_type_queue_table_event());
            self.send_queue_table_event(self.get_secondary_queue_table_event());
            self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
        }
    }
    pub fn send_queue_table_events(&mut self) {
        self.send_queue_table_event(Some(self.get_split_batch_queue_table_event()));
        self.send_queue_table_event(self.get_only_queue_type_queue_table_event());
        self.send_queue_table_event(self.get_secondary_queue_table_event());
        self.send_queue_table_event(Some(self.get_number_jobs_changed_queue_table_event()));
    }
    pub fn remove_queue_table_listener(&mut self, listener_present: bool) {
        if listener_present {
            self.queue_table_listener_count = self.queue_table_listener_count.saturating_sub(1);
        }
    }
    pub fn update_gpu(&mut self, _disable_gpu: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Boundary {
        calls: Vec<String>,
    }
    impl BatchRunTomoDialogBoundary for Boundary {
        fn batchruntomo(&mut self, method: ProcessingMethod) {
            self.calls.push(format!("run:{method}"));
        }
        fn split_batch(&mut self) {
            self.calls.push("split".into());
        }
        fn pause(&mut self) -> bool {
            self.calls.push("pause".into());
            true
        }
    }
    #[test]
    fn source_default_and_processing_method() {
        let dialog = BatchRunTomoDialog::new(8, 2, false);
        assert_eq!(dialog.get_processing_method(), ProcessingMethod::PpCpu);
        assert_eq!(dialog.cur_tab, Some(BatchRunTomoTab::Batch));
    }
    #[test]
    fn run_uses_split_batch_when_selected() {
        let mut dialog = BatchRunTomoDialog::new(8, 0, false);
        let mut boundary = Boundary::default();
        dialog.cb_split_batch.set_selected(true);
        dialog.action_performed(&mut boundary, "Run");
        assert_eq!(boundary.calls, ["split"]);
    }
    #[test]
    fn queue_maximum_prevents_single_cpu_deadlock() {
        let mut dialog = BatchRunTomoDialog::new(8, 0, true);
        dialog.queue_number_of_jobs_to_make = 8;
        dialog.queue_table_event_action(QueueTableEvent::QueueSelected {
            queue_type: QueueType::Queue,
            maximum: Some("12".into()),
        });
        assert_eq!(dialog.queue_number_of_jobs_maximum, 3);
        assert_eq!(dialog.queue_number_of_jobs_to_make, 3);
    }
    #[test]
    fn status_controls_run_and_resume_buttons() {
        let mut dialog = BatchRunTomoDialog::new(4, 0, false);
        dialog.status_changed(BatchRunTomoStatus::Running);
        assert!(!dialog.run_enabled);
        assert!(dialog.pause_enabled);
        dialog.status_changed(BatchRunTomoStatus::KilledOrPaused);
        assert!(dialog.resume_enabled);
    }
    #[test]
    fn command_parameter_exchange_preserves_delivery_and_parallel_gpu_rules() {
        let mut dialog = BatchRunTomoDialog::new(8, 4, false);
        dialog.rb_deliver_to_directory.set_selected(true);
        dialog.deliver_to_directory = Some(PathBuf::from("/data/delivered"));
        dialog.rb_gpu_machine_list_local.set_selected(true);
        let mut param = BatchruntomoParamBoundary::default();
        assert!(dialog.get_parameters_batchruntomo(&mut param, true, false, false));
        dialog.get_parameters_parallel(&mut param);
        dialog.get_parameters_split_batch(&mut param);
        assert_eq!(
            param.deliver_to_directory,
            Some(PathBuf::from("/data/delivered"))
        );
        assert_eq!(param.gpu_machine_list.as_deref(), Some("local"));
        assert_eq!(param.max_gpus_for_one_job, Some(1));
    }
    #[test]
    fn run_does_not_cross_manager_boundary_when_source_validation_fails() {
        let mut dialog = BatchRunTomoDialog::new(8, 0, false);
        let mut boundary = Boundary::default();
        dialog.rb_deliver_to_directory.set_selected(true);
        dialog.action_performed(&mut boundary, "Run");
        assert!(boundary.calls.is_empty());
    }
}
