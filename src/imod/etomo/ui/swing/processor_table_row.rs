//! `IMOD/Etomo/src/etomo/ui/swing/ProcessorTableRow.java`.
//!
//! Swing cells and the process-parameter objects are deliberately represented as
//! state at their native boundaries.  The row decisions themselves are retained
//! here, one Rust method per Java operation, so CPU/queue/GPU selection does not
//! get folded into `ProcessorTable`.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::processor_table::ColumnName;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;

pub const SECONDARY_QUEUE: &str = "SecondaryQueue";
pub const STORE_SELECTED: &str = "Selected";
pub const STORE_CPUS_SELECTED: &str = "CPUsSelected";
pub const STORE_GPUS_SELECTED: &str = "MaxGPUJobsOnQueue";
pub const DEFAULT_CPUS_SELECTED: i32 = 1;
pub const DUAL_SELECTION_MIN: i32 = 2;
pub const DROP_VALUE: i32 = 3;

/// Values consumed from Java `Node`.  `Node.java` remains the storage source
/// unit; this is its direct row-facing projection.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessorNode {
    pub name: String,
    pub num_cpus: i32,
    pub memory: Option<String>,
    pub os: Option<String>,
    pub gpu_device_array: Vec<String>,
    pub is_gpu: bool,
    /// Java `Node.isGpuLocal()` value.
    pub is_gpu_local: bool,
    /// Result of Java `Node.isLocalHost(manager, axisID, propertyUserDir)` at
    /// the storage/manager boundary used by `GpuTable.isExcludeNode`.
    pub is_local_host: bool,
    pub gpus_per_cluster_job: Option<String>,
    pub cpu_type: Option<String>,
    pub speed: Option<String>,
    pub gpu_type: Option<String>,
    pub gpu_speed: Option<String>,
    pub gpu_memory: Option<String>,
    pub gpu_ncores: Option<String>,
    pub queue_mode: QueueMode,
    pub secondary_queue: bool,
}

/// Java `QueueMode` values the row asks about.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueueMode {
    #[default]
    Invalid,
    QueueWithSingleCpu,
    Queue,
    Node,
}
impl QueueMode {
    pub fn is_type(self, queue_type: QueueType) -> bool {
        matches!(
            (self, queue_type),
            (Self::QueueWithSingleCpu, QueueType::Queue)
                | (Self::Queue, QueueType::Queue)
                | (Self::Node, QueueType::Node | QueueType::NodeWithoutGpu)
        )
    }
}

/// Java `QueueType` values used in `setHeader1NumberCPUsTitle`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueType {
    Queue,
    Node,
    NodeWithoutGpu,
}

/// `ProcesschunksParam` calls made by `getParameters` are a direct comscript
/// boundary.  Capturing them preserves their order and values for its adapter.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcesschunksParameters {
    pub gpu_processing: bool,
    pub queue_mode: Option<QueueMode>,
    pub queue: Option<String>,
    pub machine_names: Vec<(String, i32, Vec<String>)>,
    pub gpu_machines: Vec<(String, i32, Vec<String>)>,
    pub secondary_number: Option<String>,
}

/// `BatchruntomoParam` calls made by this row at the direct comscript boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchruntomoParameters {
    pub queue_mode: Option<QueueMode>,
    pub max_jobs_on_queue: Option<i32>,
    pub max_gpu_jobs_on_queue: Option<String>,
    pub cpu_machines: Vec<(String, i32)>,
    pub gpu_machines: Vec<(String, i32, Vec<String>)>,
}

/// Source-owned state of `ProcessorTableRow` cells.  The actual ToggleCell,
/// FieldCell, SpinnerCell, GridBag and Swing listener instances are native GUI
/// boundaries, while every value used by row logic is retained here.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessorTableRow {
    pub computer: String,
    pub selected: bool,
    pub secondary_queue_selected: bool,
    pub has_secondary_queue: bool,
    pub selection_enabled: bool,
    /// Java `cellComputer.isEnabled()`, distinct from the stored
    /// `selectionEnabled` policy field.
    pub computer_enabled: bool,
    pub cpus_selected: i32,
    pub successes: i32,
    pub restarts: i32,
    pub failure_reason: Option<String>,
    pub load: Vec<String>,
    pub cpu_usage: Option<String>,
    pub displayed: bool,
    pub display_queues: bool,
    pub num_rows_in_table: usize,
    pub gpu_device_array: Vec<String>,
    pub dual_selection_queue_table: bool,
    pub queue_mode: QueueMode,
    pub secondary_queue: bool,
    pub gpus_per_cluster_job: Option<String>,
    pub num_cpus: i32,
    pub memory: Option<String>,
    pub os: Option<String>,
    pub row_initialized: bool,
    pub load_warning: bool,
    pub enable_secondary_queue: bool,
    pub queue_type: Option<QueueType>,
    pub max_gpu_jobs_on_queue: Option<i32>,
    pub cpu_selection_is_spinner: bool,
    pub cpu_selection_minimum: i32,
    pub cpu_selection_maximum: i32,
    pub max_gpu_spinner_enabled: bool,
    pub queue_name: Option<String>,
    pub number_gpus: Option<String>,
    pub number_cpus: Option<i32>,
    pub load1: Option<f64>,
    pub load5: Option<f64>,
    pub users: Option<i32>,
    pub users_tooltip: Option<String>,
    pub cpu_usage_warning: bool,
    pub load1_warning: bool,
    pub load5_warning: bool,
    pub restarts_warning: bool,
    pub restarts_error: bool,
    pub computer_warning: bool,
    pub failure_tooltip: Option<String>,
    pub header1_number_cpus_title: String,
    pub displayed_cells: Vec<String>,
    pub selection_changed: bool,
    pub queue_selected_event: Option<(QueueMode, String)>,
}

impl ProcessorTableRow {
    /// Compatibility constructor used by the existing `ProcessorTable` hook.
    pub fn new(computer: impl Into<String>) -> Self {
        Self::get_computer_instance(
            ProcessorNode {
                name: computer.into(),
                // The string-only ProcessorTable hook is the pre-Node storage
                // boundary.  Do not silently clamp its selected-core value.
                num_cpus: i32::MAX,
                ..ProcessorNode::default()
            },
            1,
        )
    }

    /// Java `getComputerInstance` plus constructor, `initRow`, and listeners.
    pub fn get_computer_instance(node: ProcessorNode, num_rows_in_table: usize) -> Self {
        let mut row = Self::new_with_node(node, false, false, 0, num_rows_in_table);
        row.init_row();
        row.add_listeners();
        row
    }

    /// Java `getQueueInstance` plus constructor, `initRow`, and listeners.
    pub fn get_queue_instance(
        node: ProcessorNode,
        dual_selection_queue_table: bool,
        load_array_size: usize,
        num_rows_in_table: usize,
    ) -> Self {
        let mut row = Self::new_with_node(
            node,
            true,
            dual_selection_queue_table,
            load_array_size,
            num_rows_in_table,
        );
        row.init_row();
        row.add_listeners();
        row
    }

    /// Java private constructor.  The table/header and `ProcessorTableState`
    /// widget visibility queries are owned by `ProcessorTable`; their selected
    /// cells are supplied by its adapter through `set_displayed_cells`.
    fn new_with_node(
        node: ProcessorNode,
        display_queues: bool,
        dual_selection_queue_table: bool,
        load_array_size: usize,
        num_rows_in_table: usize,
    ) -> Self {
        let cpu_selection_is_spinner = node.num_cpus > 1
            && (!dual_selection_queue_table || node.queue_mode != QueueMode::QueueWithSingleCpu);
        let minimum = if dual_selection_queue_table {
            DUAL_SELECTION_MIN
        } else {
            0
        };
        let computer = node.name.clone();
        Self {
            computer: computer.clone(),
            selected: false,
            secondary_queue_selected: false,
            has_secondary_queue: display_queues
                && dual_selection_queue_table
                && node.secondary_queue,
            selection_enabled: true,
            computer_enabled: true,
            cpus_selected: if cpu_selection_is_spinner {
                if dual_selection_queue_table {
                    DUAL_SELECTION_MIN
                } else {
                    DEFAULT_CPUS_SELECTED
                }
            } else {
                DEFAULT_CPUS_SELECTED
            },
            successes: 0,
            restarts: 0,
            failure_reason: None,
            load: vec![String::new(); load_array_size],
            cpu_usage: None,
            displayed: false,
            display_queues,
            num_rows_in_table,
            gpu_device_array: node.gpu_device_array,
            dual_selection_queue_table,
            queue_mode: if dual_selection_queue_table {
                node.queue_mode
            } else {
                QueueMode::Invalid
            },
            secondary_queue: dual_selection_queue_table && node.secondary_queue,
            gpus_per_cluster_job: if node.is_gpu {
                node.gpus_per_cluster_job.clone()
            } else {
                None
            },
            num_cpus: node.num_cpus,
            memory: node.memory.clone(),
            os: node.os.clone(),
            row_initialized: false,
            load_warning: true,
            enable_secondary_queue: false,
            queue_type: None,
            max_gpu_jobs_on_queue: None,
            cpu_selection_is_spinner,
            cpu_selection_minimum: minimum,
            cpu_selection_maximum: node.num_cpus,
            max_gpu_spinner_enabled: false,
            queue_name: dual_selection_queue_table.then_some(computer),
            number_gpus: node.gpus_per_cluster_job,
            number_cpus: Some(node.num_cpus),
            load1: None,
            load5: None,
            users: None,
            users_tooltip: None,
            cpu_usage_warning: false,
            load1_warning: false,
            load5_warning: false,
            restarts_warning: false,
            restarts_error: false,
            computer_warning: false,
            failure_tooltip: None,
            header1_number_cpus_title: "# Cores".into(),
            displayed_cells: vec![],
            selection_changed: false,
            queue_selected_event: None,
        }
    }

    /// Java `addListeners`; listener wiring is a Swing boundary.
    pub fn add_listeners(&mut self) {
        self.row_initialized = true;
    }

    /// Java `queueSelectedAction`.
    pub fn queue_selected_action(&mut self) {
        if self.number_cpus.is_some() {
            self.queue_selected_event = Some((self.queue_mode, self.num_cpus.to_string()));
        }
    }

    /// Java `queueTableEventAction`, including inherited `QueueTableDataEvent`.
    pub fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        match event {
            QueueTableEvent::EnableSecondaryQueue => self.enable_secondary_queue = true,
            QueueTableEvent::DisableSecondaryQueue => self.enable_secondary_queue = false,
            QueueTableEvent::OnlyQueueType(queue_type) => {
                self.queue_type = Some(queue_type);
                self.set_header1_number_cpus_title(Some(queue_type));
            }
            _ => {}
        }
        self.update_display();
    }

    /// Java `QueueTableDataEvent.ONLY_QUEUE_TYPE` branch.
    pub fn queue_table_data_event_action(&mut self, queue_type: QueueType) {
        self.queue_type = Some(queue_type);
        self.set_header1_number_cpus_title(Some(queue_type));
        self.update_display();
    }

    /// Java `setHeader1NumberCPUsTitle`.
    pub fn set_header1_number_cpus_title(&mut self, queue_type: Option<QueueType>) {
        if !self.display_queues {
            return;
        }
        self.header1_number_cpus_title = if matches!(
            queue_type,
            Some(QueueType::Node | QueueType::NodeWithoutGpu)
        ) {
            "# Jobs".into()
        } else {
            "# Cores".into()
        };
    }

    /// Java private `buildGroup`.
    pub fn build_group(&self, prepend: &str) -> String {
        if prepend.is_empty() {
            format!("{}.", self.get_computer())
        } else {
            format!("{prepend}.{}.", self.get_computer())
        }
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let group = self.build_group(prepend);
        if self.has_secondary_queue() {
            props.insert(
                format!("{group}{}", Self::get_secondary_queue_key()),
                self.secondary_queue_selected.to_string(),
            );
        }
        props.insert(
            format!("{group}{STORE_SELECTED}"),
            self.is_selected().to_string(),
        );
        props.insert(
            format!("{group}{STORE_CPUS_SELECTED}"),
            self.cpus_selected.to_string(),
        );
        if self.has_secondary_queue {
            props.insert(
                format!("{group}{STORE_GPUS_SELECTED}"),
                self.max_gpu_jobs_on_queue.unwrap_or(0).to_string(),
            );
        }
    }

    /// Java private `getSecondaryQueueKey`.
    pub fn get_secondary_queue_key() -> String {
        format!("{SECONDARY_QUEUE}.{STORE_SELECTED}")
    }

    /// Java `load(Properties,String)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        let group = self.build_group(prepend);
        if self.has_secondary_queue() {
            self.secondary_queue_selected = props
                .get(&format!("{group}{}", Self::get_secondary_queue_key()))
                .is_some_and(|v| v == "true");
        }
        self.set_selected(
            props
                .get(&format!("{group}{STORE_SELECTED}"))
                .is_some_and(|v| v == "true"),
        );
        if self.cpu_selection_is_spinner && self.is_selected() {
            let default = if self.dual_selection_queue_table {
                DUAL_SELECTION_MIN
            } else {
                DEFAULT_CPUS_SELECTED
            };
            let value = props
                .get(&format!("{group}{STORE_CPUS_SELECTED}"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(default);
            self.set_cpus_selected(&value.to_string());
        }
        if self.has_secondary_queue && self.is_secondary_queue_selected() {
            self.max_gpu_jobs_on_queue = Some(
                props
                    .get(&format!("{group}{STORE_GPUS_SELECTED}"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1),
            );
        }
    }

    /// Java private `initRow` after the Node/ProcessorTableState calls.
    pub fn init_row(&mut self) {
        self.row_initialized = true;
        if !self.cpu_selection_is_spinner {
            self.cpus_selected = DEFAULT_CPUS_SELECTED;
        }
        self.update_selected();
    }

    /// Java `turnOffLoadWarning`.
    pub fn turn_off_load_warning(&mut self) {
        self.load_warning = false;
        self.cpu_usage_warning = false;
        self.load1_warning = false;
        self.load5_warning = false;
    }
    pub fn is_displayed(&self) -> bool {
        self.displayed
    }
    pub fn delete_row(&mut self) {
        self.displayed = false;
        self.displayed_cells.clear();
    }

    /// Java `display(int,Viewport)`.  GridBag work is a Swing boundary; `index`
    /// is retained so an adapter can apply the source viewport predicate.
    pub fn display(&mut self, _index: usize) {
        self.displayed = true;
        self.displayed_cells.clear();
        self.displayed_cells.push("computer".into());
        if self.has_secondary_queue {
            self.displayed_cells.push("secondaryQueue".into());
        }
        if self.queue_name.is_some() {
            self.displayed_cells.push("queueName".into());
        }
        self.displayed_cells.push(
            if self.has_secondary_queue && self.enable_secondary_queue {
                "maxGPUJobsOnQueue"
            } else {
                "CPUsSelected"
            }
            .into(),
        );
        if self.number_cpus.is_some() {
            self.displayed_cells.push("numberCpus".into());
        }
        if self.number_gpus.is_some() {
            self.displayed_cells.push("numberGpus".into());
        }
        if self.load1.is_some() {
            self.displayed_cells.push("load1".into());
        }
        if self.load5.is_some() {
            self.displayed_cells.push("load5".into());
        }
        if self.cpu_usage.is_some() {
            self.displayed_cells.push("cpuUsage".into());
        }
        if !self.load.is_empty() {
            self.displayed_cells.push("loadArray".into());
        }
    }

    /// Java `performAction`.
    pub fn perform_action(&mut self) {
        self.update_selected();
    }
    pub fn focus_gained(&mut self) {}
    pub fn focus_lost(&mut self) {}
    pub fn state_changed_cpu(&mut self) {
        self.selection_changed = true;
    }
    pub fn state_changed_computer(&mut self) {
        if self.display_queues {
            self.update_selected();
        }
    }

    /// Java `msgDropped`.
    pub fn msg_dropped(&mut self, reason: impl Into<String>) {
        self.set_selected(false);
        self.failure_reason = Some(reason.into());
        self.failure_tooltip =
            Some("This computer was dropped from the current distributed process.".into());
    }

    /// Java `setSelected`.
    pub fn set_selected(&mut self, selected: bool) {
        if !selected && self.num_rows_in_table == 1 && !self.selection_enabled {
            return;
        }
        self.selected = selected;
        self.update_selected();
    }
    pub fn enable_selection_field(&mut self, enabled: bool) {
        self.selection_enabled = enabled;
        self.update_display();
    }

    /// Java `setCPUsSelected`.
    pub fn set_cpus_selected(&mut self, input: &str) {
        let Some(mut cpus_selected) = input.parse::<i32>().ok() else {
            return;
        };
        if self.cpu_selection_is_spinner
            && self.dual_selection_queue_table
            && cpus_selected < DUAL_SELECTION_MIN
        {
            cpus_selected = DUAL_SELECTION_MIN;
        }
        self.cpus_selected = cpus_selected.clamp(
            self.cpu_selection_minimum,
            self.cpu_selection_maximum.max(self.cpu_selection_minimum),
        );
    }
    pub fn has_secondary_queue(&self) -> bool {
        self.has_secondary_queue
    }
    pub fn set_secondary_queue_selected(&mut self) {
        if self.has_secondary_queue() {
            self.secondary_queue_selected = true;
            self.update_display();
        }
    }
    pub fn secondary_queue_selected_action(&mut self) {
        self.update_display();
    }
    pub fn set_parameters(&mut self) {
        self.selected = false;
    }

    /// Java private `updateSelected`.
    pub fn update_selected(&mut self) {
        self.set_selected_error();
        self.selection_changed = true;
        self.update_display();
    }

    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        self.computer_enabled = if self.display_queues && self.dual_selection_queue_table {
            self.selection_enabled
                && self.queue_mode != QueueMode::Invalid
                && self
                    .queue_type
                    .is_none_or(|kind| self.queue_mode.is_type(kind))
        } else {
            self.selection_enabled
        };
        if !self.is_selected() && !self.cpu_selection_is_spinner {
            self.cpus_selected = 0;
        } else if self.is_selected()
            && !self.dual_selection_queue_table
            && !self.cpu_selection_is_spinner
        {
            self.cpus_selected = 1;
        }
        if self.has_secondary_queue {
            let selected = self.is_secondary_queue_selected();
            if self.max_gpu_spinner_enabled != selected {
                self.max_gpu_spinner_enabled = selected;
                if selected {
                    self.max_gpu_jobs_on_queue = Some(self.max_gpu_jobs_on_queue.unwrap_or(1));
                } else {
                    self.max_gpu_jobs_on_queue = Some(0);
                }
            }
        }
    }

    /// Java `setSelectedError`; `Utilities.isWindowsOS` is exposed to the UI
    /// adapter through `set_selected_error_for_windows`.
    pub fn set_selected_error(&mut self) {
        self.set_selected_error_for_windows(false, false);
    }
    pub fn set_selected_error_for_windows(&mut self, windows: bool, secondary_table: bool) {
        if secondary_table {
            self.computer_warning = false;
            return;
        }
        let no_load_average = if self.display_queues && !self.load.is_empty() {
            self.load[0].is_empty() || self.load[0] == "NA"
        } else if windows && self.cpu_usage.is_some() {
            self.cpu_usage.as_deref().is_none_or(str::is_empty)
        } else {
            self.load1.is_none()
        };
        self.computer_warning = self.is_selected() && no_load_average;
    }
    pub fn is_selected(&self) -> bool {
        self.computer_enabled && self.selected
    }
    pub fn is_secondary_queue_selected(&self) -> bool {
        self.has_secondary_queue() && self.enable_secondary_queue && self.secondary_queue_selected
    }

    /// Java `getParameters(ProcesschunksParam,boolean,boolean)`.
    pub fn get_processchunks_parameters(
        &self,
        param: &mut ProcesschunksParameters,
        add_gpu_machine: bool,
        secondary_table: bool,
        queue_table: bool,
        gpu_table: bool,
    ) {
        if queue_table && self.is_selected() {
            param.queue_mode = Some(self.queue_mode);
        }
        let cpus = self.get_cpus_selected();
        if cpus > 0 {
            let name = self.get_computer().to_owned();
            if !secondary_table {
                param
                    .machine_names
                    .push((name.clone(), cpus, self.gpu_device_array.clone()));
            }
            if add_gpu_machine && gpu_table {
                param
                    .gpu_machines
                    .push((name, cpus, self.gpu_device_array.clone()));
            }
        }
    }
    pub fn get_secondary_batchruntomo_parameters(
        &self,
        param: &mut BatchruntomoParameters,
        queue_table: bool,
    ) {
        if queue_table && self.is_secondary_queue_selected() {
            param.max_gpu_jobs_on_queue = Some(self.max_gpu_jobs_on_queue.unwrap_or(1).to_string());
        }
    }
    pub fn get_secondary_processchunks_parameters(
        &self,
        param: &mut ProcesschunksParameters,
        queue_table: bool,
    ) {
        if queue_table && self.is_secondary_queue_selected() {
            param.secondary_number = Some(self.max_gpu_jobs_on_queue.unwrap_or(1).to_string());
        }
    }

    /// Java `getParameters(BatchruntomoParam)`.
    pub fn get_batchruntomo_parameters(
        &self,
        param: &mut BatchruntomoParameters,
        queue_table: bool,
        cpu_table: bool,
        gpu_table: bool,
    ) {
        if queue_table && self.is_selected() {
            param.queue_mode = Some(self.queue_mode);
            param.max_jobs_on_queue = Some(self.cpus_selected);
        } else {
            let cpus = self.get_cpus_selected();
            if cpus > 0 && cpu_table {
                param
                    .cpu_machines
                    .push((self.get_computer().to_owned(), cpus));
            } else if cpus > 0 && gpu_table {
                param.gpu_machines.push((
                    self.get_computer().to_owned(),
                    cpus,
                    self.gpu_device_array.clone(),
                ));
            }
        }
    }

    pub fn get_successes(&self) -> i32 {
        self.successes
    }
    pub fn get_cpus_selected(&self) -> i32 {
        if self.is_selected() {
            self.cpus_selected
        } else {
            0
        }
    }
    pub fn equals(&self, computer: &str) -> bool {
        self.get_computer() == computer
    }
    pub fn add_success(&mut self) {
        self.successes = if self.successes == 0 {
            1
        } else {
            self.successes + 1
        };
    }
    pub fn reset_results(&mut self) {
        self.successes = 0;
        self.restarts = 0;
        self.restarts_error = false;
        self.restarts_warning = false;
    }
    pub fn add_restart(&mut self) {
        self.restarts += 1;
        self.restarts_error = self.restarts >= DROP_VALUE;
        self.restarts_warning = !self.restarts_error && self.restarts > 0;
    }

    /// Java `setLoad(double,double,int,String)`.
    pub fn set_load(&mut self, load1: f64, load5: f64, users: i32, tooltip: impl Into<String>) {
        self.set_load_cell(true, load1);
        self.set_load_cell(false, load5);
        self.computer_warning = false;
        self.users = Some(users);
        self.users_tooltip = Some(tooltip.into());
    }
    /// Java `setLoad(String[])`.
    pub fn set_load_array(&mut self, array: &[String]) {
        for (index, value) in array.iter().enumerate() {
            if index < self.load.len() {
                self.load[index] = value.clone();
            }
        }
        self.computer_warning = false;
    }
    /// Java `setCPUUsage` with nullable `ConstEtomoNumber` represented by `Option<i32>`.
    pub fn set_cpu_usage(&mut self, cpu_usage: f64, number_of_processors: impl Into<String>) {
        let processors = number_of_processors.into();
        let count = processors.parse::<f64>().ok();
        let usage = count.map_or(cpu_usage / 100.0, |value| cpu_usage * value / 100.0);
        if self.load_warning {
            self.cpu_usage_warning = cpu_usage > 75.0;
        }
        self.cpu_usage = Some(usage.to_string());
        self.computer_warning = false;
    }
    /// Java `clearLoad`.
    pub fn clear_load(&mut self, reason: impl Into<String>, tooltip: impl Into<String>) {
        self.load1 = None;
        self.load5 = None;
        self.users = None;
        self.cpu_usage = None;
        self.cpu_usage_warning = false;
        self.load1_warning = false;
        self.load5_warning = false;
        self.set_selected_error();
        self.failure_reason = Some(reason.into());
        self.failure_tooltip = Some(tooltip.into());
    }
    /// Java two-argument `clearFailureReason`.
    pub fn clear_failure_reason(&mut self, reasons: Option<(&str, &str)>) {
        if reasons.is_none_or(|(first, second)| {
            self.failure_reason.as_deref() == Some(first)
                || self.failure_reason.as_deref() == Some(second)
        }) {
            self.clear_failure_reason_all();
        }
    }
    pub fn clear_failure_reason_all(&mut self) {
        self.failure_reason = None;
        self.failure_tooltip = None;
    }
    /// Java private `setLoad(FieldCell,double,int)`.
    pub fn set_load_cell(&mut self, first: bool, load: f64) {
        let warning = self.load_warning && load >= self.num_cpus as f64;
        if first {
            self.load1_warning = warning;
            self.load1 = Some(load);
        } else {
            self.load5_warning = warning;
            self.load5 = Some(load);
        }
    }
    pub fn get_height(&self) -> i32 {
        1
    }
    pub fn get_queue_mode(&self) -> QueueMode {
        self.queue_mode
    }
    pub fn get_computer(&self) -> &str {
        self.queue_name.as_deref().unwrap_or(&self.computer)
    }

    /// Java private `add(InputCell,boolean,ColumnName,ColumnName,...)`.  GridBag
    /// placement is a Swing boundary; retaining the remainder predicate gives
    /// the adapter the exact source layout decision.
    pub fn add(
        &mut self,
        cell: impl Into<String>,
        use_cell: bool,
        column_name: ColumnName,
        last_column_name: ColumnName,
    ) {
        if use_cell {
            self.displayed_cells.push(cell.into());
            if column_name == last_column_name {
                self.displayed_cells
                    .push("GridBagConstraints.REMAINDER".into());
            }
        }
    }

    /// Existing `ProcessorTable` map protocol; Java's typed overloads above are
    /// retained separately because Rust cannot overload `getParameters`.
    pub fn get_parameters(&self, param: &mut BTreeMap<String, String>, secondary: bool) {
        if self.is_selected() {
            param.insert(
                self.get_computer().to_owned(),
                self.cpus_selected.to_string(),
            );
        }
        if secondary && self.is_secondary_queue_selected() {
            param.insert("secondaryQueue".into(), self.get_computer().to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queue_row_preserves_dual_selection_and_gpu_spinner_value() {
        let mut row = ProcessorTableRow::get_queue_instance(
            ProcessorNode {
                name: "q".into(),
                num_cpus: 8,
                queue_mode: QueueMode::Queue,
                secondary_queue: true,
                ..ProcessorNode::default()
            },
            true,
            2,
            2,
        );
        row.queue_table_data_event_action(QueueType::Queue);
        row.set_selected(true);
        row.enable_secondary_queue = true;
        row.set_secondary_queue_selected();
        row.update_display();
        assert_eq!(row.get_cpus_selected(), DUAL_SELECTION_MIN);
        assert_eq!(row.max_gpu_jobs_on_queue, Some(1));
    }
    #[test]
    fn persistence_uses_java_group_and_row_selection_rules() {
        let mut row = ProcessorTableRow::new("host");
        row.set_selected(true);
        let mut props = BTreeMap::new();
        row.store(&mut props, "ProcessorTable");
        assert_eq!(props["ProcessorTable.host.Selected"], "true");
        row.set_selected(false);
        row.load(&props, "ProcessorTable");
        assert!(row.is_selected());
    }
    #[test]
    fn load_and_restart_warnings_follow_source_thresholds() {
        let mut row = ProcessorTableRow::get_computer_instance(
            ProcessorNode {
                name: "host".into(),
                num_cpus: 1,
                ..ProcessorNode::default()
            },
            2,
        );
        row.set_selected(true);
        row.set_load(1.0, 2.0, 3, "users");
        row.add_restart();
        assert!(row.load5_warning);
        assert!(row.restarts_warning);
        row.add_restart();
        row.add_restart();
        assert!(row.restarts_error);
    }
}
