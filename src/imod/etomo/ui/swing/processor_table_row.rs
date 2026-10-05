//! `IMOD/Etomo/src/etomo/ui/swing/ProcessorTableRow.java`.
//!
//! One row of a `ProcessorTable`: a computer, GPU or queue, with its selection
//! cell and its load, CPU and result cells.
//!
//! Object model (ui.md): the row is an EDT object (`Rc`, `&self` methods, `Cell` /
//! `RefCell` fields).  It belongs to its table, so it holds the table weakly
//! (`Weak<dyn ProcessorTableVirtual>`; calls on the Java `table` go through the
//! subclass, since Java's calls are virtual).  The Java `ToggleCell cellComputer`
//! is used both as a `ToggleCell` and, cast, as a `Cell`, so the same object is
//! held through both traits.  The Java `InputCell cellCPUsSelected` is a
//! `SpinnerCell` or a `FieldCell`, which the Java tests with `instanceof` and
//! casts; it is held as an `InputCellVirtual` plus [`CpusSelectedCell`], the
//! concrete class.  The listener inner classes are closures capturing a `Weak` to
//! the row (or the table).  Layout (`GridBagLayout`/`GridBagConstraints`) is not
//! modelled; cells are added to the table panel.

use std::cell::{Cell as StdCell, RefCell};
use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::check_box_cell::CheckBoxCell;
use super::field_cell::FieldCell;
use super::gpu_table;
use super::header_cell::HeaderCell;
use super::input_cell::InputCellVirtual;
use super::processor_table::{
    FIRST_QUEUE_LABEL, NUMBER_CPUS_MAX_LABEL, NUMBER_CPUS_USED_LABEL2, ProcessorTableVirtual,
    Properties, SECONDARY_QUEUE_LABEL,
};
use super::radio_button_cell::RadioButtonCell;
use super::spinner_cell::SpinnerCell;
use super::toggle_cell::ToggleCell;
use super::viewport::Viewport;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::processchunks_param::{self, ProcesschunksParam};
use crate::imod::etomo::jdk::ButtonGroup;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::logic::processor_table_state::ProcessorTableState;
use crate::imod::etomo::logic::table_state::TableState;
use crate::imod::etomo::storage::node::Node;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::const_etomo_number::{
    self, ConstEtomoNumber, java_lang_integer_parse_int,
};
use crate::imod::etomo::ui::processor_table_field::ProcessorTableField;
use crate::imod::etomo::ui::queue_table_data_event::QueueTableDataEvent;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::table_field::TableField;
use crate::imod::etomo::util::utilities;

/// The queue mode and type (re-exported for the modules that used to take them from
/// here).
pub use crate::imod::etomo::r#type::queue_mode::QueueMode;
pub use crate::imod::etomo::r#type::queue_type::QueueType;

/// Java private static final `SECONDARY_QUEUE`.
const SECONDARY_QUEUE: &str = "SecondaryQueue";
/// Java private static final `STORE_SELECTED`.
const STORE_SELECTED: &str = "Selected";
/// Java private static final `STORE_CPUS_SELECTED`.
const STORE_CPUS_SELECTED: &str = "CPUsSelected";
/// Java private static final `STORE_GPUS_SELECTED`.
const STORE_GPUS_SELECTED: &str = "MaxGPUJobsOnQueue";
/// Java private static final `DEFAULT_CPUS_SELECTED`.
const DEFAULT_CPUS_SELECTED: i32 = 1;
/// Java private static final `DUAL_SELECTION_MIN`.
const DUAL_SELECTION_MIN: i32 = 2;

/// The concrete class of Java `cellCPUsSelected` (`instanceof SpinnerCell` /
/// `FieldCell`).
#[derive(Clone)]
pub enum CpusSelectedCell {
    /// A `SpinnerCell`.
    Spinner(Rc<SpinnerCell>),
    /// A `FieldCell`.
    Field(Rc<FieldCell>),
}

/// `Rc<dyn TableField>` for a `ProcessorTableField` constant.
fn table_field(field: ProcessorTableField) -> Option<Rc<dyn TableField>> {
    Some(Rc::new(field))
}

/// Java package-private `final class ProcessorTableRow implements Storable`.
pub struct ProcessorTableRow {
    /// This row, for the listener closures.
    self_ref: RefCell<Weak<ProcessorTableRow>>,
    /// Java `tempDisplayedFields`: temporary storage for building the row.
    temp_displayed_fields: RefCell<Vec<Rc<dyn CellVirtual>>>,

    /// Java final `cellComputer`: a computer or queue.
    cell_computer: Rc<dyn ToggleCell>,
    /// Java `(Cell) cellComputer`: the same object.
    cell_computer_cell: Rc<dyn CellVirtual>,
    /// Java final `cellSecondaryQueue`.
    cell_secondary_queue: Option<Rc<RadioButtonCell>>,
    /// Java final `cellBlankSecondaryQueue`.
    cell_blank_secondary_queue: Option<Rc<HeaderCell>>,
    /// Java final `cellQueueName`.
    cell_queue_name: Option<Rc<HeaderCell>>,
    /// Java final `cellCPUsSelected`.
    cell_cpus_selected: Rc<dyn InputCellVirtual>,
    /// The concrete class of `cellCPUsSelected`.
    cell_cpus_selected_type: CpusSelectedCell,
    /// Java final `cellNumberCpus`.
    cell_number_cpus: Option<Rc<FieldCell>>,
    /// Java final `cellNumberGpus`.
    cell_number_gpus: Option<Rc<FieldCell>>,
    /// Java final `cellBlankNumberGpus`.
    cell_blank_number_gpus: Option<Rc<HeaderCell>>,
    /// Java final `cellMaxGPUJobsOnQueue`.
    cell_max_gpu_jobs_on_queue: Option<Rc<SpinnerCell>>,
    /// Java final `cellLoad1`.
    cell_load1: Option<Rc<FieldCell>>,
    /// Java final `cellLoad5`.
    cell_load5: Option<Rc<FieldCell>>,
    /// Java final `cellCPUUsage`.
    cell_cpu_usage: Option<Rc<FieldCell>>,
    /// Java final `cellLoadArray`.
    cell_load_array: Option<Vec<Rc<FieldCell>>>,
    /// Java final `cellUsers`.
    cell_users: Option<Rc<FieldCell>>,
    /// Java final `cellCPUType`.
    cell_cpu_type: Option<Rc<FieldCell>>,
    /// Java final `cellSpeed`.
    cell_speed: Option<Rc<FieldCell>>,
    /// Java final `cellMemory`.
    cell_memory: Option<Rc<FieldCell>>,
    /// Java final `cellOS`.
    cell_os: Option<Rc<FieldCell>>,
    /// Java final `cellRestarts`.
    cell_restarts: Option<Rc<FieldCell>>,
    /// Java final `cellSuccesses`.
    cell_successes: Option<Rc<FieldCell>>,
    /// Java final `cellFailureReason`.
    cell_failure_reason: Option<Rc<FieldCell>>,
    /// Java final `cellGpuType`.
    cell_gpu_type: Option<Rc<FieldCell>>,
    /// Java final `cellGpuSpeed`.
    cell_gpu_speed: Option<Rc<FieldCell>>,
    /// Java final `cellGpuMemory`.
    cell_gpu_memory: Option<Rc<FieldCell>>,
    /// Java final `cellGpuNcores`.
    cell_gpu_ncores: Option<Rc<FieldCell>>,

    /// Java final `displayQueues`.
    display_queues: bool,
    /// Java final `numRowsInTable`.
    num_rows_in_table: i32,

    /// Java final `gpuDeviceArray`.
    gpu_device_array: Option<Vec<String>>,
    /// Java final `dualSelectionQueueTable`.
    dual_selection_queue_table: bool,
    /// Java final `queueMode`.
    queue_mode: Option<QueueMode>,
    /// Java final `secondaryQueue`.
    #[allow(dead_code)]
    secondary_queue: bool,
    /// Java final `gpusPerClusterJob`.
    gpus_per_cluster_job: Option<String>,

    /// Java `table`.
    table: Weak<dyn ProcessorTableVirtual>,
    /// Java `numCpus`.
    num_cpus: i32,
    /// Java `memory` (assigned, never read, in the Java too).
    #[allow(dead_code)]
    memory: Option<String>,
    /// Java `os`.
    os: Option<String>,
    /// Java `rowInitialized` (assigned, never read, in the Java too).
    row_initialized: StdCell<bool>,
    /// Java `displayed`.
    displayed: StdCell<bool>,
    /// Java `loadWarning`.
    load_warning: StdCell<bool>,
    /// Java `selectionEnabled`.
    selection_enabled: StdCell<bool>,
    /// Java `enableSecondaryQueue`.
    enable_secondary_queue: StdCell<bool>,
    /// Java `queueType`.
    queue_type: StdCell<Option<QueueType>>,
    /// Java `maxGPUJobsOnQueue`.
    max_gpu_jobs_on_queue: StdCell<Option<i32>>,

    /// Java `computerName`.
    computer_name: String,
    /// Java final package-private `tableState`.
    pub(crate) table_state: Rc<ProcessorTableState>,
}

impl ProcessorTableRow {
    /// Java private `ProcessorTableRow(ProcessorTable, Node, int numCpus, boolean
    /// displayQueues, boolean dualSelectionQueueTable, ButtonGroup queueButtonGroup,
    /// ButtonGroup secondaryQueueButtonGroup, int queueLoadArraySize, int
    /// numRowsInTable, ProcessorTableState)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_cpus: i32,
        display_queues: bool,
        dual_selection_queue_table: bool,
        queue_button_group: Option<&Rc<ButtonGroup>>,
        secondary_queue_button_group: Option<&Rc<ButtonGroup>>,
        queue_load_array_size: i32,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        let computer_name = node.get_name().to_string();
        let (queue_mode, secondary_queue) = if dual_selection_queue_table {
            (Some(node.get_queue_mode()), node.is_secondary_queue())
        } else {
            (None, false)
        };
        let memory = node.get_memory();
        let gpu_device_array = node.get_gpu_device_array().map(<[String]>::to_vec);
        let os = node.get_os();
        let gpus_per_cluster_job = if node.is_gpu() {
            node.get_gpus_per_cluster_job()
        } else {
            None
        };
        let max = num_cpus;
        let header1_computer_text = table.get_header1_computer_text();
        let cell_computer: Rc<dyn ToggleCell>;
        let cell_computer_cell: Rc<dyn CellVirtual>;
        let cell_secondary_queue;
        let cell_blank_secondary_queue;
        let cell_queue_name;
        let cell_max_gpu_jobs_on_queue;
        if display_queues {
            if table_state.is_use(ProcessorTableField::TwoQueuesH2) {
                let radio = RadioButtonCell::get_named_instance_button_group_string_string_string(
                    queue_button_group,
                    header1_computer_text.as_deref(),
                    Some(FIRST_QUEUE_LABEL),
                    Some(&computer_name),
                );
                cell_computer = radio.clone();
                cell_computer_cell = radio;
                if secondary_queue {
                    // Create a radio button and a spinner for the secondary queue.
                    cell_secondary_queue = Some(
                        RadioButtonCell::get_named_instance_button_group_string_string_string(
                            secondary_queue_button_group,
                            header1_computer_text.as_deref(),
                            Some(SECONDARY_QUEUE_LABEL),
                            Some(&computer_name),
                        ),
                    );
                    cell_blank_secondary_queue = None;
                    cell_max_gpu_jobs_on_queue = Some(SpinnerCell::get_int_instance(1, max));
                } else {
                    cell_secondary_queue = None;
                    cell_blank_secondary_queue = Some(HeaderCell::get_named_instance(
                        header1_computer_text.as_deref(),
                        Some(SECONDARY_QUEUE_LABEL),
                        Some(&computer_name),
                    ));
                    cell_max_gpu_jobs_on_queue = None;
                }
                cell_queue_name = Some(HeaderCell::new_void());
            } else {
                let radio = RadioButtonCell::get_named_instance_button_group_string(
                    queue_button_group,
                    header1_computer_text.as_deref(),
                );
                cell_computer = radio.clone();
                cell_computer_cell = radio;
                cell_secondary_queue = None;
                cell_blank_secondary_queue = None;
                cell_queue_name = None;
                cell_max_gpu_jobs_on_queue = None;
            }
        } else {
            let check_box =
                CheckBoxCell::get_named_instance_string(header1_computer_text.as_deref());
            cell_computer = check_box.clone();
            cell_computer_cell = check_box;
            cell_secondary_queue = None;
            cell_max_gpu_jobs_on_queue = None;
            cell_blank_secondary_queue = None;
            cell_queue_name = None;
        }
        let cell_cpus_selected_type = if num_cpus > 1
            && (!dual_selection_queue_table || queue_mode == Some(QueueMode::QueueWithSingleCpu))
        {
            let mut min = 0;
            if dual_selection_queue_table {
                min = DUAL_SELECTION_MIN;
            }
            CpusSelectedCell::Spinner(SpinnerCell::get_int_instance(min, max))
        } else {
            CpusSelectedCell::Field(FieldCell::get_ineditable_instance())
        };
        let cell_cpus_selected: Rc<dyn InputCellVirtual> = match &cell_cpus_selected_type {
            CpusSelectedCell::Spinner(cell) => cell.clone(),
            CpusSelectedCell::Field(cell) => cell.clone(),
        };
        // setHeader1NumberCPUsTitle(queueType) runs here in the Java, with queueType
        // still null; it only touches cellCPUsSelected and cellMaxGPUJobsOnQueue, so it
        // is run below, once the row exists.
        let cell_number_cpus = if table_state.is_use(ProcessorTableField::NumCpusMaxH2) {
            Some(FieldCell::get_named_ineditable_instance_string_string(
                table.getheader1_number_cpus_title().as_deref(),
                Some(NUMBER_CPUS_MAX_LABEL),
            ))
        } else {
            None
        };
        let (cell_number_gpus, cell_blank_number_gpus) =
            if table_state.is_use(ProcessorTableField::NumGpus) {
                if gpus_per_cluster_job.is_some() {
                    (
                        Some(FieldCell::get_named_ineditable_instance_string_string(
                            Some(gpu_table::NUMBER_CPUS_LABEL),
                            Some(&computer_name),
                        )),
                        None,
                    )
                } else {
                    (
                        None,
                        Some(HeaderCell::get_named_instance(
                            Some(gpu_table::NUMBER_CPUS_LABEL),
                            Some(&computer_name),
                            None,
                        )),
                    )
                }
            } else {
                (None, None)
            };
        let (cell_load1, cell_load5) = if table_state.is_use(ProcessorTableField::LoadAverageH1) {
            (
                Some(FieldCell::get_ineditable_instance()),
                Some(FieldCell::get_ineditable_instance()),
            )
        } else {
            (None, None)
        };
        let cell_cpu_usage = if table_state.is_use(ProcessorTableField::CpuUsageH1) {
            Some(FieldCell::get_ineditable_instance())
        } else {
            None
        };
        let cell_load_array = if table_state.is_use(ProcessorTableField::LoadArray0H1) {
            if !table_state.is_use(ProcessorTableField::LoadArrayXH1) {
                Some(vec![FieldCell::get_ineditable_instance()])
            } else {
                let mut cell_load_array = Vec::new();
                for _ in 0..queue_load_array_size {
                    cell_load_array.push(FieldCell::get_ineditable_instance());
                }
                Some(cell_load_array)
            }
        } else {
            None
        };
        let ineditable_if = |field: ProcessorTableField| {
            if table_state.is_use(field) {
                Some(FieldCell::get_ineditable_instance())
            } else {
                None
            }
        };
        let cell_users = ineditable_if(ProcessorTableField::UsersH1);
        let cell_cpu_type = if table_state.is_use(ProcessorTableField::TypeH1) {
            Some(FieldCell::get_named_ineditable_instance_string(Some(
                super::processor_table::CPU_TYPE_LABEL,
            )))
        } else {
            None
        };
        let cell_speed = ineditable_if(ProcessorTableField::SpeedH1);
        let cell_memory = ineditable_if(ProcessorTableField::MemoryH1);
        let cell_os = ineditable_if(ProcessorTableField::OsH1);
        let cell_gpu_type = ineditable_if(ProcessorTableField::GpuTypeH1);
        let cell_gpu_speed = ineditable_if(ProcessorTableField::GpuSpeedH1);
        let cell_gpu_memory = ineditable_if(ProcessorTableField::GpuMemoryH1);
        let cell_gpu_ncores = ineditable_if(ProcessorTableField::GpuNcoresH1);
        let (cell_restarts, cell_successes, cell_failure_reason) =
            if table_state.is_use(ProcessorTableField::RestartsH1) {
                (
                    Some(FieldCell::get_named_ineditable_instance_string(Some(
                        super::processor_table::RESTARTS_LABEL,
                    ))),
                    Some(FieldCell::get_ineditable_instance()),
                    Some(FieldCell::get_ineditable_instance()),
                )
            } else {
                (None, None, None)
            };
        let row = Rc::new(ProcessorTableRow {
            self_ref: RefCell::new(Weak::new()),
            temp_displayed_fields: RefCell::new(Vec::new()),
            cell_computer,
            cell_computer_cell,
            cell_secondary_queue,
            cell_blank_secondary_queue,
            cell_queue_name,
            cell_cpus_selected,
            cell_cpus_selected_type,
            cell_number_cpus,
            cell_number_gpus,
            cell_blank_number_gpus,
            cell_max_gpu_jobs_on_queue,
            cell_load1,
            cell_load5,
            cell_cpu_usage,
            cell_load_array,
            cell_users,
            cell_cpu_type,
            cell_speed,
            cell_memory,
            cell_os,
            cell_restarts,
            cell_successes,
            cell_failure_reason,
            cell_gpu_type,
            cell_gpu_speed,
            cell_gpu_memory,
            cell_gpu_ncores,
            display_queues,
            num_rows_in_table,
            gpu_device_array,
            dual_selection_queue_table,
            queue_mode,
            secondary_queue,
            gpus_per_cluster_job,
            table: Rc::downgrade(table),
            num_cpus,
            memory,
            os,
            row_initialized: StdCell::new(false),
            displayed: StdCell::new(false),
            load_warning: StdCell::new(true),
            selection_enabled: StdCell::new(true),
            enable_secondary_queue: StdCell::new(false),
            queue_type: StdCell::new(None),
            max_gpu_jobs_on_queue: StdCell::new(None),
            computer_name,
            table_state: table_state.clone(),
        });
        *row.self_ref.borrow_mut() = Rc::downgrade(&row);
        row.set_header1_number_cpus_title(row.queue_type.get());
        row.update_display();
        row
    }

    /// Java `table`.  The table owns its rows, so it is alive whenever a row is used.
    fn table(&self) -> Rc<dyn ProcessorTableVirtual> {
        self.table
            .upgrade()
            .expect("ProcessorTableRow: the table that owns this row is gone")
    }

    /// Java static `getComputerInstance(ProcessorTable, Node, int, int,
    /// ProcessorTableState)`.
    pub fn get_computer_instance(
        table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_cpus: i32,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        let instance = ProcessorTableRow::new(
            table,
            node,
            num_cpus,
            false,
            false,
            None,
            None,
            0,
            num_rows_in_table,
            table_state,
        );
        instance.init_row(Some(node));
        instance.add_listeners();
        instance
    }

    /// Java static `getQueueInstance(ProcessorTable, Node, int, boolean, ButtonGroup,
    /// ButtonGroup, int, int, ProcessorTableState)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_queue_instance(
        table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_cpus: i32,
        dual_selection_queue_table: bool,
        button_group: Option<&Rc<ButtonGroup>>,
        secondary_button_group: Option<&Rc<ButtonGroup>>,
        load_array_size: i32,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        let instance = ProcessorTableRow::new(
            table,
            node,
            num_cpus,
            true,
            dual_selection_queue_table,
            button_group,
            secondary_button_group,
            load_array_size,
            num_rows_in_table,
            table_state,
        );
        instance.init_row(Some(node));
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        if let CpusSelectedCell::Spinner(spinner) = &self.cell_cpus_selected_type {
            // new PTRCPUChangeListener(this)
            let adaptee = self.self_ref.borrow().clone();
            spinner.add_change_listener(Rc::new(move |_event| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.state_changed_cpu();
                }
            }));
        }
        if self.display_queues && self.dual_selection_queue_table {
            // new QueueSelectedListener(this)
            let adaptee = self.self_ref.borrow().clone();
            self.cell_computer
                .add_action_listener(Rc::new(move |_event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.queue_selected_action();
                    }
                }));
            if let Some(cell_secondary_queue) = &self.cell_secondary_queue {
                // new SecondaryQueueSelectedListener(table)
                let adaptee = self.table.clone();
                cell_secondary_queue.add_action_listener(Rc::new(move |_event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.processor_table().secondary_queue_selected_action();
                    }
                }));
            }
        }
    }

    /// Java `queueSelectedAction()`.
    pub fn queue_selected_action(&self) {
        if let Some(cell_number_cpus) = &self.cell_number_cpus
            && !cell_number_cpus.is_empty()
        {
            // queueMode is set whenever this listener is registered (dual selection).
            if let Some(queue_mode) = self.queue_mode {
                self.table().send_queue_table_event(
                    &QueueTableDataEvent::get_queue_selected_instance(
                        queue_mode,
                        cell_number_cpus.get_text_void(),
                    ),
                );
            }
        }
    }

    /// Java `queueTableEventAction(QueueTableEvent)`.
    pub fn queue_table_event_action(&self, event: &QueueTableEvent) {
        match event {
            // event instanceof QueueTableDataEvent
            QueueTableEvent::QueueSelected { .. } | QueueTableEvent::NumberJobsChanged(_) => {}
            QueueTableEvent::OnlyQueueType(queue_type) => {
                self.queue_type.set(Some(*queue_type));
                self.set_header1_number_cpus_title(self.queue_type.get());
            }
            QueueTableEvent::EnableSecondaryQueue => {
                self.enable_secondary_queue.set(true);
                self.table().processor_table().build();
            }
            QueueTableEvent::DisableSecondaryQueue => {
                self.enable_secondary_queue.set(false);
                self.table().processor_table().build();
            }
            _ => {}
        }
        self.update_display();
    }

    /// Java `setHeader1NumberCPUsTitle(QueueType)`.
    pub fn set_header1_number_cpus_title(&self, queue_type: Option<QueueType>) {
        if !self.display_queues {
            return;
        }
        let header_label1 = if queue_type == Some(QueueType::Node)
            || queue_type == Some(QueueType::NodeWithoutGpu)
        {
            Some(super::queue_table::NUMBER_JOBS_LABEL1.to_string())
        } else {
            self.table().getheader1_number_cpus_title()
        };
        self.cell_cpus_selected.set_name_string_string_string(
            header_label1.as_deref(),
            Some(NUMBER_CPUS_USED_LABEL2),
            Some(&self.computer_name),
        );
        if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue {
            cell_max_gpu_jobs_on_queue.set_name_string_string_string(
                header_label1.as_deref(),
                Some(NUMBER_CPUS_USED_LABEL2),
                Some(&self.computer_name),
            );
        }
    }

    /// Java `@Override store(Properties)`.
    pub fn store_properties(&self, props: &mut Properties) {
        self.store_properties_string(props, "");
    }

    /// Java private `buildGroup(String)`.
    fn build_group(&self, prepend: &str) -> String {
        // Java string concatenation of a null label gives "null".
        let label = self.get_computer().unwrap_or_else(|| "null".to_string());
        // `prepend == ""` is a reference comparison in Java; every caller passes the
        // literal "" or a built string, so it is an emptiness test.
        let prepend = if prepend.is_empty() {
            label
        } else {
            format!("{prepend}.{label}")
        };
        format!("{prepend}.")
    }

    /// Java `@Override store(Properties, String)`.
    pub fn store_properties_string(&self, props: &mut Properties, prepend: &str) {
        let group = self.build_group(prepend);
        if self.has_secondary_queue() {
            props.insert(
                format!("{}{}", group, ProcessorTableRow::get_secondary_queue_key()),
                self.cell_secondary_queue
                    .as_ref()
                    .unwrap()
                    .is_selected()
                    .to_string(),
            );
        }
        props.insert(
            format!("{group}{STORE_SELECTED}"),
            self.is_selected().to_string(),
        );
        match &self.cell_cpus_selected_type {
            CpusSelectedCell::Field(field) => {
                props.insert(
                    format!("{group}{STORE_CPUS_SELECTED}"),
                    field.get_value().unwrap_or_else(|| "null".to_string()),
                );
            }
            CpusSelectedCell::Spinner(spinner) => {
                props.insert(
                    format!("{group}{STORE_CPUS_SELECTED}"),
                    spinner.get_int_value().to_string(),
                );
            }
        }
        if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue {
            props.insert(
                format!("{group}{STORE_GPUS_SELECTED}"),
                cell_max_gpu_jobs_on_queue.get_int_value().to_string(),
            );
        }
    }

    /// Java `@Override load(Properties)`.  Get the objects attributes from the
    /// properties object.
    pub fn load_properties(&self, props: &Properties) {
        self.load_properties_string(props, "");
    }

    /// Java private static `getSecondaryQueueKey()`.
    fn get_secondary_queue_key() -> String {
        format!("{SECONDARY_QUEUE}.{STORE_SELECTED}")
    }

    /// Java `@Override load(Properties, String)`.  Load the computers and number of
    /// CPUs selected.
    ///
    /// Upstream bug fixed in translation (`ProcessorTableRow.java:442-449`): a stored
    /// CPU or GPU count that is not an integer makes `Integer.parseInt` throw
    /// `NumberFormatException` out of the load; here that value is skipped and the
    /// cell keeps its current value.
    pub fn load_properties_string(&self, props: &Properties, prepend: &str) {
        let group = self.build_group(prepend);
        // Boolean.valueOf(String): true only for "true", ignoring case.
        let boolean_value_of = |value: Option<&String>, default: &str| {
            value
                .map(String::as_str)
                .unwrap_or(default)
                .eq_ignore_ascii_case("true")
        };
        if self.has_secondary_queue() {
            self.cell_secondary_queue
                .as_ref()
                .unwrap()
                .set_selected(boolean_value_of(
                    props.get(&format!(
                        "{}{}",
                        group,
                        ProcessorTableRow::get_secondary_queue_key()
                    )),
                    "false",
                ));
        }
        let selected = boolean_value_of(props.get(&format!("{group}{STORE_SELECTED}")), "false");
        self.set_selected(selected);
        if let CpusSelectedCell::Spinner(spinner) = &self.cell_cpus_selected_type
            && self.is_selected()
        {
            let default = if self.dual_selection_queue_table {
                DUAL_SELECTION_MIN
            } else {
                DEFAULT_CPUS_SELECTED
            }
            .to_string();
            let value = props
                .get(&format!("{group}{STORE_CPUS_SELECTED}"))
                .cloned()
                .unwrap_or(default);
            if let Ok(value) = java_lang_integer_parse_int(&value) {
                spinner.set_value_int(value);
            }
        }
        if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue
            && self.is_secondary_queue_selected()
        {
            let value = props
                .get(&format!("{group}{STORE_GPUS_SELECTED}"))
                .cloned()
                .unwrap_or_else(|| 1.to_string());
            if let Ok(value) = java_lang_integer_parse_int(&value) {
                cell_max_gpu_jobs_on_queue.set_value_int(value);
            }
        }
    }

    /// Java private `initRow(Node)`.
    fn init_row(&self, node: Option<&Node>) {
        let table_state: Option<Rc<dyn TableState>> = Some(self.table_state.clone());
        // table state
        if let Some(cell_number_cpus) = &self.cell_number_cpus {
            cell_number_cpus.set_table_state(
                table_field(ProcessorTableField::NumCpusMaxH2),
                table_state.clone(),
            );
        }
        if let Some(cell_number_gpus) = &self.cell_number_gpus {
            cell_number_gpus.set_table_state(
                table_field(ProcessorTableField::NumGpus),
                table_state.clone(),
            );
        }
        if let Some(cell_load1) = &self.cell_load1 {
            cell_load1.set_table_state(
                table_field(ProcessorTableField::LoadAverageH1),
                table_state.clone(),
            );
            self.cell_load5.as_ref().unwrap().set_table_state(
                table_field(ProcessorTableField::LoadAverageH1),
                table_state.clone(),
            );
        }
        if let Some(cell_cpu_usage) = &self.cell_cpu_usage {
            cell_cpu_usage.set_table_state(
                table_field(ProcessorTableField::CpuUsageH1),
                table_state.clone(),
            );
        }
        if let Some(cell_load_array) = &self.cell_load_array {
            for (i, cell) in cell_load_array.iter().enumerate() {
                if i > 0 {
                    cell.set_table_state(
                        table_field(ProcessorTableField::LoadArrayXH1),
                        table_state.clone(),
                    );
                } else {
                    cell.set_table_state(
                        table_field(ProcessorTableField::LoadArray0H1),
                        table_state.clone(),
                    );
                }
            }
        }
        for (cell, field) in [
            (&self.cell_users, ProcessorTableField::UsersH1),
            (&self.cell_cpu_type, ProcessorTableField::TypeH1),
            (&self.cell_speed, ProcessorTableField::SpeedH1),
            (&self.cell_memory, ProcessorTableField::MemoryH1),
            (&self.cell_os, ProcessorTableField::OsH1),
            (&self.cell_gpu_type, ProcessorTableField::GpuTypeH1),
            (&self.cell_gpu_speed, ProcessorTableField::GpuSpeedH1),
            (&self.cell_gpu_memory, ProcessorTableField::GpuMemoryH1),
            (&self.cell_gpu_ncores, ProcessorTableField::GpuNcoresH1),
            (&self.cell_restarts, ProcessorTableField::RestartsH1),
        ] {
            if let Some(cell) = cell {
                cell.set_table_state(table_field(field), table_state.clone());
            }
        }
        // init
        self.row_initialized.set(true);
        // new ProcessorTableRowActionListener(this)
        let adaptee = self.self_ref.borrow().clone();
        self.cell_computer
            .add_action_listener(Rc::new(move |_event| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.perform_action();
                }
            }));
        if self.display_queues {
            // new PTRComputerChangeListener(this)
            let adaptee = self.self_ref.borrow().clone();
            self.cell_computer
                .add_change_listener(Rc::new(move |_event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.state_changed_computer();
                    }
                }));
        }
        if self.has_secondary_queue() {
            // new ProcessorTableRowActionListener(this)
            let adaptee = self.self_ref.borrow().clone();
            self.cell_secondary_queue
                .as_ref()
                .unwrap()
                .add_action_listener(Rc::new(move |_event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.perform_action();
                    }
                }));
        }
        match &self.cell_cpus_selected_type {
            CpusSelectedCell::Spinner(spinner) => {
                if !self.dual_selection_queue_table {
                    spinner.set_value_int(DEFAULT_CPUS_SELECTED);
                } else {
                    spinner.set_value_int(DUAL_SELECTION_MIN);
                }
                spinner.set_disabled_value(0);
            }
            CpusSelectedCell::Field(field) => field.set_value_int(1),
        }
        if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue {
            cell_max_gpu_jobs_on_queue.set_enabled(false);
            cell_max_gpu_jobs_on_queue.set_value_int(0);
        }
        if let Some(cell_number_cpus) = &self.cell_number_cpus {
            cell_number_cpus.set_value_int(self.num_cpus);
        }
        if let Some(cell_number_gpus) = &self.cell_number_gpus {
            cell_number_gpus.set_value_string(self.gpus_per_cluster_job.as_deref());
        }
        if let Some(cell_memory) = &self.cell_memory {
            cell_memory.set_editable(false);
        }
        if let Some(node) = node {
            if !self.dual_selection_queue_table {
                self.cell_computer.set_label(Some(node.get_name()));
            } else if let Some(cell_queue_name) = &self.cell_queue_name {
                // Java dereferences cellQueueName unconditionally; it is non-null
                // whenever the table uses dual selection (TWO_QUEUES_H2).
                cell_queue_name.set_text_string(Some(node.get_name()));
            }
            if let Some(cell_cpu_type) = &self.cell_cpu_type {
                cell_cpu_type.set_value_string(node.get_type().as_deref());
            }
            if let Some(cell_speed) = &self.cell_speed {
                cell_speed.set_value_string(node.get_speed().as_deref());
            }
            if let Some(cell_memory) = &self.cell_memory {
                cell_memory.set_value_string(node.get_memory().as_deref());
            }
            if let Some(cell_os) = &self.cell_os {
                cell_os.set_value_string(self.os.as_deref());
            }
            if let Some(cell_gpu_type) = &self.cell_gpu_type {
                cell_gpu_type.set_value_string(node.get_gpu_type().as_deref());
            }
            if let Some(cell_gpu_speed) = &self.cell_gpu_speed {
                cell_gpu_speed.set_value_string(node.get_gpu_speed().as_deref());
            }
            if let Some(cell_gpu_memory) = &self.cell_gpu_memory {
                cell_gpu_memory.set_value_string(node.get_gpu_memory().as_deref());
            }
            if let Some(cell_gpu_ncores) = &self.cell_gpu_ncores {
                cell_gpu_ncores.set_value_string(node.get_gpu_ncores().as_deref());
            }
        }
        self.update_selected();
    }

    /// Java `turnOffLoadWarning()`.
    pub fn turn_off_load_warning(&self) {
        self.load_warning.set(false);
        if let Some(cell_cpu_usage) = &self.cell_cpu_usage {
            cell_cpu_usage.set_warning_boolean(false);
        }
        if let Some(cell_load1) = &self.cell_load1 {
            cell_load1.set_warning_boolean(false);
            self.cell_load5.as_ref().unwrap().set_warning_boolean(false);
        }
    }

    /// Java `isDisplayed()`.
    pub fn is_displayed(&self) -> bool {
        self.displayed.get()
    }

    /// Java `deleteRow()`.
    pub fn delete_row(&self) {
        self.displayed.set(false);
    }

    // Java private `add(InputCell, boolean, ColumnName, ColumnName, JPanel,
    // GridBagLayout, GridBagConstraints)` is never called; its body only adds a cell
    // with layout constraints (see `display`).

    /// Java `display(int, Viewport)`.
    pub fn display(&self, index: i32, viewport: &Rc<Viewport>) {
        self.displayed.set(true);
        if !viewport.in_viewport(index) {
            return;
        }
        // create row
        let Some(panel) = self.table().processor_table().get_table_panel() else {
            return;
        };
        // Swing layout: layout = table.getTableLayout(); constraints =
        // table.getTableConstraints(); constraints.weighty = 0.0; weightx = 0.0;
        // gridheight = 1.
        // Set display columns
        {
            let mut temp = self.temp_displayed_fields.borrow_mut();
            temp.clear();
            temp.push(self.cell_computer_cell.clone());
            if self.has_secondary_queue() {
                temp.push(self.cell_secondary_queue.as_ref().unwrap().clone());
            }
            if let Some(cell) = &self.cell_blank_secondary_queue {
                temp.push(cell.clone());
            }
            if let Some(cell) = &self.cell_queue_name {
                temp.push(cell.clone());
            }
            if self.cell_max_gpu_jobs_on_queue.is_none() || !self.enable_secondary_queue.get() {
                let cell: Rc<dyn CellVirtual> = match &self.cell_cpus_selected_type {
                    CpusSelectedCell::Spinner(cell) => cell.clone(),
                    CpusSelectedCell::Field(cell) => cell.clone(),
                };
                temp.push(cell);
            }
            if let Some(cell) = &self.cell_max_gpu_jobs_on_queue
                && self.enable_secondary_queue.get()
            {
                temp.push(cell.clone());
            }
            if let Some(cell) = &self.cell_number_cpus
                && cell.is_display()
            {
                temp.push(cell.clone());
            }
            if let Some(cell) = &self.cell_number_gpus
                && cell.is_display()
            {
                temp.push(cell.clone());
            }
            if let Some(cell) = &self.cell_blank_number_gpus
                && cell.is_display()
            {
                temp.push(cell.clone());
            }
            if let Some(cell_load1) = &self.cell_load1 {
                if cell_load1.is_display() {
                    temp.push(cell_load1.clone());
                }
                let cell_load5 = self.cell_load5.as_ref().unwrap();
                if cell_load5.is_display() {
                    temp.push(cell_load5.clone());
                }
            }
            if let Some(cell) = &self.cell_cpu_usage
                && cell.is_display()
            {
                temp.push(cell.clone());
            }
            if let Some(cell_load_array) = &self.cell_load_array {
                for cell in cell_load_array {
                    if cell.is_display() {
                        temp.push(cell.clone());
                    }
                }
            }
            for cell in [
                &self.cell_users,
                &self.cell_cpu_type,
                &self.cell_speed,
                &self.cell_memory,
                &self.cell_os,
                &self.cell_gpu_type,
                &self.cell_gpu_speed,
                &self.cell_gpu_memory,
                &self.cell_gpu_ncores,
            ] {
                if let Some(cell) = cell
                    && cell.is_display()
                {
                    temp.push(cell.clone());
                }
            }
            if let Some(cell_restarts) = &self.cell_restarts
                && cell_restarts.is_display()
            {
                temp.push(cell_restarts.clone());
                temp.push(self.cell_successes.as_ref().unwrap().clone());
                temp.push(self.cell_failure_reason.as_ref().unwrap().clone());
            }
        }
        // Add fields to the table
        // Swing layout: constraints.gridwidth = 1; the last cell gets
        // constraints.gridwidth = GridBagConstraints.REMAINDER.
        let cells = self.temp_displayed_fields.borrow().clone();
        for cell in cells.iter() {
            cell.add(&panel);
        }
        self.temp_displayed_fields.borrow_mut().clear();
    }

    /// Java `performAction()`.
    pub fn perform_action(&self) {
        self.update_selected();
    }

    // Java public `focusGained(FocusEvent)` and `focusLost(FocusEvent)` are empty
    // and nothing registers this row as a FocusListener.

    /// Java `stateChangedCPU()`.
    pub fn state_changed_cpu(&self) {
        self.table().processor_table().msg_cpus_selected_changed();
    }

    /// Java `stateChangedComputer()`.
    pub fn state_changed_computer(&self) {
        if !self.display_queues {
            return;
        }
        // handle radio button changes
        self.update_selected();
    }

    /// Java final `msgDropped(String)`.
    pub fn msg_dropped(&self, reason: Option<&str>) {
        self.set_selected(false);
        if let Some(cell_failure_reason) = &self.cell_failure_reason {
            cell_failure_reason.set_value_string(reason);
            cell_failure_reason.set_tool_tip_text(Some(
                "This computer was dropped from the current distributed process.",
            ));
        }
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        // Do not allow the row to be unselected if it is disabled and the only row.
        if !selected && self.num_rows_in_table == 1 && !self.cell_computer.is_enabled() {
            return;
        }
        self.cell_computer.set_selected(selected);
        self.update_selected();
    }

    /// Java `enableSelectionField(boolean)`.
    pub fn enable_selection_field(&self, enabled: bool) {
        self.selection_enabled.set(enabled);
        self.update_display();
    }

    /// Java `setCPUsSelected(String)`.
    pub fn set_cpus_selected(&self, s_cpus_selected: Option<&str>) {
        let Some(mut cpus_selected) = converter::to_integer(s_cpus_selected) else {
            return;
        };
        match &self.cell_cpus_selected_type {
            CpusSelectedCell::Field(field) => field.set_value_int(cpus_selected),
            CpusSelectedCell::Spinner(spinner) => {
                if self.dual_selection_queue_table && cpus_selected < DUAL_SELECTION_MIN {
                    cpus_selected = DUAL_SELECTION_MIN;
                }
                spinner.set_value_int(cpus_selected);
            }
        }
    }

    /// Java `hasSecondaryQueue()`.
    pub fn has_secondary_queue(&self) -> bool {
        self.cell_secondary_queue.is_some()
    }

    /// Java `setSecondaryQueueSelected()`.
    pub fn set_secondary_queue_selected(&self) {
        if let Some(cell_secondary_queue) = &self.cell_secondary_queue {
            cell_secondary_queue.set_selected(true);
        }
    }

    /// Java `setParameters()`.
    pub fn set_parameters(&self) {
        self.cell_computer.set_selected(false);
    }

    /// Java private `updateSelected()`.
    fn update_selected(&self) {
        self.set_selected_error();
        self.table().processor_table().msg_cpus_selected_changed();
        self.update_display();
    }

    /// Java `updateDisplay()`.
    pub fn update_display(&self) {
        if self.table().is_queue_table() {
            let queue_type = self.queue_type.get();
            self.cell_computer.set_enabled(
                self.selection_enabled.get()
                    && (!self.dual_selection_queue_table
                        || (self.queue_mode != Some(QueueMode::Invalid)
                            && (self.queue_mode.is_none()
                                || queue_type.is_none()
                                || self.queue_mode.unwrap().is_type(queue_type.unwrap())))),
            );
        } else {
            self.cell_computer.set_enabled(self.selection_enabled.get());
        }
        if let Some(cell_secondary_queue) = &self.cell_secondary_queue {
            cell_secondary_queue.set_enabled(self.enable_secondary_queue.get());
        }
        let mut selected = self.is_selected();
        CellVirtual::set_enabled(&*self.cell_cpus_selected, selected);
        if let CpusSelectedCell::Field(field) = &self.cell_cpus_selected_type {
            if !selected {
                field.set_value_int(0);
            } else if !self.dual_selection_queue_table {
                field.set_value_int(1);
            }
        }
        if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue {
            let prev_selected = cell_max_gpu_jobs_on_queue.is_enabled();
            selected = self.is_secondary_queue_selected();
            if prev_selected != selected {
                cell_max_gpu_jobs_on_queue.set_enabled(selected);
                if selected {
                    if let Some(max_gpu_jobs_on_queue) = self.max_gpu_jobs_on_queue.get() {
                        cell_max_gpu_jobs_on_queue.set_value_int(max_gpu_jobs_on_queue);
                    } else {
                        cell_max_gpu_jobs_on_queue.set_value_int(1);
                    }
                } else {
                    self.max_gpu_jobs_on_queue
                        .set(Some(cell_max_gpu_jobs_on_queue.get_int_value()));
                    cell_max_gpu_jobs_on_queue.set_value_int(0);
                }
            }
        }
    }

    /// Java `setSelectedError()`.
    pub fn set_selected_error(&self) {
        if self.table().processor_table().is_secondary() {
            self.cell_computer.set_warning(false);
            return;
        }
        let mut noload_average = false;
        if self.display_queues && self.cell_load_array.is_some() {
            let cell0 = &self.cell_load_array.as_ref().unwrap()[0];
            noload_average = cell0.is_empty() || cell0.equals(Some("NA"));
        } else if utilities::is_windows_os() && self.cell_cpu_usage.is_some() {
            noload_average = self.cell_cpu_usage.as_ref().unwrap().is_empty();
        } else if let Some(cell_load1) = &self.cell_load1 {
            noload_average = cell_load1.is_empty();
        }
        self.cell_computer
            .set_warning(self.is_selected() && noload_average);
    }

    /// Java final `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.cell_computer.is_enabled() && self.cell_computer.is_selected()
    }

    /// Java final `isSecondaryQueueSelected()`.
    pub fn is_secondary_queue_selected(&self) -> bool {
        match &self.cell_secondary_queue {
            Some(cell_secondary_queue) => {
                cell_secondary_queue.is_enabled() && cell_secondary_queue.is_selected()
            }
            None => false,
        }
    }

    /// Java final `getParameters(ProcesschunksParam, boolean addGPUMachine, boolean
    /// secondaryTable)`.  If secondaryTable is true, most parameters will not be
    /// loaded to avoid overriding values from the primary table.  But GPU parameters
    /// that must be loaded can only come from the secondary table if the primary
    /// table is CPU.
    ///
    /// Upstream behaviour: `ProcesschunksParam.addMachineName` throws
    /// `IllegalStateException` once the command has been built, which the Java lets
    /// escape (an uncaught exception on the event thread).  The translation reports
    /// the message on standard error and continues.
    pub fn get_parameters_processchunks_param_boolean_boolean(
        &self,
        param: &ProcesschunksParam,
        add_gpu_machine: bool,
        secondary_table: bool,
    ) {
        let table = self.table();
        if table.is_queue_table() && self.is_selected() {
            param.set_queue_mode(self.queue_mode);
        }
        let num_cpus = self.get_cpus_selected();
        if num_cpus > 0 {
            // Java string conversion of a null label gives "null".
            let label = self.get_computer().unwrap_or_else(|| "null".to_string());
            if !secondary_table {
                if let Err(message) =
                    param.add_machine_name(&label, num_cpus, self.gpu_device_array.clone())
                {
                    eprintln!("java.lang.IllegalStateException: {message}");
                }
            }
            // If the GPU machine list is being added, it may come from either the
            // primary or secondary table.
            if add_gpu_machine && table.is_gpu_table() {
                param.add_gpu_machine(Some(&label), num_cpus, self.gpu_device_array.as_deref());
            }
        }
    }

    /// Java final `getSecondaryParameters(BatchruntomoParam)`.
    pub fn get_secondary_parameters_batchruntomo_param(&self, param: &mut BatchruntomoParam) {
        if self.table().is_queue_table() && self.is_secondary_queue_selected() {
            if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue {
                param.set_max_gpu_jobs_on_queue_string(
                    cell_max_gpu_jobs_on_queue.get_text().as_deref(),
                );
            } else {
                param.set_max_gpu_jobs_on_queue_int(1);
            }
        }
    }

    /// Java final `getSecondaryParameters(ProcesschunksParam)`.
    pub fn get_secondary_parameters_processchunks_param(&self, param: &ProcesschunksParam) {
        if self.table().is_queue_table() && self.is_secondary_queue_selected() {
            if let Some(cell_max_gpu_jobs_on_queue) = &self.cell_max_gpu_jobs_on_queue {
                param.set_secondary_number_string(cell_max_gpu_jobs_on_queue.get_text().as_deref());
            } else {
                param.set_secondary_number_int(1);
            }
        }
    }

    /// Java final `getParameters(BatchruntomoParam)`.
    pub fn get_parameters_batchruntomo_param(&self, param: &mut BatchruntomoParam) {
        let table = self.table();
        if table.is_queue_table() && self.is_selected() {
            param.set_queue_mode(self.queue_mode);
            match &self.cell_cpus_selected_type {
                CpusSelectedCell::Spinner(spinner) => {
                    param.set_max_jobs_on_queue_int(spinner.get_int_value());
                }
                CpusSelectedCell::Field(field) => {
                    param.set_max_jobs_on_queue_string(field.get_value().as_deref());
                }
            }
        } else {
            let num_cpus = self.get_cpus_selected();
            if num_cpus > 0 {
                if table.is_cpu_table() {
                    param.add_cpu_machine(self.get_computer().as_deref(), num_cpus);
                } else if table.is_gpu_table() {
                    param.add_gpu_machine(
                        self.get_computer().as_deref(),
                        num_cpus,
                        self.gpu_device_array.as_deref(),
                    );
                }
            }
        }
    }

    /// Java `getSuccesses()`.
    pub fn get_successes(&self) -> i32 {
        if let Some(cell_successes) = &self.cell_successes {
            return cell_successes.get_int_value();
        }
        0
    }

    /// Java `getCPUsSelected()`.
    pub fn get_cpus_selected(&self) -> i32 {
        if !self.is_selected() {
            return 0;
        }
        match &self.cell_cpus_selected_type {
            CpusSelectedCell::Spinner(spinner) => spinner.get_int_value(),
            CpusSelectedCell::Field(field) => {
                let mut cpus_selected = field.get_int_value();
                if cpus_selected == const_etomo_number::INTEGER_NULL_VALUE {
                    cpus_selected = 0;
                }
                cpus_selected
            }
        }
    }

    /// Java public `equals(String)`.
    ///
    /// Upstream bug fixed in translation (`ProcessorTableRow.java:908`): the Java
    /// calls `getComputer().equals(computer)`, which throws NullPointerException when
    /// the row has no label; such a row matches nothing here.
    pub fn equals(&self, computer: Option<&str>) -> bool {
        match self.get_computer() {
            Some(this_computer) => Some(this_computer.as_str()) == computer,
            None => false,
        }
    }

    /// Java `addSuccess()`.
    pub fn add_success(&self) {
        if let Some(cell_successes) = &self.cell_successes {
            let mut successes = cell_successes.get_int_value();
            if successes == const_etomo_number::INTEGER_NULL_VALUE {
                successes = 1;
            } else {
                successes = successes.wrapping_add(1);
            }
            cell_successes.set_value_int(successes);
        }
    }

    /// Java `resetResults()`.
    pub fn reset_results(&self) {
        if let Some(cell_restarts) = &self.cell_restarts {
            self.cell_successes.as_ref().unwrap().set_value_void();
            cell_restarts.set_value_void();
            cell_restarts.set_error_boolean(false);
            cell_restarts.set_warning_boolean(false);
        }
    }

    /// Java `addRestart()`.
    pub fn add_restart(&self) {
        if let Some(cell_restarts) = &self.cell_restarts {
            let mut restarts = cell_restarts.get_int_value();
            if restarts == const_etomo_number::INTEGER_NULL_VALUE {
                restarts = 1;
            } else {
                restarts = restarts.wrapping_add(1);
            }
            cell_restarts.set_value_int(restarts);
            if restarts >= processchunks_param::DROP_VALUE {
                cell_restarts.set_error_boolean(true);
            } else if restarts > 0 {
                cell_restarts.set_warning_boolean(true);
            }
        }
    }

    /// Java `setLoad(double, double, int, String)`.
    pub fn set_load_double_double_int_string(
        &self,
        load1: f64,
        load5: f64,
        users: i32,
        users_tooltip: Option<&str>,
    ) {
        self.set_load_field_cell_double_int(self.cell_load1.as_ref(), load1, self.num_cpus);
        self.set_load_field_cell_double_int(self.cell_load5.as_ref(), load5, self.num_cpus);
        self.cell_computer.set_warning(false);
        if let Some(cell_users) = &self.cell_users {
            cell_users.set_value_int(users);
            cell_users.set_tool_tip_text(users_tooltip);
        }
    }

    /// Java `setLoad(String[])`.
    pub fn set_load_string_array(&self, load_array: &[String]) {
        if let Some(cell_load_array) = &self.cell_load_array {
            for (i, load) in load_array.iter().enumerate() {
                if i < cell_load_array.len() {
                    cell_load_array[i].set_value_string(Some(load));
                }
            }
        }
        self.cell_computer.set_warning(false);
    }

    /// Java `setCPUUsage(double, ConstEtomoNumber)`.
    pub fn set_cpu_usage(&self, cpu_usage: f64, number_of_processors: Option<&ConstEtomoNumber>) {
        let usage = match number_of_processors {
            Some(number_of_processors) if !number_of_processors.is_null() => {
                cpu_usage * number_of_processors.get_int() as f64 / 100.0
            }
            _ => cpu_usage / 100.0,
        };
        if let Some(cell_cpu_usage) = &self.cell_cpu_usage {
            if self.load_warning.get() {
                cell_cpu_usage.set_warning_boolean(cpu_usage > 75.0);
            }
            cell_cpu_usage.set_value_double(usage);
        }
        self.cell_computer.set_warning(false);
    }

    /// Java final `clearLoad(String, String)`.
    pub fn clear_load(&self, reason: Option<&str>, tooltip: Option<&str>) {
        // Java assigns loadName ("CPU usage" / "load averages") and never reads it.
        if utilities::is_windows_os() && self.cell_cpu_usage.is_some() {
            let _load_name = "CPU usage";
            let cell_cpu_usage = self.cell_cpu_usage.as_ref().unwrap();
            cell_cpu_usage.set_value_void();
            cell_cpu_usage.set_warning_boolean(false);
        } else {
            let _load_name = "load averages";
            if let Some(cell_load1) = &self.cell_load1 {
                cell_load1.set_value_void();
                cell_load1.set_warning_boolean(false);
            }
            if let Some(cell_load5) = &self.cell_load5 {
                cell_load5.set_value_void();
                cell_load5.set_warning_boolean(false);
            }
            if let Some(cell_users) = &self.cell_users {
                cell_users.set_value_void();
            }
        }
        self.set_selected_error();
        if let Some(cell_failure_reason) = &self.cell_failure_reason {
            cell_failure_reason.set_value_string(reason);
            cell_failure_reason.set_tool_tip_text(tooltip);
        }
    }

    /// Java final `clearFailureReason(String, String)`.  Clear failure reason, if
    /// failure reason equals failureReason1 or 2.  This means that processes can only
    /// clear their own messages.  This is useful for restarting an intermittent
    /// process without losing the processchunks failure reason.
    pub fn clear_failure_reason_string_string(
        &self,
        failure_reason1: Option<&str>,
        failure_reason2: Option<&str>,
    ) {
        if let Some(cell_failure_reason) = &self.cell_failure_reason {
            let value = cell_failure_reason.get_value();
            // `value.equals(null)` is false.
            match value.as_deref() {
                None => return,
                Some(value) if Some(value) != failure_reason1 && Some(value) != failure_reason2 => {
                    return;
                }
                _ => {}
            }
            self.clear_failure_reason_void();
        }
    }

    /// Java final `clearFailureReason()`.
    pub fn clear_failure_reason_void(&self) {
        if let Some(cell_failure_reason) = &self.cell_failure_reason {
            cell_failure_reason.set_value_void();
            cell_failure_reason.set_tool_tip_text(None);
        }
    }

    /// Java private final `setLoad(FieldCell, double, int)`.
    fn set_load_field_cell_double_int(
        &self,
        cell_load: Option<&Rc<FieldCell>>,
        load: f64,
        number_cpus: i32,
    ) {
        let Some(cell_load) = cell_load else {
            return;
        };
        if self.load_warning.get() {
            cell_load.set_warning_boolean(load >= number_cpus as f64);
        }
        cell_load.set_value_double(load);
    }

    /// Java final `getHeight()`.
    pub fn get_height(&self) -> i32 {
        self.cell_computer.get_height()
    }

    /// Java final `getQueueMode()`.
    pub fn get_queue_mode(&self) -> Option<QueueMode> {
        self.queue_mode
    }

    /// Java final `getComputer()`.
    pub fn get_computer(&self) -> Option<String> {
        if !self.dual_selection_queue_table {
            return self.cell_computer.get_label();
        }
        // Java dereferences cellQueueName unconditionally; see `init_row`.
        self.cell_queue_name
            .as_ref()
            .and_then(|cell| cell.get_text())
    }
}

/// Java `implements Storable`.
impl Storable for ProcessorTableRow {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_properties(properties);
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.store_properties_string(properties, prepend);
    }
    fn load(&self, properties: &mut BTreeMap<String, String>) {
        self.load_properties(properties);
    }
    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.load_properties_string(properties, prepend);
    }
}
