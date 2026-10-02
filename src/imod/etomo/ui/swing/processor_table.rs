//! `IMOD/Etomo/src/etomo/ui/swing/ProcessorTable.java`.
//!
//! Java `public abstract class ProcessorTable implements Storable,
//! ParallelProgressDisplay, LoadDisplay, Viewable`: the table of computers, GPUs or
//! queues in the parallel processing panel.  Concrete tables: `CpuTable`,
//! `GpuTable` (extends `CpuTable`) and `QueueTable`.
//!
//! Object model (ui.md):
//! * A concrete table embeds [`ProcessorTable`] as its `base` field (`GpuTable`
//!   through `CpuTable`), implements [`ProcessorTableVirtual`] for the abstract and
//!   overridden methods, and implements `ParallelProgressDisplay`, `LoadDisplay` and
//!   `Viewable` with [`processor_table_interfaces!`], which binds them to the base
//!   class's bodies.  Code holding a Java `ProcessorTable` holds
//!   `Rc<dyn ProcessorTableVirtual>`; the interfaces are reached from it by trait
//!   upcasting (`Rc<dyn ParallelProgressDisplay>` etc.).
//! * Java runs the field initializers and the constructor body with `this` already
//!   the subclass (the constructor calls `getHeader1ComputerText()`,
//!   `getProcessorType()`, ... and hands `this` to the viewport, the load monitor and
//!   the table state).  So the subclass allocates itself first
//!   ([`ProcessorTable::new`] holds only the constructor's parameters and the
//!   defaults), connects `this` with [`ProcessorTable::set_this`], and then runs
//!   [`ProcessorTable::construct`] - the field initializers and constructor body.
//!   The fields the constructor assigns are `OnceCell`s; an unset `OnceCell` is
//!   Java's `null`.
//! * The table is owned by its `ParallelPanel`, so it holds the panel weakly.
//! * The inner class `RowList` accesses its outer instance; its methods take the
//!   outer `ProcessorTable` as a parameter.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::cell::CellVirtual;
use super::gpu_table;
use super::header_cell::HeaderCell;
use super::load_display::LoadDisplay;
use super::parallel_panel::ParallelPanel;
use super::parallel_progress_display::ParallelProgressDisplay;
use super::processor_table_row::ProcessorTableRow;
use super::ui_harness;
use super::viewable::Viewable;
use super::viewport::Viewport;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::processor_table_state::ProcessorTableState;
use crate::imod::etomo::logic::processor_type::ProcessorType;
use crate::imod::etomo::process::load_average_monitor::LoadAverageMonitor;
use crate::imod::etomo::process::load_monitor::LoadMonitor;
use crate::imod::etomo::process::queuechunk_load_monitor::QueuechunkLoadMonitor;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::node::Node;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::expander::Expander;
use crate::imod::etomo::ui::processor_table_field::ProcessorTableField;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::ui::table_field::TableField;
use crate::imod::etomo::util::event_queue::EdtRef;

/// Java `RUNNABLE_KEY`.
const RUNNABLE_KEY: &str = "ProcessorTable";
/// Java `RESOURCE_KEY`.
const RESOURCE_KEY: &str = "ProcessorResourceTable";
/// Java `CPU_TYPE_LABEL`.
pub const CPU_TYPE_LABEL: &str = "CPU Type";
/// Java `RESTARTS_LABEL`.
pub const RESTARTS_LABEL: &str = "Retries";
/// Java `NUMBER_CPUS_MAX_LABEL`.
pub const NUMBER_CPUS_MAX_LABEL: &str = "Max.";
/// Java `FIRST_QUEUE_LABEL`.
pub const FIRST_QUEUE_LABEL: &str = "1st";
/// Java `SECONDARY_QUEUE_LABEL`.
pub const SECONDARY_QUEUE_LABEL: &str = "2nd";
/// Java `NUMBER_CPUS_USED_LABEL2`.
pub const NUMBER_CPUS_USED_LABEL2: &str = "Used";

/// Java `Properties`.
pub type Properties = BTreeMap<String, String>;

/// The abstract and overridden methods of Java `ProcessorTable`, plus the interfaces
/// the class implements (as supertraits, so that a `Rc<dyn ProcessorTableVirtual>`
/// upcasts to each of them).
pub trait ProcessorTableVirtual: ParallelProgressDisplay + LoadDisplay + Viewable {
    /// The embedded `ProcessorTable` (Java `this` seen as a `ProcessorTable`).
    fn processor_table(&self) -> &ProcessorTable;

    /// Java abstract `getSize()`.
    fn get_size(&self) -> i32;

    /// Java abstract `getNode(int)`.
    fn get_node(&self, index: i32) -> Option<Arc<Node>>;

    /// Java abstract `createProcessorTableRow(ProcessorTable, Node, int,
    /// ProcessorTableState)`.
    fn create_processor_table_row(
        &self,
        processor_table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow>;

    /// Java abstract `getHeader1ComputerText()`.
    fn get_header1_computer_text(&self) -> Option<String>;

    /// Java abstract `getIntermittentCommand(String)`.
    fn get_intermittent_command_string(
        &self,
        computer: Option<&str>,
    ) -> Arc<dyn IntermittentCommand>;

    /// Java abstract `isExcludeNode(Node)`.
    fn is_exclude_node(&self, node: &Node) -> bool;

    /// Java abstract `isNiceable()`.
    fn is_niceable(&self) -> bool;

    /// Java abstract `getStorePrepend()`.
    fn get_store_prepend(&self) -> String;

    /// Java abstract `getLoadPrepend(ConstEtomoVersion)`.
    fn get_load_prepend(&self, version: &dyn ConstEtomoVersion) -> String;

    /// Java abstract `initRow(ProcessorTableRow)`.
    fn init_row(&self, row: &Rc<ProcessorTableRow>);

    /// Java abstract `getNoCpusSelectedErrorMessage()`.
    fn get_no_cpus_selected_error_message(&self) -> Option<String>;

    /// Java abstract `isQueueTable()`.
    fn is_queue_table(&self) -> bool;

    /// Java abstract `isCpuTable()`.
    fn is_cpu_table(&self) -> bool;

    /// Java abstract `isGpuTable()`.
    fn is_gpu_table(&self) -> bool;

    /// Java abstract `getParameters(ProcessingMethod, BatchruntomoParam, boolean)`.
    fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        do_validation: bool,
    ) -> bool;

    /// Java abstract `getProcessorType()`.
    fn get_processor_type(&self) -> ProcessorType;

    /// Java `getheader1NumberCPUsTitle()` (overridden by `GpuTable`).
    fn getheader1_number_cpus_title(&self) -> Option<String> {
        self.processor_table().getheader1_number_cpus_title_super()
    }

    /// Java `getParameters(ProcesschunksParam)` (overridden by `GpuTable` and
    /// `QueueTable`).
    fn get_parameters_processchunks_param(&self, param: &ProcesschunksParam) {
        self.processor_table()
            .get_parameters_processchunks_param_super(param);
    }

    /// Java `getParameters(BatchruntomoParam)` (overridden by `QueueTable`).
    fn get_parameters_batchruntomo_param(&self, param: &mut BatchruntomoParam) {
        self.processor_table()
            .get_parameters_batchruntomo_param_super(param);
    }

    /// Java `getMachineMap(BatchruntomoParam)` (overridden by `GpuTable`).
    fn get_machine_map(&self, param: &BatchruntomoParam) -> Option<HashMap<String, String>> {
        self.processor_table().get_machine_map_super(param)
    }

    /// Java `actionPerformed(ActionEvent)` (overridden by `QueueTable`).  Also the
    /// body of `ParallelProgressDisplay.actionPerformed`.
    fn action_performed_virtual(&self, _event: &ActionEvent) {}

    /// Java `addQueueTableListener(QueueTableListener)` (overridden by
    /// `QueueTable`).
    fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java `sendQueueTableEvent(QueueTableEvent)` (overridden by `QueueTable`).
    fn send_queue_table_event(&self, _event: &QueueTableEvent) {}

    /// Java `removeQueueTableListener(QueueTableListener)` (overridden by
    /// `QueueTable`).
    fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}

    /// Java `queueTableEventAction(QueueTableEvent)` (overridden by `QueueTable`).
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        self.processor_table().queue_table_event_action_super(event);
    }

    /// Java public `enableGpuQueueRows()` (overridden by `QueueTable`).
    fn enable_gpu_queue_rows(&self) {}
}

/// Binds `ParallelProgressDisplay`, `LoadDisplay` and `Viewable` of a concrete
/// table to `ProcessorTable`'s bodies (Java: the methods are inherited).
#[macro_export]
macro_rules! processor_table_interfaces {
    ($table:ty) => {
        impl $crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay
            for $table
        {
            fn action_performed(&self, event: &$crate::imod::etomo::jdk::ActionEvent) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::action_performed_virtual(self, event);
            }
            fn msg_dropped(&self, computer: Option<&str>, reason: Option<&str>) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_dropped(computer, reason);
            }
            fn add_success(&self, computer: Option<&str>) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).add_success(computer);
            }
            fn add_restart(&self, computer: Option<&str>) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).add_restart(computer);
            }
            fn msg_killing_process(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_killing_process();
            }
            fn msg_pausing_process(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_pausing_process();
            }
            fn msg_starting_process_on_selected_computers(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_starting_process_on_selected_computers();
            }
            fn msg_ending_process(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_ending_process();
            }
            fn reset_results(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).reset_results();
            }
            fn set_computer_map(
                &self,
                computer_map: Option<&std::collections::HashMap<String, String>>,
            ) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).set_computer_map(computer_map);
            }
            fn set_secondary_queue(&self, secondary_queue: Option<&str>) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).set_secondary_queue(secondary_queue);
            }
            fn msg_process_started(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_process_started();
            }
            fn is_secondary(&self) -> bool {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).is_secondary()
            }
            fn is_runnable(&self) -> bool {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).is_runnable()
            }
            fn is_limited(&self) -> bool {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).is_limited()
            }
        }

        impl $crate::imod::etomo::ui::swing::load_display::LoadDisplay for $table {
            fn set_load_string_double_double_int_string(
                &self,
                computer: Option<&str>,
                load1: f64,
                load5: f64,
                users: i32,
                users_tooltip: Option<&str>,
            ) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self)
                    .set_load_string_double_double_int_string(computer, load1, load5, users, users_tooltip);
            }
            fn msg_load_failed(
                &self,
                computer: Option<&str>,
                reason: Option<&str>,
                tooltip: Option<&str>,
            ) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_load_failed(computer, reason, tooltip);
            }
            fn msg_starting_process(
                &self,
                computer: Option<&str>,
                failure_reason1: Option<&str>,
                failure_reason2: Option<&str>,
            ) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self)
                    .msg_starting_process(computer, failure_reason1, failure_reason2);
            }
            fn set_cpu_usage(
                &self,
                computer: Option<&str>,
                cpu_usage: f64,
                number_of_processors: Option<&$crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber>,
            ) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self)
                    .set_cpu_usage(computer, cpu_usage, number_of_processors);
            }
            fn start_load(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).start_load();
            }
            fn stop_load(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).stop_load();
            }
            fn end_load(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).end_load();
            }
            fn set_load_string_string_array(&self, computer: Option<&str>, load_array: &[String]) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self)
                    .set_load_string_string_array(computer, load_array);
            }
        }

        impl $crate::imod::etomo::ui::swing::viewable::Viewable for $table {
            fn msg_viewport_paged(&self) {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).msg_viewport_paged();
            }
            fn size(&self) -> i32 {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).size()
            }
            fn get_focusable_parents(&self) -> Vec<std::rc::Rc<$crate::imod::etomo::jdk::JComponent>> {
                $crate::imod::etomo::ui::swing::processor_table::ProcessorTableVirtual::processor_table(self).get_focusable_parents()
            }
        }
    };
}

/// Java `ProcessorTable`.
pub struct ProcessorTable {
    /// Java `this`, seen through the subclass's overrides.
    this: RefCell<Option<Weak<dyn ProcessorTableVirtual>>>,
    /// Java `rootPanel`.
    root_panel: Rc<JComponent>,
    /// Java `header1Computer` (its initializer calls `getHeader1ComputerText()`).
    header1_computer: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Computer`.
    header2_computer: Rc<HeaderCell>,
    /// Java `header2NumberCPUsUsed`.
    header2_number_cpus_used: Rc<HeaderCell>,
    /// Java `rowList`.
    row_list: RowList,
    /// Java `viewport` (its initializer passes `this`).
    viewport: OnceCell<Rc<Viewport>>,
    /// Java `tempDisplayedFields`: temporary storage of the displayed fields in the
    /// current row being built.
    temp_displayed_fields: RefCell<Vec<Rc<dyn CellVirtual>>>,
    /// Java `focusableParents`.
    focusable_parents: Vec<Rc<JComponent>>,

    /// Java `axisID`.
    pub(crate) axis_id: AxisID,
    /// Java `manager`.
    pub(crate) manager: &'static dyn BaseManager,
    /// Java `parent`.
    pub(crate) parent: Weak<ParallelPanel>,

    /// Java `tableState`.
    table_state: OnceCell<Rc<ProcessorTableState>>,
    /// Java `loadMonitor` (null when etomo runs with `--noload`).
    load_monitor: OnceCell<Option<Arc<dyn LoadMonitor>>>,
    /// Java `header2PrimaryQueue` (unset: null).
    header2_primary_queue: OnceCell<Rc<HeaderCell>>,
    /// Java `header2SecondaryQueue`.
    header2_secondary_queue: OnceCell<Rc<HeaderCell>>,
    /// Java `header1NumberCPUs`.
    header1_number_cpus: OnceCell<Rc<HeaderCell>>,
    /// Java `header2NumberCPUsMax`.
    header2_number_cpus_max: OnceCell<Rc<HeaderCell>>,
    /// Java `header1NumberGPUs`.
    header1_number_gpus: OnceCell<Rc<HeaderCell>>,
    /// Java `header2NumberGPUs`.
    header2_number_gpus: OnceCell<Rc<HeaderCell>>,
    /// Java `header1Load`.
    header1_load: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Load1`.
    header2_load1: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Load5`.
    header2_load5: OnceCell<Rc<HeaderCell>>,
    /// Java `header1CPUUsage`.
    header1_cpu_usage: OnceCell<Rc<HeaderCell>>,
    /// Java `header2CPUUsage`.
    header2_cpu_usage: OnceCell<Rc<HeaderCell>>,
    /// Java `header1LoadArray`.
    header1_load_array: OnceCell<Vec<Rc<HeaderCell>>>,
    /// Java `header2LoadArray`.
    header2_load_array: OnceCell<Vec<Rc<HeaderCell>>>,
    /// Java `header1Users`.
    header1_users: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Users`.
    header2_users: OnceCell<Rc<HeaderCell>>,
    /// Java `header1CPUType`.
    header1_cpu_type: OnceCell<Rc<HeaderCell>>,
    /// Java `header2CPUType`.
    header2_cpu_type: OnceCell<Rc<HeaderCell>>,
    /// Java `header1Speed`.
    header1_speed: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Speed`.
    header2_speed: OnceCell<Rc<HeaderCell>>,
    /// Java `header1RAM`.
    header1_ram: OnceCell<Rc<HeaderCell>>,
    /// Java `header2RAM`.
    header2_ram: OnceCell<Rc<HeaderCell>>,
    /// Java `header1OS`.
    header1_os: OnceCell<Rc<HeaderCell>>,
    /// Java `header2OS`.
    header2_os: OnceCell<Rc<HeaderCell>>,
    /// Java `header1Restarts`.
    header1_restarts: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Restarts`.
    header2_restarts: OnceCell<Rc<HeaderCell>>,
    /// Java `header1Finished`.
    header1_finished: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Finished`.
    header2_finished: OnceCell<Rc<HeaderCell>>,
    /// Java `header1Failure`.
    header1_failure: OnceCell<Rc<HeaderCell>>,
    /// Java `header2Failure`.
    header2_failure: OnceCell<Rc<HeaderCell>>,
    /// Java `header1GPUType`.
    header1_gpu_type: OnceCell<Rc<HeaderCell>>,
    /// Java `header2GPUType`.
    header2_gpu_type: OnceCell<Rc<HeaderCell>>,
    /// Java `header1GPUNcores`.
    header1_gpu_ncores: OnceCell<Rc<HeaderCell>>,
    /// Java `header2GPUNcores`.
    header2_gpu_ncores: OnceCell<Rc<HeaderCell>>,
    /// Java `header1GPUSpeed`.
    header1_gpu_speed: OnceCell<Rc<HeaderCell>>,
    /// Java `header2GPUSpeed`.
    header2_gpu_speed: OnceCell<Rc<HeaderCell>>,
    /// Java `header1GPURAM`.
    header1_gpu_ram: OnceCell<Rc<HeaderCell>>,
    /// Java `header2GPURAM`.
    header2_gpu_ram: OnceCell<Rc<HeaderCell>>,

    /// Java `runnable`.
    runnable: Cell<bool>,

    /// Java `tablePanel`.
    table_panel: RefCell<Option<Rc<JComponent>>>,
    // Java `layout` (GridBagLayout) and `constraints` (GridBagConstraints): layout
    // only, not modelled.  `getTableLayout()` and `getTableConstraints()` are
    // therefore absent; the rows add their cells to `getTablePanel()`.
    /// Java `scrolling`.
    scrolling: Cell<bool>,
    /// Java `expanded`.
    expanded: Cell<bool>,
    /// Java `stopped`.
    stopped: Cell<bool>,
    /// Java `secondary`.
    secondary: Cell<bool>,
    /// Java `limited`.
    limited: Cell<bool>,
}

/// A header that the constructor always assigns.
fn header(cell: &OnceCell<Rc<HeaderCell>>) -> &Rc<HeaderCell> {
    cell.get()
        .expect("ProcessorTable: header assigned by the constructor")
}

/// `Rc<dyn TableField>` for a `ProcessorTableField` constant.
fn field(table_field: ProcessorTableField) -> Option<Rc<dyn TableField>> {
    Some(Rc::new(table_field))
}

impl ProcessorTable {
    /// The part of Java `ProcessorTable(BaseManager, ParallelPanel, AxisID, boolean,
    /// boolean, Expander, InterfaceType)` that needs no `this`: the parameters it
    /// stores and the field initializers that do not dispatch.  The subclass then
    /// calls [`set_this`](Self::set_this) and [`construct`](Self::construct).
    pub fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<ParallelPanel>,
        axis_id: AxisID,
        runnable: bool,
    ) -> ProcessorTable {
        let root_panel = JComponent::new_panel();
        ProcessorTable {
            this: RefCell::new(None),
            focusable_parents: vec![root_panel.clone()],
            root_panel,
            header1_computer: OnceCell::new(),
            header2_computer: HeaderCell::new_void(),
            header2_number_cpus_used: HeaderCell::new_string(Some(NUMBER_CPUS_USED_LABEL2)),
            row_list: RowList::new(),
            viewport: OnceCell::new(),
            temp_displayed_fields: RefCell::new(Vec::new()),
            axis_id,
            manager,
            parent,
            table_state: OnceCell::new(),
            load_monitor: OnceCell::new(),
            header2_primary_queue: OnceCell::new(),
            header2_secondary_queue: OnceCell::new(),
            header1_number_cpus: OnceCell::new(),
            header2_number_cpus_max: OnceCell::new(),
            header1_number_gpus: OnceCell::new(),
            header2_number_gpus: OnceCell::new(),
            header1_load: OnceCell::new(),
            header2_load1: OnceCell::new(),
            header2_load5: OnceCell::new(),
            header1_cpu_usage: OnceCell::new(),
            header2_cpu_usage: OnceCell::new(),
            header1_load_array: OnceCell::new(),
            header2_load_array: OnceCell::new(),
            header1_users: OnceCell::new(),
            header2_users: OnceCell::new(),
            header1_cpu_type: OnceCell::new(),
            header2_cpu_type: OnceCell::new(),
            header1_speed: OnceCell::new(),
            header2_speed: OnceCell::new(),
            header1_ram: OnceCell::new(),
            header2_ram: OnceCell::new(),
            header1_os: OnceCell::new(),
            header2_os: OnceCell::new(),
            header1_restarts: OnceCell::new(),
            header2_restarts: OnceCell::new(),
            header1_finished: OnceCell::new(),
            header2_finished: OnceCell::new(),
            header1_failure: OnceCell::new(),
            header2_failure: OnceCell::new(),
            header1_gpu_type: OnceCell::new(),
            header2_gpu_type: OnceCell::new(),
            header1_gpu_ncores: OnceCell::new(),
            header2_gpu_ncores: OnceCell::new(),
            header1_gpu_speed: OnceCell::new(),
            header2_gpu_speed: OnceCell::new(),
            header1_gpu_ram: OnceCell::new(),
            header2_gpu_ram: OnceCell::new(),
            runnable: Cell::new(runnable),
            table_panel: RefCell::new(None),
            scrolling: Cell::new(false),
            expanded: Cell::new(false),
            stopped: Cell::new(true),
            secondary: Cell::new(false),
            limited: Cell::new(false),
        }
    }

    /// Connects this `ProcessorTable` to the subclass that embeds it (Java `this`).
    pub fn set_this(&self, this: Weak<dyn ProcessorTableVirtual>) {
        *self.this.borrow_mut() = Some(this);
    }

    /// Java `this` as the subclass.
    pub fn this(&self) -> Rc<dyn ProcessorTableVirtual> {
        self.this
            .borrow()
            .as_ref()
            .and_then(Weak::upgrade)
            .expect("ProcessorTable: set_this was not called by the subclass constructor")
    }

    /// Java `parent`.  The panel owns this table, so it is alive whenever the table
    /// is used.
    fn parent(&self) -> Rc<ParallelPanel> {
        self.parent
            .upgrade()
            .expect("ProcessorTable: the ParallelPanel that owns this table is gone")
    }

    fn viewport(&self) -> &Rc<Viewport> {
        self.viewport
            .get()
            .expect("ProcessorTable: viewport assigned by the constructor")
    }

    fn table_state(&self) -> &Rc<ProcessorTableState> {
        self.table_state
            .get()
            .expect("ProcessorTable: tableState assigned by the constructor")
    }

    fn load_monitor(&self) -> Option<Arc<dyn LoadMonitor>> {
        self.load_monitor.get().cloned().flatten()
    }

    /// The field initializers that dispatch or pass `this`, and the body, of Java
    /// `ProcessorTable(BaseManager, ParallelPanel, AxisID, boolean displayQueues,
    /// boolean runnable, Expander moreLess, InterfaceType interfaceType)`.
    pub fn construct(
        &self,
        display_queues: bool,
        more_less: Option<Rc<dyn Expander>>,
        interface_type: InterfaceType,
    ) {
        let this = self.this();
        // Field initializers.
        let _ = self.header1_computer.set(HeaderCell::new_string(
            this.get_header1_computer_text().as_deref(),
        ));
        let viewable: Rc<dyn Viewable> = this.clone();
        let parallel_table_size = etomo_director::INSTANCE
            .with_user_configuration(|c| c.get_parallel_table_size().get_int());
        let _ = self.viewport.set(Viewport::new(
            Rc::downgrade(&viewable),
            parallel_table_size,
            Some("Processor"),
        ));
        // this.manager, this.parent, this.axisID, this.runnable: see `new`.
        // init
        self.viewport().init_paging();
        let load_monitor: Option<Arc<dyn LoadMonitor>> =
            if !etomo_director::ARGUMENTS.lock().unwrap().is_no_load() {
                let load_display: Rc<dyn LoadDisplay> = this.clone();
                let load_display = Arc::new(EdtRef::new(load_display));
                if display_queues {
                    // TODO(unit): needs etomo/process/QueuechunkLoadMonitor.java -
                    // `new QueuechunkLoadMonitor(this, axisID, manager)`.
                    Some(QueuechunkLoadMonitor::new(
                        load_display,
                        self.axis_id,
                        self.manager,
                    ))
                } else {
                    // TODO(unit): needs etomo/process/LoadAverageMonitor.java -
                    // `new LoadAverageMonitor(this, axisID, manager)`.
                    Some(LoadAverageMonitor::new(
                        load_display,
                        self.axis_id,
                        self.manager,
                    ))
                }
            } else {
                None
            };
        let _ = self.load_monitor.set(load_monitor);
        let display: Rc<dyn ParallelProgressDisplay> = this.clone();
        let property_user_dir = self.manager.get_property_user_dir();
        // TODO(unit): needs etomo/logic/ProcessorTableState.java.
        let table_state = ProcessorTableState::new(
            self.manager,
            self.axis_id,
            property_user_dir.as_deref(),
            Some(interface_type),
            more_less,
            Rc::downgrade(&display),
            this.get_processor_type(),
            this.is_queue_table(),
            self.manager.is_dual_selection_queue_table(),
        );
        let _ = self.table_state.set(table_state);
        let table_state = self.table_state();
        if table_state.is_use(ProcessorTableField::TwoQueuesH2) {
            let _ = self
                .header2_primary_queue
                .set(HeaderCell::new_string(Some(FIRST_QUEUE_LABEL)));
            let _ = self
                .header2_secondary_queue
                .set(HeaderCell::new_string(Some(SECONDARY_QUEUE_LABEL)));
        }
        let _ = self.header1_number_cpus.set(HeaderCell::new_string(
            this.getheader1_number_cpus_title().as_deref(),
        ));
        if table_state.is_use(ProcessorTableField::NumCpusMaxH2) {
            let _ = self
                .header2_number_cpus_max
                .set(HeaderCell::new_string(Some(NUMBER_CPUS_MAX_LABEL)));
        }
        if table_state.is_use(ProcessorTableField::NumGpus) {
            let _ = self
                .header1_number_gpus
                .set(HeaderCell::new_string(Some(gpu_table::NUMBER_CPUS_LABEL)));
            let _ = self.header2_number_gpus.set(HeaderCell::new_void());
        }
        if table_state.is_use(ProcessorTableField::LoadAverageH1) {
            let _ = self
                .header1_load
                .set(HeaderCell::new_string(Some("Load Average")));
            let _ = self
                .header2_load1
                .set(HeaderCell::new_string(Some("1 Min.")));
            let _ = self
                .header2_load5
                .set(HeaderCell::new_string(Some("5 Min.")));
        }
        if table_state.is_use(ProcessorTableField::CpuUsageH1) {
            let _ = self
                .header1_cpu_usage
                .set(HeaderCell::new_string(Some("CPU Usage")));
            let _ = self.header2_cpu_usage.set(HeaderCell::new_void());
        }
        if table_state.is_use(ProcessorTableField::LoadArray0H1) {
            // The Rust accessor returns the array itself; Java's getLoadUnitsArray()
            // never returns null either (an empty array when no units are set).
            let load_units_array: Option<Vec<String>> =
                Some(cpu_adoc::INSTANCE.get_load_units_array());
            if !table_state.is_use(ProcessorTableField::LoadArrayXH1) || load_units_array.is_none()
            {
                let _ = self
                    .header1_load_array
                    .set(vec![HeaderCell::new_string(Some("Load"))]);
                let _ = self.header2_load_array.set(vec![HeaderCell::new_void()]);
            } else {
                let load_units_array = load_units_array.unwrap();
                let mut header1_load_array = Vec::with_capacity(load_units_array.len());
                let mut header2_load_array = Vec::with_capacity(load_units_array.len());
                for load_units in &load_units_array {
                    header1_load_array.push(HeaderCell::new_string(Some(load_units)));
                    header2_load_array.push(HeaderCell::new_void());
                }
                let _ = self.header1_load_array.set(header1_load_array);
                let _ = self.header2_load_array.set(header2_load_array);
            }
        }
        if table_state.is_use(ProcessorTableField::UsersH1) {
            let _ = self
                .header1_users
                .set(HeaderCell::new_string(Some("Users")));
            let _ = self.header2_users.set(HeaderCell::new_void());
        }
        if table_state.is_use(ProcessorTableField::TypeH1) {
            let _ = self
                .header1_cpu_type
                .set(HeaderCell::new_string(Some(CPU_TYPE_LABEL)));
            let _ = self.header2_cpu_type.set(HeaderCell::new_void());
        }
        if table_state.is_use(ProcessorTableField::SpeedH1) {
            let _ = self
                .header1_speed
                .set(HeaderCell::new_string(Some("Speed")));
            let _ = self.header2_speed.set(HeaderCell::new_string(
                cpu_adoc::INSTANCE.get_speed_units().as_deref(),
            ));
        }
        if table_state.is_use(ProcessorTableField::MemoryH1) {
            let _ = self.header1_ram.set(HeaderCell::new_string(Some("RAM")));
            let _ = self.header2_ram.set(HeaderCell::new_string(
                cpu_adoc::INSTANCE.get_memory_units().as_deref(),
            ));
        }
        if table_state.is_use(ProcessorTableField::OsH1) {
            let _ = self.header1_os.set(HeaderCell::new_string(Some("OS")));
            let _ = self.header2_os.set(HeaderCell::new_void());
        }
        if table_state.is_use(ProcessorTableField::RestartsH1) {
            let _ = self
                .header1_restarts
                .set(HeaderCell::new_string(Some(RESTARTS_LABEL)));
            let _ = self.header2_restarts.set(HeaderCell::new_void());
            let label = if interface_type != InterfaceType::BatchRunTomo {
                "Chunks"
            } else {
                "Runs"
            };
            let _ = self
                .header1_finished
                .set(HeaderCell::new_string(Some(label)));
            let _ = self
                .header2_finished
                .set(HeaderCell::new_string(Some("Done")));
            let _ = self
                .header1_failure
                .set(HeaderCell::new_string(Some("Failure")));
            let _ = self
                .header2_failure
                .set(HeaderCell::new_string(Some("Reason")));
        }
        if table_state.is_use(ProcessorTableField::GpuTypeH1) {
            let _ = self
                .header1_gpu_type
                .set(HeaderCell::new_string(Some("Type")));
            let _ = self.header2_gpu_type.set(HeaderCell::new_void());
        }
        if table_state.is_use(ProcessorTableField::GpuSpeedH1) {
            let _ = self
                .header1_gpu_speed
                .set(HeaderCell::new_string(Some("Speed")));
            let _ = self.header2_gpu_speed.set(HeaderCell::new_string(
                cpu_adoc::INSTANCE.get_gpu_speed_units().as_deref(),
            ));
        }
        if table_state.is_use(ProcessorTableField::GpuMemoryH1) {
            let _ = self
                .header1_gpu_ram
                .set(HeaderCell::new_string(Some("RAM")));
            let _ = self.header2_gpu_ram.set(HeaderCell::new_string(
                cpu_adoc::INSTANCE.get_gpu_memory_units().as_deref(),
            ));
        }
        if table_state.is_use(ProcessorTableField::GpuNcoresH1) {
            let _ = self
                .header1_gpu_ncores
                .set(HeaderCell::new_string(Some("Cores")));
            let _ = self.header2_gpu_ncores.set(HeaderCell::new_void());
        }
    }

    /// Java `getFocusableParents()` (`Viewable`).
    pub fn get_focusable_parents(&self) -> Vec<Rc<JComponent>> {
        self.focusable_parents.clone()
    }

    /// Java `getheader1NumberCPUsTitle()` (the class's body).
    pub fn getheader1_number_cpus_title_super(&self) -> Option<String> {
        Some("# Cores".to_string())
    }

    /// Java `setHeader1NumberCPUsTitle()`.
    pub fn set_header1_number_cpus_title_void(&self) {
        header(&self.header1_number_cpus)
            .set_text_string(self.this().getheader1_number_cpus_title().as_deref());
    }

    /// Java `setHeader1NumberCPUsTitle(String)`.
    pub fn set_header1_number_cpus_title_string(&self, title: Option<&str>) {
        header(&self.header1_number_cpus).set_text_string(title);
    }

    /// Java `createTable()`.
    pub fn create_table(&self) {
        self.expanded.set(true);
        self.init_table();
        // Swing layout: rootPanel.setLayout(new BoxLayout(rootPanel, X_AXIS));
        // rootPanel.setBorder(LineBorder.createBlackLineBorder()).
        self.build();
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.row_list.size_void() > 0
    }

    /// Java `setExpanded(boolean)`.
    pub fn set_expanded(&self, expanded: bool) {
        if self.expanded.get() == expanded {
            return;
        }
        self.expanded.set(expanded);
        self.row_list.set_contracted_index(expanded);
        self.build();
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.root_panel.set_visible(visible);
    }

    /// Java `build()`.
    pub fn build(&self) {
        self.root_panel.remove_all();
        self.build_table();
        let table_panel = self.table_panel.borrow().clone();
        if let Some(table_panel) = table_panel {
            self.root_panel.add(&table_panel);
        }
        // Upstream NPE fixed in translation (ProcessorTable.java:388): Swing's
        // Container.add throws NullPointerException for the null paging panel
        // a viewport without one returns; nothing is added then.
        if let Some(paging_panel) = self.viewport().get_paging_panel() {
            self.root_panel.add(&paging_panel);
        }
        // configure
        let manager = self.manager;
        let axis_id = self.axis_id;
        ui_harness::with(|harness| harness.repaint_window(Some(manager), Some(axis_id)));
    }

    /// Java private `initTable()`.
    fn init_table(&self) {
        let table_state: Rc<dyn crate::imod::etomo::logic::table_state::TableState> =
            self.table_state().clone();
        let table_state = Some(table_state);
        // table state
        header(&self.header1_computer)
            .set_table_state(field(ProcessorTableField::ComputerH1), table_state.clone());
        header(&self.header1_number_cpus)
            .set_table_state(field(ProcessorTableField::NumCpusH1), table_state.clone());
        if let Some(header2_number_cpus_max) = self.header2_number_cpus_max.get() {
            header2_number_cpus_max.set_table_state(
                field(ProcessorTableField::NumCpusMaxH2),
                table_state.clone(),
            );
        }
        if let Some(header1_load) = self.header1_load.get() {
            header1_load.set_table_state(
                field(ProcessorTableField::LoadAverageH1),
                table_state.clone(),
            );
            header(&self.header2_load1).set_table_state(
                field(ProcessorTableField::LoadAverageH1),
                table_state.clone(),
            );
            header(&self.header2_load5).set_table_state(
                field(ProcessorTableField::LoadAverageH1),
                table_state.clone(),
            );
        }
        if let Some(header1_cpu_usage) = self.header1_cpu_usage.get() {
            header1_cpu_usage
                .set_table_state(field(ProcessorTableField::CpuUsageH1), table_state.clone());
            header(&self.header2_cpu_usage)
                .set_table_state(field(ProcessorTableField::CpuUsageH1), table_state.clone());
        }
        if let Some(header1_load_array) = self.header1_load_array.get() {
            let header2_load_array = self.header2_load_array.get().unwrap();
            for i in 0..header1_load_array.len() {
                if i > 0 {
                    header1_load_array[i].set_table_state(
                        field(ProcessorTableField::LoadArrayXH1),
                        table_state.clone(),
                    );
                    header2_load_array[i].set_table_state(
                        field(ProcessorTableField::LoadArrayXH1),
                        table_state.clone(),
                    );
                } else {
                    header1_load_array[i].set_table_state(
                        field(ProcessorTableField::LoadArray0H1),
                        table_state.clone(),
                    );
                    header2_load_array[i].set_table_state(
                        field(ProcessorTableField::LoadArray0H1),
                        table_state.clone(),
                    );
                }
            }
        }
        if let Some(header1_users) = self.header1_users.get() {
            header1_users.set_table_state(field(ProcessorTableField::UsersH1), table_state.clone());
            header(&self.header2_users)
                .set_table_state(field(ProcessorTableField::UsersH1), table_state.clone());
        }
        if let Some(header1_cpu_type) = self.header1_cpu_type.get() {
            header1_cpu_type
                .set_table_state(field(ProcessorTableField::TypeH1), table_state.clone());
            header(&self.header2_cpu_type)
                .set_table_state(field(ProcessorTableField::TypeH1), table_state.clone());
        }
        if let Some(header1_speed) = self.header1_speed.get() {
            header1_speed.set_table_state(field(ProcessorTableField::SpeedH1), table_state.clone());
            header(&self.header2_speed)
                .set_table_state(field(ProcessorTableField::SpeedH1), table_state.clone());
        }
        if let Some(header1_ram) = self.header1_ram.get() {
            header1_ram.set_table_state(field(ProcessorTableField::MemoryH1), table_state.clone());
            header(&self.header2_ram)
                .set_table_state(field(ProcessorTableField::MemoryH1), table_state.clone());
        }
        if let Some(header1_os) = self.header1_os.get() {
            header1_os.set_table_state(field(ProcessorTableField::OsH1), table_state.clone());
            header(&self.header2_os)
                .set_table_state(field(ProcessorTableField::OsH1), table_state.clone());
        }
        if let Some(header1_gpu_type) = self.header1_gpu_type.get() {
            header1_gpu_type
                .set_table_state(field(ProcessorTableField::GpuTypeH1), table_state.clone());
            header(&self.header2_gpu_type)
                .set_table_state(field(ProcessorTableField::GpuTypeH1), table_state.clone());
        }
        if let Some(header1_gpu_speed) = self.header1_gpu_speed.get() {
            header1_gpu_speed
                .set_table_state(field(ProcessorTableField::GpuSpeedH1), table_state.clone());
            header(&self.header2_gpu_speed)
                .set_table_state(field(ProcessorTableField::GpuSpeedH1), table_state.clone());
        }
        if let Some(header1_gpu_ram) = self.header1_gpu_ram.get() {
            header1_gpu_ram
                .set_table_state(field(ProcessorTableField::GpuMemoryH1), table_state.clone());
            header(&self.header2_gpu_ram)
                .set_table_state(field(ProcessorTableField::GpuMemoryH1), table_state.clone());
        }
        if let Some(header1_gpu_ncores) = self.header1_gpu_ncores.get() {
            header1_gpu_ncores
                .set_table_state(field(ProcessorTableField::GpuNcoresH1), table_state.clone());
            header(&self.header2_gpu_ncores)
                .set_table_state(field(ProcessorTableField::GpuNcoresH1), table_state.clone());
        }
        if let Some(header1_restarts) = self.header1_restarts.get() {
            header1_restarts
                .set_table_state(field(ProcessorTableField::RestartsH1), table_state.clone());
        }
        // loop through the nodes
        // loop on nodes
        let this = self.this();
        let size = this.get_size();
        let user_name = std::env::var("USER").ok();
        for i in 0..size {
            // get the node
            let node = this.get_node(i);
            // exclude any node with the "exclude-interface" attribute set to the
            // current interface
            if let Some(node) = node
                && !node.is_excluded_interface(self.manager.get_interface_type())
                // System.getProperty("user.name")
                && !node.is_excluded_user(user_name.as_deref())
                && !this.is_exclude_node(&node)
            {
                // create the row
                let row = this.create_processor_table_row(&this, &node, size, self.table_state());
                this.init_row(&row);
                // add the row to the rows HashedArray
                self.row_list.add(row);
            }
        }
        // try {
        {
            let storable = this.clone();
            let parameter_store = etomo_director::INSTANCE.get_parameter_store();
            // Upstream NPE fixed in translation (ProcessorTable.java:485): Java
            // dereferences a null parameter store; nothing is loaded then.
            if let Some(parameter_store) = parameter_store.as_ref() {
                parameter_store.load(&storable);
            }
        }
        self.set_tool_tip_text();
        if self.row_list.size_void() == 1 {
            self.row_list.set_selected(0, true);
            // rowList.enableSelectionField(0, false);
        }
    }

    /// Java `msgViewportPaged()` (`Viewable`).
    pub fn msg_viewport_paged(&self) {
        self.build();
        self.pack();
    }

    /// `UIHarness.INSTANCE.pack(axisID, manager)`, as the source writes it at each
    /// site.
    fn pack(&self) {
        let manager = self.manager;
        let axis_id = self.axis_id;
        ui_harness::with(|harness| harness.pack_axis_id_base_manager(Some(axis_id), Some(manager)));
    }

    /// Java `setSecondary(boolean)`.
    pub fn set_secondary(&self, input: bool) {
        if input != self.secondary.get() {
            self.secondary.set(input);
            self.build();
            self.row_list.set_selected_error();
            self.pack();
        }
    }

    /// Java private `buildTable()`.
    fn build_table(&self) {
        let table_panel = JComponent::new_panel();
        *self.table_panel.borrow_mut() = Some(table_panel.clone());
        // Swing layout: layout = new GridBagLayout(); constraints = new
        // GridBagConstraints(); tablePanel.setLayout(layout); constraints.fill = BOTH;
        // anchor = CENTER; weightx = 0.0; weighty = 0.0; gridheight = 1;
        // gridwidth = 1.
        // Header 1
        // Set display columns
        {
            let mut temp = self.temp_displayed_fields.borrow_mut();
            temp.clear();
            temp.push(header(&self.header1_computer).clone());
            temp.push(header(&self.header1_number_cpus).clone());
            if let Some(header1_number_gpus) = self.header1_number_gpus.get() {
                temp.push(header1_number_gpus.clone());
            }
            if let Some(header1_load) = self.header1_load.get()
                && header1_load.is_display()
            {
                temp.push(header1_load.clone());
            }
            if let Some(header1_cpu_usage) = self.header1_cpu_usage.get()
                && header1_cpu_usage.is_display()
            {
                temp.push(header1_cpu_usage.clone());
            }
            if let Some(header1_load_array) = self.header1_load_array.get() {
                for header1_load in header1_load_array {
                    if header1_load.is_display() {
                        temp.push(header1_load.clone());
                    }
                }
            }
            for cell in [
                &self.header1_users,
                &self.header1_cpu_type,
                &self.header1_speed,
                &self.header1_ram,
                &self.header1_os,
                &self.header1_gpu_type,
                &self.header1_gpu_speed,
                &self.header1_gpu_ram,
                &self.header1_gpu_ncores,
            ] {
                if let Some(cell) = cell.get()
                    && cell.is_display()
                {
                    temp.push(cell.clone());
                }
            }
            if let Some(header1_restarts) = self.header1_restarts.get()
                && header1_restarts.is_display()
            {
                temp.push(header1_restarts.clone());
                temp.push(header(&self.header1_finished).clone());
                temp.push(header(&self.header1_failure).clone());
            }
        }
        // Add fields to the table
        let cells = self.temp_displayed_fields.borrow().clone();
        let size = cells.len();
        for (i, cell) in cells.iter().enumerate() {
            if i == size - 1 {
                // Swing layout: constraints.gridwidth = GridBagConstraints.REMAINDER.
            } else {
                // Swing layout: constraints.gridwidth = cell.getGridwidth().
                let _ = cell.cell().get_gridwidth();
            }
            cell.add(&table_panel);
        }
        // Header 2
        // Set display columns
        {
            let mut temp = self.temp_displayed_fields.borrow_mut();
            temp.clear();
            if let Some(header2_primary_queue) = self.header2_primary_queue.get() {
                temp.push(header2_primary_queue.clone());
            }
            if let Some(header2_secondary_queue) = self.header2_secondary_queue.get() {
                temp.push(header2_secondary_queue.clone());
            }
            temp.push(self.header2_computer.clone());
            temp.push(self.header2_number_cpus_used.clone());
            if let Some(header2_number_cpus_max) = self.header2_number_cpus_max.get()
                && header2_number_cpus_max.is_display()
            {
                temp.push(header2_number_cpus_max.clone());
            }
            if let Some(header2_number_gpus) = self.header2_number_gpus.get() {
                temp.push(header2_number_gpus.clone());
            }
            if let Some(header2_load1) = self.header2_load1.get() {
                if header2_load1.is_display() {
                    temp.push(header2_load1.clone());
                }
                let header2_load5 = header(&self.header2_load5);
                if header2_load5.is_display() {
                    temp.push(header2_load5.clone());
                }
            }
            if let Some(header2_cpu_usage) = self.header2_cpu_usage.get()
                && header2_cpu_usage.is_display()
            {
                temp.push(header2_cpu_usage.clone());
            }
            if let Some(header2_load_array) = self.header2_load_array.get() {
                for header2_load in header2_load_array {
                    if header2_load.is_display() {
                        temp.push(header2_load.clone());
                    }
                }
            }
            for cell in [
                &self.header2_users,
                &self.header2_cpu_type,
                &self.header2_speed,
                &self.header2_ram,
                &self.header2_os,
                &self.header2_gpu_type,
                &self.header2_gpu_speed,
                &self.header2_gpu_ram,
                &self.header2_gpu_ncores,
            ] {
                if let Some(cell) = cell.get()
                    && cell.is_display()
                {
                    temp.push(cell.clone());
                }
            }
            if let Some(header1_restarts) = self.header1_restarts.get()
                && header1_restarts.is_display()
            {
                temp.push(header(&self.header2_restarts).clone());
                temp.push(header(&self.header2_finished).clone());
                temp.push(header(&self.header2_failure).clone());
            }
        }
        // Add fields to the table
        // Swing layout: constraints.gridwidth = 1.
        let cells = self.temp_displayed_fields.borrow().clone();
        // Swing layout: the last cell gets constraints.gridwidth = REMAINDER.
        for cell in cells.iter() {
            cell.add(&table_panel);
        }
        self.temp_displayed_fields.borrow_mut().clear();
        // add rows to the table
        self.viewport().msg_viewable_changed();
        self.row_list.display(self.expanded.get(), self.viewport());
    }

    /// Java private `add(HeaderCell, boolean, ColumnName, ColumnName)` (not called
    /// by the source).
    #[allow(dead_code)]
    fn add(
        &self,
        cell: &Rc<HeaderCell>,
        r#use: bool,
        column_name: ColumnName,
        last_column_name: ColumnName,
    ) {
        if r#use {
            if last_column_name == column_name {
                // Swing layout: constraints.gridwidth = GridBagConstraints.REMAINDER.
            }
            let table_panel = self.table_panel.borrow().clone();
            if let Some(table_panel) = table_panel {
                CellVirtual::add(&**cell, &table_panel);
            }
        }
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java `getTablePanel()`.
    pub fn get_table_panel(&self) -> Option<Rc<JComponent>> {
        self.table_panel.borrow().clone()
    }

    /// Java `resetResults()` (`ParallelProgressDisplay`).
    pub fn reset_results(&self) {
        self.row_list.reset_results();
    }

    /// Java `getTotalSuccesses()`.
    pub fn get_total_successes(&self) -> i32 {
        self.row_list.get_total_successes()
    }

    /// Java `msgCPUsSelectedChanged()`.
    pub fn msg_cpus_selected_changed(&self) {
        if !self.secondary.get() {
            self.parent().set_cpus_selected(self.get_cpus_selected());
        } else {
            self.parent()
                .set_secondary_cpus_selected(self.get_cpus_selected());
        }
    }

    /// Java `msgEndingProcess()` (`ParallelProgressDisplay`).
    pub fn msg_ending_process(&self) {
        self.parent().msg_ending_process();
    }

    /// Java `msgKillingProcess()` (`ParallelProgressDisplay`).
    pub fn msg_killing_process(&self) {
        self.parent().msg_killing_process();
    }

    /// Java `msgProcessStarted()` (`ParallelProgressDisplay`).
    pub fn msg_process_started(&self) {
        self.parent().msg_process_started();
    }

    /// Java `msgPausingProcess()` (`ParallelProgressDisplay`).
    pub fn msg_pausing_process(&self) {
        self.parent().msg_pausing_process();
    }

    /// Java `getCPUsSelected()`.
    pub fn get_cpus_selected(&self) -> i32 {
        self.row_list.get_cpus_selected()
    }

    /// Java `restartLoadMonitor()`.
    pub fn restart_load_monitor(&self) {
        if let Some(load_monitor) = self.load_monitor() {
            load_monitor.restart();
        }
    }

    /// Java `isSecondary()` (`ParallelProgressDisplay`).
    pub fn is_secondary(&self) -> bool {
        self.secondary.get()
    }

    /// Java `isRunnable()` (`ParallelProgressDisplay`).
    pub fn is_runnable(&self) -> bool {
        self.runnable.get()
    }

    /// Java `isLimited()` (`ParallelProgressDisplay`).
    pub fn is_limited(&self) -> bool {
        self.limited.get()
    }

    /// Java `setRunnable(boolean)`.
    pub fn set_runnable(&self, input: bool) {
        if input != self.runnable.get() {
            self.runnable.set(input);
            self.build();
            self.row_list.set_selected_error();
            self.pack();
        }
    }

    /// Java `setLimited(boolean)`.
    pub fn set_limited(&self, input: bool) {
        if input != self.limited.get() {
            self.limited.set(input);
            self.build();
            self.row_list.set_selected_error();
            self.pack();
        }
    }

    /// Java `getFirstSelectedIndex()`.
    pub fn get_first_selected_index(&self) -> i32 {
        self.row_list.get_first_selected_index()
    }

    /// Java `getNextSelectedIndex(int)`.
    pub fn get_next_selected_index(&self, last_index: i32) -> i32 {
        self.row_list.get_next_selected_index(last_index)
    }

    /// Java `getParameters(ProcesschunksParam)` (the class's body).
    pub fn get_parameters_processchunks_param_super(&self, param: &ProcesschunksParam) {
        self.row_list
            .get_parameters_processchunks_param(self, param);
    }

    /// Java `getParameters(BatchruntomoParam)` (the class's body).
    pub fn get_parameters_batchruntomo_param_super(&self, param: &mut BatchruntomoParam) {
        self.row_list.get_parameters_batchruntomo_param(param);
    }

    /// Java `setParameters(BatchruntomoParam)`.
    pub fn set_parameters(&self, param: &BatchruntomoParam) {
        let machine_map = self.this().get_machine_map(param);
        ParallelProgressDisplay::set_computer_map(&*self.this(), machine_map.as_ref());
    }

    /// Java `getMachineMap(BatchruntomoParam)` (the class's body).
    pub fn get_machine_map_super(
        &self,
        param: &BatchruntomoParam,
    ) -> Option<HashMap<String, String>> {
        param.get_cpu_machine_map()
    }

    /// Java `getFirstSelectedComputer()`.
    pub fn get_first_selected_computer(&self) -> Option<String> {
        self.row_list
            .get_computer(self.row_list.get_first_selected_index())
    }

    /// Java `getSelectedSecondaryQueueNode()`.
    pub fn get_selected_secondary_queue_node(&self) -> Option<Arc<Node>> {
        let row = self.row_list.get_selected_secondary_queue()?;
        let computer_name = row.get_computer()?;
        Network::get_queue(Some(&computer_name))
    }

    /// Java `size()` (`Viewable`).
    pub fn size(&self) -> i32 {
        self.row_list.size_boolean(self.expanded.get())
    }

    /// Java private `getRow(String)`.
    fn get_row(&self, computer: Option<&str>) -> Option<Rc<ProcessorTableRow>> {
        self.row_list.get_string(computer)
    }

    /// Java `getFirstSelectedRow()`.
    pub fn get_first_selected_row(&self) -> Option<Rc<ProcessorTableRow>> {
        self.row_list
            .get_int(self.row_list.get_first_selected_index())
    }

    /// Java `getFirstSelectedSecondaryQueueRow()`.
    pub fn get_first_selected_secondary_queue_row(&self) -> Option<Rc<ProcessorTableRow>> {
        self.row_list
            .get_int(self.row_list.get_first_selected_secondary_queue_index())
    }

    /// Java `addRestart(String)` (`ParallelProgressDisplay`).
    pub fn add_restart(&self, computer: Option<&str>) {
        let Some(row) = self.get_row(computer) else {
            return;
        };
        row.add_restart();
    }

    /// Java `addSuccess(String)` (`ParallelProgressDisplay`).
    pub fn add_success(&self, computer: Option<&str>) {
        let Some(row) = self.get_row(computer) else {
            return;
        };
        row.add_success();
    }

    /// Java `setComputerMap(Map<String, String>)` (`ParallelProgressDisplay`).
    pub fn set_computer_map(&self, computer_map: Option<&HashMap<String, String>>) {
        self.row_list.set_computer_map(computer_map);
        self.parent().msg_selection_changed();
    }

    /// Java `setSecondaryQueue(String)` (`ParallelProgressDisplay`).
    pub fn set_secondary_queue(&self, secondary_queue: Option<&str>) {
        self.row_list.set_secondary_queue(self, secondary_queue);
        self.parent().msg_selection_changed();
    }

    /// Java `msgDropped(String, String)` (`ParallelProgressDisplay`).
    pub fn msg_dropped(&self, computer: Option<&str>, reason: Option<&str>) {
        let Some(row) = self.get_row(computer) else {
            return;
        };
        row.msg_dropped(reason);
    }

    /// Java `getHelpMessage()`.
    pub fn get_help_message(&self) -> String {
        format!(
            "Click on check boxes in the {} column and use the spinner in the {} {} column where available.",
            header(&self.header1_computer)
                .get_text()
                .as_deref()
                .unwrap_or("null"),
            header(&self.header1_number_cpus)
                .get_text()
                .as_deref()
                .unwrap_or("null"),
            self.header2_number_cpus_used
                .get_text()
                .as_deref()
                .unwrap_or("null")
        )
    }

    /// Java `startLoad()` (`LoadDisplay`).
    pub fn start_load(&self) {
        let load_monitor = self.load_monitor();
        if load_monitor.is_none() || self.secondary.get() {
            // The secondary table should not also run the load
            return;
        }
        let load_monitor = load_monitor.unwrap();
        self.stopped.set(false);
        let mut i = 0;
        while i < self.row_list.size_void() {
            self.manager.start_load(
                Some(self.get_intermittent_command_int(i)),
                Some(load_monitor.clone()),
            );
            i += 1;
        }
    }

    /// Java private `getIntermittentCommand(int)`.
    fn get_intermittent_command_int(&self, index: i32) -> Arc<dyn IntermittentCommand> {
        let row = self
            .row_list
            .get_int(index)
            .expect("ProcessorTable: row index within the row list");
        let computer = row.get_computer();
        self.this()
            .get_intermittent_command_string(computer.as_deref())
    }

    /// Java `endLoad()` (`LoadDisplay`).
    pub fn end_load(&self) {
        if let Some(load_monitor) = self.load_monitor() {
            self.stopped.set(true);
            let mut i = 0;
            while i < self.row_list.size_void() {
                self.manager.end_load(
                    Some(self.get_intermittent_command_int(i)),
                    Some(load_monitor.clone()),
                );
                i += 1;
            }
        }
    }

    /// Java `stopLoad()` (`LoadDisplay`).
    pub fn stop_load(&self) {
        if let Some(load_monitor) = self.load_monitor() {
            self.stopped.set(true);
            let mut i = 0;
            while i < self.row_list.size_void() {
                self.manager.stop_load(
                    Some(self.get_intermittent_command_int(i)),
                    Some(load_monitor.clone()),
                );
                i += 1;
            }
        }
    }

    /// Java public `isStopped()`.
    pub fn is_stopped(&self) -> bool {
        self.stopped.get()
    }

    /// Java `setLoad(String, double, double, int, String)` (`LoadDisplay`).
    ///
    /// Upstream bug fixed (ProcessorTable.java:923-927): the source dereferences
    /// `rowList.get(computer)` without a null check, so a load report for a computer
    /// that has no row (one excluded from this table, or a table rebuilt while the
    /// monitor ran) throws NullPointerException on the event thread.  The report is
    /// ignored instead, as `addRestart`/`addSuccess`/`msgDropped` already do.
    pub fn set_load_string_double_double_int_string(
        &self,
        computer: Option<&str>,
        load1: f64,
        load5: f64,
        users: i32,
        users_tooltip: Option<&str>,
    ) {
        if let Some(row) = self.row_list.get_string(computer) {
            row.set_load_double_double_int_string(load1, load5, users, users_tooltip);
        }
    }

    /// Java `setLoad(String, String[])` (`LoadDisplay`).  Missing row: see
    /// `set_load_string_double_double_int_string`.
    pub fn set_load_string_string_array(&self, computer: Option<&str>, load_array: &[String]) {
        if let Some(row) = self.row_list.get_string(computer) {
            row.set_load_string_array(load_array);
        }
    }

    /// Java `setCPUUsage(String, double, ConstEtomoNumber)` (`LoadDisplay`).  Missing
    /// row: see `set_load_string_double_double_int_string`.
    pub fn set_cpu_usage(
        &self,
        computer: Option<&str>,
        cpu_usage: f64,
        number_of_processors: Option<&ConstEtomoNumber>,
    ) {
        if let Some(row) = self.row_list.get_string(computer) {
            row.set_cpu_usage(cpu_usage, number_of_processors);
        }
    }

    /// Java `msgLoadFailed(String, String, String)` (`LoadDisplay`).  Clears the load
    /// from the display.  Does not ask the monitor to drop the computer because
    /// processchunks handles this very well, and it is possible that the computer may
    /// still be available.  Missing row: see
    /// `set_load_string_double_double_int_string`.
    pub fn msg_load_failed(
        &self,
        computer: Option<&str>,
        reason: Option<&str>,
        tooltip: Option<&str>,
    ) {
        if let Some(row) = self.row_list.get_string(computer) {
            row.clear_load(reason, tooltip);
        }
    }

    /// Java `msgStartingProcessOnSelectedComputers()` (`ParallelProgressDisplay`).
    pub fn msg_starting_process_on_selected_computers(&self) {
        self.clear_failure_reason(true);
    }

    /// Java `msgStartingProcess(String, String, String)` (`LoadDisplay`).  Clear
    /// failure reason, if failure reason equals failureReason1 or 2.  This means that
    /// intermittent processes only clear their own messages.  Missing row: see
    /// `set_load_string_double_double_int_string`.
    pub fn msg_starting_process(
        &self,
        computer: Option<&str>,
        failure_reason1: Option<&str>,
        failure_reason2: Option<&str>,
    ) {
        if let Some(row) = self.row_list.get_string(computer) {
            row.clear_failure_reason_string_string(failure_reason1, failure_reason2);
        }
    }

    /// Java `clearFailureReason(boolean)`.
    pub fn clear_failure_reason(&self, selected_computers: bool) {
        self.row_list.clear_failure_reason(selected_computers);
    }

    /// Java `getGroupKey()`.
    pub fn get_group_key(&self) -> String {
        if self.runnable.get() {
            return RUNNABLE_KEY.to_string();
        }
        RESOURCE_KEY.to_string()
    }

    /// Java `store(Properties)` (`Storable`).
    pub fn store_properties(&self, props: &mut Properties) {
        self.store_properties_string(props, "");
    }

    /// Java `store(Properties, String)` (`Storable`).
    pub fn store_properties_string(&self, props: &mut Properties, prepend: &str) {
        // `prepend == ""` is a reference comparison in Java; every caller passes the
        // literal "" (interned) or a built string, so it is an emptiness test.
        let prepend = if prepend.is_empty() {
            self.this().get_store_prepend()
        } else {
            format!("{}.{}", prepend, self.this().get_store_prepend())
        };
        self.row_list.store(props, &prepend);
    }

    /// Java final `load(Properties)` (`Storable`).  Get the objects attributes from
    /// the properties object.
    pub fn load_properties(&self, props: &Properties) {
        self.load_properties_string(props, "");
    }

    /// Java `load(Properties, String)` (`Storable`).
    pub fn load_properties_string(&self, props: &Properties, prepend: &str) {
        let version = self.parent().get_version();
        let prepend = if prepend.is_empty() {
            self.this().get_load_prepend(&version)
        } else {
            format!("{}.{}", prepend, self.this().get_load_prepend(&version))
        };
        self.row_list.load(props, &prepend);
    }

    /// Java final `isScrolling()`.
    pub fn is_scrolling(&self) -> bool {
        self.scrolling.get()
    }

    /// Java final private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let mut text;
        text = "Select computers to use for parallel processing.";
        header(&self.header1_computer).set_tool_tip_text(Some(text));
        self.header2_computer.set_tool_tip_text(Some(text));
        text = "Select the number of CPUs to use for each computer.";
        header(&self.header1_number_cpus).set_tool_tip_text(Some(text));
        self.header2_number_cpus_used.set_tool_tip_text(Some(text));
        if let Some(header1_number_gpus) = self.header1_number_gpus.get() {
            text = "The number of GPUs per node";
            header1_number_gpus.set_tool_tip_text(Some(text));
            header(&self.header2_number_gpus).set_tool_tip_text(Some(text));
        }
        if let Some(header2_number_cpus_max) = self.header2_number_cpus_max.get() {
            header2_number_cpus_max.set_tool_tip_text(Some(
                "The maximum number of CPUs available on each computer.",
            ));
        }
        if let Some(header1_load) = self.header1_load.get() {
            header1_load.set_tool_tip_text(Some("Represents how busy each computer is."));
            header(&self.header2_load1)
                .set_tool_tip_text(Some("The load averaged over one minute."));
            header(&self.header2_load5)
                .set_tool_tip_text(Some("The load averaged over five minutes."));
        }
        if let Some(header1_cpu_usage) = self.header1_cpu_usage.get() {
            header1_cpu_usage.set_tool_tip_text(Some(
                "The CPU usage (0 to number of CPUs) averaged over one second.",
            ));
        }
        text = "The number of users logged into the computer.";
        if let Some(header1_users) = self.header1_users.get() {
            header1_users.set_tool_tip_text(Some(text));
        }
        if let Some(header2_users) = self.header2_users.get() {
            header2_users.set_tool_tip_text(Some(text));
        }
        text = "The number of times processes failed on each computer.";
        if let Some(header1_restarts) = self.header1_restarts.get() {
            header1_restarts.set_tool_tip_text(Some(text));
            header(&self.header2_restarts).set_tool_tip_text(Some(text));
            text = "The number of processes each computer completed for a distributed process.";
            header(&self.header1_finished).set_tool_tip_text(Some(text));
            header(&self.header2_finished).set_tool_tip_text(Some(text));
            text = "Reason for a failure by the load average or a process";
            header(&self.header1_failure).set_tool_tip_text(Some(text));
            header(&self.header2_failure).set_tool_tip_text(Some(text));
        }
        if let Some(header1_cpu_type) = self.header1_cpu_type.get() {
            text = "The CPU type of each computer.";
            header1_cpu_type.set_tool_tip_text(Some(text));
            header(&self.header2_cpu_type).set_tool_tip_text(Some(text));
        }
        if let Some(header1_speed) = self.header1_speed.get() {
            text = "The speed of each computer.";
            header1_speed.set_tool_tip_text(Some(text));
            header(&self.header2_speed).set_tool_tip_text(Some(text));
        }
        if let Some(header1_ram) = self.header1_ram.get() {
            text = "The amount of RAM in each computer.";
            header1_ram.set_tool_tip_text(Some(text));
            header(&self.header2_ram).set_tool_tip_text(Some(text));
        }
        if let Some(header1_os) = self.header1_os.get() {
            text = "The operating system of each computer.";
            header1_os.set_tool_tip_text(Some(text));
            header(&self.header2_os).set_tool_tip_text(Some(text));
        }
        if let Some(header1_gpu_type) = self.header1_gpu_type.get() {
            text = "The GPU type of each computer.";
            header1_gpu_type.set_tool_tip_text(Some(text));
            header(&self.header2_gpu_type).set_tool_tip_text(Some(text));
        }
        if let Some(header1_gpu_speed) = self.header1_gpu_speed.get() {
            text = "The GPU speed of each computer.";
            header1_gpu_speed.set_tool_tip_text(Some(text));
            header(&self.header2_gpu_speed).set_tool_tip_text(Some(text));
        }
        if let Some(header1_gpu_ram) = self.header1_gpu_ram.get() {
            text = "The amount of GPU RAM in each computer.";
            header1_gpu_ram.set_tool_tip_text(Some(text));
            header(&self.header2_gpu_ram).set_tool_tip_text(Some(text));
        }
        if let Some(header1_gpu_ncores) = self.header1_gpu_ncores.get() {
            text = "The number of GPU cores in each computer.";
            header1_gpu_ncores.set_tool_tip_text(Some(text));
            header(&self.header2_gpu_ncores).set_tool_tip_text(Some(text));
        }
    }

    /// Java `queueTableEventAction(QueueTableEvent)` (the class's body).
    pub fn queue_table_event_action_super(&self, event: &QueueTableEvent) {
        self.row_list.queue_table_event_action(event);
    }

    /// Java `secondaryQueueSelectedAction()`.
    pub fn secondary_queue_selected_action(&self) {
        self.row_list.secondary_queue_selected_action();
    }

    /// Java `enableQueueRow(String, boolean)`.
    pub fn enable_queue_row(&self, name: Option<&str>, enable: bool) {
        self.row_list.enable_selection_field(name, enable);
    }
}

/// Java `Storable` on a table reference: `ParameterStore.load(this)` and
/// `manager.savePreferences(axisID, table)` take the table as a `Storable`.  The
/// bodies only need shared access.
impl Storable for Rc<dyn ProcessorTableVirtual> {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.processor_table().store_properties(properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        self.processor_table()
            .store_properties_string(properties, prepend);
    }

    fn load(&self, properties: &BTreeMap<String, String>) {
        self.processor_table().load_properties(properties);
    }

    fn load_with_prepend(&self, properties: &BTreeMap<String, String>, prepend: &str) {
        self.processor_table()
            .load_properties_string(properties, prepend);
    }
}

/// Java static final nested class `ColumnName`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnName {
    /// Java `NUMBER_USED`.
    NumberUsed,
    /// Java `NUMBER`.
    Number,
    /// Java `LOAD`.
    Load,
    /// Java `TYPE`.
    Type,
    /// Java `SPEED`.
    Speed,
    /// Java `MEMORY`.
    Memory,
    /// Java `OS`.
    Os,
    /// Java `RUN`.
    Run,
    /// Java `USERS`.
    Users,
}

impl std::fmt::Display for ColumnName {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ColumnName::NumberUsed => "NUMBER_USED",
            ColumnName::Number => "NUMBER",
            ColumnName::Load => "LOAD",
            ColumnName::Type => "TYPE",
            ColumnName::Speed => "SPEED",
            ColumnName::Memory => "MEMORY",
            ColumnName::Os => "OS",
            ColumnName::Run => "RUN",
            ColumnName::Users => "USERS",
        })
    }
}

/// Java private final inner class `RowList`.  The rows are only borrowed long
/// enough to clone the `Rc` of one row: a row's methods call back into the table,
/// which reads this list again.
struct RowList {
    /// Java `list`.
    list: RefCell<Vec<Rc<ProcessorTableRow>>>,
    /// Java `contractedIndex`: contracted index for use when the table is not
    /// expanded.
    contracted_index: RefCell<Vec<Rc<ProcessorTableRow>>>,
}

impl RowList {
    /// Java private `RowList()`.
    fn new() -> RowList {
        RowList {
            list: RefCell::new(Vec::new()),
            contracted_index: RefCell::new(Vec::new()),
        }
    }

    /// Java `queueTableEventAction(QueueTableEvent)`.
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i).unwrap().queue_table_event_action(event);
            i += 1;
        }
    }

    /// Java `secondaryQueueSelectedAction()`.
    fn secondary_queue_selected_action(&self) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i).unwrap().update_display();
            i += 1;
        }
    }

    /// Java private `setComputerMap(Map<String, String>)`.  Changes the selected
    /// computers and CPUs to match computerMap.
    fn set_computer_map(&self, computer_map: Option<&HashMap<String, String>>) {
        let Some(computer_map) = computer_map else {
            return;
        };
        if computer_map.is_empty() {
            return;
        }
        let mut i = 0;
        while i < self.size_void() {
            // First unselect a computer. Then select the computer if it is in
            // computerMap.
            let row = self.get_int(i).unwrap();
            row.set_selected(false);
            let key = row.get_computer();
            // `Map.containsKey(null)` is false for the maps in use (HashMap built from
            // non-null keys).
            if let Some(key) = key
                && computer_map.contains_key(&key)
            {
                row.set_selected(true);
                row.set_cpus_selected(computer_map.get(&key).map(String::as_str));
            }
            i += 1;
        }
    }

    /// Java private `setSecondaryQueue(String)`.
    fn set_secondary_queue(&self, outer: &ProcessorTable, secondary_queue: Option<&str>) {
        if secondary_queue.is_none() || !outer.this().is_queue_table() {
            return;
        }
        let mut i = 0;
        while i < self.size_void() {
            // Select the secondary queue radio button in the row which matches the
            // secondaryQueue parameter.
            let row = self.get_int(i).unwrap();
            if row.has_secondary_queue() && row.equals(secondary_queue) {
                row.set_secondary_queue_selected();
                return;
            }
            i += 1;
        }
    }

    /// Java private `setSelectedError()`.
    fn set_selected_error(&self) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i).unwrap().set_selected_error();
            i += 1;
        }
    }

    /// Java private `add(ProcessorTableRow)`.
    fn add(&self, row: Rc<ProcessorTableRow>) {
        self.list.borrow_mut().push(row);
    }

    /// Java private `display(boolean, Viewport)`.
    fn display(&self, expanded: bool, viewport: &Rc<Viewport>) {
        let mut i = 0;
        while i < self.size_boolean(expanded) {
            let row = if expanded {
                self.get_int(i).unwrap()
            } else {
                self.contracted_index.borrow()[i as usize].clone()
            };
            row.delete_row();
            row.display(i, viewport);
            i += 1;
        }
    }

    /// Java private `size(boolean)`.
    fn size_boolean(&self, expanded: bool) -> i32 {
        if expanded {
            return self.size_void();
        }
        self.contracted_index.borrow().len() as i32
    }

    /// Java private `size()`.
    fn size_void(&self) -> i32 {
        self.list.borrow().len() as i32
    }

    /// Java private `get(int)`.
    fn get_int(&self, index: i32) -> Option<Rc<ProcessorTableRow>> {
        if index == -1 {
            return None;
        }
        Some(self.list.borrow()[index as usize].clone())
    }

    /// Java private `get(String)`.
    fn get_string(&self, computer: Option<&str>) -> Option<Rc<ProcessorTableRow>> {
        let mut i = 0;
        while i < self.size_void() {
            let row = self.get_int(i).unwrap();
            if row.equals(computer) {
                return Some(row);
            }
            i += 1;
        }
        None
    }

    /// Java private `setContractedIndex(boolean)`.
    fn set_contracted_index(&self, expanded: bool) {
        self.contracted_index.borrow_mut().clear();
        if !expanded {
            let mut i = 0;
            while i < self.size_void() {
                let row = self.get_int(i).unwrap();
                if row.is_selected() || row.is_secondary_queue_selected() {
                    self.contracted_index.borrow_mut().push(row);
                }
                i += 1;
            }
        }
    }

    /// Java private `getParameters(ProcesschunksParam)`.
    fn get_parameters_processchunks_param(
        &self,
        outer: &ProcessorTable,
        param: &ProcesschunksParam,
    ) {
        let this = outer.this();
        if outer.manager.is_add_gpu_machine_to_process_chunks() && this.is_gpu_table() {
            param.reset_gpu_machine_list();
        }
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i)
                .unwrap()
                .get_parameters_processchunks_param_boolean_boolean(
                    param,
                    outer.manager.is_add_gpu_machine_to_process_chunks(),
                    outer.is_secondary(),
                );
            i += 1;
        }
    }

    /// Java private `getParameters(BatchruntomoParam)`.
    fn get_parameters_batchruntomo_param(&self, param: &mut BatchruntomoParam) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i)
                .unwrap()
                .get_parameters_batchruntomo_param(param);
            i += 1;
        }
    }

    /// Java private `setSelected(int, boolean)`.
    fn set_selected(&self, index: i32, selected: bool) {
        self.get_int(index).unwrap().set_selected(selected);
    }

    // Java private `enableSelectionField(int, boolean)` is commented out in the
    // source.

    /// Java private `enableSelectionField(String, boolean)`.
    ///
    /// Upstream bug fixed (ProcessorTable.java:1297-1299): `get(name)` is
    /// dereferenced without a null check.  `QueueTable.enableGpuQueueRows` passes
    /// every queue in cpu.adoc, including queues `isExcludeNode` kept out of the
    /// table, so a missing row threw NullPointerException.  A missing row is skipped.
    fn enable_selection_field(&self, name: Option<&str>, enabled: bool) {
        if let Some(row) = self.get_string(name) {
            row.enable_selection_field(enabled);
        }
    }

    /// Java private `resetResults()`.
    fn reset_results(&self) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i).unwrap().reset_results();
            i += 1;
        }
    }

    /// Java private `getTotalSuccesses()`.
    fn get_total_successes(&self) -> i32 {
        let mut successes: i32 = 0;
        let mut i = 0;
        while i < self.size_void() {
            successes = successes.wrapping_add(self.get_int(i).unwrap().get_successes());
            i += 1;
        }
        successes
    }

    /// Java private `getCPUsSelected()`.
    fn get_cpus_selected(&self) -> i32 {
        let mut cpus_selected: i32 = 0;
        let mut i = 0;
        while i < self.size_void() {
            cpus_selected =
                cpus_selected.wrapping_add(self.get_int(i).unwrap().get_cpus_selected());
            i += 1;
        }
        cpus_selected
    }

    /// Java private `getFirstSelectedIndex()`.
    fn get_first_selected_index(&self) -> i32 {
        let mut i = 0;
        while i < self.size_void() {
            if self.get_int(i).unwrap().is_selected() {
                return i;
            }
            i += 1;
        }
        -1
    }

    /// Java private `getFirstSelectedSecondaryQueueIndex()`.
    fn get_first_selected_secondary_queue_index(&self) -> i32 {
        let mut i = 0;
        while i < self.size_void() {
            if self.get_int(i).unwrap().is_secondary_queue_selected() {
                return i;
            }
            i += 1;
        }
        -1
    }

    /// Java private `getSelectedSecondaryQueue()`.
    fn get_selected_secondary_queue(&self) -> Option<Rc<ProcessorTableRow>> {
        let mut i = 0;
        while i < self.size_void() {
            let row = self.get_int(i);
            if let Some(row) = row
                && row.is_secondary_queue_selected()
            {
                return Some(row);
            }
            i += 1;
        }
        None
    }

    /// Java private `getNextSelectedIndex(int)`.
    fn get_next_selected_index(&self, last_index: i32) -> i32 {
        let mut i = last_index + 1;
        while i < self.size_void() {
            if self.get_int(i).unwrap().is_selected() {
                return i;
            }
            i += 1;
        }
        -1
    }

    /// Java private `getComputer(int)`.
    fn get_computer(&self, index: i32) -> Option<String> {
        let row = self.get_int(index)?;
        row.get_computer()
    }

    /// Java private `clearFailureReason(boolean)`.
    fn clear_failure_reason(&self, selected_computers: bool) {
        let mut i = 0;
        while i < self.size_void() {
            let row = self.get_int(i).unwrap();
            if !selected_computers || row.is_selected() {
                row.clear_failure_reason_void();
            }
            i += 1;
        }
    }

    /// Java private `store(Properties, String)`.
    fn store(&self, props: &mut Properties, prepend: &str) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i)
                .unwrap()
                .store_properties_string(props, prepend);
            i += 1;
        }
    }

    /// Java private `load(Properties, String)`.
    fn load(&self, props: &Properties, prepend: &str) {
        let mut i = 0;
        while i < self.size_void() {
            self.get_int(i)
                .unwrap()
                .load_properties_string(props, prepend);
            i += 1;
        }
    }
}
