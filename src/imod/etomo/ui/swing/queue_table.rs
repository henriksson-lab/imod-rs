//! `IMOD/Etomo/src/etomo/ui/swing/QueueTable.java`.
//!
//! Java `final class QueueTable extends ProcessorTable`: the processor table
//! that lists cluster queues.  The superclass is embedded as `base` (with
//! `Deref`), the abstract and overridden members are [`ProcessorTableVirtual`],
//! and the inherited interfaces are bound with `processor_table_interfaces!`.
//! Construction follows `ProcessorTable`'s split (see `processor_table.rs`):
//! allocate, connect `this`, then run [`ProcessorTable::construct`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::parallel_panel::ParallelPanel;
use super::processor_table::{ProcessorTable, ProcessorTableVirtual};
use super::processor_table_row::ProcessorTableRow;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::queuechunk_param::QueuechunkParam;
use crate::imod::etomo::jdk::{ActionEvent, ButtonGroup};
use crate::imod::etomo::logic::processor_table_state::ProcessorTableState;
use crate::imod::etomo::logic::processor_type::ProcessorType;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::node::Node;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::r#type::queue_type::QueueType;
use crate::imod::etomo::ui::expander::Expander;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

/// Java private static final `PREPEND`.
const PREPEND: &str = ".Queue";
/// Java package-private static final `NUMBER_JOBS_LABEL1`.
pub const NUMBER_JOBS_LABEL1: &str = "# Jobs";

/// Java `final class QueueTable extends ProcessorTable`.
pub struct QueueTable {
    /// The `ProcessorTable` superclass.
    base: ProcessorTable,
    /// Java private final `dualSelectionQueueTable`.
    dual_selection_queue_table: bool,
    /// Java private `buttonGroup = null`.
    button_group: RefCell<Option<Rc<ButtonGroup>>>,
    /// Java private `secondaryButtonGroup = null`.
    secondary_button_group: RefCell<Option<Rc<ButtonGroup>>>,
    /// Java private `queueTableListenerArray = null`.
    queue_table_listener_array: RefCell<Option<Vec<Rc<dyn QueueTableListener>>>>,
}

impl Deref for QueueTable {
    type Target = ProcessorTable;
    fn deref(&self) -> &ProcessorTable {
        &self.base
    }
}

impl QueueTable {
    /// Java `QueueTable(BaseManager, ParallelPanel, AxisID, boolean runnable,
    /// Expander moreLess, InterfaceType)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<ParallelPanel>,
        axis_id: AxisID,
        runnable: bool,
        more_less: Option<Rc<dyn Expander>>,
        interface_type: InterfaceType,
    ) -> Rc<QueueTable> {
        // super(manager, parent, axisID, true, runnable, moreLess, interfaceType);
        // dualSelectionQueueTable = manager.isDualSelectionQueueTable();
        // (The superclass constructor calls no method that reads
        // dualSelectionQueueTable; `getSize` is only called by `createTable`.)
        let instance = Rc::new(QueueTable {
            base: ProcessorTable::new(manager, parent, axis_id, runnable),
            dual_selection_queue_table: manager.is_dual_selection_queue_table(),
            button_group: RefCell::new(None),
            secondary_button_group: RefCell::new(None),
            queue_table_listener_array: RefCell::new(None),
        });
        let this = Rc::downgrade(&instance) as Weak<dyn ProcessorTableVirtual>;
        instance.base.set_this(this);
        // displayQueues is true.
        instance.base.construct(true, more_less, interface_type);
        instance
    }

    /// Java `@Override getProcessorType()`.
    pub fn get_processor_type(&self) -> ProcessorType {
        ProcessorType::Queue
    }

    /// Java `@Override getStorePrepend()`.
    pub fn get_store_prepend(&self) -> String {
        format!("{}{}", self.base.get_group_key(), PREPEND)
    }

    /// Java `@Override getLoadPrepend(ConstEtomoVersion)`.
    pub fn get_load_prepend(&self, _version: &dyn ConstEtomoVersion) -> String {
        format!("{}{}", self.base.get_group_key(), PREPEND)
    }

    /// Java `@Override getSize()`.
    pub fn get_size(&self) -> i32 {
        *self.button_group.borrow_mut() = Some(ButtonGroup::new());
        if self.dual_selection_queue_table {
            *self.secondary_button_group.borrow_mut() = Some(ButtonGroup::new());
        }
        Network::get_num_queues()
    }

    /// Java `@Override getNode(int)`.
    pub fn get_node(&self, index: i32) -> Option<Arc<Node>> {
        Network::get_queue_by_index(index)
    }

    /// Java `@Override createProcessorTableRow(ProcessorTable, Node, int,
    /// ProcessorTableState)`.
    pub fn create_processor_table_row(
        &self,
        processor_table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        let cpus = node.get_cpus();
        let mut i_cpus = 0;
        // Java `cpus != null`: Node.getCpus() never returns null in the
        // translation.
        if !cpus.is_null() {
            i_cpus = cpus.get_int();
        }
        let button_group = self.button_group.borrow().clone();
        let secondary_button_group = self.secondary_button_group.borrow().clone();
        ProcessorTableRow::get_queue_instance(
            processor_table,
            node,
            i_cpus,
            self.dual_selection_queue_table,
            button_group.as_ref(),
            secondary_button_group.as_ref(),
            1.max(cpu_adoc::INSTANCE.get_load_units_array().len() as i32),
            num_rows_in_table,
            table_state,
        )
    }

    /// Java public `@Override enableGpuQueueRows()`.
    pub fn enable_gpu_queue_rows(&self) {
        // The dual selection queue table cannot use the simple isUseGpu
        // functionality.
        if self.dual_selection_queue_table {
            return;
        }
        let mut i = 0;
        while i < self.get_size() {
            // Java `getNode(i)` is non-null for every index below getSize().
            if let Some(curr_node) = self.get_node(i) {
                // When 'Use the GPU' is on, queues without a GPU should be
                // disabled.  When 'Use the GPU' is off, queues with a GPU should
                // be disabled.
                let is_cb_use_gpu = self
                    .base
                    .parent
                    .upgrade()
                    .is_some_and(|parent| parent.is_cb_use_gpu());
                // Java `Network.getQueue(currNode.getName()).isGpu()`: the queue
                // is the node just read by name.
                let is_gpu = Network::get_queue(Some(curr_node.get_name()))
                    .is_some_and(|queue| queue.is_gpu());
                if !is_gpu {
                    self.base
                        .enable_queue_row(Some(curr_node.get_name()), !is_cb_use_gpu);
                } else {
                    self.base
                        .enable_queue_row(Some(curr_node.get_name()), is_cb_use_gpu);
                }
            }
            i += 1;
        }
    }

    /// Java public `@Override actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        // Java `if (event != null)`: an event is always passed.
        // Assuming this is a GPU event because that's the only event this class
        // listens to.
        self.enable_gpu_queue_rows();
    }

    /// Java `@Override addQueueTableListener(QueueTableListener)`.
    pub fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>) {
        self.queue_table_listener_array
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener);
        // Since listeners can be added at any time, send the status to each new
        // listener.
        if let Some(row) = self.base.get_first_selected_row() {
            row.queue_selected_action();
        }
    }

    /// Java `@Override queueTableEventAction(QueueTableEvent)`.
    pub fn queue_table_event_action(&self, event: &QueueTableEvent) {
        // `event instanceof QueueTableDataEvent`: the data-bearing variants.
        match event {
            QueueTableEvent::NumberJobsChanged(jobs) => {
                if let Some(row) = self.base.get_first_selected_row() {
                    let queue_mode = row.get_queue_mode();
                    if queue_mode == Some(QueueMode::NodeWithGpu)
                        || queue_mode == Some(QueueMode::NodeWithoutGpu)
                    {
                        row.set_cpus_selected(Some(jobs.as_str()));
                    }
                }
            }
            QueueTableEvent::OnlyQueueType(queue_type) => {
                self.set_header1_number_cpus_title(Some(*queue_type));
            }
            _ => {}
        }
        self.base.queue_table_event_action_super(event);
    }

    /// Java `setHeader1NumberCPUsTitle(QueueType)`.  (Not overloaded in this
    /// class; the superclass's overloads are `set_header1_number_cpus_title_void`
    /// and `set_header1_number_cpus_title_string`.)
    pub fn set_header1_number_cpus_title(&self, queue_type: Option<QueueType>) {
        if queue_type == Some(QueueType::Node) || queue_type == Some(QueueType::NodeWithoutGpu) {
            self.base
                .set_header1_number_cpus_title_string(Some(NUMBER_JOBS_LABEL1));
        } else {
            self.base.set_header1_number_cpus_title_void();
        }
    }

    /// Java `@Override sendQueueTableEvent(QueueTableEvent)`.
    pub fn send_queue_table_event(&self, event: &QueueTableEvent) {
        // Iterate over a copy: a listener may change the list.
        let listeners = self.queue_table_listener_array.borrow().clone();
        let Some(listeners) = listeners else {
            return;
        };
        for listener in listeners {
            listener.queue_table_event_action(event);
        }
    }

    /// Java `@Override removeQueueTableListener(QueueTableListener)`.
    pub fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>) {
        if let Some(array) = self.queue_table_listener_array.borrow_mut().as_mut() {
            // ArrayList.remove(Object): the first equal element.
            if let Some(index) = array.iter().position(|item| Rc::ptr_eq(item, listener)) {
                array.remove(index);
            }
        }
    }

    /// Java `@Override getHeader1ComputerText()`.
    pub fn get_header1_computer_text(&self) -> Option<String> {
        Some("Queue".to_string())
    }

    /// Java `@Override getNoCpusSelectedErrorMessage()`.
    pub fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        Some("A queue must be selected.".to_string())
    }

    /// Java `useUsersColumn()`.
    pub fn use_users_column(&self) -> bool {
        false
    }

    /// Java `@Override isQueueTable()`.
    pub fn is_queue_table(&self) -> bool {
        true
    }

    /// Java `@Override isCpuTable()`.
    pub fn is_cpu_table(&self) -> bool {
        false
    }

    /// Java `@Override isGpuTable()`.
    pub fn is_gpu_table(&self) -> bool {
        false
    }

    /// Java `@Override getParameters(ProcesschunksParam)`.
    pub fn get_parameters_processchunks_param(&self, param: &ProcesschunksParam) {
        // Avoid loading parameters if this is a secondary table. Only some GPU
        // parameters can be loaded from the secondary table.
        if !self.base.is_secondary() {
            let queue = self.base.get_first_selected_computer();
            let node = Network::get_queue(queue.as_deref());
            if let Some(node) = &node {
                node.get_parameters_processchunks(param);
            }
            param.set_queue(queue.as_deref());
            // Set secondary queue values
            let node = self.base.get_selected_secondary_queue_node();
            if let Some(node) = &node {
                node.get_secondary_parameters_processchunks(param);
            } else {
                param.reset_secondary_queue();
            }
            let row = self.base.get_first_selected_secondary_queue_row();
            if let Some(row) = &row {
                row.get_secondary_parameters_processchunks_param(param);
            } else {
                param.reset_secondary_queue();
            }
        }
        self.base.get_parameters_processchunks_param_super(param);
    }

    /// Java `@Override getParameters(ProcessingMethod, BatchruntomoParam,
    /// boolean)`.
    pub fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        do_validation: bool,
    ) -> bool {
        if method == Some(ProcessingMethod::PpCpu) {
            param.reset_cpu_machine_list();
            self.get_parameters_batchruntomo_param(param);
        } else if method == Some(ProcessingMethod::Queue) {
            self.get_parameters_batchruntomo_param(param);
            // Avoid loading parameters if this is a secondary table. Only some GPU
            // parameters can be loaded from the secondary table.
            if !self.base.is_secondary() {
                let queue = self.base.get_first_selected_computer();
                if queue.is_none() && do_validation {
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string(
                            Some(self.base.manager),
                            "Please select a queue",
                            "No Queue Selected",
                        )
                    });
                    return false;
                }
                let node = Network::get_queue(queue.as_deref());
                if let Some(node) = &node {
                    node.get_parameters_batchruntomo(param);
                }
            }
        }
        true
    }

    /// Java `@Override getParameters(BatchruntomoParam)`.  Only one row can be
    /// selected as the primary queue.
    pub fn get_parameters_batchruntomo_param(&self, param: &mut BatchruntomoParam) {
        let row = self.base.get_first_selected_row();
        if let Some(row) = &row {
            row.get_parameters_batchruntomo_param(param);
        }
        // Set secondary queue values
        let row = self.base.get_first_selected_secondary_queue_row();
        if let Some(row) = &row {
            row.get_secondary_parameters_batchruntomo_param(param);
        } else {
            param.reset_secondary_queue();
        }
        let node = self.base.get_selected_secondary_queue_node();
        if let Some(node) = &node {
            node.get_secondary_parameters_batchruntomo(param);
        } else {
            param.reset_secondary_queue();
        }
    }

    /// Java `@Override getIntermittentCommand(String)`.
    pub fn get_intermittent_command_string(
        &self,
        computer: Option<&str>,
    ) -> Arc<dyn IntermittentCommand> {
        Arc::new(QueuechunkParam::get_load_instance(
            computer,
            self.base.axis_id,
            self.base.manager,
        )) as Arc<dyn IntermittentCommand>
    }

    /// Java `@Override isExcludeNode(Node)`.
    pub fn is_exclude_node(&self, node: &Node) -> bool {
        // Java `if (node == null) return false;`: a node is always passed.
        if self.dual_selection_queue_table
            && node.get_queue_mode() == QueueMode::Invalid
            && !node.is_secondary_queue()
        {
            return true;
        }
        false
    }

    /// Java `@Override isNiceable()`.
    pub fn is_niceable(&self) -> bool {
        true
    }

    /// Java `@Override initRow(ProcessorTableRow)` (empty).
    pub fn init_row(&self, _row: &Rc<ProcessorTableRow>) {}
}

impl ProcessorTableVirtual for QueueTable {
    fn processor_table(&self) -> &ProcessorTable {
        &self.base
    }
    fn get_size(&self) -> i32 {
        QueueTable::get_size(self)
    }
    fn get_node(&self, index: i32) -> Option<Arc<Node>> {
        QueueTable::get_node(self, index)
    }
    fn create_processor_table_row(
        &self,
        processor_table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        QueueTable::create_processor_table_row(
            self,
            processor_table,
            node,
            num_rows_in_table,
            table_state,
        )
    }
    fn get_header1_computer_text(&self) -> Option<String> {
        QueueTable::get_header1_computer_text(self)
    }
    fn get_intermittent_command_string(
        &self,
        computer: Option<&str>,
    ) -> Arc<dyn IntermittentCommand> {
        QueueTable::get_intermittent_command_string(self, computer)
    }
    fn is_exclude_node(&self, node: &Node) -> bool {
        QueueTable::is_exclude_node(self, node)
    }
    fn is_niceable(&self) -> bool {
        QueueTable::is_niceable(self)
    }
    fn get_store_prepend(&self) -> String {
        QueueTable::get_store_prepend(self)
    }
    fn get_load_prepend(&self, version: &dyn ConstEtomoVersion) -> String {
        QueueTable::get_load_prepend(self, version)
    }
    fn init_row(&self, row: &Rc<ProcessorTableRow>) {
        QueueTable::init_row(self, row)
    }
    fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        QueueTable::get_no_cpus_selected_error_message(self)
    }
    fn is_queue_table(&self) -> bool {
        QueueTable::is_queue_table(self)
    }
    fn is_cpu_table(&self) -> bool {
        QueueTable::is_cpu_table(self)
    }
    fn is_gpu_table(&self) -> bool {
        QueueTable::is_gpu_table(self)
    }
    fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        do_validation: bool,
    ) -> bool {
        QueueTable::get_parameters_processing_method_batchruntomo_param_boolean(
            self,
            method,
            param,
            do_validation,
        )
    }
    fn get_processor_type(&self) -> ProcessorType {
        QueueTable::get_processor_type(self)
    }
    fn get_parameters_processchunks_param(&self, param: &ProcesschunksParam) {
        QueueTable::get_parameters_processchunks_param(self, param)
    }
    fn get_parameters_batchruntomo_param(&self, param: &mut BatchruntomoParam) {
        QueueTable::get_parameters_batchruntomo_param(self, param)
    }
    fn get_machine_map(&self, param: &BatchruntomoParam) -> Option<HashMap<String, String>> {
        // Not overridden by QueueTable.
        self.base.get_machine_map_super(param)
    }
    fn action_performed_virtual(&self, event: &ActionEvent) {
        QueueTable::action_performed(self, event)
    }
    fn add_queue_table_listener(&self, listener: Rc<dyn QueueTableListener>) {
        QueueTable::add_queue_table_listener(self, listener)
    }
    fn send_queue_table_event(&self, event: &QueueTableEvent) {
        QueueTable::send_queue_table_event(self, event)
    }
    fn remove_queue_table_listener(&self, listener: &Rc<dyn QueueTableListener>) {
        QueueTable::remove_queue_table_listener(self, listener)
    }
    fn queue_table_event_action(&self, event: &QueueTableEvent) {
        QueueTable::queue_table_event_action(self, event)
    }
    fn enable_gpu_queue_rows(&self) {
        QueueTable::enable_gpu_queue_rows(self)
    }
}

crate::processor_table_interfaces!(QueueTable);
