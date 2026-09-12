//! `IMOD/Etomo/src/etomo/ui/swing/QueueTable.java`.
//!
//! `Network`, `Node`, `CpuAdoc`, `ButtonGroup`, and queue command parameters
//! are explicit boundaries.  The queue-specific selection and event decisions
//! remain in this source unit over the common `ProcessorTable` mechanics.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

use super::processor_table::{ProcessorTable, ProcessorTableHooks, ProcessorTableRow};
use super::processor_table_row::{
    BatchruntomoParameters, ProcesschunksParameters, ProcessorNode, QueueMode, QueueType,
};

const PREPEND: &str = ".Queue";
pub const NUMBER_JOBS_LABEL1: &str = "# Jobs";

/// Java Swing `ButtonGroup` boundary, allocated by `getSize` for queue rows.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ButtonGroupBoundary {
    pub members: Vec<String>,
}

/// Java `QueuechunkParam.getLoadInstance(computer,axisID,manager)` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuechunkParamBoundary {
    pub computer: String,
}

/// Values retained from `Network` and `CpuAdoc` for QueueTable's abstract
/// ProcessorTable calls.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QueueTableHooks {
    pub group_key: String,
    pub queues: Vec<ProcessorNode>,
    pub dual_selection_queue_table: bool,
    pub load_units_count: usize,
}

impl ProcessorTableHooks for QueueTableHooks {
    /// Java `getSize`; ButtonGroup allocation is owned by `QueueTable::get_size`
    /// because the common-table hook is immutable in Rust.
    fn get_size(&self) -> usize {
        self.queues.len()
    }

    /// Java `getNode(int)` after `Network.getQueue`.
    fn get_node(&self, index: usize) -> Option<String> {
        self.queues.get(index).map(|node| node.name.clone())
    }

    /// Java `createProcessorTableRow` after `Node.getCpus` and `CpuAdoc` lookup.
    fn create_processor_table_row(
        &self,
        node: &str,
        num_rows_in_table: usize,
    ) -> ProcessorTableRow {
        let node = self
            .queues
            .iter()
            .find(|queue| queue.name == node)
            .cloned()
            .unwrap_or_else(|| ProcessorNode {
                name: node.into(),
                ..ProcessorNode::default()
            });
        ProcessorTableRow::get_queue_instance(
            node,
            self.dual_selection_queue_table,
            self.load_units_count.max(1),
            num_rows_in_table,
        )
    }

    /// Java `getHeader1ComputerText`.
    fn get_header1_computer_text(&self) -> String {
        "Queue".into()
    }

    /// Java `getIntermittentCommand`.
    fn get_intermittent_command(&self, computer: &str) -> Option<String> {
        Some(computer.into())
    }

    /// Java `isExcludeNode`.
    fn is_exclude_node(&self, node: &str) -> bool {
        self.queues
            .iter()
            .find(|queue| queue.name == node)
            .is_some_and(|queue| {
                self.dual_selection_queue_table
                    && queue.queue_mode == QueueMode::Invalid
                    && !queue.secondary_queue
            })
    }

    /// Java `isNiceable`.
    fn is_niceable(&self) -> bool {
        true
    }

    /// Java `getStorePrepend`.
    fn get_store_prepend(&self) -> String {
        format!("{}{}", self.group_key, PREPEND)
    }

    /// Java `getLoadPrepend`.
    fn get_load_prepend(&self, _: &str) -> String {
        self.get_store_prepend()
    }

    /// Java `initRow` has an empty body.
    fn init_row(&self, _: &mut ProcessorTableRow) {}

    fn is_queue_table(&self) -> bool {
        true
    }
    fn is_cpu_table(&self) -> bool {
        false
    }
    fn is_gpu_table(&self) -> bool {
        false
    }
    fn get_no_cpus_selected_error_message(&self) -> String {
        "A queue must be selected.".into()
    }
}

/// Java `QueueTable`; the Rc/RefCell listener handles are the ownership-safe
/// equivalent of Java's retained `ArrayList<QueueTableListener>` references.
pub struct QueueTable {
    pub table: ProcessorTable<QueueTableHooks>,
    pub dual_selection_queue_table: bool,
    pub button_group: Option<ButtonGroupBoundary>,
    pub secondary_button_group: Option<ButtonGroupBoundary>,
    pub queue_table_listener_array: Vec<Rc<RefCell<dyn QueueTableListener>>>,
    /// Java `parent.isCbUseGpu()` boundary read by `enableGpuQueueRows`.
    pub cb_use_gpu: bool,
    /// Java UIHarness modal-message boundary in validated queue selection.
    pub validation_message: Option<(String, String)>,
}

impl QueueTable {
    /// Java constructor.  Manager/parent/AxisID/Expander/interface arguments are
    /// represented by their direct Network/CpuAdoc/ParallelPanel boundary values.
    pub fn new(
        group_key: impl Into<String>,
        queues: Vec<ProcessorNode>,
        dual_selection_queue_table: bool,
        load_units_count: usize,
        displayed_fields: BTreeSet<String>,
        runnable: bool,
        no_load: bool,
    ) -> Self {
        Self {
            table: ProcessorTable::new(
                QueueTableHooks {
                    group_key: group_key.into(),
                    queues,
                    dual_selection_queue_table,
                    load_units_count,
                },
                displayed_fields,
                true,
                runnable,
                no_load,
            ),
            dual_selection_queue_table,
            button_group: None,
            secondary_button_group: None,
            queue_table_listener_array: Vec::new(),
            cb_use_gpu: false,
            validation_message: None,
        }
    }

    /// Java `getProcessorType`.
    pub fn get_processor_type(&self) -> &'static str {
        "QUEUE"
    }

    /// Java `getStorePrepend`.
    pub fn get_store_prepend(&self) -> String {
        self.table.hooks.get_store_prepend()
    }

    /// Java `getLoadPrepend`.
    pub fn get_load_prepend(&self, version: &str) -> String {
        self.table.hooks.get_load_prepend(version)
    }

    /// Java `getSize`, including the ButtonGroup initialization side effect.
    pub fn get_size(&mut self) -> usize {
        self.button_group = Some(ButtonGroupBoundary::default());
        if self.dual_selection_queue_table {
            self.secondary_button_group = Some(ButtonGroupBoundary::default());
        }
        self.table.hooks.queues.len()
    }

    /// Java `getNode(int)` after the Network boundary.
    pub fn get_node(&self, index: usize) -> Option<&ProcessorNode> {
        self.table.hooks.queues.get(index)
    }

    /// Java `createProcessorTableRow`.
    pub fn create_processor_table_row(
        &self,
        node: &ProcessorNode,
        num_rows_in_table: usize,
    ) -> ProcessorTableRow {
        ProcessorTableRow::get_queue_instance(
            node.clone(),
            self.dual_selection_queue_table,
            self.table.hooks.load_units_count.max(1),
            num_rows_in_table,
        )
    }

    /// Java `createTable`, with its pre-build ButtonGroup initialization.
    pub fn create_table(&mut self) {
        self.get_size();
        self.table.create_table();
        if let Some(group) = &mut self.button_group {
            group.members = self
                .table
                .row_list
                .list
                .iter()
                .map(|row| row.computer.clone())
                .collect();
        }
        if let Some(group) = &mut self.secondary_button_group {
            group.members = self
                .table
                .row_list
                .list
                .iter()
                .filter(|row| row.has_secondary_queue)
                .map(|row| row.computer.clone())
                .collect();
        }
    }

    /// Java `enableGpuQueueRows`.
    pub fn enable_gpu_queue_rows(&mut self) {
        if self.dual_selection_queue_table {
            return;
        }
        let queues: Vec<(String, bool)> = self
            .table
            .hooks
            .queues
            .iter()
            .map(|queue| (queue.name.clone(), queue.is_gpu))
            .collect();
        for (name, is_gpu) in queues {
            self.table.enable_queue_row(
                &name,
                if is_gpu {
                    self.cb_use_gpu
                } else {
                    !self.cb_use_gpu
                },
            );
        }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self, event_present: bool) {
        if event_present {
            self.enable_gpu_queue_rows();
        }
    }

    /// Java `addQueueTableListener`.
    pub fn add_queue_table_listener(&mut self, listener: Rc<RefCell<dyn QueueTableListener>>) {
        self.queue_table_listener_array.push(listener);
        let index = self.table.get_first_selected_index();
        let event = self.table.row_list.get_mut(index).and_then(|row| {
            row.queue_selected_action();
            row.queue_selected_event
                .as_ref()
                .map(|(queue_mode, maximum)| QueueTableEvent::QueueSelected {
                    queue_mode: *queue_mode,
                    maximum: Some(maximum.clone()),
                })
        });
        if let Some(event) = event {
            self.send_queue_table_event(event);
        }
    }

    /// Java `queueTableEventAction`.
    pub fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        match &event {
            QueueTableEvent::NumberJobsChanged(jobs) => {
                if let Some(row) = self.table.get_first_selected_row() {
                    if row.get_queue_mode() == QueueMode::Node {
                        let index = self.table.get_first_selected_index();
                        if let Some(row) = self.table.row_list.get_mut(index) {
                            row.set_cpus_selected(jobs);
                        }
                    }
                }
            }
            QueueTableEvent::OnlyQueueType(queue_type) => {
                self.set_header1_number_cpus_title(*queue_type);
            }
            _ => {}
        }
        self.table.queue_table_event_action(event);
    }

    /// Java `setHeader1NumberCPUsTitle(QueueType)`.
    pub fn set_header1_number_cpus_title(&mut self, queue_type: QueueType) {
        if matches!(queue_type, QueueType::Node | QueueType::NodeWithoutGpu) {
            self.table
                .set_header1_number_cpus_title_to(NUMBER_JOBS_LABEL1);
        } else {
            self.table.set_header1_number_cpus_title();
        }
    }

    /// Java `sendQueueTableEvent`.
    pub fn send_queue_table_event(&mut self, event: QueueTableEvent) {
        for listener in &self.queue_table_listener_array {
            listener
                .borrow_mut()
                .queue_table_event_action(event.clone());
        }
        self.table.send_queue_table_event(event);
    }

    /// Java `removeQueueTableListener`.
    pub fn remove_queue_table_listener(&mut self, listener: &Rc<RefCell<dyn QueueTableListener>>) {
        self.queue_table_listener_array
            .retain(|candidate| !Rc::ptr_eq(candidate, listener));
    }

    /// Java `getHeader1ComputerText`.
    pub fn get_header1_computer_text(&self) -> &'static str {
        "Queue"
    }

    /// Java `getNoCpusSelectedErrorMessage`.
    pub fn get_no_cpus_selected_error_message(&self) -> &'static str {
        "A queue must be selected."
    }

    /// Java `useUsersColumn`.
    pub fn use_users_column(&self) -> bool {
        false
    }

    /// Java `isQueueTable`.
    pub fn is_queue_table(&self) -> bool {
        true
    }

    /// Java `isCpuTable`.
    pub fn is_cpu_table(&self) -> bool {
        false
    }

    /// Java `isGpuTable`.
    pub fn is_gpu_table(&self) -> bool {
        false
    }

    /// Java `getParameters(ProcesschunksParam)`.
    pub fn get_processchunks_parameters(&self, param: &mut ProcesschunksParameters) {
        if !self.table.is_secondary() {
            param.queue = self.table.get_first_selected_computer().map(str::to_owned);
            if let Some(row) = self.table.get_first_selected_secondary_queue_row() {
                row.get_secondary_processchunks_parameters(param, true);
            } else {
                param.secondary_number = None;
            }
        }
        for row in &self.table.row_list.list {
            row.get_processchunks_parameters(param, false, self.table.is_secondary(), true, false);
        }
    }

    /// Java `getParameters(ProcessingMethod,BatchruntomoParam,boolean)`.
    pub fn get_parameters(
        &mut self,
        method: ProcessingMethod,
        param: &mut BatchruntomoParameters,
        do_validation: bool,
    ) -> bool {
        if method == ProcessingMethod::PpCpu {
            param.cpu_machines.clear();
            self.get_batchruntomo_parameters(param);
        } else if method == ProcessingMethod::Queue {
            self.get_batchruntomo_parameters(param);
            if !self.table.is_secondary()
                && self.table.get_first_selected_computer().is_none()
                && do_validation
            {
                self.validation_message =
                    Some(("Please select a queue".into(), "No Queue Selected".into()));
                return false;
            }
        }
        true
    }

    /// Java `getParameters(BatchruntomoParam)`.
    pub fn get_batchruntomo_parameters(&self, param: &mut BatchruntomoParameters) {
        if let Some(row) = self.table.get_first_selected_row() {
            row.get_batchruntomo_parameters(param, true, false, false);
        }
        if let Some(row) = self.table.get_first_selected_secondary_queue_row() {
            row.get_secondary_batchruntomo_parameters(param, true);
        } else {
            param.max_gpu_jobs_on_queue = None;
        }
    }

    /// Java `getIntermittentCommand`.
    pub fn get_intermittent_command(&self, computer: impl Into<String>) -> QueuechunkParamBoundary {
        QueuechunkParamBoundary {
            computer: computer.into(),
        }
    }

    /// Java `isExcludeNode`.
    pub fn is_exclude_node(&self, node: &ProcessorNode) -> bool {
        self.dual_selection_queue_table
            && node.queue_mode == QueueMode::Invalid
            && !node.secondary_queue
    }

    /// Java `isNiceable`.
    pub fn is_niceable(&self) -> bool {
        true
    }

    /// Java `initRow` has an empty body.
    pub fn init_row(&self, _: &mut ProcessorTableRow) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Listener {
        events: Vec<QueueTableEvent>,
    }
    impl QueueTableListener for Listener {
        fn queue_table_event_action(&mut self, event: QueueTableEvent) {
            self.events.push(event);
        }
    }

    fn queue(name: &str, is_gpu: bool, mode: QueueMode) -> ProcessorNode {
        ProcessorNode {
            name: name.into(),
            num_cpus: 8,
            is_gpu,
            queue_mode: mode,
            ..ProcessorNode::default()
        }
    }

    #[test]
    fn source_identity_groups_and_gpu_enablement_are_preserved() {
        let mut table = QueueTable::new(
            "group",
            vec![
                queue("cpu", false, QueueMode::Queue),
                queue("gpu", true, QueueMode::Queue),
            ],
            false,
            0,
            BTreeSet::new(),
            true,
            true,
        );
        assert_eq!(table.get_processor_type(), "QUEUE");
        assert_eq!(table.get_store_prepend(), "group.Queue");
        assert_eq!(table.get_load_prepend("0.0"), "group.Queue");
        table.create_table();
        assert_eq!(table.button_group.as_ref().unwrap().members, ["cpu", "gpu"]);
        table.cb_use_gpu = true;
        table.action_performed(true);
        assert!(!table.table.get_row("cpu").unwrap().selection_enabled);
        assert!(table.table.get_row("gpu").unwrap().selection_enabled);
    }

    #[test]
    fn data_events_update_jobs_header_rows_and_listener_dispatch() {
        let mut table = QueueTable::new(
            "",
            vec![queue("node", false, QueueMode::Node)],
            true,
            1,
            BTreeSet::new(),
            true,
            true,
        );
        table.create_table();
        table.queue_table_event_action(QueueTableEvent::NumberJobsChanged("6".into()));
        assert_eq!(table.table.get_row("node").unwrap().cpus_selected, 6);
        table.queue_table_event_action(QueueTableEvent::OnlyQueueType(QueueType::Node));
        assert_eq!(table.table.header1_number_cpus, NUMBER_JOBS_LABEL1);
        let listener = Rc::new(RefCell::new(Listener::default()));
        table.add_queue_table_listener(listener.clone());
        table.send_queue_table_event(QueueTableEvent::Displayed);
        assert_eq!(
            listener.borrow().events,
            vec![
                QueueTableEvent::QueueSelected {
                    queue_mode: QueueMode::Node,
                    maximum: Some("8".into()),
                },
                QueueTableEvent::Displayed,
            ]
        );
    }

    #[test]
    fn queue_validation_and_secondary_boundary_follow_source() {
        let mut table = QueueTable::new("", vec![], false, 1, BTreeSet::new(), true, true);
        let mut params = BatchruntomoParameters::default();
        assert!(!table.get_parameters(ProcessingMethod::Queue, &mut params, true));
        assert_eq!(
            table.validation_message,
            Some(("Please select a queue".into(), "No Queue Selected".into()))
        );
        let mut chunks = ProcesschunksParameters::default();
        table.get_processchunks_parameters(&mut chunks);
        assert_eq!(chunks.secondary_number, None);
    }
}
