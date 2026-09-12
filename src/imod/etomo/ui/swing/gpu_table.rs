//! `IMOD/Etomo/src/etomo/ui/swing/GpuTable.java`.
//!
//! The Java class inherits the common CPU computer-table behaviour and changes
//! only the GPU-specific source methods below.  `Network`, `Node`,
//! `BaseManager`, and comscript parameter classes remain direct boundaries;
//! their values are represented by the explicit projections in this unit.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

use super::cpu_table::ProcessorType;
use super::processor_table::{ProcessorTable, ProcessorTableHooks, ProcessorTableRow};
use super::processor_table_row::{BatchruntomoParameters, ProcesschunksParameters, ProcessorNode};

const PREPEND: &str = ".Gpu";
pub const NUMBER_CPUS_LABEL: &str = "# GPUs";

/// Java `GpuTable` values supplied to inherited `ProcessorTable` methods.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuTableHooks {
    pub group_key: String,
    pub computers: Vec<ProcessorNode>,
}

impl ProcessorTableHooks for GpuTableHooks {
    /// Java inherited `CpuTable.getSize`.
    fn get_size(&self) -> usize {
        self.computers.len()
    }

    /// Java inherited `CpuTable.getNode(int)` after `Network.getComputer`.
    fn get_node(&self, index: usize) -> Option<String> {
        self.computers.get(index).map(|node| node.name.clone())
    }

    /// Java `createProcessorTableRow`.
    fn create_processor_table_row(
        &self,
        node: &str,
        num_rows_in_table: usize,
    ) -> ProcessorTableRow {
        let mut node = self
            .computers
            .iter()
            .find(|computer| computer.name == node)
            .cloned()
            .unwrap_or_else(|| ProcessorNode {
                name: node.into(),
                ..ProcessorNode::default()
            });
        // Java Node.getGpuNumber returns one for non-GPU nodes and for a GPU
        // node without an explicit device array.
        node.num_cpus = if node.is_gpu && !node.gpu_device_array.is_empty() {
            node.gpu_device_array.len() as i32
        } else {
            1
        };
        ProcessorTableRow::get_computer_instance(node, num_rows_in_table)
    }

    /// Java `getHeader1ComputerText`.
    fn get_header1_computer_text(&self) -> String {
        "GPU".into()
    }

    /// Java inherited `CpuTable.getIntermittentCommand` boundary projection.
    fn get_intermittent_command(&self, computer: &str) -> Option<String> {
        Some(computer.into())
    }

    /// Java `isExcludeNode` after the `Node.isLocalHost` manager boundary.
    fn is_exclude_node(&self, node: &str) -> bool {
        self.computers
            .iter()
            .find(|computer| computer.name == node)
            .is_none_or(|computer| {
                !computer.is_gpu || (computer.is_gpu_local && !computer.is_local_host)
            })
    }

    /// Java inherited `CpuTable.isNiceable`.
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

    /// Java `initRow`.
    fn init_row(&self, row: &mut ProcessorTableRow) {
        row.turn_off_load_warning();
    }

    fn is_queue_table(&self) -> bool {
        false
    }
    fn is_cpu_table(&self) -> bool {
        false
    }
    fn is_gpu_table(&self) -> bool {
        true
    }
    fn get_no_cpus_selected_error_message(&self) -> String {
        "At least one GPU must be selected.".into()
    }
}

/// Java `GpuTable`.  `table` is the inherited `ProcessorTable` state.
pub struct GpuTable {
    pub table: ProcessorTable<GpuTableHooks>,
    /// Java inherited `CpuTable.usersColumn` / `CpuAdoc.isUsersColumn` value.
    pub users_column: bool,
    /// Java `BaseManager.isAddGPUMachineToProcessChunks()` boundary.
    pub add_gpu_machine_to_processchunks: bool,
}

impl GpuTable {
    /// Java `GpuTable(BaseManager,ParallelPanel,AxisID,boolean,Expander,InterfaceType)`.
    /// Manager/UI arguments are represented by their direct Network/CpuAdoc and
    /// command-boundary values.
    pub fn new(
        group_key: impl Into<String>,
        computers: Vec<ProcessorNode>,
        users_column: bool,
        displayed_fields: BTreeSet<String>,
        runnable: bool,
        no_load: bool,
    ) -> Self {
        let mut table = ProcessorTable::new(
            GpuTableHooks {
                group_key: group_key.into(),
                computers,
            },
            displayed_fields,
            false,
            runnable,
            no_load,
        );
        // ProcessorTable's Java constructor dynamically dispatches to
        // GpuTable.getheader1NumberCPUsTitle while creating this header.
        table.set_header1_number_cpus_title_to(NUMBER_CPUS_LABEL);
        Self {
            table,
            users_column,
            add_gpu_machine_to_processchunks: false,
        }
    }

    /// Java `getProcessorType`.
    pub fn get_processor_type(&self) -> ProcessorType {
        ProcessorType::Gpu
    }

    /// Java `isCpuTable`.
    pub fn is_cpu_table(&self) -> bool {
        false
    }

    /// Java `isGpuTable`.
    pub fn is_gpu_table(&self) -> bool {
        true
    }

    /// Java `getheader1NumberCPUsTitle`.
    pub fn getheader1_number_cpus_title(&self) -> &'static str {
        NUMBER_CPUS_LABEL
    }

    /// Java `getStorePrepend`.
    pub fn get_store_prepend(&self) -> String {
        self.table.hooks.get_store_prepend()
    }

    /// Java `getLoadPrepend(ConstEtomoVersion)`.
    pub fn get_load_prepend(&self, version: &str) -> String {
        self.table.hooks.get_load_prepend(version)
    }

    /// Java `getHeader1ComputerText`.
    pub fn get_header1_computer_text(&self) -> &'static str {
        "GPU"
    }

    /// Java `getNoCpusSelectedErrorMessage`.
    pub fn get_no_cpus_selected_error_message(&self) -> &'static str {
        "At least one GPU must be selected."
    }

    /// Java `isExcludeNode`.
    pub fn is_exclude_node(&self, node: &ProcessorNode) -> bool {
        !node.is_gpu || (node.is_gpu_local && !node.is_local_host)
    }

    /// Java `getMachineMap(BatchruntomoParam)`.
    pub fn get_machine_map(&self, param: &BatchruntomoParameters) -> BTreeMap<String, String> {
        param
            .gpu_machines
            .iter()
            .map(|(name, number, _)| (name.clone(), number.to_string()))
            .collect()
    }

    /// Java `getParameters(ProcesschunksParam)`.
    pub fn get_processchunks_parameters(&self, param: &mut ProcesschunksParameters) {
        if !self.table.is_secondary() {
            param.gpu_processing = true;
        }
        if self.add_gpu_machine_to_processchunks {
            param.gpu_machines.clear();
        }
        for row in &self.table.row_list.list {
            row.get_processchunks_parameters(
                param,
                self.add_gpu_machine_to_processchunks,
                self.table.is_secondary(),
                false,
                true,
            );
        }
    }

    /// Java `getParameters(ProcessingMethod,BatchruntomoParam,boolean)`.
    pub fn get_parameters(
        &self,
        method: ProcessingMethod,
        param: &mut BatchruntomoParameters,
        _: bool,
    ) -> bool {
        if method == ProcessingMethod::PpGpu {
            param.gpu_machines.clear();
            self.get_batchruntomo_parameters(param);
        }
        true
    }

    /// Java inherited `ProcessorTable.getParameters(BatchruntomoParam)`.
    pub fn get_batchruntomo_parameters(&self, param: &mut BatchruntomoParameters) {
        for row in &self.table.row_list.list {
            row.get_batchruntomo_parameters(param, false, false, true);
        }
    }

    /// Java `enableNumberColumn`.
    pub fn enable_number_column(&self, node: &ProcessorNode) -> bool {
        node.is_gpu && node.gpu_device_array.len() > 1
    }

    /// Java `createProcessorTableRow`.
    pub fn create_processor_table_row(
        &self,
        node: &ProcessorNode,
        num_rows_in_table: usize,
    ) -> ProcessorTableRow {
        self.table
            .hooks
            .create_processor_table_row(&node.name, num_rows_in_table)
    }

    /// Java `initRow`.
    pub fn init_row(&self, row: &mut ProcessorTableRow) {
        row.turn_off_load_warning();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, devices: &[&str]) -> ProcessorNode {
        ProcessorNode {
            name: name.into(),
            is_gpu: true,
            gpu_device_array: devices.iter().map(|device| (*device).into()).collect(),
            ..ProcessorNode::default()
        }
    }

    #[test]
    fn gpu_identity_headers_and_storage_are_source_exact() {
        let table = GpuTable::new("parallel", vec![], true, BTreeSet::new(), true, false);
        assert_eq!(table.get_processor_type(), ProcessorType::Gpu);
        assert!(!table.is_cpu_table());
        assert!(table.is_gpu_table());
        assert_eq!(table.getheader1_number_cpus_title(), "# GPUs");
        assert_eq!(table.table.header1_number_cpus, "# GPUs");
        assert_eq!(table.get_header1_computer_text(), "GPU");
        assert_eq!(
            table.get_no_cpus_selected_error_message(),
            "At least one GPU must be selected."
        );
        assert_eq!(table.get_store_prepend(), "parallel.Gpu");
        assert_eq!(table.get_load_prepend("1.0"), "parallel.Gpu");
    }

    #[test]
    fn only_available_gpu_nodes_are_created_and_their_gpu_count_is_used() {
        let mut unavailable_local = gpu("remote-local", &["0"]);
        unavailable_local.is_gpu_local = true;
        let mut table = GpuTable::new(
            "group",
            vec![
                ProcessorNode {
                    name: "cpu-only".into(),
                    ..ProcessorNode::default()
                },
                unavailable_local,
                gpu("gpu-a", &["0", "1"]),
            ],
            false,
            BTreeSet::new(),
            true,
            true,
        );
        table.table.create_table();
        assert_eq!(table.table.row_list.list.len(), 1);
        let row = &table.table.row_list.list[0];
        assert_eq!(row.computer, "gpu-a");
        assert_eq!(row.num_cpus, 2);
        assert!(!row.load_warning);
        assert!(table.enable_number_column(&gpu("gpu-a", &["0", "1"])));
        assert!(!table.enable_number_column(&gpu("gpu-b", &["0"])));
    }

    #[test]
    fn gpu_parameters_follow_secondary_and_add_gpu_machine_rules() {
        let mut table = GpuTable::new(
            "group",
            vec![gpu("gpu-a", &["0", "1"])],
            false,
            BTreeSet::new(),
            true,
            true,
        );
        table.table.create_table();
        table.add_gpu_machine_to_processchunks = true;
        let mut chunks = ProcesschunksParameters {
            gpu_machines: vec![("old".into(), 1, vec![])],
            ..ProcesschunksParameters::default()
        };
        table.get_processchunks_parameters(&mut chunks);
        assert!(chunks.gpu_processing);
        assert_eq!(
            chunks.machine_names,
            vec![("gpu-a".into(), 1, vec!["0".into(), "1".into()])]
        );
        assert_eq!(chunks.gpu_machines, chunks.machine_names);

        let mut batch = BatchruntomoParameters {
            gpu_machines: vec![("old".into(), 1, vec![])],
            ..BatchruntomoParameters::default()
        };
        assert!(table.get_parameters(ProcessingMethod::PpGpu, &mut batch, true));
        assert_eq!(batch.gpu_machines, chunks.machine_names);
        assert_eq!(table.get_machine_map(&batch)["gpu-a"], "1");
    }
}
