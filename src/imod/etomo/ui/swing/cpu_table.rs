//! `IMOD/Etomo/src/etomo/ui/swing/CpuTable.java`.
//!
//! `Network`, `CpuAdoc`, `LoadAverageParam`, and Swing's `ButtonGroup` are
//! direct application/UI boundaries.  Their values and command construction are
//! retained explicitly here; the common table mechanics remain in the source
//! shaped `ProcessorTable` unit.
#![allow(dead_code)]

use std::collections::BTreeSet;

use super::processor_table::{ProcessorTable, ProcessorTableHooks, ProcessorTableRow};
use super::processor_table_row::{BatchruntomoParameters, ProcesschunksParameters, ProcessorNode};
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;

const PREPEND: &str = ".Cpu";

/// Java `ProcessorType` value selected by `getProcessorType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessorType {
    Cpu,
    Gpu,
}

/// Java `LoadAverageParam.getInstance(computer, manager)` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadAverageParamBoundary {
    pub computer: String,
}

/// Java `CpuTable` data required by the `ProcessorTable` abstract-method calls.
/// `Network.getComputer` is represented by the ordered computer projection and
/// `BaseManager.getGroupKey` by `group_key`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CpuTableHooks {
    pub group_key: String,
    pub computers: Vec<ProcessorNode>,
}

impl ProcessorTableHooks for CpuTableHooks {
    /// Java `getSize`.
    fn get_size(&self) -> usize {
        self.computers.len()
    }

    /// Java `getNode(int)`, after the direct `Network` lookup boundary.
    fn get_node(&self, index: usize) -> Option<String> {
        self.computers.get(index).map(|node| node.name.clone())
    }

    /// Java `createProcessorTableRow(ProcessorTable,Node,int,ProcessorTableState)`.
    fn create_processor_table_row(
        &self,
        node: &str,
        num_rows_in_table: usize,
    ) -> ProcessorTableRow {
        let node = self
            .computers
            .iter()
            .find(|computer| computer.name == node)
            .cloned()
            .unwrap_or_else(|| ProcessorNode {
                name: node.into(),
                ..ProcessorNode::default()
            });
        ProcessorTableRow::get_computer_instance(node, num_rows_in_table)
    }

    /// Java `getHeader1ComputerText`.
    fn get_header1_computer_text(&self) -> String {
        "Computer".into()
    }

    /// Java `getIntermittentCommand`; `LoadAverageParam` is represented by its
    /// computer-key command projection in the common table boundary.
    fn get_intermittent_command(&self, computer: &str) -> Option<String> {
        Some(computer.into())
    }

    /// Java `isExcludeNode`.
    fn is_exclude_node(&self, _: &str) -> bool {
        false
    }

    /// Java `isNiceable`.
    fn is_niceable(&self) -> bool {
        true
    }

    /// Java `getStorePrepend`.
    fn get_store_prepend(&self) -> String {
        format!("{}{}", self.group_key, PREPEND)
    }

    /// Java `getLoadPrepend(ConstEtomoVersion)`.
    fn get_load_prepend(&self, version: &str) -> String {
        let mut pieces = version.split('.');
        let major = pieces
            .next()
            .and_then(|piece| piece.parse::<u32>().ok())
            .unwrap_or(0);
        let minor = pieces
            .next()
            .and_then(|piece| piece.parse::<u32>().ok())
            .unwrap_or(0);
        if major > 1 || (major == 1 && minor >= 1) {
            format!("{}{}", self.group_key, PREPEND)
        } else {
            "ProcessorTable".into()
        }
    }

    /// Java `initRow` has an empty body.
    fn init_row(&self, _: &mut ProcessorTableRow) {}

    /// Java `isQueueTable`.
    fn is_queue_table(&self) -> bool {
        false
    }

    /// Java `isCpuTable`.
    fn is_cpu_table(&self) -> bool {
        true
    }

    /// Java `isGpuTable`.
    fn is_gpu_table(&self) -> bool {
        false
    }

    /// Java `getNoCpusSelectedErrorMessage`.
    fn get_no_cpus_selected_error_message(&self) -> String {
        "At least one computer must be selected.".into()
    }
}

/// Java `CpuTable`.  `table` is its `ProcessorTable` superclass state.
pub struct CpuTable {
    pub table: ProcessorTable<CpuTableHooks>,
    pub users_column: bool,
    /// Java `BaseManager.isAddGPUMachineToProcessChunks()` boundary read by
    /// the inherited `ProcessorTable.getParameters(ProcesschunksParam)`.
    pub add_gpu_machine_to_processchunks: bool,
}

impl CpuTable {
    /// Java `CpuTable(BaseManager,ParallelPanel,AxisID,boolean,Expander,InterfaceType)`.
    /// Swing/manager arguments are represented by their direct values: table
    /// field policy, `Network` computers, `CpuAdoc.isUsersColumn`, and no-load.
    pub fn new(
        group_key: impl Into<String>,
        computers: Vec<ProcessorNode>,
        users_column: bool,
        displayed_fields: BTreeSet<String>,
        runnable: bool,
        no_load: bool,
    ) -> Self {
        Self {
            table: ProcessorTable::new(
                CpuTableHooks {
                    group_key: group_key.into(),
                    computers,
                },
                displayed_fields,
                false,
                runnable,
                no_load,
            ),
            users_column,
            add_gpu_machine_to_processchunks: false,
        }
    }

    /// Java `getProcessorType`.
    pub fn get_processor_type(&self) -> ProcessorType {
        ProcessorType::Cpu
    }

    /// Java `isQueueTable`.
    pub fn is_queue_table(&self) -> bool {
        false
    }

    /// Java `isCpuTable`.
    pub fn is_cpu_table(&self) -> bool {
        true
    }

    /// Java `isGpuTable`.
    pub fn is_gpu_table(&self) -> bool {
        false
    }

    /// Java `getStorePrepend`.
    pub fn get_store_prepend(&self) -> String {
        format!("{}{}", self.table.hooks.group_key, PREPEND)
    }

    /// Java `getLoadPrepend(ConstEtomoVersion)`.
    pub fn get_load_prepend(&self, version: &str) -> String {
        let mut pieces = version.split('.');
        let major = pieces
            .next()
            .and_then(|piece| piece.parse::<u32>().ok())
            .unwrap_or(0);
        let minor = pieces
            .next()
            .and_then(|piece| piece.parse::<u32>().ok())
            .unwrap_or(0);
        if major > 1 || (major == 1 && minor >= 1) {
            self.get_store_prepend()
        } else {
            "ProcessorTable".into()
        }
    }

    /// Java `getSize`.
    pub fn get_size(&self) -> usize {
        self.table.hooks.computers.len()
    }

    /// Java `getParameters(ProcessingMethod,BatchruntomoParam,boolean)`.
    pub fn get_parameters(
        &self,
        method: ProcessingMethod,
        param: &mut BatchruntomoParameters,
        _: bool,
    ) -> bool {
        if method == ProcessingMethod::PpCpu {
            param.cpu_machines.clear();
            for row in &self.table.row_list.list {
                row.get_batchruntomo_parameters(param, false, true, false);
            }
        }
        true
    }

    /// Java inherited `ProcessorTable.getParameters(ProcesschunksParam)`.
    pub fn get_processchunks_parameters(&self, param: &mut ProcesschunksParameters) {
        if self.add_gpu_machine_to_processchunks {
            param.gpu_machines.clear();
        }
        for row in &self.table.row_list.list {
            row.get_processchunks_parameters(
                param,
                self.add_gpu_machine_to_processchunks,
                self.table.is_secondary(),
                false,
                false,
            );
        }
    }

    /// Java `getButtonGroup`; CPU tables do not own a Swing `ButtonGroup`.
    pub fn get_button_group(&self) -> Option<()> {
        None
    }

    /// Java `getNode(int)`, after the direct `Network.getComputer` boundary.
    pub fn get_node(&self, index: usize) -> Option<&ProcessorNode> {
        self.table.hooks.computers.get(index)
    }

    /// Java `createProcessorTableRow`.
    pub fn create_processor_table_row(
        &self,
        node: &ProcessorNode,
        num_rows_in_table: usize,
    ) -> ProcessorTableRow {
        ProcessorTableRow::get_computer_instance(node.clone(), num_rows_in_table)
    }

    /// Java `getHeader1ComputerText`.
    pub fn get_header1_computer_text(&self) -> &'static str {
        "Computer"
    }

    /// Java `getNoCpusSelectedErrorMessage`.
    pub fn get_no_cpus_selected_error_message(&self) -> &'static str {
        "At least one computer must be selected."
    }

    /// Java `getIntermittentCommand(String)`.
    pub fn get_intermittent_command(
        &self,
        computer: impl Into<String>,
    ) -> LoadAverageParamBoundary {
        LoadAverageParamBoundary {
            computer: computer.into(),
        }
    }

    /// Java `isExcludeNode`.
    pub fn is_exclude_node(&self, _: &ProcessorNode) -> bool {
        false
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

    #[test]
    fn cpu_identity_and_storage_follow_source() {
        let table = CpuTable::new("parallel", vec![], true, BTreeSet::new(), true, false);
        assert_eq!(table.get_processor_type(), ProcessorType::Cpu);
        assert!(table.is_cpu_table());
        assert!(!table.is_queue_table());
        assert!(!table.is_gpu_table());
        assert!(table.users_column);
        assert_eq!(table.get_store_prepend(), "parallel.Cpu");
        assert_eq!(table.get_load_prepend("1.0"), "ProcessorTable");
        assert_eq!(table.get_load_prepend("1.1"), "parallel.Cpu");
    }

    #[test]
    fn rows_are_computer_rows_with_source_cpu_values() {
        let mut table = CpuTable::new(
            "group",
            vec![ProcessorNode {
                name: "node-a".into(),
                num_cpus: 8,
                ..ProcessorNode::default()
            }],
            false,
            BTreeSet::new(),
            true,
            true,
        );
        assert_eq!(table.get_size(), 1);
        assert_eq!(table.get_node(0).unwrap().num_cpus, 8);
        table.table.create_table();
        assert_eq!(table.table.row_list.list.len(), 1);
        assert_eq!(table.table.row_list.list[0].num_cpus, 8);
        assert!(table.table.row_list.list[0].selected);
    }

    #[test]
    fn batchruntomo_parameters_only_change_for_pp_cpu() {
        let mut table = CpuTable::new(
            "group",
            vec![ProcessorNode {
                name: "node-a".into(),
                num_cpus: 4,
                ..ProcessorNode::default()
            }],
            false,
            BTreeSet::new(),
            true,
            true,
        );
        table.table.create_table();
        let mut param = BatchruntomoParameters {
            cpu_machines: vec![("old".into(), 1)],
            ..BatchruntomoParameters::default()
        };
        assert!(table.get_parameters(ProcessingMethod::LocalCpu, &mut param, true));
        assert_eq!(param.cpu_machines, vec![("old".into(), 1)]);
        assert!(table.get_parameters(ProcessingMethod::PpCpu, &mut param, true));
        assert_eq!(param.cpu_machines, vec![("node-a".into(), 1)]);
    }

    #[test]
    fn inherited_processchunks_path_keeps_cpu_rows_out_of_gpu_machine_list() {
        let mut table = CpuTable::new(
            "group",
            vec![ProcessorNode {
                name: "node-a".into(),
                num_cpus: 4,
                gpu_device_array: vec!["0".into()],
                ..ProcessorNode::default()
            }],
            false,
            BTreeSet::new(),
            true,
            true,
        );
        table.table.create_table();
        table.add_gpu_machine_to_processchunks = true;
        let mut param = ProcesschunksParameters {
            gpu_machines: vec![("old".into(), 1, vec![])],
            ..ProcesschunksParameters::default()
        };
        table.get_processchunks_parameters(&mut param);
        assert_eq!(
            param.machine_names,
            vec![("node-a".into(), 1, vec!["0".into()])]
        );
        assert!(param.gpu_machines.is_empty());
    }

    #[test]
    fn remaining_abstract_method_values_are_exact() {
        let table = CpuTable::new("", vec![], false, BTreeSet::new(), false, true);
        assert_eq!(table.get_button_group(), None);
        assert_eq!(table.get_header1_computer_text(), "Computer");
        assert_eq!(
            table.get_no_cpus_selected_error_message(),
            "At least one computer must be selected."
        );
        assert_eq!(table.get_intermittent_command("host").computer, "host");
        assert!(table.is_niceable());
    }
}
