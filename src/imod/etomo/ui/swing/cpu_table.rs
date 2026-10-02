//! `IMOD/Etomo/src/etomo/ui/swing/CpuTable.java`.
//!
//! Child of ProcessorTable that makes a ProcessorTable display CPUs (bug# 1422).
//!
//! Java `class CpuTable extends ProcessorTable`, itself extended by `GpuTable`.  The
//! superclass is embedded as `base` (with `Deref`), and the abstract and overridden
//! members are [`ProcessorTableVirtual`].  CpuTable's own bodies are inherent
//! methods here, so that `GpuTable` (which embeds a `CpuTable`) can reuse the ones it
//! inherits.  Construction follows `ProcessorTable`'s split: the subclass allocates
//! itself ([`CpuTable::new_fields`]), connects `this`, then runs the superclass
//! constructor body ([`ProcessorTable::construct`]).

use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::parallel_panel::ParallelPanel;
use super::processor_table::{ProcessorTable, ProcessorTableVirtual};
use super::processor_table_row::ProcessorTableRow;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::comscript::load_average_param::LoadAverageParam;
use crate::imod::etomo::jdk::ButtonGroup;
use crate::imod::etomo::logic::processor_table_state::ProcessorTableState;
use crate::imod::etomo::logic::processor_type::ProcessorType;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::node::Node;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::expander::Expander;

/// Java package-private `class CpuTable extends ProcessorTable`.
pub struct CpuTable {
    base: ProcessorTable,
    /// Java private final `PREPEND` (an instance field in the Java).
    prepend: &'static str,
    /// Java private final `usersColumn` (assigned, never read, in the Java too).
    #[allow(dead_code)]
    users_column: bool,
}

impl Deref for CpuTable {
    type Target = ProcessorTable;
    fn deref(&self) -> &ProcessorTable {
        &self.base
    }
}

impl CpuTable {
    /// The field part of Java `CpuTable(BaseManager, ParallelPanel, AxisID, boolean,
    /// Expander, InterfaceType)`: `ProcessorTable`'s own fields (see
    /// [`ProcessorTable::new`]) and `usersColumn`.  (Java assigns `usersColumn` after
    /// `super(...)`; the superclass constructor never reads it.)  A subclass
    /// (`GpuTable`) embeds the result and then runs [`ProcessorTable::construct`].
    pub fn new_fields(
        manager: &'static dyn BaseManager,
        parent: Weak<ParallelPanel>,
        axis_id: AxisID,
        runnable: bool,
    ) -> CpuTable {
        CpuTable {
            // super(manager, parent, axisID, false, runnable, moreLess, interfaceType)
            base: ProcessorTable::new(manager, parent, axis_id, runnable),
            prepend: ".Cpu",
            users_column: cpu_adoc::INSTANCE.is_users_column(),
        }
    }

    /// Java `CpuTable(BaseManager, ParallelPanel, AxisID, boolean runnable, Expander
    /// moreLess, InterfaceType)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<ParallelPanel>,
        axis_id: AxisID,
        runnable: bool,
        more_less: Option<Rc<dyn Expander>>,
        interface_type: InterfaceType,
    ) -> Rc<CpuTable> {
        let instance = Rc::new(CpuTable::new_fields(manager, parent, axis_id, runnable));
        let this = Rc::downgrade(&instance) as Weak<dyn ProcessorTableVirtual>;
        instance.base.set_this(this);
        // displayQueues is false.
        instance.base.construct(false, more_less, interface_type);
        instance
    }

    /// Java `@Override getProcessorType()`.
    pub fn get_processor_type(&self) -> ProcessorType {
        ProcessorType::Cpu
    }

    /// Java `@Override isQueueTable()`.
    pub fn is_queue_table(&self) -> bool {
        false
    }

    /// Java `@Override isCpuTable()`.
    pub fn is_cpu_table(&self) -> bool {
        true
    }

    /// Java `@Override isGpuTable()`.
    pub fn is_gpu_table(&self) -> bool {
        false
    }

    /// Java `@Override getStorePrepend()`.
    pub fn get_store_prepend(&self) -> String {
        format!("{}{}", self.base.get_group_key(), self.prepend)
    }

    /// Java `@Override getLoadPrepend(ConstEtomoVersion)`.
    pub fn get_load_prepend(&self, version: &dyn ConstEtomoVersion) -> String {
        if version.ge_string(Some("1.1")) {
            return format!("{}{}", self.base.get_group_key(), self.prepend);
        }
        "ProcessorTable".to_string()
    }

    /// Java final `@Override getSize()`.
    pub fn get_size(&self) -> i32 {
        Network::get_num_computers()
    }

    /// Java `@Override getParameters(ProcessingMethod, BatchruntomoParam, boolean)`.
    pub fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        _do_validation: bool,
    ) -> bool {
        if method == Some(ProcessingMethod::PpCpu) {
            param.reset_cpu_machine_list();
            // getParameters(param): virtual.
            self.base.this().get_parameters_batchruntomo_param(param);
        }
        true
    }

    /// Java final `getButtonGroup()`.
    pub fn get_button_group(&self) -> Option<Rc<ButtonGroup>> {
        None
    }

    /// Java final `@Override getNode(int)`.
    pub fn get_node(&self, index: i32) -> Option<Arc<Node>> {
        Network::get_computer(
            self.base.manager,
            index,
            self.base.axis_id,
            self.base.manager.get_property_user_dir().as_deref(),
        )
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
        // Node.getCpus() never returns null in the translation.
        let cpus = node.get_cpus();
        let mut i_cpus = 0;
        if !cpus.is_null() {
            i_cpus = cpus.get_int();
        }
        ProcessorTableRow::get_computer_instance(
            processor_table,
            node,
            i_cpus,
            num_rows_in_table,
            table_state,
        )
    }

    /// Java `@Override getHeader1ComputerText()`.
    pub fn get_header1_computer_text(&self) -> Option<String> {
        Some("Computer".to_string())
    }

    /// Java `@Override getNoCpusSelectedErrorMessage()`.
    pub fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        Some("At least one computer must be selected.".to_string())
    }

    /// Java final `@Override getIntermittentCommand(String)`.
    pub fn get_intermittent_command_string(
        &self,
        computer: Option<&str>,
    ) -> Arc<dyn IntermittentCommand> {
        // Java uses the (null) computer as a map key; a null row label reads "null".
        LoadAverageParam::get_instance(computer.unwrap_or("null"), self.base.manager)
            as Arc<dyn IntermittentCommand>
    }

    /// Java `@Override isExcludeNode(Node)`.
    pub fn is_exclude_node(&self, _node: &Node) -> bool {
        false
    }

    /// Java final `@Override isNiceable()`.
    pub fn is_niceable(&self) -> bool {
        true
    }

    /// Java `@Override initRow(ProcessorTableRow)` (empty).
    pub fn init_row(&self, _row: &Rc<ProcessorTableRow>) {}
}

impl ProcessorTableVirtual for CpuTable {
    fn processor_table(&self) -> &ProcessorTable {
        &self.base
    }
    fn get_size(&self) -> i32 {
        CpuTable::get_size(self)
    }
    fn get_node(&self, index: i32) -> Option<Arc<Node>> {
        CpuTable::get_node(self, index)
    }
    fn create_processor_table_row(
        &self,
        processor_table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        CpuTable::create_processor_table_row(
            self,
            processor_table,
            node,
            num_rows_in_table,
            table_state,
        )
    }
    fn get_header1_computer_text(&self) -> Option<String> {
        CpuTable::get_header1_computer_text(self)
    }
    fn get_intermittent_command_string(
        &self,
        computer: Option<&str>,
    ) -> Arc<dyn IntermittentCommand> {
        CpuTable::get_intermittent_command_string(self, computer)
    }
    fn is_exclude_node(&self, node: &Node) -> bool {
        CpuTable::is_exclude_node(self, node)
    }
    fn is_niceable(&self) -> bool {
        CpuTable::is_niceable(self)
    }
    fn get_store_prepend(&self) -> String {
        CpuTable::get_store_prepend(self)
    }
    fn get_load_prepend(&self, version: &dyn ConstEtomoVersion) -> String {
        CpuTable::get_load_prepend(self, version)
    }
    fn init_row(&self, row: &Rc<ProcessorTableRow>) {
        CpuTable::init_row(self, row)
    }
    fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        CpuTable::get_no_cpus_selected_error_message(self)
    }
    fn is_queue_table(&self) -> bool {
        CpuTable::is_queue_table(self)
    }
    fn is_cpu_table(&self) -> bool {
        CpuTable::is_cpu_table(self)
    }
    fn is_gpu_table(&self) -> bool {
        CpuTable::is_gpu_table(self)
    }
    fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        do_validation: bool,
    ) -> bool {
        CpuTable::get_parameters_processing_method_batchruntomo_param_boolean(
            self,
            method,
            param,
            do_validation,
        )
    }
    fn get_processor_type(&self) -> ProcessorType {
        CpuTable::get_processor_type(self)
    }
}

crate::processor_table_interfaces!(CpuTable);
