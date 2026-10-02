//! `IMOD/Etomo/src/etomo/ui/swing/GpuTable.java`.
//!
//! Child of CpuTable that makes a ProcessorTable display GPUs (bug# 1422).
//!
//! Java `final class GpuTable extends CpuTable`: the `CpuTable` is embedded as
//! `base` (with `Deref`); the members GpuTable overrides are in its
//! [`ProcessorTableVirtual`] implementation, and the ones it inherits forward to the
//! `CpuTable` bodies.

use std::collections::HashMap;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::cpu_table::CpuTable;
use super::parallel_panel::ParallelPanel;
use super::processor_table::{ProcessorTable, ProcessorTableVirtual};
use super::processor_table_row::ProcessorTableRow;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::intermittent_command::IntermittentCommand;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::logic::processor_table_state::ProcessorTableState;
use crate::imod::etomo::logic::processor_type::ProcessorType;
use crate::imod::etomo::storage::node::Node;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::expander::Expander;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;

/// Java private static final `PREPEND`.
const PREPEND: &str = ".Gpu";
/// Java static final `NUMBER_CPUS_LABEL`.
pub const NUMBER_CPUS_LABEL: &str = "# GPUs";

/// Java package-private `final class GpuTable extends CpuTable`.
pub struct GpuTable {
    base: CpuTable,
}

impl Deref for GpuTable {
    type Target = CpuTable;
    fn deref(&self) -> &CpuTable {
        &self.base
    }
}

impl GpuTable {
    /// Java `GpuTable(BaseManager, ParallelPanel, AxisID, boolean runnable, Expander
    /// moreLess, InterfaceType)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<ParallelPanel>,
        axis_id: AxisID,
        runnable: bool,
        more_less: Option<Rc<dyn Expander>>,
        interface_type: InterfaceType,
    ) -> Rc<GpuTable> {
        // super(manager, parent, axisID, runnable, moreLess, interfaceType)
        let instance = Rc::new(GpuTable {
            base: CpuTable::new_fields(manager, parent, axis_id, runnable),
        });
        let this = Rc::downgrade(&instance) as Weak<dyn ProcessorTableVirtual>;
        instance.processor_table().set_this(this);
        // CpuTable passes displayQueues false.
        instance
            .processor_table()
            .construct(false, more_less, interface_type);
        instance
    }

    /// Java `@Override getProcessorType()`.
    pub fn get_processor_type(&self) -> ProcessorType {
        ProcessorType::Gpu
    }

    /// Java `@Override isCpuTable()`.
    pub fn is_cpu_table(&self) -> bool {
        false
    }

    /// Java `@Override isGpuTable()`.
    pub fn is_gpu_table(&self) -> bool {
        true
    }

    /// Java `@Override getheader1NumberCPUsTitle()`.
    pub fn getheader1_number_cpus_title(&self) -> Option<String> {
        Some(NUMBER_CPUS_LABEL.to_string())
    }

    /// Java `@Override getStorePrepend()`.
    pub fn get_store_prepend(&self) -> String {
        format!("{}{}", self.processor_table().get_group_key(), PREPEND)
    }

    /// Java `@Override getLoadPrepend(ConstEtomoVersion)`.
    pub fn get_load_prepend(&self, _version: &dyn ConstEtomoVersion) -> String {
        format!("{}{}", self.processor_table().get_group_key(), PREPEND)
    }

    /// Java `@Override getHeader1ComputerText()`.
    pub fn get_header1_computer_text(&self) -> Option<String> {
        Some("GPU".to_string())
    }

    /// Java `@Override getNoCpusSelectedErrorMessage()`.
    pub fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        Some("At least one GPU must be selected.".to_string())
    }

    /// Java `@Override isExcludeNode(Node)`.
    pub fn is_exclude_node(&self, node: &Node) -> bool {
        if !node.is_gpu() {
            return true;
        }
        let table = self.processor_table();
        if node.is_gpu_local()
            && !node.is_local_host(
                table.manager,
                table.axis_id,
                table.manager.get_property_user_dir().as_deref(),
            )
        {
            return true;
        }
        false
    }

    /// Java `@Override getMachineMap(BatchruntomoParam)`.
    pub fn get_machine_map(&self, param: &BatchruntomoParam) -> Option<HashMap<String, String>> {
        param.get_gpu_machine_map()
    }

    /// Java `@Override getParameters(ProcesschunksParam)`.
    pub fn get_parameters_processchunks_param(&self, param: &ProcesschunksParam) {
        // Avoid loading parameters if this is a secondary table. Only some GPU
        // parameters can be loaded from the secondary table.
        if !ParallelProgressDisplay::is_secondary(self) {
            param.set_gpu_processing(true);
        }
        // super.getParameters(param): CpuTable does not override it.
        self.processor_table()
            .get_parameters_processchunks_param_super(param);
    }

    /// Java `@Override getParameters(ProcessingMethod, BatchruntomoParam, boolean)`.
    pub fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        _do_validation: bool,
    ) -> bool {
        if method == Some(ProcessingMethod::PpGpu) {
            param.reset_gpu_machine_list();
            // getParameters(param): virtual.
            self.processor_table()
                .this()
                .get_parameters_batchruntomo_param(param);
        }
        true
    }

    /// Java package-private `enableNumberColumn(Node)`.
    pub fn enable_number_column(&self, node: &Node) -> bool {
        // numberColumn is true if an number attribute is not defaulted to 1
        // 1436 unnecessary column (was !isDefault and was always true)
        node.get_gpu_number() > 1
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
        ProcessorTableRow::get_computer_instance(
            processor_table,
            node,
            node.get_gpu_number(),
            num_rows_in_table,
            table_state,
        )
    }

    /// Java `@Override initRow(ProcessorTableRow)`.
    pub fn init_row(&self, row: &Rc<ProcessorTableRow>) {
        row.turn_off_load_warning();
    }
}

impl ProcessorTableVirtual for GpuTable {
    fn processor_table(&self) -> &ProcessorTable {
        self.base.processor_table()
    }
    // Inherited from CpuTable.
    fn get_size(&self) -> i32 {
        self.base.get_size()
    }
    fn get_node(&self, index: i32) -> Option<Arc<Node>> {
        self.base.get_node(index)
    }
    fn get_intermittent_command_string(
        &self,
        computer: Option<&str>,
    ) -> Arc<dyn IntermittentCommand> {
        self.base.get_intermittent_command_string(computer)
    }
    fn is_niceable(&self) -> bool {
        self.base.is_niceable()
    }
    fn is_queue_table(&self) -> bool {
        self.base.is_queue_table()
    }
    // Overridden by GpuTable.
    fn create_processor_table_row(
        &self,
        processor_table: &Rc<dyn ProcessorTableVirtual>,
        node: &Node,
        num_rows_in_table: i32,
        table_state: &Rc<ProcessorTableState>,
    ) -> Rc<ProcessorTableRow> {
        GpuTable::create_processor_table_row(
            self,
            processor_table,
            node,
            num_rows_in_table,
            table_state,
        )
    }
    fn get_header1_computer_text(&self) -> Option<String> {
        GpuTable::get_header1_computer_text(self)
    }
    fn is_exclude_node(&self, node: &Node) -> bool {
        GpuTable::is_exclude_node(self, node)
    }
    fn get_store_prepend(&self) -> String {
        GpuTable::get_store_prepend(self)
    }
    fn get_load_prepend(&self, version: &dyn ConstEtomoVersion) -> String {
        GpuTable::get_load_prepend(self, version)
    }
    fn init_row(&self, row: &Rc<ProcessorTableRow>) {
        GpuTable::init_row(self, row)
    }
    fn get_no_cpus_selected_error_message(&self) -> Option<String> {
        GpuTable::get_no_cpus_selected_error_message(self)
    }
    fn is_cpu_table(&self) -> bool {
        GpuTable::is_cpu_table(self)
    }
    fn is_gpu_table(&self) -> bool {
        GpuTable::is_gpu_table(self)
    }
    fn get_parameters_processing_method_batchruntomo_param_boolean(
        &self,
        method: Option<ProcessingMethod>,
        param: &mut BatchruntomoParam,
        do_validation: bool,
    ) -> bool {
        GpuTable::get_parameters_processing_method_batchruntomo_param_boolean(
            self,
            method,
            param,
            do_validation,
        )
    }
    fn get_processor_type(&self) -> ProcessorType {
        GpuTable::get_processor_type(self)
    }
    fn getheader1_number_cpus_title(&self) -> Option<String> {
        GpuTable::getheader1_number_cpus_title(self)
    }
    fn get_parameters_processchunks_param(&self, param: &ProcesschunksParam) {
        GpuTable::get_parameters_processchunks_param(self, param)
    }
    fn get_machine_map(&self, param: &BatchruntomoParam) -> Option<HashMap<String, String>> {
        GpuTable::get_machine_map(self, param)
    }
}

crate::processor_table_interfaces!(GpuTable);
