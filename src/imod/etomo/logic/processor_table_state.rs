//! `IMOD/Etomo/src/etomo/logic/ProcessorTableState.java`.
//!
//! Collects and distributes information about the state of a processor table.
//!
//! Permanent state (per table):
//! - windows - bool (from OS)
//! - usersColumn - bool (from cpu.adoc global section)
//! - loadUnits - number of elements - int (from cpu.adoc global section)
//! - cpuType - bool (from cpu.adoc all sections)
//! - memory - bool (from cpu.adoc all sections)
//! - numberGt1 (includes gpuGt1) - bool (from cpu.adoc all sections)
//! - os - bool (from cpu.adoc all sections)
//! - speed - bool (from cpu.adoc all sections)
//! - secondary - bool (from table)
//! - type - cpu, gpu, queue - enum (from table)
//!
//! Changeable state:
//! - less - bool (from panel header)
//! - secondary - bool (from table)
//! - runnable - bool (from table)
//!
//! Information from cpu.adoc sections requires the user name, and the current
//! interface.
//!
//! Fields are displayed from left to right, and then from top row to bottom row.
//! Fields should only depend on previously displayed fields.  Checking whether a field
//! is used in a table should only rely on the permanent state.  H2 and Row fields
//! always have a grid width of 1.  (The source's per-field table of use/display rules
//! is the code below.)
//!
//! An EDT object, built as `Rc<Self>` by `ProcessorTable` and shared with its cells as
//! `Rc<dyn TableState>`.  The table owns this state and is also its `display`, so the
//! display is held as a `Weak` (Java's collector handles the cycle); it is alive
//! whenever this state is asked anything.

use std::any::Any;
use std::rc::{Rc, Weak};
use std::sync::LazyLock;

use super::processor_type::ProcessorType;
use super::table_state::{DEFAULT_GRIDWIDTH, TableState};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::ui::expander::Expander;
use crate::imod::etomo::ui::processor_table_field::ProcessorTableField;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;
use crate::imod::etomo::ui::table_field::TableField;
use crate::imod::etomo::util::utilities;

/// Java private static final `WINDOWS`.
static WINDOWS: LazyLock<bool> = LazyLock::new(utilities::is_windows_os);
/// Java private static final `USERS_COLUMN`.
static USERS_COLUMN: LazyLock<bool> = LazyLock::new(|| cpu_adoc::INSTANCE.is_users_column());
/// Java private static final `LOAD_UNITS`.
static LOAD_UNITS: LazyLock<i32> = LazyLock::new(|| cpu_adoc::INSTANCE.get_load_units());
/// Java private static final `GRIDWIDTH_2`.
const GRIDWIDTH_2: i32 = 2;
/// Java private static final `GRIDWIDTH_3`.
const GRIDWIDTH_3: i32 = 3;

/// Java `ProcessorTableState implements TableState`.
pub struct ProcessorTableState {
    /// Java private final `moreLess`.
    more_less: Option<Rc<dyn Expander>>,
    /// Java private final `display`.
    display: Weak<dyn ParallelProgressDisplay>,
    /// Java private final `processorType`.
    processor_type: ProcessorType,
    /// Java private final `gpusPerClusterJob`.
    gpus_per_cluster_job: bool,
    /// Java private final `cpuType`.
    cpu_type: bool,
    /// Java private final `numberGt1`.
    number_gt1: bool,
    /// Java private final `gpuGt1`.
    gpu_gt1: bool,
    /// Java private final `memory`.
    memory: bool,
    /// Java private final `os`.
    os: bool,
    /// Java private final `speed`.
    speed: bool,
    /// Java private final `gpuType`.
    gpu_type: bool,
    /// Java private final `gpuMemory`.
    gpu_memory: bool,
    /// Java private final `gpuNcores`.
    gpu_ncores: bool,
    /// Java private final `gpuSpeed`.
    gpu_speed: bool,
    /// Java private final `isQueueTable`.
    is_queue_table: bool,
    /// Java private final `dualSelectionQueueTable`.
    dual_selection_queue_table: bool,
}

impl ProcessorTableState {
    /// Java `ProcessorTableState(BaseManager, AxisID, String, InterfaceType, Expander,
    /// ParallelProgressDisplay, ProcessorType, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
        interface_type: Option<InterfaceType>,
        more_less: Option<Rc<dyn Expander>>,
        display: Weak<dyn ParallelProgressDisplay>,
        processor_type: ProcessorType,
        is_queue_table: bool,
        dual_selection_queue_table: bool,
    ) -> Rc<ProcessorTableState> {
        let gpus_per_cluster_job =
            cpu_adoc::INSTANCE.is_gpus_per_cluster_job(interface_type, processor_type);
        let cpu_type = cpu_adoc::INSTANCE.is_type(interface_type, processor_type);
        let number_gt1 = Network::is_number_gt1(
            interface_type,
            processor_type,
            manager,
            axis_id,
            property_user_dir,
        );
        let gpu_gt1 = Network::is_gpu_gt1(
            interface_type,
            processor_type,
            manager,
            axis_id,
            property_user_dir,
        );
        let memory = cpu_adoc::INSTANCE.is_memory(interface_type, processor_type);
        let os = cpu_adoc::INSTANCE.is_os(interface_type, processor_type);
        let speed = cpu_adoc::INSTANCE.is_speed(interface_type, processor_type);
        let gpu_type = cpu_adoc::INSTANCE.is_gpu_type(interface_type, processor_type);
        let gpu_memory = cpu_adoc::INSTANCE.is_gpu_memory(interface_type, processor_type);
        let gpu_ncores = cpu_adoc::INSTANCE.is_gpu_ncores(interface_type, processor_type);
        let gpu_speed = cpu_adoc::INSTANCE.is_gpu_speed(interface_type, processor_type);
        Rc::new(ProcessorTableState {
            more_less,
            display,
            processor_type,
            is_queue_table,
            dual_selection_queue_table,
            gpus_per_cluster_job,
            cpu_type,
            number_gt1,
            gpu_gt1,
            memory,
            os,
            speed,
            gpu_type,
            gpu_memory,
            gpu_ncores,
            gpu_speed,
        })
    }

    /// Java `isUse(ProcessorTableField)`.
    pub fn is_use(&self, table_field: ProcessorTableField) -> bool {
        if table_field == ProcessorTableField::TwoQueuesH2 {
            return self.is_queue_table && self.dual_selection_queue_table;
        }
        if table_field == ProcessorTableField::NumCpusMaxH2 {
            if self.processor_type == ProcessorType::Gpu {
                return self.gpu_gt1;
            }
            return self.number_gt1;
        }
        if table_field == ProcessorTableField::NumGpus {
            return self.dual_selection_queue_table
                && self.processor_type == ProcessorType::Queue
                && self.gpus_per_cluster_job;
        }
        if table_field == ProcessorTableField::LoadAverageH1 {
            return !*WINDOWS
                && (self.processor_type == ProcessorType::Cpu
                    || self.processor_type == ProcessorType::Gpu);
        }
        if table_field == ProcessorTableField::CpuUsageH1 {
            return *WINDOWS
                && (self.processor_type == ProcessorType::Cpu
                    || self.processor_type == ProcessorType::Gpu);
        }
        if table_field == ProcessorTableField::LoadArray0H1 {
            return self.processor_type == ProcessorType::Queue;
        }
        if table_field == ProcessorTableField::LoadArrayXH1 {
            return self.processor_type == ProcessorType::Queue && *LOAD_UNITS > 1;
        }
        if table_field == ProcessorTableField::UsersH1 {
            return *USERS_COLUMN
                && !*WINDOWS
                && (self.processor_type == ProcessorType::Cpu
                    || self.processor_type == ProcessorType::Gpu);
        }
        if table_field == ProcessorTableField::TypeH1 {
            return self.cpu_type;
        }
        if table_field == ProcessorTableField::SpeedH1 {
            return self.speed;
        }
        if table_field == ProcessorTableField::MemoryH1 {
            return self.memory;
        }
        if table_field == ProcessorTableField::OsH1 {
            return self.os;
        }
        if table_field == ProcessorTableField::RestartsH1 {
            // The secondary table never displays the restart fields.
            return !self
                .display
                .upgrade()
                .expect("ProcessorTableState: the table (display) owns its state")
                .is_secondary();
        }
        if table_field == ProcessorTableField::GpuTypeH1 {
            return self.gpu_type;
        }
        if table_field == ProcessorTableField::GpuSpeedH1 {
            return self.gpu_speed;
        }
        if table_field == ProcessorTableField::GpuMemoryH1 {
            return self.gpu_memory;
        }
        if table_field == ProcessorTableField::GpuNcoresH1 {
            return self.gpu_ncores;
        }
        true
    }
}

impl TableState for ProcessorTableState {
    /// Java `isDisplay(TableField)`.  Java compares the `TableField` against the
    /// `ProcessorTableField` constants by identity; a field of any other class matches
    /// none of them.
    fn is_display(&self, table_field: Option<&dyn TableField>) -> bool {
        let table_field: Option<ProcessorTableField> = table_field.and_then(|table_field| {
            (table_field as &dyn Any)
                .downcast_ref::<ProcessorTableField>()
                .copied()
        });
        if table_field == Some(ProcessorTableField::NumCpusMaxH2)
            || table_field == Some(ProcessorTableField::NumGpus)
        {
            return self.more_less.is_none() || self.more_less.as_ref().unwrap().is_expanded();
        }
        if table_field == Some(ProcessorTableField::LoadAverageH1)
            || table_field == Some(ProcessorTableField::CpuUsageH1)
            || table_field == Some(ProcessorTableField::LoadArray0H1)
        {
            return !self
                .display
                .upgrade()
                .expect("ProcessorTableState: the table (display) owns its state")
                .is_secondary();
        }
        if table_field == Some(ProcessorTableField::UsersH1)
            || table_field == Some(ProcessorTableField::TypeH1)
            || table_field == Some(ProcessorTableField::SpeedH1)
            || table_field == Some(ProcessorTableField::MemoryH1)
            || table_field == Some(ProcessorTableField::OsH1)
            || table_field == Some(ProcessorTableField::GpuTypeH1)
            || table_field == Some(ProcessorTableField::GpuSpeedH1)
            || table_field == Some(ProcessorTableField::GpuMemoryH1)
            || table_field == Some(ProcessorTableField::GpuNcoresH1)
        {
            return (self.more_less.is_none() || self.more_less.as_ref().unwrap().is_expanded())
                && !self
                    .display
                    .upgrade()
                    .expect("ProcessorTableState: the table (display) owns its state")
                    .is_secondary();
        }
        if table_field == Some(ProcessorTableField::RestartsH1) {
            let display = self
                .display
                .upgrade()
                .expect("ProcessorTableState: the table (display) owns its state");
            return !display.is_secondary() && display.is_runnable() && !display.is_limited();
        }
        true
    }

    /// Java `getGridwidth(TableField)`.
    ///
    /// Upstream bug fixed (ProcessorTableState.java:306): the `NUM_CPUS_H1` arm calls
    /// `moreLess.isExpanded()` without the null check every `isDisplay` arm makes, so a
    /// table built without a more/less expander throws a NullPointerException there.
    /// A null `moreLess` is treated as expanded, as `isDisplay` treats it.
    fn get_gridwidth(&self, table_field: Option<&dyn TableField>) -> i32 {
        let table_field: Option<ProcessorTableField> = table_field.and_then(|table_field| {
            (table_field as &dyn Any)
                .downcast_ref::<ProcessorTableField>()
                .copied()
        });
        if table_field == Some(ProcessorTableField::ComputerH1) {
            if self.is_queue_table && self.dual_selection_queue_table {
                return GRIDWIDTH_3;
            }
        } else if table_field == Some(ProcessorTableField::NumCpusH1) {
            if (self.more_less.is_none() || self.more_less.as_ref().unwrap().is_expanded())
                && ((self.processor_type == ProcessorType::Gpu && self.gpu_gt1)
                    || (self.processor_type != ProcessorType::Gpu && self.number_gt1))
            {
                return GRIDWIDTH_2;
            }
        } else if table_field == Some(ProcessorTableField::LoadAverageH1) {
            return GRIDWIDTH_2;
        }
        DEFAULT_GRIDWIDTH
    }
}
