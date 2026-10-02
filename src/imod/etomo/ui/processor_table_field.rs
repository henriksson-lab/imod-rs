//! `IMOD/Etomo/src/etomo/ui/ProcessorTableField.java`.
//!
//! An enum used to identify fields in the Processor Table.  The Java class has a
//! private constructor and one static instance per field, compared by identity; a Rust
//! enum has exactly those semantics.

use super::table_field::TableField;

/// Java `ProcessorTableField implements TableField`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorTableField {
    /// Java `COMPUTER_H1`.
    ComputerH1,
    /// Java `NUM_CPUS_H1`.
    NumCpusH1,
    /// Java `TWO_QUEUES_H2`.
    TwoQueuesH2,
    /// Java `NUM_CPUS_MAX_H2`.
    NumCpusMaxH2,
    /// Java `NUM_GPUS`.
    NumGpus,
    /// Java `LOAD_AVERAGE_H1`.
    LoadAverageH1,
    /// Java `CPU_USAGE_H1`.
    CpuUsageH1,
    /// Java `LOAD_ARRAY_0_H1`.
    LoadArray0H1,
    /// Java `LOAD_ARRAY_X_H1`.
    LoadArrayXH1,
    /// Java `USERS_H1`.
    UsersH1,
    /// Java `TYPE_H1`.
    TypeH1,
    /// Java `SPEED_H1`.
    SpeedH1,
    /// Java `MEMORY_H1`.
    MemoryH1,
    /// Java `OS_H1`.
    OsH1,
    /// Java `RESTARTS_H1`.
    RestartsH1,
    /// Java `GPU_TYPE_H1`.
    GpuTypeH1,
    /// Java `GPU_SPEED_H1`.
    GpuSpeedH1,
    /// Java `GPU_MEMORY_H1`.
    GpuMemoryH1,
    /// Java `GPU_NCORES_H1`.
    GpuNcoresH1,
}

impl TableField for ProcessorTableField {}
